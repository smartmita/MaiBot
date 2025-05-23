import asyncio
import threading
import random
import time
import json
import re
from typing import Dict, Optional, List, Any, Tuple
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from .sobriquet_db import SobriquetDB
from .sobriquet_mapper import build_sobriquet_mapping_prompt, format_existing_sobriquets_for_prompt
from .sobriquet_utils import select_sobriquets_for_prompt
from src.chat.person_info.person_info import person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("SobriquetManager") # 获取日志记录器实例
logger_helper = get_logger("AsyncLoopHelperSobriquet") # 异步循环辅助日志

def run_async_loop(loop: asyncio.AbstractEventLoop, coro):
    """
    在指定的事件循环中运行给定的协程直到完成，并确保循环最终安全关闭。
    """
    try:
        logger_helper.debug(f"在事件循环 {id(loop)} 中运行协程 (用于绰号处理)...")
        result = loop.run_until_complete(coro)
        logger_helper.debug(f"事件循环 {id(loop)} 中的协程已完成 (用于绰号处理)。")
        return result
    except asyncio.CancelledError: # pragma: no cover
        logger_helper.info(f"事件循环 {id(loop)} 中的协程被取消 (用于绰号处理)。")
    except Exception as e: # pragma: no cover
        logger_helper.error(f"事件循环 {id(loop)} 中发生错误 (用于绰号处理): {e}", exc_info=True)
    finally: # 确保循环被正确清理
        try:
            all_tasks = asyncio.all_tasks(loop)
            current_task = asyncio.current_task(loop) # 获取当前任务以避免取消自身
            tasks_to_cancel = [
                task for task in all_tasks if task is not current_task
            ]
            if tasks_to_cancel:
                logger_helper.info(f"正在取消事件循环 {id(loop)} 中的 {len(tasks_to_cancel)} 个剩余任务...")
                for task in tasks_to_cancel:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logger_helper.info(f"事件循环 {id(loop)} 中的剩余任务已取消。")
            if loop.is_running(): # 如果循环仍在运行，则停止它
                loop.stop()
                logger_helper.info(f"Asyncio 事件循环 {id(loop)} 已停止。")
            if not loop.is_closed(): # 如果循环未关闭，则关闭它
                loop.run_until_complete(loop.shutdown_asyncgens()) # 先关闭异步生成器
                loop.close()
                logger_helper.info(f"Asyncio 事件循环 {id(loop)} 已关闭。")
        except Exception as close_err: # pragma: no cover
            logger_helper.error(f"清理 Asyncio 事件循环 {id(loop)} 时发生错误: {close_err}", exc_info=True)

class SobriquetManager:
    """
    管理群组绰号的分析、存储和使用。
    通过LLM分析聊天内容，提取可靠和不可靠的绰号映射，并更新数据库。
    提供选择后的绰号用于Prompt注入。
    """
    _instance = None # 单例模式的实例存储
    _lock = threading.Lock() # 用于确保线程安全的锁

    def __new__(cls, *args, **kwargs): # 实现单例模式
        if not cls._instance:
            with cls._lock: # 加锁以防止并发创建多个实例
                if not cls._instance:
                    logger.info("正在创建 SobriquetManager 单例实例...")
                    cls._instance = super(SobriquetManager, cls).__new__(cls)
                    cls._instance._initialized = False # 标记为未初始化
        return cls._instance

    def __init__(self):
        """
        初始化 SobriquetManager。
        使用锁和 _initialized 标志确保实际初始化逻辑只执行一次。
        """
        if hasattr(self, "_initialized") and self._initialized: # 如果已初始化，则直接返回
            return

        with self._lock: # 加锁确保初始化过程的线程安全
            if hasattr(self, "_initialized") and self._initialized: # 再次检查，防止并发进入
                return

            logger.info("正在初始化 SobriquetManager 组件...")
            # 从全局配置中读取绰号分析是否启用的标志
            self.is_analysis_enabled = global_config.profile.enable_sobriquet_mapping

            # 初始化数据库处理器
            person_info_collection = getattr(db, "person_info", None) # 安全获取集合对象
            self.db_handler = SobriquetDB(person_info_collection)
            if not self.db_handler.is_available():
                logger.error("数据库处理器初始化失败，SobriquetManager 功能受限。")

            # 初始化LLM映射器 (如果分析功能启用)
            self.llm_mapper: Optional[LLMRequest] = None
            if self.is_analysis_enabled:
                try:
                    model_config_data = global_config.model.sobriquet_mapping # 直接访问模型配置
                    self.llm_mapper = LLMRequest(
                        model=model_config_data, # 传递模型配置对象
                        temperature=model_config_data.get('temp', 0.5), # 使用get获取可选参数，提供默认值
                        max_tokens=model_config_data.get('max_tokens', 256),
                        request_type="sobriquet_mapping", # 请求类型，可能用于区分不同LLM调用
                    )
                    logger.info("绰号映射 LLM 映射器初始化成功。")
                except AttributeError as ae: # 捕获配置路径错误
                     logger.error(f"初始化绰号映射 LLM 时配置路径错误: {ae}。请检查 global_config.model.sobriquet_mapping 是否正确。绰号分析功能禁用。", exc_info=True)
                     self.llm_mapper = None
                except KeyError as ke: # 捕获缺少关键配置项的错误
                    logger.error(f"初始化绰号映射 LLM 时缺少配置项: {ke}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
                except Exception as e: # 捕获其他初始化错误
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
            
            # 队列和线程相关配置
            self.queue_max_size = global_config.profile.sobriquet_queue_max_size # 分析任务队列的最大长度
            self.sobriquet_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size) # 初始化异步队列
            self._stop_event = threading.Event() # 用于通知后台线程停止的事件
            self._sobriquet_thread: Optional[threading.Thread] = None # 后台处理线程的引用
            self.sleep_interval = global_config.profile.sobriquet_process_sleep_interval # 后台线程轮询队列的间隔
            
            # Prompt注入相关的最低计数阈值
            self.min_count_for_prompt = global_config.profile.min_sobriquet_count_for_prompt_injection
            if self.min_count_for_prompt > 0:
                logger.info(f"将对注入Prompt的绰号应用最低映射强度阈值: {self.min_count_for_prompt}")
            
            # 事件触发式衰减因子 (用于衰减近期未被LLM确认的绰号)
            self.event_decay_factor = float(global_config.profile.sobriquet_event_decay_factor)
            if not (0.0 <= self.event_decay_factor <= 1.0):
                logger.warning(f"事件衰减因子 sobriquet_event_decay_factor ({self.event_decay_factor}) 配置无效，应在[0.0, 1.0]之间。将设置为1.0 (不衰减)。")
                self.event_decay_factor = 1.0 
            if self.event_decay_factor < 1.0:
                 logger.info(f"已配置事件触发式绰号衰减，衰减因子: {self.event_decay_factor}")
            else:
                 logger.info("事件触发式绰号衰减未启用 (衰减因子为1.0)。")

            # LLM判定的不可靠绰号的衰减因子
            self.unreliable_sobriquet_decay_factor = float(global_config.profile.unreliable_sobriquet_decay_factor)
            if not (0.0 <= self.unreliable_sobriquet_decay_factor <= 1.0): # 此因子应小于1才有衰减效果
                logger.warning(f"不可靠绰号衰减因子 unreliable_sobriquet_decay_factor ({self.unreliable_sobriquet_decay_factor}) 配置无效，应在[0.0, 1.0]之间。将设置为0.1。")
                self.unreliable_sobriquet_decay_factor = 0.1 # 提供一个合理的默认值
            logger.info(f"LLM判定的不可靠绰号将按衰减因子 {self.unreliable_sobriquet_decay_factor} 进行处理。")

            self._initialized = True # 标记为已初始化
            logger.info("SobriquetManager 初始化完成。")

    def start_processor(self):
        """启动后台绰号处理线程（如果绰号分析功能已启用且线程未运行）。"""
        if not self.is_analysis_enabled:
            logger.info("绰号分析功能已禁用，处理器未启动。")
            return
        # 如果配置为不向Prompt注入任何绰号，分析的必要性降低
        if global_config.profile.max_sobriquets_in_prompt == 0:
            logger.warning("配置中 max_sobriquets_in_prompt 为0，可能不需要进行绰号分析或注入。")

        # 检查线程是否已存在或是否仍在运行
        if self._sobriquet_thread is None or not self._sobriquet_thread.is_alive():
            logger.info("正在启动绰号处理器线程...")
            self._stop_event.clear() # 清除停止事件，允许线程运行
            self._sobriquet_thread = threading.Thread(
                target=self._run_processor_in_thread, # 线程执行的目标函数
                daemon=True, # 设置为守护线程，主程序退出时线程也会退出
            )
            self._sobriquet_thread.start() # 启动线程
            logger.info(f"绰号处理器线程已启动 (ID: {self._sobriquet_thread.ident})")
        else:
            logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        """停止后台绰号处理线程。"""
        if self._sobriquet_thread and self._sobriquet_thread.is_alive():
            logger.info("正在停止绰号处理器线程...")
            self._stop_event.set() # 设置停止事件，通知线程退出循环
            try:
                self._sobriquet_thread.join(timeout=10) # 等待线程结束，设置超时
                if self._sobriquet_thread.is_alive(): 
                    logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e: 
                logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._sobriquet_thread and not self._sobriquet_thread.is_alive():
                    logger.info("绰号处理器线程已成功停止。")
                self._sobriquet_thread = None # 清理线程引用
        else:
            logger.info("绰号处理器线程未在运行或已被清理。")

    async def trigger_sobriquet_analysis(
        self,
        anchor_message: MessageRecv, # 触发分析的锚点消息
        bot_reply: List[str],       # 机器人对此消息的回复内容
        chat_stream: Optional[ChatStream] = None, # 当前的聊天流对象
    ):
        """
        准备用于绰号分析的数据，并将其放入处理队列。
        此方法会收集当前聊天上下文中的用户ID，用于后续的事件触发式衰减和LLM参考。
        """
        if not self.is_analysis_enabled: # 如果分析功能未启用，则直接返回
            return

        # 根据配置的概率随机决定是否执行此次分析
        if random.random() > global_config.profile.sobriquet_analysis_probability:
            logger.debug("跳过绰号分析：随机概率未命中。")
            return

        current_chat_stream = chat_stream or anchor_message.chat_stream # 获取当前聊天流
        if not current_chat_stream or not current_chat_stream.group_info: # 必须是群聊才有意义
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]" # 日志前缀，方便追踪
        try:
            history_limit = global_config.profile.sobriquet_analysis_history_limit # 获取分析所需的历史消息数量
            # 获取锚点消息之前的聊天记录
            history_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=current_chat_stream.stream_id,
                timestamp=time.time(), # 以当前时间为基准
                limit=history_limit,
            )
            # 将历史消息格式化为LLM可读的字符串
            chat_history_str = await build_readable_messages(
                messages=history_messages,
                replace_bot_name=True, 
                merge_messages=False, 
                timestamp_mode="relative", 
                read_mark=0.0, 
                truncate=False, 
            )
            bot_reply_str = " ".join(bot_reply) if bot_reply else "" # 将机器人回复列表合并为字符串
            group_id = str(current_chat_stream.group_info.group_id) # 获取群组ID
            platform = current_chat_stream.platform # 获取平台信息

            # 提取当前聊天上下文中出现的所有用户ID，用于后续处理
            user_ids_in_current_context = list(set( # 使用set去重
                str(msg["user_info"]["user_id"])
                for msg in history_messages # 从历史消息中提取
                if msg.get("user_info", {}).get("user_id") # 安全获取用户ID
            ))
            # 将当前锚点消息的发送者ID也加入列表
            if anchor_message.user_info and anchor_message.user_info.user_id:
                 current_sender_id = str(anchor_message.user_info.user_id)
                 if current_sender_id not in user_ids_in_current_context: # 避免重复添加
                    user_ids_in_current_context.append(current_sender_id)
            
            logger.debug(f"{log_prefix} 为事件衰减和LLM参考收集到的上下文用户ID: {user_ids_in_current_context}")

            # 构建任务项元组，包含所有分析所需信息
            item = (chat_history_str, bot_reply_str, platform, group_id, user_ids_in_current_context)
            await self._add_to_queue(item, platform, group_id) # 将任务项加入队列

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_selected_sobriquets_for_group(
        self, platform: str, user_ids_in_context: List[str], group_id: str
    ) -> Optional[List[Tuple[str, str, str, int]]]:
        """
        获取指定用户在特定群组中，经过筛选和选择（满足最低计数阈值和数量上限）的常用绰号。
        此方法供 ProfileManager 调用，用于构建注入到主LLM的Prompt。
        """
        max_sobriquets_to_inject = global_config.profile.max_sobriquets_in_prompt # 获取最大注入数量配置
        # 如果分析未启用或最大注入数为0，则不提供绰号
        if not self.is_analysis_enabled or max_sobriquets_to_inject == 0:
            logger.debug(
                f"群组 {group_id} 的绰号注入已跳过: "
                f"分析启用状态={self.is_analysis_enabled}, "
                f"最大注入数量={max_sobriquets_to_inject}"
            )
            return None

        log_prefix = f"[{platform}:{group_id}]"
        
        try:
            all_sobriquets_data_by_actual_name: Dict[str, Dict[str, Any]] = {}
            if user_ids_in_context: # 仅当提供了上下文用户ID时才查询数据库
                # 从关系管理器获取用户在指定群组的原始绰号数据
                all_sobriquets_data_by_actual_name = await relationship_manager.get_users_group_nicknames(
                    platform, user_ids_in_context, group_id
                )

            if not all_sobriquets_data_by_actual_name: # 如果没有获取到任何数据
                logger.debug(f"{log_prefix} 未找到群组 '{group_id}' 的原始绰号数据。")
                return None

            # 使用工具函数根据权重和配置选择候选绰号
            selected_sobriquets_candidates = select_sobriquets_for_prompt(all_sobriquets_data_by_actual_name)

            if not selected_sobriquets_candidates: # 如果选择后为空
                logger.debug(f"{log_prefix} `select_sobriquets_for_prompt` 未选出任何候选绰号。")
                return None
            
            logger.debug(f"{log_prefix} 初始选出的候选绰号 (共{len(selected_sobriquets_candidates)}个): {selected_sobriquets_candidates}")

            # 应用最低映射强度阈值进行过滤
            final_selected_sobriquets: List[Tuple[str, str, str, int]] = []
            if self.min_count_for_prompt > 0: # 如果设置了最低阈值
                for sobriquet_tuple in selected_sobriquets_candidates:
                    # sobriquet_tuple 结构: (用户实际昵称, user_id, 群内常用绰号, 次数)
                    # 次数是第4个元素，索引为3
                    if sobriquet_tuple[3] >= self.min_count_for_prompt: # 只有计数大于等于阈值的才保留
                        final_selected_sobriquets.append(sobriquet_tuple)
                    else:
                        logger.debug(f"{log_prefix} 绰号 '{sobriquet_tuple[2]}' (用户ID: {sobriquet_tuple[1]}, 次数: {sobriquet_tuple[3]}) "
                                     f"因低于阈值 {self.min_count_for_prompt} 而被排除出Prompt。")
                
                if not final_selected_sobriquets and selected_sobriquets_candidates: # 如果所有都被过滤掉了
                     logger.info(f"{log_prefix} 所有初始选择的绰号都低于最低映射强度阈值 {self.min_count_for_prompt}。")
            else: # 如果阈值为0或未设置，则使用所有选出的候选者
                final_selected_sobriquets = selected_sobriquets_candidates
            
            if final_selected_sobriquets:
                logger.info(f"{log_prefix} 为群组 '{group_id}' 最终选择了以下用户常用绰号进行注入 (共{len(final_selected_sobriquets)}个): {final_selected_sobriquets}")
            else:
                logger.info(f"{log_prefix} 没有绰号满足注入条件 (阈值: {self.min_count_for_prompt})。")
                
            return final_selected_sobriquets # 返回最终筛选结果
            
        except Exception as e:
            logger.error(f"{log_prefix} 获取群组 '{group_id}' 的绰号以供注入时出错: {e}", exc_info=True)
            return None # 出错时返回None

    async def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        """将分析任务异步添加到内部处理队列。"""
        try:
            await self.sobriquet_queue.put(item) # 异步放入队列
            logger.debug(
                f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号分析队列。当前大小: {self.sobriquet_queue.qsize()}"
            )
        except asyncio.QueueFull: # pragma: no cover
            logger.warning(
                f"绰号分析队列已满 (最大={self.queue_max_size})。平台 '{platform}' 群组 '{group_id}' 的项目被丢弃。"
            )
        except Exception as e: # pragma: no cover
            logger.error(f"将项目添加到绰号分析队列时出错: {e}", exc_info=True)

    async def _analyze_and_update_sobriquets(self, item: tuple):
        """
        处理单个队列项目：执行事件衰减，获取LLM参考信息，调用 LLM 分析，并根据结果更新数据库。
        item 结构: (chat_history_str, bot_reply, platform, group_id, user_ids_for_llm_ref_and_decay)
        """
        if not isinstance(item, tuple) or len(item) != 5: # 校验任务项格式
            logger.warning(f"从队列接收到无效项目: 类型 {type(item)}, 期望长度 5 但得到 {len(item) if isinstance(item, tuple) else '非元组'}")
            return

        chat_history_str, bot_reply, platform, group_id, user_ids_for_llm_ref_and_decay = item
        log_prefix = f"[{platform}:{group_id}]" # 日志前缀
        logger.debug(f"{log_prefix} 开始处理绰号分析任务。上下文用户ID (用于衰减和LLM参考): {user_ids_for_llm_ref_and_decay}")

        # 步骤1: 事件触发式衰减 (如果衰减因子小于1.0且数据库可用)
        if self.event_decay_factor < 1.0 and self.db_handler.is_available():
            if user_ids_for_llm_ref_and_decay: # 仅当有明确的用户ID列表时执行
                try:
                    decayed_count = await self.db_handler.decay_sobriquets_in_group(
                        platform=platform,
                        group_id=group_id,
                        user_ids=user_ids_for_llm_ref_and_decay, 
                        decay_factor=self.event_decay_factor # 使用配置的衰减因子
                    )
                    if decayed_count > 0 :
                        logger.info(f"{log_prefix} 对 {len(user_ids_for_llm_ref_and_decay)} 个目标用户的共 {decayed_count} 个绰号条目应用了事件衰减。")
                    else:
                        logger.debug(f"{log_prefix} 事件衰减未影响任何绰号条目（可能目标用户无绰号或计数已为0）。")
                except Exception as e_decay:
                    logger.error(f"{log_prefix} 执行绰号事件衰减时出错: {e_decay}", exc_info=True)
            else:
                logger.debug(f"{log_prefix} 跳过事件触发式衰减，因未提供上下文用户ID列表。")

        # 步骤2: LLM 分析前的准备和调用
        if not self.llm_mapper: # 检查LLM映射器是否可用
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行绰号分析。")
            return
        if not self.db_handler.is_available(): # 再次检查数据库是否可用
            logger.error(f"{log_prefix} 数据库处理器不可用，无法更新绰号计数或获取参考信息。")
            return

        # 获取当前上下文中用户已存在的绰号信息，作为LLM分析的参考
        existing_sobriquets_for_ref_str: Optional[str] = None
        # existing_data_for_processing 用于后续处理不可靠映射时查找原始计数
        existing_data_for_processing: Dict[str, List[Dict[str, Any]]] = {} 
        if user_ids_for_llm_ref_and_decay: # 仅当有上下文用户时才获取
            try:
                existing_data_for_processing = await self.db_handler.get_existing_sobriquets_for_users_in_group(
                    platform, group_id, user_ids_for_llm_ref_and_decay
                )
                if existing_data_for_processing: 
                    # 过滤掉那些值为空列表的用户条目，避免向LLM传递过多无效信息
                    filtered_existing_data_for_prompt = {
                        uid: sobriquets for uid, sobriquets in existing_data_for_processing.items() if sobriquets
                    }
                    if filtered_existing_data_for_prompt: # 如果过滤后仍有数据
                        existing_sobriquets_for_ref_str = format_existing_sobriquets_for_prompt(filtered_existing_data_for_prompt)
                        logger.debug(f"{log_prefix} 获取到以下已存在绰号供LLM参考:\n{existing_sobriquets_for_ref_str}")
                    else:
                        logger.debug(f"{log_prefix} 上下文用户在群组 '{group_id}' 中无已记录的有效绰号供LLM参考。")
                else: # 如果数据库未返回任何相关用户的数据
                    logger.debug(f"{log_prefix} 未能从数据库获取群组 '{group_id}' 中上下文用户的已存在绰号信息。")
            except Exception as e_get_existing:
                logger.error(f"{log_prefix} 获取已存在绰号供LLM参考时出错: {e_get_existing}", exc_info=True)
        
        # 调用LLM进行分析，传入聊天记录、机器人回复和可选的参考信息
        analysis_result = await self._call_llm_for_sobriquet_analysis(
            chat_history_str, 
            bot_reply,
            existing_sobriquets_for_ref_str 
        )

        # 步骤3: 根据LLM分析结果更新数据库
        # 处理LLM判定的可靠映射
        reliable_mappings = analysis_result.get("reliable_mappings", {}) # 安全获取，默认为空字典
        if isinstance(reliable_mappings, dict) and reliable_mappings: # 确保是字典且不为空
            logger.info(f"{log_prefix} LLM找到的可靠绰号映射: {reliable_mappings}")
            for user_id_str, sobriquet_value in reliable_mappings.items():
                if not user_id_str or not sobriquet_value or not user_id_str.isdigit(): # 基本校验
                    logger.warning(f"{log_prefix} 跳过无效的可靠映射条目: user_id='{user_id_str}', sobriquet='{sobriquet_value}'")
                    continue
                user_id_int = int(user_id_str)
                try:
                    person_id = person_info_manager.get_person_id(platform, user_id_int) 
                    if not person_id: continue # 如果无法获取person_id，则跳过
                    await self.db_handler.upsert_person(person_id, user_id_int, platform) # 确保用户存在
                    # 对可靠映射，增加其计数（默认为1）
                    await self.db_handler.update_group_sobriquet_count(person_id, group_id, sobriquet_value, increment=1)
                except Exception as e: 
                    logger.exception(f"{log_prefix} 处理可靠绰号 '{sobriquet_value}' (用户 {user_id_str}) 时出错：{e}")
        
        # 处理LLM判定的不可靠或应否定映射
        unreliable_mappings = analysis_result.get("unreliable_mappings", {}) # 安全获取
        if isinstance(unreliable_mappings, dict) and unreliable_mappings: # 确保是字典且不为空
            logger.info(f"{log_prefix} LLM找到的不可靠/应否定绰号映射: {unreliable_mappings}")
            for user_id_str, sobriquet_value in unreliable_mappings.items():
                if not user_id_str or not sobriquet_value or not user_id_str.isdigit(): # 基本校验
                    logger.warning(f"{log_prefix} 跳过无效的不可靠映射条目: user_id='{user_id_str}', sobriquet='{sobriquet_value}'")
                    continue
                user_id_int = int(user_id_str)
                try:
                    person_id = person_info_manager.get_person_id(platform, user_id_int)
                    if not person_id: continue
                    await self.db_handler.upsert_person(person_id, user_id_int, platform) # 确保用户存在
                    
                    new_count: int # 用于存储处理后的计数值
                    original_count_found = False
                    original_count = 0

                    # 尝试从为LLM准备的参考数据 existing_data_for_processing 中查找该不可靠绰号的原始计数
                    if user_id_str in existing_data_for_processing:
                        for sob_entry in existing_data_for_processing[user_id_str]:
                            if sob_entry.get("name") == sobriquet_value:
                                original_count = sob_entry.get("count", 0)
                                original_count_found = True
                                break
                    
                    if original_count_found and original_count > 0:
                        # 如果在参考数据中找到，并且原计数大于0，则按配置的比例衰减
                        new_count = max(0, int(original_count * self.unreliable_sobriquet_decay_factor))
                        logger.info(f"{log_prefix} 用户 {user_id_str} 的不可靠绰号 '{sobriquet_value}' (原计数: {original_count}) 将按因子 {self.unreliable_sobriquet_decay_factor} 衰减至 {new_count}。")
                    else:
                        # 如果不在参考数据中（LLM新发现并直接标记为不可靠），或原计数为0，则直接设置为0
                        new_count = 0 
                        logger.info(f"{log_prefix} 用户 {user_id_str} 的不可靠绰号 '{sobriquet_value}' (未在参考数据中找到或原计数为0) 将设置为 {new_count}。")
                    
                    # 使用 set_group_sobriquet_value 将计算出的新计数值更新到数据库
                    await self.db_handler.set_group_sobriquet_value(person_id, group_id, sobriquet_value, value=new_count)
                    
                except Exception as e:
                    logger.exception(f"{log_prefix} 处理不可靠绰号 '{sobriquet_value}' (用户 {user_id_str}) 时出错：{e}")

        if not reliable_mappings and not unreliable_mappings: # 如果两类映射都为空
            logger.debug(f"{log_prefix} LLM 未找到任何可靠或不可靠的绰号映射，或分析失败/返回空结果。")


    async def _call_llm_for_sobriquet_analysis(
        self,
        chat_history_str: str,
        bot_reply: str,
        existing_sobriquets_str: Optional[str] = None # 接收已存在的绰号信息字符串
    ) -> Dict[str, Any]:
        """
        内部方法：调用 LLM 分析聊天记录、Bot 回复以及可选的参考信息，
        提取可靠的 和 不可靠的 用户ID-绰号 映射。
        """
        if not self.llm_mapper: # 检查LLM映射器是否可用
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {} # 返回空字典表示分析失败或无结果

        # 构建Prompt，包含所有相关信息
        prompt = build_sobriquet_mapping_prompt(chat_history_str, bot_reply, existing_sobriquets_str) 
        logger.debug(f"构建的绰号映射 Prompt (部分):\n{prompt[:400]}...") # 记录Prompt部分内容

        try:
            # 调用LLM生成响应
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")

            if not response_content: # 如果LLM返回空内容
                logger.warning("LLM 返回了空的绰号映射内容。")
                return {}

            response_content = response_content.strip() # 去除首尾空白
            # 尝试从Markdown代码块中提取JSON
            markdown_code_regex = re.compile(r"```(?:json)?\s*\n(\{.*?\})\n\s*```", re.DOTALL | re.IGNORECASE)
            match = markdown_code_regex.search(response_content)
            json_str_to_parse = ""
            if match: # 如果匹配到Markdown代码块
                json_str_to_parse = match.group(1).strip()
            elif response_content.startswith("{") and response_content.endswith("}"): # 如果直接是JSON字符串
                json_str_to_parse = response_content
            else: # 尝试通用提取被文本包围的JSON
                json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if json_match:
                    json_str_to_parse = json_match.group(0)
                else: 
                    logger.warning(f"LLM 响应似乎不包含有效的 JSON 对象。响应: {response_content}")
                    return {}
            
            if not json_str_to_parse: # 如果提取后仍为空
                logger.warning(f"未能从LLM响应中提取出JSON内容。原始响应: {response_content}")
                return {}

            result = json.loads(json_str_to_parse) # 解析JSON字符串

            if not isinstance(result, dict): # 确保解析结果是字典
                logger.warning(f"LLM 响应不是一个有效的 JSON 对象 (字典类型)。解析后类型: {type(result)}, 内容: {json_str_to_parse}")
                return {}
            
            # 初始化返回的字典结构，确保即使LLM没有返回某个键，我们也有默认的空字典
            final_result = {
                "reliable_mappings": {},
                "unreliable_mappings": {}
            }

            # 处理可靠映射
            reliable_mappings = result.get("reliable_mappings") # 安全获取
            if isinstance(reliable_mappings, dict): # 确保是字典
                final_result["reliable_mappings"] = self._filter_llm_sobriquet_results(reliable_mappings)
            elif reliable_mappings is not None: # 如果存在但不是字典，记录警告
                logger.warning(f"LLM返回的reliable_mappings不是字典类型: {reliable_mappings}")

            # 处理不可靠映射
            unreliable_mappings = result.get("unreliable_mappings") # 安全获取
            if isinstance(unreliable_mappings, dict): # 确保是字典
                final_result["unreliable_mappings"] = self._filter_llm_sobriquet_results(unreliable_mappings)
            elif unreliable_mappings is not None: # 如果存在但不是字典，记录警告
                logger.warning(f"LLM返回的unreliable_mappings不是字典类型: {unreliable_mappings}")
            
            # 记录分析结果的统计信息
            if final_result["reliable_mappings"] or final_result["unreliable_mappings"]:
                 logger.info(f"LLM分析结果: 可靠映射 {len(final_result['reliable_mappings'])} 条, 不可靠映射 {len(final_result['unreliable_mappings'])} 条。")
            else:
                logger.info("LLM分析未找到任何有效映射或所有映射均被过滤/返回空。")
            
            return final_result # 返回包含两类映射的字典

        except json.JSONDecodeError as json_err: 
            logger.error(f"解析 LLM 响应 JSON 失败: {json_err}\n待解析内容: '{json_str_to_parse}'\n原始响应: {response_content}")
            return {} # 解析失败返回空字典
        except Exception as e: 
            logger.error(f"绰号映射 LLM 调用或处理过程中发生意外错误: {e}", exc_info=True)
            return {} # 其他异常也返回空字典

    def _filter_llm_sobriquet_results(self, original_data: Dict[str, str]) -> Dict[str, str]:
        """
        过滤 LLM 返回的绰号映射结果，主要移除机器人自身和空绰号。
        """
        filtered_data = {}
        # 安全获取机器人QQ号，如果配置不存在则为None
        bot_qq_str = getattr(global_config.bot, 'qq_account', None)

        for user_id, sobriquet_val in original_data.items():
            if not isinstance(user_id, str): # 确保 user_id 是字符串
                logger.warning(f"过滤掉非字符串 user_id: {user_id}")
                continue
            if bot_qq_str and user_id == bot_qq_str: # 过滤掉机器人自己
                logger.debug(f"过滤掉机器人自身的映射: ID {user_id}")
                continue
            if not sobriquet_val or not sobriquet_val.strip(): # 过滤掉空或仅含空白的绰号
                logger.debug(f"过滤掉用户 {user_id} 的空绰号。")
                continue

            filtered_data[user_id] = sobriquet_val.strip() # 存储处理过的绰号
        return filtered_data

    def _run_processor_in_thread(self):
        """后台线程的入口函数，负责创建和管理该线程的 asyncio 事件循环。"""
        thread_id = threading.get_ident() # 获取当前线程ID
        logger.info(f"绰号处理器后台线程启动 (线程 ID: {thread_id})...")
        loop = asyncio.new_event_loop() # 为新线程创建新的事件循环
        asyncio.set_event_loop(loop) # 将此循环设置为当前线程的事件循环
        logger.info(f"(线程 ID: {thread_id}) 绰号处理 Asyncio 事件循环已创建并设置。")
        
        run_async_loop(loop, self._processing_loop()) # 运行主处理循环
        
        logger.info(f"绰号处理器后台线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
        """后台线程中运行的异步处理循环，持续从队列中获取并处理绰号分析任务。"""
        logger.info("绰号异步处理循环已启动。")
        while not self._stop_event.is_set(): # 只要停止事件未被设置，就持续循环
            try:
                # 异步等待从队列中获取任务项，设置超时以允许定期检查停止事件
                item = await asyncio.wait_for(self.sobriquet_queue.get(), timeout=self.sleep_interval)
                await self._analyze_and_update_sobriquets(item) # 处理获取到的任务
                self.sobriquet_queue.task_done() # 通知队列该任务已完成
            except asyncio.TimeoutError: # 等待超时是正常的，继续下一次循环以检查停止事件
                continue
            except asyncio.CancelledError: # pragma: no cover ; 如果任务被取消
                logger.info("绰号处理循环被取消。")
                break # 退出循环
            except Exception as e: # pragma: no cover ; 捕获其他意外错误
                logger.error(f"绰号处理循环中发生未捕获错误: {e}", exc_info=True)
                await asyncio.sleep(5) # 发生错误时短暂休眠，避免快速连续失败刷屏
        logger.info("绰号异步处理循环已结束。")

# 创建 SobriquetManager 的单例实例，供项目其他部分使用
sobriquet_manager = SobriquetManager()
