import asyncio
import threading
import random
import time
import json
import re
from typing import Dict, Optional, List, Any, Tuple, Union
from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from .sobriquet_db import SobriquetDB # 假设 SobriquetDB 已经更新为使用 group_sobriquets 和 strength
from .sobriquet_mapper import build_sobriquet_mapping_prompt, format_existing_sobriquets_for_prompt
from .sobriquet_utils import select_sobriquets_for_prompt
from src.chat.person_info.person_info import person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager # 假设 RelationshipManager 已更新
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("SobriquetManager")
logger_helper = get_logger("AsyncLoopHelperSobriquet")

def run_async_loop(loop: asyncio.AbstractEventLoop, coro):
    try:
        logger_helper.debug(f"在事件循环 {id(loop)} 中运行协程 (用于绰号处理)...")
        result = loop.run_until_complete(coro)
        logger_helper.debug(f"事件循环 {id(loop)} 中的协程已完成 (用于绰号处理)。")
        return result
    except asyncio.CancelledError:
        logger_helper.info(f"事件循环 {id(loop)} 中的协程被取消 (用于绰号处理)。")
    except Exception as e:
        logger_helper.error(f"事件循环 {id(loop)} 中发生错误 (用于绰号处理): {e}", exc_info=True)
    finally:
        try:
            all_tasks = asyncio.all_tasks(loop)
            current_task = asyncio.current_task(loop)
            tasks_to_cancel = [
                task for task in all_tasks if task is not current_task
            ]
            if tasks_to_cancel:
                logger_helper.info(f"正在取消事件循环 {id(loop)} 中的 {len(tasks_to_cancel)} 个剩余任务...")
                for task in tasks_to_cancel:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logger_helper.info(f"事件循环 {id(loop)} 中的剩余任务已取消。")
            if loop.is_running():
                loop.stop()
                logger_helper.info(f"Asyncio 事件循环 {id(loop)} 已停止。")
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
                logger_helper.info(f"Asyncio 事件循环 {id(loop)} 已关闭。")
        except Exception as close_err:
            logger_helper.error(f"清理 Asyncio 事件循环 {id(loop)} 时发生错误: {close_err}", exc_info=True)

class SobriquetManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logger.info("正在创建 SobriquetManager 单例实例...")
                    cls._instance = super(SobriquetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            logger.info("正在初始化 SobriquetManager 组件...")
            self.is_analysis_enabled = global_config.profile.enable_sobriquet_mapping

            person_info_collection = getattr(db, "person_info", None)
            self.db_handler = SobriquetDB(person_info_collection)
            if not self.db_handler.is_available():
                logger.error("数据库处理器初始化失败，SobriquetManager 功能受限。")

            self.llm_mapper: Optional[LLMRequest] = None
            if self.is_analysis_enabled:
                try:
                    model_config_data = global_config.model.sobriquet_mapping
                    self.llm_mapper = LLMRequest(
                        model=model_config_data,
                        temperature=model_config_data.get('temp', 0.5),
                        max_tokens=model_config_data.get('max_tokens', 256),
                        request_type="sobriquet_mapping",
                    )
                    logger.info("绰号映射 LLM 映射器初始化成功。")
                except AttributeError as ae:
                     logger.error(f"初始化绰号映射 LLM 时配置路径错误: {ae}。请检查 global_config.model.sobriquet_mapping 是否正确。绰号分析功能禁用。", exc_info=True)
                     self.llm_mapper = None
                except KeyError as ke:
                    logger.error(f"初始化绰号映射 LLM 时缺少配置项: {ke}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
                except Exception as e:
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
            
            self.queue_max_size = global_config.profile.sobriquet_queue_max_size
            self.sobriquet_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size)
            self._stop_event = threading.Event()
            self._sobriquet_thread: Optional[threading.Thread] = None
            self.sleep_interval = global_config.profile.sobriquet_process_sleep_interval
            
            self.min_strength_for_prompt = float(global_config.profile.min_sobriquet_strength_for_prompt_injection)
            if self.min_strength_for_prompt > 0: # 阈值仍应大于0才有意义
                logger.info(f"将对注入Prompt的绰号应用最低映射强度阈值: {self.min_strength_for_prompt:.2f}")
            
            # 事件衰减值 (数值减法)
            # 配置项名建议改为 sobriquet_event_decay_value
            self.event_decay_value = float(global_config.profile.sobriquet_event_decay_value)
            if self.event_decay_value < 0.0: # 衰减值不能为负
                logger.warning(f"事件衰减值 sobriquet_event_decay_value ({self.event_decay_value:.2f}) 配置无效 (不能为负)。将设置为0.0 (不衰减)。")
                self.event_decay_value = 0.0
            
            if self.event_decay_value > 0.0:
                 logger.info(f"已配置事件触发式绰号强度衰减，每次衰减值: {self.event_decay_value:.2f}")
            else: # event_decay_value == 0.0
                 logger.info("事件触发式绰号强度衰减未启用 (衰减值为0.0)。")

            # 不可靠绰号衰减因子 (比例)
            self.unreliable_sobriquet_decay_factor = float(global_config.profile.unreliable_sobriquet_decay_factor)
            if not (0.0 <= self.unreliable_sobriquet_decay_factor <= 1.0):
                logger.warning(f"不可靠绰号衰减因子 unreliable_sobriquet_decay_factor ({self.unreliable_sobriquet_decay_factor:.2f}) 配置无效，应在[0.0, 1.0]之间。将设置为0.1。")
                self.unreliable_sobriquet_decay_factor = 0.1
            logger.info(f"LLM判定的不可信绰号将按衰减因子 {self.unreliable_sobriquet_decay_factor:.2f} 进行强度调整。")
            
            # 可靠绰号映射的强度增量值
            self.reliable_mapping_strength_increment = float(global_config.profile.reliable_mapping_strength_increment)
            if self.reliable_mapping_strength_increment < 0: # 增量不应为负
                logger.warning(f"可靠绰号映射强度增量 reliable_mapping_strength_increment ({self.reliable_mapping_strength_increment:.2f}) 不应为负，已设为1.0。")
                self.reliable_mapping_strength_increment = 1.0
            logger.info(f"可靠绰号映射将增加强度值: {self.reliable_mapping_strength_increment:.2f}")


            self._initialized = True
            logger.info("SobriquetManager 初始化完成。")

    def start_processor(self):
        if not self.is_analysis_enabled:
            logger.info("绰号分析功能已禁用，处理器未启动。")
            return
        if global_config.profile.max_sobriquets_in_prompt == 0:
            logger.warning("配置中 max_sobriquets_in_prompt 为0，可能不需要进行绰号分析或注入。")

        if self._sobriquet_thread is None or not self._sobriquet_thread.is_alive():
            logger.info("正在启动绰号处理器线程...")
            self._stop_event.clear()
            self._sobriquet_thread = threading.Thread(
                target=self._run_processor_in_thread,
                daemon=True,
            )
            self._sobriquet_thread.start()
            logger.info(f"绰号处理器线程已启动 (ID: {self._sobriquet_thread.ident})")
        else:
            logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        if self._sobriquet_thread and self._sobriquet_thread.is_alive():
            logger.info("正在停止绰号处理器线程...")
            self._stop_event.set()
            try:
                self._sobriquet_thread.join(timeout=10)
                if self._sobriquet_thread.is_alive():
                    logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e:
                logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._sobriquet_thread and not self._sobriquet_thread.is_alive():
                    logger.info("绰号处理器线程已成功停止。")
                self._sobriquet_thread = None
        else:
            logger.info("绰号处理器线程未在运行或已被清理。")

    async def trigger_sobriquet_analysis(
        self,
        anchor_message: MessageRecv,
        bot_reply: List[str],      
        chat_stream: Optional[ChatStream] = None,
    ):
        if not self.is_analysis_enabled:
            return

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info:
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        if random.random() > global_config.profile.sobriquet_analysis_probability:
            logger.debug("跳过绰号分析：随机概率未命中。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
            history_limit = global_config.profile.sobriquet_analysis_history_limit
            history_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=current_chat_stream.stream_id,
                timestamp=time.time(),
                limit=history_limit,
            )
            chat_history_str = await build_readable_messages(
                messages=history_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
                truncate=False,
            )
            bot_reply_str = " ".join(bot_reply) if bot_reply else ""
            group_id = str(current_chat_stream.group_info.group_id)
            platform = current_chat_stream.platform

            user_ids_in_current_context = list(set(
                str(msg["user_info"]["user_id"])
                for msg in history_messages
                if msg.get("user_info", {}).get("user_id")
            ))

            logger.debug(f"{log_prefix} 为事件衰减和LLM参考收集到的上下文用户ID: {user_ids_in_current_context}")

            item = (chat_history_str, bot_reply_str, platform, group_id, user_ids_in_current_context)
            await self._add_to_queue(item, platform, group_id)

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_selected_sobriquets_for_group(
        self, platform: str, user_ids_in_context: List[str], group_id: str
    ) -> Optional[List[Tuple[str, str, str, float]]]:
        max_sobriquets_to_inject = global_config.profile.max_sobriquets_in_prompt
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
            if user_ids_in_context:
                # 假设 relationship_manager.get_users_group_sobriquets 返回正确的结构
                all_sobriquets_data_by_actual_name = await relationship_manager.get_users_group_sobriquets(
                    platform, user_ids_in_context, group_id
                )
                logger.debug(f"{log_prefix} 从 relationship_manager 获取到的原始绰号数据类型: {type(all_sobriquets_data_by_actual_name)}")


            if not isinstance(all_sobriquets_data_by_actual_name, dict):
                logger.error(f"{log_prefix} 从 relationship_manager 获取的绰号数据不是预期的字典类型，而是 {type(all_sobriquets_data_by_actual_name)}。无法继续。")
                return None
                
            if not all_sobriquets_data_by_actual_name:
                logger.debug(f"{log_prefix} 未找到群组 '{group_id}' 的原始绰号数据或数据为空。")
                return None

            selected_sobriquets_candidates = select_sobriquets_for_prompt(all_sobriquets_data_by_actual_name)

            if not selected_sobriquets_candidates:
                logger.debug(f"{log_prefix} `select_sobriquets_for_prompt` 未选出任何候选绰号。")
                return None
            
            logger.debug(f"{log_prefix} 初始选出的候选绰号 (共{len(selected_sobriquets_candidates)}个): {selected_sobriquets_candidates}")

            final_selected_sobriquets: List[Tuple[str, str, str, float]] = []
            if self.min_strength_for_prompt > 0:
                for sobriquet_tuple in selected_sobriquets_candidates:
                    if sobriquet_tuple[3] >= self.min_strength_for_prompt:
                        final_selected_sobriquets.append(sobriquet_tuple)
                    else:
                        logger.debug(f"{log_prefix} 绰号 '{sobriquet_tuple[2]}' (用户ID: {sobriquet_tuple[1]}, 强度: {sobriquet_tuple[3]:.2f}) "
                                     f"因低于阈值 {self.min_strength_for_prompt:.2f} 而被排除出Prompt。")
                
                if not final_selected_sobriquets and selected_sobriquets_candidates:
                     logger.info(f"{log_prefix} 所有初始选择的绰号都低于最低映射强度阈值 {self.min_strength_for_prompt:.2f}。")
            else: # min_strength_for_prompt <= 0，不过滤
                final_selected_sobriquets = selected_sobriquets_candidates
            
            if final_selected_sobriquets:
                # 限制最终选择注入的数量
                if len(final_selected_sobriquets) > max_sobriquets_to_inject:
                    logger.info(f"{log_prefix} 满足条件的绰号数量 ({len(final_selected_sobriquets)}) 超过最大注入限制 ({max_sobriquets_to_inject})，将截取强度最高的。")
                    # 确保已按强度排序 (select_sobriquets_for_prompt 应该已经做了)
                    final_selected_sobriquets.sort(key=lambda x: x[3], reverse=True)
                    final_selected_sobriquets = final_selected_sobriquets[:max_sobriquets_to_inject]

                logger.info(f"{log_prefix} 为群组 '{group_id}' 最终选择了以下用户常用绰号进行注入 (共{len(final_selected_sobriquets)}个): {final_selected_sobriquets}")
            else:
                logger.info(f"{log_prefix} 没有绰号满足注入条件 (强度阈值: {self.min_strength_for_prompt:.2f})。")
                
            return final_selected_sobriquets
            
        except Exception as e:
            logger.error(f"{log_prefix} 获取群组 '{group_id}' 的绰号以供注入时出错: {e}", exc_info=True)
            return None

    async def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        try:
            await self.sobriquet_queue.put(item)
            logger.debug(
                f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号分析队列。当前大小: {self.sobriquet_queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"绰号分析队列已满 (最大={self.queue_max_size})。平台 '{platform}' 群组 '{group_id}' 的项目被丢弃。"
            )
        except Exception as e:
            logger.error(f"将项目添加到绰号分析队列时出错: {e}", exc_info=True)

    async def _analyze_and_update_sobriquets(self, item: tuple):
        if not isinstance(item, tuple) or len(item) != 5:
            logger.warning(f"从队列接收到无效项目: 类型 {type(item)}, 期望长度 5 但得到 {len(item) if isinstance(item, tuple) else '非元组'}")
            return

        chat_history_str, bot_reply, platform, group_id, user_ids_for_llm_ref_and_decay = item
        log_prefix = f"[{platform}:{group_id}]"
        logger.debug(f"{log_prefix} 开始处理绰号分析任务。上下文用户ID (用于衰减和LLM参考): {user_ids_for_llm_ref_and_decay}")
        
        loop = asyncio.get_running_loop()

        # 事件触发式衰减 - 使用减法衰减值
        # 只有当 event_decay_value > 0 时才执行衰减
        if self.event_decay_value > 0.0 and self.db_handler.is_available():
            if user_ids_for_llm_ref_and_decay:
                try:
                    decayed_count = await loop.run_in_executor(
                        None,
                        self.db_handler.decay_sobriquets_in_group,
                        platform,
                        group_id,
                        user_ids_for_llm_ref_and_decay,
                        self.event_decay_value
                    )
                    if decayed_count > 0 :
                        logger.info(f"{log_prefix} 对 {len(user_ids_for_llm_ref_and_decay)} 个目标用户的共 {decayed_count} 个绰号条目应用了事件强度衰减 (衰减值: {self.event_decay_value:.2f})。")
                    else:
                        logger.debug(f"{log_prefix} 事件强度衰减未影响任何绰号条目（可能目标用户无绰号或强度已为0）。")
                except Exception as e_decay:
                    logger.error(f"{log_prefix} 执行绰号事件强度衰减时出错: {e_decay}", exc_info=True)
            else:
                logger.debug(f"{log_prefix} 跳过事件触发式强度衰减，因未提供上下文用户ID列表。")

        if not self.llm_mapper:
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行绰号分析。")
            return
        if not self.db_handler.is_available():
            logger.error(f"{log_prefix} 数据库处理器不可用，无法更新绰号强度或获取参考信息。")
            return

        existing_sobriquets_for_ref_str: Optional[str] = None
        existing_data_for_processing: Dict[str, List[Dict[str, Any]]] = {}
        if user_ids_for_llm_ref_and_decay:
            try:
                existing_data_for_processing = await loop.run_in_executor(
                    None,
                    self.db_handler.get_existing_sobriquets_for_users_in_group,
                    platform, group_id, user_ids_for_llm_ref_and_decay
                )
                if existing_data_for_processing:
                    filtered_existing_data_for_prompt = {
                        uid: sobriquets for uid, sobriquets in existing_data_for_processing.items() if sobriquets
                    }
                    if filtered_existing_data_for_prompt:
                        existing_sobriquets_for_ref_str = format_existing_sobriquets_for_prompt(filtered_existing_data_for_prompt)
                        logger.debug(f"{log_prefix} 获取到以下已存在绰号供LLM参考:\n{existing_sobriquets_for_ref_str}")
                    else:
                        logger.debug(f"{log_prefix} 上下文用户在群组 '{group_id}' 中无已记录的有效绰号供LLM参考。")
                else:
                    logger.debug(f"{log_prefix} 未能从数据库获取群组 '{group_id}' 中上下文用户的已存在绰号信息。")
            except Exception as e_get_existing:
                logger.error(f"{log_prefix} 获取已存在绰号供LLM参考时出错: {e_get_existing}", exc_info=True)
        
        analysis_result = await self._call_llm_for_sobriquet_analysis(
            chat_history_str,
            bot_reply,
            existing_sobriquets_for_ref_str
        )

        reliable_mappings = analysis_result.get("reliable_mappings", {})
        if isinstance(reliable_mappings, dict) and reliable_mappings:
            logger.info(f"{log_prefix} LLM找到的可靠绰号映射: {reliable_mappings}")
            for user_id_str, sobriquet_value in reliable_mappings.items():
                if not user_id_str or not sobriquet_value or not user_id_str.isdigit():
                    logger.warning(f"{log_prefix} 跳过无效的可靠映射条目: user_id='{user_id_str}', sobriquet='{sobriquet_value}'")
                    continue
                user_id_int = int(user_id_str)
                try:
                    person_id = person_info_manager.get_person_id(platform, user_id_int)
                    if not person_id: continue
                    await loop.run_in_executor(None, self.db_handler.upsert_person, person_id, user_id_int, platform)
                    await loop.run_in_executor(None, self.db_handler.update_group_sobriquet_strength, person_id, group_id, sobriquet_value, self.reliable_mapping_strength_increment)
                except Exception as e:
                    logger.exception(f"{log_prefix} 处理可靠绰号 '{sobriquet_value}' (用户 {user_id_str}) 时出错：{e}")
        
        unreliable_mappings = analysis_result.get("unreliable_mappings", {})
        if isinstance(unreliable_mappings, dict) and unreliable_mappings:
            logger.info(f"{log_prefix} LLM找到的不可信/应否定绰号映射: {unreliable_mappings}")
            for user_id_str, sobriquet_value in unreliable_mappings.items():
                if not user_id_str or not sobriquet_value or not user_id_str.isdigit():
                    logger.warning(f"{log_prefix} 跳过无效的不可信映射条目: user_id='{user_id_str}', sobriquet='{sobriquet_value}'")
                    continue
                user_id_int = int(user_id_str)
                try:
                    person_id = person_info_manager.get_person_id(platform, user_id_int)
                    if not person_id: continue
                    await loop.run_in_executor(None, self.db_handler.upsert_person, person_id, user_id_int, platform)
                    
                    new_strength: float
                    original_strength_found = False
                    original_strength = 0.0

                    if user_id_str in existing_data_for_processing:
                        for sob_entry in existing_data_for_processing[user_id_str]:
                            if sob_entry.get("name") == sobriquet_value:
                                original_strength = float(sob_entry.get("strength", 0.0))
                                original_strength_found = True
                                break
                    
                    if original_strength_found and original_strength > 0.0:
                        new_strength = max(0.0, original_strength * self.unreliable_sobriquet_decay_factor)
                        logger.info(f"{log_prefix} 用户 {user_id_str} 的不可信绰号 '{sobriquet_value}' (原强度: {original_strength:.2f}) 将按因子 {self.unreliable_sobriquet_decay_factor:.2f} 衰减至 {new_strength:.2f}。")
                    else:
                        new_strength = 0.0
                        logger.info(f"{log_prefix} 用户 {user_id_str} 的不可信绰号 '{sobriquet_value}' (未在参考数据中找到或原强度为0) 将设置为 {new_strength:.2f}。")
                    
                    await loop.run_in_executor(None, self.db_handler.set_group_sobriquet_strength, person_id, group_id, sobriquet_value, new_strength)
                    
                except Exception as e:
                    logger.exception(f"{log_prefix} 处理不可信绰号 '{sobriquet_value}' (用户 {user_id_str}) 时出错：{e}")

        if not reliable_mappings and not unreliable_mappings:
            logger.debug(f"{log_prefix} LLM 未找到任何可靠或不可信的绰号映射，或分析失败/返回空结果。")


    async def _call_llm_for_sobriquet_analysis(
        self,
        chat_history_str: str,
        bot_reply: str,
        existing_sobriquets_str: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.llm_mapper:
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {}

        prompt = build_sobriquet_mapping_prompt(chat_history_str, bot_reply, existing_sobriquets_str)
        logger.debug(f"构建的绰号映射 Prompt :\n{prompt}...")

        try:
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")

            if not response_content:
                logger.warning("LLM 返回了空的绰号映射内容。")
                return {}

            response_content = response_content.strip()
            markdown_code_regex = re.compile(r"```(?:json)?\s*\n(\{.*?\})\n\s*```", re.DOTALL | re.IGNORECASE)
            match = markdown_code_regex.search(response_content)
            json_str_to_parse = ""
            if match:
                json_str_to_parse = match.group(1).strip()
            elif response_content.startswith("{") and response_content.endswith("}"):
                json_str_to_parse = response_content
            else:
                json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if json_match:
                    json_str_to_parse = json_match.group(0)
                else:
                    logger.warning(f"LLM 响应似乎不包含有效的 JSON 对象。响应: {response_content}")
                    return {}
            
            if not json_str_to_parse:
                logger.warning(f"未能从LLM响应中提取出JSON内容。原始响应: {response_content}")
                return {}

            result = json.loads(json_str_to_parse)

            if not isinstance(result, dict):
                logger.warning(f"LLM 响应不是一个有效的 JSON 对象 (字典类型)。解析后类型: {type(result)}, 内容: {json_str_to_parse}")
                return {}
            
            final_result = {
                "reliable_mappings": {},
                "unreliable_mappings": {} # 之前这里是 unreliable_mappings，保持一致
            }

            reliable_mappings = result.get("reliable_mappings")
            if isinstance(reliable_mappings, dict):
                final_result["reliable_mappings"] = self._filter_llm_sobriquet_results(reliable_mappings)
            elif reliable_mappings is not None:
                logger.warning(f"LLM返回的reliable_mappings不是字典类型: {reliable_mappings}")


            unreliable_mappings = result.get("unreliable_mappings") # 之前这里是 unreliable_mappings
            if isinstance(unreliable_mappings, dict):
                final_result["unreliable_mappings"] = self._filter_llm_sobriquet_results(unreliable_mappings)
            elif unreliable_mappings is not None:
                logger.warning(f"LLM返回的unreliable_mappings不是字典类型: {unreliable_mappings}")
            
            if final_result["reliable_mappings"] or final_result["unreliable_mappings"]:
                 logger.info(f"LLM分析结果: 可靠映射 {len(final_result['reliable_mappings'])} 条, 不可信映射 {len(final_result['unreliable_mappings'])} 条。")
            else:
                logger.info("LLM分析未找到任何有效映射或所有映射均被过滤/返回空。")
            
            return final_result

        except json.JSONDecodeError as json_err:
            logger.error(f"解析 LLM 响应 JSON 失败: {json_err}\n待解析内容: '{json_str_to_parse}'\n原始响应: {response_content}")
            return {}
        except Exception as e:
            logger.error(f"绰号映射 LLM 调用或处理过程中发生意外错误: {e}", exc_info=True)
            return {}

    def _filter_llm_sobriquet_results(self, original_data: Dict[str, str]) -> Dict[str, str]:
        filtered_data = {}
        bot_qq_str = getattr(global_config.bot, 'qq_account', None)

        for user_id, sobriquet_val in original_data.items():
            if not isinstance(user_id, str):
                logger.warning(f"过滤掉非字符串 user_id: {user_id}")
                continue
            if bot_qq_str and user_id == bot_qq_str:
                logger.debug(f"过滤掉机器人自身的映射: ID {user_id}")
                continue
            if not sobriquet_val or not sobriquet_val.strip():
                logger.debug(f"过滤掉用户 {user_id} 的空绰号。")
                continue

            filtered_data[user_id] = sobriquet_val.strip()
        return filtered_data

    def _run_processor_in_thread(self):
        thread_id = threading.get_ident()
        logger.info(f"绰号处理器后台线程启动 (线程 ID: {thread_id})...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"(线程 ID: {thread_id}) 绰号处理 Asyncio 事件循环已创建并设置。")
        
        run_async_loop(loop, self._processing_loop())
        
        logger.info(f"绰号处理器后台线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
        logger.info("绰号异步处理循环已启动。")
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self.sobriquet_queue.get(), timeout=self.sleep_interval)
                await self._analyze_and_update_sobriquets(item)
                self.sobriquet_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("绰号处理循环被取消。")
                break
            except Exception as e:
                logger.error(f"绰号处理循环中发生未捕获错误: {e}", exc_info=True)
                await asyncio.sleep(5)
        logger.info("绰号异步处理循环已结束。")

sobriquet_manager = SobriquetManager()
