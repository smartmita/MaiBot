import asyncio
import threading
import random
import time
import json
import re
from typing import Dict, Optional, List, Any, Tuple
from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from .sobriquet_db import SobriquetDB # import 更新
from .sobriquet_mapper import build_sobriquet_mapping_prompt # import 更新
from .sobriquet_utils import select_sobriquets_for_prompt # import 更新
from src.chat.person_info.person_info import person_info_manager
# relationship_manager 用于获取用户实际昵称和原始绰号数据，仍然需要
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("SobriquetManager") # logger 名称更新
logger_helper = get_logger("AsyncLoopHelperSobriquet") # 为辅助函数创建单独的 logger, 名称区分


def run_async_loop(loop: asyncio.AbstractEventLoop, coro):
    """
    运行给定的协程直到完成，并确保循环最终关闭。

    Args:
        loop: 要使用的 asyncio 事件循环。
        coro: 要在循环中运行的主协程。
    """
    try:
        logger_helper.debug(f"Running coroutine in loop {id(loop)} for sobriquet processing...") # 日志内容稍作区分
        result = loop.run_until_complete(coro)
        logger_helper.debug(f"Coroutine completed in loop {id(loop)} for sobriquet processing.")
        return result
    except asyncio.CancelledError:
        logger_helper.info(f"Coroutine in loop {id(loop)} for sobriquet processing was cancelled.")
    except Exception as e:
        logger_helper.error(f"Error in async loop {id(loop)} for sobriquet processing: {e}", exc_info=True)
    finally:
        try:
            all_tasks = asyncio.all_tasks(loop)
            current_task = asyncio.current_task(loop)
            tasks_to_cancel = [
                task for task in all_tasks if task is not current_task
            ]
            if tasks_to_cancel:
                logger_helper.info(f"Cancelling {len(tasks_to_cancel)} outstanding tasks in sobriquet loop {id(loop)}...")
                for task in tasks_to_cancel:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logger_helper.info(f"Outstanding tasks cancelled in sobriquet loop {id(loop)}.")

            if loop.is_running():
                loop.stop()
                logger_helper.info(f"Asyncio sobriquet loop {id(loop)} stopped.")

            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
                logger_helper.info(f"Asyncio sobriquet loop {id(loop)} closed.")
        except Exception as close_err:
            logger_helper.error(f"Error during asyncio sobriquet loop cleanup for loop {id(loop)}: {close_err}", exc_info=True)


class SobriquetManager: # 类名更新
    """
    管理群组绰号分析、处理、存储和使用的单例类。
    封装了 LLM 调用、后台处理线程和数据库交互。
    此类专注于绰号的发现和原始数据的提供。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logger.info("正在创建 SobriquetManager 单例实例...") # 日志内容更新
                    cls._instance = super(SobriquetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            logger.info("正在初始化 SobriquetManager 组件...") # 日志内容更新
            # enable_nickname_mapping 控制的是绰号分析功能，此配置键名保持不变
            self.is_analysis_enabled = global_config.group_nickname.enable_nickname_mapping

            person_info_collection = getattr(db, "person_info", None)
            self.db_handler = SobriquetDB(person_info_collection) # 使用 SobriquetDB
            if not self.db_handler.is_available():
                logger.error("数据库处理器初始化失败，SobriquetManager 功能受限。")

            self.llm_mapper: Optional[LLMRequest] = None
            if self.is_analysis_enabled:
                try:
                    # nickname_mapping 模型配置键名保持不变
                    model_config = global_config.model.sobriquet_mapping
                    if model_config and model_config.get("name"):
                        self.llm_mapper = LLMRequest(
                            model=model_config,
                            temperature=model_config.get("temp", 0.5),
                            max_tokens=model_config.get("max_tokens", 256),
                            request_type="nickname_mapping", # request_type 保持不变，可能被外部使用
                        )
                        logger.info("绰号映射 LLM 映射器初始化成功。")
                    else:
                        logger.warning("绰号映射 LLM 配置无效或缺失 'name'，绰号分析功能禁用。")
                except KeyError as ke:
                    logger.error(f"初始化绰号映射 LLM 时缺少配置项: {ke}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
                except Exception as e:
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
            
            # 队列和线程 (用于绰号分析)
            # nickname_queue_max_size 配置键名不变
            self.queue_max_size = global_config.group_nickname.nickname_queue_max_size
            self.sobriquet_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size) # 队列名更新
            self._stop_event = threading.Event()
            self._sobriquet_thread: Optional[threading.Thread] = None # 线程名更新
            # nickname_process_sleep_interval 配置键名不变
            self.sleep_interval = global_config.group_nickname.nickname_process_sleep_interval

            self._initialized = True
            logger.info("SobriquetManager 初始化完成。") # 日志内容更新

    def start_processor(self):
        """启动后台绰号处理线程（如果已启用且未运行）。"""
        if not self.is_analysis_enabled:
            logger.info("绰号分析功能已禁用，处理器未启动。")
            return
        # max_nicknames_in_prompt 配置键名不变 (这个配置现在由ProfileManager/ProfileUtils使用，但SobriquetManager的启动可能仍受其影响)
        if global_config.group_nickname.max_nicknames_in_prompt == 0:
            # 此处逻辑是如果prompt不注入绰号，可能也不需要分析。可以保留。
            logger.error("[错误] max_nicknames_in_prompt 配置为0，绰号分析功能可能受影响或不必要。")


        if self._sobriquet_thread is None or not self._sobriquet_thread.is_alive(): # 变量更新
            logger.info("正在启动绰号处理器线程...")
            self._stop_event.clear()
            self._sobriquet_thread = threading.Thread( # 变量更新
                target=self._run_processor_in_thread,
                daemon=True,
            )
            self._sobriquet_thread.start() # 变量更新
            logger.info(f"绰号处理器线程已启动 (ID: {self._sobriquet_thread.ident})") # 变量更新
        else:
            logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        """停止后台绰号处理线程。"""
        if self._sobriquet_thread and self._sobriquet_thread.is_alive(): # 变量更新
            logger.info("正在停止绰号处理器线程...")
            self._stop_event.set()
            try:
                self._sobriquet_thread.join(timeout=10) # 变量更新
                if self._sobriquet_thread.is_alive(): # 变量更新
                    logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e:
                logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._sobriquet_thread and not self._sobriquet_thread.is_alive(): # 变量更新
                    logger.info("绰号处理器线程已成功停止。")
                self._sobriquet_thread = None # 变量更新
        else:
            logger.info("绰号处理器线程未在运行或已被清理。")

    async def trigger_sobriquet_analysis( # 方法名更新
        self,
        anchor_message: MessageRecv,
        bot_reply: List[str],
        chat_stream: Optional[ChatStream] = None,
    ):
        """
        准备数据并将其排队等待绰号分析（如果满足条件）。
        """
        if not self.is_analysis_enabled: # 依赖总开关
            return

        # nickname_analysis_probability 配置键名不变
        # if random.random() > global_config.group_nickname.nickname_analysis_probability:
        #     logger.debug("跳过绰号分析：随机概率未命中。")
        #     return

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info:
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
            # nickname_analysis_history_limit 配置键名不变
            history_limit = global_config.group_nickname.nickname_analysis_history_limit
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

            item = (chat_history_str, bot_reply_str, platform, group_id)
            await self._add_to_queue(item, platform, group_id)

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_selected_sobriquets_for_group( # 新方法：用于获取已处理和选择的绰号数据
        self, platform: str, user_ids_in_context: List[str], group_id: str
    ) -> Optional[List[Tuple[str, str, str, int]]]:
        """
        获取指定用户在特定群组中经过筛选和选择的常用绰号。
        此方法供 ProfileManager 调用。
        """
        if not self.is_analysis_enabled : # 如果总的绰号分析功能未开启，则不提供绰号
            # 或者可以考虑一个独立的“是否在prompt中显示已存储绰号”的开关
            logger.debug(f"Sobriquet analysis is disabled, not providing sobriquets for prompt. Group: {group_id}")
            return None
        if global_config.group_nickname.max_nicknames_in_prompt == 0:
             logger.debug(f"max_nicknames_in_prompt is 0, not providing sobriquets for prompt. Group: {group_id}")
             return None


        log_prefix = f"[{platform}:{group_id}]"
        selected_sobriquets_for_formatting: Optional[List[Tuple[str, str, str, int]]] = None
        try:
            # 注意：get_users_group_nicknames 的第三个参数是 group_id_str
            # 它返回的结构是 { "用户实际昵称1": {"user_id": "uid1", "nicknames": [{"绰号A": 次数}, ...]}, ... }
            # "nicknames" 这个key是数据库历史遗留，代表的就是该用户在该群的绰号列表和次数
            all_sobriquets_data_by_actual_name: Dict[str, Dict[str, Any]] = {}
            if user_ids_in_context: # 确保有用户ID才去查询
                all_sobriquets_data_by_actual_name = await relationship_manager.get_users_group_nicknames(
                    platform, user_ids_in_context, group_id
                )

            if all_sobriquets_data_by_actual_name:
                # select_sobriquets_for_prompt 内部处理权重选择和数量限制
                selected_sobriquets_for_formatting = select_sobriquets_for_prompt(all_sobriquets_data_by_actual_name)
                if selected_sobriquets_for_formatting:
                    logger.debug(f"{log_prefix} 为群组 '{group_id}' 选择了以下用户常用绰号: {selected_sobriquets_for_formatting}")
            return selected_sobriquets_for_formatting
        except Exception as e:
            logger.error(f"{log_prefix} 获取群组 '{group_id}' 的绰号以供注入时出错: {e}", exc_info=True)
            return None


    # 私有/内部方法保持不变，仅更新变量名和日志信息中的称呼
    async def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        try:
            await self.sobriquet_queue.put(item) # 队列名更新
            logger.debug(
                f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号队列。当前大小: {self.sobriquet_queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"绰号队列已满 (最大={self.queue_max_size})。平台 '{platform}' 群组 '{group_id}' 的项目被丢弃。"
            )
        except Exception as e:
            logger.error(f"将项目添加到绰号队列时出错: {e}", exc_info=True)

    async def _analyze_and_update_sobriquets(self, item: tuple): # 方法名更新
        if not isinstance(item, tuple) or len(item) != 4:
            logger.warning(f"从队列接收到无效项目: {type(item)}, 期望长度 4 但得到 {len(item) if isinstance(item, tuple) else '非元组'}")
            return

        chat_history_str, bot_reply, platform, group_id = item
        log_prefix = f"[{platform}:{group_id}]"
        logger.debug(f"{log_prefix} 开始处理绰号分析任务...")

        if not self.llm_mapper:
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行分析。")
            return
        if not self.db_handler.is_available():
            logger.error(f"{log_prefix} 数据库处理器不可用，无法更新计数。")
            return

        analysis_result = await self._call_llm_for_sobriquet_analysis(chat_history_str, bot_reply) # 方法名更新

        if analysis_result.get("is_exist") and analysis_result.get("data"):
            sobriquet_map_to_update = analysis_result["data"] # 变量名更新
            logger.info(f"{log_prefix} LLM 找到绰号映射，准备更新数据库: {sobriquet_map_to_update}")

            for user_id_str, sobriquet_value in sobriquet_map_to_update.items(): # 变量名更新
                if not user_id_str or not sobriquet_value: # 变量名更新
                    logger.warning(f"{log_prefix} 跳过无效条目: user_id='{user_id_str}', sobriquet='{sobriquet_value}'") # 日志更新
                    continue
                if not user_id_str.isdigit():
                    logger.warning(f"{log_prefix} 无效的用户ID格式 (非纯数字): '{user_id_str}'，跳过。")
                    continue
                user_id_int = int(user_id_str)

                try:
                    # person_id_manager.get_person_id 的第二个参数可以是 str 或 int，这里保持 int
                    person_id = person_info_manager.get_person_id(platform, user_id_int) # 修正：之前误写为 user_id_str
                    if not person_id:
                        logger.error(
                            f"{log_prefix} 无法为 platform='{platform}', user_id='{user_id_str}' 生成 person_id，跳过此用户。"
                        )
                        continue
                    self.db_handler.upsert_person(person_id, user_id_int, platform)
                    # 调用 SobriquetDB 中的更新方法
                    self.db_handler.update_group_sobriquet_count(person_id, group_id, sobriquet_value) # 方法及参数名更新
                except (OperationFailure, DuplicateKeyError) as db_err:
                    logger.exception(
                        f"{log_prefix} 数据库操作失败 ({type(db_err).__name__}): 用户 {user_id_str}, 绰号 {sobriquet_value}. 错误: {db_err}"
                    )
                except Exception as e:
                    logger.exception(f"{log_prefix} 处理用户 {user_id_str} 的绰号 '{sobriquet_value}' 时发生意外错误：{e}")
        else:
            logger.debug(f"{log_prefix} LLM 未找到可靠的绰号映射或分析失败。")

    async def _call_llm_for_sobriquet_analysis( # 方法名更新
        self,
        chat_history_str: str,
        bot_reply: str,
    ) -> Dict[str, Any]:
        if not self.llm_mapper:
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {"is_exist": False}

        prompt = build_sobriquet_mapping_prompt(chat_history_str, bot_reply) # 使用更新后的函数
        logger.debug(f"构建的绰号映射 Prompt (部分):\n{prompt[:300]}...") 

        try:
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")

            if not response_content:
                logger.warning("LLM 返回了空的绰号映射内容。")
                return {"is_exist": False}

            response_content = response_content.strip()
            markdown_code_regex = re.compile(r"```(?:json)?\s*\n(\{.*?\})\n\s*```", re.DOTALL | re.IGNORECASE)
            match = markdown_code_regex.search(response_content)
            if match:
                response_content = match.group(1).strip()
            elif response_content.startswith("{") and response_content.endswith("}"):
                pass
            else:
                json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(0)
                else:
                    logger.warning(f"LLM 响应似乎不包含有效的 JSON 对象。响应: {response_content}")
                    return {"is_exist": False}

            result = json.loads(response_content)

            if not isinstance(result, dict):
                logger.warning(f"LLM 响应不是一个有效的 JSON 对象 (字典类型)。响应内容: {response_content}")
                return {"is_exist": False}

            is_exist = result.get("is_exist")

            if is_exist is True:
                original_data = result.get("data")
                if isinstance(original_data, dict) and original_data:
                    logger.info(f"LLM 找到的原始绰号映射: {original_data}")
                    filtered_data = self._filter_llm_sobriquet_results(original_data) # 方法名更新
                    if not filtered_data:
                        logger.info("所有找到的绰号映射都被过滤掉了。")
                        return {"is_exist": False}
                    else:
                        logger.info(f"过滤后的绰号映射: {filtered_data}")
                        return {"is_exist": True, "data": filtered_data}
                else:
                    logger.warning(f"LLM 响应格式错误: is_exist=True 但 data 无效。原始 data: {original_data}")
                    return {"is_exist": False}
            elif is_exist is False:
                logger.info("LLM 明确指示未找到可靠的绰号映射 (is_exist=False)。")
                return {"is_exist": False}
            else:
                logger.warning(f"LLM 响应格式错误: 'is_exist' 的值 '{is_exist}' 无效。")
                return {"is_exist": False}

        except json.JSONDecodeError as json_err:
            logger.error(f"解析 LLM 响应 JSON 失败: {json_err}\n原始响应: {response_content}")
            return {"is_exist": False}
        except Exception as e:
            logger.error(f"绰号映射 LLM 调用或处理过程中发生意外错误: {e}", exc_info=True)
            return {"is_exist": False}

    def _filter_llm_sobriquet_results(self, original_data: Dict[str, str]) -> Dict[str, str]: # 方法名更新
        """过滤 LLM 返回的绰号映射结果。"""
        filtered_data = {}
        # bot.qq_account 配置键名不变
        bot_qq_str = global_config.bot.qq_account if global_config.bot.qq_account else None

        for user_id, sobriquet_val in original_data.items(): # 变量名更新
            if not isinstance(user_id, str):
                logger.warning(f"过滤掉非字符串 user_id: {user_id}")
                continue
            if bot_qq_str and user_id == bot_qq_str:
                logger.debug(f"过滤掉机器人自身的映射: ID {user_id}")
                continue
            if not sobriquet_val or not sobriquet_val.strip(): # 变量名更新
                logger.debug(f"过滤掉用户 {user_id} 的空绰号。")
                continue

            filtered_data[user_id] = sobriquet_val.strip() # 变量名更新

        return filtered_data

    def _run_processor_in_thread(self):
        """后台线程入口函数，管理绰号分析的 asyncio 事件循环。""" # 注释更新
        thread_id = threading.get_ident()
        logger.info(f"绰号处理器线程启动 (线程 ID: {thread_id})...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"(线程 ID: {thread_id}) 绰号处理 Asyncio 事件循环已创建并设置。")

        run_async_loop(loop, self._processing_loop()) # 使用辅助函数

        logger.info(f"绰号处理器线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
        """后台线程中运行的异步处理循环 (使用 asyncio.Queue)。"""
        logger.info("绰号异步处理循环已启动。")

        while not self._stop_event.is_set():
            try:
                # 等待队列项，超时时间为 sleep_interval
                item = await asyncio.wait_for(self.sobriquet_queue.get(), timeout=self.sleep_interval) # 队列名更新
                await self._analyze_and_update_sobriquets(item) # 方法名更新
                self.sobriquet_queue.task_done() # 队列名更新
            except asyncio.TimeoutError:
                # 超时是正常的，用于检查 _stop_event
                continue
            except asyncio.CancelledError:
                logger.info("绰号处理循环被取消。")
                break
            except Exception as e:
                logger.error(f"绰号处理循环出错: {e}", exc_info=True)
                await asyncio.sleep(5) 

        logger.info("绰号异步处理循环已结束。")


# 单例实例名更新
sobriquet_manager = SobriquetManager()