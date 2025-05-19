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
from .nickname_db import NicknameDB
from .nickname_mapper import build_mapping_prompt
from .nickname_utils import select_nicknames_for_prompt, format_user_info_prompt
from src.chat.person_info.person_info import person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("NicknameManager")
logger_helper = get_logger("AsyncLoopHelper")  # 为辅助函数创建单独的 logger


def run_async_loop(loop: asyncio.AbstractEventLoop, coro):
    """
    运行给定的协程直到完成，并确保循环最终关闭。

    Args:
        loop: 要使用的 asyncio 事件循环。
        coro: 要在循环中运行的主协程。
    """
    try:
        logger_helper.debug(f"Running coroutine in loop {id(loop)}...")
        result = loop.run_until_complete(coro)
        logger_helper.debug(f"Coroutine completed in loop {id(loop)}.")
        return result
    except asyncio.CancelledError:
        logger_helper.info(f"Coroutine in loop {id(loop)} was cancelled.")
        # 取消是预期行为，不视为错误
    except Exception as e:
        logger_helper.error(f"Error in async loop {id(loop)}: {e}", exc_info=True)
    finally:
        try:
            # 1. 取消所有剩余任务
            all_tasks = asyncio.all_tasks(loop)
            current_task = asyncio.current_task(loop)
            tasks_to_cancel = [
                task for task in all_tasks if task is not current_task
            ]  # 避免取消 run_until_complete 本身
            if tasks_to_cancel:
                logger_helper.info(f"Cancelling {len(tasks_to_cancel)} outstanding tasks in loop {id(loop)}...")
                for task in tasks_to_cancel:
                    task.cancel()
                # 等待取消完成
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                logger_helper.info(f"Outstanding tasks cancelled in loop {id(loop)}.")

            # 2. 停止循环 (如果仍在运行)
            if loop.is_running():
                loop.stop()
                logger_helper.info(f"Asyncio loop {id(loop)} stopped.")

            # 3. 关闭循环 (如果未关闭)
            if not loop.is_closed():
                # 在关闭前再运行一次以处理挂起的关闭回调
                loop.run_until_complete(loop.shutdown_asyncgens())  # 关闭异步生成器
                loop.close()
                logger_helper.info(f"Asyncio loop {id(loop)} closed.")
        except Exception as close_err:
            logger_helper.error(f"Error during asyncio loop cleanup for loop {id(loop)}: {close_err}", exc_info=True)


class NicknameManager:
    """
    管理群组绰号分析、处理、存储和使用的单例类。
    封装了 LLM 调用、后台处理线程和数据库交互。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logger.info("正在创建 NicknameManager 单例实例...")
                    cls._instance = super(NicknameManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化 NicknameManager。
        使用锁和标志确保实际初始化只执行一次。
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            logger.info("正在初始化 NicknameManager 组件...")
            self.is_enabled = global_config.group_nickname.enable_nickname_mapping

            # 数据库处理器
            person_info_collection = getattr(db, "person_info", None)
            self.db_handler = NicknameDB(person_info_collection)
            if not self.db_handler.is_available():
                logger.error("数据库处理器初始化失败，NicknameManager 功能受限。")

            # LLM 映射器 (用于绰号分析，不直接用于此处的 prompt 注入逻辑修改)
            self.llm_mapper: Optional[LLMRequest] = None
            if self.is_enabled: # LLM 映射器仅在功能启用时初始化
                try:
                    model_config = global_config.model.nickname_mapping
                    if model_config and model_config.get("name"):
                        self.llm_mapper = LLMRequest(
                            model=model_config,
                            temperature=model_config.get("temp", 0.5),
                            max_tokens=model_config.get("max_tokens", 256),
                            request_type="nickname_mapping",
                        )
                        logger.info("绰号映射 LLM 映射器初始化成功。")
                    else:
                        logger.warning("绰号映射 LLM 配置无效或缺失 'name'，绰号分析功能禁用。")
                        # self.is_enabled = False # 此处的 is_enabled 控制的是绰号分析，而非基础的UID-Name注入
                except KeyError as ke:
                    logger.error(f"初始化绰号映射 LLM 时缺少配置项: {ke}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
                    # self.is_enabled = False
                except Exception as e:
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，绰号分析功能禁用。", exc_info=True)
                    self.llm_mapper = None
                    # self.is_enabled = False

            # 队列和线程 (用于绰号分析)
            self.queue_max_size = global_config.group_nickname.nickname_queue_max_size
            self.nickname_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size)
            self._stop_event = threading.Event()
            self._nickname_thread: Optional[threading.Thread] = None
            self.sleep_interval = global_config.group_nickname.nickname_process_sleep_interval

            self._initialized = True
            logger.info("NicknameManager 初始化完成。")

    def start_processor(self):
        """启动后台处理线程（如果已启用且未运行）。"""
        if not self.is_enabled: # 这里的 is_enabled 指的是整个绰号特性，包括分析
            logger.info("绰号处理功能已禁用，处理器未启动。")
            return
        if global_config.group_nickname.max_nicknames_in_prompt == 0:
            logger.error("[错误] 绰号注入数量不合适，绰号处理功能已禁用！")
            # 如果 max_nicknames_in_prompt 为0，可能也意味着不应进行绰号分析
            return

        if self._nickname_thread is None or not self._nickname_thread.is_alive():
            logger.info("正在启动绰号处理器线程...")
            self._stop_event.clear()
            self._nickname_thread = threading.Thread(
                target=self._run_processor_in_thread,
                daemon=True,
            )
            self._nickname_thread.start()
            logger.info(f"绰号处理器线程已启动 (ID: {self._nickname_thread.ident})")
        else:
            logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        """停止后台处理线程。"""
        if self._nickname_thread and self._nickname_thread.is_alive():
            logger.info("正在停止绰号处理器线程...")
            self._stop_event.set()
            try:
                self._nickname_thread.join(timeout=10)
                if self._nickname_thread.is_alive():
                    logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e:
                logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._nickname_thread and not self._nickname_thread.is_alive():
                    logger.info("绰号处理器线程已成功停止。")
                self._nickname_thread = None
        else:
            logger.info("绰号处理器线程未在运行或已被清理。")

    async def trigger_nickname_analysis(
        self,
        anchor_message: MessageRecv,
        bot_reply: List[str],
        chat_stream: Optional[ChatStream] = None,
    ):
        """
        准备数据并将其排队等待绰号分析（如果满足条件）。
        """
        if not self.is_enabled: # 依赖总开关
            return

        if random.random() > global_config.group_nickname.nickname_analysis_probability: # 注意这里是大于号，原为小于号，按通常理解应为小于号触发
            logger.debug("跳过绰号分析：随机概率未命中。")
            return

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info:
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
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

            user_ids_in_history = {
                str(msg["user_info"]["user_id"]) for msg in history_messages if msg.get("user_info", {}).get("user_id")
            }
            user_name_map = {}
            if user_ids_in_history:
                try:
                    names_data = await relationship_manager.get_person_names_batch(platform, list(user_ids_in_history))
                except Exception as e:
                    logger.error(f"{log_prefix} 批量获取 person_name 时出错: {e}", exc_info=True)
                    names_data = {}
                for user_id_str_hist in user_ids_in_history: # 变量名区分
                    if user_id_str_hist in names_data:
                        user_name_map[user_id_str_hist] = names_data[user_id_str_hist]
                    else:
                        latest_nickname = next(
                            (
                                m["user_info"].get("user_nickname")
                                for m in reversed(history_messages)
                                if str(m["user_info"].get("user_id")) == user_id_str_hist and m["user_info"].get("user_nickname")
                            ),
                            None,
                        )
                        user_name_map[user_id_str_hist] = (
                            latest_nickname or f"{global_config.bot.nickname}"
                            if user_id_str_hist == global_config.bot.qq_account
                            else "未知"
                        )

            item = (chat_history_str, bot_reply_str, platform, group_id, user_name_map)
            await self._add_to_queue(item, platform, group_id)

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_nickname_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        """
        获取并格式化用于 Prompt 注入的用户信息字符串。
        始终包含 UID 和 Person Name。
        如果功能启用且在群聊中，则额外包含绰号信息。
        """
        if not chat_stream: # chat_stream 是必需的，用于获取平台信息等
            logger.warning("无效的 chat_stream，无法获取用户信息注入。")
            return ""

        log_prefix = f"[{chat_stream.stream_id}]"
        try:
            platform = chat_stream.platform # 获取平台信息

            # 1. 收集上下文中的用户ID
            user_ids_in_context = {
                str(msg["user_info"]["user_id"])
                for msg in message_list_before_now
                if msg.get("user_info", {}).get("user_id")
            }

            if not user_ids_in_context: # 如果消息列表为空或没有用户信息
                # 尝试从 chat_stream 获取最近发言者作为备选
                recent_speakers = chat_stream.get_recent_speakers(limit=5) # 可配置回顾数量
                user_ids_in_context.update(str(speaker["user_id"]) for speaker in recent_speakers if speaker.get("user_id"))
            
            if not user_ids_in_context:
                logger.debug(f"{log_prefix} 未找到上下文用户用于信息注入。")
                return ""

            # 2. 获取这些用户的 Person Name
            # 使用集合确保 user_id 唯一，转换为列表进行查询
            unique_user_ids_list = sorted(list(user_ids_in_context)) # 排序以保证输出顺序一致性
            person_names_data: Dict[str, str] = {}
            if unique_user_ids_list:
                try:
                    person_names_data = await relationship_manager.get_person_names_batch(platform, unique_user_ids_list)
                except Exception as e:
                    logger.error(f"{log_prefix} 批量获取 person_name 时出错: {e}", exc_info=True)
                    # 出错则 names_data 为空，后续逻辑会处理
            
            users_for_prompt: List[Tuple[str, str]] = [] # 存储 (user_id, person_name)
            for user_id_str in unique_user_ids_list:
                person_name = person_names_data.get(user_id_str)
                if person_name: # 只添加成功获取到 person_name 的用户
                    users_for_prompt.append((user_id_str, person_name))
                elif user_id_str == global_config.bot.qq_account:
                    users_for_prompt.append((user_id_str, global_config.bot.nickname))
                else:
                    logger.warning(f"{log_prefix} 未能获取到 user_id '{user_id_str}' 的 person_name。")
            
            if not users_for_prompt: # 如果最终没有可供显示的用户信息
                logger.debug(f"{log_prefix} 未能构建有效的用户列表用于注入。")
                return ""

            # 3. 条件性地获取绰号信息
            selected_nicknames_for_formatting: Optional[List[Tuple[str, str, str, int]]] = None
            # 检查总开关是否打开，并且当前是群聊
            if self.is_enabled and chat_stream.group_info and chat_stream.group_info.group_id:
                group_id = str(chat_stream.group_info.group_id)
                # 获取绰号数据，其键为 person_name
                all_nicknames_data_by_person_name: Dict[str, Dict[str, Any]] = {}
                if unique_user_ids_list: # 确保有用户ID列表才去查询绰号
                    try:
                        # get_users_group_nicknames 需要 user_id 列表
                        all_nicknames_data_by_person_name = await relationship_manager.get_users_group_nicknames(
                            platform, unique_user_ids_list, group_id
                        )
                    except Exception as e:
                        logger.error(f"{log_prefix} 获取群组 '{group_id}' 的绰号时出错: {e}", exc_info=True)
                
                if all_nicknames_data_by_person_name:
                    # select_nicknames_for_prompt 期望的输入是以 person_name 为键的字典
                    # 其返回值为 List[Tuple[str, str, str, int]] (person_name, user_id, nickname, count)
                    selected_nicknames_for_formatting = select_nicknames_for_prompt(all_nicknames_data_by_person_name)
                    if selected_nicknames_for_formatting:
                         logger.debug(f"{log_prefix} 为群组 '{group_id}' 选择了以下绰号进行注入: {selected_nicknames_for_formatting}")


            # 4. 调用格式化函数
            # format_user_info_prompt 现在处理两种情况
            injection_str = format_user_info_prompt(users_for_prompt, selected_nicknames_for_formatting)
            
            if injection_str:
                logger.debug(f"{log_prefix} 生成的用户信息 Prompt 注入:\n{injection_str.strip()}") # strip() 移除末尾换行符以便日志查看
            return injection_str

        except Exception as e:
            logger.error(f"{log_prefix} 获取用户信息注入时发生严重错误: {e}", exc_info=True)
            return ""


    # 私有/内部方法

    async def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        """将项目异步添加到内部处理队列 (asyncio.Queue)。"""
        try:
            await self.nickname_queue.put(item)
            logger.debug(
                f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号队列。当前大小: {self.nickname_queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"绰号队列已满 (最大={self.queue_max_size})。平台 '{platform}' 群组 '{group_id}' 的项目被丢弃。"
            )
        except Exception as e:
            logger.error(f"将项目添加到绰号队列时出错: {e}", exc_info=True)

    async def _analyze_and_update_nicknames(self, item: tuple):
        """处理单个队列项目：调用 LLM 分析并更新数据库。"""
        if not isinstance(item, tuple) or len(item) != 5:
            logger.warning(f"从队列接收到无效项目: {type(item)}")
            return

        chat_history_str, bot_reply, platform, group_id, user_name_map = item
        log_prefix = f"[{platform}:{group_id}]"
        logger.debug(f"{log_prefix} 开始处理绰号分析任务...")

        if not self.llm_mapper:
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行分析。")
            return
        if not self.db_handler.is_available():
            logger.error(f"{log_prefix} 数据库处理器不可用，无法更新计数。")
            return

        analysis_result = await self._call_llm_for_analysis(chat_history_str, bot_reply, user_name_map)

        if analysis_result.get("is_exist") and analysis_result.get("data"):
            nickname_map_to_update = analysis_result["data"]
            logger.info(f"{log_prefix} LLM 找到绰号映射，准备更新数据库: {nickname_map_to_update}")

            for user_id_str, nickname in nickname_map_to_update.items():
                if not user_id_str or not nickname:
                    logger.warning(f"{log_prefix} 跳过无效条目: user_id='{user_id_str}', nickname='{nickname}'")
                    continue
                if not user_id_str.isdigit():
                    logger.warning(f"{log_prefix} 无效的用户ID格式 (非纯数字): '{user_id_str}'，跳过。")
                    continue
                user_id_int = int(user_id_str)

                try:
                    person_id = person_info_manager.get_person_id(platform, user_id_str)
                    if not person_id:
                        logger.error(
                            f"{log_prefix} 无法为 platform='{platform}', user_id='{user_id_str}' 生成 person_id，跳过此用户。"
                        )
                        continue
                    # 确保 person 文档存在
                    self.db_handler.upsert_person(person_id, user_id_int, platform)
                    # 更新绰号计数
                    self.db_handler.update_group_nickname_count(person_id, group_id, nickname)
                except (OperationFailure, DuplicateKeyError) as db_err:
                    logger.exception(
                        f"{log_prefix} 数据库操作失败 ({type(db_err).__name__}): 用户 {user_id_str}, 绰号 {nickname}. 错误: {db_err}"
                    )
                except Exception as e:
                    logger.exception(f"{log_prefix} 处理用户 {user_id_str} 的绰号 '{nickname}' 时发生意外错误：{e}")
        else:
            logger.debug(f"{log_prefix} LLM 未找到可靠的绰号映射或分析失败。")

    async def _call_llm_for_analysis(
        self,
        chat_history_str: str,
        bot_reply: str,
        user_name_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        内部方法：调用 LLM 分析聊天记录和 Bot 回复，提取可靠的 用户ID-绰号 映射。
        """
        if not self.llm_mapper:
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {"is_exist": False}

        prompt = build_mapping_prompt(chat_history_str, bot_reply, user_name_map)
        logger.debug(f"构建的绰号映射 Prompt (部分):\n{prompt}...") # 调整日志输出长度

        try:
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")

            if not response_content:
                logger.warning("LLM 返回了空的绰号映射内容。")
                return {"is_exist": False}

            response_content = response_content.strip()
            # 增强对 JSON 代码块的提取
            markdown_code_regex = re.compile(r"```(?:json)?\s*\n(\{.*?\})\n\s*```", re.DOTALL | re.IGNORECASE)
            match = markdown_code_regex.search(response_content) # 使用 search 而非 match
            if match:
                response_content = match.group(1).strip()
            elif response_content.startswith("{") and response_content.endswith("}"):
                pass  # 可能是纯 JSON
            else:
                # 尝试提取被文本包围的 JSON 对象
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

            is_exist = result.get("is_exist") # 使用 .get() 避免 KeyError

            if is_exist is True: # 显式检查 True
                original_data = result.get("data")
                if isinstance(original_data, dict) and original_data: # 确保 data 是非空字典
                    logger.info(f"LLM 找到的原始绰号映射: {original_data}")
                    filtered_data = self._filter_llm_results(original_data, user_name_map)
                    if not filtered_data:
                        logger.info("所有找到的绰号映射都被过滤掉了。")
                        return {"is_exist": False} # 如果过滤后为空，也视为未找到
                    else:
                        logger.info(f"过滤后的绰号映射: {filtered_data}")
                        return {"is_exist": True, "data": filtered_data}
                else: # is_exist 为 True 但 data 无效
                    logger.warning(f"LLM 响应格式错误: is_exist=True 但 data 无效。原始 data: {original_data}")
                    return {"is_exist": False}
            elif is_exist is False: # 显式检查 False
                logger.info("LLM 明确指示未找到可靠的绰号映射 (is_exist=False)。")
                return {"is_exist": False}
            else: # is_exist 的值不是 True 或 False
                logger.warning(f"LLM 响应格式错误: 'is_exist' 的值 '{is_exist}' 无效。")
                return {"is_exist": False}

        except json.JSONDecodeError as json_err:
            logger.error(f"解析 LLM 响应 JSON 失败: {json_err}\n原始响应: {response_content}")
            return {"is_exist": False}
        except Exception as e:
            logger.error(f"绰号映射 LLM 调用或处理过程中发生意外错误: {e}", exc_info=True)
            return {"is_exist": False}

    def _filter_llm_results(self, original_data: Dict[str, str], user_name_map: Dict[str, str]) -> Dict[str, str]:
        """过滤 LLM 返回的绰号映射结果。"""
        filtered_data = {}
        bot_qq_str = global_config.bot.qq_account if global_config.bot.qq_account else None

        for user_id, nickname in original_data.items():
            if not isinstance(user_id, str): # 确保 user_id 是字符串
                logger.warning(f"过滤掉非字符串 user_id: {user_id}")
                continue
            if bot_qq_str and user_id == bot_qq_str: # 过滤掉机器人自己
                logger.debug(f"过滤掉机器人自身的映射: ID {user_id}")
                continue
            if not nickname or not nickname.strip(): # 过滤掉空或仅含空白的绰号
                logger.debug(f"过滤掉用户 {user_id} 的空绰号。")
                continue

            # 移除了与 person_name 比较的过滤，因为 LLM prompt 已指示不要输出与已知名称相同的词
            # person_name = user_name_map.get(user_id)
            # if person_name and person_name == nickname.strip():
            #     logger.debug(f"过滤掉用户 {user_id} 的映射: 绰号 '{nickname}' 与其名称 '{person_name}' 相同。")
            #     continue
            filtered_data[user_id] = nickname.strip() # 存储处理过的绰号

        return filtered_data

    def _run_processor_in_thread(self):
        """后台线程入口函数，使用辅助函数管理 asyncio 事件循环。"""
        thread_id = threading.get_ident()
        logger.info(f"绰号处理器线程启动 (线程 ID: {thread_id})...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"(线程 ID: {thread_id}) Asyncio 事件循环已创建并设置。")

        run_async_loop(loop, self._processing_loop())

        logger.info(f"绰号处理器线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
        """后台线程中运行的异步处理循环 (使用 asyncio.Queue)。"""
        logger.info("绰号异步处理循环已启动。")

        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self.nickname_queue.get(), timeout=self.sleep_interval)
                await self._analyze_and_update_nicknames(item)
                self.nickname_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("绰号处理循环被取消。")
                break
            except Exception as e:
                logger.error(f"绰号处理循环出错: {e}", exc_info=True)
                await asyncio.sleep(5) # 短暂休眠避免快速连续失败

        logger.info("绰号异步处理循环已结束。")


# 在模块级别创建单例实例
nickname_manager = NicknameManager()
