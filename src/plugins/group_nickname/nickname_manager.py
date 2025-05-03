import asyncio
import threading
import queue
import traceback
import time
import json
import re
from typing import Dict, Optional, List, Any

from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.plugins.models.utils_model import LLMRequest
from .nickname_db import NicknameDB
from .nickname_mapper import _build_mapping_prompt
from .nickname_utils import select_nicknames_for_prompt, format_nickname_prompt_injection

# 依赖于 person_info_manager 来生成 person_id
from ..person_info.person_info import person_info_manager

# 依赖于 relationship_manager 来获取用户名称和现有绰号
from ..person_info.relationship_manager import relationship_manager

# 导入消息和聊天流相关的类型和工具
from src.plugins.chat.chat_stream import ChatStream
from src.plugins.chat.message import MessageRecv
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

logger = get_logger("NicknameManager")


class NicknameManager:
    """
    管理群组绰号分析、处理、存储和使用的单例类。
    封装了 LLM 调用、后台处理线程和数据库交互。
    """

    _instance = None
    _lock = threading.Lock()

    # Singleton Implementation
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                # 再次检查，防止多线程并发创建实例
                if not cls._instance:
                    logger.info("正在创建 NicknameManager 单例实例...")
                    cls._instance = super(NicknameManager, cls).__new__(cls)
                    cls._instance._initialized = False  # 添加初始化标志
        return cls._instance

    def __init__(self):
        """
        初始化 NicknameManager。
        使用锁和标志确保实际初始化只执行一次。
        """
        if self._initialized:  # 如果已初始化，直接返回
            return

        with self._lock:
            # 再次检查初始化标志，防止重复初始化
            if self._initialized:
                return

            logger.info("正在初始化 NicknameManager 组件...")
            self.config = global_config
            self.is_enabled = self.config.ENABLE_NICKNAME_MAPPING

            # 数据库处理器
            person_info_collection = getattr(db, "person_info", None)
            self.db_handler = NicknameDB(person_info_collection)
            if not self.db_handler.is_available():
                logger.error("数据库处理器初始化失败，NicknameManager 功能受限。")
                self.is_enabled = False  # 如果数据库不可用，禁用功能

            # LLM 映射器
            self.llm_mapper: Optional[LLMRequest] = None
            if self.is_enabled:
                try:
                    model_config = self.config.llm_nickname_mapping
                    if model_config and model_config.get("name"):
                        self.llm_mapper = LLMRequest(
                            model=model_config,
                            temperature=model_config.get("temp", 0.5),  # 使用 get 获取并提供默认值
                            max_tokens=model_config.get("max_tokens", 256),  # 使用 get 获取并提供默认值
                            request_type="nickname_mapping",
                        )
                        logger.info("绰号映射 LLM 映射器初始化成功。")
                    else:
                        logger.warning("绰号映射 LLM 配置无效或缺失 'name'，功能禁用。")
                        self.is_enabled = False
                except KeyError as ke:
                    logger.error(f"初始化绰号映射 LLM 时缺少配置项: {ke}，功能禁用。", exc_info=True)
                    self.llm_mapper = None
                    self.is_enabled = False
                except Exception as e:
                    logger.error(f"初始化绰号映射 LLM 映射器失败: {e}，功能禁用。", exc_info=True)
                    self.llm_mapper = None
                    self.is_enabled = False

            # 队列和线程
            self.queue_max_size = getattr(self.config, "NICKNAME_QUEUE_MAX_SIZE", 100)
            self.nickname_queue: queue.Queue = queue.Queue(maxsize=self.queue_max_size)
            self._stop_event = threading.Event()
            self._nickname_thread: Optional[threading.Thread] = None
            self.sleep_interval = getattr(self.config, "NICKNAME_PROCESS_SLEEP_INTERVAL", 0.5)

            self._initialized = True  # 标记为已初始化
            logger.info("NicknameManager 初始化完成。")

    # 公共方法

    def start_processor(self):
        """启动后台处理线程（如果已启用且未运行）。"""
        if not self.is_enabled:
            logger.info("绰号处理功能已禁用，处理器未启动。")
            return
        if self._nickname_thread is None or not self._nickname_thread.is_alive():
            logger.info("正在启动绰号处理器线程...")
            self._stop_event.clear()  # 清除停止事件标志
            self._nickname_thread = threading.Thread(
                target=self._run_processor_in_thread,  # 线程执行的入口函数
                daemon=True,  # 设置为守护线程，主程序退出时自动结束
            )
            self._nickname_thread.start()
            logger.info(f"绰号处理器线程已启动 (ID: {self._nickname_thread.ident})")
        else:
            logger.warning("绰号处理器线程已在运行中。")

    def stop_processor(self):
        """停止后台处理线程。"""
        if self._nickname_thread and self._nickname_thread.is_alive():
            logger.info("正在停止绰号处理器线程...")
            self._stop_event.set()  # 设置停止事件标志
            try:
                # 可选：尝试清空队列，避免丢失未处理的任务
                # while not self.nickname_queue.empty():
                #    try:
                #        self.nickname_queue.get_nowait()
                #        self.nickname_queue.task_done()
                #    except queue.Empty:
                #        break
                # logger.info("绰号处理队列已清空。")

                self._nickname_thread.join(timeout=10)  # 等待线程结束，设置超时
                if self._nickname_thread.is_alive():
                    logger.warning("绰号处理器线程在超时后仍未停止。")
            except Exception as e:
                logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
            finally:
                if self._nickname_thread and not self._nickname_thread.is_alive():
                    logger.info("绰号处理器线程已成功停止。")
                self._nickname_thread = None  # 清理线程对象引用
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
        取代了旧的 trigger_nickname_analysis_if_needed 函数。
        """
        if not self.is_enabled:
            return  # 功能禁用则直接返回

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info:
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
            # 1. 获取历史记录
            history_limit = getattr(self.config, "NICKNAME_ANALYSIS_HISTORY_LIMIT", 30)
            history_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=current_chat_stream.stream_id,
                timestamp=time.time(),
                limit=history_limit,
            )

            # 格式化历史记录
            chat_history_str = await build_readable_messages(
                messages=history_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
                truncate=False,
            )

            # 2. 获取 Bot 回复
            bot_reply_str = " ".join(bot_reply) if bot_reply else ""

            # 3. 获取群组和平台信息
            group_id = str(current_chat_stream.group_info.group_id)
            platform = current_chat_stream.platform

            # 4. 构建用户 ID 到名称的映射 (user_name_map)
            user_ids_in_history = {
                str(msg["user_info"]["user_id"]) for msg in history_messages if msg.get("user_info", {}).get("user_id")
            }
            user_name_map = {}
            if user_ids_in_history:
                try:
                    # 使用 relationship_manager 批量获取名称
                    names_data = await relationship_manager.get_person_names_batch(platform, list(user_ids_in_history))
                except Exception as e:
                    logger.error(f"{log_prefix} 批量获取 person_name 时出错: {e}", exc_info=True)
                    names_data = {}

                # 填充 user_name_map
                for user_id in user_ids_in_history:
                    if user_id in names_data:
                        user_name_map[user_id] = names_data[user_id]
                    else:
                        # 回退查找历史记录中的 nickname
                        latest_nickname = next(
                            (
                                m["user_info"].get("user_nickname")
                                for m in reversed(history_messages)
                                if str(m["user_info"].get("user_id")) == user_id and m["user_info"].get("user_nickname")
                            ),
                            None,
                        )
                        user_name_map[user_id] = latest_nickname or f"未知({user_id})"

            # 5. 添加到内部处理队列
            item = (chat_history_str, bot_reply_str, platform, group_id, user_name_map)
            self._add_to_queue(item, platform, group_id)  # 调用私有方法入队

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_nickname_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        """
        获取并格式化用于 Prompt 注入的绰号信息字符串。
        取代了旧的 get_nickname_injection_for_prompt 函数。
        """
        if not self.is_enabled or not chat_stream or not chat_stream.group_info:
            return ""  # 功能禁用或非群聊则返回空

        log_prefix = f"[{chat_stream.stream_id}]"
        try:
            group_id = str(chat_stream.group_info.group_id)
            platform = chat_stream.platform

            # 确定上下文中的用户 ID
            user_ids_in_context = {
                str(msg["user_info"]["user_id"])
                for msg in message_list_before_now
                if msg.get("user_info", {}).get("user_id")
            }

            # 如果消息列表为空，尝试获取最近发言者
            if not user_ids_in_context:
                recent_speakers = chat_stream.get_recent_speakers(limit=5)
                user_ids_in_context.update(str(speaker["user_id"]) for speaker in recent_speakers)

            if not user_ids_in_context:
                logger.warning(f"{log_prefix} 未找到上下文用户用于绰号注入。")
                return ""

            # 使用 relationship_manager 批量获取这些用户的群组绰号
            all_nicknames_data = await relationship_manager.get_users_group_nicknames(
                platform, list(user_ids_in_context), group_id
            )

            if all_nicknames_data:
                # 使用 nickname_utils 中的工具函数进行选择和格式化
                selected_nicknames = select_nicknames_for_prompt(all_nicknames_data)
                injection_str = format_nickname_prompt_injection(selected_nicknames)
                if injection_str:
                    logger.debug(f"{log_prefix} 生成的绰号 Prompt 注入:\n{injection_str}")
                return injection_str
            else:
                return ""  # 没有获取到绰号数据

        except Exception as e:
            logger.error(f"{log_prefix} 获取绰号注入时出错: {e}", exc_info=True)
            return ""  # 出错时返回空

    # 私有/内部方法

    def _add_to_queue(self, item: tuple, platform: str, group_id: str):
        """将项目添加到内部处理队列。"""
        try:
            self.nickname_queue.put_nowait(item)
            logger.debug(
                f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号队列。当前大小: {self.nickname_queue.qsize()}"
            )
        except queue.Full:
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
        thread_id = threading.get_ident()
        log_prefix = f"[线程 {thread_id}][{platform}:{group_id}]"
        logger.debug(f"{log_prefix} 开始处理绰号分析任务...")

        if not self.llm_mapper:
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行分析。")
            return
        if not self.db_handler.is_available():
            logger.error(f"{log_prefix} 数据库处理器不可用，无法更新计数。")
            return

        # 1. 调用 LLM 分析 (逻辑从 nickname_mapper 移入)
        analysis_result = await self._call_llm_for_analysis(chat_history_str, bot_reply, user_name_map)

        # 2. 如果分析成功且找到映射，则更新数据库
        if analysis_result.get("is_exist") and analysis_result.get("data"):
            nickname_map_to_update = analysis_result["data"]
            logger.info(f"{log_prefix} LLM 找到绰号映射，准备更新数据库: {nickname_map_to_update}")

            for user_id_str, nickname in nickname_map_to_update.items():
                # 基本验证
                if not user_id_str or not nickname:
                    logger.warning(f"{log_prefix} 跳过无效条目: user_id='{user_id_str}', nickname='{nickname}'")
                    continue
                if not user_id_str.isdigit():
                    logger.warning(f"{log_prefix} 无效的用户ID格式 (非纯数字): '{user_id_str}'，跳过。")
                    continue
                user_id_int = int(user_id_str)
                # 结束验证

                try:
                    # 步骤 1: 生成 person_id
                    person_id = person_info_manager.get_person_id(platform, user_id_str)
                    if not person_id:
                        logger.error(
                            f"{log_prefix} 无法为 platform='{platform}', user_id='{user_id_str}' 生成 person_id，跳过此用户。"
                        )
                        continue

                    # 步骤 2: 确保 Person 文档存在 (调用 DB Handler)
                    self.db_handler.upsert_person(person_id, user_id_int, platform)

                    # 步骤 3: 更新群组绰号 (调用 DB Handler)
                    self.db_handler.update_group_nickname_count(person_id, group_id, nickname)

                except (OperationFailure, DuplicateKeyError) as db_err:  # 捕获特定的数据库错误
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
        (逻辑从 analyze_chat_for_nicknames 移入)
        """
        if not self.llm_mapper:  # 再次检查 LLM 映射器
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {"is_exist": False}

        prompt = _build_mapping_prompt(chat_history_str, bot_reply, user_name_map)
        logger.debug(f"构建的绰号映射 Prompt:\n{prompt[:500]}...")  # 截断日志输出

        try:
            # 调用 LLM
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")

            if not response_content:
                logger.warning("LLM 返回了空的绰号映射内容。")
                return {"is_exist": False}

            # 清理可能的 Markdown 代码块标记
            response_content = response_content.strip()
            markdown_code_regex = re.compile(r"^```(?:\w+)?\s*\n(.*?)\n\s*```$", re.DOTALL | re.IGNORECASE)
            match = markdown_code_regex.match(response_content)
            if match:
                response_content = match.group(1).strip()
            # 尝试直接解析 JSON，即使没有代码块标记
            elif response_content.startswith("{") and response_content.endswith("}"):
                pass  # 可能是纯 JSON
            else:
                # 尝试在文本中查找 JSON 对象
                json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(0)
                else:
                    logger.warning(f"LLM 响应似乎不包含有效的 JSON 对象。响应: {response_content}")
                    return {"is_exist": False}

            # 解析 JSON
            result = json.loads(response_content)

            # 结果验证和过滤
            if not isinstance(result, dict):
                logger.warning(f"LLM 响应不是一个有效的 JSON 对象 (字典类型)。响应内容: {response_content}")
                return {"is_exist": False}

            is_exist = result.get("is_exist")

            if is_exist is True:
                original_data = result.get("data")
                if isinstance(original_data, dict) and original_data:
                    logger.info(f"LLM 找到的原始绰号映射: {original_data}")
                    filtered_data = self._filter_llm_results(original_data, user_name_map)  # 调用过滤函数
                    if not filtered_data:
                        logger.info("所有找到的绰号映射都被过滤掉了。")
                        return {"is_exist": False}
                    else:
                        logger.info(f"过滤后的绰号映射: {filtered_data}")
                        return {"is_exist": True, "data": filtered_data}
                else:
                    # is_exist 为 True 但 data 缺失、不是字典或为空
                    logger.warning(f"LLM 响应格式错误: is_exist=True 但 data 无效。原始 data: {original_data}")
                    return {"is_exist": False}
            elif is_exist is False:
                logger.info("LLM 明确指示未找到可靠的绰号映射 (is_exist=False)。")
                return {"is_exist": False}
            else:  # is_exist 不是 True 或 False (包括 None)
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
        bot_qq_str = str(self.config.BOT_QQ) if hasattr(self.config, "BOT_QQ") else None

        for user_id, nickname in original_data.items():
            # 过滤条件 1: user_id 必须是字符串
            if not isinstance(user_id, str):
                logger.warning(f"过滤掉非字符串 user_id: {user_id}")
                continue
            # 过滤条件 2: 排除机器人自身
            if bot_qq_str and user_id == bot_qq_str:
                logger.debug(f"过滤掉机器人自身的映射: ID {user_id}")
                continue
            # 过滤条件 3: 排除 nickname 为空或仅包含空白的情况
            if not nickname or nickname.isspace():
                logger.debug(f"过滤掉用户 {user_id} 的空绰号。")
                continue

            # 过滤条件 4 (可选，根据 Prompt 效果决定是否保留): 排除 nickname 与已知名称相同的情况
            # person_name = user_name_map.get(user_id)
            # if person_name and person_name == nickname:
            #     logger.debug(f"过滤掉用户 {user_id} 的映射: 绰号 '{nickname}' 与其名称 '{person_name}' 相同。")
            #     continue

            # 如果通过所有过滤条件，则保留
            filtered_data[user_id] = nickname.strip()  # 保留时去除首尾空白

        return filtered_data

    # 线程相关
    def _run_processor_in_thread(self):
        """后台线程的入口函数，负责创建和运行 asyncio 事件循环。"""
        loop = None
        thread_id = threading.get_ident()
        logger.info(f"绰号处理器线程启动 (线程 ID: {thread_id})...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info(f"(线程 ID: {thread_id}) Asyncio 事件循环已创建并设置。")
            # 运行主处理循环直到停止事件被设置
            loop.run_until_complete(self._processing_loop())
        except Exception as e:
            logger.error(f"(线程 ID: {thread_id}) 运行绰号处理器线程时出错: {e}", exc_info=True)
        finally:
            # 确保循环被正确关闭
            if loop:
                try:
                    if loop.is_running():
                        logger.info(f"(线程 ID: {thread_id}) 正在停止 asyncio 循环...")
                        all_tasks = asyncio.all_tasks(loop)
                        if all_tasks:
                            logger.info(f"(线程 ID: {thread_id}) 正在取消 {len(all_tasks)} 个运行中的任务...")
                            for task in all_tasks:
                                task.cancel()
                            # 等待任务取消完成
                            loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                            logger.info(f"(线程 ID: {thread_id}) 所有任务已取消。")
                        loop.stop()
                        logger.info(f"(线程 ID: {thread_id}) 循环已停止。")
                    if not loop.is_closed():
                        loop.close()
                        logger.info(f"(线程 ID: {thread_id}) Asyncio 循环已关闭。")
                except Exception as loop_close_err:
                    logger.error(f"(线程 ID: {thread_id}) 关闭循环时出错: {loop_close_err}", exc_info=True)
            logger.info(f"绰号处理器线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
        """后台线程中运行的异步处理循环。"""
        thread_id = threading.get_ident()
        logger.info(f"绰号处理循环已启动 (线程 ID: {thread_id})。")

        while not self._stop_event.is_set():
            try:
                # 从队列中获取项目，设置超时以允许检查停止事件
                item = self.nickname_queue.get(block=True, timeout=self.sleep_interval)

                # 处理获取到的项目
                await self._analyze_and_update_nicknames(item)

                self.nickname_queue.task_done()  # 标记任务完成

            except queue.Empty:
                # 超时，队列为空，继续循环检查停止事件
                continue
            except asyncio.CancelledError:
                logger.info(f"绰号处理循环被取消 (线程 ID: {thread_id})。")
                break  # 任务被取消，退出循环
            except Exception as e:
                # 捕获处理单个项目时可能发生的其他异常
                logger.error(f"(线程 ID: {thread_id}) 绰号处理循环出错: {e}\n{traceback.format_exc()}")
                # 可以在这里添加错误处理逻辑，例如将失败的任务放回队列或记录到错误日志
                # 短暂休眠避免快速连续失败
                await asyncio.sleep(5)

        logger.info(f"绰号处理循环已结束 (线程 ID: {thread_id})。")


# 在模块级别创建单例实例
# 这使得其他模块可以通过 `from .nickname_manager import nickname_manager` 来导入和使用
nickname_manager = NicknameManager()
