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
from .nickname_mapper import _build_mapping_prompt
from .nickname_utils import select_nicknames_for_prompt, format_nickname_prompt_injection
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

    def __init__(self): # 标准单例实现
        """
        初始化 NicknameManager。
        使用锁和标志确保实际初始化只执行一次。
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return
            self._initialize_components() # 将实际初始化逻辑放入一个单独的方法

    def _initialize_components(self): # 实际的初始化逻辑
        logger.info("正在初始化 NicknameManager 组件...")
        self.is_enabled = global_config.group_nickname.enable_nickname_mapping

        person_info_collection = getattr(db, "person_info", None)
        self.db_handler = NicknameDB(person_info_collection)
        if not self.db_handler.is_available():
            logger.error("数据库处理器初始化失败，NicknameManager 功能受限。")
            self.is_enabled = False

        self.llm_mapper: Optional[LLMRequest] = None
        if self.is_enabled:
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

        self.queue_max_size = global_config.group_nickname.nickname_queue_max_size
        self.nickname_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_max_size)
        self._stop_event = threading.Event()
        self._nickname_thread: Optional[threading.Thread] = None
        self.sleep_interval = global_config.group_nickname.nickname_process_sleep_interval
        self._initialized = True
        logger.info("NicknameManager 初始化完成。")

    async def _get_user_primary_identifiers(
        self,
        uid_str: str,
        platform: str, # 保留，以备将来使用
        message_list_before_now: List[Dict],
    ) -> Tuple[str, str]:
        """
        辅助方法：为给定的 UID 获取其主要的“显示名”和“群名片”。
        严格按照 “QQ网名 > 群名片（如果QQ网名获取不到，则群名片作为显示名）> QQ号” 的优先级确定显示名。
        LLM名（person_name）不在此处使用。

        返回: (main_display_name, group_card_name_for_info)
        """
        main_display_name = f"用户{uid_str}" # 最终备用，如果其他都获取不到
        actual_group_card_name = "" # 存储从历史记录中实际获取的群名片
        latest_qq_nickname_from_history = ""

        # 从 message_list_before_now 获取最新的群名片和QQ昵称
        for msg_info in reversed(message_list_before_now): # 从最新消息开始找
            msg_user_info = msg_info.get("user_info", {})
            if str(msg_user_info.get("user_id")) == uid_str:
                # 只要找到该用户的任何一条消息，就尝试获取这两个信息
                if msg_user_info.get("user_nickname"): 
                    latest_qq_nickname_from_history = msg_user_info.get("user_nickname")
                if msg_user_info.get("user_cardname"): 
                    actual_group_card_name = msg_user_info.get("user_cardname")
            
                # 找到了该用户的最新一条相关消息，信息已提取完毕，可以跳出循环
                logger.debug(f"[PIDGetter] For UID '{uid_str}': Found in history - QQNickname='{latest_qq_nickname_from_history}', CardName='{actual_group_card_name}'")
                break 
    
        # 确定 main_display_name (您期望的“用户名”，即聊天记录中看到的，优先QQ网名)
        if latest_qq_nickname_from_history:
            main_display_name = latest_qq_nickname_from_history
            logger.debug(f"[PIDGetter] For UID '{uid_str}': Using QQNickname '{main_display_name}' as main_display_name.")
        elif actual_group_card_name: # 如果QQ网名没取到，但有群名片，则群名片作为主显示名
            main_display_name = actual_group_card_name
            logger.debug(f"[PIDGetter] For UID '{uid_str}': QQNickname empty, using CardName '{main_display_name}' as main_display_name.")
        else: # 如果两者都获取不到，则main_display_name 维持 f"用户{uid_str}"
            logger.debug(f"[PIDGetter] For UID '{uid_str}': Both QQNickname and CardName empty, main_display_name defaults to '{main_display_name}'.")

        # group_card_name_for_info 就是我们从历史记录中获取的 actual_group_card_name
        # 如果用户在群内没有设置群名片，QQ的行为通常是显示其QQ网名。
        # 在这种情况下，actual_group_card_name 可能是空的，或者等于 latest_qq_nickname_from_history。
        # 我们需要确保即使 actual_group_card_name 为空，如果 main_display_name 是QQ昵称，那么“群名片”应该显示这个QQ昵称。
        # 或者如果您的意思是，如果群里没设置，就显示QQ昵称，那 actual_group_card_name 字段就应该反映这一点。
    
        # 修正：如果实际群名片为空，并且QQ昵称存在，则群名片信息应该显示QQ昵称（模拟QQ行为）
        # 但这与“ta在这个群的群名称为”的语义有点冲突，如果群名片真的没设置，应该就是空。
        # 我们先保持 group_card_name_for_info 就是 actual_group_card_name 的原始值。
        # 如果您希望在 actual_group_card_name 为空时，让"群名称"部分显示QQ昵称，那需要在这里调整。
        # 暂时，我们让它真实反映获取到的群名片。
        group_card_name_for_info = actual_group_card_name

        logger.debug(f"[PIDGetter] Final for UID '{uid_str}': main_display_name='{main_display_name}', group_card_name_for_info='{group_card_name_for_info}'")
        return main_display_name, group_card_name_for_info


    async def get_nickname_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        if not self.is_enabled or not chat_stream or not chat_stream.group_info:
            return ""

        log_prefix = f"[{chat_stream.stream_id if chat_stream else 'UnknownStream'}:{chat_stream.group_info.group_id if chat_stream and chat_stream.group_info else 'UnknownGroup'}]"
        logger.debug(f"{log_prefix} Attempting to get nickname prompt injection (Strict Name Logic).")

        try:
            group_id_str = str(chat_stream.group_info.group_id)
            platform = chat_stream.platform
            
            current_user_ids_in_context = set()
            for msg in message_list_before_now: # 从当前上下文消息中获取用户
                msg_user_info = msg.get("user_info", {})
                uid_str = str(msg_user_info.get("user_id", ""))
                if uid_str:
                    current_user_ids_in_context.add(uid_str)
            
            if not current_user_ids_in_context and chat_stream: # 如果消息列表为空，尝试从 ChatStream 获取最近发言者
                recent_speakers = chat_stream.get_recent_speakers(limit=global_config.group_nickname.max_nicknames_in_prompt)
                current_user_ids_in_context.update(str(speaker["user_id"]) for speaker in recent_speakers)
            
            logger.debug(f"{log_prefix} User IDs in context for injection: {current_user_ids_in_context}")
            if not current_user_ids_in_context:
                logger.warning(f"{log_prefix} No user IDs found for nickname injection.")
                return ""

            # 获取这些用户已学习的绰号
            # users_with_learned_nicknames_data 的键是 person_name (LLM名)，值包含 user_id 和 nicknames 列表
            # 我们仍然需要这个数据来获取“已学习的绰号”
            users_with_learned_nicknames_data = await relationship_manager.get_users_group_nicknames(
                platform, list(current_user_ids_in_context), group_id_str
            )
            logger.debug(f"{log_prefix} Learned nicknames data from relationship_manager: {users_with_learned_nicknames_data}")

            # all_info_for_prompt 的键现在是 main_display_name (网名/群名片/用户UID)
            # 值是 {"user_id": "uid_str", "group_card_name": "card", "nicknames": [{"绰号": 次数}, ...]}
            all_info_for_prompt: Dict[str, Dict[str, Any]] = {}
            used_main_display_keys = set() # 用于确保主键的唯一性

            for uid_str in current_user_ids_in_context:
                main_display_name, group_card_name_for_info = await self._get_user_primary_identifiers(
                    uid_str, platform, message_list_before_now
                )
                
                # 处理 main_display_name 键的冲突 (极小概率，但保险)
                original_main_display_name = main_display_name
                counter = 1
                while main_display_name in used_main_display_keys:
                    main_display_name = f"{original_main_display_name}_{counter}"
                    counter += 1
                used_main_display_keys.add(main_display_name)

                if original_main_display_name != main_display_name:
                     logger.warning(f"{log_prefix} Main display name key conflict for '{original_main_display_name}' (UID: {uid_str}), new key is '{main_display_name}'.")

                # 查找该 uid_str 对应的已学习绰号
                learned_nicknames_list = []
                if users_with_learned_nicknames_data:
                    for _llm_name_key, data_val in users_with_learned_nicknames_data.items():
                        if data_val.get("user_id") == uid_str:
                            learned_nicknames_list = data_val.get("nicknames", [])
                            break 
                
                all_info_for_prompt[main_display_name] = {
                    "user_id": uid_str,
                    "group_card_name": group_card_name_for_info,
                    "nicknames": learned_nicknames_list 
                }
                logger.debug(f"{log_prefix} Compiled info for display_key '{main_display_name}' (UID: {uid_str}): {all_info_for_prompt[main_display_name]}")

            if all_info_for_prompt:
                logger.debug(f"{log_prefix} Data being passed to select_nicknames_for_prompt: {all_info_for_prompt}")
                # select_nicknames_for_prompt 的输入 all_nicknames_info_with_uid 的键现在是 main_display_name
                # 它返回的元组的第一个元素也将是这个 main_display_name
                selected_nicknames_with_info = select_nicknames_for_prompt(all_info_for_prompt)
                injection_str = format_nickname_prompt_injection(selected_nicknames_with_info)
                if injection_str:
                    logger.info(f"{log_prefix} Generated nickname prompt injection (Strict Name Logic):\n{injection_str}")
                else:
                    logger.debug(f"{log_prefix} No nickname injection string generated (select_nicknames_for_prompt returned empty or all items filtered).")
                return injection_str
            else:
                logger.warning(f"{log_prefix} No information gathered for prompt injection for any user in context.")
                return ""

        except Exception as e:
            logger.error(f"{log_prefix} Exception in get_nickname_prompt_injection (Strict Name Logic): {e}", exc_info=True)
            return ""

    async def trigger_nickname_analysis(
        self,
        anchor_message: MessageRecv,
        bot_reply: List[str],
        chat_stream: Optional[ChatStream] = None,
    ):
        """
        准备数据并将其排队等待绰号分析（如果满足条件）。
        (现在调用异步的 _add_to_queue)
        """
        if not self.is_enabled:
            return

        if random.random() < global_config.group_nickname.nickname_analysis_probability:
            logger.debug("跳过绰号分析：随机概率未命中。")
            return

        current_chat_stream = chat_stream or anchor_message.chat_stream
        if not current_chat_stream or not current_chat_stream.group_info:
            logger.debug("跳过绰号分析：非群聊或无效的聊天流。")
            return

        log_prefix = f"[{current_chat_stream.stream_id}]"
        try:
            # 1. 获取历史记录
            history_limit = global_config.group_nickname.nickname_analysis_history_limit
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

            user_name_map_for_llm: Dict[str,str] = {} 
            temp_user_ids_for_map = set()
            history_messages = get_raw_msg_before_timestamp_with_chat( # 确保 history_messages 在这里获取
                chat_id=current_chat_stream.stream_id,
                timestamp=time.time(),
                limit=global_config.group_nickname.nickname_analysis_history_limit,
            )
            for msg in history_messages: 
                msg_user_info = msg.get("user_info", {})
                uid_str = str(msg_user_info.get("user_id", ""))
                if uid_str:
                    temp_user_ids_for_map.add(uid_str)
        
            for uid_str_for_map in temp_user_ids_for_map:
                main_display_name_for_map, _ = await self._get_user_primary_identifiers( # 使用新的辅助函数
                    uid_str_for_map, 
                    current_chat_stream.platform, # 使用 current_chat_stream 的 platform
                    history_messages # 传递这段特定的历史给辅助函数
                )
                user_name_map_for_llm[uid_str_for_map] = main_display_name_for_map
        
            chat_history_str = await build_readable_messages( # chat_history_str 的构建也放在获取 history_messages 之后
                messages=history_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
                truncate=False, 
            )
            bot_reply_str = " ".join(bot_reply) if bot_reply else ""
            group_id = str(current_chat_stream.group_info.group_id)
            platform = current_chat_stream.platform # 确保 platform 从 current_chat_stream 获取

            item = (chat_history_str, bot_reply_str, platform, group_id, user_name_map_for_llm) # 使用新的 map
            await self._add_to_queue(item, platform, group_id)
            logger.debug(f"{log_prefix} Queued item for nickname analysis. User name map for LLM: {user_name_map_for_llm}")


            
            # 4. 构建用户 ID 到名称的映射 (user_name_map)
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
                for user_id in user_ids_in_history:
                    if user_id in names_data:
                        user_name_map[user_id] = names_data[user_id]
                    else:
                        latest_nickname = next(
                            (
                                m["user_info"].get("user_nickname")
                                for m in reversed(history_messages)
                                if str(m["user_info"].get("user_id")) == user_id and m["user_info"].get("user_nickname")
                            ),
                            None,
                        )
                        user_name_map[user_id] = (
                            latest_nickname or f"{global_config.bot.nickname}"
                            if user_id == global_config.bot.qq_account
                            else "未知"
                        )

            item = (chat_history_str, bot_reply_str, platform, group_id, user_name_map)
            await self._add_to_queue(item, platform, group_id)

        except Exception as e:
            logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)

    async def get_nickname_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        """
        获取并格式化用于 Prompt 注入的绰号信息字符串。
        (增加了获取和传递群名片的逻辑)
        """
        if not self.is_enabled or not chat_stream or not chat_stream.group_info:
            return ""

        log_prefix = f"[{chat_stream.stream_id}]"
        try:
            group_id_str = str(chat_stream.group_info.group_id) # group_id 应该是字符串
            platform = chat_stream.platform
            
            current_user_ids_in_context = { # 重命名以示区分
                str(msg["user_info"]["user_id"])
                for msg in message_list_before_now
                if msg.get("user_info", {}).get("user_id")
            }
            logger.debug(f"{log_prefix} User IDs in current message_list_before_now: {current_user_ids_in_context}")

            if not current_user_ids_in_context:
                # 如果 message_list_before_now 为空或不含用户信息，尝试从 chat_stream 获取最近发言者
                recent_speakers = chat_stream.get_recent_speakers(limit=global_config.group_nickname.max_nicknames_in_prompt) # 获取更多一些候选
                current_user_ids_in_context.update(str(speaker["user_id"]) for speaker in recent_speakers)
                logger.debug(f"{log_prefix} User IDs after checking recent speakers: {current_user_ids_in_context}")

            if not current_user_ids_in_context:
                logger.warning(f"{log_prefix} No user IDs found for nickname injection.")
                return ""

            # all_info_for_prompt 的键将是 person_name (LLM名或备用名)
            # 值是 {"user_id": "uid_str", "group_card_name": "card", "nicknames": [{"绰号": 次数}, ...]}
            all_info_for_prompt: Dict[str, Dict[str, Any]] = {}

            # 1. 获取已学习的绰号信息 (这可能只包含部分在 current_user_ids_in_context 中的用户)
            users_with_learned_nicknames_data = await relationship_manager.get_users_group_nicknames(
                platform, list(current_user_ids_in_context), group_id_str # 查询当前上下文中所有人的绰号
            )
            logger.debug(f"{log_prefix} Learned nicknames data from relationship_manager: {users_with_learned_nicknames_data}")

            # 2. 整合信息：优先使用 users_with_learned_nicknames_data，然后为其他在上下文中的用户补充基础信息
            for uid_str in current_user_ids_in_context:
                person_name_key = None
                user_cardname = ""
                learned_nicknames_list = []

                # 检查此 uid_str 是否在 users_with_learned_nicknames_data 的结果中
                # (需要遍历，因为 users_with_learned_nicknames_data 的键是 person_name)
                found_in_learned_data = False
                if users_with_learned_nicknames_data:
                    for pn_key, data_val in users_with_learned_nicknames_data.items():
                        if data_val.get("user_id") == uid_str:
                            person_name_key = pn_key
                            learned_nicknames_list = data_val.get("nicknames", [])
                            found_in_learned_data = True
                            # 群名片仍然需要从 message_list_before_now 或其他途径获取，因为 relationship_manager 可能不返回
                            break 
                
                # 统一获取群名片和 person_name_key (如果之前没找到)
                # 优先从当前消息上下文获取最新的群名片和用户名(LLM名 > QQ昵称)
                temp_person_name_for_key_lookup = None # 用于查找的用户名
                for msg_info in reversed(message_list_before_now): # message_list_before_now 更可靠
                    msg_user_info = msg_info.get("user_info", {})
                    if str(msg_user_info.get("user_id")) == uid_str:
                        if msg_user_info.get("user_cardname"):
                            user_cardname = msg_user_info.get("user_cardname")
                        
                        # 确定 person_name_key
                        if not person_name_key: # 如果之前没从 relationship_manager 的结果中获得 person_name
                            # 尝试从 person_info 获取LLM名
                            try:
                                temp_person_id_for_pi = person_info_manager.get_person_id(platform, int(uid_str))
                                llm_name = await person_info_manager.get_value(temp_person_id_for_pi, "person_name")
                                if llm_name:
                                    temp_person_name_for_key_lookup = llm_name
                            except ValueError: # uid_str 不是数字的情况（理论上QQ号是数字）
                                logger.warning(f"{log_prefix} User ID '{uid_str}' is not a valid integer for person_info_manager.")
                            except Exception as e_pi_name:
                                logger.warning(f"{log_prefix} Error getting person_name for UID '{uid_str}': {e_pi_name}")

                            if not temp_person_name_for_key_lookup and msg_user_info.get("user_nickname"):
                                temp_person_name_for_key_lookup = msg_user_info.get("user_nickname") # 备用QQ昵称
                            
                            if not temp_person_name_for_key_lookup:
                                temp_person_name_for_key_lookup = f"用户{uid_str}" # 最终备用
                            
                            person_name_key = temp_person_name_for_key_lookup
                        break # 找到该用户的最新消息即可

                if not person_name_key: # 如果遍历完 message_list 还是没有 person_name_key (例如该用户不在近期消息里，但在 recent_speakers 里)
                    # 再次尝试从 person_info 获取LLM名
                    try:
                        temp_person_id_for_pi = person_info_manager.get_person_id(platform, int(uid_str))
                        llm_name = await person_info_manager.get_value(temp_person_id_for_pi, "person_name")
                        if llm_name:
                            person_name_key = llm_name
                    except Exception: pass # 忽略错误
                    if not person_name_key:
                        person_name_key = f"用户{uid_str}" # 最终的最终备用

                # 确保 person_name_key 的唯一性，如果冲突则添加后缀
                original_person_name_key = person_name_key
                counter = 1
                while person_name_key in all_info_for_prompt and all_info_for_prompt[person_name_key].get("user_id") != uid_str : # 确保不是因为同一个用户被多次处理
                    person_name_key = f"{original_person_name_key}_{counter}"
                    counter +=1
                
                if original_person_name_key != person_name_key:
                     logger.warning(f"{log_prefix} Person name key conflict for '{original_person_name_key}', new key is '{person_name_key}' for user_id '{uid_str}'.")
                
                # 存入或更新 all_info_for_prompt
                # 如果用户已在 all_info_for_prompt 中（通过不同的 person_name_key 但相同的 uid_str），则不应重复添加
                # 但我们的循环是基于 uid_str，所以每个 uid_str 只会处理一次
                all_info_for_prompt[person_name_key] = {
                    "user_id": uid_str,
                    "group_card_name": user_cardname, # 这是新获取或已有的
                    "nicknames": learned_nicknames_list  # 来自 relationship_manager 或为空列表
                }
                logger.debug(f"{log_prefix} Compiled info for person_name_key '{person_name_key}' (UID: {uid_str}): card='{user_cardname}', learned_nicknames_count={len(learned_nicknames_list)}")


            if all_info_for_prompt:
                logger.debug(f"{log_prefix} Data being passed to select_nicknames_for_prompt: {all_info_for_prompt}")
                selected_nicknames_with_info = select_nicknames_for_prompt(all_info_for_prompt) # select_nicknames_for_prompt 需要能处理 nicknames 为空的情况
                injection_str = format_nickname_prompt_injection(selected_nicknames_with_info)
                if injection_str:
                    logger.info(f"{log_prefix} Generated nickname prompt injection (with group names):\n{injection_str}")
                else:
                    logger.debug(f"{log_prefix} No nickname injection string generated.")
                return injection_str
            else:
                logger.warning(f"{log_prefix} No information gathered for prompt injection for any user in context.")
                return ""

        except Exception as e:
            logger.error(f"{log_prefix} Exception in get_nickname_prompt_injection: {e}", exc_info=True)
            return ""

    # 私有/内部方法

    async def _add_to_queue(self, item: tuple, platform: str, group_id: str): # 保持不变
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

    async def _analyze_and_update_nicknames(self, item: tuple): # 保持不变
        if not isinstance(item, tuple) or len(item) != 5:
            logger.warning(f"从队列接收到无效项目: {type(item)}")
            return

        chat_history_str, bot_reply, platform, group_id, user_name_map = item
        log_prefix = f"[{platform}:{group_id}]"
        logger.debug(f"{log_prefix} 开始处理绰号分析任务 (user_name_map: {user_name_map})...")


        if not self.llm_mapper:
            logger.error(f"{log_prefix} LLM 映射器不可用，无法执行分析。")
            return
        if not self.db_handler.is_available():
            logger.error(f"{log_prefix} 数据库处理器不可用，无法更新计数。")
            return

        analysis_result = await self._call_llm_for_analysis(chat_history_str, bot_reply, user_name_map)

        if analysis_result.get("is_exist") and analysis_result.get("data"):
            nickname_map_to_update = analysis_result["data"] # 预期是 { "uid_str": "nickname", ... }
            logger.info(f"{log_prefix} LLM 找到绰号映射，准备更新数据库: {nickname_map_to_update}")

            for user_id_str_from_llm, nickname in nickname_map_to_update.items():
                if not user_id_str_from_llm or not nickname: # nickname 也不能为空
                    logger.warning(f"{log_prefix} 跳过无效条目: user_id='{user_id_str_from_llm}', nickname='{nickname}'")
                    continue
                
                person_id = None
                try:
                    # 假设 LLM 返回的 key (user_id_str_from_llm) 确实是数字UID字符串
                    person_id = person_info_manager.get_person_id(platform, int(user_id_str_from_llm))
                except ValueError:
                     logger.error(f"{log_prefix} 无法将LLM返回的键 '{user_id_str_from_llm}' 转换为整数以获取person_id。")
                     continue 
                except Exception as e_get_pid:
                    logger.error(f"{log_prefix} 获取 person_id 失败 for key '{user_id_str_from_llm}': {e_get_pid}")
                    continue

                if not person_id:
                    logger.error(f"{log_prefix} 无法为 platform='{platform}', key='{user_id_str_from_llm}' 生成 person_id，跳过此用户。")
                    continue
                try:
                    # 存储时使用 int(user_id_str_from_llm) 作为原始用户ID
                    self.db_handler.upsert_person(person_id, int(user_id_str_from_llm), platform)
                    self.db_handler.update_group_nickname_count(person_id, group_id, nickname)
                    logger.info(f"{log_prefix} Updated nickname count for person_id='{person_id}', group='{group_id}', nickname='{nickname}'")
                except (OperationFailure, DuplicateKeyError) as db_err:
                    logger.exception(f"{log_prefix} 数据库操作失败: 用户key {user_id_str_from_llm}, 绰号 {nickname}. Error: {db_err}")
                except Exception as e:
                    logger.exception(f"{log_prefix} 处理用户key {user_id_str_from_llm} 的绰号 '{nickname}' 时发生意外错误：{e}")
        else:
            logger.debug(f"{log_prefix} LLM 未找到可靠的绰号映射或分析失败。")
            
    async def _call_llm_for_analysis( # 基本保持不变
        self,
        chat_history_str: str,
        bot_reply: str,
        user_name_map: Dict[str, str], 
    ) -> Dict[str, Any]:
        if not self.llm_mapper:
            logger.error("LLM 映射器未初始化，无法执行分析。")
            return {"is_exist": False}

        prompt = _build_mapping_prompt(chat_history_str, bot_reply, user_name_map) # user_name_map 是 uid -> display_name
        logger.debug(f"构建的绰号映射 Prompt (传递给LLM的user_name_map: {user_name_map}):\n{prompt[:500]}...")
        try:
            response_content, _, _ = await self.llm_mapper.generate_response(prompt)
            logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")
            if not response_content: return {"is_exist": False}
            response_content = response_content.strip()
            match = re.match(r"^```(?:\w+)?\s*\n(.*?)\n\s*```$", response_content, re.DOTALL | re.IGNORECASE)
            if match: response_content = match.group(1).strip()
            elif not (response_content.startswith("{") and response_content.endswith("}")):
                json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if json_match: response_content = json_match.group(0)
                else: return {"is_exist": False}
            if not response_content: return {"is_exist": False}
            result = json.loads(response_content)
            if not isinstance(result, dict): return {"is_exist": False}
            is_exist = result.get("is_exist")
            if is_exist is True:
                original_data = result.get("data")
                if isinstance(original_data, dict) and original_data:
                    filtered_data = self._filter_llm_results(original_data, user_name_map) # user_name_map is uid -> display_name
                    if not filtered_data: return {"is_exist": False}
                    return {"is_exist": True, "data": filtered_data} # data is { "uid_str": "nickname" }
                return {"is_exist": False}
            elif is_exist is False: return {"is_exist": False}
            return {"is_exist": False}
        except json.JSONDecodeError as json_err: logger.error(f"解析LLM JSON失败: {json_err}\n原始: {response_content[:200]}"); return {"is_exist": False}
        except Exception as e: logger.error(f"LLM调用或处理错: {e}", exc_info=True); return {"is_exist": False}


    def _filter_llm_results(self, original_data: Dict[str, str], user_name_map_for_context: Dict[str, str]) -> Dict[str, str]: # user_name_map_for_context is uid -> display_name
        filtered_data = {}
        bot_qq_str = global_config.bot.qq_account if global_config.bot.qq_account else None

        for uid_key_from_llm, nickname_from_llm in original_data.items(): # uid_key_from_llm 应该是UID字符串
            if not isinstance(uid_key_from_llm, str):
                logger.warning(f"过滤掉LLM返回的非字符串 user_id_key: {uid_key_from_llm}")
                continue
            if bot_qq_str and uid_key_from_llm == bot_qq_str:
                logger.debug(f"过滤掉机器人自身的映射: ID {uid_key_from_llm}")
                continue
            if not nickname_from_llm or nickname_from_llm.isspace():
                logger.debug(f"过滤掉用户 {uid_key_from_llm} 的空绰号。")
                continue
            
            # 检查绰号是否与该用户的已知“主显示名”（来自user_name_map_for_context）相同
            # user_name_map_for_context 的键是 uid_str
            current_display_name = user_name_map_for_context.get(uid_key_from_llm)
            if current_display_name and current_display_name == nickname_from_llm:
                 logger.debug(f"过滤掉用户 {uid_key_from_llm} 的映射: 绰号 '{nickname_from_llm}' 与其当前显示名称 '{current_display_name}' 相同。")
                 continue
            # 未来可以扩展：如果 nickname_from_llm 也等于该用户的QQ原始昵称（如果能获取到的话），也过滤。

            filtered_data[uid_key_from_llm] = nickname_from_llm.strip()
        return filtered_data

    # start_processor, stop_processor, _run_processor_in_thread, _processing_loop 保持您文件中的版本
    def start_processor(self):
        if not self.is_enabled:
            logger.info("绰号处理功能已禁用，处理器未启动。")
            return
        if global_config.group_nickname.max_nicknames_in_prompt == 0:
            logger.error("[错误] 绰号注入数量不合适，绰号处理功能已禁用！")
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
            
    def _run_processor_in_thread(self):
        thread_id = threading.get_ident() 
        logger.info(f"绰号处理器线程启动 (线程 ID: {thread_id})...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"(线程 ID: {thread_id}) Asyncio 事件循环已创建并设置。")
        # run_async_loop 应该在您的文件中定义
        run_async_loop(loop, self._processing_loop())
        logger.info(f"绰号处理器线程结束 (线程 ID: {thread_id}).")

    async def _processing_loop(self):
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
                await asyncio.sleep(5)
        logger.info("绰号异步处理循环已结束。")


nickname_manager = NicknameManager()
