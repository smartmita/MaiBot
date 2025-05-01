import random
import time
from typing import List, Dict, Tuple, Optional
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.plugins.person_info.relationship_manager import relationship_manager
from src.plugins.chat.chat_stream import ChatStream
from src.plugins.chat.message import MessageRecv
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from .nickname_processor import add_to_nickname_queue


logger = get_logger("nickname_utils")


def select_nicknames_for_prompt(all_nicknames_info: Dict[str, List[Dict[str, int]]]) -> List[Tuple[str, str, int]]:
    """
    从给定的绰号信息中，根据映射次数加权随机选择最多 N 个绰号。
    """
    if not all_nicknames_info:
        return []

    candidates = []
    for user_name, nicknames in all_nicknames_info.items():
        if nicknames:
            for nickname_entry in nicknames:
                if isinstance(nickname_entry, dict) and len(nickname_entry) == 1:
                    nickname, count = list(nickname_entry.items())[0]
                    if isinstance(count, int) and count > 0:
                        weight = count + global_config.NICKNAME_PROBABILITY_SMOOTHING
                        candidates.append((user_name, nickname, count, weight))
                    else:
                        logger.warning(
                            f"Invalid count for nickname '{nickname}' of user '{user_name}': {count}. Skipping."
                        )
                else:
                    logger.warning(f"Invalid nickname entry format for user '{user_name}': {nickname_entry}. Skipping.")

    if not candidates:
        return []

    total_weight = sum(c[3] for c in candidates)

    if total_weight <= 0:
        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = candidates[: global_config.MAX_NICKNAMES_IN_PROMPT]
    else:
        probabilities = [c[3] / total_weight for c in candidates]
        num_to_select = min(global_config.MAX_NICKNAMES_IN_PROMPT, len(candidates))
        try:
            selected_indices = set()
            selected = []
            attempts = 0
            max_attempts = num_to_select * 5
            while len(selected) < num_to_select and attempts < max_attempts:
                chosen_index = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
                if chosen_index not in selected_indices:
                    selected_indices.add(chosen_index)
                    selected.append(candidates[chosen_index])
                attempts += 1
            if len(selected) < num_to_select:
                remaining_candidates = [c for i, c in enumerate(candidates) if i not in selected_indices]
                remaining_candidates.sort(key=lambda x: x[2], reverse=True)
                needed = num_to_select - len(selected)
                selected.extend(remaining_candidates[:needed])
        except Exception as e:
            logger.error(
                f"Error during weighted random choice for nicknames: {e}. Falling back to top N.", exc_info=True
            )
            candidates.sort(key=lambda x: x[2], reverse=True)
            selected = candidates[: global_config.MAX_NICKNAMES_IN_PROMPT]

    result = [(user, nick, count) for user, nick, count, _weight in selected]
    result.sort(key=lambda x: x[2], reverse=True)
    logger.debug(f"Selected nicknames for prompt: {result}")
    return result


def format_nickname_prompt_injection(selected_nicknames: List[Tuple[str, str, int]]) -> str:
    """
    将选中的绰号信息格式化为注入 Prompt 的字符串。
    (代码保持不变)
    """
    if not selected_nicknames:
        return ""

    prompt_lines = ["【群成员绰号信息】"]
    grouped_by_user: Dict[str, List[str]] = {}

    for user_name, nickname, _count in selected_nicknames:
        if user_name not in grouped_by_user:
            grouped_by_user[user_name] = []
        grouped_by_user[user_name].append(f"“{nickname}”")

    for user_name, nicknames in grouped_by_user.items():
        nicknames_str = "、".join(nicknames)
        prompt_lines.append(f"- {user_name}，有时被称为：{nicknames_str}")

    if len(prompt_lines) > 1:
        return "\n".join(prompt_lines) + "\n"
    else:
        return ""


async def get_nickname_injection_for_prompt(chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
    """
    获取并格式化用于 Prompt 注入的绰号信息字符串。
    """
    nickname_injection_str = ""
    if global_config.ENABLE_NICKNAME_MAPPING and chat_stream and chat_stream.group_info:
        try:
            group_id = str(chat_stream.group_info.group_id)
            user_ids_in_context = set()
            if message_list_before_now:
                for msg in message_list_before_now:
                    sender_id = msg["user_info"].get("user_id")
                    if sender_id:
                        user_ids_in_context.add(str(sender_id))
            else:
                recent_speakers = chat_stream.get_recent_speakers(limit=5)
                for speaker in recent_speakers:
                    user_ids_in_context.add(str(speaker['user_id']))
                if not user_ids_in_context:
                    logger.warning(f"[{chat_stream.stream_id}] No messages or recent speakers found for nickname injection.")

            if user_ids_in_context:
                platform = chat_stream.platform
                all_nicknames_data = await relationship_manager.get_users_group_nicknames(
                    platform, list(user_ids_in_context), group_id
                )
                if all_nicknames_data:
                    selected_nicknames = select_nicknames_for_prompt(all_nicknames_data)
                    nickname_injection_str = format_nickname_prompt_injection(selected_nicknames)
                    if nickname_injection_str:
                        logger.debug(f"[{chat_stream.stream_id}] Generated nickname info for prompt:\n{nickname_injection_str}")
        except Exception as e:
            logger.error(f"[{chat_stream.stream_id}] Error getting or formatting nickname info for prompt: {e}", exc_info=True)
            nickname_injection_str = ""
    return nickname_injection_str


# --- 新增：触发绰号分析的工具函数 ---
async def trigger_nickname_analysis_if_needed(
    anchor_message: MessageRecv,
    bot_reply: List[str],
    chat_stream: Optional[ChatStream] = None # 允许传入 chat_stream 或从 anchor_message 获取
):
    """
    如果满足条件（群聊、功能开启），则准备数据并触发绰号分析任务。

    Args:
        anchor_message: 触发回复的原始消息对象。
        bot_reply: Bot 生成的回复内容列表。
        chat_stream: 可选的 ChatStream 对象。
    """
    # 检查功能是否开启
    if not global_config.ENABLE_NICKNAME_MAPPING:
        return

    # 确定使用的 chat_stream
    current_chat_stream = chat_stream or anchor_message.chat_stream

    # 检查是否是群聊且 chat_stream 有效
    if not current_chat_stream or not current_chat_stream.group_info:
        logger.debug(f"[{current_chat_stream.stream_id if current_chat_stream else 'Unknown'}] Skipping nickname analysis: Not a group chat or invalid chat stream.")
        return

    log_prefix = f"[{current_chat_stream.stream_id}]" # 日志前缀

    try:
        # 1. 获取历史记录
        history_limit = 30  # 可配置的历史记录条数
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

        # 2. 获取 Bot 回复字符串
        bot_reply_str = " ".join(bot_reply) if bot_reply else "" # 处理空回复列表

        # 3. 获取群号和平台
        group_id = str(current_chat_stream.group_info.group_id)
        platform = current_chat_stream.platform

        # 4. 构建用户 ID 到名称的映射
        user_ids_in_history = set()
        for msg in history_messages:
            sender_id = msg["user_info"].get("user_id")
            if sender_id:
                user_ids_in_history.add(str(sender_id))

        user_name_map = {}
        if user_ids_in_history:
            try:
                # 批量获取 person_name
                names_data = await relationship_manager.get_person_names_batch(platform, list(user_ids_in_history))
            except Exception as e:
                logger.error(f"{log_prefix} Error getting person names batch: {e}", exc_info=True)
                names_data = {}

            for user_id in user_ids_in_history:
                if user_id in names_data:
                    user_name_map[user_id] = names_data[user_id]
                else:
                    # 回退查找 nickname (从后往前找最新的)
                    latest_nickname = next(
                        (
                            m["user_info"].get("user_nickname") # 从 user_info 获取
                            for m in reversed(history_messages)
                            if str(m["user_info"].get("user_id")) == user_id and m["user_info"].get("user_nickname") # 确保 nickname 存在
                        ),
                        None,
                    )
                    user_name_map[user_id] = latest_nickname or f"未知({user_id})" # 提供回退

        # 5. 添加到处理队列
        await add_to_nickname_queue(chat_history_str, bot_reply_str, platform, group_id, user_name_map)
        logger.debug(f"{log_prefix} Triggered nickname analysis for group {group_id}.")

    except Exception as e:
        logger.error(f"{log_prefix} Error triggering nickname analysis: {e}", exc_info=True)