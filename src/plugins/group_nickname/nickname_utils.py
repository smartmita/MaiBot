# GroupNickname/nickname_utils.py
import random
import time
from typing import List, Dict, Tuple, Optional, Any
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.plugins.person_info.relationship_manager import relationship_manager
from src.plugins.chat.chat_stream import ChatStream
from src.plugins.chat.message import MessageRecv
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from .nickname_processor import add_to_nickname_queue


# 获取日志记录器，命名为 "绰号工具"
logger = get_logger("nickname_utils")


def select_nicknames_for_prompt(all_nicknames_info: Dict[str, List[Dict[str, int]]]) -> List[Tuple[str, str, int]]:
    """
    从给定的绰号信息中，根据映射次数加权随机选择最多 N 个绰号。

    Args:
        all_nicknames_info: 包含用户及其绰号信息的字典，格式为
                        { "用户名1": [{"绰号A": 次数}, {"绰号B": 次数}], ... }

    Returns:
        List[Tuple[str, str, int]]: 选中的绰号列表，每个元素为 (用户名, 绰号, 次数)。
                                    按次数降序排序。
    """
    if not all_nicknames_info:
        # 如果输入为空，直接返回空列表
        return []

    candidates = [] # 候选绰号列表，包含 (用户名, 绰号, 次数, 权重)
    for user_name, nicknames in all_nicknames_info.items():
        if nicknames:
            for nickname_entry in nicknames:
                # nickname_entry 应该是 {"绰号": 次数} 格式
                if isinstance(nickname_entry, dict) and len(nickname_entry) == 1:
                    nickname, count = list(nickname_entry.items())[0]
                    # 确保次数是正整数
                    if isinstance(count, int) and count > 0:
                        # 添加平滑因子，避免概率为0，并让低频词也有机会
                        weight = count + global_config.NICKNAME_PROBABILITY_SMOOTHING
                        candidates.append((user_name, nickname, count, weight))
                    else:
                        # 日志：记录无效的绰号次数
                        logger.warning(
                            f"用户 '{user_name}' 的绰号 '{nickname}' 次数无效: {count}。已跳过。"
                        )
                else:
                    # 日志：记录无效的绰号条目格式
                    logger.warning(f"用户 '{user_name}' 的绰号条目格式无效: {nickname_entry}。已跳过。")

    if not candidates:
        # 如果没有有效的候选绰号，返回空列表
        return []

    # 计算总权重
    total_weight = sum(c[3] for c in candidates)

    if total_weight <= 0:
        # 如果所有权重都无效或为0，则按原始次数排序选择前 N 个
        logger.warning("所有候选绰号的总权重为0或负数，将按原始次数选择 Top N。")
        candidates.sort(key=lambda x: x[2], reverse=True)  # 按原始次数排序
        selected = candidates[: global_config.MAX_NICKNAMES_IN_PROMPT]
    else:
        # 计算归一化概率
        probabilities = [c[3] / total_weight for c in candidates]

        # 使用概率分布进行加权随机选择（不重复）
        num_to_select = min(global_config.MAX_NICKNAMES_IN_PROMPT, len(candidates))
        try:
            # 实现不重复加权抽样
            selected_indices = set()
            selected = []
            attempts = 0
            max_attempts = num_to_select * 5  # 设置最大尝试次数，防止无限循环

            while len(selected) < num_to_select and attempts < max_attempts:
                # 每次只选一个
                chosen_index = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
                if chosen_index not in selected_indices:
                    selected_indices.add(chosen_index)
                    selected.append(candidates[chosen_index])
                attempts += 1

            # 如果尝试多次后仍未选够，补充出现次数最多的
            if len(selected) < num_to_select:
                logger.debug(f"加权随机选择后数量不足 ({len(selected)}/{num_to_select})，补充选择次数最多的。")
                remaining_candidates = [c for i, c in enumerate(candidates) if i not in selected_indices]
                remaining_candidates.sort(key=lambda x: x[2], reverse=True) # 按原始次数排序
                needed = num_to_select - len(selected)
                selected.extend(remaining_candidates[:needed])

        except Exception as e:
            # 日志：记录加权随机选择时发生的错误，并回退到简单选择
            logger.error(
                f"绰号加权随机选择时出错: {e}。将回退到选择次数最多的 Top N。", exc_info=True
            )
            # 出错时回退到选择次数最多的 N 个
            candidates.sort(key=lambda x: x[2], reverse=True)
            selected = candidates[: global_config.MAX_NICKNAMES_IN_PROMPT]

    # 格式化输出结果为 (用户名, 绰号, 次数)
    result = [(user, nick, count) for user, nick, count, _weight in selected]
    result.sort(key=lambda x: x[2], reverse=True)  # 按次数降序

    # 日志：记录最终选中的用于 Prompt 的绰号
    logger.debug(f"为 Prompt 选择的绰号: {result}")
    return result


def format_nickname_prompt_injection(selected_nicknames: List[Tuple[str, str, int]]) -> str:
    """
    将选中的绰号信息格式化为注入 Prompt 的字符串。

    Args:
        selected_nicknames: 选中的绰号列表 (用户名, 绰号, 次数)。

    Returns:
        str: 格式化后的字符串，如果列表为空则返回空字符串。
    """
    if not selected_nicknames:
        # 如果没有选中的绰号，返回空字符串
        return ""

    prompt_lines = ["以下是聊天记录中一些成员在本群的绰号信息（按常用度排序），如果有需要提及对方，用你认为合适的方式提及："] # 注入部分的标题
    grouped_by_user: Dict[str, List[str]] = {} # 用于按用户分组

    # 按用户分组绰号
    for user_name, nickname, _count in selected_nicknames:
        if user_name not in grouped_by_user:
            grouped_by_user[user_name] = []
        # 添加中文引号以区分绰号
        grouped_by_user[user_name].append(f"“{nickname}”")

    # 构建每个用户的绰号字符串
    for user_name, nicknames in grouped_by_user.items():
        nicknames_str = "、".join(nicknames) # 使用中文顿号连接
        prompt_lines.append(f"- 你私下称呼ta为{user_name}，ta被有时被群友称为：{nicknames_str}") # 格式化输出

    # 如果只有标题行，返回空字符串，避免注入无意义的标题
    if len(prompt_lines) > 1:
        # 末尾加换行符，以便在 Prompt 中正确分隔
        return "\n".join(prompt_lines) + "\n"
    else:
        return ""


async def get_nickname_injection_for_prompt(chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
    """
    获取并格式化用于 Prompt 注入的绰号信息字符串。
    这是一个封装函数，整合了获取、选择和格式化的逻辑。

    Args:
        chat_stream: 当前的 ChatStream 对象。
        message_list_before_now: 用于确定上下文中用户的消息列表。

    Returns:
        str: 格式化后的绰号信息字符串，如果无法获取或格式化则返回空字符串。
    """
    nickname_injection_str = ""
    # 仅在群聊且功能开启时执行
    if global_config.ENABLE_NICKNAME_MAPPING and chat_stream and chat_stream.group_info:
        try:
            group_id = str(chat_stream.group_info.group_id)
            user_ids_in_context = set() # 存储上下文中出现的用户ID

            # 从消息列表中提取用户ID
            if message_list_before_now:
                for msg in message_list_before_now:
                    sender_id = msg["user_info"].get("user_id")
                    if sender_id:
                        user_ids_in_context.add(str(sender_id))
            else:
                # 如果消息列表为空，尝试获取最近发言者作为上下文用户
                recent_speakers = chat_stream.get_recent_speakers(limit=5) # 获取最近5个发言者
                for speaker in recent_speakers:
                    user_ids_in_context.add(str(speaker['user_id']))
                if not user_ids_in_context:
                    # 日志：记录未找到上下文用户
                    logger.warning(f"[{chat_stream.stream_id}] 未找到消息或最近发言者用于绰号注入。")

            # 如果找到了上下文用户
            if user_ids_in_context:
                platform = chat_stream.platform
                # --- 调用批量获取群组绰号的方法 ---
                # 使用 relationship_manager 从数据库获取数据
                all_nicknames_data = await relationship_manager.get_users_group_nicknames(
                    platform, list(user_ids_in_context), group_id
                )

                # 如果获取到了绰号数据
                if all_nicknames_data:
                    # 调用选择和格式化函数
                    selected_nicknames = select_nicknames_for_prompt(all_nicknames_data)
                    nickname_injection_str = format_nickname_prompt_injection(selected_nicknames)
                    if nickname_injection_str:
                        # 日志：记录生成的用于 Prompt 的绰号信息
                        logger.debug(f"[{chat_stream.stream_id}] 已生成用于 Prompt 的绰号信息:\n{nickname_injection_str}")

        except Exception as e:
            # 日志：记录获取或格式化绰号信息时发生的错误
            logger.error(f"[{chat_stream.stream_id}] 获取或格式化 Prompt 绰号信息时出错: {e}", exc_info=True)
            nickname_injection_str = "" # 出错时确保返回空字符串

    # 返回最终生成的字符串（可能为空）
    return nickname_injection_str


async def trigger_nickname_analysis_if_needed(
    anchor_message: MessageRecv,
    bot_reply: List[str],
    chat_stream: Optional[ChatStream] = None # 允许传入 chat_stream 或从 anchor_message 获取
):
    """
    如果满足条件（群聊、功能开启），则准备数据并触发绰号分析任务。
    将相关信息放入处理队列，由 nickname_processor 处理。

    Args:
        anchor_message: 触发回复的原始消息对象。
        bot_reply: Bot 生成的回复内容列表。
        chat_stream: 可选的 ChatStream 对象。
    """
    # 检查功能是否开启
    if not global_config.ENABLE_NICKNAME_MAPPING:
        return # 如果功能禁用，直接返回

    # 确定使用的 chat_stream
    current_chat_stream = chat_stream or anchor_message.chat_stream

    # 检查是否是群聊且 chat_stream 有效
    if not current_chat_stream or not current_chat_stream.group_info:
        # 日志：记录跳过分析的原因（非群聊或无效流）
        logger.debug(f"[{current_chat_stream.stream_id if current_chat_stream else '未知流'}] 跳过绰号分析：非群聊或无效聊天流。")
        return

    log_prefix = f"[{current_chat_stream.stream_id}]" # 用于日志的前缀

    try:
        # 1. 获取历史记录
        history_limit = 30  # 定义获取历史记录的数量限制
        history_messages = get_raw_msg_before_timestamp_with_chat(
            chat_id=current_chat_stream.stream_id,
            timestamp=time.time(), # 获取当前时间之前的记录
            limit=history_limit,
        )

        # 格式化历史记录为可读字符串
        chat_history_str = await build_readable_messages(
            messages=history_messages,
            replace_bot_name=True, # 替换机器人名称，以便 LLM 分析
            merge_messages=False, # 不合并消息，保留原始对话结构
            timestamp_mode="relative", # 使用相对时间戳
            read_mark=0.0, # 不需要已读标记
            truncate=False, # 获取完整内容进行分析
        )

        # 2. 获取 Bot 回复字符串
        bot_reply_str = " ".join(bot_reply) if bot_reply else "" # 处理空回复列表

        # 3. 获取群号和平台信息
        group_id = str(current_chat_stream.group_info.group_id)
        platform = current_chat_stream.platform

        # 4. 构建用户 ID 到名称的映射 (user_name_map)
        user_ids_in_history = set() # 存储历史记录中出现的用户ID
        for msg in history_messages:
            sender_id = msg["user_info"].get("user_id")
            if sender_id:
                user_ids_in_history.add(str(sender_id))

        user_name_map = {} # 初始化映射字典
        if user_ids_in_history:
            try:
                # 批量从数据库获取这些用户的 person_name
                names_data = await relationship_manager.get_person_names_batch(platform, list(user_ids_in_history))
            except Exception as e:
                # 日志：记录获取 person_name 时发生的错误
                logger.error(f"{log_prefix} 批量获取 person_name 时出错: {e}", exc_info=True)
                names_data = {} # 出错时使用空字典

            # 填充 user_name_map
            for user_id in user_ids_in_history:
                if user_id in names_data:
                    # 如果数据库中有 person_name，则使用它
                    user_name_map[user_id] = names_data[user_id]
                else:
                    # 如果数据库中没有，则回退查找用户在历史记录中最近使用的 nickname
                    latest_nickname = next(
                        (
                            m["user_info"].get("user_nickname") # 从 user_info 获取 nickname
                            for m in reversed(history_messages) # 从后往前找
                            # 确保消息的用户ID匹配且 nickname 存在
                            if str(m["user_info"].get("user_id")) == user_id and m["user_info"].get("user_nickname")
                        ),
                        None, # 如果找不到，返回 None
                    )
                    # 如果找到了 nickname 则使用，否则使用 "未知(ID)"
                    user_name_map[user_id] = latest_nickname or f"未知({user_id})"

        # 5. 将准备好的数据添加到绰号处理队列
        await add_to_nickname_queue(chat_history_str, bot_reply_str, platform, group_id, user_name_map)
        # 日志：记录已成功触发分析任务
        logger.debug(f"{log_prefix} 已为群组 {group_id} 触发绰号分析任务。")

    except Exception as e:
        # 日志：记录触发分析过程中发生的任何其他错误
        logger.error(f"{log_prefix} 触发绰号分析时出错: {e}", exc_info=True)