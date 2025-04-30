# GroupNickname/nickname_utils.py
import random
from typing import List, Dict, Tuple, Optional
from src.common.logger_manager import get_logger
from .config import MAX_NICKNAMES_IN_PROMPT, NICKNAME_PROBABILITY_SMOOTHING

logger = get_logger("nickname_utils")

def select_nicknames_for_prompt(
    all_nicknames_info: Dict[str, List[Dict[str, int]]]
) -> List[Tuple[str, str, int]]:
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
        return []

    candidates = []
    for user_name, nicknames in all_nicknames_info.items():
        if nicknames:
            for nickname_entry in nicknames:
                # nickname_entry 应该是 {"绰号": 次数} 格式
                if isinstance(nickname_entry, dict) and len(nickname_entry) == 1:
                    nickname, count = list(nickname_entry.items())[0]
                    # 确保次数是正整数
                    if isinstance(count, int) and count > 0:
                         # 添加平滑因子，避免概率为0，并让低频词也有机会
                        weight = count + NICKNAME_PROBABILITY_SMOOTHING
                        candidates.append((user_name, nickname, count, weight))
                    else:
                         logger.warning(f"Invalid count for nickname '{nickname}' of user '{user_name}': {count}. Skipping.")
                else:
                    logger.warning(f"Invalid nickname entry format for user '{user_name}': {nickname_entry}. Skipping.")


    if not candidates:
        return []

    # 计算总权重
    total_weight = sum(c[3] for c in candidates)

    if total_weight <= 0:
        # 如果所有权重都无效或为0，则随机选择（或按次数选择）
        candidates.sort(key=lambda x: x[2], reverse=True) # 按原始次数排序
        selected = candidates[:MAX_NICKNAMES_IN_PROMPT]
    else:
        # 计算归一化概率
        probabilities = [c[3] / total_weight for c in candidates]

        # 使用概率分布进行加权随机选择（不重复）
        num_to_select = min(MAX_NICKNAMES_IN_PROMPT, len(candidates))
        try:
            # random.choices 允许重复，我们需要不重复的选择
            # 可以使用 numpy.random.choice 或手动实现不重复加权抽样
            # 这里用一个简化的方法：多次 choices 然后去重，直到达到数量或无法再选
            selected_indices = set()
            selected = []
            attempts = 0
            max_attempts = num_to_select * 5 # 防止无限循环

            while len(selected) < num_to_select and attempts < max_attempts:
                 # 每次只选一个，避免一次选多个时概率分布变化导致的问题
                 chosen_index = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
                 if chosen_index not in selected_indices:
                      selected_indices.add(chosen_index)
                      selected.append(candidates[chosen_index])
                 attempts += 1

            # 如果尝试多次后仍未选够，补充出现次数最多的
            if len(selected) < num_to_select:
                 remaining_candidates = [c for i, c in enumerate(candidates) if i not in selected_indices]
                 remaining_candidates.sort(key=lambda x: x[2], reverse=True) # 按原始次数排序
                 needed = num_to_select - len(selected)
                 selected.extend(remaining_candidates[:needed])

        except Exception as e:
             logger.error(f"Error during weighted random choice for nicknames: {e}. Falling back to top N.", exc_info=True)
             # 出错时回退到选择次数最多的 N 个
             candidates.sort(key=lambda x: x[2], reverse=True)
             selected = candidates[:MAX_NICKNAMES_IN_PROMPT]


    # 格式化输出并按次数排序
    result = [(user, nick, count) for user, nick, count, _weight in selected]
    result.sort(key=lambda x: x[2], reverse=True) # 按次数降序

    logger.debug(f"Selected nicknames for prompt: {result}")
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
        return ""

    prompt_lines = ["以下是聊天记录中一些成员在本群的绰号信息（按常用度排序）："]
    grouped_by_user: Dict[str, List[str]] = {}

    for user_name, nickname, _count in selected_nicknames:
        if user_name not in grouped_by_user:
            grouped_by_user[user_name] = []
        # 添加引号以区分绰号
        grouped_by_user[user_name].append(f'“{nickname}”')

    for user_name, nicknames in grouped_by_user.items():
        nicknames_str = "、".join(nicknames)
        prompt_lines.append(f"{user_name}，在本群有时被称为：{nicknames_str}")

    return "\n".join(prompt_lines) + "\n" # 末尾加换行符

