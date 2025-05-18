import random
from typing import List, Dict, Tuple, Any
from src.common.logger_manager import get_logger
from src.config.config import global_config

# 这个文件现在只包含纯粹的工具函数，与状态和流程无关

logger = get_logger("nickname_utils")


def select_nicknames_for_prompt(
    all_nicknames_info_with_uid: Dict[str, Dict[str, Any]] # 修改输入类型提示
) -> List[Tuple[str, str, str, int]]: # 修改返回类型提示 (用户名, user_id, 绰号, 次数)
    """
    从给定的绰号信息中，根据映射次数加权随机选择最多 N 个绰号用于 Prompt。

    Args:
        all_nicknames_info_with_uid: 包含用户及其绰号和 UID 信息的字典，格式为
                        { "用户名1": {"user_id": "uid1", "nicknames": [{"绰号A": 次数}, {"绰号B": 次数}]}, ... }
                        注意：这里的键是 person_name。

    Returns:
        List[Tuple[str, str, str, int]]: 选中的绰号列表，每个元素为 (用户名, user_id, 绰号, 次数)。
                                    按次数降序排序。
    """
    if not all_nicknames_info_with_uid:
        return []

    candidates = []  # 存储 (用户名, user_id, 绰号, 次数, 权重)
    smoothing_factor = global_config.group_nickname.nickname_probability_smoothing

    for user_name, data in all_nicknames_info_with_uid.items():
        user_id = data.get("user_id")
        nicknames_list = data.get("nicknames")

        if not user_id or not isinstance(nicknames_list, list):
            logger.warning(f"用户 '{user_name}' 的数据格式无效或缺少 user_id/nicknames。已跳过。 Data: {data}")
            continue
            
        for nickname_entry in nicknames_list:
            if isinstance(nickname_entry, dict) and len(nickname_entry) == 1:
                nickname, count = list(nickname_entry.items())[0]
                if isinstance(count, int) and count > 0 and isinstance(nickname, str) and nickname:
                    weight = count + smoothing_factor
                    candidates.append((user_name, user_id, nickname, count, weight)) # 添加 user_id
                else:
                    logger.warning(
                        f"用户 '{user_name}' (UID: {user_id}) 的绰号条目无效: {nickname_entry} (次数非正整数或绰号为空)。已跳过。"
                    )
            else:
                logger.warning(f"用户 '{user_name}' (UID: {user_id}) 的绰号条目格式无效: {nickname_entry}。已跳过。")

    if not candidates:
        return []

    max_nicknames = global_config.group_nickname.max_nicknames_in_prompt
    num_to_select = min(max_nicknames, len(candidates))

    try:
        selected_candidates_with_weight = weighted_sample_without_replacement(candidates, num_to_select) # 使用新的辅助函数名（如果修改了它）

        if len(selected_candidates_with_weight) < num_to_select:
            logger.debug(
                f"加权随机选择后数量不足 ({len(selected_candidates_with_weight)}/{num_to_select})，尝试补充选择次数最多的。"
            )
            selected_ids = set(
                (c[0], c[1], c[2]) for c in selected_candidates_with_weight # (user_name, user_id, nickname)
            )
            remaining_candidates = [c for c in candidates if (c[0], c[1], c[2]) not in selected_ids]
            remaining_candidates.sort(key=lambda x: x[3], reverse=True)  # 按原始次数 (index 3) 排序
            needed = num_to_select - len(selected_candidates_with_weight)
            selected_candidates_with_weight.extend(remaining_candidates[:needed])

    except Exception as e:
        logger.error(f"绰号加权随机选择时出错: {e}。将回退到选择次数最多的 Top N。", exc_info=True)
        candidates.sort(key=lambda x: x[3], reverse=True) # 按原始次数 (index 3) 排序
        selected_candidates_with_weight = candidates[:num_to_select]

    # 格式化输出结果为 (用户名, user_id, 绰号, 次数)，移除权重
    result = [(user, uid, nick, count) for user, uid, nick, count, _weight in selected_candidates_with_weight]

    result.sort(key=lambda x: x[3], reverse=True) # 按次数 (index 3) 降序排序

    logger.debug(f"为 Prompt 选择的绰号 (含UID): {result}")
    return result


def format_nickname_prompt_injection(selected_nicknames_with_uid: List[Tuple[str, str, str, int]]) -> str: # 修改输入类型
    """
    将选中的绰号信息（含UID）格式化为注入 Prompt 的字符串。

    Args:
        selected_nicknames_with_uid: 选中的绰号列表 (用户名, user_id, 绰号, 次数)。

    Returns:
        str: 格式化后的字符串，如果列表为空则返回空字符串。
    """
    if not selected_nicknames_with_uid:
        return ""

    prompt_lines = ["以下是聊天记录中一些成员在本群的绰号信息（按常用度排序）与 uid 信息，供你参考："]
    # 改为: { (user_name, user_id): [绰号1, 绰号2] }
    grouped_by_user: Dict[Tuple[str, str], List[str]] = {}

    for user_name, user_id, nickname, _count in selected_nicknames_with_uid: # 解包时加入 user_id
        user_key = (user_name, user_id) # 使用 (user_name, user_id) 作为键
        if user_key not in grouped_by_user:
            grouped_by_user[user_key] = []
        grouped_by_user[user_key].append(f"“{nickname}”")

    for (user_name, user_id), nicknames_list in grouped_by_user.items():
        nicknames_str = "、".join(nicknames_list)
        # 格式化输出，例如: "- 张三(12345)，ta 可能被称为：“三儿”、“张哥”"
        prompt_lines.append(f"- {user_name}({user_id})，ta 可能被称为：{nicknames_str}")

    if len(prompt_lines) > 1:
        return "\n".join(prompt_lines) + "\n"
    else:
        return ""


def weighted_sample_without_replacement( # 函数名保持不变，但内部处理的元组结构变了
    candidates: List[Tuple[str, str, str, int, float]], k: int # 修改输入类型 (用户名, user_id, 绰号, 次数, 权重)
) -> List[Tuple[str, str, str, int, float]]: # 修改返回类型
    """
    执行不重复的加权随机抽样。使用 A-ExpJ 算法思想的简化实现。

    Args:
        candidates: 候选列表，每个元素为 (用户名, user_id, 绰号, 次数, 权重)。
        k: 需要选择的数量。

    Returns:
        List[Tuple[str, str, str, int, float]]: 选中的元素列表（包含权重）。
    """
    if k <= 0:
        return []
    n = len(candidates)
    if k >= n:
        return candidates[:]

    weighted_keys = []
    for i in range(n):
        weight = candidates[i][4] # 权重现在是第5个元素 (index 4)
        if weight <= 0:
            log_key = float("-inf")
            logger.warning(f"候选者 {candidates[i][:3]} 的权重为非正数 ({weight})，抽中概率极低。") # 日志中多显示一个元素
        else:
            log_u = -random.expovariate(1.0)
            log_key = log_u / weight
        weighted_keys.append((log_key, i))

    weighted_keys.sort(key=lambda x: x[0], reverse=True)
    selected_indices = [index for _log_key, index in weighted_keys[:k]]
    selected_items = [candidates[i] for i in selected_indices]

    return selected_items