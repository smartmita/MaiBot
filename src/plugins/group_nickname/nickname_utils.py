import random
from typing import List, Dict, Tuple, Any
from src.common.logger_manager import get_logger
from src.config.config import global_config

# 这个文件现在只包含纯粹的工具函数，与状态和流程无关

logger = get_logger("nickname_utils")


def select_nicknames_for_prompt(
    all_nicknames_info_with_uid: Dict[str, Dict[str, Any]] # 修改输入类型提示
) -> List[Tuple[str, str, str, str, int]]: # 修改返回类型提示 (用户名, user_id, 绰号, 次数)
    """
    从给定的绰号信息中，根据映射次数加权随机选择最多 N 个绰号用于 Prompt。

    Args:
        all_nicknames_info_with_uid: 包含用户及其绰号、UID 和群名称信息的字典，格式为
                        { "用户名1": {"user_id": "uid1", "group_card_name": "群名片A", "nicknames": [{"绰号A": 次数}, ...]}, ... }
                        注意：这里的键是 person_name (用户名)。

    Returns:
        List[Tuple[str, str, str, str, int]]: 选中的绰号列表，每个元素为 (用户名, user_id, 群名称, 绰号, 次数)。
                                    按次数降序排序。
    """
    if not all_nicknames_info_with_uid:
        return []

    candidates = []  # 存储 (用户名, user_id, 绰号, 次数, 权重)
    smoothing_factor = global_config.group_nickname.nickname_probability_smoothing

    for user_name, data in all_nicknames_info_with_uid.items():
        user_id = data.get("user_id")
        group_card_name = data.get("group_card_name", "") # 获取群名称，如果不存在则为空
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
        selected_candidates_with_weight = weighted_sample_without_replacement(candidates, num_to_select)

        if len(selected_candidates_with_weight) < num_to_select and candidates: # 增加 candidates 非空检查
            logger.debug(
                f"加权随机选择后数量不足 ({len(selected_candidates_with_weight)}/{num_to_select})，尝试补充选择次数最多的。"
            )
            # 构建一个集合，包含已选中的 (用户名, user_id, 群名称, 绰号) 以便去重
            # selected_candidates_with_weight 中的元组现在是 (user_name, user_id, group_card_name, nickname, count, weight)
            selected_tuples_for_set = set(
                (c[0], c[1], c[2], c[3]) for c in selected_candidates_with_weight 
            )
            
            remaining_candidates = [
                c for c in candidates 
                if (c[0], c[1], c[2], c[3]) not in selected_tuples_for_set
            ]
            remaining_candidates.sort(key=lambda x: x[4], reverse=True)  # 按原始次数 (index 4 in 6-tuple) 排序
            needed = num_to_select - len(selected_candidates_with_weight)
            selected_candidates_with_weight.extend(remaining_candidates[:needed])

    except Exception as e:
        logger.error(f"绰号加权随机选择时出错: {e}。将回退到选择次数最多的 Top N。", exc_info=True)
        candidates.sort(key=lambda x: x[4], reverse=True) # 按原始次数 (index 4 in 6-tuple) 排序
        selected_candidates_with_weight = candidates[:num_to_select]

    
    result = [(user, uid, gcn, nick, count) for user, uid, gcn, nick, count, _weight in selected_candidates_with_weight]
    result.sort(key=lambda x: x[4], reverse=True) # 按次数 (index 4 in 5-tuple result) 降序排序

    logger.debug(f"为 Prompt 选择的绰号 (含UID和群名称): {result}")
    return result


def format_nickname_prompt_injection(selected_nicknames_with_info: List[Tuple[str, str, str, str, int]]) -> str: # 修改输入类型
    """
    将选中的绰号信息（含UID和群名称）格式化为注入 Prompt 的字符串。

    Args:
        selected_nicknames_with_info: 选中的绰号列表，每个元素为 (用户名, user_id, 群名称, 绰号, 次数)。

    Returns:
        str: 格式化后的字符串，如果列表为空则返回空字符串。
    """
    if not selected_nicknames_with_info:
        return ""

    prompt_lines = ["以下是聊天记录中一些成员的详细称呼信息（按常用度排序）与 uid 信息，供你参考："]
    # 改为: { (user_name, user_id): [绰号1, 绰号2] }
    grouped_by_user_details: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for user_name, user_id, group_card_name, nickname, _count in selected_nicknames_with_info:
        user_key = (user_name, user_id) # 使用 (用户名, 用户ID) 作为分组的键
        
        if user_key not in grouped_by_user_details:
            grouped_by_user_details[user_key] = {
                "group_card_name": group_card_name, # 存储该用户的群名称（对于同一个用户，群名称应该是一致的）
                "nicknames": [] # 初始化绰号列表
            }
        
        # 添加格式化的绰号到列表中，避免重复添加（尽管上游选择逻辑可能已去重）
        formatted_nickname = f"“{nickname}”"
        if formatted_nickname not in grouped_by_user_details[user_key]["nicknames"]:
            grouped_by_user_details[user_key]["nicknames"].append(formatted_nickname)

    # 构建最终的输出行
    for (user_name, user_id), details in grouped_by_user_details.items():
        nicknames_str = "、".join(details["nicknames"])
        
        # 构建群名称部分，仅当群名称非空时添加
        card_name_part = ""
        if details["group_card_name"] and details["group_card_name"].strip():
            card_name_part = f"，ta在这个群的群名称为“{details['group_card_name']}”"
        
        # 构建绰号部分，仅当存在绰号时添加
        called_as_part = ""
        if nicknames_str:
            called_as_part = f"，ta 可能被群友称为：{nicknames_str}"
        
        # 组合行：确保只添加有内容的部分
        # 格式：- 用户名(UID) [，群名称部分] [，绰号部分]
        line = f"- {user_name}({user_id})"
        if card_name_part:
            line += card_name_part
        if called_as_part: # 只有在确实有绰号时才添加“可能被群友称为”
            line += called_as_part
        elif not card_name_part and not called_as_part: # 如果既没群名片也没绰号，但用户在列表里，也显示基础信息
            pass # - 用户名(UID) 已经有了

        prompt_lines.append(line)

    if len(prompt_lines) > 1: # 如果不仅仅只有标题行
        return "\n".join(prompt_lines) + "\n"
    else:
        return "" # 如果没有处理任何用户信息，则返回空字符串


def weighted_sample_without_replacement( 
    candidates: List[Tuple[str, str, str, str, int, float]], k: int 
) -> List[Tuple[str, str, str, str, int, float]]:
    """
    执行不重复的加权随机抽样。
    Args:
        candidates: 候选列表，每个元素为 (用户名, user_id, 群名称, 绰号, 次数, 权重)。
        k: 需要选择的数量。
    Returns:
        List[Tuple[str, str, str, str, int, float]]: 选中的元素列表（包含权重）。
    """
    if k <= 0:
        return []
    n = len(candidates)
    if k >= n:
        return candidates[:]

    weighted_keys = []
    for i in range(n):
        weight = candidates[i][5] # 权重现在是第6个元素 (index 5)
        if weight <= 0:
            log_key = float("-inf")
            logger.warning(f"候选者 {candidates[i][:4]} 的权重为非正数 ({weight})，抽中概率极低。")
        else:
            log_u = -random.expovariate(1.0)
            log_key = log_u / weight
        weighted_keys.append((log_key, i))

    weighted_keys.sort(key=lambda x: x[0], reverse=True)
    selected_indices = [index for _log_key, index in weighted_keys[:k]]
    selected_items = [candidates[i] for i in selected_indices]

    return selected_items