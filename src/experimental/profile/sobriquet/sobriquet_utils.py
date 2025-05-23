import random
from typing import List, Dict, Tuple, Any, Optional
from src.common.logger_manager import get_logger
from src.config.config import global_config


logger = get_logger("sobriquet_utils") # 日志记录器名称更新


def select_sobriquets_for_prompt( # 函数名更新
    all_sobriquets_info_by_actual_name: Dict[str, Dict[str, Any]] # 参数名更新以反映键和值的含义
) -> List[Tuple[str, str, str, int]]:
    """
    从给定的绰号信息中，根据映射次数加权随机选择最多 N 个绰号用于 Prompt。

    Args:
        all_sobriquets_info_by_actual_name: 包含用户及其绰号和 UID 信息的字典，格式为
                        { "用户实际昵称1": {"user_id": "uid1", "nicknames": [{"绰号A": 次数}, ...]}, ... }
                        注意：这里的键是用户的实际昵称。 'nicknames' 仍为数据库中对应的key，代表群内常用绰号列表。

    Returns:
        List[Tuple[str, str, str, int]]: 选中的绰号列表，每个元素为 (用户实际昵称, user_id, 群内常用绰号, 次数)。
                                    按次数降序排序。
    """
    if not all_sobriquets_info_by_actual_name:
        return []

    candidates = []  # 存储 (用户实际昵称, user_id, 群内常用绰号, 次数, 权重)
    # global_config.group_nickname... 配置路径不变
    smoothing_factor = global_config.group_nickname.nickname_probability_smoothing

    for user_actual_name, data in all_sobriquets_info_by_actual_name.items(): # 变量名更新
        user_id = data.get("user_id")
        # 'nicknames' 是从数据库结构中来的key，代表群内常用绰号列表
        sobriquets_list = data.get("nicknames") # 变量名更新

        if not user_id or not isinstance(sobriquets_list, list):
            logger.warning(f"用户实际昵称 '{user_actual_name}' 的数据格式无效或缺少 user_id/nicknames。已跳过。 Data: {data}")
            continue

        for sobriquet_entry in sobriquets_list: # 变量名更新
            # sobriquet_entry 的结构是 {"绰号名": 次数}
            if isinstance(sobriquet_entry, dict) and len(sobriquet_entry) == 1:
                group_sobriquet_str, count = list(sobriquet_entry.items())[0] # 变量名更新
                if isinstance(count, int) and count > 0 and isinstance(group_sobriquet_str, str) and group_sobriquet_str:
                    weight = count + smoothing_factor
                    candidates.append((user_actual_name, user_id, group_sobriquet_str, count, weight)) # 变量名更新
                else:
                    logger.warning(
                        f"用户实际昵称 '{user_actual_name}' (UID: {user_id}) 的群内常用绰号条目无效: {sobriquet_entry}。已跳过。"
                    )
            else:
                logger.warning(f"用户实际昵称 '{user_actual_name}' (UID: {user_id}) 的群内常用绰号条目格式无效: {sobriquet_entry}。已跳过。")


    if not candidates:
        return []

    # global_config.group_nickname... 配置路径不变
    max_sobriquets = global_config.group_nickname.max_nicknames_in_prompt # 变量名更新
    if max_sobriquets == 0:
        logger.warning("max_nicknames_in_prompt 配置为0，不选择任何绰号。")
        return []

    num_to_select = min(max_sobriquets, len(candidates))

    try:
        selected_candidates_with_weight = weighted_sample_without_replacement(candidates, num_to_select)

        if len(selected_candidates_with_weight) < num_to_select:
            logger.debug(
                f"加权随机选择后数量不足 ({len(selected_candidates_with_weight)}/{num_to_select})，尝试补充选择次数最多的。"
            )
            selected_ids = set(
                (c[0], c[1], c[2]) for c in selected_candidates_with_weight
            )
            remaining_candidates = [c for c in candidates if (c[0], c[1], c[2]) not in selected_ids]
            remaining_candidates.sort(key=lambda x: x[3], reverse=True)
            needed = num_to_select - len(selected_candidates_with_weight)
            selected_candidates_with_weight.extend(remaining_candidates[:needed])

    except Exception as e:
        logger.error(f"绰号加权随机选择时出错: {e}。将回退到选择次数最多的 Top N。", exc_info=True)
        candidates.sort(key=lambda x: x[3], reverse=True)
        selected_candidates_with_weight = candidates[:num_to_select]

    result = [(user, uid, sobriquet, count) for user, uid, sobriquet, count, _weight in selected_candidates_with_weight] # 变量名更新

    result.sort(key=lambda x: x[3], reverse=True)

    logger.debug(f"为 Prompt 选择的绰号 (含UID): {result}")
    return result


def weighted_sample_without_replacement(
    candidates: List[Tuple[Any, ...]], # 保持通用性，因为此函数可能被其他地方使用
    k: int
) -> List[Tuple[Any, ...]]:
    """
    执行不重复的加权随机抽样。使用 A-ExpJ 算法思想的简化实现。
    假设候选列表的最后一个元素是权重。
    Args:
        candidates: 候选列表，每个元组的最后一个元素应为权重。
                    例如: (item_data1, item_data2, ..., weight)。
        k: 需要选择的数量。

    Returns:
        List[Tuple[Any, ...]]: 选中的元素列表（包含权重）。
    """
    if k <= 0:
        return []
    n = len(candidates)
    if k >= n:
        return candidates[:]

    weighted_keys = []
    for i in range(n):
        candidate_item = candidates[i]
        weight = candidate_item[-1] # 假设权重是元组的最后一个元素
        
        if not isinstance(weight, (int, float)):
            logger.error(f"加权抽样时候选者 {candidate_item} 的权重格式不正确: {weight}。跳过此候选者。")
            continue

        if weight <= 0:
            log_key = float("-inf") 
            logger.warning(f"候选者 {str(candidate_item[:-1])} 的权重为非正数 ({weight})，抽中概率极低。")
        else:
            log_u = -random.expovariate(1.0)
            log_key = log_u / weight
        weighted_keys.append((log_key, i))

    weighted_keys.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = [index for _log_key, index in weighted_keys[:k]]
    
    selected_items = [candidates[i] for i in selected_indices]

    return selected_items

# 原 format_user_info_prompt 函数已移至 profile_utils.py 并重构为 format_profile_prompt_injection