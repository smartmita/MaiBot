import random
from typing import List, Dict, Tuple, Any, Optional
from src.common.logger_manager import get_logger
from src.config.config import global_config # 假设 global_config 已能正确加载配置

logger = get_logger("sobriquet_utils") # 获取日志记录器实例

def select_sobriquets_for_prompt( 
    all_sobriquets_info_by_actual_name: Dict[str, Dict[str, Any]] 
) -> List[Tuple[str, str, str, float]]: # 返回类型中的 int 改为 float
    """
    从给定的绰号信息中，根据映射强度加权随机选择最多 N 个绰号用于 Prompt。
    现在使用 'strength' (float) 而不是 'count' (int)。

    Args:
        all_sobriquets_info_by_actual_name: 包含用户及其绰号和 UID 信息的字典。
            结构示例: { 
                "用户实际昵称1": {
                    "user_id": "uid1", 
                    "nicknames": [{"绰号A": strength1_float}, {"绰号B": strength2_float}, ...] 
                }, ... 
            }
            注意：'nicknames' 键对应的值是从数据库中来的群内常用绰号列表及其强度。

    Returns:
        选中的绰号列表，每个元素为 (用户实际昵称, user_id, 群内常用绰号, 强度_float)。
        列表会按强度降序排序。
    """
    if not all_sobriquets_info_by_actual_name: # 如果输入为空，直接返回空列表
        return []

    # 候选绰号及其相关信息：(用户实际昵称, user_id, 群内常用绰号, 强度_float, 权重_float)
    candidates: List[Tuple[str, str, str, float, float]] = []  
    
    # 从全局配置中获取平滑因子，用于调整选择概率，避免强度低的绰号完全没有机会被选中
    smoothing_factor = float(global_config.profile.sobriquet_probability_smoothing) # 确保是 float

    # 遍历每个用户（以其实际昵称为键）的绰号信息
    for user_actual_name, data in all_sobriquets_info_by_actual_name.items():
        user_id = data.get("user_id") # 获取用户ID
        sobriquets_list = data.get("nicknames") # 获取该用户的绰号列表（原始数据库结构）

        # 数据有效性检查
        if not user_id or not isinstance(sobriquets_list, list):
            logger.warning(f"用户实际昵称 '{user_actual_name}' 的数据格式无效或缺少 user_id/nicknames。已跳过。数据: {data}")
            continue

        # 遍历该用户的每个绰号条目
        for sobriquet_entry in sobriquets_list:
            # 绰号条目通常是 {"绰号名": 强度_float} 格式的字典
            if isinstance(sobriquet_entry, dict) and len(sobriquet_entry) == 1:
                group_sobriquet_str, strength_val = list(sobriquet_entry.items())[0] # 提取绰号名和强度
                
                # 进一步校验绰号名和强度的有效性
                if isinstance(strength_val, (int, float)) and float(strength_val) > 0.0 and \
                   isinstance(group_sobriquet_str, str) and group_sobriquet_str:
                    
                    current_strength = float(strength_val)
                    weight = current_strength + smoothing_factor # 计算权重（强度 + 平滑因子）
                    candidates.append((user_actual_name, user_id, group_sobriquet_str, current_strength, weight))
                else:
                    logger.warning(
                        f"用户实际昵称 '{user_actual_name}' (UID: {user_id}) 的群内常用绰号条目强度无效: {sobriquet_entry}。已跳过。"
                    )
            else:
                logger.warning(f"用户实际昵称 '{user_actual_name}' (UID: {user_id}) 的群内常用绰号条目格式无效: {sobriquet_entry}。已跳过。")

    if not candidates: # 如果没有有效的候选绰号，返回空列表
        return []

    # 从配置中获取在Prompt中允许的最大绰号数量
    max_sobriquets_in_prompt = global_config.profile.max_sobriquets_in_prompt
    if max_sobriquets_in_prompt == 0: # 如果配置为0，则不选择任何绰号
        logger.warning("配置项 max_sobriquets_in_prompt 为0，不选择任何绰号。")
        return []

    # 确定实际需要选择的绰号数量（不超过候选总数和配置上限）
    num_to_select = min(max_sobriquets_in_prompt, len(candidates))

    selected_candidates_with_weight: List[Tuple[str, str, str, float, float]]
    try:
        # 执行加权随机抽样（不重复）
        selected_candidates_with_weight = weighted_sample_without_replacement(candidates, num_to_select)

        # 如果加权抽样选出的数量不足，尝试用强度最高者补充
        if len(selected_candidates_with_weight) < num_to_select:
            logger.debug(
                f"加权随机选择后数量不足 ({len(selected_candidates_with_weight)}/{num_to_select})，尝试补充选择强度最高的。"
            )
            # 找出已选中的，避免重复
            selected_ids_set = set(
                (c[0], c[1], c[2]) for c in selected_candidates_with_weight # (实际昵称, uid, 绰号) 作为唯一标识
            )
            # 筛选出未被选中的候选者
            remaining_candidates = [c for c in candidates if (c[0], c[1], c[2]) not in selected_ids_set]
            remaining_candidates.sort(key=lambda x: x[3], reverse=True) # 按原始强度降序排序
            needed_to_fill = num_to_select - len(selected_candidates_with_weight) # 计算还需补充的数量
            selected_candidates_with_weight.extend(remaining_candidates[:needed_to_fill]) # 添加补充

    except Exception as e: # 如果抽样过程中发生错误
        logger.error(f"绰号加权随机选择时出错: {e}。将回退到选择强度最高的 Top N。", exc_info=True)
        candidates.sort(key=lambda x: x[3], reverse=True) # 按原始强度降序排序
        selected_candidates_with_weight = candidates[:num_to_select] # 直接取强度最高的N个

    # 格式化输出结果，移除权重信息，只保留 (用户实际昵称, user_id, 绰号, 强度_float)
    result = [(user, uid, sobriquet, strength) for user, uid, sobriquet, strength, _weight in selected_candidates_with_weight]

    result.sort(key=lambda x: x[3], reverse=True) # 最终结果按强度降序排序

    logger.debug(f"为 Prompt 选择的绰号 (含UID和强度): {result}")
    return result


def weighted_sample_without_replacement(
    candidates: List[Tuple[Any, ...]], 
    k: int
) -> List[Tuple[Any, ...]]:
    """
    执行不重复的加权随机抽样。使用 A-ExpJ 算法思想的简化实现。
    假设候选列表的每个元组的最后一个元素是其权重。

    Args:
        candidates: 候选列表，每个元组的格式应为 (数据1, 数据2, ..., 权重)。
        k: 需要选择的数量。

    Returns:
        选中的元素列表（保持原始元组结构，包含权重）。
    """
    if k <= 0: # 如果选择数量无效，返回空
        return []
    n = len(candidates) # 候选者总数
    if k >= n: # 如果要选择的数量大于等于总数，则返回所有候选者
        return candidates[:] 

    weighted_keys: List[Tuple[float, int]] = [] # 存储 (计算出的排序键, 原始索引)
    for i in range(n):
        candidate_item = candidates[i]
        try:
            weight = float(candidate_item[-1]) # 假设权重是元组的最后一个元素，并尝试转为浮点数
        except (TypeError, ValueError, IndexError): # 处理权重无效或不存在的情况
            logger.error(f"加权抽样时，候选者 {candidate_item} 的权重格式不正确或缺失。跳过此候选者。")
            continue

        if weight <= 0: # 如果权重非正，给予极低的选中概率
            log_key = float("-inf") # 负无穷使得其排序时几乎总是在最后
            logger.warning(f"候选者 {str(candidate_item[:-1])} 的权重为非正数 ({weight})，抽中概率极低。")
        else:
            # A-ExpJ 算法核心思想：为每个项目生成一个 key = U^(1/w)，其中 U 是 (0,1) 上的随机数，w 是权重。
            # 或者等效地，key = exp(log(U)/w)。由于 log(U) 通常是负数，所以权重越大，key 越接近1（或0，取决于实现）。
            # 这里使用 expovariate(1.0) 生成一个均值为1的指数分布随机数 R，然后 key = -R/w。
            # -R 是负数，所以权重 w 越大，key 越接近0（即越大）。
            log_u = -random.expovariate(1.0) # 生成标准指数分布的负值，模拟优先级
            log_key = log_u / weight # 权重越大，log_key 越接近0（因为 log_u 是负数）
        weighted_keys.append((log_key, i)) # 存储计算出的键和原始索引

    # 按计算出的键值降序排序，键值越大（越接近0）的优先级越高
    weighted_keys.sort(key=lambda x: x[0], reverse=True)
    
    # 选择排序后的前 k 个元素的原始索引
    selected_indices = [index for _log_key, index in weighted_keys[:k]]
    
    # 根据选中的索引获取原始候选项目
    selected_items = [candidates[i] for i in selected_indices]

    return selected_items
