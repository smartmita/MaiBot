import random
from typing import List, Dict, Tuple, Any, Optional
from src.common.logger_manager import get_logger
from src.config.config import global_config


logger = get_logger("nickname_utils")


def select_nicknames_for_prompt(
    all_nicknames_info_with_uid: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, str, str, int]]:
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
    if max_nicknames == 0: # 处理 max_nicknames_in_prompt 配置为0的边界情况
        logger.warning("max_nicknames_in_prompt 配置为0，不选择任何绰号。")
        return []

    num_to_select = min(max_nicknames, len(candidates))

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


def format_user_info_prompt(
    users_data: List[Tuple[str, str]], # 改为接收 (user_id, person_name) 元组列表
    selected_nicknames: Optional[List[Tuple[str, str, str, int]]] = None # (person_name, user_id, nickname, count)
) -> str:
    """
    将用户基本信息和可选的绰号信息格式化为注入 Prompt 的字符串。

    Args:
        users_data: 用户信息列表，每个元素为 (user_id, person_name)。
        selected_nicknames: 可选的已选绰号列表 (person_name, user_id, 绰号, 次数)。
                            此列表中的绰号应已按常用度排序。

    Returns:
        str: 格式化后的字符串，如果列表为空则返回空字符串。
    """
    if not users_data:
        return ""

    prompt_lines = ["以下是聊天记录中存在的对象的信息，与聊天记录中的 uid 一一映射，供你参考："]
    
    nicknames_map: Dict[str, List[str]] = {} # Key: user_id, Value: List of formatted nickname strings ("“nickname”")
    if selected_nicknames: # 仅当提供了绰号信息时才构建映射
        for _p_name, u_id, nickname, _count in selected_nicknames: # 迭代已按全局频率排序的绰号
            if u_id not in nicknames_map:
                nicknames_map[u_id] = []
            nicknames_map[u_id].append(f"“{nickname}”")
        # nicknames_map[u_id] 中的绰号列表将自然地按其在 selected_nicknames 中的顺序排列（即按常用度）

    for user_id, person_name in users_data:
        line = f"uid:{user_id}，名为“{person_name}“"
        if selected_nicknames and user_id in nicknames_map: # 如果有绰号信息且当前用户有绰号
            nicknames_str = "、".join(nicknames_map[user_id])
            line += f"，ta 在本群常被称为：{nicknames_str}"
        prompt_lines.append(line)

    if len(prompt_lines) > 1: # 确保除了标题行还有其他内容
        return "\n".join(prompt_lines) + "\n"
    else:
        return "" # 如果只有标题行（例如 users_data 为空，虽然前面有检查），则返回空


def weighted_sample_without_replacement(
    candidates: List[Tuple[str, str, str, int, float]], k: int
) -> List[Tuple[str, str, str, int, float]]:
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
        return candidates[:] # 如果 k 大于等于候选数量，直接返回所有候选者

    weighted_keys = []
    for i in range(n):
        # username, user_id, nickname, count, weight = candidates[i] # 解包以提高可读性
        _user_name, _user_id, _nickname, _count, weight = candidates[i] # 使用下划线表示暂不使用的变量
        if weight <= 0:
            # 如果权重为0或负数，赋予一个极小的对数键值，使其几乎不可能被选中
            # 但仍需避免 log(0) 或除以0的错误
            log_key = float("-inf") 
            logger.warning(f"候选者 {candidates[i][:3]} 的权重为非正数 ({weight})，抽中概率极低。")
        else:
            # random.expovariate(lambd) 返回一个服从指数分布的随机数，其均值为 1/lambd
            # 这里 lambd=1.0，所以均值为1.0
            log_u = -random.expovariate(1.0) # 生成标准指数分布的负值，模拟优先级
            log_key = log_u / weight # 权重越大，log_key 越接近0（因为 log_u 是负数）
        weighted_keys.append((log_key, i)) # 存储 (计算出的键, 原始索引)

    # 按计算出的键值降序排序，键值越大（越接近0）的优先级越高
    weighted_keys.sort(key=lambda x: x[0], reverse=True)
    
    # 选择前 k 个元素的原始索引
    selected_indices = [index for _log_key, index in weighted_keys[:k]]
    
    # 根据选中的索引获取原始候选项目
    selected_items = [candidates[i] for i in selected_indices]

    return selected_items

# 原 format_nickname_prompt_injection 函数已被新的 format_user_info_prompt 函数取代，
# 因为新的函数能够处理两种情况（仅用户信息，或用户信息+绰号）。
# 如果旧函数没有其他地方调用，可以安全地认为它已被新逻辑覆盖。
