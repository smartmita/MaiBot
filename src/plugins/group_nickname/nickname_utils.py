import random
from typing import List, Dict, Tuple, Any, Optional
from src.common.logger_manager import get_logger
from src.config.config import global_config


logger = get_logger("nickname_utils")


def select_nicknames_for_prompt(
    all_nicknames_info_by_actual_nickname: Dict[str, Dict[str, Any]] # 参数名修改以反映键的含义
) -> List[Tuple[str, str, str, int]]:
    """
    从给定的绰号信息中，根据映射次数加权随机选择最多 N 个绰号用于 Prompt。

    Args:
        all_nicknames_info_by_actual_nickname: 包含用户及其绰号和 UID 信息的字典，格式为
                        { "用户实际昵称1": {"user_id": "uid1", "nicknames": [{"绰号A": 次数}, ...]}, ... }
                        注意：这里的键是用户的实际昵称。

    Returns:
        List[Tuple[str, str, str, int]]: 选中的绰号列表，每个元素为 (用户实际昵称, user_id, 群内常用绰号, 次数)。
                                    按次数降序排序。
    """
    if not all_nicknames_info_by_actual_nickname:
        return []

    candidates = []  # 存储 (用户实际昵称, user_id, 群内常用绰号, 次数, 权重)
    smoothing_factor = global_config.group_nickname.nickname_probability_smoothing

    # 修改：迭代时，user_key 现在是用户的实际昵称
    for user_actual_nickname, data in all_nicknames_info_by_actual_nickname.items():
        user_id = data.get("user_id")
        nicknames_list = data.get("nicknames") # 这是群内常用绰号列表

        if not user_id or not isinstance(nicknames_list, list):
            logger.warning(f"用户实际昵称 '{user_actual_nickname}' 的数据格式无效或缺少 user_id/nicknames。已跳过。 Data: {data}")
            continue

        for nickname_entry in nicknames_list:
            if isinstance(nickname_entry, dict) and len(nickname_entry) == 1:
                group_nickname_str, count = list(nickname_entry.items())[0] # 这是群内常用绰号
                if isinstance(count, int) and count > 0 and isinstance(group_nickname_str, str) and group_nickname_str:
                    weight = count + smoothing_factor
                    # 修改：存储 user_actual_nickname
                    candidates.append((user_actual_nickname, user_id, group_nickname_str, count, weight))
                else:
                    logger.warning(
                        f"用户实际昵称 '{user_actual_nickname}' (UID: {user_id}) 的群内常用绰号条目无效: {nickname_entry}。已跳过。"
                    )
            else:
                logger.warning(f"用户实际昵称 '{user_actual_nickname}' (UID: {user_id}) 的群内常用绰号条目格式无效: {nickname_entry}。已跳过。")


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
    users_data_with_extras: List[Dict[str, Any]], # 修改：接收包含更丰富信息的用户数据列表
    selected_group_nicknames: Optional[List[Tuple[str, str, str, int]]] = None,
    is_group_chat: bool = False # 新增参数，判断是否为群聊
) -> str:
    """
    将用户基本信息、群名片、群头衔和可选的群内常用绰号信息格式化为注入 Prompt 的字符串。

    Args:
        users_data_with_extras: 用户信息列表，每个元素为字典，期望包含:
            "user_id": str,
            "actual_nickname": str,
            "group_cardname": Optional[str], (仅群聊相关)
            "group_titlename": Optional[str]  (仅群聊相关)
        selected_group_nicknames: 可选的已选群内常用绰号列表
                                  元组格式: (actual_nickname_key, user_id, group_nickname_str, count)。
        is_group_chat: bool, 指示当前是否为群聊上下文。

    Returns:
        str: 格式化后的字符串，如果列表为空则返回空字符串。
    """
    if not users_data_with_extras:
        return ""

    prompt_lines = ["以下是聊天记录中存在的对象的信息，与聊天记录中的 uid 一一映射，供你参考："]

    # 构建 user_id 到其群内常用绰号字符串列表的映射 (这部分逻辑不变)
    group_nicknames_map_by_uid: Dict[str, List[str]] = {}
    if selected_group_nicknames: # 只有提供了 selected_group_nicknames 时才处理
        for _actual_nick_key, u_id, group_nickname_str, _count in selected_group_nicknames:
            if u_id not in group_nicknames_map_by_uid:
                group_nicknames_map_by_uid[u_id] = []
            group_nicknames_map_by_uid[u_id].append(f"“{group_nickname_str}”")

    for user_data in users_data_with_extras:
        user_id = user_data.get("user_id")
        actual_nickname = user_data.get("actual_nickname")

        if not user_id or actual_nickname is None: # actual_nickname 可以是空字符串，但不应是None
            logger.warning(f"format_user_info_prompt 跳过无效用户数据: {user_data}")
            continue

        if user_id == global_config.bot.qq_account: # 对机器人本身的特殊处理
            line = f"uid:{user_id}，这是你，你的昵称为“{actual_nickname}”"
        else:
            line = f"uid:{user_id}，用户昵称为“{actual_nickname}”"

        # 处理群名片和群头衔 (仅群聊)
        if is_group_chat:
            group_cardname = user_data.get("group_cardname")
            group_titlename = user_data.get("group_titlename")

            if group_cardname: # 如果存在群名片
                line += f"，在本群的群名片为“{group_cardname}”"

            if group_titlename: # 如果存在群头衔
                line += f"，群特殊头衔为“{group_titlename}”"

        # 处理群内常用绰号 (原有逻辑，但现在依赖 is_group_chat 和 selected_group_nicknames)
        # 确保 selected_group_nicknames 只有在群聊且功能启用时才应被传入并处理
        if is_group_chat and selected_group_nicknames and user_id in group_nicknames_map_by_uid:
            group_nicknames_str_joined = "、".join(group_nicknames_map_by_uid[user_id])
            line += f"，ta 在本群常被称为：{group_nicknames_str_joined}"

        prompt_lines.append(line)

    if len(prompt_lines) > 1: # 确保除了标题行还有其他内容
        return "\n".join(prompt_lines) + "\n"
    else:
        return ""



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
