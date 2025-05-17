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

    candidates_with_nicknames = [] 
    users_without_nicknames_info = [] # 存储 (用户名, user_id, 群名称)

    smoothing_factor = global_config.group_nickname.nickname_probability_smoothing

    for user_name_key, data in all_nicknames_info_with_uid.items():
        user_id = data.get("user_id")
        group_card_name = data.get("group_card_name", "")
        nicknames_list = data.get("nicknames") # 这可能是一个空列表

        if not user_id: # user_id 是必须的
            logger.warning(f"用户 '{user_name_key}' 的数据缺少 user_id。已跳过。 Data: {data}")
            continue

        if nicknames_list and isinstance(nicknames_list, list): # 如果有绰号列表
            has_valid_nickname_for_this_user = False
            for nickname_entry in nicknames_list:
                if isinstance(nickname_entry, dict) and len(nickname_entry) == 1:
                    nickname, count = list(nickname_entry.items())[0]
                    if isinstance(count, int) and count > 0 and isinstance(nickname, str) and nickname:
                        weight = count + smoothing_factor
                        candidates_with_nicknames.append((user_name_key, user_id, group_card_name, nickname, count, weight))
                        has_valid_nickname_for_this_user = True
                    else:
                        logger.warning(f"用户 '{user_name_key}' (UID: {user_id}) 的绰号条目无效: {nickname_entry}。")
                else:
                    logger.warning(f"用户 '{user_name_key}' (UID: {user_id}) 的绰号条目格式无效: {nickname_entry}。")
            
            if not has_valid_nickname_for_this_user: # 有nicknames_list但里面没有效绰号
                 users_without_nicknames_info.append((user_name_key, user_id, group_card_name))
        else: # nicknames_list 为空或无效
            users_without_nicknames_info.append((user_name_key, user_id, group_card_name))


    selected_final_output = [] # 最终给 format_nickname_prompt_injection 的列表 (用户名, user_id, 群名称, 绰号, 次数)
    
    # 处理有绰号的用户
    if candidates_with_nicknames:
        max_nicknames_config = global_config.group_nickname.max_nicknames_in_prompt
        num_to_select_from_candidates = min(max_nicknames_config, len(candidates_with_nicknames))
        
        selected_candidates_with_weight = []
        try:
            selected_candidates_with_weight = weighted_sample_without_replacement(candidates_with_nicknames, num_to_select_from_candidates)
            if len(selected_candidates_with_weight) < num_to_select_from_candidates and candidates_with_nicknames:
                selected_tuples_for_set = set(
                    (c[0], c[1], c[2], c[3]) for c in selected_candidates_with_weight
                )
                remaining_candidates = [
                    c for c in candidates_with_nicknames
                    if (c[0], c[1], c[2], c[3]) not in selected_tuples_for_set
                ]
                remaining_candidates.sort(key=lambda x: x[4], reverse=True)
                needed = num_to_select_from_candidates - len(selected_candidates_with_weight)
                selected_candidates_with_weight.extend(remaining_candidates[:needed])
        except Exception as e:
            logger.error(f"绰号加权随机选择时出错: {e}。回退到选择次数最多的。", exc_info=True)
            temp_sorted_candidates = sorted(candidates_with_nicknames, key=lambda x: x[4], reverse=True)
            selected_candidates_with_weight = temp_sorted_candidates[:num_to_select_from_candidates]

        for user, uid, gcn, nick, count, _weight in selected_candidates_with_weight:
            selected_final_output.append((user, uid, gcn, nick, count))

    # 添加没有学习到绰号但仍在上下文中的用户信息
    # 我们希望总共输出的用户条目（不是绰号条目）不超过 max_nicknames_in_prompt
    # 先统计已选中有绰号的用户数量
    users_already_added_with_nicknames = {item[1] for item in selected_final_output} # set of user_ids
    
    remaining_slots_for_users = global_config.group_nickname.max_nicknames_in_prompt - len(users_already_added_with_nicknames)

    if remaining_slots_for_users > 0 and users_without_nicknames_info:
        # 随机选择一些没有绰号的用户来填充，或者按某种顺序
        # 这里简单地按列表顺序取，也可以打乱
        random.shuffle(users_without_nicknames_info) 
        for user_name, user_id, group_card_name in users_without_nicknames_info:
            if user_id not in users_already_added_with_nicknames and remaining_slots_for_users > 0:
                selected_final_output.append((user_name, user_id, group_card_name, "", 0)) # 绰号为空，次数为0
                users_already_added_with_nicknames.add(user_id)
                remaining_slots_for_users -= 1
            if remaining_slots_for_users <= 0:
                break
                
    # 按用户名（或某种一致的顺序）排序，使得输出更稳定，可选
    # selected_final_output.sort(key=lambda x: x[0]) 
    
    # 如果希望最终输出的绰号多的用户排前面，可以再按次数排一次
    # 但由于混合了无绰号用户，这个排序可能意义不大，除非你希望有绰号的绝对优先
    selected_final_output.sort(key=lambda x: x[4], reverse=True) # 按次数降序

    logger.debug(f"为 Prompt 选择的最终信息 (含无绰号用户): {selected_final_output}")
    return selected_final_output


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

        # 只有当 nickname 字段本身非空时，才将其视为一个有效的绰号进行添加
        if nickname and nickname.strip(): # 检查 nickname 是否有实际内容
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
        if details["nicknames"]: # 直接检查列表是否为空
            called_as_part = f"，ta 可能被群友称为：{nicknames_str}"
        
        line = f"- {user_name}({user_id})"
        if card_name_part:
            line += card_name_part
        if called_as_part: # 只有在确实有有效绰号时才添加
            line += called_as_part
        
        prompt_lines.append(line)

    if len(prompt_lines) > 1:
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