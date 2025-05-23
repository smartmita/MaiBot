from typing import List, Dict, Tuple, Any, Optional
from src.common.logger_manager import get_logger
from src.config.config import global_config

logger = get_logger("profile_utils")

def format_profile_prompt_injection(
    users_profile_data: List[Dict[str, Any]],
    selected_group_sobriquets: Optional[List[Tuple[str, str, str, int]]] = None,
    is_group_chat: bool = False
) -> str:
    """
    将用户画像信息（基本信息、性别、群名片、群头衔和可选的群内常用绰号）格式化为注入 Prompt 的字符串。

    Args:
        users_profile_data: 用户画像信息列表，每个元素为字典，期望包含:
            "user_id": str,
            "actual_name": str,      // 用户在平台的实际名称/昵称
            "gender_mark": str,      // 用户性别标记
            "group_cardname": Optional[str], // 用户在本群的群名片 (仅群聊)
            "group_titlename": Optional[str] // 用户在本群的群头衔 (仅群聊)
        selected_group_sobriquets: 可选的已选群内常用绰号列表。
                                   元组格式: (actual_name_key, user_id, group_sobriquet_str, count)。
                                   actual_name_key 是获取绰号时使用的用户实际名称。
        is_group_chat: bool, 指示当前是否为群聊上下文。

    Returns:
        str: 格式化后的字符串，如果列表为空则返回空字符串。
    """
    if not users_profile_data:
        return ""

    prompt_lines = ["以下是聊天记录中存在的对象的信息，与聊天记录中的 uid 一一映射，供你参考："]

    # 构建 user_id 到其群内常用绰号字符串列表的映射
    group_sobriquets_map_by_uid: Dict[str, List[str]] = {}
    if is_group_chat and selected_group_sobriquets:
        for _actual_name_key, u_id, group_sobriquet_str, _count in selected_group_sobriquets:
            if u_id not in group_sobriquets_map_by_uid:
                group_sobriquets_map_by_uid[u_id] = []
            group_sobriquets_map_by_uid[u_id].append(f"“{group_sobriquet_str}”")

    for user_data in users_profile_data:
        user_id = user_data.get("user_id")
        actual_name = user_data.get("actual_name") # 字段名更新为 actual_name
        gender_mark = user_data.get("gender_mark", "未知")

        if not user_id or actual_name is None: # actual_name 检查
            logger.warning(f"format_profile_prompt_injection 跳过无效用户数据 (缺少 user_id 或 actual_name): {user_data}")
            continue
        
        # bot.qq_account 和 bot.nickname 配置键名不变
        # identity.gender 配置键名不变
        if user_id == global_config.bot.qq_account:
            bot_gender_from_config = global_config.identity.gender
            line = f"uid:{user_id}，这是你，你的昵称为“{actual_name}”，性别为“{bot_gender_from_config}”"
        else:
            line = f"uid:{user_id}，用户昵称为“{actual_name}”，性别为“{gender_mark}”"

        if is_group_chat:
            group_cardname = user_data.get("group_cardname")
            group_titlename = user_data.get("group_titlename")
            if group_cardname and group_cardname != actual_name: # 仅当群名片和昵称不同时显示，避免冗余
                line += f"，在本群的群名片为“{group_cardname}”"
            if group_titlename:
                line += f"，群特殊头衔为“{group_titlename}”"

            if user_id in group_sobriquets_map_by_uid:
                sobriquets_str_joined = "、".join(group_sobriquets_map_by_uid[user_id])
                line += f"，ta 在本群常被称为：{sobriquets_str_joined}"
        
        prompt_lines.append(line)

    if len(prompt_lines) > 1:
        return "\n".join(prompt_lines) + "\n"
    else:
        return ""