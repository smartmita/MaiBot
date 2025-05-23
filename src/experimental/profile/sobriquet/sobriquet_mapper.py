from typing import Dict, List, Any, Optional
from src.common.logger_manager import get_logger

logger = get_logger("sobriquet_mapper") # 日志记录器

def format_existing_sobriquets_for_prompt(existing_sobriquets: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    将已存在的用户绰号数据格式化为字符串，以便注入Prompt。
    使用 'strength' (强度)。

    Args:
        existing_sobriquets: 一个字典，键是用户ID（字符串），值是该用户已有的绰号列表 
                             (每个绰号是一个字典，如{"name": "昵称", "strength": 10.5})。
    Returns:
        格式化后的字符串，用于Prompt。
    """
    if not existing_sobriquets:
        return "当前上下文中用户无已记录的常用绰号。"
    
    lines = ["当前上下文中用户已记录的常用绰号（供你参考，帮助判断新提及的是否为已知绰号或新绰号，以及哪些可能是旧的或不再使用的）："]
    for user_id, sobriquets in existing_sobriquets.items():
        if sobriquets:
            valid_sobriquet_strs = []
            for s_entry in sobriquets:
                if isinstance(s_entry, dict) and "name" in s_entry and "strength" in s_entry:
                    strength_val = float(s_entry.get('strength', 0.0))
                    valid_sobriquet_strs.append(f"“{s_entry.get('name')}”(强度: {strength_val:.1f})")
            
            if valid_sobriquet_strs:
                lines.append(f"用户ID {user_id}: {', '.join(valid_sobriquet_strs)}")
            else:
                lines.append(f"用户ID {user_id}: 无有效已记录绰号")
        else:
            lines.append(f"用户ID {user_id}: 无已记录绰号")
    return "\n".join(lines)

def build_sobriquet_mapping_prompt(
    chat_history_str: str, 
    bot_reply: str,
    existing_sobriquets_str: Optional[str] = None
) -> str:
    """
    构建用于 LLM 进行绰号映射分析的 Prompt。
    要求LLM同时输出可靠和“不可信/应否定”的映射。

    Args:
        chat_history_str: 格式化后的聊天历史记录字符串。
        bot_reply: Bot 的最新回复字符串。
        existing_sobriquets_str: 可选的，当前上下文中用户已存在的绰号信息字符串 (包含强度)。

    Returns:
        str: 构建好的 Prompt 字符串。
    """
    reference_info_section = ""
    if existing_sobriquets_str:
        reference_info_section = f"\n\n已知信息参考（上下文中用户的既有绰号记录，包含当前强度）：\n---\n{existing_sobriquets_str}\n---\n"

    prompt = f"""
任务：仔细分析以下“聊天记录”、“你的最新回复”以及可选的“已知信息参考”，识别用户ID与绰号之间的映射关系。你需要区分哪些映射是基于当前聊天内容清晰、明确、可靠的，以及哪些是**必定不可信或应明确否定**的。

聊天记录：
---
{chat_history_str}
---

你的最新回复：
{bot_reply}
{reference_info_section}
分析要求与输出格式：
1.  **可靠映射 (reliable_mappings)**：
    * 找出聊天记录和“你的最新回复”中，那些在当前上下文中**非常清晰、无歧义地指向特定用户ID**的绰号。
    * 这些是你高度确信的映射关系。如果“已知信息参考”中已存在此绰号，当前聊天内容应能进一步确认其适用性。
2.  **不可信或应否定的映射 (unreliable_mappings)**：
    * 此类别用于那些你**有较高把握认定为错误、不适用、或已被用户明确否认**的绰号与用户的关联。
    * **主要场景包括**：
        * 用户在聊天中**明确表示不再使用某个绰号**，或否认某个称呼是自己的。例如：“别叫我‘二狗’了，我现在是‘啸天’”。此时，“二狗”对于该用户就属于此类。
        * 上下文强烈暗示某个称呼是**误用、玩笑性质且非公认绰号，或明显不适用于目标用户**。
        * 根据“已知信息参考”，一个历史低强度绰号与当前高强度常用绰号冲突，且无任何迹象表明该历史绰号被重新启用或依然有效。
    * **目标**：识别出那些**不应被视为该用户有效绰号**的称呼。这与“可靠映射”中可能存在的轻微不确定性不同，这里指的是你有理由相信这个映射是**错误或无效**的。
3.  通用规则：
    * **不要**输出你自己（名称通常后带"(你)"的用户）的任何映射。
    * **不要**输出与用户已知平台昵称（通常在聊天记录中已明确标出，如 "张三(12345)" 中的 "张三"）完全相同的词语作为绰号。
    * **不要**将在“你的最新回复”中你对他人使用的称呼或绰号进行映射（只分析聊天记录中他人对用户的称呼）。
    * **不要**输出指代不明或过于通用的词语（如“大佬”、“兄弟”、“那个谁”等，除非上下文能非常明确地指向特定用户并满足可靠性要求）。
4.  输出JSON对象格式如下：
    ```json
    {{
        "reliable_mappings": {{
            "用户A数字id": "可靠绰号_A1",
            "用户B数字id": "可靠绰号_B1"
        }},
        "unreliable_mappings": {{
            "用户C数字id": "不可信或应否定绰号_C1",
            "用户D数字id": "不可信或应否定绰号_D1"
        }}
    }}
    ```
    * `"reliable_mappings"` 和 `"unreliable_mappings"` 字段的键都必须是用户的**数字 ID (字符串形式)**，值是对应的**绰号 (字符串形式)**。
    * 如果某类映射不存在，则对应的字段值必须是一个空对象 `{{}}`。
    * 对于 `reliable_mappings`，只包含你能**高度确认**映射关系的条目。
    * 对于 `unreliable_mappings`，包含那些你确信**不正确或应被明确否定**的映射。
5.  请**仅**输出上述格式的 JSON 对象，不要包含任何额外的解释、注释或代码块标记之外的文本。

输出：
"""
    return prompt
