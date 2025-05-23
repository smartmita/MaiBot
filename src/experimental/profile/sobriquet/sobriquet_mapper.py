from typing import Dict, List, Any, Optional
from src.common.logger_manager import get_logger

logger = get_logger("sobriquet_mapper") # 日志记录器

def format_existing_sobriquets_for_prompt(existing_sobriquets: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    将已存在的用户绰号数据格式化为字符串，以便注入Prompt。

    Args:
        existing_sobriquets: 一个字典，键是用户ID（字符串），值是该用户已有的绰号列表 
                             (每个绰号是一个字典，如{"name": "昵称", "count": 10})。
                             示例: {"123": [{"name": "小明", "count": 20}, {"name": "明仔", "count": 5}]}
    Returns:
        格式化后的字符串，用于Prompt。
    """
    if not existing_sobriquets:
        return "当前上下文中用户无已记录的常用绰号。" # 如果没有数据，返回提示信息
    
    lines = ["当前上下文中用户已记录的常用绰号（供你参考，帮助判断新提及的是否为已知绰号或新绰号，以及哪些可能是旧的或不再使用的）："]
    for user_id, sobriquets in existing_sobriquets.items():
        if sobriquets: # 确保绰号列表不为空
            valid_sobriquet_strs = []
            for s_entry in sobriquets: # 遍历该用户的每个绰号条目
                if isinstance(s_entry, dict) and "name" in s_entry and "count" in s_entry:
                    # 格式化每个绰号及其计数
                    valid_sobriquet_strs.append(f"“{s_entry.get('name')}”(提及{s_entry.get('count',0)}次)")
            
            if valid_sobriquet_strs: # 如果存在有效的绰号字符串
                lines.append(f"用户ID {user_id}: {', '.join(valid_sobriquet_strs)}")
            else:
                lines.append(f"用户ID {user_id}: 无有效已记录绰号") # 如果用户的绰号列表无效或为空
        else:
            lines.append(f"用户ID {user_id}: 无已记录绰号") # 如果用户ID存在但没有绰号数据
    return "\n".join(lines)

def build_sobriquet_mapping_prompt(
    chat_history_str: str, 
    bot_reply: str,
    existing_sobriquets_str: Optional[str] = None # 新增参数：已存在的绰号信息字符串
) -> str:
    """
    构建用于 LLM 进行绰号映射分析的 Prompt。
    要求LLM同时输出可靠和不可靠的映射，并参考已存在的绰号信息。

    Args:
        chat_history_str: 格式化后的聊天历史记录字符串。
        bot_reply: Bot 的最新回复字符串。
        existing_sobriquets_str: 可选的，当前上下文中用户已存在的绰号信息字符串。

    Returns:
        str: 构建好的 Prompt 字符串。
    """
    reference_info_section = "" # 初始化参考信息部分
    if existing_sobriquets_str: # 如果提供了已存在的绰号信息
        reference_info_section = f"\n\n已知信息参考（上下文中用户的既有绰号记录）：\n---\n{existing_sobriquets_str}\n---\n"

    # 构建完整的Prompt字符串
    prompt = f"""
任务：仔细分析以下“聊天记录”、“你的最新回复”以及可选的“已知信息参考”，识别用户ID与绰号之间的映射关系。你需要区分哪些映射是基于当前聊天内容清晰、明确、可靠的，哪些是模糊、不确定、或者基于已知信息判断可能已过时或不准确的（应被否定或标记为不可靠）。

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
2.  **不可靠或需否定的映射 (unreliable_mappings)**：
    * 找出那些在当前聊天中被提及，但其与特定用户ID的关联**模糊、不确定、可能是猜测**的称呼。
    * 也包括那些你根据上下文（例如用户明确否认）或对比“已知信息参考”（例如一个低频历史绰号在当前无明显使用迹象，或与高频常用绰号冲突且无合理解释）判断，某用户**明确不再使用或不适用**于某个曾被提及的称呼的情况。
    * 例如，如果“已知信息参考”显示用户A常用“小明”(高频)，但当前聊天有人偶然叫他“大明”(低频，且上下文不强烈支持“大明”是新的稳定称呼)，则“大明”可能归为此类。
    * 或者，如果聊天中有人说“我早不用‘二狗’这个外号了”，那么“二狗”对于该用户就是一个需要在此处报告的“应否定映射”。
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
            "用户C数字id": "存疑或应否定绰号_C1",
            "用户D数字id": "存疑或应否定绰号_D1"
        }}
    }}
    ```
    * `"reliable_mappings"` 和 `"unreliable_mappings"` 字段的键都必须是用户的**数字 ID (字符串形式)**，值是对应的**绰号 (字符串形式)**。
    * 如果某类映射不存在，则对应的字段值必须是一个空对象 `{{}}`。例如，如果没有不可靠映射，则为 `"unreliable_mappings": {{}}`。
    * 对于 `reliable_mappings`，只包含你能**高度确认**映射关系的条目。
    * 对于 `unreliable_mappings`，包含那些你认为关联性弱、存疑、或根据上下文应被否定的映射。
5.  请**仅**输出上述格式的 JSON 对象，不要包含任何额外的解释、注释或代码块标记之外的文本。

输出：
"""
    return prompt
