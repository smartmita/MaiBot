import json
from typing import Dict, Any, Optional
from src.common.logger_manager import get_logger
from src.plugins.models.utils_model import LLMRequest
# 从全局配置导入
from src.config.config import global_config


logger = get_logger("nickname_mapper")

llm_mapper: Optional[LLMRequest] = None
if global_config.ENABLE_NICKNAME_MAPPING: # 使用全局开关
    try:
        # 从全局配置获取模型设置
        model_config = global_config.llm_nickname_mapping
        if not model_config or not model_config.get("name"):
            logger.error("在全局配置中未找到有效的 'llm_nickname_mapping' 配置或缺少 'name' 字段。")
        else:
            llm_args = {
                "model": model_config.get("name"), # 必须有 name
                "temperature": model_config.get("temp", 0.5), # 使用 temp 字段
                "max_tokens": model_config.get("max_tokens", 200), # max_tokens 是可选的，取决于 LLMRequest 实现
                "api_key": model_config.get("key"), # 使用 key 字段
                "base_url": model_config.get("base_url"), # 使用 base_url 字段
                "request_type": "nickname_mapping"
            }
            # 清理 None 值参数
            llm_args = {k: v for k, v in llm_args.items() if v is not None}

            llm_mapper = LLMRequest(**llm_args)
            logger.info("绰号映射 LLM 初始化成功 (使用全局配置)。")

    except Exception as e:
        logger.error(f"使用全局配置初始化绰号映射 LLM 失败: {e}", exc_info=True)
        llm_mapper = None
# --- 结束修改 ---

def _build_mapping_prompt(chat_history_str: str, bot_reply: str, user_name_map: Dict[str, str]) -> str:
    # ... (函数内容不变) ...
    user_list_str = "\n".join([f"- {uid}: {name}" for uid, name in user_name_map.items()])

    prompt = f"""
任务：分析以下聊天记录和 Bot 的最新回复，判断其中是否包含用户绰号，并确定绰号与用户 ID 之间是否存在明确的一一对应关系。

已知用户信息：
{user_list_str}

聊天记录：
---
{chat_history_str}
---

Bot 最新回复：
{bot_reply}

分析要求：
1.  识别聊天记录和 Bot 回复中出现的可能是用户绰号的词语。
2.  判断这些绰号是否能明确地指向某个特定的用户 ID。一个绰号必须在上下文中清晰地与某个发言人或被提及的人关联起来。
3.  如果能建立可靠的一一映射关系，请输出一个 JSON 对象，格式如下：
    {{
        "is_exist": true,
        "data": {{
            "用户ID_A": "绰号_A",
            "用户ID_B": "绰号_B"
        }}
    }}
    其中 "data" 字段的键是用户的 ID，值是对应的绰号。只包含你能确认映射关系的绰号。
4.  如果无法建立任何可靠的一一映射关系（例如，绰号指代不明、没有出现绰号、或无法确认绰号与用户的关联），请输出 JSON 对象：
    {{
        "is_exist": false
    }}
5.  请严格按照 JSON 格式输出，不要包含任何额外的解释或文本。

输出：
"""
    return prompt


async def analyze_chat_for_nicknames(
    chat_history_str: str,
    bot_reply: str,
    user_name_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    调用 LLM 分析聊天记录和 Bot 回复，提取可靠的 用户ID-绰号 映射。
    """
    # --- [修改] 使用全局配置开关 ---
    if not global_config.ENABLE_NICKNAME_MAPPING:
    # --- 结束修改 ---
        logger.debug("绰号映射功能已禁用。")
        return {"is_exist": False}

    if llm_mapper is None:
        logger.error("绰号映射 LLM 未初始化。无法执行分析。")
        return {"is_exist": False}

    prompt = _build_mapping_prompt(chat_history_str, bot_reply, user_name_map)
    logger.debug(f"构建的绰号映射 Prompt:\n{prompt}")

    try:
        # 调用 LLM
        response_content, _, _ = await llm_mapper.generate_response(prompt)
        logger.debug(f"LLM 原始响应 (绰号映射): {response_content}")

        # ... (解析 LLM 响应的逻辑不变) ...
        if not response_content:
            logger.warning("LLM 返回了空的绰号映射内容。")
            return {"is_exist": False}

        response_content = response_content.strip()
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()

        try:
            result = json.loads(response_content)
            if isinstance(result, dict) and "is_exist" in result:
                if result["is_exist"] is True:
                    if "data" in result and isinstance(result["data"], dict):
                        if not result["data"]:
                            logger.debug("LLM 指示 is_exist=True 但 data 为空。视为 False 处理。")
                            return {"is_exist": False}
                        logger.info(f"找到绰号映射: {result['data']}")
                        return {"is_exist": True, "data": result["data"]}
                    else:
                        logger.warning("LLM 响应格式错误: is_exist 为 True 但 'data' 缺失或不是字典。")
                        return {"is_exist": False}
                elif result["is_exist"] is False:
                    logger.info("LLM 未找到可靠的绰号映射。")
                    return {"is_exist": False}
                else:
                    logger.warning("LLM 响应格式错误: 'is_exist' 不是布尔值。")
                    return {"is_exist": False}
            else:
                logger.warning("LLM 响应格式错误: 缺少 'is_exist' 键或不是字典。")
                return {"is_exist": False}
        except json.JSONDecodeError as json_err:
            logger.error(f"解析 LLM 响应 JSON 失败: {json_err}\n原始响应: {response_content}")
            return {"is_exist": False}

    except Exception as e:
        logger.error(f"绰号映射 LLM 调用或处理过程中出错: {e}", exc_info=True)
        return {"is_exist": False}

