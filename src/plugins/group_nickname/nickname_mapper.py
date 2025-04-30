# GroupNickname/nickname_mapper.py
import json
from typing import Dict, Any, Tuple, List, Optional
from src.common.logger_manager import get_logger # 假设你的日志管理器路径
from src.plugins.models.utils_model import LLMRequest # 假设你的 LLM 请求工具路径
from .config import LLM_MODEL_NICKNAME_MAPPING, ENABLE_NICKNAME_MAPPING

logger = get_logger("nickname_mapper")

# 初始化用于绰号映射的 LLM 实例
# 注意：这里的初始化方式可能需要根据你的 LLMRequest 实现进行调整
try:
    # 尝试使用字典解包来传递参数
    llm_mapper = LLMRequest(
        model=LLM_MODEL_NICKNAME_MAPPING.get("model_name", "default_model"),
        temperature=LLM_MODEL_NICKNAME_MAPPING.get("temperature", 0.5),
        max_tokens=LLM_MODEL_NICKNAME_MAPPING.get("max_tokens", 200),
        api_key=LLM_MODEL_NICKNAME_MAPPING.get("api_key"),
        base_url=LLM_MODEL_NICKNAME_MAPPING.get("base_url"),
        request_type="nickname_mapping" # 定义一个请求类型用于区分
    )
    logger.info("Nickname mapping LLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize nickname mapping LLM: {e}", exc_info=True)
    llm_mapper = None # 初始化失败则置为 None

def _build_mapping_prompt(chat_history_str: str, bot_reply: str, user_name_map: Dict[str, str]) -> str:
    """
    构建用于 LLM 分析绰号映射的 Prompt。

    Args:
        chat_history_str: 格式化后的聊天记录字符串。
        bot_reply: Bot 的回复内容。
        user_name_map: 用户 ID 到已知名称（如 person_name 或 nickname）的映射。

    Returns:
        str: 构建好的 Prompt。
    """
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

    Args:
        chat_history_str: 格式化后的聊天记录字符串。
        bot_reply: Bot 的回复内容。
        user_name_map: 用户 ID 到已知名称（如 person_name 或 nickname）的映射。

    Returns:
        Dict[str, Any]: 分析结果，格式为 { "is_exist": bool, "data": Optional[Dict[str, str]] }。
                       如果出错，返回 {"is_exist": False}。
    """
    if not ENABLE_NICKNAME_MAPPING:
        logger.debug("Nickname mapping feature is disabled.")
        return {"is_exist": False}

    if llm_mapper is None:
        logger.error("Nickname mapping LLM is not initialized. Cannot perform analysis.")
        return {"is_exist": False}

    prompt = _build_mapping_prompt(chat_history_str, bot_reply, user_name_map)
    logger.debug(f"Nickname mapping prompt built:\n{prompt}") # 调试日志

    try:
        # --- 调用 LLM ---
        # 注意：这里的调用方式需要根据你的 LLMRequest 实现进行调整
        # 可能需要使用 generate_response_sync 或其他同步方法，因为这将在独立进程中运行
        # 或者如果 LLMRequest 支持异步，确保在异步环境中调用
        # response_content, _, _ = await llm_mapper.generate_response(prompt)

        # 假设 llm_mapper 有一个同步的 generate 方法或在异步环境中调用
        # 这里暂时使用 await，如果你的 LLMRequest 不支持，需要修改
        response_content, _, _ = await llm_mapper.generate_response(prompt)


        logger.debug(f"LLM raw response for nickname mapping: {response_content}")

        # --- 解析 LLM 响应 ---
        if not response_content:
            logger.warning("LLM returned empty content for nickname mapping.")
            return {"is_exist": False}

        # 尝试去除可能的代码块标记
        response_content = response_content.strip()
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()

        try:
            result = json.loads(response_content)
            # 基本验证
            if isinstance(result, dict) and "is_exist" in result:
                if result["is_exist"] is True:
                    if "data" in result and isinstance(result["data"], dict):
                        # 过滤掉 data 为空的情况
                        if not result["data"]:
                            logger.debug("LLM indicated is_exist=True but data is empty. Treating as False.")
                            return {"is_exist": False}
                        logger.info(f"Nickname mapping found: {result['data']}")
                        return {"is_exist": True, "data": result["data"]}
                    else:
                        logger.warning("LLM response format error: is_exist is True but 'data' is missing or not a dict.")
                        return {"is_exist": False}
                elif result["is_exist"] is False:
                    logger.info("No reliable nickname mapping found by LLM.")
                    return {"is_exist": False}
                else:
                    logger.warning("LLM response format error: 'is_exist' is not a boolean.")
                    return {"is_exist": False}
            else:
                logger.warning("LLM response format error: Missing 'is_exist' key or not a dict.")
                return {"is_exist": False}
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM response as JSON: {json_err}\nRaw response: {response_content}")
            return {"is_exist": False}

    except Exception as e:
        logger.error(f"Error during nickname mapping LLM call or processing: {e}", exc_info=True)
        return {"is_exist": False}

