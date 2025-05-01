import json
from typing import Dict, Any, Optional
from src.common.logger_manager import get_logger
from src.plugins.models.utils_model import LLMRequest
from src.config.config import global_config


logger = get_logger("nickname_mapper")

llm_mapper: Optional[LLMRequest] = None
if global_config.ENABLE_NICKNAME_MAPPING:  # 使用全局开关
    try:
        # 从全局配置获取模型设置
        model_config = global_config.llm_nickname_mapping
        if not model_config or not model_config.get("name"):
            logger.error("在全局配置中未找到有效的 'llm_nickname_mapping' 配置或缺少 'name' 字段。")
        else:
            llm_mapper = LLMRequest(  # <-- LLM 初始化
                model=global_config.llm_nickname_mapping,
                temperature=global_config.llm_nickname_mapping["temp"],
                max_tokens=256,
                request_type="nickname_mapping",
            )
            logger.info("绰号映射 LLM 初始化成功 (使用全局配置)。")

    except Exception as e:
        logger.error(f"使用全局配置初始化绰号映射 LLM 失败: {e}", exc_info=True)
        llm_mapper = None


def _build_mapping_prompt(chat_history_str: str, bot_reply: str, user_name_map: Dict[str, str]) -> str:
    """构建用于 LLM 绰号映射的 Prompt"""
    # user_name_map 包含了 user_id 到 person_name (或 fallback nickname) 的映射
    user_list_str = "\n".join([f"- {uid}: {name}" for uid, name in user_name_map.items()])
    # print(f"\n\n\nKnown User Info for LLM:\n{user_list_str}\n\n\n\n") # Debugging print
    prompt = f"""
任务：分析以下聊天记录和你的最新回复，判断其中是否包含用户绰号，并确定绰号与用户 ID 之间是否存在明确的一一对应关系。

已知用户信息（ID: 名称）：
{user_list_str}

聊天记录：
---
{chat_history_str}
---

你的最新回复：
{bot_reply}

分析要求：
1.  识别聊天记录和你发言中出现的可能是用户绰号的词语。
2.  判断这些绰号是否能明确地指向某个特定的用户 ID。一个绰号必须在上下文中清晰地与某个发言人或被提及的人关联起来。
3.  如果能建立可靠的一一映射关系，请输出一个 JSON 对象，格式如下：
    {{
        "is_exist": true,
        "data": {{
            "用户A数字id": "绰号_A",
            "用户B数字id": "绰号_B"
        }}
    }}
    其中 "data" 字段的键是用户的 ID (字符串形式)，值是对应的绰号。只包含你能确认映射关系的绰号。
4.  如果无法建立任何可靠的一一映射关系（例如，绰号指代不明、没有出现绰号、或无法确认绰号与用户的关联），请输出 JSON 对象：
    {{
        "is_exist": false
    }}
5.  在“已知用户信息”列表中，你的昵称后面可能包含"(你)"，这表示是你自己，不需要输出你自身的绰号映射。请确保不要将你自己的ID和任何词语映射为绰号。
6.  不要输出与用户名称相同的绰号，不要输出你发言中对他人的绰号映射。
7.  请严格按照 JSON 格式输出，不要包含任何额外的解释或文本。

输出：
"""
    return prompt


async def analyze_chat_for_nicknames(
    chat_history_str: str,
    bot_reply: str,
    user_name_map: Dict[str, str],  # 这个 map 包含了 user_id -> person_name 的信息
) -> Dict[str, Any]:
    """
    调用 LLM 分析聊天记录和 Bot 回复，提取可靠的 用户ID-绰号 映射，并进行过滤。
    """
    if not global_config.ENABLE_NICKNAME_MAPPING:
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

        if not response_content:
            logger.warning("LLM 返回了空的绰号映射内容。")
            return {"is_exist": False}

        # 清理可能的 Markdown 代码块标记
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
                    original_data = result.get("data")  # 使用 .get() 更安全
                    if isinstance(original_data, dict) and original_data:  # 确保 data 是非空字典
                        logger.info(f"LLM 找到的原始绰号映射: {original_data}")

                        # --- 开始过滤 ---
                        filtered_data = {}
                        bot_qq_str = str(global_config.BOT_QQ)  # 将机器人QQ转为字符串以便比较

                        for user_id, nickname in original_data.items():
                            # 检查 user_id 是否是字符串，以防万一
                            if not isinstance(user_id, str):
                                logger.warning(f"LLM 返回的 user_id '{user_id}' 不是字符串，跳过。")
                                continue

                            # 条件 1: 排除机器人自身
                            if user_id == bot_qq_str:
                                logger.debug(f"过滤掉机器人自身的映射: ID {user_id}")
                                continue

                            # 有了改名工具后，该过滤器已不适合了，尝试通过修改 prompt 获得更好的结果
                            # # 条件 2: 排除 nickname 与 person_name 相同的情况
                            # person_name = user_name_map.get(user_id) # 从传入的映射中查找 person_name
                            # if person_name and person_name == nickname:
                            #     logger.debug(f"过滤掉用户 {user_id} 的映射: 绰号 '{nickname}' 与其名称 '{person_name}' 相同。")
                            #     continue

                            # 如果通过所有过滤条件，则保留
                            filtered_data[user_id] = nickname
                        # --- 结束过滤 ---

                        # 检查过滤后是否还有数据
                        if not filtered_data:
                            logger.info("所有找到的绰号映射都被过滤掉了。")
                            return {"is_exist": False}
                        else:
                            logger.info(f"过滤后的绰号映射: {filtered_data}")
                            return {"is_exist": True, "data": filtered_data}  # 返回过滤后的数据

                    else:
                        # is_exist 为 True 但 data 缺失、不是字典或为空
                        if "data" not in result:
                            logger.warning("LLM 响应格式错误: is_exist 为 True 但 'data' 键缺失。")
                        elif not isinstance(result.get("data"), dict):
                            logger.warning("LLM 响应格式错误: is_exist 为 True 但 'data' 不是字典。")
                        else:  # data 为空字典
                            logger.debug("LLM 指示 is_exist=True 但 data 为空字典。视为 False 处理。")
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
