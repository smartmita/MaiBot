import traceback
import json
import re
from typing import Dict, Any, Optional, Tuple, List, Union
from src.common.logger_manager import get_logger  # 确认 logger 的导入路径
from src.plugins.memory_system.Hippocampus import HippocampusManager
from src.plugins.heartFC_chat.heartflow_prompt_builder import prompt_builder  # 确认 prompt_builder 的导入路径
from src.plugins.chat.chat_stream import ChatStream
from ..person_info.person_info import person_info_manager
import math

logger = get_logger("pfc_utils")


async def retrieve_contextual_info(text: str, private_name: str) -> Tuple[str, str]:
    """
    根据输入文本检索相关的记忆和知识。

    Args:
        text: 用于检索的上下文文本 (例如聊天记录)。
        private_name: 私聊对象的名称，用于日志记录。

    Returns:
        Tuple[str, str]: (检索到的记忆字符串, 检索到的知识字符串)
    """
    retrieved_memory_str = "无相关记忆。"
    retrieved_knowledge_str = "无相关知识。"
    memory_log_msg = "未自动检索到相关记忆。"
    knowledge_log_msg = "未自动检索到相关知识。"

    if not text or text == "还没有聊天记录。" or text == "[构建聊天记录出错]":
        logger.debug(f"[私聊][{private_name}] (retrieve_contextual_info) 无有效上下文，跳过检索。")
        return retrieved_memory_str, retrieved_knowledge_str

    # 1. 检索记忆 (逻辑来自原 _get_memory_info)
    try:
        related_memory = await HippocampusManager.get_instance().get_memory_from_text(
            text=text,
            max_memory_num=2,
            max_memory_length=2,
            max_depth=3,
            fast_retrieval=False,
        )
        if related_memory:
            related_memory_info = ""
            for memory in related_memory:
                related_memory_info += memory[1] + "\n"
            if related_memory_info:
                # 注意：原版提示信息可以根据需要调整
                retrieved_memory_str = f"你回忆起：\n{related_memory_info.strip()}\n(以上是你的回忆，供参考)\n"
                memory_log_msg = f"自动检索到记忆: {related_memory_info.strip()[:100]}..."
            else:
                memory_log_msg = "自动检索记忆返回为空。"
        logger.debug(f"[私聊][{private_name}] (retrieve_contextual_info) 记忆检索: {memory_log_msg}")

    except Exception as e:
        logger.error(
            f"[私聊][{private_name}] (retrieve_contextual_info) 自动检索记忆时出错: {e}\n{traceback.format_exc()}"
        )
        retrieved_memory_str = "检索记忆时出错。\n"

    # 2. 检索知识 (逻辑来自原 action_planner 和 reply_generator)
    try:
        # 使用导入的 prompt_builder 实例及其方法
        knowledge_result = await prompt_builder.get_prompt_info(
            message=text,
            threshold=0.38,  # threshold 可以根据需要调整
        )
        if knowledge_result:
            retrieved_knowledge_str = knowledge_result  # 直接使用返回结果
            knowledge_log_msg = "自动检索到相关知识。"
        logger.debug(f"[私聊][{private_name}] (retrieve_contextual_info) 知识检索: {knowledge_log_msg}")

    except Exception as e:
        logger.error(
            f"[私聊][{private_name}] (retrieve_contextual_info) 自动检索知识时出错: {e}\n{traceback.format_exc()}"
        )
        retrieved_knowledge_str = "检索知识时出错。\n"

    return retrieved_memory_str, retrieved_knowledge_str


def get_items_from_json(
    content: str,
    private_name: str,
    *items: str,
    default_values: Optional[Dict[str, Any]] = None,
    required_types: Optional[Dict[str, type]] = None,
    allow_array: bool = True,
) -> Tuple[bool, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """从文本中提取JSON内容并获取指定字段

    Args:
        content: 包含JSON的文本
        private_name: 私聊名称
        *items: 要提取的字段名
        default_values: 字段的默认值，格式为 {字段名: 默认值}
        required_types: 字段的必需类型，格式为 {字段名: 类型}
        allow_array: 是否允许解析JSON数组

    Returns:
        Tuple[bool, Union[Dict[str, Any], List[Dict[str, Any]]]]: (是否成功, 提取的字段字典或字典列表)
    """
    cleaned_content = content.strip()
    result: Union[Dict[str, Any], List[Dict[str, Any]]] = {}  # 初始化类型
    # 匹配 ```json ... ``` 或 ``` ... ```
    markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content, re.IGNORECASE)
    if markdown_match:
        cleaned_content = markdown_match.group(1).strip()
        logger.debug(f"[私聊][{private_name}] 已去除 Markdown 标记，剩余内容: {cleaned_content[:100]}...")
    # --- 新增结束 ---

    # 设置默认值
    default_result: Dict[str, Any] = {}  # 用于单对象时的默认值
    if default_values:
        default_result.update(default_values)
        result = default_result.copy()  # 先用默认值初始化

    # 首先尝试解析为JSON数组
    if allow_array:
        try:
            # 尝试直接解析清理后的内容为列表
            json_array = json.loads(cleaned_content)

            if isinstance(json_array, list):
                valid_items_list: List[Dict[str, Any]] = []
                for item in json_array:
                    if not isinstance(item, dict):
                        logger.warning(f"[私聊][{private_name}] JSON数组中的元素不是字典: {item}")
                        continue

                    current_item_result = default_result.copy()  # 每个元素都用默认值初始化
                    valid_item = True

                    # 提取并验证字段
                    for field in items:
                        if field in item:
                            current_item_result[field] = item[field]
                        elif field not in default_result:  # 如果字段不存在且没有默认值
                            logger.warning(f"[私聊][{private_name}] JSON数组元素缺少必要字段 '{field}': {item}")
                            valid_item = False
                            break  # 这个元素无效

                    if not valid_item:
                        continue

                    # 验证类型
                    if required_types:
                        for field, expected_type in required_types.items():
                            # 检查 current_item_result 中是否存在该字段 (可能来自 item 或 default_values)
                            if field in current_item_result and not isinstance(
                                current_item_result[field], expected_type
                            ):
                                logger.warning(
                                    f"[私聊][{private_name}] JSON数组元素字段 '{field}' 类型错误 (应为 {expected_type.__name__}, 实际为 {type(current_item_result[field]).__name__}): {item}"
                                )
                                valid_item = False
                                break

                    if not valid_item:
                        continue

                    # 验证字符串不为空 (只检查 items 中要求的字段)
                    for field in items:
                        if (
                            field in current_item_result
                            and isinstance(current_item_result[field], str)
                            and not current_item_result[field].strip()
                        ):
                            logger.warning(f"[私聊][{private_name}] JSON数组元素字段 '{field}' 不能为空字符串: {item}")
                            valid_item = False
                            break

                    if valid_item:
                        valid_items_list.append(current_item_result)  # 只添加完全有效的项

                if valid_items_list:  # 只有当列表不为空时才认为是成功
                    logger.debug(f"[私聊][{private_name}] 成功解析JSON数组，包含 {len(valid_items_list)} 个有效项目。")
                    return True, valid_items_list
                else:
                    # 如果列表为空（可能所有项都无效），则继续尝试解析为单个对象
                    logger.debug(f"[私聊][{private_name}] 解析为JSON数组，但未找到有效项目，尝试解析单个JSON对象。")
                    # result 重置回单个对象的默认值
                    result = default_result.copy()

        except json.JSONDecodeError:
            logger.debug(f"[私聊][{private_name}] JSON数组直接解析失败，尝试解析单个JSON对象")
            # result 重置回单个对象的默认值
            result = default_result.copy()
        except Exception as e:
            logger.error(f"[私聊][{private_name}] 尝试解析JSON数组时发生未知错误: {str(e)}")
            # result 重置回单个对象的默认值
            result = default_result.copy()

    # 尝试解析为单个JSON对象
    try:
        # 尝试直接解析清理后的内容
        json_data = json.loads(cleaned_content)
        if not isinstance(json_data, dict):
            logger.error(f"[私聊][{private_name}] 解析为单个对象，但结果不是字典类型: {type(json_data)}")
            return False, default_result  # 返回失败和默认值

    except json.JSONDecodeError:
        # 如果直接解析失败，尝试用正则表达式查找 JSON 对象部分 (作为后备)
        # 这个正则比较简单，可能无法处理嵌套或复杂的 JSON
        json_pattern = r"\{[\s\S]*?\}"  # 使用非贪婪匹配
        json_match = re.search(json_pattern, cleaned_content)
        if json_match:
            try:
                potential_json_str = json_match.group()
                json_data = json.loads(potential_json_str)
                if not isinstance(json_data, dict):
                    logger.error(f"[私聊][{private_name}] 正则提取后解析，但结果不是字典类型: {type(json_data)}")
                    return False, default_result
                logger.debug(f"[私聊][{private_name}] 通过正则提取并成功解析JSON对象。")
            except json.JSONDecodeError:
                logger.error(f"[私聊][{private_name}] 正则提取的部分 '{potential_json_str[:100]}...' 无法解析为JSON。")
                return False, default_result
        else:
            logger.error(
                f"[私聊][{private_name}] 无法在返回内容中找到有效的JSON对象部分。原始内容: {cleaned_content[:100]}..."
            )
            return False, default_result

    # 提取并验证字段 (适用于单个JSON对象)
    # 确保 result 是字典类型用于更新
    if not isinstance(result, dict):
        result = default_result.copy()  # 如果之前是列表，重置为字典

    valid_single_object = True
    for item in items:
        if item in json_data:
            result[item] = json_data[item]
        elif item not in default_result:  # 如果字段不存在且没有默认值
            logger.error(f"[私聊][{private_name}] JSON对象缺少必要字段 '{item}'。JSON内容: {json_data}")
            valid_single_object = False
            break  # 这个对象无效

    if not valid_single_object:
        return False, default_result

    # 验证类型
    if required_types:
        for field, expected_type in required_types.items():
            if field in result and not isinstance(result[field], expected_type):
                logger.error(
                    f"[私聊][{private_name}] JSON对象字段 '{field}' 类型错误 (应为 {expected_type.__name__}, 实际为 {type(result[field]).__name__})"
                )
                valid_single_object = False
                break

    if not valid_single_object:
        return False, default_result

    # 验证字符串不为空 (只检查 items 中要求的字段)
    for field in items:
        if field in result and isinstance(result[field], str) and not result[field].strip():
            logger.error(f"[私聊][{private_name}] JSON对象字段 '{field}' 不能为空字符串")
            valid_single_object = False
            break

    if valid_single_object:
        logger.debug(f"[私聊][{private_name}] 成功解析并验证了单个JSON对象。")
        return True, result  # 返回提取并验证后的字典
    else:
        return False, default_result  # 验证失败


async def get_person_id(private_name: str, chat_stream: ChatStream):
    private_user_id_str: Optional[str] = None
    private_platform_str: Optional[str] = None
    private_nickname_str = private_name

    if chat_stream.user_info:
        private_user_id_str = str(chat_stream.user_info.user_id)
        private_platform_str = chat_stream.user_info.platform
        logger.info(
            f"[私聊][{private_name}] 从 ChatStream 获取到私聊对象信息: ID={private_user_id_str}, Platform={private_platform_str}, Name={private_nickname_str}"
        )
    elif chat_stream.group_info is None and private_name:
        pass

    if private_user_id_str and private_platform_str:
        try:
            private_user_id_int = int(private_user_id_str)
            # person_id = person_info_manager.get_person_id( # get_person_id 可能只查询，不创建
            #     private_platform_str,
            #     private_user_id_int
            # )
            # 使用 get_or_create_person 确保用户存在
            person_id = await person_info_manager.get_or_create_person(
                platform=private_platform_str,
                user_id=private_user_id_int,
                nickname=private_name,  # 使用传入的 private_name 作为昵称
            )
            if person_id is None:  # 如果 get_or_create_person 返回 None，说明创建失败
                logger.error(f"[私聊][{private_name}] get_or_create_person 未能获取或创建 person_id。")
                return None  # 返回 None 表示失败

            return person_id, private_platform_str, private_user_id_str  # 返回获取或创建的 person_id
        except ValueError:
            logger.error(f"[私聊][{private_name}] 无法将 private_user_id_str ('{private_user_id_str}') 转换为整数。")
            return None  # 返回 None 表示失败
        except Exception as e_pid:
            logger.error(f"[私聊][{private_name}] 获取或创建 person_id 时出错: {e_pid}")
            return None  # 返回 None 表示失败
    else:
        logger.warning(
            f"[私聊][{private_name}] 未能确定私聊对象的 user_id 或 platform，无法获取 person_id。将在收到消息后尝试。"
        )
        return None  # 返回 None 表示失败


async def adjust_relationship_value_nonlinear(old_value: float, raw_adjustment: float) -> float:
    # 限制 old_value 范围
    old_value = max(-1000, min(1000, old_value))
    value = raw_adjustment

    if old_value >= 0:
        if value >= 0:
            value = value * math.cos(math.pi * old_value / 2000)
            if old_value > 500:
                rdict = await person_info_manager.get_specific_value_list("relationship_value", lambda x: x > 700)
                high_value_count = len(rdict)
                if old_value > 700:
                    value *= 3 / (high_value_count + 2)
                else:
                    value *= 3 / (high_value_count + 3)
        elif value < 0:
            value = value * math.exp(old_value / 2000)
        else:
            value = 0
    else:
        if value >= 0:
            value = value * math.exp(old_value / 2000)
        elif value < 0:
            value = value * math.cos(math.pi * old_value / 2000)
        else:
            value = 0

    return value
