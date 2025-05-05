import traceback
import json
import re
from typing import Dict, Any, Optional, Tuple, List, Union
from src.common.logger_manager import get_logger  # 确认 logger 的导入路径
from src.plugins.memory_system.Hippocampus import HippocampusManager
from src.plugins.heartFC_chat.heartflow_prompt_builder import prompt_builder  # 确认 prompt_builder 的导入路径

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
    content = content.strip()
    result = {}

    # 设置默认值
    if default_values:
        result.update(default_values)

    # 首先尝试解析为JSON数组
    if allow_array:
        try:
            # 尝试找到文本中的JSON数组
            array_pattern = r"\[[\s\S]*\]"
            array_match = re.search(array_pattern, content)
            if array_match:
                array_content = array_match.group()
                json_array = json.loads(array_content)

                # 确认是数组类型
                if isinstance(json_array, list):
                    # 验证数组中的每个项目是否包含所有必需字段
                    valid_items = []
                    for item in json_array:
                        if not isinstance(item, dict):
                            continue

                        # 检查是否有所有必需字段
                        if all(field in item for field in items):
                            # 验证字段类型
                            if required_types:
                                type_valid = True
                                for field, expected_type in required_types.items():
                                    if field in item and not isinstance(item[field], expected_type):
                                        type_valid = False
                                        break

                                if not type_valid:
                                    continue

                            # 验证字符串字段不为空
                            string_valid = True
                            for field in items:
                                if isinstance(item[field], str) and not item[field].strip():
                                    string_valid = False
                                    break

                            if not string_valid:
                                continue

                            valid_items.append(item)

                    if valid_items:
                        return True, valid_items
        except json.JSONDecodeError:
            logger.debug(f"[私聊][{private_name}]JSON数组解析失败，尝试解析单个JSON对象")
        except Exception as e:
            logger.debug(f"[私聊][{private_name}]尝试解析JSON数组时出错: {str(e)}")

    # 尝试解析JSON对象
    try:
        json_data = json.loads(content)
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试查找和提取JSON部分
        json_pattern = r"\{[^{}]*\}"
        json_match = re.search(json_pattern, content)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.error(f"[私聊][{private_name}]提取的JSON内容解析失败")
                return False, result
        else:
            logger.error(f"[私聊][{private_name}]无法在返回内容中找到有效的JSON")
            return False, result

    # 提取字段
    for item in items:
        if item in json_data:
            result[item] = json_data[item]

    # 验证必需字段
    if not all(item in result for item in items):
        logger.error(f"[私聊][{private_name}]JSON缺少必要字段，实际内容: {json_data}")
        return False, result

    # 验证字段类型
    if required_types:
        for field, expected_type in required_types.items():
            if field in result and not isinstance(result[field], expected_type):
                logger.error(f"[私聊][{private_name}]{field} 必须是 {expected_type.__name__} 类型")
                return False, result

    # 验证字符串字段不为空
    for field in items:
        if isinstance(result[field], str) and not result[field].strip():
            logger.error(f"[私聊][{private_name}]{field} 不能为空")
            return False, result

    return True, result
