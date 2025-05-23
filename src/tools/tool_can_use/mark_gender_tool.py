from src.config.config import global_config
from src.tools.tool_can_use.base_tool import BaseTool, register_tool
from src.common.logger_manager import get_logger
from typing import Dict, Any, Optional
from src.chat.person_info.person_info import person_info_manager # 导入 person_info_manager
from src.chat.message_receive.chat_stream import ChatStream # 导入 ChatStream

logger = get_logger("mark_gender_tool")

class MarkGenderTool(BaseTool):
    """标记或更新用户性别的工具"""

    name = "mark_user_gender" # 工具的唯一名称
    description = "当你从对话中明确或大概判断出某个用户的性别时，使用此工具来标记或更新该用户的性别信息。性别可以是：男、女、可能为男、可能为女、未知。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "要标记性别的用户的UID。"},
            "gender_mark": {"type": "string", "description": "推断出的性别标记 (例如: '男', '女', '可能为男', '可能为女', '未知')。"},
            "reasoning": {"type": "string", "description": "简要说明你为什么做出这个性别判断。"}
        },
        "required": ["user_id", "gender_mark", "reasoning"],
    }

    # 修改 execute 方法以接受 chat_stream
    async def execute(self, function_args: Dict[str, Any], chat_stream: Optional[ChatStream] = None) -> Dict[str, Any]:
        
        user_id_str = function_args.get("user_id")
        gender_mark_value = function_args.get("gender_mark")
        reasoning = function_args.get("reasoning", "未提供原因")

        if not global_config.experimental.enable_gender_marking_tool:
            return {"type": "gender_mark_disabled", "id": user_id_str, "content": "性别标记功能未启用。"}

        if not user_id_str or not gender_mark_value:
            logger.warning(f"MarkGenderTool: 缺少 user_id 或 gender_mark 参数。 Args: {function_args}")
            return {"type": "gender_mark_update", "id": user_id_str or "未知用户", "content": "标记性别失败：缺少必要参数。"}

        if not chat_stream or not chat_stream.platform:
            logger.warning(f"MarkGenderTool: 缺少有效的 chat_stream 或 platform 信息，无法确定平台。")
            return {"type": "gender_mark_update", "id": user_id_str, "content": "标记性别失败：无法确定平台。"}

        platform = chat_stream.platform

        allowed_gender_marks = ["男", "女", "可能为男", "可能为女", "未知"]
        if gender_mark_value not in allowed_gender_marks:
            logger.warning(f"MarkGenderTool: 无效的性别标记 '{gender_mark_value}'。允许的值: {allowed_gender_marks}")
            return {"type": "gender_mark_update", "id": user_id_str, "content": f"标记性别失败：无效的性别标记 '{gender_mark_value}'。"}

        try:
            # PersonInfoManager.get_person_id 期望 user_id 是 int，但工具参数通常是 string。
            # 需要确保 person_info_manager.get_person_id 可以处理字符串，或者在这里转换。
            # 你提供的 person_info.py 中的 get_person_id 接受 (platform: str, user_id: int)
            # 我们需要确保 user_id_str 被正确地转换为 int (如果它是数字的话)
            try:
                user_id_for_get_person_id = int(user_id_str)
            except ValueError:
                logger.error(f"MarkGenderTool: user_id '{user_id_str}' 无法转换为整数，无法获取 person_id。")
                return {"type": "gender_mark_update", "id": user_id_str, "content": f"标记性别失败：用户ID '{user_id_str}' 格式无效。"}

            person_id = person_info_manager.get_person_id(platform, user_id_for_get_person_id)

            if not person_id:
                logger.error(f"MarkGenderTool: 无法为 platform='{platform}', user_id='{user_id_str}' 获取 person_id。")
                return {"type": "gender_mark_update", "id": user_id_str, "content": f"标记性别失败：无法找到用户 {user_id_str}。"}

            # 使用 PersonInfoManager 的 update_one_field 方法
            # 这个方法需要 person_id, 字段名, 新值, 和一个可选的 data 字典 (用于创建新用户时)
            # 因为我们是更新现有用户的字段，data 可以不传或传一个包含平台和用户ID的最小字典
            # 确保用户存在，如果不存在，update_one_field 应该能处理（它内部会调用 create_person_info）
            user_data_for_update = {
                "platform": platform,
                "user_id": user_id_for_get_person_id # 传递整数型 user_id
            }
            await person_info_manager.update_one_field(person_id, "gender_mark", gender_mark_value, data=user_data_for_update)

            # person_info_manager.update_one_field 没有明确的成功/失败返回值，我们假设它如果没抛异常就是成功了
            logger.info(f"MarkGenderTool: 已请求将用户 {user_id_str} (PersonID: {person_id}) 的性别标记更新为 '{gender_mark_value}'。原因: {reasoning}")
            return {
                "type": "gender_mark_update_success", 
                "id": user_id_str, 
                "content": f"已记录用户 {user_id_str} 的性别为 '{gender_mark_value}'。"
            }

        except Exception as e:
            logger.error(f"MarkGenderTool: 执行工具时发生错误: {str(e)}", exc_info=True)
            return {"type": "gender_mark_update_error", "id": user_id_str, "content": f"标记性别时发生内部错误: {str(e)}"}

register_tool(MarkGenderTool)