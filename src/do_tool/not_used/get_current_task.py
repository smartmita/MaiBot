from src.do_tool.tool_can_use.base_tool import BaseTool
from src.plugins.schedule.schedule_generator import bot_schedule
from src.common.logger import get_module_logger
from typing import Any
from datetime import datetime

logger = get_module_logger("get_current_task_tool")


class GetCurrentTaskTool(BaseTool):
    """获取当前正在做的事情/最近的任务工具"""

    name = "get_schedule"
    description = "获取当前正在做的事情，或者某个时间点/时间段的日程信息"
    parameters = {
        "type": "object",
        "properties": {
            "start_time": {"type": "string", "description": "开始时间，格式为'HH:MM'，填写current则获取当前任务"},
            "end_time": {"type": "string", "description": "结束时间，格式为'HH:MM'，填写current则获取当前任务"},
        },
        "required": ["start_time", "end_time"],
    }

    async def execute(self, function_args: dict[str, Any], message_txt: str = "") -> dict[str, Any]:
        """执行获取当前任务或指定时间段的日程信息

        Args:
            function_args: 工具参数
            message_txt: 原始消息文本，此工具不使用

        Returns:
            dict: 工具执行结果
        """
        start_time = function_args.get("start_time")
        end_time = function_args.get("end_time")

        # 如果 start_time 或 end_time 为 "current"，则获取当前任务
        if start_time == "current" or end_time == "current":
            current_task = bot_schedule.get_current_num_task(num=1, time_info=True)
            current_time = datetime.now().strftime("%H:%M:%S")
            current_date = datetime.now().strftime("%Y-%m-%d")
            if current_task:
                task_info = f"{current_date} {current_time}，你在{current_task}"
            else:
                task_info = f"{current_time} {current_date}，没在做任何事情"
        # 如果提供了时间范围，则获取该时间段的日程信息
        elif start_time and end_time:
            tasks = await bot_schedule.get_task_from_time_to_time(start_time, end_time)
            if tasks:
                task_list = []
                for task in tasks:
                    task_time = task[0].strftime("%H:%M")
                    task_content = task[1]
                    task_list.append(f"{task_time}时，{task_content}")
                task_info = "\n".join(task_list)
            else:
                task_info = f"在 {start_time} 到 {end_time} 之间没有找到日程信息"
        else:
            task_info = "请提供有效的开始时间和结束时间"
        return {"name": "get_current_task", "content": f"日程信息: {task_info}"}
