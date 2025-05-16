import time
import os
from typing import List, Optional, Dict, Any


class CycleDetail:
    """循环信息记录类"""

    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.thinking_id = ""
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.timers: Dict[str, float] = {}

        # 新字段
        self.loop_observation_info: Dict[str, Any] = {}
        self.loop_process_info: Dict[str, Any] = {}
        self.loop_plan_info: Dict[str, Any] = {}
        self.loop_action_info: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """将循环信息转换为字典格式"""
        return {
            "cycle_id": self.cycle_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timers": self.timers,
            "thinking_id": self.thinking_id,
            "loop_observation_info": self.loop_observation_info,
            "loop_process_info": self.loop_process_info,
            "loop_plan_info": self.loop_plan_info,
            "loop_action_info": self.loop_action_info,
        }

    def complete_cycle(self):
        """完成循环，记录结束时间"""
        self.end_time = time.time()

    def set_thinking_id(self, thinking_id: str):
        """设置思考消息ID"""
        self.thinking_id = thinking_id

    def set_loop_info(self, loop_info: Dict[str, Any]):
        """设置循环信息"""
        self.loop_observation_info = loop_info["loop_observation_info"]
        self.loop_processor_info = loop_info["loop_processor_info"]
        self.loop_plan_info = loop_info["loop_plan_info"]
        self.loop_action_info = loop_info["loop_action_info"]

    @staticmethod
    def list_cycles(stream_id: str, base_dir: str = "log_debug") -> List[str]:
        """
        列出指定stream_id的所有循环文件

        参数:
            stream_id: 聊天流ID
            base_dir: 基础目录，默认为log_debug

        返回:
            List[str]: 文件路径列表
        """
        try:
            stream_dir = os.path.join(base_dir, stream_id)
            if not os.path.exists(stream_dir):
                return []

            files = [
                os.path.join(stream_dir, f)
                for f in os.listdir(stream_dir)
                if f.startswith("cycle_") and f.endswith(".txt")
            ]
            return sorted(files)
        except Exception as e:
            print(f"列出循环文件时出错: {e}")
            return []
