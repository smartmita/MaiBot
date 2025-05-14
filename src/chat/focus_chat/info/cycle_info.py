from typing import Dict, Optional, Any
from dataclasses import dataclass
from .info_base import InfoBase


@dataclass
class CycleInfo(InfoBase):
    """循环信息类

    用于记录和管理心跳循环的相关信息，包括循环ID、时间信息、动作信息等。
    继承自 InfoBase 类，使用字典存储具体数据。

    Attributes:
        type (str): 信息类型标识符，固定为 "cycle"

    Data Fields:
        cycle_id (str): 当前循环的唯一标识符
        start_time (str): 循环开始的时间
        end_time (str): 循环结束的时间
        action (str): 在循环中采取的动作
        action_data (Dict[str, Any]): 动作相关的详细数据
        reason (str): 触发循环的原因
        observe_info (str): 当前的回复信息
    """

    type: str = "cycle"

    def get_type(self) -> str:
        """获取信息类型"""
        return self.type

    def get_data(self) -> Dict[str, str]:
        """获取信息数据"""
        return self.data

    def get_info(self, key: str) -> Optional[str]:
        """获取特定属性的信息

        Args:
            key: 要获取的属性键名

        Returns:
            属性值，如果键不存在则返回 None
        """
        return self.data.get(key)

    def set_cycle_id(self, cycle_id: str) -> None:
        """设置循环ID

        Args:
            cycle_id (str): 循环的唯一标识符
        """
        self.data["cycle_id"] = cycle_id

    def set_start_time(self, start_time: str) -> None:
        """设置开始时间

        Args:
            start_time (str): 循环开始的时间，建议使用标准时间格式
        """
        self.data["start_time"] = start_time

    def set_end_time(self, end_time: str) -> None:
        """设置结束时间

        Args:
            end_time (str): 循环结束的时间，建议使用标准时间格式
        """
        self.data["end_time"] = end_time

    def set_action(self, action: str) -> None:
        """设置采取的动作

        Args:
            action (str): 在循环中执行的动作名称
        """
        self.data["action"] = action

    def set_action_data(self, action_data: Dict[str, Any]) -> None:
        """设置动作数据

        Args:
            action_data (Dict[str, Any]): 动作相关的详细数据，将被转换为字符串存储
        """
        self.data["action_data"] = str(action_data)

    def set_reason(self, reason: str) -> None:
        """设置原因

        Args:
            reason (str): 触发循环的原因说明
        """
        self.data["reason"] = reason

    def set_observe_info(self, observe_info: str) -> None:
        """设置回复信息

        Args:
            observe_info (str): 当前的回复信息
        """
        self.data["observe_info"] = observe_info

    def get_cycle_id(self) -> Optional[str]:
        """获取循环ID

        Returns:
            Optional[str]: 循环的唯一标识符，如果未设置则返回 None
        """
        return self.get_info("cycle_id")

    def get_start_time(self) -> Optional[str]:
        """获取开始时间

        Returns:
            Optional[str]: 循环开始的时间，如果未设置则返回 None
        """
        return self.get_info("start_time")

    def get_end_time(self) -> Optional[str]:
        """获取结束时间

        Returns:
            Optional[str]: 循环结束的时间，如果未设置则返回 None
        """
        return self.get_info("end_time")

    def get_action(self) -> Optional[str]:
        """获取采取的动作

        Returns:
            Optional[str]: 在循环中执行的动作名称，如果未设置则返回 None
        """
        return self.get_info("action")

    def get_action_data(self) -> Optional[str]:
        """获取动作数据

        Returns:
            Optional[str]: 动作相关的详细数据（字符串形式），如果未设置则返回 None
        """
        return self.get_info("action_data")

    def get_reason(self) -> Optional[str]:
        """获取原因

        Returns:
            Optional[str]: 触发循环的原因说明，如果未设置则返回 None
        """
        return self.get_info("reason")

    def get_observe_info(self) -> Optional[str]:
        """获取回复信息

        Returns:
            Optional[str]: 当前的回复信息，如果未设置则返回 None
        """
        return self.get_info("observe_info")
