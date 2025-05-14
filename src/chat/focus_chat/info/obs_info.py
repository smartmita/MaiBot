from typing import Dict, Optional
from dataclasses import dataclass
from .info_base import InfoBase


@dataclass
class ObsInfo(InfoBase):
    """OBS信息类

    用于记录和管理OBS相关的信息，包括说话消息、截断后的说话消息和聊天类型。
    继承自 InfoBase 类，使用字典存储具体数据。

    Attributes:
        type (str): 信息类型标识符，固定为 "obs"

    Data Fields:
        talking_message (str): 说话消息内容
        talking_message_str_truncate (str): 截断后的说话消息内容
        chat_type (str): 聊天类型，可以是 "private"（私聊）、"group"（群聊）或 "other"（其他）
    """

    type: str = "obs"

    def set_talking_message(self, message: str) -> None:
        """设置说话消息

        Args:
            message (str): 说话消息内容
        """
        self.data["talking_message"] = message

    def set_talking_message_str_truncate(self, message: str) -> None:
        """设置截断后的说话消息

        Args:
            message (str): 截断后的说话消息内容
        """
        self.data["talking_message_str_truncate"] = message

    def set_previous_chat_info(self, message: str) -> None:
        """设置之前聊天信息

        Args:
            message (str): 之前聊天信息内容
        """
        self.data["previous_chat_info"] = message

    def set_chat_type(self, chat_type: str) -> None:
        """设置聊天类型

        Args:
            chat_type (str): 聊天类型，可以是 "private"（私聊）、"group"（群聊）或 "other"（其他）
        """
        if chat_type not in ["private", "group", "other"]:
            chat_type = "other"
        self.data["chat_type"] = chat_type

    def set_chat_target(self, chat_target: str) -> None:
        """设置聊天目标

        Args:
            chat_target (str): 聊天目标，可以是 "private"（私聊）、"group"（群聊）或 "other"（其他）
        """
        self.data["chat_target"] = chat_target

    def get_talking_message(self) -> Optional[str]:
        """获取说话消息

        Returns:
            Optional[str]: 说话消息内容，如果未设置则返回 None
        """
        return self.get_info("talking_message")

    def get_talking_message_str_truncate(self) -> Optional[str]:
        """获取截断后的说话消息

        Returns:
            Optional[str]: 截断后的说话消息内容，如果未设置则返回 None
        """
        return self.get_info("talking_message_str_truncate")

    def get_chat_type(self) -> str:
        """获取聊天类型

        Returns:
            str: 聊天类型，默认为 "other"
        """
        return self.get_info("chat_type") or "other"

    def get_type(self) -> str:
        """获取信息类型

        Returns:
            str: 当前信息对象的类型标识符
        """
        return self.type

    def get_data(self) -> Dict[str, str]:
        """获取所有信息数据

        Returns:
            Dict[str, str]: 包含所有信息数据的字典
        """
        return self.data

    def get_info(self, key: str) -> Optional[str]:
        """获取特定属性的信息

        Args:
            key: 要获取的属性键名

        Returns:
            Optional[str]: 属性值，如果键不存在则返回 None
        """
        return self.data.get(key)
