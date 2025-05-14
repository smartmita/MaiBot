from typing import Dict, Optional
from dataclasses import dataclass
from .info_base import InfoBase


@dataclass
class ChatInfo(InfoBase):
    """聊天信息类

    用于记录和管理聊天相关的信息，包括聊天ID、名称和类型等。
    继承自 InfoBase 类，使用字典存储具体数据。

    Attributes:
        type (str): 信息类型标识符，固定为 "chat"

    Data Fields:
        chat_id (str): 聊天的唯一标识符
        chat_name (str): 聊天的名称
        chat_type (str): 聊天的类型
    """

    type: str = "chat"

    def set_chat_id(self, chat_id: str) -> None:
        """设置聊天ID

        Args:
            chat_id (str): 聊天的唯一标识符
        """
        self.data["chat_id"] = chat_id

    def set_chat_name(self, chat_name: str) -> None:
        """设置聊天名称

        Args:
            chat_name (str): 聊天的名称
        """
        self.data["chat_name"] = chat_name

    def set_chat_type(self, chat_type: str) -> None:
        """设置聊天类型

        Args:
            chat_type (str): 聊天的类型
        """
        self.data["chat_type"] = chat_type

    def get_chat_id(self) -> Optional[str]:
        """获取聊天ID

        Returns:
            Optional[str]: 聊天的唯一标识符，如果未设置则返回 None
        """
        return self.get_info("chat_id")

    def get_chat_name(self) -> Optional[str]:
        """获取聊天名称

        Returns:
            Optional[str]: 聊天的名称，如果未设置则返回 None
        """
        return self.get_info("chat_name")

    def get_chat_type(self) -> Optional[str]:
        """获取聊天类型

        Returns:
            Optional[str]: 聊天的类型，如果未设置则返回 None
        """
        return self.get_info("chat_type")

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
