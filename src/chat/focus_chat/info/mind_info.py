from typing import Dict, Any
from dataclasses import dataclass, field
from .info_base import InfoBase


@dataclass
class MindInfo(InfoBase):
    """思维信息类

    用于存储和管理当前思维状态的信息。

    Attributes:
        type (str): 信息类型标识符，默认为 "mind"
        data (Dict[str, Any]): 包含 current_mind 的数据字典
    """

    type: str = "mind"
    data: Dict[str, Any] = field(default_factory=lambda: {"current_mind": ""})

    def get_current_mind(self) -> str:
        """获取当前思维状态

        Returns:
            str: 当前思维状态
        """
        return self.get_info("current_mind") or ""

    def set_current_mind(self, mind: str) -> None:
        """设置当前思维状态

        Args:
            mind: 要设置的思维状态
        """
        self.data["current_mind"] = mind
