from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type
from src.common.logger_manager import get_logger

logger = get_logger("base_action")

# 全局动作注册表
_ACTION_REGISTRY: Dict[str, Type["BaseAction"]] = {}
_DEFAULT_ACTIONS: Dict[str, str] = {}


def register_action(cls):
    """
    动作注册装饰器
    
    用法:
        @register_action
        class MyAction(BaseAction):
            action_name = "my_action"
            action_description = "我的动作"
            ...
    """
    # 检查类是否有必要的属性
    if not hasattr(cls, "action_name") or not hasattr(cls, "action_description"):
        logger.error(f"动作类 {cls.__name__} 缺少必要的属性: action_name 或 action_description")
        return cls
    
    action_name = getattr(cls, "action_name")
    action_description = getattr(cls, "action_description")
    is_default = getattr(cls, "default", False)
    
    if not action_name or not action_description:
        logger.error(f"动作类 {cls.__name__} 的 action_name 或 action_description 为空")
        return cls
    
    # 将动作类注册到全局注册表
    _ACTION_REGISTRY[action_name] = cls
    
    # 如果是默认动作，添加到默认动作集
    if is_default:
        _DEFAULT_ACTIONS[action_name] = action_description
    
    logger.info(f"已注册动作: {action_name} -> {cls.__name__}，默认: {is_default}")
    return cls


class BaseAction(ABC):
    """动作基类接口

    所有具体的动作类都应该继承这个基类，并实现handle_action方法。
    """

    def __init__(self, action_data: dict, reasoning: str, cycle_timers: dict, thinking_id: str):
        """初始化动作

        Args:
            action_name: 动作名称
            action_data: 动作数据
            reasoning: 执行该动作的理由
            cycle_timers: 计时器字典
            thinking_id: 思考ID
        """
        #每个动作必须实现
        self.action_name:str = "base_action"
        self.action_description:str = "基础动作"
        self.action_parameters:dict = {}
        self.action_require:list[str] = []
        
        self.default:bool = False
        
        
        self.action_data = action_data
        self.reasoning = reasoning
        self.cycle_timers = cycle_timers
        self.thinking_id = thinking_id

    @abstractmethod
    async def handle_action(self) -> Tuple[bool, str]:
        """处理动作的抽象方法，需要被子类实现

        Returns:
            Tuple[bool, str]: (是否执行成功, 回复文本)
        """
        pass
