from typing import Dict, List, Optional, Callable, Coroutine, Type, Any, Union
import os
import importlib
from src.chat.focus_chat.planners.actions.base_action import BaseAction, _ACTION_REGISTRY, _DEFAULT_ACTIONS
from src.chat.heart_flow.observation.observation import Observation
from src.chat.focus_chat.expressors.default_expressor import DefaultExpressor
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.focus_chat.heartFC_Cycleinfo import CycleDetail
from src.common.logger_manager import get_logger

# 导入动作类，确保装饰器被执行
from src.chat.focus_chat.planners.actions.reply_action import ReplyAction
from src.chat.focus_chat.planners.actions.no_reply_action import NoReplyAction

logger = get_logger("action_factory")

# 定义动作信息类型
ActionInfo = Dict[str, Any]


class ActionManager:
    """
    动作管理器，用于管理各种类型的动作
    """

    def __init__(self):
        """初始化动作管理器"""
        # 所有注册的动作集合
        self._registered_actions: Dict[str, ActionInfo] = {}
        # 当前正在使用的动作集合，默认加载默认动作
        self._using_actions: Dict[str, ActionInfo] = {}
        # 临时备份原始使用中的动作
        self._original_actions_backup: Optional[Dict[str, ActionInfo]] = None
        
        # 默认动作集，仅作为快照，用于恢复默认
        self._default_actions: Dict[str, ActionInfo] = {}
        
        # 加载所有已注册动作
        self._load_registered_actions()
        
        # 初始化时将默认动作加载到使用中的动作
        self._using_actions = self._default_actions.copy()
        
        # logger.info(f"当前可用动作: {list(self._using_actions.keys())}")
        # for action_name, action_info in self._using_actions.items():
            # logger.info(f"动作名称: {action_name}, 动作信息: {action_info}")
            

    def _load_registered_actions(self) -> None:
        """
        加载所有通过装饰器注册的动作
        """
        try:
            # 从_ACTION_REGISTRY获取所有已注册动作
            for action_name, action_class in _ACTION_REGISTRY.items():
                # 获取动作相关信息
                action_description:str = getattr(action_class, "action_description", "")
                action_parameters:dict[str:str] = getattr(action_class, "action_parameters", {})
                action_require:list[str] = getattr(action_class, "action_require", [])
                is_default:bool = getattr(action_class, "default", False)
                
                if action_name and action_description:
                    # 创建动作信息字典
                    action_info = {
                        "description": action_description,
                        "parameters": action_parameters,
                        "require": action_require
                    }
                    
                    # 注册2
                    print("注册2")
                    print(action_info)
                    
                    # 添加到所有已注册的动作
                    self._registered_actions[action_name] = action_info
                    
                    # 添加到默认动作（如果是默认动作）
                    if is_default:
                        self._default_actions[action_name] = action_info
                
            logger.info(f"所有注册动作: {list(self._registered_actions.keys())}")
            logger.info(f"默认动作: {list(self._default_actions.keys())}")
            # for action_name, action_info in self._default_actions.items():
                # logger.info(f"动作名称: {action_name}, 动作信息: {action_info}")
            
        except Exception as e:
            logger.error(f"加载已注册动作失败: {e}")

    def create_action(
        self,
        action_name: str,
        action_data: dict,
        reasoning: str,
        cycle_timers: dict,
        thinking_id: str,
        observations: List[Observation],
        expressor: DefaultExpressor,
        chat_stream: ChatStream,
        current_cycle: CycleDetail,
        log_prefix: str,
        on_consecutive_no_reply_callback: Callable[[], Coroutine[None, None, None]],
        total_no_reply_count: int = 0,
        total_waiting_time: float = 0.0,
        shutting_down: bool = False,
    ) -> Optional[BaseAction]:
        """
        创建动作处理器实例

        Args:
            action_name: 动作名称
            action_data: 动作数据
            reasoning: 执行理由
            cycle_timers: 计时器字典
            thinking_id: 思考ID
            observations: 观察列表
            expressor: 表达器
            chat_stream: 聊天流
            current_cycle: 当前循环信息
            log_prefix: 日志前缀
            on_consecutive_no_reply_callback: 连续不回复回调
            total_no_reply_count: 连续不回复计数
            total_waiting_time: 累计等待时间
            shutting_down: 是否正在关闭

        Returns:
            Optional[BaseAction]: 创建的动作处理器实例，如果动作名称未注册则返回None
        """
        # 检查动作是否在当前使用的动作集中
        if action_name not in self._using_actions:
            logger.warning(f"当前不可用的动作类型: {action_name}")
            return None
            
        handler_class = _ACTION_REGISTRY.get(action_name)
        if not handler_class:
            logger.warning(f"未注册的动作类型: {action_name}")
            return None

        try:
            # 创建动作实例并传递所有必要参数
            instance = handler_class(
                action_name=action_name,
                action_data=action_data,
                reasoning=reasoning,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                observations=observations,
                on_consecutive_no_reply_callback=on_consecutive_no_reply_callback,
                current_cycle=current_cycle,
                log_prefix=log_prefix,
                total_no_reply_count=total_no_reply_count,
                total_waiting_time=total_waiting_time,
                shutting_down=shutting_down,
                expressor=expressor,
                chat_stream=chat_stream,
            )
            
            return instance

        except Exception as e:
            logger.error(f"创建动作处理器实例失败: {e}")
            return None

    def get_registered_actions(self) -> Dict[str, ActionInfo]:
        """获取所有已注册的动作集"""
        return self._registered_actions.copy()

    def get_default_actions(self) -> Dict[str, ActionInfo]:
        """获取默认动作集"""
        return self._default_actions.copy()
        
    def get_using_actions(self) -> Dict[str, ActionInfo]:
        """获取当前正在使用的动作集"""
        return self._using_actions.copy()

    def add_action_to_using(self, action_name: str) -> bool:
        """
        添加已注册的动作到当前使用的动作集
        
        Args:
            action_name: 动作名称
            
        Returns:
            bool: 添加是否成功
        """
        if action_name not in self._registered_actions:
            logger.warning(f"添加失败: 动作 {action_name} 未注册")
            return False
            
        if action_name in self._using_actions:
            logger.info(f"动作 {action_name} 已经在使用中")
            return True
            
        self._using_actions[action_name] = self._registered_actions[action_name]
        logger.info(f"添加动作 {action_name} 到使用集")
        return True

    def remove_action_from_using(self, action_name: str) -> bool:
        """
        从当前使用的动作集中移除指定动作
        
        Args:
            action_name: 动作名称
            
        Returns:
            bool: 移除是否成功
        """
        if action_name not in self._using_actions:
            logger.warning(f"移除失败: 动作 {action_name} 不在当前使用的动作集中")
            return False
            
        del self._using_actions[action_name]
        logger.info(f"已从使用集中移除动作 {action_name}")
        return True

    def add_action(self, action_name: str, description: str, parameters: Dict = None, require: List = None) -> bool:
        """
        添加新的动作到注册集
        
        Args:
            action_name: 动作名称
            description: 动作描述
            parameters: 动作参数定义，默认为空字典
            require: 动作依赖项，默认为空列表
            
        Returns:
            bool: 添加是否成功
        """
        if action_name in self._registered_actions:
            return False
            
        if parameters is None:
            parameters = {}
        if require is None:
            require = []
            
        action_info = {
            "description": description,
            "parameters": parameters,
            "require": require
        }
        
        self._registered_actions[action_name] = action_info
        return True

    def remove_action(self, action_name: str) -> bool:
        """从注册集移除指定动作"""
        if action_name not in self._registered_actions:
            return False
        del self._registered_actions[action_name]
        # 如果在使用集中也存在，一并移除
        if action_name in self._using_actions:
            del self._using_actions[action_name]
        return True

    def temporarily_remove_actions(self, actions_to_remove: List[str]) -> None:
        """临时移除使用集中的指定动作，备份原始使用集"""
        if self._original_actions_backup is None:
            self._original_actions_backup = self._using_actions.copy()
        for name in actions_to_remove:
            self._using_actions.pop(name, None)

    def restore_actions(self) -> None:
        """恢复之前备份的原始使用集"""
        if self._original_actions_backup is not None:
            self._using_actions = self._original_actions_backup.copy()
            self._original_actions_backup = None
    
    def restore_default_actions(self) -> None:
        """恢复默认动作集到使用集"""
        self._using_actions = self._default_actions.copy()
        self._original_actions_backup = None

    def get_action(self, action_name: str) -> Optional[Type[BaseAction]]:
        """
        获取指定动作的处理器类
        
        Args:
            action_name: 动作名称
            
        Returns:
            Optional[Type[BaseAction]]: 动作处理器类，如果不存在则返回None
        """
        return _ACTION_REGISTRY.get(action_name)


# 创建全局实例
ActionFactory = ActionManager()
