from typing import Dict, Optional
import asyncio
from src.common.logger_manager import get_logger
from .idle_chat import IdleChat
import traceback

logger = get_logger("pfc_idle_chat_manager")

class IdleChatManager:
    """空闲聊天管理器
    
    用于管理所有私聊用户的空闲聊天实例。
    采用单例模式，确保全局只有一个管理器实例。
    """
    
    _instance: Optional["IdleChatManager"] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __init__(self):
        """初始化空闲聊天管理器"""
        self._idle_chats: Dict[str, IdleChat] = {}  # stream_id -> IdleChat
        self._active_conversations_count: Dict[str, int] = {}  # stream_id -> count
        
    @classmethod
    def get_instance(cls) -> "IdleChatManager":
        """获取管理器单例 (同步版本)

        Returns:
            IdleChatManager: 管理器实例
        """
        if not cls._instance:
            # 在同步环境中创建实例
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def get_instance_async(cls) -> "IdleChatManager":
        """获取管理器单例 (异步版本)

        Returns:
            IdleChatManager: 管理器实例
        """
        if not cls._instance:
            async with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance
    
    async def get_or_create_idle_chat(self, stream_id: str, private_name: str) -> IdleChat:
        """获取或创建空闲聊天实例

        Args:
            stream_id: 聊天流ID
            private_name: 私聊用户名称

        Returns:
            IdleChat: 空闲聊天实例
        """
        if stream_id not in self._idle_chats:
            idle_chat = IdleChat(stream_id, private_name)
            self._idle_chats[stream_id] = idle_chat
            # 初始化活跃对话计数
            if stream_id not in self._active_conversations_count:
                self._active_conversations_count[stream_id] = 0
            idle_chat.start()  # 启动空闲检测
            logger.info(f"[私聊][{private_name}]创建并启动新的空闲聊天实例")
        return self._idle_chats[stream_id]
    
    async def remove_idle_chat(self, stream_id: str) -> None:
        """移除空闲聊天实例

        Args:
            stream_id: 聊天流ID
        """
        if stream_id in self._idle_chats:
            idle_chat = self._idle_chats[stream_id]
            idle_chat.stop()  # 停止空闲检测
            del self._idle_chats[stream_id]
            if stream_id in self._active_conversations_count:
                del self._active_conversations_count[stream_id]
            logger.info(f"[私聊][{idle_chat.private_name}]移除空闲聊天实例")
    
    async def notify_conversation_start(self, stream_id: str) -> None:
        """通知对话开始

        Args:
            stream_id: 聊天流ID
        """
        try:
            if stream_id not in self._idle_chats:
                logger.warning(f"对话开始通知: {stream_id} 没有对应的IdleChat实例，将创建一个")
                # 从stream_id尝试提取private_name
                private_name = stream_id
                if stream_id.startswith("private_"):
                    parts = stream_id.split("_")
                    if len(parts) >= 2:
                        private_name = parts[1]  # 取第二部分作为名称
                await self.get_or_create_idle_chat(stream_id, private_name)
            
            if stream_id not in self._active_conversations_count:
                self._active_conversations_count[stream_id] = 0
            
            # 增加计数前记录当前值，用于日志
            old_count = self._active_conversations_count[stream_id]
            self._active_conversations_count[stream_id] += 1
            new_count = self._active_conversations_count[stream_id]
            
            # 确保IdleChat实例存在
            idle_chat = self._idle_chats.get(stream_id)
            if idle_chat:
                await idle_chat.increment_active_instances()
                logger.debug(f"对话开始通知: {stream_id}, 计数从{old_count}增加到{new_count}")
            else:
                logger.error(f"对话开始通知: {stream_id}, 计数增加但IdleChat不存在! 计数:{old_count}->{new_count}")
        except Exception as e:
            logger.error(f"对话开始通知处理失败: {stream_id}, 错误: {e}")
            logger.error(traceback.format_exc())
    
    async def notify_conversation_end(self, stream_id: str) -> None:
        """通知对话结束

        Args:
            stream_id: 聊天流ID
        """
        try:
            # 记录当前计数用于日志
            old_count = self._active_conversations_count.get(stream_id, 0)
            
            # 安全减少计数，避免负数
            if stream_id in self._active_conversations_count and self._active_conversations_count[stream_id] > 0:
                self._active_conversations_count[stream_id] -= 1
            else:
                # 如果计数已经为0或不存在，设置为0
                self._active_conversations_count[stream_id] = 0
                
            new_count = self._active_conversations_count.get(stream_id, 0)
            
            # 确保IdleChat实例存在
            idle_chat = self._idle_chats.get(stream_id)
            if idle_chat:
                await idle_chat.decrement_active_instances()
                logger.debug(f"对话结束通知: {stream_id}, 计数从{old_count}减少到{new_count}")
            else:
                logger.warning(f"对话结束通知: {stream_id}, 计数减少但IdleChat不存在! 计数:{old_count}->{new_count}")
                
            # 检查是否所有对话都结束了，帮助调试
            all_counts = sum(self._active_conversations_count.values())
            if all_counts == 0:
                logger.info(f"所有对话实例都已结束，当前总活跃计数为0")
        except Exception as e:
            logger.error(f"对话结束通知处理失败: {stream_id}, 错误: {e}")
            logger.error(traceback.format_exc())
    
    def get_idle_chat(self, stream_id: str) -> Optional[IdleChat]:
        """获取空闲聊天实例

        Args:
            stream_id: 聊天流ID

        Returns:
            Optional[IdleChat]: 空闲聊天实例，如果不存在则返回None
        """
        return self._idle_chats.get(stream_id)
    
    def get_active_conversations_count(self, stream_id: str) -> int:
        """获取指定流的活跃对话计数

        Args:
            stream_id: 聊天流ID

        Returns:
            int: 活跃对话计数
        """
        return self._active_conversations_count.get(stream_id, 0)
    
    def get_all_active_conversations_count(self) -> int:
        """获取所有活跃对话总计数

        Returns:
            int: 活跃对话总计数
        """
        return sum(self._active_conversations_count.values()) 