"""
PFC_idle 包 - 用于空闲时主动聊天的功能模块

该包包含以下主要组件：
- IdleChat: 根据关系和活跃度进行智能主动聊天
- IdleChatManager: 管理多个聊天实例的空闲状态
- IdleConversation: 处理与空闲聊天相关的功能，与主Conversation类解耦
"""

from .idle_chat import IdleChat
from .idle_chat_manager import IdleChatManager
from .idle_conversation import IdleConversation, get_idle_conversation_instance, initialize_idle_conversation

__all__ = [
    'IdleChat',
    'IdleChatManager',
    'IdleConversation',
    'get_idle_conversation_instance',
    'initialize_idle_conversation'
] 