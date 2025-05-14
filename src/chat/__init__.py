"""
MaiMBot插件系统
包含聊天、情绪、记忆、日程等功能模块
"""

from src.chat.message_receive.chat_stream import chat_manager
from src.chat.emoji_system.emoji_manager import emoji_manager
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.normal_chat.willing.willing_manager import willing_manager

# 导出主要组件供外部使用
__all__ = [
    "chat_manager",
    "emoji_manager",
    "relationship_manager",
    "willing_manager",
]
