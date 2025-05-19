from enum import Enum
from typing import Literal


class ConversationState(Enum):
    """对话状态"""

    INIT = "初始化"
    RETHINKING = "重新思考"
    ANALYZING = "分析历史"
    PLANNING = "规划目标"
    GENERATING = "生成回复"
    CHECKING = "检查回复"
    SENDING = "发送消息"
    FETCHING = "获取知识"
    WAITING = "等待"
    LISTENING = "倾听"
    ENDED = "结束"
    JUDGING = "判断"
    IGNORED = "屏蔽"
    ERROR = "错误"


ActionType = Literal[
    "direct_reply",
    "send_new_message",
    "send_memes",
    "wait",
    "listening",
    "rethink_goal",
    "end_conversation",
    "block_and_ignore",
    "say_goodbye",
    "reply_after_wait_timeout"  # <--- 新增动作类型
]
