from typing import Optional


class ConversationInfo:
    def __init__(self):
        self.done_action = []
        self.goal_list = []
        self.knowledge_list = []
        self.memory_list = []
        self.last_successful_reply_action: Optional[str] = None
        self.last_reply_rejection_reason: Optional[str] = None  # 用于存储上次回复被拒原因
        self.last_rejected_reply_content: Optional[str] = None  # 用于存储上次被拒的回复内容
        self.my_message_count: int = 0  # 用于存储连续发送了多少条消息