from typing import Optional, List, Dict, Any


class ConversationInfo:
    def __init__(self):
        self.done_action: List[Dict[str, Any]] = []  # 建议明确类型
        self.goal_list: List[Dict[str, Any]] = []  # 建议明确类型
        self.knowledge_list: List[Any] = []  # 建议明确类型
        self.memory_list: List[Any] = []  # 建议明确类型
        self.last_successful_reply_action: Optional[str] = None
        self.last_reply_rejection_reason: Optional[str] = None  # 用于存储上次回复被拒原因
        self.last_rejected_reply_content: Optional[str] = None  # 用于存储上次被拒的回复内容
        self.my_message_count: int = 0  # 用于存储连续发送了多少条消息
        self.person_id: Optional[str] = None  # 私聊对象的唯一ID
        self.relationship_text: Optional[str] = "你们还不熟悉。"  # 与当前对话者的关系描述文本
        self.current_emotion_text: Optional[str] = "心情平静。"  # 机器人当前的情绪描述文本
        self.current_instance_message_count: int = 0  # 当前私聊实例中的消息计数
        self.other_new_messages_during_planning_count: int = 0  # 在计划阶段期间收到的其他新消息计数
        self.current_emoji_query: Optional[str] = None  # 表情包
        self.wait_has_timed_out: bool = False  # 标记上一个 wait 动作是否超时
        self.last_wait_duration_minutes: Optional[float] = None # 上一次等待超时的时长（分钟）
        self.current_pfc_thought: Optional[str] = None
        self.pfc_structured_info: Dict[str, Any] = {} # 用来存储SubMind输出的结构化信息
        self.previous_pfc_thought: Optional[str] = None # 用于存储上一轮SubMind的想法，供下一轮参考
        self.retrieved_historical_chat_for_submind: Optional[str] = None # 存储给SubMind的回想历史聊天记录
