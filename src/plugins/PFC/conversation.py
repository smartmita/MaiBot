import time
import asyncio
import traceback
from typing import Dict, Any, Optional

from src.common.logger_manager import get_logger
from maim_message import UserInfo
from src.plugins.chat.chat_stream import chat_manager, ChatStream
from ..chat.message import Message  # 假设 Message 类型被 _convert_to_message 使用
from src.config.config import global_config
from ..person_info.person_info import person_info_manager
from ..person_info.relationship_manager import relationship_manager
from ..moods.moods import MoodManager

from .pfc_relationship import PfcRelationshipUpdater, PfcRepationshipTranslator
from .pfc_emotion import PfcEmotionUpdater

# 导入 PFC 内部组件和类型
from .pfc_types import ConversationState
from .pfc import GoalAnalyzer
from .chat_observer import ChatObserver
from .message_sender import DirectMessageSender
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from .reply_generator import ReplyGenerator
from .PFC_idle.idle_chat import IdleChat
from .pfc_KnowledgeFetcher import KnowledgeFetcher
from .waiter import Waiter
from .reply_checker import ReplyChecker

# >>> 新增导入 <<<
from .conversation_loop import run_conversation_loop  # 导入新的循环函数

from rich.traceback import install

install(extra_lines=3)

logger = get_logger("pfc_conversation")

# 时区配置移到 loop 文件或更全局的位置，这里不再需要
# configured_tz = getattr(global_config, 'TIME_ZONE', 'Asia/Shanghai')
# TIME_ZONE = tz.gettz(configured_tz)
# if TIME_ZONE is None:
#     logger.error(f"配置的时区 '{configured_tz}' 无效，将使用默认时区 'Asia/Shanghai'")
#     TIME_ZONE = tz.gettz('Asia/Shanghai')


class Conversation:
    """
    对话类，负责管理单个私聊对话的状态和核心逻辑流程。
    """

    def __init__(self, stream_id: str, private_name: str):
        """
        初始化对话实例的基本属性。
        核心组件的实例化将由 PFCManager 通过调用 conversation_initializer 中的函数完成。
        """
        self.stream_id: str = stream_id
        self.private_name: str = private_name
        self.state: ConversationState = ConversationState.INIT
        self.should_continue: bool = False  # Manager 会在初始化后设置
        self.ignore_until_timestamp: Optional[float] = None
        self.generated_reply: str = ""
        self.chat_stream: Optional[ChatStream] = None

        self.person_info_mng = person_info_manager
        self.relationship_mng = relationship_manager
        self.mood_mng = MoodManager.get_instance()

        self.relationship_updater: Optional[PfcRelationshipUpdater] = None
        self.relationship_translator: Optional[PfcRepationshipTranslator] = None
        self.emotion_updater: Optional[PfcEmotionUpdater] = None
        self.action_planner: Optional[ActionPlanner] = None
        self.goal_analyzer: Optional[GoalAnalyzer] = None
        self.reply_generator: Optional[ReplyGenerator] = None
        self.knowledge_fetcher: Optional[KnowledgeFetcher] = None
        self.waiter: Optional[Waiter] = None
        self.direct_sender: Optional[DirectMessageSender] = None
        self.idle_chat: Optional[IdleChat] = None
        self.chat_observer: Optional[ChatObserver] = None
        self.observation_info: Optional[ObservationInfo] = None
        self.conversation_info: Optional[ConversationInfo] = None
        self.reply_checker: Optional[ReplyChecker] = None

        self._initialized: bool = False  # Manager 会在初始化成功后设为 True

        self.bot_qq_str: Optional[str] = str(global_config.BOT_QQ) if global_config.BOT_QQ else None
        if not self.bot_qq_str:
            logger.error(f"[私聊][{self.private_name}] 严重错误：未能从配置中获取 BOT_QQ ID！")

    # _initialize 和 _load_initial_history 方法已被移除

    async def start(self):
        """
        启动对话流程。创建并启动核心的规划与行动循环 (`run_conversation_loop`)。
        """
        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 对话实例未被 Manager 正确初始化，无法启动规划循环。")
            return

        if not self.should_continue:
            logger.warning(f"[私聊][{self.private_name}] 对话实例已被 Manager 标记为不应继续，无法启动规划循环。")
            return

        logger.info(f"[私聊][{self.private_name}] 对话系统启动，准备创建规划循环任务...")
        try:
            # >>> 修改后的调用 <<<
            _loop_task = asyncio.create_task(run_conversation_loop(self))
            logger.info(f"[私聊][{self.private_name}] 规划循环任务已创建。")
        except Exception as task_err:
            logger.error(f"[私聊][{self.private_name}] 创建规划循环任务时出错: {task_err}")
            await self.stop()

    async def stop(self):
        """
        停止对话实例并清理相关资源。
        """
        logger.info(f"[私聊][{self.private_name}] 正在停止对话实例: {self.stream_id}")
        self.should_continue = False

        # 最终关系评估
        if (
            self._initialized
            and self.relationship_updater
            and self.conversation_info
            and self.observation_info
            and self.chat_observer
        ):
            try:
                logger.info(f"[私聊][{self.private_name}] 准备执行最终关系评估...")
                await self.relationship_updater.update_relationship_final(
                    conversation_info=self.conversation_info,
                    observation_info=self.observation_info,
                    chat_observer_for_history=self.chat_observer,
                )
                logger.info(f"[私聊][{self.private_name}] 最终关系评估已调用。")
            except Exception as e_final_rel:
                logger.error(f"[私聊][{self.private_name}] 调用最终关系评估时出错: {e_final_rel}")
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"[私聊][{self.private_name}] 跳过最终关系评估，实例未完全初始化或缺少组件。")

        # 停止其他组件
        if self.idle_chat:
            # 减少活跃实例计数，而不是停止IdleChat
            await self.idle_chat.decrement_active_instances()
            logger.info(f"[私聊][{self.private_name}] 已减少IdleChat活跃实例计数")
        if self.observation_info and self.chat_observer:
            self.observation_info.unbind_from_chat_observer()
        if self.mood_mng and hasattr(self.mood_mng, "stop_mood_update") and self.mood_mng._running:  # type: ignore
            self.mood_mng.stop_mood_update()  # type: ignore
            logger.info(f"[私聊][{self.private_name}] MoodManager 后台更新已停止。")

        self._initialized = False  # 标记为未初始化
        logger.info(f"[私聊][{self.private_name}] 对话实例 {self.stream_id} 已停止。")

    # _plan_and_action_loop 方法已被移除

    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Optional[Message]:
        """将从数据库或其他来源获取的消息字典转换为内部使用的 Message 对象"""
        # 这个方法似乎没有被其他内部方法调用，但为了完整性暂时保留
        try:
            chat_stream_to_use = self.chat_stream or chat_manager.get_stream(self.stream_id)
            if not chat_stream_to_use:
                logger.error(
                    f"[私聊][{self.private_name}] 无法确定 ChatStream for stream_id {self.stream_id}，无法转换消息。"
                )
                return None

            user_info_dict = msg_dict.get("user_info", {})
            user_info: Optional[UserInfo] = None
            if isinstance(user_info_dict, dict):
                try:
                    user_info = UserInfo.from_dict(user_info_dict)
                except Exception as e:
                    logger.warning(
                        f"[私聊][{self.private_name}] 从字典创建 UserInfo 时出错: {e}, dict: {user_info_dict}"
                    )
            if not user_info:
                logger.warning(
                    f"[私聊][{self.private_name}] 消息缺少有效的 UserInfo，无法转换。 msg_id: {msg_dict.get('message_id')}"
                )
                return None

            return Message(
                message_id=msg_dict.get("message_id", f"gen_{time.time()}"),
                chat_stream=chat_stream_to_use,
                time=msg_dict.get("time", time.time()),
                user_info=user_info,
                processed_plain_text=msg_dict.get("processed_plain_text", ""),
                detailed_plain_text=msg_dict.get("detailed_plain_text", ""),
            )
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}] 转换消息时出错: {e}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            return None
