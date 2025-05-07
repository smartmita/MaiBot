import time
import asyncio
import datetime
import traceback
import json # 确保导入 json 模块
from typing import Dict, Any, Optional, Set, List
from dateutil import tz

from src.common.logger_manager import get_logger
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from maim_message import UserInfo
from src.plugins.chat.chat_stream import chat_manager, ChatStream
from ..chat.message import Message
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
from .idle_conversation_starter import IdleConversationStarter
from .pfc_KnowledgeFetcher import KnowledgeFetcher
from .waiter import Waiter
from .reply_checker import ReplyChecker

# 导入富文本回溯，用于更好的错误展示
from rich.traceback import install

install(extra_lines=3)

# 获取当前模块的日志记录器
logger = get_logger("pfc_conversation")

# 确保 global_config.TIME_ZONE 存在且有效，否则使用默认值
configured_tz = getattr(global_config, 'TIME_ZONE', 'Asia/Shanghai') # 使用 getattr 安全访问
TIME_ZONE = tz.gettz(configured_tz)
if TIME_ZONE is None: # 如果 gettz 返回 None，说明时区字符串无效
    logger.error(f"配置的时区 '{configured_tz}' 无效，将使用默认时区 'Asia/Shanghai'")
    TIME_ZONE = tz.gettz('Asia/Shanghai')

class Conversation:
    """
    对话类，负责管理单个私聊对话的状态和核心逻辑流程。
    包含对话的初始化、启动、停止、规划循环以及动作处理。
    """

    def __init__(self, stream_id: str, private_name: str):
        """
        初始化对话实例。

        Args:
            stream_id (str): 唯一的聊天流 ID。
            private_name (str): 私聊对象的名称，用于日志和区分。
        """
        self.stream_id: str = stream_id
        self.private_name: str = private_name
        self.state: ConversationState = ConversationState.INIT  # 对话的初始状态
        self.should_continue: bool = False  # 标记对话循环是否应该继续运行
        self.ignore_until_timestamp: Optional[float] = None  # 如果设置了，忽略此时间戳之前的活动
        self.generated_reply: str = ""  # 存储最近生成的回复内容
        self.chat_stream: Optional[ChatStream] = None  # 关联的聊天流对象

        # --- 新增：初始化管理器实例 ---
        self.person_info_mng = person_info_manager
        self.relationship_mng = relationship_manager
        self.mood_mng = MoodManager.get_instance() # MoodManager 是单例

        # 初始化所有核心组件为 None，将在 _initialize 中创建
        self.relationship_updater: Optional[PfcRelationshipUpdater] = None # 新增
        self.relationship_translator: Optional[PfcRepationshipTranslator] = None
        self.emotion_updater: Optional[PfcEmotionUpdater] = None       # 新增
        self.action_planner: Optional[ActionPlanner] = None
        self.goal_analyzer: Optional[GoalAnalyzer] = None
        self.reply_generator: Optional[ReplyGenerator] = None
        self.knowledge_fetcher: Optional[KnowledgeFetcher] = None
        self.waiter: Optional[Waiter] = None
        self.direct_sender: Optional[DirectMessageSender] = None
        self.idle_conversation_starter: Optional[IdleConversationStarter] = None
        self.chat_observer: Optional[ChatObserver] = None
        self.observation_info: Optional[ObservationInfo] = None
        self.conversation_info: Optional[ConversationInfo] = None
        self.reply_checker: Optional[ReplyChecker] = None  # 回复检查器

        # 内部状态标志
        self._initializing: bool = False  # 标记是否正在初始化，防止并发问题
        self._initialized: bool = False  # 标记是否已成功初始化

        # 缓存机器人自己的 QQ 号字符串，避免重复转换
        self.bot_qq_str: Optional[str] = str(global_config.BOT_QQ) if global_config.BOT_QQ else None
        if not self.bot_qq_str:
            # 这是一个严重问题，记录错误
            logger.error(f"[私聊][{self.private_name}] 严重错误：未能从配置中获取 BOT_QQ ID！PFC 可能无法正常工作。")

    async def _initialize(self):
        """
        异步初始化对话实例及其所有依赖的核心组件。
        这是一个关键步骤，确保所有部分都准备就绪才能开始对话循环。
        """
        # 防止重复初始化
        if self._initialized or self._initializing:
            logger.warning(f"[私聊][{self.private_name}] 尝试重复初始化或正在初始化中。")
            return

        self._initializing = True  # 标记开始初始化
        logger.info(f"[私聊][{self.private_name}] 开始初始化对话实例: {self.stream_id}")

        try:
            # 1. 初始化核心功能组件
            logger.debug(f"[私聊][{self.private_name}] 初始化 ActionPlanner...")
            self.action_planner = ActionPlanner(self.stream_id, self.private_name)

            self.relationship_updater = PfcRelationshipUpdater(
                private_name=self.private_name,
                bot_name=global_config.BOT_NICKNAME # 或者 self.name (如果 Conversation 类有 self.name)
            )
            self.relationship_translator = PfcRepationshipTranslator(private_name=self.private_name)
            logger.info(f"[私聊][{self.private_name}] PfcRelationship 初始化完成。")

            self.emotion_updater = PfcEmotionUpdater(
                private_name=self.private_name,
                bot_name=global_config.BOT_NICKNAME # 或者 self.name
            )
            logger.info(f"[私聊][{self.private_name}] PfcEmotion 初始化完成。")

            logger.debug(f"[私聊][{self.private_name}] 初始化 GoalAnalyzer...")
            self.goal_analyzer = GoalAnalyzer(self.stream_id, self.private_name)

            logger.debug(f"[私聊][{self.private_name}] 初始化 ReplyGenerator...")
            self.reply_generator = ReplyGenerator(self.stream_id, self.private_name)

            logger.debug(f"[私聊][{self.private_name}] 初始化 KnowledgeFetcher...")
            self.knowledge_fetcher = KnowledgeFetcher(self.private_name)

            logger.debug(f"[私聊][{self.private_name}] 初始化 Waiter...")
            self.waiter = Waiter(self.stream_id, self.private_name)

            logger.debug(f"[私聊][{self.private_name}] 初始化 DirectMessageSender...")
            self.direct_sender = DirectMessageSender(self.private_name)

            logger.debug(f"[私聊][{self.private_name}] 初始化 ReplyChecker...")
            self.reply_checker = ReplyChecker(self.stream_id, self.private_name)

            # 获取关联的 ChatStream
            logger.debug(f"[私聊][{self.private_name}] 获取 ChatStream...")
            self.chat_stream = chat_manager.get_stream(self.stream_id)
            if not self.chat_stream:
                # 获取不到 ChatStream 是一个严重问题，因为无法发送消息
                logger.error(
                    f"[私聊][{self.private_name}] 初始化错误：无法从 chat_manager 获取 stream_id {self.stream_id} 的 ChatStream。"
                )
                raise ValueError(f"无法获取 stream_id {self.stream_id} 的 ChatStream")

            # 初始化空闲对话启动器
            logger.debug(f"[私聊][{self.private_name}] 初始化 IdleConversationStarter...")
            self.idle_conversation_starter = IdleConversationStarter(self.stream_id, self.private_name)

            # 2. 初始化信息存储和观察组件
            logger.debug(f"[私聊][{self.private_name}] 获取 ChatObserver 实例...")
            self.chat_observer = ChatObserver.get_instance(self.stream_id, self.private_name)

            logger.debug(f"[私聊][{self.private_name}] 初始化 ObservationInfo...")
            self.observation_info = ObservationInfo(self.private_name)
            # 确保 ObservationInfo 知道机器人的 ID
            if not self.observation_info.bot_id:
                logger.warning(f"[私聊][{self.private_name}] ObservationInfo 未能自动获取 bot_id，尝试手动设置。")
                self.observation_info.bot_id = self.bot_qq_str

            logger.debug(f"[私聊][{self.private_name}] 初始化 ConversationInfo...")
            self.conversation_info = ConversationInfo()

            # 3. 绑定观察者和信息处理器
            logger.debug(f"[私聊][{self.private_name}] 绑定 ObservationInfo 到 ChatObserver...")
            self.observation_info.bind_to_chat_observer(self.chat_observer)

            # 4. 加载初始聊天记录
            await self._load_initial_history()

            # 4.1 加载用户数据
            # 尝试从 observation_info 获取，这依赖于 _load_initial_history 的实现
            private_user_id_str: Optional[str] = None
            private_platform_str: Optional[str] = None
            private_nickname_str = self.private_name

            if self.observation_info and self.observation_info.last_message_sender and self.observation_info.last_message_sender != self.bot_qq_str:
            # 如果历史记录最后一条不是机器人发的，那么发送者就是对方
            # 假设 observation_info 中已经有了 sender_user_id, sender_platform, sender_name
            # 这些字段应该在 observation_info.py 的 update_from_message 中从非机器人消息填充
            # 并且 _load_initial_history 处理历史消息时也应该填充它们
            # 这里的逻辑是：取 observation_info 中最新记录的非机器人发送者的信息
                if self.observation_info.sender_user_id and self.observation_info.sender_platform:
                    private_user_id_str = self.observation_info.sender_user_id
                    private_platform_str = self.observation_info.sender_platform
                    logger.info(f"[私聊][{self.private_name}] 从 ObservationInfo 获取到私聊对象信息: ID={private_user_id_str}, Platform={private_platform_str}, Name={private_nickname_str}")

            if not private_user_id_str and self.chat_stream: # 如果 observation_info 中没有，尝试从 chat_stream (通常代表对方)
                if self.chat_stream.user_info and str(self.chat_stream.user_info.user_id) != self.bot_qq_str : # 确保不是机器人自己
                    private_user_id_str = str(self.chat_stream.user_info.user_id)
                    private_platform_str = self.chat_stream.user_info.platform
                    logger.info(f"[私聊][{self.private_name}] 从 ChatStream 获取到私聊对象信息: ID={private_user_id_str}, Platform={private_platform_str}, Name={private_nickname_str}")
                elif self.chat_stream.group_info is None and self.private_name: # 私聊场景，且 private_name 可能有用
                    # 这是一个备选方案，如果 private_name 直接是 user_id
                    # 你需要确认 private_name 的确切含义和格式
                    # logger.warning(f"[私聊][{self.private_name}] 尝试使用 private_name ('{self.private_name}') 作为 user_id，平台默认为 'qq'")
                    # private_user_id_str = self.private_name
                    # private_platform_str = "qq" # 假设平台是qq
                    # private_nickname_str = self.private_name # 昵称也暂时用 private_name
                    pass # 暂时不启用此逻辑，依赖 observation_info 或 chat_stream.user_info

            if private_user_id_str and private_platform_str:
                try:
                   # 将 user_id 转换为整数类型，因为 person_info_manager.get_person_id 需要 int
                    private_user_id_int = int(private_user_id_str)
                    self.conversation_info.person_id = self.person_info_mng.get_person_id(
                        private_platform_str,
                        private_user_id_int # 使用转换后的整数ID
                    )
                    logger.info(f"[私聊][{self.private_name}] 获取到 person_id: {self.conversation_info.person_id} for {private_platform_str}:{private_user_id_str}")

                    # 确保用户在数据库中存在，如果不存在则创建
                    # get_or_create_person 内部处理 person_id 的生成，所以我们直接传 platform 和 user_id
                    await self.person_info_mng.get_or_create_person(
                        platform=private_platform_str,
                        user_id=private_user_id_int, # 使用转换后的整数ID
                        nickname=private_nickname_str if private_nickname_str else "未知用户",
                        # user_cardname 和 user_avatar 如果能从 chat_stream.user_info 或 observation_info 获取也应传入
                        # user_cardname = self.chat_stream.user_info.card if self.chat_stream and self.chat_stream.user_info else None,
                        # user_avatar = self.chat_stream.user_info.avatar if self.chat_stream and self.chat_stream.user_info else None
                    )
                except ValueError:
                    logger.error(f"[私聊][{self.private_name}] 无法将 private_user_id_str ('{private_user_id_str}') 转换为整数。")
                except Exception as e_pid:
                    logger.error(f"[私聊][{self.private_name}] 获取或创建 person_id 时出错: {e_pid}")
            else:
                logger.warning(f"[私聊][{self.private_name}] 未能确定私聊对象的 user_id 或 platform，无法获取 person_id。将在收到消息后尝试。")

            # 5. 启动需要后台运行的组件
            logger.debug(f"[私聊][{self.private_name}] 启动 ChatObserver...")
            self.chat_observer.start()
            if self.idle_conversation_starter:
                logger.debug(f"[私聊][{self.private_name}] 启动 IdleConversationStarter...")
                self.idle_conversation_starter.start()
                logger.info(f"[私聊][{self.private_name}] 空闲对话检测器已启动")

             # 5.1 启动 MoodManager 的后台更新
            if self.mood_mng and hasattr(self.mood_mng, 'start_mood_update') and not self.mood_mng._running:
                self.mood_mng.start_mood_update(update_interval=global_config.mood_update_interval) # 使用配置的更新间隔
                logger.info(f"[私聊][{self.private_name}] MoodManager 已启动后台更新，间隔: {global_config.mood_update_interval} 秒。")
            elif self.mood_mng and self.mood_mng._running:
                 logger.info(f"[私聊][{self.private_name}] MoodManager 已在运行中。")
            else:
                logger.warning(f"[私聊][{self.private_name}] MoodManager 未能启动，相关功能可能受限。")

            # --- 在初始化完成前，尝试加载一次关系和情绪信息 ---
            if self.conversation_info and self.conversation_info.person_id:
                try:
                    # 1. 获取数值型关系值
                    numeric_relationship_value = await self.person_info_mng.get_value(
                    self.conversation_info.person_id,
                    "relationship_value"
                    )
                    # 确保是浮点数
                    if not isinstance(numeric_relationship_value, (int, float)):
                        from bson.decimal128 import Decimal128
                        if isinstance(numeric_relationship_value, Decimal128):
                            numeric_relationship_value = float(numeric_relationship_value.to_decimal())
                        else:
                            numeric_relationship_value = 0.0

                    # 2. 使用PFC内部翻译函数
                    self.conversation_info.relationship_text = await self.relationship_translator.translate_relationship_value_to_text(numeric_relationship_value)
                    logger.info(f"[私聊][{self.private_name}] 初始化时加载关系文本: {self.conversation_info.relationship_text}")
                except Exception as e_init_rel:
                    logger.error(f"[私聊][{self.private_name}] 初始化时加载关系文本出错: {e_init_rel}")
                    self.conversation_info.relationship_text = "你们的关系是：普通。" # 保留默认值
            if self.conversation_info and self.mood_mng:
                try:
                   self.conversation_info.current_emotion_text = self.mood_mng.get_prompt()
                   logger.info(f"[私聊][{self.private_name}] 初始化时加载情绪文本: {self.conversation_info.current_emotion_text}")
                except Exception as e_init_emo:
                   logger.error(f"[私聊][{self.private_name}] 初始化时加载情绪文本出错: {e_init_emo}")
                # 保留默认值


            # 6. 标记初始化成功并设置运行状态
            self._initialized = True
            self.should_continue = True  # 初始化成功，标记可以继续运行循环
            self.state = ConversationState.ANALYZING  # 设置初始状态为分析

            logger.info(f"[私聊][{self.private_name}] 对话实例 {self.stream_id} 初始化完成。")

        except Exception as e:
            # 捕获初始化过程中的任何异常
            logger.error(f"[私聊][{self.private_name}] 初始化对话实例失败: {e}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            self.should_continue = False  # 初始化失败，标记不能继续
            self._initialized = False  # 确保标记为未初始化
            # 尝试停止可能部分启动的组件
            await self.stop()
            raise  # 将异常重新抛出，通知调用者初始化失败
        finally:
            # 无论成功与否，都要清除正在初始化的标记
            self._initializing = False

    async def _load_initial_history(self):
        """加载并处理初始的聊天记录"""
        if not self.observation_info:
            logger.warning(f"[私聊][{self.private_name}] ObservationInfo 未初始化，无法加载历史记录。")
            return

        try:
            logger.info(f"[私聊][{self.private_name}] 为 {self.stream_id} 加载初始聊天记录...")
            # 从聊天核心获取原始消息列表
            initial_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=self.stream_id,
                timestamp=time.time(),
                limit=30,  # limit 可以根据需要调整或配置
            )

            if initial_messages:
                # 更新 ObservationInfo 中的历史记录列表和计数
                self.observation_info.chat_history = initial_messages
                self.observation_info.chat_history_count = len(initial_messages)

                # 获取最后一条消息的信息
                last_msg = initial_messages[-1]
                self.observation_info.last_message_time = last_msg.get("time")
                self.observation_info.last_message_id = last_msg.get("message_id")

                # 安全地解析最后一条消息的发送者信息
                last_user_info_dict = last_msg.get("user_info", {})
                if isinstance(last_user_info_dict, dict):
                    try:
                        last_user_info = UserInfo.from_dict(last_user_info_dict)
                        # 存储发送者的 user_id 字符串
                        self.observation_info.last_message_sender = (
                            str(last_user_info.user_id) if last_user_info else None
                        )
                    except Exception as e:
                        logger.warning(f"[私聊][{self.private_name}] 解析最后一条消息的用户信息时出错: {e}")
                        self.observation_info.last_message_sender = None
                else:
                    # 如果 user_info 不是字典，也标记为未知
                    self.observation_info.last_message_sender = None

                # 存储最后一条消息的文本内容
                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")

                # 构建用于 Prompt 的历史记录字符串 (只使用最近的一部分)
                history_slice_for_str = initial_messages[-30:]  # 可配置
                self.observation_info.chat_history_str = await build_readable_messages(
                    history_slice_for_str,
                    replace_bot_name=True,
                    merge_messages=False,
                    timestamp_mode="relative",
                    read_mark=0.0,  # read_mark 可能需要根据实际情况调整
                )

                # 更新 ChatObserver 和 IdleStarter 的时间戳
                if self.chat_observer:
                    # 更新观察者的最后消息时间，避免重复处理这些初始消息
                    self.chat_observer.last_message_time = self.observation_info.last_message_time
                if self.idle_conversation_starter and self.observation_info.last_message_time:
                    # 更新空闲计时器的起始时间
                    await self.idle_conversation_starter.update_last_message_time(
                        self.observation_info.last_message_time
                    )

                logger.info(
                    f"[私聊][{self.private_name}] 成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}"
                )
            else:
                # 如果没有历史记录
                logger.info(f"[私聊][{self.private_name}] 没有找到初始聊天记录。")
                self.observation_info.chat_history_str = "还没有聊天记录。"  # 设置默认提示

        except Exception as load_err:
            # 捕获加载过程中的异常
            logger.error(f"[私聊][{self.private_name}] 加载初始聊天记录时出错: {load_err}")
            # 即使出错，也设置一个提示，避免后续使用 None 值
            if self.observation_info:
                self.observation_info.chat_history_str = "[加载聊天记录出错]"

    async def start(self):
        """
        启动对话流程。
        会检查实例是否已初始化，如果未初始化会尝试初始化。
        成功后，创建并启动核心的规划与行动循环 (`_plan_and_action_loop`)。
        """
        # 检查是否已初始化，如果未初始化则尝试进行初始化
        if not self._initialized:
            logger.warning(f"[私聊][{self.private_name}] 对话实例未初始化，尝试初始化...")
            try:
                await self._initialize()
                # 在尝试初始化后，再次检查状态
                if not self._initialized:
                    logger.error(f"[私聊][{self.private_name}] 初始化失败，无法启动规划循环。")
                    return  # 初始化失败，明确停止
            except Exception as init_err:
                logger.error(f"[私聊][{self.private_name}] 初始化过程中发生未捕获错误: {init_err}，无法启动。")
                return  # 初始化异常，明确停止

        # 再次检查 should_continue 标志，确保初始化成功且未被外部停止
        if not self.should_continue:
            logger.warning(
                f"[私聊][{self.private_name}] 对话实例已被标记为不应继续 (可能由于初始化失败或已被停止)，无法启动规划循环。"
            )
            return

        logger.info(f"[私聊][{self.private_name}] 对话系统启动，准备创建规划循环任务...")
        # 使用 asyncio.create_task 在后台启动主循环
        try:
            logger.debug(f"[私聊][{self.private_name}] 正在创建 _plan_and_action_loop 任务...")
            # 创建任务，但不等待其完成，让它在后台运行
            _loop_task = asyncio.create_task(self._plan_and_action_loop())
            # 可以选择性地添加完成回调来处理任务结束或异常
            # loop_task.add_done_callback(self._handle_loop_completion)
            logger.info(f"[私聊][{self.private_name}] 规划循环任务已创建。")
        except Exception as task_err:
            logger.error(f"[私聊][{self.private_name}] 创建规划循环任务时出错: {task_err}")
            # 如果创建任务失败，可能需要停止实例
            await self.stop()

    async def stop(self):
        """
        停止对话实例并清理相关资源。
        会停止后台任务、解绑观察者等。
        """
        logger.info(f"[私聊][{self.private_name}] 正在停止对话实例: {self.stream_id}")
        self.should_continue = False  # 设置标志，让主循环退出

        # --- 新增：在对话结束时调用最终关系更新 ---
        if self._initialized and self.relationship_updater and self.conversation_info and self.observation_info and self.chat_observer:
            try:
                logger.info(f"[私聊][{self.private_name}] 准备执行最终关系评估...")
                await self.relationship_updater.update_relationship_final(
                    conversation_info=self.conversation_info,
                    observation_info=self.observation_info,
                    chat_observer_for_history=self.chat_observer
                )
                logger.info(f"[私聊][{self.private_name}] 最终关系评估已调用。")
            except Exception as e_final_rel:
                logger.error(f"[私聊][{self.private_name}] 调用最终关系评估时出错: {e_final_rel}")
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"[私聊][{self.private_name}] 跳过最终关系评估，因为实例未完全初始化或缺少必要组件。")

        # 停止空闲对话检测器
        if self.idle_conversation_starter:
            self.idle_conversation_starter.stop()

        # 解绑 ObservationInfo 与 ChatObserver
        if self.observation_info and self.chat_observer:
            self.observation_info.unbind_from_chat_observer()

        if self.mood_mng and hasattr(self.mood_mng, 'stop_mood_update') and self.mood_mng._running:
            self.mood_mng.stop_mood_update()
            logger.info(f"[私聊][{self.private_name}] MoodManager 后台更新已停止。")

        # ChatObserver 是单例，通常不由单个 Conversation 停止
        # 如果需要，可以在管理器层面处理 ChatObserver 的生命周期

        # 标记为未初始化
        self._initialized = False
        logger.info(f"[私聊][{self.private_name}] 对话实例 {self.stream_id} 已停止。")

    async def _plan_and_action_loop(self):
        """
        核心的规划与行动循环 (PFC Loop)。
        持续运行，根据当前状态规划下一步行动，处理新消息中断，执行动作，直到被停止。
        """
        logger.info(f"[私聊][{self.private_name}] 进入 _plan_and_action_loop 循环。")

        # 循环前再次确认初始化状态
        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 尝试在未初始化状态下运行规划循环，退出。")
            return  # 明确退出
        force_reflect_and_act = False
        # 主循环，只要 should_continue 为 True 就一直运行
        while self.should_continue:
            loop_iter_start_time = time.time()  # 记录本次循环开始时间
            logger.debug(f"[私聊][{self.private_name}] 开始新一轮循环迭代 ({loop_iter_start_time:.2f})")
            try:
        # 重新获取 TIME_ZONE 以防在 __init__ 中导入失败
                if 'TIME_ZONE' not in locals() or TIME_ZONE is None:
                    from dateutil import tz
                    try:
                       from ...config.config import global_config
                    except ImportError:
                        global_config = None
                        TIME_ZONE = tz.tzlocal()
                    else:
                        configured_tz = getattr(global_config, 'TIME_ZONE', 'Asia/Shanghai')
                        TIME_ZONE = tz.gettz(configured_tz)
                        if TIME_ZONE is None: TIME_ZONE = tz.gettz('Asia/Shanghai')

                current_time = datetime.datetime.now(TIME_ZONE)
                if self.observation_info: # 确保 observation_info 存在
                    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z%z") # 包含时区信息的格式
                    self.observation_info.current_time_str = time_str
                    logger.debug(f"[私聊][{self.private_name}] 更新 ObservationInfo 当前时间: {time_str}")
                else:
                    logger.warning(f"[私聊][{self.private_name}] ObservationInfo 未初始化，无法更新当前时间。")
            except Exception as time_update_err:
                logger.error(f"[私聊][{self.private_name}] 更新 ObservationInfo 当前时间时出错: {time_update_err}")
                 # --- 更新时间代码结束 ---

            # --- 处理忽略状态 ---
            if self.ignore_until_timestamp and loop_iter_start_time < self.ignore_until_timestamp:
                # 如果当前处于忽略状态
                if self.idle_conversation_starter and self.idle_conversation_starter._running:
                    # 暂停空闲检测器
                    self.idle_conversation_starter.stop()
                    logger.debug(f"[私聊][{self.private_name}] 对话被暂时忽略，暂停空闲对话检测")
                # 计算需要睡眠的时间，最多30秒或直到忽略结束
                sleep_duration = min(30, self.ignore_until_timestamp - loop_iter_start_time)
                await asyncio.sleep(sleep_duration)
                continue  # 跳过本次循环的后续步骤，直接进入下一次迭代检查
            elif self.ignore_until_timestamp and loop_iter_start_time >= self.ignore_until_timestamp:
                # 如果忽略时间已到
                logger.info(f"[私聊][{self.private_name}] 忽略时间已到 {self.stream_id}，准备结束对话。")
                self.ignore_until_timestamp = None  # 清除忽略时间戳
                await self.stop()  # 调用 stop 方法来结束整个对话实例
                continue  # 跳过本次循环的后续步骤
            else:
                # 如果不在忽略状态，确保空闲检测器在运行
                if self.idle_conversation_starter and not self.idle_conversation_starter._running:
                    self.idle_conversation_starter.start()
                    logger.debug(f"[私聊][{self.private_name}] 恢复空闲对话检测")

            # --- 核心规划与行动逻辑 ---
            try:
                if self.conversation_info and self._initialized: # 确保 conversation_info 和实例已初始化
                    # 更新关系文本
                    if self.conversation_info.person_id: # 确保 person_id 已获取
                        try:
                            # 1.直接从 person_info_manager 获取数值型的 relationship_value
                            numeric_relationship_value = await self.person_info_mng.get_value(
                                self.conversation_info.person_id,
                                "relationship_value"
                            )
                        
                            # 确保 relationship_value 是浮点数 (可以复用 relationship_manager 中的 ensure_float，或者在这里简单处理)
                            if not isinstance(numeric_relationship_value, (int, float)):
                                # 尝试从 Decimal128 转换 (如果你的数据库用的是这个)
                                from bson.decimal128 import Decimal128
                                if isinstance(numeric_relationship_value, Decimal128):
                                    numeric_relationship_value = float(numeric_relationship_value.to_decimal())
                                else: # 其他类型，或转换失败，给默认值
                                    logger.warning(f"[私聊][{self.private_name}] 获取的 relationship_value 类型未知 ({type(numeric_relationship_value)}) 或转换失败，默认为0.0")
                                    numeric_relationship_value = 0.0
                        
                            logger.debug(f"[私聊][{self.private_name}] 获取到数值型关系值: {numeric_relationship_value}")
                        
                            # 2. 使用PFC内部的翻译函数将其转换为文本描述
                            simplified_relationship_text = await self.relationship_translator.translate_relationship_value_to_text(numeric_relationship_value)
                            self.conversation_info.relationship_text = simplified_relationship_text
                        
                            logger.debug(f"[私聊][{self.private_name}] 更新后关系文本 (PFC内部翻译): {self.conversation_info.relationship_text}")

                        except Exception as e_rel:
                            logger.error(f"[私聊][{self.private_name}] 更新关系文本(PFC内部翻译)时出错: {e_rel}")
                            self.conversation_info.relationship_text = "你们的关系是：普通。" # 出错时的默认值
                    elif self.observation_info and self.observation_info.sender_user_id and self.observation_info.sender_platform:
                        # 如果 person_id 之前没获取到，在这里尝试再次获取 (这部分逻辑保持，因为 person_id 是必须的)
                        try:
                            private_user_id_int = int(self.observation_info.sender_user_id)
                            self.conversation_info.person_id = self.person_info_mng.get_person_id(
                                self.observation_info.sender_platform,
                                private_user_id_int
                            )
                            await self.person_info_mng.get_or_create_person( #确保用户存在
                                platform=self.observation_info.sender_platform,
                                user_id=private_user_id_int,
                                nickname=self.observation_info.sender_name if self.observation_info.sender_name else "未知用户",
                            )
                            if self.conversation_info.person_id: # 如果成功获取 person_id，则再次尝试更新关系文本
                                numeric_relationship_value = await self.person_info_mng.get_value(
                                    self.conversation_info.person_id,
                                    "relationship_value"
                                )
                                if not isinstance(numeric_relationship_value, (int, float)):
                                    from bson.decimal128 import Decimal128
                                    if isinstance(numeric_relationship_value, Decimal128):
                                        numeric_relationship_value = float(numeric_relationship_value.to_decimal())
                                    else:
                                        numeric_relationship_value = 0.0
                                self.conversation_info.relationship_text = await self.relationship_translator.translate_relationship_value_to_text(numeric_relationship_value)
                                logger.debug(f"[私聊][{self.private_name}] (备用逻辑)更新后关系文本: {self.conversation_info.relationship_text}")

                        except ValueError:
                            logger.error(f"[私聊][{self.private_name}] 循环中无法将 sender_user_id ('{self.observation_info.sender_user_id}') 转换为整数。")
                            self.conversation_info.relationship_text = "你们的关系是：普通。"
                        except Exception as e_pid_loop:
                            logger.error(f"[私聊][{self.private_name}] 循环中获取 person_id 并更新关系时出错: {e_pid_loop}")
                            self.conversation_info.relationship_text = "你们的关系是：普通。"
                    else:
                         # 如果 person_id 仍无法确定
                        self.conversation_info.relationship_text = "你们的关系是：普通。" # 或 "你们的关系是：未知。"
                
                     # 更新情绪文本
                    if self.mood_mng:
                        self.conversation_info.current_emotion_text = self.mood_mng.get_prompt()
                        logger.debug(f"[私聊][{self.private_name}] 更新情绪文本: {self.conversation_info.current_emotion_text}")
                    else:
            # 如果 mood_mng 未初始化，使用 ConversationInfo 中的默认值
                        pass # self.conversation_info.current_emotion_text 会保持其在 ConversationInfo 中的默认值
    ### 标记新增/修改区域 结束 ###

                # 1. 检查核心组件是否都已初始化
                if not all([self.action_planner, self.observation_info, self.conversation_info]):
                    logger.error(f"[私聊][{self.private_name}] 核心组件未初始化，无法继续规划循环。将等待5秒后重试...")
                    await asyncio.sleep(5)
                    continue  # 跳过本次迭代

                # 2. 记录规划开始时间并重置临时状态
                planning_start_time = time.time()
                logger.debug(f"[私聊][{self.private_name}] --- 开始规划 ({planning_start_time:.2f}) ---")
                # 重置上一轮存储的“规划期间他人新消息数”
                self.conversation_info.other_new_messages_during_planning_count = 0

                # 3. 调用 ActionPlanner 进行规划
                logger.debug(f"[私聊][{self.private_name}] 调用 ActionPlanner.plan...")
                # 传入当前观察信息、对话信息和上次成功回复的动作类型
                action, reason = await self.action_planner.plan(
                    self.observation_info,
                    self.conversation_info, # type: ignore
                    self.conversation_info.last_successful_reply_action, # type: ignore
                    use_reflect_prompt=force_reflect_and_act # 使用标志
                )
                force_reflect_and_act = False
                planning_duration = time.time() - planning_start_time
                logger.debug(
                    f"[私聊][{self.private_name}] ActionPlanner.plan 完成 (耗时: {planning_duration:.3f} 秒)，初步规划动作: {action}"
                )

                # 4. 检查规划期间是否有新消息到达
                current_unprocessed_messages = getattr(self.observation_info, "unprocessed_messages", [])
                new_messages_during_planning: List[Dict[str, Any]] = []
                other_new_messages_during_planning: List[Dict[str, Any]] = []

                # 遍历当前所有未处理的消息
                for msg in current_unprocessed_messages:
                    msg_time = msg.get("time")
                    sender_id = msg.get("user_info", {}).get("user_id")
                    # 检查消息时间是否在本次规划开始之后
                    if msg_time and msg_time >= planning_start_time:
                        new_messages_during_planning.append(msg)
                        # 同时检查是否是来自他人的消息
                        if sender_id != self.bot_qq_str:
                            other_new_messages_during_planning.append(msg)

                new_msg_count = len(new_messages_during_planning)  # 规划期间所有新消息数
                other_new_msg_count = len(other_new_messages_during_planning)  # 规划期间他人新消息数
                logger.debug(
                    f"[私聊][{self.private_name}] 规划期间收到新消息总数: {new_msg_count}, 来自他人: {other_new_msg_count}"
                )

                if self.conversation_info and other_new_msg_count > 0: # 如果有来自他人的新消息
                    self.conversation_info.current_instance_message_count += other_new_msg_count
                    logger.debug(f"[私聊][{self.private_name}] 用户发送新消息，实例消息计数增加到: {self.conversation_info.current_instance_message_count}")
                    
                    # 调用增量关系更新
                    if self.relationship_updater:
                        await self.relationship_updater.update_relationship_incremental(
                            conversation_info=self.conversation_info,
                            observation_info=self.observation_info,
                            chat_observer_for_history=self.chat_observer
                        )
                    
                    # 调用情绪更新
                    if self.emotion_updater and other_new_messages_during_planning:
                        # 取最后一条用户消息作为情绪更新的上下文事件
                        last_user_msg = other_new_messages_during_planning[-1]
                        last_user_msg_text = last_user_msg.get("processed_plain_text", "用户发了新消息")
                        
                        sender_name_for_event = getattr(self.observation_info, 'sender_name', '对方')
                        if not sender_name_for_event: # 如果 observation_info 中还没有，尝试从消息中取
                            user_info_dict = last_user_msg.get("user_info", {})
                            sender_name_for_event = user_info_dict.get("user_nickname", "对方")

                        event_desc = f"用户【{sender_name_for_event}】发送了新消息: '{last_user_msg_text[:30]}...'"
                        await self.emotion_updater.update_emotion_based_on_context(
                            conversation_info=self.conversation_info,
                            observation_info=self.observation_info,
                            chat_observer_for_history=self.chat_observer,
                            event_description=event_desc
                        )

                # 5. 根据动作类型和新消息数量，判断是否需要中断当前规划
                should_interrupt: bool = False
                interrupt_reason: str = ""

                if action in ["wait", "listening"]:
                    # 规则：对于 wait/listen，任何新消息（无论来自谁）都应该中断
                    if new_msg_count > 0:
                        should_interrupt = True
                        interrupt_reason = f"规划 {action} 期间收到 {new_msg_count} 条新消息"
                        logger.info(f"[私聊][{self.private_name}] 中断 '{action}'，原因: {interrupt_reason}。")
                else:
                    # 规则：对于其他动作，检查来自他人的新消息是否超过阈值 2
                    interrupt_threshold: int = 2
                    if other_new_msg_count > interrupt_threshold:
                        should_interrupt = True
                        interrupt_reason = f"规划 {action} 期间收到 {other_new_msg_count} 条来自他人的新消息 (阈值 >{interrupt_threshold})"
                        logger.info(f"[私聊][{self.private_name}] 中断 '{action}'，原因: {interrupt_reason}。")

                # 6. 如果需要中断，则记录取消信息，重置状态，并进入下一次循环
                if should_interrupt:
                    logger.info(f"[私聊][{self.private_name}] 执行中断，重新规划...")
                    # 记录被取消的动作到历史记录
                    cancel_record = {
                        "action": action,
                        "plan_reason": reason,
                        "status": "cancelled_due_to_new_messages",  # 标记取消原因
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "final_reason": interrupt_reason,
                    }
                    # 安全地添加到 done_action 列表
                    if not hasattr(self.conversation_info, "done_action"):
                        self.conversation_info.done_action = []
                    self.conversation_info.done_action.append(cancel_record)

                    # 重置追问状态，因为当前动作被中断了
                    self.conversation_info.last_successful_reply_action = None
                    # 将状态设置回分析，准备处理新消息并重新规划
                    self.state = ConversationState.ANALYZING
                    await asyncio.sleep(0.1)  # 短暂等待，避免CPU空转
                    continue  # 直接进入下一次循环迭代

                # 7. 如果未中断，存储规划期间的他人新消息数，并执行动作
                logger.debug(f"[私聊][{self.private_name}] 未中断，调用 _handle_action 执行动作 '{action}'...")
                # 将计算出的“规划期间他人新消息数”存入 conversation_info，供 _handle_action 使用
                self.conversation_info.other_new_messages_during_planning_count = other_new_msg_count
                # 调用动作处理函数
                await self._handle_action(action, reason, self.observation_info, self.conversation_info)
                logger.debug(f"[私聊][{self.private_name}] _handle_action 完成。")

                # --- 新增逻辑：检查是否因为RG决定不发送而需要反思 ---
                last_action_record = (
                    self.conversation_info.done_action[-1] if self.conversation_info.done_action else {} # type: ignore
                )
                if last_action_record.get("action") == "send_new_message" and \
                    last_action_record.get("status") == "done_no_reply":
                    logger.info(f"[私聊][{self.private_name}] 检测到 ReplyGenerator 决定不发送消息，将在下一轮强制使用反思Prompt。")
                    force_reflect_and_act = True # 设置标志，下一轮使用反思prompt
                    # 不需要立即 continue，让循环自然进入下一轮，下一轮的 plan 会用这个标志

                # 8. 检查是否需要结束整个对话（例如目标达成或执行了结束动作）
                goal_ended: bool = False
                # 检查最新的目标是否是“结束对话”
                if hasattr(self.conversation_info, "goal_list") and self.conversation_info.goal_list:
                    last_goal_item = self.conversation_info.goal_list[-1]
                    current_goal: Optional[str] = None
                    if isinstance(last_goal_item, dict):
                        current_goal = last_goal_item.get("goal")
                    elif isinstance(last_goal_item, str):
                        current_goal = last_goal_item
                    if isinstance(current_goal, str) and current_goal == "结束对话":
                        goal_ended = True

                # 检查最后执行的动作是否是结束类型且成功完成
                last_action_record = (
                    self.conversation_info.done_action[-1] if self.conversation_info.done_action else {}
                )
                action_ended: bool = (
                    last_action_record.get("action") in ["end_conversation", "say_goodbye"]
                    and last_action_record.get("status") == "done"
                )

                # 如果满足任一结束条件，则停止循环
                if goal_ended or action_ended:
                    logger.info(
                        f"[私聊][{self.private_name}] 检测到结束条件 (目标结束: {goal_ended}, 动作结束: {action_ended})，停止循环。"
                    )
                    await self.stop()  # 调用 stop 来停止实例
                    continue  # 跳过后续，虽然 stop 会设置 should_continue=False

            except asyncio.CancelledError:
                # 处理任务被取消的情况
                logger.info(f"[私聊][{self.private_name}] PFC 主循环任务被取消。")
                await self.stop()  # 确保资源被清理
                break  # 明确退出循环
            except Exception as loop_err:
                # 捕获循环中的其他未预期错误
                logger.error(f"[私聊][{self.private_name}] PFC 主循环出错: {loop_err}")
                logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
                self.state = ConversationState.ERROR  # 设置错误状态
                # 可以在这里添加更复杂的错误恢复逻辑，或者简单等待后重试
                await asyncio.sleep(5)  # 等待一段时间，避免错误状态下快速空转

            # --- 控制循环频率 ---
            loop_duration = time.time() - loop_iter_start_time  # 计算本次循环耗时
            min_loop_interval = 0.1  # 设置最小循环间隔（秒），防止CPU占用过高
            logger.debug(f"[私聊][{self.private_name}] 循环迭代耗时: {loop_duration:.3f} 秒。")
            if loop_duration < min_loop_interval:
                # 如果循环太快，则睡眠一段时间
                await asyncio.sleep(min_loop_interval - loop_duration)

        # 循环结束后的日志
        logger.info(f"[私聊][{self.private_name}] PFC 循环已退出 for stream_id: {self.stream_id}")

    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Optional[Message]:
        """将从数据库或其他来源获取的消息字典转换为内部使用的 Message 对象"""
        try:
            # 优先使用实例自身的 chat_stream，如果不存在则尝试从管理器获取
            chat_stream_to_use = self.chat_stream or chat_manager.get_stream(self.stream_id)
            if not chat_stream_to_use:
                logger.error(
                    f"[私聊][{self.private_name}] 无法确定 ChatStream for stream_id {self.stream_id}，无法转换消息。"
                )
                return None  # 无法确定聊天流，返回 None

            # 解析用户信息字典
            user_info_dict = msg_dict.get("user_info", {})
            user_info: Optional[UserInfo] = None
            if isinstance(user_info_dict, dict):
                try:
                    # 使用 UserInfo 类的方法从字典创建对象
                    user_info = UserInfo.from_dict(user_info_dict)
                except Exception as e:
                    # 解析失败记录警告
                    logger.warning(
                        f"[私聊][{self.private_name}] 从字典创建 UserInfo 时出错: {e}, dict: {user_info_dict}"
                    )
            if not user_info:
                # 如果没有有效的 UserInfo，记录警告并返回 None
                logger.warning(
                    f"[私聊][{self.private_name}] 消息缺少有效的 UserInfo，无法转换。 msg_id: {msg_dict.get('message_id')}"
                )
                return None

            # 创建并返回 Message 对象
            return Message(
                message_id=msg_dict.get("message_id", f"gen_{time.time()}"),  # 如果没有ID，生成一个临时的
                chat_stream=chat_stream_to_use,
                time=msg_dict.get("time", time.time()),  # 如果没有时间戳，使用当前时间
                user_info=user_info,  # 使用解析出的 UserInfo 对象
                processed_plain_text=msg_dict.get("processed_plain_text", ""),  # 获取处理后的纯文本
                detailed_plain_text=msg_dict.get("detailed_plain_text", ""),  # 获取详细纯文本
            )
        except Exception as e:
            # 捕获转换过程中的任何异常
            logger.error(f"[私聊][{self.private_name}] 转换消息时出错: {e}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            return None  # 转换失败返回 None

    async def _handle_action(
        self, action: str, reason: str, observation_info: ObservationInfo, conversation_info: ConversationInfo
    ):
        """
        处理由 ActionPlanner 规划出的具体行动。
        包括生成回复、调用检查器、发送消息、等待、思考目标等，并包含重试逻辑。
        根据执行结果和规则更新对话状态。
        """
        # 检查初始化状态
        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 尝试在未初始化状态下处理动作 '{action}'。")
            return

        logger.info(f"[私聊][{self.private_name}] 开始处理动作: {action}, 原因: {reason}")
        action_start_time = time.time()  # 记录动作开始时间

        # --- 准备动作历史记录条目 ---
        current_action_record = {
            "action": action,
            "plan_reason": reason,  # 记录规划时的原因
            "status": "start",  # 初始状态为“开始”
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 记录开始时间
            "final_reason": None,  # 最终结果的原因，将在 finally 中设置
        }
        # 安全地添加到历史记录列表
        if not hasattr(conversation_info, "done_action"):
            conversation_info.done_action = []
        conversation_info.done_action.append(current_action_record)
        # 获取当前记录在列表中的索引，方便后续更新状态
        action_index = len(conversation_info.done_action) - 1

        # --- 初始化动作执行状态变量 ---
        action_successful: bool = False  # 标记动作是否成功执行
        final_status: str = "recall"  # 动作最终状态，默认为 recall (表示未成功或需重试)
        final_reason: str = "动作未成功执行"  # 动作最终原因
        need_replan_from_checker: bool = False  # 标记是否由 ReplyChecker 要求重新规划

        try:
            # --- 根据不同的 action 类型执行相应的逻辑 ---

            # 1. 处理需要生成、检查、发送的动作
            if action in ["direct_reply", "send_new_message"]:
                max_reply_attempts: int = 3  # 最多尝试次数 (可配置)
                reply_attempt_count: int = 0
                is_suitable: bool = False  # 标记回复是否合适
                generated_content: str = ""  # 存储生成的回复
                check_reason: str = "未进行检查"  # 存储检查结果原因

                # --- [核心修复] 引入重试循环 ---
                is_send_decision_from_rg = False # 标记是否由 reply_generator 决定发送
                
                while reply_attempt_count < max_reply_attempts and not is_suitable and not need_replan_from_checker:
                    reply_attempt_count += 1
                    log_prefix = f"[私聊][{self.private_name}] 尝试生成/检查 '{action}' 回复 (第 {reply_attempt_count}/{max_reply_attempts} 次)..."
                    logger.info(log_prefix)

                    self.state = ConversationState.GENERATING
                    if not self.reply_generator:
                        raise RuntimeError("ReplyGenerator 未初始化")
                    
                    raw_llm_output = await self.reply_generator.generate(
                        observation_info, conversation_info, action_type=action
                    )
                    logger.debug(f"{log_prefix} ReplyGenerator.generate 返回: '{raw_llm_output}'")

                    should_send_reply = True # 默认对于 direct_reply 是要发送的
                    text_to_process = raw_llm_output # 默认情况下，处理原始输出

                    if action == "send_new_message":
                        is_send_decision_from_rg = True # 标记 send_new_message 的决策来自RG
                        try:
                            # 使用 pfc_utils.py 中的 get_items_from_json 来解析
                            # 注意：get_items_from_json 目前主要用于提取固定字段的字典。
                            # reply_generator 返回的是一个顶级JSON对象。
                            # 我们需要稍微调整用法或增强 get_items_from_json。
                            # 简单起见，这里我们先直接用 json.loads，后续可以优化。
                            
                            parsed_json = None
                            try:
                                parsed_json = json.loads(raw_llm_output)
                            except json.JSONDecodeError:
                                logger.error(f"{log_prefix} ReplyGenerator 返回的不是有效的JSON: {raw_llm_output}")
                                # 如果JSON解析失败，视为RG决定不发送，并给出原因
                                conversation_info.last_reply_rejection_reason = "回复生成器未返回有效JSON"
                                conversation_info.last_rejected_reply_content = raw_llm_output
                                should_send_reply = False
                                text_to_process = "no" # 或者一个特定的错误标记
                            
                            if parsed_json:
                                send_decision = parsed_json.get("send", "no").lower()
                                generated_text_from_json = parsed_json.get("txt", "no")

                                if send_decision == "yes":
                                    should_send_reply = True
                                    text_to_process = generated_text_from_json
                                    logger.info(f"{log_prefix} ReplyGenerator 决定发送消息。内容: '{text_to_process[:100]}...'")
                                else:
                                    should_send_reply = False
                                    text_to_process = "no" # 保持和 prompt 中一致，txt 为 "no"
                                    logger.info(f"{log_prefix} ReplyGenerator 决定不发送消息。")
                                    # 此时，我们应该跳出重试循环，并触发 action_planner 的反思 prompt
                                    # 将此信息传递到循环外部进行处理
                                    break # 跳出 while 循环

                        except Exception as e_json: # 更广泛地捕获解析相关的错误
                            logger.error(f"{log_prefix} 解析 ReplyGenerator 的JSON输出时出错: {e_json}, 输出: {raw_llm_output}")
                            conversation_info.last_reply_rejection_reason = f"解析回复生成器JSON输出错误: {e_json}"
                            conversation_info.last_rejected_reply_content = raw_llm_output
                            should_send_reply = False
                            text_to_process = "no"
                    
                    if not should_send_reply and action == "send_new_message": # 如果RG决定不发送 (send_new_message特定逻辑)
                        break # 直接跳出重试循环，后续逻辑会处理这种情况

                    generated_content_for_check_or_send = text_to_process

                    if not generated_content_for_check_or_send or generated_content_for_check_or_send.startswith("抱歉") or (action == "send_new_message" and generated_content_for_check_or_send == "no"):
                        logger.warning(f"{log_prefix} 生成内容无效或为错误提示 (或send:no)，将进行下一次尝试 (如果适用)。")
                        check_reason = "生成内容无效或选择不发送"
                        conversation_info.last_reply_rejection_reason = check_reason
                        conversation_info.last_rejected_reply_content = generated_content_for_check_or_send
                        if action == "direct_reply": # direct_reply 失败时才继续尝试
                             await asyncio.sleep(0.5)
                             continue
                        else: # send_new_message 如果是 no，不应该继续尝试，上面已经break了
                            pass # 理论上不会到这里如果上面break了

                    self.state = ConversationState.CHECKING
                    if not self.reply_checker:
                        raise RuntimeError("ReplyChecker 未初始化")

                    current_goal_str = ""
                    if conversation_info.goal_list:
                        goal_item = conversation_info.goal_list[-1]
                        if isinstance(goal_item, dict):
                            current_goal_str = goal_item.get("goal", "")
                        elif isinstance(goal_item, str):
                            current_goal_str = goal_item
                    
                    chat_history_for_check = getattr(observation_info, "chat_history", [])
                    chat_history_text_for_check = getattr(observation_info, "chat_history_str", "")
                    current_retry_for_checker = reply_attempt_count - 1
                    current_time_value_for_check = observation_info.current_time_str or "获取时间失败"

                    if global_config.enable_pfc_reply_checker:
                        logger.debug(f"{log_prefix} 调用 ReplyChecker 检查 (配置已启用)...")
                        is_suitable, check_reason, need_replan_from_checker = await self.reply_checker.check(
                            reply=generated_content_for_check_or_send,
                            goal=current_goal_str,
                            chat_history=chat_history_for_check,
                            chat_history_text=chat_history_text_for_check,
                            current_time_str=current_time_value_for_check,
                            retry_count=current_retry_for_checker,
                        )
                        logger.info(
                            f"{log_prefix} ReplyChecker 结果: 合适={is_suitable}, 原因='{check_reason}', 需重规划={need_replan_from_checker}"
                        )
                    else:
                        is_suitable = True
                        check_reason = "ReplyChecker 已通过配置关闭"
                        need_replan_from_checker = False
                        logger.info(f"{log_prefix} [配置关闭] ReplyChecker 已跳过，默认回复为合适。")
                    
                    if not is_suitable:
                        conversation_info.last_reply_rejection_reason = check_reason
                        conversation_info.last_rejected_reply_content = generated_content_for_check_or_send
                        if not need_replan_from_checker and reply_attempt_count < max_reply_attempts:
                            logger.warning(f"{log_prefix} 回复不合适，原因: {check_reason}。将进行下一次尝试。")
                            await asyncio.sleep(0.5)
                        # 如果需要重规划或达到最大次数，循环会在下次判断时自动结束
                
                # --- 循环结束后处理 ---
                if action == "send_new_message" and not should_send_reply and is_send_decision_from_rg:
                    # 这是 reply_generator 决定不发送的情况
                    logger.info(f"[私聊][{self.private_name}] 动作 '{action}': ReplyGenerator 决定不发送消息。将调用 ActionPlanner 进行反思。")
                    final_status = "done_no_reply" # 一个新的状态，表示动作完成但无回复
                    final_reason = "回复生成器决定不发送消息"
                    action_successful = True # 动作本身（决策）是成功的

                    # 清除追问状态，因为没有实际发送
                    conversation_info.last_successful_reply_action = None
                    conversation_info.my_message_count = 0 # 重置连续发言计数

                    # !!! 触发 ActionPlanner 使用 PROMPT_REFLECT_AND_ACT !!!
                    if not self.action_planner:
                        raise RuntimeError("ActionPlanner 未初始化")
                    
                    logger.info(f"[私聊][{self.private_name}] {global_config.BOT_NICKNAME}本来想发一条新消息，但是想想还是算了。现在重新规划...")
                    # 调用 action_planner.plan 并传入 use_reflect_prompt=True
                    new_action, new_reason = await self.action_planner.plan(
                        observation_info,
                        conversation_info,
                        last_successful_reply_action=None, # 因为没发送，所以没有成功的回复动作
                        use_reflect_prompt=True
                    )
                    # 记录这次特殊的“反思”动作
                    reflect_action_record = {
                        "action": f"reflect_after_no_send ({new_action})", # 记录原始意图和新规划
                        "plan_reason": f"RG决定不发送后，AP规划: {new_reason}",
                        "status": "delegated", # 标记为委托给新的规划
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    conversation_info.done_action.append(reflect_action_record)

                    logger.info(f"[私聊][{self.private_name}] 反思后的新规划动作: {new_action}, 原因: {new_reason}")
                    # **暂定方案：**
                    # _handle_action 在这种情况下返回一个特殊标记。
                    # 为了不立即修改返回类型，我们暂时在这里记录日志，并在 _plan_and_action_loop 中添加逻辑。
                    # _handle_action 会将 action_successful 设置为 True，final_status 为 "done_no_reply"。
                    # _plan_and_action_loop 之后会检查这个状态。

                    # (这里的 final_status, final_reason, action_successful 已在上面设置)

                elif is_suitable: # 适用于 direct_reply 或 send_new_message (RG决定发送且检查通过)
                    logger.info(f"[私聊][{self.private_name}] 动作 '{action}': 找到合适的回复，准备发送。")
                    conversation_info.last_reply_rejection_reason = None
                    conversation_info.last_rejected_reply_content = None
                    self.generated_reply = generated_content_for_check_or_send
                    timestamp_before_sending = time.time()
                    logger.debug(
                        f"[私聊][{self.private_name}] 动作 '{action}': 记录发送前时间戳: {timestamp_before_sending:.2f}"
                    )
                    self.state = ConversationState.SENDING
                    send_success = await self._send_reply() # _send_reply 内部会更新 my_message_count
                    send_end_time = time.time()

                    if send_success:
                        action_successful = True
                        logger.info(f"[私聊][{self.private_name}] 动作 '{action}': 成功发送回复.")
                        if self.idle_conversation_starter:
                            await self.idle_conversation_starter.update_last_message_time(send_end_time)

                        current_unprocessed_messages = getattr(observation_info, "unprocessed_messages", [])
                        message_ids_to_clear: Set[str] = set()
                        for msg in current_unprocessed_messages:
                            msg_time = msg.get("time")
                            msg_id = msg.get("message_id")
                            sender_id_info = msg.get("user_info", {})
                            sender_id = str(sender_id_info.get("user_id")) if sender_id_info else None
                            
                            if (
                                msg_id
                                and msg_time
                                and sender_id != self.bot_qq_str # 确保是对方的消息
                                and msg_time < timestamp_before_sending # 只清理发送前的
                            ):
                                message_ids_to_clear.add(msg_id)
                        
                        if message_ids_to_clear:
                            logger.debug(
                                f"[私聊][{self.private_name}] 准备清理 {len(message_ids_to_clear)} 条发送前(他人)消息: {message_ids_to_clear}"
                            )
                            await observation_info.clear_processed_messages(message_ids_to_clear)
                        else:
                            logger.debug(f"[私聊][{self.private_name}] 没有需要清理的发送前(他人)消息。")

                        other_new_msg_count_during_planning = getattr(
                            conversation_info, "other_new_messages_during_planning_count", 0
                        )

                        if other_new_msg_count_during_planning > 0 and action == "direct_reply":
                            logger.info(
                                f"[私聊][{self.private_name}] 因规划期间收到 {other_new_msg_count_during_planning} 条他人新消息，下一轮强制使用【初始回复】逻辑。"
                            )
                            conversation_info.last_successful_reply_action = None
                            # conversation_info.my_message_count = 0 # 不在这里重置，因为刚发了一条
                        elif action == "direct_reply" or action == "send_new_message": # 成功发送后
                            logger.info(
                                f"[私聊][{self.private_name}] 成功执行 '{action}', 下一轮【允许】使用追问逻辑。"
                            )
                            conversation_info.last_successful_reply_action = action
                        
                        if conversation_info:
                            conversation_info.current_instance_message_count += 1
                            logger.debug(f"[私聊][{self.private_name}] 实例消息计数(机器人发送后)增加到: {conversation_info.current_instance_message_count}")
                            
                            if self.relationship_updater:
                                await self.relationship_updater.update_relationship_incremental(
                                    conversation_info=conversation_info,
                                    observation_info=observation_info,
                                    chat_observer_for_history=self.chat_observer
                                )
                            
                            sent_reply_summary = self.generated_reply[:50] if self.generated_reply else "空回复"
                            event_for_emotion_update = f"你刚刚发送了消息: '{sent_reply_summary}...'"
                            if self.emotion_updater:
                                await self.emotion_updater.update_emotion_based_on_context(
                                    conversation_info=conversation_info,
                                    observation_info=observation_info,
                                    chat_observer_for_history=self.chat_observer,
                                    event_description=event_for_emotion_update
                                )
                    else: # 发送失败
                        logger.error(f"[私聊][{self.private_name}] 动作 '{action}': 发送回复失败。")
                        final_status = "recall"
                        final_reason = "发送回复时失败"
                        conversation_info.last_successful_reply_action = None
                        conversation_info.my_message_count = 0 # 发送失败，重置计数

                elif need_replan_from_checker:
                    logger.warning(
                        f"[私聊][{self.private_name}] 动作 '{action}' 因 ReplyChecker 要求而被取消，将重新规划。原因: {check_reason}"
                    )
                    final_status = "recall"
                    final_reason = f"回复检查要求重新规划: {check_reason}"
                    conversation_info.last_successful_reply_action = None
                    # my_message_count 保持不变，因为没有成功发送

                else: # 达到最大尝试次数仍未找到合适回复
                    logger.warning(
                        f"[私聊][{self.private_name}] 动作 '{action}': 达到最大尝试次数 ({max_reply_attempts})，未能生成/检查通过合适的回复。最终原因: {check_reason}"
                    )
                    final_status = "recall"
                    final_reason = f"尝试{max_reply_attempts}次后失败: {check_reason}"
                    conversation_info.last_successful_reply_action = None
                    # my_message_count 保持不变

            # 2. 处理发送告别语动作 (保持简单，不加重试)
            elif action == "say_goodbye":
                self.state = ConversationState.GENERATING
                if not self.reply_generator:
                    raise RuntimeError("ReplyGenerator 未初始化")
                # 生成告别语
                generated_content = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type=action
                )
                logger.info(f"[私聊][{self.private_name}] 动作 '{action}': 生成内容: '{generated_content[:100]}...'")

                # 检查生成内容
                if not generated_content or generated_content.startswith("抱歉"):
                    logger.warning(f"[私聊][{self.private_name}] 动作 '{action}': 生成内容为空或为错误提示，取消发送。")
                    final_reason = "生成内容无效"
                    # 即使生成失败，也按计划结束对话
                    final_status = "done"  # 标记为 done，因为目的是结束
                    self.should_continue = False
                    logger.info(f"[私聊][{self.private_name}] 告别语生成失败，仍按计划结束对话。")
                else:
                    # 发送告别语
                    self.generated_reply = generated_content
                    timestamp_before_sending = time.time()
                    logger.debug(
                        f"[私聊][{self.private_name}] 动作 '{action}': 记录发送前时间戳: {timestamp_before_sending:.2f}"
                    )
                    self.state = ConversationState.SENDING
                    send_success = await self._send_reply()
                    send_end_time = time.time()

                    if send_success:
                        action_successful = True  # 标记成功
                        # final_status 和 final_reason 会在 finally 中设置
                        logger.info(f"[私聊][{self.private_name}] 成功发送告别语，即将停止对话实例。")
                        # 更新空闲计时器
                        if self.idle_conversation_starter:
                            await self.idle_conversation_starter.update_last_message_time(send_end_time)
                        # 清理发送前的消息 (虽然通常是最后一条，但保持逻辑一致)
                        current_unprocessed_messages = getattr(observation_info, "unprocessed_messages", [])
                        message_ids_to_clear: Set[str] = set()
                        for msg in current_unprocessed_messages:
                            msg_time = msg.get("time")
                            msg_id = msg.get("message_id")
                            sender_id = msg.get("user_info", {}).get("user_id")
                            if (
                                msg_id
                                and msg_time
                                and sender_id != self.bot_qq_str
                                and msg_time < timestamp_before_sending
                            ):
                                message_ids_to_clear.add(msg_id)
                        if message_ids_to_clear:
                            await observation_info.clear_processed_messages(message_ids_to_clear)

                        ### [[[ 新增：调用关系和情绪更新 ]]] ###
                        if conversation_info: # 确保 conversation_info 存在
                            conversation_info.current_instance_message_count += 1
                            logger.debug(f"[私聊][{self.private_name}] 实例消息计数(告别语后)增加到: {conversation_info.current_instance_message_count}")
                            
                            # 告别通常是结束，可以不进行增量关系更新，但情绪可以更新
                            # if self.relationship_updater:
                            # await self.relationship_updater.update_relationship_incremental(...)
                        
                        sent_reply_summary = self.generated_reply[:50] if self.generated_reply else "空回复"
                        event_for_emotion_update = f"你发送了告别消息: '{sent_reply_summary}...'"
                        if self.emotion_updater:
                            await self.emotion_updater.update_emotion_based_on_context(
                                conversation_info=conversation_info,
                                observation_info=observation_info,
                                chat_observer_for_history=self.chat_observer,
                                event_description=event_for_emotion_update
                            )
                        # 发送成功后结束对话
                        self.should_continue = False
                    else:
                        # 发送失败
                        logger.error(f"[私聊][{self.private_name}] 动作 '{action}': 发送告别语失败。")
                        final_status = "recall"
                        final_reason = "发送告别语失败"
                        # 发送失败不能结束对话
                        self.should_continue = True

            # 3. 处理重新思考目标动作
            elif action == "rethink_goal":
                self.state = ConversationState.RETHINKING
                if not self.goal_analyzer:
                    raise RuntimeError("GoalAnalyzer 未初始化")
                # 调用 GoalAnalyzer 分析并更新目标
                await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
                action_successful = True  # 标记成功
                event_for_emotion_update = "你重新思考了对话目标和方向"
                if self.emotion_updater and conversation_info and observation_info: # 确保updater和info都存在
                    await self.emotion_updater.update_emotion_based_on_context(
                        conversation_info=conversation_info,
                        observation_info=observation_info,
                        chat_observer_for_history=self.chat_observer,
                        event_description=event_for_emotion_update
                    )

            # 4. 处理倾听动作
            elif action == "listening":
                self.state = ConversationState.LISTENING
                if not self.waiter:
                    raise RuntimeError("Waiter 未初始化")
                logger.info(f"[私聊][{self.private_name}] 动作 'listening': 进入倾听状态...")
                # 调用 Waiter 的倾听等待方法，内部会处理超时
                await self.waiter.wait_listening(conversation_info)
                action_successful = True  # listening 动作本身执行即视为成功，后续由新消息或超时驱动
                event_for_emotion_update = "你决定耐心倾听对方的发言"
                if self.emotion_updater and conversation_info and observation_info:
                    await self.emotion_updater.update_emotion_based_on_context(
                        conversation_info=conversation_info,
                        observation_info=observation_info,
                        chat_observer_for_history=self.chat_observer,
                        event_description=event_for_emotion_update
                    )

            # 5. 处理结束对话动作
            elif action == "end_conversation":
                logger.info(f"[私聊][{self.private_name}] 动作 'end_conversation': 收到最终结束指令，停止对话...")
                action_successful = True  # 标记成功
                self.should_continue = False  # 设置标志以退出循环

            # 6. 处理屏蔽忽略动作
            elif action == "block_and_ignore":
                logger.info(f"[私聊][{self.private_name}] 动作 'block_and_ignore': 不想再理你了...")
                ignore_duration_seconds = 10 * 60  # 忽略 10 分钟，可配置
                self.ignore_until_timestamp = time.time() + ignore_duration_seconds
                logger.info(
                    f"[私聊][{self.private_name}] 将忽略此对话直到: {datetime.datetime.fromtimestamp(self.ignore_until_timestamp)}"
                )
                self.state = ConversationState.IGNORED  # 设置忽略状态
                action_successful = True  # 标记成功
                event_for_emotion_update = "当前对话让你感到不适，你决定暂时不再理会对方"
                if self.emotion_updater and conversation_info and observation_info:
                    # 可以让LLM判断此时的情绪，或者直接设定一个倾向（比如厌恶、不耐烦）
                    # 这里还是让LLM判断
                    await self.emotion_updater.update_emotion_based_on_context(
                        conversation_info=conversation_info,
                        observation_info=observation_info,
                        chat_observer_for_history=self.chat_observer,
                        event_description=event_for_emotion_update
                    )

            # 7. 处理等待动作
            elif action == "wait":
                self.state = ConversationState.WAITING
                if not self.waiter:
                    raise RuntimeError("Waiter 未初始化")
                logger.info(f"[私聊][{self.private_name}] 动作 'wait': 进入等待状态...")
                # 调用 Waiter 的常规等待方法，内部处理超时
                # wait 方法返回是否超时 (True=超时, False=未超时/被新消息中断)
                timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True  # wait 动作本身执行即视为成功
                event_for_emotion_update = ""
                if timeout_occurred: # 假设 timeout_occurred 能正确反映是否超时
                    event_for_emotion_update = "你等待对方回复，但对方长时间没有回应"
                else:
                    event_for_emotion_update = "你选择等待对方的回复（对方可能很快回复了）"
                
                if self.emotion_updater and conversation_info and observation_info:
                     await self.emotion_updater.update_emotion_based_on_context(
                        conversation_info=conversation_info,
                        observation_info=observation_info,
                        chat_observer_for_history=self.chat_observer,
                        event_description=event_for_emotion_update
                    )
                # wait 动作完成后不需要清理消息，等待新消息或超时触发重新规划
                logger.debug(f"[私聊][{self.private_name}] Wait 动作完成，无需在此清理消息。")

            # 8. 处理未知的动作类型
            else:
                logger.warning(f"[私聊][{self.private_name}] 未知的动作类型: {action}")
                final_status = "recall"  # 未知动作标记为 recall
                final_reason = f"未知的动作类型: {action}"

            # --- 重置非回复动作的追问状态 ---
            # 确保执行完非回复动作后，下一次规划不会错误地进入追问逻辑
            if action not in ["direct_reply", "send_new_message", "say_goodbye"]:
                conversation_info.last_successful_reply_action = None
                # 清理可能残留的拒绝信息
                conversation_info.last_reply_rejection_reason = None
                conversation_info.last_rejected_reply_content = None

        except asyncio.CancelledError:
            # 处理任务被取消的异常
            logger.warning(f"[私聊][{self.private_name}] 处理动作 '{action}' 时被取消。")
            final_status = "cancelled"
            final_reason = "动作处理被取消"
            # 取消时也重置追问状态
            conversation_info.last_successful_reply_action = None
            raise  # 重新抛出 CancelledError，让上层知道任务被取消
        except Exception as handle_err:
            # 捕获处理动作过程中的其他所有异常
            logger.error(f"[私聊][{self.private_name}] 处理动作 '{action}' 时出错: {handle_err}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            final_status = "error"  # 标记为错误状态
            final_reason = f"处理动作时出错: {handle_err}"
            self.state = ConversationState.ERROR  # 设置对话状态为错误
            # 出错时重置追问状态
            conversation_info.last_successful_reply_action = None

        finally:
            # --- 无论成功与否，都执行 ---

            # 1. 重置临时存储的计数值
            # 确保这个值只在当前规划周期内有效
            conversation_info.other_new_messages_during_planning_count = 0

            # 2. 更新动作历史记录的最终状态和原因
            # 优化：如果动作成功但状态仍是默认的 recall，则更新为 done
            if final_status == "recall" and action_successful:
                final_status = "done"
                # 根据动作类型设置更具体的成功原因
                if action == "wait":
                    # 检查是否是因为超时结束的（需要 waiter 返回值，或者检查 goal_list）
                    # 这里简化处理，直接使用通用成功原因
                    timeout_occurred = (
                        any("分钟，" in g.get("goal", "") for g in conversation_info.goal_list if isinstance(g, dict))
                        if conversation_info.goal_list
                        else False
                    )
                    final_reason = "等待完成" + (" (超时)" if timeout_occurred else " (收到新消息或中断)")
                elif action == "listening":
                    final_reason = "进入倾听状态"
                elif action in ["rethink_goal", "end_conversation", "block_and_ignore"]:
                    final_reason = f"成功执行 {action}"
                elif action in ["direct_reply", "send_new_message", "say_goodbye"]:
                    # 如果是因为发送成功，设置原因
                    final_reason = "成功发送"
                else:
                    # 其他未知但标记成功的动作
                    final_reason = "动作成功完成"

            elif final_status == "recall" and not action_successful:
                # 如果最终是 recall 且未成功，且不是因为检查不通过（比如生成失败），确保原因合理
                # 保留之前的逻辑，检查是否已有更具体的失败原因
                if not final_reason or final_reason == "动作未成功执行":
                    # 检查是否有 checker 的原因
                    checker_reason = conversation_info.last_reply_rejection_reason
                    if checker_reason:
                        final_reason = f"回复检查不通过: {checker_reason}"
                    else:
                        final_reason = "动作执行失败或被取消"  # 通用失败原因

            # 更新历史记录字典
            if conversation_info.done_action and action_index < len(conversation_info.done_action):
                # 使用 update 方法更新字典，更安全
                conversation_info.done_action[action_index].update(
                    {
                        "status": final_status,  # 最终状态
                        "time_completed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 完成时间
                        "final_reason": final_reason,  # 最终原因
                        "duration_ms": int((time.time() - action_start_time) * 1000),  # 记录耗时（毫秒）
                    }
                )
                logger.debug(
                    f"[私聊][{self.private_name}] 动作 '{action}' 最终状态: {final_status}, 原因: {final_reason}"
                )
            else:
                # 如果索引无效或列表为空，记录错误
                logger.error(f"[私聊][{self.private_name}] 无法更新动作历史记录，索引 {action_index} 无效或列表为空。")

    async def _send_reply(self) -> bool:
        """发送 `self.generated_reply` 中的内容到聊天流"""
        # 检查是否有内容可发送
        if not self.generated_reply:
            logger.warning(f"[私聊][{self.private_name}] 没有生成回复内容，无法发送。")
            return False
        # 检查发送器和聊天流是否已初始化
        if not self.direct_sender:
            logger.error(f"[私聊][{self.private_name}] DirectMessageSender 未初始化，无法发送。")
            return False
        if not self.chat_stream:
            logger.error(f"[私聊][{self.private_name}] ChatStream 未初始化，无法发送。")
            return False

        try:
            reply_content = self.generated_reply
            # 调用发送器发送消息，不指定回复对象
            await self.direct_sender.send_message(
                chat_stream=self.chat_stream,
                content=reply_content,
                reply_to_message=None,  # 私聊通常不需要引用回复
            )
            # 自身发言数量累计 +1
            self.conversation_info.my_message_count += 1
            # 发送成功后，将状态设置回分析，准备下一轮规划
            self.state = ConversationState.ANALYZING
            return True  # 返回成功
        except Exception as e:
            # 捕获发送过程中的异常
            logger.error(f"[私聊][{self.private_name}] 发送消息时失败: {str(e)}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            self.state = ConversationState.ERROR  # 发送失败标记错误状态
            return False  # 返回失败

    async def _send_timeout_message(self):
        """在等待超时后发送一条结束消息"""
        # 检查发送器和聊天流
        if not self.direct_sender or not self.chat_stream:
            logger.warning(f"[私聊][{self.private_name}] 发送器或聊天流未初始化，无法发送超时消息。")
            return
        try:
            # 定义超时消息内容，可以考虑配置化或由 LLM 生成
            timeout_content = "我们好像很久没说话了，先这样吧~"
            # 发送超时消息
            await self.direct_sender.send_message(
                chat_stream=self.chat_stream, content=timeout_content, reply_to_message=None
            )
            logger.info(f"[私聊][{self.private_name}] 已发送超时结束消息。")
            # 发送超时消息后，通常意味着对话结束，调用 stop
            await self.stop()
        except Exception as e:
            # 捕获发送超时消息的异常
            logger.error(f"[私聊][{self.private_name}] 发送超时消息失败: {str(e)}")
