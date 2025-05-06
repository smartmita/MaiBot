import time
import asyncio
import datetime
import traceback
from typing import Dict, Any, Optional, Set, List
from src.common.logger_manager import get_logger
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from maim_message import UserInfo
from src.plugins.chat.chat_stream import chat_manager, ChatStream
from ..chat.message import Message
from ...config.config import global_config

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

        # 初始化所有核心组件为 None，将在 _initialize 中创建
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
            # 可以在这里抛出异常或采取其他错误处理措施

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

            # 5. 启动需要后台运行的组件
            logger.debug(f"[私聊][{self.private_name}] 启动 ChatObserver...")
            self.chat_observer.start()
            if self.idle_conversation_starter:
                logger.debug(f"[私聊][{self.private_name}] 启动 IdleConversationStarter...")
                self.idle_conversation_starter.start()
                logger.info(f"[私聊][{self.private_name}] 空闲对话检测器已启动")

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
                history_slice_for_str = initial_messages[-20:]  # 可配置
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

        # 停止空闲对话检测器
        if self.idle_conversation_starter:
            self.idle_conversation_starter.stop()

        # 解绑 ObservationInfo 与 ChatObserver
        if self.observation_info and self.chat_observer:
            self.observation_info.unbind_from_chat_observer()

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

        # 主循环，只要 should_continue 为 True 就一直运行
        while self.should_continue:
            loop_iter_start_time = time.time()  # 记录本次循环开始时间
            logger.debug(f"[私聊][{self.private_name}] 开始新一轮循环迭代 ({loop_iter_start_time:.2f})")

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
                    self.observation_info, self.conversation_info, self.conversation_info.last_successful_reply_action
                )
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
                # 根据 Message 类的定义，可能还需要其他字段
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
        包括生成回复、调用检查器、发送消息、等待、思考目标等。
        并根据执行结果和规则更新对话状态。
        """
        # 检查初始化状态
        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 尝试在未初始化状态下处理动作 '{action}'。")
            return

        logger.info(f"[私聊][{self.private_name}] 开始处理动作: {action}, 原因: {reason}")
        action_start_time = time.time()  # 记录动作开始时间，用于计算耗时

        # --- 准备动作历史记录 ---
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

            # 1. 处理需要生成并可能发送消息的动作
            if action in ["direct_reply", "send_new_message"]:
                # --- a. 生成回复 ---
                self.state = ConversationState.GENERATING  # 更新对话状态
                if not self.reply_generator:
                    # 检查依赖组件是否存在
                    raise RuntimeError("ReplyGenerator 未初始化")
                # 调用 ReplyGenerator 生成回复内容
                generated_content = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type=action
                )
                logger.info(
                    f"[私聊][{self.private_name}] 动作 '{action}': 生成内容: '{generated_content[:100]}...'"
                )  # 日志中截断长内容

                # 检查生成内容是否有效
                if not generated_content or generated_content.startswith("抱歉"):
                    # 如果生成失败或返回错误提示
                    logger.warning(f"[私聊][{self.private_name}] 动作 '{action}': 生成内容为空或为错误提示，标记失败。")
                    final_reason = "生成内容无效"
                    final_status = "recall"  # 标记为 recall
                    # 重置追问状态，因为本次回复失败
                    conversation_info.last_successful_reply_action = None
                else:
                    # --- b. 检查回复 (如果生成成功) ---
                    self.state = ConversationState.CHECKING  # 更新状态为检查中
                    if not self.reply_checker:
                        raise RuntimeError("ReplyChecker 未初始化")

                    # 准备检查所需的上下文信息
                    current_goal_str: str = ""  # 当前对话目标字符串
                    if conversation_info.goal_list:
                        # 通常检查最新的目标
                        goal_item = conversation_info.goal_list[-1]
                        if isinstance(goal_item, dict):
                            current_goal_str = goal_item.get("goal", "")
                        elif isinstance(goal_item, str):
                            current_goal_str = goal_item
                    # 获取用于检查的聊天记录 (列表和字符串形式)
                    chat_history_for_check: List[Dict[str, Any]] = getattr(observation_info, "chat_history", [])
                    chat_history_text_for_check: str = getattr(observation_info, "chat_history_str", "")
                    # 当前重试次数 (如果未来加入重试逻辑，这里需要传递实际次数)
                    retry_count: int = 0

                    logger.debug(f"[私聊][{self.private_name}] 调用 ReplyChecker 检查回复...")
                    # 调用 ReplyChecker 的 check 方法
                    is_suitable, check_reason, need_replan_from_checker = await self.reply_checker.check(
                        reply=generated_content,
                        goal=current_goal_str,
                        chat_history=chat_history_for_check,  # 传递列表形式的历史记录
                        chat_history_text=chat_history_text_for_check,  # 传递文本形式的历史记录
                        retry_count=retry_count,
                    )
                    logger.info(
                        f"[私聊][{self.private_name}] ReplyChecker 检查结果: 合适={is_suitable}, 原因='{check_reason}', 需重规划={need_replan_from_checker}"
                    )

                    # --- c. 处理检查结果 ---
                    if not is_suitable or need_replan_from_checker:
                        # 如果检查不通过或 Checker 要求重新规划
                        # 记录拒绝原因和内容，供下次生成时参考
                        conversation_info.last_reply_rejection_reason = check_reason
                        conversation_info.last_rejected_reply_content = generated_content
                        # 设置最终状态和原因
                        final_reason = f"回复检查不通过: {check_reason}"
                        final_status = "recall"  # 标记为 recall
                        # 重置追问状态
                        conversation_info.last_successful_reply_action = None
                        logger.warning(f"[私聊][{self.private_name}] 动作 '{action}' 因回复检查失败而被拒绝。")
                        # 注意：如果 need_replan_from_checker 为 True，后续逻辑会因 final_status 为 recall 而可能触发重新规划
                    else:
                        # 如果检查通过
                        # 清除上次的拒绝信息
                        conversation_info.last_reply_rejection_reason = None
                        conversation_info.last_rejected_reply_content = None

                        # --- d. 发送回复 ---
                        self.generated_reply = generated_content  # 存储待发送内容
                        timestamp_before_sending = time.time()  # 记录发送前时间戳
                        logger.debug(
                            f"[私聊][{self.private_name}] 动作 '{action}': 回复检查通过，记录发送前时间戳: {timestamp_before_sending:.2f}"
                        )
                        self.state = ConversationState.SENDING  # 更新状态为发送中
                        # 调用内部发送方法
                        send_success = await self._send_reply()
                        send_end_time = time.time()  # 记录发送结束时间

                        if send_success:
                            # 如果发送成功
                            action_successful = True  # 标记动作成功
                            logger.info(f"[私聊][{self.private_name}] 动作 '{action}': 成功发送回复.")
                            # 更新空闲计时器
                            if self.idle_conversation_starter:
                                await self.idle_conversation_starter.update_last_message_time(send_end_time)

                            # --- e. 清理已处理消息 ---
                            current_unprocessed_messages = getattr(observation_info, "unprocessed_messages", [])
                            message_ids_to_clear: Set[str] = set()
                            # 遍历所有未处理消息
                            for msg in current_unprocessed_messages:
                                msg_time = msg.get("time")
                                msg_id = msg.get("message_id")
                                sender_id = msg.get("user_info", {}).get("user_id")
                                # 规则：只清理【发送前】收到的、【来自他人】的消息
                                if (
                                    msg_id
                                    and msg_time
                                    and sender_id != self.bot_qq_str
                                    and msg_time < timestamp_before_sending
                                ):
                                    message_ids_to_clear.add(msg_id)
                            # 如果有需要清理的消息，调用清理方法
                            if message_ids_to_clear:
                                logger.debug(
                                    f"[私聊][{self.private_name}] 准备清理 {len(message_ids_to_clear)} 条发送前(他人)消息: {message_ids_to_clear}"
                                )
                                await observation_info.clear_processed_messages(message_ids_to_clear)
                            else:
                                logger.debug(f"[私聊][{self.private_name}] 没有需要清理的发送前(他人)消息。")

                            # --- f. 决定下一轮规划类型 ---
                            # 从 conversation_info 获取【规划期间】收到的【他人】新消息数量
                            other_new_msg_count_during_planning = getattr(
                                conversation_info, "other_new_messages_during_planning_count", 0
                            )

                            # 规则：如果规划期间收到他人新消息 (0 < count <= 2)，则下一轮强制初始回复
                            if other_new_msg_count_during_planning > 0:
                                logger.info(
                                    f"[私聊][{self.private_name}] 因规划期间收到 {other_new_msg_count_during_planning} 条他人新消息，下一轮强制使用【初始回复】逻辑。"
                                )
                                conversation_info.last_successful_reply_action = None  # 强制初始回复
                            else:
                                # 规则：如果规划期间【没有】收到他人新消息，则允许追问
                                logger.info(
                                    f"[私聊][{self.private_name}] 规划期间无他人新消息，下一轮【允许】使用追问逻辑 (基于 '{action}')。"
                                )
                                conversation_info.last_successful_reply_action = action  # 允许追问

                        else:
                            # 如果发送失败
                            logger.error(f"[私聊][{self.private_name}] 动作 '{action}': 发送回复失败。")
                            final_status = "recall"  # 发送失败，标记为 recall
                            final_reason = "发送回复时失败"
                            # 重置追问状态
                            conversation_info.last_successful_reply_action = None

            # 2. 处理发送告别语动作
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
                    final_status = "done"
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

            # 4. 处理倾听动作
            elif action == "listening":
                self.state = ConversationState.LISTENING
                if not self.waiter:
                    raise RuntimeError("Waiter 未初始化")
                logger.info(f"[私聊][{self.private_name}] 动作 'listening': 进入倾听状态...")
                # 调用 Waiter 的倾听等待方法，内部会处理超时
                await self.waiter.wait_listening(conversation_info)
                action_successful = True  # listening 动作本身执行即视为成功，后续由新消息或超时驱动

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

            # 7. 处理等待动作
            elif action == "wait":
                self.state = ConversationState.WAITING
                if not self.waiter:
                    raise RuntimeError("Waiter 未初始化")
                logger.info(f"[私聊][{self.private_name}] 动作 'wait': 进入等待状态...")
                # 调用 Waiter 的常规等待方法，内部处理超时
                timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True  # wait 动作本身执行即视为成功
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
                if not final_reason or final_reason == "动作未成功执行":
                    # 排除已经被 checker 设置的原因
                    if not conversation_info.last_reply_rejection_reason:
                        final_reason = "动作执行失败或被取消"  # 提供一个更通用的失败原因

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
