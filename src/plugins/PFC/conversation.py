import time
import asyncio
import datetime

# from .message_storage import MongoDBMessageStorage
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

from ...config.config import global_config # 确保导入 global_config
from typing import Dict, Any, Optional, Set # 引入 Set
from ..chat.message import Message
from .pfc_types import ConversationState
# 确保导入 ChatObserver 和 GoalAnalyzer (如果 pfc.py 中定义了它们)
# from .pfc import ChatObserver, GoalAnalyzer # 可能需要调整导入路径
from .chat_observer import ChatObserver # 导入 ChatObserver
from .pfc import GoalAnalyzer # 导入 GoalAnalyzer
from .message_sender import DirectMessageSender
from src.common.logger_manager import get_logger
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo  # 确保导入 ConversationInfo
from .reply_generator import ReplyGenerator
from ..chat.chat_stream import ChatStream
from maim_message import UserInfo
from src.plugins.chat.chat_stream import chat_manager
from .pfc_KnowledgeFetcher import KnowledgeFetcher
from .waiter import Waiter

import traceback
from rich.traceback import install

install(extra_lines=3)

logger = get_logger("pfc")


class Conversation:
    """对话类，负责管理单个对话的状态和行为"""

    def __init__(self, stream_id: str, private_name: str):
        """初始化对话实例

        Args:
            stream_id: 聊天流ID
        """
        self.stream_id = stream_id
        self.private_name = private_name
        self.state = ConversationState.INIT
        self.should_continue = False
        self.ignore_until_timestamp: Optional[float] = None

        # 回复相关
        self.generated_reply = ""

        # 初始化 bot_id
        self.bot_id = str(global_config.BOT_QQ) # 从配置中获取

    async def _initialize(self):
        """初始化实例，注册所有组件"""

        try:
            self.action_planner = ActionPlanner(self.stream_id, self.private_name)
            self.goal_analyzer = GoalAnalyzer(self.stream_id, self.private_name)
            self.reply_generator = ReplyGenerator(self.stream_id, self.private_name)
            self.knowledge_fetcher = KnowledgeFetcher(self.private_name)
            self.waiter = Waiter(self.stream_id, self.private_name)
            self.direct_sender = DirectMessageSender(self.private_name)

            # 获取聊天流信息
            self.chat_stream = chat_manager.get_stream(self.stream_id)

            self.stop_action_planner = False
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]初始化对话实例：注册运行组件失败: {e}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            raise

        try:
            # 决策所需要的信息，包括自身自信和观察信息两部分
            # 注册观察器和观测信息
            self.chat_observer = ChatObserver.get_instance(self.stream_id, self.private_name)
            self.chat_observer.start()
            self.observation_info = ObservationInfo(self.private_name)
            # --- 在绑定前设置 bot_id ---
            self.observation_info.bot_id = self.bot_id
            # --- 设置结束 ---
            self.observation_info.bind_to_chat_observer(self.chat_observer)

            self.conversation_info = ConversationInfo()
            # --- 初始化上次拒绝回复的信息 ---
            self.conversation_info.last_reply_rejection_reason = None
            self.conversation_info.last_rejected_reply_content = None

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]初始化对话实例：注册信息组件失败: {e}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            raise
        try:
            logger.info(f"[私聊][{self.private_name}]为 {self.stream_id} 加载初始聊天记录...")
            initial_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=self.stream_id,
                timestamp=time.time(),
                limit=30,  # 加载最近30条作为初始上下文，可以调整
            )
            chat_talking_prompt = await build_readable_messages(
                initial_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
            if initial_messages:
                # 将加载的消息填充到 ObservationInfo 的 chat_history
                self.observation_info.chat_history = initial_messages
                self.observation_info.chat_history_str = chat_talking_prompt + "\n"
                self.observation_info.chat_history_count = len(initial_messages)

                # 更新 ObservationInfo 中的时间戳等信息
                last_msg = initial_messages[-1]
                self.observation_info.last_message_time = last_msg.get("time")
                # 确保 last_msg['user_info'] 是字典
                last_user_info_dict = last_msg.get("user_info", {})
                if isinstance(last_user_info_dict, dict):
                     last_user_info = UserInfo.from_dict(last_user_info_dict)
                     self.observation_info.last_message_sender = str(last_user_info.user_id) # 确保是字符串
                else:
                     logger.warning(f"Initial message user_info is not a dict: {last_user_info_dict}")
                     self.observation_info.last_message_sender = None

                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")

                logger.info(
                    f"[私聊][{self.private_name}]成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}"
                )

                # 让 ChatObserver 从加载的最后一条消息之后开始同步
                self.chat_observer.last_message_time = self.observation_info.last_message_time
                self.chat_observer.last_message_read = last_msg  # 更新 observer 的最后读取记录
            else:
                logger.info(f"[私聊][{self.private_name}]没有找到初始聊天记录。")

        except Exception as load_err:
            logger.error(f"[私聊][{self.private_name}]加载初始聊天记录时出错: {load_err}")
            # 出错也要继续，只是没有历史记录而已
        # 组件准备完成，启动该论对话
        self.should_continue = True
        asyncio.create_task(self.start())

    async def start(self):
        """开始对话流程"""
        try:
            logger.info(f"[私聊][{self.private_name}]对话系统启动中...")
            asyncio.create_task(self._plan_and_action_loop())
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]启动对话系统失败: {e}")
            raise

    async def _plan_and_action_loop(self):
        """思考步，PFC核心循环模块"""
        while self.should_continue:
            # 忽略逻辑
            if self.ignore_until_timestamp and time.time() < self.ignore_until_timestamp:
                await asyncio.sleep(30)
                continue
            elif self.ignore_until_timestamp and time.time() >= self.ignore_until_timestamp:
                logger.info(f"[私聊][{self.private_name}]忽略时间已到 {self.stream_id}，准备结束对话。")
                self.ignore_until_timestamp = None
                self.should_continue = False
                continue
            try:
                # --- 记录规划开始时的时间戳和未处理消息的 ID 集合 ---
                planning_marker_time = time.time()
                # 获取规划开始时未处理消息的 ID 集合
                initial_unprocessed_ids: Set[str] = {
                    msg.get("message_id") for msg in self.observation_info.unprocessed_messages if msg.get("message_id")
                }
                logger.debug(f"[私聊][{self.private_name}]规划开始标记时间: {planning_marker_time}, 初始未处理消息ID数: {len(initial_unprocessed_ids)}")

                # --- 调用 Action Planner ---
                action, reason = await self.action_planner.plan(
                    self.observation_info, self.conversation_info, self.conversation_info.last_successful_reply_action
                )

                # --- 规划后，精确计算规划期间收到的“用户”新消息数 ---
                current_unprocessed_messages = self.observation_info.unprocessed_messages
                new_messages_during_planning = []
                for msg in current_unprocessed_messages:
                    msg_id = msg.get("message_id")
                    # 检查消息ID是否不在初始集合中，且消息时间戳晚于规划开始时间（增加时间判断以防万一）
                    if msg_id and msg_id not in initial_unprocessed_ids and msg.get("time", 0) >= planning_marker_time:
                        new_messages_during_planning.append(msg)

                # 计算这些新消息中来自用户的数量
                new_user_messages_count = 0
                for msg in new_messages_during_planning:
                    user_info_dict = msg.get("user_info", {})
                    sender_id = None
                    if isinstance(user_info_dict, dict):
                        sender_id = str(user_info_dict.get("user_id")) # 确保是字符串
                    # 检查发送者ID是否不是机器人ID
                    if sender_id and sender_id != self.bot_id:
                        new_user_messages_count += 1

                logger.debug(f"[私聊][{self.private_name}]规划期间共收到新消息: {len(new_messages_during_planning)} 条, 其中用户消息: {new_user_messages_count} 条")

                # --- 根据用户新消息数决定是否重新规划 ---
                planning_buffer = 2 # 用户指定的缓冲值
                if new_user_messages_count > planning_buffer:
                    logger.info(
                        f"[私聊][{self.private_name}]规划期间收到 {new_user_messages_count} 条用户新消息 (超过缓冲 {planning_buffer})，放弃当前计划 '{action}'，立即重新规划"
                    )
                    self.conversation_info.last_successful_reply_action = None
                    await asyncio.sleep(0.1)
                    continue # 重新进入循环进行规划

                # --- 如果规划期间用户新消息未超限，则继续执行规划的动作 ---
                # 将 planning_marker_time 和 new_user_messages_count 传递给 _handle_action
                await self._handle_action(action, reason, self.observation_info, self.conversation_info, planning_marker_time, new_user_messages_count)

                # 检查是否需要结束对话 (逻辑不变)
                goal_ended = False
                if hasattr(self.conversation_info, "goal_list") and self.conversation_info.goal_list:
                    for goal_item in self.conversation_info.goal_list:
                        current_goal = None # 初始化
                        if isinstance(goal_item, dict):
                            current_goal = goal_item.get("goal")
                        elif isinstance(goal_item, str): # 处理直接是字符串的情况
                            current_goal = goal_item

                        if current_goal == "结束对话":
                            goal_ended = True
                            break

                if goal_ended:
                    self.should_continue = False
                    logger.info(f"[私聊][{self.private_name}]检测到'结束对话'目标，停止循环。")

            except Exception as loop_err:
                logger.error(f"[私聊][{self.private_name}]PFC主循环出错: {loop_err}")
                logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
                await asyncio.sleep(1)

            if self.should_continue:
                await asyncio.sleep(0.1)

        logger.info(f"[私聊][{self.private_name}]PFC 循环结束 for stream_id: {self.stream_id}")


    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Message:
        """将消息字典转换为Message对象"""
        try:
            chat_info = msg_dict.get("chat_info")
            if chat_info and isinstance(chat_info, dict):
                chat_stream = ChatStream.from_dict(chat_info)
            elif self.chat_stream:
                chat_stream = self.chat_stream
            else:
                chat_stream = chat_manager.get_stream(self.stream_id)
                if not chat_stream:
                    raise ValueError(f"无法确定 ChatStream for stream_id {self.stream_id}")

            user_info_dict = msg_dict.get("user_info", {})
            if isinstance(user_info_dict, dict):
                 user_info = UserInfo.from_dict(user_info_dict)
            else:
                 logger.warning(f"Message user_info is not a dict: {user_info_dict}")
                 user_info = UserInfo(user_id="unknown", user_nickname="Unknown", platform="unknown")

            return Message(
                message_id=msg_dict.get("message_id", f"gen_{time.time()}"),
                chat_stream=chat_stream,
                time=msg_dict.get("time", time.time()),
                user_info=user_info,
                processed_plain_text=msg_dict.get("processed_plain_text", ""),
                detailed_plain_text=msg_dict.get("detailed_plain_text", ""),
            )
        except Exception as e:
            logger.warning(f"[私聊][{self.private_name}]转换消息时出错: {e}")
            raise ValueError(f"无法将字典转换为 Message 对象: {e}") from e

    # --- 修改：_handle_action 接收 planning_marker_time 和 new_user_messages_count ---
    async def _handle_action(
        self, action: str, reason: str, observation_info: ObservationInfo, conversation_info: ConversationInfo, planning_marker_time: float, new_user_messages_during_planning: int
    ):
        """处理规划的行动"""

        logger.debug(f"[私聊][{self.private_name}]执行行动: {action}, 原因: {reason}")

        # 记录action历史
        current_action_record = {
            "action": action,
            "plan_reason": reason,
            "status": "start",
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "final_reason": None,
        }
        if not hasattr(conversation_info, "done_action"):
            conversation_info.done_action = []
        conversation_info.done_action.append(current_action_record)
        action_index = len(conversation_info.done_action) - 1

        action_successful = False

        # --- 根据不同的 action 执行 ---

        if action == "send_new_message":
            max_reply_attempts = 3
            reply_attempt_count = 0
            is_suitable = False
            need_replan = False
            check_reason = "未进行尝试"
            final_reply_to_send = ""

            while reply_attempt_count < max_reply_attempts and not is_suitable:
                reply_attempt_count += 1
                logger.info(
                    f"[私聊][{self.private_name}]尝试生成追问回复 (第 {reply_attempt_count}/{max_reply_attempts} 次)..."
                )
                self.state = ConversationState.GENERATING

                # 1. 生成回复
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="send_new_message"
                )
                logger.info(
                    f"[私聊][{self.private_name}]第 {reply_attempt_count} 次生成的追问回复: {self.generated_reply}"
                )

                # 2. 检查回复
                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = ""
                    if conversation_info.goal_list:
                        first_goal = conversation_info.goal_list[0]
                        if isinstance(first_goal, dict):
                            current_goal_str = first_goal.get("goal", "")
                        elif isinstance(first_goal, str):
                            current_goal_str = first_goal

                    is_suitable, check_reason, need_replan = await self.reply_generator.check_reply(
                        reply=self.generated_reply,
                        goal=current_goal_str,
                        chat_history=observation_info.chat_history,
                        chat_history_str=observation_info.chat_history_str,
                        retry_count=reply_attempt_count - 1,
                    )
                    logger.info(
                        f"[私聊][{self.private_name}]第 {reply_attempt_count} 次追问检查结果: 合适={is_suitable}, 原因='{check_reason}', 需重新规划={need_replan}"
                    )

                    if not is_suitable:
                         setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                         setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    else:
                         setattr(conversation_info, 'last_reply_rejection_reason', None)
                         setattr(conversation_info, 'last_rejected_reply_content', None)

                    if is_suitable:
                        final_reply_to_send = self.generated_reply
                        break
                    elif need_replan:
                        logger.warning(
                            f"[私聊][{self.private_name}]第 {reply_attempt_count} 次追问检查建议重新规划，停止尝试。原因: {check_reason}"
                        )
                        break
                except Exception as check_err:
                    logger.error(
                        f"[私聊][{self.private_name}]第 {reply_attempt_count} 次调用 ReplyChecker (追问) 时出错: {check_err}"
                    )
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                    setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    break

            # 循环结束，处理最终结果
            if is_suitable:
                # 发送合适的回复
                self.generated_reply = final_reply_to_send
                send_success = await self._send_reply()

                if send_success:
                    # 发送成功后，标记处理过的消息
                    await observation_info.mark_messages_processed_up_to(planning_marker_time)

                    # --- 核心逻辑修改：根据规划期间收到的“用户”新消息数决定下一步状态 ---
                    if new_user_messages_during_planning > 0:
                        logger.info(f"[私聊][{self.private_name}] 发送追问成功后，检测到规划期间有 {new_user_messages_during_planning} 条用户新消息，强制重置回复状态以进行新规划。")
                        self.conversation_info.last_successful_reply_action = None # 强制重新规划
                    else:
                        # 只有在规划期间没有用户新消息时，才设置追问状态
                        logger.info(f"[私聊][{self.private_name}] 发送追问成功，规划期间无用户新消息，允许下次进入追问状态。")
                        self.conversation_info.last_successful_reply_action = "send_new_message"
                    # --- 核心逻辑修改结束 ---

                    action_successful = True
                else:
                    logger.error(f"[私聊][{self.private_name}]发送追问回复失败")
                    if action_index < len(conversation_info.done_action):
                        conversation_info.done_action[action_index].update(
                            {"status": "recall", "final_reason": f"发送追问回复失败: {final_reply_to_send}"}
                        )
                    self.conversation_info.last_successful_reply_action = None

            elif need_replan:
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，追问回复决定打回动作决策。打回原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"追问尝试{reply_attempt_count}次后打回: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None

            else:
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，未能生成合适的追问回复。最终原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"追问尝试{reply_attempt_count}次后失败: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None

                # 执行 Wait 操作
                logger.info(f"[私聊][{self.private_name}]由于无法生成合适追问回复，执行 'wait' 操作...")
                self.state = ConversationState.WAITING
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                await self.waiter.wait(self.conversation_info)
                wait_action_record = {
                    "action": "wait",
                    "plan_reason": "因 send_new_message 多次尝试失败而执行的后备等待",
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "final_reason": None,
                }
                conversation_info.done_action.append(wait_action_record)
                action_successful = True

        elif action == "direct_reply":
            max_reply_attempts = 3
            reply_attempt_count = 0
            is_suitable = False
            need_replan = False
            check_reason = "未进行尝试"
            final_reply_to_send = ""

            while reply_attempt_count < max_reply_attempts and not is_suitable:
                reply_attempt_count += 1
                logger.info(
                    f"[私聊][{self.private_name}]尝试生成首次回复 (第 {reply_attempt_count}/{max_reply_attempts} 次)..."
                )
                self.state = ConversationState.GENERATING

                # 1. 生成回复
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="direct_reply"
                )
                logger.info(
                    f"[私聊][{self.private_name}]第 {reply_attempt_count} 次生成的首次回复: {self.generated_reply}"
                )

                # 2. 检查回复
                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = ""
                    if conversation_info.goal_list:
                        first_goal = conversation_info.goal_list[0]
                        if isinstance(first_goal, dict):
                            current_goal_str = first_goal.get("goal", "")
                        elif isinstance(first_goal, str):
                            current_goal_str = first_goal

                    is_suitable, check_reason, need_replan = await self.reply_generator.check_reply(
                        reply=self.generated_reply,
                        goal=current_goal_str,
                        chat_history=observation_info.chat_history,
                        chat_history_str=observation_info.chat_history_str,
                        retry_count=reply_attempt_count - 1,
                    )
                    logger.info(
                        f"[私聊][{self.private_name}]第 {reply_attempt_count} 次首次回复检查结果: 合适={is_suitable}, 原因='{check_reason}', 需重新规划={need_replan}"
                    )

                    if not is_suitable:
                         setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                         setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    else:
                         setattr(conversation_info, 'last_reply_rejection_reason', None)
                         setattr(conversation_info, 'last_rejected_reply_content', None)

                    if is_suitable:
                        final_reply_to_send = self.generated_reply
                        break
                    elif need_replan:
                        logger.warning(
                            f"[私聊][{self.private_name}]第 {reply_attempt_count} 次首次回复检查建议重新规划，停止尝试。原因: {check_reason}"
                        )
                        break
                except Exception as check_err:
                    logger.error(
                        f"[私聊][{self.private_name}]第 {reply_attempt_count} 次调用 ReplyChecker (首次回复) 时出错: {check_err}"
                    )
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                    setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    break

            # 循环结束，处理最终结果
            if is_suitable:
                # 发送合适的回复
                self.generated_reply = final_reply_to_send
                send_success = await self._send_reply()

                if send_success:
                    # 发送成功后，标记处理过的消息
                    await observation_info.mark_messages_processed_up_to(planning_marker_time)

                    # --- 核心逻辑修改：根据规划期间收到的“用户”新消息数决定下一步状态 ---
                    if new_user_messages_during_planning > 0:
                        logger.info(f"[私聊][{self.private_name}] 发送首次回复成功后，检测到规划期间有 {new_user_messages_during_planning} 条用户新消息，强制重置回复状态以进行新规划。")
                        self.conversation_info.last_successful_reply_action = None # 强制重新规划
                    else:
                        # 只有在规划期间没有用户新消息时，才设置追问状态
                        logger.info(f"[私聊][{self.private_name}] 发送首次回复成功，规划期间无用户新消息，允许下次进入追问状态。")
                        self.conversation_info.last_successful_reply_action = "direct_reply"
                    # --- 核心逻辑修改结束 ---

                    action_successful = True
                else:
                    logger.error(f"[私聊][{self.private_name}]发送首次回复失败")
                    if action_index < len(conversation_info.done_action):
                        conversation_info.done_action[action_index].update(
                            {"status": "recall", "final_reason": f"发送首次回复失败: {final_reply_to_send}"}
                        )
                    self.conversation_info.last_successful_reply_action = None

            elif need_replan:
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，首次回复决定打回动作决策。打回原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"首次回复尝试{reply_attempt_count}次后打回: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None

            else:
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，未能生成合适的首次回复。最终原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"首次回复尝试{reply_attempt_count}次后失败: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None

                # 执行 Wait 操作
                logger.info(f"[私聊][{self.private_name}]由于无法生成合适首次回复，执行 'wait' 操作...")
                self.state = ConversationState.WAITING
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                await self.waiter.wait(self.conversation_info)
                wait_action_record = {
                    "action": "wait",
                    "plan_reason": "因 direct_reply 多次尝试失败而执行的后备等待",
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "final_reason": None,
                }
                conversation_info.done_action.append(wait_action_record)
                action_successful = True

        # --- 其他动作的处理逻辑保持不变，但确保在成功后调用 mark_messages_processed_up_to ---
        elif action == "rethink_goal":
            self.state = ConversationState.RETHINKING
            try:
                if not hasattr(self, "goal_analyzer"):
                    logger.error(f"[私聊][{self.private_name}]GoalAnalyzer 未初始化，无法重新思考目标。")
                    raise AttributeError("GoalAnalyzer not initialized")
                await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                action_successful = True
            except Exception as rethink_err:
                logger.error(f"[私聊][{self.private_name}]重新思考目标时出错: {rethink_err}")
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"重新思考目标失败: {rethink_err}"}
                    )
                self.conversation_info.last_successful_reply_action = None

        elif action == "listening":
            self.state = ConversationState.LISTENING
            logger.info(f"[私聊][{self.private_name}]倾听对方发言...")
            try:
                if not hasattr(self, "waiter"):
                    logger.error(f"[私聊][{self.private_name}]Waiter 未初始化，无法倾听。")
                    raise AttributeError("Waiter not initialized")
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                await self.waiter.wait_listening(conversation_info)
                action_successful = True
            except Exception as listen_err:
                logger.error(f"[私聊][{self.private_name}]倾听时出错: {listen_err}")
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"倾听失败: {listen_err}"}
                    )
                self.conversation_info.last_successful_reply_action = None

        elif action == "say_goodbye":
            self.state = ConversationState.GENERATING
            logger.info(f"[私聊][{self.private_name}]执行行动: 生成并发送告别语...")
            try:
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="say_goodbye"
                )
                logger.info(f"[私聊][{self.private_name}]生成的告别语: {self.generated_reply}")
                if self.generated_reply:
                    send_success = await self._send_reply()
                    if send_success:
                        await observation_info.mark_messages_processed_up_to(planning_marker_time)
                        action_successful = True
                        logger.info(f"[私聊][{self.private_name}]告别语已发送。")
                    else:
                        logger.warning(f"[私聊][{self.private_name}]发送告别语失败。")
                        action_successful = False
                        if action_index < len(conversation_info.done_action):
                           conversation_info.done_action[action_index].update(
                               {"status": "recall", "final_reason": "发送告别语失败"}
                           )
                else:
                    logger.warning(f"[私聊][{self.private_name}]未能生成告别语内容，无法发送。")
                    action_successful = False
                    if action_index < len(conversation_info.done_action):
                        conversation_info.done_action[action_index].update(
                            {"status": "recall", "final_reason": "未能生成告别语内容"}
                        )
                self.should_continue = False
                logger.info(f"[私聊][{self.private_name}]发送告别语流程结束，即将停止对话实例。")
            except Exception as goodbye_err:
                logger.error(f"[私聊][{self.private_name}]生成或发送告别语时出错: {goodbye_err}")
                logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
                self.should_continue = False
                action_successful = False
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"生成或发送告别语时出错: {goodbye_err}"}
                    )

        elif action == "end_conversation":
            self.should_continue = False
            logger.info(f"[私聊][{self.private_name}]收到最终结束指令，停止对话...")
            await observation_info.mark_messages_processed_up_to(planning_marker_time)
            action_successful = True

        elif action == "block_and_ignore":
            logger.info(f"[私聊][{self.private_name}]不想再理你了...")
            ignore_duration_seconds = 10 * 60
            self.ignore_until_timestamp = time.time() + ignore_duration_seconds
            logger.info(
                f"[私聊][{self.private_name}]将忽略此对话直到: {datetime.datetime.fromtimestamp(self.ignore_until_timestamp)}"
            )
            self.state = ConversationState.IGNORED
            await observation_info.mark_messages_processed_up_to(planning_marker_time)
            action_successful = True

        else:  # 对应 'wait' 动作
            self.state = ConversationState.WAITING
            logger.info(f"[私聊][{self.private_name}]等待更多信息...")
            try:
                if not hasattr(self, "waiter"):
                    logger.error(f"[私聊][{self.private_name}]Waiter 未初始化，无法等待。")
                    raise AttributeError("Waiter not initialized")
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                _timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True
            except Exception as wait_err:
                logger.error(f"[私聊][{self.private_name}]等待时出错: {wait_err}")
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"等待失败: {wait_err}"}
                    )
                self.conversation_info.last_successful_reply_action = None

        # --- 更新 Action History 状态 ---
        if action_successful:
            if action_index < len(conversation_info.done_action):
                 # 只有在明确不需要强制重新规划时，才在非回复动作后重置状态
                 # 注意：这里的条件与回复动作后的逻辑略有不同，因为非回复动作本身就不会进入追问
                 if action not in ["direct_reply", "send_new_message"]:
                      self.conversation_info.last_successful_reply_action = None

                 conversation_info.done_action[action_index].update(
                     {
                         "status": "done",
                         "time": datetime.datetime.now().strftime("%H:%M:%S"),
                     }
                 )
            else:
                 logger.error(f"[私聊][{self.private_name}]尝试更新无效的 action_index: {action_index}，当前 done_action 长度: {len(conversation_info.done_action)}")

    async def _send_reply(self) -> bool:
        """发送回复，并返回发送是否成功"""
        if not self.generated_reply:
            logger.warning(f"[私聊][{self.private_name}]没有生成回复内容，无法发送。")
            return False

        try:
            reply_content = self.generated_reply

            if not hasattr(self, "direct_sender") or not self.direct_sender:
                logger.error(f"[私聊][{self.private_name}]DirectMessageSender 未初始化，无法发送回复。")
                return False
            if not self.chat_stream:
                logger.error(f"[私聊][{self.private_name}]ChatStream 未初始化，无法发送回复。")
                return False

            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=reply_content)

            self.state = ConversationState.ANALYZING
            return True

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]发送消息或更新状态时失败: {str(e)}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            self.state = ConversationState.ANALYZING
            return False

    async def _send_timeout_message(self):
        """发送超时结束消息"""
        try:
            if not hasattr(self, 'observation_info') or not self.observation_info.chat_history:
                 logger.warning(f"[私聊][{self.private_name}]无法获取聊天历史，无法发送超时消息。")
                 return

            messages = self.observation_info.chat_history[-1:]
            if not messages:
                return

            latest_message_dict = messages[0]
            if not self.chat_stream:
                 logger.error(f"[私聊][{self.private_name}]ChatStream 未初始化，无法发送超时消息。")
                 return
            latest_message = self._convert_to_message(latest_message_dict)

            await self.direct_sender.send_message(
                chat_stream=self.chat_stream, content="[自动消息] 对方长时间未响应，对话已超时。", reply_to_message=latest_message
            )
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]发送超时消息失败: {str(e)}")

