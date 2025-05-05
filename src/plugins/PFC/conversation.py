import time
import asyncio
import datetime

# from .message_storage import MongoDBMessageStorage
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

from ...config.config import global_config # 确保导入 global_config
from typing import Dict, Any, Optional
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
                # --- 记录规划开始时的时间戳和未处理消息数 ---
                # 使用 time.time() 获取当前时间戳作为标记点，更准确反映规划开始的时刻
                planning_marker_time = time.time()
                initial_unprocessed_count = self.observation_info.new_messages_count
                logger.debug(f"[私聊][{self.private_name}]规划开始标记时间: {planning_marker_time}, 初始未处理: {initial_unprocessed_count}")


                # --- 调用 Action Planner ---
                # 传递 self.conversation_info.last_successful_reply_action
                action, reason = await self.action_planner.plan(
                    self.observation_info, self.conversation_info, self.conversation_info.last_successful_reply_action
                )

                # --- 规划后检查是否有 *过多* 新消息到达 ---
                # 检查规划期间（调用 plan 函数的时间段内）新收到的消息数
                current_unprocessed_count = self.observation_info.new_messages_count
                # planning_buffer = 2 # 用户指定的缓冲值
                planning_buffer = 2 # 使用用户指定的缓冲值
                new_messages_during_planning = current_unprocessed_count - initial_unprocessed_count

                if new_messages_during_planning > planning_buffer:
                    logger.info(
                        f"[私聊][{self.private_name}]规划期间收到 {new_messages_during_planning} 条新消息 (超过缓冲 {planning_buffer})，放弃当前计划 '{action}'，立即重新规划"
                    )
                    # 重置上次成功回复状态，因为要响应新消息
                    self.conversation_info.last_successful_reply_action = None
                    # 记录被放弃的规划动作 (可选)
                    # current_action_record = { ... status: "recalled_before_execution" ... }
                    # conversation_info.done_action.append(current_action_record)
                    await asyncio.sleep(0.1) # 短暂等待
                    continue # 重新进入循环进行规划

                # --- 如果规划期间新消息未超限，则继续执行规划的动作 ---
                # 将 planning_marker_time 传递给 _handle_action
                await self._handle_action(action, reason, self.observation_info, self.conversation_info, planning_marker_time)

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
                await asyncio.sleep(1) # 发生错误时等待一段时间

            if self.should_continue:
                await asyncio.sleep(0.1) # 保持循环间的短暂间隔

        logger.info(f"[私聊][{self.private_name}]PFC 循环结束 for stream_id: {self.stream_id}")

    # --- 移除 _check_new_messages_during_action 方法 ---
    # 这个方法不再需要，因为检查逻辑已合并到 _plan_and_action_loop 中，
    # 并且 _handle_action 内部不再需要打断执行。
    # def _check_new_messages_during_action(self, planning_marker_time: float, buffer: int = 0) -> bool:
    #     # ... (旧代码) ...
    #     pass

    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Message:
        """将消息字典转换为Message对象"""
        try:
            # 尝试从 msg_dict 直接获取 chat_stream，如果失败则从全局 chat_manager 获取
            chat_info = msg_dict.get("chat_info")
            if chat_info and isinstance(chat_info, dict):
                chat_stream = ChatStream.from_dict(chat_info)
            elif self.chat_stream:  # 使用实例变量中的 chat_stream
                chat_stream = self.chat_stream
            else:  # Fallback: 尝试从 manager 获取 (可能需要 stream_id)
                chat_stream = chat_manager.get_stream(self.stream_id)
                if not chat_stream:
                    raise ValueError(f"无法确定 ChatStream for stream_id {self.stream_id}")

            user_info_dict = msg_dict.get("user_info", {})
            if isinstance(user_info_dict, dict):
                 user_info = UserInfo.from_dict(user_info_dict)
            else:
                 logger.warning(f"Message user_info is not a dict: {user_info_dict}")
                 # 根据需要返回默认 UserInfo 或抛出错误
                 user_info = UserInfo(user_id="unknown", user_nickname="Unknown", platform="unknown")


            return Message(
                message_id=msg_dict.get("message_id", f"gen_{time.time()}"),  # 提供默认 ID
                chat_stream=chat_stream,  # 使用确定的 chat_stream
                time=msg_dict.get("time", time.time()),  # 提供默认时间
                user_info=user_info,
                processed_plain_text=msg_dict.get("processed_plain_text", ""),
                detailed_plain_text=msg_dict.get("detailed_plain_text", ""),
            )
        except Exception as e:
            logger.warning(f"[私聊][{self.private_name}]转换消息时出错: {e}")
            # 可以选择返回 None 或重新抛出异常，这里选择重新抛出以指示问题
            raise ValueError(f"无法将字典转换为 Message 对象: {e}") from e

    # --- 修改：_handle_action 接收 planning_marker_time ---
    async def _handle_action(
        self, action: str, reason: str, observation_info: ObservationInfo, conversation_info: ConversationInfo, planning_marker_time: float
    ):
        """处理规划的行动"""

        logger.debug(f"[私聊][{self.private_name}]执行行动: {action}, 原因: {reason}")

        # 记录action历史 (逻辑不变)
        current_action_record = {
            "action": action,
            "plan_reason": reason,
            "status": "start",
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "final_reason": None,
        }
        # 确保 done_action 列表存在
        if not hasattr(conversation_info, "done_action"):
            conversation_info.done_action = []
        conversation_info.done_action.append(current_action_record)
        action_index = len(conversation_info.done_action) - 1

        action_successful = False  # 用于标记动作是否成功完成

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

                # --- 移除生成前的检查 ---
                # if self._check_new_messages_during_action(planning_marker_time): ...

                # 1. 生成回复 (调用 generate 时传入 action_type)
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="send_new_message"
                )
                logger.info(
                    f"[私聊][{self.private_name}]第 {reply_attempt_count} 次生成的追问回复: {self.generated_reply}"
                )

                # --- 移除检查前的检查 ---
                # if self._check_new_messages_during_action(planning_marker_time): ...

                # 2. 检查回复 (逻辑不变)
                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = "" # 初始化
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

                    # 记录失败原因和内容到 conversation_info
                    if not is_suitable:
                         # 确保属性存在
                         setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                         setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    else:
                         # 清除上次失败记录
                         setattr(conversation_info, 'last_reply_rejection_reason', None)
                         setattr(conversation_info, 'last_rejected_reply_content', None)

                    if is_suitable:
                        final_reply_to_send = self.generated_reply
                        break
                    elif need_replan:
                        logger.warning(
                            f"[私聊][{self.private_name}]第 {reply_attempt_count} 次追问检查建议重新规划，停止尝试。原因: {check_reason}"
                        )
                        break # 跳出循环，后续会处理 need_replan
                except Exception as check_err:
                    logger.error(
                        f"[私聊][{self.private_name}]第 {reply_attempt_count} 次调用 ReplyChecker (追问) 时出错: {check_err}"
                    )
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    # 记录失败
                    setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                    setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    break # 出错也跳出循环

            # 循环结束，处理最终结果
            if is_suitable:
                # --- 移除发送前的检查 ---
                # if self._check_new_messages_during_action(planning_marker_time): ...

                # 发送合适的回复
                self.generated_reply = final_reply_to_send
                send_success = await self._send_reply()  # 调用发送函数，获取发送结果

                if send_success:
                    # --- 发送成功后，标记处理过的消息 ---
                    # 使用 planning_marker_time 标记规划开始前的消息为已处理
                    await observation_info.mark_messages_processed_up_to(planning_marker_time)
                    # 更新状态: 标记上次成功是 send_new_message
                    self.conversation_info.last_successful_reply_action = "send_new_message"
                    action_successful = True  # 标记动作成功
                else:
                    # 发送失败处理
                    logger.error(f"[私聊][{self.private_name}]发送追问回复失败")
                    # 确保 action_index 有效
                    if action_index < len(conversation_info.done_action):
                        conversation_info.done_action[action_index].update(
                            {"status": "recall", "final_reason": f"发送追问回复失败: {final_reply_to_send}"}
                        )
                    self.conversation_info.last_successful_reply_action = None # 发送失败，重置状态

            elif need_replan:
                # 打回动作决策
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，追问回复决定打回动作决策。打回原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"追问尝试{reply_attempt_count}次后打回: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None # 重置成功状态

            else:
                # 追问失败
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
                # --- Wait 操作也需要标记处理过的消息 ---
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
                action_successful = True # Wait 本身算成功完成

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

                # --- 移除生成前的检查 ---
                # if self._check_new_messages_during_action(planning_marker_time): ...

                # 1. 生成回复
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="direct_reply"
                )
                logger.info(
                    f"[私聊][{self.private_name}]第 {reply_attempt_count} 次生成的首次回复: {self.generated_reply}"
                )

                # --- 移除检查前的检查 ---
                # if self._check_new_messages_during_action(planning_marker_time): ...

                # 2. 检查回复
                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = "" # 初始化
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

                    # 记录失败原因和内容
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
                        break # 跳出循环
                except Exception as check_err:
                    logger.error(
                        f"[私聊][{self.private_name}]第 {reply_attempt_count} 次调用 ReplyChecker (首次回复) 时出错: {check_err}"
                    )
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    # 记录失败
                    setattr(conversation_info, 'last_reply_rejection_reason', check_reason)
                    setattr(conversation_info, 'last_rejected_reply_content', self.generated_reply)
                    break # 出错也跳出循环

            # 循环结束，处理最终结果
            if is_suitable:
                 # --- 移除发送前的检查 ---
                 # if self._check_new_messages_during_action(planning_marker_time): ...

                # 发送合适的回复
                self.generated_reply = final_reply_to_send
                send_success = await self._send_reply() # 调用发送函数

                if send_success:
                    # --- 发送成功后，标记处理过的消息 ---
                    await observation_info.mark_messages_processed_up_to(planning_marker_time)
                    # 更新状态: 标记上次成功是 direct_reply
                    self.conversation_info.last_successful_reply_action = "direct_reply"
                    action_successful = True  # 标记动作成功
                else:
                    # 发送失败处理
                    logger.error(f"[私聊][{self.private_name}]发送首次回复失败")
                    if action_index < len(conversation_info.done_action):
                        conversation_info.done_action[action_index].update(
                            {"status": "recall", "final_reason": f"发送首次回复失败: {final_reply_to_send}"}
                        )
                    self.conversation_info.last_successful_reply_action = None # 发送失败，重置状态

            elif need_replan:
                # 打回动作决策
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，首次回复决定打回动作决策。打回原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"首次回复尝试{reply_attempt_count}次后打回: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None # 重置成功状态

            else:
                # 首次回复失败
                logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，未能生成合适的首次回复。最终原因: {check_reason}"
                )
                if action_index < len(conversation_info.done_action):
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"首次回复尝试{reply_attempt_count}次后失败: {check_reason}"}
                    )
                self.conversation_info.last_successful_reply_action = None

                # 执行 Wait 操作 (保持原有逻辑)
                logger.info(f"[私聊][{self.private_name}]由于无法生成合适首次回复，执行 'wait' 操作...")
                self.state = ConversationState.WAITING
                # --- Wait 操作也需要标记处理过的消息 ---
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
                action_successful = True # Wait 本身算成功

        elif action == "rethink_goal":
            self.state = ConversationState.RETHINKING
            try:
                if not hasattr(self, "goal_analyzer"):
                    logger.error(f"[私聊][{self.private_name}]GoalAnalyzer 未初始化，无法重新思考目标。")
                    raise AttributeError("GoalAnalyzer not initialized")

                # --- 移除 rethink_goal 前的检查 ---
                # if self._check_new_messages_during_action(planning_marker_time): ...

                await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
                # --- rethink_goal 后标记处理过的消息 ---
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

                # --- listening 动作开始前标记处理过的消息 ---
                # 因为 listening 本身是等待行为，需要在开始等待前处理掉规划时看到的消息
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                await self.waiter.wait_listening(conversation_info)
                action_successful = True  # Listening 完成就算成功
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
                 # --- 移除告别前的检查 ---
                 # if self._check_new_messages_during_action(planning_marker_time): ...

                # 1. 生成告别语
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="say_goodbye"
                )
                logger.info(f"[私聊][{self.private_name}]生成的告别语: {self.generated_reply}")

                # 2. 发送告别语
                if self.generated_reply:
                    send_success = await self._send_reply()
                    if send_success:
                        # --- 发送成功后标记处理过的消息 ---
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

                # 3. 结束对话
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
            # --- 结束对话前标记处理过的消息 ---
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
             # --- 屏蔽前标记处理过的消息 ---
            await observation_info.mark_messages_processed_up_to(planning_marker_time)
            action_successful = True

        else:  # 对应 'wait' 动作
            self.state = ConversationState.WAITING
            logger.info(f"[私聊][{self.private_name}]等待更多信息...")
            try:
                if not hasattr(self, "waiter"):
                    logger.error(f"[私聊][{self.private_name}]Waiter 未初始化，无法等待。")
                    raise AttributeError("Waiter not initialized")

                # --- Wait 开始前标记处理过的消息 ---
                await observation_info.mark_messages_processed_up_to(planning_marker_time)
                _timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True  # Wait 完成就算成功
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
                 conversation_info.done_action[action_index].update(
                     {
                         "status": "done",
                         "time": datetime.datetime.now().strftime("%H:%M:%S"),
                     }
                 )
                 if action not in ["direct_reply", "send_new_message"]:
                     self.conversation_info.last_successful_reply_action = None
            else:
                 logger.error(f"[私聊][{self.private_name}]尝试更新无效的 action_index: {action_index}，当前 done_action 长度: {len(conversation_info.done_action)}")

    async def _send_reply(self) -> bool:
        """发送回复，并返回发送是否成功"""
        if not self.generated_reply:
            logger.warning(f"[私聊][{self.private_name}]没有生成回复内容，无法发送。")
            return False # 发送失败

        try:
            reply_content = self.generated_reply

            if not hasattr(self, "direct_sender") or not self.direct_sender:
                logger.error(f"[私聊][{self.private_name}]DirectMessageSender 未初始化，无法发送回复。")
                return False # 发送失败
            if not self.chat_stream:
                logger.error(f"[私聊][{self.private_name}]ChatStream 未初始化，无法发送回复。")
                return False # 发送失败

            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=reply_content)

            # 触发 observer 更新的逻辑保持注释，依赖自动轮询
            # self.chat_observer.trigger_update()
            # await self.chat_observer.wait_for_update()

            self.state = ConversationState.ANALYZING
            return True # 发送成功

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]发送消息或更新状态时失败: {str(e)}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            self.state = ConversationState.ANALYZING
            return False # 发送失败

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

