# -*- coding: utf-8 -*-
# File: conversation.py
import time
import asyncio
import datetime

from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from typing import Dict, Any, Optional, Set # <-- 添加 Set 类型提示
from ..chat.message import Message
from .pfc_types import ConversationState
from .pfc import ChatObserver, GoalAnalyzer
from .message_sender import DirectMessageSender
from src.common.logger_manager import get_logger
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from .reply_generator import ReplyGenerator
from ..chat.chat_stream import ChatStream
from .idle_conversation_starter import IdleConversationStarter
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
        """初始化对话实例"""
        self.stream_id = stream_id
        self.private_name = private_name
        self.state = ConversationState.INIT
        self.should_continue = False
        self.ignore_until_timestamp: Optional[float] = None
        self.generated_reply = ""

    async def _initialize(self):
        """初始化实例，注册所有组件 (保持不变)"""
        try:
            self.action_planner = ActionPlanner(self.stream_id, self.private_name)
            self.goal_analyzer = GoalAnalyzer(self.stream_id, self.private_name)
            self.reply_generator = ReplyGenerator(self.stream_id, self.private_name)
            self.knowledge_fetcher = KnowledgeFetcher(self.private_name)
            self.waiter = Waiter(self.stream_id, self.private_name)
            self.direct_sender = DirectMessageSender(self.private_name)
            self.chat_stream = chat_manager.get_stream(self.stream_id)
            self.idle_conversation_starter = IdleConversationStarter(self.stream_id, self.private_name)
            self.stop_action_planner = False
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]初始化对话实例：注册运行组件失败: {e}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            raise

        try:
            self.chat_observer = ChatObserver.get_instance(self.stream_id, self.private_name)
            self.chat_observer.start()
            self.observation_info = ObservationInfo(self.private_name)
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
                limit=30,
            )
            chat_talking_prompt = await build_readable_messages(
                initial_messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
            if initial_messages:
                self.observation_info.chat_history = initial_messages
                self.observation_info.chat_history_str = chat_talking_prompt + "\n"
                self.observation_info.chat_history_count = len(initial_messages)
                last_msg = initial_messages[-1]
                self.observation_info.last_message_time = last_msg.get("time")
                last_user_info = UserInfo.from_dict(last_msg.get("user_info", {}))
                self.observation_info.last_message_sender = last_user_info.user_id
                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")
                logger.info(
                    f"[私聊][{self.private_name}]成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}"
                )
                # --- 注意: 下面两行保持不变，但其健壮性依赖于 ChatObserver 的实现 ---
                self.chat_observer.last_message_time = self.observation_info.last_message_time
                self.chat_observer.last_message_read = last_msg
                await self.idle_conversation_starter.update_last_message_time(self.observation_info.last_message_time)
            else:
                logger.info(f"[私聊][{self.private_name}]没有找到初始聊天记录。")

        except Exception as load_err:
            logger.error(f"[私聊][{self.private_name}]加载初始聊天记录时出错: {load_err}")

        self.should_continue = True
        asyncio.create_task(self.start())
        # 启动空闲对话检测器
        self.idle_conversation_starter.start()
        logger.info(f"[私聊][{self.private_name}]空闲对话检测器已启动")
        asyncio.create_task(self.start())
    async def start(self):
        """开始对话流程 (保持不变)"""
        try:
            logger.info(f"[私聊][{self.private_name}]对话系统启动中...")
            asyncio.create_task(self._plan_and_action_loop())
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]启动对话系统失败: {e}")
            raise

    async def _plan_and_action_loop(self):
        """思考步，PFC核心循环模块"""
        while self.should_continue:
            # 忽略逻辑 (保持不变)
            if self.ignore_until_timestamp and time.time() < self.ignore_until_timestamp:
                # 暂停空闲对话检测器，避免在忽略期间触发
                if hasattr(self, 'idle_conversation_starter') and self.idle_conversation_starter._running:
                    self.idle_conversation_starter.stop()
                    logger.debug(f"[私聊][{self.private_name}]对话被暂时忽略，暂停空闲对话检测")
                await asyncio.sleep(30)
                continue
            elif self.ignore_until_timestamp and time.time() >= self.ignore_until_timestamp:
                logger.info(f"[私聊][{self.private_name}]忽略时间已到 {self.stream_id}，准备结束对话。")
                self.ignore_until_timestamp = None
                self.should_continue = False
                continue
            else:
                # 确保空闲对话检测器在正常对话时是启动的
                if hasattr(self, 'idle_conversation_starter') and not self.idle_conversation_starter._running:
                    self.idle_conversation_starter.start()
                    logger.debug(f"[私聊][{self.private_name}]恢复空闲对话检测")

            try:
                # --- [修改点 1] 在规划前记录未处理消息的 ID 集合 ---
                message_ids_before_planning = set()
                initial_unprocessed_message_count = 0
                if hasattr(self.observation_info, "unprocessed_messages"):
                    message_ids_before_planning = {msg.get("message_id") for msg in self.observation_info.unprocessed_messages if msg.get("message_id")}
                    initial_unprocessed_message_count = len(self.observation_info.unprocessed_messages)
                    logger.debug(f"[私聊][{self.private_name}]规划开始，当前未处理消息数: {initial_unprocessed_message_count}, IDs: {message_ids_before_planning}")
                else:
                    logger.warning(f"[私聊][{self.private_name}]ObservationInfo missing 'unprocessed_messages' before planning.")

                # --- 调用 Action Planner (保持不变) ---
                action, reason = await self.action_planner.plan(
                    self.observation_info, self.conversation_info, self.conversation_info.last_successful_reply_action
                )

                # --- [修改点 2] 规划后检查是否有 *过多* 新消息到达 ---
                current_unprocessed_messages = []
                current_unprocessed_message_count = 0
                if hasattr(self.observation_info, "unprocessed_messages"):
                    current_unprocessed_messages = self.observation_info.unprocessed_messages
                    current_unprocessed_message_count = len(current_unprocessed_messages)
                else:
                    logger.warning(f"[私聊][{self.private_name}]ObservationInfo missing 'unprocessed_messages' after planning.")

                # 计算规划期间实际新增的消息数量
                new_messages_during_planning_count = 0
                for msg in current_unprocessed_messages:
                    msg_id = msg.get("message_id")
                    if msg_id and msg_id not in message_ids_before_planning:
                        new_messages_during_planning_count += 1

                logger.debug(f"[私聊][{self.private_name}]规划结束，当前未处理消息数: {current_unprocessed_message_count}, 规划期间新增: {new_messages_during_planning_count}")

                # **核心逻辑：判断是否中断** (保持不变)
                if new_messages_during_planning_count > 2:
                    logger.info(
                        f"[私聊][{self.private_name}]规划期间新增消息数 ({new_messages_during_planning_count}) 超过阈值(2)，取消本次行动 '{action}'，重新规划"
                    )
                    self.conversation_info.last_successful_reply_action = None
                    await asyncio.sleep(0.1)
                    continue # 跳过本轮后续处理，直接进入下一轮循环重新规划

                # --- 执行动作 (移除 message_ids_before_planning 参数传递) ---
                await self._handle_action(action, reason, self.observation_info, self.conversation_info)

                # --- 检查是否需要结束对话 (逻辑保持不变) ---
                goal_ended = False
                if hasattr(self.conversation_info, "goal_list") and self.conversation_info.goal_list:
                    for goal_item in self.conversation_info.goal_list:
                        current_goal = None
                        if isinstance(goal_item, dict):
                            current_goal = goal_item.get("goal")
                        elif isinstance(goal_item, str):
                             current_goal = goal_item
                        if isinstance(current_goal, str) and current_goal == "结束对话":
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


    # --- 移除 _check_interrupt_before_sending 方法 ---
    # def _check_interrupt_before_sending(self, message_ids_before_planning: set) -> bool:
    #     ... (旧代码移除)

    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Message:
        """将消息字典转换为Message对象 (保持不变)"""
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

            user_info = UserInfo.from_dict(msg_dict.get("user_info", {}))

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

    # --- [修改点 3] 修改 _handle_action 签名并调整内部逻辑 (移除 message_ids_before_planning 参数) ---
    async def _handle_action(
        self,
        action: str,
        reason: str,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo
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
        if not hasattr(conversation_info, "done_action"):
            conversation_info.done_action = []
        conversation_info.done_action.append(current_action_record)
        action_index = len(conversation_info.done_action) - 1

        action_successful = False
        reply_sent = False

        # --- 根据不同的 action 执行 ---
        if action == "direct_reply" or action == "send_new_message":
            # 合并 reply 和 follow-up 的生成/检查逻辑 (保持不变)
            max_reply_attempts = 3
            reply_attempt_count = 0
            is_suitable = False
            need_replan = False
            check_reason = "未进行尝试"
            final_reply_to_send = ""

            while reply_attempt_count < max_reply_attempts and not is_suitable:
                reply_attempt_count += 1
                log_prefix = f"[私聊][{self.private_name}]尝试生成 '{action}' 回复 (第 {reply_attempt_count}/{max_reply_attempts} 次)..."
                logger.info(log_prefix)
                self.state = ConversationState.GENERATING

                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type=action
                )
                logger.info(f"{log_prefix} 生成内容: {self.generated_reply}")

                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = ""
                    if hasattr(conversation_info, 'goal_list') and conversation_info.goal_list:
                         goal_item = conversation_info.goal_list[0]
                         if isinstance(goal_item, dict):
                              current_goal_str = goal_item.get('goal', '')
                         elif isinstance(goal_item, str):
                              current_goal_str = goal_item
                    if is_suitable:
                        # 检查是否有新消息
                        if self._check_new_messages_after_planning():
                            logger.info(f"[私聊][{self.private_name}]生成追问回复期间收到新消息，取消发送，重新规划行动")
                            conversation_info.done_action[action_index].update(
                                {"status": "recall", "final_reason": f"有新消息，取消发送追问: {final_reply_to_send}"}
                            )
                            return  # 直接返回，重新规划

                        # 发送合适的回复
                        self.generated_reply = final_reply_to_send
                        # --- 在这里调用 _send_reply ---
                        await self._send_reply()  # <--- 调用恢复后的函数

                        # 更新状态: 标记上次成功是 send_new_message
                        self.conversation_info.last_successful_reply_action = "send_new_message"
                        action_successful = True  # 标记动作成功

                        # 更新最后消息时间，重置空闲检测计时器
                        await self.idle_conversation_starter.update_last_message_time()

                    elif need_replan:
                        # 打回动作决策
                        logger.warning(
                            f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，追问回复决定打回动作决策。打回原因: {check_reason}"
                        )
                        conversation_info.done_action[action_index].update(
                            {"status": "recall",
                             "final_reason": f"追问尝试{reply_attempt_count}次后打回: {check_reason}"}
                        )
                    chat_history_for_check = getattr(observation_info, 'chat_history', [])
                    chat_history_str_for_check = getattr(observation_info, 'chat_history_str', '')


                    is_suitable, check_reason, need_replan = await self.reply_generator.check_reply(
                        reply=self.generated_reply,
                        goal=current_goal_str,
                        chat_history=chat_history_for_check,
                        chat_history_str=chat_history_str_for_check,
                        retry_count=reply_attempt_count - 1,
                    )
                    logger.info(
                        f"{log_prefix} 检查结果: 合适={is_suitable}, 原因='{check_reason}', 需重新规划={need_replan}"
                    )

                    if not is_suitable or need_replan:
                        conversation_info.last_reply_rejection_reason = check_reason
                        conversation_info.last_rejected_reply_content = self.generated_reply
                    else:
                        conversation_info.last_reply_rejection_reason = None
                        conversation_info.last_rejected_reply_content = None

                    if is_suitable:
                        final_reply_to_send = self.generated_reply
                        break
                    elif need_replan:
                        logger.warning(
                            f"{log_prefix} 检查建议重新规划，停止尝试。原因: {check_reason}"
                        )
                        break
                except Exception as check_err:
                    logger.error(
                        f"{log_prefix} 调用 ReplyChecker 时出错: {check_err}"
                    )
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    conversation_info.last_reply_rejection_reason = f"检查过程出错: {check_err}"
                    conversation_info.last_rejected_reply_content = self.generated_reply
                    break

            # --- 处理生成和检查的结果 ---
            if is_suitable:
                # --- [修改点 4] 记录发送前时间戳 ---
                timestamp_before_sending = time.time()
                logger.debug(f"[私聊][{self.private_name}]准备发送回复，记录发送前时间戳: {timestamp_before_sending}")

                # 确认发送
                self.generated_reply = final_reply_to_send
                send_success = await self._send_reply() # 调用发送函数
                await self.idle_conversation_starter.update_last_message_time()

                if send_success:
                    action_successful = True
                    reply_sent = True
                    logger.info(f"[私聊][{self.private_name}]成功发送 '{action}' 回复.")
                    conversation_info.last_reply_rejection_reason = None
                    conversation_info.last_rejected_reply_content = None

                    # --- [修改点 5] 基于时间戳处理消息和决定下一轮 prompt 类型 ---
                    current_unprocessed_messages = getattr(observation_info, 'unprocessed_messages', [])
                    message_ids_to_clear: Set[str] = set() # 使用 Set 类型
                    new_messages_during_sending_count = 0

                    for msg in current_unprocessed_messages:
                        msg_time = msg.get('time')
                        msg_id = msg.get('message_id')
                        if msg_id and msg_time: # 确保时间和 ID 存在
                            if msg_time < timestamp_before_sending:
                                message_ids_to_clear.add(msg_id)
                            else:
                                # 时间戳大于等于发送前时间戳，视为新消息
                                new_messages_during_sending_count += 1

                    logger.debug(f"[私聊][{self.private_name}]回复发送后，检测到 {len(message_ids_to_clear)} 条发送前消息待清理，{new_messages_during_sending_count} 条发送期间/之后的新消息。")

                    # 清理发送前到达的消息
                    if message_ids_to_clear:
                       await observation_info.clear_processed_messages(message_ids_to_clear)
                    else:
                       logger.debug(f"[私聊][{self.private_name}]没有需要清理的发送前消息。")




                    # 根据发送期间是否有新消息，决定下次规划用哪个 prompt
                    if new_messages_during_sending_count > 0:
                        logger.info(f"[私聊][{self.private_name}]检测到 {new_messages_during_sending_count} 条在发送期间/之后到达的新消息，下一轮将使用首次回复逻辑处理。")
                        self.conversation_info.last_successful_reply_action = None # 强制下一轮用 PROMPT_INITIAL_REPLY
                    else:
                        logger.info(f"[私聊][{self.private_name}]发送期间/之后无新消息，下一轮将根据 '{action}' 使用追问逻辑。")
                        self.conversation_info.last_successful_reply_action = action # 保持状态，下一轮可能用 PROMPT_FOLLOW_UP

                else: # 发送失败
                     logger.error(f"[私聊][{self.private_name}]发送 '{action}' 回复失败。")
                     action_successful = False
                     self.conversation_info.last_successful_reply_action = None
                     conversation_info.done_action[action_index].update(
                         {"status": "recall", "final_reason": "发送回复时失败"}
                     )

            elif need_replan:
                 # 检查后打回动作决策 (保持不变)
                 logger.warning(
                    f"[私聊][{self.private_name}]'{action}' 回复检查后决定打回动作决策 (尝试 {reply_attempt_count} 次)。打回原因: {check_reason}"
                 )
                 conversation_info.done_action[action_index].update(
                     {"status": "recall", "final_reason": f"'{action}' 尝试{reply_attempt_count}次后打回: {check_reason}"}
                 )
                 self.conversation_info.last_successful_reply_action = None

            else: # 多次尝试后仍然不合适 (保持不变)
                 logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，未能生成合适的 '{action}' 回复。最终原因: {check_reason}"
                 )
                 conversation_info.done_action[action_index].update(
                     {"status": "recall", "final_reason": f"'{action}' 尝试{reply_attempt_count}次后失败: {check_reason}"}
                 )
                 self.conversation_info.last_successful_reply_action = None

                 if action == "send_new_message":
                     logger.info(f"[私聊][{self.private_name}]由于无法生成合适追问回复，执行 'wait' 操作...")
                     self.state = ConversationState.WAITING
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
                     self.conversation_info.last_successful_reply_action = None

        # --- 处理其他动作 (保持不变，确保状态重置) ---
        elif action == "rethink_goal":
            self.state = ConversationState.RETHINKING
            try:
                if not hasattr(self, "goal_analyzer"):
                    raise AttributeError("GoalAnalyzer not initialized")
                await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
                action_successful = True
            except Exception as rethink_err:
                logger.error(f"[私聊][{self.private_name}]重新思考目标时出错: {rethink_err}")
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"重新思考目标失败: {rethink_err}"}
                )
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None


        elif action == "listening":
            self.state = ConversationState.LISTENING
            logger.info(f"[私聊][{self.private_name}]倾听对方发言...")
            try:
                if not hasattr(self, "waiter"):
                    raise AttributeError("Waiter not initialized")
                await self.waiter.wait_listening(conversation_info)
                action_successful = True
            except Exception as listen_err:
                logger.error(f"[私聊][{self.private_name}]倾听时出错: {listen_err}")
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"倾听失败: {listen_err}"}
                )
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None

        elif action == "say_goodbye":
            self.state = ConversationState.GENERATING
            logger.info(f"[私聊][{self.private_name}]执行行动: 生成并发送告别语...")
            try:
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="say_goodbye"
                )
                logger.info(f"[私聊][{self.private_name}]生成的告别语: {self.generated_reply}")

                if self.generated_reply:
                    # --- [修改点 6] 告别语发送前记录时间戳 ---
                    timestamp_before_sending_goodbye = time.time()
                    send_success = await self._send_reply()
                    if send_success:
                        action_successful = True
                        reply_sent = True
                        logger.info(f"[私聊][{self.private_name}]告别语已发送。")

                        # --- [修改点 7] 告别语发送后也处理未读消息 ---
                        # （虽然通常之后就结束了，但以防万一）
                        current_unprocessed_messages_goodbye = getattr(observation_info, 'unprocessed_messages', [])
                        message_ids_to_clear_goodbye: Set[str] = set()
                        for msg in current_unprocessed_messages_goodbye:
                           msg_time = msg.get('time')
                           msg_id = msg.get('message_id')
                           if msg_id and msg_time and msg_time < timestamp_before_sending_goodbye:
                               message_ids_to_clear_goodbye.add(msg_id)
                        if message_ids_to_clear_goodbye:
                           await observation_info.clear_processed_messages(message_ids_to_clear_goodbye)

                        self.should_continue = False # 正常结束
                        logger.info(f"[私聊][{self.private_name}]发送告别语流程结束，即将停止对话实例。")
                    else:
                         logger.warning(f"[私聊][{self.private_name}]发送告别语失败。")
                         action_successful = False
                         self.should_continue = True # 发送失败不能结束
                         conversation_info.done_action[action_index].update(
                              {"status": "recall", "final_reason": "发送告别语失败"}
                         )
                         self.conversation_info.last_successful_reply_action = None

                else:
                    logger.warning(f"[私聊][{self.private_name}]未能生成告别语内容，无法发送。")
                    action_successful = False
                    self.should_continue = True
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": "未能生成告别语内容"}
                    )
                    self.conversation_info.last_successful_reply_action = None

            except Exception as goodbye_err:
                logger.error(f"[私聊][{self.private_name}]生成或发送告别语时出错: {goodbye_err}")
                logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
                action_successful = False
                self.should_continue = True
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"生成或发送告别语时出错: {goodbye_err}"}
                )
                self.conversation_info.last_successful_reply_action = None

        elif action == "end_conversation":
            self.should_continue = False
            logger.info(f"[私聊][{self.private_name}]收到最终结束指令，停止对话...")
            action_successful = True
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None


        elif action == "block_and_ignore":
            logger.info(f"[私聊][{self.private_name}]不想再理你了...")
            ignore_duration_seconds = 10 * 60
            self.ignore_until_timestamp = time.time() + ignore_duration_seconds
            logger.info(
                f"[私聊][{self.private_name}]将忽略此对话直到: {datetime.datetime.fromtimestamp(self.ignore_until_timestamp)}"
            )
            self.state = ConversationState.IGNORED
            action_successful = True
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None


        else:  # 对应 'wait' 动作
            self.state = ConversationState.WAITING
            logger.info(f"[私聊][{self.private_name}]等待更多信息...")
            try:
                if not hasattr(self, "waiter"):
                    raise AttributeError("Waiter not initialized")
                _timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True
            except Exception as wait_err:
                logger.error(f"[私聊][{self.private_name}]等待时出错: {wait_err}")
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"等待失败: {wait_err}"}
                )
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None


        # --- 更新 Action History 状态 (保持不变) ---
        if action_successful:
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )
            logger.debug(f"[私聊][{self.private_name}]动作 '{action}' 标记为 'done'")
        else:
            logger.debug(f"[私聊][{self.private_name}]动作 '{action}' 标记为 'recall' 或失败")


    async def _send_reply(self) -> bool:
        """发送回复，并返回是否发送成功 (保持不变)"""
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
            logger.error(f"[私聊][{self.private_name}]发送消息时失败: {str(e)}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            self.state = ConversationState.ANALYZING
            return False


    async def _send_timeout_message(self):
        """发送超时结束消息 (保持不变)"""
        try:
            messages = self.chat_observer.get_cached_messages(limit=1)
            if not messages:
                return
            latest_message = self._convert_to_message(messages[0])
            await self.direct_sender.send_message(
                chat_stream=self.chat_stream, content="TODO:超时消息", reply_to_message=latest_message
            )
            # 停止空闲对话检测器
            if hasattr(self, 'idle_conversation_starter'):
                self.idle_conversation_starter.stop()

            if hasattr(self, 'chat_observer'):
                self.chat_observer.stop()
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]发送超时消息失败: {str(e)}")
