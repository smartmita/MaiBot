# -*- coding: utf-8 -*-
# File: conversation.py
import time
import asyncio
import datetime

# from .message_storage import MongoDBMessageStorage
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat

# from ...config.config import global_config
from typing import Dict, Any, Optional
from ..chat.message import Message
from .pfc_types import ConversationState
from .pfc import ChatObserver, GoalAnalyzer # pfc.py 包含了 GoalAnalyzer，无需重复导入
from .message_sender import DirectMessageSender
from src.common.logger_manager import get_logger
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo  # 确保导入 ConversationInfo
from .reply_generator import ReplyGenerator
from ..chat.chat_stream import ChatStream
from maim_message import UserInfo
from src.plugins.chat.chat_stream import chat_manager
from .pfc_KnowledgeFetcher import KnowledgeFetcher # 注意：这里是 PFC_KnowledgeFetcher.py
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
            self.observation_info.bind_to_chat_observer(self.chat_observer)
            # print(self.chat_observer.get_cached_messages(limit=)

            self.conversation_info = ConversationInfo()
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]初始化对话实例：注册信息组件失败: {e}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            raise
        try:
            logger.info(f"[私聊][{self.private_name}]为 {self.stream_id} 加载初始聊天记录...")
            initial_messages = get_raw_msg_before_timestamp_with_chat(  #
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
                last_user_info = UserInfo.from_dict(last_msg.get("user_info", {}))
                self.observation_info.last_message_sender = last_user_info.user_id
                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")

                logger.info(
                    f"[私聊][{self.private_name}]成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}"
                )

                # 让 ChatObserver 从加载的最后一条消息之后开始同步
                # **** 注意：这里的 last_message_time 设置可能需要 review ****
                # 如果数据库消息时间戳可能不完全连续，直接设置 last_message_time 可能导致 observer 错过消息
                # 更稳妥的方式是让 observer 自己管理其内部的 last_message_time 或 last_message_id
                # 暂时保留，但标记为潜在问题点。如果 observer 逻辑是可靠的，则此行 OK。
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
            # 忽略逻辑 (保持不变)
            if self.ignore_until_timestamp and time.time() < self.ignore_until_timestamp:
                await asyncio.sleep(30)
                continue
            elif self.ignore_until_timestamp and time.time() >= self.ignore_until_timestamp:
                logger.info(f"[私聊][{self.private_name}]忽略时间已到 {self.stream_id}，准备结束对话。")
                self.ignore_until_timestamp = None
                self.should_continue = False
                continue

            try:
                # --- [修改点 1] 在规划前记录新消息状态 ---
                # 记录规划开始时未处理消息的 ID 集合，用于后续判断哪些是新来的
                message_ids_before_planning = set()
                initial_unprocessed_message_count = 0 # <-- 新增：记录规划前的未处理消息数
                if hasattr(self.observation_info, "unprocessed_messages"):
                    message_ids_before_planning = {msg.get("message_id") for msg in self.observation_info.unprocessed_messages if msg.get("message_id")}
                    initial_unprocessed_message_count = len(self.observation_info.unprocessed_messages) # <-- 获取初始数量
                    logger.debug(f"[私聊][{self.private_name}]规划开始，当前未处理消息数: {initial_unprocessed_message_count}, IDs: {message_ids_before_planning}")
                else:
                    logger.warning(
                        f"[私聊][{self.private_name}]ObservationInfo missing 'unprocessed_messages' before planning."
                    )


                # --- 调用 Action Planner (保持不变) ---
                action, reason = await self.action_planner.plan(
                    self.observation_info, self.conversation_info, self.conversation_info.last_successful_reply_action
                )

                # --- [修改点 2] 规划后检查是否有 *过多* 新消息到达 ---
                current_unprocessed_messages = []
                current_unprocessed_message_count = 0
                if hasattr(self.observation_info, "unprocessed_messages"):
                    current_unprocessed_messages = self.observation_info.unprocessed_messages
                    current_unprocessed_message_count = len(current_unprocessed_messages) # <-- 获取当前数量
                else:
                    logger.warning(
                        f"[私聊][{self.private_name}]ObservationInfo missing 'unprocessed_messages' after planning."
                    )

                # 计算规划期间实际新增的消息数量
                new_messages_during_planning_count = 0
                new_message_ids_during_planning = set()
                for msg in current_unprocessed_messages:
                    msg_id = msg.get("message_id")
                    if msg_id and msg_id not in message_ids_before_planning:
                        new_messages_during_planning_count += 1
                        new_message_ids_during_planning.add(msg_id)

                logger.debug(f"[私聊][{self.private_name}]规划结束，当前未处理消息数: {current_unprocessed_message_count}, 规划期间新增: {new_messages_during_planning_count}")

                # **核心逻辑：判断是否中断**
                # 这里的 +2 是根据你的需求来的，代表允许的缓冲
                # 我们比较的是 *规划期间新增的消息数* 是否超过阈值
                if new_messages_during_planning_count > 2:
                    logger.info(
                        f"[私聊][{self.private_name}]规划期间新增消息数 ({new_messages_during_planning_count}) 超过阈值(2)，取消本次行动 '{action}'，重新规划"
                    )
                    # 中断时，重置上次回复状态，因为需要基于最新消息重新决策
                    self.conversation_info.last_successful_reply_action = None
                    # **重要**: 中断时不清空未处理消息，留给下一轮规划处理
                    await asyncio.sleep(0.1) # 短暂暂停避免CPU空转
                    continue # 跳过本轮后续处理，直接进入下一轮循环重新规划

                # --- [修改点 3] 准备执行动作，处理规划时已知的消息 ---
                # 如果决定要回复 (direct_reply 或 send_new_message)，并且规划开始时就有未处理消息
                # 这表示 LLM 规划时已经看到了这些消息
                # 我们需要在发送回复 *后* 清理掉这些规划时已知的消息
                # 注意：这里不再立即清理，清理逻辑移到 _handle_action 成功发送后
                messages_known_during_planning = []
                if action in ["direct_reply", "send_new_message"] and initial_unprocessed_message_count > 0:
                     messages_known_during_planning = [
                        msg for msg_id in message_ids_before_planning
                        if (msg := next((m for m in self.observation_info.unprocessed_messages if m.get("message_id") == msg_id), None)) is not None
                     ]
                     logger.debug(f"[私聊][{self.private_name}]规划时已知 {len(messages_known_during_planning)} 条消息，将在回复成功后清理。")


                # --- 执行动作 ---
                # 将规划时已知需要清理的消息ID集合传递给 _handle_action
                await self._handle_action(action, reason, self.observation_info, self.conversation_info, message_ids_before_planning)

                # --- 检查是否需要结束对话 (逻辑保持不变) ---
                goal_ended = False
                if hasattr(self.conversation_info, "goal_list") and self.conversation_info.goal_list:
                    for goal_item in self.conversation_info.goal_list:
                        current_goal = None # 初始化 current_goal
                        if isinstance(goal_item, dict):
                            current_goal = goal_item.get("goal")
                        elif isinstance(goal_item, str): # 处理直接是字符串的情况
                             current_goal = goal_item

                        # 确保 current_goal 是字符串再比较
                        if isinstance(current_goal, str) and current_goal == "结束对话":
                            goal_ended = True
                            break

                if goal_ended:
                    self.should_continue = False
                    logger.info(f"[私聊][{self.private_name}]检测到'结束对话'目标，停止循环。")

            except Exception as loop_err:
                logger.error(f"[私聊][{self.private_name}]PFC主循环出错: {loop_err}")
                logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
                await asyncio.sleep(1) # 出错时暂停一下

            # 循环间隔
            if self.should_continue:
                await asyncio.sleep(0.1) # 非阻塞的短暂停

        logger.info(f"[私聊][{self.private_name}]PFC 循环结束 for stream_id: {self.stream_id}")


    # --- [修改点 4] 修改 _check_new_messages_after_planning ---
    # 重命名并修改逻辑，用于在 *发送前* 检查是否有过多新消息（兜底检查）
    def _check_interrupt_before_sending(self, message_ids_before_planning: set) -> bool:
        """在发送回复前，最后检查一次是否有过多新消息导致需要中断"""
        if not hasattr(self, "observation_info") or not hasattr(self.observation_info, "unprocessed_messages"):
            logger.warning(
                f"[私聊][{self.private_name}]ObservationInfo 未初始化或缺少 'unprocessed_messages' 属性，无法检查新消息。"
            )
            return False

        current_unprocessed_messages = self.observation_info.unprocessed_messages
        new_messages_count = 0
        for msg in current_unprocessed_messages:
             msg_id = msg.get("message_id")
             if msg_id and msg_id not in message_ids_before_planning:
                 new_messages_count += 1

        # 使用与规划后检查相同的阈值
        if new_messages_count > 2:
            logger.info(
                f"[私聊][{self.private_name}]准备发送时发现新增消息数 ({new_messages_count}) 超过阈值(2)，取消发送并重新规划"
            )
            if hasattr(self, "conversation_info"):
                self.conversation_info.last_successful_reply_action = None
            else:
                logger.warning(
                    f"[私聊][{self.private_name}]ConversationInfo 未初始化，无法重置 last_successful_reply_action。"
                )
            return True # 需要中断
        return False # 不需要中断


    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Message:
        """将消息字典转换为Message对象 (保持不变)"""
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

            user_info = UserInfo.from_dict(msg_dict.get("user_info", {}))

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

    # --- [修改点 5] 修改 _handle_action 签名并调整内部逻辑 ---
    async def _handle_action(
        self,
        action: str,
        reason: str,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        message_ids_before_planning: set # <-- 接收规划前的消息ID集合
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
        reply_sent = False # <-- 新增：标记是否成功发送了回复

        # --- 根据不同的 action 执行 ---
        if action == "direct_reply" or action == "send_new_message":
            # 合并 direct_reply 和 send_new_message 的大部分逻辑
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

                # 1. 生成回复 (传入 action_type)
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type=action
                )
                logger.info(f"{log_prefix} 生成内容: {self.generated_reply}")

                # 2. 检查回复 (逻辑不变)
                self.state = ConversationState.CHECKING
                try:
                    current_goal_str = "" # 初始化
                    if hasattr(conversation_info, 'goal_list') and conversation_info.goal_list:
                         goal_item = conversation_info.goal_list[0] # 取第一个目标
                         if isinstance(goal_item, dict):
                              current_goal_str = goal_item.get('goal', '')
                         elif isinstance(goal_item, str):
                              current_goal_str = goal_item

                    # 确保 chat_history 和 chat_history_str 存在
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

                    # 更新拒绝原因和内容 (仅在不合适或需要重规划时)
                    if not is_suitable or need_replan:
                        conversation_info.last_reply_rejection_reason = check_reason
                        conversation_info.last_rejected_reply_content = self.generated_reply
                    else:
                        # 检查通过，清空上次拒绝记录
                        conversation_info.last_reply_rejection_reason = None
                        conversation_info.last_rejected_reply_content = None

                    if is_suitable:
                        final_reply_to_send = self.generated_reply
                        break # 检查通过，跳出循环
                    elif need_replan:
                        logger.warning(
                            f"{log_prefix} 检查建议重新规划，停止尝试。原因: {check_reason}"
                        )
                        break # 需要重新规划，跳出循环
                except Exception as check_err:
                    logger.error(
                        f"{log_prefix} 调用 ReplyChecker 时出错: {check_err}"
                    )
                    check_reason = f"第 {reply_attempt_count} 次检查过程出错: {check_err}"
                    conversation_info.last_reply_rejection_reason = f"检查过程出错: {check_err}"
                    conversation_info.last_rejected_reply_content = self.generated_reply # 记录出错时尝试的内容
                    break # 检查出错，跳出循环

            # --- 处理生成和检查的结果 ---
            if is_suitable:
                # --- [修改点 6] 发送前最后检查是否需要中断 ---
                if self._check_interrupt_before_sending(message_ids_before_planning):
                    logger.info(f"[私聊][{self.private_name}]生成回复后、发送前发现过多新消息，取消发送，重新规划")
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": f"发送前发现过多新消息，取消发送: {final_reply_to_send}"}
                    )
                    self.conversation_info.last_successful_reply_action = None # 重置状态
                    return # 直接返回，主循环会重新规划

                # 确认发送
                self.generated_reply = final_reply_to_send
                send_success = await self._send_reply() # 调用发送函数

                if send_success:
                    action_successful = True
                    reply_sent = True # 标记回复已发送
                    logger.info(f"[私聊][{self.private_name}]成功发送 '{action}' 回复.")
                    # 清空上次拒绝记录 (再次确保)
                    conversation_info.last_reply_rejection_reason = None
                    conversation_info.last_rejected_reply_content = None

                    # --- [修改点 7] 发送成功后，处理新消息并决定下一轮 prompt 类型 ---
                    # 获取发送后的最新未处理消息列表
                    final_unprocessed_messages = getattr(observation_info, 'unprocessed_messages', [])
                    final_unprocessed_count = len(final_unprocessed_messages)

                    # 计算在生成和发送期间新增的消息数
                    new_messages_during_generation_count = 0
                    for msg in final_unprocessed_messages:
                        msg_id = msg.get("message_id")
                        # 如果消息 ID 不在规划前的集合中，说明是新来的
                        if msg_id and msg_id not in message_ids_before_planning:
                             new_messages_during_generation_count += 1

                    logger.debug(f"[私聊][{self.private_name}]回复发送后，当前未处理消息数: {final_unprocessed_count}, 其中生成/发送期间新增: {new_messages_during_generation_count}")

                    # 根据生成期间是否有新消息，决定下次规划用哪个 prompt
                    if new_messages_during_generation_count > 0:
                        # 有 1 条或更多新消息在生成期间到达
                        logger.info(f"[私聊][{self.private_name}]检测到 {new_messages_during_generation_count} 条在生成/发送期间到达的新消息，下一轮将使用首次回复逻辑处理。")
                        self.conversation_info.last_successful_reply_action = None # 强制下一轮用 PROMPT_INITIAL_REPLY
                    else:
                        # 没有新消息在生成期间到达
                        logger.info(f"[私聊][{self.private_name}]生成/发送期间无新消息，下一轮将根据 '{action}' 使用追问逻辑。")
                        self.conversation_info.last_successful_reply_action = action # 保持状态，下一轮可能用 PROMPT_FOLLOW_UP

                    # --- [修改点 8] 清理规划时已知的消息 ---
                    # 只有在回复成功发送后，才清理掉那些在规划时就已经看到的消息
                    if message_ids_before_planning:
                        await observation_info.clear_processed_messages(message_ids_before_planning)


                else: # 发送失败
                     logger.error(f"[私聊][{self.private_name}]发送 '{action}' 回复失败。")
                     # 发送失败，也认为动作未成功，重置状态
                     action_successful = False
                     self.conversation_info.last_successful_reply_action = None
                     conversation_info.done_action[action_index].update(
                         {"status": "recall", "final_reason": "发送回复时失败"}
                     )

            elif need_replan:
                 # 检查后决定打回动作决策
                 logger.warning(
                    f"[私聊][{self.private_name}]'{action}' 回复检查后决定打回动作决策 (尝试 {reply_attempt_count} 次)。打回原因: {check_reason}"
                 )
                 conversation_info.done_action[action_index].update(
                     {"status": "recall", "final_reason": f"'{action}' 尝试{reply_attempt_count}次后打回: {check_reason}"}
                 )
                 self.conversation_info.last_successful_reply_action = None # 重置状态

            else: # 多次尝试后仍然不合适 (is_suitable is False and not need_replan)
                 logger.warning(
                    f"[私聊][{self.private_name}]经过 {reply_attempt_count} 次尝试，未能生成合适的 '{action}' 回复。最终原因: {check_reason}"
                 )
                 conversation_info.done_action[action_index].update(
                     {"status": "recall", "final_reason": f"'{action}' 尝试{reply_attempt_count}次后失败: {check_reason}"}
                 )
                 self.conversation_info.last_successful_reply_action = None # 重置状态

                 # 如果是 send_new_message 失败，则执行 wait (保持原 fallback 逻辑)
                 if action == "send_new_message":
                     logger.info(f"[私聊][{self.private_name}]由于无法生成合适追问回复，执行 'wait' 操作...")
                     self.state = ConversationState.WAITING
                     await self.waiter.wait(self.conversation_info)
                     wait_action_record = {
                         "action": "wait",
                         "plan_reason": "因 send_new_message 多次尝试失败而执行的后备等待",
                         "status": "done", # wait 本身算完成
                         "time": datetime.datetime.now().strftime("%H:%M:%S"),
                         "final_reason": None,
                     }
                     conversation_info.done_action.append(wait_action_record)
                     action_successful = True # fallback wait 成功
                     # 注意： fallback wait 成功后，last_successful_reply_action 仍然是 None

        # --- 处理其他动作 (保持大部分不变，主要是确保状态重置) ---
        elif action == "rethink_goal":
            self.state = ConversationState.RETHINKING
            try:
                if not hasattr(self, "goal_analyzer"):
                    logger.error(f"[私聊][{self.private_name}]GoalAnalyzer 未初始化，无法重新思考目标。")
                    raise AttributeError("GoalAnalyzer not initialized")
                await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
                action_successful = True
            except Exception as rethink_err:
                logger.error(f"[私聊][{self.private_name}]重新思考目标时出错: {rethink_err}")
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"重新思考目标失败: {rethink_err}"}
                )
            # 无论成功失败，非回复动作都重置 last_successful_reply_action
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None # 清除拒绝原因
            conversation_info.last_rejected_reply_content = None


        elif action == "listening":
            self.state = ConversationState.LISTENING
            logger.info(f"[私聊][{self.private_name}]倾听对方发言...")
            try:
                if not hasattr(self, "waiter"):
                    logger.error(f"[私聊][{self.private_name}]Waiter 未初始化，无法倾听。")
                    raise AttributeError("Waiter not initialized")
                await self.waiter.wait_listening(conversation_info)
                action_successful = True
            except Exception as listen_err:
                logger.error(f"[私聊][{self.private_name}]倾听时出错: {listen_err}")
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"倾听失败: {listen_err}"}
                )
            # 无论成功失败，非回复动作都重置
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None

        elif action == "say_goodbye":
            self.state = ConversationState.GENERATING
            logger.info(f"[私聊][{self.private_name}]执行行动: 生成并发送告别语...")
            try:
                # 1. 生成告别语
                self.generated_reply = await self.reply_generator.generate(
                    observation_info, conversation_info, action_type="say_goodbye"
                )
                logger.info(f"[私聊][{self.private_name}]生成的告别语: {self.generated_reply}")

                # 2. 发送告别语
                if self.generated_reply:
                    # --- [修改点 9] 告别前也检查中断 ---
                    if self._check_interrupt_before_sending(message_ids_before_planning):
                         logger.info(f"[私聊][{self.private_name}]发送告别语前发现过多新消息，取消发送，重新规划")
                         conversation_info.done_action[action_index].update(
                              {"status": "recall", "final_reason": "发送告别语前发现过多新消息"}
                         )
                         self.should_continue = True # 不能结束，需要重规划
                         self.conversation_info.last_successful_reply_action = None # 重置状态
                         return

                    send_success = await self._send_reply()
                    if send_success:
                        action_successful = True
                        reply_sent = True # 标记发送成功
                        logger.info(f"[私聊][{self.private_name}]告别语已发送。")
                        # 发送告别语成功后，通常意味着对话结束
                        self.should_continue = False
                        logger.info(f"[私聊][{self.private_name}]发送告别语流程结束，即将停止对话实例。")
                    else:
                         logger.warning(f"[私聊][{self.private_name}]发送告别语失败。")
                         action_successful = False
                         # 发送失败不应结束对话，可能需要重试或做其他事
                         self.should_continue = True
                         conversation_info.done_action[action_index].update(
                              {"status": "recall", "final_reason": "发送告别语失败"}
                         )
                         self.conversation_info.last_successful_reply_action = None # 重置状态

                else:
                    logger.warning(f"[私聊][{self.private_name}]未能生成告别语内容，无法发送。")
                    action_successful = False
                    self.should_continue = True # 未能生成也不能结束
                    conversation_info.done_action[action_index].update(
                        {"status": "recall", "final_reason": "未能生成告别语内容"}
                    )
                    self.conversation_info.last_successful_reply_action = None

            except Exception as goodbye_err:
                logger.error(f"[私聊][{self.private_name}]生成或发送告别语时出错: {goodbye_err}")
                logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
                action_successful = False
                self.should_continue = True # 出错也不能结束
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"生成或发送告别语时出错: {goodbye_err}"}
                )
                self.conversation_info.last_successful_reply_action = None

        elif action == "end_conversation":
            self.should_continue = False
            logger.info(f"[私聊][{self.private_name}]收到最终结束指令，停止对话...")
            action_successful = True
            # 结束对话也重置状态
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
            # 忽略也重置状态
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None


        else:  # 对应 'wait' 动作
            self.state = ConversationState.WAITING
            logger.info(f"[私聊][{self.private_name}]等待更多信息...")
            try:
                if not hasattr(self, "waiter"):
                    logger.error(f"[私聊][{self.private_name}]Waiter 未初始化，无法等待。")
                    raise AttributeError("Waiter not initialized")
                _timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True
            except Exception as wait_err:
                logger.error(f"[私聊][{self.private_name}]等待时出错: {wait_err}")
                conversation_info.done_action[action_index].update(
                    {"status": "recall", "final_reason": f"等待失败: {wait_err}"}
                )
            # 无论成功失败，非回复动作都重置
            self.conversation_info.last_successful_reply_action = None
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None


        # --- 更新 Action History 状态 ---
        if action_successful:
            conversation_info.done_action[action_index].update(
                {
                    "status": "done",
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            )
            # **注意**: last_successful_reply_action 的更新逻辑已经移到各自的动作处理中
            logger.debug(f"[私聊][{self.private_name}]动作 '{action}' 标记为 'done'")
        else:
            # 如果动作是 recall 状态，在各自的处理逻辑中已经更新了 done_action 的 final_reason
            logger.debug(f"[私聊][{self.private_name}]动作 '{action}' 标记为 'recall' 或失败")

    # --- [修改点 10] _send_reply 返回布尔值表示成功与否 ---
    async def _send_reply(self) -> bool:
        """发送回复，并返回是否发送成功"""
        if not self.generated_reply:
            logger.warning(f"[私聊][{self.private_name}]没有生成回复内容，无法发送。")
            return False # 发送失败

        try:
            reply_content = self.generated_reply

            # 检查依赖项
            if not hasattr(self, "direct_sender") or not self.direct_sender:
                logger.error(f"[私聊][{self.private_name}]DirectMessageSender 未初始化，无法发送回复。")
                return False # 发送失败
            if not self.chat_stream:
                logger.error(f"[私聊][{self.private_name}]ChatStream 未初始化，无法发送回复。")
                return False # 发送失败

            # 发送消息
            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=reply_content)

            # 发送成功后，可以考虑触发 observer 更新，但需谨慎避免竞争条件或重复处理
            # 暂时注释掉，依赖 observer 的自然更新周期
            # self.chat_observer.trigger_update()
            # await self.chat_observer.wait_for_update()

            self.state = ConversationState.ANALYZING  # 更新状态 (例如，可以改为 IDLE 或 WAITING)
            return True # 发送成功

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]发送消息时失败: {str(e)}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            self.state = ConversationState.ANALYZING # 或者设置为 ERROR 状态？
            return False # 发送失败


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
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]发送超时消息失败: {str(e)}")