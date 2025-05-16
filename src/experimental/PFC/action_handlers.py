from abc import ABC, abstractmethod
import time
import asyncio
import datetime
import traceback
import json
from typing import Optional, Set, TYPE_CHECKING, List, Tuple # 确保导入 List 和 Tuple

from src.chat.emoji_system.emoji_manager import emoji_manager
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.utils.chat_message_builder import build_readable_messages
from PFC.pfc_types import ConversationState
from PFC.observation_info import ObservationInfo
from PFC.conversation_info import ConversationInfo
from src.chat.utils.utils_image import image_path_to_base64
from maim_message import Seg, UserInfo
from src.chat.message_receive.message import MessageSending, MessageSet
from src.chat.message_receive.message_sender import message_manager
# PFC.message_sender 已经包含 DirectMessageSender，这里不再需要单独导入

if TYPE_CHECKING:
    from PFC.conversation import Conversation

logger = get_logger("pfc_action_handlers")


class ActionHandler(ABC):
    """处理动作的抽象基类。"""
    def __init__(self, conversation: "Conversation"):
        self.conversation = conversation
        self.logger = logger

    @abstractmethod
    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        """
        执行动作。

        返回:
            一个元组，包含:
            - action_successful (bool): 动作是否成功。
            - final_status (str): 动作的最终状态。
            - final_reason (str): 最终状态的原因。
        """
        pass

    async def _send_reply_or_segments(self, segments_data: list[Seg], content_for_log: str) -> bool:
        """
        辅助函数，用于发送消息（文本或图片段）。
        """
        if not self.conversation.direct_sender:
            self.logger.error(f"[私聊][{self.conversation.private_name}] DirectMessageSender 未初始化，无法发送。")
            return False
        if not self.conversation.chat_stream:
            self.logger.error(f"[私聊][{self.conversation.private_name}] ChatStream 未初始化，无法发送。")
            return False

        try:
            final_segments = Seg(type="seglist", data=segments_data)
            await self.conversation.direct_sender.send_message(
                chat_stream=self.conversation.chat_stream,
                segments=final_segments,
                reply_to_message=None,
                content=content_for_log
            )
            if self.conversation.conversation_info:
                self.conversation.conversation_info.my_message_count += 1
            self.conversation.state = ConversationState.ANALYZING
            return True
        except Exception as e:
            self.logger.error(f"[私聊][{self.conversation.private_name}] 发送消息时失败: {str(e)}")
            self.logger.error(f"[私聊][{self.conversation.private_name}] {traceback.format_exc()}")
            self.conversation.state = ConversationState.ERROR
            return False

    async def _update_bot_message_in_history(
        self,
        send_time: float,
        message_content: str, # 对于图片，这应该是描述性的文本
        observation_info: ObservationInfo,
        message_id_prefix: str = "bot_sent_"
    ):
        """在机器人发送消息后，更新 ObservationInfo 中的聊天记录。"""
        if not self.conversation.bot_qq_str:
            self.logger.warning(f"[私聊][{self.conversation.private_name}] Bot QQ ID 未知，无法更新机器人消息历史。")
            return

        bot_message_dict = {
            "message_id": f"{message_id_prefix}{send_time}",
            "time": send_time,
            "user_info": {
                "user_id": self.conversation.bot_qq_str,
                "user_nickname": global_config.BOT_NICKNAME,
                "platform": self.conversation.chat_stream.platform if self.conversation.chat_stream else "unknown_platform",
            },
            "processed_plain_text": message_content,
            "detailed_plain_text": message_content,
        }
        observation_info.chat_history.append(bot_message_dict)
        observation_info.chat_history_count = len(observation_info.chat_history)
        self.logger.debug(
            f"[私聊][{self.conversation.private_name}] {global_config.BOT_NICKNAME}发送的消息已添加到 chat_history。当前历史数: {observation_info.chat_history_count}"
        )

        max_history_len = getattr(global_config, "pfc_max_chat_history_for_checker", 50)
        if len(observation_info.chat_history) > max_history_len:
            observation_info.chat_history = observation_info.chat_history[-max_history_len:]
            observation_info.chat_history_count = len(observation_info.chat_history)

        history_slice_for_str = observation_info.chat_history[-30:]
        try:
            observation_info.chat_history_str = await build_readable_messages(
                history_slice_for_str,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
        except Exception as e_build_hist:
            self.logger.error(f"[私聊][{self.conversation.private_name}] 更新 chat_history_str 时出错: {e_build_hist}")
            observation_info.chat_history_str = "[构建聊天记录出错]"

    async def _update_post_send_states(
        self,
        send_time: float,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        action_type: str, # "direct_reply", "send_new_message", "say_goodbye", "send_memes"
        event_description_for_emotion: str
    ):
        """处理发送消息成功后的通用状态更新。"""
        if self.conversation.idle_chat:
            await self.conversation.idle_chat.update_last_message_time(send_time)

        # 清理已处理的未读消息 (只清理在发送这条回复之前的、来自他人的消息)
        current_unprocessed_messages = getattr(observation_info, "unprocessed_messages", [])
        message_ids_to_clear: Set[str] = set()
        timestamp_before_sending = send_time - 0.001 # 确保是发送前的时间
        for msg in current_unprocessed_messages:
            msg_time = msg.get("time")
            msg_id = msg.get("message_id")
            sender_id_info = msg.get("user_info", {})
            sender_id = str(sender_id_info.get("user_id")) if sender_id_info else None

            if (
                msg_id
                and msg_time
                and sender_id != self.conversation.bot_qq_str
                and msg_time < timestamp_before_sending
            ):
                message_ids_to_clear.add(msg_id)

        if message_ids_to_clear:
            self.logger.debug(
                f"[私聊][{self.conversation.private_name}] 准备清理 {len(message_ids_to_clear)} 条发送前(他人)消息: {message_ids_to_clear}"
            )
            await observation_info.clear_processed_messages(message_ids_to_clear)
        else:
            self.logger.debug(f"[私聊][{self.conversation.private_name}] 没有需要清理的发送前(他人)消息。")

        # 更新追问状态
        other_new_msg_count_during_planning = getattr(
            conversation_info, "other_new_messages_during_planning_count", 0
        )
        if action_type in ["direct_reply", "send_new_message", "send_memes"]:
            if other_new_msg_count_during_planning > 0 and action_type == "direct_reply":
                self.logger.debug(
                    f"[私聊][{self.conversation.private_name}] 因规划期间收到 {other_new_msg_count_during_planning} 条他人新消息，下一轮强制使用【初始回复】逻辑。"
                )
                conversation_info.last_successful_reply_action = None
            else:
                self.logger.debug(
                    f"[私聊][{self.conversation.private_name}] 成功执行 '{action_type}', 下一轮【允许】使用追问逻辑。"
                )
                conversation_info.last_successful_reply_action = action_type

        # 更新实例消息计数和关系/情绪
        conversation_info.current_instance_message_count += 1
        self.logger.debug(
            f"[私聊][{self.conversation.private_name}] 实例消息计数({global_config.BOT_NICKNAME}发送后)增加到: {conversation_info.current_instance_message_count}"
        )
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_description_for_emotion)

    async def _update_relationship_and_emotion(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        event_description: str
    ):
        """辅助方法：更新关系和情绪状态。"""
        if self.conversation.relationship_updater and self.conversation.chat_observer:
            await self.conversation.relationship_updater.update_relationship_incremental(
                conversation_info=conversation_info,
                observation_info=observation_info,
                chat_observer_for_history=self.conversation.chat_observer,
            )
        if self.conversation.emotion_updater and self.conversation.chat_observer:
            await self.conversation.emotion_updater.update_emotion_based_on_context(
                conversation_info=conversation_info,
                observation_info=observation_info,
                chat_observer_for_history=self.conversation.chat_observer,
                event_description=event_description,
            )


class BaseTextReplyHandler(ActionHandler):
    """
    处理基于文本的回复动作的基类，包含生成-检查-重试的循环。
    适用于 DirectReplyHandler 和 SendNewMessageHandler。
    """
    async def _generate_and_check_text_reply_loop(
        self,
        action_type: str, # "direct_reply" or "send_new_message"
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        max_attempts: int
    ) -> Tuple[bool, Optional[str], str, bool, bool]:
        """
        管理生成文本回复并检查其适用性的循环。
        对于 send_new_message，它还处理来自 ReplyGenerator 的初始 JSON 决策。

        返回:
            is_suitable (bool): 是否找到了合适的回复或作出了发送决策。
            generated_content (Optional[str]): 要发送的内容；如果 ReplyGenerator 决定不发送 (send_new_message)，则为 None。
            check_reason (str): 检查器或生成失败的原因。
            need_replan_from_checker (bool): 如果检查器要求重新规划。
            should_send_reply_for_new_message (bool): 特定于 send_new_message，如果 ReplyGenerator 决定发送则为 True。
        """
        reply_attempt_count = 0
        is_suitable = False
        generated_content_to_send: Optional[str] = None
        final_check_reason = "未开始检查"
        need_replan = False
        # should_send_reply_for_new_message 仅用于 send_new_message 动作类型
        should_send_reply_for_new_message = True if action_type == "direct_reply" else False # direct_reply 总是尝试发送

        while reply_attempt_count < max_attempts and not is_suitable and not need_replan:
            reply_attempt_count += 1
            log_prefix = f"[私聊][{self.conversation.private_name}] 尝试生成/检查 '{action_type}' (第 {reply_attempt_count}/{max_attempts} 次)..."
            self.logger.info(log_prefix)

            self.conversation.state = ConversationState.GENERATING
            if not self.conversation.reply_generator:
                # 这个应该在 Conversation 初始化时就保证了，但以防万一
                raise RuntimeError(f"ReplyGenerator 未为 {self.conversation.private_name} 初始化")

            raw_llm_output = await self.conversation.reply_generator.generate(
                observation_info, conversation_info, action_type=action_type
            )
            self.logger.debug(f"{log_prefix} ReplyGenerator.generate 返回: '{raw_llm_output}'")

            current_content_for_check = raw_llm_output

            if action_type == "send_new_message":
                parsed_json = None
                try:
                    parsed_json = json.loads(raw_llm_output)
                except json.JSONDecodeError:
                    self.logger.error(f"{log_prefix} ReplyGenerator 返回的不是有效的JSON: {raw_llm_output}")
                    conversation_info.last_reply_rejection_reason = "回复生成器未返回有效JSON"
                    conversation_info.last_rejected_reply_content = raw_llm_output
                    should_send_reply_for_new_message = False # 标记不发送
                    is_suitable = True # 决策已做出（不发送），所以认为是 "suitable" 以跳出循环
                    final_check_reason = "回复生成器JSON解析失败，决定不发送"
                    generated_content_to_send = None # 明确不发送内容
                    break # 跳出重试循环

                if parsed_json:
                    send_decision = parsed_json.get("send", "no").lower()
                    generated_text_from_json = parsed_json.get("txt", "") # 如果不发送，txt可能是"no"

                    if send_decision == "yes":
                        should_send_reply_for_new_message = True
                        current_content_for_check = generated_text_from_json
                        self.logger.info(f"{log_prefix} ReplyGenerator 决定发送消息。内容初步为: '{current_content_for_check[:100]}...'")
                    else: # send_decision is "no"
                        should_send_reply_for_new_message = False
                        is_suitable = True # 决策已做出（不发送）
                        final_check_reason = "回复生成器决定不发送"
                        generated_content_to_send = None
                        self.logger.info(f"{log_prefix} ReplyGenerator 决定不发送消息。")
                        break # 跳出重试循环
            
            # 如果是 direct_reply 或者 send_new_message 且决定要发送，则检查内容
            if not current_content_for_check or current_content_for_check.startswith("抱歉") or current_content_for_check.strip() == "" or (action_type == "send_new_message" and current_content_for_check == "no" and should_send_reply_for_new_message):
                warning_msg = f"{log_prefix} 生成内容无效或为错误提示"
                if action_type == "send_new_message" and current_content_for_check == "no" and should_send_reply_for_new_message:
                     warning_msg += " (ReplyGenerator决定发送但文本为'no')"
                self.logger.warning(warning_msg + "，将进行下一次尝试 (如果适用)。")
                final_check_reason = "生成内容无效"
                conversation_info.last_reply_rejection_reason = final_check_reason
                conversation_info.last_rejected_reply_content = current_content_for_check
                await asyncio.sleep(0.5)
                continue

            # --- 内容检查 ---
            self.conversation.state = ConversationState.CHECKING
            if not self.conversation.reply_checker:
                raise RuntimeError(f"ReplyChecker 未为 {self.conversation.private_name} 初始化")

            current_goal_str = ""
            if conversation_info.goal_list:
                goal_item = conversation_info.goal_list[-1]
                current_goal_str = goal_item.get("goal", "") if isinstance(goal_item, dict) else str(goal_item)

            chat_history_for_check = getattr(observation_info, "chat_history", [])
            chat_history_text_for_check = getattr(observation_info, "chat_history_str", "")
            current_time_value_for_check = observation_info.current_time_str or "获取时间失败"

            if global_config.enable_pfc_reply_checker:
                self.logger.debug(f"{log_prefix} 调用 ReplyChecker 检查 (配置已启用)...")
                is_suitable_check, reason_check, need_replan_check = await self.conversation.reply_checker.check(
                    reply=current_content_for_check, goal=current_goal_str,
                    chat_history=chat_history_for_check, chat_history_text=chat_history_text_for_check,
                    current_time_str=current_time_value_for_check, retry_count=(reply_attempt_count - 1)
                )
                self.logger.info(
                    f"{log_prefix} ReplyChecker 结果: 合适={is_suitable_check}, 原因='{reason_check}', 需重规划={need_replan_check}"
                )
            else:
                is_suitable_check, reason_check, need_replan_check = True, "ReplyChecker 已通过配置关闭", False
                self.logger.debug(f"{log_prefix} [配置关闭] ReplyChecker 已跳过，默认回复为合适。")

            is_suitable = is_suitable_check
            final_check_reason = reason_check
            need_replan = need_replan_check

            if not is_suitable:
                conversation_info.last_reply_rejection_reason = final_check_reason
                conversation_info.last_rejected_reply_content = current_content_for_check
                if final_check_reason == "机器人尝试发送重复消息" and not need_replan:
                    self.logger.warning(f"{log_prefix} 回复因自身重复被拒绝。将重试。")
                elif not need_replan and reply_attempt_count < max_attempts:
                    self.logger.warning(f"{log_prefix} 回复不合适: {final_check_reason}。将重试。")
                else: # 需要重规划或达到最大次数
                    self.logger.warning(f"{log_prefix} 回复不合适且(需要重规划或已达最大次数): {final_check_reason}")
                    break # 结束循环
                await asyncio.sleep(0.5) # 重试前暂停
            else: # is_suitable is True
                generated_content_to_send = current_content_for_check
                conversation_info.last_reply_rejection_reason = None
                conversation_info.last_rejected_reply_content = None
                break # 成功，跳出循环
        
        # 确保即使循环结束，如果 should_send_reply_for_new_message 为 False，则 is_suitable 也为 True（表示决策完成）
        if action_type == "send_new_message" and not should_send_reply_for_new_message:
            is_suitable = True # 决策已完成（不发送）
            generated_content_to_send = None # 确认不发送任何内容

        return is_suitable, generated_content_to_send, final_check_reason, need_replan, should_send_reply_for_new_message


class DirectReplyHandler(BaseTextReplyHandler):
    """处理直接回复动作的处理器。"""
    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            return False, "error", "DirectReply 的 ObservationInfo 或 ConversationInfo 为空"

        action_successful = False
        final_status = "recall"
        final_reason = "直接回复动作未成功执行"
        max_reply_attempts: int = getattr(global_config, "pfc_max_reply_attempts", 3)

        is_suitable, generated_content, check_reason, need_replan, _ = await self._generate_and_check_text_reply_loop(
            action_type="direct_reply",
            observation_info=observation_info,
            conversation_info=conversation_info,
            max_attempts=max_reply_attempts
        )

        if is_suitable and generated_content:
            self.conversation.generated_reply = generated_content
            timestamp_before_sending = time.time()
            self.conversation.state = ConversationState.SENDING
            text_segment = Seg(type="text", data=self.conversation.generated_reply)
            send_success = await self._send_reply_or_segments([text_segment], self.conversation.generated_reply)
            send_end_time = time.time()

            if send_success:
                action_successful = True
                final_status = "done"
                final_reason = "成功发送直接回复"
                await self._update_bot_message_in_history(send_end_time, self.conversation.generated_reply, observation_info)
                event_desc = f"你直接回复了消息: '{self.conversation.generated_reply[:50]}...'"
                await self._update_post_send_states(send_end_time, observation_info, conversation_info, "direct_reply", event_desc)
            else:
                final_status = "recall"; final_reason = "发送直接回复时失败"; action_successful = False
                conversation_info.last_successful_reply_action = None
                conversation_info.my_message_count = 0
        elif need_replan:
            final_status = "recall"; final_reason = f"回复检查要求重新规划: {check_reason}"
            conversation_info.last_successful_reply_action = None
        else: # 达到最大尝试次数或生成内容无效
            final_status = "max_checker_attempts_failed"
            final_reason = f"达到最大回复尝试次数或生成内容无效，检查原因: {check_reason}"
            action_successful = False
            conversation_info.last_successful_reply_action = None
        
        return action_successful, final_status, final_reason


class SendNewMessageHandler(BaseTextReplyHandler):
    """处理发送新消息动作的处理器。"""
    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            return False, "error", "SendNewMessage 的 ObservationInfo 或 ConversationInfo 为空"

        action_successful = False
        final_status = "recall"
        final_reason = "发送新消息动作未成功执行"
        max_reply_attempts: int = getattr(global_config, "pfc_max_reply_attempts", 3)

        is_suitable, generated_content, check_reason, need_replan, should_send = await self._generate_and_check_text_reply_loop(
            action_type="send_new_message",
            observation_info=observation_info,
            conversation_info=conversation_info,
            max_attempts=max_reply_attempts
        )

        if not should_send: # ReplyGenerator 决定不发送
            self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'send_new_message': ReplyGenerator 决定不发送。原因: {check_reason}")
            final_status = "done_no_reply"
            final_reason = check_reason if check_reason else "回复生成器决定不发送消息"
            action_successful = True # 决策本身是成功的
            conversation_info.last_successful_reply_action = None
            conversation_info.my_message_count = 0
        elif is_suitable and generated_content: # 决定发送且内容合适
            self.conversation.generated_reply = generated_content
            timestamp_before_sending = time.time()
            self.conversation.state = ConversationState.SENDING
            text_segment = Seg(type="text", data=self.conversation.generated_reply)
            send_success = await self._send_reply_or_segments([text_segment], self.conversation.generated_reply)
            send_end_time = time.time()

            if send_success:
                action_successful = True
                final_status = "done"
                final_reason = "成功发送新消息"
                await self._update_bot_message_in_history(send_end_time, self.conversation.generated_reply, observation_info)
                event_desc = f"你发送了一条新消息: '{self.conversation.generated_reply[:50]}...'"
                await self._update_post_send_states(send_end_time, observation_info, conversation_info, "send_new_message", event_desc)
            else:
                final_status = "recall"; final_reason = "发送新消息时失败"; action_successful = False
                conversation_info.last_successful_reply_action = None
                conversation_info.my_message_count = 0
        elif need_replan:
            final_status = "recall"; final_reason = f"回复检查要求重新规划: {check_reason}"
            conversation_info.last_successful_reply_action = None
        else: # 达到最大尝试次数或生成内容无效
            final_status = "max_checker_attempts_failed"
            final_reason = f"达到最大回复尝试次数或生成内容无效，检查原因: {check_reason}"
            action_successful = False
            conversation_info.last_successful_reply_action = None
            
        return action_successful, final_status, final_reason


class SayGoodbyeHandler(ActionHandler):
    """处理发送告别语动作的处理器。"""
    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            return False, "error", "SayGoodbye 的 ObservationInfo 或 ConversationInfo 为空"

        action_successful = False
        final_status = "recall"
        final_reason = "告别语动作未成功执行"

        self.conversation.state = ConversationState.GENERATING
        if not self.conversation.reply_generator:
            raise RuntimeError("ReplyGenerator 未初始化")

        generated_content = await self.conversation.reply_generator.generate(
            observation_info, conversation_info, action_type="say_goodbye"
        )
        self.logger.info(
            f"[私聊][{self.conversation.private_name}] 动作 'say_goodbye': 生成内容: '{generated_content[:100]}...'"
        )

        if not generated_content or generated_content.startswith("抱歉"):
            self.logger.warning(
                f"[私聊][{self.conversation.private_name}] 动作 'say_goodbye': 生成内容为空或为错误提示，取消发送。"
            )
            final_reason = "生成告别内容无效"
            final_status = "done"
            self.conversation.should_continue = False
            action_successful = True # 即使不发送，结束对话的决策也算完成
        else:
            self.conversation.generated_reply = generated_content
            self.conversation.state = ConversationState.SENDING
            text_segment = Seg(type="text", data=self.conversation.generated_reply)
            send_success = await self._send_reply_or_segments([text_segment], self.conversation.generated_reply)
            send_end_time = time.time()

            if send_success:
                action_successful = True
                final_status = "done"
                final_reason = "成功发送告别语"
                self.conversation.should_continue = False
                await self._update_bot_message_in_history(send_end_time, self.conversation.generated_reply, observation_info)
                event_desc = f"你发送了告别消息: '{self.conversation.generated_reply[:50]}...'"
                await self._update_post_send_states(send_end_time, observation_info, conversation_info, "say_goodbye", event_desc)
            else:
                final_status = "recall"; final_reason = "发送告别语失败"; action_successful = False
                self.conversation.should_continue = True # 发送失败则不结束

        return action_successful, final_status, final_reason


class SendMemesHandler(ActionHandler):
    """处理发送表情包动作的处理器。"""
    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            return False, "error", "SendMemes 的 ObservationInfo 或 ConversationInfo 为空"

        action_successful = False
        final_status = "recall"
        final_reason_prefix = "发送表情包"
        final_reason = f"{final_reason_prefix}失败：未知原因"
        self.conversation.state = ConversationState.GENERATING

        emoji_query = conversation_info.current_emoji_query
        if not emoji_query:
            final_reason = f"{final_reason_prefix}失败：缺少表情包查询语句"
            conversation_info.last_successful_reply_action = None
            return False, "recall", final_reason

        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'send_memes': 使用查询 '{emoji_query}' 获取表情包...")
        try:
            emoji_result = await emoji_manager.get_emoji_for_text(emoji_query)
            if emoji_result:
                emoji_path, emoji_description = emoji_result
                self.logger.info(f"获取到表情包: {emoji_path}, 描述: {emoji_description}")

                if not self.conversation.chat_stream: raise RuntimeError("ChatStream 未初始化")
                image_b64_content = image_path_to_base64(emoji_path)
                if not image_b64_content: raise ValueError(f"无法转换图片 {emoji_path} 为Base64")

                image_segment = Seg(type="image", data={"file": f"base64://{image_b64_content}"})
                log_content_for_meme = f"[表情: {emoji_description}]"
                send_success = await self._send_reply_or_segments([image_segment], log_content_for_meme)
                send_end_time = time.time()

                if send_success:
                    action_successful = True
                    final_status = "done"
                    final_reason = f"{final_reason_prefix}成功发送 ({emoji_description})"
                    await self._update_bot_message_in_history(send_end_time, log_content_for_meme, observation_info, "bot_meme_")
                    event_desc = f"你发送了一个表情包 ({emoji_description})"
                    await self._update_post_send_states(send_end_time, observation_info, conversation_info, "send_memes", event_desc)
                else:
                    final_status = "recall"; final_reason = f"{final_reason_prefix}失败：发送时出错"
            else:
                final_reason = f"{final_reason_prefix}失败：未找到合适表情包"
                conversation_info.last_successful_reply_action = None
        except Exception as e:
            self.logger.error(f"处理表情包动作时出错: {e}", exc_info=True)
            final_status = "error"; final_reason = f"{final_reason_prefix}失败：处理时出错 ({e})"
            conversation_info.last_successful_reply_action = None

        return action_successful, final_status, final_reason


class RethinkGoalHandler(ActionHandler):
    """处理重新思考目标动作的处理器。"""
    async def execute(self, reason: str, observation_info: Optional[ObservationInfo], conversation_info: Optional[ConversationInfo], action_start_time: float, current_action_record: dict) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info: return False, "error", "RethinkGoal 缺少信息"
        self.conversation.state = ConversationState.RETHINKING
        if not self.conversation.goal_analyzer: raise RuntimeError("GoalAnalyzer 未初始化")
        await self.conversation.goal_analyzer.analyze_goal(conversation_info, observation_info)
        event_desc = "你重新思考了对话目标和方向"
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", "成功重新思考目标"

class ListeningHandler(ActionHandler):
    """处理倾听动作的处理器。"""
    async def execute(self, reason: str, observation_info: Optional[ObservationInfo], conversation_info: Optional[ConversationInfo], action_start_time: float, current_action_record: dict) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info: return False, "error", "Listening 缺少信息"
        self.conversation.state = ConversationState.LISTENING
        if not self.conversation.waiter: raise RuntimeError("Waiter 未初始化")
        await self.conversation.waiter.wait_listening(conversation_info)
        event_desc = "你决定耐心倾听对方的发言"
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", "进入倾听状态"

class EndConversationHandler(ActionHandler):
    """处理结束对话动作的处理器。"""
    async def execute(self, reason: str, observation_info: Optional[ObservationInfo], conversation_info: Optional[ConversationInfo], action_start_time: float, current_action_record: dict) -> tuple[bool, str, str]:
        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'end_conversation': 收到最终结束指令，停止对话...")
        self.conversation.should_continue = False
        return True, "done", "对话结束指令已执行"

class BlockAndIgnoreHandler(ActionHandler):
    """处理屏蔽并忽略动作的处理器。"""
    async def execute(self, reason: str, observation_info: Optional[ObservationInfo], conversation_info: Optional[ConversationInfo], action_start_time: float, current_action_record: dict) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info: return False, "error", "BlockAndIgnore 缺少信息"
        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'block_and_ignore': 不想再理你了...")
        ignore_duration_seconds = 10 * 60
        self.conversation.ignore_until_timestamp = time.time() + ignore_duration_seconds
        self.conversation.state = ConversationState.IGNORED
        event_desc = "当前对话让你感到不适，你决定暂时不再理会对方"
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", f"已屏蔽并忽略对话 {ignore_duration_seconds // 60} 分钟"

class WaitHandler(ActionHandler):
    """处理等待动作的处理器。"""
    async def execute(self, reason: str, observation_info: Optional[ObservationInfo], conversation_info: Optional[ConversationInfo], action_start_time: float, current_action_record: dict) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info: return False, "error", "Wait 缺少信息"
        self.conversation.state = ConversationState.WAITING
        if not self.conversation.waiter: raise RuntimeError("Waiter 未初始化")
        timeout_occurred = await self.conversation.waiter.wait(conversation_info)
        event_desc = "你等待对方回复，但对方长时间没有回应" if timeout_occurred else "你选择等待对方的回复"
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", "等待动作完成"

class UnknownActionHandler(ActionHandler):
    """处理未知动作的处理器。"""
    async def execute(self, reason: str, observation_info: Optional[ObservationInfo], conversation_info: Optional[ConversationInfo], action_start_time: float, current_action_record: dict) -> tuple[bool, str, str]:
        action_name = current_action_record.get("action", "未知")
        self.logger.warning(f"[私聊][{self.conversation.private_name}] 未知的动作类型: {action_name}")
        return False, "recall", f"未知的动作类型: {action_name}"
