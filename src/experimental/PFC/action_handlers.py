from abc import ABC, abstractmethod
import time
import asyncio
import traceback
import json # 保持导入，以防未来某些处理器可能需要处理JSON
import random
from typing import Optional, Set, TYPE_CHECKING, List, Tuple, Dict, Any # 确保导入Any

from src.chat.emoji_system.emoji_manager import emoji_manager
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.utils.chat_message_builder import build_readable_messages
from .pfc_types import ConversationState

# 确保导入路径正确
try:
    from .observation_info import ObservationInfo
    from .conversation_info import ConversationInfo
except ImportError: # Fallback for type hinting if direct import fails
    ObservationInfo = Optional[Any] # type: ignore
    ConversationInfo = Optional[Any] # type: ignore

from src.chat.utils.utils_image import image_path_to_base64
from maim_message import Seg # type: ignore

if TYPE_CHECKING:
    from .conversation import Conversation

logger = get_logger("pfc_action_handlers")


class ActionHandler(ABC):
    """
    处理动作的抽象基类。
    每个具体的动作处理器都应继承此类并实现 execute 方法。
    """

    def __init__(self, conversation: "Conversation"):
        """
        初始化动作处理器。

        Args:
            conversation (Conversation): 当前对话实例。
        """
        self.conversation = conversation
        self.logger = logger # 使用模块级别的logger

    @abstractmethod
    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict,
    ) -> tuple[bool, str, str]:
        """
        执行具体的动作逻辑。

        Args:
            reason (str): 执行此动作的规划原因。
            observation_info (Optional[ObservationInfo]): 当前的观察信息。
            conversation_info (Optional[ConversationInfo]): 当前的对话信息。
            action_start_time (float): 动作开始的时间戳。
            current_action_record (dict): 用于记录此动作执行情况的字典。

        Returns:
            tuple[bool, str, str]: 一个元组，包含:
                - action_successful (bool): 动作是否成功执行。
                - final_status (str): 动作的最终状态 (如 "done", "recall", "error")。
                - final_reason (str): 动作最终状态的原因或描述。
        """
        pass

    async def _send_reply_or_segments(self, segments_data: list[Seg], content_for_log: str) -> bool:
        """
        内部辅助函数，用于将构造好的消息段发送出去。
        (此方法保持不变)
        """
        if not self.conversation.direct_sender:
            self.logger.error(f"[私聊][{self.conversation.private_name}] DirectMessageSender 未初始化，无法发送。")
            return False
        if not self.conversation.chat_stream:
            self.logger.error(f"[私聊][{self.conversation.private_name}] ChatStream 未初始化，无法发送。")
            return False

        try:
            final_segments = Seg(type="seglist", data=segments_data) # 将Seg对象列表包装在type="seglist"的Seg对象中
            await self.conversation.direct_sender.send_message( # 调用实际的发送方法
                chat_stream=self.conversation.chat_stream,
                segments=final_segments,
                reply_to_message=None,  # 私聊通常不引用回复
                content=content_for_log,  # 用于发送器内部的日志记录
            )
            return True
        except Exception as e:
            self.logger.error(f"[私聊][{self.conversation.private_name}] 发送消息时失败: {str(e)}")
            self.logger.error(f"[私聊][{self.conversation.private_name}] {traceback.format_exc()}")
            self.conversation.state = ConversationState.ERROR # 发送失败则标记错误状态
            return False

    async def _update_bot_message_in_history(
        self,
        send_time: float,
        message_content: str,
        observation_info: Optional[ObservationInfo], # 明确接收 ObservationInfo
        message_id_prefix: str = "bot_sent_",
    ):
        """
        在机器人成功发送消息后，将该消息添加到 ObservationInfo 的聊天历史中。
        """
        if not self.conversation.bot_qq_str: # 检查机器人QQ号是否存在
            self.logger.warning(f"[私聊][{self.conversation.private_name}] Bot QQ ID 未知，无法更新机器人消息历史。")
            return
        if not observation_info: # 检查 observation_info 是否有效
            self.logger.warning(f"[私聊][{self.conversation.private_name}] ObservationInfo 未提供，无法更新机器人消息历史。")
            return

        bot_message_dict: Dict[str, Any] = { # 明确字典类型
            "message_id": f"{message_id_prefix}{send_time:.3f}",
            "time": send_time,
            "user_info": {
                "user_id": self.conversation.bot_qq_str,
                "user_nickname": global_config.bot.nickname,
                "platform": self.conversation.chat_stream.platform if self.conversation.chat_stream else "unknown_platform",
            },
            "processed_plain_text": message_content,
            "detailed_plain_text": message_content, # 详细文本也使用相同内容
        }
        observation_info.chat_history.append(bot_message_dict) # 添加到历史记录
        observation_info.chat_history_count = len(observation_info.chat_history) # 更新计数
        self.logger.debug(
            f"[私聊][{self.conversation.private_name}] {global_config.bot.nickname}发送的消息 ('{message_content[:30]}...')已添加到 chat_history。当前历史数: {observation_info.chat_history_count}"
        )

        max_history_len = global_config.pfc.pfc_max_chat_history_for_checker # 从配置获取最大历史长度
        if len(observation_info.chat_history) > max_history_len: # 如果超过最大长度
            observation_info.chat_history = observation_info.chat_history[-max_history_len:] # 截取最新的部分
            observation_info.chat_history_count = len(observation_info.chat_history) # 更新计数

        history_slice_for_str = observation_info.chat_history[-global_config.pfc.pfc_recent_history_display_count:]
        try:
            observation_info.chat_history_str = await build_readable_messages( # 构建可读的聊天历史字符串
                history_slice_for_str,
                replace_bot_name=True, merge_messages=False, timestamp_mode="relative", read_mark=0.0,
            )
        except Exception as e_build_hist:
            self.logger.error(f"[私聊][{self.conversation.private_name}] 更新 chat_history_str 时出错: {e_build_hist}")
            observation_info.chat_history_str = "[构建聊天记录出错]"


    async def _update_post_send_states(
        self,
        observation_info: Optional[ObservationInfo], # 明确类型
        conversation_info: Optional[ConversationInfo], # 明确类型
        action_type: str,
        event_description_for_emotion: str,
    ):
        """
        在成功发送一条或多条消息（文本/表情）后，处理通用的状态更新。
        """
        if not observation_info or not conversation_info: # 防御性检查
            self.logger.warning(f"[私聊][{self.conversation.private_name}] _update_post_send_states: ObservationInfo 或 ConversationInfo 为空，跳过更新。")
            return

        current_event_time = time.time() # 获取当前事件时间

        if self.conversation.idle_chat: # 如果存在主动聊天实例
            await self.conversation.idle_chat.update_last_message_time(current_event_time) # 更新最后消息时间

        # 清理已处理的他人消息
        current_unprocessed_messages = getattr(observation_info, "unprocessed_messages", [])
        message_ids_to_clear: Set[str] = set() # 明确类型
        timestamp_before_current_interaction_completion = current_event_time - 0.001 # 定义交互完成前的时间戳

        for msg in current_unprocessed_messages: # 遍历未处理消息
            msg_time = msg.get("time")
            msg_id = msg.get("message_id")
            sender_id_info = msg.get("user_info", {})
            sender_id = str(sender_id_info.get("user_id")) if sender_id_info else None
            if (msg_id and msg_time and sender_id != self.conversation.bot_qq_str and
                msg_time < timestamp_before_current_interaction_completion): # 如果是他人消息且在本次交互完成前
                message_ids_to_clear.add(msg_id) # 添加到待清理集合

        if message_ids_to_clear: # 如果有需要清理的消息
            self.logger.debug(
                f"[私聊][{self.conversation.private_name}] 准备清理 {len(message_ids_to_clear)} 条交互完成前(他人)消息: {message_ids_to_clear}"
            )
            await observation_info.clear_processed_messages(message_ids_to_clear) # 清理消息

        # 更新追问状态 (last_successful_reply_action)
        other_new_msg_count_during_planning = getattr(conversation_info, "other_new_messages_during_planning_count", 0)
        if action_type in ["direct_reply", "send_new_message", "send_memes", "reply_after_wait_timeout"]: # 如果是回复类动作
            if other_new_msg_count_during_planning > 0 and action_type == "direct_reply": # 如果是直接回复且规划期间有新消息
                conversation_info.last_successful_reply_action = None # 则下次不应追问
            else:
                conversation_info.last_successful_reply_action = action_type # 否则记录本次成功动作为下次追问依据
        
        # 更新关系和情绪状态
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_description_for_emotion)


    async def _update_relationship_and_emotion(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        event_description: str
    ):
        """
        辅助方法：调用关系更新器和情绪更新器。
        """
        # 更新关系值（增量）
        if self.conversation.relationship_updater and self.conversation.chat_observer: # 确保组件存在
            await self.conversation.relationship_updater.update_relationship_incremental(
                conversation_info=conversation_info,
                observation_info=observation_info,
                chat_observer_for_history=self.conversation.chat_observer,
            )
        # 更新情绪状态
        if self.conversation.emotion_updater and self.conversation.chat_observer: # 确保组件存在
            await self.conversation.emotion_updater.update_emotion_based_on_context(
                conversation_info=conversation_info,
                observation_info=observation_info,
                chat_observer_for_history=self.conversation.chat_observer,
                event_description=event_description,
            )

    async def _fetch_and_prepare_emoji_segment(self, emoji_query: str) -> Optional[Tuple[Seg, str, str]]:
        """
        根据表情查询字符串获取表情图片，准备发送。
        (此方法保持不变)
        """
        self.logger.info(f"[私聊][{self.conversation.private_name}] 尝试获取表情，查询: '{emoji_query}'")
        try:
            emoji_result = await emoji_manager.get_emoji_for_text(emoji_query) # 获取表情
            if emoji_result: # 如果成功获取
                emoji_path, full_emoji_description = emoji_result
                self.logger.info(f"获取到表情包: {emoji_path}, 描述: {full_emoji_description}")
                emoji_b64_content = image_path_to_base64(emoji_path) # 转为Base64
                if not emoji_b64_content: # 如果转换失败
                    self.logger.error(f"无法将图片 {emoji_path} 转换为Base64。")
                    return None
                emoji_segment = Seg(type="emoji", data=emoji_b64_content) # 创建Seg对象
                log_content_for_emoji = full_emoji_description[-20:] + "..." if len(full_emoji_description) > 20 else full_emoji_description # 日志截断
                return emoji_segment, full_emoji_description, log_content_for_emoji
            else: # 如果未获取到表情
                self.logger.warning(f"未能根据查询 '{emoji_query}' 获取到合适的表情包。")
                return None
        except Exception as e:
            self.logger.error(f"获取或准备表情图片时出错: {e}", exc_info=True)
            return None


class BaseTextReplyHandler(ActionHandler):
    """
    处理基于文本的回复动作的基类，包含生成-检查-重试的循环。
    适用于 DirectReplyHandler 和 SendNewMessageHandler。
    """
    async def _generate_and_check_text_reply_loop(
        self,
        action_type: str,
        observation_info: Optional[ObservationInfo], # 明确类型
        conversation_info: Optional[ConversationInfo], # 明确类型
        max_attempts: int,
    ) -> Tuple[bool, Optional[str], str, bool]: # 返回值减少一个布尔型
        """
        管理生成文本回复并检查其适用性的循环。
        ReplyGenerator 现在直接返回文本。
        """
        reply_attempt_count = 0 # 初始化尝试次数
        is_suitable = False # 标记回复是否合适
        generated_content_to_send: Optional[str] = None # 最终要发送的文本
        final_check_reason = "未开始检查" # 默认检查原因
        need_replan = False # 是否需要重新规划

        # 确保 observation_info 和 conversation_info 有效
        if not observation_info or not conversation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] _generate_and_check_text_reply_loop: ObservationInfo 或 ConversationInfo 为空。")
            return False, None, "内部信息缺失，无法生成或检查回复", True # 通常这种错误需要重新规划

        while reply_attempt_count < max_attempts and not is_suitable and not need_replan:
            reply_attempt_count += 1 # 增加尝试次数
            log_prefix = f"[私聊][{self.conversation.private_name}] 尝试生成/检查 '{action_type}' (第 {reply_attempt_count}/{max_attempts} 次)..."
            self.logger.info(log_prefix)

            self.conversation.state = ConversationState.GENERATING # 设置状态为生成中
            if not self.conversation.reply_generator: # 检查ReplyGenerator是否存在
                self.logger.error(f"ReplyGenerator 未为 {self.conversation.private_name} 初始化!")
                final_check_reason = "ReplyGenerator未初始化"
                need_replan = True # 严重错误，需要重新规划
                break # 无法继续

            # 调用 ReplyGenerator.generate()，它现在接收 observation_info, conversation_info, action_type
            current_content_for_check = await self.conversation.reply_generator.generate(
                observation_info=observation_info,
                conversation_info=conversation_info,
                action_type=action_type
            )
            self.logger.debug(f"{log_prefix} ReplyGenerator.generate 返回: '{current_content_for_check}'")

            # 检查生成的内容是否有效
            if (not current_content_for_check or
                current_content_for_check.startswith("抱歉") or # 通常是ReplyGenerator内部出错的标志
                not current_content_for_check.strip() or # 检查是否为空或只有空白
                current_content_for_check.strip().lower() in ["嗯...", "嗯."]): # 这些是ReplyGenerator的默认无效输出
                warning_msg = f"{log_prefix} 生成内容无效或为通用错误/默认提示: '{current_content_for_check}'"
                self.logger.warning(warning_msg + "，将进行下一次尝试 (如果适用)。")
                final_check_reason = "生成内容无效或为通用错误提示"
                conversation_info.last_reply_rejection_reason = final_check_reason # 记录拒绝原因
                conversation_info.last_rejected_reply_content = current_content_for_check # 记录被拒内容
                await asyncio.sleep(0.5) # 短暂休眠后重试
                continue # 进入下一次循环

            # --- 内容检查 ---
            self.conversation.state = ConversationState.CHECKING # 设置状态为检查中
            if not self.conversation.reply_checker: # 检查ReplyChecker是否存在
                self.logger.error(f"ReplyChecker 未为 {self.conversation.private_name} 初始化!")
                final_check_reason = "ReplyChecker未初始化"
                need_replan = True # 严重错误
                break # 无法继续

            current_goal_str = "" # 初始化当前目标字符串
            if conversation_info.goal_list: # 如果目标列表存在
                goal_item = conversation_info.goal_list[-1] # 获取最后一个目标
                current_goal_str = goal_item.get("goal", "") if isinstance(goal_item, dict) else str(goal_item)

            chat_history_for_check = getattr(observation_info, "chat_history", [])
            chat_history_text_for_check = getattr(observation_info, "chat_history_str", "")
            current_time_value_for_check = observation_info.current_time_str or "获取时间失败"

            is_suitable_check, reason_check, need_replan_check = True, "ReplyChecker已通过配置关闭", False
            if global_config.pfc.enable_pfc_reply_checker: # 如果启用了ReplyChecker
                self.logger.debug(f"{log_prefix} 调用 ReplyChecker 检查 (配置已启用)...")
                is_suitable_check, reason_check, need_replan_check = await self.conversation.reply_checker.check(
                    reply=current_content_for_check,
                    goal=current_goal_str,
                    chat_history=chat_history_for_check,
                    chat_history_text=chat_history_text_for_check,
                    current_time_str=current_time_value_for_check,
                    retry_count=(reply_attempt_count - 1),
                )
                self.logger.info(
                    f"{log_prefix} ReplyChecker 结果: 合适={is_suitable_check}, 原因='{reason_check}', 需重规划={need_replan_check}"
                )
            else: # 如果未启用ReplyChecker
                self.logger.debug(f"{log_prefix} [配置关闭] ReplyChecker 已跳过，默认回复为合适。")

            is_suitable = is_suitable_check # 更新内容是否合适
            final_check_reason = reason_check # 更新检查原因
            need_replan = need_replan_check # 更新是否需要重规划

            if not is_suitable: # 如果内容不合适
                conversation_info.last_reply_rejection_reason = final_check_reason # 记录拒绝原因
                conversation_info.last_rejected_reply_content = current_content_for_check # 记录被拒内容
                if final_check_reason == "机器人尝试发送与历史发言相似的消息 (内容规范化后相同)" and not need_replan:
                    self.logger.warning(f"{log_prefix} 回复因自身重复被拒绝。将重试。")
                elif not need_replan and reply_attempt_count < max_attempts: # 如果不需要重规划且还有尝试次数
                    self.logger.warning(f"{log_prefix} 回复不合适: {final_check_reason}。将重试。")
                else: # 需要重规划或已达到最大尝试次数
                    self.logger.warning(f"{log_prefix} 回复不合适且(需要重规划或已达最大次数): {final_check_reason}")
                    break # 结束循环
                await asyncio.sleep(0.5) # 重试前暂停
            else: # 内容合适
                generated_content_to_send = current_content_for_check # 设置最终要发送的内容
                conversation_info.last_reply_rejection_reason = None # 清除上次拒绝原因
                conversation_info.last_rejected_reply_content = None # 清除上次拒绝内容
                break # 成功，跳出循环
        
        return is_suitable, generated_content_to_send, final_check_reason, need_replan


    async def _process_and_send_reply_with_optional_emoji(
        self,
        action_type: str,
        observation_info: Optional[ObservationInfo], # 明确类型
        conversation_info: Optional[ConversationInfo], # 明确类型
        max_reply_attempts: int,
    ) -> Tuple[bool, bool, List[str], Optional[str], bool, str]: # 返回值减少一个布尔型
        """
        核心共享方法：处理文本生成/检查，获取表情，并按顺序发送。
        """
        sent_text_successfully = False # 标记文本是否成功发送
        sent_emoji_successfully = False # 标记表情是否成功发送
        final_reason_parts: List[str] = [] # 存储最终原因的各个部分
        full_emoji_description_if_sent: Optional[str] = None # 如果表情发送成功，其完整描述

        if not observation_info or not conversation_info: # 再次检查参数有效性
            self.logger.error(f"[私聊][{self.conversation.private_name}] _process_and_send_reply_with_optional_emoji: ObservationInfo 或 ConversationInfo 为空。")
            return False, False, ["内部信息缺失"], None, True, "内部信息缺失" # 返回错误状态

        # 1. 处理文本部分
        (
            is_suitable_text,
            generated_text_content,
            text_check_reason,
            need_replan_text,
        ) = await self._generate_and_check_text_reply_loop(
            action_type=action_type,
            observation_info=observation_info,
            conversation_info=conversation_info,
            max_attempts=max_reply_attempts,
        )

        text_to_send: Optional[str] = None # 初始化要发送的文本
        if is_suitable_text and generated_text_content: # 如果文本合适且有内容
            text_to_send = generated_text_content # 准备发送
        
        # 2. 处理表情部分
        emoji_prepared_info: Optional[Tuple[Seg, str, str]] = None # 初始化表情准备信息
        emoji_query = conversation_info.current_emoji_query # 获取当前表情查询
        if emoji_query: # 如果有表情查询
            emoji_prepared_info = await self._fetch_and_prepare_emoji_segment(emoji_query) # 获取并准备表情
            conversation_info.current_emoji_query = None # 清理查询，避免重复使用

        # 3. 决定发送顺序并发送
        send_order: List[str] = [] # 初始化发送顺序列表
        if text_to_send and emoji_prepared_info: # 如果文本和表情都有
            send_order = ["text", "emoji"] if random.random() < 0.5 else ["emoji", "text"] # 随机顺序
        elif text_to_send: # 只有文本
            send_order = ["text"]
        elif emoji_prepared_info: # 只有表情
            send_order = ["emoji"]

        for item_type in send_order: # 按顺序发送
            current_send_time = time.time() # 获取当前发送时间
            if item_type == "text" and text_to_send: # 如果是发送文本且有文本内容
                self.conversation.generated_reply = text_to_send # 记录生成的回复
                text_segment = Seg(type="text", data=text_to_send) # 创建文本Seg对象
                if await self._send_reply_or_segments([text_segment], text_to_send): # 发送
                    sent_text_successfully = True # 标记文本发送成功
                    await self._update_bot_message_in_history(current_send_time, text_to_send, observation_info) # 更新历史记录
                    if self.conversation.conversation_info: # 更新计数
                        self.conversation.conversation_info.current_instance_message_count += 1
                        self.conversation.conversation_info.my_message_count += 1
                    final_reason_parts.append(f"成功发送文本 ('{text_to_send[:20]}...')") # 添加成功原因
                else: # 如果发送失败
                    final_reason_parts.append("发送文本失败")
                    break # 中断发送流程
            elif item_type == "emoji" and emoji_prepared_info: # 如果是发送表情且已准备好表情
                emoji_segment, full_emoji_desc, log_emoji_desc = emoji_prepared_info
                if await self._send_reply_or_segments([emoji_segment], log_emoji_desc): # 发送
                    sent_emoji_successfully = True # 标记表情发送成功
                    full_emoji_description_if_sent = full_emoji_desc # 记录完整描述
                    await self._update_bot_message_in_history(current_send_time, full_emoji_desc, observation_info, "bot_emoji_") # 更新历史记录
                    if self.conversation.conversation_info: # 更新计数
                        self.conversation.conversation_info.current_instance_message_count += 1
                        self.conversation.conversation_info.my_message_count += 1
                    final_reason_parts.append(f"成功发送表情 ({full_emoji_desc})") # 添加成功原因
                else: # 如果发送失败
                    final_reason_parts.append("发送表情失败")
                    if not text_to_send: break # 如果只有表情且表情发送失败，则中断

        return (
            sent_text_successfully,
            sent_emoji_successfully,
            final_reason_parts,
            full_emoji_description_if_sent,
            need_replan_text,
            text_check_reason if not is_suitable_text else "文本检查通过或未执行",
        )


class DirectReplyHandler(BaseTextReplyHandler):
    """处理直接回复动作（direct_reply）的处理器。"""

    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict,
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] DirectReplyHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行直接回复"

        action_successful = False # 初始化动作成功标记
        final_status = "recall" # 默认最终状态
        final_reason = "直接回复动作未成功执行" # 默认最终原因
        if hasattr(global_config.pfc, 'pfc_max_reply_attempts'):
            max_reply_attempts: int = global_config.pfc.pfc_max_reply_attempts
        else:
            max_reply_attempts: int = 3 # 默认值
            self.logger.warning( # 假设在类方法中，所以用 self.logger
                f"[私聊][{self.conversation.private_name}] 配置项 'pfc_max_reply_attempts' 未在 global_config.pfc 中找到，使用默认值: {max_reply_attempts}"
    )

        (
            sent_text_successfully,
            sent_emoji_successfully,
            reason_parts,
            full_emoji_desc_if_sent, # 修改变量名以接收
            need_replan_from_text_check,
            text_check_failure_reason,
        ) = await self._process_and_send_reply_with_optional_emoji(
            action_type="direct_reply",
            observation_info=observation_info,
            conversation_info=conversation_info,
            max_reply_attempts=max_reply_attempts,
        )

        if sent_text_successfully or sent_emoji_successfully: # 如果文本或表情任一发送成功
            action_successful = True # 标记动作成功
            final_status = "done" # 设置状态为完成
            final_reason = "; ".join(reason_parts) if reason_parts else "成功完成操作" # 组合原因

            event_desc_parts = [] # 初始化事件描述部分
            if sent_text_successfully and self.conversation.generated_reply: # 如果文本发送成功
                event_desc_parts.append(f"你回复了: '{self.conversation.generated_reply[:30]}...'")
            if sent_emoji_successfully and full_emoji_desc_if_sent: # 如果表情发送成功
                event_desc_parts.append(f"并发送了表情: '{full_emoji_desc_if_sent}'")
            event_desc = " ".join(event_desc_parts) if event_desc_parts else "机器人发送了消息" # 组合事件描述
            await self._update_post_send_states(observation_info, conversation_info, "direct_reply", event_desc) # 更新发送后状态
        elif need_replan_from_text_check: # 如果文本检查要求重新规划
            final_status = "recall" # 设置状态为需要重新调用（规划）
            final_reason = f"文本回复检查要求重新规划: {text_check_failure_reason}"
            conversation_info.last_successful_reply_action = None # 清除上次成功回复动作
        else: # 其他失败情况（例如达到最大尝试次数）
            final_status = "max_checker_attempts_failed" if not need_replan_from_text_check else "recall"
            final_reason = f"直接回复失败。文本检查: {text_check_failure_reason}. " + ("; ".join(reason_parts) if reason_parts else "")
            action_successful = False # 标记动作失败
            conversation_info.last_successful_reply_action = None # 清除上次成功回复动作
        
        return action_successful, final_status, final_reason.strip()


class SendNewMessageHandler(BaseTextReplyHandler):
    """处理发送新消息动作（send_new_message）的处理器。"""

    async def execute(
        self,
        reason: str,
        observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo],
        action_start_time: float,
        current_action_record: dict,
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] SendNewMessageHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行发送新消息"

        action_successful = False # 初始化动作成功标记
        final_status = "recall" # 默认最终状态
        final_reason = "发送新消息动作未成功执行" # 默认最终原因
        config_item_name = 'pfc_max_reply_attempts'
        default_value = 3
        max_reply_attempts: int = getattr(global_config.pfc, config_item_name, default_value)


        (
            sent_text_successfully,
            sent_emoji_successfully,
            reason_parts,
            full_emoji_desc_if_sent, # 修改变量名
            need_replan_from_text_check,
            text_check_failure_reason,
        ) = await self._process_and_send_reply_with_optional_emoji(
            action_type="send_new_message",
            observation_info=observation_info,
            conversation_info=conversation_info,
            max_reply_attempts=max_reply_attempts,
        )

        # ReplyGenerator不再决定是否发送，ActionPlanner决定了send_new_message就意味着要尝试发送
        # 所以不再需要 rg_decided_not_to_send_text 相关的逻辑分支
        if sent_text_successfully or sent_emoji_successfully: # 如果文本或表情任一发送成功
            action_successful = True # 标记动作成功
            final_status = "done" # 设置状态为完成
            final_reason = "; ".join(reason_parts) if reason_parts else "成功完成操作" # 组合原因

            event_desc_parts = [] # 初始化事件描述部分
            if sent_text_successfully and self.conversation.generated_reply: # 如果文本发送成功
                event_desc_parts.append(f"你发送了新消息: '{self.conversation.generated_reply[:30]}...'")
            if sent_emoji_successfully and full_emoji_desc_if_sent: # 如果表情发送成功
                event_desc_parts.append(f"并发送了表情: '{full_emoji_desc_if_sent}'")
            event_desc = " ".join(event_desc_parts) if event_desc_parts else "机器人发送了消息" # 组合事件描述
            await self._update_post_send_states(observation_info, conversation_info, "send_new_message", event_desc) # 更新发送后状态
        elif need_replan_from_text_check: # 如果文本检查要求重新规划
            final_status = "recall" # 设置状态为需要重新调用（规划）
            final_reason = f"发送新消息的文本检查要求重新规划: {text_check_failure_reason}"
            conversation_info.last_successful_reply_action = None # 清除上次成功回复动作
        else: # 文本和表情都未能发送，或者文本检查失败且不需重规划
            final_status = "max_checker_attempts_failed" if not need_replan_from_text_check else "recall"
            final_reason = f"发送新消息失败。文本检查: {text_check_failure_reason}. " + ("; ".join(reason_parts) if reason_parts else "")
            action_successful = False # 标记动作失败
            conversation_info.last_successful_reply_action = None # 清除上次成功回复动作
        
        # 如果动作最终没有发出任何实质内容，确保 last_successful_reply_action 被清除
        if final_status != "done" and conversation_info:
            conversation_info.last_successful_reply_action = None

        return action_successful, final_status, final_reason.strip()


class SayGoodbyeHandler(ActionHandler):
    """处理发送告别语动作（say_goodbye）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] SayGoodbyeHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行告别"

        action_successful = False # 初始化动作成功标记
        final_status = "recall" # 默认最终状态
        final_reason = "告别语动作未成功执行" # 默认最终原因

        self.conversation.state = ConversationState.GENERATING # 设置状态为生成中
        if not self.conversation.reply_generator: # 检查ReplyGenerator是否存在
            self.logger.error(f"ReplyGenerator 未为 {self.conversation.private_name} 初始化!")
            return False, "error", "ReplyGenerator未初始化"

        # 调用 ReplyGenerator.generate() 获取告别语文本
        generated_content = await self.conversation.reply_generator.generate(
            observation_info=observation_info,
            conversation_info=conversation_info,
            action_type="say_goodbye"
        )
        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'say_goodbye': 生成内容: '{generated_content[:100]}...'")

        if not generated_content or generated_content.startswith("抱歉") or not generated_content.strip() or generated_content.strip().lower() in ["嗯...", "嗯."]:
            self.logger.warning(f"[私聊][{self.conversation.private_name}] 动作 'say_goodbye': 生成内容无效，取消发送。")
            final_reason = "生成告别内容无效，对话直接结束"
            final_status = "done" # 即使不发送，结束对话的决策也算完成
            self.conversation.should_continue = False # 标记对话结束
            action_successful = True # 动作（决策结束）本身算成功
        else: # 如果生成内容有效
            self.conversation.generated_reply = generated_content # 记录生成的回复
            self.conversation.state = ConversationState.SENDING # 设置状态为发送中
            text_segment = Seg(type="text", data=self.conversation.generated_reply) # 创建文本Seg对象
            send_success = await self._send_reply_or_segments([text_segment], self.conversation.generated_reply) # 发送
            send_end_time = time.time() # 记录发送结束时间

            if send_success: # 如果发送成功
                action_successful = True # 标记动作成功
                final_status = "done" # 设置状态为完成
                final_reason = "成功发送告别语" # 设置最终原因
                self.conversation.should_continue = False # 标记对话结束
                if self.conversation.conversation_info: # 更新计数
                    self.conversation.conversation_info.current_instance_message_count += 1
                    self.conversation.conversation_info.my_message_count += 1
                await self._update_bot_message_in_history(send_end_time, self.conversation.generated_reply, observation_info) # 更新历史记录
                event_desc = f"你发送了告别消息: '{self.conversation.generated_reply[:50]}...'" # 事件描述
                await self._update_post_send_states(observation_info, conversation_info, "say_goodbye", event_desc) # 更新发送后状态
            else: # 如果发送失败
                final_status = "recall" # 设置状态为需要重新调用（规划）
                final_reason = "发送告别语失败，对话保持" # 设置最终原因
                action_successful = False # 标记动作失败

        return action_successful, final_status, final_reason


class SendMemesHandler(ActionHandler): # (此Handler逻辑基本不变)
    """处理发送表情包动作（send_memes）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] SendMemesHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法发送表情包"

        action_successful = False
        final_status = "recall"
        final_reason_prefix = "发送表情包"
        final_reason = f"{final_reason_prefix}失败：未知原因"
        self.conversation.state = ConversationState.GENERATING # 或 SENDING_MEME

        emoji_query = conversation_info.current_emoji_query
        if not emoji_query:
            final_reason = f"{final_reason_prefix}失败：缺少表情包查询语句"
            return False, "recall", final_reason
        
        conversation_info.current_emoji_query = None # 清理

        emoji_prepared_info = await self._fetch_and_prepare_emoji_segment(emoji_query)

        if emoji_prepared_info:
            emoji_segment, full_emoji_description, log_emoji_description = emoji_prepared_info
            send_success = await self._send_reply_or_segments([emoji_segment], log_emoji_description)
            send_end_time = time.time()

            if send_success:
                action_successful = True
                final_status = "done"
                final_reason = f"{final_reason_prefix}成功发送 ({full_emoji_description})"
                if self.conversation.conversation_info:
                    self.conversation.conversation_info.current_instance_message_count += 1
                    self.conversation.conversation_info.my_message_count += 1
                await self._update_bot_message_in_history(send_end_time, full_emoji_description, observation_info, "bot_meme_")
                event_desc = f"你发送了一个表情包 ({full_emoji_description})"
                await self._update_post_send_states(observation_info, conversation_info, "send_memes", event_desc)
            else:
                final_status = "recall"
                final_reason = f"{final_reason_prefix}失败：发送时出错"
        else:
            final_reason = f"{final_reason_prefix}失败：未找到或准备表情失败 ({emoji_query})"

        return action_successful, final_status, final_reason


class RethinkGoalHandler(ActionHandler): # (保持不变)
    """处理重新思考目标动作（rethink_goal）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] RethinkGoalHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法重新思考目标"
        self.conversation.state = ConversationState.RETHINKING
        if not self.conversation.goal_analyzer:
            self.logger.error(f"GoalAnalyzer 未为 {self.conversation.private_name} 初始化!")
            return False, "error", "GoalAnalyzer未初始化"
        await self.conversation.goal_analyzer.analyze_goal(conversation_info, observation_info)
        event_desc = "你重新思考了对话目标和方向"
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", "成功重新思考目标"


class ListeningHandler(ActionHandler): # (保持不变)
    """处理倾听动作（listening）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] ListeningHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行倾听"
        self.conversation.state = ConversationState.LISTENING

        if not self.conversation.waiter:
            self.logger.error(f"Waiter 未为 {self.conversation.private_name} 初始化!")
            return False, "error", "Waiter未初始化"
        
        if conversation_info:
            conversation_info.wait_has_timed_out = False
            conversation_info.last_wait_duration_minutes = None

        timeout_occurred, wait_duration_minutes = await self.conversation.waiter.wait_listening(conversation_info)
        
        event_desc = f"你耐心倾听对方的发言"
        if timeout_occurred and conversation_info:
            conversation_info.wait_has_timed_out = True
            conversation_info.last_wait_duration_minutes = wait_duration_minutes
            event_desc = f"你耐心倾听，但对方长时间（约{wait_duration_minutes:.1f}分钟）没有继续发言"
            self.logger.info(f"[私聊][{self.conversation.private_name}] 倾听等待超时，时长: {wait_duration_minutes:.1f} 分钟。")

        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", "进入倾听状态"


class EndConversationHandler(ActionHandler): # (保持不变)
    """处理结束对话动作（end_conversation）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'end_conversation': 收到最终结束指令，停止对话...")
        self.conversation.should_continue = False
        return True, "done", "对话结束指令已执行"


class BlockAndIgnoreHandler(ActionHandler): # (保持不变)
    """处理屏蔽并忽略对话动作（block_and_ignore）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] BlockAndIgnoreHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行屏蔽"
        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'block_and_ignore': 不想再理你了...")
        if hasattr(global_config.pfc, 'pfc_block_ignore_duration_seconds'):
            ignore_duration_seconds: int = global_config.pfc.pfc_block_ignore_duration_seconds
        else:
            ignore_duration_seconds: int = 600 # 默认值
            self.logger.warning(
                f"[私聊][{self.conversation.private_name}] 配置项 'pfc_block_ignore_duration_seconds' 未在 global_config.pfc 中找到，使用默认值: {ignore_duration_seconds}"
    )
        self.conversation.ignore_until_timestamp = time.time() + ignore_duration_seconds
        self.conversation.state = ConversationState.IGNORED
        event_desc = "当前对话让你感到不适，你决定暂时不再理会对方"
        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", f"已屏蔽并忽略对话 {ignore_duration_seconds // 600} 分钟"


class WaitHandler(ActionHandler): # (保持不变)
    """处理等待动作（wait）的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        if not conversation_info or not observation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] WaitHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行等待"
        self.conversation.state = ConversationState.WAITING

        if not self.conversation.waiter:
            self.logger.error(f"Waiter 未为 {self.conversation.private_name} 初始化!")
            return False, "error", "Waiter未初始化"

        if conversation_info:
            conversation_info.wait_has_timed_out = False
            conversation_info.last_wait_duration_minutes = None

        timeout_occurred, wait_duration_minutes = await self.conversation.waiter.wait(conversation_info)
        
        event_desc = "你选择等待对方的回复"
        if timeout_occurred and conversation_info:
            conversation_info.wait_has_timed_out = True
            conversation_info.last_wait_duration_minutes = wait_duration_minutes
            event_desc = f"你等待对方回复，但对方长时间（约{wait_duration_minutes:.1f}分钟）没有回应"
            self.logger.info(f"[私聊][{self.conversation.private_name}] 等待超时，时长: {wait_duration_minutes:.1f} 分钟。")

        await self._update_relationship_and_emotion(observation_info, conversation_info, event_desc)
        return True, "done", "等待动作完成"


class ReplyAfterWaitTimeoutHandler(ActionHandler): # MODIFIED
    """处理在等待超时后进行回复的动作"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict,
    ) -> tuple[bool, str, str]:
        if not observation_info or not conversation_info:
            self.logger.error(f"[私聊][{self.conversation.private_name}] ReplyAfterWaitTimeoutHandler: ObservationInfo 或 ConversationInfo 为空。")
            return False, "error", "内部信息缺失，无法执行等待后回复"

        action_successful = False # 初始化动作成功标记
        final_status = "recall" # 默认最终状态
        final_reason = "等待超时后回复动作未成功执行" # 默认最终原因

        self.conversation.state = ConversationState.GENERATING # 设置状态为生成中
        if not self.conversation.reply_generator: # 检查ReplyGenerator是否存在
            self.logger.error(f"ReplyGenerator 未为 {self.conversation.private_name} 初始化!")
            return False, "error", "ReplyGenerator未初始化"

        # 调用 ReplyGenerator.generate() 获取回复文本
        generated_content = await self.conversation.reply_generator.generate(
            observation_info=observation_info,
            conversation_info=conversation_info,
            action_type="reply_after_wait_timeout" # 明确动作类型
        )
        self.logger.info(f"[私聊][{self.conversation.private_name}] 动作 'reply_after_wait_timeout': 生成内容: '{generated_content[:100]}...'")

        if not generated_content or generated_content.startswith("抱歉") or not generated_content.strip() or generated_content.strip().lower() in ["嗯...", "嗯."]:
            self.logger.warning(f"[私聊][{self.conversation.private_name}] 'reply_after_wait_timeout': 生成内容无效或为空。")
            final_reason = "等待超时后回复内容生成无效"
            # 即使生成无效，也认为“尝试回复”这个动作完成了，但没有发出东西，状态应为recall以便重新规划
            return False, "recall", final_reason

        # 内容检查 (如果启用)
        self.conversation.state = ConversationState.CHECKING # 设置状态为检查中
        if not self.conversation.reply_checker: # 检查ReplyChecker是否存在
            self.logger.error(f"ReplyChecker 未为 {self.conversation.private_name} 初始化!")
            return False, "error", "ReplyChecker未初始化"

        is_suitable, check_reason, need_replan_from_check = True, "ReplyChecker已通过配置关闭", False # 默认值
        if global_config.pfc.enable_pfc_reply_checker: # 如果启用了ReplyChecker
            current_goal_str = "" # 初始化当前目标字符串
            if conversation_info.goal_list: # 如果目标列表存在
                goal_item = conversation_info.goal_list[-1] # 获取最后一个目标
                current_goal_str = goal_item.get("goal", "") if isinstance(goal_item, dict) else str(goal_item)
            is_suitable, check_reason, need_replan_from_check = await self.conversation.reply_checker.check(
                reply=generated_content, goal=current_goal_str,
                chat_history=getattr(observation_info, "chat_history", []),
                chat_history_text=getattr(observation_info, "chat_history_str", ""),
                current_time_str=observation_info.current_time_str or "获取时间失败",
                retry_count=0 # 对于超时回复，通常只检查一次
            )
            self.logger.info(f"[私聊][{self.conversation.private_name}] 'reply_after_wait_timeout' Checker结果: 合适={is_suitable}, 原因='{check_reason}', 重规划={need_replan_from_check}")
        
        if not is_suitable: # 如果检查不通过
            conversation_info.last_reply_rejection_reason = check_reason # 记录拒绝原因
            conversation_info.last_rejected_reply_content = generated_content # 记录被拒内容
            final_reason = f"等待超时后回复检查不通过: {check_reason}"
            # 如果检查不通过，通常意味着需要重新规划或放弃回复
            return False, "recall", final_reason

        # 发送回复 (如果通过检查)
        emoji_prepared_info: Optional[Tuple[Seg, str, str]] = None # 初始化表情准备信息
        emoji_query = conversation_info.current_emoji_query # 获取当前表情查询
        if emoji_query: # 如果有表情查询
            emoji_prepared_info = await self._fetch_and_prepare_emoji_segment(emoji_query) # 获取并准备表情
            conversation_info.current_emoji_query = None # 清理查询

        segments_to_send: List[Seg] = [Seg(type="text", data=generated_content)] # 初始化待发送Seg列表
        log_content_parts: List[str] = [f"文本:'{generated_content[:20]}...'"] # 初始化日志内容部分
        full_emoji_desc_for_event: Optional[str] = None # 初始化事件用表情描述

        if emoji_prepared_info: # 如果准备了表情
            emoji_segment, full_emoji_desc, _ = emoji_prepared_info
            segments_to_send.append(emoji_segment) # 添加表情Seg
            log_content_parts.append(f"表情:'{full_emoji_desc}'") # 添加表情到日志
            full_emoji_desc_for_event = full_emoji_desc # 记录完整描述

        self.conversation.state = ConversationState.SENDING # 设置状态为发送中
        self.conversation.generated_reply = generated_content # 记录生成的回复（主要文本部分）
        full_log_content = " ".join(log_content_parts) # 组合完整日志内容

        send_success = await self._send_reply_or_segments(segments_to_send, full_log_content) # 发送
        send_time = time.time() # 记录发送时间

        if send_success: # 如果发送成功
            action_successful = True # 标记动作成功
            final_status = "done" # 设置状态为完成
            final_reason = f"成功发送等待超时后的回复: {full_log_content}" # 设置最终原因
            await self._update_bot_message_in_history(send_time, generated_content, observation_info) # 更新机器人发送的文本消息到历史
            if emoji_prepared_info and full_emoji_desc_for_event: # 如果也发送了表情
                 await self._update_bot_message_in_history(send_time + 0.001, f"(附带表情: {full_emoji_desc_for_event})", observation_info, "bot_emoji_accompany_") # 单独记录表情（或合并描述）

            if conversation_info: # 更新对话信息
                conversation_info.last_successful_reply_action = "reply_after_wait_timeout" # 记录成功动作类型
                conversation_info.last_reply_rejection_reason = None # 清除拒绝原因
                conversation_info.last_rejected_reply_content = None # 清除被拒内容
                conversation_info.my_message_count += 1 # 增加机器人发言计数
                conversation_info.current_instance_message_count += 1 # 增加实例消息计数
                conversation_info.wait_has_timed_out = False # 超时状态已被处理，重置
                conversation_info.last_wait_duration_minutes = None # 清除等待时长

            event_desc = f"你在等待对方许久未回应后，发送了消息: '{generated_content[:30]}...'" # 事件描述
            if emoji_prepared_info: event_desc += f" 并可能附带了表情。"
            await self._update_post_send_states(observation_info, conversation_info, "reply_after_wait_timeout", event_desc) # 更新发送后状态
        else: # 如果发送失败
            final_reason = f"发送等待超时后的回复失败: {full_log_content}"
            # 发送失败，保持 wait_has_timed_out = True (如果之前是True的话)，让Planner下次能感知到
        return action_successful, final_status, final_reason


class UnknownActionHandler(ActionHandler): # (保持不变)
    """处理未知或无效动作的处理器。"""
    async def execute(
        self, reason: str, observation_info: Optional[ObservationInfo],
        conversation_info: Optional[ConversationInfo], action_start_time: float,
        current_action_record: dict
    ) -> tuple[bool, str, str]:
        action_name = current_action_record.get("action", "未知动作类型") # 从记录中获取动作名
        self.logger.warning(f"[私聊][{self.conversation.private_name}] 接收到未知的动作类型: {action_name}")
        return False, "recall", f"未知的动作类型: {action_name}" # 标记为需要重新规划