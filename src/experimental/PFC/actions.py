import time
import asyncio
import datetime
import traceback
import json
from typing import Optional, Set, TYPE_CHECKING
from src.chat.emoji_system.emoji_manager import emoji_manager
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.utils.chat_message_builder import build_readable_messages
from .pfc_types import ConversationState
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from src.chat.utils.utils_image import image_path_to_base64  # 假设路径正确
from maim_message import Seg, UserInfo  # 从 maim_message 导入 Seg 和 UserInfo
from src.chat.message_receive.message import MessageSending, MessageSet  # PFC 的发送器依赖这些
from src.chat.message_receive.message_sender import message_manager  # PFC 的发送器依赖这个

if TYPE_CHECKING:
    from .conversation import Conversation  # 用于类型提示以避免循环导入

logger = get_logger("pfc_actions")


async def _send_reply_internal(conversation_instance: "Conversation") -> bool:
    """
    内部辅助函数，用于发送 conversation_instance.generated_reply 中的内容。
    这之前是 Conversation 类中的 _send_reply 方法。
    """
    # 检查是否有内容可发送
    if not conversation_instance.generated_reply:
        logger.warning(f"[私聊][{conversation_instance.private_name}] 没有生成回复内容，无法发送。")
        return False
    # 检查发送器和聊天流是否已初始化
    if not conversation_instance.direct_sender:
        logger.error(f"[私聊][{conversation_instance.private_name}] DirectMessageSender 未初始化，无法发送。")
        return False
    if not conversation_instance.chat_stream:
        logger.error(f"[私聊][{conversation_instance.private_name}] ChatStream 未初始化，无法发送。")
        return False

    try:
        reply_content = conversation_instance.generated_reply
        # 调用发送器发送消息，不指定回复对象
        await conversation_instance.direct_sender.send_message(
            chat_stream=conversation_instance.chat_stream,
            content=reply_content,
            reply_to_message=None,  # 私聊通常不需要引用回复
        )
        # 自身发言数量累计 +1
        if conversation_instance.conversation_info:  # 确保 conversation_info 存在
            conversation_instance.conversation_info.my_message_count += 1
        # 发送成功后，将状态设置回分析，准备下一轮规划
        conversation_instance.state = ConversationState.ANALYZING
        return True  # 返回成功
    except Exception as e:
        # 捕获发送过程中的异常
        logger.error(f"[私聊][{conversation_instance.private_name}] 发送消息时失败: {str(e)}")
        logger.error(f"[私聊][{conversation_instance.private_name}] {traceback.format_exc()}")
        conversation_instance.state = ConversationState.ERROR  # 发送失败标记错误状态
        return False  # 返回失败


async def handle_action(
    conversation_instance: "Conversation",
    action: str,
    reason: str,
    observation_info: Optional[ObservationInfo],
    conversation_info: Optional[ConversationInfo],
):
    """
    处理由 ActionPlanner 规划出的具体行动。
    这之前是 Conversation 类中的 _handle_action 方法。
    """
    # 检查初始化状态
    if not conversation_instance._initialized:
        logger.error(f"[私聊][{conversation_instance.private_name}] 尝试在未初始化状态下处理动作 '{action}'。")
        return

    # 确保 observation_info 和 conversation_info 不为 None
    if not observation_info:
        logger.error(f"[私聊][{conversation_instance.private_name}] ObservationInfo 为空，无法处理动作 '{action}'。")
        # 在 conversation_info 和 done_action 存在时更新状态
        if conversation_info and hasattr(conversation_info, "done_action") and conversation_info.done_action:
            conversation_info.done_action[-1].update(
                {
                    "status": "error",
                    "final_reason": "ObservationInfo is None",
                }
            )
        conversation_instance.state = ConversationState.ERROR
        return
    if not conversation_info:  # conversation_info 在这里是必需的
        logger.error(f"[私聊][{conversation_instance.private_name}] ConversationInfo 为空，无法处理动作 '{action}'。")
        conversation_instance.state = ConversationState.ERROR
        return

    logger.info(f"[私聊][{conversation_instance.private_name}] 开始处理动作: {action}, 原因: {reason}")
    action_start_time = time.time()  # 记录动作开始时间

    # --- 准备动作历史记录条目 ---
    current_action_record = {
        "action": action,
        "plan_reason": reason,  # 记录规划时的原因
        "status": "start",  # 初始状态为"开始"
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 记录开始时间
        "final_reason": None,  # 最终结果的原因，将在 finally 中设置
    }
    # 安全地添加到历史记录列表
    if not hasattr(conversation_info, "done_action") or conversation_info.done_action is None:  # 防御性检查
        conversation_info.done_action = []
    conversation_info.done_action.append(current_action_record)
    # 获取当前记录在列表中的索引，方便后续更新状态
    action_index = len(conversation_info.done_action) - 1

    # --- 初始化动作执行状态变量 ---
    action_successful: bool = False  # 标记动作是否成功执行
    final_status: str = "recall"  # 动作最终状态，默认为 recall (表示未成功或需重试)
    final_reason: str = "动作未成功执行"  # 动作最终原因

    # 在此声明变量以避免 UnboundLocalError
    is_suitable: bool = False
    generated_content_for_check_or_send: str = ""
    check_reason: str = "未进行检查"
    need_replan_from_checker: bool = False
    should_send_reply: bool = True  # 默认需要发送 (对于 direct_reply)
    is_send_decision_from_rg: bool = False  # 标记 send_new_message 的决策是否来自 ReplyGenerator

    try:
        # --- 根据不同的 action 类型执行相应的逻辑 ---

        # 1. 处理需要生成、检查、发送的动作
        if action in ["direct_reply", "send_new_message"]:
            max_reply_attempts: int = getattr(global_config, "pfc_max_reply_attempts", 3)  # 最多尝试次数 (可配置)
            reply_attempt_count: int = 0
            # is_suitable, generated_content_for_check_or_send, check_reason, need_replan_from_checker, should_send_reply, is_send_decision_from_rg 已在外部声明

            while reply_attempt_count < max_reply_attempts and not is_suitable and not need_replan_from_checker:
                reply_attempt_count += 1
                log_prefix = f"[私聊][{conversation_instance.private_name}] 尝试生成/检查 '{action}' 回复 (第 {reply_attempt_count}/{max_reply_attempts} 次)..."
                logger.info(log_prefix)

                conversation_instance.state = ConversationState.GENERATING
                if not conversation_instance.reply_generator:
                    raise RuntimeError("ReplyGenerator 未初始化")

                raw_llm_output = await conversation_instance.reply_generator.generate(
                    observation_info, conversation_info, action_type=action
                )
                logger.debug(f"{log_prefix} ReplyGenerator.generate 返回: '{raw_llm_output}'")

                text_to_process = raw_llm_output  # 默认情况下，处理原始输出

                if action == "send_new_message":
                    is_send_decision_from_rg = True  # 标记这是 send_new_message 的决策过程
                    parsed_json = None
                    try:
                        # 尝试解析JSON
                        parsed_json = json.loads(raw_llm_output)
                    except json.JSONDecodeError:
                        logger.error(f"{log_prefix} ReplyGenerator 返回的不是有效的JSON: {raw_llm_output}")
                        # 如果JSON解析失败，视为RG决定不发送，并给出原因
                        conversation_info.last_reply_rejection_reason = "回复生成器未返回有效JSON"
                        conversation_info.last_rejected_reply_content = raw_llm_output
                        should_send_reply = False
                        text_to_process = "no"  # 或者一个特定的错误标记

                    if parsed_json:  # 如果成功解析
                        send_decision = parsed_json.get("send", "no").lower()
                        generated_text_from_json = parsed_json.get("txt", "no")

                        if send_decision == "yes":
                            should_send_reply = True
                            text_to_process = generated_text_from_json
                            logger.info(f"{log_prefix} ReplyGenerator 决定发送消息。内容: '{text_to_process[:100]}...'")
                        else:  # send_decision is "no"
                            should_send_reply = False
                            text_to_process = "no"  # 保持和 prompt 中一致，txt 为 "no"
                            logger.info(f"{log_prefix} ReplyGenerator 决定不发送消息。")
                            # 既然RG决定不发送，就直接跳出重试循环
                            break

                # 如果 ReplyGenerator 在 send_new_message 动作中决定不发送，则跳出重试循环
                if action == "send_new_message" and not should_send_reply:
                    break

                generated_content_for_check_or_send = text_to_process

                # 检查生成的内容是否有效
                if (
                    not generated_content_for_check_or_send
                    or generated_content_for_check_or_send.startswith("抱歉")
                    or generated_content_for_check_or_send.strip() == ""
                    or (
                        action == "send_new_message"
                        and generated_content_for_check_or_send == "no"
                        and should_send_reply
                    )
                ):  # RG决定发送但文本为"no"或空
                    warning_msg = f"{log_prefix} 生成内容无效或为错误提示"
                    if action == "send_new_message" and generated_content_for_check_or_send == "no":  # 特殊情况日志
                        warning_msg += " (ReplyGenerator决定发送但文本为'no')"

                    logger.warning(warning_msg + "，将进行下一次尝试 (如果适用)。")
                    check_reason = "生成内容无效或选择不发送"  # 统一原因
                    conversation_info.last_reply_rejection_reason = check_reason
                    conversation_info.last_rejected_reply_content = generated_content_for_check_or_send

                    await asyncio.sleep(0.5)  # 暂停一下
                    continue  # 直接进入下一次循环尝试

                # --- 内容检查 ---
                conversation_instance.state = ConversationState.CHECKING
                if not conversation_instance.reply_checker:
                    raise RuntimeError("ReplyChecker 未初始化")

                # 准备检查器所需参数
                current_goal_str = ""
                if conversation_info.goal_list:  # 确保 goal_list 存在且不为空
                    goal_item = conversation_info.goal_list[-1]
                    if isinstance(goal_item, dict):
                        current_goal_str = goal_item.get("goal", "")
                    elif isinstance(goal_item, str):
                        current_goal_str = goal_item

                chat_history_for_check = getattr(observation_info, "chat_history", [])
                chat_history_text_for_check = getattr(observation_info, "chat_history_str", "")
                current_retry_for_checker = reply_attempt_count - 1  # retry_count 从0开始
                current_time_value_for_check = observation_info.current_time_str or "获取时间失败"

                # 调用检查器
                if global_config.enable_pfc_reply_checker:
                    logger.debug(f"{log_prefix} 调用 ReplyChecker 检查 (配置已启用)...")
                    (
                        is_suitable,
                        check_reason,
                        need_replan_from_checker,
                    ) = await conversation_instance.reply_checker.check(
                        reply=generated_content_for_check_or_send,
                        goal=current_goal_str,
                        chat_history=chat_history_for_check,  # 使用完整的历史记录列表
                        chat_history_text=chat_history_text_for_check,  # 可以是截断的文本
                        current_time_str=current_time_value_for_check,
                        retry_count=current_retry_for_checker,  # 传递当前重试次数
                    )
                    logger.info(
                        f"{log_prefix} ReplyChecker 结果: 合适={is_suitable}, 原因='{check_reason}', 需重规划={need_replan_from_checker}"
                    )
                else:  # 如果配置关闭
                    is_suitable = True
                    check_reason = "ReplyChecker 已通过配置关闭"
                    need_replan_from_checker = False
                    logger.debug(f"{log_prefix} [配置关闭] ReplyChecker 已跳过，默认回复为合适。")

                # 处理检查结果
                if not is_suitable:
                    conversation_info.last_reply_rejection_reason = check_reason
                    conversation_info.last_rejected_reply_content = generated_content_for_check_or_send

                    # 如果是机器人自身复读，且检查器认为不需要重规划 (这是新版 ReplyChecker 的逻辑)
                    if check_reason == "机器人尝试发送重复消息" and not need_replan_from_checker:
                        logger.warning(
                            f"{log_prefix} 回复因自身重复被拒绝: {check_reason}。将使用相同 Prompt 类型重试。"
                        )
                        if reply_attempt_count < max_reply_attempts:  # 还有尝试次数
                            await asyncio.sleep(0.5)  # 暂停一下
                            continue  # 进入下一次重试
                        else:  # 达到最大次数
                            logger.warning(f"{log_prefix} 即使是复读，也已达到最大尝试次数。")
                            break  # 结束循环，按失败处理
                    elif (
                        not need_replan_from_checker and reply_attempt_count < max_reply_attempts
                    ):  # 其他不合适原因，但无需重规划，且可重试
                        logger.warning(f"{log_prefix} 回复不合适，原因: {check_reason}。将进行下一次尝试。")
                        await asyncio.sleep(0.5)  # 暂停一下
                        continue  # 进入下一次重试
                    else:  # 需要重规划，或达到最大次数
                        logger.warning(f"{log_prefix} 回复不合适且(需要重规划或已达最大次数)。原因: {check_reason}")
                        break  # 结束循环，将在循环外部处理
                else:  # is_suitable is True
                    # 找到了合适的回复
                    conversation_info.last_reply_rejection_reason = None  # 清除之前的拒绝原因
                    conversation_info.last_rejected_reply_content = None
                    break  # 成功，跳出循环

            # --- 循环结束后处理 ---
            if action == "send_new_message" and not should_send_reply and is_send_decision_from_rg:
                # 这是 reply_generator 决定不发送的情况
                logger.info(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': ReplyGenerator 决定不发送消息。"
                )
                final_status = "done_no_reply"  # 一个新的状态，表示动作完成但无回复
                final_reason = "回复生成器决定不发送消息"
                action_successful = True  # 动作本身（决策）是成功的

                # 清除追问状态，因为没有实际发送
                conversation_info.last_successful_reply_action = None
                conversation_info.my_message_count = 0  # 重置连续发言计数
                # 后续的 plan 循环会检测到这个 "done_no_reply" 状态并使用反思 prompt

            elif is_suitable:  # 适用于 direct_reply 或 (send_new_message 且 RG决定发送并通过检查)
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 找到合适的回复，准备发送。"
                )
                # conversation_info.last_reply_rejection_reason = None # 已在循环内清除
                # conversation_info.last_rejected_reply_content = None
                conversation_instance.generated_reply = generated_content_for_check_or_send  # 使用检查通过的内容
                timestamp_before_sending = time.time()
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 记录发送前时间戳: {timestamp_before_sending:.2f}"
                )
                conversation_instance.state = ConversationState.SENDING
                send_success = await _send_reply_internal(conversation_instance)  # 调用重构后的发送函数
                send_end_time = time.time()  # 记录发送完成时间

                if send_success:
                    action_successful = True
                    final_status = "done"  # 明确设置 final_status
                    final_reason = "成功发送"  # 明确设置 final_reason
                    logger.debug(f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 成功发送回复.")

                    # --- 新增：将机器人发送的消息添加到 ObservationInfo 的 chat_history ---
                    if (
                        observation_info and conversation_instance.bot_qq_str
                    ):  # 确保 observation_info 和 bot_qq_str 存在
                        bot_message_dict = {
                            "message_id": f"bot_sent_{send_end_time}",  # 生成一个唯一ID
                            "time": send_end_time,
                            "user_info": {  # 构造机器人的 UserInfo
                                "user_id": conversation_instance.bot_qq_str,
                                "user_nickname": global_config.BOT_NICKNAME,  # 或者 conversation_instance.name
                                "platform": conversation_instance.chat_stream.platform
                                if conversation_instance.chat_stream
                                else "unknown_platform",
                            },
                            "processed_plain_text": conversation_instance.generated_reply,
                            "detailed_plain_text": conversation_instance.generated_reply,  # 简单处理
                            # 根据你的消息字典结构，可能还需要其他字段
                        }
                        observation_info.chat_history.append(bot_message_dict)
                        observation_info.chat_history_count = len(observation_info.chat_history)
                        logger.debug(
                            f"[私聊][{conversation_instance.private_name}] {global_config.BOT_NICKNAME}发送的消息已添加到 chat_history。当前历史数: {observation_info.chat_history_count}"
                        )

                        # 可选：如果 chat_history 过长，进行修剪 (例如，保留最近N条)
                        max_history_len = getattr(global_config, "pfc_max_chat_history_for_checker", 50)  # 例如，可配置
                        if len(observation_info.chat_history) > max_history_len:
                            observation_info.chat_history = observation_info.chat_history[-max_history_len:]
                            observation_info.chat_history_count = len(observation_info.chat_history)  # 更新计数

                        # 更新 chat_history_str (如果 ReplyChecker 也依赖这个字符串)
                        # 这个更新可能比较消耗资源，如果 checker 只用列表，可以考虑优化此处
                        history_slice_for_str = observation_info.chat_history[-30:]  # 例如最近30条
                        try:
                            observation_info.chat_history_str = await build_readable_messages(
                                history_slice_for_str,
                                replace_bot_name=True,
                                merge_messages=False,
                                timestamp_mode="relative",
                                read_mark=0.0,
                            )
                        except Exception as e_build_hist:
                            logger.error(
                                f"[私聊][{conversation_instance.private_name}] 更新 chat_history_str 时出错: {e_build_hist}"
                            )
                            observation_info.chat_history_str = "[构建聊天记录出错]"
                    # --- 新增结束 ---

                    # 更新 idle_chat 的最后消息时间
                    # (避免在发送消息后很快触发主动聊天)
                    if conversation_instance.idle_chat:
                        await conversation_instance.idle_chat.update_last_message_time(send_end_time)

                    # 清理已处理的未读消息 (只清理在发送这条回复之前的、来自他人的消息)
                    current_unprocessed_messages = getattr(observation_info, "unprocessed_messages", [])
                    message_ids_to_clear: Set[str] = set()
                    for msg in current_unprocessed_messages:
                        msg_time = msg.get("time")
                        msg_id = msg.get("message_id")
                        sender_id_info = msg.get("user_info", {})  # 安全获取 user_info
                        sender_id = str(sender_id_info.get("user_id")) if sender_id_info else None  # 安全获取 sender_id

                        if (
                            msg_id  # 确保 msg_id 存在
                            and msg_time  # 确保 msg_time 存在
                            and sender_id != conversation_instance.bot_qq_str  # 确保是对方的消息
                            and msg_time < timestamp_before_sending  # 只清理发送前的
                        ):
                            message_ids_to_clear.add(msg_id)

                    if message_ids_to_clear:
                        logger.debug(
                            f"[私聊][{conversation_instance.private_name}] 准备清理 {len(message_ids_to_clear)} 条发送前(他人)消息: {message_ids_to_clear}"
                        )
                        await observation_info.clear_processed_messages(message_ids_to_clear)
                    else:
                        logger.debug(f"[私聊][{conversation_instance.private_name}] 没有需要清理的发送前(他人)消息。")

                    # 更新追问状态 和 关系/情绪状态
                    other_new_msg_count_during_planning = getattr(
                        conversation_info, "other_new_messages_during_planning_count", 0
                    )

                    # 如果是 direct_reply 且规划期间有他人新消息，则下次不追问
                    if other_new_msg_count_during_planning > 0 and action == "direct_reply":
                        logger.debug(
                            f"[私聊][{conversation_instance.private_name}] 因规划期间收到 {other_new_msg_count_during_planning} 条他人新消息，下一轮强制使用【初始回复】逻辑。"
                        )
                        conversation_info.last_successful_reply_action = None
                        # conversation_info.my_message_count 不在此处重置，因为它刚发了一条
                    elif action == "direct_reply" or action == "send_new_message":  # 成功发送后
                        logger.debug(
                            f"[私聊][{conversation_instance.private_name}] 成功执行 '{action}', 下一轮【允许】使用追问逻辑。"
                        )
                        conversation_info.last_successful_reply_action = action

                    # 更新实例消息计数和关系/情绪
                    if conversation_info:  # 再次确认
                        conversation_info.current_instance_message_count += 1
                        logger.debug(
                            f"[私聊][{conversation_instance.private_name}] 实例消息计数({global_config.BOT_NICKNAME}发送后)增加到: {conversation_info.current_instance_message_count}"
                        )

                        if conversation_instance.relationship_updater:  # 确保存在
                            await conversation_instance.relationship_updater.update_relationship_incremental(
                                conversation_info=conversation_info,
                                observation_info=observation_info,
                                chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                            )

                        sent_reply_summary = (
                            conversation_instance.generated_reply[:50]
                            if conversation_instance.generated_reply
                            else "空回复"
                        )
                        event_for_emotion_update = f"你刚刚发送了消息: '{sent_reply_summary}...'"
                        if conversation_instance.emotion_updater:  # 确保存在
                            await conversation_instance.emotion_updater.update_emotion_based_on_context(
                                conversation_info=conversation_info,
                                observation_info=observation_info,
                                chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                                event_description=event_for_emotion_update,
                            )
                else:  # 发送失败
                    logger.error(f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 发送回复失败。")
                    final_status = "recall"  # 标记为 recall 或 error
                    final_reason = "发送回复时失败"
                    action_successful = False  # 确保 action_successful 为 False
                    # 发送失败，重置追问状态和计数
                    conversation_info.last_successful_reply_action = None
                    conversation_info.my_message_count = 0

            elif need_replan_from_checker:  # 如果检查器要求重规划
                logger.warning(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}' 因 ReplyChecker 要求而被取消，将重新规划。原因: {check_reason}"
                )
                final_status = "recall"  # 标记为 recall
                final_reason = f"回复检查要求重新规划: {check_reason}"
                # 重置追问状态，因为没有成功发送
                conversation_info.last_successful_reply_action = None
                # my_message_count 保持不变，因为没有成功发送

            else:  # 达到最大尝试次数仍未找到合适回复 (is_suitable is False and not need_replan_from_checker)
                logger.warning(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 达到最大回复尝试次数 ({max_reply_attempts})，ReplyChecker 仍判定不合适。最终检查原因: {check_reason}"
                )
                final_status = "max_checker_attempts_failed"
                final_reason = f"达到最大回复尝试次数({max_reply_attempts})，ReplyChecker仍判定不合适: {check_reason}"
                action_successful = False
                if conversation_info:  # 确保 conversation_info 存在
                    conversation_info.last_successful_reply_action = None
                # my_message_count 保持不变

        # 2. 处理发送告别语动作 (保持简单，不加重试)
        elif action == "say_goodbye":
            conversation_instance.state = ConversationState.GENERATING
            if not conversation_instance.reply_generator:
                raise RuntimeError("ReplyGenerator 未初始化")
            # 生成告别语
            generated_content = await conversation_instance.reply_generator.generate(
                observation_info,
                conversation_info,
                action_type=action,  # action_type='say_goodbye'
            )
            logger.info(
                f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 生成内容: '{generated_content[:100]}...'"
            )

            # 检查生成内容
            if not generated_content or generated_content.startswith("抱歉"):
                logger.warning(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 生成内容为空或为错误提示，取消发送。"
                )
                final_reason = "生成内容无效"
                # 即使生成失败，也按计划结束对话
                final_status = "done"  # 标记为 done，因为目的是结束
                conversation_instance.should_continue = False  # 停止对话
                logger.info(f"[私聊][{conversation_instance.private_name}] 告别语生成失败，仍按计划结束对话。")
            else:
                # 发送告别语
                conversation_instance.generated_reply = generated_content
                timestamp_before_sending = time.time()
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 记录发送前时间戳: {timestamp_before_sending:.2f}"
                )
                conversation_instance.state = ConversationState.SENDING
                send_success = await _send_reply_internal(conversation_instance)  # 调用重构后的发送函数
                send_end_time = time.time()

                if send_success:
                    action_successful = True  # 标记成功
                    # final_status 和 final_reason 会在 finally 中设置
                    logger.info(f"[私聊][{conversation_instance.private_name}] 成功发送告别语，即将停止对话实例。")
                    # 更新 idle_chat 的最后消息时间
                    # (避免在发送消息后很快触发主动聊天)
                    if conversation_instance.idle_chat:
                        await conversation_instance.idle_chat.update_last_message_time(send_end_time)
                    # 清理发送前的消息 (虽然通常是最后一条，但保持逻辑一致)
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
                            and sender_id != conversation_instance.bot_qq_str  # 不是自己的消息
                            and msg_time < timestamp_before_sending  # 发送前
                        ):
                            message_ids_to_clear.add(msg_id)
                    if message_ids_to_clear:
                        await observation_info.clear_processed_messages(message_ids_to_clear)

                    # 更新关系和情绪
                    if conversation_info:  # 确保 conversation_info 存在
                        conversation_info.current_instance_message_count += 1
                        logger.debug(
                            f"[私聊][{conversation_instance.private_name}] 实例消息计数(告别语后)增加到: {conversation_info.current_instance_message_count}"
                        )

                    sent_reply_summary = (
                        conversation_instance.generated_reply[:50]
                        if conversation_instance.generated_reply
                        else "空回复"
                    )
                    event_for_emotion_update = f"你发送了告别消息: '{sent_reply_summary}...'"
                    if conversation_instance.emotion_updater:  # 确保存在
                        await conversation_instance.emotion_updater.update_emotion_based_on_context(
                            conversation_info=conversation_info,
                            observation_info=observation_info,
                            chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                            event_description=event_for_emotion_update,
                        )
                    # 发送成功后结束对话
                    conversation_instance.should_continue = False
                else:
                    # 发送失败
                    logger.error(f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 发送告别语失败。")
                    final_status = "recall"  # 或 "error"
                    final_reason = "发送告别语失败"
                    # 发送失败不能结束对话，让其自然流转或由其他逻辑结束
                    conversation_instance.should_continue = True  # 保持 should_continue

        # 3. 处理重新思考目标动作
        elif action == "rethink_goal":
            conversation_instance.state = ConversationState.RETHINKING
            if not conversation_instance.goal_analyzer:
                raise RuntimeError("GoalAnalyzer 未初始化")
            # 调用 GoalAnalyzer 分析并更新目标
            await conversation_instance.goal_analyzer.analyze_goal(conversation_info, observation_info)
            action_successful = True  # 标记成功
            event_for_emotion_update = "你重新思考了对话目标和方向"
            if (
                conversation_instance.emotion_updater and conversation_info and observation_info
            ):  # 确保updater和info都存在
                await conversation_instance.emotion_updater.update_emotion_based_on_context(
                    conversation_info=conversation_info,
                    observation_info=observation_info,
                    chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                    event_description=event_for_emotion_update,
                )

        # 4. 处理倾听动作
        elif action == "listening":
            conversation_instance.state = ConversationState.LISTENING
            if not conversation_instance.waiter:
                raise RuntimeError("Waiter 未初始化")
            logger.info(f"[私聊][{conversation_instance.private_name}] 动作 'listening': 进入倾听状态...")
            # 调用 Waiter 的倾听等待方法，内部会处理超时
            await conversation_instance.waiter.wait_listening(conversation_info)  # 直接传递 conversation_info
            action_successful = True  # listening 动作本身执行即视为成功，后续由新消息或超时驱动
            event_for_emotion_update = "你决定耐心倾听对方的发言"
            if conversation_instance.emotion_updater and conversation_info and observation_info:  # 确保都存在
                await conversation_instance.emotion_updater.update_emotion_based_on_context(
                    conversation_info=conversation_info,
                    observation_info=observation_info,
                    chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                    event_description=event_for_emotion_update,
                )

        # 5. 处理结束对话动作
        elif action == "end_conversation":
            logger.info(
                f"[私聊][{conversation_instance.private_name}] 动作 'end_conversation': 收到最终结束指令，停止对话..."
            )
            action_successful = True  # 标记成功
            conversation_instance.should_continue = False  # 设置标志以退出循环

        # 6. 处理屏蔽忽略动作
        elif action == "block_and_ignore":
            logger.info(f"[私聊][{conversation_instance.private_name}] 动作 'block_and_ignore': 不想再理你了...")
            ignore_duration_seconds = 10 * 60  # 忽略 10 分钟，可配置
            conversation_instance.ignore_until_timestamp = time.time() + ignore_duration_seconds
            logger.info(
                f"[私聊][{conversation_instance.private_name}] 将忽略此对话直到: {datetime.datetime.fromtimestamp(conversation_instance.ignore_until_timestamp)}"
            )
            conversation_instance.state = ConversationState.IGNORED  # 设置忽略状态
            action_successful = True  # 标记成功
            event_for_emotion_update = "当前对话让你感到不适，你决定暂时不再理会对方"
            if conversation_instance.emotion_updater and conversation_info and observation_info:  # 确保都存在
                await conversation_instance.emotion_updater.update_emotion_based_on_context(
                    conversation_info=conversation_info,
                    observation_info=observation_info,
                    chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                    event_description=event_for_emotion_update,
                )

        # X. 处理发送表情包动作
        elif action == "send_memes":
            conversation_instance.state = ConversationState.GENERATING
            final_reason_prefix = "发送表情包"
            action_successful = False  # 先假设不成功

            # 确保 conversation_info 和 observation_info 存在
            if not conversation_info or not observation_info:
                logger.error(
                    f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': ConversationInfo 或 ObservationInfo 为空。"
                )
                final_status = "error"
                final_reason = f"{final_reason_prefix}失败：内部信息缺失"
                # done_action 的更新会在 finally 中处理
                # 理论上这不应该发生，因为调用 handle_action 前应该有检查
                # 但作为防御性编程，可以加上
                if conversation_info:  # 即使另一个为空，也尝试更新
                    conversation_info.last_successful_reply_action = None
                # 直接跳到 finally 块
                # 注意：此处不能直接 return，否则 finally 不会被完整执行。
                # 而是让后续的 final_status 和 action_successful 决定流程。
                # 这里我们通过设置 action_successful = False 和 final_status = "error" 来让 finally 处理
                # 更好的方式可能是直接在 finally 前面抛出异常，但为了简化，我们先这样。
                # 此处保持 action_successful = False，后续的 finally 会处理状态。
                pass  # 让代码继续到 try...except...finally 的末尾

            else:  # conversation_info 和 observation_info 都存在
                emoji_query = conversation_info.current_emoji_query
                if not emoji_query:
                    logger.warning(
                        f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': emoji_query 为空，无法获取表情包。"
                    )
                    final_status = "recall"
                    final_reason = f"{final_reason_prefix}失败：缺少表情包查询语句"
                    conversation_info.last_successful_reply_action = None
                else:
                    logger.info(
                        f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': 使用查询 '{emoji_query}' 获取表情包..."
                    )
                    try:
                        emoji_result = await emoji_manager.get_emoji_for_text(emoji_query)

                        if emoji_result:
                            emoji_path, emoji_description = emoji_result
                            logger.info(
                                f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': 获取到表情包: {emoji_path}, 描述: {emoji_description}"
                            )

                            if not conversation_instance.chat_stream:
                                logger.error(
                                    f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': ChatStream 未初始化，无法发送。"
                                )
                                raise RuntimeError("ChatStream 未初始化")

                            image_b64_content = image_path_to_base64(emoji_path)
                            if not image_b64_content:
                                logger.error(
                                    f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': 无法将图片 {emoji_path} 转换为 base64。"
                                )
                                raise ValueError(f"无法将图片 {emoji_path} 转换为Base64")

                            # --- 统一 Seg 构造方式 (与群聊类似) ---
                            # 直接使用 type="emoji" 和 base64 数据
                            message_segment_for_emoji = Seg(type="emoji", data=image_b64_content)
                            # --------------------------------------

                            bot_user_info = UserInfo(
                                user_id=global_config.BOT_QQ,
                                user_nickname=global_config.BOT_NICKNAME,
                                platform=conversation_instance.chat_stream.platform,
                            )
                            message_id_emoji = f"pfc_meme_{round(time.time(), 3)}"

                            # --- 直接使用 DirectMessageSender (如果其 send_message 适配单个 Seg) ---
                            # 或者如果 DirectMessageSender.send_message 需要 content: str，
                            # 我们就需要调整 DirectMessageSender 或这里的逻辑。
                            # 假设 DirectMessageSender 能被改造或其依赖的 message_manager 能处理 Seg 对象。
                            # 我们先按照 PFC/message_sender.py 的结构来尝试构造 MessageSending
                            # PFC/message_sender.py 中的 DirectMessageSender.send_message(content: str)
                            # 它内部是 segments = Seg(type="seglist", data=[Seg(type="text", data=content)])
                            # 这意味着 DirectMessageSender 目前只设计来发送文本。

                            # *** 为了与群聊发送逻辑一致，并假设底层 message_manager 可以处理任意 Seg ***
                            # 我们需要绕过 DirectMessageSender 的 send_message(content: str)
                            # 或者修改 DirectMessageSender 以接受 Seg 对象。
                            # 更简单的做法是直接调用与群聊相似的底层发送机制，即构造 MessageSending 并使用 message_manager

                            message_to_send = MessageSending(
                                message_id=message_id_emoji,
                                chat_stream=conversation_instance.chat_stream,
                                bot_user_info=bot_user_info,
                                sender_info=None,  # 表情通常不是对特定消息的回复
                                message_segment=message_segment_for_emoji,  # 直接使用构造的 Seg
                                reply=None,
                                is_head=True,
                                is_emoji=True,
                                thinking_start_time=action_start_time,  # 使用动作开始时间作为近似
                            )

                            await message_to_send.process()  # 消息预处理

                            message_set_emoji = MessageSet(conversation_instance.chat_stream, message_id_emoji)
                            message_set_emoji.add_message(message_to_send)
                            await message_manager.add_message(message_set_emoji)  # 使用全局管理器发送

                            logger.info(
                                f"[私聊][{conversation_instance.private_name}] PFC 动作 'send_memes': 表情包已发送: {emoji_path} ({emoji_description})"
                            )
                            action_successful = True  # 标记发送成功
                            # final_status 和 final_reason 会在 finally 中设置

                            # --- 后续成功处理逻辑 (与之前相同，但要确保 conversation_info 存在) ---
                            if conversation_info:
                                conversation_info.my_message_count += 1
                                conversation_info.last_successful_reply_action = action
                                conversation_info.current_instance_message_count += 1
                                logger.debug(
                                    f"[私聊][{conversation_instance.private_name}] 成功执行 '{action}', my_message_count: {conversation_info.my_message_count}, 下一轮将使用【追问】逻辑。"
                                )

                            current_send_time = time.time()
                            if conversation_instance.idle_chat:
                                await conversation_instance.idle_chat.update_last_message_time(current_send_time)

                            if observation_info and conversation_instance.bot_qq_str:
                                bot_meme_message_dict = {
                                    "message_id": message_id_emoji,
                                    "time": current_send_time,
                                    "user_info": bot_user_info.to_dict(),
                                    "processed_plain_text": f"[表情包: {emoji_description}]",
                                    "detailed_plain_text": f"[表情包: {emoji_path} - {emoji_description}]",
                                    "raw_message": "[CQ:image,file=base64://...]",  # 示例
                                }
                                observation_info.chat_history.append(bot_meme_message_dict)
                                observation_info.chat_history_count = len(observation_info.chat_history)
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
                                except Exception as e_build_hist_meme:
                                    logger.error(
                                        f"[私聊][{conversation_instance.private_name}] 更新 chat_history_str (表情包后) 时出错: {e_build_hist_meme}"
                                    )

                            current_unprocessed_messages_emoji = observation_info.unprocessed_messages
                            message_ids_to_clear_emoji: Set[str] = set()
                            for msg_item in current_unprocessed_messages_emoji:
                                msg_time_item = msg_item.get("time")
                                msg_id_item = msg_item.get("message_id")
                                sender_id_info_item = msg_item.get("user_info", {})
                                sender_id_item = (
                                    str(sender_id_info_item.get("user_id")) if sender_id_info_item else None
                                )
                                if (
                                    msg_id_item
                                    and msg_time_item
                                    and sender_id_item != conversation_instance.bot_qq_str
                                    and msg_time_item < current_send_time
                                ):
                                    message_ids_to_clear_emoji.add(msg_id_item)
                            if message_ids_to_clear_emoji:
                                await observation_info.clear_processed_messages(message_ids_to_clear_emoji)

                            if conversation_instance.relationship_updater and conversation_info:
                                await conversation_instance.relationship_updater.update_relationship_incremental(
                                    conversation_info=conversation_info,
                                    observation_info=observation_info,
                                    chat_observer_for_history=conversation_instance.chat_observer,
                                )
                            event_for_emotion_update_emoji = f"你发送了一个表情包 ({emoji_description})"
                            if conversation_instance.emotion_updater and conversation_info:
                                await conversation_instance.emotion_updater.update_emotion_based_on_context(
                                    conversation_info=conversation_info,
                                    observation_info=observation_info,
                                    chat_observer_for_history=conversation_instance.chat_observer,
                                    event_description=event_for_emotion_update_emoji,
                                )
                        else:  # emoji_result is None
                            logger.warning(
                                f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': 未能根据查询 '{emoji_query}' 获取到合适的表情包。"
                            )
                            final_status = "recall"
                            final_reason = f"{final_reason_prefix}失败：未找到合适表情包"
                            action_successful = False
                            if conversation_info:
                                conversation_info.last_successful_reply_action = None
                                conversation_info.current_emoji_query = None

                    except Exception as get_send_emoji_err:
                        logger.error(
                            f"[私聊][{conversation_instance.private_name}] 动作 'send_memes': 处理过程中出错: {get_send_emoji_err}"
                        )
                        logger.error(traceback.format_exc())
                        final_status = "recall"  # 或 "error"
                        final_reason = f"{final_reason_prefix}失败：处理表情包时出错 ({get_send_emoji_err})"
                        action_successful = False
                        if conversation_info:
                            conversation_info.last_successful_reply_action = None
                            conversation_info.current_emoji_query = None

        # 7. 处理等待动作
        elif action == "wait":
            conversation_instance.state = ConversationState.WAITING
            if not conversation_instance.waiter:
                raise RuntimeError("Waiter 未初始化")
            logger.info(f"[私聊][{conversation_instance.private_name}] 动作 'wait': 进入等待状态...")
            # 调用 Waiter 的常规等待方法，内部处理超时
            # wait 方法返回是否超时 (True=超时, False=未超时/被新消息中断)
            timeout_occurred = await conversation_instance.waiter.wait(conversation_info)  # 直接传递 conversation_info
            action_successful = True  # wait 动作本身执行即视为成功
            event_for_emotion_update = ""
            if timeout_occurred:  # 假设 timeout_occurred 能正确反映是否超时
                event_for_emotion_update = "你等待对方回复，但对方长时间没有回应"
            else:
                event_for_emotion_update = "你选择等待对方的回复（对方可能很快回复了）"

            if conversation_instance.emotion_updater and conversation_info and observation_info:  # 确保都存在
                await conversation_instance.emotion_updater.update_emotion_based_on_context(
                    conversation_info=conversation_info,
                    observation_info=observation_info,
                    chat_observer_for_history=conversation_instance.chat_observer,  # 确保 chat_observer 存在
                    event_description=event_for_emotion_update,
                )
            # wait 动作完成后不需要清理消息，等待新消息或超时触发重新规划
            logger.debug(f"[私聊][{conversation_instance.private_name}] Wait 动作完成，无需在此清理消息。")

        # 8. 处理未知的动作类型
        else:
            logger.warning(f"[私聊][{conversation_instance.private_name}] 未知的动作类型: {action}")
            final_status = "recall"  # 未知动作标记为 recall
            final_reason = f"未知的动作类型: {action}"

        # --- 重置非回复动作的追问状态 ---
        # 确保执行完非回复动作后，下一次规划不会错误地进入追问逻辑
        if action not in ["direct_reply", "send_new_message", "say_goodbye", "send_memes"]:
            conversation_info.last_successful_reply_action = None
            # 清理可能残留的拒绝信息
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None

        if action != "send_memes" or not action_successful:
            if conversation_info and hasattr(conversation_info, "current_emoji_query"):
                conversation_info.current_emoji_query = None

    except asyncio.CancelledError:
        # 处理任务被取消的异常
        logger.warning(f"[私聊][{conversation_instance.private_name}] 处理动作 '{action}' 时被取消。")
        final_status = "cancelled"
        final_reason = "动作处理被取消"
        # 取消时也重置追问状态
        if conversation_info:  # 确保 conversation_info 存在
            conversation_info.last_successful_reply_action = None
        raise  # 重新抛出 CancelledError，让上层知道任务被取消
    except Exception as handle_err:
        # 捕获处理动作过程中的其他所有异常
        logger.error(f"[私聊][{conversation_instance.private_name}] 处理动作 '{action}' 时出错: {handle_err}")
        logger.error(f"[私聊][{conversation_instance.private_name}] {traceback.format_exc()}")
        final_status = "error"  # 标记为错误状态
        final_reason = f"处理动作时出错: {handle_err}"
        conversation_instance.state = ConversationState.ERROR  # 设置对话状态为错误
        # 出错时重置追问状态
        if conversation_info:  # 确保 conversation_info 存在
            conversation_info.last_successful_reply_action = None

    finally:
        # --- 统一更新动作历史记录的最终状态和原因 ---
        # (确保 conversation_info 和 done_action[action_index] 有效)
        if (
            conversation_info
            and hasattr(conversation_info, "done_action")
            and conversation_info.done_action
            and action_index < len(conversation_info.done_action)
        ):
            # 确定最终状态和原因
            if action_successful:  # 如果动作本身标记为成功
                if final_status not in ["done", "done_no_reply"]:  # 如果没有被特定成功状态覆盖
                    final_status = "done"
                if not final_reason or final_reason == "动作未成功执行":
                    if action == "send_memes":
                        final_reason = f"{final_reason_prefix}成功发送"
                    # ... (其他动作的默认成功原因) ...
                    else:
                        final_reason = f"动作 {action} 成功完成"
            else:  # action_successful is False
                if final_status in ["recall", "start", "unknown"]:  # 如果状态还是初始或未定
                    # 尝试从 conversation_info 获取更具体的失败原因
                    specific_rejection_reason = getattr(conversation_info, "last_reply_rejection_reason", None)
                    if specific_rejection_reason and (not final_reason or final_reason == "动作未成功执行"):
                        final_reason = f"执行失败: {specific_rejection_reason}"
                    elif not final_reason or final_reason == "动作未成功执行":
                        final_reason = f"动作 {action} 执行失败或被中止"
                # 如果 final_status 已经是 "error", "cancelled", "max_checker_attempts_failed" 等，则保留

            conversation_info.done_action[action_index].update(
                {
                    "status": final_status,
                    "time_completed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "final_reason": final_reason,
                    "duration_ms": int((time.time() - action_start_time) * 1000),
                }
            )
        else:
            logger.error(
                f"[私聊][{conversation_instance.private_name}] 无法更新动作历史记录，索引 {action_index} 无效或列表为空。"
            )

        # --- 统一设置 ConversationState ---
        if final_status == "done" or final_status == "done_no_reply" or final_status == "recall":
            # "recall" 状态也应该回到 ANALYZING 准备重新规划
            conversation_instance.state = ConversationState.ANALYZING
        elif final_status == "error" or final_status == "max_checker_attempts_failed":
            conversation_instance.state = ConversationState.ERROR
        # 对于 "cancelled", "listening", "waiting", "ignored", "ended" 等状态，
        # 它们应该在各自的动作逻辑内部或者由外部 (如 loop) 来决定下一个 ConversationState。
        # 例如，end_conversation/say_goodbye 会设置 should_continue=False，loop 会退出。
        # listening/wait 会在动作完成后（可能因为新消息或超时）使 loop 自然进入下一轮 ANALYZING。
        # cancelled 会让 loop 捕获异常并停止。

        # 重置非回复动作的追问状态 (确保 send_memes 被视为回复动作)
        if action not in [
            "direct_reply",
            "send_new_message",
            "say_goodbye",
            "send_memes",
        ]:  # <--- 把 send_memes 加到这里
            if conversation_info:
                conversation_info.last_successful_reply_action = None
                conversation_info.last_reply_rejection_reason = None
                conversation_info.last_rejected_reply_content = None

        # 清理表情查询（如果动作不是send_memes但查询还存在，或者send_memes失败了）
        if action != "send_memes" or not action_successful:
            if conversation_info and hasattr(conversation_info, "current_emoji_query"):
                conversation_info.current_emoji_query = None

        log_final_reason_msg = final_reason if final_reason else "无明确原因"
        if (
            final_status == "done"
            and action_successful
            and action in ["direct_reply", "send_new_message"]  # send_memes 的发送内容不同
            and hasattr(conversation_instance, "generated_reply")
            and conversation_instance.generated_reply
        ):
            log_final_reason_msg += f" (发送内容: '{conversation_instance.generated_reply[:30]}...')"
        elif (
            final_status == "done" and action_successful and action == "send_memes"
            # emoji_description 在 send_memes 内部获取，这里不再重复记录到 log_final_reason_msg，
            # 因为 logger.info 已经记录过发送的表情描述
        ):
            pass

        logger.info(
            f"[私聊][{conversation_instance.private_name}] 动作 '{action}' 处理完成。最终状态: {final_status}, 原因: {log_final_reason_msg}"
        )
