import time
import asyncio
import datetime
import traceback
import json
from typing import Optional, Set, TYPE_CHECKING

from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.plugins.utils.chat_message_builder import build_readable_messages
from .pfc_types import ConversationState
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo

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
                logger.info(f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 找到合适的回复，准备发送。")
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
                    logger.info(f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 成功发送回复.")

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

                    # 更新 idle_conversation_starter 的最后消息时间
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
                    f"[私聊][{conversation_instance.private_name}] 动作 '{action}': 达到最大尝试次数 ({max_reply_attempts})，未能生成/检查通过合适的回复。最终原因: {check_reason}"
                )
                final_status = "recall"  # 标记为 recall
                final_reason = f"尝试{max_reply_attempts}次后失败: {check_reason}"
                action_successful = False  # 确保 action_successful 为 False
                # 重置追问状态
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
                    # 更新 idle_conversation_starter 的最后消息时间
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
        if action not in ["direct_reply", "send_new_message", "say_goodbye"]:
            conversation_info.last_successful_reply_action = None
            # 清理可能残留的拒绝信息
            conversation_info.last_reply_rejection_reason = None
            conversation_info.last_rejected_reply_content = None

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
        # --- 无论成功与否，都执行 ---

        # 1. 重置临时存储的计数值
        if conversation_info:  # 确保 conversation_info 存在
            conversation_info.other_new_messages_during_planning_count = 0

        # 2. 更新动作历史记录的最终状态和原因
        # 优化：如果动作成功但状态仍是默认的 recall，则更新为 done
        if action_successful:
            # 如果动作标记为成功，但 final_status 仍然是初始的 "recall" 或者 "start"
            # (因为可能在try块中成功执行了但没有显式更新 final_status 为 "done")
            # 或者是 "done_no_reply" 这种特殊的成功状态
            if (
                final_status in ["recall", "start"] and action != "send_new_message"
            ):  # send_new_message + no_reply 是特殊成功
                final_status = "done"
                if not final_reason or final_reason == "动作未成功执行":  # 避免覆盖已有的具体成功原因
                    # 为不同类型的成功动作提供更具体的默认成功原因
                    if action == "wait":
                        # 检查 conversation_info.goal_list 是否存在且不为空
                        timeout_occurred = (
                            any(
                                "分钟，" in g.get("goal", "")
                                for g in conversation_info.goal_list
                                if isinstance(g, dict)
                            )
                            if conversation_info and conversation_info.goal_list
                            else False
                        )
                        final_reason = "等待完成" + (" (超时)" if timeout_occurred else " (收到新消息或中断)")
                    elif action == "listening":
                        final_reason = "进入倾听状态"
                    elif action in ["rethink_goal", "end_conversation", "block_and_ignore", "say_goodbye"]:
                        final_reason = f"成功执行 {action}"
                    elif action in ["direct_reply", "send_new_message"]:  # 正常发送成功的case
                        final_reason = "成功发送"
                    else:
                        final_reason = f"动作 {action} 成功完成"
            # 如果已经是 "done" 或 "done_no_reply"，则保留它们和它们对应的 final_reason

        else:  # action_successful is False
            # 如果动作标记为失败，且 final_status 还是 "recall" (初始值) 或 "start"
            if final_status in ["recall", "start"]:
                # 尝试从 conversation_info 中获取更具体的失败原因（例如 checker 的原因）
                # 这个 specific_rejection_reason 是在 try 块中被设置的
                specific_rejection_reason = getattr(conversation_info, "last_reply_rejection_reason", None)
                rejected_content = getattr(conversation_info, "last_rejected_reply_content", None)

                if specific_rejection_reason:  # 如果有更具体的原因
                    final_reason = f"执行失败: {specific_rejection_reason}"
                    if (
                        rejected_content and specific_rejection_reason == "机器人尝试发送重复消息"
                    ):  # 对复读提供更清晰的日志
                        final_reason += f" (内容: '{rejected_content[:30]}...')"
                elif not final_reason or final_reason == "动作未成功执行":  # 如果没有更具体的原因，且当前原因还是默认的
                    final_reason = f"动作 {action} 执行失败或被意外中止"
            # 如果 final_status 已经是 "error" 或 "cancelled"，则保留它们和它们对应的 final_reason

        # 更新 done_action 中的记录
        # 防御性检查，确保 conversation_info, done_action 存在，并且索引有效
        if (
            conversation_info
            and hasattr(conversation_info, "done_action")
            and conversation_info.done_action
            and action_index < len(conversation_info.done_action)
        ):
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

        # 最终日志输出
        log_final_reason = final_reason if final_reason else "无明确原因"
        # 为成功发送的动作添加发送内容摘要
        if (
            final_status == "done"
            and action_successful
            and action in ["direct_reply", "send_new_message"]
            and hasattr(conversation_instance, "generated_reply")
            and conversation_instance.generated_reply
        ):
            log_final_reason += f" (发送内容: '{conversation_instance.generated_reply[:30]}...')"

        logger.info(
            f"[私聊][{conversation_instance.private_name}] 动作 '{action}' 处理完成。最终状态: {final_status}, 原因: {log_final_reason}"
        )
