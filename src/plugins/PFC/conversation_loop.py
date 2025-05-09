import time
import asyncio
import datetime
import traceback
from typing import Dict, Any, List, TYPE_CHECKING
from dateutil import tz

from src.common.logger_manager import get_logger
from src.config.config import global_config
from .pfc_types import ConversationState # 需要导入 ConversationState
from . import actions # 需要导入 actions 模块

if TYPE_CHECKING:
    from .conversation import Conversation

logger = get_logger("pfc_loop")

# 时区配置
configured_tz = getattr(global_config, "TIME_ZONE", "Asia/Shanghai")
TIME_ZONE = tz.gettz(configured_tz)
if TIME_ZONE is None:
    logger.error(f"配置的时区 '{configured_tz}' 无效，将使用默认时区 'Asia/Shanghai'")
    TIME_ZONE = tz.gettz("Asia/Shanghai")


async def run_conversation_loop(conversation_instance: "Conversation"):
    """
    核心的规划与行动循环 (PFC Loop)。
    """
    logger.debug(f"[私聊][{conversation_instance.private_name}] 进入 run_conversation_loop 循环。")

    if not conversation_instance._initialized:
        logger.error(f"[私聊][{conversation_instance.private_name}] 尝试在未初始化状态下运行规划循环，退出。")
        return

    # 注意：force_reflect_and_act 是在主循环的每次迭代开始前确定的，
    # 它在 action_planner.plan 使用后会被重置为 False。
    # 如果在一次迭代中被设为 True (例如因为LLM任务中断)，它会影响下一次迭代的 plan 调用。
    _force_reflect_and_act_next_iter = False

    while conversation_instance.should_continue:
        loop_iter_start_time = time.time()
        current_force_reflect_and_act = _force_reflect_and_act_next_iter
        _force_reflect_and_act_next_iter = False # 默认下一轮迭代不强制反思

        logger.debug(f"[私聊][{conversation_instance.private_name}] 开始新一轮循环迭代 ({loop_iter_start_time:.2f}), force_reflect_next_iter: {current_force_reflect_and_act}")

        # 更新当前时间
        try:
            global TIME_ZONE
            if TIME_ZONE is None:
                configured_tz_loop = getattr(global_config, "TIME_ZONE", "Asia/Shanghai")
                TIME_ZONE = tz.gettz(configured_tz_loop)
                if TIME_ZONE is None:
                    logger.error(f"循环中: 配置的时区 '{configured_tz_loop}' 无效，将使用 'Asia/Shanghai'")
                    TIME_ZONE = tz.gettz("Asia/Shanghai")

            current_time_dt = datetime.datetime.now(TIME_ZONE)
            if conversation_instance.observation_info:
                time_str = current_time_dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
                conversation_instance.observation_info.current_time_str = time_str
            else:
                logger.warning(
                    f"[私聊][{conversation_instance.private_name}] ObservationInfo 未初始化，无法更新当前时间。"
                )
        except Exception as time_update_err:
            logger.error(
                f"[私聊][{conversation_instance.private_name}] 更新 ObservationInfo 当前时间时出错: {time_update_err}"
            )

        # 处理忽略状态
        if (
            conversation_instance.ignore_until_timestamp
            and loop_iter_start_time < conversation_instance.ignore_until_timestamp
        ):
            if conversation_instance.idle_chat and conversation_instance.idle_chat._running:
                logger.debug(f"[私聊][{conversation_instance.private_name}] 对话被暂时忽略，暂停对该用户的主动聊天")
            sleep_duration = min(30, conversation_instance.ignore_until_timestamp - loop_iter_start_time)
            await asyncio.sleep(sleep_duration)
            continue
        elif (
            conversation_instance.ignore_until_timestamp
            and loop_iter_start_time >= conversation_instance.ignore_until_timestamp
        ):
            logger.info(
                f"[私聊][{conversation_instance.private_name}] 忽略时间已到 {conversation_instance.stream_id}，准备结束对话。"
            )
            conversation_instance.ignore_until_timestamp = None
            await conversation_instance.stop()
            continue
        else:
            pass

        # 核心规划与行动逻辑
        try:
            if conversation_instance.conversation_info and conversation_instance._initialized:
                if (
                    conversation_instance.conversation_info.person_id
                    and conversation_instance.relationship_translator
                    and conversation_instance.person_info_mng
                ):
                    try:
                        numeric_relationship_value = await conversation_instance.person_info_mng.get_value(
                            conversation_instance.conversation_info.person_id, "relationship_value"
                        )
                        if not isinstance(numeric_relationship_value, (int, float)):
                            from bson.decimal128 import Decimal128
                            if isinstance(numeric_relationship_value, Decimal128):
                                numeric_relationship_value = float(numeric_relationship_value.to_decimal())
                            else:
                                numeric_relationship_value = 0.0
                        conversation_instance.conversation_info.relationship_text = (
                            await conversation_instance.relationship_translator.translate_relationship_value_to_text(
                                numeric_relationship_value
                            )
                        )
                    except Exception as e_rel:
                        logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) 更新关系文本时出错: {e_rel}")
                        conversation_instance.conversation_info.relationship_text = "你们的关系是：普通。"
                if conversation_instance.mood_mng:
                    conversation_instance.conversation_info.current_emotion_text = (
                        conversation_instance.mood_mng.get_prompt()
                    )

            if not all(
                [
                    conversation_instance.action_planner,
                    conversation_instance.observation_info,
                    conversation_instance.conversation_info,
                ]
            ):
                logger.error(
                    f"[私聊][{conversation_instance.private_name}] 核心组件未初始化，无法继续规划循环。将等待5秒后重试..."
                )
                await asyncio.sleep(5)
                continue

            planning_start_time = time.time()
            logger.debug(
                f"[私聊][{conversation_instance.private_name}] --- (Loop) 开始规划 ({planning_start_time:.2f}) ---"
            )
            if conversation_instance.conversation_info:
                conversation_instance.conversation_info.other_new_messages_during_planning_count = 0

            action, reason = await conversation_instance.action_planner.plan(
                conversation_instance.observation_info,
                conversation_instance.conversation_info,
                conversation_instance.conversation_info.last_successful_reply_action
                if conversation_instance.conversation_info
                else None,
                use_reflect_prompt=current_force_reflect_and_act, # 使用当前迭代的强制反思标志
            )
            # current_force_reflect_and_act 已经被用于本次plan, _force_reflect_and_act_next_iter 默认为 False

            logger.debug(
                f"[私聊][{conversation_instance.private_name}] (Loop) ActionPlanner.plan 完成，初步规划动作: {action}"
            )

            current_unprocessed_messages_after_plan = getattr(conversation_instance.observation_info, "unprocessed_messages", [])
            new_messages_during_action_planning: List[Dict[str, Any]] = []
            other_new_messages_during_action_planning: List[Dict[str, Any]] = []

            for msg_ap in current_unprocessed_messages_after_plan:
                msg_time_ap = msg_ap.get("time")
                sender_id_info_ap = msg_ap.get("user_info", {})
                sender_id_ap = str(sender_id_info_ap.get("user_id")) if sender_id_info_ap else None
                if msg_time_ap and msg_time_ap >= planning_start_time:
                    new_messages_during_action_planning.append(msg_ap)
                    if sender_id_ap != conversation_instance.bot_qq_str:
                        other_new_messages_during_action_planning.append(msg_ap)
            
            new_msg_count_action_planning = len(new_messages_during_action_planning)
            other_new_msg_count_action_planning = len(other_new_messages_during_action_planning)

            if conversation_instance.conversation_info and other_new_msg_count_action_planning > 0:
                pass # 计数更新等通常在消息实际处理后

            should_interrupt_action_planning: bool = False
            interrupt_reason_action_planning: str = ""
            if action in ["wait", "listening"] and new_msg_count_action_planning > 0:
                should_interrupt_action_planning = True
                interrupt_reason_action_planning = f"规划 {action} 期间收到 {new_msg_count_action_planning} 条新消息"
            elif other_new_msg_count_action_planning > 2: 
                should_interrupt_action_planning = True
                interrupt_reason_action_planning = f"规划 {action} 期间收到 {other_new_msg_count_action_planning} 条来自他人的新消息"

            if should_interrupt_action_planning:
                logger.info(
                    f"[私聊][{conversation_instance.private_name}] (Loop) 中断 '{action}' (在ActionPlanner.plan后)，原因: {interrupt_reason_action_planning}。重新规划..."
                )
                cancel_record_ap = {
                    "action": action,
                    "plan_reason": reason,
                    "status": "cancelled_due_to_new_messages_during_action_plan",
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "final_reason": interrupt_reason_action_planning,
                }
                if conversation_instance.conversation_info:
                    if not hasattr(conversation_instance.conversation_info, "done_action") or conversation_instance.conversation_info.done_action is None:
                        conversation_instance.conversation_info.done_action = []
                    conversation_instance.conversation_info.done_action.append(cancel_record_ap)
                    conversation_instance.conversation_info.last_successful_reply_action = None
                conversation_instance.state = ConversationState.ANALYZING
                await asyncio.sleep(0.1)
                continue

            if action in ["direct_reply", "send_new_message"]:
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] (Loop) 动作 '{action}' 需要LLM生成，进入监控执行模式..."
                )
                llm_call_start_time = time.time()
                
                if conversation_instance.conversation_info:
                    conversation_instance.conversation_info.other_new_messages_during_planning_count = other_new_msg_count_action_planning

                llm_action_task = asyncio.create_task(
                    actions.handle_action(
                        conversation_instance,
                        action,
                        reason,
                        conversation_instance.observation_info,
                        conversation_instance.conversation_info,
                    )
                )

                interrupted_during_llm = False
                llm_task_cancelled_by_us = False

                while not llm_action_task.done():
                    try:
                        await asyncio.wait_for(llm_action_task, timeout=1.5)
                    except asyncio.TimeoutError:
                        current_time_for_check = time.time()
                        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) LLM Monitor Timeout. llm_call_start_time: {llm_call_start_time:.2f}, current_check_time: {current_time_for_check:.2f}")
                        
                        current_unprocessed_messages_during_llm = getattr(conversation_instance.observation_info, "unprocessed_messages", [])
                        other_new_messages_this_check: List[Dict[str, Any]] = []
                        
                        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) Checking unprocessed_messages (count: {len(current_unprocessed_messages_during_llm)}):")
                        for msg_llm in current_unprocessed_messages_during_llm:
                            msg_time_llm = msg_llm.get("time")
                            sender_id_info_llm = msg_llm.get("user_info", {})
                            sender_id_llm = str(sender_id_info_llm.get("user_id")) if sender_id_info_llm else None
                            is_new_enough = msg_time_llm and msg_time_llm >= llm_call_start_time
                            is_other_sender = sender_id_llm != conversation_instance.bot_qq_str
                            
                            logger.debug(f"  - Msg ID: {msg_llm.get('message_id')}, Time: {msg_time_llm:.2f if msg_time_llm else 'N/A'}, Sender: {sender_id_llm}. New enough? {is_new_enough}. Other sender? {is_other_sender}.")

                            if is_new_enough and is_other_sender:
                                other_new_messages_this_check.append(msg_llm)
                        
                        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) Found {len(other_new_messages_this_check)} 'other_new_messages_this_check'.")

                        if len(other_new_messages_this_check) > 2:
                            logger.info(
                                f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 执行期间收到 {len(other_new_messages_this_check)} 条来自他人的新消息，发起中断。"
                            )
                            if not llm_action_task.done():
                                llm_action_task.cancel()
                                llm_task_cancelled_by_us = True
                            break # 从监控循环中断
                    # except asyncio.CancelledError: # 这个由外部的 await llm_action_task 捕获
                    #     pass

                # 监控循环结束后，等待任务的最终结果
                try:
                    await llm_action_task
                    logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务最终完成 (未被取消或未发生错误)。")
                except asyncio.CancelledError:
                    logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务最终确认被取消。")
                    interrupted_during_llm = True 
                    # actions.handle_action 内部的finally块会更新done_action状态为cancelled
                except Exception as e_llm_final:
                    logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务执行时发生最终错误: {e_llm_final}")
                    logger.error(traceback.format_exc())
                    interrupted_during_llm = True
                    conversation_instance.state = ConversationState.ERROR
                    # actions.handle_action 内部的finally块会更新done_action状态为error

                if interrupted_during_llm:
                    conversation_instance.state = ConversationState.ANALYZING
                    # 如果是我们因为新消息主动取消的，下一轮不强制反思，正常规划
                    # 如果是其他原因（例如LLM内部错误导致任务提前结束并被视为中断），也可以考虑不强制反思
                    # _force_reflect_and_act_next_iter 保持其默认的 False
                    logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作中断，准备重新规划。llm_task_cancelled_by_us: {llm_task_cancelled_by_us}")
                    await asyncio.sleep(0.1)
                    continue
            else:
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] (Loop) 执行非LLM类动作 '{action}'..."
                )
                if conversation_instance.conversation_info:
                    conversation_instance.conversation_info.other_new_messages_during_planning_count = other_new_msg_count_action_planning
                
                await actions.handle_action(
                    conversation_instance,
                    action,
                    reason,
                    conversation_instance.observation_info,
                    conversation_instance.conversation_info,
                )
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 非LLM类动作 '{action}' 完成。")

            last_action_record = {}
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                last_action_record = conversation_instance.conversation_info.done_action[-1]
            
            # 只有当 ReplyGenerator 明确决定不发送时，才强制下一轮反思
            if (
                last_action_record.get("action") == "send_new_message"
                and last_action_record.get("status") == "done_no_reply" 
            ):
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 检测到 ReplyGenerator 决定不发送消息，下一轮将强制反思。")
                _force_reflect_and_act_next_iter = True


            goal_ended: bool = False
            if (
                conversation_instance.conversation_info
                and hasattr(conversation_instance.conversation_info, "goal_list")
                and conversation_instance.conversation_info.goal_list
            ):
                last_goal_item = conversation_instance.conversation_info.goal_list[-1]
                current_goal = (
                    last_goal_item.get("goal")
                    if isinstance(last_goal_item, dict)
                    else (last_goal_item if isinstance(last_goal_item, str) else None)
                )
                if current_goal == "结束对话":
                    goal_ended = True

            last_action_record_for_end_check = {} # 重新获取最新的记录
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                last_action_record_for_end_check = conversation_instance.conversation_info.done_action[-1]
            action_ended: bool = (
                last_action_record_for_end_check.get("action") in ["end_conversation", "say_goodbye"]
                and last_action_record_for_end_check.get("status") == "done"
            )

            if goal_ended or action_ended:
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 检测到结束条件，停止循环。")
                await conversation_instance.stop()
                continue

        except asyncio.CancelledError:
            logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 主循环任务被取消。")
            await conversation_instance.stop()
            break
        except Exception as loop_err:
            logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 主循环出错: {loop_err}")
            logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) {traceback.format_exc()}")
            conversation_instance.state = ConversationState.ERROR
            await asyncio.sleep(5)

        loop_duration = time.time() - loop_iter_start_time
        min_loop_interval = 0.1
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 循环迭代耗时: {loop_duration:.3f} 秒。")
        if loop_duration < min_loop_interval:
            await asyncio.sleep(min_loop_interval - loop_duration)

    logger.info(
        f"[私聊][{conversation_instance.private_name}] (Loop) PFC 循环已退出 for stream_id: {conversation_instance.stream_id}"
    )
