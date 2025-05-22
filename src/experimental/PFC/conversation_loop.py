import time
import asyncio
import datetime # 确保导入 datetime
import traceback
from typing import Dict, Any, List, TYPE_CHECKING, Optional

from dateutil import tz # 确保导入 tz

from src.common.logger_manager import get_logger
from src.config.config import global_config
from .pfc_types import ConversationState
from . import actions

# 为 SubMind 调用新增的导入
from .pfc_utils import retrieve_contextual_info, build_chat_history_text

if TYPE_CHECKING:
    from .conversation import Conversation
    # 如果 SubMind 类定义在不同的地方，确保这里的导入路径正确
    # from experimental.Legacy_HFC.heart_flow.sub_mind import SubMind

logger = get_logger("pfc_loop")

# 时区配置 (从 global_config 或默认值获取)
# TIME_ZONE_STR = global_config.schedule.time_zone or "Asia/Shanghai" # 从配置读取
# TIME_ZONE = tz.gettz(TIME_ZONE_STR)
# if TIME_ZONE is None:
#     logger.error(f"无法加载配置的时区 '{TIME_ZONE_STR}'，将使用默认的 'Asia/Shanghai'")
#     TIME_ZONE = tz.gettz("Asia/Shanghai")
# 改为在循环内部获取和检查，以确保 global_config 已加载
TIME_ZONE: Optional[datetime.tzinfo] = None


try:
    MAX_CONSECUTIVE_LLM_ACTION_FAILURES = global_config.pfc.pfc_max_consecutive_llm_action_failures
except AttributeError:
    MAX_CONSECUTIVE_LLM_ACTION_FAILURES = 3
    logger.warning("Config 'global_config.pfc.pfc_max_consecutive_llm_action_failures' not found, using default value: 3")

async def run_conversation_loop(conversation_instance: "Conversation"):
    """
    核心的规划与行动循环 (PFC Loop)。
    """
    global TIME_ZONE # 允许修改全局的TIME_ZONE变量（在此模块作用域内）

    logger.debug(f"[私聊][{conversation_instance.private_name}] 进入 run_conversation_loop 循环。")

    if not conversation_instance._initialized:
        logger.error(f"[私聊][{conversation_instance.private_name}] 尝试在未初始化状态下运行规划循环，退出。")
        await conversation_instance.stop() # 确保停止
        return

    _force_reflect_and_act_next_iter = False

    while conversation_instance.should_continue:
        loop_iter_start_time = time.time()
        current_force_reflect_and_act = _force_reflect_and_act_next_iter
        _force_reflect_and_act_next_iter = False

        logger.debug(
            f"[私聊][{conversation_instance.private_name}] 开始新一轮循环迭代 ({loop_iter_start_time:.2f}), "
            f"force_reflect_next_iter: {current_force_reflect_and_act}, "
            f"consecutive_llm_failures: {conversation_instance.consecutive_llm_action_failures}"
        )

        try:
            # 确保 TIME_ZONE 已初始化
            if TIME_ZONE is None:
                time_zone_str_from_config = global_config.schedule.time_zone or "Asia/Shanghai"
                TIME_ZONE = tz.gettz(time_zone_str_from_config)
                if TIME_ZONE is None:
                    logger.error(f"无法加载配置的时区 '{time_zone_str_from_config}'，临时使用 'Asia/Shanghai'")
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

        if (
            conversation_instance.ignore_until_timestamp
            and loop_iter_start_time < conversation_instance.ignore_until_timestamp
        ):
            if conversation_instance.idle_chat and hasattr(conversation_instance.idle_chat, '_running') and conversation_instance.idle_chat._running: # type: ignore
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
            await conversation_instance.stop() # 这会设置 should_continue = False
            continue # 应该会因为 should_continue 退出循环
        else:
            pass # 正常流程

        try:
            # 更新关系和情绪文本 (这些通常在ActionHandler的post_send_states中更新，但规划前也确保一下)
            if conversation_instance.conversation_info and conversation_instance._initialized:
                if (
                    conversation_instance.conversation_info.person_id and
                    conversation_instance.relationship_translator and
                    conversation_instance.person_info_mng
                ):
                    try:
                        numeric_relationship_value = await conversation_instance.person_info_mng.get_value(
                            conversation_instance.conversation_info.person_id, "relationship_value"
                        )
                        if not isinstance(numeric_relationship_value, (int, float)):
                            from bson.decimal128 import Decimal128 # type: ignore
                            if isinstance(numeric_relationship_value, Decimal128):
                                numeric_relationship_value = float(numeric_relationship_value.to_decimal())
                            else: numeric_relationship_value = 0.0
                        conversation_instance.conversation_info.relationship_text = \
                            await conversation_instance.relationship_translator.translate_relationship_value_to_text(numeric_relationship_value)
                    except Exception as e_rel_loop:
                        logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) 更新关系文本时出错: {e_rel_loop}")
                        if conversation_instance.conversation_info : conversation_instance.conversation_info.relationship_text = "你们的关系是：普通。"
                if conversation_instance.mood_mng and conversation_instance.conversation_info:
                    conversation_instance.conversation_info.current_emotion_text = conversation_instance.mood_mng.get_mood_prompt()

            if not all(
                [
                    conversation_instance.action_planner,
                    conversation_instance.observation_info,
                    conversation_instance.conversation_info,
                    conversation_instance.sub_mind_instance_for_pfc, # 确保SubMind实例也已初始化
                    conversation_instance.chat_stream # 确保ChatStream已初始化
                ]
            ):
                logger.error(
                    f"[私聊][{conversation_instance.private_name}] 核心组件未完全初始化，无法继续规划循环。将等待5秒后重试..."
                )
                await asyncio.sleep(5)
                continue

            # ========== SubMind 调用开始 ==========
            logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) PFC的SubMind准备开始思考...")
            try:
                # 1. 准备 SubMind 需要的 PFC 上下文参数
                pfc_done_actions_param = conversation_instance.conversation_info.done_action
                previous_pfc_thought_param = conversation_instance.conversation_info.previous_pfc_thought

                # 2. 获取并设置“回想的聊天记录”
                chat_history_text_for_context_retrieval = await build_chat_history_text(
                    conversation_instance.observation_info, # type: ignore
                    conversation_instance.private_name
                )
                query_for_private_history_parts = []
                if conversation_instance.observation_info and conversation_instance.observation_info.unprocessed_messages:
                    for msg_dict in conversation_instance.observation_info.unprocessed_messages:
                        text_content = msg_dict.get("processed_plain_text", "")
                        if text_content.strip(): query_for_private_history_parts.append(text_content)
                query_for_private_history_str = " ".join(query_for_private_history_parts).strip() or chat_history_text_for_context_retrieval

                short_term_hist_earliest_time: Optional[float] = None
                if conversation_instance.observation_info and conversation_instance.observation_info.chat_history:
                    display_limit = global_config.pfc.pfc_recent_history_display_count
                    relevant_history_slice = conversation_instance.observation_info.chat_history[-display_limit:]
                    if relevant_history_slice:
                        short_term_hist_earliest_time = relevant_history_slice[0].get('time')

                _, _, retrieved_historical_chat_str_val = await retrieve_contextual_info(
                    text=chat_history_text_for_context_retrieval,
                    private_name=conversation_instance.private_name,
                    chat_id=conversation_instance.stream_id,
                    historical_chat_query_text=query_for_private_history_str,
                    current_short_term_history_earliest_time=short_term_hist_earliest_time
                )
                # 将获取到的历史回忆存起来，供SubMind的Prompt使用
                if conversation_instance.conversation_info:
                    conversation_instance.conversation_info.retrieved_historical_chat_for_submind = retrieved_historical_chat_str_val

                # 3. 调用 SubMind 的思考方法
                # 确保 sub_mind_instance_for_pfc 存在
                if conversation_instance.sub_mind_instance_for_pfc:
                    pfc_current_thought_val, _, tool_calls_str_from_submind, pfc_structured_info_dict_val = \
                        await conversation_instance.sub_mind_instance_for_pfc.do_thinking_before_reply(
                            pfc_done_actions=pfc_done_actions_param,
                            previous_pfc_thought=previous_pfc_thought_param,
                            retrieved_historical_chat_str_pfc=retrieved_historical_chat_str_val
                            # HFC参数都为None
                        )
                    if tool_calls_str_from_submind:
                        logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) PFC SubMind 在本轮思考中使用了工具: {tool_calls_str_from_submind}")
                        # TODO: 未来可以考虑是否需要基于此进行PFC主循环内的工具处理或SubMind递归调用
                        # 当前：SubMind内部已处理工具执行并将结果放入structured_info

                    # 4. 存储思考结果到 ConversationInfo
                    if conversation_instance.conversation_info:
                        conversation_instance.conversation_info.current_pfc_thought = pfc_current_thought_val
                        conversation_instance.conversation_info.pfc_structured_info = pfc_structured_info_dict_val
                        conversation_instance.conversation_info.previous_pfc_thought = pfc_current_thought_val
                    logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) PFC SubMind 思考完成: '{str(pfc_current_thought_val)[:100]}...'")
                else:
                    logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) PFC的SubMind实例未初始化！")
                    if conversation_instance.conversation_info:
                        conversation_instance.conversation_info.current_pfc_thought = "[SubMind实例错误]"
                        conversation_instance.conversation_info.pfc_structured_info = {}

            except Exception as submind_err:
                logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) 调用 PFC SubMind 时出错: {submind_err}")
                logger.error(traceback.format_exc())
                if conversation_instance.conversation_info:
                    conversation_instance.conversation_info.current_pfc_thought = "[PFCSubMind 思考出错]"
                    conversation_instance.conversation_info.pfc_structured_info = {}
            # ========== SubMind 调用结束 ==========


            planning_start_time = time.time() # 规划开始时间点
            logger.debug(
                f"[私聊][{conversation_instance.private_name}] --- (Loop) 开始规划 ({planning_start_time:.2f}) ---"
            )
            if conversation_instance.conversation_info:
                conversation_instance.conversation_info.other_new_messages_during_planning_count = 0 # 重置

            action: str
            reason: str
            if conversation_instance.action_planner and conversation_instance.observation_info and conversation_instance.conversation_info:
                action, reason = await conversation_instance.action_planner.plan(
                    conversation_instance.observation_info,
                    conversation_instance.conversation_info,
                    conversation_instance.conversation_info.last_successful_reply_action,
                    use_reflect_prompt=current_force_reflect_and_act,
                )
            else:
                logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) ActionPlanner 或其依赖未初始化，无法规划！")
                action, reason = "wait", "内部错误，ActionPlanner无法执行"


            logger.debug(
                f"[私聊][{conversation_instance.private_name}] (Loop) ActionPlanner.plan 完成，初步规划动作: {action}"
            )

            # --- 规划后新消息检查与处理 ---
            # (这部分逻辑与原文件基本一致，用于处理在ActionPlanner思考期间收到的新消息)
            current_unprocessed_messages_after_plan = getattr(
                conversation_instance.observation_info, "unprocessed_messages", []
            )
            new_messages_during_action_planning: List[Dict[str, Any]] = []
            other_new_messages_during_action_planning: List[Dict[str, Any]] = []

            for msg_ap in current_unprocessed_messages_after_plan:
                msg_time_ap = msg_ap.get("time")
                sender_id_info_ap = msg_ap.get("user_info", {})
                sender_id_ap = str(sender_id_info_ap.get("user_id")) if sender_id_info_ap else None
                # planning_start_time 是 ActionPlanner.plan() 开始的时间
                if msg_time_ap and msg_time_ap >= planning_start_time:
                    new_messages_during_action_planning.append(msg_ap)
                    if sender_id_ap != conversation_instance.bot_qq_str:
                        other_new_messages_during_action_planning.append(msg_ap)
            
            new_msg_count_action_planning = len(new_messages_during_action_planning)
            other_new_msg_count_action_planning = len(other_new_messages_during_action_planning)

            if conversation_instance.conversation_info and other_new_msg_count_action_planning > 0:
                # 将规划期间收到的他人新消息数存入conversation_info，供后续判断
                conversation_instance.conversation_info.other_new_messages_during_planning_count = other_new_msg_count_action_planning


            should_interrupt_action_planning: bool = False
            interrupt_reason_action_planning: str = ""
            if action in ["wait", "listening"] and new_msg_count_action_planning > 0:
                should_interrupt_action_planning = True
                interrupt_reason_action_planning = f"规划 {action} 期间收到 {new_msg_count_action_planning} 条新消息"
            elif other_new_msg_count_action_planning > global_config.pfc.pfc_message_buffer_size:
                should_interrupt_action_planning = True
                interrupt_reason_action_planning = (
                    f"规划 {action} 期间收到 {other_new_msg_count_action_planning} 条来自他人的新消息 (超过缓冲区大小)"
                )

            if should_interrupt_action_planning:
                logger.info(
                    f"[私聊][{conversation_instance.private_name}] (Loop) 中断 '{action}' (在ActionPlanner.plan后)，原因: {interrupt_reason_action_planning}。重新规划..."
                )
                cancel_record_ap = {
                    "action": action, "plan_reason": reason,
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
                await asyncio.sleep(0.1) # 短暂等待，让其他任务（如ChatObserver）有机会运行
                continue # 直接进入下一次主循环迭代

            # --- LLM Action Handling with Shield and Failure Count ---
            # (这部分复杂的LLM动作监控和处理逻辑与原文件保持一致)
            if action in ["direct_reply", "send_new_message", "reply_after_wait_timeout"]: # reply_after_wait_timeout 也是LLM驱动的
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 动作 '{action}' 需要LLM生成，进入监控执行模式...")
                llm_call_start_time = time.time()

                # other_new_messages_during_planning_count 已经在上面计算并存入 conversation_info

                llm_action_task = asyncio.create_task(
                    actions.handle_action(
                        conversation_instance,
                        action,
                        reason,
                        conversation_instance.observation_info,
                        conversation_instance.conversation_info,
                    )
                )
                interrupted_by_new_messages = False
                llm_action_completed_successfully = False
                action_outcome_processed = False

                while not llm_action_task.done() and not action_outcome_processed:
                    try:
                        await asyncio.wait_for(asyncio.shield(llm_action_task), timeout=1.5)
                        action_outcome_processed = True
                    except asyncio.TimeoutError:
                        current_unprocessed_messages_during_llm = getattr(conversation_instance.observation_info, "unprocessed_messages", [])
                        other_new_messages_this_check: List[Dict[str, Any]] = []
                        for msg_llm in current_unprocessed_messages_during_llm:
                            msg_time_llm = msg_llm.get("time")
                            sender_id_info_llm = msg_llm.get("user_info", {})
                            sender_id_llm = str(sender_id_info_llm.get("user_id")) if sender_id_info_llm else None
                            if msg_time_llm and msg_time_llm >= llm_call_start_time and sender_id_llm != conversation_instance.bot_qq_str:
                                other_new_messages_this_check.append(msg_llm)
                        if len(other_new_messages_this_check) > global_config.pfc.pfc_message_buffer_size:
                            logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 执行期间收到 {len(other_new_messages_this_check)} 条来自他人的新消息，将取消LLM任务。")
                            if not llm_action_task.done(): llm_action_task.cancel()
                            interrupted_by_new_messages = True
                            action_outcome_processed = True
                
                action_final_status_in_history = "unknown"
                try:
                    await llm_action_task
                    if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                        if conversation_instance.conversation_info.done_action:
                            action_final_status_in_history = conversation_instance.conversation_info.done_action[-1].get("status", "unknown")
                    if action_final_status_in_history in ["done", "done_no_reply"]:
                        llm_action_completed_successfully = True
                        conversation_instance.consecutive_llm_action_failures = 0
                    else: # 任务完成但未成功
                        if not interrupted_by_new_messages: conversation_instance.consecutive_llm_action_failures += 1
                except asyncio.CancelledError:
                    logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务最终确认被取消。")
                    if not interrupted_by_new_messages: conversation_instance.consecutive_llm_action_failures += 1
                except Exception as e_llm_final:
                    logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务执行时发生最终错误: {e_llm_final}", exc_info=True)
                    conversation_instance.state = ConversationState.ERROR
                    if not interrupted_by_new_messages: conversation_instance.consecutive_llm_action_failures += 1
                
                # --- Post LLM Action Task Handling ---
                if not llm_action_completed_successfully:
                   
                    last_action_record_detail = {}
                    last_action_final_status_detail = "unknown"
                    # 从 conversation_info.done_action 获取上一个动作的最终状态
                    if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                        if conversation_instance.conversation_info.done_action:
                            last_action_record_detail = conversation_instance.conversation_info.done_action[-1]
                            last_action_final_status_detail = last_action_record_detail.get("status", "unknown")

                    if last_action_final_status_detail == "max_checker_attempts_failed":
                        # ... (处理ReplyChecker最大尝试失败的逻辑，强制wait)
                        
                        logger.warning(f"[私聊][{conversation_instance.private_name}] (Loop) 原规划动作因达到ReplyChecker最大尝试次数而失败。将强制执行 'wait' 动作。")
                        action_to_perform_now = "wait"
                        reason_for_forced_wait = f"原动作因ReplyChecker多次判定不合适而失败，现强制等待。"
                        if conversation_instance.conversation_info:
                            conversation_instance.conversation_info.last_successful_reply_action = None
                            conversation_instance.consecutive_llm_action_failures = 0 # 重置，因为这是checker的锅
                        await actions.handle_action(conversation_instance, action_to_perform_now, reason_for_forced_wait, conversation_instance.observation_info, conversation_instance.conversation_info)
                        _force_reflect_and_act_next_iter = False
                        await asyncio.sleep(0.1)
                        continue
                    elif conversation_instance.consecutive_llm_action_failures >= MAX_CONSECUTIVE_LLM_ACTION_FAILURES:
                        # ... (处理LLM连续失败的逻辑，强制wait)
                        logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) LLM相关动作连续失败或被取消 {conversation_instance.consecutive_llm_action_failures} 次。将强制等待并重置计数器。")
                        forced_wait_action_on_consecutive_failure = "wait"
                        reason_for_consecutive_failure_wait = f"LLM连续失败{conversation_instance.consecutive_llm_action_failures}次，强制等待"
                        conversation_instance.consecutive_llm_action_failures = 0
                        if conversation_instance.conversation_info: conversation_instance.conversation_info.last_successful_reply_action = None
                        await actions.handle_action(conversation_instance, forced_wait_action_on_consecutive_failure, reason_for_consecutive_failure_wait, conversation_instance.observation_info, conversation_instance.conversation_info)
                        _force_reflect_and_act_next_iter = False
                        conversation_instance.state = ConversationState.ANALYZING
                        await asyncio.sleep(1) # 给点时间让状态生效
                        continue
                    else: # 其他LLM失败或中断情况
                        conversation_instance.state = ConversationState.ANALYZING
                        logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作中断/失败，准备重新规划。Interrupted: {interrupted_by_new_messages}, Failures: {conversation_instance.consecutive_llm_action_failures}")
                        await asyncio.sleep(0.1)
                        continue
                    # END POST LLM FAILURE HANDLING
            else: # 非LLM驱动的动作 (如 wait, listening, end_conversation, rethink_goal, send_memes 等)
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 执行非LLM类动作 '{action}'...")
                conversation_instance.consecutive_llm_action_failures = 0 # 重置LLM失败计数
                # other_new_messages_during_planning_count 已经在上面存入 conversation_info
                await actions.handle_action(
                    conversation_instance, action, reason,
                    conversation_instance.observation_info,
                    conversation_instance.conversation_info,
                )
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 非LLM类动作 '{action}' 完成。")

            # --- 检查是否需要强制反思或结束对话 (逻辑与原文件一致) ---
            last_action_record = {}
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                if conversation_instance.conversation_info.done_action:
                    last_action_record = conversation_instance.conversation_info.done_action[-1]

            if (last_action_record.get("action") == "send_new_message" and
                last_action_record.get("status") == "done_no_reply"):
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 检测到 ReplyGenerator 决定不发送消息，下一轮将强制反思。")
                _force_reflect_and_act_next_iter = True
            
            goal_ended: bool = False
            if (conversation_instance.conversation_info and
                hasattr(conversation_instance.conversation_info, "goal_list") and
                conversation_instance.conversation_info.goal_list):
                last_goal_item = conversation_instance.conversation_info.goal_list[-1]
                current_goal = (last_goal_item.get("goal") if isinstance(last_goal_item, dict) else
                                (last_goal_item if isinstance(last_goal_item, str) else None))
                if current_goal == "结束对话": goal_ended = True

            last_action_record_for_end_check = {}
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                if conversation_instance.conversation_info.done_action:
                    last_action_record_for_end_check = conversation_instance.conversation_info.done_action[-1]
            
            action_ended: bool = (
                last_action_record_for_end_check.get("action") in ["end_conversation", "say_goodbye"] and
                last_action_record_for_end_check.get("status") == "done"
            )

            if goal_ended or action_ended:
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 检测到结束条件，停止循环。Goal ended: {goal_ended}, Action ended: {action_ended}")
                await conversation_instance.stop() # 这会设置 should_continue = False
                # continue # 应该会因为 should_continue 退出循环

        except asyncio.CancelledError:
            logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 主循环任务被取消。")
            await conversation_instance.stop() # 确保停止
            break # 明确跳出 while 循环
        except Exception as loop_err:
            logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 主循环出错: {loop_err}")
            logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) {traceback.format_exc()}")
            conversation_instance.state = ConversationState.ERROR
            # 考虑是否在这里也调用 stop() 或添加更强的错误恢复机制
            await asyncio.sleep(5) # 发生错误时等待一段时间再尝试

        loop_duration = time.time() - loop_iter_start_time
        min_loop_interval = global_config.pfc.get('pfc_min_loop_interval_seconds', 0.1) # 从配置读取或默认0.1秒
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 循环迭代耗时: {loop_duration:.3f} 秒。")
        if loop_duration < min_loop_interval:
            await asyncio.sleep(min_loop_interval - loop_duration)

    logger.info(
        f"[私聊][{conversation_instance.private_name}] (Loop) PFC 循环已退出 for stream_id: {conversation_instance.stream_id}"
    )