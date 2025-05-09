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

# 时区配置 (从 conversation.py 移过来，或者考虑放到更全局的配置模块)
configured_tz = getattr(global_config, "TIME_ZONE", "Asia/Shanghai")
TIME_ZONE = tz.gettz(configured_tz)
if TIME_ZONE is None:
    logger.error(f"配置的时区 '{configured_tz}' 无效，将使用默认时区 'Asia/Shanghai'")
    TIME_ZONE = tz.gettz("Asia/Shanghai")


async def run_conversation_loop(conversation_instance: "Conversation"):
    """
    核心的规划与行动循环 (PFC Loop)。
    之前是 Conversation 类中的 _plan_and_action_loop 方法。
    """
    logger.debug(f"[私聊][{conversation_instance.private_name}] 进入 run_conversation_loop 循环。")

    if not conversation_instance._initialized:
        logger.error(f"[私聊][{conversation_instance.private_name}] 尝试在未初始化状态下运行规划循环，退出。")
        return

    force_reflect_and_act = False # 用于强制使用反思 prompt 的标志

    while conversation_instance.should_continue:
        loop_iter_start_time = time.time()
        logger.debug(f"[私聊][{conversation_instance.private_name}] 开始新一轮循环迭代 ({loop_iter_start_time:.2f})")

        # 更新当前时间
        try:
            global TIME_ZONE # 引用全局 TIME_ZONE
            if TIME_ZONE is None: # 如果还未加载成功
                configured_tz_loop = getattr(global_config, "TIME_ZONE", "Asia/Shanghai")
                TIME_ZONE = tz.gettz(configured_tz_loop)
                if TIME_ZONE is None:
                    logger.error(f"循环中: 配置的时区 '{configured_tz_loop}' 无效，将使用 'Asia/Shanghai'")
                    TIME_ZONE = tz.gettz("Asia/Shanghai")

            current_time_dt = datetime.datetime.now(TIME_ZONE)
            if conversation_instance.observation_info:
                time_str = current_time_dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
                conversation_instance.observation_info.current_time_str = time_str
                logger.debug(f"[私聊][{conversation_instance.private_name}] 更新 ObservationInfo 当前时间: {time_str}")
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
            # 更新关系和情绪文本 (在每次循环开始时进行)
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

            # 检查核心组件
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

            # 规划
            planning_start_time = time.time() # 这是 ActionPlanner.plan 开始的时间
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
                use_reflect_prompt=force_reflect_and_act,
            )
            force_reflect_and_act = False # 重置反思标志
            logger.debug(
                f"[私聊][{conversation_instance.private_name}] (Loop) ActionPlanner.plan 完成，初步规划动作: {action}"
            )

            # 检查在 ActionPlanner.plan 期间是否有中断 (这部分逻辑保持不变)
            # 注意：这里的 planning_start_time 是 action_planner.plan 开始的时间
            current_unprocessed_messages_after_plan = getattr(conversation_instance.observation_info, "unprocessed_messages", [])
            new_messages_during_action_planning: List[Dict[str, Any]] = []
            other_new_messages_during_action_planning: List[Dict[str, Any]] = []

            for msg_ap in current_unprocessed_messages_after_plan:
                msg_time_ap = msg_ap.get("time")
                sender_id_info_ap = msg_ap.get("user_info", {})
                sender_id_ap = str(sender_id_info_ap.get("user_id")) if sender_id_info_ap else None
                if msg_time_ap and msg_time_ap >= planning_start_time: # 使用 action_planner.plan 的开始时间
                    new_messages_during_action_planning.append(msg_ap)
                    if sender_id_ap != conversation_instance.bot_qq_str:
                        other_new_messages_during_action_planning.append(msg_ap)
            
            new_msg_count_action_planning = len(new_messages_during_action_planning)
            other_new_msg_count_action_planning = len(other_new_messages_during_action_planning)

            # 更新因 ActionPlanner.plan 期间新消息而产生的计数和状态 (这部分逻辑也基本保持)
            if conversation_instance.conversation_info and other_new_msg_count_action_planning > 0:
                # (如果需要，这里可以更新实例消息计数、关系、情绪等，但通常这些在消息实际处理后更新更合适)
                # conversation_instance.conversation_info.current_instance_message_count += other_new_msg_count_action_planning
                pass


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
                # 记录中断的动作
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
                await asyncio.sleep(0.1) # 短暂休眠再开始下一轮
                continue # 跳过本轮的 actions.handle_action，直接进入下一轮循环重新规划

            # 如果 ActionPlanner.plan 后没有中断，则准备执行动作
            # 【核心修改点】对于需要LLM生成回复的动作，进行特殊处理
            if action in ["direct_reply", "send_new_message"]:
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] (Loop) 动作 '{action}' 需要LLM生成，进入监控执行模式..."
                )
                llm_call_start_time = time.time() # LLM实际调用开始的时间
                
                # 将 conversation_info 中用于 action_planner 中断的计数值传递或更新，以供 handle_action 使用
                # actions.handle_action 内部可能也需要知道这些信息
                if conversation_instance.conversation_info:
                    # 注意：这个字段可能在 actions.handle_action 中被使用和重置
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
                while not llm_action_task.done():
                    try:
                        await asyncio.wait_for(llm_action_task, timeout=5) # 每1.5秒检查一次
                    except asyncio.TimeoutError:
                        # LLM任务仍在运行，检查新消息
                        current_unprocessed_messages_during_llm = getattr(conversation_instance.observation_info, "unprocessed_messages", [])
                        other_new_messages_this_check: List[Dict[str, Any]] = []
                        
                        # 打印调试信息，与用户提供的一致
                        # print(111111111111111111111111) # 和用户调试信息一致
                        for msg_llm in current_unprocessed_messages_during_llm:
                            # print(msg_llm) # 和用户调试信息一致
                            msg_time_llm = msg_llm.get("time")
                            sender_id_info_llm = msg_llm.get("user_info", {})
                            sender_id_llm = str(sender_id_info_llm.get("user_id")) if sender_id_info_llm else None
                            
                            if msg_time_llm and msg_time_llm >= llm_call_start_time: # 注意这里用 llm_call_start_time
                                if sender_id_llm != conversation_instance.bot_qq_str:
                                    other_new_messages_this_check.append(msg_llm)
                                    # print("添加成功！\n") # 和用户调试信息一致
                        
                        # print(other_new_messages_this_check) # 和用户调试信息一致
                        # print(len(other_new_messages_this_check)) # 和用户调试信息一致

                        if len(other_new_messages_this_check) > 2: # 用户的重新规划条件
                            logger.info(
                                f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 执行期间收到 {len(other_new_messages_this_check)} 条来自他人的新消息，中断并重新规划。"
                            )
                            llm_action_task.cancel()
                            interrupted_during_llm = True
                            
                            # 记录中断的动作到 history
                            cancel_record_llm = {
                                "action": action,
                                "plan_reason": reason, # 使用规划时得到的 reason
                                "status": "cancelled_due_to_new_messages_during_llm",
                                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "final_reason": f"LLM处理期间收到{len(other_new_messages_this_check)}条新用户消息",
                            }
                            if conversation_instance.conversation_info:
                                if not hasattr(conversation_instance.conversation_info, "done_action") or conversation_instance.conversation_info.done_action is None:
                                    conversation_instance.conversation_info.done_action = []
                                conversation_instance.conversation_info.done_action.append(cancel_record_llm)
                                conversation_instance.conversation_info.last_successful_reply_action = None # 因为没有成功回复
                                # 可以在这里也更新 current_instance_message_count 和相关情绪/关系，如果这些新消息确实是用户的
                                conversation_instance.conversation_info.current_instance_message_count += len(other_new_messages_this_check)
                                # (此处可以添加调用关系/情绪更新的逻辑，如果需要的话)

                            conversation_instance.state = ConversationState.ANALYZING # 准备重新规划
                            force_reflect_and_act = True # 下一轮强制使用初始/反思型规划
                            break # 跳出监控循环
                    except asyncio.CancelledError:
                        logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务被取消。")
                        interrupted_during_llm = True # 标记为中断
                        # conversation_instance.state 和 force_reflect_and_act 已在上面处理 cancellation 的地方设置
                        break # 跳出监控循环
                    except Exception as e_llm_task:
                        logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务执行时出错: {e_llm_task}")
                        logger.error(traceback.format_exc())
                        interrupted_during_llm = True # 标记为中断，按错误处理
                        conversation_instance.state = ConversationState.ERROR
                        # 记录错误的动作到 history
                        error_record_llm = {
                            "action": action,
                            "plan_reason": reason,
                            "status": "error_during_llm_action",
                            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "final_reason": f"LLM动作执行时发生错误: {str(e_llm_task)}",
                        }
                        if conversation_instance.conversation_info:
                            if not hasattr(conversation_instance.conversation_info, "done_action") or conversation_instance.conversation_info.done_action is None:
                                conversation_instance.conversation_info.done_action = []
                            conversation_instance.conversation_info.done_action.append(error_record_llm)
                            conversation_instance.conversation_info.last_successful_reply_action = None
                        break # 跳出监控循环
                
                if interrupted_during_llm:
                    await asyncio.sleep(0.1) # 短暂休眠
                    continue # 如果LLM任务被中断，则直接进入下一轮PFC主循环以重新规划

                # 如果LLM任务正常完成 (没有被中断或出错跳出)
                # actions.handle_action 内部会处理其结果和状态更新
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) LLM动作 '{action}' 任务正常完成。")

            else: # 对于非LLM生成类的动作，直接执行
                logger.debug(
                    f"[私聊][{conversation_instance.private_name}] (Loop) 执行非LLM类动作 '{action}'..."
                )
                if conversation_instance.conversation_info: # 确保传递最新的计数值
                    conversation_instance.conversation_info.other_new_messages_during_planning_count = other_new_msg_count_action_planning
                
                await actions.handle_action(
                    conversation_instance,
                    action,
                    reason,
                    conversation_instance.observation_info,
                    conversation_instance.conversation_info,
                )
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 非LLM类动作 '{action}' 完成。")

            # 检查是否需要反思 (这部分逻辑保持不变)
            last_action_record = {}
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                last_action_record = conversation_instance.conversation_info.done_action[-1]
            if (
                last_action_record.get("action") == "send_new_message"
                and last_action_record.get("status") == "done_no_reply"
            ):
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 检测到需反思，设置标志。")
                force_reflect_and_act = True

            # 检查结束条件 (这部分逻辑保持不变)
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

            last_action_record_for_end_check = {}
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
            await asyncio.sleep(5) # 出错后等待一段时间

        # 控制循环频率
        loop_duration = time.time() - loop_iter_start_time
        min_loop_interval = 0.1 # 最小循环间隔，避免过于频繁的空转
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 循环迭代耗时: {loop_duration:.3f} 秒。")
        if loop_duration < min_loop_interval:
            await asyncio.sleep(min_loop_interval - loop_duration)

    logger.info(
        f"[私聊][{conversation_instance.private_name}] (Loop) PFC 循环已退出 for stream_id: {conversation_instance.stream_id}"
    )