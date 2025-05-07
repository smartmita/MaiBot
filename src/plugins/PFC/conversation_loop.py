import time
import asyncio
import datetime
import traceback
from typing import Dict, Any, Optional, Set, List, TYPE_CHECKING
from dateutil import tz

from src.common.logger_manager import get_logger
from src.config.config import global_config
from .pfc_types import ConversationState # 需要导入 ConversationState
from . import actions # 需要导入 actions 模块

if TYPE_CHECKING:
    from .conversation import Conversation

logger = get_logger("pfc_loop")

# 时区配置 (从 conversation.py 移过来，或者考虑放到更全局的配置模块)
configured_tz = getattr(global_config, 'TIME_ZONE', 'Asia/Shanghai')
TIME_ZONE = tz.gettz(configured_tz)
if TIME_ZONE is None:
    logger.error(f"配置的时区 '{configured_tz}' 无效，将使用默认时区 'Asia/Shanghai'")
    TIME_ZONE = tz.gettz('Asia/Shanghai')


async def run_conversation_loop(conversation_instance: 'Conversation'):
    """
    核心的规划与行动循环 (PFC Loop)。
    之前是 Conversation 类中的 _plan_and_action_loop 方法。
    """
    logger.info(f"[私聊][{conversation_instance.private_name}] 进入 run_conversation_loop 循环。")

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
                configured_tz_loop = getattr(global_config, 'TIME_ZONE', 'Asia/Shanghai')
                TIME_ZONE = tz.gettz(configured_tz_loop)
                if TIME_ZONE is None:
                    logger.error(f"循环中: 配置的时区 '{configured_tz_loop}' 无效，将使用 'Asia/Shanghai'")
                    TIME_ZONE = tz.gettz('Asia/Shanghai')
            
            current_time_dt = datetime.datetime.now(TIME_ZONE)
            if conversation_instance.observation_info: 
                time_str = current_time_dt.strftime("%Y-%m-%d %H:%M:%S %Z%z") 
                conversation_instance.observation_info.current_time_str = time_str
                logger.debug(f"[私聊][{conversation_instance.private_name}] 更新 ObservationInfo 当前时间: {time_str}")
            else:
                logger.warning(f"[私聊][{conversation_instance.private_name}] ObservationInfo 未初始化，无法更新当前时间。")
        except Exception as time_update_err:
            logger.error(f"[私聊][{conversation_instance.private_name}] 更新 ObservationInfo 当前时间时出错: {time_update_err}")

        # 处理忽略状态
        if conversation_instance.ignore_until_timestamp and loop_iter_start_time < conversation_instance.ignore_until_timestamp:
            if conversation_instance.idle_conversation_starter and conversation_instance.idle_conversation_starter._running:
                conversation_instance.idle_conversation_starter.stop()
                logger.debug(f"[私聊][{conversation_instance.private_name}] 对话被暂时忽略，暂停空闲对话检测")
            sleep_duration = min(30, conversation_instance.ignore_until_timestamp - loop_iter_start_time)
            await asyncio.sleep(sleep_duration)
            continue 
        elif conversation_instance.ignore_until_timestamp and loop_iter_start_time >= conversation_instance.ignore_until_timestamp:
            logger.info(f"[私聊][{conversation_instance.private_name}] 忽略时间已到 {conversation_instance.stream_id}，准备结束对话。")
            conversation_instance.ignore_until_timestamp = None 
            await conversation_instance.stop() # 调用 Conversation 实例的 stop 方法
            continue 
        else:
            if conversation_instance.idle_conversation_starter and not conversation_instance.idle_conversation_starter._running:
                conversation_instance.idle_conversation_starter.start()
                logger.debug(f"[私聊][{conversation_instance.private_name}] 恢复空闲对话检测")

        # 核心规划与行动逻辑
        try:
            # 更新关系和情绪文本 (在每次循环开始时进行)
            if conversation_instance.conversation_info and conversation_instance._initialized: 
                # 更新关系
                if conversation_instance.conversation_info.person_id and conversation_instance.relationship_translator and conversation_instance.person_info_mng: 
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
                        conversation_instance.conversation_info.relationship_text = await conversation_instance.relationship_translator.translate_relationship_value_to_text(numeric_relationship_value)
                    except Exception as e_rel:
                        logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) 更新关系文本时出错: {e_rel}")
                        conversation_instance.conversation_info.relationship_text = "你们的关系是：普通。" 
                # 更新情绪
                if conversation_instance.mood_mng:
                    conversation_instance.conversation_info.current_emotion_text = conversation_instance.mood_mng.get_prompt() # type: ignore
            
            # 检查核心组件
            if not all([conversation_instance.action_planner, conversation_instance.observation_info, conversation_instance.conversation_info]):
                logger.error(f"[私聊][{conversation_instance.private_name}] 核心组件未初始化，无法继续规划循环。将等待5秒后重试...")
                await asyncio.sleep(5)
                continue

            # 规划
            planning_start_time = time.time()
            logger.debug(f"[私聊][{conversation_instance.private_name}] --- (Loop) 开始规划 ({planning_start_time:.2f}) ---")
            if conversation_instance.conversation_info:
                conversation_instance.conversation_info.other_new_messages_during_planning_count = 0

            action, reason = await conversation_instance.action_planner.plan(
                conversation_instance.observation_info,
                conversation_instance.conversation_info, 
                conversation_instance.conversation_info.last_successful_reply_action if conversation_instance.conversation_info else None, 
                use_reflect_prompt=force_reflect_and_act 
            )
            force_reflect_and_act = False
            logger.debug(
                f"[私聊][{conversation_instance.private_name}] (Loop) ActionPlanner.plan 完成，初步规划动作: {action}"
            )

            # 检查中断
            current_unprocessed_messages = getattr(conversation_instance.observation_info, "unprocessed_messages", [])
            new_messages_during_planning: List[Dict[str, Any]] = []
            other_new_messages_during_planning: List[Dict[str, Any]] = []

            for msg in current_unprocessed_messages:
                msg_time = msg.get("time")
                sender_id_info = msg.get("user_info", {})
                sender_id = str(sender_id_info.get("user_id")) if sender_id_info else None
                if msg_time and msg_time >= planning_start_time:
                    new_messages_during_planning.append(msg)
                    if sender_id != conversation_instance.bot_qq_str:
                        other_new_messages_during_planning.append(msg)
            
            new_msg_count = len(new_messages_during_planning)
            other_new_msg_count = len(other_new_messages_during_planning)
            
            if conversation_instance.conversation_info and other_new_msg_count > 0: 
                conversation_instance.conversation_info.current_instance_message_count += other_new_msg_count
                # 触发关系和情绪更新（如果需要）
                if conversation_instance.relationship_updater and conversation_instance.observation_info and conversation_instance.chat_observer:
                    await conversation_instance.relationship_updater.update_relationship_incremental(
                        conversation_info=conversation_instance.conversation_info,
                        observation_info=conversation_instance.observation_info,
                        chat_observer_for_history=conversation_instance.chat_observer
                    )
                if conversation_instance.emotion_updater and other_new_messages_during_planning and conversation_instance.observation_info and conversation_instance.chat_observer:
                    last_user_msg = other_new_messages_during_planning[-1]
                    last_user_msg_text = last_user_msg.get("processed_plain_text", "用户发了新消息")
                    sender_name_for_event = getattr(conversation_instance.observation_info, 'sender_name', '对方')
                    event_desc = f"用户【{sender_name_for_event}】发送了新消息: '{last_user_msg_text[:30]}...'"
                    await conversation_instance.emotion_updater.update_emotion_based_on_context(
                        conversation_info=conversation_instance.conversation_info,
                        observation_info=conversation_instance.observation_info,
                        chat_observer_for_history=conversation_instance.chat_observer,
                        event_description=event_desc
                    )

            should_interrupt: bool = False
            interrupt_reason: str = ""
            if action in ["wait", "listening"] and new_msg_count > 0:
                should_interrupt = True
                interrupt_reason = f"规划 {action} 期间收到 {new_msg_count} 条新消息"
            elif other_new_msg_count > 2: # Threshold for other actions
                should_interrupt = True
                interrupt_reason = f"规划 {action} 期间收到 {other_new_msg_count} 条来自他人的新消息"

            if should_interrupt:
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 中断 '{action}'，原因: {interrupt_reason}。重新规划...")
                cancel_record = { "action": action, "plan_reason": reason, "status": "cancelled_due_to_new_messages", "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "final_reason": interrupt_reason, }
                if conversation_instance.conversation_info:
                    if not hasattr(conversation_instance.conversation_info, "done_action") or conversation_instance.conversation_info.done_action is None: conversation_instance.conversation_info.done_action = []
                    conversation_instance.conversation_info.done_action.append(cancel_record)
                    conversation_instance.conversation_info.last_successful_reply_action = None
                conversation_instance.state = ConversationState.ANALYZING
                await asyncio.sleep(0.1)
                continue

            # 执行动作 (调用 actions 模块的函数)
            logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 未中断，调用 actions.handle_action 执行动作 '{action}'...")
            if conversation_instance.conversation_info:
                conversation_instance.conversation_info.other_new_messages_during_planning_count = other_new_msg_count
                
            await actions.handle_action(conversation_instance, action, reason, conversation_instance.observation_info, conversation_instance.conversation_info)
            logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) actions.handle_action 完成。")

            # 检查是否需要反思
            last_action_record = {}
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                last_action_record = conversation_instance.conversation_info.done_action[-1]
            if last_action_record.get("action") == "send_new_message" and last_action_record.get("status") == "done_no_reply":
                logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) 检测到需反思，设置标志。")
                force_reflect_and_act = True
            
            # 检查结束条件
            goal_ended: bool = False
            if conversation_instance.conversation_info and hasattr(conversation_instance.conversation_info, "goal_list") and conversation_instance.conversation_info.goal_list:
                last_goal_item = conversation_instance.conversation_info.goal_list[-1]
                current_goal = last_goal_item.get("goal") if isinstance(last_goal_item, dict) else (last_goal_item if isinstance(last_goal_item, str) else None)
                if current_goal == "结束对话": goal_ended = True

            last_action_record_for_end_check = {}
            if conversation_instance.conversation_info and conversation_instance.conversation_info.done_action:
                 last_action_record_for_end_check = conversation_instance.conversation_info.done_action[-1]
            action_ended: bool = ( last_action_record_for_end_check.get("action") in ["end_conversation", "say_goodbye"] and last_action_record_for_end_check.get("status") == "done" )

            if goal_ended or action_ended:
                logger.info( f"[私聊][{conversation_instance.private_name}] (Loop) 检测到结束条件，停止循环。" )
                await conversation_instance.stop() # 调用 Conversation 的 stop
                continue # 虽然会 break，但 continue 更明确

        except asyncio.CancelledError:
            logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 主循环任务被取消。")
            await conversation_instance.stop() # 调用 Conversation 的 stop
            break 
        except Exception as loop_err:
            logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 主循环出错: {loop_err}")
            logger.error(f"[私聊][{conversation_instance.private_name}] (Loop) {traceback.format_exc()}")
            conversation_instance.state = ConversationState.ERROR
            await asyncio.sleep(5)

        # 控制循环频率
        loop_duration = time.time() - loop_iter_start_time
        min_loop_interval = 0.1
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Loop) 循环迭代耗时: {loop_duration:.3f} 秒。")
        if loop_duration < min_loop_interval:
            await asyncio.sleep(min_loop_interval - loop_duration)

    logger.info(f"[私聊][{conversation_instance.private_name}] (Loop) PFC 循环已退出 for stream_id: {conversation_instance.stream_id}")