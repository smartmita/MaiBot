import time
import asyncio
import datetime
import traceback
from typing import Dict, Any, Optional, Set, List

from src.common.logger_manager import get_logger
from src.plugins.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from maim_message import UserInfo
from src.plugins.chat.chat_stream import chat_manager, ChatStream
from ..chat.message import Message # 假设 Message 类在这里
from ...config.config import global_config # 导入全局配置

from .pfc_types import ConversationState
from .pfc import GoalAnalyzer # 假设 GoalAnalyzer 在 pfc.py
from .chat_observer import ChatObserver
from .message_sender import DirectMessageSender
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo # 导入修改后的 ConversationInfo
from .reply_generator import ReplyGenerator
from .idle_conversation_starter import IdleConversationStarter
from .pfc_KnowledgeFetcher import KnowledgeFetcher # 假设 KnowledgeFetcher 在这里
from .waiter import Waiter

from rich.traceback import install
install(extra_lines=3)

logger = get_logger("pfc_conversation")

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
        self.chat_stream: Optional[ChatStream] = None

        # 初始化组件为 None
        self.action_planner: Optional[ActionPlanner] = None
        self.goal_analyzer: Optional[GoalAnalyzer] = None
        self.reply_generator: Optional[ReplyGenerator] = None
        self.knowledge_fetcher: Optional[KnowledgeFetcher] = None
        self.waiter: Optional[Waiter] = None
        self.direct_sender: Optional[DirectMessageSender] = None
        self.idle_conversation_starter: Optional[IdleConversationStarter] = None
        self.chat_observer: Optional[ChatObserver] = None
        self.observation_info: Optional[ObservationInfo] = None
        self.conversation_info: Optional[ConversationInfo] = None # 使用 ConversationInfo

        self._initializing = False
        self._initialized = False
        # 在初始化时获取机器人QQ号字符串，避免重复转换
        self.bot_qq_str = str(global_config.BOT_QQ) if global_config.BOT_QQ else None
        if not self.bot_qq_str:
            logger.error(f"[私聊][{self.private_name}] 严重错误：未能从配置中获取 BOT_QQ ID！PFC 可能无法正常工作。")


    async def _initialize(self):
        """异步初始化对话实例及其所有组件"""

        if self._initialized or self._initializing:
            logger.warning(f"[私聊][{self.private_name}] 尝试重复初始化或正在初始化中。")
            return
        self._initializing = True
        logger.info(f"[私聊][{self.private_name}] 开始初始化对话实例: {self.stream_id}")
        try:
            self.action_planner = ActionPlanner(self.stream_id, self.private_name)
            self.goal_analyzer = GoalAnalyzer(self.stream_id, self.private_name)
            self.reply_generator = ReplyGenerator(self.stream_id, self.private_name)
            self.knowledge_fetcher = KnowledgeFetcher(self.private_name)
            self.waiter = Waiter(self.stream_id, self.private_name)
            self.direct_sender = DirectMessageSender(self.private_name)
            self.chat_stream = chat_manager.get_stream(self.stream_id)
            if not self.chat_stream:
                raise ValueError(f"无法获取 stream_id {self.stream_id} 的 ChatStream")
            self.idle_conversation_starter = IdleConversationStarter(self.stream_id, self.private_name)
            self.chat_observer = ChatObserver.get_instance(self.stream_id, self.private_name)
            self.observation_info = ObservationInfo(self.private_name)
            if not self.observation_info.bot_id:
                logger.warning(f"[私聊][{self.private_name}] ObservationInfo 未能自动获取 bot_id，尝试手动设置。")
                self.observation_info.bot_id = self.bot_qq_str
            self.conversation_info = ConversationInfo()
            self.observation_info.bind_to_chat_observer(self.chat_observer)
            await self._load_initial_history()
            self.chat_observer.start()
            if self.idle_conversation_starter:
                self.idle_conversation_starter.start()
                logger.info(f"[私聊][{self.private_name}] 空闲对话检测器已启动")
            self._initialized = True
            self.should_continue = True
            self.state = ConversationState.ANALYZING
            logger.info(f"[私聊][{self.private_name}] 对话实例 {self.stream_id} 初始化完成。")
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}] 初始化对话实例失败: {e}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            self.should_continue = False
            await self.stop()
            raise
        finally:
            self._initializing = False


    async def _load_initial_history(self):
        """加载初始聊天记录"""

        if not self.observation_info: return
        try:
            logger.info(f"[私聊][{self.private_name}] 为 {self.stream_id} 加载初始聊天记录...")
            initial_messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=self.stream_id, timestamp=time.time(), limit=30,
            )
            if initial_messages:
                self.observation_info.chat_history = initial_messages
                self.observation_info.chat_history_count = len(initial_messages)
                last_msg = initial_messages[-1]
                self.observation_info.last_message_time = last_msg.get("time")
                self.observation_info.last_message_id = last_msg.get("message_id")
                last_user_info_dict = last_msg.get("user_info", {})
                if isinstance(last_user_info_dict, dict):
                    try:
                        last_user_info = UserInfo.from_dict(last_user_info_dict)
                        self.observation_info.last_message_sender = str(last_user_info.user_id) if last_user_info else None
                    except Exception as e:
                        logger.warning(f"[私聊][{self.private_name}] 解析最后一条消息的用户信息时出错: {e}")
                        self.observation_info.last_message_sender = None
                else: self.observation_info.last_message_sender = None
                self.observation_info.last_message_content = last_msg.get("processed_plain_text", "")
                history_slice_for_str = initial_messages[-20:]
                self.observation_info.chat_history_str = await build_readable_messages(
                    history_slice_for_str, replace_bot_name=True, merge_messages=False, timestamp_mode="relative", read_mark=0.0
                )
                if self.chat_observer: self.chat_observer.last_message_time = self.observation_info.last_message_time
                if self.idle_conversation_starter and self.observation_info.last_message_time:
                    await self.idle_conversation_starter.update_last_message_time(self.observation_info.last_message_time)
                logger.info(f"[私聊][{self.private_name}] 成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {self.observation_info.last_message_time}")
            else:
                logger.info(f"[私聊][{self.private_name}] 没有找到初始聊天记录。")
                self.observation_info.chat_history_str = "还没有聊天记录。"
        except Exception as load_err:
            logger.error(f"[私聊][{self.private_name}] 加载初始聊天记录时出错: {load_err}")
            if self.observation_info: self.observation_info.chat_history_str = "[加载聊天记录出错]"


    async def start(self):
        """开始对话流程"""

        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 对话实例未初始化，无法启动。")
            try:
                await self._initialize()
                if not self._initialized: return
            except Exception: return
        if not self.should_continue:
            logger.warning(f"[私聊][{self.private_name}] 对话实例已被标记为不应继续，无法启动。")
            return
        logger.info(f"[私聊][{self.private_name}] 对话系统启动，开始规划循环...")
        asyncio.create_task(self._plan_and_action_loop())


    async def stop(self):
        """停止对话实例并清理资源"""

        logger.info(f"[私聊][{self.private_name}] 正在停止对话实例: {self.stream_id}")
        self.should_continue = False
        if self.idle_conversation_starter: self.idle_conversation_starter.stop()
        if self.observation_info and self.chat_observer: self.observation_info.unbind_from_chat_observer()
        self._initialized = False
        logger.info(f"[私聊][{self.private_name}] 对话实例 {self.stream_id} 已停止。")


    async def _plan_and_action_loop(self):
        """思考步，PFC核心循环模块 - 实现精细化中断逻辑"""

        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 尝试在未初始化状态下运行规划循环。")
            return
        while self.should_continue:
            current_loop_start_time = time.time()
            # --- 忽略逻辑 ---
            if self.ignore_until_timestamp and current_loop_start_time < self.ignore_until_timestamp:
                if self.idle_conversation_starter and self.idle_conversation_starter._running:
                    self.idle_conversation_starter.stop(); logger.debug(f"[私聊][{self.private_name}] 对话被暂时忽略，暂停空闲对话检测")
                sleep_duration = min(30, self.ignore_until_timestamp - current_loop_start_time)
                await asyncio.sleep(sleep_duration)
                continue
            elif self.ignore_until_timestamp and current_loop_start_time >= self.ignore_until_timestamp:
                logger.info(f"[私聊][{self.private_name}] 忽略时间已到 {self.stream_id}，准备结束对话。")
                self.ignore_until_timestamp = None; await self.stop(); continue
            else:
                if self.idle_conversation_starter and not self.idle_conversation_starter._running:
                    self.idle_conversation_starter.start(); logger.debug(f"[私聊][{self.private_name}] 恢复空闲对话检测")
            # --- 核心规划与行动逻辑 ---
            try:
                if not all([self.action_planner, self.observation_info, self.conversation_info]):
                    logger.error(f"[私聊][{self.private_name}] 核心组件未初始化，无法继续规划循环。"); await asyncio.sleep(5); continue
                # --- 1. 记录规划开始时间 ---
                planning_start_time = time.time()
                logger.debug(f"[私聊][{self.private_name}] --- 开始新一轮规划 ({planning_start_time:.2f}) ---")
                self.conversation_info.other_new_messages_during_planning_count = 0
                # --- 2. 调用 Action Planner ---
                action, reason = await self.action_planner.plan(self.observation_info, self.conversation_info, self.conversation_info.last_successful_reply_action)
                planning_duration = time.time() - planning_start_time
                logger.debug(f"[私聊][{self.private_name}] 规划耗时: {planning_duration:.3f} 秒，初步规划动作: {action}")
                # --- 3. 检查规划期间的新消息 ---
                current_unprocessed_messages = getattr(self.observation_info, 'unprocessed_messages', [])
                new_messages_during_planning: List[Dict[str, Any]] = []
                other_new_messages_during_planning: List[Dict[str, Any]] = []
                for msg in current_unprocessed_messages:
                    msg_time = msg.get('time')
                    sender_id = msg.get("user_info", {}).get("user_id")
                    if msg_time and msg_time >= planning_start_time:
                        new_messages_during_planning.append(msg)
                        if sender_id != self.bot_qq_str: other_new_messages_during_planning.append(msg)
                new_msg_count = len(new_messages_during_planning); other_new_msg_count = len(other_new_messages_during_planning)
                logger.debug(f"[私聊][{self.private_name}] 规划期间收到新消息总数: {new_msg_count}, 来自他人: {other_new_msg_count}")
                # --- 4. 执行中断检查 ---
                should_interrupt = False; interrupt_reason = ""
                if action in ["wait", "listening"]:
                    if new_msg_count > 0: should_interrupt = True; interrupt_reason = f"规划 {action} 期间收到 {new_msg_count} 条新消息"; logger.info(f"[私聊][{self.private_name}] 中断 '{action}'，原因: {interrupt_reason}。")
                else:
                    interrupt_threshold = 2
                    if other_new_msg_count > interrupt_threshold: should_interrupt = True; interrupt_reason = f"规划 {action} 期间收到 {other_new_msg_count} 条来自他人的新消息 (阈值 >{interrupt_threshold})"; logger.info(f"[私聊][{self.private_name}] 中断 '{action}'，原因: {interrupt_reason}。")
                if should_interrupt:
                    logger.info(f"[私聊][{self.private_name}] 执行中断，重新规划...")
                    cancel_record = {"action": action, "plan_reason": reason, "status": "cancelled_due_to_new_messages", "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "final_reason": interrupt_reason}
                    if not hasattr(self.conversation_info, "done_action"): self.conversation_info.done_action = []
                    self.conversation_info.done_action.append(cancel_record)
                    self.conversation_info.last_successful_reply_action = None; self.state = ConversationState.ANALYZING; await asyncio.sleep(0.1); continue
                # --- 5. 如果未中断，存储状态并执行动作 ---
                logger.debug(f"[私聊][{self.private_name}] 未中断，继续执行动作 '{action}'")
                self.conversation_info.other_new_messages_during_planning_count = other_new_msg_count
                await self._handle_action(action, reason, self.observation_info, self.conversation_info)
                # --- 6. 检查是否需要结束对话 ---
                goal_ended = False
                if hasattr(self.conversation_info, "goal_list") and self.conversation_info.goal_list:
                    last_goal_item = self.conversation_info.goal_list[-1]; current_goal = None
                    if isinstance(last_goal_item, dict): current_goal = last_goal_item.get("goal")
                    elif isinstance(last_goal_item, str): current_goal = last_goal_item
                    if isinstance(current_goal, str) and current_goal == "结束对话": goal_ended = True
                last_action_record = self.conversation_info.done_action[-1] if self.conversation_info.done_action else {}
                action_ended = last_action_record.get("action") in ["end_conversation", "say_goodbye"] and last_action_record.get("status") == "done"
                if goal_ended or action_ended:
                    logger.info(f"[私聊][{self.private_name}] 检测到结束条件 (目标结束: {goal_ended}, 动作结束: {action_ended})，停止循环。"); await self.stop(); continue
            except asyncio.CancelledError: logger.info(f"[私聊][{self.private_name}] PFC 主循环被取消。"); await self.stop(); break
            except Exception as loop_err: logger.error(f"[私聊][{self.private_name}] PFC 主循环出错: {loop_err}\n{traceback.format_exc()}"); self.state = ConversationState.ERROR; await asyncio.sleep(5)
            # 控制循环频率
            loop_duration = time.time() - current_loop_start_time; min_loop_interval = 0.1
            if loop_duration < min_loop_interval: await asyncio.sleep(min_loop_interval - loop_duration)
        logger.info(f"[私聊][{self.private_name}] PFC 循环结束 for stream_id: {self.stream_id}")


    def _convert_to_message(self, msg_dict: Dict[str, Any]) -> Optional[Message]:
        """将消息字典转换为Message对象"""

        try:
            chat_stream_to_use = self.chat_stream or chat_manager.get_stream(self.stream_id)
            if not chat_stream_to_use: logger.error(f"[私聊][{self.private_name}] 无法确定 ChatStream for stream_id {self.stream_id}，无法转换消息。"); return None
            user_info_dict = msg_dict.get("user_info", {}); user_info: Optional[UserInfo] = None
            if isinstance(user_info_dict, dict):
                try: user_info = UserInfo.from_dict(user_info_dict)
                except Exception as e: logger.warning(f"[私聊][{self.private_name}] 从字典创建 UserInfo 时出错: {e}, dict: {user_info_dict}")
            if not user_info: logger.warning(f"[私聊][{self.private_name}] 消息缺少有效的 UserInfo，无法转换。 msg_id: {msg_dict.get('message_id')}"); return None
            return Message(message_id=msg_dict.get("message_id", f"gen_{time.time()}"), chat_stream=chat_stream_to_use, time=msg_dict.get("time", time.time()), user_info=user_info, processed_plain_text=msg_dict.get("processed_plain_text", ""), detailed_plain_text=msg_dict.get("detailed_plain_text", ""))
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 转换消息时出错: {e}\n{traceback.format_exc()}"); return None


    async def _handle_action(
        self,
        action: str,
        reason: str,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo
    ):
        """处理规划的行动 - 实现精细化后续状态设置"""
        if not self._initialized:
            logger.error(f"[私聊][{self.private_name}] 尝试在未初始化状态下处理动作 '{action}'。")
            return

        logger.info(f"[私聊][{self.private_name}] 开始处理动作: {action}, 原因: {reason}")
        action_start_time = time.time()

        # 记录action历史
        current_action_record = {
            "action": action, "plan_reason": reason, "status": "start",
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "final_reason": None,
        }
        if not hasattr(conversation_info, "done_action"): conversation_info.done_action = []
        conversation_info.done_action.append(current_action_record)
        action_index = len(conversation_info.done_action) - 1

        action_successful = False # 初始化动作成功状态
        final_status = "recall" # 默认失败状态
        final_reason = "动作未成功执行" # 默认失败原因

        try:
            # --- 根据不同的 action 执行 ---
            if action in ["direct_reply", "send_new_message", "say_goodbye"]:
                # --- 生成回复逻辑 ---
                self.state = ConversationState.GENERATING
                if not self.reply_generator: raise RuntimeError("ReplyGenerator 未初始化")
                generated_content = await self.reply_generator.generate(observation_info, conversation_info, action_type=action)
                logger.info(f"[私聊][{self.private_name}] 动作 '{action}': 生成内容: '{generated_content[:100]}...'")

                if not generated_content or generated_content.startswith("抱歉"):
                    logger.warning(f"[私聊][{self.private_name}] 动作 '{action}': 生成内容为空或为错误提示，取消发送。")
                    final_reason = "生成内容无效"
                    if action == "say_goodbye": final_status = "done"; self.should_continue = False; logger.info(f"[私聊][{self.private_name}] 告别语生成失败，仍按计划结束对话。")
                    else: final_status = "recall"; conversation_info.last_successful_reply_action = None
                else:
                    # --- 发送回复逻辑 ---
                    self.generated_reply = generated_content
                    timestamp_before_sending = time.time()
                    logger.debug(f"[私聊][{self.private_name}] 动作 '{action}': 记录发送前时间戳: {timestamp_before_sending:.2f}")
                    self.state = ConversationState.SENDING
                    send_success = await self._send_reply()
                    send_end_time = time.time()

                    if send_success:
                        action_successful = True # <--- 标记动作成功
                        # final_status 和 final_reason 在 finally 块中根据 action_successful 设置
                        logger.info(f"[私聊][{self.private_name}] 动作 '{action}': 成功发送回复.")
                        if self.idle_conversation_starter: await self.idle_conversation_starter.update_last_message_time(send_end_time)

                        # --- 清理已处理消息 ---
                        current_unprocessed_messages = getattr(observation_info, 'unprocessed_messages', [])
                        message_ids_to_clear: Set[str] = set()
                        for msg in current_unprocessed_messages:
                            msg_time = msg.get('time'); msg_id = msg.get('message_id'); sender_id = msg.get("user_info", {}).get("user_id")
                            if msg_id and msg_time and sender_id != self.bot_qq_str and msg_time < timestamp_before_sending: message_ids_to_clear.add(msg_id)
                        if message_ids_to_clear: logger.debug(f"[私聊][{self.private_name}] 准备清理 {len(message_ids_to_clear)} 条发送前(他人)消息: {message_ids_to_clear}"); await observation_info.clear_processed_messages(message_ids_to_clear)
                        else: logger.debug(f"[私聊][{self.private_name}] 没有需要清理的发送前(他人)消息。")

                        # --- 决定下一轮规划类型 ---
                        other_new_msg_count_during_planning = getattr(conversation_info, 'other_new_messages_during_planning_count', 0)
                        if other_new_msg_count_during_planning > 0:
                            logger.info(f"[私聊][{self.private_name}] 因规划期间收到 {other_new_msg_count_during_planning} 条他人新消息，下一轮强制使用【初始回复】逻辑。")
                            conversation_info.last_successful_reply_action = None
                        else:
                            logger.info(f"[私聊][{self.private_name}] 规划期间无他人新消息，下一轮【允许】使用追问逻辑 (基于 '{action}')。")
                            conversation_info.last_successful_reply_action = action

                        # 清除上次拒绝信息
                        conversation_info.last_reply_rejection_reason = None; conversation_info.last_rejected_reply_content = None
                        if action == "say_goodbye": self.should_continue = False; logger.info(f"[私聊][{self.private_name}] 成功发送告别语，即将停止对话实例。")
                    else:
                        # 发送失败
                        logger.error(f"[私聊][{self.private_name}] 动作 '{action}': 发送回复失败。")
                        final_status = "recall"; final_reason = "发送回复时失败" # 发送失败直接设置状态
                        conversation_info.last_successful_reply_action = None
                        if action == "say_goodbye": self.should_continue = True

            # --- 其他动作处理 ---
            elif action == "rethink_goal":
                self.state = ConversationState.RETHINKING
                if not self.goal_analyzer: raise RuntimeError("GoalAnalyzer 未初始化")
                await self.goal_analyzer.analyze_goal(conversation_info, observation_info)
                action_successful = True # <--- 标记动作成功
            elif action == "listening":
                self.state = ConversationState.LISTENING
                if not self.waiter: raise RuntimeError("Waiter 未初始化")
                logger.info(f"[私聊][{self.private_name}] 动作 'listening': 进入倾听状态...")
                await self.waiter.wait_listening(conversation_info)
                action_successful = True # <--- 标记动作成功
            elif action == "end_conversation":
                logger.info(f"[私聊][{self.private_name}] 动作 'end_conversation': 收到最终结束指令，停止对话...")
                action_successful = True # <--- 标记动作成功
                self.should_continue = False
            elif action == "block_and_ignore":
                logger.info(f"[私聊][{self.private_name}] 动作 'block_and_ignore': 不想再理你了...")
                ignore_duration_seconds = 10 * 60
                self.ignore_until_timestamp = time.time() + ignore_duration_seconds
                logger.info(f"[私聊][{self.private_name}] 将忽略此对话直到: {datetime.datetime.fromtimestamp(self.ignore_until_timestamp)}")
                self.state = ConversationState.IGNORED
                action_successful = True # <--- 标记动作成功
            elif action == "wait":
                self.state = ConversationState.WAITING
                if not self.waiter: raise RuntimeError("Waiter 未初始化")
                logger.info(f"[私聊][{self.private_name}] 动作 'wait': 进入等待状态...")
                timeout_occurred = await self.waiter.wait(self.conversation_info)
                action_successful = True # <--- 标记动作成功
                # wait 的 reason 在 finally 中设置
                logger.debug(f"[私聊][{self.private_name}] Wait 动作完成，无需在此清理消息。")
            else:
                logger.warning(f"[私聊][{self.private_name}] 未知的动作类型: {action}")
                final_status = "recall"; final_reason = f"未知的动作类型: {action}" # 未知动作直接失败

            # --- 重置非回复动作的追问状态 ---
            if action not in ["direct_reply", "send_new_message", "say_goodbye"]:
                 conversation_info.last_successful_reply_action = None
                 conversation_info.last_reply_rejection_reason = None
                 conversation_info.last_rejected_reply_content = None

        except asyncio.CancelledError:
             logger.warning(f"[私聊][{self.private_name}] 处理动作 '{action}' 时被取消。")
             final_status = "cancelled"; final_reason = "动作处理被取消"
             conversation_info.last_successful_reply_action = None
             raise
        except Exception as handle_err:
            logger.error(f"[私聊][{self.private_name}] 处理动作 '{action}' 时出错: {handle_err}")
            logger.error(f"[私聊][{self.private_name}] {traceback.format_exc()}")
            final_status = "error"; final_reason = f"处理动作时出错: {handle_err}"
            self.state = ConversationState.ERROR
            conversation_info.last_successful_reply_action = None

        finally:
            # --- 重置临时计数值 ---
            conversation_info.other_new_messages_during_planning_count = 0

            # --- 更新 Action History 状态 (优化) ---
            # 如果状态仍然是默认的 recall，但 action_successful 为 True，则更新为 done
            if final_status == "recall" and action_successful:
                final_status = "done"
                # 设置成功的 reason (可以根据动作类型细化)
                if action == "wait":
                     # 检查是否是因为超时结束的（需要 waiter 返回值，或者检查 goal_list）
                     timeout_occurred = any("分钟，" in g.get("goal","") for g in conversation_info.goal_list if isinstance(g, dict)) if conversation_info.goal_list else False
                     final_reason = "等待完成" + (" (超时)" if timeout_occurred else " (收到新消息或中断)")
                elif action == "listening":
                     final_reason = "进入倾听状态"
                elif action in ["rethink_goal", "end_conversation", "block_and_ignore"]:
                     final_reason = f"成功执行 {action}" # 通用成功原因
                else: # 默认为发送成功
                     final_reason = "成功发送"

            # 更新历史记录
            if conversation_info.done_action and action_index < len(conversation_info.done_action):
                 conversation_info.done_action[action_index].update(
                     {
                         "status": final_status,
                         "time_completed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "final_reason": final_reason,
                         "duration_ms": int((time.time() - action_start_time) * 1000)
                     }
                 )
                 logger.debug(f"[私聊][{self.private_name}] 动作 '{action}' 最终状态: {final_status}, 原因: {final_reason}")
            else:
                 logger.error(f"[私聊][{self.private_name}] 无法更新动作历史记录，索引 {action_index} 无效或列表为空。")


    async def _send_reply(self) -> bool:
        """发送生成的回复"""

        if not self.generated_reply: logger.warning(f"[私聊][{self.private_name}] 没有生成回复内容，无法发送。"); return False
        if not self.direct_sender: logger.error(f"[私聊][{self.private_name}] DirectMessageSender 未初始化，无法发送。"); return False
        if not self.chat_stream: logger.error(f"[私聊][{self.private_name}] ChatStream 未初始化，无法发送。"); return False
        try:
            reply_content = self.generated_reply
            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=reply_content, reply_to_message=None)
            self.state = ConversationState.ANALYZING
            return True
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}] 发送消息时失败: {str(e)}\n{traceback.format_exc()}")
            self.state = ConversationState.ERROR
            return False

    # _send_timeout_message 方法可以保持不变
    async def _send_timeout_message(self):
        """发送超时结束消息"""

        if not self.direct_sender or not self.chat_stream: logger.warning(f"[私聊][{self.private_name}] 发送器或聊天流未初始化，无法发送超时消息。"); return
        try:
            timeout_content = "我们好像很久没说话了，先这样吧~"
            await self.direct_sender.send_message(chat_stream=self.chat_stream, content=timeout_content, reply_to_message=None)
            logger.info(f"[私聊][{self.private_name}] 已发送超时结束消息。")
            await self.stop()
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 发送超时消息失败: {str(e)}")
