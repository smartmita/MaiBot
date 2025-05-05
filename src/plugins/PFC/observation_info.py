# -*- coding: utf-8 -*-
# File: observation_info.py
from typing import List, Optional, Dict, Any, Set
from maim_message import UserInfo
import time
from src.common.logger import get_module_logger
# 移除旧的 ChatObserver 导入，因为它现在通过类型提示和方法参数传入
# from .chat_observer import ChatObserver
from .chat_states import NotificationHandler, NotificationType, Notification
from src.plugins.utils.chat_message_builder import build_readable_messages
import traceback  # 导入 traceback 用于调试

# 确保 ChatObserver 类型可用，即使不直接导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .chat_observer import ChatObserver


logger = get_module_logger("observation_info")


class ObservationInfoHandler(NotificationHandler):
    """ObservationInfo的通知处理器"""

    def __init__(self, observation_info: "ObservationInfo", private_name: str):
        """初始化处理器

        Args:
            observation_info: 要更新的ObservationInfo实例
            private_name: 私聊对象的名称，用于日志记录
        """
        self.observation_info = observation_info
        # 将 private_name 存储在 handler 实例中
        self.private_name = private_name

    async def handle_notification(self, notification: Notification):  # 添加类型提示
        # 获取通知类型和数据
        notification_type = notification.type
        data = notification.data

        try:  # 添加错误处理块
            if notification_type == NotificationType.NEW_MESSAGE:
                # 处理新消息通知
                # logger.debug(f"[私聊][{self.private_name}]收到新消息通知data: {data}") # 可以在需要时取消注释
                message_id = data.get("message_id")
                processed_plain_text = data.get("processed_plain_text")
                detailed_plain_text = data.get("detailed_plain_text")
                user_info_dict = data.get("user_info")  # 先获取字典
                time_value = data.get("time")

                # 确保 user_info 是字典类型再创建 UserInfo 对象
                user_info = None
                if isinstance(user_info_dict, dict):
                    try:
                        user_info = UserInfo.from_dict(user_info_dict)
                    except Exception as e:
                        logger.error(
                            f"[私聊][{self.private_name}]从字典创建 UserInfo 时出错: {e}, 字典内容: {user_info_dict}"
                        )
                        # 可以选择在这里返回或记录错误，避免后续代码出错
                        return
                elif user_info_dict is not None:
                    logger.warning(
                        f"[私聊][{self.private_name}]收到的 user_info 不是预期的字典类型: {type(user_info_dict)}"
                    )
                    # 根据需要处理非字典情况，这里暂时返回
                    return

                message = {
                    "message_id": message_id,
                    "processed_plain_text": processed_plain_text,
                    "detailed_plain_text": detailed_plain_text,
                    "user_info": user_info_dict,  # 存储原始字典或 UserInfo 对象，取决于你的 update_from_message 如何处理
                    "time": time_value,
                }
                # 传递 UserInfo 对象（如果成功创建）或原始字典
                await self.observation_info.update_from_message(message, user_info)  # 修改：传递 user_info 对象

            elif notification_type == NotificationType.COLD_CHAT:
                # 处理冷场通知
                is_cold = data.get("is_cold", False)
                await self.observation_info.update_cold_chat_status(is_cold, time.time())  # 修改：改为 await 调用

            elif notification_type == NotificationType.ACTIVE_CHAT:
                # 处理活跃通知 (通常由 COLD_CHAT 的反向状态处理)
                is_active = data.get("is_active", False)
                self.observation_info.is_cold_chat = not is_active # Corrected variable name

            elif notification_type == NotificationType.BOT_SPEAKING:
                # 处理机器人说话通知 (按需实现)
                self.observation_info.is_typing = False
                self.observation_info.last_bot_speak_time = time.time()

            elif notification_type == NotificationType.USER_SPEAKING:
                # 处理用户说话通知
                self.observation_info.is_typing = False
                self.observation_info.last_user_speak_time = time.time()

            elif notification_type == NotificationType.MESSAGE_DELETED:
                # 处理消息删除通知
                message_id = data.get("message_id")
                # 从 unprocessed_messages 中移除被删除的消息
                original_count = len(self.observation_info.unprocessed_messages)
                self.observation_info.unprocessed_messages = [
                    msg for msg in self.observation_info.unprocessed_messages if msg.get("message_id") != message_id
                ]
                # --- [修改点 11] 更新 new_messages_count ---
                self.observation_info.new_messages_count = len(self.observation_info.unprocessed_messages)
                if self.observation_info.new_messages_count < original_count:
                    logger.info(f"[私聊][{self.private_name}]移除了未处理的消息 (ID: {message_id}), 当前未处理数: {self.observation_info.new_messages_count}")


            elif notification_type == NotificationType.USER_JOINED:
                # 处理用户加入通知 (如果适用私聊场景)
                user_id = data.get("user_id")
                if user_id:
                    self.observation_info.active_users.add(str(user_id))  # 确保是字符串

            elif notification_type == NotificationType.USER_LEFT:
                # 处理用户离开通知 (如果适用私聊场景)
                user_id = data.get("user_id")
                if user_id:
                    self.observation_info.active_users.discard(str(user_id))  # 确保是字符串

            elif notification_type == NotificationType.ERROR:
                # 处理错误通知
                error_msg = data.get("error", "未提供错误信息")
                logger.error(f"[私聊][{self.private_name}]收到错误通知: {error_msg}")

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]处理通知时发生错误: {e}")
            logger.error(traceback.format_exc())  # 打印详细堆栈信息


# @dataclass <-- 这个，不需要了（递黄瓜）
class ObservationInfo:
    """决策信息类，用于收集和管理来自chat_observer的通知信息 (手动实现 __init__)"""

    # 类型提示保留，可用于文档和静态分析
    private_name: str
    chat_history: List[Dict[str, Any]]
    chat_history_str: str
    unprocessed_messages: List[Dict[str, Any]]
    active_users: Set[str]
    last_bot_speak_time: Optional[float]
    last_user_speak_time: Optional[float]
    last_message_time: Optional[float]
    last_message_id: Optional[str]
    last_message_content: str
    last_message_sender: Optional[str]
    bot_id: Optional[str]
    chat_history_count: int
    new_messages_count: int
    cold_chat_start_time: Optional[float]
    cold_chat_duration: float
    is_typing: bool
    is_cold_chat: bool # Corrected variable name
    changed: bool
    chat_observer: Optional['ChatObserver'] # Use forward reference
    handler: Optional[ObservationInfoHandler]

    def __init__(self, private_name: str):
        """
        手动初始化 ObservationInfo 的所有实例变量。
        """

        # 接收的参数
        self.private_name: str = private_name

        # data_list
        self.chat_history: List[Dict[str, Any]] = []
        self.chat_history_str: str = ""
        self.unprocessed_messages: List[Dict[str, Any]] = []
        self.active_users: Set[str] = set()

        # data
        self.last_bot_speak_time: Optional[float] = None
        self.last_user_speak_time: Optional[float] = None
        self.last_message_time: Optional[float] = None
        self.last_message_id: Optional[str] = None
        self.last_message_content: str = ""
        self.last_message_sender: Optional[str] = None
        self.bot_id: Optional[str] = None # 需要在某个地方设置 bot_id，例如从 global_config 获取
        self.chat_history_count: int = 0
        self.new_messages_count: int = 0
        self.cold_chat_start_time: Optional[float] = None
        self.cold_chat_duration: float = 0.0

        # state
        self.is_typing: bool = False
        self.is_cold_chat: bool = False # Corrected variable name
        self.changed: bool = False

        # 关联对象
        self.chat_observer: Optional['ChatObserver'] = None # Use forward reference

        self.handler: ObservationInfoHandler = ObservationInfoHandler(self, self.private_name)

        # --- 初始化 bot_id ---
        from ...config.config import global_config # 移动到 __init__ 内部以避免循环导入问题
        self.bot_id = str(global_config.BOT_QQ) if global_config.BOT_QQ else None

    def bind_to_chat_observer(self, chat_observer: ChatObserver):
        """绑定到指定的chat_observer (保持不变)"""
        if self.chat_observer:
            logger.warning(f"[私聊][{self.private_name}]尝试重复绑定 ChatObserver")
            return

        self.chat_observer = chat_observer
        try:
            if not self.handler:
                logger.error(f"[私聊][{self.private_name}] 尝试绑定时 handler 未初始化！")
                self.chat_observer = None
                return

            self.chat_observer.notification_manager.register_handler(
                target="observation_info", notification_type=NotificationType.NEW_MESSAGE, handler=self.handler
            )
            self.chat_observer.notification_manager.register_handler(
                target="observation_info", notification_type=NotificationType.COLD_CHAT, handler=self.handler
            )
            # --- [修改点 12] 注册 MESSAGE_DELETED ---
            self.chat_observer.notification_manager.register_handler(
                 target="observation_info", notification_type=NotificationType.MESSAGE_DELETED, handler=self.handler
             )
            logger.info(f"[私聊][{self.private_name}]成功绑定到 ChatObserver")
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]绑定到 ChatObserver 时出错: {e}")
            self.chat_observer = None

    def unbind_from_chat_observer(self):
        """解除与chat_observer的绑定 (保持不变)"""
        if (
            self.chat_observer and hasattr(self.chat_observer, "notification_manager") and self.handler
        ):
            try:
                self.chat_observer.notification_manager.unregister_handler(
                    target="observation_info", notification_type=NotificationType.NEW_MESSAGE, handler=self.handler
                )
                self.chat_observer.notification_manager.unregister_handler(
                    target="observation_info", notification_type=NotificationType.COLD_CHAT, handler=self.handler
                )
                # --- [修改点 13] 注销 MESSAGE_DELETED ---
                self.chat_observer.notification_manager.unregister_handler(
                     target="observation_info", notification_type=NotificationType.MESSAGE_DELETED, handler=self.handler
                 )
                logger.info(f"[私聊][{self.private_name}]成功从 ChatObserver 解绑")
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]从 ChatObserver 解绑时出错: {e}")
            finally:
                self.chat_observer = None
        else:
            logger.warning(f"[私聊][{self.private_name}]尝试解绑时 ChatObserver 不存在、无效或 handler 未设置")

    async def update_from_message(self, message: Dict[str, Any], user_info: Optional[UserInfo]):
        """从消息更新信息 (保持不变)"""
        message_time = message.get("time")
        message_id = message.get("message_id")
        processed_text = message.get("processed_plain_text", "")

        if message_time and message_time > (self.last_message_time or 0):
            self.last_message_time = message_time
            self.last_message_id = message_id
            self.last_message_content = processed_text
            self.is_cold_chat = False
            self.cold_chat_start_time = None
            self.cold_chat_duration = 0.0

            if user_info:
                sender_id = str(user_info.user_id)
                self.last_message_sender = sender_id
                if sender_id == self.bot_id:
                    self.last_bot_speak_time = message_time
                else:
                    self.last_user_speak_time = message_time
                    self.active_users.add(sender_id)
            else:
                logger.warning(
                    f"[私聊][{self.private_name}]处理消息更新时缺少有效的 UserInfo 对象, message_id: {message_id}"
                )
                self.last_message_sender = None

            # --- [修改点 14] 添加到未处理列表，并更新计数 ---
            # 检查消息是否已存在于未处理列表中，避免重复添加
            if not any(msg.get("message_id") == message_id for msg in self.unprocessed_messages):
                 self.unprocessed_messages.append(message)
                 self.new_messages_count = len(self.unprocessed_messages)
                 logger.debug(f"[私聊][{self.private_name}]添加新未处理消息 ID: {message_id}, 当前未处理数: {self.new_messages_count}")
                 self.update_changed()
            else:
                 logger.warning(f"[私聊][{self.private_name}]尝试重复添加未处理消息 ID: {message_id}")

        else:
            pass


    def update_changed(self):
        """标记状态已改变，并重置标记 (保持不变)"""
        self.changed = True

    async def update_cold_chat_status(self, is_cold: bool, current_time: float):
        """更新冷场状态 (保持不变)"""
        if is_cold != self.is_cold_chat:
            self.is_cold_chat = is_cold
            if is_cold:
                self.cold_chat_start_time = (
                    self.last_message_time or current_time
                )
                logger.info(f"[私聊][{self.private_name}]进入冷场状态，开始时间: {self.cold_chat_start_time}")
            else:
                if self.cold_chat_start_time:
                    self.cold_chat_duration = current_time - self.cold_chat_start_time
                    logger.info(f"[私聊][{self.private_name}]结束冷场状态，持续时间: {self.cold_chat_duration:.2f} 秒")
                self.cold_chat_start_time = None
            self.update_changed()

        if self.is_cold_chat and self.cold_chat_start_time:
            self.cold_chat_duration = current_time - self.cold_chat_start_time

    def get_active_duration(self) -> float:
        """获取当前活跃时长 (保持不变)"""
        if not self.last_message_time:
            return 0.0
        return time.time() - self.last_message_time

    def get_user_response_time(self) -> Optional[float]:
        """获取用户最后响应时间 (保持不变)"""
        if not self.last_user_speak_time:
            return None
        return time.time() - self.last_user_speak_time

    def get_bot_response_time(self) -> Optional[float]:
        """获取机器人最后响应时间 (保持不变)"""
        if not self.last_bot_speak_time:
            return None
        return time.time() - self.last_bot_speak_time

    # --- [修改点 15] 重命名并修改 clear_unprocessed_messages ---
    # async def clear_unprocessed_messages(self): <-- 旧方法注释掉或删除
    async def clear_processed_messages(self, message_ids_to_clear: Set[str]):
        """将指定ID的未处理消息移入历史记录，并更新相关状态"""
        if not message_ids_to_clear:
            logger.debug(f"[私聊][{self.private_name}]没有需要清理的消息 ID。")
            return

        messages_to_move = []
        remaining_messages = []
        cleared_count = 0

        # 分离要清理和要保留的消息
        for msg in self.unprocessed_messages:
            if msg.get("message_id") in message_ids_to_clear:
                messages_to_move.append(msg)
                cleared_count += 1
            else:
                remaining_messages.append(msg)

        if not messages_to_move:
            logger.debug(f"[私聊][{self.private_name}]未找到与 ID 列表匹配的未处理消息进行清理。")
            return

        logger.debug(f"[私聊][{self.private_name}]准备清理 {cleared_count} 条已处理消息...")

        # 将要移动的消息添加到历史记录
        max_history_len = 100
        self.chat_history.extend(messages_to_move)
        if len(self.chat_history) > max_history_len:
            self.chat_history = self.chat_history[-max_history_len:]

        # 更新历史记录字符串 (仅使用最近一部分生成)
        history_slice_for_str = self.chat_history[-20:] # 例如最近20条
        try:
            self.chat_history_str = await build_readable_messages(
                history_slice_for_str,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]构建聊天记录字符串时出错: {e}")
            self.chat_history_str = "[构建聊天记录出错]"

        # 更新未处理消息列表和计数
        self.unprocessed_messages = remaining_messages
        self.new_messages_count = len(self.unprocessed_messages)
        self.chat_history_count = len(self.chat_history)

        logger.info(f"[私聊][{self.private_name}]已清理 {cleared_count} 条消息，剩余未处理 {self.new_messages_count} 条，当前历史记录 {self.chat_history_count} 条。")

        self.update_changed() # 状态改变