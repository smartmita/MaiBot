import traceback

from maim_message import UserInfo
from src.config.config import global_config
from src.common.logger_manager import get_logger
from ..chat.chat_stream import chat_manager
from typing import Optional, Dict, Any
from .pfc_manager import PFCManager
from src.plugins.chat.message import MessageRecv
from src.plugins.storage.storage import MessageStorage
from datetime import datetime


logger = get_logger("pfc_processor")


async def _handle_error(error: Exception, context: str, message: Optional[MessageRecv] = None) -> None:
    """统一的错误处理函数

    Args:
        error: 捕获到的异常
        context: 错误发生的上下文描述
        message: 可选的消息对象，用于记录相关消息内容
    """
    logger.error(f"{context}: {error}")
    logger.error(traceback.format_exc())
    if message and hasattr(message, "raw_message"):
        logger.error(f"相关消息原始内容: {message.raw_message}")


class PFCProcessor:
    """PFC 处理器，负责处理接收到的信息并计数"""

    def __init__(self):
        """初始化 PFC 处理器，创建消息存储实例"""
        self.storage = MessageStorage()
        self.pfc_manager = PFCManager.get_instance()

    async def process_message(self, message_data: Dict[str, Any]) -> None:
        """处理接收到的原始消息数据

        主要流程:
        1. 消息解析与初始化
        2. 过滤检查
        3. 消息存储
        4. 创建 PFC 流
        5. 日志记录

        Args:
            message_data: 原始消息字符串
        """
        message = None
        try:
            # 1. 消息解析与初始化
            message = MessageRecv(message_data)
            groupinfo = message.message_info.group_info
            userinfo = message.message_info.user_info
            messageinfo = message.message_info

            logger.trace(f"准备为{userinfo.user_id}创建/获取聊天流")
            chat = await chat_manager.get_or_create_stream(
                platform=messageinfo.platform,
                user_info=userinfo,
                group_info=groupinfo,
            )
            message.update_chat_stream(chat)

            # 2. 过滤检查
            # 处理消息
            await message.process()
            # 过滤词/正则表达式过滤
            if self._check_ban_words(message.processed_plain_text, userinfo) or self._check_ban_regex(
                message.raw_message, userinfo
            ):
                return

            # 3. 消息存储
            await self.storage.store_message(message, chat)
            logger.trace(f"存储成功: {message.processed_plain_text}")

            # 4. 创建 PFC 聊天流
            await self._create_pfc_chat(message)

            # 5. 日志记录
            # 将时间戳转换为datetime对象
            current_time = datetime.fromtimestamp(message.message_info.time).strftime("%H:%M:%S")
            logger.info(
                f"[{current_time}][私聊]{message.message_info.user_info.user_nickname}: {message.processed_plain_text}"
            )

        except Exception as e:
            await _handle_error(e, "消息处理失败", message)

    async def _create_pfc_chat(self, message: MessageRecv):
        try:
            chat_id = str(message.chat_stream.stream_id)
            private_name = str(message.message_info.user_info.user_nickname)

            if global_config.enable_pfc_chatting:
                await self.pfc_manager.get_or_create_conversation(chat_id, private_name)

        except Exception as e:
            logger.error(f"创建PFC聊天失败: {e}")

    @staticmethod
    def _check_ban_words(text: str, userinfo: UserInfo) -> bool:
        """检查消息中是否包含过滤词"""
        for word in global_config.ban_words:
            if word in text:
                logger.info(f"[私聊]{userinfo.user_nickname}:{text}")
                logger.info(f"[过滤词识别]消息中含有{word}，filtered")
                return True
        return False

    @staticmethod
    def _check_ban_regex(text: str, userinfo: UserInfo) -> bool:
        """检查消息是否匹配过滤正则表达式"""
        for pattern in global_config.ban_msgs_regex:
            if pattern.search(text):
                logger.info(f"[私聊]{userinfo.user_nickname}:{text}")
                logger.info(f"[正则表达式过滤]消息匹配到{pattern}，filtered")
                return True
        return False
