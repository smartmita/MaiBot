from src.common.logger import get_module_logger
from .chat_observer import ChatObserver
from .conversation_info import ConversationInfo
from src.config.config import global_config
from typing import Tuple
import time
import asyncio

logger = get_module_logger("pfc_waiter")

# --- 在这里设定你想要的超时时间（秒） ---
# 例如： 120 秒 = 2 分钟
DESIRED_TIMEOUT_SECONDS = 300


class Waiter:
    """等待处理类"""

    def __init__(self, stream_id: str, private_name: str):
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.name = global_config.bot.nickname
        self.private_name = private_name
        # self.wait_accumulated_time = 0 # 不再需要累加计时

    async def wait(self, conversation_info: ConversationInfo) -> bool:
        """等待用户新消息或超时
        Returns:
            Tuple[bool, float]: (是否超时, 等待的分钟数)
        """
        wait_start_time = time.time()
        logger.info(f"[私聊][{self.private_name}]进入常规等待状态 (超时: {DESIRED_TIMEOUT_SECONDS} 秒)...")
        wait_duration_minutes: float = 0.0 # <--- 初始化

        while True:
            if self.chat_observer.new_message_after(wait_start_time):
                logger.info(f"[私聊][{self.private_name}]等待结束，收到新消息")
                return False, 0.0  # <--- 返回 False 和 0.0 分钟

            elapsed_time = time.time() - wait_start_time
            if elapsed_time > DESIRED_TIMEOUT_SECONDS:
                wait_duration_minutes = elapsed_time / 60.0 # <--- 计算分钟数
                logger.info(f"[私聊][{self.private_name}]等待超过 {DESIRED_TIMEOUT_SECONDS} 秒 ({wait_duration_minutes:.1f} 分钟)...添加思考目标。")
                wait_goal = {
                    "goal": f"你等待了{wait_duration_minutes:.1f}分钟，注意可能在对方看来聊天已经结束，思考接下来要做什么",
                    "reasoning": "对方很久没有回复你的消息了",
                }
                # 确保 goal_list 存在
                if not hasattr(conversation_info, 'goal_list') or conversation_info.goal_list is None:
                    conversation_info.goal_list = []
                conversation_info.goal_list.append(wait_goal)
                logger.info(f"[私聊][{self.private_name}]添加目标: {wait_goal}")
                return True, wait_duration_minutes # <--- 返回 True 和等待的分钟数

            await asyncio.sleep(5)
            logger.debug(
                f"[私聊][{self.private_name}]等待中..."
            )  # 可以考虑把这个频繁日志注释掉，只在超时或收到消息时输出

    async def wait_listening(self, conversation_info: ConversationInfo) -> Tuple[bool, float]: # <--- 修改返回类型
        """倾听用户发言或超时
        Returns:
            Tuple[bool, float]: (是否超时, 等待的分钟数)
        """
        wait_start_time = time.time()
        logger.info(f"[私聊][{self.private_name}]进入倾听等待状态 (超时: {DESIRED_TIMEOUT_SECONDS} 秒)...")
        wait_duration_minutes: float = 0.0 # <--- 初始化

        while True:
            if self.chat_observer.new_message_after(wait_start_time):
                logger.info(f"[私聊][{self.private_name}]倾听等待结束，收到新消息")
                return False, 0.0 # <--- 返回 False 和 0.0 分钟

            elapsed_time = time.time() - wait_start_time
            if elapsed_time > DESIRED_TIMEOUT_SECONDS:
                wait_duration_minutes = elapsed_time / 60.0 # <--- 计算分钟数
                logger.info(f"[私聊][{self.private_name}]倾听等待超过 {DESIRED_TIMEOUT_SECONDS} 秒 ({wait_duration_minutes:.1f} 分钟)...添加思考目标。")
                wait_goal = {
                    "goal": f"你等待了{wait_duration_minutes:.1f}分钟，对方似乎话说一半突然消失了，可能忙去了？也可能忘记了回复？要问问吗？还是结束对话？或继续等待？思考接下来要做什么",
                    "reasoning": "对方话说一半消失了，很久没有回复",
                }
                if not hasattr(conversation_info, 'goal_list') or conversation_info.goal_list is None:
                    conversation_info.goal_list = []
                conversation_info.goal_list.append(wait_goal)
                logger.info(f"[私聊][{self.private_name}]添加目标: {wait_goal}")
                return True, wait_duration_minutes # <--- 返回 True 和等待的分钟数

            await asyncio.sleep(5)
            logger.debug(f"[私聊][{self.private_name}]倾听等待中...")  # 同上，可以考虑注释掉
