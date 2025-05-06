import asyncio
import time
import random
import traceback
from typing import List, TYPE_CHECKING, Optional, Dict, Any

from src.common.logger import get_module_logger
from ..models.utils_model import LLMRequest
from ...config.config import global_config
from .chat_observer import ChatObserver
from .message_sender import DirectMessageSender
from ..chat.chat_stream import ChatStream
from src.individuality.individuality import Individuality
from src.plugins.utils.chat_message_builder import build_readable_messages

if TYPE_CHECKING:
    from ..chat.message import Message
    # 仍然需要类型提示
    from .conversation import Conversation
    from .pfc_manager import PFCManager # 仅用于类型提示

logger = get_module_logger("idle_conversation")

class IdleConversationStarter:
    """长时间无对话主动发起对话的组件"""

    def __init__(self, stream_id: str, private_name: str):
        """初始化空闲对话启动器"""
        self.stream_id: str = stream_id
        self.private_name: str = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.message_sender = DirectMessageSender(private_name)
        self._lock: asyncio.Lock = asyncio.Lock()
        self.llm = LLMRequest(
            model=global_config.llm_normal, temperature=0.8, max_tokens=500, request_type="idle_conversation_starter"
        )
        self.personality_info: str = Individuality.get_instance().get_prompt(x_person=2, level=3)
        self.name: str = global_config.BOT_NICKNAME
        idle_config = getattr(global_config, 'idle_conversation', {})
        self.enabled: bool = idle_config.get('enable_idle_conversation', True)
        self.idle_check_interval: int = idle_config.get('idle_check_interval', 10)
        self.min_idle_time: int = idle_config.get('min_idle_time', 60)
        self.max_idle_time: int = idle_config.get('max_idle_time', 120)
        self.last_message_time: float = time.time()
        self.actual_idle_threshold: int = self._get_new_threshold()
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    def _get_new_threshold(self) -> int:
        """计算一个新的随机空闲阈值"""
        try:
            min_t = max(10, self.min_idle_time); max_t = max(min_t, self.max_idle_time)
            return random.randint(min_t, max_t)
        except ValueError:
            logger.warning(f"[私聊][{self.private_name}] idle_time 配置无效 ({self.min_idle_time}, {self.max_idle_time})，使用默认值 60-120。")
            return random.randint(60, 120)

    def start(self) -> None:
        """启动空闲对话检测"""
        if not self.enabled: logger.info(f"[私聊][{self.private_name}] 主动发起对话功能已禁用"); return
        if self._running: logger.debug(f"[私聊][{self.private_name}] 主动发起对话功能已在运行中"); return
        self._running = True
        self._task = asyncio.create_task(self._check_idle_loop())
        logger.info(f"[私聊][{self.private_name}] 启动空闲对话检测，当前阈值: {self.actual_idle_threshold}秒")

    def stop(self) -> None:
        """停止空闲对话检测"""
        if not self._running: return
        self._running = False
        if self._task: self._task.cancel(); self._task = None
        logger.info(f"[私聊][{self.private_name}] 停止空闲对话检测")

    async def update_last_message_time(self, message_time: Optional[float] = None) -> None:
        """更新最后一条消息的时间，并重置阈值"""
        async with self._lock:
            new_time = message_time or time.time()
            if new_time > self.last_message_time:
                self.last_message_time = new_time
                self.actual_idle_threshold = self._get_new_threshold()
                logger.debug(f"[私聊][{self.private_name}] 更新最后消息时间: {self.last_message_time:.2f}，新阈值: {self.actual_idle_threshold}秒")

    def reload_config(self) -> None:
        """重新加载配置"""
        try:
            idle_config = getattr(global_config, 'idle_conversation', {})
            self.enabled = idle_config.get('enable_idle_conversation', True)
            self.idle_check_interval = idle_config.get('idle_check_interval', 10)
            self.min_idle_time = idle_config.get('min_idle_time', 60)
            self.max_idle_time = idle_config.get('max_idle_time', 120)
            logger.debug(f"[私聊][{self.private_name}] 重新加载主动对话配置: 启用={self.enabled}, 检查间隔={self.idle_check_interval}秒, 阈值范围=[{self.min_idle_time}, {self.max_idle_time}]秒")
            async def update_threshold_async():
                async with self._lock:
                    self.actual_idle_threshold = self._get_new_threshold()
                    logger.debug(f"[私聊][{self.private_name}] 配置重载后更新空闲检测阈值为: {self.actual_idle_threshold}秒")
            asyncio.create_task(update_threshold_async())
            if self.enabled and not self._running: self.start()
            elif not self.enabled and self._running: self.stop()
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 重新加载配置时出错: {str(e)}")

    async def _check_idle_loop(self) -> None:
        """检查空闲状态的循环"""
        try:
            config_reload_counter = 0; config_reload_interval = 60
            while self._running:
                await asyncio.sleep(self.idle_check_interval)
                config_reload_counter = (config_reload_counter + 1) % config_reload_interval
                if config_reload_counter == 0: self.reload_config()
                if not self.enabled: continue
                current_time = time.time()
                async with self._lock: last_msg_time = self.last_message_time; threshold = self.actual_idle_threshold
                idle_time = current_time - last_msg_time
                if idle_time >= threshold:
                    logger.info(f"[私聊][{self.private_name}] 检测到长时间({idle_time:.0f}秒 >= {threshold}秒)无对话，尝试主动发起聊天")
                    try:
                        await self._initiate_conversation()
                        await self.update_last_message_time() # 主动发起后立即更新时间
                    except Exception as initiate_err:
                        logger.error(f"[私聊][{self.private_name}] 主动发起对话过程中出错: {initiate_err}\n{traceback.format_exc()}")
                        await self.update_last_message_time() # 即使出错也更新时间
        except asyncio.CancelledError: logger.debug(f"[私聊][{self.private_name}] 空闲对话检测任务被取消")
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 空闲对话检测循环发生未预期错误: {str(e)}\n{traceback.format_exc()}")

    async def _initiate_conversation(self) -> None:
        """生成并发送主动对话内容 (修复循环导入)"""
        # --- 将 PFCManager 的导入移到这里 ---
        from .pfc_manager import PFCManager
        # 仅在需要时导入类型提示，避免运行时错误
        if TYPE_CHECKING:
            from .conversation import Conversation

        pfc_manager: 'PFCManager' = PFCManager.get_instance() # 获取管理器实例
        conversation: Optional['Conversation'] = None
        chat_stream: Optional[ChatStream] = None

        try:
            # --- 1. 获取或创建 Conversation 实例 ---
            logger.debug(f"[私聊][{self.private_name}] 尝试获取或创建 Conversation 实例...")
            conversation = await pfc_manager.get_or_create_conversation(self.stream_id, self.private_name)
            if not conversation or not conversation._initialized:
                logger.error(f"[私聊][{self.private_name}] 无法获取或创建有效的 Conversation 实例，取消主动发起对话。")
                return
            chat_stream = conversation.chat_stream
            if not chat_stream:
                logger.error(f"[私聊][{self.private_name}] 无法从 Conversation 实例获取 ChatStream，取消主动发起对话。")
                return

            # --- 2. 获取聊天历史用于生成 ---
            messages: List[Dict[str, Any]] = []
            chat_history_text = "最近没有聊天记录。"
            if conversation.observation_info:
                messages = conversation.observation_info.chat_history[-12:]
                if messages:
                    chat_history_text = await build_readable_messages(messages, replace_bot_name=True, merge_messages=False, timestamp_mode="relative", read_mark=0.0)
            else: logger.warning(f"[私聊][{self.private_name}] Conversation 实例缺少 ObservationInfo，无法获取准确聊天记录。")

            # --- 3. 构建 Prompt 并生成内容 ---
            prompt = f"""{self.personality_info}。你的名字是{self.name}。
你正在与用户 {self.private_name} 进行QQ私聊, 但已经有一段时间没有对话了。
你想要主动发起一个友好的对话，可以说说自己在做的事情或者询问对方在做什么。
请基于以下之前的对话历史，生成一条自然、友好、符合你个性的主动对话消息。
这条消息应该能够引起用户的兴趣，重新开始对话。

最近的对话历史（可能已经过去了很久）：
{chat_history_text}

请直接输出一条消息，不要有任何额外的解释或引导文字。消息要简短自然，就像是在日常聊天中的开场白。
消息内容尽量简短,不要超过30个字。可以适当添加符合你人设的语气词或表情符号（如果合适）。
"""
            # logger.debug(f"[私聊][{self.private_name}] 发送到 LLM 的主动对话 Prompt: \n{prompt}") # 日志可能过长
            content = ""
            try:
                llm_response, _ = await asyncio.wait_for(self.llm.generate_response_async(prompt), timeout=30)
                content = llm_response.strip().strip('"\'')
                logger.debug(f"[私聊][{self.private_name}] LLM 生成的主动对话内容: {content}")
            except asyncio.TimeoutError: logger.error(f"[私聊][{self.private_name}] 生成主动对话内容超时"); return
            except Exception as llm_err: logger.error(f"[私聊][{self.private_name}] 生成主动对话内容失败: {str(llm_err)}"); return
            if not content: logger.error(f"[私聊][{self.private_name}] 生成的主动对话内容为空"); return

            # --- 4. 发送消息 ---
            logger.info(f"[私聊][{self.private_name}] 准备发送主动对话消息: {content}")
            await self.message_sender.send_message(chat_stream=chat_stream, content=content, reply_to_message=None)

            # --- 5. 发送成功后的处理 ---
            logger.success(f"[私聊][{self.private_name}] 成功主动发起对话: {content}")
            # 更新时间戳的操作移至外层 _check_idle_loop
            # 触发 ChatObserver 更新
            if conversation.chat_observer:
                logger.debug(f"[私聊][{self.private_name}] 触发 ChatObserver 更新...")
                conversation.chat_observer.trigger_update()
            else: logger.warning(f"[私聊][{self.private_name}] Conversation 实例缺少 ChatObserver，无法触发更新。")

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}] 主动发起对话的整体流程出错: {str(e)}\n{traceback.format_exc()}")