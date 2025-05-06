from typing import List, Tuple, TYPE_CHECKING, Optional, Union
import asyncio
import time
import random
from src.common.logger import get_module_logger
from ..models.utils_model import LLMRequest
from ...config.config import global_config
from .chat_observer import ChatObserver
from .message_sender import DirectMessageSender
from ..chat.chat_stream import ChatStream
from maim_message import UserInfo
from src.individuality.individuality import Individuality
from src.plugins.utils.chat_message_builder import build_readable_messages

if TYPE_CHECKING:
    from ..chat.message import Message
    from .conversation import Conversation

logger = get_module_logger("idle_conversation")

class IdleConversationStarter:
    """长时间无对话主动发起对话的组件
    
    该组件会在一段时间没有对话后，自动生成一条消息发送给用户，以保持对话的活跃度。
    时间阈值会在配置的最小和最大值之间随机选择，每次发送消息后都会重置。
    """
    
    def __init__(self, stream_id: str, private_name: str):
        """初始化空闲对话启动器
        
        Args:
            stream_id: 聊天流ID
            private_name: 私聊用户名称
        """
        self.stream_id: str = stream_id
        self.private_name: str = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.message_sender = DirectMessageSender(private_name)
        
        # 添加异步锁，保护对共享变量的访问
        self._lock: asyncio.Lock = asyncio.Lock()
        
        # LLM请求对象，用于生成主动对话内容
        self.llm = LLMRequest(
            model=global_config.llm_normal, temperature=0.8, max_tokens=500, request_type="idle_conversation_starter"
        )
        
        # 个性化信息
        self.personality_info: str = Individuality.get_instance().get_prompt(x_person=2, level=3)
        self.name: str = global_config.BOT_NICKNAME
        self.nick_name: List[str] = global_config.BOT_ALIAS_NAMES
        
        # 从配置文件读取配置参数，或使用默认值
        self.enabled: bool = getattr(global_config, 'idle_conversation', {}).get('enable_idle_conversation', True)
        self.idle_check_interval: int = getattr(global_config, 'idle_conversation', {}).get('idle_check_interval', 10)
        self.min_idle_time: int = getattr(global_config, 'idle_conversation', {}).get('min_idle_time', 60)
        self.max_idle_time: int = getattr(global_config, 'idle_conversation', {}).get('max_idle_time', 120)
        
        # 计算实际触发阈值（在min和max之间随机）
        self.actual_idle_threshold: int = random.randint(self.min_idle_time, self.max_idle_time)
        
        # 工作状态
        self.last_message_time: float = time.time()
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
    
    def start(self) -> None:
        """启动空闲对话检测
        
        如果功能被禁用或已经在运行，则不会启动。
        """
        # 如果功能被禁用，则不启动
        if not self.enabled:
            logger.info(f"[私聊][{self.private_name}]主动发起对话功能已禁用")
            return
            
        if self._running:
            logger.debug(f"[私聊][{self.private_name}]主动发起对话功能已在运行中")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._check_idle_loop())
        logger.info(f"[私聊][{self.private_name}]启动空闲对话检测，阈值设置为{self.actual_idle_threshold}秒")
    
    def stop(self) -> None:
        """停止空闲对话检测
        
        取消当前运行的任务并重置状态。
        """
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info(f"[私聊][{self.private_name}]停止空闲对话检测")
    
    async def update_last_message_time(self, message_time: Optional[float] = None) -> None:
        """更新最后一条消息的时间
        
        Args:
            message_time: 消息时间戳，如果为None则使用当前时间
        """
        async with self._lock:
            self.last_message_time = message_time or time.time()
            # 重新随机化下一次触发的时间阈值
            self.actual_idle_threshold = random.randint(self.min_idle_time, self.max_idle_time)
            logger.debug(f"[私聊][{self.private_name}]更新最后消息时间: {self.last_message_time}，新阈值: {self.actual_idle_threshold}秒")
    
    def reload_config(self) -> None:
        """重新加载配置
        
        从配置文件重新读取所有参数，以便动态调整空闲对话检测的行为。
        """
        try:
            # 从配置文件重新读取参数
            self.enabled = getattr(global_config, 'idle_conversation', {}).get('enable_idle_conversation', True)
            self.idle_check_interval = getattr(global_config, 'idle_conversation', {}).get('idle_check_interval', 10)
            self.min_idle_time = getattr(global_config, 'idle_conversation', {}).get('min_idle_time', 7200)
            self.max_idle_time = getattr(global_config, 'idle_conversation', {}).get('max_idle_time', 18000)
            
            logger.debug(f"[私聊][{self.private_name}]重新加载主动对话配置: 启用={self.enabled}, 检查间隔={self.idle_check_interval}秒, 最短间隔={self.min_idle_time}秒, 最长间隔={self.max_idle_time}秒")
            
            # 重新计算实际阈值
            async def update_threshold():
                async with self._lock:
                    self.actual_idle_threshold = random.randint(self.min_idle_time, self.max_idle_time)
                    logger.debug(f"[私聊][{self.private_name}]更新空闲检测阈值为: {self.actual_idle_threshold}秒")
            
            # 创建一个任务来异步更新阈值
            asyncio.create_task(update_threshold())
            
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]重新加载配置时出错: {str(e)}")
    
    async def _check_idle_loop(self) -> None:
        """检查空闲状态的循环
        
        定期检查是否长时间无对话，如果达到阈值则尝试主动发起对话。
        """
        try:
            config_reload_counter = 0
            config_reload_interval = 100  # 每100次检查重新加载一次配置
            
            while self._running:
                # 定期重新加载配置
                config_reload_counter += 1
                if config_reload_counter >= config_reload_interval:
                    self.reload_config()
                    config_reload_counter = 0
                
                # 检查是否启用了主动对话功能
                if not self.enabled:
                    # 如果禁用了功能，就等待一段时间后再次检查配置
                    await asyncio.sleep(self.idle_check_interval)
                    continue
                
                # 使用锁保护对共享变量的读取
                current_time = time.time()
                async with self._lock:
                    idle_time = current_time - self.last_message_time
                    threshold = self.actual_idle_threshold
                
                if idle_time >= threshold:
                    logger.info(f"[私聊][{self.private_name}]检测到长时间({idle_time:.0f}秒)无对话，尝试主动发起聊天")
                    await self._initiate_conversation()
                    # 更新时间，避免连续触发
                    await self.update_last_message_time()
                
                # 等待下一次检查
                await asyncio.sleep(self.idle_check_interval)
        
        except asyncio.CancelledError:
            logger.debug(f"[私聊][{self.private_name}]空闲对话检测任务被取消")
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]空闲对话检测出错: {str(e)}")
            # 尝试重新启动检测循环
            if self._running:
                logger.info(f"[私聊][{self.private_name}]尝试重新启动空闲对话检测")
                self._task = asyncio.create_task(self._check_idle_loop())
    
    async def _initiate_conversation(self) -> None:
        """生成并发送主动对话内容
        
        获取聊天历史记录，使用LLM生成合适的开场白，然后发送消息。
        """
        try:
            # 获取聊天历史记录，用于生成更合适的开场白
            messages = self.chat_observer.get_cached_messages(limit=12)  # 获取最近12条消息
            chat_history_text = await build_readable_messages(
                messages,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
            
            # 构建提示词
            prompt = f"""{self.personality_info}。你的名字是{self.name}。
            你正在与用户{self.private_name}进行QQ私聊,
            但已经有一段时间没有对话了。
            你想要主动发起一个友好的对话，可以说说自己在做的事情或者询问对方在做什么。
            请基于以下之前的对话历史，生成一条自然、友好、符合你个性的主动对话消息。
            这条消息应该能够引起用户的兴趣，重新开始对话。
            最近的对话历史（可能已经过去了很久）：
            {chat_history_text}
            请直接输出一条消息，不要有任何额外的解释或引导文字。消息要简短自然，就像是在日常聊天中的开场白。
            消息内容尽量简短,不要超过20个字,不要添加任何表情符号。
            """
            
            # 尝试生成回复，添加超时处理
            try:
                content, _ = await asyncio.wait_for(
                    self.llm.generate_response_async(prompt),
                    timeout=30  # 30秒超时
                )
            except asyncio.TimeoutError:
                logger.error(f"[私聊][{self.private_name}]生成主动对话内容超时")
                return
            except Exception as llm_err:
                logger.error(f"[私聊][{self.private_name}]生成主动对话内容失败: {str(llm_err)}")
                return
            
            # 清理结果
            content = content.strip()
            content = content.strip('"\'')
            
            if not content:
                logger.error(f"[私聊][{self.private_name}]生成的主动对话内容为空")
                return
                
            # 统一错误处理，从这里开始所有操作都在同一个try-except块中
            logger.debug(f"[私聊][{self.private_name}]成功生成主动对话内容: {content}，准备发送")
            
            from .pfc_manager import PFCManager
            from src.plugins.chat.chat_stream import chat_manager
            
            # 获取当前实例
            pfc_manager = PFCManager.get_instance()
            
            # 结束当前对话实例（如果存在）
            current_conversation = await pfc_manager.get_conversation(self.stream_id)
            if current_conversation:
                logger.info(f"[私聊][{self.private_name}]结束当前对话实例，准备创建新实例")
                try:
                    await current_conversation.stop()
                    await pfc_manager.remove_conversation(self.stream_id)
                except Exception as e:
                    logger.warning(f"[私聊][{self.private_name}]结束当前对话实例时出错: {str(e)}，继续创建新实例")
            
            # 创建新的对话实例
            logger.info(f"[私聊][{self.private_name}]创建新的对话实例以发送主动消息")
            new_conversation = None
            try:
                new_conversation = await pfc_manager.get_or_create_conversation(self.stream_id, self.private_name)
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]创建新对话实例失败: {str(e)}")
                return
            
            # 确保新对话实例已初始化完成
            chat_stream = await self._get_chat_stream(new_conversation)
            if not chat_stream:
                logger.error(f"[私聊][{self.private_name}]无法获取有效的聊天流，取消发送主动消息")
                return
            
            # 发送消息
            try:
                await self.message_sender.send_message(
                    chat_stream=chat_stream,
                    content=content,
                    reply_to_message=None
                )
                
                # 更新空闲会话启动器的最后消息时间
                await self.update_last_message_time()
                
                # 如果新对话实例有一个聊天观察者，请触发更新
                if new_conversation and hasattr(new_conversation, 'chat_observer'):
                    logger.info(f"[私聊][{self.private_name}]触发聊天观察者更新")
                    try:
                        new_conversation.chat_observer.trigger_update()
                    except Exception as e:
                        logger.warning(f"[私聊][{self.private_name}]触发聊天观察者更新失败: {str(e)}")
                
                logger.success(f"[私聊][{self.private_name}]成功主动发起对话: {content}")
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]发送主动对话消息失败: {str(e)}")
        
        except Exception as e:
            # 顶级异常处理，确保任何未捕获的异常都不会导致整个进程崩溃
            logger.error(f"[私聊][{self.private_name}]主动发起对话过程中发生未预期的错误: {str(e)}")
    
    async def _get_chat_stream(self, conversation: Optional['Conversation'] = None) -> Optional[ChatStream]:
        """获取可用的聊天流
        
        尝试多种方式获取聊天流：
        1. 从传入的对话实例中获取
        2. 从全局聊天管理器中获取
        3. 创建一个新的聊天流
        
        Args:
            conversation: 对话实例，可以为None
            
        Returns:
            Optional[ChatStream]: 如果成功获取则返回聊天流，否则返回None
        """
        chat_stream = None
        
        # 1. 尝试从对话实例获取
        if conversation and hasattr(conversation, 'should_continue'):
            # 等待一小段时间，确保初始化完成
            retry_count = 0
            max_retries = 10
            while not conversation.should_continue and retry_count < max_retries:
                await asyncio.sleep(0.5)
                retry_count += 1
                logger.debug(f"[私聊][{self.private_name}]等待新对话实例初始化完成: 尝试 {retry_count}/{max_retries}")
            
            if not conversation.should_continue:
                logger.warning(f"[私聊][{self.private_name}]新对话实例初始化可能未完成，但仍将尝试获取聊天流")
            
            # 尝试使用对话实例的聊天流
            if hasattr(conversation, 'chat_stream') and conversation.chat_stream:
                logger.info(f"[私聊][{self.private_name}]使用新对话实例的聊天流")
                return conversation.chat_stream
        
        # 2. 尝试从聊天管理器获取
        from src.plugins.chat.chat_stream import chat_manager
        try:
            logger.info(f"[私聊][{self.private_name}]尝试从chat_manager获取聊天流")
            chat_stream = chat_manager.get_stream(self.stream_id)
            if chat_stream:
                return chat_stream
        except Exception as e:
            logger.warning(f"[私聊][{self.private_name}]从chat_manager获取聊天流失败: {str(e)}")
        
        # 3. 创建新的聊天流
        try:
            logger.warning(f"[私聊][{self.private_name}]无法获取现有聊天流，创建新的聊天流")
            # 创建用户信息对象
            user_info = UserInfo(
                user_id=global_config.BOT_QQ,
                user_nickname=global_config.BOT_NICKNAME,
                platform="qq"
            )
            # 创建聊天流
            return ChatStream(self.stream_id, "qq", user_info)
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]创建新聊天流失败: {str(e)}")
            return None 