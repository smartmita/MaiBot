# TODO: 优化 idle 逻辑 增强其与 PFC 模式的联动（在做了在做了TAT）
from typing import Optional, Dict
import asyncio
import time
import traceback
from datetime import datetime
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from src.chat.message_receive.chat_stream import chat_manager
from src.chat.message_receive.chat_stream import ChatStream

from ..chat_observer import ChatObserver
from ..message_sender import DirectMessageSender
from ..pfc_relationship import PfcRepationshipTranslator, PfcRelationshipUpdater
from maim_message import Seg
from rich.traceback import install
from ..pfc_utils import build_chat_history_text
from .idle_weight import get_user_relationship_data
from .idle_conversation import IdleConversation
# 导入日程系统
from src.experimental.Legacy_HFC.schedule.schedule_generator import bot_schedule

install(extra_lines=3)

logger = get_logger("pfc_idle_chat")

class IdleChat:
    """主动聊天组件（测试中）
    负责生成主动聊天的具体内容，根据用户特点和聊天历史生成合适的开场白
    """

    # 单例模式实现
    _instances: Dict[str, "IdleChat"] = {}

    @classmethod
    def get_instance(cls, stream_id: str, private_name: str) -> "IdleChat":
        """获取IdleChat实例（单例模式）

        Args:
            stream_id: 聊天流ID
            private_name: 私聊用户名称

        Returns:
            IdleChat: IdleChat实例
        """
        key = f"{private_name}:{stream_id}"
        if key not in cls._instances:
            cls._instances[key] = cls(stream_id, private_name)
            logger.info(f"[私聊][{private_name}]创建新的IdleChat实例")
        return cls._instances[key]

    def __init__(self, stream_id: str, private_name: str):
        """初始化主动聊天组件

        Args:
            stream_id: 聊天流ID
            private_name: 私聊用户名称
        """
        self.stream_id = stream_id
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.message_sender = DirectMessageSender(private_name)
        
        # 初始化关系组件
        self.relationship_translator = PfcRepationshipTranslator(private_name)
        self.relationship_updater = PfcRelationshipUpdater(private_name, global_config.bot.nickname)

        # LLM请求（推理模型），用于生成主动对话内容
        self.llm = LLMRequest(
            model=global_config.model.idle_chat, 
            temperature=global_config.model.idle_chat["temp"], 
            max_tokens=global_config.model.idle_chat["max_tokens"], 
            request_type="idle_chat")
        
        # 最后消息时间
        self.last_message_time = time.time()

    async def update_last_message_time(self, current_time=None):
        """更新最后消息时间
        
        Args:
            current_time: 当前时间，如果为None则使用当前系统时间
        """
        self.last_message_time = current_time if current_time is not None else time.time()
        logger.debug(f"[私聊][{self.private_name}]更新最后消息时间: {self.last_message_time}")

    async def _get_chat_stream(self) -> Optional[ChatStream]:
        """获取聊天流实例"""
        try:
            existing_chat_stream = chat_manager.get_stream(self.stream_id)
            if existing_chat_stream:
                logger.info(f"[私聊][{self.private_name}]从chat_manager找到现有聊天流")
                
                # 记录现有聊天流的用户信息
                if hasattr(existing_chat_stream, "user_info") and existing_chat_stream.user_info:
                    user_info = existing_chat_stream.user_info
                    logger.info(f"[私聊][{self.private_name}]现有聊天流用户信息: {user_info}")
                    
                    if hasattr(user_info, "user_id"):
                        logger.info(f"[私聊][{self.private_name}]现有聊天流user_id: {user_info.user_id}")
                    if hasattr(user_info, "user_nickname"):
                        logger.info(f"[私聊][{self.private_name}]现有聊天流user_nickname: {user_info.user_nickname}")
                
                return existing_chat_stream
            
            logger.debug(f"[私聊][{self.private_name}]未找到聊天流")
            return None
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]获取聊天流时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def _initiate_chat(self) -> None:
        """生成并发送主动聊天消息"""
        try:
            # 从数据库直接加载聊天历史记录
            current_time = time.time()
            messages = await self.chat_observer.message_storage.get_messages_before(
                self.stream_id, 
                current_time,  
                limit=30  # 增大获取的历史消息数量，提供更多上下文
            )
            
            # 创建一个临时的ObservationInfo对象用于构建历史记录文本
            class TempObservationInfo:
                def __init__(self, messages):
                    self.chat_history = messages
                    self.chat_history_str = None
                    self.new_messages_count = 0
                    self.unprocessed_messages = []
                    
            # 创建临时观察信息对象
            observation_info = TempObservationInfo(messages)
            
            # 使用pfc_utils中的方法获取正确格式的聊天历史文本
            chat_history_text = await build_chat_history_text(observation_info, self.private_name)
            
            # 获取关系数据
            relationship_description = "普通"  # 设置默认值
            try:
                platform = "qq"  # 假设用户来自QQ平台
                user_id = None
                
                # 尝试获取聊天流并从中提取user_id
                try:
                    # 首先尝试获取当前实例的聊天流
                    chat_stream = await self._get_chat_stream()
                    logger.info(f"[私聊][{self.private_name}]获取到聊天流: {chat_stream}")
                    
                    if chat_stream and hasattr(chat_stream, "user_info") and chat_stream.user_info:
                        if hasattr(chat_stream.user_info, "user_id") and chat_stream.user_info.user_id:
                            user_id = chat_stream.user_info.user_id
                            logger.info(f"[私聊][{self.private_name}]从chat_stream成功获取到user_id: {user_id}")
                except Exception as e:
                    logger.error(f"[私聊][{self.private_name}]从chat_stream获取user_id失败: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # 如果从chat_stream获取失败，回退到使用private_name
                if not user_id:
                    try:
                        user_id = int(self.private_name)
                    except ValueError:
                        # 如果不能转换为整数，使用字符串值
                        user_id = self.private_name
                
                # 从idle_weight模块获取用户关系数据
                _, _, relationship_description = await get_user_relationship_data(
                    user_id, platform, self.private_name
                )
            
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]获取关系数据失败: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 构建提示词
            current_time = datetime.now().strftime("%H:%M")
            
            # 从全局配置中获取人格信息
            personality_core = global_config.personality.personality_core
            personality_sides = global_config.personality.personality_sides
            
            # 准备人格侧面信息
            personality_sides_text = ""
            if personality_sides:
                personality_sides_text = "\n".join([f"- {side}" for side in personality_sides])
            
            # 获取当前日程活动信息
            current_activity = ""
            try:
                if hasattr(bot_schedule, 'today_done_list') and bot_schedule.today_done_list:
                    # 获取最近的活动
                    current_activity = bot_schedule.get_current_num_task(1, True).strip()
                    if current_activity:
                        logger.info(f"[私聊][{self.private_name}]获取到最近日程活动: {current_activity}")
                
                # 如果从today_done_list没有获取到活动，尝试从日程表中获取
                if not current_activity and hasattr(bot_schedule, 'today_schedule_text') and bot_schedule.today_schedule_text:
                    # 从完整日程中提取当前时间段的活动
                    hour_now = current_time.split(":")[0]
                    minute_now = current_time.split(":")[1]
                    schedule_lines = bot_schedule.today_schedule_text.split("\n")
                    
                    # 首先尝试精确匹配当前小时
                    for line in schedule_lines:
                        if (hour_now + ":" in line or hour_now + "时" in line) and len(line.strip()) > 5:
                            current_activity = line.strip()
                            logger.info(f"[私聊][{self.private_name}]从日程表中获取到当前时间活动: {current_activity}")
                            break
                    
                    # 如果没找到，尝试查找最近的时间段
                    if not current_activity:
                        current_hour = int(hour_now)
                        # 查找前后1小时内的活动
                        for h in [current_hour, current_hour-1, current_hour+1]:
                            h_str = str(h).zfill(2)  # 确保两位数格式
                            for line in schedule_lines:
                                if (h_str + ":" in line or h_str + "时" in line) and len(line.strip()) > 5:
                                    current_activity = line.strip()
                                    logger.info(f"[私聊][{self.private_name}]从日程表中获取到附近时间活动: {current_activity}")
                                    break
                            if current_activity:
                                break
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]获取日程活动时出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            prompt = f"""你是{global_config.bot.nickname}。
            你正在与用户{self.private_name}进行QQ私聊，你们的关系是{relationship_description}
            你的人格核心特点：{personality_core}
            {f"你的一些个性特点：\n{personality_sides_text}" if personality_sides_text else ""}
            {f"根据你的日程，你现在正在: {current_activity}" if current_activity else ""}
            你想要主动发起对话。
            请基于以下之前的对话历史，生成一条自然、友好、符合关系程度的主动对话消息。
            这条消息应能够引起用户的兴趣，重新开始对话。
            最近的对话历史（并不是现在的对话）：
            {chat_history_text}
            {"如果你决定告诉对方你在做什么，请自然地融入你的日程活动信息。" if current_activity else "请你根据对话历史决定是告诉对方你正在做的事情，还是询问对方正在做的事情"}
            请直接输出一条消息，不要有任何额外的解释或引导文字，不要输出表情包
            消息内容尽量简短
            """
            
            # 记录完整的prompt
            logger.info(f"[私聊][{self.private_name}]生成的完整prompt:\n{'-'*50}\n{prompt}\n{'-'*50}")

            # 生成回复
            logger.debug(f"[私聊][{self.private_name}]开始生成主动聊天内容")
            try:
                content, _ = await asyncio.wait_for(self.llm.generate_response_async(prompt), timeout=30)
                logger.debug(f"[私聊][{self.private_name}]成功生成主动聊天内容: {content}")
            except asyncio.TimeoutError:
                logger.error(f"[私聊][{self.private_name}]生成主动聊天内容超时")
                return
            except Exception as llm_err:
                logger.error(f"[私聊][{self.private_name}]生成主动聊天内容失败: {str(llm_err)}")
                logger.error(traceback.format_exc())
                return
                
            # 清理结果
            content = content.strip()
            content = content.strip("\"'")

            if not content:
                logger.error(f"[私聊][{self.private_name}]生成的主动聊天内容为空")
                return

            # 获取聊天流
            chat_stream = await self._get_chat_stream()
            if not chat_stream:
                logger.error(f"[私聊][{self.private_name}]无法获取有效的聊天流，取消发送主动消息")
                return

            # 发送消息
            try:
                segments = Seg(type="seglist", data=[Seg(type="text", data=content)])
                logger.debug(f"[私聊][{self.private_name}]准备发送主动聊天消息: {content}")
                await self.message_sender.send_message(
                    chat_stream=chat_stream, segments=segments, reply_to_message=None, content=content
                )
                logger.info(f"[私聊][{self.private_name}]成功主动发起聊天: {content}")
                
                # 将此用户添加到待回复列表
                from .idle_manager import IdleManager
                async with IdleManager._global_lock:
                    IdleManager._pending_replies[self.private_name] = time.time()
                    IdleManager._tried_users.add(self.private_name)
                    logger.info(f"[私聊][{self.private_name}]已添加到等待回复列表中")
                
                # 成功发送消息后，启动对话实例
                logger.info(f"[私聊][{self.private_name}]尝试为用户启动PFC对话实例")
                try:
                    conversation = await IdleConversation.start_conversation_for_user(self.stream_id, self.private_name)
                    if conversation:
                        logger.info(f"[私聊][{self.private_name}]成功启动PFC对话实例")
                    else:
                        logger.warning(f"[私聊][{self.private_name}]未能成功启动PFC对话实例")
                except Exception as conv_err:
                    logger.error(f"[私聊][{self.private_name}]启动PFC对话实例时出错: {str(conv_err)}")
                    logger.error(traceback.format_exc())
                
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]发送主动聊天消息失败: {str(e)}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]主动发起聊天过程中发生未预期的错误: {str(e)}")
            logger.error(traceback.format_exc())
