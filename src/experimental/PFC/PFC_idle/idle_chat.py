from typing import Optional, Dict, Any
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from src.chat.message_receive.chat_stream import chat_manager
from src.chat.message_receive.chat_stream import ChatStream

from ..chat_observer import ChatObserver
from ..message_sender import DirectMessageSender
from ..pfc_relationship import PfcRepationshipTranslator, PfcRelationshipUpdater
from maim_message import Seg # type: ignore
from rich.traceback import install
from ..pfc_utils import build_chat_history_text, get_items_from_json
from .idle_weight import get_user_relationship_data
from .idle_conversation import IdleConversation
from .idle_planner import IdlePlanner
from ....experimental.Legacy_HFC.schedule.schedule_generator import bot_schedule

install(extra_lines=3)

logger = get_logger("pfc_idle_chat")

class IdleChat:
    """主动聊天组件（测试中）
    负责生成主动聊天的具体内容，根据用户特点和聊天历史生成合适的开场白
    """

    # 单例模式实现
    _instances: Dict[str, "IdleChat"] = {}
    
    # 类级别的配置
    MAX_LLM_RETRIES = 5  # 生成聊天内容的最大重试次数
    LLM_RETRY_DELAY = 15  # 重试间隔（秒）
    LLM_TIMEOUT = 30  # LLM调用超时（秒）
    
    # 存储下一次计划的对话时间
    _next_idle_times: Dict[str, Dict[str, Any]] = {}

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
        
        # 初始化 IdlePlanner
        self.idle_planner = IdlePlanner(private_name=self.private_name)
        
        # 最后消息时间 (更准确地说，应该是上次尝试主动聊天的决策时间)
        self.last_message_time = time.time()
        
        # 记录上次发送的主题，避免重复
        self.last_topics = []

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
                
                if hasattr(existing_chat_stream, "user_info") and existing_chat_stream.user_info:
                    user_info = existing_chat_stream.user_info
                
                return existing_chat_stream
            
            logger.debug(f"[私聊][{self.private_name}]未找到聊天流")
            return None
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]获取聊天流时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def _decide_next_chat_time(self, chat_history_text: str, relationship_description: str, current_activity: str) -> None:
        """决定下一次主动对话的时机
        
        Args:
            chat_history_text: 聊天历史文本
            relationship_description: 关系描述
            current_activity: 当前活动
        """
        try:
            today_schedule = ""
            try:
                # 获取全天日程
                if hasattr(bot_schedule, 'today_schedule_text') and bot_schedule.today_schedule_text:
                    today_schedule = bot_schedule.today_schedule_text.strip()
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]获取全天日程时出错: {str(e)}")
                
            bot_nickname = global_config.bot.nickname
            
            # 创建用于决策的LLM请求
            next_chat_llm = LLMRequest(
                model=global_config.model.normal, 
                temperature=0.3, 
                max_tokens=300, 
                request_type="idle_next_chat_decision"
            )
            
            # 构建提示词
            prompt = f"""你是{bot_nickname}，你刚刚向用户{self.private_name}发送了主动聊天消息，现在需要决定下一次主动聊天的最佳时机。
            你和{self.private_name}的关系是：{relationship_description}
            这是你们之前的对话历史：
            {chat_history_text}
            {"根据你的日程，你现在正在: " + current_activity if current_activity else ""}
            
            {"以下是你今天的完整日程安排：\\n" + today_schedule if today_schedule else ""}
            
            根据用户的情况、你们的关系以及以上上下文，请分析并决定下一次主动聊天的最佳时间。
            考虑因素：
            1. 用户可能的作息时间（基于聊天历史推测）
            2. 关系亲密度（关系越好，可以更频繁聊天）
            3. 之前聊天的质量和回复情况
            4. 你的日程安排
            
            请严格按照以下JSON格式输出你的决策：
            {{
                "next_idle_time": "YYYY-MM-DD HH:MM:SS",  // 下次主动聊天的时间（24小时制），必须是未来的时间
                "reason": "详细说明为什么选择这个时间点",  // 选择这个时间的原因
                "user": "{self.private_name}",  // 用户名称
                "fallback_hours": 1,  // 如果指定的时间已经过期，建议几小时后再次聊天（0-12之间的整数）
                "fallback_minutes": 30  // 额外的分钟数（0-59之间的整数），与fallback_hours组合使用
            }}
            
            只输出格式正确的JSON，不要有任何其他内容。
            """
            
            # 实现多次重试
            success = False
            result = None
            
            for attempt in range(self.MAX_LLM_RETRIES):
                try:
                    logger.debug(f"[私聊][{self.private_name}]开始生成下次主动聊天时间决策 (尝试 {attempt+1}/{self.MAX_LLM_RETRIES})")
                    content, _ = await asyncio.wait_for(
                        next_chat_llm.generate_response_async(prompt), 
                        timeout=self.LLM_TIMEOUT
                    )
                    
                    # 添加LLM原始输出的调试日志
                    logger.debug(f"[私聊][{self.private_name}]LLM原始输出: {content}")
                    
                    success, result = get_items_from_json(
                        content,
                        self.private_name,
                        "next_idle_time", "reason", "user", "fallback_hours", "fallback_minutes",
                        default_values={
                            "next_idle_time": (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                            "reason": "默认安排在2小时后进行下次对话",
                            "user": self.private_name,
                            "fallback_hours": 1,
                            "fallback_minutes": 30
                        }
                    )
                    
                    if not success:
                        logger.warning(f"[私聊][{self.private_name}]解析下次主动聊天时间JSON失败，尝试重试")
                        await asyncio.sleep(self.LLM_RETRY_DELAY)
                        continue
                    
                    # 验证时间格式
                    try:
                        next_time_str = result["next_idle_time"]
                        next_time = datetime.strptime(next_time_str, "%Y-%m-%d %H:%M:%S")
                        
                        # 添加调试信息
                        logger.debug(f"[私聊][{self.private_name}]LLM返回的下次聊天时间: {next_time_str}, 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 确保时间在将来
                        now = datetime.now()
                        if next_time <= now:
                            logger.warning(f"[私聊][{self.private_name}]LLM返回的时间 {next_time_str} 已经过期（当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}），将使用备选时间间隔")
                            
                            # 使用LLM提供的fallback_hours和fallback_minutes而不是发送新请求
                            fallback_hours = 1  # 默认小时
                            fallback_minutes = 30  # 默认分钟
                            try:
                                # 处理小时部分
                                if "fallback_hours" in result and isinstance(result["fallback_hours"], (int, str)):
                                    fallback_value = result["fallback_hours"]
                                    if isinstance(fallback_value, str):
                                        # 尝试将字符串转换为整数
                                        import re
                                        numbers = re.findall(r'\d+', fallback_value)
                                        if numbers:
                                            fallback_candidate = int(numbers[0])
                                            if 0 <= fallback_candidate <= 12:
                                                fallback_hours = fallback_candidate
                                    elif isinstance(fallback_value, int) and 0 <= fallback_value <= 12:
                                        fallback_hours = fallback_value
                                
                                # 处理分钟部分
                                if "fallback_minutes" in result and isinstance(result["fallback_minutes"], (int, str)):
                                    fallback_value = result["fallback_minutes"]
                                    if isinstance(fallback_value, str):
                                        # 尝试将字符串转换为整数
                                        import re
                                        numbers = re.findall(r'\d+', fallback_value)
                                        if numbers:
                                            fallback_candidate = int(numbers[0])
                                            if 0 <= fallback_candidate <= 59:
                                                fallback_minutes = fallback_candidate
                                    elif isinstance(fallback_value, int) and 0 <= fallback_value <= 59:
                                        fallback_minutes = fallback_value
                                
                                # 记录fallback值的调试信息
                                logger.debug(f"[私聊][{self.private_name}]使用LLM提供的备选间隔: {fallback_hours}小时{fallback_minutes}分钟")
                                
                                # 确保至少有15分钟的间隔
                                if fallback_hours == 0 and fallback_minutes < 15:
                                    fallback_minutes = 15
                                    logger.warning(f"[私聊][{self.private_name}]备选时间间隔过短，调整为至少15分钟")
                                
                            except Exception as e:
                                logger.warning(f"[私聊][{self.private_name}]处理fallback时间间隔时出错: {str(e)}")
                                
                            next_time = now + timedelta(hours=fallback_hours, minutes=fallback_minutes)
                            next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
                            logger.warning(f"[私聊][{self.private_name}]已根据备选策略设置下次聊天时间为当前时间+{fallback_hours}小时{fallback_minutes}分钟: {next_time_str}")
                            
                        # 检查时间是否在活动时间内（6点到24点）
                        next_hour = next_time.hour
                        if not (6 <= next_hour < 24):
                            # 如果在非活动时间，调整到第二天早上8点
                            if next_hour < 6:
                                # 当天早上8点
                                adjusted_time = next_time.replace(hour=8, minute=0, second=0)
                            else:
                                # 第二天早上8点
                                adjusted_time = (next_time + timedelta(days=1)).replace(hour=8, minute=0, second=0)
                            
                            next_time = adjusted_time
                            next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
                            logger.info(f"[私聊][{self.private_name}]计划的下次聊天时间调整到活动时间内: {next_time_str}")
                            
                        # 保存决策结果
                        IdleChat._next_idle_times[self.private_name] = {
                            "next_idle_time": next_time,
                            "next_idle_time_str": next_time_str,
                            "reason": result["reason"],
                            "user": self.private_name,
                            "chat_instance": self
                        }
                        
                        logger.info(f"[私聊][{self.private_name}]成功计划下次主动聊天时间: {next_time_str}, 原因: {result['reason']}")
                        break
                        
                    except ValueError as ve:
                        logger.error(f"[私聊][{self.private_name}]下次主动聊天时间格式错误: {result['next_idle_time']}, 错误: {str(ve)}")
                        await asyncio.sleep(self.LLM_RETRY_DELAY)
                        
                except asyncio.TimeoutError:
                    logger.error(f"[私聊][{self.private_name}]生成下次主动聊天时间决策超时 (尝试 {attempt+1}/{self.MAX_LLM_RETRIES})")
                    await asyncio.sleep(self.LLM_RETRY_DELAY)
                except Exception as llm_err:
                    logger.error(f"[私聊][{self.private_name}]生成下次主动聊天时间决策失败 (尝试 {attempt+1}/{self.MAX_LLM_RETRIES}): {str(llm_err)}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(self.LLM_RETRY_DELAY)
                    
            # 如果所有尝试都失败，使用默认值
            if not success or result is None:
                logger.warning(f"[私聊][{self.private_name}]在{self.MAX_LLM_RETRIES}次尝试后仍未能生成下次主动聊天时间决策，使用默认值")
                
                # 使用默认值：当前时间+2小时
                now = datetime.now()
                next_time = now + timedelta(hours=2)
                
                # 检查时间是否在活动时间内（6点到24点）
                next_hour = next_time.hour
                if not (6 <= next_hour < 24):
                    # 调整到第二天早上8点
                    if next_hour < 6:
                        next_time = next_time.replace(hour=8, minute=0, second=0)
                    else:
                        next_time = (next_time + timedelta(days=1)).replace(hour=8, minute=0, second=0)
                
                next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 保存决策结果
                IdleChat._next_idle_times[self.private_name] = {
                    "next_idle_time": next_time,
                    "next_idle_time_str": next_time_str,
                    "reason": "由于LLM决策失败，默认安排在合适的时间进行下次对话",
                    "user": self.private_name,
                    "chat_instance": self
                }
                
                logger.info(f"[私聊][{self.private_name}]使用默认计划时间: {next_time_str}")
                    
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]决定下次主动聊天时间过程中发生未预期的错误: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _initiate_chat(self) -> None:
        """决策是否发起聊天，如果决策是，则生成并发送主动聊天消息"""
        try:
            # 1. 收集决策所需信息
            current_time_for_history = time.time()
            messages = await self.chat_observer.message_storage.get_messages_before(
                self.stream_id, 
                current_time_for_history,  
                limit=30
            )
            
            class TempObservationInfo:
                def __init__(self, messages):
                    self.chat_history = messages
                    self.chat_history_str = None # 将由 build_chat_history_text 填充
                    self.new_messages_count = 0
                    self.unprocessed_messages = []
                    self.bot_id = str(global_config.bot.qq_account)
                    self.current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
            observation_info = TempObservationInfo(messages)
            chat_history_text = await build_chat_history_text(observation_info, self.private_name)
            
            relationship_description = "普通"
            user_id_for_relation = self.private_name # 默认使用 private_name
            try:
                chat_stream_for_id = await self._get_chat_stream()
                if chat_stream_for_id and hasattr(chat_stream_for_id, "user_info") and chat_stream_for_id.user_info and hasattr(chat_stream_for_id.user_info, "user_id") and chat_stream_for_id.user_info.user_id:
                    user_id_for_relation = chat_stream_for_id.user_info.user_id
                
                _, _, relationship_description = await get_user_relationship_data(
                    user_id_for_relation, "qq", self.private_name 
                )
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]获取关系数据失败: {str(e)}")

            bot_nickname = global_config.bot.nickname
            personality_core = global_config.personality.personality_core
            personality_sides = global_config.personality.personality_sides
            
            # 准备人格侧面信息
            personality_sides_text = ""
            if personality_sides:
                personality_sides_text = "\n".join([f"- {side}" for side in personality_sides])
            
            # 获取日程信息 - 同时获取当前活动和全天日程
            current_activity = ""
            today_schedule = ""
            try:
                # 获取全天日程
                if hasattr(bot_schedule, 'today_schedule_text') and bot_schedule.today_schedule_text:
                    today_schedule = bot_schedule.today_schedule_text.strip()
                    
                # 获取当前活动
                if hasattr(bot_schedule, 'today_done_list') and bot_schedule.today_done_list:
                    # 获取最近的活动
                    current_activity_candidate = bot_schedule.get_current_num_task(1, True)
                    if current_activity_candidate:
                        current_activity = current_activity_candidate.strip()
                
                # 如果从today_done_list没有获取到活动，尝试从日程表中提取当前活动
                if not current_activity and today_schedule:
                    # 从完整日程中提取当前时间段的活动
                    current_time_str = datetime.now().strftime("%H:%M")
                    hour_now = current_time_str.split(":")[0]
                    minute_now = current_time_str.split(":")[1]
                    schedule_lines = today_schedule.split("\n")
                    
                    # 首先尝试精确匹配当前小时
                    for line in schedule_lines:
                        if (hour_now + ":" in line or hour_now + "时" in line) and len(line.strip()) > 5:
                            current_activity = line.strip()
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
                                    break
                            if current_activity:
                                break
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]获取日程活动时出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 2. 调用 IdlePlanner 进行决策 - 不单独计算时间描述，依赖IdleManager的冷却机制（问就是懒）
            logger.info(f"[私聊][{self.private_name}] 调用 IdlePlanner 进行主动聊天决策...")
            # 不再使用实例本地的last_message_time，改为从IdleManager全局冷却机制中获取状态
            from .idle_manager import IdleManager
            global_last_trigger = IdleManager._last_global_trigger_time
            
            should_chat, decision_reason, suggested_topics = await self.idle_planner.decide_to_chat_async(
                chat_history_text=chat_history_text,
                relationship_description=relationship_description,
                bot_nickname=bot_nickname,
                bot_personality_core=personality_core,
                bot_personality_sides_text=personality_sides_text,
                bot_current_activity=current_activity,
                last_interaction_timestamp=global_last_trigger  # 使用全局触发时间替代实例本地时间
            )

            # 不再更新实例本地的时间戳，全部依赖IdleManager的全局冷却机制
            # 当决策为"要聊天"时，IdleManager中的_last_global_trigger_time会自动更新
            pass

            if not should_chat:
                logger.info(f"[私聊][{self.private_name}] IdlePlanner 决策不发起主动聊天。原因: {decision_reason}")
                return
            
            logger.info(f"[私聊][{self.private_name}] IdlePlanner 决策发起主动聊天。原因: {decision_reason}")
            if suggested_topics:
                logger.info(f"[私聊][{self.private_name}] 建议的话题: {', '.join(suggested_topics)}")
            
            # 3. 如果决策为是，则生成并发送消息
            # 添加建议话题到提示中
            topics_text = ""
            if suggested_topics:
                # 过滤掉上次已使用的话题
                filtered_topics = [t for t in suggested_topics if t not in self.last_topics]
                if filtered_topics:
                    topics_text = f"建议的聊天话题：{', '.join(filtered_topics)}\n尝试自然地围绕这些话题展开对话。"
                else:
                    # 如果所有话题都是上次用过的，还是使用原始话题，但给模型提示
                    topics_text = f"建议的聊天话题（这些是重复话题，请尝试换个角度）：{', '.join(suggested_topics)}"
                
                # 记录本次使用的话题
                self.last_topics = suggested_topics
            
            prompt_for_initiation = f"""你是{global_config.bot.nickname}。
            你正在与用户{self.private_name}进行QQ私聊，你们的关系是{relationship_description}
            你的人格核心特点：{personality_core}
            {f"你的一些个性特点：\n{personality_sides_text}" if personality_sides_text else ""}
            {f"根据你的日程，你现在正在: {current_activity}" if current_activity else ""}
            
            {f"以下是你今天的完整日程安排：\\n{today_schedule}" if today_schedule else ""}
            你已经决定主动发起对话（因为{decision_reason}）。
            {topics_text if topics_text else ""}
            请基于以下之前的对话历史，生成一条自然、友好、符合关系程度的主动对话消息。
            这条消息应能够引起用户的兴趣，重新开始对话。
            最近的对话历史（并不是现在的对话）：
            {chat_history_text}
            {"如果你决定告诉对方你在做什么，请自然地融入你的日程活动信息。" if current_activity else "请你根据对话历史决定是告诉对方你正在做的事情，还是询问对方正在做的事情"}
            请直接输出一条消息，不要有任何额外的解释或引导文字，不要输出表情包和句号等标点符号
            消息内容尽量简短
            """
            
            # 实现多次重试
            content = None
            for attempt in range(self.MAX_LLM_RETRIES):
                try:
                    logger.debug(f"[私聊][{self.private_name}]开始生成主动聊天内容 (尝试 {attempt+1}/{self.MAX_LLM_RETRIES})")
                    content, _ = await asyncio.wait_for(
                        self.llm.generate_response_async(prompt_for_initiation), 
                        timeout=self.LLM_TIMEOUT
                    )
                    logger.debug(f"[私聊][{self.private_name}]成功生成主动聊天内容: {content}")
                    break  # 成功生成，跳出循环
                except asyncio.TimeoutError:
                    logger.error(f"[私聊][{self.private_name}]生成主动聊天内容超时 (尝试 {attempt+1}/{self.MAX_LLM_RETRIES})")
                    await asyncio.sleep(self.LLM_RETRY_DELAY)
                except Exception as llm_err:
                    logger.error(f"[私聊][{self.private_name}]生成主动聊天内容失败 (尝试 {attempt+1}/{self.MAX_LLM_RETRIES}): {str(llm_err)}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(self.LLM_RETRY_DELAY)
            
            # 如果所有重试都失败，直接取消任务
            if not content:
                logger.error(f"[私聊][{self.private_name}]在{self.MAX_LLM_RETRIES}次尝试后仍未能生成主动聊天内容，取消本次主动聊天任务")
                return  # 直接返回，不再继续后续的发送消息逻辑
                
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
                
                # 在发送消息时已更新了IdleManager._last_global_trigger_time
                pass

                from .idle_manager import IdleManager # 延迟导入以避免循环依赖
                async with IdleManager._global_lock:
                    IdleManager._pending_replies[self.private_name] = time.time()
                    IdleManager._tried_users.add(self.private_name)
                    logger.info(f"[私聊][{self.private_name}]已添加到等待回复列表中")
                
                # 记录成功发送的对话尝试
                await self.idle_planner.record_successful_chat()
                
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
                
                # 在成功发送消息后，决定下一次主动对话的时机
                logger.info(f"[私聊][{self.private_name}]开始决定下一次主动对话的时机")
                await self._decide_next_chat_time(chat_history_text, relationship_description, current_activity)
                
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]发送主动聊天消息失败: {str(e)}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]主动发起聊天过程中发生未预期的错误: {str(e)}")
            logger.error(traceback.format_exc())
