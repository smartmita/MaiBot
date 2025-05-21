'''
idle_planner.py

负责决策是否要发起主动聊天的模块。
'''
import time
import traceback
import asyncio
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

# 修正导入路径
from ....common.logger_manager import get_logger
from ....config.config import global_config
from ....chat.models.utils_model import LLMRequest
from ..pfc_utils import get_items_from_json 

logger = get_logger("pfc_idle_planner")

PROMPT_IDLE_DECISION = """
你现在是{bot_nickname}
你正在严谨地评估是否应该主动向用户{private_name}发起一次闲聊。

【背景信息】
- 你和{private_name}的关系是：{relationship_description}
- 距离你们上次有效互动大约已经过去了：{last_interaction_time_desc}
- {bot_nickname}的人格核心：{bot_personality_core}
- {bot_nickname} 的一些个性特点：
{bot_personality_sides_text}
- {bot_nickname} 当前可能正在做的事情或所处的状态：{bot_current_activity}
- 最近的聊天记录回顾（如果太少，则表示你们很久没聊了或者这是初次考虑主动聊天）：
{chat_history_text}
{user_interests_text}

【决策任务】
请你综合以上所有信息，审慎判断现在是否是一个合适的时机，以及是否有充分的、积极的理由让{bot_nickname}主动向{private_name}发起对话。
请重点考虑以下几点：
1.  **用户打扰**：现在发起对话是否可能会打扰到用户？（例如，如果对方最近有未回复你的消息，或者对话明显是对方希望结束的，则不宜主动发起）
2.  **关系与个性**：主动发起对话是否符合{bot_nickname}的个性和你们当前的关系阶段？
3.  **开场白潜力**：基于现有信息，是否能构思出一个自然的、不突兀的、可能引起对方兴趣的开场话题？（你不需要生成具体的开场白内容，只需要评估这种可能性）
4.  **互动频率**：考虑到上次互动时间，现在发起是否显得过于频繁或过于疏远？
5.  **积极性**：发起对话的目的是否积极和健康，例如是真诚的问候、分享有趣的事情，而不是无意义的骚扰或刷存在感（在关系足够好时可以适当刷存在感）

严格按照JSON格式输出你的决策和详细理由：
{{
    "should_chat": "yes/no",  // 'yes' 表示你认为应该主动发起对话, 'no' 表示不应该
    "reason": "请详细阐述你做出此决策的核心原因和考量，务必具体结合上述给出的背景信息进行分析，解释为什么是'yes'或'no'。",
    "suggested_topics": ["建议的聊天话题1", "建议的聊天话题2"] // 如果决定聊天，提供2-3个可能的话题建议
}}

注意：请严格按照JSON格式输出，不要包含任何其他内容。
"""

class IdlePlanner:
    """
    决策是否发起主动聊天的规划器。
    """
    # 类级别的缓存和统计
    _decision_cache = {}  # 缓存决策结果
    _user_cooldowns = {}  # 用户冷却时间
    _user_interests = {}  # 用户兴趣模型
    _decision_stats = {   # 全局决策统计
        "total_decisions": 0,
        "positive_decisions": 0,
        "llm_failures": 0,
        "successful_chats": 0,  # 用户回复的对话次数
    }
    _conversation_topics = {}  # 用户对话主题历史
    
    def __init__(self, private_name: str):
        self.private_name = private_name
        self.default_cooldown = 3600  # 默认冷却时间（秒）
        self.max_retries = 5  # 最大重试次数
        self.retry_delay = 15  # 重试延迟（秒）
        self.llm_timeout = 30  # LLM调用超时时间（秒）
        
        # 初始化用户特定的数据
        if private_name not in self._user_interests:
            self._user_interests[private_name] = []
        
        if private_name not in self._conversation_topics:
            self._conversation_topics[private_name] = []
            
        try:
            self.llm = LLMRequest(
                model=global_config.model.pfc_action_planner, 
                temperature=global_config.model.pfc_action_planner.get("temp", 0.5),
                max_tokens=global_config.model.pfc_action_planner.get("max_tokens", 300),
                request_type="idle_planning_decision"
            )
        except TypeError as e:
            logger.error(f"[私聊][{self.private_name}] 初始化 IdlePlanner LLMRequest 时配置错误: {e}")
            raise
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}] 初始化 IdlePlanner LLMRequest 时发生未知错误: {e}")
            raise
    
    async def can_chat_now(self) -> bool:
        """检查是否已经超过冷却时间"""
        if self.private_name not in self._user_cooldowns:
            return True
        
        last_time = self._user_cooldowns.get(self.private_name, 0)
        time_since_last = time.time() - last_time
        
        if time_since_last > self.default_cooldown:
            return True
            
        logger.debug(f"[私聊][{self.private_name}] 冷却时间未到，还需等待 {(self.default_cooldown - time_since_last)/60:.1f} 分钟")
        return False
    
    def update_cooldown(self):
        """更新用户冷却时间"""
        self._user_cooldowns[self.private_name] = time.time()
        logger.debug(f"[私聊][{self.private_name}] 已更新冷却时间")
    
    async def record_decision(self, decision: bool):
        """记录决策结果"""
        self._decision_stats["total_decisions"] += 1
        if decision:
            self._decision_stats["positive_decisions"] += 1
        
        # 记录用户特定的决策
        logger.debug(f"[私聊][{self.private_name}] 决策记录：总计{self._decision_stats['total_decisions']}次，"
                   f"肯定决策{self._decision_stats['positive_decisions']}次，"
                   f"LLM失败{self._decision_stats['llm_failures']}次")
    
    async def record_successful_chat(self):
        """记录用户回复的成功对话"""
        self._decision_stats["successful_chats"] += 1
        logger.info(f"[私聊][{self.private_name}] 成功对话记录更新：{self._decision_stats['successful_chats']}次")
    
    async def extract_user_interests(self, chat_history_text: str):
        """从聊天历史中提取用户兴趣点（简化版）"""
        # 这里可以实现更复杂的兴趣提取算法
        # 简单起见，这里只是示例性地从历史中提取一些关键词
        if not chat_history_text or len(chat_history_text) < 10:
            return
            
        # 简单的兴趣提取逻辑（示例）
        interests = []
        common_interests = ["游戏", "音乐", "电影", "美食", "旅行", "动漫", "工作", "学习"]
        for interest in common_interests:
            if interest in chat_history_text and interest not in self._user_interests[self.private_name]:
                interests.append(interest)
                
        # 更新用户兴趣（最多保留5个）
        if interests:
            self._user_interests[self.private_name].extend(interests)
            self._user_interests[self.private_name] = self._user_interests[self.private_name][-5:]
            logger.debug(f"[私聊][{self.private_name}] 提取到用户兴趣: {interests}")
    
    def get_user_interests_text(self) -> str:
        """获取用户兴趣文本描述"""
        interests = self._user_interests.get(self.private_name, [])
        if not interests:
            return "- 暂无明确的用户兴趣信息。"
            
        return f"- 用户可能的兴趣点：{', '.join(interests)}"
    
    async def fallback_decision(self, hours_since_last: float) -> Tuple[bool, str, list]:
        """当LLM调用失败时的后备决策策略"""
        logger.warning(f"[私聊][{self.private_name}] 使用后备决策策略")

        return False, "我不打扰，我走了哈", []

    async def decide_to_chat_async(
        self,
        chat_history_text: str,
        relationship_description: str,
        bot_nickname: str,
        bot_personality_core: str,
        bot_personality_sides_text: str,
        bot_current_activity: str,
        last_interaction_timestamp: float
    ) -> Tuple[bool, str, list]:
        """
        使用LLM决定是否主动发起聊天，并获取原因和话题建议。
        参数:
            last_interaction_timestamp: 上次互动的时间戳（秒）
        返回值: (是否聊天, 理由, 建议话题列表)
        """
        # 计算距离上次互动的小时数
        hours_since_last = (time.time() - last_interaction_timestamp) / 3600
        
        # 生成描述性文本用于提示词
        if hours_since_last < 0.0166:  # 小于1分钟
            last_interaction_time_desc = f"约{int(hours_since_last * 3600)}秒前"
        elif hours_since_last < 1:  # 小于1小时
            last_interaction_time_desc = f"约{int(hours_since_last * 60)}分钟前"
        elif hours_since_last < 24:  # 小于1天
            last_interaction_time_desc = f"约{int(hours_since_last)}小时前"
        else:  # 大于等于1天
            last_interaction_time_desc = f"约{int(hours_since_last / 24)}天前"
        
        # 提取用户兴趣
        await self.extract_user_interests(chat_history_text)
        user_interests_text = self.get_user_interests_text()
        
        # 准备缓存键
        cache_key = f"{self.private_name}:{time.strftime('%Y%m%d%H')}"
        
        # 检查缓存
        if cache_key in self._decision_cache:
            cache_time = self._decision_cache[cache_key]['time']
            if (time.time() - cache_time) < 3600:  # 缓存1小时内有效
                logger.info(f"[私聊][{self.private_name}] 使用缓存的决策结果")
                return self._decision_cache[cache_key]['result']
        
        # 构建提示
        prompt = PROMPT_IDLE_DECISION.format(
            bot_nickname=bot_nickname,
            private_name=self.private_name,
            relationship_description=relationship_description,
            last_interaction_time_desc=last_interaction_time_desc, 
            bot_personality_core=bot_personality_core,
            bot_personality_sides_text=bot_personality_sides_text if bot_personality_sides_text.strip() else "无特别的个性侧面信息。",
            bot_current_activity=bot_current_activity if bot_current_activity.strip() else "目前没有特定的活动信息。",
            chat_history_text=chat_history_text if chat_history_text.strip() else "最近没有聊天记录。",
            user_interests_text=user_interests_text
        )

        logger.debug(f"[私聊][{self.private_name}] IdlePlanner 发送给LLM的决策提示词:\n------\n{prompt}\n------")

        # LLM决策尝试（带重试逻辑）
        for attempt in range(self.max_retries):
            try:
                llm_start_time = time.time()
                content, _ = await asyncio.wait_for(
                    self.llm.generate_response_async(prompt),
                    timeout=self.llm_timeout
                )
                llm_duration = time.time() - llm_start_time
                logger.debug(f"[私聊][{self.private_name}] IdlePlanner LLM 耗时: {llm_duration:.3f} 秒, 原始返回: {content}")

                success, result_dict = get_items_from_json(
                    content,
                    self.private_name,
                    "should_chat", "reason", "suggested_topics", 
                    default_values={
                        "should_chat": "no", 
                        "reason": "LLM返回格式错误或未提供决策信息，默认不发起聊天。",
                        "suggested_topics": []
                    },
                    required_types={"should_chat": str, "reason": str, "suggested_topics": list}
                )

                should_chat_str = result_dict.get("should_chat", "no").lower()
                reason = result_dict.get("reason", "LLM未提供明确原因。")
                suggested_topics = result_dict.get("suggested_topics", [])

                if not success:
                    logger.warning(f"[私聊][{self.private_name}] IdlePlanner未能成功解析LLM的JSON输出，尝试重试。原始输出: {content}")
                    await asyncio.sleep(self.retry_delay)
                    continue

                decision = should_chat_str == "yes"
                logger.info(f"[私聊][{self.private_name}] IdlePlanner LLM决策结果: 是否聊天={decision}, 原因: {reason}")
                
                # 更新冷却时间
                self.update_cooldown()
                
                # 记录决策结果
                await self.record_decision(decision)
                
                # 缓存结果
                self._decision_cache[cache_key] = {
                    'time': time.time(),
                    'result': (decision, reason, suggested_topics)
                }
                
                # 如果决定聊天，更新对话主题
                if decision and suggested_topics:
                    if self.private_name in self._conversation_topics:
                        self._conversation_topics[self.private_name].extend(suggested_topics)
                        self._conversation_topics[self.private_name] = self._conversation_topics[self.private_name][-10:] # 保留最近10个
                
                return decision, reason, suggested_topics

            except asyncio.TimeoutError:
                logger.warning(f"[私聊][{self.private_name}] IdlePlanner LLM调用超时 (尝试 {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}] IdlePlanner 在调用LLM时出错 (尝试 {attempt+1}/{self.max_retries}): {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(self.retry_delay)
        
        # 所有尝试都失败，直接取消任务
        self._decision_stats["llm_failures"] += 1
        logger.error(f"[私聊][{self.private_name}] IdlePlanner 在{self.max_retries}次尝试后仍然失败，取消主动聊天任务")
        return False, f"IdlePlanner在{self.max_retries}次尝试后仍然失败，取消主动聊天任务", []

async def main_test():
    planner = IdlePlanner(private_name="测试用户")
    decision, reason, topics = await planner.decide_to_chat_async(
        chat_history_text="用户：昨天天气真好啊！\n我：是啊，阳光明媚的。",
        relationship_description="普通朋友",
        bot_nickname="Plutor",
        bot_personality_core="友好、乐于助人",
        bot_personality_sides_text="- 喜欢音乐\n- 有点小迷糊",
        bot_current_activity="正在看心理学书籍",
        last_interaction_timestamp=time.time() - 3 * 3600
    )
    print(f"测试决策：是否聊天 -> {decision}")
    print(f"测试原因：{reason}")
    print(f"建议话题：{topics}")

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main_test()) # 取消注释以进行本地测试
    pass
