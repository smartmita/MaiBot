import asyncio
import difflib
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import jieba # type: ignore

# PFC 类型定义 (如果PFC相关类不在标准路径，静态分析器可能会报错，但不影响代码逻辑)
try:
    from experimental.PFC.conversation_info import ConversationInfo
    from experimental.PFC.observation_info import ObservationInfo
    from src.chat.message_receive.chat_stream import ChatStream
except ImportError:
    ConversationInfo = Optional[Any] # type: ignore
    ObservationInfo = Optional[Any] # type: ignore
    ChatStream = Optional[Any] # type: ignore

# HFC 类型定义
from .observation import ChattingObservation
from .chat_state_info import ChatStateInfo # HFC的ChatStateInfo
from ..heartFC_Cycleinfo import CycleInfo # HFC的CycleInfo

# 通用模块导入
from src.chat.knowledge.knowledge_lib import qa_manager
from src.chat.memory_system.Hippocampus import HippocampusManager
from src.chat.message_receive.chat_stream import chat_manager
from src.chat.models.utils_model import LLMRequest
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.utils.chat_message_builder import (
    build_readable_messages,
    get_raw_msg_before_timestamp_with_chat,
)
from src.chat.utils.json_utils import process_llm_tool_calls, safe_json_dumps
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.individuality.individuality import Individuality
from src.tools.tool_use import ToolUser
from ..schedule.schedule_generator import bot_schedule
# nickname_manager 的导入路径可能需要根据你的项目结构调整
from src.experimental.profile.sobriquet.nickname_manager import nickname_manager


logger = get_logger("sub_mind")

# --- 辅助函数 ---
def parse_knowledge_and_get_max_relevance(knowledge_str: str) -> Tuple[Optional[str], float]:
    """
    解析 qa_manager.get_knowledge 返回的字符串，提取所有知识的文本和最高的相关性得分。
    返回: (原始知识字符串, 最高相关性得分)，如果无有效相关性则返回 (原始知识字符串, 0.0)
    """
    if not knowledge_str:
        return None, 0.0

    max_relevance = 0.0
    relevance_scores = re.findall(r"该条知识对于问题的相关性：([0-9.]+)", knowledge_str)

    if relevance_scores:
        try:
            max_relevance = max(float(score) for score in relevance_scores)
        except ValueError:
            logger.warning(f"解析相关性得分时出错: {relevance_scores}")
            return knowledge_str, 0.0
    else:
        logger.debug(f"在知识字符串中未找到明确的相关性得分标记: '{knowledge_str[:100]}...'")
        return knowledge_str, 0.0 # 即使没有标记，也返回原始字符串和0.0相关性

    return knowledge_str, max_relevance

def calculate_similarity(text_a: str, text_b: str) -> float:
    """计算两个文本字符串的相似度。"""
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()

def calculate_replacement_probability(similarity: float) -> float:
    """根据相似度计算替换的概率。"""
    if similarity <= 0.4:
        return 0.0
    elif similarity >= 0.9:
        return 1.0
    elif 0.4 < similarity <= 0.6:
        probability = 3.5 * similarity - 1.4
        return max(0.0, probability)
    else:  # 0.6 < similarity < 0.9
        probability = similarity + 0.1
        return min(1.0, max(0.0, probability))

# --- Prompt模板定义 ---
def init_prompt():
    # HFC Group Chat Prompt (保留HFC群聊模板)
    group_prompt_hfc = """
<identity>
    <bot_name>你的名字是{bot_name}。</bot_name>
    <personality_profile>{prompt_personality}</personality_profile>
</identity>
<group_nicknames>
{nickname_info_hfc}
</group_nicknames>
<knowledge_base>
    <structured_information>{extra_info}</structured_information>
    <social_relationships>{relation_prompt_hfc}</social_relationships>
</knowledge_base>
<recent_internal_state>
    <previous_thoughts_and_actions>{last_loop_prompt_hfc}</previous_thoughts_and_actions>
    <recent_reply_history>{cycle_info_block_hfc}</recent_reply_history>
    <current_schedule>你现在正在做的事情是：{schedule_info}</current_schedule>
    <current_mood>你现在{mood_info}</current_mood>
</recent_internal_state>
<live_chat_context>
    <timestamp>现在是{time_now}。</timestamp>
    <chat_log>你正在上网，和qq群里的网友们聊天，以下是正在进行的聊天内容：
{chat_observe_info_hfc}</chat_log>
</live_chat_context>
<thinking_guidance>
请仔细阅读当前聊天内容，分析讨论话题和群成员关系，分析你刚刚发言和别人对你的发言的反应，思考你要不要回复或发言。然后思考你是否需要使用函数工具。
注意耐心：
  -请特别关注对话的自然流转和对方的输入状态。如果感觉对方可能正在打字或思考，或者其发言明显未结束（比如话说到一半），请耐心等待，避免过早打断或急于追问。
  -如果你发送消息后对方没有立即回应，请优先考虑对方是否正在忙碌或话题已自然结束，内心想法应倾向于“耐心等待”或“思考对方是否在忙”，而非立即追问，除非追问非常必要且不会打扰。
思考并输出你真实的内心想法。
</thinking_guidance>
<output_requirements_for_inner_thought>
1. 根据聊天内容生成你的内心想法，{hf_do_next}
   - 如果你决定回复或发言，必须**明确写出**你准备发送的消息的具体内容是什么。
   - 如果你觉得不想继续专注在这个群聊（例如感到无聊、疲惫、话题不感兴趣或想去关注其他事情），你的内心想法应该明确表达出这种倾向（例如：“不太想继续聊了。”或“感觉没什么好说的了”或“有点累了，想结束和这个群的专注互动了。”）
   - 如果你现在很忙或没时间，也可以倾向选择不回复。
2. 不要分点、不要使用表情符号
3. 避免多余符号(冒号、引号、括号等)
4. 语言简洁自然，不要浮夸
5. 不要把注意力放在别人发的表情包上，它们只是一种辅助表达方式
6. 注意分辨群里谁在跟谁说话，你不一定是当前聊天的主角，消息中的“你”不一定指的是你（{bot_name}），也可能是别人
7. 默认使用中文
</output_requirements_for_inner_thought>
<tool_usage_instructions>
1. 输出想法后考虑是否需要使用工具
2. 工具可获取信息或执行操作
3. 如需处理消息或回复，请使用工具。
</tool_usage_instructions>
"""
    Prompt(group_prompt_hfc, "sub_heartflow_prompt_before") # HFC群聊模板名保持

    # HFC Private Chat Prompt (原sub_heartflow_prompt_private_before，现已废弃，PFC接管私聊)
    # 我们不再注册这个模板，或者可以注释掉它
    # private_prompt_hfc_old = """..."""
    # Prompt(private_prompt_hfc_old, "sub_heartflow_prompt_private_before")

    # PFC Private Chat Prompt (新增，用于PFC私聊)
    private_prompt_pfc = """
{extra_info}
{relation_prompt_pfc}
你的名字是{bot_name},{prompt_personality}

<目标回顾>
你当前的主要对话目标是：
{pfc_current_goals_formatted_string}
</目标回顾>

<你的上一次思考与行动回顾>
{pfc_last_thought_and_action_prompt}
</你的上一次思考与行动回顾>

现在是{time_now}，你正在上网，和 {private_chat_partner_name_pfc} 私聊。
以下是你们当前的聊天内容：
{current_chat_observe_info_pfc}

你还回想起一些过去相关的聊天片段：
{retrieved_historical_chat_str_pfc}
你现在正在做的事情是：{schedule_info}
你现在{mood_info}

请仔细阅读以上所有信息，特别是当前的聊天内容、你的目标、以及你之前的思考和行动。
思考并输出你此刻真实的内心想法。这个想法应该是对当前局面的分析、你的感受、或者你接下来想在对话中达成的细微意图。
输出要求：
1. 根据聊天内容和你当前的目标生成你的内心想法，{hf_do_next}
2. 不要分点、不要使用表情符号
3. 避免多余符号(冒号、引号、括号等)
4. 语言简洁自然，不要浮夸
5. 如果你刚发言，对方没有回复你，请谨慎回复，你的想法可以是如何等待或者观察。
6. 不要把注意力放在别人发的表情包上，它们只是一种辅助表达方式
工具使用说明：
1. 输出想法后考虑是否需要使用工具
2. 工具可获取信息或执行操作
3. 如需处理消息或回复，请使用工具。
"""
    Prompt(private_prompt_pfc, "sub_heartflow_prompt_private_pfc")

    # Last Loop Prompt (通用)
    last_loop_t = """
刚刚你的内心想法是：{current_thinking_info}
{if_replan_prompt}
"""
    Prompt(last_loop_t, "last_loop")

init_prompt() # 模块加载时注册Prompt

class SubMind:
    def __init__(self,
                 subheartflow_id: str,
                 # HFC specific
                 chat_state: Optional[ChatStateInfo] = None,
                 observations: Optional[List[ChattingObservation]] = None,
                 # PFC specific
                 pfc_conversation_info: Optional[ConversationInfo] = None,
                 pfc_observation_info: Optional[ObservationInfo] = None,
                 pfc_chat_stream: Optional[ChatStream] = None
                ):
        self.stream_id = subheartflow_id
        self.log_prefix = f"[{chat_manager.get_stream_name(self.stream_id) or self.stream_id}] SubMind "

        self.is_pfc_context = pfc_conversation_info is not None and pfc_observation_info is not None

        if self.is_pfc_context:
            self.pfc_conversation_info = pfc_conversation_info
            self.pfc_observation_info = pfc_observation_info
            self.pfc_chat_stream = pfc_chat_stream # PFC会传入ChatStream实例
            self.chat_state = None
            self.observations = []
            logger.debug(f"{self.log_prefix}以PFC模式初始化。")
        else: # HFC Mode
            self.chat_state = chat_state
            self.observations = observations if observations is not None else []
            self.pfc_conversation_info = None
            self.pfc_observation_info = None
            self.pfc_chat_stream = None
            logger.debug(f"{self.log_prefix}以HFC模式初始化。")

        self.current_mind: str = ""
        self.past_mind: List[str] = []
        self.structured_info: List[Dict[str, Any]] = []
        self.structured_info_str: str = ""
        self._update_structured_info_str()

        self.llm_model = LLMRequest(
            model=global_config.model.sub_heartflow,
            temperature=global_config.model.sub_heartflow["temp"],
            request_type="sub_heart_flow",
        )
        self.knowledge_retrieval_steps = [
            {"name": "latest_1_msg", "limit": 1, "relevance_threshold": 0.075},
            {"name": "latest_2_msgs", "limit": 2, "relevance_threshold": 0.065},
            {"name": "short_window_3_msgs", "limit": 3, "relevance_threshold": 0.050},
            {"name": "medium_window_8_msgs", "limit": 8, "relevance_threshold": 0.030},
        ]
        self.last_active_time: Optional[float] = None

    def _update_structured_info_str(self):
        if not self.structured_info:
            self.structured_info_str = ""
            return
        lines = ["【信息】"]
        for item in self.structured_info:
            type_str = item.get("type", "未知类型")
            content_str = item.get("content", "")
            if type_str == "info": lines.append(f"刚刚: {content_str}")
            elif type_str == "memory": lines.append(f"{content_str}")
            elif type_str == "comparison_result": lines.append(f"数字大小比较结果: {content_str}")
            elif type_str == "time_info": lines.append(f"{content_str}")
            elif type_str == "lpmm_knowledge": lines.append(f"你知道：{content_str}")
            else: lines.append(f"{type_str}的信息: {content_str}")
        self.structured_info_str = "\n".join(lines)
        # logger.debug(f"{self.log_prefix} 更新 structured_info_str: \n{self.structured_info_str}") # 日志可能过于频繁

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]], tool_instance: ToolUser):
        new_structured_items = []
        for tool_call in tool_calls:
            try:
                result = await tool_instance._execute_tool_call(tool_call) # type: ignore
                if result:
                    new_item = {
                        "type": result.get("type", "unknown_tool_result"),
                        "id": result.get("id", f"tool_result_{time.time()}"),
                        "content": result.get("content", ""),
                        "ttl": 3,
                    }
                    new_structured_items.append(new_item)
            except Exception as tool_e:
                logger.error(f"{self.log_prefix}工具执行失败: {tool_e}", exc_info=True)
        if new_structured_items:
            self.structured_info.extend(new_structured_items)
            logger.debug(f"{self.log_prefix}工具调用收集到新的结构化信息: {safe_json_dumps(new_structured_items, ensure_ascii=False)}")
            self._update_structured_info_str()

    def update_current_mind(self, response: str):
        if self.current_mind:
            self.past_mind.append(self.current_mind)
            if len(self.past_mind) > 5:
                self.past_mind.pop(0)
        self.current_mind = response

    async def do_thinking_before_reply(
        self,
        pfc_done_actions: Optional[List[Dict[str, Any]]] = None,
        previous_pfc_thought: Optional[str] = None,
        retrieved_historical_chat_str_pfc: Optional[str] = None,
        history_cycle: Optional[List[CycleInfo]] = None,
        tool_calls_str: Optional[str] = None,
        pass_mind: Optional[str] = None
    ) -> Tuple[str, List[str], Optional[str], Dict[str, Any]]:

        self.last_active_time = time.time()

        # 1. 清理 structured_info
        processed_info_to_keep = []
        for item in self.structured_info:
            if not tool_calls_str and not pass_mind:
                if item.get("type") == "lpmm_knowledge": item["ttl"] = 0
                else: item["ttl"] -= 1
            if item["ttl"] > 0: processed_info_to_keep.append(item)
            else: logger.debug(f"{self.log_prefix} 移除过期的structured_info项: {item.get('id', '未知ID')}")
        self.structured_info = processed_info_to_keep
        # structured_info_str 会在知识/记忆检索后，或工具执行后更新

        # 2. 准备通用上下文变量
        individuality = Individuality.get_instance()
        prompt_personality = individuality.get_prompt(x_person=2, level=3)
        bot_name = individuality.name
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        current_schedule_info = bot_schedule.get_current_num_task(num=1, time_info=False) or "当前没有什么特别的安排。"
        local_random = random.Random()
        current_minute = int(time.strftime("%M"))
        local_random.seed(current_minute + sum(ord(c) for c in self.stream_id[:5]))
        hf_options = [
            ("可以参考之前的想法，在原来想法的基础上继续思考，但是也要注意话题的推进，**不要在一个话题上停留太久或揪着一个话题不放，除非你觉得真的有必要**",0.3,),
            ("可以参考之前的想法，在原来的想法上**尝试新的话题**", 0.3),
            ("不要太深入，注意话题的推进，**不要在一个话题上停留太久或揪着一个话题不放，除非你觉得真的有必要**", 0.2),
            ("进行深入思考，但是注意话题的推进，**不要在一个话题上停留太久或揪着一个话题不放，除非你觉得真的有必要**",0.2,),
            ("可以参考之前的想法继续思考，并结合你自身的人设，知识，信息，回忆等等", 0.08), ]
        hf_do_next = local_random.choices([opt[0] for opt in hf_options], weights=[opt[1] for opt in hf_options], k=1)[0]

        # ---- 初始化特定于上下文的变量 ----
        mood_info: str = "心情平静"
        # PFC
        private_chat_partner_name_pfc_val: str = "对方"
        current_chat_observe_info_pfc_val: str = "当前没有聊天内容。"
        if_historical_recall_exists_str_pfc_val: str = ""
        formatted_historical_recall_pfc_val: str = ""
        pfc_last_thought_and_action_prompt_str_val: str = "这是你本次对话的第一次深入思考。\n"
        pfc_current_goals_formatted_string_str_val: str = "当前没有明确的对话目标。"
        relation_prompt_pfc_val: str = f"你和{private_chat_partner_name_pfc_val}的关系很普通。"
        # HFC
        chat_observe_info_hfc_val: str = "当前没有聊天内容。"
        is_group_chat_hfc_val: bool = False
        chat_target_name_hfc_val: str = "对方"
        last_loop_prompt_hfc_val: str = ""
        cycle_info_block_hfc_val: str = "\n【近期回复历史】\n(最近没有连续文本回复)\n"
        nickname_info_hfc_val: str = ""
        relation_prompt_hfc_val: str = "关系未知。"

        # ---- 根据 self.is_pfc_context 填充特定上下文 ----
        if self.is_pfc_context:
            # --- PFC 上下文获取和准备 ---
            if not self.pfc_observation_info or not self.pfc_conversation_info:
                logger.error(f"{self.log_prefix} PFC模式但缺少PFC上下文对象！")
                return "(PFC上下文错误)", self.past_mind, None, {}

            mood_info = self.pfc_conversation_info.current_emotion_text or mood_info
            private_chat_partner_name_pfc_val = self.pfc_observation_info.private_name or private_chat_partner_name_pfc_val
            current_chat_observe_info_pfc_val = self.pfc_observation_info.chat_history_str or current_chat_observe_info_pfc_val

            if retrieved_historical_chat_str_pfc and retrieved_historical_chat_str_pfc.strip():
                formatted_historical_recall_pfc_val = retrieved_historical_chat_str_pfc
                if_historical_recall_exists_str_pfc_val = ""

            _previous_thought_for_prompt = previous_pfc_thought if previous_pfc_thought else "你之前没有特别的想法。"
            _if_replan_prompt_for_pfc = "这是你针对当前对话的第一次深入思考和行动规划。\n"
            if pfc_done_actions and len(pfc_done_actions) > 0:
                last_action = pfc_done_actions[-1]
                action_type = last_action.get("action", "未知动作")
                plan_reason = last_action.get("plan_reason", "未知原因")
                status = last_action.get("status", "未知状态")
                final_reason = last_action.get("final_reason", "")
                _if_replan_prompt_for_pfc = (
                    f"基于这个想法（或之前的决策），你上次的行动是：'{action_type}' (当时规划的原因: '{plan_reason}')，"
                    f"该行动的最终状态是：'{status}'"
                    f"{f'，备注信息：{final_reason}' if final_reason and final_reason != plan_reason else ''}。\n"
                )
            pfc_last_thought_and_action_prompt_str_val = (await global_prompt_manager.get_prompt_async("last_loop")).format(
                current_thinking_info=_previous_thought_for_prompt,
                if_replan_prompt=_if_replan_prompt_for_pfc
            )

            if self.pfc_conversation_info.goal_list:
                goals_text_parts = []
                for goal_item in self.pfc_conversation_info.goal_list[-3:]:
                    goal = goal_item.get('goal', '未知目标')
                    reason = goal_item.get('reasoning', '无具体原因')
                    goals_text_parts.append(f"- 目标: {goal} (原因: {reason})")
                if goals_text_parts:
                    pfc_current_goals_formatted_string_str_val = "\n".join(goals_text_parts)

            if self.pfc_conversation_info.person_id:
                relation_prompt_pfc_val = await relationship_manager.build_relationship_info(self.pfc_conversation_info.person_id, is_id=True)
            if not relation_prompt_pfc_val.strip():
                relation_prompt_pfc_val = f"你和{private_chat_partner_name_pfc_val}的关系很普通。"

            if not tool_calls_str and not pass_mind:
                chat_words_pfc = set(jieba.cut(current_chat_observe_info_pfc_val))
                keywords_pfc = [word for word in chat_words_pfc if len(word) > 1][:5]
                logger.debug(f"{self.log_prefix} PFC模式下提取的关键词: {keywords_pfc}")
                existing_topics_pfc = {item["id"] for item in self.structured_info if item["type"] == "memory"}
                filtered_keywords_pfc = [k for k in keywords_pfc if k not in existing_topics_pfc]
                if filtered_keywords_pfc:
                    related_memory_pfc = await HippocampusManager.get_instance().get_memory_from_topic(
                        valid_keywords=filtered_keywords_pfc, max_memory_num=3, max_memory_length=2, max_depth=3
                    )
                    if related_memory_pfc:
                        for topic, memory in related_memory_pfc:
                            self.structured_info.append({"type": "memory", "id": topic, "content": memory, "ttl": 3})
                        logger.debug(f"{self.log_prefix} PFC模式下，添加了 {len(related_memory_pfc)} 条记忆到structured_info。")

                raw_knowledge_str_pfc = qa_manager.get_knowledge(current_chat_observe_info_pfc_val)
                if raw_knowledge_str_pfc:
                    knowledge_content_pfc, max_relevance_pfc = parse_knowledge_and_get_max_relevance(raw_knowledge_str_pfc)
                    if max_relevance_pfc >= 0.03:
                        self.structured_info.append({
                            "type": "lpmm_knowledge", "id": f"lpmm_knowledge_pfc_{time.time()}",
                            "content": knowledge_content_pfc, "ttl": 1,
                        })
                        logger.debug(f"{self.log_prefix} PFC模式下，添加了相关性为 {max_relevance_pfc:.2f} 的知识到structured_info。")
            self._update_structured_info_str()

        else: # ---- HFC 上下文获取和准备 (仅群聊) ----
            if not self.observations or not self.chat_state or not hasattr(self.observations[0], 'is_group_chat'):
                logger.error(f"{self.log_prefix} HFC模式但缺少HFC上下文对象！")
                return "(HFC上下文错误)", self.past_mind, None, {}

            hfc_observation: ChattingObservation = self.observations[0] # type: ignore
            is_group_chat_hfc_val = hfc_observation.is_group_chat

            if not is_group_chat_hfc_val: # 如果不是HFC群聊（即HFC私聊，现已废弃），则报错或返回
                logger.error(f"{self.log_prefix} HFC模式下检测到私聊上下文，但HFC私聊逻辑已被移除。请检查调用流程或配置文件。")
                return "(HFC私聊路径已废弃)", self.past_mind, None, {}

            # --- HFC 群聊逻辑开始 ---
            chat_observe_info_hfc_val = hfc_observation.get_observe_info()
            mood_info = self.chat_state.mood

            _previous_mind_for_hfc = self.current_mind if self.current_mind else ""
            if pass_mind: _previous_mind_for_hfc = pass_mind

            _if_replan_prompt_for_hfc = ""
            if history_cycle and history_cycle[-1]:
                last_cycle = history_cycle[-1]
                _action_type_hfc = last_cycle.action_type
                _reasoning_hfc = last_cycle.reasoning
                _is_replan_hfc = last_cycle.replanned
                if _is_replan_hfc:
                    _if_replan_prompt_for_hfc = f"但是你有了上述想法之后，有了新消息，你决定重新思考后，你做了：{_action_type_hfc}\n因为：{_reasoning_hfc}\n"
                else:
                    _if_replan_prompt_for_hfc = f"出于这个想法，你刚才做了：{_action_type_hfc}\n因为：{_reasoning_hfc}\n"

            if tool_calls_str:
                _if_replan_prompt_for_hfc = f"出于这个想法，你刚刚调用了 {tool_calls_str} 工具，获取的内容在 <structured_information> 中。"
                if history_cycle and history_cycle[-1]:
                     _last_action_hfc_tool = history_cycle[-1].action_type
                     _last_reasoning_hfc_tool = history_cycle[-1].reasoning
                     _if_replan_prompt_for_hfc += f"而你上一次行动为：{_last_action_hfc_tool}\n因为：{_last_reasoning_hfc_tool}\n"

            if _previous_mind_for_hfc or _if_replan_prompt_for_hfc:
                 last_loop_prompt_hfc_val = (await global_prompt_manager.get_prompt_async("last_loop")).format(
                    current_thinking_info=_previous_mind_for_hfc,
                    if_replan_prompt=_if_replan_prompt_for_hfc
                )

            if history_cycle:
                _consecutive_text_replies_hfc = 0; _responses_for_prompt_hfc = []; _recent_active_cycles_hfc = []
                for _cycle_hfc in reversed(history_cycle):
                    if _cycle_hfc.action_taken: _recent_active_cycles_hfc.append(_cycle_hfc)
                    if len(_recent_active_cycles_hfc) == 3: break
                for _cycle_hfc_active in _recent_active_cycles_hfc:
                    if _cycle_hfc_active.action_type == "text_reply":
                        _consecutive_text_replies_hfc +=1
                        _response_text_hfc = _cycle_hfc_active.response_info.get("response_text", [])
                        _formatted_response_hfc = "[空回复]" if not _response_text_hfc else " ".join(_response_text_hfc)
                        _responses_for_prompt_hfc.append(_formatted_response_hfc)
                    else: break
                if _consecutive_text_replies_hfc >=3: cycle_info_block_hfc_val = f'你已经连续回复了三条消息（最近: "{_responses_for_prompt_hfc[0]}"，第二近: "{_responses_for_prompt_hfc[1]}"，第三近: "{_responses_for_prompt_hfc[2]}"）。你回复的有点多了，请注意'
                elif _consecutive_text_replies_hfc == 2: cycle_info_block_hfc_val = f'你已经连续回复了两条消息（最近: "{_responses_for_prompt_hfc[0]}"，第二近: "{_responses_for_prompt_hfc[1]}"），请注意'
                elif _consecutive_text_replies_hfc == 1: cycle_info_block_hfc_val = f'你刚刚已经回复一条消息（内容: "{_responses_for_prompt_hfc[0]}"）'
                if cycle_info_block_hfc_val and cycle_info_block_hfc_val != "\n【近期回复历史】\n(最近没有连续文本回复)\n":
                    cycle_info_block_hfc_val = f"\n【近期回复历史】\n{cycle_info_block_hfc_val}\n"

            if not tool_calls_str and not pass_mind:
                chat_words_hfc = set(jieba.cut(chat_observe_info_hfc_val))
                keywords_hfc = [word for word in chat_words_hfc if len(word) > 1][:5]
                logger.debug(f"{self.log_prefix} HFC群聊模式下提取的关键词: {keywords_hfc}")
                existing_topics_hfc = {item["id"] for item in self.structured_info if item["type"] == "memory"}
                filtered_keywords_hfc = [k for k in keywords_hfc if k not in existing_topics_hfc]
                if filtered_keywords_hfc:
                    related_memory_hfc = await HippocampusManager.get_instance().get_memory_from_topic(
                        valid_keywords=filtered_keywords_hfc, max_memory_num=3, max_memory_length=2, max_depth=3
                    )
                    if related_memory_hfc:
                        for topic, memory_text_hfc in related_memory_hfc:
                            self.structured_info.append({"type": "memory", "id": topic, "content": memory_text_hfc, "ttl": 3})
                        logger.debug(f"{self.log_prefix} HFC群聊模式下，添加了 {len(related_memory_hfc)} 条记忆到structured_info。")

                _final_knowledge_to_add_hfc = None
                if hfc_observation: # 确保hfc_observation存在
                    for step_config in self.knowledge_retrieval_steps: # 阶梯式检索
                        _step_name_hfc = step_config["name"]; _limit_hfc = step_config["limit"]; _threshold_hfc = step_config["relevance_threshold"]
                        try:
                            _context_messages_dicts_hfc = get_raw_msg_before_timestamp_with_chat(chat_id=hfc_observation.chat_id, timestamp=time.time(), limit=_limit_hfc)
                            if _context_messages_dicts_hfc:
                                _current_context_text_hfc = await build_readable_messages(messages=_context_messages_dicts_hfc, timestamp_mode="lite")
                                if _current_context_text_hfc:
                                    _raw_knowledge_str_hfc_step = qa_manager.get_knowledge(_current_context_text_hfc)
                                    if _raw_knowledge_str_hfc_step:
                                        _knowledge_content_hfc, _max_relevance_hfc = parse_knowledge_and_get_max_relevance(_raw_knowledge_str_hfc_step)
                                        if _max_relevance_hfc >= _threshold_hfc: _final_knowledge_to_add_hfc = _knowledge_content_hfc; break
                        except Exception as e_step_hfc: logger.error(f"{self.log_prefix} HFC阶梯检索阶段 '{_step_name_hfc}' 发生错误: {e_step_hfc}")
                    if not _final_knowledge_to_add_hfc and chat_observe_info_hfc_val: # 完整窗口回退
                        _raw_knowledge_str_hfc_full = qa_manager.get_knowledge(chat_observe_info_hfc_val)
                        if _raw_knowledge_str_hfc_full:
                            _knowledge_content_hfc_full, _ = parse_knowledge_and_get_max_relevance(_raw_knowledge_str_hfc_full)
                            _final_knowledge_to_add_hfc = _knowledge_content_hfc_full
                    if _final_knowledge_to_add_hfc:
                        self.structured_info.append({"type": "lpmm_knowledge", "id": f"lpmm_knowledge_hfc_{time.time()}", "content": _final_knowledge_to_add_hfc, "ttl": 1})
                        logger.debug(f"{self.log_prefix} HFC群聊模式下，添加了知识 '{str(_final_knowledge_to_add_hfc)[:50]}...' 到structured_info。")
            self._update_structured_info_str()

            if is_group_chat_hfc_val: # HFC群聊才需要nickname_info
                _hfc_chat_stream_for_nick = None
                if self.observations and self.observations[0] and self.observations[0].chat_id: # type: ignore
                    _hfc_chat_stream_for_nick = chat_manager.get_stream(self.observations[0].chat_id) # type: ignore
                if _hfc_chat_stream_for_nick:
                    message_list_for_nicknames = get_raw_msg_before_timestamp_with_chat(
                        chat_id=_hfc_chat_stream_for_nick.stream_id,
                        timestamp=time.time(),
                        limit=global_config.chat.observation_context_size,
                    )
                    nickname_info_hfc_val = await nickname_manager.get_nickname_prompt_injection(
                        _hfc_chat_stream_for_nick, message_list_for_nicknames
                    )
                else: nickname_info_hfc_val = "[获取群成员绰号信息失败(stream获取失败)]"

            if self.observations and self.observations[0] and hasattr(self.observations[0], 'person_list'): # type: ignore
                hfc_person_list = self.observations[0].person_list # type: ignore
                for person in hfc_person_list:
                    relation_prompt_hfc_val += await relationship_manager.build_relationship_info(person, is_id=True) # type: ignore
            if not relation_prompt_hfc_val.strip(): relation_prompt_hfc_val = "你和大家关系一般。"
            # --- HFC 群聊逻辑结束 ---

        # ---- 3. 准备工具 (通用) ----
        tool_instance = ToolUser()
        tools = tool_instance._define_tools()

        # ---- 4. 构建最终提示词 ----
        prompt = ""
        if self.is_pfc_context:
            template_name = "sub_heartflow_prompt_private_pfc"
            prompt = (await global_prompt_manager.get_prompt_async(template_name)).format(
                extra_info=self.structured_info_str,
                relation_prompt_pfc=relation_prompt_pfc_val,
                bot_name=bot_name,
                prompt_personality=prompt_personality,
                pfc_current_goals_formatted_string=pfc_current_goals_formatted_string_str_val,
                pfc_last_thought_and_action_prompt=pfc_last_thought_and_action_prompt_str_val,
                time_now=time_now,
                private_chat_partner_name_pfc=private_chat_partner_name_pfc_val,
                current_chat_observe_info_pfc=current_chat_observe_info_pfc_val,
                if_historical_recall_exists=if_historical_recall_exists_str_pfc_val,
                retrieved_historical_chat_str_pfc=formatted_historical_recall_pfc_val,
                end_if_historical_recall_exists=if_historical_recall_exists_str_pfc_val.replace("<!--", "<!-- /"),
                schedule_info=current_schedule_info,
                mood_info=mood_info,
                hf_do_next=hf_do_next
            )
            if not formatted_historical_recall_pfc_val: # 清理PFC条件标记
                prompt = re.sub(r".*?", "", prompt, flags=re.DOTALL)
            else:
                prompt = prompt.replace("", "").replace("", "")
        else: # HFC 场景 (只剩下群聊)
            template_name = "sub_heartflow_prompt_before" # HFC群聊模板
            prompt = (await global_prompt_manager.get_prompt_async(template_name)).format(
                extra_info=self.structured_info_str,
                prompt_personality=prompt_personality,
                relation_prompt_hfc=relation_prompt_hfc_val, # HFC变量
                bot_name=bot_name,
                time_now=time_now,
                chat_observe_info_hfc=chat_observe_info_hfc_val, # HFC变量
                mood_info=mood_info, # HFC情绪
                hf_do_next=hf_do_next,
                last_loop_prompt_hfc=last_loop_prompt_hfc_val, # HFC变量
                cycle_info_block_hfc=cycle_info_block_hfc_val, # HFC变量
                nickname_info_hfc=nickname_info_hfc_val, # HFC变量
                schedule_info=current_schedule_info,
            )

        # ---- 5. 执行LLM请求并处理响应 (通用) ----
        content = ""
        _reasoning_content_llm = ""
        llm_tool_calls_from_response = None
        tool_calls_str_output = tool_calls_str

        try:
            logger.debug(f"{self.log_prefix} SubMind 最终发送给LLM的Prompt (is_pfc={self.is_pfc_context}): \n------\n{prompt}\n------")
            response_text_llm, _reasoning_content_llm, llm_tool_calls_from_response = await self.llm_model.generate_response_tool_async(
                prompt=prompt, tools=tools
            )
            logger.debug(f"{self.log_prefix} SubMind 从LLM收到的原始响应: '{str(response_text_llm)[:200]}...'")
            content = response_text_llm if response_text_llm else ""

            if llm_tool_calls_from_response:
                success_tc, valid_tool_calls_tc, error_msg_tc = process_llm_tool_calls(
                    llm_tool_calls_from_response, log_prefix=f"{self.log_prefix} SubMind "
                )
                if success_tc and valid_tool_calls_tc:
                    tool_calls_str_output = ", ".join(
                        [call.get("function", {}).get("name", "未知工具") for call in valid_tool_calls_tc]
                    )
                    logger.info(f"{self.log_prefix} SubMind 模型请求调用{len(valid_tool_calls_tc)}个工具: {tool_calls_str_output}")
                    await self._execute_tool_calls(valid_tool_calls_tc, tool_instance)
                elif not success_tc:
                    logger.warning(f"{self.log_prefix} SubMind 处理工具调用时出错: {error_msg_tc}")
                    tool_calls_str_output = None
            else:
                logger.info(f"{self.log_prefix} SubMind 未使用工具")
                tool_calls_str_output = None
        except Exception as e:
            logger.error(f"{self.log_prefix} SubMind 执行LLM请求或处理响应时出错: {e}", exc_info=True)
            content = "(SubMind思考过程中出现错误)"
            tool_calls_str_output = None

        # ---- 6. 应用概率性去重 (通用) ----
        if global_config.chat.allow_remove_duplicates:
            comparison_base_mind = ""
            if self.is_pfc_context:
                comparison_base_mind = previous_pfc_thought if previous_pfc_thought else self.current_mind
            else: # HFC
                comparison_base_mind = pass_mind if pass_mind else self.current_mind

            if comparison_base_mind and content:
                new_content_before_dedup = content
                try:
                    similarity = calculate_similarity(comparison_base_mind, new_content_before_dedup)
                    replacement_prob = calculate_replacement_probability(similarity)
                    logger.debug(f"{self.log_prefix} 新旧想法相似度: {similarity:.2f}, 替换概率: {replacement_prob:.2f} (比较对象: '{comparison_base_mind[:30]}...', 新想法: '{new_content_before_dedup[:30]}...')")
                    if random.random() < replacement_prob:
                        _yu_qi_ci_liebiao_dedup = ["嗯", "哦", "啊", "唉", "哈", "唔"]
                        _zhuan_jie_liebiao_dedup = ["但是", "不过", "然而", "可是", "只是", "然后", "接着", "此外", "而且", "另外"]
                        if similarity == 1.0:
                            logger.debug(f"{self.log_prefix} 想法完全重复 (相似度 1.0)，执行特殊处理...")
                            if len(new_content_before_dedup) > 1:
                                _split_point = max(1, len(new_content_before_dedup) // 2 + random.randint(-len(new_content_before_dedup) // 4, len(new_content_before_dedup) // 4))
                                _truncated_content = new_content_before_dedup[:_split_point]
                            else: _truncated_content = new_content_before_dedup
                            _yu_qi_ci = random.choice(_yu_qi_ci_liebiao_dedup)
                            _zhuan_jie_ci = random.choice(_zhuan_jie_liebiao_dedup)
                            content = f"{_yu_qi_ci}{_zhuan_jie_ci}，{_truncated_content}"
                            logger.debug(f"{self.log_prefix} 想法重复，特殊处理后: {content}")
                        else:
                            logger.debug(f"{self.log_prefix} 执行概率性去重 (概率: {replacement_prob:.2f})...")
                            _matcher = difflib.SequenceMatcher(None, comparison_base_mind, new_content_before_dedup)
                            _deduplicated_parts = []
                            _last_match_end_in_b = 0
                            for _i_match, _j_match, _n_match in _matcher.get_matching_blocks():
                                if _last_match_end_in_b < _j_match: _deduplicated_parts.append(new_content_before_dedup[_last_match_end_in_b:_j_match])
                                _last_match_end_in_b = _j_match + _n_match
                            _deduplicated_content = "".join(_deduplicated_parts).strip()
                            if _deduplicated_content:
                                _prefix_str_dedup = ""
                                if random.random() < 0.3: _prefix_str_dedup += random.choice(_yu_qi_ci_liebiao_dedup)
                                if random.random() < 0.7: _prefix_str_dedup += random.choice(_zhuan_jie_liebiao_dedup)
                                if _prefix_str_dedup: content = f"{_prefix_str_dedup}，{_deduplicated_content}"
                                else: content = _deduplicated_content
                                logger.debug(f"{self.log_prefix} 去重并处理后: {content}")
                            else:
                                logger.warning(f"{self.log_prefix} 去重后内容为空，保留LLM输出: {new_content_before_dedup}")
                                content = new_content_before_dedup
                    else: logger.debug(f"{self.log_prefix} 未执行概率性去重 (概率: {replacement_prob:.2f})")
                except Exception as e_dedup:
                    logger.error(f"{self.log_prefix} 应用概率性去重时出错: {e_dedup}", exc_info=True)
                    content = new_content_before_dedup
            else: logger.debug(f"{self.log_prefix} 无先前想法或当前想法为空，跳过概率性去重。")

        # ---- 7. 更新思考状态并返回结果 ----
        if not content:
            default_thought = "(PFC SubMind: 不知道该想些什么...)" if self.is_pfc_context else "(HFC SubMind: 不知道该想些什么...)"
            content = default_thought
            logger.warning(f"{self.log_prefix} LLM返回空结果或处理后为空，思考失败。")

        logger.info(f"{self.log_prefix} SubMind 最终思考结果 (is_pfc={self.is_pfc_context}): '{content[:100]}...'")
        self.update_current_mind(content)

        structured_info_dict = {f"item_{i}": info_item for i, info_item in enumerate(self.structured_info)}
        return self.current_mind, self.past_mind, tool_calls_str_output, structured_info_dict