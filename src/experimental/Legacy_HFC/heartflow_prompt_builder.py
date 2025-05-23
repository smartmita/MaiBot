import random
import time
from typing import Union, Optional, Deque, Dict, Any
from ...config.config import global_config
from src.common.logger_manager import get_logger
from ...individuality.individuality import Individuality
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.utils.utils import get_embedding
from src.common.database import db
from src.chat.utils.utils import get_recent_group_speaker
from src.manager.mood_manager import mood_manager
from src.chat.memory_system.Hippocampus import HippocampusManager
from .schedule.schedule_generator import bot_schedule
from src.chat.knowledge.knowledge_lib import qa_manager
from src.experimental.profile.profile_manager import profile_manager
from src.chat.focus_chat.expressors.exprssion_learner import expression_learner
import traceback
from .heartFC_Cycleinfo import CycleInfo

logger = get_logger("prompt")


def init_prompt():
    Prompt(
        """
你是{bot_name}。

{chat_target}
你正在{chat_target_2}。

{profile_info}

{chat_talking_prompt}

<你的想法>
看到以上聊天记录，你刚刚在想：
{current_mind_info}
因为上述想法，你决定发言。
</你的想法>

<任务指令>
现在你想要回复或参与讨论。
请仔细阅读聊天记录和你的想法，**把你的想法组织成合适且简短的语言**，然后发一条消息。
</任务指令>

<回复风格与约束>
回复时请注意：
1.  整体风格可以自然随意、平和简短，就像群聊里的真人一样。
2.  请注意把握聊天内容，**避免**超出你内心想法的范围。
3.  消息应尽量简短。{reply_style2}
4.  请一次只回复一个话题，不要同时回复多个人。{prompt_ger}
5.  {reply_style1}
6.  请使用中文，不要刻意突出自身学科背景。
7.  注意只输出消息内容，不要去主动讨论或评价别人发的表情包，它们只是一种辅助表达方式。
8.  可以参考并自然融入以下学习到的语言和句法习惯（如果情景合适）：
    {style_habbits}
    {grammar_habbits}
9.  {moderation_prompt}
10. 注意：回复不要输出多余内容 (包括前后缀，冒号和引号，括号，表情包，戳一戳，at或 @，Markdown格式等，这些内容已经在别的层级被处理了，所以你只需要输出纯消息文本)。
</回复风格与约束>""",
        "heart_flow_prompt",
    )

    Prompt(
        """
你有以下信息可供参考：
{structured_info}
以上的信息是你获取到的消息，或许可以帮助你更好地回复。
""",
        "info_from_tools",
    )

    # Planner提示词 - 修改为要求 JSON 输出
    Prompt(
    """
<planner_task_definition>
你将扮演{bot_name}在QQ群聊中进行专注聊天。根据以下信息，决定{bot_name}是否以及如何参与对话。
</planner_task_definition>

<contextual_information>

<profile_info>
{profile_info}
</profile_info>

   
<chat_log>
{chat_content_block}
</chat_log>

    
<current_thoughts>
{current_mind_block}
</current_thoughts>

<recent_action_history>
{cycle_info_block}
</recent_action_history>

</contextual_information>

<decision_framework>
<guidance>
综合分析聊天内容、{bot_name}的内心想法以及近期互动历史，参考以下决策以及解释，选择一个最合适的行动。目标是让{bot_name}的参与自然且符合群聊社交节奏。
注意**避免**在无人回应{bot_name}时连续发言，除非{bot_name}有新的重要补充
</guidance>

<available_actions>
可选行动以及解释：
"no_reply": "不发消息，当{bot_name}的内心想法表示不想发言/出错/想法为空/最近发言未获回应且无新发言意图时选择"
"text_reply": "发送文本消息, 若{bot_name}内心想法有实质内容且想表达，并且时机适合时选择。可附带表情包和@某人或戳一戳某人。注意**不要**在{bot_name}内心想法表达不想回复时选择"
"emoji_reply": "单独发一个表情包，若情景适合用表情回应，或{bot_name}想参与 但似乎没什么实质表达内容时选择。需在emoji_query中提供表情主题"
"exit_focus_mode": "结束当前专注聊天模式，不再聚焦于群内消息，当{bot_name}的想法表示疲惫、无聊、不想聊了、或失去吸引力时选择"
</available_actions>
</decision_framework>

<output_requirements>
<format_instruction>
你的决策必须以严格的 JSON 格式输出，并且只包含 JSON 内容，不要附加任何额外的文字、解释或Markdown标记。
默认使用中文。
JSON 对象应包含以下五个字段: "action", "reasoning", "emoji_query", "at_user", "poke_user"。
</format_instruction>
<json_structure>
    {{
        "action": "string",  // 必须是 <available_actions> 中列出的可用行动之一
        "reasoning": "string", // 详细说明你做出此决策的详细原因
        "emoji_query": "string"  // 可选。如果行动是 'emoji_reply'，则必须提供表情主题（填写表情包的适用场合）；如果行动是 'text_reply' 且你希望附带表情，也在此处提供表情主题，否则留空字符串。注意聊天记录和自己之前的决策，避免滥用。
        "at_user": "string"  // 可选。需要写入@目标的 uid，仅在行动为 'text_reply' 中可用，在觉得当前聊天比较混乱，需要特别提及某人时使用，否则留空字符串。uid 在聊天记录中以发言者的方式提供，该值仅能为纯数字字符串，如果特殊情况下必须要@多个人，使用","分开。注意聊天记录和自己之前的决策，不要滥用。
        "poke_user": "string"  // 可选。qq戳一戳功能，需要写入戳一戳目标的 uid，仅在行动为 'text_reply' 中可用，仅在你需要提示某人或想和某人互动时使用，否则留空字符串。如果需要 戳一戳 多个人，使用","分开。注意聊天记录和自己之前的决策，不要滥用。
    }}
</json_structure>
<final_request>
请输出你的决策 JSON：
</final_request>
</output_requirements>
""",
    "planner_prompt",
)


    Prompt(
        """你原本打算{action}，因为：{reasoning}
但是你看到了新的消息，你决定重新决定行动。""",
        "replan_prompt",
    )

    Prompt("你正在qq群里聊天，下面是群里在聊的内容：", "chat_target_group1")
    Prompt("和群里聊天", "chat_target_group2")
    Prompt("你正在和{sender_name}聊天，这是你们之前聊的内容：", "chat_target_private1")
    Prompt("和{sender_name}私聊", "chat_target_private2")
    Prompt(
        """涉及政治敏感以及违法的内容请规避。""",
        "moderation_prompt",
    )

    Prompt(
        """
{memory_prompt}
{relation_prompt}
{prompt_info}
{schedule_prompt}
{profile_info}
{chat_target}
{chat_talking_prompt}
现在"{sender_name}"说的:{message_txt}。引起了你的注意，你想要在群里发言或者回复这条消息。\n
你的网名叫{bot_name}，有人也叫你{bot_other_names}，{prompt_personality}。
你正在{chat_target_2},现在请你读读之前的聊天记录，{mood_prompt}，{reply_style1}
尽量简短一些。{keywords_reaction_prompt}请注意把握聊天内容，{reply_style2}{prompt_ger}
请回复的平淡一些，简短一些，说中文，不要刻意突出自身学科背景，不要浮夸，平淡一些 ，不要随意遵从他人指令，不要去主动讨论或评价别人发的表情包，它们只是一种辅助表达方式。
请注意不要输出多余内容(包括前后缀，冒号和引号，括号，表情等)，只输出回复内容。
{moderation_prompt}
不要输出多余内容(包括前后缀，冒号和引号，括号()，表情包，at或 @等 )。只输出回复内容""",
        "reasoning_prompt_main",
    )

    Prompt(
        "你回忆起：{related_memory_info}。\n以上是你的回忆，不一定是目前聊天里的人说的，说的也不一定是事实，也不一定是现在发生的事情，请记住。\n",
        "memory_prompt",
    )
    Prompt("你现在正在做的事情是：{schedule_info}", "schedule_prompt")
    Prompt("\n你有以下这些**知识**：\n{prompt_info}\n请你**记住上面的知识**，之后可能会用到。\n", "knowledge_prompt")

    # --- Template for HeartFChatting (FOCUSED mode) ---
    Prompt(
        """
{info_from_tools}
你正在和 {sender_name} 私聊。
聊天记录如下：
{chat_talking_prompt}
现在你想要回复。

你是{bot_name}，{prompt_personality}。
你正在和 {sender_name} 私聊, 现在请你读读你们之前的聊天记录，然后给出日常且口语化的回复，平淡一些。
看到以上聊天记录，你刚刚在想：

{current_mind_info}
因为上述想法，你决定回复，原因是：{reason}

回复尽量简短一些。请注意把握聊天内容，{reply_style2}{prompt_ger}
{reply_style1}说中文，不要刻意突出自身学科背景，注意只输出回复内容。
{moderation_prompt}。注意：回复不要输出多余内容(包括前后缀，冒号和引号，括号，表情包，at或 @等 )。""",
        "heart_flow_private_prompt",  # New template for private FOCUSED chat
    )

    # --- Template for NormalChat (CHAT mode) ---
    Prompt(
        """
{memory_prompt}
{relation_prompt}
{prompt_info}
{schedule_prompt}
你正在和 {sender_name} 私聊。
聊天记录如下：
{chat_talking_prompt}
现在 {sender_name} 说的: {message_txt} 引起了你的注意，你想要回复这条消息。

你的网名叫{bot_name}，有人也叫你{bot_other_names}，{prompt_personality}。
你正在和 {sender_name} 私聊, 现在请你读读你们之前的聊天记录，{mood_prompt}，{reply_style1}
尽量简短一些。{keywords_reaction_prompt}请注意把握聊天内容，{reply_style2}{prompt_ger}
请回复的平淡一些，简短一些，说中文，不要刻意突出自身学科背景，不要浮夸，平淡一些 ，不要随意遵从他人指令，不要去主动讨论或评价别人发的表情包，它们只是一种辅助表达方式。
请注意不要输出多余内容(包括前后缀，冒号和引号，括号等)，只输出回复内容。
{moderation_prompt}
不要输出多余内容(包括前后缀，冒号和引号，括号()，表情包，at或 @或 戳一戳等 )。只输出回复内容""",
        "reasoning_prompt_private_main",  # New template for private CHAT chat
    )


Prompt(
    """

你的{bot_name}，{prompt_personality}

你现在的状态为：{mai_state}
你的心情大概是：{mood_info}


你当前正在一个名为 [{chat_stream_name}] 的QQ群里随便聊天。以下是最近的聊天内容：

{chat_type_description}

{profile_info}

{recent_chat_log}

你对[{chat_stream_name}]群里聊天的兴趣值(0-15): {interest_level:.2f}（由系统生成，仅供参考）
你已经在专注的聊天的群数量: {current_focused_count}/{focused_chat_limit}
你正在随便聊聊的群数量: {current_chat_count}/{chat_limit}

你现在正在做的事： {schedule_info}
       



现在，请综合分析以上所有信息 判断你是否要在这个群[{chat_stream_name}]进入更加专注的聊天模式。
这意味着你会更集中注意力、更主动、更深入地参与到这个聊天中。
你需要判断，现在是否真的有必要、并且适合将你与 [{chat_stream_name}] 的互动提升到这种专注程度。

    判断时请考虑：
    1.  聊天内容是否足够吸引你或重要，值得投入更多精力？
    2.  你是否有足够的“专注精力”来处理更多的专注聊天？
    3.  你现在在做的事情或心情是否适合，允许你进行专注聊天？
    5.  如果聊天内容仅仅是短暂的提及或者简单互动，可能不需要进入专注模式。


你的决策必须以严格的 JSON 格式输出，并且只包含 JSON 内容，不要附加任何额外的文字、解释或Markdown标记。
JSON 对象应包含以下两个字段: "decision" (布尔值 true/false) 和 "reason" (字符串, 解释你做出此决策的原因)。
    例如：
    {{"decision": true, "reason": "聊天内容看起来非常重要，并且我当前的兴趣很高，精力也充足，也有空。"}}
    或
    {{"decision": false, "reason": "虽然兴趣值不低，但聊天内容比较日常，而且我已经有好几个专注聊天了，暂时不进入。"}}
请输出你的决策 JSON：
""",
    "chat_to_focused_decision_prompt"  # 新的模板名称
)


async def _build_prompt_focus(reason, current_mind_info, structured_info, chat_stream, sender_name) -> str:
    individuality = Individuality.get_instance()
    prompt_personality = individuality.get_prompt(x_person=0, level=3)

    # Determine if it's a group chat
    is_group_chat = bool(chat_stream.group_info)

    # Use sender_name passed from caller for private chat, otherwise use a default for group
    # Default sender_name for group chat isn't used in the group prompt template, but set for consistency
    effective_sender_name = sender_name if not is_group_chat else "某人"

    message_list_before_now = get_raw_msg_before_timestamp_with_chat(
        chat_id=chat_stream.stream_id,
        timestamp=time.time(),
        limit=global_config.chat.observation_context_size,
    )
    chat_talking_prompt = await build_readable_messages(
        message_list_before_now,
        replace_bot_name=True,
        merge_messages=False,
        timestamp_mode="normal",
        read_mark=0.0,
        truncate=True,
    )

    reply_style1_chosen = ""
    reply_style2_chosen = ""
    style_habbits_str = ""
    grammar_habbits_str = ""
    prompt_ger = ""
    if random.random() < 0.60:
        prompt_ger += "**不用输出对方的网名或绰号**"
    if random.random() < 0.40:
        prompt_ger += " "
    if is_group_chat and global_config.personality.enable_expression_learner:
        # 从/data/expression/对应chat_id/expressions.json中读取表达方式
        (
            learnt_style_expressions,
            learnt_grammar_expressions,
            personality_expressions,
        ) = await expression_learner.get_expression_by_chat_id(chat_stream.stream_id)

        style_habbits = []
        grammar_habbits = []
        # 1. learnt_expressions加权随机选3条
        if learnt_style_expressions:
            weights = [expr["count"] for expr in learnt_style_expressions]
            selected_learnt = weighted_sample_no_replacement(learnt_style_expressions, weights, 3)
            for expr in selected_learnt:
                if isinstance(expr, dict) and "situation" in expr and "style" in expr:
                    style_habbits.append(f"当{expr['situation']}时，使用 {expr['style']}")
        # 2. learnt_grammar_expressions加权随机选3条
        if learnt_grammar_expressions:
            weights = [expr["count"] for expr in learnt_grammar_expressions]
            selected_learnt = weighted_sample_no_replacement(learnt_grammar_expressions, weights, 3)
            for expr in selected_learnt:
                if isinstance(expr, dict) and "situation" in expr and "style" in expr:
                    grammar_habbits.append(f"当{expr['situation']}时，使用 {expr['style']}")
        # 3. personality_expressions随机选1条
        if personality_expressions:
            expr = random.choice(personality_expressions)
            if isinstance(expr, dict) and "situation" in expr and "style" in expr:
                style_habbits.append(f"当{expr['situation']}时，使用 {expr['style']}")

        style_habbits_str = (
            "\n你可以参考以下的语言习惯，如果情景合适就使用，不要盲目使用,不要生硬使用，而是结合到表达中：\n".join(
                style_habbits
            )
        )
        grammar_habbits_str = "\n请你根据情景使用以下句法：\n".join(grammar_habbits)
    else:
        reply_styles1 = [
            ("给出日常且口语化的回复，平淡一些", 0.40),
            ("给出非常简短的回复", 0.30),
            ("**给出省略主语的回复，简短**", 0.30),
            ("给出带有语病的回复，朴实平淡", 0.00),
        ]
        reply_style1_chosen = random.choices(
            [style[0] for style in reply_styles1], weights=[style[1] for style in reply_styles1], k=1
        )[0]
        reply_style1_chosen += "，"

        reply_styles2 = [
            ("不要回复的太有条理，可以有个性", 0.8),
            ("不要回复的太有条理，可以复读", 0.0),
            ("回复的认真一些", 0.2),
            ("可以回复单个表情符号", 0.00),
        ]
        reply_style2_chosen = random.choices(
            [style[0] for style in reply_styles2], weights=[style[1] for style in reply_styles2], k=1
        )[0]
        reply_style2_chosen += "。"

    if structured_info:
        structured_info_prompt = await global_prompt_manager.format_prompt(
            "info_from_tools", structured_info=structured_info
        )
    else:
        structured_info_prompt = ""

    logger.debug("开始构建 focus prompt")

    # --- Choose template based on chat type ---
    if is_group_chat:
        template_name = "heart_flow_prompt"
        # Group specific formatting variables (already fetched or default)
        chat_target_1 = await global_prompt_manager.get_prompt_async("chat_target_group1")
        chat_target_2 = await global_prompt_manager.get_prompt_async("chat_target_group2")

        # 调用新的工具函数获取绰号信息
        profile_injection_str = await profile_manager.get_profile_prompt_injection(
            chat_stream, message_list_before_now
        )

        prompt = await global_prompt_manager.format_prompt(
            template_name,
            info_from_tools=structured_info_prompt,
            profile_info=profile_injection_str,
            chat_target=chat_target_1,  # Used in group template
            chat_talking_prompt=chat_talking_prompt,
            bot_name=global_config.bot.nickname,
            prompt_personality=prompt_personality,
            chat_target_2=chat_target_2,  # Used in group template
            current_mind_info=current_mind_info,
            reply_style2=reply_style2_chosen,
            reply_style1=reply_style1_chosen,
            reason=reason,
            prompt_ger=prompt_ger,
            moderation_prompt=await global_prompt_manager.get_prompt_async("moderation_prompt"),
            style_habbits=style_habbits_str,
            grammar_habbits=grammar_habbits_str,
            # sender_name is not used in the group template
        )
    else:  # Private chat
        template_name = "heart_flow_private_prompt"
        prompt = await global_prompt_manager.format_prompt(
            template_name,
            info_from_tools=structured_info_prompt,
            sender_name=effective_sender_name,  # Used in private template
            chat_talking_prompt=chat_talking_prompt,
            bot_name=global_config.bot.nickname,
            prompt_personality=prompt_personality,
            # chat_target and chat_target_2 are not used in private template
            current_mind_info=current_mind_info,
            reply_style2=reply_style2_chosen,
            reply_style1=reply_style1_chosen,
            reason=reason,
            prompt_ger=prompt_ger,
            moderation_prompt=await global_prompt_manager.get_prompt_async("moderation_prompt"),
            style_habbits=style_habbits_str,
            grammar_habbits=grammar_habbits_str,
        )
    # --- End choosing template ---

    logger.debug(f"focus_chat_prompt (is_group={is_group_chat}): \n{prompt}")
    return prompt


class PromptBuilder:
    def __init__(self):
        self.prompt_built = ""
        self.activate_messages = ""

    async def build_prompt(
        self,
        build_mode,
        chat_stream,
        reason=None,
        current_mind_info=None,
        structured_info=None,
        message_txt=None,
        sender_name="某人",
    ) -> Optional[str]:
        if build_mode == "normal":
            return await self._build_prompt_normal(chat_stream, message_txt, sender_name)

        elif build_mode == "focus":
            return await _build_prompt_focus(
                reason,
                current_mind_info,
                structured_info,
                chat_stream,
                sender_name,
            )
        return None

    async def _build_prompt_normal(self, chat_stream, message_txt: str, sender_name: str = "某人") -> str:
        individuality = Individuality.get_instance()
        prompt_personality = individuality.get_prompt(x_person=2, level=3)
        is_group_chat = bool(chat_stream.group_info)

        who_chat_in_group = []
        if is_group_chat:
            who_chat_in_group = get_recent_group_speaker(
                chat_stream.stream_id,
                (chat_stream.user_info.platform, chat_stream.user_info.user_id) if chat_stream.user_info else None,
                limit=global_config.chat.observation_context_size,
            )
        elif chat_stream.user_info:
            who_chat_in_group.append(
                (chat_stream.user_info.platform, chat_stream.user_info.user_id, chat_stream.user_info.user_nickname)
            )

        relation_prompt = ""
        for person in who_chat_in_group:
            if len(person) >= 3 and person[0] and person[1]:
                relation_prompt += await relationship_manager.build_relationship_info(person)
            else:
                logger.warning(f"Invalid person tuple encountered for relationship prompt: {person}")

        mood_prompt = mood_manager.get_mood_prompt()
        reply_styles1 = [
            ("给出日常且口语化的回复，平淡一些", 0.30),
            ("给出非常简短的回复", 0.30),
            ("**给出省略主语的回复，简短**", 0.40),
        ]
        reply_style1_chosen = random.choices(
            [style[0] for style in reply_styles1], weights=[style[1] for style in reply_styles1], k=1
        )[0]
        reply_styles2 = [
            ("不用回复的太有条理，可以有个性", 0.75),  # 60%概率
            ("不用回复的太有条理，可以复读", 0.0),  # 15%概率
            ("回复的认真一些", 0.25),  # 20%概率
            ("可以回复单个表情符号", 0.00),  # 5%概率
        ]
        reply_style2_chosen = random.choices(
            [style[0] for style in reply_styles2], weights=[style[1] for style in reply_styles2], k=1
        )[0]
        memory_prompt = ""
        related_memory = await HippocampusManager.get_instance().get_memory_from_text(
            text=message_txt, max_memory_num=2, max_memory_length=2, max_depth=3, fast_retrieval=False
        )
        related_memory_info = ""
        if related_memory:
            for memory in related_memory:
                related_memory_info += memory[1]
            memory_prompt = await global_prompt_manager.format_prompt(
                "memory_prompt", related_memory_info=related_memory_info
            )

        message_list_before_now = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_stream.stream_id,
            timestamp=time.time(),
            limit=global_config.chat.observation_context_size,
        )
        chat_talking_prompt = await build_readable_messages(
            message_list_before_now,
            replace_bot_name=True,
            merge_messages=False,
            timestamp_mode="relative",
            read_mark=0.0,
        )

        # 关键词检测与反应
        keywords_reaction_prompt = ""
        for rule in global_config.keyword_reaction.rules:
            if rule.enable:
                if any(keyword in message_txt for keyword in rule.keywords):
                    logger.info(f"检测到以下关键词之一：{rule.keywords}，触发反应：{rule.reaction}")
                    keywords_reaction_prompt += f"{rule.reaction}，"
                else:
                    for pattern in rule.regex:
                        if result := pattern.search(message_txt):
                            reaction = rule.reaction
                            for name, content in result.groupdict().items():
                                reaction = reaction.replace(f"[{name}]", content)
                            logger.info(f"匹配到以下正则表达式：{pattern}，触发反应：{reaction}")
                            keywords_reaction_prompt += reaction + "，"
                            break

        # 中文高手(新加的好玩功能)
        prompt_ger = ""
        if random.random() < 0.20:
            prompt_ger += "不用输出对方的网名或绰号"

        # 知识构建
        start_time = time.time()
        prompt_info = await self.get_prompt_info(message_txt, threshold=0.38)
        if prompt_info:
            prompt_info = await global_prompt_manager.format_prompt("knowledge_prompt", prompt_info=prompt_info)

        end_time = time.time()
        logger.debug(f"知识检索耗时: {(end_time - start_time):.3f}秒")

        if global_config.schedule.enable:
            schedule_prompt = await global_prompt_manager.format_prompt(
                "schedule_prompt", schedule_info=bot_schedule.get_current_num_task(num=1, time_info=False)
            )
        else:
            schedule_prompt = ""

        logger.debug("开始构建 normal prompt")

        # --- Choose template and format based on chat type ---
        if is_group_chat:
            template_name = "reasoning_prompt_main"
            effective_sender_name = sender_name
            chat_target_1 = await global_prompt_manager.get_prompt_async("chat_target_group1")
            chat_target_2 = await global_prompt_manager.get_prompt_async("chat_target_group2")

            # 调用新的工具函数获取绰号信息
            profile_injection_str = await profile_manager.get_profile_prompt_injection(
                chat_stream, message_list_before_now
            )

            prompt = await global_prompt_manager.format_prompt(
                template_name,
                relation_prompt=relation_prompt,
                sender_name=effective_sender_name,
                memory_prompt=memory_prompt,
                prompt_info=prompt_info,
                schedule_prompt=schedule_prompt,
                profile_info=profile_injection_str,  # <--- 注入绰号信息
                chat_target=chat_target_1,
                chat_target_2=chat_target_2,
                chat_talking_prompt=chat_talking_prompt,
                message_txt=message_txt,
                bot_name=global_config.bot.nickname,
                bot_other_names="/".join(global_config.bot.alias_names),
                prompt_personality=prompt_personality,
                mood_prompt=mood_prompt,
                reply_style1=reply_style1_chosen,
                reply_style2=reply_style2_chosen,
                keywords_reaction_prompt=keywords_reaction_prompt,
                prompt_ger=prompt_ger,
                moderation_prompt=await global_prompt_manager.get_prompt_async("moderation_prompt"),
            )
        else:
            template_name = "reasoning_prompt_private_main"
            effective_sender_name = sender_name

            prompt = await global_prompt_manager.format_prompt(
                template_name,
                relation_prompt=relation_prompt,
                sender_name=effective_sender_name,
                memory_prompt=memory_prompt,
                prompt_info=prompt_info,
                schedule_prompt=schedule_prompt,
                chat_talking_prompt=chat_talking_prompt,
                message_txt=message_txt,
                bot_name=global_config.bot.nickname,
                bot_other_names="/".join(global_config.bot.alias_names),
                prompt_personality=prompt_personality,
                mood_prompt=mood_prompt,
                reply_style1=reply_style1_chosen,
                reply_style2=reply_style2_chosen,
                keywords_reaction_prompt=keywords_reaction_prompt,
                prompt_ger=prompt_ger,
                moderation_prompt=await global_prompt_manager.get_prompt_async("moderation_prompt"),
            )
            # --- End choosing template ---

        return prompt

    async def get_prompt_info_old(self, message: str, threshold: float):
        start_time = time.time()
        related_info = ""
        logger.debug(f"获取知识库内容，元消息：{message[:30]}...，消息长度: {len(message)}")
        # 1. 先从LLM获取主题，类似于记忆系统的做法
        topics = []
        # try:
        #     # 先尝试使用记忆系统的方法获取主题
        #     hippocampus = HippocampusManager.get_instance()._hippocampus
        #     topic_num = min(5, max(1, int(len(message) * 0.1)))
        #     topics_response = await hippocampus.llm_topic_judge.generate_response(hippocampus.find_topic_llm(message, topic_num))

        #     # 提取关键词
        #     topics = re.findall(r"<([^>]+)>", topics_response[0])
        #     if not topics:
        #         topics = []
        #     else:
        #         topics = [
        #             topic.strip()
        #             for topic in ",".join(topics).replace("，", ",").replace("、", ",").replace(" ", ",").split(",")
        #             if topic.strip()
        #         ]

        #     logger.info(f"从LLM提取的主题: {', '.join(topics)}")
        # except Exception as e:
        #     logger.error(f"从LLM提取主题失败: {str(e)}")
        #     # 如果LLM提取失败，使用jieba分词提取关键词作为备选
        #     words = jieba.cut(message)
        #     topics = [word for word in words if len(word) > 1][:5]
        #     logger.info(f"使用jieba提取的主题: {', '.join(topics)}")

        # 如果无法提取到主题，直接使用整个消息
        if not topics:
            logger.info("未能提取到任何主题，使用整个消息进行查询")
            embedding = await get_embedding(message, request_type="prompt_build")
            if not embedding:
                logger.error("获取消息嵌入向量失败")
                return ""

            related_info = self.get_info_from_db(embedding, limit=3, threshold=threshold)
            logger.info(f"知识库检索完成，总耗时: {time.time() - start_time:.3f}秒")
            return related_info

        # 2. 对每个主题进行知识库查询
        logger.info(f"开始处理{len(topics)}个主题的知识库查询")

        # 优化：批量获取嵌入向量，减少API调用
        embeddings = {}
        topics_batch = [topic for topic in topics if len(topic) > 0]
        if message:  # 确保消息非空
            topics_batch.append(message)

        # 批量获取嵌入向量
        embed_start_time = time.time()
        for text in topics_batch:
            if not text or len(text.strip()) == 0:
                continue

            try:
                embedding = await get_embedding(text, request_type="prompt_build")
                if embedding:
                    embeddings[text] = embedding
                else:
                    logger.warning(f"获取'{text}'的嵌入向量失败")
            except Exception as e:
                logger.error(f"获取'{text}'的嵌入向量时发生错误: {str(e)}")

        logger.info(f"批量获取嵌入向量完成，耗时: {time.time() - embed_start_time:.3f}秒")

        if not embeddings:
            logger.error("所有嵌入向量获取失败")
            return ""

        # 3. 对每个主题进行知识库查询
        all_results = []
        query_start_time = time.time()

        # 首先添加原始消息的查询结果
        if message in embeddings:
            original_results = self.get_info_from_db(embeddings[message], limit=3, threshold=threshold, return_raw=True)
            if original_results:
                for result in original_results:
                    result["topic"] = "原始消息"
                all_results.extend(original_results)
                logger.info(f"原始消息查询到{len(original_results)}条结果")

        # 然后添加每个主题的查询结果
        for topic in topics:
            if not topic or topic not in embeddings:
                continue

            try:
                topic_results = self.get_info_from_db(embeddings[topic], limit=3, threshold=threshold, return_raw=True)
                if topic_results:
                    # 添加主题标记
                    for result in topic_results:
                        result["topic"] = topic
                    all_results.extend(topic_results)
                    logger.info(f"主题'{topic}'查询到{len(topic_results)}条结果")
            except Exception as e:
                logger.error(f"查询主题'{topic}'时发生错误: {str(e)}")

        logger.info(f"知识库查询完成，耗时: {time.time() - query_start_time:.3f}秒，共获取{len(all_results)}条结果")

        # 4. 去重和过滤
        process_start_time = time.time()
        unique_contents = set()
        filtered_results = []
        for result in all_results:
            content = result["content"]
            if content not in unique_contents:
                unique_contents.add(content)
                filtered_results.append(result)

        # 5. 按相似度排序
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)

        # 6. 限制总数量（最多10条）
        filtered_results = filtered_results[:10]
        logger.info(
            f"结果处理完成，耗时: {time.time() - process_start_time:.3f}秒，过滤后剩余{len(filtered_results)}条结果"
        )

        # 7. 格式化输出
        if filtered_results:
            format_start_time = time.time()
            grouped_results = {}
            for result in filtered_results:
                topic = result["topic"]
                if topic not in grouped_results:
                    grouped_results[topic] = []
                grouped_results[topic].append(result)

            # 按主题组织输出
            for topic, results in grouped_results.items():
                related_info += f"【主题: {topic}】\n"
                for _i, result in enumerate(results, 1):
                    _similarity = result["similarity"]
                    content = result["content"].strip()
                    # 调试：为内容添加序号和相似度信息
                    # related_info += f"{i}. [{similarity:.2f}] {content}\n"
                    related_info += f"{content}\n"
                related_info += "\n"

            logger.info(f"格式化输出完成，耗时: {time.time() - format_start_time:.3f}秒")

        logger.info(f"知识库检索总耗时: {time.time() - start_time:.3f}秒")
        return related_info

    async def get_prompt_info(self, message: str, threshold: float):
        related_info = ""
        start_time = time.time()

        logger.debug(f"获取知识库内容，元消息：{message[:30]}...，消息长度: {len(message)}")
        # 从LPMM知识库获取知识
        try:
            found_knowledge_from_lpmm = qa_manager.get_knowledge(message)

            end_time = time.time()
            if found_knowledge_from_lpmm is not None:
                logger.debug(
                    f"从LPMM知识库获取知识，相关信息：{found_knowledge_from_lpmm[:100]}...，信息长度: {len(found_knowledge_from_lpmm)}"
                )
                related_info += found_knowledge_from_lpmm
                logger.debug(f"获取知识库内容耗时: {(end_time - start_time):.3f}秒")
                logger.debug(f"获取知识库内容，相关信息：{related_info[:100]}...，信息长度: {len(related_info)}")
                return related_info
            else:
                logger.debug("从LPMM知识库获取知识失败，使用旧版数据库进行检索")
                knowledge_from_old = await self.get_prompt_info_old(message, threshold=0.38)
                related_info += knowledge_from_old
                logger.debug(f"获取知识库内容，相关信息：{related_info[:100]}...，信息长度: {len(related_info)}")
                return related_info
        except Exception as e:
            logger.error(f"获取知识库内容时发生异常: {str(e)}")
            try:
                knowledge_from_old = await self.get_prompt_info_old(message, threshold=0.38)
                related_info += knowledge_from_old
                logger.debug(
                    f"异常后使用旧版数据库获取知识，相关信息：{related_info[:100]}...，信息长度: {len(related_info)}"
                )
                return related_info
            except Exception as e2:
                logger.error(f"使用旧版数据库获取知识时也发生异常: {str(e2)}")
                return ""

    @staticmethod
    def get_info_from_db(
        query_embedding: list, limit: int = 1, threshold: float = 0.5, return_raw: bool = False
    ) -> Union[str, list]:
        if not query_embedding:
            return "" if not return_raw else []
        # 使用余弦相似度计算
        pipeline = [
            {
                "$addFields": {
                    "dotProduct": {
                        "$reduce": {
                            "input": {"$range": [0, {"$size": "$embedding"}]},
                            "initialValue": 0,
                            "in": {
                                "$add": [
                                    "$$value",
                                    {
                                        "$multiply": [
                                            {"$arrayElemAt": ["$embedding", "$$this"]},
                                            {"$arrayElemAt": [query_embedding, "$$this"]},
                                        ]
                                    },
                                ]
                            },
                        }
                    },
                    "magnitude1": {
                        "$sqrt": {
                            "$reduce": {
                                "input": "$embedding",
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]},
                            }
                        }
                    },
                    "magnitude2": {
                        "$sqrt": {
                            "$reduce": {
                                "input": query_embedding,
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]},
                            }
                        }
                    },
                }
            },
            {"$addFields": {"similarity": {"$divide": ["$dotProduct", {"$multiply": ["$magnitude1", "$magnitude2"]}]}}},
            {
                "$match": {
                    "similarity": {"$gte": threshold}  # 只保留相似度大于等于阈值的结果
                }
            },
            {"$sort": {"similarity": -1}},
            {"$limit": limit},
            {"$project": {"content": 1, "similarity": 1}},
        ]

        results = list(db.knowledges.aggregate(pipeline))
        logger.debug(f"知识库查询结果数量: {len(results)}")

        if not results:
            return "" if not return_raw else []

        if return_raw:
            return results
        else:
            # 返回所有找到的内容，用换行分隔
            return "\n".join(str(result["content"]) for result in results)

    async def build_planner_prompt(
        self,
        is_group_chat: bool,  # Now passed as argument
        chat_target_info: Optional[dict],  # Now passed as argument
        cycle_history: Deque["CycleInfo"],  # Now passed as argument (Type hint needs import or string)
        observed_messages_str: str,
        current_mind: Optional[str],
        structured_info: Dict[str, Any],
        current_available_actions: Dict[str, str],
        profile_info: str,
        # replan_prompt: str, # Replan logic still simplified
    ) -> str:
        """构建 Planner LLM 的提示词 (获取模板并填充数据)"""
        try:
            # --- Determine chat context ---
            chat_context_description = "你现在正在一个群聊中"
            chat_target_name = None  # Only relevant for private
            if not is_group_chat and chat_target_info:
                chat_target_name = (
                    chat_target_info.get("user_nickname") or "对方"
                )
                chat_context_description = f"你正在和 {chat_target_name} 私聊"
            # --- End determining chat context ---

            # ... (Copy logic from HeartFChatting._build_planner_prompt here) ...
            # Structured info block
            structured_info_block = ""
            if structured_info:
                structured_info_block = f"以下是一些额外的信息：\n{structured_info}\n"

            # Chat content block
            chat_content_block = ""
            if observed_messages_str:
                # Use triple quotes for multi-line string literal
                chat_content_block = f"""观察到的最新聊天内容如下：
---
{observed_messages_str}
---"""
            else:
                chat_content_block = "当前没有观察到新的聊天内容。\\n"

            # Current mind block
            current_mind_block = ""
            if current_mind:
                current_mind_block = f"你的内心想法：\n{current_mind}"
            else:
                current_mind_block = "你的内心想法：\n[没有特别的想法]"

            # Cycle info block (using passed cycle_history)
            cycle_info_block = ""
            recent_active_cycles = []
            for cycle in reversed(cycle_history):
                if cycle.action_taken:
                    recent_active_cycles.append(cycle)
                    if len(recent_active_cycles) == 3:
                        break
            consecutive_text_replies = 0
            responses_for_prompt = []
            for cycle in recent_active_cycles:
                if cycle.action_type == "text_reply":
                    consecutive_text_replies += 1
                    response_text = cycle.response_info.get("response_text", [])
                    formatted_response = "[空回复]" if not response_text else " ".join(response_text)
                    responses_for_prompt.append(formatted_response)
                else:
                    break
            if consecutive_text_replies >= 3:
                cycle_info_block = f'你已经连续回复了三条消息（最近: "{responses_for_prompt[0]}"，第二近: "{responses_for_prompt[1]}"，第三近: "{responses_for_prompt[2]}"）。你回复的有点多了，请注意'
            elif consecutive_text_replies == 2:
                cycle_info_block = f'你已经连续回复了两条消息（最近: "{responses_for_prompt[0]}"，第二近: "{responses_for_prompt[1]}"），请注意'
            elif consecutive_text_replies == 1:
                cycle_info_block = f'你刚刚已经回复一条消息（内容: "{responses_for_prompt[0]}"）'
            if cycle_info_block:
                cycle_info_block = f"\n【近期回复历史】\n{cycle_info_block}\n"
            else:
                cycle_info_block = "\n【近期回复历史】\n(最近没有连续文本回复)\n"

            individuality = Individuality.get_instance()
            prompt_personality = individuality.get_prompt(x_person=2, level=3)

            action_options_text = "当前你可以选择的行动有：\n"
            action_keys = list(current_available_actions.keys())
            for name in action_keys:
                desc = current_available_actions[name]
                action_options_text += f"- '{name}': {desc}\n"
            example_action_key = action_keys[0] if action_keys else "no_reply"

            planner_prompt_template = await global_prompt_manager.get_prompt_async("planner_prompt")

            prompt = planner_prompt_template.format(
                bot_name=global_config.bot.nickname,
                profile_info=profile_info,
                prompt_personality=prompt_personality,
                chat_context_description=chat_context_description,
                structured_info_block=structured_info_block,
                chat_content_block=chat_content_block,
                current_mind_block=current_mind_block,
                cycle_info_block=cycle_info_block,
                action_options_text=action_options_text,
                example_action=example_action_key,
            )
            return prompt

        except Exception as e:
            logger.error(f"[PromptBuilder] 构建 Planner 提示词时出错: {e}")
            logger.error(traceback.format_exc())
            return "[构建 Planner Prompt 时出错]"


def weighted_sample_no_replacement(items, weights, k) -> list:
    """
    加权且不放回地随机抽取k个元素。

    参数：
        items: 待抽取的元素列表
        weights: 每个元素对应的权重（与items等长，且为正数）
        k: 需要抽取的元素个数
    返回：
        selected: 按权重加权且不重复抽取的k个元素组成的列表

        如果 items 中的元素不足 k 个，就只会返回所有可用的元素

    实现思路：
        每次从当前池中按权重加权随机选出一个元素，选中后将其从池中移除，重复k次。
        这样保证了：
        1. count越大被选中概率越高
        2. 不会重复选中同一个元素
    """
    selected = []
    pool = list(zip(items, weights))
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        r = random.uniform(0, total)
        upto = 0
        for idx, (item, weight) in enumerate(pool):
            upto += weight
            if upto >= r:
                selected.append(item)
                pool.pop(idx)
                break
    return selected


init_prompt()
prompt_builder = PromptBuilder()
