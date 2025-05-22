import random
import json # 保持导入，以防未来需要处理JSON输入或输出（尽管当前设计是纯文本）
from datetime import datetime # 保持导入，可能用于时间相关的上下文处理
from typing import Optional, List, Dict, Any # 从 typing 导入

from src.common.logger_manager import get_logger
from src.chat.models.utils_model import LLMRequest
from src.config.config import global_config
# 确保导入PFC相关的Info类，路径可能需要根据你的项目结构调整
try:
    from .observation_info import ObservationInfo
    from .conversation_info import ConversationInfo
except ImportError: # Fallback for type hinting if direct import fails
    ObservationInfo = Optional[Any] # type: ignore
    ConversationInfo = Optional[Any] # type: ignore

from .pfc_utils import build_chat_history_text # 假设此函数用于构建聊天记录字符串

logger = get_logger("pfc_reply_generator_contextual") # 更新logger名称

# --- 为不同的回复场景定义新的、包含上下文的Prompt模板 ---

PROMPT_RG_DIRECT_REPLY = """
当前时间：{current_time_str}
你是{bot_name}，你正在和{sender_name}在QQ上私聊。
你与对方的关系是：{relationship_text}
你现在的心情是：{current_emotion_text}

当前对话目标：
{goals_str}

最近的聊天记录：
{chat_history_text}

你刚刚经过思考，内心的想法是：
---
{current_mind_info}
---
基于以上想法，你想回复对方。

现在，请你基于以上的聊天记录和背景信息，**把你的想法组织成合适简短的语言**，回复对方。
回复要点：
1. 清晰表达“内心想法”的核心含义。
2. 保持对话的流畅和自然过渡。
3. 符合你({bot_name})的个性和当前的情绪。
4. 除非“内心想法”中明确指示，否则避免开启全新的、不相关的话题。
5. {reply_style_directives}

请直接输出你组织好的纯文本回复内容，不需要任何额外的前后缀、引号或自身思考的描述。
"""

PROMPT_RG_SEND_NEW_MESSAGE = """
当前时间：{current_time_str}
你是{bot_name}，你正在和{sender_name}在QQ上私聊。**并且你刚刚已经回复了对方**。
你与对方的关系是：{relationship_text}
你现在的心情是：{current_emotion_text}

当前对话目标：
{goals_str}

最近的聊天记录（包括你上一句说的话）：
{chat_history_text}

你的“内心想法”是：
---
{current_mind_info}
---
基于以上想法，你想再发一条消息。

现在，请你基于以上的聊天记录和背景信息，**把你的想法组织成合适简短的语言**，再给对方发一条消息。
这条消息应该：
1. 与之前的消息连贯，像是对话的自然延伸。
2. 符合你({bot_name})的个性和当前的情绪。
3. {reply_style_directives}

请直接输出你组织好的纯文本回复内容，不需要任何额外的前后缀、引号或自身思考的描述。
"""

PROMPT_RG_SAY_GOODBYE = """
当前时间：{current_time_str}
你是{bot_name}，你正在和{sender_name}在QQ上私聊。对话似乎即将结束。
你与对方的关系是：{relationship_text}
你现在的心情是：{current_emotion_text}

最近的聊天记录：
{chat_history_text}

你此刻的“内心想法”是：
---
{current_mind_info}
---
基于以上想法，你想最后再发一条消息。

现在，请将这个“内心想法”组织成一句简短、礼貌且自然的最后的消息。
这条消息应该：
1. 从你自己的角度发言。
2. 符合你的性格特征和身份细节。
3. 通俗易懂，自然流畅，通常很简短。
4. 自然地为这场对话画上句号，避免开启新话题或显得冗长、刻意。

请直接输出你组织好的纯文本告别消息内容，不需要任何额外的前后缀、引号或自身思考的描述。
"""

PROMPT_RG_REPLY_AFTER_WAIT_TIMEOUT = """
当前时间：{current_time_str}
你是{bot_name}，你正在和{sender_name}在QQ上私聊。你之前等待对方回复但超时了（大约等待了 {last_wait_duration_minutes:.1f} 分钟）。
你与对方的关系是：{relationship_text}
你现在的心情是：{current_emotion_text}

当前对话目标（可能已包含提示你处理等待超时的目标）：
{goals_str}

最近的聊天记录（包括你等待前发送的消息）：
{chat_history_text}

考虑到对方长时间未回应，你此刻的“内心想法”是：
---
{current_mind_info}
---

现在，考虑到对方长时间未回复，请将这个“内心想法”组织成一句自然的回复，尝试打破沉默或重新与对方建立连接。
这条消息可以尝试重新引起对方的注意，或者礼貌地询问对方是否还在，或者表达你准备结束对话的意图等。
请注意语气，既要表达出你注意到了对方的沉默，又不要显得过于急躁或认真的指责。
回复要点：
1. 可以适当地提及等待或询问对方情况，但语气要温和。
2. 符合你({bot_name})的个性和当前的情绪。
3. {reply_style_directives}

请直接输出你组织好的纯文本回复内容，不需要任何额外的前后缀、引号或自身思考的描述。
"""

REPLY_STYLE_DIRECTIVES_EXAMPLES = [
    "注意在私聊中，通常不会提及对方网名，建议省略主语或用 你 来代替"
]

class ReplyGenerator:
    def __init__(self, stream_id: str, private_name: str):
        self.llm = LLMRequest(
            model=global_config.model.pfc_chat,
            temperature=global_config.model.pfc_chat.get("temp", 0.75), # 语言组织时温度可以略高一点点以增加自然度
            max_tokens=global_config.model.pfc_chat.get("max_tokens", 200), # 回复通常不需要太长
            request_type="pfc_reply_organization_contextual",
        )
        self.name = global_config.bot.nickname
        self.private_name = private_name

    async def generate(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        action_type: str
    ) -> str:

        logger.debug(
            f"[私聊][{self.private_name}] ReplyGenerator 开始基于SubMind想法和上下文组织回复 (动作类型: {action_type})"
        )

        # 1. 从 ConversationInfo 中获取 SubMind 的 "内心想法"
        current_mind_info_val = conversation_info.current_pfc_thought
        if not current_mind_info_val:
            logger.warning(f"[私聊][{self.private_name}] ReplyGenerator: SubMind的内心想法为空，无法生成回复。")
            return "嗯..." # 或其他合适的默认回复

        # 2. 从 Info 对象中准备Prompt所需的其他上下文
        current_time_str_val = observation_info.current_time_str if observation_info else "未知时间"
        # 对于私聊，sender_name 就是对方的名字，即 self.private_name
        sender_name_val = self.private_name
        relationship_text_val = conversation_info.relationship_text or "你们还不熟悉。"
        current_emotion_text_val = conversation_info.current_emotion_text or "心情平静。"

        goals_str_parts = []
        if conversation_info.goal_list:
            # 只取最近的目标给ReplyGenerator，避免Prompt过长
            for goal_item in conversation_info.goal_list[-1:]: # 例如只取最后一个目标
                goal = goal_item.get('goal', '未知目标')
                reason = goal_item.get('reasoning', '无具体原因')
                goals_str_parts.append(f"- {goal} (因: {reason})")
        goals_str_val = "\n".join(goals_str_parts) if goals_str_parts else "当前没有明确的对话目标。"

        # 使用 build_chat_history_text 来获取最新的聊天记录字符串
        # 注意：build_chat_history_text 内部会处理 ObservationInfo 中的 chat_history 和 unprocessed_messages
        chat_history_text_val = await build_chat_history_text(observation_info, self.private_name)
        if not chat_history_text_val.strip() or chat_history_text_val == "还没有聊天记录。\n":
            chat_history_text_val = "（你们还没有开始聊天，或者最近没有聊天记录。）"


        # 3. 根据 action_type 选择 Prompt 模板
        prompt_template_to_use = ""
        format_params: Dict[str, Any] = { # 明确类型
            "current_time_str": current_time_str_val,
            "bot_name": self.name,
            "sender_name": sender_name_val,
            "relationship_text": relationship_text_val,
            "current_emotion_text": current_emotion_text_val,
            "goals_str": goals_str_val,
            "chat_history_text": chat_history_text_val,
            "current_mind_info": current_mind_info_val,
            "reply_style_directives": random.choice(REPLY_STYLE_DIRECTIVES_EXAMPLES)
        }

        if action_type == "direct_reply":
            prompt_template_to_use = PROMPT_RG_DIRECT_REPLY
        elif action_type == "send_new_message":
            prompt_template_to_use = PROMPT_RG_SEND_NEW_MESSAGE
        elif action_type == "say_goodbye":
            prompt_template_to_use = PROMPT_RG_SAY_GOODBYE
        elif action_type == "reply_after_wait_timeout":
            prompt_template_to_use = PROMPT_RG_REPLY_AFTER_WAIT_TIMEOUT
            # 为这个特定模板添加等待时长参数
            format_params["last_wait_duration_minutes"] = conversation_info.last_wait_duration_minutes or 0.0
        else:
            logger.warning(f"[私聊][{self.private_name}] ReplyGenerator 收到未支持的action_type: {action_type}，将使用直接回复的组织方式。")
            prompt_template_to_use = PROMPT_RG_DIRECT_REPLY # 默认

        # 4. 格式化并调用LLM
        try:
            prompt = prompt_template_to_use.format(**format_params)
        except KeyError as e:
            logger.error(f"[私聊][{self.private_name}] ReplyGenerator 格式化Prompt时缺少键: {e}。模板标识符: {action_type}, 实际模板片段: {prompt_template_to_use[:150]}..., 参数键: {list(format_params.keys())}", exc_info=True)
            return "抱歉，我组织语言的时候出了一点小问题。" # 返回一个对用户友好的错误信息
        
        logger.debug(f"[私聊][{self.private_name}] ReplyGenerator 发送到LLM的语言组织Prompt ({action_type}):\n------\n{prompt}\n------")

        try:
            generated_reply_text, _ = await self.llm.generate_response_async(prompt)
            logger.debug(f"[私聊][{self.private_name}] ReplyGenerator LLM原始生成内容: '{generated_reply_text}'")

            if not generated_reply_text:
                generated_reply_text = "嗯。" # 如果LLM返回空，给一个非常简短的默认回复
                logger.warning(f"[私聊][{self.private_name}] ReplyGenerator LLM返回空，使用默认简单回复。")
            
            # 清理回复文本
            cleaned_reply = generated_reply_text.strip()
            # 移除可能的LLM引导性前缀，例如 "你的回复："
            # 使用更健壮的方式移除，避免大小写问题和多种可能的前缀
            prefixes_to_check = [f"{self.name}：", f"{self.name}:", "你的回复：", "回复：", "机器人回复：", f"好的，这是组织后的回复：", f"这是组织好的回复："]
            for prefix in prefixes_to_check:
                if cleaned_reply.lower().startswith(prefix.lower()):
                    cleaned_reply = cleaned_reply[len(prefix):].strip()
                    break # 找到并移除一个即可

            # 移除首尾可能存在的引号
            cleaned_reply = cleaned_reply.strip('"').strip("'")
            
            # 确保不会返回空的或只有空白的字符串
            if not cleaned_reply.strip():
                logger.warning(f"[私聊][{self.private_name}] ReplyGenerator 清理后的回复为空，原始生成: '{generated_reply_text}'，返回默认回复。")
                return "嗯..."

            return cleaned_reply

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}] ReplyGenerator 调用LLM或处理回复时出错: {e}", exc_info=True)
            return "抱歉，我现在有点混乱，不知道该怎么表达了。"