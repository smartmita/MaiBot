# -*- coding: utf-8 -*-
# File: action_planner.py
import time
import traceback
from typing import Tuple, Optional, Dict, Any, List

from src.common.logger_manager import get_logger
from src.individuality.individuality import Individuality
from src.plugins.utils.chat_message_builder import build_readable_messages
from ..models.utils_model import LLMRequest
from ...config.config import global_config

# 确保导入路径正确
from .pfc_utils import get_items_from_json, retrieve_contextual_info
from .chat_observer import ChatObserver
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo

logger = get_logger("pfc_action_planner")


# --- 定义 Prompt 模板  ---

# Prompt(1): 首次回复或非连续回复时的决策 Prompt
PROMPT_INITIAL_REPLY = """{persona_text}。现在你在参与一场QQ私聊，请根据以下【所有信息】审慎且灵活的决策下一步行动，可以回复，可以倾听，可以调取知识，甚至可以屏蔽对方：

【当前对话目标】
{goals_str}
【最近行动历史概要】
{action_history_summary}
【你想起来的相关知识】
{retrieved_knowledge_str}
【上一次行动的详细情况和结果】
{last_action_context}
【时间和超时提示】
{time_since_last_bot_message_info}{timeout_context}
【最近的对话记录】(包括你已成功发送的消息 和 新收到的消息)
{chat_history_text}
【你的的回忆】
{retrieved_memory_str}

------
可选行动类型以及解释：
listening: 倾听对方发言，当你认为对方话才说到一半，发言明显未结束时选择
direct_reply: 直接回复对方 (当有新消息需要处理时，通常应选择此项)
rethink_goal: 思考一个对话目标，当你觉得目前对话需要目标，或当前目标不再适用，或话题卡住时选择。注意私聊的环境是灵活的，有可能需要经常选择
end_conversation: 结束对话，对方长时间没回复或者当你觉得对话告一段落时可以选择
block_and_ignore: 更加极端的结束对话方式，直接结束对话并在一段时间内无视对方所有发言（屏蔽），当对话让你感到十分不适，或你遭到各类骚扰时选择

请以JSON格式输出你的决策：
{{
    "action": "选择的行动类型 (必须是上面列表中的一个)",
    "reason": "选择该行动的详细原因 (必须有解释你是如何根据“上一次行动结果”、“对话记录”和自身设定人设做出合理判断的)"
}}

注意：请严格按照JSON格式输出，不要包含任何其他内容。"""

# Prompt(2): 上一次成功回复后，决定继续发言时的决策 Prompt
PROMPT_FOLLOW_UP = """{persona_text}。现在你在参与一场QQ私聊，刚刚你已经回复了对方，请根据以下【所有信息】审慎且灵活的决策下一步行动，可以继续发送新消息，可以等待，可以倾听，可以调取知识，甚至可以屏蔽对方：

【当前对话目标】
{goals_str}
【最近行动历史概要】
{action_history_summary}
【你想起来的相关知识】
{retrieved_knowledge_str}
【上一次行动的详细情况和结果】
{last_action_context}
【时间和超时提示】
{time_since_last_bot_message_info}{timeout_context}
【最近的对话记录】(包括你已成功发送的消息 和 新收到的消息)
{chat_history_text}
【你的的回忆】
{retrieved_memory_str}
------
可选行动类型以及解释：
wait: 暂时不说话，留给对方交互空间，等待对方回复（尤其是在你刚发言后、或上次发言因重复、发言过多被拒时、或不确定做什么时，这是不错的选择）。**重要：仅当没有未读消息时才能选择此项。**
listening: 倾听对方发言（虽然你刚发过言，但如果对方立刻回复且明显话没说完，可以选择这个）
send_new_message: 发送一条新消息继续对话，允许适当的追问、补充、深入话题，或开启相关新话题。**但是避免在因重复被拒后立即使用，也不要在对方没有回复的情况下过多的“消息轰炸”或重复发言**
rethink_goal: 思考一个对话目标，当你觉得目前对话需要目标，或当前目标不再适用，或话题卡住时选择。注意私聊的环境是灵活的，有可能需要经常选择
end_conversation: 结束对话，对方长时间没回复或者当你觉得对话告一段落时可以选择
block_and_ignore: 更加极端的结束对话方式，直接结束对话并在一段时间内无视对方所有发言（屏蔽），当对话让你感到十分不适，或你遭到各类骚扰时选择

请以JSON格式输出你的决策：
{{
    "action": "选择的行动类型 (必须是上面列表中的一个)",
    "reason": "选择该行动的详细原因 (必须有解释你是如何根据“上一次行动结果”、“对话记录”和自身设定人设做出合理判断的。请说明你为什么选择继续发言而不是等待，以及打算发送什么类型的新消息连续发言，必须记录已经发言了几次)"
}}

注意：请严格按照JSON格式输出，不要包含任何其他内容。"""

# 新增：Prompt(3): 决定是否在结束对话前发送告别语
PROMPT_END_DECISION = """{persona_text}。刚刚你决定结束一场 QQ 私聊。

【你们之前的聊天记录】
{chat_history_text}

你觉得你们的对话已经完整结束了吗？有时候，在对话自然结束后再说点什么可能会有点奇怪，但有时也可能需要一条简短的消息来圆满结束。
如果觉得确实有必要再发一条简短、自然、符合你人设的告别消息（比如 "好，下次再聊~" 或 "嗯，先这样吧"），就输出 "yes"。
如果觉得当前状态下直接结束对话更好，没有必要再发消息，就输出 "no"。

请以 JSON 格式输出你的选择：
{{
    "say_bye": "yes/no",
    "reason": "选择 yes 或 no 的原因和内心想法 (简要说明)"
}}

注意：请严格按照 JSON 格式输出，不要包含任何其他内容。"""


class ActionPlanner:
    """行动规划器"""

    def __init__(self, stream_id: str, private_name: str):
        """初始化行动规划器"""
        self.stream_id = stream_id
        self.private_name = private_name
        # 初始化 LLM 请求对象
        try:
             llm_config = global_config.llm_PFC_action_planner
             if not isinstance(llm_config, dict):
                 raise TypeError(f"LLM config 'llm_PFC_action_planner' is not a dictionary: {llm_config}")

             self.llm = LLMRequest(
                 model=llm_config,
                 temperature=llm_config.get("temp", 0.7),
                 max_tokens=1500,
                 request_type="action_planning",
             )
        except TypeError as e:
             logger.error(f"[私聊][{self.private_name}] 初始化 LLMRequest 时配置错误: {e}")
             raise
        except Exception as e:
             logger.error(f"[私聊][{self.private_name}] 初始化 LLMRequest 时发生未知错误: {e}")
             raise

        # 获取个性化信息和机器人名称
        self.personality_info = Individuality.get_instance().get_prompt(x_person=2, level=3)
        self.name = global_config.BOT_NICKNAME
        # 获取 ChatObserver 实例 (单例模式)
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)


    async def plan(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        last_successful_reply_action: Optional[str],
    ) -> Tuple[str, str]:
        """
        规划下一步行动。

        Args:
            observation_info: 观察信息，包含聊天记录、未读消息等。
            conversation_info: 对话信息，包含目标、历史动作等。
            last_successful_reply_action: 上一次成功的回复动作类型 ('direct_reply' 或 'send_new_message' 或 None)。

        Returns:
            Tuple[str, str]: (规划的行动类型, 行动原因)。
        """
        logger.info(f"[私聊][{self.private_name}] 开始规划行动...")
        plan_start_time = time.time()

        # --- 1. 准备 Prompt 输入信息 ---
        try:
            time_since_last_bot_message_info = self._get_bot_last_speak_time_info(observation_info)
            timeout_context = self._get_timeout_context(conversation_info)
            goals_str = self._build_goals_string(conversation_info)
            chat_history_text = await self._build_chat_history_text(observation_info)
            persona_text = f"你的名字是{self.name}，{self.personality_info}。"
            action_history_summary, last_action_context = self._build_action_history_context(conversation_info)
            retrieved_memory_str, retrieved_knowledge_str = await retrieve_contextual_info(
                chat_history_text, self.private_name
            )
            logger.info(
                 f"[私聊][{self.private_name}] (ActionPlanner) 检索完成。记忆: {'有' if '回忆起' in retrieved_memory_str else '无'} / 知识: {'有' if retrieved_knowledge_str and '无相关知识' not in retrieved_knowledge_str and '出错' not in retrieved_knowledge_str else '无'}"
            )
        except Exception as prep_err:
             logger.error(f"[私聊][{self.private_name}] 准备 Prompt 输入时出错: {prep_err}")
             logger.error(traceback.format_exc())
             return "wait", f"准备行动规划输入时出错: {prep_err}"

        # --- 2. 选择并格式化 Prompt ---
        try:
            if last_successful_reply_action in ["direct_reply", "send_new_message"]:
                prompt_template = PROMPT_FOLLOW_UP
                log_msg = "使用 PROMPT_FOLLOW_UP (追问决策)"
            else:
                prompt_template = PROMPT_INITIAL_REPLY
                log_msg = "使用 PROMPT_INITIAL_REPLY (首次/非连续回复决策)"
            logger.debug(f"[私聊][{self.private_name}] {log_msg}")

            prompt = prompt_template.format(
                persona_text=persona_text,
                goals_str=goals_str if goals_str.strip() else "- 目前没有明确对话目标，请考虑设定一个。",
                action_history_summary=action_history_summary,
                last_action_context=last_action_context,
                time_since_last_bot_message_info=time_since_last_bot_message_info,
                timeout_context=timeout_context,
                chat_history_text=chat_history_text if chat_history_text.strip() else "还没有聊天记录。",
                retrieved_memory_str=retrieved_memory_str if retrieved_memory_str else "无相关记忆。",
                retrieved_knowledge_str=retrieved_knowledge_str if retrieved_knowledge_str else "无相关知识。",
            )
            logger.debug(f"[私聊][{self.private_name}] 发送到LLM的最终提示词:\n------\n{prompt}\n------")
        except KeyError as fmt_key_err:
            logger.error(f"[私聊][{self.private_name}] 格式化 Prompt 时缺少键: {fmt_key_err}")
            return "wait", f"格式化 Prompt 时出错 (缺少键: {fmt_key_err})"
        except Exception as fmt_err:
            logger.error(f"[私聊][{self.private_name}] 格式化 Prompt 时发生未知错误: {fmt_err}")
            return "wait", f"格式化 Prompt 时出错: {fmt_err}"

        # --- 3. 调用 LLM 进行初步规划 ---
        try:
            llm_start_time = time.time()
            content, _ = await self.llm.generate_response_async(prompt)
            llm_duration = time.time() - llm_start_time
            logger.debug(f"[私聊][{self.private_name}] LLM (行动规划) 耗时: {llm_duration:.3f} 秒, 原始返回: {content}")

            success, initial_result = get_items_from_json(
                content,
                self.private_name,
                "action",
                "reason",
                default_values={"action": "wait", "reason": "LLM返回格式错误或未提供原因，默认等待"},
            )

            initial_action = initial_result.get("action", "wait")
            initial_reason = initial_result.get("reason", "LLM未提供原因，默认等待")
            logger.info(f"[私聊][{self.private_name}] LLM 初步规划行动: {initial_action}, 原因: {initial_reason}")
        except Exception as llm_err:
            logger.error(f"[私聊][{self.private_name}] 调用 LLM 或解析初步规划结果时出错: {llm_err}")
            logger.error(traceback.format_exc())
            return "wait", f"行动规划 LLM 调用或解析出错: {llm_err}"

        # --- 4. 处理特殊动作 (end_conversation) ---
        final_action = initial_action
        final_reason = initial_reason

        if initial_action == "end_conversation":
            try:
                final_action, final_reason = await self._handle_end_conversation_decision(
                    persona_text, chat_history_text, initial_reason
                )
            except Exception as end_dec_err:
                 logger.error(f"[私聊][{self.private_name}] 处理结束对话决策时出错: {end_dec_err}")
                 logger.warning(f"[私聊][{self.private_name}] 结束决策出错，将按原计划执行 end_conversation")
                 final_action = "end_conversation" # 保持原计划
                 final_reason = initial_reason

        # --- [移除] 不再需要在这里检查 wait 动作的约束 ---
        # elif initial_action == "wait":
        #     # ... (移除之前的检查逻辑) ...
        #     final_action = "wait"
        #     final_reason = initial_reason

        # --- 5. 验证最终行动类型 ---
        valid_actions = [
            "direct_reply", "send_new_message", "wait", "listening",
            "rethink_goal", "end_conversation", "block_and_ignore", "say_goodbye"
        ]
        if final_action not in valid_actions:
            logger.warning(f"[私聊][{self.private_name}] LLM 返回了未知的行动类型: '{final_action}'，强制改为 wait")
            final_reason = f"(原始行动'{final_action}'无效，已强制改为wait) {final_reason}"
            final_action = "wait" # 遇到无效动作，默认等待

        plan_duration = time.time() - plan_start_time
        logger.success(f"[私聊][{self.private_name}] 最终规划行动: {final_action} (总耗时: {plan_duration:.3f} 秒)")
        logger.info(f"[私聊][{self.private_name}] 行动原因: {final_reason}")
        return final_action, final_reason


    # --- Helper methods for preparing prompt inputs  ---

    def _get_bot_last_speak_time_info(self, observation_info: ObservationInfo) -> str:
        """获取机器人上次发言时间提示"""
        
        time_info = ""
        try:
            if not observation_info or not observation_info.bot_id: return ""
            bot_id_str = str(observation_info.bot_id)
            if hasattr(observation_info, "chat_history") and observation_info.chat_history:
                for msg in reversed(observation_info.chat_history):
                    if not isinstance(msg, dict): continue
                    sender_info = msg.get("user_info", {})
                    sender_id = str(sender_info.get("user_id")) if isinstance(sender_info, dict) else None
                    msg_time = msg.get("time")
                    if sender_id == bot_id_str and msg_time:
                        time_diff = time.time() - msg_time
                        if time_diff < 60.0:
                            time_info = f"提示：你上一条成功发送的消息是在 {time_diff:.1f} 秒前。\n"
                        break
        except AttributeError as e: logger.warning(f"[私聊][{self.private_name}] 获取 Bot 上次发言时间时属性错误: {e}")
        except Exception as e: logger.warning(f"[私聊][{self.private_name}] 获取 Bot 上次发言时间时出错: {e}")
        return time_info

    def _get_timeout_context(self, conversation_info: ConversationInfo) -> str:
        """获取超时提示信息"""
        
        timeout_context = ""
        try:
            if hasattr(conversation_info, "goal_list") and conversation_info.goal_list:
                last_goal_item = conversation_info.goal_list[-1]
                last_goal_text = ""
                if isinstance(last_goal_item, dict): last_goal_text = last_goal_item.get("goal", "")
                elif isinstance(last_goal_item, str): last_goal_text = last_goal_item
                if isinstance(last_goal_text, str) and "分钟，" in last_goal_text and "思考接下来要做什么" in last_goal_text:
                    wait_time_str = last_goal_text.split("分钟，")[0].replace("你等待了","").strip()
                    timeout_context = f"重要提示：对方已经长时间（约 {wait_time_str} 分钟）没有回复你的消息了，请基于此情况规划下一步。\n"
                    logger.debug(f"[私聊][{self.private_name}] 检测到超时目标: {last_goal_text}")
        except AttributeError as e: logger.warning(f"[私聊][{self.private_name}] 检查超时目标时属性错误: {e}")
        except Exception as e: logger.warning(f"[私聊][{self.private_name}] 检查超时目标时出错: {e}")
        return timeout_context

    def _build_goals_string(self, conversation_info: ConversationInfo) -> str:
        """构建对话目标字符串"""
        
        goals_str = ""
        try:
            if hasattr(conversation_info, "goal_list") and conversation_info.goal_list:
                recent_goals = conversation_info.goal_list[-3:]
                for goal_item in recent_goals:
                    goal = "目标内容缺失"; reasoning = "没有明确原因"
                    if isinstance(goal_item, dict):
                        goal = goal_item.get("goal", goal); reasoning = goal_item.get("reasoning", reasoning)
                    elif isinstance(goal_item, str): goal = goal_item
                    goal = str(goal) if goal is not None else "目标内容缺失"
                    reasoning = str(reasoning) if reasoning is not None else "没有明确原因"
                    goals_str += f"- 目标：{goal}\n  原因：{reasoning}\n"
                if not goals_str: goals_str = "- 目前没有明确对话目标，请考虑设定一个。\n"
            else: goals_str = "- 目前没有明确对话目标，请考虑设定一个。\n"
        except AttributeError as e: logger.warning(f"[私聊][{self.private_name}] 构建对话目标字符串时属性错误: {e}"); goals_str = "- 获取对话目标时出错。\n"
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 构建对话目标字符串时出错: {e}"); goals_str = "- 构建对话目标时出错。\n"
        return goals_str

    async def _build_chat_history_text(self, observation_info: ObservationInfo) -> str:
        """构建聊天历史记录文本 (包含未处理消息)"""
        
        chat_history_text = ""
        try:
            if hasattr(observation_info, "chat_history_str") and observation_info.chat_history_str: chat_history_text = observation_info.chat_history_str
            elif hasattr(observation_info, "chat_history") and observation_info.chat_history:
                 history_slice = observation_info.chat_history[-20:]
                 chat_history_text = await build_readable_messages(history_slice, replace_bot_name=True, merge_messages=False, timestamp_mode="relative", read_mark=0.0)
            else: chat_history_text = "还没有聊天记录。\n"
            unread_count = getattr(observation_info, 'new_messages_count', 0)
            unread_messages = getattr(observation_info, 'unprocessed_messages', [])
            if unread_count > 0 and unread_messages:
                from ...config.config import global_config
                bot_qq_str = str(global_config.BOT_QQ)
                other_unread_messages = [msg for msg in unread_messages if msg.get("user_info", {}).get("user_id") != bot_qq_str]
                other_unread_count = len(other_unread_messages)
                if other_unread_count > 0:
                    new_messages_str = await build_readable_messages(other_unread_messages, replace_bot_name=True, merge_messages=False, timestamp_mode="relative", read_mark=0.0)
                    chat_history_text += f"\n--- 以下是 {other_unread_count} 条你需要处理的新消息 ---\n{new_messages_str}\n------\n"
                    logger.debug(f"[私聊][{self.private_name}] 向 LLM 追加了 {other_unread_count} 条未读消息。")
        except AttributeError as e: logger.warning(f"[私聊][{self.private_name}] 构建聊天记录文本时属性错误: {e}"); chat_history_text = "[获取聊天记录时出错]\n"
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 处理聊天记录时发生未知错误: {e}"); chat_history_text = "[处理聊天记录时出错]\n"
        return chat_history_text


    def _build_action_history_context(self, conversation_info: ConversationInfo) -> Tuple[str, str]:
        """构建行动历史概要和上一次行动详细情况"""
        
        action_history_summary = "你最近执行的行动历史：\n"; last_action_context = "关于你【上一次尝试】的行动：\n"
        action_history_list: List[Dict[str, Any]] = []
        try:
            if hasattr(conversation_info, "done_action") and conversation_info.done_action: action_history_list = conversation_info.done_action[-5:]
        except AttributeError as e: logger.warning(f"[私聊][{self.private_name}] 获取行动历史时属性错误: {e}")
        except Exception as e: logger.error(f"[私聊][{self.private_name}] 访问行动历史时出错: {e}")
        if not action_history_list:
            action_history_summary += "- 还没有执行过行动。\n"; last_action_context += "- 这是你规划的第一个行动。\n"
        else:
            for i, action_data in enumerate(action_history_list):
                if not isinstance(action_data, dict): logger.warning(f"[私聊][{self.private_name}] 行动历史记录格式错误，跳过: {action_data}"); continue
                action_type = action_data.get("action", "未知动作"); plan_reason = action_data.get("plan_reason", "未知规划原因")
                status = action_data.get("status", "未知状态"); final_reason = action_data.get("final_reason", "")
                action_time = action_data.get("time", "未知时间")
                reason_text = f", 最终原因: “{final_reason}”" if final_reason else ""
                summary_line = f"- 时间:{action_time}, 尝试:'{action_type}', 状态:{status}{reason_text}"
                action_history_summary += summary_line + "\n"
                if i == len(action_history_list) - 1:
                    last_action_context += f"- 上次【规划】的行动是: '{action_type}'\n"
                    last_action_context += f"- 当时规划的【原因】是: {plan_reason}\n"
                    if status == "done": last_action_context += "- 该行动已【成功执行】。\n"
                    elif status == "recall" or status == "error" or status.startswith("cancelled"):
                        last_action_context += "- 但该行动最终【未能成功执行/被取消/出错】。\n"
                        if final_reason: last_action_context += f"- 【重要】失败/取消/错误原因是: “{final_reason}”\n"
                        else: last_action_context += "- 【重要】失败/取消/错误原因未明确记录。\n"
                    elif status == "start": last_action_context += "- 该行动【正在执行中】或【未完成】。\n"
                    else: last_action_context += f"- 该行动当前状态未知: {status}\n"
        return action_history_summary, last_action_context

    # --- Helper method for handling end_conversation decision  ---

    async def _handle_end_conversation_decision(
        self, persona_text: str, chat_history_text: str, initial_reason: str
    ) -> Tuple[str, str]:
        """处理结束对话前的告别决策"""
        logger.info(f"[私聊][{self.private_name}] 初步规划结束对话，进入告别决策...")
        end_decision_prompt = PROMPT_END_DECISION.format(persona_text=persona_text, chat_history_text=chat_history_text)
        logger.debug(f"[私聊][{self.private_name}] 发送到LLM的结束决策提示词:\n------\n{end_decision_prompt}\n------")
        llm_start_time = time.time()
        end_content, _ = await self.llm.generate_response_async(end_decision_prompt)
        llm_duration = time.time() - llm_start_time
        logger.debug(f"[私聊][{self.private_name}] LLM (结束决策) 耗时: {llm_duration:.3f} 秒, 原始返回: {end_content}")
        end_success, end_result = get_items_from_json(end_content, self.private_name, "say_bye", "reason", default_values={"say_bye": "no", "reason": "结束决策LLM返回格式错误，默认不告别"}, required_types={"say_bye": str, "reason": str})
        say_bye_decision = end_result.get("say_bye", "no").lower()
        end_decision_reason = end_result.get("reason", "未提供原因")
        if end_success and say_bye_decision == "yes":
            logger.info(f"[私聊][{self.private_name}] 结束决策: yes, 准备生成告别语. 原因: {end_decision_reason}")
            final_action = "say_goodbye"; final_reason = f"决定发送告别语 (原因: {end_decision_reason})。原结束理由: {initial_reason}"
            return final_action, final_reason
        else:
            logger.info(f"[私聊][{self.private_name}] 结束决策: no, 直接结束对话. 原因: {end_decision_reason}")
            return "end_conversation", initial_reason
