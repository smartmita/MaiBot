import time
from typing import Tuple, Optional, Union  # 增加了 Optional
from src.plugins.memory_system.Hippocampus import HippocampusManager
from src.plugins.knowledge.knowledge_lib import qa_manager
from src.common.database import db
from src.plugins.chat.utils import get_embedding

# import jieba # 如果需要旧版知识库的回退，可能需要
# import re    # 如果需要旧版知识库的回退，可能需要
from src.common.logger_manager import get_logger
from ..models.utils_model import LLMRequest
from ...config.config import global_config
from .chat_observer import ChatObserver
from .pfc_utils import get_items_from_json
from src.individuality.individuality import Individuality
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from src.plugins.utils.chat_message_builder import build_readable_messages


logger = get_logger("pfc_action_planner")


# --- 定义 Prompt 模板 ---

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
direct_reply: 直接回复对方
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
wait: 暂时不说话，留给对方交互空间，等待对方回复（尤其是在你刚发言后、或上次发言因重复、发言过多被拒时、或不确定做什么时，这是不错的选择）
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


# ActionPlanner 类定义，顶格
class ActionPlanner:
    """行动规划器"""

    def __init__(self, stream_id: str, private_name: str):
        self.llm = LLMRequest(
            model=global_config.llm_PFC_action_planner,
            temperature=global_config.llm_PFC_action_planner["temp"],
            max_tokens=1500,
            request_type="action_planning",
        )
        self.personality_info = Individuality.get_instance().get_prompt(x_person=2, level=3)
        self.name = global_config.BOT_NICKNAME
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)

    async def _get_memory_info(self, text: str) -> str:
        """根据文本自动检索相关记忆"""
        memory_prompt = ""
        related_memory_info = ""
        try:
            related_memory = await HippocampusManager.get_instance().get_memory_from_text(
                text=text,
                max_memory_num=2,  # 最多获取 2 条记忆
                max_memory_length=2,  # 每条记忆长度限制（这个参数含义可能需确认）
                max_depth=3,  # 搜索深度
                fast_retrieval=False,  # 是否快速检索
            )
            if related_memory:
                for memory in related_memory:
                    # memory[0] 是记忆ID, memory[1] 是记忆内容
                    related_memory_info += memory[1] + "\n"  # 将记忆内容拼接起来
                if related_memory_info:
                    memory_prompt = f"你回忆起：\n{related_memory_info.strip()}\n(以上是你的回忆，供参考)\n"
                    logger.debug(
                        f"[私聊]决策层[{self.private_name}]自动检索到记忆: {related_memory_info.strip()[:100]}..."
                    )
                else:
                    logger.debug(f"[私聊]决策层[{self.private_name}]自动检索记忆返回为空。")
            else:
                logger.debug(f"[私聊]决策层[{self.private_name}]未自动检索到相关记忆。")
        except Exception as e:
            logger.error(f"[私聊]决策层[{self.private_name}]自动检索记忆时出错: {e}")
            # memory_prompt = "检索记忆时出错。\n" # 可以选择是否提示错误
        return memory_prompt

    async def _get_prompt_info_old(self, message: str, threshold: float) -> str:
        """
        旧版的知识检索方法，根据消息文本从旧知识库（knowledges collection）检索。
        (移植并自 heartflow_prompt_builder.py)
        """
        related_info = ""
        start_time = time.time()
        logger.debug(f"[私聊]决策层[{self.private_name}]开始使用旧版知识检索，消息: {message[:30]}...")

        # 简化处理：直接使用整个消息进行查询，不再提取主题
        query_text = message.strip()
        if not query_text:
            logger.debug(f"[私聊]决策层[{self.private_name}]旧版知识检索：消息为空，跳过。")
            return ""

        embedding = None
        try:
            embedding = await get_embedding(query_text, request_type="pfc_implicit_knowledge")
        except Exception as e:
            logger.error(f"[私聊]决策层[{self.private_name}]旧版知识检索：获取嵌入向量时出错: {str(e)}")

        if not embedding:
            logger.error(f"[私聊]决策层[{self.private_name}]旧版知识检索：获取嵌入向量失败。")
            return ""

        # 调用我们之前添加的 get_info_from_db 函数
        results = get_info_from_db(embedding, limit=5, threshold=threshold, return_raw=True)  # 最多查 5 条

        logger.info(
            f"[私聊][{self.private_name}]旧版知识库查询完成，耗时: {time.time() - start_time:.3f}秒，获取{len(results)}条结果"
        )

        # 去重和格式化
        unique_contents = set()
        final_results_content = []
        for result in results:
            content = result.get("content", "").strip()
            # similarity = result.get("similarity", 0.0)
            if content and content not in unique_contents:
                unique_contents.add(content)
                # 可以选择性地加入相似度信息，或者只加内容
                # final_results_content.append(f"[{similarity:.2f}] {content}")
                final_results_content.append(content)

        if final_results_content:
            related_info = "\n".join(final_results_content)
            logger.debug(f"[私聊][{self.private_name}]旧版知识检索格式化后内容: {related_info[:100]}...")
        else:
            logger.debug(f"[私聊][{self.private_name}]旧版知识检索未找到合适结果或结果为空。")

        logger.info(f"[私聊][{self.private_name}]旧版知识检索总耗时: {time.time() - start_time:.3f}秒")
        return related_info

    async def _get_prompt_info(self, message: str, threshold: float = 0.38) -> str:
        """
        自动检索相关知识的主函数。优先使用 LPMM，失败则回退到旧版。
        (移植自 heartflow_prompt_builder.py)
        """
        related_info = ""
        start_time = time.time()
        message = message.strip()
        if not message:
            logger.debug(f"[私聊][{self.private_name}]自动知识检索：输入消息为空。")
            return ""

        logger.debug(f"[私聊][{self.private_name}]开始自动知识检索，消息: {message[:30]}...")

        # 1. 尝试从 LPMM 知识库获取知识
        try:
            found_knowledge_from_lpmm = qa_manager.get_knowledge(message)
            if found_knowledge_from_lpmm and found_knowledge_from_lpmm.strip():
                related_info = found_knowledge_from_lpmm.strip()
                logger.info(f"[私聊][{self.private_name}]从 LPMM 知识库获取到知识，长度: {len(related_info)}")
                logger.debug(f"[私聊][{self.private_name}]LPMM 知识内容: {related_info[:100]}...")
                # LPMM 成功获取，直接返回
                logger.info(f"[私聊][{self.private_name}]自动知识检索(LPMM)耗时: {time.time() - start_time:.3f}秒")
                return related_info
            else:
                logger.debug(f"[私聊][{self.private_name}]LPMM 知识库未返回有效知识，尝试旧版数据库检索。")
        except Exception as e:
            logger.error(
                f"[私聊][{self.private_name}]调用 LPMM 知识库 (qa_manager.get_knowledge) 时发生异常: {str(e)}，尝试旧版数据库检索。"
            )

        # 2. 如果 LPMM 失败或无结果，尝试旧版数据库
        try:
            knowledge_from_old = await self._get_prompt_info_old(message, threshold=threshold)
            if knowledge_from_old and knowledge_from_old.strip():
                related_info = knowledge_from_old.strip()
                logger.info(f"[私聊][{self.private_name}]从旧版数据库检索到知识，长度: {len(related_info)}")
                # 旧版成功获取，返回
                logger.info(f"[私聊][{self.private_name}]自动知识检索(旧版)耗时: {time.time() - start_time:.3f}秒")
                return related_info
            else:
                logger.debug(f"[私聊][{self.private_name}]旧版数据库也未检索到有效知识。")

        except Exception as e2:
            logger.error(
                f"[私聊][{self.private_name}]调用旧版知识库检索 (_get_prompt_info_old) 时也发生异常: {str(e2)}"
            )

        # 如果两种方法都失败或无结果
        logger.info(
            f"[私聊][{self.private_name}]自动知识检索总耗时: {time.time() - start_time:.3f}秒，未找到任何相关知识。"
        )
        return ""  # 返回空字符串

    # 修改 plan 方法签名，增加 last_successful_reply_action 参数
    async def plan(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        last_successful_reply_action: Optional[str],
    ) -> Tuple[str, str]:
        """规划下一步行动

        Args:
            observation_info: 决策信息
            conversation_info: 对话信息
            last_successful_reply_action: 上一次成功的回复动作类型 ('direct_reply' 或 'send_new_message' 或 None)

        Returns:
            Tuple[str, str]: (行动类型, 行动原因)
        """
        # --- 获取 Bot 上次发言时间信息 ---
        # (这部分逻辑不变)
        time_since_last_bot_message_info = ""
        try:
            bot_id = str(global_config.BOT_QQ)
            if hasattr(observation_info, "chat_history") and observation_info.chat_history:
                for i in range(len(observation_info.chat_history) - 1, -1, -1):
                    msg = observation_info.chat_history[i]
                    if not isinstance(msg, dict):
                        continue
                    sender_info = msg.get("user_info", {})
                    sender_id = str(sender_info.get("user_id")) if isinstance(sender_info, dict) else None
                    msg_time = msg.get("time")
                    if sender_id == bot_id and msg_time:
                        time_diff = time.time() - msg_time
                        if time_diff < 60.0:
                            time_since_last_bot_message_info = (
                                f"提示：你上一条成功发送的消息是在 {time_diff:.1f} 秒前。\n"
                            )
                        break
            else:
                logger.debug(
                    f"[私聊][{self.private_name}]Observation info chat history is empty or not available for bot time check."
                )
        except AttributeError:
            logger.warning(
                f"[私聊][{self.private_name}]ObservationInfo object might not have chat_history attribute yet for bot time check."
            )
        except Exception as e:
            logger.warning(f"[私聊][{self.private_name}]获取 Bot 上次发言时间时出错: {e}")

        # --- 获取超时提示信息 ---
        # (这部分逻辑不变)
        timeout_context = ""
        try:
            if hasattr(conversation_info, "goal_list") and conversation_info.goal_list:
                last_goal_dict = conversation_info.goal_list[-1]
                if isinstance(last_goal_dict, dict) and "goal" in last_goal_dict:
                    last_goal_text = last_goal_dict["goal"]
                    if isinstance(last_goal_text, str) and "分钟，思考接下来要做什么" in last_goal_text:
                        try:
                            timeout_minutes_text = last_goal_text.split("，")[0].replace("你等待了", "")
                            timeout_context = f"重要提示：对方已经长时间（{timeout_minutes_text}）没有回复你的消息了（这可能代表对方繁忙/不想回复/没注意到你的消息等情况，或在对方看来本次聊天已告一段落），请基于此情况规划下一步。\n"
                        except Exception:
                            timeout_context = "重要提示：对方已经长时间没有回复你的消息了（这可能代表对方繁忙/不想回复/没注意到你的消息等情况，或在对方看来本次聊天已告一段落），请基于此情况规划下一步。\n"
            else:
                logger.debug(
                    f"[私聊][{self.private_name}]Conversation info goal_list is empty or not available for timeout check."
                )
        except AttributeError:
            logger.warning(
                f"[私聊][{self.private_name}]ConversationInfo object might not have goal_list attribute yet for timeout check."
            )
        except Exception as e:
            logger.warning(f"[私聊][{self.private_name}]检查超时目标时出错: {e}")

        # --- 构建通用 Prompt 参数 ---
        logger.debug(
            f"[私聊][{self.private_name}]开始规划行动：当前目标: {getattr(conversation_info, 'goal_list', '不可用')}"
        )

        # 构建对话目标 (goals_str)
        goals_str = ""
        try:
            if hasattr(conversation_info, "goal_list") and conversation_info.goal_list:
                for goal_reason in conversation_info.goal_list:
                    if isinstance(goal_reason, dict):
                        goal = goal_reason.get("goal", "目标内容缺失")
                        reasoning = goal_reason.get("reasoning", "没有明确原因")
                    else:
                        goal = str(goal_reason)
                        reasoning = "没有明确原因"

                    goal = str(goal) if goal is not None else "目标内容缺失"
                    reasoning = str(reasoning) if reasoning is not None else "没有明确原因"
                    goals_str += f"- 目标：{goal}\n  原因：{reasoning}\n"

                if not goals_str:
                    goals_str = "- 目前没有明确对话目标，请考虑设定一个。\n"
            else:
                goals_str = "- 目前没有明确对话目标，请考虑设定一个。\n"
        except AttributeError:
            logger.warning(
                f"[私聊][{self.private_name}]ConversationInfo object might not have goal_list attribute yet."
            )
            goals_str = "- 获取对话目标时出错。\n"
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]构建对话目标字符串时出错: {e}")
            goals_str = "- 构建对话目标时出错。\n"

        # --- 知识信息字符串构建开始 ---
        # knowledge_info_str = "【已获取的相关知识和记忆】\n"
        # try:
        # 检查 conversation_info 是否有 knowledge_list 并且不为空
        # if hasattr(conversation_info, "knowledge_list") and conversation_info.knowledge_list:
        # 最多只显示最近的 5 条知识，防止 Prompt 过长
        # recent_knowledge = conversation_info.knowledge_list[-5:]
        # for i, knowledge_item in enumerate(recent_knowledge):
        # if isinstance(knowledge_item, dict):
        # query = knowledge_item.get("query", "未知查询")
        # knowledge = knowledge_item.get("knowledge", "无知识内容")
        # source = knowledge_item.get("source", "未知来源")
        # 只取知识内容的前 2000 个字，避免太长
        # knowledge_snippet = knowledge[:2000] + "..." if len(knowledge) > 2000 else knowledge
        # knowledge_info_str += (
        # f"{i + 1}. 关于 '{query}' 的知识 (来源: {source}):\n   {knowledge_snippet}\n"
        # )
        # else:
        # 处理列表里不是字典的异常情况
        # knowledge_info_str += f"{i + 1}. 发现一条格式不正确的知识记录。\n"

        # if not recent_knowledge:  # 如果 knowledge_list 存在但为空
        # knowledge_info_str += "- 暂无相关知识和记忆。\n"

        # else:
        # 如果 conversation_info 没有 knowledge_list 属性，或者列表为空
        # knowledge_info_str += "- 暂无相关知识记忆。\n"
        # except AttributeError:
        # logger.warning(f"[私聊][{self.private_name}]ConversationInfo 对象可能缺少 knowledge_list 属性。")
        # knowledge_info_str += "- 获取知识列表时出错。\n"
        # except Exception as e:
        # logger.error(f"[私聊][{self.private_name}]构建知识信息字符串时出错: {e}")
        # knowledge_info_str += "- 处理知识列表时出错。\n"
        # --- 知识信息字符串构建结束 ---

        # 获取聊天历史记录 (chat_history_text)
        try:
            if hasattr(observation_info, "chat_history") and observation_info.chat_history:
                chat_history_text = observation_info.chat_history_str
                if not chat_history_text:
                    chat_history_text = "还没有聊天记录。\n"
            else:
                chat_history_text = "还没有聊天记录。\n"

            if hasattr(observation_info, "new_messages_count") and observation_info.new_messages_count > 0:
                if hasattr(observation_info, "unprocessed_messages") and observation_info.unprocessed_messages:
                    new_messages_list = observation_info.unprocessed_messages
                    new_messages_str = await build_readable_messages(
                        new_messages_list,
                        replace_bot_name=True,
                        merge_messages=False,
                        timestamp_mode="relative",
                        read_mark=0.0,
                    )
                    chat_history_text += (
                        f"\n--- 以下是 {observation_info.new_messages_count} 条新消息 ---\n{new_messages_str}"
                    )
                else:
                    logger.warning(
                        f"[私聊][{self.private_name}]ObservationInfo has new_messages_count > 0 but unprocessed_messages is empty or missing."
                    )
        except AttributeError:
            logger.warning(
                f"[私聊][{self.private_name}]ObservationInfo object might be missing expected attributes for chat history."
            )
            chat_history_text = "获取聊天记录时出错。\n"
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]处理聊天记录时发生未知错误: {e}")
            chat_history_text = "处理聊天记录时出错。\n"

        # 构建 Persona 文本 (persona_text)
        persona_text = f"你的名字是{self.name}，{self.personality_info}。"

        # 构建行动历史和上一次行动结果 (action_history_summary, last_action_context)
        # (这部分逻辑不变)
        action_history_summary = "你最近执行的行动历史：\n"
        last_action_context = "关于你【上一次尝试】的行动：\n"
        action_history_list = []
        try:
            if hasattr(conversation_info, "done_action") and conversation_info.done_action:
                action_history_list = conversation_info.done_action[-5:]
            else:
                logger.debug(f"[私聊][{self.private_name}]Conversation info done_action is empty or not available.")
        except AttributeError:
            logger.warning(
                f"[私聊][{self.private_name}]ConversationInfo object might not have done_action attribute yet."
            )
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]访问行动历史时出错: {e}")

        if not action_history_list:
            action_history_summary += "- 还没有执行过行动。\n"
            last_action_context += "- 这是你规划的第一个行动。\n"
        else:
            for i, action_data in enumerate(action_history_list):
                action_type = "未知"
                plan_reason = "未知"
                status = "未知"
                final_reason = ""
                action_time = ""

                if isinstance(action_data, dict):
                    action_type = action_data.get("action", "未知")
                    plan_reason = action_data.get("plan_reason", "未知规划原因")
                    status = action_data.get("status", "未知")
                    final_reason = action_data.get("final_reason", "")
                    action_time = action_data.get("time", "")
                elif isinstance(action_data, tuple):
                    # 假设旧格式兼容
                    if len(action_data) > 0:
                        action_type = action_data[0]
                    if len(action_data) > 1:
                        plan_reason = action_data[1]  # 可能是规划原因或最终原因
                    if len(action_data) > 2:
                        status = action_data[2]
                    if status == "recall" and len(action_data) > 3:
                        final_reason = action_data[3]
                    elif status == "done" and action_type in ["direct_reply", "send_new_message"]:
                        plan_reason = "成功发送"  # 简化显示

                reason_text = f", 失败/取消原因: {final_reason}" if final_reason else ""
                summary_line = f"- 时间:{action_time}, 尝试行动:'{action_type}', 状态:{status}{reason_text}"
                action_history_summary += summary_line + "\n"

                if i == len(action_history_list) - 1:
                    last_action_context += f"- 上次【规划】的行动是: '{action_type}'\n"
                    last_action_context += f"- 当时规划的【原因】是: {plan_reason}\n"
                    if status == "done":
                        last_action_context += "- 该行动已【成功执行】。\n"
                        # 记录这次成功的行动类型，供下次决策
                        # self.last_successful_action_type = action_type # 不在这里记录，由 conversation 控制
                    elif status == "recall":
                        last_action_context += "- 但该行动最终【未能执行/被取消】。\n"
                        if final_reason:
                            last_action_context += f"- 【重要】失败/取消的具体原因是: “{final_reason}”\n"
                        else:
                            last_action_context += "- 【重要】失败/取消原因未明确记录。\n"
                        # self.last_successful_action_type = None # 行动失败，清除记录
                    else:
                        last_action_context += f"- 该行动当前状态: {status}\n"
                        # self.last_successful_action_type = None # 非完成状态，清除记录

                retrieved_memory_str_planner = ""
        retrieved_knowledge_str_planner = ""
        retrieval_context = chat_history_text  # 使用聊天记录作为检索上下文
        if retrieval_context and retrieval_context != "还没有聊天记录。" and retrieval_context != "[构建聊天记录出错]":
            try:
                logger.debug(f"[私聊][{self.private_name}] (ActionPlanner) 开始自动检索记忆...")
                retrieved_memory_str_planner = await self._get_memory_info(text=retrieval_context)
                logger.info(
                    f"[私聊][{self.private_name}] (ActionPlanner) 自动检索记忆 {'完成' if retrieved_memory_str_planner else '无结果'}。"
                )

                logger.debug(f"[私聊][{self.private_name}] (ActionPlanner) 开始自动知识检索...")
                retrieved_knowledge_str_planner = await self._get_prompt_info(message=retrieval_context)
                logger.info(
                    f"[私聊][{self.private_name}] (ActionPlanner) 自动检索知识 {'完成' if retrieved_knowledge_str_planner else '无结果'}。"
                )
            except Exception as retrieval_err:
                logger.error(f"[私聊][{self.private_name}] (ActionPlanner) 自动检索时出错: {retrieval_err}")
                retrieved_memory_str_planner = "检索记忆时出错。\n"
                retrieved_knowledge_str_planner = "检索知识时出错。\n"
        else:
            logger.debug(f"[私聊][{self.private_name}] (ActionPlanner) 无有效聊天记录，跳过自动检索。")
            retrieved_memory_str_planner = "无聊天记录无法检索记忆。\n"
            retrieved_knowledge_str_planner = "无聊天记录无法检索知识。\n"

        # --- 选择 Prompt ---
        if last_successful_reply_action in ["direct_reply", "send_new_message"]:
            prompt_template = PROMPT_FOLLOW_UP
            logger.debug(f"[私聊][{self.private_name}]使用 PROMPT_FOLLOW_UP (追问决策)")
        else:
            prompt_template = PROMPT_INITIAL_REPLY
            logger.debug(f"[私聊][{self.private_name}]使用 PROMPT_INITIAL_REPLY (首次/非连续回复决策)")

        # --- 格式化最终的 Prompt ---
        prompt = prompt_template.format(
            persona_text=persona_text,
            goals_str=goals_str if goals_str.strip() else "- 目前没有明确对话目标，请考虑设定一个。",
            action_history_summary=action_history_summary,
            last_action_context=last_action_context,
            time_since_last_bot_message_info=time_since_last_bot_message_info,
            timeout_context=timeout_context,
            chat_history_text=chat_history_text if chat_history_text.strip() else "还没有聊天记录。",
            # knowledge_info_str=knowledge_info_str,
            retrieved_memory_str=retrieved_memory_str_planner if retrieved_memory_str_planner else "无相关记忆。",
            retrieved_knowledge_str=retrieved_knowledge_str_planner
            if retrieved_knowledge_str_planner
            else "无相关知识。",
        )

        logger.debug(f"[私聊][{self.private_name}]发送到LLM的最终提示词:\n------\n{prompt}\n------")
        try:
            content, _ = await self.llm.generate_response_async(prompt)
            logger.debug(f"[私聊][{self.private_name}]LLM (行动规划) 原始返回内容: {content}")

            # --- 初始行动规划解析 ---
            success, initial_result = get_items_from_json(
                content,
                self.private_name,
                "action",
                "reason",
                default_values={"action": "wait", "reason": "LLM返回格式错误或未提供原因，默认等待"},
            )

            initial_action = initial_result.get("action", "wait")
            initial_reason = initial_result.get("reason", "LLM未提供原因，默认等待")

            # 检查是否需要进行结束对话决策 ---
            if initial_action == "end_conversation":
                logger.info(f"[私聊][{self.private_name}]初步规划结束对话，进入告别决策...")

                # 使用新的 PROMPT_END_DECISION
                end_decision_prompt = PROMPT_END_DECISION.format(
                    persona_text=persona_text,  # 复用之前的 persona_text
                    chat_history_text=chat_history_text,  # 复用之前的 chat_history_text
                )

                logger.debug(
                    f"[私聊][{self.private_name}]发送到LLM的结束决策提示词:\n------\n{end_decision_prompt}\n------"
                )
                try:
                    end_content, _ = await self.llm.generate_response_async(end_decision_prompt)  # 再次调用LLM
                    logger.debug(f"[私聊][{self.private_name}]LLM (结束决策) 原始返回内容: {end_content}")

                    # 解析结束决策的JSON
                    end_success, end_result = get_items_from_json(
                        end_content,
                        self.private_name,
                        "say_bye",
                        "reason",
                        default_values={"say_bye": "no", "reason": "结束决策LLM返回格式错误，默认不告别"},
                        required_types={"say_bye": str, "reason": str},  # 明确类型
                    )

                    say_bye_decision = end_result.get("say_bye", "no").lower()  # 转小写方便比较
                    end_decision_reason = end_result.get("reason", "未提供原因")

                    if end_success and say_bye_decision == "yes":
                        # 决定要告别，返回新的 'say_goodbye' 动作
                        logger.info(
                            f"[私聊][{self.private_name}]结束决策: yes, 准备生成告别语. 原因: {end_decision_reason}"
                        )
                        # 注意：这里的 reason 可以考虑拼接初始原因和结束决策原因，或者只用结束决策原因
                        final_action = "say_goodbye"
                        final_reason = f"决定发送告别语。决策原因: {end_decision_reason} (原结束理由: {initial_reason})"
                        return final_action, final_reason
                    else:
                        # 决定不告别 (包括解析失败或明确说no)
                        logger.info(
                            f"[私聊][{self.private_name}]结束决策: no, 直接结束对话. 原因: {end_decision_reason}"
                        )
                        # 返回原始的 'end_conversation' 动作
                        final_action = "end_conversation"
                        final_reason = initial_reason  # 保持原始的结束理由
                        return final_action, final_reason

                except Exception as end_e:
                    logger.error(f"[私聊][{self.private_name}]调用结束决策LLM或处理结果时出错: {str(end_e)}")
                    # 出错时，默认执行原始的结束对话
                    logger.warning(f"[私聊][{self.private_name}]结束决策出错，将按原计划执行 end_conversation")
                    return "end_conversation", initial_reason  # 返回原始动作和原因

            else:
                action = initial_action
                reason = initial_reason

                # 验证action类型 (保持不变)
                valid_actions = [
                    "direct_reply",
                    "send_new_message",
                    "wait",
                    "listening",
                    "rethink_goal",
                    "end_conversation",  # 仍然需要验证，因为可能从上面决策后返回
                    "block_and_ignore",
                    "say_goodbye",  # 也要验证这个新动作
                ]
                if action not in valid_actions:
                    logger.warning(f"[私聊][{self.private_name}]LLM返回了未知的行动类型: '{action}'，强制改为 wait")
                    reason = f"(原始行动'{action}'无效，已强制改为wait) {reason}"
                    action = "wait"

                logger.info(f"[私聊][{self.private_name}]规划的行动: {action}")
                logger.info(f"[私聊][{self.private_name}]行动原因: {reason}")
                return action, reason

        except Exception as e:
            # 外层异常处理保持不变
            logger.error(f"[私聊][{self.private_name}]规划行动时调用 LLM 或处理结果出错: {str(e)}")
            return "wait", f"行动规划处理中发生错误，暂时等待: {str(e)}"


def get_info_from_db(
    query_embedding: list, limit: int = 1, threshold: float = 0.5, return_raw: bool = False
) -> Union[str, list]:
    """
    从旧知识库 (knowledges collection) 中根据嵌入向量相似度检索信息。
    (移植自 heartflow_prompt_builder.py)
    """
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
        # 防止除以零错误，添加一个小的 epsilon
        {
            "$addFields": {
                "similarity": {
                    "$divide": ["$dotProduct", {"$max": [{"$multiply": ["$magnitude1", "$magnitude2"]}, 1e-9]}]
                }
            }
        },
        {
            "$match": {
                "similarity": {"$gte": threshold}  # 只保留相似度大于等于阈值的结果
            }
        },
        {"$sort": {"similarity": -1}},
        {"$limit": limit},
        {"$project": {"content": 1, "similarity": 1}},
    ]

    try:
        results = list(db.knowledges.aggregate(pipeline))
        # 注意：这里的 logger 需要能访问到，或者在这个函数里获取 logger 实例
        # logger.debug(f"旧知识库查询结果数量: {len(results)}") # 暂时注释掉，避免 logger 未定义
    except Exception as e:
        logger.debug(f"执行旧知识库聚合查询时出错: {e}")
        results = []

    if not results:
        return "" if not return_raw else []

    if return_raw:
        return results
    else:
        # 返回所有找到的内容，用换行分隔
        return "\n".join(str(result["content"]) for result in results)
