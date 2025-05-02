# 用于访问记忆系统
from src.plugins.memory_system.Hippocampus import HippocampusManager
# 用于访问新的知识库 (LPMM)
from src.plugins.knowledge.knowledge_lib import qa_manager
# 用于访问数据库 (旧知识库需要)
from src.common.database import db
# 用于获取文本的嵌入向量 (旧知识库需要)
from src.plugins.chat.utils import get_embedding
# 可能用于旧知识库提取主题 (如果需要回退到旧方法)
# import jieba # 如果报错说找不到 jieba，可能需要安装: pip install jieba
# import re    # 正则表达式库，通常 Python 自带
from typing import Tuple, List, Dict, Any,Union
from src.common.logger import get_module_logger
from ..models.utils_model import LLMRequest
from ...config.config import global_config
from .chat_observer import ChatObserver
from .reply_checker import ReplyChecker
from src.individuality.individuality import Individuality
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from src.plugins.utils.chat_message_builder import build_readable_messages
import time

logger = get_module_logger("reply_generator")

# --- 定义 Prompt 模板 ---

# Prompt for direct_reply (首次回复)
PROMPT_DIRECT_REPLY = """{persona_text}。现在你在参与一场QQ私聊，请根据以下信息生成一条回复：

当前对话目标：{goals_str}

你有以下这些知识：
{retrieved_knowledge_str}
请你**记住上面的知识**，在回复中有可能会用到。

最近的聊天记录：
{chat_history_text}

{retrieved_memory_str}


请根据上述信息，结合聊天记录，回复对方。该回复应该：
1. 符合对话目标，以"你"的角度发言（不要自己与自己对话！）
2. 符合你的性格特征和身份细节
3. 通俗易懂，自然流畅，像正常聊天一样，简短（通常20字以内，除非特殊情况）
4. 可以适当利用相关知识和回忆，但**不要生硬引用**，若无必要，也可以不利用
5. 自然、得体，结合聊天记录逻辑合理，且没有重复表达同质内容

请注意把握聊天内容，不要回复的太有条理，可以有个性。请分清"你"和对方说的话，不要把"你"说的话当做对方说的话，这是你自己说的话。
可以回复得自然随意自然一些，就像真人一样，注意把握聊天内容，整体风格可以平和、简短，不要刻意突出自身学科背景，不要说你说过的话，可以简短，多简短都可以，但是避免冗长。
请你注意不要输出多余内容(包括前后缀，冒号和引号，括号，表情等)，只输出回复内容。
不要输出多余内容(包括前后缀，冒号和引号，括号，表情包，at或 @等 )。

请直接输出回复内容，不需要任何额外格式。"""

# Prompt for send_new_message (追问/补充)
PROMPT_SEND_NEW_MESSAGE = """{persona_text}。现在你在参与一场QQ私聊，**刚刚你已经发送了一条或多条消息**，现在请根据以下信息再发一条新消息： 

当前对话目标：{goals_str}

你有以下这些知识：
{retrieved_knowledge_str}
请你**记住上面的知识**，在发消息时有可能会用到。

最近的聊天记录：
{chat_history_text}

{retrieved_memory_str}

请根据上述信息，结合聊天记录，继续发一条新消息（例如对之前消息的补充，深入话题，或追问等等）。该消息应该： 
1. 符合对话目标，以"你"的角度发言（不要自己与自己对话！）
2. 符合你的性格特征和身份细节
3. 通俗易懂，自然流畅，像正常聊天一样，简短（通常20字以内，除非特殊情况）
4. 可以适当利用相关知识和回忆，但**不要生硬引用**，若无必要，也可以不利用
5. 跟之前你发的消息自然的衔接，逻辑合理，且没有重复表达同质内容或部分重叠内容

请注意把握聊天内容，不用太有条理，可以有个性。请分清"你"和对方说的话，不要把"你"说的话当做对方说的话，这是你自己说的话。
这条消息可以自然随意自然一些，就像真人一样，注意把握聊天内容，整体风格可以平和、简短，不要刻意突出自身学科背景，不要说你说过的话，可以简短，多简短都可以，但是避免冗长。
请你注意不要输出多余内容(包括前后缀，冒号和引号，括号，表情等)，只输出消息内容。
不要输出多余内容(包括前后缀，冒号和引号，括号，表情包，at或 @等 )。

请直接输出回复内容，不需要任何额外格式。"""

# Prompt for say_goodbye (告别语生成)
PROMPT_FAREWELL = """{persona_text}。你在参与一场 QQ 私聊，现在对话似乎已经结束，你决定再发一条最后的消息来圆满结束。

最近的聊天记录：
{chat_history_text}

请根据上述信息，结合聊天记录，构思一条**简短、自然、符合你人设**的最后的消息。
这条消息应该：
1. 从你自己的角度发言。
2. 符合你的性格特征和身份细节。
3. 通俗易懂，自然流畅，通常很简短。
4. 自然地为这场对话画上句号，避免开启新话题或显得冗长、刻意。

请像真人一样随意自然，**简洁是关键**。
不要输出多余内容（包括前后缀、冒号、引号、括号、表情包、at或@等）。

请直接输出最终的告别消息内容，不需要任何额外格式。"""


class ReplyGenerator:
    """回复生成器"""

    def __init__(self, stream_id: str, private_name: str):
        self.llm = LLMRequest(
            model=global_config.llm_PFC_chat,
            temperature=global_config.llm_PFC_chat["temp"],
            max_tokens=300,
            request_type="reply_generation",
        )
        self.personality_info = Individuality.get_instance().get_prompt(x_person=2, level=3)
        self.name = global_config.BOT_NICKNAME
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.reply_checker = ReplyChecker(stream_id, private_name)
    async def _get_memory_info(self, text: str) -> str:
        """根据文本自动检索相关记忆"""
        memory_prompt = ""
        related_memory_info = ""
        try:
            related_memory = await HippocampusManager.get_instance().get_memory_from_text(
                text=text,
                max_memory_num=2, # 最多获取 2 条记忆
                max_memory_length=2, # 每条记忆长度限制（这个参数含义可能需确认）
                max_depth=3, # 搜索深度
                fast_retrieval=False # 是否快速检索
            )
            if related_memory:
                for memory in related_memory:
                    # memory[0] 是记忆ID, memory[1] 是记忆内容
                    related_memory_info += memory[1] + "\n" # 将记忆内容拼接起来
                if related_memory_info:
                    memory_prompt = f"你回忆起：\n{related_memory_info.strip()}\n(以上是你的回忆，不一定是目前聊天里的人说的，回忆中别人说的事情也不一定是准确的，请记住)\n"
                    logger.debug(f"[私聊][{self.private_name}]自动检索到记忆: {related_memory_info.strip()[:100]}...")
                else:
                    logger.debug(f"[私聊][{self.private_name}]自动检索记忆返回为空。")
            else:
                logger.debug(f"[私聊][{self.private_name}]未自动检索到相关记忆。")
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]自动检索记忆时出错: {e}")
            # memory_prompt = "检索记忆时出错。\n" # 可以选择是否提示错误
        return memory_prompt

    async def _get_prompt_info_old(self, message: str, threshold: float) -> str:
        """
        旧版的知识检索方法，根据消息文本从旧知识库（knowledges collection）检索。
        (移植并简化自 heartflow_prompt_builder.py)
        """
        related_info = ""
        start_time = time.time()
        logger.debug(f"[私聊][{self.private_name}]开始使用旧版知识检索，消息: {message[:30]}...")

        # 简化处理：直接使用整个消息进行查询，不再提取主题
        query_text = message.strip()
        if not query_text:
            logger.debug(f"[私聊][{self.private_name}]旧版知识检索：消息为空，跳过。")
            return ""

        embedding = None
        try:
            embedding = await get_embedding(query_text, request_type="pfc_implicit_knowledge")
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]旧版知识检索：获取嵌入向量时出错: {str(e)}")

        if not embedding:
            logger.error(f"[私聊][{self.private_name}]旧版知识检索：获取嵌入向量失败。")
            return ""

        # 调用我们之前添加的 get_info_from_db 函数
        results = get_info_from_db(embedding, limit=5, threshold=threshold, return_raw=True) # 最多查 5 条

        logger.info(f"[私聊][{self.private_name}]旧版知识库查询完成，耗时: {time.time() - start_time:.3f}秒，获取{len(results)}条结果")

        # 去重和格式化
        unique_contents = set()
        final_results_content = []
        for result in results:
            content = result.get("content", "").strip()
            similarity = result.get("similarity", 0.0)
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
            logger.error(f"[私聊][{self.private_name}]调用 LPMM 知识库 (qa_manager.get_knowledge) 时发生异常: {str(e)}，尝试旧版数据库检索。")

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
            logger.error(f"[私聊][{self.private_name}]调用旧版知识库检索 (_get_prompt_info_old) 时也发生异常: {str(e2)}")
        # 如果两种方法都失败或无结果
        logger.info(f"[私聊][{self.private_name}]自动知识检索总耗时: {time.time() - start_time:.3f}秒，未找到任何相关知识。")
        return "" # 返回空字符串
    
    # 修改 generate 方法签名，增加 action_type 参数
    async def generate(
        self, observation_info: ObservationInfo, conversation_info: ConversationInfo, action_type: str
    ) -> str:
        """生成回复

        Args:
            observation_info: 观察信息
            conversation_info: 对话信息
            action_type: 当前执行的动作类型 ('direct_reply' 或 'send_new_message')

        Returns:
            str: 生成的回复
        """
        # 构建提示词
        logger.debug(
            f"[私聊][{self.private_name}]开始生成回复 (动作类型: {action_type})：当前目标: {conversation_info.goal_list}"
        )

        # --- 构建通用 Prompt 参数 ---
        # (这部分逻辑基本不变)

        # 构建对话目标 (goals_str)
        goals_str = ""
        if conversation_info.goal_list:
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
        else:
            goals_str = "- 目前没有明确对话目标\n"  # 简化无目标情况

        # --- 新增：构建知识信息字符串 ---
        # knowledge_info_str = "【供参考的相关知识和记忆】\n"  # 稍微改下标题，表明是供参考
        # try:
            # 检查 conversation_info 是否有 knowledge_list 并且不为空
            # if hasattr(conversation_info, "knowledge_list") and conversation_info.knowledge_list:
                # 最多只显示最近的 5 条知识
                # recent_knowledge = conversation_info.knowledge_list[-5:]
                # for i, knowledge_item in enumerate(recent_knowledge):
                    # if isinstance(knowledge_item, dict):
                        # query = knowledge_item.get("query", "未知查询")
                        # knowledge = knowledge_item.get("knowledge", "无知识内容")
                        # source = knowledge_item.get("source", "未知来源")
                        # 只取知识内容的前 2000 个字
                        # knowledge_snippet = knowledge[:2000] + "..." if len(knowledge) > 2000 else knowledge
                        # knowledge_info_str += (
                            # f"{i + 1}. 关于 '{query}' (来源: {source}): {knowledge_snippet}\n"  # 格式微调，更简洁
                        # )
                    # else:
                        # knowledge_info_str += f"{i + 1}. 发现一条格式不正确的知识记录。\n"

                # if not recent_knowledge:
                    # knowledge_info_str += "- 暂无。\n"  # 更简洁的提示

            # else:
                # knowledge_info_str += "- 暂无。\n"
        # except AttributeError:
            # logger.warning(f"[私聊][{self.private_name}]ConversationInfo 对象可能缺少 knowledge_list 属性。")
            # knowledge_info_str += "- 获取知识列表时出错。\n"
        # except Exception as e:
            # logger.error(f"[私聊][{self.private_name}]构建知识信息字符串时出错: {e}")
            # knowledge_info_str += "- 处理知识列表时出错。\n"

        # 获取聊天历史记录 (chat_history_text)
        chat_history_text = observation_info.chat_history_str
        if observation_info.new_messages_count > 0 and observation_info.unprocessed_messages:
            new_messages_list = observation_info.unprocessed_messages
            new_messages_str = await build_readable_messages(
                new_messages_list,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
            chat_history_text += f"\n--- 以下是 {observation_info.new_messages_count} 条新消息 ---\n{new_messages_str}"
        elif not chat_history_text:
            chat_history_text = "还没有聊天记录。"

        # 构建 Persona 文本 (persona_text)
        persona_text = f"你的名字是{self.name}，{self.personality_info}。"
        retrieved_memory_str = ""
        retrieved_knowledge_str = ""
        # 使用 chat_history_text 作为检索的上下文，因为它包含了最近的对话和新消息
        retrieval_context = chat_history_text
        if retrieval_context and retrieval_context != "还没有聊天记录。" and retrieval_context != "[构建聊天记录出错]":
            try:
                # 提取记忆
                logger.debug(f"[私聊][{self.private_name}]开始自动检索记忆...")
                retrieved_memory_str = await self._get_memory_info(text=retrieval_context)
                if retrieved_memory_str:
                     logger.info(f"[私聊][{self.private_name}]自动检索到记忆片段。")
                else:
                     logger.info(f"[私聊][{self.private_name}]未自动检索到相关记忆。")

                # 提取知识
                logger.debug(f"[私聊][{self.private_name}]开始自动检索知识...")
                retrieved_knowledge_str = await self._get_prompt_info(message=retrieval_context)
                if retrieved_knowledge_str:
                     logger.info(f"[私聊][{self.private_name}]自动检索到相关知识。")
                else:
                     logger.info(f"[私聊][{self.private_name}]未自动检索到相关知识。")

            except Exception as retrieval_err:
                logger.error(f"[私聊][{self.private_name}]在自动检索记忆/知识时发生错误: {retrieval_err}")
                retrieved_memory_str = "检索记忆时出错。\n"
                retrieved_knowledge_str = "检索知识时出错。\n"
        else:
            logger.debug(f"[私聊][{self.private_name}]聊天记录为空或无效，跳过自动记忆/知识检索。")
            retrieved_memory_str = "无聊天记录，无法自动检索记忆。\n"
            retrieved_knowledge_str = "无聊天记录，无法自动检索知识。\n"

        # --- 选择 Prompt ---
        if action_type == "send_new_message":
            prompt_template = PROMPT_SEND_NEW_MESSAGE
            logger.info(f"[私聊][{self.private_name}]使用 PROMPT_SEND_NEW_MESSAGE (追问生成)")
        elif action_type == "say_goodbye":  # 处理告别动作
            prompt_template = PROMPT_FAREWELL
            logger.info(f"[私聊][{self.private_name}]使用 PROMPT_FAREWELL (告别语生成)")
        else:  # 默认使用 direct_reply 的 prompt (包括 'direct_reply' 或其他未明确处理的类型)
            prompt_template = PROMPT_DIRECT_REPLY
            logger.info(f"[私聊][{self.private_name}]使用 PROMPT_DIRECT_REPLY (首次/非连续回复生成)")

        # --- 格式化最终的 Prompt ---
        prompt = prompt_template.format(
            persona_text=persona_text,
            goals_str=goals_str,
            chat_history_text=chat_history_text,
            # knowledge_info_str=knowledge_info_str,
            retrieved_memory_str=retrieved_memory_str if retrieved_memory_str else "无相关记忆。", # 如果为空则提示无
            retrieved_knowledge_str=retrieved_knowledge_str if retrieved_knowledge_str else "无相关知识。" # 如果为空则提示无
        )

        # --- 调用 LLM 生成 ---
        logger.debug(f"[私聊][{self.private_name}]发送到LLM的生成提示词:\n------\n{prompt}\n------")
        try:
            content, _ = await self.llm.generate_response_async(prompt)
            logger.debug(f"[私聊][{self.private_name}]生成的回复: {content}")
            # 移除旧的检查新消息逻辑，这应该由 conversation 控制流处理
            return content

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]生成回复时出错: {e}")
            return "抱歉，我现在有点混乱，让我重新思考一下..."

    # check_reply 方法保持不变
    async def check_reply(
        self, reply: str, goal: str, chat_history: List[Dict[str, Any]], chat_history_str: str, retry_count: int = 0
    ) -> Tuple[bool, str, bool]:
        """检查回复是否合适
        (此方法逻辑保持不变)
        """
        return await self.reply_checker.check(reply, goal, chat_history, chat_history_str, retry_count)

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
        {"$addFields": {"similarity": {"$divide": ["$dotProduct", {"$max": [{"$multiply": ["$magnitude1", "$magnitude2"]}, 1e-9]}]}}},
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
        # logger.error(f"执行旧知识库聚合查询时出错: {e}") # 暂时注释掉
        results = []

    if not results:
        return "" if not return_raw else []

    if return_raw:
        return results
    else:
        # 返回所有找到的内容，用换行分隔
        return "\n".join(str(result["content"]) for result in results)
