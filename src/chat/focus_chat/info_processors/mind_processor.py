from src.chat.heart_flow.observation.chatting_observation import ChattingObservation
from src.chat.heart_flow.observation.observation import Observation
from src.chat.models.utils_model import LLMRequest
from src.config.config import global_config
import time
import traceback
from src.common.logger_manager import get_logger
from src.individuality.individuality import Individuality
import random
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.utils.json_utils import safe_json_dumps
from src.chat.message_receive.chat_stream import chat_manager
import difflib
from src.chat.person_info.relationship_manager import relationship_manager
from .base_processor import BaseProcessor
from src.chat.focus_chat.info.mind_info import MindInfo
from typing import List, Optional
from src.chat.heart_flow.observation.hfcloop_observation import HFCloopObservation
from src.chat.focus_chat.info_processors.processor_utils import (
    calculate_similarity,
    calculate_replacement_probability,
    get_spark,
)
from typing import Dict
from src.chat.focus_chat.info.info_base import InfoBase

logger = get_logger("processor")


def init_prompt():
    # --- Group Chat Prompt ---
    group_prompt = """
你的名字是{bot_name}
{memory_str}
{extra_info}
{relation_prompt}
{cycle_info_block}
现在是{time_now}，你正在上网，和qq群里的网友们聊天，以下是正在进行的聊天内容：
{chat_observe_info}

以下是你之前对聊天的观察和规划，你的名字是{bot_name}：
{last_mind}

现在请你继续输出观察和规划，输出要求：
1. 先关注未读新消息的内容和近期回复历史
2. 根据新信息，修改和删除之前的观察和规划
3. 根据聊天内容继续输出观察和规划，{hf_do_next}
4. 注意群聊的时间线索，话题由谁发起，进展状况如何，思考聊天的时间线。
6. 语言简洁自然，不要分点，不要浮夸，不要修辞，仅输出思考内容就好"""
    Prompt(group_prompt, "sub_heartflow_prompt_before")

    # --- Private Chat Prompt ---
    private_prompt = """
{memory_str}
{extra_info}
{relation_prompt}
你的名字是{bot_name},{prompt_personality},你现在{mood_info}
{cycle_info_block}
现在是{time_now}，你正在上网，和 {chat_target_name} 私聊，以下是你们的聊天内容：
{chat_observe_info}
以下是你之前对聊天的观察和规划：
{last_mind}
请仔细阅读聊天内容，想想你和 {chat_target_name} 的关系，回顾你们刚刚的交流,你刚刚发言和对方的反应，思考聊天的主题。
请思考你要不要回复以及如何回复对方。
思考并输出你的内心想法
输出要求：
1. 根据聊天内容生成你的想法，{hf_do_next}
2. 不要分点、不要使用表情符号
3. 避免多余符号(冒号、引号、括号等)
4. 语言简洁自然，不要浮夸
5. 如果你刚发言，对方没有回复你，请谨慎回复"""
    Prompt(private_prompt, "sub_heartflow_prompt_private_before")


class MindProcessor(BaseProcessor):
    log_prefix = "聊天思考"

    def __init__(self, subheartflow_id: str):
        super().__init__()

        self.subheartflow_id = subheartflow_id

        self.llm_model = LLMRequest(
            model=global_config.model.sub_heartflow,
            temperature=global_config.model.sub_heartflow["temp"],
            max_tokens=800,
            request_type="sub_heart_flow",
        )

        self.current_mind = ""
        self.past_mind = []
        self.structured_info = []
        self.structured_info_str = ""

        name = chat_manager.get_stream_name(self.subheartflow_id)
        self.log_prefix = f"[{name}] "
        self._update_structured_info_str()

    def _update_structured_info_str(self):
        """根据 structured_info 更新 structured_info_str"""
        if not self.structured_info:
            self.structured_info_str = ""
            return

        lines = ["【信息】"]
        for item in self.structured_info:
            # 简化展示，突出内容和类型，包含TTL供调试
            type_str = item.get("type", "未知类型")
            content_str = item.get("content", "")

            if type_str == "info":
                lines.append(f"刚刚: {content_str}")
            elif type_str == "memory":
                lines.append(f"{content_str}")
            elif type_str == "comparison_result":
                lines.append(f"数字大小比较结果: {content_str}")
            elif type_str == "time_info":
                lines.append(f"{content_str}")
            elif type_str == "lpmm_knowledge":
                lines.append(f"你知道：{content_str}")
            else:
                lines.append(f"{type_str}的信息: {content_str}")

        self.structured_info_str = "\n".join(lines)
        logger.debug(f"{self.log_prefix} 更新 structured_info_str: \n{self.structured_info_str}")

    async def process_info(
        self, observations: Optional[List[Observation]] = None, running_memorys: Optional[List[Dict]] = None, *infos
    ) -> List[InfoBase]:
        """处理信息对象

        Args:
            *infos: 可变数量的InfoBase类型的信息对象

        Returns:
            List[InfoBase]: 处理后的结构化信息列表
        """
        current_mind = await self.do_thinking_before_reply(observations, running_memorys)

        mind_info = MindInfo()
        mind_info.set_current_mind(current_mind)

        return [mind_info]

    async def do_thinking_before_reply(
        self, observations: Optional[List[Observation]] = None, running_memorys: Optional[List[Dict]] = None
    ):
        """
        在回复前进行思考，生成内心想法并收集工具调用结果

        参数:
            observations: 观察信息

        返回:
            如果return_prompt为False:
                tuple: (current_mind, past_mind) 当前想法和过去的想法列表
            如果return_prompt为True:
                tuple: (current_mind, past_mind, prompt) 当前想法、过去的想法列表和使用的prompt
        """

        # ---------- 0. 更新和清理 structured_info ----------
        if self.structured_info:
            updated_info = []
            for item in self.structured_info:
                item["ttl"] -= 1
                if item["ttl"] > 0:
                    updated_info.append(item)
                else:
                    logger.debug(f"{self.log_prefix} 移除过期的 structured_info 项: {item['id']}")
            self.structured_info = updated_info
            self._update_structured_info_str()
        logger.debug(
            f"{self.log_prefix} 当前完整的 structured_info: {safe_json_dumps(self.structured_info, ensure_ascii=False)}"
        )

        memory_str = ""
        if running_memorys:
            memory_str = "以下是当前在聊天中，你回忆起的记忆：\n"
            for running_memory in running_memorys:
                memory_str += f"{running_memory['topic']}: {running_memory['content']}\n"

        # ---------- 1. 准备基础数据 ----------
        # 获取现有想法和情绪状态
        previous_mind = self.current_mind if self.current_mind else ""

        if observations is None:
            observations = []
        for observation in observations:
            if isinstance(observation, ChattingObservation):
                # 获取聊天元信息
                is_group_chat = observation.is_group_chat
                chat_target_info = observation.chat_target_info
                chat_target_name = "对方"  # 私聊默认名称
                if not is_group_chat and chat_target_info:
                    # 优先使用 user_nickname，最后回退到默认值
                    chat_target_name = (
                        chat_target_info.get("user_nickname") or chat_target_name
                    )
                # 获取聊天内容
                chat_observe_info = observation.get_observe_info()
                person_list = observation.person_list
            if isinstance(observation, HFCloopObservation):
                hfcloop_observe_info = observation.get_observe_info()

        # ---------- 3. 准备个性化数据 ----------
        # 获取个性化信息
        individuality = Individuality.get_instance()

        relation_prompt = ""
        for person in person_list:
            relation_prompt += await relationship_manager.build_relationship_info(person, is_id=True)

        # 构建个性部分
        # prompt_personality = individuality.get_prompt(x_person=2, level=2)

        # 获取当前时间
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        spark_prompt = get_spark()

        # ---------- 5. 构建最终提示词 ----------
        template_name = "sub_heartflow_prompt_before" if is_group_chat else "sub_heartflow_prompt_private_before"
        logger.debug(f"{self.log_prefix} 使用{'群聊' if is_group_chat else '私聊'}思考模板")

        prompt = (await global_prompt_manager.get_prompt_async(template_name)).format(
            memory_str=memory_str,
            extra_info=self.structured_info_str,
            # prompt_personality=prompt_personality,
            relation_prompt=relation_prompt,
            bot_name=individuality.name,
            time_now=time_now,
            chat_observe_info=chat_observe_info,
            # mood_info="mood_info",
            hf_do_next=spark_prompt,
            last_mind=previous_mind,
            cycle_info_block=hfcloop_observe_info,
            chat_target_name=chat_target_name,
        )

        # 在构建完提示词后，生成最终的prompt字符串
        final_prompt = prompt

        content = ""  # 初始化内容变量

        try:
            # 调用LLM生成响应
            response, _ = await self.llm_model.generate_response_async(prompt=final_prompt)

            # 直接使用LLM返回的文本响应作为 content
            content = response if response else ""

        except Exception as e:
            # 处理总体异常
            logger.error(f"{self.log_prefix} 执行LLM请求或处理响应时出错: {e}")
            logger.error(traceback.format_exc())
            content = "思考过程中出现错误"

        # 记录初步思考结果
        logger.debug(f"{self.log_prefix} 思考prompt: \n{final_prompt}\n")

        # 处理空响应情况
        if not content:
            content = "(不知道该想些什么...)"
            logger.warning(f"{self.log_prefix} LLM返回空结果，思考失败。")

        # ---------- 8. 更新思考状态并返回结果 ----------
        logger.info(f"{self.log_prefix} 思考结果: {content}")
        # 更新当前思考内容
        self.update_current_mind(content)

        return content

    def update_current_mind(self, response):
        if self.current_mind:  # 只有当 current_mind 非空时才添加到 past_mind
            self.past_mind.append(self.current_mind)
        self.current_mind = response

    def de_similar(self, previous_mind, new_content):
        try:
            similarity = calculate_similarity(previous_mind, new_content)
            replacement_prob = calculate_replacement_probability(similarity)
            logger.debug(f"{self.log_prefix} 新旧想法相似度: {similarity:.2f}, 替换概率: {replacement_prob:.2f}")

            # 定义词语列表 (移到判断之前)
            yu_qi_ci_liebiao = ["嗯", "哦", "啊", "唉", "哈", "唔"]
            zhuan_zhe_liebiao = ["但是", "不过", "然而", "可是", "只是"]
            cheng_jie_liebiao = ["然后", "接着", "此外", "而且", "另外"]
            zhuan_jie_ci_liebiao = zhuan_zhe_liebiao + cheng_jie_liebiao

            if random.random() < replacement_prob:
                # 相似度非常高时，尝试去重或特殊处理
                if similarity == 1.0:
                    logger.debug(f"{self.log_prefix} 想法完全重复 (相似度 1.0)，执行特殊处理...")
                    # 随机截取大约一半内容
                    if len(new_content) > 1:  # 避免内容过短无法截取
                        split_point = max(
                            1, len(new_content) // 2 + random.randint(-len(new_content) // 4, len(new_content) // 4)
                        )
                        truncated_content = new_content[:split_point]
                    else:
                        truncated_content = new_content  # 如果只有一个字符或者为空，就不截取了

                    # 添加语气词和转折/承接词
                    yu_qi_ci = random.choice(yu_qi_ci_liebiao)
                    zhuan_jie_ci = random.choice(zhuan_jie_ci_liebiao)
                    content = f"{yu_qi_ci}{zhuan_jie_ci}，{truncated_content}"
                    logger.debug(f"{self.log_prefix} 想法重复，特殊处理后: {content}")

                else:
                    # 相似度较高但非100%，执行标准去重逻辑
                    logger.debug(f"{self.log_prefix} 执行概率性去重 (概率: {replacement_prob:.2f})...")
                    logger.debug(
                        f"{self.log_prefix} previous_mind类型: {type(previous_mind)}, new_content类型: {type(new_content)}"
                    )

                    matcher = difflib.SequenceMatcher(None, previous_mind, new_content)
                    logger.debug(f"{self.log_prefix} matcher类型: {type(matcher)}")

                    deduplicated_parts = []
                    last_match_end_in_b = 0

                    # 获取并记录所有匹配块
                    matching_blocks = matcher.get_matching_blocks()
                    logger.debug(f"{self.log_prefix} 匹配块数量: {len(matching_blocks)}")
                    logger.debug(
                        f"{self.log_prefix} 匹配块示例(前3个): {matching_blocks[:3] if len(matching_blocks) > 3 else matching_blocks}"
                    )

                    # get_matching_blocks()返回形如[(i, j, n), ...]的列表，其中i是a中的索引，j是b中的索引，n是匹配的长度
                    for idx, match in enumerate(matching_blocks):
                        if not isinstance(match, tuple):
                            logger.error(f"{self.log_prefix} 匹配块 {idx} 不是元组类型，而是 {type(match)}: {match}")
                            continue

                        try:
                            _i, j, n = match  # 解包元组为三个变量
                            logger.debug(f"{self.log_prefix} 匹配块 {idx}: i={_i}, j={j}, n={n}")

                            if last_match_end_in_b < j:
                                # 确保添加的是字符串，而不是元组
                                try:
                                    non_matching_part = new_content[last_match_end_in_b:j]
                                    logger.debug(
                                        f"{self.log_prefix} 添加非匹配部分: '{non_matching_part}', 类型: {type(non_matching_part)}"
                                    )
                                    if not isinstance(non_matching_part, str):
                                        logger.warning(
                                            f"{self.log_prefix} 非匹配部分不是字符串类型: {type(non_matching_part)}"
                                        )
                                        non_matching_part = str(non_matching_part)
                                    deduplicated_parts.append(non_matching_part)
                                except Exception as e:
                                    logger.error(f"{self.log_prefix} 处理非匹配部分时出错: {e}")
                                    logger.error(traceback.format_exc())
                            last_match_end_in_b = j + n
                        except Exception as e:
                            logger.error(f"{self.log_prefix} 处理匹配块时出错: {e}")
                            logger.error(traceback.format_exc())

                    logger.debug(f"{self.log_prefix} 去重前部分列表: {deduplicated_parts}")
                    logger.debug(f"{self.log_prefix} 列表元素类型: {[type(part) for part in deduplicated_parts]}")

                    # 确保所有元素都是字符串
                    deduplicated_parts = [str(part) for part in deduplicated_parts]

                    # 防止列表为空
                    if not deduplicated_parts:
                        logger.warning(f"{self.log_prefix} 去重后列表为空，添加空字符串")
                        deduplicated_parts = [""]

                    logger.debug(f"{self.log_prefix} 处理后的部分列表: {deduplicated_parts}")

                    try:
                        deduplicated_content = "".join(deduplicated_parts).strip()
                        logger.debug(f"{self.log_prefix} 拼接后的去重内容: '{deduplicated_content}'")
                    except Exception as e:
                        logger.error(f"{self.log_prefix} 拼接去重内容时出错: {e}")
                        logger.error(traceback.format_exc())
                        deduplicated_content = ""

                    if deduplicated_content:
                        # 根据概率决定是否添加词语
                        prefix_str = ""
                        if random.random() < 0.3:  # 30% 概率添加语气词
                            prefix_str += random.choice(yu_qi_ci_liebiao)
                        if random.random() < 0.7:  # 70% 概率添加转折/承接词
                            prefix_str += random.choice(zhuan_jie_ci_liebiao)

                        # 组合最终结果
                        if prefix_str:
                            content = f"{prefix_str}，{deduplicated_content}"  # 更新 content
                            logger.debug(f"{self.log_prefix} 去重并添加引导词后: {content}")
                        else:
                            content = deduplicated_content  # 更新 content
                            logger.debug(f"{self.log_prefix} 去重后 (未添加引导词): {content}")
                    else:
                        logger.warning(f"{self.log_prefix} 去重后内容为空，保留原始LLM输出: {new_content}")
                        content = new_content  # 保留原始 content
            else:
                logger.debug(f"{self.log_prefix} 未执行概率性去重 (概率: {replacement_prob:.2f})")
                # content 保持 new_content 不变

        except Exception as e:
            logger.error(f"{self.log_prefix} 应用概率性去重或特殊处理时出错: {e}")
            logger.error(traceback.format_exc())
            # 出错时保留原始 content
            content = new_content

        return content


init_prompt()
