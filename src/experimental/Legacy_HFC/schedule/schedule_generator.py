import datetime
import os
import sys
import asyncio
import json # <--- 新增导入
import re
from dateutil import tz

# 添加项目根目录到 Python 路径
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.common.database import db  # noqa: E402
from src.common.logger import get_module_logger, SCHEDULE_STYLE_CONFIG, LogConfig  # noqa: E402
from src.chat.models.utils_model import LLMRequest  # noqa: E402
from src.config.config import global_config  # noqa: E402
from src.individuality.individuality import Individuality
from src.chat.knowledge.knowledge_lib import qa_manager # 用于知识库检索
from src.chat.memory_system.Hippocampus import HippocampusManager # 用于记忆检索 (如果需要)

TIME_ZONE = tz.gettz(global_config.schedule.time_zone)  # 设置时区


schedule_config = LogConfig(
    # 使用海马体专用样式
    console_format=SCHEDULE_STYLE_CONFIG["console_format"],
    file_format=SCHEDULE_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("scheduler", config=schedule_config)


def parse_knowledge_and_get_max_relevance(knowledge_str: str) -> tuple[str | None, float]: # 返回类型修改
    """
    解析 qa_manager.get_knowledge 返回的字符串，提取所有知识的文本和最高的相关性得分。
    返回: (原始知识字符串, 最高相关性得分)，如果无有效相关性则返回 (原始知识字符串, 0.0)
    """
    if not knowledge_str:
        return None, 0.0 # 如果输入为空，返回 None 和 0.0

    max_relevance = 0.0
    # 正则表达式匹配 "该条知识对于问题的相关性：数字"
    # 我们需要捕获数字部分
    relevance_scores = re.findall(r"该条知识对于问题的相关性：([0-9.]+)", knowledge_str)

    if relevance_scores:
        try:
            max_relevance = max(float(score) for score in relevance_scores)
        except ValueError:
            logger.warning(f"解析相关性得分时出错: {relevance_scores}")
            return knowledge_str, 0.0  # 出错时返回0.0
    else:
        # 如果没有找到 "该条知识对于问题的相关性：" 这样的模式，
        # 说明可能 qa_manager 返回的格式有变，或者没有有效的知识。
        # 在这种情况下，我们无法确定相关性，保守起见返回0.0
        logger.debug(f"在知识字符串中未找到明确的相关性得分标记: '{knowledge_str[:100]}...'")
        # 即使没有相关性标记，也返回原始字符串，因为可能仍然有用
        return knowledge_str, 0.0

    return knowledge_str, max_relevance


class ScheduleGenerator:
    # enable_output: bool = True

    def __init__(self):
        # 使用离线LLM模型
        self.enable_output = None

        self.llm_schedule_initial_generator = LLMRequest(
            model=global_config.model.schedule_initial_generator, 
            temperature=global_config.model.schedule_initial_generator.get("temp", 0.7), 
            max_tokens=global_config.model.schedule_initial_generator.get("max_tokens", 7000),
            request_type="schedule_initial_generation", 
        )
        self.llm_keyword_extractor = LLMRequest(
            model=global_config.model.schedule_keyword_extractor, 
            temperature=global_config.model.schedule_keyword_extractor.get("temp", 0.3),
            max_tokens=global_config.model.schedule_keyword_extractor.get("max_tokens", 1000),
            request_type="schedule_keyword_extraction",
        )
        self.llm_schedule_refiner = LLMRequest(
            model=global_config.model.schedule_refiner, 
            temperature=global_config.model.schedule_refiner.get("temp", 0.7),
            max_tokens=global_config.model.schedule_refiner.get("max_tokens", 7000),
            request_type="schedule_refinement",
        )
        self.llm_schedule_current_activity = LLMRequest( 
            model=global_config.model.schedule_current_activity, 
            temperature=global_config.model.schedule_current_activity.get("temp", 0.5),
            max_tokens=global_config.model.schedule_current_activity.get("max_tokens", 2048),
            request_type="schedule_current_activity", 
        )

        self.today_schedule_text = ""
        self.today_done_list = []

        self.yesterday_schedule_text = ""
        self.yesterday_done_list = []

        self.name = ""
        self.behavior = ""
        self.individuality = Individuality.get_instance()

        self.start_time = datetime.datetime.now(TIME_ZONE)

        self.schedule_doing_update_interval = 300  # 最好大于60

    def initialize(
        self,
        name: str = "bot_name",
        personality: str = "你是一个爱国爱党的新时代青年", # 这个参数没用了
        behavior: str = "你非常外向，喜欢尝试新事物和人交流",
        interval: int = 60,
    ):
        """初始化日程系统"""
        self.name = name
        self.behavior = behavior
        self.schedule_doing_update_interval = interval
        # self.personality = personality

    async def mai_schedule_start(self):
        """启动日程系统，每5分钟执行一次move_doing，并在日期变化时重新检查日程"""
        try:
            if global_config.schedule.enable:
                logger.info(f"日程系统启动/刷新时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                # 初始化日程
                await self.check_and_create_today_schedule()
                # self.print_schedule()

                while True:
                    # print(self.get_current_num_task(1, True))

                    current_time = datetime.datetime.now(TIME_ZONE)

                    # 检查是否需要重新生成日程（日期变化）
                    if current_time.date() != self.start_time.date():
                        logger.info("检测到日期变化，重新生成日程")
                        self.start_time = current_time
                        await self.check_and_create_today_schedule()
                        # self.print_schedule()

                    # 执行当前活动
                    # mind_thinking = heartflow.current_state.current_mind

                    await self.move_doing()

                    await asyncio.sleep(self.schedule_doing_update_interval)
            else:
                logger.info("日程系统未启用")

        except Exception as e:
            logger.error(f"日程系统运行时出错: {str(e)}")
            logger.exception("详细错误信息：")

    async def check_and_create_today_schedule(self):
        """检查昨天的日程，并确保今天有日程安排

        Returns:
            tuple: (today_schedule_text, today_schedule) 今天的日程文本和解析后的日程字典
        """
        today = datetime.datetime.now(TIME_ZONE)
        yesterday = today - datetime.timedelta(days=1)

        # 先检查昨天的日程
        self.yesterday_schedule_text, self.yesterday_done_list = self.load_schedule_from_db(yesterday)
        if self.yesterday_schedule_text:
            logger.debug(f"已加载{yesterday.strftime('%Y-%m-%d')}的日程")

        # 检查今天的日程
        # today_schedule_text_from_db, self.today_done_list = self.load_schedule_from_db(today) # 先不直接赋值给 self.today_schedule_text
        # 修改：分别加载日程文本和已完成列表
        raw_today_schedule_text_from_db, loaded_today_done_list = self.load_schedule_from_db(today)
        self.today_done_list = loaded_today_done_list if loaded_today_done_list is not None else []


        if not raw_today_schedule_text_from_db: # 如果数据库中没有今天的日程
            logger.info(f"{today.strftime('%Y-%m-%d')}的日程不存在，准备生成新的日程")
            try:
                # === 步骤 1: 生成初版日程表 ===
                initial_schedule_text = await self.generate_daily_schedule_initial(target_date=today)
                if not initial_schedule_text:
                    logger.error("初版日程生成失败，无法继续。")
                    self.today_schedule_text = "" # 确保置空
                    self.save_today_schedule_to_db() # 保存空日程或标记错误
                    return

                logger.info(f"初版日程已生成，长度: {len(initial_schedule_text)}")
                logger.debug(f"初版日程内容预览:\n{initial_schedule_text[:500]}...")

                # === 步骤 2: 提取关键词并检索知识库与记忆 ===
                keywords = await self.extract_keywords_from_schedule(initial_schedule_text)
                if not keywords:
                    logger.warning("未能从初版日程中提取到关键词。将直接使用初版日程。")
                    self.today_schedule_text = initial_schedule_text
                else:
                    logger.info(f"从日程中提取到关键词: {keywords}")
                    retrieved_knowledge_and_memory = await self.retrieve_knowledge_for_keywords(keywords)

                    if not retrieved_knowledge_and_memory:
                        logger.info("未检索到相关的知识库或记忆内容。将直接使用初版日程。")
                        self.today_schedule_text = initial_schedule_text
                    else:
                        logger.info("检索到相关的知识库或记忆内容。")
                        logger.debug(f"检索到的内容:\n{retrieved_knowledge_and_memory[:1000]}...") # 打印部分内容

                        # === 步骤 3: 结合知识库和记忆，要求LLM修改日程 ===
                        refined_schedule_text = await self.refine_schedule_with_knowledge(
                            initial_schedule_text,
                            retrieved_knowledge_and_memory
                        )
                        if not refined_schedule_text:
                            logger.error("结合知识库修改日程失败，将使用初版日程。")
                            self.today_schedule_text = initial_schedule_text
                        else:
                            logger.info("日程已结合知识库和记忆进行修改。")
                            logger.debug(f"修改后的日程内容预览:\n{refined_schedule_text[:500]}...")
                            self.today_schedule_text = refined_schedule_text

            except Exception as e:
                logger.error(f"生成或优化日程时发生错误: {str(e)}")
                logger.exception("详细错误信息:")
                self.today_schedule_text = "" # 出错时确保为空
        else: # 如果数据库中已有今天的日程 (可能是之前运行生成并保存的)
            logger.info(f"已从数据库加载{today.strftime('%Y-%m-%d')}的日程。")
            self.today_schedule_text = raw_today_schedule_text_from_db
            # self.today_done_list 已在前面加载


        # === 步骤 4: 保存最终日程表 ===
        self.save_today_schedule_to_db()
        if self.today_schedule_text:
            logger.info(f"最终使用的日程表内容预览:\n{self.today_schedule_text[:500]}...")
        else:
            logger.warning("今日日程最终为空。")

    def construct_daytime_prompt(self, target_date: datetime.datetime):
        date_str = target_date.strftime("%Y-%m-%d")
        weekday = target_date.strftime("%A")
        prompt_personality_description = self.individuality.get_prompt(x_person=0, level=3)
        bot_name_to_use = self.name

        prompt = f"你是{bot_name_to_use}。\n{prompt_personality_description}\n你的行为习惯大概是：{self.behavior}\n"
        prompt += f"你昨天的日程是：{self.yesterday_schedule_text}\n"
        prompt += f"请为你生成{date_str}（{weekday}），也就是今天的日程安排，结合你的个人特点和行为习惯以及昨天的安排\n"
        prompt += "推测你的日程安排，包括你一天都在做什么，从起床到睡眠，有什么发现和思考，具体一些，详细一些，需要1500字以上，精确到每半个小时，记得写明时间\n"  # noqa: E501
        prompt += "直接返回你的日程，现实一点，不要浮夸，从起床到睡觉，不要输出其他内容："
        return prompt
    

    def construct_doing_prompt(self, time: datetime.datetime, mind_thinking: str = ""):
        now_time = time.strftime("%H:%M")
        previous_doings = self.get_current_num_task(5, True)
        prompt_personality_description = self.individuality.get_prompt(x_person=0, level=3)
        bot_name_to_use = self.name

        prompt = f"你是{bot_name_to_use}。\n{prompt_personality_description}\n你的行为习惯大概是：{self.behavior}\n"
        prompt += f"你今天的日程是：{self.today_schedule_text}\n"
        if previous_doings:
            prompt += f"你之前做了的事情是：{previous_doings}，从之前到现在已经过去了{self.schedule_doing_update_interval / 60}分钟了\n"  # noqa: E501
        if mind_thinking:
            prompt += f"你脑子里在想：{mind_thinking}\n"
        prompt += f"现在是{now_time}，结合你的个人特点和行为习惯,注意关注你今天的日程安排和想法安排你接下来做什么，现实一点，不要浮夸"
        prompt += "安排你接下来做什么，具体一些，详细一些\n"
        prompt += "直接返回你在做的事情，注意是当前时间，不要输出其他内容："
        return prompt
    
    def construct_keyword_extraction_prompt(self, schedule_text: str) -> str:
        """构建用于从日程表中提取关键词的提示词"""
        prompt = (
            f"这是一段机器人的一天日程安排文本：\n"
            f"```text\n{schedule_text}\n```\n"
            f"请从上述日程安排文本中提取出所有的名词（包括人名、地名、事件名、物品名等专有名词和普通名词）。"
            f"你需要将提取到的名词以JSON列表的格式返回，例如：[\"名词1\", \"名词2\", \"名词3\"]。\n"
            f"请确保只返回JSON格式的列表，不要包含其他任何文字说明或解释。"
        )
        return prompt
    
    def construct_schedule_refinement_prompt(self, initial_schedule: str, knowledge_and_memory: str) -> str:
        """构建用于结合知识库和记忆修改日程的提示词"""
        prompt_personality_description = self.individuality.get_prompt(x_person=0, level=3)
        bot_name_to_use = self.name

        prompt = (
            f"你是{bot_name_to_use}。\n"
            f"{prompt_personality_description}\n"
            f"你的行为习惯大概是：{self.behavior}\n\n"
            f"这是你今天的一份初步日程安排（由一个没那么好的llm生成）：\n"
            f"```text\n{initial_schedule}\n```\n\n"
            f"为了让这份日程更贴合你的角色设定和背景故事，这里有一些相关的知识库和记忆信息：\n"
            f"```text\n{knowledge_and_memory}\n```\n\n"
            f"请仔细阅读初步日程以及相关的知识库和记忆信息。\n"
            f"你的任务是：根据这些知识库和记忆信息，修改并完善上述初步日程安排，使其更符合你的角色特点、背景故事和已知信息。\n"
            f"修改时请注意：\n"
            f"1. 保持日程的整体结构和时间线索基本不变，除非知识库/记忆明确指示了需要调整时间或顺序的事件。\n"
            f"2. 可以将知识库和记忆中的相关情节、人物、地点、特定行为或思考方式自然地融入到日程的描述中。\n"
            f"3. 如果初步日程中的某些活动与知识库/记忆或你的人设有冲突，请以知识库/记忆和你的人设为准进行修正。\n"
            f"4. 如果知识库/记忆中包含初步日程未提及但对你角色非常重要的日常活动或思考，请适当补充。\n"
            f"5. 修改后的日程应该仍然是一份详细的、精确到每半小时左右的日程安排，保持与初步日程相似的格式和详细程度。\n"
            f"6. 输出的语言风格应与你 ({bot_name_to_use}) 的角色个性一致。\n"
            f"7. 请直接返回修改后的完整日程安排文本，不要包含其他任何文字说明或解释。\n"
            f"8. 如果你认为有不合理，或夸张的地方，也可以改动，使其更加合理。\n"
            f"9. 如果你认为初步日程已经很好地结合了所提供的知识库和记忆，或者知识库和记忆与日程内容关联不大，可以只做微小调整或不作调整。但在这种情况下，也请尽量确保日程中没有与知识库/记忆明显矛盾的地方。\n\n"
            f"修改后的日程安排："
        )
        return prompt

    async def refine_schedule_with_knowledge(self, initial_schedule: str, knowledge_and_memory: str) -> str:
        """
        使用LLM结合知识库和记忆修改日程表。
        """
        if not initial_schedule:
            logger.error("初版日程为空，无法进行优化。")
            return ""
        if not knowledge_and_memory: # 如果没有知识，直接返回原日程
            logger.info("没有可供参考的知识库或记忆，直接返回初版日程。")
            return initial_schedule

        prompt = self.construct_schedule_refinement_prompt(initial_schedule, knowledge_and_memory)
        logger.debug(f"用于优化日程的Prompt:\n{prompt}")
        try:
            refined_schedule, _ = await self.llm_schedule_refiner.generate_response_async(prompt)
            if not refined_schedule:
                logger.warning("日程优化LLM未返回任何内容。")
                return initial_schedule # 出错或无返回时，返回原日程
            return refined_schedule
        except Exception as e:
            logger.error(f"调用LLM优化日程时出错: {e}")
            logger.exception("详细错误信息:")
            return initial_schedule # 出错时，返回原日程
    

    async def extract_keywords_from_schedule(self, schedule_text: str) -> list[str]:
        """
        使用LLM从日程表中提取关键词（名词）。
        """
        if not schedule_text:
            return []

        prompt = self.construct_keyword_extraction_prompt(schedule_text)
        logger.debug(f"用于提取关键词的Prompt:\n{prompt}")
        try:
            response_text, _ = await self.llm_keyword_extractor.generate_response_async(prompt)
            if not response_text:
                logger.warning("关键词提取LLM未返回任何内容。")
                return []

            logger.debug(f"关键词提取LLM原始返回:\n{response_text}")

            # 尝试解析JSON
            try:
                # 尝试去除可能的 markdown 代码块标记
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                elif cleaned_response.startswith("```"):
                     cleaned_response = cleaned_response[3:]
                     if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]

                cleaned_response = cleaned_response.strip()

                keywords = json.loads(cleaned_response)
                if isinstance(keywords, list) and all(isinstance(kw, str) for kw in keywords):
                    return list(set(kw.strip() for kw in keywords if kw.strip())) # 去重并移除空字符串
                else:
                    logger.warning(f"关键词提取LLM返回的不是有效的JSON字符串列表: {response_text}")
                    return []
            except json.JSONDecodeError:
                logger.warning(f"无法解析关键词提取LLM返回的JSON: {response_text}")
                # 可以尝试用其他方式从文本中提取，例如简单的逗号分隔，但LLM应该能返回JSON
                # 作为后备，可以尝试从文本中提取所有看起来像名词的词，但这会复杂很多
                return []
        except Exception as e:
            logger.error(f"调用LLM提取日程关键词时出错: {e}")
            logger.exception("详细错误信息:")
            return []
        

    async def retrieve_knowledge_for_keywords(self, keywords: list[str]) -> str:
        """
        根据关键词列表检索相关的知识库和记忆。
        参考 sub_mind.py 中的 get_prompt_info 逻辑。
        """
        if not keywords:
            return ""

        all_retrieved_texts = []
        # 知识库检索阈值，参考 sub_mind.py 中的 lpmm 相关逻辑或 get_prompt_info_old
        # qa_manager.get_knowledge 内部似乎已经有相关性判断逻辑
        # 我们主要关注收集返回的知识文本

        # --- 知识库检索 ---
        # 我们对每个关键词独立检索，然后汇总。或者可以将关键词组合成一句话再检索，取决于效果。
        # 为了简单起见，这里对每个关键词独立检索。
        # 注意：qa_manager.get_knowledge 的输入是 "message" (即上下文文本)，而不是单个关键词。
        # 所以，更好的做法可能是将日程原文和提取的关键词结合起来作为检索的上下文，
        # 或者，如果 qa_manager 支持基于关键词的检索，则使用该方式。
        # 假设我们现在依然用 "message" 的方式，可以将关键词嵌入到一个问句中，或者直接使用关键词本身。

        # 简单实现：对每个关键词进行检索
        retrieved_knowledge_contents = set() # 用于去重

        for keyword in keywords:
            if not keyword.strip(): # 跳过空关键词
                continue
            try:
                # logger.debug(f"为关键词 '{keyword}' 检索知识库...")
                # qa_manager.get_knowledge 需要一个 "message" 作为输入。
                # 我们可以简单地用关键词本身，或者构造一个包含关键词的简单问句。
                # 这里我们尝试直接用关键词，如果效果不好，可以调整。
                # 或者，更好的做法是，如果日程的主要内容与特定关键词相关，
                # 那么在检索该关键词时，也把日程的相应段落作为上下文传入。
                # 为简化，先直接用关键词。
                # found_knowledge = qa_manager.get_knowledge(keyword) # qa_manager.get_knowledge 不是异步的
                loop = asyncio.get_event_loop()
                found_knowledge = await loop.run_in_executor(None, qa_manager.get_knowledge, keyword)


                if found_knowledge:
                    # qa_manager.get_knowledge 返回的已经是处理过的字符串，包含相关性和内容
                    # 我们需要从中提取纯文本内容，并可能根据相关性过滤
                    # sub_mind.py 中的 parse_knowledge_and_get_max_relevance 是一个好的参考
                    knowledge_content, relevance = parse_knowledge_and_get_max_relevance(found_knowledge)
                    # 你可以设定一个阈值，例如：
                    knowledge_relevance_threshold = global_config.schedule.knowledge_relevance_threshold # 假设配置中新增日程用阈值
                    if relevance >= knowledge_relevance_threshold:
                        logger.info(f"关键词 '{keyword}' 检索到相关知识 (相关性: {relevance:.4f})")
                        retrieved_knowledge_contents.add(knowledge_content.strip())
                    else:
                        logger.debug(f"关键词 '{keyword}' 检索到的知识相关性 ({relevance:.4f}) 低于阈值 {knowledge_relevance_threshold}，已忽略。")
            except Exception as e:
                logger.error(f"为关键词 '{keyword}' 检索知识库时出错: {e}")
                logger.exception("详细错误信息:")

        if retrieved_knowledge_contents:
            all_retrieved_texts.append("--- 相关知识库信息 ---")
            for content in retrieved_knowledge_contents:
                all_retrieved_texts.append(content)
            all_retrieved_texts.append("--- 结束知识库信息 ---")


        # --- 记忆检索 (可选，如果需要) ---
        # 你可以参照 sub_mind.py 中 HippocampusManager 的使用方式来添加记忆检索
        # 例如:
        # hippocampus_instance = HippocampusManager.get_instance()
        # retrieved_memories_contents = set()
        # for keyword in keywords:
        #     if not keyword.strip():
        #         continue
        #     try:
        #         # 假设 get_memory_from_text 或类似方法，输入关键词或包含关键词的文本
        #         # related_memories = await hippocampus_instance.get_memory_from_text(keyword, max_memory_num=1, fast_retrieval=True) # 根据需要调整参数
        #         related_memories = await hippocampus_instance.get_memory_from_topic(valid_keywords=[keyword], max_memory_num=1)
        #
        #         if related_memories:
        #             for topic_or_id, memory_text in related_memories: # Hippocampus 返回的是元组列表
        #                 logger.info(f"关键词 '{keyword}' 检索到相关记忆: {topic_or_id}")
        #                 retrieved_memories_contents.add(memory_text.strip())
        #     except Exception as e:
        #         logger.error(f"为关键词 '{keyword}' 检索记忆时出错: {e}")
        #         logger.exception("详细错误信息:")
        #
        # if retrieved_memories_contents:
        #     all_retrieved_texts.append("\n--- 相关记忆信息 ---")
        #     for content in retrieved_memories_contents:
        #         all_retrieved_texts.append(content)
        #     all_retrieved_texts.append("--- 结束记忆信息 ---")

        return "\n\n".join(all_retrieved_texts)
    

    async def generate_daily_schedule_initial(
        self,
        target_date: datetime.datetime = None,
    ) -> str: # 返回类型改为 str
        """生成每日日程的初始版本"""
        daytime_prompt = self.construct_daytime_prompt(target_date)
        logger.debug(f"用于生成初版日程的Prompt:\n{daytime_prompt}")
        try:
            # --- 修改：使用新的LLM实例 ---
            daytime_response, _ = await self.llm_schedule_initial_generator.generate_response_async(daytime_prompt)
            if not daytime_response:
                logger.error("LLM未能生成初版日程内容。")
                return ""
            return daytime_response
        except Exception as e:
            logger.error(f"调用LLM生成初版日程时出错: {e}")
            logger.exception("详细错误信息:")
            return ""

    def print_schedule(self):
        """打印完整的日程安排"""
        if not self.today_schedule_text:
            logger.warning("今日日程有误，将在下次运行时重新生成")
            db.schedule.delete_one({"date": datetime.datetime.now(TIME_ZONE).strftime("%Y-%m-%d")})
        else:
            logger.info("=== 今日日程安排 ===")
            logger.info(self.today_schedule_text)
            logger.info("==================")
            self.enable_output = False

    async def update_today_done_list(self):
        # 更新数据库中的 today_done_list
        today_str = datetime.datetime.now(TIME_ZONE).strftime("%Y-%m-%d")
        existing_schedule = db.schedule.find_one({"date": today_str})

        if existing_schedule:
            # 更新数据库中的 today_done_list
            db.schedule.update_one({"date": today_str}, {"$set": {"today_done_list": self.today_done_list}})
            logger.debug(f"已更新{today_str}的已完成活动列表")
        else:
            logger.warning(f"未找到{today_str}的日程记录")

    async def move_doing(self, mind_thinking: str = ""):
        try:
            current_time = datetime.datetime.now(TIME_ZONE)
            if mind_thinking:
                doing_prompt = self.construct_doing_prompt(current_time, mind_thinking)
            else:
                doing_prompt = self.construct_doing_prompt(current_time)

            doing_response, _ = await self.llm_schedule_current_activity.generate_response_async(doing_prompt)
            self.today_done_list.append((current_time, doing_response))

            await self.update_today_done_list()

            logger.info(f"当前活动: {doing_response}")

            return doing_response
        except GeneratorExit:
            logger.warning("日程生成被中断")
            return "日程生成被中断"
        except Exception as e:
            logger.error(f"生成日程时发生错误: {str(e)}")
            return "生成日程时发生错误"

    async def get_task_from_time_to_time(self, start_time: str, end_time: str):
        """获取指定时间范围内的任务列表

        Args:
            start_time (str): 开始时间，格式为"HH:MM"
            end_time (str): 结束时间，格式为"HH:MM"

        Returns:
            list: 时间范围内的任务列表
        """
        result = []
        for task in self.today_done_list:
            task_time = task[0]  # 获取任务的时间戳
            task_time_str = task_time.strftime("%H:%M")

            # 检查任务时间是否在指定范围内
            if self._time_diff(start_time, task_time_str) >= 0 and self._time_diff(task_time_str, end_time) >= 0:
                result.append(task)

        return result

    def get_current_num_task(self, num=1, time_info=False):
        """获取最新加入的指定数量的日程

        Args:
            num (int): 需要获取的日程数量，默认为1
            time_info (bool): 是否包含时间信息，默认为False

        Returns:
            list: 最新加入的日程列表
        """
        if not self.today_done_list:
            return []

        # 确保num不超过列表长度
        num = min(num, len(self.today_done_list))
        pre_doings = ""
        for doing in self.today_done_list[-num:]:
            if time_info:
                time_str = doing[0].strftime("%H:%M")
                pre_doings += time_str + "时，" + doing[1] + "\n"
            else:
                pre_doings += doing[1] + "\n"

        # 返回最新的num条日程
        return pre_doings

    def save_today_schedule_to_db(self):
        """保存日程到数据库，同时初始化 today_done_list"""
        date_str = datetime.datetime.now(TIME_ZONE).strftime("%Y-%m-%d")
        schedule_data = {
            "date": date_str,
            "schedule": self.today_schedule_text,
            "today_done_list": self.today_done_list if hasattr(self, "today_done_list") else [],
        }
        # 使用 upsert 操作，如果存在则更新，不存在则插入
        db.schedule.update_one({"date": date_str}, {"$set": schedule_data}, upsert=True)
        logger.debug(f"已保存{date_str}的日程到数据库")

    @staticmethod
    def load_schedule_from_db(date: datetime.datetime):
        """从数据库加载日程，同时加载 today_done_list"""
        date_str = date.strftime("%Y-%m-%d")
        existing_schedule = db.schedule.find_one({"date": date_str})

        if existing_schedule:
            schedule_text = existing_schedule["schedule"]
            return schedule_text, existing_schedule.get("today_done_list", [])
        else:
            logger.debug(f"{date_str}的日程不存在")
            return None, None


async def main():
    # 使用示例
    scheduler = ScheduleGenerator()
    scheduler.initialize(
        name="{global_config.bot.nickname}",
        personality="你叫{global_config.bot.nickname}，你19岁，是一个大二的女大学生，你有一头黑色短发，你会刷贴吧，你现在在学习心理学",
        behavior="你比较内向，一般熬夜比较晚，然后第二天早上10点起床吃早午饭",
        interval=60,
    )
    await scheduler.mai_schedule_start()


# 当作为组件导入时使用的实例
bot_schedule = ScheduleGenerator()

if __name__ == "__main__":
    import asyncio

    # 当直接运行此文件时执行
    asyncio.run(main())
