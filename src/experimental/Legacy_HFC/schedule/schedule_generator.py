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

# 规范化函数
def more_robust_normalize(text: str) -> str:
    if not text:
        return ""
    # 尝试移除 "第X条知识：" 这样的前缀，如果存在的话
    text = re.sub(r"^\s*第\d+条知识：\s*", "", text).strip()
    # 移除大部分非中文、非英文、非数字的特殊符号，但保留中文常用标点
    # 你可以根据你的知识内容特点调整这个正则表达式
    text = re.sub(r"[^\w\s\u4e00-\u9fa5，。！？；：]", "", text, flags=re.UNICODE)
    # 统一小写（主要针对可能存在的英文）
    text = text.lower()
    # 移除多余空格
    return " ".join(text.strip().split())

def extract_individual_knowledge_pieces(knowledge_blob: str, source_keyword_for_log: str = "未知") -> list[tuple[str, float]]:
    """
    从 qa_manager 返回的包含多条知识的字符串中提取独立的知识片段及其相关性。
    返回一个列表，每个元素是 (独立知识文本, 该知识的相关性)。
    """
    pieces = []
    if not knowledge_blob or not knowledge_blob.strip():
        return pieces

    # 主要模式：匹配 "第X条知识：" 开头，直到 "该条知识对于问题的相关性："
    # (?s) 或 re.DOTALL 使 . 匹配换行符
    # 使用非贪婪匹配 .*?
    pattern = re.compile(
        r"(?:第\d+条知识：\s*)?(?P<content>.+?)\s*该条知识对于问题的相关性：\s*(?P<relevance>[0-9.]+)",
        re.DOTALL
    )

    # 为了处理一个大块文本可能被错误地解析为多个小块，或者一个小块被遗漏的情况，
    # 我们改为使用 finditer 来查找所有匹配项。
    matches = list(pattern.finditer(knowledge_blob))

    if matches:
        for match in matches:
            content = match.group("content").strip()
            # 再次去除可能仍存在的 "第X条知识：" (如果正则表达式的第一个可选组匹配失败但内容里还有)
            content = re.sub(r"^第\d+条知识：\s*", "", content).strip()
            try:
                relevance_val = float(match.group("relevance"))
                if content: # 确保内容不为空
                    pieces.append((content, relevance_val))
                else:
                    logger.debug(f"提取到空内容知识片段，来源关键词: {source_keyword_for_log}，相关性: {relevance_val}，已忽略。")
            except ValueError:
                logger.warning(f"无法解析相关性值: '{match.group('relevance')}'，来源关键词: {source_keyword_for_log}。该片段被忽略。")
    elif knowledge_blob.strip(): # 如果正则没有匹配到任何东西，但原始字符串不为空
        # 尝试将整个 blob 作为一个条目，并尝试用 parse_knowledge_and_get_max_relevance 获取其总体相关性
        # 注意：parse_knowledge_and_get_max_relevance 本身也会返回整个 blob 作为 content
        # 所以我们直接使用原始 blob 和提取出的相关性
        _original_blob_content, overall_relevance = parse_knowledge_and_get_max_relevance(knowledge_blob)
        if _original_blob_content.strip(): # 确保解析后的内容不为空
             # 这里 _original_blob_content 理论上就是 knowledge_blob，但我们用解析函数返回的确保一致性
            logger.debug(f"未能按 '第X条' 格式拆分知识块 (来源: {source_keyword_for_log})，将其视为单个条目。相关性: {overall_relevance:.4f}")
            pieces.append((_original_blob_content.strip(), overall_relevance if overall_relevance > 0 else 0.01)) # 给一个低默认值
        else:
            logger.warning(f"知识块 (来源: {source_keyword_for_log}) 既不能按 '第X条' 拆分，整体解析后内容也为空，已忽略。原始块: {knowledge_blob[:200]}...")


    # 日志记录，有多少 piece 被成功提取
    if pieces:
        logger.debug(f"从关键词 '{source_keyword_for_log}' 的知识块中成功提取 {len(pieces)} 个独立知识片段。")
    elif knowledge_blob.strip(): # 有原始输入但没提取出任何东西
        logger.warning(f"未能从关键词 '{source_keyword_for_log}' 的知识块中提取任何有效片段。原始块: {knowledge_blob[:200]}...")
        
    return pieces


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
        加强去重逻辑，针对拆分后的独立知识条目进行去重。
        """
        if not keywords:
            return ""

        # 用于存储所有不重复的 (独立知识文本, 相关性, 来源关键词等元数据)
        # 我们用一个字典来存储，键是规范化后的文本签名，值是包含原始内容和最高相关性的字典
        # 这样如果后续有相同签名但相关性更高的，可以更新
        unique_knowledge_map = {} # str_signature -> {"content": original_piece_content, "relevance": highest_relevance_for_this_content, "source_keywords": set_of_keywords}


        for keyword in keywords:
            if not keyword.strip():
                continue
            try:
                loop = asyncio.get_event_loop()
                # found_knowledge_str 是从 qa_manager 获取的原始字符串块
                found_knowledge_str = await loop.run_in_executor(None, qa_manager.get_knowledge, keyword)

                if found_knowledge_str and found_knowledge_str.strip():
                    # 1. 将原始字符串块拆分为独立的知识片段
                    # 将当前关键词传递给拆分函数，用于更详细的日志记录
                    individual_pieces = extract_individual_knowledge_pieces(found_knowledge_str, source_keyword_for_log=keyword)
                    
                    if not individual_pieces:
                        logger.debug(f"关键词 '{keyword}' 未能拆分出有效知识片段。")
                        continue # 如果拆分不出任何有效片段，处理下一个关键词

                    for piece_content, piece_relevance in individual_pieces:
                        if not piece_content.strip(): # 跳过内容为空的片段
                            continue

                        knowledge_relevance_threshold = global_config.schedule.get("knowledge_relevance_threshold", 0.35) # 从配置读取，提供默认值
                        
                        if piece_relevance >= knowledge_relevance_threshold:
                            # 2. 对每个独立知识片段的文本内容进行规范化
                            normalized_piece_signature = more_robust_normalize(piece_content)

                            if not normalized_piece_signature: # 如果规范化后为空，也跳过
                                logger.debug(f"独立知识片段规范化后为空，原始片段: '{piece_content[:50]}...'，已忽略。")
                                continue

                            # 3. 检查是否已经添加过，或者是否需要更新（如果新的相关性更高）
                            if normalized_piece_signature not in unique_knowledge_map or \
                               piece_relevance > unique_knowledge_map[normalized_piece_signature]["relevance"]:
                                
                                if normalized_piece_signature not in unique_knowledge_map:
                                    logger.info(f"新增独立知识 (来自关键词 '{keyword}', 相关性: {piece_relevance:.4f})。内容: {piece_content[:50]}...")
                                    unique_knowledge_map[normalized_piece_signature] = {
                                        "content": piece_content.strip(),
                                        "relevance": piece_relevance,
                                        "source_keywords": {keyword} # 初始化来源关键词集合
                                    }
                                else: # normalized_piece_signature 已存在，但当前 piece_relevance 更高
                                    logger.info(f"更新独立知识 (来自关键词 '{keyword}', 新相关性: {piece_relevance:.4f} > 旧相关性: {unique_knowledge_map[normalized_piece_signature]['relevance']:.4f})。内容: {piece_content[:50]}...")
                                    unique_knowledge_map[normalized_piece_signature]["relevance"] = piece_relevance
                                    # 可以选择是否更新内容，如果认为高相关性的版本更好
                                    # unique_knowledge_map[normalized_piece_signature]["content"] = piece_content.strip() 
                                    unique_knowledge_map[normalized_piece_signature]["source_keywords"].add(keyword)
                            
                            # 如果签名已存在且当前相关性不高，可以记录一下来源（可选）
                            elif normalized_piece_signature in unique_knowledge_map:
                                unique_knowledge_map[normalized_piece_signature]["source_keywords"].add(keyword)
                                logger.debug(f"独立知识片段内容已存在 (来自 '{keyword}', 相关性 {piece_relevance:.4f})，未更新。")
                        else:
                            logger.debug(f"独立知识片段 (来自 '{keyword}') 相关性 ({piece_relevance:.4f}) 低于阈值 {knowledge_relevance_threshold}，已忽略。内容: {piece_content[:50]}...")
                else:
                    logger.debug(f"关键词 '{keyword}' 未检索到任何知识内容或内容为空。")
            except Exception as e:
                logger.error(f"为关键词 '{keyword}' 检索知识库或处理时出错: {e}")
                logger.exception("详细错误信息:")

        # --- 从 unique_knowledge_map 中提取最终的知识项列表 ---
        # unique_knowledge_map 的值是 {"content": ..., "relevance": ..., "source_keywords": ...}
        final_knowledge_items_list = list(unique_knowledge_map.values())

        # --- 组装最终的知识文本 ---
        all_retrieved_texts = []
        if final_knowledge_items_list:
            # 按相关性排序所有收集到的独立知识片段
            final_knowledge_items_list.sort(key=lambda x: x["relevance"], reverse=True)
            
            MAX_ITEMS_FOR_PROMPT = global_config.schedule.get("max_knowledge_items_for_schedule_prompt", 5)
            logger.info(f"总共收集到 {len(final_knowledge_items_list)} 条去重后的独立知识，将选取最多 {MAX_ITEMS_FOR_PROMPT} 条。")

            all_retrieved_texts.append("--- 相关知识库信息 ---")
            count = 0
            for item in final_knowledge_items_list:
                if count >= MAX_ITEMS_FOR_PROMPT:
                    logger.debug(f"已达到知识条目上限 {MAX_ITEMS_FOR_PROMPT}，停止添加。")
                    break
                
                # 打印来源关键词信息，方便调试
                source_kw_str = ", ".join(list(item['source_keywords'])[:3]) # 最多显示3个来源关键词
                if len(item['source_keywords']) > 3:
                    source_kw_str += f" 等{len(item['source_keywords'])}个"

                logger.debug(f"最终选用知识 (相关性: {item['relevance']:.4f}, 来源: [{source_kw_str}]): {item['content'][:100]}...")
                all_retrieved_texts.append(item['content']) # 只添加内容到最终prompt
                count += 1
            all_retrieved_texts.append("--- 结束知识库信息 ---")
        else:
            logger.info("经过所有关键词检索和去重后，没有最终采纳的知识片段。")


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
