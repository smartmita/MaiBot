import datetime
import os
import sys
import asyncio
import json
from dateutil import tz
from typing import List, Dict, Any, Optional # 新增类型提示

# 添加项目根目录到 Python 路径
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(root_path)

from src.common.database import db  # noqa: E402
from src.common.logger_manager import get_module_logger
from src.common.logger import get_module_logger, SCHEDULE_STYLE_CONFIG, LogConfig  # noqa: E402
from src.chat.models.utils_model import LLMRequest  # noqa: E402
from src.config.config import global_config  # noqa: E402
from src.individuality.individuality import Individuality

TIME_ZONE = tz.gettz(global_config.schedule.time_zone)  # 设置时区


schedule_config = LogConfig(
    # 使用海马体专用样式
    console_format=SCHEDULE_STYLE_CONFIG["console_format"],
    file_format=SCHEDULE_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("scheduler", config=schedule_config)


class ScheduleGenerator:
    # enable_output: bool = True

    def __init__(self):
        # 使用离线LLM模型
        self.enable_output = None
        self.llm_scheduler_all = LLMRequest(
            model=global_config.model.scheduler_all,
            temperature=global_config.model.scheduler_all["temp"],
            max_tokens=3000,
            request_type="schedule_outline",
        )
        self.llm_scheduler_doing = LLMRequest(
            model=global_config.model.scheduler_doing,
            temperature=global_config.model.scheduler_doing["temp"],
            max_tokens=2048,
            request_type="schedule_activity",
        )

        # self.today_schedule_text = "" # <--- 考虑移除或改变其用途
        self.today_outline_plan: List[Dict[str, str]] = []
        self.today_done_list: List[tuple[datetime.datetime, str]] = []

        self.yesterday_outline_plan: List[Dict[str, str]] = [] # <--- 新增：存储昨天的概要计划
        self.yesterday_done_list: List[tuple[datetime.datetime, str]] = []

        self.name = ""
        self.behavior = ""
        self.individuality = Individuality.get_instance()
        self.start_time = datetime.datetime.now(TIME_ZONE)
        self.schedule_doing_update_interval = 300

    def initialize(
        self,
        name: str = "bot_name",
        behavior: str = "你非常外向，喜欢尝试新事物和人交流",
        interval: int = 60,
    ):
        """初始化日程系统"""
        self.name = name
        self.behavior = behavior
        self.schedule_doing_update_interval = interval

    async def check_and_create_today_schedule(self):
        """检查昨天的日程，并确保今天有概要日程安排。"""
        today = datetime.datetime.now(TIME_ZONE)
        yesterday = today - datetime.timedelta(days=1)

        # 1. 加载昨天的日程信息
        self.yesterday_outline_plan, self.yesterday_done_list = self._load_schedule_from_db(yesterday)
        if self.yesterday_outline_plan:
            logger.debug(f"已加载 {yesterday.strftime('%Y-%m-%d')} 的概要日程和已完成活动。")
        else:
            self.yesterday_outline_plan = [] # 确保有默认值
            logger.debug(f"{yesterday.strftime('%Y-%m-%d')} 的概要日程不存在。")
        if not self.yesterday_done_list: # 即使概要计划不存在，也确保done_list有默认值
            self.yesterday_done_list = []


        # 2. 检查并生成今天的概要日程
        loaded_today_outline, loaded_today_done = self._load_schedule_from_db(today)
        self.today_done_list = loaded_today_done if loaded_today_done is not None else []

        if loaded_today_outline:
            self.today_outline_plan = loaded_today_outline
            logger.info(f"已加载 {today.strftime('%Y-%m-%d')} 的概要日程。")
        else:
            logger.info(f"{today.strftime('%Y-%m-%d')} 的概要日程不存在，准备生成新的概要日程。")
            await self._generate_daily_outline(target_date=today)
            # _generate_daily_outline 内部会保存到数据库和 self.today_outline_plan

        # 如果今天没有任何已完成的活动，并且概要计划存在，可以考虑初始化第一个 "当前活动"
        # (这部分逻辑可以根据需求调整，或者放在 mai_schedule_start 的首次 move_doing 调用中)
        if not self.today_done_list and self.today_outline_plan:
             logger.debug(f"今天尚无已完成活动，将在首次 move_doing 时决定当前活动。")


    def _format_outline_for_prompt(self, outline_plan: List[Dict[str, str]]) -> str:
        """将结构化的概要计划格式化为易读的文本，用于Prompt。"""
        if not outline_plan:
            return "无特定概要计划。"
        formatted_lines = []
        for item in outline_plan:
            time_hour = item.get("time_hour", "未知时间")
            activity_summary = item.get("activity_summary", "未定")
            formatted_lines.append(f"- {time_hour}: {activity_summary}")
        return "\n".join(formatted_lines)

    def _format_done_list_for_prompt(self, done_list: List[tuple[datetime.datetime, str]], limit: int = 5) -> str:
        """将已完成的活动列表格式化为易读的文本，用于Prompt，只取最近几条。"""
        if not done_list:
            return "昨天没有记录已完成的活动。"
        recent_done = done_list[-limit:]
        formatted_lines = []
        for timestamp, activity in recent_done:
            formatted_lines.append(f"- {timestamp.strftime('%H:%M')}: {activity}")
        if not formatted_lines: # 以防万一 limit=0 或 done_list 为空
            return "昨天没有记录已完成的活动。"
        return "回顾昨天完成的部分活动：\n" + "\n".join(formatted_lines)


    async def _generate_daily_outline(self, target_date: datetime.datetime):
        """为指定日期生成概要日程计划。"""
        date_str = target_date.strftime("%Y-%m-%d")
        weekday = target_date.strftime("%A") # %A 是星期几的全名，如 "Saturday"

        prompt_personality = self.individuality.get_prompt(x_person=2, level=3) # 假设参数与之前讨论一致

        # 准备昨天日程的参考信息
        yesterday_outline_str = self._format_outline_for_prompt(self.yesterday_outline_plan)
        yesterday_done_str = self._format_done_list_for_prompt(self.yesterday_done_list)

        prompt = f"""你是{self.name}，一位{prompt_personality}
你的日常行为习惯大致是：{self.behavior}

回顾你昨天的概要日程：
{yesterday_outline_str}
以及昨天实际完成的部分活动：
{yesterday_done_str}

请为今天 ({date_str}，{weekday}) 生成一个概要性的日程计划。
计划应主要覆盖从早上约9点到晚上约22点的主要活动安排，请尽量按【小时】为单位规划每个小时的概要活动。
如果某些小时没有特别的安排，可以描述为“弹性时间”或“处理杂务”。
请避免规划深夜和凌晨（例如 00:00 至 08:00）的活动，这些时间段通常用于休息。

请严格以JSON列表格式输出，每个列表项为一个包含 "time_hour" (例如 "09:00", "10:00"...) 和 "activity_summary" (该小时的活动概要描述) 的对象。
例如：
[
  {{"time_hour": "09:00", "activity_summary": "阅读行业新闻，整理今日待办"}},
  {{"time_hour": "10:00", "activity_summary": "核心工作任务A - 方案设计阶段"}},
  {{"time_hour": "11:00", "activity_summary": "参与团队线上会议，讨论项目进展"}},
  {{"time_hour": "12:00", "activity_summary": "午餐与短暂休息"}},
  {{"time_hour": "13:00", "activity_summary": "核心工作任务B - 代码实现"}},
  {{"time_hour": "14:00", "activity_summary": "回复邮件和即时消息，处理协作请求"}},
  {{"time_hour": "15:00", "activity_summary": "学习新技术X，观看教程视频"}},
  {{"time_hour": "16:00", "activity_summary": "弹性时间，处理突发事务或整理笔记"}},
  {{"time_hour": "17:00", "activity_summary": "今日工作总结与明日计划初步构思"}},
  {{"time_hour": "18:00", "activity_summary": "晚餐时间"}},
  {{"time_hour": "19:00", "activity_summary": "个人兴趣项目 - Y项目开发"}},
  {{"time_hour": "20:00", "activity_summary": "在线浏览或参与社群讨论"}},
  {{"time_hour": "21:00", "activity_summary": "阅读书籍或文章，放松身心"}},
  {{"time_hour": "22:00", "activity_summary": "准备休息，结束今日活动"}}
]
确保只输出JSON内容，不要包含任何额外的解释或Markdown标记。
"""
        logger.debug(f"生成概要日程的Prompt:\n{prompt}")

        try:
            outline_response_text, _ = await self.llm_scheduler_all.generate_response_async(prompt)
            logger.debug(f"LLM返回的概要日程文本: {outline_response_text}")

            # 清理 LLM 可能返回的 markdown 代码块标记
            if outline_response_text.startswith("```json"):
                outline_response_text = outline_response_text[7:]
                if outline_response_text.endswith("```"):
                    outline_response_text = outline_response_text[:-3]
            outline_response_text = outline_response_text.strip()

            parsed_outline = json.loads(outline_response_text)
            if isinstance(parsed_outline, list) and all(isinstance(item, dict) and "time_hour" in item and "activity_summary" in item for item in parsed_outline):
                self.today_outline_plan = parsed_outline
                logger.info(f"成功生成并解析了 {date_str} 的概要日程，共 {len(self.today_outline_plan)} 条。")
            else:
                logger.error(f"LLM返回的概要日程JSON格式不正确: {parsed_outline}")
                self.today_outline_plan = [] # 使用空列表作为回退
        except json.JSONDecodeError as e:
            logger.error(f"解析LLM返回的概要日程JSON时出错: {e}. LLM原始输出: {outline_response_text}")
            self.today_outline_plan = [] # 解析失败时使用空列表
        except Exception as e:
            logger.error(f"生成概要日程时发生意外错误: {e}")
            logger.exception("详细错误信息：")
            self.today_outline_plan = [] # 其他错误时使用空列表

        self._save_schedule_to_db() # 保存新生成的或空的概要计划


    def _save_schedule_to_db(self):
        """保存当前实例的日程信息 (概要计划和已完成列表) 到数据库。"""
        date_str = self.start_time.strftime("%Y-%m-%d") # 使用 self.start_time 确保日期一致性
        schedule_data = {
            "date": date_str,
            "outline_plan": self.today_outline_plan, # 新字段
            "done_list": self.today_done_list, # 修改字段名以更清晰
        }
        db.schedule.update_one({"date": date_str}, {"$set": schedule_data}, upsert=True)
        logger.debug(f"已保存 {date_str} 的日程信息到数据库 (概要计划: {len(self.today_outline_plan)} 项, 已完成: {len(self.today_done_list)} 项)。")

    def _load_schedule_from_db(self, date: datetime.datetime) -> tuple[Optional[List[Dict[str, str]]], Optional[List[tuple[datetime.datetime, str]]]]:
        """从数据库加载指定日期的日程信息 (概要计划和已完成列表)。"""
        date_str = date.strftime("%Y-%m-%d")
        existing_schedule = db.schedule.find_one({"date": date_str})

        if existing_schedule:
            outline_plan = existing_schedule.get("outline_plan")
            done_list_raw = existing_schedule.get("done_list") # 之前可能是 today_done_list

            # done_list 中的时间戳可能需要从数据库格式转换回 datetime 对象
            # (如果它们是以字符串或其他可序列化格式存储的话)
            # 这里假设它们直接以 Python datetime 对象兼容的方式存储，或者在保存时已处理
            # 如果没有，需要在这里添加转换逻辑
            # 例如，如果时间戳存为 ISO 格式字符串:
            # done_list_typed = []
            # if done_list_raw:
            #     for item in done_list_raw:
            #         if isinstance(item, (list, tuple)) and len(item) == 2:
            #             try:
            #                 # 假设 item[0] 是时间戳，item[1] 是活动描述
            #                 # 如果 item[0] 是字符串，需要解析
            #                 # t_stamp = datetime.datetime.fromisoformat(item[0]).replace(tzinfo=TIME_ZONE) if isinstance(item[0], str) else item[0]
            #                 # done_list_typed.append((t_stamp, item[1]))
            #                 # 暂时简化，假设已经是 (datetime, str) 元组列表
            #                  done_list_typed.append(item)
            #             except Exception as e:
            #                 logger.warning(f"从数据库加载 done_list 时，解析时间戳失败: {item[0]}, error: {e}")
            #         else:
            #             logger.warning(f"从数据库加载 done_list 时，发现格式不正确的项: {item}")


            # 确保返回的是正确类型，即使数据库中字段不存在或为null
            final_outline_plan = outline_plan if isinstance(outline_plan, list) else None
            final_done_list = done_list_raw if isinstance(done_list_raw, list) else None

            logger.debug(f"从数据库加载了 {date_str} 的日程 (概要计划: {len(final_outline_plan) if final_outline_plan else 0} 项, 已完成: {len(final_done_list) if final_done_list else 0} 项)。")
            return final_outline_plan, final_done_list
        else:
            logger.debug(f"{date_str} 的日程记录不存在于数据库。")
            return None, None
        
    def _get_current_mind(self) -> str:
        """
        获取机器人当前的主要想法。
        具体的实现方式取决于您的项目结构。
        这里提供一个占位符实现，您需要根据实际情况修改。
        """
        # 方案1: 如果 heartflow 是全局可访问的单例
        try:
            from src.chat.heart_flow.heartflow import heartflow # 尝试导入
            if heartflow and hasattr(heartflow, 'current_mind'):
                return heartflow.current_mind
            else:
                logger.warning("无法获取全局 heartflow 实例的 current_mind。")
                return "暂时没什么特别的想法。"
        except ImportError:
            logger.warning("无法导入全局 heartflow 实例。")
            return "暂时没什么特别的想法。"
        except Exception as e:
            logger.error(f"获取 current_mind 时出错: {e}")
            return "暂时没什么特别的想法。"


    def construct_doing_prompt(
        self,
        current_time_dt: datetime.datetime,
        current_outline_plan: List[Dict[str, str]],
        # mind_thinking: str, # 改为从内部获取或可选
    ) -> str:
        """构造决定当前具体做什么的Prompt，并请求判断是否需要更新概要计划。"""
        now_time_str = current_time_dt.strftime("%H:%M")
        weekday_str = current_time_dt.strftime("%A")
        previous_doings_str = self._format_done_list_for_prompt(self.today_done_list, limit=3) # 只看最近3条完成的
        formatted_outline_plan_str = self._format_outline_for_prompt(current_outline_plan)
        prompt_personality = self.individuality.get_prompt(x_person=2, level=3)
        mind_thinking = self._get_current_mind() # 从内部获取当前想法

        prompt = f"""你是{self.name}，一位{prompt_personality}
你的日常行为习惯大致是：{self.behavior}

今天是{weekday_str}。
你当前的概要日程计划是：
{formatted_outline_plan_str}

你最近完成的几项活动是：
{previous_doings_str}
距离上一项记录的活动可能已经过去一段时间（约 {self.schedule_doing_update_interval / 60:.0f} 分钟）。

你脑子里可能在想：{mind_thinking}

现在是 {now_time_str}。
请根据你的概要计划、你之前的活动、当前的想法以及你的个性和行为习惯，具体决定你接下来半小时到一小时内要做什么。
你的决定应该是具体且可执行的活动。

同时，请判断你的这个决定是否会显著影响或改变后续的概要日程计划。
例如，如果你决定做一件计划外的重要事情，或者某项计划内活动需要比预期更长的时间，这可能就需要调整概要计划。

请严格以JSON格式输出你的决定，包含以下字段：
- "current_activity": "string" (你当前决定做的具体事情，例如："开始阅读关于XX技术的文档" 或 "回复YY群组的消息")
- "duration_suggestion": "string" (这项活动大概持续多久，例如: "约30分钟", "大约1小时", "直到完成XX任务")
- "impacts_outline_plan": boolean (这个决定是否会导致后续的概要日程计划需要调整？true 或 false)
- "outline_update_suggestion": "string" (如果impacts_outline_plan为true，请提供调整概要计划的简要理由或建议，例如：“下午的学习时间需要用来处理紧急工作”或“原定休息时间推迟”。如果为false，则此字段留空字符串。)

输出示例1 (不影响概要计划):
{{
  "current_activity": "继续完成上午未完成的报告撰写工作",
  "duration_suggestion": "大约1小时",
  "impacts_outline_plan": false,
  "outline_update_suggestion": ""
}}

输出示例2 (可能影响概要计划):
{{
  "current_activity": "突发！需要立即处理一个紧急的客户请求",
  "duration_suggestion": "预计花费1-2小时",
  "impacts_outline_plan": true,
  "outline_update_suggestion": "原定下午的学习计划可能需要推迟或取消，优先处理客户请求。"
}}

确保只输出JSON内容，不要包含任何额外的解释或Markdown标记。
"""
        logger.debug(f"构造 doing_prompt:\n{prompt}")
        return prompt

    async def move_doing(self) -> Optional[str]: # 移除 mind_thinking 参数
        """决定当前做什么，记录活动，并根据需要触发概要计划的更新。"""
        current_time = datetime.datetime.now(TIME_ZONE)

        # 1. 构造Prompt
        # mind_thinking_for_prompt = mind_thinking if mind_thinking else self._get_current_mind()
        doing_prompt = self.construct_doing_prompt(
            current_time_dt=current_time,
            current_outline_plan=self.today_outline_plan,
            # mind_thinking=mind_thinking_for_prompt,
        )

        # 2. 调用LLM获取当前活动决定
        try:
            activity_decision_text, _ = await self.llm_scheduler_doing.generate_response_async(doing_prompt)
            logger.debug(f"LLM返回的当前活动决定文本: {activity_decision_text}")

            # 清理LLM可能返回的markdown
            if activity_decision_text.startswith("```json"):
                activity_decision_text = activity_decision_text[7:]
                if activity_decision_text.endswith("```"):
                    activity_decision_text = activity_decision_text[:-3]
            activity_decision_text = activity_decision_text.strip()

            activity_decision = json.loads(activity_decision_text)

            current_activity = activity_decision.get("current_activity", "思考下一步行动...")
            # duration_suggestion = activity_decision.get("duration_suggestion", "未知时长") # 可以记录，但暂时不直接使用
            impacts_outline = activity_decision.get("impacts_outline_plan", False)
            update_suggestion = activity_decision.get("outline_update_suggestion", "")

        except json.JSONDecodeError as e:
            logger.error(f"解析LLM返回的当前活动JSON时出错: {e}. LLM原始输出: {activity_decision_text}")
            current_activity = "解析活动决定时出错，暂时休息一下"
            impacts_outline = False
            update_suggestion = ""
        except Exception as e:
            logger.error(f"获取当前活动决定时发生意外错误: {e}")
            logger.exception("详细错误信息：")
            current_activity = "决定当前活动时出错，稍后重试"
            impacts_outline = False
            update_suggestion = ""

        # 3. 记录当前完成的活动
        self.today_done_list.append((current_time, current_activity))
        logger.info(f"当前活动 ({current_time.strftime('%H:%M')}): {current_activity}")

        # 4. 触发概要计划更新（如果需要）
        if impacts_outline:
            logger.info(f"当前活动 '{current_activity}' 可能影响概要计划，建议调整：'{update_suggestion}'。开始更新概要计划...")
            await self._update_daily_outline(reason_for_update=update_suggestion, activity_that_triggered_update=current_activity)
        else:
            logger.debug(f"当前活动 '{current_activity}' 不影响概要计划，或LLM未建议调整。")

        # 5. 统一保存日程信息（包括更新的done_list和可能更新的outline_plan）
        self._save_schedule_to_db()

        return current_activity


    async def _update_daily_outline(self, reason_for_update: str, activity_that_triggered_update: str):
        """根据新的情况更新或重新生成当日后续的概要计划。"""
        current_time_str = datetime.datetime.now(TIME_ZONE).strftime("%H:%M")
        prompt_personality = self.individuality.get_prompt(x_person=2, level=3)
        current_outline_json_string = json.dumps(self.today_outline_plan, ensure_ascii=False, indent=2)
        # 获取最近的已完成活动作为上下文
        recent_done_activities = self._format_done_list_for_prompt(self.today_done_list, limit=3)


        prompt = f"""你是{self.name}，一位{prompt_personality}
你的日常行为习惯大致是：{self.behavior}

你当前的概要日程计划是：
{current_outline_json_string}

你最近完成的活动包括：
{recent_done_activities}

现在是 {current_time_str}。
由于你刚刚决定要进行活动：“{activity_that_triggered_update}”，这导致了以下情况或考量： “{reason_for_update}”。
这可能需要你调整从现在开始的后续概要日程计划。

请根据这个新的情况，审视并更新或重新规划从当前时间点 ({current_time_str}) 开始的后续概要日程计划。
请确保新的概要计划与你的人格、行为习惯以及当前的时间相符。
新的计划应主要覆盖从当前小时往后到晚上约22点的主要活动安排，请尽量按【小时】为单位规划。
如果某些后续小时没有特别的安排，可以描述为“弹性时间”或“处理杂务”。
请避免规划深夜和凌晨（例如 00:00 至 08:00）的活动。

请严格以JSON列表格式输出【更新后的完整当日概要计划】，保持原有结构 (每个列表项为一个包含 "time_hour" 和 "activity_summary" 的对象)。
你需要输出的是从早上开始到晚上结束的【一整天】的更新后的概要计划，而不仅仅是修改的部分。
例如，如果现在是14:00，你需要输出从09:00（或其他你的活动开始时间）到22:00的完整计划，其中14:00之后的部分会根据当前情况调整。

确保只输出JSON内容，不要包含任何额外的解释或Markdown标记。
"""
        logger.debug(f"更新概要日程的Prompt:\n{prompt}")

        try:
            updated_outline_response_text, _ = await self.llm_scheduler_all.generate_response_async(prompt) # 使用 llm_scheduler_all
            logger.debug(f"LLM返回的更新后概要日程文本: {updated_outline_response_text}")

            if updated_outline_response_text.startswith("```json"):
                updated_outline_response_text = updated_outline_response_text[7:]
                if updated_outline_response_text.endswith("```"):
                    updated_outline_response_text = updated_outline_response_text[:-3]
            updated_outline_response_text = updated_outline_response_text.strip()

            parsed_updated_outline = json.loads(updated_outline_response_text)

            if isinstance(parsed_updated_outline, list) and \
               all(isinstance(item, dict) and "time_hour" in item and "activity_summary" in item for item in parsed_updated_outline):
                self.today_outline_plan = parsed_updated_outline
                logger.info(f"成功更新并解析了当日的概要日程，共 {len(self.today_outline_plan)} 条。")
            else:
                logger.error(f"LLM返回的更新后概要日程JSON格式不正确: {parsed_updated_outline}")
                # 保留旧的概要计划，避免错误覆盖
        except json.JSONDecodeError as e:
            logger.error(f"解析LLM返回的更新后概要日程JSON时出错: {e}. LLM原始输出: {updated_outline_response_text}")
        except Exception as e:
            logger.error(f"更新概要日程时发生意外错误: {e}")
            logger.exception("详细错误信息：")

    async def mai_schedule_start(self):
        """启动日程系统，每隔一段时间执行一次move_doing，并在日期变化时重新检查日程"""
        try:
            if global_config.schedule.enable:
                logger.info(f"日程系统启动/刷新时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                await self.check_and_create_today_schedule() # 确保当天有概要计划

                # 首次启动时，如果 done_list 为空，可以立即执行一次 move_doing 来确定初始活动
                if not self.today_done_list:
                    logger.info("日程系统首次启动或新的一天，立即尝试决定当前活动。")
                    await self.move_doing() # 不传递 mind_thinking

                while True:
                    await asyncio.sleep(self.schedule_doing_update_interval) # 先等待，再执行

                    current_time_loop = datetime.datetime.now(TIME_ZONE)
                    if current_time_loop.date() != self.start_time.date():
                        logger.info("检测到日期变化，重新生成日程概要...")
                        self.start_time = current_time_loop # 更新 start_time 为新的一天
                        await self.check_and_create_today_schedule()
                        # 新的一天开始，如果 done_list 为空，也立即执行一次 move_doing
                        if not self.today_done_list:
                             logger.info("新的一天开始，立即尝试决定当前活动。")
                             await self.move_doing() # 不传递 mind_thinking
                             continue # 开始下一个循环的等待

                    logger.debug(f"到达预定间隔，执行 move_doing...")
                    await self.move_doing() # 不传递 mind_thinking

            else:
                logger.info("日程系统未启用")

        except asyncio.CancelledError:
            logger.info("日程系统主循环被取消。")
        except Exception as e:
            logger.error(f"日程系统主循环运行时出错: {str(e)}")
            logger.exception("详细错误信息：")


    def _time_diff(self, time_str1: str, time_str2: str) -> float:
        """计算两个 HH:MM 格式时间字符串的差异（分钟），time1 - time2"""
        try:
            h1, m1 = map(int, time_str1.split(':'))
            h2, m2 = map(int, time_str2.split(':'))
            return (h1 * 60 + m1) - (h2 * 60 + m2)
        except ValueError:
            logger.warning(f"无法解析时间字符串进行比较: '{time_str1}', '{time_str2}'")
            return float('-inf') # 或其他错误指示

    async def get_task_from_time_to_time(self, start_time_str: str, end_time_str: str) -> List[tuple[datetime.datetime, str]]:
        """获取指定时间范围内的已完成活动列表。"""
        result: List[tuple[datetime.datetime, str]] = []
        if not isinstance(start_time_str, str) or not isinstance(end_time_str, str):
            logger.warning("get_task_from_time_to_time: start_time_str 和 end_time_str 必须是字符串。")
            return result

        try:
            # 将输入的 "HH:MM" 字符串转换为当天的 datetime 对象，以便与 done_list 中的时间戳进行更可靠的比较
            today_date = datetime.datetime.now(TIME_ZONE).date()
            start_dt_obj = datetime.datetime.strptime(f"{today_date} {start_time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=TIME_ZONE)
            end_dt_obj = datetime.datetime.strptime(f"{today_date} {end_time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=TIME_ZONE)
        except ValueError:
            logger.warning(f"无法将时间字符串 '{start_time_str}', '{end_time_str}' 转换为有效的 datetime 对象。")
            return result

        for task_timestamp, activity_description in self.today_done_list:
            if start_dt_obj <= task_timestamp <= end_dt_obj:
                result.append((task_timestamp, activity_description))
        return result



    def get_current_num_task(self, num: int = 1, time_info: bool = False) -> str:
        """
        获取最新完成的指定数量的活动，并以字符串形式返回。
        这个方法主要用于向LLM提供上下文，字符串格式可能更适合Prompt。
        """
        if not self.today_done_list:
            return "最近没有已完成的活动。" if not time_info else "最近没有已完成的活动记录时间。"

        actual_num = min(num, len(self.today_done_list))
        if actual_num == 0:
            return "最近没有已完成的活动。" if not time_info else "最近没有已完成的活动记录时间。"

        selected_tasks = self.today_done_list[-actual_num:]
        
        output_lines = []
        for task_timestamp, activity_description in selected_tasks:
            if time_info:
                time_str = task_timestamp.strftime("%H:%M")
                output_lines.append(f"{time_str}时，{activity_description}")
            else:
                output_lines.append(activity_description)
        
        return "\n".join(output_lines)
    
    def print_schedule_outline(self):
        """打印当前的概要日程计划和最近的已完成活动。"""
        logger.info("=== 今日概要日程计划 ===")
        if not self.today_outline_plan:
            logger.info("  (当前没有概要计划)")
        else:
            for item in self.today_outline_plan:
                time_hour = item.get("time_hour", "未知时间")
                activity_summary = item.get("activity_summary", "未定")
                logger.info(f"  - {time_hour}: {activity_summary}")
        
        logger.info("\n=== 最近完成的活动 ===")
        if not self.today_done_list:
            logger.info("  (今日尚无已完成的活动)")
        else:
            # 只打印最近5条，避免过多输出
            for task_timestamp, activity_description in self.today_done_list[-5:]:
                logger.info(f"  - {task_timestamp.strftime('%Y-%m-%d %H:%M')}: {activity_description}")
        logger.info("======================")
        # self.enable_output = False # 这个属性的具体作用尚不明确，暂时保留注释
    

async def main():
    # 使用示例
    scheduler = ScheduleGenerator()
    scheduler.initialize(
        name="麦麦",
        # personality="你叫麦麦，你19岁，是一个大二的女大学生，你有一头黑色短发，你会刷贴吧，你现在在学习心理学",
        behavior="你比较内向，一般熬夜比较晚，然后第二天早上10点起床吃早午饭",
        interval=60,
    )
    try:
        await scheduler.mai_schedule_start()
    except KeyboardInterrupt:
        logger.info("通过Ctrl+C手动停止日程测试。")


# 当作为组件导入时使用的实例
bot_schedule = ScheduleGenerator()

if __name__ == "__main__":
    from src.common.logger_manager import setup_logging
    setup_logging() # 假设有一个全局的日志设置函数

    # 当直接运行此文件时执行
    asyncio.run(main())
