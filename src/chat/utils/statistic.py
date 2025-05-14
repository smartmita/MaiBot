from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple, List

from src.common.logger import get_module_logger
from src.manager.async_task_manager import AsyncTask

from ...common.database import db
from src.manager.local_store_manager import local_storage

logger = get_module_logger("maibot_statistic")

# 统计数据的键
TOTAL_REQ_CNT = "total_requests"
TOTAL_COST = "total_cost"
REQ_CNT_BY_TYPE = "requests_by_type"
REQ_CNT_BY_USER = "requests_by_user"
REQ_CNT_BY_MODEL = "requests_by_model"
IN_TOK_BY_TYPE = "in_tokens_by_type"
IN_TOK_BY_USER = "in_tokens_by_user"
IN_TOK_BY_MODEL = "in_tokens_by_model"
OUT_TOK_BY_TYPE = "out_tokens_by_type"
OUT_TOK_BY_USER = "out_tokens_by_user"
OUT_TOK_BY_MODEL = "out_tokens_by_model"
TOTAL_TOK_BY_TYPE = "tokens_by_type"
TOTAL_TOK_BY_USER = "tokens_by_user"
TOTAL_TOK_BY_MODEL = "tokens_by_model"
COST_BY_TYPE = "costs_by_type"
COST_BY_USER = "costs_by_user"
COST_BY_MODEL = "costs_by_model"
ONLINE_TIME = "online_time"
TOTAL_MSG_CNT = "total_messages"
MSG_CNT_BY_CHAT = "messages_by_chat"


class OnlineTimeRecordTask(AsyncTask):
    """在线时间记录任务"""

    def __init__(self):
        super().__init__(task_name="Online Time Record Task", run_interval=60)

        self.record_id: str | None = None
        """记录ID"""

        self._init_database()  # 初始化数据库

    @staticmethod
    def _init_database():
        """初始化数据库"""
        if "online_time" not in db.list_collection_names():
            # 初始化数据库（在线时长）
            db.create_collection("online_time")
            # 创建索引
            if ("end_timestamp", 1) not in db.online_time.list_indexes():
                db.online_time.create_index([("end_timestamp", 1)])

    async def run(self):
        try:
            if self.record_id:
                # 如果有记录，则更新结束时间
                db.online_time.update_one(
                    {"_id": self.record_id},
                    {
                        "$set": {
                            "end_timestamp": datetime.now() + timedelta(minutes=1),
                        }
                    },
                )
            else:
                # 如果没有记录，检查一分钟以内是否已有记录
                current_time = datetime.now()
                recent_record = db.online_time.find_one(
                    {"end_timestamp": {"$gte": current_time - timedelta(minutes=1)}}
                )

                if not recent_record:
                    # 若没有记录，则插入新的在线时间记录
                    self.record_id = db.online_time.insert_one(
                        {
                            "start_timestamp": current_time,
                            "end_timestamp": current_time + timedelta(minutes=1),
                        }
                    ).inserted_id
                else:
                    # 如果有记录，则更新结束时间
                    self.record_id = recent_record["_id"]
                    db.online_time.update_one(
                        {"_id": self.record_id},
                        {
                            "$set": {
                                "end_timestamp": current_time + timedelta(minutes=1),
                            }
                        },
                    )
        except Exception:
            logger.exception("在线时间记录失败")


def _format_online_time(online_seconds: int) -> str:
    """
    格式化在线时间
    :param online_seconds: 在线时间（秒）
    :return: 格式化后的在线时间字符串
    """
    total_oneline_time = timedelta(seconds=int(online_seconds)) #确保是整数

    days = total_oneline_time.days
    hours = total_oneline_time.seconds // 3600
    minutes = (total_oneline_time.seconds // 60) % 60
    seconds = total_oneline_time.seconds % 60
    if days > 0:
        # 如果在线时间超过1天，则格式化为"X天X小时X分钟"
        total_oneline_time_str = f"{total_oneline_time.days}天{hours}小时{minutes}分钟{seconds}秒"
    elif hours > 0:
        # 如果在线时间超过1小时，则格式化为"X小时X分钟X秒"
        total_oneline_time_str = f"{hours}小时{minutes}分钟{seconds}秒"
    else:
        # 其他情况格式化为"X分钟X秒"
        total_oneline_time_str = f"{minutes}分钟{seconds}秒"

    return total_oneline_time_str


class StatisticOutputTask(AsyncTask):
    """统计输出任务"""

    SEP_LINE = "-" * 84

    def __init__(self, record_file_path: str = "maibot_statistics.html"):
        # 延迟300秒启动，运行间隔300秒
        super().__init__(task_name="Statistics Data Output Task", wait_before_start=0, run_interval=300)

        self.name_mapping: Dict[str, Tuple[str, float]] = {}
        """
            联系人/群聊名称映射 {聊天ID: (联系人/群聊名称, 记录时间（timestamp）)}
            注：设计记录时间的目的是方便更新名称，使联系人/群聊名称保持最新
        """

        self.record_file_path: str = record_file_path
        """
        记录文件路径
        """

        now = datetime.now() # Renamed to avoid conflict with 'now' in methods
        if "deploy_time" in local_storage:
            # 如果存在部署时间，则使用该时间作为全量统计的起始时间
            deploy_time = datetime.fromtimestamp(local_storage["deploy_time"])
        else:
            # 否则，使用最大时间范围，并记录部署时间为当前时间
            deploy_time = datetime(2000, 1, 1)
            local_storage["deploy_time"] = now.timestamp()

        self.stat_period: List[Tuple[str, timedelta, str]] = [
            ("all_time", now - deploy_time, "自部署以来"),  # 必须保留"all_time"
            ("last_7_days", timedelta(days=7), "最近7天"),
            ("last_24_hours", timedelta(days=1), "最近24小时"),
            ("last_hour", timedelta(hours=1), "最近1小时"),
        ]
        """
        统计时间段 [(统计名称, 统计时间段, 统计描述), ...]
        """

    def _statistic_console_output(self, stats: Dict[str, Any], now: datetime):
        """
        输出统计数据到控制台
        :param stats: 统计数据
        :param now: 基准当前时间
        """
        # 输出最近一小时的统计数据
        last_hour_stats = stats.get("last_hour", {}) # Ensure 'last_hour' key exists

        output = [
            self.SEP_LINE,
            f"  最近1小时的统计数据  (自{now.strftime('%Y-%m-%d %H:%M:%S')}开始，详细信息见文件：{self.record_file_path})",
            self.SEP_LINE,
            self._format_total_stat(last_hour_stats),
            "",
            self._format_model_classified_stat(last_hour_stats),
            "",
            self._format_chat_stat(last_hour_stats),
            self.SEP_LINE,
            "",
        ]

        logger.info("\n" + "\n".join(output))

    async def run(self):
        try:
            now = datetime.now()
            # 收集统计数据
            stats = self._collect_all_statistics(now)

            # 输出统计数据到控制台
            if "last_hour" in stats: # Check if stats for last_hour were successfully collected
                self._statistic_console_output(stats, now)
            else:
                logger.warning("无法输出最近一小时统计数据到控制台，因为数据缺失。")
            # 输出统计数据到html文件
            self._generate_html_report(stats, now)
        except Exception as e:
            logger.exception(f"输出统计数据过程中发生异常，错误信息：{e}")

    # -- 以下为统计数据收集方法 --

    @staticmethod
    def _collect_model_request_for_period(collect_period: List[Tuple[str, datetime]]) -> Dict[str, Any]:
        """
        收集指定时间段的LLM请求统计数据

        :param collect_period: 统计时间段 [(period_key, start_datetime), ...]
        """
        if not collect_period: 
            return {}

        collect_period.sort(key=lambda x: x[1], reverse=True)

        stats = {
            period_key: {
                TOTAL_REQ_CNT: 0,
                REQ_CNT_BY_TYPE: defaultdict(int),
                REQ_CNT_BY_USER: defaultdict(int),
                REQ_CNT_BY_MODEL: defaultdict(int),
                IN_TOK_BY_TYPE: defaultdict(int),
                IN_TOK_BY_USER: defaultdict(int),
                IN_TOK_BY_MODEL: defaultdict(int),
                OUT_TOK_BY_TYPE: defaultdict(int),
                OUT_TOK_BY_USER: defaultdict(int),
                OUT_TOK_BY_MODEL: defaultdict(int),
                TOTAL_TOK_BY_TYPE: defaultdict(int),
                TOTAL_TOK_BY_USER: defaultdict(int),
                TOTAL_TOK_BY_MODEL: defaultdict(int),
                TOTAL_COST: 0.0,
                COST_BY_TYPE: defaultdict(float),
                COST_BY_USER: defaultdict(float),
                COST_BY_MODEL: defaultdict(float),
            }
            for period_key, _ in collect_period
        }
        
        # Determine the overall earliest start time for the database query
        # This assumes collect_period is not empty, which is checked at the beginning.
        overall_earliest_start_time = min(p[1] for p in collect_period)

        for record in db.llm_usage.find({"timestamp": {"$gte": overall_earliest_start_time}}):
            record_timestamp = record.get("timestamp")
            if not isinstance(record_timestamp, datetime): # Ensure timestamp is a datetime object
                try: # Attempt conversion if it's a number (e.g. Unix timestamp)
                    record_timestamp = datetime.fromtimestamp(float(record_timestamp))
                except (ValueError, TypeError):
                    logger.warning(f"Skipping LLM usage record with invalid timestamp: {record.get('_id')}")
                    continue


            for idx, (_current_period_key, period_start_time) in enumerate(collect_period):
                if record_timestamp >= period_start_time:
                    for period_key_to_update, _ in collect_period[idx:]:
                        stats[period_key_to_update][TOTAL_REQ_CNT] += 1

                        request_type = record.get("request_type", "unknown")
                        user_id = str(record.get("user_id", "unknown"))
                        model_name = record.get("model_name", "unknown")

                        stats[period_key_to_update][REQ_CNT_BY_TYPE][request_type] += 1
                        stats[period_key_to_update][REQ_CNT_BY_USER][user_id] += 1
                        stats[period_key_to_update][REQ_CNT_BY_MODEL][model_name] += 1

                        prompt_tokens = record.get("prompt_tokens", 0)
                        completion_tokens = record.get("completion_tokens", 0)
                        total_tokens = prompt_tokens + completion_tokens

                        stats[period_key_to_update][IN_TOK_BY_TYPE][request_type] += prompt_tokens
                        stats[period_key_to_update][IN_TOK_BY_USER][user_id] += prompt_tokens
                        stats[period_key_to_update][IN_TOK_BY_MODEL][model_name] += prompt_tokens

                        stats[period_key_to_update][OUT_TOK_BY_TYPE][request_type] += completion_tokens
                        stats[period_key_to_update][OUT_TOK_BY_USER][user_id] += completion_tokens
                        stats[period_key_to_update][OUT_TOK_BY_MODEL][model_name] += completion_tokens

                        stats[period_key_to_update][TOTAL_TOK_BY_TYPE][request_type] += total_tokens
                        stats[period_key_to_update][TOTAL_TOK_BY_USER][user_id] += total_tokens
                        stats[period_key_to_update][TOTAL_TOK_BY_MODEL][model_name] += total_tokens

                        cost = record.get("cost", 0.0)
                        stats[period_key_to_update][TOTAL_COST] += cost
                        stats[period_key_to_update][COST_BY_TYPE][request_type] += cost
                        stats[period_key_to_update][COST_BY_USER][user_id] += cost
                        stats[period_key_to_update][COST_BY_MODEL][model_name] += cost
                    break  

        return stats

    @staticmethod
    def _collect_online_time_for_period(collect_period: List[Tuple[str, datetime]], now: datetime) -> Dict[str, Any]:
        """
        收集指定时间段的在线时间统计数据

        :param collect_period: 统计时间段 [(period_key, start_datetime), ...]
        :param now: 当前时间，用于校准end_timestamp
        """
        if not collect_period:
            return {}

        collect_period.sort(key=lambda x: x[1], reverse=True)

        stats = {
            period_key: {
                ONLINE_TIME: 0.0,
            }
            for period_key, _ in collect_period
        }
        
        overall_earliest_start_time = min(p[1] for p in collect_period)

        for record in db.online_time.find({"end_timestamp": {"$gte": overall_earliest_start_time}}):
            record_end_timestamp: datetime = record.get("end_timestamp")
            record_start_timestamp: datetime = record.get("start_timestamp")

            if not isinstance(record_end_timestamp, datetime) or not isinstance(record_start_timestamp, datetime):
                logger.warning(f"Skipping online_time record with invalid timestamps: {record.get('_id')}")
                continue

            actual_end_timestamp = min(record_end_timestamp, now)

            for idx, (_current_period_key, period_start_time) in enumerate(collect_period):
                if record_start_timestamp < now and actual_end_timestamp > period_start_time:
                    overlap_start = max(record_start_timestamp, period_start_time)
                    overlap_end = min(actual_end_timestamp, now) 

                    if overlap_end > overlap_start: 
                        duration_seconds = (overlap_end - overlap_start).total_seconds()
                        for period_key_to_update, _ in collect_period[idx:]:
                            stats[period_key_to_update][ONLINE_TIME] += duration_seconds
                        break 

        return stats

    def _collect_message_count_for_period(self, collect_period: List[Tuple[str, datetime]]) -> Dict[str, Any]:
        """
        收集指定时间段的消息统计数据

        :param collect_period: 统计时间段 [(period_key, start_datetime), ...]
        """
        if not collect_period:
            return {}

        collect_period.sort(key=lambda x: x[1], reverse=True)

        stats = {
            period_key: {
                TOTAL_MSG_CNT: 0,
                MSG_CNT_BY_CHAT: defaultdict(int),
            }
            for period_key, _ in collect_period
        }

        overall_earliest_start_timestamp_float = min(p[1].timestamp() for p in collect_period)
        
        for message in db.messages.find({"time": {"$gte": overall_earliest_start_timestamp_float}}):
            chat_info = message.get("chat_info", {}) 
            user_info = message.get("user_info", {}) 
            message_time_ts = message.get("time")  

            if message_time_ts is None: 
                logger.warning(f"Skipping message record with no timestamp: {message.get('_id')}")
                continue
            
            try:
                message_datetime = datetime.fromtimestamp(float(message_time_ts))
            except (ValueError, TypeError):
                logger.warning(f"Skipping message record with invalid time format: {message.get('_id')}")
                continue


            group_info = chat_info.get("group_info")
            chat_id = None
            chat_name = None

            if group_info and group_info.get("group_id"):
                gid = group_info.get('group_id')
                chat_id = f"g{gid}"
                chat_name = group_info.get("group_name", f"群聊 {gid}")
            elif user_info and user_info.get("user_id"):
                uid = user_info['user_id']
                chat_id = f"u{uid}"
                chat_name = user_info.get("user_nickname", f"用户 {uid}")
            
            if not chat_id: 
                continue

            current_mapping = self.name_mapping.get(chat_id)
            if current_mapping:
                if chat_name != current_mapping[0] and message_time_ts > current_mapping[1]:
                    self.name_mapping[chat_id] = (chat_name, message_time_ts)
            else:
                self.name_mapping[chat_id] = (chat_name, message_time_ts)


            for idx, (_current_period_key, period_start_time) in enumerate(collect_period):
                if message_datetime >= period_start_time:
                    for period_key_to_update, _ in collect_period[idx:]:
                        stats[period_key_to_update][TOTAL_MSG_CNT] += 1
                        stats[period_key_to_update][MSG_CNT_BY_CHAT][chat_id] += 1
                    break 

        return stats

    def _collect_all_statistics(self, now: datetime) -> Dict[str, Dict[str, Any]]:
        """
        收集各时间段的统计数据
        :param now: 基准当前时间
        """
        # Correctly determine deploy_time
        if "deploy_time" in local_storage:
            try:
                deploy_time = datetime.fromtimestamp(local_storage["deploy_time"])
            except (TypeError, ValueError):
                logger.error("Invalid deploy_time in local_storage. Resetting.")
                deploy_time = datetime(2000, 1, 1)
                local_storage["deploy_time"] = now.timestamp()
        else:
            deploy_time = datetime(2000, 1, 1)
            local_storage["deploy_time"] = now.timestamp()

        # Rebuild stat_period based on the current 'now' and determined 'deploy_time'
        current_stat_periods_config = [
            ("all_time", now - deploy_time if now > deploy_time else timedelta(seconds=0), "自部署以来"),
            ("last_7_days", timedelta(days=7), "最近7天"),
            ("last_24_hours", timedelta(days=1), "最近24小时"),
            ("last_hour", timedelta(hours=1), "最近1小时"),
        ]
        self.stat_period = current_stat_periods_config # Update instance's stat_period if needed elsewhere

        stat_start_timestamp_config = []
        for period_name, delta, _ in current_stat_periods_config:
            start_dt = deploy_time if period_name == "all_time" else now - delta
            stat_start_timestamp_config.append((period_name, start_dt))

        # 收集各类数据
        model_req_stat = self._collect_model_request_for_period(stat_start_timestamp_config)
        online_time_stat = self._collect_online_time_for_period(stat_start_timestamp_config, now)
        message_count_stat = self._collect_message_count_for_period(stat_start_timestamp_config)

        final_stats = {}
        for period_key, _ in stat_start_timestamp_config:
            final_stats[period_key] = {}
            final_stats[period_key].update(model_req_stat.get(period_key, {}))
            final_stats[period_key].update(online_time_stat.get(period_key, {}))
            final_stats[period_key].update(message_count_stat.get(period_key, {}))
            
            for stat_field_key in [
                TOTAL_REQ_CNT, REQ_CNT_BY_TYPE, REQ_CNT_BY_USER, REQ_CNT_BY_MODEL,
                IN_TOK_BY_TYPE, IN_TOK_BY_USER, IN_TOK_BY_MODEL,
                OUT_TOK_BY_TYPE, OUT_TOK_BY_USER, OUT_TOK_BY_MODEL,
                TOTAL_TOK_BY_TYPE, TOTAL_TOK_BY_USER, TOTAL_TOK_BY_MODEL,
                TOTAL_COST, COST_BY_TYPE, COST_BY_USER, COST_BY_MODEL,
                ONLINE_TIME, TOTAL_MSG_CNT, MSG_CNT_BY_CHAT
            ]:
                if stat_field_key not in final_stats[period_key]:
                    # Initialize with appropriate default type if key is missing
                    if "BY_" in stat_field_key: # These are usually defaultdicts
                        final_stats[period_key][stat_field_key] = defaultdict(int if "CNT" in stat_field_key or "TOK" in stat_field_key else float)
                    elif "CNT" in stat_field_key or "TOK" in stat_field_key :
                         final_stats[period_key][stat_field_key] = 0
                    elif "COST" in stat_field_key or ONLINE_TIME == stat_field_key:
                         final_stats[period_key][stat_field_key] = 0.0
        return final_stats

    # -- 以下为统计数据格式化方法 --

    @staticmethod
    def _format_total_stat(stats: Dict[str, Any]) -> str:
        """
        格式化总统计数据
        """
        output = [
            f"总在线时间: {_format_online_time(stats.get(ONLINE_TIME, 0))}",
            f"总消息数: {stats.get(TOTAL_MSG_CNT, 0)}",
            f"总请求数: {stats.get(TOTAL_REQ_CNT, 0)}",
            f"总花费: {stats.get(TOTAL_COST, 0.0):.4f}¥",
            "",
        ]
        return "\n".join(output)

    @staticmethod
    def _format_model_classified_stat(stats: Dict[str, Any]) -> str:
        """
        格式化按模型分类的统计数据
        """
        if stats.get(TOTAL_REQ_CNT, 0) > 0:
            data_fmt = "{:<32}  {:>10}  {:>12}  {:>12}  {:>12}  {:>9.4f}¥"
            output = [
                "按模型分类统计:",
                " 模型名称                          调用次数    输入Token     输出Token     Token总量     累计花费",
            ]
            req_cnt_by_model = stats.get(REQ_CNT_BY_MODEL, {})
            in_tok_by_model = stats.get(IN_TOK_BY_MODEL, defaultdict(int))
            out_tok_by_model = stats.get(OUT_TOK_BY_MODEL, defaultdict(int))
            total_tok_by_model = stats.get(TOTAL_TOK_BY_MODEL, defaultdict(int))
            cost_by_model = stats.get(COST_BY_MODEL, defaultdict(float))

            for model_name, count in sorted(req_cnt_by_model.items()):
                name = model_name[:29] + "..." if len(model_name) > 32 else model_name
                in_tokens = in_tok_by_model[model_name]
                out_tokens = out_tok_by_model[model_name]
                tokens = total_tok_by_model[model_name]
                cost = cost_by_model[model_name]
                output.append(data_fmt.format(name, count, in_tokens, out_tokens, tokens, cost))

            output.append("")
            return "\n".join(output)
        else:
            return ""

    def _format_chat_stat(self, stats: Dict[str, Any]) -> str:
        """
        格式化聊天统计数据
        """
        if stats.get(TOTAL_MSG_CNT, 0) > 0:
            output = ["聊天消息统计:", " 联系人/群组名称                  消息数量"]
            msg_cnt_by_chat = stats.get(MSG_CNT_BY_CHAT, {}) 
            for chat_id, count in sorted(msg_cnt_by_chat.items()):
                chat_name_display = self.name_mapping.get(chat_id, (f"未知 ({chat_id})", None))[0]
                output.append(f"{chat_name_display[:32]:<32}  {count:>10}")

            output.append("")
            return "\n".join(output)
        else:
            return ""

    def _generate_html_report(self, stat_collection: dict[str, Any], now: datetime):
        """
        生成HTML格式的统计报告
        :param stat_collection: 包含所有时间段统计数据的字典 {period_key: stats_dict}
        :param now: 基准当前时间
        """
        # Correctly get deploy_time_dt for display purposes
        if "deploy_time" in local_storage:
            try:
                deploy_time_dt = datetime.fromtimestamp(local_storage["deploy_time"])
            except (TypeError, ValueError):
                logger.error("Invalid deploy_time in local_storage for HTML report. Using default.")
                deploy_time_dt = datetime(2000,1,1) # Fallback
        else:
            # This should ideally not happen if __init__ or _collect_all_statistics ran
            logger.warning("deploy_time not found in local_storage for HTML report. Using default.")
            deploy_time_dt = datetime(2000, 1, 1) # Fallback

        tab_list_html = []
        tab_content_html_list = []

        for period_key, period_delta, period_display_name in self.stat_period: # Use self.stat_period as defined by _collect_all_statistics
            tab_list_html.append(
                f'<button class="tab-link" onclick="showTab(event, \'{period_key}\')">{period_display_name}</button>'
            )

            current_period_stats = stat_collection.get(period_key, {}) 

            if period_key == "all_time":
                start_time_dt_for_period = deploy_time_dt
            else:
                # Ensure period_delta is a timedelta object
                if isinstance(period_delta, timedelta):
                    start_time_dt_for_period = now - period_delta
                else: # Fallback if period_delta is not as expected (e.g. from old self.stat_period)
                    logger.warning(f"period_delta for {period_key} is not a timedelta. Using 'now'. Type: {type(period_delta)}")
                    start_time_dt_for_period = now


            html_content_for_tab = f"""
            <div id="{period_key}" class="tab-content">
                <p class="info-item">
                    <strong>统计时段: </strong>
                    {start_time_dt_for_period.strftime("%Y-%m-%d %H:%M:%S")} ~ {now.strftime("%Y-%m-%d %H:%M:%S")}
                </p>
                <p class="info-item"><strong>总在线时间: </strong>{_format_online_time(current_period_stats.get(ONLINE_TIME, 0))}</p>
                <p class="info-item"><strong>总消息数: </strong>{current_period_stats.get(TOTAL_MSG_CNT, 0)}</p>
                <p class="info-item"><strong>总请求数: </strong>{current_period_stats.get(TOTAL_REQ_CNT, 0)}</p>
                <p class="info-item"><strong>总花费: </strong>{current_period_stats.get(TOTAL_COST, 0.0):.4f} ¥</p>
            """

            html_content_for_tab += "<h2>按模型分类统计</h2><table><thead><tr><th>模型名称</th><th>调用次数</th><th>输入Token</th><th>输出Token</th><th>Token总量</th><th>累计花费</th></tr></thead><tbody>"
            req_cnt_by_model = current_period_stats.get(REQ_CNT_BY_MODEL, {})
            in_tok_by_model = current_period_stats.get(IN_TOK_BY_MODEL, defaultdict(int))
            out_tok_by_model = current_period_stats.get(OUT_TOK_BY_MODEL, defaultdict(int))
            total_tok_by_model = current_period_stats.get(TOTAL_TOK_BY_MODEL, defaultdict(int))
            cost_by_model = current_period_stats.get(COST_BY_MODEL, defaultdict(float))
            if req_cnt_by_model:
                for model_name, count in sorted(req_cnt_by_model.items()):
                    html_content_for_tab += (
                        f"<tr>"
                        f"<td>{model_name}</td>"
                        f"<td>{count}</td>"
                        f"<td>{in_tok_by_model[model_name]}</td>"
                        f"<td>{out_tok_by_model[model_name]}</td>"
                        f"<td>{total_tok_by_model[model_name]}</td>"
                        f"<td>{cost_by_model[model_name]:.4f} ¥</td>"
                        f"</tr>"
                    )
            else:
                html_content_for_tab += "<tr><td colspan='6'>无数据</td></tr>"
            html_content_for_tab += "</tbody></table>"

            html_content_for_tab += "<h2>按请求类型分类统计</h2><table><thead><tr><th>请求类型</th><th>调用次数</th><th>输入Token</th><th>输出Token</th><th>Token总量</th><th>累计花费</th></tr></thead><tbody>"
            req_cnt_by_type = current_period_stats.get(REQ_CNT_BY_TYPE, {})
            in_tok_by_type = current_period_stats.get(IN_TOK_BY_TYPE, defaultdict(int))
            out_tok_by_type = current_period_stats.get(OUT_TOK_BY_TYPE, defaultdict(int))
            total_tok_by_type = current_period_stats.get(TOTAL_TOK_BY_TYPE, defaultdict(int))
            cost_by_type = current_period_stats.get(COST_BY_TYPE, defaultdict(float))
            if req_cnt_by_type:
                for req_type, count in sorted(req_cnt_by_type.items()):
                    html_content_for_tab += (
                        f"<tr>"
                        f"<td>{req_type}</td>"
                        f"<td>{count}</td>"
                        f"<td>{in_tok_by_type[req_type]}</td>"
                        f"<td>{out_tok_by_type[req_type]}</td>"
                        f"<td>{total_tok_by_type[req_type]}</td>"
                        f"<td>{cost_by_type[req_type]:.4f} ¥</td>"
                        f"</tr>"
                    )
            else:
                html_content_for_tab += "<tr><td colspan='6'>无数据</td></tr>"
            html_content_for_tab += "</tbody></table>"

            html_content_for_tab += "<h2>按用户分类统计</h2><table><thead><tr><th>用户ID/名称</th><th>调用次数</th><th>输入Token</th><th>输出Token</th><th>Token总量</th><th>累计花费</th></tr></thead><tbody>"
            req_cnt_by_user = current_period_stats.get(REQ_CNT_BY_USER, {})
            in_tok_by_user = current_period_stats.get(IN_TOK_BY_USER, defaultdict(int))
            out_tok_by_user = current_period_stats.get(OUT_TOK_BY_USER, defaultdict(int))
            total_tok_by_user = current_period_stats.get(TOTAL_TOK_BY_USER, defaultdict(int))
            cost_by_user = current_period_stats.get(COST_BY_USER, defaultdict(float))
            if req_cnt_by_user:
                for user_id, count in sorted(req_cnt_by_user.items()):
                    user_display_name = self.name_mapping.get(user_id, (user_id, None))[0] 
                    html_content_for_tab += (
                        f"<tr>"
                        f"<td>{user_display_name}</td>"
                        f"<td>{count}</td>"
                        f"<td>{in_tok_by_user[user_id]}</td>"
                        f"<td>{out_tok_by_user[user_id]}</td>"
                        f"<td>{total_tok_by_user[user_id]}</td>"
                        f"<td>{cost_by_user[user_id]:.4f} ¥</td>"
                        f"</tr>"
                    )
            else:
                html_content_for_tab += "<tr><td colspan='6'>无数据</td></tr>"
            html_content_for_tab += "</tbody></table>"

            html_content_for_tab += "<h2>聊天消息统计</h2><table><thead><tr><th>联系人/群组名称</th><th>消息数量</th></tr></thead><tbody>"
            msg_cnt_by_chat = current_period_stats.get(MSG_CNT_BY_CHAT, {})
            if msg_cnt_by_chat:
                for chat_id, count in sorted(msg_cnt_by_chat.items()):
                    chat_name_display = self.name_mapping.get(chat_id, (f"未知/归档聊天 ({chat_id})", None))[0]
                    html_content_for_tab += f"<tr><td>{chat_name_display}</td><td>{count}</td></tr>"
            else:
                html_content_for_tab += "<tr><td colspan='2'>无数据</td></tr>"
            html_content_for_tab += "</tbody></table></div>" 

            tab_content_html_list.append(html_content_for_tab)


        html_template = (
            """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MaiBot运行统计报告</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f6;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 20px auto;
            background-color: #fff;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        h1 {
            text-align: center;
            font-size: 2em;
        }
        h2 {
            font-size: 1.5em;
            margin-top: 30px;
        }
        p {
            margin-bottom: 10px;
        }
        .info-item {
            background-color: #ecf0f1;
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 0.95em;
        }
        .info-item strong {
            color: #2980b9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
            word-break: break-all; 
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.8em;
            color: #7f8c8d;
        }
        .tabs {
            overflow: hidden;
            background: #ecf0f1;
            display: flex; 
            flex-wrap: wrap; 
            margin-bottom: -1px; 
        }
        .tabs button {
            background: inherit; 
            border: 1px solid #ccc; 
            border-bottom: none; 
            outline: none;
            padding: 14px 16px; 
            cursor: pointer;
            transition: 0.3s; 
            font-size: 16px;
            margin-right: 2px; 
            border-radius: 4px 4px 0 0; 
        }
        .tabs button:hover {
            background-color: #d4dbdc;
        }
        .tabs button.active {
            background-color: #fff; 
            border-color: #ccc;
            border-bottom: 1px solid #fff; 
            position: relative;
            z-index: 1;
        }
        .tab-content {
            display: none;
            padding: 20px;
            background-color: #fff;
            border: 1px solid #ccc;
            border-top: none; 
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
"""
            + f"""
    <div class="container">
        <h1>MaiBot运行统计报告</h1>
        <p class="info-item"><strong>统计截止时间:</strong> {now.strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="tabs">
            {"".join(tab_list_html)}
        </div>

        {"".join(tab_content_html_list)}
        
        <div class="footer">
            <p>Generated by MaiBot Statistics Module</p>
        </div>
    </div>
"""
            + """
<script>
    let i, tab_content, tab_links;
    tab_content = document.getElementsByClassName("tab-content");
    tab_links = document.getElementsByClassName("tab-link");
    
    if (tab_content.length > 0 && tab_links.length > 0) { 
        tab_content[0].classList.add("active");
        tab_links[0].classList.add("active");
    }

    function showTab(evt, tabName) {
        for (i = 0; i < tab_content.length; i++) {
            tab_content[i].classList.remove("active");
        }
        for (i = 0; i < tab_links.length; i++) {
            tab_links[i].classList.remove("active");
        }
        const currentTabContent = document.getElementById(tabName);
        if (currentTabContent) { 
             currentTabContent.classList.add("active");
        }
        if (evt.currentTarget) { 
            evt.currentTarget.classList.add("active");
        }
    }
</script>
</body>
</html>
        """
        )

        try:
            with open(self.record_file_path, "w", encoding="utf-8") as f:
                f.write(html_template)
            logger.info(f"统计报告已生成: {self.record_file_path}")
        except IOError as e:
            logger.error(f"无法写入统计报告文件 {self.record_file_path}: {e}")
