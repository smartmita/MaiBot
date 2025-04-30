from src.common.logger_manager import get_logger
from ..chat.chat_stream import ChatStream
import math
from bson.decimal128 import Decimal128
from .person_info import person_info_manager
import time
import random
from typing import List, Dict, Any, Optional, Tuple # 确保导入了 List, Dict, Optional, Tuple

logger = get_logger("relation")


class RelationshipManager:
    def __init__(self):
        self.positive_feedback_value = 0  # 正反馈系统
        self.gain_coefficient = [1.0, 1.0, 1.1, 1.2, 1.4, 1.7, 1.9, 2.0]
        self._mood_manager = None

    @property
    def mood_manager(self):
        if self._mood_manager is None:
            from ..moods.moods import MoodManager  # 延迟导入
            self._mood_manager = MoodManager.get_instance()
        return self._mood_manager

    def positive_feedback_sys(self, label: str, stance: str):
        """正反馈系统，通过正反馈系数增益情绪变化，根据情绪再影响关系变更"""
        positive_list = ["开心", "惊讶", "害羞"]
        negative_list = ["愤怒", "悲伤", "恐惧", "厌恶"]

        if label in positive_list:
            if 7 > self.positive_feedback_value >= 0:
                self.positive_feedback_value += 1
            elif self.positive_feedback_value < 0:
                self.positive_feedback_value = 0
        elif label in negative_list:
            if -7 < self.positive_feedback_value <= 0:
                self.positive_feedback_value -= 1
            elif self.positive_feedback_value > 0:
                self.positive_feedback_value = 0

        if abs(self.positive_feedback_value) > 1:
            logger.info(f"触发mood变更增益，当前增益系数：{self.gain_coefficient[abs(self.positive_feedback_value)]}")

    def mood_feedback(self, value):
        """情绪反馈"""
        mood_manager = self.mood_manager
        mood_gain = mood_manager.get_current_mood().valence ** 2 * math.copysign(
            1, value * mood_manager.get_current_mood().valence
        )
        value += value * mood_gain
        logger.info(f"当前relationship增益系数：{mood_gain:.3f}")
        return value

    def feedback_to_mood(self, mood_value):
        """对情绪的反馈"""
        coefficient = self.gain_coefficient[abs(self.positive_feedback_value)]
        if mood_value > 0 and self.positive_feedback_value > 0 or mood_value < 0 and self.positive_feedback_value < 0:
            return mood_value * coefficient
        else:
            return mood_value / coefficient

    @staticmethod
    async def is_known_some_one(platform, user_id):
        """判断是否认识某人"""
        is_known = person_info_manager.is_person_known(platform, user_id)
        return is_known

    @staticmethod
    async def is_qved_name(platform, user_id):
        """判断是否已经命名"""
        person_id = person_info_manager.get_person_id(platform, user_id)
        # 优化：直接检查 person_name 字段是否存在且不为 None 或空字符串
        person_name = await person_info_manager.get_value(person_id, "person_name")
        return bool(person_name) # 如果 person_name 非空则返回 True

    @staticmethod
    async def get_person_name(platform: str, user_id: str) -> Optional[str]:
        """获取单个用户的 person_name"""
        person_id = person_info_manager.get_person_id(platform, str(user_id)) # 确保 user_id 是字符串
        return await person_info_manager.get_value(person_id, "person_name")

    # --- [新增] 批量获取用户名称 ---
    @staticmethod
    async def get_person_names_batch(platform: str, user_ids: List[str]) -> Dict[str, str]:
        """
        批量获取多个用户的 person_name。

        Args:
            platform (str): 平台名称。
            user_ids (List[str]): 用户 ID 列表。

        Returns:
            Dict[str, str]: 映射 {user_id: person_name}，只包含成功获取到名称的用户。
        """
        if not user_ids:
            return {}

        person_ids = [person_info_manager.get_person_id(platform, str(uid)) for uid in user_ids] # 确保 uid 是字符串
        names_map = {}
        try:
            # 使用 $in 操作符批量查询
            cursor = person_info_manager.collection.find(
                {"person_id": {"$in": person_ids}},
                {"_id": 0, "person_id": 1, "person_name": 1} # 只查询需要的字段
            )
            async for doc in cursor:
                # 从 person_id 反向推导出原始 user_id
                # 注意：这依赖于 get_person_id 的实现方式，假设它是 platform_userid 格式
                original_user_id = doc.get("person_id", "").split("_", 1)[-1]
                person_name = doc.get("person_name")
                if original_user_id and person_name:
                    names_map[original_user_id] = person_name
            logger.debug(f"Batch get person names for {len(user_ids)} users, found {len(names_map)} names.")
        except Exception as e:
            logger.error(f"Error during batch get person names: {e}", exc_info=True)
        return names_map
    # --- 结束新增 ---

    # --- [新增] 批量获取用户群组绰号 ---
    @staticmethod
    async def get_users_group_nicknames(platform: str, user_ids: List[str], group_id: str) -> Dict[str, List[Dict[str, int]]]:
        """
        批量获取多个用户在指定群组的绰号信息。

        Args:
            platform (str): 平台名称。
            user_ids (List[str]): 用户 ID 列表。
            group_id (str): 群组 ID。

        Returns:
            Dict[str, List[Dict[str, int]]]: 映射 {person_name: [{"绰号A": 次数}, ...]}
                                            只包含成功获取到绰号信息的用户。
                                            键是用户的 person_name。
        """
        if not user_ids or not group_id:
            return {}

        person_ids = [person_info_manager.get_person_id(platform, str(uid)) for uid in user_ids]
        nicknames_data = {}
        group_id_str = str(group_id) # 确保 group_id 是字符串

        try:
            # 查询包含目标 person_id 且 group_nickname 字段存在的文档
            cursor = person_info_manager.collection.find(
                {
                    "person_id": {"$in": person_ids},
                    "group_nickname": {"$elemMatch": {group_id_str: {"$exists": True}}} # 确保该群组的条目存在
                },
                {"_id": 0, "person_id": 1, "person_name": 1, "group_nickname": 1} # 查询所需字段
            )

            async for doc in cursor:
                person_name = doc.get("person_name")
                if not person_name: # 如果没有 person_name，则跳过此用户
                    continue

                group_nicknames_list = doc.get("group_nickname", [])
                user_group_nicknames = []
                # 遍历 group_nickname 列表，找到对应 group_id 的条目
                for group_entry in group_nicknames_list:
                    if group_id_str in group_entry and isinstance(group_entry[group_id_str], list):
                        # 提取该群组的绰号列表 [{"绰号": 次数}, ...]
                        user_group_nicknames = group_entry[group_id_str]
                        break # 找到后即可退出内层循环

                if user_group_nicknames: # 确保列表非空
                    # 过滤掉格式不正确的条目
                    valid_nicknames = []
                    for item in user_group_nicknames:
                        if isinstance(item, dict) and len(item) == 1:
                            key, value = list(item.items())[0]
                            if isinstance(key, str) and isinstance(value, int):
                                valid_nicknames.append(item)
                            else:
                                logger.warning(f"Invalid nickname format in DB for user {person_name}, group {group_id_str}: {item}")
                        else:
                            logger.warning(f"Invalid nickname entry format in DB for user {person_name}, group {group_id_str}: {item}")

                    if valid_nicknames:
                        nicknames_data[person_name] = valid_nicknames # 使用 person_name 作为 key

            logger.debug(f"Batch get group nicknames for {len(user_ids)} users in group {group_id_str}, found data for {len(nicknames_data)} users.")

        except Exception as e:
            logger.error(f"Error during batch get group nicknames: {e}", exc_info=True)

        return nicknames_data
    # --- 结束新增 ---


    @staticmethod
    async def first_knowing_some_one(platform, user_id, user_nickname, user_cardname, user_avatar):
        """初次认识某人或更新信息"""
        person_id = person_info_manager.get_person_id(platform, user_id)
        # 首次认识时，除了更新 nickname，也应该设置初始关系值等
        initial_data = {
            "platform": platform,
            "user_id": user_id,
            "nickname": user_nickname,
            "konw_time": int(time.time()),
            "relationship_value": 0.0, # 设置初始关系值为 0
            "msg_interval": -1, # 初始消息间隔设为 -1 或其他标记
            "msg_interval_list": [],
            "group_nickname": [] # 初始化为空列表
        }
        # 使用 update_one 并结合 $setOnInsert 来避免覆盖已有数据
        await person_info_manager.collection.update_one(
            {"person_id": person_id},
            {
                "$set": {"nickname": user_nickname}, # 总是更新 nickname
                "$setOnInsert": initial_data # 仅在插入新文档时设置这些初始值
            },
            upsert=True
        )
        # 尝试获取或生成 person_name
        await person_info_manager.qv_person_name(person_id, user_nickname, user_cardname, user_avatar)


    async def calculate_update_relationship_value(self, chat_stream: ChatStream, label: str, stance: str) -> tuple:
        """计算并变更关系值"""
        stancedict = {"支持": 0, "中立": 1, "反对": 2}
        valuedict = {
            "开心": 1.5, "愤怒": -2.0, "悲伤": -0.5, "惊讶": 0.6, "害羞": 2.0,
            "平静": 0.3, "恐惧": -1.5, "厌恶": -1.0, "困惑": 0.5,
        }

        person_id = person_info_manager.get_person_id(chat_stream.user_info.platform, chat_stream.user_info.user_id)
        data = { # 这个 data 似乎是用于 setOnInsert 的，应该在 first_knowing 时处理
            "platform": chat_stream.user_info.platform,
            "user_id": chat_stream.user_info.user_id,
            "nickname": chat_stream.user_info.user_nickname,
            "konw_time": int(time.time()),
        }
        old_value = await person_info_manager.get_value(person_id, "relationship_value")
        old_value = self.ensure_float(old_value, person_id) # 确保是 float

        # 限制旧值范围
        old_value = max(min(old_value, 1000), -1000)

        value_change = 0.0 # 初始化变化量
        base_value = valuedict.get(label, 0.0) # 获取基础情绪值

        # 应用立场影响和关系值衰减/增强逻辑
        if base_value > 0 and stancedict.get(stance, 1) != 2: # 正面情绪且非反对
            value_change = base_value * math.cos(math.pi * old_value / 2000)
            if old_value > 500: # 高关系值增长减缓
                rdict = await person_info_manager.get_specific_value_list("relationship_value", lambda x: x > 700)
                high_value_count = len(rdict)
                # 注意：这里的减缓因子可能需要调整
                value_change *= 3 / (high_value_count + (2 if old_value > 700 else 3))
        elif base_value < 0 and stancedict.get(stance, 1) != 0: # 负面情绪且非支持
            # 关系好时负面影响更大，关系差时负面影响减弱
            value_change = base_value * math.exp(old_value / 2000) if old_value >= 0 else base_value * math.cos(math.pi * old_value / 2000)
        # else: 立场冲突或情绪平静，基础变化为 0

        # 应用正反馈系统和情绪反馈
        self.positive_feedback_sys(label, stance)
        value_change = self.mood_feedback(value_change) # 应用当前情绪对关系变化的影响
        value_change = self.feedback_to_mood(value_change) # 应用连续反馈对关系变化的影响

        new_value = old_value + value_change
        # 再次限制新值范围
        new_value = max(min(new_value, 1000), -1000)
        actual_change = new_value - old_value # 记录实际变化量

        level_num = self.calculate_level_num(new_value)
        relationship_level = ["厌恶", "冷漠", "一般", "友好", "喜欢", "暧昧"]
        logger.info(
            f"用户: {chat_stream.user_info.user_nickname} "
            f"当前关系: {relationship_level[level_num]}, "
            f"关系值: {old_value:.2f}, "
            f"立场情感: {stance}-{label}, "
            f"变更: {actual_change:+.5f}, "
            f"新值: {new_value:.2f}"
        )

        # 更新数据库，只更新 relationship_value
        await person_info_manager.update_one_field(person_id, "relationship_value", new_value)

        return chat_stream.user_info.user_nickname, actual_change, relationship_level[level_num]


    async def build_relationship_info(self, person, is_id: bool = False) -> str:
        """构建用于 Prompt 的关系信息字符串"""
        if is_id:
            person_id = person
            # 如果只有 person_id，需要反查 platform 和 user_id 来获取 person_name
            # 这依赖于 person_id 的格式，假设是 platform_userid
            try:
                platform, user_id_str = person_id.split("_", 1)
                person_name = await self.get_person_name(platform, user_id_str)
            except ValueError:
                logger.warning(f"Invalid person_id format for prompt building: {person_id}")
                person_name = None
        else:
            platform, user_id, _ = person # 解包元组
            person_id = person_info_manager.get_person_id(platform, user_id)
            person_name = await self.get_person_name(platform, user_id)

        if not person_name:
            person_name = f"用户({person_id})" # 回退显示 ID

        relationship_value = await person_info_manager.get_value(person_id, "relationship_value")
        relationship_value = self.ensure_float(relationship_value, person_id) # 确保是 float
        level_num = self.calculate_level_num(relationship_value)

        # 定义关系等级和对应的行为描述
        relationship_levels = ["厌恶", "冷漠以对", "认识", "友好对待", "喜欢", "暧昧"]
        relation_prompt_list = ["忽视的回应", "冷淡回复", "保持理性", "愿意回复", "积极回复", "友善和包容的回复"]

        # 根据等级和随机性决定是否输出及输出内容
        if level_num == 2: # "一般"关系不特别提示
            return ""
        elif level_num in [0, 5] or random.random() < 0.6: # 极好/极差 或 60% 概率
            # 修正索引，确保在列表范围内
            level_idx = max(0, min(level_num, len(relationship_levels) - 1))
            prompt_idx = max(0, min(level_num, len(relation_prompt_list) - 1))
            return f"你{relationship_levels[level_idx]}{person_name}，打算{relation_prompt_list[prompt_idx]}。\n"
        else:
            return ""

    @staticmethod
    def calculate_level_num(relationship_value) -> int:
        """关系等级计算"""
        # 确保 value 是 float
        try:
            value = float(relationship_value.to_decimal() if isinstance(relationship_value, Decimal128) else relationship_value)
        except (ValueError, TypeError, AttributeError):
            value = 0.0 # 转换失败默认为 0

        # 阈值判断
        if value < -227: return 0
        elif value < -73: return 1
        elif value < 227: return 2
        elif value < 587: return 3
        elif value < 900: return 4
        else: return 5 # >= 900

    @staticmethod
    def ensure_float(value, person_id):
        """确保返回浮点数，转换失败返回0.0"""
        if isinstance(value, (float, int)): # 直接处理 float 和 int
            return float(value)
        try:
            # 尝试处理 Decimal128 或其他可转换为 float 的类型
            return float(value.to_decimal() if isinstance(value, Decimal128) else value)
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"[关系管理] {person_id} 值转换失败（原始值：{value}），已重置为0")
            # 在转换失败时，尝试在数据库中将该字段重置为 0.0
            try:
                person_info_manager.update_one_field(person_id, "relationship_value", 0.0)
            except Exception as db_err:
                logger.error(f"Failed to reset relationship_value for {person_id} in DB: {db_err}")
            return 0.0


relationship_manager = RelationshipManager()
