from src.common.logger_manager import get_logger
from ..message_receive.chat_stream import ChatStream
import math
from bson.decimal128 import Decimal128
from .person_info import person_info_manager
import time
import random
from typing import List, Dict, Any
from ...common.database import db
from maim_message import UserInfo

from ...manager.mood_manager import mood_manager

# import re
# import traceback


logger = get_logger("relation")


class RelationshipManager:
    def __init__(self):
        self.positive_feedback_value = 0  # 正反馈系统
        self.gain_coefficient = [1.0, 1.0, 1.1, 1.2, 1.4, 1.7, 1.9, 2.0]
        self._mood_manager = None

    @property
    def mood_manager(self):
        if self._mood_manager is None:
            self._mood_manager = mood_manager
        return self._mood_manager

    def positive_feedback_sys(self, label: str, stance: str):
        """正反馈系统，通过正反馈系数增益情绪变化，根据情绪再影响关系变更"""

        positive_list = [
            "开心",
            "惊讶",
            "害羞",
        ]

        negative_list = [
            "愤怒",
            "悲伤",
            "恐惧",
            "厌恶",
        ]

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
        mood_gain = mood_manager.current_mood.valence**2 * math.copysign(1, value * mood_manager.current_mood.valence)
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
    async def is_known_some_one(platform, user_id, nickname):
        """判断是否认识某人"""
        is_known = await person_info_manager.is_person_known(platform, user_id, nickname)
        return is_known

    @staticmethod
    async def get_actual_nicknames_batch(platform: str, user_ids: List[str]) -> Dict[str, str]:
        """
        批量获取多个用户的实际平台昵称 (nickname)。

        Args:
            platform (str): 平台名称。
            user_ids (List[str]): 用户 ID 列表 (期望是字符串形式的 UID)。

        Returns:
            Dict[str, str]: 映射 {user_id: actual_nickname}。
                            其中 user_id 是输入列表中的原始 ID 字符串。
        """
        if not user_ids:
            logger.debug("get_actual_nicknames_batch 调用时 user_ids 为空，返回空字典。")
            return {}

        person_id_to_original_uid_map: Dict[str, str] = {}
        person_ids_to_query: List[str] = []

        for uid in user_ids:
            uid_str = str(uid) # 确保输入的UID是字符串
            person_id = person_info_manager.get_person_id(platform, uid_str)
            person_ids_to_query.append(person_id)
            # 即使 person_id 可能因哈希碰撞（理论上极低概率）重复，
            # 对于给定的 (platform, uid_str) 组合，person_id 是确定的。
            # 我们用 person_id 作为查找键，并映射回原始的 uid_str。
            if person_id not in person_id_to_original_uid_map:
                 person_id_to_original_uid_map[person_id] = uid_str
            # 如果在输入 user_ids 列表中有重复的 uid，这里会用最后一个 uid_str 覆盖，
            # 但通常批量查询的 user_ids 应该是唯一的。

        actual_nicknames_map: Dict[str, str] = {}
        if not person_ids_to_query: # 如果处理后没有有效的 person_id 可供查询
            return {}

        try:
            cursor = db.person_info.find(
                {"person_id": {"$in": person_ids_to_query}},
                {"_id": 0, "person_id": 1, "nickname": 1}  # 只需要 person_id 和 nickname
            )

            async for doc in cursor:
                db_person_id = doc.get("person_id")
                actual_nickname = doc.get("nickname") # 这是用户的实际平台昵称

                # 通过 person_id 从映射中找回原始输入的 user_id
                original_input_uid = person_id_to_original_uid_map.get(db_person_id)

                if original_input_uid and actual_nickname is not None: # 确保昵称存在（可以是空字符串）
                    actual_nicknames_map[original_input_uid] = actual_nickname
                elif original_input_uid and actual_nickname is None:
                    logger.debug(f"用户 (UID: {original_input_uid}, PersonID: {db_person_id}) 在数据库中 'nickname' 字段为 null 或缺失。")
                # 如果 original_input_uid 为 None，意味着数据库返回了一个我们没有请求的 person_id，这不应发生。

            found_count = len(actual_nicknames_map)
            requested_count = len(set(uid_str for uid_str in user_ids)) # 计算去重后的请求数量
            logger.debug(f"批量获取 {requested_count} 个用户的实际昵称，成功从数据库匹配并返回 {found_count} 个。")

        except AttributeError as e:
            logger.error(f"批量获取实际昵称时访问数据库出错 (AttributeError): {e}。请检查common/database.py和集合名称。")
        except Exception as e:
            logger.error(f"批量获取用户实际昵称时发生未知错误: {e}", exc_info=True)
        
        return actual_nicknames_map

    @staticmethod
    async def get_users_group_nicknames(
        platform: str, user_ids: List[str], group_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个用户在指定群组的绰号信息和 user_id。

        Args:
            platform (str): 平台名称。
            user_ids (List[str]): 用户 ID 列表。
            group_id (str): 群组 ID。

        Returns:
            Dict[str, Dict[str, Any]]: 映射 {nickname: {"user_id": "uid", "nicknames": [{"绰号A": 次数}, ...]} }
                                       其中 nickname 是用户在平台上的实际昵称。
        """
        if not user_ids or not group_id:
            return {}

        # 创建 person_id 到原始 user_id (字符串形式) 的映射
        person_ids_map = {person_info_manager.get_person_id(platform, str(uid)): str(uid) for uid in user_ids}
        person_ids_to_query = list(person_ids_map.keys())
        
        nicknames_data_by_actual_nickname = {} # 存储结果的字典，键为用户的实际昵称
        group_id_str = str(group_id)

        try:
            # 从数据库查询时，确保包含 nickname 字段
            cursor = db.person_info.find(
                {"person_id": {"$in": person_ids_to_query}},
                {
                    "_id": 0,
                    "person_id": 1,
                    "nickname": 1,  # 获取用户的实际平台昵称
                    "group_nicknames": 1,
                    "user_id": 1
                },
            )

            async for doc in cursor: # 使用 async for
                actual_nickname_from_db = doc.get("nickname") # 获取用户的实际平台昵称

                user_id_val = doc.get("user_id")
                original_user_id_from_doc = None # 用于存储从文档中解析出的 user_id

                # 解析文档中的 user_id (这部分逻辑保持不变)
                if isinstance(user_id_val, (int, float)):
                    original_user_id_from_doc = str(user_id_val)
                elif isinstance(user_id_val, str):
                    if "_" in user_id_val: # 假设 user_id 可能存储为 "platform_actualuid" 格式
                        original_user_id_from_doc = user_id_val.split("_", 1)[-1]
                    else:
                        original_user_id_from_doc = user_id_val

                current_person_id_from_doc = doc.get("person_id")
                # 优先使用请求时映射的 user_id，如果找不到，则使用从文档中解析的 user_id
                # 这确保了即使数据库中的 user_id 字段格式多样，也能尽量关联回请求时的 uid
                actual_user_id_for_output = person_ids_map.get(current_person_id_from_doc, original_user_id_from_doc)

                # 关键检查：确保我们有有效的实际昵称 (actual_nickname_from_db) 和用户ID (actual_user_id_for_output)
                if not actual_nickname_from_db or not actual_user_id_for_output:
                    logger.warning(
                        f"跳过处理，因为从数据库获取的实际昵称 ('{actual_nickname_from_db}') 或 "
                        f"解析的用户ID ('{actual_user_id_for_output}') 无效。 "
                        f"文档 Person ID: {current_person_id_from_doc}"
                    )
                    continue

                group_nicknames_list_from_doc = doc.get("group_nicknames", [])
                target_group_nicknames_raw = [] # 存储目标群组的原始绰号列表

                # 提取特定群组的绰号 (这部分逻辑保持不变)
                for group_entry in group_nicknames_list_from_doc:
                    if isinstance(group_entry, dict) and group_entry.get("group_id") == group_id_str:
                        nicknames_raw_for_group = group_entry.get("nicknames", [])
                        if isinstance(nicknames_raw_for_group, list):
                            target_group_nicknames_raw = nicknames_raw_for_group
                        break

                valid_nicknames_formatted = [] # 存储格式化且有效的绰号
                if target_group_nicknames_raw:
                    for item in target_group_nicknames_raw:
                        if (
                            isinstance(item, dict)
                            and isinstance(item.get("name"), str)
                            and isinstance(item.get("count"), int)
                            and item["count"] > 0 # 确保绰号计数有效
                        ):
                            valid_nicknames_formatted.append({item["name"]: item["count"]})
                        else:
                            logger.warning(
                                f"数据库中用户 (实际昵称: {actual_nickname_from_db}, UID: {actual_user_id_for_output}) "
                                f"在群组 {group_id_str} 中的绰号格式无效或 count <= 0: {item}"
                            )

                # 如果找到了有效的绰号，则以用户的实际昵称 (actual_nickname_from_db) 为键存储
                if valid_nicknames_formatted:
                    nicknames_data_by_actual_nickname[actual_nickname_from_db] = {
                        "user_id": actual_user_id_for_output,
                        "nicknames": valid_nicknames_formatted
                    }

            logger.debug(
                f"批量获取群组 {group_id_str} 中 {len(user_ids)} 个用户的绰号和UID (以实际昵称为键)，"
                f"找到 {len(nicknames_data_by_actual_nickname)} 个用户的数据。"
            )

        except AttributeError as e: # 捕获数据库对象属性错误
            logger.error(f"访问数据库时出错: {e}。请检查 common/database.py 和集合名称 'person_info'。")
        except Exception as e: # 捕获其他潜在错误
            logger.error(f"批量获取群组绰号时发生未知错误: {e}", exc_info=True)

        return nicknames_data_by_actual_nickname


    @staticmethod
    async def first_knowing_some_one(
        platform: str, user_id: str, user_nickname: str, user_cardname: str, user_avatar: str
    ):
        """判断是否认识某人"""
        person_id = person_info_manager.get_person_id(platform, user_id)
        data = {
            "platform": platform,
            "user_id": user_id,
            "nickname": user_nickname,
            "konw_time": int(time.time()),
        }
        await person_info_manager.update_one_field(
            person_id=person_id, field_name="nickname", value=user_nickname, data=data
        )

    async def calculate_update_relationship_value(self, user_info: UserInfo, platform: str, label: str, stance: str):
        """计算并变更关系值
        新的关系值变更计算方式：
            将关系值限定在-1000到1000
            对于关系值的变更，期望：
                1.向两端逼近时会逐渐减缓
                2.关系越差，改善越难，关系越好，恶化越容易
                3.人维护关系的精力往往有限，所以当高关系值用户越多，对于中高关系值用户增长越慢
                4.连续正面或负面情感会正反馈

        返回：
            用户昵称，变更值，变更后关系等级

        """
        stancedict = {
            "支持": 0,
            "中立": 1,
            "反对": 2,
        }

        valuedict = {
            "开心": 1.5,
            "愤怒": -2.0,
            "悲伤": -0.5,
            "惊讶": 0.6,
            "害羞": 2.0,
            "平静": 0.3,
            "恐惧": -1.5,
            "厌恶": -1.0,
            "困惑": 0.5,
        }

        person_id = person_info_manager.get_person_id(platform, user_info.user_id)
        data = {
            "platform": platform,
            "user_id": user_info.user_id,
            "nickname": user_info.user_nickname,
            "konw_time": int(time.time()),
        }
        old_value = await person_info_manager.get_value(person_id, "relationship_value")
        old_value = self.ensure_float(old_value, person_id)

        if old_value > 1000:
            old_value = 1000
        elif old_value < -1000:
            old_value = -1000

        value = valuedict[label]
        if old_value >= 0:
            if valuedict[label] >= 0 and stancedict[stance] != 2:
                value = value * math.cos(math.pi * old_value / 2000)
                if old_value > 500:
                    rdict = await person_info_manager.get_specific_value_list("relationship_value", lambda x: x > 700)
                    high_value_count = len(rdict)
                    if old_value > 700:
                        value *= 3 / (high_value_count + 2)  # 排除自己
                    else:
                        value *= 3 / (high_value_count + 3)
            elif valuedict[label] < 0 and stancedict[stance] != 0:
                value = value * math.exp(old_value / 2000)
            else:
                value = 0
        elif old_value < 0:
            if valuedict[label] >= 0 and stancedict[stance] != 2:
                value = value * math.exp(old_value / 2000)
            elif valuedict[label] < 0 and stancedict[stance] != 0:
                value = value * math.cos(math.pi * old_value / 2000)
            else:
                value = 0

        self.positive_feedback_sys(label, stance)
        value = self.mood_feedback(value)

        level_num = self.calculate_level_num(old_value + value)
        relationship_level = ["厌恶", "冷漠", "一般", "友好", "喜欢", "依赖"]
        logger.info(
            f"用户: {user_info.user_nickname}"
            f"当前关系: {relationship_level[level_num]}, "
            f"关系值: {old_value:.2f}, "
            f"当前立场情感: {stance}-{label}, "
            f"变更: {value:+.5f}"
        )

        await person_info_manager.update_one_field(person_id, "relationship_value", old_value + value, data)

    async def calculate_update_relationship_value_with_reason(
        self, chat_stream: ChatStream, label: str, stance: str, reason: str
    ) -> tuple:
        """计算并变更关系值
        新的关系值变更计算方式：
            将关系值限定在-1000到1000
            对于关系值的变更，期望：
                1.向两端逼近时会逐渐减缓
                2.关系越差，改善越难，关系越好，恶化越容易
                3.人维护关系的精力往往有限，所以当高关系值用户越多，对于中高关系值用户增长越慢
                4.连续正面或负面情感会正反馈

        返回：
            用户昵称，变更值，变更后关系等级

        """
        stancedict = {
            "支持": 0,
            "中立": 1,
            "反对": 2,
        }

        valuedict = {
            "开心": 1.5,
            "愤怒": -2.0,
            "悲伤": -0.5,
            "惊讶": 0.6,
            "害羞": 2.0,
            "平静": 0.3,
            "恐惧": -1.5,
            "厌恶": -1.0,
            "困惑": 0.5,
        }

        person_id = person_info_manager.get_person_id(chat_stream.user_info.platform, chat_stream.user_info.user_id)
        data = {
            "platform": chat_stream.user_info.platform,
            "user_id": chat_stream.user_info.user_id,
            "nickname": chat_stream.user_info.user_nickname,
            "konw_time": int(time.time()),
        }
        old_value = await person_info_manager.get_value(person_id, "relationship_value")
        old_value = self.ensure_float(old_value, person_id)

        if old_value > 1000:
            old_value = 1000
        elif old_value < -1000:
            old_value = -1000

        value = valuedict[label]
        if old_value >= 0:
            if valuedict[label] >= 0 and stancedict[stance] != 2:
                value = value * math.cos(math.pi * old_value / 2000)
                if old_value > 500:
                    rdict = await person_info_manager.get_specific_value_list("relationship_value", lambda x: x > 700)
                    high_value_count = len(rdict)
                    if old_value > 700:
                        value *= 3 / (high_value_count + 2)  # 排除自己
                    else:
                        value *= 3 / (high_value_count + 3)
            elif valuedict[label] < 0 and stancedict[stance] != 0:
                value = value * math.exp(old_value / 2000)
            else:
                value = 0
        elif old_value < 0:
            if valuedict[label] >= 0 and stancedict[stance] != 2:
                value = value * math.exp(old_value / 2000)
            elif valuedict[label] < 0 and stancedict[stance] != 0:
                value = value * math.cos(math.pi * old_value / 2000)
            else:
                value = 0

        self.positive_feedback_sys(label, stance)
        value = self.mood_feedback(value)

        level_num = self.calculate_level_num(old_value + value)
        relationship_level = ["厌恶", "冷漠", "一般", "友好", "喜欢", "依赖"]
        logger.info(
            f"用户: {chat_stream.user_info.user_nickname}"
            f"当前关系: {relationship_level[level_num]}, "
            f"关系值: {old_value:.2f}, "
            f"当前立场情感: {stance}-{label}, "
            f"变更: {value:+.5f}"
        )

        await person_info_manager.update_one_field(person_id, "relationship_value", old_value + value, data)

        return chat_stream.user_info.user_nickname, value, relationship_level[level_num]

    async def build_relationship_info(self, person, is_id: bool = False) -> str:
        if is_id:
            person_id = person
        else:
            # print(f"person: {person}")
            person_id = person_info_manager.get_person_id(person[0], person[1])
        user_id = await person_info_manager.get_value(person_id, "user_id")
        relationship_value = await person_info_manager.get_value(person_id, "relationship_value")
        level_num = self.calculate_level_num(relationship_value)

        if level_num == 0 or level_num == 5:
            relationship_level = ["厌恶", "冷漠以对", "认识", "友好对待", "喜欢", "依赖"]
            relation_prompt2_list = [
                "忽视的回应",
                "冷淡回复",
                "保持理性",
                "愿意回复",
                "积极回复",
                "友善和包容的回复",
            ]
            return f"你{relationship_level[level_num]}{user_id}，打算{relation_prompt2_list[level_num]}。\n"
        elif level_num == 2:
            return ""
        else:
            if random.random() < 0.6:
                relationship_level = ["厌恶", "冷漠以对", "认识", "友好对待", "喜欢", "依赖"]
                relation_prompt2_list = [
                    "忽视的回应",
                    "冷淡回复",
                    "保持理性",
                    "愿意回复",
                    "积极回复",
                    "友善和包容的回复",
                ]
                return f"你{relationship_level[level_num]}{user_id}，打算{relation_prompt2_list[level_num]}。\n"
            else:
                return ""

    @staticmethod
    def calculate_level_num(relationship_value) -> int:
        """关系等级计算"""
        if -1000 <= relationship_value < -227:
            level_num = 0
        elif -227 <= relationship_value < -73:
            level_num = 1
        elif -73 <= relationship_value < 227:
            level_num = 2
        elif 227 <= relationship_value < 587:
            level_num = 3
        elif 587 <= relationship_value < 900:
            level_num = 4
        elif 900 <= relationship_value <= 1000:
            level_num = 5
        else:
            level_num = 5 if relationship_value > 1000 else 0
        return level_num

    @staticmethod
    def ensure_float(value, person_id):
        """确保返回浮点数，转换失败返回0.0"""
        if isinstance(value, float):
            return value
        try:
            return float(value.to_decimal() if isinstance(value, Decimal128) else value)
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"[关系管理] {person_id}值转换失败（原始值：{value}），已重置为0")
            return 0.0


relationship_manager = RelationshipManager()
