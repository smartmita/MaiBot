from pymongo.collection import Collection
from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from typing import Optional

# 日志记录器名称更新
logger = get_logger("sobriquet_db")


class SobriquetDB:  # 类名更新
    """
    处理与群组绰号相关的数据库操作 (MongoDB)。
    封装了对 'person_info' 集合的读写操作。
    """

    def __init__(self, person_info_collection: Optional[Collection]):
        """
        初始化 SobriquetDB 处理器。 # 名称更新

        Args:
            person_info_collection: MongoDB 'person_info' 集合对象。
                                    如果为 None，则数据库操作将被禁用。
        """
        if person_info_collection is None:
            # 日志消息中的类名更新
            logger.error("未提供 person_info 集合，SobriquetDB 操作将被禁用。")
            self.person_info_collection = None
        else:
            self.person_info_collection = person_info_collection
            logger.info("SobriquetDB 初始化成功。") # 名称更新

    def is_available(self) -> bool:
        """检查数据库集合是否可用。"""
        return self.person_info_collection is not None

    def upsert_person(self, person_id: str, user_id_int: int, platform: str):
        """
        确保数据库中存在指定 person_id 的文档 (Upsert)。
        如果文档不存在，则使用提供的用户信息创建它。

        Args:
            person_id: 要查找或创建的 person_id。
            user_id_int: 用户的整数 ID。
            platform: 平台名称。

        Returns:
            UpdateResult 或 None: MongoDB 更新操作的结果，如果数据库不可用则返回 None。

        Raises:
            DuplicateKeyError: 如果发生重复键错误 (理论上不应由 upsert 触发)。
            Exception: 其他数据库操作错误。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 upsert_person。")
            return None
        try:
            # 关键步骤：基于 person_id 执行 Upsert
            result = self.person_info_collection.update_one(
                {"person_id": person_id},
                {
                    "$setOnInsert": {
                        "person_id": person_id,
                        "user_id": user_id_int,
                        "platform": platform,
                        "group_nicknames": [],  # 初始化 group_nicknames 数组 (数据库字段名不变)
                    }
                },
                upsert=True,
            )
            if result.upserted_id:
                logger.debug(f"Upsert 创建了新的 person 文档: {person_id}")
            return result
        except DuplicateKeyError as dk_err:
            logger.error(
                f"数据库操作失败 (DuplicateKeyError): person_id {person_id}. 错误: {dk_err}. 这不应该发生，请检查 person_id 生成逻辑和数据库状态。"
            )
            raise
        except Exception as e:
            logger.exception(f"对 person_id {person_id} 执行 Upsert 时失败: {e}")
            raise

    def update_group_sobriquet_count(self, person_id: str, group_id_str: str, sobriquet: str): # 参数名 nickname -> sobriquet
        """
        尝试更新 person_id 文档中特定群组的绰号计数，或添加新条目。
        按顺序尝试：增加计数 -> 添加绰号 -> 添加群组。

        Args:
            person_id: 目标文档的 person_id。
            group_id_str: 目标群组的 ID (字符串)。
            sobriquet: 要更新或添加的绰号。 # 名称更新
        """
        if not self.is_available():
            # 日志消息更新
            logger.error("数据库集合不可用，无法执行 update_group_sobriquet_count。")
            return

        try:
            # 3a. 尝试增加现有群组中现有绰号的计数
            result_inc = self.person_info_collection.update_one(
                {
                    "person_id": person_id,
                    # 数据库字段名 "group_nicknames", "nicknames.name" 不变
                    "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet}},
                },
                # 数据库字段名 "group_nicknames.$[group].nicknames.$[nick].count" 不变
                {"$inc": {"group_nicknames.$[group].nicknames.$[nick].count": 1}},
                array_filters=[
                    {"group.group_id": group_id_str},
                    {"nick.name": sobriquet}, # 匹配值使用新的变量名 sobriquet
                ],
            )
            if result_inc.modified_count > 0:
                # logger.debug(f"成功增加 person_id {person_id} 在群组 {group_id_str} 中绰号 '{sobriquet}' 的计数。") # 日志消息更新
                return

            # 3b. 如果上一步未修改 (绰号不存在于该群组)，尝试将新绰号添加到现有群组
            result_push_nick = self.person_info_collection.update_one(
                {
                    "person_id": person_id,
                    "group_nicknames.group_id": group_id_str,
                },
                # 数据库字段名 "group_nicknames.$[group].nicknames", "name", "count" 不变
                {"$push": {"group_nicknames.$[group].nicknames": {"name": sobriquet, "count": 1}}},
                array_filters=[{"group.group_id": group_id_str}],
            )
            if result_push_nick.modified_count > 0:
                # 日志消息更新
                logger.debug(f"成功为 person_id {person_id} 在现有群组 {group_id_str} 中添加新绰号 '{sobriquet}'。")
                return

            # 3c. 如果上一步也未修改 (群组条目本身不存在)，则添加新的群组条目和绰号
            # 确保 group_nicknames 数组存在 (作为保险措施)
            self.person_info_collection.update_one(
                {"person_id": person_id, "group_nicknames": {"$exists": False}}, # 数据库字段名不变
                {"$set": {"group_nicknames": []}}, # 数据库字段名不变
            )
            # 推送新的群组对象到 group_nicknames 数组
            result_push_group = self.person_info_collection.update_one(
                {
                    "person_id": person_id,
                    "group_nicknames.group_id": {"$ne": group_id_str}, # 数据库字段名不变
                },
                {
                    "$push": {
                        "group_nicknames": { # 数据库字段名不变
                            "group_id": group_id_str,
                            "nicknames": [{"name": sobriquet, "count": 1}], # 数据库字段名 name, count 不变
                        }
                    }
                },
            )
            if result_push_group.modified_count > 0:
                # 日志消息更新
                logger.debug(f"为 person_id {person_id} 添加了新的群组 {group_id_str} 和绰号 '{sobriquet}'。")

        except (OperationFailure, DuplicateKeyError) as db_err:
            logger.exception(
                # 日志消息更新
                f"数据库操作失败 ({type(db_err).__name__}): person_id {person_id}, 群组 {group_id_str}, 绰号 {sobriquet}. 错误: {db_err}"
            )
        except Exception as e:
            logger.exception(
                # 日志消息更新
                f"更新群组绰号计数时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}. Error: {e}"
            )