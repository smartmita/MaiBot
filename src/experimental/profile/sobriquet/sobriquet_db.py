from pymongo.collection import Collection
from pymongo import UpdateOne
from pymongo.errors import OperationFailure, DuplicateKeyError # MongoDB特定的异常类型
from src.common.logger_manager import get_logger
from typing import Optional, List, Dict, Any # 类型提示

logger = get_logger("sobriquet_db") # 获取日志记录器实例

class SobriquetDB:
    """
    处理与群组绰号相关的数据库操作 (MongoDB)。
    封装了对 'person_info' 集合的读写操作。
    核心改动：
    1. 绰号的 'count' (次数) 改为 'strength' (强度)，并使用浮点数存储。
    2. 存储字段 'group_nicknames' 改为 'group_sobriquets'。
    3. 内部数组 'nicknames' 改为 'sobriquets'。
    4. 衰减机制改为按固定值减法衰减强度。
    """

    def __init__(self, person_info_collection: Optional[Collection]):
        """
        初始化 SobriquetDB 处理器。

        Args:
            person_info_collection: MongoDB 'person_info' 集合对象 (同步驱动)。
        """
        if person_info_collection is None:
            logger.error("未提供 person_info 集合，SobriquetDB 操作将被禁用。")
            self.person_info_collection = None
        else:
            self.person_info_collection = person_info_collection
            logger.info("SobriquetDB 初始化成功 (使用同步模式)。")

    def is_available(self) -> bool:
        """检查数据库集合是否可用。"""
        return self.person_info_collection is not None

    def upsert_person(self, person_id: str, user_id_int: int, platform: str):
        """
        确保数据库中存在指定 person_id 的文档 (Upsert)。
        如果文档不存在，则使用提供的用户信息创建它。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 upsert_person。")
            return None
        try:
            # $setOnInsert 中使用新的字段名 group_sobriquets
            result = self.person_info_collection.update_one(
                {"person_id": person_id},
                {
                    "$setOnInsert": {
                        "person_id": person_id,
                        "user_id": user_id_int,
                        "platform": platform,
                        "group_sobriquets": [], # 修改字段名
                    }
                },
                upsert=True,
            )
            if result.upserted_id:
                logger.debug(f"Upsert 操作为 person_id '{person_id}' 创建了新的 person 文档。")
            return result
        except DuplicateKeyError as dk_err:
            logger.error(
                f"数据库操作失败 (DuplicateKeyError): person_id {person_id}. 错误: {dk_err}."
            )
            raise
        except Exception as e:
            logger.exception(f"对 person_id {person_id} 执行 Upsert 时失败: {e}")
            raise

    def update_group_sobriquet_strength(self, person_id: str, group_id_str: str, sobriquet: str, increment: float = 1.0):
        """
        更新或添加用户在特定群组的绰号强度。默认为增加1.0。
        如果绰号或群组不存在，会尝试创建它们（仅当 increment > 0 时）。
        强度值不会低于0.0。
        使用 'group_sobriquets' 和 'sobriquets' 字段。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 update_group_sobriquet_strength。")
            return

        try:
            if increment != 0.0:
                # 尝试增加现有绰号的强度
                result_inc = self.person_info_collection.update_one(
                    {
                        "person_id": person_id,
                        # 查询条件中使用新的字段名
                        "group_sobriquets": {"$elemMatch": {"group_id": group_id_str, "sobriquets.name": sobriquet}},
                    },
                    # 更新操作中使用新的字段名
                    {"$inc": {"group_sobriquets.$[group].sobriquets.$[nick].strength": increment}},
                    array_filters=[
                        {"group.group_id": group_id_str},
                        {"nick.name": sobriquet},
                    ],
                )
                if result_inc.modified_count > 0:
                    logger.debug(f"已更新 person_id '{person_id}' 在群组 '{group_id_str}' 中绰号 '{sobriquet}' 的强度 (增量: {increment})。")
                    # 如果减少强度后小于0，则设置为0
                    if increment < 0.0:
                        self.person_info_collection.update_one(
                             {
                                "person_id": person_id,
                                "group_sobriquets": {"$elemMatch": {"group_id": group_id_str, "sobriquets.name": sobriquet, "sobriquets.strength": {"$lt": 0.0}}},
                             },
                             {"$set": {"group_sobriquets.$[group].sobriquets.$[nick].strength": 0.0}},
                             array_filters=[
                                {"group.group_id": group_id_str},
                                {"nick.name": sobriquet},
                            ],
                        )
                    return

            if increment > 0.0: # 仅当增量为正时才添加新绰号或群组
                # 尝试向现有群组添加新绰号
                result_push_nick = self.person_info_collection.update_one(
                    {
                        "person_id": person_id,
                        "group_sobriquets.group_id": group_id_str, # 查询条件
                    },
                    # $push 操作中使用新的字段名
                    {"$push": {"group_sobriquets.$[group].sobriquets": {"name": sobriquet, "strength": increment}}},
                    array_filters=[{"group.group_id": group_id_str}],
                )
                if result_push_nick.modified_count > 0:
                    logger.debug(f"成功为 person_id '{person_id}' 在现有群组 '{group_id_str}' 中添加新绰号 '{sobriquet}' (初始强度: {increment})。")
                    return

                # 如果群组也不存在，则添加新群组和新绰号
                # 确保 group_sobriquets 数组存在
                self.person_info_collection.update_one(
                    {"person_id": person_id},
                    {"$setOnInsert": {"group_sobriquets": []}}, # 修改字段名
                    upsert=False
                )
                self.person_info_collection.update_one(
                    {"person_id": person_id, "group_sobriquets": {"$exists": False}}, # 修改字段名
                    {"$set": {"group_sobriquets": []}} # 修改字段名
                )
                
                result_push_group = self.person_info_collection.update_one(
                    {"person_id": person_id},
                    {
                        "$push": {
                            # $push 操作中使用新的字段名
                            "group_sobriquets": {
                                "group_id": group_id_str,
                                "sobriquets": [{"name": sobriquet, "strength": increment}], # 修改字段名
                            }
                        }
                    },
                )
                if result_push_group.modified_count > 0:
                    logger.debug(f"为 person_id '{person_id}' 添加了新的群组 '{group_id_str}' 和绰号 '{sobriquet}' (初始强度: {increment})。")
                else:
                    logger.warning(
                        f"尝试为 person_id '{person_id}' 添加新群组 '{group_id_str}' 和绰号 '{sobriquet}' 失败。"
                        "这可能表示person_id文档不存在或group_sobriquets未初始化。请确保先调用 upsert_person。"
                    )
        except Exception as e:
            logger.exception(
                f"更新群组绰号强度时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}. Error: {e}"
            )

    def set_group_sobriquet_strength(self, person_id: str, group_id_str: str, sobriquet: str, strength_value: float):
        """
        将用户在特定群组的特定绰号的强度值直接设置为指定值。
        如果绰号或群组不存在，会尝试创建它们。确保强度值不低于0.0。
        使用 'group_sobriquets' 和 'sobriquets' 字段。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 set_group_sobriquet_strength。")
            return
        
        target_strength = max(0.0, float(strength_value))

        try:
            result_set_existing = self.person_info_collection.update_one(
                {
                    "person_id": person_id,
                    "group_sobriquets": {"$elemMatch": {"group_id": group_id_str, "sobriquets.name": sobriquet}},
                },
                {"$set": {"group_sobriquets.$[group].sobriquets.$[nick].strength": target_strength}},
                array_filters=[
                    {"group.group_id": group_id_str},
                    {"nick.name": sobriquet},
                ],
            )
            if result_set_existing.modified_count > 0:
                logger.debug(f"已将 person_id '{person_id}' 在群组 '{group_id_str}' 中绰号 '{sobriquet}' 的强度设置为 {target_strength}。")
                return

            group_exists_doc = self.person_info_collection.find_one(
                {"person_id": person_id, "group_sobriquets.group_id": group_id_str}, # 修改字段名
                {"_id": 1}
            )

            if not group_exists_doc:
                self.person_info_collection.update_one(
                    {"person_id": person_id},
                    # 修改字段名
                    {"$push": {"group_sobriquets": {"group_id": group_id_str, "sobriquets": []}}},
                    upsert=False
                )
                logger.debug(f"为 person_id '{person_id}' 添加了群组框架 '{group_id_str}' 以便设置绰号强度。")
            
            self.person_info_collection.update_one(
                {
                    "person_id": person_id,
                    "group_sobriquets.group_id": group_id_str, # 修改字段名
                },
                {
                    # 修改字段名
                    "$pull": {"group_sobriquets.$[group].sobriquets": {"name": sobriquet}},
                },
                array_filters=[{"group.group_id": group_id_str}]
            )
            result_add_new_nick_strength = self.person_info_collection.update_one(
                 {
                    "person_id": person_id,
                    "group_sobriquets.group_id": group_id_str, # 修改字段名
                },
                # 修改字段名
                {"$push": {"group_sobriquets.$[group].sobriquets": {"name": sobriquet, "strength": target_strength}}},
                array_filters=[{"group.group_id": group_id_str}]
            )

            if result_add_new_nick_strength.modified_count > 0:
                logger.debug(f"成功为 person_id '{person_id}' 在群组 '{group_id_str}' 中设置/添加绰号 '{sobriquet}' 强度为 {target_strength}。")
            else:
                logger.warning(
                    f"尝试为 person_id '{person_id}' 在群组 '{group_id_str}' 中设置/添加绰号 '{sobriquet}' 强度为 {target_strength} 失败。"
                    "这通常不应发生，除非 person_id 文档不存在。请确保先调用 upsert_person。"
                )
        except Exception as e:
            logger.exception(
                f"设置群组绰号强度值时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}, strength {target_strength}. Error: {e}"
            )

    def decay_sobriquets_in_group(self, platform: str, group_id: str, user_ids: Optional[List[str]], decay_value: float) -> int:
        """
        对指定平台、群组和用户列表内的绰号强度应用减法衰减。
        强度不会低于0.0。使用 'group_sobriquets' 和 'sobriquets' 字段。

        Args:
            platform: 平台标识。
            group_id: 群组ID。
            user_ids: 可选的用户ID列表 (字符串形式)。
            decay_value: 每次衰减的强度值 (应为正数)。

        Returns:
            int: 实际被衰减影响的绰号条目数量。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 decay_sobriquets_in_group。")
            return 0
        
        if not user_ids:
            logger.debug(f"未提供用户ID (user_ids 为空或None)，跳过在群组 '{group_id}' 中的绰号强度衰减。")
            return 0

        if not (decay_value > 0.0):
            logger.warning(f"无效的衰减值 '{decay_value}' (应为正数)，跳过绰号强度衰减。")
            return 0
        
        log_prefix = f"[{platform}:{group_id}]"
        total_nicknames_decayed_count = 0 # 变量名保持，含义是影响的条目数
        bulk_operations = []

        user_ids_int = []
        for uid_str in user_ids:
            if uid_str.isdigit():
                user_ids_int.append(int(uid_str))
            else:
                logger.warning(f"{log_prefix} 无效的用户ID格式 (非纯数字): '{uid_str}'，在衰减操作中跳过此ID。")
        
        if not user_ids_int:
            logger.debug(f"{log_prefix} 没有有效的数字用户ID可供衰减。")
            return 0

        try:
            person_cursor = self.person_info_collection.find(
                {"platform": platform, "user_id": {"$in": user_ids_int}}
            )

            for person_doc in person_cursor:
                person_id = person_doc["person_id"]
                # 修改字段名
                current_group_sobriquets_list = person_doc.get("group_sobriquets", [])
                person_doc_needs_update = False
                new_group_sobriquets_for_person = []

                for group_entry in current_group_sobriquets_list:
                    processed_group_entry = group_entry.copy()
                    if processed_group_entry.get("group_id") == group_id:
                        # 修改字段名
                        current_sobriquets_in_group = processed_group_entry.get("sobriquets", [])
                        new_sobriquets_for_this_group_entry = [] # 修改变量名
                        group_entry_modified_locally = False

                        for nick_entry in current_sobriquets_in_group: # nick_entry 保持，指代单个绰号条目
                            original_strength = float(nick_entry.get("strength", 0.0))
                            if original_strength > 10.0:
                                new_strength = max(10.0, original_strength - decay_value)
                                if new_strength < original_strength:
                                    total_nicknames_decayed_count += 1
                                    group_entry_modified_locally = True
                                new_sobriquets_for_this_group_entry.append({"name": nick_entry["name"], "strength": new_strength})
                            elif original_strength > 0.0:
                                new_strength = max(0.0, original_strength - decay_value)
                                if new_strength < original_strength:
                                    total_nicknames_decayed_count += 1
                                    group_entry_modified_locally = True
                                new_sobriquets_for_this_group_entry.append({"name": nick_entry["name"], "strength": new_strength})
                            else:
                                new_sobriquets_for_this_group_entry.append({"name": nick_entry["name"], "strength": 0.0})
                        
                        if group_entry_modified_locally:
                            processed_group_entry["sobriquets"] = new_sobriquets_for_this_group_entry # 修改字段名
                            person_doc_needs_update = True
                    new_group_sobriquets_for_person.append(processed_group_entry)

                if person_doc_needs_update:
                    bulk_operations.append(
                        UpdateOne(
                            {"person_id": person_id},
                            # 修改字段名
                            {"$set": {"group_sobriquets": new_group_sobriquets_for_person}}
                        )
                    )
            
            if bulk_operations:
                logger.debug(f"{log_prefix} 准备对 {len(bulk_operations)} 个用户的绰号数据执行批量强度衰减操作 (衰减值: {decay_value})。总共约 {total_nicknames_decayed_count} 个绰号将被影响。")
                result = self.person_info_collection.bulk_write(bulk_operations, ordered=False)
                logger.info(f"{log_prefix} 批量绰号强度衰减完成。匹配文档数: {result.matched_count}, 成功修改文档数: {result.modified_count}。实际衰减绰号条目数: {total_nicknames_decayed_count}")
                return total_nicknames_decayed_count
            else:
                logger.debug(f"{log_prefix} 在指定用户范围内，没有需要进行强度衰减的绰号。")
                return 0
        except Exception as e:
            logger.error(f"{log_prefix} 衰减群组 '{group_id}' 的绰号强度时发生错误: {e}", exc_info=True)
            return 0

    def get_existing_sobriquets_for_users_in_group(
        self, platform: str, group_id: str, user_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取指定用户列表在特定群组中已记录的绰号及其强度。
        返回的字典中，键是用户ID字符串。每个绰号条目包含 "name" 和 "strength"。
        使用 'group_sobriquets' 和 'sobriquets' 字段。
        """
        result_map: Dict[str, List[Dict[str, Any]]] = {uid_str: [] for uid_str in user_ids}
        
        if not self.is_available() or not user_ids:
            return result_map

        user_ids_int = [int(uid) for uid in user_ids if uid.isdigit()]
        if not user_ids_int:
            return result_map
        
        try:
            person_cursor = self.person_info_collection.find(
                {"platform": platform, "user_id": {"$in": user_ids_int}},
                # 修改投影字段名
                {"user_id": 1, "group_sobriquets": 1}
            )
            for person_doc in person_cursor:
                original_user_id_str = str(person_doc.get("user_id"))
                
                if original_user_id_str not in result_map:
                    continue

                # 修改字段名
                group_sobriquets_list = person_doc.get("group_sobriquets", [])
                for group_entry in group_sobriquets_list:
                    if group_entry.get("group_id") == group_id:
                        valid_sobriquets = [] # 修改变量名
                        # 修改字段名
                        for nick_entry in group_entry.get("sobriquets", []): # nick_entry 保持，指代单个绰号条目
                            if isinstance(nick_entry, dict) and "name" in nick_entry and "strength" in nick_entry:
                                valid_sobriquets.append({"name": nick_entry["name"], "strength": float(nick_entry["strength"])})
                        if valid_sobriquets:
                           result_map[original_user_id_str] = valid_sobriquets
                        break
            return result_map
        except Exception as e:
            logger.error(f"为群组 '{group_id}' 用户 {user_ids} 获取已存在绰号强度时出错: {e}", exc_info=True)
            return result_map
