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
    此类中的方法现在是同步的，以匹配同步的PyMongo驱动。
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
            logger.info("SobriquetDB 初始化成功 (使用同步模式)。") # 明确是同步模式

    def is_available(self) -> bool:
        """检查数据库集合是否可用。"""
        return self.person_info_collection is not None

    def upsert_person(self, person_id: str, user_id_int: int, platform: str): # 改为同步方法
        """
        确保数据库中存在指定 person_id 的文档 (Upsert)。
        如果文档不存在，则使用提供的用户信息创建它。此方法为同步。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 upsert_person。")
            return None 
        try:
            result = self.person_info_collection.update_one( # 移除 await
                {"person_id": person_id},
                {
                    "$setOnInsert": { 
                        "person_id": person_id,
                        "user_id": user_id_int, 
                        "platform": platform,
                        "group_nicknames": [], 
                    }
                },
                upsert=True, 
            )
            if result.upserted_id: 
                logger.debug(f"Upsert 操作为 person_id '{person_id}' 创建了新的 person 文档。")
            return result # 直接返回结果
        except DuplicateKeyError as dk_err: 
            logger.error(
                f"数据库操作失败 (DuplicateKeyError): person_id {person_id}. 错误: {dk_err}."
            )
            raise 
        except Exception as e: 
            logger.exception(f"对 person_id {person_id} 执行 Upsert 时失败: {e}")
            raise

    def update_group_sobriquet_count(self, person_id: str, group_id_str: str, sobriquet: str, increment: int = 1): # 改为同步方法
        """
        更新或添加用户在特定群组的绰号计数。默认为增加1。
        如果绰号或群组不存在，会尝试创建它们（仅当 increment > 0 时）。
        此方法为同步。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 update_group_sobriquet_count。")
            return

        try:
            if increment != 0:
                result_inc = self.person_info_collection.update_one( # 移除 await
                    { 
                        "person_id": person_id,
                        "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet}},
                    },
                    {"$inc": {"group_nicknames.$[group].nicknames.$[nick].count": increment}},
                    array_filters=[ 
                        {"group.group_id": group_id_str}, 
                        {"nick.name": sobriquet},         
                    ],
                )
                if result_inc.modified_count > 0:
                    logger.debug(f"已更新 person_id '{person_id}' 在群组 '{group_id_str}' 中绰号 '{sobriquet}' 的计数 (增量: {increment})。")
                    if increment < 0:
                        self.person_info_collection.update_one( # 移除 await
                             {
                                "person_id": person_id,
                                "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet, "nicknames.count": {"$lt": 0}}},
                             },
                             {"$set": {"group_nicknames.$[group].nicknames.$[nick].count": 0}}, 
                             array_filters=[
                                {"group.group_id": group_id_str},
                                {"nick.name": sobriquet},
                            ],
                        )
                    return 

            if increment > 0:
                result_push_nick = self.person_info_collection.update_one( # 移除 await
                    { 
                        "person_id": person_id,
                        "group_nicknames.group_id": group_id_str, 
                    },
                    {"$push": {"group_nicknames.$[group].nicknames": {"name": sobriquet, "count": increment}}},
                    array_filters=[{"group.group_id": group_id_str}], 
                )
                if result_push_nick.modified_count > 0:
                    logger.debug(f"成功为 person_id '{person_id}' 在现有群组 '{group_id_str}' 中添加新绰号 '{sobriquet}' (初始计数: {increment})。")
                    return

                self.person_info_collection.update_one( # 移除 await
                    {"person_id": person_id},
                    {"$setOnInsert": {"group_nicknames": []}}, 
                    upsert=False 
                )
                self.person_info_collection.update_one( # 移除 await
                    {"person_id": person_id, "group_nicknames": {"$exists": False}},
                    {"$set": {"group_nicknames": []}}
                )
                
                result_push_group = self.person_info_collection.update_one( # 移除 await
                    {"person_id": person_id}, 
                    { 
                        "$push": {
                            "group_nicknames": {
                                "group_id": group_id_str,
                                "nicknames": [{"name": sobriquet, "count": increment}],
                            }
                        }
                    },
                )
                if result_push_group.modified_count > 0:
                    logger.debug(f"为 person_id '{person_id}' 添加了新的群组 '{group_id_str}' 和绰号 '{sobriquet}' (初始计数: {increment})。")
                else:
                    logger.warning(
                        f"尝试为 person_id '{person_id}' 添加新群组 '{group_id_str}' 和绰号 '{sobriquet}' 失败。"
                        "这可能表示person_id文档不存在。请确保先调用 upsert_person。"
                    )
        except Exception as e: 
            logger.exception(
                f"更新群组绰号计数时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}. Error: {e}"
            )

    def set_group_sobriquet_value(self, person_id: str, group_id_str: str, sobriquet: str, value: int): # 改为同步方法
        """
        将用户在特定群组的特定绰号的计数值直接设置为指定值。
        如果绰号或群组不存在，会尝试创建它们。确保计数值不低于0。
        此方法为同步。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 set_group_sobriquet_value。")
            return
        
        target_value = max(0, int(value)) 

        try:
            result_set_existing = self.person_info_collection.update_one( # 移除 await
                { 
                    "person_id": person_id,
                    "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet}},
                },
                {"$set": {"group_nicknames.$[group].nicknames.$[nick].count": target_value}},
                array_filters=[
                    {"group.group_id": group_id_str},
                    {"nick.name": sobriquet},
                ],
            )
            if result_set_existing.modified_count > 0:
                logger.debug(f"已将 person_id '{person_id}' 在群组 '{group_id_str}' 中绰号 '{sobriquet}' 的计数设置为 {target_value}。")
                return

            group_exists_doc = self.person_info_collection.find_one( # 移除 await
                {"person_id": person_id, "group_nicknames.group_id": group_id_str},
                {"_id": 1} 
            )

            if not group_exists_doc: 
                self.person_info_collection.update_one( # 移除 await
                    {"person_id": person_id},
                    {"$push": {"group_nicknames": {"group_id": group_id_str, "nicknames": []}}},
                    upsert=False 
                )
                logger.debug(f"为 person_id '{person_id}' 添加了群组框架 '{group_id_str}' 以便设置绰号值。")
            
            self.person_info_collection.update_one( # 移除 await
                { 
                    "person_id": person_id,
                    "group_nicknames.group_id": group_id_str,
                },
                { 
                    "$pull": {"group_nicknames.$[group].nicknames": {"name": sobriquet}}, 
                },
                array_filters=[{"group.group_id": group_id_str}]
            )
            result_add_new_nick_value = self.person_info_collection.update_one( # 移除 await
                 { 
                    "person_id": person_id,
                    "group_nicknames.group_id": group_id_str,
                },
                {"$push": {"group_nicknames.$[group].nicknames": {"name": sobriquet, "count": target_value}}},
                array_filters=[{"group.group_id": group_id_str}]
            )

            if result_add_new_nick_value.modified_count > 0:
                logger.debug(f"成功为 person_id '{person_id}' 在群组 '{group_id_str}' 中设置/添加绰号 '{sobriquet}' 计数为 {target_value}。")
            else:
                logger.warning(
                    f"尝试为 person_id '{person_id}' 在群组 '{group_id_str}' 中设置/添加绰号 '{sobriquet}' 计数为 {target_value} 失败。"
                    "这通常不应发生，除非 person_id 文档不存在。请确保先调用 upsert_person。"
                )
        except Exception as e: 
            logger.exception(
                f"设置群组绰号计数值时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}, value {target_value}. Error: {e}"
            )

    def decay_sobriquets_in_group(self, platform: str, group_id: str, user_ids: Optional[List[str]], decay_factor: float) -> int: # 改为同步方法
        """
        对指定平台、群组和用户列表内的绰号计数应用衰减因子。
        此方法为同步。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 decay_sobriquets_in_group。")
            return 0
        
        if not user_ids: 
            logger.debug(f"未提供用户ID (user_ids 为空或None)，跳过在群组 '{group_id}' 中的绰号衰减。")
            return 0

        if not (0.0 <= decay_factor <= 1.0): 
            logger.warning(f"无效的衰减因子 '{decay_factor}' (应在[0.0, 1.0]之间)，跳过绰号衰减。")
            return 0
        
        if decay_factor == 1.0: 
            logger.debug(f"衰减因子为1.0，不执行实际的绰号衰减操作。")
            return 0

        log_prefix = f"[{platform}:{group_id}]"
        total_nicknames_decayed_count = 0 
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
            person_cursor = self.person_info_collection.find( # 移除 await
                {"platform": platform, "user_id": {"$in": user_ids_int}}
            )

            for person_doc in person_cursor: # 改为同步 for 循环
                person_id = person_doc["person_id"]
                current_group_nicknames_list = person_doc.get("group_nicknames", [])
                person_doc_needs_update = False 
                new_group_nicknames_for_person = [] 

                for group_entry in current_group_nicknames_list: 
                    processed_group_entry = group_entry.copy() 
                    if processed_group_entry.get("group_id") == group_id:
                        current_nicknames_in_group = processed_group_entry.get("nicknames", [])
                        new_nicknames_for_this_group_entry = []
                        group_entry_modified = False 

                        for nick_entry in current_nicknames_in_group: 
                            original_count = nick_entry.get("count", 0)
                            if original_count > 0: 
                                new_count = max(0, int(original_count * decay_factor)) 
                                if new_count < original_count: 
                                    total_nicknames_decayed_count += 1
                                    group_entry_modified = True
                                new_nicknames_for_this_group_entry.append({"name": nick_entry["name"], "count": new_count})
                            else: 
                                new_nicknames_for_this_group_entry.append({"name": nick_entry["name"], "count": 0}) 
                        
                        if group_entry_modified: 
                            processed_group_entry["nicknames"] = new_nicknames_for_this_group_entry
                            person_doc_needs_update = True 
                    new_group_nicknames_for_person.append(processed_group_entry) 

                if person_doc_needs_update: 
                    bulk_operations.append(
                        UpdateOne(
                            {"person_id": person_id}, 
                            {"$set": {"group_nicknames": new_group_nicknames_for_person}} 
                        )
                    )
            
            if bulk_operations:
                logger.debug(f"{log_prefix} 准备对 {len(bulk_operations)} 个用户的绰号数据执行批量衰减操作。总共约 {total_nicknames_decayed_count} 个绰号将被影响。")
                result = self.person_info_collection.bulk_write(bulk_operations, ordered=False) # 移除 await
                logger.info(f"{log_prefix} 批量绰号衰减完成。匹配文档数: {result.matched_count}, 成功修改文档数: {result.modified_count}。实际衰减绰号条目数: {total_nicknames_decayed_count}")
                return total_nicknames_decayed_count 
            else:
                logger.debug(f"{log_prefix} 在指定用户范围内，没有需要衰减的绰号。")
                return 0
        except Exception as e: 
            logger.error(f"{log_prefix} 衰减群组 '{group_id}' 的绰号时发生错误: {e}", exc_info=True)
            return 0

    def get_existing_sobriquets_for_users_in_group( # 改为同步方法
        self, platform: str, group_id: str, user_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取指定用户列表在特定群组中已记录的绰号及其计数。
        返回的字典中，键是用户ID字符串。
        此方法为同步。
        """
        result_map: Dict[str, List[Dict[str, Any]]] = {uid_str: [] for uid_str in user_ids}
        
        if not self.is_available() or not user_ids:
            return result_map 

        user_ids_int = [int(uid) for uid in user_ids if uid.isdigit()]
        if not user_ids_int: 
            return result_map
        
        try:
            person_cursor = self.person_info_collection.find( # 移除 await
                {"platform": platform, "user_id": {"$in": user_ids_int}},
                {"user_id": 1, "group_nicknames": 1} 
            )
            for person_doc in person_cursor: # 改为同步 for 循环
                original_user_id_str = str(person_doc.get("user_id")) 
                
                if original_user_id_str not in result_map: 
                    continue 

                group_nicknames_list = person_doc.get("group_nicknames", [])
                found_target_group = False
                for group_entry in group_nicknames_list: 
                    if group_entry.get("group_id") == group_id: 
                        valid_nicknames = []
                        for nick in group_entry.get("nicknames", []): 
                            if isinstance(nick, dict) and "name" in nick and "count" in nick:
                                valid_nicknames.append({"name": nick["name"], "count": nick["count"]})
                        if valid_nicknames: 
                           result_map[original_user_id_str] = valid_nicknames
                        found_target_group = True
                        break 
                # 如果未找到目标群组的记录，result_map 中该用户的条目将保持为空列表（初始化时已设置）
            return result_map
        except Exception as e:
            logger.error(f"为群组 '{group_id}' 用户 {user_ids} 获取已存在绰号时出错: {e}", exc_info=True)
            return result_map
