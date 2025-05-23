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
    假设使用的 'person_info_collection' 是一个异步MongoDB集合对象 (例如 Motor)。
    """

    def __init__(self, person_info_collection: Optional[Collection]):
        """
        初始化 SobriquetDB 处理器。

        Args:
            person_info_collection: 异步 MongoDB 'person_info' 集合对象。
                                    如果为 None，则数据库操作将被禁用。
        """
        if person_info_collection is None:
            logger.error("未提供 person_info 集合，SobriquetDB 操作将被禁用。")
            self.person_info_collection = None
        else:
            self.person_info_collection = person_info_collection
            logger.info("SobriquetDB 初始化成功。")

    def is_available(self) -> bool:
        """检查数据库集合是否可用。"""
        return self.person_info_collection is not None

    async def upsert_person(self, person_id: str, user_id_int: int, platform: str):
        """
        确保数据库中存在指定 person_id 的文档 (Upsert)。
        如果文档不存在，则使用提供的用户信息创建它。此方法为异步。

        Args:
            person_id: 要查找或创建的 person_id (全局唯一的人物ID)。
            user_id_int: 用户在该平台的数字 ID。
            platform: 平台名称 (例如 "qq", "wechat")。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 upsert_person。")
            return None # 或者可以抛出异常
        try:
            # 使用 $setOnInsert 确保只有在插入新文档时才设置这些字段
            result = await self.person_info_collection.update_one(
                {"person_id": person_id}, # 查询条件
                {
                    "$setOnInsert": { # 仅在插入时设置的字段
                        "person_id": person_id,
                        "user_id": user_id_int, 
                        "platform": platform,
                        "group_nicknames": [], # 初始化群组绰号列表
                    }
                },
                upsert=True, # 如果找不到匹配的文档，则插入一个新文档
            )
            if result.upserted_id: # 如果执行了插入操作
                logger.debug(f"Upsert 操作为 person_id '{person_id}' 创建了新的 person 文档。")
            return result
        except DuplicateKeyError as dk_err: 
            # 理论上 upsert=True 不应该频繁触发此错误，除非 person_id 索引有问题或并发冲突
            logger.error(
                f"数据库操作失败 (DuplicateKeyError): person_id {person_id}. 错误: {dk_err}."
            )
            raise # 将异常重新抛出，让上层处理
        except Exception as e: 
            logger.exception(f"对 person_id {person_id} 执行 Upsert 时失败: {e}")
            raise

    async def update_group_sobriquet_count(self, person_id: str, group_id_str: str, sobriquet: str, increment: int = 1):
        """
        更新或添加用户在特定群组的绰号计数。默认为增加1。
        如果绰号或群组不存在，会尝试创建它们（仅当 increment > 0 时）。
        此方法为异步。

        Args:
            person_id: 目标人物的全局唯一ID。
            group_id_str: 目标群组的 ID (字符串)。
            sobriquet: 要更新或添加的绰号。
            increment: 计数的增量，可以为负数以减少计数。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 update_group_sobriquet_count。")
            return

        try:
            # 步骤1: 尝试对已存在的绰号增加/减少计数
            if increment != 0:
                result_inc = await self.person_info_collection.update_one(
                    { # 查询条件：匹配 person_id 和特定的群组及绰号
                        "person_id": person_id,
                        "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet}},
                    },
                    # 更新操作：增加 group_nicknames 数组中匹配元素的 nicknames 数组中匹配元素的 count 值
                    {"$inc": {"group_nicknames.$[group].nicknames.$[nick].count": increment}},
                    array_filters=[ # 定义数组元素的匹配条件
                        {"group.group_id": group_id_str}, # 匹配群组
                        {"nick.name": sobriquet},         # 匹配绰号名
                    ],
                )
                if result_inc.modified_count > 0:
                    logger.debug(f"已更新 person_id '{person_id}' 在群组 '{group_id_str}' 中绰号 '{sobriquet}' 的计数 (增量: {increment})。")
                    # 如果是减少计数，确保计数不会低于0
                    if increment < 0:
                        await self.person_info_collection.update_one(
                             {
                                "person_id": person_id,
                                "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet, "nicknames.count": {"$lt": 0}}},
                             },
                             {"$set": {"group_nicknames.$[group].nicknames.$[nick].count": 0}}, # 将负数计数修正为0
                             array_filters=[
                                {"group.group_id": group_id_str},
                                {"nick.name": sobriquet},
                            ],
                        )
                    return # 操作完成

            # 步骤2: 如果是增加计数 (increment > 0) 且步骤1未成功 (说明绰号可能不存在于该群组)
            if increment > 0:
                # 尝试将新绰号添加到已存在的群组条目中
                result_push_nick = await self.person_info_collection.update_one(
                    { # 查询条件：匹配 person_id 和群组ID
                        "person_id": person_id,
                        "group_nicknames.group_id": group_id_str, 
                    },
                    # 更新操作：向匹配群组的 nicknames 数组中添加新绰号
                    {"$push": {"group_nicknames.$[group].nicknames": {"name": sobriquet, "count": increment}}},
                    array_filters=[{"group.group_id": group_id_str}], # 确保操作正确的群组元素
                )
                if result_push_nick.modified_count > 0:
                    logger.debug(f"成功为 person_id '{person_id}' 在现有群组 '{group_id_str}' 中添加新绰号 '{sobriquet}' (初始计数: {increment})。")
                    return

                # 步骤3: 如果群组条目本身也不存在，则添加新的群组条目和绰号
                # 首先确保 person 文档存在且 group_nicknames 字段已初始化为数组
                await self.person_info_collection.update_one(
                    {"person_id": person_id},
                    {"$setOnInsert": {"group_nicknames": []}}, 
                    upsert=False # 不应在此创建person文档，它应由 upsert_person 创建
                )
                # 再次确保 group_nicknames 字段存在 (以防万一)
                await self.person_info_collection.update_one(
                    {"person_id": person_id, "group_nicknames": {"$exists": False}},
                    {"$set": {"group_nicknames": []}}
                )
                
                # 添加新的群组对象到 group_nicknames 数组
                result_push_group = await self.person_info_collection.update_one(
                    {"person_id": person_id}, # 定位到 person 文档
                    { # 更新操作：添加包含新群组和新绰号的对象
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
                    # 此处失败可能意味着 person_id 文档不存在，或者并发写入冲突
                    logger.warning(
                        f"尝试为 person_id '{person_id}' 添加新群组 '{group_id_str}' 和绰号 '{sobriquet}' 失败。"
                        "这可能表示person_id文档不存在。请确保先调用 upsert_person。"
                    )
        except Exception as e: 
            logger.exception(
                f"更新群组绰号计数时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}. Error: {e}"
            )

    async def set_group_sobriquet_value(self, person_id: str, group_id_str: str, sobriquet: str, value: int):
        """
        将用户在特定群组的特定绰号的计数值直接设置为指定值。
        如果绰号或群组不存在，会尝试创建它们。确保计数值不低于0。
        此方法为异步。

        Args:
            person_id: 目标人物的全局唯一ID。
            group_id_str: 目标群组的 ID (字符串)。
            sobriquet: 目标绰号。
            value: 要设置的目标计数值。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 set_group_sobriquet_value。")
            return
        
        target_value = max(0, int(value)) # 确保计数值不低于0且为整数

        try:
            # 步骤1: 尝试更新已存在的绰号的计数值
            result_set_existing = await self.person_info_collection.update_one(
                { # 查询条件：匹配 person_id, group_id, 和 sobriquet name
                    "person_id": person_id,
                    "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": sobriquet}},
                },
                # 更新操作：设置匹配到的绰号的 count
                {"$set": {"group_nicknames.$[group].nicknames.$[nick].count": target_value}},
                array_filters=[
                    {"group.group_id": group_id_str},
                    {"nick.name": sobriquet},
                ],
            )
            if result_set_existing.modified_count > 0:
                logger.debug(f"已将 person_id '{person_id}' 在群组 '{group_id_str}' 中绰号 '{sobriquet}' 的计数设置为 {target_value}。")
                return

            # 步骤2: 如果绰号不存在，但群组可能存在，尝试在该群组中添加此绰号并设置计数值
            # 首先确保群组对象存在于 group_nicknames 数组中
            # 使用 $addToSet 确保 group_id 不重复添加，但它不适用于更新内部数组，所以先检查
            group_exists_doc = await self.person_info_collection.find_one(
                {"person_id": person_id, "group_nicknames.group_id": group_id_str},
                {"_id": 1} # 只需要知道是否存在
            )

            if not group_exists_doc: # 如果群组不存在，则先添加群组框架
                await self.person_info_collection.update_one(
                    {"person_id": person_id},
                    {"$push": {"group_nicknames": {"group_id": group_id_str, "nicknames": []}}},
                    upsert=False # 假设 person_id 已由 upsert_person 创建
                )
                logger.debug(f"为 person_id '{person_id}' 添加了群组框架 '{group_id_str}' 以便设置绰号值。")
            
            # 现在群组应该存在了，向该群组的 nicknames 数组添加新绰号（或更新，如果并发时刚被创建）
            # 为了确保原子性地设置值，如果绰号可能已存在但之前的update_one未匹配到（例如并发），
            # 一个健壮的做法是先 $pull 掉同名绰号，再 $push 新的。
            await self.person_info_collection.update_one(
                { # 定位到正确的 person 和 group
                    "person_id": person_id,
                    "group_nicknames.group_id": group_id_str,
                },
                { # 从 nicknames 数组中移除名为 sobriquet 的条目
                    "$pull": {"group_nicknames.$[group].nicknames": {"name": sobriquet}}, 
                },
                array_filters=[{"group.group_id": group_id_str}]
            )
            # 然后添加（或重新添加）带有目标计数值的绰号
            result_add_new_nick_value = await self.person_info_collection.update_one(
                 { 
                    "person_id": person_id,
                    "group_nicknames.group_id": group_id_str,
                },
                # 向 nicknames 数组中添加新绰号条目
                {"$push": {"group_nicknames.$[group].nicknames": {"name": sobriquet, "count": target_value}}},
                array_filters=[{"group.group_id": group_id_str}]
            )

            if result_add_new_nick_value.modified_count > 0:
                logger.debug(f"成功为 person_id '{person_id}' 在群组 '{group_id_str}' 中设置/添加绰号 '{sobriquet}' 计数为 {target_value}。")
            else:
                # 如果 person_id 不存在，upsert_person 应该先被调用
                logger.warning(
                    f"尝试为 person_id '{person_id}' 在群组 '{group_id_str}' 中设置/添加绰号 '{sobriquet}' 计数为 {target_value} 失败。"
                    "这通常不应发生，除非 person_id 文档不存在。请确保先调用 upsert_person。"
                )
        except Exception as e: 
            logger.exception(
                f"设置群组绰号计数值时发生意外错误: person_id {person_id}, group {group_id_str}, nick {sobriquet}, value {target_value}. Error: {e}"
            )

    async def decay_sobriquets_in_group(self, platform: str, group_id: str, user_ids: Optional[List[str]], decay_factor: float) -> int:
        """
        对指定平台、群组和用户列表内的绰号计数应用衰减因子。
        此方法为异步。

        Args:
            platform: 平台名称。
            group_id: 目标群组的 ID (字符串)。
            user_ids: 目标用户的ID列表 (字符串格式)。如果为 None 或空，则不执行操作。
            decay_factor: 衰减因子 (例如 0.98)。计数将乘以该因子。

        Returns:
            int: 实际被衰减（计数值发生改变）的绰号条目总数。
        """
        if not self.is_available():
            logger.error("数据库集合不可用，无法执行 decay_sobriquets_in_group。")
            return 0
        
        if not user_ids: # 如果没有提供用户ID，则不执行衰减
            logger.debug(f"未提供用户ID (user_ids 为空或None)，跳过在群组 '{group_id}' 中的绰号衰减。")
            return 0

        if not (0.0 <= decay_factor <= 1.0): # 衰减因子应在 [0.0, 1.0] 范围内
            logger.warning(f"无效的衰减因子 '{decay_factor}' (应在[0.0, 1.0]之间)，跳过绰号衰减。")
            return 0
        
        if decay_factor == 1.0: # 如果衰减因子为1，则无需操作
            logger.debug(f"衰减因子为1.0，不执行实际的绰号衰减操作。")
            return 0


        log_prefix = f"[{platform}:{group_id}]"
        total_nicknames_decayed_count = 0 # 记录实际衰减的绰号数量
        bulk_operations = [] # 存储批量更新操作

        # 将字符串用户ID列表转换为整数列表，用于数据库查询
        user_ids_int = []
        for uid_str in user_ids:
            if uid_str.isdigit():
                user_ids_int.append(int(uid_str))
            else:
                logger.warning(f"{log_prefix} 无效的用户ID格式 (非纯数字): '{uid_str}'，在衰减操作中跳过此ID。")
        
        if not user_ids_int: # 如果没有有效的数字ID，则不执行
            logger.debug(f"{log_prefix} 没有有效的数字用户ID可供衰减。")
            return 0

        try:
            # 1. 查找所有与指定平台和用户ID列表匹配的 person 文档
            person_cursor = self.person_info_collection.find(
                {"platform": platform, "user_id": {"$in": user_ids_int}}
            )

            async for person_doc in person_cursor: # 异步迭代查询结果
                person_id = person_doc["person_id"]
                current_group_nicknames_list = person_doc.get("group_nicknames", [])
                person_doc_needs_update = False # 标记此文档是否需要更新
                new_group_nicknames_for_person = [] # 构建更新后的群组绰号列表

                for group_entry in current_group_nicknames_list: # 遍历每个群组条目
                    processed_group_entry = group_entry.copy() # 操作副本
                    # 检查是否是目标群组
                    if processed_group_entry.get("group_id") == group_id:
                        current_nicknames_in_group = processed_group_entry.get("nicknames", [])
                        new_nicknames_for_this_group_entry = []
                        group_entry_modified = False # 标记此群组条目是否被修改

                        for nick_entry in current_nicknames_in_group: # 遍历群组内的每个绰号
                            original_count = nick_entry.get("count", 0)
                            if original_count > 0: # 只对计数大于0的绰号进行衰减
                                new_count = max(0, int(original_count * decay_factor)) # 应用衰减并确保不低于0
                                if new_count < original_count: # 如果计数值实际减少
                                    total_nicknames_decayed_count += 1
                                    group_entry_modified = True
                                new_nicknames_for_this_group_entry.append({"name": nick_entry["name"], "count": new_count})
                            else: # 如果原始计数已经是0或更低，则保持为0
                                new_nicknames_for_this_group_entry.append({"name": nick_entry["name"], "count": 0})
                        
                        if group_entry_modified: # 如果此群组的绰号有变化
                            processed_group_entry["nicknames"] = new_nicknames_for_this_group_entry
                            person_doc_needs_update = True # 整个person文档需要更新
                    new_group_nicknames_for_person.append(processed_group_entry) # 添加处理后（或未变）的群组条目

                if person_doc_needs_update: # 如果此用户的绰号数据有变动
                    # 准备一个更新操作，替换整个 group_nicknames 数组
                    bulk_operations.append(
                        UpdateOne(
                            {"person_id": person_id}, # 查询条件
                            {"$set": {"group_nicknames": new_group_nicknames_for_person}} # 更新内容
                        )
                    )
            
            # 2. 如果有需要执行的更新操作，则进行批量写入
            if bulk_operations:
                logger.debug(f"{log_prefix} 准备对 {len(bulk_operations)} 个用户的绰号数据执行批量衰减操作。总共约 {total_nicknames_decayed_count} 个绰号将被影响。")
                result = await self.person_info_collection.bulk_write(bulk_operations, ordered=False) # 执行批量异步写入
                logger.info(f"{log_prefix} 批量绰号衰减完成。匹配文档数: {result.matched_count}, 成功修改文档数: {result.modified_count}。实际衰减绰号条目数: {total_nicknames_decayed_count}")
                return total_nicknames_decayed_count # 返回实际衰减的绰号数量
            else:
                logger.debug(f"{log_prefix} 在指定用户范围内，没有需要衰减的绰号。")
                return 0
        except Exception as e: 
            logger.error(f"{log_prefix} 衰减群组 '{group_id}' 的绰号时发生错误: {e}", exc_info=True)
            return 0

    async def get_existing_sobriquets_for_users_in_group(
        self, platform: str, group_id: str, user_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取指定用户列表在特定群组中已记录的绰号及其计数。
        返回的字典中，键是用户ID字符串。
        此方法为异步。
        """
        # 初始化结果映射，确保所有查询的用户ID都有一个条目，即使它们没有绰号数据
        result_map: Dict[str, List[Dict[str, Any]]] = {uid_str: [] for uid_str in user_ids}
        
        if not self.is_available() or not user_ids:
            return result_map # 如果数据库不可用或用户ID列表为空，返回初始化的空结果

        # 将字符串用户ID列表转换为整数列表，用于数据库查询
        user_ids_int = [int(uid) for uid in user_ids if uid.isdigit()]
        if not user_ids_int: # 如果没有有效的数字ID，返回初始化的空结果
            return result_map
        
        try:
            # 查询匹配平台和用户ID列表的person文档，只获取需要的字段以提高效率
            person_cursor = self.person_info_collection.find(
                {"platform": platform, "user_id": {"$in": user_ids_int}},
                {"user_id": 1, "group_nicknames": 1} # 投影：只获取 user_id 和 group_nicknames
            )
            async for person_doc in person_cursor: # 异步迭代结果
                original_user_id_str = str(person_doc.get("user_id")) # 从文档中获取user_id并转回字符串
                
                # 确保这个从数据库返回的user_id是我们最初查询的user_id之一
                if original_user_id_str not in result_map: 
                    continue # 如果不是，则跳过（理论上不应发生，除非数据不一致）

                group_nicknames_list = person_doc.get("group_nicknames", [])
                found_target_group = False
                for group_entry in group_nicknames_list: # 遍历该用户的群组绰号列表
                    if group_entry.get("group_id") == group_id: # 找到目标群组
                        valid_nicknames = []
                        for nick in group_entry.get("nicknames", []): # 遍历该群组内的绰号
                            # 确保绰号条目格式正确
                            if isinstance(nick, dict) and "name" in nick and "count" in nick:
                                valid_nicknames.append({"name": nick["name"], "count": nick["count"]})
                        if valid_nicknames: # 如果找到了有效的绰号
                           result_map[original_user_id_str] = valid_nicknames
                        found_target_group = True
                        break # 已找到目标群组，无需再遍历此人的其他群组
                
                # 如果遍历完所有群组都未找到目标群组的绰号记录，确保该用户ID在result_map中对应空列表（已在初始化时完成）

            return result_map
        except Exception as e:
            logger.error(f"为群组 '{group_id}' 用户 {user_ids} 获取已存在绰号时出错: {e}", exc_info=True)
            # 出错时，为所有最初查询的用户返回空列表（已在初始化时完成）
            return result_map
