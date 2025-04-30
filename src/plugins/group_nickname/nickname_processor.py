# GroupNickname/nickname_processor.py
import asyncio
import time
import traceback
# 明确导入 Event 和 Queue
from multiprocessing import Process, Queue as mpQueue
# 尝试从 synchronize 导入 Event
from multiprocessing.synchronize import Event as mpEvent
from typing import Dict, Any, Tuple, Optional, List

from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure

# 假设你的项目结构允许这样导入
try:
    from src.common.logger_manager import get_logger
    from src.config.config import global_config
    from .config import (
        ENABLE_NICKNAME_MAPPING, DB_COLLECTION_PERSON_INFO,
        NICKNAME_QUEUE_MAX_SIZE, NICKNAME_PROCESS_SLEEP_INTERVAL,
        get_stop_event, set_stop_event
    )
    from .nickname_mapper import analyze_chat_for_nicknames
except ImportError:
    # 提供备选导入路径或记录错误，以便调试
    print("Error: Failed to import necessary modules. Please check your project structure and PYTHONPATH.")
    # 在无法导入时，定义临时的 get_logger 以避免 NameError，但这只是权宜之计
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    # 定义临时的全局配置，这同样是权宜之计
    class MockGlobalConfig:
        mongodb_uri = "mongodb://localhost:27017/" # 示例 URI
        mongodb_database = "your_db_name" # 示例数据库名
    global_config = MockGlobalConfig()
    # 定义临时的配置变量
    ENABLE_NICKNAME_MAPPING = True
    DB_COLLECTION_PERSON_INFO = "person_info"
    NICKNAME_QUEUE_MAX_SIZE = 100
    NICKNAME_PROCESS_SLEEP_INTERVAL = 0.5
    # 使用导入的 mpEvent
    _stop_event_internal = mpEvent()
    def get_stop_event(): return _stop_event_internal
    def set_stop_event(): _stop_event_internal.set()
    # 定义临时的 analyze_chat_for_nicknames
    async def analyze_chat_for_nicknames(*args, **kwargs): return {"is_exist": False}


logger = get_logger("nickname_processor")

# --- 数据库连接 ---
mongo_client: Optional[MongoClient] = None
person_info_collection = None

def _initialize_db():
    """初始化数据库连接（在子进程中调用）"""
    global mongo_client, person_info_collection
    if mongo_client is None:
        try:
            mongo_uri = global_config.mongodb_uri
            if not mongo_uri:
                raise ValueError("MongoDB URI not found in global config.")

            mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command('ping')
            db = mongo_client[global_config.mongodb_database]
            person_info_collection = db[DB_COLLECTION_PERSON_INFO]
            logger.info("Nickname processor: Database connection initialized successfully.")
        except (ConnectionFailure, ValueError, OperationFailure) as e:
            logger.error(f"Nickname processor: Failed to initialize database connection: {e}", exc_info=True)
            mongo_client = None
            person_info_collection = None
        except Exception as e:
            logger.error(f"Nickname processor: An unexpected error occurred during DB initialization: {e}", exc_info=True)
            mongo_client = None
            person_info_collection = None


def _close_db():
    """关闭数据库连接"""
    global mongo_client
    if mongo_client:
        try:
            mongo_client.close()
            logger.info("Nickname processor: Database connection closed.")
        except Exception as e:
            logger.error(f"Nickname processor: Error closing database connection: {e}", exc_info=True)
        finally:
            mongo_client = None


# --- 数据库更新逻辑 ---
async def update_nickname_counts(group_id: str, nickname_map: Dict[str, str]):
    """
    更新数据库中用户的群组绰号计数。

    Args:
        group_id (str): 群组 ID。
        nickname_map (Dict[str, str]): 需要更新的 {用户ID: 绰号} 映射。
    """
    if not person_info_collection:
        logger.error("Database collection is not initialized. Cannot update nickname counts.")
        return
    if not nickname_map:
        logger.debug("Empty nickname map provided for update.")
        return

    logger.info(f"Attempting to update nickname counts for group '{group_id}' with map: {nickname_map}")

    for user_id, nickname in nickname_map.items():
        if not user_id or not nickname:
            logger.warning(f"Skipping invalid entry in nickname map: user_id='{user_id}', nickname='{nickname}'")
            continue

        group_id_str = str(group_id) # 确保是字符串

        try:
            # a. 确保用户文档存在 group_nickname 字段且为 list
            person_info_collection.update_one(
                {"person_id": user_id},
                {"$setOnInsert": {"group_nickname": []}}, # 如果字段不存在则创建为空列表
                upsert=True
            )

            # b. 确保特定 group_id 的条目存在
            update_result = person_info_collection.update_one(
                {"person_id": user_id, f"group_nickname.{group_id_str}": {"$exists": False}},
                {"$push": {"group_nickname": {group_id_str: []}}} # 如果不存在则添加
            )
            if update_result.modified_count > 0:
                logger.debug(f"Added group entry for group '{group_id_str}' for user '{user_id}'.")

            # c. 确保特定 nickname 存在于 group_id 的数组中，并增加计数
            update_result = person_info_collection.update_one(
                {
                    "person_id": user_id,
                    "group_nickname": {
                        "$elemMatch": {
                                group_id_str: {"$elemMatch": {nickname: {"$exists": True}}}
                        }
                    }
                },
                {"$inc": {f"group_nickname.$[group].$[nick].{nickname}": 1}},
                array_filters=[
                    {f"group.{group_id_str}": {"$exists": True}},
                    {f"nick.{nickname}": {"$exists": True}}
                ]
            )

            if update_result.matched_count == 0:
                # nickname 不存在，添加 nickname 并设置次数为 1
                add_nick_result = person_info_collection.update_one(
                    {"person_id": user_id, f"group_nickname.{group_id_str}": {"$exists": True}},
                    {"$push": {f"group_nickname.$[group].{group_id_str}": {nickname: 1}}},
                    array_filters=[{f"group.{group_id_str}": {"$exists": True}}]
                )
                if add_nick_result.modified_count > 0:
                    logger.debug(f"Added nickname '{nickname}' with count 1 for user '{user_id}' in group '{group_id_str}'.")
                else:
                    logger.warning(f"Failed to add nickname '{nickname}' for user '{user_id}' in group '{group_id_str}'. Update result: {add_nick_result.raw_result}")

            elif update_result.modified_count > 0:
                logger.debug(f"Incremented count for nickname '{nickname}' for user '{user_id}' in group '{group_id_str}'.")
            else:
                logger.warning(f"Nickname increment operation matched but did not modify for user '{user_id}', nickname '{nickname}'. Update result: {update_result.raw_result}")

        except OperationFailure as op_err:
            logger.error(f"Database operation failed for user {user_id}, group {group_id_str}, nickname {nickname}: {op_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error updating nickname for user {user_id}, group {group_id_str}, nickname {nickname}: {e}", exc_info=True)


# --- 队列和进程 ---
# 使用明确导入的类型
nickname_queue: mpQueue[Tuple[str, str, str, Dict[str, str]]] = mpQueue(maxsize=NICKNAME_QUEUE_MAX_SIZE)
_nickname_process: Optional[Process] = None

async def add_to_nickname_queue(
    chat_history_str: str,
    bot_reply: str,
    group_id: Optional[str], # 群聊时需要
    user_name_map: Dict[str, str] # 用户ID到名字的映射
):
    """将需要分析的数据放入队列。"""
    if not ENABLE_NICKNAME_MAPPING:
        return

    if group_id is None:
        logger.debug("Skipping nickname mapping for private chat.")
        return # 私聊暂时不处理绰号映射

    try:
        item = (chat_history_str, bot_reply, str(group_id), user_name_map) # 确保 group_id 是字符串
        # 使用 put_nowait，如果队列满则会抛出 Full 异常
        nickname_queue.put_nowait(item)
        logger.debug(f"Added item to nickname queue for group {group_id}.")
    # 捕获 queue.Full 异常
    except Exception as e:
        # 检查异常类型是否为队列满（需要导入 queue 模块或处理 Full 异常）
        # from queue import Full # 如果 nickname_queue 是 asyncio.Queue
        # if isinstance(e, Full):
        #     logger.warning("Nickname processing queue is full. Discarding new item.")
        # else:
        #     logger.error(f"Error adding item to nickname queue: {e}", exc_info=True)
        # 由于 multiprocessing.Queue 的 Full 异常在不同环境下可能不同，这里暂时捕获通用异常
        logger.warning(f"Failed to add item to nickname queue (possibly full): {e}", exc_info=True)


# 使用从 synchronize 导入的 mpEvent
async def _nickname_processing_loop(queue: mpQueue, stop_event: mpEvent): # 使用 mpEvent
    """独立进程中的主循环，处理队列任务。"""
    _initialize_db() # 初始化数据库连接
    logger.info("Nickname processing loop started.")

    while not stop_event.is_set():
        try:
            if not queue.empty():
                # 从队列获取任务
                chat_history_str, bot_reply, group_id, user_name_map = queue.get()
                logger.debug(f"Processing nickname mapping task for group {group_id}...")

                # 调用 LLM 分析
                analysis_result = await analyze_chat_for_nicknames(chat_history_str, bot_reply, user_name_map)

                # 如果找到映射，更新数据库
                if analysis_result.get("is_exist") and analysis_result.get("data"):
                    await update_nickname_counts(group_id, analysis_result["data"])

                # 短暂 sleep 避免 CPU 占用过高
                await asyncio.sleep(0.05) # 稍微减少 sleep 时间

            else:
                # 队列为空时休眠
                await asyncio.sleep(NICKNAME_PROCESS_SLEEP_INTERVAL)

        except asyncio.CancelledError:
            logger.info("Nickname processing loop cancelled.")
            break # 响应取消请求
        except Exception as e:
            logger.error(f"Error in nickname processing loop: {e}\n{traceback.format_exc()}")
            # 发生错误时也休眠一下，防止快速连续出错
            await asyncio.sleep(5)

    _close_db() # 关闭数据库连接
    logger.info("Nickname processing loop finished.")


# 使用从 synchronize 导入的 mpEvent
def _run_processor_process(queue: mpQueue, stop_event: mpEvent): # 使用 mpEvent
    """进程启动函数，运行异步循环。"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_nickname_processing_loop(queue, stop_event))
        loop.close()
    except Exception as e:
        logger.error(f"Error running nickname processor process: {e}", exc_info=True)

def start_nickname_processor():
    """启动绰号映射处理进程。"""
    global _nickname_process
    if not ENABLE_NICKNAME_MAPPING:
        logger.info("Nickname mapping feature is disabled. Processor not started.")
        return

    if _nickname_process is None or not _nickname_process.is_alive():
        logger.info("Starting nickname processor process...")
        stop_event = get_stop_event()
        stop_event.clear()
        # 传递明确导入的类型
        _nickname_process = Process(target=_run_processor_process, args=(nickname_queue, stop_event), daemon=True)
        _nickname_process.start()
        logger.info(f"Nickname processor process started with PID: {_nickname_process.pid}")
    else:
        logger.warning("Nickname processor process is already running.")

def stop_nickname_processor():
    """停止绰号映射处理进程。"""
    global _nickname_process
    if _nickname_process and _nickname_process.is_alive():
        logger.info("Stopping nickname processor process...")
        set_stop_event()
        try:
            _nickname_process.join(timeout=10)
            if _nickname_process.is_alive():
                logger.warning("Nickname processor process did not stop gracefully after 10 seconds. Terminating...")
                _nickname_process.terminate()
                _nickname_process.join(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping nickname processor process: {e}", exc_info=True)
        finally:
            if _nickname_process and not _nickname_process.is_alive():
                logger.info("Nickname processor process stopped successfully.")
            else:
                logger.error("Failed to stop nickname processor process.")
            _nickname_process = None
    else:
        logger.info("Nickname processor process is not running.")

# 可以在应用启动时调用 start_nickname_processor()
# 在应用关闭时调用 stop_nickname_processor()
