# nickname_processor.py (多线程版本 - 使用全局 config)

import asyncio
import traceback
import threading
import queue
from typing import Dict, Optional, Any

# 数据库和日志导入
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from src.common.logger_manager import get_logger
from src.common.database import db # 使用全局 db

logger = get_logger("nickname_processor")

# --- 恢复导入全局 config ---
try:
    from src.config.config import global_config # <--- 直接导入全局配置
except ImportError:
    logger.critical("无法导入 global_config！")
    global_config = None # 设置为 None
# ---------------------------

# 绰号分析函数导入
from .nickname_mapper import analyze_chat_for_nicknames

# --- 使用 threading.Event ---
_stop_event = threading.Event()
# --------------------------

# --- 数据库更新逻辑 (使用全局 db) ---
async def update_nickname_counts(group_id: str, nickname_map: Dict[str, str]):
    """更新数据库中用户的群组绰号计数 (使用全局 db)"""
    person_info_collection = db.person_info
    # ... (函数体保持不变, 参考之前的版本) ...
    if not nickname_map: logger.debug("提供的用于更新的绰号映射为空。"); return
    logger.info(f"尝试更新群组 '{group_id}' 的绰号计数，映射为: {nickname_map}")
    for user_id_str, nickname in nickname_map.items():
        if not user_id_str or not nickname: logger.warning(f"跳过无效条目: user_id='{user_id_str}', nickname='{nickname}'"); continue
        group_id_str = str(group_id)
        try: user_id_int = int(user_id_str)
        except ValueError: logger.warning(f"无效的用户ID格式: '{user_id_str}'，跳过。"); continue
        try:
            person_info_collection.update_one({"user_id": user_id_int},{"$setOnInsert": {"user_id": user_id_int}}, upsert=True)
            person_info_collection.update_one({"user_id": user_id_int, "group_nicknames": {"$exists": False}}, {"$set": {"group_nicknames": []}})
            update_result = person_info_collection.update_one({"user_id": user_id_int, "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": nickname}}}, {"$inc": {"group_nicknames.$[group].nicknames.$[nick].count": 1}}, array_filters=[{"group.group_id": group_id_str}, {"nick.name": nickname}])
            if update_result.modified_count > 0: continue
            update_result = person_info_collection.update_one({"user_id": user_id_int, "group_nicknames.group_id": group_id_str}, {"$push": {"group_nicknames.$[group].nicknames": {"name": nickname, "count": 1}}}, array_filters=[{"group.group_id": group_id_str}])
            if update_result.modified_count > 0: continue
            update_result = person_info_collection.update_one({"user_id": user_id_int, "group_nicknames.group_id": {"$ne": group_id_str}}, {"$push": {"group_nicknames": {"group_id": group_id_str, "nicknames": [{"name": nickname, "count": 1}]}}})
        except OperationFailure as op_err: logger.exception(f"数据库操作失败: 用户 {user_id_str}, 群组 {group_id_str}, 绰号 {nickname}")
        except Exception as e: logger.exception(f"更新用户 {user_id_str} 的绰号 {nickname} 时发生意外错误")


# --- 使用 queue.Queue ---
# --- 修改：直接使用 global_config ---
queue_max_size = getattr(global_config, 'NICKNAME_QUEUE_MAX_SIZE', 100) if global_config else 100
# --------------------------------
nickname_queue: queue.Queue = queue.Queue(maxsize=queue_max_size)
# ----------------------

_nickname_thread: Optional[threading.Thread] = None

# --- add_to_nickname_queue (使用全局 config) ---
async def add_to_nickname_queue(
    chat_history_str: str,
    bot_reply: str,
    group_id: Optional[str],
    user_name_map: Dict[str, str]
):
    """将需要分析的数据放入队列。"""
    # --- 修改：使用全局 config ---
    if not global_config or not global_config.ENABLE_NICKNAME_MAPPING:
    # ---------------------------
        return
    if group_id is None: logger.debug("私聊跳过绰号映射。"); return
    try:
        item = (chat_history_str, bot_reply, str(group_id), user_name_map)
        nickname_queue.put_nowait(item)
        logger.debug(f"已将项目添加到群组 {group_id} 的绰号队列。当前大小: {nickname_queue.qsize()}")
    except queue.Full: logger.warning(f"无法将项目添加到绰号队列：队列已满 (maxsize={nickname_queue.maxsize})。")
    except Exception as e: logger.warning(f"无法将项目添加到绰号队列: {e}", exc_info=True)


# --- _nickname_processing_loop (使用全局 config) ---
async def _nickname_processing_loop(q: queue.Queue, stop_event: threading.Event):
    """独立线程中的主循环，处理队列任务 (使用全局 db 和 config)。"""
    thread_id = threading.get_ident()
    logger.info(f"绰号处理循环已启动 (线程 ID: {thread_id})。")
    # --- 修改：使用全局 config ---
    sleep_interval = getattr(global_config, 'NICKNAME_PROCESS_SLEEP_INTERVAL', 0.5) if global_config else 0.5
    # ---------------------------

    while not stop_event.is_set():
        try:
            item = q.get(block=True, timeout=sleep_interval)
            if isinstance(item, tuple) and len(item) == 4:
                chat_history_str, bot_reply, group_id, user_name_map = item
                logger.debug(f"(线程 ID: {thread_id}) 正在处理群组 {group_id} 的绰号映射任务...")
                # analyze_chat_for_nicknames 内部也应使用 global_config
                analysis_result = await analyze_chat_for_nicknames(chat_history_str, bot_reply, user_name_map)
                if analysis_result.get("is_exist") and analysis_result.get("data"):
                    await update_nickname_counts(group_id, analysis_result["data"])
            else:
                logger.warning(f"(线程 ID: {thread_id}) 从队列接收到意外的项目类型: {type(item)}")
            q.task_done()
        except queue.Empty: continue
        except asyncio.CancelledError: logger.info(f"绰号处理循环已取消 (线程 ID: {thread_id})。"); break
        except Exception as e: logger.error(f"(线程 ID: {thread_id}) 绰号处理循环出错: {e}\n{traceback.format_exc()}"); await asyncio.sleep(5)
    logger.info(f"绰号处理循环已结束 (线程 ID: {thread_id})。")


# --- _run_processor_thread (保持不变，不处理 db 或 config) ---
def _run_processor_thread(q: queue.Queue, stop_event: threading.Event):
    """线程启动函数，运行异步循环。"""
    loop = None
    thread_id = threading.get_ident()
    logger.info(f"Nickname processor thread starting (Thread ID: {thread_id})...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"(Thread ID: {thread_id}) Asyncio event loop created and set.")
        loop.run_until_complete(_nickname_processing_loop(q, stop_event))
    except Exception as e: logger.error(f"(Thread ID: {thread_id}) Error running nickname processor thread: {e}", exc_info=True)
    finally:
        if loop:
            try:
                if loop.is_running():
                    all_tasks = asyncio.all_tasks(loop)
                    if all_tasks:
                        logger.info(f"(Thread ID: {thread_id}) Cancelling {len(all_tasks)} tasks...")
                        for task in all_tasks: task.cancel()
                        loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                    loop.stop()
                loop.close()
                logger.info(f"(Thread ID: {thread_id}) Asyncio loop closed.")
            except Exception as loop_close_err: logger.error(f"(Thread ID: {thread_id}) Error closing loop: {loop_close_err}", exc_info=True)
        logger.info(f"Nickname processor thread finished (Thread ID: {thread_id}).")


# --- start_nickname_processor (使用全局 config) ---
def start_nickname_processor():
    """启动绰号映射处理线程。"""
    global _nickname_thread
    # --- 修改：使用全局 config ---
    if not global_config or not global_config.ENABLE_NICKNAME_MAPPING:
    # ---------------------------
        logger.info("绰号映射功能已禁用或无法获取配置。处理器未启动。")
        return

    if _nickname_thread is None or not _nickname_thread.is_alive():
        logger.info("正在启动绰号处理器线程...")
        stop_event = get_stop_event()
        stop_event.clear()
        _nickname_thread = threading.Thread(
            target=_run_processor_thread,
            args=(nickname_queue, stop_event),
            daemon=True
        )
        _nickname_thread.start()
        logger.info(f"绰号处理器线程已启动 (Thread ID: {_nickname_thread.ident})")
    else:
        logger.warning("绰号处理器线程已在运行中。")

# --- stop_nickname_processor (保持不变) ---
def stop_nickname_processor():
    """停止绰号映射处理线程。"""
    global _nickname_thread
    if _nickname_thread and _nickname_thread.is_alive():
        logger.info("正在停止绰号处理器线程...")
        set_stop_event()
        try:
            _nickname_thread.join(timeout=10)
            if _nickname_thread.is_alive(): logger.warning("绰号处理器线程在 10 秒后未结束。")
        except Exception as e: logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
        finally:
            if _nickname_thread and not _nickname_thread.is_alive(): logger.info("绰号处理器线程已成功停止。")
            else: logger.warning("停止绰号处理器线程：线程可能仍在运行。")
            _nickname_thread = None
    else:
        logger.info("绰号处理器线程未在运行或已被清理。")

# --- Event 控制函数 (保持不变) ---
def get_stop_event() -> threading.Event:
    """获取全局停止事件"""
    return _stop_event

def set_stop_event():
    """设置全局停止事件，通知子线程退出"""
    _stop_event.set()
