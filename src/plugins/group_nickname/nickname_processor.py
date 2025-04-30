# GroupNickname/nickname_processor.py
import asyncio
import traceback
from multiprocessing import Process, Queue as mpQueue, Event
from typing import Dict, Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from src.common.logger_manager import get_logger # 导入日志管理器
from src.config.config import global_config # 导入全局配置
from .nickname_mapper import analyze_chat_for_nicknames # 导入绰号分析函数
from src.common.database import db # 导入数据库初始化和关闭函数

logger = get_logger("nickname_processor") # 获取日志记录器实例
# --- 运行时状态 (用于安全停止进程) ---
_stop_event = Event()

# --- 数据库连接 ---
mongo_client: Optional[MongoClient] = None # MongoDB 客户端实例
person_info_collection = None # 用户信息集合对象


# --- 数据库更新逻辑 ---
async def update_nickname_counts(group_id: str, nickname_map: Dict[str, str]):
    """
    更新数据库中用户的群组绰号计数。
    """
    person_info_collection = db.person_info

    if not person_info_collection: # 理论上 db 对象总是可用，但保留检查
        logger.error("无法访问数据库集合 'person_info'。无法更新绰号计数。")
        return
    if not nickname_map:
        logger.debug("提供的用于更新的绰号映射为空。")
        return

    logger.info(f"尝试更新群组 '{group_id}' 的绰号计数，映射为: {nickname_map}")

    for user_id, nickname in nickname_map.items():
        if not user_id or not nickname:
            logger.warning(f"跳过绰号映射中的无效条目: user_id='{user_id}', nickname='{nickname}'")
            continue

        group_id_str = str(group_id)

        try:
            # a. 确保用户文档存在 group_nickname 字段且为 list
            person_info_collection.update_one(
                {"person_id": user_id},
                {"$setOnInsert": {"group_nickname": []}},
                upsert=True
            )

            # b. 确保特定 group_id 的条目存在
            update_result = person_info_collection.update_one(
                {"person_id": user_id, f"group_nickname.{group_id_str}": {"$exists": False}},
                {"$push": {"group_nickname": {group_id_str: []}}}
            )
            if update_result.modified_count > 0:
                logger.debug(f"为用户 '{user_id}' 添加了群组 '{group_id_str}' 的条目。")

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
                    logger.debug(f"为用户 '{user_id}' 在群组 '{group_id_str}' 中添加了绰号 '{nickname}'，计数为 1。")
                else:
                    logger.warning(f"未能为用户 '{user_id}' 在群组 '{group_id_str}' 中添加绰号 '{nickname}'。更新结果: {add_nick_result.raw_result}")

            elif update_result.modified_count > 0:
                logger.debug(f"用户 '{user_id}' 在群组 '{group_id_str}' 中的绰号 '{nickname}' 计数已增加。")
            else:
                logger.warning(f"绰号增加操作匹配但未修改用户 '{user_id}' 的绰号 '{nickname}'。更新结果: {update_result.raw_result}")

        except OperationFailure as op_err:
            logger.error(f"数据库操作失败: 用户 {user_id}, 群组 {group_id_str}, 绰号 {nickname}: {op_err}", exc_info=True)
        except Exception as e:
            logger.error(f"更新用户 {user_id} 的绰号 {nickname} 时发生意外错误: {e}", exc_info=True)


# --- 队列和进程 ---
nickname_queue: mpQueue = mpQueue(maxsize=global_config.NICKNAME_QUEUE_MAX_SIZE)
_nickname_process: Optional[Process] = None

async def add_to_nickname_queue(
    chat_history_str: str,
    bot_reply: str,
    group_id: Optional[str],
    user_name_map: Dict[str, str]
):
    """将需要分析的数据放入队列。"""
    if not global_config.ENABLE_NICKNAME_MAPPING:
        return

    if group_id is None:
        logger.debug("私聊跳过绰号映射。")
        return

    try:
        item = (chat_history_str, bot_reply, str(group_id), user_name_map)
        nickname_queue.put_nowait(item)
        logger.debug(f"已将项目添加到群组 {group_id} 的绰号队列。")
    except Exception as e:
        logger.warning(f"无法将项目添加到绰号队列（可能已满）: {e}", exc_info=True)


async def _nickname_processing_loop(queue: mpQueue, stop_event):
    """独立进程中的主循环，处理队列任务。"""

    logger.info("绰号处理循环已启动。")

    while not stop_event.is_set():
        logger.info("绰号处理循环正在运行...")
        try:
            if not queue.empty():
                item = queue.get()
                if isinstance(item, tuple) and len(item) == 4:
                    chat_history_str, bot_reply, group_id, user_name_map = item
                    logger.debug(f"正在处理群组 {group_id} 的绰号映射任务...")

                    analysis_result = await analyze_chat_for_nicknames(chat_history_str, bot_reply, user_name_map)

                    if analysis_result.get("is_exist") and analysis_result.get("data"):
                        await update_nickname_counts(group_id, analysis_result["data"])
                else:
                    logger.warning(f"从队列接收到意外的项目类型: {type(item)}")

                await asyncio.sleep(5)
            else:
                await asyncio.sleep(global_config.NICKNAME_PROCESS_SLEEP_INTERVAL)

        except asyncio.CancelledError:
            logger.info("绰号处理循环已取消。")
            break
        except Exception as e:
            logger.error(f"绰号处理循环出错: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5)

    logger.info("绰号处理循环已结束。")


def _run_processor_process(queue: mpQueue, stop_event):
    """进程启动函数，运行异步循环。"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_nickname_processing_loop(queue, stop_event))
        loop.close()
    except Exception as e:
        logger.error(f"运行绰号处理器进程时出错: {e}", exc_info=True)

def start_nickname_processor():
    """启动绰号映射处理进程。"""
    global _nickname_process
    if not global_config.ENABLE_NICKNAME_MAPPING:
        logger.info("绰号映射功能已禁用。处理器未启动。")
        return

    if _nickname_process is None or not _nickname_process.is_alive():
        logger.info("正在启动绰号处理器进程...")
        stop_event = get_stop_event()
        stop_event.clear()
        _nickname_process = Process(target=_run_processor_process, args=(nickname_queue, stop_event), daemon=True)
        _nickname_process.start()
        logger.info(f"绰号处理器进程已启动，PID: {_nickname_process.pid}")
    else:
        logger.warning("绰号处理器进程已在运行中。")

def stop_nickname_processor():
    """停止绰号映射处理进程。"""
    global _nickname_process
    if _nickname_process and _nickname_process.is_alive():
        logger.info("正在停止绰号处理器进程...")
        set_stop_event() # 发送停止信号
        try:
            _nickname_process.join(timeout=10)
            if _nickname_process.is_alive():
                logger.warning("绰号处理器进程在 10 秒后未优雅停止。正在终止...")
                _nickname_process.terminate()
                _nickname_process.join(timeout=5)
        except Exception as e:
            logger.error(f"停止绰号处理器进程时出错: {e}", exc_info=True)
        finally:
            if _nickname_process and not _nickname_process.is_alive():
                logger.info("绰号处理器进程已成功停止。")
            else:
                logger.error("未能停止绰号处理器进程。")
            _nickname_process = None
    else:
        logger.info("绰号处理器进程未在运行。")

# 可以在应用启动时调用 start_nickname_processor()
# 可以在应用关闭时调用 stop_nickname_processor()
def get_stop_event():
    """获取全局停止事件"""
    return _stop_event

def set_stop_event():
    """设置全局停止事件，通知子进程退出"""
    _stop_event.set()