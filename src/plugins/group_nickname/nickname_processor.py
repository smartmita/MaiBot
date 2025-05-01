import os
import asyncio
import traceback
from multiprocessing import Process, Queue as mpQueue, Event
from typing import Dict, Optional

from pymongo import MongoClient
from pymongo.errors import OperationFailure

from src.common.logger_manager import get_logger # 导入日志管理器
from src.config.config import global_config # 导入全局配置
from .nickname_mapper import analyze_chat_for_nicknames # 导入绰号分析函数
from src.common.database import get_db, close_db

logger = get_logger("nickname_processor") # 获取日志记录器实例
# --- 运行时状态 (用于安全停止进程) ---
_stop_event = Event()

# --- 数据库连接 ---
mongo_client: Optional[MongoClient] = None # MongoDB 客户端实例
person_info_collection = None # 用户信息集合对象

# --- 数据库更新逻辑 (使用推荐的新结构) ---
async def update_nickname_counts(group_id: str, nickname_map: Dict[str, str], current_db):
    """
    更新数据库中用户的群组绰号计数。
    使用传入的数据库实例。
    """
    # 从传入的 db 实例获取 collection
    person_info_collection = current_db.person_info # <--- 使用 current_db

    if not nickname_map:
        logger.debug("提供的用于更新的绰号映射为空。")
        return

    logger.info(f"尝试更新群组 '{group_id}' 的绰号计数，映射为: {nickname_map}")

    for user_id_str, nickname in nickname_map.items():
        if not user_id_str or not nickname:
            logger.warning(f"跳过绰号映射中的无效条目: user_id='{user_id_str}', nickname='{nickname}'")
            continue

        group_id_str = str(group_id)
        try:
            user_id_int = int(user_id_str)
        except ValueError:
            logger.warning(f"无效的用户ID格式: '{user_id_str}'，跳过。")
            continue

        try:
            # 确保后续所有的数据库操作都使用从 current_db 获取的 person_info_collection
            person_info_collection.update_one(
                {"user_id": user_id_int},
                {"$setOnInsert": {"user_id": user_id_int}},
                upsert=True
            )
            # 确保 group_nicknames 字段存在且为数组 (如果不存在则创建)
            person_info_collection.update_one(
                {"user_id": user_id_int, "group_nicknames": {"$exists": False}},
                {"$set": {"group_nicknames": []}}
            )


            # 尝试直接增加现有绰号的计数
            # 条件：用户存在，且 group_nicknames 数组中存在一个元素其 group_id 匹配，且该元素的 nicknames 数组中存在一个元素的 name 匹配
            update_result = person_info_collection.update_one(
                {
                    "user_id": user_id_int,
                    "group_nicknames": {             # <--- 确保使用 group_nicknames
                        "$elemMatch": {"group_id": group_id_str, "nicknames.name": nickname}
                    }
                },
                {                                     # <--- 确保使用 group_nicknames
                    "$inc": {"group_nicknames.$[group].nicknames.$[nick].count": 1}
                },
                array_filters=[
                    {"group.group_id": group_id_str},
                    {"nick.name": nickname}
                ]
            )

            if update_result.modified_count > 0:
                logger.debug(f"用户 '{user_id_str}' 在群组 '{group_id_str}' 中的绰号 '{nickname}' 计数已增加。")
                continue # 处理完成，进行下一次循环

            # 如果未修改任何内容，尝试将新绰号添加到现有群组的 nicknames 数组中
            # 条件：用户存在，且 group_nicknames 数组中存在一个元素其 group_id 匹配
            update_result = person_info_collection.update_one(
                {
                    "user_id": user_id_int,
                    "group_nicknames.group_id": group_id_str
                },
                {
                    "$push": {"group_nicknames.$[group].nicknames": {"name": nickname, "count": 1}}
                },
                array_filters=[
                    {"group.group_id": group_id_str}
                ]
            )

            if update_result.modified_count > 0:
                logger.debug(f"为用户 '{user_id_str}' 在群组 '{group_id_str}' 中添加了新绰号 '{nickname}'，计数为 1。")
                continue # 处理完成，进行下一次循环

            # 如果未修改任何内容，说明群组条目本身可能不存在于 group_nicknames 数组中，尝试添加新的群组条目
            # 条件：用户存在，且 group_nicknames 数组中 *不包含* 指定 group_id 的元素
            update_result = person_info_collection.update_one(
                {
                    "user_id": user_id_int,
                    "group_nicknames.group_id": {"$ne": group_id_str}
                },
                {
                    "$push": {
                        "group_nicknames": {
                            "group_id": group_id_str,
                            "nicknames": [{"name": nickname, "count": 1}]
                        }
                    }
                }
                # 注意：这里不需要 upsert=True，因为步骤1已确保用户存在。
                # 如果字段 group_nicknames 不存在，$push 会自动创建它。
            )

            # 记录日志（无论修改与否，因为可能是因为组已存在但无匹配导致没修改）
            if update_result.modified_count > 0:
                logger.debug(f"为用户 '{user_id_str}' 添加了新群组 '{group_id_str}' 条目和绰号 '{nickname}'。")
            else:
                # 到这里还没成功，可能意味着群组已存在但之前的步骤意外失败，或者有并发问题
                logger.warning(f"未能为用户 '{user_id_str}' 更新或添加群组 '{group_id_str}' 的绰号 '{nickname}'。可能群组已存在但前面的步骤未成功修改。UpdateResult: {update_result.raw_result}")


        except OperationFailure as op_err:
            # 使用 logger.exception 来记录数据库操作错误，自动包含 traceback
            logger.exception(f"数据库操作失败: 用户 {user_id_str}, 群组 {group_id_str}, 绰号 {nickname}") # <--- 修改了日志记录方式
        except Exception as e:
            # 记录其他意外错误
            logger.exception(f"更新用户 {user_id_str} 的绰号 {nickname} 时发生意外错误") # <--- 修改了日志记录方式

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


async def _nickname_processing_loop(queue: mpQueue, stop_event, current_db):
    """独立进程中的主循环，处理队列任务，使用传入的数据库连接。"""
    pid = os.getpid() # 获取进程ID用于日志
    logger.info(f"绰号处理循环已启动 (PID: {pid})。 使用数据库: {current_db.name}")

    while not stop_event.is_set():
        try:
            if not queue.empty():
                # 或者使用 queue.get(timeout=...) 来避免忙等待，并处理 Empty 异常
                item = queue.get()
                if isinstance(item, tuple) and len(item) == 4:
                    chat_history_str, bot_reply, group_id, user_name_map = item
                    logger.debug(f"(PID: {pid}) 正在处理群组 {group_id} 的绰号映射任务...")

                    analysis_result = await analyze_chat_for_nicknames(chat_history_str, bot_reply, user_name_map)

                    if analysis_result.get("is_exist") and analysis_result.get("data"):
                        # 将数据库实例传递下去
                        await update_nickname_counts(group_id, analysis_result["data"], current_db)
                else:
                    logger.warning(f"(PID: {pid}) 从队列接收到意外的项目类型: {type(item)}")
                # 处理完一个任务后短暂休眠，避免CPU空转
                await asyncio.sleep(0.1)
            else:
                # 队列为空时，休眠更长时间
                await asyncio.sleep(global_config.NICKNAME_PROCESS_SLEEP_INTERVAL)

        except asyncio.CancelledError:
            logger.info(f"绰号处理循环已取消 (PID: {pid})。")
            break
        except Exception as e:
            logger.error(f"(PID: {pid}) 绰号处理循环出错: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5) # 出错后等待一段时间

    logger.info(f"绰号处理循环已结束 (PID: {pid})。")


def _run_processor_process(queue: mpQueue, stop_event):
    """进程启动函数，管理自己的数据库连接并运行异步循环。"""
    db_instance = None # 初始化数据库实例变量
    loop = None
    pid = os.getpid()
    logger.info(f"绰号处理器进程启动中 (PID: {pid})...")

    try:
        # 调用 get_db() 会触发此进程的懒加载逻辑
        logger.info(f"子进程 (PID: {pid}) - 即将调用 get_db()")
        db_instance = get_db()
        logger.info(f"子进程 (PID: {pid}) - 完成 get_db(), 连接到数据库: {db_instance.name}")
        logger.info(f"绰号处理器进程 (PID: {pid}) 已获取数据库连接: {db_instance.name}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 将获取到的数据库实例传递给异步循环
        logger.info(f"子进程 (PID: {pid}) - 即将运行 _nickname_processing_loop")
        loop.run_until_complete(_nickname_processing_loop(queue, stop_event, db_instance))
        logger.info(f"子进程 (PID: {pid}) - 完成 _nickname_processing_loop")

    except Exception as e:
        logger.error(f"(PID: {pid}) 运行绰号处理器进程时出错: {e}", exc_info=True)
    finally:
        # --- 清理工作 ---
        if loop:
            try:
                # 关闭事件循环
                if loop.is_running():
                    loop.stop() # 先停止
                loop.close()
                logger.info(f"(PID: {pid}) asyncio 事件循环已关闭。")
            except Exception as loop_close_err:
                logger.error(f"(PID: {pid}) 关闭 asyncio 事件循环时出错: {loop_close_err}", exc_info=True)

        try:
            close_db()
            logger.info(f"(PID: {pid}) 数据库连接已通过 close_db() 关闭。")
        except Exception as db_close_err:
            logger.error(f"(PID: {pid}) 关闭数据库连接时出错: {db_close_err}", exc_info=True)
        logger.info(f"绰号处理器进程已结束 (PID: {pid})。")

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