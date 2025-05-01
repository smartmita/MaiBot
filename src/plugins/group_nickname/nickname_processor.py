import asyncio
import traceback
import threading
import queue
from typing import Dict, Optional
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, DuplicateKeyError  # 引入 DuplicateKeyError
from src.common.logger_manager import get_logger
from src.common.database import db  # 使用全局 db
from .nickname_mapper import analyze_chat_for_nicknames
from src.config.config import global_config

logger = get_logger("nickname_processor")

_stop_event = threading.Event()


def _upsert_person(collection: Collection, person_id: str, user_id_int: int, platform: str):
    """
    确保数据库中存在指定 person_id 的文档 (Upsert)。
    如果文档不存在，则使用提供的用户信息创建它。

    Args:
        collection: MongoDB 集合对象 (person_info)。
        person_id: 要查找或创建的 person_id。
        user_id_int: 用户的整数 ID。
        platform: 平台名称。

    Returns:
        UpdateResult: MongoDB 更新操作的结果。

    Raises:
        DuplicateKeyError: 如果发生重复键错误 (理论上不应由 upsert 触发)。
        Exception: 其他数据库操作错误。
    """
    try:
        # 关键步骤：基于 person_id 执行 Upsert
        # 如果文档不存在，它会被创建，并设置 $setOnInsert 中的字段。
        # 如果文档已存在，此操作不会修改任何内容。
        result = collection.update_one(
            {"person_id": person_id},
            {
                "$setOnInsert": {
                    "person_id": person_id,
                    "user_id": user_id_int,  # 确保这里使用传入的 user_id_int
                    "platform": platform,
                    "group_nicknames": [],  # 初始化 group_nicknames 数组
                }
            },
            upsert=True,
        )
        if result.upserted_id:
            logger.debug(f"Upsert on person_id created new document: {person_id}")
        # else:
        #    logger.debug(f"Upsert on person_id found existing document: {person_id}")
        return result
    except DuplicateKeyError as dk_err:
        # 这个错误理论上不应该再由 upsert 触发。
        # 如果仍然出现，可能指示 person_id 生成逻辑问题或非常罕见的 MongoDB 内部情况。
        logger.error(
            f"数据库操作失败 (DuplicateKeyError): person_id {person_id}. 错误: {dk_err}. 这不应该发生，请检查 person_id 生成逻辑和数据库状态。"
        )
        raise  # 将异常向上抛出，让调用者处理
    except Exception as e:
        logger.exception(f"对 person_id {person_id} 执行 Upsert 时失败: {e}")
        raise  # 将异常向上抛出


def _update_group_nickname(collection: Collection, person_id: str, group_id_str: str, nickname: str):
    """
    尝试更新 person_id 文档中特定群组的绰号计数，或添加新条目。
    按顺序尝试：增加计数 -> 添加绰号 -> 添加群组。

    Args:
        collection: MongoDB 集合对象 (person_info)。
        person_id: 目标文档的 person_id。
        group_id_str: 目标群组的 ID (字符串)。
        nickname: 要更新或添加的绰号。
    """
    # 3a. 尝试增加现有群组中现有绰号的计数
    result_inc = collection.update_one(
        {
            "person_id": person_id,
            "group_nicknames": {"$elemMatch": {"group_id": group_id_str, "nicknames.name": nickname}},
        },
        {"$inc": {"group_nicknames.$[group].nicknames.$[nick].count": 1}},
        array_filters=[
            {"group.group_id": group_id_str},
            {"nick.name": nickname},
        ],
    )
    if result_inc.modified_count > 0:
        # logger.debug(f"成功增加 person_id {person_id} 在群组 {group_id_str} 中绰号 '{nickname}' 的计数。")
        return  # 成功增加计数，操作完成

    # 3b. 如果上一步未修改 (绰号不存在于该群组)，尝试将新绰号添加到现有群组
    result_push_nick = collection.update_one(
        {
            "person_id": person_id,
            "group_nicknames.group_id": group_id_str,  # 检查群组是否存在
        },
        {"$push": {"group_nicknames.$[group].nicknames": {"name": nickname, "count": 1}}},
        array_filters=[{"group.group_id": group_id_str}],
    )
    if result_push_nick.modified_count > 0:
        logger.debug(f"成功为 person_id {person_id} 在现有群组 {group_id_str} 中添加新绰号 '{nickname}'。")
        return  # 成功添加绰号，操作完成

    # 3c. 如果上一步也未修改 (群组条目本身不存在)，则添加新的群组条目和绰号
    # 确保 group_nicknames 数组存在 (作为保险措施)
    collection.update_one(
        {"person_id": person_id, "group_nicknames": {"$exists": False}},
        {"$set": {"group_nicknames": []}},
    )
    # 推送新的群组对象到 group_nicknames 数组
    result_push_group = collection.update_one(
        {
            "person_id": person_id,
            "group_nicknames.group_id": {"$ne": group_id_str},  # 确保该群组 ID 尚未存在
        },
        {
            "$push": {
                "group_nicknames": {
                    "group_id": group_id_str,
                    "nicknames": [{"name": nickname, "count": 1}],
                }
            }
        },
    )
    if result_push_group.modified_count > 0:
        logger.debug(f"为 person_id {person_id} 添加了新的群组 {group_id_str} 和绰号 '{nickname}'。")
    # else:
    # 如果连添加群组也失败 (例如 group_id 已存在但之前的步骤都未匹配，理论上不太可能)，
    # 可能需要进一步的日志或错误处理，但这通常意味着数据状态异常。
    # logger.warning(f"尝试为 person_id {person_id} 添加新群组 {group_id_str} 失败，可能群组已存在但结构不符合预期。")


async def update_nickname_counts(platform: str, group_id: str, nickname_map: Dict[str, str]):
    """
    更新数据库中用户的群组绰号计数 (使用全局 db)。
    通过调用辅助函数来处理 person 文档的 upsert 和绰号更新。

    Args:
        platform (str): 平台名称 (e.g., 'qq')。
        group_id (str): 群组 ID。
        nickname_map (Dict[str, str]): 用户 ID (字符串) 到绰号的映射。
    """
    try:
        from ..person_info.person_info import person_info_manager
    except ImportError:
        logger.error("无法导入 person_info_manager，无法生成 person_id！")
        return

    person_info_collection = db.person_info
    if not nickname_map:
        logger.debug("提供的用于更新的绰号映射为空。")
        return

    logger.info(f"尝试更新平台 '{platform}' 群组 '{group_id}' 的绰号计数，映射为: {nickname_map}")

    for user_id_str, nickname in nickname_map.items():
        # --- 基本验证 ---
        if not user_id_str or not nickname:
            logger.warning(f"跳过无效条目: user_id='{user_id_str}', nickname='{nickname}'")
            continue
        group_id_str = str(group_id)
        try:
            user_id_int = int(user_id_str)
        except ValueError:
            logger.warning(f"无效的用户ID格式: '{user_id_str}'，跳过。")
            continue
        # --- 结束验证 ---

        try:
            # --- 步骤 1: 生成 person_id ---
            person_id = person_info_manager.get_person_id(platform, user_id_str)
            if not person_id:
                logger.error(f"无法为 platform='{platform}', user_id='{user_id_str}' 生成 person_id，跳过此用户。")
                continue

            # --- 步骤 2: 确保 Person 文档存在 (调用辅助函数) ---
            _upsert_person(person_info_collection, person_id, user_id_int, platform)

            # --- 步骤 3: 更新群组绰号 (调用辅助函数) ---
            _update_group_nickname(person_info_collection, person_id, group_id_str, nickname)

        # --- 统一处理数据库操作可能抛出的异常 ---
        except (OperationFailure, DuplicateKeyError) as db_err:  # 捕获特定的数据库错误
            logger.exception(
                f"数据库操作失败 ({type(db_err).__name__}): 用户 {user_id_str}, 群组 {group_id_str}, 绰号 {nickname}. 错误: {db_err}"
            )
        except Exception as e:
            # 捕获其他所有可能的错误 (例如 person_id 生成、辅助函数内部未捕获的错误等)
            logger.exception(f"处理用户 {user_id_str} 的绰号 '{nickname}' 时发生意外错误：{e}")


# --- 使用 queue.Queue ---
queue_max_size = getattr(global_config, "NICKNAME_QUEUE_MAX_SIZE", 100)
nickname_queue: queue.Queue = queue.Queue(maxsize=queue_max_size)

_nickname_thread: Optional[threading.Thread] = None


# --- add_to_nickname_queue (保持不变，已包含 platform) ---
async def add_to_nickname_queue(
    chat_history_str: str, bot_reply: str, platform: str, group_id: Optional[str], user_name_map: Dict[str, str]
):
    """将需要分析的数据放入队列。"""
    if not global_config or not global_config.ENABLE_NICKNAME_MAPPING:
        return
    if group_id is None:
        logger.debug("私聊跳过绰号映射。")
        return
    try:
        item = (chat_history_str, bot_reply, platform, str(group_id), user_name_map)
        nickname_queue.put_nowait(item)
        logger.debug(
            f"已将项目添加到平台 '{platform}' 群组 '{group_id}' 的绰号队列。当前大小: {nickname_queue.qsize()}"
        )
    except queue.Full:
        logger.warning(f"无法将项目添加到绰号队列：队列已满 (maxsize={nickname_queue.maxsize})。")
    except Exception as e:
        logger.warning(f"无法将项目添加到绰号队列: {e}", exc_info=True)


# --- _nickname_processing_loop (保持不变，已包含 platform) ---
async def _nickname_processing_loop(q: queue.Queue, stop_event: threading.Event):
    """独立线程中的主循环，处理队列任务 (使用全局 db 和 config)。"""
    thread_id = threading.get_ident()
    logger.info(f"绰号处理循环已启动 (线程 ID: {thread_id})。")
    sleep_interval = getattr(global_config, "NICKNAME_PROCESS_SLEEP_INTERVAL", 0.5)

    while not stop_event.is_set():
        try:
            item = q.get(block=True, timeout=sleep_interval)

            if isinstance(item, tuple) and len(item) == 5:
                chat_history_str, bot_reply, platform, group_id, user_name_map = item
                logger.debug(f"(线程 ID: {thread_id}) 正在处理平台 '{platform}' 群组 '{group_id}' 的绰号映射任务...")
                analysis_result = await analyze_chat_for_nicknames(chat_history_str, bot_reply, user_name_map)
                if analysis_result.get("is_exist") and analysis_result.get("data"):
                    await update_nickname_counts(platform, group_id, analysis_result["data"])
            else:
                logger.warning(f"(线程 ID: {thread_id}) 从队列接收到意外的项目类型或长度: {type(item)}, 内容: {item}")

            q.task_done()

        except queue.Empty:
            continue
        except asyncio.CancelledError:
            logger.info(f"绰号处理循环已取消 (线程 ID: {thread_id})。")
            break
        except Exception as e:
            logger.error(f"(线程 ID: {thread_id}) 绰号处理循环出错: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5)

    logger.info(f"绰号处理循环已结束 (线程 ID: {thread_id})。")


# --- _run_processor_thread (保持不变) ---
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
    except Exception as e:
        logger.error(f"(Thread ID: {thread_id}) Error running nickname processor thread: {e}", exc_info=True)
    finally:
        if loop:
            try:
                if loop.is_running():
                    logger.info(f"(Thread ID: {thread_id}) Stopping the asyncio loop...")
                    all_tasks = asyncio.all_tasks(loop)
                    if all_tasks:
                        logger.info(f"(Thread ID: {thread_id}) Cancelling {len(all_tasks)} running tasks...")
                        for task in all_tasks:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                        logger.info(f"(Thread ID: {thread_id}) All tasks cancelled.")
                    loop.stop()
                    logger.info(f"(Thread ID: {thread_id}) Loop stopped.")
                if not loop.is_closed():
                    loop.close()
                    logger.info(f"(Thread ID: {thread_id}) Asyncio loop closed.")
            except Exception as loop_close_err:
                logger.error(f"(Thread ID: {thread_id}) Error closing loop: {loop_close_err}", exc_info=True)
        logger.info(f"Nickname processor thread finished (Thread ID: {thread_id}).")


# --- start_nickname_processor (保持不变) ---
def start_nickname_processor():
    """启动绰号映射处理线程。"""
    global _nickname_thread
    if not global_config or not global_config.ENABLE_NICKNAME_MAPPING:
        logger.info("绰号映射功能已禁用或无法获取配置。处理器未启动。")
        return

    if _nickname_thread is None or not _nickname_thread.is_alive():
        logger.info("正在启动绰号处理器线程...")
        stop_event = get_stop_event()
        stop_event.clear()
        _nickname_thread = threading.Thread(
            target=_run_processor_thread, args=(nickname_queue, stop_event), daemon=True
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
            if _nickname_thread.is_alive():
                logger.warning("绰号处理器线程在 10 秒后未结束。")
        except Exception as e:
            logger.error(f"停止绰号处理器线程时出错: {e}", exc_info=True)
        finally:
            if _nickname_thread and not _nickname_thread.is_alive():
                logger.info("绰号处理器线程已成功停止。")
            else:
                logger.warning("停止绰号处理器线程：线程可能仍在运行或未正确清理。")
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
