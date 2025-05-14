from fastapi import APIRouter
from strawberry.fastapi import GraphQLRouter
import os
import sys

# from src.chat.heart_flow.heartflow import heartflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from src.config.config import BotConfig
from src.common.logger_manager import get_logger
from src.api.reload_config import reload_config as reload_config_func
from src.common.server import global_server
from src.api.apiforgui import (
    get_all_subheartflow_ids,
    forced_change_subheartflow_status,
    get_subheartflow_cycle_info,
    get_all_states,
)
from src.chat.heart_flow.sub_heartflow import ChatState
from src.api.basic_info_api import get_all_basic_info  # 新增导入

# import uvicorn
# import os


router = APIRouter()


logger = get_logger("api")

# maiapi = FastAPI()
logger.info("麦麦API服务器已启动")
graphql_router = GraphQLRouter(schema=None, path="/")  # Replace `None` with your actual schema

router.include_router(graphql_router, prefix="/graphql", tags=["GraphQL"])


@router.post("/config/reload")
async def reload_config():
    return await reload_config_func()


@router.get("/gui/subheartflow/get/all")
async def get_subheartflow_ids():
    """获取所有子心流的ID列表"""
    return await get_all_subheartflow_ids()


@router.post("/gui/subheartflow/forced_change_status")
async def forced_change_subheartflow_status_api(subheartflow_id: str, status: ChatState):  # noqa
    """强制改变子心流的状态"""
    # 参数检查
    if not isinstance(status, ChatState):
        logger.warning(f"无效的状态参数: {status}")
        return {"status": "failed", "reason": "invalid status"}
    logger.info(f"尝试将子心流 {subheartflow_id} 状态更改为 {status.value}")
    success = await forced_change_subheartflow_status(subheartflow_id, status)
    if success:
        logger.info(f"子心流 {subheartflow_id} 状态更改为 {status.value} 成功")
        return {"status": "success"}
    else:
        logger.error(f"子心流 {subheartflow_id} 状态更改为 {status.value} 失败")
        return {"status": "failed"}


@router.get("/stop")
async def force_stop_maibot():
    """强制停止MAI Bot"""
    from bot import request_shutdown

    success = await request_shutdown()
    if success:
        logger.info("MAI Bot已强制停止")
        return {"status": "success"}
    else:
        logger.error("MAI Bot强制停止失败")
        return {"status": "failed"}


@router.get("/gui/subheartflow/cycleinfo")
async def get_subheartflow_cycle_info_api(subheartflow_id: str, history_len: int):
    """获取子心流的循环信息"""
    cycle_info = await get_subheartflow_cycle_info(subheartflow_id, history_len)
    if cycle_info:
        return {"status": "success", "data": cycle_info}
    else:
        logger.warning(f"子心流 {subheartflow_id} 循环信息未找到")
        return {"status": "failed", "reason": "subheartflow not found"}


@router.get("/gui/get_all_states")
async def get_all_states_api():
    """获取所有状态"""
    all_states = await get_all_states()
    if all_states:
        return {"status": "success", "data": all_states}
    else:
        logger.warning("获取所有状态失败")
        return {"status": "failed", "reason": "failed to get all states"}


@router.get("/info")
async def get_system_basic_info():
    """获取系统基本信息"""
    logger.info("请求系统基本信息")
    try:
        info = get_all_basic_info()
        return {"status": "success", "data": info}
    except Exception as e:
        logger.error(f"获取系统基本信息失败: {e}")
        return {"status": "failed", "reason": str(e)}


def start_api_server():
    """启动API服务器"""
    global_server.register_router(router, prefix="/api/v1")
