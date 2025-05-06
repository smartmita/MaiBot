from fastapi import APIRouter
from strawberry.fastapi import GraphQLRouter

# from src.config.config import BotConfig
from src.common.logger_manager import get_logger
from src.api.reload_config import reload_config as reload_config_func
from src.common.server import global_server
from .apiforgui import get_all_subheartflow_ids, forced_change_subheartflow_status
from src.heart_flow.sub_heartflow import ChatState
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


def start_api_server():
    """启动API服务器"""
    global_server.register_router(router, prefix="/api/v1")
