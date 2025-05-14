from src.chat.heart_flow.heartflow import heartflow
from src.chat.heart_flow.sub_heartflow import ChatState
from src.common.logger_manager import get_logger

logger = get_logger("api")


async def get_all_subheartflow_ids() -> list:
    """获取所有子心流的ID列表"""
    all_subheartflows = heartflow.subheartflow_manager.get_all_subheartflows()
    return [subheartflow.subheartflow_id for subheartflow in all_subheartflows]


async def forced_change_subheartflow_status(subheartflow_id: str, status: ChatState) -> bool:
    """强制改变子心流的状态"""
    subheartflow = await heartflow.get_or_create_subheartflow(subheartflow_id)
    if subheartflow:
        return await heartflow.force_change_subheartflow_status(subheartflow_id, status)
    return False


async def get_subheartflow_cycle_info(subheartflow_id: str, history_len: int) -> dict:
    """获取子心流的循环信息"""
    subheartflow_cycle_info = await heartflow.api_get_subheartflow_cycle_info(subheartflow_id, history_len)
    logger.debug(f"子心流 {subheartflow_id} 循环信息: {subheartflow_cycle_info}")
    if subheartflow_cycle_info:
        return subheartflow_cycle_info
    else:
        logger.warning(f"子心流 {subheartflow_id} 循环信息未找到")
        return None


async def get_all_states():
    """获取所有状态"""
    all_states = await heartflow.api_get_all_states()
    logger.debug(f"所有状态: {all_states}")
    return all_states
