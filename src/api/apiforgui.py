from src.heart_flow.heartflow import heartflow
from src.heart_flow.sub_heartflow import ChatState


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
