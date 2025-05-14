from typing import List, Optional, Any
from src.chat.focus_chat.info.obs_info import ObsInfo
from src.chat.heart_flow.observation.observation import Observation
from src.chat.focus_chat.info.info_base import InfoBase
from .base_processor import BaseProcessor
from src.common.logger_manager import get_logger
from src.chat.heart_flow.observation.chatting_observation import ChattingObservation
from src.chat.heart_flow.observation.hfcloop_observation import HFCloopObservation
from src.chat.focus_chat.info.cycle_info import CycleInfo
from datetime import datetime
from typing import Dict
from src.chat.models.utils_model import LLMRequest
from src.config.config import global_config

logger = get_logger("observation")


class ChattingInfoProcessor(BaseProcessor):
    """观察处理器

    用于处理Observation对象，将其转换为ObsInfo对象。
    """

    def __init__(self):
        """初始化观察处理器"""
        self.llm_summary = LLMRequest(
            model=global_config.llm_observation, temperature=0.7, max_tokens=300, request_type="chat_observation"
        )
        super().__init__()

    async def process_info(
        self,
        observations: Optional[List[Observation]] = None,
        running_memorys: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[InfoBase]:
        """处理Observation对象

        Args:
            infos: InfoBase对象列表
            observations: 可选的Observation对象列表
            **kwargs: 其他可选参数

        Returns:
            List[InfoBase]: 处理后的ObsInfo实例列表
        """
        # print(f"observations: {observations}")
        processed_infos = []

        # 处理Observation对象
        if observations:
            for obs in observations:
                # print(f"obs: {obs}")
                if isinstance(obs, ChattingObservation):
                    obs_info = ObsInfo()

                    await self.chat_compress(obs)

                    # 设置说话消息
                    if hasattr(obs, "talking_message_str"):
                        obs_info.set_talking_message(obs.talking_message_str)

                    # 设置截断后的说话消息
                    if hasattr(obs, "talking_message_str_truncate"):
                        obs_info.set_talking_message_str_truncate(obs.talking_message_str_truncate)

                    if hasattr(obs, "mid_memory_info"):
                        obs_info.set_previous_chat_info(obs.mid_memory_info)

                    # 设置聊天类型
                    is_group_chat = obs.is_group_chat
                    if is_group_chat:
                        chat_type = "group"
                    else:
                        chat_type = "private"
                        obs_info.set_chat_target(obs.chat_target_info.get("person_name", "某人"))
                    obs_info.set_chat_type(chat_type)

                    # logger.debug(f"聊天信息处理器处理后的信息: {obs_info}")

                    processed_infos.append(obs_info)
                if isinstance(obs, HFCloopObservation):
                    obs_info = CycleInfo()
                    obs_info.set_observe_info(obs.observe_info)
                    processed_infos.append(obs_info)

        return processed_infos

    async def chat_compress(self, obs: ChattingObservation):
        if obs.compressor_prompt:
            try:
                summary_result, _, _ = await self.llm_summary.generate_response(obs.compressor_prompt)
                summary = "没有主题的闲聊"  # 默认值
                if summary_result:  # 确保结果不为空
                    summary = summary_result
            except Exception as e:
                logger.error(f"总结主题失败 for chat {obs.chat_id}: {e}")

            mid_memory = {
                "id": str(int(datetime.now().timestamp())),
                "theme": summary,
                "messages": obs.oldest_messages,  # 存储原始消息对象
                "readable_messages": obs.oldest_messages_str,
                # "timestamps": oldest_timestamps,
                "chat_id": obs.chat_id,
                "created_at": datetime.now().timestamp(),
            }

            obs.mid_memorys.append(mid_memory)
            if len(obs.mid_memorys) > obs.max_mid_memory_len:
                obs.mid_memorys.pop(0)  # 移除最旧的

            mid_memory_str = "之前聊天的内容概述是：\n"
            for mid_memory_item in obs.mid_memorys:  # 重命名循环变量以示区分
                time_diff = int((datetime.now().timestamp() - mid_memory_item["created_at"]) / 60)
                mid_memory_str += (
                    f"距离现在{time_diff}分钟前(聊天记录id:{mid_memory_item['id']})：{mid_memory_item['theme']}\n"
                )
            obs.mid_memory_info = mid_memory_str

            obs.compressor_prompt = ""
            obs.oldest_messages = []
            obs.oldest_messages_str = ""
