from src.chat.heart_flow.observation.chatting_observation import ChattingObservation
from src.chat.heart_flow.observation.working_observation import WorkingObservation
from src.chat.heart_flow.observation.hfcloop_observation import HFCloopObservation
from src.chat.models.utils_model import LLMRequest
from src.config.config import global_config
from src.common.logger_manager import get_logger
from src.chat.utils.prompt_builder import Prompt
from datetime import datetime
from src.chat.memory_system.Hippocampus import HippocampusManager
from typing import List, Dict
import difflib


logger = get_logger("memory_activator")


def init_prompt():
    # --- Group Chat Prompt ---
    memory_activator_prompt = """
    你是一个记忆分析器，你需要根据以下信息来进行会议
    以下是一场聊天中的信息，请根据这些信息，总结出几个关键词作为记忆回忆的触发词
    
    {obs_info_text}
    
    请输出一个json格式，包含以下字段：
    {{
        "keywords": ["关键词1", "关键词2", "关键词3",......]
    }}
    不要输出其他多余内容，只输出json格式就好
    """

    Prompt(memory_activator_prompt, "memory_activator_prompt")


class MemoryActivator:
    def __init__(self):
        # TODO: API-Adapter修改标记
        self.summary_model = LLMRequest(
            model=global_config.model.summary, temperature=0.7, max_tokens=50, request_type="chat_observation"
        )
        self.running_memory = []

    async def activate_memory(self, observations) -> List[Dict]:
        """
        激活记忆

        Args:
            observations: 现有的进行观察后的 观察列表

        Returns:
            List[Dict]: 激活的记忆列表
        """
        obs_info_text = ""
        for observation in observations:
            if isinstance(observation, ChattingObservation):
                obs_info_text += observation.get_observe_info()
            elif isinstance(observation, WorkingObservation):
                working_info = observation.get_observe_info()
                for working_info_item in working_info:
                    obs_info_text += f"{working_info_item['type']}: {working_info_item['content']}\n"
            elif isinstance(observation, HFCloopObservation):
                obs_info_text += observation.get_observe_info()

        # prompt = await global_prompt_manager.format_prompt(
        #     "memory_activator_prompt",
        #     obs_info_text=obs_info_text,
        # )

        # logger.debug(f"prompt: {prompt}")

        # response = await self.summary_model.generate_response(prompt)

        # logger.debug(f"response: {response}")

        # # 只取response的第一个元素（字符串）
        # response_str = response[0]
        # keywords = list(get_keywords_from_json(response_str))

        # #调用记忆系统获取相关记忆
        # related_memory = await HippocampusManager.get_instance().get_memory_from_topic(
        #     valid_keywords=keywords, max_memory_num=3, max_memory_length=2, max_depth=3
        # )
        related_memory = await HippocampusManager.get_instance().get_memory_from_text(
            text=obs_info_text, max_memory_num=3, max_memory_length=2, max_depth=3, fast_retrieval=True
        )

        # logger.debug(f"获取到的记忆: {related_memory}")

        # 激活时，所有已有记忆的duration+1，达到3则移除
        for m in self.running_memory[:]:
            m["duration"] = m.get("duration", 1) + 1
        self.running_memory = [m for m in self.running_memory if m["duration"] < 3]

        if related_memory:
            for topic, memory in related_memory:
                # 检查是否已存在相同topic或相似内容（相似度>=0.7）的记忆
                exists = any(
                    m["topic"] == topic or difflib.SequenceMatcher(None, m["content"], memory).ratio() >= 0.7
                    for m in self.running_memory
                )
                if not exists:
                    self.running_memory.append(
                        {"topic": topic, "content": memory, "timestamp": datetime.now().isoformat(), "duration": 1}
                    )
                    logger.debug(f"添加新记忆: {topic} - {memory}")

        # 限制同时加载的记忆条数，最多保留最后3条
        if len(self.running_memory) > 3:
            self.running_memory = self.running_memory[-3:]

        return self.running_memory


init_prompt()
