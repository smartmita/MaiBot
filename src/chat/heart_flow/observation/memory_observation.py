from src.chat.heart_flow.observation.observation import Observation
from datetime import datetime
from src.common.logger_manager import get_logger
import traceback

# Import the new utility function
from src.chat.memory_system.Hippocampus import HippocampusManager
import jieba
from typing import List

logger = get_logger("memory")


class MemoryObservation(Observation):
    def __init__(self, observe_id):
        super().__init__(observe_id)
        self.observe_info: str = ""
        self.context: str = ""
        self.running_memory: List[dict] = []

    def get_observe_info(self):
        for memory in self.running_memory:
            self.observe_info += f"{memory['topic']}:{memory['content']}\n"
        return self.observe_info

    async def observe(self):
        # ---------- 2. 获取记忆 ----------
        try:
            # 从聊天内容中提取关键词
            chat_words = set(jieba.cut(self.context))
            # 过滤掉停用词和单字词
            keywords = [word for word in chat_words if len(word) > 1]
            # 去重并限制数量
            keywords = list(set(keywords))[:5]

            logger.debug(f"取的关键词: {keywords}")

            # 调用记忆系统获取相关记忆
            related_memory = await HippocampusManager.get_instance().get_memory_from_topic(
                valid_keywords=keywords, max_memory_num=3, max_memory_length=2, max_depth=3
            )

            logger.debug(f"获取到的记忆: {related_memory}")

            if related_memory:
                for topic, memory in related_memory:
                    # 将记忆添加到 running_memory
                    self.running_memory.append(
                        {"topic": topic, "content": memory, "timestamp": datetime.now().isoformat()}
                    )
                    logger.debug(f"添加新记忆: {topic} - {memory}")

        except Exception as e:
            logger.error(f"观察 记忆时出错: {e}")
            logger.error(traceback.format_exc())
