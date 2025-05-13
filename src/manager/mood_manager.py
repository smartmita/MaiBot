import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

from ..config.config import global_config
from ..common.logger_manager import get_logger
from ..manager.async_task_manager import AsyncTask
from ..individuality.individuality import Individuality

logger = get_logger("mood")


@dataclass
class MoodState:
    valence: float
    """愉悦度 (-1.0 到 1.0)，-1表示极度负面，1表示极度正面"""
    arousal: float
    """唤醒度 (-1.0 到 1.0)，-1表示抑制，1表示兴奋"""
    text: str
    """心情的文本描述"""


@dataclass
class MoodChangeHistory:
    valence_direction_factor: int
    """愉悦度变化的系数（正为增益，负为抑制）"""
    arousal_direction_factor: int
    """唤醒度变化的系数（正为增益，负为抑制）"""


class MoodUpdateTask(AsyncTask):
    def __init__(self):
        super().__init__(
            task_name="Mood Update Task",
            wait_before_start=global_config.mood_update_interval,
            run_interval=global_config.mood_update_interval,
        )

        # 从配置文件获取衰减率
        self.decay_rate_valence: float = 1 - global_config.mood_decay_rate
        """愉悦度衰减率"""
        self.decay_rate_arousal: float = 1 - global_config.mood_decay_rate
        """唤醒度衰减率"""

        self.last_update = time.time()
        """上次更新时间"""

    async def run(self):
        current_time = time.time()
        time_diff = current_time - self.last_update
        agreeableness_factor = 1  # 宜人性系数
        agreeableness_bias = 0  # 宜人性偏置
        neuroticism_factor = 0.5  # 神经质系数
        # 获取人格特质
        personality = Individuality.get_instance().personality
        if personality:
            # 神经质：影响情绪变化速度
            neuroticism_factor = 1 + (personality.neuroticism - 0.5) * 0.4
            agreeableness_factor = 1 + (personality.agreeableness - 0.5) * 0.4

            # 宜人性：影响情绪基准线
            if personality.agreeableness < 0.2:
                agreeableness_bias = (personality.agreeableness - 0.2) * 0.5
            elif personality.agreeableness > 0.8:
                agreeableness_bias = (personality.agreeableness - 0.8) * 0.5
            else:
                agreeableness_bias = 0

        # 分别计算正向和负向的衰减率
        if mood_manager.current_mood.valence >= 0:
            # 正向情绪衰减
            decay_rate_positive = self.decay_rate_valence * (1 / agreeableness_factor)
            valence_target = 0 + agreeableness_bias
            new_valence = valence_target + (mood_manager.current_mood.valence - valence_target) * math.exp(
                -decay_rate_positive * time_diff * neuroticism_factor
            )
        else:
            # 负向情绪衰减
            decay_rate_negative = self.decay_rate_valence * agreeableness_factor
            valence_target = 0 + agreeableness_bias
            new_valence = valence_target + (mood_manager.current_mood.valence - valence_target) * math.exp(
                -decay_rate_negative * time_diff * neuroticism_factor
            )

        # Arousal 向中性（0）回归
        arousal_target = 0
        new_arousal = arousal_target + (mood_manager.current_mood.arousal - arousal_target) * math.exp(
            -self.decay_rate_arousal * time_diff * neuroticism_factor
        )

        mood_manager.set_current_mood(new_valence, new_arousal)

        self.last_update = current_time


class MoodPrintTask(AsyncTask):
    def __init__(self):
        super().__init__(
            task_name="Mood Print Task",
            wait_before_start=60,
            run_interval=60,
        )

    async def run(self):
        # 打印当前心情
        logger.info(
            f"愉悦度: {mood_manager.current_mood.valence:.2f}, "
            f"唤醒度: {mood_manager.current_mood.arousal:.2f}, "
            f"心情: {mood_manager.current_mood.text}"
        )


class MoodManager:
    # TODO: 改进，使用具有实验支持的新情绪模型

    EMOTION_FACTOR_MAP: Dict[str, Tuple[float, float]] = {
        "开心": (0.21, 0.6),
        "害羞": (0.15, 0.2),
        "愤怒": (-0.24, 0.8),
        "恐惧": (-0.21, 0.7),
        "悲伤": (-0.21, 0.3),
        "厌恶": (-0.12, 0.4),
        "惊讶": (0.06, 0.7),
        "困惑": (0.0, 0.6),
        "平静": (0.03, 0.5),
    }
    """
    情绪词映射表 {mood: (valence, arousal)}
    将情绪描述词映射到愉悦度和唤醒度的元组
    """

    EMOTION_POINT_MAP: Dict[Tuple[float, float], str] = {
        # 第一象限：高唤醒，正愉悦
        (0.5, 0.4): "兴奋",
        (0.3, 0.6): "快乐",
        (0.2, 0.3): "满足",
        # 第二象限：高唤醒，负愉悦
        (-0.5, 0.4): "愤怒",
        (-0.3, 0.6): "焦虑",
        (-0.2, 0.3): "烦躁",
        # 第三象限：低唤醒，负愉悦
        (-0.5, -0.4): "悲伤",
        (-0.3, -0.3): "疲倦",
        (-0.4, -0.7): "疲倦",
        # 第四象限：低唤醒，正愉悦
        (0.2, -0.1): "平静",
        (0.3, -0.2): "安宁",
        (0.5, -0.4): "放松",
    }
    """
    情绪文本映射表 {(valence, arousal): mood}
    将量化的情绪状态元组映射到文本描述
    """

    def __init__(self):
        self.current_mood = MoodState(
            valence=0.0,
            arousal=0.0,
            text="平静",
        )
        """当前情绪状态"""

        self.mood_change_history: MoodChangeHistory = MoodChangeHistory(
            valence_direction_factor=0,
            arousal_direction_factor=0,
        )
        """情绪变化历史"""

        self._lock = asyncio.Lock()
        """异步锁，用于保护线程安全"""

    def set_current_mood(self, new_valence: float, new_arousal: float):
        """
        设置当前情绪状态
        :param new_valence: 新的愉悦度
        :param new_arousal: 新的唤醒度
        """
        # 限制范围
        self.current_mood.valence = max(-1.0, min(new_valence, 1.0))
        self.current_mood.arousal = max(-1.0, min(new_arousal, 1.0))

        closest_mood = None
        min_distance = float("inf")

        for (v, a), text in self.EMOTION_POINT_MAP.items():
            # 计算当前情绪状态与每个情绪文本的欧氏距离
            distance = math.sqrt((self.current_mood.valence - v) ** 2 + (self.current_mood.arousal - a) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_mood = text

        if closest_mood:
            self.current_mood.text = closest_mood

    def update_current_mood(self, valence_delta: float, arousal_delta: float):
        """
        根据愉悦度和唤醒度变化量更新当前情绪状态
        :param valence_delta: 愉悦度变化量
        :param arousal_delta: 唤醒度变化量
        """
        # 计算连续增益/抑制
        # 规则：多次相同方向的变化会有更大的影响系数，反方向的变化会清零影响系数（系数的正负号由变化方向决定）
        if valence_delta * self.mood_change_history.valence_direction_factor > 0:
            # 如果方向相同，则根据变化方向改变系数
            if valence_delta > 0:
                self.mood_change_history.valence_direction_factor += 1  # 若为正向，则增加
            else:
                self.mood_change_history.valence_direction_factor -= 1  # 若为负向，则减少
        else:
            # 如果方向不同，则重置计数
            self.mood_change_history.valence_direction_factor = 0

        if arousal_delta * self.mood_change_history.arousal_direction_factor > 0:
            # 如果方向相同，则根据变化方向改变系数
            if arousal_delta > 0:
                self.mood_change_history.arousal_direction_factor += 1  # 若为正向，则增加计数
            else:
                self.mood_change_history.arousal_direction_factor -= 1  # 若为负向，则减少计数
        else:
            # 如果方向不同，则重置计数
            self.mood_change_history.arousal_direction_factor = 0

        # 计算增益/抑制的结果
        # 规则：如果当前情绪状态与变化方向相同，则增益；否则抑制
        if self.current_mood.valence * self.mood_change_history.valence_direction_factor > 0:
            valence_delta = valence_delta * (1.01 ** abs(self.mood_change_history.valence_direction_factor))
        else:
            valence_delta = valence_delta * (0.99 ** abs(self.mood_change_history.valence_direction_factor))

        if self.current_mood.arousal * self.mood_change_history.arousal_direction_factor > 0:
            arousal_delta = arousal_delta * (1.01 ** abs(self.mood_change_history.arousal_direction_factor))
        else:
            arousal_delta = arousal_delta * (0.99 ** abs(self.mood_change_history.arousal_direction_factor))

        self.set_current_mood(
            new_valence=self.current_mood.valence + valence_delta,
            new_arousal=self.current_mood.arousal + arousal_delta,
        )

    def get_mood_prompt(self) -> str:
        """
        根据当前情绪状态生成提示词
        """
        base_prompt = f"当前心情：{self.current_mood.text}。"

        # 根据情绪状态添加额外的提示信息
        if self.current_mood.valence > 0.5:
            base_prompt += "你现在心情很好，"
        elif self.current_mood.valence < -0.5:
            base_prompt += "你现在心情不太好，"

        if self.current_mood.arousal > 0.4:
            base_prompt += "情绪比较激动。"
        elif self.current_mood.arousal < -0.4:
            base_prompt += "情绪比较平静。"

        return base_prompt

    def get_arousal_multiplier(self) -> float:
        """
        根据当前情绪状态返回唤醒度乘数
        """
        if self.current_mood.arousal > 0.4:
            multiplier = 1 + min(0.15, (self.current_mood.arousal - 0.4) / 3)
            return multiplier
        elif self.current_mood.arousal < -0.4:
            multiplier = 1 - min(0.15, ((0 - self.current_mood.arousal) - 0.4) / 3)
            return multiplier
        return 1.0

    def update_mood_from_emotion(self, emotion: str, intensity: float = 1.0) -> None:
        """
        根据情绪词更新心情状态
        :param emotion: 情绪词（如'开心', '悲伤'等位于self.EMOTION_FACTOR_MAP中的键）
        :param intensity: 情绪强度（0.0-1.0）
        """
        if emotion not in self.EMOTION_FACTOR_MAP:
            logger.error(f"[情绪更新] 未知情绪词: {emotion}")
            return

        valence_change, arousal_change = self.EMOTION_FACTOR_MAP[emotion]
        old_valence = self.current_mood.valence
        old_arousal = self.current_mood.arousal
        old_mood = self.current_mood.text

        self.update_current_mood(valence_change, arousal_change)  # 更新当前情绪状态

        logger.info(
            f"[情绪变化] {emotion}(强度:{intensity:.2f}) | 愉悦度:{old_valence:.2f}->{self.current_mood.valence:.2f}, 唤醒度:{old_arousal:.2f}->{self.current_mood.arousal:.2f} | 心情:{old_mood}->{self.current_mood.text}"
        )


mood_manager = MoodManager()
"""全局情绪管理器"""
