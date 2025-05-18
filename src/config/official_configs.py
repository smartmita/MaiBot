from dataclasses import dataclass, field
from typing import Any

from src.config.config_base import ConfigBase

"""
须知：
1. 本文件中记录了所有的配置项
2. 所有新增的class都需要继承自ConfigBase
3. 所有新增的class都应在config.py中的Config类中添加字段
4. 对于新增的字段，若为可选项，则应在其后添加field()并设置default_factory或default
"""


@dataclass
class BotConfig(ConfigBase):
    """QQ机器人配置类"""

    qq_account: str
    """QQ账号"""

    nickname: str
    """昵称"""

    alias_names: list[str] = field(default_factory=lambda: [])
    """别名列表"""


@dataclass
class ChatTargetConfig(ConfigBase):
    """
    聊天目标配置类
    此类中有聊天的群组和用户配置
    """

    talk_allowed_groups: set[str] = field(default_factory=lambda: set())
    """允许聊天的群组列表"""

    talk_frequency_down_groups: set[str] = field(default_factory=lambda: set())
    """降低聊天频率的群组列表"""

    ban_user_id: set[str] = field(default_factory=lambda: set())
    """禁止聊天的用户列表"""


@dataclass
class PersonalityConfig(ConfigBase):
    """人格配置类"""

    personality_core: str
    """核心人格"""

    expression_style: str
    """表达风格"""

    enable_expression_learner: bool = True
    """是否启用新发言习惯注入，关闭则启用旧方法"""

    personality_sides: list[str] = field(default_factory=lambda: [])
    """人格侧写"""

    personality_detail_level: int = 0
    """人设消息注入 prompt 详细等级 (0: 采用默认配置, 1: 核心/随机细节, 2: 核心+随机侧面/全部细节, 3: 全部)"""

@dataclass
class IdentityConfig(ConfigBase):
    """个体特征配置类"""

    height: int = 170
    """身高（单位：厘米）"""

    weight: float = 50.0
    """体重（单位：千克）"""

    age: int = 18
    """年龄（单位：岁）"""

    gender: str = "女"
    """性别（男/女）"""

    appearance: str = "可爱"
    """外貌描述"""

    identity_detail: list[str] = field(default_factory=lambda: [])
    """身份特征"""


@dataclass
class PlatformsConfig(ConfigBase):
    """平台配置类"""

    qq: str
    """QQ适配器连接URL配置"""


@dataclass
class ChatConfig(ConfigBase):
    """聊天配置类"""

    allow_focus_mode: bool = True
    """是否允许专注聊天状态"""

    base_normal_chat_num: int = 3
    """最多允许多少个群进行普通聊天"""

    base_focused_chat_num: int = 2
    """最多允许多少个群进行专注聊天"""

    observation_context_size: int = 12
    """可观察到的最长上下文大小，超过这个值的上下文会被压缩"""

    message_buffer: bool = True
    """消息缓冲器"""

    ban_words: set[str] = field(default_factory=lambda: set())
    """过滤词列表"""

    ban_msgs_regex: set[str] = field(default_factory=lambda: set())
    """过滤正则表达式列表"""

    allow_remove_duplicates: bool = True
    """是否开启心流去重（如果发现心流截断问题严重可尝试关闭）"""


@dataclass
class NormalChatConfig(ConfigBase):
    """普通聊天配置类"""

    reasoning_model_probability: float = 0.3
    """
    发言时选择推理模型的概率（0-1之间）
    选择普通模型的概率为 1 - reasoning_model_probability
    """

    emoji_chance: float = 0.2
    """发送表情包的基础概率"""

    thinking_timeout: int = 120
    """最长思考时间"""

    willing_mode: str = "classical"
    """意愿模式"""

    response_willing_amplifier: float = 1.0
    """回复意愿放大系数"""

    response_interested_rate_amplifier: float = 1.0
    """回复兴趣度放大系数"""

    down_frequency_rate: float = 3.0
    """降低回复频率的群组回复意愿降低系数"""

    emoji_response_penalty: float = 0.0
    """表情包回复惩罚系数"""

    mentioned_bot_inevitable_reply: bool = False
    """提及 bot 必然回复"""

    at_bot_inevitable_reply: bool = False
    """@bot 必然回复"""


@dataclass
class FocusChatConfig(ConfigBase):
    """专注聊天配置类"""

    reply_trigger_threshold: float = 3.0
    """心流聊天触发阈值，越低越容易触发"""

    default_decay_rate_per_second: float = 0.98
    """默认衰减率，越大衰减越快"""

    consecutive_no_reply_threshold: int = 3
    """连续不回复的次数阈值"""

    compressed_length: int = 5
    """心流上下文压缩的最短压缩长度，超过心流观察到的上下文长度，会压缩，最短压缩长度为5"""

    compress_length_limit: int = 5
    """最多压缩份数，超过该数值的压缩上下文会被删除"""


@dataclass
class EmojiConfig(ConfigBase):
    """表情包配置类"""

    max_reg_num: int = 200
    """表情包最大注册数量"""

    do_replace: bool = True
    """达到最大注册数量时替换旧表情包"""

    check_interval: int = 120
    """表情包检查间隔（分钟）"""

    save_pic: bool = False
    """是否保存图片"""

    cache_emoji: bool = True
    """是否缓存表情包"""

    steal_emoji: bool = True
    """是否偷取表情包，让{global_config.bot.nickname}可以发送她保存的这些表情包"""

    content_filtration: bool = False
    """是否开启表情包过滤"""

    filtration_prompt: str = "符合公序良俗"
    """表情包过滤要求"""


@dataclass
class MemoryConfig(ConfigBase):
    """记忆配置类"""

    memory_build_interval: int = 600
    """记忆构建间隔（秒）"""

    memory_build_distribution: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ] = field(default_factory=lambda: (6.0, 3.0, 0.6, 32.0, 12.0, 0.4))
    """记忆构建分布，参数：分布1均值，标准差，权重，分布2均值，标准差，权重"""

    memory_build_sample_num: int = 8
    """记忆构建采样数量"""

    memory_build_sample_length: int = 40
    """记忆构建采样长度"""

    memory_compress_rate: float = 0.1
    """记忆压缩率"""

    forget_memory_interval: int = 1000
    """记忆遗忘间隔（秒）"""

    memory_forget_time: int = 24
    """记忆遗忘时间（小时）"""

    memory_forget_percentage: float = 0.01
    """记忆遗忘比例"""

    consolidate_memory_interval: int = 1000
    """记忆整合间隔（秒）"""

    consolidation_similarity_threshold: float = 0.7
    """整合相似度阈值"""

    consolidate_memory_percentage: float = 0.01
    """整合检查节点比例"""

    memory_ban_words: list[str] = field(default_factory=lambda: ["表情包", "图片", "回复", "聊天记录"])
    """不允许记忆的词列表"""

    long_message_auto_truncate: bool = True
    """HFC 模式过长消息自动截断"""


@dataclass
class MoodConfig(ConfigBase):
    """情绪配置类"""

    mood_update_interval: int = 1
    """情绪更新间隔（秒）"""

    mood_decay_rate: float = 0.95
    """情绪衰减率"""

    mood_intensity_factor: float = 0.7
    """情绪强度因子"""


@dataclass
class KeywordRuleConfig(ConfigBase):
    """关键词规则配置类"""

    enable: bool = True
    """是否启用关键词规则"""

    keywords: list[str] = field(default_factory=lambda: [])
    """关键词列表"""

    regex: list[str] = field(default_factory=lambda: [])
    """正则表达式列表"""

    reaction: str = ""
    """关键词触发的反应"""


@dataclass
class KeywordReactionConfig(ConfigBase):
    """关键词配置类"""

    enable: bool = True
    """是否启用关键词反应"""

    rules: list[KeywordRuleConfig] = field(default_factory=lambda: [])
    """关键词反应规则列表"""


@dataclass
class ChineseTypoConfig(ConfigBase):
    """中文错别字配置类"""

    enable: bool = True
    """是否启用中文错别字生成器"""

    error_rate: float = 0.01
    """单字替换概率"""

    min_freq: int = 9
    """最小字频阈值"""

    tone_error_rate: float = 0.1
    """声调错误概率"""

    word_replace_rate: float = 0.006
    """整词替换概率"""


@dataclass
class ResponseSplitterConfig(ConfigBase):
    """回复分割器配置类"""

    enable: bool = True
    """是否启用回复分割器"""

    max_length: int = 256
    """回复允许的最大长度"""

    max_sentence_num: int = 3
    """回复允许的最大句子数"""

    enable_kaomoji_protection: bool = False
    """是否启用颜文字保护"""


@dataclass
class TelemetryConfig(ConfigBase):
    """遥测配置类"""

    enable: bool = True
    """是否启用遥测"""


@dataclass
class ExperimentalConfig(ConfigBase):
    """实验功能配置类"""

    enable_friend_chat: bool = False
    """是否启用好友聊天"""

    talk_allowed_private: set[str] = field(default_factory=lambda: set())
    """允许聊天的私聊列表"""

    enable_Legacy_HFC: bool = False
    """是否启用 Legacy_HFC 处理器"""

    enable_friend_whitelist: bool = True
    """是否启用好友白名单"""

    rename_person: bool = True
    """是否启用改名工具"""

    api_polling_max_retries: int = 3
    """API轮询最大重试次数"""

    enable_always_relative_history: bool = False
    """聊天记录总是使用 relative 模式"""

    name_display_mode: int = 1
    """
    聊天记录用户名称显示模式。
    1: 老模式 (LLM名称 > '昵称：'+群名片 > QQ昵称)。
    2: 优先显示群名片 (群名片 > LLM名称 > QQ昵称)。
    """


@dataclass
class ScheduleConfig(ConfigBase):
    """日程配置类"""

    enable: bool = False
    """是否启用日程生成"""

    prompt_schedule_gen: str = "无日程"
    """日程生成提示"""

    schedule_doing_update_interval: int = 300
    """日程表更新间隔 单位秒"""

    schedule_temperature: float = 0.5
    """日程表温度，建议0.5-1.0"""

    time_zone: str = "Asia/Shanghai"
    """时区"""


@dataclass
class GroupNicknameConfig(ConfigBase):
    """绰号处理系统配置类"""

    enable_nickname_mapping: bool = False
    """绰号映射功能总开关"""

    max_nicknames_in_prompt: int = 10
    """Prompt 中最多注入的绰号数量"""

    nickname_probability_smoothing: int = 1
    """绰号加权随机选择的平滑因子"""

    nickname_queue_max_size: int = 100
    """绰号处理队列最大容量"""

    nickname_process_sleep_interval: float = 5.0
    """绰号处理进程休眠间隔（秒）"""

    nickname_analysis_history_limit: int = 30
    """绰号处理可见最大上下文"""

    nickname_analysis_probability: float = 0.1
    """绰号随机概率命中"""


@dataclass
class PFCConfig(ConfigBase):
    """PFC配置类"""

    enable: bool = False
    """是否启用PFC"""

    pfc_message_buffer_size: int = 2
    """PFC 聊天消息缓冲数量"""

    pfc_recent_history_display_count: int = 18
    """PFC 对话最大可见上下文"""

    enable_pfc_reply_checker: bool = True
    """是否启用 PFC 的回复检查器"""

    pfc_max_reply_attempts: int = 3
    """发言最多尝试次数"""

    pfc_max_chat_history_for_checker: int = 30
    """checker聊天记录最大可见上文长度"""

    pfc_emotion_update_intensity: float = 0.6
    """情绪更新强度"""

    pfc_emotion_history_count: int = 5
    """情绪更新最大可见上下文长度"""

    pfc_relationship_incremental_interval: int = 10
    """关系值增值强度"""

    pfc_relationship_incremental_msg_count: int = 10
    """会话中，关系值判断最大可见上下文"""

    pfc_relationship_incremental_default_change: float = 1.0
    """会话中，关系值默认更新值"""

    pfc_relationship_incremental_max_change: float = 5.0
    """会话中，关系值最大可变值"""

    pfc_relationship_final_msg_count: int = 30
    """会话结束时，关系值判断最大可见上下文"""

    pfc_relationship_final_default_change: float = 5.0
    """会话结束时，关系值默认更新值"""

    pfc_relationship_final_max_change: float = 50.0
    """会话结束时，关系值最大可变值"""

    pfc_historical_fallback_exclude_seconds: int = 45
    """pfc 翻看聊天记录排除最近时长"""

    enable_idle_chat: bool = True
    """是否启用 pfc 主动发言"""

    idle_check_interval: int = 10
    """主动发言检查间隔（分钟）"""

    min_cooldown: int = 7200
    """主动发言最短冷却时间（秒）"""

    max_cooldown: int = 18000
    """主动发言最长冷却时间（秒）"""


@dataclass
class ModelConfig(ConfigBase):
    """模型配置类"""

    model_max_output_length: int = 800
    """最大回复长度"""

    reasoning: dict[str, Any] = field(default_factory=lambda: {})
    """推理模型配置"""

    normal: dict[str, Any] = field(default_factory=lambda: {})
    """普通模型配置"""

    topic_judge: dict[str, Any] = field(default_factory=lambda: {})
    """主题判断模型配置"""

    summary: dict[str, Any] = field(default_factory=lambda: {})
    """摘要模型配置"""

    vlm: dict[str, Any] = field(default_factory=lambda: {})
    """视觉语言模型配置"""

    heartflow: dict[str, Any] = field(default_factory=lambda: {})
    """心流模型配置"""

    observation: dict[str, Any] = field(default_factory=lambda: {})
    """观察模型配置"""

    sub_heartflow: dict[str, Any] = field(default_factory=lambda: {})
    """子心流模型配置"""

    plan: dict[str, Any] = field(default_factory=lambda: {})
    """计划模型配置"""

    embedding: dict[str, Any] = field(default_factory=lambda: {})
    """嵌入模型配置"""

    pfc_action_planner: dict[str, Any] = field(default_factory=lambda: {})
    """PFC动作规划模型配置"""

    pfc_chat: dict[str, Any] = field(default_factory=lambda: {})
    """PFC聊天模型配置"""

    # pfc_reply_checker: dict[str, Any] = field(default_factory=lambda: {})
    # """PFC回复检查模型配置"""

    tool_use: dict[str, Any] = field(default_factory=lambda: {})
    """工具使用模型配置"""

    nickname_mapping: dict[str, Any] = field(default_factory=lambda: {})
    """绰号映射LLM配置"""

    scheduler_all: dict[str, Any] = field(default_factory=lambda: {})
    """全局日程LLM配置"""

    scheduler_doing: dict[str, Any] = field(default_factory=lambda: {})
    """当前活动日程LLM配置"""

    PFC_relationship_eval: dict[str, Any] = field(default_factory=lambda: {})
    """PFC关系评估LLM配置"""
