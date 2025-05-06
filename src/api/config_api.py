from typing import List, Optional, Dict, Any
import strawberry
from packaging.version import Version
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@strawberry.type
class APIBotConfig:
    """机器人配置类"""

    INNER_VERSION: Version  # 配置文件内部版本号
    MAI_VERSION: str  # 硬编码的版本信息

    # bot
    BOT_QQ: Optional[int]  # 机器人QQ号
    BOT_NICKNAME: Optional[str]  # 机器人昵称
    BOT_ALIAS_NAMES: List[str]  # 机器人别名列表

    # group
    talk_allowed_groups: List[int]  # 允许回复消息的群号列表
    talk_frequency_down_groups: List[int]  # 降低回复频率的群号列表
    ban_user_id: List[int]  # 禁止回复和读取消息的QQ号列表

    # personality
    personality_core: str  # 人格核心特点描述
    personality_sides: List[str]  # 人格细节描述列表

    # identity
    identity_detail: List[str]  # 身份特点列表
    age: int  # 年龄（岁）
    gender: str  # 性别
    appearance: str  # 外貌特征描述

    # schedule
    ENABLE_SCHEDULE_GEN: bool  # 是否启用日程生成
    ENABLE_SCHEDULE_INTERACTION: bool  # 是否启用日程交互
    PROMPT_SCHEDULE_GEN: str  # 日程生成提示词
    SCHEDULE_DOING_UPDATE_INTERVAL: int  # 日程进行中更新间隔
    SCHEDULE_TEMPERATURE: float  # 日程生成温度
    TIME_ZONE: str  # 时区

    # platforms
    platforms: Dict[str, str]  # 平台信息

    # chat
    allow_focus_mode: bool  # 是否允许专注模式
    base_normal_chat_num: int  # 基础普通聊天次数
    base_focused_chat_num: int  # 基础专注聊天次数
    observation_context_size: int  # 观察上下文大小
    message_buffer: bool  # 是否启用消息缓冲
    ban_words: List[str]  # 禁止词列表
    ban_msgs_regex: List[str]  # 禁止消息的正则表达式列表

    # normal_chat
    MODEL_R1_PROBABILITY: float  # 模型推理概率
    MODEL_V3_PROBABILITY: float  # 模型普通概率
    emoji_chance: float  # 表情符号出现概率
    thinking_timeout: int  # 思考超时时间
    willing_mode: str  # 意愿模式
    response_willing_amplifier: float  # 回复意愿放大器
    response_interested_rate_amplifier: float  # 回复兴趣率放大器
    down_frequency_rate: float  # 降低频率率
    emoji_response_penalty: float  # 表情回复惩罚
    mentioned_bot_inevitable_reply: bool  # 提到机器人时是否必定回复
    at_bot_inevitable_reply: bool  # @机器人时是否必定回复

    # focus_chat
    reply_trigger_threshold: float  # 回复触发阈值
    default_decay_rate_per_second: float  # 默认每秒衰减率
    consecutive_no_reply_threshold: int  # 连续不回复阈值

    # compressed
    compressed_length: int  # 压缩长度
    compress_length_limit: int  # 压缩长度限制

    # emoji
    max_emoji_num: int  # 最大表情符号数量
    max_reach_deletion: bool  # 达到最大数量时是否删除
    EMOJI_CHECK_INTERVAL: int  # 表情检查间隔
    EMOJI_REGISTER_INTERVAL: Optional[int]  # 表情注册间隔（兼容性保留）
    EMOJI_SAVE: bool  # 是否保存表情
    EMOJI_CHECK: bool  # 是否检查表情
    EMOJI_CHECK_PROMPT: str  # 表情检查提示词

    # memory
    build_memory_interval: int  # 构建记忆间隔
    memory_build_distribution: List[float]  # 记忆构建分布
    build_memory_sample_num: int  # 构建记忆样本数量
    build_memory_sample_length: int  # 构建记忆样本长度
    memory_compress_rate: float  # 记忆压缩率
    forget_memory_interval: int  # 忘记记忆间隔
    memory_forget_time: int  # 记忆忘记时间
    memory_forget_percentage: float  # 记忆忘记百分比
    consolidate_memory_interval: int  # 巩固记忆间隔
    consolidation_similarity_threshold: float  # 巩固相似度阈值
    consolidation_check_percentage: float  # 巩固检查百分比
    memory_ban_words: List[str]  # 记忆禁止词列表

    # mood
    mood_update_interval: float  # 情绪更新间隔
    mood_decay_rate: float  # 情绪衰减率
    mood_intensity_factor: float  # 情绪强度因子

    # keywords_reaction
    keywords_reaction_enable: bool  # 是否启用关键词反应
    keywords_reaction_rules: List[Dict[str, Any]]  # 关键词反应规则

    # chinese_typo
    chinese_typo_enable: bool  # 是否启用中文错别字
    chinese_typo_error_rate: float  # 中文错别字错误率
    chinese_typo_min_freq: int  # 中文错别字最小频率
    chinese_typo_tone_error_rate: float  # 中文错别字声调错误率
    chinese_typo_word_replace_rate: float  # 中文错别字单词替换率

    # response_splitter
    enable_response_splitter: bool  # 是否启用回复分割器
    response_max_length: int  # 回复最大长度
    response_max_sentence_num: int  # 回复最大句子数
    enable_kaomoji_protection: bool  # 是否启用颜文字保护

    model_max_output_length: int  # 模型最大输出长度

    # remote
    remote_enable: bool  # 是否启用远程功能

    # experimental
    enable_friend_chat: bool  # 是否启用好友聊天
    talk_allowed_private: List[int]  # 允许私聊的QQ号列表
    enable_pfc_chatting: bool  # 是否启用PFC聊天

    # 模型配置
    llm_reasoning: Dict[str, Any]  # 推理模型配置
    llm_normal: Dict[str, Any]  # 普通模型配置
    llm_topic_judge: Dict[str, Any]  # 主题判断模型配置
    llm_summary: Dict[str, Any]  # 总结模型配置
    llm_emotion_judge: Optional[Dict[str, Any]]  # 情绪判断模型配置（兼容性保留）
    embedding: Dict[str, Any]  # 嵌入模型配置
    vlm: Dict[str, Any]  # VLM模型配置
    moderation: Optional[Dict[str, Any]]  # 审核模型配置（兼容性保留）
    llm_observation: Dict[str, Any]  # 观察模型配置
    llm_sub_heartflow: Dict[str, Any]  # 子心流模型配置
    llm_heartflow: Dict[str, Any]  # 心流模型配置
    llm_plan: Optional[Dict[str, Any]]  # 计划模型配置
    llm_PFC_action_planner: Optional[Dict[str, Any]]  # PFC行动计划模型配置
    llm_PFC_chat: Optional[Dict[str, Any]]  # PFC聊天模型配置
    llm_PFC_reply_checker: Optional[Dict[str, Any]]  # PFC回复检查模型配置
    llm_tool_use: Optional[Dict[str, Any]]  # 工具使用模型配置

    api_urls: Optional[Dict[str, str]]  # API地址配置


@strawberry.type
class APIEnvConfig:
    """环境变量配置"""

    HOST: str  # 服务主机地址
    PORT: int  # 服务端口

    PLUGINS: List[str]  # 插件列表

    MONGODB_HOST: str  # MongoDB 主机地址
    MONGODB_PORT: int  # MongoDB 端口
    DATABASE_NAME: str  # 数据库名称

    CHAT_ANY_WHERE_BASE_URL: str  # ChatAnywhere 基础URL
    SILICONFLOW_BASE_URL: str  # SiliconFlow 基础URL
    DEEP_SEEK_BASE_URL: str  # DeepSeek 基础URL

    DEEP_SEEK_KEY: Optional[str]  # DeepSeek API Key
    CHAT_ANY_WHERE_KEY: Optional[str]  # ChatAnywhere API Key
    SILICONFLOW_KEY: Optional[str]  # SiliconFlow API Key

    SIMPLE_OUTPUT: Optional[bool]  # 是否简化输出
    CONSOLE_LOG_LEVEL: Optional[str]  # 控制台日志等级
    FILE_LOG_LEVEL: Optional[str]  # 文件日志等级
    DEFAULT_CONSOLE_LOG_LEVEL: Optional[str]  # 默认控制台日志等级
    DEFAULT_FILE_LOG_LEVEL: Optional[str]  # 默认文件日志等级

    @strawberry.field
    def get_env(self) -> str:
        return "env"


print("当前路径：")
print(ROOT_PATH)
