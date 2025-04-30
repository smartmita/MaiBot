import threading

# 功能总开关
ENABLE_NICKNAME_MAPPING = False # 设置为 False 可完全禁用此功能

# --- LLM 相关配置 (示例，你需要根据实际情况修改) ---
# 用于绰号映射分析的 LLM 模型配置
LLM_MODEL_NICKNAME_MAPPING = {
    "model_name": "your_llm_model_for_mapping", # 替换成你用于分析的模型名称
    "api_key": "YOUR_API_KEY", # 如果需要
    "base_url": "YOUR_API_BASE", # 如果需要
    "temperature": 0.5,
    "max_tokens": 200,
}

# --- 数据库相关配置 (如果需要独立配置) ---
# 例如，如果数据库连接信息不同或需要特定集合名称
DB_COLLECTION_PERSON_INFO = "person_info" # 你的用户信息集合名称

# --- Prompt 注入配置 ---
MAX_NICKNAMES_IN_PROMPT = 10 # Prompt 中最多注入的绰号数量
NICKNAME_PROBABILITY_SMOOTHING = 1 # 用于加权随机选择的平滑因子 (防止概率为0)

# --- 进程控制 ---
NICKNAME_QUEUE_MAX_SIZE = 100 # 进程间通信队列的最大容量
NICKNAME_PROCESS_SLEEP_INTERVAL = 0.5 # 映射进程在队列为空时的休眠时间（秒）


# --- 运行时状态 (用于安全停止进程) ---
_stop_event = threading.Event()

def get_stop_event():
    """获取全局停止事件"""
    return _stop_event

def set_stop_event():
    """设置全局停止事件，通知子进程退出"""
    _stop_event.set()

