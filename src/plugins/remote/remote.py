import requests
import time
import uuid
import platform
import os
import json
import threading
from src.common.logger import get_module_logger, LogConfig, REMOTE_STYLE_CONFIG
from src.config.config import global_config


remote_log_config = LogConfig(
    console_format=REMOTE_STYLE_CONFIG["console_format"],
    file_format=REMOTE_STYLE_CONFIG["file_format"],
)
logger = get_module_logger("remote", config=remote_log_config)

# --- 使用向上导航的方式定义路径 ---

# 1. 获取当前文件 (remote.py) 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 从当前目录向上导航三级找到项目根目录
#    (src/plugins/remote/ -> src/plugins/ -> src/ -> project_root)
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

# 3. 定义 data 目录的路径 (位于项目根目录下)
data_dir = os.path.join(root_dir, "data")

# 4. 定义 UUID 文件在 data 目录下的完整路径
UUID_FILE = os.path.join(data_dir, "client_uuid.json")

# --- 路径定义结束 ---


# 生成或获取客户端唯一ID
def get_unique_id():
    # --- 在尝试读写 UUID_FILE 之前确保 data 目录存在 ---
    # 将目录检查和创建逻辑移到这里，在首次需要写入前执行
    try:
        # exist_ok=True 意味着如果目录已存在也不会报错
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        # 处理可能的权限错误等
        logger.error(f"无法创建数据目录 {data_dir}: {e}")
        # 根据你的错误处理逻辑，可能需要在这里返回错误或抛出异常
        # 暂且返回 None 或抛出，避免继续执行导致问题
        raise RuntimeError(f"无法创建必要的数据目录 {data_dir}") from e
    # --- 目录检查结束 ---

    # 检查是否已经有保存的UUID
    if os.path.exists(UUID_FILE):
        try:
            with open(UUID_FILE, "r", encoding="utf-8") as f:  # 指定 encoding
                data = json.load(f)
                if "client_id" in data:
                    logger.debug(f"从本地文件读取客户端ID: {UUID_FILE}")
                    return data["client_id"]
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"读取UUID文件 {UUID_FILE} 出错: {e}，将生成新的UUID")
        except Exception as e:  # 捕捉其他可能的异常
            logger.error(f"读取UUID文件 {UUID_FILE} 时发生未知错误: {e}")

    # 如果没有保存的UUID或读取出错，则生成新的
    client_id = generate_unique_id()
    logger.info(f"生成新的客户端ID: {client_id}")

    # 保存UUID到文件
    try:
        # 再次确认目录存在 (虽然理论上前面已创建，但更保险)
        os.makedirs(data_dir, exist_ok=True)
        with open(UUID_FILE, "w", encoding="utf-8") as f:  # 指定 encoding
            json.dump({"client_id": client_id}, f, indent=4)  # 添加 indent 使json可读
        logger.info(f"已保存新生成的客户端ID到本地文件: {UUID_FILE}")
    except IOError as e:
        logger.error(f"保存UUID时出错: {UUID_FILE} - {e}")
    except Exception as e:  # 捕捉其他可能的异常
        logger.error(f"保存UUID文件 {UUID_FILE} 时发生未知错误: {e}")

    return client_id


# 生成客户端唯一ID
def generate_unique_id():
    # 结合主机名、系统信息和随机UUID生成唯一ID
    system_info = platform.system()
    unique_id = f"{system_info}-{uuid.uuid4()}"
    return unique_id


def send_heartbeat(server_url, client_id):
    """向服务器发送心跳"""
    sys = platform.system()
    try:
        headers = {"Client-ID": client_id, "User-Agent": f"HeartbeatClient/{client_id[:8]}"}
        data = json.dumps(
            {"system": sys, "Version": global_config.MAI_VERSION},
        )
        logger.debug(f"正在发送心跳到服务器: {server_url}")
        logger.debug(f"心跳数据: {data}")
        response = requests.post(f"{server_url}/api/clients", headers=headers, data=data)

        if response.status_code == 201:
            data = response.json()
            logger.debug(f"心跳发送成功。服务器响应: {data}")
            return True
        else:
            logger.debug(f"心跳发送失败。状态码: {response.status_code}, 响应内容: {response.text}")
            return False

    except requests.RequestException as e:
        # 如果请求异常，可能是网络问题，不记录错误
        logger.debug(f"发送心跳时出错: {e}")
        return False


class HeartbeatThread(threading.Thread):
    """心跳线程类"""

    def __init__(self, server_url, interval):
        super().__init__(daemon=True)  # 设置为守护线程，主程序结束时自动结束
        self.server_url = server_url
        self.interval = interval
        self.client_id = get_unique_id()
        self.running = True
        self.stop_event = threading.Event()  # 添加事件对象用于可中断的等待
        self.last_heartbeat_time = 0  # 记录上次发送心跳的时间

    def run(self):
        """线程运行函数"""
        logger.debug(f"心跳线程已启动，客户端ID: {self.client_id}")

        while self.running:
            # 发送心跳
            if send_heartbeat(self.server_url, self.client_id):
                logger.info(f"{self.interval}秒后发送下一次心跳...")
            else:
                logger.info(f"{self.interval}秒后重试...")

            self.last_heartbeat_time = time.time()

            # 使用可中断的等待代替 sleep
            # 每秒检查一次是否应该停止或发送心跳
            remaining_wait = self.interval
            while remaining_wait > 0 and self.running:
                # 每次最多等待1秒，便于及时响应停止请求
                wait_time = min(1, remaining_wait)
                if self.stop_event.wait(wait_time):
                    break  # 如果事件被设置，立即退出等待
                remaining_wait -= wait_time

                # 检查是否由于外部原因导致间隔异常延长
                if time.time() - self.last_heartbeat_time >= self.interval * 1.5:
                    logger.warning("检测到心跳间隔异常延长，立即发送心跳")
                    break

    def stop(self):
        """停止线程"""
        self.running = False
        self.stop_event.set()  # 设置事件，中断等待
        logger.debug("心跳线程已收到停止信号")


def main():
    if global_config.remote_enable:
        """主函数，启动心跳线程"""
        # 配置
        server_url = "http://hyybuth.xyz:10058"
        # server_url = "http://localhost:10058"
        heartbeat_interval = 300  # 5分钟（秒）

        # 创建并启动心跳线程
        heartbeat_thread = HeartbeatThread(server_url, heartbeat_interval)
        heartbeat_thread.start()

        return heartbeat_thread  # 返回线程对象，便于外部控制
    return None
