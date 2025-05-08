import requests
import time
import uuid
import platform
import os
import json
import threading
import subprocess

# from loguru import logger
from src.common.logger_manager import get_logger
from src.config.config import global_config

logger = get_logger("remote")

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
    # 基于机器码生成唯一ID，同一台机器上生成的UUID是固定的，只要机器码不变
    import hashlib

    system_info = platform.system()
    machine_code = None

    try:
        if system_info == "Windows":
            # 使用wmic命令获取主机UUID（更稳定）
            result = subprocess.check_output(
                "wmic csproduct get uuid", shell=True, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL
            )
            lines = result.decode(errors="ignore").splitlines()
            # 过滤掉空行和表头，只取有效UUID
            uuids = [line.strip() for line in lines if line.strip() and line.strip().lower() != "uuid"]
            if uuids:
                uuid_val = uuids[0]
                # logger.debug(f"主机UUID: {uuid_val}")
                # 增加无效值判断
                if uuid_val and uuid_val.lower() not in ["to be filled by o.e.m.", "none", "", "standard"]:
                    machine_code = uuid_val
        elif system_info == "Linux":
            # 优先读取 /etc/machine-id，其次 /var/lib/dbus/machine-id，取第一个非空且内容有效的
            for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        code = f.read().strip()
                        # 只要内容非空且不是全0
                        if code and set(code) != {"0"}:
                            machine_code = code
                            break
        elif system_info == "Darwin":
            # macOS: 使用IOPlatformUUID
            result = subprocess.check_output(
                "ioreg -rd1 -c IOPlatformExpertDevice | awk '/IOPlatformUUID/'", shell=True
            )
            uuid_line = result.decode(errors="ignore")
            # 解析出 "IOPlatformUUID" = "xxxx-xxxx-xxxx-xxxx"
            import re

            m = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', uuid_line)
            if m:
                uuid_val = m.group(1)
                logger.debug(f"IOPlatformUUID: {uuid_val}")
                if uuid_val and uuid_val.lower() not in ["to be filled by o.e.m.", "none", "", "standard"]:
                    machine_code = uuid_val
    except Exception as e:
        logger.debug(f"获取机器码失败: {e}")

    # 如果主板序列号无效，尝试用MAC地址
    if not machine_code:
        try:
            mac = uuid.getnode()
            if (mac >> 40) % 2 == 0:  # 不是本地伪造MAC
                machine_code = str(mac)
        except Exception as e:
            logger.debug(f"获取MAC地址失败: {e}")

    def md5_to_uuid(md5hex):
        # 将32位md5字符串格式化为8-4-4-4-12的UUID格式
        return f"{md5hex[0:8]}-{md5hex[8:12]}-{md5hex[12:16]}-{md5hex[16:20]}-{md5hex[20:32]}"

    if machine_code:
        # print(f"machine_code={machine_code!r}")  # 可用于调试
        md5 = hashlib.md5(machine_code.encode("utf-8")).hexdigest()
        uuid_str = md5_to_uuid(md5)
    else:
        uuid_str = str(uuid.uuid4())

    unique_id = f"{system_info}-{uuid_str}"
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


# --- 测试用例 ---
if __name__ == "__main__":
    print("测试唯一ID生成：")
    print("唯一ID:", get_unique_id())
