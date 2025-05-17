import asyncio
import hashlib
import os
import sys
from pathlib import Path
import time
import platform
import traceback
from dotenv import load_dotenv
from src.common.logger_manager import get_logger

# from src.common.logger import LogConfig, CONFIRM_STYLE_CONFIG
from src.common.crash_logger import install_crash_handler
from src.main import MainSystem
from rich.traceback import install
from src.plugins.group_nickname.nickname_manager import nickname_manager
from src.experimental.PFC.PFC_idle.idle_chat import IdleChat
from src.chat.message_receive.chat_stream import chat_manager
import atexit
from src.config.config import global_config
from src.manager.async_task_manager import async_task_manager, AsyncTask

install(extra_lines=3)

# 设置工作目录为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"已设置工作目录为: {script_dir}")


logger = get_logger("main")
confirm_logger = get_logger("confirm")
# 获取没有加载env时的环境变量
env_mask = {key: os.getenv(key) for key in os.environ}

uvicorn_server = None
driver = None
app = None
loop = None

# shutdown_requested = False  # 新增全局变量


async def request_shutdown() -> bool:
    """请求关闭程序"""
    try:
        if loop and not loop.is_closed():
            try:
                loop.run_until_complete(graceful_shutdown())
            except Exception as ge:  # 捕捉优雅关闭时可能发生的错误
                logger.error(f"优雅关闭时发生错误: {ge}")
                return False
        return True
    except Exception as e:
        logger.error(f"请求关闭程序时发生错误: {e}")
        return False


def easter_egg():
    # 彩蛋
    from colorama import init, Fore

    init()
    text = "你又一次唤醒了这个女孩，但之后，你们将何去何从呢？"
    rainbow_colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
    rainbow_text = ""
    for i, char in enumerate(text):
        rainbow_text += rainbow_colors[i % len(rainbow_colors)] + char
    print(rainbow_text)


def load_env():
    # 直接加载生产环境变量配置
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
        logger.success("成功加载环境变量配置")
    else:
        logger.error("未找到.env文件，请确保文件存在")
        raise FileNotFoundError("未找到.env文件，请确保文件存在")


def scan_provider(env_config: dict):
    provider = {}

    # 利用未初始化 env 时获取的 env_mask 来对新的环境变量集去重
    # 避免 GPG_KEY 这样的变量干扰检查
    env_config = dict(filter(lambda item: item[0] not in env_mask, env_config.items()))

    # 遍历 env_config 的所有键
    for key in env_config:
        # 检查键是否符合 {provider}_BASE_URL 或 {provider}_KEY 的格式
        if key.endswith("_BASE_URL") or key.endswith("_KEY"):
            # 提取 provider 名称
            provider_name = key.split("_", 1)[0]  # 从左分割一次，取第一部分

            # 初始化 provider 的字典（如果尚未初始化）
            if provider_name not in provider:
                provider[provider_name] = {"url": None, "key": None}

            # 根据键的类型填充 url 或 key
            if key.endswith("_BASE_URL"):
                provider[provider_name]["url"] = env_config[key]
            elif key.endswith("_KEY"):
                provider[provider_name]["key"] = env_config[key]

    # 检查每个 provider 是否同时存在 url 和 key
    for provider_name, config in provider.items():
        if config["url"] is None or config["key"] is None:
            logger.error(f"provider 内容：{config}\nenv_config 内容：{env_config}")
            raise ValueError(f"请检查 '{provider_name}' 提供商配置是否丢失 BASE_URL 或 KEY 环境变量")


async def graceful_shutdown():
    try:
        logger.info(f"正在优雅关闭{global_config.bot.nickname}...")

        # 停止所有异步任务
        await async_task_manager.stop_and_wait_all_tasks()

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"{global_config.bot.nickname}关闭失败: {e}")


def check_eula():
    eula_confirm_file = Path("eula.confirmed")
    privacy_confirm_file = Path("privacy.confirmed")
    eula_file = Path("EULA.md")
    privacy_file = Path("PRIVACY.md")

    eula_updated = True
    privacy_updated = True

    eula_confirmed = False
    privacy_confirmed = False

    # 首先计算当前EULA文件的哈希值
    if eula_file.exists():
        with open(eula_file, "r", encoding="utf-8") as f:
            eula_content = f.read()
        eula_new_hash = hashlib.md5(eula_content.encode("utf-8")).hexdigest()
    else:
        logger.error("EULA.md 文件不存在")
        raise FileNotFoundError("EULA.md 文件不存在")

    # 首先计算当前隐私条款文件的哈希值
    if privacy_file.exists():
        with open(privacy_file, "r", encoding="utf-8") as f:
            privacy_content = f.read()
        privacy_new_hash = hashlib.md5(privacy_content.encode("utf-8")).hexdigest()
    else:
        logger.error("PRIVACY.md 文件不存在")
        raise FileNotFoundError("PRIVACY.md 文件不存在")

    # 检查EULA确认文件是否存在
    if eula_confirm_file.exists():
        with open(eula_confirm_file, "r", encoding="utf-8") as f:
            confirmed_content = f.read()
        if eula_new_hash == confirmed_content:
            eula_confirmed = True
            eula_updated = False
    if eula_new_hash == os.getenv("EULA_AGREE"):
        eula_confirmed = True
        eula_updated = False

    # 检查隐私条款确认文件是否存在
    if privacy_confirm_file.exists():
        with open(privacy_confirm_file, "r", encoding="utf-8") as f:
            confirmed_content = f.read()
        if privacy_new_hash == confirmed_content:
            privacy_confirmed = True
            privacy_updated = False
    if privacy_new_hash == os.getenv("PRIVACY_AGREE"):
        privacy_confirmed = True
        privacy_updated = False

    # 如果EULA或隐私条款有更新，提示用户重新确认
    if eula_updated or privacy_updated:
        confirm_logger.critical("EULA或隐私条款内容已更新，请在阅读后重新确认，继续运行视为同意更新后的以上两款协议")
        confirm_logger.critical(
            f'输入"同意"或"confirmed"或设置环境变量"EULA_AGREE={eula_new_hash}"和"PRIVACY_AGREE={privacy_new_hash}"继续运行'
        )
        while True:
            user_input = input().strip().lower()
            if user_input in ["同意", "confirmed"]:
                # print("确认成功，继续运行")
                # print(f"确认成功，继续运行{eula_updated} {privacy_updated}")
                if eula_updated:
                    logger.info(f"更新EULA确认文件{eula_new_hash}")
                    eula_confirm_file.write_text(eula_new_hash, encoding="utf-8")
                if privacy_updated:
                    logger.info(f"更新隐私条款确认文件{privacy_new_hash}")
                    privacy_confirm_file.write_text(privacy_new_hash, encoding="utf-8")
                break
            else:
                confirm_logger.critical('请输入"同意"或"confirmed"以继续运行')
        return
    elif eula_confirmed and privacy_confirmed:
        return


def raw_main():
    # 利用 TZ 环境变量设定程序工作的时区
    if platform.system().lower() != "windows":
        time.tzset()

    # 安装崩溃日志处理器
    install_crash_handler()

    check_eula()
    print("检查EULA和隐私条款完成")

    easter_egg()

    load_env()

    env_config = {key: os.getenv(key) for key in os.environ}
    scan_provider(env_config)

    # 确保 NicknameManager 单例实例存在并已初始化
    # (单例模式下，导入时或第一次调用时会自动初始化)
    _ = nickname_manager  # 显式引用一次

    # 启动 NicknameManager 的后台处理器线程
    logger.info("准备启动绰号处理管理器...")
    nickname_manager.start_processor()  # 调用实例的方法
    logger.info("已调用启动绰号处理管理器。")

    # 注册 NicknameManager 的停止方法到 atexit，确保程序退出时线程能被清理
    atexit.register(nickname_manager.stop_processor)  # 注册实例的方法
    logger.info("已注册绰号处理管理器的退出处理程序。")
    
    # 初始化和启动IdleChat系统，遍历所有聊天流启动空闲聊天功能
    class InitIdleChatTask(AsyncTask):
        """初始化空闲聊天系统的任务"""
        
        def __init__(self):
            super().__init__(task_name="init_idle_chat_system", wait_before_start=10)
        
        async def run(self):
            logger.info("准备初始化空闲聊天系统...")
            try:
                # 直接调用IdleChat的initialize_all_streams方法
                await IdleChat.initialize_all_streams()
                logger.info("空闲聊天系统初始化完成")
            except Exception as e:
                logger.error(f"初始化空闲聊天系统时发生错误: {e}")
                logger.error(traceback.format_exc())
    
    # 创建空闲聊天系统初始化任务，但不立即添加到任务管理器
    # 将其作为MainSystem的属性，在事件循环启动后添加
    idle_chat_init_task = InitIdleChatTask()
    logger.info("已创建空闲聊天系统初始化任务")
    
    # 创建MainSystem实例并设置IdleChat任务
    main_system = MainSystem()
    main_system.idle_chat_init_task = idle_chat_init_task
    
    # 返回MainSystem实例
    return main_system


if __name__ == "__main__":
    exit_code = 0  # 用于记录程序最终的退出状态
    try:
        # 获取MainSystem实例
        main_system = raw_main()

        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 执行初始化和任务调度
            loop.run_until_complete(main_system.initialize())
            
            # 在事件循环已经启动的情况下添加IdleChat初始化任务
            if hasattr(main_system, 'idle_chat_init_task'):
                loop.run_until_complete(async_task_manager.add_task(main_system.idle_chat_init_task))
                logger.info("已添加空闲聊天系统初始化任务到任务管理器")
                
            loop.run_until_complete(main_system.schedule_tasks())
            
        except KeyboardInterrupt:
            # loop.run_until_complete(global_api.stop())
            logger.warning("收到中断信号，正在优雅关闭...")
            if loop and not loop.is_closed():
                try:
                    loop.run_until_complete(graceful_shutdown())
                except Exception as ge:  # 捕捉优雅关闭时可能发生的错误
                    logger.error(f"优雅关闭时发生错误: {ge}")
        # 新增：检测外部请求关闭

        # except Exception as e: # 将主异常捕获移到外层 try...except
        #     logger.error(f"事件循环内发生错误: {str(e)} {str(traceback.format_exc())}")
        #     exit_code = 1
        # finally: # finally 块移到最外层，确保 loop 关闭和暂停总是执行
        #     if loop and not loop.is_closed():
        #         loop.close()
        #     # 在这里添加 input() 来暂停
        #     input("按 Enter 键退出...") # <--- 添加这行
        #     sys.exit(exit_code) # <--- 使用记录的退出码

    except Exception as e:
        logger.error(f"主程序发生异常: {str(e)} {str(traceback.format_exc())}")
        exit_code = 1  # 标记发生错误
    finally:
        # 确保 loop 在任何情况下都尝试关闭（如果存在且未关闭）
        if "loop" in locals() and loop and not loop.is_closed():
            loop.close()
            logger.info("事件循环已关闭")
        # 在程序退出前暂停，让你有机会看到输出
        # input("按 Enter 键退出...")  # <--- 添加这行
        sys.exit(exit_code)  # <--- 使用记录的退出码
