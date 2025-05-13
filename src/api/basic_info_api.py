import platform
import psutil
import sys
import os


def get_system_info():
    """获取操作系统信息"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def get_python_version():
    """获取 Python 版本信息"""
    return sys.version


def get_cpu_usage():
    """获取系统总CPU使用率"""
    return psutil.cpu_percent(interval=1)


def get_process_cpu_usage():
    """获取当前进程CPU使用率"""
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=1)


def get_memory_usage():
    """获取系统内存使用情况 (单位 MB)"""
    mem = psutil.virtual_memory()
    bytes_to_mb = lambda x: round(x / (1024 * 1024), 2)  # noqa
    return {
        "total_mb": bytes_to_mb(mem.total),
        "available_mb": bytes_to_mb(mem.available),
        "percent": mem.percent,
        "used_mb": bytes_to_mb(mem.used),
        "free_mb": bytes_to_mb(mem.free),
    }


def get_process_memory_usage():
    """获取当前进程内存使用情况 (单位 MB)"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    bytes_to_mb = lambda x: round(x / (1024 * 1024), 2)  # noqa
    return {
        "rss_mb": bytes_to_mb(mem_info.rss),  # Resident Set Size: 实际使用物理内存
        "vms_mb": bytes_to_mb(mem_info.vms),  # Virtual Memory Size: 虚拟内存大小
        "percent": process.memory_percent(),  # 进程内存使用百分比
    }


def get_disk_usage(path="/"):
    """获取指定路径磁盘使用情况 (单位 GB)"""
    disk = psutil.disk_usage(path)
    bytes_to_gb = lambda x: round(x / (1024 * 1024 * 1024), 2)  # noqa
    return {
        "total_gb": bytes_to_gb(disk.total),
        "used_gb": bytes_to_gb(disk.used),
        "free_gb": bytes_to_gb(disk.free),
        "percent": disk.percent,
    }


def get_all_basic_info():
    """获取所有基本信息并封装返回"""
    # 对于进程CPU使用率，需要先初始化
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # 初始化调用
    process_cpu = process.cpu_percent(interval=0.1)  # 短暂间隔获取

    return {
        "system_info": get_system_info(),
        "python_version": get_python_version(),
        "cpu_usage_percent": get_cpu_usage(),
        "process_cpu_usage_percent": process_cpu,
        "memory_usage": get_memory_usage(),
        "process_memory_usage": get_process_memory_usage(),
        "disk_usage_root": get_disk_usage("/"),
    }


def get_all_basic_info_string() -> str:
    """获取所有基本信息并以带解释的字符串形式返回"""
    info = get_all_basic_info()

    sys_info = info["system_info"]
    mem_usage = info["memory_usage"]
    proc_mem_usage = info["process_memory_usage"]
    disk_usage = info["disk_usage_root"]

    # 对进程内存使用百分比进行格式化，保留两位小数
    proc_mem_percent = round(proc_mem_usage["percent"], 2)

    output_string = f"""[系统信息]
  - 操作系统: {sys_info["system"]} (例如: Windows, Linux)
  - 发行版本: {sys_info["release"]} (例如: 11, Ubuntu 20.04)
  - 详细版本: {sys_info["version"]}
  - 硬件架构: {sys_info["machine"]} (例如: AMD64)
  - 处理器信息: {sys_info["processor"]}

[Python 环境]
  - Python 版本: {info["python_version"]}

[CPU 状态]
  - 系统总 CPU 使用率: {info["cpu_usage_percent"]}%
  - 当前进程 CPU 使用率: {info["process_cpu_usage_percent"]}%

[系统内存使用情况]
  - 总物理内存: {mem_usage["total_mb"]} MB
  - 可用物理内存: {mem_usage["available_mb"]} MB
  - 物理内存使用率: {mem_usage["percent"]}%
  - 已用物理内存: {mem_usage["used_mb"]} MB
  - 空闲物理内存: {mem_usage["free_mb"]} MB

[当前进程内存使用情况]
  - 实际使用物理内存 (RSS): {proc_mem_usage["rss_mb"]} MB
  - 占用虚拟内存 (VMS): {proc_mem_usage["vms_mb"]} MB
  - 进程内存使用率: {proc_mem_percent}%

[磁盘使用情况 (根目录)]
  - 总空间: {disk_usage["total_gb"]} GB
  - 已用空间: {disk_usage["used_gb"]} GB
  - 可用空间: {disk_usage["free_gb"]} GB
  - 磁盘使用率: {disk_usage["percent"]}%
"""
    return output_string


if __name__ == "__main__":
    print(f"System Info: {get_system_info()}")
    print(f"Python Version: {get_python_version()}")
    print(f"CPU Usage: {get_cpu_usage()}%")
    # 第一次调用 process.cpu_percent() 会返回0.0或一个无意义的值，需要间隔一段时间再调用
    # 或者在初始化Process对象后，先调用一次cpu_percent(interval=None)，然后再调用cpu_percent(interval=1)
    current_process = psutil.Process(os.getpid())
    current_process.cpu_percent(interval=None)  # 初始化
    print(f"Process CPU Usage: {current_process.cpu_percent(interval=1)}%")  # 实际获取

    memory_usage_info = get_memory_usage()
    print(
        f"Memory Usage: Total={memory_usage_info['total_mb']}MB, Used={memory_usage_info['used_mb']}MB, Percent={memory_usage_info['percent']}%"
    )

    process_memory_info = get_process_memory_usage()
    print(
        f"Process Memory Usage: RSS={process_memory_info['rss_mb']}MB, VMS={process_memory_info['vms_mb']}MB, Percent={process_memory_info['percent']}%"
    )

    disk_usage_info = get_disk_usage("/")
    print(
        f"Disk Usage (Root): Total={disk_usage_info['total_gb']}GB, Used={disk_usage_info['used_gb']}GB, Percent={disk_usage_info['percent']}%"
    )

    print("\n--- All Basic Info (JSON) ---")
    all_info = get_all_basic_info()
    import json

    print(json.dumps(all_info, indent=4, ensure_ascii=False))

    print("\n--- All Basic Info (String with Explanations) ---")
    info_string = get_all_basic_info_string()
    print(info_string)
