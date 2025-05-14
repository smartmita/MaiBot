import os
import time
from typing import List, Dict, Any, Tuple
from src.chat.focus_chat.heartFC_Cycleinfo import CycleInfo
from src.common.logger_manager import get_logger

logger = get_logger("cycle_analyzer")


class CycleAnalyzer:
    """循环信息分析类，提供查询和分析CycleInfo的工具"""

    def __init__(self, base_dir: str = "log_debug"):
        """
        初始化分析器

        参数:
            base_dir: 存储CycleInfo的基础目录，默认为log_debug
        """
        self.base_dir = base_dir

    def list_streams(self) -> List[str]:
        """
        获取所有聊天流ID列表

        返回:
            List[str]: 聊天流ID列表
        """
        try:
            if not os.path.exists(self.base_dir):
                return []

            return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        except Exception as e:
            logger.error(f"获取聊天流列表时出错: {e}")
            return []

    def get_stream_cycle_count(self, stream_id: str) -> int:
        """
        获取指定聊天流的循环数量

        参数:
            stream_id: 聊天流ID

        返回:
            int: 循环数量
        """
        try:
            files = CycleInfo.list_cycles(stream_id, self.base_dir)
            return len(files)
        except Exception as e:
            logger.error(f"获取聊天流循环数量时出错: {e}")
            return 0

    def get_stream_cycles(self, stream_id: str, start: int = 0, limit: int = -1) -> List[str]:
        """
        获取指定聊天流的循环文件列表

        参数:
            stream_id: 聊天流ID
            start: 起始索引，默认为0
            limit: 返回的最大数量，默认为-1（全部）

        返回:
            List[str]: 循环文件路径列表
        """
        try:
            files = CycleInfo.list_cycles(stream_id, self.base_dir)
            if limit < 0:
                return files[start:]
            else:
                return files[start : start + limit]
        except Exception as e:
            logger.error(f"获取聊天流循环文件列表时出错: {e}")
            return []

    def get_cycle_content(self, filepath: str) -> str:
        """
        获取循环文件的内容

        参数:
            filepath: 文件路径

        返回:
            str: 文件内容
        """
        try:
            if not os.path.exists(filepath):
                return f"文件不存在: {filepath}"

            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取循环文件内容时出错: {e}")
            return f"读取文件出错: {e}"

    def analyze_stream_cycles(self, stream_id: str) -> Dict[str, Any]:
        """
        分析指定聊天流的所有循环，生成统计信息

        参数:
            stream_id: 聊天流ID

        返回:
            Dict[str, Any]: 统计信息
        """
        try:
            files = CycleInfo.list_cycles(stream_id, self.base_dir)
            if not files:
                return {"error": "没有找到循环记录"}

            total_cycles = len(files)
            action_counts = {"text_reply": 0, "emoji_reply": 0, "no_reply": 0, "unknown": 0}
            total_duration = 0
            tool_usage = {}

            for filepath in files:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                    # 解析动作类型
                    for line in content.split("\n"):
                        if line.startswith("动作:"):
                            action = line[3:].strip()
                            action_counts[action] = action_counts.get(action, 0) + 1

                        # 解析耗时
                        elif line.startswith("耗时:"):
                            try:
                                duration = float(line[3:].strip().split("秒")[0])
                                total_duration += duration
                            except Exception as e:
                                logger.error(f"解析耗时时出错: {e}")
                                pass

                        # 解析工具使用
                        elif line.startswith("使用的工具:"):
                            tools = line[6:].strip().split(", ")
                            for tool in tools:
                                tool_usage[tool] = tool_usage.get(tool, 0) + 1

            avg_duration = total_duration / total_cycles if total_cycles > 0 else 0

            return {
                "总循环数": total_cycles,
                "动作统计": action_counts,
                "平均耗时": f"{avg_duration:.2f}秒",
                "总耗时": f"{total_duration:.2f}秒",
                "工具使用次数": tool_usage,
            }
        except Exception as e:
            logger.error(f"分析聊天流循环时出错: {e}")
            return {"error": f"分析出错: {e}"}

    def get_latest_cycles(self, count: int = 10) -> List[Tuple[str, str]]:
        """
        获取所有聊天流中最新的几个循环

        参数:
            count: 获取的数量，默认为10

        返回:
            List[Tuple[str, str]]: 聊天流ID和文件路径的元组列表
        """
        try:
            all_cycles = []
            streams = self.list_streams()

            for stream_id in streams:
                files = CycleInfo.list_cycles(stream_id, self.base_dir)
                for filepath in files:
                    try:
                        # 从文件名中提取时间戳
                        filename = os.path.basename(filepath)
                        timestamp_str = filename.split("_", 2)[2].split(".")[0]
                        timestamp = time.mktime(time.strptime(timestamp_str, "%Y%m%d_%H%M%S"))
                        all_cycles.append((timestamp, stream_id, filepath))
                    except Exception as e:
                        logger.error(f"从文件名中提取时间戳时出错: {e}")
                        continue

            # 按时间戳排序，取最新的count个
            all_cycles.sort(reverse=True)
            return [(item[1], item[2]) for item in all_cycles[:count]]
        except Exception as e:
            logger.error(f"获取最新循环时出错: {e}")
            return []


# 使用示例
if __name__ == "__main__":
    analyzer = CycleAnalyzer()

    # 列出所有聊天流
    streams = analyzer.list_streams()
    print(f"找到 {len(streams)} 个聊天流: {streams}")

    # 分析第一个聊天流的循环
    if streams:
        stream_id = streams[0]
        stats = analyzer.analyze_stream_cycles(stream_id)
        print(f"\n聊天流 {stream_id} 的统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 获取最新的循环
        cycles = analyzer.get_stream_cycles(stream_id, limit=1)
        if cycles:
            print("\n最新循环内容:")
            print(analyzer.get_cycle_content(cycles[0]))

    # 获取所有聊天流中最新的3个循环
    latest_cycles = analyzer.get_latest_cycles(3)
    print(f"\n所有聊天流中最新的 {len(latest_cycles)} 个循环:")
    for stream_id, filepath in latest_cycles:
        print(f"  聊天流 {stream_id}: {os.path.basename(filepath)}")
