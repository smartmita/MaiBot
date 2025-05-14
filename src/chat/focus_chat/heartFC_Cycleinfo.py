import time
import os
import json
from typing import List, Optional, Dict, Any


class CycleDetail:
    """循环信息记录类"""

    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.action_taken = False
        self.action_type = "unknown"
        self.reasoning = ""
        self.timers: Dict[str, float] = {}
        self.thinking_id = ""
        self.replanned = False

        # 添加响应信息相关字段
        self.response_info: Dict[str, Any] = {
            "response_text": [],  # 回复的文本列表
            "emoji_info": "",  # 表情信息
            "anchor_message_id": "",  # 锚点消息ID
            "reply_message_ids": [],  # 回复消息ID列表
            "sub_mind_thinking": "",  # 子思维思考内容
            "in_mind_reply": [],  # 子思维思考内容
        }

        # 添加SubMind相关信息
        self.submind_info: Dict[str, Any] = {
            "prompt": "",  # SubMind输入的prompt
            "structured_info": "",  # 结构化信息
            "result": "",  # SubMind的思考结果
        }

        # 添加ToolUse相关信息
        self.tooluse_info: Dict[str, Any] = {
            "prompt": "",  # 工具使用的prompt
            "tools_used": [],  # 使用了哪些工具
            "tool_results": [],  # 工具获得的信息
        }

        # 添加Planner相关信息
        self.planner_info: Dict[str, Any] = {
            "prompt": "",  # 规划器的prompt
            "response": "",  # 规划器的原始回复
            "parsed_result": {},  # 解析后的结果
        }

    def to_dict(self) -> Dict[str, Any]:
        """将循环信息转换为字典格式"""
        return {
            "cycle_id": self.cycle_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "action_taken": self.action_taken,
            "action_type": self.action_type,
            "reasoning": self.reasoning,
            "timers": self.timers,
            "thinking_id": self.thinking_id,
            "response_info": self.response_info,
            "submind_info": self.submind_info,
            "tooluse_info": self.tooluse_info,
            "planner_info": self.planner_info,
        }

    def complete_cycle(self):
        """完成循环，记录结束时间"""
        self.end_time = time.time()

    def set_action_info(
        self, action_type: str, reasoning: str, action_taken: bool, action_data: Optional[Dict[str, Any]] = None
    ):
        """设置动作信息"""
        self.action_type = action_type
        self.action_data = action_data
        self.reasoning = reasoning
        self.action_taken = action_taken

    def set_thinking_id(self, thinking_id: str):
        """设置思考消息ID"""
        self.thinking_id = thinking_id

    def set_response_info(
        self,
        response_text: Optional[List[str]] = None,
        emoji_info: Optional[str] = None,
        anchor_message_id: Optional[str] = None,
        reply_message_ids: Optional[List[str]] = None,
        sub_mind_thinking: Optional[str] = None,
    ):
        """设置响应信息"""
        if response_text is not None:
            self.response_info["response_text"] = response_text
        if emoji_info is not None:
            self.response_info["emoji_info"] = emoji_info
        if anchor_message_id is not None:
            self.response_info["anchor_message_id"] = anchor_message_id
        if reply_message_ids is not None:
            self.response_info["reply_message_ids"] = reply_message_ids
        if sub_mind_thinking is not None:
            self.response_info["sub_mind_thinking"] = sub_mind_thinking

    def set_submind_info(
        self,
        prompt: Optional[str] = None,
        structured_info: Optional[str] = None,
        result: Optional[str] = None,
    ):
        """设置SubMind信息"""
        if prompt is not None:
            self.submind_info["prompt"] = prompt
        if structured_info is not None:
            self.submind_info["structured_info"] = structured_info
        if result is not None:
            self.submind_info["result"] = result

    def set_tooluse_info(
        self,
        prompt: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ):
        """设置ToolUse信息"""
        if prompt is not None:
            self.tooluse_info["prompt"] = prompt
        if tools_used is not None:
            self.tooluse_info["tools_used"] = tools_used
        if tool_results is not None:
            self.tooluse_info["tool_results"] = tool_results

    def set_planner_info(
        self,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        parsed_result: Optional[Dict[str, Any]] = None,
    ):
        """设置Planner信息"""
        if prompt is not None:
            self.planner_info["prompt"] = prompt
        if response is not None:
            self.planner_info["response"] = response
        if parsed_result is not None:
            self.planner_info["parsed_result"] = parsed_result

    @staticmethod
    def save_to_file(cycle_info: "CycleDetail", stream_id: str, base_dir: str = "log_debug") -> str:
        """
        将CycleInfo保存到文件

        参数:
            cycle_info: CycleInfo对象
            stream_id: 聊天流ID
            base_dir: 基础目录，默认为log_debug

        返回:
            str: 保存的文件路径
        """
        try:
            # 创建目录结构
            stream_dir = os.path.join(base_dir, stream_id)
            os.makedirs(stream_dir, exist_ok=True)

            # 生成文件名和路径
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(cycle_info.start_time))
            filename = f"cycle_{cycle_info.cycle_id}_{timestamp}.txt"
            filepath = os.path.join(stream_dir, filename)

            # 格式化输出成易读的格式
            with open(filepath, "w", encoding="utf-8") as f:
                # 写入基本信息
                f.write(f"循环ID: {cycle_info.cycle_id}\n")
                f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cycle_info.start_time))}\n")
                if cycle_info.end_time:
                    f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cycle_info.end_time))}\n")
                    duration = cycle_info.end_time - cycle_info.start_time
                    f.write(f"耗时: {duration:.2f}秒\n")
                f.write(f"动作: {cycle_info.action_type}\n")
                f.write(f"原因: {cycle_info.reasoning}\n")
                f.write(f"执行状态: {'已执行' if cycle_info.action_taken else '未执行'}\n")
                f.write(f"思考ID: {cycle_info.thinking_id}\n")
                f.write(f"是否为重新规划: {'是' if cycle_info.replanned else '否'}\n\n")

                # 写入计时器信息
                if cycle_info.timers:
                    f.write("== 计时器信息 ==\n")
                    for name, elapsed in cycle_info.timers.items():
                        formatted_time = f"{elapsed * 1000:.2f}毫秒" if elapsed < 1 else f"{elapsed:.2f}秒"
                        f.write(f"{name}: {formatted_time}\n")
                    f.write("\n")

                # 写入响应信息
                f.write("== 响应信息 ==\n")
                f.write(f"锚点消息ID: {cycle_info.response_info['anchor_message_id']}\n")
                if cycle_info.response_info["response_text"]:
                    f.write("回复文本:\n")
                    for i, text in enumerate(cycle_info.response_info["response_text"]):
                        f.write(f"  [{i + 1}] {text}\n")
                if cycle_info.response_info["emoji_info"]:
                    f.write(f"表情信息: {cycle_info.response_info['emoji_info']}\n")
                if cycle_info.response_info["reply_message_ids"]:
                    f.write(f"回复消息ID: {', '.join(cycle_info.response_info['reply_message_ids'])}\n")
                f.write("\n")

                # 写入SubMind信息
                f.write("== SubMind信息 ==\n")
                f.write(f"结构化信息:\n{cycle_info.submind_info['structured_info']}\n\n")
                f.write(f"思考结果:\n{cycle_info.submind_info['result']}\n\n")
                f.write("SubMind Prompt:\n")
                f.write(f"{cycle_info.submind_info['prompt']}\n\n")

                # 写入ToolUse信息
                f.write("== 工具使用信息 ==\n")
                if cycle_info.tooluse_info["tools_used"]:
                    f.write(f"使用的工具: {', '.join(cycle_info.tooluse_info['tools_used'])}\n")
                else:
                    f.write("未使用工具\n")

                if cycle_info.tooluse_info["tool_results"]:
                    f.write("工具结果:\n")
                    for i, result in enumerate(cycle_info.tooluse_info["tool_results"]):
                        f.write(f"  [{i + 1}] 类型: {result.get('type', '未知')}, 内容: {result.get('content', '')}\n")
                f.write("\n")
                f.write("工具执行 Prompt:\n")
                f.write(f"{cycle_info.tooluse_info['prompt']}\n\n")

                # 写入Planner信息
                f.write("== Planner信息 ==\n")
                f.write("Planner Prompt:\n")
                f.write(f"{cycle_info.planner_info['prompt']}\n\n")
                f.write("原始回复:\n")
                f.write(f"{cycle_info.planner_info['response']}\n\n")
                f.write("解析结果:\n")
                f.write(f"{json.dumps(cycle_info.planner_info['parsed_result'], ensure_ascii=False, indent=2)}\n")

            return filepath
        except Exception as e:
            print(f"保存CycleInfo到文件时出错: {e}")
            return ""

    @staticmethod
    def load_from_file(filepath: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载CycleInfo信息（只加载JSON格式的数据，不解析文本格式）

        参数:
            filepath: 文件路径

        返回:
            Optional[Dict[str, Any]]: 加载的CycleInfo数据，失败则返回None
        """
        try:
            if not os.path.exists(filepath):
                print(f"文件不存在: {filepath}")
                return None

            # 尝试从文件末尾读取JSON数据
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 查找"解析结果:"后的JSON数据
            for i, line in enumerate(lines):
                if "解析结果:" in line and i + 1 < len(lines):
                    # 尝试解析后面的行
                    json_data = ""
                    for j in range(i + 1, len(lines)):
                        json_data += lines[j]

                    try:
                        return json.loads(json_data)
                    except json.JSONDecodeError:
                        continue

            # 如果没有找到JSON数据，则返回None
            return None
        except Exception as e:
            print(f"从文件加载CycleInfo时出错: {e}")
            return None

    @staticmethod
    def list_cycles(stream_id: str, base_dir: str = "log_debug") -> List[str]:
        """
        列出指定stream_id的所有循环文件

        参数:
            stream_id: 聊天流ID
            base_dir: 基础目录，默认为log_debug

        返回:
            List[str]: 文件路径列表
        """
        try:
            stream_dir = os.path.join(base_dir, stream_id)
            if not os.path.exists(stream_dir):
                return []

            files = [
                os.path.join(stream_dir, f)
                for f in os.listdir(stream_dir)
                if f.startswith("cycle_") and f.endswith(".txt")
            ]
            return sorted(files)
        except Exception as e:
            print(f"列出循环文件时出错: {e}")
            return []
