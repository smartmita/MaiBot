from typing import Tuple, List, Dict, Any
from src.common.logger import get_module_logger
from src.config.config import global_config  # 为了获取 BOT_QQ
from .chat_observer import ChatObserver

logger = get_module_logger("reply_checker")


class ReplyChecker:
    """回复检查器 - 新版：仅检查机器人自身发言的精确重复"""

    def __init__(self, stream_id: str, private_name: str):
        # self.llm = LLMRequest(...) # <--- 移除 LLM 初始化
        self.name = global_config.BOT_NICKNAME
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        # self.max_retries = 3 # 这个 max_retries 属性在当前设计下不再由 checker 控制，而是由 conversation.py 控制
        self.bot_qq_str = str(global_config.BOT_QQ)  # 获取机器人QQ号用于识别自身消息

    async def check(
        self,
        reply: str,
        goal: str,
        chat_history: List[Dict[str, Any]],
        chat_history_text: str,
        current_time_str: str,
        retry_count: int = 0,
    ) -> Tuple[bool, str, bool]:
        """检查生成的回复是否与机器人之前的发言完全一致（长度大于4）

        Args:
            reply: 待检查的机器人回复内容
            goal: 当前对话目标 (新逻辑中未使用)
            chat_history: 对话历史记录 (包含用户和机器人的消息字典列表)
            chat_history_text: 对话历史记录的文本格式 (新逻辑中未使用)
            current_time_str: 当前时间的字符串格式 (新逻辑中未使用)
            retry_count: 当前重试次数 (新逻辑中未使用)

        Returns:
            Tuple[bool, str, bool]: (是否合适, 原因, 是否需要重新规划)
                                    对于重复消息: (False, "机器人尝试发送重复消息", False)
                                    对于非重复消息: (True, "消息内容未与机器人历史发言重复。", False)
        """
        if not self.bot_qq_str:
            logger.error(f"[私聊][{self.private_name}] ReplyChecker: BOT_QQ 未配置，无法检查机器人自身消息。")
            return True, "BOT_QQ未配置，跳过重复检查。", False  # 无法检查则默认通过

        if len(reply) <= 4:
            return True, "消息长度小于等于4字符，跳过重复检查。", False

        try:
            match_found = False  # <--- 用于调试
            for i, msg_dict in enumerate(chat_history):  # <--- 添加索引用于日志
                if not isinstance(msg_dict, dict):
                    continue

                user_info_data = msg_dict.get("user_info")
                if not isinstance(user_info_data, dict):
                    continue

                sender_id = str(user_info_data.get("user_id"))

                if sender_id == self.bot_qq_str:
                    historical_message_text = msg_dict.get("processed_plain_text", "")
                    # <--- 新增详细对比日志 --- START --->
                    logger.debug(
                        f"[私聊][{self.private_name}] ReplyChecker: 历史记录 #{i} (机器人): '{historical_message_text}' (长度 {len(historical_message_text)})"
                    )
                    if reply == historical_message_text:
                        logger.warning(f"[私聊][{self.private_name}] ReplyChecker: !!! 精确匹配成功 !!!")
                        logger.warning(f"[私聊][{self.private_name}] ReplyChecker 检测到机器人自身重复消息: '{reply}'")
                        match_found = True  # <--- 标记找到
                        return (False, "机器人尝试发送重复消息", False)
                    # <--- 新增详细对比日志 --- END --->

            if not match_found:  # <--- 根据标记判断
                logger.debug(f"[私聊][{self.private_name}] ReplyChecker: 未找到重复。")  # <--- 新增日志
            return (True, "消息内容未与机器人历史发言重复。", False)

        except Exception as e:
            import traceback

            logger.error(f"[私聊][{self.private_name}] ReplyChecker 检查重复时出错: 类型={type(e)}, 值={e}")
            logger.error(f"[私聊][{self.private_name}]{traceback.format_exc()}")
            # 发生未知错误时，为安全起见，默认通过，并记录原因
            return (True, f"检查重复时发生内部错误: {str(e)}", False)
