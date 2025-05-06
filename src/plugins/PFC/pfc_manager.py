import time
import asyncio # 引入 asyncio
import traceback
from typing import Dict, Optional

from src.common.logger import get_module_logger
from .conversation import Conversation

logger = get_module_logger("pfc_manager")


class PFCManager:
    """PFC对话管理器，负责管理所有对话实例"""

    # 单例模式
    _instance = None

    # 会话实例管理
    _instances: Dict[str, Conversation] = {}
    _initializing: Dict[str, bool] = {} # 用于防止并发初始化同一个 stream_id

    @classmethod
    def get_instance(cls) -> "PFCManager":
        """获取管理器单例"""
        if cls._instance is None:
            cls._instance = PFCManager()
        return cls._instance

    async def get_or_create_conversation(self, stream_id: str, private_name: str) -> Optional[Conversation]:
        """获取或创建对话实例，并确保其启动"""

        # 检查是否正在初始化 (防止并发问题)
        if self._initializing.get(stream_id, False):
            logger.debug(f"[私聊][{private_name}] 会话实例正在初始化中，请稍候: {stream_id}")
            # 可以选择等待一小段时间或直接返回 None
            await asyncio.sleep(0.5) # 短暂等待，让初始化有机会完成
            # 再次检查实例是否存在
            if stream_id in self._instances and self._instances[stream_id]._initialized:
                 logger.debug(f"[私聊][{private_name}] 初始化已完成，返回现有实例: {stream_id}")
                 return self._instances[stream_id]
            else:
                 logger.warning(f"[私聊][{private_name}] 等待后实例仍未初始化完成或不存在。")
                 return None # 避免返回未完成的实例

        # 检查是否已有活动实例
        if stream_id in self._instances:
            instance = self._instances[stream_id]
            # 检查忽略状态
            if (hasattr(instance, "ignore_until_timestamp") and
                    instance.ignore_until_timestamp and
                    time.time() < instance.ignore_until_timestamp):
                logger.debug(f"[私聊][{private_name}] 会话实例当前处于忽略状态: {stream_id}")
                return None # 处于忽略状态，不返回实例

            # 检查是否已初始化且应继续运行
            if instance._initialized and instance.should_continue:
                logger.debug(f"[私聊][{private_name}] 使用现有活动会话实例: {stream_id}")
                return instance
            else:
                # 如果实例存在但未初始化或不应继续，清理旧实例
                logger.warning(f"[私聊][{private_name}] 发现无效或已停止的旧实例，清理并重新创建: {stream_id}")
                await self._cleanup_conversation(instance)
                # 从字典中移除，确保下面能创建新的
                if stream_id in self._instances: del self._instances[stream_id]
                if stream_id in self._initializing: del self._initializing[stream_id]


        # --- 创建并初始化新实例 ---
        conversation_instance: Optional[Conversation] = None
        try:
            logger.info(f"[私聊][{private_name}] 创建新的对话实例: {stream_id}")
            self._initializing[stream_id] = True # 标记开始初始化

            # 创建实例
            conversation_instance = Conversation(stream_id, private_name)
            self._instances[stream_id] = conversation_instance # 立即存入字典

            # **启动实例初始化**
            # _initialize_conversation 会调用 conversation._initialize()
            await self._initialize_conversation(conversation_instance)

            # --- 关键修复：在初始化成功后调用 start() ---
            if conversation_instance._initialized and conversation_instance.should_continue:
                logger.info(f"[私聊][{private_name}] 初始化成功，调用 conversation.start() 启动主循环...")
                await conversation_instance.start() # 确保调用 start 方法
            else:
                # 如果 _initialize_conversation 内部初始化失败
                logger.error(f"[私聊][{private_name}] 初始化未成功完成，无法启动实例 {stream_id}。")
                # 清理可能部分创建的实例
                await self._cleanup_conversation(conversation_instance)
                if stream_id in self._instances: del self._instances[stream_id]
                conversation_instance = None # 返回 None 表示失败

        except Exception as e:
            logger.error(f"[私聊][{private_name}] 创建或启动会话实例时发生严重错误: {stream_id}, 错误: {e}")
            logger.error(traceback.format_exc())
            # 确保清理
            if conversation_instance:
                await self._cleanup_conversation(conversation_instance)
            if stream_id in self._instances: del self._instances[stream_id]
            conversation_instance = None # 返回 None

        finally:
            # 确保初始化标记被清除
             if stream_id in self._initializing:
                 self._initializing[stream_id] = False

        return conversation_instance

    async def _initialize_conversation(self, conversation: Conversation):
        """(内部方法) 初始化会话实例的核心逻辑"""
        stream_id = conversation.stream_id
        private_name = conversation.private_name
        try:
            logger.info(f"[私聊][{private_name}] 管理器开始调用 conversation._initialize(): {stream_id}")
            await conversation._initialize() # 调用实例自身的初始化方法
            # 注意：初始化成功与否由 conversation._initialized 和 conversation.should_continue 标志决定
            if conversation._initialized:
                 logger.info(f"[私聊][{private_name}] conversation._initialize() 调用完成，实例标记为已初始化: {stream_id}")
            else:
                 logger.warning(f"[私聊][{private_name}] conversation._initialize() 调用完成，但实例未成功标记为已初始化: {stream_id}")

        except Exception as e:
            # _initialize 内部应该处理自己的异常，但这里也捕获以防万一
            logger.error(f"[私聊][{private_name}] 调用 conversation._initialize() 时发生未捕获错误: {stream_id}, 错误: {e}")
            logger.error(traceback.format_exc())
            # 确保实例状态反映失败
            conversation._initialized = False
            conversation.should_continue = False


    async def _cleanup_conversation(self, conversation: Conversation):
        """清理会话实例的资源"""
        if not conversation: return
        stream_id = conversation.stream_id
        private_name = conversation.private_name
        logger.info(f"[私聊][{private_name}] 开始清理会话实例资源: {stream_id}")
        try:
            # 调用 conversation 的 stop 方法来停止其内部组件
            if hasattr(conversation, 'stop') and callable(conversation.stop):
                await conversation.stop() # stop 方法应处理内部组件的停止
            else:
                 logger.warning(f"[私聊][{private_name}] Conversation 对象缺少 stop 方法，可能无法完全清理资源。")
                 # 尝试手动停止已知组件 (作为后备)
                 if hasattr(conversation, 'idle_conversation_starter') and conversation.idle_conversation_starter:
                     conversation.idle_conversation_starter.stop()
                 if hasattr(conversation, 'observation_info') and conversation.observation_info:
                     conversation.observation_info.unbind_from_chat_observer()
                 # ChatObserver 是单例，不在此处停止

            logger.info(f"[私聊][{private_name}] 会话实例 {stream_id} 资源已清理")
        except Exception as e:
            logger.error(f"[私聊][{private_name}] 清理会话实例资源时失败: {stream_id}, 错误: {e}")
            logger.error(traceback.format_exc())

    async def get_conversation(self, stream_id: str) -> Optional[Conversation]:
        """获取已存在的会话实例 (只读)"""
        instance = self._instances.get(stream_id)
        if instance and instance._initialized and instance.should_continue:
             # 检查忽略状态
             if (hasattr(instance, "ignore_until_timestamp") and
                 instance.ignore_until_timestamp and
                 time.time() < instance.ignore_until_timestamp):
                 return None # 忽略期间不返回
             return instance
        return None # 不存在或无效则返回 None

    async def remove_conversation(self, stream_id: str):
        """移除并清理会话实例"""
        if stream_id in self._instances:
            instance_to_remove = self._instances[stream_id]
            logger.info(f"[管理器] 准备移除并清理会话实例: {stream_id}")
            try:
                # 先从字典中移除引用，防止新的请求获取到正在清理的实例
                del self._instances[stream_id]
                if stream_id in self._initializing: del self._initializing[stream_id]
                # 清理资源
                await self._cleanup_conversation(instance_to_remove)
                logger.info(f"[管理器] 会话实例 {stream_id} 已成功移除并清理")
            except Exception as e:
                logger.error(f"[管理器] 移除或清理会话实例 {stream_id} 时失败: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"[管理器] 尝试移除不存在的会话实例: {stream_id}")

