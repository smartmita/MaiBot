import traceback
import logging
import asyncio
from typing import Optional, Dict
from src.common.logger_manager import get_logger
import time

logger = get_logger("pfc_idle_conversation")

class IdleConversation:
    """
    处理Idle聊天相关的功能，将这些功能从主Conversation类中分离出来，
    以减少代码量并方便维护。
    """

    def __init__(self):
        """初始化IdleConversation实例"""
        self._idle_chat_manager = None
        self._running = False
        self._active_streams: Dict[str, bool] = {}  # 跟踪活跃的流
        self._monitor_task = None  # 用于后台监控的任务
        self._lock = asyncio.Lock()  # 用于线程安全操作
        self._initialization_in_progress = False  # 防止并发初始化

    async def initialize(self):
        """初始化Idle聊天管理器"""
        # 防止并发初始化
        if self._initialization_in_progress:
            logger.debug("IdleConversation正在初始化中，等待完成")
            return False
            
        if self._idle_chat_manager is not None:
            logger.debug("IdleConversation已初始化，无需重复操作")
            return True
            
        # 标记开始初始化
        self._initialization_in_progress = True
        
        try:
            # 从PFCManager获取IdleChatManager实例
            from ..pfc_manager import PFCManager
            pfc_manager = PFCManager.get_instance()
            self._idle_chat_manager = pfc_manager.get_idle_chat_manager()
            logger.debug("IdleConversation初始化完成，已获取IdleChatManager实例")
            return True
        except Exception as e:
            logger.error(f"初始化IdleConversation时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # 无论成功或失败，都清除初始化标志
            self._initialization_in_progress = False
    
    async def start(self):
        """启动IdleConversation，创建后台监控任务"""
        if self._running:
            logger.debug("IdleConversation已经在运行")
            return False
            
        if not self._idle_chat_manager:
            success = await self.initialize()
            if not success:
                logger.error("无法启动IdleConversation：初始化失败")
                return False
        
        try:
            self._running = True
            # 创建后台监控任务，使用try-except块来捕获可能的异常
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    self._monitor_task = asyncio.create_task(self._monitor_loop())
                    logger.info("IdleConversation启动成功，后台监控任务已创建")
                else:
                    logger.warning("事件循环不活跃，跳过监控任务创建")
            except RuntimeError:
                # 如果没有活跃的事件循环，记录警告但继续执行
                logger.warning("没有活跃的事件循环，IdleConversation将不会启动监控任务")
                # 尽管没有监控任务，但仍然将running设为True表示IdleConversation已启动
            
            return True
        except Exception as e:
            self._running = False
            logger.error(f"启动IdleConversation失败: {e}")
            logger.error(traceback.format_exc())
            return False
        
    async def stop(self):
        """停止IdleConversation的后台任务"""
        if not self._running:
            return
            
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            try:
                self._monitor_task.cancel()
                try:
                    await asyncio.wait_for(self._monitor_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("停止IdleConversation监控任务超时")
                except asyncio.CancelledError:
                    pass  # 正常取消
            except Exception as e:
                logger.error(f"停止IdleConversation监控任务时出错: {e}")
                logger.error(traceback.format_exc())
                
        self._monitor_task = None
        logger.info("IdleConversation已停止")
        
    async def _monitor_loop(self):
        """后台监控循环，定期检查活跃的会话并执行必要的操作"""
        try:
            while self._running:
                try:
                    # 同步活跃流计数到IdleChatManager
                    if self._idle_chat_manager:
                        await self._sync_active_streams_to_manager()
                    
                    # 这里可以添加定期检查逻辑，如查询空闲状态等
                    active_count = len(self._active_streams)
                    logger.debug(f"IdleConversation监控中，当前活跃流数量: {active_count}")
                    
                except Exception as e:
                    logger.error(f"IdleConversation监控循环出错: {e}")
                    logger.error(traceback.format_exc())
                
                # 每30秒执行一次监控
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            logger.info("IdleConversation监控任务已取消")
        except Exception as e:
            logger.error(f"IdleConversation监控任务异常退出: {e}")
            logger.error(traceback.format_exc())
            self._running = False
            
    async def _sync_active_streams_to_manager(self):
        """同步活跃流计数到IdleChatManager和IdleChat"""
        try:
            if not self._idle_chat_manager:
                return
                
            # 获取当前的活跃流列表
            async with self._lock:
                active_streams = list(self._active_streams.keys())
                
            # 对每个活跃流，确保IdleChatManager和IdleChat中的计数是正确的
            for stream_id in active_streams:
                # 获取当前IdleChatManager中的计数
                manager_count = self._idle_chat_manager.get_active_conversations_count(stream_id)
                
                # 由于我们的活跃流字典只记录是否活跃(值为True)，所以计数应该是1
                if manager_count != 1:
                    # 修正IdleChatManager中的计数
                    old_count = manager_count
                    self._idle_chat_manager._active_conversations_count[stream_id] = 1
                    logger.warning(f"同步调整IdleChatManager中的计数: stream_id={stream_id}, {old_count}->1")
                    
                    # 同时修正IdleChat中的计数
                    idle_chat = self._idle_chat_manager.get_idle_chat(stream_id)
                    if idle_chat:
                        if getattr(idle_chat, "active_instances_count", 0) != 1:
                            old_count = getattr(idle_chat, "active_instances_count", 0)
                            idle_chat.active_instances_count = 1
                            logger.warning(f"同步调整IdleChat中的计数: stream_id={stream_id}, {old_count}->1")
            
            # 检查IdleChatManager中有没有多余的计数(conversation中已不存在但manager中还有)
            for stream_id, count in list(self._idle_chat_manager._active_conversations_count.items()):
                if count > 0 and stream_id not in active_streams:
                    # 重置为0
                    self._idle_chat_manager._active_conversations_count[stream_id] = 0
                    logger.warning(f"重置IdleChatManager中的多余计数: stream_id={stream_id}, {count}->0")
                    
                    # 同时修正IdleChat中的计数
                    idle_chat = self._idle_chat_manager.get_idle_chat(stream_id)
                    if idle_chat and getattr(idle_chat, "active_instances_count", 0) > 0:
                        old_count = getattr(idle_chat, "active_instances_count", 0)
                        idle_chat.active_instances_count = 0
                        logger.warning(f"同步重置IdleChat中的计数: stream_id={stream_id}, {old_count}->0")
                        
            # 日志记录同步结果
            total_active = len(active_streams)
            total_manager = sum(self._idle_chat_manager._active_conversations_count.values())
            logger.debug(f"同步后的计数: IdleConversation活跃流={total_active}, IdleChatManager总计数={total_manager}")
            
        except Exception as e:
            logger.error(f"同步活跃流计数失败: {e}")
            logger.error(traceback.format_exc())

    async def get_or_create_idle_chat(self, stream_id: str, private_name: str):
        """
        获取或创建IdleChat实例
        
        Args:
            stream_id: 聊天流ID
            private_name: 私聊对象名称，用于日志
            
        Returns:
            bool: 操作是否成功
        """
        # 确保IdleConversation已启动
        if not self._running:
            await self.start()
            
        if not self._idle_chat_manager:
            # 如果尚未初始化，尝试初始化
            success = await self.initialize()
            if not success:
                logger.warning(f"[私聊][{private_name}] 获取或创建IdleChat失败：IdleChatManager未初始化")
                return False
                
        try:
            # 创建IdleChat实例
            idle_chat = await self._idle_chat_manager.get_or_create_idle_chat(stream_id, private_name)
            logger.debug(f"[私聊][{private_name}] 已创建或获取IdleChat实例")
            return True
        except Exception as e:
            logger.warning(f"[私聊][{private_name}] 创建或获取IdleChat实例失败: {e}")
            logger.warning(traceback.format_exc())
            return False

    async def notify_conversation_start(self, stream_id: str, private_name: str) -> bool:
        """
        通知空闲聊天管理器对话开始
        
        Args:
            stream_id: 聊天流ID
            private_name: 私聊对象名称，用于日志
            
        Returns:
            bool: 通知是否成功
        """
        try:
            # 确保IdleConversation已启动
            if not self._running:
                success = await self.start()
                if not success:
                    logger.warning(f"[私聊][{private_name}] 启动IdleConversation失败，无法通知对话开始")
                    return False
                
            if not self._idle_chat_manager:
                # 如果尚未初始化，尝试初始化
                success = await self.initialize()
                if not success:
                    logger.warning(f"[私聊][{private_name}] 通知对话开始失败：IdleChatManager未初始化")
                    return False
                    
            try:
                # 确保IdleChat实例已创建 - 这是关键步骤，要先创建IdleChat
                await self.get_or_create_idle_chat(stream_id, private_name)
                
                # 先记录活跃状态 - 这是权威源
                async with self._lock:
                    self._active_streams[stream_id] = True
                    
                # 然后同步到IdleChatManager
                if self._idle_chat_manager:
                    await self._idle_chat_manager.notify_conversation_start(stream_id)
                    logger.info(f"[私聊][{private_name}] 已通知空闲聊天管理器对话开始")
                else:
                    logger.warning(f"[私聊][{private_name}] IdleChatManager不存在，但已记录活跃状态")
                    
                # 立即进行一次同步，确保数据一致性
                await self._sync_active_streams_to_manager()
                
                return True
            except Exception as e:
                logger.warning(f"[私聊][{private_name}] 通知空闲聊天管理器对话开始失败: {e}")
                logger.warning(traceback.format_exc())
                # 即使通知失败，也应记录活跃状态
                async with self._lock:
                    self._active_streams[stream_id] = True
                return False
        except Exception as outer_e:
            logger.error(f"[私聊][{private_name}] 处理对话开始通知时发生严重错误: {outer_e}")
            logger.error(traceback.format_exc())
            return False

    async def notify_conversation_end(self, stream_id: str, private_name: str) -> bool:
        """
        通知空闲聊天管理器对话结束
        
        Args:
            stream_id: 聊天流ID
            private_name: 私聊对象名称，用于日志
            
        Returns:
            bool: 通知是否成功
        """
        try:
            # 先从自身的活跃流中移除 - 这是权威源
            was_active = False
            async with self._lock:
                if stream_id in self._active_streams:
                    del self._active_streams[stream_id]
                    was_active = True
                    logger.debug(f"[私聊][{private_name}] 已从活跃流中移除 {stream_id}")
                    
            if not self._idle_chat_manager:
                # 如果尚未初始化，尝试初始化
                success = await self.initialize()
                if not success:
                    logger.warning(f"[私聊][{private_name}] 通知对话结束失败：IdleChatManager未初始化")
                    return False
                    
            try:
                # 然后同步到IdleChatManager
                if self._idle_chat_manager:
                    # 无论如何都尝试通知
                    await self._idle_chat_manager.notify_conversation_end(stream_id)
                    
                    # 立即进行一次同步，确保数据一致性
                    await self._sync_active_streams_to_manager()
                    
                    logger.info(f"[私聊][{private_name}] 已通知空闲聊天管理器对话结束")
                    
                    # 检查当前活跃流数量
                    active_count = len(self._active_streams)
                    if active_count == 0:
                        logger.info(f"[私聊][{private_name}] 当前无活跃流，可能会触发主动聊天")
                    
                    # 额外调用：如果实例存在且只有在确实移除了活跃流的情况下才触发检查
                    if was_active:
                        idle_chat = self._idle_chat_manager.get_idle_chat(stream_id)
                        if idle_chat:
                            # 直接触发IdleChat检查，而不是等待下一个循环
                            logger.info(f"[私聊][{private_name}] 对话结束，手动触发一次主动聊天检查")
                            asyncio.create_task(self._trigger_idle_chat_check(idle_chat, stream_id, private_name))
                    
                    return True
                else:
                    logger.warning(f"[私聊][{private_name}] IdleChatManager不存在，但已更新活跃状态")
                    return False
            except Exception as e:
                logger.warning(f"[私聊][{private_name}] 通知空闲聊天管理器对话结束失败: {e}")
                logger.warning(traceback.format_exc())
                return False
        except Exception as outer_e:
            logger.error(f"[私聊][{private_name}] 处理对话结束通知时发生严重错误: {outer_e}")
            logger.error(traceback.format_exc())
            return False
            
    async def _trigger_idle_chat_check(self, idle_chat, stream_id: str, private_name: str):
        """在对话结束后，手动触发一次IdleChat的检查"""
        try:
            # 确保活跃计数与IdleConversation一致
            async with self._lock:
                is_active_in_conversation = stream_id in self._active_streams
                
            # 强制使IdleChat的计数与IdleConversation一致
            if is_active_in_conversation:
                # 如果在IdleConversation中是活跃的，IdleChat的计数应该是1
                if idle_chat.active_instances_count != 1:
                    old_count = idle_chat.active_instances_count
                    idle_chat.active_instances_count = 1
                    logger.warning(f"[私聊][{private_name}] 修正IdleChat计数: {old_count}->1")
            else:
                # 如果在IdleConversation中不是活跃的，IdleChat的计数应该是0
                if idle_chat.active_instances_count != 0:
                    old_count = idle_chat.active_instances_count
                    idle_chat.active_instances_count = 0
                    logger.warning(f"[私聊][{private_name}] 修正IdleChat计数: {old_count}->0")
                
            # 等待1秒，让任何正在进行的处理完成
            await asyncio.sleep(1)
            
            # 只有当stream不再活跃时才触发检查
            if not is_active_in_conversation:
                # 尝试触发一次检查
                if hasattr(idle_chat, "_should_trigger"):
                    should_trigger = await idle_chat._should_trigger()
                    logger.info(f"[私聊][{private_name}] 手动触发主动聊天检查结果: {should_trigger}")
                    
                    # 如果应该触发，直接调用_initiate_chat
                    if should_trigger and hasattr(idle_chat, "_initiate_chat"):
                        logger.info(f"[私聊][{private_name}] 手动触发主动聊天")
                        await idle_chat._initiate_chat()
                        # 更新最后触发时间
                        idle_chat.last_trigger_time = time.time()
                else:
                    logger.warning(f"[私聊][{private_name}] IdleChat没有_should_trigger方法，无法触发检查")
        except Exception as e:
            logger.error(f"[私聊][{private_name}] 手动触发主动聊天检查时出错: {e}")
            logger.error(traceback.format_exc())

    def is_stream_active(self, stream_id: str) -> bool:
        """检查指定的stream是否活跃"""
        return stream_id in self._active_streams
        
    def get_active_streams_count(self) -> int:
        """获取当前活跃的stream数量"""
        return len(self._active_streams)
        
    @property
    def is_running(self) -> bool:
        """检查IdleConversation是否正在运行"""
        return self._running

    @property
    def idle_chat_manager(self):
        """获取IdleChatManager实例"""
        return self._idle_chat_manager

# 创建单例实例
_instance: Optional[IdleConversation] = None
_instance_lock = asyncio.Lock()
_initialization_in_progress = False  # 防止并发初始化

async def initialize_idle_conversation() -> IdleConversation:
    """初始化并启动IdleConversation单例实例"""
    global _initialization_in_progress
    
    # 防止并发初始化
    if _initialization_in_progress:
        logger.debug("IdleConversation全局初始化正在进行中，等待完成")
        return get_idle_conversation_instance()
        
    # 标记正在初始化
    _initialization_in_progress = True
    
    try:
        instance = get_idle_conversation_instance()
        
        # 如果实例已经在运行，避免重复初始化
        if getattr(instance, '_running', False):
            logger.debug("IdleConversation已在运行状态，无需重新初始化")
            _initialization_in_progress = False
            return instance
            
        # 初始化实例
        success = await instance.initialize()
        if not success:
            logger.error("IdleConversation初始化失败")
            _initialization_in_progress = False
            return instance
            
        # 启动实例
        success = await instance.start()
        if not success:
            logger.error("IdleConversation启动失败")
        else:
            # 启动成功，进行初始检查
            logger.info("IdleConversation启动成功，执行初始化后检查")
            # 这里可以添加一些启动后的检查，如果需要
            
            # 创建一个异步任务，定期检查系统状态
            asyncio.create_task(periodic_system_check(instance))
        
        return instance
    except Exception as e:
        logger.error(f"初始化并启动IdleConversation时出错: {e}")
        logger.error(traceback.format_exc())
        # 重置标志，允许下次再试
        _initialization_in_progress = False
        return get_idle_conversation_instance()  # 返回实例，即使初始化失败
    finally:
        # 清除初始化标志
        _initialization_in_progress = False

async def periodic_system_check(instance: IdleConversation):
    """定期检查系统状态，确保主动聊天功能正常工作"""
    try:
        # 等待10秒，让系统完全启动
        await asyncio.sleep(10)
        
        while getattr(instance, '_running', False):
            try:
                # 检查活跃流数量
                active_streams_count = len(getattr(instance, '_active_streams', {}))
                
                # 如果IdleChatManager存在，检查其中的活跃对话计数
                idle_chat_manager = getattr(instance, '_idle_chat_manager', None)
                if idle_chat_manager and hasattr(idle_chat_manager, 'get_all_active_conversations_count'):
                    manager_count = idle_chat_manager.get_all_active_conversations_count()
                    
                    # 如果两者不一致，记录警告
                    if active_streams_count != manager_count:
                        logger.warning(f"检测到计数不一致: IdleConversation记录的活跃流数量({active_streams_count}) 与 IdleChatManager记录的活跃对话数({manager_count})不匹配")
                        
                        # 如果IdleChatManager记录的计数为0但自己的记录不为0，进行修正
                        if manager_count == 0 and active_streams_count > 0:
                            logger.warning(f"检测到可能的计数错误，尝试修正：清空IdleConversation的活跃流记录")
                            async with instance._lock:
                                instance._active_streams.clear()
                
                # 检查计数如果为0，帮助日志输出
                if active_streams_count == 0:
                    logger.debug("当前没有活跃的对话流，应该可以触发主动聊天")
                
            except Exception as check_err:
                logger.error(f"执行系统检查时出错: {check_err}")
                logger.error(traceback.format_exc())
                
            # 每60秒检查一次
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        logger.debug("系统检查任务被取消")
    except Exception as e:
        logger.error(f"系统检查任务异常退出: {e}")
        logger.error(traceback.format_exc())

def get_idle_conversation_instance() -> IdleConversation:
    """获取IdleConversation的单例实例"""
    global _instance
    if _instance is None:
        _instance = IdleConversation()
    return _instance 