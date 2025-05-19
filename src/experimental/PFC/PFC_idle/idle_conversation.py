import asyncio
import traceback
from typing import Optional, Any

from src.common.logger_manager import get_logger


logger = get_logger("pfc_idle_conversation")

class IdleConversation:
    """
    负责在idle_chat发送消息后启动对话实例
    实现idle_chat与PFC系统的联动
    目标是实现idle_chat与PFC系统的联动，确保idle_chat发送消息后，
    PFC系统能够正确地启动对话实例，并确保对话实例能够正确地执行wait操作（并未实现TAT）
    """
    
    @staticmethod
    async def start_conversation_for_user(stream_id: str, private_name: str) -> Optional[Any]:
        """
        为用户启动一个对话实例。
        当idle_chat成功发送消息后调用此方法，为对应用户创建PFC对话实例。
        
        Args:
            stream_id (str): 聊天流ID
            private_name (str): 私聊用户名称
            
        Returns:
            Optional[Conversation]: 成功创建的对话实例，如果创建失败则返回None
        """
        try:
            logger.info(f"[私聊][{private_name}] 主动聊天消息发送成功，尝试启动对话实例")
            
            # 在方法内部导入，避免循环导入
            from ..pfc_manager import PFCManager
            # 也在方法内部导入Conversation类
            from ..conversation import Conversation
            
            # 获取PFC管理器实例
            pfc_manager = PFCManager.get_instance()
            
            # 检查是否已存在对话实例
            existing_conversation = await pfc_manager.get_conversation(stream_id)
            if existing_conversation:
                logger.info(f"[私聊][{private_name}] 已存在活动对话实例，无需创建新实例")
                # 即使是已存在的实例，也添加idle标记以确保安全
                if hasattr(existing_conversation, "conversation_info"):
                    existing_conversation.conversation_info.idle_chat_initiated = True
                    existing_conversation.conversation_info.force_wait_only = True
                    logger.info(f"[私聊][{private_name}] 为已存在的对话实例添加了idle_chat标记")
                return existing_conversation
            
            # 创建新的对话实例
            logger.info(f"[私聊][{private_name}] 通过PFC管理器创建新对话实例")
            conversation = await pfc_manager.get_or_create_conversation(stream_id, private_name)
            
            if conversation and conversation._initialized:
                # 配置对话实例直接进入wait状态，跳过rethink_goal环节
                logger.info(f"[私聊][{private_name}] 对话实例成功创建并初始化，配置直接进入wait状态")
                
                # 设置一个初始行动来跳过rethink_goal
                if hasattr(conversation, "conversation_info"):
                    # 创建初始wait行动
                    initial_action = {
                        "action": "wait",
                        "status": "start",
                        "plan_reason": "主动聊天已发送消息，等待用户回复",
                        "time": "主动聊天启动", 
                        "final_reason": ""
                    }
                    
                    # 添加到done_action列表，表示已经有一个wait行动
                    if hasattr(conversation.conversation_info, "done_action"):
                        if not conversation.conversation_info.done_action:
                            conversation.conversation_info.done_action = []
                        conversation.conversation_info.done_action.append(initial_action)
                        logger.info(f"[私聊][{private_name}] 已将初始行动设置为wait，跳过rethink_goal环节")
                    else:
                        logger.warning(f"[私聊][{private_name}] conversation_info缺少done_action属性，无法设置初始wait行动")
                
                    # 添加标记，表明这个对话实例是由idle_chat启动的，应仅限于wait状态
                    conversation.conversation_info.idle_chat_initiated = True
                    conversation.conversation_info.force_wait_only = True
                    logger.info(f"[私聊][{private_name}] 已标记对话实例为idle_chat启动，限制仅能执行wait操作")

                    # 直接修改conversation的行为，确保强制wait状态
                    try:
                        # 使用更通用的方法，通过直接设置next_action字段来强制wait状态
                        if hasattr(conversation, "current_action"):
                            conversation._original_current_action = conversation.current_action
                            conversation.current_action = IdleConversation._create_wait_action()
                            logger.info(f"[私聊][{private_name}] 已设置conversation.current_action为wait")
                        
                        # 检查conversation是否有next_action字段，如果有则覆盖
                        if hasattr(conversation, "next_action"):
                            conversation._original_next_action = conversation.next_action
                            conversation.next_action = IdleConversation._create_wait_action()
                            logger.info(f"[私聊][{private_name}] 已设置conversation.next_action为wait")
                        
                        # 使用猴子补丁修改_update_conversation_action方法，确保任何action更新都被拦截
                        if hasattr(conversation, "_update_conversation_action"):
                            original_update_action = conversation._update_conversation_action
                            
                            async def force_wait_update_action(action_info):
                                # 检查是否应强制使用wait操作
                                if (hasattr(conversation, "conversation_info") and 
                                    hasattr(conversation.conversation_info, "force_wait_only") and 
                                    conversation.conversation_info.force_wait_only):
                                    logger.info(f"[私聊][{private_name}] 拦截到action更新尝试，强制改为wait操作")
                                    # 替换为wait操作
                                    action_info = IdleConversation._create_wait_action()
                                
                                # 调用原始方法
                                return await original_update_action(action_info)
                            
                            # 替换方法
                            conversation._update_conversation_action = force_wait_update_action
                            logger.info(f"[私聊][{private_name}] 已覆盖_update_conversation_action方法")
                        
                    except Exception as e:
                        logger.error(f"[私聊][{private_name}] 修改conversation行为时出错: {str(e)}")
                        logger.error(traceback.format_exc())
                    
                    # 添加标志函数到conversation，其他代码可以检查这些函数
                    conversation.is_idle_chat_conversation = lambda: True
                    conversation.should_force_wait = lambda: True
                    logger.info(f"[私聊][{private_name}] 已添加辅助方法到conversation实例")
                else:
                    logger.warning(f"[私聊][{private_name}] 对话实例缺少conversation_info属性，无法设置初始wait行动")
                
                return conversation
            else:
                logger.error(f"[私聊][{private_name}] 对话实例创建或初始化失败")
                return None
                
        except Exception as e:
            logger.error(f"[私聊][{private_name}] 启动对话实例时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def _create_wait_action():
        """
        创建一个wait操作
        """
        wait_action = {
            "action": "wait",
            "status": "start",
            "plan_reason": "主动聊天限制，只能执行wait操作",
            "time": "idle_chat锁定", 
            "final_reason": ""
        }
        return wait_action
    
    @staticmethod
    async def cleanup_conversation(stream_id: str, private_name: str) -> bool:
        """
        清理用户的对话实例。
        
        Args:
            stream_id (str): 聊天流ID
            private_name (str): 私聊用户名称
            
        Returns:
            bool: 清理成功返回True，否则返回False
        """
        try:
            logger.info(f"[私聊][{private_name}] 尝试清理对话实例")
            
            # 在方法内部导入，避免循环导入
            from ..pfc_manager import PFCManager
            
            # 获取PFC管理器实例
            pfc_manager = PFCManager.get_instance()
            
            # 移除对话实例
            await pfc_manager.remove_conversation(stream_id)
            
            logger.info(f"[私聊][{private_name}] 对话实例已清理")
            return True
            
        except Exception as e:
            logger.error(f"[私聊][{private_name}] 清理对话实例时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False 