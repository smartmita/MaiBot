import time
import traceback
from typing import TYPE_CHECKING

from src.common.logger_manager import get_logger
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_before_timestamp_with_chat
from maim_message import UserInfo
from src.chat.message_receive.chat_stream import chat_manager
from src.config.config import global_config

# 导入 PFC 内部组件和类型
from .pfc_types import ConversationState
from .pfc import GoalAnalyzer
from .chat_observer import ChatObserver
from .message_sender import DirectMessageSender
from .action_planner import ActionPlanner
from .observation_info import ObservationInfo
from .conversation_info import ConversationInfo
from .reply_generator import ReplyGenerator
from .PFC_idle.idle_chat import IdleChat
from .PFC_idle.idle_manager import IdleManager
from .waiter import Waiter
from .pfc_utils import get_person_id
from .reply_checker import ReplyChecker
from .pfc_relationship import PfcRelationshipUpdater, PfcRepationshipTranslator
from .pfc_emotion import PfcEmotionUpdater
from experimental.Legacy_HFC.heart_flow.sub_mind import SubMind


if TYPE_CHECKING:
    from .conversation import Conversation  # 用于类型提示以避免循环导入

logger = get_logger("pfc_initializer")


async def load_initial_history(conversation_instance: "Conversation"):
    """
    加载并处理初始的聊天记录。
    之前是 Conversation 类中的 _load_initial_history 方法。
    """
    if not conversation_instance.observation_info:  # 确保 ObservationInfo 已初始化
        logger.warning(f"[私聊][{conversation_instance.private_name}] ObservationInfo 未初始化，无法加载历史记录。")
        return

    try:
        logger.debug(
            f"[私聊][{conversation_instance.private_name}] 为 {conversation_instance.stream_id} 加载初始聊天记录..."
        )
        # 从聊天核心获取原始消息列表
        initial_messages = get_raw_msg_before_timestamp_with_chat(
            chat_id=conversation_instance.stream_id,
            timestamp=time.time(),
            limit=30,  # limit 可以根据需要调整或配置
        )

        if initial_messages:
            # 更新 ObservationInfo 中的历史记录列表和计数
            conversation_instance.observation_info.chat_history = initial_messages
            conversation_instance.observation_info.chat_history_count = len(initial_messages)

            # 获取最后一条消息的信息
            last_msg = initial_messages[-1]
            conversation_instance.observation_info.last_message_time = last_msg.get("time")
            conversation_instance.observation_info.last_message_id = last_msg.get("message_id")

            # 安全地解析最后一条消息的发送者信息
            last_user_info_dict = last_msg.get("user_info", {})
            if isinstance(last_user_info_dict, dict):
                try:
                    last_user_info = UserInfo.from_dict(last_user_info_dict)
                    # 存储发送者的 user_id 字符串
                    conversation_instance.observation_info.last_message_sender = (
                        str(last_user_info.user_id) if last_user_info else None
                    )
                except Exception as e:
                    logger.warning(
                        f"[私聊][{conversation_instance.private_name}] 解析最后一条消息的用户信息时出错: {e}"
                    )
                    conversation_instance.observation_info.last_message_sender = None
            else:
                # 如果 user_info 不是字典，也标记为未知
                conversation_instance.observation_info.last_message_sender = None

            # 存储最后一条消息的文本内容
            conversation_instance.observation_info.last_message_content = last_msg.get("processed_plain_text", "")

            # 构建用于 Prompt 的历史记录字符串 (只使用最近的一部分)
            history_slice_for_str = initial_messages[-30:]  # 可配置
            conversation_instance.observation_info.chat_history_str = await build_readable_messages(
                history_slice_for_str,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,  # read_mark 可能需要根据实际情况调整
            )

            # 更新 ChatObserver 和 IdleChat 的时间戳
            if conversation_instance.chat_observer:
                # 更新观察者的最后消息时间，避免重复处理这些初始消息
                conversation_instance.chat_observer.last_message_time = (
                    conversation_instance.observation_info.last_message_time
                )
            if conversation_instance.idle_chat and conversation_instance.observation_info.last_message_time:
                # 更新空闲计时器的起始时间
                await conversation_instance.idle_chat.update_last_message_time(
                    conversation_instance.observation_info.last_message_time
                )

            logger.info(
                f"[私聊][{conversation_instance.private_name}] 成功加载 {len(initial_messages)} 条初始聊天记录。最后一条消息时间: {conversation_instance.observation_info.last_message_time}"
            )
        else:
            # 如果没有历史记录
            logger.info(f"[私聊][{conversation_instance.private_name}] 没有找到初始聊天记录。")
            conversation_instance.observation_info.chat_history_str = "还没有聊天记录。"  # 设置默认提示

    except Exception as load_err:
        # 捕获加载过程中的异常
        logger.error(f"[私聊][{conversation_instance.private_name}] 加载初始聊天记录时出错: {load_err}")
        # 即使出错，也设置一个提示，避免后续使用 None 值
        if conversation_instance.observation_info:
            conversation_instance.observation_info.chat_history_str = "[加载聊天记录出错]"


async def initialize_core_components(conversation_instance: "Conversation"):
    """
    异步初始化对话实例及其所有依赖的核心组件。
    之前是 Conversation 类中的 _initialize 方法。
    """
    # 防止重复初始化 (在 PFCManager层面已经有 _initializing 标志，这里可以简化或移除)
    # if conversation_instance._initialized or conversation_instance._initializing_flag_from_manager: # 假设 manager 设置了一个标志
    # logger.warning(f"[私聊][{conversation_instance.private_name}] 尝试重复初始化或正在初始化中 (initializer)。")
    # return

    # conversation_instance._initializing_flag_from_manager = True # 标记开始初始化
    logger.debug(
        f"[私聊][{conversation_instance.private_name}] (Initializer) 开始初始化对话实例核心组件: {conversation_instance.stream_id}"
    )

    try:
        # ===== 步骤 0: 优先初始化Info对象和ChatStream =====
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 ObservationInfo...")
        conversation_instance.observation_info = ObservationInfo(conversation_instance.private_name)
        if not conversation_instance.observation_info.bot_id:
            logger.warning(
                f"[私聊][{conversation_instance.private_name}] (Initializer) ObservationInfo 未能自动获取 bot_id，尝试手动设置。"
            )
            conversation_instance.observation_info.bot_id = conversation_instance.bot_qq_str

        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 ConversationInfo...")
        conversation_instance.conversation_info = ConversationInfo()

        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 获取 ChatStream...")
        # ChatStream 的获取是后续很多组件的基础，也应该尽早
        chat_stream_instance = chat_manager.get_stream(conversation_instance.stream_id)
        if not chat_stream_instance:
            logger.error(
                f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化错误：无法从 chat_manager 获取 stream_id {conversation_instance.stream_id} 的 ChatStream。"
            )
            # 抛出异常，让PFCManager知道初始化失败
            raise ValueError(f"无法获取 stream_id {conversation_instance.stream_id} 的 ChatStream")
        conversation_instance.chat_stream = chat_stream_instance
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ChatStream 获取成功。")


        # ===== 步骤 1: 初始化核心功能组件 =====
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 SubMind for PFC...")
        # 现在 conversation_info 和 observation_info 肯定已经存在了
        conversation_instance.sub_mind_instance_for_pfc = SubMind(
            subheartflow_id=conversation_instance.stream_id,
            chat_state=None,
            observations=None,
            pfc_conversation_info=conversation_instance.conversation_info,
            pfc_observation_info=conversation_instance.observation_info,
            pfc_chat_stream=conversation_instance.chat_stream # 确保 chat_stream 已被正确赋值
        )
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) 为PFC创建的SubMind实例已初始化。")
        
        # ActionPlanner 和其他组件的初始化不应该在 SubMind 的 else 分支中
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 ActionPlanner...")
        conversation_instance.action_planner = ActionPlanner(
            conversation_instance.stream_id, conversation_instance.private_name
        )
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ActionPlanner 初始化完成。")


        conversation_instance.relationship_updater = PfcRelationshipUpdater(
            private_name=conversation_instance.private_name, bot_name=global_config.bot.nickname
        )
        conversation_instance.relationship_translator = PfcRepationshipTranslator(
            private_name=conversation_instance.private_name
        )
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) PfcRelationship 初始化完成。")

        conversation_instance.emotion_updater = PfcEmotionUpdater(
            private_name=conversation_instance.private_name, bot_name=global_config.bot.nickname
        )
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) PfcEmotion 初始化完成。")

        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 GoalAnalyzer...")
        conversation_instance.goal_analyzer = GoalAnalyzer(
            conversation_instance.stream_id, conversation_instance.private_name
        )
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) GoalAnalyzer 初始化完成。")


        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 ReplyGenerator...")
        conversation_instance.reply_generator = ReplyGenerator(
            conversation_instance.stream_id, conversation_instance.private_name
        )
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ReplyGenerator 初始化完成。")


        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 Waiter...")
        conversation_instance.waiter = Waiter(conversation_instance.stream_id, conversation_instance.private_name)
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) Waiter 初始化完成。")


        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 DirectMessageSender...")
        conversation_instance.direct_sender = DirectMessageSender(conversation_instance.private_name)
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) DirectMessageSender 初始化完成。")


        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 ReplyChecker...")
        conversation_instance.reply_checker = ReplyChecker(
            conversation_instance.stream_id, conversation_instance.private_name
        )
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ReplyChecker 初始化完成。")


        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化 IdleChat...")
        conversation_instance.idle_chat = IdleChat.get_instance(
            conversation_instance.stream_id, conversation_instance.private_name
        )
        # IdleManager 的活跃实例计数器更新是必要的
        await IdleManager._global_lock.acquire()
        try:
            IdleManager._global_active_instances_count += 1
            logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) IdleChat实例已获取并增加活跃计数，当前计数：{IdleManager._global_active_instances_count}")
        finally:
            IdleManager._global_lock.release()
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) IdleChat 初始化完成。")


        # ===== 步骤 2: 初始化信息存储和观察组件 =====
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 获取 ChatObserver 实例...")
        conversation_instance.chat_observer = ChatObserver.get_instance(
            conversation_instance.stream_id, conversation_instance.private_name
        )
        logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ChatObserver 实例获取完成。")


        # ===== 步骤 3: 绑定观察者和信息处理器 =====
        logger.debug(
            f"[私聊][{conversation_instance.private_name}] (Initializer) 绑定 ObservationInfo 到 ChatObserver..."
        )
        # 确保 observation_info 和 chat_observer 都存在
        if conversation_instance.observation_info and conversation_instance.chat_observer:
            conversation_instance.observation_info.bind_to_chat_observer(conversation_instance.chat_observer)
            logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ObservationInfo 绑定 ChatObserver 完成。")
        else:
            logger.error(f"[私聊][{conversation_instance.private_name}] (Initializer) ObservationInfo 或 ChatObserver 未初始化，无法绑定！")


        # ===== 步骤 4: 加载初始聊天记录 =====
        await load_initial_history(conversation_instance) # 这个函数内部会使用 observation_info

        # ===== 步骤 4.1: 加载用户数据 (person_id, 关系文本, 情绪文本) =====
        # person_id
        if conversation_instance.conversation_info and conversation_instance.chat_stream:
            person_id_tuple = await get_person_id(
                private_name=conversation_instance.private_name,
                chat_stream=conversation_instance.chat_stream,
            )
            if person_id_tuple:
                conversation_instance.conversation_info.person_id = person_id_tuple[0]
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 获取到 person_id: {conversation_instance.conversation_info.person_id}")
            else:
                logger.warning(f"[私聊][{conversation_instance.private_name}] (Initializer) 未能获取 person_id。")
        
        # 关系文本
        if (conversation_instance.conversation_info and
            conversation_instance.conversation_info.person_id and
            conversation_instance.relationship_translator and
            conversation_instance.person_info_mng):
            try:
                numeric_relationship_value = await conversation_instance.person_info_mng.get_value(
                    conversation_instance.conversation_info.person_id, "relationship_value"
                )
                # ... (处理Decimal128的逻辑不变) ...
                if not isinstance(numeric_relationship_value, (int, float)):
                    from bson.decimal128 import Decimal128
                    if isinstance(numeric_relationship_value, Decimal128):
                        numeric_relationship_value = float(numeric_relationship_value.to_decimal())
                    else:
                        numeric_relationship_value = 0.0
                conversation_instance.conversation_info.relationship_text = (
                    await conversation_instance.relationship_translator.translate_relationship_value_to_text(
                        numeric_relationship_value
                    )
                )
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化时加载关系文本: {conversation_instance.conversation_info.relationship_text}")
            except Exception as e_init_rel: # ... (错误处理不变)
                logger.error(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化时加载关系文本出错: {e_init_rel}")
                if conversation_instance.conversation_info : conversation_instance.conversation_info.relationship_text = "你们的关系是：普通。"

        # 情绪文本
        if conversation_instance.conversation_info and conversation_instance.mood_mng:
            try:
                conversation_instance.conversation_info.current_emotion_text = (
                    conversation_instance.mood_mng.get_mood_prompt()
                )
                logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化时加载情绪文本: {conversation_instance.conversation_info.current_emotion_text}")
            except Exception as e_init_emo: # ... (错误处理不变)
                logger.error(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化时加载情绪文本出错: {e_init_emo}")


        # ===== 步骤 5: 启动需要后台运行的组件 =====
        logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) 启动 ChatObserver...")
        if conversation_instance.chat_observer:
            conversation_instance.chat_observer.start() # ChatObserver 需要启动其内部循环
            logger.info(f"[私聊][{conversation_instance.private_name}] (Initializer) ChatObserver 已启动。")


        # IdleChat 实例在 get_instance 时如果不存在会被创建，其全局检查循环由 IdleManager 统一管理启动
        # 此处不需要单独启动 IdleChat 的某个循环
        if conversation_instance.idle_chat:
            logger.debug(f"[私聊][{conversation_instance.private_name}] (Initializer) IdleChat 实例已通过 get_instance 获取/创建。")


        # ===== 步骤 6: 设置最终状态 =====
        conversation_instance.state = ConversationState.ANALYZING # 设置初始状态为分析
        logger.info(
            f"[私聊][{conversation_instance.private_name}] (Initializer) 对话实例 {conversation_instance.stream_id} 核心组件初始化完成，状态设为 ANALYZING。"
        )

    except ValueError as ve: # 捕获 ValueError (例如 get_stream 失败)
        logger.error(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化对话实例核心组件时发生值错误: {ve}", exc_info=True)
        raise # 重新抛出，让PFCManager知道初始化失败
    except Exception as e:
        logger.error(f"[私聊][{conversation_instance.private_name}] (Initializer) 初始化对话实例核心组件时发生未知错误: {e}", exc_info=True)
        raise # 重新抛出，让PFCManager知道初始化失败
