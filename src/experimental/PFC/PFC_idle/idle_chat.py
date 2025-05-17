# TODO: 优化 idle 逻辑 增强其与 PFC 模式的联动
from typing import Optional, Dict, Set, List
import asyncio
import time
import random
import traceback
from datetime import datetime
from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.models.utils_model import LLMRequest
from src.chat.message_receive.chat_stream import chat_manager, ChatManager
from src.chat.person_info.person_info import person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.utils.chat_message_builder import build_readable_messages

# from ...schedule.schedule_generator import bot_schedule
from ..chat_observer import ChatObserver
from ..message_sender import DirectMessageSender
from src.chat.message_receive.chat_stream import ChatStream
from ..pfc_relationship import PfcRepationshipTranslator, PfcRelationshipUpdater
from maim_message import UserInfo, Seg
from rich.traceback import install
from bson.decimal128 import Decimal128

install(extra_lines=3)

logger = get_logger("pfc_idle_chat")

class IdleChat:
    """主动聊天组件（测试中）

    在以下条件都满足时触发主动聊天：
    1. 当前没有任何活跃的对话实例
    2. 在指定的活动时间内（7:00-23:00）
    3. 根据关系值动态调整触发概率
    4. 上次触发后已经过了足够的冷却时间
    """

    # 单例模式实现
    _instances: Dict[str, "IdleChat"] = {}

    # 全局共享状态，用于跟踪未回复的用户
    _pending_replies: Dict[str, float] = {}  # 用户名 -> 发送时间
    _tried_users: Set[str] = set()  # 已尝试过的用户集合
    _global_lock = asyncio.Lock()  # 保护共享状态的全局锁
    _initialization_task = None  # 初始化任务
    
    # 全局共享时间状态
    _last_global_trigger_time: float = time.time() - 2000  # 最后一次全局触发时间，初始设为过去时间
    _global_active_instances_count: int = 0  # 全局活跃实例计数
    _global_check_task = None  # 全局检查任务

    @classmethod
    def get_instance(cls, stream_id: str, private_name: str) -> "IdleChat":
        """获取IdleChat实例（单例模式）

        Args:
            stream_id: 聊天流ID
            private_name: 私聊用户名称

        Returns:
            IdleChat: IdleChat实例
        """
        key = f"{private_name}:{stream_id}"
        if key not in cls._instances:
            cls._instances[key] = cls(stream_id, private_name)
            # 创建实例时自动启动检测
            # cls._instances[key].start()  # 不再启动独立检测，改为全局检测
            logger.info(f"[私聊][{private_name}]创建新的IdleChat实例")
        return cls._instances[key]

    @classmethod
    async def start_global_check(cls):
        """启动全局检查任务，所有实例共享一个时间判断"""
        if cls._global_check_task is None or cls._global_check_task.done():
            cls._global_check_task = asyncio.create_task(cls._global_check_loop())
            logger.info("已启动全局主动聊天检查任务")
    
    @classmethod
    async def _global_check_loop(cls):
        """全局检查循环，替代各实例独立的检查循环"""
        try:
            check_interval = global_config.pfc.idle_check_interval * 60  # 检查间隔（默认10分钟，转换为秒）
            min_cooldown = global_config.pfc.min_cooldown  # 最短冷却时间
            max_cooldown = global_config.pfc.max_cooldown  # 最长冷却时间
            active_hours_start = 6  # 活动开始时间
            active_hours_end = 24  # 活动结束时间
            
            while True:
                # 检查是否启用了主动聊天功能
                if not global_config.pfc.enable_idle_chat:
                    logger.debug("主动聊天功能已禁用，等待启用")
                    await asyncio.sleep(60)  # 每分钟检查一次配置变更
                    continue
                
                # 获取可用的实例列表
                active_instances = list(cls._instances.values())
                if not active_instances:
                    logger.debug("暂无可用的主动聊天实例，等待下一次检查")
                    await asyncio.sleep(check_interval)
                    continue
                
                # 检查是否在活动时间内
                current_hour = datetime.now().hour
                if not (active_hours_start <= current_hour < active_hours_end):
                    logger.debug(f"当前时间 {current_hour}:00 不在活动时间内 ({active_hours_start}:00-{active_hours_end}:00)，等待下一次检查")
                    await asyncio.sleep(check_interval)
                    continue
                
                # 检查是否有活跃实例
                if cls._global_active_instances_count > 0:
                    logger.debug(f"存在活跃实例 ({cls._global_active_instances_count})，不触发主动聊天")
                    await asyncio.sleep(check_interval)
                    continue
                
                # 检查冷却时间
                current_time = time.time()
                time_since_last_trigger = current_time - cls._last_global_trigger_time
                if time_since_last_trigger < min_cooldown:
                    time_left = min_cooldown - time_since_last_trigger
                    logger.debug(f"全局冷却时间未到(已过{time_since_last_trigger:.0f}秒/需要{min_cooldown}秒)，还需等待{time_left:.0f}秒")
                    await asyncio.sleep(min(time_left + 1, check_interval))  # 等待剩余冷却时间或检查间隔，取较小值
                    continue
                
                # 基础几率
                base_trigger_probability = 0.3
                
                # 强制触发检查 - 如果超过最大冷却时间，增加触发概率
                if time_since_last_trigger > max_cooldown * 2:
                    force_probability = min(0.8, base_trigger_probability * 3)
                    if random.random() < force_probability:
                        logger.info(f"超过最大冷却时间({time_since_last_trigger:.0f}秒)，强制触发主动聊天选择")
                        selected_instance = await cls._select_instance_to_trigger(active_instances)
                        if selected_instance:
                            logger.info(f"选择了实例 {selected_instance.private_name} 进行主动聊天，立即启动")
                            # 更新全局触发时间
                            cls._last_global_trigger_time = time.time()
                            # 创建任务执行聊天，避免阻塞检查循环
                            asyncio.create_task(selected_instance._initiate_chat())
                            cls._last_global_trigger_time = time.time()
                            await asyncio.sleep(check_interval)
                            continue
                
                # 正常触发检查
                should_trigger = random.random() < base_trigger_probability
                if should_trigger:
                    logger.info("随机触发主动聊天检查，开始选择实例")
                    selected_instance = await cls._select_instance_to_trigger(active_instances)
                    if selected_instance:
                        logger.info(f"选择了实例 {selected_instance.private_name} 进行主动聊天，立即启动")
                        # 更新全局触发时间
                        cls._last_global_trigger_time = time.time()
                        # 创建任务执行聊天，避免阻塞检查循环
                        asyncio.create_task(selected_instance._initiate_chat())
                
                # 等待下一次检查
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.debug("全局主动聊天检测任务被取消")
        except Exception as e:
            logger.error(f"全局主动聊天检测出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 尝试重新启动检测循环
            cls._global_check_task = asyncio.create_task(cls._global_check_loop())
    
    @classmethod
    async def _select_instance_to_trigger(cls, instances: List["IdleChat"]) -> Optional["IdleChat"]:
        """基于关系值和其他因素选择一个实例来触发主动聊天
        
        Returns:
            Optional[IdleChat]: 选定的实例，如果没有合适的实例则返回None
        """
        if not instances:
            return None
        
        try:
            logger.info(f"开始选择实例，共有 {len(instances)} 个候选实例")
            # 获取并记录所有实例的关系值
            instances_with_rel = []
            for instance in instances:
                try:
                    # 查看是否在待回复列表中
                    in_pending = instance.private_name in cls._pending_replies
                    
                    # 获取关系数据
                    try:
                        platform = "qq"  # 假设用户来自QQ平台
                        # 获取关系值
                        person_id = person_info_manager.get_person_id(platform, instance.private_name)
                        rel_value = await person_info_manager.get_value(person_id, "relationship_value")
                        # 确保关系值是浮点数
                        if isinstance(rel_value, Decimal128):
                            rel_value = float(rel_value.to_decimal())
                        elif rel_value is None:
                            rel_value = 0.0
                        else:
                            rel_value = float(rel_value)
                        
                        # 计算关系等级
                        rel_level_num = relationship_manager.calculate_level_num(rel_value)
                        relationship_level = ["厌恶", "冷漠", "一般", "友好", "喜欢", "依赖"]
                        rel_level_name = relationship_level[rel_level_num]
                        
                        logger.info(f"[私聊][{instance.private_name}]关系值: {rel_value:.2f}, 关系等级: {rel_level_name}")
                        relationship_description = rel_level_name
                    except Exception as e:
                        logger.error(f"[私聊][{instance.private_name}]获取关系数据失败: {str(e)}")
                        logger.error(traceback.format_exc())
                        # 使用默认关系描述
                        relationship_description = "普通"
                    
                    # 记录实例信息
                    instances_with_rel.append({
                        "instance": instance,
                        "rel_value": rel_value,
                        "rel_level": rel_level_num,
                        "in_pending": in_pending,
                        "weight": 0  # 初始权重为0
                    })
                    
                except Exception as e:
                    logger.error(f"处理实例 {instance.private_name} 时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 如果没有获取到有效数据，随机选择一个实例
            if not instances_with_rel:
                logger.warning(f"没有任何有效实例，将随机选择一个")
                return random.choice(instances)
            
            logger.info(f"成功获取 {len(instances_with_rel)} 个实例的数据")
            
            # 计算每个实例的权重
            for data in instances_with_rel:
                # 基础权重：关系等级0=1, 1=2, 2=3, 3=4, 4=5, 5=6
                base_weight = data["rel_level"] + 1
                
                # 如果在待回复列表中，大幅降低权重
                if data["in_pending"]:
                    base_weight *= 0.1
                
                # 最终权重
                data["weight"] = base_weight
                logger.debug(f"实例 {data['instance'].private_name} 的权重: {data['weight']}")
            
            # 按权重进行加权随机选择
            total_weight = sum(data["weight"] for data in instances_with_rel)
            if total_weight <= 0:
                # 如果所有权重都是0，随机选择
                logger.warning(f"所有实例的权重都为0，将随机选择一个实例")
                selected = random.choice(instances_with_rel)
            else:
                # 加权随机选择
                rand_val = random.uniform(0, total_weight)
                current = 0
                selected = instances_with_rel[0]
                for data in instances_with_rel:
                    current += data["weight"]
                    if rand_val <= current:
                        selected = data
                        break
            
            logger.info(f"选择了用户 {selected['instance'].private_name} 进行主动聊天")
            return selected["instance"]
            
        except Exception as e:
            logger.error(f"选择实例时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 出错时随机选择一个实例
            logger.warning(f"由于错误，将随机选择一个实例")
            return random.choice(instances)

    @classmethod
    async def initialize_all_streams(cls):
        """初始化所有可用的聊天流的主动聊天功能
        
        从数据库加载所有聊天流，并为每个私聊流创建一个IdleChat实例
        """
        try:
            logger.info("开始初始化所有可用聊天流的主动聊天功能")
            
            # 获取所有聊天流 - 使用现有的load_all_streams方法
            await chat_manager.load_all_streams()
            all_streams = chat_manager.streams
            
            logger.info(f"共加载 {len(all_streams)} 个聊天流，开始筛选私聊流")
            
            # 改进私聊流的识别逻辑 - 基于ChatStream对象的属性而非stream_id
            private_streams = {}
            for stream_id, stream in all_streams.items():
                # 详细日志记录当前流信息，帮助调试
                stream_info = f"ID: {stream_id}"
                if hasattr(stream, "user_info") and stream.user_info:
                    stream_info += f", user: {stream.user_info.user_nickname if hasattr(stream.user_info, 'user_nickname') else 'None'}"
                if hasattr(stream, "group_info") and stream.group_info:
                    stream_info += f", group: {stream.group_info.group_name if hasattr(stream.group_info, 'group_name') else 'None'}"
                logger.debug(f"检查流: {stream_info}")
                
                # 私聊判断标准: 有user_info但没有group_info
                if (hasattr(stream, "user_info") and stream.user_info and 
                    (not hasattr(stream, "group_info") or stream.group_info is None)):
                    logger.info(f"找到私聊流: {stream_info}")
                    private_streams[stream_id] = stream
                # 备用判断: 流ID中包含私聊相关标识
                elif stream_id and isinstance(stream_id, str) and ("private" in stream_id.lower() or "direct" in stream_id.lower()):
                    logger.info(f"基于ID找到可能的私聊流: {stream_info}")
                    private_streams[stream_id] = stream
            
            logger.info(f"找到 {len(private_streams)} 个私聊流，开始创建IdleChat实例")
            
            # 为每个私聊流创建IdleChat实例
            for stream_id, stream in private_streams.items():
                try:
                    # 获取私聊用户名称
                    private_name = None
                    
                    # 尝试获取用户昵称或ID
                    if hasattr(stream, "user_info") and stream.user_info:
                        if hasattr(stream.user_info, "user_nickname") and stream.user_info.user_nickname:
                            private_name = stream.user_info.user_nickname
                        elif hasattr(stream.user_info, "user_id") and stream.user_info.user_id:
                            private_name = str(stream.user_info.user_id)
                    
                    # 如果无法从user_info获取名称，尝试从stream_id解析
                    if not private_name and stream_id and isinstance(stream_id, str):
                        # 尝试各种可能的格式解析
                        if "_" in stream_id:
                            parts = stream_id.split("_")
                            # 尝试找到可能是用户ID的部分
                            for part in parts:
                                if part.isdigit() or (part and not any(kw in part.lower() for kw in ["private", "direct", "chat"])):
                                    private_name = part
                                    break
                    
                    # 如果仍无法获取名称，使用stream_id作为最后手段
                    if not private_name:
                        private_name = stream_id
                        logger.warning(f"无法获取流 {stream_id} 的用户名称，使用stream_id作为替代")
                    
                    # 创建IdleChat实例
                    cls.get_instance(stream_id, private_name)
                    logger.info(f"为用户 {private_name} 创建了IdleChat实例")
                except Exception as e:
                    logger.error(f"为流 {stream_id} 创建IdleChat实例时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            logger.success(f"所有可用聊天流的主动聊天功能初始化完成，共创建 {len(private_streams)} 个实例")
            
            # 启动全局检查任务
            await cls.start_global_check()
            logger.info("已启动全局主动聊天检查任务")
            
        except Exception as e:
            logger.error(f"初始化所有聊天流时出错: {str(e)}")
            logger.error(traceback.format_exc())

    @classmethod
    async def register_user_response(cls, private_name: str) -> None:
        """注册用户已回复

        当用户回复消息时调用此方法，将用户从待回复列表中移除

        Args:
            private_name: 私聊用户名称
        """
        async with cls._global_lock:
            if private_name in cls._pending_replies:
                del cls._pending_replies[private_name]
                logger.info(f"[私聊][{private_name}]已回复主动聊天消息，从待回复列表中移除")

    @classmethod
    async def get_next_available_user(cls) -> Optional[str]:
        """获取下一个可用于主动聊天的用户

        优先选择未尝试过的用户，其次是已尝试但超时未回复的用户

        Returns:
            Optional[str]: 下一个可用的用户名，如果没有则返回None
        """
        async with cls._global_lock:
            current_time = time.time()
            timeout_threshold = 120 # 500秒未回复视为超时

            # 清理超时未回复的用户
            for user, send_time in list(cls._pending_replies.items()):
                if current_time - send_time > timeout_threshold:
                    logger.info(f"[私聊][{user}]超过{timeout_threshold}秒未回复，标记为超时")
                    del cls._pending_replies[user]

            # 获取所有实例中的用户
            all_users = set()
            for key in cls._instances:
                user = key.split(":", 1)[0]
                all_users.add(user)

            # 优先选择未尝试过的用户
            untried_users = all_users - cls._tried_users
            if untried_users:
                next_user = random.choice(list(untried_users))
                cls._tried_users.add(next_user)
                return next_user

            # 如果所有用户都已尝试过，重置尝试集合，从头开始
            if len(cls._tried_users) >= len(all_users):
                cls._tried_users.clear()
                logger.info("[私聊]所有用户都已尝试过，重置尝试列表")
                # 随机选择一个不在待回复列表中的用户
                available_users = all_users - set(cls._pending_replies.keys())
                if available_users:
                    next_user = random.choice(list(available_users))
                    cls._tried_users.add(next_user)
                    return next_user

            return None

    def __init__(self, stream_id: str, private_name: str):
        """初始化主动聊天组件

        Args:
            stream_id: 聊天流ID
            private_name: 私聊用户名称
        """
        self.stream_id = stream_id
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.message_sender = DirectMessageSender(private_name)
        
        # 初始化关系组件
        self.relationship_translator = PfcRepationshipTranslator(private_name)
        self.relationship_updater = PfcRelationshipUpdater(private_name, global_config.bot.nickname)

        # LLM请求对象，用于生成主动对话内容
        self.llm = LLMRequest(model=global_config.model.normal, temperature=0.5, max_tokens=500, request_type="idle_chat")

        # 配置参数 - 从global_config加载
        self.min_cooldown = global_config.pfc.min_cooldown  # 最短冷却时间（默认2小时）
        self.max_cooldown = global_config.pfc.max_cooldown  # 最长冷却时间（默认5小时）
        self.check_interval = global_config.pfc.idle_check_interval * 60  # 检查间隔（默认10分钟，转换为秒）
        self.active_hours_start = 6  # 活动开始时间
        self.active_hours_end = 24  # 活动结束时间

    def start(self) -> None:
        """启动主动聊天检测"""
        logger.info(f"[私聊][{self.private_name}]主动聊天功能已通过全局检测机制启动")

    def stop(self) -> None:
        """停止主动聊天检测"""
        logger.info(f"[私聊][{self.private_name}]主动聊天功能已停止")

    async def increment_active_instances(self) -> None:
        """增加活跃实例计数

        当创建新的对话实例时调用此方法
        """
        async with self.__class__._global_lock:
            self.__class__._global_active_instances_count += 1
            logger.debug(f"[私聊][{self.private_name}]全局活跃实例数+1，当前：{self.__class__._global_active_instances_count}")

    async def decrement_active_instances(self) -> None:
        """减少活跃实例计数

        当对话实例结束时调用此方法
        """
        async with self.__class__._global_lock:
            self.__class__._global_active_instances_count = max(0, self.__class__._global_active_instances_count - 1)
            logger.debug(f"[私聊][{self.private_name}]全局活跃实例数-1，当前：{self.__class__._global_active_instances_count}")

    async def update_last_message_time(self, message_time: Optional[float] = None) -> None:
        """更新最后一条消息的时间

        Args:
            message_time: 消息时间戳，如果为None则使用当前时间
        """
        async with self.__class__._global_lock:
            self.__class__._last_global_trigger_time = message_time or time.time()
            logger.debug(f"[私聊][{self.private_name}]更新全局最后消息时间: {self.__class__._last_global_trigger_time:.2f}")

        # 当用户发送消息时，也应该注册响应
        await self.__class__.register_user_response(self.private_name)

    async def _get_chat_stream(self) -> Optional[ChatStream]:
        """获取聊天流实例"""
        try:
            existing_chat_stream = chat_manager.get_stream(self.stream_id)
            if existing_chat_stream:
                logger.debug(f"[私聊][{self.private_name}]从chat_manager找到现有聊天流")
                return existing_chat_stream

            # 如果没有现有聊天流，则创建新的
            logger.debug(f"[私聊][{self.private_name}]未找到现有聊天流，创建新聊天流")
            # 创建用户信息对象
            user_info = UserInfo(
                user_id=self.private_name,  # 使用私聊用户的ID
                user_nickname=self.private_name,  # 使用私聊用户的名称
                platform="qq",
            )
            # 创建聊天流
            new_stream = ChatStream(self.stream_id, "qq", user_info)
            # 将新创建的聊天流添加到管理器中
            chat_manager.register_stream(new_stream)
            logger.debug(f"[私聊][{self.private_name}]成功创建并注册新聊天流")
            return new_stream
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]创建/获取聊天流失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def _initiate_chat(self) -> None:
        """生成并发送主动聊天消息"""
        try:
            # 获取聊天历史记录
            messages = self.chat_observer.get_cached_messages(limit=12)
            
            # 立即获取历史文本
            chat_history_text = await build_readable_messages(
                messages, replace_bot_name=True, merge_messages=False, timestamp_mode="relative", read_mark=0.0
            )
            
            # 添加触发判定条件
            # 1. 检查是否有活跃对话实例存在
            if self.__class__._global_active_instances_count > 0:
                logger.debug(f"[私聊][{self.private_name}]存在活跃实例，取消主动聊天")
                return
            
            # 2. 检查是否在待回复列表中
            if self.private_name in self.__class__._pending_replies:
                logger.debug(f"[私聊][{self.private_name}]已在待回复列表中，取消主动聊天")
                return
                
            # 此时已通过所有条件判断，可以继续生成和发送消息
            logger.info(f"[私聊][{self.private_name}]通过所有触发条件，开始准备主动聊天内容")
            
            # 获取关系数据
            try:
                platform = "qq"  # 假设用户来自QQ平台
                # 获取关系值
                person_id = person_info_manager.get_person_id(platform, self.private_name)
                rel_value = await person_info_manager.get_value(person_id, "relationship_value")
                # 确保关系值是浮点数
                if isinstance(rel_value, Decimal128):
                    rel_value = float(rel_value.to_decimal())
                elif rel_value is None:
                    rel_value = 0.0
                else:
                    rel_value = float(rel_value)
                
                # 计算关系等级
                rel_level_num = relationship_manager.calculate_level_num(rel_value)
                relationship_level = ["厌恶", "冷漠", "一般", "友好", "喜欢", "依赖"]
                rel_level_name = relationship_level[rel_level_num]
                
                logger.info(f"[私聊][{self.private_name}]关系值: {rel_value:.2f}, 关系等级: {rel_level_name}")
                relationship_description = rel_level_name
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]获取关系数据失败: {str(e)}")
                logger.error(traceback.format_exc())
                # 使用默认关系描述
                relationship_description = "普通"
            
            # 构建提示词
            current_time = datetime.now().strftime("%H:%M")
            prompt = f"""你是{global_config.bot.nickname}。
            你正在与用户{self.private_name}进行QQ私聊，你们的关系是{relationship_description}
            现在时间{current_time}
            
            你想要主动发起对话。
            请基于以下之前的对话历史，生成一条自然、友好、符合关系程度的主动对话消息。
            这条消息应能够引起用户的兴趣，重新开始对话。
            最近的对话历史（并不是现在的对话）：
            {chat_history_text}
            请你严格根据对话历史决定是告诉对方你正在做的事情，还是询问对方正在做的事情
            请直接输出一条消息，不要有任何额外的解释或引导文字
            消息内容尽量简短
            """

            # 生成回复
            logger.debug(f"[私聊][{self.private_name}]开始生成主动聊天内容")
            try:
                content, _ = await asyncio.wait_for(self.llm.generate_response_async(prompt), timeout=30)
                logger.debug(f"[私聊][{self.private_name}]成功生成主动聊天内容: {content}")
            except asyncio.TimeoutError:
                logger.error(f"[私聊][{self.private_name}]生成主动聊天内容超时")
                return
            except Exception as llm_err:
                logger.error(f"[私聊][{self.private_name}]生成主动聊天内容失败: {str(llm_err)}")
                logger.error(traceback.format_exc())
                return
                
            # 清理结果
            content = content.strip()
            content = content.strip("\"'")

            if not content:
                logger.error(f"[私聊][{self.private_name}]生成的主动聊天内容为空")
                return

            # 获取聊天流
            chat_stream = await self._get_chat_stream()
            if not chat_stream:
                logger.error(f"[私聊][{self.private_name}]无法获取有效的聊天流，取消发送主动消息")
                return

            # 发送消息
            try:
                segments = Seg(type="seglist", data=[Seg(type="text", data=content)])
                logger.debug(f"[私聊][{self.private_name}]准备发送主动聊天消息: {content}")
                await self.message_sender.send_message(
                    chat_stream=chat_stream, segments=segments, reply_to_message=None, content=content
                )
                logger.info(f"[私聊][{self.private_name}]成功主动发起聊天: {content}")
                
                # 将此用户添加到待回复列表
                async with self.__class__._global_lock:
                    self.__class__._pending_replies[self.private_name] = time.time()
                    self.__class__._tried_users.add(self.private_name)
                    logger.info(f"[私聊][{self.private_name}]已添加到等待回复列表中")
                
            except Exception as e:
                logger.error(f"[私聊][{self.private_name}]发送主动聊天消息失败: {str(e)}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]主动发起聊天过程中发生未预期的错误: {str(e)}")
            logger.error(traceback.format_exc())

# 启动自动初始化任务
if __name__ == "__main__":
    asyncio.run(IdleChat.initialize_all_streams())
else:
    # 在导入时，使用create_task异步启动，避免阻塞导入过程
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，则创建任务
            asyncio.create_task(IdleChat.initialize_all_streams())
            logger.info("已在现有事件循环中启动主动聊天初始化任务")
        else:
            # 如果事件循环未运行，则需要另外处理
            logger.info("事件循环未运行，将在适当时机启动主动聊天初始化任务")
            # 可以在这里添加稍后启动的逻辑，或者由外部调用 IdleChat.start_initialization_task()
    except RuntimeError:
        logger.warning("无法获取事件循环，将由外部调用启动主动聊天初始化任务")
