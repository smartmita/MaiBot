import asyncio
import hashlib
import time
import copy
from typing import Dict, Optional


from ...common.database import db
from maim_message import GroupInfo, UserInfo

from src.common.logger_manager import get_logger
from rich.traceback import install

install(extra_lines=3)


logger = get_logger("chat_stream")


class ChatStream:
    """聊天流对象，存储一个完整的聊天上下文"""

    def __init__(
        self,
        stream_id: str,
        platform: str,
        user_info: UserInfo,
        group_info: Optional[GroupInfo] = None,
        data: dict = None,
    ):
        self.stream_id = stream_id
        self.platform = platform
        self.user_info = user_info
        self.group_info = group_info
        self.create_time = data.get("create_time", time.time()) if data else time.time()
        self.last_active_time = data.get("last_active_time", self.create_time) if data else self.create_time
        self.saved = False

    def to_dict(self) -> dict:
        """转换为字典格式"""
        result = {
            "stream_id": self.stream_id,
            "platform": self.platform,
            "user_info": self.user_info.to_dict() if self.user_info else None,
            "group_info": self.group_info.to_dict() if self.group_info else None,
            "create_time": self.create_time,
            "last_active_time": self.last_active_time,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ChatStream":
        """从字典创建实例"""
        user_info = UserInfo.from_dict(data.get("user_info", {})) if data.get("user_info") else None
        group_info = GroupInfo.from_dict(data.get("group_info", {})) if data.get("group_info") else None

        return cls(
            stream_id=data["stream_id"],
            platform=data["platform"],
            user_info=user_info,
            group_info=group_info,
            data=data,
        )

    def update_active_time(self):
        """更新最后活跃时间"""
        self.last_active_time = time.time()
        self.saved = False


class ChatManager:
    """聊天管理器，管理所有聊天流"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.streams: Dict[str, ChatStream] = {}  # stream_id -> ChatStream
            self._ensure_collection()
            self._initialized = True
            # 在事件循环中启动初始化
            # asyncio.create_task(self._initialize())
            # # 启动自动保存任务
            # asyncio.create_task(self._auto_save_task())

    async def _initialize(self):
        """异步初始化"""
        try:
            await self.load_all_streams()
            logger.success(f"聊天管理器已启动，已加载 {len(self.streams)} 个聊天流")
        except Exception as e:
            logger.error(f"聊天管理器启动失败: {str(e)}")

    async def _auto_save_task(self):
        """定期自动保存所有聊天流"""
        while True:
            await asyncio.sleep(300)  # 每5分钟保存一次
            try:
                await self._save_all_streams()
                logger.info("聊天流自动保存完成")
            except Exception as e:
                logger.error(f"聊天流自动保存失败: {str(e)}")

    @staticmethod
    def _ensure_collection():
        """确保数据库集合存在并创建索引"""
        if "chat_streams" not in db.list_collection_names():
            db.create_collection("chat_streams")
            # 创建索引
            db.chat_streams.create_index([("stream_id", 1)], unique=True)
            db.chat_streams.create_index([("platform", 1), ("user_info.user_id", 1), ("group_info.group_id", 1)])

    @staticmethod
    def _generate_stream_id(platform: str, user_info: UserInfo, group_info: Optional[GroupInfo] = None) -> str:
        """生成聊天流唯一ID"""
        if group_info:
            # 组合关键信息
            components = [platform, str(group_info.group_id)]
        else:
            components = [platform, str(user_info.user_id), "private"]

        # 使用MD5生成唯一ID
        key = "_".join(components)
        return hashlib.md5(key.encode()).hexdigest()

    async def get_or_create_stream(
        self, platform: str, user_info: UserInfo, group_info: Optional[GroupInfo] = None
    ) -> ChatStream:
        """获取或创建聊天流

        Args:
            platform: 平台标识
            user_info: 用户信息
            group_info: 群组信息（可选）

        Returns:
            ChatStream: 聊天流对象
        """
        stream_id = self._generate_stream_id(platform, user_info, group_info)

        # mes_name: 当前调用此函数时，根据传入参数确定的最新聊天名称
        current_mes_name = None
        if group_info and group_info.group_name:
            current_mes_name = group_info.group_name
        elif user_info and user_info.user_nickname: # private chat
            current_mes_name = f"{user_info.user_nickname}的私聊"

        # 检查内存中是否存在
        if stream_id in self.streams:
            stream_instance = self.streams[stream_id] # 获取内存中的实际stream对象

            # stream_name: 从内存中获取的、可能存在的旧名称，用于比较
            # 这个名称是 get_stream_name() 在此函数更新stream_instance前会返回的名称
            old_effective_stream_name = None
            if stream_instance.group_info and stream_instance.group_info.group_name:
                old_effective_stream_name = stream_instance.group_info.group_name
            elif stream_instance.user_info and stream_instance.user_info.user_nickname:
                old_effective_stream_name = f"{stream_instance.user_info.user_nickname}的私聊"

            # 判断名称是否发生变化
            if current_mes_name is not None and old_effective_stream_name != current_mes_name:
                logger.info(f"聊天流名称变更 (来自内存): ID '{stream_id}' 从 '{old_effective_stream_name}' 变为 '{current_mes_name}'. 更新数据库.")
                # 更新内存中的stream_instance对象的 用户信息和群组信息（这将包含新的名称）
                stream_instance.user_info = user_info
                if group_info:
                    stream_instance.group_info = group_info
                else: # 处理从群聊变私聊或群信息消失的情况
                    stream_instance.group_info = None
                stream_instance.update_active_time() # 更新活跃时间也会将saved标记为False
                await self._save_stream(stream_instance) # 保存更新到数据库
            else:
                # 名称未变或无法确定当前新名称，仅更新活动时间和确保信息最新
                stream_instance.user_info = user_info
                if group_info:
                    stream_instance.group_info = group_info
                else:
                    stream_instance.group_info = None
                stream_instance.update_active_time()

            # self.streams[stream_id] 已经是更新后的 stream_instance
            return copy.deepcopy(self.streams[stream_id]) # 返回一个副本，避免外部直接修改缓存

        # 检查数据库中是否存在
        data = db.chat_streams.find_one({"stream_id": stream_id})
        if data:
            stream_from_db = ChatStream.from_dict(data)

            # stream_name: 从数据库加载的、可能存在的旧名称，用于比较
            db_stream_name = None
            if stream_from_db.group_info and stream_from_db.group_info.group_name:
                db_stream_name = stream_from_db.group_info.group_name
            elif stream_from_db.user_info and stream_from_db.user_info.user_nickname:
                db_stream_name = f"{stream_from_db.user_info.user_nickname}的私聊"

            # 判断名称是否发生变化
            if current_mes_name is not None and db_stream_name != current_mes_name:
                logger.info(f"聊天流名称变更 (来自数据库): ID '{stream_id}' 从 '{db_stream_name}' 变为 '{current_mes_name}'. 更新数据库.")
                # 更新从数据库加载的stream对象的 用户信息和群组信息
                stream_from_db.user_info = user_info
                if group_info:
                    stream_from_db.group_info = group_info
                else:
                    stream_from_db.group_info = None
                # stream_from_db.update_active_time() # 将在下面统一处理
                # stream_from_db.saved = False # update_active_time会处理

            # 统一更新信息和活跃时间
            stream_from_db.user_info = user_info
            if group_info:
                stream_from_db.group_info = group_info
            else:
                stream_from_db.group_info = None
            stream_from_db.update_active_time()

            self.streams[stream_id] = stream_from_db # 存入内存缓存
            await self._save_stream(stream_from_db) # 保存到数据库（如果名称变化或活跃时间更新）
            return copy.deepcopy(stream_from_db)

        # 创建新的聊天流
        # 此时，新流的名称自然就是 current_mes_name
        new_stream = ChatStream(
            stream_id=stream_id,
            platform=platform,
            user_info=user_info,
            group_info=group_info,
        )
        self.streams[stream_id] = new_stream
        await self._save_stream(new_stream)
        logger.info(f"创建了新的聊天流: ID '{stream_id}', 名称 '{current_mes_name if current_mes_name else '未知'}'")
        return copy.deepcopy(new_stream)

    def get_stream(self, stream_id: str) -> Optional[ChatStream]:
        """通过stream_id获取聊天流"""
        return self.streams.get(stream_id)

    def get_stream_by_info(
        self, platform: str, user_info: UserInfo, group_info: Optional[GroupInfo] = None
    ) -> Optional[ChatStream]:
        """通过信息获取聊天流"""
        stream_id = self._generate_stream_id(platform, user_info, group_info)
        return self.streams.get(stream_id)

    def get_stream_name(self, stream_id: str) -> Optional[str]:
        """根据 stream_id 获取聊天流名称"""
        stream = self.get_stream(stream_id)
        if not stream:
            return None

        if stream.group_info and stream.group_info.group_name:
            return stream.group_info.group_name
        elif stream.user_info and stream.user_info.user_nickname:
            return f"{stream.user_info.user_nickname}的私聊"
        else:
            # 如果没有群名或用户昵称，返回 None 或其他默认值
            return None

    @staticmethod
    async def _save_stream(stream: ChatStream):
        """保存聊天流到数据库"""
        if not stream.saved:
            db.chat_streams.update_one({"stream_id": stream.stream_id}, {"$set": stream.to_dict()}, upsert=True)
            stream.saved = True

    async def _save_all_streams(self):
        """保存所有聊天流"""
        for stream in self.streams.values():
            await self._save_stream(stream)

    async def load_all_streams(self):
        """从数据库加载所有聊天流"""
        all_streams = db.chat_streams.find({})
        for data in all_streams:
            stream = ChatStream.from_dict(data)
            self.streams[stream.stream_id] = stream


# 创建全局单例
chat_manager = ChatManager()