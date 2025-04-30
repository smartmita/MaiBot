import asyncio
import base64
import hashlib
import os
import random
import time
import traceback
from typing import Optional, Tuple
from PIL import Image
import io
import re

from ...common.database import db
from ...config.config import global_config
from ..chat.utils_image import image_path_to_base64, image_manager
from ..models.utils_model import LLMRequest
from src.common.logger_manager import get_logger


logger = get_logger("emoji")

BASE_DIR = os.path.join("data")
EMOJI_DIR = os.path.join(BASE_DIR, "emoji")  # 表情包存储目录
EMOJI_REGISTED_DIR = os.path.join(BASE_DIR, "emoji_registed")  # 已注册的表情包注册目录
MAX_EMOJI_FOR_PROMPT = 20  # 最大允许的表情包描述数量于图片替换的 prompt 中


"""
还没经过测试，有些地方数据库和内存数据同步可能不完全

"""


class MaiEmoji:
    """定义一个表情包"""

    def __init__(self, filename: str, path: str):
        self.path = path  # 存储目录路径
        self.filename = filename
        self.embedding = []
        self.hash = ""  # 初始为空，在创建实例时会计算
        self.description = ""
        self.emotion = []
        self.usage_count = 0
        self.last_used_time = time.time()
        self.register_time = time.time()
        self.is_deleted = False  # 标记是否已被删除
        self.format = ""

    async def initialize_hash_format(self):
        """从文件创建表情包实例

        参数:
            file_path: 文件的完整路径

        返回:
            MaiEmoji: 创建的表情包实例，如果失败则返回None
        """
        try:
            file_path = os.path.join(self.path, self.filename)
            if not os.path.exists(file_path):
                logger.error(f"[错误] 表情包文件不存在: {file_path}")
                return None

            image_base64 = image_path_to_base64(file_path)
            if image_base64 is None:
                logger.error(f"[错误] 无法读取图片: {file_path}")
                return None

            # 计算哈希值
            image_bytes = base64.b64decode(image_base64)
            self.hash = hashlib.md5(image_bytes).hexdigest()

            # 获取图片格式
            self.format = Image.open(io.BytesIO(image_bytes)).format.lower()

        except Exception as e:
            logger.error(f"[错误] 初始化表情包失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def register_to_db(self):
        """
        注册表情包
        将表情包对应的文件，从当前路径移动到EMOJI_REGISTED_DIR目录下
        并修改对应的实例属性，然后将表情包信息保存到数据库中
        """
        try:
            # 确保目标目录存在
            os.makedirs(EMOJI_REGISTED_DIR, exist_ok=True)

            # 源路径是当前实例的完整路径
            source_path = os.path.join(self.path, self.filename)
            # 目标路径
            destination_path = os.path.join(EMOJI_REGISTED_DIR, self.filename)

            # 检查源文件是否存在
            if not os.path.exists(source_path):
                logger.error(f"[错误] 源文件不存在: {source_path}")
                return False

            # --- 文件移动 ---
            try:
                # 如果目标文件已存在，先删除 (确保移动成功)
                if os.path.exists(destination_path):
                    os.remove(destination_path)

                os.rename(source_path, destination_path)
                logger.debug(f"[移动] 文件从 {source_path} 移动到 {destination_path}")
                # 更新实例的路径属性为新目录
                self.path = EMOJI_REGISTED_DIR
            except Exception as move_error:
                logger.error(f"[错误] 移动文件失败: {str(move_error)}")
                return False  # 文件移动失败，不继续

            # --- 数据库操作 ---
            try:
                # 准备数据库记录 for emoji collection
                emoji_record = {
                    "filename": self.filename,
                    "path": os.path.join(self.path, self.filename),  # 使用更新后的路径
                    "embedding": self.embedding,
                    "description": self.description,
                    "emotion": self.emotion,  # 添加情感标签字段
                    "hash": self.hash,
                    "format": self.format,
                    "timestamp": int(self.register_time),  # 使用实例的注册时间
                    "usage_count": self.usage_count,
                    "last_used_time": self.last_used_time,
                }

                # 使用upsert确保记录存在或被更新
                db["emoji"].update_one({"hash": self.hash}, {"$set": emoji_record}, upsert=True)

                logger.success(f"[注册] 表情包信息保存到数据库: {self.emotion}")

                return True

            except Exception as db_error:
                logger.error(f"[错误] 保存数据库失败: {str(db_error)}")
                # 考虑是否需要将文件移回？为了简化，暂时只记录错误
                return False

        except Exception as e:
            logger.error(f"[错误] 注册表情包失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def delete(self):
        """删除表情包

        删除表情包的文件和数据库记录

        返回:
            bool: 是否成功删除
        """
        try:
            # 1. 删除文件
            if os.path.exists(os.path.join(self.path, self.filename)):
                try:
                    os.remove(os.path.join(self.path, self.filename))
                    logger.debug(f"[删除] 文件: {os.path.join(self.path, self.filename)}")
                except Exception as e:
                    logger.error(f"[错误] 删除文件失败 {os.path.join(self.path, self.filename)}: {str(e)}")
                    # 继续执行，即使文件删除失败也尝试删除数据库记录

            # 2. 删除数据库记录
            result = db.emoji.delete_one({"hash": self.hash})
            deleted_in_db = result.deleted_count > 0

            if deleted_in_db:
                logger.info(f"[删除] 表情包 {self.filename} 无对应文件，已删除")

                # 3. 标记对象已被删除
                self.is_deleted = True
                return True
            else:
                logger.error(f"[错误] 删除表情包记录失败: {self.hash}")
                return False

        except Exception as e:
            logger.error(f"[错误] 删除表情包失败: {str(e)}")
            return False


class EmojiManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        self._scan_task = None
        self.vlm = LLMRequest(model=global_config.vlm, temperature=0.3, max_tokens=1000, request_type="emoji")
        self.llm_emotion_judge = LLMRequest(
            model=global_config.llm_normal, max_tokens=600, request_type="emoji"
        )  # 更高的温度，更少的token（后续可以根据情绪来调整温度）

        self.emoji_num = 0
        self.emoji_num_max = global_config.max_emoji_num
        self.emoji_num_max_reach_deletion = global_config.max_reach_deletion
        self.emoji_objects: list[MaiEmoji] = []  # 存储MaiEmoji对象的列表，使用类型注解明确列表元素类型

        logger.info("启动表情包管理器")

    def _ensure_emoji_dir(self):
        """确保表情存储目录存在"""
        os.makedirs(EMOJI_DIR, exist_ok=True)

    def initialize(self):
        """初始化数据库连接和表情目录"""
        if not self._initialized:
            try:
                self._ensure_emoji_collection()
                self._ensure_emoji_dir()
                self._initialized = True
                # 更新表情包数量
                # 启动时执行一次完整性检查
                # await self.check_emoji_file_integrity()
            except Exception:
                logger.exception("初始化表情管理器失败")

    def _ensure_db(self):
        """确保数据库已初始化"""
        if not self._initialized:
            self.initialize()
        if not self._initialized:
            raise RuntimeError("EmojiManager not initialized")

    @staticmethod
    def _ensure_emoji_collection():
        """确保emoji集合存在并创建索引

        这个函数用于确保MongoDB数据库中存在emoji集合,并创建必要的索引。

        索引的作用是加快数据库查询速度:
        - embedding字段的2dsphere索引: 用于加速向量相似度搜索,帮助快速找到相似的表情包
        - tags字段的普通索引: 加快按标签搜索表情包的速度
        - filename字段的唯一索引: 确保文件名不重复,同时加快按文件名查找的速度

        没有索引的话,数据库每次查询都需要扫描全部数据,建立索引后可以大大提高查询效率。
        """
        if "emoji" not in db.list_collection_names():
            db.create_collection("emoji")
            db.emoji.create_index([("embedding", "2dsphere")])
            db.emoji.create_index([("filename", 1)], unique=True)

    def record_usage(self, hash: str):
        """记录表情使用次数"""
        try:
            db.emoji.update_one({"hash": hash}, {"$inc": {"usage_count": 1}})
            for emoji in self.emoji_objects:
                if emoji.hash == hash:
                    emoji.usage_count += 1
                    break

        except Exception as e:
            logger.error(f"记录表情使用失败: {str(e)}")

    async def get_emoji_for_text(self, text_emotion: str) -> Optional[Tuple[str, str]]:
        """根据文本内容获取相关表情包
        Args:
            text_emotion: 输入的情感描述文本
        Returns:
            Optional[Tuple[str, str]]: (表情包文件路径, 表情包描述)，如果没有找到则返回None
        """
        try:
            self._ensure_db()
            time_start = time.time()

            # 获取所有表情包
            all_emojis = self.emoji_objects

            if not all_emojis:
                logger.warning("数据库中没有任何表情包")
                return None

            # 计算每个表情包与输入文本的最大情感相似度
            emoji_similarities = []
            for emoji in all_emojis:
                emotions = emoji.emotion
                if not emotions:
                    continue

                # 计算与每个emotion标签的相似度，取最大值
                max_similarity = 0
                for emotion in emotions:
                    # 使用编辑距离计算相似度
                    distance = self._levenshtein_distance(text_emotion, emotion)
                    max_len = max(len(text_emotion), len(emotion))
                    similarity = 1 - (distance / max_len if max_len > 0 else 0)
                    max_similarity = max(max_similarity, similarity)

                emoji_similarities.append((emoji, max_similarity))

            # 按相似度降序排序
            emoji_similarities.sort(key=lambda x: x[1], reverse=True)

            # 获取前5个最相似的表情包
            top_5_emojis = emoji_similarities[:10] if len(emoji_similarities) > 10 else emoji_similarities

            if not top_5_emojis:
                logger.warning("未找到匹配的表情包")
                return None

            # 从前5个中随机选择一个
            selected_emoji, similarity = random.choice(top_5_emojis)

            # 更新使用次数
            self.record_usage(selected_emoji.hash)

            time_end = time.time()

            logger.info(
                f"找到[{text_emotion}]表情包,用时:{time_end - time_start:.2f}秒: {selected_emoji.description}  (相似度: {similarity:.4f})"
            )
            return selected_emoji.path, f"[ {selected_emoji.description} ]"

        except Exception as e:
            logger.error(f"[错误] 获取表情包失败: {str(e)}")
            return None

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离

        Args:
            s1: 第一个字符串
            s2: 第二个字符串

        Returns:
            int: 编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    async def check_emoji_file_integrity(self):
        """检查表情包文件完整性
        遍历self.emoji_objects中的所有对象，检查文件是否存在
        如果文件已被删除，则执行对象的删除方法并从列表中移除
        """
        try:
            if not self.emoji_objects:
                logger.warning("[检查] emoji_objects为空，跳过完整性检查")
                return

            total_count = len(self.emoji_objects)
            self.emoji_num = total_count
            removed_count = 0
            # 使用列表复制进行遍历，因为我们会在遍历过程中修改列表
            for emoji in self.emoji_objects[:]:
                try:
                    # 检查文件是否存在
                    if not os.path.exists(emoji.path):
                        logger.warning(f"[检查] 表情包文件已被删除: {emoji.path}")
                        # 执行表情包对象的删除方法
                        await emoji.delete()
                        # 从列表中移除该对象
                        self.emoji_objects.remove(emoji)
                        # 更新计数
                        self.emoji_num -= 1
                        removed_count += 1
                        continue

                    if emoji.description == None:
                        logger.warning(f"[检查] 表情包文件已被删除: {emoji.path}")
                        # 执行表情包对象的删除方法
                        await emoji.delete()
                        # 从列表中移除该对象
                        self.emoji_objects.remove(emoji)
                        # 更新计数
                        self.emoji_num -= 1
                        removed_count += 1
                        continue

                except Exception as item_error:
                    logger.error(f"[错误] 处理表情包记录时出错: {str(item_error)}")
                    continue

            await self.clean_unused_emojis(EMOJI_REGISTED_DIR, self.emoji_objects)
            # 输出清理结果
            if removed_count > 0:
                logger.success(f"[清理] 已清理 {removed_count} 个失效的表情包记录")
                logger.info(f"[统计] 清理前: {total_count} | 清理后: {len(self.emoji_objects)}")
            else:
                logger.info(f"[检查] 已检查 {total_count} 个表情包记录，全部完好")

        except Exception as e:
            logger.error(f"[错误] 检查表情包完整性失败: {str(e)}")
            logger.error(traceback.format_exc())

    async def start_periodic_check_register(self):
        """定期检查表情包完整性和数量"""
        await self.get_all_emoji_from_db()
        while True:
            logger.info("[扫描] 开始检查表情包完整性...")
            await self.check_emoji_file_integrity()
            await self.clear_temp_emoji()
            logger.info("[扫描] 开始扫描新表情包...")

            # 检查表情包目录是否存在
            if not os.path.exists(EMOJI_DIR):
                logger.warning(f"[警告] 表情包目录不存在: {EMOJI_DIR}")
                os.makedirs(EMOJI_DIR, exist_ok=True)
                logger.info(f"[创建] 已创建表情包目录: {EMOJI_DIR}")
                await asyncio.sleep(global_config.EMOJI_CHECK_INTERVAL * 60)
                continue

            # 检查目录是否为空
            files = os.listdir(EMOJI_DIR)
            if not files:
                logger.warning(f"[警告] 表情包目录为空: {EMOJI_DIR}")
                await asyncio.sleep(global_config.EMOJI_CHECK_INTERVAL * 60)
                continue

            # 检查是否需要处理表情包(数量超过最大值或不足)
            if (self.emoji_num > self.emoji_num_max and global_config.max_reach_deletion) or (
                self.emoji_num < self.emoji_num_max
            ):
                try:
                    # 获取目录下所有图片文件
                    files_to_process = [
                        f
                        for f in files
                        if os.path.isfile(os.path.join(EMOJI_DIR, f))
                        and f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
                    ]

                    # 处理每个符合条件的文件
                    for filename in files_to_process:
                        # 尝试注册表情包
                        success = await self.register_emoji_by_filename(filename)
                        if success:
                            # 注册成功则跳出循环
                            break
                        else:
                            # 注册失败则删除对应文件
                            file_path = os.path.join(EMOJI_DIR, filename)
                            os.remove(file_path)
                            logger.warning(f"[清理] 删除注册失败的表情包文件: {filename}")
                except Exception as e:
                    logger.error(f"[错误] 扫描表情包目录失败: {str(e)}")

            await asyncio.sleep(global_config.EMOJI_CHECK_INTERVAL * 60)

    async def get_all_emoji_from_db(self):
        """获取所有表情包并初始化为MaiEmoji类对象

        参数:
            hash: 可选，如果提供则只返回指定哈希值的表情包

        返回:
            list[MaiEmoji]: 表情包对象列表
        """
        try:
            self._ensure_db()

            # 获取所有表情包
            all_emoji_data = list(db.emoji.find())

            # 将数据库记录转换为MaiEmoji对象
            emoji_objects = []
            for emoji_data in all_emoji_data:
                emoji = MaiEmoji(
                    filename=emoji_data.get("filename", ""),
                    path=emoji_data.get("path", ""),
                )

                # 设置额外属性
                emoji.hash = emoji_data.get("hash", "")
                emoji.usage_count = emoji_data.get("usage_count", 0)
                emoji.last_used_time = emoji_data.get("last_used_time", emoji_data.get("timestamp", time.time()))
                emoji.register_time = emoji_data.get("timestamp", time.time())
                emoji.description = emoji_data.get("description", "")
                emoji.emotion = emoji_data.get("emotion", [])  # 添加情感标签的加载
                emoji_objects.append(emoji)

            # 存储到EmojiManager中
            self.emoji_objects = emoji_objects

        except Exception as e:
            logger.error(f"[错误] 获取所有表情包对象失败: {str(e)}")

    async def get_emoji_from_db(self, hash=None):
        """获取所有表情包并初始化为MaiEmoji类对象

        参数:
            hash: 可选，如果提供则只返回指定哈希值的表情包

        返回:
            list[MaiEmoji]: 表情包对象列表
        """
        try:
            self._ensure_db()

            # 准备查询条件
            query = {}
            if hash:
                query = {"hash": hash}

            # 获取所有表情包
            all_emoji_data = list(db.emoji.find(query))

            # 将数据库记录转换为MaiEmoji对象
            emoji_objects = []
            for emoji_data in all_emoji_data:
                emoji = MaiEmoji(
                    filename=emoji_data.get("filename", ""),
                    path=emoji_data.get("path", ""),
                )

                # 设置额外属性
                emoji.usage_count = emoji_data.get("usage_count", 0)
                emoji.last_used_time = emoji_data.get("last_used_time", emoji_data.get("timestamp", time.time()))
                emoji.register_time = emoji_data.get("timestamp", time.time())
                emoji.description = emoji_data.get("description", "")
                emoji.emotion = emoji_data.get("emotion", [])  # 添加情感标签的加载

                emoji_objects.append(emoji)

            # 存储到EmojiManager中
            self.emoji_objects = emoji_objects

            return emoji_objects

        except Exception as e:
            logger.error(f"[错误] 获取所有表情包对象失败: {str(e)}")
            return []

    async def get_emoji_from_manager(self, hash) -> MaiEmoji:
        """从EmojiManager中获取表情包

        参数:
            hash:如果提供则只返回指定哈希值的表情包
        """
        for emoji in self.emoji_objects:
            if emoji.hash == hash:
                return emoji
        return None

    async def delete_emoji(self, emoji_hash: str) -> bool:
        """根据哈希值删除表情包

        Args:
            emoji_hash: 表情包的哈希值

        Returns:
            bool: 是否成功删除
        """
        try:
            self._ensure_db()

            # 从emoji_objects中查找表情包对象
            emoji = await self.get_emoji_from_manager(emoji_hash)

            if not emoji:
                logger.warning(f"[警告] 未找到哈希值为 {emoji_hash} 的表情包")
                return False

            # 使用MaiEmoji对象的delete方法删除表情包
            success = await emoji.delete()

            if success:
                # 从emoji_objects列表中移除该对象
                self.emoji_objects = [e for e in self.emoji_objects if e.hash != emoji_hash]
                # 更新计数
                self.emoji_num -= 1
                logger.info(f"[统计] 当前表情包数量: {self.emoji_num}")

                return True
            else:
                logger.error(f"[错误] 删除表情包失败: {emoji_hash}")
                return False

        except Exception as e:
            logger.error(f"[错误] 删除表情包失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _emoji_objects_to_readable_list(self, emoji_objects):
        """将表情包对象列表转换为可读的字符串列表

        参数:
            emoji_objects: MaiEmoji对象列表

        返回:
            list[str]: 可读的表情包信息字符串列表
        """
        emoji_info_list = []
        for i, emoji in enumerate(emoji_objects):
            # 转换时间戳为可读时间
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(emoji.register_time))
            # 构建每个表情包的信息字符串
            emoji_info = (
                f"编号: {i + 1}\n描述: {emoji.description}\n使用次数: {emoji.usage_count}\n添加时间: {time_str}\n"
            )
            emoji_info_list.append(emoji_info)
        return emoji_info_list

    async def replace_a_emoji(self, new_emoji: MaiEmoji):
        """替换一个表情包

        Args:
            new_emoji: 新表情包对象

        Returns:
            bool: 是否成功替换表情包
        """
        try:
            self._ensure_db()

            # 获取所有表情包对象
            emoji_objects = self.emoji_objects
            # 计算每个表情包的选择概率
            probabilities = [1 / (emoji.usage_count + 1) for emoji in emoji_objects]
            # 归一化概率，确保总和为1
            total_probability = sum(probabilities)
            normalized_probabilities = [p / total_probability for p in probabilities]

            # 使用概率分布选择最多20个表情包
            selected_emojis = random.choices(
                emoji_objects, weights=normalized_probabilities, k=min(MAX_EMOJI_FOR_PROMPT, len(emoji_objects))
            )

            # 将表情包信息转换为可读的字符串
            emoji_info_list = self._emoji_objects_to_readable_list(selected_emojis)

            # 构建提示词
            prompt = (
                f"{global_config.BOT_NICKNAME}的表情包存储已满({self.emoji_num}/{self.emoji_num_max})，"
                f"需要决定是否删除一个旧表情包来为新表情包腾出空间。\n\n"
                f"新表情包信息：\n"
                f"描述: {new_emoji.description}\n\n"
                f"现有表情包列表：\n" + "\n".join(emoji_info_list) + "\n\n"
                "请决定：\n"
                "1. 是否要删除某个现有表情包来为新表情包腾出空间？\n"
                "2. 如果要删除，应该删除哪一个(给出编号)？\n"
                "请只回答：'不删除'或'删除编号X'(X为表情包编号)。"
            )

            # 调用大模型进行决策
            decision, _ = await self.llm_emotion_judge.generate_response_async(prompt, temperature=0.8)
            logger.info(f"[决策] 结果: {decision}")

            # 解析决策结果
            if "不删除" in decision:
                logger.info("[决策] 不删除任何表情包")
                return False

            # 尝试从决策中提取表情包编号
            match = re.search(r"删除编号(\d+)", decision)
            if match:
                emoji_index = int(match.group(1)) - 1  # 转换为0-based索引

                # 检查索引是否有效
                if 0 <= emoji_index < len(selected_emojis):
                    emoji_to_delete = selected_emojis[emoji_index]

                    # 删除选定的表情包
                    logger.info(f"[决策] 删除表情包: {emoji_to_delete.description}")
                    delete_success = await self.delete_emoji(emoji_to_delete.hash)

                    if delete_success:
                        # 修复：等待异步注册完成
                        register_success = await new_emoji.register_to_db()
                        if register_success:
                            self.emoji_objects.append(new_emoji)
                            self.emoji_num += 1
                            logger.success(f"[成功] 注册: {new_emoji.filename}")
                            return True
                        else:
                            logger.error(f"[错误] 注册表情包到数据库失败: {new_emoji.filename}")
                            return False
                    else:
                        logger.error("[错误] 删除表情包失败，无法完成替换")
                        return False
                else:
                    logger.error(f"[错误] 无效的表情包编号: {emoji_index + 1}")
            else:
                logger.error(f"[错误] 无法从决策中提取表情包编号: {decision}")

            return False

        except Exception as e:
            logger.error(f"[错误] 替换表情包失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def build_emoji_description(self, image_base64: str) -> Tuple[str, list]:
        """获取表情包描述和情感列表

        Args:
            image_base64: 图片的base64编码

        Returns:
            Tuple[str, list]: 返回表情包描述和情感列表
        """
        try:
            # 解码图片并获取格式
            image_bytes = base64.b64decode(image_base64)
            image_format = Image.open(io.BytesIO(image_bytes)).format.lower()

            # 调用AI获取描述
            if image_format == "gif" or image_format == "GIF":
                image_base64 = image_manager.transform_gif(image_base64)
                prompt = "这是一个动态图表情包，每一张图代表了动态图的某一帧，黑色背景代表透明，描述一下表情包表达的情感和内容，描述细节，从互联网梗,meme的角度去分析"
                description, _ = await self.vlm.generate_response_for_image(prompt, image_base64, "jpg")
            else:
                prompt = "这是一个表情包，请详细描述一下表情包所表达的情感和内容，描述细节，从互联网梗,meme的角度去分析"
                description, _ = await self.vlm.generate_response_for_image(prompt, image_base64, image_format)

            # 审核表情包
            if global_config.EMOJI_CHECK:
                prompt = f'''
                    这是一个表情包，请对这个表情包进行审核，标准如下：
                    1. 必须符合"{global_config.EMOJI_CHECK_PROMPT}"的要求
                    2. 不能是色情、暴力、等违法违规内容，必须符合公序良俗
                    3. 不能是任何形式的截图，聊天记录或视频截图
                    4. 不要出现5个以上文字
                    请回答这个表情包是否满足上述要求，是则回答是，否则回答否，不要出现任何其他内容
                '''
                content, _ = await self.vlm.generate_response_for_image(prompt, image_base64, image_format)
                if content == "否":
                    return None, []

            # 分析情感含义
            emotion_prompt = f"""
            请你识别这个表情包的含义和适用场景，给我简短的描述，每个描述不要超过15个字
            这是一个基于这个表情包的描述：'{description}'
            你可以关注其幽默和讽刺意味，动用贴吧，微博，小红书的知识，必须从互联网梗,meme的角度去分析
            请直接输出描述，不要出现任何其他内容，如果有多个描述，可以用逗号分隔
            """
            emotions_text, _ = await self.llm_emotion_judge.generate_response_async(emotion_prompt, temperature=0.7)

            # 处理情感列表
            emotions = [e.strip() for e in emotions_text.split(",") if e.strip()]

            # 根据情感标签数量随机选择喵~超过5个选3个，超过2个选2个
            if len(emotions) > 5:
                emotions = random.sample(emotions, 3)
            elif len(emotions) > 2:
                emotions = random.sample(emotions, 2)

            return f"[表情包：{description}]", emotions

        except Exception as e:
            logger.error(f"获取表情包描述失败: {str(e)}")
            return "", []

    async def register_emoji_by_filename(self, filename: str) -> bool:
        """读取指定文件名的表情包图片，分析并注册到数据库

        Args:
            filename: 表情包文件名，必须位于EMOJI_DIR目录下

        Returns:
            bool: 注册是否成功
        """
        try:
            # 使用MaiEmoji类创建表情包实例
            new_emoji = MaiEmoji(filename, EMOJI_DIR)
            await new_emoji.initialize_hash_format()
            emoji_base64 = image_path_to_base64(os.path.join(EMOJI_DIR, filename))
            description, emotions = await self.build_emoji_description(emoji_base64)
            if description == "" or description == None:
                return False
            new_emoji.description = description
            new_emoji.emotion = emotions

            # 检查是否已经注册过
            # 对比内存中是否存在相同哈希值的表情包
            if await self.get_emoji_from_manager(new_emoji.hash):
                logger.warning(f"[警告] 表情包已存在: {filename}")
                return False

            if self.emoji_num >= self.emoji_num_max:
                logger.warning(f"表情包数量已达到上限({self.emoji_num}/{self.emoji_num_max})")
                replaced = await self.replace_a_emoji(new_emoji)
                if not replaced:
                    logger.error("[错误] 替换表情包失败，无法完成注册")
                    return False
                return True
            else:
                # 修复：等待异步注册完成
                register_success = await new_emoji.register_to_db()
                if register_success:
                    self.emoji_objects.append(new_emoji)
                    self.emoji_num += 1
                    logger.success(f"[成功] 注册: {filename}")
                    return True
                else:
                    logger.error(f"[错误] 注册表情包到数据库失败: {filename}")
                    return False

        except Exception as e:
            logger.error(f"[错误] 注册表情包失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def clear_temp_emoji(self):
        """每天清理临时表情包
        清理/data/emoji和/data/image目录下的所有文件
        当目录中文件数超过50时，会全部删除
        """

        logger.info("[清理] 开始清理缓存...")

        # 清理emoji目录
        emoji_dir = os.path.join(BASE_DIR, "emoji")
        if os.path.exists(emoji_dir):
            files = os.listdir(emoji_dir)
            # 如果文件数超过50就全部删除
            if len(files) > 50:
                for filename in files:
                    file_path = os.path.join(emoji_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.debug(f"[清理] 删除: {filename}")

        # 清理image目录
        image_dir = os.path.join(BASE_DIR, "image")
        if os.path.exists(image_dir):
            files = os.listdir(image_dir)
            # 如果文件数超过50就全部删除
            if len(files) > 50:
                for filename in files:
                    file_path = os.path.join(image_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.debug(f"[清理] 删除图片: {filename}")

        logger.success("[清理] 完成")

    async def clean_unused_emojis(self, emoji_dir, emoji_objects):
        """清理未使用的表情包文件
        遍历指定文件夹中的所有文件，删除未在emoji_objects列表中的文件
        """
        # 首先检查目录是否存在喵~
        if not os.path.exists(emoji_dir):
            logger.warning(f"[清理] 表情包目录不存在，跳过清理: {emoji_dir}")
            return

        # 获取所有表情包路径
        emoji_paths = {emoji.path for emoji in emoji_objects}

        # 遍历文件夹中的所有文件
        for file_name in os.listdir(emoji_dir):
            file_path = os.path.join(emoji_dir, file_name)

            # 检查文件是否在表情包路径列表中
            if file_path not in emoji_paths:
                try:
                    # 删除未在表情包列表中的文件
                    os.remove(file_path)
                    logger.info(f"[清理] 删除未使用的表情包文件: {file_path}")
                except Exception as e:
                    logger.error(f"[错误] 删除文件时出错: {str(e)}")


# 创建全局单例
emoji_manager = EmojiManager()
