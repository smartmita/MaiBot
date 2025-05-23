import asyncio
import threading
from typing import Dict, Optional, List, Any, Tuple

from src.common.logger_manager import get_logger
from src.config.config import global_config
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.person_info.person_info import person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager
from .sobriquet.sobriquet_manager import sobriquet_manager # 导入新的 SobriquetManager
from .profile_utils import format_profile_prompt_injection # 导入新的格式化函数

logger = get_logger("ProfileManager")


class ProfileManager:
    """
    管理用户画像信息获取和整合的单例类。
    协调从各个来源（如 SobriquetManager, PersonInfoManager, RelationshipManager）获取数据，
    并使用 ProfileUtils 中的工具函数格式化为 Prompt 注入内容。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logger.info("正在创建 ProfileManager 单例实例...")
                    cls._instance = super(ProfileManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return
            logger.info("正在初始化 ProfileManager 组件...")
            # ProfileManager 本身可以有一个总开关，或者依赖于子模块的开关
            self.is_enabled = global_config.profile.profile_system_enabled

            # 这里可以初始化其他依赖，如果 ProfileManager 需要管理更多类型的画像信息
            # sobriquet_manager 已经是单例，在导入时创建/获取

            self._initialized = True
            logger.info("ProfileManager 初始化完成。")

    async def get_profile_prompt_injection(self, chat_stream: ChatStream, message_list_before_now: List[Dict]) -> str:
        """
        获取并格式化用于 Prompt 注入的整合用户画像信息字符串。
        包含 UID、用户实际昵称、性别、群名片、群头衔（仅群聊），以及在群聊中可选的群内常用绰号。
        """
        if not self.is_enabled:
            logger.info("ProfileManager 已禁用，不提供画像信息注入。")
            return ""
        if not chat_stream:
            logger.warning("无效的 chat_stream，无法获取用户画像信息注入。")
            return ""

        log_prefix = f"[{chat_stream.stream_id}]"
        try:
            platform = chat_stream.platform
            is_group = bool(chat_stream.group_info and chat_stream.group_info.group_id)

            user_ids_in_context_str = {
                str(msg["user_info"]["user_id"])
                for msg in message_list_before_now
                if msg.get("user_info", {}).get("user_id")
            }

            if not user_ids_in_context_str:
                logger.debug(f"{log_prefix} 未找到上下文用户用于画像信息注入。")
                return ""

            unique_user_ids_list_str = sorted(list(user_ids_in_context_str))

            # 1. 获取用户实际平台昵称 (actual_name)
            actual_names_map: Dict[str, str] = {}
            if unique_user_ids_list_str:
                try:
                    actual_names_map = await relationship_manager.get_actual_nicknames_batch(platform, unique_user_ids_list_str)
                except Exception as e:
                    logger.error(f"{log_prefix} 批量获取用户实际昵称时出错: {e}", exc_info=True)

            # 2. 构建用户基础数据列表，并获取群名片、群头衔 (如果适用) 和性别
            users_profile_data: List[Dict[str, Any]] = []
            latest_user_info_map: Dict[str, Dict[str, Any]] = {}
            if is_group:
                for msg_dict in reversed(message_list_before_now):
                    user_info = msg_dict.get("user_info")
                    if user_info and isinstance(user_info, dict):
                        msg_user_id_str = str(user_info.get("user_id"))
                        if msg_user_id_str in unique_user_ids_list_str and msg_user_id_str not in latest_user_info_map:
                            latest_user_info_map[msg_user_id_str] = user_info
                        if len(latest_user_info_map) == len(unique_user_ids_list_str):
                            break
            
            for user_id_str in unique_user_ids_list_str:
                user_data_entry: Dict[str, Any] = {"user_id": user_id_str}
                
                # 设置实际昵称 (actual_name)
                actual_name = actual_names_map.get(user_id_str)
                if actual_name is not None:
                    user_data_entry["actual_name"] = actual_name
                elif user_id_str == global_config.bot.qq_account: # bot.qq_account 配置名不变
                    user_data_entry["actual_name"] = global_config.bot.nickname # bot.nickname 配置名不变
                else:
                    fallback_name = next(
                        (
                            m["user_info"].get("user_nickname")
                            for m in reversed(message_list_before_now)
                            if str(m["user_info"].get("user_id")) == user_id_str and m["user_info"].get("user_nickname")
                        ),
                        None
                    )
                    if fallback_name:
                        user_data_entry["actual_name"] = fallback_name
                    else:
                        logger.warning(f"{log_prefix} 未能获取 user_id '{user_id_str}' 的实际平台昵称，也未在上下文中找到备用昵称。")
                        user_data_entry["actual_name"] = f"用户{user_id_str[:4]}" # 提供一个默认值

                # 获取性别标记 (gender_mark)
                gender_mark_value = "未知"
                try:
                    user_id_for_gender_lookup = int(user_id_str)
                    person_id = person_info_manager.get_person_id(platform, user_id_for_gender_lookup)
                    if person_id:
                        # "gender_mark" 是 person_info 中的字段键名，保持不变
                        fetched_gender = await person_info_manager.get_value(person_id, "gender_mark")
                        if fetched_gender is not None:
                            gender_mark_value = fetched_gender
                except ValueError:
                    logger.warning(f"{log_prefix} 用户ID '{user_id_str}' 无法转为整数，无法获取其 person_id 以查询性别。")
                except Exception as e_gender:
                    logger.error(f"{log_prefix} 获取用户 {user_id_str} 性别标记时出错: {e_gender}", exc_info=True)
                user_data_entry["gender_mark"] = gender_mark_value

                # 获取群名片和群头衔
                if is_group:
                    user_specific_info = latest_user_info_map.get(user_id_str)
                    if user_specific_info:
                        user_data_entry["group_cardname"] = user_specific_info.get("user_cardname")
                        user_data_entry["group_titlename"] = user_specific_info.get("user_titlename")
                
                users_profile_data.append(user_data_entry)

            if not users_profile_data:
                logger.debug(f"{log_prefix} 未能构建有效的用户画像基础数据列表。")
                return ""

            # 3. 获取群组常用绰号 (如果适用)
            selected_group_sobriquets: Optional[List[Tuple[str, str, str, int]]] = None
            if is_group:
                group_id_str = str(chat_stream.group_info.group_id)
                # 调用 SobriquetManager 获取已选择的绰号
                # sobriquet_manager.is_analysis_enabled 和 max_nicknames_in_prompt 的检查会在 get_selected_sobriquets_for_group 内部进行
                selected_group_sobriquets = await sobriquet_manager.get_selected_sobriquets_for_group(
                    platform, unique_user_ids_list_str, group_id_str
                )

            # 4. 格式化所有信息为 Prompt 注入字符串
            injection_str = format_profile_prompt_injection(
                users_profile_data,
                selected_group_sobriquets,
                is_group
            )

            if injection_str:
                logger.debug(f"{log_prefix} 生成的用户画像 Prompt 注入:\n{injection_str.strip()}")
            return injection_str

        except Exception as e:
            logger.error(f"{log_prefix} 获取用户画像信息注入时发生严重错误: {e}", exc_info=True)
            return ""


# 单例实例
profile_manager = ProfileManager()