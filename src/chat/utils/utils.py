import random
import re
import regex
import time
from collections import Counter

import jieba
import numpy as np
from maim_message import UserInfo
from pymongo.errors import PyMongoError

from src.common.logger import get_module_logger
from src.manager.mood_manager import mood_manager
from ..message_receive.message import MessageRecv
from ..models.utils_model import LLMRequest
from .typo_generator import ChineseTypoGenerator
from ...common.database import db
from ...config.config import global_config

logger = get_module_logger("chat_utils")

# --- 全局常量和预编译正则表达式 ---
# \p{L} 匹配任何语言中的任何种类的字母字符。
_L_REGEX = regex.compile(r"\p{L}")
# \p{Han} 匹配汉字。
_HAN_CHAR_REGEX = regex.compile(r"\p{Han}")
# \p{Nd} 匹配十进制数字字符。
_Nd_REGEX = regex.compile(r"\p{Nd}")

# 书名号占位符的前缀，用于在处理文本时临时替换书名号。
BOOK_TITLE_PLACEHOLDER_PREFIX = "__BOOKTITLE_"
# 定义句子分隔符集合。
SEPARATORS = {"。", "，", ",", " ", ";", "\xa0", "\n", ".", "—", "！", "？"}
# 已知的以点号结尾的英文缩写词，用于避免错误地将缩写词中的点号作为句子结束符。
KNOWN_ABBREVIATIONS_ENDING_WITH_DOT = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Messrs.", "Mmes.", "Capt.", "Gov.",
    "Inc.", "Ltd.", "Corp.", "Co.", "PLC", "vs.", "etc.", "i.e.", "e.g.", "viz.",
    "al.", "et al.", "ca.", "cf.", "No.", "Vol.", "pp.", "fig.", "figs.", "ed.",
    "Ph.D.", "M.D.", "B.A.", "M.A.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.",
    "Aug.", "Sep.", "Oct.", "Nov.", "Dec.", "Mon.", "Tue.", "Wed.", "Thu.", "Fri.",
    "Sat.", "Sun.", "U.S.", "U.K.", "E.U.", "U.S.A.", "U.S.S.R.", "Ave.", "Blvd.",
    "Rd.", "Ln.", "approx.", "dept.", "appt.", "श्री.", # 印地语中的 Shri.
}

# --- 辅助函数 ---

def is_letter_not_han(char_str: str) -> bool:
    """
    检查单个字符是否为“字母”且“非汉字”。
    例如拉丁字母、西里尔字母、韩文等返回True。
    汉字、数字、标点、空格等返回False。

    Args:
        char_str:待检查的单个字符。

    Returns:
        bool: 如果字符是字母且非汉字则为True，否则为False。
    """
    if not isinstance(char_str, str) or len(char_str) != 1:
        return False # 输入必须是单个字符的字符串
    is_letter = _L_REGEX.fullmatch(char_str) is not None
    if not is_letter:
        return False # 如果不是字母，直接返回False
    # 是字母，则进一步判断是否为汉字
    is_han = _HAN_CHAR_REGEX.fullmatch(char_str) is not None
    return not is_han # 是字母且不是汉字


def is_han_character(char_str: str) -> bool:
    """
    检查单个字符是否为汉字 (使用 Unicode \p{Han} 属性)。

    Args:
        char_str: 待检查的单个字符。

    Returns:
        bool: 如果字符是汉字则为True，否则为False。
    """
    if not isinstance(char_str, str) or len(char_str) != 1:
        return False
    return _HAN_CHAR_REGEX.fullmatch(char_str) is not None


def is_digit(char_str: str) -> bool:
    """
    检查单个字符是否为Unicode数字 (十进制数字)。

    Args:
        char_str: 待检查的单个字符。

    Returns:
        bool: 如果字符是Unicode数字则为True，否则为False。
    """
    if not isinstance(char_str, str) or len(char_str) != 1:
        return False
    return _Nd_REGEX.fullmatch(char_str) is not None


def is_relevant_word_char(char_str: str) -> bool:
    """
    检查字符是否为“相关词语字符”（即非汉字字母或数字）。
    此函数用于判断在非中文语境下，空格两侧的字符是否应被视为构成一个连续词语的部分，
    从而决定该空格是否作为分割点。
    例如拉丁字母、西里尔字母、数字等返回True。
    汉字、标点、纯空格等返回False。

    Args:
        char_str: 待检查的单个字符。

    Returns:
        bool: 如果字符是非汉字字母或数字则为True，否则为False。
    """
    if not isinstance(char_str, str) or len(char_str) != 1:
        return False
    # 检查是否为Unicode字母
    if _L_REGEX.fullmatch(char_str):
        # 如果是字母，则检查是否非汉字
        return not _HAN_CHAR_REGEX.fullmatch(char_str)
    # 检查是否为Unicode数字
    if _Nd_REGEX.fullmatch(char_str):
        return True  # 数字本身被视为相关词语字符
    return False


def is_english_letter(char: str) -> bool:
    """
    检查单个字符是否为英文字母（忽略大小写）。

    Args:
        char: 待检查的单个字符。

    Returns:
        bool: 如果字符是英文字母则为True，否则为False。
    """
    return "a" <= char.lower() <= "z"


def protect_book_titles(text: str) -> tuple[str, dict[str, str]]:
    """
    保护文本中的书名号内容，将其替换为唯一的占位符。
    返回保护后的文本和占位符到原始内容的映射。

    Args:
        text: 原始输入文本。

    Returns:
        tuple[str, dict[str, str]]: 一个元组，包含：
            - protected_text (str): 书名号被占位符替换后的文本。
            - book_title_mapping (dict): 占位符到原始书名号内容（含书名号本身）的映射。
    """
    book_title_mapping = {}
    # 正则表达式匹配《内容》形式的书名号，使用非贪婪匹配 (.*?) 以正确处理。
    book_title_pattern = re.compile(r"《(.*?)》")

    def replace_func(match):
        # 为每个匹配到的书名号生成一个唯一的占位符。
        placeholder = f"{BOOK_TITLE_PLACEHOLDER_PREFIX}{len(book_title_mapping)}__"
        # 存储占位符和原始书名号（包括《》）的映射关系。
        book_title_mapping[placeholder] = match.group(0)
        return placeholder

    protected_text = book_title_pattern.sub(replace_func, text)
    return protected_text, book_title_mapping

def recover_book_titles(sentences: list[str], book_title_mapping: dict[str, str]) -> list[str]:
    """
    将句子列表中的书名号占位符恢复为原始的书名号内容。

    Args:
        sentences: 包含可能书名号占位符的句子列表。
        book_title_mapping: 占位符到原始书名号内容的映射。

    Returns:
        list[str]: 书名号占位符被恢复后的句子列表。
    """
    recovered_sentences = []
    if not sentences: # 如果输入句子列表为空，直接返回空列表
        return []
    for sentence in sentences:
        if not isinstance(sentence, str): # 添加类型检查，确保每个元素都是字符串
            recovered_sentences.append(sentence) # 如果不是字符串，直接添加（或选择跳过/记录错误）
            continue
        # 遍历映射，将句子中的每个占位符替换回其原始书名号内容。
        for placeholder, original_content in book_title_mapping.items():
            sentence = sentence.replace(placeholder, original_content)
        recovered_sentences.append(sentence)
    return recovered_sentences


def db_message_to_str(message_dict: dict) -> str:
    logger.debug(f"message_dict: {message_dict}")
    time_str = time.strftime("%m-%d %H:%M:%S", time.localtime(message_dict["time"]))
    try:
        name = "[(%s)%s]%s" % (
            message_dict["user_id"],
            message_dict.get("user_nickname", ""),
            message_dict.get("user_cardname", ""),
        )
    except Exception:
        name = message_dict.get("user_nickname", "") or f"用户{message_dict['user_id']}"
    content = message_dict.get("processed_plain_text", "")
    result = f"[{time_str}] {name}: {content}\n"
    logger.debug(f"result: {result}")
    return result


def is_mentioned_bot_in_message(message: MessageRecv) -> tuple[bool, float]:
    """检查消息是否提到了机器人"""
    keywords = [global_config.bot.nickname]
    nicknames = global_config.bot.alias_names
    reply_probability = 0.0
    is_at = False
    is_mentioned = False

    if (
        message.message_info.additional_config is not None
        and message.message_info.additional_config.get("is_mentioned") is not None
    ):
        try:
            reply_probability = float(message.message_info.additional_config.get("is_mentioned"))
            is_mentioned = True
            return is_mentioned, reply_probability
        except Exception as e:
            logger.warning(e)
            logger.warning(
                f"消息中包含不合理的设置 is_mentioned: {message.message_info.additional_config.get('is_mentioned')}"
            )

    # 判断是否被@
    if re.search(f"@[\\s\\S]*?（id:{global_config.bot.qq_account}）", message.processed_plain_text):
        is_at = True
        is_mentioned = True

    if is_at and global_config.normal_chat.at_bot_inevitable_reply:
        reply_probability = 1.0
        logger.info("被@，回复概率设置为100%")
    else:
        if not is_mentioned:
            # 判断是否被回复
            if re.match(
                f"\\[回复 [\\s\\S]*?\\({str(global_config.bot.qq_account)}\\)：[\\s\\S]*?\\]，说：",
                message.processed_plain_text,
            ):
                is_mentioned = True
            else:
                # 判断内容中是否被提及
                message_content = re.sub(r"@[\s\S]*?（(\d+)）", "", message.processed_plain_text)
                message_content = re.sub(r"\[回复 [\s\S]*?\(((\d+)|未知id)\)：[\s\S]*?]，说：", "", message_content)
                for keyword in keywords:
                    if keyword in message_content:
                        is_mentioned = True
                for nickname in nicknames:
                    if nickname in message_content:
                        is_mentioned = True
        if is_mentioned and global_config.normal_chat.mentioned_bot_inevitable_reply:
            reply_probability = 1.0
            logger.info("被提及，回复概率设置为100%")
    return is_mentioned, reply_probability


async def get_embedding(text, request_type="embedding"):
    """获取文本的embedding向量"""
    # TODO: API-Adapter修改标记
    llm = LLMRequest(model=global_config.model.embedding, request_type=request_type)
    # return llm.get_embedding_sync(text)
    try:
        embedding = await llm.get_embedding(text)
    except Exception as e:
        logger.error(f"获取embedding失败: {str(e)}")
        embedding = None
    return embedding


def get_recent_group_detailed_plain_text(chat_stream_id: str, limit: int = 12, combine=False):
    recent_messages = list(
        db.messages.find(
            {"chat_id": chat_stream_id},
            {
                "time": 1,  # 返回时间字段
                "chat_id": 1,
                "chat_info": 1,
                "user_info": 1,
                "message_id": 1,  # 返回消息ID字段
                "detailed_plain_text": 1,  # 返回处理后的文本字段
            },
        )
        .sort("time", -1)
        .limit(limit)
    )

    if not recent_messages:
        return []

    message_detailed_plain_text = ""
    message_detailed_plain_text_list = []

    # 反转消息列表，使最新的消息在最后
    recent_messages.reverse()

    if combine:
        for msg_db_data in recent_messages:
            message_detailed_plain_text += str(msg_db_data["detailed_plain_text"])
        return message_detailed_plain_text
    else:
        for msg_db_data in recent_messages:
            message_detailed_plain_text_list.append(msg_db_data["detailed_plain_text"])
        return message_detailed_plain_text_list


def get_recent_group_speaker(chat_stream_id: int, sender, limit: int = 12) -> list:
    # 获取当前群聊记录内发言的人
    recent_messages = list(
        db.messages.find(
            {"chat_id": chat_stream_id},
            {
                "user_info": 1,
            },
        )
        .sort("time", -1)
        .limit(limit)
    )

    if not recent_messages:
        return []

    who_chat_in_group = []
    for msg_db_data in recent_messages:
        user_info = UserInfo.from_dict(msg_db_data["user_info"])
        if (
            (user_info.platform, user_info.user_id) != sender
            and user_info.user_id != global_config.bot.qq_account
            and (user_info.platform, user_info.user_id, user_info.user_nickname) not in who_chat_in_group
            and len(who_chat_in_group) < 5
        ):  # 排除重复，排除消息发送者，排除bot，限制加载的关系数目
            who_chat_in_group.append((user_info.platform, user_info.user_id, user_info.user_nickname))

    return who_chat_in_group


def split_into_sentences_w_remove_punctuation(original_text: str) -> list[str]:
    """
    将输入文本分割成句子列表。
    此过程包括：
    1. 保护书名号。
    2. 文本预处理（如处理换行符）。
    3. 基于分隔符将文本切分为初步的段落(segments)。
    4. 根据段落内容和分隔符类型，构建初步的句子列表(preliminary_final_sentences)，
       特别处理汉字间的空格作为分割点。
    5. 对初步句子列表进行可能的合并（基于随机概率和文本长度）。
    6. 对合并后的句子进行随机标点移除。
    7. 恢复书名号。
    8. 返回最终处理过的句子列表。

    Args:
        original_text: 原始输入文本。

    Returns:
        list[str]: 分割和处理后的句子列表。
    """
    # 步骤1: 保护书名号，将其替换为占位符，并获取映射关系。
    text, local_book_title_mapping = protect_book_titles(original_text)
    perform_book_title_recovery_here = True # 控制是否在本函数末尾执行恢复，主要用于调试

    # 步骤2: 文本预处理
    text = regex.sub(r"\n\s*\n+", "\n", text)  # 合并多个连续的换行符（及其间的空格）为一个换行符。
    text = regex.sub(r"\n\s*([—。.,，;\s\xa0！？])", r"\1", text) # 移除分隔符前的换行符和空格。
    text = regex.sub(r"([—。.,，;\s\xa0！？])\s*\n", r"\1", text) # 移除分隔符后的换行符和空格。

    def replace_han_newline(match):
        """辅助函数，用于将汉字之间的单个换行符替换为逗号。"""
        char1 = match.group(1)
        char2 = match.group(2)
        if is_han_character(char1) and is_han_character(char2):
            return char1 + "，" + char2  # 汉字间的换行符替换为逗号
        return match.group(0) # 其他情况保持不变

    text = regex.sub(r"(.)\n(.)", replace_han_newline, text) # 应用上述替换规则

    len_text = len(text) # 使用保护书名号后的文本长度进行后续判断

    # 特殊情况处理：如果原始文本（保护后）本身就是一个书名号占位符，
    # 后续逻辑可能会将其作为单个元素处理，这里先标记，确保它能被正确恢复。
    # (此处的 'pass' 意味着具体处理逻辑在后续的 restructured section 中统一进行)
    if local_book_title_mapping and text in local_book_title_mapping: # 注意：这里应该是 text in local_book_title_mapping.keys()
        pass


    # 对于非常短且不含书名号的文本的提前返回逻辑。
    if len_text < 3 and not local_book_title_mapping:
        stripped_text = text.strip()
        if not stripped_text: # 如果剥离空格后为空，返回空列表
            return []
        # 如果剥离后只有一个字符且该字符是分隔符，也视为空（或无效）输入
        if len(stripped_text) == 1 and stripped_text in SEPARATORS:
            return []
        # 对于极短文本，不应用随机标点移除，直接返回其剥离空格后的内容
        return [stripped_text]


    # 步骤3: 基于分隔符将文本切分为初步的段落(segments)
    # segments 列表中的每个元素是一个元组 (content, separator_char)
    segments = []
    current_segment = "" # 当前正在构建的段落内容
    i = 0
    while i < len(text):
        char = text[i] # 当前字符
        if char in SEPARATORS: # 如果当前字符是分隔符
            can_split_current_char = True # 默认情况下，当前分隔符可以用于分割

            # 特殊分隔符处理逻辑
            if char == ".": # 处理点号 '.'
                can_split_this_dot = True # 默认点号可以分割
                # 规则1: 小数点 (数字.数字) - 不分割
                if 0 < i < len_text - 1 and is_digit(text[i - 1]) and is_digit(text[i + 1]):
                    can_split_this_dot = False
                # 规则2: 西文缩写/域名内部的点 (非汉字字母.非汉字字母) - 不分割
                elif 0 < i < len_text - 1 and is_letter_not_han(text[i - 1]) and is_letter_not_han(text[i + 1]):
                    can_split_this_dot = False
                # 规则3: 已知缩写词的末尾点 (例如 "e.g. ", "U.S.A. ") - 不分割
                else:
                    potential_abbreviation_word = current_segment + char # 构造包含当前点号的潜在词语
                    # 检查是否是已知缩写词，并且其后是空格或文本末尾
                    is_followed_by_space = i + 1 < len_text and text[i + 1] == " "
                    is_at_end_of_text = i + 1 == len_text
                    if potential_abbreviation_word in KNOWN_ABBREVIATIONS_ENDING_WITH_DOT and \
                        (is_followed_by_space or is_at_end_of_text):
                        can_split_this_dot = False
                can_split_current_char = can_split_this_dot
            elif char == " " or char == "\xa0":  # 处理空格或NBSP (非断行空格)
                # 规则：非中文单词内部的空格不分割 (例如 "hello world", "слово1 слово2")
                if 0 < i < len_text - 1: # 确保空格前后都有字符
                    prev_char = text[i - 1]
                    next_char = text[i + 1]
                    # 如果空格前后都是“相关词语字符”（非汉字字母或数字），则不分割
                    if is_relevant_word_char(prev_char) and is_relevant_word_char(next_char):
                        can_split_current_char = False
            # 特殊分隔符处理逻辑结束

            if can_split_current_char: # 如果决定在此处分割
                if current_segment: # 如果当前段落有内容，则将其与分隔符一起存入segments
                    segments.append((current_segment, char))
                # 如果当前段落为空，但分隔符不是简单的排版空格 (除非是换行符这种有意义的空行分隔)
                # 这用于处理连续分隔符或以分隔符开头的情况
                elif char not in [" ", "\xa0"] or char == "\n":
                    segments.append(("", char)) # 添加空内容和该分隔符
                current_segment = ""  # 重置当前段落内容
            else: # 如果不分割，则将当前分隔符加入到当前段落内容中
                current_segment += char
        else: # 如果当前字符不是分隔符，则加入当前段落内容
            current_segment += char
        i += 1

    if current_segment:  # 处理文本末尾剩余的段落内容（它没有后续分隔符）
        segments.append((current_segment, "")) # 使用空字符串作为其分隔符标记

    # 步骤3.1: 过滤segments列表
    # 移除仅由空格组成的segment内容，但保留其后的有效分隔符（如换行符）。
    filtered_segments = []
    for content, sep in segments:
        stripped_content = content.strip() # 移除内容两端的空白
        if stripped_content: # 如果剥离后仍有内容，则保留
            filtered_segments.append((stripped_content, sep))
        # 如果内容为空，但分隔符本身有意义（不是普通空格，或者是换行符）
        elif sep and (sep not in [" ", "\xa0"] or sep == "\n"):
            filtered_segments.append(("", sep)) # 保留空内容和该有意义的分隔符
    segments = filtered_segments # 更新segments为过滤后的列表

    # 步骤4: 构建初步的句子列表 (preliminary_final_sentences)
    # 此阶段基于segments中的内容和分隔符类型，尝试组装成句子。
    # 关键逻辑：识别强终止符，并特别处理汉字间的空格作为分割点。
    preliminary_final_sentences = []
    current_sentence_build = "" # 当前正在构建的句子
    num_segments = len(segments)
    for k, (content, sep) in enumerate(segments): # 遍历每个 (内容, 分隔符) 对
        current_sentence_build += content  # 首先将段落内容加入当前句子构建

        # 判断分隔符的类型
        is_strong_terminator = sep in {"。", ".", "！", "？", "\n", "—"} # 是否为强句子终止符
        is_space_separator = sep in [" ", "\xa0"] # 是否为空格类分隔符

        append_sep_to_current = is_strong_terminator # 默认只有强终止符会附加到句子末尾
        should_split_now = False # 标记是否应在当前分隔符处立即分割句子

        if is_strong_terminator: # 如果是强终止符，则应立即分割
            should_split_now = True
        elif is_space_separator:  # 如果分隔符是空格
            # 检查是否为“汉字-空格-汉字”模式，若是，则也应分割
            if current_sentence_build: # 确保当前构建的句子有内容
                last_char_of_build_stripped = current_sentence_build.strip() # 获取去除尾部空格的句子内容
                # 检查当前句子末尾字符是否为汉字
                if last_char_of_build_stripped and is_han_character(last_char_of_build_stripped[-1]):
                    # 检查下一个segment (如果存在) 的内容的第一个字符是否是汉字
                    if k + 1 < num_segments:
                        next_content_tuple = segments[k+1]
                        if next_content_tuple: # 确保元组存在
                            next_content = next_content_tuple[0] # 获取下一个段落的内容
                            if next_content and is_han_character(next_content[0]):
                                should_split_now = True # 满足汉字-空格-汉字，应分割
                                append_sep_to_current = False  # 此时，该空格作为分割符，不应附加到句子末尾

            if not should_split_now: # 如果不是因汉字间空格而分割（即普通空格连接）
                # 避免在句子开头或已存在尾部空格时重复添加空格
                if current_sentence_build and not current_sentence_build.endswith(" ") and not current_sentence_build.endswith("\xa0"):
                    current_sentence_build += " "  # 将此空格作为连接符加入（统一用普通空格）
                append_sep_to_current = False # 该空格已作为连接符处理，不作为独立分隔符附加

        if should_split_now: # 如果决定在当前位置分割句子
            if append_sep_to_current and sep: # 如果需要附加分隔符（通常是强终止符）
                current_sentence_build += sep

            stripped_sentence = current_sentence_build.strip() # 清理句子两端空格
            if stripped_sentence: # 确保句子不为空
                preliminary_final_sentences.append(stripped_sentence)
            current_sentence_build = "" # 重置句子构建器
        elif sep and not is_space_separator:  # 如果是其他弱分隔符 (如 ',', ';')
            current_sentence_build += sep # 将其加入当前句子
            # 如果这是最后一个segment，且当前构建的句子有内容，则也视为一个完整句子
            if k == num_segments - 1 and current_sentence_build.strip():
                preliminary_final_sentences.append(current_sentence_build.strip())
                current_sentence_build = ""
        # 如果 sep 是空字符串 (通常是最后一个 segment 的情况)，则 current_sentence_build 已有内容，
        # 等待循环结束后的统一处理。

    if current_sentence_build.strip(): # 处理循环结束后剩余的正在构建的句子
        preliminary_final_sentences.append(current_sentence_build.strip())

    # 再次清理，确保列表中的句子都是非空字符串
    preliminary_final_sentences = [s for s in preliminary_final_sentences if s.strip()]

    # --- RESTRUCTURED SECTION FOR MERGING, PUNCTUATION REMOVAL, AND BOOK TITLE RECOVERY ---
    # 此部分统一处理句子的合并、随机标点移除和最终的书名号恢复。
    intermediate_sentences_placeholders = [] # 存储待恢复书名号的中间句子列表

    if not preliminary_final_sentences:
        # 情况1: 初步句子列表为空
        # 这可能发生在原始文本非常短、仅包含分隔符，或者仅包含一个书名号（已被替换为占位符）
        # 如果原文是单个书名号占位符，则应保留它以供恢复。
        # 注意: text 是经过 protect_book_titles 处理后的文本。
        # local_book_title_mapping.keys() 包含所有占位符。
        if local_book_title_mapping and text in local_book_title_mapping.keys():
            intermediate_sentences_placeholders = [text] # 此时 text 就是那个占位符
        # else (其他导致 preliminary_final_sentences 为空的情况)，intermediate_sentences_placeholders 保持为空列表

    elif len(preliminary_final_sentences) == 1:
        # 情况2: 初步句子列表只有一个句子
        s = preliminary_final_sentences[0].strip() # 获取该句子并去除两端空格
        if s:
            s = random_remove_punctuation(s) # 对该句子进行随机标点移除
        intermediate_sentences_placeholders = [s] if s else [] # 如果处理后仍有内容，则加入列表

    else: # 情况3: 初步句子列表有多个句子，需要进行可能的随机合并
        final_sentences_merged = [] # 存储合并后的句子

        # 根据原始文本长度（未保护书名号前）决定合并强度
        original_len_for_strength = len(original_text)
        split_strength = 0.5 # 默认分割强度
        if original_len_for_strength < 12:
            split_strength = 0.5
        elif original_len_for_strength < 32:
            split_strength = 0.7
        else:
            split_strength = 0.9
        actual_merge_probability = 1.0 - split_strength # 合并概率与分割强度互补

        temp_sentence = "" # 临时存储正在合并的句子
        if preliminary_final_sentences: # 确保有句子可以合并
            temp_sentence = preliminary_final_sentences[0] # 从第一个句子开始
            for i_merge in range(1, len(preliminary_final_sentences)): # 遍历后续句子
                current_sentence_to_merge = preliminary_final_sentences[i_merge]
                should_merge_based_on_punctuation = True # 默认可以合并
                # 如果前一个句子以强终止符结尾，则不应与后一个句子合并
                if temp_sentence and \
                    (temp_sentence.endswith("。") or temp_sentence.endswith(".") or \
                    temp_sentence.endswith("!") or temp_sentence.endswith("?") or \
                    temp_sentence.endswith("—")):
                    should_merge_based_on_punctuation = False

                # 根据合并概率和标点规则决定是否合并
                if random.random() < actual_merge_probability and temp_sentence and should_merge_based_on_punctuation:
                    # 合并时，如果需要，在两句子间添加空格
                    if not temp_sentence.endswith(" ") and not current_sentence_to_merge.startswith(" "):
                        temp_sentence += " " 
                    temp_sentence += current_sentence_to_merge
                else: # 不合并，则将已构建的 temp_sentence 加入列表，并开始新的 temp_sentence
                    if temp_sentence:
                        final_sentences_merged.append(temp_sentence)
                    temp_sentence = current_sentence_to_merge
            if temp_sentence: # 将最后一个构建的（或未合并的）句子加入列表
                final_sentences_merged.append(temp_sentence)

        # 对合并后的每个句子进行清理和随机标点移除
        processed_temp = []
        for sentence_val in final_sentences_merged:
            s_loop = sentence_val.strip()
            # 移除句末可能存在的逗号
            if s_loop.endswith(",") or s_loop.endswith("，"):
                s_loop = s_loop[:-1].strip()
            if s_loop: # 确保句子不为空
                s_loop = random_remove_punctuation(s_loop) # 随机标点移除
            if s_loop: # 再次确保句子不为空
                processed_temp.append(s_loop)
        intermediate_sentences_placeholders = processed_temp

    # 统一的书名号恢复步骤
    final_sentences_recovered = []
    if perform_book_title_recovery_here and local_book_title_mapping:
        # 如果有书名号映射且需要恢复，则调用恢复函数
        final_sentences_recovered = recover_book_titles(intermediate_sentences_placeholders, local_book_title_mapping)
    else: # 否则，直接使用中间结果
        final_sentences_recovered = intermediate_sentences_placeholders

    # 返回最终结果，并再次过滤空字符串
    return [s for s in final_sentences_recovered if s.strip()]


def random_remove_punctuation(text: str) -> str:
    """随机处理标点符号，模拟人类打字习惯

    Args:
        text: 要处理的文本

    Returns:
        str: 处理后的文本
    """
    result = ""
    text_len = len(text)

    for i, char in enumerate(text):
        if char == "。" and i == text_len - 1:  # 结尾的句号
            if random.random() > 0.1:  # 90%概率删除结尾句号
                continue
        # elif char == "，":
        #     rand = random.random()
        #     if rand < 0.25:  # 25%概率删除逗号
        #         continue
        #     elif rand < 0.2:  # 20%概率把逗号变成空格
        #         result += " "
        #         continue
        result += char
    return result


def process_llm_response(text: str) -> list[str]:
    # 先保护颜文字
    if global_config.response_splitter.enable_kaomoji_protection:
        protected_text, kaomoji_mapping = protect_kaomoji(text)
        logger.trace(f"保护颜文字后的文本: {protected_text}")
    else:
        protected_text = text
        kaomoji_mapping = {}
    # 提取被 [] 包裹且包含中文的内容
    pattern = re.compile(r"[\[](?=.*[一-鿿]).*?[\]）]")
    # _extracted_contents = pattern.findall(text)
    _extracted_contents = pattern.findall(protected_text)  # 在保护后的文本上查找
    # 去除 () 和 [] 及其包裹的内容
    cleaned_text = pattern.sub("", protected_text)

    if cleaned_text == "":
        return ["呃呃"]

    logger.debug(f"{text}去除括号处理后的文本: {cleaned_text}")
    # 对清理后的文本进行进一步处理
    max_length = global_config.response_splitter.max_length * 2
    max_sentence_num = global_config.response_splitter.max_sentence_num
    # 如果基本上是中文，则进行长度过滤
    if get_western_ratio(cleaned_text) < 0.1:
        if len(cleaned_text) > max_length:
            logger.warning(f"回复过长 ({len(cleaned_text)} 字符)，返回默认回复")
            return ["懒得说"]

    typo_generator = ChineseTypoGenerator(
        error_rate=global_config.chinese_typo.error_rate,
        min_freq=global_config.chinese_typo.min_freq,
        tone_error_rate=global_config.chinese_typo.tone_error_rate,
        word_replace_rate=global_config.chinese_typo.word_replace_rate,
    )

    if global_config.response_splitter.enable:
        split_sentences = split_into_sentences_w_remove_punctuation(cleaned_text)
    else:
        split_sentences = [cleaned_text]

    sentences = []
    for sentence in split_sentences:
        if global_config.chinese_typo.enable:
            typoed_text, typo_corrections = typo_generator.create_typo_sentence(sentence)
            sentences.append(typoed_text)
            if typo_corrections:
                sentences.append(typo_corrections)
        else:
            sentences.append(sentence)

    if len(sentences) > max_sentence_num:
        logger.warning(f"分割后消息数量过多 ({len(sentences)} 条)，返回默认回复")
        return [f"{global_config.bot.nickname}不知道哦"]

    # if extracted_contents:
    #     for content in extracted_contents:
    #         sentences.append(content)

    # 在所有句子处理完毕后，对包含占位符的列表进行恢复
    if global_config.response_splitter.enable_kaomoji_protection:
        sentences = recover_kaomoji(sentences, kaomoji_mapping)

    return sentences


def calculate_typing_time(
    input_string: str,
    thinking_start_time: float,
    chinese_time: float = 0.2,
    english_time: float = 0.1,
    is_emoji: bool = False,
) -> float:
    """
    计算输入字符串所需的时间，中文和英文字符有不同的输入时间
        input_string (str): 输入的字符串
        chinese_time (float): 中文字符的输入时间，默认为0.2秒
        english_time (float): 英文字符的输入时间，默认为0.1秒
        is_emoji (bool): 是否为emoji，默认为False

    特殊情况：
    - 如果只有一个中文字符，将使用3倍的中文输入时间
    - 在所有输入结束后，额外加上回车时间0.3秒
    - 如果is_emoji为True，将使用固定1秒的输入时间
    """
    mood_arousal = mood_manager.current_mood.arousal
    typing_speed_multiplier = 1.5**mood_arousal
    chinese_time *= 1 / typing_speed_multiplier
    english_time *= 1 / typing_speed_multiplier

    # 使用 is_han_character 进行判断
    chinese_chars = sum(1 for char in input_string if is_han_character(char))

    if chinese_chars == 1 and len(input_string.strip()) == 1:
        return chinese_time * 3 + 0.3

    total_time = 0
    for char in input_string:
        if is_han_character(char):  # 使用 is_han_character 进行判断
            total_time += chinese_time
        else:
            total_time += english_time

    if is_emoji:
        total_time = 1

    if time.time() - thinking_start_time > 10:
        total_time = 1

    return total_time


def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


def text_to_vector(text):
    """将文本转换为词频向量"""
    # 分词
    words = jieba.lcut(text)
    # 统计词频
    word_freq = Counter(words)
    return word_freq


def find_similar_topics_simple(text: str, topics: list, top_k: int = 5) -> list:
    """使用简单的余弦相似度计算文本相似度"""
    # 将输入文本转换为词频向量
    text_vector = text_to_vector(text)

    # 计算每个主题的相似度
    similarities = []
    for topic in topics:
        topic_vector = text_to_vector(topic)
        # 获取所有唯一词
        all_words = set(text_vector.keys()) | set(topic_vector.keys())
        # 构建向量
        v1 = [text_vector.get(word, 0) for word in all_words]
        v2 = [topic_vector.get(word, 0) for word in all_words]
        # 计算相似度
        similarity = cosine_similarity(v1, v2)
        similarities.append((topic, similarity))

    # 按相似度降序排序并返回前k个
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]


def truncate_message(message: str, max_length=20) -> str:
    """截断消息，使其不超过指定长度"""
    if len(message) > max_length:
        return message[:max_length] + "..."
    return message


def protect_kaomoji(sentence):
    """ "
    识别并保护句子中的颜文字（含括号与无括号），将其替换为占位符，
    并返回替换后的句子和占位符到颜文字的映射表。
    Args:
        sentence (str): 输入的原始句子
    Returns:
        tuple: (处理后的句子, {占位符: 颜文字})
    """
    kaomoji_pattern = re.compile(
        r"("
        r"[(\[（【]"  # 左括号
        r"[^()\[\]（）【】]*?"  # 非括号字符（惰性匹配）
        r"[^一-龥a-zA-Z0-9\s]"  # 非中文、非英文、非数字、非空格字符（必须包含至少一个）
        r"[^()\[\]（）【】]*?"  # 非括号字符（惰性匹配）
        r"[)\]）】"  # 右括号
        r"]"
        r")"
        r"|"
        r"([▼▽・ᴥω･﹏^><≧≦￣｀´∀ヮДд︿﹀へ｡ﾟ╥╯╰︶︹•⁄]{2,15})"
    )

    kaomoji_matches = kaomoji_pattern.findall(sentence)
    placeholder_to_kaomoji = {}

    for idx, match in enumerate(kaomoji_matches):
        kaomoji = match[0] if match[0] else match[1]
        placeholder = f"__KAOMOJI_{idx}__"
        sentence = sentence.replace(kaomoji, placeholder, 1)
        placeholder_to_kaomoji[placeholder] = kaomoji

    return sentence, placeholder_to_kaomoji


def recover_kaomoji(sentences, placeholder_to_kaomoji):
    """
    根据映射表恢复句子中的颜文字。
    Args:
        sentences (list): 含有占位符的句子列表
        placeholder_to_kaomoji (dict): 占位符到颜文字的映射表
    Returns:
        list: 恢复颜文字后的句子列表
    """
    recovered_sentences = []
    for sentence in sentences:
        for placeholder, kaomoji in placeholder_to_kaomoji.items():
            sentence = sentence.replace(placeholder, kaomoji)
        recovered_sentences.append(sentence)
    return recovered_sentences


def get_western_ratio(paragraph):
    """计算段落中字母数字字符的西文比例
    原理：检查段落中字母数字字符的西文比例
    通过is_english_letter函数判断每个字符是否为西文
    只检查字母数字字符，忽略标点符号和空格等非字母数字字符

    Args:
        paragraph: 要检查的文本段落

    Returns:
        float: 西文字符比例(0.0-1.0)，如果没有字母数字字符则返回0.0
    """
    alnum_chars = [char for char in paragraph if char.isalnum()]
    if not alnum_chars:
        return 0.0

    western_count = sum(1 for char in alnum_chars if is_english_letter(char))  # 保持使用 is_english_letter
    return western_count / len(alnum_chars)


def count_messages_between(start_time: float, end_time: float, stream_id: str) -> tuple[int, int]:
    """计算两个时间点之间的消息数量和文本总长度

    Args:
        start_time (float): 起始时间戳 (不包含)
        end_time (float): 结束时间戳 (包含)
        stream_id (str): 聊天流ID

    Returns:
        tuple[int, int]: (消息数量, 文本总长度)
    """
    count = 0
    total_length = 0

    # 参数校验 (可选但推荐)
    if start_time >= end_time:
        # logger.debug(f"开始时间 {start_time} 大于或等于结束时间 {end_time}，返回 0, 0")
        return 0, 0
    if not stream_id:
        logger.error("stream_id 不能为空")
        return 0, 0

    # 直接查询时间范围内的消息
    # time > start_time AND time <= end_time
    query = {"chat_id": stream_id, "time": {"$gt": start_time, "$lte": end_time}}

    try:
        # 执行查询
        messages_cursor = db.messages.find(query)

        # 遍历结果计算数量和长度
        for msg in messages_cursor:
            count += 1
            total_length += len(msg.get("processed_plain_text", ""))

        # logger.debug(f"查询范围 ({start_time}, {end_time}] 内找到 {count} 条消息，总长度 {total_length}")
        return count, total_length

    except PyMongoError as e:
        logger.error(f"查询 stream_id={stream_id} 在 ({start_time}, {end_time}] 范围内的消息时出错: {e}")
        return 0, 0
    except Exception as e:  # 保留一个通用异常捕获以防万一
        logger.error(f"计算消息数量时发生意外错误: {e}")
        return 0, 0


def translate_timestamp_to_human_readable(timestamp: float, mode: str = "normal") -> str:
    """将时间戳转换为人类可读的时间格式

    Args:
        timestamp: 时间戳
        mode: 转换模式，"normal"为标准格式，"relative"为相对时间格式

    Returns:
        str: 格式化后的时间字符串
    """
    if mode == "normal":
        return time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(timestamp))
    elif mode == "relative":
        now = time.time()
        diff = now - timestamp

        if diff < 20:
            return "刚刚:\n"
        elif diff < 60:
            return f"{int(diff)}秒前:\n"
        elif diff < 3600:
            return f"{int(diff / 60)}分钟前:\n"
        elif diff < 86400:
            return f"{int(diff / 3600)}小时前:\n"
        elif diff < 86400 * 2:
            return f"{int(diff / 86400)}天前:\n"
        else:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) + ":\n"
    else:  # mode = "lite" or unknown
        # 只返回时分秒格式，喵~
        return time.strftime("%H:%M:%S", time.localtime(timestamp))


def parse_text_timestamps(text: str, mode: str = "normal") -> str:
    """解析文本中的时间戳并转换为可读时间格式

    Args:
        text: 包含时间戳的文本，时间戳应以[]包裹
        mode: 转换模式，传递给translate_timestamp_to_human_readable，"normal"或"relative"

    Returns:
        str: 替换后的文本

    转换规则:
    - normal模式: 将文本中所有时间戳转换为可读格式
    - lite模式:
        - 第一个和最后一个时间戳必须转换
        - 以5秒为间隔划分时间段，每段最多转换一个时间戳
        - 不转换的时间戳替换为空字符串
    """
    # 匹配[数字]或[数字.数字]格式的时间戳
    pattern = r"\[(\d+(?:\.\d+)?)\]"

    # 找出所有匹配的时间戳
    matches = list(re.finditer(pattern, text))

    if not matches:
        return text

    # normal模式: 直接转换所有时间戳
    if mode == "normal":
        result_text = text
        for match in matches:
            timestamp = float(match.group(1))
            readable_time = translate_timestamp_to_human_readable(timestamp, "normal")
            # 由于替换会改变文本长度，需要使用正则替换而非直接替换
            pattern_instance = re.escape(match.group(0))
            result_text = re.sub(pattern_instance, readable_time, result_text, count=1)
        return result_text
    else:
        # lite模式: 按5秒间隔划分并选择性转换
        result_text = text

        # 提取所有时间戳及其位置
        timestamps = [(float(m.group(1)), m) for m in matches]
        timestamps.sort(key=lambda x: x[0])  # 按时间戳升序排序

        if not timestamps:
            return text

        # 获取第一个和最后一个时间戳
        first_timestamp, first_match = timestamps[0]
        last_timestamp, last_match = timestamps[-1]

        # 将时间范围划分成5秒间隔的时间段
        time_segments = {}

        # 对所有时间戳按15秒间隔分组
        for ts, match in timestamps:
            segment_key = int(ts // 15)  # 将时间戳除以15取整，作为时间段的键
            if segment_key not in time_segments:
                time_segments[segment_key] = []
            time_segments[segment_key].append((ts, match))

        # 记录需要转换的时间戳
        to_convert = []

        # 从每个时间段中选择一个时间戳进行转换
        for _, segment_timestamps in time_segments.items():
            # 选择这个时间段中的第一个时间戳
            to_convert.append(segment_timestamps[0])

        # 确保第一个和最后一个时间戳在转换列表中
        first_in_list = False
        last_in_list = False

        for ts, _ in to_convert:
            if ts == first_timestamp:
                first_in_list = True
            if ts == last_timestamp:
                last_in_list = True

        if not first_in_list:
            to_convert.append((first_timestamp, first_match))
        if not last_in_list:
            to_convert.append((last_timestamp, last_match))

        # 创建需要转换的时间戳集合，用于快速查找
        to_convert_set = {match.group(0) for _, match in to_convert}

        # 首先替换所有不需要转换的时间戳为空字符串
        for _, match in timestamps:
            if match.group(0) not in to_convert_set:
                pattern_instance = re.escape(match.group(0))
                result_text = re.sub(pattern_instance, "", result_text, count=1)

        # 按照时间戳原始顺序排序，避免替换时位置错误
        to_convert.sort(key=lambda x: x[1].start())

        # 执行替换
        # 由于替换会改变文本长度，从后向前替换
        to_convert.reverse()
        for ts, match in to_convert:
            readable_time = translate_timestamp_to_human_readable(ts, "relative")
            pattern_instance = re.escape(match.group(0))
            result_text = re.sub(pattern_instance, readable_time, result_text, count=1)

        return result_text