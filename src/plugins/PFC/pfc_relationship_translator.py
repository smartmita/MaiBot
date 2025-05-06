# 直接完整导入群聊的relationship_manager.py可能不可取，因为对于PFC的planner来说，其暗示了选择回复，所以新建代码文件来适配PFC的决策层面
from src.common.logger_manager import get_logger

logger = get_logger("pfc_relationship_translator")

# 这个函数复用 relationship_manager.py 中的 calculate_level_num 的逻辑
def _calculate_relationship_level_num(relationship_value: float) -> int:
    """
    根据关系值计算关系等级编号 (0-5)。
    这里的阈值应与 relationship_manager.py 中的保持一致
    """
    if not isinstance(relationship_value, (int, float)):
        logger.warning(f"传入的 relationship_value '{relationship_value}' 不是有效的数值类型，默认为0。")
        relationship_value = 0.0

    if -1000 <= relationship_value < -227:
        level_num = 0  # 厌恶
    elif -227 <= relationship_value < -73:
        level_num = 1  # 冷漠
    elif -73 <= relationship_value < 227:
        level_num = 2  # 普通/认识
    elif 227 <= relationship_value < 587:
        level_num = 3  # 友好
    elif 587 <= relationship_value < 900:
        level_num = 4  # 喜欢
    elif 900 <= relationship_value <= 1000:
        level_num = 5  # 暧昧
    else:
        # 超出范围的值处理
        if relationship_value > 1000:
            level_num = 5
        elif relationship_value < -1000:
            level_num = 0
        else: # 理论上不会到这里，除非前面的条件逻辑有误
            logger.warning(f"关系值 {relationship_value} 未落入任何预设范围，默认为普通。")
            level_num = 2 
    return level_num

def translate_relationship_value_to_text(relationship_value: float) -> str:
    """
    将数值型的关系值转换为PFC私聊场景下简洁的关系描述文本。
    """
    level_num = _calculate_relationship_level_num(relationship_value)

    relationship_descriptions = [
        "厌恶",   # level_num 0
        "冷漠",   # level_num 1
        "初识",   # level_num 2
        "友好",   # level_num 3
        "喜欢",   # level_num 4
        "暧昧"    # level_num 5
    ]

    if 0 <= level_num < len(relationship_descriptions):
        description = relationship_descriptions[level_num]
    else:
        description = "普通" # 默认或错误情况
        logger.warning(f"计算出的 level_num ({level_num}) 无效，关系描述默认为 '普通'")

    return f"你们的关系是：{description}。"