"""
权重计算模块 - 为主动聊天功能提供权重和触发概率计算

此模块包含：
1. 基于关系等级的权重计算函数
2. 根据关系调整触发概率的函数
"""
from typing import Dict, Any, List, Tuple, Optional
import logging
import traceback

# 获取日志记录器
logger = logging.getLogger("pfc_idle_chat")

# 导入关系管理器
from src.chat.person_info.person_info import person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager

# 关系等级范围，用于替代直接访问relationship_manager.level_ranges
DEFAULT_LEVEL_RANGES = [-100, -50, 0, 50, 100, 150]

def get_relationship_level_ranges() -> List[float]:
    """获取关系等级的阈值范围
    
    Returns:
        List[float]: 关系等级的阈值范围列表
    """
    try:
        # 尝试从relationship_manager获取level_ranges属性
        if hasattr(relationship_manager, 'level_ranges'):
            return relationship_manager.level_ranges
        
        # 如果没有这个属性，使用默认值
        logger.warning("RelationshipManager对象没有level_ranges属性，使用默认值")
        return DEFAULT_LEVEL_RANGES
    except Exception as e:
        logger.error(f"获取关系等级范围时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return DEFAULT_LEVEL_RANGES

def calculate_user_weight(
    relationship_level: int, 
    relationship_value: float, 
    level_ranges: List[float] = None, 
    in_pending: bool = False
) -> float:
    """计算用户的权重，用于主动聊天的用户选择

    Args:
        relationship_level: 关系等级 (0=厌恶, 1=冷漠, 2=一般, 3=友好, 4=喜欢, 5=依赖)
        relationship_value: 精确关系值
        level_ranges: 关系等级的范围边界值列表，默认为None时自动获取
        in_pending: 是否在待回复列表中，默认为False

    Returns:
        float: 计算得到的权重值
    """
    # 如果未提供等级范围，则自动获取
    if level_ranges is None:
        level_ranges = get_relationship_level_ranges()
        
    # 对关系等级应用指数增长权重，让高关系等级的用户有明显更高的权重
    if relationship_level <= 1:  # 厌恶和冷漠
        # 低关系给予较低的权重，但仍有一定概率被选中
        base_weight = max(0.3, 0.5 * (relationship_level + 1))
    elif relationship_level == 2:  # 一般
        # 中等关系
        base_weight = 2.0
    elif relationship_level == 3:  # 友好
        # 友好关系，权重开始显著增加
        base_weight = 4.0
    elif relationship_level == 4:  # 喜欢
        # 喜欢关系，较高权重
        base_weight = 7.0
    else:  # 依赖 (relationship_level == 5)
        # 依赖关系，极高权重
        base_weight = 12.0
    
    # 微调：在同一关系等级内，基于精确关系值进行微调（±20%）
    if relationship_level < len(level_ranges) - 1:
        curr_min = level_ranges[relationship_level]
        next_min = level_ranges[relationship_level + 1]
        # 计算在当前等级内的相对位置 (0.0 到 1.0)
        if next_min > curr_min:
            relative_pos = min(1.0, max(0.0, (relationship_value - curr_min) / (next_min - curr_min)))
            # 应用±20%的微调
            fine_tune_factor = 0.8 + (0.4 * relative_pos)  # 0.8 到 1.2
            base_weight *= fine_tune_factor
    
    # 如果在待回复列表中，降低权重
    if in_pending:
        base_weight *= 0.1
    
    return base_weight

def calculate_base_trigger_probability(relationship_level: int) -> float:
    """基于关系等级计算基础触发概率

    Args:
        relationship_level: 关系等级 (0=厌恶, 1=冷漠, 2=一般, 3=友好, 4=喜欢, 5=依赖)

    Returns:
        float: 基础触发概率
    """
    # 基础触发概率，基于关系等级动态调整
    probability_map = {
        0: 0.1,   # 厌恶
        1: 0.15,  # 冷漠
        2: 0.25,  # 一般
        3: 0.35,  # 友好
        4: 0.45,  # 喜欢
        5: 0.6    # 依赖
    }
    return probability_map.get(relationship_level, 0.3)

def process_instances_weights(
    instances_with_rel: List[Dict[str, Any]],
    level_ranges: List[float] = None
) -> None:
    """处理所有实例的权重计算

    Args:
        instances_with_rel: 包含实例和关系数据的列表
        level_ranges: 关系等级的范围边界值列表，默认为None时自动获取
    """
    # 如果未提供等级范围，则自动获取
    if level_ranges is None:
        level_ranges = get_relationship_level_ranges()
        
    # 计算每个实例的权重
    for data in instances_with_rel:
        # 获取关系数据
        relationship_level = data["relationship_level"]
        relationship_value = data["relationship_value"]
        in_pending = data["in_pending"]
        
        # 计算权重
        base_weight = calculate_user_weight(
            relationship_level, 
            relationship_value, 
            level_ranges, 
            in_pending
        )
        
        # 记录日志
        logger.debug(
            f"[私聊][{data['instance'].private_name}] "
            f"关系等级: {relationship_level}, "
            f"关系值: {relationship_value:.2f}, "
            f"基础权重: {base_weight:.2f}"
        )
        
        # 最终权重
        data["weight"] = base_weight
    
    # 记录所有用户的权重情况
    weight_info = "\n".join([
        f"- {data['instance'].private_name}: "
        f"关系等级={data['relationship_level']}, "
        f"权重={data['weight']:.2f}" for data in instances_with_rel
    ])
    logger.info(f"所有用户的权重分配:\n{weight_info}")

def find_max_relationship_user(instances_with_rel: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """找出具有最高关系值的用户

    Args:
        instances_with_rel: 包含实例和关系数据的列表

    Returns:
        Dict[str, Any]: 最高关系值的用户数据，如果列表为空则返回None
    """
    if not instances_with_rel:
        return None
    return max(instances_with_rel, key=lambda x: x["relationship_value"])

async def get_user_relationship_data(user_id, platform="qq", private_name=None) -> Tuple[float, int, str]:
    """获取用户的关系数据
    
    Args:
        user_id: 用户ID
        platform: 平台，默认为"qq"
        private_name: 用户名称，用于日志记录
        
    Returns:
        Tuple[float, int, str]: (关系值，关系等级数字，关系等级描述)
    """
    try:
        log_prefix = f"[私聊][{private_name}]" if private_name else "[私聊]"
        
        # 默认关系数据
        relationship_value = 0.0
        relationship_level_num = 2  # 对应 "一般"
        relationship_description = "普通"
        
        # 生成person_id并获取关系值
        person_id = person_info_manager.get_person_id(platform, user_id)
        if private_name:
            logger.info(f"{log_prefix}生成的person_id: {person_id}")
        
        # 使用person_info_manager获取关系值
        relationship_value = await person_info_manager.get_value(person_id, "relationship_value")
        if private_name:
            logger.info(f"{log_prefix}获取到原始关系值: {relationship_value}")
        
        # 确保关系值为浮点类型
        relationship_value = ensure_float_relationship(relationship_value, person_id)
        
        # 计算关系等级
        relationship_level_num = calculate_relationship_level_num(relationship_value)
        
        # 获取关系等级描述
        relationship_level_names = ["厌恶", "冷漠", "一般", "友好", "喜欢", "依赖"]
        relationship_description = relationship_level_names[relationship_level_num]
        
        if private_name:
            logger.info(f"{log_prefix}关系值: {relationship_value:.2f}, 关系等级: {relationship_description}")
        
        return relationship_value, relationship_level_num, relationship_description
    
    except Exception as e:
        if private_name:
            logger.error(f"[私聊][{private_name}]获取关系数据失败: {str(e)}")
            logger.error(traceback.format_exc())
        else:
            logger.error(f"获取关系数据失败: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 使用默认关系数据
        return 0.0, 2, "普通"

def ensure_float_relationship(value, person_id=None) -> float:
    """确保关系值为浮点类型
    
    这个函数是对relationship_manager.ensure_float的封装，
    提供错误处理以防止relationship_manager不存在ensure_float方法
    
    Args:
        value: 关系值
        person_id: 用户ID，用于日志记录
        
    Returns:
        float: 转换为浮点类型的关系值
    """
    try:
        if hasattr(relationship_manager, 'ensure_float'):
            return relationship_manager.ensure_float(value, person_id)
        
        # 如果relationship_manager没有ensure_float方法，手动实现转换
        try:
            # 处理各种可能的类型
            if hasattr(value, 'value'):  # 处理Decimal128等有value属性的类型
                return float(value.value)
            else:
                return float(value)
        except (ValueError, TypeError):
            logger.warning(f"无法将关系值 '{value}' 转换为浮点数，使用默认值0.0")
            return 0.0
    except Exception as e:
        logger.error(f"确保关系值为浮点类型时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return 0.0

def calculate_relationship_level_num(relationship_value: float) -> int:
    """计算关系等级数字
    
    这个函数是对relationship_manager.calculate_level_num的封装，
    提供错误处理以防止relationship_manager不存在calculate_level_num方法
    
    Args:
        relationship_value: 关系值
        
    Returns:
        int: 关系等级数字 (0-5)
    """
    try:
        if hasattr(relationship_manager, 'calculate_level_num'):
            return relationship_manager.calculate_level_num(relationship_value)
        
        # 如果relationship_manager没有calculate_level_num方法，手动实现计算
        level_ranges = get_relationship_level_ranges()
        
        for i in range(len(level_ranges) - 1, -1, -1):
            if relationship_value >= level_ranges[i]:
                return i
        
        # 如果小于所有范围，返回最低等级
        return 0
    except Exception as e:
        logger.error(f"计算关系等级时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return 2  # 默认为"一般" 