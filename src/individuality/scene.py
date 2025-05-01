import json
import os
from typing import Any


def load_scenes() -> dict[str, Any]:
    """
    从JSON文件加载场景数据

    Returns:
        Dict: 包含所有场景的字典
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "template_scene.json")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


PERSONALITY_SCENES = load_scenes()


def get_scene_by_factor(factor: str) -> dict | None:
    """
    根据人格因子获取对应的情景测试

    Args:
        factor (str): 人格因子名称

    Returns:
        dict: 包含情景描述的字典
    """
    return PERSONALITY_SCENES.get(factor, None)


def get_all_scenes() -> dict:
    """
    获取所有情景测试

    Returns:
        Dict: 所有情景测试的字典
    """
    return PERSONALITY_SCENES
