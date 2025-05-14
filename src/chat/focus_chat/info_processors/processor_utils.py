import difflib
import random
import time


def calculate_similarity(text_a: str, text_b: str) -> float:
    """
    计算两个文本字符串的相似度。
    """
    if not text_a or not text_b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, text_a, text_b)
    return matcher.ratio()


def calculate_replacement_probability(similarity: float) -> float:
    """
    根据相似度计算替换的概率。
    规则：
    - 相似度 <= 0.4: 概率 = 0
    - 相似度 >= 0.9: 概率 = 1
    - 相似度 == 0.6: 概率 = 0.7
    - 0.4 < 相似度 <= 0.6: 线性插值 (0.4, 0) 到 (0.6, 0.7)
    - 0.6 < 相似度 < 0.9: 线性插值 (0.6, 0.7) 到 (0.9, 1.0)
    """
    if similarity <= 0.4:
        return 0.0
    elif similarity >= 0.9:
        return 1.0
    elif 0.4 < similarity <= 0.6:
        # p = 3.5 * s - 1.4
        probability = 3.5 * similarity - 1.4
        return max(0.0, probability)
    else:  # 0.6 < similarity < 0.9
        # p = s + 0.1
        probability = similarity + 0.1
        return min(1.0, max(0.0, probability))


def get_spark():
    local_random = random.Random()
    current_minute = int(time.strftime("%M"))
    local_random.seed(current_minute)

    hf_options = [
        ("可以参考之前的想法，在原来想法的基础上继续思考", 0.2),
        ("可以参考之前的想法，在原来的想法上尝试新的话题", 0.4),
        ("不要太深入", 0.2),
        ("进行深入思考", 0.2),
    ]
    # 加权随机选择思考指导
    hf_do_next = local_random.choices(
        [option[0] for option in hf_options], weights=[option[1] for option in hf_options], k=1
    )[0]

    return hf_do_next
