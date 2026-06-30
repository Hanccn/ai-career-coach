"""
职业探索模块 —— AI 引导用户发现适合的职业方向
"""
from src.api_client import chat
from src.prompts import CAREER_EXPLORER_SYSTEM, CAREER_EXPLORER_USER


def explore_career(answers: dict) -> str:
    """基于用户的专业、兴趣、工作方式偏好推荐职业方向

    Args:
        answers: {"major": "...", "interest": "...", "style": "..."}

    Returns:
        AI 生成的职业推荐报告（Markdown）
    """
    prompt = CAREER_EXPLORER_USER.format(
        major=answers.get("major", "未指定"),
        interest=answers.get("interest", "未指定"),
        style=answers.get("style", "未指定"),
    )

    result = chat([
        {"role": "system", "content": CAREER_EXPLORER_SYSTEM},
        {"role": "user", "content": prompt},
    ], temperature=0.8, max_tokens=2048)

    return result
