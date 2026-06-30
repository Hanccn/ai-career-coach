"""
学习路线规划 —— 从 JD 分析到行动计划

输入：岗位名称 + JD 分析摘要
输出：按周划分的学习路线（默认 3 周）
"""
from src.api_client import chat, chat_stream
from src.prompts import ROADMAP_SYSTEM, ROADMAP_USER


def generate_roadmap(role: str, jd_summary: str, weeks: int = 3):
    """
    生成学习路线规划

    Args:
        role: 目标岗位名称
        jd_summary: JD 分析结果（摘要）
        weeks: 学习周期（周数）

    Returns:
        AI 生成的学习路线（Markdown 字符串）
    """
    messages = [
        {"role": "system", "content": ROADMAP_SYSTEM},
        {"role": "user", "content": ROADMAP_USER.format(
            role=role,
            jd_summary=jd_summary[:3000],  # 截断过长输入
            weeks=weeks,
        )},
    ]

    return chat(messages, temperature=0.7, max_tokens=3072)
