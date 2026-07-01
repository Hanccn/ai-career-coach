from __future__ import annotations
"""
JD 分析 —— 单份分析 + 多份横向对比
"""
from src.api_client import chat, chat_stream
from src.prompts import (
    JD_ANALYSIS_SYSTEM, JD_ANALYSIS_USER,
    JD_COMPARE_SYSTEM, JD_COMPARE_USER,
    GAP_SYSTEM, GAP_USER,
)


def analyze_jd(jd_text: str, stream: bool = False):
    """
    分析单份岗位 JD
    """
    messages = [
        {"role": "system", "content": JD_ANALYSIS_SYSTEM},
        {"role": "user", "content": JD_ANALYSIS_USER.format(jd_text=jd_text)},
    ]

    if stream:
        return chat_stream(messages, temperature=0.5, max_tokens=2048)
    else:
        return chat(messages, temperature=0.5, max_tokens=2048)


def compare_jds(jd_list: list[str]):
    """
    横向对比多份 JD，找共性差异

    Args:
        jd_list: JD 文本列表，每项是一份完整 JD

    Returns:
        对比分析结果（Markdown）
    """
    # 格式化 JD 列表
    formatted = ""
    for i, jd in enumerate(jd_list):
        formatted += f"\n---\n### JD-{i+1}\n{jd[:2000]}\n"

    messages = [
        {"role": "system", "content": JD_COMPARE_SYSTEM},
        {"role": "user", "content": JD_COMPARE_USER.format(
            count=len(jd_list),
            jd_list=formatted,
        )},
    ]

    return chat(messages, temperature=0.5, max_tokens=3072)


def analyze_gap(jd_analysis: str, resume_text: str):
    """
    JD 分析结果 vs 简历 → 差距分析
    """
    messages = [
        {"role": "system", "content": GAP_SYSTEM},
        {"role": "user", "content": GAP_USER.format(
            jd_analysis=jd_analysis[:3500],
            resume_text=resume_text[:3500],
        )},
    ]
    return chat(messages, temperature=0.6, max_tokens=4096)
