from __future__ import annotations
"""
简历优化 —— 轻量功能

输入：简历文本 + 目标岗位（可选）
输出：具体修改建议 + 匹配度分析
"""
from src.api_client import chat, chat_stream
from src.prompts import RESUME_SYSTEM, RESUME_USER


def optimize_resume(resume_text: str, target_jd: str = "", stream: bool = False):
    """
    优化简历

    Args:
        resume_text: 简历原文
        target_jd: 目标岗位 JD（可选）
        stream: 是否流式输出

    Returns:
        AI 优化建议（字符串或生成器）
    """
    messages = [
        {"role": "system", "content": RESUME_SYSTEM},
        {"role": "user", "content": RESUME_USER.format(
            resume_text=resume_text,
            target_jd=target_jd or "（未提供，请仅针对简历本身给出优化建议）",
        )},
    ]

    if stream:
        return chat_stream(messages, temperature=0.5, max_tokens=2048)
    else:
        return chat(messages, temperature=0.5, max_tokens=2048)
