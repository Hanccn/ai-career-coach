"""
岗位百科模块 —— AI 详细介绍岗位的日常工作、能力要求、发展路径
"""
from src.api_client import chat
from src.prompts import ROLE_ENCYCLOPEDIA_SYSTEM, ROLE_ENCYCLOPEDIA_USER


def describe_role(role_name: str) -> str:
    """获取某个岗位的详细介绍

    Args:
        role_name: 岗位名称，如 "产品经理（通用）"

    Returns:
        AI 生成的岗位百科（Markdown）
    """
    prompt = ROLE_ENCYCLOPEDIA_USER.format(role_name=role_name)

    result = chat([
        {"role": "system", "content": ROLE_ENCYCLOPEDIA_SYSTEM},
        {"role": "user", "content": prompt},
    ], temperature=0.6, max_tokens=2048)

    return result
