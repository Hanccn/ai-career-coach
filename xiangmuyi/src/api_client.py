"""
DeepSeek API 客户端封装
使用 OpenAI 兼容接口，支持 deepseek-chat 和 deepseek-reasoner

密钥来源优先级：st.secrets > 环境变量 > .env
"""
from __future__ import annotations
import os
from openai import OpenAI
from dotenv import load_dotenv

# 多路径尝试加载 .env（Streamlit 运行时 cwd 可能不是项目根目录）
for _try_dir in [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # src/../ = 项目根
    os.getcwd(),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai-career-coach"),
]:
    _env_path = os.path.join(_try_dir, ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)
        break
else:
    load_dotenv()  # 兜底：从 cwd 找

_client = None


def _get_env_config(key: str, default: str = "") -> str:
    """纯环境变量取值（不触发 Streamlit，安全在 import 阶段调用）"""
    return os.getenv(key, default)


# 模块级常量（仅从环境变量取值，不会触发 st.secrets）
DEEPSEEK_API_KEY = _get_env_config("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _get_env_config("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _get_env_config("DEEPSEEK_MODEL", "deepseek-chat")


def _resolve_api_key() -> str:
    """懒加载解析 API Key：先环境变量，再 st.secrets（仅在已初始化的 Streamlit 上下文中）"""
    if DEEPSEEK_API_KEY:
        return DEEPSEEK_API_KEY
    try:
        import streamlit as st
        return st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return ""


def get_client() -> OpenAI:
    """获取 DeepSeek API 客户端（懒加载单例）"""
    global _client
    if _client is None:
        key = _resolve_api_key()
        if not key:
            raise ValueError(
                "请在 .env 文件中设置 DEEPSEEK_API_KEY。\n"
                "1. 复制 .env.example 为 .env\n"
                "2. 填入你的 DeepSeek API Key\n"
                "3. 从 https://platform.deepseek.com 获取"
            )
        import httpx
        _client = OpenAI(
            api_key=key,
            base_url=DEEPSEEK_BASE_URL,
            http_client=httpx.Client(trust_env=False),  # 跳过系统代理，避免 TLS 中间人问题
        )
    return _client


def chat(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: str = None,
) -> str:
    """
    调用 DeepSeek Chat API
    """
    client = get_client()
    model = model or DEEPSEEK_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def chat_stream(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: str = None,
):
    """
    流式调用 DeepSeek Chat API（生成器）
    """
    client = get_client()
    model = model or DEEPSEEK_MODEL

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
