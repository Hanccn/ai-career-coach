from __future__ import annotations
"""
AI 模拟面试 —— 深度功能

核心逻辑：
1. 动态追问：AI 根据候选人回答质量决定下一题方向
2. 隐式评估：每轮记录简短评价，不展示给候选人
3. 最终报告：6 轮结束后生成结构化评估报告
"""
import streamlit as st
from src.api_client import chat
from src.prompts import (
    INTERVIEWER_SYSTEM,
    INTERVIEW_FIRST_QUESTION,
    INTERVIEW_NEXT_QUESTION,
    INTERVIEW_EVALUATION,
    INTERVIEW_MAX_ROUNDS,
    INTERVIEW_ROLES,
)


def init_interview_state():
    """初始化面试相关的 session state"""
    defaults = {
        "interview_active": False,
        "interview_round": 0,
        "interview_history": [],  # 完整对话 [{role, content}, ...]
        "interview_assessments": [],  # 隐式评估 [str, ...]
        "interview_evaluation": None,  # 最终报告
        "interview_role": None,
        "waiting_for_answer": False,
        "current_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_interview(role_name: str, jd_context: str = ""):
    """开始一场新的模拟面试

    Args:
        role_name: 岗位名称
        jd_context: 岗位 JD 原文（可选，有则围绕 JD 提问）
    """
    st.session_state.interview_active = True
    st.session_state.interview_round = 0
    st.session_state.interview_history = []
    st.session_state.interview_assessments = []
    st.session_state.interview_evaluation = None
    st.session_state.interview_role = role_name
    st.session_state.waiting_for_answer = True

    # JD 上下文格式化
    if jd_context.strip():
        jd_section = f"""## 岗位 JD 参考
以下是候选人申请的岗位 JD，你的面试问题必须围绕以下要求展开：

{jd_context[:2000]}

请紧扣这份 JD 中的技能要求和职责描述来提问。"""
    else:
        jd_section = "（候选人未提供具体 JD，请根据「{role}」岗位的通用要求提问。）".format(role=role_name)

    # 动态注入岗位 + JD 到 Prompt
    system_prompt = INTERVIEWER_SYSTEM.format(role=role_name, jd_context=jd_section)
    jd_for_first_q = f"以下是岗位JD参考：\n{jd_context[:1500]}" if jd_context.strip() else "（无具体JD，请基于岗位通用要求提问）"
    first_q_prompt = INTERVIEW_FIRST_QUESTION.format(role=role_name, jd_context=jd_for_first_q)

    # 生成第一个问题
    with st.spinner("面试官正在准备第一个问题..."):
        first_question = chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_q_prompt},
        ], temperature=0.85)

    st.session_state.current_question = first_question
    st.session_state._system_prompt = system_prompt  # 缓存，后续追问复用
    st.session_state._has_jd_context = bool(jd_context.strip())  # 标记有 JD 上下文
    st.session_state.interview_round = 1
    st.session_state.interview_history.append(
        {"role": "assistant", "content": first_question}
    )


def submit_answer(answer: str):
    """候选人提交答案，生成下一题或结束面试"""
    if not answer.strip():
        return

    current_round = st.session_state.interview_round
    last_question = st.session_state.interview_history[-1]["content"]

    # 保存答案
    st.session_state.interview_history.append(
        {"role": "user", "content": answer}
    )

    if current_round >= INTERVIEW_MAX_ROUNDS:
        # 最后一轮 → 生成评估
        with st.spinner("面试结束，正在生成评估报告..."):
            evaluation = _generate_evaluation()
        st.session_state.interview_evaluation = evaluation
        st.session_state.interview_active = False
        st.session_state.interview_completed = True  # 标记：需要记录数据
        st.session_state.waiting_for_answer = False
        st.rerun()
    else:
        # 生成下一题
        with st.spinner("面试官正在思考下一个问题..."):
            next_data = _generate_next_question(
                history=st.session_state.interview_history,
                last_question=last_question,
                last_answer=answer,
            )

        st.session_state.current_question = next_data["question"]
        st.session_state.interview_assessments.append(next_data["assessment"])
        st.session_state.interview_round = current_round + 1
        st.session_state.interview_history.append(
            {"role": "assistant", "content": next_data["question"]}
        )
        st.rerun()


def _generate_next_question(history: list, last_question: str, last_answer: str) -> dict:
    """生成下一个面试问题 + 隐式评估"""
    # 格式化对话历史
    history_text = _format_history(history[:-1])  # 不包含最后一轮

    prompt = INTERVIEW_NEXT_QUESTION.format(
        history=history_text,
        last_question=last_question,
        last_answer=last_answer,
    )

    response = chat([
        {"role": "system", "content": st.session_state.get("_system_prompt", INTERVIEWER_SYSTEM.format(role=st.session_state.interview_role))},
        {"role": "user", "content": prompt},
    ], temperature=0.85, max_tokens=1024)

    # 解析 AI 输出
    assessment = ""
    question = ""

    if "[HIDDEN_ASSESSMENT]" in response and "[NEXT_QUESTION]" in response:
        parts = response.split("[NEXT_QUESTION]")
        assessment_part = parts[0].replace("[HIDDEN_ASSESSMENT]", "").strip()
        question = parts[1].strip()
        assessment = assessment_part
    else:
        # 容错：AI 没按格式输出
        assessment = "评估解析失败"
        question = response.strip()

    return {"assessment": assessment, "question": question}


def _generate_evaluation() -> str:
    """生成最终面试评估报告"""
    history_text = _format_history(st.session_state.interview_history)

    # 拼接隐式评估作为参考
    assessments_text = "\n".join(
        f"第{i+1}轮评估：{a}"
        for i, a in enumerate(st.session_state.interview_assessments)
    )

    full_context = f"""对话记录：
{history_text}

---
各轮次隐式评估参考：
{assessments_text}"""

    evaluation = chat([
        {"role": "system", "content": st.session_state.get("_system_prompt", INTERVIEWER_SYSTEM.format(role=st.session_state.interview_role))},
        {"role": "user", "content": INTERVIEW_EVALUATION.format(full_history=full_context)},
    ], temperature=0.6, max_tokens=3072)

    return evaluation


def _format_history(history: list) -> str:
    """格式化对话历史为可读文本"""
    lines = []
    for msg in history:
        role = "👤 候选人" if msg["role"] == "user" else "🎯 面试官"
        lines.append(f"{role}：{msg['content']}")
    return "\n\n".join(lines)


def reset_interview():
    """重置面试状态"""
    st.session_state.interview_active = False
    st.session_state.interview_round = 0
    st.session_state.interview_history = []
    st.session_state.interview_assessments = []
    st.session_state.interview_evaluation = None
    st.session_state.interview_role = None
    st.session_state.waiting_for_answer = False
    st.session_state.current_question = ""
