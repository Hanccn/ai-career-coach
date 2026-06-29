from __future__ import annotations
"""
产品数据统计与面试进步追踪
"""
import streamlit as st
import re
from datetime import datetime


def init_stats_state():
    """初始化统计相关 session state"""
    defaults = {
        "total_jd_analyses": 0,
        "total_resume_optimizations": 0,
        "total_interviews": 0,
        "total_roadmaps": 0,
        "roles_covered": set(),  # 面试过的岗位集合
        "interview_scores": [],  # [{role, score, date, evaluation}, ...]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def record_jd_analysis():
    st.session_state.total_jd_analyses += 1


def record_resume_optimization():
    st.session_state.total_resume_optimizations += 1


def record_roadmap():
    st.session_state.total_roadmaps += 1


def record_interview(role: str, evaluation_text: str):
    """面试完成后记录评分和岗位"""
    st.session_state.total_interviews += 1
    if isinstance(st.session_state.roles_covered, set):
        st.session_state.roles_covered.add(role)

    # 从评估报告提取综合评分
    score = _parse_score(evaluation_text)

    st.session_state.interview_scores.append({
        "role": role,
        "score": score,
        "date": datetime.now().strftime("%m/%d %H:%M"),
        "evaluation": evaluation_text,
    })


def _parse_score(text: str) -> int | None:
    """从评估报告中提取综合评分"""
    # 匹配 "| **综合** | **X** |" 或 "| 综合 | X |"
    patterns = [
        r"\*\*综合\*\*\s*\|\s*\*?\*?(\d+)\*?\*?",
        r"综合\s*\|\s*(\d+)",
        r"综合评分[：:]\s*(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def get_score_trend() -> list:
    """返回评分趋势数据"""
    return [
        {"index": i + 1, "role": s["role"], "score": s["score"], "date": s["date"]}
        for i, s in enumerate(st.session_state.interview_scores)
        if s["score"] is not None
    ]


def get_latest_score() -> int | None:
    """最近一次面试评分"""
    scores = st.session_state.interview_scores
    if scores and scores[-1]["score"] is not None:
        return scores[-1]["score"]
    return None


def get_score_delta() -> int | None:
    """最近一次与上一次的评分变化"""
    scores = [s["score"] for s in st.session_state.interview_scores if s["score"] is not None]
    if len(scores) >= 2:
        return scores[-1] - scores[-2]
    return None
