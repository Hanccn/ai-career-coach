"""
产品数据统计、面试进步追踪、使用记录持久化
"""
import streamlit as st
import re
import json
import os
from datetime import datetime

# ── JSON 持久化路径 ──
def _history_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usage_history.json")


def _load_history():
    path = _history_path()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"jd_analyses": [], "resumes": [], "roadmaps": [], "interviews": []}


def _save_history(data):
    try:
        with open(_history_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# ── Session State 初始化 ──
def init_stats_state():
    defaults = {
        "total_jd_analyses": 0,
        "total_resume_optimizations": 0,
        "total_interviews": 0,
        "total_roadmaps": 0,
        "roles_covered": set(),
        "interview_scores": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 从持久化文件恢复计数
    hist = _load_history()
    st.session_state.total_jd_analyses = len(hist.get("jd_analyses", []))
    st.session_state.total_resume_optimizations = len(hist.get("resumes", []))
    st.session_state.total_roadmaps = len(hist.get("roadmaps", []))
    st.session_state.total_interviews = len(hist.get("interviews", []))
    st.session_state.roles_covered = set(
        r.get("role", "") for r in hist.get("interviews", []) if r.get("role")
    )
    st.session_state.interview_scores = [
        {"role": r.get("role", ""), "score": r.get("score"), "date": r.get("date", ""), "evaluation": r.get("evaluation", "")}
        for r in hist.get("interviews", [])
    ]


# ── 记录函数 ──
def record_jd_analysis():
    st.session_state.total_jd_analyses += 1
    hist = _load_history()
    jd_text = st.session_state.get("jd_last_input", "")
    result = st.session_state.get("jd_last_result", "")
    guessed = st.session_state.get("_last_guessed_role", "")
    if not guessed:
        combined = (result + jd_text).lower()
        kw_map = {
            "AI产品经理": ["ai产品", "大模型", "人工智能产品"],
            "产品经理（通用）": ["产品经理", "产品实习", "prd", "需求分析"],
            "数据分析师": ["数据分析", "数据产品", "数据运营", "sql", "bi"],
            "前端开发工程师": ["前端", "react", "vue", "web前端"],
            "后端开发工程师": ["后端", "java", "python开发", "服务端"],
            "算法工程师": ["算法", "机器学习", "深度学习", "nlp", "cv"],
            "用户运营": ["用户运营", "用户增长", "社群运营"],
            "UI/UX设计师": ["ui设计", "ux", "交互设计", "视觉设计", "figma"],
        }
        for role, keys in kw_map.items():
            if any(k in combined for k in keys):
                guessed = role
                break
        if not guessed:
            guessed = "产品经理（通用）"

    hist.setdefault("jd_analyses", []).append({
        "date": datetime.now().strftime("%m/%d %H:%M"),
        "role": guessed,
        "jd_snippet": jd_text[:200],
        "jd_full": jd_text[:3000],
        "result": result[:3000],
    })
    _save_history(hist)


def record_jd_compare(count=0):
    """多份 JD 对比也计入"""
    for _ in range(count):
        st.session_state.total_jd_analyses += 1


def record_resume_optimization():
    st.session_state.total_resume_optimizations += 1
    hist = _load_history()
    resume_text = st.session_state.get("user_resume", "")
    hist.setdefault("resumes", []).append({
        "date": datetime.now().strftime("%m/%d %H:%M"),
        "resume_snippet": resume_text[:200],
    })
    _save_history(hist)


def record_roadmap():
    st.session_state.total_roadmaps += 1
    hist = _load_history()
    hist.setdefault("roadmaps", []).append({
        "date": datetime.now().strftime("%m/%d %H:%M"),
        "role": st.session_state.get("roadmap_role_input", ""),
        "weeks": st.session_state.get("_roadmap_weeks", 3),
    })
    _save_history(hist)


def record_interview(role, evaluation_text):
    st.session_state.total_interviews += 1
    if isinstance(st.session_state.roles_covered, set):
        st.session_state.roles_covered.add(role)

    score = _parse_score(evaluation_text)
    entry = {
        "role": role,
        "score": score,
        "date": datetime.now().strftime("%m/%d %H:%M"),
        "evaluation": evaluation_text[:5000],
        "rounds": st.session_state.get("interview_round", 0),
    }
    st.session_state.interview_scores.append(entry)
    hist = _load_history()
    hist.setdefault("interviews", []).append(entry)
    _save_history(hist)


# ── 历史查询 ──
def get_jd_history():
    return list(reversed(_load_history().get("jd_analyses", [])))


def get_interview_history():
    return list(reversed(_load_history().get("interviews", [])))


def get_all_history():
    return _load_history()


def clear_all_history():
    _save_history({"jd_analyses": [], "resumes": [], "roadmaps": [], "interviews": []})
    init_stats_state()


# ── 评分工具 ──
def _parse_score(text):
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


def get_score_trend():
    return [
        {"index": i + 1, "role": s["role"], "score": s["score"], "date": s["date"]}
        for i, s in enumerate(st.session_state.interview_scores)
        if s["score"] is not None
    ]


def get_latest_score():
    scores = st.session_state.interview_scores
    if scores and scores[-1]["score"] is not None:
        return scores[-1]["score"]
    return None


def get_score_delta():
    scores = [s["score"] for s in st.session_state.interview_scores if s["score"] is not None]
    if len(scores) >= 2:
        return scores[-1] - scores[-2]
    return None
