"""
AI 求职助手 —— 产品经理实习项目 #3
技术栈：Streamlit + DeepSeek API

运行方式：
    streamlit run app.py
"""
from __future__ import annotations
import streamlit as st
import sys, os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.insert(0, BASE_DIR)

from src.api_client import DEEPSEEK_API_KEY
from src.jd_analyzer import analyze_jd, compare_jds, analyze_gap
from src.resume import optimize_resume
from src.interview import (
    init_interview_state, start_interview, submit_answer,
    reset_interview, INTERVIEW_MAX_ROUNDS,
)
from src.prompts import ROLE_DATABASE
from src.roadmap import generate_roadmap
from src.stats import (
    init_stats_state, record_jd_analysis, record_resume_optimization,
    record_roadmap, record_interview, record_jd_compare,
    get_score_trend, get_latest_score, get_score_delta,
    get_jd_history, get_interview_history, clear_all_history,
)

# ═══════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="AI 求职助手 | JobCoach",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════
# 全局 CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
/* ── 字体导入 ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── 全局重置 ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1e293b;
}
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.5rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

/* ── 主容器 padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 0 !important;
}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.25rem !important;
}

/* ── 按钮 ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
    transition: all 0.15s ease;
    border: none !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
}
.stButton > button[kind="secondary"] {
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #e2e8f0 !important;
}

/* ── 卡片容器 ── */
.glass-card {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: 16px;
    padding: 28px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
}

/* ── 功能选择卡片 ── */
.feature-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 20px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    height: 100%;
}
.feature-card:hover {
    border-color: #a5b4fc;
    box-shadow: 0 4px 24px rgba(99,102,241,0.1);
    transform: translateY(-2px);
}
.feature-card .emoji {
    font-size: 40px;
    display: block;
    margin-bottom: 12px;
}
.feature-card .title {
    font-size: 17px;
    font-weight: 650;
    color: #1e293b;
    margin-bottom: 6px;
}
.feature-card .desc {
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.5;
}
.feature-card .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin-bottom: 10px;
}
.badge-deep    { background: #ede9fe; color: #7c3aed; }
.badge-light   { background: #f1f5f9; color: #64748b; }

/* ── 面试对话框 ── */
.chat-wrap { margin: 8px 0; }
.chat-msg {
    display: flex;
    gap: 12px;
    margin: 16px 0;
    animation: fadeInUp 0.3s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-msg.assistant { justify-content: flex-start; }
.chat-msg.user      { justify-content: flex-end; }
.chat-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.chat-avatar.ai     { background: #ede9fe; }
.chat-avatar.human  { background: #dbeafe; }
.chat-bubble {
    max-width: 70%;
    padding: 14px 18px;
    border-radius: 18px;
    line-height: 1.65;
    font-size: 15px;
}
.chat-bubble.ai {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-bottom-left-radius: 4px;
    color: #334155;
}
.chat-bubble.human {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    border-bottom-right-radius: 4px;
}

/* ── 面试进度 ── */
.round-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #f1f5f9;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 14px; font-weight: 500;
    margin-bottom: 20px;
}
.round-pill .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%     { opacity: 0.5; transform: scale(1.3); }
}

/* ── 面试岗位选择卡 ── */
.role-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 20px 20px;
    text-align: center;
    transition: all 0.2s;
}
.role-card:hover {
    border-color: #c4b5fd;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
}

/* ── 面试评估报告 ── */
.eval-header {
    background: linear-gradient(135deg, #faf5ff 0%, #ede9fe 100%);
    border-radius: 20px;
    padding: 32px 28px;
    margin-bottom: 28px;
    text-align: center;
}
.eval-header .score-ring {
    display: inline-block;
    width: 80px; height: 80px;
    line-height: 80px;
    border-radius: 50%;
    background: conic-gradient(#6366f1 0deg, #6366f1 252deg, #e2e8f0 252deg);
    text-align: center;
    font-size: 28px; font-weight: 800;
    color: #6366f1;
}

/* ── 标签 ── */
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px 4px 2px 0;
}
.tag-purple { background: #ede9fe; color: #7c3aed; }
.tag-blue   { background: #dbeafe; color: #2563eb; }
.tag-green  { background: #dcfce7; color: #16a34a; }
.tag-amber  { background: #fef3c7; color: #d97706; }
.tag-red    { background: #fee2e2; color: #dc2626; }

/* ── 输入区 ── */
textarea {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 14px !important;
}
textarea:focus {
    border-color: #a5b4fc !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── 分割线 ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 24px 0;
}

/* ── 底部 ── */
.footer-bar {
    text-align: center;
    padding: 40px 0 20px;
    color: #94a3b8;
    font-size: 12px;
}

/* ── 信息提示框 ── */
.info-tip {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
}

/* ── 折叠面板 ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
}

/* ── 隐藏 Streamlit 默认元素 ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
/* 去掉 text_input / text_area 的 "Press Enter to apply/reply" */
small { display: none !important; }
[data-baseweb="input"] + div { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── 强制清楚侧边栏折叠状态 ──
st.markdown("""
<script>
localStorage.removeItem("stSidebarState");
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# 初始化 Session State
# ═══════════════════════════════════════════
init_interview_state()
init_stats_state()
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ═══════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:28px;">📄</span>
        <div>
            <div style="font-size:18px;font-weight:700;color:#1e293b;">JobCoach</div>
            <div style="font-size:12px;color:#94a3b8;">AI 求职助手</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 导航项
    nav_items = [
        ("🏠", "首页", "home"),
        ("📋", "JD 分析", "jd"),
        ("✂️", "简历优化", "resume"),
        ("🗺️", "学习路线", "roadmap"),
        ("💬", "AI 模拟面试", "interview"),
        ("👤", "我的", "profile"),
    ]

    for emoji, label, key in nav_items:
        active = st.session_state.current_page == key
        btn_label = f"{emoji}  {label}"
        btn_type = "primary" if active else "secondary"
        if st.button(btn_label, key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state.current_page = key
            st.rerun()



# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════
def _guess_role(result: str, jd_text: str) -> str:
    """从 JD 分析和原始文本中猜测岗位名称——JD 标题优先，AI 分析辅助"""
    jd_lower = jd_text[:500].lower()  # JD 前半段通常有岗位名
    combined = (result + jd_text).lower()

    # 优先从 JD 原文标题匹配（不依赖 AI 结果）
    keywords = {
        "HR（招聘方向）": ["hr实习", "人力资源", "hrbp", "招聘专员", "员工关系"],
        "AI产品经理": ["ai产品", "大模型", "ai native", "人工智能产品经理"],
        "UI/UX设计师": ["ui设计", "ux设计", "交互设计", "视觉设计"],
        "算法工程师": ["算法工程师", "机器学习工程师", "深度学习", "nlp工程师"],
        "前端开发工程师": ["前端开发", "web前端", "前端工程师"],
        "后端开发工程师": ["后端开发", "服务端", "后端工程师", "java开发"],
        "数据分析师": ["数据分析师", "数据产品经理", "bi工程师"],
        "产品经理（通用）": ["产品经理", "产品实习", "产品需求", "prd"],
        "用户运营": ["用户运营", "用户增长", "社群运营"],
    }
    # 先看 JD 标题
    for role, keys in keywords.items():
        if any(k in jd_lower for k in keys):
            return role
    # 再看 AI 分析结果
    for role, keys in keywords.items():
        if any(k in result.lower() for k in keys):
            return role
    return "产品经理（通用）"


def _show_flow_nav(current_step: int):
    """
    页面底部流程导航——只留下一步
    current_step: 1=JD分析, 2=简历优化, 3=差距分析, 4=学习路线, 5=模拟面试
    """
    FLOW = [
        (1, "JD 分析", "jd"),
        (2, "简历优化", "resume"),
        (3, "学习路线", "roadmap"),
        (4, "模拟面试", "interview"),
    ]

    if current_step < 4:
        next_label, next_page = FLOW[current_step][1], FLOW[current_step][2]
        st.divider()
        _, bc, _ = st.columns([2, 1, 2])
        with bc:
            if st.button("下一步：" + next_label + " → ", key="flow_next_" + str(current_step),
                         use_container_width=True, type="primary"):
                # 从学习路线 → 模拟面试时，带上 JD 上下文
                if next_page == "interview":
                    jd_text = st.session_state.get("jd_last_input", "")
                    if jd_text.strip():
                        st.session_state.interview_jd_context = jd_text
                        # 自动识别岗位
                        result = st.session_state.get("jd_last_result", "")
                        auto_role = _guess_role(result, jd_text)
                        st.session_state.interview_selected_role = auto_role
                        st.session_state.interview_confirm_mode = True
                    else:
                        st.session_state.interview_confirm_mode = False
                        st.session_state.interview_selected_role = None
                        st.session_state.interview_jd_context = ""
                st.session_state.current_page = next_page
                st.rerun()


def _show_resume_input_compact():
    """差距分析入口——输入文字或上传 PDF"""
    st.markdown("""
    <div style="margin-top:12px;font-size:12px;color:#a8a29e;">贴简历文字或上传 PDF，直接跟这份 JD 做差距分析</div>
    """, unsafe_allow_html=True)
    inline_resume = st.text_area(
        "简历文字",
        placeholder="粘贴你的简历…",
        height=100, key="jd_inline_resume", label_visibility="collapsed",
    )
    if inline_resume.strip():
        st.session_state.user_resume = inline_resume
        st.rerun()

    # PDF 上传——CSS 拉到输入框内部右下角
    st.markdown("""
    <style>
    #jd-pdf-upload-area { margin-top: -34px; position: relative; z-index: 5; display: flex; justify-content: flex-end; padding-right: 8px; }
    #jd-pdf-upload-area button { font-size: 11px !important; padding: 2px 10px !important; border-radius: 6px !important; background: #f5f5f4 !important; border: 1px solid #e7e5e4 !important; color: #78716c !important; }
    #jd-pdf-upload-area section { justify-content: flex-end; }
    </style>
    <div id="jd-pdf-upload-area">
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF", type=["pdf"], key="jd_pdf_upload", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    if uploaded is not None:
            _parse_pdf(uploaded)


def _parse_pdf(file):
    """解析 PDF 文件为文本并存入 user_resume"""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            st.session_state.user_resume = text.strip()
            st.success(f"已从 PDF 提取 {len(text)} 字")
            st.rerun()
        else:
            st.warning("PDF 中没有可提取的文字（可能是扫描版图片）")
    except Exception as e:
        st.warning(f"PDF 解析失败：{e}")


def _show_single_result(result: str, jd_text: str):
    """展示单份 JD 分析结果 + 操作入口"""
    tags = []
    for kw in ["SQL", "Python", "数据分析", "产品设计", "用户调研", "PRD", "大模型", "NLP", "原型设计", "项目管理", "沟通", "逻辑", "竞品分析", "Excel"]:
        if kw.lower() in result.lower():
            tags.append(kw)
    tags = tags[:6]

    st.divider()
    st.markdown("### 分析结果")

    if tags:
        tag_html = "".join(f'<span class="tag tag-purple" style="margin:2px 4px 2px 0;">{t}</span>' for t in tags)
        st.markdown(f'<div style="padding:0 0 12px;">{tag_html}</div>', unsafe_allow_html=True)

    st.markdown(result)

    st.divider()
    guessed_role = _guess_role(result, jd_text)
    c_act1, c_act2 = st.columns(2)
    with c_act1:
        st.markdown("""
        <div style="padding:16px 20px;background:#fafaf9;border:1px solid #e7e5e4;border-radius:12px;height:100%;">
            <div style="font-weight:550;font-size:14px;color:#1c1917;margin-bottom:4px;">学习路线</div>
            <div style="font-size:12px;color:#a8a29e;margin-bottom:12px;">基于分析结果生成准备计划</div>
        """, unsafe_allow_html=True)
        if st.button("学习路线规划 →", key="jd_to_roadmap", type="secondary", use_container_width=True):
            st.session_state._roadmap_role = guessed_role
            st.session_state._roadmap_summary = result
            st.session_state.current_page = "roadmap"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c_act2:
        st.markdown(f"""
        <div style="padding:16px 20px;background:#fafaf9;border:1px solid #e7e5e4;border-radius:12px;height:100%;">
            <div style="font-weight:550;font-size:14px;color:#1c1917;margin-bottom:4px;">模拟面试</div>
            <div style="font-size:12px;color:#a8a29e;margin-bottom:12px;">检验你对这份 JD 的理解</div>
        """, unsafe_allow_html=True)
        if st.button(f"模拟面试：{guessed_role} →", key="jd_to_interview", type="primary", use_container_width=True):
            st.session_state.interview_selected_role = guessed_role
            st.session_state.interview_confirm_mode = True
            st.session_state.interview_jd_context = jd_text
            st.session_state.current_page = "interview"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── JD-简历差距分析 ──
    user_resume = st.session_state.get("user_resume", "")
    if user_resume.strip():
        _show_gap_section(result, user_resume)
    else:
        _show_resume_input_compact()

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns([3, 1])
    with ca:
        if st.button("重新分析", use_container_width=True, key="jd_redo"):
            st.session_state.jd_last_result = None
            st.session_state.jd_last_input = ""
            st.rerun()
    with cb:
        st.download_button("导出",
            data=f"# JD 分析报告\n\n{result}\n\n---\n原始JD：\n{jd_text}",
            file_name="JD分析报告.md", mime="text/markdown", use_container_width=True)


def _show_gap_section(jd_analysis: str, resume: str):
    """在 JD 分析结果下方展示差距分析区域"""
    if "gap_result" not in st.session_state:
        st.session_state.gap_result = None
    if "gap_running" not in st.session_state:
        st.session_state.gap_running = False

    st.divider()
    st.markdown("### 简历差距分析")
    st.markdown('<div style="font-size:12px;color:#a8a29e;margin-bottom:8px;">对照你的简历，看跟这份 JD 差在哪</div>', unsafe_allow_html=True)

    if not st.session_state.gap_running and not st.session_state.gap_result:
        if st.button("开始差距分析", key="run_gap", type="primary", use_container_width=True):
            st.session_state.gap_running = True
            st.rerun()

    if st.session_state.gap_running:
        with st.spinner("正在对比简历和 JD …"):
            result = analyze_gap(jd_analysis, resume)
        st.session_state.gap_result = result
        st.session_state.gap_running = False
        st.rerun()

    if st.session_state.gap_result:
        st.markdown(st.session_state.gap_result)
        if st.button("重新分析差距", key="gap_redo", use_container_width=True):
            st.session_state.gap_result = None
            st.rerun()


def _show_cached_result():
    """展示已缓存的 JD 分析结果"""
    result = st.session_state.jd_last_result
    jd_text = st.session_state.jd_last_input
    st.divider()
    st.markdown("### 分析结果")
    st.markdown(result)

    guessed_role = _guess_role(result, jd_text)
    c_act1, c_act2 = st.columns(2)
    with c_act1:
        st.markdown("""
        <div style="padding:16px 20px;background:#fafaf9;border:1px solid #e7e5e4;border-radius:12px;height:100%;margin-top:16px;">
            <div style="font-weight:550;font-size:14px;color:#1c1917;margin-bottom:4px;">学习路线</div>
            <div style="font-size:12px;color:#a8a29e;margin-bottom:12px;">基于分析结果生成准备计划</div>
        """, unsafe_allow_html=True)
        if st.button("学习路线规划 →", key="jd_to_roadmap2", type="secondary", use_container_width=True):
            st.session_state._roadmap_role = guessed_role
            st.session_state._roadmap_summary = result
            st.session_state.current_page = "roadmap"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c_act2:
        st.markdown(f"""
        <div style="padding:16px 20px;background:#fafaf9;border:1px solid #e7e5e4;border-radius:12px;height:100%;margin-top:16px;">
            <div style="font-weight:550;font-size:14px;color:#1c1917;margin-bottom:4px;">模拟面试</div>
            <div style="font-size:12px;color:#a8a29e;margin-bottom:12px;">检验你对这份 JD 的理解</div>
        """, unsafe_allow_html=True)
        if st.button(f"模拟面试：{guessed_role} →", key="jd_to_interview2", type="primary", use_container_width=True):
            st.session_state.interview_selected_role = guessed_role
            st.session_state.interview_confirm_mode = True
            st.session_state.interview_jd_context = jd_text
            st.session_state.current_page = "interview"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns([3, 1])
    with ca:
        if st.button("重新分析", use_container_width=True, key="jd_redo2"):
            st.session_state.jd_last_result = None
            st.session_state.jd_last_input = ""
            st.rerun()
    with cb:
        st.download_button("导出",
            data=f"# JD 分析报告\n\n{result}\n\n---\n原始JD：\n{jd_text}",
            file_name="JD分析报告.md", mime="text/markdown", use_container_width=True)

    # gap analysis
    user_resume = st.session_state.get("user_resume", "")
    if user_resume.strip():
        _show_gap_section(result, user_resume)
    else:
        _show_resume_input_compact()


# ═══════════════════════════════════════════
# ── 页面：首页 ──
# ═══════════════════════════════════════════
if st.session_state.current_page == "home":
    # hero —— 干净，无大 emoji
    st.markdown("""
    <div style="text-align:center;padding:48px 20px 40px;">
        <h1 style="font-size:2rem!important;font-weight:700!important;margin-bottom:6px;color:#0f172a;letter-spacing:-0.02em;">
            为你的求职准备<span style="color:#334155;border-bottom:2px solid #d4d4d8;padding-bottom:2px;">添砖加瓦</span>
        </h1>
        <p style="font-size:15px;color:#78716c;max-width:460px;margin:12px auto 0;line-height:1.7;">
            专注求职这件事
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 全流程指导 ──
    st.markdown("""
    <div style="margin:0 auto 28px;max-width:720px;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:4px;">
            <div style="text-align:center;flex:1;">
                <div style="width:28px;height:28px;border-radius:50%;background:#f5f5f4;border:1px solid #d6d3d1;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:#78716c;margin-bottom:6px;">1</div>
                <div style="font-size:12px;font-weight:550;color:#1c1917;">分析 JD</div>
                <div style="font-size:10px;color:#a8a29e;margin-top:1px;">拆解岗位要求</div>
            </div>
            <div style="color:#d6d3d1;font-size:12px;padding-top:8px;">→</div>
            <div style="text-align:center;flex:1;">
                <div style="width:28px;height:28px;border-radius:50%;background:#f5f5f4;border:1px solid #d6d3d1;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:#78716c;margin-bottom:6px;">2</div>
                <div style="font-size:12px;font-weight:550;color:#1c1917;">上传简历</div>
                <div style="font-size:10px;color:#a8a29e;margin-top:1px;">文字或 PDF</div>
            </div>
            <div style="color:#d6d3d1;font-size:12px;padding-top:8px;">→</div>
            <div style="text-align:center;flex:1;">
                <div style="width:28px;height:28px;border-radius:50%;background:#f5f5f4;border:1px solid #d6d3d1;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:#78716c;margin-bottom:6px;">3</div>
                <div style="font-size:12px;font-weight:550;color:#1c1917;">差距分析</div>
                <div style="font-size:10px;color:#a8a29e;margin-top:1px;">简历 vs JD</div>
            </div>
            <div style="color:#d6d3d1;font-size:12px;padding-top:8px;">→</div>
            <div style="text-align:center;flex:1;">
                <div style="width:28px;height:28px;border-radius:50%;background:#f5f5f4;border:1px solid #d6d3d1;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:#78716c;margin-bottom:6px;">4</div>
                <div style="font-size:12px;font-weight:550;color:#1c1917;">学习路线</div>
                <div style="font-size:10px;color:#a8a29e;margin-top:1px;">制定准备计划</div>
            </div>
            <div style="color:#d6d3d1;font-size:12px;padding-top:8px;">→</div>
            <div style="text-align:center;flex:1;">
                <div style="width:28px;height:28px;border-radius:50%;background:#f5f5f4;border:1px solid #d6d3d1;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:#78716c;margin-bottom:6px;">5</div>
                <div style="font-size:12px;font-weight:550;color:#1c1917;">模拟面试</div>
                <div style="font-size:10px;color:#a8a29e;margin-top:1px;">检验准备成果</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 四个圆形入口 ──
    c1, c2, c3, c4 = st.columns(4)
    entries = [
        ("📋", "JD 分析", "拆解岗位要求", "jd"),
        ("📝", "简历优化", "STAR 改写建议", "resume"),
        ("🗺️", "学习路线", "按周规划准备", "roadmap"),
        ("💬", "模拟面试", "6 轮动态追问", "interview"),
    ]
    for col, (icon, title, desc, page) in zip([c1, c2, c3, c4], entries):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:16px 8px;">
                <div style="
                    width:64px;height:64px;border-radius:50%;
                    background:#f5f5f4;border:1px dashed #d6d3d1;
                    display:inline-flex;align-items:center;justify-content:center;
                    font-size:26px;margin-bottom:10px;
                    transition:border-color 0.2s;
                ">{icon}</div>
                <div style="font-weight:600;font-size:15px;color:#1c1917;margin-bottom:2px;">{title}</div>
                <div style="font-size:12px;color:#a8a29e;margin-bottom:10px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"{title} →", key=f"go_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()


# ═══════════════════════════════════════════
# ── 页面：JD 分析 ──
# ═══════════════════════════════════════════
elif st.session_state.current_page == "jd":
    if st.button("← 首页", key="jd_home"):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h1>JD 智能分析</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── 模式切换 ──
    if "jd_mode" not in st.session_state:
        st.session_state.jd_mode = "single"  # "single" | "compare"
    if "jd_last_result" not in st.session_state:
        st.session_state.jd_last_result = None
        st.session_state.jd_last_input = ""
    if "jd_compare_slots" not in st.session_state:
        st.session_state.jd_compare_slots = ["", "", ""]

    cm1, cm2 = st.columns(2)
    with cm1:
        if st.button("分析单份 JD", key="mode_single",
                     type="primary" if st.session_state.jd_mode == "single" else "secondary",
                     use_container_width=True):
            st.session_state.jd_mode = "single"
            st.rerun()
    with cm2:
        if st.button("横向对比多份 JD", key="mode_compare",
                     type="primary" if st.session_state.jd_mode == "compare" else "secondary",
                     use_container_width=True):
            st.session_state.jd_mode = "compare"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════
    # 模式 A：单份分析
    # ═══════════════════════════════
    if st.session_state.jd_mode == "single":
        if st.session_state.jd_last_result is None:
            st.markdown("")

        if "jd_clear_seed" not in st.session_state:
            st.session_state.jd_clear_seed = 0

        # 填入示例用 pending 机制
        pending = st.session_state.pop("_jd_pending", "") or st.session_state.get("jd_last_input", "")
        jd_text = st.text_area(
            "粘贴岗位 JD",
            placeholder="从招聘网站直接复制粘贴，支持任意格式…",
            height=240,
            label_visibility="collapsed",
            key=f"jd_box_{st.session_state.jd_clear_seed}",
            value=pending,
        )
        if jd_text.strip():
            st.session_state.jd_last_input = jd_text

        EXAMPLE_JD = """【岗位名称】HR实习生
【工作职责】
1. 协助招聘流程推进，包括简历筛选、面试安排与跟进
2. 参与员工入职培训及新员工融入计划的执行
3. 维护招聘渠道，整理并分析招聘数据
4. 支持员工关系、企业文化活动等日常HR事务
【任职要求】
1. 2026届及以上在校生，每周实习4天以上
2. 对人力资源工作有基本认知，人力资源/心理学/管理学相关专业优先
3. 熟练使用Excel等办公软件，具备数据整理能力
4. 优秀的沟通表达和亲和力，做事细致有条理
5. 有HR实习经验或校园招聘经历优先"""

        c_ct, c_cl = st.columns([20, 1])
        with c_ct: st.caption(f"{len(jd_text)} 字")
        with c_cl:
            if jd_text.strip() and st.button("✕", key="clear_jd", help="清空输入", use_container_width=True):
                st.session_state.jd_clear_seed += 1
                st.rerun()
        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("填入示例", key="fill_example_jd", use_container_width=True):
                st.session_state._jd_pending = EXAMPLE_JD
                st.session_state.jd_clear_seed += 1
                st.rerun()
        with btn_c2:
            analyze_btn = st.button(
                "开始分析", type="primary", use_container_width=True,
                disabled=not DEEPSEEK_API_KEY or not jd_text.strip(),
            )

        if analyze_btn and jd_text.strip():
            with st.spinner("正在分析…"):
                result = analyze_jd(jd_text)

            st.session_state.jd_last_result = result
            st.session_state.jd_last_input = jd_text
            record_jd_analysis(_guess_role(result, jd_text))
            st.session_state.gap_result = None
            st.session_state.gap_running = False
            _show_single_result(result, jd_text)

        elif st.session_state.jd_last_result:
            _show_cached_result()

    # ═══════════════════════════════════════
    # 模式 B：多份横向对比
    # ═══════════════════════════════════════
    else:
        st.markdown("""
        <div style="font-size:12px;color:#a8a29e;margin-bottom:12px;">
            粘贴 2–5 份相似岗位的 JD，AI 帮你提取共性要求、差异点和行业趋势
        </div>
        """, unsafe_allow_html=True)

        # 插槽数
        slot_count = st.radio("JD 数量", [2, 3, 4, 5], index=1, horizontal=True, label_visibility="collapsed")
        while len(st.session_state.jd_compare_slots) < slot_count:
            st.session_state.jd_compare_slots.append("")
        while len(st.session_state.jd_compare_slots) > slot_count:
            st.session_state.jd_compare_slots.pop()

        for i in range(slot_count):
            st.session_state.jd_compare_slots[i] = st.text_area(
                f"JD-{i+1}",
                value=st.session_state.jd_compare_slots[i],
                placeholder=f"粘贴第 {i+1} 份 JD…",
                height=130,
                key=f"jd_compare_{i}",
                label_visibility="collapsed",
            )

        filled_count = sum(1 for s in st.session_state.jd_compare_slots if s.strip())
        st.caption(f"已填 {filled_count}/{slot_count} 份")

        compare_btn = st.button(
            "开始对比", type="primary", use_container_width=True,
            disabled=not DEEPSEEK_API_KEY or filled_count < 2,
        )

        if "jd_compare_result" not in st.session_state:
            st.session_state.jd_compare_result = None

        if compare_btn:
            with st.spinner("正在横向对比…"):
                jd_list = [s for s in st.session_state.jd_compare_slots if s.strip()]
                result = compare_jds(jd_list)
                record_jd_compare(len(jd_list))
                st.session_state.jd_compare_result = result

            st.divider()
            st.markdown("### 横向对比结果")
            st.markdown(result)

            ca, cb = st.columns([3, 1])
            with ca:
                if st.button("重新对比", use_container_width=True, key="compare_redo"):
                    st.session_state.jd_compare_result = None
                    st.rerun()
            with cb:
                st.download_button("导出",
                    data=f"# JD 横向对比报告\n\n{result}",
                    file_name="JD横向对比报告.md", mime="text/markdown", use_container_width=True)

        elif st.session_state.jd_compare_result:
            st.divider()
            st.markdown("### 横向对比结果")
            st.markdown(st.session_state.jd_compare_result)
            ca, cb = st.columns([3, 1])
            with ca:
                if st.button("重新对比", use_container_width=True, key="compare_redo2"):
                    st.session_state.jd_compare_result = None
                    st.rerun()
            with cb:
                st.download_button("导出",
                    data=f"# JD 横向对比报告\n\n{st.session_state.jd_compare_result}",
                    file_name="JD横向对比报告.md", mime="text/markdown", use_container_width=True)

    if st.session_state.jd_mode == "single":
        _show_flow_nav(1)

# ═══════════════════════════════════════════
# ── 页面：简历优化 ──
# ═══════════════════════════════════════════
elif st.session_state.current_page == "resume":
    if st.button("← 首页", key="resume_home"):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1>简历优化</h1>
    </div>
    """, unsafe_allow_html=True)

    # 从 JD 分析页跳过来时，自动填入最后分析的 JD
    bridge_jd = st.session_state.get("jd_last_input", "")

    # 清除后重建 widget 的种子
    if "resume_clear_seed" not in st.session_state:
        st.session_state.resume_clear_seed = 0
    if "resume_target_clear_seed" not in st.session_state:
        st.session_state.resume_target_clear_seed = 0

    resume_text = st.text_area(
        "简历内容",
        placeholder="把你的简历文字粘贴在这里…",
        height=280, label_visibility="collapsed",
        key=f"resume_box_{st.session_state.resume_clear_seed}",
    )
    if resume_text.strip():
        st.session_state.user_resume = resume_text
        if "gap_result" in st.session_state:
            st.session_state.gap_result = None

    cr1, cr2 = st.columns([20, 1])
    with cr2:
        if resume_text.strip() and st.button("✕", key="clear_resume", help="清空简历", use_container_width=True):
            st.session_state.user_resume = ""
            st.session_state.resume_clear_seed += 1
            st.rerun()

    target_jd_text = st.text_area(
        "目标岗位（可选）",
        placeholder="留空也可以，AI 会针对简历本身给出优化建议。\n想看到匹配度分析的话，就把目标 JD 也贴进来。",
        height=160, label_visibility="collapsed",
        key=f"resume_target_box_{st.session_state.resume_target_clear_seed}",
    )
    if bridge_jd and not target_jd_text:
        st.caption("已自动填入刚才分析的 JD")
    if target_jd_text.strip():
        st.session_state.resume_target_jd = target_jd_text

    ct1, ct2 = st.columns([20, 1])
    with ct2:
        if target_jd_text.strip() and st.button("✕", key="clear_target_jd", help="清空目标岗位", use_container_width=True):
            st.session_state.resume_target_jd = ""
            st.session_state.resume_target_clear_seed += 1
            st.rerun()

    optimize_btn = st.button(
        "优化我的简历", type="primary", use_container_width=True,
        disabled=not DEEPSEEK_API_KEY or not resume_text.strip(),
    )

    if optimize_btn and resume_text.strip():
        with st.spinner("正在分析…"):
            result = optimize_resume(resume_text, target_jd_text)
        record_resume_optimization()

        st.divider()
        st.markdown("### 优化报告")
        st.markdown(result)

        ca, cb = st.columns([3, 1])
        with ca:
            if st.button("重新优化", use_container_width=True, key="resume_redo"):
                st.rerun()
        with cb:
            st.download_button("导出",
                data=f"# 简历优化报告\n\n{result}\n\n---\n原始简历：\n{resume_text}",
                file_name="简历优化报告.md", mime="text/markdown", use_container_width=True)

    _show_flow_nav(2)

# ═══════════════════════════════════════════
# ── 页面：学习路线 ──
# ═══════════════════════════════════════════
elif st.session_state.current_page == "roadmap":
    if st.button("← 首页", key="roadmap_home"):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h1>学习路线规划</h1>
    </div>
    """, unsafe_allow_html=True)

    # 初始化状态
    if "roadmap_result" not in st.session_state:
        st.session_state.roadmap_result = None

    # JD 桥接（从 JD 分析页跳过来时自动填入）或手动输入
    bridge_role = st.session_state.pop("_roadmap_role", "")
    bridge_summary = st.session_state.pop("_roadmap_summary", "")

    st.markdown('<div style="font-size:13px;color:#a8a29e;margin-bottom:4px;">目标岗位</div>', unsafe_allow_html=True)
    rr1, rr2 = st.columns([20, 1])
    role_saved = st.session_state.get("roadmap_role_input", "")
    with rr1:
        role = st.text_input(
            "目标岗位",
            value=bridge_role if bridge_role else role_saved,
            placeholder="输入你想准备的岗位，如：AI产品经理、前端开发…",
            label_visibility="collapsed",
        )
    st.session_state.roadmap_role_input = role
    with rr2:
        if role.strip() and st.button("✕", key="clear_roadmap_role", help="清空岗位", use_container_width=True):
            st.session_state.roadmap_role_input = ""
            st.rerun()

    st.markdown('<div style="font-size:13px;color:#a8a29e;margin:16px 0 4px;">JD 分析摘要（选填）</div>', unsafe_allow_html=True)
    st.caption("从 JD 分析页跳过来会自动填入，留空 AI 会根据岗位名直接生成")
    jd_saved = st.session_state.get("roadmap_jd_input", "")
    jd_summary = st.text_area(
        "JD 分析摘要",
        value=bridge_summary if bridge_summary else jd_saved,
        placeholder="可以把 JD 分析结果粘贴过来，让学习路线更有针对性…",
        height=100,
        label_visibility="collapsed",
    )
    st.session_state.roadmap_jd_input = jd_summary
    rs1, rs2 = st.columns([20, 1])
    with rs2:
        if jd_summary.strip() and st.button("✕", key="clear_roadmap_jd", help="清空摘要", use_container_width=True):
            st.session_state.roadmap_jd_input = ""
            st.rerun()

    st.markdown('<div style="font-size:13px;color:#a8a29e;margin:16px 0 4px;">学习周期</div>', unsafe_allow_html=True)
    weeks = st.selectbox("学习周期", [2, 3, 4, 6, 8], index=1, label_visibility="collapsed",
                         format_func=lambda w: f"{w} 周" + ("（快速冲刺）" if w == 2 else "（推荐）" if w == 3 else "（扎实准备）" if w == 4 else "（深度提升）" if w == 6 else "（系统转型）"))

    gen_btn = st.button("生成学习路线", type="primary", use_container_width=True,
                        disabled=not DEEPSEEK_API_KEY or not role.strip())

    if gen_btn and role.strip():
        with st.spinner("正在规划…"):
            result = generate_roadmap(role, jd_summary, weeks)
        st.session_state.roadmap_result = result
        st.session_state._roadmap_weeks = weeks
        record_roadmap()

    if st.session_state.roadmap_result:
        st.divider()
        st.markdown("### 学习路线")
        st.markdown(st.session_state.roadmap_result)

        ca, cb = st.columns([3, 1])
        with ca:
            if st.button("重新生成", use_container_width=True, key="roadmap_redo"):
                st.session_state.roadmap_result = None
                st.rerun()
        with cb:
            st.download_button("导出",
                data=f"# 学习路线规划\n\n目标岗位：{role}\n\n{st.session_state.roadmap_result}",
                file_name="学习路线规划.md", mime="text/markdown", use_container_width=True)

    _show_flow_nav(3)

# ═══════════════════════════════════════════
# ── 页面：AI 模拟面试 ──
# ═══════════════════════════════════════════
elif st.session_state.current_page == "interview":
    if st.button("← 首页", key="interview_home"):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1>AI 模拟面试</h1>
        <p style="color:#64748b;font-size:15px;margin-top:-4px;">
            真实 PM 实习面试体验 —— 动态追问、多维度评估、详细反馈
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 状态 0：搜索 / 选择岗位，未开始 ──
    if not st.session_state.interview_active and st.session_state.interview_evaluation is None:

        # 初始化 session 状态
        if "interview_search_query" not in st.session_state:
            st.session_state.interview_search_query = ""
        if "interview_selected_role" not in st.session_state:
            st.session_state.interview_selected_role = None
        if "interview_confirm_mode" not in st.session_state:
            st.session_state.interview_confirm_mode = False
        if "interview_jd_context" not in st.session_state:
            st.session_state.interview_jd_context = ""

        # 确认页模式
        if st.session_state.interview_confirm_mode and st.session_state.interview_selected_role:
            role = st.session_state.interview_selected_role
            cat = next((r["category"] for r in ROLE_DATABASE if r["name"] == role), "")
            jd_context = st.session_state.get("interview_jd_context", "")
            if st.button("← 返回", key="back_to_search"):
                st.session_state.interview_confirm_mode = False
                st.session_state.interview_selected_role = None
                st.session_state.interview_jd_context = ""
                st.rerun()

            st.markdown(f"""
            <div style="text-align:center;padding:40px 20px 24px;">
                <div style="
                    width:72px;height:72px;border-radius:50%;
                    background:#f5f5f4;border:1px dashed #d6d3d1;
                    display:inline-flex;align-items:center;justify-content:center;
                    font-size:30px;margin-bottom:16px;
                ">💬</div>
                <h2 style="font-weight:700;color:#0f172a;margin-bottom:4px;">{role}</h2>
                <p style="font-size:14px;color:#a8a29e;">{cat} · 实习面试模拟</p>
            </div>
            """, unsafe_allow_html=True)

            # ── JD 上下文提示 ──
            if jd_context.strip():
                with st.expander("📋 面试将基于这份 JD 提问", expanded=False):
                    st.caption(f"JD 内容（{len(jd_context)} 字）")
                    st.markdown(jd_context[:1200] + ("…" if len(jd_context) > 1200 else ""))
                    st.caption("AI 面试官会围绕这份 JD 中的技能和职责进行提问。")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("""
                <div style="text-align:center;padding:12px;">
                    <div style="font-weight:600;font-size:14px;color:#1c1917;">6 轮问答</div>
                    <div style="font-size:12px;color:#a8a29e;">动态追问，模拟真实面试节奏</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div style="text-align:center;padding:12px;">
                    <div style="font-weight:600;font-size:14px;color:#1c1917;">多维评估</div>
                    <div style="font-size:12px;color:#a8a29e;">专业基础·逻辑·执行·沟通·自驱</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown("""
                <div style="text-align:center;padding:12px;">
                    <div style="font-weight:600;font-size:14px;color:#1c1917;">详细报告</div>
                    <div style="font-size:12px;color:#a8a29e;">结构化反馈 + 两周改进路线</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
            with c_btn2:
                if st.button("开始面试", key="confirm_start", use_container_width=True, type="primary"):
                    start_interview(role, jd_context=jd_context)
                    st.session_state.interview_confirm_mode = False
                    st.session_state.interview_selected_role = None
                    st.session_state.interview_jd_context = ""
                    st.rerun()

        else:
            # ── 面试进步追踪（有历史记录时显示）──
            scores = get_score_trend()
            if len(scores) >= 1:
                with st.expander("📈 面试进步追踪", expanded=len(scores) >= 2):
                    # 趋势图
                    if len(scores) >= 2:
                        import pandas as pd
                        df = pd.DataFrame(scores)
                        df["label"] = df.apply(lambda r: f"#{r['index']} {r['role']}", axis=1)
                        st.line_chart(df.set_index("label")["score"], use_container_width=True, height=200)

                    # 历史列表
                    st.markdown('<div style="font-size:11px;color:#a8a29e;margin:8px 0 4px;">面试历史</div>', unsafe_allow_html=True)
                    for s in reversed(scores[-8:]):  # 最近 8 次
                        color = "#1c1917" if s["score"] is not None and s["score"] >= 6 else "#dc2626"
                        score_str = str(s["score"]) if s["score"] is not None else "—"
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;padding:6px 0;font-size:13px;border-bottom:1px solid #f5f5f4;">
                            <span style="color:#1c1917;">#{s['index']} {s['role']}</span>
                            <span style="display:flex;gap:16px;">
                                <span style="color:#a8a29e;">{s['date']}</span>
                                <span style="font-weight:600;color:{color};">{score_str}/10</span>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

            # ── 标题 ──
            st.markdown("""
            <div style="text-align:center;margin-bottom:20px;">
                <span style="font-weight:600;font-size:17px;color:#0f172a;">选择你要模拟的面试岗位</span>
            </div>
            """, unsafe_allow_html=True)

            # ── 搜索框 + 右端图标 ──
            sc1, sc2 = st.columns([20, 1], gap="small")
            with sc1:
                query = st.text_input(
                    "搜索岗位",
                    value=st.session_state.interview_search_query,
                    placeholder="输入岗位名称，如：产品经理、前端开发、数据分析…",
                    key="role_search",
                    label_visibility="collapsed",
                )
            with sc2:
                if query.strip():
                    # 有内容 → 显示清除 ×
                    if st.button("✕", key="clear_search", help="清除搜索内容", use_container_width=True):
                        st.session_state.interview_search_query = ""
                        st.rerun()
                else:
                    # 无内容 → 显示放大镜
                    st.markdown("""
                    <div style="display:flex;align-items:center;justify-content:center;height:40px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a8a29e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state.interview_search_query = query.strip()

            # ── 搜索结果下拉面板 ──
            q = query.lower()
            if q:
                filtered = [r for r in ROLE_DATABASE if q in r["name"].lower() or q in r["category"].lower()]

                # 下拉面板容器
                st.markdown("""
                <div style="
                    max-width:520px;margin:0 auto 8px;
                    border:1px solid #e7e5e4;border-radius:12px;
                    background:#fff;padding:4px 0;
                    box-shadow:0 4px 16px rgba(0,0,0,0.04);
                ">
                """, unsafe_allow_html=True)

                if filtered:
                    st.markdown(f'<div style="font-size:11px;color:#a8a29e;padding:8px 16px 4px;">找到 {len(filtered)} 个岗位</div>', unsafe_allow_html=True)
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for r in filtered:
                        grouped[r["category"]].append(r["name"])

                    for cat, roles in grouped.items():
                        st.markdown(f'<div style="font-size:11px;color:#a8a29e;padding:8px 16px 4px;text-transform:uppercase;letter-spacing:0.06em;">{cat}</div>', unsafe_allow_html=True)
                        # 按钮在容器内，每行4个
                        for i in range(0, len(roles), 4):
                            chunk = roles[i:i+4]
                            btn_cols = st.columns(4)
                            for j, role_name in enumerate(chunk):
                                with btn_cols[j]:
                                    if st.button(role_name, key=f"role_drop_{role_name}", use_container_width=True):
                                        st.session_state.interview_selected_role = role_name
                                        st.session_state.interview_confirm_mode = True
                                        st.rerun()
                else:
                    st.markdown("""
                    <div style="text-align:center;padding:24px;color:#a8a29e;font-size:13px;">
                        没有匹配的岗位，换个关键词试试
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # ── 三个示例横条（始终保留）──
            st.markdown("""
            <div style="text-align:center;margin:16px 0 8px;">
                <span style="font-size:12px;color:#d6d3d1;">示例岗位方向</span>
            </div>
            """, unsafe_allow_html=True)

            examples = [
                ("产品类", "产品经理 / AI产品经理 / 策略产品经理 …", "拆解需求、设计功能、数据驱动决策",
                 '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#78716c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>'),
                ("技术类", "前端 / 后端 / 算法 / 测试 / 安全 …", "技术深度、项目经验、协作沟通",
                 '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#78716c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>'),
                ("综合类", "运营 / 市场 / 设计 / 金融 / 咨询 …", "业务理解、执行力、跨部门协同",
                 '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#78716c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'),
            ]

            for cat_label, roles_text, focus, icon_svg in examples:
                st.markdown(f"""
                <div style="
                    background:#fafaf9;border:1px solid #e7e5e4;border-radius:12px;
                    padding:14px 20px;margin:6px 0;display:flex;align-items:center;gap:14px;
                ">
                    <div style="
                        min-width:40px;height:40px;border-radius:10px;
                        background:#f5f5f4;border:1px dashed #d6d3d1;
                        display:flex;align-items:center;justify-content:center;
                    ">{icon_svg}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="font-weight:550;font-size:14px;color:#1c1917;margin-bottom:2px;">{roles_text}</div>
                        <div style="font-size:12px;color:#a8a29e;">{focus}</div>
                    </div>
                    <span style="font-size:12px;color:#d6d3d1;">→</span>
                </div>
                """, unsafe_allow_html=True)

    # ── 状态 1：面试中 ──
    elif st.session_state.interview_active:
        current_round = st.session_state.interview_round

        # 顶部状态栏
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
            <div class="round-pill">
                <span class="dot"></span>
                第 {current_round} / {INTERVIEW_MAX_ROUNDS} 轮
            </div>
            <span style="font-size:14px;color:#64748b;">
                {st.session_state.interview_role}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # 进度条
        st.progress(current_round / INTERVIEW_MAX_ROUNDS)

        # 对话历史
        for msg in st.session_state.interview_history:
            if msg["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-msg assistant">
                    <div class="chat-avatar ai">🎯</div>
                    <div class="chat-bubble ai">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-msg user">
                    <div class="chat-bubble human">{msg['content']}</div>
                    <div class="chat-avatar human">👤</div>
                </div>
                """, unsafe_allow_html=True)

        # 输入区（表单模式——Enter 直接提交）
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form(key=f"answer_form_{current_round}", clear_on_submit=True):
            answer = st.text_area(
                "输入你的回答",
                placeholder="像在真实面试中那样回答… Shift+Enter 换行，Enter 直接提交",
                height=110,
                label_visibility="collapsed",
            )

            fc1, fc2, fc3 = st.columns([2.5, 1, 1])
            with fc1:
                label = "📤 提交并查看评估" if current_round >= INTERVIEW_MAX_ROUNDS else "📤 提交回答（Enter）"
                submitted = st.form_submit_button(label, type="primary", use_container_width=True)
            with fc2:
                pass
            with fc3:
                pass

        if submitted:
            if not answer.strip():
                st.warning("请输入你的回答")
            else:
                submit_answer(answer)

        # 提前结束放在 form 外面
        _, ec, _ = st.columns([2.5, 1, 1])
        with ec:
            if st.button("🛑 提前结束", use_container_width=True, key="end_early",
                         help="基于已有对话生成评估报告"):
                st.session_state.interview_history.append(
                    {"role": "user", "content": "(提前结束面试)"}
                )
                with st.spinner("正在生成评估报告…"):
                    from src.interview import _generate_evaluation
                    st.session_state.interview_evaluation = _generate_evaluation()
                st.session_state.interview_completed = True
                st.session_state.interview_active = False
                st.session_state.waiting_for_answer = False
                st.rerun()

    # ── 状态 2：面试完成，展示评估 ──
    elif st.session_state.interview_evaluation:
        # 首次展示评估时记录数据
        if st.session_state.get("interview_completed"):
            record_interview(st.session_state.interview_role or "未指定", st.session_state.interview_evaluation)
            st.session_state.interview_completed = False

        st.markdown("""
        <div style="text-align:center;padding:12px 0 4px;">
            <span style="font-size:48px;">🎉</span>
        </div>
        """, unsafe_allow_html=True)

        # 对话回顾
        with st.expander("📝 查看完整对话记录", expanded=False):
            for msg in st.session_state.interview_history:
                role = "🎯 面试官" if msg["role"] == "assistant" else "👤 你"
                bg = "#f8fafc" if msg["role"] == "assistant" else "#ede9fe"
                st.markdown(f"""
                <div style="background:{bg};padding:12px 16px;border-radius:10px;margin:6px 0;">
                    <span style="font-weight:600;font-size:13px;">{role}</span>
                    <div style="margin-top:4px;">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

        # 评估报告
        st.divider()
        st.markdown("## 📊 面试评估报告")
        st.markdown(f'<div class="glass-card">{st.session_state.interview_evaluation}</div>', unsafe_allow_html=True)

        # 底部操作
        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔄 再来一次面试", type="primary", use_container_width=True):
                reset_interview()
                st.rerun()
        with col2:
            st.download_button("📥 导出面试报告",
                data=f"# AI 模拟面试报告\n\n岗位：{st.session_state.interview_role}\n\n## 完整对话\n\n" +
                     "\n\n".join(
                         f"**{'面试官' if m['role']=='assistant' else '候选人'}**：{m['content']}"
                         for m in st.session_state.interview_history
                     ) +
                     f"\n\n---\n\n{st.session_state.interview_evaluation}",
                file_name="模拟面试报告.md", mime="text/markdown", use_container_width=True)

    if not st.session_state.interview_active:
        _show_flow_nav(4)

# ═══════════════════════════════════════════
# ── 页面：我的（个人主页）──
# ═══════════════════════════════════════════
elif st.session_state.current_page == "profile":
    if st.button("← 首页", key="profile_home"):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1>我的</h1>
    </div>
    """, unsafe_allow_html=True)

    jd_hist = get_jd_history()
    iv_hist = get_interview_history()

    # ── 历史分析记录 ──
    st.markdown("### 📋 历史分析")
    if jd_hist:
        for i, item in enumerate(jd_hist):
            with st.expander(f"{item['role']} · {item['date']}", expanded=(i == 0)):
                st.caption(f"JD 原文（{len(item.get('jd_full', ''))} 字）")
                st.markdown(item.get("jd_full", item.get("jd_snippet", "")).replace("\n", "\n\n")[:3000])
                st.caption("分析结果")
                st.markdown(item.get("result", "")[:3000])
    else:
        st.markdown('<div style="color:#a8a29e;font-size:14px;padding:24px 0;text-align:center;">暂无分析记录<br><span style="font-size:12px;">去 JD 分析页贴一份岗位描述就会出现在这里</span></div>', unsafe_allow_html=True)

    st.divider()

    # ── 面试记录 ──
    st.markdown("### 🎯 面试记录")
    if iv_hist:
        # 趋势图
        scores_valid = [h for h in iv_hist if h.get("score") is not None]
        if len(scores_valid) >= 2:
            import pandas as pd
            df_data = []
            for h in reversed(scores_valid):
                df_data.append({"面试": h["role"], "评分": h["score"]})
            df = pd.DataFrame(df_data)
            if len(df) >= 2:
                st.line_chart(df.set_index("面试"), use_container_width=True, height=180)

        for h in iv_hist:
            color = "#1c1917" if h.get("score") and h["score"] >= 6 else "#dc2626"
            score_str = str(h["score"]) if h.get("score") is not None else "—"
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;margin:4px 0;background:#fafaf9;border:1px solid #e7e5e4;border-radius:10px;">
                <span style="font-weight:550;color:#1c1917;">{h['role']}</span>
                <span style="display:flex;gap:20px;">
                    <span style="color:#a8a29e;font-size:13px;">{h['date']}</span>
                    <span style="font-weight:600;color:{color};">{score_str}/10</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

        # 导出
        st.markdown("<br>", unsafe_allow_html=True)
        export_md = "# 我的求职记录\n\n## 面试记录\n\n"
        for h in iv_hist:
            export_md += f"- **{h['role']}** · {h['date']} · 评分 {h.get('score', '—')}/10\n"
            export_md += f"  轮次: {h.get('rounds', '—')}\n\n"
            export_md += f"  {h.get('evaluation', '')}\n\n---\n\n"
        st.download_button("📥 导出面试记录", data=export_md, file_name="求职记录.md", mime="text/markdown", use_container_width=True)
    else:
        st.markdown('<div style="color:#a8a29e;font-size:14px;padding:24px 0;text-align:center;">暂无面试记录<br><span style="font-size:12px;">完成一次模拟面试后会自动记录下来</span></div>', unsafe_allow_html=True)

    # ── 清除数据 ──
    st.divider()
    with st.expander("⚙️ 数据管理", expanded=False):
        st.caption("所有数据存储在你本地浏览器和文件中，不会被上传。")
        if st.button("清除全部记录", type="secondary", use_container_width=True):
            clear_all_history()
            st.rerun()

# ═══════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════
st.markdown("""
<div class="footer-bar">
    JobCoach v2.1 · AI 求职助手
</div>
""", unsafe_allow_html=True)
