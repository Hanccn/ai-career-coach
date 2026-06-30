"""AI Career Copilot —— Streamlit Cloud 入口
实际代码在 xiangmuyi/ 子目录"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "xiangmuyi"))
os.chdir(os.path.join(os.path.dirname(__file__), "xiangmuyi"))
exec(open(os.path.join(os.path.dirname(__file__), "xiangmuyi", "app.py"), encoding="utf-8").read())
