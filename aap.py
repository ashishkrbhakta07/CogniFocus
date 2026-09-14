"""
AI STUDY DISTRACTION DETECTOR
Modern SaaS Streamlit Dashboard inspired by CogniAI Minimalist Design.
Turn study activity into actionable focus insights — instantly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

from focus import calculate_focus_score
from data_manager import (
    load_sessions, save_session, get_today_summary, 
    get_activity_breakdown, ensure_data_dir
)
from mlmodel import DistractionPredictor
from datagenerator import generate_sample_data

# Page configuration
st.set_page_config(
    page_title="CogniFocus | AI Study Distraction Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching the CogniAI Dribbble Design (Warm Ivory, Orange Accents, Charcoal Pills, Soft Shadows)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-page: #fbfaf8;
        --bg-surface: #ffffff;
        --accent-orange: #ff6a3d;
        --accent-orange-hover: #f05a2d;
        --accent-orange-light: #fff3ee;
        --accent-dark: #18181b;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --text-subtle: #94a3b8;
        --border-card: #f0eee6;
        --shadow-soft: 0 10px 30px rgba(0, 0, 0, 0.04);
        --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.03);
    }
    
    .stApp {
        background-color: #faf9f6;
        background-image: radial-gradient(at 10% 10%, rgba(255, 237, 226, 0.6) 0px, transparent 50%),
                          radial-gradient(at 90% 80%, rgba(254, 243, 199, 0.4) 0px, transparent 50%);
        color: #0f172a;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Remove extra top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    /* Top Navbar */
    .cogni-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        background: #ffffff;
        border: 1px solid #f0eee6;
        border-radius: 9999px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.03);
        margin-bottom: 28px;
    }
    .cogni-brand {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #18181b;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .cogni-brand span {
        color: #ff6a3d;
    }
    
    /* Hero Header */
    .hero-box {
        text-align: center;
        max-width: 820px;
        margin: 0 auto 36px auto;
        padding: 10px 0;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fff0eb;
        border: 1px solid #ffd8cc;
        color: #ff6a3d;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        color: #18181b;
        margin-bottom: 14px;
    }
    .hero-title span {
        color: #ff6a3d;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        line-height: 1.6;
        margin: 0 auto;
    }

    /* Metric Cards - CogniAI Style */
    .cogni-card {
        background: #ffffff;
        border: 1px solid #f0eee6;
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: var(--shadow-card);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .cogni-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.06);
    }
    .cogni-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .cogni-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .cogni-card-val {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #18181b;
        line-height: 1.1;
    }
    .cogni-card-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 8px;
    }

    /* Badges */
    .pill-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .pill-orange { background: #fff0eb; color: #ff6a3d; border: 1px solid #ffd8cc; }
    .pill-green { background: #ecfdf5; color: #10b981; border: 1px solid #a7f3d0; }
    .pill-red { background: #fef2f2; color: #ef4444; border: 1px solid #fecaca; }
    .pill-dark { background: #18181b; color: #ffffff; }

    /* Reality Alert Banner (CogniAI Warm Style) */
    .reality-banner-danger {
        background: #ffffff;
        border-left: 5px solid #ff6a3d;
        border-top: 1px solid #f0eee6;
        border-right: 1px solid #f0eee6;
        border-bottom: 1px solid #f0eee6;
        border-radius: 16px;
        padding: 16px 20px;
        color: #18181b;
        box-shadow: var(--shadow-card);
        margin: 20px 0;
        font-size: 0.95rem;
    }
    .reality-banner-danger strong {
        color: #ff6a3d;
    }

    /* Control Box */
    .control-box {
        background: #ffffff;
        border: 1px solid #f0eee6;
        border-radius: 20px;
        padding: 24px;
        box-shadow: var(--shadow-card);
        margin-bottom: 24px;
    }
    .control-box-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #18181b;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Diagnostic Box */
    .diag-box {
        background: #ffffff;
        border: 1px solid #f0eee6;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .diag-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .diag-content {
        font-size: 1.05rem;
        font-weight: 600;
        color: #18181b;
        line-height: 1.4;
    }

    /* Custom Streamlit widget overrides */
    div.stButton > button {
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        border: 1px solid #f0eee6 !important;
        background-color: #ffffff !important;
        color: #18181b !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: #ff6a3d !important;
        color: #ff6a3d !important;
        background-color: #fff9f6 !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #ff6a3d !important;
        border-color: #ff6a3d !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(255, 106, 61, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #f05a2d !important;
        color: #ffffff !important;
    }

    /* Slider accent */
    div[data-baseweb="slider"] {
        accent-color: #ff6a3d !important;
    }
    
    /* Radio navbar */
    div[role="radiogroup"] {
        background: #ffffff;
        padding: 6px;
        border-radius: 9999px;
        border: 1px solid #f0eee6;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- INITIALIZE CENTRAL REACTIVE STATE ----------------
if "study_min" not in st.session_state:
    st.session_state.study_min = 70
if "break_min" not in st.session_state:
    st.session_state.break_min = 20
if "phone_min" not in st.session_state:
    st.session_state.phone_min = 20
if "talk_min" not in st.session_state:
    st.session_state.talk_min = 10
if "social_min" not in st.session_state:
    st.session_state.social_min = 0
if "wander_min" not in st.session_state:
    st.session_state.wander_min = 0
if "subject" not in st.session_state:
    st.session_state.subject = "Mathematics (Calculus)"
if "nav_view" not in st.session_state:
    st.session_state.nav_view = "📊 Dashboard & Insights"

# Initialize Data & ML Model
ensure_data_dir()
df_sessions = load_sessions()

if df_sessions.empty:
    with st.spinner("Initializing study database with sample history..."):
        generate_sample_data(30)
        df_sessions = load_sessions()

predictor = DistractionPredictor()
predictor.train(df_sessions)


# ---------------- TOP NAVIGATION BAR (CogniAI Style) ----------------
top_nav_col1, top_nav_col2, top_nav_col3 = st.columns([1.2, 2.6, 1.2])

with top_nav_col1:
    st.markdown("""
    <div style="font-size: 1.45rem; font-weight: 800; color: #18181b; padding-top: 6px;">
        Cogni<span style="color: #ff6a3d;">Focus</span>
    </div>
    """, unsafe_allow_html=True)

with top_nav_col2:
    # Pill Navigation Buttons
    nav_cols = st.columns(4)
    with nav_cols[0]:
        if st.button("📊 Overview", use_container_width=True, type="primary" if st.session_state.nav_view == "📊 Dashboard & Insights" else "secondary"):
            st.session_state.nav_view = "📊 Dashboard & Insights"
            st.rerun()
    with nav_cols[1]:
        if st.button("⏱️ Live Tracker", use_container_width=True, type="primary" if st.session_state.nav_view == "⏱️ Live Session Tracker" else "secondary"):
            st.session_state.nav_view = "⏱️ Live Session Tracker"
            st.rerun()
    with nav_cols[2]:
        if st.button("🔮 ML Forecast", use_container_width=True, type="primary" if st.session_state.nav_view == "🔮 AI Risk Forecaster" else "secondary"):
            st.session_state.nav_view = "🔮 AI Risk Forecaster"
            st.rerun()
    with nav_cols[3]:
        if st.button("📜 History", use_container_width=True, type="primary" if st.session_state.nav_view == "📜 Session History & Data" else "secondary"):
            st.session_state.nav_view = "📜 Session History & Data"
            st.rerun()

with top_nav_col3:
    if st.button("🔄 Reset 30d Data", use_container_width=True):
        generate_sample_data(30)
        st.success("Refreshed sample history!")
        st.rerun()

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# ---------------- HERO HEADLINE SECTION (CogniAI Typography) ----------------
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">⚡ AI-POWERED FOCUS INTELLIGENCE</div>
    <div class="hero-title">
        Turn Study Activity into <span>Actionable Insights</span> — Instantly
    </div>
    <div class="hero-subtitle">
        Harness machine learning analytics to transform raw study sessions into clear focus scores. Understand your distractions, identify risky time windows, and study smarter.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- CALCULATE LIVE REACTIVE FOCUS METRICS ----------------
current_breakdown = {
    "studying": st.session_state.study_min,
    "phone": st.session_state.phone_min,
    "talking": st.session_state.talk_min,
    "break": st.session_state.break_min,
    "social_media": st.session_state.social_min,
    "wandering": st.session_state.wander_min
}
live_result = calculate_focus_score(current_breakdown)

# Hourly Risk from Predictor based on current session
hourly_risk = predictor.predict_hourly_trend(
    prev_score=live_result["focus_score"],
    planned_duration=live_result["total_minutes"]
)
peak_risk_info = predictor.get_peak_distraction_window(hourly_risk)


# ==============================================================================
# VIEW 1: DASHBOARD & INSIGHTS (CogniAI Minimalist Layout)
# ==============================================================================
if st.session_state.nav_view == "📊 Dashboard & Insights":

    # 1. LIVE INTERACTIVE CONTROL CENTER (Presets & Sliders)
    st.markdown("""
    <div class="control-box">
        <div class="control-box-title">
            <span>⚡ Interactive Activity Control Center</span>
            <span class="pill-badge pill-orange">REAL-TIME CONNECTED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preset Scenario Buttons (CogniAI Pill Style)
    pres_cols = st.columns(4)
    with pres_cols[0]:
        if st.button("🎯 Slide 5 Benchmark (120m)", use_container_width=True):
            st.session_state.study_min = 70
            st.session_state.phone_min = 20
            st.session_state.talk_min = 10
            st.session_state.break_min = 20
            st.session_state.social_min = 0
            st.session_state.wander_min = 0
            st.rerun()
    with pres_cols[1]:
        if st.button("⚡ Deep Flow State (90m)", use_container_width=True):
            st.session_state.study_min = 90
            st.session_state.phone_min = 0
            st.session_state.talk_min = 0
            st.session_state.break_min = 10
            st.session_state.social_min = 0
            st.session_state.wander_min = 0
            st.rerun()
    with pres_cols[2]:
        if st.button("📱 Phone Trap Evening (45m)", use_container_width=True):
            st.session_state.study_min = 45
            st.session_state.phone_min = 35
            st.session_state.talk_min = 10
            st.session_state.break_min = 25
            st.session_state.social_min = 5
            st.session_state.wander_min = 0
            st.rerun()
    with pres_cols[3]:
        if st.button("🗣️ Group Disruption (50m)", use_container_width=True):
            st.session_state.study_min = 50
            st.session_state.phone_min = 15
            st.session_state.talk_min = 35
            st.session_state.break_min = 20
            st.session_state.social_min = 0
            st.session_state.wander_min = 0
            st.rerun()

    # Live Sliders
    sl1, sl2, sl3, sl4 = st.columns(4)
    with sl1:
        st.session_state.study_min = st.slider("📚 Active Study", 0, 240, st.session_state.study_min, step=5)
    with sl2:
        st.session_state.phone_min = st.slider("📱 Phone Leak", 0, 180, st.session_state.phone_min, step=5)
    with sl3:
        st.session_state.break_min = st.slider("☕ Rest Break", 0, 120, st.session_state.break_min, step=5)
    with sl4:
        st.session_state.talk_min = st.slider("🗣️ Talking / Interruption", 0, 120, st.session_state.talk_min, step=5)

    # Recalculate live
    current_breakdown = {
        "studying": st.session_state.study_min,
        "phone": st.session_state.phone_min,
        "talking": st.session_state.talk_min,
        "break": st.session_state.break_min,
        "social_media": st.session_state.social_min,
        "wandering": st.session_state.wander_min
    }
    live_result = calculate_focus_score(current_breakdown)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 2. KEY METRICS GRID (Clean white cards, soft borders & CogniAI typography)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        score = live_result["focus_score"]
        status_pill = "pill-green" if score >= 80 else ("pill-orange" if score >= 70 else "pill-red")
        score_color = "#10b981" if score >= 80 else ("#ff6a3d" if score >= 70 else "#ef4444")
        st.markdown(f"""
        <div class="cogni-card">
            <div class="cogni-card-header">
                <span class="cogni-card-title">Focus Score</span>
                <span class="pill-badge {status_pill}">{live_result['status_label']}</span>
            </div>
            <div class="cogni-card-val" style="color: {score_color};">{score}<span style="font-size: 1.2rem; color: #94a3b8; font-weight: 500;">/100</span></div>
            <div class="cogni-card-sub">Real-time cognitive efficiency</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="cogni-card">
            <div class="cogni-card-header">
                <span class="cogni-card-title">Active Study Time</span>
                <span class="pill-badge pill-orange">{live_result['study_ratio_pct']}% Focus</span>
            </div>
            <div class="cogni-card-val" style="color: #18181b;">{live_result['active_minutes']:.0f}<span style="font-size: 1.1rem; color: #94a3b8; font-weight: 500;"> min</span></div>
            <div class="cogni-card-sub">Out of {live_result['total_minutes']:.0f}m total desk time</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        dist_color = "#ef4444" if live_result["biggest_distraction"] != "None" else "#10b981"
        st.markdown(f"""
        <div class="cogni-card">
            <div class="cogni-card-header">
                <span class="cogni-card-title">Top Distraction</span>
                <span class="pill-badge pill-red">Leak Vector</span>
            </div>
            <div class="cogni-card-val" style="color: {dist_color}; font-size: 2.1rem;">{live_result['biggest_distraction']}</div>
            <div class="cogni-card-sub">{live_result['distraction_minutes']:.0f} mins lost off-task</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="cogni-card">
            <div class="cogni-card-header">
                <span class="cogni-card-title">Risk Window</span>
                <span class="pill-badge pill-orange">{peak_risk_info['peak_prob']}% Risk</span>
            </div>
            <div class="cogni-card-val" style="color: #ea580c; font-size: 1.7rem; padding-top: 5px;">{peak_risk_info['window']}</div>
            <div class="cogni-card-sub">Circadian evening fatigue</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. REALITY ALERT BANNER (CogniAI Style)
    if live_result["study_ratio_pct"] < 70:
        st.markdown(f"""
        <div class="reality-banner-danger">
            ⚠️ <strong>TIME ≠ FOCUS:</strong> Out of <strong>{live_result['total_minutes']:.0f} minutes</strong> at your desk, 
            <strong>ONLY {live_result['study_ratio_pct']}%</strong> was active productive study time!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="reality-banner-danger" style="border-left-color: #10b981;">
            🎯 <strong>HIGH FOCUS EFFICIENCY:</strong> <strong>{live_result['study_ratio_pct']}%</strong> of your session was productive deep learning!
        </div>
        """, unsafe_allow_html=True)

    # 4. CHARTS & AI COACHING SECTION (CogniAI Card Layout)
    chart_c, diag_c = st.columns([1.15, 1.0])

    with chart_c:
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 800; color: #18181b; margin-bottom: 12px;">
            🍩 Activity Distribution
        </div>
        """, unsafe_allow_html=True)

        donut_labels = ["Studying", "Phone", "Talking", "Break"]
        donut_values = [
            st.session_state.study_min,
            st.session_state.phone_min,
            st.session_state.talk_min,
            st.session_state.break_min
        ]
        if st.session_state.social_min > 0:
            donut_labels.append("Social Media")
            donut_values.append(st.session_state.social_min)
        if st.session_state.wander_min > 0:
            donut_labels.append("Mind Wandering")
            donut_values.append(st.session_state.wander_min)

        # CogniAI Warm Palette
        warm_colors = {
            "Studying": "#ff6a3d",      # CogniAI Orange
            "Phone": "#ef4444",         # Red
            "Talking": "#3b82f6",       # Blue
            "Break": "#cbd5e1",         # Soft Gray
            "Social Media": "#8b5cf6",  # Purple
            "Mind Wandering": "#f59e0b" # Amber
        }
        colors = [warm_colors.get(l, "#94a3b8") for l in donut_labels]

        fig_donut = go.Figure(data=[go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=.65,
            marker=dict(colors=colors, line=dict(color='#ffffff', width=3)),
            textinfo='label+percent',
            hoverinfo='label+value+percent',
            textfont=dict(size=12, color='#18181b', family="Plus Jakarta Sans")
        )])
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            font=dict(color='#0f172a')
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with diag_c:
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 800; color: #18181b; margin-bottom: 12px;">
            💡 AI Focus Diagnostic
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="diag-box">
            <div class="diag-label" style="color: #ff6a3d;">Diagnostic Insight</div>
            <div class="diag-content">
                "{live_result['ai_insight']}"
            </div>
        </div>

        <div class="diag-box">
            <div class="diag-label" style="color: #ea580c;">ML Predicted Threat Window</div>
            <div class="diag-content" style="color: #ea580c;">
                🕒 <strong>{peak_risk_info['window']}</strong> ({peak_risk_info['peak_prob']}% risk surge)
            </div>
        </div>

        <div class="diag-box">
            <div class="diag-label" style="color: #10b981;">Prescribed Intervention</div>
            <div class="diag-content" style="color: #047857;">
                ✅ <strong>{live_result['suggested_action']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("💾 Save & Log Session to History", type="primary", use_container_width=True):
            now = datetime.now()
            sess_record = {
                "session_id": f"sess_{int(now.timestamp())}",
                "timestamp": now.isoformat(),
                "date": str(date.today()),
                "hour_of_day": now.hour,
                "day_of_week": now.strftime("%A"),
                "subject": st.session_state.subject,
                "total_minutes": live_result["total_minutes"],
                "studying_minutes": st.session_state.study_min,
                "phone_minutes": st.session_state.phone_min,
                "talking_minutes": st.session_state.talk_min,
                "break_minutes": st.session_state.break_min,
                "social_media_minutes": st.session_state.social_min,
                "wandering_minutes": st.session_state.wander_min,
                "focus_score": live_result["focus_score"],
                "study_ratio_pct": live_result["study_ratio_pct"],
                "biggest_distraction": live_result["biggest_distraction"]
            }
            save_session(sess_record)
            st.success(f"✅ Logged! Focus Score: {live_result['focus_score']}/100")


# ==============================================================================
# VIEW 2: LIVE SESSION TRACKER (Clean CogniAI Style)
# ==============================================================================
elif st.session_state.nav_view == "⏱️ Live Session Tracker":
    st.markdown("""
    <div style="font-size: 1.6rem; font-weight: 800; color: #18181b; margin-bottom: 6px;">
        ⏱️ Interactive Study Session Tracker
    </div>
    <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 24px;">
        Track active study and distraction minutes in real time. Connected directly with the Overview Dashboard.
    </div>
    """, unsafe_allow_html=True)

    tc1, tc2 = st.columns([1.1, 1.0])

    with tc1:
        st.markdown('<div class="control-box">', unsafe_allow_html=True)
        st.session_state.subject = st.text_input("Study Subject / Goal", value=st.session_state.subject)
        
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.session_state.study_min = st.number_input("📚 Active Study (Mins)", 0, 480, st.session_state.study_min, 5)
            st.session_state.break_min = st.number_input("☕ Intentional Break (Mins)", 0, 120, st.session_state.break_min, 5)
        with c_t2:
            st.session_state.phone_min = st.number_input("📱 Phone Distraction (Mins)", 0, 240, st.session_state.phone_min, 5)
            st.session_state.talk_min = st.number_input("🗣️ Talking / Interruption (Mins)", 0, 180, st.session_state.talk_min, 5)

        with st.expander("Additional Categories (Social Media & Mind Wandering)"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.session_state.social_min = st.number_input("🌐 Social Media (Mins)", 0, 180, st.session_state.social_min, 5)
            with sc2:
                st.session_state.wander_min = st.number_input("💭 Mind Wandering (Mins)", 0, 180, st.session_state.wander_min, 5)

        if st.button("💾 Save Session Record", type="primary", use_container_width=True):
            now = datetime.now()
            sess_record = {
                "session_id": f"sess_{int(now.timestamp())}",
                "timestamp": now.isoformat(),
                "date": str(date.today()),
                "hour_of_day": now.hour,
                "day_of_week": now.strftime("%A"),
                "subject": st.session_state.subject,
                "total_minutes": live_result["total_minutes"],
                "studying_minutes": st.session_state.study_min,
                "phone_minutes": st.session_state.phone_min,
                "talking_minutes": st.session_state.talk_min,
                "break_minutes": st.session_state.break_min,
                "social_media_minutes": st.session_state.social_min,
                "wandering_minutes": st.session_state.wander_min,
                "focus_score": live_result["focus_score"],
                "study_ratio_pct": live_result["study_ratio_pct"],
                "biggest_distraction": live_result["biggest_distraction"]
            }
            save_session(sess_record)
            st.success(f"Session saved! Focus Score: {live_result['focus_score']}/100")
        st.markdown('</div>', unsafe_allow_html=True)

    with tc2:
        score = live_result["focus_score"]
        score_color = "#10b981" if score >= 80 else ("#ff6a3d" if score >= 70 else "#ef4444")
        st.markdown(f"""
        <div class="cogni-card" style="padding: 28px; text-align: center;">
            <div class="cogni-card-title" style="margin-bottom: 8px;">Real-Time Focus Score</div>
            <div style="font-size: 4.2rem; font-weight: 800; color: {score_color}; line-height: 1;">
                {score}<span style="font-size: 1.5rem; color: #94a3b8; font-weight: 500;">/100</span>
            </div>
            <div style="margin-top: 8px;"><span class="pill-badge pill-orange">{live_result['status_label']}</span></div>

            <hr style="border: 0; border-top: 1px solid #f0eee6; margin: 20px 0;">

            <div style="text-align: left; display: flex; flex-direction: column; gap: 10px; font-size: 0.95rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">Session Duration:</span>
                    <strong>{live_result['total_minutes']:.0f} Minutes</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">Active Study Share:</span>
                    <strong style="color: #ff6a3d;">{live_result['study_ratio_pct']}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">Primary Attention Leak:</span>
                    <strong style="color: #ef4444;">{live_result['biggest_distraction']}</strong>
                </div>
            </div>

            <div class="diag-box" style="margin-top: 18px; text-align: left;">
                <div class="diag-label" style="color: #ff6a3d;">AI Advice</div>
                <div style="font-size: 0.95rem; color: #18181b; font-weight: 600;">
                    "{live_result['ai_insight']}"
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: AI RISK FORECASTER (Slide 8 Machine Learning - CogniAI Style)
# ==============================================================================
elif st.session_state.nav_view == "🔮 AI Risk Forecaster":
    st.markdown("""
    <div style="font-size: 1.6rem; font-weight: 800; color: #18181b; margin-bottom: 6px;">
        🤖 Machine Learning: Distraction Risk Forecaster
    </div>
    <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 24px;">
        Learns from previous study history to forecast high-risk distraction windows (Slide 8).
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2 = st.columns([1.2, 1.0])

    with fc1:
        st.markdown('<div class="control-box">', unsafe_allow_html=True)
        st.markdown('<div class="control-box-title">⚙️ Model Input Parameters</div>', unsafe_allow_html=True)
        sim_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)
        sim_duration = st.slider("Planned Study Length (Minutes)", 30, 240, int(live_result["total_minutes"] or 90), step=15)
        sim_prev_score = st.slider("Previous Session Focus Score", 0, 100, live_result["focus_score"], step=5)
        st.markdown('</div>', unsafe_allow_html=True)

    with fc2:
        st.markdown(f"""
        <div class="cogni-card" style="border-left: 5px solid #ff6a3d;">
            <div class="cogni-card-header">
                <span class="cogni-card-title">PROTOTYPE PREDICTION (SLIDE 8)</span>
                <span class="pill-badge pill-red">PEAK RISK</span>
            </div>
            <div style="font-size: 1.65rem; font-weight: 800; color: #18181b; margin: 10px 0;">
                HIGH PROBABILITY OF DISTRACTION BETWEEN {peak_risk_info['window']}
            </div>
            <div style="font-size: 0.88rem; color: #64748b;">
                ⚠️ Note: The prediction is an early machine learning prototype, not a medical or psychological diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    hours = [17, 18, 19, 20, 21, 22, 23]
    trend_data = predictor.predict_hourly_trend(
        day_of_week=sim_day,
        prev_score=sim_prev_score,
        planned_duration=sim_duration,
        hours_to_evaluate=hours
    )
    peak_info = predictor.get_peak_distraction_window(trend_data)

    st.markdown("""
    <div style="font-size: 1.15rem; font-weight: 800; color: #18181b; margin: 16px 0 10px 0;">
        📊 Hourly Distraction Risk Probability (5 PM – 11 PM)
    </div>
    """, unsafe_allow_html=True)

    hours_labels = [entry["display_hour"] for entry in trend_data]
    probs = [entry["probability_pct"] for entry in trend_data]

    # Warm CogniAI Orange Area Line Chart
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(
        x=hours_labels,
        y=probs,
        mode='lines+markers+text',
        name='Distraction Risk',
        line=dict(color='#ff6a3d', width=3.5, shape='spline'),
        marker=dict(size=9, color='#ff6a3d', line=dict(color='#ffffff', width=2)),
        fill='tozeroy',
        fillcolor='rgba(255, 106, 61, 0.12)',
        text=[f"{p}%" for p in probs],
        textposition="top center",
        textfont=dict(color="#18181b", size=12, family="Plus Jakarta Sans")
    ))

    fig_ml.add_hline(y=50, line_dash="dash", line_color="rgba(0,0,0,0.2)", annotation_text="High Risk (50%)", annotation_position="bottom right")

    fig_ml.update_layout(
        xaxis=dict(title="Hour of Day", gridcolor="#f0eee6", tickfont=dict(color="#64748b")),
        yaxis=dict(title="Distraction Probability (%)", range=[0, 105], gridcolor="#f0eee6", tickfont=dict(color="#64748b")),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        margin=dict(t=30, b=30, l=40, r=20),
        font=dict(family="Plus Jakarta Sans", color="#18181b")
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    # 3 Strategic Defense Cards
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""
        <div class="cogni-card">
            <div class="cogni-card-title">Critical Threat Window</div>
            <div class="cogni-card-val" style="color: #ea580c; font-size: 1.8rem; margin: 6px 0;">{peak_info['window']}</div>
            <div class="cogni-card-sub">Peak Risk Probability: <strong>{peak_info['peak_prob']}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown("""
        <div class="cogni-card">
            <div class="cogni-card-title">Primary Threat Vector</div>
            <div class="cogni-card-val" style="color: #ff6a3d; font-size: 1.8rem; margin: 6px 0;">Smartphone</div>
            <div class="cogni-card-sub">Passive scrolling & notifications</div>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown("""
        <div class="cogni-card">
            <div class="cogni-card-title">Proactive Protocol</div>
            <div class="cogni-card-val" style="color: #10b981; font-size: 1.8rem; margin: 6px 0;">25m Block</div>
            <div class="cogni-card-sub">Phone in another room during block</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 4: SESSION HISTORY & DATA (CogniAI Style)
# ==============================================================================
elif st.session_state.nav_view == "📜 Session History & Data":
    st.markdown("""
    <div style="font-size: 1.6rem; font-weight: 800; color: #18181b; margin-bottom: 6px;">
        📜 Session History & Analytics
    </div>
    <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 24px;">
        Long-term focus score progression and raw study session logs.
    </div>
    """, unsafe_allow_html=True)

    if df_sessions.empty:
        st.info("No study sessions recorded yet.")
    else:
        df_display = df_sessions.copy()
        df_display["display_date"] = pd.to_datetime(df_display["timestamp"]).dt.strftime("%b %d, %H:%M")

        fig_trend = px.line(
            df_display.tail(20),
            x="display_date",
            y="focus_score",
            markers=True,
            title="Focus Score Progression (Recent Sessions)",
            labels={"display_date": "Session Date", "focus_score": "Focus Score (/100)"}
        )
        fig_trend.update_traces(line_color="#ff6a3d", marker=dict(size=7, color="#18181b"))
        fig_trend.update_layout(
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            xaxis=dict(gridcolor="#f0eee6"),
            yaxis=dict(range=[0, 105], gridcolor="#f0eee6"),
            font=dict(family="Plus Jakarta Sans", color="#0f172a")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.dataframe(
            df_sessions[[
                "date", "subject", "total_minutes", "studying_minutes", 
                "phone_minutes", "talking_minutes", "break_minutes", 
                "focus_score", "biggest_distraction"
            ]].tail(25),
            use_container_width=True
        )

        csv_data = df_sessions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Historical Data as CSV",
            data=csv_data,
            file_name="cognifocus_study_sessions.csv",
            mime="text/csv"
        )
