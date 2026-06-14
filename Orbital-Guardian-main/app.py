"""
app.py  –  Orbital Guardian
AI-Powered Space Debris Detection, Tracking & Collision Prevention
Team LEGION  |  ASTRAVA Hackathon 2026

PERFORMANCE FIXES vs original:
1. fetch_tle_group() wrapped in @st.cache_data(ttl=300) — called ONCE per 5 min
2. load_data() cached with @st.cache_data(ttl=30) — no redundant ML training
3. Trajectory: predict_trajectory() result stored in session_state — not called twice
4. Altitude profile uses cached trajectory data (no second SGP4 propagation)
5. auto_refresh: uses st.rerun() properly — no blocking time.sleep()
6. RF model is module-level singleton in predictor.py — trains once per process
7. IsolationForest on trajectory: n_estimators=50 (not 150) for small datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
import time

from data_fetcher import (
    fetch_tle_group, get_positions_df, GROUPS, compute_collision_risks
)
from predictor import (
    detect_anomalies, enrich_risks_with_ml,
    predict_trajectory, find_overpasses, orbital_stats
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Orbital Guardian",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:ital,wght@0,100;0,300;0,400;0,600;1,300&display=swap');

/* ═══════════════ ROOT VARIABLES ═══════════════ */
:root {
    --neon-green: #00ff88;
    --neon-blue: #00d4ff;
    --neon-purple: #a855f7;
    --neon-pink: #ff00aa;
    --danger-red: #ff2244;
    --warn-orange: #ff8800;
    --bg-void: #020509;
    --bg-deep: #030c14;
    --bg-mid: #071020;
    --bg-panel: #0a1628;
    --border-dim: #0f2a45;
    --border-glow: rgba(0,255,136,0.25);
    --text-bright: #e8f4fd;
    --text-mid: #7fb3d3;
    --text-dim: #3d6b8a;
}

/* ═══════════════ GLOBAL RESET ═══════════════ */
html, body, [class*="css"] {
    background-color: var(--bg-void) !important;
    color: var(--text-bright) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ═══════════════ APP BACKGROUND — DEEP SPACE ═══════════════ */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,212,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(168,85,247,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 100% 100% at 50% 50%, #020509 0%, #010307 100%);
}

/* Scanline effect */
.stApp::before {
    content:""; position:fixed; top:0; left:0; width:100%; height:100%;
    background:
        repeating-linear-gradient(
            0deg,
            transparent, transparent 3px,
            rgba(0,255,136,0.008) 3px, rgba(0,255,136,0.008) 4px
        );
    pointer-events:none; z-index:9998;
}

/* Star field noise overlay */
.stApp::after {
    content:""; position:fixed; top:0; left:0; width:100%; height:100%;
    background-image:
        radial-gradient(1px 1px at 15% 25%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 75%, rgba(0,255,136,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 15%, rgba(0,212,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 55%, rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 45%, rgba(168,85,247,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 85%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 92% 30%, rgba(0,255,136,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 8% 70%, rgba(0,212,255,0.25) 0%, transparent 100%),
        radial-gradient(2px 2px at 50% 40%, rgba(255,255,255,0.15) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 90%, rgba(255,255,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 5%, rgba(0,255,136,0.3) 0%, transparent 100%);
    pointer-events:none; z-index:9997; opacity:0.6;
}

/* ═══════════════ SIDEBAR ═══════════════ */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #040d1a 0%, #020810 40%, #030c14 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15) !important;
    box-shadow: 4px 0 40px rgba(0,212,255,0.06) !important;
}

[data-testid="stSidebar"]::before {
    content:"";
    position:absolute; top:0; left:0; right:0; height:180px;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,255,136,0.08) 0%, transparent 70%);
    pointer-events:none;
}

/* ═══════════════ SIDEBAR LOGO ═══════════════ */
.sidebar-logo {
    font-family:'Orbitron',monospace; font-size:1.3rem; font-weight:900;
    background: linear-gradient(135deg, #00ff88 0%, #00d4ff 60%, #a855f7 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; padding:14px 0 2px; letter-spacing:0.15em;
    text-shadow: none;
    filter: drop-shadow(0 0 12px rgba(0,255,136,0.4));
}

.sidebar-emblem {
    text-align:center; font-size:2.2rem; margin-bottom:4px;
    filter: drop-shadow(0 0 16px rgba(0,255,136,0.6));
    animation: float 4s ease-in-out infinite;
}

@keyframes float {
    0%,100% { transform:translateY(0px); }
    50% { transform:translateY(-6px); }
}

.sidebar-tag {
    font-family:'Share Tech Mono',monospace; font-size:0.58rem; color:#00d4ff99;
    text-align:center; letter-spacing:0.18em; margin-bottom:6px;
    text-transform:uppercase;
}

.sidebar-divider {
    height:1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), rgba(0,212,255,0.3), transparent);
    margin:14px 0;
}

.sidebar-section-label {
    font-family:'Share Tech Mono',monospace; font-size:0.6rem;
    color:var(--text-dim); letter-spacing:0.2em; text-transform:uppercase;
    margin: 12px 0 6px; padding-left:2px;
}

/* ═══════════════ SIDEBAR WIDGETS ═══════════════ */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stToggle label {
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.65rem !important;
    color:var(--text-mid) !important;
    letter-spacing:0.1em !important;
    text-transform:uppercase !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(0,212,255,0.04) !important;
    border-color: rgba(0,212,255,0.2) !important;
    border-radius:6px !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: var(--neon-green) !important;
    box-shadow: 0 0 10px rgba(0,255,136,0.5) !important;
}

/* ═══════════════ MAIN HEADER ═══════════════ */
.og-title {
    font-family:'Orbitron',monospace; font-size:2.8rem; font-weight:900;
    background: linear-gradient(135deg, #00ff88 0%, #00d4ff 40%, #a855f7 75%, #ff00aa 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:0.06em; line-height:1.05;
    filter: drop-shadow(0 0 20px rgba(0,255,136,0.3));
    animation: title-shimmer 8s ease-in-out infinite;
}

@keyframes title-shimmer {
    0%,100% { filter:drop-shadow(0 0 20px rgba(0,255,136,0.3)); }
    50% { filter:drop-shadow(0 0 35px rgba(0,212,255,0.45)); }
}

.og-sub {
    font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    color:rgba(0,255,136,0.6); letter-spacing:0.18em; margin-top:6px;
    display:flex; align-items:center; gap:8px;
}

.og-badge {
    display:inline-block;
    background: rgba(0,255,136,0.08);
    border:1px solid rgba(0,255,136,0.25);
    border-radius:3px; padding:2px 8px;
    font-family:'Share Tech Mono',monospace; font-size:0.6rem;
    color:#00ff88cc; letter-spacing:0.12em;
}

/* ═══════════════ LIVE DOT ═══════════════ */
.live-dot {
    display:inline-block; width:9px; height:9px;
    background:#00ff88; border-radius:50%;
    box-shadow: 0 0 6px #00ff88, 0 0 12px rgba(0,255,136,0.5);
    animation:pulse-live 1.6s ease-in-out infinite; margin-right:6px;
    vertical-align:middle;
}

@keyframes pulse-live {
    0%,100%{ opacity:1; box-shadow:0 0 6px #00ff88, 0 0 12px rgba(0,255,136,0.5); }
    50%{ opacity:0.3; box-shadow:0 0 2px #00ff88; }
}

/* ═══════════════ METRIC CARDS ═══════════════ */
.metric-card {
    background: linear-gradient(145deg, rgba(10,22,40,0.95) 0%, rgba(7,16,32,0.98) 100%);
    border:1px solid rgba(30,58,95,0.6);
    border-top: 2px solid transparent;
    border-radius:14px; padding:20px 22px;
    position:relative; overflow:hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(10px);
}

.metric-card::before {
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, #00ff88, #00d4ff);
}

.metric-card::after {
    content:""; position:absolute;
    top:-40%; right:-20%; width:120px; height:120px;
    background: radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 70%);
    border-radius:50%;
}

.metric-card:hover {
    transform:translateY(-3px);
    box-shadow:
        0 8px 32px rgba(0,255,136,0.12),
        0 0 0 1px rgba(0,255,136,0.15),
        inset 0 1px 0 rgba(0,255,136,0.08);
}

.metric-card .label {
    font-family:'Share Tech Mono',monospace; font-size:0.62rem;
    color:var(--neon-blue); letter-spacing:0.2em;
    text-transform:uppercase; margin-bottom:6px;
}

.metric-card .value {
    font-family:'Orbitron',monospace; font-size:1.9rem; font-weight:700;
    background: linear-gradient(135deg, #00ff88, #00d4ff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.15; margin:2px 0 6px;
}

.metric-card .delta {
    font-family:'Share Tech Mono',monospace; font-size:0.62rem;
    color:var(--text-dim); letter-spacing:0.08em;
}

.metric-card .corner-decor {
    position:absolute; bottom:12px; right:14px;
    font-size:1.6rem; opacity:0.08;
}

/* Danger metric variant */
.metric-card.danger::before {
    background: linear-gradient(90deg, #ff2244, #ff8800);
}
.metric-card.danger .value {
    background: linear-gradient(135deg, #ff2244, #ff8800);
    -webkit-background-clip:text;
}

/* Warning variant */
.metric-card.warn::before {
    background: linear-gradient(90deg, #ff8800, #ffcc00);
}
.metric-card.warn .value {
    background: linear-gradient(135deg, #ff8800, #ffcc00);
    -webkit-background-clip:text;
}

/* ═══════════════ ALERT BOXES ═══════════════ */
.alert-critical {
    background: linear-gradient(135deg, rgba(45,10,10,0.9), rgba(25,5,5,0.95));
    border:1px solid rgba(255,34,68,0.4); border-left:3px solid #ff2244;
    border-radius:10px; padding:16px 20px; margin:10px 0;
    font-family:'Share Tech Mono',monospace; font-size:0.8rem;
    animation:pulse-alert-red 2.5s ease-in-out infinite;
    position:relative; overflow:hidden;
}
.alert-critical::before {
    content:"🚨"; position:absolute; right:16px; top:50%; transform:translateY(-50%);
    font-size:1.8rem; opacity:0.4; animation:spin-slow 8s linear infinite;
}
@keyframes spin-slow { from{transform:translateY(-50%) rotate(0deg)} to{transform:translateY(-50%) rotate(360deg)} }
@keyframes pulse-alert-red {
    0%,100%{ box-shadow:0 0 10px rgba(255,34,68,0.2), inset 0 0 20px rgba(255,34,68,0.03); }
    50%{ box-shadow:0 0 25px rgba(255,34,68,0.35), inset 0 0 30px rgba(255,34,68,0.06); }
}

.alert-high {
    background: linear-gradient(135deg, rgba(45,26,10,0.9), rgba(25,15,5,0.95));
    border:1px solid rgba(255,136,0,0.35); border-left:3px solid #ff8800;
    border-radius:10px; padding:16px 20px; margin:10px 0;
    font-family:'Share Tech Mono',monospace; font-size:0.8rem;
    animation:pulse-alert-orange 3s ease-in-out infinite;
}
@keyframes pulse-alert-orange {
    0%,100%{ box-shadow:0 0 8px rgba(255,136,0,0.15); }
    50%{ box-shadow:0 0 20px rgba(255,136,0,0.25); }
}

.alert-safe {
    background: linear-gradient(135deg, rgba(10,45,26,0.9), rgba(5,25,15,0.95));
    border:1px solid rgba(0,255,136,0.25); border-left:3px solid #00ff88;
    border-radius:10px; padding:16px 20px; margin:10px 0;
    font-family:'Share Tech Mono',monospace; font-size:0.8rem;
    box-shadow: 0 0 15px rgba(0,255,136,0.06);
}

/* ═══════════════ SECTION HEADERS ═══════════════ */
.section-header {
    font-family:'Orbitron',monospace; font-size:0.85rem; font-weight:700;
    color:#00d4ff; letter-spacing:0.15em; text-transform:uppercase;
    padding-bottom:10px; margin-bottom:18px;
    position:relative;
}
.section-header::after {
    content:""; position:absolute; bottom:0; left:0;
    width:100%; height:1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.5), rgba(0,255,136,0.3), transparent);
}

/* ═══════════════ TABS ═══════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(4,13,26,0.8) !important;
    border-bottom: 1px solid rgba(30,58,95,0.4) !important;
    gap: 3px !important;
    padding: 0 4px !important;
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab"] {
    font-family:'Orbitron',monospace !important;
    font-size:0.65rem !important;
    color:rgba(74,158,255,0.7) !important;
    border-radius:8px 8px 0 0 !important;
    padding:10px 18px !important;
    letter-spacing:0.06em !important;
    transition: all 0.25s ease !important;
    background:transparent !important;
    border:none !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color:#00d4ff !important;
    background: rgba(0,212,255,0.04) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(13,32,64,0.9), rgba(10,22,40,0.95)) !important;
    color:#00ff88 !important;
    border-top:2px solid #00ff88 !important;
    box-shadow: 0 -4px 20px rgba(0,255,136,0.1) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top:20px !important;
}

/* ═══════════════ BUTTONS ═══════════════ */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,212,255,0.05)) !important;
    border: 1px solid rgba(0,255,136,0.35) !important;
    color: #00ff88 !important;
    font-family:'Orbitron',monospace !important;
    font-size:0.68rem !important;
    letter-spacing:0.12em !important;
    border-radius:8px !important;
    padding:10px 20px !important;
    transition:all 0.3s ease !important;
    text-transform:uppercase !important;
    position:relative !important;
    overflow:hidden !important;
}

.stButton > button::before {
    content:""; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.08), transparent);
    transition:left 0.4s ease;
}

.stButton > button:hover {
    box-shadow:
        0 0 20px rgba(0,255,136,0.25),
        0 0 40px rgba(0,255,136,0.08),
        inset 0 0 20px rgba(0,255,136,0.04) !important;
    border-color: rgba(0,255,136,0.6) !important;
    transform:translateY(-1px) !important;
}

.stButton > button:hover::before { left:100%; }

.stButton > button:active { transform:translateY(0) !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.1)) !important;
    border-color: rgba(0,255,136,0.5) !important;
}

/* ═══════════════ SELECTBOX & INPUTS ═══════════════ */
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
.stTextInput input {
    background: rgba(7,16,32,0.8) !important;
    border-color: rgba(30,58,95,0.6) !important;
    color: var(--text-bright) !important;
    border-radius:8px !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.8rem !important;
    transition:border-color 0.2s ease !important;
}

[data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.08) !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    background: linear-gradient(90deg, #00ff88, #00d4ff) !important;
}

/* ═══════════════ DATAFRAMES ═══════════════ */
[data-testid="stDataFrame"] {
    border-radius:10px !important;
    overflow:hidden !important;
    border:1px solid rgba(30,58,95,0.5) !important;
}

[data-testid="stDataFrame"] table thead th {
    background: rgba(7,16,32,0.95) !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.65rem !important;
    color:#00d4ff !important;
    letter-spacing:0.12em !important;
    text-transform:uppercase !important;
    border-bottom:1px solid rgba(0,212,255,0.2) !important;
}

[data-testid="stDataFrame"] table tbody tr:hover td {
    background:rgba(0,255,136,0.04) !important;
}

/* ═══════════════ INFO BOX ═══════════════ */
.info-box {
    background: rgba(0,212,255,0.03);
    border:1px solid rgba(0,212,255,0.15);
    border-left:3px solid rgba(0,212,255,0.5);
    border-radius:8px; padding:12px 16px; margin:8px 0;
    font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    color:var(--text-mid); line-height:1.6;
    backdrop-filter:blur(4px);
}

/* ═══════════════ SPINNER ═══════════════ */
.stSpinner > div {
    border-top-color: #00ff88 !important;
    border-right-color: rgba(0,255,136,0.3) !important;
}

/* ═══════════════ SCROLLBAR ═══════════════ */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:rgba(4,13,26,0.5); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00ff8866, #00d4ff44);
    border-radius:3px;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #00ff88aa, #00d4ff88); }

/* ═══════════════ PROGRESS BAR ═══════════════ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ff88, #00d4ff, #a855f7) !important;
}

/* ═══════════════ STATUS BADGE ═══════════════ */
.status-badge {
    display:inline-block; padding:3px 10px;
    border-radius:20px; font-family:'Share Tech Mono',monospace;
    font-size:0.62rem; letter-spacing:0.1em; font-weight:600;
}
.status-badge.online {
    background:rgba(0,255,136,0.1); border:1px solid rgba(0,255,136,0.3); color:#00ff88;
}
.status-badge.warning {
    background:rgba(255,136,0,0.1); border:1px solid rgba(255,136,0,0.3); color:#ff8800;
}
.status-badge.critical {
    background:rgba(255,34,68,0.1); border:1px solid rgba(255,34,68,0.3); color:#ff2244;
}

/* ═══════════════ TOGGLE ═══════════════ */
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background-color:#00ff88 !important;
}

/* ═══════════════ MULTISELECT ═══════════════ */
[data-baseweb="tag"] {
    background:rgba(0,255,136,0.12) !important;
    border:1px solid rgba(0,255,136,0.3) !important;
    border-radius:4px !important;
}

/* ═══════════════ TOOLTIPS & HOVER EFFECTS ═══════════════ */
.stMarkdown a {
    color:#00d4ff !important;
    text-decoration:none !important;
    border-bottom:1px solid rgba(0,212,255,0.3) !important;
    transition:border-color 0.2s !important;
}
.stMarkdown a:hover { border-color:#00d4ff !important; }

/* ═══════════════ SUCCESS / WARNING / ERROR BOXES ═══════════════ */
[data-testid="stAlert"] {
    border-radius:10px !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.78rem !important;
    backdrop-filter:blur(4px) !important;
}

/* ═══════════════ CORNER GLOW ACCENTS ═══════════════ */
.corner-accent-tl {
    position:fixed; top:0; left:0; width:300px; height:300px;
    background:radial-gradient(circle at top left, rgba(0,255,136,0.05) 0%, transparent 70%);
    pointer-events:none; z-index:1;
}
.corner-accent-br {
    position:fixed; bottom:0; right:0; width:300px; height:300px;
    background:radial-gradient(circle at bottom right, rgba(168,85,247,0.05) 0%, transparent 70%);
    pointer-events:none; z-index:1;
}

/* ═══════════════ PLOTLY CHARTS ═══════════════ */
.js-plotly-plot .plotly .modebar {
    background:rgba(7,16,32,0.8) !important;
    border-radius:6px !important;
}
.js-plotly-plot .plotly .modebar-btn path {
    fill:rgba(0,212,255,0.6) !important;
}

/* ═══════════════ HUD GRID OVERLAY ═══════════════ */
.hud-grid {
    position:fixed; top:0; left:0; width:100%; height:100%;
    background-image:
        linear-gradient(rgba(0,212,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.018) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events:none; z-index:0;
    mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
}

/* ═══════════════ ANIMATED ORBIT RING ═══════════════ */
.orbit-ring {
    position:fixed; top:50%; left:50%;
    width:800px; height:800px;
    border:1px solid rgba(0,255,136,0.025);
    border-radius:50%;
    transform:translate(-50%,-50%) rotateX(70deg);
    pointer-events:none; z-index:0;
    animation: orbit-spin 60s linear infinite;
}
.orbit-ring-2 {
    position:fixed; top:50%; left:50%;
    width:600px; height:600px;
    border:1px solid rgba(0,212,255,0.02);
    border-radius:50%;
    transform:translate(-50%,-50%) rotateX(70deg);
    pointer-events:none; z-index:0;
    animation: orbit-spin 40s linear infinite reverse;
}
@keyframes orbit-spin {
    from { transform: translate(-50%,-50%) rotateX(70deg) rotateZ(0deg); }
    to   { transform: translate(-50%,-50%) rotateX(70deg) rotateZ(360deg); }
}

/* ═══════════════ MAP MODE TOGGLE PILL ═══════════════ */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.68rem !important;
    color:var(--text-mid) !important;
    letter-spacing:0.08em !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {
    padding:4px 0 !important;
}

/* ═══════════════ ENHANCED METRIC CARDS - 3D DEPTH ═══════════════ */
.metric-card {
    box-shadow:
        0 4px 24px rgba(0,0,0,0.5),
        0 1px 0 rgba(0,255,136,0.06) inset,
        0 -1px 0 rgba(0,0,0,0.4) inset !important;
    transform-style: preserve-3d;
}

/* ═══════════════ GLOWING CHART BORDER ═══════════════ */
.js-plotly-plot {
    border-radius:12px;
    box-shadow:
        0 0 0 1px rgba(0,212,255,0.08),
        0 8px 32px rgba(0,0,0,0.4),
        0 0 60px rgba(0,212,255,0.03);
    overflow:hidden;
    transition: box-shadow 0.4s ease;
}
.js-plotly-plot:hover {
    box-shadow:
        0 0 0 1px rgba(0,212,255,0.18),
        0 8px 40px rgba(0,0,0,0.5),
        0 0 80px rgba(0,212,255,0.06);
}

/* ═══════════════ SECTION HEADER ENHANCED ═══════════════ */
.section-header {
    position:relative;
}
.section-header::before {
    content: "◈";
    margin-right:8px;
    color:rgba(0,212,255,0.5);
    font-size:0.7rem;
}

/* ═══════════════ BUTTON GLOW ═══════════════ */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.1)) !important;
    border: 1px solid rgba(0,255,136,0.4) !important;
    color:#00ff88 !important;
    font-family:'Orbitron',monospace !important;
    font-size:0.7rem !important;
    letter-spacing:0.1em !important;
    border-radius:8px !important;
    transition:all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.1) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(0,255,136,0.25), rgba(0,212,255,0.2)) !important;
    box-shadow: 0 0 30px rgba(0,255,136,0.25), 0 4px 16px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ═══════════════ DATAFRAME NEON THEME ═══════════════ */
[data-testid="stDataFrame"] {
    border-radius:10px !important;
    border:1px solid rgba(0,212,255,0.1) !important;
    overflow:hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
}

/* ═══════════════ TICKER ANIMATION ═══════════════ */
.live-ticker {
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem;
    color:rgba(0,255,136,0.45);
    letter-spacing:0.15em;
    overflow:hidden;
    white-space:nowrap;
    border-top:1px solid rgba(0,255,136,0.08);
    padding:4px 0;
    margin-top:8px;
}
.ticker-inner {
    display:inline-block;
    animation: ticker-scroll 30s linear infinite;
}
@keyframes ticker-scroll {
    0%   { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

/* ═══════════════ NEON GLOW TEXT ACCENT ═══════════════ */
.neon-text-green {
    color:#00ff88;
    text-shadow: 0 0 10px rgba(0,255,136,0.6), 0 0 20px rgba(0,255,136,0.3);
}
.neon-text-blue {
    color:#00d4ff;
    text-shadow: 0 0 10px rgba(0,212,255,0.6), 0 0 20px rgba(0,212,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# HUD overlay elements — grid, orbit rings, corner accents
st.markdown("""
<div class="corner-accent-tl"></div>
<div class="corner-accent-br"></div>
<div class="hud-grid"></div>
<div class="orbit-ring"></div>
<div class="orbit-ring-2"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FETCHERS
# CRITICAL: These @st.cache_data wrappers prevent redundant network calls
# and model training. fetch_tle is called 3x in this app — caching saves 2 calls.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch_tle(group_key: str) -> list:
    """Cached TLE fetch — refreshes every 5 minutes."""
    return fetch_tle_group(group_key)


@st.cache_data(ttl=30, show_spinner=False)
def load_data(group_key: str, limit: int, thresh: float):
    """
    Main data pipeline: fetch TLEs → compute positions → AI analysis.
    Cached for 30 seconds so rapid interactions don't re-run everything.
    """
    df = get_positions_df(group_key, limit=limit)
    if df.empty:
        return df, [], {}
    df = detect_anomalies(df, n_estimators=150)   # full fleet — 150 trees ok
    df, pairs = compute_collision_risks(df, threshold_km=thresh)
    pairs = enrich_risks_with_ml(df, pairs)        # RF model cached in predictor.py
    stats = orbital_stats(df)
    return df, pairs, stats


def get_map_style(mode: str):
    """Return geo layout dict for the selected map mode."""
    if "Satellite" in mode:
        return dict(
            showland=True,
            # landcolor="rgba(0,0,0,0)",  <-- I REMOVED THIS ONE
            showocean=True,
            oceancolor="rgba(0,0,0,0)",
            showcountries=True,
            countrycolor="rgba(0,255,136,0.5)",
            showcoastlines=True,
            coastlinecolor="rgba(0,212,255,0.6)",
            showframe=False,
            bgcolor="#030712",
            # Use OpenStreetMap satellite tiles via mapbox-style URL trick in Plotly
            # Plotly scattergeo doesn't support raster tiles, so we simulate
            # by using the "white-bg" scope and overlaying our own style.
            # For true satellite we switch to go.Scattermapbox below.
            landcolor="#1a2e1a", # <-- KEPT THIS ONE
            showlakes=True,
            lakecolor="#050d18",
            showrivers=True,
            rivercolor="#0a1628",
            showsubunits=True,
            subunitcolor="rgba(0,255,136,0.2)",
        )
    elif "Natural" in mode:
        return dict(
            showland=True,
            landcolor="#1a3320",
            showocean=True,
            oceancolor="#0a1e35",
            showcountries=True,
            countrycolor="rgba(0,212,255,0.3)",
            showcoastlines=True,
            coastlinecolor="rgba(0,255,136,0.4)",
            showframe=False,
            bgcolor="#030712",
            showlakes=True,
            lakecolor="#0a1628",
        )
    else:  # Normal dark space
        return dict(
            showland=True,
            landcolor="#0d1b2a",
            showocean=True,
            oceancolor="#050d18",
            showcountries=True,
            countrycolor="#1e3a5f",
            showcoastlines=True,
            coastlinecolor="#244060",
            showframe=False,
            bgcolor="#030712",
            showlakes=True,
            lakecolor="#030e1a",
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-emblem">🛰️</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-logo">ORBITAL<br>GUARDIAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">◈ AI SPACE SITUATIONAL AWARENESS ◈</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:8px;margin:8px 0 4px">
        <span class="status-badge online">● ONLINE</span>
        <span class="status-badge online">SGP4 ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">◦ satellite group</div>', unsafe_allow_html=True)
    group_key = st.selectbox("", list(GROUPS.keys()), label_visibility="collapsed")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">◦ analysis parameters</div>', unsafe_allow_html=True)
    collision_thresh = st.slider("Collision Threshold (km)", 20, 500, 200)
    pred_minutes     = st.slider("Prediction Window (min)", 30, 180, 90)
    sat_limit        = st.slider("Max Satellites", 10, 80, 50)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">◦ observer location</div>', unsafe_allow_html=True)
    obs_lat = st.number_input("Latitude",  value=17.44, min_value=-90.0,  max_value=90.0,  step=0.1)
    obs_lon = st.number_input("Longitude", value=78.49, min_value=-180.0, max_value=180.0, step=0.1)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">◦ map display mode</div>', unsafe_allow_html=True)

    map_mode = st.radio(
        "",
        ["🌍 Normal (Dark Space)", "🛰️ Live Satellite", "🌐 Natural Earth"],
        label_visibility="collapsed",
        index=0,
    )
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:rgba(0,212,255,0.4);
    padding:6px 8px;background:rgba(0,212,255,0.03);border:1px solid rgba(0,212,255,0.1);
    border-radius:6px;line-height:1.8;margin-top:4px'>
    🛰️ Satellite = real imagery<br>🌍 Normal = space ops HUD<br>🌐 Natural = classic globe
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">◦ globe settings</div>', unsafe_allow_html=True)
    globe_proj = st.selectbox(
        "",
        ["orthographic", "natural earth", "mercator", "robinson", "azimuthal equal area"],
        label_visibility="collapsed",
        index=0,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    auto_refresh = st.toggle("⟳  Auto Refresh (30s)", value=False)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;line-height:2.2;padding:4px 2px'>
    <div style='color:rgba(0,212,255,0.5);letter-spacing:0.15em;margin-bottom:4px'>DATA SOURCES</div>
    <div style='color:rgba(100,160,200,0.6)'>▸ CelesTrak (Live TLE Feed)</div>
    <div style='color:rgba(100,160,200,0.6)'>▸ SGP4 Propagator (NASA/NORAD)</div>
    <div style='color:rgba(100,160,200,0.6)'>▸ Fallback TLE Library</div>
    <div style='color:rgba(0,212,255,0.5);letter-spacing:0.15em;margin:8px 0 4px'>AI ENGINE</div>
    <div style='color:rgba(100,160,200,0.6)'>▸ Isolation Forest (Anomaly)</div>
    <div style='color:rgba(100,160,200,0.6)'>▸ Random Forest (Collision ML)</div>
    <div style='color:rgba(100,160,200,0.6)'>▸ 3D ECI Proximity Detection</div>
    <div style='color:rgba(0,255,136,0.25);letter-spacing:0.15em;margin:10px 0 2px;text-align:center'>
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─</div>
    <div style='color:rgba(0,255,136,0.35);text-align:center;letter-spacing:0.12em'>
    TEAM LEGION · ASTRAVA 2026</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
h1, h2 = st.columns([4, 1])
with h1:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
    st.markdown(f"""
    <div style="margin-bottom:4px">
        <span class="og-badge">ASTRAVA 2026</span>&nbsp;
        <span class="og-badge">TEAM LEGION</span>&nbsp;
        <span class="og-badge">v2.0</span>&nbsp;
        <span class="og-badge" style="border-color:rgba(168,85,247,0.3);color:#a855f7aa;background:rgba(168,85,247,0.06)">SGP4 ENGINE</span>
    </div>
    <div class="og-title">ORBITAL GUARDIAN</div>
    <div class="og-sub">
        <span class="live-dot"></span>
        <span>LIVE&nbsp;·&nbsp;AI SPACE DEBRIS DETECTION &amp; COLLISION PREVENTION</span>
        <span style="color:rgba(0,212,255,0.4)">|</span>
        <span style="color:rgba(0,212,255,0.7)">{now_str}</span>
    </div>
    <div class="live-ticker">
        <span class="ticker-inner">
        ◈ ORBITAL GUARDIAN MISSION CONTROL &nbsp;&nbsp;▸&nbsp;&nbsp;
        MONITORING LEO / MEO / GEO ORBITAL SHELLS &nbsp;&nbsp;▸&nbsp;&nbsp;
        AI COLLISION AVOIDANCE SYSTEM ACTIVE &nbsp;&nbsp;▸&nbsp;&nbsp;
        ISOLATION FOREST ANOMALY DETECTION ONLINE &nbsp;&nbsp;▸&nbsp;&nbsp;
        RANDOM FOREST ML RISK CLASSIFIER DEPLOYED &nbsp;&nbsp;▸&nbsp;&nbsp;
        3D ECI PROXIMITY DETECTION RUNNING &nbsp;&nbsp;▸&nbsp;&nbsp;
        CELESTRAK TLE FEED CONNECTED &nbsp;&nbsp;▸&nbsp;&nbsp;
        TEAM LEGION · ASTRAVA HACKATHON 2026 &nbsp;&nbsp;◈
        </span>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("⟳  REFRESH DATA", type="primary"):
        st.cache_data.clear()
        st.rerun()

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (cached — fast on subsequent renders)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("⟳  Fetching orbital data & running AI analysis..."):
    df, risk_pairs, stats = load_data(group_key, sat_limit, collision_thresh)

if df.empty:
    st.error("⚠️  No satellite data available. Check your internet connection.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# METRIC ROW
# ══════════════════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns(5)
metric_configs = [
    (m1, "🛰️ TRACKED",     stats.get("total", 0),            "satellites active",     "🛰️", ""),
    (m2, "⚡ RISK PAIRS",  len(risk_pairs),                   f"within {collision_thresh}km", "⚠️", "danger" if len(risk_pairs) > 0 else ""),
    (m3, "◎ AVG SPEED",   f"{stats.get('avg_speed', 0)}", "km/s orbital velocity",    "◎", ""),
    (m4, "▲ AVG ALTITUDE", f"{stats.get('avg_alt_km', 0)}", "km LEO/MEO mean orbit",   "▲", ""),
    (m5, "⊘ ANOMALIES",   stats.get("anomalies", 0),         "AI-flagged objects",      "⊘", "warn" if stats.get("anomalies", 0) > 0 else ""),
]
for col, label, value, delta, icon, variant in metric_configs:
    with col:
        variant_class = f"metric-card {variant}" if variant else "metric-card"
        st.markdown(f"""
        <div class="{variant_class}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="delta">{delta}</div>
            <div class="corner-decor">{icon}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="height:24px"></div>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;opacity:0.4">
    <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3))"></div>
    <div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#00d4ff;letter-spacing:0.2em">MISSION CONTROL</div>
    <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(0,212,255,0.3),transparent)"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍  LIVE GLOBE",
    "⚠️  COLLISION RISKS",
    "🔮  TRAJECTORY AI",
    "📡  OVERPASS FINDER",
    "📊  ANALYTICS",
])


# ── TAB 1: LIVE GLOBE ─────────────────────────────────────────────────────────
with tab1:
    lc1, lc2 = st.columns([3, 1])
    with lc1:

        # ── Map mode badge
        mode_badge_color = {"🛰️ Live Satellite": "#00ff88", "🌐 Natural Earth": "#00d4ff", "🌍 Normal (Dark Space)": "#a855f7"}
        badge_col = mode_badge_color.get(map_mode, "#00d4ff")
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
            <div class="section-header" style="margin-bottom:0;padding-bottom:0;border:none">
                REAL-TIME SATELLITE POSITIONS  ·  SGP4 PROPAGATED
            </div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.62rem;
                padding:4px 14px;border:1px solid {badge_col}55;border-radius:20px;
                background:{badge_col}11;color:{badge_col};letter-spacing:0.12em;
                animation:pulse-live 2s ease-in-out infinite;white-space:nowrap">
                {map_mode}
            </div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,rgba(0,212,255,0.5),rgba(0,255,136,0.3),transparent);margin-bottom:18px"></div>
        """, unsafe_allow_html=True)

        color_map = {"🟢 SAFE": "#00ff88", "🟡 MEDIUM": "#ffcc00", "🟠 HIGH": "#ff8800", "🔴 CRITICAL": "#ff2222"}
        size_map  = {"🟢 SAFE": 6, "🟡 MEDIUM": 9, "🟠 HIGH": 13, "🔴 CRITICAL": 16}

        # ── SATELLITE MODE: use Scattermapbox with real satellite tiles
        if "Satellite" in map_mode:
            fig = go.Figure()
            for level, color in color_map.items():
                if "risk_level" in df.columns:
                    sub = df[df["risk_level"] == level]
                else:
                    sub = df if level == "🟢 SAFE" else pd.DataFrame()
                if sub.empty:
                    continue
                fig.add_trace(go.Scattermapbox(
                    lat=sub["lat"], lon=sub["lon"],
                    mode="markers",
                    marker=dict(
                        size=size_map[level] + 2,
                        color=color,
                        opacity=0.92,
                    ),
                    name=level,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Lat: %{lat:.2f}°  Lon: %{lon:.2f}°<br>"
                        "Alt: %{customdata[1]} km  |  Speed: %{customdata[2]} km/s<br>"
                        "Status: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=list(zip(
                        sub["name"], sub["alt_km"], sub["speed_km_s"],
                        sub["risk_level"] if "risk_level" in sub.columns else ["SAFE"] * len(sub),
                    )),
                ))

            # Anomaly rings on mapbox
            if "is_anomaly" in df.columns:
                anom = df[df["is_anomaly"]]
                if not anom.empty:
                    fig.add_trace(go.Scattermapbox(
                        lat=anom["lat"], lon=anom["lon"],
                        mode="markers",
                        marker=dict(size=22, color="rgba(255,0,255,0.25)"),
                        name="⚠️ Anomaly",
                        hovertemplate="<b>%{customdata}</b><br>AI anomaly detected<extra></extra>",
                        customdata=anom["name"],
                    ))

            # Observer
            fig.add_trace(go.Scattermapbox(
                lat=[obs_lat], lon=[obs_lon],
                mode="markers+text",
                marker=dict(size=16, color="#00d4ff"),
                text=["◉ YOU"],
                textposition="top right",
                textfont=dict(color="#00d4ff", size=12),
                name="Observer",
            ))

            fig.update_layout(
                mapbox=dict(
                    style="satellite-streets",
                    center=dict(lat=20, lon=80),
                    zoom=1.2,
                    accesstoken="YOUR_MAPBOX_TOKEN",
                ),
                paper_bgcolor="#030712",
                font=dict(color="#94a3b8", family="Share Tech Mono"),
                height=620,
                legend=dict(bgcolor="rgba(7,16,32,0.85)", bordercolor="#1e3a5f",
                            font=dict(size=11, family="Share Tech Mono"), x=0.01, y=0.01),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Satellite mode glow note
            st.markdown("""
            <div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;
            color:rgba(0,255,136,0.5);padding:6px 12px;background:rgba(0,255,136,0.03);
            border:1px solid rgba(0,255,136,0.1);border-radius:6px;margin-top:-8px'>
            🛰️ LIVE SATELLITE IMAGERY — Real-world terrain visible beneath orbital objects
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── NORMAL / NATURAL EARTH: Scattergeo with custom dark theme
            geo_style = get_map_style(map_mode)
            fig = go.Figure()
            for level, color in color_map.items():
                if "risk_level" in df.columns:
                    sub = df[df["risk_level"] == level]
                else:
                    sub = df if level == "🟢 SAFE" else pd.DataFrame()
                if sub.empty:
                    continue
                fig.add_trace(go.Scattergeo(
                    lat=sub["lat"], lon=sub["lon"],
                    mode="markers",
                    marker=dict(
                        size=size_map[level],
                        color=color,
                        opacity=0.92,
                        line=dict(color="rgba(255,255,255,0.3)", width=0.8),
                        symbol="circle",
                    ),
                    name=level,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Lat: %{lat:.2f}°  Lon: %{lon:.2f}°<br>"
                        "Alt: %{customdata[1]} km  |  Speed: %{customdata[2]} km/s<br>"
                        "Status: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=list(zip(
                        sub["name"], sub["alt_km"], sub["speed_km_s"],
                        sub["risk_level"] if "risk_level" in sub.columns else ["SAFE"] * len(sub),
                    )),
                ))

            # Anomaly rings
            if "is_anomaly" in df.columns:
                anom = df[df["is_anomaly"]]
                if not anom.empty:
                    fig.add_trace(go.Scattergeo(
                        lat=anom["lat"], lon=anom["lon"], mode="markers",
                        marker=dict(size=20, color="rgba(255,0,255,0.12)", line=dict(color="#ff00ff", width=2.5)),
                        name="⚠️ Anomaly",
                        hovertemplate="<b>%{customdata}</b><br>AI anomaly detected<extra></extra>",
                        customdata=anom["name"],
                    ))

            # Observer
            fig.add_trace(go.Scattergeo(
                lat=[obs_lat], lon=[obs_lon], mode="markers+text",
                marker=dict(size=14, color="#00d4ff", symbol="star",
                            line=dict(color="white", width=1.5)),
                text=["◉ YOU"], textposition="top center",
                textfont=dict(color="#00d4ff", size=12, family="Share Tech Mono"), name="Observer",
            ))

            fig.update_layout(
            geo=dict(
            **geo_style,
            projection_type=globe_proj,
            center=dict(lat=20, lon=80),
            lonaxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.06)"),
            lataxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.06)")
            ),
                paper_bgcolor="#030712",
                font=dict(color="#94a3b8", family="Share Tech Mono"),
                height=620,
                legend=dict(bgcolor="rgba(7,16,32,0.85)", bordercolor="#1e3a5f",
                            font=dict(size=11, family="Share Tech Mono"), x=0, y=0),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with lc2:
        st.markdown('<div class="section-header">OBJECT LIST</div>', unsafe_allow_html=True)
        cols = ["name", "alt_km", "speed_km_s"]
        if "risk_level" in df.columns:
            cols.append("risk_level")
        disp = df[cols].copy()
        disp.columns = ["Name", "Alt(km)", "Spd(km/s)"] + (["Status"] if "risk_level" in df.columns else [])
        st.dataframe(disp, height=480, use_container_width=True, hide_index=True)
        st.markdown("""
        <div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;margin-top:8px'>
        <span style='color:#00ff88'>●</span> SAFE &nbsp;
        <span style='color:#ffcc00'>●</span> MEDIUM<br>
        <span style='color:#ff8800'>●</span> HIGH &nbsp;
        <span style='color:#ff2222'>●</span> CRITICAL<br>
        <span style='color:#ff00ff'>○</span> AI ANOMALY &nbsp;
        <span style='color:#00d4ff'>★</span> YOU
        </div>""", unsafe_allow_html=True)


# ── TAB 2: COLLISION RISKS ────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">AI COLLISION RISK ANALYSIS  ·  3D ECI PROXIMITY + ML PROBABILITY</div>', unsafe_allow_html=True)

    if risk_pairs:
        critical = [p for p in risk_pairs if "CRITICAL" in p.get("Level", "")]
        high     = [p for p in risk_pairs if "HIGH"     in p.get("Level", "")]
        if critical:
            st.markdown(f'<div class="alert-critical">🚨 CRITICAL — {len(critical)} pair(s) at extreme proximity. ML models indicate elevated collision probability.</div>', unsafe_allow_html=True)
        elif high:
            st.markdown(f'<div class="alert-high">⚠️ HIGH RISK — {len(high)} pair(s) within danger threshold.</div>', unsafe_allow_html=True)

        st.dataframe(pd.DataFrame(risk_pairs), use_container_width=True, hide_index=True)

        rdf = pd.DataFrame(risk_pairs)
        fig2 = go.Figure(go.Bar(
            x=(rdf["Object A"].str[:16] + " ↔ " + rdf["Object B"].str[:16]),
            y=rdf["Distance (km)"],
            marker=dict(
                color=rdf["Distance (km)"],
                colorscale=[[0, "#ff0000"], [0.4, "#ff8800"], [0.7, "#ffcc00"], [1, "#00ff88"]],
                line=dict(color="#1e3a5f", width=1),
            ),
            text=rdf.get("ML Prob", ""),
            textposition="outside",
        ))
        fig2.update_layout(
            title="Close-Approach Distances (km)  ·  ML Collision Probability",
            paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
            font=dict(color="#94a3b8", family="Share Tech Mono", size=10),
            xaxis=dict(gridcolor="#1e3a5f", tickangle=-30),
            yaxis=dict(gridcolor="#1e3a5f", title="Distance (km)"),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

        if "risk_score" in df.columns:
            fig_sc = px.scatter(
                df, x="alt_km", y="speed_km_s", color="risk_score",
                color_continuous_scale="RdYlGn_r", hover_name="name",
                title="Risk Distribution — Altitude vs Speed",
                labels={"alt_km": "Alt (km)", "speed_km_s": "Speed (km/s)", "risk_score": "Risk"},
            )
            fig_sc.update_layout(paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
                                  font=dict(color="#94a3b8"), height=360)
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.markdown(f'<div class="alert-safe">✅ ALL CLEAR — No objects within {collision_thresh} km. Orbital environment nominal.</div>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Histogram(
            x=df["alt_km"], nbinsx=25,
            marker=dict(color="#00ff88", line=dict(color="#030712", width=0.5)),
        ))
        fig3.add_vline(x=400,  line_dash="dot", line_color="#00d4ff", annotation_text="ISS ~400km",     annotation_font_color="#00d4ff")
        fig3.add_vline(x=550,  line_dash="dot", line_color="#ffcc00", annotation_text="Starlink ~550km", annotation_font_color="#ffcc00")
        fig3.add_vline(x=2000, line_dash="dot", line_color="#ff8800", annotation_text="MEO boundary",   annotation_font_color="#ff8800")
        fig3.update_layout(
            title="Altitude Distribution", paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
            font=dict(color="#94a3b8", family="Share Tech Mono"),
            xaxis=dict(gridcolor="#1e3a5f", title="Altitude (km)"),
            yaxis=dict(gridcolor="#1e3a5f", title="Count"),
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── TAB 3: TRAJECTORY AI ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">AI TRAJECTORY PREDICTION  ·  SGP4 PROPAGATION + ANOMALY DETECTION</div>', unsafe_allow_html=True)

    # Use cached TLEs — no extra network call
    sats_list = cached_fetch_tle(group_key)
    sat_names = [s["name"] for s in sats_list[:40]]

    tc1, tc2 = st.columns([2, 1])
    with tc1:
        selected = st.multiselect(
            "Select objects to predict (max 4):",
            sat_names,
            default=sat_names[:min(3, len(sat_names))],
        )
    with tc2:
        show_anom = st.toggle("Show Anomaly Points", value=True)

    st.markdown("""
    <div class="info-box">
    ℹ️ Each satellite's full 90-min ground track is computed using the same SGP4
    algorithm used by NASA/NORAD. Magenta rings = Isolation Forest anomalies.
    </div>""", unsafe_allow_html=True)

    if st.button("🔮  RUN TRAJECTORY PREDICTION"):
        if not selected:
            st.warning("Select at least one satellite.")
        else:
            # ── Compute trajectories — store in session_state to avoid double computation
            traj_cache = {}
            colors = ["#00ff88", "#00d4ff", "#ff8800", "#ff00ff"]

            prog = st.progress(0, text="Computing trajectories...")
            for idx, sname in enumerate(selected[:4]):
                sat_d = next((s for s in sats_list if s["name"] == sname), None)
                if not sat_d:
                    continue
                prog.progress((idx + 0.5) / len(selected[:4]), text=f"Propagating {sname}...")
                # predict_trajectory now includes anomaly detection internally
                tdf = predict_trajectory(sat_d, minutes=pred_minutes, step=2)
                if not tdf.empty:
                    traj_cache[sname] = tdf
                prog.progress((idx + 1) / len(selected[:4]))
            prog.empty()

            if not traj_cache:
                st.error("Could not compute trajectories. TLE data may be invalid.")
                st.stop()

            # ── Ground track map (uses cached traj — NOT recomputed)
            fig_t = go.Figure()
            for idx, (sname, tdf) in enumerate(traj_cache.items()):
                col = colors[idx % 4]

                fig_t.add_trace(go.Scattergeo(
                    lat=tdf["lat"], lon=tdf["lon"], mode="lines",
                    line=dict(color=col, width=2.5), opacity=0.85, name=sname,
                    hovertemplate=(
                        f"<b>{sname}</b><br>"
                        "T+%{customdata[0]}min | %{customdata[1]}<br>"
                        "Alt: %{customdata[2]}km  Speed: %{customdata[3]} km/s"
                        "<extra></extra>"
                    ),
                    customdata=list(zip(
                        tdf["min_ahead"], tdf["time_utc"],
                        tdf["alt_km"], tdf["speed_km_s"],
                    )),
                ))
                # Current position marker (T+0)
                fig_t.add_trace(go.Scattergeo(
                    lat=[tdf.iloc[0]["lat"]], lon=[tdf.iloc[0]["lon"]],
                    mode="markers",
                    marker=dict(size=14, color=col, symbol="star",
                                line=dict(color="white", width=1)),
                    name=f"NOW: {sname}", showlegend=False,
                ))
                # Anomaly rings
                if show_anom and "is_anomaly" in tdf.columns:
                    ap = tdf[tdf["is_anomaly"]]
                    if not ap.empty:
                        fig_t.add_trace(go.Scattergeo(
                            lat=ap["lat"], lon=ap["lon"], mode="markers",
                            marker=dict(size=14, color="rgba(0,0,0,0)",
                                        line=dict(color="#ff00ff", width=2.5)),
                            name=f"⚠️ Anomaly: {sname}", showlegend=True,
                            hovertemplate=(
                                f"<b>{sname} — AI ANOMALY</b><br>"
                                "T+%{customdata[0]}min | %{customdata[1]}<br>"
                                "Score: %{customdata[2]}<extra></extra>"
                            ),
                            customdata=list(zip(
                                ap["min_ahead"], ap["time_utc"],
                                ap.get("anomaly_score", [0]*len(ap)),
                            )),
                        ))

            geo_style_t = get_map_style(map_mode)
            fig_t.update_layout(
                geo=dict(
                    **geo_style_t,
                    projection_type=globe_proj,
                    # Added correct Plotly gridline syntax
                    lonaxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.06)"),
                    lataxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.06)"),
                ),
                title=dict(
                    text=f"Predicted Ground Tracks — Next {pred_minutes} min  ·  SGP4",
                    font=dict(family="Orbitron", color="#00d4ff", size=13),
                ),
                paper_bgcolor="#030712",
                font=dict(color="#94a3b8", family="Share Tech Mono"),
                height=580,
                legend=dict(bgcolor="rgba(7,16,32,0.9)", bordercolor="#1e3a5f"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_t, use_container_width=True)

            # ── Altitude profile (uses SAME cached data — no second SGP4 run)
            st.markdown('<div class="section-header">ALTITUDE & SPEED PROFILE</div>', unsafe_allow_html=True)
            ac1, ac2 = st.columns(2)

            with ac1:
                fig_alt = go.Figure()
                for idx, (sname, tdf) in enumerate(traj_cache.items()):
                    fig_alt.add_trace(go.Scatter(
                        x=tdf["min_ahead"], y=tdf["alt_km"],
                        mode="lines", name=sname,
                        line=dict(color=colors[idx % 4], width=2),
                        hovertemplate="<b>" + sname + "</b><br>T+%{x}min → %{y}km<extra></extra>",
                    ))
                fig_alt.update_layout(
                    title="Altitude (km) vs Time", paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
                    font=dict(color="#94a3b8", family="Share Tech Mono"),
                    xaxis=dict(title="Minutes Ahead", gridcolor="#1e3a5f"),
                    yaxis=dict(title="Altitude (km)", gridcolor="#1e3a5f"),
                    legend=dict(bgcolor="#0d1b2a"), height=280,
                )
                st.plotly_chart(fig_alt, use_container_width=True)

            with ac2:
                fig_spd = go.Figure()
                for idx, (sname, tdf) in enumerate(traj_cache.items()):
                    fig_spd.add_trace(go.Scatter(
                        x=tdf["min_ahead"], y=tdf["speed_km_s"],
                        mode="lines", name=sname,
                        line=dict(color=colors[idx % 4], width=2),
                        hovertemplate="<b>" + sname + "</b><br>T+%{x}min → %{y} km/s<extra></extra>",
                    ))
                fig_spd.update_layout(
                    title="Speed (km/s) vs Time", paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
                    font=dict(color="#94a3b8", family="Share Tech Mono"),
                    xaxis=dict(title="Minutes Ahead", gridcolor="#1e3a5f"),
                    yaxis=dict(title="Speed (km/s)", gridcolor="#1e3a5f"),
                    legend=dict(bgcolor="#0d1b2a"), height=280,
                )
                st.plotly_chart(fig_spd, use_container_width=True)

            # ── Anomaly summary table
            anom_rows = []
            for sname, tdf in traj_cache.items():
                if "is_anomaly" in tdf.columns:
                    ap = tdf[tdf["is_anomaly"]]
                    for _, row in ap.iterrows():
                        anom_rows.append({
                            "Satellite": sname,
                            "T+ (min)":  row["min_ahead"],
                            "Time (UTC)": row["time_utc"],
                            "Alt (km)":  row["alt_km"],
                            "Score":     row.get("anomaly_score", "—"),
                        })
            if anom_rows:
                st.markdown('<div class="section-header" style="margin-top:8px">DETECTED ANOMALIES ON TRAJECTORY</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(anom_rows), use_container_width=True, hide_index=True)
    else:
        st.info("⬆️  Select satellites above and click **RUN TRAJECTORY PREDICTION**")


# ── TAB 4: OVERPASS FINDER ────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">SATELLITE OVERPASS PREDICTOR  ·  ECI → AzEl COORDINATE TRANSFORM</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    ℹ️ Overpass times are computed using the correct ECI→SEZ→AzEl coordinate transform
    (same method as Heavens-Above and NASA Orbital Viewer). All times are in UTC.
    GEO satellites (GOES, Himawari, Meteosat) are stationary — they don't "pass over".
    </div>""", unsafe_allow_html=True)

    ov_c1, ov_c2, ov_c3 = st.columns([1, 1, 1])
    with ov_c1:
        op_lat = st.number_input("Observer Latitude",  value=obs_lat,  min_value=-90.0,  max_value=90.0,  step=0.01, key="op_lat", format="%.4f")
    with ov_c2:
        op_lon = st.number_input("Observer Longitude", value=obs_lon,  min_value=-180.0, max_value=180.0, step=0.01, key="op_lon", format="%.4f")
    with ov_c3:
        op_min_el = st.number_input("Min Elevation (°)", value=10.0, min_value=0.0, max_value=89.0, step=1.0, key="op_min_el")

    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:0.78rem;color:#4a9eff;margin-bottom:8px">'
        f'📍 Observer: <b style="color:#00ff88">{op_lat:.4f}°N  {op_lon:.4f}°E</b> · '
        f'Min elevation: <b style="color:#00ff88">{op_min_el:.0f}°</b></div>',
        unsafe_allow_html=True,
    )

    oc1, oc2 = st.columns([2, 1])
    with oc1:
        # Use cached TLEs — avoid duplicate network call
        sats_list2 = cached_fetch_tle(group_key)
        op_name = st.selectbox("Select satellite:", [s["name"] for s in sats_list2[:50]])
    with oc2:
        op_hours = st.slider("Search window (hours)", 6, 72, 24)

    if st.button("📡  FIND OVERPASSES"):
        sat_d = next((s for s in sats_list2 if s["name"] == op_name), None)
        if not sat_d:
            st.error("Satellite data not found.")
        else:
            with st.spinner(f"Computing overpasses for {op_name} over next {op_hours}h..."):
                passes = find_overpasses(
                    sat_d, op_lat, op_lon,
                    hours=op_hours,
                    min_elevation=op_min_el,
                )

            if passes:
                st.success(f"✅  Found **{len(passes)}** overpass(es) in next {op_hours} hours")

                pass_df = pd.DataFrame([{
                    "Start (UTC)":    p["start_utc"],
                    "End (UTC)":      p.get("end_utc", "—"),
                    "Max Elev (°)":   p["max_elevation_deg"],
                    "Peak Time":      p.get("max_el_time_utc", "—"),
                    "Duration (min)": p["duration_min"],
                    "Start Az (°)":   p["azimuth_deg"],
                    "Range (km)":     p.get("range_km", "—"),
                    "Visibility": (
                        "✅ GREAT" if p["max_elevation_deg"] > 45 else
                        "👍 GOOD"  if p["max_elevation_deg"] > 30 else
                        "⚡ FAIR"  if p["max_elevation_deg"] > 15 else
                        "👁️ LOW"
                    ),
                } for p in passes])

                st.dataframe(pass_df, use_container_width=True, hide_index=True)

                # ── Ground-track map for next 4 passes
                st.markdown('<div class="section-header" style="margin-top:16px">GROUND TRACK — NEXT PASSES</div>', unsafe_allow_html=True)

                fig_ov = go.Figure()
                colors_ov = ["#00ff88", "#00d4ff", "#ff8800", "#ff00ff",
                             "#ffcc00", "#a855f7", "#ec4899", "#22d3ee"]

                from sgp4.api import Satrec as _Satrec, jday as _jday
                from data_fetcher import eci_to_geodetic as _etg

                for pidx, p in enumerate(passes[:4]):
                    start_dt = p["start_dt"]
                    dur      = p["duration_min"]
                    col_ov   = colors_ov[pidx % len(colors_ov)]
                    label    = f"Pass {pidx+1}: {p['start_utc']} (max {p['max_elevation_deg']}°)"

                    # Compute track ±2 min around pass (use same sgp4 object)
                    track_rows = []
                    try:
                        _sat = _Satrec.twoline2rv(sat_d["tle1"], sat_d["tle2"])
                        t_cur = start_dt - timedelta(minutes=2)
                        t_end = start_dt + timedelta(minutes=dur + 2)
                        while t_cur <= t_end:
                            _jd, _fr = _jday(t_cur.year, t_cur.month, t_cur.day,
                                             t_cur.hour, t_cur.minute, t_cur.second)
                            _e, _r, _ = _sat.sgp4(_jd, _fr)
                            if _e == 0:
                                _lat, _lon, _alt = _etg(_r, _jd + _fr)
                                track_rows.append({
                                    "lat": _lat, "lon": _lon, "alt": round(_alt, 1),
                                    "t": t_cur.strftime("%H:%M UTC"),
                                })
                            t_cur += timedelta(minutes=1)
                    except Exception:
                        pass

                    if track_rows:
                        tdf_ov = pd.DataFrame(track_rows)
                        fig_ov.add_trace(go.Scattergeo(
                            lat=tdf_ov["lat"], lon=tdf_ov["lon"],
                            mode="lines+markers",
                            line=dict(color=col_ov, width=2.5),
                            marker=dict(size=4, color=col_ov),
                            name=label,
                            hovertemplate=(
                                "<b>" + label + "</b><br>"
                                "Lat:%{lat:.2f}° Lon:%{lon:.2f}°<br>"
                                "Alt: %{customdata[0]} km  Time: %{customdata[1]}<extra></extra>"
                            ),
                            customdata=list(zip(tdf_ov["alt"], tdf_ov["t"])),
                            opacity=0.85,
                        ))
                        # Mark pass start with triangle
                        vis_start = track_rows[min(2, len(track_rows)-1)]
                        fig_ov.add_trace(go.Scattergeo(
                            lat=[vis_start["lat"]], lon=[vis_start["lon"]],
                            mode="markers",
                            marker=dict(size=14, color=col_ov, symbol="triangle-up",
                                        line=dict(color="white", width=1.5)),
                            showlegend=False,
                            hovertemplate=f"<b>Pass {pidx+1} starts</b><br>{vis_start['t']}<extra></extra>",
                        ))

                # Observer marker
                fig_ov.add_trace(go.Scattergeo(
                    lat=[op_lat], lon=[op_lon],
                    mode="markers+text",
                    marker=dict(size=14, color="#00d4ff", symbol="star",
                                line=dict(color="white", width=1.5)),
                    text=["YOU"], textposition="top right",
                    textfont=dict(color="#00d4ff", size=12, family="Share Tech Mono"),
                    name="Observer",
                    hovertemplate=f"Observer<br>{op_lat:.4f}°N  {op_lon:.4f}°E<extra></extra>",
                ))

                geo_style_ov = get_map_style(map_mode)
                fig_ov.update_layout(
                geo=dict(
                    **geo_style_ov,
                    projection_type=globe_proj,
                    center=dict(lat=op_lat, lon=op_lon),
                    lataxis_range=[max(op_lat - 60, -90), min(op_lat + 60, 90)],
                    lonaxis_range=[op_lon - 80, op_lon + 80],
                    # Added correct Plotly gridline syntax
                    lonaxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.07)"),
                    lataxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.07)"),
                ),
                paper_bgcolor="#030712",
                    font=dict(color="#94a3b8", family="Share Tech Mono"),
                    height=540,
                    legend=dict(bgcolor="rgba(7,16,32,0.9)", bordercolor="#1e3a5f",
                                font=dict(size=10, family="Share Tech Mono")),
                    margin=dict(l=0, r=0, t=10, b=0),
                    title=dict(
                        text=f"Overpass Ground Tracks — {op_name}",
                        font=dict(family="Orbitron", color="#00d4ff", size=13),
                    ),
                )
                st.plotly_chart(fig_ov, use_container_width=True)

                # ── Elevation bar chart
                st.markdown('<div class="section-header" style="margin-top:8px">MAX ELEVATION PER PASS</div>', unsafe_allow_html=True)
                fig_el = go.Figure(go.Bar(
                    x=[f"Pass {i+1}<br>{p['start_utc']}" for i, p in enumerate(passes)],
                    y=pass_df["Max Elev (°)"],
                    marker=dict(
                        color=pass_df["Max Elev (°)"],
                        colorscale=[[0, "#1e3a5f"], [0.3, "#00d4ff"], [0.7, "#00ff88"], [1, "#ffffff"]],
                        line=dict(color="#1e3a5f", width=1),
                    ),
                    text=[f"{v}°  {vis}" for v, vis in zip(pass_df["Max Elev (°)"], pass_df["Visibility"])],
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=10),
                ))
                fig_el.add_hline(y=45, line_dash="dot", line_color="#00ff88",
                                  annotation_text="Great (45°)", annotation_font_color="#00ff88")
                fig_el.add_hline(y=30, line_dash="dot", line_color="#00d4ff",
                                  annotation_text="Good (30°)", annotation_font_color="#00d4ff")
                fig_el.add_hline(y=op_min_el, line_dash="dot", line_color="#ff8800",
                                  annotation_text=f"Min threshold ({op_min_el:.0f}°)",
                                  annotation_font_color="#ff8800")
                fig_el.update_layout(
                    paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
                    font=dict(color="#94a3b8", family="Share Tech Mono"),
                    xaxis=dict(gridcolor="#1e3a5f", tickangle=-20),
                    yaxis=dict(gridcolor="#1e3a5f", title="Max Elevation (°)", range=[0, 95]),
                    height=350,
                )
                st.plotly_chart(fig_el, use_container_width=True)

            else:
                st.warning(
                    f"⚠️ No passes above **{op_min_el:.0f}°** elevation for **{op_name}** "
                    f"over the next **{op_hours}h** from ({op_lat:.2f}°, {op_lon:.2f}°).  \n\n"
                    "**Suggestions:**\n"
                    "- Try a longer search window (48–72 hours)\n"
                    "- Lower the minimum elevation (try 5°)\n"
                    "- GEO satellites (GOES, Himawari, Meteosat) never pass overhead\n"
                    "- LEO satellites (ISS, Starlink, NOAA) pass 4–6 times per day\n"
                    "- Try **Space Stations** group and select ISS for best results"
                )


# ── TAB 5: ANALYTICS ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">FLEET ANALYTICS & AI INSIGHTS</div>', unsafe_allow_html=True)

    ac1, ac2 = st.columns(2)

    with ac1:
        fig_pie = go.Figure(go.Pie(
            labels=["LEO (<2000km)", "MEO (2000–35786km)", "GEO (>35786km)"],
            values=[
                max(stats.get("leo_count", 1), 1),
                max(stats.get("meo_count", 0), 1),
                max(stats.get("geo_count", 0), 1),
            ],
            hole=0.55,
            marker=dict(
                colors=["#00ff88", "#00d4ff", "#7c3aed"],
                line=dict(color="#030712", width=2),
            ),
        ))
        fig_pie.update_layout(
            title="Orbit Zone Distribution",
            paper_bgcolor="#030712",
            font=dict(color="#94a3b8", family="Share Tech Mono"),
            legend=dict(bgcolor="#0d1b2a"),
            height=340,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ac2:
        fig_spd2 = go.Figure(go.Histogram(
            x=df["speed_km_s"], nbinsx=20,
            marker=dict(color="#00d4ff", line=dict(color="#030712", width=0.5)),
            opacity=0.85,
        ))
        fig_spd2.update_layout(
            title="Orbital Speed Distribution",
            paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
            font=dict(color="#94a3b8", family="Share Tech Mono"),
            xaxis=dict(title="Speed (km/s)", gridcolor="#1e3a5f"),
            yaxis=dict(title="Count", gridcolor="#1e3a5f"),
            height=340,
        )
        st.plotly_chart(fig_spd2, use_container_width=True)

    if "is_anomaly" in df.columns and df["is_anomaly"].any():
        st.markdown('<div class="section-header" style="margin-top:8px">AI-FLAGGED ANOMALIES (ISOLATION FOREST)</div>', unsafe_allow_html=True)
        adf = df[df["is_anomaly"]][["name", "alt_km", "speed_km_s", "lat", "lon", "anomaly_score"]].copy()
        adf.columns = ["Name", "Alt(km)", "Speed(km/s)", "Lat", "Lon", "Anomaly Score"]
        st.dataframe(
            adf.sort_values("Anomaly Score", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.markdown('<div class="section-header" style="margin-top:8px">ORBITAL PARAMETER MAP</div>', unsafe_allow_html=True)
    color_col = "risk_score" if "risk_score" in df.columns else "alt_km"
    fig_map = px.scatter(
        df, x="alt_km", y="speed_km_s", color=color_col,
        hover_name="name", color_continuous_scale="Plasma",
        title="Altitude vs Speed — All Tracked Objects",
        labels={"alt_km": "Alt (km)", "speed_km_s": "Speed (km/s)", color_col: "Risk"},
    )
    fig_map.update_layout(
        paper_bgcolor="#030712", plot_bgcolor="#0d1b2a",
        font=dict(color="#94a3b8", family="Share Tech Mono"),
        height=400,
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTO REFRESH — non-blocking
# ══════════════════════════════════════════════════════════════════════════════
if auto_refresh:
    # Store last refresh time in session_state to avoid blocking
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, 30 - int(elapsed))

    st.sidebar.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:0.62rem;'
        f'color:rgba(0,255,136,0.5);text-align:center;padding:8px;'
        f'border:1px solid rgba(0,255,136,0.1);border-radius:6px;'
        f'background:rgba(0,255,136,0.03);">'
        f'<span style="color:#00ff88">⟳</span> Next sync in '
        f'<span style="color:#00ff88;font-weight:700">{remaining}s</span></div>',
        unsafe_allow_html=True,
    )

    if elapsed >= 30:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()
    else:
        # Re-check every 5 seconds using meta refresh trick
        time.sleep(min(5, remaining + 0.1))
        st.rerun()
