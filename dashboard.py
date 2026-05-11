"""
Fartcoin Alpha — Trade Desk Dashboard (v3)

Two-speed framework: LGBM/LSTM model signals + systematic settlement signals.
Decision-first layout: identify scenario → run playbook.

4 tabs:
  1. 🎯 Trade Desk    — Scenario + dual signal cards + execution playbook
  2. 🤖 Signal Engine — Model deep dive (LGBM + LSTM + chart)
  3. 🌐 Market        — HMM regime + funding + OI/LSR + BTC + session
  4. 📓 Journal       — Equity by speed, win rates, trade log

Run: streamlit run dashboard.py --server.port 8501
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timezone
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

from market_state import (
    SESSION_MAP, HOURLY_BIAS, ASIA_SUB, ASIA_DAY_BPS,
    classify_session, classify_asia_sub, compute_market_state,
    determine_action, _next_positive_hour,
)
from projections import compute_projections
from alerts import evaluate_alerts, evaluate_projection_alerts
from coin_config import COIN_CONFIG, DEFAULT_COIN

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Alpha Trade Desk",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 0.8rem; padding-bottom: 1rem; }
  [data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.05rem !important; font-weight: 700; }
  button[data-baseweb="tab"] { font-size: 0.88rem; padding: 6px 16px; font-weight: 600; }
  .signal-card { border-radius: 10px; padding: 16px 20px; border: 1.5px solid; margin-bottom: 0; }
  .playbook-box { border-radius: 8px; padding: 16px 20px; background: #0d1117;
                  border: 1px solid #30363d; margin-top: 14px; }
  .scenario-badge { border-radius: 10px; padding: 20px 28px; margin-bottom: 16px; }
  .pill { display:inline-block; padding:2px 10px; border-radius:20px;
          font-size:0.75rem; font-weight:600; margin-right:4px; }
  .section-label { font-size:0.68rem; font-weight:700; letter-spacing:1.2px;
                   color:#546e7a; text-transform:uppercase; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"
st_autorefresh(interval=300_000, key="data_refresh")
utc_now = datetime.now(timezone.utc)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    _coin_options = list(COIN_CONFIG.keys())
    selected_coin = st.selectbox("Coin", _coin_options,
                                 index=_coin_options.index(DEFAULT_COIN))
    st.markdown("---")
    st.caption(f"🕐 {utc_now.strftime('%H:%M UTC')} · Auto-refresh 5min")

_cfg       = COIN_CONFIG[selected_coin]
_cmc_sym   = _cfg["cmc_symbol"]
_perp_sym  = _cfg["perp_symbol"]
_disp_name = _cfg["display_name"]
_emoji     = _cfg["emoji"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data(cmc_sym, perp_sym):
    data = {}
    index_files = {
        "ohlcv":              f"{cmc_sym}_ohlcv_hourly.csv",
        "ohlcv_daily":        f"{cmc_sym}_ohlcv.csv",
        "btc":                "bitcoin_cg_chart.csv",
        "funding":            f"{perp_sym}_funding.csv",
        "lsr":                f"{perp_sym}_lsr.csv",
        "oi":                 f"{perp_sym}_oi.csv",
        "taker":              f"{perp_sym}_taker.csv",
        "signals":            f"signals_{cmc_sym}.csv",
        "news_sentiment":     "news_sentiment_hourly.csv",
        "holder_concentration": "holder_concentration_history.csv",
        "exchange_flow":      "exchange_flow_history.csv",
        "cross_exchange_funding": "coinalyze_funding_history.csv",
        "predicted_funding":  "coinalyze_predicted_funding.csv",
        "liquidations":       "coinalyze_liquidations.csv",
        "derivatives_history": "derivatives_history.csv",
        "fear_greed":         "fear_greed_history.csv",
        "sentiment_history":  "sentiment_history.csv",
    }
    for key, fname in index_files.items():
        f = DATA_DIR / fname
        if f.exists():
            try:
                data[key] = pd.read_csv(f, index_col=0, parse_dates=True)
            except Exception:
                pass
    flat_files = {
        "derivatives": f"{cmc_sym}_derivatives_snapshot.csv",
        "trades":      "trades.csv",
    }
    for key, fname in flat_files.items():
        f = DATA_DIR / fname
        if f.exists():
            try:
                data[key] = pd.read_csv(f)
            except Exception:
                pass
    return data


data = load_all_data(_cmc_sym, _perp_sym)


@st.cache_data(ttl=300)
def compute_state(cmc_sym, _cache_key):
    _data   = load_all_data(cmc_sym, _perp_sym)
    _mkt    = compute_market_state(_data)
    _action = determine_action(_mkt)
    _proj   = compute_projections(_data, _mkt)
    _sa     = evaluate_alerts(_mkt, _action)
    _pa     = evaluate_projection_alerts(_proj, _mkt)
    return _mkt, _action, _proj, _sa + _pa


_ohlcv_key = str(data["ohlcv"].index[-1]) if "ohlcv" in data and not data["ohlcv"].empty else "none"
mkt, action, proj, all_alerts = compute_state(_cmc_sym, _ohlcv_key)

# Data freshness
_freshness_candidates = [
    DATA_DIR / "FARTCOIN_ohlcv_hourly.csv",
    DATA_DIR / "FARTCOINUSDT_oi.csv",
    DATA_DIR / "FARTCOINUSDT_lsr.csv",
    DATA_DIR / "FARTCOINUSDT_funding.csv",
]
_data_age_min = None
for _f in _freshness_candidates:
    if _f.exists():
        _age = (utc_now - datetime.fromtimestamp(
            os.path.getmtime(_f), tz=timezone.utc)).total_seconds() / 60
        if _data_age_min is None or _age < _data_age_min:
            _data_age_min = _age

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute shortcuts
# ─────────────────────────────────────────────────────────────────────────────

direction   = action["direction"]
conviction  = action["conviction"]
composite   = mkt["composite"]
session     = mkt["session"]
fart_price  = mkt["fart_price"]
btc_price   = mkt.get("btc_price", 0)
risk_score  = mkt.get("risk_score", 0)
avg_funding = mkt.get("avg_funding", 0)
total_oi    = mkt.get("total_oi", 0)
risk_label  = "HIGH" if risk_score >= 4 else "MOD" if risk_score >= 2 else "LOW"

opp    = proj.get("opportunity", {})
_hmm   = proj.get("hmm_regime",  {})
_sr    = proj.get("support_resistance", {})

_score         = opp.get("score", 0)
_tier          = opp.get("tier", "WATCH")
_size_pct      = opp.get("size_pct", 0)
_kelly_f       = opp.get("kelly_fraction", 0)
_meta_prob     = opp.get("meta_prob", 0.5)
_p4_prob       = opp.get("p4_prob")
_p8_prob       = opp.get("p8_prob")
_both_agree    = opp.get("both_agree")
_lstm_prob     = opp.get("lstm_prob")
_lstm_trade    = opp.get("lstm_trade", 0)
_triple_agree  = opp.get("triple_agreement", 0)

_hmm_regime = _hmm.get("regime_label", "STEADY_STATE")
_hmm_conf   = _hmm.get("confidence", 0)
_hmm_hours  = _hmm.get("hours_in_regime", 0)

_ns = _sr.get("nearest_support")    or {}
_nr = _sr.get("nearest_resistance") or {}

desk_setups  = proj.get("desk_setups", {})
_ds_active   = desk_setups.get("active_signals", [])
_ds_all      = desk_setups.get("signals", [])
_sys_active  = len(_ds_active) > 0
_sys_dir     = _ds_active[0].get("direction") if _ds_active else None
_next_settle = desk_setups.get("next_settlement", "—")
_mins_settle = desk_setups.get("mins_to_next", None)
_cur_fund    = desk_setups.get("current_funding_rate") or avg_funding

# Freshness badge
_fresh_str = ""
if _data_age_min is not None:
    if _data_age_min < 120:
        _fresh_str = f"Data {_data_age_min:.0f}m ago"
    elif _data_age_min < 240:
        _fresh_str = f"Data {_data_age_min/60:.1f}h ago"
    else:
        _fresh_str = f"⚠ Data {_data_age_min/60:.1f}h old"

# ─────────────────────────────────────────────────────────────────────────────
# Scenario classification  (the core of the two-speed framework)
# ─────────────────────────────────────────────────────────────────────────────

_model_active = _tier in ("TRADE", "HIGH CONVICTION", "FULL SEND")
_gated        = _tier in ("BLOCKED", "BLOCKED (SESSION)")
_hakai        = _hmm_regime == "HAKAI"

if _gated or _hakai:
    _scenario = "STAND_ASIDE"
elif _model_active and _sys_active:
    _scenario = "CONFLUENCE" if _sys_dir == direction else "CONFLICT"
elif _triple_agree or _tier == "FULL SEND":
    _scenario = "FULL_SEND"
elif _tier == "HIGH CONVICTION":
    _scenario = "MODEL_HC"
elif _tier == "TRADE":
    _scenario = "MODEL_TRADE"
elif _sys_active:
    _scenario = "SYSTEMATIC_ONLY"
elif _tier == "WATCH":
    _scenario = "WATCH"
else:
    _scenario = "STAND_ASIDE"

# Scenario display config
_SCENARIOS = {
    "CONFLUENCE":     {"label": "CONFLUENCE",           "icon": "⚡",
                       "color": "#00c853", "bg": "#001a09",
                       "desc": "Model signal + systematic setup align — highest conviction entry"},
    "FULL_SEND":      {"label": "FULL SEND",             "icon": "🟢",
                       "color": "#4caf50", "bg": "#0a1a0a",
                       "desc": "Triple ensemble agreement — LGBM 4h + 8h + LSTM all fire"},
    "MODEL_HC":       {"label": "HIGH CONVICTION",       "icon": "🔵",
                       "color": "#29b6f6", "bg": "#00111a",
                       "desc": "Model strong — both LGBM horizons agree above confidence threshold"},
    "MODEL_TRADE":    {"label": "MODEL SIGNAL",          "icon": "🟡",
                       "color": "#f9a825", "bg": "#1a1200",
                       "desc": "Model signal active — dual-horizon LGBM agreement, judgment execution"},
    "SYSTEMATIC_ONLY":{"label": "DESK SETUP",            "icon": "⚙️",
                       "color": "#ff7043", "bg": "#1a0a00",
                       "desc": "Rule-based systematic signal — mechanical execution, fixed size"},
    "WATCH":          {"label": "WATCH — NOT YET",       "icon": "👁",
                       "color": "#e65100", "bg": "#130900",
                       "desc": "Setup forming but not confirmed — do not enter, monitor only"},
    "CONFLICT":       {"label": "CONFLICTING SIGNALS",   "icon": "⚠️",
                       "color": "#ef5350", "bg": "#150000",
                       "desc": "Model and systematic signals disagree on direction — stand aside"},
    "STAND_ASIDE":    {"label": "STAND ASIDE",           "icon": "⛔",
                       "color": "#546e7a", "bg": "#111618",
                       "desc": "No edge — session gate, HAKAI regime, or below threshold"},
}
_scn = _SCENARIOS.get(_scenario, _SCENARIOS["STAND_ASIDE"])

# Playbook parameters
if _scenario == "CONFLUENCE":
    _play_dir    = direction
    _play_size   = min(_size_pct + 15, 40)
    _play_hold   = max(_size_pct // 10, 4)  # rough: larger positions = longer hold
    _play_style  = "Confluence execution — size up vs model-only, prioritise clean entry"
    _play_hold_h = 4
elif _scenario == "FULL_SEND":
    _play_dir    = direction
    _play_size   = _size_pct
    _play_hold_h = 8
    _play_style  = "Triple ensemble — highest conviction, consider 8h hold window"
elif _scenario in ("MODEL_HC", "MODEL_TRADE"):
    _play_dir    = direction
    _play_size   = _size_pct
    _play_hold_h = 4
    _play_style  = ("Judgment execution — check HMM regime before entry, adjust size if STEADY_STATE"
                    if _scenario == "MODEL_TRADE" else
                    "High conviction — full Kelly size, standard 4h hold")
elif _scenario == "SYSTEMATIC_ONLY":
    _sys_sig     = _ds_active[0] if _ds_active else {}
    _play_dir    = _sys_sig.get("direction", "LONG")
    _play_size   = _sys_sig.get("size_pct", 20)
    _play_hold_h = _sys_sig.get("hold_h", 4)
    _play_style  = "Mechanical execution — fixed size, do not adjust based on model or regime"
else:
    _play_dir    = direction
    _play_size   = 0
    _play_hold_h = 0
    _play_style  = ""

# Entry / stop / target
_entry      = fart_price
_ns_price   = _ns.get("price", 0)
_nr_price   = _nr.get("price", 0)
_stop_p     = _ns_price if _play_dir == "LONG" else _nr_price
_target_p   = _nr_price if _play_dir == "LONG" else _ns_price
_stop_pct   = (_stop_p   - _entry) / (_entry + 1e-9) * 100 if _stop_p   else 0
_target_pct = (_target_p - _entry) / (_entry + 1e-9) * 100 if _target_p else 0
_rr_val     = abs(_target_pct / (_stop_pct + 1e-9)) if _stop_pct else 0

# ─────────────────────────────────────────────────────────────────────────────
# Helper: probability bar
# ─────────────────────────────────────────────────────────────────────────────

def _prob_bar(prob, label, gated=False):
    if gated or prob is None:
        return (f'<div style="flex:1;text-align:center">'
                f'<div style="color:#444;font-size:0.75rem;margin-bottom:4px">{label}</div>'
                f'<div style="background:#1a1a1a;border-radius:4px;height:7px;width:100%"></div>'
                f'<div style="color:#444;font-size:0.85rem;margin-top:4px">— gated</div>'
                f'</div>')
    pct = int(prob * 100)
    c   = "#4caf50" if prob >= 0.50 else "#ef5350"
    tc  = "#a5d6a7" if prob >= 0.50 else "#ef9a9a"
    chk = "✓" if prob >= 0.50 else "✗"
    return (f'<div style="flex:1;text-align:center">'
            f'<div style="color:#aaa;font-size:0.75rem;margin-bottom:4px">{label}</div>'
            f'<div style="background:#1e1e1e;border-radius:4px;height:7px;width:100%">'
            f'<div style="background:{c};height:7px;border-radius:4px;width:{pct}%"></div></div>'
            f'<div style="color:{tc};font-size:0.88rem;font-weight:700;margin-top:4px">'
            f'{pct}% {chk}</div>'
            f'</div>')


def _severity_color(sev):
    return {"high": "#c62828", "medium": "#e65100", "low": "#1b5e20"}.get(sev, "#37474f")


def _playbook_cell(label, value, color="#fff"):
    return (f'<div style="background:#111618;border-radius:6px;padding:10px 12px">'
            f'<div style="font-size:0.68rem;color:#546e7a;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{color}">{value}</div>'
            f'</div>')


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATUS BAR (above all tabs)
# ─────────────────────────────────────────────────────────────────────────────

_time_col   = "#ff8f00" if _data_age_min and _data_age_min >= 240 else "#546e7a"
_hmm_col    = "#ef9a9a" if _hakai else "#a5d6a7" if _hmm_regime == "ACCUMULATION" else "#90caf9"
_risk_col   = "#ef9a9a" if risk_score >= 4 else "#f9a825" if risk_score >= 2 else "#a5d6a7"
_fund_col   = "#ef9a9a" if avg_funding > 0.0003 else "#f9a825" if avg_funding > 0.0001 else "#a5d6a7"

st.markdown(
    f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
    f'padding:8px 18px;margin-bottom:14px;display:flex;flex-wrap:wrap;gap:14px;'
    f'align-items:center;font-size:0.8rem">'
    f'<span style="color:#aaa">{_emoji} <b style="color:#fff">{_disp_name}</b> '
    f'<b style="color:#64b5f6">${fart_price:.5f}</b></span>'
    f'<span style="color:#555">|</span>'
    f'<span style="color:#aaa">BTC <b style="color:#fff">${btc_price:,.0f}</b></span>'
    f'<span style="color:#555">|</span>'
    f'<span style="color:#aaa">HMM <b style="color:{_hmm_col}">{_hmm_regime} {_hmm_conf:.0%}</b></span>'
    f'<span style="color:#555">|</span>'
    f'<span style="color:#aaa">Funding <b style="color:{_fund_col}">{avg_funding:.5f}</b></span>'
    f'<span style="color:#555">|</span>'
    f'<span style="color:#aaa">Session <b style="color:#fff">{session}</b></span>'
    f'<span style="color:#555">|</span>'
    f'<span style="color:#aaa">Risk <b style="color:{_risk_col}">{risk_label} {risk_score}/7</b></span>'
    f'<span style="color:#555">|</span>'
    f'<span style="color:{_time_col}">{_fresh_str or utc_now.strftime("%H:%M UTC")}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Alerts
for al in all_alerts:
    sev = al.get("severity", "low")
    c   = _severity_color(sev)
    lbl = {"high": "HIGH", "medium": "MED", "low": "LOW"}.get(sev, sev.upper())
    st.markdown(
        f'<div style="background:{c}22;border-left:4px solid {c};border-radius:6px;'
        f'padding:7px 14px;margin-bottom:4px;font-size:0.85rem">'
        f'<b style="color:{c}">[{lbl}]</b> {al["title"]}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_desk, tab_engine, tab_mkt, tab_journal = st.tabs([
    "🎯 Trade Desk",
    "🤖 Signal Engine",
    "🌐 Market",
    "📓 Journal",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRADE DESK
# ═════════════════════════════════════════════════════════════════════════════

with tab_desk:

    # ── 1a. Scenario badge ────────────────────────────────────────────────────
    _dir_icon = "📈" if _play_dir == "LONG" else ("📉" if _play_dir == "SHORT" else "⏸")
    _weekend  = " 🏖 WEEKEND" if mkt.get("is_weekend") else ""

    st.markdown(
        f'<div class="scenario-badge" style="background:{_scn["bg"]};'
        f'border:2px solid {_scn["color"]};border-radius:12px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        f'<div>'
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:1.2px;'
        f'color:{_scn["color"]};margin-bottom:4px">CURRENT SCENARIO</div>'
        f'<div style="font-size:2.2rem;font-weight:900;color:{_scn["color"]};letter-spacing:0.3px">'
        f'{_scn["icon"]} {_scn["label"]}{_weekend}</div>'
        f'<div style="color:#9e9e9e;font-size:0.9rem;margin-top:4px">{_scn["desc"]}</div>'
        f'</div>'
        f'<div style="text-align:right;min-width:160px">'
        + (
            f'<div style="font-size:2.8rem;font-weight:900;color:{_scn["color"]}">'
            f'{_dir_icon} {_play_dir}</div>'
            f'<div style="color:#aaa;font-size:0.85rem">'
            f'{_play_size}% position · {_play_hold_h}h hold</div>'
            if _play_size > 0 else
            f'<div style="font-size:2.0rem;font-weight:700;color:#546e7a">⏸ NO TRADE</div>'
            f'<div style="color:#546e7a;font-size:0.85rem">Flat — wait</div>'
        ) +
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 1b. Dual signal cards ─────────────────────────────────────────────────
    card_l, card_r = st.columns(2)

    # LEFT: Model signal
    with card_l:
        st.markdown('<div class="section-label">Speed 1 · Model Signal</div>',
                    unsafe_allow_html=True)

        _m_color  = "#4caf50" if _model_active else "#37474f"
        _m_bg     = "#0a1a0a" if _model_active else "#111618"
        _m_status = _tier if _tier else "NO TRADE"
        _m_dir_ic = ("📈" if direction == "LONG" else "📉") if _model_active else "⏸"
        _m_hit    = {"TRADE": "73.9%", "HIGH CONVICTION": "73.9%",
                     "FULL SEND": "97.7% (triple)"}.get(_tier, "—")
        _m_sharpe = {"TRADE": "5.11", "HIGH CONVICTION": "5.11",
                     "FULL SEND": "9.67 (triple)"}.get(_tier, "—")

        st.markdown(
            f'<div class="signal-card" style="background:{_m_bg};border-color:{_m_color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:10px">'
            f'<div>'
            f'<div style="color:{_m_color};font-weight:800;font-size:1rem">'
            f'{_m_dir_ic} {_m_status}</div>'
            f'<div style="color:#9e9e9e;font-size:0.75rem">LGBM dual-horizon + LSTM</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="color:#aaa;font-size:0.78rem">Kelly size</div>'
            f'<div style="color:#fff;font-weight:700;font-size:1.1rem">{_size_pct}%</div>'
            f'</div>'
            f'</div>'
            + (
                f'<div style="display:flex;gap:20px;margin-bottom:10px">'
                + _prob_bar(_p4_prob, "4h Model", _gated)
                + _prob_bar(_p8_prob, "8h Model", _gated)
                + (_prob_bar(_lstm_prob, "LSTM", _gated) if _lstm_prob is not None else "")
                + f'</div>'
                if _p4_prob is not None else ""
            ) +
            f'<div style="display:flex;gap:16px;font-size:0.8rem;color:#9e9e9e">'
            f'<span>Hist hit: <b style="color:#fff">{_m_hit}</b></span>'
            f'<span>Sharpe: <b style="color:#fff">{_m_sharpe}</b></span>'
            f'<span>Score: <b style="color:#fff">{_score}/100</b></span>'
            f'</div>'
            + (
                f'<div style="margin-top:8px;background:#0f200f;border-radius:6px;'
                f'padding:6px 10px;font-size:0.78rem;color:#a5d6a7;font-weight:600">'
                f'🟢 TRIPLE AGREEMENT — 97.7% hist. hit rate</div>'
                if _triple_agree and not _gated else ""
            ) +
            f'</div>',
            unsafe_allow_html=True,
        )

    # RIGHT: Systematic signal
    with card_r:
        st.markdown('<div class="section-label">Speed 2 · Systematic Setup</div>',
                    unsafe_allow_html=True)

        if _sys_active:
            _ss = _ds_active[0]
            _ss_dir   = _ss.get("direction", "LONG")
            _ss_col   = "#4caf50" if _ss_dir == "LONG" else "#ef5350"
            _ss_bg    = "#0a1a0a" if _ss_dir == "LONG" else "#1a0505"
            _ss_icon  = "📈" if _ss_dir == "LONG" else "📉"
            _ss_hit   = _ss.get("hit_rate", 0)
            _ss_sh    = _ss.get("sharpe", 0)
            _ss_size  = _ss.get("size_pct", 20)
            _ss_hold  = _ss.get("hold_h", 4)
            _ss_freq  = _ss.get("trades_per_month", "?")
            _ss_fund  = _ss.get("last_settle_rate") or _ss.get("current_funding_rate") or 0
            _ss_mins  = _ss.get("mins_since_settlement") or _ss.get("mins_to_settlement") or 0

            st.markdown(
                f'<div class="signal-card" style="background:{_ss_bg};border-color:{_ss_col}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-bottom:10px">'
                f'<div>'
                f'<div style="color:{_ss_col};font-weight:800;font-size:1rem">'
                f'{_ss_icon} {_ss.get("label","").upper()}</div>'
                f'<div style="color:#9e9e9e;font-size:0.75rem">Settlement-cycle rule · mechanical</div>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<div style="color:#aaa;font-size:0.78rem">Fixed size</div>'
                f'<div style="color:#fff;font-weight:700;font-size:1.1rem">{_ss_size}%</div>'
                f'</div>'
                f'</div>'
                f'<div style="background:#1a1a1a;border-radius:6px;padding:8px 12px;'
                f'margin-bottom:10px;font-size:0.8rem;color:#ccc">'
                f'{_ss.get("trigger","")}</div>'
                f'<div style="display:flex;gap:16px;font-size:0.8rem;color:#9e9e9e">'
                f'<span>Hit: <b style="color:#fff">{_ss_hit:.0%}</b></span>'
                f'<span>Sharpe: <b style="color:#fff">{_ss_sh:.2f}</b></span>'
                f'<span>Hold: <b style="color:#fff">{_ss_hold}h</b></span>'
                f'<span>~{_ss_freq}/mo</span>'
                f'</div>'
                f'<div style="margin-top:8px;font-size:0.75rem;color:#78909c">'
                f'Funding at trigger: {_ss_fund:.6f} · {_ss_mins:.0f}min timing</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            # Show inactive state + what would trigger
            _p95 = 0.000243
            _p99 = 0.000530
            _fund_pct_p95 = (_cur_fund / _p95 * 100) if _p95 else 0
            _fund_pct_p99 = (_cur_fund / _p99 * 100) if _p99 else 0

            st.markdown(
                f'<div class="signal-card" style="background:#111618;border-color:#263238">'
                f'<div style="color:#546e7a;font-weight:700;font-size:0.9rem;margin-bottom:10px">'
                f'⏸ NO SYSTEMATIC SETUP</div>'
                f'<div style="font-size:0.8rem;color:#78909c;margin-bottom:12px">'
                f'Next settlement: <b style="color:#b0bec5">{_next_settle}</b>'
                + (f' ({_mins_settle:.0f}min away)' if _mins_settle else '') +
                f'</div>'
                f'<div class="section-label" style="margin-bottom:6px">Trigger thresholds</div>'
                f'<div style="margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;'
                f'color:#aaa;margin-bottom:3px">'
                f'<span>📈 Post-Bounce (p95): {_p95:.6f}</span>'
                f'<span style="color:#78909c">{min(_fund_pct_p95, 100):.0f}% there</span>'
                f'</div>'
                f'<div style="background:#1e1e1e;border-radius:3px;height:5px">'
                f'<div style="background:#4caf50;height:5px;border-radius:3px;'
                f'width:{min(_fund_pct_p95,100):.0f}%"></div></div>'
                f'</div>'
                f'<div>'
                f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;'
                f'color:#aaa;margin-bottom:3px">'
                f'<span>📉 Extreme Fade (p99): {_p99:.6f}</span>'
                f'<span style="color:#78909c">{min(_fund_pct_p99, 100):.0f}% there</span>'
                f'</div>'
                f'<div style="background:#1e1e1e;border-radius:3px;height:5px">'
                f'<div style="background:#ef5350;height:5px;border-radius:3px;'
                f'width:{min(_fund_pct_p99,100):.0f}%"></div></div>'
                f'</div>'
                f'<div style="margin-top:10px;font-size:0.75rem;color:#546e7a">'
                f'Current funding: <b style="color:#b0bec5">{_cur_fund:.6f}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 1c. Confluence callout ────────────────────────────────────────────────
    if _scenario == "CONFLUENCE":
        _sys_sig_conf = _ds_active[0] if _ds_active else {}
        st.markdown(
            f'<div style="background:#001a09;border:1.5px solid #00c853;border-radius:8px;'
            f'padding:12px 18px;margin-top:10px">'
            f'<div style="color:#00c853;font-weight:800;font-size:0.9rem;margin-bottom:4px">'
            f'⚡ CONFLUENCE — BOTH SPEEDS AGREE</div>'
            f'<div style="color:#a5d6a7;font-size:0.85rem">'
            f'Model ({_tier}) + {_sys_sig_conf.get("label","Systematic")} both signal '
            f'<b>{direction}</b>. '
            f'Combined position: <b>{_play_size}%</b> (model {_size_pct}% + 15% systematic premium). '
            f'Historical confluence occurs ~2/month — treat as elevated priority.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif _scenario == "CONFLICT":
        st.markdown(
            f'<div style="background:#150000;border:1.5px solid #ef5350;border-radius:8px;'
            f'padding:12px 18px;margin-top:10px">'
            f'<div style="color:#ef5350;font-weight:800;font-size:0.9rem;margin-bottom:4px">'
            f'⚠️ SIGNAL CONFLICT — STAND ASIDE</div>'
            f'<div style="color:#ef9a9a;font-size:0.85rem">'
            f'Model says <b>{direction}</b> · Systematic says <b>{_sys_dir}</b>. '
            f'Conflicting signals cancel edge. Flat until one resolves.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 1d. Execution playbook ────────────────────────────────────────────────
    if _play_size > 0:
        st.markdown(
            f'<div class="playbook-box">'
            f'<div class="section-label">Execution Playbook</div>'
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;'
            f'margin-bottom:14px">'
            + _playbook_cell("Direction", f'{_dir_icon} {_play_dir}',
                             "#4caf50" if _play_dir == "LONG" else "#ef5350")
            + _playbook_cell("Size", f"{_play_size}%", _scn["color"])
            + _playbook_cell("Hold", f"{_play_hold_h}h", "#aaa")
            + (_playbook_cell("Entry", f"${_entry:.5f}", "#fff") if _entry else "")
            + (_playbook_cell("R/R", f"{_rr_val:.1f}x", "#f9a825") if _rr_val else "")
            + f'</div>'
            f'<div style="display:flex;gap:24px;font-size:0.83rem;color:#9e9e9e;'
            f'border-top:1px solid #21262d;padding-top:10px">'
            + (
                f'<span>🛑 Stop: <b style="color:#ef9a9a">${_stop_p:.5f} ({_stop_pct:+.1f}%)</b></span>'
                f'<span>🎯 Target: <b style="color:#a5d6a7">${_target_p:.5f} ({_target_pct:+.1f}%)</b></span>'
                if _stop_p and _target_p else ""
            ) +
            f'<span style="margin-left:auto;font-style:italic;color:#546e7a">'
            f'{_play_style}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 1e. Funding thermometer (always visible) ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Funding Rate · Settlement Trigger Monitor</div>',
                unsafe_allow_html=True)

    _p95_t = 0.000243
    _p99_t = 0.000530
    _bybit_floor = 0.000050
    _fund_display = _cur_fund if _cur_fund else 0
    _fund_pct_of_p99 = min(_fund_display / _p99_t * 100, 130) if _p99_t else 0

    # Zone: floor → p95 → p99 → max
    _zone_color = ("#ef5350" if _fund_display >= _p99_t else
                   "#f9a825" if _fund_display >= _p95_t else
                   "#4caf50" if _fund_display >= _bybit_floor else
                   "#546e7a")
    _zone_label = ("🔴 EXTREME — fade trigger" if _fund_display >= _p99_t else
                   "🟡 ELEVATED — bounce trigger" if _fund_display >= _p95_t else
                   "🟢 NORMAL" if _fund_display >= _bybit_floor else
                   "⚫ FLOOR")

    # Build bar HTML: 3 segments (floor→p95, p95→p99, p99+)
    _seg1_width = min(_fund_display / _p95_t * 60, 60)   # 0→60% of bar = floor to p95
    _seg2_width = min(max(_fund_display - _p95_t, 0) / (_p99_t - _p95_t) * 25, 25)  # 25% = p95→p99
    _seg3_width = min(max(_fund_display - _p99_t, 0) / _p99_t * 15, 15)  # 15% = beyond p99

    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
        f'padding:14px 18px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:10px">'
        f'<div style="font-size:1.6rem;font-weight:800;color:{_zone_color}">'
        f'{_fund_display:.6f}</div>'
        f'<div style="text-align:right">'
        f'<div style="color:{_zone_color};font-size:0.85rem;font-weight:700">{_zone_label}</div>'
        f'<div style="color:#546e7a;font-size:0.75rem">Next settle: {_next_settle}'
        + (f' · {_mins_settle:.0f}min away' if _mins_settle else '') +
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="position:relative;height:14px;background:#1e1e1e;border-radius:7px;'
        f'overflow:visible;margin-bottom:8px">'
        f'<div style="position:absolute;left:0;height:14px;border-radius:7px 0 0 7px;'
        f'width:{_seg1_width:.1f}%;background:linear-gradient(90deg,#37474f,#f9a825)"></div>'
        + (f'<div style="position:absolute;left:{_seg1_width:.1f}%;height:14px;'
           f'width:{_seg2_width:.1f}%;background:#f9a825"></div>'
           if _seg2_width > 0 else "") +
        (f'<div style="position:absolute;left:{_seg1_width+_seg2_width:.1f}%;height:14px;'
         f'border-radius:0 7px 7px 0;width:{_seg3_width:.1f}%;background:#ef5350"></div>'
         if _seg3_width > 0 else "") +
        f'<div style="position:absolute;left:60%;top:-18px;font-size:0.65rem;color:#f9a825">p95</div>'
        f'<div style="position:absolute;left:85%;top:-18px;font-size:0.65rem;color:#ef5350">p99</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#546e7a">'
        f'<span>Floor {_bybit_floor:.5f}</span>'
        f'<span style="color:#f9a825">p95 {_p95_t:.6f} → Bounce LONG</span>'
        f'<span style="color:#ef5350">p99 {_p99_t:.6f} → Extreme Fade SHORT</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 1f. Next opportunity (when no signal active) ──────────────────────────
    if _scenario in ("STAND_ASIDE", "WATCH") and not _sys_active:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">What Would Fire Next</div>',
                    unsafe_allow_html=True)
        nxt_l, nxt_r = st.columns(2)
        with nxt_l:
            _score_gap = max(0, 55 - _score)
            st.markdown(
                f'<div style="background:#111618;border-radius:8px;padding:14px 18px;'
                f'border-left:3px solid #37474f">'
                f'<div style="color:#546e7a;font-size:0.72rem;font-weight:700;'
                f'letter-spacing:1px;margin-bottom:8px">MODEL SIGNAL NEEDS</div>'
                f'<div style="color:#ccc;font-size:0.9rem">Score: <b>{_score}/100</b> '
                f'(needs <b style="color:#f9a825">55+</b> for TRADE)</div>'
                f'<div style="color:#9e9e9e;font-size:0.8rem;margin-top:4px">'
                f'+{_score_gap} pts to TRADE · '
                f'+{max(0, 70 - _score)} to HIGH CONVICTION</div>'
                f'<div style="background:#1e1e1e;border-radius:4px;height:6px;'
                f'margin-top:8px;width:100%">'
                f'<div style="background:#f9a825;height:6px;border-radius:4px;'
                f'width:{min(_score, 100)}%"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with nxt_r:
            _p95_gap   = max(0, _p95_t - _fund_display)
            _p99_gap   = max(0, _p99_t - _fund_display)
            _settle_str = f'{_mins_settle:.0f}min' if _mins_settle else '—'
            st.markdown(
                f'<div style="background:#111618;border-radius:8px;padding:14px 18px;'
                f'border-left:3px solid #37474f">'
                f'<div style="color:#546e7a;font-size:0.72rem;font-weight:700;'
                f'letter-spacing:1px;margin-bottom:8px">SYSTEMATIC SIGNAL NEEDS</div>'
                f'<div style="color:#ccc;font-size:0.9rem">'
                f'Next settle: <b style="color:#b0bec5">{_next_settle}</b> ({_settle_str})</div>'
                f'<div style="color:#9e9e9e;font-size:0.8rem;margin-top:4px">'
                f'Funding +{_p95_gap:.6f} → Bounce LONG<br>'
                f'Funding +{_p99_gap:.6f} + within 2h → Extreme Fade SHORT</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 1g. All systematic signal details (expander) ─────────────────────────
    if _ds_all:
        with st.expander("⚙️ All systematic signals — status detail"):
            for _sig in _ds_all:
                _ic = "🟢" if _sig.get("active") else "⚫"
                st.markdown(
                    f"{_ic} **{_sig.get('label', _sig['id'])}** "
                    f"({_sig.get('direction','?')}, {_sig.get('hold_h','?')}h hold, "
                    f"{_sig.get('size_pct',20)}% · "
                    f"hit {_sig.get('hit_rate',0):.0%} · Sharpe {_sig.get('sharpe',0):.2f}) — "
                    f"{_sig.get('reason', '')}"
                )



# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNAL ENGINE
# ═════════════════════════════════════════════════════════════════════════════

with tab_engine:
    signals = data.get("signals")
    ohlcv   = data.get("ohlcv")

    if signals is None or ohlcv is None:
        st.warning("Run signal_engine.py to generate signals.")
    else:
        price_col = "price" if "price" in ohlcv.columns else "close"

        # ── Model panel ───────────────────────────────────────────────────────
        st.markdown("#### Model Agreement Panel")
        eng_l, eng_r = st.columns([2, 1])

        with eng_l:
            _agree_bg    = "#0a1a0a" if _both_agree else "#1a0000"
            _agree_col   = "#4caf50" if _both_agree else "#ef5350"
            _agree_badge = "BOTH AGREE ✓✓" if _both_agree else "NOT ALIGNED ✗"
            if _gated:
                _agree_badge = "GATED"
                _agree_col   = "#546e7a"
                _agree_bg    = "#111618"
            elif _both_agree and _p4_prob and _p8_prob and min(_p4_prob, _p8_prob) >= 0.65:
                _agree_badge = "STRONG CONSENSUS ✓✓"
                _agree_col   = "#00c853"

            st.markdown(
                f'<div style="background:{_agree_bg};border-radius:10px;padding:16px 20px">'
                f'<div style="display:flex;align-items:center;gap:20px;margin-bottom:12px">'
                + _prob_bar(_p4_prob, "4h Model", _gated)
                + f'<div style="text-align:center;min-width:120px">'
                  f'<div style="color:{_agree_col};font-weight:800;font-size:0.85rem">'
                  f'{_agree_badge}</div>'
                  f'<div style="color:#666;font-size:0.75rem;margin-top:2px">Score {_score}/100</div>'
                  f'</div>'
                + _prob_bar(_p8_prob, "8h Model", _gated)
                + (_prob_bar(_lstm_prob, "LSTM-64", _gated) if _lstm_prob is not None else "")
                + f'</div>'
                + (f'<div style="background:#0f200f;border-radius:6px;padding:7px 12px;'
                   f'font-size:0.8rem;color:#a5d6a7;font-weight:600">'
                   f'🟢 TRIPLE AGREEMENT · 97.7% hist. hit · Sharpe 9.67</div>'
                   if _triple_agree and not _gated else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

        with eng_r:
            st.metric("Score", f"{_score}/100")
            st.metric("Tier", _tier)
            st.metric("Kelly", f"{_kelly_f:.0%}")
            st.metric("p (meta)", f"{_meta_prob:.0%}")

        # ── Why this call ─────────────────────────────────────────────────────
        _hakai_exit = _hmm.get("hakai_exit_h", 24)
        _reasons    = []
        if _hakai:
            _reasons.append(("🚫", f"HMM HAKAI ({_hmm_conf:.0%}, {_hmm_hours}h) — distribution, no entries"))
        elif _hmm_regime == "ACCUMULATION" and isinstance(_hakai_exit, (int,float)) and _hakai_exit <= 6:
            _reasons.append(("🔥", f"ACCUMULATION — fresh HAKAI exit {_hakai_exit:.0f}h ago. Prime window."))
        elif _hmm_regime == "ACCUMULATION":
            _reasons.append(("⬆️", f"ACCUMULATION ({_hmm_conf:.0%}, {_hmm_hours}h) — amplified conviction"))
        else:
            _reasons.append(("⚪", f"STEADY STATE ({_hmm_conf:.0%}) — neutral, raise bar to 60%+"))

        _comp_str = "bullish" if composite > 0.3 else "bearish" if composite < -0.3 else "weak"
        _reasons.append(("📊", f"Composite {composite:+.3f} — {_comp_str}"))

        sig_cols = [c for c in signals.columns if c.startswith("sig_")]
        sig_vals = {c: float(signals[c].dropna().iloc[-1]) for c in sig_cols
                    if len(signals[c].dropna()) > 0}
        _top_drv = opp.get("top_drivers", [])
        for feat, imp in (_top_drv[:3] if _top_drv else []):
            _fd = feat.replace("sig_", "").replace("_", " ").upper()
            _reasons.append(("🔑", f"Driver: {_fd} (importance {imp:.0f})"))

        if avg_funding > 0.0003:
            _reasons.append(("💰", f"Funding {avg_funding:.5f} — crowded longs, carry drag"))
        elif avg_funding < 0:
            _reasons.append(("💰", f"Funding {avg_funding:.5f} — shorts crowded, upward bias"))

        if _sr.get("available"):
            _ns_d = abs(_ns.get("distance_pct", 0))
            _nr_d = abs(_nr.get("distance_pct", 0))
            if _ns_d < 1.0:
                _reasons.append(("🟢", f"At support ${_ns.get('price',0):.5f} ({_ns_d:.1f}%)"))
            if _nr_d < 1.0:
                _reasons.append(("🔴", f"At resistance ${_nr.get('price',0):.5f} ({_nr_d:.1f}%)"))

        with st.expander("🧠 Why this call", expanded=True):
            for _ic, _tx in _reasons:
                st.markdown(f"{_ic} {_tx}")

        # ── Signal components ─────────────────────────────────────────────────
        with st.expander("📡 Signal components (drill-down)", expanded=False):
            if sig_vals:
                _sv_items = list(sig_vals.items())
                _sv_cols  = st.columns(min(len(_sv_items), 5))
                for _i, (_cn, _v) in enumerate(_sv_items[:10]):
                    _lbl = _cn.replace("sig_", "").replace("_", " ").title()
                    _ic  = "🟢" if _v > 0.2 else "🔴" if _v < -0.2 else "⚪"
                    _sv_cols[_i % len(_sv_cols)].metric(f"{_ic} {_lbl}", f"{_v:+.3f}")

        # ── Price + Composite + Volume chart ──────────────────────────────────
        st.markdown("#### Price · Composite · Volume")
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.3, 0.15], vertical_spacing=0.03,
            subplot_titles=[f"{_disp_name} Price", "Composite Signal", "Volume"],
        )
        fig.add_trace(go.Scatter(
            x=ohlcv.index, y=ohlcv[price_col], name="Price",
            line=dict(color="#64b5f6", width=1.5)), row=1, col=1)

        # S/R overlays
        if _sr.get("available", False):
            _cur_p  = _sr.get("current_price", 0)
            _levels = _sr.get("levels", [])
            _x0, _x1 = ohlcv.index[0], ohlcv.index[-1]
            _va_lo, _va_hi = _sr.get("value_area_low"), _sr.get("value_area_high")
            if _va_lo and _va_hi:
                fig.add_shape(type="rect", x0=_x0, x1=_x1, y0=_va_lo, y1=_va_hi,
                              xref="x", yref="y",
                              fillcolor="rgba(100,181,246,0.06)", line_width=0,
                              layer="below", row=1, col=1)
            for _lv in _levels:
                _lp  = _lv.get("price", 0)
                _lt  = _lv.get("type", "support")
                _ls  = _lv.get("strength", 0.5)
                _rgb = (f"rgba(239,154,154,{0.4+0.5*_ls:.2f})" if _lt == "resistance"
                        else f"rgba(165,214,167,{0.4+0.5*_ls:.2f})")
                fig.add_shape(type="line", x0=_x0, x1=_x1, y0=_lp, y1=_lp,
                              line=dict(color=_rgb, width=0.8+1.2*_ls,
                                        dash="solid" if _ls >= 0.7 else "dot"),
                              row=1, col=1)
                _dist = (_lp - _cur_p) / (_cur_p + 1e-9) * 100
                _lbl  = f"{'R' if _lt=='resistance' else 'S'} ${_lp:.4f} {_dist:+.1f}%"
                fig.add_annotation(x=_x1, y=_lp, xref="x", yref="y",
                                   text=_lbl, showarrow=False,
                                   font=dict(size=8, color=_rgb),
                                   bgcolor="rgba(15,20,30,0.7)",
                                   xanchor="right", yanchor="middle", row=1, col=1)

        comp_s = signals["composite"]
        fig.add_trace(go.Scatter(
            x=comp_s.index, y=comp_s, name="Composite",
            line=dict(color="#90caf9", width=1)), row=2, col=1)
        for level, color, dash in [(0.4,"#66bb6a","dash"),(0.2,"#a5d6a7","dot"),
                                    (-0.2,"#ef9a9a","dot"),(-0.4,"#e57373","dash")]:
            fig.add_hline(y=level, line_color=color, line_dash=dash, row=2, col=1)
        fig.add_hline(y=composite, line_color="yellow", line_dash="dot",
                      annotation_text=f"NOW {composite:+.3f}", row=2, col=1)
        fig.add_trace(go.Bar(
            x=ohlcv.index, y=ohlcv["volume"], name="Volume",
            marker_color="rgba(120,120,120,0.25)"), row=3, col=1)
        fig.update_layout(height=600, showlegend=False, hovermode="x unified",
                          template="plotly_dark", margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ── S/R detail ────────────────────────────────────────────────────────
        if _sr.get("available", False):
            with st.expander("🧱 Support & Resistance levels", expanded=False):
                _c1, _c2, _c3 = st.columns(3)
                if _ns:
                    _c1.metric("🟢 Nearest Support",   f"${_ns['price']:.5f}",
                               f"-{abs(_ns.get('distance_pct',0)):.2f}% · str={_ns.get('strength',0):.2f}")
                if _nr:
                    _c2.metric("🔴 Nearest Resistance", f"${_nr['price']:.5f}",
                               f"+{abs(_nr.get('distance_pct',0)):.2f}% · str={_nr.get('strength',0):.2f}")
                _c3.metric("⚖️ R/R", f"{_sr.get('risk_reward',0):.2f}x")
                _lv_rows = []
                for _lv in _sr.get("levels", []):
                    _lv_rows.append({
                        "Type":     _lv.get("type","").title(),
                        "Price":    f"${_lv['price']:.5f}",
                        "Dist%":    f"{(_lv['price']-fart_price)/fart_price*100:+.2f}%",
                        "Strength": f"{_lv.get('strength',0):.2f}",
                        "Touches":  _lv.get("touches", 0),
                        "Methods":  ", ".join(_lv.get("methods",[])),
                    })
                if _lv_rows:
                    st.dataframe(pd.DataFrame(_lv_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

with tab_mkt:
    mkt_tabs = st.tabs(["🧠 Regime & Funding", "📐 Derivatives", "₿ BTC & Correlation", "⏰ Session"])

    # ── 3a. Regime & Funding ─────────────────────────────────────────────────
    with mkt_tabs[0]:
        r1l, r1r = st.columns(2)

        with r1l:
            st.markdown("**HMM Regime**")
            _hmm_colors = {"HAKAI":"#b71c1c","ACCUMULATION":"#1b5e20","STEADY_STATE":"#1565c0"}
            _hmm_icons  = {"HAKAI":"🔴","ACCUMULATION":"🟢","STEADY_STATE":"🔵"}
            _hc = _hmm_colors.get(_hmm_regime, "#37474f")
            _hi = _hmm_icons.get(_hmm_regime, "⚪")
            st.markdown(
                f'<div style="background:{_hc}22;border:1.5px solid {_hc};border-radius:8px;'
                f'padding:16px 20px">'
                f'<div style="font-size:1.4rem;font-weight:800;color:{_hc}">'
                f'{_hi} {_hmm_regime}</div>'
                f'<div style="color:#aaa;margin-top:4px">Confidence: {_hmm_conf:.0%} · '
                f'{_hmm_hours}h in regime</div>'
                f'<div style="font-size:0.8rem;color:#9e9e9e;margin-top:6px">'
                f'{_hmm.get("description","")[:160]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            r1a, r1b, r1c = st.columns(3)
            r1a.metric("Regime", _hmm_regime.replace("_"," "))
            r1b.metric("Confidence", f"{_hmm_conf:.0%}")
            r1c.metric("Hours in", f"{_hmm_hours}h")

        with r1r:
            st.markdown("**Funding Rate**")
            funding_df = data.get("funding")
            if funding_df is not None and not funding_df.empty:
                _fund_col2 = "fundingRate" if "fundingRate" in funding_df.columns else funding_df.columns[0]
                _fund_recent = funding_df[_fund_col2].dropna().tail(60)
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    x=_fund_recent.index, y=_fund_recent.values,
                    fill="tozeroy", fillcolor="rgba(100,181,246,0.12)",
                    line=dict(color="#64b5f6", width=1.5), name="Funding"))
                fig_f.add_hline(y=0.000243, line_color="#f9a825", line_dash="dash",
                                annotation_text="p95 (Bounce)", line_width=1)
                fig_f.add_hline(y=0.000530, line_color="#ef5350", line_dash="dash",
                                annotation_text="p99 (Fade)", line_width=1)
                fig_f.add_hline(y=0, line_color="#546e7a", line_width=0.5)
                fig_f.update_layout(template="plotly_dark", height=220,
                                    margin=dict(t=10, b=10), showlegend=False,
                                    yaxis_title="Funding Rate")
                st.plotly_chart(fig_f, use_container_width=True)
            ra, rb, rc = st.columns(3)
            ra.metric("Avg Funding", f"{avg_funding:.5f}")
            rb.metric("vs p95",  f"{avg_funding/0.000243:.1f}x" if avg_funding else "0.0x")
            rc.metric("vs p99",  f"{avg_funding/0.000530:.1f}x" if avg_funding else "0.0x")

        st.markdown("---")

        # OI + LSR + BSR row
        st.markdown("**Open Interest · Long/Short Ratio · Buy/Sell Ratio**")
        oi_df  = data.get("oi")
        lsr_df = data.get("lsr")
        taker_df = data.get("taker")

        der1, der2, der3 = st.columns(3)
        if oi_df is not None and not oi_df.empty:
            _oi_col = [c for c in oi_df.columns if "oi" in c.lower() or "open" in c.lower()]
            if _oi_col:
                _oi_s = oi_df[_oi_col[0]].dropna().tail(168)
                fig_oi = go.Figure(go.Scatter(x=_oi_s.index, y=_oi_s.values,
                                              fill="tozeroy",
                                              fillcolor="rgba(100,181,246,0.1)",
                                              line=dict(color="#64b5f6", width=1.5)))
                fig_oi.update_layout(template="plotly_dark", height=180,
                                     margin=dict(t=10, b=10), title="Open Interest",
                                     showlegend=False)
                der1.plotly_chart(fig_oi, use_container_width=True)

        if lsr_df is not None and not lsr_df.empty:
            _lsr_col = [c for c in lsr_df.columns if "ratio" in c.lower() or "lsr" in c.lower()]
            if _lsr_col:
                _lsr_s = lsr_df[_lsr_col[0]].dropna().tail(168)
                fig_lsr = go.Figure(go.Scatter(x=_lsr_s.index, y=_lsr_s.values,
                                               line=dict(color="#a5d6a7", width=1.5)))
                fig_lsr.add_hline(y=1.0, line_color="#546e7a", line_dash="dot")
                fig_lsr.update_layout(template="plotly_dark", height=180,
                                      margin=dict(t=10, b=10), title="Long/Short Ratio",
                                      showlegend=False)
                der2.plotly_chart(fig_lsr, use_container_width=True)

        if taker_df is not None and not taker_df.empty:
            _bsr_col = [c for c in taker_df.columns if "ratio" in c.lower() or "bsr" in c.lower()]
            if _bsr_col:
                _bsr_s = taker_df[_bsr_col[0]].dropna().tail(168)
                _bsr_c = ["#66bb6a" if v > 0.5 else "#e57373" for v in _bsr_s.values]
                fig_bsr = go.Figure(go.Bar(x=_bsr_s.index, y=_bsr_s.values,
                                           marker_color=_bsr_c))
                fig_bsr.add_hline(y=0.5, line_color="#546e7a", line_dash="dot")
                fig_bsr.update_layout(template="plotly_dark", height=180,
                                      margin=dict(t=10, b=10), title="Buy/Sell Ratio",
                                      showlegend=False)
                der3.plotly_chart(fig_bsr, use_container_width=True)

    # ── 3b. Derivatives (exchange OI / funding divergence) ───────────────────
    with mkt_tabs[1]:
        deriv = data.get("derivatives")
        if deriv is None:
            st.info("No derivatives snapshot. Run automation.py to refresh.")
        else:
            active = deriv[deriv["open_interest_usd"] > 10_000].copy()
            st.subheader(f"Cross-Exchange Positioning — {len(active)} venues")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("**OI by Exchange**")
                oi_top = active.nlargest(12, "open_interest_usd")
                fig_oix = go.Figure(go.Bar(
                    y=oi_top["exchange"].str[:22], x=oi_top["open_interest_usd"]/1e6,
                    orientation="h", marker_color="#42a5f5",
                    text=[f"${v/1e6:.1f}M" for v in oi_top["open_interest_usd"]],
                    textposition="outside"))
                fig_oix.update_layout(template="plotly_dark", height=380,
                                      xaxis_title="OI ($M)", margin=dict(l=160, t=10))
                st.plotly_chart(fig_oix, use_container_width=True)
            with dc2:
                st.markdown("**Funding Divergence**")
                fr_s = active.sort_values("funding_rate")
                fig_fr = go.Figure(go.Bar(
                    y=fr_s["exchange"].str[:22], x=fr_s["funding_rate"],
                    orientation="h",
                    marker_color=["#e57373" if x < 0 else "#66bb6a" for x in fr_s["funding_rate"]]))
                fig_fr.add_vline(x=0, line_color="#546e7a")
                fig_fr.update_layout(template="plotly_dark",
                                     height=max(380, len(fr_s)*18),
                                     xaxis_title="Funding", margin=dict(l=160, t=10))
                st.plotly_chart(fig_fr, use_container_width=True)

            with st.expander("Full exchange table"):
                display_df = active[[
                    "exchange","funding_rate","open_interest_usd","volume_24h_usd"]
                ].sort_values("open_interest_usd", ascending=False)
                st.dataframe(display_df.style.format({
                    "funding_rate": "{:.6f}",
                    "open_interest_usd": "${:,.0f}",
                    "volume_24h_usd": "${:,.0f}",
                }), use_container_width=True)

    # ── 3c. BTC & Correlation ─────────────────────────────────────────────────
    with mkt_tabs[2]:
        btc_data = data.get("btc")
        ohlcv_d  = data.get("ohlcv")
        if btc_data is None or ohlcv_d is None:
            st.info("Missing BTC or OHLCV data.")
        else:
            btc_col  = "price" if "price" in btc_data.columns else "close"
            fart_col = "price" if "price" in ohlcv_d.columns else "close"
            btc_h    = btc_data[[btc_col]].resample("1h").last().dropna()
            btc_h.columns = ["btc_price"]
            btc_h["btc_ret"] = btc_h["btc_price"].pct_change()
            fart_h   = ohlcv_d[[fart_col]].resample("1h").last().dropna()
            fart_h.columns = ["fart_price"]
            fart_h["fart_ret"] = fart_h["fart_price"].pct_change()
            merged  = btc_h.join(fart_h, how="inner").dropna()
            corr    = merged["btc_ret"].corr(merged["fart_ret"])
            valid   = merged[["btc_ret","fart_ret"]].dropna()
            beta    = np.polyfit(valid["btc_ret"], valid["fart_ret"], 1)[0] if len(valid)>50 else 0
            merged["roll_corr"] = merged["btc_ret"].rolling(24).corr(merged["fart_ret"])

            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Correlation (all)", f"{corr:.3f}")
            bm2.metric("Beta", f"{beta:.2f}x")
            _btc_lead = proj.get("btc_lead_lag", {})
            bm3.metric("BTC 2h Return", f"{_btc_lead.get('btc_2h_ret',0)*100:+.2f}%")
            bm4.metric("7d BTC Corr",   f"{_btc_lead.get('btc_corr_7d',0):.3f}")

            fig_corr = go.Figure(go.Scatter(
                x=merged.index, y=merged["roll_corr"],
                line=dict(color="#64b5f6", width=1.5), fill="tozeroy",
                fillcolor="rgba(100,181,246,0.08)"))
            fig_corr.add_hline(y=0.7,  line_color="#66bb6a", line_dash="dash",
                               annotation_text="High corr")
            fig_corr.add_hline(y=0.3,  line_color="#f9a825", line_dash="dot")
            fig_corr.add_hline(y=0,    line_color="#546e7a")
            fig_corr.update_layout(template="plotly_dark", height=260,
                                   title="Rolling 24h BTC Correlation",
                                   margin=dict(t=30, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── 3d. Session ───────────────────────────────────────────────────────────
    with mkt_tabs[3]:
        ohlcv_s = data.get("ohlcv")
        if ohlcv_s is None:
            st.info("No OHLCV data.")
        else:
            price_col_s = "price" if "price" in ohlcv_s.columns else "close"
            df_s = ohlcv_s.copy()
            df_s["ret"]    = df_s[price_col_s].pct_change()
            df_s["hour"]   = df_s.index.hour
            df_s["session"] = df_s["hour"].apply(classify_session)

            st.markdown("#### Session Performance")
            scols = st.columns(4)
            _s_styles = {
                "bullish":     ("#1b5e20", "✅ FAVORABLE"),
                "conditional": ("#1565c0", "🔀 CONDITIONAL"),
                "neutral":     ("#e65100", "⚠️ VOLATILE"),
                "bearish":     ("#c62828", "⛔ AVOID"),
            }
            for i, (sess, info) in enumerate(SESSION_MAP.items()):
                fc, label = _s_styles.get(info["bias"], ("#37474f", ""))
                scols[i].markdown(
                    f'<div style="background:{fc}22;padding:14px;border-radius:8px;'
                    f'border-left:4px solid {fc}">'
                    f'<b style="color:#fff">{sess}</b><br>'
                    f'<span style="color:#aaa;font-size:0.78rem">{info["et"]} ET</span><br>'
                    f'<span style="font-size:1.5rem;font-weight:800;color:{fc}">'
                    f'{info["avg_bps"]:+.1f}<span style="font-size:0.8rem"> bps/hr</span></span><br>'
                    f'<span style="font-size:0.72rem;color:{fc}">{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

            now_h      = utc_now.hour
            hourly_ret = df_s.groupby("hour")["ret"].mean() * 10000
            hourly_vol = df_s.groupby("hour")["ret"].apply(lambda x: x.abs().mean()) * 100
            sc1, sc2   = st.columns(2)
            with sc1:
                st.markdown("#### Avg Return by Hour (UTC)")
                fig_r = go.Figure(go.Bar(
                    x=hourly_ret.index, y=hourly_ret.values,
                    marker_color=["#66bb6a" if v>5 else "#e57373" if v<-10 else "#90a4ae"
                                  for v in hourly_ret]))
                fig_r.add_vline(x=now_h, line_color="#ffeb3b", line_width=2,
                                annotation_text=f"NOW ({now_h:02d}h)")
                fig_r.update_layout(template="plotly_dark", height=280,
                                    xaxis_title="Hour UTC", yaxis_title="bps",
                                    xaxis=dict(dtick=1), margin=dict(t=10,b=10))
                st.plotly_chart(fig_r, use_container_width=True)
            with sc2:
                st.markdown("#### Volatility by Hour")
                fig_v = go.Figure(go.Bar(x=hourly_vol.index, y=hourly_vol.values,
                                         marker_color="#ffb74d"))
                fig_v.add_vline(x=now_h, line_color="#ffeb3b", line_width=2)
                fig_v.update_layout(template="plotly_dark", height=280,
                                    xaxis_title="Hour UTC", yaxis_title="|Move|%",
                                    xaxis=dict(dtick=1), margin=dict(t=10,b=10))
                st.plotly_chart(fig_v, use_container_width=True)

            st.markdown("---")
            st.markdown("""
#### Timing Reference
| Rule | Hour (UTC) | ET | Edge |
|------|-----------|-----|------|
| ⭐ **Best entry** | 10:00 UTC | 6:00 AM | +23 bps bias |
| ✅ **Strong window** | 19:00–22:00 UTC | 3–6 PM | +15–19 bps avg |
| 🔴 **Kill zone #1** | 18:00 UTC | 2:00 PM | −0.46% avg ALL days |
| 🔴 **Kill zone #2** | 03:00 Thu | 11 PM Wed | −0.97% avg |
| ⚠️ **Thursday** | All day | — | −0.63% avg 4h, 41% hit |
| ⚠️ **Sunday** | All day | — | −0.40% avg 4h, 44% hit |
""")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — JOURNAL
# ═════════════════════════════════════════════════════════════════════════════

with tab_journal:
    _jpath = DATA_DIR / "trade_journal.csv"
    if not _jpath.exists():
        st.info("No trade journal yet. Signals will be logged after the first pipeline run.")
    else:
        try:
            jdf = pd.read_csv(_jpath)
            jdf_res = jdf[jdf["outcome_4h"].astype(str).str.strip() != ""].copy()
            jdf_res["outcome_4h"] = pd.to_numeric(jdf_res["outcome_4h"], errors="coerce")
            jdf_res["timestamp"]  = pd.to_datetime(jdf_res["timestamp"], utc=True, errors="coerce")
            jdf_res = jdf_res.dropna(subset=["outcome_4h","timestamp"]).sort_values("timestamp")

            # Speed classification
            _model_tiers = {"TRADE", "HIGH CONVICTION", "FULL SEND"}
            jdf_res["speed"] = jdf_res["tier"].apply(
                lambda t: "Speed 1 (Model)" if t in _model_tiers else "Speed 2 (Systematic)"
            )

            # ── Summary metrics ───────────────────────────────────────────────
            j1, j2, j3, j4 = st.columns(4)
            _total = len(jdf_res)
            _wr    = (jdf_res["outcome_4h"] > 0).mean()
            _avg   = jdf_res["outcome_4h"].mean()
            _cum   = jdf_res["outcome_4h"].sum()
            j1.metric("Total Resolved", _total)
            j2.metric("Overall Hit Rate", f"{_wr:.0%}")
            j3.metric("Avg Return / Trade", f"{_avg:+.2f}%")
            j4.metric("Cumulative P&L", f"{_cum:+.2f}%")

            st.markdown("---")

            # ── Equity curve: Speed 1 vs Speed 2 ─────────────────────────────
            st.markdown("#### Equity Curve — Speed 1 (Model) vs Speed 2 (Systematic)")
            st.caption("Speed 2 systematic signals not yet tracked in journal — shown when live trades are logged.")

            fig_eq = go.Figure()

            _speed1 = jdf_res[jdf_res["speed"] == "Speed 1 (Model)"].copy()
            _speed2 = jdf_res[jdf_res["speed"] == "Speed 2 (Systematic)"].copy()

            # All entries baseline
            fig_eq.add_trace(go.Scatter(
                x=jdf_res["timestamp"], y=jdf_res["outcome_4h"].cumsum(),
                name="All signals", line=dict(color="#546e7a", width=1.2, dash="dot")))

            # Speed 1 tiers
            _tier_colors = {
                "TRADE":          "#f9a825",
                "HIGH CONVICTION": "#29b6f6",
                "FULL SEND":      "#66bb6a",
            }
            for _t, _tc in _tier_colors.items():
                _sub = _speed1[_speed1["tier"] == _t]
                if len(_sub) >= 2:
                    _sub_idx = _sub.set_index("timestamp")
                    fig_eq.add_trace(go.Scatter(
                        x=_sub_idx.index, y=_sub_idx["outcome_4h"].cumsum(),
                        name=f"Speed 1 · {_t}",
                        line=dict(color=_tc, width=2.2)))

            # Speed 2 (when available)
            if len(_speed2) >= 2:
                fig_eq.add_trace(go.Scatter(
                    x=_speed2["timestamp"], y=_speed2["outcome_4h"].cumsum(),
                    name="Speed 2 · Systematic",
                    line=dict(color="#ff7043", width=2.2, dash="dash")))

            fig_eq.update_layout(
                height=300, template="plotly_dark",
                margin=dict(l=40, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=dict(gridcolor="#1e2a30"),
                yaxis=dict(gridcolor="#1e2a30", ticksuffix="%"),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── Rolling stats ─────────────────────────────────────────────────
            if len(jdf_res) >= 5:
                _n_roll = min(20, len(jdf_res))
                _recent = jdf_res.tail(_n_roll)
                _rwr    = (_recent["outcome_4h"] > 0).mean()
                _wins   = jdf_res["outcome_4h"][jdf_res["outcome_4h"] > 0]
                _losses = jdf_res["outcome_4h"][jdf_res["outcome_4h"] <= 0]
                _aw     = float(_wins.mean()) if len(_wins) else 0
                _al     = float(abs(_losses.mean())) if len(_losses) else 0.001

                rs1, rs2, rs3, rs4 = st.columns(4)
                rs1.metric(f"Rolling WR (last {_n_roll})", f"{_rwr:.0%}")
                rs2.metric("Avg Win",     f"+{_aw:.2f}%")
                rs3.metric("Avg Loss",    f"-{_al:.2f}%")
                rs4.metric("W/L Ratio",   f"{_aw/_al:.2f}x")

            st.markdown("---")

            # ── Scenario performance table ─────────────────────────────────────
            st.markdown("#### Performance by Tier")
            _perf_rows = []
            for _t in jdf_res["tier"].dropna().unique():
                _sub = jdf_res[jdf_res["tier"] == _t]
                _n   = len(_sub)
                _h   = (_sub["outcome_4h"] > 0).mean()
                _a   = _sub["outcome_4h"].mean()
                _e   = _sub["outcome_4h"] - 0.0045
                _sh  = (_e.mean()/_e.std()*np.sqrt(365*24/4)
                        if _e.std() > 0 else 0)
                _spd = "Speed 1" if _t in _model_tiers else "Speed 2"
                _perf_rows.append({
                    "Speed":  _spd,
                    "Tier":   _t,
                    "Trades": _n,
                    "Hit %":  f"{_h:.0%}",
                    "Avg Ret":f"{_a:+.2f}%",
                    "Sharpe": f"{_sh:+.2f}",
                })
            if _perf_rows:
                _pf = pd.DataFrame(_perf_rows).sort_values(["Speed","Tier"])
                st.dataframe(_pf, use_container_width=True, hide_index=True)

            # ── Trade log ─────────────────────────────────────────────────────
            with st.expander("📋 Full trade log"):
                _log_cols = [c for c in ["timestamp","tier","direction","score",
                                         "outcome_4h","hit","speed"]
                             if c in jdf_res.columns]
                st.dataframe(jdf_res[_log_cols].sort_values(
                    "timestamp", ascending=False
                ).head(100), use_container_width=True, hide_index=True)

        except Exception as _je:
            st.error(f"Journal load error: {_je}")
