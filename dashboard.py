"""
Fartcoin Alpha — Trade Desk Dashboard (v2)

Redesigned for analyst usability:
- Live alert bar at top
- All projections (including external data) in one place
- Consistent dark theme, scannable layout

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

# Inject a tighter global style
st.markdown("""
<style>
  /* Reduce default Streamlit padding */
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  /* Metric label smaller */
  [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; font-weight: 700; }
  /* Tab styling */
  button[data-baseweb="tab"] { font-size: 0.85rem; padding: 6px 14px; }
  /* Alert card base */
  .alert-card { border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 0.9rem; }
  .proj-card  { border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; border-left: 4px solid; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"

# Auto-refresh every 5 minutes
st_autorefresh(interval=300_000, key="data_refresh")

utc_now = datetime.now(timezone.utc)

# ─────────────────────────────────────────────────────────────────────────────
# Coin selector (sidebar)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Coin")
    _coin_options = list(COIN_CONFIG.keys())
    selected_coin = st.selectbox(
        "Select coin",
        _coin_options,
        index=_coin_options.index(DEFAULT_COIN),
        label_visibility="collapsed",
    )

_cfg        = COIN_CONFIG[selected_coin]
_cmc_sym    = _cfg["cmc_symbol"]
_perp_sym   = _cfg["perp_symbol"]
_disp_name  = _cfg["display_name"]
_emoji      = _cfg["emoji"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (includes all external collector files)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data(cmc_sym, perp_sym):
    data = {}

    # Core files (index = timestamp)
    index_files = {
        "ohlcv":              f"{cmc_sym}_ohlcv_hourly.csv",
        "ohlcv_daily":        f"{cmc_sym}_ohlcv.csv",
        "btc":                "bitcoin_cg_chart.csv",
        "funding":            f"{perp_sym}_funding.csv",
        "lsr":                f"{perp_sym}_lsr.csv",
        "oi":                 f"{perp_sym}_oi.csv",
        "taker":              f"{perp_sym}_taker.csv",
        "signals":            f"signals_{cmc_sym}.csv",
        # External collector files
        "news_sentiment":         "news_sentiment_hourly.csv",
        "holder_concentration":   "holder_concentration_history.csv",
        "exchange_flow":          "exchange_flow_history.csv",
        "cross_exchange_funding":  "coinalyze_funding_history.csv",
        "predicted_funding":      "coinalyze_predicted_funding.csv",
        "liquidations":           "coinalyze_liquidations.csv",
        "derivatives_history":    "derivatives_history.csv",
        "fear_greed":             "fear_greed_history.csv",
        "sentiment_history":      "sentiment_history.csv",
    }
    for key, fname in index_files.items():
        f = DATA_DIR / fname
        if f.exists():
            try:
                data[key] = pd.read_csv(f, index_col=0, parse_dates=True)
            except Exception:
                pass

    # Flat CSV files (no datetime index)
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

# Computed state
mkt    = compute_market_state(data)
action = determine_action(mkt)
proj   = compute_projections(data, mkt)

# Active alerts (both signal-level and projection-level)
_sig_alerts  = evaluate_alerts(mkt, action)
_proj_alerts = evaluate_projection_alerts(proj, mkt)
all_alerts   = _sig_alerts + _proj_alerts

# Data freshness
_signals_file = DATA_DIR / f"signals_{_cmc_sym}.csv"
_data_age_min = None
if _signals_file.exists():
    _mtime = datetime.fromtimestamp(os.path.getmtime(_signals_file), tz=timezone.utc)
    _data_age_min = (utc_now - _mtime).total_seconds() / 60


# ─────────────────────────────────────────────────────────────────────────────
# Helper: colored projection card
# ─────────────────────────────────────────────────────────────────────────────

def _proj_card(title, body, color="#1565c0", icon=""):
    st.markdown(
        f'<div class="proj-card" style="border-color:{color};background:{color}18">'
        f'<b>{icon} {title}</b><br><span style="color:#ccc">{body}</span></div>',
        unsafe_allow_html=True,
    )


def _severity_color(sev):
    return {"high": "#c62828", "medium": "#e65100", "low": "#1b5e20"}.get(sev, "#37474f")


# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE — shortcuts used across header + all tabs
# ─────────────────────────────────────────────────────────────────────────────

direction  = action["direction"]
conviction = action["conviction"]
composite  = mkt["composite"]
session    = mkt["session"]
fart_price = mkt["fart_price"]
btc_price  = mkt.get("btc_price", 0)
risk_score = mkt.get("risk_score", 0)
risk_label = "HIGH" if risk_score >= 4 else "MOD" if risk_score >= 2 else "LOW"
avg_funding= mkt.get("avg_funding", 0)
total_oi   = mkt.get("total_oi", 0)

# Projection shortcuts (available globally so Tab 1 can use them too)
opp        = proj.get("opportunity", {})
_hmm       = proj.get("hmm_regime", {})
_sr        = proj.get("support_resistance", {})

_score     = opp.get("score", 0)
_tier      = opp.get("tier", "WATCH")
_size_pct  = opp.get("size_pct", 0)
_kelly_f   = opp.get("kelly_fraction", 0)
_meta_prob = opp.get("meta_prob", 0.5)
_top_drv   = opp.get("top_drivers", [])

_hmm_regime = _hmm.get("regime_label", "STEADY_STATE")
_hmm_conf   = _hmm.get("confidence", 0)
_hmm_hours  = _hmm.get("hours_in_regime", 0)

_ns  = _sr.get("nearest_support")   or {}
_nr  = _sr.get("nearest_resistance") or {}
_rr  = _sr.get("risk_reward", 1.0)

# ── Unified action state (single truth for the desk) ─────────────────────────
# Combines meta-model tier + direction into one clear instruction.
if _tier in ("BLOCKED", "BLOCKED (SESSION)"):
    if _hmm_regime == "HAKAI":
        _brief_label  = "STAND ASIDE"
        _brief_reason = (f"HAKAI regime active ({_hmm_conf:.0%} conf, {_hmm_hours}h). "
                         f"Distribution phase — smart money exiting. No new longs or shorts.")
    else:
        _brief_label  = "STAND ASIDE"
        _brief_reason = (f"Session gate ({session}). This window has no historical edge. "
                         f"Wait for Asia / London / NYC open.")
    _brief_icon  = "⛔"
    _brief_color = "#7f0000"
    _brief_bg    = "#2d0000"
elif _tier == "PASS":
    _brief_label  = "NO TRADE"
    _brief_reason = f"Signal below threshold (score {_score}/100). Not enough edge to cover carry cost. Stay flat."
    _brief_icon  = "⏸"
    _brief_color = "#546e7a"
    _brief_bg    = "#1a2226"
elif _tier == "WATCH":
    _brief_label  = "WATCH — NOT YET"
    _brief_reason = f"Setup forming but not confirmed (score {_score}/100). Monitor — do not enter yet."
    _brief_icon  = "👁"
    _brief_color = "#e65100"
    _brief_bg    = "#1a1000"
elif direction == "LONG" and _tier in ("TRADE", "HIGH CONVICTION", "FULL SEND"):
    _brief_label  = "LONG"
    _brief_reason = f"{conviction} conviction. Score {_score}/100 — {_tier}. Kelly size: {_size_pct}%."
    _brief_icon  = "⬆"
    _brief_color = "#1b5e20"
    _brief_bg    = "#001200"
elif direction == "SHORT" and _tier in ("TRADE", "HIGH CONVICTION", "FULL SEND"):
    _brief_label  = "SHORT"
    _brief_reason = f"{conviction} conviction. Score {_score}/100 — {_tier}. Kelly size: {_size_pct}%."
    _brief_icon  = "⬇"
    _brief_color = "#b71c1c"
    _brief_bg    = "#1a0000"
else:
    _brief_label  = "NO TRADE"
    _brief_reason = f"Composite {composite:+.3f} — no directional edge. Wait for confirmation."
    _brief_icon  = "⏸"
    _brief_color = "#546e7a"
    _brief_bg    = "#1a2226"

# ── Trade setup levels (entry / stop / target) ────────────────────────────────
_entry  = fart_price
_stop   = _ns.get("price", 0) if _brief_label == "LONG"  else _nr.get("price", 0)
_target = _nr.get("price", 0) if _brief_label == "LONG"  else _ns.get("price", 0)
_stop_pct   = (_stop   - _entry) / (_entry + 1e-9) * 100 if _stop   else 0
_target_pct = (_target - _entry) / (_entry + 1e-9) * 100 if _target else 0
_setup_rr   = abs(_target_pct / (_stop_pct + 1e-9)) if _stop_pct else 0

# ── Freshness badge ───────────────────────────────────────────────────────────
_fresh_str = ""
if _data_age_min is not None:
    _fresh_str = (f"Data {_data_age_min:.0f}m ago" if _data_age_min < 35
                  else f"⚠ Data {_data_age_min:.0f}m old — refresh needed")

# ─────────────────────────────────────────────────────────────────────────────
# TRADE DESK BRIEF — the single card the desk reads first
# ─────────────────────────────────────────────────────────────────────────────

_weekend_note  = " 🏖 WEEKEND" if mkt.get("is_weekend") else ""
_hmm_pill_col  = "#ef9a9a" if _hmm_regime == "HAKAI" else "#a5d6a7" if _hmm_regime == "ACCUMULATION" else "#90caf9"
_fund_pill_col = "#ef9a9a" if avg_funding > 0.3 else "#a5d6a7" if avg_funding < 0 else "#ffffff"
_risk_pill_col = "#ef9a9a" if risk_score >= 4 else "#f9a825" if risk_score >= 2 else "#a5d6a7"
_time_pill     = utc_now.strftime("%H:%M UTC") + (f" · {_fresh_str}" if _fresh_str else "")
_stale_html    = (
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#ff8f00">⚠ {_fresh_str}</span>'
    if _data_age_min and _data_age_min >= 35 else
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">{_time_pill}</span>'
)
_levels_html = ""
if _brief_label in ("LONG", "SHORT") and _stop and _target:
    _levels_html = (
        f'<div style="margin-top:12px;display:flex;gap:24px;font-size:0.88rem">'
        f'<span style="color:#aaa">Entry <b style="color:#fff">${_entry:.5f}</b></span>'
        f'<span style="color:#aaa">Stop <b style="color:#ef9a9a">${_stop:.5f} ({_stop_pct:+.1f}%)</b></span>'
        f'<span style="color:#aaa">Target <b style="color:#a5d6a7">${_target:.5f} ({_target_pct:+.1f}%)</b></span>'
        f'<span style="color:#aaa">R/R <b style="color:#fff">{_setup_rr:.1f}x</b></span>'
        f'</div>'
    )

_brief_html = (
    f'<div style="background:{_brief_bg};border:2px solid {_brief_color};border-radius:12px;padding:18px 24px;margin-bottom:14px">'
    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
    f'<div>'
    f'<div style="font-size:2rem;font-weight:900;color:{_brief_color};letter-spacing:0.5px">{_brief_icon} {_brief_label}{_weekend_note}</div>'
    f'<div style="color:#ccc;font-size:0.95rem;margin-top:4px">{_brief_reason}</div>'
    f'{_levels_html}'
    f'</div>'
    f'<div style="text-align:right;min-width:140px">'
    f'<div style="font-size:2.5rem;font-weight:900;color:{_brief_color}">{_size_pct}<span style="font-size:1rem;color:#888">% size</span></div>'
    f'<div style="color:#aaa;font-size:0.8rem">Score {_score}/100 · {_tier}</div>'
    f'<div style="color:#aaa;font-size:0.8rem">Kelly {_kelly_f:.0%} · p={_meta_prob:.0%}</div>'
    f'</div>'
    f'</div>'
    f'<div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:8px;font-size:0.78rem">'
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">HMM: <b style="color:{_hmm_pill_col}">{_hmm_regime} {_hmm_conf:.0%}</b></span>'
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">Session: <b style="color:#fff">{session}</b></span>'
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">FART: <b style="color:#64b5f6">${fart_price:.5f}</b></span>'
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">BTC: <b style="color:#fff">${btc_price:,.0f}</b></span>'
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">Funding: <b style="color:{_fund_pill_col}">{avg_funding:.4f}</b></span>'
    f'<span style="background:#263238;padding:3px 10px;border-radius:20px;color:#b0bec5">Risk: <b style="color:{_risk_pill_col}">{risk_label} {risk_score}/7</b></span>'
    f'{_stale_html}'
    f'</div>'
    f'</div>'
)
st.markdown(_brief_html, unsafe_allow_html=True)

# ── Alerts bar ────────────────────────────────────────────────────────────────
if all_alerts:
    for al in all_alerts:
        sev   = al.get("severity", "low")
        color = _severity_color(sev)
        label = {"high": "HIGH", "medium": "MED", "low": "LOW"}.get(sev, sev.upper())
        st.markdown(
            f'<div class="alert-card" style="background:{color}22;border-left:4px solid {color};margin-top:6px">'
            f'<b style="color:{color}">[{label}]</b> {al["title"]}</div>',
            unsafe_allow_html=True,
        )

# ── Inline notes ──────────────────────────────────────────────────────────────
_notes = []
if action.get("funding_note"):  _notes.append(("💰", action["funding_note"], "warning"))
if action.get("btc_note"):      _notes.append(("₿",  action["btc_note"],     "info"))
if action.get("session_note"):  _notes.append(("⏰", action["session_note"],  "error"))
if action.get("asia_note"):     _notes.append(("🌏", action["asia_note"],    "info"))
if action.get("weekend_note"):  _notes.append(("🏖️", action["weekend_note"], "warning"))
for _ic, _nt, _kd in _notes:
    getattr(st, _kd)(f"{_ic} {_nt}")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_signal, tab_proj, tab_timing, tab_exchange, tab_btc, tab_rules = st.tabs([
    "📊 Signal",
    "🔮 Projections",
    "⏰ Timing",
    "🏦 Exchange",
    "₿ BTC",
    "📋 Rules",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

with tab_signal:
    signals = data.get("signals")
    ohlcv   = data.get("ohlcv")

    if signals is None or ohlcv is None:
        st.warning("Run signal_engine.py to generate signals.")
    else:
        price_col = "price" if "price" in ohlcv.columns else "close"

        # ── Why this call? (key reasons) ─────────────────────────────────────
        sig_cols = [c for c in signals.columns if c.startswith("sig_")]
        sig_vals = {c: float(signals[c].dropna().iloc[-1]) for c in sig_cols
                    if len(signals[c].dropna()) > 0}

        _reasons = []
        # HMM regime
        _hakai_exit = _hmm.get("hakai_exit_h", 24)
        if _hmm_regime == "HAKAI":
            _reasons.append(("🚫", f"HMM HAKAI ({_hmm_conf:.0%}, {_hmm_hours}h) — distribution phase, smart money exiting"))
        elif _hmm_regime == "ACCUMULATION" and isinstance(_hakai_exit, (int, float)) and _hakai_exit <= 6:
            _reasons.append(("🔥", f"HMM ACCUMULATION — fresh HAKAI exit {_hakai_exit:.0f}h ago ({_hmm_conf:.0%}). Prime entry window."))
        elif _hmm_regime == "ACCUMULATION":
            _reasons.append(("⬆️", f"HMM ACCUMULATION ({_hmm_conf:.0%}, {_hmm_hours}h) — institutional buying, amplified conviction"))
        else:
            _reasons.append(("⚪", f"HMM STEADY STATE ({_hmm_conf:.0%}) — neutral regime, raise bar to 60%+"))
        # Composite
        _comp_str = "bullish" if composite > 0.3 else "bearish" if composite < -0.3 else "neutral/weak"
        _reasons.append(("📊", f"Composite {composite:+.3f} — {_comp_str}"))
        # Top model drivers
        for feat, imp in (_top_drv[:2] if _top_drv else []):
            _fd = feat.replace("sig_", "").replace("_", " ").upper()
            _reasons.append(("🔑", f"Top driver: {_fd} (importance {imp:.0f})"))
        # Funding
        if avg_funding > 0.3:
            _reasons.append(("💰", f"Funding {avg_funding:.4f} — longs are heavy, downward pressure expected"))
        elif avg_funding < 0:
            _reasons.append(("💰", f"Funding {avg_funding:.4f} — shorts are heavy, upward pressure expected"))
        # S/R context
        if _sr.get("available"):
            _ns_dist = abs(_ns.get("distance_pct", 0))
            _nr_dist = abs(_nr.get("distance_pct", 0))
            if _ns_dist < 1.0:
                _reasons.append(("🟢", f"Price at strong support ${_ns.get('price',0):.5f} ({_ns_dist:.1f}% below) — str {_ns.get('strength',0):.2f}"))
            if _nr_dist < 1.0:
                _reasons.append(("🔴", f"Price at resistance ${_nr.get('price',0):.5f} ({_nr_dist:.1f}% above) — str {_nr.get('strength',0):.2f}"))
        # Session
        if session in ("Late NYC",):
            _reasons.append(("⏰", "Late NYC (20-24h UTC) — historically no edge, session-gated"))
        # Alerts
        for _al in all_alerts[:2]:
            _reasons.append(("🚨", _al.get("title", "")[:80]))

        # Render reasons as a clean list
        if _reasons:
            _why_html = '<div style="background:#1a2226;border-radius:8px;padding:14px 18px;margin-bottom:14px">'
            _why_html += '<div style="font-size:0.72rem;color:#607d8b;font-weight:700;letter-spacing:1px;margin-bottom:10px">WHY THIS CALL</div>'
            for _icon, _txt in _reasons:
                _why_html += f'<div style="margin-bottom:6px;font-size:0.88rem;color:#ccc">{_icon} {_txt}</div>'
            _why_html += '</div>'
            st.markdown(_why_html, unsafe_allow_html=True)

        # ── Price + Composite chart ───────────────────────────────────────────
        # ── Signal components (collapsed — drill-down only) ──────────────────
        with st.expander("📡 Signal Components (drill-down)", expanded=False):
            if sig_vals:
                _sc = st.columns(len(sig_vals))
                for _i, (_cn, _v) in enumerate(sig_vals.items()):
                    _lbl = _cn.replace("sig_", "").replace("_", " ").title()
                    _ico = "🟢" if _v > 0.2 else "🔴" if _v < -0.2 else "⚪"
                    _sc[_i].metric(f"{_ico} {_lbl}", f"{_v:+.3f}")

        # ── Price + S/R chart ─────────────────────────────────────────────────
        st.markdown("#### Price vs Composite Signal")
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.03,
            subplot_titles=[f"{_disp_name} Price", "Composite Signal", "Volume"],
        )

        fig.add_trace(go.Scatter(
            x=ohlcv.index, y=ohlcv[price_col], name="Price",
            line=dict(color="#64b5f6", width=1.5)), row=1, col=1)

        # ── S/R level overlays ────────────────────────────────────────────────
        if _sr.get("available", False):
            _cur_p  = _sr.get("current_price", 0)
            _levels = _sr.get("levels", [])
            _x0     = ohlcv.index[0]
            _x1     = ohlcv.index[-1]

            # Value area shading first (behind lines)
            _va_lo = _sr.get("value_area_low")
            _va_hi = _sr.get("value_area_high")
            if _va_lo and _va_hi:
                fig.add_shape(
                    type="rect",
                    x0=_x0, x1=_x1,
                    y0=_va_lo, y1=_va_hi,
                    xref="x", yref="y",
                    fillcolor="rgba(100,181,246,0.07)",
                    line_width=0,
                    layer="below",
                    row=1, col=1,
                )
                fig.add_annotation(
                    x=_x1, y=_va_hi,
                    xref="x", yref="y",
                    text="Value Area (70% vol)",
                    showarrow=False,
                    font=dict(size=8, color="rgba(100,181,246,0.6)"),
                    xanchor="right", yanchor="bottom",
                    row=1, col=1,
                )

            # S/R level lines + labels
            for _lv in _levels:
                _lv_price    = _lv.get("price", 0)
                _lv_type     = _lv.get("type", "support")
                _lv_strength = _lv.get("strength", 0.5)
                _lv_methods  = _lv.get("methods", [])
                _lv_touches  = _lv.get("touches", 0)
                _dist_pct    = (_lv_price - _cur_p) / (_cur_p + 1e-9) * 100

                # Color: bright for strong, muted for weak
                if _lv_type == "resistance":
                    _rgb = f"rgba(239,154,154,{0.4 + 0.5 * _lv_strength:.2f})"
                else:
                    _rgb = f"rgba(165,214,167,{0.4 + 0.5 * _lv_strength:.2f})"

                _lw   = 0.8 + 1.4 * _lv_strength   # 0.8–2.2px
                _dash = "solid" if _lv_strength >= 0.7 else "dot"

                # Horizontal line
                fig.add_shape(
                    type="line",
                    x0=_x0, x1=_x1,
                    y0=_lv_price, y1=_lv_price,
                    xref="x", yref="y",
                    line=dict(color=_rgb, width=_lw, dash=_dash),
                    row=1, col=1,
                )

                # Label at right edge
                _prefix  = "R" if _lv_type == "resistance" else "S"
                _meth_short = "+".join(
                    m.replace("volume_node", "vol").replace("round_number", "rnd")
                    for m in _lv_methods
                )
                _label = (f"{_prefix} ${_lv_price:.4f}  {_dist_pct:+.1f}%"
                          + (f"  [{_meth_short}]" if _meth_short else "")
                          + (f"  t={_lv_touches}" if _lv_touches else ""))
                fig.add_annotation(
                    x=_x1, y=_lv_price,
                    xref="x", yref="y",
                    text=_label,
                    showarrow=False,
                    font=dict(size=8, color=_rgb),
                    bgcolor="rgba(15,20,30,0.65)",
                    xanchor="right",
                    yanchor="middle",
                    row=1, col=1,
                )

        comp = signals["composite"]
        fig.add_trace(go.Scatter(
            x=comp.index, y=comp, name="Composite",
            line=dict(color="#90caf9", width=1)), row=2, col=1)

        for level, color, dash in [
            (0.4,  "#66bb6a", "dash"), (0.2,  "#a5d6a7", "dot"),
            (-0.2, "#ef9a9a", "dot"),  (-0.4, "#e57373", "dash"),
        ]:
            fig.add_hline(y=level, line_color=color, line_dash=dash, row=2, col=1)

        fig.add_hrect(y0=0.4, y1=0.7,  fillcolor="#66bb6a", opacity=0.06, row=2, col=1)
        fig.add_hrect(y0=-0.7, y1=-0.4, fillcolor="#e57373", opacity=0.06, row=2, col=1)
        fig.add_hline(y=composite, line_color="yellow", line_dash="dot",
                      annotation_text=f"NOW {composite:+.3f}", row=2, col=1)

        fig.add_trace(go.Bar(
            x=ohlcv.index, y=ohlcv["volume"], name="Volume",
            marker_color="rgba(120,120,120,0.3)"), row=3, col=1)

        fig.update_layout(
            height=650, showlegend=False, hovermode="x unified",
            template="plotly_dark", margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── S/R Summary Panel ─────────────────────────────────────────────────
        if _sr.get("available", False):
            with st.expander("🧱 Support & Resistance Levels", expanded=False):
                _cur_p   = _sr.get("current_price", 0)
                _levels  = _sr.get("levels", [])
                _va_lo   = _sr.get("value_area_low")
                _va_hi   = _sr.get("value_area_high")
                _ns_dict = _ns
                _nr_dict = _nr
                _rr      = _sr.get("risk_reward", 1.0)

                _c1, _c2, _c3 = st.columns(3)
                if _ns_dict:
                    _dist = abs(_ns_dict.get("distance_pct", 0))
                    _c1.metric(
                        "🟢 Nearest Support",
                        f"${_ns_dict['price']:.5f}",
                        f"-{_dist:.2f}% | str={_ns_dict['strength']:.2f}",
                    )
                if _nr_dict:
                    _dist = abs(_nr_dict.get("distance_pct", 0))
                    _c2.metric(
                        "🔴 Nearest Resistance",
                        f"${_nr_dict['price']:.5f}",
                        f"+{_dist:.2f}% | str={_nr_dict['strength']:.2f}",
                    )
                _c3.metric("⚖️ Risk/Reward", f"{_rr:.2f}x", "nearest R / nearest S")

                if _va_lo and _va_hi:
                    _in_va = _va_lo <= _cur_p <= _va_hi
                    st.markdown(
                        f"**Value Area (70% vol):** ${_va_lo:.5f} – ${_va_hi:.5f}"
                        + (f"  ← *price inside value area*" if _in_va else "  ← *price outside value area*")
                    )

                # Level table
                if _levels:
                    _lv_rows = []
                    for _lv in sorted(_levels, key=lambda l: l["price"], reverse=True):
                        _dp = (_lv["price"] - _cur_p) / (_cur_p + 1e-9) * 100
                        _lv_rows.append({
                            "Type":       "🔴 R" if _lv["type"] == "resistance" else "🟢 S",
                            "Price":      f"${_lv['price']:.5f}",
                            "Dist %":     f"{_dp:+.2f}%",
                            "Strength":   f"{'⬆' if _lv['strength'] > 0.7 else '▶' if _lv['strength'] > 0.4 else '▼'} {_lv['strength']:.2f}",
                            "Touches":    _lv.get("touches", "—"),
                            "Bounce%":    f"{_lv.get('bounce_rate', 0)*100:.0f}%",
                            "Methods":    "+".join(_lv.get("methods", [])),
                        })
                    st.dataframe(
                        pd.DataFrame(_lv_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

        # ── Historical trades ─────────────────────────────────────────────────
        trades = data.get("trades")
        if trades is not None and not trades.empty:
            with st.expander("📜 Historical Trades"):
                st.dataframe(trades, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORWARD PROJECTIONS  (all models incl. external data)
# ═════════════════════════════════════════════════════════════════════════════

with tab_proj:

    # ── Row 0: Exit plan reminder ─────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#263238;border-left:4px solid #546e7a;'
        f'border-radius:6px;padding:10px 16px;margin-bottom:14px;font-size:0.9rem">'
        f'📌 <b>Exit Plan:</b> {action["exit_plan"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Opportunity Score Meter ───────────────────────────────────────────────
    if opp.get("available"):
        score     = _score
        tier      = _tier
        size_pct  = _size_pct
        hmm_label = opp.get("hmm_label", "—")
        meta_prob = _meta_prob
        top_drv   = _top_drv

        tier_colors = {
            "BLOCKED":          ("#b71c1c", "🚫"),
            "PASS":             ("#37474f", "⏸️"),
            "WATCH":            ("#e65100", "👁️"),
            "TRADE":            ("#1b5e20", "✅"),
            "HIGH CONVICTION":  ("#00695c", "🔥"),
            "FULL SEND":        ("#004d40", "⚡"),
        }
        tier_col, tier_icon = tier_colors.get(tier, ("#37474f", ""))

        # Score bar fill colour (red→yellow→green)
        bar_color = "#b71c1c" if score < 35 else "#e65100" if score < 50 else "#f9a825" if score < 60 else "#2e7d32"

        st.markdown("#### 🎯 Opportunity Score")
        opp_left, opp_right = st.columns([2, 1])
        with opp_left:
            st.markdown(
                f'<div style="background:#1a1a2e;border:1px solid {tier_col};border-radius:10px;padding:16px 20px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
                f'<span style="font-size:2rem;font-weight:900;color:{bar_color}">{score}<span style="font-size:1rem;color:#888">/100</span></span>'
                f'<span style="background:{tier_col};color:#fff;border-radius:6px;padding:4px 14px;'
                f'font-size:0.9rem;font-weight:700">{tier_icon} {tier}</span>'
                f'</div>'
                f'<div style="background:#263238;border-radius:6px;height:10px;margin-bottom:10px">'
                f'<div style="background:{bar_color};width:{score}%;height:100%;border-radius:6px;'
                f'transition:width 0.5s"></div></div>'
                f'<div style="display:flex;gap:20px;font-size:0.82rem;color:#aaa">'
                f'<span>Meta-prob: <b style="color:#fff">{meta_prob:.0%}</b></span>'
                f'<span>HMM: <b style="color:#fff">{hmm_label}</b></span>'
                f'<span>Position size: <b style="color:{bar_color}">{size_pct}%</b></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with opp_right:
            if top_drv:
                drv_html = '<div style="background:#1a1a2e;border:1px solid #37474f;border-radius:8px;padding:12px">'
                drv_html += '<div style="font-size:0.75rem;color:#607d8b;margin-bottom:6px">TOP SIGNAL DRIVERS</div>'
                for feat, imp in top_drv[:4]:
                    feat_display = feat.replace("sig_", "").replace("_", " ").upper()
                    bar_w = int(imp / (top_drv[0][1] + 1e-9) * 100)
                    drv_html += (
                        f'<div style="margin-bottom:5px">'
                        f'<div style="font-size:0.72rem;color:#ccc">{feat_display}</div>'
                        f'<div style="background:#263238;border-radius:3px;height:5px">'
                        f'<div style="background:#00d4aa;width:{bar_w}%;height:100%;border-radius:3px"></div>'
                        f'</div></div>'
                    )
                drv_html += '</div>'
                st.markdown(drv_html, unsafe_allow_html=True)
        st.markdown("")

    # ── Trade Setups Panel ────────────────────────────────────────────────────
    trade_setups = proj.get("trade_setups", [])
    active_setups   = [s for s in trade_setups if s.get("active") and s["id"] != "NO_TRADE"]
    inactive_setups = [s for s in trade_setups if not s.get("active") and s["id"] != "NO_TRADE"]

    st.markdown("#### 🎯 Trade Setups")
    if not active_setups:
        st.markdown(
            '<div style="background:#263238;border-left:4px solid #546e7a;border-radius:6px;'
            'padding:12px 16px;font-size:0.9rem;color:#90a4ae">'
            '🚫 <b>NO ACTIVE SETUPS</b> — Model below 55% threshold. '
            'Not enough edge to cover carry costs. Wait for a setup to activate.</div>',
            unsafe_allow_html=True,
        )
    else:
        _setup_dir_colors = {"short": "#b71c1c", "long": "#1b5e20", "flat": "#37474f"}
        _setup_dir_icons  = {"short": "⬇️ SHORT", "long": "⬆️ LONG", "flat": "⏸️ FLAT"}

        ts_cols = st.columns(min(len(active_setups), 3))
        for i, setup in enumerate(active_setups[:3]):
            col = ts_cols[i % len(ts_cols)]
            sd   = setup.get("direction", "flat")
            sc   = _setup_dir_colors.get(sd, "#37474f")
            si   = _setup_dir_icons.get(sd, sd.upper())
            conf = setup.get("confidence", 0)
            conf_bar = int(conf * 100)
            conf_color = "#1b5e20" if conf >= 0.70 else "#e65100" if conf >= 0.50 else "#546e7a"

            col.markdown(
                f'<div class="proj-card" style="border-color:{sc};background:{sc}22">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<b style="color:{sc};font-size:1rem">{si}</b>'
                f'<span style="background:{conf_color};color:#fff;border-radius:4px;'
                f'padding:2px 8px;font-size:0.75rem;font-weight:700">{conf:.0%} conf</span>'
                f'</div>'
                f'<div style="font-size:0.78rem;color:#aaa;margin:2px 0 6px">'
                f'<b>{setup["id"].replace("_", " ")}</b>'
                f'</div>'
                f'<div style="font-size:0.85rem;margin-bottom:8px">'
                f'{setup.get("trigger","")}'
                f'</div>'
                f'<div style="font-size:0.78rem;color:#aaa">'
                f'Target: <b style="color:#fff">{setup.get("target_pct",0):+.2f}%</b>'
                f'&nbsp;|&nbsp; {setup.get("stop_note","")}</div>'
                f'<div style="font-size:0.72rem;color:#607d8b;margin-top:4px">'
                f'📊 {setup.get("historical_edge","")}</div>'
                f'<div style="font-size:0.72rem;color:#546e7a;margin-top:2px">'
                f'🕐 {setup.get("time_window","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Pending setups (inactive — waiting for trigger)
    if inactive_setups:
        with st.expander(f"⏳ {len(inactive_setups)} pending setup(s) — waiting for trigger"):
            for setup in inactive_setups:
                sd = setup.get("direction", "flat")
                sc = _setup_dir_colors.get(sd, "#37474f") if active_setups else "#37474f"
                st.markdown(
                    f'**{setup["id"].replace("_"," ")}** ({sd.upper()}) — '
                    f'{setup.get("trigger","")} · '
                    f'*{setup.get("time_window","")}*'
                )

    st.markdown("---")

    # ── Row 1: Probability + Manipulation cycle ───────────────────────────────
    r1a, r1b = st.columns([3, 2])

    prob_data       = proj.get("probability", {})
    prob_val        = prob_data.get("prob_positive_4h", 0.5)
    exp_move        = prob_data.get("expected_move_pct", 0)
    prob_conv       = prob_data.get("conviction", "LOW")
    entry_rec       = prob_data.get("entry_recommendation", "")
    above_threshold = prob_data.get("above_bybit_threshold", False)
    bybit_entry_thr = prob_data.get("bybit_entry_threshold", 0.55)
    bybit_be        = prob_data.get("bybit_break_even", 0.61)
    bybit_carry     = prob_data.get("bybit_carry_4h_pct", 0.45)

    with r1a:
        st.markdown("#### Probability Model (Bybit-calibrated)")

        # Color: green above break-even, amber in entry zone, red below entry, grey neutral
        if prob_val >= bybit_be:
            pcolor = "#1b5e20"    # strong green
        elif prob_val >= bybit_entry_thr:
            pcolor = "#e65100"    # amber — above entry threshold but below break-even
        elif prob_val <= (1 - bybit_be):
            pcolor = "#b71c1c"    # strong red
        elif prob_val <= (1 - bybit_entry_thr):
            pcolor = "#880e4f"    # dark pink bearish
        else:
            pcolor = "#37474f"    # grey — no edge zone

        # Entry/no-trade banner
        if above_threshold:
            entry_banner = (
                '<div style="background:#1b5e20;color:#fff;border-radius:4px;padding:3px 10px;'
                'font-weight:800;font-size:0.82rem;margin-bottom:6px;display:inline-block">'
                f'✅ ENTRY SIGNAL — prob {prob_val:.0%} ≥ {bybit_entry_thr:.0%} threshold</div><br>'
            )
        elif prob_val <= (1 - bybit_entry_thr):
            entry_banner = (
                '<div style="background:#880e4f;color:#fff;border-radius:4px;padding:3px 10px;'
                'font-weight:800;font-size:0.82rem;margin-bottom:6px;display:inline-block">'
                f'⚠️ BEARISH — prob {prob_val:.0%} ≤ {1-bybit_entry_thr:.0%}</div><br>'
            )
        else:
            entry_banner = (
                '<div style="background:#37474f;color:#ccc;border-radius:4px;padding:3px 10px;'
                'font-weight:700;font-size:0.82rem;margin-bottom:6px;display:inline-block">'
                f'🚫 NO TRADE — {prob_val:.0%} is between {1-bybit_entry_thr:.0%}–{bybit_entry_thr:.0%} (no edge after carry)</div><br>'
            )

        # Threshold gauge bar (simple inline HTML)
        bar_pct = int(prob_val * 100)
        be_pct  = int(bybit_be * 100)
        ent_pct = int(bybit_entry_thr * 100)
        gauge_bar = (
            f'<div style="position:relative;height:8px;background:#333;border-radius:4px;margin:6px 0 10px">'
            f'<div style="position:absolute;left:0;width:{bar_pct}%;height:100%;background:{pcolor};border-radius:4px"></div>'
            f'<div style="position:absolute;left:{ent_pct}%;top:-3px;width:2px;height:14px;background:#ff9800" title="Entry threshold {bybit_entry_thr:.0%}"></div>'
            f'<div style="position:absolute;left:{be_pct}%;top:-3px;width:2px;height:14px;background:#4caf50" title="Break-even {bybit_be:.0%}"></div>'
            f'</div>'
            f'<div style="font-size:0.7rem;color:#888;display:flex;justify-content:space-between">'
            f'<span>0%</span>'
            f'<span style="color:#ff9800">▲ Entry {bybit_entry_thr:.0%}</span>'
            f'<span style="color:#4caf50">▲ B/E {bybit_be:.0%}</span>'
            f'<span>100%</span>'
            f'</div>'
        )

        # BTC divergence badge
        btc_div_badge = ""
        if prob_data.get("btc_divergence"):
            btc_div_badge = (
                '<div style="background:#e65100;color:#fff;border-radius:4px;padding:3px 10px;'
                'font-weight:700;font-size:0.78rem;margin-bottom:6px;display:inline-block">'
                '⚠ BTC DIVERGENCE — FART lagging BTC rally (bearish confirmation)</div><br>'
            )

        st.markdown(
            f'<div class="proj-card" style="border-color:{pcolor};background:{pcolor}22">'
            f'{entry_banner}'
            f'{btc_div_badge}'
            f'<span style="font-size:2rem;font-weight:800;color:{pcolor}">{prob_val:.0%}</span>'
            f'&nbsp;&nbsp;probability of positive 4h return<br>'
            f'<b>Expected move:</b> {exp_move:+.2f}% &nbsp;|&nbsp; '
            f'<b>Conviction:</b> {prob_conv} &nbsp;|&nbsp; '
            f'<b>n=</b>{prob_data.get("model_n_train", 0):,}<br>'
            f'{gauge_bar}'
            f'<span style="font-size:0.75rem;color:#888">'
            f'Bybit carry cost: <b>{bybit_carry:.2f}%/4h</b> &nbsp;|&nbsp; '
            f'Entry threshold: <b>{bybit_entry_thr:.0%}</b> &nbsp;|&nbsp; '
            f'Break-even: <b>{bybit_be:.0%}</b>'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    cycle = proj.get("manipulation_cycle", {})
    phase = cycle.get("phase", "DORMANT")
    with r1b:
        st.markdown("#### Manipulation Cycle")
        _ccolors = {
            "SPIKE_IN_PROGRESS": "#b71c1c",
            "BUILDUP":           "#e65100",
            "QUIET_ACCUMULATION":"#1565c0",
            "DORMANT":           "#37474f",
        }
        _phase_icons = {
            "SPIKE_IN_PROGRESS": "🚨 EXIT",
            "BUILDUP":           "⚡ BUILDUP",
            "QUIET_ACCUMULATION":"🔍 QUIET ACCUM",
            "DORMANT":           "😴 DORMANT",
        }
        ccolor   = _ccolors.get(phase, "#37474f")
        phase_lbl = _phase_icons.get(phase, phase)
        est      = cycle.get("est_hours_to_move")
        est_txt  = f"Est. move: ~{est}h" if est is not None and phase != "SPIKE_IN_PROGRESS" else ""
        exit_banner = (
            '<div style="background:#b71c1c;color:#fff;border-radius:4px;padding:4px 10px;'
            'font-weight:800;font-size:0.85rem;margin-bottom:6px">⛔ DO NOT ENTER — CLOSE POSITIONS</div>'
            if phase == "SPIKE_IN_PROGRESS" else ""
        )
        st.markdown(
            f'<div class="proj-card" style="border-color:{ccolor};background:{ccolor}22">'
            f'{exit_banner}'
            f'<span style="font-size:1.2rem;font-weight:800;color:{ccolor}">{phase_lbl}</span> '
            f'({cycle.get("confidence", 0):.0%} conf)<br>'
            f'{est_txt}<br>'
            f'<span style="font-size:0.8rem;color:#aaa">{cycle.get("description","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Row 2: Price fan chart ────────────────────────────────────────────────
    ci  = proj.get("confidence_intervals", {})
    h4  = ci.get("h4")
    h8  = ci.get("h8")

    ohlcv_chart = data.get("ohlcv")
    if h4 and h8 and ohlcv_chart is not None:
        price_col = "price" if "price" in ohlcv_chart.columns else "close"
        recent    = ohlcv_chart[price_col].iloc[-24:]

        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(
            x=list(range(-len(recent), 0)), y=recent.values,
            mode="lines", name="History", line=dict(color="#90caf9", width=2),
        ))

        hours_fwd = [0, 4, 8]
        cur = float(recent.iloc[-1])
        centers  = [cur, h4["center"],  h8["center"]]
        high_95  = [cur, h4["high_95"], h8["high_95"]]
        low_95   = [cur, h4["low_95"],  h8["low_95"]]
        high_68  = [cur, h4["high_68"], h8["high_68"]]
        low_68   = [cur, h4["low_68"],  h8["low_68"]]

        fig_fan.add_trace(go.Scatter(x=hours_fwd, y=high_95, mode="lines",
                                     line=dict(width=0), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=hours_fwd, y=low_95, mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(100,181,246,0.12)", name="95% CI"))
        fig_fan.add_trace(go.Scatter(x=hours_fwd, y=high_68, mode="lines",
                                     line=dict(width=0), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=hours_fwd, y=low_68, mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(100,181,246,0.30)", name="68% CI"))
        fig_fan.add_trace(go.Scatter(x=hours_fwd, y=centers, mode="lines+markers",
                                     name="Expected", line=dict(color="#ffeb3b", dash="dash", width=2)))
        fig_fan.add_vline(x=0, line_color="#546e7a", line_dash="dash")
        thin_vol = h4.get("thin_volume", False)
        vol_ratio_val = h4.get("volume_ratio", 1.0)
        vol_note = (
            f" ⚠ THIN VOLUME ({vol_ratio_val:.0%} of 24h avg) — bands widened, expect wider spreads"
            if thin_vol else ""
        )
        fig_fan.update_layout(
            title=f"4h: ${h4['low_68']:.4f} – ${h4['high_68']:.4f} (68%) | "
                  f"${h4['low_95']:.4f} – ${h4['high_95']:.4f} (95%){vol_note}",
            xaxis_title="Hours (0 = now)", yaxis_title="Price ($)",
            template="plotly_dark", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_fan, use_container_width=True)
        if thin_vol:
            st.warning(f"⚠ Volume at {vol_ratio_val:.0%} of 24h avg — reduce position size 25-30%, expect wide spreads")

    st.markdown("---")

    # ── Row 3: Mean reversion — Funding & LSR ────────────────────────────────
    st.markdown("#### Mean Reversion Models")
    mr = proj.get("mean_reversion", {})
    mr_c1, mr_c2 = st.columns(2)

    with mr_c1:
        fr_data = mr.get("funding")
        if fr_data and fr_data.get("projected_path"):
            path = fr_data["projected_path"]
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Scatter(
                x=list(range(1, len(path)+1)), y=path, mode="lines",
                name="Projected", line=dict(color="#4dd0e1", dash="dash", width=2)))
            fig_fr.add_hline(y=fr_data["mean"], line_dash="dot",
                             annotation_text="Mean", line_color="#546e7a")
            fig_fr.update_layout(
                title=f"Funding Rate  (½-life {fr_data['half_life_h']:.1f}h, real {fr_data.get('current_real',0):.4f})",
                xaxis_title="Hours", yaxis_title="Rate",
                template="plotly_dark", height=260, margin=dict(t=36, b=10),
            )
            st.plotly_chart(fig_fr, use_container_width=True)
            st.caption(fr_data.get("description", ""))
        else:
            st.info("No funding reversion data")

    with mr_c2:
        lsr_data = mr.get("lsr")
        if lsr_data and lsr_data.get("projected_path"):
            path = lsr_data["projected_path"]
            pct  = lsr_data.get("percentile", 0.5)
            # Calibrated: p85+ = extreme (forced unwind), p75+ = high crowding
            if pct >= 0.85:
                bar_color = "#b71c1c"
            elif pct >= 0.75:
                bar_color = "#e65100"
            elif pct <= 0.15:
                bar_color = "#1565c0"
            elif pct <= 0.25:
                bar_color = "#1976d2"
            else:
                bar_color = "#90caf9"

            fig_lsr = go.Figure()
            fig_lsr.add_trace(go.Scatter(
                x=list(range(1, len(path)+1)), y=path, mode="lines",
                name="Projected", line=dict(color=bar_color, dash="dash", width=2)))
            fig_lsr.add_hline(y=lsr_data.get("median", 1.0), line_dash="dot",
                              annotation_text="Median", line_color="#546e7a")
            # Add extremity threshold lines
            fig_lsr.add_hline(y=float(lsr_data.get("current", 1.0)),
                              line_color=bar_color, line_dash="dot", line_width=1,
                              annotation_text=f"Now {lsr_data['current']:.3f}")
            fig_lsr.update_layout(
                title=f"Long/Short Ratio  (now {lsr_data['current']:.3f}, {pct:.0%}ile) — "
                      f"reversion in ~{lsr_data.get('avg_revert_time_h',8):.0f}h ({lsr_data.get('revert_rate',0):.0%} rate)",
                xaxis_title="Hours", yaxis_title="LSR",
                template="plotly_dark", height=260, margin=dict(t=36, b=10),
            )
            st.plotly_chart(fig_lsr, use_container_width=True)

            # Extremity action alert
            lsr_action = lsr_data.get("lsr_action", "")
            if pct >= 0.85:
                st.error(f"🔴 FORCED UNWIND SIGNAL — {lsr_action}")
            elif pct >= 0.75:
                st.warning(f"⚠ {lsr_action}")
            elif pct <= 0.15:
                st.error(f"🔵 SHORT SQUEEZE RISK — {lsr_action}")
            elif pct <= 0.25:
                st.warning(f"⚠ {lsr_action}")

            st.caption(lsr_data.get("description", ""))
        else:
            st.info("No LSR reversion data")

    st.markdown("---")

    # ── Row 4: BTC Lead-Lag + Session Edge ───────────────────────────────────
    st.markdown("#### Context Models")
    ctx_c1, ctx_c2 = st.columns(2)

    btc_ll    = proj.get("btc_lead_lag", {})
    btc_2h    = btc_ll.get("btc_2h_return_pct", 0)
    proj_fart = btc_ll.get("projected_fart_move_pct", 0)
    btc_conf  = btc_ll.get("confidence", 0)

    with ctx_c1:
        st.markdown("**BTC Lead-Lag**")
        ll_color = "#b71c1c" if abs(btc_2h) > 2 else "#e65100" if abs(btc_2h) > 1 else "#1565c0"

        # Badge logic — divergence takes priority over standard override
        btc_ov_type = prob_data.get("btc_override_type", "")
        btc_is_div  = prob_data.get("btc_divergence", False)
        btc_ov_badge = ""
        if btc_is_div:
            btc_ov_badge = (
                '<div style="background:#e65100;color:#fff;border-radius:4px;padding:3px 10px;'
                'font-size:0.78rem;font-weight:800;margin-bottom:6px">'
                '⚠ BTC DIVERGENCE — FART lagging BTC rally. '
                'If FART doesn\'t follow within 1-2h → SHORT the divergence.</div>'
            )
        elif btc_ov_type == "BTC_DUMP_LOW_FUND_BUY":
            btc_ov_badge = '<div style="background:#1b5e20;color:#fff;border-radius:4px;padding:3px 10px;font-size:0.78rem;font-weight:800;margin-bottom:6px">⭐ HIGH-CONVICTION LONG — BTC dump + low funding override active</div>'
        elif btc_ov_type == "BTC_RALLY_HIGH_FUND_FADE":
            btc_ov_badge = '<div style="background:#b71c1c;color:#fff;border-radius:4px;padding:3px 10px;font-size:0.78rem;font-weight:800;margin-bottom:6px">⚠ DON\'T CHASE — BTC rally + high funding, 24% hist win rate</div>'
        elif btc_ov_type == "BTC_DIRECTION":
            btc_ov_badge = '<div style="background:#e65100;color:#fff;border-radius:4px;padding:3px 10px;font-size:0.78rem;font-weight:800;margin-bottom:6px">⚡ BTC DIRECTION OVERRIDE active</div>'

        if btc_ov_badge:
            st.markdown(btc_ov_badge, unsafe_allow_html=True)
        _proj_card(
            f"BTC {btc_2h:+.1f}% → FART {proj_fart:+.1f}% projected",
            btc_ll.get("description", "No data"),
            color=ll_color, icon="₿",
        )
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("BTC 2h", f"{btc_2h:+.1f}%")
        bc2.metric("FART proj", f"{proj_fart:+.1f}%")
        bc3.metric("Confidence", f"{btc_conf:.0%}")

    sess_cond    = proj.get("session_conditional", {})
    edge         = sess_cond.get("combined_edge_pct", 0)
    quality      = sess_cond.get("quality", "")
    dow_name     = sess_cond.get("day_of_week", "")
    dow_bias     = sess_cond.get("day_bias", "neutral")
    vol_pct_avg  = sess_cond.get("volume_pct_of_avg", 1.0)
    thin_vol_ses = vol_pct_avg < 0.70

    with ctx_c2:
        st.markdown("**Session-Conditional Edge**")
        se_color = "#1b5e20" if edge > 0.5 else "#b71c1c" if edge < -0.5 else "#e65100"

        # Volume status pill
        if vol_pct_avg < 0.50:
            vol_pill = f'<span style="background:#b71c1c;color:#fff;border-radius:3px;padding:1px 7px;font-size:0.72rem;font-weight:700">VERY THIN VOL {vol_pct_avg:.0%}</span>'
        elif vol_pct_avg < 0.70:
            vol_pill = f'<span style="background:#e65100;color:#fff;border-radius:3px;padding:1px 7px;font-size:0.72rem;font-weight:700">THIN VOL {vol_pct_avg:.0%}</span>'
        else:
            vol_pill = f'<span style="background:#37474f;color:#ccc;border-radius:3px;padding:1px 7px;font-size:0.72rem">Vol {vol_pct_avg:.0%} of avg</span>'

        _proj_card(
            f"Edge: {edge:+.2f}%  [{quality}]",
            sess_cond.get("description", "No data"),
            color=se_color, icon="⏰",
        )
        st.markdown(vol_pill, unsafe_allow_html=True)

        if thin_vol_ses:
            size_cut = "40%+" if vol_pct_avg < 0.50 else "25-30%"
            st.warning(f"⚠ Thin volume ({vol_pct_avg:.0%} of 24h avg) — reduce position size {size_cut}")

        # Day-of-week bias badge
        if dow_bias == "bearish":
            st.markdown(
                f'<span style="background:#b71c1c22;border:1px solid #b71c1c;color:#ef9a9a;'
                f'border-radius:4px;padding:3px 10px;font-size:0.8rem">'
                f'⚠ {dow_name} — historical bearish bias</span>',
                unsafe_allow_html=True,
            )
        elif dow_bias == "bullish":
            st.markdown(
                f'<span style="background:#1b5e2022;border:1px solid #1b5e20;color:#a5d6a7;'
                f'border-radius:4px;padding:3px 10px;font-size:0.8rem">'
                f'✅ {dow_name} — historical bullish bias</span>',
                unsafe_allow_html=True,
            )
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Combined Edge", f"{edge:+.2f}%")
        sc2.metric("Samples", f"n={sess_cond.get('n_samples',0)}")
        sc3.metric("Volume", f"{vol_pct_avg:.0%} of avg")

    st.markdown("---")

    # ── Row 5: External data projections ─────────────────────────────────────
    st.markdown("#### External Data Signals")
    ext_c1, ext_c2, ext_c3, ext_c4 = st.columns(4)

    # — News / Sentiment ——————————————————————————————————————————————————
    news = proj.get("news_sentiment", {})
    with ext_c1:
        st.markdown("**Sentiment**")
        if not news.get("available"):
            st.info("No sentiment data — run external_collectors.py")
        else:
            na = news.get("assessment", "NEUTRAL")
            nd = news.get("divergence", "NONE")
            na_color = "#b71c1c" if na in ("DANGER","CAUTION") else \
                       "#1b5e20" if na == "BULLISH" else "#1565c0"
            fg_val   = news.get("fear_greed_value", 0)
            fg_class = news.get("fear_greed_class", "")
            fg_color = "#e57373" if fg_val < 30 else "#66bb6a" if fg_val > 60 else "#ffb74d"
            st.markdown(
                f'<div class="proj-card" style="border-color:{na_color};background:{na_color}18">'
                f'<b style="color:{na_color}">{na}</b>'
                f'{"  |  Divergence: <b>" + nd + "</b>" if nd != "NONE" else ""}<br>'
                f'<span style="color:{fg_color}">Fear&Greed: <b>{fg_val} ({fg_class})</b></span><br>'
                f'Community bullish: <b>{news.get("cg_sentiment_up_pct",0):.0f}%</b> &nbsp;|&nbsp; '
                f'Composite: <b>{news.get("sentiment_composite",0):.3f}</b><br>'
                f'<span style="font-size:0.8rem;color:#aaa">{news.get("description","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # — On-chain / Exchange Flow ——————————————————————————————————————————
    onchain = proj.get("onchain_flow", {})
    with ext_c2:
        st.markdown("**On-Chain Flow**")
        if not onchain.get("available"):
            st.info("No on-chain data — run external_collectors.py")
        else:
            oa = onchain.get("assessment", "NEUTRAL")
            oc = "#b71c1c" if "DUMP" in oa or "INFLOW" in oa else \
                 "#1b5e20" if "ACCUM" in oa or "WITHDRAWAL" in oa else "#1565c0"
            net = onchain.get("net_flow_tokens", 0)
            st.markdown(
                f'<div class="proj-card" style="border-color:{oc};background:{oc}18">'
                f'<b style="color:{oc}">{oa}</b><br>'
                f'Net flow: <b>{net:+,.0f} tokens</b><br>'
                f'Whale transfers: <b>{onchain.get("whale_transfers",0)}</b> &nbsp;|&nbsp; '
                f'Gini: <b>{onchain.get("gini",0):.3f}</b> ({onchain.get("gini_trend","?")})<br>'
                f'Top-10 holders: <b>{onchain.get("top10_pct",0):.1f}%</b><br>'
                f'<span style="font-size:0.8rem;color:#aaa">{onchain.get("description","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # — Cross-Exchange / Squeeze ——————————————————————————————————————————
    cx = proj.get("cross_exchange", {})
    with ext_c3:
        st.markdown("**Cross-Exchange / Squeeze**")
        if not cx.get("available"):
            st.info("No cross-exchange data — run external_collectors.py")
        else:
            ca = cx.get("assessment", "NORMAL")
            sq = cx.get("squeeze_risk", "LOW")
            cc = "#b71c1c" if sq == "HIGH" or "SQUEEZE" in ca else \
                 "#e65100" if sq == "MEDIUM" else "#1565c0"
            pf = cx.get("predicted_funding", 0)
            st.markdown(
                f'<div class="proj-card" style="border-color:{cc};background:{cc}18">'
                f'<b style="color:{cc}">{ca}</b><br>'
                f'Squeeze risk: <b style="color:{cc}">{sq}</b><br>'
                f'Predicted funding: <b>{pf:.4f}</b><br>'
                f'Funding arb: <b>{cx.get("funding_arb","?")}</b> &nbsp;|&nbsp; '
                f'Liq z-score: <b>{cx.get("liq_zscore",0):.2f}</b><br>'
                f'<span style="font-size:0.8rem;color:#aaa">{cx.get("description","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # — Coinglass OI Momentum + Funding Spread (Bybit-aware) ———————————————
    cg = proj.get("coinglass_oi_funding", {})
    with ext_c4:
        st.markdown("**OI / Funding (Bybit-calibrated)**")
        if not cg.get("available"):
            st.info("No Coinglass data — run: python3 external_collectors.py --source coinglass")
        else:
            cga          = cg.get("assessment", "NORMAL")
            bybit_rate   = cg.get("bybit_rate", 0)
            binance_rate = cg.get("binance_rate", 0)
            daily_carry  = cg.get("bybit_daily_carry_pct", bybit_rate * 3)
            bb_div       = cg.get("binance_bybit_divergence", "NORMAL")

            # Colour map — includes new Bybit-specific assessments
            _cg_colors = {
                "OI_PRICE_DIV_LONG":        "#1b5e20",
                "PASSIVE_ACCUM":            "#2e7d32",
                "EXTREME_SHORT_FUNDING":    "#1565c0",
                "BINANCE_BEARISH_VS_BYBIT": "#b71c1c",
                "BOTH_CROWDED_LONG":        "#c62828",
                "OI_BUILDING_PRICE_WEAK":   "#b71c1c",   # NEW: longs trapped
                "OI_SPIKE_CAUTION":         "#b71c1c",
                "OI_SURGE_CAUTION":         "#c62828",
                "OI_TREND_CHASE_BEARISH":   "#e65100",
                "EXTREME_LONG":             "#c62828",
                "HIGH_LONG":                "#e65100",
                "SETTLEMENT_IMMINENT":      "#f57f17",
                "OI_BUILDING":              "#546e7a",
                "FUNDING_SPREAD":           "#546e7a",
                "NORMAL":                   "#37474f",
            }
            cg_color = _cg_colors.get(cga, "#37474f")
            cg_icon = (
                "⭐" if cga in ("OI_PRICE_DIV_LONG", "PASSIVE_ACCUM", "EXTREME_SHORT_FUNDING")
                else "🔴" if cga == "OI_BUILDING_PRICE_WEAK"
                else "⚠" if cga in ("OI_SPIKE_CAUTION", "OI_TREND_CHASE_BEARISH",
                                    "EXTREME_LONG", "HIGH_LONG", "BINANCE_BEARISH_VS_BYBIT",
                                    "BOTH_CROWDED_LONG")
                else "⚡" if cga == "SETTLEMENT_IMMINENT" else ""
            )

            m5      = cg.get("m5_oi_chg", 0)
            m15     = cg.get("m15_oi_chg", 0)
            h1      = cg.get("h1_oi_chg", 0)
            oi_div  = cg.get("oi_price_divergence", "NORMAL")
            spread  = cg.get("spread_pct", 0)
            settle  = cg.get("mins_to_settle", 999)
            settle_str = f"⚡ {settle:.0f}min to settle" if cg.get("settlement_imminent") else ""

            # Carry cost pill colour: red if >1%/day, orange if >0.5%
            carry_color = "#b71c1c" if daily_carry > 1.0 else "#e65100" if daily_carry > 0.5 else "#37474f"

            # Divergence badge
            div_badge = ""
            if bb_div == "BINANCE_BEARISH_VS_BYBIT":
                div_badge = (
                    '<span style="background:#b71c1c;color:#fff;border-radius:3px;'
                    'padding:1px 6px;font-size:0.72rem;font-weight:700">'
                    '⚠ BINANCE BEARISH vs BYBIT</span><br>'
                )
            elif bb_div == "BOTH_CROWDED_LONG":
                div_badge = (
                    '<span style="background:#c62828;color:#fff;border-radius:3px;'
                    'padding:1px 6px;font-size:0.72rem;font-weight:700">'
                    '⚠ BOTH EXCHANGES CROWDED LONG</span><br>'
                )

            st.markdown(
                f'<div class="proj-card" style="border-color:{cg_color};background:{cg_color}18">'
                f'<b style="color:{cg_color}">{cg_icon} {cga}</b><br>'
                f'{div_badge}'
                f'OI: 5m <b>{m5:+.1f}%</b> · 15m <b>{m15:+.1f}%</b> · 1h <b>{h1:+.1f}%</b><br>'
                f'Divergence: <b>{oi_div}</b><br>'
                f'Bybit: <b style="color:{carry_color}">{bybit_rate:+.3f}%/8h</b> '
                f'· carry: <b style="color:{carry_color}">{daily_carry:+.2f}%/day</b><br>'
                f'Binance: <b>{binance_rate:+.3f}%</b> · Spread: <b>{spread:.3f}%</b><br>'
                f'{"<b>" + settle_str + "</b><br>" if settle_str else ""}'
                f'<span style="font-size:0.78rem;color:#aaa">{cg.get("description","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Contextual callouts
            if cga == "OI_PRICE_DIV_LONG":
                st.success("⭐ SPOT DIP: historically +0.96% avg 4h, 70% hit rate")
            elif cga == "OI_BUILDING_PRICE_WEAK":
                st.error("🔴 LONGS TRAPPED: OI building while price falls — exhaustion fingerprint (-0.25% avg 4h, 44% hit)")
            elif cga == "BINANCE_BEARISH_VS_BYBIT":
                st.warning("⚠ Informed money (Binance) is bearish while Bybit longs pay floor rate")
            elif daily_carry >= 1.5:
                st.error(f"🔴 Bybit carry {daily_carry:+.2f}%/day — longs need strong conviction to overcome cost")

    st.markdown("---")

    # ── Row 6: Settlement Cycle + Liquidation Cascade ────────────────────────
    st.markdown("#### Structural Pattern Models")
    struct_c1, struct_c2 = st.columns(2)

    # — Settlement Cycle ——————————————————————————————————————————————————
    settlement = proj.get("funding_settlement", {})
    with struct_c1:
        st.markdown("**Funding Settlement Cycle**")
        s_phase        = settlement.get("phase", "MID_CYCLE")
        s_mins         = settlement.get("mins_to_settlement", 999)
        s_effect       = settlement.get("expected_effect", "UNKNOWN")
        s_conf         = settlement.get("confidence", 0)
        s_funding_sign = settlement.get("current_funding_sign", "NEUTRAL")
        s_pre          = settlement.get("pre_ret_mean")
        s_post         = settlement.get("post_ret_mean")
        s_n            = settlement.get("historical_n", 0)
        s_dubai_time   = settlement.get("dubai_settlement_time", "")
        s_trade_setup  = settlement.get("trade_setup")
        s_setup_desc   = settlement.get("trade_setup_desc", "")

        # Color based on phase and expected effect
        if s_phase == "PRE_SETTLEMENT":
            s_color = "#b71c1c" if "DOWN" in s_effect else "#1b5e20" if "UP" in s_effect else "#e65100"
        elif s_phase == "JUST_SETTLED":
            s_color = "#1b5e20" if "UP" in s_effect else "#b71c1c" if "DOWN" in s_effect else "#1565c0"
        elif s_trade_setup and "FADE" in s_trade_setup:
            s_color = "#e65100"   # upcoming short setup
        else:
            s_color = "#37474f"

        _phase_labels = {
            "PRE_SETTLEMENT": "⚡ PRE-SETTLEMENT",
            "JUST_SETTLED":   "✅ JUST SETTLED",
            "MID_CYCLE":      "🔄 MID-CYCLE",
        }
        s_label     = _phase_labels.get(s_phase, s_phase)
        next_settle = settlement.get("next_settlement_utc", "?")
        dubai_str   = f" · {s_dubai_time}" if s_dubai_time else ""

        pre_str  = f"Pre avg: {s_pre:+.2f}%" if s_pre is not None else "Pre: n/a"
        post_str = f"Post avg: {s_post:+.2f}%" if s_post is not None else "Post: n/a"

        # Trade setup callout block
        setup_html = ""
        if s_setup_desc:
            setup_color = "#b71c1c" if "FADE" in (s_trade_setup or "") else "#1b5e20" if "LONG" in (s_trade_setup or "") else "#e65100"
            setup_html = (
                f'<div style="background:{setup_color}22;border-left:3px solid {setup_color};'
                f'border-radius:4px;padding:6px 10px;margin-top:8px;font-size:0.82rem">'
                f'<b>🎯 {(s_trade_setup or "").replace("_"," ")}</b><br>'
                f'{s_setup_desc}'
                f'</div>'
            )

        st.markdown(
            f'<div class="proj-card" style="border-color:{s_color};background:{s_color}18">'
            f'<b style="color:{s_color};font-size:1.1rem">{s_label}</b><br>'
            f'Next settlement: <b>{next_settle}{dubai_str}</b> ({s_mins:.0f}min away)<br>'
            f'Funding: <b>{s_funding_sign}</b> &nbsp;|&nbsp; '
            f'Expected: <b>{s_effect}</b><br>'
            f'{pre_str} &nbsp;|&nbsp; {post_str} &nbsp;|&nbsp; n={s_n}<br>'
            f'Confidence: <b>{s_conf:.0%}</b><br>'
            f'{setup_html}'
            f'<span style="font-size:0.78rem;color:#aaa;margin-top:6px;display:block">'
            f'{settlement.get("description","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Phase", s_phase.replace("_", " "))
        sc2.metric("Next Settle", f"{next_settle}{dubai_str}")
        sc3.metric("Confidence", f"{s_conf:.0%}")

    # — Liquidation Cascade ——————————————————————————————————————————————
    liq_cas = proj.get("liq_cascade", {})
    with struct_c2:
        st.markdown("**Liquidation Cascade Detector**")
        lc_state    = liq_cas.get("state", "NORMAL")
        lc_detected = liq_cas.get("cascade_detected", False)
        lc_z        = liq_cas.get("liq_zscore", 0)
        lc_candles  = liq_cas.get("candles_since_cascade")
        lc_wick     = liq_cas.get("wick_ratio", 0)
        lc_vol_r    = liq_cas.get("volume_ratio", 0)
        lc_4h_avg   = liq_cas.get("post_cascade_avg_4h", 0)
        lc_hit      = liq_cas.get("post_cascade_hit_rate", 0)
        lc_n        = liq_cas.get("historical_n", 0)
        lc_conf     = liq_cas.get("confidence", 0)

        _lc_colors = {
            "CASCADE_IN_PROGRESS": "#b71c1c",
            "POST_CASCADE_ENTRY":  "#1b5e20",
            "POST_CASCADE_WATCH":  "#e65100",
            "NORMAL":              "#37474f",
        }
        _lc_icons = {
            "CASCADE_IN_PROGRESS": "⚡ CASCADE IN PROGRESS",
            "POST_CASCADE_ENTRY":  "✅ POST-CASCADE ENTRY",
            "POST_CASCADE_WATCH":  "👀 POST-CASCADE WATCH",
            "NORMAL":              "😴 NO CASCADE",
        }
        lc_color = _lc_colors.get(lc_state, "#37474f")
        lc_label = _lc_icons.get(lc_state, lc_state)

        candle_str = f"Candles since: <b>{lc_candles}</b><br>" if lc_candles is not None else ""
        hist_str   = f"Historical: <b>{lc_4h_avg:+.1f}%</b> avg 4h, <b>{lc_hit:.0%}</b> hit (n={lc_n})" if lc_n >= 5 else ""

        st.markdown(
            f'<div class="proj-card" style="border-color:{lc_color};background:{lc_color}18">'
            f'<b style="color:{lc_color};font-size:1.1rem">{lc_label}</b><br>'
            f'Liq z-score: <b>{lc_z:.1f}σ</b> &nbsp;|&nbsp; '
            f'Wick ratio: <b>{lc_wick:.1f}x</b> &nbsp;|&nbsp; '
            f'Vol spike: <b>{lc_vol_r:.1f}x</b><br>'
            f'{candle_str}'
            f'{hist_str + "<br>" if hist_str else ""}'
            f'Confidence: <b>{lc_conf:.0%}</b><br>'
            f'<span style="font-size:0.8rem;color:#aaa">{liq_cas.get("description","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("State", lc_state.replace("_", " "))
        lc2.metric("Liq z-score", f"{lc_z:.1f}σ")
        lc3.metric("4h Post-Cas avg", f"{lc_4h_avg:+.1f}%" if lc_n >= 5 else "N/A")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SESSION TIMING
# ═════════════════════════════════════════════════════════════════════════════

with tab_timing:
    ohlcv = data.get("ohlcv")
    if ohlcv is None:
        st.warning("No OHLCV data.")
    else:
        price_col = "price" if "price" in ohlcv.columns else "close"
        df        = ohlcv.copy()
        df["return"]     = df[price_col].pct_change()
        df["abs_return"] = df["return"].abs()
        df["hour"]       = df.index.hour
        df["session"]    = df["hour"].apply(classify_session)

        # Session cards
        st.markdown("#### Session Performance")
        scols = st.columns(4)
        _s_styles = {
            "bullish":     ("#e8f5e9", "#2e7d32", "✅ FAVORABLE"),
            "conditional": ("#e3f2fd", "#1565c0", "🔀 CONDITIONAL"),
            "neutral":     ("#fff3e0", "#e65100", "⚠️ VOLATILE"),
            "bearish":     ("#ffebee", "#c62828", "⛔ AVOID"),
        }
        for i, (sess, info) in enumerate(SESSION_MAP.items()):
            bg, fc, label = _s_styles.get(info["bias"], ("#424242", "#fff", ""))
            scols[i].markdown(
                f'<div style="background:{bg};padding:14px;border-radius:8px;'
                f'border-left:5px solid {fc}">'
                f'<b style="color:{fc}">{sess}</b><br>'
                f'<span style="color:#555;font-size:0.8rem">{info["et"]} ET</span><br>'
                f'<span style="font-size:1.6rem;font-weight:800;color:{fc}">'
                f'{info["avg_bps"]:+.1f}<span style="font-size:0.8rem"> bps/hr</span></span><br>'
                f'<span style="font-size:0.75rem;color:{fc}">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Hourly return + volatility charts
        now_h     = utc_now.hour
        hourly_ret = df.groupby("hour")["return"].mean() * 10000
        hourly_vol = df.groupby("hour")["abs_return"].mean() * 100

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("#### Avg Return by Hour (UTC)")
            r_colors = ["#66bb6a" if v > 5 else "#e57373" if v < -10 else "#90a4ae"
                        for v in hourly_ret]
            fig_r = go.Figure(go.Bar(x=hourly_ret.index, y=hourly_ret.values,
                                     marker_color=r_colors))
            fig_r.add_vline(x=now_h, line_color="#ffeb3b", line_width=2,
                            annotation_text=f"NOW ({now_h:02d}h)")
            # Session shading
            fig_r.add_vrect(x0=-0.5, x1=7.5,  fillcolor="blue",   opacity=0.04)
            fig_r.add_vrect(x0=7.5,  x1=12.5, fillcolor="green",  opacity=0.04)
            fig_r.add_vrect(x0=12.5, x1=20.5, fillcolor="orange", opacity=0.04)
            fig_r.add_vrect(x0=20.5, x1=23.5, fillcolor="purple", opacity=0.04)
            fig_r.update_layout(
                template="plotly_dark", height=300,
                xaxis_title="Hour (UTC)", yaxis_title="bps",
                xaxis=dict(dtick=1), margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with ch2:
            st.markdown("#### Volatility by Hour (Avg |Move|%)")
            fig_v = go.Figure(go.Bar(x=hourly_vol.index, y=hourly_vol.values,
                                     marker_color="#ffb74d"))
            fig_v.add_vline(x=now_h, line_color="#ffeb3b", line_width=2,
                            annotation_text="NOW")
            fig_v.update_layout(
                template="plotly_dark", height=300,
                xaxis_title="Hour (UTC)", yaxis_title="|Move| %",
                xaxis=dict(dtick=1), margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_v, use_container_width=True)

        # Asia deep dive
        st.markdown("---")
        st.markdown("#### 🌏 Asia Deep Dive")
        a1, a2 = st.columns(2)
        with a1:
            early = df[df["hour"].isin([0,1,2,3])]["return"].mean() * 10000
            late  = df[df["hour"].isin([4,5,6,7])]["return"].mean() * 10000
            for label, val, note in [
                ("Early Asia (00-04 UTC / 8pm-12am ET)", early,
                 "00:00–01:00 UTC are the worst hours"),
                ("Late Asia (04-08 UTC / 12am-4am ET)", late,
                 "04:00–05:00 slightly positive; 07:00 UTC dumps before London"),
            ]:
                col = "#e57373" if val < 0 else "#66bb6a"
                st.markdown(
                    f'<div style="background:#1e272e;border-radius:8px;padding:12px;'
                    f'margin-bottom:8px">'
                    f'<b>{label}</b><br>'
                    f'<span style="font-size:1.4rem;font-weight:800;color:{col}">'
                    f'{val:+.1f} bps/hr</span><br>'
                    f'<span style="font-size:0.78rem;color:#aaa">{note}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with a2:
            day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            day_vals  = [ASIA_DAY_BPS[d] for d in day_order]
            fig_day = go.Figure(go.Bar(
                x=day_order, y=day_vals,
                marker_color=["#66bb6a" if v > 0 else "#e57373" for v in day_vals],
                text=[f"{v:+.0f}" for v in day_vals], textposition="outside",
            ))
            fig_day.add_hline(y=0, line_color="#546e7a", line_width=0.5)
            fig_day.update_layout(
                title="Asia by Day (Mon/Wed/Fri tradeable | Thu/Sat/Sun avoid)",
                template="plotly_dark", height=260, margin=dict(t=36, b=10),
            )
            st.plotly_chart(fig_day, use_container_width=True)

        # Weekend Insight
        if mkt.get("is_weekend"):
            st.markdown("---")
            st.markdown("#### 📉 Weekend Market Dynamics")
            w1, w2 = st.columns(2)
            with w1:
                st.info("Weekend liquidity is 30-50% lower than weekday averages. This amplifies every order, making manipulation (spoof-walls, stop-hunts) much cheaper for Market Makers.")
            with w2:
                st.warning("Historically, Sunday (22:00 UTC) marks the 'Sunday Dump' as TradFi markets prepare for Monday open. Exit weekend longs before this window.")

        # ── Kill Zones Deep Dive ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎯 Kill Zones — Market Maker Stop Hunt Analysis")
        st.caption(
            "Kill zones are recurring hours where market makers consistently hunt stops. "
            "Identified from 90 days / 2,128 hourly observations. "
            "Avoid entering positions 30 min before these windows."
        )

        # Per-hour stats computed from OHLCV
        df_h = df.copy()
        df_h["dow"] = df_h.index.dayofweek
        df_h["fwd_4h"] = df_h[price_col].pct_change(4).shift(-4) * 100
        df_h = df_h.dropna(subset=["return", "fwd_4h"])

        hourly_stats = {}
        for h in range(24):
            sub = df_h[df_h["hour"] == h]
            hourly_stats[h] = {
                "n": len(sub),
                "mean_1h": sub["return"].mean() * 100,
                "hit_1h": (sub["return"] > 0).mean(),
                "mean_4h": sub["fwd_4h"].mean(),
                "hit_4h": (sub["fwd_4h"] > 0).mean(),
            }

        # Classify each hour
        KILL_HOURS   = {0, 1, 3, 7, 11, 18}   # from bias table + data confirmation
        STRONG_HOURS = {10, 19, 20, 22}

        kz_now = now_h in KILL_HOURS
        st_now = now_h in STRONG_HOURS

        # Current hour status banner
        hrs_until_kill   = min((h - now_h) % 24 for h in KILL_HOURS)
        hrs_until_strong = min((h - now_h) % 24 for h in STRONG_HOURS)
        next_kill_h   = (now_h + hrs_until_kill) % 24
        next_strong_h = (now_h + hrs_until_strong) % 24

        kz_c1, kz_c2, kz_c3 = st.columns(3)
        with kz_c1:
            if kz_now:
                st.error(f"🚨 IN KILL ZONE NOW ({now_h:02d}:00 UTC) — AVOID NEW ENTRIES")
            else:
                st.markdown(
                    f'<div style="background:#1e272e;border-left:4px solid #e57373;'
                    f'padding:12px;border-radius:6px">'
                    f'<b style="color:#e57373">Next Kill Zone</b><br>'
                    f'<span style="font-size:1.4rem;font-weight:800">{next_kill_h:02d}:00 UTC</span><br>'
                    f'<span style="color:#aaa">in ~{hrs_until_kill}h</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with kz_c2:
            if st_now:
                st.success(f"✅ IN STRONG WINDOW NOW ({now_h:02d}:00 UTC) — FAVORABLE FOR ENTRY")
            else:
                st.markdown(
                    f'<div style="background:#1e272e;border-left:4px solid #66bb6a;'
                    f'padding:12px;border-radius:6px">'
                    f'<b style="color:#66bb6a">Next Strong Window</b><br>'
                    f'<span style="font-size:1.4rem;font-weight:800">{next_strong_h:02d}:00 UTC</span><br>'
                    f'<span style="color:#aaa">in ~{hrs_until_strong}h</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with kz_c3:
            cur_stats = hourly_stats.get(now_h, {})
            kz_bg = "#b71c1c22" if kz_now else "#1b5e2022" if st_now else "#263238"
            kz_bd = "#b71c1c" if kz_now else "#66bb6a" if st_now else "#546e7a"
            st.markdown(
                f'<div style="background:{kz_bg};border-left:4px solid {kz_bd};'
                f'padding:12px;border-radius:6px">'
                f'<b>Current Hour Stats ({now_h:02d}:00 UTC)</b><br>'
                f'1h avg: <b>{cur_stats.get("mean_1h",0):+.2f}%</b> · '
                f'hit: <b>{cur_stats.get("hit_1h",0.5):.0%}</b><br>'
                f'4h avg: <b>{cur_stats.get("mean_4h",0):+.2f}%</b> · '
                f'hit: <b>{cur_stats.get("hit_4h",0.5):.0%}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Kill zone detail cards
        st.markdown("#### Confirmed Kill Zones")
        kz_cols = st.columns(len(KILL_HOURS))
        kz_defs = {
            0:  {"et": "8 PM",  "note": "Asia open stop hunt. OI falling = -0.45% avg."},
            1:  {"et": "9 PM",  "note": "Sat/Sun/Thu worst (-0.71%, -0.42%, -0.22%)."},
            3:  {"et": "11 PM", "note": "Thu/Wed most reliable: -0.97%/-0.63%, 15% hit."},
            7:  {"et": "3 AM",  "note": "London open fake-out. Mon/Tue/Sat/-0.35-0.70%."},
            11: {"et": "7 AM",  "note": "Pre-NYC flush. Wed worst (-0.15%, 31% hit)."},
            18: {"et": "2 PM",  "note": "MOST CONSISTENT. All days negative. -0.46% avg, 39% hit."},
        }
        for i, h in enumerate(sorted(KILL_HOURS)):
            stats = hourly_stats[h]
            info  = kz_defs.get(h, {})
            is_now = (now_h == h)
            border = "#ff1744" if is_now else "#b71c1c"
            kz_cols[i].markdown(
                f'<div style="background:#1a0000;border:2px solid {border};'
                f'border-radius:8px;padding:12px;text-align:center">'
                f'{"🚨 " if is_now else ""}'
                f'<span style="font-size:1.3rem;font-weight:800;color:#ef9a9a">{h:02d}:00 UTC</span><br>'
                f'<span style="color:#aaa;font-size:0.75rem">{info.get("et","")} ET</span><br>'
                f'<span style="font-size:1.1rem;color:#e57373;font-weight:700">'
                f'{stats["mean_1h"]:+.2f}%</span> avg<br>'
                f'<span style="color:#ef9a9a">{stats["hit_1h"]:.0%} hit rate</span><br>'
                f'<span style="font-size:0.72rem;color:#888">{info.get("note","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Strong Windows (High-Probability Entry)")
        sw_cols = st.columns(len(STRONG_HOURS))
        sw_defs = {
            10: {"et": "6 AM",  "note": "Pre-NYC build. +23.4 bps bias. Strong vs 90d data."},
            19: {"et": "3 PM",  "note": "Post-kill bounce. Mon +0.44%, 77% hit. Best follow-through."},
            20: {"et": "4 PM",  "note": "NYSE close flow. Wed: +1.19% avg, 69% hit."},
            22: {"et": "6 PM",  "note": "Late NYC ramp. Fri: +0.57% avg, 75% hit."},
        }
        for i, h in enumerate(sorted(STRONG_HOURS)):
            stats = hourly_stats[h]
            info  = sw_defs.get(h, {})
            is_now = (now_h == h)
            border = "#00e676" if is_now else "#2e7d32"
            sw_cols[i].markdown(
                f'<div style="background:#001a00;border:2px solid {border};'
                f'border-radius:8px;padding:12px;text-align:center">'
                f'{"✅ " if is_now else ""}'
                f'<span style="font-size:1.3rem;font-weight:800;color:#a5d6a7">{h:02d}:00 UTC</span><br>'
                f'<span style="color:#aaa;font-size:0.75rem">{info.get("et","")} ET</span><br>'
                f'<span style="font-size:1.1rem;color:#66bb6a;font-weight:700">'
                f'{stats["mean_1h"]:+.2f}%</span> avg<br>'
                f'<span style="color:#a5d6a7">{stats["hit_1h"]:.0%} hit rate</span><br>'
                f'<span style="font-size:0.72rem;color:#888">{info.get("note","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Hour × Day heatmap
        st.markdown("---")
        st.markdown("#### Hour × Day Heatmap (1h Avg Return %, 90 days)")
        st.caption("Red = kill zone. Green = strong window. Most reliable combos labeled.")

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heat_z = np.zeros((7, 24))
        heat_text = [["" for _ in range(24)] for _ in range(7)]
        for h in range(24):
            for d in range(7):
                sub = df_h[(df_h["hour"] == h) & (df_h["dow"] == d)]["return"] * 100
                if len(sub) >= 4:
                    val = sub.mean()
                    heat_z[d, h] = val
                    if abs(val) > 0.5:
                        heat_text[d][h] = f"{val:+.1f}%"

        fig_heat = go.Figure(go.Heatmap(
            z=heat_z,
            x=list(range(24)),
            y=dow_names,
            text=heat_text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#b71c1c"], [0.35, "#c62828"], [0.48, "#263238"],
                [0.52, "#263238"], [0.65, "#1b5e20"], [1.0, "#00c853"],
            ],
            zmid=0,
            zmin=-1.5, zmax=1.5,
            colorbar=dict(title="1h Ret %"),
        ))
        # Mark kill zones
        for h in KILL_HOURS:
            fig_heat.add_vline(x=h, line_color="#ef9a9a", line_width=1, line_dash="dot")
        for h in STRONG_HOURS:
            fig_heat.add_vline(x=h, line_color="#66bb6a", line_width=1, line_dash="dot")
        fig_heat.add_vline(x=now_h, line_color="#ffeb3b", line_width=2,
                           annotation_text=f"NOW", annotation_font_color="#ffeb3b")
        fig_heat.update_layout(
            template="plotly_dark", height=300,
            xaxis=dict(title="Hour UTC", dtick=1),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Most reliable combos table
        with st.expander("📊 Top 10 Kill Zone Combos (Hour × Day)", expanded=False):
            combo_data = []
            for h in range(24):
                for d in range(7):
                    sub = df_h[(df_h["hour"]==h) & (df_h["dow"]==d)]["return"].dropna() * 100
                    if len(sub) >= 8:
                        consistency = abs((sub > 0).mean() - 0.5) * 2
                        combo_data.append({
                            "Hour (UTC)": f"{h:02d}:00",
                            "ET": f"{(h-4)%24:02d}:00",
                            "Day": dow_names[d],
                            "n": len(sub),
                            "Avg 1h Ret": f"{sub.mean():+.2f}%",
                            "Hit Rate": f"{(sub>0).mean():.0%}",
                            "Consistency": f"{consistency:.2f}",
                            "Type": "🔴 Kill" if sub.mean() < -0.4 else ("🟢 Strong" if sub.mean() > 0.4 else "⚪ Neutral"),
                        })
            combo_df = pd.DataFrame(combo_data)
            kills_df = combo_df[combo_df["Type"] == "🔴 Kill"].nlargest(10, "n")
            st.dataframe(kills_df, use_container_width=True, hide_index=True)

        with st.expander("📊 Top 10 Strong Combos (Hour × Day)", expanded=False):
            combo_df2 = pd.DataFrame(combo_data) if combo_data else pd.DataFrame()
            if not combo_df2.empty:
                strong_df = combo_df2[combo_df2["Type"] == "🟢 Strong"].nlargest(10, "n")
                st.dataframe(strong_df, use_container_width=True, hide_index=True)

        # Timing playbook — updated
        st.markdown("---")
        st.markdown("#### Timing Playbook")
        st.markdown("""
| Rule | Hour (UTC) | ET Time | Data |
|------|-----------|---------|------|
| ⭐ **Best entry** | 10:00 UTC | 6:00 AM | +23.4 bps bias |
| ✅ **Strong window** | 19:00–22:00 UTC | 3–6 PM | +15–19 bps avg |
| ✅ **Best combo** | 21:00 Tue / 07:00 Wed | 5 PM Tue / 3 AM Wed | +1.60% / 85% hit |
| 🔴 **Kill zone #1** | 18:00 UTC | 2:00 PM | −0.46% avg, 39% hit, ALL days |
| 🔴 **Kill zone #2** | 03:00 Thu | 11 PM Wed | −0.97% avg, 15% hit |
| 🔴 **Kill zone #3** | 23:00 Wed | 7 PM Wed | −1.22% avg, 23% hit |
| 🔴 **Asia bleed** | 00:00–01:00 UTC | 8–9 PM | −0.12–0.29% avg |
| ⚠️ **Thursday** | All day | — | −0.63% avg 4h, 41% hit |
| ⚠️ **Sunday** | All day | — | −0.40% avg 4h, 44% hit |
| 📏 **Hold window** | — | — | 4–8h max, decays after |
""")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXCHANGE INTEL
# ═════════════════════════════════════════════════════════════════════════════

with tab_exchange:
    deriv = data.get("derivatives")
    if deriv is None:
        st.warning("No derivatives data.")
    else:
        active = deriv[deriv["open_interest_usd"] > 10_000].copy()
        st.subheader(f"Cross-Exchange Positioning — {len(active)} Active Venues")

        ec1, ec2 = st.columns(2)

        with ec1:
            st.markdown("#### OI Distribution")
            oi_top = active.nlargest(12, "open_interest_usd")
            fig_oi = go.Figure(go.Bar(
                y=oi_top["exchange"].str[:22],
                x=oi_top["open_interest_usd"] / 1e6,
                orientation="h", marker_color="#42a5f5",
                text=[f"${v/1e6:.1f}M" for v in oi_top["open_interest_usd"]],
                textposition="outside",
            ))
            fig_oi.update_layout(
                template="plotly_dark", height=420,
                xaxis_title="OI ($M)", margin=dict(l=160, t=10),
            )
            st.plotly_chart(fig_oi, use_container_width=True)

            total_oi_ex = active["open_interest_usd"].sum()
            top2 = active.nlargest(2, "open_interest_usd")
            st.metric(
                "Top-2 OI share",
                f"{top2['open_interest_usd'].sum()/total_oi_ex:.0%}",
                delta=f"{top2.iloc[0]['exchange'][:12]} + {top2.iloc[1]['exchange'][:12]}",
            )

        with ec2:
            st.markdown("#### Funding Rate Divergence")
            fr_sorted = active.sort_values("funding_rate")
            fig_fr2 = go.Figure(go.Bar(
                y=fr_sorted["exchange"].str[:22],
                x=fr_sorted["funding_rate"],
                orientation="h",
                marker_color=["#e57373" if x < 0 else "#66bb6a"
                               for x in fr_sorted["funding_rate"]],
            ))
            fig_fr2.add_vline(x=0, line_color="#546e7a", line_width=1)
            fig_fr2.update_layout(
                template="plotly_dark",
                height=max(400, len(fr_sorted) * 18),
                xaxis_title="Funding Rate", margin=dict(l=160, t=10),
            )
            st.plotly_chart(fig_fr2, use_container_width=True)

        # Churning
        st.markdown("#### Churning Detection (Vol / OI)")
        st.caption("Ratio > 10x = suspicious. Possible wash trading or rapid position cycling.")
        active["vol_oi"] = active["volume_24h_usd"] / active["open_interest_usd"].replace(0, np.nan)
        churning = active.nlargest(10, "vol_oi")
        fig_ch = go.Figure(go.Bar(
            y=churning["exchange"].str[:22], x=churning["vol_oi"],
            orientation="h",
            marker_color=["#e57373" if v > 10 else "#ffb74d" if v > 5 else "#66bb6a"
                          for v in churning["vol_oi"]],
            text=[f"{v:.1f}x" for v in churning["vol_oi"]], textposition="outside",
        ))
        fig_ch.update_layout(
            template="plotly_dark", height=340,
            xaxis_title="Vol / OI", margin=dict(l=160, t=10),
        )
        st.plotly_chart(fig_ch, use_container_width=True)

        with st.expander("Full Exchange Table"):
            display_df = active[[
                "exchange","funding_rate","open_interest_usd",
                "volume_24h_usd","basis_pct","spread","price",
            ]].sort_values("open_interest_usd", ascending=False)
            st.dataframe(display_df.style.format({
                "funding_rate": "{:.6f}", "open_interest_usd": "${:,.0f}",
                "volume_24h_usd": "${:,.0f}", "basis_pct": "{:.4f}%",
                "spread": "{:.4f}", "price": "${:.4f}",
            }), use_container_width=True, height=500)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BTC CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

with tab_btc:
    btc  = data.get("btc")
    ohlcv = data.get("ohlcv")

    if btc is None or ohlcv is None:
        st.warning("Missing BTC or FART data.")
    else:
        btc_col  = "price" if "price" in btc.columns else "close"
        fart_col = "price" if "price" in ohlcv.columns else "close"

        btc_h  = btc[[btc_col,"volume"]].resample("1h").last().dropna()
        btc_h.columns = ["btc_price","btc_volume"]
        btc_h["btc_return"] = btc_h["btc_price"].pct_change()

        fart_h = ohlcv[[fart_col]].resample("1h").last().dropna()
        fart_h.columns = ["fart_price"]
        fart_h["fart_return"] = fart_h["fart_price"].pct_change()

        merged = btc_h.join(fart_h, how="inner").dropna(subset=["btc_return","fart_return"])
        corr   = merged["btc_return"].corr(merged["fart_return"])
        valid  = merged[["btc_return","fart_return"]].dropna()
        beta   = np.polyfit(valid["btc_return"], valid["fart_return"], 1)[0] if len(valid) > 50 else 0
        merged["rolling_corr"] = merged["btc_return"].rolling(24).corr(merged["fart_return"])

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Correlation", f"{corr:.3f}")
        bm2.metric("Beta", f"{beta:.2f}x")
        bm3.metric("BTC Regime", mkt.get("btc_regime","?"))
        bm4.metric("BTC 24h", f"{mkt.get('btc_ret_24h',0):.1%}")

        st.info(
            f"FART moves **{beta:.1f}x** BTC on average. "
            f"When correlation goes negative, moves are **2.5x bigger** — manipulation signal."
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("#### Normalized Price Overlay")
            btc_norm  = merged["btc_price"]  / merged["btc_price"].iloc[0]  * 100
            fart_norm = merged["fart_price"] / merged["fart_price"].iloc[0] * 100
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(x=merged.index, y=btc_norm,
                                        name="BTC", line=dict(color="#f7931a", width=2)))
            fig_ov.add_trace(go.Scatter(x=merged.index, y=fart_norm,
                                        name="FART", line=dict(color="#42a5f5", width=2)))
            fig_ov.update_layout(
                template="plotly_dark", height=320,
                yaxis_title="Indexed (start=100)", hovermode="x unified",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_ov, use_container_width=True)

        with bc2:
            st.markdown("#### Rolling 24h Correlation")
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=merged.index, y=merged["rolling_corr"],
                                        line=dict(color="#42a5f5", width=1.5)))
            fig_rc.add_hline(y=0, line_color="#e57373", line_dash="dash")
            fig_rc.add_hrect(y0=-1, y1=0, fillcolor="#e57373", opacity=0.05)
            fig_rc.update_layout(
                template="plotly_dark", height=320,
                yaxis_title="24h Correlation", yaxis_range=[-1,1],
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_rc, use_container_width=True)

        # Regime table
        st.markdown("#### FART Behavior by BTC Regime")
        merged["btc_ret_24h"] = merged["btc_price"].pct_change(24)

        def _regime(r):
            if pd.isna(r): return None
            if r >  0.03: return "BTC Strong Rally (>3%)"
            if r >  0.01: return "BTC Mild Rally (1-3%)"
            if r > -0.01: return "BTC Flat (−1%→1%)"
            if r > -0.03: return "BTC Mild Dump (−3→−1%)"
            return "BTC Strong Dump (<−3%)"

        merged["regime"] = merged["btc_ret_24h"].apply(_regime)
        regime_order = [
            "BTC Strong Rally (>3%)", "BTC Mild Rally (1-3%)",
            "BTC Flat (−1%→1%)", "BTC Mild Dump (−3→−1%)", "BTC Strong Dump (<−3%)",
        ]
        regime_stats = merged.groupby("regime").agg(
            avg_fart_bps=("fart_return", lambda x: x.mean() * 10000),
            observations=("fart_return", "count"),
        ).reindex(regime_order).dropna()
        regime_stats["action"] = [
            "✅ LONG bias — tailwind",
            "✅ Mild long bias",
            "⚪ Neutral — use other signals",
            "⚠️ Caution on longs",
            "⛔ AVOID longs — 1.6x downside beta",
        ][:len(regime_stats)]
        st.dataframe(regime_stats, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRADE RULES
# ═════════════════════════════════════════════════════════════════════════════

with tab_rules:
    c_l, c_r = st.columns(2)

    with c_l:
        st.markdown("""
### Entry Rules

| Composite | Conviction | Notes |
|-----------|------------|-------|
| > +0.4 / < −0.4 | HIGH | Primary trigger. 88% hit rate (4h) |
| +0.3→0.4 / −0.3→−0.4 | MEDIUM | 65% hit rate — use with confirming factors |
| +0.2→0.3 / −0.2→−0.3 | LOW | Only with session + BTC alignment |

### Timing Filters

| Filter | Rule |
|--------|------|
| **Session** | Prefer London or Late NYC |
| **Asia** | Longs OK Mon/Wed/Fri Late Asia. SHORT signals = highest quality. |
| **Kill zone** | No new entries at 18:00 UTC (2pm ET) |
| **Day of week** | Prefer Mon/Tue/Fri |

### Exit Rules

| Condition | Action |
|-----------|--------|
| 4–8 hours elapsed | Close position |
| Composite flips sign | Immediate close |
| Hit 18:00 UTC | Close or tighten stop |
| 24h hold | MUST close — signal reverses |
""")

    with c_r:
        st.markdown("""
### Conviction Modifiers

| Factor | Upgrades | Downgrades |
|--------|----------|------------|
| Funding | Aligns with direction | Opposes direction |
| BTC regime | Rally+long / Dump+short | Dump+long / Rally+short |
| Session | London / Late NYC | Asia |
| Volume spike | Vol > 2× avg | — |

### Risk Management

| Rule | Detail |
|------|--------|
| **Position sizing** | HIGH = full, MED = 50%, LOW = 25% |
| **Stop loss** | −3% from entry |
| **Take profit** | +1% at 4h, trail from there |
| **Correlation break** | 24h corr goes negative → cut 50% |

### The MM Playbook

1. **Accumulate** — quiet period (2-3 days low volume)
2. **Engineer** — push price on thin books (Asia or 2pm ET)
3. **Cascade** — trigger liquidations
4. **Harvest** — exit at inflated/deflated prices
5. **Reset** — mean-reverts within 24h

**Our edge:** detect Stage 1–2 via composite, enter BEFORE the cascade.
Signal: 50 bps quintile spread, 88% hit rate at high conviction.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"UTC {utc_now.strftime('%Y-%m-%d %H:%M')} | "
    f"Auto-refreshes every 5 min | Not financial advice."
)
