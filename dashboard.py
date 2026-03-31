"""
Fartcoin Alpha — Trade Desk Dashboard

Actionable dashboard for the trading desk. Shows:
- Current trade signal and recommended action
- Entry/exit timing by trading session
- BTC regime context
- Cross-exchange positioning for manipulation detection

Run: streamlit run dashboard.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FART Trade Desk",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent / "data"

SESSION_MAP = {
    "Asia":     {"utc": "00:00-08:00", "et": "8pm-4am",  "bias": "conditional", "avg_bps": -9.3},
    "London":   {"utc": "08:00-13:00", "et": "4am-9am",  "bias": "bullish", "avg_bps": 4.5},
    "NYC":      {"utc": "13:00-21:00", "et": "9am-5pm",  "bias": "neutral", "avg_bps": -2.0},
    "Late NYC": {"utc": "21:00-00:00", "et": "5pm-8pm",  "bias": "bullish", "avg_bps": 10.7},
}

HOURLY_BIAS = {
    0: -21.8, 1: -17.5, 2: 1.1, 3: -17.4, 4: 7.9, 5: 5.1,
    6: -9.7, 7: -21.7, 8: 7.3, 9: 6.0, 10: 23.4, 11: -15.1,
    12: 0.8, 13: -9.2, 14: -9.9, 15: -3.8, 16: 5.3, 17: -5.6,
    18: -26.5, 19: 15.8, 20: 18.2, 21: 10.0, 22: 18.9, 23: 3.2,
}

# Asia sub-sessions and day-of-week patterns
ASIA_SUB = {
    "Early Asia (00-04)": {"et": "8pm-12am", "avg_bps": -15.9, "hours": [0, 1, 2, 3]},
    "Late Asia (04-08)":  {"et": "12am-4am", "avg_bps": -5.4,  "hours": [4, 5, 6, 7]},
}
ASIA_DAY_BPS = {
    "Mon": 3.2, "Tue": -13.1, "Wed": 2.6, "Thu": -34.8,
    "Fri": 10.9, "Sat": -24.5, "Sun": -17.3,
}


def classify_session(hour):
    if 0 <= hour < 8: return "Asia"
    elif 8 <= hour < 13: return "London"
    elif 13 <= hour < 21: return "NYC"
    else: return "Late NYC"


def classify_asia_sub(hour):
    if 0 <= hour < 4: return "Early Asia"
    elif 4 <= hour < 8: return "Late Asia"
    return None


def get_current_utc():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_all_data():
    data = {}
    files = {
        "ohlcv": "FARTCOIN_ohlcv_hourly.csv",
        "ohlcv_daily": "FARTCOIN_ohlcv.csv",
        "derivatives": "FARTCOIN_derivatives_snapshot.csv",
        "signals": "signals.csv",
        "trades": "trades.csv",
        "btc": "bitcoin_cg_chart.csv",
        "funding": "FARTCOINUSDT_funding.csv",
        "lsr": "FARTCOINUSDT_lsr.csv",
    }
    for key, fname in files.items():
        f = DATA_DIR / fname
        if f.exists():
            if key in ("derivatives", "trades"):
                data[key] = pd.read_csv(f)
            else:
                data[key] = pd.read_csv(f, index_col=0, parse_dates=True)
    return data


data = load_all_data()


# ---------------------------------------------------------------------------
# Computed state
# ---------------------------------------------------------------------------

def compute_market_state():
    """Compute current market state for action panel."""
    state = {}
    now = get_current_utc()
    state["utc_hour"] = now.hour
    state["session"] = classify_session(now.hour)
    state["session_info"] = SESSION_MAP[state["session"]]
    state["hourly_bias_bps"] = HOURLY_BIAS.get(now.hour, 0)
    state["weekday"] = now.strftime("%A")
    state["weekday_short"] = now.strftime("%a")
    state["asia_sub"] = classify_asia_sub(now.hour)
    state["asia_day_bps"] = ASIA_DAY_BPS.get(state["weekday_short"], 0)

    # Compute Late NYC carry (prior 3h momentum)
    ohlcv_carry = data.get("ohlcv")
    if ohlcv_carry is not None and state["session"] == "Asia":
        pc = "price" if "price" in ohlcv_carry.columns else "close"
        ret_3h = ohlcv_carry[pc].pct_change(3).iloc[-1] if len(ohlcv_carry) > 3 else 0
        state["late_nyc_carry"] = ret_3h
    else:
        state["late_nyc_carry"] = 0

    # Signal state
    signals = data.get("signals")
    if signals is not None:
        latest = signals.dropna(subset=["composite"]).iloc[-1]
        state["composite"] = latest["composite"]
        state["signals"] = {c: latest[c] for c in signals.columns if c.startswith("sig_") and not np.isnan(latest[c])}
    else:
        state["composite"] = 0
        state["signals"] = {}

    # Derivatives
    deriv = data.get("derivatives")
    if deriv is not None:
        active = deriv[deriv["open_interest_usd"] > 10000]
        state["avg_funding"] = active["funding_rate"].mean()
        state["total_oi"] = active["open_interest_usd"].sum()
        state["total_vol"] = active["volume_24h_usd"].sum()
        state["oi_vol_ratio"] = state["total_oi"] / state["total_vol"] if state["total_vol"] > 0 else 0
        state["funding_range"] = active["funding_rate"].max() - active["funding_rate"].min()
        hhi = ((active["open_interest_usd"] / state["total_oi"]) ** 2).sum()
        state["hhi"] = hhi

        # Risk score
        risk = 0
        if abs(state["avg_funding"]) > 0.01: risk += 2
        if state["oi_vol_ratio"] < 0.5: risk += 1
        if hhi > 0.15: risk += 2
        if state["funding_range"] > 0.05: risk += 1
        top_share = active["open_interest_usd"].max() / state["total_oi"]
        if top_share > 0.3: risk += 1
        state["risk_score"] = risk
    else:
        state["avg_funding"] = 0
        state["risk_score"] = 0

    # BTC context
    btc = data.get("btc")
    ohlcv = data.get("ohlcv")
    if btc is not None and ohlcv is not None:
        btc_col = "price" if "price" in btc.columns else "close"
        fart_col = "price" if "price" in ohlcv.columns else "close"
        btc_ret_24h = btc[btc_col].pct_change(24).iloc[-1] if len(btc) > 24 else 0
        state["btc_price"] = btc[btc_col].iloc[-1]
        state["btc_ret_24h"] = btc_ret_24h
        if btc_ret_24h > 0.03: state["btc_regime"] = "Strong Rally"
        elif btc_ret_24h > 0.01: state["btc_regime"] = "Mild Rally"
        elif btc_ret_24h > -0.01: state["btc_regime"] = "Flat"
        elif btc_ret_24h > -0.03: state["btc_regime"] = "Mild Dump"
        else: state["btc_regime"] = "Strong Dump"

        state["fart_price"] = ohlcv[fart_col].iloc[-1]
    else:
        state["btc_regime"] = "Unknown"
        state["btc_price"] = 0
        state["fart_price"] = 0

    return state


def determine_action(state):
    """Determine the trade action based on all signals."""
    composite = state.get("composite", 0)
    session = state.get("session", "")
    funding = state.get("avg_funding", 0)
    btc_regime = state.get("btc_regime", "")
    hourly_bias = state.get("hourly_bias_bps", 0)

    # Primary signal
    if composite > 0.4:
        direction = "LONG"
        conviction = "HIGH"
    elif composite > 0.3:
        direction = "LONG"
        conviction = "MEDIUM"
    elif composite > 0.2:
        direction = "LONG"
        conviction = "LOW"
    elif composite < -0.4:
        direction = "SHORT"
        conviction = "HIGH"
    elif composite < -0.3:
        direction = "SHORT"
        conviction = "MEDIUM"
    elif composite < -0.2:
        direction = "SHORT"
        conviction = "LOW"
    else:
        direction = "FLAT"
        conviction = "N/A"

    # Session filter — Asia is CONDITIONAL, not blanket bearish
    session_ok = True
    session_note = ""
    asia_note = ""

    if session == "Asia":
        weekday_short = state.get("weekday_short", "")
        asia_sub = state.get("asia_sub", "")
        asia_day_bps = state.get("asia_day_bps", 0)
        carry = state.get("late_nyc_carry", 0)

        # Asia-specific intelligence
        good_asia_day = weekday_short in ("Mon", "Wed", "Fri")
        bad_asia_day = weekday_short in ("Thu", "Sat", "Sun")
        late_asia_window = asia_sub == "Late Asia"  # 04-08 UTC is less negative

        if direction == "LONG":
            if bad_asia_day:
                session_ok = False
                session_note = f"{weekday_short} Asia is toxic ({asia_day_bps:+.0f} bps/hr). Wait for London open."
            elif good_asia_day and late_asia_window:
                session_ok = True
                asia_note = f"{weekday_short} Asia (Late sub-session) is tradeable ({asia_day_bps:+.0f} bps/hr)."
            elif good_asia_day:
                session_ok = True
                asia_note = f"{weekday_short} Asia is positive ({asia_day_bps:+.0f} bps/hr). Early Asia is noisier — smaller size."
                if conviction == "LOW":
                    session_ok = False
                    session_note = "Low conviction + Early Asia = skip. Wait for Late Asia (04:00 UTC) or London."
            else:
                # Tue — moderate negative
                if hourly_bias < -15:
                    session_ok = False
                    session_note = f"Current hour has strong negative bias ({hourly_bias:+.0f} bps). Wait."
                else:
                    asia_note = f"{weekday_short} Asia is mixed. Proceed with caution."

            # Carry effect: after mild Late NYC up, Asia dumps hardest
            if carry > 0.005 and carry <= 0.02:
                asia_note += f" Late NYC carry +{carry:.1%} — CAUTION: mild positive carry → Asia tends to give back (-1.5% avg)."
                if conviction != "HIGH":
                    conviction = "LOW"

        elif direction == "SHORT":
            # SHORT signals during Asia are actually the BEST (0.83% avg return, IC=0.075)
            asia_note = f"SHORT signals during Asia are high-quality (IC=0.075, avg return 0.83%). Composite signal is MORE predictive during Asia."
            if bad_asia_day:
                conviction = "HIGH" if conviction in ("HIGH", "MEDIUM") else "MEDIUM"
                asia_note += f" {weekday_short} Asia bleeds ({asia_day_bps:+.0f} bps/hr) — tailwind for shorts."

    elif direction == "LONG" and hourly_bias < -15:
        session_ok = False
        session_note = f"Current hour ({state['utc_hour']:02d}:00 UTC) has strong negative bias ({hourly_bias:+.0f} bps). Wait."

    # Funding rate context
    funding_note = ""
    if funding > 0.01:
        funding_note = f"Funding heavily positive ({funding:.4f}). Longs crowded — contrarian bearish pressure."
        if direction == "LONG":
            conviction = "LOW"  # downgrade
    elif funding < -0.01:
        funding_note = f"Funding heavily negative ({funding:.4f}). Shorts crowded — contrarian bullish pressure."
        if direction == "SHORT":
            conviction = "LOW"

    # BTC context
    btc_note = ""
    if "Dump" in btc_regime and direction == "LONG":
        btc_note = f"BTC in {btc_regime} mode. Fart has 1.6x beta — high risk for longs."
        conviction = "LOW"
    elif "Rally" in btc_regime and direction == "LONG":
        btc_note = f"BTC in {btc_regime} mode. Tailwind for longs (1.6x beta)."

    # Timing
    if direction != "FLAT":
        if session in ("London", "Late NYC"):
            timing = "NOW — favorable session"
        elif session == "Asia" and session_ok:
            timing = f"CONDITIONAL — {asia_note.split('.')[0]}."
        elif session == "Asia" and not session_ok:
            timing = "WAIT — " + (session_note or "Unfavorable Asia conditions. Queue for London open (08:00 UTC / 4am ET)")
        else:
            if hourly_bias > 5:
                timing = "NOW — current hour has positive bias"
            elif hourly_bias < -10:
                timing = f"WAIT — {state['utc_hour']:02d}:00 UTC has {hourly_bias:+.0f} bps bias. Next window: {_next_positive_hour(state['utc_hour']):02d}:00 UTC"
            else:
                timing = "ACCEPTABLE — neutral hour"
    else:
        timing = "No trade. Wait for signal."

    # Exit plan
    if direction != "FLAT":
        exit_plan = "Exit within 4-8 hours (signal decays after 8h, reverses by 24h). " \
                    "Hard stop if composite flips sign. Kill zone: 18:00 UTC (2pm ET)."
    else:
        exit_plan = "N/A"

    return {
        "direction": direction,
        "conviction": conviction,
        "session_ok": session_ok,
        "session_note": session_note,
        "asia_note": asia_note,
        "funding_note": funding_note,
        "btc_note": btc_note,
        "timing": timing,
        "exit_plan": exit_plan,
    }


def _next_positive_hour(current_hour):
    """Find the next hour with positive historical bias."""
    for offset in range(1, 24):
        h = (current_hour + offset) % 24
        if HOURLY_BIAS.get(h, 0) > 5:
            return h
    return (current_hour + 1) % 24


mkt = compute_market_state()
action = determine_action(mkt)


# =========================================================================
# MAIN LAYOUT — ACTION PANEL (always visible at top)
# =========================================================================

st.markdown("### 🎯 FARTCOIN TRADE DESK")

# --- Action Banner ---
direction = action["direction"]
if direction == "LONG":
    color = "#1b5e20" if action["conviction"] == "HIGH" else "#388e3c" if action["conviction"] == "MEDIUM" else "#66bb6a"
    st.markdown(f"""<div style="background:{color};color:white;padding:20px;border-radius:10px;margin-bottom:16px">
    <h2 style="margin:0;color:white">⬆️ LONG — {action['conviction']} CONVICTION</h2>
    <p style="margin:8px 0 0 0;font-size:16px;color:white"><b>Timing:</b> {action['timing']}</p>
    <p style="margin:4px 0 0 0;font-size:14px;color:rgba(255,255,255,0.9)"><b>Exit:</b> {action['exit_plan']}</p>
    </div>""", unsafe_allow_html=True)
elif direction == "SHORT":
    color = "#b71c1c" if action["conviction"] == "HIGH" else "#d32f2f" if action["conviction"] == "MEDIUM" else "#ef5350"
    st.markdown(f"""<div style="background:{color};color:white;padding:20px;border-radius:10px;margin-bottom:16px">
    <h2 style="margin:0;color:white">⬇️ SHORT — {action['conviction']} CONVICTION</h2>
    <p style="margin:8px 0 0 0;font-size:16px;color:white"><b>Timing:</b> {action['timing']}</p>
    <p style="margin:4px 0 0 0;font-size:14px;color:rgba(255,255,255,0.9)"><b>Exit:</b> {action['exit_plan']}</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div style="background:#424242;color:white;padding:20px;border-radius:10px;margin-bottom:16px">
    <h2 style="margin:0;color:white">⏸️ NO TRADE — WAIT FOR SIGNAL</h2>
    <p style="margin:8px 0 0 0;font-size:14px;color:rgba(255,255,255,0.8)">Composite: {mkt['composite']:.3f} (need > 0.2 for LONG or < -0.2 for SHORT)</p>
    </div>""", unsafe_allow_html=True)

# --- Context Alerts ---
if action.get("asia_note"):
    st.info(f"🌏 **Asia Intel:** {action['asia_note']}")
if action["funding_note"]:
    st.warning(f"💰 **Funding:** {action['funding_note']}")
if action["btc_note"]:
    st.info(f"₿ **BTC Context:** {action['btc_note']}")
if action["session_note"]:
    st.error(f"⏰ **Session Warning:** {action['session_note']}")

# --- Key Metrics Row ---
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("FART Price", f"${mkt['fart_price']:.4f}")
col2.metric("Composite", f"{mkt['composite']:+.3f}")
col3.metric("Funding Rate", f"{mkt.get('avg_funding', 0):.4f}")
col4.metric("BTC", f"${mkt.get('btc_price', 0):,.0f}", delta=f"{mkt.get('btc_ret_24h', 0):.1%} 24h")
col5.metric("Total OI", f"${mkt.get('total_oi', 0)/1e6:.0f}M")
col6.metric("Session", f"{mkt['session']} ({mkt.get('session_info', {}).get('et', '')})")
risk_score = mkt.get("risk_score", 0)
risk_label = "HIGH" if risk_score >= 4 else "MOD" if risk_score >= 2 else "LOW"
col7.metric("Manip. Risk", f"{risk_label} ({risk_score}/7)")

st.markdown("---")

# =========================================================================
# TABS
# =========================================================================

tab_signal, tab_timing, tab_exchange, tab_btc, tab_rules = st.tabs([
    "📊 Signal Dashboard",
    "⏰ Session Timing",
    "🏦 Exchange Intel",
    "₿ BTC Context",
    "📋 Trade Rules",
])

# =========================================================================
# TAB 1: Signal Dashboard
# =========================================================================

with tab_signal:
    signals = data.get("signals")
    ohlcv = data.get("ohlcv")

    if signals is not None and ohlcv is not None:
        price_col = "price" if "price" in ohlcv.columns else "close"

        # --- Price + Composite overlaid ---
        st.subheader("Price Action vs Composite Signal")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.3, 0.2],
                            vertical_spacing=0.04,
                            subplot_titles=["FARTCOIN Price", "Composite Signal", "Volume"])

        fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv[price_col], name="Price",
                                 line=dict(color="#1f77b4", width=1.5)), row=1, col=1)

        # Signal with colored zones
        comp = signals["composite"]
        fig.add_trace(go.Scatter(x=comp.index, y=comp, name="Composite",
                                 line=dict(color="#1f77b4", width=1)), row=2, col=1)
        fig.add_hline(y=0.4, line_color="green", line_dash="dash", row=2, col=1,
                      annotation_text="LONG ENTRY (0.4)")
        fig.add_hline(y=0.2, line_color="green", line_dash="dot", row=2, col=1,
                      annotation_text="Low conviction (0.2)")
        fig.add_hline(y=-0.2, line_color="red", line_dash="dot", row=2, col=1)
        fig.add_hline(y=-0.4, line_color="red", line_dash="dash", row=2, col=1,
                      annotation_text="SHORT ENTRY (-0.4)")
        fig.add_hrect(y0=0.4, y1=0.6, fillcolor="green", opacity=0.08, row=2, col=1)
        fig.add_hrect(y0=-0.6, y1=-0.4, fillcolor="red", opacity=0.08, row=2, col=1)

        fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], name="Volume",
                             marker_color="rgba(100,100,100,0.3)"), row=3, col=1)

        fig.update_layout(height=700, showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- Signal Component Gauges ---
        st.subheader("Signal Components (Current)")
        sig_cols = [c for c in signals.columns if c.startswith("sig_") and not np.isnan(signals[c].dropna().iloc[-1] if len(signals[c].dropna()) > 0 else np.nan)]
        n_sigs = len(sig_cols)
        if n_sigs > 0:
            cols = st.columns(min(n_sigs, 7))
            for i, col_name in enumerate(sig_cols):
                val = signals[col_name].dropna().iloc[-1] if len(signals[col_name].dropna()) > 0 else 0
                label = col_name.replace("sig_", "").replace("_", " ").title()
                with cols[i % len(cols)]:
                    # Color based on value
                    if val > 0.2: icon = "🟢"
                    elif val < -0.2: icon = "🔴"
                    else: icon = "⚪"
                    st.metric(f"{icon} {label}", f"{val:+.3f}")

        # --- Historical trade performance ---
        trades = data.get("trades")
        if trades is not None and not trades.empty:
            st.subheader("Historical Trades")
            st.dataframe(trades, use_container_width=True, hide_index=True)

    else:
        st.warning("Run signal_engine.py to generate signals.")


# =========================================================================
# TAB 2: Session Timing
# =========================================================================

with tab_timing:
    st.subheader("When to Trade — Session & Hour Analysis")

    ohlcv = data.get("ohlcv")
    if ohlcv is not None:
        price_col = "price" if "price" in ohlcv.columns else "close"
        df = ohlcv.copy()
        df["return"] = df[price_col].pct_change()
        df["abs_return"] = df["return"].abs()
        df["hour"] = df.index.hour
        df["session"] = df["hour"].apply(classify_session)

        # --- Session Cards ---
        st.markdown("#### Session Performance (Your NYC Time)")
        cols = st.columns(4)
        for i, (sess, info) in enumerate(SESSION_MAP.items()):
            with cols[i]:
                bps = info["avg_bps"]
                bias = info["bias"]
                if bias == "bullish":
                    st.markdown(f"""<div style="background:#e8f5e9;padding:15px;border-radius:8px;border-left:4px solid #388e3c">
                    <h4 style="margin:0">{sess}</h4>
                    <p style="margin:4px 0;color:#666">{info['et']} ET</p>
                    <p style="margin:0;font-size:24px;color:#388e3c"><b>{bps:+.1f} bps/hr</b></p>
                    <p style="margin:4px 0 0 0;color:#388e3c">✅ FAVORABLE for entry</p>
                    </div>""", unsafe_allow_html=True)
                elif bias == "conditional":
                    st.markdown(f"""<div style="background:#e3f2fd;padding:15px;border-radius:8px;border-left:4px solid #1976d2">
                    <h4 style="margin:0">{sess}</h4>
                    <p style="margin:4px 0;color:#666">{info['et']} ET</p>
                    <p style="margin:0;font-size:24px;color:#1976d2"><b>{bps:+.1f} bps/hr avg</b></p>
                    <p style="margin:4px 0 0 0;color:#1976d2">🔀 CONDITIONAL — depends on day & carry</p>
                    </div>""", unsafe_allow_html=True)
                elif bias == "neutral":
                    st.markdown(f"""<div style="background:#fff3e0;padding:15px;border-radius:8px;border-left:4px solid #ff9800">
                    <h4 style="margin:0">{sess}</h4>
                    <p style="margin:4px 0;color:#666">{info['et']} ET</p>
                    <p style="margin:0;font-size:24px;color:#ff9800"><b>{bps:+.1f} bps/hr</b></p>
                    <p style="margin:4px 0 0 0;color:#ff9800">⚠️ CAUTION — most volatile</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:#ffebee;padding:15px;border-radius:8px;border-left:4px solid #d32f2f">
                    <h4 style="margin:0">{sess}</h4>
                    <p style="margin:4px 0;color:#666">{info['et']} ET</p>
                    <p style="margin:0;font-size:24px;color:#d32f2f"><b>{bps:+.1f} bps/hr</b></p>
                    <p style="margin:4px 0 0 0;color:#d32f2f">⛔ AVOID for longs</p>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # ASIA SESSION DEEP DIVE
        # ---------------------------------------------------------------
        st.markdown("---")
        st.subheader("🌏 Asia Session Deep Dive")
        st.caption("Asia isn't uniformly bearish. The desk needs to know WHEN Asia is tradeable.")

        # Sub-session breakdown
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown("#### Asia Sub-Sessions")
            early = df[df["hour"].isin([0, 1, 2, 3])]["return"].mean() * 10000
            late = df[df["hour"].isin([4, 5, 6, 7])]["return"].mean() * 10000
            st.markdown(f"""<div style="background:#ffebee;padding:12px;border-radius:8px;margin-bottom:8px">
            <b>Early Asia</b> (00-04 UTC / 8pm-12am ET): <span style="color:#d32f2f;font-size:18px"><b>{early:+.1f} bps/hr</b></span><br>
            <small>This is where Late NYC momentum dies. 00:00 and 01:00 UTC are the worst hours.</small>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#fff8e1;padding:12px;border-radius:8px">
            <b>Late Asia</b> (04-08 UTC / 12am-4am ET): <span style="color:#f57f17;font-size:18px"><b>{late:+.1f} bps/hr</b></span><br>
            <small>04:00 and 05:00 UTC are actually positive. 07:00 UTC dumps ahead of London.</small>
            </div>""", unsafe_allow_html=True)

        with col_a2:
            st.markdown("#### Asia by Day of Week")
            day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            day_vals = [ASIA_DAY_BPS[d] for d in day_order]
            colors = ["#388e3c" if v > 0 else "#d32f2f" for v in day_vals]
            fig_day = go.Figure(go.Bar(x=day_order, y=day_vals, marker_color=colors,
                                       text=[f"{v:+.0f}" for v in day_vals], textposition="outside"))
            fig_day.add_hline(y=0, line_color="black", line_width=0.5)
            fig_day.update_layout(height=250, yaxis_title="Avg bps/hr",
                                  title="Mon/Wed/Fri = tradeable | Thu/Sat/Sun = toxic")
            st.plotly_chart(fig_day, use_container_width=True)

        # Carry effect
        st.markdown("#### Late NYC Carry Effect")
        st.markdown("""
        | Late NYC Outcome | Next Asia Avg Return | Interpretation |
        |-----------------|---------------------|----------------|
        | **Strong UP (>2%)** | -0.43% | Partial give-back, but not catastrophic |
        | **Mild UP (0-2%)** | **-1.52%** | **WORST outcome** — Asia dumps hardest after mild Late NYC gains |
        | **Flat** | -0.43% | Normal Asia bleed |
        | **Mild DOWN** | -0.47% | Normal Asia bleed |
        | **Strong DOWN (<-2%)** | -0.57% | Continued selling |

        **Key insight:** The carry correlation is **-0.045** (weakly negative). A strong Late NYC move does NOT predict Asia continuation.
        The MOST dangerous setup is a mild +0-2% Late NYC — Asia gives it ALL back and then some.
        """)

        # Signal quality during Asia
        st.markdown("#### Composite Signal Works BETTER During Asia")
        st.markdown("""
        | Metric | Asia Session | All Other Sessions |
        |--------|-------------|-------------------|
        | **IC (composite → 4h return)** | **0.075** | 0.053 |
        | **LONG signals (>0.2) hit rate** | 57% | — |
        | **SHORT signals (<-0.2) avg return** | **+0.83%** | — |

        **Bottom line for the desk:** When you DO get a signal during Asia, trust it more than usual.
        SHORT signals during Asia are the highest-quality trades in the entire dataset.
        """)

        st.markdown("---")

        col_l, col_r = st.columns(2)

        # --- Hourly Returns Chart ---
        with col_l:
            st.markdown("#### Return by Hour (UTC)")
            hourly_ret = df.groupby("hour")["return"].mean() * 10000
            colors = ["#388e3c" if v > 5 else "#d32f2f" if v < -10 else "#9e9e9e" for v in hourly_ret]

            fig = go.Figure(go.Bar(x=hourly_ret.index, y=hourly_ret.values, marker_color=colors))
            # Highlight current hour
            now_h = get_current_utc().hour
            fig.add_vline(x=now_h, line_color="blue", line_width=3, line_dash="solid",
                          annotation_text=f"NOW ({now_h:02d}:00 UTC)")
            fig.add_vrect(x0=-0.5, x1=7.5, fillcolor="blue", opacity=0.03)
            fig.add_vrect(x0=7.5, x1=12.5, fillcolor="green", opacity=0.03)
            fig.add_vrect(x0=12.5, x1=20.5, fillcolor="orange", opacity=0.03)
            fig.add_vrect(x0=20.5, x1=23.5, fillcolor="purple", opacity=0.03)
            fig.update_layout(height=350, xaxis_title="Hour (UTC)", yaxis_title="Avg Return (bps)",
                              xaxis=dict(dtick=1))
            st.plotly_chart(fig, use_container_width=True)

        # --- Volatility Chart ---
        with col_r:
            st.markdown("#### Volatility by Hour (Where the Big Moves Are)")
            hourly_vol = df.groupby("hour")["abs_return"].mean() * 100
            fig2 = go.Figure(go.Bar(x=hourly_vol.index, y=hourly_vol.values, marker_color="#ff9800"))
            fig2.add_vline(x=now_h, line_color="blue", line_width=3,
                           annotation_text="NOW")
            fig2.update_layout(height=350, xaxis_title="Hour (UTC)", yaxis_title="Avg |Move| (%)",
                               xaxis=dict(dtick=1))
            st.plotly_chart(fig2, use_container_width=True)

        # --- Key Timing Rules ---
        st.markdown("#### Timing Playbook")
        st.markdown("""
        | Rule | Detail | Your Time (ET) |
        |------|--------|----------------|
        | **Best entry window** | 10:00 UTC (+23.4 bps avg) | **6:00 AM ET** |
        | **Second window** | 19:00-22:00 UTC (+15-19 bps avg) | **3-6 PM ET** |
        | **Kill zone — AVOID** | 18:00 UTC (-26.5 bps avg) | **2:00 PM ET** |
        | **Asia bleed — AVOID longs** | 00:00-07:00 UTC | **8 PM - 3 AM ET** |
        | **Hold window** | 4-8 hours max | Signal decays after 8h |
        | **Hard exit** | Composite flips sign | Immediate close |
        | **Best days** | Mon, Tue, Fri | |
        | **Worst days** | Thu, Sat, Sun | Dump days |
        """)

        # --- Day of Week ---
        daily = data.get("ohlcv_daily")
        if daily is not None:
            st.markdown("#### Returns by Day of Week")
            daily_c = daily.copy()
            daily_c["return"] = daily_c["close"].pct_change()
            daily_c["weekday"] = daily_c.index.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            wd_mean = daily_c.groupby("weekday")["return"].mean().reindex(day_order) * 100
            colors = ["#388e3c" if v > 0 else "#d32f2f" for v in wd_mean]
            fig3 = go.Figure(go.Bar(x=[d[:3] for d in day_order], y=wd_mean.values, marker_color=colors))
            fig3.add_hline(y=0, line_color="black", line_width=0.5)
            # Highlight today
            today = get_current_utc().strftime("%A")
            fig3.update_layout(height=300, yaxis_title="Avg Return (%)",
                               title=f"Today is {today}")
            st.plotly_chart(fig3, use_container_width=True)


# =========================================================================
# TAB 3: Exchange Intel
# =========================================================================

with tab_exchange:
    deriv = data.get("derivatives")
    if deriv is None:
        st.warning("No derivatives data.")
    else:
        active = deriv[deriv["open_interest_usd"] > 10000].copy()

        st.subheader(f"Cross-Exchange Positioning — {len(active)} Active Exchanges")

        col_l, col_r = st.columns(2)

        # --- OI Distribution ---
        with col_l:
            st.markdown("#### Where Is the Money? (OI)")
            oi_top = active.nlargest(12, "open_interest_usd")
            fig = go.Figure(go.Bar(
                y=oi_top["exchange"].str[:22], x=oi_top["open_interest_usd"] / 1e6,
                orientation="h", marker_color="#1f77b4",
                text=[f"${v/1e6:.1f}M" for v in oi_top["open_interest_usd"]],
                textposition="outside",
            ))
            fig.update_layout(height=400, xaxis_title="OI ($M)", margin=dict(l=180))
            st.plotly_chart(fig, use_container_width=True)

            total_oi = active["open_interest_usd"].sum()
            top_2 = active.nlargest(2, "open_interest_usd")
            top_2_share = top_2["open_interest_usd"].sum() / total_oi
            st.metric("Top 2 Exchanges OI Share", f"{top_2_share:.0%}",
                      delta=f"{top_2.iloc[0]['exchange'][:15]} + {top_2.iloc[1]['exchange'][:15]}")

        # --- Funding Rates ---
        with col_r:
            st.markdown("#### Funding Rate Divergence")
            fr_sorted = active.sort_values("funding_rate")
            colors = ["#d32f2f" if x < 0 else "#388e3c" for x in fr_sorted["funding_rate"]]
            fig2 = go.Figure(go.Bar(
                y=fr_sorted["exchange"].str[:22], x=fr_sorted["funding_rate"],
                orientation="h", marker_color=colors,
            ))
            fig2.add_vline(x=0, line_color="black", line_width=1)
            fig2.update_layout(height=max(400, len(fr_sorted) * 18), xaxis_title="Funding Rate",
                               margin=dict(l=180))
            st.plotly_chart(fig2, use_container_width=True)

        # --- Churning Detection ---
        st.markdown("#### Churning Detection (Volume / OI Ratio)")
        st.caption("Ratio > 10x is suspicious. Could indicate wash trading or rapid position cycling.")
        active["vol_oi"] = active["volume_24h_usd"] / active["open_interest_usd"].replace(0, np.nan)
        churning = active.nlargest(10, "vol_oi")
        fig3 = go.Figure(go.Bar(
            y=churning["exchange"].str[:22], x=churning["vol_oi"],
            orientation="h",
            marker_color=["#d32f2f" if v > 10 else "#ff9800" if v > 5 else "#388e3c" for v in churning["vol_oi"]],
            text=[f"{v:.1f}x" for v in churning["vol_oi"]], textposition="outside",
        ))
        fig3.update_layout(height=350, xaxis_title="Volume / OI", margin=dict(l=180))
        st.plotly_chart(fig3, use_container_width=True)

        # --- Full table ---
        with st.expander("Full Exchange Data"):
            display_df = active[["exchange", "funding_rate", "open_interest_usd", "volume_24h_usd",
                                 "basis_pct", "spread", "price"]].sort_values("open_interest_usd", ascending=False)
            st.dataframe(display_df.style.format({
                "funding_rate": "{:.6f}", "open_interest_usd": "${:,.0f}",
                "volume_24h_usd": "${:,.0f}", "basis_pct": "{:.4f}%",
                "spread": "{:.4f}", "price": "${:.4f}",
            }), use_container_width=True, height=500)


# =========================================================================
# TAB 4: BTC Context
# =========================================================================

with tab_btc:
    btc = data.get("btc")
    ohlcv = data.get("ohlcv")

    if btc is not None and ohlcv is not None:
        btc_col = "price" if "price" in btc.columns else "close"
        fart_col = "price" if "price" in ohlcv.columns else "close"

        btc_h = btc[[btc_col, "volume"]].resample("1h").last().dropna()
        btc_h.columns = ["btc_price", "btc_volume"]
        btc_h["btc_return"] = btc_h["btc_price"].pct_change()

        fart_h = ohlcv[[fart_col]].resample("1h").last().dropna()
        fart_h.columns = ["fart_price"]
        fart_h["fart_return"] = fart_h["fart_price"].pct_change()

        merged = btc_h.join(fart_h, how="inner").dropna(subset=["btc_return", "fart_return"])
        corr = merged["btc_return"].corr(merged["fart_return"])
        valid = merged[["btc_return", "fart_return"]].dropna()
        beta = np.polyfit(valid["btc_return"], valid["fart_return"], 1)[0] if len(valid) > 50 else 0
        merged["rolling_corr"] = merged["btc_return"].rolling(24).corr(merged["fart_return"])

        # Top metrics
        st.subheader("BTC-FART Relationship")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlation", f"{corr:.3f}")
        col2.metric("Beta", f"{beta:.2f}x")
        col3.metric("BTC Regime", mkt.get("btc_regime", "?"))
        col4.metric("BTC 24h", f"{mkt.get('btc_ret_24h', 0):.1%}")

        st.markdown(f"""
        **What this means for trading:**
        - FART moves **{beta:.1f}x** BTC on average. A 1% BTC move → ~{beta:.1f}% FART move.
        - When BTC and FART **decorrelate**, FART moves are **2.5x bigger** than normal → **manipulation signal**.
        - BTC dumps: FART falls {beta:.1f}x harder. BTC rallies: FART pumps {beta:.1f}x harder.
        - Late NYC session (5-8pm ET) has the highest beta (1.94x) — most leveraged to BTC.
        """)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Price Overlay (Normalized)")
            btc_norm = merged["btc_price"] / merged["btc_price"].iloc[0] * 100
            fart_norm = merged["fart_price"] / merged["fart_price"].iloc[0] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged.index, y=btc_norm, name="BTC",
                                     line=dict(color="#f7931a", width=2)))
            fig.add_trace(go.Scatter(x=merged.index, y=fart_norm, name="FART",
                                     line=dict(color="#1f77b4", width=2)))
            fig.update_layout(height=350, yaxis_title="Indexed (100=start)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("#### Rolling Correlation (Decorrelation = MM Activity)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=merged.index, y=merged["rolling_corr"],
                                      line=dict(color="#1f77b4", width=1)))
            fig2.add_hline(y=0, line_color="red", line_dash="dash")
            fig2.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.07)
            fig2.update_layout(height=350, yaxis_title="24h Correlation", yaxis_range=[-1, 1])
            st.plotly_chart(fig2, use_container_width=True)

        # --- BTC Regime Table ---
        st.markdown("#### How FART Behaves in Each BTC Environment")
        merged["btc_ret_24h"] = merged["btc_price"].pct_change(24)

        def regime(r):
            if pd.isna(r): return None
            if r > 0.03: return "BTC Strong Rally (>3%)"
            if r > 0.01: return "BTC Mild Rally (1-3%)"
            if r > -0.01: return "BTC Flat (-1% to 1%)"
            if r > -0.03: return "BTC Mild Dump (-3 to -1%)"
            return "BTC Strong Dump (<-3%)"

        merged["regime"] = merged["btc_ret_24h"].apply(regime)
        regime_order = ["BTC Strong Rally (>3%)", "BTC Mild Rally (1-3%)", "BTC Flat (-1% to 1%)",
                        "BTC Mild Dump (-3 to -1%)", "BTC Strong Dump (<-3%)"]
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
    else:
        st.warning("Missing BTC or FART data.")


# =========================================================================
# TAB 5: Trade Rules
# =========================================================================

with tab_rules:
    st.subheader("Trade Execution Rules")

    st.markdown("""
    ### Entry Rules

    | Condition | Threshold | Notes |
    |-----------|-----------|-------|
    | **Composite signal** | > +0.4 for LONG, < -0.4 for SHORT | Primary trigger. 88% hit rate at 0.4 (4h window) |
    | **Medium conviction** | +0.3 to +0.4 / -0.3 to -0.4 | 65% hit rate — use with confirming factors |
    | **Low conviction** | +0.2 to +0.3 / -0.2 to -0.3 | Only trade with session + BTC alignment |

    ### Timing Filters (MUST pass before entry)

    | Filter | Rule | Reason |
    |--------|------|--------|
    | **Session** | Prefer London (4-9am ET) or Late NYC (5-8pm ET) | Best historical returns |
    | **Asia — conditional** | Longs OK on Mon/Wed/Fri in Late Asia (12am-4am ET). Avoid Thu/Sat/Sun Asia entirely. SHORT signals during Asia are highest quality. | See Asia Deep Dive |
    | **Kill zone** | No new entries at 2pm ET (18:00 UTC) | Worst single hour (-26.5 bps) |
    | **Day of week** | Prefer Mon/Tue/Fri | Thu/Sat/Sun are dump days |

    ### Conviction Modifiers

    | Factor | Upgrades | Downgrades |
    |--------|----------|------------|
    | **Funding rate** | Aligns with direction (neg funding + long) | Opposes direction (pos funding + long) |
    | **BTC regime** | Rally + long, or Dump + short | Dump + long, or Rally + short |
    | **Session** | London or Late NYC | Asia |
    | **Volume spike** | Volume > 2x avg (positioning) | — |

    ### Exit Rules

    | Condition | Action |
    |-----------|--------|
    | **4-8 hours elapsed** | Close position. Signal decays after 8h. |
    | **Composite flips sign** | Immediate close. Signal reversed. |
    | **Hit 18:00 UTC (2pm ET)** | Close or tighten stop. Kill zone. |
    | **24h hold** | MUST close. Signal goes negative after 24h (mean reversion). |

    ### Risk Management

    | Rule | Detail |
    |------|--------|
    | **Position sizing** | Scale with conviction: HIGH = full size, MED = 50%, LOW = 25% |
    | **Stop loss** | -3% from entry (big moves average 5-7%, stops need room) |
    | **Take profit** | +1% at 4h (avg return at high conviction), trail from there |
    | **Correlation break** | If 24h BTC correlation goes negative, reduce position 50% — MM activity likely |
    | **Manipulation cycle** | After 2-3 quiet days + sudden pump > 10%, expect reversal next day. Fade it. |

    ### The MM Playbook (What We're Trading Against)

    1. **Accumulate** — MMs build positions during quiet periods (low volume, 2-3 days)
    2. **Engineer** — Push price on thin order books (usually during Asia or 2pm ET)
    3. **Cascade** — Trigger liquidations (leveraged positions get wiped)
    4. **Harvest** — Exit the other side at inflated/deflated prices
    5. **Reset** — Price mean-reverts within 24h. MMs re-enter.

    **Our edge:** We detect Stage 1-2 via the composite signal and enter BEFORE the cascade.
    Signal has 50 bps quintile spread and 88% hit rate at high conviction over 4h.
    """)


# =========================================================================
# Footer
# =========================================================================

st.markdown("---")
st.caption(f"Last data refresh: check data/ timestamps | UTC now: {get_current_utc().strftime('%Y-%m-%d %H:%M')} | Not financial advice.")
