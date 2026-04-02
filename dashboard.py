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

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FART Trade Desk",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
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
# Data loading  (includes all external collector files)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data():
    data = {}

    # Core files (index = timestamp)
    index_files = {
        "ohlcv":              "FARTCOIN_ohlcv_hourly.csv",
        "ohlcv_daily":        "FARTCOIN_ohlcv.csv",
        "btc":                "bitcoin_cg_chart.csv",
        "funding":            "FARTCOINUSDT_funding.csv",
        "lsr":                "FARTCOINUSDT_lsr.csv",
        "oi":                 "FARTCOINUSDT_oi.csv",
        "taker":              "FARTCOINUSDT_taker.csv",
        "signals":            "signals.csv",
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
        "derivatives": "FARTCOIN_derivatives_snapshot.csv",
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


data = load_all_data()

# Computed state
mkt    = compute_market_state(data)
action = determine_action(mkt)
proj   = compute_projections(data, mkt)

# Active alerts (both signal-level and projection-level)
_sig_alerts  = evaluate_alerts(mkt, action)
_proj_alerts = evaluate_projection_alerts(proj, mkt)
all_alerts   = _sig_alerts + _proj_alerts

# Data freshness
_signals_file = DATA_DIR / "signals.csv"
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
# HEADER — single line of key metrics
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

# Direction banner
_dir_colors = {
    "LONG":  {"HIGH": "#1b5e20", "MEDIUM": "#2e7d32", "LOW": "#388e3c"},
    "SHORT": {"HIGH": "#b71c1c", "MEDIUM": "#c62828", "LOW": "#d32f2f"},
    "FLAT":  {"HIGH": "#37474f", "MEDIUM": "#37474f", "LOW": "#37474f"},
}
_banner_color = _dir_colors.get(direction, _dir_colors["FLAT"]).get(conviction, "#37474f")
_dir_icon = {"LONG": "⬆️", "SHORT": "⬇️", "FLAT": "⏸️"}.get(direction, "⏸️")

freshness_html = ""
if _data_age_min is not None:
    if _data_age_min < 35:
        freshness_html = f'<span style="font-size:0.75rem;opacity:0.7">Data {_data_age_min:.0f}m ago</span>'
    else:
        freshness_html = f'<span style="font-size:0.75rem;color:#ff8f00">⚠ Data {_data_age_min:.0f}m old</span>'

st.markdown(
    f"""<div style="background:{_banner_color};color:white;padding:14px 20px;
    border-radius:10px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
    <div>
      <span style="font-size:1.5rem;font-weight:800">{_dir_icon} {direction} — {conviction} CONVICTION</span>
      <span style="margin-left:20px;font-size:0.95rem;opacity:0.9">{action['timing']}</span>
    </div>
    <div style="text-align:right;font-size:0.8rem;opacity:0.85">
      {freshness_html}<br>
      {utc_now.strftime('%H:%M UTC')}
    </div>
    </div>""",
    unsafe_allow_html=True,
)

# Metric strip
mc = st.columns(8)
mc[0].metric("FART", f"${fart_price:.4f}")
mc[1].metric("Composite", f"{composite:+.3f}")
mc[2].metric("Funding", f"{avg_funding:.4f}")
mc[3].metric("BTC", f"${btc_price:,.0f}")
mc[4].metric("BTC Regime", mkt.get("btc_regime", "?"))
mc[5].metric("OI", f"${total_oi/1e6:.0f}M")
mc[6].metric("Session", f"{session} ({mkt.get('session_info',{}).get('et','')})")
mc[7].metric("Manip Risk", f"{risk_label} {risk_score}/7")

# Inline notes (funding, BTC, session warnings) — compact
_notes = []
if action.get("funding_note"):  _notes.append(("💰", action["funding_note"], "warning"))
if action.get("btc_note"):      _notes.append(("₿",  action["btc_note"],     "info"))
if action.get("session_note"):  _notes.append(("⏰", action["session_note"],  "error"))
if action.get("asia_note"):     _notes.append(("🌏", action["asia_note"],    "info"))
for icon, note, kind in _notes:
    getattr(st, kind)(f"{icon} {note}")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE ALERTS BAR  (always visible before tabs)
# ─────────────────────────────────────────────────────────────────────────────

if all_alerts:
    st.markdown(f"#### 🚨 Active Alerts ({len(all_alerts)})")
    for al in all_alerts:
        sev   = al.get("severity", "low")
        color = _severity_color(sev)
        label = {"high": "HIGH", "medium": "MED", "low": "LOW"}.get(sev, sev.upper())
        st.markdown(
            f'<div class="alert-card" style="background:{color}22;border-left:4px solid {color}">'
            f'<b style="color:{color}">[{label}]</b> {al["title"]}</div>',
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No active alerts")

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

        # ── Signal component gauges (top of tab) ─────────────────────────────
        st.markdown("#### Signal Components — Current Reading")
        sig_cols = [c for c in signals.columns if c.startswith("sig_")]
        sig_vals = {c: float(signals[c].dropna().iloc[-1]) for c in sig_cols
                    if len(signals[c].dropna()) > 0}

        if sig_vals:
            cols = st.columns(len(sig_vals))
            for i, (col_name, val) in enumerate(sig_vals.items()):
                label = col_name.replace("sig_", "").replace("_", " ").title()
                icon  = "🟢" if val > 0.2 else "🔴" if val < -0.2 else "⚪"
                cols[i].metric(f"{icon} {label}", f"{val:+.3f}")

        st.markdown("---")

        # ── Price + Composite chart ───────────────────────────────────────────
        st.markdown("#### Price vs Composite Signal")
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.03,
            subplot_titles=["FARTCOIN Price", "Composite Signal", "Volume"],
        )

        fig.add_trace(go.Scatter(
            x=ohlcv.index, y=ohlcv[price_col], name="Price",
            line=dict(color="#64b5f6", width=1.5)), row=1, col=1)

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

    # ── Row 1: Probability + Manipulation cycle ───────────────────────────────
    r1a, r1b = st.columns([3, 2])

    prob_data = proj.get("probability", {})
    prob_val  = prob_data.get("prob_positive_4h", 0.5)
    exp_move  = prob_data.get("expected_move_pct", 0)
    prob_conv = prob_data.get("conviction", "LOW")

    with r1a:
        st.markdown("#### Probability Model")
        pcolor = "#1b5e20" if prob_val > 0.6 else "#b71c1c" if prob_val < 0.4 else "#1565c0"
        st.markdown(
            f'<div class="proj-card" style="border-color:{pcolor};background:{pcolor}22">'
            f'<span style="font-size:2rem;font-weight:800;color:{pcolor}">{prob_val:.0%}</span>'
            f'&nbsp;&nbsp;probability of positive 4h return<br>'
            f'<b>Expected move:</b> {exp_move:+.2f}% &nbsp;|&nbsp; '
            f'<b>Conviction:</b> {prob_conv} &nbsp;|&nbsp; '
            f'<b>n=</b>{prob_data.get("model_n_train", 0):,}<br>'
            f'<span style="font-size:0.8rem;color:#aaa">{prob_data.get("description","")}</span>'
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
        ccolor = _ccolors.get(phase, "#37474f")
        est    = cycle.get("est_hours_to_move")
        est_txt = f"Est. move: ~{est}h" if est is not None else ""
        st.markdown(
            f'<div class="proj-card" style="border-color:{ccolor};background:{ccolor}22">'
            f'<span style="font-size:1.2rem;font-weight:800;color:{ccolor}">{phase}</span> '
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
        fig_fan.update_layout(
            title=f"4h: ${h4['low_68']:.4f} – ${h4['high_68']:.4f} (68%) | "
                  f"${h4['low_95']:.4f} – ${h4['high_95']:.4f} (95%)",
            xaxis_title="Hours (0 = now)", yaxis_title="Price ($)",
            template="plotly_dark", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_fan, use_container_width=True)

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
            bar_color = "#e57373" if pct < 0.3 else "#66bb6a" if pct > 0.7 else "#90caf9"
            fig_lsr = go.Figure()
            fig_lsr.add_trace(go.Scatter(
                x=list(range(1, len(path)+1)), y=path, mode="lines",
                name="Projected", line=dict(color=bar_color, dash="dash", width=2)))
            fig_lsr.add_hline(y=lsr_data.get("median", 1.0), line_dash="dot",
                              annotation_text="Median", line_color="#546e7a")
            fig_lsr.update_layout(
                title=f"Long/Short Ratio  (now {lsr_data['current']:.3f}, {pct:.0%}ile)",
                xaxis_title="Hours", yaxis_title="LSR",
                template="plotly_dark", height=260, margin=dict(t=36, b=10),
            )
            st.plotly_chart(fig_lsr, use_container_width=True)
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
        _proj_card(
            f"BTC +{btc_2h:+.1f}% → FART {proj_fart:+.1f}% projected",
            btc_ll.get("description", "No data"),
            color=ll_color, icon="₿",
        )
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("BTC 2h", f"{btc_2h:+.1f}%")
        bc2.metric("FART proj", f"{proj_fart:+.1f}%")
        bc3.metric("Confidence", f"{btc_conf:.0%}")

    sess_cond = proj.get("session_conditional", {})
    edge      = sess_cond.get("combined_edge_pct", 0)
    quality   = sess_cond.get("quality", "")
    with ctx_c2:
        st.markdown("**Session-Conditional Edge**")
        se_color = "#1b5e20" if edge > 0.5 else "#b71c1c" if edge < -0.5 else "#e65100"
        _proj_card(
            f"Edge: {edge:+.2f}%  [{quality}]",
            sess_cond.get("description", "No data"),
            color=se_color, icon="⏰",
        )
        sc1, sc2 = st.columns(2)
        sc1.metric("Combined Edge", f"{edge:+.2f}%")
        sc2.metric("Samples", f"n={sess_cond.get('n_samples',0)}")

    st.markdown("---")

    # ── Row 5: External data projections ─────────────────────────────────────
    st.markdown("#### External Data Signals")
    ext_c1, ext_c2, ext_c3 = st.columns(3)

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

        # Timing playbook table
        st.markdown("---")
        st.markdown("#### Timing Playbook")
        st.markdown("""
| Rule | Detail | ET Time |
|------|--------|---------|
| **Best entry** | 10:00 UTC (+23.4 bps avg) | 6:00 AM |
| **2nd window** | 19:00–22:00 UTC (+15–19 bps) | 3–6 PM |
| **Kill zone — AVOID** | 18:00 UTC (−26.5 bps avg) | 2:00 PM |
| **Asia bleed — AVOID longs** | 00:00–07:00 UTC | 8 PM – 3 AM |
| **Hold window** | 4–8 hours max | Signal decays after 8h |
| **Hard exit** | Composite flips sign | Immediate |
| **Best days** | Mon · Tue · Fri | |
| **Worst days** | Thu · Sat · Sun | Dump days |
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
