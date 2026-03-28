"""
Fartcoin Alpha Dashboard — Live Manipulation Detection

Streamlit dashboard for monitoring market maker behavior,
cross-exchange anomalies, BTC correlation, and signal state.

Run: streamlit run dashboard.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fartcoin Alpha",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"


def classify_session(hour):
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 21:
        return "NYC"
    else:
        return "Late NYC"


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_all_data():
    data = {}

    f = DATA_DIR / "FARTCOIN_ohlcv_hourly.csv"
    if f.exists():
        data["ohlcv"] = pd.read_csv(f, index_col=0, parse_dates=True)

    f = DATA_DIR / "FARTCOIN_ohlcv.csv"
    if f.exists():
        data["ohlcv_daily"] = pd.read_csv(f, index_col=0, parse_dates=True)

    f = DATA_DIR / "FARTCOIN_derivatives_snapshot.csv"
    if f.exists():
        data["derivatives"] = pd.read_csv(f)

    f = DATA_DIR / "signals.csv"
    if f.exists():
        data["signals"] = pd.read_csv(f, index_col=0, parse_dates=True)

    f = DATA_DIR / "trades.csv"
    if f.exists():
        data["trades"] = pd.read_csv(f)

    f = DATA_DIR / "bitcoin_cg_chart.csv"
    if f.exists():
        data["btc"] = pd.read_csv(f, index_col=0, parse_dates=True)

    f = DATA_DIR / "FARTCOINUSDT_funding.csv"
    if f.exists():
        data["funding"] = pd.read_csv(f, index_col=0, parse_dates=True)

    return data


data = load_all_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("💨 Fartcoin Alpha")
st.sidebar.caption("Manipulation Detection Framework")

page = st.sidebar.radio("Navigate", [
    "📊 Market Overview",
    "🏦 Cross-Exchange",
    "📈 BTC Correlation",
    "🎯 Signals & Trades",
    "⏰ Session Analysis",
])

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Show data freshness
if "derivatives" in data:
    snap_time = data["derivatives"].get("snapshot_time", pd.Series(["unknown"])).iloc[0]
    st.sidebar.info(f"Snapshot: {snap_time[:19] if isinstance(snap_time, str) else 'N/A'}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources**")
st.sidebar.markdown("- CoinMarketCap (OHLCV)")
st.sidebar.markdown("- CoinGecko (Derivatives)")
st.sidebar.markdown("- BTC correlation (CG)")


# =========================================================================
# PAGE 1: Market Overview
# =========================================================================

if page == "📊 Market Overview":
    st.title("📊 Market Overview")

    # --- Top Metrics ---
    deriv = data.get("derivatives")
    ohlcv = data.get("ohlcv")

    if deriv is not None:
        active = deriv[deriv["open_interest_usd"] > 10000]
        avg_fr = active["funding_rate"].mean()
        total_oi = active["open_interest_usd"].sum()
        total_vol = active["volume_24h_usd"].sum()
        oi_vol = total_oi / total_vol if total_vol > 0 else 0
        hhi = ((active["open_interest_usd"] / total_oi) ** 2).sum()
        fr_range = active["funding_rate"].max() - active["funding_rate"].min()

        # Risk score
        risk = 0
        if abs(avg_fr) > 0.01: risk += 2
        if oi_vol < 0.5: risk += 1
        if hhi > 0.15: risk += 2
        if fr_range > 0.05: risk += 1
        top_share = active["open_interest_usd"].max() / total_oi
        if top_share > 0.3: risk += 1
        risk_label = "🔴 HIGH" if risk >= 4 else "🟡 MODERATE" if risk >= 2 else "🟢 LOW"

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Manipulation Risk", risk_label)
        col2.metric("Avg Funding Rate", f"{avg_fr:.4f}", delta="Longs pay" if avg_fr > 0 else "Shorts pay")
        col3.metric("Total OI", f"${total_oi/1e6:.1f}M")
        col4.metric("24h Volume", f"${total_vol/1e6:.1f}M")
        col5.metric("OI/Vol Ratio", f"{oi_vol:.2f}x")

        st.markdown("---")

    # --- Price Chart ---
    if ohlcv is not None:
        st.subheader("Price & Volume (90d Hourly)")
        price_col = "price" if "price" in ohlcv.columns else "close"

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv[price_col], name="Price",
                                 line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
        fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], name="Volume",
                             marker_color="rgba(100,100,100,0.3)"), row=2, col=1)
        fig.update_layout(height=500, showlegend=False,
                          xaxis2_title="Date", yaxis_title="Price ($)", yaxis2_title="Volume ($)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Current State Summary ---
    if deriv is not None:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Risk Factors")
            if abs(avg_fr) > 0.01:
                direction = "BEARISH (contrarian)" if avg_fr > 0 else "BULLISH (contrarian)"
                st.error(f"🚨 Extreme funding rate ({avg_fr:.4f}) → {direction}")
            else:
                st.success(f"✅ Funding rate normal ({avg_fr:.4f})")

            if oi_vol < 0.5:
                st.warning(f"⚠️ High churning — OI/Vol ratio {oi_vol:.2f}x")
            if fr_range > 0.05:
                st.warning(f"⚠️ Funding divergence across exchanges: {fr_range:.4f}")
            if hhi > 0.15:
                st.error(f"🚨 OI concentrated (HHI={hhi:.4f})")
            else:
                st.success(f"✅ OI fragmented across exchanges (HHI={hhi:.4f})")

        with col_b:
            st.subheader("Signal State")
            signals = data.get("signals")
            if signals is not None:
                latest = signals.dropna(subset=["composite"]).iloc[-1]
                comp = latest["composite"]
                if comp > 0.4:
                    st.success(f"🟢 LONG signal active (composite: {comp:.3f})")
                elif comp < -0.4:
                    st.error(f"🔴 SHORT signal active (composite: {comp:.3f})")
                else:
                    st.info(f"⚪ Neutral (composite: {comp:.3f})")

                for col in [c for c in signals.columns if c.startswith("sig_")]:
                    val = latest.get(col, 0)
                    if not np.isnan(val):
                        bar_color = "green" if val > 0.1 else "red" if val < -0.1 else "gray"
                        st.markdown(f"**{col.replace('sig_', '')}**: `{val:+.3f}`")

    # --- Daily return distribution ---
    daily = data.get("ohlcv_daily")
    if daily is not None:
        st.subheader("Daily Return Distribution")
        returns = daily["close"].pct_change().dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=30, marker_color="#1f77b4",
                                    marker_line_color="white", marker_line_width=1))
        fig.add_vline(x=0, line_color="red", line_width=2)
        fig.update_layout(height=300, xaxis_title="Daily Return (%)",
                          yaxis_title="Count",
                          title=f"Mean={returns.mean():.2f}%, Std={returns.std():.1f}%")
        st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# PAGE 2: Cross-Exchange Analysis
# =========================================================================

elif page == "🏦 Cross-Exchange":
    st.title("🏦 Cross-Exchange Derivatives Analysis")

    deriv = data.get("derivatives")
    if deriv is None:
        st.warning("No derivatives data. Run data_collector.py first.")
    else:
        active = deriv[deriv["open_interest_usd"] > 10000].copy()
        st.caption(f"{len(active)} exchanges with OI > $10k (of {len(deriv)} total)")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 OI Distribution", "💰 Funding Rates",
                                           "🔄 Churning", "📐 Basis & Spread"])

        with tab1:
            st.subheader("Open Interest by Exchange")
            oi_sorted = active.sort_values("open_interest_usd", ascending=True).tail(15)
            fig = go.Figure(go.Bar(
                y=oi_sorted["exchange"].str[:25],
                x=oi_sorted["open_interest_usd"] / 1e6,
                orientation="h",
                marker_color="#1f77b4",
                text=[f"${v/1e6:.1f}M" for v in oi_sorted["open_interest_usd"]],
                textposition="outside",
            ))
            fig.update_layout(height=500, xaxis_title="Open Interest ($M)",
                              margin=dict(l=200))
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart
            top10 = active.nlargest(10, "open_interest_usd")
            other = pd.DataFrame([{
                "exchange": "Others",
                "open_interest_usd": active["open_interest_usd"].sum() - top10["open_interest_usd"].sum()
            }])
            pie_data = pd.concat([top10[["exchange", "open_interest_usd"]], other])
            fig2 = px.pie(pie_data, values="open_interest_usd", names="exchange",
                          title="OI Market Share", hole=0.4)
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("Funding Rates by Exchange")
            fr_sorted = active.sort_values("funding_rate")
            colors = ["#d32f2f" if x < 0 else "#388e3c" for x in fr_sorted["funding_rate"]]
            fig = go.Figure(go.Bar(
                y=fr_sorted["exchange"].str[:25],
                x=fr_sorted["funding_rate"],
                orientation="h",
                marker_color=colors,
            ))
            fig.add_vline(x=0, line_color="black", line_width=1)
            fig.update_layout(height=max(400, len(fr_sorted) * 20),
                              xaxis_title="Funding Rate",
                              margin=dict(l=200))
            st.plotly_chart(fig, use_container_width=True)

            # Funding rate table
            st.subheader("Detailed View")
            display_cols = ["exchange", "funding_rate", "open_interest_usd", "volume_24h_usd", "basis_pct"]
            st.dataframe(active[display_cols].sort_values("funding_rate", ascending=False)
                         .style.format({
                             "funding_rate": "{:.6f}",
                             "open_interest_usd": "${:,.0f}",
                             "volume_24h_usd": "${:,.0f}",
                             "basis_pct": "{:.4f}%",
                         }),
                         use_container_width=True, height=400)

        with tab3:
            st.subheader("Volume / OI Ratio (Churning Detection)")
            active["vol_oi_ratio"] = active["volume_24h_usd"] / active["open_interest_usd"].replace(0, np.nan)
            churning = active.nlargest(15, "vol_oi_ratio")

            fig = go.Figure(go.Bar(
                y=churning["exchange"].str[:25],
                x=churning["vol_oi_ratio"],
                orientation="h",
                marker_color=["#d32f2f" if v > 10 else "#ff9800" if v > 5 else "#388e3c"
                               for v in churning["vol_oi_ratio"]],
                text=[f"{v:.1f}x" for v in churning["vol_oi_ratio"]],
                textposition="outside",
            ))
            fig.update_layout(height=500, xaxis_title="Volume / OI Ratio",
                              margin=dict(l=200))
            st.plotly_chart(fig, use_container_width=True)

            agg_ratio = active["volume_24h_usd"].sum() / active["open_interest_usd"].sum()
            if agg_ratio > 5:
                st.error(f"🚨 Aggregate churning extremely high: {agg_ratio:.1f}x")
            elif agg_ratio > 2:
                st.warning(f"⚠️ High churning: {agg_ratio:.1f}x")
            else:
                st.success(f"✅ Normal activity: {agg_ratio:.1f}x")

        with tab4:
            st.subheader("Basis Spread (Perp Premium/Discount)")
            basis_sorted = active.sort_values("basis_pct", ascending=False).head(20)
            fig = go.Figure(go.Bar(
                y=basis_sorted["exchange"].str[:25],
                x=basis_sorted["basis_pct"],
                orientation="h",
                marker_color=["#388e3c" if b > 0 else "#d32f2f" for b in basis_sorted["basis_pct"]],
            ))
            fig.add_vline(x=0, line_color="black", line_width=1)
            fig.update_layout(height=500, xaxis_title="Basis (%)",
                              margin=dict(l=200))
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# PAGE 3: BTC Correlation
# =========================================================================

elif page == "📈 BTC Correlation":
    st.title("📈 Bitcoin Correlation Analysis")

    btc = data.get("btc")
    ohlcv = data.get("ohlcv")

    if btc is None or ohlcv is None:
        st.warning("Missing BTC or FART data.")
    else:
        # Prepare merged data
        btc_price_col = "price" if "price" in btc.columns else "close"
        fart_price_col = "price" if "price" in ohlcv.columns else "close"

        btc_h = btc[[btc_price_col, "volume"]].resample("1h").last().dropna()
        btc_h.columns = ["btc_price", "btc_volume"]
        btc_h["btc_return"] = btc_h["btc_price"].pct_change()

        fart_h = ohlcv[[fart_price_col, "volume"]].resample("1h").last().dropna()
        fart_h.columns = ["fart_price", "fart_volume"]
        fart_h["fart_return"] = fart_h["fart_price"].pct_change()

        merged = btc_h.join(fart_h, how="inner").dropna(subset=["btc_return", "fart_return"])

        # --- Top Metrics ---
        corr = merged["btc_return"].corr(merged["fart_return"])
        valid = merged[["btc_return", "fart_return"]].dropna()
        beta = np.polyfit(valid["btc_return"], valid["fart_return"], 1)[0] if len(valid) > 50 else 0
        merged["rolling_corr"] = merged["btc_return"].rolling(24).corr(merged["fart_return"])
        decor_hours = (merged["rolling_corr"] < 0).sum()
        decor_pct = decor_hours / len(merged) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlation", f"{corr:.3f}")
        col2.metric("Beta", f"{beta:.2f}x", delta="amplifies BTC" if beta > 1 else "dampens BTC")
        col3.metric("Decorrelated Hours", f"{decor_hours}", delta=f"{decor_pct:.1f}% of time")
        col4.metric("Observations", f"{len(merged):,}")

        st.markdown("---")

        # --- Normalized Price Overlay ---
        st.subheader("Normalized Price Comparison")
        btc_norm = merged["btc_price"] / merged["btc_price"].iloc[0] * 100
        fart_norm = merged["fart_price"] / merged["fart_price"].iloc[0] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged.index, y=btc_norm, name="BTC",
                                 line=dict(color="#f7931a", width=2)))
        fig.add_trace(go.Scatter(x=merged.index, y=fart_norm, name="FARTCOIN",
                                 line=dict(color="#1f77b4", width=2)))
        fig.update_layout(height=400, yaxis_title="Indexed (100 = start)",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)

        # --- Rolling Correlation ---
        with col_l:
            st.subheader("Rolling 24h Correlation")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=merged.index, y=merged["rolling_corr"],
                                      line=dict(color="#1f77b4", width=1), name="24h corr"))
            fig2.add_hline(y=0, line_color="red", line_dash="dash")
            fig2.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.05)
            fig2.update_layout(height=350, yaxis_title="Correlation", yaxis_range=[-1, 1])
            st.plotly_chart(fig2, use_container_width=True)

            if decor_pct > 5:
                st.error(f"🚨 Frequent decorrelation ({decor_pct:.1f}%) — MM manipulation periods")
            elif decor_pct > 1:
                st.warning(f"⚠️ Occasional decorrelation ({decor_pct:.1f}%)")
            else:
                st.success(f"✅ Rare decorrelation ({decor_pct:.1f}%)")

            # Decorrelation move size
            decor_mask = merged["rolling_corr"] < 0
            if decor_mask.sum() > 0:
                decor_vol = merged.loc[decor_mask, "fart_return"].abs().mean() * 10000
                normal_vol = merged.loc[~decor_mask, "fart_return"].abs().mean() * 10000
                st.metric("Avg |move| during decorrelation", f"{decor_vol:.0f} bps",
                          delta=f"{decor_vol/normal_vol:.1f}x vs normal")

        # --- Scatter Plot ---
        with col_r:
            st.subheader("Return Scatter (Hourly)")
            fig3 = go.Figure()
            fig3.add_trace(go.Scattergl(
                x=merged["btc_return"] * 100, y=merged["fart_return"] * 100,
                mode="markers", marker=dict(size=3, opacity=0.3, color="#1f77b4"),
                name="Hourly returns"
            ))
            # Regression line
            x_line = np.linspace(merged["btc_return"].min(), merged["btc_return"].max(), 100)
            z = np.polyfit(valid["btc_return"], valid["fart_return"], 1)
            fig3.add_trace(go.Scatter(
                x=x_line * 100, y=(z[0] * x_line + z[1]) * 100,
                mode="lines", line=dict(color="red", width=2),
                name=f"β={z[0]:.2f}"
            ))
            fig3.update_layout(height=350, xaxis_title="BTC Return (%)",
                               yaxis_title="FART Return (%)")
            st.plotly_chart(fig3, use_container_width=True)

        # --- Lead/Lag ---
        st.subheader("Lead/Lag Analysis")
        lags = range(-8, 9)
        lag_corrs = [merged["btc_return"].shift(lag).corr(merged["fart_return"]) for lag in lags]
        colors = ["#388e3c" if c > 0 else "#d32f2f" for c in lag_corrs]
        fig4 = go.Figure(go.Bar(x=list(lags), y=lag_corrs, marker_color=colors))
        fig4.update_layout(height=300, xaxis_title="Lag (hrs, + = BTC leads)",
                           yaxis_title="Correlation")
        st.plotly_chart(fig4, use_container_width=True)

        peak_lag = list(lags)[np.argmax([abs(c) for c in lag_corrs])]
        if peak_lag > 0:
            st.info(f"📍 BTC leads Fartcoin by ~{peak_lag}h")
        elif peak_lag < 0:
            st.warning(f"📍 Fartcoin leads BTC by ~{abs(peak_lag)}h — unusual")
        else:
            st.info("📍 Simultaneous movement — no clear lead/lag")

        # --- BTC Regime ---
        st.subheader("Fartcoin Behavior by BTC Regime")
        merged["btc_ret_24h"] = merged["btc_price"].pct_change(24)

        def regime(r):
            if pd.isna(r): return None
            if r > 0.03: return "Strong Rally (>3%)"
            if r > 0.01: return "Mild Rally (1-3%)"
            if r > -0.01: return "Flat (-1% to 1%)"
            if r > -0.03: return "Mild Dump (-3 to -1%)"
            return "Strong Dump (<-3%)"

        merged["regime"] = merged["btc_ret_24h"].apply(regime)
        regime_order = ["Strong Rally (>3%)", "Mild Rally (1-3%)", "Flat (-1% to 1%)",
                        "Mild Dump (-3 to -1%)", "Strong Dump (<-3%)"]
        regime_means = merged.groupby("regime")["fart_return"].mean().reindex(regime_order).dropna() * 10000
        reg_colors = ["#1b5e20", "#66bb6a", "#9e9e9e", "#ef5350", "#b71c1c"][:len(regime_means)]

        fig5 = go.Figure(go.Bar(
            y=regime_means.index, x=regime_means.values,
            orientation="h", marker_color=reg_colors,
            text=[f"{v:.1f} bps" for v in regime_means.values],
            textposition="outside",
        ))
        fig5.add_vline(x=0, line_color="black", line_width=1)
        fig5.update_layout(height=300, xaxis_title="Avg FART Return (bps/hr)")
        st.plotly_chart(fig5, use_container_width=True)

        # --- Asymmetric Beta ---
        st.subheader("Asymmetric Response")
        btc_up = merged[merged["btc_return"] > 0.001]
        btc_down = merged[merged["btc_return"] < -0.001]

        if len(btc_up) > 20 and len(btc_down) > 20:
            up_beta = np.polyfit(btc_up["btc_return"], btc_up["fart_return"], 1)[0]
            down_beta = np.polyfit(btc_down["btc_return"], btc_down["fart_return"], 1)[0]

            col_u, col_d = st.columns(2)
            col_u.metric("Upside Beta", f"{up_beta:.2f}x", delta=f"{len(btc_up)} hours")
            col_d.metric("Downside Beta", f"{down_beta:.2f}x", delta=f"{len(btc_down)} hours")

            if down_beta > up_beta * 1.3:
                st.error("🚨 Falls harder than it rises — MMs amplify BTC downmoves")
            elif up_beta > down_beta * 1.3:
                st.warning("⚠️ Pumps harder than it dumps — MMs ride BTC rallies to exit")
            else:
                st.info("Symmetric beta — similar amplification in both directions")


# =========================================================================
# PAGE 4: Signals & Trades
# =========================================================================

elif page == "🎯 Signals & Trades":
    st.title("🎯 Signal Engine & Trade Generation")

    signals = data.get("signals")
    trades = data.get("trades")

    if signals is None:
        st.warning("No signals data. Run signal_engine.py first.")
    else:
        # --- Composite Signal Chart ---
        st.subheader("Composite Signal Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=signals.index, y=signals["composite"],
                                 line=dict(color="#1f77b4", width=1), name="Composite"))
        fig.add_hline(y=0.4, line_color="green", line_dash="dash", annotation_text="Long Entry")
        fig.add_hline(y=-0.4, line_color="red", line_dash="dash", annotation_text="Short Entry")
        fig.add_hline(y=0, line_color="gray", line_width=0.5)
        fig.add_hrect(y0=0.4, y1=signals["composite"].max() + 0.1, fillcolor="green", opacity=0.05)
        fig.add_hrect(y0=signals["composite"].min() - 0.1, y1=-0.4, fillcolor="red", opacity=0.05)
        fig.update_layout(height=400, yaxis_title="Score", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- Individual Signals ---
        st.subheader("Individual Signal Breakdown")
        sig_cols = [c for c in signals.columns if c.startswith("sig_")]

        fig2 = make_subplots(rows=len(sig_cols), cols=1, shared_xaxes=True,
                             subplot_titles=[c.replace("sig_", "").replace("_", " ").title()
                                             for c in sig_cols],
                             vertical_spacing=0.04)
        colors = px.colors.qualitative.Set2
        for i, col in enumerate(sig_cols):
            fig2.add_trace(go.Scatter(
                x=signals.index, y=signals[col],
                line=dict(color=colors[i % len(colors)], width=1),
                name=col.replace("sig_", ""),
            ), row=i+1, col=1)
            fig2.add_hline(y=0, line_color="gray", line_width=0.3, row=i+1, col=1)

        fig2.update_layout(height=200 * len(sig_cols), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Signal Stats ---
        st.subheader("Signal Statistics")
        stats = signals[sig_cols + ["composite"]].describe().T
        stats["current"] = signals[sig_cols + ["composite"]].iloc[-1]
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)

        # --- Trades ---
        st.subheader("Generated Trades")
        if trades is not None and not trades.empty:
            st.dataframe(trades, use_container_width=True)
            st.metric("Total Trades", len(trades))
        else:
            st.info("No trades generated with current thresholds.")

        # --- Backtest Quick Stats ---
        st.subheader("Quick Backtest (Threshold Analysis)")
        ohlcv = data.get("ohlcv")
        if ohlcv is not None:
            price_col = "price" if "price" in ohlcv.columns else "close"
            price = ohlcv[price_col]
            fwd_4h = price.pct_change(4).shift(-4)
            fwd_4h.name = "fwd_ret_4h"
            bt = signals[["composite"]].join(fwd_4h, how="left").dropna()

            thresh_results = []
            for t in [0.1, 0.2, 0.3, 0.4]:
                longs = bt[bt["composite"] > t]["fwd_ret_4h"]
                shorts = bt[bt["composite"] < -t]["fwd_ret_4h"]
                thresh_results.append({
                    "Threshold": f"±{t}",
                    "Long Trades": len(longs),
                    "Long Hit Rate": f"{(longs > 0).mean():.0%}" if len(longs) > 0 else "N/A",
                    "Long Avg Ret": f"{longs.mean():.3%}" if len(longs) > 0 else "N/A",
                    "Short Trades": len(shorts),
                    "Short Hit Rate": f"{(-shorts > 0).mean():.0%}" if len(shorts) > 0 else "N/A",
                    "Short Avg Ret": f"{(-shorts).mean():.3%}" if len(shorts) > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(thresh_results), use_container_width=True, hide_index=True)


# =========================================================================
# PAGE 5: Session Analysis
# =========================================================================

elif page == "⏰ Session Analysis":
    st.title("⏰ Trading Session Analysis")

    ohlcv = data.get("ohlcv")
    if ohlcv is None:
        st.warning("No hourly data available.")
    else:
        price_col = "price" if "price" in ohlcv.columns else "close"
        df = ohlcv.copy()
        df["return"] = df[price_col].pct_change()
        df["abs_return"] = df["return"].abs()
        df["hour"] = df.index.hour
        df["session"] = df["hour"].apply(classify_session)

        # --- Session Metrics ---
        session_order = ["Asia", "London", "NYC", "Late NYC"]
        session_stats = df.groupby("session")["return"].agg(["mean", "std", "count"])
        session_stats["mean_bps"] = session_stats["mean"] * 10000
        session_stats = session_stats.reindex(session_order)

        st.subheader("Session Performance (Your NYC Time)")
        col1, col2, col3, col4 = st.columns(4)
        session_info = {
            "Asia": ("8pm-4am ET", col1),
            "London": ("4am-9am ET", col2),
            "NYC": ("9am-5pm ET", col3),
            "Late NYC": ("5pm-8pm ET", col4),
        }
        for sess, (time_str, col) in session_info.items():
            if sess in session_stats.index:
                bps = session_stats.loc[sess, "mean_bps"]
                col.metric(f"{sess}", f"{bps:+.1f} bps/hr",
                           delta=time_str)

        st.markdown("---")

        col_l, col_r = st.columns(2)

        # --- Hourly Returns Heatmap ---
        with col_l:
            st.subheader("Average Return by Hour (UTC)")
            hourly_ret = df.groupby("hour")["return"].mean() * 10000
            colors = ["#388e3c" if v > 0 else "#d32f2f" for v in hourly_ret]
            fig = go.Figure(go.Bar(x=hourly_ret.index, y=hourly_ret.values,
                                   marker_color=colors))
            fig.update_layout(height=350, xaxis_title="Hour (UTC)",
                              yaxis_title="Avg Return (bps)",
                              xaxis=dict(dtick=1))
            # Add session shading
            fig.add_vrect(x0=-0.5, x1=7.5, fillcolor="blue", opacity=0.03,
                          annotation_text="Asia", annotation_position="top left")
            fig.add_vrect(x0=7.5, x1=12.5, fillcolor="green", opacity=0.03,
                          annotation_text="London", annotation_position="top left")
            fig.add_vrect(x0=12.5, x1=20.5, fillcolor="orange", opacity=0.03,
                          annotation_text="NYC", annotation_position="top left")
            fig.add_vrect(x0=20.5, x1=23.5, fillcolor="purple", opacity=0.03,
                          annotation_text="Late", annotation_position="top left")
            st.plotly_chart(fig, use_container_width=True)

        # --- Volatility by Hour ---
        with col_r:
            st.subheader("Avg |Move| by Hour (UTC)")
            hourly_vol = df.groupby("hour")["abs_return"].mean() * 100
            fig2 = go.Figure(go.Bar(x=hourly_vol.index, y=hourly_vol.values,
                                    marker_color="#ff9800"))
            fig2.update_layout(height=350, xaxis_title="Hour (UTC)",
                               yaxis_title="Avg |Return| (%)",
                               xaxis=dict(dtick=1))
            fig2.add_vrect(x0=-0.5, x1=7.5, fillcolor="blue", opacity=0.03)
            fig2.add_vrect(x0=7.5, x1=12.5, fillcolor="green", opacity=0.03)
            fig2.add_vrect(x0=12.5, x1=20.5, fillcolor="orange", opacity=0.03)
            fig2.add_vrect(x0=20.5, x1=23.5, fillcolor="purple", opacity=0.03)
            st.plotly_chart(fig2, use_container_width=True)

        # --- Big Moves by Session ---
        st.subheader("Big Move Distribution (>3% hourly)")
        big = df[df["abs_return"] > 0.03]
        if not big.empty:
            big_counts = big.groupby("session").size().reindex(session_order, fill_value=0)
            total_counts = df.groupby("session").size().reindex(session_order)
            freq = big_counts / total_counts * 100

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=session_order, y=big_counts.values, name="Count",
                                  marker_color="#d32f2f", yaxis="y"))
            fig3.add_trace(go.Scatter(x=session_order, y=freq.values, name="% of hours",
                                      line=dict(color="#1f77b4", width=3), yaxis="y2",
                                      mode="lines+markers"))
            fig3.update_layout(
                height=350,
                yaxis=dict(title="Big Move Count"),
                yaxis2=dict(title="% of Session Hours", overlaying="y", side="right"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # --- Session Transitions ---
        st.subheader("Session Transitions (Where MMs Hand Off)")
        transitions = [
            ("Asia → London", 7, 8, 9),
            ("London → NYC", 12, 13, 14),
            ("NYC → Late NYC", 17, 18, 19),
            ("Late NYC → Asia", 23, 0, 1),
        ]

        trans_data = []
        for name, h_b, h_m, h_a in transitions:
            b = df[df["hour"] == h_b]["return"].mean() * 10000
            m = df[df["hour"] == h_m]["return"].mean() * 10000
            a = df[df["hour"] == h_a]["return"].mean() * 10000
            pattern = "🔄 REVERSAL" if np.sign(b) != np.sign(a) else "➡️ CONTINUATION"
            trans_data.append({
                "Transition": name,
                "Before (bps)": f"{b:+.1f}",
                "Boundary (bps)": f"{m:+.1f}",
                "After (bps)": f"{a:+.1f}",
                "Pattern": pattern,
            })
        st.dataframe(pd.DataFrame(trans_data), use_container_width=True, hide_index=True)

        # --- Day of Week ---
        st.subheader("Returns by Day of Week")
        daily = data.get("ohlcv_daily")
        if daily is not None:
            daily_c = daily.copy()
            daily_c["return"] = daily_c["close"].pct_change()
            daily_c["weekday"] = daily_c.index.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            wd_mean = daily_c.groupby("weekday")["return"].mean().reindex(day_order) * 100
            colors = ["#388e3c" if v > 0 else "#d32f2f" for v in wd_mean]
            fig4 = go.Figure(go.Bar(x=[d[:3] for d in day_order], y=wd_mean.values,
                                    marker_color=colors))
            fig4.update_layout(height=300, yaxis_title="Avg Return (%)")
            fig4.add_hline(y=0, line_color="black", line_width=0.5)
            st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("Built for alpha detection. Not financial advice.")
