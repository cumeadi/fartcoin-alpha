"""
Market State — Fartcoin Alpha Framework

Shared market state logic used by both the dashboard and automation pipeline.
No Streamlit dependency — pure Python + pandas.
"""

import numpy as np
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Session Constants
# ---------------------------------------------------------------------------

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


def _next_positive_hour(current_hour):
    """Find the next hour with positive historical bias."""
    for offset in range(1, 24):
        h = (current_hour + offset) % 24
        if HOURLY_BIAS.get(h, 0) > 5:
            return h
    return (current_hour + 1) % 24


# ---------------------------------------------------------------------------
# Market State Computation
# ---------------------------------------------------------------------------

def compute_market_state(data):
    """Compute current market state from data dict. No Streamlit dependency."""
    state = {}
    now = datetime.now(timezone.utc)
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


# ---------------------------------------------------------------------------
# Action Determination
# ---------------------------------------------------------------------------

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

        good_asia_day = weekday_short in ("Mon", "Wed", "Fri")
        bad_asia_day = weekday_short in ("Thu", "Sat", "Sun")
        late_asia_window = asia_sub == "Late Asia"

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
                if hourly_bias < -15:
                    session_ok = False
                    session_note = f"Current hour has strong negative bias ({hourly_bias:+.0f} bps). Wait."
                else:
                    asia_note = f"{weekday_short} Asia is mixed. Proceed with caution."

            if carry > 0.005 and carry <= 0.02:
                asia_note += f" Late NYC carry +{carry:.1%} — CAUTION: mild positive carry → Asia tends to give back (-1.5% avg)."
                if conviction != "HIGH":
                    conviction = "LOW"

        elif direction == "SHORT":
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
            conviction = "LOW"
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
