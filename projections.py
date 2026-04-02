"""
Forward Projection Engine — Fartcoin Alpha Framework

6 projection sub-models that transform reactive signals into forward-looking
intelligence. Pure Python + pandas + numpy + scipy — no ML libraries.

Entry point: compute_projections(data, market_state) -> dict
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timezone

from market_state import (
    HOURLY_BIAS, SESSION_MAP, classify_session,
)


# =========================================================================
# 1. Probabilistic Forward Returns
# =========================================================================

def _logistic(z):
    """Numerically stable logistic function."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _neg_log_likelihood(theta, X, y):
    """Negative log-likelihood for logistic regression."""
    p = _logistic(X @ theta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _build_feature_matrix(df, for_prediction=False, state=None):
    """
    Build enriched feature matrix for probability model.
    Features: composite, |composite|, composite², session dummies,
    hourly_bias, funding z-score, BTC regime flag.
    """
    composite = df["composite"].values

    features = [
        np.ones(len(df)),            # intercept
        composite,                   # signal direction
        np.abs(composite),           # signal magnitude
        composite ** 2,              # non-linearity
    ]

    if for_prediction and state is not None:
        # For single-row prediction, build from state
        hour = state.get("utc_hour", 12)
        session = state.get("session", "NYC")
        features.append(np.array([HOURLY_BIAS.get(hour, 0) / 100]))  # hourly bias (scaled)
        features.append(np.array([1.0 if session == "London" else 0.0]))
        features.append(np.array([1.0 if session == "Late NYC" else 0.0]))
        features.append(np.array([1.0 if session == "Asia" else 0.0]))

        # Funding z-score proxy: use sig_funding if available
        sig_funding = state.get("signals", {}).get("sig_funding", 0)
        features.append(np.array([sig_funding]))

        # BTC regime: bullish=+1, bearish=-1, flat=0
        btc_regime = state.get("btc_regime", "Unknown")
        btc_flag = 1.0 if "Rally" in btc_regime else (-1.0 if "Dump" in btc_regime else 0.0)
        features.append(np.array([btc_flag]))

        return np.array([f[0] for f in features]).reshape(1, -1)

    # For training: build from DataFrame columns
    if "hour" in df.columns:
        hourly_bias_vals = df["hour"].map(lambda h: HOURLY_BIAS.get(h, 0) / 100).values
    else:
        hourly_bias_vals = np.zeros(len(df))
    features.append(hourly_bias_vals)

    if "session" in df.columns:
        features.append((df["session"] == "London").astype(float).values)
        features.append((df["session"] == "Late NYC").astype(float).values)
        features.append((df["session"] == "Asia").astype(float).values)
    else:
        features.extend([np.zeros(len(df))] * 3)

    if "sig_funding" in df.columns:
        features.append(df["sig_funding"].fillna(0).values)
    else:
        features.append(np.zeros(len(df)))

    # BTC regime not available in historical training — use zero
    features.append(np.zeros(len(df)))

    return np.column_stack(features)


def _project_probabilistic_return(data, state):
    """
    Calibrate composite score → probability of positive 4h return.
    Enhanced with session, hourly bias, funding, and BTC regime features.
    """
    result = {
        "prob_positive_4h": 0.5,
        "expected_move_pct": 0.0,
        "horizon_hours": 4,
        "model_n_train": 0,
        "description": "Insufficient data for probability estimate.",
    }

    signals = data.get("signals")
    ohlcv = data.get("ohlcv")
    if signals is None or ohlcv is None:
        return result

    price_col = "price" if "price" in ohlcv.columns else "close"

    # Align signals with price data on common index
    common_idx = signals.index.intersection(ohlcv.index)
    sig_cols = [c for c in signals.columns if c in ["composite", "sig_funding"]]
    df = signals.loc[common_idx, sig_cols].copy()
    df["fwd_ret_4h"] = ohlcv.loc[common_idx, price_col].pct_change(4).shift(-4)
    df["hour"] = df.index.hour if hasattr(df.index, 'hour') else 12
    df["session"] = df["hour"].apply(classify_session)
    df = df.dropna(subset=["fwd_ret_4h", "composite"])

    if len(df) < 50:
        return result

    # Build enriched feature matrix
    X = _build_feature_matrix(df)
    y = (df["fwd_ret_4h"].values > 0).astype(float)

    # Fit logistic regression via L-BFGS-B
    theta0 = np.zeros(X.shape[1])
    try:
        res = minimize(_neg_log_likelihood, theta0, args=(X, y),
                       method="L-BFGS-B", options={"maxiter": 300})
        theta = res.x
    except Exception:
        return result

    # Predict for current state
    x_now = _build_feature_matrix(df.iloc[-1:], for_prediction=True, state=state)
    prob = float(_logistic(x_now @ theta).ravel()[0])

    # Expected magnitude: session-specific volatility × sqrt(horizon)
    session = state.get("session", "NYC")
    session_hours = [h for h in range(24) if classify_session(h) == session]
    if session_hours and hasattr(ohlcv.index, 'hour'):
        mask = ohlcv.index.hour.isin(session_hours)
        session_vol = ohlcv.loc[mask, price_col].pct_change().std()
    else:
        session_vol = ohlcv[price_col].pct_change().std()

    if np.isnan(session_vol) or session_vol == 0:
        session_vol = 0.02

    expected_magnitude = session_vol * np.sqrt(4) * 100  # in pct

    # Direction-adjusted expected move
    if prob > 0.5:
        expected_move = expected_magnitude * (prob - 0.5) * 2
    else:
        expected_move = -expected_magnitude * (0.5 - prob) * 2

    # Apply session-direction filter from backtest findings
    session_warning = ""
    if session == "London" and prob > 0.5:
        # London LONG confirmed negative edge (-0.35%) — downgrade
        prob = 0.5 + (prob - 0.5) * 0.5  # shrink toward 0.5
        expected_move *= 0.5
        session_warning = " (London LONG edge is weak — conviction reduced)"
    elif session == "Late NYC" and prob < 0.5:
        # Late NYC SHORT is best edge (+1.07%) — upgrade
        prob = prob * 0.85  # push further bearish
        expected_move *= 1.3
        session_warning = " (Late NYC SHORT has strong historical edge — conviction boosted)"

    # Conviction label
    if prob > 0.65:
        conviction = "HIGH"
    elif prob > 0.55:
        conviction = "MODERATE"
    elif prob < 0.35:
        conviction = "HIGH (bearish)"
    elif prob < 0.45:
        conviction = "MODERATE (bearish)"
    else:
        conviction = "LOW"

    result.update({
        "prob_positive_4h": round(prob, 4),
        "expected_move_pct": round(expected_move, 2),
        "model_n_train": len(df),
        "conviction": conviction,
        "description": (
            f"{prob:.0%} probability of positive 4h return. "
            f"Expected move: {expected_move:+.2f}%. "
            f"Conviction: {conviction}. "
            f"(Model: {X.shape[1]} features, {len(df)} obs)"
            f"{session_warning}"
        ),
    })
    return result


# =========================================================================
# 2. Mean-Reversion Timing (Funding + LSR)
# =========================================================================

def _estimate_ar1(series):
    """Estimate AR(1) coefficient and half-life from a time series."""
    s = series.dropna()
    if len(s) < 30:
        return {"phi": 0.5, "half_life_h": 10, "mean": 0}

    mean = s.mean()
    y = s.values[1:]
    y_lag = s.values[:-1]

    var_lag = np.var(y_lag)
    if var_lag < 1e-15:
        return {"phi": 0.5, "half_life_h": 10, "mean": float(mean)}

    phi = np.cov(y, y_lag)[0, 1] / var_lag
    phi = np.clip(phi, 0.01, 0.999)  # ensure mean-reverting

    half_life = -np.log(2) / np.log(phi)
    return {
        "phi": float(phi),
        "half_life_h": float(half_life),
        "mean": float(mean),
    }


def _project_decay_path(current, mean, phi, steps=24):
    """Project AR(1) decay path forward."""
    path = []
    val = current
    for _ in range(steps):
        val = mean + phi * (val - mean)
        path.append(val)
    return path


def _project_mean_reversion(data, state):
    """Project funding rate and LSR mean-reversion timing."""
    result = {"funding": None, "lsr": None}

    # --- Funding ---
    funding_df = data.get("funding")
    if funding_df is not None and not funding_df.empty:
        fr_series = funding_df["fundingRate"]
        ar1 = _estimate_ar1(fr_series)
        current_fr = float(fr_series.iloc[-1])
        path = _project_decay_path(current_fr, ar1["mean"], ar1["phi"], 24)

        # Time to cross neutral (mean)
        cross_time = None
        threshold = ar1["mean"]
        for t, v in enumerate(path, 1):
            if (current_fr > threshold and v <= threshold) or \
               (current_fr < threshold and v >= threshold):
                cross_time = t
                break

        # Also check real derivatives funding if available
        real_funding = state.get("avg_funding", 0)

        result["funding"] = {
            "current_synthetic": round(current_fr, 6),
            "current_real": round(real_funding, 6),
            "mean": round(ar1["mean"], 6),
            "phi": round(ar1["phi"], 4),
            "half_life_h": round(ar1["half_life_h"], 1),
            "projected_cross_time_h": cross_time,
            "projected_path": [round(v, 6) for v in path],
            "description": (
                f"Funding half-life: {ar1['half_life_h']:.1f}h. "
                f"Current: {current_fr:.6f} (real: {real_funding:.4f}). "
                f"{'Projected to cross neutral in ' + str(cross_time) + 'h.' if cross_time else 'Near equilibrium.'}"
            ),
        }

    # --- LSR (percentile-rank reversion model) ---
    lsr_df = data.get("lsr")
    if lsr_df is not None and not lsr_df.empty:
        col = "longShortRatio" if "longShortRatio" in lsr_df.columns else lsr_df.columns[0]
        lsr_series = lsr_df[col].dropna()
        current_lsr = float(lsr_series.iloc[-1])
        median_lsr = float(lsr_series.median())

        # Percentile rank of current value
        percentile = (lsr_series < current_lsr).mean()

        # Measure empirical reversion: for each historical extreme,
        # how many hours until it returned to median?
        p90 = lsr_series.quantile(0.90)
        p10 = lsr_series.quantile(0.10)

        # Estimate reversion time from empirical data
        revert_times = []
        vals = lsr_series.values
        for i in range(len(vals) - 1):
            is_extreme = vals[i] > p90 or vals[i] < p10
            if not is_extreme:
                continue
            # Find first return to median zone (25th-75th percentile)
            p25 = lsr_series.quantile(0.25)
            p75 = lsr_series.quantile(0.75)
            for j in range(i + 1, min(i + 25, len(vals))):
                if p25 <= vals[j] <= p75:
                    revert_times.append(j - i)
                    break

        avg_revert_time = np.mean(revert_times) if revert_times else 12
        revert_rate = len(revert_times) / max(1, sum(1 for v in vals if v > p90 or v < p10))

        # Project path: use empirical median-reversion curve
        # Exponential decay toward median with empirical time constant
        tau = avg_revert_time / np.log(2) if avg_revert_time > 0 else 5
        path = []
        for t in range(1, 25):
            projected = median_lsr + (current_lsr - median_lsr) * np.exp(-t / tau)
            path.append(projected)

        # Time to cross equilibrium (1.0)
        cross_time = None
        for t, v in enumerate(path, 1):
            if (current_lsr > 1.0 and v <= 1.0) or (current_lsr < 1.0 and v >= 1.0):
                cross_time = t
                break

        # Extremity assessment
        if percentile > 0.90:
            extremity = f"EXTREME LONG bias (top {(1-percentile)*100:.0f}% of history)"
        elif percentile > 0.75:
            extremity = f"High long bias ({percentile:.0%} percentile)"
        elif percentile < 0.10:
            extremity = f"EXTREME SHORT bias (bottom {percentile*100:.0f}% of history)"
        elif percentile < 0.25:
            extremity = f"High short bias ({percentile:.0%} percentile)"
        else:
            extremity = f"Normal range ({percentile:.0%} percentile)"

        result["lsr"] = {
            "current": round(current_lsr, 4),
            "median": round(median_lsr, 4),
            "percentile": round(percentile, 4),
            "avg_revert_time_h": round(avg_revert_time, 1),
            "revert_rate": round(revert_rate, 2),
            "projected_cross_time_h": cross_time,
            "projected_path": [round(v, 4) for v in path],
            "description": (
                f"LSR: {current_lsr:.4f} — {extremity}. "
                f"Median: {median_lsr:.4f}. "
                f"Avg reversion time from extremes: {avg_revert_time:.0f}h "
                f"(revert rate: {revert_rate:.0%}). "
                f"{'Projected to cross equilibrium in ' + str(cross_time) + 'h.' if cross_time else 'Near equilibrium.'}"
            ),
        }

    return result


# =========================================================================
# 3. Hourly Manipulation Cycle Detector
# =========================================================================

def _detect_hourly_manipulation_cycle(data, state):
    """
    4-phase state machine on hourly data:
    DORMANT → QUIET_ACCUMULATION → BUILDUP → SPIKE_IN_PROGRESS
    """
    result = {
        "phase": "DORMANT",
        "description": "No manipulation pattern detected.",
        "hours_in_phase": 0,
        "est_hours_to_move": None,
        "confidence": 0.0,
    }

    ohlcv = data.get("ohlcv")
    oi_df = data.get("oi")
    if ohlcv is None or oi_df is None or len(ohlcv) < 12:
        return result

    price_col = "price" if "price" in ohlcv.columns else "close"
    oi_col = oi_df.columns[0]  # Usually 'openInterest' or similar

    # Recent window: last 12 hours
    recent_ohlcv = ohlcv.iloc[-12:]
    recent_oi = oi_df.iloc[-12:]

    # Volume ratio: current vs rolling 24h mean
    vol_24h_mean = ohlcv["volume"].iloc[-24:].mean() if len(ohlcv) >= 24 else ohlcv["volume"].mean()
    recent_vol_ratios = recent_ohlcv["volume"] / vol_24h_mean

    # OI change over 4h windows
    oi_4h_pct = recent_oi[oi_col].pct_change(4)

    # Check each phase condition from most advanced to least
    last_3_vol_ratio = recent_vol_ratios.iloc[-3:].values if len(recent_vol_ratios) >= 3 else []
    last_vol_ratio = recent_vol_ratios.iloc[-1] if len(recent_vol_ratios) > 0 else 1.0
    last_oi_4h_change = oi_4h_pct.iloc[-1] if len(oi_4h_pct) > 0 else 0

    # Get signal state
    signals = data.get("signals")
    oi_accel_sig = 0
    if signals is not None and "sig_oi_accel" in signals.columns:
        oi_accel_sig = signals["sig_oi_accel"].iloc[-1] if len(signals) > 0 else 0

    # Phase: SPIKE_IN_PROGRESS
    if last_vol_ratio > 1.5 and abs(last_oi_4h_change) > 0.05:
        # Count hours in spike
        hours_in = sum(1 for v in recent_vol_ratios.values[-6:] if v > 1.3)
        result = {
            "phase": "SPIKE_IN_PROGRESS",
            "description": (
                f"Manufactured move underway. Volume {last_vol_ratio:.1f}x normal, "
                f"OI changed {last_oi_4h_change:+.1%} in 4h. "
                f"This is the liquidation/squeeze phase."
            ),
            "hours_in_phase": hours_in,
            "est_hours_to_move": 0,
            "confidence": 0.85,
        }

    # Phase: BUILDUP
    elif oi_accel_sig > 0.5 and last_oi_4h_change > 0.02:
        quiet_hours = sum(1 for v in last_3_vol_ratio if v < 0.8) if len(last_3_vol_ratio) > 0 else 0
        result = {
            "phase": "BUILDUP",
            "description": (
                f"Position buildup detected. OI accel signal: {oi_accel_sig:.2f}, "
                f"OI +{last_oi_4h_change:.1%} in 4h. "
                f"Manufactured move likely within 2-4 hours."
            ),
            "hours_in_phase": max(1, quiet_hours),
            "est_hours_to_move": 3,
            "confidence": 0.65,
        }

    # Phase: QUIET_ACCUMULATION
    elif len(last_3_vol_ratio) >= 3 and all(v < 0.7 for v in last_3_vol_ratio) and last_oi_4h_change > 0.02:
        result = {
            "phase": "QUIET_ACCUMULATION",
            "description": (
                f"Quiet accumulation phase. Volume {np.mean(last_3_vol_ratio):.1f}x normal "
                f"for 3+ hours while OI rising +{last_oi_4h_change:.1%}. "
                f"Positions building under the radar."
            ),
            "hours_in_phase": 3,
            "est_hours_to_move": 6,
            "confidence": 0.45,
        }

    return result


# =========================================================================
# 4. Session-Conditional Projections
# =========================================================================

def _project_session_conditional(data, state):
    """
    Compute conditional historical return given current session + signal direction.
    """
    session = state.get("session", "NYC")
    direction = "LONG" if state.get("composite", 0) > 0 else "SHORT"
    hour = state.get("utc_hour", 12)

    result = {
        "session": session,
        "direction": direction,
        "conditional_avg_return_pct": 0.0,
        "session_bias_4h_bps": 0.0,
        "combined_edge_pct": 0.0,
        "n_samples": 0,
        "description": "Insufficient data.",
    }

    # Sum forward hourly bias for next 4 hours
    bias_sum = sum(HOURLY_BIAS.get((hour + i) % 24, 0) for i in range(1, 5))

    ohlcv = data.get("ohlcv")
    signals = data.get("signals")
    if ohlcv is None or signals is None:
        result["session_bias_4h_bps"] = round(bias_sum, 1)
        result["description"] = f"Session bias next 4h: {bias_sum:+.1f} bps total."
        return result

    price_col = "price" if "price" in ohlcv.columns else "close"

    # Build filtered returns — align signals with ohlcv on common index
    common_idx = signals.index.intersection(ohlcv.index)
    df = signals.loc[common_idx, ["composite"]].copy()
    fwd_ret = ohlcv.loc[common_idx, price_col].pct_change(4).shift(-4)
    df["fwd_ret_4h"] = fwd_ret
    df["hour"] = df.index.hour if hasattr(df.index, 'hour') else 12
    df["session"] = df["hour"].apply(classify_session)
    df = df.dropna(subset=["fwd_ret_4h"])

    # Filter by current session and signal direction
    if direction == "LONG":
        mask = (df["session"] == session) & (df["composite"] > 0.1)
    else:
        mask = (df["session"] == session) & (df["composite"] < -0.1)

    filtered = df[mask]
    n = len(filtered)

    if n > 10:
        cond_avg = filtered["fwd_ret_4h"].mean() * 100  # pct
    else:
        cond_avg = 0

    combined_edge = cond_avg + (bias_sum / 100)  # convert bps to pct

    # Session-direction quality assessment from backtest evidence
    quality = "NEUTRAL"
    warning = ""

    # Backtest-confirmed edge rankings:
    # Best:  Late NYC SHORT +1.07% (69% hit), Asia SHORT +0.35%, NYC LONG +0.40%
    # Worst: London LONG -0.35% (41% hit)
    if session == "London" and direction == "LONG":
        quality = "AVOID"
        warning = " London LONG has confirmed NEGATIVE edge (-0.35% avg, 41% hit rate)."
    elif session == "Late NYC" and direction == "SHORT":
        quality = "STRONG"
        warning = " Late NYC SHORT is the highest-edge combo (+1.07% avg, 69% hit rate)."
    elif session == "Asia" and direction == "SHORT":
        quality = "FAVORABLE"
        warning = " Asia SHORT has positive edge (+0.35% avg, confirmed by backtest)."
    elif session == "NYC" and direction == "LONG":
        quality = "FAVORABLE"
        warning = " NYC LONG has positive edge (+0.40% avg)."

    result.update({
        "conditional_avg_return_pct": round(cond_avg, 3),
        "session_bias_4h_bps": round(bias_sum, 1),
        "combined_edge_pct": round(combined_edge, 3),
        "n_samples": n,
        "quality": quality,
        "description": (
            f"{direction} during {session}: avg 4h return {cond_avg:+.2f}% "
            f"(n={n}). Session bias next 4h: {bias_sum:+.0f} bps. "
            f"Combined edge: {combined_edge:+.2f}%. "
            f"Quality: {quality}.{warning}"
        ),
    })
    return result


# =========================================================================
# 5. BTC Lead-Lag Projection
# =========================================================================

def _project_btc_lead_lag(data, state):
    """
    BTC moves concurrently with Fart (0h lag, corr=0.50, beta=1.7x).
    Backtest confirmed: 91% direction accuracy during big BTC moves.
    """
    result = {
        "btc_2h_return_pct": 0.0,
        "beta": 1.70,
        "projected_fart_move_pct": 0.0,
        "rolling_corr_24h": 0.0,
        "confidence": 0.0,
        "direction_accuracy": 0.0,
        "description": "No BTC data available.",
    }

    btc = data.get("btc")
    ohlcv = data.get("ohlcv")
    if btc is None or ohlcv is None or len(btc) < 25:
        return result

    btc_col = "price" if "price" in btc.columns else "close"
    fart_col = "price" if "price" in ohlcv.columns else "close"

    # Round both to nearest hour for alignment
    btc_hourly = btc[[btc_col]].copy()
    btc_hourly.index = btc_hourly.index.round("h")
    btc_hourly = btc_hourly[~btc_hourly.index.duplicated(keep="last")]

    fart_hourly = ohlcv[[fart_col]].copy()
    fart_hourly.index = fart_hourly.index.round("h")
    fart_hourly = fart_hourly[~fart_hourly.index.duplicated(keep="last")]

    common = btc_hourly.index.intersection(fart_hourly.index)
    if len(common) < 50:
        return result

    # Use concurrent (0h lag) — backtest confirmed strongest correlation
    btc_ret_2h = btc_hourly.loc[common, btc_col].pct_change(2)
    fart_ret_4h = fart_hourly.loc[common, fart_col].pct_change(4)

    aligned = pd.DataFrame({"btc": btc_ret_2h, "fart": fart_ret_4h}).dropna()
    if len(aligned) < 50:
        return result

    # Estimate beta via polyfit
    try:
        coeffs = np.polyfit(aligned["btc"].values, aligned["fart"].values, 1)
        beta = coeffs[0]
    except Exception:
        beta = 1.70

    # Current BTC 2h return
    btc_2h_ret = float(btc_ret_2h.iloc[-1]) if not pd.isna(btc_ret_2h.iloc[-1]) else 0

    # Rolling 24h correlation on hourly returns
    btc_1h = btc_hourly.loc[common, btc_col].pct_change().dropna()
    fart_1h = fart_hourly.loc[common, fart_col].pct_change().dropna()
    common_1h = btc_1h.index.intersection(fart_1h.index)
    if len(common_1h) >= 24:
        rolling_corr = float(btc_1h.loc[common_1h].rolling(24).corr(fart_1h.loc[common_1h]).iloc[-1])
    else:
        rolling_corr = float(btc_1h.loc[common_1h].corr(fart_1h.loc[common_1h]))
    rolling_corr = rolling_corr if not np.isnan(rolling_corr) else 0.0

    # Direction accuracy: overall and during big moves
    dir_acc_all = ((aligned["btc"] > 0) == (aligned["fart"] > 0)).mean()
    big_moves = aligned[aligned["btc"].abs() > 0.01]
    dir_acc_big = ((big_moves["btc"] > 0) == (big_moves["fart"] > 0)).mean() if len(big_moves) > 5 else dir_acc_all

    # Projected Fart move
    projected_move = btc_2h_ret * beta * 100

    # Confidence: higher for big BTC moves (where accuracy is 91%)
    base_conf = min(1.0, abs(rolling_corr))
    if abs(btc_2h_ret) > 0.01:
        confidence = min(1.0, base_conf * 1.3)  # boost for big moves
    elif abs(btc_2h_ret) > 0.005:
        confidence = base_conf
    else:
        confidence = base_conf * 0.5  # low confidence for small moves

    # Description
    if abs(btc_2h_ret) > 0.02:
        urgency = "MAJOR"
    elif abs(btc_2h_ret) > 0.01:
        urgency = "SIGNIFICANT"
    elif abs(btc_2h_ret) > 0.005:
        urgency = "Notable"
    else:
        urgency = "Minor"

    direction = "rallied" if btc_2h_ret > 0 else "dropped"

    result.update({
        "btc_2h_return_pct": round(btc_2h_ret * 100, 2),
        "beta": round(beta, 2),
        "projected_fart_move_pct": round(projected_move, 2),
        "rolling_corr_24h": round(rolling_corr, 3),
        "confidence": round(confidence, 2),
        "direction_accuracy": round(dir_acc_big, 3),
        "description": (
            f"{urgency}: BTC {direction} {abs(btc_2h_ret)*100:.1f}% in last 2h. "
            f"With {beta:.1f}x beta, projected FART response: {projected_move:+.1f}%. "
            f"Correlation: {rolling_corr:.2f}. "
            f"Direction accuracy: {dir_acc_big:.0%} (big moves). "
            f"Confidence: {confidence:.0%}."
        ),
    })
    return result


# =========================================================================
# 6. Confidence Intervals
# =========================================================================

def _compute_confidence_intervals(data, state, prob_result):
    """
    Session-specific volatility-based confidence bands at 4h and 8h.
    """
    result = {"h4": None, "h8": None}

    ohlcv = data.get("ohlcv")
    if ohlcv is None or len(ohlcv) < 25:
        return result

    price_col = "price" if "price" in ohlcv.columns else "close"
    current_price = float(ohlcv[price_col].iloc[-1])
    session = state.get("session", "NYC")
    hour = state.get("utc_hour", 12)

    # Session-specific hourly volatility
    session_hours = [h for h in range(24) if classify_session(h) == session]
    if session_hours and hasattr(ohlcv.index, 'hour'):
        mask = ohlcv.index.hour.isin(session_hours)
        hourly_vol = ohlcv.loc[mask, price_col].pct_change().std()
    else:
        hourly_vol = ohlcv[price_col].pct_change().std()

    if np.isnan(hourly_vol) or hourly_vol == 0:
        hourly_vol = 0.02  # fallback 2% hourly vol

    # Point estimate direction from probability model
    expected_move_pct = prob_result.get("expected_move_pct", 0) / 100

    for horizon, label in [(4, "h4"), (8, "h8")]:
        vol_h = hourly_vol * np.sqrt(horizon)
        center = current_price * (1 + expected_move_pct * (horizon / 4))

        result[label] = {
            "horizon_hours": horizon,
            "center": round(center, 6),
            "current_price": round(current_price, 6),
            "vol_pct": round(vol_h * 100, 2),
            "low_68": round(center * (1 - vol_h), 6),
            "high_68": round(center * (1 + vol_h), 6),
            "low_95": round(center * (1 - 2 * vol_h), 6),
            "high_95": round(center * (1 + 2 * vol_h), 6),
        }

    return result


# =========================================================================
# Model 7: News Sentiment Momentum
# =========================================================================

def _project_news_sentiment(data, market_state):
    """
    Assess market sentiment from 3 free sources:
      - Alternative.me Fear & Greed Index (market-wide baseline)
      - CoinGecko community sentiment (FARTCOIN-specific votes)
      - CoinPaprika short-term price action (15m/1h spike detection)

    Also reads the combined sentiment_history.csv for divergence detection
    (e.g., FART pumping while market is in Extreme Fear = manipulation).

    Falls back to news_sentiment_hourly.csv for the rolling sentiment composite.
    """
    # Try sentiment_history first (richer data from new collectors)
    try:
        from pathlib import Path as _P
        _sent_file = _P(__file__).parent / "data" / "sentiment_history.csv"
        if _sent_file.exists():
            sent_df = pd.read_csv(_sent_file, parse_dates=["timestamp"])
            if not sent_df.empty:
                latest = sent_df.iloc[-1]

                fear_greed = int(latest.get("fear_greed_value", 50))
                fg_class = latest.get("fear_greed_class", "Neutral")
                fg_trend = latest.get("fear_greed_trend", "UNKNOWN")
                cg_up = float(latest.get("cg_sentiment_up_pct", 50))
                cp_15m = float(latest.get("cp_pct_change_15m", 0))
                cp_1h = float(latest.get("cp_pct_change_1h", 0))
                cp_vol_chg = float(latest.get("cp_volume_change_24h", 0))
                composite = float(latest.get("sentiment_composite", 0))
                divergence = latest.get("divergence", "NONE")
                div_desc = latest.get("divergence_desc", "")

                # Determine assessment
                if divergence == "PUMP_IN_FEAR":
                    assessment = "DANGER"
                    description = div_desc
                elif divergence == "VOLUME_PUMP":
                    assessment = "CAUTION"
                    description = div_desc
                elif divergence == "DUMP_IN_OPTIMISM":
                    assessment = "DANGER"
                    description = div_desc
                elif composite > 0.3:
                    assessment = "BULLISH_MOMENTUM"
                    description = (
                        f"Sentiment bullish (composite: {composite:+.2f}). "
                        f"Fear & Greed: {fear_greed} ({fg_class}). "
                        f"Community: {cg_up:.0f}% bullish. "
                        f"1h move: {cp_1h:+.1f}%.")
                elif composite < -0.3:
                    assessment = "BEARISH_MOMENTUM"
                    description = (
                        f"Sentiment bearish (composite: {composite:+.2f}). "
                        f"Fear & Greed: {fear_greed} ({fg_class}). "
                        f"Community: {cg_up:.0f}% bullish. "
                        f"1h move: {cp_1h:+.1f}%.")
                else:
                    assessment = "NEUTRAL"
                    description = (
                        f"Sentiment neutral (composite: {composite:+.2f}). "
                        f"Fear & Greed: {fear_greed} ({fg_class}, {fg_trend}). "
                        f"Community: {cg_up:.0f}% bullish. "
                        f"1h: {cp_1h:+.1f}%, Vol change: {cp_vol_chg:+.0f}%.")

                return {
                    "available": True,
                    "fear_greed_value": fear_greed,
                    "fear_greed_class": fg_class,
                    "fear_greed_trend": fg_trend,
                    "cg_sentiment_up_pct": round(cg_up, 1),
                    "cp_pct_change_15m": round(cp_15m, 2),
                    "cp_pct_change_1h": round(cp_1h, 2),
                    "cp_volume_change_24h": round(cp_vol_chg, 1),
                    "sentiment_composite": round(composite, 3),
                    "divergence": divergence,
                    "assessment": assessment,
                    "high_buzz": divergence != "NONE",
                    "description": description,
                }
    except Exception:
        pass  # Fall through to legacy path

    # Legacy fallback: news_sentiment_hourly.csv
    news = data.get("news_sentiment")
    if news is None or (isinstance(news, pd.DataFrame) and news.empty):
        return {
            "available": False,
            "description": "No sentiment data. Run: python3 external_collectors.py --source sentiment",
        }

    df = news.copy()
    latest = df.iloc[-1] if len(df) > 0 else {}
    current_buzz = float(latest.get("news_buzz", 0))
    current_sentiment = float(latest.get("news_sentiment", 0))

    sentiment_6h = df["news_sentiment"].tail(6).mean() if len(df) >= 6 else current_sentiment
    sentiment_24h = df["news_sentiment"].tail(24).mean() if len(df) >= 24 else current_sentiment
    sentiment_delta = sentiment_6h - sentiment_24h

    high_buzz = current_buzz > 1.5
    divergence = high_buzz and current_sentiment < -0.2

    if divergence:
        assessment = "DANGER"
    elif current_sentiment > 0.2:
        assessment = "BULLISH_MOMENTUM"
    elif current_sentiment < -0.2:
        assessment = "BEARISH_MOMENTUM"
    else:
        assessment = "NEUTRAL"

    description = (f"Sentiment: {current_sentiment:+.2f} (composite). "
                   f"6h avg: {sentiment_6h:+.2f}, 24h avg: {sentiment_24h:+.2f}.")

    return {
        "available": True,
        "current_sentiment": round(current_sentiment, 3),
        "sentiment_6h": round(sentiment_6h, 3),
        "sentiment_24h": round(sentiment_24h, 3),
        "sentiment_delta": round(sentiment_delta, 3),
        "news_buzz": round(current_buzz, 2),
        "assessment": assessment,
        "high_buzz": high_buzz,
        "divergence": divergence,
        "description": description,
    }


# =========================================================================
# Model 8: On-Chain Flow (Helius)
# =========================================================================

def _project_onchain_flow(data, market_state):
    """
    Assess on-chain signals: holder concentration + exchange flow.

    Key signals:
      - Gini coefficient trend (rising = accumulation by whales)
      - Net exchange flow (positive = withdrawals = bullish)
      - Whale transfer count (sudden spike = position changes)

    Returns dict with on-chain assessment.
    """
    holders = data.get("holder_concentration")
    flow = data.get("exchange_flow")

    if (holders is None or (isinstance(holders, pd.DataFrame) and holders.empty)) and \
       (flow is None or (isinstance(flow, pd.DataFrame) and flow.empty)):
        return {
            "available": False,
            "description": "No on-chain data available. Set HELIUS_API_KEY.",
        }

    result = {"available": True}

    # --- Holder concentration analysis ---
    if holders is not None and isinstance(holders, pd.DataFrame) and not holders.empty:
        latest = holders.iloc[-1]
        result["gini"] = round(float(latest.get("gini", 0)), 4)
        result["top10_pct"] = round(float(latest.get("top10_pct", 0)), 1)
        result["top20_pct"] = round(float(latest.get("top20_pct", 0)), 1)
        result["exchange_held_pct"] = round(float(latest.get("exchange_held_pct", 0)), 1)
        result["total_holders"] = int(latest.get("total_holders", 0))

        # Trend: compare latest to earlier if we have history
        if len(holders) >= 3:
            prev_gini = holders["gini"].iloc[-3] if "gini" in holders.columns else None
            if prev_gini is not None and not pd.isna(prev_gini):
                gini_delta = result["gini"] - float(prev_gini)
                result["gini_trend"] = "CONCENTRATING" if gini_delta > 0.005 else \
                                       "DISTRIBUTING" if gini_delta < -0.005 else "STABLE"
                result["gini_delta"] = round(gini_delta, 4)
            else:
                result["gini_trend"] = "UNKNOWN"
        else:
            result["gini_trend"] = "INSUFFICIENT_DATA"
    else:
        result["gini_trend"] = "NO_DATA"

    # --- Exchange flow analysis ---
    if flow is not None and isinstance(flow, pd.DataFrame) and not flow.empty:
        latest_flow = flow.iloc[-1]
        net_flow = float(latest_flow.get("net_flow_tokens", 0))
        whale_transfers = int(latest_flow.get("whale_transfers", 0))

        result["net_flow_tokens"] = round(net_flow, 0)
        result["whale_transfers"] = whale_transfers
        result["flow_direction"] = "BULLISH" if net_flow > 0 else "BEARISH" if net_flow < 0 else "NEUTRAL"

        # Recent flow trend
        if len(flow) >= 3:
            recent_net = flow["net_flow_tokens"].tail(3).sum()
            result["flow_trend_3_snapshots"] = round(float(recent_net), 0)
        else:
            result["flow_trend_3_snapshots"] = round(net_flow, 0)
    else:
        result["flow_direction"] = "NO_DATA"
        net_flow = 0
        whale_transfers = 0

    # --- Overall assessment ---
    gini_trend = result.get("gini_trend", "UNKNOWN")
    flow_dir = result.get("flow_direction", "UNKNOWN")

    if gini_trend == "CONCENTRATING" and flow_dir == "BEARISH":
        assessment = "WHALE_DUMPING"
        desc = (f"Whales concentrating (Gini trend: {gini_trend}) AND sending to exchanges. "
                f"Net flow: {net_flow:+,.0f} tokens. High dump risk.")
    elif gini_trend == "CONCENTRATING" and flow_dir == "BULLISH":
        assessment = "WHALE_ACCUMULATING"
        desc = (f"Whales accumulating (Gini trend: {gini_trend}) AND withdrawing from exchanges. "
                f"Net flow: {net_flow:+,.0f} tokens. Bullish signal.")
    elif whale_transfers >= 3:
        assessment = "HIGH_WHALE_ACTIVITY"
        desc = (f"{whale_transfers} whale transfers detected. "
                f"Net flow: {net_flow:+,.0f} tokens. Watch for volatility.")
    elif flow_dir == "BEARISH" and abs(net_flow) > 50000:
        assessment = "EXCHANGE_INFLOW"
        desc = (f"Significant tokens moving to exchanges ({net_flow:+,.0f}). "
                f"Potential selling pressure ahead.")
    elif flow_dir == "BULLISH" and abs(net_flow) > 50000:
        assessment = "EXCHANGE_OUTFLOW"
        desc = (f"Tokens being withdrawn from exchanges ({net_flow:+,.0f}). "
                f"Reducing available supply — bullish.")
    else:
        assessment = "NEUTRAL"
        desc = (f"On-chain flow neutral. Gini: {result.get('gini', 'N/A')}, "
                f"Net flow: {net_flow:+,.0f} tokens.")

    result["assessment"] = assessment
    result["description"] = desc

    return result


# =========================================================================
# Model 9: Cross-Exchange Derivatives Intelligence (Coinalyze)
# =========================================================================

def _project_cross_exchange_derivatives(data, market_state):
    """
    Multi-exchange derivatives signals:
      - Funding spread: divergence across exchanges = arb opportunity
      - Predicted funding: what funding WILL be (pre-settlement signal)
      - Liquidation clusters: where are the stops?

    Returns dict with cross-exchange assessment.
    """
    cx_funding = data.get("cross_exchange_funding")
    predicted = data.get("predicted_funding")
    liquidations = data.get("liquidations")

    has_any = any(
        d is not None and isinstance(d, pd.DataFrame) and not d.empty
        for d in [cx_funding, predicted, liquidations]
    )

    if not has_any:
        return {
            "available": False,
            "description": "No Coinalyze data available. Set COINALYZE_API_KEY.",
        }

    result = {"available": True}

    # --- Cross-exchange funding spread ---
    if cx_funding is not None and isinstance(cx_funding, pd.DataFrame) and not cx_funding.empty:
        latest = cx_funding.iloc[-1]
        spread = float(latest.get("funding_spread", 0))
        mean_funding = float(latest.get("mean_funding", 0))

        result["funding_spread"] = round(spread, 6)
        result["mean_funding_cross_ex"] = round(mean_funding, 6)

        # High spread = funding arb opportunity, suggests manipulation
        if spread > 0.001:
            result["funding_arb"] = "HIGH_SPREAD"
            result["funding_arb_desc"] = (
                f"Cross-exchange funding spread: {spread:.4%}. "
                f"Market makers likely arbing this — expect convergence.")
        else:
            result["funding_arb"] = "NORMAL"
    else:
        result["funding_arb"] = "NO_DATA"

    # --- Predicted funding (pre-settlement) ---
    if predicted is not None and isinstance(predicted, pd.DataFrame) and not predicted.empty:
        latest_pred = predicted.iloc[-1]
        mean_pred = float(latest_pred.get("mean_predicted", 0))
        result["predicted_funding"] = round(mean_pred, 6)

        # Compare predicted to current funding
        current_funding = market_state.get("avg_funding", 0)
        funding_shift = mean_pred - current_funding

        if abs(mean_pred) > 0.0005:
            direction = "longs will pay" if mean_pred > 0 else "shorts will pay"
            result["predicted_funding_desc"] = (
                f"Predicted funding: {mean_pred:.4%} ({direction}). "
                f"Shift from current: {funding_shift:+.4%}.")
            if abs(mean_pred) > 0.001:
                result["squeeze_risk"] = "HIGH"
            else:
                result["squeeze_risk"] = "MODERATE"
        else:
            result["predicted_funding_desc"] = "Predicted funding near zero — balanced market."
            result["squeeze_risk"] = "LOW"
    else:
        result["squeeze_risk"] = "NO_DATA"

    # --- Liquidation analysis ---
    if liquidations is not None and isinstance(liquidations, pd.DataFrame) and not liquidations.empty:
        latest_liq = liquidations.iloc[-1]
        liq_zscore = float(latest_liq.get("liq_zscore", 0))
        liq_ratio = float(latest_liq.get("liq_ratio", 0.5))
        total_liq = float(latest_liq.get("total_liquidations", 0))

        result["liq_zscore"] = round(liq_zscore, 2)
        result["liq_ratio_long"] = round(liq_ratio, 3)  # % of liquidations that are longs
        result["total_liquidations"] = round(total_liq, 0)

        if liq_zscore > 2.0:
            result["liq_event"] = "CASCADE"
            result["liq_desc"] = (
                f"Liquidation cascade detected ({liq_zscore:.1f}σ above normal). "
                f"{'Longs' if liq_ratio > 0.6 else 'Shorts'} getting wiped "
                f"({liq_ratio:.0%} long liqs). Forced selling/buying in progress.")
        elif liq_zscore > 1.0:
            result["liq_event"] = "ELEVATED"
            result["liq_desc"] = (
                f"Elevated liquidations ({liq_zscore:.1f}σ). "
                f"Long/short ratio: {liq_ratio:.0%}/{1-liq_ratio:.0%}.")
        else:
            result["liq_event"] = "NORMAL"
            result["liq_desc"] = "Liquidation levels normal."
    else:
        result["liq_event"] = "NO_DATA"

    # --- Overall assessment ---
    squeeze = result.get("squeeze_risk", "NO_DATA")
    liq_event = result.get("liq_event", "NO_DATA")
    arb = result.get("funding_arb", "NO_DATA")

    if squeeze == "HIGH" and liq_event == "CASCADE":
        assessment = "SQUEEZE_IN_PROGRESS"
        desc = "⚠️ Squeeze in progress: extreme predicted funding + liquidation cascade."
    elif squeeze == "HIGH":
        assessment = "SQUEEZE_BUILDING"
        desc = f"Predicted funding extreme ({result.get('predicted_funding', 0):.4%}). Squeeze conditions building."
    elif liq_event == "CASCADE":
        assessment = "LIQUIDATION_CASCADE"
        desc = result.get("liq_desc", "Liquidation cascade detected.")
    elif arb == "HIGH_SPREAD":
        assessment = "FUNDING_ARB"
        desc = result.get("funding_arb_desc", "Cross-exchange funding divergence.")
    else:
        parts = []
        if squeeze != "NO_DATA":
            parts.append(f"Squeeze risk: {squeeze}")
        if liq_event != "NO_DATA":
            parts.append(f"Liquidations: {liq_event}")
        assessment = "NORMAL"
        desc = "Cross-exchange derivatives normal. " + ". ".join(parts)

    result["assessment"] = assessment
    result["description"] = desc

    return result


# =========================================================================
# Public API
# =========================================================================

def compute_projections(data, market_state):
    """
    Run all 6 projection sub-models.

    Args:
        data: dict of DataFrames from signal_engine.load_data()
        market_state: dict from market_state.compute_market_state()

    Returns:
        dict with keys: probability, mean_reversion, manipulation_cycle,
        session_conditional, btc_lead_lag, confidence_intervals, summary
    """
    prob = _project_probabilistic_return(data, market_state)
    mr = _project_mean_reversion(data, market_state)
    cycle = _detect_hourly_manipulation_cycle(data, market_state)
    session = _project_session_conditional(data, market_state)
    btc = _project_btc_lead_lag(data, market_state)
    ci = _compute_confidence_intervals(data, market_state, prob)

    # New external data models
    news = _project_news_sentiment(data, market_state)
    onchain = _project_onchain_flow(data, market_state)
    cx_derivs = _project_cross_exchange_derivatives(data, market_state)

    # Build summary string for Slack briefings
    parts = []
    parts.append(f"*Probability:* {prob['description']}")

    if mr.get("funding"):
        parts.append(f"*Funding Reversion:* {mr['funding']['description']}")
    if mr.get("lsr"):
        parts.append(f"*LSR Reversion:* {mr['lsr']['description']}")

    if cycle["phase"] != "DORMANT":
        parts.append(f"*Manipulation:* {cycle['description']}")

    parts.append(f"*Session Edge:* {session['description']}")
    parts.append(f"*BTC Lead-Lag:* {btc['description']}")

    # External data summaries (only if available)
    if news.get("available"):
        parts.append(f"*News Sentiment:* {news['description']}")
    if onchain.get("available"):
        parts.append(f"*On-Chain Flow:* {onchain['description']}")
    if cx_derivs.get("available"):
        parts.append(f"*Cross-Exchange:* {cx_derivs['description']}")

    if ci.get("h4"):
        h4 = ci["h4"]
        parts.append(
            f"*4h Range:* ${h4['low_68']:.4f} – ${h4['high_68']:.4f} (68%) | "
            f"${h4['low_95']:.4f} – ${h4['high_95']:.4f} (95%)"
        )

    summary = "\n".join(parts)

    return {
        "probability": prob,
        "mean_reversion": mr,
        "manipulation_cycle": cycle,
        "session_conditional": session,
        "btc_lead_lag": btc,
        "confidence_intervals": ci,
        "news_sentiment": news,
        "onchain_flow": onchain,
        "cross_exchange": cx_derivs,
        "summary": summary,
    }
