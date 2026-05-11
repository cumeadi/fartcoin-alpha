"""
Forward Projection Engine — Fartcoin Alpha Framework

Sub-models:
  1. Probabilistic Forward Returns (LightGBM)
  2. Mean Reversion (funding + LSR)
  3. Manipulation Cycle Detector
  4. Session Conditional
  5. BTC Lead-Lag
  6. Confidence Intervals
  7. Ghost Long Detector  ← NEW: Binance/Bybit funding velocity divergence
  8. HMM Regime Switcher  ← NEW: 3-state hidden Markov regime classifier
  9. VPIN Proxy           ← NEW: OI-based informed-flow toxicity

Entry point: compute_projections(data, market_state) -> dict
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from hmm_engine import label_current as _hmm_label_current
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

try:
    from trade_scorer import (build_meta_features, score_live as _score_live_meta,
                               score_live_lstm_raw as _score_live_lstm_raw)
    _SCORER_AVAILABLE = True
except ImportError:
    _SCORER_AVAILABLE = False
    _score_live_lstm_raw = None

try:
    from support_resistance import compute_sr_levels as _compute_sr_levels
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False

try:
    from systematic_signals import compute_settlement_signals as _compute_settlement_signals
    _SYSTEMATIC_AVAILABLE = True
except ImportError:
    _SYSTEMATIC_AVAILABLE = False
    _compute_settlement_signals = None

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
    hourly_bias, funding z-score, BTC regime flag, day-of-week dummies.
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

        # Day-of-week: Thu/Fri bearish bias, Monday bullish bias
        import datetime as _dt
        now_utc = _dt.datetime.utcnow()
        dow = now_utc.weekday()  # 0=Mon ... 6=Sun
        features.append(np.array([1.0 if dow in (3, 4) else 0.0]))  # Thu/Fri
        features.append(np.array([1.0 if dow == 0 else 0.0]))       # Monday

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

    # Day-of-week features: Thu/Fri bearish, Monday bullish
    if hasattr(df.index, 'dayofweek'):
        dow = pd.Series(df.index.dayofweek)
        features.append(dow.isin([3, 4]).astype(float).values)  # Thu/Fri
        features.append((dow == 0).astype(float).values)         # Monday
    else:
        features.extend([np.zeros(len(df))] * 2)

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

    # ── Carry-adjusted label ──────────────────────────────────────────────────
    # Bybit FARTCOIN: 0.5%/8h floor = 0.25% per 4h holding window
    # Round-trip Bybit taker fees + spread ≈ 0.20%
    # Label = 1 only when return exceeds total 4h cost of being long on Bybit
    BYBIT_CARRY_4H  = 0.0025   # 0.25%  — half settlement cycle
    BYBIT_SLIP_RT   = 0.0020   # 0.20%  — round-trip taker fees + spread
    BYBIT_COST_4H   = BYBIT_CARRY_4H + BYBIT_SLIP_RT  # 0.45% total hurdle
    y = (df["fwd_ret_4h"].values > BYBIT_COST_4H).astype(float)

    # ── Extra feature columns on training df ─────────────────────────────────
    # Price momentum
    _px = ohlcv.loc[common_idx, price_col]
    df["mom_1h"] = _px.pct_change(1).reindex(df.index).fillna(0)
    df["mom_4h"] = _px.pct_change(4).reindex(df.index).fillna(0)
    df["mom_8h"] = _px.pct_change(8).reindex(df.index).fillna(0)

    # Realized vol ratio: recent (6h) vs baseline (24h) — pre-move detector
    _ret1h = _px.pct_change(1).reindex(df.index)
    _vol6  = _ret1h.rolling(6, min_periods=3).std()
    _vol24 = _ret1h.rolling(24, min_periods=12).std().replace(0, np.nan)
    df["vol_ratio"] = (_vol6 / _vol24).fillna(1.0).clip(0, 5)

    # Bybit settlement proximity (0:00 / 08:00 / 16:00 UTC)
    def _mins_to_bybit_settle(ts):
        if not hasattr(ts, "hour"):
            return 240
        total_mins = ts.hour * 60 + ts.minute
        for settle_h in [0, 8, 16, 24]:
            diff = settle_h * 60 - total_mins
            if diff > 0:
                return diff
        return 480

    if hasattr(df.index, "hour"):
        _mts = [_mins_to_bybit_settle(ts) for ts in df.index]
        df["near_settle"] = [1.0 if m <= 30 else 0.0 for m in _mts]
        df["post_settle"] = [1.0 if m >= 450 else 0.0 for m in _mts]
    else:
        df["near_settle"] = 0.0
        df["post_settle"] = 0.0

    # Additional signal features (all already computed by signal_engine)
    for _sig in ["sig_oi_accel", "sig_lsr", "sig_taker", "sig_oi_divergence"]:
        if _sig in signals.columns:
            df[_sig] = signals.loc[common_idx, _sig].reindex(df.index).fillna(0)
        else:
            df[_sig] = 0.0

    EXTRA_COLS = [
        "mom_1h", "mom_4h", "mom_8h", "vol_ratio",
        "near_settle", "post_settle",
        "sig_oi_accel", "sig_lsr", "sig_taker", "sig_oi_divergence",
    ]

    # Drop rows where key extras are NaN
    df = df.dropna(subset=["fwd_ret_4h", "composite", "vol_ratio"])
    y  = (df["fwd_ret_4h"].values > BYBIT_COST_4H).astype(float)  # recompute after dropna

    # ── Build full feature matrix ─────────────────────────────────────────────
    X_base  = _build_feature_matrix(df)
    X_extra = df[EXTRA_COLS].fillna(0).values
    X       = np.hstack([X_base, X_extra])

    # ── Prediction vector — current extra feature values ──────────────────────
    x_now_base = _build_feature_matrix(df.iloc[-1:], for_prediction=True, state=state)

    # Momentum from latest OHLCV
    _cur_mom1h = float(_px.pct_change(1).iloc[-1]) if len(_px) > 1 else 0.0
    _cur_mom4h = float(_px.pct_change(4).iloc[-1]) if len(_px) > 4 else 0.0
    _cur_mom8h = float(_px.pct_change(8).iloc[-1]) if len(_px) > 8 else 0.0
    _cur_vol_r = float((_vol6.iloc[-1] / _vol24.iloc[-1])
                       if not np.isnan(_vol6.iloc[-1]) and not np.isnan(_vol24.iloc[-1])
                       and _vol24.iloc[-1] > 0 else 1.0)
    _cur_vol_r = min(max(_cur_vol_r, 0), 5)

    # Settlement proximity from current UTC time
    import datetime as _dt
    _now = _dt.datetime.utcnow()
    _cur_mts = _mins_to_bybit_settle(_now)
    _cur_near_settle = 1.0 if _cur_mts <= 30 else 0.0
    _cur_post_settle = 1.0 if _cur_mts >= 450 else 0.0

    # Signal values from latest row
    def _latest_sig(col):
        if col in signals.columns:
            v = signals[col].dropna()
            return float(v.iloc[-1]) if len(v) > 0 else 0.0
        return 0.0

    x_extra_now = np.array([[
        _cur_mom1h, _cur_mom4h, _cur_mom8h, _cur_vol_r,
        _cur_near_settle, _cur_post_settle,
        _latest_sig("sig_oi_accel"), _latest_sig("sig_lsr"),
        _latest_sig("sig_taker"), _latest_sig("sig_oi_divergence"),
    ]])
    x_now = np.hstack([x_now_base, x_extra_now])

    # ── Fit model ─────────────────────────────────────────────────────────────
    if _LGBM_AVAILABLE:
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=5,
            num_leaves=20,
            min_child_samples=max(10, len(df) // 30),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        try:
            clf.fit(X, y)
            prob = float(clf.predict_proba(x_now)[0][1])
            feat_imp = dict(enumerate(clf.feature_importances_.tolist()))
        except Exception:
            return result
        model_label = "LightGBM-v2"
    else:
        # Fallback: logistic regression via L-BFGS-B (base features only)
        theta0 = np.zeros(X_base.shape[1])
        try:
            res = minimize(_neg_log_likelihood, theta0, args=(X_base, y),
                           method="L-BFGS-B", options={"maxiter": 300})
            theta = res.x
        except Exception:
            return result
        prob = float(_logistic(x_now_base @ theta).ravel()[0])
        feat_imp = {}
        model_label = "LogReg"

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

    # Session-direction filters removed — backtest showed they were overfit
    session_warning = ""

    # Momentum override: if price has already moved >3% in last 4h, don't fade it
    # This week's lesson: composite said LONG during -5% to -7% crashes
    recent_ret_4h = ohlcv[price_col].pct_change(4).iloc[-1] if len(ohlcv) > 4 else 0
    if not pd.isna(recent_ret_4h):
        if recent_ret_4h < -0.03 and prob > 0.5:
            # Price crashed >3% but model says LONG — override to neutral
            prob = 0.5
            expected_move = 0
            session_warning = " (MOMENTUM OVERRIDE: price dropped >3% in 4h — do not fade)"
        elif recent_ret_4h > 0.03 and prob < 0.5:
            # Price rallied >3% but model says SHORT — override to neutral
            prob = 0.5
            expected_move = 0
            session_warning = " (MOMENTUM OVERRIDE: price rallied >3% in 4h — do not fade)"

    # ── Bybit-calibrated thresholds ───────────────────────────────────────────
    # Bybit fixed funding floor = +0.50%/8h = +1.50%/day = +0.25%/4h carry
    # Round-trip slippage est = 0.20% → total 4h hurdle = 0.45%
    # Backtested sweet spot: model prob ≥ 55% = win rate 71%, Sharpe 8.19
    # Real break-even probability = 61% (when carry adjusted out)
    BYBIT_ENTRY_THRESHOLD = 0.55   # minimum for a trade entry signal
    BYBIT_BREAK_EVEN      = 0.61   # carry-adjusted break-even probability
    above_bybit_threshold = prob >= BYBIT_ENTRY_THRESHOLD

    # Conviction label — Bybit-aware
    if prob >= BYBIT_BREAK_EVEN:
        conviction = "HIGH"            # >61% — genuinely above carry cost
    elif prob >= BYBIT_ENTRY_THRESHOLD:
        conviction = "MODERATE"        # 55-61% — above entry threshold, below break-even
    elif prob <= (1 - BYBIT_BREAK_EVEN):   # ≤39%
        conviction = "HIGH (bearish)"
    elif prob <= (1 - BYBIT_ENTRY_THRESHOLD):   # ≤45%
        conviction = "MODERATE (bearish)"
    else:
        conviction = "LOW"             # 45-55% — no edge after carry

    # Entry recommendation
    if prob >= BYBIT_ENTRY_THRESHOLD:
        entry_rec = "ENTER LONG ✅"
    elif prob <= (1 - BYBIT_ENTRY_THRESHOLD):
        entry_rec = "ENTER SHORT / REDUCE ⚠️"
    else:
        entry_rec = "NO TRADE — below threshold 🚫"

    n_features = X.shape[1] if "X" in dir() else len(x_now.ravel())

    result.update({
        "prob_positive_4h": round(prob, 4),
        "expected_move_pct": round(expected_move, 2),
        "model_n_train": len(df),
        "conviction": conviction,
        "entry_recommendation": entry_rec,
        "above_bybit_threshold": above_bybit_threshold,
        "bybit_entry_threshold": BYBIT_ENTRY_THRESHOLD,
        "bybit_break_even": BYBIT_BREAK_EVEN,
        "bybit_carry_4h_pct": round(BYBIT_COST_4H * 100, 3),
        "description": (
            f"{prob:.0%} prob of +4h return. Expected move: {expected_move:+.2f}%. "
            f"Conviction: {conviction}. "
            f"Min entry: {BYBIT_ENTRY_THRESHOLD:.0%} | Break-even after carry: {BYBIT_BREAK_EVEN:.0%} | "
            f"Carry cost: {BYBIT_COST_4H*100:.2f}%/4h. "
            f"{entry_rec}. "
            f"(Model: {model_label}, {n_features} features, {len(df)} obs)"
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
                f"Funding expected to normalize in ~{ar1['half_life_h']:.1f}h. "
                f"Current: {current_fr:.6f} (real: {real_funding:.4f}). "
                f"{'Expected to reach neutral in ' + str(cross_time) + 'h.' if cross_time else 'Near normal levels.'}"
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
        # Live calibration (Apr 2026): p84 confirmed 8h mean reversion at 93% hit rate.
        # Thresholds lowered from p90/p10 to p80/p20 to catch extremes earlier.
        if percentile > 0.85:
            extremity = f"Longs extremely heavy (top {(1-percentile)*100:.0f}% of history)"
            lsr_action = (
                f"⚠ LONG UNWIND RISK — Longs are at {percentile:.0%} percentile. "
                f"Historically, longs get forced out within ~{avg_revert_time:.0f}h "
                f"({revert_rate:.0%} of the time). Lean short."
            )
        elif percentile > 0.75:
            extremity = f"Longs heavy ({percentile:.0%} percentile)"
            lsr_action = (
                f"Longs are elevated. Pullback risk within ~{avg_revert_time:.0f}h "
                f"({revert_rate:.0%} hit rate). Avoid adding longs."
            )
        elif percentile < 0.15:
            extremity = f"Shorts extremely heavy (bottom {percentile*100:.0f}% of history)"
            lsr_action = (
                f"⚠ SHORT SQUEEZE RISK — Shorts are at {percentile:.0%} percentile. "
                f"Shorts tend to get squeezed within ~{avg_revert_time:.0f}h ({revert_rate:.0%} of the time)."
            )
        elif percentile < 0.25:
            extremity = f"Shorts heavy ({percentile:.0%} percentile)"
            lsr_action = (
                f"Shorts are elevated. Squeeze risk within ~{avg_revert_time:.0f}h "
                f"({revert_rate:.0%} hit rate). Avoid adding shorts."
            )
        else:
            extremity = f"Normal range ({percentile:.0%} percentile)"
            lsr_action = ""

        result["lsr"] = {
            "current": round(current_lsr, 4),
            "median": round(median_lsr, 4),
            "percentile": round(percentile, 4),
            "avg_revert_time_h": round(avg_revert_time, 1),
            "revert_rate": round(revert_rate, 2),
            "projected_cross_time_h": cross_time,
            "projected_path": [round(v, 4) for v in path],
            "lsr_action": lsr_action,
            "description": (
                f"LSR: {current_lsr:.4f} — {extremity}. "
                f"Median: {median_lsr:.4f}. "
                f"Avg reversion time from extremes: {avg_revert_time:.0f}h "
                f"(revert rate: {revert_rate:.0%}). "
                f"{'Expected to reach neutral in ' + str(cross_time) + 'h.' if cross_time else 'Near normal levels.'}"
                + (f" {lsr_action}" if lsr_action else "")
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

    # Phase: SPIKE_IN_PROGRESS — this is an EXIT signal, not entry
    # Backtest: SPIKE events averaged -5.32% this week (0% hit rate)
    # The manufactured move is ENDING — distribution is happening
    if last_vol_ratio > 1.5 and abs(last_oi_4h_change) > 0.05:
        hours_in = sum(1 for v in recent_vol_ratios.values[-6:] if v > 1.3)
        result = {
            "phase": "SPIKE_IN_PROGRESS",
            "description": (
                f"EXIT SIGNAL — Spike phase detected. Volume {last_vol_ratio:.1f}x normal, "
                f"OI changed {last_oi_4h_change:+.1%} in 4h. "
                f"This is distribution/exhaustion — the move is ending, not starting. "
                f"Close positions or tighten stops. Do NOT enter new trades."
            ),
            "hours_in_phase": hours_in,
            "est_hours_to_move": 0,
            "confidence": 0.85,
            "action": "EXIT",
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

    # Day-of-week — always compute so it's available even on early return
    import datetime as _dt_sc
    _dow_now = _dt_sc.datetime.utcnow().weekday()
    _DOW_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                  4: "Friday", 5: "Saturday", 6: "Sunday"}
    _day_name = _DOW_NAMES.get(_dow_now, "")
    _day_bias = "bearish" if _dow_now in (3, 4) else ("bullish" if _dow_now == 0 else "neutral")

    result = {
        "session": session,
        "direction": direction,
        "conditional_avg_return_pct": 0.0,
        "session_bias_4h_bps": 0.0,
        "combined_edge_pct": 0.0,
        "n_samples": 0,
        "day_of_week": _day_name,
        "day_bias": _day_bias,
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

    # Volume context: thin market reduces signal quality and widens slippage
    # Live calibration: -37% volume confirmed wider spreads and less reliable signals
    vol_warning = ""
    volume_pct_of_avg = 1.0
    if ohlcv is not None and "volume" in ohlcv.columns and len(ohlcv) >= 24:
        try:
            _vol24_avg = ohlcv["volume"].iloc[-24:].mean()
            _vol_cur = ohlcv["volume"].iloc[-1]
            if _vol24_avg > 0:
                volume_pct_of_avg = float(_vol_cur / _vol24_avg)
                if volume_pct_of_avg < 0.50:
                    vol_warning = f" ⚠ VERY THIN VOLUME ({volume_pct_of_avg:.0%} of 24h avg) — reduce position size 40%+, expect wide spreads."
                elif volume_pct_of_avg < 0.70:
                    vol_warning = f" ⚠ THIN VOLUME ({volume_pct_of_avg:.0%} of 24h avg) — reduce position size 25-30%."
        except Exception:
            pass

    # Session-direction quality assessment from backtest evidence
    quality = "NEUTRAL"
    warning = ""

    # Quality based on this observation's edge strength, not hard-coded session labels
    # (hard-coded labels were overfit — inverted in live trading)
    warning = ""
    if combined_edge > 0.3 and n >= 30:
        quality = "STRONG"
    elif combined_edge > 0.1:
        quality = "FAVORABLE"
    elif combined_edge < -0.2:
        quality = "WEAK"
    else:
        quality = "NEUTRAL"

    # Downgrade quality one tier when volume is thin (signals are noisier)
    if volume_pct_of_avg < 0.70 and quality == "STRONG":
        quality = "FAVORABLE"
    elif volume_pct_of_avg < 0.70 and quality == "FAVORABLE":
        quality = "NEUTRAL"

    # Day-of-week note for description
    day_note = ""
    if _dow_now in (3, 4):
        day_note = f" ⚠ {_day_name} has bearish bias historically."
    elif _dow_now == 0:
        day_note = f" {_day_name} has bullish bias historically."

    result.update({
        "conditional_avg_return_pct": round(cond_avg, 3),
        "session_bias_4h_bps": round(bias_sum, 1),
        "combined_edge_pct": round(combined_edge, 3),
        "n_samples": n,
        "quality": quality,
        "volume_pct_of_avg": round(volume_pct_of_avg, 2),
        "day_of_week": _day_name,
        "day_bias": _day_bias,
        "description": (
            f"{direction} during {session}: avg 4h return {cond_avg:+.2f}% "
            f"(n={n}). Session bias next 4h: {bias_sum:+.0f} bps. "
            f"Combined edge: {combined_edge:+.2f}%. "
            f"Quality: {quality}.{day_note}{warning}{vol_warning}"
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

    # Volume adjustment: thin markets → widen bands (lower reliability)
    # Live calibration: -37% volume = bands widened 20%, center less reliable
    vol_24h_avg = ohlcv["volume"].iloc[-24:].mean() if "volume" in ohlcv.columns and len(ohlcv) >= 24 else None
    vol_current = ohlcv["volume"].iloc[-1] if "volume" in ohlcv.columns and len(ohlcv) > 0 else None
    volume_ratio = 1.0
    thin_volume = False
    if vol_24h_avg is not None and vol_current is not None and vol_24h_avg > 0:
        volume_ratio = float(vol_current / vol_24h_avg)
        thin_volume = volume_ratio < 0.65  # <65% of 24h avg = thin

    # When volume is thin: widen confidence bands (market more erratic)
    vol_expansion = 1.0
    if volume_ratio < 0.65:
        vol_expansion = 1.25   # 25% wider bands
    elif volume_ratio < 0.80:
        vol_expansion = 1.12   # 12% wider

    # Point estimate direction from probability model
    expected_move_pct = prob_result.get("expected_move_pct", 0) / 100

    for horizon, label in [(4, "h4"), (8, "h8")]:
        vol_h = hourly_vol * np.sqrt(horizon) * vol_expansion
        center = current_price * (1 + expected_move_pct * (horizon / 4))

        result[label] = {
            "horizon_hours": horizon,
            "center": round(center, 6),
            "current_price": round(current_price, 6),
            "vol_pct": round(vol_h * 100, 2),
            "volume_ratio": round(volume_ratio, 2),
            "thin_volume": thin_volume,
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

        # Bybit FARTCOIN has a fixed funding floor of +0.5% per 8h.
        # Thresholds must be calibrated above that floor to be meaningful.
        # MODERATE = above floor, HIGH = genuinely extreme for Bybit perps.
        BYBIT_FUNDING_FLOOR = 0.005   # 0.5% — Bybit minimum for FARTCOIN
        SQUEEZE_HIGH_THRESHOLD = 0.008   # >0.8% is truly elevated above floor
        SQUEEZE_MOD_THRESHOLD  = 0.006   # >0.6% is modestly elevated

        if abs(mean_pred) > BYBIT_FUNDING_FLOOR * 0.5:
            direction = "longs will pay" if mean_pred > 0 else "shorts will pay"
            result["predicted_funding_desc"] = (
                f"Predicted funding: {mean_pred:.4%} ({direction}). "
                f"Shift from current: {funding_shift:+.4%}. "
                f"[Bybit floor: +0.50% — readings at floor are not signals]")
            if abs(mean_pred) > SQUEEZE_HIGH_THRESHOLD:
                result["squeeze_risk"] = "HIGH"
            elif abs(mean_pred) > SQUEEZE_MOD_THRESHOLD:
                result["squeeze_risk"] = "MODERATE"
            else:
                result["squeeze_risk"] = "LOW"   # at/near floor — not a signal
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
# 10. Coinglass OI Momentum + Funding Spread
# =========================================================================

def _project_coinglass_oi_funding(data, state):
    """
    Derive signals from Coinglass real-time OI momentum and cross-exchange
    funding spread.

    BACKTEST-CALIBRATED SIGNAL INTERPRETATIONS (90 days, 2128 obs):
      - OI spike (>2% in 1h): historically BEARISH (-0.12% to -0.25% avg 4h, 44-47% hit)
        → Not a trend-follow signal. High OI growth = crowded setup prone to reversal.
      - OI flat + price falling: BEST BUY signal (+0.96% avg 4h, 70% hit)
        → Spot-driven drop with no new leveraged shorts = temporary dislocation
      - OI rising quietly (PASSIVE_ACCUM): mildly bullish (+0.36%, 55% hit)
        → Steady accumulation without price chasing is the healthy version
      - OI + price both surging (TREND_CHASE): bearish (-0.25%, 44% hit)
        → Crowded, exhaustion risk
      - Funding extreme short (<p5=-0.87%): contrarian long (+0.72%, 58% hit)
      - Funding extreme long (>p95=+0.86%): contrarian short (-0.51%, 47% hit)
      - Funding spread > 1% across exchanges → arb/manipulation risk
      - Settlement within 30min → price pinning or spike risk
    """
    result = {
        "available": False,
        "assessment": "NO_DATA",
        "description": "Coinglass data not available.",
    }

    cg_oi = data.get("coinglass_oi")
    cg_fund = data.get("coinglass_funding")

    if (cg_oi is None or cg_oi.empty) and (cg_fund is None or cg_fund.empty):
        return result

    parts = []
    oi_assessment = "NORMAL"
    fund_assessment = "NORMAL"

    # --- OI Momentum ---
    if cg_oi is not None and not cg_oi.empty:
        latest_oi = cg_oi.iloc[-1]

        oi_usd       = float(latest_oi.get("oi_usd", 0))
        m5_oi        = float(latest_oi.get("m5_oi_chg", 0))
        m15_oi       = float(latest_oi.get("m15_oi_chg", 0))
        h1_oi        = float(latest_oi.get("h1_oi_chg", 0))
        h4_oi        = float(latest_oi.get("h4_oi_chg", 0))
        oi_vol_ratio = float(latest_oi.get("oi_vol_ratio", 0))
        oi_spike     = str(latest_oi.get("oi_spike", "NORMAL"))
        oi_div       = str(latest_oi.get("oi_vol_divergence", "NORMAL"))
        leverage_flag = str(latest_oi.get("leverage_flag", "NORMAL"))
        oi_direction  = str(latest_oi.get("direction_flag", "NEUTRAL"))
        avg_funding   = float(latest_oi.get("avg_funding", 0)) * 100  # to pct

        # --- OI/Price Divergence: best signal from backtest ---
        # Requires current price data to compare OI vs price direction
        ohlcv = data.get("ohlcv")
        price_col = "price" if ohlcv is not None and "price" in ohlcv.columns else "close"
        price_1h_chg = 0.0
        if ohlcv is not None and len(ohlcv) > 1:
            try:
                price_1h_chg = float(ohlcv[price_col].pct_change(1).iloc[-1] * 100)
            except Exception:
                price_1h_chg = 0.0

        # Also compute 4h price change for divergence detection
        price_4h_chg = 0.0
        if ohlcv is not None and len(ohlcv) > 4:
            try:
                price_4h_chg = float(ohlcv[price_col].pct_change(4).iloc[-1] * 100)
            except Exception:
                price_4h_chg = 0.0

        oi_price_divergence = "NORMAL"
        if abs(h1_oi) < 0.5 and price_1h_chg < -1.0:
            # BEST BUY: OI flat + price falling → spot-driven dip, not leveraged selling
            # Historically: +0.96% avg 4h, 70% hit rate
            oi_price_divergence = "SPOT_DIP_BUY"
            oi_assessment = "OI_PRICE_DIV_LONG"
            parts.append(
                f"⭐ OI flat ({h1_oi:+.1f}%/1h) + price down {price_1h_chg:+.1f}% "
                f"→ SPOT DIP (historically +0.96% avg 4h, 70% hit rate)"
            )
        elif h4_oi > 3.0 and price_4h_chg < -0.5:
            # NEW (calibrated Apr 2026): OI building over 4h while price declining.
            # Longs are adding into a falling price — classic exhaustion/distribution fingerprint.
            # Not quite TREND_CHASE (price isn't surging), but worse — longs being trapped.
            # Historical edge: -0.25% avg 4h, 44% hit (similar to trend-chase, structural weakness)
            oi_price_divergence = "OI_BUILDING_PRICE_WEAK"
            oi_assessment = "OI_BUILDING_PRICE_WEAK"
            parts.append(
                f"🔴 OI building +{h4_oi:.1f}%/4h while price {price_4h_chg:+.1f}% — "
                f"longs adding into weakness (exhaustion fingerprint, historically -0.25% avg 4h, 44% hit)"
            )
        elif h1_oi > 2.0 and abs(price_1h_chg) < 0.5:
            # PASSIVE ACCUM: OI building quietly, price not reacting yet
            # Historically: +0.36% avg 4h, 55% hit rate
            oi_price_divergence = "PASSIVE_ACCUM"
            oi_assessment = "PASSIVE_ACCUM"
            parts.append(
                f"OI rising {h1_oi:+.1f}%/1h quietly — passive accumulation "
                f"(historically +0.36% avg 4h, 55% hit)"
            )
        elif h1_oi > 2.0 and price_1h_chg > 0.5:
            # TREND CHASE: both OI and price rising fast — exhaustion risk
            # Historically: -0.25% avg 4h, 44% hit
            oi_price_divergence = "TREND_CHASE"
            oi_assessment = "OI_TREND_CHASE_BEARISH"
            parts.append(
                f"⚠ OI {h1_oi:+.1f}%/1h + price {price_1h_chg:+.1f}% — trend chase "
                f"(historically -0.25% avg 4h, 44% hit — exhaustion risk)"
            )
        elif oi_spike in ("SPIKE_5M", "SPIKE_15M"):
            # OI SPIKE without price context: historically bearish (-0.12% to -0.14%, 46-47% hit)
            oi_assessment = "OI_SPIKE_CAUTION"
            parts.append(
                f"⚠ OI spike: {m5_oi:+.1f}% in 5m / {m15_oi:+.1f}% in 15m "
                f"— historically bearish (-0.14% avg 4h, 46% hit)"
            )
        elif oi_spike == "SURGE_1H" or abs(h1_oi) > 5:
            # Rapid OI surge: also historically a fade signal
            oi_assessment = "OI_SURGE_CAUTION"
            parts.append(
                f"⚠ OI surge {h1_oi:+.1f}%/1h — rapid leverage buildup "
                f"(historically bearish, 44-47% hit)"
            )
        elif abs(h4_oi) > 5:
            oi_assessment = "OI_BUILDING"
            parts.append(f"OI building {h4_oi:+.1f}% over 4h (monitor for reversal)")

        result["oi_price_divergence"] = oi_price_divergence
        result["price_1h_chg"] = round(price_1h_chg, 3)

        if oi_div == "DELEVERAGE":
            parts.append("OI dropping + volume spike (forced unwind — potential volatility)")

        if leverage_flag == "HIGH":
            parts.append(f"OI/Vol ratio {oi_vol_ratio:.2f} — highly leveraged (fragile)")
        elif leverage_flag == "ELEVATED":
            parts.append(f"OI/Vol ratio {oi_vol_ratio:.2f} — elevated leverage")

        result["oi_usd"] = round(oi_usd / 1e6, 1)
        result["m5_oi_chg"] = m5_oi
        result["m15_oi_chg"] = m15_oi
        result["h1_oi_chg"] = h1_oi
        result["h4_oi_chg"] = h4_oi
        result["oi_vol_ratio"] = oi_vol_ratio
        result["oi_direction"] = oi_direction
        result["oi_assessment"] = oi_assessment
        result["avg_funding_pct"] = round(avg_funding, 4)

    # --- Funding Spread ---
    if cg_fund is not None and not cg_fund.empty:
        latest_f = cg_fund.iloc[-1]

        mean_rate       = float(latest_f.get("mean_rate_pct", 0))
        max_rate        = float(latest_f.get("max_rate_pct", 0))
        min_rate        = float(latest_f.get("min_rate_pct", 0))
        spread          = float(latest_f.get("spread_pct", 0))
        mean_predicted  = float(latest_f.get("mean_predicted_pct", 0))
        pred_delta      = float(latest_f.get("pred_vs_current_delta", 0))
        mins_to_settle  = float(latest_f.get("min_mins_to_settle", 999))
        fund_extreme    = str(latest_f.get("funding_extreme", "NORMAL"))
        fund_divergence = str(latest_f.get("funding_divergence", "NORMAL"))
        predicted_shift = str(latest_f.get("predicted_shift", "STABLE"))
        settlement_imm  = bool(latest_f.get("settlement_imminent", False))

        binance_rate   = float(latest_f.get("binance_rate", 0))
        bybit_rate     = float(latest_f.get("bybit_rate", 0))
        okx_rate       = float(latest_f.get("okx_rate", 0))
        hl_rate        = float(latest_f.get("hyperliquid_rate", 0))

        if fund_extreme in ("EXTREME_LONG", "HIGH_LONG"):
            fund_assessment = fund_extreme
            parts.append(
                f"Funding extreme: mean {mean_rate:+.3f}% "
                f"(Binance {binance_rate:+.3f}% / Bybit {bybit_rate:+.3f}% / OKX {okx_rate:+.3f}%)"
            )
        elif fund_extreme == "EXTREME_SHORT":
            # Only flag as actionable if a primary traded exchange (Bybit/Binance/OKX) is negative.
            # The min_rate can be dragged down by obscure venues not relevant to our execution.
            _primary_negative = bybit_rate < -0.1 or binance_rate < -0.1 or okx_rate < -0.1
            if _primary_negative:
                fund_assessment = "EXTREME_SHORT"
                _neg_exchanges = ", ".join(
                    e for e, r in [("Bybit", bybit_rate), ("Binance", binance_rate), ("OKX", okx_rate)]
                    if r < -0.1
                )
                parts.append(
                    f"Funding negative on primary exchange(s): {_neg_exchanges} "
                    f"(mean {mean_rate:+.3f}%) — shorts are heavy, upward pressure likely"
                )
            else:
                # Negative rate is on a minor venue only — not actionable for Bybit traders
                parts.append(
                    f"Funding spread: min {min_rate:+.3f}% (minor venue) vs "
                    f"Bybit {bybit_rate:+.3f}% / Binance {binance_rate:+.3f}% — "
                    f"cross-exchange arb only, not a primary market signal"
                )
                fund_assessment = "FUNDING_SPREAD"

        if fund_divergence in ("HIGH_SPREAD", "ELEVATED_SPREAD"):
            parts.append(
                f"Cross-exchange spread {spread:.3f}% "
                f"(max: {max_rate:+.3f}% / min: {min_rate:+.3f}%) — cross-exchange gap, possible manipulation"
            )
            if fund_assessment == "NORMAL":
                fund_assessment = "FUNDING_SPREAD"

        if predicted_shift == "RATE_RISING":
            parts.append(f"Predicted funding rising to {mean_predicted:+.3f}% (Δ{pred_delta:+.3f}%)")
        elif predicted_shift == "RATE_FALLING":
            parts.append(f"Predicted funding falling to {mean_predicted:+.3f}% (Δ{pred_delta:+.3f}%)")

        if settlement_imm:
            parts.append(f"⚡ Settlement in {mins_to_settle:.0f}min — price may pin or spike at settlement")
            if fund_assessment == "NORMAL":
                fund_assessment = "SETTLEMENT_IMMINENT"

        result["mean_rate_pct"]      = mean_rate
        result["spread_pct"]         = spread
        result["mean_predicted_pct"] = mean_predicted
        result["pred_delta"]         = pred_delta
        result["mins_to_settle"]     = mins_to_settle
        result["settlement_imminent"] = settlement_imm
        result["fund_assessment"]    = fund_assessment
        result["predicted_shift"]    = predicted_shift
        result["bybit_rate"]         = bybit_rate
        result["binance_rate"]       = binance_rate

        # --- Bybit carry cost context ---
        # Bybit FARTCOIN fixed floor = +0.5%/8h = 1.5%/day = 10.5%/week
        # Any long must overcome this carry to be profitable
        bybit_daily_carry = bybit_rate * 3   # 3 settlements per day
        result["bybit_daily_carry_pct"] = round(bybit_daily_carry, 4)
        if bybit_rate >= 0.005:
            parts.append(
                f"⚠ Bybit carry: {bybit_rate:+.3f}%/8h = {bybit_daily_carry:+.2f}%/day "
                f"({bybit_daily_carry * 7:+.1f}%/wk) — longs must gain >{bybit_daily_carry:.2f}%/day just to break even"
            )

        # --- Binance/Bybit divergence signal (NEW) ---
        # When Binance funding diverges significantly from Bybit's floor:
        #   Binance << Bybit floor → Binance traders are bearish (informed flow)
        #   Binance >> Bybit floor → Binance traders are very long (crowded)
        binance_vs_bybit = binance_rate - bybit_rate
        result["binance_vs_bybit_spread"] = round(binance_vs_bybit, 4)

        BYBIT_FLOOR = 0.5   # pct
        if binance_rate < (BYBIT_FLOOR - 0.3):
            # Binance is materially below Bybit floor — informed money is bearish
            divergence_signal = "BINANCE_BEARISH_VS_BYBIT"
            parts.append(
                f"⚠ Binance/Bybit divergence: Binance {binance_rate:+.3f}% vs Bybit {bybit_rate:+.3f}% "
                f"(Δ{binance_vs_bybit:+.3f}%) — Binance traders are positioned bearish while Bybit longs pay high carry"
            )
            result["binance_bybit_divergence"] = divergence_signal
        elif binance_rate > (BYBIT_FLOOR + 0.5):
            # Binance is materially above Bybit floor — both crowded long
            divergence_signal = "BOTH_CROWDED_LONG"
            parts.append(
                f"Both Binance ({binance_rate:+.3f}%) and Bybit ({bybit_rate:+.3f}%) elevated "
                f"— longs are heavy on both exchanges, reversal risk elevated"
            )
            result["binance_bybit_divergence"] = divergence_signal
        else:
            result["binance_bybit_divergence"] = "NORMAL"

    # Overall assessment (priority: actionable signals first)
    # Pull Binance/Bybit divergence if computed
    _bb_div = result.get("binance_bybit_divergence", "NORMAL")

    # Positive signals
    if oi_assessment == "OI_PRICE_DIV_LONG":
        overall = "OI_PRICE_DIV_LONG"          # best buy signal (70% hit)
    elif oi_assessment == "PASSIVE_ACCUM":
        overall = "PASSIVE_ACCUM"               # steady accumulation (55% hit)
    elif fund_assessment == "EXTREME_SHORT":
        overall = "EXTREME_SHORT_FUNDING"       # contrarian long — primary exchange confirmed
    # Negative / caution signals
    elif _bb_div == "BINANCE_BEARISH_VS_BYBIT":
        overall = "BINANCE_BEARISH_VS_BYBIT"    # informed money vs Bybit longs — bearish edge
    elif oi_assessment in ("OI_SPIKE_CAUTION", "OI_SURGE_CAUTION", "OI_TREND_CHASE_BEARISH"):
        overall = oi_assessment
    elif fund_assessment in ("EXTREME_LONG", "HIGH_LONG") or _bb_div == "BOTH_CROWDED_LONG":
        overall = fund_assessment if fund_assessment in ("EXTREME_LONG", "HIGH_LONG") \
                  else "BOTH_CROWDED_LONG"      # contrarian bearish
    # Structural alerts
    elif fund_assessment == "SETTLEMENT_IMMINENT":
        overall = "SETTLEMENT_IMMINENT"
    elif oi_assessment == "OI_BUILDING":
        overall = "OI_BUILDING"
    elif fund_assessment == "FUNDING_SPREAD":
        overall = "FUNDING_SPREAD"
    else:
        overall = "NORMAL"

    description = " | ".join(parts) if parts else "OI and funding within normal ranges."

    result.update({
        "available": True,
        "assessment": overall,
        "description": description,
    })
    return result


# =========================================================================
# Model 11: Funding Settlement Cycle Analyzer
# =========================================================================

def _project_funding_settlement_cycle(data, state):
    """
    Analyze funding settlement patterns at 00:00, 08:00, 16:00 UTC.

    Key structural behavior:
    - Positive funding + pre-settlement: longs close to avoid paying → price dips
    - Positive funding + post-settlement: pressure releases → bounce likely
    - Negative funding + pre-settlement: shorts close to avoid paying → price spikes
    - Negative funding + post-settlement: pressure releases → pullback likely

    Uses historical OHLCV to compute mean pre/post returns by funding sign.
    """
    import datetime as _dt

    SETTLEMENT_HOURS = [0, 8, 16]
    PRE_WINDOW = 1   # candles before settlement (1h resolution = 1 candle = 1h before)
    POST_WINDOW = 2  # candles after settlement to check effect

    result = {
        "mins_to_settlement": None,
        "next_settlement_utc": None,
        "phase": "MID_CYCLE",
        "current_funding_sign": "NEUTRAL",
        "pre_ret_mean": None,
        "post_ret_mean": None,
        "historical_n": 0,
        "expected_effect": "UNKNOWN",
        "confidence": 0.0,
        "description": "Insufficient data for settlement cycle analysis.",
    }

    ohlcv = data.get("ohlcv")
    funding_df = data.get("funding")

    if ohlcv is None or len(ohlcv) < 48:
        return result

    price_col = "price" if "price" in ohlcv.columns else "close"

    # --- Time to next settlement ---
    now_utc = _dt.datetime.utcnow()
    current_hour = now_utc.hour
    current_min = now_utc.minute

    # Find next settlement hour
    upcoming = [h for h in SETTLEMENT_HOURS if h > current_hour]
    next_settle_h = upcoming[0] if upcoming else SETTLEMENT_HOURS[0] + 24
    today = now_utc.replace(minute=0, second=0, microsecond=0)
    if upcoming:
        next_settle_dt = today.replace(hour=next_settle_h)
    else:
        next_settle_dt = (today + _dt.timedelta(days=1)).replace(hour=SETTLEMENT_HOURS[0])

    mins_to_settle = (next_settle_dt - now_utc).total_seconds() / 60
    result["mins_to_settlement"] = round(mins_to_settle, 0)
    result["next_settlement_utc"] = next_settle_dt.strftime("%H:%M UTC")

    # Phase classification
    if mins_to_settle <= 30:
        phase = "PRE_SETTLEMENT"
    elif mins_to_settle >= (8 * 60 - 60):
        phase = "JUST_SETTLED"
    else:
        phase = "MID_CYCLE"
    result["phase"] = phase

    # --- Current funding sign ---
    avg_funding = state.get("avg_funding", 0)
    if avg_funding > 0.0001:
        funding_sign = "POSITIVE"
    elif avg_funding < -0.0001:
        funding_sign = "NEGATIVE"
    else:
        funding_sign = "NEUTRAL"
    result["current_funding_sign"] = funding_sign

    # --- Historical analysis: compute pre/post returns at each settlement ---
    if not hasattr(ohlcv.index, "hour"):
        return result

    prices = ohlcv[price_col].dropna()
    if len(prices) < 48:
        return result

    pre_rets_pos, post_rets_pos = [], []   # when funding was positive at settlement
    pre_rets_neg, post_rets_neg = [], []   # when funding was negative at settlement

    # Get funding series for historical sign classification
    fund_series = None
    if funding_df is not None and not funding_df.empty:
        fund_col = funding_df.columns[0]
        fund_series = funding_df[fund_col]

    price_idx = prices.index
    for i in range(PRE_WINDOW, len(prices) - POST_WINDOW):
        ts = price_idx[i]
        if not hasattr(ts, 'hour'):
            continue
        if ts.hour not in SETTLEMENT_HOURS:
            continue

        pre_ret = (prices.iloc[i] - prices.iloc[i - PRE_WINDOW]) / prices.iloc[i - PRE_WINDOW]
        post_ret = (prices.iloc[i + POST_WINDOW] - prices.iloc[i]) / prices.iloc[i]

        # Get funding sign at this settlement (use nearest value)
        if fund_series is not None and len(fund_series) > 0:
            try:
                nearest_fund = fund_series.asof(ts) if hasattr(fund_series.index, 'freq') else \
                               fund_series.iloc[fund_series.index.searchsorted(ts, side="right") - 1]
                f_sign = "POSITIVE" if float(nearest_fund) > 0 else "NEGATIVE"
            except Exception:
                f_sign = "POSITIVE" if avg_funding > 0 else "NEGATIVE"
        else:
            # Use current funding as proxy for historical (imperfect but usable)
            f_sign = funding_sign

        if f_sign == "POSITIVE":
            pre_rets_pos.append(pre_ret)
            post_rets_pos.append(post_ret)
        else:
            pre_rets_neg.append(pre_ret)
            post_rets_neg.append(post_ret)

    # Select the relevant stats based on current funding sign
    if funding_sign == "POSITIVE":
        pre_rets = pre_rets_pos
        post_rets = post_rets_pos
    elif funding_sign == "NEGATIVE":
        pre_rets = pre_rets_neg
        post_rets = post_rets_neg
    else:
        pre_rets = pre_rets_pos + pre_rets_neg
        post_rets = post_rets_pos + post_rets_neg

    n = len(pre_rets)
    result["historical_n"] = n

    if n < 5:
        result["description"] = (
            f"Settlement in {mins_to_settle:.0f}min ({next_settle_dt.strftime('%H:%M UTC')}). "
            f"Insufficient settlement history (n={n}) to compute pattern stats."
        )
        return result

    pre_mean = float(np.mean(pre_rets) * 100)
    post_mean = float(np.mean(post_rets) * 100)
    result["pre_ret_mean"] = round(pre_mean, 3)
    result["post_ret_mean"] = round(post_mean, 3)

    # Confidence: based on consistency and sample size
    pre_consistency = (np.array(pre_rets) < 0).mean() if pre_mean < 0 else (np.array(pre_rets) > 0).mean()
    post_consistency = (np.array(post_rets) > 0).mean() if post_mean > 0 else (np.array(post_rets) < 0).mean()
    confidence = float(np.mean([pre_consistency, post_consistency]) * min(1.0, n / 20))
    result["confidence"] = round(confidence, 2)

    # Dubai-time settlement labels (UTC+4)
    dubai_settle_h = (next_settle_dt.hour + 4) % 24
    dubai_settle_str = f"{dubai_settle_h:02d}:00 Dubai"

    # Expected effect at current phase
    trade_setup = None
    trade_setup_desc = ""

    if phase == "PRE_SETTLEMENT":
        effect_dir = "DOWN" if pre_mean < 0 else "UP"
        result["expected_effect"] = f"PRE_{effect_dir}"
        action_note = (
            f"Pre-settlement with {funding_sign.lower()} funding: "
            f"historically {pre_mean:+.2f}% in last hour before settlement "
            f"({pre_consistency:.0%} consistent, n={n})."
        )
        # Trade setup: pre-settlement micro-long (positive funding only)
        if funding_sign == "POSITIVE" and pre_mean > 0:
            trade_setup = "PRE_SETTLEMENT_MICRO_LONG"
            trade_setup_desc = (
                f"⚡ PRE-SETTLEMENT MICRO-LONG — price historically +{pre_mean:.2f}% "
                f"in last 60min before {result['next_settlement_utc']} ({dubai_settle_str}). "
                f"Only valid if model ≥55%. Tight stop — exit AT settlement."
            )
        elif funding_sign == "POSITIVE" and pre_mean < 0:
            trade_setup = "PRE_SETTLEMENT_FADE"
            trade_setup_desc = (
                f"PRE-SETTLEMENT FADE — positive funding, price historically {pre_mean:.2f}% "
                f"into settlement. Longs closing to avoid paying."
            )

    elif phase == "JUST_SETTLED":
        effect_dir = "UP" if post_mean > 0 else "DOWN"
        result["expected_effect"] = f"POST_{effect_dir}"
        action_note = (
            f"Just settled with {funding_sign.lower()} funding: "
            f"historically {post_mean:+.2f}% in 2h after settlement "
            f"({post_consistency:.0%} consistent, n={n})."
        )
        if funding_sign == "POSITIVE" and post_mean < 0:
            trade_setup = "POST_SETTLEMENT_FADE"
            trade_setup_desc = (
                f"⭐ POST-SETTLEMENT FADE — funding just reset. Historically {post_mean:.2f}% "
                f"in 2h after settlement with positive funding (n={n}). "
                f"Short on any bounce. Target: -{abs(post_mean):.2f}%."
            )
        elif post_mean > 0 and post_consistency > 0.54:
            # Only show bounce when the historical hit rate is genuinely above coin-flip.
            # Negative funding → 49.6% actual hit rate = no edge. Skip.
            trade_setup = "POST_SETTLEMENT_BOUNCE"
            trade_setup_desc = (
                f"POST-SETTLEMENT BOUNCE — historically +{post_mean:.2f}% in 2h "
                f"after settlement (n={n}, {post_consistency:.0%} hit). Only if model ≥55%."
            )

    else:
        result["expected_effect"] = "MID_CYCLE"
        action_note = (
            f"Mid-cycle ({mins_to_settle:.0f}min to next settlement). "
            f"Pre-settlement avg: {pre_mean:+.2f}% | Post-settlement avg: {post_mean:+.2f}%."
        )
        # Mid-cycle: flag upcoming setup window (regardless of funding sign)
        pre_window_mins = max(0, mins_to_settle - 60)
        if abs(post_mean) > 0.05:
            post_dir_word = "fade (short)" if post_mean < 0 else "bounce (long)"
            trade_setup = "UPCOMING_POST_SETTLEMENT_FADE" if post_mean < 0 else "UPCOMING_POST_SETTLEMENT_BOUNCE"
            trade_setup_desc = (
                f"📅 UPCOMING SETUP — POST-SETTLEMENT {post_dir_word.upper()} at "
                f"{result['next_settlement_utc']} ({dubai_settle_str}, in {mins_to_settle:.0f}min). "
                f"Historically {post_mean:+.2f}% avg in 2h after settlement "
                f"(n={n}, confidence {confidence:.0%}). "
                f"Price often {'pumps into settlement — short once it peaks' if post_mean < 0 else 'dips into settlement — buy once it stops falling'}."
            )
        if abs(pre_mean) > 0.05 and mins_to_settle <= 90:
            pre_dir_word = "long" if pre_mean > 0 else "short"
            trade_setup = f"UPCOMING_PRE_SETTLEMENT_{pre_dir_word.upper()}"
            trade_setup_desc = (
                f"⏰ PRE-SETTLEMENT WINDOW OPENING — {mins_to_settle:.0f}min to "
                f"{result['next_settlement_utc']} ({dubai_settle_str}). "
                f"Historically {pre_mean:+.2f}% in last 60min before settlement "
                f"({'only if model ≥55%' if pre_mean > 0 else 'short setup confirmed'})."
            )

    result["trade_setup"] = trade_setup
    result["trade_setup_desc"] = trade_setup_desc
    result["dubai_settlement_time"] = dubai_settle_str

    result["description"] = (
        f"Settlement cycle: {result['next_settlement_utc']} ({dubai_settle_str}, "
        f"{mins_to_settle:.0f}min away). "
        f"Funding: {funding_sign} ({avg_funding:.4f}). "
        f"{action_note} Confidence: {confidence:.0%}."
        + (f" {trade_setup_desc}" if trade_setup_desc else "")
    )
    return result


# =========================================================================
# Model 12: Liquidation Cascade Detector
# =========================================================================

def _detect_liquidation_cascade(data, state):
    """
    Detect liquidation cascade signatures from OHLCV + liquidation data.

    Cascade fingerprint (OHLCV wick detection):
    - Lower wick > 2x candle body = forced sellers hit market
    - Volume spike > 2x 20-period avg = capitulation volume
    - Recovery: price closes near high of candle body

    Post-cascade entry window (2-4 candles after wick):
    - Forced sellers exhausted, only organic flow remains
    - Historically bullish: mean +0.8% to +1.4% over next 4h

    Also checks Coinalyze/Coinglass liquidation z-score if available.
    """
    result = {
        "state": "NORMAL",
        "cascade_detected": False,
        "candles_since_cascade": None,
        "wick_ratio": 0.0,
        "volume_ratio": 0.0,
        "liq_zscore": 0.0,
        "post_cascade_avg_4h": 0.0,
        "post_cascade_hit_rate": 0.0,
        "historical_n": 0,
        "confidence": 0.0,
        "description": "No liquidation cascade detected.",
    }

    ohlcv = data.get("ohlcv")
    if ohlcv is None or len(ohlcv) < 30:
        return result

    price_col = "price" if "price" in ohlcv.columns else "close"
    open_col = "open" if "open" in ohlcv.columns else None
    high_col = "high" if "high" in ohlcv.columns else None
    low_col = "low" if "low" in ohlcv.columns else None
    vol_col = "volume" if "volume" in ohlcv.columns else None

    close = ohlcv[price_col].values
    has_ohlc = all(c in ohlcv.columns for c in ["open", "high", "low"])
    has_vol = vol_col in ohlcv.columns if vol_col else False

    if not has_ohlc:
        # Can't detect wicks without OHLC — use liquidation data only
        pass
    else:
        opens = ohlcv["open"].values
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values

        # --- Historical cascade analysis ---
        # For each candle, compute wick ratio and volume ratio
        vol_20 = None
        if has_vol:
            volumes = ohlcv[vol_col].values
            # Rolling 20-period average volume
            vol_rolling = pd.Series(volumes).rolling(20, min_periods=5).mean().values

        cascade_idxs = []
        for i in range(5, len(close) - 1):
            body = abs(close[i] - opens[i])
            lower_wick = min(opens[i], close[i]) - lows[i]  # always >= 0
            if body < 1e-12:
                continue
            wr = lower_wick / body

            vol_ratio = 1.0
            if has_vol and vol_rolling[i] > 0:
                vol_ratio = volumes[i] / vol_rolling[i]

            # Cascade condition: wick ratio > 2 AND volume spike > 1.8x
            if wr > 2.0 and vol_ratio > 1.8:
                cascade_idxs.append((i, wr, vol_ratio))

        # --- Post-cascade performance (historical) ---
        post_4h_rets = []
        for idx, wr, vr in cascade_idxs:
            if idx + 4 < len(close):
                ret_4h = (close[idx + 4] - close[idx]) / close[idx]
                post_4h_rets.append(ret_4h)

        n = len(post_4h_rets)
        result["historical_n"] = n
        if n >= 5:
            result["post_cascade_avg_4h"] = round(float(np.mean(post_4h_rets) * 100), 2)
            result["post_cascade_hit_rate"] = round(float((np.array(post_4h_rets) > 0).mean()), 3)

        # --- Current state: find most recent cascade ---
        if cascade_idxs:
            last_cascade_idx, last_wr, last_vr = cascade_idxs[-1]
            candles_since = len(close) - 1 - last_cascade_idx
            result["candles_since_cascade"] = candles_since
            result["wick_ratio"] = round(last_wr, 2)
            result["volume_ratio"] = round(last_vr, 2)

            if candles_since <= 1:
                result["state"] = "CASCADE_IN_PROGRESS"
                result["cascade_detected"] = True
            elif candles_since <= 4:
                result["state"] = "POST_CASCADE_ENTRY"
                result["cascade_detected"] = True
            elif candles_since <= 12:
                result["state"] = "POST_CASCADE_WATCH"
                result["cascade_detected"] = True

    # --- Augment with Coinalyze liquidation z-score if available ---
    liquidations = data.get("liquidations")
    if liquidations is not None and not liquidations.empty:
        latest_liq = liquidations.iloc[-1]
        liq_z = float(latest_liq.get("liq_zscore", 0))
        liq_ratio = float(latest_liq.get("liq_ratio", 0.5))
        result["liq_zscore"] = round(liq_z, 2)
        result["liq_ratio_long"] = round(liq_ratio, 3)

        # Override state if liquidation data confirms cascade
        if liq_z > 2.5 and result["state"] == "NORMAL":
            result["state"] = "CASCADE_IN_PROGRESS"
            result["cascade_detected"] = True
        elif liq_z > 1.5 and result["state"] in ("NORMAL", "POST_CASCADE_WATCH"):
            if result["state"] == "NORMAL":
                result["state"] = "POST_CASCADE_WATCH"
                result["cascade_detected"] = True

    # Also check Coinglass liquidation data if available
    cg_liq = data.get("coinglass_liquidations")
    if cg_liq is not None and not cg_liq.empty:
        latest_cg = cg_liq.iloc[-1]
        cg_long_liq = float(latest_cg.get("long_liquidations_usd", 0))
        cg_short_liq = float(latest_cg.get("short_liquidations_usd", 0))
        cg_total = cg_long_liq + cg_short_liq
        cg_z = float(latest_cg.get("liq_zscore", 0))
        result["cg_total_liquidations_usd"] = round(cg_total, 0)
        result["cg_liq_zscore"] = round(cg_z, 2)

        if cg_z > 2.5:
            result["state"] = "CASCADE_IN_PROGRESS"
            result["cascade_detected"] = True
            if result["liq_zscore"] == 0.0:
                result["liq_zscore"] = cg_z

    # --- Confidence ---
    n = result["historical_n"]
    if result["cascade_detected"]:
        base_conf = min(1.0, max(result["wick_ratio"] / 5, result["liq_zscore"] / 3))
        result["confidence"] = round(min(1.0, base_conf * (1 + min(n, 20) / 20)), 2)

    # --- Description ---
    state = result["state"]
    if state == "CASCADE_IN_PROGRESS":
        result["description"] = (
            f"⚡ LIQUIDATION CASCADE IN PROGRESS — "
            f"Wick ratio: {result['wick_ratio']:.1f}x | "
            f"Volume: {result['volume_ratio']:.1f}x avg | "
            f"Liq z-score: {result['liq_zscore']:.1f}σ. "
            f"Do NOT enter long here — wait for wick to confirm."
        )
    elif state == "POST_CASCADE_ENTRY":
        n_hist = result["historical_n"]
        avg = result["post_cascade_avg_4h"]
        hr = result["post_cascade_hit_rate"]
        result["description"] = (
            f"✅ POST-CASCADE ENTRY WINDOW — {result['candles_since_cascade']} candle(s) after wick. "
            f"Forced sellers exhausted. Historical: {avg:+.1f}% avg 4h return, "
            f"{hr:.0%} hit rate (n={n_hist}). High-probability long setup."
        )
    elif state == "POST_CASCADE_WATCH":
        result["description"] = (
            f"👀 Post-cascade watching ({result['candles_since_cascade']} candles after wick). "
            f"Entry window closing. Monitor for re-test."
        )
    elif result["cascade_detected"]:
        result["description"] = (
            f"Cascade detected (z-score: {result['liq_zscore']:.1f}σ). "
            f"Monitor for entry setup."
        )
    else:
        result["description"] = "No liquidation cascade pattern detected. Market structure normal."

    return result


# =========================================================================
# Public API
# =========================================================================
# Trade Setup Synthesiser
# =========================================================================

def _compute_trade_setups(prob, mr, session, btc, settlement, cg_oi_fund, liq_cascade, market_state):
    """
    Synthesise sub-model outputs into a prioritised list of concrete trade setups
    for the current session. Each setup has: id, direction, trigger, confidence,
    target_pct, stop_note, time_window, session_window, active.

    Calibrated against live data from Apr 2026 Dubai desk session.
    """
    import datetime as _dt
    setups = []
    now_utc = _dt.datetime.utcnow()
    dubai_hour = (now_utc.hour + 4) % 24

    prob_val = prob.get("prob_positive_4h", 0.5)
    lsr_data = mr.get("lsr", {}) or {}
    lsr_pct = lsr_data.get("percentile", 0.5)
    lsr_val = lsr_data.get("current", 1.0)
    settlement_mins = settlement.get("mins_to_settlement") or 999
    settlement_phase = settlement.get("phase", "MID_CYCLE")
    post_mean = settlement.get("post_ret_mean") or 0.0
    pre_mean = settlement.get("pre_ret_mean") or 0.0
    settlement_conf = settlement.get("confidence", 0.0)
    funding_sign = settlement.get("current_funding_sign", "NEUTRAL")
    avg_funding = market_state.get("avg_funding", 0)
    dubai_settle = settlement.get("dubai_settlement_time", "")
    oi_assessment = cg_oi_fund.get("oi_assessment", "NORMAL") if cg_oi_fund.get("available") else "NORMAL"
    btc_div = prob.get("btc_divergence", False)
    cascade = liq_cascade.get("cascade_detected", False)
    post_cascade_4h = liq_cascade.get("post_cascade_avg_4h", 0)

    # ── Setup 1: POST-SETTLEMENT FADE ─────────────────────────────────────────
    # Best structural trade of the day. Data shows:
    #   Positive funding settle → 54.5% short hit rate, −0.21% avg (REAL EDGE)
    #   Negative funding settle → 49.6% long hit rate, −0.06% avg (COIN FLIP — skip)
    # Bounce direction (long) only shown when settlement_conf > 0.54 (above coin-flip).
    _is_fade_dir  = (post_mean < 0)
    _bounce_valid = (post_mean > 0 and settlement_conf > 0.54)
    if abs(post_mean) > 0.05 and settlement_conf > 0.4 and (_is_fade_dir or _bounce_valid):
        post_direction = "short" if post_mean < 0 else "long"
        post_action_verb = "short once the move stalls" if post_mean < 0 else "buy the pullback"

        if settlement_phase == "JUST_SETTLED":
            active = True
            trigger = (
                f"Settlement just happened. {post_direction.capitalize()} on the next "
                f"{'bounce' if post_mean < 0 else 'pullback'}."
            )
            time_note = "Active now"
        elif settlement_mins <= 180:
            active = True
            trigger = (
                f"Price often {'pumps' if post_mean < 0 else 'dips'} into {dubai_settle} settlement. "
                f"{post_action_verb.capitalize()} when price stops {'rising' if post_mean < 0 else 'falling'}."
            )
            time_note = f"{settlement_mins:.0f}min to setup ({dubai_settle})"
        else:
            active = False
            trigger = f"Wait — setup opens ~60min before {dubai_settle}"
            time_note = f"Opens in ~{max(0, settlement_mins - 60):.0f}min"

        setups.append({
            "id": "POST_SETTLEMENT_FADE",
            "direction": post_direction,
            "trigger": trigger,
            "target_pct": round(abs(post_mean), 2),
            "stop_note": "Stop above the pre-settlement high" if post_mean < 0 else "Stop below the pre-settlement low",
            "confidence": round(settlement_conf, 2),
            "historical_edge": f"{post_mean:+.2f}% avg post-settlement (n={settlement.get('historical_n', 0)})",
            "time_window": time_note,
            "active": active,
            "priority": 1,
        })

    # ── Setup 2: LSR EXHAUSTION SHORT ─────────────────────────────────────────
    # LSR at extreme levels → structural forced unwind within ~8h (93% hit rate)
    if lsr_pct > 0.80 and prob_val < 0.50:
        conf = round(min(0.85, lsr_pct * lsr_data.get("revert_rate", 0.93)), 2)
        setups.append({
            "id": "LSR_EXHAUSTION_SHORT",
            "direction": "short",
            "trigger": (
                f"LSR at {lsr_val:.2f} ({lsr_pct:.0%} percentile) — longs are structurally crowded. "
                f"Enter short on any 0.5-1% bounce from current price."
            ),
            "target_pct": round(lsr_data.get("avg_revert_time_h", 8) * 0.05, 2),
            "stop_note": "Stop above recent swing high, reduce if LSR drops below 1.5",
            "confidence": conf,
            "historical_edge": (
                f"LSR reversion within ~{lsr_data.get('avg_revert_time_h', 8):.0f}h, "
                f"{lsr_data.get('revert_rate', 0.93):.0%} hit rate"
            ),
            "time_window": f"~{lsr_data.get('avg_revert_time_h', 8):.0f}h window",
            "active": True,
            "priority": 2,
        })

    # ── Setup 3: BTC DIVERGENCE RESOLUTION ────────────────────────────────────
    # FART not following BTC rally despite BTC being up = structural weakness
    if btc_div and prob_val < 0.47:
        btc_2h = btc.get("btc_2h_return_pct", 0)
        setups.append({
            "id": "BTC_DIVERGENCE_RESOLUTION",
            "direction": "short",
            "trigger": (
                f"BTC rallied {btc_2h:+.1f}% but FART model is bearish at {prob_val:.0%}. "
                f"Confirm: if FART price doesn't follow BTC within 1-2h, short the divergence. "
                f"Entry: when FART stalls or rolls over while BTC holds."
            ),
            "target_pct": round(abs(prob.get("expected_move_pct", 1.0)), 2),
            "stop_note": "Stop if FART breaks out above the BTC-implied level",
            "confidence": round(btc.get("confidence", 0.5) * 0.8, 2),
            "historical_edge": "FART-BTC divergence: 91% directional accuracy on resolution",
            "time_window": "Confirm within 1-2h of BTC move",
            "active": True,
            "priority": 3,
        })

    # ── Setup 4: OI BUILDING PRICE WEAK ───────────────────────────────────────
    if oi_assessment == "OI_BUILDING_PRICE_WEAK":
        setups.append({
            "id": "OI_BUILDING_PRICE_WEAK",
            "direction": "short",
            "trigger": (
                "OI building while price is declining — longs being trapped. "
                "Enter short on failed recovery attempts (price bounces but can't reclaim prior high)."
            ),
            "target_pct": 0.50,
            "stop_note": "Stop above OI-buildup range high",
            "confidence": 0.55,
            "historical_edge": "-0.25% avg 4h, 44% hit rate (same as trend-chase exhaustion)",
            "time_window": "4-8h window",
            "active": True,
            "priority": 4,
        })

    # ── Setup 5: POST-SETTLEMENT MICRO-LONG (conditional) ────────────────────
    # Only if model crosses 55% — pre-settlement, positive funding, historically +pre_mean
    if funding_sign == "POSITIVE" and pre_mean > 0.05 and settlement_mins <= 90 and prob_val >= 0.55:
        setups.append({
            "id": "PRE_SETTLEMENT_MICRO_LONG",
            "direction": "long",
            "trigger": (
                f"Model ≥55% + pre-settlement window ({settlement_mins:.0f}min to {dubai_settle}). "
                f"Enter long now, exit AT settlement. Do NOT hold through."
            ),
            "target_pct": round(pre_mean, 2),
            "stop_note": "Hard exit at settlement regardless of P&L",
            "confidence": round(settlement_conf * 0.7, 2),
            "historical_edge": f"Pre-settlement avg: +{pre_mean:.2f}% with positive funding (n={settlement.get('historical_n', 0)})",
            "time_window": f"Active now — exit at {dubai_settle}",
            "active": True,
            "priority": 5,
        })

    # ── Setup 6: POST-CASCADE ENTRY ───────────────────────────────────────────
    if cascade and post_cascade_4h > 0:
        setups.append({
            "id": "POST_CASCADE_ENTRY",
            "direction": "long",
            "trigger": (
                "Liquidation cascade just occurred. Enter long as sellers exhaust. "
                "Look for stabilisation candle with lower wick."
            ),
            "target_pct": round(post_cascade_4h, 2),
            "stop_note": "Stop below cascade low",
            "confidence": round(liq_cascade.get("confidence", 0.6), 2),
            "historical_edge": f"Post-cascade avg: +{post_cascade_4h:.2f}% in 4h ({liq_cascade.get('post_cascade_hit_rate', 0):.0%} hit rate)",
            "time_window": "4h window post-cascade",
            "active": True,
            "priority": 6,
        })

    # ── NO TRADE scenario ─────────────────────────────────────────────────────
    if not setups or all(not s["active"] for s in setups):
        setups.append({
            "id": "NO_TRADE",
            "direction": "flat",
            "trigger": f"Model at {prob_val:.0%} — below 55% threshold, no edge after Bybit carry costs.",
            "target_pct": 0,
            "stop_note": "N/A",
            "confidence": 0,
            "historical_edge": "No valid setup detected",
            "time_window": "Wait for setup",
            "active": False,
            "priority": 99,
        })

    # Sort by priority
    setups.sort(key=lambda s: s["priority"])
    return setups


# =========================================================================
# 7. Ghost Long Detector — Funding Velocity Cross-Venue Divergence
# =========================================================================

def _detect_ghost_long(data):
    """
    Ghost Long: Binance funding rate collapsing while Bybit stays pinned at floor.
    MMs absorb sell pressure on Bybit to keep the carry trade alive, then flush.

    Signal: ΔF_mom = ∂/∂t(F_Binance - F_Bybit)
    If the spread velocity is sharply negative (Binance leading lower while Bybit
    stays elevated), longs on Bybit are the bag-holders.

    Returns dict with keys:
      state, spread_now, spread_velocity, velocity_zscore, description
    """
    result = {
        "state": "NORMAL",
        "spread_now": None,
        "spread_velocity": None,
        "velocity_zscore": None,
        "description": "Cross-venue funding velocity: no data.",
        "bearish_signal": False,
    }

    fund_snap = data.get("coinglass_funding")
    if fund_snap is None or len(fund_snap) < 4:
        return result

    try:
        df = fund_snap.copy().sort_values("timestamp")
        # Require both Binance and Bybit columns
        if "binance_rate" not in df.columns or "bybit_rate" not in df.columns:
            return result

        df = df.dropna(subset=["binance_rate", "bybit_rate"])
        if len(df) < 3:
            return result

        # Spread: Binance - Bybit (positive = Binance more bullish)
        df["spread"] = df["binance_rate"] - df["bybit_rate"]

        # Velocity: first difference of spread (change per snapshot interval)
        df["spread_vel"] = df["spread"].diff()

        spread_now  = float(df["spread"].iloc[-1])
        vel_now     = float(df["spread_vel"].iloc[-1])

        # Z-score velocity relative to history
        vel_std = df["spread_vel"].std()
        vel_zscore = (vel_now / vel_std) if vel_std > 1e-9 else 0.0

        result["spread_now"]       = round(spread_now, 4)
        result["spread_velocity"]  = round(vel_now, 4)
        result["velocity_zscore"]  = round(vel_zscore, 2)

        bybit_rate_now    = float(df["bybit_rate"].iloc[-1])
        binance_rate_now  = float(df["binance_rate"].iloc[-1])

        # Ghost Long conditions:
        # 1. Bybit at/near floor (≥ +0.40%) but Binance collapsing (rate falling fast)
        # 2. Spread velocity is sharply negative (Binance leading lower)
        bybit_elevated  = bybit_rate_now >= 0.40
        binance_falling = vel_zscore < -1.5
        spread_negative = spread_now < -0.20   # Binance already below Bybit — informed bearish

        if bybit_elevated and binance_falling:
            result["state"] = "GHOST_LONG"
            result["bearish_signal"] = True
            result["description"] = (
                f"⚠ GHOST LONG: Bybit pinned at {bybit_rate_now:+.3f}% while Binance "
                f"funding collapsing (vel {vel_now:+.4f}, {vel_zscore:+.1f}σ). "
                f"MMs absorbing sell pressure on Bybit to keep carry trade alive. "
                f"High-probability FLUSH signal — fade longs."
            )
        elif spread_negative and vel_zscore < -1.0:
            result["state"] = "INFORMED_BEARISH"
            result["bearish_signal"] = True
            result["description"] = (
                f"Binance leading lower: spread {spread_now:+.3f}% "
                f"(Binance {binance_rate_now:+.3f}% vs Bybit {bybit_rate_now:+.3f}%). "
                f"Informed venue bearish vs retail venue. Velocity {vel_zscore:+.1f}σ."
            )
        elif vel_zscore > 1.5 and spread_now > 0.20:
            result["state"] = "INFORMED_BULLISH"
            result["description"] = (
                f"Binance funding accelerating higher vs Bybit: spread {spread_now:+.3f}%, "
                f"velocity {vel_zscore:+.1f}σ. Informed venue bullish."
            )
        else:
            result["state"] = "NORMAL"
            result["description"] = (
                f"Cross-venue funding velocity normal. "
                f"Spread: {spread_now:+.3f}% (Binance {binance_rate_now:+.3f}% "
                f"vs Bybit {bybit_rate_now:+.3f}%). Velocity: {vel_zscore:+.1f}σ."
            )
    except Exception as e:
        result["description"] = f"Ghost Long detector error: {e}"

    return result


# =========================================================================
# 8. HMM Regime Switcher — 3-State Market Regime Classifier
# =========================================================================

def _classify_hmm_regime(data, market_state):
    """
    Delegates entirely to hmm_engine.label_current() — single source of truth.
    """
    if not _HMM_AVAILABLE:
        return {
            "regime": 0, "regime_label": "STEADY_STATE",
            "confidence": 0.0, "conviction_multiplier": 0.5,
            "hours_in_regime": 0, "description": "HMM unavailable.",
            "available": False,
        }
    return _hmm_label_current(data, lookback=500)


# =========================================================================
# 9. VPIN Proxy — OI-Based Informed Flow Toxicity
# =========================================================================

def _compute_vpin_proxy(data):
    """
    True VPIN requires tick-level buy/sell volume. Since our taker data is synthetic,
    we compute an OI-based informed-flow toxicity proxy:

      OI imbalance (rising OI + no price move) = stealth accumulation (toxic long)
      OI collapse + price spike = distribution (hakai end-of-move)
      OI + price co-moving = trend (less toxic — directional)

    VPIN_proxy = |OI_pct_change| * (1 - |price_OI_correlation|)
    High values = OI moving without price = informed positioning (toxic flow)

    Returns dict with: vpin_proxy, vpin_zscore, toxicity, description
    """
    result = {
        "vpin_proxy": None,
        "vpin_zscore": None,
        "toxicity": "NORMAL",
        "description": "VPIN proxy: insufficient data.",
        "available": False,
    }

    ohlcv = data.get("ohlcv")
    oi_df = data.get("oi")

    if ohlcv is None or oi_df is None or len(ohlcv) < 24:
        return result

    try:
        price_col = "price" if "price" in ohlcv.columns else "close"
        oi_col    = oi_df.columns[0]

        n = min(200, len(ohlcv), len(oi_df))
        prices  = ohlcv[price_col].iloc[-n:].values.astype(float)
        oi_vals = oi_df[oi_col].iloc[-n:].values.astype(float)

        # Bucket-based VPIN proxy over 8-hour volume buckets
        bucket = 8
        vpin_series = []
        for i in range(bucket, n):
            price_bucket = prices[i-bucket:i]
            oi_bucket    = oi_vals[i-bucket:i]

            price_ret = np.diff(price_bucket) / (price_bucket[:-1] + 1e-9)
            oi_ret    = np.diff(oi_bucket)    / (np.abs(oi_bucket[:-1]) + 1e-9)

            if len(price_ret) < 2:
                continue

            # Imbalance: |OI change| weighted by inverse price-OI correlation
            corr = np.corrcoef(price_ret, oi_ret)[0, 1] if len(price_ret) > 1 else 0.0
            corr = 0.0 if np.isnan(corr) else corr

            # High |OI_ret| with low |corr| = toxic (OI moving without price explanation)
            vpin_val = np.mean(np.abs(oi_ret)) * (1.0 - abs(corr))
            vpin_series.append(vpin_val)

        if len(vpin_series) < 5:
            return result

        vpin_arr  = np.array(vpin_series)
        vpin_now  = vpin_arr[-1]
        vpin_mean = vpin_arr.mean()
        vpin_std  = vpin_arr.std() + 1e-9
        vpin_z    = (vpin_now - vpin_mean) / vpin_std

        # Classify toxicity
        if vpin_z > 2.0:
            toxicity = "EXTREME"
            desc = (
                f"⚠ EXTREME TOXIC FLOW: VPIN proxy {vpin_now:.4f} (+{vpin_z:.1f}σ). "
                f"OI moving sharply without price correlation — informed MMs positioning "
                f"before a stop-hunt. Treat any Dormant/Buildup signal as HIGH PRIORITY."
            )
        elif vpin_z > 1.0:
            toxicity = "ELEVATED"
            desc = (
                f"ELEVATED TOXIC FLOW: VPIN proxy {vpin_now:.4f} (+{vpin_z:.1f}σ). "
                f"Above-average OI imbalance vs price. Market makers may be offloading to retail. "
                f"A Buildup signal here has higher conversion probability."
            )
        elif vpin_z < -1.0:
            toxicity = "LOW"
            desc = (
                f"Low toxic flow: VPIN proxy {vpin_now:.4f} ({vpin_z:.1f}σ). "
                f"OI and price moving together — directional, less manipulated."
            )
        else:
            toxicity = "NORMAL"
            desc = (
                f"VPIN proxy normal: {vpin_now:.4f} ({vpin_z:+.1f}σ). "
                f"No unusual informed-flow activity."
            )

        result.update({
            "vpin_proxy": round(float(vpin_now), 6),
            "vpin_zscore": round(float(vpin_z), 2),
            "toxicity": toxicity,
            "description": desc,
            "available": True,
        })

    except Exception as e:
        result["description"] = f"VPIN proxy error: {e}"

    return result


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

    # ── New structural intelligence models ────────────────────────────────
    ghost_long = _detect_ghost_long(data)
    hmm_regime = _classify_hmm_regime(data, market_state)
    vpin       = _compute_vpin_proxy(data)
    sr_levels  = _compute_sr_levels(data) if _SR_AVAILABLE else {"available": False}

    # ── HMM regime adjusts conviction on probability model ───────────────
    # In HAKAI regime, suppress entries regardless of composite score.
    # In ACCUMULATION, amplify signal conviction.
    # In STEADY_STATE, raise the bar (require stronger composite).
    if hmm_regime["available"]:
        regime_label = hmm_regime["regime_label"]
        mult = hmm_regime["conviction_multiplier"]
        raw_prob = prob["prob_positive_4h"]
        if regime_label == "HAKAI":
            # Distribution phase — clamp probability toward 0.5 (no edge)
            prob["prob_positive_4h"] = 0.5 + (raw_prob - 0.5) * mult
            prob["description"] += f" [HMM HAKAI: conviction {mult}x — exit phase, no new entries]"
        elif regime_label == "ACCUMULATION":
            # Amplify signal strength
            prob["prob_positive_4h"] = 0.5 + (raw_prob - 0.5) * mult
            prob["prob_positive_4h"] = min(max(prob["prob_positive_4h"], 0.0), 1.0)
            prob["description"] += f" [HMM ACCUMULATION: conviction {mult}x — full-send regime]"
        elif regime_label == "STEADY_STATE":
            # Compress signal — only very strong signals matter
            prob["prob_positive_4h"] = 0.5 + (raw_prob - 0.5) * mult
            prob["description"] += f" [HMM STEADY STATE: conviction {mult}x — raise bar to 0.60+]"

    # ── Ghost Long upgrades bearish conviction ────────────────────────────
    if ghost_long.get("bearish_signal"):
        current_p = prob["prob_positive_4h"]
        if current_p > 0.45:  # Only suppress if model wasn't already bearish
            prob["prob_positive_4h"] = min(current_p, 0.42)
            prob["description"] += (
                f" [GHOST LONG OVERRIDE: Bybit/Binance velocity divergence → "
                f"bearish. Fade longs.]"
            )

    # ── VPIN + Manipulation phase synergy ────────────────────────────────
    if vpin.get("toxicity") == "EXTREME" and cycle.get("phase") in ("DORMANT", "BUILDUP"):
        cycle["description"] += (
            f" ⚠ VPIN CONFIRMATION: Extreme toxic flow ({vpin['vpin_zscore']:+.1f}σ) "
            f"in {cycle['phase']} phase = HIGH-PRIORITY STOP-HUNT SIGNAL."
        )
        cycle["confidence"] = min(cycle.get("confidence", 0.5) + 0.15, 0.95)

    # ── Meta-model opportunity score ─────────────────────────────────────
    opportunity = {"score": 50, "tier": "WATCH", "available": False}
    if _SCORER_AVAILABLE:
        try:
            meta_df = build_meta_features(data)
            # Pass the live HMM label so score_live() can enforce the HAKAI gate
            # even when the walk-forward rolling HMM disagrees (they use different
            # history windows — live full-history label is authoritative).
            _live_hmm_lbl = hmm_regime.get("regime_label") if hmm_regime.get("available") else None
            opportunity = _score_live_meta(meta_df, live_hmm_label=_live_hmm_lbl)

            # ── Triple-ensemble escalation ────────────────────────────────────
            # When LGBM fires AND LSTM-raw lb=64+HMM independently agrees,
            # escalate to FULL SEND (97.7% hit rate in walk-forward backtest).
            _lgbm_fires = opportunity.get("tier") in ("TRADE", "HIGH CONVICTION", "FULL SEND")
            if _lgbm_fires and _score_live_lstm_raw is not None:
                try:
                    _lstm_result = _score_live_lstm_raw(data, lookback=64,
                                                         live_hmm_label=_live_hmm_lbl)
                    _lstm_fires  = _lstm_result.get("trade", 0) == 1
                    opportunity["lstm_prob"]  = _lstm_result.get("prob", 0.0)
                    opportunity["lstm_trade"] = int(_lstm_fires)
                    opportunity["triple_agreement"] = int(_lgbm_fires and _lstm_fires)
                    if _lgbm_fires and _lstm_fires:
                        # Escalate tier to FULL SEND and boost score
                        opportunity["tier"]  = "FULL SEND"
                        opportunity["score"] = max(opportunity.get("score", 80), 95)
                        opportunity["triple_escalated"] = True
                except Exception as _le:
                    opportunity["lstm_error"] = str(_le)

        except Exception as _e:
            opportunity["error"] = str(_e)

    # --- Funding level context for BTC interaction signals ---
    # Get current funding percentile from historical data
    _funding_data = data.get("funding")
    _ohlcv_data = data.get("ohlcv")
    _fund_pct_val = 0.0
    _fund_extreme_low = False
    _fund_extreme_high = False
    if _funding_data is not None and len(_funding_data) > 50:
        try:
            _fund_col = _funding_data.columns[0]
            _cur_fund = float(_funding_data[_fund_col].iloc[-1])
            _fund_ptile = float((_funding_data[_fund_col] < _cur_fund).mean())
            _fund_pct_val = _cur_fund * 100
            _fund_extreme_low  = _fund_ptile < 0.25   # bottom quartile
            _fund_extreme_high = _fund_ptile > 0.75   # top quartile
        except Exception:
            pass

    # BTC OVERRIDE: When BTC has a big move (>1.5% in 2h) with high confidence,
    # override the probability model. BTC had 100% accuracy on big moves this week.
    #
    # BACKTEST-CALIBRATED (90 days):
    #   BTC dump >2% + Fund LOW:  +1.22% avg 4h, 68% hit — STRONG BUY
    #   BTC rally >2% + Fund HIGH: -1.27% avg 4h, 24% hit — STRONG AVOID
    btc_abs = abs(btc.get("btc_2h_return_pct", 0))
    btc_conf = btc.get("confidence", 0)
    btc_2h_ret = btc.get("btc_2h_return_pct", 0)

    # Special case: BTC dump + low funding — best setup in the data (68% hit, +1.22% avg)
    if btc_2h_ret < -2.0 and _fund_extreme_low and btc_conf > 0.4:
        prob["prob_positive_4h"] = max(prob["prob_positive_4h"], 0.68)
        prob["expected_move_pct"] = abs(prob.get("expected_move_pct", 1.0))
        prob["btc_override"] = True
        prob["btc_override_type"] = "BTC_DUMP_LOW_FUND_BUY"
        prob["description"] += (
            f" [⭐ BTC DUMP+LOW FUND OVERRIDE: BTC {btc_2h_ret:+.1f}% + fund low "
            f"→ HIGH-CONV LONG (68% hist hit, +1.22% avg 4h)]"
        )
    # Special case: BTC rally + high funding — worst setup (24% hit, -1.27% avg) — don't chase
    elif btc_2h_ret > 2.0 and _fund_extreme_high and prob["prob_positive_4h"] > 0.5:
        prob["prob_positive_4h"] = min(prob["prob_positive_4h"], 0.40)
        prob["expected_move_pct"] = -abs(prob.get("expected_move_pct", 1.0))
        prob["btc_override"] = True
        prob["btc_override_type"] = "BTC_RALLY_HIGH_FUND_FADE"
        prob["description"] += (
            f" [⚠ BTC RALLY+HIGH FUND OVERRIDE: BTC +{btc_2h_ret:.1f}% + fund high "
            f"→ DON'T CHASE (24% hist hit, -1.27% avg 4h)]"
        )
    # Standard BTC direction override for large moves
    elif btc_abs > 1.5 and btc_conf > 0.5:
        btc_direction = "up" if btc_2h_ret > 0 else "down"

        # DIVERGENCE GUARD (calibrated Apr 2026):
        # When BTC is rallying BUT both major exchanges are crowded long (BOTH_CROWDED_LONG)
        # AND LSR is at extreme (>p75), FART's failure to follow BTC is itself a bearish signal.
        # These are the conditions where the bullish BTC override backfires.
        # Observed: BTC +0.8% but FART model 42% → price stalled, bearish resolution.
        _cg_oi_fund = data.get("coinglass_oi") or {}
        _lsr_data = data.get("lsr")
        _lsr_pct = 0.5
        if _lsr_data is not None and not (isinstance(_lsr_data, pd.DataFrame) and _lsr_data.empty):
            try:
                _lsr_col = "longShortRatio" if "longShortRatio" in _lsr_data.columns else _lsr_data.columns[0]
                _lsr_s = _lsr_data[_lsr_col].dropna()
                _lsr_pct = float((_lsr_s < float(_lsr_s.iloc[-1])).mean())
            except Exception:
                pass

        _both_crowded = False
        try:
            _avg_fund_pct = float(market_state.get("avg_funding", 0)) * 100
            _both_crowded = (_avg_fund_pct > 0.4 and _fund_extreme_high)  # both sides elevated
        except Exception:
            pass

        _fart_lagging_btc = (
            btc_direction == "up"
            and prob["prob_positive_4h"] < 0.47
            and _lsr_pct > 0.75
            and _both_crowded
        )

        if _fart_lagging_btc:
            # FART is NOT following BTC rally despite BTC being up — structural weakness signal
            # Keep/reinforce the bearish signal, suppress the BTC bullish override
            prob["btc_divergence"] = True
            prob["btc_divergence_type"] = "FART_LAGGING_BTC_RALLY"
            prob["description"] += (
                f" [⚠ BTC DIVERGENCE: BTC +{btc_abs:.1f}% but FART model at "
                f"{prob['prob_positive_4h']:.0%} with crowded longs (LSR p{_lsr_pct:.0%}) — "
                f"FART failing to follow BTC = structural weakness. "
                f"If FART doesn't catch up within 1-2h, bearish resolution likely. "
                f"BTC bullish override SUPPRESSED.]"
            )
        elif btc_direction == "up" and prob["prob_positive_4h"] < 0.6:
            prob["prob_positive_4h"] = max(prob["prob_positive_4h"], 0.65)
            prob["expected_move_pct"] = abs(prob["expected_move_pct"])
            prob["btc_override"] = True
            prob["btc_override_type"] = "BTC_DIRECTION"
            prob["description"] += (
                f" [BTC OVERRIDE: BTC +{btc_abs:.1f}% → forcing bullish bias, "
                f"confidence {btc_conf:.0%}]"
            )
        elif btc_direction == "down" and prob["prob_positive_4h"] > 0.4:
            prob["prob_positive_4h"] = min(prob["prob_positive_4h"], 0.35)
            prob["expected_move_pct"] = -abs(prob["expected_move_pct"])
            prob["btc_override"] = True
            prob["btc_override_type"] = "BTC_DIRECTION"
            prob["description"] += (
                f" [BTC OVERRIDE: BTC -{btc_abs:.1f}% → forcing bearish bias, "
                f"confidence {btc_conf:.0%}]"
            )

    ci = _compute_confidence_intervals(data, market_state, prob)

    # New external data models
    news = _project_news_sentiment(data, market_state)
    onchain = _project_onchain_flow(data, market_state)
    cx_derivs = _project_cross_exchange_derivatives(data, market_state)
    cg_oi_fund = _project_coinglass_oi_funding(data, market_state)

    # New structural models
    settlement = _project_funding_settlement_cycle(data, market_state)
    liq_cascade = _detect_liquidation_cascade(data, market_state)

    # Systematic rule-based signals (validated walk-forward)
    desk_setups: dict = {}
    if _SYSTEMATIC_AVAILABLE and _compute_settlement_signals is not None:
        try:
            desk_setups = _compute_settlement_signals(data)
        except Exception as _se:
            desk_setups = {"any_active": False, "signals": [], "active_signals": [],
                           "summary": f"systematic_signals error: {_se}"}

    # Synthesise trade setups from all sub-models
    trade_setups = _compute_trade_setups(
        prob, mr, session, btc, settlement, cg_oi_fund, liq_cascade, market_state
    )

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
    if cg_oi_fund.get("available"):
        parts.append(f"*OI/Funding:* {cg_oi_fund['description']}")

    # New structural intelligence
    if hmm_regime.get("available"):
        parts.append(f"*HMM Regime:* {hmm_regime['description']}")
    if ghost_long.get("state") != "NORMAL":
        parts.append(f"*Ghost Long:* {ghost_long['description']}")
    if vpin.get("available") and vpin.get("toxicity") in ("ELEVATED", "EXTREME"):
        parts.append(f"*VPIN Proxy:* {vpin['description']}")
    if opportunity.get("available"):
        parts.append(f"*Opportunity Score:* {opportunity['description']}")

    # Settlement cycle: always include (key trade window context)
    parts.append(f"*Settlement Cycle:* {settlement['description']}")
    if liq_cascade.get("cascade_detected"):
        parts.append(f"*Liq Cascade:* {liq_cascade['description']}")

    # Systematic rule-based signals
    if desk_setups.get("any_active"):
        for _sig in desk_setups.get("active_signals", []):
            parts.append(
                f"*⚡ Desk Setup [{_sig['id']}]:* {_sig['description']}"
            )

    if ci.get("h4"):
        h4 = ci["h4"]
        thin_note = " ⚠ THIN VOLUME — bands widened" if h4.get("thin_volume") else ""
        parts.append(
            f"*4h Range:* ${h4['low_68']:.4f} – ${h4['high_68']:.4f} (68%) | "
            f"${h4['low_95']:.4f} – ${h4['high_95']:.4f} (95%){thin_note}"
        )

    # Trade setups summary
    active_setups = [s for s in trade_setups if s["active"]]
    if active_setups:
        setup_lines = []
        for s in active_setups:
            setup_lines.append(
                f"  • [{s['id']}] {s['direction'].upper()} — {s['trigger']} "
                f"(conf: {s['confidence']:.0%}, target: {s['target_pct']:+.2f}%)"
            )
        parts.append("*Trade Setups:*\n" + "\n".join(setup_lines))

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
        "coinglass_oi_funding": cg_oi_fund,
        "funding_settlement": settlement,
        "liq_cascade": liq_cascade,
        "ghost_long": ghost_long,
        "hmm_regime": hmm_regime,
        "vpin_proxy": vpin,
        "support_resistance": sr_levels,
        "opportunity": opportunity,
        "trade_setups": trade_setups,
        "desk_setups": desk_setups,
        "summary": summary,
    }
