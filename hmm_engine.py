"""
HMM Engine — Single Source of Truth for Market Regime Classification

Both projections.py and trade_scorer.py import from here.
No HMM code lives anywhere else.

3 regimes:
  0  STEADY_STATE   — funding neutral, OI flat. Conviction 0.5x. Avoid trading.
  1  ACCUMULATION   — low/negative funding, rising OI, quiet price. Conviction 1.5x.
  2  HAKAI          — OI spike + elevated vol + price breakout. Conviction 0.3x.
                      HARD GATE: no new entries in this regime.

Public API
----------
  fit_regime_model(X)        → (model, regime_map)
  map_states(model, X)       → (states, posteriors, regime_map)
  label_current(data)        → dict  (for live use in projections.py)
  roll_regime_series(data, lookback, step)  → np.ndarray  (for backtesting)
  build_feature_matrix(data) → np.ndarray  shape (n, 4)
"""

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except ImportError:
    _HMM_OK = False

N_STATES   = 3
N_ITER     = 100
RANDOM_SEED = 42
TOL        = 1e-2

LABELS     = {0: "STEADY_STATE", 1: "ACCUMULATION", 2: "HAKAI"}
MULTIPLIERS = {0: 0.5, 1: 1.5, 2: 0.3}
DESCRIPTIONS = {
    0: (
        "HMM Regime: STEADY STATE (conf {conf:.0%}, {h}h). "
        "Funding neutral, OI flat. Signal conviction halved — "
        "raise composite bar to 0.60+. Avoid new positions."
    ),
    1: (
        "HMM Regime: ACCUMULATION 🟢 (conf {conf:.0%}, {h}h). "
        "Low/negative funding + quiet OI build. Conviction 1.5x — "
        "composite 0.4 is a full-send in this regime. Long bias."
    ),
    2: (
        "HMM Regime: HAKAI ⚡ (conf {conf:.0%}, {h}h). "
        "OI spike + elevated vol. Distribution/exhaustion phase. "
        "Conviction 0.3x — tighten stops NOW. No new entries."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction (shared)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(data, n=None):
    """
    Build the 4-feature matrix used by the HMM:
      [funding_z, oi_z, price_z, vol_ratio_z]

    Args:
        data: dict of DataFrames from signal_engine.load_data()
        n:    number of rows to use (None = all)

    Returns:
        X  np.ndarray  shape (n, 4)
        arrays tuple   (fund_z, oi_z, px_z, vol_z) — individual series
    """
    ohlcv   = data["ohlcv"]
    oi_df   = data["oi"]
    fund_df = data["funding"]

    price_col = "price" if "price" in ohlcv.columns else "close"
    oi_col    = oi_df.columns[0]
    fund_col  = fund_df.columns[0]

    if n is None:
        n = min(len(ohlcv), len(oi_df), len(fund_df))

    prices  = ohlcv[price_col].values[:n].astype(float)
    funding = fund_df[fund_col].values[:n].astype(float)
    oi_vals = oi_df[oi_col].values[:n].astype(float)
    vol     = ohlcv["volume"].values[:n].astype(float)

    # Funding z-score
    fm, fs  = np.nanmean(funding), np.nanstd(funding) + 1e-9
    fund_z  = (funding - fm) / fs

    # OI 1h momentum z-score
    oi_chg  = np.diff(oi_vals, prepend=oi_vals[0]) / (np.abs(oi_vals) + 1e-9)
    om, os_ = np.nanmean(oi_chg), np.nanstd(oi_chg) + 1e-9
    oi_z    = (oi_chg - om) / os_

    # Price return z-score
    px_ret  = np.diff(prices, prepend=prices[0]) / (prices + 1e-9)
    pm, ps  = np.nanmean(px_ret), np.nanstd(px_ret) + 1e-9
    px_z    = (px_ret - pm) / ps

    # Volume ratio z-score
    vol_ma  = np.convolve(vol, np.ones(24) / 24, mode="same")
    vol_r   = vol / (vol_ma + 1e-9)
    vol_z   = (vol_r - 1.0) / (np.nanstd(vol_r) + 1e-9)

    X = np.column_stack([fund_z, oi_z, px_z, vol_z])
    X = np.nan_to_num(X, nan=0.0)

    return X, (fund_z, oi_z, px_z, vol_z)


# ─────────────────────────────────────────────────────────────────────────────
# Core HMM fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_regime_model(X):
    """
    Fit a 3-state Gaussian HMM on feature matrix X.

    Returns:
        model      trained GaussianHMM
        regime_map dict {hmm_state_int → economic_regime_int (0/1/2)}
    """
    if not _HMM_OK:
        return None, {0: 0, 1: 1, 2: 2}

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        tol=TOL,
    )
    model.fit(X)
    regime_map = _build_regime_map(model)
    return model, regime_map


def _build_regime_map(model):
    """
    Map raw HMM state indices to economic regimes using state means:
      - HAKAI (2)       = highest combined OI_z + vol_z  (features 1 & 3)
      - ACCUMULATION (1) = lowest funding_z               (feature 0)
      - STEADY_STATE (0) = the remainder
    """
    means = model.means_   # (n_states, n_features)

    hakai_scores = means[:, 1] + means[:, 3]   # oi_z + vol_z
    hakai_s      = int(np.argmax(hakai_scores))

    funding_order = np.argsort(means[:, 0])     # ascending funding
    accum_s = int(funding_order[0])
    if accum_s == hakai_s:
        accum_s = int(funding_order[1])

    steady_s = [s for s in range(N_STATES) if s not in (hakai_s, accum_s)]
    steady_s = steady_s[0] if steady_s else (3 - hakai_s - accum_s)

    return {hakai_s: 2, accum_s: 1, steady_s: 0}


def map_states(model, X, regime_map):
    """
    Predict states for X and map to economic regime integers.

    Returns:
        regimes     np.ndarray int  (0/1/2 per row)
        posteriors  np.ndarray float (n_rows, N_STATES) — posterior probs
    """
    raw_states = model.predict(X)
    posteriors = model.predict_proba(X)
    regimes    = np.array([regime_map.get(int(s), 0) for s in raw_states])

    # Remap posterior columns to economic regime order
    econ_post = np.zeros((len(X), N_STATES))
    for raw_s, econ_s in regime_map.items():
        econ_post[:, econ_s] += posteriors[:, raw_s]

    return regimes, econ_post


# ─────────────────────────────────────────────────────────────────────────────
# Live label (projections.py uses this)
# ─────────────────────────────────────────────────────────────────────────────

def label_current(data, lookback=500):
    """
    Train HMM on last `lookback` rows, label the current row.

    Returns dict with keys:
      regime, regime_label, confidence, conviction_multiplier,
      hours_in_regime, description, available
    """
    result = {
        "regime": 0, "regime_label": "STEADY_STATE",
        "confidence": 0.0, "conviction_multiplier": 0.5,
        "hours_in_regime": 0, "description": "HMM: insufficient data.",
        "available": False,
    }

    if not _HMM_OK:
        return result

    try:
        ohlcv = data.get("ohlcv")
        oi_df = data.get("oi")
        fund_df = data.get("funding")
        if ohlcv is None or oi_df is None or fund_df is None:
            return result
        if min(len(ohlcv), len(oi_df), len(fund_df)) < 48:
            return result

        n  = min(lookback, len(ohlcv), len(oi_df), len(fund_df))
        X, _ = build_feature_matrix(data, n=n)

        model, regime_map = fit_regime_model(X)
        if model is None:
            return result

        regimes, posteriors = map_states(model, X, regime_map)

        current_regime  = int(regimes[-1])
        confidence      = float(posteriors[-1, current_regime])

        # Cap confidence at 95% — 100% is almost always a degenerate model
        confidence = min(confidence, 0.95)

        # Count trailing hours in current regime
        trailing = 0
        for r in reversed(regimes):
            if r == current_regime:
                trailing += 1
            else:
                break

        label = LABELS[current_regime]
        desc  = DESCRIPTIONS[current_regime].format(conf=confidence, h=trailing)

        result.update({
            "regime":               current_regime,
            "regime_label":         label,
            "confidence":           round(confidence, 3),
            "conviction_multiplier": MULTIPLIERS[current_regime],
            "hours_in_regime":      trailing,
            "description":          desc,
            "available":            True,
        })
    except Exception as e:
        result["description"] = f"HMM error: {e}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward series (trade_scorer.py uses this)
# ─────────────────────────────────────────────────────────────────────────────

def roll_regime_series(data, lookback=400, step=6):
    """
    Compute regime label for every row using a strictly causal rolling window.
    Train on rows [i-lookback, i), label the STEP rows ending at i.

    Returns np.ndarray int shape (n,), values 0/1/2.
    No lookahead — each row is labeled using only past data.
    """
    if not _HMM_OK:
        n = len(data["ohlcv"])
        return np.zeros(n, dtype=int)

    X, _ = build_feature_matrix(data)
    n     = len(X)
    regimes = np.full(n, -1, dtype=int)

    for i in range(lookback, n, step):
        window = X[i - lookback: i]
        try:
            model, regime_map = fit_regime_model(window)
            if model is None:
                continue
            # Label the window to get the mapping, then label the step slice
            step_slice = X[max(0, i - step): i]
            step_regimes, _ = map_states(model, step_slice, regime_map)
            for j, r in enumerate(step_regimes):
                idx = i - step + j
                if 0 <= idx < n:
                    regimes[idx] = int(r)
        except Exception:
            continue

    # Forward-fill any gaps (first `lookback` rows stay unlabeled → STEADY)
    for i in range(n):
        if regimes[i] == -1:
            regimes[i] = regimes[i - 1] if i > 0 else 0

    return regimes
