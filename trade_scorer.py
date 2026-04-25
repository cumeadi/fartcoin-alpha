"""
Trade Opportunity Scorer — Meta-Model Stack

Sits on top of all existing sub-models and learns which COMBINATIONS of signals
predict a carry-adjusted profitable trade.

Architecture:
  Layer 1 (sub-models)  →  composite, HMM regime proxy, VPIN proxy, Ghost Long
                            velocity, LSR pct, OI momentum, funding z-score,
                            session encoding, BTC lead-lag
  Layer 2 (meta-model)  →  LightGBM trained walk-forward on Layer 1 outputs
  Output                →  opportunity_score (0-100), tier, component breakdown,
                            recommended position size %

Walk-forward validation:
  - Train window: 400h rolling (≈17 days)
  - Step:         6h (no overlapping test windows)
  - Target:       fwd_ret_4h > CARRY_COST (0.45%)
  - Minimum obs:  200 before first prediction

The key insight vs. the existing LightGBM:
  The existing model predicts direction from raw market microstructure features.
  This meta-model predicts trade SUCCESS from *processed signal outputs* —
  including the HMM regime state, which regime-gates every other signal.

Run:
    python3 trade_scorer.py --coin FARTCOIN
    python3 trade_scorer.py --coin ZEC
"""

import argparse
import warnings
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

try:
    import lightgbm as lgb
    _LGBM_OK = True
except ImportError:
    _LGBM_OK = False

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except ImportError:
    _HMM_OK = False

from signal_engine import load_data, compute_all_signals
from coin_config import get_config, DEFAULT_COIN

CARRY_COST  = 0.0045   # 0.45% / 4h — Bybit floor
TRAIN_WIN   = 400      # rolling training window in hours
STEP        = 6        # walk-forward step size
MIN_TRAIN   = 200      # minimum rows before first prediction
HMM_STATES  = 3

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _hmm_regime_series(fund_z, oi_z, px_z, vol_z, lookback=400):
    """
    Compute rolling HMM regime for each row.
    Uses past `lookback` rows to train, then labels the current row.
    Returns int array: 0=STEADY, 1=ACCUMULATION, 2=HAKAI
    """
    n = len(fund_z)
    regimes = np.full(n, -1, dtype=int)

    if not _HMM_OK:
        return regimes

    X = np.column_stack([fund_z, oi_z, px_z, vol_z])
    X = np.nan_to_num(X, nan=0.0)

    for i in range(lookback, n, STEP):
        window = X[max(0, i - lookback): i]
        try:
            model = GaussianHMM(
                n_components=HMM_STATES,
                covariance_type="diag",
                n_iter=80,
                random_state=42,
                tol=1e-2,
            )
            model.fit(window)
            state_seq   = model.predict(window)
            state_means = model.means_  # shape (n_states, n_features)

            # Map states to economic regimes by feature profiles
            # Feature 0 = fund_z: ACCUMULATION has lowest (most negative funding)
            # Feature 2 = oi_z + feature 3 = vol_z: HAKAI has highest OI+vol
            hakai_score = state_means[:, 1] + state_means[:, 3]   # oi_z + vol_z
            hakai_s     = int(np.argmax(hakai_score))
            accum_s     = int(np.argmin(state_means[:, 0]))        # lowest funding_z
            if accum_s == hakai_s:
                order   = np.argsort(state_means[:, 0])
                accum_s = int(order[1] if order[0] == hakai_s else order[0])
            steady_s    = [s for s in range(HMM_STATES) if s not in (hakai_s, accum_s)]
            steady_s    = steady_s[0] if steady_s else (3 - hakai_s - accum_s)

            regime_map  = {hakai_s: 2, accum_s: 1, steady_s: 0}

            # Label the range [i-STEP, i)
            last_states = model.predict(X[max(0, i - STEP): i])
            for j, s in enumerate(last_states):
                idx = i - STEP + j
                if 0 <= idx < n:
                    regimes[idx] = regime_map.get(int(s), 0)

        except Exception:
            continue

    # Forward-fill any unlabeled rows
    for i in range(n):
        if regimes[i] == -1:
            regimes[i] = regimes[i - 1] if i > 0 else 0

    return regimes


def build_meta_features(data):
    """
    Build the full feature matrix for the meta-model.

    Returns DataFrame aligned to OHLCV index with columns:
      composite, sig_*, hmm_regime, vpin_proxy, funding_velocity,
      lsr_pct, oi_4h_pct, funding_z, btc_2h_ret, session_enc,
      hour_sin, hour_cos, dow_sin, dow_cos, fwd_ret_4h, fwd_ret_8h, target_4h
    """
    ohlcv   = data["ohlcv"].copy()
    oi_df   = data["oi"].copy()
    fund_df = data["funding"].copy()
    lsr_df  = data["lsr"].copy()
    btc_df  = data.get("btc")
    signals = data.get("signals")

    price_col = "price" if "price" in ohlcv.columns else "close"
    oi_col    = oi_df.columns[0]
    fund_col  = fund_df.columns[0]
    lsr_col   = lsr_df.columns[0]

    # ── Align everything to OHLCV index ─────────────────────────────────────
    n = len(ohlcv)
    prices  = ohlcv[price_col].values.astype(float)
    funding = fund_df[fund_col].values[:n].astype(float)
    oi_vals = oi_df[oi_col].values[:n].astype(float)
    lsr_val = lsr_df[lsr_col].values[:n].astype(float)

    # ── Core sub-signals ────────────────────────────────────────────────────
    df = pd.DataFrame(index=ohlcv.index)

    # Composite + sub-signals from signal engine
    if signals is not None and len(signals) > 0:
        for col in ["composite"] + [c for c in signals.columns if c.startswith("sig_")]:
            if col in signals.columns:
                df[col] = signals[col].values[:n] if len(signals) >= n else np.nan
    else:
        df["composite"] = 0.0

    # ── Regime proxy features (used to build HMM + as raw features) ─────────
    fund_mean = np.nanmean(funding)
    fund_std  = np.nanstd(funding) + 1e-9
    fund_z    = (funding - fund_mean) / fund_std

    oi_4h     = np.diff(oi_vals, prepend=oi_vals[0]) / (np.abs(oi_vals) + 1e-9)
    oi_z      = (oi_4h - np.nanmean(oi_4h)) / (np.nanstd(oi_4h) + 1e-9)

    px_ret    = np.diff(prices, prepend=prices[0]) / (prices + 1e-9)
    px_z      = (px_ret - np.nanmean(px_ret)) / (np.nanstd(px_ret) + 1e-9)

    vol       = ohlcv["volume"].values.astype(float)
    vol_ma    = np.convolve(vol, np.ones(24) / 24, mode="same")
    vol_ratio = vol / (vol_ma + 1e-9)
    vol_z     = (vol_ratio - 1.0) / (np.nanstd(vol_ratio) + 1e-9)

    df["funding_z"]   = fund_z
    df["oi_4h_pct"]   = oi_4h
    df["oi_z"]        = oi_z
    df["px_z"]        = px_z
    df["vol_ratio"]   = vol_ratio

    # ── LSR percentile (rolling 200h) ────────────────────────────────────────
    lsr_s = pd.Series(lsr_val)
    df["lsr_pct"] = lsr_s.rolling(200, min_periods=20).rank(pct=True).values

    # ── OI 1h & 4h change ────────────────────────────────────────────────────
    df["oi_1h_pct"] = pd.Series(oi_vals).pct_change(1).values
    df["oi_4h_chg"] = pd.Series(oi_vals).pct_change(4).values

    # ── Funding velocity proxy (synthetic: ΔfundingRate 4h) ─────────────────
    df["funding_vel"] = pd.Series(funding).diff(4).values / (fund_std + 1e-9)

    # ── VPIN proxy: rolling 8h buckets ──────────────────────────────────────
    vpin_vals = np.zeros(n)
    bucket = 8
    for i in range(bucket, n):
        px_b  = prices[i - bucket: i]
        oi_b  = oi_vals[i - bucket: i]
        pr    = np.diff(px_b) / (px_b[:-1] + 1e-9)
        or_   = np.diff(oi_b) / (np.abs(oi_b[:-1]) + 1e-9)
        if len(pr) > 1:
            corr = np.corrcoef(pr, or_)[0, 1]
            corr = 0.0 if np.isnan(corr) else corr
            vpin_vals[i] = np.mean(np.abs(or_)) * (1.0 - abs(corr))
    vpin_mean = np.nanmean(vpin_vals[vpin_vals > 0])
    vpin_std  = np.nanstd(vpin_vals[vpin_vals > 0]) + 1e-9
    df["vpin_z"] = (vpin_vals - vpin_mean) / vpin_std

    # ── BTC lead-lag ─────────────────────────────────────────────────────────
    if btc_df is not None and len(btc_df) > 0:
        btc_col   = "price" if "price" in btc_df.columns else btc_df.columns[0]
        btc_prices = btc_df[btc_col].values[:n].astype(float)
        btc_2h    = np.diff(btc_prices, prepend=btc_prices[0]) / (btc_prices + 1e-9)
        df["btc_2h_ret"] = btc_2h
    else:
        df["btc_2h_ret"] = 0.0

    # ── Session & time encoding (cyclic) ─────────────────────────────────────
    try:
        hours = pd.DatetimeIndex(ohlcv.index).hour
        dows  = pd.DatetimeIndex(ohlcv.index).dayofweek
    except Exception:
        hours = np.zeros(n, dtype=int)
        dows  = np.zeros(n, dtype=int)

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dows / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dows / 7)

    # Session encoding: 0=Asia, 1=London, 2=NYC, 3=Late NYC
    session_enc = np.where((hours >= 8) & (hours < 13), 1,
                  np.where((hours >= 13) & (hours < 17), 2,
                  np.where((hours >= 17) & (hours < 22), 3, 0)))
    df["session_enc"] = session_enc

    # ── HMM regime (walk-forward, no lookahead) ───────────────────────────────
    print("  [meta] Computing rolling HMM regime labels...", flush=True)
    regimes = _hmm_regime_series(fund_z, oi_z, px_z, vol_z, lookback=TRAIN_WIN)
    df["hmm_regime"]   = regimes
    df["hmm_hakai"]    = (regimes == 2).astype(float)
    df["hmm_accum"]    = (regimes == 1).astype(float)
    df["hmm_steady"]   = (regimes == 0).astype(float)

    # ── Forward returns (target) ──────────────────────────────────────────────
    df["fwd_ret_4h"] = pd.Series(prices, index=ohlcv.index).pct_change(4).shift(-4).values
    df["fwd_ret_8h"] = pd.Series(prices, index=ohlcv.index).pct_change(8).shift(-8).values
    df["target_4h"]  = (df["fwd_ret_4h"] > CARRY_COST).astype(int)
    df["target_8h"]  = (df["fwd_ret_8h"] > CARRY_COST * 2).astype(int)

    return df.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward meta-model
# ─────────────────────────────────────────────────────────────────────────────

META_FEATURES = [
    "composite",
    "sig_funding", "sig_oi_divergence", "sig_oi_accel",
    "sig_lsr", "sig_taker", "sig_volume_spike",
    "funding_z", "funding_vel",
    "oi_4h_pct", "oi_1h_pct", "oi_4h_chg",
    "lsr_pct",
    "vpin_z",
    "btc_2h_ret",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_enc",
    "hmm_hakai", "hmm_accum", "hmm_steady",
    "vol_ratio",
]


def walk_forward_meta(df):
    """
    Walk-forward train/test of the meta-model.

    Returns results DataFrame with columns:
      timestamp, fwd_ret_4h, target_4h, meta_prob, meta_hit,
      hmm_regime, is_hakai, is_accum
    """
    if not _LGBM_OK:
        raise RuntimeError("lightgbm required for meta-model training")

    available_features = [f for f in META_FEATURES if f in df.columns]
    results = []

    n       = len(df)
    start   = MIN_TRAIN + TRAIN_WIN
    indices = list(range(start, n - 8, STEP))

    print(f"  [meta] Walk-forward: {len(indices)} eval points, {len(available_features)} features")

    for k, i in enumerate(indices):
        if k % 50 == 0:
            pct = k / len(indices) * 100
            print(f"    [{pct:3.0f}%] {df.index[i] if hasattr(df.index[i], 'strftime') else i}", flush=True)

        train = df.iloc[i - TRAIN_WIN: i]
        row   = df.iloc[i]

        X_train = train[available_features].values
        y_train = train["target_4h"].values

        if y_train.sum() < 10 or (1 - y_train).sum() < 10:
            continue

        try:
            model = lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(X_train, y_train)

            X_test = row[available_features].values.reshape(1, -1)
            prob   = float(model.predict_proba(X_test)[0, 1])

            results.append({
                "timestamp":  df.index[i],
                "fwd_ret_4h": float(row["fwd_ret_4h"]),
                "target_4h":  int(row["target_4h"]),
                "meta_prob":  prob,
                "meta_hit":   int(prob > 0.55 and row["fwd_ret_4h"] > CARRY_COST),
                "meta_trade": int(prob > 0.55),
                "hmm_regime": int(row["hmm_regime"]),
                "is_hakai":   int(row["hmm_hakai"] > 0.5),
                "is_accum":   int(row["hmm_accum"] > 0.5),
                "vpin_z":     float(row["vpin_z"]),
                "composite":  float(row["composite"]),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Live scorer (uses last trained model implicitly via full-history features)
# ─────────────────────────────────────────────────────────────────────────────

def score_live(df, projections=None):
    """
    Train meta-model on all available history (last TRAIN_WIN rows as train),
    then score the current row. Returns opportunity score dict.
    """
    if not _LGBM_OK:
        return {"score": 50, "tier": "WATCH", "available": False}

    available_features = [f for f in META_FEATURES if f in df.columns]
    n = len(df)
    if n < MIN_TRAIN + 10:
        return {"score": 50, "tier": "WATCH", "available": False}

    train = df.iloc[-(TRAIN_WIN + 1): -1]
    row   = df.iloc[-1]

    X_train = train[available_features].values
    y_train = train["target_4h"].values

    if y_train.sum() < 5:
        return {"score": 50, "tier": "WATCH", "available": False}

    try:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=8,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_train, y_train)

        X_now = row[available_features].values.reshape(1, -1)
        prob  = float(model.predict_proba(X_now)[0, 1])

        # Map probability → 0-100 score
        # 0.50 → 50, 0.65 → 75, 0.35 → 25, etc.
        score = int(np.clip((prob - 0.20) / 0.60 * 100, 0, 100))

        # ── Tier classification ──────────────────────────────────────────────
        hmm = int(row.get("hmm_regime", 0))
        if int(row.get("hmm_hakai", 0)) == 1:
            tier   = "BLOCKED"   # HAKAI — never enter
            score  = min(score, 25)
        elif score >= 78:
            tier = "FULL SEND"
        elif score >= 65:
            tier = "HIGH CONVICTION"
        elif score >= 55:
            tier = "TRADE"
        elif score >= 45:
            tier = "WATCH"
        else:
            tier = "PASS"

        # ── Position size recommendation ─────────────────────────────────────
        size_pct = {
            "BLOCKED": 0,
            "PASS": 0,
            "WATCH": 0,
            "TRADE": 50,
            "HIGH CONVICTION": 75,
            "FULL SEND": 100,
        }.get(tier, 0)

        # ── Feature importance for component breakdown ────────────────────────
        fi    = model.feature_importances_
        top_k = 5
        top_idx   = np.argsort(fi)[::-1][:top_k]
        top_feats = [(available_features[j], round(float(fi[j]), 3)) for j in top_idx]

        # ── SHAP-style direction (positive = bullish contribution) ────────────
        # Simple: sign of feature × weight of feature on current row
        current_vals = row[available_features].values
        feature_directions = {}
        for j, fname in enumerate(available_features):
            direction = "↑" if current_vals[j] * fi[j] > 0 else "↓"
            feature_directions[fname] = direction

        return {
            "score":           score,
            "tier":            tier,
            "meta_prob":       round(prob, 4),
            "size_pct":        size_pct,
            "hmm_regime":      hmm,
            "hmm_label":       ["STEADY_STATE", "ACCUMULATION", "HAKAI"][min(hmm, 2)],
            "top_drivers":     top_feats,
            "available":       True,
            "description":     (
                f"Opportunity Score: {score}/100 — {tier}. "
                f"Meta-prob: {prob:.0%}. HMM: {['STEADY_STATE','ACCUMULATION','HAKAI'][min(hmm,2)]}. "
                f"Recommended size: {size_pct}% of max position."
            ),
        }
    except Exception as e:
        return {"score": 50, "tier": "WATCH", "available": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring & reporting
# ─────────────────────────────────────────────────────────────────────────────

def score_results(results, coin):
    """Print walk-forward backtest summary."""
    if results.empty:
        print("No results.")
        return

    n_total   = len(results)
    n_trade   = results["meta_trade"].sum()
    trade_pct = n_trade / n_total

    all_hit   = (results["fwd_ret_4h"] > CARRY_COST).mean()
    meta_hit  = results.loc[results["meta_trade"] == 1, "fwd_ret_4h"].apply(lambda r: r > CARRY_COST).mean() if n_trade > 0 else 0
    lift      = meta_hit - all_hit

    # By regime
    hakai_rows  = results[results["is_hakai"] == 1]
    accum_rows  = results[results["is_accum"] == 1]
    steady_rows = results[(results["is_hakai"] == 0) & (results["is_accum"] == 0)]

    accum_trade_rows = accum_rows[accum_rows["meta_trade"] == 1]
    accum_hit  = (accum_trade_rows["fwd_ret_4h"] > CARRY_COST).mean() if len(accum_trade_rows) > 0 else 0

    # Average return when trading
    avg_ret_trade = results.loc[results["meta_trade"] == 1, "fwd_ret_4h"].mean() if n_trade > 0 else 0
    avg_ret_all   = results["fwd_ret_4h"].mean()

    # Sharpe (simplified, per-trade)
    trade_rets = results.loc[results["meta_trade"] == 1, "fwd_ret_4h"]
    sharpe     = (trade_rets.mean() / (trade_rets.std() + 1e-9)) * np.sqrt(252 / 4) if len(trade_rets) > 1 else 0

    print(f"\n{'='*70}")
    print(f"  TRADE OPPORTUNITY SCORER — {coin}")
    print(f"  {n_total} eval points | carry cost: {CARRY_COST:.2%}/4h")
    print(f"{'='*70}")
    print(f"\n  ── Overall ──")
    print(f"  Baseline hit rate (all bars):   {all_hit:.1%}  (n={n_total})")
    print(f"  Meta-model trades:              {trade_pct:.1%} of bars (n={n_trade})")
    print(f"  Meta hit rate (when trading):   {meta_hit:.1%}")
    print(f"  Lift over baseline:             {lift:+.1%}")
    print(f"  Avg return when trading:        {avg_ret_trade:+.3%}  (vs {avg_ret_all:+.3%} all bars)")
    print(f"  Annualised Sharpe (trades):     {sharpe:+.2f}")
    print(f"\n  ── By HMM Regime ──")
    print(f"  STEADY_STATE  bars:  {len(steady_rows):4d} | trade rate: {steady_rows['meta_trade'].mean():.1%}")
    print(f"  ACCUMULATION  bars:  {len(accum_rows):4d} | trade rate: {accum_rows['meta_trade'].mean():.1%} | hit rate when trading: {accum_hit:.1%}")
    print(f"  HAKAI         bars:  {len(hakai_rows):4d} | trade rate: {hakai_rows['meta_trade'].mean():.1%}  (should be ~0)")

    # Monthly breakdown
    results["month"] = pd.to_datetime(results["timestamp"]).dt.to_period("M")
    monthly = results.groupby("month").apply(
        lambda g: pd.Series({
            "hit_rate": (g.loc[g["meta_trade"]==1, "fwd_ret_4h"] > CARRY_COST).mean() if g["meta_trade"].sum() > 0 else np.nan,
            "n_trades": g["meta_trade"].sum(),
        })
    )
    print(f"\n  ── Monthly Performance ──")
    for period, row in monthly.iterrows():
        bar = "█" * int(row["hit_rate"] * 20) if not np.isnan(row["hit_rate"]) else ""
        print(f"  {period}   {row['hit_rate']:5.1%}  {bar}  (n={int(row['n_trades'])})")

    print(f"\n{'='*70}\n")


def plot_results(results, coin):
    """Rolling hit rate, cumulative PnL, regime overlay."""
    if len(results) < 20:
        return

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)

    timestamps = pd.to_datetime(results["timestamp"])
    trade_mask = results["meta_trade"] == 1
    trade_rows = results[trade_mask]

    # ── Panel 1: Rolling hit rate ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    window = 30
    rolling_hit = (
        pd.Series((trade_rows["fwd_ret_4h"] > CARRY_COST).values, index=trade_rows.index)
        .rolling(window, min_periods=5).mean()
    )
    ax1.plot(trade_rows.index, rolling_hit, color="#00d4aa", linewidth=1.5, label=f"{window}-trade rolling hit rate")
    ax1.axhline(0.55, color="white", linestyle="--", alpha=0.5, label="55% target")
    ax1.axhline(0.396, color="#ff6b6b", linestyle=":", alpha=0.5, label=f"Baseline {0.396:.1%}")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Hit Rate")
    ax1.set_title(f"{coin} — Meta-Model Rolling Hit Rate (trades only)")
    ax1.legend(fontsize=8)
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="white"); ax1.yaxis.label.set_color("white"); ax1.title.set_color("white")

    # ── Panel 2: Cumulative PnL ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    cum_all   = results["fwd_ret_4h"].cumsum()
    cum_trade = trade_rows["fwd_ret_4h"].cumsum()
    ax2.plot(results.index, cum_all,   color="#888888", linewidth=1,   label="Buy & hold all bars", alpha=0.6)
    ax2.plot(trade_rows.index, cum_trade, color="#00d4aa", linewidth=1.5, label="Meta-model trades only")
    ax2.axhline(0, color="white", linestyle="-", alpha=0.2)
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title("Cumulative PnL (meta-model vs. all bars)")
    ax2.legend(fontsize=8)
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white"); ax2.yaxis.label.set_color("white"); ax2.title.set_color("white")

    # ── Panel 3: HMM Regime overlay ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    colors = {0: "#888888", 1: "#00d4aa", 2: "#ff6b6b"}
    labels = {0: "Steady", 1: "Accumulation", 2: "HAKAI"}
    for regime_id in [0, 1, 2]:
        mask = results["hmm_regime"] == regime_id
        ax3.scatter(
            results.index[mask], results.loc[mask, "fwd_ret_4h"],
            color=colors[regime_id], alpha=0.4, s=10, label=labels[regime_id]
        )
    ax3.axhline(CARRY_COST, color="yellow", linestyle="--", alpha=0.5, label=f"Carry {CARRY_COST:.2%}")
    ax3.axhline(0, color="white", alpha=0.2)
    ax3.set_ylabel("4h Forward Return")
    ax3.set_title("4h Returns Colored by HMM Regime")
    ax3.legend(fontsize=8)
    ax3.set_facecolor("#1a1a2e")
    ax3.tick_params(colors="white"); ax3.yaxis.label.set_color("white"); ax3.title.set_color("white")

    fig.patch.set_facecolor("#0d0d1a")
    out = OUTPUT_DIR / f"meta_model_{coin}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trade Opportunity Scorer — Meta-Model Backtest")
    parser.add_argument("--coin", default=DEFAULT_COIN)
    args = parser.parse_args()

    cfg     = get_config(args.coin)
    perp    = cfg["perp_symbol"]
    cmc     = cfg["cmc_symbol"]
    cg      = cfg["cg_coin_id"]

    print(f"\n{'='*70}")
    print(f"  TRADE OPPORTUNITY SCORER  |  Coin: {args.coin}")
    print(f"  Train window: {TRAIN_WIN}h  |  Step: {STEP}h  |  Carry: {CARRY_COST:.2%}/4h")
    print(f"{'='*70}\n")

    print("[1/4] Loading data...")
    data = load_data(perp_symbol=perp, cmc_symbol=cmc, cg_coin_id=cg)
    signals = compute_all_signals(data)
    data["signals"] = signals
    print(f"  OHLCV: {len(data['ohlcv'])} rows")

    print("\n[2/4] Building meta-features...")
    df = build_meta_features(data)
    print(f"  Feature matrix: {df.shape}  |  Target base rate: {df['target_4h'].mean():.1%}")

    print("\n[3/4] Walk-forward evaluation...")
    results = walk_forward_meta(df)
    out_csv = OUTPUT_DIR / f"meta_model_{args.coin}.csv"
    results.to_csv(out_csv, index=False)
    print(f"  Results saved → {out_csv}")

    print("\n[4/4] Scoring & plotting...")
    score_results(results, args.coin)
    plot_results(results, args.coin)

    # Live score
    print("  [Live score]")
    live = score_live(df)
    print(f"  Score: {live['score']}/100  |  Tier: {live['tier']}  |  Size: {live['size_pct']}%")
    print(f"  HMM: {live.get('hmm_label')}  |  Meta-prob: {live.get('meta_prob', 0):.1%}")
    if live.get("top_drivers"):
        print("  Top drivers:", live["top_drivers"])


if __name__ == "__main__":
    main()
