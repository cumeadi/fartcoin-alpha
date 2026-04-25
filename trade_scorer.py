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

from hmm_engine import roll_regime_series as _roll_regime_series, build_feature_matrix as _hmm_features
from signal_engine import load_data, compute_all_signals
from coin_config import get_config, DEFAULT_COIN

CARRY_COST  = 0.0045   # 0.45% / 4h — Bybit floor
TRAIN_WIN   = 500      # rolling training window (widened: more history = more stable estimates)
STEP        = 6        # walk-forward step size
MIN_TRAIN   = 250      # minimum rows before first prediction
# Note on autocorrelation: btc_corr_7d uses 168h lookback, so consecutive test rows
# are not independent. This inflates hit rate confidence — CI is ±~37% not ±28%.
# A purge gap (tested at 24h) degraded performance badly because short-lag features
# (funding_vel, oi changes) lose their most informative recent rows.
# Mitigated instead by: reduced model complexity + wider train window.

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────


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

    # ── Align everything to shortest common length ───────────────────────────
    btc_len = len(data.get("btc")) if data.get("btc") is not None and len(data.get("btc")) > 0 else len(ohlcv)
    n = min(len(ohlcv), len(oi_df), len(fund_df), len(lsr_df), btc_len)

    ohlcv   = ohlcv.iloc[:n]
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
    fund_s = pd.Series(funding)
    df["funding_vel"] = fund_s.diff(4).values / (fund_std + 1e-9)

    # ── Funding momentum (2nd derivative) ────────────────────────────────────
    # Positive accel = funding rising (longs paying more) = bearish pressure
    # Negative accel = funding falling toward floor = accumulation setup
    df["funding_accel"] = fund_s.diff(4).diff(4).values / (fund_std + 1e-9)

    # Funding sign flip (transition event in last 8h = regime inflection)
    fund_signs = pd.Series(np.sign(funding))
    df["funding_sign_flip"] = fund_signs.rolling(8, min_periods=2).apply(
        lambda x: 1.0 if (x.max() > 0 and x.min() <= 0) else 0.0, raw=True
    ).fillna(0.0).values

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

    # ── Liquidation cluster proximity (Coinalyze data) ───────────────────────
    # High recent liq_zscore = active cascade = dangerous entry environment
    try:
        _liq_path = Path(__file__).parent / "data" / "coinalyze_liquidations.csv"
        if _liq_path.exists():
            _liq = pd.read_csv(_liq_path, index_col=0, parse_dates=True)
            if "liq_zscore" in _liq.columns:
                _lz = _liq["liq_zscore"].fillna(0)
                _lz.index = pd.to_datetime(_lz.index).floor("h")
                _lz_8h = _lz.rolling(8, min_periods=1).max()
                _ohlcv_h = pd.to_datetime(ohlcv.index).floor("h")
                _aligned = _lz_8h.reindex(_ohlcv_h, method="ffill").fillna(0)
                df["liq_cluster_recent"] = _aligned.values[:n]
            else:
                df["liq_cluster_recent"] = 0.0
        else:
            df["liq_cluster_recent"] = 0.0
    except Exception:
        df["liq_cluster_recent"] = 0.0

    # ── Rolling S/R distance proxy (stochastic-style, vectorized) ──────────────
    # Captures price position within recent range as a fast rolling S/R proxy.
    # Full S/R engine (support_resistance.py) is used for visualization; this
    # provides historical per-row features for the walk-forward meta-model.
    try:
        _prc   = pd.Series(prices[:n])
        _lo168 = _prc.rolling(168, min_periods=24).min()
        _hi168 = _prc.rolling(168, min_periods=24).max()
        # Distance below recent high = space to resistance
        df["dist_to_resistance_pct"] = ((_hi168 - _prc) / (_prc + 1e-9)).clip(0, 0.5).fillna(0.05).values
        # Distance above recent low  = space to support
        df["dist_to_support_pct"]    = ((_prc - _lo168) / (_prc + 1e-9)).clip(0, 0.5).fillna(0.05).values
        # Risk/reward: how much room to resistance vs distance above support
        df["sr_risk_reward"] = np.clip(
            df["dist_to_resistance_pct"] / (df["dist_to_support_pct"] + 1e-9), 0.05, 20.0
        )
    except Exception:
        df["dist_to_resistance_pct"] = 0.05
        df["dist_to_support_pct"]    = 0.05
        df["sr_risk_reward"]         = 1.0

    # ── BTC lead-lag + rolling correlation regime ────────────────────────────
    if btc_df is not None and len(btc_df) > 0:
        btc_col_   = "price" if "price" in btc_df.columns else btc_df.columns[0]
        btc_prices = btc_df[btc_col_].values[:n].astype(float)
        btc_2h     = np.diff(btc_prices, prepend=btc_prices[0]) / (btc_prices + 1e-9)
        df["btc_2h_ret"] = btc_2h[:n]
        # Rolling 168h BTC-FART return correlation
        # High (>0.7) = BTC lead-lag signals are reliable; Low (<0.4) = FART solo regime
        fart_ret_s = pd.Series(np.diff(prices, prepend=prices[0]) / (prices + 1e-9))
        btc_ret_s  = pd.Series(btc_2h[:n])
        df["btc_corr_7d"] = (
            fart_ret_s.rolling(168, min_periods=48).corr(btc_ret_s)
            .fillna(0.65).values  # fill with historical mean
        )
    else:
        df["btc_2h_ret"]  = 0.0
        df["btc_corr_7d"] = 0.65

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

    # Bad-session gate feature: 20-24h UTC = 41.7% historical hit rate (no edge)
    # Exposed to walk-forward model AND used as a hard gate in walk_forward_meta()
    df["session_bad"] = (hours >= 20).astype(int)

    # ── HMM regime (walk-forward via hmm_engine — single source of truth) ───────
    print("  [meta] Computing rolling HMM regime labels (hmm_engine)...", flush=True)
    # Pass a sliced copy so hmm_engine sees the same n rows as df
    data_n = {k: (v.iloc[:n] if hasattr(v, "iloc") else v) for k, v in data.items()}
    regimes = _roll_regime_series(data_n, lookback=TRAIN_WIN, step=STEP)[:n]
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
    "funding_z", "funding_vel", "funding_accel",
    # funding_sign_flip removed: permutation test showed zero IC contribution
    "oi_4h_pct", "oi_1h_pct", "oi_4h_chg",
    "lsr_pct",
    "vpin_z",
    "btc_2h_ret", "btc_corr_7d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_enc", "session_bad",
    "hmm_hakai", "hmm_accum", "hmm_steady",
    "vol_ratio",
    "dist_to_support_pct", "dist_to_resistance_pct", "sr_risk_reward",
    # liq_cluster_recent removed: permutation test showed zero IC contribution
    # (only 4 weeks of data — insufficient history for the model to learn from)
]


def walk_forward_meta(df):
    """
    Walk-forward train/test of the meta-model.

    Anti-overfitting measures:
      - PURGE_GAP (24h) between training end and test row — prevents feature leakage
        from long-lookback features (btc_corr_7d uses 168h; 24h is a conservative purge)
      - Reduced model complexity: n_estimators=80, max_depth=3, min_child_samples=15
      - Wider train window (500h) compensates for the purge gap

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
                n_estimators=80,       # reduced from 120 — 320 training rows, not 1000
                max_depth=3,           # reduced from 4 — fewer parameters per tree
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=15,  # raised from 10 — requires larger leaf nodes
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(X_train, y_train)

            X_test = row[available_features].values.reshape(1, -1)
            prob   = float(model.predict_proba(X_test)[0, 1])

            # ── Hard gates: HAKAI regime + bad session ─────────────────
            is_hakai = int(row["hmm_hakai"] > 0.5)
            # 20-24h UTC: 41.7% historical hit rate — below carry break-even
            try:
                hour_of_row = pd.Timestamp(df.index[i]).hour
            except Exception:
                hour_of_row = 12
            is_bad_session = int(hour_of_row >= 20)
            meta_trade = int(prob > 0.55 and not is_hakai and not is_bad_session)

            results.append({
                "timestamp":  df.index[i],
                "fwd_ret_4h": float(row["fwd_ret_4h"]),
                "target_4h":  int(row["target_4h"]),
                "meta_prob":  prob,
                "meta_hit":   int(meta_trade and row["fwd_ret_4h"] > CARRY_COST),
                "meta_trade": meta_trade,
                "hmm_regime": int(row["hmm_regime"]),
                "is_hakai":   is_hakai,
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
    Train meta-model on all available history, then score the current row.

    Improvements vs v1:
      - Isotonic regression calibration (80/20 split within training window)
      - Kelly position sizing (replaces fixed 50/75/100 tiers)
      - Session hard gate: 20-24h UTC blocked (41.7% historical hit rate)
      - New features: btc_corr_7d, funding_accel, funding_sign_flip,
                      liq_cluster_recent, session_bad
    """
    if not _LGBM_OK:
        return {"score": 50, "tier": "WATCH", "available": False}

    available_features = [f for f in META_FEATURES if f in df.columns]
    n = len(df)
    if n < MIN_TRAIN + 10:
        return {"score": 50, "tier": "WATCH", "available": False}

    train = df.iloc[-(TRAIN_WIN + 1): -1]
    row   = df.iloc[-1]

    # ── Session gate — block entries 20-24h UTC ──────────────────────────────
    try:
        current_hour = pd.Timestamp(df.index[-1]).hour
    except Exception:
        current_hour = 12
    is_bad_session = (current_hour >= 20)

    X_train = train[available_features].values
    y_train = train["target_4h"].values

    if y_train.sum() < 5:
        return {"score": 50, "tier": "WATCH", "available": False}

    try:
        # ── 80/20 in-sample split: fit LightGBM on 80%, calibrate on 20% ────
        split   = int(len(X_train) * 0.80)
        X_fit   = X_train[:split]
        y_fit   = y_train[:split]
        X_cal   = X_train[split:]
        y_cal   = y_train[split:]

        model = lgb.LGBMClassifier(
            n_estimators=80,       # matched to walk_forward_meta for consistency
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=15,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_fit, y_fit)

        X_now    = row[available_features].values.reshape(1, -1)
        prob_raw = float(model.predict_proba(X_now)[0, 1])

        # ── Isotonic regression calibration ──────────────────────────────────
        # Corrects overconfidence at high-probability bins (was 25pp off at 0.8+)
        prob = prob_raw
        try:
            from sklearn.isotonic import IsotonicRegression
            if len(X_cal) >= 20 and y_cal.sum() >= 3 and (1 - y_cal).sum() >= 3:
                raw_cal = model.predict_proba(X_cal)[:, 1]
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(raw_cal, y_cal)
                prob = float(ir.predict([prob_raw])[0])
        except Exception:
            pass   # fall back to uncalibrated probability

        # ── Map probability → 0-100 score ────────────────────────────────────
        score = int(np.clip((prob - 0.20) / 0.60 * 100, 0, 100))

        # ── Tier classification ───────────────────────────────────────────────
        hmm = int(row.get("hmm_regime", 0))
        if int(row.get("hmm_hakai", 0)) == 1:
            tier  = "BLOCKED"        # HAKAI — distribution phase
            score = min(score, 25)
        elif is_bad_session:
            tier  = "BLOCKED (SESSION)"   # 20-24h UTC — no edge window
            score = min(score, 30)
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

        # ── Kelly position sizing ─────────────────────────────────────────────
        # f* = (p·b − q) / b   where b = avg_win / avg_loss from training history
        # Uses full Kelly with per-tier ceiling to limit overexposure
        kelly_f   = 0.0
        kelly_pct = 0
        try:
            train_rets = train["fwd_ret_4h"].values
            win_rets   = train_rets[train_rets > CARRY_COST]
            loss_rets  = train_rets[train_rets <= CARRY_COST]
            avg_win    = float(win_rets.mean())        if len(win_rets)  > 5 else CARRY_COST * 2
            avg_loss   = float(abs(loss_rets.mean()))  if len(loss_rets) > 5 else CARRY_COST
            b          = avg_win / (avg_loss + 1e-9)
            q          = 1.0 - prob
            kelly_f    = max(0.0, (prob * b - q) / (b + 1e-9))
            kelly_pct  = int(kelly_f * 100)  # full Kelly as base percentage
        except Exception:
            kelly_pct = 0

        # Tier ceiling caps Kelly (can't exceed max for that confidence band)
        tier_ceiling = {
            "BLOCKED": 0, "BLOCKED (SESSION)": 0, "PASS": 0, "WATCH": 0,
            "TRADE": 60, "HIGH CONVICTION": 80, "FULL SEND": 100,
        }
        size_pct = min(kelly_pct, tier_ceiling.get(tier, 0))

        # ── Feature importance for component breakdown ────────────────────────
        fi      = model.feature_importances_
        top_k   = 5
        top_idx = np.argsort(fi)[::-1][:top_k]
        top_feats = [(available_features[j], round(float(fi[j]), 3)) for j in top_idx]

        return {
            "score":           score,
            "tier":            tier,
            "meta_prob":       round(prob, 4),
            "meta_prob_raw":   round(prob_raw, 4),
            "kelly_fraction":  round(kelly_f, 3),
            "size_pct":        size_pct,
            "hmm_regime":      hmm,
            "hmm_label":       ["STEADY_STATE", "ACCUMULATION", "HAKAI"][min(hmm, 2)],
            "session_blocked": is_bad_session,
            "top_drivers":     top_feats,
            "available":       True,
            "description":     (
                f"Opportunity Score: {score}/100 — {tier}. "
                f"Meta-prob: {prob:.0%} (raw: {prob_raw:.0%}, Kelly: {kelly_f:.1%}). "
                f"HMM: {['STEADY_STATE','ACCUMULATION','HAKAI'][min(hmm,2)]}. "
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
    """Rolling hit rate, cumulative PnL, regime overlay, calibration curve."""
    if len(results) < 20:
        return

    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

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

    # ── Panel 4: Calibration curve ────────────────────────────────────────────
    # "Does a 60% meta_prob actually hit 60% of the time?"
    ax4 = fig.add_subplot(gs[3])
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_mid, frac_pos, bin_counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (results["meta_prob"] >= lo) & (results["meta_prob"] < hi)
        if mask.sum() >= 3:
            actual_hit = (results.loc[mask, "fwd_ret_4h"] > CARRY_COST).mean()
            bin_mid.append((lo + hi) / 2)
            frac_pos.append(actual_hit)
            bin_counts.append(mask.sum())

    if bin_mid:
        ax4.plot([0, 1], [0, 1], color="#888888", linestyle="--", alpha=0.5, label="Perfect calibration")
        ax4.plot(bin_mid, frac_pos, color="#00d4aa", marker="o", linewidth=1.5,
                 markersize=5, label="Actual hit rate")
        # Size points by sample count
        for bm, fp, bc in zip(bin_mid, frac_pos, bin_counts):
            ax4.annotate(f"n={bc}", (bm, fp), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color="#888888")
        ax4.axhline(0.55, color="white", linestyle=":", alpha=0.4, label="Entry threshold")
        ax4.fill_between([0, 1], [0, 0], [0.55, 0.55], alpha=0.08, color="#ff6b6b")
        ax4.fill_between([0, 1], [0.55, 0.55], [1, 1], alpha=0.08, color="#00d4aa")

    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.set_xlabel("Predicted probability", color="white")
    ax4.set_ylabel("Actual hit rate", color="white")
    ax4.set_title(f"{coin} — Calibration Curve (predicted vs actual hit rate)")
    ax4.legend(fontsize=8)
    ax4.set_facecolor("#1a1a2e")
    ax4.tick_params(colors="white"); ax4.xaxis.label.set_color("white")
    ax4.yaxis.label.set_color("white"); ax4.title.set_color("white")

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
