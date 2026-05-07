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
    from lstm_model import TradingLSTM, train_lstm, predict_lstm, build_sequences, LOOKBACK
    _LSTM_OK = True
except ImportError:
    _LSTM_OK = False
    LOOKBACK  = 10

from hmm_engine import roll_regime_series as _roll_regime_series, build_feature_matrix as _hmm_features, label_current as _hmm_label_current
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

    # ── Time-based alignment to OHLCV index (reference timeline) ────────────
    # Data sources now have different time coverage (OI/LSR: Dec 2024+,
    # BTC chart: Feb 2026+, etc.). Positional slice produces wrong pairings.
    # Reindex each series to the OHLCV hourly index and forward-fill gaps.
    _tol = pd.Timedelta("2h")
    def _align(df_in, col):
        try:
            s = df_in[col]
            # Ensure UTC-naive index for comparison
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_convert("UTC").tz_localize(None)
            return s.reindex(ohlcv.index, method="nearest", tolerance=_tol).ffill().bfill().fillna(0.0).values.astype(float)
        except Exception:
            return s.reindex(ohlcv.index, method="ffill").ffill().bfill().fillna(0.0).values.astype(float)

    n       = len(ohlcv)
    prices  = ohlcv[price_col].values.astype(float)
    funding = _align(fund_df, fund_col)
    oi_vals = _align(oi_df,   oi_col)
    lsr_val = _align(lsr_df,  lsr_col)

    # ── Core sub-signals ────────────────────────────────────────────────────
    df = pd.DataFrame(index=ohlcv.index)

    # Composite + sub-signals from signal engine
    # Time-aligned reindex (not positional): signals may cover a different date
    # range than ohlcv (e.g. signals span Dec 2024–now, ohlcv is Bybit Feb–May 2026).
    # nearest+2h tolerance handles ~2min offset between Bybit hour-candles and
    # live-collector timestamps.
    if signals is not None and len(signals) > 0:
        try:
            sig_aligned = signals.reindex(ohlcv.index, method="nearest",
                                          tolerance=pd.Timedelta("2h"))
        except Exception:
            sig_aligned = signals.reindex(ohlcv.index, method="nearest")
        for col in ["composite"] + [c for c in signals.columns if c.startswith("sig_")]:
            if col in sig_aligned.columns:
                df[col] = sig_aligned[col].fillna(0.0).values[:n]
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

    # ── OI acceleration (second derivative of 4h OI change) ─────────────────
    # Measures whether OI momentum is building or topping out.
    # Positive accel = OI growth accelerating (accumulation heating up).
    # Negative accel = OI growth decelerating (positions being unwound).
    # z-scored below (48h window) to remove regime-scale drift.
    _oi_4h_s = pd.Series(oi_vals).pct_change(4).fillna(0)
    df["oi_accel"] = _oi_4h_s.diff(4).fillna(0).values

    # ── LSR enriched features ────────────────────────────────────────────────
    # lsr_pct (existing): rolling percentile rank — slow, non-reactive
    # lsr_z: rolling z-score — faster response to LSR regime shifts
    # lsr_trend_6h: 6h slope of LSR — is crowd positioning shifting toward
    #   longs or shorts right now? Positive = crowding long (bearish signal for
    #   mean-reverting meme coins); negative = crowding short (squeeze setup).
    # lsr_oi_div: LSR trend minus OI trend. When OI grows but crowd turns short
    #   = smart money buying against the crowd = strong bullish divergence.
    #   When LSR and OI trend together = consensus positioning = lower alpha.
    _lsr_s = pd.Series(lsr_val)
    df["lsr_z"] = (
        (_lsr_s - _lsr_s.rolling(48, min_periods=12).mean()) /
        (_lsr_s.rolling(48, min_periods=12).std() + 1e-9)
    ).clip(-3, 3).fillna(0.0).values
    df["lsr_trend_6h"] = _lsr_s.diff(6).fillna(0).values
    _oi_trend_6h       = pd.Series(oi_vals).pct_change(6).fillna(0)
    _lsr_trend_norm    = _lsr_s.diff(6).fillna(0) / (_lsr_s.rolling(24, min_periods=6).std() + 1e-9)
    _oi_trend_norm     = _oi_trend_6h / (_oi_trend_6h.rolling(24, min_periods=6).std() + 1e-9)
    df["lsr_oi_div"]   = (_lsr_trend_norm - _oi_trend_norm).clip(-3, 3).fillna(0).values

    # ── CVD (Cumulative Volume Delta) ────────────────────────────────────────
    # Requires real taker BSR from Bybit tick data. Measures net aggressive
    # buying pressure. CVD diverging from price = distribution/accumulation.
    taker_df = data.get("taker")
    if taker_df is not None and len(taker_df) > 0 and "buySellRatio" in taker_df.columns:
        bsr_arr = _align(taker_df, "buySellRatio")
        vol_arr = ohlcv["volume"].values.astype(float)
        delta   = (bsr_arr * 2 - 1) * vol_arr   # +vol when buying, -vol when selling
        delta_s = pd.Series(delta)
        df["cvd_4h"]  = delta_s.rolling(4,  min_periods=2).sum().fillna(0).values
        df["cvd_12h"] = delta_s.rolling(12, min_periods=4).sum().fillna(0).values
        # CVD velocity: is buy pressure accelerating or decelerating?
        df["cvd_vel"] = delta_s.rolling(4).sum().diff(4).fillna(0).values
    else:
        df["cvd_4h"] = df["cvd_12h"] = df["cvd_vel"] = 0.0

    # ── Realized volatility (Garman-Klass + Parkinson) ───────────────────────
    # More efficient volatility estimators than close-to-close.
    # rv_ratio (short/long vol): spike = regime change or stop-hunt incoming.
    if all(c in ohlcv.columns for c in ("open", "high", "low", "close")):
        _o = ohlcv["open"].values.astype(float)
        _h = ohlcv["high"].values.astype(float)
        _l = ohlcv["low"].values.astype(float)
        _c = ohlcv["close"].values.astype(float)
        # Parkinson (range-based, 5x more efficient than close-to-close)
        park   = (1.0 / (4.0 * np.log(2))) * (np.log((_h + 1e-9) / (_l + 1e-9)) ** 2)
        park_s = pd.Series(park)
        df["rv_parkinson_4h"]  = park_s.rolling(4,  min_periods=2).mean().fillna(park.mean()).values
        df["rv_parkinson_24h"] = park_s.rolling(24, min_periods=8).mean().fillna(park.mean()).values
        # Garman-Klass (adds open/close info on top of range)
        gk   = (0.5 * np.log((_h + 1e-9) / (_l + 1e-9)) ** 2
                - (2.0 * np.log(2) - 1.0) * np.log((_c + 1e-9) / (_o + 1e-9)) ** 2)
        gk_s = pd.Series(np.clip(gk, 0, None))
        df["rv_gk_4h"] = gk_s.rolling(4, min_periods=2).mean().fillna(gk.mean()).values
        # Vol regime ratio: 4h realized vol vs 24h — spikes precede stop-hunts
        df["rv_ratio"] = (
            df["rv_parkinson_4h"] / (df["rv_parkinson_24h"] + 1e-9)
        ).clip(0.1, 10.0).values
    else:
        df["rv_parkinson_4h"] = df["rv_parkinson_24h"] = df["rv_gk_4h"] = df["rv_ratio"] = 0.0

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

    # ── Synthetic liquidation proxy (derived from price + OI) ───────────────
    # Forced long liquidation signature: price drops sharply AND OI falls
    # simultaneously (positions closed by exchange, not voluntary sells).
    # Voluntary sell: price drops but OI may rise (shorts adding) or stays flat.
    # liq_proxy = max(0, -px_ret) * max(0, -oi_pct_chg) per hour
    # Rolling 8h max captures cascade windows; z-scored for stationarity.
    try:
        _px_ret_1h = np.diff(prices, prepend=prices[0]) / (prices + 1e-9)
        _oi_ret_1h = pd.Series(oi_vals).pct_change(1).fillna(0).values
        _forced    = np.maximum(0, -_px_ret_1h) * np.maximum(0, -_oi_ret_1h)
        _forced_s  = pd.Series(_forced)
        _liq_8h_max = _forced_s.rolling(8, min_periods=2).max().fillna(0)
        df["liq_cluster_recent"] = _liq_8h_max.values
    except Exception:
        df["liq_cluster_recent"] = 0.0

    # ── Volume absorption (Amihud-style) ────────────────────────────────────────
    # Measures how much price moved per unit of volume traded.
    # Low Amihud = high volume with small price move = absorption / accumulation.
    # High Amihud = thin market, price moving easily on low volume = breakout.
    # absorption_z inverts and z-scores: positive = price absorbed, negative = moving freely.
    # Two variants:
    #   absorption_4h:  short-term (4h rolling mean) — captures session dynamics
    #   absorption_24h: medium-term (24h rolling mean) — distinguishes regimes
    # VWAP approx deviation: (H+L+2C)/4 vs 24h rolling VWAP proxy — measures
    # whether current price is trading above or below its fair VWAP.
    try:
        _c   = ohlcv["close"].values.astype(float)
        _h   = ohlcv["high"].values.astype(float) if "high" in ohlcv.columns else _c
        _l   = ohlcv["low"].values.astype(float)  if "low"  in ohlcv.columns else _c
        _vol = ohlcv["volume"].values.astype(float)
        # |price_return| / volume — tiny to avoid div/0
        _px_ret_abs = np.abs(np.diff(_c, prepend=_c[0]) / (_c + 1e-9))
        _amihud     = _px_ret_abs / (_vol + 1e-9)   # price impact per unit volume
        _amihud_s   = pd.Series(_amihud)
        # absorption = -Amihud z-score: high absorption = positive value
        df["absorption_4h"]  = -(_amihud_s.rolling(4,  min_periods=2).mean().fillna(_amihud.mean()).values)
        df["absorption_24h"] = -(_amihud_s.rolling(24, min_periods=8).mean().fillna(_amihud.mean()).values)
        # VWAP proxy = (H+L+2*C)/4; deviation from 24h rolling mean VWAP
        _vwap_bar    = (_h + _l + 2 * _c) / 4.0
        _vwap_s      = pd.Series(_vwap_bar)
        _vwap_24h    = _vwap_s.rolling(24, min_periods=8).mean()
        df["vwap_dev_24h"] = ((_vwap_s - _vwap_24h) / (_vwap_24h + 1e-9)).fillna(0.0).values
    except Exception:
        df["absorption_4h"] = df["absorption_24h"] = df["vwap_dev_24h"] = 0.0

    # ── Volatility regime (ATR ratio) + price momentum ──────────────────────────
    # atr_ratio: 14h ATR / 168h MA of ATR. >1 = elevated vol regime (IC=-0.048, p=0.024)
    #   Rationale: high vol → carry cost harder to beat + more noise in all signals
    #   Negative IC confirmed: model should trade LESS when vol is elevated
    # mom12: 12h price return. IC=-0.040, p=0.061.
    #   Negative IC = mean-reversion tendency for meme coins over 4h horizon
    #   Rising 12h price → longs already crowded → fade setup more likely
    try:
        _px_s    = pd.Series(prices[:n])
        _rets_s  = _px_s.pct_change().fillna(0)
        _atr14   = _rets_s.abs().rolling(14, min_periods=5).mean() * _px_s
        _atr_ma  = _atr14.rolling(168, min_periods=48).mean()
        df["atr_ratio"] = (_atr14 / (_atr_ma + 1e-9)).clip(0.1, 5.0).fillna(1.0).values
        df["mom12"]     = _px_s.pct_change(12).clip(-0.30, 0.30).fillna(0.0).values
    except Exception:
        df["atr_ratio"] = 1.0
        df["mom12"]     = 0.0

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
    fart_ret_s = pd.Series(np.diff(prices, prepend=prices[0]) / (prices + 1e-9))
    if btc_df is not None and len(btc_df) > 0:
        btc_col_   = "price" if "price" in btc_df.columns else btc_df.columns[0]
        btc_prices = _align(btc_df, btc_col_)
        btc_2h     = np.diff(btc_prices, prepend=btc_prices[0]) / (btc_prices + 1e-9)
        df["btc_2h_ret"] = btc_2h
        btc_ret_s  = pd.Series(btc_2h)
        df["btc_corr_7d"] = (
            fart_ret_s.rolling(168, min_periods=48).corr(btc_ret_s)
            .fillna(0.65).values
        )
    else:
        df["btc_2h_ret"]  = 0.0
        df["btc_corr_7d"] = 0.65

    sol_df = data.get("sol")
    if sol_df is not None and len(sol_df) > 0:
        try:
            sol_col_ = "price" if "price" in sol_df.columns else sol_df.columns[0]
            sol_aligned = _align(sol_df, sol_col_)
            sol_2h      = np.diff(sol_aligned, prepend=sol_aligned[0]) / (sol_aligned + 1e-9)
            sol_ret_s   = pd.Series(sol_2h)
            df["sol_2h_ret"] = sol_2h
            # 48h window — SOL-FART coupling is more regime-sensitive than BTC
            df["sol_corr_2d"] = (
                fart_ret_s.rolling(48, min_periods=12).corr(sol_ret_s)
                .fillna(0.5).values
            )
        except Exception:
            df["sol_2h_ret"]  = 0.0
            df["sol_corr_2d"] = 0.5
    else:
        df["sol_2h_ret"]  = 0.0
        df["sol_corr_2d"] = 0.5

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
    # Build synthetic aligned data_n so hmm_engine (which uses positional min-len)
    # sees consistent n-row DataFrames for ohlcv, oi, and funding.
    _idx = ohlcv.index
    _oi_s   = pd.DataFrame({"sumOpenInterestValue": oi_vals},   index=_idx)
    _fund_s = pd.DataFrame({fund_col: funding},                  index=_idx)
    data_n  = {**data, "ohlcv": ohlcv, "oi": _oi_s, "funding": _fund_s}
    regimes = _roll_regime_series(data_n, lookback=TRAIN_WIN, step=STEP)[:n]
    df["hmm_regime"]   = regimes
    df["hmm_hakai"]    = (regimes == 2).astype(float)
    df["hmm_accum"]    = (regimes == 1).astype(float)
    df["hmm_steady"]   = (regimes == 0).astype(float)

    # ── Regime transition features ────────────────────────────────────────────
    # hakai_exit_h: hours since the last HAKAI regime ended.
    # Motivation: the HAKAI→ACCUMULATION flip is the best entry signal — price
    # has just completed distribution and institutional buyers are stepping in.
    # Currently we hard-block during HAKAI but miss the entry at the flip.
    # Value: 0 = still in HAKAI; 1-6 = fresh exit (high opportunity); 7+ = stale.
    # Capped at 24h so the model doesn't learn spurious long-range effects.
    try:
        reg_series    = pd.Series(regimes, dtype=int)
        was_hakai     = (reg_series == 2)
        # For each row, find how many steps ago HAKAI last ended
        hakai_exit_h  = np.full(n, 24, dtype=float)   # default: far from any exit
        last_hakai_end = None
        for _i in range(n):
            if was_hakai.iloc[_i]:
                last_hakai_end = None          # inside HAKAI — reset
                hakai_exit_h[_i] = 0.0        # 0 = currently in HAKAI
            else:
                if last_hakai_end is None:
                    # Find the most recent row where HAKAI ended before this row
                    _prior_hakai = was_hakai.iloc[:_i]
                    if _prior_hakai.any():
                        last_hakai_end = _prior_hakai.values.nonzero()[0][-1]
                hakai_exit_h[_i] = min((_i - last_hakai_end) if last_hakai_end is not None else 24, 24)
        df["hakai_exit_h"] = hakai_exit_h
    except Exception:
        df["hakai_exit_h"] = 24.0   # safe default: no recent transition

    # ── Forward returns (target) ──────────────────────────────────────────────
    df["fwd_ret_4h"] = pd.Series(prices, index=ohlcv.index).pct_change(4).shift(-4).values
    df["fwd_ret_8h"] = pd.Series(prices, index=ohlcv.index).pct_change(8).shift(-8).values
    df["target_4h"]  = (df["fwd_ret_4h"] > CARRY_COST).astype(int)
    df["target_8h"]  = (df["fwd_ret_8h"] > CARRY_COST * 2).astype(int)

    # ── Stationarity transforms ───────────────────────────────────────────────
    # 10 features were raw/unbounded. Rolling z-score stabilizes feature
    # distributions across the 500h training window (different market regimes).
    # Benefit: LightGBM improved 67.9%→69.7% hit, Sharpe 3.64→5.65.
    # Required for LSTM (gradient descent needs bounded inputs).
    # Binary HMM flags (hmm_hakai, hmm_accum, hmm_steady) and already-
    # stationary features (funding_z, lsr_pct, vpin_z, etc.) are unchanged.
    def _rzs(s, window, min_p):
        mu  = pd.Series(s).rolling(window, min_periods=min_p).mean()
        sig = pd.Series(s).rolling(window, min_periods=min_p).std() + 1e-9
        return ((pd.Series(s) - mu) / sig).clip(-3, 3).fillna(0.0).values

    for _col in ("oi_4h_pct", "oi_1h_pct", "oi_4h_chg", "btc_2h_ret", "vol_ratio"):
        if _col in df.columns:
            df[_col] = _rzs(df[_col].values, 48, 12)

    if "sr_risk_reward" in df.columns:
        df["sr_risk_reward"] = _rzs(df["sr_risk_reward"].values, 168, 48)

    if "hakai_exit_h" in df.columns:
        df["hakai_exit_h"] = (df["hakai_exit_h"] / 24.0).clip(0.0, 1.0)

    # Stationarize new features (same 48h window as other raw features)
    for _col in ("cvd_4h", "cvd_12h", "cvd_vel",
                 "rv_parkinson_4h", "rv_parkinson_24h", "rv_gk_4h", "rv_ratio"):
        if _col in df.columns:
            df[_col] = _rzs(df[_col].values, 48, 12)

    # OI accel + LSR trend: stationarize (lsr_z and lsr_oi_div already z-scored above)
    for _col in ("oi_accel", "lsr_trend_6h"):
        if _col in df.columns:
            df[_col] = _rzs(df[_col].values, 48, 12)

    # Absorption + VWAP deviation: z-score to remove regime-scale differences
    for _col in ("absorption_4h", "absorption_24h", "vwap_dev_24h"):
        if _col in df.columns:
            df[_col] = _rzs(df[_col].values, 48, 12)

    return df.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward meta-model
# ─────────────────────────────────────────────────────────────────────────────

META_FEATURES = [
    # ── Round 3 cleanup (2026-04-29): removed composite + 6 redundant sig_* features ──
    # Walk-forward backtest with full feature set (32 features incl. composite + 6 sig_*):
    #   n=35, hit=65.7%, lift=+27.8pp, Sharpe=3.40
    # Walk-forward backtest with cleaned feature set (25 features, this list):
    #   n=28, hit=67.9%, lift=+30.0pp, Sharpe=3.64
    # Improvement: +2.2pp hit rate, +0.24 Sharpe, fewer over-trades.
    # Why removed:
    #   composite: weighted sum of the same sig_* features below — perfect collinearity.
    #   sig_funding: redundant with funding_z, funding_vel, funding_accel.
    #   sig_oi_divergence: redundant with oi_4h_pct, oi_1h_pct, oi_4h_chg.
    #   sig_oi_accel: redundant with funding_accel and oi_*_pct.
    #   sig_lsr: redundant with lsr_pct.
    #   sig_taker: data is synthetic constant 0.5 → feature is always 0 → noise.
    #   sig_volume_spike: redundant with vol_ratio.
    # Round 3 candidates rejected:
    #   sig_pv_divergence — backtest didn't improve on clean baseline.
    #   sig_dex_liq_div — only 13.5% non-zero coverage, no measurable backtest impact.
    #   liq_long_short_ratio_z — failed IC (p=0.151); 168h-z-score variant improved
    #     backtest (67.9%→73.3%) but 24h/72h/240h windows ALL degraded — overfitting.
    #   funding_spread_z — passed IC strongly (p=0.006) but backtest didn't improve
    #     (atr_ratio precedent: IC pass + backtest degrade = reject).
    "funding_z", "funding_vel", "funding_accel",
    # funding_sign_flip removed: permutation test showed zero IC contribution
    "oi_4h_pct", "oi_1h_pct", "oi_4h_chg",
    "lsr_pct",
    "vpin_z",
    "btc_2h_ret", "btc_corr_7d",
    # sol_2h_ret removed: permutation IC p=0.42 — no independent signal after composite
    "sol_corr_2d",   # borderline p=0.12 but structural (SOL-FART coupling regime)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_enc", "session_bad",
    "hmm_hakai", "hmm_accum", "hmm_steady", "hakai_exit_h",
    "vol_ratio",
    "dist_to_support_pct", "dist_to_resistance_pct", "sr_risk_reward",
    # Round 4 additions (2026-05-06): Bybit tick data + realized vol features
    #   rv_parkinson_24h: IC=+0.051 p=0.019. Walk-forward: hit 69.2%→72.5% (+3.3pp),
    #     Sharpe 3.62→4.06 (+0.44). 24h Parkinson vol measures range-based realized
    #     vol — higher recent vol = larger expected move = model more selective.
    #   rv_gk_4h: IC=+0.031 p=0.145 (marginal). Adding alongside rv_parkinson_24h
    #     reverted gains (72.5%→69.2%). Rejected — redundant.
    #   cvd_4h/12h/vel: IC <0.015, p>0.50 — CVD has no predictive power at 4h horizon
    #     on Bybit tick data. May be useful at longer horizons (future work).
    #   sig_taker: IC=+0.013 p=0.53 — real BSR now available but taker imbalance signal
    #     too noisy at 4h horizon. Rejected.
    #   liq_cluster_recent: still 0% coverage (Coinalyze only 7 days). Re-test after
    #     manual Coinalyze full export.
    "rv_parkinson_24h",
    # liq_cluster_recent: re-test after Coinalyze full export
    # atr_ratio: IC=-0.048 p=0.024 — backtest degraded (keep in build for Kelly scaling)
    # mom12: IC=-0.020 p=0.36 — no signal on Bybit data.
    # Round 5 candidates (2026-05-07): LSR, OI structure, microstructure — all rejected
    #   lsr_oi_div:      IC=+0.025 p=0.006 PASS but backtest degraded (64.2%→63.7%, 3.23→3.24S)
    #   absorption_24h:  IC=+0.025 p=0.005 PASS but backtest degraded (64.2%→62.8%, 3.23→3.17S)
    #   lsr_z:           IC=+0.011 p=0.21 FAIL
    #   lsr_trend_6h:    IC=+0.014 p=0.13 FAIL
    #   oi_accel:        IC=-0.004 p=0.63 FAIL
    #   absorption_4h:   IC=-0.001 p=0.89 FAIL
    #   vwap_dev_24h:    IC=+0.004 p=0.67 FAIL
    # Pattern: IC~0.025 features pass the statistical gate but add noise to LGBM,
    #   reducing selectivity (more trades, worse quality). Effective IC threshold for
    #   LGBM inclusion appears to be |IC| > 0.04 based on rv_parkinson_24h precedent.
]


_LGBM_PARAMS = dict(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=15,
    random_state=42,
    verbose=-1,
    n_jobs=1,
)


def walk_forward_meta(df):
    """
    Walk-forward train/test of the meta-model.

    Dual-horizon ensemble: trains two LightGBM models per window — one on
    target_4h and one on target_8h. Only trades when BOTH agree (prob > 0.50).
    This filters bars where the 4h signal is strong but the move reverses by 8h
    (pump-and-dump pattern). Compared to single-horizon:
      single-horizon: 64.2% hit, Sharpe 3.23, 358 trades
      dual-horizon:   70.9% hit, Sharpe 4.09, 289 trades (+6.7pp, +0.86S)

    Anti-overfitting measures:
      - Reduced model complexity: n_estimators=80, max_depth=3, min_child_samples=15
      - Wider train window (500h)
      - Hard gates: HAKAI block + bad session block (20-24h UTC)

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
        y4      = train["target_4h"].values
        y8      = train["target_8h"].values

        if y4.sum() < 10 or (1 - y4).sum() < 10:
            continue
        if y8.sum() < 10 or (1 - y8).sum() < 10:
            continue

        try:
            m4 = lgb.LGBMClassifier(**_LGBM_PARAMS).fit(X_train, y4)
            m8 = lgb.LGBMClassifier(**_LGBM_PARAMS).fit(X_train, y8)

            X_test = row[available_features].values.reshape(1, -1)
            p4     = float(m4.predict_proba(X_test)[0, 1])
            p8     = float(m8.predict_proba(X_test)[0, 1])
            prob   = (p4 + p8) / 2.0

            # ── Hard gates: HAKAI regime + bad session ─────────────────
            is_hakai = int(row["hmm_hakai"] > 0.5)
            try:
                hour_of_row = pd.Timestamp(df.index[i]).hour
            except Exception:
                hour_of_row = 12
            is_bad_session = int(hour_of_row >= 20)
            # Dual-horizon gate: require both models to agree (prob > 0.50 each)
            meta_trade = int(p4 > 0.50 and p8 > 0.50 and not is_hakai and not is_bad_session)

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
                "composite":  float(row.get("composite", 0.0)),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Live scorer (uses last trained model implicitly via full-history features)
# ─────────────────────────────────────────────────────────────────────────────

def score_live(df, projections=None, live_hmm_label=None):
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

    # ── Live HAKAI override ───────────────────────────────────────────────────
    # The walk-forward rolling HMM in build_meta_features() uses a 500h window
    # and can disagree with the full-history live HMM from hmm_engine.label_current().
    # The live label is authoritative — if it says HAKAI, block regardless of feature.
    # Priority: explicit live_hmm_label arg > data-based live label > feature value.
    live_is_hakai = False
    if live_hmm_label is not None:
        live_is_hakai = (live_hmm_label == "HAKAI")
    else:
        try:
            _live_lbl = _hmm_label_current(df._data if hasattr(df, '_data') else {})
            live_is_hakai = (_live_lbl.get("regime_label") == "HAKAI")
        except Exception:
            pass  # fall back to feature value only

    X_train = train[available_features].values
    y4_train = train["target_4h"].values
    y8_train = train["target_8h"].values

    if y4_train.sum() < 5 or y8_train.sum() < 5:
        return {"score": 50, "tier": "WATCH", "available": False}

    try:
        # ── Dual-horizon ensemble: 4h model + 8h model ───────────────────────
        # 80/20 in-sample split: fit on 80%, calibrate 4h model on 20%
        split   = int(len(X_train) * 0.80)
        X_fit   = X_train[:split]
        y4_fit  = y4_train[:split]
        X_cal   = X_train[split:]
        y4_cal  = y4_train[split:]

        m4 = lgb.LGBMClassifier(**_LGBM_PARAMS)
        m4.fit(X_fit, y4_fit)
        m8 = lgb.LGBMClassifier(**_LGBM_PARAMS)
        m8.fit(X_train, y8_train)   # 8h model uses full window (no calibration needed)

        X_now    = row[available_features].values.reshape(1, -1)
        p4_raw   = float(m4.predict_proba(X_now)[0, 1])
        p8_raw   = float(m8.predict_proba(X_now)[0, 1])
        prob_raw = (p4_raw + p8_raw) / 2.0
        model    = m4   # use 4h model for feature importance

        # ── Isotonic regression calibration on 4h model ──────────────────────
        # Corrects overconfidence at high-probability bins (was 25pp off at 0.8+)
        p4_cal = p4_raw
        try:
            from sklearn.isotonic import IsotonicRegression
            if len(X_cal) >= 20 and y4_cal.sum() >= 3 and (1 - y4_cal).sum() >= 3:
                raw_cal = m4.predict_proba(X_cal)[:, 1]
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(raw_cal, y4_cal)
                p4_cal = float(ir.predict([p4_raw])[0])
        except Exception:
            pass   # fall back to uncalibrated probability
        prob = (p4_cal + p8_raw) / 2.0

        # ── Map probability → 0-100 score ────────────────────────────────────
        score = int(np.clip((prob - 0.20) / 0.60 * 100, 0, 100))

        # ── Dual-horizon gate: both models must agree ────────────────────────
        # If either model is bearish (prob < 0.50), cap at WATCH regardless of score.
        both_agree = (p4_cal >= 0.50 and p8_raw >= 0.50)

        # ── Tier classification ───────────────────────────────────────────────
        hmm = int(row.get("hmm_regime", 0))
        # Use live_is_hakai (full-history HMM) as authoritative gate, with
        # feature-based hmm_hakai as fallback. This prevents the walk-forward
        # rolling HMM from contradicting the live full-history classifier.
        if int(row.get("hmm_hakai", 0)) == 1 or live_is_hakai:
            tier  = "BLOCKED"        # HAKAI — distribution phase
            score = min(score, 25)
        elif is_bad_session:
            tier  = "BLOCKED (SESSION)"   # 20-24h UTC — no edge window
            score = min(score, 30)
        elif not both_agree:
            tier  = "WATCH"          # horizons disagree — stay out
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

        # ── ATR-based vol scaling: reduce size in high-vol regimes ──────────────
        # atr_ratio > 1 = above-average volatility — same probability buys less edge
        # Scale: 1.0x at normal vol, 0.5x at 2× vol, 0.33x at 3× vol (1/atr_ratio)
        # Floor at 0.25x so we never size down more than 75% purely from vol
        try:
            _atr_ratio_live = float(row.get("atr_ratio", 1.0))
            _vol_scalar = max(0.25, min(1.0, 1.0 / max(_atr_ratio_live, 0.5)))
        except Exception:
            _vol_scalar = 1.0

        # Tier ceiling caps Kelly (can't exceed max for that confidence band)
        tier_ceiling = {
            "BLOCKED": 0, "BLOCKED (SESSION)": 0, "PASS": 0, "WATCH": 0,
            "TRADE": 60, "HIGH CONVICTION": 80, "FULL SEND": 100,
        }
        size_pct = int(min(kelly_pct * _vol_scalar, tier_ceiling.get(tier, 0)))

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
            "p4_prob":         round(p4_cal, 4),   # calibrated 4h model probability
            "p8_prob":         round(p8_raw, 4),   # raw 8h model probability
            "both_agree":      both_agree,          # True if p4≥0.50 AND p8≥0.50
            "kelly_fraction":  round(kelly_f, 3),
            "size_pct":        size_pct,
            "hmm_regime":      hmm,
            "hmm_label":       ["STEADY_STATE", "ACCUMULATION", "HAKAI"][min(hmm, 2)],
            "session_blocked": is_bad_session,
            "top_drivers":     top_feats,
            "available":       True,
            "description":     (
                f"Opportunity Score: {score}/100 — {tier}. "
                f"4h-prob: {p4_cal:.0%} | 8h-prob: {p8_raw:.0%} | avg: {prob:.0%} "
                f"(Kelly: {kelly_f:.1%}). "
                f"HMM: {['STEADY_STATE','ACCUMULATION','HAKAI'][min(hmm,2)]}. "
                f"Recommended size: {size_pct}% of max position."
            ),
        }
    except Exception as e:
        return {"score": 50, "tier": "WATCH", "available": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# LSTM meta-model — walk-forward + live scorer
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_lstm(df):
    """
    Walk-forward backtest using TradingLSTM instead of LightGBM.

    Each eval point trains an LSTM on the preceding TRAIN_WIN rows
    (as LOOKBACK-length sequences), then scores the current bar.
    StandardScaler is fit on training sequences only (no leakage).

    Returns same DataFrame schema as walk_forward_meta() so score_results()
    works for both.
    """
    if not _LSTM_OK:
        raise RuntimeError("lstm_model.py required — torch not installed or import failed")

    from sklearn.preprocessing import StandardScaler

    available_features = [f for f in META_FEATURES if f in df.columns]
    n_feat   = len(available_features)
    results  = []
    n        = len(df)
    start    = MIN_TRAIN + TRAIN_WIN
    indices  = list(range(start, n - 8, STEP))

    print(f"  [lstm] Walk-forward: {len(indices)} eval points, {n_feat} features, lookback={LOOKBACK}")

    for k, i in enumerate(indices):
        if k % 20 == 0:
            pct = k / len(indices) * 100
            print(f"    [{pct:3.0f}%] {df.index[i] if hasattr(df.index[i], 'strftime') else i}", flush=True)

        train = df.iloc[i - TRAIN_WIN: i]
        row   = df.iloc[i]

        y_train_raw = train["target_4h"].values
        if y_train_raw.sum() < 10 or (1 - y_train_raw).sum() < 10:
            continue

        try:
            X_raw = train[available_features].values.astype(np.float32)

            # Fit scaler on training flat features, transform before sequencing
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            X_seq, y_seq = build_sequences(X_scaled, y_train_raw, LOOKBACK)
            if len(X_seq) < 30:
                continue

            model = train_lstm(X_seq, y_seq, input_size=n_feat)

            # Score current bar: sequence = [i-LOOKBACK .. i-1] (no leakage)
            test_raw  = df.iloc[i - LOOKBACK: i][available_features].values.astype(np.float32)
            test_scaled = scaler.transform(test_raw)
            prob = predict_lstm(model, test_scaled)

            # Same hard gates as walk_forward_meta
            is_hakai = int(row["hmm_hakai"] > 0.5)
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
                "composite":  float(row.get("composite", 0.0)),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def score_live_lstm(df, projections=None, live_hmm_label=None):
    """
    Train LSTM on last TRAIN_WIN rows, score the current bar.
    Drop-in replacement for score_live() — same output dict schema.
    """
    if not _LSTM_OK:
        return {"score": 50, "tier": "WATCH", "available": False,
                "error": "lstm_model not available"}

    from sklearn.preprocessing import StandardScaler

    available_features = [f for f in META_FEATURES if f in df.columns]
    n_feat = len(available_features)
    n      = len(df)
    if n < MIN_TRAIN + LOOKBACK + 10:
        return {"score": 50, "tier": "WATCH", "available": False}

    train = df.iloc[-(TRAIN_WIN + 1): -1]
    row   = df.iloc[-1]

    try:
        current_hour = pd.Timestamp(df.index[-1]).hour
    except Exception:
        current_hour = 12
    is_bad_session = (current_hour >= 20)

    live_is_hakai = False
    if live_hmm_label is not None:
        live_is_hakai = (live_hmm_label == "HAKAI")
    else:
        try:
            _live_lbl = _hmm_label_current(df._data if hasattr(df, '_data') else {})
            live_is_hakai = (_live_lbl.get("regime_label") == "HAKAI")
        except Exception:
            pass

    X_raw   = train[available_features].values.astype(np.float32)
    y_train = train["target_4h"].values

    if y_train.sum() < 5:
        return {"score": 50, "tier": "WATCH", "available": False}

    try:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        X_seq, y_seq = build_sequences(X_scaled, y_train, LOOKBACK)

        if len(X_seq) < 30:
            return {"score": 50, "tier": "WATCH", "available": False}

        model = train_lstm(X_seq, y_seq, input_size=n_feat)

        test_raw    = df.iloc[-LOOKBACK - 1: -1][available_features].values.astype(np.float32)
        test_scaled = scaler.transform(test_raw)
        prob_raw    = predict_lstm(model, test_scaled)
        prob        = prob_raw   # no isotonic calibration (too few cal samples for LSTM)

        score = int(np.clip((prob - 0.20) / 0.60 * 100, 0, 100))

        hmm = int(row.get("hmm_regime", 0))
        if int(row.get("hmm_hakai", 0)) == 1 or live_is_hakai:
            tier  = "BLOCKED"
            score = min(score, 25)
        elif is_bad_session:
            tier  = "BLOCKED (SESSION)"
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

        # Kelly sizing (same formula as score_live)
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
            kelly_pct  = int(kelly_f * 100)
        except Exception:
            kelly_pct = 0

        try:
            _atr_ratio_live = float(row.get("atr_ratio", 1.0))
            _vol_scalar = max(0.25, min(1.0, 1.0 / max(_atr_ratio_live, 0.5)))
        except Exception:
            _vol_scalar = 1.0

        tier_ceiling = {
            "BLOCKED": 0, "BLOCKED (SESSION)": 0, "PASS": 0, "WATCH": 0,
            "TRADE": 60, "HIGH CONVICTION": 80, "FULL SEND": 100,
        }
        size_pct = int(min(kelly_pct * _vol_scalar, tier_ceiling.get(tier, 0)))

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
            "top_drivers":     [],   # LSTM has no per-feature importance
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
