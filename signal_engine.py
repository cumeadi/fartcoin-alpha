"""
Signal Engine — Fartcoin Alpha Framework

Computes manipulation-detection signals from raw market data.
Each signal returns a normalized score [-1, +1] where:
  +1 = strong bullish manipulation signal (price likely to pump)
  -1 = strong bearish manipulation signal (price likely to dump)
   0 = neutral / no signal

The composite score combines all signals into a single actionable metric.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_data(perp_symbol="FARTCOINUSDT", cmc_symbol="FARTCOIN",
              cg_coin_id="fartcoin"):
    """Load all CSVs into a dict of DataFrames. Prefer hourly CG data."""
    data = {}

    # Prefer CoinGecko hourly chart over CMC daily
    cg_hourly = DATA_DIR / f"{cmc_symbol}_ohlcv_hourly.csv"
    cmc_daily = DATA_DIR / f"{cmc_symbol}_ohlcv.csv"
    if cg_hourly.exists():
        df = pd.read_csv(cg_hourly, index_col=0, parse_dates=True)
        data["ohlcv"] = df
        print(f"  Loaded ohlcv (CG hourly): {len(df)} rows")
    elif cmc_daily.exists():
        df = pd.read_csv(cmc_daily, index_col=0, parse_dates=True)
        data["ohlcv"] = df
        print(f"  Loaded ohlcv (CMC daily): {len(df)} rows")

    # Also keep CMC daily for daily-level analysis
    if cmc_daily.exists():
        data["ohlcv_daily"] = pd.read_csv(cmc_daily, index_col=0, parse_dates=True)

    # Perps signals (synthetic or real)
    other_files = {
        "funding": DATA_DIR / f"{perp_symbol}_funding.csv",
        "oi": DATA_DIR / f"{perp_symbol}_oi.csv",
        "lsr": DATA_DIR / f"{perp_symbol}_lsr.csv",
        "taker": DATA_DIR / f"{perp_symbol}_taker.csv",
    }
    for key, path in other_files.items():
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            data[key] = df
            print(f"  Loaded {key}: {len(df)} rows")
        else:
            print(f"  MISSING: {path.name}")

    # Derivatives snapshot (cross-exchange)
    deriv_file = DATA_DIR / f"{cmc_symbol}_derivatives_snapshot.csv"
    if deriv_file.exists():
        data["derivatives"] = pd.read_csv(deriv_file)
        print(f"  Loaded derivatives snapshot: {len(data['derivatives'])} tickers")

    # Derivatives history (accumulated polls)
    hist_file = DATA_DIR / "derivatives_history.csv"
    if hist_file.exists():
        data["derivatives_history"] = pd.read_csv(hist_file, parse_dates=["timestamp"])
        print(f"  Loaded derivatives history: {len(data['derivatives_history'])} snapshots")

    return data


# ---------------------------------------------------------------------------
# Signal 1: Funding Rate Extremes
# ---------------------------------------------------------------------------
# When funding is extremely positive, longs are paying shorts heavily.
# This means the market is overcrowded long → squeeze incoming → SHORT signal.
# When funding is extremely negative → overcrowded short → LONG signal.
#
# MM playbook: Push price up while funding is deeply negative (free money
# from shorts paying you). Or push price down when funding is deeply positive.

def signal_funding_rate(funding_df, lookback=20, z_threshold=1.5):
    """
    Score based on z-score of current funding rate vs recent history.
    Extreme positive funding → bearish (return negative score).
    Extreme negative funding → bullish (return positive score).
    """
    if funding_df is None or funding_df.empty:
        return pd.Series(dtype=float)

    fr = funding_df["fundingRate"]
    rolling_mean = fr.rolling(lookback).mean()
    rolling_std = fr.rolling(lookback).std()
    z_score = (fr - rolling_mean) / rolling_std.replace(0, np.nan)

    # Invert: high funding = bearish signal, low funding = bullish signal
    signal = -z_score.clip(-3, 3) / 3  # normalize to [-1, 1]
    signal.name = "sig_funding"
    return signal


# ---------------------------------------------------------------------------
# Signal 2: Open Interest Divergence
# ---------------------------------------------------------------------------
# OI rising + price flat = positions being built before a move.
# OI rising + price rising = trend confirmation (less useful for manipulation).
# OI rising + price falling = shorts building → potential squeeze up.
#
# The key divergence: OI changes significantly while price doesn't.

def signal_oi_divergence(oi_df, ohlcv_df, lookback=24):
    """
    Score based on OI rate of change vs price rate of change.
    High OI change + low price change = manipulation setup.
    Direction inferred from funding/LSR context.
    """
    if oi_df is None or ohlcv_df is None:
        return pd.Series(dtype=float)

    oi = oi_df["sumOpenInterestValue"].resample("1h").last().ffill()
    # Resample price to hourly to match OI
    price = ohlcv_df["close"].resample("1h").last().ffill()

    # Align indices
    common = oi.index.intersection(price.index)
    if len(common) < lookback + 1:
        return pd.Series(dtype=float)

    oi = oi.loc[common]
    price = price.loc[common]

    oi_pct = oi.pct_change(lookback).fillna(0)
    price_pct = price.pct_change(lookback).fillna(0).abs()

    # Divergence: OI moved a lot, price didn't
    # Higher ratio = more "hidden" positioning
    divergence = (oi_pct.abs() / (price_pct + 0.001))  # avoid div/0

    # Normalize using percentile rank
    signal = divergence.rank(pct=True) * 2 - 1  # map [0,1] to [-1,1]
    signal.name = "sig_oi_divergence"
    return signal


# ---------------------------------------------------------------------------
# Signal 3: OI Acceleration
# ---------------------------------------------------------------------------
# Not just is OI rising, but is it ACCELERATING? Sudden spikes in OI
# often precede manufactured moves.

def signal_oi_acceleration(oi_df, lookback=12):
    """Rate of change of the rate of change of OI. Spikes = positioning."""
    if oi_df is None or oi_df.empty:
        return pd.Series(dtype=float)

    oi = oi_df["sumOpenInterestValue"]
    oi_roc = oi.pct_change(lookback)
    oi_accel = oi_roc.diff(lookback)

    z = (oi_accel - oi_accel.rolling(48).mean()) / oi_accel.rolling(48).std().replace(0, np.nan)
    signal = z.clip(-3, 3) / 3
    signal.name = "sig_oi_accel"
    return signal


# ---------------------------------------------------------------------------
# Signal 4: Long/Short Ratio Extremes
# ---------------------------------------------------------------------------
# When top traders are overwhelmingly long → contrarian short signal.
# When overwhelmingly short → contrarian long signal.
# MMs know where retail is positioned and will hunt those stops.

def signal_lsr_extreme(lsr_df, lookback=48, z_threshold=1.5):
    """Contrarian signal from top-trader long/short ratio."""
    if lsr_df is None or lsr_df.empty:
        return pd.Series(dtype=float)

    ratio = lsr_df["longShortRatio"]
    z = (ratio - ratio.rolling(lookback).mean()) / ratio.rolling(lookback).std().replace(0, np.nan)

    # Contrarian: high LSR (everyone long) → bearish, low LSR → bullish
    signal = -z.clip(-3, 3) / 3
    signal.name = "sig_lsr"
    return signal


# ---------------------------------------------------------------------------
# Signal 5: Taker Aggression Imbalance
# ---------------------------------------------------------------------------
# Taker buy/sell ratio shows who is aggressively crossing the spread.
# Sudden shifts in taker aggression often precede manufactured moves.

def signal_taker_imbalance(taker_df, lookback=24):
    """Taker buy/sell ratio deviation from mean."""
    if taker_df is None or taker_df.empty:
        return pd.Series(dtype=float)

    ratio = taker_df["buySellRatio"]
    z = (ratio - ratio.rolling(lookback).mean()) / ratio.rolling(lookback).std().replace(0, np.nan)

    signal = z.clip(-3, 3) / 3  # positive = aggressive buying
    signal.name = "sig_taker"
    return signal


# ---------------------------------------------------------------------------
# Signal 6: Volume Spike Anomaly
# ---------------------------------------------------------------------------
# Volume spikes on meme coins that DON'T correspond to broader market moves
# are suspicious. A 5x volume day on Fartcoin while BTC is flat = someone
# is engineering liquidity.

def signal_volume_spike(ohlcv_df, lookback=20, spike_threshold=2.5):
    """Z-score of volume vs rolling average. High = suspicious activity."""
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.Series(dtype=float)

    vol = ohlcv_df["volume"]
    vol_mean = vol.rolling(lookback).mean()
    vol_std = vol.rolling(lookback).std()
    z = (vol - vol_mean) / vol_std.replace(0, np.nan)

    # Volume spikes are directionally ambiguous — just flag magnitude
    signal = z.clip(0, 3) / 3  # 0 to 1 (no negative; spike = notable)
    signal.name = "sig_volume_spike"
    return signal


# ---------------------------------------------------------------------------
# Signal 7: Price-Volume Divergence
# ---------------------------------------------------------------------------
# Price rising on declining volume = weak rally, likely to reverse.
# Price falling on declining volume = weak selloff, likely to bounce.
# This catches the "manufactured move on thin liquidity" pattern.

def signal_price_volume_divergence(ohlcv_df, lookback=10):
    """Correlation between price change and volume change over lookback."""
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.Series(dtype=float)

    price_chg = ohlcv_df["close"].pct_change()
    vol_chg = ohlcv_df["volume"].pct_change()

    rolling_corr = price_chg.rolling(lookback).corr(vol_chg)

    # Negative correlation (price up + volume down) = suspicious
    # Map: -1 corr → signal +1 (manipulation likely), +1 corr → signal -1
    signal = -rolling_corr.fillna(0)
    signal.name = "sig_pv_divergence"
    return signal


# ---------------------------------------------------------------------------
# Composite Signal
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "sig_funding": 0.20,       # Funding rate extremes
    "sig_oi_divergence": 0.20, # OI building without price move
    "sig_oi_accel": 0.15,      # OI acceleration spikes
    "sig_lsr": 0.15,           # Contrarian long/short
    "sig_taker": 0.15,         # Taker aggression
    "sig_volume_spike": 0.05,  # Volume anomalies (directionally ambiguous)
    "sig_pv_divergence": 0.10, # Price-volume divergence
}


def compute_composite(signals_df, weights=None):
    """
    Weighted composite of all signals.
    Returns a score in [-1, 1] where:
      > +0.5  = strong long setup (MMs likely to push up)
      < -0.5  = strong short setup (MMs likely to push down)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    composite = pd.Series(0.0, index=signals_df.index)
    for col, w in weights.items():
        if col in signals_df.columns:
            composite += signals_df[col].fillna(0) * w

    composite.name = "composite"
    return composite


def compute_all_signals(data):
    """Run all signal functions on loaded data, return combined DataFrame."""
    signals = []

    if "funding" in data:
        signals.append(signal_funding_rate(data["funding"]))

    if "oi" in data and "ohlcv" in data:
        signals.append(signal_oi_divergence(data["oi"], data["ohlcv"]))

    if "oi" in data:
        signals.append(signal_oi_acceleration(data["oi"]))

    if "lsr" in data:
        signals.append(signal_lsr_extreme(data["lsr"]))

    if "taker" in data:
        signals.append(signal_taker_imbalance(data["taker"]))

    if "ohlcv" in data:
        signals.append(signal_volume_spike(data["ohlcv"]))
        signals.append(signal_price_volume_divergence(data["ohlcv"]))

    # Combine all signals on a common index
    signals = [s for s in signals if not s.empty]
    if not signals:
        print("No signals computed — check your data.")
        return pd.DataFrame()

    df = pd.concat(signals, axis=1)
    df["composite"] = compute_composite(df)
    return df


# ---------------------------------------------------------------------------
# Entry/Exit Logic
# ---------------------------------------------------------------------------

def generate_trades(signals_df, entry_threshold=0.4, exit_threshold=0.1,
                    min_hold_periods=4):
    """
    Simple threshold-based trade generation.

    LONG entry:  composite > +entry_threshold
    SHORT entry: composite < -entry_threshold
    EXIT:        |composite| < exit_threshold OR signal flips

    Returns DataFrame of trades with entry/exit times and signals.
    """
    trades = []
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_time = None
    entry_score = None
    hold_count = 0

    for ts, row in signals_df.iterrows():
        score = row["composite"]
        if np.isnan(score):
            continue

        if position == 0:
            if score > entry_threshold:
                position = 1
                entry_time = ts
                entry_score = score
                hold_count = 0
            elif score < -entry_threshold:
                position = -1
                entry_time = ts
                entry_score = score
                hold_count = 0
        else:
            hold_count += 1
            should_exit = (
                hold_count >= min_hold_periods
                and (
                    abs(score) < exit_threshold
                    or (position == 1 and score < 0)
                    or (position == -1 and score > 0)
                )
            )
            if should_exit:
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_score": entry_score,
                    "exit_score": score,
                    "hold_periods": hold_count,
                })
                position = 0

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()

    print("\nComputing signals...")
    signals = compute_all_signals(data)

    if not signals.empty:
        signals.to_csv(DATA_DIR / "signals.csv")
        print(f"\nSignals computed: {len(signals)} rows")
        print(f"Columns: {list(signals.columns)}")
        print(f"\nComposite score stats:")
        print(signals["composite"].describe())

        print("\n--- Recent Signal Snapshot ---")
        print(signals.tail(10).round(3))

        print("\n--- Generated Trades ---")
        trades = generate_trades(signals)
        if not trades.empty:
            trades.to_csv(DATA_DIR / "trades.csv", index=False)
            print(trades.to_string(index=False))
            print(f"\nTotal trades: {len(trades)}")
        else:
            print("No trades generated with current thresholds.")
    else:
        print("No signals to compute. Run data_collector.py first.")
