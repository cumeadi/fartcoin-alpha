"""
systematic_backtest.py — Backtest systematic rule-based signals

Three signals:
  1. Settlement Fade  — SHORT 2h before 4h settlement when funding is elevated
  2. Funding Spike Reversion — SHORT when funding z-score > threshold
  3. Ghost Long — LONG when Binance-Bybit spread velocity > 2σ (mini test)

Run:
  python3 systematic_backtest.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
CARRY_1H  = 0.0045 / 4   # carry cost per 1h   (≈ 0.001125%)
CARRY_2H  = 0.0045 / 2   # carry cost per 2h
CARRY_4H  = 0.0045        # carry cost per 4h


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlcv() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "FARTCOIN_ohlcv_hourly.csv",
                     index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)
    return df


def _load_funding() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "FARTCOINUSDT_funding.csv",
                     index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)
    return df


def _score(df: pd.DataFrame, name: str, direction: str,
           hold_h: int, carry: float) -> pd.DataFrame:
    """
    Print backtest summary. Expects df with:
      signal (0/1), fwd_ret (actual price return over hold_h in the
                             direction of the trade, sign-adjusted),
      timestamp index
    """
    trades = df[df["signal"] == 1].copy()
    n      = len(trades)
    n_all  = len(df)

    if n == 0:
        print(f"\n{name}: no signals fired.")
        return trades

    baseline_hit = (df["fwd_ret_raw"] > 0).mean()
    hit          = (trades["fwd_ret"] > carry).mean()   # direction-adjusted
    avg_ret      = trades["fwd_ret"].mean()
    wins         = trades[trades["fwd_ret"] > carry]["fwd_ret"]
    losses       = trades[trades["fwd_ret"] <= carry]["fwd_ret"]
    avg_win      = wins.mean()  if len(wins)  > 0 else 0
    avg_loss     = losses.mean() if len(losses) > 0 else 0
    wl_ratio     = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    bars_per_yr  = 365 * 24 / hold_h
    excess       = trades["fwd_ret"] - carry
    sharpe       = (excess.mean() / excess.std() * np.sqrt(bars_per_yr)
                    if excess.std() > 0 else 0)

    # Monthly breakdown
    trades["ym"] = trades.index.to_period("M")
    monthly = trades.groupby("ym").apply(
        lambda g: pd.Series({
            "n":    len(g),
            "hit":  (g["fwd_ret"] > carry).mean(),
        })
    )

    trades_per_month = n / max((df.index.max() - df.index.min()).days / 30, 1)

    print(f"\n{'='*72}")
    print(f"  {name}  [{direction} | {hold_h}h hold]")
    print(f"  {n_all} bars  |  carry cost: {carry*100:.3f}%/{hold_h}h  |  "
          f"~{trades_per_month:.0f} trades/month")
    print(f"{'='*72}")
    print(f"  Baseline hit rate (all bars):   {baseline_hit:.1%}  (n={n_all})")
    print(f"  Signal fires:                   {n/n_all:.1%} of bars (n={n})")
    print(f"  Hit rate (direction-adjusted):  {hit:.1%}")
    print(f"  Lift over baseline:             {(hit - baseline_hit)*100:+.1f}pp")
    print(f"  Avg return per trade:           {avg_ret*100:+.3f}%")
    print(f"  Avg win / Avg loss:             {avg_win*100:+.3f}% / {avg_loss*100:+.3f}%")
    print(f"  W/L ratio:                      {wl_ratio:.2f}x")
    print(f"  Annualised Sharpe:              {sharpe:+.2f}")
    print(f"\n  ── Monthly Performance ──")
    for period, row in monthly.iterrows():
        bar  = "█" * int(row["hit"] * 20)
        print(f"  {period}   {row['hit']:.1%}  {bar:<20}  (n={int(row['n'])})")
    print(f"{'='*72}")

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# 1. Settlement Fade
# ─────────────────────────────────────────────────────────────────────────────

def backtest_settlement_fade(
    funding_threshold_pct: float = 0.0,   # any positive funding
    entry_offset_h: int = 2,              # enter N hours before settlement
    min_funding_pct: float = 0.005,       # ignore floor (0.5% = Bybit floor)
) -> pd.DataFrame:
    """
    SHORT {entry_offset_h}h before each 4h funding settlement when
    funding rate > min_funding_pct.

    Logic:
      - funding rate > min_funding_pct  → longs crowded, will close before paying
      - Enter SHORT at close of (settlement_time - entry_offset_h)
      - Exit at close of settlement_time
      - P&L = -(price change) since we're short
    """
    ohlcv   = _load_ohlcv()
    funding = _load_funding()

    close   = ohlcv["close"]
    results = []

    for ts, row in funding.iterrows():
        rate = float(row["fundingRate"])

        # Gate: only trade elevated positive funding
        # Ignore Bybit floor (always 0.005) — only real price signal
        # above floor
        if rate <= min_funding_pct:
            continue

        entry_ts = ts - pd.Timedelta(hours=entry_offset_h)

        if entry_ts not in close.index or ts not in close.index:
            continue

        price_entry = float(close.loc[entry_ts])
        price_exit  = float(close.loc[ts])

        if price_entry <= 0:
            continue

        raw_ret    = (price_exit - price_entry) / price_entry   # raw price change
        trade_ret  = -raw_ret                                    # SHORT = invert

        results.append({
            "timestamp":    entry_ts,
            "settle_ts":    ts,
            "funding_rate": rate,
            "price_entry":  price_entry,
            "price_exit":   price_exit,
            "fwd_ret_raw":  raw_ret,
            "fwd_ret":      trade_ret,   # direction-adjusted
            "signal":       1,
        })

    df = pd.DataFrame(results).set_index("timestamp") if results else pd.DataFrame()

    # Build "all bars" baseline using all ohlcv bars
    all_df = pd.DataFrame({
        "fwd_ret_raw": close.pct_change(entry_offset_h).shift(-entry_offset_h),
        "signal":      0,
    }, index=ohlcv.index).dropna()

    # For baseline hit rate context
    all_df["fwd_ret"] = all_df["fwd_ret_raw"]  # long-biased baseline

    if df.empty:
        print("Settlement Fade: no trades (check threshold)")
        return all_df

    # Attach signal to full bar df
    all_df = all_df.copy()
    all_df.loc[all_df.index.isin(df.index), "signal"] = 1
    all_df.loc[all_df.index.isin(df.index), "fwd_ret"] = df["fwd_ret"].reindex(
        all_df.index[all_df.index.isin(df.index)]
    ).values

    return _score(df, "Settlement Fade (SHORT into 4h settlement)",
                  "SHORT", entry_offset_h, CARRY_2H)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Funding Spike Reversion
# ─────────────────────────────────────────────────────────────────────────────

def backtest_funding_spike(
    z_threshold: float = 1.5,
    hold_h: int = 4,
    entry_delay_h: int = 1,   # wait 1h after spike to let it confirm
    roll_window: int = 168,   # 7-day rolling window for z-score
) -> pd.DataFrame:
    """
    SHORT when funding rate z-score (rolling {roll_window}h) > z_threshold.

    Intuition: extreme positive funding = longs very crowded.  Mean reversion
    tends to follow.  Entry is delayed by {entry_delay_h}h to let the spike
    confirm rather than catching the top of a legitimate move.
    """
    ohlcv   = _load_ohlcv()
    funding = _load_funding()

    # Reindex funding to hourly OHLCV timestamps (forward-fill)
    funding_h = funding["fundingRate"].reindex(ohlcv.index).ffill().fillna(0)

    # Rolling z-score
    roll_mean = funding_h.rolling(roll_window, min_periods=48).mean()
    roll_std  = funding_h.rolling(roll_window, min_periods=48).std().clip(lower=1e-9)
    funding_z = (funding_h - roll_mean) / roll_std

    close  = ohlcv["close"]
    # Forward return over hold_h (LONG direction; we invert for SHORT)
    fwd_ret_raw = close.shift(-hold_h) / close - 1

    results = []
    for ts in ohlcv.index:
        z = funding_z.loc[ts]
        if z < z_threshold:
            continue
        if pd.isna(z) or pd.isna(fwd_ret_raw.loc[ts]):
            continue

        # Entry delay: use the bar entry_delay_h hours after the spike
        entry_ts = ts + pd.Timedelta(hours=entry_delay_h)
        if entry_ts not in close.index:
            continue
        exit_ts = entry_ts + pd.Timedelta(hours=hold_h)
        if exit_ts not in close.index:
            continue

        raw_ret   = (close.loc[exit_ts] - close.loc[entry_ts]) / close.loc[entry_ts]
        trade_ret = -raw_ret  # SHORT

        results.append({
            "timestamp":  ts,
            "entry_ts":   entry_ts,
            "funding_z":  z,
            "funding_rate": float(funding_h.loc[ts]),
            "fwd_ret_raw":  raw_ret,
            "fwd_ret":      trade_ret,
            "signal":       1,
        })

    df = pd.DataFrame(results).set_index("timestamp") if results else pd.DataFrame()

    # Baseline
    baseline = pd.DataFrame({
        "fwd_ret_raw": fwd_ret_raw,
        "signal":      0,
    }, index=ohlcv.index).dropna()
    baseline["fwd_ret"] = baseline["fwd_ret_raw"]

    if df.empty:
        print(f"Funding Spike Reversion (z>{z_threshold}): no trades")
        return baseline

    # Sensitivity sweep across z thresholds
    print(f"\n  Sensitivity: z_threshold sweep")
    print(f"  {'z>':>6}  {'Trades':>7}  {'Hit%':>7}  {'AvgRet':>8}  {'Sharpe':>7}")
    for z_thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        sub = df[df["funding_z"] >= z_thr]
        if len(sub) < 5:
            break
        hit = (sub["fwd_ret"] > CARRY_4H).mean()
        avg = sub["fwd_ret"].mean()
        exc = sub["fwd_ret"] - CARRY_4H
        sh  = (exc.mean() / exc.std() * np.sqrt(365 * 24 / hold_h)
               if exc.std() > 0 else 0)
        print(f"  {z_thr:>6.1f}  {len(sub):>7}  {hit:>7.1%}  {avg*100:>+7.3f}%  {sh:>+7.2f}")

    return _score(df, f"Funding Spike Reversion (z>{z_threshold}, {entry_delay_h}h delay)",
                  "SHORT", hold_h, CARRY_4H)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ghost Long (cross-exchange spread) — mini validation
# ─────────────────────────────────────────────────────────────────────────────

def backtest_ghost_long(
    velocity_z_threshold: float = 1.5,
    hold_h: int = 4,
) -> pd.DataFrame:
    """
    LONG when Binance-Bybit funding spread velocity > velocity_z_threshold.

    Data: coinalyze_funding_history.csv (col A = Binance, col 6 = Bybit).
    NOTE: Only ~168 hours of data (7 days). Results are directional
    validation only — flag clearly as preliminary.
    """
    ohlcv = _load_ohlcv()

    cx_path = DATA_DIR / "coinalyze_funding_history.csv"
    if not cx_path.exists():
        print("Ghost Long: coinalyze_funding_history.csv not found — skipping")
        return pd.DataFrame()

    cx = pd.read_csv(cx_path, parse_dates=["timestamp"])
    cx["timestamp"] = cx["timestamp"].dt.tz_localize(None)
    cx = cx.set_index("timestamp").sort_index()

    if "A" not in cx.columns or "6" not in cx.columns:
        print("Ghost Long: expected columns A (Binance) and 6 (Bybit) not found")
        return pd.DataFrame()

    cx["spread"]    = cx["A"] - cx["6"]   # Binance - Bybit
    cx["velocity"]  = cx["spread"].diff()

    roll_v_mean = cx["velocity"].rolling(24, min_periods=6).mean()
    roll_v_std  = cx["velocity"].rolling(24, min_periods=6).std().clip(lower=1e-12)
    cx["vel_z"] = (cx["velocity"] - roll_v_mean) / roll_v_std

    close   = ohlcv["close"]
    results = []

    for ts, row in cx.iterrows():
        if row["vel_z"] < velocity_z_threshold:
            continue
        if ts not in close.index:
            # Nearest hourly
            nearest = close.index.asof(ts)
            if pd.isna(nearest):
                continue
            ts_entry = nearest
        else:
            ts_entry = ts

        ts_exit = ts_entry + pd.Timedelta(hours=hold_h)
        if ts_exit not in close.index:
            continue

        raw_ret = (close.loc[ts_exit] - close.loc[ts_entry]) / close.loc[ts_entry]

        results.append({
            "timestamp":  ts_entry,
            "spread":     float(row["spread"]),
            "vel_z":      float(row["vel_z"]),
            "fwd_ret_raw": raw_ret,
            "fwd_ret":     raw_ret,   # LONG: no inversion
            "signal":      1,
        })

    if not results:
        print(f"Ghost Long (vel_z>{velocity_z_threshold}): no trades in available data")
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("timestamp")

    # Baseline from the same 7-day window
    window_start = cx.index.min()
    window_end   = cx.index.max()
    baseline_close = close.loc[window_start:window_end]
    baseline_fwd   = baseline_close.shift(-hold_h) / baseline_close - 1

    print(f"\n{'='*72}")
    print(f"  Ghost Long (LONG | {hold_h}h hold)  ⚠ PRELIMINARY — 7-day sample")
    print(f"  {len(cx)} hourly obs | {len(df)} trades | range: "
          f"{window_start.date()} → {window_end.date()}")
    print(f"{'='*72}")
    bh = (baseline_fwd > 0).mean()
    hit = (df["fwd_ret"] > CARRY_4H).mean()
    avg = df["fwd_ret"].mean()
    print(f"  Baseline hit (this window):     {bh:.1%}  (n={len(baseline_fwd.dropna())})")
    print(f"  Signal fires:                   {len(df)/len(cx):.1%} of bars (n={len(df)})")
    print(f"  Hit rate when trading:          {hit:.1%}")
    print(f"  Lift over baseline:             {(hit - bh)*100:+.1f}pp")
    print(f"  Avg return per trade:           {avg*100:+.3f}%")
    print(f"\n  ⚠  7-day sample is not statistically significant.")
    print(f"     Directional signal looks {'POSITIVE' if hit > 0.55 else 'WEAK'}.")
    print(f"     Recommend 3+ months of continuous collection before trading live.")
    print(f"{'='*72}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*72)
    print("  SYSTEMATIC SIGNALS BACKTEST")
    print("  Fartcoin Alpha Framework")
    print("="*72)

    print("\n\n>>> SIGNAL 1: SETTLEMENT FADE")
    print("    Entry threshold: funding > Bybit floor (0.005% = 0.5%/8h)")
    sf = backtest_settlement_fade(min_funding_pct=0.0051)

    print("\n\n>>> SIGNAL 2: FUNDING SPIKE REVERSION")
    print("    Entry: 1h after spike confirms | Hold: 4h")
    fsr = backtest_funding_spike(z_threshold=1.5, hold_h=4, entry_delay_h=1)

    print("\n\n>>> SIGNAL 3: GHOST LONG (preliminary)")
    print("    Binance-Bybit spread velocity > 1.5σ | Hold: 4h")
    gl = backtest_ghost_long(velocity_z_threshold=1.5, hold_h=4)

    # Combined summary
    print("\n\n" + "="*72)
    print("  COMBINED SIGNAL SUMMARY")
    print("="*72)
    rows = []
    if not sf.empty:
        hit = (sf["fwd_ret"] > CARRY_2H).mean()
        exc = sf["fwd_ret"] - CARRY_2H
        sh  = exc.mean() / exc.std() * np.sqrt(365 * 24 / 2) if exc.std() > 0 else 0
        total_days = max((sf.index.max() - sf.index.min()).days, 1)
        rows.append(("Settlement Fade", len(sf), f"{len(sf)/(total_days/30):.0f}/mo",
                     f"{hit:.1%}", f"{sf['fwd_ret'].mean()*100:+.3f}%", f"{sh:+.2f}"))
    if not fsr.empty:
        hit = (fsr["fwd_ret"] > CARRY_4H).mean()
        exc = fsr["fwd_ret"] - CARRY_4H
        sh  = exc.mean() / exc.std() * np.sqrt(365 * 24 / 4) if exc.std() > 0 else 0
        total_days = max((fsr.index.max() - fsr.index.min()).days, 1)
        rows.append(("Funding Spike Reversion", len(fsr), f"{len(fsr)/(total_days/30):.0f}/mo",
                     f"{hit:.1%}", f"{fsr['fwd_ret'].mean()*100:+.3f}%", f"{sh:+.2f}"))
    if not gl.empty:
        hit = (gl["fwd_ret"] > CARRY_4H).mean()
        rows.append(("Ghost Long ⚠ prelim.", len(gl), "N/A",
                     f"{hit:.1%}", f"{gl['fwd_ret'].mean()*100:+.3f}%", "N/A"))

    if rows:
        print(f"  {'Signal':<28} {'Trades':>7} {'Freq':>7} {'Hit%':>7} {'AvgRet':>8} {'Sharpe':>8}")
        print(f"  {'-'*68}")
        for r in rows:
            print(f"  {r[0]:<28} {r[1]:>7} {r[2]:>7} {r[3]:>7} {r[4]:>8} {r[5]:>8}")
    print("="*72)
