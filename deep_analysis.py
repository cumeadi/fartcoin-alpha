"""
Deep Analysis — Fartcoin Alpha Framework

Combines:
1. CoinGecko hourly price/volume (2161 data points)
2. CMC daily OHLCV with real high/low/open/close (89 days)
3. Real derivatives snapshot (68 tickers across exchanges)
4. Synthetic perps signals calibrated against real current values

This is the main analysis script — runs all tests and produces the report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from signal_engine import load_data, compute_all_signals

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all():
    """Load every dataset we have."""
    data = load_data()

    # Also load CMC daily for OHLCV analysis (has real candles)
    cmc_path = DATA_DIR / "FARTCOIN_ohlcv.csv"
    if cmc_path.exists():
        data["ohlcv_daily"] = pd.read_csv(cmc_path, index_col=0, parse_dates=True)
        print(f"  Loaded CMC daily OHLCV: {len(data['ohlcv_daily'])} rows")

    return data


# ---------------------------------------------------------------------------
# Analysis 1: Cross-Exchange Manipulation Signals (from real derivatives data)
# ---------------------------------------------------------------------------

def analyze_derivatives_snapshot(deriv_df):
    """Deep analysis of the 68-ticker derivatives snapshot."""
    print("\n" + "=" * 70)
    print("CROSS-EXCHANGE DERIVATIVES ANALYSIS")
    print("=" * 70)

    # Filter to meaningful exchanges (OI > $10k)
    active = deriv_df[deriv_df["open_interest_usd"] > 10000].copy()
    print(f"\nActive exchanges (OI > $10k): {len(active)} of {len(deriv_df)} total")

    # --- Funding Rate Analysis ---
    print("\n--- FUNDING RATE BY EXCHANGE ---")
    fr_sorted = active.sort_values("funding_rate", ascending=False)
    print(fr_sorted[["exchange", "funding_rate", "open_interest_usd", "volume_24h_usd"]].to_string(index=False))

    avg_fr = active["funding_rate"].mean()
    print(f"\nWeighted avg funding rate: {avg_fr:.6f}")
    print(f"Funding rate interpretation:")
    if avg_fr > 0.01:
        print(f"  >> HEAVILY POSITIVE ({avg_fr:.4f}) — Longs paying shorts.")
        print(f"  >> Crowded long trade. HIGH probability of SHORT squeeze setup by MMs.")
        print(f"  >> Signal: BEARISH (contrarian)")
    elif avg_fr < -0.01:
        print(f"  >> HEAVILY NEGATIVE ({avg_fr:.4f}) — Shorts paying longs.")
        print(f"  >> Crowded short trade. HIGH probability of LONG squeeze by MMs.")
        print(f"  >> Signal: BULLISH (contrarian)")
    else:
        print(f"  >> NEUTRAL ({avg_fr:.4f}) — No extreme crowding.")

    # --- OI Concentration (where is the manipulation likely happening?) ---
    print("\n--- OPEN INTEREST CONCENTRATION ---")
    total_oi = active["open_interest_usd"].sum()
    active["oi_share"] = active["open_interest_usd"] / total_oi * 100
    oi_sorted = active.sort_values("open_interest_usd", ascending=False)
    print(oi_sorted[["exchange", "open_interest_usd", "oi_share"]].head(10).to_string(index=False))
    print(f"\nTotal OI across exchanges: ${total_oi:,.0f}")

    # Herfindahl index (market concentration)
    hhi = ((active["oi_share"] / 100) ** 2).sum()
    print(f"OI Herfindahl Index: {hhi:.4f}")
    if hhi > 0.25:
        print("  >> HIGHLY CONCENTRATED — one exchange dominates. Easier to manipulate.")
    elif hhi > 0.15:
        print("  >> MODERATELY CONCENTRATED — a few exchanges dominate.")
    else:
        print("  >> FRAGMENTED — OI spread across many exchanges. Harder to manipulate.")

    # --- Volume vs OI (churning analysis) ---
    print("\n--- VOLUME / OI RATIO (Churning Detection) ---")
    active["vol_oi_ratio"] = active["volume_24h_usd"] / active["open_interest_usd"].replace(0, np.nan)
    vol_oi = active.sort_values("vol_oi_ratio", ascending=False)
    print(vol_oi[["exchange", "volume_24h_usd", "open_interest_usd", "vol_oi_ratio"]].head(10).to_string(index=False))

    avg_ratio = active["volume_24h_usd"].sum() / total_oi
    print(f"\nAggregate Volume/OI ratio: {avg_ratio:.2f}x")
    if avg_ratio > 5:
        print("  >> EXTREMELY HIGH churning — possible wash trading or rapid position turnover.")
    elif avg_ratio > 2:
        print("  >> HIGH activity relative to positions — active trading/speculation.")
    else:
        print("  >> NORMAL — positions are relatively sticky.")

    # --- Basis Spread (perp premium/discount) ---
    print("\n--- BASIS SPREAD (Perp vs Spot) ---")
    basis_sorted = active.sort_values("basis_pct", ascending=False)
    print(basis_sorted[["exchange", "basis_pct", "price", "index_price"]].head(10).to_string(index=False))

    avg_basis = active["basis_pct"].mean()
    print(f"\nAvg basis: {avg_basis:.4f}%")
    if avg_basis > 0.1:
        print("  >> PERP PREMIUM — futures trading above spot. Bullish pressure.")
    elif avg_basis < -0.1:
        print("  >> PERP DISCOUNT — futures trading below spot. Bearish pressure.")
    else:
        print("  >> NEUTRAL basis.")

    # --- Spread Analysis (liquidity fragmentation) ---
    print("\n--- SPREAD ANALYSIS ---")
    spread_sorted = active.sort_values("spread")
    print(spread_sorted[["exchange", "spread", "volume_24h_usd"]].head(10).to_string(index=False))

    return active


# ---------------------------------------------------------------------------
# Analysis 2: Trading Session Analysis (NYC / London / Asia)
# ---------------------------------------------------------------------------

def classify_session(hour):
    """Map UTC hour to trading session."""
    if 0 <= hour < 8:
        return "Asia (00-08 UTC)"
    elif 8 <= hour < 13:
        return "London (08-13 UTC)"
    elif 13 <= hour < 21:
        return "NYC (13-21 UTC)"
    else:
        return "Late NYC / Pre-Asia (21-00 UTC)"


def analyze_by_session(cg_chart):
    """Break down returns, volume, and manipulation patterns by trading session."""
    print("\n" + "=" * 70)
    print("TRADING SESSION ANALYSIS (NYC / LONDON / ASIA)")
    print("=" * 70)

    df = cg_chart.copy()
    price_col = "price" if "price" in df.columns else "close"
    df["return"] = df[price_col].pct_change()
    df["abs_return"] = df["return"].abs()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(24).mean()
    df["hour"] = df.index.hour
    df["session"] = df["hour"].apply(classify_session)

    # --- Session Returns ---
    print("\n--- RETURNS BY SESSION ---")
    session_order = ["Asia (00-08 UTC)", "London (08-13 UTC)",
                     "NYC (13-21 UTC)", "Late NYC / Pre-Asia (21-00 UTC)"]
    stats = df.groupby("session")["return"].agg(["mean", "std", "count"])
    stats["mean_bps"] = stats["mean"] * 10000
    stats["sharpe"] = stats["mean"] / stats["std"] * np.sqrt(365 * 24)
    stats["cum_return"] = df.groupby("session")["return"].sum()
    stats = stats.reindex(session_order)
    print(stats[["mean_bps", "std", "sharpe", "cum_return", "count"]].round(4).to_string())

    best = stats["mean_bps"].idxmax()
    worst = stats["mean_bps"].idxmin()
    print(f"\n  Best session:  {best} ({stats.loc[best, 'mean_bps']:.1f} bps/hr avg)")
    print(f"  Worst session: {worst} ({stats.loc[worst, 'mean_bps']:.1f} bps/hr avg)")

    # --- Volatility by Session ---
    print("\n--- VOLATILITY BY SESSION ---")
    vol_stats = df.groupby("session")["abs_return"].agg(["mean", "median", "max"])
    vol_stats = vol_stats.reindex(session_order)
    vol_stats = vol_stats * 100  # to percent
    vol_stats.columns = ["avg_move_%", "median_move_%", "max_move_%"]
    print(vol_stats.round(4).to_string())

    most_volatile = vol_stats["avg_move_%"].idxmax()
    print(f"\n  Most volatile session: {most_volatile} ({vol_stats.loc[most_volatile, 'avg_move_%']:.3f}% avg move)")

    # --- Big Moves by Session ---
    print("\n--- BIG MOVES (>3%) BY SESSION ---")
    big = df[df["abs_return"] > 0.03]
    if not big.empty:
        session_counts = big.groupby("session").size().reindex(session_order, fill_value=0)
        total_counts = df.groupby("session").size().reindex(session_order)
        freq = (session_counts / total_counts * 100)
        for s in session_order:
            print(f"  {s}: {session_counts.get(s, 0)} big moves ({freq.get(s, 0):.2f}% of hours)")

    # --- Volume by Session ---
    print("\n--- VOLUME BY SESSION ---")
    vol_by_session = df.groupby("session")["volume"].agg(["mean", "sum"])
    vol_by_session["share"] = vol_by_session["sum"] / vol_by_session["sum"].sum() * 100
    vol_by_session = vol_by_session.reindex(session_order)
    print(vol_by_session[["share"]].round(1).to_string())

    # --- Session Transition Moves ---
    print("\n--- SESSION TRANSITION ANALYSIS ---")
    print("(What happens at session boundaries — where MMs hand off)")
    transitions = [
        ("Asia → London", 7, 8, 9),
        ("London → NYC", 12, 13, 14),
        ("NYC → Asia", 20, 21, 22),
    ]
    for name, h_before, h_boundary, h_after in transitions:
        before = df[df["hour"] == h_before]["return"].mean() * 10000
        boundary = df[df["hour"] == h_boundary]["return"].mean() * 10000
        after = df[df["hour"] == h_after]["return"].mean() * 10000
        reversal = "REVERSAL" if np.sign(before) != np.sign(after) else "CONTINUATION"
        print(f"  {name}: {before:+.1f} bps → {boundary:+.1f} bps → {after:+.1f} bps  [{reversal}]")

    return df


# ---------------------------------------------------------------------------
# Analysis 3: Bitcoin Correlation & MM Behavior
# ---------------------------------------------------------------------------

def analyze_btc_correlation(fart_data, btc_path=None):
    """
    How does BTC movement affect MM behavior on Fartcoin?
    Key questions:
    - Do MMs use BTC dumps as cover for Fartcoin manipulation?
    - Does Fartcoin lead or lag BTC?
    - Is the correlation constant or does it break during manipulation?
    - Do Fartcoin moves amplify BTC moves (beta)?
    """
    print("\n" + "=" * 70)
    print("BITCOIN CORRELATION & MM BEHAVIOR ANALYSIS")
    print("=" * 70)

    if btc_path is None:
        btc_path = DATA_DIR / "bitcoin_cg_chart.csv"

    if not btc_path.exists():
        print("  No BTC data available. Skipping.")
        return None

    btc = pd.read_csv(btc_path, index_col=0, parse_dates=True)
    fart = fart_data.copy()
    fart_price_col = "price" if "price" in fart.columns else "close"

    # Align timestamps (merge on nearest hour)
    btc["btc_price"] = btc["price"] if "price" in btc.columns else btc["close"]
    btc["btc_return"] = btc["btc_price"].pct_change()
    btc["btc_volume"] = btc["volume"]

    fart["fart_price"] = fart[fart_price_col]
    fart["fart_return"] = fart["fart_price"].pct_change()

    # Merge on index (both hourly, close enough timestamps)
    # Resample both to hourly to align
    btc_h = btc[["btc_price", "btc_return", "btc_volume"]].resample("1h").last().dropna()
    fart_h = fart[["fart_price", "fart_return"]].resample("1h").last().dropna()
    if "volume" in fart.columns:
        fart_h["fart_volume"] = fart["volume"].resample("1h").last()

    merged = btc_h.join(fart_h, how="inner").dropna(subset=["btc_return", "fart_return"])
    print(f"\n  Merged dataset: {len(merged)} hourly observations")
    print(f"  Period: {merged.index[0]} to {merged.index[-1]}")

    # --- 1. Overall Correlation ---
    print("\n--- OVERALL CORRELATION ---")
    corr = merged["btc_return"].corr(merged["fart_return"])
    rank_corr = merged["btc_return"].rank().corr(merged["fart_return"].rank())
    print(f"  Pearson correlation:  {corr:.4f}")
    print(f"  Spearman (rank):     {rank_corr:.4f}")

    if corr > 0.5:
        print("  >> STRONG positive correlation — Fartcoin follows BTC closely")
    elif corr > 0.3:
        print("  >> MODERATE correlation — moves together but with independence")
    elif corr > 0.1:
        print("  >> WEAK correlation — largely independent")
    else:
        print("  >> NEAR ZERO — Fartcoin decoupled from BTC")

    # --- 2. Beta (amplification) ---
    print("\n--- BETA (Amplification Factor) ---")
    from numpy.polynomial.polynomial import polyfit
    # Simple OLS: fart_return = alpha + beta * btc_return
    valid = merged[["btc_return", "fart_return"]].dropna()
    if len(valid) > 50:
        beta_coeffs = np.polyfit(valid["btc_return"], valid["fart_return"], 1)
        beta = beta_coeffs[0]
        alpha = beta_coeffs[1]
        print(f"  Beta:  {beta:.2f}x (Fartcoin moves {beta:.1f}x BTC on average)")
        print(f"  Alpha: {alpha*10000:.1f} bps/hr (excess return independent of BTC)")
        if beta > 2:
            print("  >> HIGH BETA — Fartcoin amplifies BTC moves significantly")
            print("  >> MMs can use small BTC moves to trigger large Fart moves")
        elif beta > 1:
            print("  >> MODERATE BETA — amplifies BTC but not extreme")

    # --- 3. Rolling Correlation (when does correlation break?) ---
    print("\n--- ROLLING CORRELATION (24h window) ---")
    merged["rolling_corr_24h"] = merged["btc_return"].rolling(24).corr(merged["fart_return"])
    merged["rolling_corr_72h"] = merged["btc_return"].rolling(72).corr(merged["fart_return"])

    rc = merged["rolling_corr_24h"].dropna()
    print(f"  Mean 24h corr:   {rc.mean():.4f}")
    print(f"  Std 24h corr:    {rc.std():.4f}")
    print(f"  Min 24h corr:    {rc.min():.4f} (at {rc.idxmin()})")
    print(f"  Max 24h corr:    {rc.max():.4f} (at {rc.idxmax()})")

    # Periods where correlation breaks (potential manipulation)
    decorrelated = merged[merged["rolling_corr_24h"] < 0]
    if not decorrelated.empty:
        print(f"\n  Hours with NEGATIVE 24h correlation: {len(decorrelated)} ({len(decorrelated)/len(merged)*100:.1f}%)")
        print("  >> When Fart moves OPPOSITE to BTC = independent force (MM manipulation)")

        # What happens during decorrelation?
        decor_fart_ret = decorrelated["fart_return"].mean() * 10000
        decor_fart_vol = abs(decorrelated["fart_return"]).mean() * 10000
        normal_fart_vol = abs(merged[merged["rolling_corr_24h"] >= 0]["fart_return"]).mean() * 10000
        print(f"  Avg Fart return during decorrelation: {decor_fart_ret:.1f} bps")
        print(f"  Avg |Fart move| during decorrelation: {decor_fart_vol:.1f} bps")
        print(f"  Avg |Fart move| during normal corr:   {normal_fart_vol:.1f} bps")
        if decor_fart_vol > normal_fart_vol:
            print("  >> BIGGER moves during decorrelation — supports manipulation hypothesis!")

    # --- 4. Asymmetric Response (Does Fart react differently to BTC up vs down?) ---
    print("\n--- ASYMMETRIC RESPONSE ---")
    btc_up = merged[merged["btc_return"] > 0.001]
    btc_down = merged[merged["btc_return"] < -0.001]
    btc_flat = merged[abs(merged["btc_return"]) <= 0.001]

    if len(btc_up) > 20 and len(btc_down) > 20:
        up_beta = np.polyfit(btc_up["btc_return"], btc_up["fart_return"], 1)[0]
        down_beta = np.polyfit(btc_down["btc_return"], btc_down["fart_return"], 1)[0]
        flat_fart = btc_flat["fart_return"].mean() * 10000

        print(f"  When BTC UP:   Fart beta = {up_beta:.2f}x (n={len(btc_up)})")
        print(f"  When BTC DOWN: Fart beta = {down_beta:.2f}x (n={len(btc_down)})")
        print(f"  When BTC FLAT: Fart avg return = {flat_fart:.1f} bps (n={len(btc_flat)})")

        if down_beta > up_beta * 1.3:
            print("  >> FARTCOIN FALLS HARDER THAN IT RISES with BTC")
            print("  >> MMs likely amplify BTC downmoves to trigger liquidations")
        elif up_beta > down_beta * 1.3:
            print("  >> FARTCOIN PUMPS HARDER THAN IT DUMPS with BTC")
            print("  >> MMs may be using BTC rallies to pump and exit positions")

    # --- 5. Lead/Lag Analysis ---
    print("\n--- LEAD/LAG ANALYSIS ---")
    print("(Does BTC lead Fartcoin or vice versa?)")
    for lag in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        shifted_btc = merged["btc_return"].shift(lag)
        c = shifted_btc.corr(merged["fart_return"])
        direction = "BTC leads" if lag > 0 else "Fart leads" if lag < 0 else "Simultaneous"
        bar = "█" * int(abs(c) * 50)
        print(f"  Lag {lag:+d}h ({direction:>14}): corr = {c:+.4f}  {bar}")

    # Find peak lag
    lags = range(-8, 9)
    lag_corrs = [(lag, merged["btc_return"].shift(lag).corr(merged["fart_return"])) for lag in lags]
    peak_lag, peak_corr = max(lag_corrs, key=lambda x: abs(x[1]))
    print(f"\n  Peak correlation at lag {peak_lag:+d}h ({peak_corr:.4f})")
    if peak_lag > 0:
        print(f"  >> BTC leads Fartcoin by ~{peak_lag}h — trade Fartcoin after BTC moves")
    elif peak_lag < 0:
        print(f"  >> Fartcoin leads BTC by ~{abs(peak_lag)}h — unusual, possible MM front-running")
    else:
        print(f"  >> Simultaneous — no clear lead/lag")

    # --- 6. BTC Regime Analysis ---
    print("\n--- BTC REGIME ANALYSIS ---")
    print("(How does Fartcoin behave in different BTC environments?)")
    merged["btc_ret_24h"] = merged["btc_price"].pct_change(24)
    merged["fart_ret_24h"] = merged["fart_price"].pct_change(24)

    # Classify BTC regime
    def btc_regime(ret):
        if pd.isna(ret): return None
        if ret > 0.03: return "BTC Strong Rally (>3%)"
        if ret > 0.01: return "BTC Mild Rally (1-3%)"
        if ret > -0.01: return "BTC Flat (-1% to 1%)"
        if ret > -0.03: return "BTC Mild Dump (-3 to -1%)"
        return "BTC Strong Dump (<-3%)"

    merged["btc_regime"] = merged["btc_ret_24h"].apply(btc_regime)
    regime_order = ["BTC Strong Rally (>3%)", "BTC Mild Rally (1-3%)", "BTC Flat (-1% to 1%)",
                    "BTC Mild Dump (-3 to -1%)", "BTC Strong Dump (<-3%)"]

    regime_stats = merged.groupby("btc_regime").agg(
        fart_ret_mean=("fart_return", "mean"),
        fart_ret_std=("fart_return", "std"),
        fart_24h_ret_mean=("fart_ret_24h", "mean"),
        count=("fart_return", "count"),
    ).reindex(regime_order).dropna()

    regime_stats["fart_ret_bps"] = regime_stats["fart_ret_mean"] * 10000
    regime_stats["fart_24h_pct"] = regime_stats["fart_24h_ret_mean"] * 100
    print(regime_stats[["fart_ret_bps", "fart_24h_pct", "fart_ret_std", "count"]].round(4).to_string())

    # --- 7. Session-Specific BTC Correlation ---
    print("\n--- BTC CORRELATION BY TRADING SESSION ---")
    merged["hour"] = merged.index.hour
    merged["session"] = merged["hour"].apply(classify_session)

    session_order_list = ["Asia (00-08 UTC)", "London (08-13 UTC)",
                          "NYC (13-21 UTC)", "Late NYC / Pre-Asia (21-00 UTC)"]
    for s in session_order_list:
        sess = merged[merged["session"] == s]
        if len(sess) > 30:
            c = sess["btc_return"].corr(sess["fart_return"])
            b = np.polyfit(sess["btc_return"].dropna(), sess["fart_return"].dropna(), 1)[0] if len(sess.dropna()) > 10 else 0
            print(f"  {s}: corr={c:.4f}, beta={b:.2f}x (n={len(sess)})")

    return merged


# ---------------------------------------------------------------------------
# Analysis 4: Hourly Price Pattern Analysis
# ---------------------------------------------------------------------------

def analyze_hourly_patterns(cg_chart):
    """Analyze hourly price/volume for intraday manipulation patterns."""
    print("\n" + "=" * 70)
    print("HOURLY PRICE/VOLUME PATTERN ANALYSIS")
    print("=" * 70)

    df = cg_chart.copy()
    price_col = "price" if "price" in df.columns else "close"
    df["return"] = df[price_col].pct_change()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(24).mean()
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek  # 0=Mon, 6=Sun

    # --- Hour-of-day patterns ---
    print("\n--- HOUR-OF-DAY RETURN PATTERNS ---")
    hourly = df.groupby("hour")["return"].agg(["mean", "std", "count"])
    hourly["sharpe"] = hourly["mean"] / hourly["std"] * np.sqrt(365 * 24)
    hourly["mean_bps"] = hourly["mean"] * 10000
    print(hourly[["mean_bps", "std", "sharpe", "count"]].round(4).to_string())

    best_hour = hourly["mean_bps"].idxmax()
    worst_hour = hourly["mean_bps"].idxmin()
    print(f"\nBest hour (UTC): {best_hour}:00 ({hourly.loc[best_hour, 'mean_bps']:.1f} bps avg)")
    print(f"Worst hour (UTC): {worst_hour}:00 ({hourly.loc[worst_hour, 'mean_bps']:.1f} bps avg)")

    # --- Hour-of-day volume patterns ---
    print("\n--- HOUR-OF-DAY VOLUME PATTERNS ---")
    vol_hourly = df.groupby("hour")["vol_ratio"].mean()
    print("Avg volume ratio by hour (>1 = above 24h avg):")
    for h in range(24):
        bar = "█" * int(vol_hourly.get(h, 0) * 20)
        print(f"  {h:02d}:00  {vol_hourly.get(h, 0):.2f}x  {bar}")

    # --- Big move clustering ---
    print("\n--- BIG HOURLY MOVES (>5% in 1 hour) ---")
    big = df[abs(df["return"]) > 0.05].copy()
    if not big.empty:
        big["next_4h_ret"] = df[price_col].pct_change(4).shift(-4)
        for ts, row in big.iterrows():
            direction = "UP" if row["return"] > 0 else "DOWN"
            reversal = ""
            if not np.isnan(row.get("next_4h_ret", np.nan)):
                if np.sign(row["return"]) != np.sign(row["next_4h_ret"]):
                    reversal = " ← REVERSED"
                else:
                    reversal = " ← CONTINUED"
            print(f"  {ts}: {direction} {abs(row['return']):.1%}, vol={row.get('vol_ratio', 0):.1f}x{reversal}")
    else:
        print("  No moves >5% in a single hour.")

    # Check >3% moves
    med_moves = df[abs(df["return"]) > 0.03].copy()
    if not med_moves.empty:
        print(f"\n  Moves >3%: {len(med_moves)} occurrences")
        med_moves["next_4h_ret"] = df[price_col].pct_change(4).shift(-4)
        reversals = 0
        continuations = 0
        for _, row in med_moves.iterrows():
            if not np.isnan(row.get("next_4h_ret", np.nan)):
                if np.sign(row["return"]) != np.sign(row["next_4h_ret"]):
                    reversals += 1
                else:
                    continuations += 1
        total = reversals + continuations
        if total > 0:
            print(f"  Reversed within 4h: {reversals}/{total} ({reversals/total:.0%})")
            print(f"  Continued within 4h: {continuations}/{total} ({continuations/total:.0%})")

    # --- Volume-Price Correlation by Hour ---
    print("\n--- VOLUME PRECEDING PRICE MOVES ---")
    df["fwd_abs_ret_4h"] = abs(df["return"]).rolling(4).sum().shift(-4)
    corr = df["vol_ratio"].corr(df["fwd_abs_ret_4h"])
    print(f"  Volume ratio → next 4h |return| correlation: {corr:.4f}")
    if corr > 0.1:
        print("  >> Volume spikes predict subsequent volatility — positioning signal!")
    elif corr < -0.05:
        print("  >> Low volume precedes big moves — quiet accumulation signal!")

    return df


# ---------------------------------------------------------------------------
# Analysis 3: Signal Backtest with proper forward returns
# ---------------------------------------------------------------------------

def backtest_signals_hourly(data):
    """Backtest signals using hourly price data."""
    print("\n" + "=" * 70)
    print("SIGNAL BACKTEST (Hourly Resolution)")
    print("=" * 70)

    signals = compute_all_signals(data)
    if signals.empty:
        print("No signals computed.")
        return

    # Get hourly price for forward returns
    ohlcv = data["ohlcv"]
    price = ohlcv["price"] if "price" in ohlcv.columns else ohlcv["close"]

    # Compute forward returns at various horizons
    for h in [1, 2, 4, 8, 12, 24]:
        fwd = price.pct_change(h).shift(-h)
        fwd.name = f"fwd_ret_{h}h"
        signals = signals.join(fwd, how="left")

    # Drop rows where we don't have both signal and forward returns
    horizons = [c for c in signals.columns if c.startswith("fwd_ret_")]
    signal_cols = [c for c in signals.columns if c.startswith("sig_")]

    print(f"\nSignal columns: {signal_cols}")
    print(f"Forward return horizons: {horizons}")

    # --- IC Analysis ---
    print("\n--- INFORMATION COEFFICIENT (IC) BY SIGNAL ---")
    print(f"{'Signal':<25} {'1h IC':>8} {'2h IC':>8} {'4h IC':>8} {'8h IC':>8} {'12h IC':>8} {'24h IC':>8}")
    print("-" * 85)
    for sig in signal_cols + ["composite"]:
        ics = []
        for hz in horizons:
            clean = signals[[sig, hz]].dropna()
            if len(clean) > 50:
                ic = clean[sig].corr(clean[hz])
                ics.append(f"{ic:8.4f}")
            else:
                ics.append(f"{'N/A':>8}")
        print(f"{sig:<25} {'  '.join(ics)}")

    # --- Threshold Backtest ---
    print("\n--- THRESHOLD BACKTEST ---")
    for thresh in [0.2, 0.3, 0.4]:
        for hz in ["fwd_ret_4h", "fwd_ret_8h", "fwd_ret_24h"]:
            if hz not in signals.columns:
                continue
            clean = signals[["composite", hz]].dropna()

            longs = clean[clean["composite"] > thresh][hz]
            shorts = clean[clean["composite"] < -thresh][hz]

            if len(longs) > 5:
                hit = (longs > 0).mean()
                avg = longs.mean()
                print(f"  LONG  (>{thresh:+.1f}) @ {hz}: n={len(longs)}, hit={hit:.0%}, avg_ret={avg:+.3%}")

            if len(shorts) > 5:
                short_pnl = -shorts
                hit = (short_pnl > 0).mean()
                avg = short_pnl.mean()
                print(f"  SHORT (<{-thresh:+.1f}) @ {hz}: n={len(shorts)}, hit={hit:.0%}, avg_ret={avg:+.3%}")

    # --- Quintile Analysis ---
    print("\n--- COMPOSITE SIGNAL QUINTILE ANALYSIS (4h forward) ---")
    clean = signals[["composite", "fwd_ret_4h"]].dropna()
    if len(clean) > 100:
        clean["quintile"] = pd.qcut(clean["composite"], 5, labels=["Q1(bear)", "Q2", "Q3(neutral)", "Q4", "Q5(bull)"],
                                     duplicates="drop")
        q_stats = clean.groupby("quintile")["fwd_ret_4h"].agg(["mean", "std", "count"])
        q_stats["mean_bps"] = q_stats["mean"] * 10000
        q_stats["sharpe"] = q_stats["mean"] / q_stats["std"] * np.sqrt(365 * 6)
        print(q_stats.round(4).to_string())
        spread = q_stats["mean"].iloc[-1] - q_stats["mean"].iloc[0]
        print(f"\nQ5-Q1 spread: {spread*10000:.1f} bps ({spread*100:.3f}%)")
        if abs(spread) > 0.001:
            print("  >> MEANINGFUL spread — signal has predictive value!")
        else:
            print("  >> Weak spread — signal needs refinement.")

    return signals


# ---------------------------------------------------------------------------
# Analysis 4: Manipulation Cycle Detection
# ---------------------------------------------------------------------------

def detect_manipulation_cycles(data):
    """
    Look for the full manipulation cycle:
    1. Quiet accumulation (low volume)
    2. Price push (volume spike)
    3. Liquidation cascade (extreme volatility)
    4. Mean reversion (profit taking)
    """
    print("\n" + "=" * 70)
    print("MANIPULATION CYCLE DETECTION")
    print("=" * 70)

    daily = data.get("ohlcv_daily")
    if daily is None or daily.empty:
        print("Need daily OHLCV data.")
        return

    df = daily.copy()
    df["return"] = df["close"].pct_change()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(10).mean()
    df["range_pct"] = (df["high"] - df["low"]) / df["open"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = abs(df["close"] - df["open"]) / df["open"]

    # Detect phases
    df["quiet"] = df["vol_ratio"] < 0.7  # below-average volume
    df["spike"] = df["vol_ratio"] > 1.5  # above-average volume
    df["big_move"] = abs(df["return"]) > 0.10
    df["stop_hunt"] = (df["upper_wick"] > 0.4) | (df["lower_wick"] > 0.4)

    # Look for sequences: quiet → spike → big_move
    cycles = []
    for i in range(2, len(df)):
        # Pattern: 1-3 quiet days → volume spike + big move
        quiet_window = df["quiet"].iloc[max(0, i-3):i]
        if quiet_window.sum() >= 2 and df["big_move"].iloc[i]:
            entry_date = df.index[i]
            ret = df["return"].iloc[i]
            vol = df["vol_ratio"].iloc[i]

            # What happened after?
            fwd_1d = df["return"].iloc[i+1] if i+1 < len(df) else np.nan
            fwd_3d = (df["close"].iloc[min(i+3, len(df)-1)] / df["close"].iloc[i] - 1) if i+3 < len(df) else np.nan
            reversed_1d = not np.isnan(fwd_1d) and np.sign(ret) != np.sign(fwd_1d)

            cycles.append({
                "date": entry_date,
                "move": ret,
                "volume_ratio": vol,
                "quiet_days_before": int(quiet_window.sum()),
                "stop_hunt": bool(df["stop_hunt"].iloc[i]),
                "fwd_1d_return": fwd_1d,
                "fwd_3d_return": fwd_3d,
                "reversed_next_day": reversed_1d,
            })

    if cycles:
        cycles_df = pd.DataFrame(cycles)
        print("\n--- DETECTED MANIPULATION CYCLES ---")
        print("(Quiet accumulation → big move)")
        print(cycles_df.to_string(index=False))

        # Stats
        n = len(cycles_df)
        reversals = cycles_df["reversed_next_day"].sum()
        stop_hunts = cycles_df["stop_hunt"].sum()
        print(f"\nTotal cycles detected: {n}")
        print(f"Reversed next day: {reversals}/{n} ({reversals/n:.0%})")
        print(f"Involved stop hunt: {stop_hunts}/{n} ({stop_hunts/n:.0%})")
        print(f"Avg move size: {cycles_df['move'].abs().mean():.1%}")
        print(f"Avg volume ratio on move day: {cycles_df['volume_ratio'].mean():.2f}x")

        # The tradeable signal
        pumps = cycles_df[cycles_df["move"] > 0]
        dumps = cycles_df[cycles_df["move"] < 0]
        if len(pumps) > 0:
            print(f"\nPump cycles ({len(pumps)}):")
            print(f"  Avg 1d follow-through: {pumps['fwd_1d_return'].mean():+.1%}")
            print(f"  Avg 3d follow-through: {pumps['fwd_3d_return'].mean():+.1%}")
        if len(dumps) > 0:
            print(f"\nDump cycles ({len(dumps)}):")
            print(f"  Avg 1d follow-through: {dumps['fwd_1d_return'].mean():+.1%}")
            print(f"  Avg 3d follow-through: {dumps['fwd_3d_return'].mean():+.1%}")
    else:
        print("No clear manipulation cycles detected.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comprehensive(data, signals_df=None):
    """Generate comprehensive analysis charts."""
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("FARTCOIN MANIPULATION ANALYSIS", fontsize=18, fontweight="bold", y=0.98)

    # --- 1. Price + Volume (daily) ---
    ax1 = fig.add_subplot(4, 2, 1)
    daily = data.get("ohlcv_daily")
    if daily is not None:
        ax1.plot(daily.index, daily["close"], "b-", linewidth=1)
        ax1.set_title("Daily Price")
        ax1.set_ylabel("Price ($)")
        ax1b = ax1.twinx()
        ax1b.bar(daily.index, daily["volume"], alpha=0.3, color="gray", width=0.8)
        ax1b.set_ylabel("Volume ($)")

    # --- 2. Cross-Exchange OI Distribution ---
    ax2 = fig.add_subplot(4, 2, 2)
    deriv = data.get("derivatives")
    if deriv is not None:
        active = deriv[deriv["open_interest_usd"] > 100000].sort_values("open_interest_usd", ascending=True)
        if not active.empty:
            ax2.barh(active["exchange"].str[:20], active["open_interest_usd"] / 1e6, color="steelblue")
            ax2.set_xlabel("Open Interest ($M)")
            ax2.set_title("OI by Exchange (>$100k)")

    # --- 3. Cross-Exchange Funding Rates ---
    ax3 = fig.add_subplot(4, 2, 3)
    if deriv is not None:
        active = deriv[deriv["open_interest_usd"] > 100000].sort_values("funding_rate")
        if not active.empty:
            colors = ["red" if x < 0 else "green" for x in active["funding_rate"]]
            ax3.barh(active["exchange"].str[:20], active["funding_rate"], color=colors)
            ax3.axvline(0, color="black", linewidth=0.5)
            ax3.set_xlabel("Funding Rate")
            ax3.set_title("Funding Rate by Exchange")

    # --- 4. Hourly Volume Pattern ---
    ax4 = fig.add_subplot(4, 2, 4)
    cg = data.get("ohlcv")
    if cg is not None and "volume" in cg.columns:
        hourly_vol = cg.copy()
        hourly_vol["hour"] = hourly_vol.index.hour
        vol_by_hour = hourly_vol.groupby("hour")["volume"].mean()
        ax4.bar(vol_by_hour.index, vol_by_hour.values / 1e6, color="steelblue")
        ax4.set_xlabel("Hour (UTC)")
        ax4.set_ylabel("Avg Volume ($M)")
        ax4.set_title("Average Volume by Hour")

    # --- 5. Daily Return Distribution ---
    ax5 = fig.add_subplot(4, 2, 5)
    if daily is not None:
        returns = daily["close"].pct_change().dropna()
        ax5.hist(returns * 100, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax5.axvline(0, color="red", linewidth=1)
        ax5.set_xlabel("Daily Return (%)")
        ax5.set_title(f"Return Distribution (mean={returns.mean()*100:.2f}%, std={returns.std()*100:.1f}%)")

    # --- 6. Weekday Returns ---
    ax6 = fig.add_subplot(4, 2, 6)
    if daily is not None:
        daily_ret = daily.copy()
        daily_ret["return"] = daily_ret["close"].pct_change()
        daily_ret["weekday"] = daily_ret.index.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd_mean = daily_ret.groupby("weekday")["return"].mean().reindex(day_order)
        colors = ["green" if x > 0 else "red" for x in wd_mean]
        ax6.bar(range(7), wd_mean * 100, color=colors)
        ax6.set_xticks(range(7))
        ax6.set_xticklabels([d[:3] for d in day_order])
        ax6.set_ylabel("Avg Return (%)")
        ax6.set_title("Returns by Day of Week")
        ax6.axhline(0, color="black", linewidth=0.5)

    # --- 7. Composite Signal Over Time ---
    ax7 = fig.add_subplot(4, 2, 7)
    if signals_df is not None and "composite" in signals_df.columns:
        comp = signals_df["composite"].dropna()
        ax7.plot(comp.index, comp.values, linewidth=0.5, alpha=0.8, color="steelblue")
        ax7.axhline(0.4, color="green", linestyle="--", alpha=0.5, label="Long entry")
        ax7.axhline(-0.4, color="red", linestyle="--", alpha=0.5, label="Short entry")
        ax7.axhline(0, color="gray", alpha=0.3)
        ax7.set_title("Composite Signal Over Time")
        ax7.set_ylabel("Score")
        ax7.legend(fontsize=8)

    # --- 8. Stop Hunt Analysis ---
    ax8 = fig.add_subplot(4, 2, 8)
    if daily is not None:
        d = daily.copy()
        rng = d["high"] - d["low"]
        d["upper_wick"] = (d["high"] - d[["open", "close"]].max(axis=1)) / rng.replace(0, np.nan)
        d["lower_wick"] = (d[["open", "close"]].min(axis=1) - d["low"]) / rng.replace(0, np.nan)
        ax8.scatter(d["upper_wick"] * 100, d["lower_wick"] * 100,
                    c=d["close"].pct_change() * 100, cmap="RdYlGn", alpha=0.6, s=40)
        ax8.set_xlabel("Upper Wick (% of range)")
        ax8.set_ylabel("Lower Wick (% of range)")
        ax8.set_title("Stop Hunt Map (color = daily return)")
        ax8.axhline(40, color="red", linestyle="--", alpha=0.3)
        ax8.axvline(40, color="red", linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "deep_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nCharts saved to {OUTPUT_DIR / 'deep_analysis.png'}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_btc_analysis(merged, fart_data):
    """Generate BTC correlation charts."""
    if merged is None or merged.empty:
        return

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("BTC vs FARTCOIN — MANIPULATION CORRELATION", fontsize=18, fontweight="bold", y=0.98)

    # 1. Price overlay (normalized)
    ax1 = fig.add_subplot(3, 2, 1)
    btc_norm = merged["btc_price"] / merged["btc_price"].iloc[0] * 100
    fart_norm = merged["fart_price"] / merged["fart_price"].iloc[0] * 100
    ax1.plot(merged.index, btc_norm, "orange", label="BTC", linewidth=1)
    ax1.plot(merged.index, fart_norm, "steelblue", label="FART", linewidth=1)
    ax1.set_title("Normalized Price (base=100)")
    ax1.legend()
    ax1.set_ylabel("Indexed Price")

    # 2. Rolling correlation
    ax2 = fig.add_subplot(3, 2, 2)
    if "rolling_corr_24h" in merged.columns:
        ax2.plot(merged.index, merged["rolling_corr_24h"], "steelblue", linewidth=0.8, label="24h")
    if "rolling_corr_72h" in merged.columns:
        ax2.plot(merged.index, merged["rolling_corr_72h"], "orange", linewidth=1, label="72h")
    ax2.axhline(0, color="red", linewidth=1, linestyle="--")
    ax2.set_title("Rolling Correlation (BTC vs FART)")
    ax2.set_ylabel("Correlation")
    ax2.set_ylim(-1, 1)
    ax2.legend()

    # 3. Scatter: BTC return vs FART return
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.scatter(merged["btc_return"] * 100, merged["fart_return"] * 100,
                alpha=0.2, s=10, color="steelblue")
    # Regression line
    valid = merged[["btc_return", "fart_return"]].dropna()
    if len(valid) > 50:
        z = np.polyfit(valid["btc_return"], valid["fart_return"], 1)
        x_line = np.linspace(valid["btc_return"].min(), valid["btc_return"].max(), 100)
        ax3.plot(x_line * 100, (z[0] * x_line + z[1]) * 100, "red", linewidth=2,
                 label=f"beta={z[0]:.2f}")
    ax3.set_xlabel("BTC Return (%)")
    ax3.set_ylabel("FART Return (%)")
    ax3.set_title("Return Scatter (hourly)")
    ax3.legend()
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.axvline(0, color="gray", linewidth=0.5)

    # 4. Lead/Lag correlation
    ax4 = fig.add_subplot(3, 2, 4)
    lags = range(-8, 9)
    lag_corrs = [merged["btc_return"].shift(lag).corr(merged["fart_return"]) for lag in lags]
    colors = ["green" if c > 0 else "red" for c in lag_corrs]
    ax4.bar(list(lags), lag_corrs, color=colors)
    ax4.set_xlabel("Lag (hours, positive = BTC leads)")
    ax4.set_ylabel("Correlation")
    ax4.set_title("Lead/Lag Correlation")
    ax4.axhline(0, color="black", linewidth=0.5)

    # 5. BTC regime → Fartcoin 24h return
    ax5 = fig.add_subplot(3, 2, 5)
    if "btc_regime" in merged.columns:
        regime_order = ["BTC Strong Rally (>3%)", "BTC Mild Rally (1-3%)", "BTC Flat (-1% to 1%)",
                        "BTC Mild Dump (-3 to -1%)", "BTC Strong Dump (<-3%)"]
        regime_means = merged.groupby("btc_regime")["fart_return"].mean().reindex(regime_order).dropna()
        colors = ["darkgreen", "lightgreen", "gray", "salmon", "darkred"][:len(regime_means)]
        ax5.barh(range(len(regime_means)), regime_means * 10000, color=colors)
        ax5.set_yticks(range(len(regime_means)))
        ax5.set_yticklabels([r.replace("BTC ", "") for r in regime_means.index], fontsize=9)
        ax5.set_xlabel("Avg FART Return (bps/hr)")
        ax5.set_title("Fartcoin Behavior by BTC Regime")
        ax5.axvline(0, color="black", linewidth=0.5)

    # 6. Session correlation
    ax6 = fig.add_subplot(3, 2, 6)
    if "session" in merged.columns:
        session_order_list = ["Asia (00-08 UTC)", "London (08-13 UTC)",
                              "NYC (13-21 UTC)", "Late NYC / Pre-Asia (21-00 UTC)"]
        sess_corrs = []
        sess_labels = []
        for s in session_order_list:
            sess = merged[merged["session"] == s]
            if len(sess) > 30:
                c = sess["btc_return"].corr(sess["fart_return"])
                sess_corrs.append(c)
                sess_labels.append(s.split("(")[0].strip())
        if sess_corrs:
            colors = ["green" if c > 0.3 else "orange" if c > 0 else "red" for c in sess_corrs]
            ax6.barh(range(len(sess_corrs)), sess_corrs, color=colors)
            ax6.set_yticks(range(len(sess_corrs)))
            ax6.set_yticklabels(sess_labels)
            ax6.set_xlabel("Correlation")
            ax6.set_title("BTC-FART Correlation by Session")
            ax6.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "btc_correlation.png", dpi=150, bbox_inches="tight")
    print(f"\nBTC charts saved to {OUTPUT_DIR / 'btc_correlation.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("FARTCOIN DEEP ANALYSIS — MANIPULATION DETECTION REPORT")
    print("=" * 70)

    data = load_all()

    # 1. Cross-exchange derivatives analysis
    if "derivatives" in data:
        analyze_derivatives_snapshot(data["derivatives"])

    # 2. Trading session analysis (NYC / London / Asia)
    if "ohlcv" in data:
        analyze_by_session(data["ohlcv"])

    # 3. Hourly pattern analysis
    if "ohlcv" in data:
        analyze_hourly_patterns(data["ohlcv"])

    # 4. BTC Correlation & MM Behavior
    btc_merged = None
    if "ohlcv" in data:
        btc_merged = analyze_btc_correlation(data["ohlcv"])

    # 5. Signal backtest
    signals = backtest_signals_hourly(data)

    # 6. Manipulation cycle detection
    detect_manipulation_cycles(data)

    # 7. Charts
    print("\n[Generating charts...]")
    plot_comprehensive(data, signals)
    if btc_merged is not None:
        plot_btc_analysis(btc_merged, data["ohlcv"])

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY — ACTIONABLE SIGNALS")
    print("=" * 70)

    if "derivatives" in data:
        deriv = data["derivatives"]
        active = deriv[deriv["open_interest_usd"] > 10000]
        avg_fr = active["funding_rate"].mean()
        total_oi = active["open_interest_usd"].sum()
        total_vol = active["volume_24h_usd"].sum()
        oi_vol_ratio = total_oi / total_vol if total_vol > 0 else 0

        print(f"\n  Current State:")
        print(f"    Avg Funding Rate:  {avg_fr:.6f}")
        print(f"    Total OI:          ${total_oi:,.0f}")
        print(f"    24h Volume:        ${total_vol:,.0f}")
        print(f"    OI/Volume Ratio:   {oi_vol_ratio:.2f}x")

        print(f"\n  Manipulation Risk Assessment:")
        risk_score = 0
        if abs(avg_fr) > 0.01:
            risk_score += 2
            print(f"    [HIGH] Extreme funding rate ({avg_fr:.4f})")
        if oi_vol_ratio < 0.5:
            risk_score += 1
            print(f"    [MED]  High churning (OI/Vol = {oi_vol_ratio:.2f})")
        hhi = ((active["open_interest_usd"] / total_oi) ** 2).sum()
        if hhi > 0.15:
            risk_score += 2
            print(f"    [HIGH] OI concentration (HHI = {hhi:.4f})")

        top_ex = active.loc[active["open_interest_usd"].idxmax(), "exchange"]
        top_share = active["open_interest_usd"].max() / total_oi
        if top_share > 0.3:
            risk_score += 1
            print(f"    [MED]  {top_ex} holds {top_share:.0%} of OI")

        fr_range = active["funding_rate"].max() - active["funding_rate"].min()
        if fr_range > 0.05:
            risk_score += 1
            print(f"    [MED]  Funding rate divergence across exchanges ({fr_range:.4f})")

        print(f"\n  Overall Manipulation Risk: {'HIGH' if risk_score >= 4 else 'MODERATE' if risk_score >= 2 else 'LOW'} ({risk_score}/7)")


if __name__ == "__main__":
    main()
