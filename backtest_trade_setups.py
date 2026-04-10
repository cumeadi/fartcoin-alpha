"""
Backtest Trade Setup Logic — Fartcoin Alpha Framework

Validates the 6 trade setup types defined in _compute_trade_setups() (projections.py)
against 2,162 hourly rows of historical data (Jan 8 – Apr 8, 2026).

Each setup's trigger conditions are reconstructed from available historical CSVs.
Where live data (e.g. Coinglass OI series) doesn't exist historically, we use
documented proxies.

Run:
    python3 backtest_trade_setups.py

Outputs:
    output/backtest_setups_summary.csv   — one row per setup, all metrics
    output/backtest_setups_monthly.csv   — hit rate by month per setup
    output/backtest_setups_equity.png    — cumulative PnL curves
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — mirror projections.py calibration
# ---------------------------------------------------------------------------
CARRY_COST          = 0.0045        # 0.45% per 4h (Bybit funding floor)
LSR_EXHAUSTION_PCT  = 0.80          # rolling percentile trigger
LSR_ROLLING_WINDOW  = 90 * 24      # 90 days in hours for percentile calc
SETTLE_HOURS_UTC    = {0, 8, 16}    # Bybit funding settlement UTC hours
PRE_SETTLE_WINDOW   = 2             # hours before settle for micro-long
WICK_THRESHOLD      = 0.03          # 3%+ lower wick for cascade proxy
SHARPE_ANNUALISE    = np.sqrt(24 * 365)  # hourly → annual

SETUP_NAMES = [
    "POST_SETTLEMENT_FADE",
    "LSR_EXHAUSTION_SHORT",
    "BTC_DIVERGENCE_PROXY",
    "OI_BUILDING_PRICE_WEAK",
    "PRE_SETTLEMENT_MICRO_LONG",
    "POST_CASCADE_ENTRY",
]


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    """
    Merge OHLCV + signals + LSR + funding into a single hourly DataFrame.
    All sources are resampled / deduped to hourly frequency, then inner-joined
    on a clean UTC hour index.
    """
    # --- OHLCV ---
    ohlcv = pd.read_csv(DATA_DIR / "FARTCOIN_ohlcv_hourly.csv", parse_dates=["timestamp"])
    ohlcv = ohlcv.set_index("timestamp").sort_index()
    # Resample to clean hourly (take last value in each hour bucket)
    ohlcv = ohlcv[["open", "high", "low", "close", "volume"]].resample("1h").last().ffill()

    # --- Signals ---
    sig = pd.read_csv(DATA_DIR / "signals.csv", parse_dates=["timestamp"])
    sig = sig.set_index("timestamp").sort_index()
    # Keep columns we need; deduplicate by taking mean within each hour
    sig_cols = ["sig_funding", "sig_lsr", "sig_oi_accel", "sig_oi_divergence",
                "sig_volume_spike", "sig_pv_divergence", "composite"]
    sig = sig[[c for c in sig_cols if c in sig.columns]]
    sig = sig.resample("1h").mean()

    # --- LSR ---
    lsr = pd.read_csv(DATA_DIR / "FARTCOINUSDT_lsr.csv", parse_dates=["timestamp"])
    lsr = lsr.set_index("timestamp").sort_index()
    lsr = lsr[["longShortRatio"]].resample("1h").last().ffill()

    # --- Funding ---
    fund = pd.read_csv(DATA_DIR / "FARTCOINUSDT_funding.csv", parse_dates=["timestamp"])
    fund = fund.set_index("timestamp").sort_index()
    fund = fund[["fundingRate"]].resample("1h").last().ffill()

    # --- Merge ---
    df = ohlcv.join(sig, how="left").join(lsr, how="left").join(fund, how="left")
    df = df.dropna(subset=["close"])  # must have price
    df.index = df.index.tz_localize(None)  # ensure tz-naive
    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add settlement labels, LSR percentile, price features, wick ratio."""
    df = df.copy()

    # Settlement phase labels
    utc_hour = df.index.hour
    df["is_settle_hour"]   = utc_hour.isin(SETTLE_HOURS_UTC)
    df["hours_since_settle"] = utc_hour.map(lambda h: min((h - s) % 8 for s in SETTLE_HOURS_UTC))
    df["hours_until_settle"] = utc_hour.map(lambda h: min((s - h) % 8 for s in SETTLE_HOURS_UTC))

    # Session label
    df["session"] = utc_hour.map(_classify_session)

    # Month label
    df["month"] = df.index.to_period("M").astype(str)

    # Price features
    df["price_4h_chg"]  = df["close"].pct_change(4)
    df["price_1h_chg"]  = df["close"].pct_change(1)
    df["volume_ratio"]  = df["volume"] / df["volume"].rolling(24, min_periods=8).mean()

    # Lower wick ratio: (open - low) / open  — proxy for liquidation cascade
    df["wick_lower"]    = (df["open"] - df["low"]).clip(lower=0) / df["open"].replace(0, np.nan)

    # 4h pre-settlement return: return over 4h leading into a settlement candle
    df["pre_4h_ret"] = df["close"].pct_change(4)

    # LSR rolling percentile (90-day window)
    if "longShortRatio" in df.columns:
        df["lsr_pct"] = (
            df["longShortRatio"]
            .rolling(LSR_ROLLING_WINDOW, min_periods=168)  # min 1 week
            .rank(pct=True)
        )
    else:
        df["lsr_pct"] = np.nan

    return df


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward return columns at 1h, 2h, 4h, 8h horizons."""
    df = df.copy()
    price = df["close"]
    for h in [1, 2, 4, 8]:
        df[f"fwd_ret_{h}h"] = price.shift(-h) / price - 1
    return df


def _classify_session(hour: int) -> str:
    if 0 <= hour < 8:   return "Asia"
    if 8 <= hour < 13:  return "London"
    if 13 <= hour < 21: return "NYC"
    return "Late NYC"


# ---------------------------------------------------------------------------
# 3. Setup Triggers
# ---------------------------------------------------------------------------

def trigger_post_settlement_fade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trigger on settlement candles where price moved strongly into settlement
    (≥0.3%), then fade the move. Direction depends on sign of pre_4h_ret.
    Outcome measured as 4h forward return (sign-corrected for direction).
    """
    settle = df[df["is_settle_hour"]].copy()
    # Strong pre-move condition
    mask = settle["pre_4h_ret"].abs() >= 0.003
    triggered = settle[mask].copy()
    # Trade direction: fade the move
    triggered["direction"] = np.where(triggered["pre_4h_ret"] > 0, "short", "long")
    # PnL: short → negate fwd_ret; long → keep fwd_ret
    triggered["outcome"] = np.where(
        triggered["direction"] == "short",
        -triggered["fwd_ret_4h"],
        triggered["fwd_ret_4h"],
    )
    triggered["setup"] = "POST_SETTLEMENT_FADE"
    return triggered.dropna(subset=["outcome"])


def trigger_lsr_exhaustion_short(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trigger when LSR is in top 20th percentile (crowded longs) AND composite
    signal is negative. Short trade; profit when price falls.
    """
    mask = (df["lsr_pct"] > LSR_EXHAUSTION_PCT) & (df["composite"] < 0)
    triggered = df[mask].copy()
    triggered["direction"] = "short"
    triggered["outcome"]   = -triggered["fwd_ret_4h"]
    triggered["setup"]     = "LSR_EXHAUSTION_SHORT"
    return triggered.dropna(subset=["outcome"])


def trigger_btc_divergence_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    No BTC data historically → proxy with funding+LSR crowding.
    Trigger when composite is in the bottom decile AND both funding and LSR
    signals are negative (crowded, weakening market). Short trade.
    Composite is scaled to ~[-0.17, +0.23]; use p10 ≈ -0.086 as threshold.
    """
    sig_fund = df.get("sig_funding", pd.Series(np.nan, index=df.index))
    sig_lsr  = df.get("sig_lsr",     pd.Series(np.nan, index=df.index))
    # Use dynamic p10 threshold to stay regime-adaptive
    comp_p10 = df["composite"].quantile(0.10)
    mask = (
        (sig_fund < -0.15) &
        (sig_lsr  < -0.10) &
        (df["composite"] < comp_p10)
    )
    triggered = df[mask].copy()
    triggered["direction"] = "short"
    triggered["outcome"]   = -triggered["fwd_ret_4h"]
    triggered["setup"]     = "BTC_DIVERGENCE_PROXY"
    return triggered.dropna(subset=["outcome"])


def trigger_oi_building_price_weak(df: pd.DataFrame) -> pd.DataFrame:
    """
    OI proxy: price falling over 4h on above-average volume = longs being
    built / trapped. Short trade.
    """
    mask = (
        (df["price_4h_chg"] < -0.005) &
        (df["volume_ratio"] > 1.1)
    )
    triggered = df[mask].copy()
    triggered["direction"] = "short"
    triggered["outcome"]   = -triggered["fwd_ret_4h"]
    triggered["setup"]     = "OI_BUILDING_PRICE_WEAK"
    return triggered.dropna(subset=["outcome"])


def trigger_pre_settlement_micro_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enter long when ≤2h before settlement, composite in the upper half (>p50),
    funding signal positive. 2h hold into settlement.
    Composite p75 ≈ +0.043; use that as minimum conviction threshold.
    """
    sig_fund = df.get("sig_funding", pd.Series(np.nan, index=df.index))
    comp_p75 = df["composite"].quantile(0.75)
    mask = (
        (df["hours_until_settle"] <= PRE_SETTLE_WINDOW) &
        (df["hours_until_settle"] > 0) &  # exclude settle candle itself
        (sig_fund > 0.15) &
        (df["composite"] > comp_p75)
    )
    triggered = df[mask].copy()
    triggered["direction"] = "long"
    triggered["outcome"]   = triggered["fwd_ret_2h"]
    triggered["setup"]     = "PRE_SETTLEMENT_MICRO_LONG"
    return triggered.dropna(subset=["outcome"])


def trigger_post_cascade_entry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cascade proxy: the OHLCV CSV contains price snapshots only (open==high==low==close),
    so intra-bar wick detection is impossible. Instead, detect cascade via a sharp
    1h price drop followed by a reversal — i.e. previous-hour return was below -2%
    AND current-hour return is positive. Enter long on confirmed reversal.
    """
    prev_1h_ret = df["price_1h_chg"].shift(1)  # last hour's return
    curr_1h_ret = df["price_1h_chg"]            # this hour's return
    mask = (
        (prev_1h_ret < -0.02) &   # prior candle: ≥2% drop (cascade)
        (curr_1h_ret > 0.0)        # this candle: positive recovery
    )
    triggered = df[mask].copy()
    triggered["direction"] = "long"
    triggered["outcome"]   = triggered["fwd_ret_4h"]
    triggered["setup"]     = "POST_CASCADE_ENTRY"
    return triggered.dropna(subset=["outcome"])


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_setup(triggered: pd.DataFrame, name: str) -> dict:
    """Compute per-setup metrics from triggered rows with outcome column."""
    if len(triggered) == 0:
        return _empty_result(name)

    out = triggered["outcome"]
    direction = triggered["direction"].iloc[0]

    # Hit = outcome beats carry cost (longs) or zero (shorts)
    threshold = CARRY_COST if direction == "long" else 0.0
    hit = out > threshold

    n = len(out)
    hit_rate = hit.mean()
    avg_ret  = out.mean()
    med_ret  = out.median()
    std_ret  = out.std()
    sharpe   = (avg_ret / std_ret * SHARPE_ANNUALISE) if std_ret > 0 else 0.0
    carry_edge = avg_ret - threshold

    return {
        "setup":              name,
        "direction":          direction,
        "n_trades":           n,
        "hit_rate":           round(hit_rate, 4),
        "avg_return":         round(avg_ret, 6),
        "median_return":      round(med_ret, 6),
        "carry_adj_edge":     round(carry_edge, 6),
        "sharpe_annual":      round(sharpe, 3),
        "max_win":            round(out.max(), 6),
        "max_loss":           round(out.min(), 6),
        "std_return":         round(std_ret, 6),
    }


def _empty_result(name: str) -> dict:
    return {
        "setup": name, "direction": "—", "n_trades": 0,
        "hit_rate": None, "avg_return": None, "median_return": None,
        "carry_adj_edge": None, "sharpe_annual": None,
        "max_win": None, "max_loss": None, "std_return": None,
    }


def monthly_breakdown(triggered: pd.DataFrame, name: str) -> pd.DataFrame:
    """Hit rate by calendar month for a triggered setup DataFrame."""
    if len(triggered) == 0:
        return pd.DataFrame()

    direction = triggered["direction"].iloc[0]
    threshold = CARRY_COST if direction == "long" else 0.0

    triggered = triggered.copy()
    triggered["hit"]   = triggered["outcome"] > threshold
    triggered["month"] = triggered.index.to_period("M").astype(str)

    monthly = triggered.groupby("month")["hit"].agg(
        hit_rate="mean", n_trades="count"
    ).reset_index()
    monthly["setup"] = name
    return monthly


def session_breakdown(triggered: pd.DataFrame, name: str) -> pd.DataFrame:
    """Hit rate by trading session."""
    if len(triggered) == 0 or "session" not in triggered.columns:
        return pd.DataFrame()

    direction = triggered["direction"].iloc[0]
    threshold = CARRY_COST if direction == "long" else 0.0

    triggered = triggered.copy()
    triggered["hit"] = triggered["outcome"] > threshold

    session = triggered.groupby("session")["hit"].agg(
        hit_rate="mean", n_trades="count"
    ).reset_index()
    session["setup"] = name
    return session


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_results(all_triggered: list[pd.DataFrame], summary: pd.DataFrame):
    """
    Two-panel chart:
      Top: cumulative PnL curves per setup
      Bottom: hit rate bar chart per setup
    """
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.45)

    ax_eq  = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    colors = sns.color_palette("tab10", n_colors=len(SETUP_NAMES))

    # --- Equity curves ---
    for i, (triggered, name) in enumerate(zip(all_triggered, SETUP_NAMES)):
        if len(triggered) == 0:
            continue
        out = triggered["outcome"].sort_index()
        cum = (1 + out).cumprod()
        ax_eq.plot(cum.reset_index(drop=True), label=name, color=colors[i], linewidth=1.5)

    ax_eq.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_eq.set_title("Cumulative PnL by Setup (indexed at 1.0)", fontsize=13, fontweight="bold")
    ax_eq.set_xlabel("Trade #")
    ax_eq.set_ylabel("Equity (1 = flat)")
    ax_eq.legend(fontsize=8, loc="upper left")

    # --- Hit rate bars ---
    valid = summary[summary["n_trades"] > 0].copy()
    if len(valid) > 0:
        x   = range(len(valid))
        bars = ax_bar.bar(
            x,
            valid["hit_rate"].astype(float),
            color=[colors[SETUP_NAMES.index(n)] if n in SETUP_NAMES else "#888"
                   for n in valid["setup"]],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        ax_bar.axhline(0.5,  color="#ff6b6b", linewidth=1.2, linestyle="--", label="50% breakeven")
        ax_bar.axhline(0.60, color="#51cf66", linewidth=1.0, linestyle=":",  label="60% target")
        ax_bar.set_xticks(list(x))
        ax_bar.set_xticklabels(valid["setup"], rotation=20, ha="right", fontsize=8)
        ax_bar.set_ylabel("Hit Rate")
        ax_bar.set_title("Hit Rate per Setup (carry-adjusted)", fontsize=11, fontweight="bold")
        ax_bar.set_ylim(0, 1)
        ax_bar.legend(fontsize=8)

        # Annotate n_trades
        for rect, (_, row) in zip(bars, valid.iterrows()):
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                f"n={int(row['n_trades'])}",
                ha="center", va="bottom", fontsize=7, color="white"
            )

    fig.suptitle("Fartcoin Alpha — Trade Setup Backtest (Jan–Apr 2026)",
                 fontsize=14, fontweight="bold", y=0.98)

    out_path = OUTPUT_DIR / "backtest_setups_equity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"\n[Chart saved] {out_path}")


# ---------------------------------------------------------------------------
# 6. Main Orchestrator
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 60)
    print("  FARTCOIN ALPHA — TRADE SETUP BACKTEST")
    print("=" * 60)

    # Load & feature-engineer
    print("\n[1/4] Loading and merging data...")
    df = load_and_merge()
    df = add_labels(df)
    df = add_forward_returns(df)
    print(f"      Rows available: {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Trigger each setup
    print("\n[2/4] Triggering setups...")
    trigger_fns = [
        trigger_post_settlement_fade,
        trigger_lsr_exhaustion_short,
        trigger_btc_divergence_proxy,
        trigger_oi_building_price_weak,
        trigger_pre_settlement_micro_long,
        trigger_post_cascade_entry,
    ]

    all_triggered = []
    summary_rows  = []
    monthly_rows  = []
    session_rows  = []

    for fn, name in zip(trigger_fns, SETUP_NAMES):
        triggered = fn(df)
        all_triggered.append(triggered)
        result  = evaluate_setup(triggered, name)
        monthly = monthly_breakdown(triggered, name)
        session = session_breakdown(triggered, name)
        summary_rows.append(result)
        if len(monthly) > 0:
            monthly_rows.append(monthly)
        if len(session) > 0:
            session_rows.append(session)
        print(f"      {name}: {len(triggered)} triggers")

    summary = pd.DataFrame(summary_rows)

    # Save CSVs
    print("\n[3/4] Saving outputs...")
    summary_path = OUTPUT_DIR / "backtest_setups_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"      Summary → {summary_path}")

    if monthly_rows:
        monthly_df = pd.concat(monthly_rows, ignore_index=True)
        monthly_path = OUTPUT_DIR / "backtest_setups_monthly.csv"
        monthly_df.to_csv(monthly_path, index=False)
        print(f"      Monthly → {monthly_path}")

    if session_rows:
        session_df = pd.concat(session_rows, ignore_index=True)
        session_path = OUTPUT_DIR / "backtest_setups_session.csv"
        session_df.to_csv(session_path, index=False)
        print(f"      Session → {session_path}")

    # Plot
    print("\n[4/4] Plotting...")
    plot_results(all_triggered, summary)

    # Print summary table
    _print_summary(summary)

    # Print monthly detail
    if monthly_rows:
        _print_monthly(pd.concat(monthly_rows))

    return summary


def _print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 80)
    print(f"  {'SETUP':<28} {'DIR':<6} {'N':>5}  {'HIT%':>6}  {'AVG_RET':>8}  "
          f"{'EDGE':>8}  {'SHARPE':>7}  {'MAX_WIN':>8}  {'MAX_LOSS':>9}")
    print("-" * 80)
    for _, row in summary.iterrows():
        if row["n_trades"] == 0:
            print(f"  {row['setup']:<28} {'—':<6} {'0':>5}  {'—':>6}  {'—':>8}  "
                  f"{'—':>8}  {'—':>7}  {'—':>8}  {'—':>9}")
            continue
        hit_str  = f"{row['hit_rate']:.1%}" if row["hit_rate"] is not None else "—"
        avg_str  = f"{row['avg_return']*100:+.3f}%" if row["avg_return"] is not None else "—"
        edge_str = f"{row['carry_adj_edge']*100:+.3f}%" if row["carry_adj_edge"] is not None else "—"
        sh_str   = f"{row['sharpe_annual']:+.2f}" if row["sharpe_annual"] is not None else "—"
        mw_str   = f"{row['max_win']*100:.2f}%" if row["max_win"] is not None else "—"
        ml_str   = f"{row['max_loss']*100:.2f}%" if row["max_loss"] is not None else "—"

        # Flag validated setups
        validated = (
            row["n_trades"] >= 20 and
            row["hit_rate"] is not None and
            row["hit_rate"] >= 0.60
        )
        flag = " ✓" if validated else "  "
        print(f"{flag} {row['setup']:<28} {row['direction']:<6} {int(row['n_trades']):>5}  "
              f"{hit_str:>6}  {avg_str:>8}  {edge_str:>8}  {sh_str:>7}  {mw_str:>8}  {ml_str:>9}")

    print("=" * 80)
    print("  ✓ = validated (n ≥ 20, hit_rate ≥ 60% carry-adjusted)")
    print(f"  Carry cost applied: {CARRY_COST*100:.2f}%/4h (Bybit floor)\n")


def _print_monthly(monthly_df: pd.DataFrame):
    print("\n  MONTHLY HIT RATE BREAKDOWN")
    print("-" * 60)
    for setup in SETUP_NAMES:
        sub = monthly_df[monthly_df["setup"] == setup]
        if len(sub) == 0:
            continue
        row_str = "  ".join(
            f"{r['month']} {r['hit_rate']:.0%}(n={int(r['n_trades'])})"
            for _, r in sub.iterrows()
        )
        print(f"  {setup[:28]:<28}  {row_str}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_backtest()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
