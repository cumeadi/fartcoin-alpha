"""
Backtest Engine — Fartcoin Alpha Framework

Tests the signal framework against historical data to validate
whether the manipulation signals actually predict price movements.

Key metrics:
- Hit rate: % of trades that were profitable
- Avg return per trade
- Sharpe ratio of the signal-based strategy
- Signal-to-noise: do high-conviction signals outperform low-conviction?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from signal_engine import load_data, compute_all_signals, DEFAULT_WEIGHTS

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def merge_signals_with_price(signals_df, ohlcv_df):
    """Align signals with forward returns for backtesting."""
    price = ohlcv_df["close"].resample("1h").last().ffill()

    # Forward returns at various horizons
    for h in [1, 4, 8, 12, 24]:
        price_fwd = price.shift(-h)
        ret = (price_fwd / price - 1)
        ret.name = f"fwd_ret_{h}h"
        signals_df = signals_df.join(ret, how="left")

    return signals_df


def analyze_signal_predictiveness(df, signal_col, ret_col="fwd_ret_4h", n_bins=5):
    """
    Core test: bin the signal into quintiles and measure average forward
    returns in each bin. If the signal works, top quintile should have
    significantly different returns than bottom quintile.
    """
    clean = df[[signal_col, ret_col]].dropna()
    if len(clean) < 50:
        return None

    clean["bin"] = pd.qcut(clean[signal_col], n_bins, labels=False, duplicates="drop")
    result = clean.groupby("bin")[ret_col].agg(["mean", "std", "count"])
    result["sharpe"] = result["mean"] / result["std"].replace(0, np.nan) * np.sqrt(24)  # annualize hourly

    # Spread: top bin return minus bottom bin return
    spread = result["mean"].iloc[-1] - result["mean"].iloc[0]

    return {
        "signal": signal_col,
        "return_col": ret_col,
        "bin_stats": result,
        "spread": spread,
        "ic": clean[signal_col].corr(clean[ret_col]),  # information coefficient
        "ic_rank": clean[signal_col].rank().corr(clean[ret_col].rank()),  # rank IC
        "n_obs": len(clean),
    }


def threshold_backtest(df, entry_threshold=0.4, exit_threshold=0.1,
                       fwd_horizon="fwd_ret_4h"):
    """
    Simple threshold-based backtest:
    - Go LONG when composite > +threshold
    - Go SHORT when composite < -threshold
    - Measure realized forward returns
    """
    longs = df[df["composite"] > entry_threshold][fwd_horizon].dropna()
    shorts = df[df["composite"] < -entry_threshold][fwd_horizon].dropna()

    results = {}
    if len(longs) > 0:
        results["long"] = {
            "count": len(longs),
            "mean_return": longs.mean(),
            "median_return": longs.median(),
            "hit_rate": (longs > 0).mean(),
            "sharpe": longs.mean() / longs.std() * np.sqrt(252) if longs.std() > 0 else 0,
            "max_return": longs.max(),
            "max_drawdown": longs.min(),
        }

    if len(shorts) > 0:
        # For shorts, we profit when price goes DOWN, so negate returns
        short_pnl = -shorts
        results["short"] = {
            "count": len(shorts),
            "mean_return": short_pnl.mean(),
            "median_return": short_pnl.median(),
            "hit_rate": (short_pnl > 0).mean(),
            "sharpe": short_pnl.mean() / short_pnl.std() * np.sqrt(252) if short_pnl.std() > 0 else 0,
            "max_return": short_pnl.max(),
            "max_drawdown": short_pnl.min(),
        }

    return results


def sensitivity_analysis(df, thresholds=None, horizons=None):
    """
    Sweep entry thresholds and forward horizons to find optimal params.
    This tells you: what conviction level + holding period maximizes alpha?
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    if horizons is None:
        horizons = ["fwd_ret_1h", "fwd_ret_4h", "fwd_ret_8h", "fwd_ret_12h", "fwd_ret_24h"]

    rows = []
    for thresh in thresholds:
        for hz in horizons:
            if hz not in df.columns:
                continue
            results = threshold_backtest(df, entry_threshold=thresh, fwd_horizon=hz)
            for direction, stats in results.items():
                rows.append({
                    "threshold": thresh,
                    "horizon": hz,
                    "direction": direction,
                    **stats,
                })

    return pd.DataFrame(rows)


def weight_optimization(df, n_iterations=1000, ret_col="fwd_ret_4h"):
    """
    Monte Carlo weight optimization.
    Randomly sample weight vectors and evaluate which combination
    of signals produces the best composite score.
    """
    signal_cols = [c for c in df.columns if c.startswith("sig_")]
    if not signal_cols or ret_col not in df.columns:
        return None

    clean = df[signal_cols + [ret_col]].dropna()
    if len(clean) < 50:
        return None

    best_ic = -1
    best_weights = None
    results = []

    rng = np.random.default_rng(42)
    for _ in range(n_iterations):
        # Random weights that sum to 1
        raw = rng.dirichlet(np.ones(len(signal_cols)))
        weights = dict(zip(signal_cols, raw))

        # Compute composite with these weights
        composite = sum(clean[col] * w for col, w in weights.items())
        ic = composite.corr(clean[ret_col])

        results.append({"ic": ic, **weights})

        if abs(ic) > abs(best_ic):
            best_ic = ic
            best_weights = weights

    return {
        "best_ic": best_ic,
        "best_weights": best_weights,
        "all_results": pd.DataFrame(results),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_signal_analysis(df, output_dir=OUTPUT_DIR):
    """Generate analysis charts."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Fartcoin Alpha Signal Analysis", fontsize=16, fontweight="bold")

    signal_cols = [c for c in df.columns if c.startswith("sig_")]

    # 1. Composite score over time
    ax = axes[0, 0]
    if "composite" in df.columns:
        comp = df["composite"].dropna()
        ax.plot(comp.index, comp.values, linewidth=0.5, alpha=0.8)
        ax.axhline(0.4, color="green", linestyle="--", alpha=0.5, label="Long threshold")
        ax.axhline(-0.4, color="red", linestyle="--", alpha=0.5, label="Short threshold")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.set_title("Composite Signal Over Time")
        ax.legend(fontsize=8)
        ax.set_ylabel("Score")

    # 2. Signal correlation heatmap
    ax = axes[0, 1]
    if signal_cols:
        corr = df[signal_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title("Signal Correlations")

    # 3. Signal distribution
    ax = axes[1, 0]
    for col in signal_cols:
        if col in df.columns:
            df[col].dropna().hist(ax=ax, bins=50, alpha=0.4, label=col)
    ax.set_title("Signal Distributions")
    ax.legend(fontsize=7)

    # 4. Composite vs forward returns scatter
    ax = axes[1, 1]
    if "composite" in df.columns and "fwd_ret_4h" in df.columns:
        clean = df[["composite", "fwd_ret_4h"]].dropna()
        ax.scatter(clean["composite"], clean["fwd_ret_4h"],
                   alpha=0.1, s=2, color="steelblue")
        # Add trend line
        z = np.polyfit(clean["composite"], clean["fwd_ret_4h"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(clean["composite"].min(), clean["composite"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"slope={z[0]:.4f}")
        ax.set_xlabel("Composite Signal")
        ax.set_ylabel("4h Forward Return")
        ax.set_title("Signal vs Forward Returns")
        ax.legend()

    # 5. Cumulative return of strategy
    ax = axes[2, 0]
    if "composite" in df.columns and "fwd_ret_4h" in df.columns:
        clean = df[["composite", "fwd_ret_4h"]].dropna()
        # Long when composite > 0.3, short when < -0.3
        position = np.where(clean["composite"] > 0.3, 1,
                           np.where(clean["composite"] < -0.3, -1, 0))
        strat_ret = position * clean["fwd_ret_4h"].values
        cum_ret = (1 + pd.Series(strat_ret, index=clean.index)).cumprod()
        ax.plot(cum_ret.index, cum_ret.values, linewidth=1)
        ax.set_title("Cumulative Strategy Return")
        ax.set_ylabel("Growth of $1")

    # 6. Hit rate by signal strength
    ax = axes[2, 1]
    if "composite" in df.columns and "fwd_ret_4h" in df.columns:
        clean = df[["composite", "fwd_ret_4h"]].dropna()
        bins = pd.qcut(clean["composite"], 10, labels=False, duplicates="drop")
        hit_rates = clean.groupby(bins).apply(
            lambda g: (g["fwd_ret_4h"] * np.sign(g["composite"].mean()) > 0).mean()
        )
        ax.bar(hit_rates.index, hit_rates.values, color="steelblue")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Signal Strength Decile")
        ax.set_ylabel("Hit Rate")
        ax.set_title("Hit Rate by Signal Strength")

    plt.tight_layout()
    fig.savefig(output_dir / "signal_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Chart saved to {output_dir / 'signal_analysis.png'}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 60)
    print("FARTCOIN ALPHA FRAMEWORK — BACKTEST")
    print("=" * 60)

    print("\n[1] Loading data...")
    data = load_data()

    print("\n[2] Computing signals...")
    signals = compute_all_signals(data)
    if signals.empty:
        print("No signals. Run data_collector.py first.")
        return

    print("\n[3] Merging with forward returns...")
    if "ohlcv" not in data:
        print("Need OHLCV data for backtesting.")
        return
    bt = merge_signals_with_price(signals, data["ohlcv"])

    print("\n[4] Individual Signal Predictiveness:")
    print("-" * 50)
    signal_cols = [c for c in bt.columns if c.startswith("sig_")]
    for col in signal_cols:
        result = analyze_signal_predictiveness(bt, col)
        if result:
            print(f"\n  {col}:")
            print(f"    IC (Pearson):  {result['ic']:.4f}")
            print(f"    IC (Rank):     {result['ic_rank']:.4f}")
            print(f"    Spread (Q5-Q1): {result['spread']*100:.3f}%")
            print(f"    Observations:  {result['n_obs']}")

    print("\n[5] Threshold Backtest (composite signal):")
    print("-" * 50)
    for hz in ["fwd_ret_1h", "fwd_ret_4h", "fwd_ret_8h", "fwd_ret_24h"]:
        if hz not in bt.columns:
            continue
        results = threshold_backtest(bt, fwd_horizon=hz)
        for direction, stats in results.items():
            print(f"\n  {direction.upper()} @ {hz}:")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

    print("\n[6] Sensitivity Analysis:")
    print("-" * 50)
    sens = sensitivity_analysis(bt)
    if not sens.empty:
        # Show top 10 by Sharpe
        top = sens.nlargest(10, "sharpe")
        print(top[["threshold", "horizon", "direction", "count",
                    "mean_return", "hit_rate", "sharpe"]].to_string(index=False))
        sens.to_csv(DATA_DIR / "sensitivity.csv", index=False)

    print("\n[7] Weight Optimization (Monte Carlo):")
    print("-" * 50)
    opt = weight_optimization(bt)
    if opt:
        print(f"  Best IC: {opt['best_ic']:.4f}")
        print(f"  Optimal weights:")
        for sig, w in sorted(opt["best_weights"].items(), key=lambda x: -x[1]):
            print(f"    {sig}: {w:.3f}")

    print("\n[8] Generating charts...")
    plot_signal_analysis(bt)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_backtest()
