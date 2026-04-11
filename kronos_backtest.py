"""
Kronos Zero-Shot Backtest — Fartcoin Alpha Framework

Step 1 of the Kronos integration plan:
  Run Kronos-mini zero-shot on historical OHLCV and measure:
  1. Kronos direction accuracy alone (4h & 8h horizons)
  2. LightGBM composite direction accuracy alone
  3. Agreement hit rate — when BOTH models agree on direction

If agreement hit rate > either model alone by ≥5pp, Kronos earns its place
in the live projections pipeline.

Run:
    python3 kronos_backtest.py
    python3 kronos_backtest.py --coin ZEC
    python3 kronos_backtest.py --coin FARTCOIN --context 200 --step 6

Outputs:
    output/kronos_backtest_{coin}.csv   — per-step results
    output/kronos_backtest_{coin}.png   — accuracy curves
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Add Kronos model code to path
sys.path.insert(0, str(PROJECT_DIR / "kronos_model"))
from kronos import KronosTokenizer, Kronos, KronosPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CARRY_COST      = 0.0045   # 0.45%/4h — Bybit floor; a long needs to beat this
HORIZONS        = [4, 8]   # hours ahead to evaluate
LOOKBACK        = 200      # hourly candles used as context per inference
STEP            = 6        # walk-forward step size in hours (avoids overlapping windows)
MIN_CONTEXT     = 48       # minimum history required before first prediction
LGBM_THRESHOLD  = 0.03     # composite > this = LightGBM says UP (calibrated to ~50th pct)

KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
KRONOS_MODEL_ID     = "NeoQuasar/Kronos-mini"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(coin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load OHLCV and signals for the given coin. Returns (ohlcv_df, signals_df)."""
    from coin_config import get_config
    cfg = get_config(coin)
    cmc = cfg["cmc_symbol"]
    perp = cfg["perp_symbol"]

    # OHLCV
    ohlcv_path = DATA_DIR / f"{cmc}_ohlcv_hourly.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")
    ohlcv = pd.read_csv(ohlcv_path, parse_dates=["timestamp"])
    ohlcv = ohlcv.set_index("timestamp").sort_index()
    ohlcv = ohlcv[["open", "high", "low", "close", "volume"]].resample("1h").last().ffill()
    ohlcv = ohlcv.dropna(subset=["close"])

    # Signals (for LightGBM composite proxy)
    sig_path = DATA_DIR / f"signals_{cmc}.csv"
    if not sig_path.exists():
        sig_path = DATA_DIR / "signals.csv"  # fallback
    if sig_path.exists():
        sig = pd.read_csv(sig_path, parse_dates=["timestamp"])
        sig = sig.set_index("timestamp").sort_index()
        sig = sig[["composite"]].resample("1h").mean()
    else:
        # No signals file — create empty df aligned to ohlcv
        sig = pd.DataFrame({"composite": np.nan}, index=ohlcv.index)

    # Align
    sig = sig.reindex(ohlcv.index, method="ffill")
    return ohlcv, sig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Kronos model (cached — load once)
# ─────────────────────────────────────────────────────────────────────────────

def load_kronos(max_context: int = 2048) -> KronosPredictor:
    """Download and initialise Kronos-mini with Tokenizer-2k."""
    print(f"  Loading tokenizer from {KRONOS_TOKENIZER_ID}...")
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID)

    print(f"  Loading model from {KRONOS_MODEL_ID}...")
    model = Kronos.from_pretrained(KRONOS_MODEL_ID)
    model.eval()

    device = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda:0" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-forward evaluation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    predictor: KronosPredictor,
    lookback: int,
    step: int,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Walk-forward validation. At each step:
      - Feed last `lookback` candles to Kronos → get 4h and 8h forecasts
      - Read LightGBM composite direction from signals
      - Record actual forward returns
      - Score: direction correct vs actual (long = fwd_ret > carry_cost, short = fwd_ret < 0)
    """
    max_horizon = max(horizons)
    rows = []

    # Build index of evaluation points
    eval_indices = list(range(lookback, len(ohlcv) - max_horizon, step))
    n_eval = len(eval_indices)

    print(f"\n  Walk-forward: {n_eval} evaluation points "
          f"(lookback={lookback}h, step={step}h, horizon={horizons})")

    for i, idx in enumerate(eval_indices):
        if i % 20 == 0:
            pct = i / n_eval * 100
            ts  = ohlcv.index[idx].strftime("%Y-%m-%d %H:%M")
            print(f"    [{pct:4.0f}%] {ts} ({i}/{n_eval})", end="\r")

        # Context window
        ctx_df   = ohlcv.iloc[idx - lookback : idx].copy()
        ctx_ts   = ctx_df.index.to_series().reset_index(drop=True)
        current_price = ohlcv["close"].iloc[idx]
        ts_now   = ohlcv.index[idx]

        # Future timestamps for prediction
        freq_h   = pd.tseries.frequencies.to_offset("1h")
        fut_ts   = pd.Series([ts_now + freq_h * (h + 1) for h in range(max_horizon)])

        # ── Kronos prediction ──
        try:
            pred_df = predictor.predict(
                df=ctx_df,
                x_timestamp=ctx_ts,
                y_timestamp=fut_ts,
                pred_len=max_horizon,
                T=1.0,
                top_p=0.9,
                sample_count=3,   # 3 paths averaged — balance speed vs noise
                verbose=False,
            )
        except Exception as e:
            # Skip this step if Kronos errors (e.g. NaN in context)
            continue

        # ── LightGBM composite ──
        try:
            lgbm_composite = signals["composite"].reindex([ts_now], method="nearest").iloc[0]
        except Exception:
            lgbm_composite = np.nan

        lgbm_up = lgbm_composite > LGBM_THRESHOLD if not np.isnan(lgbm_composite) else None

        # ── Record results per horizon ──
        row = {"timestamp": ts_now, "current_price": current_price,
               "lgbm_composite": lgbm_composite}

        for h in horizons:
            fwd_price  = ohlcv["close"].iloc[idx + h]
            fwd_ret    = fwd_price / current_price - 1

            # Kronos direction: predicted close h steps ahead vs current
            kronos_close = pred_df["close"].iloc[h - 1]
            kronos_up    = kronos_close > current_price

            # Predicted high/low over the horizon (for stop/target quality check)
            kronos_high = pred_df["high"].iloc[:h].max()
            kronos_low  = pred_df["low"].iloc[:h].min()

            # Agreement
            agree = (lgbm_up is not None) and (kronos_up == lgbm_up)

            # Hit: did taking the indicated direction beat carry cost?
            # Long hit: fwd_ret > carry_cost | Short hit: fwd_ret < 0
            kronos_long_hit = fwd_ret > CARRY_COST  if kronos_up  else fwd_ret < 0
            lgbm_long_hit   = fwd_ret > CARRY_COST  if lgbm_up    else fwd_ret < 0
            agree_hit       = kronos_long_hit if agree else None

            # Target/stop quality: did actual price reach predicted high before low?
            long_target_hit = fwd_price >= kronos_high * 0.998  # within 0.2%
            stop_hit        = fwd_price <= kronos_low  * 1.002

            row.update({
                f"fwd_ret_{h}h":          round(fwd_ret, 6),
                f"kronos_close_{h}h":     round(kronos_close, 6),
                f"kronos_up_{h}h":        kronos_up,
                f"kronos_hit_{h}h":       kronos_long_hit,
                f"lgbm_up_{h}h":          lgbm_up,
                f"lgbm_hit_{h}h":         lgbm_long_hit,
                f"agree_{h}h":            agree,
                f"agree_hit_{h}h":        agree_hit,
                f"kronos_high_{h}h":      round(kronos_high, 6),
                f"kronos_low_{h}h":       round(kronos_low, 6),
                f"target_hit_{h}h":       long_target_hit,
                f"stop_hit_{h}h":         stop_hit,
            })

        rows.append(row)

    print(f"\n  Done. {len(rows)} evaluation points completed.")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Score and print results
# ─────────────────────────────────────────────────────────────────────────────

def score(results: pd.DataFrame, coin: str):
    print(f"\n{'='*70}")
    print(f"  KRONOS ZERO-SHOT BACKTEST — {coin}")
    print(f"  {len(results)} evaluation points | "
          f"carry cost: {CARRY_COST*100:.2f}%/4h | step: {STEP}h")
    print(f"{'='*70}")

    for h in HORIZONS:
        kr_hit   = results[f"kronos_hit_{h}h"].mean()
        lg_hit   = results[f"lgbm_hit_{h}h"].dropna().mean()
        ag_mask  = results[f"agree_{h}h"]
        ag_hit   = results.loc[ag_mask, f"agree_hit_{h}h"].dropna().mean()
        n_agree  = ag_mask.sum()
        n_total  = len(results)

        # Lift = agreement hit rate minus the best single model
        best_solo = max(kr_hit, lg_hit)
        lift      = (ag_hit - best_solo) * 100 if not np.isnan(ag_hit) else 0

        print(f"\n  ── {h}h Horizon ──")
        print(f"  Kronos alone:    {kr_hit:.1%}  (n={n_total})")
        print(f"  LightGBM alone:  {lg_hit:.1%}  (n={results[f'lgbm_hit_{h}h'].notna().sum()})")
        print(f"  Agreement:       {ag_hit:.1%}  (n={n_agree}, "
              f"{n_agree/n_total:.0%} of trades) | lift: {lift:+.1f}pp")

        verdict = (
            "✅ VALIDATED — wire into pipeline" if ag_hit >= 0.60 and lift >= 5 and n_agree >= 20
            else "⚠️  MARGINAL — useful but not conclusive" if ag_hit >= 0.55 and n_agree >= 10
            else "❌ NO EDGE — skip integration"
        )
        print(f"  Verdict: {verdict}")

    # Stop / target quality (4h)
    th4 = results["target_hit_4h"].mean() if "target_hit_4h" in results else None
    sh4 = results["stop_hit_4h"].mean()   if "stop_hit_4h"   in results else None
    if th4 is not None:
        print(f"\n  ── Price Path Quality (4h) ──")
        print(f"  Predicted high reached by actual:  {th4:.1%}")
        print(f"  Predicted low breached by actual:  {sh4:.1%}")
        print(f"  → High accuracy: how reliable Kronos targets are as take-profit levels")

    # Monthly breakdown
    results["month"] = pd.to_datetime(results["timestamp"]).dt.to_period("M").astype(str)
    print(f"\n  ── Monthly Hit Rate (4h, Agreement only) ──")
    monthly = (
        results[results["agree_4h"]]
        .groupby("month")
        .agg(
            agree_hit=("agree_hit_4h", "mean"),
            n=("agree_hit_4h", "count"),
        )
    )
    for mo, row in monthly.iterrows():
        bar = "█" * int(row["agree_hit"] * 20)
        print(f"  {mo}  {bar:<20} {row['agree_hit']:.0%}  (n={int(row['n'])})")

    print(f"{'='*70}\n")
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: pd.DataFrame, coin: str):
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ts = pd.to_datetime(results["timestamp"])

    for i, h in enumerate(HORIZONS):
        ax = fig.add_subplot(gs[0, i])

        # Rolling 30-step hit rates
        window = 30
        kr_roll = results[f"kronos_hit_{h}h"].rolling(window).mean()
        lg_roll = results[f"lgbm_hit_{h}h"].rolling(window).mean()
        ag_data = results.loc[results[f"agree_{h}h"], f"agree_hit_{h}h"].copy()
        ag_ts   = ts[results[f"agree_{h}h"]]
        ag_roll = ag_data.rolling(min(window, len(ag_data))).mean()

        ax.plot(ts, kr_roll, label="Kronos", color="#4fc3f7", linewidth=1.5)
        ax.plot(ts, lg_roll, label="LightGBM", color="#a5d6a7", linewidth=1.5)
        ax.plot(ag_ts, ag_roll, label="Agreement", color="#ffb74d", linewidth=2.0)
        ax.axhline(0.50, color="#ef9a9a", linewidth=1, linestyle="--", alpha=0.7, label="50%")
        ax.axhline(0.60, color="#80cbc4", linewidth=0.8, linestyle=":", alpha=0.7, label="60%")
        ax.set_title(f"{h}h Hit Rate (rolling {window})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Hit Rate")
        ax.set_ylim(0.2, 0.9)
        ax.legend(fontsize=7)

    # Agreement frequency over time
    ax3 = fig.add_subplot(gs[1, 0])
    agree_roll = results["agree_4h"].astype(float).rolling(30).mean()
    ax3.fill_between(ts, agree_roll, alpha=0.6, color="#7986cb")
    ax3.set_title("Agreement Rate (4h, rolling 30)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Fraction agreeing")
    ax3.set_ylim(0, 1)

    # Cumulative PnL comparison
    ax4 = fig.add_subplot(gs[1, 1])
    for label, col, color in [
        ("Kronos", f"kronos_hit_{HORIZONS[0]}h", "#4fc3f7"),
        ("LightGBM", f"lgbm_hit_{HORIZONS[0]}h", "#a5d6a7"),
    ]:
        pnl = results[col].map({True: CARRY_COST, False: -CARRY_COST}).cumsum()
        ax4.plot(ts, pnl, label=label, color=color, linewidth=1.5)

    ag_pnl = (
        results.loc[results["agree_4h"], f"agree_hit_4h"]
        .map({True: CARRY_COST, False: -CARRY_COST})
        .cumsum()
        .reindex(results.index, method="ffill")
        .fillna(0)
    )
    ax4.plot(ts, ag_pnl, label="Agreement", color="#ffb74d", linewidth=2.0)
    ax4.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    ax4.set_title(f"Cumulative PnL (4h, {CARRY_COST*100:.2f}% per step)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Cumulative %")
    ax4.legend(fontsize=7)

    fig.suptitle(f"Kronos Zero-Shot Backtest — {coin} (Jan–Apr 2026)",
                 fontsize=13, fontweight="bold", y=0.98)

    out = OUTPUT_DIR / f"kronos_backtest_{coin}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[Chart saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos zero-shot backtest")
    parser.add_argument("--coin",    default="FARTCOIN", help="Coin to backtest")
    parser.add_argument("--context", type=int, default=LOOKBACK,
                        help=f"Context window in hours (default: {LOOKBACK})")
    parser.add_argument("--step",    type=int, default=STEP,
                        help=f"Walk-forward step size in hours (default: {STEP})")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  KRONOS ZERO-SHOT BACKTEST  |  Coin: {args.coin}")
    print(f"  Context: {args.context}h  |  Step: {args.step}h")
    print(f"{'='*70}\n")

    # Load data
    print("[1/4] Loading historical data...")
    ohlcv, signals = load_data(args.coin)
    print(f"  OHLCV: {len(ohlcv)} rows ({ohlcv.index[0].date()} → {ohlcv.index[-1].date()})")
    print(f"  Signals: {signals['composite'].notna().sum()} non-null composite rows")

    # Load model
    print("\n[2/4] Loading Kronos-mini...")
    predictor = load_kronos(max_context=2048)

    # Walk-forward
    print("\n[3/4] Running walk-forward evaluation...")
    results = walk_forward(
        ohlcv=ohlcv,
        signals=signals,
        predictor=predictor,
        lookback=args.context,
        step=args.step,
        horizons=HORIZONS,
    )

    # Save raw results
    results_path = OUTPUT_DIR / f"kronos_backtest_{args.coin}.csv"
    results.to_csv(results_path, index=False)
    print(f"  Raw results → {results_path}")

    # Score
    print("\n[4/4] Scoring results...")
    score(results, args.coin)

    # Plot
    plot_results(results, args.coin)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
