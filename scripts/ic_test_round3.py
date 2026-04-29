"""
Permutation IC Test — Round 3 Candidates

Tests 4 candidate features for inclusion in META_FEATURES:
  1. sig_pv_divergence    (already in df, free win)
  2. sig_dex_liq_div      (already in df, marginal coverage)
  3. liq_long_short_ratio_z  (engineered from coinalyze_liquidations.csv)
  4. funding_spread_z        (engineered from coinalyze_funding_history.csv)

Threshold: p < 0.10 to advance to Phase 2 (walk-forward backtest)
Seed: 42 (reproducible across rounds)
N_PERM: 1000

Run from project root: python3 scripts/ic_test_round3.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from signal_engine import load_data, compute_all_signals
from trade_scorer import build_meta_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_PERM   = 1000
SEED     = 42

# ── Load data + INJECT SIGNALS + build feature matrix ─────────────────────────
# CRITICAL: load_data() does NOT load signals_FARTCOIN.csv into data['signals'].
# build_meta_features() requires signals injected via compute_all_signals(data),
# otherwise all sig_* META_FEATURES are silently dropped and composite=0.
# This is what automation.py and trade_scorer.__main__ do before calling build_meta_features.
print("Loading data + computing signals + building feature matrix...")
data = load_data()
signals = compute_all_signals(data)
data["signals"] = signals
print(f"  Signals computed: {len(signals)} rows | sig_* cols: {[c for c in signals.columns if c.startswith('sig_')]}")
df   = build_meta_features(data)
n    = len(df)
ohlcv = data["ohlcv"]
print(f"  Feature matrix: {df.shape}")

target = df["fwd_ret_4h"].values
valid_target = ~np.isnan(target)

# ── Candidate 1 & 2: free wins (already in df via signal_engine) ──────────────
candidates = {}

if "sig_pv_divergence" in df.columns:
    s = df["sig_pv_divergence"].values
    candidates["sig_pv_divergence"] = (s, f"{(s != 0).sum()}/{n} non-zero, var={s.std():.3f}")
else:
    print("WARN: sig_pv_divergence missing from df")

if "sig_dex_liq_div" in df.columns:
    s = df["sig_dex_liq_div"].values
    candidates["sig_dex_liq_div"] = (s, f"{(s != 0).sum()}/{n} non-zero, var={s.std():.3f}")
else:
    print("WARN: sig_dex_liq_div missing from df")

# ── Candidate 3: liq_long_short_ratio_z ──────────────────────────────────────
liq = pd.read_csv(os.path.join(DATA_DIR, "coinalyze_liquidations.csv"),
                  index_col=0, parse_dates=True)
liq.index = liq.index.tz_localize(None) if liq.index.tz is not None else liq.index
ohlcv_idx = ohlcv.index.tz_localize(None) if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None else ohlcv.index
liq_ratio = liq["liq_ratio"]
liq_aligned = liq_ratio.reindex(ohlcv_idx[:n], method="ffill").fillna(0.5)
liq_z = ((liq_aligned - liq_aligned.rolling(168, min_periods=24).mean()) /
         (liq_aligned.rolling(168, min_periods=24).std() + 1e-9)).clip(-3, 3).fillna(0)
candidates["liq_long_short_ratio_z"] = (
    liq_z.values,
    f"raw rows: {len(liq)} | aligned non-zero: {(liq_z != 0).sum()}/{n} | var={liq_z.std():.3f}"
)

# ── Candidate 4: funding_spread_z ─────────────────────────────────────────────
fh = pd.read_csv(os.path.join(DATA_DIR, "coinalyze_funding_history.csv"),
                 index_col=0, parse_dates=True)
fh.index = fh.index.tz_localize(None) if fh.index.tz is not None else fh.index
exch_cols = [c for c in fh.columns if c not in ("mean_funding", "funding_spread")]
spread = fh[exch_cols].max(axis=1) - fh[exch_cols].min(axis=1)
spread_aligned = spread.reindex(ohlcv_idx[:n], method="ffill").fillna(0)
spread_z = ((spread_aligned - spread_aligned.rolling(168, min_periods=24).mean()) /
            (spread_aligned.rolling(168, min_periods=24).std() + 1e-9)).clip(-3, 3).fillna(0)
candidates["funding_spread_z"] = (
    spread_z.values,
    f"raw rows: {len(fh)} | aligned non-zero: {(spread_z != 0).sum()}/{n} | var={spread_z.std():.3f}"
)

# ── Permutation IC test (1000 shuffles, seed=42) ──────────────────────────────
print("\n" + "=" * 78)
print(f"PERMUTATION IC TEST  (N={N_PERM} shuffles, seed={SEED})")
print("=" * 78)
hdr = "%-26s %8s %8s %8s %12s   %s" % ("Feature", "IC", "z-score", "p-value", "verdict", "diagnostics")
print(hdr)
print("-" * 110)

np.random.seed(SEED)
results = {}

for name, (feat, diag) in candidates.items():
    feat_arr = np.array(feat, dtype=float)
    mask = valid_target & ~np.isnan(feat_arr)
    f_v = feat_arr[mask]
    t_v = target[mask]
    if len(f_v) < 100 or f_v.std() < 1e-10:
        verdict = "INSUFFICIENT"
        ic = z = p = float('nan')
    else:
        ic = float(np.corrcoef(f_v, t_v)[0, 1])
        perm_ics = np.array([
            np.corrcoef(np.random.permutation(f_v), t_v)[0, 1]
            for _ in range(N_PERM)
        ])
        z = (ic - perm_ics.mean()) / (perm_ics.std() + 1e-12)
        p = float((np.abs(perm_ics) >= np.abs(ic)).mean())
        verdict = "✅ ADVANCE" if p < 0.10 else ("⚠️ MARGINAL" if p < 0.20 else "❌ REJECT")

    results[name] = {"ic": ic, "z": z, "p": p, "verdict": verdict, "diag": diag}
    print("%-26s %8.4f %8.2f %8.3f %12s   %s" % (name, ic, z, p, verdict, diag))

print()
advance = [n for n, r in results.items() if r["verdict"].startswith("✅")]
print(f"Advancing to Phase 2 (walk-forward backtest): {advance if advance else 'NONE'}")
