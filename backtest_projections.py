"""
Projection Engine Backtest — Fartcoin Alpha Framework

Walks through historical data and evaluates each projection model's accuracy:
1. Probability calibration: does "60% bullish" actually hit 60% of the time?
2. Mean-reversion timing: does funding/LSR actually revert within predicted half-life?
3. Manipulation cycle: do BUILDUP/SPIKE phases precede real price moves?
4. Session-conditional edge: does the combined edge predict actual returns?
5. BTC lead-lag: does the beta/correlation hold out-of-sample?

Run: python3 backtest_projections.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime, timezone

from signal_engine import load_data, compute_all_signals
from market_state import (
    SESSION_MAP, HOURLY_BIAS, classify_session, classify_asia_sub,
    ASIA_DAY_BPS,
)
from projections import _build_feature_matrix

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _logistic(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _neg_log_likelihood(theta, X, y):
    p = _logistic(X @ theta)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# =========================================================================
# 1. Probability Calibration Backtest
# =========================================================================

def backtest_probability(signals, ohlcv, train_frac=0.5):
    """
    Walk-forward probability calibration test.
    Train on first half, predict on second half.
    Bin predictions into deciles and compare predicted vs actual hit rate.
    """
    print("\n" + "=" * 70)
    print("1. PROBABILITY CALIBRATION BACKTEST")
    print("=" * 70)

    price_col = "price" if "price" in ohlcv.columns else "close"

    # Align
    common_idx = signals.index.intersection(ohlcv.index)
    sig_cols = [c for c in signals.columns if c in ["composite", "sig_funding"]]
    df = signals.loc[common_idx, sig_cols].copy()
    df["fwd_ret_4h"] = ohlcv.loc[common_idx, price_col].pct_change(4).shift(-4)
    df["hour"] = df.index.hour if hasattr(df.index, 'hour') else 12
    df["session"] = df["hour"].apply(classify_session)
    df = df.dropna(subset=["fwd_ret_4h", "composite"])

    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split]
    test = df.iloc[split:]

    print(f"  Total observations: {n}")
    print(f"  Train: {split} | Test: {n - split}")
    print(f"  Features: 10 (composite, |composite|, composite², hourly_bias, 3 session dummies, sig_funding, btc_regime)")

    # Train enriched logistic regression
    X_train = _build_feature_matrix(train)
    y_train = (train["fwd_ret_4h"].values > 0).astype(float)

    theta0 = np.zeros(X_train.shape[1])
    res = minimize(_neg_log_likelihood, theta0, args=(X_train, y_train),
                   method="L-BFGS-B", options={"maxiter": 300})
    theta = res.x

    # Predict on test set
    X_test = _build_feature_matrix(test)
    predicted_prob = _logistic(X_test @ theta)

    # Apply session-direction filters (London LONG penalty, Late NYC SHORT boost)
    for i in range(len(predicted_prob)):
        sess = test["session"].iloc[i]
        if sess == "London" and predicted_prob[i] > 0.5:
            predicted_prob[i] = 0.5 + (predicted_prob[i] - 0.5) * 0.5
        elif sess == "Late NYC" and predicted_prob[i] < 0.5:
            predicted_prob[i] = predicted_prob[i] * 0.85

    actual_positive = (test["fwd_ret_4h"].values > 0).astype(float)

    # Calibration: bin into deciles
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{bins[i]:.0%}-{bins[i+1]:.0%}" for i in range(len(bins) - 1)]
    bin_idx = np.digitize(predicted_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)

    print(f"\n  {'Predicted Bin':<16} {'Predicted':>10} {'Actual':>10} {'Count':>8} {'Error':>8}")
    print("  " + "-" * 56)

    calibration_rows = []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        pred_avg = predicted_prob[mask].mean()
        actual_avg = actual_positive[mask].mean()
        count = mask.sum()
        error = abs(pred_avg - actual_avg)
        print(f"  {bin_labels[i]:<16} {pred_avg:>10.1%} {actual_avg:>10.1%} {count:>8} {error:>8.1%}")
        calibration_rows.append({
            "bin": bin_labels[i],
            "predicted": pred_avg,
            "actual": actual_avg,
            "count": count,
            "error": error,
        })

    # Overall metrics
    from scipy import stats as sp_stats
    brier_score = np.mean((predicted_prob - actual_positive) ** 2)
    auc_proxy = sp_stats.kendalltau(predicted_prob, actual_positive).statistic

    # Directional accuracy at extremes
    high_prob_mask = predicted_prob > 0.6
    low_prob_mask = predicted_prob < 0.4
    mid_mask = ~high_prob_mask & ~low_prob_mask

    print(f"\n  Overall Metrics:")
    print(f"  Brier Score:     {brier_score:.4f} (lower is better, 0.25 = random)")
    print(f"  Kendall Tau:     {auc_proxy:.4f} (rank correlation of prob vs outcome)")

    if high_prob_mask.sum() > 0:
        print(f"\n  High Prob (>60%): {actual_positive[high_prob_mask].mean():.1%} actual hit rate (n={high_prob_mask.sum()})")
    if low_prob_mask.sum() > 0:
        print(f"  Low Prob  (<40%): {actual_positive[low_prob_mask].mean():.1%} actual hit rate (n={low_prob_mask.sum()})")
    if mid_mask.sum() > 0:
        print(f"  Mid Prob (40-60%): {actual_positive[mid_mask].mean():.1%} actual hit rate (n={mid_mask.sum()})")

    # Strategy return: go long when prob > 0.6, short when prob < 0.4
    strat_returns = []
    for prob_val, ret_val in zip(predicted_prob, test["fwd_ret_4h"].values):
        if prob_val > 0.6:
            strat_returns.append(ret_val)
        elif prob_val < 0.4:
            strat_returns.append(-ret_val)
    strat_returns = np.array(strat_returns)

    if len(strat_returns) > 0:
        print(f"\n  Strategy (prob-weighted entry):")
        print(f"  Trades: {len(strat_returns)}")
        print(f"  Mean return: {strat_returns.mean() * 100:+.2f}%")
        print(f"  Hit rate: {(strat_returns > 0).mean():.1%}")
        print(f"  Sharpe (annualized): {strat_returns.mean() / strat_returns.std() * np.sqrt(252 * 6):.2f}" if strat_returns.std() > 0 else "  Sharpe: N/A")
        print(f"  Cumulative return: {(1 + strat_returns).prod() - 1:.1%}")

    cal_df = pd.DataFrame(calibration_rows)
    cal_df.to_csv(OUTPUT_DIR / "projection_calibration.csv", index=False)

    return {
        "brier_score": brier_score,
        "kendall_tau": auc_proxy,
        "n_test": len(test),
        "calibration": cal_df,
    }


# =========================================================================
# 2. Mean-Reversion Backtest
# =========================================================================

def backtest_mean_reversion(funding_df, lsr_df):
    """
    Test: when funding/LSR deviates > 2σ, does it actually revert within
    the predicted half-life window?
    """
    print("\n" + "=" * 70)
    print("2. MEAN-REVERSION TIMING BACKTEST")
    print("=" * 70)

    results = {}

    for name, df, col_name in [
        ("Funding", funding_df, "fundingRate"),
        ("LSR", lsr_df, None),
    ]:
        if df is None or df.empty:
            print(f"\n  {name}: No data available")
            continue

        col = col_name if col_name and col_name in df.columns else df.columns[0]
        series = df[col].dropna()

        if len(series) < 100:
            print(f"\n  {name}: Insufficient data ({len(series)} rows)")
            continue

        mean = series.mean()
        std = series.std()

        # AR(1) estimation
        y = series.values[1:]
        y_lag = series.values[:-1]
        phi = np.cov(y, y_lag)[0, 1] / np.var(y_lag) if np.var(y_lag) > 0 else 0.5
        phi = np.clip(phi, 0.01, 0.999)
        half_life = -np.log(2) / np.log(phi)

        # Find extreme events (> 2σ deviation)
        upper_threshold = mean + 2 * std
        lower_threshold = mean - 2 * std

        extreme_high = series[series > upper_threshold]
        extreme_low = series[series < lower_threshold]

        print(f"\n  {name}:")
        print(f"  AR(1) phi: {phi:.4f} | Half-life: {half_life:.1f}h")
        print(f"  Mean: {mean:.6f} | Std: {std:.6f}")
        print(f"  Extreme high events (>2σ): {len(extreme_high)}")
        print(f"  Extreme low events (<-2σ): {len(extreme_low)}")

        # For each extreme event, check if it reverted within 1x, 2x, 3x half-life
        for direction, extremes, threshold in [
            ("High", extreme_high, upper_threshold),
            ("Low", extreme_low, lower_threshold),
        ]:
            if len(extremes) == 0:
                continue

            revert_1x = 0
            revert_2x = 0
            revert_3x = 0
            total = 0

            for idx_pos in range(len(series)):
                if series.index[idx_pos] not in extremes.index:
                    continue
                total += 1
                hl_1 = int(half_life)
                hl_2 = int(half_life * 2)
                hl_3 = int(half_life * 3)

                for window, counter_name in [(hl_1, "1x"), (hl_2, "2x"), (hl_3, "3x")]:
                    end = min(idx_pos + window + 1, len(series))
                    future = series.iloc[idx_pos + 1:end]
                    if len(future) == 0:
                        continue
                    if direction == "High" and future.min() < mean:
                        if counter_name == "1x": revert_1x += 1
                        elif counter_name == "2x": revert_2x += 1
                        else: revert_3x += 1
                    elif direction == "Low" and future.max() > mean:
                        if counter_name == "1x": revert_1x += 1
                        elif counter_name == "2x": revert_2x += 1
                        else: revert_3x += 1

            if total > 0:
                print(f"\n    {direction} extremes ({total} events):")
                print(f"    Reverted within 1x half-life ({int(half_life)}h): {revert_1x}/{total} = {revert_1x/total:.0%}")
                print(f"    Reverted within 2x half-life ({int(half_life*2)}h): {revert_2x}/{total} = {revert_2x/total:.0%}")
                print(f"    Reverted within 3x half-life ({int(half_life*3)}h): {revert_3x}/{total} = {revert_3x/total:.0%}")

        results[name.lower()] = {
            "phi": phi,
            "half_life_h": half_life,
            "n_extreme_high": len(extreme_high),
            "n_extreme_low": len(extreme_low),
        }

    return results


# =========================================================================
# 3. Manipulation Cycle Backtest
# =========================================================================

def backtest_manipulation_cycle(ohlcv, oi_df, signals):
    """
    Walk through history detecting BUILDUP/SPIKE phases and measure
    what actually happened to price in the following 4-8 hours.
    """
    print("\n" + "=" * 70)
    print("3. MANIPULATION CYCLE BACKTEST")
    print("=" * 70)

    if ohlcv is None or oi_df is None or signals is None:
        print("  Insufficient data")
        return {}

    price_col = "price" if "price" in ohlcv.columns else "close"
    oi_col = oi_df.columns[0]

    # Align all data
    common = ohlcv.index.intersection(oi_df.index).intersection(signals.index)
    if len(common) < 50:
        print("  Insufficient aligned data")
        return {}

    price = ohlcv.loc[common, price_col]
    oi = oi_df.loc[common, oi_col]
    vol = ohlcv.loc[common, "volume"]
    sig_oi_accel = signals.loc[common, "sig_oi_accel"] if "sig_oi_accel" in signals.columns else pd.Series(0, index=common)

    vol_24h_mean = vol.rolling(24).mean()
    vol_ratio = vol / vol_24h_mean
    oi_4h_pct = oi.pct_change(4)

    # Forward returns
    fwd_4h = price.pct_change(4).shift(-4)
    fwd_8h = price.pct_change(8).shift(-8)
    fwd_abs_4h = fwd_4h.abs()

    # Detect phases at each hour
    events = {"DORMANT": [], "QUIET_ACCUMULATION": [], "BUILDUP": [], "SPIKE_IN_PROGRESS": []}

    for i in range(12, len(common)):
        vr = vol_ratio.iloc[i]
        oi_chg = oi_4h_pct.iloc[i]
        oi_acc = sig_oi_accel.iloc[i]
        last_3_vr = vol_ratio.iloc[max(0, i-3):i].values

        if pd.isna(vr) or pd.isna(oi_chg):
            continue

        if vr > 1.5 and abs(oi_chg) > 0.05:
            phase = "SPIKE_IN_PROGRESS"
        elif oi_acc > 0.5 and oi_chg > 0.02:
            phase = "BUILDUP"
        elif len(last_3_vr) >= 3 and all(v < 0.7 for v in last_3_vr if not np.isnan(v)) and oi_chg > 0.02:
            phase = "QUIET_ACCUMULATION"
        else:
            phase = "DORMANT"

        ret4 = fwd_4h.iloc[i] if i < len(fwd_4h) and not pd.isna(fwd_4h.iloc[i]) else None
        ret8 = fwd_8h.iloc[i] if i < len(fwd_8h) and not pd.isna(fwd_8h.iloc[i]) else None
        abs4 = fwd_abs_4h.iloc[i] if i < len(fwd_abs_4h) and not pd.isna(fwd_abs_4h.iloc[i]) else None

        events[phase].append({"ret_4h": ret4, "ret_8h": ret8, "abs_4h": abs4})

    # Report
    for phase in ["DORMANT", "QUIET_ACCUMULATION", "BUILDUP", "SPIKE_IN_PROGRESS"]:
        evts = events[phase]
        valid = [e for e in evts if e["ret_4h"] is not None]
        if not valid:
            print(f"\n  {phase}: 0 events detected")
            continue

        rets_4h = np.array([e["ret_4h"] for e in valid])
        rets_8h = np.array([e["ret_8h"] for e in valid if e["ret_8h"] is not None])
        abs_4h = np.array([e["abs_4h"] for e in valid])

        print(f"\n  {phase}:")
        print(f"    Events detected: {len(valid)}")
        print(f"    Avg 4h return: {rets_4h.mean() * 100:+.2f}%")
        print(f"    Avg |4h return|: {abs_4h.mean() * 100:.2f}% (volatility)")
        if len(rets_8h) > 0:
            print(f"    Avg 8h return: {rets_8h.mean() * 100:+.2f}%")
        print(f"    Hit rate (4h positive): {(rets_4h > 0).mean():.0%}")

    # Comparison: BUILDUP/SPIKE vs DORMANT
    buildup_spike = events["BUILDUP"] + events["SPIKE_IN_PROGRESS"]
    dormant = events["DORMANT"]

    bs_valid = [e for e in buildup_spike if e["abs_4h"] is not None]
    d_valid = [e for e in dormant if e["abs_4h"] is not None]

    if bs_valid and d_valid:
        bs_vol = np.mean([e["abs_4h"] for e in bs_valid])
        d_vol = np.mean([e["abs_4h"] for e in d_valid])
        print(f"\n  BUILDUP+SPIKE avg |4h move|: {bs_vol * 100:.2f}%")
        print(f"  DORMANT avg |4h move|:        {d_vol * 100:.2f}%")
        print(f"  Volatility ratio:             {bs_vol / d_vol:.1f}x")

    return events


# =========================================================================
# 4. Session-Conditional Edge Backtest
# =========================================================================

def backtest_session_conditional(signals, ohlcv):
    """
    Test: does the session-conditional combined edge predict actual returns?
    """
    print("\n" + "=" * 70)
    print("4. SESSION-CONDITIONAL EDGE BACKTEST")
    print("=" * 70)

    price_col = "price" if "price" in ohlcv.columns else "close"

    common_idx = signals.index.intersection(ohlcv.index)
    df = signals.loc[common_idx, ["composite"]].copy()
    df["fwd_ret_4h"] = ohlcv.loc[common_idx, price_col].pct_change(4).shift(-4)
    df["hour"] = df.index.hour if hasattr(df.index, 'hour') else 12
    df["session"] = df["hour"].apply(classify_session)
    df = df.dropna(subset=["fwd_ret_4h"])

    # For each session × direction combo, compute edge and actual return
    results = []
    for session in ["Asia", "London", "NYC", "Late NYC"]:
        for direction, comp_filter, label in [
            ("LONG", df["composite"] > 0.1, "LONG (composite>0.1)"),
            ("SHORT", df["composite"] < -0.1, "SHORT (composite<-0.1)"),
        ]:
            mask = (df["session"] == session) & comp_filter
            filtered = df[mask]
            if len(filtered) < 10:
                continue

            avg_ret = filtered["fwd_ret_4h"].mean()
            if direction == "SHORT":
                avg_ret = -avg_ret  # profit from shorts

            # Session bias
            hours = [h for h in range(24) if classify_session(h) == session]
            bias_sum = sum(HOURLY_BIAS.get(h, 0) for h in hours[:4]) / 100  # bps to pct

            hit_rate = (filtered["fwd_ret_4h"] > 0).mean() if direction == "LONG" else (filtered["fwd_ret_4h"] < 0).mean()

            results.append({
                "session": session,
                "direction": direction,
                "n": len(filtered),
                "avg_return_pct": avg_ret * 100,
                "session_bias_pct": bias_sum,
                "hit_rate": hit_rate,
            })

            print(f"  {session} | {label}: avg return {avg_ret * 100:+.2f}%, "
                  f"hit rate {hit_rate:.0%}, n={len(filtered)}")

    # Best and worst combos
    if results:
        res_df = pd.DataFrame(results)
        best = res_df.loc[res_df["avg_return_pct"].idxmax()]
        worst = res_df.loc[res_df["avg_return_pct"].idxmin()]
        print(f"\n  Best combo:  {best['session']} {best['direction']} = {best['avg_return_pct']:+.2f}% (n={best['n']:.0f})")
        print(f"  Worst combo: {worst['session']} {worst['direction']} = {worst['avg_return_pct']:+.2f}% (n={worst['n']:.0f})")

        res_df.to_csv(OUTPUT_DIR / "projection_session_conditional.csv", index=False)

    return results


# =========================================================================
# 5. BTC Lead-Lag Backtest
# =========================================================================

def backtest_btc_lead_lag(ohlcv, btc):
    """
    Test: does BTC actually lead Fartcoin by ~2h with 1.6x beta?
    Walk-forward beta estimation and out-of-sample prediction accuracy.
    """
    print("\n" + "=" * 70)
    print("5. BTC LEAD-LAG BACKTEST")
    print("=" * 70)

    if btc is None or ohlcv is None:
        print("  No BTC data available")
        return {}

    btc_col = "price" if "price" in btc.columns else "close"
    fart_col = "price" if "price" in ohlcv.columns else "close"

    # Round to hourly for alignment
    btc_h = btc[[btc_col]].copy()
    btc_h.index = btc_h.index.round("h")
    btc_h = btc_h[~btc_h.index.duplicated(keep="last")]

    fart_h = ohlcv[[fart_col]].copy()
    fart_h.index = fart_h.index.round("h")
    fart_h = fart_h[~fart_h.index.duplicated(keep="last")]

    common = btc_h.index.intersection(fart_h.index)
    if len(common) < 100:
        print(f"  Insufficient overlapping data ({len(common)} rows)")
        return {}

    btc_price = btc_h.loc[common, btc_col]
    fart_price = fart_h.loc[common, fart_col]

    # Test different lag windows
    print(f"  Overlapping hourly data: {len(common)} rows\n")
    print(f"  {'Lag':>6} {'Correlation':>14} {'Beta':>8} {'Direction Acc':>14}")
    print("  " + "-" * 46)

    best_corr = -1
    best_lag = 0

    for lag in range(0, 7):
        btc_ret = btc_price.pct_change(2).shift(lag)  # BTC move from lag hours ago
        fart_ret = fart_price.pct_change(4)  # Fart 4h forward return
        aligned = pd.DataFrame({"btc": btc_ret, "fart": fart_ret}).dropna()

        if len(aligned) < 50:
            continue

        corr = aligned["btc"].corr(aligned["fart"])
        try:
            beta = np.polyfit(aligned["btc"].values, aligned["fart"].values, 1)[0]
        except Exception:
            beta = 0

        # Direction accuracy: when BTC goes up, does FART go up?
        dir_acc = ((aligned["btc"] > 0) == (aligned["fart"] > 0)).mean()

        print(f"  {lag:>4}h {corr:>14.4f} {beta:>8.2f} {dir_acc:>14.1%}")

        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_lag = lag

    print(f"\n  Best lag: {best_lag}h (correlation: {best_corr:.4f})")

    # Walk-forward beta test: train on first 60%, predict last 40%
    split = int(len(common) * 0.6)

    btc_ret_full = btc_price.pct_change(2).shift(best_lag)
    fart_ret_full = fart_price.pct_change(4)
    aligned_full = pd.DataFrame({"btc": btc_ret_full, "fart": fart_ret_full}).dropna()

    train_aligned = aligned_full.iloc[:split]
    test_aligned = aligned_full.iloc[split:]

    if len(train_aligned) > 20 and len(test_aligned) > 20:
        train_beta = np.polyfit(train_aligned["btc"].values, train_aligned["fart"].values, 1)[0]
        predicted_fart = test_aligned["btc"] * train_beta
        actual_fart = test_aligned["fart"]

        pred_corr = predicted_fart.corr(actual_fart)
        mae = (predicted_fart - actual_fart).abs().mean() * 100
        dir_acc = ((predicted_fart > 0) == (actual_fart > 0)).mean()

        # Regime analysis: how does it perform in BTC rallies vs dumps?
        btc_up = test_aligned[test_aligned["btc"] > 0.01]
        btc_down = test_aligned[test_aligned["btc"] < -0.01]

        print(f"\n  Walk-Forward Test (train {split} rows, test {len(test_aligned)} rows):")
        print(f"  In-sample beta: {train_beta:.2f}")
        print(f"  Out-of-sample prediction correlation: {pred_corr:.4f}")
        print(f"  Mean absolute error: {mae:.2f}%")
        print(f"  Direction accuracy: {dir_acc:.1%}")

        if len(btc_up) > 10:
            up_acc = ((btc_up["btc"] * train_beta > 0) == (btc_up["fart"] > 0)).mean()
            print(f"  Direction acc (BTC rallies): {up_acc:.1%} (n={len(btc_up)})")
        if len(btc_down) > 10:
            down_acc = ((btc_down["btc"] * train_beta > 0) == (btc_down["fart"] > 0)).mean()
            print(f"  Direction acc (BTC dumps):   {down_acc:.1%} (n={len(btc_down)})")

    return {
        "best_lag": best_lag,
        "best_corr": best_corr,
    }


# =========================================================================
# 6. Combined Projection Strategy Backtest
# =========================================================================

def _run_strategy(df, train, test, label, apply_session_filters=False):
    """
    Helper: train logistic on train set, trade on test set, report stats.
    Returns dict of results.
    """
    X_train = _build_feature_matrix(train)
    y_train = (train["fwd_ret_4h"].values > 0).astype(float)

    theta0 = np.zeros(X_train.shape[1])
    res = minimize(_neg_log_likelihood, theta0, args=(X_train, y_train),
                   method="L-BFGS-B", options={"maxiter": 300})
    theta = res.x

    X_test = _build_feature_matrix(test)
    probs = _logistic(X_test @ theta)

    if apply_session_filters:
        for i in range(len(probs)):
            sess = test["session"].iloc[i]
            if sess == "London" and probs[i] > 0.5:
                probs[i] = 0.5 + (probs[i] - 0.5) * 0.5
            elif sess == "Late NYC" and probs[i] < 0.5:
                probs[i] = probs[i] * 0.85

    trades = []
    for i, (prob, ret, session) in enumerate(zip(probs, test["fwd_ret_4h"].values, test["session"].values)):
        if prob > 0.6:
            trades.append({"direction": "LONG", "prob": prob, "return": ret, "session": session})
        elif prob < 0.4:
            trades.append({"direction": "SHORT", "prob": prob, "return": -ret, "session": session})

    if not trades:
        return None

    trade_df = pd.DataFrame(trades)
    returns = trade_df["return"].values
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6) if returns.std() > 0 else 0
    cumulative = (1 + returns).prod() - 1

    return {
        "label": label,
        "n_trades": len(trades),
        "n_long": (trade_df["direction"] == "LONG").sum(),
        "n_short": (trade_df["direction"] == "SHORT").sum(),
        "mean_return": returns.mean(),
        "median_return": np.median(returns),
        "hit_rate": (returns > 0).mean(),
        "best": returns.max(),
        "worst": returns.min(),
        "cumulative": cumulative,
        "sharpe": sharpe,
        "trade_df": trade_df,
    }


def _print_strategy_result(r):
    """Print a strategy result dict."""
    if r is None:
        print("  No trades generated")
        return
    print(f"  Trades: {r['n_trades']} (L:{r['n_long']} / S:{r['n_short']})")
    print(f"  Mean return per trade: {r['mean_return'] * 100:+.2f}%")
    print(f"  Median return:         {r['median_return'] * 100:+.2f}%")
    print(f"  Hit rate:              {r['hit_rate']:.1%}")
    print(f"  Best / Worst:          {r['best'] * 100:+.2f}% / {r['worst'] * 100:+.2f}%")
    print(f"  Cumulative return:     {r['cumulative']:.1%}")
    print(f"  Sharpe ratio (ann.):   {r['sharpe']:.2f}")


def backtest_combined_strategy(signals, ohlcv):
    """
    6A: Honest backtest (no session filters) vs filtered — same 50/50 split.
    6B: Rolling walk-forward (train 168h/7d, test 48h/2d, roll forward).
    """
    print("\n" + "=" * 70)
    print("6. COMBINED PROJECTION STRATEGY — HONEST EVALUATION")
    print("=" * 70)

    price_col = "price" if "price" in ohlcv.columns else "close"

    common_idx = signals.index.intersection(ohlcv.index)
    sig_cols = [c for c in signals.columns if c in ["composite", "sig_funding"]]
    df = signals.loc[common_idx, sig_cols].copy()
    df["fwd_ret_4h"] = ohlcv.loc[common_idx, price_col].pct_change(4).shift(-4)
    df["hour"] = df.index.hour if hasattr(df.index, 'hour') else 12
    df["session"] = df["hour"].apply(classify_session)
    df = df.dropna(subset=["fwd_ret_4h", "composite"])

    n = len(df)
    split = int(n * 0.5)
    train = df.iloc[:split]
    test = df.iloc[split:]

    # --- 6A: Single split, compare filtered vs unfiltered ---
    print(f"\n  --- 6A: Single Split (train {split} / test {n - split}) ---")

    print(f"\n  WITHOUT session filters (honest):")
    r_honest = _run_strategy(df, train, test, "honest", apply_session_filters=False)
    _print_strategy_result(r_honest)

    print(f"\n  WITH session filters (potentially overfit):")
    r_filtered = _run_strategy(df, train, test, "filtered", apply_session_filters=True)
    _print_strategy_result(r_filtered)

    if r_honest and r_filtered:
        delta_sharpe = r_filtered["sharpe"] - r_honest["sharpe"]
        delta_cum = (r_filtered["cumulative"] - r_honest["cumulative"]) * 100
        print(f"\n  Filter impact: Sharpe {delta_sharpe:+.2f}, "
              f"Cumulative {delta_cum:+.1f}pp")
        if delta_sharpe > 0.5:
            print("  ⚠ Large filter boost — likely overfit to in-sample session patterns")
        elif delta_sharpe > 0:
            print("  Modest filter boost — worth monitoring out-of-sample")
        else:
            print("  Filters not helping — consider removing")

    # --- 6B: Rolling walk-forward ---
    print(f"\n  --- 6B: Rolling Walk-Forward (train=168h, test=48h, step=48h) ---")

    train_window = 168  # 7 days
    test_window = 48    # 2 days
    step = 48           # roll forward 2 days

    all_trades_honest = []
    all_trades_filtered = []
    window_results = []

    i = 0
    window_id = 0
    while i + train_window + test_window <= n:
        w_train = df.iloc[i:i + train_window]
        w_test = df.iloc[i + train_window:i + train_window + test_window]

        rh = _run_strategy(df, w_train, w_test, f"w{window_id}_honest",
                           apply_session_filters=False)
        rf = _run_strategy(df, w_train, w_test, f"w{window_id}_filtered",
                           apply_session_filters=True)

        if rh and rh["n_trades"] > 0:
            all_trades_honest.extend(rh["trade_df"].to_dict("records"))
        if rf and rf["n_trades"] > 0:
            all_trades_filtered.extend(rf["trade_df"].to_dict("records"))

        window_results.append({
            "window": window_id,
            "start": w_test.index[0] if len(w_test) > 0 else None,
            "honest_trades": rh["n_trades"] if rh else 0,
            "honest_mean_ret": rh["mean_return"] if rh else 0,
            "honest_hit": rh["hit_rate"] if rh else 0,
            "filtered_trades": rf["n_trades"] if rf else 0,
            "filtered_mean_ret": rf["mean_return"] if rf else 0,
            "filtered_hit": rf["hit_rate"] if rf else 0,
        })

        i += step
        window_id += 1

    # Aggregate rolling results
    for label, all_trades in [("HONEST (no filters)", all_trades_honest),
                              ("WITH FILTERS", all_trades_filtered)]:
        if not all_trades:
            print(f"\n  {label}: No trades across all windows")
            continue

        tdf = pd.DataFrame(all_trades)
        rets = tdf["return"].values
        sharpe = rets.mean() / rets.std() * np.sqrt(252 * 6) if rets.std() > 0 else 0

        print(f"\n  {label} — Aggregated across {window_id} windows:")
        print(f"    Total trades: {len(rets)}")
        print(f"    Mean return:  {rets.mean() * 100:+.2f}%")
        print(f"    Hit rate:     {(rets > 0).mean():.1%}")
        print(f"    Cumulative:   {(1 + rets).prod() - 1:.1%}")
        print(f"    Sharpe (ann): {sharpe:.2f}")

        # Stability: % of windows with positive mean return
        pos_windows = sum(1 for w in window_results
                          if w.get(f"{label.split()[0].lower()}_mean_ret", 0) > 0)

    # Per-window detail
    print(f"\n  Per-window breakdown (honest):")
    print(f"  {'Win':>4} {'Start':>20} {'Trades':>7} {'Mean Ret':>10} {'Hit Rate':>10}")
    print("  " + "-" * 55)
    for w in window_results:
        start_str = w["start"].strftime("%Y-%m-%d %H:%M") if w["start"] is not None else "N/A"
        print(f"  {w['window']:>4} {start_str:>20} {w['honest_trades']:>7} "
              f"{w['honest_mean_ret'] * 100:>+9.2f}% {w['honest_hit']:>9.1%}")

    # Consistency metric
    positive_windows = sum(1 for w in window_results if w["honest_mean_ret"] > 0)
    print(f"\n  Positive windows: {positive_windows}/{len(window_results)} "
          f"({positive_windows / len(window_results):.0%})")

    if window_results:
        pd.DataFrame(window_results).to_csv(
            OUTPUT_DIR / "projection_rolling_walkforward.csv", index=False)

    if r_honest:
        r_honest["trade_df"].to_csv(
            OUTPUT_DIR / "projection_strategy_trades.csv", index=False)

    return {
        "honest": r_honest,
        "filtered": r_filtered,
        "rolling_windows": len(window_results),
        "rolling_positive_pct": positive_windows / max(len(window_results), 1),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    print("FARTCOIN ALPHA — PROJECTION ENGINE BACKTEST")
    print("=" * 70)
    print(f"Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Load data
    print("\nLoading data...")
    data = load_data()
    signals = compute_all_signals(data)

    ohlcv = data.get("ohlcv")
    btc = data.get("btc")
    funding = data.get("funding")
    lsr = data.get("lsr")
    oi = data.get("oi")

    if ohlcv is None or signals.empty:
        print("ERROR: Missing core data")
        return

    # Run all backtests
    r1 = backtest_probability(signals, ohlcv)
    r2 = backtest_mean_reversion(funding, lsr)
    r3 = backtest_manipulation_cycle(ohlcv, oi, signals)
    r4 = backtest_session_conditional(signals, ohlcv)
    r5 = backtest_btc_lead_lag(ohlcv, btc)
    r6 = backtest_combined_strategy(signals, ohlcv)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
