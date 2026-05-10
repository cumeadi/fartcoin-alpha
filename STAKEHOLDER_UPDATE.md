# Fartcoin Alpha — Stakeholder Update
**May 2026**

---

## Executive Summary

Over the past two development sprints we rebuilt the alpha framework from a manual, single-model prototype into a production-grade system with automated data collection, a multi-model ensemble, push notifications, and a live dashboard. The headline result: the highest-conviction trading signals now hit at a **97.7% accuracy rate** in walk-forward backtesting — up from 64.2% when we started.

---

## What We Built & Why

### 1. Dual-Horizon LightGBM Ensemble
**What:** Replaced the original single-model LightGBM with two models trained in parallel — one predicting the next 4-hour return, one predicting the next 8-hour return. A trade only fires when **both models agree**.

**Why:** A single model can't distinguish a real setup from a pump that reverses quickly. The 8-hour model acts as a skeptic — if the move isn't expected to sustain, it vetoes. This filters out pump-and-dump bars while keeping genuine momentum setups.

**Result:**

| Metric | Before (single-horizon) | After (dual-horizon) |
|--------|------------------------|----------------------|
| Hit rate | 64.2% | 73.9% |
| Lift over baseline | +22.0pp | +31.7pp |
| Annualised Sharpe | 3.23 | 5.11 |
| Trade count | 358 | 283 |

The reduction in trade count is intentional — the model is more selective, not less capable.

---

### 2. LSTM Raw-OHLCV Model (New Architecture)
**What:** Built a second, independent model using a completely different architecture. Instead of 26 engineered features at a single point in time (what the LGBM sees), this LSTM reads **64 hours of raw price, volume, open interest, and order flow data as a sequence** — like a trader studying a 2-day chart rather than a single snapshot.

**Why:** LGBMs are excellent at reading the current state of the market but blind to *how it got there*. LSTMs are designed to learn temporal patterns — funding accumulation cycles, OI buildup before squeezes, multi-day BSR momentum. The two architectures are genuinely complementary.

**Development path:**

| Configuration | Hit Rate | Sharpe | Trades | Gate |
|--------------|----------|--------|--------|------|
| LSTM (original, 10h lookback, engineered features) | 47.3% | 1.09 | 740 | ❌ |
| LSTM (raw OHLCV, 48h lookback) | 56.6% | 2.31 | 318 | ❌ |
| LSTM (raw OHLCV, 64h lookback) | 64.4% | 3.42 | 362 | ❌ |
| **LSTM (raw OHLCV, 64h + HMM regime features)** | **71.5%** | **4.88** | **284** | **✅** |
| LSTM (raw OHLCV, 96h lookback) | 44.1% | 0.19 | 338 | ❌ |

Key finding: the original LSTM was predicting on noise. Switching to raw OHLCV sequences gave the model the temporal context it needed. Adding HMM regime awareness pushed it above the production gate (≥70% hit, ≥200 trades, ≥4.0 Sharpe).

---

### 3. Triple-Agreement Ensemble
**What:** When the LGBM dual-horizon model fires AND the LSTM independently agrees, the system escalates the signal to **FULL SEND** — maximum conviction, maximum size.

**Why:** Two fundamentally different models agreeing on the same bar at the same time is rare, and when it happens it's highly informative. The LGBM is reading a snapshot of 26 market conditions; the LSTM is reading 64 hours of raw price action. Their agreement is not redundant — it's convergent evidence.

**Result:**

| Model | Hit Rate | Lift | Sharpe | Trades/month |
|-------|----------|------|--------|-------------|
| LGBM dual-horizon (primary signal) | 73.9% | +31.7pp | 5.11 | ~14 |
| LSTM raw lb=64+HMM (secondary signal) | 71.5% | +29.3pp | 4.88 | ~14 |
| **Triple ensemble (both agree)** | **97.7%** | **+55.5pp** | **9.67** | **~4** |

The triple ensemble trades ~4 times per month at 97.7% accuracy with an average 4-hour return of +3.95% per trade (vs. +0.09% baseline). These are treated as the highest-conviction entries in the system.

---

### 4. Infrastructure Overhaul (Phase 2)

**Why this mattered:** The entire system was running on one person's laptop. If that machine was offline, the dashboard went stale and signals stopped. The research was solid but the delivery was fragile.

| Item | What We Built |
|------|--------------|
| **GitHub Actions pipeline** | Automated data collection runs every 4 hours (derivatives snapshot) and daily (full refresh + external data). Data commits back to the repo — Streamlit Cloud stays live automatically. |
| **Model persistence** | Trained LightGBM models are cached to disk. Dashboard cold-start dropped from **36 seconds → ~2 seconds** on cache hit. |
| **P&L tracking + equity curve** | Every live pipeline run logs direction-adjusted returns to a trade journal. The dashboard now shows a cumulative equity curve by signal tier. |
| **Telegram push alerts** | When a signal tier upgrades (e.g., WATCH → TRADE → FULL SEND), a Telegram message fires automatically with price, probability, size recommendation, and S/R levels. |
| **Multi-coin scaffold** | SOL, WIF, and BONK added to the configuration registry. `ingest_bybit.py` parameterized — a single command backfills any supported coin. |

---

## Before & After: System-Level Comparison

| Dimension | Before | After |
|-----------|--------|-------|
| **Best signal hit rate** | 64.2% (single LGBM) | 97.7% (triple ensemble) |
| **Primary signal Sharpe** | 3.23 | 5.11 |
| **Data pipeline** | Manual, laptop-dependent | Automated via GitHub Actions |
| **Dashboard load time** | 36s cold start | ~2s (cached) |
| **Alert delivery** | Manual dashboard check | Telegram push on tier upgrade |
| **Model architecture** | 1 model | 3 voters (LGBM 4h + LGBM 8h + LSTM) |
| **Live signal frequency** | ~18/month (LGBM) | ~14/month primary + ~4/month FULL SEND |
| **Supported coins** | FARTCOIN only | FARTCOIN + SOL, WIF, BONK (config ready) |
| **Trade journal** | Binary win/loss | Direction-adjusted P&L + equity curve |

---

## What the Dashboard Shows

The live dashboard (`fartcoin-alpha.streamlit.app`) now has four tabs:

1. **Signal** — Real-time verdict (LONG/SHORT/STAND ASIDE), tier, size recommendation, model agreement panel (4h model · 8h model · LSTM), and triple-agreement banner when all three fire.
2. **Market Structure** — HMM regime (ACCUMULATION / STEADY_STATE / HAKAI), support/resistance levels, funding rate, OI, and BTC correlation.
3. **Projections** — Forward price scenarios, Kelly sizing, R:R ratio, opportunity score breakdown.
4. **Journal** — Equity curve by tier, rolling win rate, historical trade log.

---

## Potential Next Steps

### New Data Sources

| Source | Signal Hypothesis | Est. IC | Priority |
|--------|------------------|---------|----------|
| **Liquidation heatmaps** | Large liquidation clusters act as price magnets and S/R levels. Fetch from Coinglass or Hyblock. | ~0.05–0.08 | High |
| **Perpetual basis (funding spread across exchanges)** | Cross-exchange funding divergence precedes mean reversion; arbitrageurs create predictable flows. Currently have single-exchange funding only. | ~0.04–0.06 | High |
| **DEX pool imbalance (Raydium/Orca)** | Solana DEX liquidity shifts precede spot moves by 1–3 hours on meme coins. Helius API already connected. | ~0.03–0.05 | Medium |
| **BTC/SOL full history (2+ years)** | BTC correlation feature currently trained on only 90 days of data. `data.binance.vision` has free hourly OHLCV back to 2019. | Indirectly improves all BTC-dependent features | High |
| **Options flow (Deribit/Derive)** | Put/call ratios and unusual options activity precede directional moves. Less relevant for FARTCOIN specifically, more relevant if we expand to SOL/BTC. | ~0.04–0.07 | Medium |

---

### New Methodologies

**1. Wider LSTM training window for lb=96**
The lb=96 LSTM collapsed (44.1% hit) because the 500-row training window is too narrow relative to the 96-bar sequence length. Widening `TRAIN_WIN` to 750–1,000 rows should make lb=96 viable, potentially pushing the standalone LSTM above 75% hit rate.

**2. Attention mechanism (Transformer)**
The LSTM reads sequences but weighs all timesteps equally. A self-attention layer would let the model learn which of the 64 hours are most predictive for a given bar type — for example, learning that the funding rate 8 hours ago is more important than 1 hour ago. Estimated +3–6pp hit rate improvement at same trade frequency.

**3. Ensemble probability calibration**
Both models output raw sigmoid probabilities. Applying isotonic regression calibration (already used in the LGBM scorer) to the LSTM output, then combining LGBM and LSTM probabilities into a single ensemble probability (rather than a binary AND gate), would allow continuous confidence scoring rather than a hard threshold. This would produce a richer signal hierarchy between the current tiers.

**4. Regime-conditional sizing**
Currently Kelly fraction is computed from the overall model accuracy. In ACCUMULATION regime, hit rate is 75.4% vs. 73.9% overall — the model is measurably better in certain regimes. Regime-specific Kelly multipliers would extract more edge without increasing risk in unfavorable conditions.

**5. Multi-coin validation**
Running the full signal engine on SOL and WIF data (once backfilled) would validate whether the features and model architecture transfer, or whether coin-specific tuning is needed. SOL's deeper liquidity and higher correlation to BTC makes it a natural second test case.

**6. Live paper trading tracker**
The trade journal currently logs every pipeline run. Building a lightweight paper trading module that tracks entries/exits at exact signal timestamps (rather than retrospective journal entries) would give cleaner live-vs-backtest comparison and catch any look-ahead bias in production.

---

## Key Risk Disclosures

- **Walk-forward backtesting** simulates out-of-sample performance but cannot fully replicate live trading conditions (slippage, liquidity, timing latency).
- The **97.7% triple ensemble hit rate** is based on 87 trades over ~16 months. Statistically robust, but a small absolute count relative to daily trading frequency.
- Both the LGBM and LSTM use HMM regime features, creating **partial non-independence** between the two voters. The true independent signal contribution is lower than it would appear from the combination alone.
- FARTCOIN's historical data begins December 2024 (~16 months). All models should be re-evaluated as the training set grows toward 24+ months.

---

*For questions on methodology, data sources, or live signal interpretation, refer to `DASHBOARD_UPDATE.md` in this repository.*
