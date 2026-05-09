# Alpha Trade Desk — Dashboard Update & Trading Guide

**Version:** May 2026 · Dual-Horizon Model  
**For:** Trading team

---

## What Changed and Why

The dashboard has been rebuilt around one insight: **two independent models confirming the same trade is fundamentally more reliable than one model with high confidence.** The previous version showed a single probability number. This version shows you the full picture — whether both the 4-hour and 8-hour models agree, how strong each conviction is, and what the risk geometry looks like before you decide.

Additionally, a field-tested finding from our backtest history is now surfaced at the top of every session: the model's historical hit rate at the current tier. You no longer need to remember that TRADE-tier signals have historically won 64% of the time — the dashboard tells you.

---

## Model Performance Benchmarks

| Tier | Min Score | Backtest Hit Rate | Sharpe | Avg Trades/Month |
|------|-----------|------------------|--------|-----------------|
| FULL SEND | 78+ | ~75–80% | 4.0+ | ~8 |
| HIGH CONVICTION | 65–77 | ~70–75% | 4.0+ | ~15 |
| TRADE | 55–64 | ~64% | 3.2 | ~25 |
| WATCH | — | — | — | Monitor only |
| PASS | <45 | — | — | No action |
| BLOCKED | Gate active | — | — | Hard stop |

**Baseline (random):** ~42% hit rate. The model must clear 0.45% gain per trade to beat carry cost.

---

## The Four Tabs at a Glance

| Tab | What's Here | When to Use |
|-----|-------------|-------------|
| **⚡ Edge** | Brief card, model agreement, signal breakdown, price chart | Every session — primary decision tab |
| **⏰ Timing** | Session windows, kill zones, funding schedule | When timing a specific entry |
| **🌐 Context** | BTC regime, exchange flows, OI/LSR, cross-exchange funding | When reading market structure |
| **📋 Reference** | Rules playbook, position sizing table, hold windows | When in doubt, check here first |

---

## Reading the Brief Card (Top of Edge Tab)

The brief card is the first and most important thing on screen. Here's what each element means:

```
┌──────────────────────────────────────────────────────────────────┐
│  🟢 LONG — HIGH CONVICTION                            80% size   │
│  Signal above threshold + both models agree.       Score 72/100  │
│  Entry $0.2480 / Stop $0.2390 / Target $0.2620      Kelly 18%   │
│                                                                    │
│  4h: ████████░░ 67% ✓  |  BOTH AGREE ✓✓  |  8h: █████████░ 73% ✓ │
│                                                                    │
│  HMM: ACCUM  Session: NYC  FART: $0.2480  BTC: $81k  Funding: 0.05│
└──────────────────────────────────────────────────────────────────┘
📊 Recent performance at this tier: Last 20 · HIGH CONVICTION → 74% hit
```

### Size % (top right)
The recommended position size as a percentage of your tradeable capital. This is already Kelly-adjusted and volatility-scaled — it accounts for current ATR relative to the historical average. Do not manually override this upward. If it says 40%, that is the mathematically optimal size given current edge and vol.

### Score and Tier
The score (0–100) is the raw model output. The tier is the human-readable label derived from it:
- **Score ≥ 78** → FULL SEND: highest conviction, full Kelly sizing
- **Score 65–77** → HIGH CONVICTION: strong signal, 80% of Kelly ceiling
- **Score 55–64** → TRADE: signal present, 60% of Kelly ceiling
- **Score 45–54** → WATCH: insufficient edge, no position
- **Score < 45** → PASS: model sees no edge
- **BLOCKED** → hard gate active (see below)

### Kelly %
The fraction of capital the Kelly criterion recommends given the model's edge estimate. The `Size %` shown is Kelly × vol scalar × tier ceiling. In practice, many traders use half-Kelly (i.e. halve the Size % shown) for additional risk management — this is valid and recommended for new positions.

### Model Agreement Row (4h / 8h bars)
This is new and critical. The two bars show the independent probability estimate from each model:

| What you see | What it means | Action |
|---|---|---|
| Both bars green + **BOTH AGREE ✓✓** | Strong consensus — not a pump-and-dump | Trade the full recommended size |
| Both green but one is 50–55% | Agreement at minimum threshold | Valid trade, consider half-size |
| 4h green, 8h red + **DIVERGE** | 4h signal bullish but 8h skeptical — reversal risk | Do not trade |
| Both red + **BOTH SKEPTICAL** | No edge in either horizon | Stay flat |
| Bars grayed + **GATED** | HMM or session gate is active | Hard stop regardless of bars |

The 4h model predicts whether the trade will be profitable within 4 hours. The 8h model predicts whether the move is sustained through 8 hours. When the 8h model is skeptical, it means the model has learned that this pattern historically reverses — classic pump-and-dump structure. **Never trade when the models diverge.**

### Entry / Stop / Target Levels
Always visible regardless of tier:
- **TRADE tier and above:** these are your live levels. Set your alerts here before the candle closes.
- **WATCH tier:** labeled "Pre-plan if signal fires." The levels are already calculated — use this time to pre-size and pre-set your alerts so you can act immediately if the signal upgrades.
- **PASS / BLOCKED:** shows nearest support and resistance as informational context only.

### Hit Rate Widget
The line below the brief card shows the model's live hit rate for the current tier over the last 20 resolved trades. Use this to calibrate your conviction:
- If the live hit rate is tracking close to the backtest benchmark (e.g. 70–75% for HIGH CONVICTION), the model is performing as expected.
- If the live hit rate has drifted significantly below the backtest (e.g. 55% for a tier that should be 70%), consider halving your size until performance recovers.

---

## The Hard Gates — When to Never Trade

These override everything. When any gate is active, the brief card shows **BLOCKED** regardless of score.

### HAKAI Regime (HMM gate)
The Hidden Markov Model has classified the current market as a **distribution phase** — smart money is exiting, and new longs are being absorbed by the exit. Key signs: OI spike + elevated realized volatility + price breakout that is not being confirmed by LSR. This is the single most common pattern preceding a sharp dump.

**Rule: Zero new positions in HAKAI. No exceptions.**

The gate clears when the HMM transitions back to ACCUMULATION or STEADY_STATE. The dashboard shows hours since the last HAKAI exit — treat the first 3–4 hours after exit as elevated risk.

### Session Gate (20:00–00:00 UTC)
The model has no statistical edge in this window. Liquidity is fragmenting across Asian pre-market and late US trading — the signal-to-noise ratio collapses and spreads widen. The backtest shows near-random outcomes in this window.

**Rule: No new positions between 20:00 and 00:00 UTC.**

### Diverging Models (WATCH override)
If the two models disagree — even if the score is above TRADE threshold — the tier is forced to WATCH. This is intentional: the 8h model's skepticism is a learned pattern, not a coincidence.

---

## Decision Framework: Step by Step

Before entering any trade, run through this checklist in order. Stop at the first "No":

```
1. Is the brief card BLOCKED?          → If yes: stop. No trade.
2. Is HMM in HAKAI?                    → If yes: stop. No trade.
3. Is it 20:00–00:00 UTC?              → If yes: stop. No trade.
4. Do BOTH models agree (✓✓)?          → If no: stop. No trade.
5. Is the tier TRADE or higher?        → If no: watch and set alert.
6. Is funding extreme (> 0.30)?        → If yes: reduce size by 50%.
7. Is Risk Score ≥ 4?                  → If yes: reduce size by 50%.
8. Is BTC in a downtrend regime?       → If yes: reduce size by 25%.
──────────────────────────────────────
   Adjusted size = Size% × multipliers above
   Enter with the pre-calculated Entry level.
   Set stop at the pre-calculated Stop level. Non-negotiable.
   Target: pre-calculated Target level or 4h max hold, whichever comes first.
```

---

## Position Sizing Rules

| Condition | Size Adjustment |
|-----------|----------------|
| FULL SEND, both agree strongly (both >65%) | 100% of shown size |
| HIGH CONVICTION, both agree | 80% of shown size |
| TRADE tier | 60% of shown size |
| Funding > 0.20 | × 0.75 |
| Funding > 0.30 | × 0.50 |
| Risk Score 3–4 | × 0.75 |
| Risk Score ≥ 4 | × 0.50 |
| BTC in downtrend | × 0.75 |
| Weekend session | × 0.50 (liquidity 30–50% lower) |
| Live hit rate tracking 10pp+ below backtest | × 0.50 until recovery |

**Never exceed 100% of the shown size.** The Kelly formula already encodes the model's edge estimate. Going above it is mathematically expected to reduce long-run returns.

---

## Common Mistakes to Avoid

**Trading WATCH tier signals.**  
WATCH means the model sees a potential setup but lacks sufficient edge to justify the carry cost. The correct response is to set a price alert at the Entry level and wait for the next refresh. If the signal upgrades to TRADE, you trade it then — not before.

**Ignoring the 8h bar when it's low.**  
A strong 4h score with a weak 8h probability (e.g. 4h: 72%, 8h: 38%) is the model's way of saying "this looks like a pump that will reverse within 8 hours." This is one of the model's most valuable outputs. The old dashboard didn't show this at all. Take it seriously.

**Manually adjusting the stop.**  
The stop level is derived from the support/resistance model, not a fixed percentage. Moving your stop lower to "give the trade room" defeats the purpose — the S/R model has already accounted for normal retracement distance.

**Staying in a trade when HMM flips to HAKAI.**  
If you're in a long position and the regime flips to HAKAI mid-trade, treat this as an exit signal, not a hold. Distribution phases rarely resolve in the long direction.

**Trading during late session (20:00–00:00 UTC) because "the setup looks good."**  
The session gate exists because the model has zero historical edge in this window across 2,000+ backtested bars. A setup that looks good visually is not a reason to override a statistically validated gate.

---

## Using the Timing Tab

The Timing tab shows session windows and kill zones. The most important kill zones to know:

| Window | Risk Level | Notes |
|--------|-----------|-------|
| 00:00–02:00 UTC | High | Asia open — often sees manipulative wicks |
| 08:00–09:00 UTC | Medium | Europe open — directional bias sets for the session |
| 13:30–14:30 UTC | High | US equity open — cross-asset volatility spike |
| 20:00–00:00 UTC | Gated | Model has no edge — hard stop |

**Best entry windows:** 02:00–07:00 UTC (Asia mid-session quiet), 10:00–13:00 UTC (London mid-session), 15:00–19:00 UTC (NY mid-session).

---

## Using the Context Tab

The Context tab shows BTC regime, exchange flows, and cross-exchange funding. Use it to answer one question: **is the broader market environment supportive?**

- **BTC in uptrend + FART accumulation regime** = highest probability environment
- **BTC flat + FART steady-state** = reduced size, wait for clearer signal
- **BTC in downtrend** = cut size by 25% on all longs regardless of score
- **Cross-exchange funding diverging** (Bybit vs Binance spread > 0.05) = elevated manipulation risk, reduce size

---

## Data Freshness

The dashboard shows a freshness badge in the top-right corner of the brief card. It reflects how recently the underlying data was updated:

- **"Data 45m ago"** — fresh, no concern
- **"Data 2.9h ago"** — normal for a daily pipeline, all signals valid
- **"⚠ Data 4.5h old"** — data is stale, live price is still current but historical signals may be lagging

The live FART and BTC prices shown in the pills are always fetched in real-time regardless of data age. The staleness warning relates to OHLCV history, OI, and LSR — the inputs to the signal engine.

---

## Reference: Signal Glossary

| Term | Definition |
|------|-----------|
| **Composite** | Weighted average of all sub-signals. Positive = bullish lean, negative = bearish. Used as input to the model, not a standalone signal. |
| **HMM Regime** | Hidden Markov Model classification: ACCUMULATION (favorable), STEADY_STATE (neutral), HAKAI (hard gate). Updated every bar. |
| **HAKAI** | Japanese for "destruction." Distribution phase — smart money exiting. No new longs. |
| **VPIN** | Volume-synchronized probability of informed trading. Elevated VPIN = informed flow in the market (can be either direction). |
| **LSR** | Long/Short Ratio. Ratio of long to total open contracts. >0.55 = crowded long. |
| **OI** | Open Interest. Rising OI + rising price = new money entering long. Rising OI + falling price = new short pressure. |
| **Funding** | Perpetual futures funding rate. Positive = longs paying shorts. High positive funding = crowded long, contrarian pressure. |
| **BSR** | Buy/Sell Ratio from taker flow. >0.55 = buyers taking liquidity (bullish). |
| **Kelly fraction** | Mathematically optimal fraction of capital to risk given estimated edge. The displayed Size% is already Kelly × vol scalar × tier ceiling. |
| **Carry cost** | 0.45% per 4-hour trade — the minimum return needed to be profitable after funding and fees. The model only trades signals that historically exceed this. |
| **4h / 8h targets** | The model is trained to predict whether the 4h forward return and 8h forward return each exceed 0.45% / 0.90% respectively. Both must be predicted positive to trade. |

---

*Last updated: May 2026. Model: Dual-horizon LightGBM ensemble (70.9% hit, 4.09 Sharpe, 289 trades in walk-forward validation).*
