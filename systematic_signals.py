"""
systematic_signals.py — Rule-based systematic trading signals

Two validated signals (walk-forward backtested, Dec 2024 – May 2026):

  1. POST_SETTLE_BOUNCE   LONG  4h  ~8/mo   Sharpe +4.21  hit 63.6%
     Fire LONG 0–4h after a settlement where fundingRate > p95 (0.000243).
     When funding was elevated, longs paid — pressure releases → bounce.

  2. EXTREME_FADE         SHORT 2h  ~4/mo   Sharpe +11.99 hit 80.0%
     Fire SHORT 0–2h before a settlement where funding > p99 (0.00053).
     Extreme crowding: longs race to exit before paying → price dips into settlement.

These complement the model signals (LGBM dual-horizon + LSTM triple ensemble)
by providing additional setups on the "between-model-signal" days.
Sized at 20% position — smaller than high-conviction model trades.

Key constants (do not tune without re-running systematic_backtest.py):
  P95_THRESHOLD   = 0.000243   raw fundingRate (Bybit decimal, not %)
  P99_THRESHOLD   = 0.000530
  SETTLE_HOURS    = {0, 4, 8, 12, 16, 20}  UTC
  BOUNCE_WINDOW_H = 4          hours AFTER settlement to stay active
  FADE_WINDOW_H   = 2          hours BEFORE settlement to be in position
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import numpy as np


# ── Validated thresholds (from walk-forward backtest) ─────────────────────────
P95_THRESHOLD   = 0.000243   # post-settle bounce gate
P99_THRESHOLD   = 0.000530   # extreme fade gate
SETTLE_HOURS    = {0, 4, 8, 12, 16, 20}   # UTC settlement hours
BOUNCE_WINDOW_H = 4
FADE_WINDOW_H   = 2

# Backtest stats (for display)
_BOUNCE_STATS = {"hit_rate": 0.636, "sharpe": 4.21, "trades_per_month": 8,
                 "avg_ret_pct": 0.68, "hold_h": 4, "size_pct": 20}
_FADE_STATS   = {"hit_rate": 0.800, "sharpe": 11.99, "trades_per_month": 4,
                 "avg_ret_pct": 1.12, "hold_h": 2, "size_pct": 20}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _settlement_times_near(now_utc: datetime) -> tuple[datetime, datetime]:
    """Return (last_settlement, next_settlement) relative to now_utc."""
    h = now_utc.hour
    # Find the last settlement hour ≤ now
    last_h = max((s for s in SETTLE_HOURS if s <= h), default=max(SETTLE_HOURS))
    if last_h > h:
        # Rolled over midnight: last settlement was yesterday
        from datetime import timedelta
        last_dt = now_utc.replace(hour=last_h, minute=0, second=0, microsecond=0) \
                  - timedelta(days=1)
    else:
        last_dt = now_utc.replace(hour=last_h, minute=0, second=0, microsecond=0)

    # Next settlement hour > now
    next_h_candidates = sorted(s for s in SETTLE_HOURS if s > h)
    if next_h_candidates:
        next_h = next_h_candidates[0]
        next_dt = now_utc.replace(hour=next_h, minute=0, second=0, microsecond=0)
    else:
        from datetime import timedelta
        next_h = min(SETTLE_HOURS)
        next_dt = (now_utc + timedelta(days=1)).replace(
            hour=next_h, minute=0, second=0, microsecond=0
        )
    return last_dt, next_dt


def _get_funding_at(funding_df: pd.DataFrame, target_dt: datetime,
                    tolerance_h: int = 2) -> float | None:
    """
    Return the funding rate closest to target_dt within tolerance_h hours.
    Returns None if no data available.
    """
    if funding_df is None or funding_df.empty:
        return None
    idx = funding_df.index
    # Strip tz if needed
    target_naive = target_dt.replace(tzinfo=None) if target_dt.tzinfo else target_dt
    window = funding_df[
        (idx >= target_naive - pd.Timedelta(hours=tolerance_h)) &
        (idx <= target_naive + pd.Timedelta(hours=tolerance_h))
    ]
    if window.empty:
        return None
    col = "fundingRate" if "fundingRate" in window.columns else window.columns[0]
    time_diffs = pd.Series(
        [(t - target_naive).total_seconds().__abs__() for t in window.index],
        index=window.index,
    )
    closest = window.iloc[time_diffs.values.argmin()]
    return float(closest[col])


def _latest_funding(funding_df: pd.DataFrame) -> float | None:
    """Return the most recent fundingRate available."""
    if funding_df is None or funding_df.empty:
        return None
    col = "fundingRate" if "fundingRate" in funding_df.columns else funding_df.columns[0]
    return float(funding_df[col].iloc[-1])


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_settlement_signals(data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate both systematic signals against the current market state.

    Parameters
    ----------
    data : dict
        Standard pipeline `data` dict. Uses:
          data["funding"]    — DataFrame with fundingRate column, DatetimeIndex
          data.get("now_utc") — datetime override for testing (default: utcnow)

    Returns
    -------
    dict with keys:
      "signals"          : list[dict]  — all signal dicts (active or inactive)
      "active_signals"   : list[dict]  — only active signals
      "any_active"       : bool
      "summary"          : str         — one-liner for Telegram / dashboard header
      "last_settlement"  : str         — HH:MM UTC
      "next_settlement"  : str         — HH:MM UTC
      "mins_to_next"     : float
      "last_funding_rate": float | None
    """
    funding_df = data.get("funding")

    # Current time (allow override for testing)
    now_raw = data.get("now_utc") or datetime.now(timezone.utc)
    now_utc = now_raw.replace(tzinfo=None) if now_raw.tzinfo else now_raw

    last_settle, next_settle = _settlement_times_near(now_utc)
    mins_to_next = (next_settle - now_utc).total_seconds() / 60
    mins_since_last = (now_utc - last_settle).total_seconds() / 60

    # Funding rates
    last_settle_rate = _get_funding_at(funding_df, last_settle, tolerance_h=1)
    latest_rate      = _latest_funding(funding_df)

    # ── Signal 1: POST_SETTLE_BOUNCE ──────────────────────────────────────────
    bounce_active = False
    bounce_reason = ""
    if last_settle_rate is not None:
        rate_ok   = last_settle_rate > P95_THRESHOLD
        timing_ok = 0 <= mins_since_last <= BOUNCE_WINDOW_H * 60
        if rate_ok and timing_ok:
            bounce_active = True
            bounce_reason = (
                f"Last settlement funding {last_settle_rate:.6f} > p95 threshold "
                f"({P95_THRESHOLD}); {mins_since_last:.0f}min since settlement "
                f"(window: 0–{BOUNCE_WINDOW_H*60:.0f}min)"
            )
        else:
            reasons = []
            if not rate_ok:
                reasons.append(
                    f"funding {last_settle_rate:.6f} ≤ p95 {P95_THRESHOLD}"
                )
            if not timing_ok:
                reasons.append(
                    f"{mins_since_last:.0f}min since settlement (window closes at {BOUNCE_WINDOW_H*60:.0f}min)"
                )
            bounce_reason = "Inactive: " + "; ".join(reasons)
    else:
        bounce_reason = "Inactive: no funding data for last settlement"

    bounce_signal: dict = {
        "id":          "POST_SETTLE_BOUNCE",
        "label":       "Post-Settlement Bounce",
        "active":      bounce_active,
        "direction":   "LONG",
        "trigger":     f"Funding >p95 ({P95_THRESHOLD}) at last settlement + within {BOUNCE_WINDOW_H}h",
        "hold_h":      _BOUNCE_STATS["hold_h"],
        "size_pct":    _BOUNCE_STATS["size_pct"],
        "hit_rate":    _BOUNCE_STATS["hit_rate"],
        "sharpe":      _BOUNCE_STATS["sharpe"],
        "trades_per_month": _BOUNCE_STATS["trades_per_month"],
        "avg_ret_pct": _BOUNCE_STATS["avg_ret_pct"],
        "last_settle_rate": last_settle_rate,
        "mins_since_settlement": round(mins_since_last, 1),
        "reason":      bounce_reason,
        "description": (
            f"🟢 POST-SETTLEMENT BOUNCE ACTIVE — LONG {_BOUNCE_STATS['size_pct']}% | "
            f"Hold {_BOUNCE_STATS['hold_h']}h | "
            f"Funding {last_settle_rate:.6f} > p95 | "
            f"Hist: {_BOUNCE_STATS['hit_rate']:.0%} hit, Sharpe {_BOUNCE_STATS['sharpe']:.2f}"
        ) if bounce_active else (
            f"Post-Settlement Bounce: inactive ({bounce_reason})"
        ),
    }

    # ── Signal 2: EXTREME_FADE ────────────────────────────────────────────────
    # Use latest funding as the best estimate of what the next settlement rate will be
    fade_active = False
    fade_reason = ""
    fade_rate   = latest_rate  # proxy for upcoming settlement rate

    if fade_rate is not None:
        rate_ok   = fade_rate > P99_THRESHOLD
        timing_ok = 0 <= mins_to_next <= FADE_WINDOW_H * 60
        if rate_ok and timing_ok:
            fade_active = True
            fade_reason = (
                f"Current funding {fade_rate:.6f} > p99 threshold "
                f"({P99_THRESHOLD}); {mins_to_next:.0f}min to settlement "
                f"(window: 0–{FADE_WINDOW_H*60:.0f}min)"
            )
        else:
            reasons = []
            if not rate_ok:
                reasons.append(
                    f"funding {fade_rate:.6f} ≤ p99 {P99_THRESHOLD}"
                )
            if not timing_ok:
                reasons.append(
                    f"{mins_to_next:.0f}min to next settlement (window opens at {FADE_WINDOW_H*60:.0f}min)"
                )
            fade_reason = "Inactive: " + "; ".join(reasons)
    else:
        fade_reason = "Inactive: no current funding data"

    fade_signal: dict = {
        "id":          "EXTREME_FADE",
        "label":       "Extreme Settlement Fade",
        "active":      fade_active,
        "direction":   "SHORT",
        "trigger":     f"Funding >p99 ({P99_THRESHOLD}) within {FADE_WINDOW_H}h of settlement",
        "hold_h":      _FADE_STATS["hold_h"],
        "size_pct":    _FADE_STATS["size_pct"],
        "hit_rate":    _FADE_STATS["hit_rate"],
        "sharpe":      _FADE_STATS["sharpe"],
        "trades_per_month": _FADE_STATS["trades_per_month"],
        "avg_ret_pct": _FADE_STATS["avg_ret_pct"],
        "current_funding_rate": fade_rate,
        "mins_to_settlement": round(mins_to_next, 1),
        "reason":      fade_reason,
        "description": (
            f"🔴 EXTREME FADE ACTIVE — SHORT {_FADE_STATS['size_pct']}% | "
            f"Hold {_FADE_STATS['hold_h']}h | "
            f"Funding {fade_rate:.6f} > p99 | "
            f"Hist: {_FADE_STATS['hit_rate']:.0%} hit, Sharpe {_FADE_STATS['sharpe']:.2f}"
        ) if fade_active else (
            f"Extreme Fade: inactive ({fade_reason})"
        ),
    }

    # ── Aggregate ─────────────────────────────────────────────────────────────
    signals = [bounce_signal, fade_signal]
    active  = [s for s in signals if s["active"]]

    if active:
        labels = " + ".join(s["label"] for s in active)
        dirs   = " / ".join(s["direction"] for s in active)
        summary = f"⚡ DESK SETUP ACTIVE: {labels} | {dirs}"
    else:
        summary = (
            f"No systematic setups active. "
            f"Next settlement: {next_settle.strftime('%H:%M')} UTC "
            f"({mins_to_next:.0f}min). "
            f"Current funding: {latest_rate:.6f}" if latest_rate else
            f"Next settlement: {next_settle.strftime('%H:%M')} UTC"
        )

    return {
        "signals":           signals,
        "active_signals":    active,
        "any_active":        bool(active),
        "summary":           summary,
        "last_settlement":   last_settle.strftime("%H:%M UTC"),
        "next_settlement":   next_settle.strftime("%H:%M UTC"),
        "mins_to_next":      round(mins_to_next, 1),
        "last_funding_rate": last_settle_rate,
        "current_funding_rate": latest_rate,
    }
