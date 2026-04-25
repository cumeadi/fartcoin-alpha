"""
Support & Resistance Engine

Three complementary methods — majority-vote levels are strongest:

  1. Volume Profile (HVN)  — price zones where most volume traded.
                             High-volume nodes = market spent time here = institutional memory.
  2. Swing Highs/Lows      — price levels where reversals occurred.
                             Multi-touch swings = market repeatedly respected the level.
  3. Psychological / Round — $0.20, $0.15, $0.25 etc. Magnet effect from retail orders.

Each candidate level is scored 0–1 on four dimensions:
  volume_score  — normalised volume at this price zone
  touch_count   — how many times price visited (capped at 8 for normalisation)
  recency       — exponential decay: levels touched recently score higher
  bounce_rate   — fraction of visits that resulted in a bounce (vs break)

Final strength = weighted sum of these four. Levels within MERGE_PCT are clustered.

Public API
----------
  compute_sr_levels(data, lookback=600, n_levels=8)
      → dict  (see docstring below for schema)

  get_sr_features(data)
      → dict  with scalar features for trade_scorer.py
              dist_to_support_pct, dist_to_resistance_pct, sr_risk_reward,
              nearest_support_strength, nearest_resistance_strength
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from scipy.signal import argrelmax, argrelmin, find_peaks
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ── Tuning constants ──────────────────────────────────────────────────────────
N_BINS       = 200      # price histogram resolution
VOL_LOOKBACK = 600      # hours of OHLCV used for volume profile  (~25 days)
SWING_ORDER  = 8        # local max/min window (8h each side)
MERGE_PCT    = 0.008    # levels within 0.8% are merged into one zone
TOUCH_TOL    = 0.005    # 0.5% — price within this of a level = "touch"
BOUNCE_BACK  = 3        # candles to check after a touch for a bounce
BOUNCE_MOVE  = 0.003    # 0.3% move away = counted as a bounce

# Score weights (must sum to 1.0)
W_VOLUME   = 0.35
W_TOUCHES  = 0.30
W_RECENCY  = 0.20
W_BOUNCE   = 0.15

# Strength labels
def _strength_label(s):
    if s >= 0.75:  return "STRONG"
    if s >= 0.50:  return "MODERATE"
    if s >= 0.30:  return "WEAK"
    return "MINOR"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Volume profile peaks
# ─────────────────────────────────────────────────────────────────────────────

def _volume_profile_levels(prices, volumes, current_price, lookback=VOL_LOOKBACK):
    """
    Return list of (price, normalised_vol_score) tuples at HVN peaks.
    """
    if not _SCIPY_OK or len(prices) < 48:
        return []

    n   = min(lookback, len(prices), len(volumes))
    px  = prices[-n:]
    vol = volumes[-n:]

    lo  = px.min() * 0.997
    hi  = px.max() * 1.003
    bins = np.linspace(lo, hi, N_BINS + 1)
    profile = np.zeros(N_BINS)

    for p, v in zip(px, vol):
        idx = int((p - lo) / (hi - lo) * N_BINS)
        idx = max(0, min(idx, N_BINS - 1))
        profile[idx] += v

    # Smooth lightly (3-bin boxcar) to avoid micro-peaks
    profile_s = np.convolve(profile, np.ones(3) / 3, mode="same")

    peaks, _ = find_peaks(
        profile_s,
        height=np.percentile(profile_s[profile_s > 0], 55),
        distance=2,
        prominence=np.percentile(profile_s[profile_s > 0], 30),
    )

    bin_mid = (bins[:-1] + bins[1:]) / 2
    max_vol = profile_s[peaks].max() if len(peaks) > 0 else 1.0

    results = []
    for i in peaks:
        vol_score = float(profile_s[i] / (max_vol + 1e-9))
        results.append((float(bin_mid[i]), vol_score))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Swing highs / lows
# ─────────────────────────────────────────────────────────────────────────────

def _swing_levels(prices, timestamps):
    """
    Return list of (price, timestamp_idx, type) for all local extrema.
    type = 'high' | 'low'
    """
    if not _SCIPY_OK or len(prices) < SWING_ORDER * 2 + 1:
        return []

    highs = argrelmax(prices, order=SWING_ORDER)[0]
    lows  = argrelmin(prices, order=SWING_ORDER)[0]

    results = []
    for i in highs:
        results.append((float(prices[i]), i, "high"))
    for i in lows:
        results.append((float(prices[i]), i, "low"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Psychological / round number levels
# ─────────────────────────────────────────────────────────────────────────────

def _round_levels(current_price, price_min, price_max):
    """
    Return round-number levels in the visible price range.
    For sub-$1 assets, use 0.01 and 0.005 steps.
    """
    if current_price < 0.10:
        steps = [0.005, 0.01]
    elif current_price < 1.0:
        steps = [0.01, 0.05]
    elif current_price < 10:
        steps = [0.5, 1.0]
    else:
        steps = [5.0, 10.0]

    levels = []
    for step in steps:
        lo_mult = int(price_min / step)
        hi_mult = int(price_max / step) + 2
        for mult in range(lo_mult, hi_mult):
            lvl = mult * step
            if price_min * 0.95 <= lvl <= price_max * 1.05:
                levels.append(round(lvl, 6))

    return list(set(levels))


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Touch count, recency, bounce rate
# ─────────────────────────────────────────────────────────────────────────────

def _score_level(price_level, prices, timestamps, current_idx):
    """
    For a candidate level, compute:
      touches, last_touch_h, recency_score, bounce_rate
    """
    n         = len(prices)
    touches   = []
    bounces   = 0
    breaks    = 0

    for i in range(1, n - BOUNCE_BACK):
        if abs(prices[i] - price_level) / (price_level + 1e-9) <= TOUCH_TOL:
            hours_ago = current_idx - i
            touches.append(hours_ago)

            # Determine if this touch was a bounce or break
            future_max_move = max(
                abs(prices[i + j] - prices[i]) / (prices[i] + 1e-9)
                for j in range(1, BOUNCE_BACK + 1)
            )
            if future_max_move > BOUNCE_MOVE:
                # Check direction: bounce = price moved away from level
                avg_future = np.mean(prices[i + 1: i + BOUNCE_BACK + 1])
                is_above   = avg_future > price_level
                is_support = price_level < np.mean(prices[max(0, i-10): i + 1])
                if (is_support and is_above) or (not is_support and not is_above):
                    bounces += 1
                else:
                    breaks += 1
            # else: small move, inconclusive — don't count

    n_touches = len(touches)
    if n_touches == 0:
        return {"touches": 0, "last_touch_h": 9999, "recency_score": 0.0, "bounce_rate": 0.5}

    last_touch_h = min(touches)   # most recent

    # Recency: exponential decay, half-life ~168h (7 days)
    recency_score = float(np.exp(-last_touch_h / 168.0))

    # Bounce rate
    denom = bounces + breaks
    bounce_rate = bounces / denom if denom > 0 else 0.5

    return {
        "touches":      n_touches,
        "last_touch_h": last_touch_h,
        "recency_score": recency_score,
        "bounce_rate":  bounce_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Merge nearby levels
# ─────────────────────────────────────────────────────────────────────────────

def _merge_levels(candidates, merge_pct=MERGE_PCT):
    """
    Merge candidate levels within merge_pct of each other.
    Each candidate: dict with 'price', 'vol_score', 'is_swing', 'is_round'.
    Returns merged list sorted by price.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x["price"])
    merged = []
    current_group = [candidates[0]]

    for c in candidates[1:]:
        ref = current_group[-1]["price"]
        if abs(c["price"] - ref) / (ref + 1e-9) <= merge_pct:
            current_group.append(c)
        else:
            merged.append(current_group)
            current_group = [c]
    merged.append(current_group)

    result = []
    for group in merged:
        # Weighted average price (by vol_score)
        prices_g = [g["price"] for g in group]
        weights  = [g.get("vol_score", 0.1) + 0.1 for g in group]
        avg_price = float(np.average(prices_g, weights=weights))
        max_vol   = max(g.get("vol_score", 0.0) for g in group)
        methods   = []
        if any(g.get("is_vol")   for g in group): methods.append("volume_node")
        if any(g.get("is_swing") for g in group): methods.append("swing")
        if any(g.get("is_round") for g in group): methods.append("round_number")

        result.append({
            "price":     avg_price,
            "vol_score": max_vol,
            "methods":   methods,
            "n_sources": len(group),  # how many methods agree → stronger
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def compute_sr_levels(data, lookback=VOL_LOOKBACK, n_levels=8):
    """
    Compute support & resistance levels from OHLCV data.

    Args:
        data:     dict from signal_engine.load_data()
        lookback: hours of history to use for volume profile
        n_levels: max levels to return on each side of current price

    Returns dict:
        {
          "levels": [
            {
              "price": float,
              "type": "support" | "resistance",
              "strength": float (0-1),
              "strength_label": "STRONG" | "MODERATE" | "WEAK" | "MINOR",
              "touches": int,
              "bounce_rate": float,
              "last_touch_h": int,
              "methods": list[str],
              "distance_pct": float,   # from current price
            }, ...
          ],
          "nearest_support":    {"price", "distance_pct", "strength", "strength_label"},
          "nearest_resistance": {"price", "distance_pct", "strength", "strength_label"},
          "risk_reward":        float,   # abs(resistance_dist) / abs(support_dist)
          "value_area_low":     float,   # 70% of volume between these two prices
          "value_area_high":    float,
          "current_price":      float,
          "available":          bool,
        }
    """
    _empty = {
        "levels": [], "nearest_support": None, "nearest_resistance": None,
        "risk_reward": 1.0, "value_area_low": None, "value_area_high": None,
        "current_price": None, "available": False,
    }

    try:
        ohlcv = data.get("ohlcv")
        if ohlcv is None or len(ohlcv) < 48:
            return _empty

        price_col = "price" if "price" in ohlcv.columns else "close"
        prices_s  = ohlcv[price_col].dropna()
        volumes_s = ohlcv["volume"].dropna()

        # Align lengths
        n = min(len(prices_s), len(volumes_s))
        prices  = prices_s.values[:n].astype(float)
        volumes = volumes_s.values[:n].astype(float)
        timestamps = np.arange(n)

        current_price = float(prices[-1])
        current_idx   = n - 1
        price_min     = prices[-lookback:].min() if n > lookback else prices.min()
        price_max     = prices[-lookback:].max() if n > lookback else prices.max()

        # ── Collect candidates from all methods ───────────────────────────────
        candidates = []

        # 1. Volume profile HVN
        vol_levels = _volume_profile_levels(prices, volumes, current_price, lookback)
        for price_lvl, vol_sc in vol_levels:
            candidates.append({
                "price": price_lvl, "vol_score": vol_sc,
                "is_vol": True, "is_swing": False, "is_round": False,
            })

        # 2. Swing highs/lows
        swing_lvls = _swing_levels(prices, timestamps)
        for price_lvl, _, _ in swing_lvls:
            if price_min * 0.97 <= price_lvl <= price_max * 1.03:
                candidates.append({
                    "price": price_lvl, "vol_score": 0.0,
                    "is_vol": False, "is_swing": True, "is_round": False,
                })

        # 3. Round numbers
        round_lvls = _round_levels(current_price, price_min, price_max)
        for price_lvl in round_lvls:
            candidates.append({
                "price": price_lvl, "vol_score": 0.0,
                "is_vol": False, "is_swing": False, "is_round": True,
            })

        if not candidates:
            return _empty

        # ── Merge nearby candidates ────────────────────────────────────────────
        merged = _merge_levels(candidates)

        # ── Score each merged level ────────────────────────────────────────────
        max_touches = max((
            _score_level(m["price"], prices, timestamps, current_idx)["touches"]
            for m in merged
        ), default=1) or 1

        scored = []
        for m in merged:
            stats = _score_level(m["price"], prices, timestamps, current_idx)
            n_touches = stats["touches"]

            # Multi-source bonus: level confirmed by 2+ methods is inherently stronger
            source_bonus = min(0.15 * (m["n_sources"] - 1), 0.25)

            vol_sc  = m["vol_score"]
            touch_n = min(n_touches, 8) / 8.0
            recency = stats["recency_score"]
            bounce  = stats["bounce_rate"]

            strength = (
                W_VOLUME  * vol_sc  +
                W_TOUCHES * touch_n +
                W_RECENCY * recency +
                W_BOUNCE  * bounce  +
                source_bonus
            )
            strength = min(float(strength), 1.0)

            pct_from_cur = (m["price"] - current_price) / (current_price + 1e-9) * 100
            level_type   = "support" if m["price"] < current_price else "resistance"

            scored.append({
                "price":          round(m["price"], 6),
                "type":           level_type,
                "strength":       round(strength, 3),
                "strength_label": _strength_label(strength),
                "touches":        n_touches,
                "bounce_rate":    round(stats["bounce_rate"], 2),
                "last_touch_h":   stats["last_touch_h"],
                "methods":        m["methods"],
                "distance_pct":   round(pct_from_cur, 2),
            })

        # ── Sort: supports by distance (closest first), resistances same ───────
        supports    = sorted(
            [s for s in scored if s["type"] == "support"],
            key=lambda x: abs(x["distance_pct"])
        )
        resistances = sorted(
            [s for s in scored if s["type"] == "resistance"],
            key=lambda x: abs(x["distance_pct"])
        )

        # Keep n_levels closest on each side, ranked by strength within that set
        top_supports    = sorted(supports[:n_levels],    key=lambda x: -x["strength"])
        top_resistances = sorted(resistances[:n_levels], key=lambda x: -x["strength"])
        all_levels = top_supports + top_resistances

        # ── Nearest levels ─────────────────────────────────────────────────────
        nearest_sup = supports[0] if supports else None
        nearest_res = resistances[0] if resistances else None

        risk_reward = 1.0
        if nearest_sup and nearest_res:
            sup_dist = abs(nearest_sup["distance_pct"])
            res_dist = abs(nearest_res["distance_pct"])
            risk_reward = round(res_dist / (sup_dist + 1e-9), 2)

        # ── Value area (where 70% of volume traded in last lookback) ───────────
        val_low, val_high = _value_area(prices, volumes, lookback)

        return {
            "levels":             all_levels,
            "nearest_support":    {
                "price":          nearest_sup["price"],
                "distance_pct":   nearest_sup["distance_pct"],
                "strength":       nearest_sup["strength"],
                "strength_label": nearest_sup["strength_label"],
            } if nearest_sup else None,
            "nearest_resistance": {
                "price":          nearest_res["price"],
                "distance_pct":   nearest_res["distance_pct"],
                "strength":       nearest_res["strength"],
                "strength_label": nearest_res["strength_label"],
            } if nearest_res else None,
            "risk_reward":        risk_reward,
            "value_area_low":     val_low,
            "value_area_high":    val_high,
            "current_price":      round(current_price, 6),
            "available":          True,
        }

    except Exception as e:
        return {**_empty, "error": str(e)}


def _value_area(prices, volumes, lookback):
    """
    Compute the value area: the price range containing 70% of volume.
    Returns (val_low, val_high).
    """
    try:
        n   = min(lookback, len(prices), len(volumes))
        px  = prices[-n:]
        vol = volumes[-n:]
        total_vol = vol.sum()
        target    = total_vol * 0.70

        lo = px.min() * 0.997
        hi = px.max() * 1.003
        bins = np.linspace(lo, hi, 101)
        profile = np.zeros(100)
        for p, v in zip(px, vol):
            idx = int((p - lo) / (hi - lo) * 100)
            idx = max(0, min(idx, 99))
            profile[idx] += v

        # Find peak bin and expand outward until 70% captured
        peak = int(np.argmax(profile))
        lo_i, hi_i = peak, peak
        cumvol = profile[peak]
        while cumvol < target:
            expand_lo = profile[lo_i - 1] if lo_i > 0 else 0
            expand_hi = profile[hi_i + 1] if hi_i < 99 else 0
            if expand_lo >= expand_hi and lo_i > 0:
                lo_i -= 1
                cumvol += profile[lo_i]
            elif hi_i < 99:
                hi_i += 1
                cumvol += profile[hi_i]
            else:
                break

        bin_mid = (bins[:-1] + bins[1:]) / 2
        return round(float(bin_mid[lo_i]), 6), round(float(bin_mid[hi_i]), 6)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Scalar feature extractor for trade_scorer.py
# ─────────────────────────────────────────────────────────────────────────────

def get_sr_features(data):
    """
    Return scalar features for the meta-model:
        dist_to_support_pct        — negative = price is above support (normal)
        dist_to_resistance_pct     — positive = price is below resistance (normal)
        sr_risk_reward             — resistance_dist / support_dist
        nearest_support_strength   — 0-1
        nearest_resistance_strength— 0-1
        inside_value_area          — 1 if price within value area, 0 if outside
    """
    defaults = {
        "dist_to_support_pct":         -5.0,
        "dist_to_resistance_pct":       5.0,
        "sr_risk_reward":               1.0,
        "nearest_support_strength":     0.5,
        "nearest_resistance_strength":  0.5,
        "inside_value_area":            1.0,
    }
    try:
        sr = compute_sr_levels(data, lookback=400, n_levels=5)
        if not sr.get("available"):
            return defaults

        cur = sr["current_price"]
        sup = sr.get("nearest_support")
        res = sr.get("nearest_resistance")
        val_lo = sr.get("value_area_low")
        val_hi = sr.get("value_area_high")

        return {
            "dist_to_support_pct":
                sup["distance_pct"] if sup else -5.0,
            "dist_to_resistance_pct":
                res["distance_pct"] if res else 5.0,
            "sr_risk_reward":
                sr["risk_reward"],
            "nearest_support_strength":
                sup["strength"] if sup else 0.5,
            "nearest_resistance_strength":
                res["strength"] if res else 0.5,
            "inside_value_area":
                1.0 if (val_lo and val_hi and val_lo <= cur <= val_hi) else 0.0,
        }
    except Exception:
        return defaults
