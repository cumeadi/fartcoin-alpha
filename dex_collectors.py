"""
DEX Collectors — Fartcoin Alpha Framework

Pulls liquidity, volume, and pair data from DEXes via the free DexScreener API.
This allows us to track DEX liquidity manipulation (e.g. MMs pulling liquidity).
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Fartcoin Mint Address
FARTCOIN_MINT = "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump"
HISTORY_FILE = DATA_DIR / "dex_history.csv"


def _safe_request(url, params=None, headers=None, timeout=30, name="API"):
    """Make a request with basic error handling."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            print(f"  [{name}] Rate limited — retry later")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [{name}] Request failed: {e}")
        return None


def fetch_dexscreener_data(mint=FARTCOIN_MINT):
    """
    Fetch all liquidity pools for the given token from DexScreener.
    Aggregates liquidity and volume to get a macro view of DEX activity.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] Polling DexScreener...")
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
    data = _safe_request(url, name="DexScreener")
    
    if data is None or "pairs" not in data or not data["pairs"]:
        print("  [DexScreener] No pairs found or API failed.")
        return pd.DataFrame()

    pairs = data["pairs"]
    
    rows = []
    total_liquidity = 0
    total_volume_24h = 0
    
    for pair in pairs:
        # Ignore extremely small or inactive pools (e.g., <$1k liquidity)
        liq_usd = float(pair.get("liquidity", {}).get("usd", 0))
        vol_24h = float(pair.get("volume", {}).get("h24", 0))
        
        if liq_usd < 1000:
            continue
            
        dex_id = pair.get("dexId", "unknown")
        pair_addr = pair.get("pairAddress", "")
        price_usd = float(pair.get("priceUsd", 0))
        
        total_liquidity += liq_usd
        total_volume_24h += vol_24h
        
        rows.append({
            "dex": dex_id,
            "pair_address": pair_addr,
            "price_usd": price_usd,
            "liquidity_usd": liq_usd,
            "volume_24h": vol_24h,
            "txns_24h_buys": pair.get("txns", {}).get("h24", {}).get("buys", 0),
            "txns_24h_sells": pair.get("txns", {}).get("h24", {}).get("sells", 0),
        })

    if not rows:
        print("  [DexScreener] No active pairs found with >$1k liquidity.")
        return pd.DataFrame()

    # Create a summary snapshot
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    # Assume price is volume-weighted or just from the biggest pool
    df_pairs = pd.DataFrame(rows)
    df_pairs.sort_values("liquidity_usd", ascending=False, inplace=True)
    best_price = df_pairs.iloc[0]["price_usd"]
    
    # Weekend detection for tagging
    from market_state import is_weekend
    is_we = is_weekend(now_dt)
    
    # Aggregate buy/sell txn counts across all active pools
    total_buys  = int(df_pairs["txns_24h_buys"].sum())
    total_sells = int(df_pairs["txns_24h_sells"].sum())
    total_txns  = total_buys + total_sells
    dex_buy_pct = total_buys / total_txns if total_txns > 0 else 0.5

    snapshot = {
        "timestamp": now,
        "total_dex_liquidity_usd": total_liquidity,
        "total_dex_volume_24h_usd": total_volume_24h,
        "weighted_price_usd": best_price,
        "active_pools": len(rows),
        "top_dex": df_pairs.iloc[0]["dex"],
        "top_pool_liquidity": df_pairs.iloc[0]["liquidity_usd"],
        "total_buys_24h": total_buys,
        "total_sells_24h": total_sells,
        "dex_buy_pct": round(dex_buy_pct, 4),   # >0.55 = buy pressure, <0.45 = sell pressure
        "is_weekend": is_we,
    }

    print(f"  [DexScreener] Total Liquidity: ${total_liquidity:,.0f} across {len(rows)} pools {'(WEEKEND)' if is_we else ''}")
    print(f"  [DexScreener] 24h DEX Volume: ${total_volume_24h:,.0f} | Buy%: {dex_buy_pct:.1%} ({total_buys:,} buys / {total_sells:,} sells)")
    
    snapshot_df = pd.DataFrame([snapshot])
    
    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([existing, snapshot_df], ignore_index=True)
    else:
        combined = snapshot_df
        
    combined.to_csv(HISTORY_FILE, index=False)
    
    # Save the detailed breakdown for the latest poll
    df_pairs.to_csv(DATA_DIR / "dex_pairs_latest.csv", index=False)
    
    return combined

if __name__ == "__main__":
    fetch_dexscreener_data()
