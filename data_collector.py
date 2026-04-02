"""
Data Collector for Fartcoin Alpha Framework

Data sources:
  1. CoinMarketCap API — historical daily OHLCV (spot)
  2. CoinGecko API (free) — real-time derivatives snapshots across all exchanges
     (funding rates, OI, volume, spread, basis)
  3. CoinGecko historical spot — price + volume chart data
  4. Polling accumulator — builds historical derivatives dataset over time
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CMC_BASE = "https://pro-api.coinmarketcap.com"
CG_BASE = "https://api.coingecko.com/api/v3"


def get_cmc_headers():
    api_key = os.environ.get("CMC_API_KEY")
    if not api_key:
        raise ValueError("Set CMC_API_KEY environment variable")
    return {"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}


def _safe_fetch(name, func, *args, **kwargs):
    """Wrapper that catches errors so collection continues."""
    try:
        df = func(*args, **kwargs)
        print(f"       Got {len(df)} rows")
        return df
    except Exception as e:
        print(f"       FAILED: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CoinMarketCap endpoints
# ---------------------------------------------------------------------------

def fetch_cmc_ohlcv_historical(symbol="FARTCOIN", days=90):
    """Daily OHLCV from CMC."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    resp = requests.get(
        f"{CMC_BASE}/v2/cryptocurrency/ohlcv/historical",
        headers=get_cmc_headers(),
        params={
            "symbol": symbol,
            "convert": "USD",
            "time_start": start.strftime("%Y-%m-%d"),
            "time_end": end.strftime("%Y-%m-%d"),
            "interval": "daily",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    quotes = data["data"][symbol][0]["quotes"] if symbol in data.get("data", {}) else []
    rows = []
    for q in quotes:
        usd = q["quote"]["USD"]
        rows.append({
            "timestamp": q["time_open"],
            "open": usd["open"],
            "high": usd["high"],
            "low": usd["low"],
            "close": usd["close"],
            "volume": usd["volume"],
            "market_cap": usd["market_cap"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.to_csv(DATA_DIR / f"{symbol}_ohlcv.csv")
    return df


# ---------------------------------------------------------------------------
# CoinGecko: Derivatives snapshots (free, no key, no geo-blocking)
# ---------------------------------------------------------------------------

def fetch_cg_derivatives_tickers(coin_filter="FARTCOIN"):
    """
    Pull ALL derivative tickers from CoinGecko, filter to our coin.
    Returns real funding rates, OI, volume, spread, basis across exchanges.
    """
    try:
        resp = requests.get(f"{CG_BASE}/derivatives", timeout=30)
        if resp.status_code == 429:
            print(f"       CoinGecko rate limited — waiting 60s...")
            time.sleep(60)
            resp = requests.get(f"{CG_BASE}/derivatives", timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"       CoinGecko derivatives failed: {e}")
        return pd.DataFrame()
    all_tickers = resp.json()

    # Filter to our coin's perpetual contracts
    fart_tickers = [
        t for t in all_tickers
        if t.get("index_id", "").upper() == coin_filter.upper()
        and t.get("contract_type") == "perpetual"
    ]

    if not fart_tickers:
        print(f"       No perpetual tickers found for {coin_filter}")
        return pd.DataFrame()

    rows = []
    for t in fart_tickers:
        rows.append({
            "exchange": t.get("market", ""),
            "symbol": t.get("symbol", ""),
            "price": float(t.get("price") or 0),
            "price_change_24h_pct": float(t.get("price_percentage_change_24h") or 0),
            "funding_rate": float(t.get("funding_rate") or 0),
            "open_interest_usd": float(t.get("open_interest") or 0),
            "volume_24h_usd": float(t.get("volume_24h") or 0),
            "spread": float(t.get("spread") or 0),
            "basis_pct": float(t.get("basis") or 0),
            "index_price": float(t.get("index") or 0),
            "last_traded": t.get("last_traded_at"),
        })

    df = pd.DataFrame(rows)
    df["snapshot_time"] = datetime.utcnow().isoformat()

    # Save raw snapshot
    snapshot_file = DATA_DIR / f"{coin_filter}_derivatives_snapshot.csv"
    df.to_csv(snapshot_file, index=False)

    return df


def fetch_cg_historical_chart(coin_id="fartcoin", days=90):
    """
    Historical price + volume from CoinGecko (free, no key).
    Gives us hourly data for <=90 days, daily for >90.
    """
    resp = requests.get(
        f"{CG_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Parse prices
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    prices.set_index("timestamp", inplace=True)

    # Parse volumes
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
    volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms")
    volumes.set_index("timestamp", inplace=True)

    # Parse market caps
    mcaps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
    mcaps["timestamp"] = pd.to_datetime(mcaps["timestamp"], unit="ms")
    mcaps.set_index("timestamp", inplace=True)

    df = prices.join(volumes).join(mcaps)
    df.to_csv(DATA_DIR / f"{coin_id}_cg_chart.csv")
    return df


def fetch_cg_exchange_details(exchange_id, coin_filter="FARTCOIN"):
    """Get detailed tickers from a specific exchange."""
    resp = requests.get(
        f"{CG_BASE}/derivatives/exchanges/{exchange_id}",
        params={"include_tickers": "unexpired"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    tickers = data.get("tickers", [])
    fart_tickers = [
        t for t in tickers
        if t.get("base", "").upper() == coin_filter.upper()
        and t.get("contract_type") == "perpetual"
    ]
    return fart_tickers


# ---------------------------------------------------------------------------
# Cross-Exchange Analysis (the manipulation detection layer)
# ---------------------------------------------------------------------------

def analyze_cross_exchange(snapshot_df):
    """
    Analyze derivatives snapshot for cross-exchange anomalies.
    This is where manipulation signals live.
    """
    if snapshot_df.empty:
        return {}

    analysis = {}

    # Filter to exchanges with meaningful OI
    active = snapshot_df[snapshot_df["open_interest_usd"] > 10000].copy()
    if active.empty:
        active = snapshot_df.copy()

    # --- Funding Rate Divergence ---
    # If one exchange has wildly different funding, MMs may be arbing it
    fr = active["funding_rate"]
    analysis["funding_rate_mean"] = fr.mean()
    analysis["funding_rate_std"] = fr.std()
    analysis["funding_rate_range"] = fr.max() - fr.min()
    analysis["funding_rate_max_exchange"] = active.loc[fr.idxmax(), "exchange"] if len(fr) > 0 else ""
    analysis["funding_rate_min_exchange"] = active.loc[fr.idxmin(), "exchange"] if len(fr) > 0 else ""

    # --- OI Concentration ---
    # If OI is concentrated on one exchange, that's where the manipulation happens
    total_oi = active["open_interest_usd"].sum()
    if total_oi > 0:
        active["oi_share"] = active["open_interest_usd"] / total_oi
        top_exchange = active.loc[active["oi_share"].idxmax()]
        analysis["total_oi_usd"] = total_oi
        analysis["top_oi_exchange"] = top_exchange["exchange"]
        analysis["top_oi_share"] = top_exchange["oi_share"]
        analysis["oi_herfindahl"] = (active["oi_share"] ** 2).sum()  # concentration index

    # --- Volume Concentration ---
    total_vol = active["volume_24h_usd"].sum()
    if total_vol > 0:
        active["vol_share"] = active["volume_24h_usd"] / total_vol
        top_vol = active.loc[active["vol_share"].idxmax()]
        analysis["total_volume_24h_usd"] = total_vol
        analysis["top_volume_exchange"] = top_vol["exchange"]
        analysis["top_volume_share"] = top_vol["vol_share"]

    # --- OI/Volume Ratio ---
    # High OI relative to volume = positions are sticky (not trading, just holding)
    # Low OI relative to volume = lots of churning (wash trading?)
    if total_vol > 0:
        analysis["oi_to_volume_ratio"] = total_oi / total_vol

    # --- Basis Spread ---
    # Large basis = perp price deviating from spot index
    # Positive basis = perp premium (bullish pressure)
    # Negative basis = perp discount (bearish pressure)
    basis = active["basis_pct"]
    analysis["avg_basis_pct"] = basis.mean()
    analysis["basis_range"] = basis.max() - basis.min()

    # --- Spread Analysis ---
    # Tight spreads on some exchanges, wide on others = liquidity fragmentation
    spreads = active["spread"]
    analysis["avg_spread"] = spreads.mean()
    analysis["spread_range"] = spreads.max() - spreads.min()

    return analysis


# ---------------------------------------------------------------------------
# Historical Derivatives Accumulator
# ---------------------------------------------------------------------------
# CoinGecko only gives snapshots. We poll periodically and append to build
# our own historical dataset.

HISTORY_FILE = DATA_DIR / "derivatives_history.csv"


def append_derivatives_snapshot(snapshot_df):
    """Append current snapshot to historical file."""
    if snapshot_df.empty:
        return

    # Compute aggregates for this snapshot
    active = snapshot_df[snapshot_df["open_interest_usd"] > 10000]
    if active.empty:
        active = snapshot_df

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "avg_funding_rate": active["funding_rate"].mean(),
        "max_funding_rate": active["funding_rate"].max(),
        "min_funding_rate": active["funding_rate"].min(),
        "funding_rate_spread": active["funding_rate"].max() - active["funding_rate"].min(),
        "total_oi_usd": active["open_interest_usd"].sum(),
        "total_volume_24h_usd": active["volume_24h_usd"].sum(),
        "oi_herfindahl": ((active["open_interest_usd"] / active["open_interest_usd"].sum()) ** 2).sum()
        if active["open_interest_usd"].sum() > 0 else 0,
        "avg_basis_pct": active["basis_pct"].mean(),
        "avg_spread": active["spread"].mean(),
        "avg_price": active["price"].mean(),
        "n_exchanges": len(active),
    }

    new_row = pd.DataFrame([row])

    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(HISTORY_FILE, index=False)
    return combined


# ---------------------------------------------------------------------------
# Synthetic perps signals from spot data (fallback)
# ---------------------------------------------------------------------------

def generate_synthetic_perps_signals(ohlcv_df, symbol="FARTCOINUSDT"):
    """Generate proxy perps signals from spot OHLCV when exchange APIs are blocked."""
    if ohlcv_df is None or ohlcv_df.empty:
        return {}

    df = ohlcv_df.copy()
    returns = df["close"].pct_change()

    # Synthetic Funding Rate: short-term vs long-term momentum
    momentum_5d = returns.rolling(5).mean()
    momentum_20d = returns.rolling(20).mean()
    funding_proxy = momentum_5d - momentum_20d
    funding_df = pd.DataFrame({"fundingRate": funding_proxy}, index=df.index)
    funding_df.to_csv(DATA_DIR / f"{symbol}_funding.csv")
    print(f"       Synthetic funding: {len(funding_df.dropna())} rows")

    # Synthetic OI: volume * volatility
    volatility = returns.rolling(10).std()
    vol_norm = df["volume"] / df["volume"].rolling(20).mean()
    oi_proxy = vol_norm * (1 + volatility * 100)
    oi_df = pd.DataFrame({
        "sumOpenInterest": oi_proxy,
        "sumOpenInterestValue": oi_proxy * df["close"],
    }, index=df.index)
    oi_df.to_csv(DATA_DIR / f"{symbol}_oi.csv")
    print(f"       Synthetic OI: {len(oi_df.dropna())} rows")

    # Synthetic LSR: RSI-based
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    lsr_proxy = rsi / (100 - rsi).replace(0, np.nan)
    lsr_df = pd.DataFrame({"longShortRatio": lsr_proxy}, index=df.index)
    lsr_df.to_csv(DATA_DIR / f"{symbol}_lsr.csv")
    print(f"       Synthetic LSR: {len(lsr_df.dropna())} rows")

    # Synthetic Taker: close position in range
    rng = df["high"] - df["low"]
    close_pos = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    taker_df = pd.DataFrame({"buySellRatio": close_pos.fillna(0.5)}, index=df.index)
    taker_df.to_csv(DATA_DIR / f"{symbol}_taker.csv")
    print(f"       Synthetic taker: {len(taker_df.dropna())} rows")

    return {"funding": funding_df, "oi": oi_df, "lsr": lsr_df, "taker": taker_df}


# ---------------------------------------------------------------------------
# Master collector
# ---------------------------------------------------------------------------

def collect_all(cmc_symbol="FARTCOIN", perp_symbol="FARTCOINUSDT",
                cg_coin_id="fartcoin", days=90):
    """Pull all available data from CMC + CoinGecko."""
    results = {}

    # --- Step 1: CMC historical OHLCV ---
    print("[1/4] Fetching CMC OHLCV (90d daily)...")
    ohlcv = _safe_fetch("CMC OHLCV", fetch_cmc_ohlcv_historical, cmc_symbol, days)
    results["ohlcv"] = ohlcv

    # --- Step 2: CoinGecko historical chart (hourly for 90d) ---
    print("[2/4] Fetching CoinGecko historical chart (hourly)...")
    cg_chart = _safe_fetch("CG Chart", fetch_cg_historical_chart, cg_coin_id, days)
    results["cg_chart"] = cg_chart

    # --- Step 2b: CoinGecko BTC chart (hourly for correlation analysis) ---
    print("[2b/4] Fetching CoinGecko BTC chart (hourly)...")
    btc_chart = _safe_fetch("CG BTC Chart", fetch_cg_historical_chart, "bitcoin", days)
    results["btc_chart"] = btc_chart

    # --- Step 3: CoinGecko derivatives snapshot (ALL exchanges) ---
    print("[3/4] Fetching CoinGecko derivatives snapshot...")
    deriv = _safe_fetch("CG Derivatives", fetch_cg_derivatives_tickers, cmc_symbol)
    results["derivatives"] = deriv

    if not deriv.empty:
        # Append to historical accumulator
        append_derivatives_snapshot(deriv)

        # Run cross-exchange analysis
        print("\n  --- CROSS-EXCHANGE ANALYSIS ---")
        analysis = analyze_cross_exchange(deriv)
        for k, v in analysis.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")
        results["cross_exchange"] = analysis

    # --- Step 4: Generate perps signals ---
    # Use CoinGecko snapshot for current state + synthetic historical
    print("\n[4/4] Generating perps signal history from spot data...")
    # Use CG hourly chart if available (more granular than CMC daily)
    if not cg_chart.empty:
        # Build OHLCV-like df from CG chart for signal generation
        cg_ohlcv = pd.DataFrame({
            "close": cg_chart["price"],
            "volume": cg_chart["volume"],
            "market_cap": cg_chart.get("market_cap", cg_chart["price"]),
            "open": cg_chart["price"],
            "high": cg_chart["price"],
            "low": cg_chart["price"],
        })
        synth = generate_synthetic_perps_signals(cg_ohlcv, perp_symbol)
        results.update(synth)

        # Override the ohlcv with CG hourly data for better signal resolution
        cg_ohlcv.to_csv(DATA_DIR / f"{cmc_symbol}_ohlcv_hourly.csv")
        results["ohlcv_hourly"] = cg_ohlcv
    elif not ohlcv.empty:
        synth = generate_synthetic_perps_signals(ohlcv, perp_symbol)
        results.update(synth)

    # --- Enrich synthetic funding with real current funding rate ---
    if not deriv.empty and "funding" in results:
        real_fr = deriv[deriv["open_interest_usd"] > 10000]["funding_rate"].mean()
        print(f"\n  Real current avg funding rate: {real_fr:.6f}")
        print(f"  (Synthetic latest: {results['funding']['fundingRate'].dropna().iloc[-1]:.6f})")

    print(f"\nAll data saved to {DATA_DIR}/")
    return results


# ---------------------------------------------------------------------------
# Polling mode: run periodically to build derivatives history
# ---------------------------------------------------------------------------

def poll_once(coin_filter="FARTCOIN"):
    """Single poll of derivatives data — call this from a cron/scheduler."""
    print(f"[{datetime.utcnow().isoformat()}] Polling derivatives...")
    snapshot = fetch_cg_derivatives_tickers(coin_filter)
    if not snapshot.empty:
        append_derivatives_snapshot(snapshot)
        analysis = analyze_cross_exchange(snapshot)
        print(f"  OI: ${analysis.get('total_oi_usd', 0):,.0f} | "
              f"Funding: {analysis.get('avg_funding_rate', 0):.6f} | "
              f"Basis: {analysis.get('avg_basis_pct', 0):.4f}%")
    return snapshot


if __name__ == "__main__":
    collect_all()
