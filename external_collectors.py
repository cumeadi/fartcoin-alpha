"""
External Data Collectors — Fartcoin Alpha Framework

New data sources beyond Binance/CoinGecko:
  1. CryptoPanic   — news sentiment (vote-based, free API)
  2. Helius        — Solana on-chain: holder distribution, whale transfers
  3. Coinalyze     — multi-exchange derivatives (aggregated OI, funding, liquidations)

Each collector:
  - Fetches data from the API
  - Saves raw CSV to data/
  - Returns a DataFrame for pipeline consumption

Usage:
  python3 external_collectors.py              # run all collectors
  python3 external_collectors.py --source X   # run one: cryptopanic | helius | coinalyze
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# FARTCOIN identifiers
FARTCOIN_MINT = "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump"
FARTCOIN_TICKER = "FARTCOIN"

# Known exchange deposit wallets (Solana) — expand this over time
# These are well-known hot wallets; transfers TO these = selling pressure
KNOWN_EXCHANGE_WALLETS = {
    # Binance hot wallets (Solana)
    "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9": "Binance",
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM": "Binance",
    # Bybit
    "AC5RDfQFmDS1deWZos921JfqscXdByf6BKHAbXeRY1ij": "Bybit",
    # OKX
    "5VCwKtCXgCDuQosUzavYqJ1XJquoUVj3gBHXZBZaACnW": "OKX",
    # Gate.io
    "u6PJ8DtQuPFnfmwHbGFULQ4u4EgjDiyYKjVEsynXq2w": "Gate.io",
}


def _safe_request(url, params=None, headers=None, timeout=30, name="API"):
    """Make a request with error handling and rate-limit awareness."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"  [{name}] Rate limited — retry after {retry_after}s")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [{name}] Request failed: {e}")
        return None


# ===========================================================================
# 1. Sentiment & Hype Detection (3 free APIs, no keys required)
#    - Alternative.me Fear & Greed Index (market-wide baseline)
#    - CoinGecko community sentiment (FARTCOIN-specific votes + watchlist)
#    - CoinPaprika granular price/volume (15m/1h spike detection)
# ===========================================================================

SENTIMENT_HISTORY_FILE = DATA_DIR / "sentiment_history.csv"


def fetch_fear_greed_index(limit=30):
    """
    Alternative.me Crypto Fear & Greed Index.

    API: https://api.alternative.me/fng/
    Free: no key, no signup, no rate limit issues.

    Returns 0-100 score: 0 = Extreme Fear, 100 = Extreme Greed.
    Key signal: Fartcoin pumping while market is in Extreme Fear = manipulation.
    """
    url = "https://api.alternative.me/fng/"
    data = _safe_request(url, params={"limit": limit, "format": "json"},
                         name="Fear & Greed")
    if data is None:
        return {}

    entries = data.get("data", [])
    if not entries:
        print("  [Fear & Greed] No data returned")
        return {}

    latest = entries[0]
    result = {
        "value": int(latest.get("value", 50)),
        "classification": latest.get("value_classification", "Neutral"),
        "timestamp": datetime.fromtimestamp(
            int(latest.get("timestamp", 0)), tz=timezone.utc
        ).isoformat(),
    }

    # Build history for trend analysis
    history = []
    for entry in entries:
        history.append({
            "timestamp": datetime.fromtimestamp(
                int(entry.get("timestamp", 0)), tz=timezone.utc
            ).isoformat(),
            "fear_greed_value": int(entry.get("value", 50)),
            "fear_greed_class": entry.get("value_classification", ""),
        })

    if history:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(DATA_DIR / "fear_greed_history.csv", index=False)

    # Trend: is fear increasing or decreasing?
    if len(entries) >= 7:
        recent_avg = np.mean([int(e["value"]) for e in entries[:3]])
        older_avg = np.mean([int(e["value"]) for e in entries[3:7]])
        result["trend"] = "IMPROVING" if recent_avg > older_avg + 5 else \
                          "DETERIORATING" if recent_avg < older_avg - 5 else "STABLE"
        result["recent_3d_avg"] = round(recent_avg, 1)
        result["prior_4d_avg"] = round(older_avg, 1)
    else:
        result["trend"] = "UNKNOWN"

    print(f"  [Fear & Greed] Value: {result['value']} ({result['classification']}), "
          f"Trend: {result.get('trend', 'N/A')}")

    return result


def fetch_coingecko_community(coin_id="fartcoin"):
    """
    CoinGecko community & market data for FARTCOIN.

    API: https://api.coingecko.com/api/v3/coins/{id}
    Free: no key, ~10-30 req/min.

    Key signals:
      - sentiment_votes_up/down_percentage (community vote ratio)
      - watchlist_portfolio_users (track changes = hype wave detection)
      - price_change_percentage_24h (for divergence with Fear & Greed)
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "false",
        "sparkline": "false",
    }

    data = _safe_request(url, params=params, name="CoinGecko Community")
    if data is None:
        return {}

    sentiment_up = data.get("sentiment_votes_up_percentage", 50)
    sentiment_down = data.get("sentiment_votes_down_percentage", 50)
    watchlist_users = data.get("watchlist_portfolio_users", 0)

    market = data.get("market_data", {})
    price = market.get("current_price", {}).get("usd", 0)
    price_change_24h = market.get("price_change_percentage_24h", 0)
    price_change_7d = market.get("price_change_percentage_7d", 0)
    volume_24h = market.get("total_volume", {}).get("usd", 0)
    market_cap = market.get("market_cap", {}).get("usd", 0)
    market_cap_rank = data.get("market_cap_rank", 0)

    result = {
        "sentiment_up_pct": round(sentiment_up or 50, 1),
        "sentiment_down_pct": round(sentiment_down or 50, 1),
        "sentiment_score": round((sentiment_up or 50) / 100, 3),  # normalize to 0-1
        "watchlist_users": watchlist_users or 0,
        "price_usd": price,
        "price_change_24h_pct": round(price_change_24h or 0, 2),
        "price_change_7d_pct": round(price_change_7d or 0, 2),
        "volume_24h_usd": volume_24h,
        "market_cap_usd": market_cap,
        "market_cap_rank": market_cap_rank or 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"  [CoinGecko] Sentiment: {result['sentiment_up_pct']:.0f}% bullish, "
          f"Watchlist: {result['watchlist_users']:,}, "
          f"24h change: {result['price_change_24h_pct']:+.1f}%")

    # Pause to avoid CoinGecko rate limits when poll_once hits CG next
    time.sleep(2)

    return result


def fetch_coinpaprika_granular(coin_id="fartcoin-fartcoin"):
    """
    CoinPaprika ticker data with granular price change intervals.

    API: https://api.coinpaprika.com/v1/tickers/{coin_id}
    Free: no key, 20 req/sec, top 2000 coins.

    Key signals:
      - percent_change_15m / 1h / 6h (short-term pump detection)
      - volume_24h_change_24h (volume spike without news = manipulation)
      - beta_value (market sensitivity — FARTCOIN beta ~2.65)
    """
    url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"
    data = _safe_request(url, name="CoinPaprika")
    if data is None:
        return {}

    quotes = data.get("quotes", {}).get("USD", {})

    result = {
        "price_usd": quotes.get("price", 0),
        "volume_24h_usd": quotes.get("volume_24h", 0),
        "volume_24h_change_pct": quotes.get("volume_24h_change_24h", 0),
        "market_cap_usd": quotes.get("market_cap", 0),
        "pct_change_15m": quotes.get("percent_change_15m", 0),
        "pct_change_1h": quotes.get("percent_change_1h", 0),
        "pct_change_6h": quotes.get("percent_change_6h", 0),
        "pct_change_12h": quotes.get("percent_change_12h", 0),
        "pct_change_24h": quotes.get("percent_change_24h", 0),
        "pct_change_7d": quotes.get("percent_change_7d", 0),
        "pct_change_30d": quotes.get("percent_change_30d", 0),
        "ath_price": quotes.get("ath_price", 0),
        "pct_from_ath": quotes.get("percent_from_price_ath", 0),
        "beta_value": data.get("beta_value", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Detect short-term pumps/dumps
    pct_15m = abs(result["pct_change_15m"] or 0)
    pct_1h = abs(result["pct_change_1h"] or 0)
    vol_change = result["volume_24h_change_pct"] or 0

    if pct_15m > 3 or pct_1h > 5:
        result["short_term_move"] = "SPIKE"
    elif pct_15m > 1.5 or pct_1h > 3:
        result["short_term_move"] = "ELEVATED"
    else:
        result["short_term_move"] = "NORMAL"

    if vol_change > 50:
        result["volume_anomaly"] = "SPIKE"
    elif vol_change > 25:
        result["volume_anomaly"] = "ELEVATED"
    else:
        result["volume_anomaly"] = "NORMAL"

    print(f"  [CoinPaprika] 15m: {result['pct_change_15m']:+.2f}%, "
          f"1h: {result['pct_change_1h']:+.2f}%, "
          f"Vol change: {vol_change:+.0f}%, "
          f"Beta: {result['beta_value']:.2f}")

    return result


def collect_sentiment():
    """
    Collect all sentiment data from the 3 free APIs and combine into
    a single snapshot for signal consumption.

    Saves:
      - data/sentiment_history.csv (appending each snapshot)
      - data/news_sentiment_hourly.csv (for projections.py compatibility)
    """
    print("  Collecting sentiment data...")

    # Fetch all three
    fear_greed = fetch_fear_greed_index(limit=7)
    cg_community = fetch_coingecko_community()
    paprika = fetch_coinpaprika_granular()

    if not fear_greed and not cg_community and not paprika:
        print("  [Sentiment] All sources failed")
        return pd.DataFrame()

    # Build combined snapshot
    now = datetime.now(timezone.utc)
    snapshot = {
        "timestamp": now.isoformat(),
        # Fear & Greed
        "fear_greed_value": fear_greed.get("value", 50),
        "fear_greed_class": fear_greed.get("classification", "Neutral"),
        "fear_greed_trend": fear_greed.get("trend", "UNKNOWN"),
        # CoinGecko community
        "cg_sentiment_up_pct": cg_community.get("sentiment_up_pct", 50),
        "cg_watchlist_users": cg_community.get("watchlist_users", 0),
        "cg_price_change_24h": cg_community.get("price_change_24h_pct", 0),
        # CoinPaprika granular
        "cp_pct_change_15m": paprika.get("pct_change_15m", 0),
        "cp_pct_change_1h": paprika.get("pct_change_1h", 0),
        "cp_pct_change_6h": paprika.get("pct_change_6h", 0),
        "cp_volume_change_24h": paprika.get("volume_24h_change_pct", 0),
        "cp_beta": paprika.get("beta_value", 0),
        "cp_short_term_move": paprika.get("short_term_move", "NORMAL"),
        "cp_volume_anomaly": paprika.get("volume_anomaly", "NORMAL"),
    }

    # --- Compute composite sentiment score ---
    # Normalize each source to [-1, +1]:
    #   Fear & Greed: 0-100 → map to [-1, +1] where 50 = 0
    fg_norm = (snapshot["fear_greed_value"] - 50) / 50

    #   CoinGecko sentiment: 0-100% bullish → map to [-1, +1]
    cg_norm = (snapshot["cg_sentiment_up_pct"] - 50) / 50

    #   CoinPaprika 1h change: clip to [-10, +10]% → map to [-1, +1]
    cp_1h = np.clip(snapshot["cp_pct_change_1h"], -10, 10) / 10

    # Weighted composite: market sentiment (30%) + community (30%) + price action (40%)
    composite = 0.3 * fg_norm + 0.3 * cg_norm + 0.4 * cp_1h
    snapshot["sentiment_composite"] = round(composite, 3)

    # --- Divergence detection ---
    # Fartcoin pumping while market is fearful = manipulation signal
    fart_pumping = snapshot["cp_pct_change_1h"] > 2
    market_fearful = snapshot["fear_greed_value"] < 30
    volume_spike = snapshot["cp_volume_change_24h"] > 50

    if fart_pumping and market_fearful:
        snapshot["divergence"] = "PUMP_IN_FEAR"
        snapshot["divergence_desc"] = (
            f"FART pumping ({snapshot['cp_pct_change_1h']:+.1f}% 1h) "
            f"while market is in {snapshot['fear_greed_class']} "
            f"(FG={snapshot['fear_greed_value']}). Possible manipulation.")
    elif fart_pumping and volume_spike:
        snapshot["divergence"] = "VOLUME_PUMP"
        snapshot["divergence_desc"] = (
            f"FART pumping ({snapshot['cp_pct_change_1h']:+.1f}% 1h) "
            f"with volume spike ({snapshot['cp_volume_change_24h']:+.0f}%). "
            f"Watch for reversal.")
    elif snapshot["cp_pct_change_1h"] < -3 and snapshot["cg_sentiment_up_pct"] > 65:
        snapshot["divergence"] = "DUMP_IN_OPTIMISM"
        snapshot["divergence_desc"] = (
            f"FART dumping ({snapshot['cp_pct_change_1h']:+.1f}% 1h) "
            f"despite bullish community sentiment ({snapshot['cg_sentiment_up_pct']:.0f}% bullish). "
            f"Possible whale distribution.")
    else:
        snapshot["divergence"] = "NONE"
        snapshot["divergence_desc"] = ""

    # Append to history
    new_row = pd.DataFrame([snapshot])
    if SENTIMENT_HISTORY_FILE.exists():
        existing = pd.read_csv(SENTIMENT_HISTORY_FILE)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(SENTIMENT_HISTORY_FILE, index=False)

    # Also write news_sentiment_hourly.csv for backward compatibility with projections.py
    # Map our composite into the format the news sentiment model expects
    hourly_row = {
        "timestamp": now.isoformat(),
        "news_count": 1,  # we have data
        "news_sentiment": snapshot["sentiment_composite"],
        "news_sentiment_std": 0,
        "vote_intensity": snapshot["cg_watchlist_users"],
        "news_buzz": 1.0 if snapshot["divergence"] != "NONE" else 0.0,
    }
    hourly_df = pd.DataFrame([hourly_row])
    hourly_df.set_index("timestamp", inplace=True)

    hourly_file = DATA_DIR / "news_sentiment_hourly.csv"
    if hourly_file.exists():
        existing_h = pd.read_csv(hourly_file, index_col=0)
        combined_h = pd.concat([existing_h, hourly_df])
        # Keep last 168 rows (7 days at 30min polling)
        combined_h = combined_h.tail(336)
    else:
        combined_h = hourly_df
    combined_h.to_csv(hourly_file)

    print(f"  [Sentiment] Composite: {composite:+.3f} | "
          f"Divergence: {snapshot['divergence']}")

    return new_row


# ===========================================================================
# 2. Helius — Solana On-Chain Data
# ===========================================================================

def fetch_helius_holders(mint=FARTCOIN_MINT, top_n=50):
    """
    Fetch top token holders and compute concentration metrics.

    API: https://mainnet.helius-rpc.com/
    Free tier: 1M credits/month, DAS calls = 10 credits each.

    Returns:
      - holders_df: top N holders with balance and % share
      - metrics: dict with gini, top10_pct, top20_pct, holder_count
    """
    api_key = os.environ.get("HELIUS_API_KEY", "")
    if not api_key:
        print("  [Helius] No API key — set HELIUS_API_KEY env var")
        print("  [Helius] Get free key at: https://www.helius.dev/")
        return pd.DataFrame(), {}

    url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"

    # Use getTokenLargestAccounts (standard Solana RPC, works reliably)
    payload = {
        "jsonrpc": "2.0",
        "id": "fartcoin-holders",
        "method": "getTokenLargestAccounts",
        "params": [mint],
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [Helius] Request failed: {e}")
        return pd.DataFrame(), {}

    accounts = data.get("result", {}).get("value", [])

    if not accounts:
        print("  [Helius] No token accounts found")
        return pd.DataFrame(), {}

    total_holders = len(accounts)  # top 20 from RPC

    rows = []
    for acc in accounts:
        amount = float(acc.get("amount", 0))
        decimals = int(acc.get("decimals", 6))
        balance = amount / (10 ** decimals) if decimals > 0 else amount

        token_account = acc.get("address", "")
        rows.append({
            "owner": token_account,  # token account address (not owner wallet)
            "token_account": token_account,
            "balance": balance,
            "is_exchange": token_account in KNOWN_EXCHANGE_WALLETS,
            "exchange_name": KNOWN_EXCHANGE_WALLETS.get(token_account, ""),
        })

    df = pd.DataFrame(rows)
    df.sort_values("balance", ascending=False, inplace=True)
    df = df.head(top_n)

    # Compute concentration metrics
    total_supply = df["balance"].sum()  # approximate from top holders
    metrics = {}

    if total_supply > 0:
        shares = df["balance"].values / total_supply
        # Gini coefficient
        n = len(shares)
        if n > 1:
            sorted_shares = np.sort(shares)
            index = np.arange(1, n + 1)
            metrics["gini"] = (2 * np.sum(index * sorted_shares) / (n * np.sum(sorted_shares))) - (n + 1) / n
        else:
            metrics["gini"] = 1.0

        metrics["top10_pct"] = shares[:10].sum() * 100 if len(shares) >= 10 else shares.sum() * 100
        metrics["top20_pct"] = shares[:20].sum() * 100 if len(shares) >= 20 else shares.sum() * 100
        metrics["exchange_held_pct"] = df[df["is_exchange"]]["balance"].sum() / total_supply * 100

    metrics["total_holders"] = total_holders
    metrics["snapshot_time"] = datetime.now(timezone.utc).isoformat()

    # Save
    df.to_csv(DATA_DIR / "helius_holders.csv", index=False)

    # Append concentration metrics to history
    _append_holder_metrics(metrics)

    print(f"  [Helius] Top {len(df)} holders fetched. "
          f"Gini: {metrics.get('gini', 0):.3f}, "
          f"Top 10: {metrics.get('top10_pct', 0):.1f}%, "
          f"Total holders: {total_holders}")

    return df, metrics


def _append_holder_metrics(metrics):
    """Append holder concentration metrics to historical tracking file."""
    history_file = DATA_DIR / "holder_concentration_history.csv"
    new_row = pd.DataFrame([metrics])

    if history_file.exists():
        existing = pd.read_csv(history_file)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(history_file, index=False)


def fetch_helius_recent_transfers(mint=FARTCOIN_MINT, min_usd=10000):
    """
    Fetch recent large token transfers to detect whale movements.

    Uses getSignaturesForAddress + parsed transaction data.
    Flags transfers TO known exchange wallets (potential sells).

    Returns DataFrame with:
        timestamp, from_wallet, to_wallet, amount, is_to_exchange, exchange_name
    """
    api_key = os.environ.get("HELIUS_API_KEY", "")
    if not api_key:
        print("  [Helius] No API key — set HELIUS_API_KEY")
        return pd.DataFrame()

    url = f"https://api.helius.xyz/v0/addresses/{mint}/transactions?api-key={api_key}&limit=100"

    data = _safe_request(url, name="Helius Transactions")
    if data is None:
        return pd.DataFrame()

    rows = []
    for tx in data:
        ts = tx.get("timestamp", 0)
        tx_type = tx.get("type", "")
        desc = tx.get("description", "")

        # Parse token transfers from the transaction
        token_transfers = tx.get("tokenTransfers", [])
        for transfer in token_transfers:
            if transfer.get("mint", "") != mint:
                continue

            amount = float(transfer.get("tokenAmount", 0))
            from_addr = transfer.get("fromUserAccount", "")
            to_addr = transfer.get("toUserAccount", "")

            to_exchange = to_addr in KNOWN_EXCHANGE_WALLETS
            from_exchange = from_addr in KNOWN_EXCHANGE_WALLETS

            rows.append({
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "signature": tx.get("signature", "")[:20],
                "type": tx_type,
                "from_wallet": from_addr[:12] + "..." if from_addr else "",
                "to_wallet": to_addr[:12] + "..." if to_addr else "",
                "from_full": from_addr,
                "to_full": to_addr,
                "amount": amount,
                "is_to_exchange": to_exchange,
                "is_from_exchange": from_exchange,
                "exchange_name": (
                    KNOWN_EXCHANGE_WALLETS.get(to_addr, "")
                    or KNOWN_EXCHANGE_WALLETS.get(from_addr, "")
                ),
                "flow_direction": (
                    "TO_EXCHANGE" if to_exchange
                    else "FROM_EXCHANGE" if from_exchange
                    else "WALLET_TO_WALLET"
                ),
            })

    if not rows:
        print("  [Helius] No token transfers found in recent transactions")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", ascending=False, inplace=True)

    # Save
    df.to_csv(DATA_DIR / "helius_transfers.csv", index=False)

    # Summary stats
    to_exchange = df[df["is_to_exchange"]]
    from_exchange = df[df["is_from_exchange"]]
    print(f"  [Helius] {len(df)} transfers found")
    print(f"    To exchanges: {len(to_exchange)} transfers, "
          f"{to_exchange['amount'].sum():,.0f} tokens")
    print(f"    From exchanges: {len(from_exchange)} transfers, "
          f"{from_exchange['amount'].sum():,.0f} tokens")

    # Compute net flow signal
    net_flow = from_exchange["amount"].sum() - to_exchange["amount"].sum()
    flow_signal = "BULLISH (net withdrawal)" if net_flow > 0 else "BEARISH (net deposit)"
    print(f"    Net flow: {net_flow:+,.0f} tokens ({flow_signal})")

    # Append flow summary to history
    _append_flow_metrics(df)

    return df


def _append_flow_metrics(transfers_df):
    """Compute and append hourly flow metrics."""
    if transfers_df.empty:
        return

    to_ex = transfers_df[transfers_df["is_to_exchange"]]["amount"].sum()
    from_ex = transfers_df[transfers_df["is_from_exchange"]]["amount"].sum()

    row = {
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "transfers_count": len(transfers_df),
        "to_exchange_tokens": to_ex,
        "from_exchange_tokens": from_ex,
        "net_flow_tokens": from_ex - to_ex,  # positive = withdrawal = bullish
        "largest_transfer": transfers_df["amount"].max(),
        "whale_transfers": (transfers_df["amount"] > 100000).sum(),
    }

    history_file = DATA_DIR / "exchange_flow_history.csv"
    new_row = pd.DataFrame([row])

    if history_file.exists():
        existing = pd.read_csv(history_file)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(history_file, index=False)


# ===========================================================================
# 3. Coinalyze — Multi-Exchange Derivatives
# ===========================================================================

# Coinalyze API: https://api.coinalyze.net/v1/
# Free: 40 requests/minute, no key needed for basic endpoints.
# Key required for some endpoints (free signup).

COINALYZE_BASE = "https://api.coinalyze.net/v1"

# FARTCOIN futures symbols across exchanges
# Format: BASE_QUOTE_EXCHANGE  (Coinalyze convention)
COINALYZE_SYMBOLS = [
    "FARTCOINUSDT_PERP.A",   # Binance
    "FARTCOINUSDT.6",         # Bybit
    "FARTCOINUSDT_PERP.3",   # OKX
    "FARTCOINUSDT_PERP.4",   # Bitget
]


def _coinalyze_headers():
    """Get Coinalyze auth headers if API key is available."""
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if api_key:
        return {"api_key": api_key}
    return {}


def fetch_coinalyze_oi_history(hours=168):
    """
    Fetch aggregated open interest history across exchanges.

    API: /open-interest-history
    Returns hourly OI for each exchange + combined.
    """
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if not api_key:
        print("  [Coinalyze] No API key — set COINALYZE_API_KEY env var")
        print("  [Coinalyze] Get free key at: https://coinalyze.net/")
        return pd.DataFrame()

    symbols = ",".join(COINALYZE_SYMBOLS)
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (hours * 3600)

    url = f"{COINALYZE_BASE}/open-interest-history"
    params = {
        "symbols": symbols,
        "interval": "1hour",
        "from": start,
        "to": now,
        "api_key": api_key,
    }

    data = _safe_request(url, params=params, name="Coinalyze OI")
    if data is None:
        return pd.DataFrame()

    all_rows = []
    for series in data:
        symbol = series.get("symbol", "unknown")
        exchange = symbol.split(".")[-1] if "." in symbol else "unknown"

        for point in series.get("history", []):
            all_rows.append({
                "timestamp": datetime.fromtimestamp(point["t"], tz=timezone.utc),
                "exchange": exchange,
                "symbol": symbol,
                "open_interest": point.get("o", 0),
                "open_interest_value": point.get("v", 0),
            })

    if not all_rows:
        print("  [Coinalyze] No OI history returned")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Pivot to get per-exchange and combined OI
    pivot = df.pivot_table(
        index="timestamp", columns="exchange",
        values="open_interest_value", aggfunc="sum"
    )
    pivot["total_oi"] = pivot.sum(axis=1)
    pivot.to_csv(DATA_DIR / "coinalyze_oi_history.csv")

    print(f"  [Coinalyze] OI history: {len(pivot)} hourly rows, "
          f"{len(pivot.columns) - 1} exchanges")

    return pivot


def fetch_coinalyze_funding_history(hours=168):
    """
    Fetch funding rate history across exchanges.

    API: /funding-rate-history
    Reveals cross-exchange funding divergence (arb opportunities).
    """
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if not api_key:
        print("  [Coinalyze] No API key — set COINALYZE_API_KEY")
        return pd.DataFrame()

    symbols = ",".join(COINALYZE_SYMBOLS)
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (hours * 3600)

    url = f"{COINALYZE_BASE}/funding-rate-history"
    params = {
        "symbols": symbols,
        "interval": "1hour",
        "from": start,
        "to": now,
        "api_key": api_key,
    }

    data = _safe_request(url, params=params, name="Coinalyze Funding")
    if data is None:
        return pd.DataFrame()

    all_rows = []
    for series in data:
        symbol = series.get("symbol", "unknown")
        exchange = symbol.split(".")[-1] if "." in symbol else "unknown"

        for point in series.get("history", []):
            all_rows.append({
                "timestamp": datetime.fromtimestamp(point["t"], tz=timezone.utc),
                "exchange": exchange,
                "funding_rate": point.get("o", 0),
            })

    if not all_rows:
        print("  [Coinalyze] No funding history returned")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(
        index="timestamp", columns="exchange",
        values="funding_rate", aggfunc="mean"
    )
    pivot["mean_funding"] = pivot.mean(axis=1)
    pivot["funding_spread"] = pivot.drop(columns=["mean_funding"], errors="ignore").max(axis=1) - \
                               pivot.drop(columns=["mean_funding"], errors="ignore").min(axis=1)
    pivot.to_csv(DATA_DIR / "coinalyze_funding_history.csv")

    print(f"  [Coinalyze] Funding history: {len(pivot)} hourly rows")
    if len(pivot) > 0:
        latest_spread = pivot["funding_spread"].iloc[-1]
        print(f"    Latest cross-exchange funding spread: {latest_spread:.6f}")

    return pivot


def fetch_coinalyze_liquidations(hours=168):
    """
    Fetch liquidation history — shows where leveraged positions get wiped.

    API: /liquidation-history
    Large liquidation clusters = forced selling/buying = manipulation fuel.
    """
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if not api_key:
        print("  [Coinalyze] No API key — set COINALYZE_API_KEY")
        return pd.DataFrame()

    symbols = ",".join(COINALYZE_SYMBOLS)
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (hours * 3600)

    url = f"{COINALYZE_BASE}/liquidation-history"
    params = {
        "symbols": symbols,
        "interval": "1hour",
        "from": start,
        "to": now,
        "api_key": api_key,
    }

    data = _safe_request(url, params=params, name="Coinalyze Liquidations")
    if data is None:
        return pd.DataFrame()

    all_rows = []
    for series in data:
        symbol = series.get("symbol", "unknown")
        exchange = symbol.split(".")[-1] if "." in symbol else "unknown"

        for point in series.get("history", []):
            all_rows.append({
                "timestamp": datetime.fromtimestamp(point["t"], tz=timezone.utc),
                "exchange": exchange,
                "long_liquidations": point.get("l", 0),
                "short_liquidations": point.get("s", 0),
            })

    if not all_rows:
        print("  [Coinalyze] No liquidation data returned")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Aggregate across exchanges per hour
    hourly = df.groupby("timestamp").agg({
        "long_liquidations": "sum",
        "short_liquidations": "sum",
    })
    hourly["total_liquidations"] = hourly["long_liquidations"] + hourly["short_liquidations"]
    hourly["liq_ratio"] = hourly["long_liquidations"] / hourly["total_liquidations"].replace(0, np.nan)

    # Spike detection: z-score vs trailing 24h
    rolling_mean = hourly["total_liquidations"].rolling(24, min_periods=1).mean()
    rolling_std = hourly["total_liquidations"].rolling(24, min_periods=1).std().replace(0, 1)
    hourly["liq_zscore"] = (hourly["total_liquidations"] - rolling_mean) / rolling_std

    hourly.to_csv(DATA_DIR / "coinalyze_liquidations.csv")
    print(f"  [Coinalyze] Liquidation history: {len(hourly)} hourly rows")

    if len(hourly) > 0:
        latest = hourly.iloc[-1]
        print(f"    Latest hour: {latest['long_liquidations']:,.0f} long liqs, "
              f"{latest['short_liquidations']:,.0f} short liqs")

    return hourly


def fetch_coinalyze_predicted_funding():
    """
    Fetch predicted funding rates — unique to Coinalyze.
    Shows what funding WILL be at next settlement, before it happens.

    API: /predicted-funding-rate-history
    Signal: if predicted funding is extreme, squeeze is incoming.
    """
    api_key = os.environ.get("COINALYZE_API_KEY", "")
    if not api_key:
        print("  [Coinalyze] No API key — set COINALYZE_API_KEY")
        return pd.DataFrame()

    symbols = ",".join(COINALYZE_SYMBOLS)
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (72 * 3600)  # Last 3 days

    url = f"{COINALYZE_BASE}/predicted-funding-rate-history"
    params = {
        "symbols": symbols,
        "interval": "1hour",
        "from": start,
        "to": now,
        "api_key": api_key,
    }

    data = _safe_request(url, params=params, name="Coinalyze Predicted Funding")
    if data is None:
        return pd.DataFrame()

    all_rows = []
    for series in data:
        symbol = series.get("symbol", "unknown")
        exchange = symbol.split(".")[-1] if "." in symbol else "unknown"

        for point in series.get("history", []):
            all_rows.append({
                "timestamp": datetime.fromtimestamp(point["t"], tz=timezone.utc),
                "exchange": exchange,
                "predicted_funding": point.get("o", 0),
            })

    if not all_rows:
        print("  [Coinalyze] No predicted funding data")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(
        index="timestamp", columns="exchange",
        values="predicted_funding", aggfunc="mean"
    )
    pivot["mean_predicted"] = pivot.mean(axis=1)
    pivot.to_csv(DATA_DIR / "coinalyze_predicted_funding.csv")

    print(f"  [Coinalyze] Predicted funding: {len(pivot)} rows")

    return pivot


# ===========================================================================
# Master Collector
# ===========================================================================

def collect_all_external():
    """Run all external data collectors."""
    print("=" * 70)
    print("EXTERNAL DATA COLLECTION")
    print("=" * 70)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    results = {}

    # --- 1. Sentiment & Hype Detection ---
    print("\n[1/3] Sentiment — Fear & Greed + CoinGecko + CoinPaprika")
    print("-" * 40)
    results["sentiment"] = collect_sentiment()

    # --- 2. Helius On-Chain ---
    print("\n[2/3] Helius — Solana On-Chain")
    print("-" * 40)
    holders_df, holder_metrics = fetch_helius_holders()
    results["holders"] = holders_df
    results["holder_metrics"] = holder_metrics

    print()
    transfers_df = fetch_helius_recent_transfers()
    results["transfers"] = transfers_df

    # --- 3. Coinalyze Multi-Exchange Derivatives ---
    print("\n[3/3] Coinalyze — Multi-Exchange Derivatives")
    print("-" * 40)
    results["coinalyze_oi"] = fetch_coinalyze_oi_history()
    results["coinalyze_funding"] = fetch_coinalyze_funding_history()
    results["coinalyze_liquidations"] = fetch_coinalyze_liquidations()
    results["coinalyze_predicted_funding"] = fetch_coinalyze_predicted_funding()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)

    for name, data in results.items():
        if isinstance(data, pd.DataFrame):
            status = f"{len(data)} rows" if not data.empty else "EMPTY"
        elif isinstance(data, dict):
            status = f"{len(data)} metrics" if data else "EMPTY"
        else:
            status = "N/A"
        print(f"  {name:30s} {status}")

    print(f"\nAll data saved to {DATA_DIR}/")
    return results


def collect_light_external():
    """
    Light poll: just the fast, incremental data for scheduled runs.
    Skips historical fetches, focuses on latest state.
    """
    results = {}

    # Sentiment snapshot
    results["sentiment"] = collect_sentiment()

    # Holder concentration snapshot
    holders_df, metrics = fetch_helius_holders()
    results["holder_metrics"] = metrics

    # Recent transfers
    results["transfers"] = fetch_helius_recent_transfers()

    # Predicted funding (last 3 days, lightweight)
    results["predicted_funding"] = fetch_coinalyze_predicted_funding()

    return results


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="External data collectors")
    parser.add_argument("--source", choices=["sentiment", "helius", "coinalyze", "all"],
                        default="all", help="Which source to collect from")
    args = parser.parse_args()

    if args.source == "sentiment":
        collect_sentiment()
    elif args.source == "helius":
        fetch_helius_holders()
        fetch_helius_recent_transfers()
    elif args.source == "coinalyze":
        fetch_coinalyze_oi_history()
        fetch_coinalyze_funding_history()
        fetch_coinalyze_liquidations()
        fetch_coinalyze_predicted_funding()
    else:
        collect_all_external()
