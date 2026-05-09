"""
Bybit public tick data ingestion + Bybit V5 REST OI/LSR refresh.

Reads daily .csv.gz files from public.bybit.com/trading/<SYMBOL>/
Builds hourly OHLCV candles and real taker BSR, appends to pipeline CSVs.
Also fetches latest OI and LSR from Bybit V5 REST API and appends to their CSVs.

Functions
---------
ingest_yesterday(symbol)   — download yesterday's tick file + refresh OI/LSR (for cron)
ingest_range(start, end, symbol) — bulk download a date range (for backfill)
refresh_oi_lsr(hours, symbol)    — pull the last N hours of OI + LSR and append

All functions default to FARTCOINUSDT for backward compatibility.
"""

import csv
import datetime
import gzip
import io
import re
import time

import numpy as np
import pandas as pd
import requests

BYBIT_API       = "https://api.bybit.com"
DEFAULT_SYMBOL  = "FARTCOINUSDT"
DATA_DIR        = __import__("pathlib").Path(__file__).parent / "data"

# Legacy module-level constants (kept for backward compatibility)
BASE_URL   = f"https://public.bybit.com/trading/{DEFAULT_SYMBOL}/"
SYMBOL     = DEFAULT_SYMBOL
OHLCV_PATH = DATA_DIR / "FARTCOIN_ohlcv_hourly.csv"
TAKER_PATH = DATA_DIR / "FARTCOINUSDT_taker.csv"
OI_PATH    = DATA_DIR / "FARTCOINUSDT_oi.csv"
LSR_PATH   = DATA_DIR / "FARTCOINUSDT_lsr.csv"


def _get_paths(symbol: str = DEFAULT_SYMBOL) -> tuple:
    """Return (base_url, ohlcv_path, taker_path, oi_path, lsr_path) for a symbol.

    Symbol is expected in FARTCOINUSDT / SOLUSDT / WIFUSDT format.
    The cmc_symbol (FARTCOIN / SOL / WIF) is derived by stripping trailing USDT.
    """
    cmc = symbol.replace("USDT", "") if symbol.endswith("USDT") else symbol
    base_url   = f"https://public.bybit.com/trading/{symbol}/"
    ohlcv_path = DATA_DIR / f"{cmc}_ohlcv_hourly.csv"
    taker_path = DATA_DIR / f"{symbol}_taker.csv"
    oi_path    = DATA_DIR / f"{symbol}_oi.csv"
    lsr_path   = DATA_DIR / f"{symbol}_lsr.csv"
    return base_url, ohlcv_path, taker_path, oi_path, lsr_path


def _download_day(date: datetime.date, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Download one day of tick data and return as DataFrame."""
    base_url, *_ = _get_paths(symbol)
    fname = f"{symbol}{date.isoformat()}.csv.gz"
    url   = base_url + fname
    resp  = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise FileNotFoundError(f"HTTP {resp.status_code}: {url}")

    rows = []
    with gzip.open(io.BytesIO(resp.content), "rt") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "ts":    float(row["timestamp"]),
                "price": float(row["price"]),
                "size":  float(row["size"]),
                "side":  row["side"],
            })
    return pd.DataFrame(rows)


def _build_hourly(ticks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert tick DataFrame to hourly OHLCV + BSR DataFrames."""
    ticks["dt"] = pd.to_datetime(ticks["ts"], unit="s", utc=True).dt.floor("1h")

    ohlcv = ticks.groupby("dt").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
    ).sort_index()

    buy_vol   = ticks[ticks["side"] == "Buy"].groupby("dt")["size"].sum()
    total_vol = ticks.groupby("dt")["size"].sum()
    bsr = (buy_vol / total_vol.clip(lower=1e-12)).reindex(ohlcv.index).fillna(0.5).clip(0, 1)

    ohlcv.index = ohlcv.index.tz_convert("UTC").tz_localize(None)
    ohlcv.index.name = "timestamp"
    ohlcv["market_cap"] = 0.0

    taker = bsr.to_frame("buySellRatio")
    taker.index = taker.index.tz_convert("UTC").tz_localize(None)
    taker.index.name = "timestamp"

    return ohlcv, taker


def _append(new_ohlcv: pd.DataFrame, new_taker: pd.DataFrame,
            symbol: str = DEFAULT_SYMBOL) -> int:
    """Append new rows to existing CSVs. Returns number of new rows added."""
    _, ohlcv_path, taker_path, _, _ = _get_paths(symbol)
    added = 0
    for path, new_df in [(ohlcv_path, new_ohlcv), (taker_path, new_taker)]:
        if path.exists():
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_df
        combined.to_csv(path)
        if path == ohlcv_path:
            added = len(new_df)
    return added


def _fetch_oi_page(end_time_ms: int, limit: int = 200,
                   symbol: str = DEFAULT_SYMBOL) -> list[dict]:
    """Fetch one page of hourly OI from Bybit V5 REST. Returns list of dicts."""
    params = {
        "category":     "linear",
        "symbol":       symbol,
        "intervalTime": "1h",
        "limit":        limit,
        "endTime":      end_time_ms,
    }
    r = requests.get(f"{BYBIT_API}/v5/market/open-interest", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit OI error: {data}")
    return data["result"]["list"]


def _fetch_lsr_page(end_time_ms: int, limit: int = 200,
                    symbol: str = DEFAULT_SYMBOL) -> list[dict]:
    """Fetch one page of hourly LSR from Bybit V5 REST. Returns list of dicts."""
    params = {
        "category": "linear",
        "symbol":   symbol,
        "period":   "1h",
        "limit":    limit,
        "endTime":  end_time_ms,
    }
    r = requests.get(f"{BYBIT_API}/v5/market/account-ratio", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit LSR error: {data}")
    return data["result"]["list"]


def _page_to_df(rows: list[dict], value_key: str, col_name: str) -> pd.DataFrame:
    """Convert Bybit API page rows to a timestamped DataFrame."""
    records = []
    for row in rows:
        ts_ms = int(row["timestamp"])
        val   = float(row[value_key])
        records.append({"timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"), col_name: val})
    df = pd.DataFrame(records).set_index("timestamp")
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df.sort_index()


def _append_csv(path, new_df: pd.DataFrame) -> int:
    """Append new_df rows to existing CSV; deduplicate and sort. Returns rows added."""
    if path.exists():
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df
    combined.to_csv(path)
    return len(new_df)


def refresh_oi_lsr(hours: int = 48,
                   symbol: str = DEFAULT_SYMBOL) -> tuple[int, int]:
    """
    Pull the last `hours` of OI and LSR from Bybit REST and append to CSVs.
    Returns (oi_rows_added, lsr_rows_added).
    """
    _, _, _, oi_path, lsr_path = _get_paths(symbol)
    now_ms   = int(time.time() * 1000)
    # Bybit returns up to 200 rows per page; 48h fits in one page
    limit    = min(hours, 200)

    oi_rows  = _fetch_oi_page(now_ms, limit=limit, symbol=symbol)
    lsr_rows = _fetch_lsr_page(now_ms, limit=limit, symbol=symbol)

    oi_df  = _page_to_df(oi_rows,  "openInterest", "sumOpenInterestValue")
    lsr_df = _page_to_df(lsr_rows, "buyRatio",     "longShortRatio")

    oi_added  = _append_csv(oi_path,  oi_df)
    lsr_added = _append_csv(lsr_path, lsr_df)

    print(f"  Bybit OI refresh ({symbol}):  +{oi_added} rows")
    print(f"  Bybit LSR refresh ({symbol}): +{lsr_added} rows")
    return oi_added, lsr_added


def ingest_yesterday(symbol: str = DEFAULT_SYMBOL) -> int:
    """Download yesterday's Bybit tick file + refresh OI/LSR, append to pipeline CSVs."""
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    ticks = _download_day(yesterday, symbol=symbol)
    ohlcv, taker = _build_hourly(ticks)
    added = _append(ohlcv, taker, symbol=symbol)
    print(f"  Bybit ingest {yesterday} ({symbol}): +{added} hourly bars")

    try:
        refresh_oi_lsr(hours=48, symbol=symbol)
    except Exception as e:
        print(f"  Bybit OI/LSR refresh: SKIP ({e})")

    return added


def ingest_range(start: datetime.date, end: datetime.date,
                 symbol: str = DEFAULT_SYMBOL) -> int:
    """Bulk download date range. Returns total hourly bars added."""
    base_url, *_ = _get_paths(symbol)
    r = requests.get(base_url, timeout=10)
    available = set(re.findall(rf"{re.escape(symbol)}(\d{{4}}-\d{{2}}-\d{{2}})\.csv\.gz", r.text))

    all_ticks = []
    d = start
    while d <= end:
        if d.isoformat() in available:
            try:
                all_ticks.append(_download_day(d, symbol=symbol))
            except Exception as e:
                print(f"  SKIP {d}: {e}")
        d += datetime.timedelta(days=1)

    if not all_ticks:
        return 0

    ticks = pd.concat(all_ticks, ignore_index=True)
    ohlcv, taker = _build_hourly(ticks)
    added = _append(ohlcv, taker, symbol=symbol)
    print(f"  Bybit ingest {start}→{end} ({symbol}): +{added} hourly bars")
    return added


if __name__ == "__main__":
    import sys
    _sym = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_SYMBOL
    if len(sys.argv) >= 3:
        s = datetime.date.fromisoformat(sys.argv[1])
        e = datetime.date.fromisoformat(sys.argv[2])
        ingest_range(s, e, symbol=_sym)
    else:
        ingest_yesterday(symbol=_sym)
