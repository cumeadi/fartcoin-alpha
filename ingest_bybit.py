"""
Bybit public tick data ingestion + Bybit V5 REST OI/LSR refresh.

Reads daily .csv.gz files from public.bybit.com/trading/FARTCOINUSDT/
Builds hourly OHLCV candles and real taker BSR, appends to pipeline CSVs.
Also fetches latest OI and LSR from Bybit V5 REST API and appends to their CSVs.

Functions
---------
ingest_yesterday()         — download yesterday's tick file + refresh OI/LSR (for cron)
ingest_range(start, end)   — bulk download a date range (for backfill)
refresh_oi_lsr(hours)      — pull the last N hours of OI + LSR and append
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

BASE_URL    = "https://public.bybit.com/trading/FARTCOINUSDT/"
BYBIT_API   = "https://api.bybit.com"
SYMBOL      = "FARTCOINUSDT"
DATA_DIR    = __import__("pathlib").Path(__file__).parent / "data"
OHLCV_PATH  = DATA_DIR / "FARTCOIN_ohlcv_hourly.csv"
TAKER_PATH  = DATA_DIR / "FARTCOINUSDT_taker.csv"
OI_PATH     = DATA_DIR / "FARTCOINUSDT_oi.csv"
LSR_PATH    = DATA_DIR / "FARTCOINUSDT_lsr.csv"


def _download_day(date: datetime.date) -> pd.DataFrame:
    """Download one day of tick data and return as DataFrame."""
    fname = f"FARTCOINUSDT{date.isoformat()}.csv.gz"
    url   = BASE_URL + fname
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


def _append(new_ohlcv: pd.DataFrame, new_taker: pd.DataFrame) -> int:
    """Append new rows to existing CSVs. Returns number of new rows added."""
    added = 0
    for path, new_df in [(OHLCV_PATH, new_ohlcv), (TAKER_PATH, new_taker)]:
        if path.exists():
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_df
        combined.to_csv(path)
        if path == OHLCV_PATH:
            added = len(new_df)
    return added


def _fetch_oi_page(end_time_ms: int, limit: int = 200) -> list[dict]:
    """Fetch one page of hourly OI from Bybit V5 REST. Returns list of dicts."""
    params = {
        "category":     "linear",
        "symbol":       SYMBOL,
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


def _fetch_lsr_page(end_time_ms: int, limit: int = 200) -> list[dict]:
    """Fetch one page of hourly LSR from Bybit V5 REST. Returns list of dicts."""
    params = {
        "category": "linear",
        "symbol":   SYMBOL,
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


def refresh_oi_lsr(hours: int = 48) -> tuple[int, int]:
    """
    Pull the last `hours` of OI and LSR from Bybit REST and append to CSVs.
    Returns (oi_rows_added, lsr_rows_added).
    """
    now_ms   = int(time.time() * 1000)
    # Bybit returns up to 200 rows per page; 48h fits in one page
    limit    = min(hours, 200)

    oi_rows  = _fetch_oi_page(now_ms, limit=limit)
    lsr_rows = _fetch_lsr_page(now_ms, limit=limit)

    oi_df  = _page_to_df(oi_rows,  "openInterest", "sumOpenInterestValue")
    lsr_df = _page_to_df(lsr_rows, "buyRatio",     "longShortRatio")

    oi_added  = _append_csv(OI_PATH,  oi_df)
    lsr_added = _append_csv(LSR_PATH, lsr_df)

    print(f"  Bybit OI refresh:  +{oi_added} rows")
    print(f"  Bybit LSR refresh: +{lsr_added} rows")
    return oi_added, lsr_added


def ingest_yesterday() -> int:
    """Download yesterday's Bybit tick file + refresh OI/LSR, append to pipeline CSVs."""
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    ticks = _download_day(yesterday)
    ohlcv, taker = _build_hourly(ticks)
    added = _append(ohlcv, taker)
    print(f"  Bybit ingest {yesterday}: +{added} hourly bars")

    try:
        refresh_oi_lsr(hours=48)
    except Exception as e:
        print(f"  Bybit OI/LSR refresh: SKIP ({e})")

    return added


def ingest_range(start: datetime.date, end: datetime.date) -> int:
    """Bulk download date range. Returns total hourly bars added."""
    r = requests.get(BASE_URL, timeout=10)
    available = set(re.findall(r"FARTCOINUSDT(\d{4}-\d{2}-\d{2})\.csv\.gz", r.text))

    all_ticks = []
    d = start
    while d <= end:
        if d.isoformat() in available:
            try:
                all_ticks.append(_download_day(d))
            except Exception as e:
                print(f"  SKIP {d}: {e}")
        d += datetime.timedelta(days=1)

    if not all_ticks:
        return 0

    ticks = pd.concat(all_ticks, ignore_index=True)
    ohlcv, taker = _build_hourly(ticks)
    added = _append(ohlcv, taker)
    print(f"  Bybit ingest {start}→{end}: +{added} hourly bars")
    return added


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        s = datetime.date.fromisoformat(sys.argv[1])
        e = datetime.date.fromisoformat(sys.argv[2])
        ingest_range(s, e)
    else:
        ingest_yesterday()
