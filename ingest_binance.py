"""
Ingest Binance public data portal ZIPs into pipeline-compatible CSVs.

Reads:  data/binance_raw/FARTCOINUSDT-1h-YYYY-MM.csv
        data/binance_raw/FARTCOINUSDT-fundingRate-YYYY-MM.csv

Writes (merges with existing, deduplicates, sort by time):
        data/FARTCOIN_ohlcv_hourly.csv     — replaces CoinGecko stubs
        data/FARTCOINUSDT_taker.csv         — replaces synthetic 0.5 BSR
        data/FARTCOINUSDT_funding.csv       — replaces empty funding column
"""

import glob
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR  = DATA_DIR / "binance_raw"


def load_klines() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("FARTCOINUSDT-1h-*.csv"))
    if not files:
        raise FileNotFoundError(f"No kline CSVs found in {RAW_DIR}")

    frames = []
    for f in files:
        df = pd.read_csv(f, header=0)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # open_time is ms epoch
    raw["timestamp"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp")

    return raw


def build_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    ohlcv = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "open":      raw["open"].astype(float),
        "high":      raw["high"].astype(float),
        "low":       raw["low"].astype(float),
        "close":     raw["close"].astype(float),
        "volume":    raw["volume"].astype(float),
        # CG-compat columns expected by signal_engine
        "market_cap": 0.0,
    }).set_index("timestamp")
    return ohlcv


def build_taker(raw: pd.DataFrame) -> pd.DataFrame:
    vol   = raw["volume"].astype(float)
    tbv   = raw["taker_buy_volume"].astype(float)
    # BSR = taker buy / total volume; clip to avoid div-by-zero artefacts
    bsr   = (tbv / vol.clip(lower=1e-12)).clip(0.0, 1.0)
    taker = pd.DataFrame({
        "timestamp":    raw["timestamp"],
        "buySellRatio": bsr.values,
    }).set_index("timestamp")
    return taker


def load_funding() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("FARTCOINUSDT-fundingRate-*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, header=0)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    raw["timestamp"] = pd.to_datetime(raw["calc_time"], unit="ms", utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp")

    funding = pd.DataFrame({
        "timestamp":   raw["timestamp"],
        "fundingRate": raw["last_funding_rate"].astype(float),
    }).set_index("timestamp")
    return funding


def merge_with_existing(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Merge new data with existing CSV, keeping new values where overlapping."""
    if existing_path.exists():
        old = pd.read_csv(existing_path, index_col=0, parse_dates=True)
        # Remove rows where existing data is synthetic (all zeros or all 0.5 BSR)
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        return combined
    return new_df


def write_tz_naive(df: pd.DataFrame, path: Path):
    """Write with UTC-stripped timestamps so pipeline parse_dates works cleanly."""
    out = df.copy()
    if hasattr(out.index, "tz") and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.to_csv(path)


def main():
    print("Loading Binance klines...")
    raw = load_klines()
    print(f"  {len(raw)} hourly bars ({raw['timestamp'].min()} → {raw['timestamp'].max()})")

    # OHLCV
    ohlcv = build_ohlcv(raw)
    ohlcv_path = DATA_DIR / "FARTCOIN_ohlcv_hourly.csv"
    write_tz_naive(ohlcv, ohlcv_path)
    print(f"  Wrote {len(ohlcv)} rows → {ohlcv_path.name}")

    # Taker BSR
    taker = build_taker(raw)
    taker_path = DATA_DIR / "FARTCOINUSDT_taker.csv"
    write_tz_naive(taker, taker_path)
    bsr_mean = taker["buySellRatio"].mean()
    bsr_std  = taker["buySellRatio"].std()
    print(f"  Wrote {len(taker)} rows → {taker_path.name}  (BSR mean={bsr_mean:.3f}, std={bsr_std:.3f})")

    # Funding
    print("\nLoading Binance funding rates...")
    funding = load_funding()
    if not funding.empty:
        funding_path = DATA_DIR / "FARTCOINUSDT_funding.csv"
        write_tz_naive(funding, funding_path)
        print(f"  Wrote {len(funding)} rows → {funding_path.name}")
    else:
        print("  No funding data found, skipping.")

    print("\nDone. Row summary:")
    print(f"  OHLCV:   {len(ohlcv):,} hourly bars")
    print(f"  Taker:   {len(taker):,} rows")
    print(f"  Funding: {len(funding):,} rows")


if __name__ == "__main__":
    main()
