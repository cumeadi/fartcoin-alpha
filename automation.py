"""
Automation Pipeline — Fartcoin Alpha Framework

CLI orchestrator for scheduled tasks. Two modes:
  --mode light   poll_once (derivatives only) + recompute signals + check alerts
  --mode full    collect_all (full data refresh) + recompute signals + check alerts

Outputs JSON to stdout for the scheduled task prompt to parse and forward to Slack.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import poll_once, collect_all
from signal_engine import load_data, compute_all_signals
from market_state import compute_market_state, determine_action
from alerts import evaluate_alerts, evaluate_projection_alerts, compute_atr_cooldown_multiplier
from projections import compute_projections
from coin_config import get_config, DEFAULT_COIN
try:
    from external_collectors import collect_light_external, collect_all_external
    HAS_EXTERNAL = True
except ImportError:
    HAS_EXTERNAL = False


DATA_DIR = Path(__file__).parent / "data"


def run_pipeline(mode="light", coin=DEFAULT_COIN):
    """Run the full automation pipeline."""
    cfg = get_config(coin)
    cmc_sym  = cfg["cmc_symbol"]
    perp_sym = cfg["perp_symbol"]
    cg_id    = cfg["cg_coin_id"]

    # Step 1: Collect data
    if mode == "full":
        print("Collecting all data...", file=sys.stderr)
        collect_all(cmc_symbol=cmc_sym, perp_symbol=perp_sym, cg_coin_id=cg_id)
    elif mode == "snapshot":
        print("Snapshot mode — skipping OHLCV refresh, using cached data...", file=sys.stderr)
        # No data collection — only external snapshot collectors ran above
    else:
        print("Light poll (derivatives snapshot)...", file=sys.stderr)
        poll_once(coin_filter=cmc_sym)

    # Step 2: Recompute signals
    print("Computing signals...", file=sys.stderr)
    data = load_data(perp_symbol=perp_sym, cmc_symbol=cmc_sym, cg_coin_id=cg_id)
    signals = compute_all_signals(data)
    signals_file = DATA_DIR / f"signals_{cmc_sym}.csv"
    if not signals.empty:
        signals.to_csv(signals_file)
        print(f"Signals saved: {len(signals)} rows", file=sys.stderr)

    # Step 3: Reload data with fresh signals
    data["signals"] = signals

    # Step 4: Compute market state + action
    mkt = compute_market_state(data)
    action = determine_action(mkt)

    # Step 4b: Calibrate vol-adjusted cooldown multiplier from ATR
    compute_atr_cooldown_multiplier(data.get("ohlcv"))

    # Step 5: Evaluate alerts
    triggered_alerts = evaluate_alerts(mkt, action)

    # Step 5b: Compute forward projections
    print("Computing projections...", file=sys.stderr)
    projections = compute_projections(data, mkt)

    # Step 5c: Evaluate projection alerts
    proj_alerts = evaluate_projection_alerts(projections, mkt)
    triggered_alerts.extend(proj_alerts)

    # Step 6: Build output
    output = {
        "mode": mode,
        "market_state": {
            "session": mkt.get("session"),
            "composite": round(mkt.get("composite", 0), 4),
            "direction": action["direction"],
            "conviction": action["conviction"],
            "timing": action["timing"],
            "btc_regime": mkt.get("btc_regime", "Unknown"),
            "btc_price": round(mkt.get("btc_price", 0), 2),
            "fart_price": round(mkt.get("fart_price", 0), 6),
            "avg_funding": round(mkt.get("avg_funding", 0), 6),
            "total_oi": round(mkt.get("total_oi", 0), 2) if "total_oi" in mkt else None,
            "risk_score": mkt.get("risk_score", 0),
            "session_info": mkt.get("session_info"),
        },
        "action": action,
        "projections": projections,
        "alerts": triggered_alerts,
    }

    return output


def _log_trade_journal(output, data_dir):
    """
    Append this pipeline run to the trade journal CSV.

    Columns:
      timestamp, mode, direction, conviction, composite, score, tier,
      kelly_fraction, size_pct, hmm_regime, fart_price, btc_price,
      avg_funding, session, alerts_n

    On the next run, also resolves the previous row's outcome:
      outcome_4h (actual 4h price change %), hit (1/0 vs carry cost)
    """
    import csv
    from datetime import datetime, timezone

    journal_path = data_dir / "trade_journal.csv"
    ms    = output["market_state"]
    proj  = output.get("projections", {})
    opp   = proj.get("opportunity", {})
    hmm   = proj.get("hmm_regime", {})

    now   = datetime.now(timezone.utc).isoformat()
    price = ms.get("fart_price", 0)

    row_data = {
        "timestamp":      now,
        "mode":           output.get("mode", ""),
        "direction":      ms.get("direction", ""),
        "conviction":     ms.get("conviction", ""),
        "composite":      ms.get("composite", 0),
        "score":          opp.get("score", 0),
        "tier":           opp.get("tier", ""),
        "kelly_fraction": opp.get("kelly_fraction", 0),
        "size_pct":       opp.get("size_pct", 0),
        "hmm_regime":     hmm.get("regime_label", ""),
        "hmm_conf":       hmm.get("confidence", 0),
        "fart_price":     price,
        "btc_price":      ms.get("btc_price", 0),
        "avg_funding":    ms.get("avg_funding", 0),
        "session":        ms.get("session", ""),
        "alerts_n":       len(output.get("alerts", [])),
        "outcome_4h":     "",   # filled in by next run
        "hit":            "",   # filled in by next run
    }

    # ── Resolve previous row's outcome ───────────────────────────────────────
    if journal_path.exists():
        try:
            prev_df = pd.read_csv(journal_path)
            if len(prev_df) > 0 and "fart_price" in prev_df.columns:
                last_row = prev_df.iloc[-1]
                if str(last_row.get("outcome_4h", "")).strip() == "" and float(last_row.get("fart_price", 0)) > 0:
                    last_price = float(last_row["fart_price"])
                    outcome    = (price - last_price) / last_price
                    hit        = 1 if outcome > 0.0045 else 0
                    prev_df.iloc[-1, prev_df.columns.get_loc("outcome_4h")] = round(outcome * 100, 4)
                    prev_df.iloc[-1, prev_df.columns.get_loc("hit")]        = hit
                    prev_df.to_csv(journal_path, index=False)
                    print(f"  [journal] Resolved last entry: outcome={outcome:+.3%}, hit={hit}", file=sys.stderr)
        except Exception as _e:
            print(f"  [journal] Outcome resolution failed: {_e}", file=sys.stderr)

    # ── Append new row ────────────────────────────────────────────────────────
    try:
        fieldnames = list(row_data.keys())
        write_header = not journal_path.exists()
        with open(journal_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"  [journal] Logged: {ms.get('direction')} @ ${price:.6f} | score={opp.get('score',0)} | {opp.get('tier','')}", file=sys.stderr)
    except Exception as _e:
        print(f"  [journal] Write failed: {_e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Fartcoin Alpha Automation Pipeline")
    parser.add_argument("--mode", choices=["light", "full", "snapshot"], default="light",
                        help="light = poll_once + signals; full = collect_all + signals; "
                             "snapshot = coinglass funding + OI only (fast, for Ghost Long data)")
    parser.add_argument("--external", action="store_true",
                        help="Also run external collectors (CryptoPanic, Helius, Coinalyze)")
    parser.add_argument("--coin", default=DEFAULT_COIN,
                        help=f"Coin to analyse (default: {DEFAULT_COIN})")
    args = parser.parse_args()

    # Optionally run external collectors first
    if args.external and HAS_EXTERNAL:
        print("Running external collectors...", file=sys.stderr)
        if args.mode == "full":
            collect_all_external(coin=args.coin)
        elif args.mode == "snapshot":
            # Lightweight: just funding + OI snapshots for Ghost Long accumulation
            from external_collectors import fetch_coinglass_funding_snapshot, fetch_coinglass_oi_snapshot
            print(f"  [snapshot] Collecting Coinglass funding + OI for {args.coin}...", file=sys.stderr)
            fetch_coinglass_funding_snapshot(coin=args.coin)
            fetch_coinglass_oi_snapshot(coin=args.coin)
        else:
            collect_light_external(coin=args.coin)

    output = run_pipeline(args.mode, coin=args.coin)

    # Log to trade journal (non-snapshot runs only — snapshot has no fresh price)
    if args.mode != "snapshot":
        _log_trade_journal(output, DATA_DIR)

    # Print JSON to stdout (for scheduled task to parse)
    print(json.dumps(output, indent=2, default=str))

    # Summary to stderr
    n_alerts = len(output["alerts"])
    ms = output["market_state"]
    opp = output.get("projections", {}).get("opportunity", {})
    print(f"\n--- Pipeline Complete ---", file=sys.stderr)
    print(f"Mode: {args.mode}", file=sys.stderr)
    print(f"Signal: {ms['direction']} ({ms['conviction']}, composite={ms['composite']:+.4f})", file=sys.stderr)
    print(f"Session: {ms['session']}", file=sys.stderr)
    print(f"Opportunity: score={opp.get('score',0)}/100 tier={opp.get('tier','')} kelly={opp.get('kelly_fraction',0):.1%} size={opp.get('size_pct',0)}%", file=sys.stderr)
    print(f"Alerts triggered: {n_alerts}", file=sys.stderr)


if __name__ == "__main__":
    main()
