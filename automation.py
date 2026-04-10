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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import poll_once, collect_all
from signal_engine import load_data, compute_all_signals
from market_state import compute_market_state, determine_action
from alerts import evaluate_alerts, evaluate_projection_alerts
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


def main():
    parser = argparse.ArgumentParser(description="Fartcoin Alpha Automation Pipeline")
    parser.add_argument("--mode", choices=["light", "full"], default="light",
                        help="light = poll_once + signals; full = collect_all + signals")
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
        else:
            collect_light_external(coin=args.coin)

    output = run_pipeline(args.mode, coin=args.coin)

    # Print JSON to stdout (for scheduled task to parse)
    print(json.dumps(output, indent=2, default=str))

    # Summary to stderr
    n_alerts = len(output["alerts"])
    ms = output["market_state"]
    print(f"\n--- Pipeline Complete ---", file=sys.stderr)
    print(f"Mode: {args.mode}", file=sys.stderr)
    print(f"Signal: {ms['direction']} ({ms['conviction']}, composite={ms['composite']:+.4f})", file=sys.stderr)
    print(f"Session: {ms['session']}", file=sys.stderr)
    print(f"Alerts triggered: {n_alerts}", file=sys.stderr)


if __name__ == "__main__":
    main()
