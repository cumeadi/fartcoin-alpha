"""
Alert Engine — Fartcoin Alpha Framework

Evaluates alert rules against current market state and manages cooldowns
to prevent duplicate notifications.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "alert_state.json"


def _load_cooldowns():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_cooldowns(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _is_cooled_down(state, rule_name, cooldown_hours):
    last_fired = state.get(rule_name)
    if last_fired is None:
        return True
    last_dt = datetime.fromisoformat(last_fired)
    return datetime.now(timezone.utc) - last_dt > timedelta(hours=cooldown_hours)


def _fire(state, rule_name):
    state[rule_name] = datetime.now(timezone.utc).isoformat()


def evaluate_alerts(market_state, action):
    """
    Evaluate all alert rules against current market state.

    Returns list of alert dicts: {type, severity, title, message}
    """
    state = _load_cooldowns()
    alerts = []

    composite = market_state.get("composite", 0)
    signals = market_state.get("signals", {})
    avg_funding = market_state.get("avg_funding", 0)
    session = market_state.get("session", "")
    session_info = market_state.get("session_info", {})
    fart_price = market_state.get("fart_price", 0)
    btc_price = market_state.get("btc_price", 0)
    btc_regime = market_state.get("btc_regime", "Unknown")
    btc_ret_24h = market_state.get("btc_ret_24h", 0)

    direction = action.get("direction", "FLAT")
    conviction = action.get("conviction", "N/A")
    timing = action.get("timing", "")
    exit_plan = action.get("exit_plan", "")

    # --- Rule 1: Signal Entry (composite crosses threshold) ---
    if abs(composite) > 0.4 and _is_cooled_down(state, "signal_entry", 2):
        sig_parts = " | ".join(f"{k.replace('sig_', '').title()} {v:+.2f}" for k, v in signals.items())
        msg = (
            f"*FARTCOIN SIGNAL ALERT*\n\n"
            f"*Direction:* {direction}\n"
            f"*Conviction:* {conviction} (composite: {composite:+.2f})\n\n"
            f"*Session:* {session} ({session_info.get('et', '')})\n"
            f"*Timing:* {timing}\n"
            f"*BTC:* ${btc_price:,.0f} ({btc_ret_24h:+.1%} 24h) — {btc_regime}\n\n"
            f"*Entry:* ${fart_price:.4f}\n"
            f"*Exit:* {exit_plan}\n\n"
            f"*Signals:* {sig_parts}"
        )
        alerts.append({
            "type": "signal_entry",
            "severity": "high",
            "title": f"FARTCOIN {direction} — {conviction} conviction",
            "message": msg,
        })
        _fire(state, "signal_entry")

    # --- Rule 2: Funding Extreme ---
    if abs(avg_funding) > 0.01 and _is_cooled_down(state, "funding_extreme", 4):
        if avg_funding > 0:
            bias = "Longs crowded — contrarian bearish pressure"
        else:
            bias = "Shorts crowded — contrarian bullish pressure"
        msg = (
            f"*FARTCOIN FUNDING ALERT*\n\n"
            f"*Avg Funding Rate:* {avg_funding:.4f}\n"
            f"*Implication:* {bias}\n"
            f"*Session:* {session}\n"
            f"*Composite:* {composite:+.2f}"
        )
        alerts.append({
            "type": "funding_extreme",
            "severity": "medium",
            "title": f"Funding extreme: {avg_funding:.4f}",
            "message": msg,
        })
        _fire(state, "funding_extreme")

    # --- Rule 3: Manipulation Phase (OI accel spike + volume spike) ---
    oi_accel = signals.get("sig_oi_accel", 0)
    vol_spike = signals.get("sig_volume_spike", 0)
    if oi_accel > 0.6 and vol_spike > 0.4 and _is_cooled_down(state, "manipulation_phase", 2):
        msg = (
            f"*FARTCOIN MANIPULATION DETECTED*\n\n"
            f"*OI Acceleration:* {oi_accel:+.2f} (spike)\n"
            f"*Volume Spike:* {vol_spike:+.2f}\n"
            f"*Composite:* {composite:+.2f} ({direction})\n"
            f"*Session:* {session}\n\n"
            f"Significant position building detected with abnormal volume. "
            f"Manufactured move likely within 2-4 hours."
        )
        alerts.append({
            "type": "manipulation_phase",
            "severity": "high",
            "title": "Manipulation phase detected",
            "message": msg,
        })
        _fire(state, "manipulation_phase")

    _save_cooldowns(state)
    return alerts


def evaluate_projection_alerts(projections, market_state):
    """
    Evaluate projection-based alert rules.

    Args:
        projections: dict from projections.compute_projections()
        market_state: dict from market_state.compute_market_state()

    Returns list of alert dicts: {type, severity, title, message}
    """
    state = _load_cooldowns()
    alerts = []

    session = market_state.get("session", "")

    # --- Rule P1: High-probability directional projection ---
    prob_data = projections.get("probability", {})
    prob = prob_data.get("prob_positive_4h", 0.5)
    if (prob > 0.65 or prob < 0.35) and _is_cooled_down(state, "proj_high_prob", 4):
        direction = "BULLISH" if prob > 0.65 else "BEARISH"
        msg = (
            f"*FARTCOIN PROJECTION ALERT*\n\n"
            f"*Direction:* {direction}\n"
            f"*Probability:* {prob:.0%} chance of positive 4h return\n"
            f"*Expected Move:* {prob_data.get('expected_move_pct', 0):+.2f}%\n"
            f"*Session:* {session}\n\n"
            f"{prob_data.get('description', '')}"
        )
        alerts.append({
            "type": "proj_high_prob",
            "severity": "high",
            "title": f"Projection: {direction} ({prob:.0%})",
            "message": msg,
        })
        _fire(state, "proj_high_prob")

    # --- Rule P2: Funding reversion imminent ---
    mr_data = projections.get("mean_reversion", {})
    fr_data = mr_data.get("funding")
    if fr_data and _is_cooled_down(state, "proj_funding_reversion", 6):
        real_funding = abs(fr_data.get("current_real", 0))
        cross_time = fr_data.get("projected_cross_time_h")
        if real_funding > 0.15 and cross_time is not None and cross_time < 8:
            msg = (
                f"*FARTCOIN FUNDING REVERSION ALERT*\n\n"
                f"*Real Funding:* {fr_data['current_real']:.4f}\n"
                f"*Half-life:* {fr_data['half_life_h']:.1f}h\n"
                f"*Projected neutral cross:* {cross_time}h\n\n"
                f"{fr_data.get('description', '')}"
            )
            alerts.append({
                "type": "proj_funding_reversion",
                "severity": "medium",
                "title": f"Funding reversion in ~{cross_time}h",
                "message": msg,
            })
            _fire(state, "proj_funding_reversion")

    # --- Rule P3: Manipulation cycle phase change ---
    cycle = projections.get("manipulation_cycle", {})
    phase = cycle.get("phase", "DORMANT")
    if phase in ("BUILDUP", "SPIKE_IN_PROGRESS") and _is_cooled_down(state, "proj_manipulation_cycle", 3):
        est = cycle.get("est_hours_to_move")
        msg = (
            f"*FARTCOIN MANIPULATION CYCLE ALERT*\n\n"
            f"*Phase:* {phase}\n"
            f"*Confidence:* {cycle.get('confidence', 0):.0%}\n"
            f"{'*Est. time to move:* ' + str(est) + 'h' if est is not None else ''}\n\n"
            f"{cycle.get('description', '')}"
        )
        alerts.append({
            "type": "proj_manipulation_cycle",
            "severity": "high",
            "title": f"Manipulation: {phase}",
            "message": msg,
        })
        _fire(state, "proj_manipulation_cycle")

    # --- Rule P4: BTC lead-lag significant move ---
    btc_data = projections.get("btc_lead_lag", {})
    btc_2h = abs(btc_data.get("btc_2h_return_pct", 0))
    btc_conf = btc_data.get("confidence", 0)
    if btc_2h > 2 and btc_conf > 0.5 and _is_cooled_down(state, "proj_btc_lead_lag", 3):
        msg = (
            f"*FARTCOIN BTC LEAD-LAG ALERT*\n\n"
            f"*BTC 2h Move:* {btc_data.get('btc_2h_return_pct', 0):+.1f}%\n"
            f"*Projected FART Response:* {btc_data.get('projected_fart_move_pct', 0):+.1f}%\n"
            f"*Beta:* {btc_data.get('beta', 0):.1f}x\n"
            f"*Correlation:* {btc_data.get('rolling_corr_24h', 0):.2f}\n\n"
            f"{btc_data.get('description', '')}"
        )
        alerts.append({
            "type": "proj_btc_lead_lag",
            "severity": "medium",
            "title": f"BTC moved {btc_data.get('btc_2h_return_pct', 0):+.1f}% — FART response projected",
            "message": msg,
        })
        _fire(state, "proj_btc_lead_lag")

    # --- Rule P5: News sentiment danger (divergence or extreme assessment) ---
    news = projections.get("news_sentiment", {})
    news_divergence = news.get("divergence", "NONE")
    news_dangerous = news_divergence in ("PUMP_IN_FEAR", "VOLUME_PUMP", "DUMP_IN_OPTIMISM") or \
                     news.get("assessment") in ("DANGER", "CAUTION")
    if news.get("available") and news_dangerous and _is_cooled_down(state, "proj_news_danger", 4):
        msg = (
            f"*FARTCOIN NEWS SENTIMENT ALERT*\n\n"
            f"*Assessment:* {news['assessment']}\n"
            f"*News Buzz:* {news.get('news_buzz', 0):.1f}σ above normal\n"
            f"*Sentiment:* {news.get('current_sentiment', 0):.2f}\n"
            f"*Session:* {session}\n\n"
            f"{news.get('description', '')}"
        )
        alerts.append({
            "type": "proj_news_danger",
            "severity": "high",
            "title": "News: high buzz + negative sentiment",
            "message": msg,
        })
        _fire(state, "proj_news_danger")

    # --- Rule P6: Whale activity / exchange flow ---
    onchain = projections.get("onchain_flow", {})
    if onchain.get("available"):
        assessment = onchain.get("assessment", "NEUTRAL")
        if assessment in ("WHALE_DUMPING", "EXCHANGE_INFLOW") and \
                _is_cooled_down(state, "proj_whale_dump", 4):
            msg = (
                f"*FARTCOIN ON-CHAIN ALERT*\n\n"
                f"*Assessment:* {assessment}\n"
                f"*Net Flow:* {onchain.get('net_flow_tokens', 0):+,.0f} tokens\n"
                f"*Whale Transfers:* {onchain.get('whale_transfers', 0)}\n\n"
                f"{onchain.get('description', '')}"
            )
            alerts.append({
                "type": "proj_whale_dump",
                "severity": "high",
                "title": f"On-Chain: {assessment}",
                "message": msg,
            })
            _fire(state, "proj_whale_dump")

        elif assessment == "WHALE_ACCUMULATING" and \
                _is_cooled_down(state, "proj_whale_accumulate", 6):
            msg = (
                f"*FARTCOIN ON-CHAIN ALERT*\n\n"
                f"*Assessment:* WHALE ACCUMULATING\n"
                f"*Net Flow:* {onchain.get('net_flow_tokens', 0):+,.0f} tokens (withdrawals)\n"
                f"*Gini Trend:* {onchain.get('gini_trend', 'N/A')}\n\n"
                f"{onchain.get('description', '')}"
            )
            alerts.append({
                "type": "proj_whale_accumulate",
                "severity": "medium",
                "title": "On-Chain: Whale accumulation detected",
                "message": msg,
            })
            _fire(state, "proj_whale_accumulate")

    # --- Rule P7: Squeeze conditions (Coinalyze) ---
    # NOTE: Bybit FARTCOIN has a fixed +0.5%/8h funding floor.
    # SQUEEZE_BUILDING only fires when squeeze_risk == HIGH (>0.8% threshold),
    # meaning it is genuinely elevated ABOVE the floor, not just at it.
    cx = projections.get("cross_exchange", {})
    if cx.get("available"):
        cx_assessment = cx.get("assessment", "NORMAL")
        cx_squeeze    = cx.get("squeeze_risk", "LOW")
        # Only alert on SQUEEZE_BUILDING if risk is genuinely HIGH (above Bybit floor)
        squeeze_alert = (
            cx_assessment in ("SQUEEZE_IN_PROGRESS", "LIQUIDATION_CASCADE")
            or (cx_assessment == "SQUEEZE_BUILDING" and cx_squeeze == "HIGH")
        )
        if squeeze_alert and _is_cooled_down(state, "proj_squeeze", 3):
            pred_fund = cx.get("predicted_funding", 0)
            msg = (
                f"*FARTCOIN SQUEEZE / LIQUIDATION ALERT*\n\n"
                f"*Assessment:* {cx_assessment}\n"
                f"*Predicted Funding:* {pred_fund:.4%} "
                f"{'(above Bybit floor — genuine signal)' if pred_fund > 0.008 else '(near Bybit floor)'}\n"
                f"*Liq Z-Score:* {cx.get('liq_zscore', 0):.1f}σ\n"
                f"*Squeeze Risk:* {cx_squeeze}\n\n"
                f"{cx.get('description', '')}"
            )
            alerts.append({
                "type": "proj_squeeze",
                "severity": "high",
                "title": f"Cross-Exchange: {cx_assessment}",
                "message": msg,
            })
            _fire(state, "proj_squeeze")

    # --- Rule P8: Coinglass OI spike or settlement imminent ---
    cg = projections.get("coinglass_oi_funding", {})
    if cg.get("available"):
        cg_assessment = cg.get("assessment", "NORMAL")
        alert_conditions = (
            cg_assessment in (
                # Positive signals (opportunity alerts)
                "OI_PRICE_DIV_LONG", "PASSIVE_ACCUM", "EXTREME_SHORT_FUNDING",
                # Caution / bearish signals — Bybit-aware
                "OI_SPIKE_CAUTION", "OI_SURGE_CAUTION", "OI_TREND_CHASE_BEARISH",
                "EXTREME_LONG", "HIGH_LONG", "BINANCE_BEARISH_VS_BYBIT",
                "BOTH_CROWDED_LONG",
                # Structural alerts
                "SETTLEMENT_IMMINENT",
            )
        )
        if alert_conditions and _is_cooled_down(state, "proj_oi_spike", 2):
            severity = "high" if cg_assessment in (
                "OI_SPIKE_CAUTION", "OI_TREND_CHASE_BEARISH", "SETTLEMENT_IMMINENT",
                "EXTREME_LONG", "OI_PRICE_DIV_LONG", "BINANCE_BEARISH_VS_BYBIT",
            ) else "medium"
            m5 = cg.get("m5_oi_chg", 0)
            m15 = cg.get("m15_oi_chg", 0)
            mins_settle = cg.get("mins_to_settle", 999)
            mean_rate   = cg.get("mean_rate_pct", 0)
            bybit_rate  = cg.get("bybit_rate", 0)
            binance_rate = cg.get("binance_rate", 0)
            daily_carry = cg.get("bybit_daily_carry_pct", bybit_rate * 3)
            msg = (
                f"*FARTCOIN OI / FUNDING ALERT*\n\n"
                f"*Assessment:* {cg_assessment}\n"
                f"*OI Change:* 5m: {m5:+.1f}% / 15m: {m15:+.1f}%\n"
                f"*OI/Vol Ratio:* {cg.get('oi_vol_ratio', 0):.2f} "
                f"({cg.get('oi_direction', 'N/A')})\n"
                f"*Bybit Rate:* {bybit_rate:+.3f}%/8h (carry: {daily_carry:+.2f}%/day)\n"
                f"*Binance Rate:* {binance_rate:+.3f}% "
                f"({'↓ bearish vs Bybit' if binance_rate < bybit_rate - 0.3 else '≈ aligned'})\n"
            )
            if cg.get("settlement_imminent"):
                msg += f"*Settlement in:* {mins_settle:.0f} min ⚡\n"
            if cg.get("spread_pct", 0) > 1.0:
                msg += f"*Funding Spread:* {cg.get('spread_pct', 0):.3f}%\n"
            msg += f"\n{cg.get('description', '')}"
            alerts.append({
                "type": "proj_oi_spike",
                "severity": severity,
                "title": f"Coinglass OI/Funding: {cg_assessment}",
                "message": msg,
            })
            _fire(state, "proj_oi_spike")

    # --- Rule P9: Funding Settlement Cycle pre-settlement warning ---
    settlement = projections.get("funding_settlement", {})
    s_phase = settlement.get("phase", "MID_CYCLE")
    s_conf = settlement.get("confidence", 0)
    s_effect = settlement.get("expected_effect", "UNKNOWN")
    s_mins = settlement.get("mins_to_settlement", 999)
    if s_phase == "PRE_SETTLEMENT" and s_conf > 0.5 and s_mins <= 30 and \
            _is_cooled_down(state, "proj_settlement_cycle", 8):
        direction_word = "BEARISH pressure" if "DOWN" in s_effect else "BULLISH pressure" if "UP" in s_effect else "price action"
        msg = (
            f"*FARTCOIN SETTLEMENT CYCLE ALERT*\n\n"
            f"*Phase:* PRE-SETTLEMENT ({s_mins:.0f}min to next settlement)\n"
            f"*Funding:* {settlement.get('current_funding_sign','?')} → {direction_word}\n"
            f"*Historical pre-settlement avg:* {settlement.get('pre_ret_mean', 0):+.2f}%\n"
            f"*Post-settlement avg:* {settlement.get('post_ret_mean', 0):+.2f}%\n"
            f"*Confidence:* {s_conf:.0%} (n={settlement.get('historical_n',0)})\n\n"
            f"{settlement.get('description', '')}"
        )
        alerts.append({
            "type": "proj_settlement_cycle",
            "severity": "medium",
            "title": f"Settlement in {s_mins:.0f}min — {direction_word}",
            "message": msg,
        })
        _fire(state, "proj_settlement_cycle")

    # --- Rule P10: Liquidation Cascade entry window ---
    liq_cas = projections.get("liq_cascade", {})
    lc_state = liq_cas.get("state", "NORMAL")
    lc_conf = liq_cas.get("confidence", 0)
    if lc_state == "CASCADE_IN_PROGRESS" and _is_cooled_down(state, "proj_cascade_active", 2):
        msg = (
            f"*FARTCOIN LIQUIDATION CASCADE ALERT*\n\n"
            f"*State:* CASCADE IN PROGRESS — do NOT enter long\n"
            f"*Liq z-score:* {liq_cas.get('liq_zscore', 0):.1f}σ\n"
            f"*Wick ratio:* {liq_cas.get('wick_ratio', 0):.1f}x body\n"
            f"*Volume spike:* {liq_cas.get('volume_ratio', 0):.1f}x avg\n\n"
            f"Wait for wick candle to complete, then watch for post-cascade entry.\n"
            f"{liq_cas.get('description', '')}"
        )
        alerts.append({
            "type": "proj_cascade_active",
            "severity": "high",
            "title": f"Liquidation cascade: {liq_cas.get('liq_zscore',0):.1f}σ",
            "message": msg,
        })
        _fire(state, "proj_cascade_active")

    elif lc_state == "POST_CASCADE_ENTRY" and lc_conf > 0.4 and \
            _is_cooled_down(state, "proj_cascade_entry", 4):
        avg_4h = liq_cas.get("post_cascade_avg_4h", 0)
        hit = liq_cas.get("post_cascade_hit_rate", 0)
        n = liq_cas.get("historical_n", 0)
        msg = (
            f"*FARTCOIN POST-CASCADE ENTRY ALERT*\n\n"
            f"*State:* POST-CASCADE ENTRY WINDOW\n"
            f"*Candles since cascade:* {liq_cas.get('candles_since_cascade', '?')}\n"
            f"*Historical 4h return:* {avg_4h:+.1f}% avg, {hit:.0%} hit rate (n={n})\n"
            f"*Confidence:* {lc_conf:.0%}\n\n"
            f"Forced sellers exhausted. High-probability long setup.\n"
            f"{liq_cas.get('description', '')}"
        )
        alerts.append({
            "type": "proj_cascade_entry",
            "severity": "high",
            "title": f"Post-cascade entry window ({hit:.0%} hist hit rate)",
            "message": msg,
        })
        _fire(state, "proj_cascade_entry")

    _save_cooldowns(state)
    return alerts
