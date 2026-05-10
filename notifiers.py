"""
notifiers.py — Fartcoin Alpha Push Notifications

Sends Telegram messages when a signal tier upgrades.
Only fires when tier rank increases (no repeat spam on same tier).

Required env vars:
  TELEGRAM_BOT_TOKEN   — bot token from @BotFather
  TELEGRAM_CHAT_ID     — target chat / channel ID

Fails silently if env vars are missing.
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── Tier rank ──────────────────────────────────────────────────────────────────
_TIER_RANK = {
    "NO TRADE":         0,
    "WATCH":            1,
    "TRADE":            2,
    "HIGH CONVICTION":  3,
    "FULL SEND":        4,
}

# Only alert on these tiers
_ALERT_TIERS = {"TRADE", "HIGH CONVICTION", "FULL SEND"}

_TIER_EMOJI = {
    "TRADE":            "🟡",
    "HIGH CONVICTION":  "🔵",
    "FULL SEND":        "🟢",
}

_DIR_EMOJI = {
    "LONG":  "📈",
    "SHORT": "📉",
}


# ── State persistence ──────────────────────────────────────────────────────────
def _load_prev_tier() -> str:
    """Return previous tier label from data/prev_tier.json, or 'NO TRADE'."""
    path = DATA_DIR / "prev_tier.json"
    try:
        if path.exists():
            return json.loads(path.read_text()).get("tier", "NO TRADE")
    except Exception:
        pass
    return "NO TRADE"


def _save_prev_tier(tier: str) -> None:
    """Persist current tier to data/prev_tier.json."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "prev_tier.json").write_text(
            json.dumps({"tier": tier, "updated": datetime.now(timezone.utc).isoformat()})
        )
    except Exception:
        pass


# ── Telegram sender ────────────────────────────────────────────────────────────
def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    """POST text message to Telegram Bot API. Returns True on success."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Message builder ────────────────────────────────────────────────────────────
def _build_message(output: dict, prev_tier: str) -> str:
    """Format a concise Telegram alert from pipeline output."""
    ms   = output.get("market_state", {})
    proj = output.get("projections", {})
    opp  = proj.get("opportunity", {})
    hmm  = proj.get("hmm_regime", {})

    tier      = opp.get("tier", "")
    direction = ms.get("direction", "LONG")
    score     = opp.get("score", 0)
    price     = ms.get("fart_price", 0)
    btc_price = ms.get("btc_price", 0)
    composite = ms.get("composite", 0)
    p4        = opp.get("p4h", 0)
    p8        = opp.get("p8h", 0)
    size_pct  = opp.get("size_pct", 0)
    kelly     = opp.get("kelly_fraction", 0)
    funding   = ms.get("avg_funding", 0)
    session   = ms.get("session", "")
    conviction= ms.get("conviction", "")

    # Support / resistance / R:R
    sr_data  = proj.get("support_resistance", {})
    support  = sr_data.get("nearest_support", None)
    resist   = sr_data.get("nearest_resistance", None)
    rr       = opp.get("rr_ratio", None)

    # HMM regime
    regime   = hmm.get("regime_label", "")
    hmm_conf = hmm.get("confidence", 0)

    t_emoji = _TIER_EMOJI.get(tier, "⚪")
    d_emoji = _DIR_EMOJI.get(direction, "")
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

    triple  = opp.get("triple_agreement", 0)
    lstm_p  = opp.get("lstm_prob")

    lines = [
        f"{t_emoji} <b>FARTCOIN SIGNAL — {tier}</b>",
        f"{d_emoji} Direction: <b>{direction}</b> ({conviction})",
        f"",
        f"📊 Score: <b>{score}/100</b>  |  Composite: {composite:+.4f}",
        f"💰 Entry: <b>${price:.6f}</b>  (BTC ${btc_price:,.0f})",
        f"📐 Size: <b>{size_pct}%</b>  Kelly: {kelly:.1%}",
        f"",
        f"🎯 P(4h): {p4:.1%}  |  P(8h): {p8:.1%}",
        (f"🟢 <b>TRIPLE AGREEMENT</b>  |  LSTM p={lstm_p:.0%}  (97.7% hist. hit rate)"
         if triple and lstm_p is not None else ""),
    ]

    if support is not None and resist is not None:
        lines.append(f"🛡 Support: ${support:.6f}  |  Resist: ${resist:.6f}")
    if rr is not None:
        lines.append(f"⚖️  R:R  {rr:.2f}:1")

    funding_pct = funding * 100
    lines += [
        f"",
        f"💸 Funding: {funding_pct:+.4f}%/hr",
        f"🧠 HMM: {regime} ({hmm_conf:.0%})" if regime else "",
        f"🕐 Session: {session}  |  {now_str}",
    ]

    # Alerts summary
    alerts = output.get("alerts", [])
    if alerts:
        top = alerts[0]
        alert_msg = top.get("message", "") if isinstance(top, dict) else str(top)
        lines.append(f"⚠️  Alert: {alert_msg[:80]}")

    # Footer
    lines += [
        f"",
        f"<i>Prev tier: {prev_tier} → {tier}</i>",
    ]

    return "\n".join(l for l in lines if l is not None)


# ── Main entry point ───────────────────────────────────────────────────────────
def notify_telegram(output: dict) -> None:
    """
    Send a Telegram alert if the signal tier has upgraded.

    Reads previous tier from data/prev_tier.json.
    Only fires when:
      (a) current tier is TRADE / HIGH CONVICTION / FULL SEND, AND
      (b) tier rank is strictly higher than the previous run.

    Fails silently if env vars are not set.
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    proj = output.get("projections", {})
    opp  = proj.get("opportunity", {})
    tier = opp.get("tier", "NO TRADE")

    # Persist new tier regardless of whether we send
    prev_tier = _load_prev_tier()
    _save_prev_tier(tier)

    # Gate 1: must be an alert-worthy tier
    if tier not in _ALERT_TIERS:
        return

    # Gate 2: must be an upgrade (not a repeat or downgrade)
    curr_rank = _TIER_RANK.get(tier, 0)
    prev_rank = _TIER_RANK.get(prev_tier, 0)
    if curr_rank <= prev_rank:
        return

    # Gate 3: must have credentials
    if not token or not chat_id:
        return

    msg = _build_message(output, prev_tier)
    ok  = _send_telegram(token, chat_id, msg)

    # Also send any triggered alerts (up to 3 extra)
    if ok:
        alerts = output.get("alerts", [])
        for alert in alerts[:3]:
            if isinstance(alert, dict):
                detail = alert.get("message", "")
                severity = alert.get("severity", "")
                if detail:
                    _send_telegram(token, chat_id,
                                   f"⚡ <b>Alert [{severity}]:</b> {detail}")
