"""
Coin configuration registry for the Fartcoin Alpha Framework.
Add new coins here; all other modules read from this dict.
"""

COIN_CONFIG = {
    "FARTCOIN": {
        "cmc_symbol":        "FARTCOIN",
        "perp_symbol":       "FARTCOINUSDT",
        "cg_coin_id":        "fartcoin",
        "cp_coin_id":        "fartcoin-fartcoin",
        "coinglass_ticker":  "FARTCOIN",
        "coinalyze_symbols": [
            "FARTCOINUSDT_PERP.A",  # Binance
            "FARTCOINUSDT.6",       # Bybit
            "FARTCOINUSDT_PERP.3",  # OKX
            "FARTCOINUSDT_PERP.4",  # Bitget
        ],
        "blockchain":        "solana",
        "display_name":      "FARTCOIN",
        "emoji":             "💨",
    },
    "ZEC": {
        "cmc_symbol":        "ZEC",
        "perp_symbol":       "ZECUSDT",
        "cg_coin_id":        "zcash",
        "cp_coin_id":        "zcash",
        "coinglass_ticker":  "ZEC",
        "coinalyze_symbols": [
            "ZECUSDT_PERP.A",  # Binance
            "ZECUSDT.6",       # Bybit
            "ZECUSDT_PERP.3",  # OKX
        ],
        "blockchain":        "zcash",  # Helius (Solana) skipped for non-solana coins
        "display_name":      "Zcash",
        "emoji":             "🛡",
    },
}

DEFAULT_COIN = "FARTCOIN"


def get_config(coin: str) -> dict:
    """Return config dict for coin, raising ValueError if unknown."""
    if coin not in COIN_CONFIG:
        raise ValueError(f"Unknown coin '{coin}'. Available: {list(COIN_CONFIG.keys())}")
    return COIN_CONFIG[coin]
