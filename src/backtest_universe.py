from __future__ import annotations

import math
from typing import Any

from config.access import require_config_value, require_live_value
from utils import symbol_to_coin

POSITION_SIDES = ("long", "short")


def backtest_side_enabled(config: dict, pside: str) -> bool:
    bot_cfg = require_config_value(config, f"bot.{pside}")
    total_wallet_exposure_limit = float(bot_cfg["total_wallet_exposure_limit"])
    n_positions = int(round(float(bot_cfg["n_positions"])))
    return (
        math.isfinite(total_wallet_exposure_limit)
        and total_wallet_exposure_limit > 0.0
        and n_positions > 0
    )


def normalize_backtest_coin(coin: Any) -> str:
    return symbol_to_coin(str(coin), verbose=False)


def _normalize_coin_list(coins: Any) -> list[str]:
    normalized = []
    seen = set()
    for coin in coins or []:
        coin_key = normalize_backtest_coin(coin)
        if not coin_key or coin_key in seen:
            continue
        seen.add(coin_key)
        normalized.append(coin_key)
    return sorted(normalized)


def effective_backtest_approved_coins_by_side(config: dict) -> dict[str, list[str]]:
    approved = require_live_value(config, "approved_coins")
    return {
        pside: (
            _normalize_coin_list(approved.get(pside, []))
            if backtest_side_enabled(config, pside)
            else []
        )
        for pside in POSITION_SIDES
    }


def effective_backtest_data_coins(config: dict) -> list[str]:
    approved_by_side = effective_backtest_approved_coins_by_side(config)
    return sorted(set().union(*(set(approved_by_side[pside]) for pside in POSITION_SIDES)))
