from __future__ import annotations

import math
from typing import Any

from config.access import require_config_value, require_live_value
from config.shared_bot import require_grouped_bot_value
from utils import heuristic_symbol_to_coin, looks_like_exact_market_identifier


POSITION_SIDES = ("long", "short")


def _required_side_value(bot_cfg: dict, pside: str, key: str):
    return require_grouped_bot_value(bot_cfg, pside, key, prefer_flat=True)


def backtest_side_enabled(config: dict, pside: str) -> bool:
    bot_cfg = require_config_value(config, f"bot.{pside}")
    total_wallet_exposure_limit = float(
        _required_side_value(bot_cfg, pside, "total_wallet_exposure_limit")
    )
    n_positions = int(
        round(float(_required_side_value(bot_cfg, pside, "n_positions")))
    )
    return (
        math.isfinite(total_wallet_exposure_limit)
        and total_wallet_exposure_limit > 0.0
        and n_positions > 0
    )


def normalize_backtest_coin(coin: Any) -> str:
    raw = str(coin).strip()
    # Keep explicitly exchange-scoped identities and exact CCXT symbols
    # lossless.  Other unqualified inputs retain the established canonical
    # coin keys used by datasets, overrides, and Rust payloads.
    return raw if looks_like_exact_market_identifier(raw) else heuristic_symbol_to_coin(raw)


def _normalize_coin_list(coins: Any) -> list[str]:
    if not isinstance(coins, (list, tuple, set)):
        raise TypeError("backtest approved coin sides must be explicit list/tuple/set values")
    normalized = []
    seen = set()
    for coin in coins:
        coin_key = normalize_backtest_coin(coin)
        if not coin_key or coin_key in seen:
            continue
        seen.add(coin_key)
        normalized.append(coin_key)
    return sorted(normalized)


def effective_backtest_approved_coins_by_side(config: dict) -> dict[str, list[str]]:
    approved = require_live_value(config, "approved_coins")
    if not isinstance(approved, dict):
        raise TypeError("live.approved_coins must be a normalized per-side mapping for backtest")
    missing = [pside for pside in POSITION_SIDES if pside not in approved]
    if missing:
        missing_paths = ", ".join(f"live.approved_coins.{pside}" for pside in missing)
        raise KeyError(f"missing required {missing_paths}")
    return {
        pside: (
            _normalize_coin_list(approved[pside])
            if backtest_side_enabled(config, pside)
            else []
        )
        for pside in POSITION_SIDES
    }


def effective_backtest_data_coins(config: dict) -> list[str]:
    approved_by_side = effective_backtest_approved_coins_by_side(config)
    return sorted(set().union(*(set(approved_by_side[pside]) for pside in POSITION_SIDES)))
