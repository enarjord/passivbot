"""Efficient bot_params generation without deepcopy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotParamsTemplate:
    """Template for efficiently generating bot_params_list without deepcopy.

    Instead of deepcopying the entire config for each trial, this class:
    1. Stores base values for all BotParams fields (just floats)
    2. Merges sampled params via dict operations (no deepcopy)
    3. Clones per coin with overrides (shallow copy of flat dicts)
    """

    # Base values for long/short (flat dicts of floats)
    base_long: dict[str, Any]
    base_short: dict[str, Any]

    # Coin list for this exchange
    coins: list[str]

    # Coin overrides: coin -> pside -> field -> value
    coin_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict, coins: list[str]) -> "BotParamsTemplate":
        """Create template from config.

        Args:
            config: Full config dict with 'bot' section
            coins: List of coins for this exchange

        Returns:
            BotParamsTemplate ready to generate bot_params_list
        """
        bot = config.get("bot", {})

        # Extract base long/short params (shallow copy - they're flat dicts of floats)
        base_long = dict(bot.get("long", {}))
        base_short = dict(bot.get("short", {}))

        # Extract coin overrides
        coin_overrides = {}
        for coin, overrides in config.get("coin_overrides", {}).items():
            if coin not in coins:
                continue
            bot_overrides = overrides.get("bot", {})
            if bot_overrides:
                coin_overrides[coin] = {
                    "long": dict(bot_overrides.get("long", {})),
                    "short": dict(bot_overrides.get("short", {})),
                }

        return cls(
            base_long=base_long,
            base_short=base_short,
            coins=coins,
            coin_overrides=coin_overrides,
        )

    def build_bot_params_list(self, sampled_params: dict[str, float]) -> list[dict]:
        """Build bot_params_list for all coins from sampled params.

        Args:
            sampled_params: Dict of param_name -> value (e.g., 'long_entry_grid_spacing_pct' -> 0.01)

        Returns:
            List of bot_params dicts, one per coin, ready for Rust backtest
        """
        # Start with base values and merge sampled params
        # Use dict unpacking for efficiency (no deepcopy needed - values are floats)
        long_params = {**self.base_long}
        short_params = {**self.base_short}

        # Apply sampled params based on prefix
        for name, value in sampled_params.items():
            if name.startswith("long_"):
                long_params[name[5:]] = value
            elif name.startswith("short_"):
                short_params[name[6:]] = value

        # Build per-coin params
        bot_params_list = []
        for coin in self.coins:
            # Shallow copy for each coin (just floats, so this is safe and fast)
            coin_long = dict(long_params)
            coin_short = dict(short_params)

            # Apply coin-specific overrides if any
            if coin in self.coin_overrides:
                overrides = self.coin_overrides[coin]
                coin_long.update(overrides.get("long", {}))
                coin_short.update(overrides.get("short", {}))

            # Set wallet_exposure_limit to -1.0 if not overridden
            # (This matches prep_backtest_args behavior)
            if coin not in self.coin_overrides or \
               "wallet_exposure_limit" not in self.coin_overrides.get(coin, {}).get("long", {}):
                coin_long["wallet_exposure_limit"] = -1.0
            if coin not in self.coin_overrides or \
               "wallet_exposure_limit" not in self.coin_overrides.get(coin, {}).get("short", {}):
                coin_short["wallet_exposure_limit"] = -1.0

            bot_params_list.append({"long": coin_long, "short": coin_short})

        return bot_params_list
