"""Decode authored coin flag strings for the explicit v8 converter."""

import argparse

from config.overrides import expand_PB_mode, set_nested_value_safe


def parse_released_v8_coin_flags(config) -> dict:
    key_map = {
        "short_mode": ["live", "forced_mode_short"],
        "long_mode": ["live", "forced_mode_long"],
        "WE_limit_long": ["bot", "long", "wallet_exposure_limit_ratio"],
        "WE_limit_short": ["bot", "short", "wallet_exposure_limit_ratio"],
        "leverage": ["live", "leverage"],
    }
    if (
        not isinstance(config, dict)
        or "live" not in config
        or "coin_flags" not in config["live"]
    ):
        return {}
    flags = config["live"]["coin_flags"]
    if not isinstance(flags, dict):
        return {}
    result = {}
    for coin in flags:
        result[coin] = {}
        if not isinstance(flags[coin], str):
            continue
        parser = _build_flag_argparser()
        keysvals = vars(parser.parse_args(flags[coin].split()))
        if live_config_path := keysvals.get("live_config_path"):
            set_nested_value_safe(
                result[coin],
                ["override_config_path"],
                live_config_path,
                create_missing=True,
            )
        for key, value in keysvals.items():
            if value and key in key_map:
                set_nested_value_safe(
                    result[coin], key_map[key], value, create_missing=True
                )
    return result


def _build_flag_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coin_flags", add_help=False)
    parser.add_argument("-sm", type=expand_PB_mode, dest="short_mode", default=None)
    parser.add_argument("-lm", type=expand_PB_mode, dest="long_mode", default=None)
    parser.add_argument("-lw", type=float, dest="WE_limit_long", default=None)
    parser.add_argument("-sw", type=float, dest="WE_limit_short", default=None)
    parser.add_argument("-lev", type=float, dest="leverage", default=None)
    parser.add_argument("-lc", type=str, dest="live_config_path", default=None)
    return parser
