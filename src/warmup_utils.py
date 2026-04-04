import math
from typing import Iterator, Tuple

from config.access import require_config_value, require_live_value
from config.optimize_bounds import flatten_optimize_bounds
from config.shared_bot import flatten_shared_bot_side
from config.strategy import (
    build_runtime_strategy_side,
    get_active_strategy_config,
    normalize_strategy_kind,
)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _require_max_warmup_minutes(config: dict) -> float:
    return _to_float(require_live_value(config, "max_warmup_minutes"))


def _accumulate_max_minutes(max_minutes: float, *values) -> tuple[float, bool]:
    for value in values:
        numeric = _to_float(value)
        if not math.isfinite(numeric):
            return max_minutes, False
        max_minutes = max(max_minutes, numeric)
    return max_minutes, True


def _iter_param_sets(config: dict) -> Iterator[Tuple[str, dict, dict]]:
    bot_cfg = config.get("bot", {})
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    strategy_cfg = get_active_strategy_config(config, strategy_kind=strategy_kind)
    base_long = flatten_shared_bot_side(bot_cfg.get("long", {}) or {})
    base_short = flatten_shared_bot_side(bot_cfg.get("short", {}) or {})
    default_long_strategy = (
        build_runtime_strategy_side(
            strategy_cfg.get("long", {}),
            strategy_kind=strategy_kind,
            pside="long",
        )
        if strategy_cfg.get("long")
        else {}
    )
    default_short_strategy = (
        build_runtime_strategy_side(
            strategy_cfg.get("short", {}),
            strategy_kind=strategy_kind,
            pside="short",
        )
        if strategy_cfg.get("short")
        else {}
    )
    yield (
        "__default__",
        base_long,
        base_short,
        default_long_strategy,
        default_short_strategy,
    )

    coin_overrides = config.get("coin_overrides", {})
    for coin, overrides in coin_overrides.items():
        bot_overrides = overrides.get("bot", {})
        long_params = dict(base_long)
        short_params = dict(base_short)
        long_params.update(flatten_shared_bot_side(bot_overrides.get("long", {}) or {}))
        short_params.update(flatten_shared_bot_side(bot_overrides.get("short", {}) or {}))
        yield (
            coin,
            long_params,
            short_params,
            (
                build_runtime_strategy_side(
                    strategy_cfg.get("long", {}),
                    strategy_kind=strategy_kind,
                    pside="long",
                    override_side=bot_overrides.get("long", {}),
                )
                if strategy_cfg.get("long") or bot_overrides.get("long", {}).get("strategy")
                else {}
            ),
            (
                build_runtime_strategy_side(
                    strategy_cfg.get("short", {}),
                    strategy_kind=strategy_kind,
                    pside="short",
                    override_side=bot_overrides.get("short", {}),
                )
                if strategy_cfg.get("short") or bot_overrides.get("short", {}).get("strategy")
                else {}
            ),
        )


def compute_backtest_warmup_minutes(config: dict) -> int:
    """Mirror Rust warmup span calculation (see calc_warmup_bars)."""

    def _extract_bound_max(bounds: dict, key: str) -> tuple[float, bool]:
        if key not in bounds:
            return 0.0, True
        entry = bounds[key]
        candidates = [entry] if isinstance(entry, (list, tuple)) else [[entry]]
        max_val = 0.0
        for candidate in candidates:
            for val in candidate:
                max_val, is_valid = _accumulate_max_minutes(max_val, val)
                if not is_valid:
                    return max_val, False
        return max_val, True

    max_minutes = 0.0
    minute_fields = [
        "ema_span_0",
        "ema_span_1",
        "forager_volume_ema_span",
        "forager_volatility_ema_span",
    ]

    for _, long_params, short_params, long_strategy, short_strategy in _iter_param_sets(config):
        strategy_params = (long_strategy, short_strategy)
        for params, strategy in zip((long_params, short_params), strategy_params):
            for field in minute_fields:
                max_minutes, is_valid = _accumulate_max_minutes(max_minutes, params.get(field))
                if not is_valid:
                    return 0
            max_minutes, is_valid = _accumulate_max_minutes(
                max_minutes,
                _to_float(strategy.get("ema_span_0")),
                _to_float(strategy.get("ema_span_1")),
            )
            if not is_valid:
                return 0
            log_span_minutes = _to_float(strategy.get("entry_volatility_ema_span_hours")) * 60.0
            max_minutes, is_valid = _accumulate_max_minutes(max_minutes, log_span_minutes)
            if not is_valid:
                return 0

    bounds = flatten_optimize_bounds(
        config.get("optimize", {}).get("bounds", {}),
        strategy_kind=config.get("live", {}).get("strategy_kind"),
    )
    bound_keys_minutes = [
        "long_ema_span_0",
        "long_ema_span_1",
        "long_forager_volume_ema_span",
        "long_forager_volatility_ema_span",
        "short_ema_span_0",
        "short_ema_span_1",
        "short_forager_volume_ema_span",
        "short_forager_volatility_ema_span",
    ]
    bound_keys_hours = [
        "long_entry_volatility_ema_span_hours",
        "short_entry_volatility_ema_span_hours",
    ]

    for key in bound_keys_minutes:
        extracted, is_valid = _extract_bound_max(bounds, key)
        if not is_valid:
            return 0
        max_minutes = max(max_minutes, extracted)
    for key in bound_keys_hours:
        extracted, is_valid = _extract_bound_max(bounds, key)
        if not is_valid:
            return 0
        max_minutes = max(max_minutes, extracted * 60.0)

    warmup_ratio = float(require_config_value(config, "live.warmup_ratio"))
    limit = _require_max_warmup_minutes(config)

    if not math.isfinite(max_minutes):
        return 0
    warmup_minutes = max_minutes * max(0.0, warmup_ratio)
    if limit > 0:
        warmup_minutes = min(warmup_minutes, limit)
    return int(math.ceil(warmup_minutes)) if warmup_minutes > 0.0 else 0


def compute_per_coin_warmup_minutes(config: dict) -> dict:
    warmup_ratio = float(require_config_value(config, "live.warmup_ratio"))
    limit = _require_max_warmup_minutes(config)
    per_coin = {}
    minute_fields = [
        "ema_span_0",
        "ema_span_1",
        "forager_volume_ema_span",
        "forager_volatility_ema_span",
    ]
    for coin, long_params, short_params, long_strategy, short_strategy in _iter_param_sets(config):
        max_minutes = 0.0
        strategy_params = (long_strategy, short_strategy)
        for params, strategy in zip((long_params, short_params), strategy_params):
            for field in minute_fields:
                max_minutes, is_valid = _accumulate_max_minutes(max_minutes, params.get(field))
                if not is_valid:
                    per_coin[coin] = 0
                    break
            else:
                max_minutes, is_valid = _accumulate_max_minutes(
                    max_minutes,
                    _to_float(strategy.get("ema_span_0")),
                    _to_float(strategy.get("ema_span_1")),
                )
                if not is_valid:
                    per_coin[coin] = 0
                    break
                max_minutes, is_valid = _accumulate_max_minutes(
                    max_minutes,
                    _to_float(strategy.get("entry_volatility_ema_span_hours")) * 60.0,
                )
                if not is_valid:
                    per_coin[coin] = 0
                    break
                continue
            break
        if coin in per_coin:
            continue
        if not math.isfinite(max_minutes):
            per_coin[coin] = 0
            continue
        warmup_minutes = max_minutes * max(0.0, warmup_ratio)
        if limit > 0:
            warmup_minutes = min(warmup_minutes, limit)
        per_coin[coin] = int(math.ceil(warmup_minutes)) if warmup_minutes > 0.0 else 0
    return per_coin


__all__ = [
    "compute_backtest_warmup_minutes",
    "compute_per_coin_warmup_minutes",
]
