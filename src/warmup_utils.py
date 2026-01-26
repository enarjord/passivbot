import math
from typing import Iterator, Tuple

from config_utils import require_config_value, require_live_value


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _require_max_warmup_minutes(config: dict) -> float:
    return _to_float(require_live_value(config, "max_warmup_minutes"))


def _iter_param_sets(config: dict) -> Iterator[Tuple[str, dict, dict]]:
    bot_cfg = config.get("bot", {})
    base_long = dict(bot_cfg.get("long", {}) or {})
    base_short = dict(bot_cfg.get("short", {}) or {})
    yield "__default__", base_long, base_short

    coin_overrides = config.get("coin_overrides", {})
    for coin, overrides in coin_overrides.items():
        bot_overrides = overrides.get("bot", {})
        long_params = dict(base_long)
        short_params = dict(base_short)
        long_params.update(bot_overrides.get("long", {}) or {})
        short_params.update(bot_overrides.get("short", {}) or {})
        yield coin, long_params, short_params


def compute_backtest_warmup_minutes(config: dict) -> int:
    """Mirror Rust warmup span calculation (see calc_warmup_bars)."""

    def _extract_bound_max(bounds: dict, key: str) -> float:
        if key not in bounds:
            return 0.0
        entry = bounds[key]
        candidates = [entry] if isinstance(entry, (list, tuple)) else [[entry]]
        max_val = 0.0
        for candidate in candidates:
            for val in candidate:
                max_val = max(max_val, _to_float(val))
        return max_val

    max_minutes = 0.0
    minute_fields = [
        "ema_span_0",
        "ema_span_1",
        "filter_volume_ema_span",
        "filter_volatility_ema_span",
    ]

    for _, long_params, short_params in _iter_param_sets(config):
        for params in (long_params, short_params):
            for field in minute_fields:
                max_minutes = max(max_minutes, _to_float(params.get(field)))
            log_span_minutes = _to_float(params.get("entry_volatility_ema_span_hours")) * 60.0
            max_minutes = max(max_minutes, log_span_minutes)

    bounds = config.get("optimize", {}).get("bounds", {})
    bound_keys_minutes = [
        "long_ema_span_0",
        "long_ema_span_1",
        "long_filter_volume_ema_span",
        "long_filter_volatility_ema_span",
        "short_ema_span_0",
        "short_ema_span_1",
        "short_filter_volume_ema_span",
        "short_filter_volatility_ema_span",
    ]
    bound_keys_hours = [
        "long_entry_volatility_ema_span_hours",
        "short_entry_volatility_ema_span_hours",
    ]

    for key in bound_keys_minutes:
        max_minutes = max(max_minutes, _extract_bound_max(bounds, key))
    for key in bound_keys_hours:
        max_minutes = max(max_minutes, _extract_bound_max(bounds, key) * 60.0)

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
        "filter_volume_ema_span",
        "filter_volatility_ema_span",
    ]
    for coin, long_params, short_params in _iter_param_sets(config):
        max_minutes = 0.0
        for params in (long_params, short_params):
            for field in minute_fields:
                max_minutes = max(max_minutes, _to_float(params.get(field)))
            max_minutes = max(
                max_minutes,
                _to_float(params.get("entry_volatility_ema_span_hours")) * 60.0,
            )
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
