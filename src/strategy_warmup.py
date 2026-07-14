from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


MISSING = object()

STRATEGY_WARMUP_1M_PROBE_KEYS = (
    "ema_span_0",
    "ema_span_1",
    "volatility_ema_span_1m",
    "offset_volatility_ema_span_1m",
    "entry_volatility_ema_span_1m",
)

STRATEGY_WARMUP_M1_LOG_RANGE_PROBE_KEYS = (
    "volatility_ema_span_1m",
    "offset_volatility_ema_span_1m",
    "entry_volatility_ema_span_1m",
)

STRATEGY_WARMUP_H1_PROBE_KEYS = (
    "volatility_ema_span_1h",
    "entry_volatility_ema_span_1h",
    "offset_volatility_ema_span_1h",
)

STRATEGY_WARMUP_1M_BOUND_LOCAL_KEYS = STRATEGY_WARMUP_1M_PROBE_KEYS
STRATEGY_WARMUP_H1_BOUND_LOCAL_KEYS = (
    "volatility_ema_span_1h",
    "entry_volatility_ema_span_1h",
    "entry_volatility_ema_span_hours",
    "offset_volatility_ema_span_1h",
)

STRATEGY_WARMUP_PROBE_PATHS = {
    "ema_span_0": (("ema_span_0",),),
    "ema_span_1": (("ema_span_1",),),
    "volatility_ema_span_1m": (
        ("volatility_ema_span_1m",),
        ("entry_volatility_ema_span_1m",),
        ("offset_volatility_ema_span_1m",),
    ),
    "entry_volatility_ema_span_1m": (
        ("entry_volatility_ema_span_1m",),
        ("volatility_ema_span_1m",),
    ),
    "offset_volatility_ema_span_1m": (("offset_volatility_ema_span_1m",),),
    "volatility_ema_span_1h": (
        ("volatility_ema_span_1h",),
        ("offset_volatility_ema_span_1h",),
        ("entry_volatility_ema_span_1h",),
        ("entry", "volatility_ema_span_hours"),
    ),
    "entry_volatility_ema_span_1h": (
        ("entry", "volatility_ema_span_hours"),
        ("entry_volatility_ema_span_1h",),
        ("volatility_ema_span_1h",),
    ),
    "offset_volatility_ema_span_1h": (("offset_volatility_ema_span_1h",),),
}


@dataclass(frozen=True)
class StrategyWarmupRequirements:
    max_1m_span_minutes: float
    max_m1_log_range_span_minutes: float
    max_h1_span_hours: float


def _format_symbol(symbol: str | None) -> str:
    return str(symbol) if symbol else "unknown"


def _path_get(mapping: dict, path: tuple[str, ...]) -> tuple[bool, Any]:
    current = mapping
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False, MISSING
        current = current[part]
    return True, current


def positive_finite_warmup_value(
    value: Any,
    *,
    context: str,
    symbol: str | None = None,
    error_label: str = "warmup",
) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid {error_label} value for {context} "
            f"symbol={_format_symbol(symbol)}: {value!r}"
        ) from exc
    if not math.isfinite(val):
        raise ValueError(
            f"invalid {error_label} value for {context} "
            f"symbol={_format_symbol(symbol)}: {value!r}"
        )
    return val if val > 0.0 else 0.0


def strategy_warmup_value(
    strategy_params: dict,
    key: str,
    *,
    pside: str,
    symbol: str | None = None,
    optional: bool = True,
    error_label: str = "warmup",
) -> float:
    paths = STRATEGY_WARMUP_PROBE_PATHS.get(key, ((key,),))
    found = False
    max_value = 0.0
    for path in paths:
        present, raw = _path_get(strategy_params, path)
        if not present:
            continue
        found = True
        max_value = max(
            max_value,
            positive_finite_warmup_value(
                raw,
                context=f"strategy {pside}.{'.'.join(path)}",
                symbol=symbol,
                error_label=error_label,
            ),
        )
    if found or optional:
        return max_value
    rendered = " | ".join(".".join(path) for path in paths)
    raise KeyError(f"missing required strategy warmup path for {pside}: {rendered}")


def strategy_warmup_requirements(
    strategy_params: dict,
    *,
    pside: str,
    symbol: str | None = None,
    error_label: str = "warmup",
) -> StrategyWarmupRequirements:
    if not isinstance(strategy_params, dict):
        raise TypeError(
            f"strategy warmup expected dict for {pside}; got {type(strategy_params).__name__}"
        )
    max_1m_span = 0.0
    max_m1_log_range_span = 0.0
    max_h1_span = 0.0
    for key in STRATEGY_WARMUP_1M_PROBE_KEYS:
        max_1m_span = max(
            max_1m_span,
            strategy_warmup_value(
                strategy_params,
                key,
                pside=pside,
                symbol=symbol,
                error_label=error_label,
            ),
        )
    for key in STRATEGY_WARMUP_M1_LOG_RANGE_PROBE_KEYS:
        max_m1_log_range_span = max(
            max_m1_log_range_span,
            strategy_warmup_value(
                strategy_params,
                key,
                pside=pside,
                symbol=symbol,
                error_label=error_label,
            ),
        )
    for key in STRATEGY_WARMUP_H1_PROBE_KEYS:
        max_h1_span = max(
            max_h1_span,
            strategy_warmup_value(
                strategy_params,
                key,
                pside=pside,
                symbol=symbol,
                error_label=error_label,
            ),
        )
    return StrategyWarmupRequirements(
        max_1m_span_minutes=max_1m_span,
        max_m1_log_range_span_minutes=max_m1_log_range_span,
        max_h1_span_hours=max_h1_span,
    )


def iter_strategy_warmup_flat_bound_keys(unit: str) -> tuple[str, ...]:
    local_keys = (
        STRATEGY_WARMUP_1M_BOUND_LOCAL_KEYS
        if unit == "1m"
        else STRATEGY_WARMUP_H1_BOUND_LOCAL_KEYS
    )
    return tuple(f"{pside}_{key}" for pside in ("long", "short") for key in local_keys)


def strategy_abs_max_weight(
    params: dict,
    *,
    paths: tuple[tuple[str, ...], ...],
    pside: str,
    context: str,
    symbol: str | None,
    optional: bool,
    error_label: str = "warmup",
) -> float:
    found = False
    max_abs = 0.0
    for path in paths:
        present, raw = _path_get(params, path)
        if not present:
            continue
        found = True
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {error_label} value for strategy {pside}.{'.'.join(path)} "
                f"symbol={_format_symbol(symbol)}: {raw!r}"
            ) from exc
        if not math.isfinite(val):
            raise ValueError(
                f"invalid {error_label} value for strategy {pside}.{'.'.join(path)} "
                f"symbol={_format_symbol(symbol)}: {raw!r}"
            )
        max_abs = max(max_abs, abs(val))
    if found or optional:
        return max_abs
    rendered = " | ".join(".".join(path) for path in paths)
    raise KeyError(f"missing required strategy parameter path for {context}: {rendered}")


__all__ = [
    "STRATEGY_WARMUP_1M_PROBE_KEYS",
    "STRATEGY_WARMUP_M1_LOG_RANGE_PROBE_KEYS",
    "STRATEGY_WARMUP_H1_PROBE_KEYS",
    "iter_strategy_warmup_flat_bound_keys",
    "positive_finite_warmup_value",
    "strategy_abs_max_weight",
    "strategy_warmup_requirements",
    "strategy_warmup_value",
]
