"""Configuration and candle plumbing for the Rust directional-efficiency experiment."""

from __future__ import annotations

import math
import logging
from numbers import Real

import numpy as np

ONE_MIN_MS = 60_000
MAX_LOOKBACK = 10_080
FIELDS = {
    "forager_directional_efficiency_lookback_minutes": (1, MAX_LOOKBACK, True),
    "forager_directional_efficiency_penalty": (0, 1, False),
    "risk_directional_efficiency_lookback_minutes": (1, MAX_LOOKBACK, True),
    "risk_directional_efficiency_cooldown_minutes": (0, MAX_LOOKBACK, False),
}


def validate_params(params: dict, *, path: str) -> None:
    # Only canonical hydration supplies defaults. Partial override validation checks present keys.
    for key, (low, high, integer) in FIELDS.items():
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{path}.{key} must be numeric")
        if (
            not math.isfinite(value)
            or not low <= value <= high
            or (integer and value != int(value))
        ):
            raise ValueError(
                f"{path}.{key} must be {'an integer ' if integer else ''}in [{low}, {high}]"
            )


def required_windows(params: dict) -> set[int]:
    windows = set()
    for group, control in (("forager", "penalty"), ("risk", "cooldown_minutes")):
        if params.get(f"{group}_directional_efficiency_{control}", 0) > 0:
            windows.add(int(params[f"{group}_directional_efficiency_lookback_minutes"]))
    return windows


def validate_bounds(config: dict) -> None:
    from config.optimize_bounds import flatten_optimize_bounds
    from optimization.bounds import Bound

    bounds = flatten_optimize_bounds(
        config.get("optimize", {}).get("bounds", {}),
        strategy_kind=config.get("live", {}).get("strategy_kind"),
    )
    for key, value in bounds.items():
        field = key.split("_", 1)[-1]
        if field not in FIELDS:
            continue
        bound = Bound.from_config(key, value)
        validate_params({field: bound.low}, path="optimize.bounds")
        validate_params({field: bound.high}, path="optimize.bounds")
        if FIELDS[field][2] and bound.low != bound.high:
            if bound.step is None or bound.step < 1 or bound.step != int(bound.step):
                raise ValueError(f"optimize.bounds.{key} requires an integer step >= 1")
        if field.endswith(("_penalty", "_cooldown_minutes")) and bound.high > 0:
            if config["backtest"]["candle_interval_minutes"] != 1:
                raise ValueError(
                    "directional efficiency bounds require 1 minute backtest candles"
                )


def reject_gpu_directional_efficiency(config: dict) -> None:
    """GPU proxies do not implement this feature; never silently score a different strategy."""
    controls = (
        "directional_efficiency_penalty",
        "directional_efficiency_cooldown_minutes",
    )

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(str(key).endswith(name) for name in controls):
                    values = child[:2] if isinstance(child, (list, tuple)) else [child]
                    if any(isinstance(v, Real) and v > 0 for v in values):
                        raise ValueError(
                            "directional efficiency requires the exact CPU optimizer (backend=pymoo); GPU proxies are unsupported"
                        )
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(config)


def values_from_candles(
    candles: np.ndarray, windows: set[int], end_ts: int
) -> list[list[float]]:
    """An incomplete window is unavailable, never shortened or filled with invented prices."""
    import passivbot_rust as pbr

    result = []
    for window in sorted(windows):
        start = end_ts - window * ONE_MIN_MS
        rows = candles[(candles["ts"] >= start) & (candles["ts"] <= end_ts)]
        expected = np.arange(start, end_ts + ONE_MIN_MS, ONE_MIN_MS, dtype=np.int64)
        if len(rows) != window + 1 or not np.array_equal(rows["ts"], expected):
            continue
        # Rust validates prices and owns the calculation in both runtimes.
        result.append(
            [
                float(window),
                float(
                    pbr.calc_directional_efficiency(rows["c"].astype(float).tolist())
                ),
            ]
        )
    return result


async def load_symbol_values(
    cm,
    symbol: str,
    windows: set[int],
    *,
    now_ms: int,
    allow_remote_fetch: bool,
    ranking_max_age_ms: int = 0,
):
    if not windows:
        return [], []
    end_ts = (int(now_ms) // ONE_MIN_MS) * ONE_MIN_MS - ONE_MIN_MS
    extra_bars = math.ceil(max(0, ranking_max_age_ms) / ONE_MIN_MS)
    from candlestick_manager import OhlcvFetchError
    from ccxt import NetworkError

    try:
        candles = await cm.get_candles(
            symbol,
            start_ts=end_ts - (max(windows) + extra_bars) * ONE_MIN_MS,
            end_ts=end_ts,
            timeframe="1m",
            strict=False,
            allow_remote_fetch=allow_remote_fetch,
            fill_leading_gaps=False,
            fill_trailing_gaps=False,
            allow_provisional_internal_gaps=False,
        )
    except (OhlcvFetchError, NetworkError, TimeoutError, OSError) as exc:
        logging.warning(
            "[directional_efficiency] candle input unavailable symbol=%s error_type=%s",
            symbol,
            type(exc).__name__,
        )
        return [], []
    current = values_from_candles(candles, windows, end_ts)
    # Ranking may carry a real completed window within the existing forager staleness budget.
    # DCA never consumes that stale value.
    finalized = candles[candles["ts"] <= end_ts]
    if len(finalized) == 0:
        return current, []
    last_ts = int(finalized["ts"][-1])
    if end_ts - last_ts > ranking_max_age_ms:
        return current, []
    ranking = (
        current
        if last_ts == end_ts
        else values_from_candles(finalized, windows, last_ts)
    )
    return current, ranking
