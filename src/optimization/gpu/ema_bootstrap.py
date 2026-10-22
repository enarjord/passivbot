"""Private integer-count transport from original float64 ordinary EMA spans.

Rust owns the count arithmetic and derives price-band spans before interval scaling.
Python resolves candidate values and explicit coin pins before either is packed as float32.
"""

import numpy as np

from optimization.gpu import model

BOOTSTRAP_COUNT_COLS = 10


def effective_end_steps(end_steps, *, candidates, available):
    """Match the replay kernel's exclusive endpoint before input admission."""
    values = (np.full(candidates, available - 1, dtype=np.int32) if end_steps is None
              else np.asarray(end_steps, dtype=np.int32))
    if values.shape != (candidates,):
        raise ValueError(f"expected one multi-coin end step per candidate, got {values.shape}")
    return np.ascontiguousarray(np.clip(values, 1, available - 1), dtype=np.int32)


def validate_hourly_bootstrap_inputs(
    params, keys, coin_overrides, *, first_observed, coin_ranges,
    requested_start, warmup_bars, available, end_steps=None, side_activation=None,
):
    """Admit hourly-dependent GPU policies only with a real planning-time basis.

    This is a dataset capability boundary: all configured hourly inputs of an
    active policy must exist before its first planning boundary. It does not
    simulate selection, positions or eventual fills to predict consumption.
    """
    matrix = np.asarray(params, dtype=np.float64)
    width = len(keys)
    if keys == model.EMA_ANCHOR_MULTICOIN_PARAM_KEYS:
        strategy = model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS
        hour = "offset_volatility_ema_span_1h"
        weights = ("offset_volatility_1h_weight",)
        wel_column = model.EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
    elif keys == model.TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS:
        strategy = tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
        hour = "volatility_ema_span_1h"
        weights = tuple(key for key in strategy if key.endswith("volatility_1h_weight"))
        wel_column = model.TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
    else:
        raise ValueError("Unsupported GPU hourly EMA parameter layout")
    ends = effective_end_steps(end_steps, candidates=len(matrix), available=available)
    if side_activation is not None and (len(side_activation) != len(coin_overrides)
            or any(type(flag) is not bool for flag in side_activation)):
        raise TypeError("hourly bootstrap activation must contain one boolean per side")
    for side, overrides in enumerate(coin_overrides):
        if side_activation is not None and not side_activation[side]:
            continue
        policy = matrix[:, side * width:(side + 1) * width]
        enabled = ((policy[:, keys.index("total_wallet_exposure_limit")] > 0.0)
                   & (np.rint(policy[:, keys.index("n_positions")]) > 0.0))
        for coin, pins in enumerate(overrides):
            if pins[wel_column] == 0.0:
                continue
            def effective(key):
                pin = pins[strategy.index(key)]
                return pin if np.isfinite(pin) else policy[:, keys.index(key)]
            required = enabled & (effective(hour) > 0.0)
            weighted = np.zeros(len(matrix), dtype=bool)
            for key in weights:
                weighted |= effective(key) != 0.0
            required &= weighted
            if not np.any(required):
                continue
            if first_observed is None or coin_ranges is None:
                raise ValueError("GPU hourly EMA admission metadata is missing; rebuild prepared data")
            first, last, trade_start = coin_ranges[coin]
            planning = max(2, int(warmup_bars) + 1, int(requested_start), int(first), int(trade_start))
            missing = required & (planning < ends) & (planning <= last) & (first_observed[coin] > planning)
            if np.any(missing):
                rows = np.flatnonzero(missing)[:8].tolist()
                raise ValueError(
                    "GPU replay unavailable hourly EMA at first planning boundary; "
                    f"candidate rows {rows}, side {side}, coin {coin}, candle {planning}"
                )


def pack_bootstrap_counts(params, keys, coin_overrides, *, interval_minutes, available):
    """Return contiguous int32 [candidate, side, coin, channel] metadata."""
    import passivbot_rust as pbr

    helper = getattr(pbr, "gpu_ema_bootstrap_counts_py", None)
    if helper is None:
        raise RuntimeError("GPU replay requires a rebuilt native EMA bootstrap-count API")
    matrix = np.asarray(params, dtype=np.float64)
    sides = len(coin_overrides)
    width = len(keys)
    if matrix.ndim != 2 or matrix.shape[1] != sides * width:
        raise ValueError("GPU EMA bootstrap candidate width does not match its sides")
    if keys == model.EMA_ANCHOR_MULTICOIN_PARAM_KEYS:
        strategy_keys = model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS
        minute, hour = "offset_volatility_ema_span_1m", "offset_volatility_ema_span_1h"
        unstuck_start = model.EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN
    elif keys == model.TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS:
        strategy_keys = tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
        minute, hour = "volatility_ema_span_1m", "volatility_ema_span_1h"
        unstuck_start = model.TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN
    else:
        raise ValueError("Unsupported GPU EMA bootstrap parameter layout")
    leaves = ("ema_span_0", "ema_span_1", "unstuck_ema_span_0", "unstuck_ema_span_1",
              minute, hour, "forager_volume_ema_span_1m", "forager_volatility_ema_span_1m")
    pins = (strategy_keys.index("ema_span_0"), strategy_keys.index("ema_span_1"),
            unstuck_start, unstuck_start + 1, strategy_keys.index(minute),
            strategy_keys.index(hour), None, None)
    coins = len(coin_overrides[0])
    if not (0 < available <= np.iinfo(np.int32).max):
        raise ValueError("GPU EMA bootstrap history must fit a positive int32 sample count")
    if matrix.shape[0] * sides * coins * BOOTSTRAP_COUNT_COLS > np.iinfo(np.int32).max:
        raise ValueError("GPU EMA bootstrap buffer exceeds int32 shader indexing")
    original = np.empty((len(matrix), sides, coins, len(leaves)), dtype=np.float64)
    for side, overrides in enumerate(coin_overrides):
        overrides = np.asarray(overrides, dtype=np.float64)
        if overrides.ndim != 2 or len(overrides) != coins:
            raise ValueError("GPU EMA bootstrap coin views must share the ordered coin layout")
        for column, (key, pin) in enumerate(zip(leaves, pins)):
            values = matrix[:, side * width + keys.index(key), None]
            original[:, side, :, column] = (values if pin is None else
                np.where(np.isfinite(overrides[None, :, pin]), overrides[None, :, pin], values))
    counts = np.asarray(helper(original.reshape(-1, len(leaves)), float(interval_minutes),
                               int(available)))
    expected = (len(matrix) * sides * coins, BOOTSTRAP_COUNT_COLS)
    if counts.dtype != np.int32 or counts.shape != expected or np.any(counts < 1) or np.any(counts > available):
        raise ValueError("Native GPU EMA bootstrap counts violate the private int32 ABI")
    return np.ascontiguousarray(counts.reshape(len(matrix), sides, coins, BOOTSTRAP_COUNT_COLS))
