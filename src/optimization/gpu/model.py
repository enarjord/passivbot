from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


GAP_BINS = 128
GAP_MAX_MINUTES = 4_000_000.0
MPS_MULTICOIN_MAX_COINS = 64

EMA_ANCHOR_PARAM_KEYS = (
    "base_qty_pct",
    "ema_span_0",
    "ema_span_1",
    "entry_double_down_factor",
    "offset",
    "offset_psize_weight",
    "offset_volatility_1h_weight",
    "offset_volatility_1m_weight",
    "offset_volatility_ema_span_1h",
    "offset_volatility_ema_span_1m",
    "entry_cooldown_minutes",
    "total_wallet_exposure_limit",
)

EMA_ANCHOR_MULTICOIN_PARAM_KEYS = (
    *EMA_ANCHOR_PARAM_KEYS,
    "forager_volume_ema_span_1m",
    "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct",
    "forager_score_weights_volume",
    "forager_score_weights_ema_readiness",
    "forager_score_weights_volatility",
    "n_positions",
)

TRAILING_MARTINGALE_PARAM_KEYS = (
    "ema_span_0",
    "ema_span_1",
    "volatility_ema_span_1h",
    "volatility_ema_span_1m",
    "entry_double_down_factor",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
    "entry_threshold_base_pct",
    "entry_threshold_we_weight",
    "entry_threshold_volatility_1h_weight",
    "entry_threshold_volatility_1m_weight",
    "entry_retracement_base_pct",
    "entry_retracement_we_weight",
    "entry_retracement_volatility_1h_weight",
    "entry_retracement_volatility_1m_weight",
    "close_qty_pct",
    "close_threshold_base_pct",
    "close_threshold_we_weight",
    "close_threshold_volatility_1h_weight",
    "close_threshold_volatility_1m_weight",
    "close_retracement_base_pct",
    "close_retracement_volatility_1h_weight",
    "close_retracement_volatility_1m_weight",
    "entry_cooldown_minutes",
    "total_wallet_exposure_limit",
    "gate_initial",
    "gate_reentry",
)

TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS = (
    *TRAILING_MARTINGALE_PARAM_KEYS,
    "forager_volume_ema_span_1m",
    "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct",
    "forager_score_weights_volume",
    "forager_score_weights_ema_readiness",
    "forager_score_weights_volatility",
    "n_positions",
)

TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS = (
    ("ema_span_0", ("ema_span_0",)),
    ("ema_span_1", ("ema_span_1",)),
    ("volatility_ema_span_1h", ("volatility_ema_span_1h",)),
    ("volatility_ema_span_1m", ("volatility_ema_span_1m",)),
    ("entry_double_down_factor", ("entry", "double_down_factor")),
    ("entry_initial_ema_dist", ("entry", "initial_ema_dist")),
    ("entry_initial_qty_pct", ("entry", "initial_qty_pct")),
    ("entry_threshold_base_pct", ("entry", "threshold_base_pct")),
    ("entry_threshold_we_weight", ("entry", "threshold_we_weight")),
    (
        "entry_threshold_volatility_1h_weight",
        ("entry", "threshold_volatility_1h_weight"),
    ),
    (
        "entry_threshold_volatility_1m_weight",
        ("entry", "threshold_volatility_1m_weight"),
    ),
    ("entry_retracement_base_pct", ("entry", "retracement_base_pct")),
    ("entry_retracement_we_weight", ("entry", "retracement_we_weight")),
    (
        "entry_retracement_volatility_1h_weight",
        ("entry", "retracement_volatility_1h_weight"),
    ),
    (
        "entry_retracement_volatility_1m_weight",
        ("entry", "retracement_volatility_1m_weight"),
    ),
    ("close_qty_pct", ("close", "qty_pct")),
    ("close_threshold_base_pct", ("close", "threshold_base_pct")),
    ("close_threshold_we_weight", ("close", "threshold_we_weight")),
    (
        "close_threshold_volatility_1h_weight",
        ("close", "threshold_volatility_1h_weight"),
    ),
    (
        "close_threshold_volatility_1m_weight",
        ("close", "threshold_volatility_1m_weight"),
    ),
    ("close_retracement_base_pct", ("close", "retracement_base_pct")),
    (
        "close_retracement_volatility_1h_weight",
        ("close", "retracement_volatility_1h_weight"),
    ),
    (
        "close_retracement_volatility_1m_weight",
        ("close", "retracement_volatility_1m_weight"),
    ),
)

GPU_STRATEGY_PARAM_KEYS = {
    "ema_anchor": EMA_ANCHOR_PARAM_KEYS,
    "trailing_martingale": TRAILING_MARTINGALE_PARAM_KEYS,
}


def flatten_trailing_martingale_params(strategy: dict, risk: dict) -> dict:
    """Flatten Rust's nested TM payload into the Metal row contract."""

    entry = strategy.get("entry", {})
    close = strategy.get("close", {})
    mode = str(entry.get("ema_gate_mode", "all")).strip().lower()
    if mode not in {"disabled", "all", "initial", "reentry"}:
        raise ValueError(f"unsupported trailing_martingale entry.ema_gate_mode={mode!r}")
    flattened = {
        "ema_span_0": strategy.get("ema_span_0"),
        "ema_span_1": strategy.get("ema_span_1"),
        "volatility_ema_span_1h": strategy.get("volatility_ema_span_1h"),
        "volatility_ema_span_1m": strategy.get("volatility_ema_span_1m"),
        "entry_cooldown_minutes": float(
            risk.get("entry_cooldown_minutes", 0.0) or 0.0
        ),
        "total_wallet_exposure_limit": float(
            risk["total_wallet_exposure_limit"]
        ),
        "gate_initial": float(mode in {"all", "initial"}),
        "gate_reentry": float(mode in {"all", "reentry"}),
    }
    for prefix, values, keys in (
        (
            "entry",
            entry,
            (
                "double_down_factor",
                "initial_ema_dist",
                "initial_qty_pct",
                "threshold_base_pct",
                "threshold_we_weight",
                "threshold_volatility_1h_weight",
                "threshold_volatility_1m_weight",
                "retracement_base_pct",
                "retracement_we_weight",
                "retracement_volatility_1h_weight",
                "retracement_volatility_1m_weight",
            ),
        ),
        (
            "close",
            close,
            (
                "qty_pct",
                "threshold_base_pct",
                "threshold_we_weight",
                "threshold_volatility_1h_weight",
                "threshold_volatility_1m_weight",
                "retracement_base_pct",
                "retracement_volatility_1h_weight",
                "retracement_volatility_1m_weight",
            ),
        ),
    ):
        for key in keys:
            flattened[f"{prefix}_{key}"] = values.get(key)
    return {key: flattened[key] for key in TRAILING_MARTINGALE_PARAM_KEYS}


def gpu_side_enabled(config: dict, side: str) -> bool:
    """Match Rust/backtest global side eligibility, including approved coins."""

    risk = config.get("bot", {}).get(side, {}).get("risk", {})
    total_exposure = float(risk.get("total_wallet_exposure_limit", 0.0) or 0.0)
    n_positions_raw = float(risk.get("n_positions", 0) or 0)
    if (
        not math.isfinite(total_exposure)
        or not math.isfinite(n_positions_raw)
        or total_exposure <= 0.0
        or int(round(n_positions_raw)) <= 0
    ):
        return False
    approved = config.get("live", {}).get("approved_coins", {})
    if isinstance(approved, dict):
        return bool(approved.get(side, []))
    return True


@dataclass(frozen=True)
class ProxyMarket:
    qty_step: float
    price_step: float
    min_qty: float
    min_cost: float
    c_mult: float
    maker_fee: float


@dataclass(frozen=True)
class ProxyRun:
    starting_balance: float
    warmup_bars: int
    trade_start_idx: int
    requested_start_ts_ms: int
    guard_ts_ms: int
    first_ts_ms: int
    interval_ms: int
    liquidation_threshold: float
    first_valid_idx: int
    last_valid_idx: int


def _build_hourly_log_range(high, low, timestamps, run: ProxyRun):
    n = len(timestamps)
    derived_timestamps = (
        int(timestamps[0]) + np.arange(n, dtype=np.int64) * run.interval_ms
    )
    hour_idx = derived_timestamps // 3_600_000
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = hour_idx[1:] > hour_idx[:-1]
    hour_log_range = np.zeros(n, dtype=np.float32)
    hour_valid = np.zeros(n, dtype=bool)
    last_boundary = 0
    last_hour_boundary_ms = (int(timestamps[0]) // 3_600_000) * 3_600_000
    first_valid = max(0, run.first_valid_idx)
    latest = None
    for k in range(1, n):
        if not boundary[k]:
            continue
        current_ts = int(derived_timestamps[k])
        window_start_ms = max(int(timestamps[0]), last_hour_boundary_ms)
        if current_ts > window_start_ms + run.interval_ms:
            start = max(last_boundary, first_valid)
            end = min(k - 1, run.last_valid_idx)
            if end >= start:
                h_segment = high[start : end + 1]
                l_segment = low[start : end + 1]
                finite = np.isfinite(h_segment) & np.isfinite(l_segment)
                if finite.any():
                    highest = h_segment[finite].max()
                    lowest = l_segment[finite].min()
                    if highest > 0.0 and lowest > 0.0:
                        latest = math.log(highest / lowest)
        if latest is not None:
            hour_log_range[k] = latest
            hour_valid[k] = True
        last_boundary = k
        last_hour_boundary_ms = (current_ts // 3_600_000) * 3_600_000
    return hour_log_range, hour_valid


def _strict_fill_tick_boundaries(
    high, low, price_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Encode Rust's strict candle/order comparisons as integer tick boundaries."""

    if not np.isfinite(price_step) or price_step <= 0.0:
        raise ValueError(
            f"MPS proxy requires a positive finite price_step, got {price_step}"
        )
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    finite = np.isfinite(high) & np.isfinite(low)
    safe_high = np.where(finite, high, 0.0)
    safe_low = np.where(finite, low, 0.0)
    high_fill_max = np.floor(safe_high / price_step).astype(np.int64)
    low_nonfill_max = np.floor(safe_low / price_step).astype(np.int64)

    decimal_multiplier = None
    multiplier = 1.0
    for places in range(16):
        scaled_step = abs(price_step) * multiplier
        rounded_step = np.floor(scaled_step + 0.5)
        if rounded_step >= 1.0 and abs(scaled_step - rounded_step) <= max(
            abs(scaled_step), 1.0
        ) * 1e-12:
            decimal_multiplier = 10.0 ** max(places, 10)
            break
        multiplier *= 10.0

    def rust_tick_prices(ticks):
        prices = ticks.astype(np.float64) * price_step
        if decimal_multiplier is None:
            return prices
        scaled = prices * decimal_multiplier
        return (
            np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
            / decimal_multiplier
        )

    # Division can land one tick to either side near an exact boundary. Repair
    # against Rust's step-decimal-preserving order-price rounding contract.
    for _ in range(2):
        high_fill_max -= (rust_tick_prices(high_fill_max) >= safe_high).astype(
            np.int64
        )
        high_fill_max += (rust_tick_prices(high_fill_max + 1) < safe_high).astype(
            np.int64
        )
        low_nonfill_max -= (rust_tick_prices(low_nonfill_max) > safe_low).astype(
            np.int64
        )
        low_nonfill_max += (
            rust_tick_prices(low_nonfill_max + 1) <= safe_low
        ).astype(np.int64)

    high_fill_max[~finite] = 0
    low_nonfill_max[~finite] = 0
    i32 = np.iinfo(np.int32)
    if (
        high_fill_max.min(initial=0) < i32.min
        or high_fill_max.max(initial=0) > i32.max
        or low_nonfill_max.min(initial=0) < i32.min
        or low_nonfill_max.max(initial=0) > i32.max
    ):
        raise ValueError("MPS proxy candle price ticks exceed signed 32-bit range")
    return high_fill_max.astype(np.int32), low_nonfill_max.astype(np.int32)


def _directional_touch_ticks(
    prices, price_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode Rust-compatible down/up/nearest ticks for candle touch prices."""

    if not np.isfinite(price_step) or price_step <= 0.0:
        raise ValueError(
            f"MPS proxy requires a positive finite price_step, got {price_step}"
        )
    prices = np.asarray(prices, dtype=np.float64)
    finite = np.isfinite(prices) & (prices > 0.0)
    safe = np.where(finite, prices, 0.0)
    step_counts = safe / price_step
    nearest_ticks = np.floor(step_counts + 0.5).astype(np.int64)

    decimal_multiplier = None
    multiplier = 1.0
    for places in range(16):
        scaled_step = abs(price_step) * multiplier
        rounded_step = np.floor(scaled_step + 0.5)
        if rounded_step >= 1.0 and abs(scaled_step - rounded_step) <= max(
            abs(scaled_step), 1.0
        ) * 1e-12:
            decimal_multiplier = 10.0 ** max(places, 10)
            break
        multiplier *= 10.0

    nearest_prices = nearest_ticks.astype(np.float64) * price_step
    if decimal_multiplier is not None:
        scaled = nearest_prices * decimal_multiplier
        nearest_prices = (
            np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
            / decimal_multiplier
        )
    tolerance = (
        np.finfo(np.float64).eps
        * np.maximum(np.abs(safe), np.abs(nearest_prices))
        * 4.0
    )
    aligned = finite & (np.abs(safe - nearest_prices) <= tolerance)
    down_ticks = np.where(aligned, nearest_ticks, np.floor(step_counts)).astype(
        np.int64
    )
    up_ticks = np.where(aligned, nearest_ticks, np.ceil(step_counts)).astype(np.int64)
    down_ticks[~finite] = 0
    up_ticks[~finite] = 0
    nearest_ticks[~finite] = 0
    i32 = np.iinfo(np.int32)
    if (
        down_ticks.min(initial=0) < i32.min
        or down_ticks.max(initial=0) > i32.max
        or up_ticks.min(initial=0) < i32.min
        or up_ticks.max(initial=0) > i32.max
        or nearest_ticks.min(initial=0) < i32.min
        or nearest_ticks.max(initial=0) > i32.max
    ):
        raise ValueError("MPS proxy candle touch ticks exceed signed 32-bit range")
    return (
        down_ticks.astype(np.int32),
        up_ticks.astype(np.int32),
        nearest_ticks.astype(np.int32),
    )


def _minimum_entry_qty_encoding(
    prices, market: ProxyMarket
) -> tuple[np.ndarray, np.ndarray]:
    """Encode Rust-compatible float64 touch minima for float32 Metal comparisons."""

    prices = np.asarray(prices, dtype=np.float64)
    finite = np.isfinite(prices) & (prices > 0.0)
    safe = np.where(finite, prices, 1.0)
    raw_min = np.maximum(
        float(market.min_qty),
        float(market.min_cost) / safe / float(market.c_mult),
    )
    raw_steps = raw_min / float(market.qty_step)
    nearest_steps = np.floor(raw_steps + 0.5)
    nearest_qty = nearest_steps * float(market.qty_step)
    representation_tolerance = (
        np.finfo(np.float64).eps
        * np.maximum(np.abs(raw_min), np.abs(nearest_qty))
        * 4.0
    )
    aligned = (
        (nearest_steps > 0.0)
        & (np.abs(raw_steps - nearest_steps) <= 1e-8)
        & (
            (nearest_qty >= raw_min)
            | (raw_min - nearest_qty <= representation_tolerance)
        )
    )
    minimum_qty = np.where(
        raw_min == 0.0,
        0.0,
        np.where(
            aligned,
            np.maximum(nearest_qty, raw_min),
            np.ceil(raw_steps) * float(market.qty_step),
        ),
    )
    minimum_qty = np.where(finite, minimum_qty, 0.0)
    rounded = minimum_qty.astype(np.float32)
    if (
        np.any(~np.isfinite(minimum_qty))
        or np.any(minimum_qty < 0.0)
        or np.any(~np.isfinite(rounded))
    ):
        raise ValueError("MPS proxy minimum touch quantities exceed float32 range")
    relation = np.sign(minimum_qty - rounded.astype(np.float64)).astype(np.int32)
    rounded_bits = np.ascontiguousarray(rounded).view(np.int32)
    return rounded_bits, relation


def build_mps_data(high, low, close, timestamps_ms, run: ProxyRun, market: ProxyMarket):
    """Prepare immutable minute data and keep it resident on Apple MPS.

    Torch is imported here, rather than at module import time, so normal CPU
    bot, backtest, and optimizer startup never depends on the optional GPU
    runtime.
    """

    try:
        import torch
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - exercised without the optional extra
        raise ModuleNotFoundError(
            "Apple MPS optimization requires the optional 'gpu-mps' dependencies; "
            "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
        ) from exc

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if not (len(high) == len(low) == len(close) == len(timestamps)):
        raise ValueError("MPS price and timestamp arrays must have matching lengths")
    if len(close) < 3:
        raise ValueError("MPS proxy requires at least three candles")

    n = len(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_range = np.log(high / low)
    log_range = np.where(np.isfinite(log_range), log_range, 0.0).astype(np.float32)

    first_valid = max(0, run.first_valid_idx)
    hour_log_range, hour_valid = _build_hourly_log_range(
        high, low, timestamps, run
    )

    day_idx = ((timestamps // 86_400_000) - (timestamps[0] // 86_400_000)).astype(
        np.int32
    )
    valid = np.zeros(n, dtype=bool)
    last_valid = min(run.last_valid_idx, n - 1)
    valid[first_valid : last_valid + 1] = (
        np.isfinite(close[first_valid : last_valid + 1])
        & (close[first_valid : last_valid + 1] > 0.0)
        & np.isfinite(high[first_valid : last_valid + 1])
        & np.isfinite(low[first_valid : last_valid + 1])
    )
    indices = np.arange(n, dtype=np.int64)
    can_generate = (
        (indices > max(1, run.warmup_bars))
        & (timestamps[0] + indices * run.interval_ms >= run.guard_ts_ms)
        & (indices >= run.trade_start_idx)
        & valid
    )
    high_fill_max_tick, low_nonfill_max_tick = _strict_fill_tick_boundaries(
        high, low, market.price_step
    )
    touch_down_tick, touch_up_tick, touch_nearest_tick = _directional_touch_ticks(
        close, market.price_step
    )
    touch_min_qty_bits, touch_min_qty_relation = _minimum_entry_qty_encoding(
        close, market
    )

    def tensor(values, *, dtype=None):
        return torch.as_tensor(values, dtype=dtype, device="mps")

    return {
        "high_f": tensor(np.where(np.isfinite(high), high, 0.0).astype(np.float32)),
        "low_f": tensor(np.where(np.isfinite(low), low, 0.0).astype(np.float32)),
        "close_f": tensor(np.where(np.isfinite(close), close, 0.0).astype(np.float32)),
        "log_range": tensor(log_range),
        "hour_log_range": tensor(hour_log_range),
        "hour_valid": tensor(hour_valid),
        "valid": tensor(valid),
        "can_gen": tensor(can_generate),
        "day_idx": tensor(day_idx),
        "high_fill_max_tick": tensor(high_fill_max_tick),
        "low_nonfill_max_tick": tensor(low_nonfill_max_tick),
        "touch_down_tick": tensor(touch_down_tick),
        "touch_up_tick": tensor(touch_up_tick),
        "touch_nearest_tick": tensor(touch_nearest_tick),
        "touch_min_qty_bits": tensor(touch_min_qty_bits),
        "touch_min_qty_relation": tensor(touch_min_qty_relation),
        "n_days": int(day_idx[-1]) + 1,
        "ts0": int(timestamps[0]),
        "times_relative": True,
        "n": n,
    }


def build_mps_multicoin_data(
    hlcvs,
    timestamps_ms,
    runs: list[ProxyRun],
    markets: list[ProxyMarket],
):
    """Pack compact multicoin inputs for the persistent Apple MPS kernel.

    Raw OHLCV stays float32 for unified-memory efficiency. Strict fill and
    order-book touch comparisons are encoded from the original float64 data as
    integer ticks, avoiding the most consequential float32 boundary collapse.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise ModuleNotFoundError(
            "Apple MPS optimization requires the optional 'gpu-mps' dependencies; "
            "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
        ) from exc

    values = np.asarray(hlcvs)
    if values.ndim != 3 or values.shape[2] < 4:
        raise ValueError(
            "MPS multicoin proxy expects HLCVs shaped (candles, coins, >=4); "
            f"got {values.shape}"
        )
    candle_count, coin_count, _channels = values.shape
    if not 2 <= coin_count <= MPS_MULTICOIN_MAX_COINS:
        raise ValueError(
            "MPS multicoin proxy supports 2.."
            f"{MPS_MULTICOIN_MAX_COINS} coins; got {coin_count}"
        )
    if len(runs) != coin_count or len(markets) != coin_count:
        raise ValueError(
            "MPS multicoin run/market settings must match the prepared coin count"
        )
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if len(timestamps) != candle_count:
        raise ValueError(
            f"MPS multicoin timestamps length {len(timestamps)} != {candle_count} candles"
        )
    if candle_count < 3:
        raise ValueError("MPS multicoin proxy requires at least three candles")
    intervals = np.diff(timestamps)
    interval_ms = int(runs[0].interval_ms)
    if np.any(intervals != interval_ms):
        raise ValueError("MPS multicoin proxy requires a continuous candle timeline")
    if interval_ms != 60_000:
        raise ValueError("MPS multicoin proxy currently supports one-minute candles only")

    bars = np.ascontiguousarray(values[:, :, :4], dtype=np.float32)
    bars[~np.isfinite(bars)] = 0.0
    fill_ticks = np.empty((candle_count, coin_count, 2), dtype=np.int32)
    touch_ticks = np.empty((candle_count, coin_count, 2), dtype=np.int32)
    touch_nearest_ticks = np.empty((candle_count, coin_count), dtype=np.int32)
    touch_min_qty_bits = np.empty((candle_count, coin_count), dtype=np.int32)
    touch_min_qty_relation = np.empty((candle_count, coin_count), dtype=np.int32)
    coin_settings = np.empty((coin_count, 11), dtype=np.float32)
    for coin, (run, market) in enumerate(zip(runs, markets)):
        if run.interval_ms != interval_ms:
            raise ValueError("MPS multicoin runs must use one shared candle interval")
        high = values[:, coin, 0].astype(np.float64, copy=False)
        low = values[:, coin, 1].astype(np.float64, copy=False)
        close = values[:, coin, 2].astype(np.float64, copy=False)
        high_fill, low_nonfill = _strict_fill_tick_boundaries(
            high, low, market.price_step
        )
        touch_down, touch_up, touch_nearest = _directional_touch_ticks(
            close, market.price_step
        )
        fill_ticks[:, coin, 0] = high_fill
        fill_ticks[:, coin, 1] = low_nonfill
        touch_ticks[:, coin, 0] = touch_down
        touch_ticks[:, coin, 1] = touch_up
        touch_nearest_ticks[:, coin] = touch_nearest
        min_qty_bits, min_qty_relation = _minimum_entry_qty_encoding(close, market)
        touch_min_qty_bits[:, coin] = min_qty_bits
        touch_min_qty_relation[:, coin] = min_qty_relation
        seed_index = min(max(int(run.first_valid_idx), 0), candle_count - 1)
        seed_close = float(close[seed_index])
        high_seed = float(values[seed_index, coin, 0])
        low_seed = float(values[seed_index, coin, 1])
        volume_seed = max(float(values[seed_index, coin, 3]), 0.0)
        typical_seed = (
            (high_seed + low_seed + seed_close) / 3.0
            if high_seed > 0.0 and low_seed > 0.0 and seed_close > 0.0
            else max(seed_close, 1.0)
        )
        coin_settings[coin] = (
            market.qty_step,
            market.price_step,
            market.min_qty,
            market.min_cost,
            market.c_mult,
            market.maker_fee,
            run.first_valid_idx,
            run.last_valid_idx,
            run.trade_start_idx,
            seed_close if np.isfinite(seed_close) and seed_close > 0.0 else 0.0,
            volume_seed * typical_seed,
        )

    invariant_bytes = (
        bars.nbytes
        + fill_ticks.nbytes
        + touch_ticks.nbytes
        + touch_nearest_ticks.nbytes
        + touch_min_qty_bits.nbytes
        + touch_min_qty_relation.nbytes
    )
    recommended = None
    recommended_fn = getattr(torch.mps, "recommended_max_memory", None)
    if callable(recommended_fn):
        recommended = int(recommended_fn())
    if recommended and invariant_bytes > int(recommended * 0.45):
        raise MemoryError(
            "MPS multicoin invariant tensors would consume "
            f"{invariant_bytes / 2**30:.2f} GiB, above the 45% safety limit of "
            f"the device's {recommended / 2**30:.2f} GiB recommended working set"
        )

    def tensor(array, *, dtype=None):
        return torch.as_tensor(array, dtype=dtype, device="mps").contiguous()

    first_day = int(timestamps[0] // 86_400_000)
    last_day = int(timestamps[-1] // 86_400_000)
    return {
        "bars": tensor(bars, dtype=torch.float32),
        "fill_ticks": tensor(fill_ticks, dtype=torch.int32),
        "touch_ticks": tensor(touch_ticks, dtype=torch.int32),
        "touch_nearest_ticks": tensor(touch_nearest_ticks, dtype=torch.int32),
        "touch_min_qty_bits": tensor(touch_min_qty_bits, dtype=torch.int32),
        "touch_min_qty_relation": tensor(
            touch_min_qty_relation, dtype=torch.int32
        ),
        "coin_settings": tensor(coin_settings, dtype=torch.float32),
        "n": candle_count,
        "n_coins": coin_count,
        "n_days": last_day - first_day + 1,
        "ts0": int(timestamps[0]),
        "start_minute_of_day": int((timestamps[0] // 60_000) % 1440),
        "start_minute_of_hour": int((timestamps[0] // 60_000) % 60),
        "invariant_bytes": invariant_bytes,
    }
