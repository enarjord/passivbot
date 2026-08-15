from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


GAP_BINS = 128
GAP_MAX_MINUTES = 4_000_000.0

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
    guard_ts_ms: int
    first_ts_ms: int
    interval_ms: int
    liquidation_threshold: float
    first_valid_idx: int
    last_valid_idx: int


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

    hour_idx = timestamps // 3_600_000
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = hour_idx[1:] > hour_idx[:-1]
    hour_log_range = np.zeros(n, dtype=np.float32)
    hour_valid = np.zeros(n, dtype=bool)
    last_boundary = 0
    first_valid = max(0, run.first_valid_idx)
    latest = None
    for k in range(1, n):
        if not boundary[k]:
            continue
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
        "n_days": int(day_idx[-1]) + 1,
        "ts0": int(timestamps[0]),
        "times_relative": True,
        "n": n,
    }
