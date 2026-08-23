from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
import torch

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    GAP_BINS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
    trailing_martingale_shader_topology,
)


MPS_DAILY_COLS = 8
MPS_MULTICOIN_DAILY_COLS = 9
MPS_SCALAR_COLS = 32
MPS_MULTICOIN_BASE_SCALAR_COLS = 61
MPS_MULTICOIN_EMA_TAIL_SCALAR_COLS = 63
MPS_MULTICOIN_SCALAR_COLS = 65
MPS_DIRECTIONAL_BASE_SCALAR_COLS = 66
MPS_DIRECTIONAL_EMA_TAIL_SCALAR_COLS = 68
MPS_DIRECTIONAL_SCALAR_COLS = 70
MPS_MULTICOIN_FUSED_BASE_SCALAR_COLS = 66
MPS_MULTICOIN_FUSED_EMA_TAIL_SCALAR_COLS = 68
MPS_MULTICOIN_FUSED_SCALAR_COLS = 70
MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY = 2048
MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS = 7

_HSL_EMA_TAIL_DEFINE = "#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 1\n"
_HSL_RAW_DRAWDOWN_DEFINE = "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n"
_RECOVERY_DISTRIBUTION_DEFINE = (
    "#define PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED 1\n"
)


def _with_hsl_ema_tail(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "#ifndef PASSIVBOT_HSL_EMA_TAIL_ENABLED" not in source:
        raise RuntimeError("MPS source is missing the HSL EMA-tail feature guard")
    return _HSL_EMA_TAIL_DEFINE + source


def _with_hsl_features(
    source: str,
    *,
    ema_tail_enabled: bool,
    raw_drawdown_enabled: bool,
) -> str:
    source = _with_hsl_ema_tail(source, ema_tail_enabled)
    if not raw_drawdown_enabled:
        return source
    if "#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED" not in source:
        raise RuntimeError("MPS source is missing the HSL raw-drawdown feature guard")
    return _HSL_RAW_DRAWDOWN_DEFINE + source


def _with_recovery_distribution(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED" not in source:
        raise RuntimeError(
            "MPS source is missing the strategy-equity recovery-distribution feature guard"
        )
    return _RECOVERY_DISTRIBUTION_DEFINE + source


def _encode_max_realized_loss_pct(value: float) -> float:
    """Encode a float64 loss fraction without loosening its Metal budget."""

    if value >= 1.0:
        return 1.0
    encoded = np.float32(value)
    if float(encoded) > value:
        encoded = np.nextafter(encoded, np.float32(-np.inf))
    return float(encoded)


def _scalar_column_or_zero(scalars, index: int):
    if scalars.shape[1] > index:
        return scalars[:, index]
    return torch.zeros_like(scalars[:, 0])


@lru_cache(maxsize=8)
def _shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        _with_recovery_distribution(
            _with_hsl_features(
                passivbot_rust.mps_ema_anchor_source_py(),
                ema_tail_enabled=hsl_ema_tail_enabled,
                raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            ),
            recovery_distribution_enabled,
        )
    )


@lru_cache(maxsize=8)
def _trailing_martingale_shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        _with_recovery_distribution(
            _with_hsl_features(
                passivbot_rust.mps_trailing_martingale_source_py(),
                ema_tail_enabled=hsl_ema_tail_enabled,
                raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            ),
            recovery_distribution_enabled,
        )
    )


@lru_cache(maxsize=2)
def _trailing_martingale_long_no_hsl_shader_library(
    recovery_distribution_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        _with_recovery_distribution(
            passivbot_rust.mps_trailing_martingale_long_no_hsl_source_py(),
            recovery_distribution_enabled,
        )
    )


@lru_cache(maxsize=2)
def _trailing_martingale_short_no_hsl_shader_library(
    recovery_distribution_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        _with_recovery_distribution(
            passivbot_rust.mps_trailing_martingale_short_no_hsl_source_py(),
            recovery_distribution_enabled,
        )
    )


@lru_cache(maxsize=4)
def _ema_anchor_multicoin_shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        _with_hsl_features(
            passivbot_rust.mps_ema_anchor_multicoin_source_py(),
            ema_tail_enabled=hsl_ema_tail_enabled,
            raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        )
    )


@lru_cache(maxsize=4)
def _trailing_martingale_multicoin_shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        _with_hsl_features(
            passivbot_rust.mps_trailing_martingale_multicoin_source_py(),
            ema_tail_enabled=hsl_ema_tail_enabled,
            raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        )
    )


@lru_cache(maxsize=1)
def _strategy_eq_recovery_distribution_shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        passivbot_rust.mps_strategy_eq_recovery_distribution_source_py()
    )


@lru_cache(maxsize=2)
def _strategy_eq_recovery_distribution_buffers(
    batch_size: int, sample_capacity: int
):
    shape = (int(batch_size), int(sample_capacity))
    return (
        torch.empty(shape, dtype=torch.int32, device="mps"),
        torch.empty(shape, dtype=torch.int32, device="mps"),
        torch.empty(
            (int(batch_size), MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device="mps",
        ),
        torch.tensor(shape, dtype=torch.int32, device="mps"),
    )


def strategy_eq_recovery_distribution_from_samples(
    strategy_equity_samples, *, sample_interval_days: float = 1.0
):
    """Approximate exact recovery summaries from uniformly spaced proxy samples."""

    if strategy_equity_samples.device.type != "mps":
        raise ValueError("strategy-equity recovery distribution requires an MPS tensor")
    if strategy_equity_samples.dtype != torch.float32:
        raise ValueError("strategy-equity recovery distribution requires float32 input")
    if strategy_equity_samples.ndim != 2:
        raise ValueError(
            "strategy-equity recovery distribution expects a batch-by-day matrix"
        )
    sample_interval_days = float(sample_interval_days)
    if not np.isfinite(sample_interval_days) or sample_interval_days <= 0.0:
        raise ValueError("recovery sample interval must be finite and positive")
    matrix = strategy_equity_samples.contiguous()
    batch_size, sample_capacity = (int(value) for value in matrix.shape)
    if batch_size == 0:
        return torch.empty(
            (0, MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device="mps",
        )
    if sample_capacity == 0:
        return torch.zeros(
            (batch_size, MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device="mps",
        )
    stack, histogram, output, sizes = _strategy_eq_recovery_distribution_buffers(
        batch_size, sample_capacity
    )
    histogram.zero_()
    library = _strategy_eq_recovery_distribution_shader_library()
    library.passivbot_strategy_eq_recovery_distribution(
        matrix,
        stack,
        histogram,
        output,
        sizes,
        threads=(batch_size, 1, 1),
    )
    return output * sample_interval_days


def _decode_outputs(daily, scalars, gaps) -> dict:
    active_days = torch.isfinite(daily[:, :, 1]) & (daily[:, :, 1] < float("inf"))

    def timestamp_column(index: int):
        values = scalars[:, index]
        return torch.where(
            values >= 0.0, values, torch.full_like(values, float("nan"))
        )

    return {
        "day_end_eq": daily[:, :, 0],
        "day_min_eq": torch.where(
            active_days,
            daily[:, :, 1],
            torch.full_like(daily[:, :, 1], float("inf")),
        ),
        "day_max_dd": daily[:, :, 2],
        "day_volume": daily[:, :, 3],
        "day_has_fill": daily[:, :, 4] > 0.0,
        "day_min_balance": torch.where(
            active_days,
            daily[:, :, 5],
            torch.full_like(daily[:, :, 5], float("inf")),
        ),
        "day_net_pnl": daily[:, :, 6],
        "day_last_fill_balance": daily[:, :, 7],
        "day_fill_count": daily[:, :, 8],
        "max_dd": scalars[:, 0],
        "held_max_ms": scalars[:, 1],
        "gap_hist": gaps,
        "gap_max_ms": scalars[:, 2],
        "first_fill_ts": timestamp_column(3),
        "last_fill_ts": timestamp_column(4),
        "recovery_max_ms": scalars[:, 5],
        "last_high_ts": timestamp_column(6),
        "first_eq_ts": timestamp_column(7),
        "last_eq_ts": timestamp_column(8),
        "liq_step": scalars[:, 9].to(torch.int64),
        "balance": scalars[:, 10],
        "psize": scalars[:, 11],
        "pprice": scalars[:, 12],
        "alive": scalars[:, 13] > 0.0,
        "open_positions": scalars[:, 14],
        "short_psize": scalars[:, 15],
        "short_pprice": scalars[:, 16],
        "profit_sum": scalars[:, 18],
        "loss_sum": scalars[:, 19],
        "position_unchanged_max_ms": scalars[:, 20],
        "entry_initial_balance_pct": scalars[:, 21],
        "total_wallet_exposure_max": scalars[:, 22],
        "total_wallet_exposure_mean": scalars[:, 23],
        "fill_count": scalars[:, 24],
        "fill_count_entry": scalars[:, 25],
        "fill_count_long": scalars[:, 26],
        "fills_active_days_count": scalars[:, 27],
        "pnl_recovery_max_ms": scalars[:, 28],
        "held_sum_ms": scalars[:, 29],
        "held_count": scalars[:, 30],
        "account_recovery_max_ms": scalars[:, 31],
        "hsl_long_enabled": scalars[:, 32] > 0.0,
        "hsl_short_enabled": scalars[:, 33] > 0.0,
        "hsl_triggers_long": scalars[:, 34],
        "hsl_triggers_short": scalars[:, 35],
        "hsl_restarts_long": scalars[:, 36],
        "hsl_restarts_short": scalars[:, 37],
        "hsl_tier_samples_total": scalars[:, 38],
        "hsl_tier_samples_yellow": scalars[:, 39],
        "hsl_tier_samples_orange": scalars[:, 40],
        "hsl_tier_samples_red": scalars[:, 41],
        "hsl_duration_sum_steps": scalars[:, 42],
        "hsl_duration_max_steps": scalars[:, 43],
        "hsl_duration_count": scalars[:, 44],
        "hsl_trigger_drawdown_sum": scalars[:, 45],
        "hsl_trigger_drawdown_count": scalars[:, 46],
        "hsl_flatten_time_sum_steps": scalars[:, 47],
        "hsl_flatten_time_count": scalars[:, 48],
        "hsl_restart_retrigger_count": scalars[:, 49],
        "hsl_halt_to_restart_equity_loss": scalars[:, 50],
        "hsl_panic_close_loss_sum": scalars[:, 51],
        "hsl_panic_close_loss_max": scalars[:, 52],
        "hsl_panic_loss_drawdown_min": scalars[:, 53],
        "hsl_panic_loss_drawdown_sum": scalars[:, 54],
        "hsl_panic_loss_drawdown_max": scalars[:, 55],
        "hsl_panic_loss_drawdown_count": scalars[:, 56],
        "hsl_drawdown_ema_max_long": scalars[:, 57],
        "hsl_drawdown_ema_max_short": scalars[:, 58],
        "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 59],
        "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 60],
        "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 61
        ),
        "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 62
        ),
        "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 63),
        "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 64),
    }


def _decode_multicoin_fused_outputs(daily, scalars, gaps) -> dict:
    output = _decode_outputs(daily, scalars, gaps)
    long_entry_initial_balance_pct = output.pop("entry_initial_balance_pct")
    output.update(
        {
            "entry_initial_balance_pct_long": long_entry_initial_balance_pct,
            "entry_initial_balance_pct_short": scalars[:, 59],
            "profit_sum_long": scalars[:, 60],
            "loss_sum_long": scalars[:, 61],
            "profit_sum_short": scalars[:, 62],
            "loss_sum_short": scalars[:, 63],
            "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 64],
            "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 65],
            "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
                scalars, 66
            ),
            "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
                scalars, 67
            ),
            "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 68),
            "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 69),
        }
    )
    return output


def _decode_directional_outputs(daily, scalars, gaps) -> dict:
    active_days = torch.isfinite(daily[:, :, 1]) & (daily[:, :, 1] < float("inf"))

    def timestamp_column(index: int):
        values = scalars[:, index]
        return torch.where(
            values >= 0.0, values, torch.full_like(values, float("nan"))
        )

    return {
        "day_end_eq": daily[:, :, 0],
        "day_min_eq": torch.where(
            active_days,
            daily[:, :, 1],
            torch.full_like(daily[:, :, 1], float("inf")),
        ),
        "day_max_dd": daily[:, :, 2],
        "day_volume": daily[:, :, 3],
        "day_has_fill": daily[:, :, 4] > 0.0,
        "day_net_pnl": daily[:, :, 5],
        "day_last_fill_balance": daily[:, :, 6],
        "day_fill_count": daily[:, :, 7],
        "max_dd": scalars[:, 0],
        "held_max_ms": scalars[:, 1],
        "gap_hist": gaps,
        "gap_max_ms": scalars[:, 2],
        "first_fill_ts": timestamp_column(3),
        "last_fill_ts": timestamp_column(4),
        "recovery_max_ms": scalars[:, 5],
        "last_high_ts": timestamp_column(6),
        "first_eq_ts": timestamp_column(7),
        "last_eq_ts": timestamp_column(8),
        "liq_step": scalars[:, 9].to(torch.int64),
        "balance": scalars[:, 10],
        "psize": scalars[:, 11],
        "pprice": scalars[:, 12],
        "alive": scalars[:, 13] > 0.0,
        "short_psize": scalars[:, 15],
        "short_pprice": scalars[:, 16],
        "hsl_long_enabled": scalars[:, 18] > 0.0,
        "hsl_short_enabled": scalars[:, 19] > 0.0,
        "hsl_triggers_long": scalars[:, 20],
        "hsl_triggers_short": scalars[:, 21],
        "hsl_restarts_long": scalars[:, 22],
        "hsl_restarts_short": scalars[:, 23],
        "hsl_tier_samples_total": scalars[:, 24],
        "hsl_tier_samples_yellow": scalars[:, 25],
        "hsl_tier_samples_orange": scalars[:, 26],
        "hsl_tier_samples_red": scalars[:, 27],
        "hsl_duration_sum_steps": scalars[:, 28],
        "hsl_duration_max_steps": scalars[:, 29],
        "hsl_duration_count": scalars[:, 30],
        "hsl_trigger_drawdown_sum": scalars[:, 31],
        "hsl_trigger_drawdown_count": scalars[:, 32],
        "hsl_flatten_time_sum_steps": scalars[:, 33],
        "hsl_flatten_time_count": scalars[:, 34],
        "hsl_restart_retrigger_count": scalars[:, 35],
        "hsl_halt_to_restart_equity_loss": scalars[:, 36],
        "hsl_panic_close_loss_sum": scalars[:, 37],
        "hsl_panic_close_loss_max": scalars[:, 38],
        "hsl_panic_loss_drawdown_min": scalars[:, 39],
        "hsl_panic_loss_drawdown_sum": scalars[:, 40],
        "hsl_panic_loss_drawdown_max": scalars[:, 41],
        "hsl_panic_loss_drawdown_count": scalars[:, 42],
        "profit_sum": scalars[:, 43],
        "loss_sum": scalars[:, 44],
        "position_unchanged_max_ms": scalars[:, 45],
        "entry_initial_balance_pct_long": scalars[:, 46],
        "entry_initial_balance_pct_short": scalars[:, 47],
        "total_wallet_exposure_max": scalars[:, 48],
        "total_wallet_exposure_mean": scalars[:, 49],
        "fill_count": scalars[:, 50],
        "fill_count_entry": scalars[:, 51],
        "fill_count_long": scalars[:, 52],
        "fills_active_days_count": scalars[:, 53],
        "pnl_recovery_max_ms": scalars[:, 54],
        "held_sum_ms": scalars[:, 55],
        "held_count": scalars[:, 56],
        "account_recovery_max_ms": scalars[:, 57],
        "profit_sum_long": scalars[:, 58],
        "loss_sum_long": scalars[:, 59],
        "profit_sum_short": scalars[:, 60],
        "loss_sum_short": scalars[:, 61],
        "hsl_drawdown_ema_max_long": scalars[:, 62],
        "hsl_drawdown_ema_max_short": scalars[:, 63],
        "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 64],
        "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 65],
        "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 66
        ),
        "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 67
        ),
        "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 68),
        "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 69),
    }


class MpsEmaAnchorRunner:
    """Persistent single-coin Metal runner with invariant data resident on MPS."""

    def __init__(
        self,
        market: ProxyMarket,
        run: ProxyRun,
        data: dict,
        *,
        long_enabled: bool = True,
        short_enabled: bool = False,
        hedge_mode: bool = True,
        filter_by_min_effective_cost: bool = False,
        max_realized_loss_pct: float = 1.0,
        taker_fee: float | None = None,
        market_order_slippage_pct: float = 0.0,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        pnl_lookback_bars: int = 0,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
    ):
        self.market = market
        self.run_config = run
        self.long_enabled = bool(long_enabled)
        self.short_enabled = bool(short_enabled)
        self.hedge_mode = bool(hedge_mode)
        if not self.long_enabled and not self.short_enabled:
            raise ValueError("MPS EMA proxy requires at least one enabled side")
        max_realized_loss_pct = float(max_realized_loss_pct)
        if not np.isfinite(max_realized_loss_pct) or max_realized_loss_pct < 0.0:
            raise ValueError(
                "max_realized_loss_pct must be finite and non-negative"
            )
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        taker_fee = market.maker_fee if taker_fee is None else float(taker_fee)
        market_order_slippage_pct = float(market_order_slippage_pct)
        pnl_lookback_bars = int(pnl_lookback_bars)
        if not np.isfinite(taker_fee):
            raise ValueError("taker_fee must be finite")
        if (
            not np.isfinite(market_order_slippage_pct)
            or market_order_slippage_pct < 0.0
        ):
            raise ValueError(
                "market_order_slippage_pct must be finite and non-negative"
            )
        if pnl_lookback_bars < 0:
            raise ValueError("pnl_lookback_bars must be non-negative")
        self.pnl_lookback_bars = pnl_lookback_bars
        self.hsl_ema_tail_enabled = bool(hsl_ema_tail_enabled)
        self.hsl_raw_drawdown_enabled = bool(hsl_raw_drawdown_enabled)
        self.recovery_distribution_enabled = bool(recovery_distribution_enabled)
        self.rolling_capacity = (
            MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY
            if pnl_lookback_bars > 0
            else 1
        )
        self.n = int(data["n"])
        self.n_days = int(data["n_days"])
        self.recovery_stride = (
            max(1, int(np.ceil(3_600_000.0 / float(run.interval_ms))))
            if self.recovery_distribution_enabled
            else 0
        )
        self.n_recovery_samples = (
            max(1, (self.n + self.recovery_stride - 1) // self.recovery_stride)
            if self.recovery_distribution_enabled
            else 1
        )
        self.bars = (
            torch.stack(
                [
                    data["high_f"],
                    data["low_f"],
                    data["close_f"],
                    data["log_range"],
                    data["hour_log_range"],
                ],
                dim=1,
            )
            .to(dtype=torch.float32, device="mps")
            .contiguous()
        )
        self.flags = (
            torch.stack(
                [
                    data["valid"].to(torch.int32),
                    data["can_gen"].to(torch.int32),
                    data["day_idx"].to(torch.int32),
                    data["hour_valid"].to(torch.int32),
                    data["high_fill_max_tick"].to(torch.int32),
                    data["low_nonfill_max_tick"].to(torch.int32),
                    data["touch_down_tick"].to(torch.int32),
                    data["touch_up_tick"].to(torch.int32),
                    data["touch_nearest_tick"].to(torch.int32),
                    data["touch_min_qty_bits"].to(torch.int32),
                    data["touch_min_qty_relation"].to(torch.int32),
                ],
                dim=1,
            )
            .to(device="mps")
            .contiguous()
        )
        liq_floor = max(0.0, run.starting_balance) * max(0.0, run.liquidation_threshold)
        self.settings = torch.tensor(
            [
                market.qty_step,
                market.price_step,
                market.min_qty,
                market.min_cost,
                market.c_mult,
                market.maker_fee,
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                float(self.long_enabled),
                float(self.short_enabled),
                float(self.hedge_mode),
                float(bool(filter_by_min_effective_cost)),
                data["max_effective_min_cost"],
                encoded_max_realized_loss_pct,
                taker_fee,
                market_order_slippage_pct,
                float(bool(hsl_panic_market_long)),
                float(bool(hsl_panic_market_short)),
            ],
            dtype=torch.float32,
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._rolling_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._recovery_buffers: dict[int, torch.Tensor] = {}
        self._sizes: dict[tuple[int, int], torch.Tensor] = {}
        self.last_profile: dict[str, float] = {}

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                f"expected directional EMA parameter matrix with {expected} columns, got {got}"
            )
        return np.ascontiguousarray(params, dtype=np.float32)

    def _output_buffers(self, batch_size: int):
        if batch_size not in self._buffers:
            # Optimizer generations use one fixed batch size.  Keep only the
            # active allocation so benchmark/tuning calls with several sizes
            # do not retain every large daily-output buffer.
            self._buffers = {
                batch_size: (
                    torch.zeros(
                        (batch_size, self.n_days, MPS_DAILY_COLS),
                        dtype=torch.float32,
                        device="mps",
                    ),
                    torch.zeros(
                        (
                            batch_size,
                            MPS_DIRECTIONAL_SCALAR_COLS
                            if self.hsl_raw_drawdown_enabled
                            else (
                                MPS_DIRECTIONAL_EMA_TAIL_SCALAR_COLS
                                if self.hsl_ema_tail_enabled
                                else MPS_DIRECTIONAL_BASE_SCALAR_COLS
                            ),
                        ),
                        dtype=torch.float32,
                        device="mps",
                    ),
                    torch.zeros(
                        (batch_size, GAP_BINS), dtype=torch.int32, device="mps"
                    ),
                )
            }
        else:
            for buffer in self._buffers[batch_size]:
                buffer.zero_()
        # An untouched day has no valid equity sample. The kernel overwrites
        # this sentinel whenever it flushes an active day.
        self._buffers[batch_size][0][:, :, 1].fill_(float("inf"))
        return self._buffers[batch_size]

    def _hsl_rolling_buffers(self, batch_size: int):
        if batch_size not in self._rolling_buffers:
            # The kernel overwrites every active ring/deque slot.  Avoid
            # zeroing this large scratch allocation between generations.
            shape = (batch_size, 2, self.rolling_capacity, 2)
            self._rolling_buffers = {
                batch_size: (
                    torch.empty(shape, dtype=torch.float32, device="mps"),
                    torch.empty(shape, dtype=torch.int32, device="mps"),
                )
            }
        return self._rolling_buffers[batch_size]

    def _recovery_sample_buffer(self, batch_size: int):
        if batch_size not in self._recovery_buffers:
            self._recovery_buffers = {
                batch_size: torch.full(
                    (batch_size, self.n_recovery_samples),
                    float("nan"),
                    dtype=torch.float32,
                    device="mps",
                )
            }
        else:
            self._recovery_buffers[batch_size].fill_(float("nan"))
        return self._recovery_buffers[batch_size]

    def run(self, params: np.ndarray, *, profile: bool = False) -> dict:
        started = time.perf_counter()
        matrix = self._pack_params(params)
        packed = time.perf_counter()
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        rolling_pnl_values, rolling_pnl_indices = self._hsl_rolling_buffers(
            batch_size
        )
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
        sizes_key = (batch_size, int(matrix.shape[1]))
        if sizes_key not in self._sizes:
            size_values = [
                batch_size,
                self.n,
                self.n_days,
                matrix.shape[1],
                self.run_config.first_valid_idx,
                self.rolling_capacity,
                self.pnl_lookback_bars,
            ]
            if self.recovery_distribution_enabled:
                size_values.extend(
                    [self.recovery_stride, self.n_recovery_samples]
                )
            self._sizes[sizes_key] = torch.tensor(
                size_values,
                dtype=torch.int32,
                device="mps",
            )
        prepared = time.perf_counter()
        library = _shader_library(
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.recovery_distribution_enabled,
        )
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        kernel_args = (
            self.bars,
            self.flags,
            params_mps,
            self.settings,
            self._sizes[sizes_key],
            daily,
            scalars,
            gaps,
            rolling_pnl_values,
            rolling_pnl_indices,
        )
        if self.recovery_distribution_enabled:
            kernel_args += (recovery_samples,)
        library.passivbot_ema_anchor(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )
        if profile:
            torch.mps.synchronize()
        finished = time.perf_counter()
        self.last_profile = {
            "cpu_pack_seconds": packed - started,
            "upload_and_zero_seconds": prepared - packed,
            "compile_seconds": compiled - prepared,
            "pre_dispatch_sync_seconds": dispatched - compiled,
            "kernel_seconds": finished - dispatched,
        }
        output = _decode_directional_outputs(daily, scalars, gaps)
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        return output


class MpsEmaAnchorMulticoinRunner:
    """Persistent single-side multi-coin EMA Anchor screening runner on MPS."""

    coin_override_cols = 29
    coin_override_label = "EMA"
    scalar_cols = MPS_MULTICOIN_SCALAR_COLS

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        side: str,
        coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        market_order_slippage_pct: float = 0.0,
        hsl_panic_market: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
    ):
        if side not in {"long", "short"}:
            raise ValueError(
                f"MPS multicoin runner side must be long or short, got {side!r}"
            )
        self.side = side
        self.collect_coin_fill_counts = bool(collect_coin_fill_counts)
        self.hsl_ema_tail_enabled = bool(hsl_ema_tail_enabled)
        self.hsl_raw_drawdown_enabled = bool(hsl_raw_drawdown_enabled)
        fused = self.scalar_cols == MPS_MULTICOIN_FUSED_SCALAR_COLS
        if self.hsl_raw_drawdown_enabled:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_SCALAR_COLS
            )
        elif self.hsl_ema_tail_enabled:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_EMA_TAIL_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_EMA_TAIL_SCALAR_COLS
            )
        else:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_BASE_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_BASE_SCALAR_COLS
            )
        self.run_config = run
        self.n = int(data["n"])
        self.n_coins = int(data["n_coins"])
        self.n_days = int(data["n_days"])
        self.bars = data["bars"]
        self.fill_ticks = data["fill_ticks"]
        self.touch_ticks = data["touch_ticks"]
        self.touch_nearest_ticks = data["touch_nearest_ticks"]
        self.touch_min_qty_bits = data["touch_min_qty_bits"]
        self.touch_min_qty_relation = data["touch_min_qty_relation"]
        self.coin_settings = data["coin_settings"]
        if coin_overrides is None:
            coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols), np.nan, dtype=np.float32
            )
        coin_overrides = np.asarray(coin_overrides, dtype=np.float32)
        if coin_overrides.shape != (self.n_coins, self.coin_override_cols):
            raise ValueError(
                f"expected multicoin {self.coin_override_label} override matrix shaped "
                f"({self.n_coins}, {self.coin_override_cols}), "
                f"got {coin_overrides.shape}"
            )
        self.coin_overrides = torch.as_tensor(
            np.ascontiguousarray(coin_overrides), device="mps"
        )
        forager_score_hysteresis_pct = float(forager_score_hysteresis_pct)
        if not np.isfinite(forager_score_hysteresis_pct) or (
            forager_score_hysteresis_pct < 0.0
        ):
            raise ValueError(
                "forager_score_hysteresis_pct must be finite and non-negative"
            )
        max_realized_loss_pct = float(max_realized_loss_pct)
        if not np.isfinite(max_realized_loss_pct) or max_realized_loss_pct < 0.0:
            raise ValueError(
                "max_realized_loss_pct must be finite and non-negative"
            )
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        market_order_slippage_pct = float(market_order_slippage_pct)
        if (
            not np.isfinite(market_order_slippage_pct)
            or market_order_slippage_pct < 0.0
        ):
            raise ValueError(
                "market_order_slippage_pct must be finite and non-negative"
            )
        liq_floor = max(0.0, run.starting_balance) * max(
            0.0, run.liquidation_threshold
        )
        self.settings = torch.tensor(
            [
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                float(side == "short"),
                forager_score_hysteresis_pct,
                encoded_max_realized_loss_pct,
                float(self.collect_coin_fill_counts),
                market_order_slippage_pct,
                float(bool(hsl_panic_market)),
            ],
            dtype=torch.float32,
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self._sizes: dict[tuple[int, int], torch.Tensor] = {}
        self._full_end_steps: dict[int, torch.Tensor] = {}
        self.last_profile: dict[str, float] = {}
        self.start_minute_of_day = int(data["start_minute_of_day"])
        self.start_minute_of_hour = int(data["start_minute_of_hour"])
        self.requested_start_idx = max(
            0,
            int(
                (run.guard_ts_ms - int(data["ts0"]) + run.interval_ms - 1)
                // run.interval_ms
            ),
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                f"expected multicoin EMA parameter matrix with {expected} columns, got {got}"
            )
        return np.ascontiguousarray(params, dtype=np.float32)

    def _output_buffers(self, batch_size: int):
        if batch_size not in self._buffers:
            self._buffers = {
                batch_size: (
                    torch.zeros(
                        (batch_size, self.n_days, MPS_MULTICOIN_DAILY_COLS),
                        dtype=torch.float32,
                        device="mps",
                    ),
                    torch.zeros(
                        (batch_size, self.scalar_cols),
                        dtype=torch.float32,
                        device="mps",
                    ),
                    torch.zeros(
                        (batch_size, GAP_BINS), dtype=torch.int32, device="mps"
                    ),
                    torch.zeros(
                        (batch_size, self.n_coins)
                        if self.collect_coin_fill_counts
                        else (1,),
                        dtype=torch.float32,
                        device="mps",
                    ),
                )
            }
        else:
            for buffer in self._buffers[batch_size]:
                buffer.zero_()
        self._buffers[batch_size][0][:, :, 1].fill_(float("inf"))
        self._buffers[batch_size][0][:, :, 5].fill_(float("inf"))
        return self._buffers[batch_size]

    def _end_steps(self, end_steps: np.ndarray | None, batch_size: int):
        if end_steps is None:
            if batch_size not in self._full_end_steps:
                self._full_end_steps[batch_size] = torch.full(
                    (batch_size,), self.n - 1, dtype=torch.int32, device="mps"
                )
            return self._full_end_steps[batch_size]
        values = np.asarray(end_steps, dtype=np.int32)
        if values.shape != (batch_size,):
            raise ValueError(
                f"expected one multi-coin end step per candidate, got {values.shape}"
            )
        values = np.clip(values, 1, self.n - 1)
        return torch.as_tensor(
            np.ascontiguousarray(values), dtype=torch.int32, device="mps"
        )

    def _dispatch(
        self,
        library,
        params_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        *,
        batch_size: int,
    ) -> None:
        library.passivbot_ema_anchor_multicoin(
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.coin_settings,
            self.coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
            daily,
            scalars,
            gaps,
            coin_fill_counts,
            threads=(batch_size, 1, 1),
        )

    def _library(self):
        return _ema_anchor_multicoin_shader_library(
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_outputs(daily, scalars, gaps)

    def run(
        self,
        params: np.ndarray,
        *,
        profile: bool = False,
        end_steps: np.ndarray | None = None,
    ) -> dict:
        started = time.perf_counter()
        matrix = self._pack_params(params)
        packed = time.perf_counter()
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        end_steps_mps = self._end_steps(end_steps, batch_size)
        daily, scalars, gaps, coin_fill_counts = self._output_buffers(batch_size)
        sizes_key = (batch_size, int(matrix.shape[1]))
        if sizes_key not in self._sizes:
            self._sizes[sizes_key] = torch.tensor(
                [
                    batch_size,
                    self.n,
                    self.n_coins,
                    self.n_days,
                    self.requested_start_idx,
                    self.run_config.warmup_bars,
                    self.start_minute_of_day,
                    self.start_minute_of_hour,
                ],
                dtype=torch.int32,
                device="mps",
            )
        prepared = time.perf_counter()
        library = self._library()
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        self._dispatch(
            library,
            params_mps,
            self._sizes[sizes_key],
            end_steps_mps,
            daily,
            scalars,
            gaps,
            coin_fill_counts,
            batch_size=batch_size,
        )
        if profile:
            torch.mps.synchronize()
        finished = time.perf_counter()
        self.last_profile = {
            "cpu_pack_seconds": packed - started,
            "upload_and_zero_seconds": prepared - packed,
            "compile_seconds": compiled - prepared,
            "pre_dispatch_sync_seconds": dispatched - compiled,
            "kernel_seconds": finished - dispatched,
        }
        output = self._decode(daily, scalars, gaps)
        if self.collect_coin_fill_counts:
            output["coin_fill_counts"] = coin_fill_counts
        return output


class MpsEmaAnchorMulticoinFusedRunner(MpsEmaAnchorMulticoinRunner):
    """Persistent dual-side shared-account EMA Anchor runner on Apple MPS."""

    scalar_cols = MPS_MULTICOIN_FUSED_SCALAR_COLS

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        long_coin_overrides: np.ndarray | None = None,
        short_coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        market_order_slippage_pct: float = 0.0,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
    ):
        super().__init__(
            run,
            data,
            side="long",
            coin_overrides=long_coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            market_order_slippage_pct=market_order_slippage_pct,
            hsl_panic_market=hsl_panic_market_long,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        )
        if short_coin_overrides is None:
            short_coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols),
                np.nan,
                dtype=np.float32,
            )
        short_coin_overrides = np.asarray(short_coin_overrides, dtype=np.float32)
        expected_shape = (self.n_coins, self.coin_override_cols)
        if short_coin_overrides.shape != expected_shape:
            raise ValueError(
                "expected fused multicoin EMA short override matrix shaped "
                f"{expected_shape}, got {short_coin_overrides.shape}"
            )
        self.short_coin_overrides = torch.as_tensor(
            np.ascontiguousarray(short_coin_overrides), device="mps"
        )
        max_realized_loss_pct = float(max_realized_loss_pct)
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        market_order_slippage_pct = float(market_order_slippage_pct)
        liq_floor = max(0.0, run.starting_balance) * max(
            0.0, run.liquidation_threshold
        )
        self.settings = torch.tensor(
            [
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                0.0,
                float(forager_score_hysteresis_pct),
                encoded_max_realized_loss_pct,
                float(self.collect_coin_fill_counts),
                market_order_slippage_pct,
                float(bool(hsl_panic_market_long)),
                float(bool(hsl_panic_market_short)),
            ],
            dtype=torch.float32,
            device="mps",
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected fused multicoin EMA parameter matrix with "
                f"{expected} columns, got {got}"
            )
        return np.ascontiguousarray(params, dtype=np.float32)

    def _dispatch(
        self,
        library,
        params_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        *,
        batch_size: int,
    ) -> None:
        library.passivbot_ema_anchor_multicoin_fused(
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.coin_settings,
            self.coin_overrides,
            self.short_coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
            daily,
            scalars,
            gaps,
            coin_fill_counts,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps)


class MpsEmaAnchorMulticoinLongRunner(MpsEmaAnchorMulticoinRunner):
    """Compatibility wrapper for the original long-only multicoin runner."""

    def __init__(self, run: ProxyRun, data: dict):
        super().__init__(run, data, side="long")


class MpsEmaAnchorMulticoinShortRunner(MpsEmaAnchorMulticoinRunner):
    """Short-only multicoin EMA Anchor screening runner."""

    def __init__(self, run: ProxyRun, data: dict):
        super().__init__(run, data, side="short")


class MpsTrailingMartingaleMulticoinRunner(MpsEmaAnchorMulticoinRunner):
    """Persistent single-side multi-coin Trailing Martingale proxy on MPS."""

    coin_override_cols = 44
    coin_override_label = "Trailing Martingale"

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        side: str,
        coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        market_order_slippage_pct: float = 0.0,
        hsl_panic_market: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
    ):
        super().__init__(
            run,
            data,
            side=side,
            coin_overrides=coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            market_order_slippage_pct=market_order_slippage_pct,
            hsl_panic_market=hsl_panic_market,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected multicoin Trailing Martingale parameter matrix with "
                f"{expected} columns, got {got}"
            )
        return np.ascontiguousarray(params, dtype=np.float32)

    def _library(self):
        return _trailing_martingale_multicoin_shader_library(
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
        )

    def _dispatch(
        self,
        library,
        params_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        *,
        batch_size: int,
    ) -> None:
        library.passivbot_trailing_martingale_multicoin(
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.touch_nearest_ticks,
            self.touch_min_qty_bits,
            self.touch_min_qty_relation,
            self.coin_settings,
            self.coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
            daily,
            scalars,
            gaps,
            coin_fill_counts,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_outputs(daily, scalars, gaps)


class MpsTrailingMartingaleMulticoinFusedRunner(
    MpsTrailingMartingaleMulticoinRunner
):
    """Persistent dual-side shared-account Trailing Martingale runner on MPS."""

    scalar_cols = MPS_MULTICOIN_FUSED_SCALAR_COLS

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        long_coin_overrides: np.ndarray | None = None,
        short_coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        market_order_slippage_pct: float = 0.0,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
    ):
        super().__init__(
            run,
            data,
            side="long",
            coin_overrides=long_coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            market_order_slippage_pct=market_order_slippage_pct,
            hsl_panic_market=hsl_panic_market_long,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        )
        if short_coin_overrides is None:
            short_coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols),
                np.nan,
                dtype=np.float32,
            )
        short_coin_overrides = np.asarray(short_coin_overrides, dtype=np.float32)
        expected_shape = (self.n_coins, self.coin_override_cols)
        if short_coin_overrides.shape != expected_shape:
            raise ValueError(
                "expected fused multicoin Trailing Martingale short override "
                f"matrix shaped {expected_shape}, got {short_coin_overrides.shape}"
            )
        self.short_coin_overrides = torch.as_tensor(
            np.ascontiguousarray(short_coin_overrides), device="mps"
        )
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            float(max_realized_loss_pct)
        )
        liq_floor = max(0.0, run.starting_balance) * max(
            0.0, run.liquidation_threshold
        )
        self.settings = torch.tensor(
            [
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                0.0,
                float(forager_score_hysteresis_pct),
                encoded_max_realized_loss_pct,
                float(self.collect_coin_fill_counts),
                float(market_order_slippage_pct),
                float(bool(hsl_panic_market_long)),
                float(bool(hsl_panic_market_short)),
            ],
            dtype=torch.float32,
            device="mps",
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected fused multicoin Trailing Martingale parameter matrix "
                f"with {expected} columns, got {got}"
            )
        return np.ascontiguousarray(params, dtype=np.float32)

    def _dispatch(
        self,
        library,
        params_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        *,
        batch_size: int,
    ) -> None:
        library.passivbot_trailing_martingale_multicoin_fused(
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.touch_nearest_ticks,
            self.touch_min_qty_bits,
            self.touch_min_qty_relation,
            self.coin_settings,
            self.coin_overrides,
            self.short_coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
            daily,
            scalars,
            gaps,
            coin_fill_counts,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps)


class MpsTrailingMartingaleRunner(MpsEmaAnchorRunner):
    """Persistent single-coin trailing-martingale runner on Apple MPS."""

    def __init__(self, *args, hsl_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.shader_topology = trailing_martingale_shader_topology(
            long_enabled=self.long_enabled,
            short_enabled=self.short_enabled,
            hsl_enabled=bool(hsl_enabled),
        )
        # The specialized no-HSL kernels retain the original 66-column ABI.
        # Every EMA-tail metric is identically zero when HSL is disabled, so
        # keep that faster topology and let the decoder synthesize zeroes.
        if self.shader_topology != "generic":
            self.hsl_ema_tail_enabled = False
            self.hsl_raw_drawdown_enabled = False

    def _shader_library(self):
        if self.shader_topology == "long_no_hsl":
            return _trailing_martingale_long_no_hsl_shader_library(
                self.recovery_distribution_enabled
            )
        if self.shader_topology == "short_no_hsl":
            return _trailing_martingale_short_no_hsl_shader_library(
                self.recovery_distribution_enabled
            )
        return _trailing_martingale_shader_library(
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.recovery_distribution_enabled,
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected directional trailing-martingale parameter matrix with "
                f"{expected} columns, got {got}"
            )
        return np.ascontiguousarray(params, dtype=np.float32)

    def run(self, params: np.ndarray, *, profile: bool = False) -> dict:
        started = time.perf_counter()
        matrix = self._pack_params(params)
        packed = time.perf_counter()
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        rolling_pnl_values, rolling_pnl_indices = self._hsl_rolling_buffers(
            batch_size
        )
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
        sizes_key = (batch_size, int(matrix.shape[1]))
        if sizes_key not in self._sizes:
            size_values = [
                batch_size,
                self.n,
                self.n_days,
                matrix.shape[1],
                self.run_config.first_valid_idx,
                self.rolling_capacity,
                self.pnl_lookback_bars,
            ]
            if self.recovery_distribution_enabled:
                size_values.extend(
                    [self.recovery_stride, self.n_recovery_samples]
                )
            self._sizes[sizes_key] = torch.tensor(
                size_values,
                dtype=torch.int32,
                device="mps",
            )
        prepared = time.perf_counter()
        library = self._shader_library()
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        kernel_args = (
            self.bars,
            self.flags,
            params_mps,
            self.settings,
            self._sizes[sizes_key],
            daily,
            scalars,
            gaps,
            rolling_pnl_values,
            rolling_pnl_indices,
        )
        if self.recovery_distribution_enabled:
            kernel_args += (recovery_samples,)
        library.passivbot_trailing_martingale(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )
        if profile:
            torch.mps.synchronize()
        finished = time.perf_counter()
        self.last_profile = {
            "cpu_pack_seconds": packed - started,
            "upload_and_zero_seconds": prepared - packed,
            "compile_seconds": compiled - prepared,
            "pre_dispatch_sync_seconds": dispatched - compiled,
            "kernel_seconds": finished - dispatched,
        }
        output = _decode_directional_outputs(daily, scalars, gaps)
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        return output
