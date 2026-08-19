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
)


MPS_DAILY_COLS = 5
MPS_MULTICOIN_DAILY_COLS = 6
MPS_SCALAR_COLS = 18


@lru_cache(maxsize=1)
def _shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(passivbot_rust.mps_ema_anchor_source_py())


@lru_cache(maxsize=1)
def _trailing_martingale_shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py()
    )


@lru_cache(maxsize=1)
def _ema_anchor_multicoin_shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py()
    )


@lru_cache(maxsize=1)
def _trailing_martingale_multicoin_shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    )


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
    ):
        self.market = market
        self.run_config = run
        self.long_enabled = bool(long_enabled)
        self.short_enabled = bool(short_enabled)
        self.hedge_mode = bool(hedge_mode)
        if not self.long_enabled and not self.short_enabled:
            raise ValueError("MPS EMA proxy requires at least one enabled side")
        self.n = int(data["n"])
        self.n_days = int(data["n_days"])
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
            ],
            dtype=torch.float32,
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
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
                        (batch_size, MPS_SCALAR_COLS), dtype=torch.float32, device="mps"
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

    def run(self, params: np.ndarray, *, profile: bool = False) -> dict:
        started = time.perf_counter()
        matrix = self._pack_params(params)
        packed = time.perf_counter()
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        sizes_key = (batch_size, int(matrix.shape[1]))
        if sizes_key not in self._sizes:
            self._sizes[sizes_key] = torch.tensor(
                [
                    batch_size,
                    self.n,
                    self.n_days,
                    matrix.shape[1],
                    self.run_config.first_valid_idx,
                ],
                dtype=torch.int32,
                device="mps",
            )
        prepared = time.perf_counter()
        library = _shader_library()
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        library.passivbot_ema_anchor(
            self.bars,
            self.flags,
            params_mps,
            self.settings,
            self._sizes[sizes_key],
            daily,
            scalars,
            gaps,
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
        }


class MpsEmaAnchorMulticoinRunner:
    """Persistent single-side multi-coin EMA Anchor screening runner on MPS."""

    coin_override_cols = 12
    coin_override_label = "EMA"

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        side: str,
        coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
    ):
        if side not in {"long", "short"}:
            raise ValueError(
                f"MPS multicoin runner side must be long or short, got {side!r}"
            )
        self.side = side
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
            ],
            dtype=torch.float32,
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._sizes: dict[tuple[int, int], torch.Tensor] = {}
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
                        (batch_size, MPS_SCALAR_COLS),
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
        self._buffers[batch_size][0][:, :, 1].fill_(float("inf"))
        self._buffers[batch_size][0][:, :, 5].fill_(float("inf"))
        return self._buffers[batch_size]

    def run(self, params: np.ndarray, *, profile: bool = False) -> dict:
        started = time.perf_counter()
        matrix = self._pack_params(params)
        packed = time.perf_counter()
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
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
        library = _ema_anchor_multicoin_shader_library()
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        library.passivbot_ema_anchor_multicoin(
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.coin_settings,
            self.coin_overrides,
            params_mps,
            self.settings,
            self._sizes[sizes_key],
            daily,
            scalars,
            gaps,
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
        return _decode_outputs(daily, scalars, gaps)


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

    coin_override_cols = 25
    coin_override_label = "Trailing Martingale"

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        side: str,
        coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
    ):
        super().__init__(
            run,
            data,
            side=side,
            coin_overrides=coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
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

    def run(self, params: np.ndarray, *, profile: bool = False) -> dict:
        started = time.perf_counter()
        matrix = self._pack_params(params)
        packed = time.perf_counter()
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
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
        library = _trailing_martingale_multicoin_shader_library()
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
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
            self._sizes[sizes_key],
            daily,
            scalars,
            gaps,
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
        return _decode_outputs(daily, scalars, gaps)


class MpsTrailingMartingaleRunner(MpsEmaAnchorRunner):
    """Persistent single-coin trailing-martingale runner on Apple MPS."""

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
        sizes_key = (batch_size, int(matrix.shape[1]))
        if sizes_key not in self._sizes:
            self._sizes[sizes_key] = torch.tensor(
                [
                    batch_size,
                    self.n,
                    self.n_days,
                    matrix.shape[1],
                    self.run_config.first_valid_idx,
                ],
                dtype=torch.int32,
                device="mps",
            )
        prepared = time.perf_counter()
        library = _trailing_martingale_shader_library()
        compiled = time.perf_counter()
        if profile:
            torch.mps.synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        library.passivbot_trailing_martingale(
            self.bars,
            self.flags,
            params_mps,
            self.settings,
            self._sizes[sizes_key],
            daily,
            scalars,
            gaps,
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
        active_days = torch.isfinite(daily[:, :, 1]) & (
            daily[:, :, 1] < float("inf")
        )

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
        }
