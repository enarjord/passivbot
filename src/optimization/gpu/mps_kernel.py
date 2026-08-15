from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
import torch

from optimization.gpu.model import (
    EMA_ANCHOR_PARAM_KEYS,
    GAP_BINS,
    ProxyMarket,
    ProxyRun,
)


MPS_DAILY_COLS = 5
MPS_SCALAR_COLS = 15


@lru_cache(maxsize=1)
def _shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(passivbot_rust.mps_ema_anchor_source_py())


class MpsEmaAnchorRunner:
    """Persistent single-coin Metal runner with invariant data resident on MPS."""

    def __init__(
        self,
        market: ProxyMarket,
        run: ProxyRun,
        data: dict,
    ):
        self.market = market
        self.run_config = run
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
            ],
            dtype=torch.float32,
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._sizes: dict[tuple[int, int], torch.Tensor] = {}
        self.last_profile: dict[str, float] = {}

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        if params.ndim != 2 or params.shape[1] != len(EMA_ANCHOR_PARAM_KEYS):
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                f"expected EMA parameter matrix with {len(EMA_ANCHOR_PARAM_KEYS)} columns, got {got}"
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
        }
