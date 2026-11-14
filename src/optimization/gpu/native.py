"""Native GPU execution facade; dataset buffers and replay handles stay on the worker.

The native worker selects factual HSL replay from effective request policies.
Transport ownership alone does not establish general numerical acceptance; the
optimizer acceptance map records the validated scope and remaining gates.
"""

from contextlib import contextmanager
import json
import logging
from types import SimpleNamespace

from config.gpu import GPU_TUNING_DEFAULT, resolve_gpu_tuning_mode
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import GpuBacktestService


_DEFAULT = object()


def cuda_initial_candidate_width():
    """Start near one wave of one-candidate CUDA blocks on the selected GPU.

    The hardware ceiling is an upper estimate before kernel register and
    shared-memory limits. Warm replays tune the width for each dataset.
    """
    import cupy
    import torch
    from optimization.gpu.model import CUDA_MAX_BATCH_CANDIDATES
    from optimization.gpu.runtime import gpu_device

    if gpu_device(torch) != "cuda":
        raise ValueError("optimize.gpu.tuning_mode='max' requires NVIDIA CUDA")
    attributes = cupy.cuda.Device(torch.cuda.current_device()).attributes
    blocks = attributes["MultiProcessorCount"] * attributes["MaxBlocksPerMultiprocessor"]
    return min(CUDA_MAX_BATCH_CANDIDATES, 1 << (blocks.bit_length() - 1))


def _metal_system_headroom():
    """Decline growth when unified-memory telemetry is unavailable or pressured."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
        if (isinstance(total, bool) or not isinstance(total, int) or total <= 0
                or isinstance(available, bool) or not isinstance(available, int)
                or not 0 <= available <= total):
            return False
        return available >= max(256 * 1024**2, total * 0.1)
    except Exception:
        # This optional host observation only permits a larger allocation. Its
        # loss cannot fail accepted work or invent available memory.
        return False


class NativeBacktestService:
    """Register prepared scenarios and exchange requests/futures, without device handles.

    One active dataset and one replay's scratch are resident at a time. Packed immutable
    inputs are cached on disk, including shared scenario subsets, while compatible
    scenarios reuse packing. Future residency/device routing belongs behind this API.
    """

    _required_device = None

    def __init__(self, *, batch_size=_DEFAULT, max_pending=_DEFAULT, max_batch_delay=None,
                 max_dispatch_candidate_bars=500_000_000, interrupt_check=None,
                 tuning_mode=GPU_TUNING_DEFAULT, input_memory_budget=None, on_inputs_prepared=None,
                 auto_request_capacity=None, auto_batch_size_limit=None, max_cohort_width=None):
        from optimization.gpu.autotune import is_auto
        from optimization.gpu.execution_tuning import ExecutionBatchTuner
        from optimization.gpu.model import CUDA_MAX_BATCH_CANDIDATES, CUDA_MAX_INITIAL_BATCH_CANDIDATES

        tuning_mode = resolve_gpu_tuning_mode(tuning_mode)
        self._max_throughput = tuning_mode == "max"
        if batch_size is _DEFAULT:
            batch_size = None if self._max_throughput else 64
        if max_pending is _DEFAULT:
            max_pending = CUDA_MAX_BATCH_CANDIDATES if self._max_throughput else 1024
        if max_cohort_width is not None and (
            not self._max_throughput or type(max_cohort_width) is not int or max_cohort_width < 1
        ):
            raise ValueError("CUDA max cohort width must be a positive integer in max mode")
        automatic = is_auto(batch_size) and tuning_mode != "off"
        diagnostics = logging.getLogger().isEnabledFor(logging.INFO)
        self._batch_tuner = (
            ExecutionBatchTuner(
                initial=(max_cohort_width or CUDA_MAX_INITIAL_BATCH_CANDIDATES) if self._max_throughput else 64,
                headroom=self._max_headroom if self._max_throughput else self._headroom,
                diagnostics=diagnostics,
                expandable_producer=max_cohort_width is not None,
            ) if automatic else None
        )
        requested_width = max_pending if automatic else (64 if is_auto(batch_size) else batch_size)
        if auto_batch_size_limit is not None:
            if type(auto_batch_size_limit) is not int or auto_batch_size_limit < 1:
                raise ValueError("automatic batch size limit must be a positive integer")
            if automatic:
                requested_width = min(requested_width, auto_batch_size_limit)
        if auto_request_capacity is not None and (
            type(auto_request_capacity) is not int or auto_request_capacity < 1
        ):
            raise ValueError("automatic request capacity must be a positive integer")
        # Candidate-major admission can place several scenario requests per
        # candidate in the queue. Transport capacity is independent of the
        # dispatch width in automatic policies and CUDA max.
        request_capacity = (
            max(max_pending, auto_request_capacity)
            if (automatic or self._max_throughput) and auto_request_capacity is not None else max_pending
        )
        self._batch_policy = self._batch_tuner or ExecutionBatchTuner(initial=requested_width, enabled=False)
        if (isinstance(max_dispatch_candidate_bars, bool)
                or not isinstance(max_dispatch_candidate_bars, int) or max_dispatch_candidate_bars < 1):
            raise ValueError("dispatch budget must be a positive integer")
        if interrupt_check is not None and not callable(interrupt_check):
            raise TypeError("interrupt_check must be callable")
        if on_inputs_prepared is not None and not callable(on_inputs_prepared):
            raise TypeError("input preparation callback must be callable")
        self._on_inputs_prepared = on_inputs_prepared
        self._input_memory_budget = input_memory_budget
        self._dispatch_budget = max_dispatch_candidate_bars
        self._interrupt_check = interrupt_check
        self._prepared_cache = {}
        self._subset_cache = {}
        self._factual_capacities = {}
        self._residency = None
        self._executor = GpuBacktestService(
            batch_size=requested_width, max_pending=request_capacity,
            max_batch_delay=(0.005 if max_batch_delay is None and tuning_mode == "off"
                             else max_batch_delay),
            worker_context=self._worker_scope,
            batch_policy=self._batch_policy,
            accumulate_after_preparation=self._max_throughput,
            diagnostics=diagnostics,
        )
        self._batch_size = self._executor.batch_size

    def diagnostics_snapshot(self):
        return self._executor.diagnostics_snapshot()

    @property
    def cohort_width(self):
        return self._batch_tuner.cohort_width if self._batch_tuner is not None else None

    @property
    def accepted_cohort_width(self):
        return self._batch_tuner.accepted_cohort_width if self._batch_tuner is not None else None

    @staticmethod
    def _headroom():
        # Queried on the GPU owner, only before a growth trial. Producer errors
        # propagate; tuning never reruns a failed simulation or substitutes CPU work.
        import torch
        from optimization.gpu.runtime import gpu_device
        if gpu_device(torch) == "mps":
            total = torch.mps.recommended_max_memory()
            free = total - torch.mps.driver_allocated_memory()
            return free >= max(256 * 1024**2, total * 0.2) and _metal_system_headroom()
        else:
            free, total = torch.cuda.mem_get_info()
        return free >= max(256 * 1024**2, total * 0.2)

    @staticmethod
    def _max_headroom():
        import torch
        from optimization.gpu.residency import cuda_max_vram_reserve_bytes

        free, total = torch.cuda.mem_get_info()
        return free > cuda_max_vram_reserve_bytes(total)

    @staticmethod
    def _dispatch_ceiling(replay):
        # Keep runner references out of the suspended factory context: another
        # dataset must be able to release this replay's tensors and scratch.
        from optimization.gpu.autotune import history_dispatch_ceiling

        return history_dispatch_ceiling(replay, replay.dispatch_batch_size)

    @contextmanager
    def _worker_scope(self):
        from optimization.gpu.runtime import gpu_device
        from optimization.gpu.residency import cuda_residency_scope, metal_residency_scope

        device = gpu_device()
        if self._max_throughput and device != "cuda":
            raise RuntimeError("GPU tuning mode 'max' requires an NVIDIA CUDA device")
        if self._required_device is not None and device != self._required_device:
            raise RuntimeError("CUDA backtest service requires an NVIDIA CUDA device")
        if self._input_memory_budget is None:
            import torch
            from optimization.gpu.residency import (
                capture_mps_input_memory_budget, capture_cuda_input_memory_budget,
            )
            if device == "mps":
                self._input_memory_budget = capture_mps_input_memory_budget(torch)
            elif self._max_throughput:
                self._input_memory_budget = capture_cuda_input_memory_budget(torch, max_throughput=True)
            else:
                self._input_memory_budget = capture_cuda_input_memory_budget(torch)
        if device == "cuda":
            from optimization.gpu.residency import CudaInputMemoryBudget

            if (not isinstance(self._input_memory_budget, CudaInputMemoryBudget)
                    or self._input_memory_budget.max_throughput != self._max_throughput):
                raise ValueError("CUDA input memory budget does not match the GPU tuning mode")
        if self._batch_tuner is not None and device == "mps":
            # Device selection stays on the owner, before prepared replay limits
            # create a controller and clamp its initial width.
            from optimization.gpu.model import MPS_TM_MULTICOIN_CHUNK_CANDIDATES
            self._batch_tuner.initial = MPS_TM_MULTICOIN_CHUNK_CANDIDATES
        scope = metal_residency_scope if device == "mps" else cuda_residency_scope
        memory_options = ({"input_memory_budget": self._input_memory_budget}
                          if device == "cuda" else {})
        try:
            with scope(**memory_options) as residency:
                self._residency = residency
                yield
        finally:
            self._residency = None
            self._prepared_cache.clear()
            self._subset_cache.clear()
            self._factual_capacities.clear()

    def register_dataset(self, dataset_id, dataset: PreparedGpuDataset):
        if not isinstance(dataset, PreparedGpuDataset):
            raise TypeError("dataset must be CPU-prepared shared-array inputs")
        capacity_hints = {}

        @contextmanager
        def factory():
            from optimization.gpu.service import SharedAccountReplay

            with dataset.attach() as (candles, btc, timestamps), dataset.attach_selection() as selection:
                indices = dataset.coin_indices
                if indices != tuple(range(dataset.hlcvs.shape[1])):
                    key = (dataset.hlcvs, dataset.time_range, indices)
                    if key not in self._subset_cache:
                        self._subset_cache[key] = self._residency.prepare_coin_subset(candles, indices)
                    candles = self._subset_cache[key]
                replay = SharedAccountReplay(
                    config=dataset.execution_config(), mss=json.loads(dataset.markets_json),
                    hlcvs=candles, btc=btc, timestamps=timestamps, exchange=dataset.exchange,
                    needed_metrics=dataset.metrics, batch_size=self._batch_size,
                    max_dispatch_candidate_bars=self._dispatch_budget,
                    interrupt_check=self._interrupt_check, prepared_data_cache=self._prepared_cache,
                    factual_hsl=True, selection=selection,
                    input_memory_budget=self._input_memory_budget,
                    factual_capacity_hints=capacity_hints,
                    automatic_scratch_budget=self._batch_tuner is not None,
                    max_throughput=self._max_throughput,
                )
                try:
                    def constrain(*, split=False):
                        transport = None
                        if self._max_throughput and self._batch_tuner is not None:
                            transport = (replay.data["stream_tick_inputs"], replay.data["compact_hourly_ranges"])
                        self._batch_policy.constrain(dataset_id, self._dispatch_ceiling(replay),
                                                     split=split, transport=transport)

                    if self._on_inputs_prepared is not None:
                        self._on_inputs_prepared()
                    # Discover physical limits after claiming one ownership request.
                    # The executor fills the first dispatch up to the prepared
                    # ceiling before simulation. Runners remain worker-owned.
                    self._residency.activate(replay)
                    constrain()
                    self._batch_policy.width(dataset_id, self._batch_size)
                    def evaluate(candidates):
                        results = replay.evaluate_results(candidates)
                        replay._remember_factual_capacities()
                        # Successful replay may have learned a larger factual
                        # history, or switched effective HSL policies off/on.
                        # Apply its current physical limit before another claim.
                        constrain(split=self._max_throughput and replay.last_evaluation_split)
                        return results
                    yield SimpleNamespace(evaluate=evaluate)
                finally:
                    replay = None

        self._executor.register_dataset_factory(dataset_id, factory)
        self._factual_capacities[dataset_id] = capacity_hints

    def submit(self, request):
        return self._executor.submit(request)

    def set_preparing(self, dataset_ids):
        self._executor.set_preparing(dataset_ids)

    def close(self, *, cancel_pending=False):
        try:
            self._executor.close(cancel_pending=cancel_pending)
        finally:
            self._factual_capacities.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(cancel_pending=exc_type is not None)


class CudaBacktestService(NativeBacktestService):
    """CUDA-only entry point retained for explicit device qualification."""

    _required_device = "cuda"
