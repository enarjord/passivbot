"""CUDA execution facade; dataset buffers and replay handles stay on the worker.

This adapter is under development. Transport ownership does not establish parity
acceptance of the existing replay engine, and optimizer cutover remains separate.
"""

from contextlib import contextmanager
import json
from types import SimpleNamespace

from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import GpuBacktestService


class CudaBacktestService:
    """Register prepared scenarios and exchange requests/futures, without device handles.

    One active dataset and one replay's scratch are resident at a time. Packed immutable
    inputs are cached on disk, including shared scenario subsets, while compatible
    scenarios reuse packing. Future residency/device routing belongs behind this API.
    """

    def __init__(self, *, batch_size=64, max_pending=1024, max_batch_delay=None,
                 max_dispatch_candidate_bars=500_000_000, interrupt_check=None,
                 tuning_mode="auto"):
        from optimization.gpu.autotune import is_auto
        from optimization.gpu.execution_tuning import ExecutionBatchTuner

        if isinstance(tuning_mode, str):
            tuning_mode = tuning_mode.strip().lower()
        if not isinstance(tuning_mode, str) or tuning_mode not in {"auto", "refresh", "off"}:
            raise ValueError("GPU tuning mode must be auto, refresh, or off")
        automatic = is_auto(batch_size) and tuning_mode != "off"
        self._batch_tuner = ExecutionBatchTuner(headroom=self._headroom) if automatic else None
        requested_width = max_pending if automatic else (64 if is_auto(batch_size) else batch_size)
        self._batch_policy = self._batch_tuner or ExecutionBatchTuner(initial=requested_width, enabled=False)
        if (isinstance(max_dispatch_candidate_bars, bool)
                or not isinstance(max_dispatch_candidate_bars, int) or max_dispatch_candidate_bars < 1):
            raise ValueError("dispatch budget must be a positive integer")
        if interrupt_check is not None and not callable(interrupt_check):
            raise TypeError("interrupt_check must be callable")
        self._dispatch_budget = max_dispatch_candidate_bars
        self._interrupt_check = interrupt_check
        self._prepared_cache = {}
        self._subset_cache = {}
        self._residency = None
        self._executor = GpuBacktestService(
            batch_size=requested_width, max_pending=max_pending,
            max_batch_delay=(0.005 if max_batch_delay is None and tuning_mode == "off"
                             else max_batch_delay),
            worker_context=self._worker_scope,
            batch_policy=self._batch_policy,
        )
        self._batch_size = self._executor.batch_size

    @staticmethod
    def _headroom():
        # Queried on the CUDA owner, only before a growth trial. Producer errors
        # propagate; tuning never reruns a failed simulation or substitutes CPU work.
        import torch
        free, total = torch.cuda.mem_get_info()
        return free >= max(256 * 1024**2, total * 0.2)

    @staticmethod
    def _dispatch_ceiling(replay):
        # Keep runner references out of the suspended factory context: another
        # dataset must be able to release this replay's tensors and scratch.
        from optimization.gpu.autotune import history_dispatch_ceiling

        return history_dispatch_ceiling(replay, replay.dispatch_batch_size)

    @contextmanager
    def _worker_scope(self):
        from optimization.gpu.runtime import gpu_device
        from optimization.gpu.residency import cuda_residency_scope

        if gpu_device() != "cuda":
            raise RuntimeError("CUDA backtest service requires an NVIDIA CUDA device")
        try:
            with cuda_residency_scope() as residency:
                self._residency = residency
                yield
        finally:
            self._residency = None
            self._prepared_cache.clear()
            self._subset_cache.clear()

    def register_dataset(self, dataset_id, dataset: PreparedGpuDataset):
        if not isinstance(dataset, PreparedGpuDataset):
            raise TypeError("dataset must be CPU-prepared shared-array inputs")

        @contextmanager
        def factory():
            from optimization.gpu.service import MpsMulticoinProxy

            with dataset.attach() as (candles, btc, timestamps):
                indices = dataset.coin_indices
                if indices != tuple(range(dataset.hlcvs.shape[1])):
                    key = (dataset.hlcvs, dataset.time_range, indices)
                    if key not in self._subset_cache:
                        self._subset_cache[key] = self._residency.prepare_coin_subset(candles, indices)
                    candles = self._subset_cache[key]
                replay = MpsMulticoinProxy(
                    config=json.loads(dataset.config_json), mss=json.loads(dataset.markets_json),
                    hlcvs=candles, btc=btc, timestamps=timestamps, exchange=dataset.exchange,
                    needed_metrics=dataset.metrics, batch_size=self._batch_size,
                    max_dispatch_candidate_bars=self._dispatch_budget,
                    interrupt_check=self._interrupt_check, prepared_data_cache=self._prepared_cache,
                )
                try:
                    # Discover physical limits after claiming one ownership request.
                    # The executor fills the first dispatch up to the prepared
                    # ceiling before simulation. Runners remain worker-owned.
                    self._residency.activate(replay)
                    self._batch_policy.constrain(dataset_id, self._dispatch_ceiling(replay))
                    self._batch_policy.width(dataset_id, self._batch_size)
                    yield SimpleNamespace(evaluate=replay.evaluate_results)
                finally:
                    del replay

        self._executor.register_dataset_factory(dataset_id, factory)

    def submit(self, request):
        return self._executor.submit(request)

    def close(self, *, cancel_pending=False):
        self._executor.close(cancel_pending=cancel_pending)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(cancel_pending=exc_type is not None)
