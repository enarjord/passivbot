"""Bounded asynchronous execution, independent of optimizer/search policy.

A registered replay owns its prepared dataset and implements ``evaluate(candidates)``.
The device service serializes access to that mutable replay, grouping compatible queued
requests into short microbatches. Callers consume individual futures; they do not observe
dispatch boundaries. No GPU runtime or evolutionary dependency is imported here.
Prepared-input factories can create and release replay resources on the owning worker;
preconstructed registration remains available as a transitional adapter.

This service does not certify a replay's simulation semantics. In particular, wrapping an
existing screening replay does not make its results authoritative.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import math
import logging
from threading import Condition, Thread, current_thread
import time
from typing import ContextManager, Protocol

from optimization.gpu.coalescing import BatchCoalescer


class BatchReplay(Protocol):
    def evaluate(self, candidates: list[dict]) -> list[ReplayResult | Mapping]: ...


@dataclass(frozen=True)
class ReplayResult:
    """Compact simulator output, before the executor attaches request identity."""

    metrics: Mapping[str, float]
    liquidated: bool


@dataclass(frozen=True)
class BacktestRequest:
    request_id: str
    dataset_id: str
    parameters: Mapping


@dataclass(frozen=True)
class BacktestResult:
    request_id: str
    dataset_id: str
    metrics: dict[str, float]
    # The legacy metric-only adapter does not establish terminal-event provenance.
    liquidated: bool | None = None


class BacktestQueueFull(RuntimeError):
    """Admission would exceed the queued-plus-running request capacity."""


@dataclass(eq=False)
class _Pending:
    request: BacktestRequest
    future: Future


class GpuBacktestService:
    """One owning execution thread; backend-specific replay handles stay internal.

    ``submit`` is nonblocking: backpressure raises ``BacktestQueueFull`` without
    admitting a request. Request IDs are unique among outstanding requests; effective
    candidate deduplication belongs to the caller. Futures may be consumed with
    ``as_completed``. Replay failures poison this service and fail all outstanding
    requests, so queued work cannot continue after a producer failure.

    Closing drains accepted work by default. ``cancel_pending=True`` cancels queued
    requests, then joins the currently running bounded dispatch. An executing replay
    may additionally use its own interrupt callback; threads are never force-killed.
    Factory cleanup failures are reported by ``close``. If execution also failed, its
    original exception remains authoritative and secondary cleanup failure is logged.

    ``max_batch_delay=None`` adapts accumulation from successful warm replay cost
    and observed submission bursts, with bounded absolute and idle-tail deadlines.
    A numeric delay keeps the fixed accumulation policy.
    """

    def __init__(
        self, *, batch_size: int = 64, max_pending: int = 1024, max_batch_delay: float | None = 0.005,
        worker_context: Callable[[], ContextManager] | None = None,
        batch_policy=None,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if max_batch_delay is not None and (not math.isfinite(max_batch_delay) or max_batch_delay < 0):
            raise ValueError("max_batch_delay must be finite and non-negative")
        if worker_context is not None and not callable(worker_context):
            raise TypeError("worker_context must be a resource-context factory")
        if batch_policy is not None and any(
            not callable(getattr(batch_policy, name, None)) for name in ("width", "observe")
        ):
            raise TypeError("batch_policy must implement width and observe")
        self.batch_size = min(batch_size, max_pending)
        self.max_pending = max_pending
        self.max_batch_delay = None if max_batch_delay is None else float(max_batch_delay)
        self._coalescing = BatchCoalescer() if max_batch_delay is None else None
        self._condition = Condition()
        self._replays: dict[str, BatchReplay] = {}
        self._factories: dict[str, Callable[[], ContextManager[BatchReplay]]] = {}
        self._queue: deque[_Pending] = deque()
        self._outstanding: dict[str, _Pending] = {}
        self._closing = False
        self._failure: BaseException | None = None
        self._cleanup_failure: BaseException | None = None
        self._thread: Thread | None = None
        self._worker_context = worker_context
        self._batch_policy = batch_policy

    def register_dataset(self, dataset_id: str, replay: BatchReplay) -> None:
        """Take exclusive replay ownership until close; never replace a handle.

        The caller must not evaluate or mutate the replay while it is registered.
        Device construction/residency adapters remain responsible for preparing it.
        """
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not callable(getattr(replay, "evaluate", None)):
            raise TypeError("replay must implement evaluate(candidates)")
        with self._condition:
            self._require_open()
            if dataset_id in self._replays or dataset_id in self._factories:
                raise ValueError(f"dataset already registered: {dataset_id}")
            self._replays[dataset_id] = replay

    def register_dataset_factory(
        self, dataset_id: str, factory: Callable[[], ContextManager[BatchReplay]]
    ) -> None:
        """Register prepared inputs without constructing a mutable device replay.

        The factory enters once on the owning worker, on first use. Its resource
        context remains open for reuse and exits on that same worker during shutdown.
        Factories must retain immutable prepared inputs for the service lifetime.
        Neither registration nor closing an unused service invokes a factory.
        """
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not callable(factory):
            raise TypeError("dataset factory must be callable")
        with self._condition:
            self._require_open()
            if dataset_id in self._replays or dataset_id in self._factories:
                raise ValueError(f"dataset already registered: {dataset_id}")
            self._factories[dataset_id] = factory

    def submit(self, request: BacktestRequest) -> Future[BacktestResult]:
        if not isinstance(request.request_id, str) or not request.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(request.dataset_id, str) or not request.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(request.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        # Caller mutation after admission must not change a queued simulation.
        snapshot = BacktestRequest(
            request.request_id, request.dataset_id, deepcopy(dict(request.parameters))
        )
        with self._condition:
            self._require_open()
            if snapshot.dataset_id not in self._replays and snapshot.dataset_id not in self._factories:
                raise ValueError(f"dataset not registered: {snapshot.dataset_id}")
            if snapshot.request_id in self._outstanding:
                raise ValueError(f"request already outstanding: {snapshot.request_id}")
            if len(self._outstanding) >= self.max_pending:
                raise BacktestQueueFull("GPU backtest request capacity exhausted")
            if self._coalescing is not None:
                active = any(item.request.dataset_id == snapshot.dataset_id
                             for item in self._outstanding.values())
                self._coalescing.arrived(snapshot.dataset_id, time.monotonic(), active=active)
            future: Future[BacktestResult] = Future()
            entry = _Pending(snapshot, future)
            self._outstanding[snapshot.request_id] = entry
            self._queue.append(entry)
            future.add_done_callback(
                lambda completed, request_id=snapshot.request_id: self._finished(request_id, completed)
            )
            if self._thread is None:
                try:
                    worker = Thread(target=self._run, name="gpu-backtests", daemon=False)
                    worker.start()
                except BaseException:
                    self._queue.remove(entry)
                    del self._outstanding[snapshot.request_id]
                    future.cancel()
                    future.set_running_or_notify_cancel()
                    raise
                self._thread = worker
            self._condition.notify_all()
        return future

    def _require_open(self) -> None:
        if self._failure is not None:
            raise RuntimeError("GPU backtest service failed") from self._failure
        if self._closing:
            raise RuntimeError("GPU backtest service is closed")

    def _finished(self, request_id: str, future: Future) -> None:
        with self._condition:
            entry = self._outstanding.get(request_id)
            if entry is None or entry.future is not future:
                return
            del self._outstanding[request_id]
            if future.cancelled():
                # Remove cancelled payloads immediately, even during a long dispatch.
                try:
                    self._queue.remove(entry)
                except ValueError:
                    pass  # Already claimed by the owning execution thread.
                else:
                    # Executor notification is still required for wait/as_completed,
                    # even when cancelled work never reaches the replay thread.
                    future.set_running_or_notify_cancel()
            self._condition.notify_all()

    def _release(self, entry: _Pending) -> None:
        # Complete admission bookkeeping before future completion wakes consumers.
        with self._condition:
            if self._outstanding.get(entry.request.request_id) is entry:
                del self._outstanding[entry.request.request_id]
            self._condition.notify_all()

    def _next_batch(self) -> list[_Pending]:
        with self._condition:
            while True:
                while not self._queue:
                    if self._closing:
                        return []
                    self._condition.wait()
                dataset_id = self._queue[0].request.dataset_id
                width = (self.batch_size if self._batch_policy is None
                         else self._batch_policy.width(dataset_id, self.batch_size))
                if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= self.batch_size:
                    raise ValueError("GPU batch policy width must respect the dispatch ceiling")
                started = time.monotonic()
                deadline = (None if self._coalescing is not None
                            else started + self.max_batch_delay)
                # The oldest request chooses the dataset, so locality grouping can
                # never starve another dataset behind newly arriving requests.
                while not self._closing:
                    compatible = sum(item.request.dataset_id == dataset_id for item in self._queue)
                    if not compatible or compatible >= width or len(self._queue) >= self.max_pending:
                        break
                    until = (self._coalescing.deadline(dataset_id, started)
                             if self._coalescing is not None else deadline)
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                    if not self._queue:
                        break
                batch = []
                remaining_queue = deque()
                while self._queue:
                    entry = self._queue.popleft()
                    if entry.request.dataset_id != dataset_id or len(batch) >= width:
                        remaining_queue.append(entry)
                    elif entry.future.set_running_or_notify_cancel():
                        batch.append(entry)
                self._queue = remaining_queue
                if batch:
                    return batch

    @staticmethod
    def _results(batch: list[_Pending], rows) -> list[BacktestResult]:
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise RuntimeError("GPU replay result cardinality does not match requests")
        results = []
        for entry, row in zip(batch, rows):
            liquidated = None
            if isinstance(row, ReplayResult):
                if not isinstance(row.liquidated, bool):
                    raise RuntimeError("GPU replay liquidation status must be Boolean")
                liquidated, row = row.liquidated, row.metrics
            if not isinstance(row, Mapping) or not row:
                raise RuntimeError("GPU replay must return non-empty metric mappings")
            metrics = {}
            for name, value in row.items():
                if not isinstance(name, str) or not name:
                    raise RuntimeError("GPU replay metric names must be non-empty strings")
                number = float(value)
                if math.isnan(number):
                    raise RuntimeError(f"GPU replay returned NaN metric: {name}")
                # Infinity may be an explicit insufficient-sample metric sentinel.
                # Its interpretation remains the canonical metric/scoring owner's job.
                metrics[name] = number
            results.append(
                BacktestResult(entry.request.request_id, entry.request.dataset_id, metrics, liquidated)
            )
        return results

    def _run(self) -> None:
        batch: list[_Pending] = []
        resources = ExitStack()
        context_entered = False
        failure_info = (None, None, None)
        try:
            while True:
                batch = self._next_batch()
                if not batch:
                    return
                if self._worker_context is not None and not context_entered:
                    resources.enter_context(self._worker_context())
                    context_entered = True
                dataset_id = batch[0].request.dataset_id
                if dataset_id not in self._replays:
                    replay = resources.enter_context(self._factories[dataset_id]())
                    if not callable(getattr(replay, "evaluate", None)):
                        raise TypeError("dataset factory must yield a replay implementing evaluate(candidates)")
                    self._replays[dataset_id] = replay
                replay = self._replays[dataset_id]
                timed = self._batch_policy is not None or self._coalescing is not None
                started = time.perf_counter() if timed else 0.0
                try:
                    rows = replay.evaluate([dict(item.request.parameters) for item in batch])
                finally:
                    del replay
                # Validate the whole producer batch before releasing any success.
                results = self._results(batch, rows)
                if timed:
                    seconds = time.perf_counter() - started
                    with self._condition:
                        if self._coalescing is not None:
                            self._coalescing.observe(dataset_id, len(batch), seconds)
                        backlog = sum(item.request.dataset_id == dataset_id for item in self._queue)
                        closing = self._closing
                if self._batch_policy is not None:
                    self._batch_policy.observe(
                        dataset_id, len(batch), seconds, backlog=backlog, closing=closing,
                    )
                for entry, result in zip(batch, results):
                    self._release(entry)
                    entry.future.set_result(result)
                del entry
                batch = []
        except BaseException as error:
            failure_info = (type(error), error, error.__traceback__)
            self._fail(error, batch)
        finally:
            # Drop service references before exiting their owning resource contexts.
            for dataset_id in self._factories:
                self._replays.pop(dataset_id, None)
            try:
                resources.__exit__(*failure_info)
            except BaseException as error:
                self._cleanup_failure = error
                if self._failure is not None:
                    logging.exception("GPU replay cleanup failed after execution failure")
                self._fail(error, [])

    def _fail(self, error: BaseException, batch: list[_Pending]) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
            error = self._failure
            self._closing = True
            abandoned = list(self._queue)
            self._queue.clear()
            self._condition.notify_all()
        for entry in batch:
            if not entry.future.done():
                self._release(entry)
                entry.future.set_exception(error)
        for entry in abandoned:
            if entry.future.set_running_or_notify_cancel():
                self._release(entry)
                entry.future.set_exception(error)

    def close(self, *, cancel_pending: bool = False) -> None:
        if current_thread() is self._thread:
            raise RuntimeError("close must not run inside an execution-thread callback")
        with self._condition:
            self._closing = True
            queued = list(self._queue) if cancel_pending else []
            # Claim cancellation before waking an accumulating worker. Its queue
            # cannot become a running dispatch after close requested cancellation.
            if cancel_pending:
                self._queue.clear()
            self._condition.notify_all()
        # User callbacks run outside the service lock. These detached entries
        # cannot be claimed by the worker, and still need executor notification.
        for entry in queued:
            entry.future.cancel()
            entry.future.set_running_or_notify_cancel()
        if self._thread is not None:
            self._thread.join()
        self._replays.clear()
        self._factories.clear()
        if self._cleanup_failure is not None:
            raise self._failure

    def __enter__(self) -> GpuBacktestService:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(cancel_pending=exc_type is not None)
