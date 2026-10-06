"""Bounded asynchronous execution, independent of optimizer/search policy.

A registered replay owns its prepared dataset and implements ``evaluate(candidates)``.
The device service serializes access to that mutable replay, combining adjacent compatible
requests into short microbatches. Callers consume individual futures; they do not observe
dispatch boundaries. No GPU runtime or evolutionary dependency is imported here.

This service does not certify a replay's simulation semantics. In particular, wrapping an
existing screening replay does not make its results authoritative.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import math
from threading import Condition, Thread, current_thread
import time
from typing import Protocol


class BatchReplay(Protocol):
    def evaluate(self, candidates: list[dict]) -> list[dict]: ...


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
    """

    def __init__(
        self, *, batch_size: int = 64, max_pending: int = 1024, max_batch_delay: float = 0.005
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if not math.isfinite(max_batch_delay) or max_batch_delay < 0:
            raise ValueError("max_batch_delay must be finite and non-negative")
        self.batch_size = min(batch_size, max_pending)
        self.max_pending = max_pending
        self.max_batch_delay = float(max_batch_delay)
        self._condition = Condition()
        self._replays: dict[str, BatchReplay] = {}
        self._queue: deque[_Pending] = deque()
        self._outstanding: dict[str, _Pending] = {}
        self._closing = False
        self._failure: BaseException | None = None
        self._thread: Thread | None = None

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
            if dataset_id in self._replays:
                raise ValueError(f"dataset already registered: {dataset_id}")
            self._replays[dataset_id] = replay

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
            if snapshot.dataset_id not in self._replays:
                raise ValueError(f"dataset not registered: {snapshot.dataset_id}")
            if snapshot.request_id in self._outstanding:
                raise ValueError(f"request already outstanding: {snapshot.request_id}")
            if len(self._outstanding) >= self.max_pending:
                raise BacktestQueueFull("GPU backtest request capacity exhausted")
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
                deadline = time.monotonic() + self.max_batch_delay
                # FIFO grouping bounds starvation; another dataset ends this batch.
                while len(self._queue) < self.batch_size and not self._closing:
                    if any(item.request.dataset_id != dataset_id for item in self._queue):
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                    if not self._queue:
                        break
                batch = []
                while (
                    self._queue
                    and len(batch) < self.batch_size
                    and self._queue[0].request.dataset_id == dataset_id
                ):
                    entry = self._queue.popleft()
                    if entry.future.set_running_or_notify_cancel():
                        batch.append(entry)
                if batch:
                    return batch

    @staticmethod
    def _results(batch: list[_Pending], rows) -> list[BacktestResult]:
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise RuntimeError("GPU replay result cardinality does not match requests")
        results = []
        for entry, row in zip(batch, rows):
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
                BacktestResult(entry.request.request_id, entry.request.dataset_id, metrics)
            )
        return results

    def _run(self) -> None:
        batch: list[_Pending] = []
        try:
            while True:
                batch = self._next_batch()
                if not batch:
                    return
                replay = self._replays[batch[0].request.dataset_id]
                rows = replay.evaluate([dict(item.request.parameters) for item in batch])
                # Validate the whole producer batch before releasing any success.
                results = self._results(batch, rows)
                for entry, result in zip(batch, results):
                    self._release(entry)
                    entry.future.set_result(result)
                del entry
                batch = []
        except BaseException as error:
            with self._condition:
                self._failure = error
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

    def __enter__(self) -> GpuBacktestService:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(cancel_pending=exc_type is not None)
