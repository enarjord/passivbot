"""Offline execution-service lifecycle; no trading or GPU emulation."""

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import gc
import weakref

import pytest

from optimization.gpu.executor import (
    BacktestQueueFull,
    BacktestRequest,
    GpuBacktestService,
)


class Replay:
    def __init__(self):
        self.calls = []

    def evaluate(self, candidates):
        self.calls.append(candidates)
        return [{"gain": item["value"]} for item in candidates]


class GatedReplay(Replay):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()

    def evaluate(self, candidates):
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("test replay was not released")
        return super().evaluate(candidates)


def request(index, dataset="market", **params):
    return BacktestRequest(str(index), dataset, {"value": index, **params})


def test_executor_import_is_independent_of_gpu_and_search_dependencies():
    source = """
import builtins
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'cupy', 'numpy', 'pymoo', 'optimize', 'backtest'}:
        raise AssertionError(name)
    if name.startswith('optimization.backends'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
from optimization.gpu.executor import GpuBacktestService
with GpuBacktestService() as service:
    pass
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr


def test_adjacent_requests_microbatch_and_keep_identity():
    replay = Replay()
    with GpuBacktestService(batch_size=3, max_batch_delay=1) as service:
        service.register_dataset("market", replay)
        futures = [service.submit(request(i)) for i in range(3)]
        results = [future.result(timeout=3) for future in futures]
    assert len(replay.calls) == 1
    assert [result.request_id for result in results] == ["0", "1", "2"]
    assert [result.metrics for result in results] == [{"gain": float(i)} for i in range(3)]
    assert all(result.dataset_id == "market" for result in results)


def test_completion_does_not_wait_for_all_accepted_requests():
    slow = GatedReplay()
    fast = Replay()
    service = GpuBacktestService(batch_size=1)
    service.register_dataset("fast", fast)
    service.register_dataset("slow", slow)
    try:
        first = service.submit(request(1, "fast"))
        second = service.submit(request(2, "slow"))
        assert slow.started.wait(3)
        assert first.result(timeout=1).metrics == {"gain": 1.0}
        assert not second.done()
    finally:
        slow.release.set()
        service.close()
    assert second.result().metrics == {"gain": 2.0}


def test_mixed_datasets_preserve_fifo_and_never_share_replay():
    first, second = Replay(), Replay()
    with GpuBacktestService(batch_size=8, max_batch_delay=1) as service:
        service.register_dataset("first", first)
        service.register_dataset("second", second)
        futures = [
            service.submit(request(1, "first")),
            service.submit(request(2, "second")),
            service.submit(request(3, "first")),
        ]
        # Close ends batch accumulation and drains the remaining singleton.
        service.close()
    assert [future.result().metrics["gain"] for future in futures] == [1, 2, 3]
    assert [row["value"] for batch in first.calls for row in batch] == [1, 3]
    assert [row["value"] for batch in second.calls for row in batch] == [2]


def test_capacity_covers_running_work_and_cancellation_releases_queued_payloads():
    replay = GatedReplay()
    service = GpuBacktestService(batch_size=1, max_pending=2)
    service.register_dataset("market", replay)
    try:
        running = service.submit(request(1))
        assert replay.started.wait(3)
        queued = service.submit(request(2))
        with pytest.raises(BacktestQueueFull):
            service.submit(request(3))
        assert queued.cancel()
        assert list(as_completed([queued], timeout=1)) == [queued]
        replacement = service.submit(request(3))
        with pytest.raises(ValueError, match="already outstanding"):
            service.submit(request(1))
        assert not running.cancel()
    finally:
        replay.release.set()
        service.close()
    assert running.result().request_id == "1"
    assert replacement.result().request_id == "3"
    assert [row["value"] for batch in replay.calls for row in batch] == [1, 3]


def test_queued_request_snapshots_caller_parameters():
    replay = GatedReplay()
    service = GpuBacktestService(batch_size=1)
    service.register_dataset("market", replay)
    try:
        service.submit(request(0))
        assert replay.started.wait(3)
        values = {"value": 1, "nested": {"setting": 7}}
        future = service.submit(BacktestRequest("next", "market", values))
        values["value"] = 99
        values["nested"]["setting"] = 99
    finally:
        replay.release.set()
        service.close()
    assert future.result().metrics == {"gain": 1.0}
    assert replay.calls[-1][0]["nested"] == {"setting": 7}


@pytest.mark.parametrize(
    "bad_rows", [[], [{"gain": 1}, {}], [{"gain": 1}, {"gain": float("nan")}]]
)
def test_malformed_batch_has_no_partial_success_and_poisoned_admission(bad_rows):
    class BrokenReplay:
        def evaluate(self, candidates):
            return bad_rows

    with GpuBacktestService(batch_size=2, max_batch_delay=1) as service:
        service.register_dataset("market", BrokenReplay())
        futures = [service.submit(request(i)) for i in range(2)]
        for future in futures:
            with pytest.raises(RuntimeError):
                future.result(timeout=3)
        with pytest.raises(RuntimeError, match="service failed"):
            service.submit(request(3))


def test_producer_failure_reaches_running_and_queued_requests():
    failure = MemoryError("device allocation failed")

    class BrokenReplay(GatedReplay):
        def evaluate(self, candidates):
            super().evaluate(candidates)
            raise failure

    replay = BrokenReplay()
    service = GpuBacktestService(batch_size=1)
    service.register_dataset("market", replay)
    first = service.submit(request(1))
    assert replay.started.wait(3)
    queued = service.submit(request(2))
    replay.release.set()
    service.close()
    assert first.exception() is failure
    assert queued.exception() is failure
    assert len(replay.calls) == 1


def test_cancel_pending_close_preserves_running_completion_and_joins():
    replay = GatedReplay()
    service = GpuBacktestService(batch_size=1)
    service.register_dataset("market", replay)
    first = service.submit(request(1))
    assert replay.started.wait(3)
    queued = service.submit(request(2))
    closer = Thread(target=lambda: service.close(cancel_pending=True))
    closer.start()
    with pytest.raises(CancelledError):
        queued.result(timeout=3)
    assert closer.is_alive()
    replay.release.set()
    closer.join(timeout=3)
    assert not closer.is_alive()
    assert first.result().metrics == {"gain": 1.0}
    with pytest.raises(RuntimeError, match="closed"):
        service.submit(request(3))


def test_concurrent_submitters_consume_all_completions_without_identity_loss():
    with GpuBacktestService(batch_size=8, max_pending=64) as service:
        service.register_dataset("market", Replay())
        with ThreadPoolExecutor(max_workers=4) as callers:
            futures = list(callers.map(lambda i: service.submit(request(i)), range(40)))
        results = [future.result() for future in as_completed(futures, timeout=5)]
    assert {result.request_id for result in results} == {str(i) for i in range(40)}


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"max_pending": 0},
        {"max_batch_delay": -1},
        {"max_batch_delay": float("nan")},
    ],
)
def test_invalid_execution_limits_are_rejected(options):
    with pytest.raises(ValueError):
        GpuBacktestService(**options)


def test_dataset_registration_and_request_validation():
    with GpuBacktestService() as service:
        service.register_dataset("market", Replay())
        with pytest.raises(ValueError, match="already registered"):
            service.register_dataset("market", Replay())
        with pytest.raises(ValueError, match="not registered"):
            service.submit(request(1, "missing"))
        with pytest.raises(ValueError, match="request_id"):
            service.submit(BacktestRequest("", "market", {}))
        with pytest.raises(TypeError, match="mapping"):
            service.submit(BacktestRequest("1", "market", []))


def test_explicit_infinite_metric_sentinels_are_not_fabricated():
    with GpuBacktestService(batch_size=1) as service:
        service.register_dataset("market", Replay())
        result = service.submit(request(float("inf"))).result(timeout=3)
    assert result.metrics["gain"] == float("inf")


def test_cancel_close_claims_waiting_batch_before_notifying_worker(monkeypatch):
    replay = Replay()
    service = GpuBacktestService(batch_size=8, max_batch_delay=30)
    service.register_dataset("market", replay)
    accumulating = Event()
    original_wait = service._condition.wait

    def waiting(timeout=None):
        if timeout is not None:
            accumulating.set()
        return original_wait(timeout)

    monkeypatch.setattr(service._condition, "wait", waiting)
    future = service.submit(request(1))
    assert accumulating.wait(3)
    original_notify = service._condition.notify_all

    def notified():
        if service._closing:
            assert not service._queue
        original_notify()

    monkeypatch.setattr(service._condition, "notify_all", notified)
    service.close(cancel_pending=True)
    assert future.cancelled()
    assert not replay.calls


def test_retained_cancelled_futures_do_not_retain_parameter_snapshots():
    snapshots = []

    class Payload:
        def __deepcopy__(self, memo):
            clone = Payload()
            snapshots.append(weakref.ref(clone))
            return clone

    replay = GatedReplay()
    service = GpuBacktestService(batch_size=1, max_pending=2)
    service.register_dataset("market", replay)
    try:
        service.submit(request(0))
        assert replay.started.wait(3)
        retained = []
        for i in range(1, 20):
            future = service.submit(request(i, payload=Payload()))
            assert future.cancel()
            retained.append(future)
        gc.collect()
        assert all(reference() is None for reference in snapshots)
        assert len(list(as_completed(retained, timeout=1))) == len(retained)
    finally:
        replay.release.set()
        service.close()


def test_completed_future_releases_capacity_before_notifying_consumers(monkeypatch):
    from optimization.gpu import executor

    publishing, release = Event(), Event()
    real_future = executor.Future

    class PausedFuture(real_future):
        def _invoke_callbacks(self):
            publishing.set()
            if not release.wait(3):
                raise TimeoutError("test completion callbacks not released")
            super()._invoke_callbacks()

    monkeypatch.setattr(executor, "Future", PausedFuture)
    service = GpuBacktestService(batch_size=1, max_pending=1)
    service.register_dataset("market", Replay())
    try:
        first = service.submit(request(1))
        assert publishing.wait(3)
        assert first.result(timeout=1).metrics == {"gain": 1.0}
        # Publication happened, but no cleanup callback has run yet.
        replacement = service.submit(request(1))
    finally:
        release.set()
        service.close()
    assert replacement.result().request_id == "1"


def test_failed_thread_start_rolls_back_admission_and_can_retry(monkeypatch):
    from optimization.gpu import executor

    error = RuntimeError("cannot start new thread")
    real_thread = executor.Thread

    class BrokenThread(real_thread):
        def start(self):
            raise error

    service = GpuBacktestService(batch_size=1, max_pending=1)
    service.register_dataset("market", Replay())
    monkeypatch.setattr(executor, "Thread", BrokenThread)
    with pytest.raises(RuntimeError) as raised:
        service.submit(request(1))
    assert raised.value is error
    monkeypatch.setattr(executor, "Thread", real_thread)
    assert service.submit(request(1)).result(timeout=3).request_id == "1"
    service.close()
