"""Offline execution-service lifecycle; no trading or GPU emulation."""

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread, get_ident
from queue import Queue
from types import SimpleNamespace
import time
import gc
import weakref

import pytest

from optimization.gpu.executor import (
    BacktestQueueFull,
    BacktestRequest,
    GpuBacktestService,
    ReplayResult,
)


class Replay:
    def __init__(self):
        self.calls = []

    def evaluate(self, candidates):
        self.calls.append(candidates)
        return [ReplayResult({"gain": item["value"]}, False) for item in candidates]


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


def test_fixed_service_remains_untimed_without_optional_diagnostics(monkeypatch):
    from optimization.gpu import executor as module

    def forbidden():
        pytest.fail("fixed service must not acquire replay clocks by default")
    monkeypatch.setattr(module.time, "perf_counter", forbidden)
    with GpuBacktestService(batch_size=1, max_batch_delay=0) as service:
        service.register_dataset("market", Replay())
        assert service.submit(request(1)).result(timeout=3).metrics == {"gain": 1}
        assert service.diagnostics_snapshot() is None


def test_optional_owner_diagnostics_separate_preparation_and_validated_execution(monkeypatch):
    from optimization.gpu import executor as module

    tick = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: tick[0])
    @contextmanager
    def worker():
        tick[0] += 2
        yield
    class TimedReplay(Replay):
        def evaluate(self, candidates):
            tick[0] += 7
            return super().evaluate(candidates)
    @contextmanager
    def factory():
        tick[0] += 3
        yield TimedReplay()
    with GpuBacktestService(batch_size=1, max_batch_delay=0, worker_context=worker,
                             diagnostics=True) as service:
        service.register_dataset_factory("market", factory)
        assert service.submit(request(1)).result(timeout=3).metrics == {"gain": 1}
        snapshot = service.diagnostics_snapshot()
        assert snapshot["batches"] == snapshot["rows"] == snapshot["last_count"] == 1
        assert snapshot["execution_seconds"] == 7
        assert snapshot["preparation_seconds"] == 5
        assert snapshot["accumulation_seconds"] >= 0
        snapshot["rows"] = -1
        assert service.diagnostics_snapshot()["rows"] == 1


def test_owner_diagnostics_do_not_count_an_invalid_batch_as_completed():
    failure = ValueError("original producer failure")
    def failed(_rows):
        raise failure
    with GpuBacktestService(max_batch_delay=0, diagnostics=True) as service:
        service.register_dataset("market", SimpleNamespace(evaluate=failed))
        with pytest.raises(ValueError) as raised:
            service.submit(request(0)).result(timeout=3)
        assert raised.value is failure
        snapshot = service.diagnostics_snapshot()
        assert snapshot["batches"] == snapshot["rows"] == 0
        assert snapshot["execution_seconds"] == 0


@pytest.mark.parametrize("clock", ["monotonic", "perf_counter"])
def test_optional_owner_clock_failure_does_not_fail_successful_futures(monkeypatch, clock):
    from optimization.gpu import executor as module

    original = getattr(module.time, clock)
    calls = [0]
    def unavailable():
        calls[0] += 1
        # The first monotonic read belongs to the existing fixed scheduler;
        # its second read is only the new accumulation diagnostic.
        if calls[0] == (2 if clock == "monotonic" else 1):
            raise OSError("optional clock unavailable")
        return original()
    monkeypatch.setattr(module.time, clock, unavailable)
    with GpuBacktestService(batch_size=1, max_batch_delay=0, diagnostics=True) as service:
        service.register_dataset("market", Replay())
        assert service.submit(request(1)).result(timeout=3).metrics == {"gain": 1}
        assert service.diagnostics_snapshot() is None


def test_required_policy_timing_failure_remains_authoritative(monkeypatch):
    from optimization.gpu import executor as module

    failure = OSError("required policy clock unavailable")
    def unavailable():
        raise failure
    policy = SimpleNamespace(width=lambda *_args: 1,
                             observe=lambda *_args, **_kwargs: pytest.fail("no valid timing evidence"))
    monkeypatch.setattr(module.time, "perf_counter", unavailable)
    with GpuBacktestService(batch_size=1, max_batch_delay=0, batch_policy=policy,
                             diagnostics=True) as service:
        service.register_dataset("market", Replay())
        with pytest.raises(OSError) as raised:
            service.submit(request(1)).result(timeout=3)
        assert raised.value is failure


def test_service_changes_batch_width_only_between_successful_dispatches():
    class Policy:
        def __init__(self):
            self.widths = {"a": 2, "b": 1}
            self.completed = []
        def width(self, dataset, ceiling):
            return self.widths[dataset]
        def observe(self, dataset, count, seconds, *, backlog, closing):
            assert seconds > 0
            self.completed.append((dataset, count, backlog))
            self.widths[dataset] = 4 if len(self.completed) == 1 else 1

    policy, replay = Policy(), GatedReplay()
    service = GpuBacktestService(batch_size=4, max_pending=16, max_batch_delay=0.05,
                                 batch_policy=policy)
    service.register_dataset("a", replay)
    service.register_dataset("b", replay)
    try:
        futures = [service.submit(request(i, "a")) for i in range(2)]
        assert replay.started.wait(3)
        futures.extend(service.submit(request(i, dataset)) for i, dataset in enumerate(
            ("a", "b", "a", "b", "a", "a", "b", "a"), start=2
        ))
        replay.release.set()
        assert [future.result(timeout=3).metrics for future in futures] == [
            {"gain": i} for i in range(10)
        ]
    finally:
        replay.release.set()
        service.close()
    assert policy.completed[:2] == [("a", 2, 5), ("a", 4, 1)]
    assert [len(batch) for batch in replay.calls] == [2, 4, 1, 1, 1, 1]
    assert sorted(item["value"] for batch in replay.calls for item in batch) == list(range(10))


@pytest.mark.parametrize("width", [0, 5, True])
def test_invalid_policy_width_fails_before_constructing_or_running_replay(width):
    class Policy:
        def width(self, *_args):
            return width
        def observe(self, *_args, **_kwargs):
            pytest.fail("invalid policy must never evaluate")
    replay = Replay()
    with GpuBacktestService(batch_size=4, batch_policy=Policy()) as service:
        service.register_dataset("market", replay)
        future = service.submit(request(0))
        with pytest.raises(ValueError, match="dispatch ceiling"):
            future.result(timeout=3)
    assert not replay.calls


def test_first_dispatch_uses_prepared_capacity_without_repeating_or_reordering_work():
    from optimization.gpu.execution_tuning import ExecutionBatchTuner

    policy = ExecutionBatchTuner(initial=8, enabled=False)
    preparing, release = Event(), Event()
    replay, other = Replay(), Replay()

    @contextmanager
    def factory():
        preparing.set()
        assert release.wait(3)
        policy.constrain("market", 2)
        yield replay

    service = GpuBacktestService(batch_size=8, max_pending=16, max_batch_delay=0,
                                 batch_policy=policy)
    service.register_dataset_factory("market", factory)
    service.register_dataset("other", other)
    try:
        first = service.submit(request(0))
        assert preparing.wait(3)
        cancelled = service.submit(request(1))
        remaining = [service.submit(request(i, dataset)) for i, dataset in
                     ((2, "other"), (3, "market"), (4, "market"))]
        assert cancelled.cancel()
        release.set()
        assert [future.result(timeout=3).metrics for future in [first, *remaining]] == [
            {"gain": i} for i in (0, 2, 3, 4)
        ]
    finally:
        release.set()
        service.close()
    assert [[item["value"] for item in batch] for batch in replay.calls] == [[0, 3], [4]]
    assert other.calls == [[{"value": 2}]]
    assert cancelled.cancelled()


def test_cancel_pending_during_preparation_does_not_fill_first_dispatch():
    from optimization.gpu.execution_tuning import ExecutionBatchTuner

    policy = ExecutionBatchTuner(initial=8, enabled=False)
    preparing, release, cancelled = Event(), Event(), Event()
    replay = Replay()

    @contextmanager
    def factory():
        preparing.set()
        assert release.wait(3)
        policy.constrain("market", 2)
        yield replay

    service = GpuBacktestService(batch_size=8, max_pending=16, max_batch_delay=0,
                                 batch_policy=policy)
    service.register_dataset_factory("market", factory)
    closer = None
    try:
        first = service.submit(request(0))
        assert preparing.wait(3)
        pending = service.submit(request(1))
        pending.add_done_callback(lambda _future: cancelled.set())
        closer = Thread(target=lambda: service.close(cancel_pending=True))
        closer.start()
        assert cancelled.wait(3)
        assert pending.cancelled()
        release.set()
        assert first.result(timeout=3).metrics == {"gain": 0}
    finally:
        release.set()
        if closer is not None:
            closer.join(3)
            assert not closer.is_alive()
        service.close()
    assert replay.calls == [[{"value": 0}]]


@pytest.mark.parametrize("prepared_width", [0, 1, 9, True])
def test_invalid_prepared_capacity_fails_claimed_work_without_simulation(prepared_width):
    class Policy:
        prepared = False

        def width(self, *_args):
            return prepared_width if self.prepared else 2

        def observe(self, *_args, **_kwargs):
            pytest.fail("invalid capacity must not supply execution evidence")

    policy, replay = Policy(), Replay()
    exited = Event()

    @contextmanager
    def factory():
        policy.prepared = True
        try:
            yield replay
        finally:
            exited.set()

    with GpuBacktestService(batch_size=8, max_batch_delay=0, batch_policy=policy) as service:
        service.register_dataset_factory("market", factory)
        # Make two requests available before the owner can claim either.
        with service._condition:
            futures = [service.submit(request(i)) for i in range(2)]
        for future in futures:
            with pytest.raises(ValueError, match="dispatch ceiling"):
                future.result(timeout=3)
    assert exited.is_set()
    assert not replay.calls


@pytest.mark.parametrize("malformed", [False, True])
def test_failed_producer_work_never_supplies_tuning_evidence(malformed):
    failure = RuntimeError("producer failed")
    class Policy:
        def width(self, *_args):
            return 1
        def observe(self, *_args, **_kwargs):
            pytest.fail("failed work must never tune")
    class FailingReplay:
        def evaluate(self, _candidates):
            if malformed:
                return [ReplayResult({"gain": float("nan")}, False)]
            raise failure
    with GpuBacktestService(batch_policy=Policy()) as service:
        service.register_dataset("market", FailingReplay())
        future = service.submit(request(0))
        if malformed:
            with pytest.raises(RuntimeError, match="NaN metric"):
                future.result(timeout=3)
        else:
            assert future.exception(timeout=3) is failure


def test_worker_context_surrounds_all_dataset_resources_on_owner():
    from contextlib import contextmanager
    from threading import get_ident
    events = []

    @contextmanager
    def context(label):
        events.append(("enter", label, get_ident()))
        try:
            yield Replay()
        finally:
            events.append(("exit", label, get_ident()))

    with GpuBacktestService(batch_size=1, worker_context=lambda: context("worker")) as service:
        service.register_dataset_factory("a", lambda: context("a"))
        service.register_dataset_factory("b", lambda: context("b"))
        assert events == []
        for i, name in enumerate(("a", "b", "a")):
            assert service.submit(request(i, name)).result(timeout=3).metrics == {"gain": i}
    owner = service._thread.ident
    assert events == [("enter", name, owner) for name in ("worker", "a", "b")] + [
        ("exit", name, owner) for name in ("b", "a", "worker")
    ]


def test_unused_worker_context_stays_lazy_and_invalid_factory_is_rejected():
    def forbidden():
        pytest.fail("unused service must not enter a device context")
    with GpuBacktestService(worker_context=forbidden):
        pass
    with pytest.raises(TypeError, match="worker_context"):
        GpuBacktestService(worker_context=object())


@pytest.mark.parametrize("stage", ["setup", "cleanup"])
def test_worker_context_failures_poison_admission_and_surface_cleanup(stage):
    from contextlib import contextmanager

    @contextmanager
    def context():
        if stage == "setup":
            raise RuntimeError("context setup")
        yield
        if stage == "cleanup":
            raise RuntimeError("context cleanup")

    service = GpuBacktestService(batch_size=1, worker_context=context)
    service.register_dataset("market", Replay())
    future = service.submit(request(1))
    if stage == "setup":
        with pytest.raises(RuntimeError, match="context setup"):
            future.result(timeout=3)
        with pytest.raises(RuntimeError, match="service failed"):
            service.submit(request(2))
        service.close()
    else:
        assert future.result(timeout=3).metrics == {"gain": 1}
        with pytest.raises(RuntimeError, match="context cleanup"):
            service.close()
        with pytest.raises(RuntimeError, match="service failed"):
            service.submit(request(2))


def test_factory_constructs_reuses_and_releases_resources_on_worker():
    events = []
    main_thread = get_ident()

    @contextmanager
    def factory():
        events.append(("enter", get_ident()))
        replay = Replay()
        try:
            yield replay
        finally:
            events.append(("exit", get_ident(), len(replay.calls)))

    service = GpuBacktestService(batch_size=1)
    service.register_dataset_factory("market", factory)
    assert not events
    for i in range(2):
        assert service.submit(request(i)).result(timeout=3).metrics == {"gain": i}
    assert len(events) == 1
    service.close()
    assert events == [("enter", service._thread.ident), ("exit", service._thread.ident, 2)]
    assert service._thread.ident != main_thread


def test_unused_and_cancelled_factories_are_not_entered():
    calls = []

    @contextmanager
    def factory():
        calls.append("entered")
        yield Replay()

    with GpuBacktestService() as unused:
        unused.register_dataset_factory("unused", factory)
    replay = GatedReplay()
    service = GpuBacktestService(batch_size=1)
    service.register_dataset("slow", replay)
    service.register_dataset_factory("unused", factory)
    try:
        service.submit(request(0, "slow"))
        assert replay.started.wait(3)
        cancelled = service.submit(request(1, "unused"))
        assert cancelled.cancel()
    finally:
        replay.release.set()
        service.close()
    assert not calls


def test_factory_setup_failure_fails_running_and_queued_requests():
    started, release = Event(), Event()
    failure = MemoryError("resident dataset allocation failed")

    @contextmanager
    def factory():
        started.set()
        assert release.wait(3)
        raise failure
        yield  # pragma: no cover - context setup deliberately never reaches yield

    service = GpuBacktestService(batch_size=1)
    service.register_dataset_factory("market", factory)
    first = service.submit(request(0))
    assert started.wait(3)
    queued = service.submit(request(1))
    release.set()
    service.close()
    assert first.exception() is failure
    assert queued.exception() is failure
    with pytest.raises(RuntimeError, match="service failed"):
        service.submit(request(2))


@pytest.mark.parametrize("context_valid", [False, True])
def test_invalid_factory_output_is_fatal_and_entered_context_is_released(context_valid):
    errors = []

    @contextmanager
    def invalid():
        try:
            yield object()
        except BaseException as error:
            errors.append(error)
            raise

    with GpuBacktestService(batch_size=1) as service:
        service.register_dataset_factory("market", invalid if context_valid else lambda: Replay())
        future = service.submit(request(0))
        with pytest.raises(TypeError):
            future.result(timeout=3)
    assert len(errors) == int(context_valid)
    if errors:
        assert errors[0] is future.exception()


@pytest.mark.parametrize("producer_fails", [False, True])
def test_factory_cleanup_failure_is_reported_without_replacing_producer_error(producer_fails, caplog):
    producer_error = MemoryError("simulation failed")
    cleanup_error = RuntimeError("device cleanup failed")

    class BrokenReplay(Replay):
        def evaluate(self, candidates):
            if producer_fails:
                raise producer_error
            return super().evaluate(candidates)

    @contextmanager
    def factory():
        try:
            yield BrokenReplay()
        finally:
            raise cleanup_error

    service = GpuBacktestService(batch_size=1)
    service.register_dataset_factory("market", factory)
    result = service.submit(request(0))
    if producer_fails:
        assert result.exception(timeout=3) is producer_error
    else:
        assert result.result(timeout=3).metrics == {"gain": 0}
    expected = producer_error if producer_fails else cleanup_error
    with pytest.raises(type(expected)) as raised:
        service.close()
    assert raised.value is expected
    assert service._failure is expected
    assert not service._replays and not service._factories
    if producer_fails:
        assert "cleanup failed after execution failure" in caplog.text


def test_all_factory_contexts_exit_even_if_one_cleanup_fails():
    released = []
    failure = RuntimeError("second dataset cleanup failed")

    @contextmanager
    def first():
        try:
            yield Replay()
        finally:
            released.append(("first", get_ident()))

    @contextmanager
    def second():
        try:
            yield Replay()
        finally:
            released.append(("second", get_ident()))
            raise failure

    service = GpuBacktestService(batch_size=1)
    service.register_dataset_factory("first", first)
    service.register_dataset_factory("second", second)
    for i, dataset in enumerate(("first", "second")):
        service.submit(request(i, dataset)).result(timeout=3)
    with pytest.raises(RuntimeError) as raised:
        service.close()
    assert raised.value is failure
    assert released == [("second", service._thread.ident), ("first", service._thread.ident)]


def test_factory_and_preconstructed_registration_share_identity_validation():
    with GpuBacktestService() as service:
        with pytest.raises(TypeError, match="must be callable"):
            service.register_dataset_factory("invalid", None)
        with pytest.raises(ValueError, match="non-empty"):
            service.register_dataset_factory("", lambda: None)
        service.register_dataset_factory("factory", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            service.register_dataset("factory", Replay())
        service.register_dataset("direct", Replay())
        with pytest.raises(ValueError, match="already registered"):
            service.register_dataset_factory("direct", lambda: None)


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
    assert all(result.liquidated is False for result in results)


def test_structured_replay_preserves_terminal_status_per_request():
    class StructuredReplay(Replay):
        def evaluate(self, candidates):
            return [ReplayResult(row.metrics, bool(item["value"] % 2)) for row, item in
                    zip(super().evaluate(candidates), candidates)]
    with GpuBacktestService(batch_size=3, max_batch_delay=1) as service:
        service.register_dataset("market", StructuredReplay())
        futures = [service.submit(request(i)) for i in range(3)]
        results = [future.result(timeout=3) for future in futures]
    assert [row.metrics for row in results] == [{"gain": float(i)} for i in range(3)]
    assert [row.liquidated for row in results] == [False, True, False]


def test_metric_only_producer_cannot_supply_execution_results():
    class MetricOnlyReplay:
        def evaluate(self, candidates):
            return [{"gain": item["value"]} for item in candidates]

    with GpuBacktestService(batch_size=2, max_batch_delay=1) as service:
        service.register_dataset("market", MetricOnlyReplay())
        futures = [service.submit(request(index)) for index in range(2)]
        for future in futures:
            with pytest.raises(RuntimeError, match="structured results with terminal status"):
                future.result(timeout=3)


@pytest.mark.parametrize("bad_status", [None, 0, "true"])
def test_invalid_terminal_status_rejects_the_complete_producer_batch(bad_status):
    class BrokenReplay(Replay):
        def evaluate(self, candidates):
            rows = super().evaluate(candidates)
            rows[-1] = ReplayResult(rows[-1].metrics, bad_status)
            return rows
    with GpuBacktestService(batch_size=2, max_batch_delay=1) as service:
        service.register_dataset("market", BrokenReplay())
        futures = [service.submit(request(i)) for i in range(2)]
        for future in futures:
            with pytest.raises(RuntimeError, match="liquidation status"):
                future.result(timeout=3)
        with pytest.raises(RuntimeError, match="service failed"):
            service.submit(request(2))


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


def test_automatic_coalescing_tracks_accepted_arrivals_and_successful_work():
    replay = GatedReplay()
    service = GpuBacktestService(batch_size=1, max_pending=2, max_batch_delay=None)
    service.register_dataset("market", replay)
    try:
        first = service.submit(request(1))
        assert replay.started.wait(3)
        queued = service.submit(request(2))
        stream = service._coalescing._streams["market"]
        prior = (stream.last_arrival, tuple(stream.gaps))
        with pytest.raises(BacktestQueueFull):
            service.submit(request(3))
        with pytest.raises(ValueError, match="already outstanding"):
            service.submit(request(1))
        assert (stream.last_arrival, tuple(stream.gaps)) == prior
        assert queued.cancel()
        # Keep both accepted replays alive until their original identities return.
        third = service.submit(request(3))
        replay.release.set()
        assert first.result(timeout=3).request_id == "1"
        assert third.result(timeout=3).request_id == "3"
        assert stream.seen_counts == {1}
        assert stream.replay_seconds is not None
    finally:
        replay.release.set()
        service.close(cancel_pending=True)


def test_invalid_producer_output_does_not_train_automatic_coalescing():
    class BrokenReplay(Replay):
        def evaluate(self, candidates):
            return [ReplayResult({}, False) for _ in candidates]

    with GpuBacktestService(batch_size=1, max_batch_delay=None) as service:
        service.register_dataset("market", BrokenReplay())
        with pytest.raises(RuntimeError, match="non-empty"):
            service.submit(request(1)).result(timeout=3)
        stream = service._coalescing._streams["market"]
        assert not stream.seen_counts
        assert stream.replay_seconds is None


def test_mixed_datasets_preserve_order_within_dataset_and_never_share_replay():
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


@pytest.mark.parametrize("delay", [0, None])
def test_interleaved_requests_group_by_oldest_dataset_without_starving_other_work(delay):
    calls = []

    class OrderedReplay(Replay):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def evaluate(self, candidates):
            calls.append((self.name, [item["value"] for item in candidates]))
            return super().evaluate(candidates)

    service = GpuBacktestService(batch_size=2, max_batch_delay=delay)
    for name in ("a", "b"):
        service.register_dataset(name, OrderedReplay(name))
    try:
        with service._condition:
            futures = [service.submit(request(i, name)) for i, name in
                       ((1, "a"), (2, "b"), (3, "a"), (4, "b"), (5, "a"), (6, "a"), (7, "b"))]
        assert [future.result(timeout=3).request_id for future in futures] == list(map(str, range(1, 8)))
    finally:
        service.close()
    assert calls == [("a", [1, 3]), ("b", [2, 4]), ("a", [5, 6]), ("b", [7])]


def test_full_mixed_queue_dispatches_partial_batch_without_waiting_for_more_admission():
    service = GpuBacktestService(batch_size=3, max_pending=3, max_batch_delay=30)
    first, second = Replay(), Replay()
    service.register_dataset("a", first)
    service.register_dataset("b", second)
    try:
        with service._condition:
            futures = [service.submit(request(i, name)) for i, name in ((1, "a"), (2, "b"), (3, "a"))]
        assert futures[0].result(timeout=3).request_id == "1"
        assert futures[2].result(timeout=3).request_id == "3"
        # Closing ends the final dataset's bounded accumulation period.
        service.close()
        assert futures[1].result().request_id == "2"
    finally:
        service.close()
    assert [[row["value"] for row in batch] for batch in first.calls] == [[1, 3]]


@pytest.mark.parametrize("automatic", [False, True])
def test_cancelling_accumulating_dataset_reselects_other_ready_work(monkeypatch, automatic):
    service = GpuBacktestService(batch_size=2, max_batch_delay=None if automatic else 30)
    if automatic:
        monkeypatch.setattr(service._coalescing, "deadline", lambda _dataset, started: started + 30)
    service.register_dataset("a", Replay())
    service.register_dataset("b", Replay())
    waiting = Event()
    original_wait = service._condition.wait

    def wait(timeout=None):
        if timeout is not None:
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(service._condition, "wait", wait)
    try:
        cancelled = service.submit(request(1, "a"))
        assert waiting.wait(3)
        futures = [service.submit(request(i, "b")) for i in (2, 3)]
        assert cancelled.cancel()
        assert [future.result(timeout=3).request_id for future in futures] == ["2", "3"]
    finally:
        service.close(cancel_pending=True)


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
    "bad_rows", [[], [ReplayResult({"gain": 1}, False), ReplayResult({}, False)],
                 [ReplayResult({"gain": 1}, False), ReplayResult({"gain": float("nan")}, False)]]
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


@pytest.mark.parametrize("automatic", [False, True])
def test_cancel_close_claims_waiting_batch_before_notifying_worker(monkeypatch, automatic):
    replay = Replay()
    service = GpuBacktestService(batch_size=8, max_batch_delay=None if automatic else 30)
    if automatic:
        monkeypatch.setattr(service._coalescing, "deadline", lambda _dataset, started: started + 30)
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


def test_automatic_close_drains_accepted_work_without_waiting_for_deadline(monkeypatch):
    replay = Replay()
    service = GpuBacktestService(batch_size=8, max_batch_delay=None)
    service.register_dataset("market", replay)
    monkeypatch.setattr(service._coalescing, "deadline", lambda _dataset, started: started + 30)
    accumulating = Event()
    original_wait = service._condition.wait

    def waiting(timeout=None):
        if timeout is not None:
            accumulating.set()
        return original_wait(timeout)

    monkeypatch.setattr(service._condition, "wait", waiting)
    future = service.submit(request(1))
    assert accumulating.wait(3)
    service.close()
    assert future.result().metrics == {"gain": 1.0}
    assert len(replay.calls) == 1


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


def warm_preparing_service(service, dataset, now):
    policy = service._coalescing
    policy.arrived(dataset, now - 1, active=False)
    for step in range(1, 5):
        policy.arrived(dataset, now - 1 + step * 0.2, active=True)
    policy.observe(dataset, 4, 2)


def test_preparing_producer_fills_through_gaps_beyond_the_ordinary_delay(monkeypatch):
    import optimization.gpu.executor as executor

    clock = [100.0]
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter,
    ))
    replay, waiting = Replay(), Queue()
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    service.register_dataset("market", replay)
    warm_preparing_service(service, "market", clock[0])
    ordinary = service._coalescing.preparing_deadline
    def deadline(dataset, started):
        value = ordinary(dataset, started)
        waiting.put(value)
        return value
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"market"})
    try:
        futures = [service.submit(request(0))]
        waiting.get(timeout=2)
        for index in (1, 2):
            clock[0] += 0.7  # Exceeds the learned 0.4 s idle tail while CPU work remains.
            # Recheck before a fresh submit could refresh last_arrival. The old
            # arrival-tail policy must release a partial batch at this boundary.
            with service._condition:
                service._condition.notify_all()
            assert waiting.get(timeout=2) > clock[0]
            with service._condition:
                assert not replay.calls and all(not future.done() for future in futures)
            futures.append(service.submit(request(index)))
            assert waiting.get(timeout=2) > clock[0]
        assert clock[0] > 100.5 and not replay.calls
        clock[0] += 0.7
        with service._condition:
            service._condition.notify_all()
        assert waiting.get(timeout=2) > clock[0]
        with service._condition:
            assert not replay.calls and all(not future.done() for future in futures)
        futures.append(service.submit(request(3)))
        assert [future.result(timeout=2).request_id for future in futures] == ["0", "1", "2", "3"]
        assert [[row["value"] for row in batch] for batch in replay.calls] == [[0, 1, 2, 3]]
    finally:
        service.close(cancel_pending=True)


def test_abandoned_preparing_hint_expires_at_the_original_absolute_deadline(monkeypatch):
    import optimization.gpu.executor as executor

    clock, waiting, replay = [100.0], Queue(), Replay()
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter,
    ))
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    service.register_dataset("market", replay)
    warm_preparing_service(service, "market", clock[0])
    ordinary = service._coalescing.preparing_deadline
    def deadline(dataset, started):
        value = ordinary(dataset, started)
        waiting.put(value)
        return value
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"market"})
    try:
        future = service.submit(request(1))
        assert waiting.get(timeout=2) == 103.0
        clock[0] = 104.0
        service.set_preparing({"market"})  # Repeating readiness cannot renew its lease.
        with service._condition:
            service._condition.notify_all()
        assert future.result(timeout=2).request_id == "1"
        assert len(replay.calls) == 1
    finally:
        service.close(cancel_pending=True)


@pytest.mark.parametrize("seal", [True, False])
def test_cold_preparing_hint_survives_gaps_until_sealed_or_expired(monkeypatch, seal):
    import optimization.gpu.executor as executor

    clock, waiting, replay = [100.0], Queue(), Replay()
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter,
    ))
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    service.register_dataset("market", replay)
    ordinary = service._coalescing.preparing_deadline

    def deadline(dataset, started):
        value = ordinary(dataset, started)
        waiting.put(value)
        return value

    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"market"})
    try:
        first = service.submit(request(1))
        assert waiting.get(timeout=3) == 160
        clock[0] = 101  # Exceeds the ordinary five-millisecond idle delay.
        second = service.submit(request(2))
        assert waiting.get(timeout=3) == 160
        assert not replay.calls and not first.done()
        if seal:
            service.set_preparing(())
        else:
            # Repeating readiness cannot renew the original deadline. A new
            # arrival wakes the worker after the abandoned hint has expired.
            clock[0] = 161
            service.set_preparing({"market"})
            third = service.submit(request(3))
            assert third.result(timeout=3).request_id == "3"
        assert first.result(timeout=3).request_id == "1"
        assert second.result(timeout=3).request_id == "2"
        batches = [[row["value"] for row in batch] for batch in replay.calls]
        assert batches == ([[1, 2]] if seal else [[1, 2, 3]])
    finally:
        service.close(cancel_pending=True)


def test_sealing_preparing_producer_releases_a_partial_batch(monkeypatch):
    replay, waiting = Replay(), Event()
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    service.register_dataset("market", replay)
    def deadline(_dataset, started):
        waiting.set()
        return started + 30
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"market"})
    try:
        future = service.submit(request(1))
        assert waiting.wait(2)
        assert not future.done()
        service.set_preparing(())
        assert future.result(timeout=2).request_id == "1"
    finally:
        service.close(cancel_pending=True)


@pytest.mark.parametrize("cold", [False, True])
def test_unrelated_ready_dataset_keeps_the_ordinary_fairness_bound(monkeypatch, cold):
    import optimization.gpu.executor as executor

    clock, waiting, order = [100.0], Queue(), []
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter,
    ))
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    for dataset in ("a", "b"):
        class OrderedReplay(Replay):
            def evaluate(self, candidates, dataset=dataset):
                order.append(dataset)
                return super().evaluate(candidates)
        service.register_dataset(dataset, OrderedReplay())
    if not cold:
        warm_preparing_service(service, "a", clock[0])
    ordinary = service._coalescing.preparing_deadline
    def deadline(dataset, started):
        value = ordinary(dataset, started)
        waiting.put(value)
        return value
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"a"})
    try:
        first = service.submit(request(1, "a"))
        waiting.get(timeout=2)
        clock[0] = 100.3
        second = service.submit(request(2, "a"))
        waiting.get(timeout=2)
        clock[0] = 100.6
        other = service.submit(request(3, "b"))
        assert [future.result(timeout=2).request_id for future in (first, second, other)] == ["1", "2", "3"]
        assert order == ["a", "b"]
    finally:
        service.close(cancel_pending=True)


@pytest.mark.parametrize("cancel_close", [False, True])
def test_close_releases_or_cancels_preparing_work_without_waiting(monkeypatch, cancel_close):
    replay, waiting = Replay(), Event()
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    service.register_dataset("market", replay)
    def deadline(_dataset, started):
        waiting.set()
        return started + 30
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"market"})
    future = service.submit(request(1))
    assert waiting.wait(2)
    service.close(cancel_pending=cancel_close)
    assert future.cancelled() if cancel_close else future.result(timeout=2).request_id == "1"
    assert bool(replay.calls) is not cancel_close
    service.set_preparing(())  # Cleanup is safe after close.


def test_preparing_hint_does_not_override_an_explicit_delay(monkeypatch):
    replay, waiting = Replay(), Event()
    service = GpuBacktestService(batch_size=4, max_batch_delay=30)
    service.register_dataset("market", replay)
    wait = service._condition.wait
    def observe_wait(timeout=None):
        if timeout is not None:
            waiting.set()
        return wait(timeout)
    monkeypatch.setattr(service._condition, "wait", observe_wait)
    try:
        service.set_preparing(())
        future = service.submit(request(1))
        assert waiting.wait(2)
        assert not future.done()
        service.close()
        assert future.result(timeout=2).request_id == "1"
    finally:
        service.close(cancel_pending=True)


def test_preparing_hint_validates_before_mutating_and_empty_work_stays_lazy():
    constructed = []
    @contextmanager
    def factory():
        constructed.append(True)
        yield Replay()
    service = GpuBacktestService(max_batch_delay=None)
    service.register_dataset_factory("market", factory)
    service.set_preparing({"market"})
    for invalid in ("market", {"other"}, {1}):
        with pytest.raises(ValueError, match="preparing datasets"):
            service.set_preparing(invalid)
        assert service._preparing == frozenset({"market"})
    service.close()
    assert not constructed and service._thread is None


def test_cancelling_preparing_oldest_reselects_the_next_dataset(monkeypatch):
    waiting, first_replay, second_replay = Queue(), Replay(), Replay()
    service = GpuBacktestService(batch_size=4, max_batch_delay=None)
    service.register_dataset("a", first_replay)
    service.register_dataset("b", second_replay)
    def deadline(dataset, started):
        waiting.put(dataset)
        return started + 30
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"a", "b"})
    try:
        first = service.submit(request(1, "a"))
        assert waiting.get(timeout=2) == "a"
        second = service.submit(request(2, "b"))
        assert first.cancel()
        # The old dataset can wake once before its cancellation is observed.
        while waiting.get(timeout=2) != "b":
            pass
        service.set_preparing(())
        assert second.result(timeout=2).request_id == "2"
        assert first.cancelled() and not first_replay.calls
        assert len(second_replay.calls) == 1
    finally:
        service.close(cancel_pending=True)


def test_full_mixed_queue_releases_preparing_work_before_more_admission(monkeypatch):
    waiting, dispatched, order = Event(), Event(), []
    service = GpuBacktestService(batch_size=3, max_pending=3, max_batch_delay=None)
    for dataset in ("a", "b", "c"):
        class OrderedReplay(Replay):
            def evaluate(self, candidates, dataset=dataset):
                order.append(dataset)
                dispatched.set()
                return super().evaluate(candidates)
        service.register_dataset(dataset, OrderedReplay())
    def deadline(_dataset, started):
        waiting.set()
        return started + 30
    monkeypatch.setattr(service._coalescing, "preparing_deadline", deadline)
    service.set_preparing({"a", "b", "c"})
    try:
        futures = [service.submit(request(1, "a"))]
        assert waiting.wait(2)
        futures.extend(service.submit(request(index, dataset))
                       for index, dataset in ((2, "b"), (3, "c")))
        assert dispatched.wait(2) and order[0] == "a"
        service.set_preparing(())
        assert [future.result(timeout=2).request_id for future in futures] == ["1", "2", "3"]
        assert order == ["a", "b", "c"]
    finally:
        service.close(cancel_pending=True)


def test_late_preparing_hint_preserves_original_future_failure():
    failure, failed = ValueError("original replay exception"), Event()
    class BrokenReplay:
        def evaluate(self, candidates):
            raise failure
    service = GpuBacktestService(batch_size=1, max_batch_delay=None)
    service.register_dataset("market", BrokenReplay())
    service.set_preparing({"market"})
    try:
        future = service.submit(request(1))
        future.add_done_callback(lambda _future: failed.set())
        assert failed.wait(2)
        service.set_preparing({"market"})
        assert service._preparing == frozenset()
        with pytest.raises(ValueError) as raised:
            future.result(timeout=2)
        assert raised.value is failure
        with pytest.raises(RuntimeError, match="service failed"):
            service.submit(request(2))
    finally:
        service.close(cancel_pending=True)
    service.set_preparing({"market"})  # Closed readiness has no admission effect.
    assert service._preparing == frozenset()
