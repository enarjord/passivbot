import os
import pickle
import signal
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from optimization import native_planner_workers as preparation
from optimization.native_planner_workers import (
    CandidatePreparationError,
    CandidatePreparationQueueFull,
    OrderedCandidateWorkers,
    _Snapshot,
    _plan_parameters,
)
from test_native_planning import inputs


def planner_inputs(**kwargs):
    import passivbot_rust
    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("real native Rust extension required for spawned planners")
    return inputs(**kwargs)


def test_planner_snapshot_excludes_mapped_and_shared_arrays():
    planner, vector = planner_inputs(suite=True, fixed=True, mirror=True)
    # The normal evaluator unpickler attaches these. A planner snapshot carries
    # none of them, even when initially captured from a populated evaluator.
    sample = np.zeros(500_000)
    planner.base.shared_hlcvs_np["binance"] = sample
    evaluator = planner.scorer.evaluator
    evaluator._master_arrays["hlcvs"]["binance"] = sample
    for context in evaluator.contexts:
        context.timestamps = {"binance": sample}
        context.attachments = {"hlcvs": {}, "btc": {}}
        context.shared_hlcvs_np = {"binance": sample}
        context.shared_btc_np = {}
    detached = pickle.dumps(_Snapshot.capture(planner))
    assert len(detached) < sample.nbytes
    replica = pickle.loads(detached).build_planner()
    assert replica.prepare("candidate", vector) == planner.prepare("candidate", vector)


@pytest.mark.parametrize("suite", [False, True])
def test_spawned_plans_match_canonical_fields_and_shape_in_order(suite):
    planner, vector = planner_inputs(suite=suite, fixed=True, mirror=True)
    rng = np.random.default_rng(1)
    broken_order = [f"candidate:{index}" for index in (4, 0, 3, 1, 2)]
    expected = []
    with OrderedCandidateWorkers(planner, workers=2, max_pending=3) as workers:
        for index, candidate_id in enumerate(broken_order):
            vector = [rng.uniform(bound.low, bound.high) for bound in planner.base.bounds]
            scenarios = ["stress"] if suite and index % 2 else None
            expected.append(planner.prepare(candidate_id, vector, scenarios=scenarios))
            workers.submit(candidate_id, vector, scenarios=scenarios)
            # The worker prepares a detached vector as submitted.
            vector[:] = ()
            if index == 2:
                assert workers.pending_count == 3
                with pytest.raises(CandidatePreparationQueueFull):
                    workers.submit("overflow", ())
            if index >= 2:
                plan = workers.poll(timeout=20)
                assert plan == expected[index - 2]
                assert all(type(row.parameters) is MappingProxyType for row in plan.requests)
                assert _plan_parameters(plan, freeze=False) == _plan_parameters(expected[index - 2], freeze=False)
        assert [workers.poll(timeout=20) for _ in range(2)] == expected[-2:]
        assert workers.pending_count == 0 and workers.poll() is None
        stats = workers.resource_snapshot()
        assert len(stats) == 2 and sum(item["plans"] for item in stats.values()) == len(expected)
        assert all(item["cpu_seconds"] > 0 and item["peak_sampled_rss_bytes"] > 0
                   and item["peak_plan_bytes"] > 0 for item in stats.values())


def test_ordered_failure_discards_future_rows_without_masking_planner_error():
    planner, vector = planner_inputs(suite=True)
    with OrderedCandidateWorkers(planner, workers=2, max_pending=3) as workers:
        workers.submit("first", vector)
        workers.submit("bad", ())
        workers.submit("later", vector)
        assert workers.poll(timeout=20).candidate_id == "first"
        with pytest.raises(CandidatePreparationError, match="bad: ValueError: candidate must have") as caught:
            workers.poll(timeout=20)
        assert caught.value.candidate_id == "bad" and "native_planning.py" in caught.value.remote_traceback
        with pytest.raises(RuntimeError, match="unavailable"):
            workers.poll()


def test_interrupt_preempts_delivery_and_reaps_owned_worker():
    planner, vector = planner_inputs()
    workers = OrderedCandidateWorkers(planner, workers=1, max_pending=2)
    with pytest.raises(KeyboardInterrupt), workers:
        process = workers._workers[0].process
        pid = process.pid
        workers.submit("running", vector)
        workers.submit("queued", vector)

        def interrupt():
            raise KeyboardInterrupt

        workers.poll(timeout=20, interrupt_check=interrupt)
    assert workers.pending_count == 0
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_dead_worker_fails_its_ordered_row_after_other_worker_result():
    planner, vector = planner_inputs()
    with OrderedCandidateWorkers(planner, workers=2, max_pending=3) as workers:
        workers.submit("first", vector)
        workers.submit("dies", vector)
        pid = workers._workers[1].process.pid
        os.kill(pid, signal.SIGKILL)
        workers.submit("later", vector)
        assert workers.poll(timeout=20).candidate_id == "first"
        with pytest.raises(CandidatePreparationError, match="dies:"):
            workers.poll(timeout=20)


def _wait_without_initializing(connection):
    with connection:
        connection.recv_bytes()
        connection.recv()


def _started_pids(monkeypatch, workers):
    pids = []
    start = workers._start_process

    def track(process):
        start(process)
        pids.append(process.pid)

    monkeypatch.setattr(workers, "_start_process", track)
    return pids


def _assert_reaped(pids, workers, count):
    assert len(pids) == count and workers.pending_count == 0
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    with pytest.raises(RuntimeError, match="unavailable"):
        workers.submit("later", ())


def test_every_worker_is_verified_before_admission(monkeypatch):
    planner, vector = planner_inputs(suite=True)
    workers = OrderedCandidateWorkers(planner, workers=2, max_pending=3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("SKIP_RUST_COMPILE", raising=False)
    with workers:
        assert len(workers.resource_snapshot()) == 2
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        assert "SKIP_RUST_COMPILE" not in os.environ
        workers.submit("ready", vector)
        assert workers.poll(timeout=20) == planner.prepare("ready", vector)


def test_startup_preserves_rust_initializer_fault_and_reaps_workers(monkeypatch):
    planner, _ = planner_inputs()
    snapshot = replace(_Snapshot.capture(planner), rust_artifact_sha256="0" * 64)
    workers = OrderedCandidateWorkers(planner, workers=2, max_pending=3)
    workers._snapshot = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    pids = _started_pids(monkeypatch, workers)

    with pytest.raises(CandidatePreparationError, match="startup failed: RuntimeError: .*Rust extension differs") as caught:
        with workers:
            pytest.fail("A failed initializer must prevent plan submission")
    assert caught.value.candidate_id is None
    assert "build_planner" in caught.value.remote_traceback
    _assert_reaped(pids, workers, 2)


def test_startup_interrupt_reaps_workers_and_preserves_the_interrupt(monkeypatch):
    planner, _ = planner_inputs()
    ready_checks = 0
    original = KeyboardInterrupt("interrupted while waiting for initializers")

    def interrupt():
        nonlocal ready_checks
        if len(pids) == 2:
            ready_checks += 1
            if ready_checks == 2:
                raise original

    workers = OrderedCandidateWorkers(planner, workers=2, max_pending=3, interrupt_check=interrupt)
    pids = _started_pids(monkeypatch, workers)
    monkeypatch.setattr(preparation, "_prepare_worker", _wait_without_initializing)
    with pytest.raises(KeyboardInterrupt, match="waiting for initializers") as caught:
        with workers:
            pytest.fail("An interrupted startup must prevent plan submission")
    assert caught.value is original
    _assert_reaped(pids, workers, 2)


def test_startup_deadline_bounds_an_unresponsive_worker(monkeypatch):
    planner, _ = planner_inputs()
    workers = OrderedCandidateWorkers(planner, workers=1, max_pending=2)
    pids = _started_pids(monkeypatch, workers)
    monkeypatch.setattr(preparation, "_STARTUP_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(preparation, "_prepare_worker", _wait_without_initializing)
    with pytest.raises(TimeoutError, match="startup deadline"):
        with workers:
            pytest.fail("An unresponsive initializer must prevent plan submission")
    _assert_reaped(pids, workers, 1)
