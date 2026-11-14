import os
import pickle
import signal
from types import MappingProxyType

import numpy as np
import pytest

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
