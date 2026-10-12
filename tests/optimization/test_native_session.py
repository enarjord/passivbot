from concurrent.futures import Future
from dataclasses import replace
from threading import Thread, get_ident
from types import SimpleNamespace

import pytest

from config.schema import get_template_config
from optimization.gpu.executor import BacktestRequest, BacktestResult, BacktestQueueFull
from optimization.native_planning import CandidatePlan
from optimization.native_results import CanonicalResultScorer, ResultSlot
from optimization.native_session import CandidateQueueFull, NativeEvaluationSession
from optimize import Evaluator, SuiteEvaluator


METRICS = ("adg_strategy_eq", "drawdown_worst_usd")


class Service:
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.requests, self.futures = {}, {}

    def submit(self, request):
        if sum(not future.done() for future in self.futures.values()) >= self.capacity:
            raise BacktestQueueFull("busy")
        future = Future()
        self.requests[request.request_id], self.futures[request.request_id] = request, future
        return future

    def complete(self, request_id, failure=None):
        future, request = self.futures[request_id], self.requests[request_id]
        if failure is not None:
            future.set_exception(failure)
        else:
            future.set_result(BacktestResult(request.request_id, request.dataset_id,
                                            {"adg_strategy_eq": 0.01, "drawdown_worst_usd": 0.1}, False))


def scorer():
    config = get_template_config()
    config["optimize"]["scoring"] = [{"metric": "adg_strategy_eq", "goal": "max"}]
    config["optimize"]["limits"] = []
    return CanonicalResultScorer(Evaluator({"binance": None, "bybit": None}, {}, {}, config))


def plan(index, *, key=None):
    slots = tuple(ResultSlot(f"{index}:{venue}", venue, "base", venue, METRICS) for venue in ("binance", "bybit"))
    requests = tuple(BacktestRequest(slot.request_id, slot.dataset_id, {}) for slot in slots)
    return CandidatePlan(str(index), (float(index),), key or str(index), "full", requests, slots)


def test_bounded_admission_refills_service_and_scores_on_cpu_poller(monkeypatch):
    service, value = Service(), scorer()
    caller = get_ident()
    original = value.score
    def scored(*args):
        assert get_ident() == caller
        return original(*args)
    monkeypatch.setattr(value, "score", scored)
    session = NativeEvaluationSession(service, value, max_candidates=2)
    session.admit(plan(0))
    session.admit(plan(1))
    with pytest.raises(CandidateQueueFull):
        session.admit(plan(2))
    assert session.poll() == []
    assert len(service.requests) == 2
    producer = Thread(target=lambda: [service.complete(name) for name in tuple(service.requests)])
    producer.start()
    producer.join()
    assert [item.candidate_id for item in session.poll()] == ["0"]
    assert len(service.requests) == 4
    assert session.active_candidate_ids == ("1",)
    for request_id in ("1:binance", "1:bybit"):
        service.complete(request_id)
    assert [item.candidate_id for item in session.poll()] == ["1"]
    assert session.active_candidate_ids == ()
    assert session.pending_request_count == 0


def test_effective_duplicates_share_pending_work_and_bounded_snapshot_cache():
    service = Service()
    session = NativeEvaluationSession(service, scorer(), cache_size=1)
    session.admit(plan(0))
    session.admit(plan(1, key="0"))
    session.poll()
    assert len(service.requests) == 2
    for name in tuple(service.requests):
        service.complete(name)
    completed = session.poll()
    assert [item.candidate_id for item in completed] == ["0", "1"]
    assert [item.payload["evaluation_vector"] for item in completed] == [[0.0], [1.0]]
    completed[0].payload["metrics"]["objectives"] = {}
    session.admit(plan(2, key="0"))
    cached = session.poll()[0]
    assert cached.payload["metrics"]["objectives"]
    assert len(service.requests) == 2
    session.admit(plan(3))
    session.poll()
    for name in ("3:binance", "3:bybit"):
        service.complete(name)
    session.poll()
    assert len(session._cache) == 1
    session.admit(plan(4, key="0"))
    session.poll()
    assert len(service.requests) == 6
    session.stop_admission()


def test_stop_cancels_waiting_work_and_preserves_completed_candidates():
    service = Service(capacity=4)
    session = NativeEvaluationSession(service, scorer())
    session.admit(plan(0))
    session.admit(plan(1))
    session.poll()
    for name in ("0:binance", "0:bybit"):
        service.complete(name)
    service.futures["1:binance"].set_running_or_notify_cancel()
    session.stop_admission()
    assert service.futures["1:bybit"].cancelled()
    assert [item.candidate_id for item in session.poll()] == ["0"]
    service.complete("1:binance")
    assert session.poll() == []
    assert session.pending_request_count == 0
    assert session.active_candidate_ids == ("1",)
    with pytest.raises(RuntimeError, match="stopped"):
        session.admit(plan(2))


def test_producer_failure_stops_new_admission_and_cancels_other_work():
    service = Service()
    session = NativeEvaluationSession(service, scorer())
    session.admit(plan(0))
    session.poll()
    failure = RuntimeError("device failure")
    service.complete("0:binance", failure)
    with pytest.raises(RuntimeError) as raised:
        session.poll()
    assert raised.value is failure
    assert service.futures["0:bybit"].cancelled()
    with pytest.raises(RuntimeError, match="session failed"):
        session.admit(plan(1))


@pytest.mark.parametrize("future_failure", [False, True])
def test_submission_failure_preserves_prior_completion_and_original_future_error(future_failure):
    original = ValueError("original producer failure")
    submission = RuntimeError("service failed during replenishment")
    class FailedAfterReady(Service):
        def submit(self, request):
            count = len(self.requests)
            if count >= (3 if future_failure else 2):
                raise submission
            future = super().submit(request)
            self.complete(request.request_id, original if count == 2 else None)
            return future
    session = NativeEvaluationSession(FailedAfterReady(capacity=4), scorer())
    session.admit(plan(0))
    session.admit(plan(1))
    assert [row.candidate_id for row in session.poll()] == ["0"]
    with pytest.raises(ValueError if future_failure else RuntimeError) as raised:
        session.poll()
    assert raised.value is (original if future_failure else submission)


def test_cpu_admission_snapshots_payload_before_waiting_for_device_capacity():
    service = Service()
    session = NativeEvaluationSession(service, scorer())
    candidate = plan(0)
    parameters = {"value": [0.01]}
    candidate = replace(candidate, requests=tuple(replace(request, parameters=parameters)
                                                 for request in candidate.requests))
    session.admit(candidate)
    parameters["value"][0] = 0.05
    session.poll()
    assert all(request.parameters == {"value": [0.01]} for request in service.requests.values())
    session.stop_admission()


def test_failed_snapshot_does_not_partially_admit_candidate():
    class InvalidPayload:
        def __deepcopy__(self, _memo):
            raise ValueError("cannot snapshot")
    session = NativeEvaluationSession(Service(), scorer())
    candidate = plan(0)
    candidate = replace(candidate, requests=tuple(replace(request, parameters={"value": InvalidPayload()})
                                                 for request in candidate.requests))
    with pytest.raises(ValueError, match="cannot snapshot"):
        session.admit(candidate)
    assert session.active_candidate_ids == ()
    assert session.pending_request_count == 0
    session.admit(plan(0))
    session.stop_admission()


def suite_scorer():
    return CanonicalResultScorer(SuiteEvaluator(scorer().base, [SimpleNamespace(
        label=label, exchanges=["binance", "bybit"],
    ) for label in ("base", "stress")], {"default": "mean"}))


def suite_plan(index, *, key="same", labels=("base", "stress"), namespace=""):
    slots = tuple(ResultSlot(f"{index}:{label}:{venue}", f"{namespace}{label}:{venue}",
                             label, venue, METRICS) for label in labels for venue in ("binance", "bybit"))
    requests = tuple(BacktestRequest(slot.request_id, slot.dataset_id, {}) for slot in slots)
    return CandidatePlan(str(index), (float(index),), key,
                         "full" if len(labels) == 2 else "screening", requests, slots)


def test_screening_rows_reuse_simulations_but_cannot_satisfy_full_suite_score(monkeypatch):
    service = Service()
    value = suite_scorer()
    scored = []
    original = value.score
    def score(vector, analyses):
        scored.append(tuple(analyses))
        return original(vector, analyses)
    monkeypatch.setattr(value, "score", score)
    session = NativeEvaluationSession(service, value)
    session.admit(suite_plan(0, labels=("base",)))
    session.poll()
    for name in tuple(service.requests):
        service.complete(name)
    screening = session.poll()[0]
    with pytest.raises(ValueError, match="screening"):
        screening.require_full()
    session.admit(suite_plan(1))
    assert session.poll() == []
    assert len(service.requests) == 4
    assert scored == [("base",)]
    assert not any(name.startswith("1:base") for name in service.requests)
    for name in ("1:stress:binance", "1:stress:bybit"):
        service.complete(name)
    assert session.poll()[0].require_full()["fitness"]
    assert scored == [("base",), ("base", "stress")]


@pytest.mark.parametrize("changed", ["candidate", "dataset", "parameters"])
def test_reused_rows_require_identical_effective_candidate_and_dataset(changed):
    service = Service(capacity=4)
    session = NativeEvaluationSession(service, suite_scorer())
    session.admit(suite_plan(0, labels=("base",)))
    session.poll()
    for name in tuple(service.requests):
        service.complete(name)
    session.poll()
    promoted = suite_plan(1, key="different" if changed == "candidate" else "same",
                          namespace="different:" if changed == "dataset" else "")
    if changed == "parameters":
        promoted = replace(promoted, requests=tuple(replace(request, parameters={"value":0.5})
                                                   for request in promoted.requests))
    session.admit(promoted)
    session.poll()
    assert len(service.requests) == 6
    session.stop_admission()


def test_row_cache_eviction_only_repeats_work_and_cached_rows_are_snapshots():
    service = Service(capacity=4)
    value = suite_scorer()
    session = NativeEvaluationSession(service, value, cache_size=1)
    session.admit(suite_plan(0, labels=("base",)))
    session.poll()
    for name in tuple(service.requests):
        service.complete(name)
    session.poll()
    assert len(session._row_cache) == 1
    # Mutation after consumption must not corrupt retained simulator evidence.
    service.futures["0:base:bybit"].result().metrics["adg_strategy_eq"] = 999
    session.admit(suite_plan(1))
    session.poll()
    assert len(service.requests) == 5  # One evicted base row repeats on the service.
    for name in tuple(service.requests):
        if name.startswith("1:"):
            service.complete(name)
    result = session.poll()[0].require_full()
    assert result["fitness"] == pytest.approx((-0.01,))
    assert len(session._row_cache) == 1


def test_misbound_future_result_is_rejected_before_it_can_enter_row_cache():
    service = Service()
    session = NativeEvaluationSession(service, scorer())
    candidate = plan(0)
    session.admit(candidate)
    session.poll()
    first, second = candidate.requests
    service.futures[first.request_id].set_result(BacktestResult(
        second.request_id, second.dataset_id, {"adg_strategy_eq":0.01, "drawdown_worst_usd":0.1}, False,
    ))
    with pytest.raises(RuntimeError, match="submitted request"):
        session.poll()
    assert not session._row_cache


def test_screening_subsets_do_not_share_pending_work_or_completed_cache():
    base = scorer().base
    value = CanonicalResultScorer(SuiteEvaluator(base, [SimpleNamespace(
        label=label, exchanges=["binance", "bybit"],
    ) for label in ("base", "stress")], {"default": "mean"}))
    service = Service(capacity=4)
    session = NativeEvaluationSession(service, value)
    for index, label in enumerate(("base", "stress")):
        candidate = plan(index, key="same-complete-candidate")
        slots = tuple(replace(slot, scenario=label, dataset_id=f"{label}:{slot.exchange}")
                      for slot in candidate.slots)
        requests = tuple(replace(request, dataset_id=slot.dataset_id) for request, slot in
                         zip(candidate.requests, slots, strict=True))
        session.admit(replace(candidate, stage="screening", requests=requests, slots=slots))
    session.poll()
    assert len(service.requests) == 4
    for name in tuple(service.requests):
        service.complete(name)
    completed = session.poll() + session.poll()
    assert {row.candidate_id for row in completed} == {"0", "1"}
    assert len(session._cache) == 2


def test_unexpected_service_cancellation_is_fatal_and_cancels_other_work():
    service = Service()
    session = NativeEvaluationSession(service, scorer())
    session.admit(plan(0))
    session.poll()
    service.futures["0:binance"].cancel()
    with pytest.raises(RuntimeError, match="cancelled unexpectedly"):
        session.poll()
    assert service.futures["0:bybit"].cancelled()


def test_request_slot_mismatch_is_rejected_before_admission():
    session = NativeEvaluationSession(Service(), scorer())
    candidate = plan(0)
    candidate = replace(candidate, requests=(replace(candidate.requests[0], dataset_id="wrong"),
                                             candidate.requests[1]))
    with pytest.raises(ValueError, match="exactly match"):
        session.admit(candidate)
    assert session.active_candidate_ids == ()
    assert session.pending_request_count == 0


def test_notification_bound_does_not_limit_suite_fan_in_to_completion_batch_size():
    service = Service()
    session = NativeEvaluationSession(service, scorer())
    session.admit(plan(0))
    session.poll(max_completions=1)
    for name in tuple(service.requests):
        service.complete(name)
    # Two venue notifications must be consumed to complete this one candidate.
    assert [row.candidate_id for row in session.poll(max_results=2, max_completions=1)] == ["0"]
    assert session.pending_request_count == 0


def test_small_completion_batches_preserve_ready_aliases_without_more_backtests():
    service = Service()
    session = NativeEvaluationSession(service, scorer(), max_candidates=8)
    for index in range(6):
        session.admit(plan(index, key="shared"))
    session.poll(max_completions=2)
    for name in tuple(service.requests):
        service.complete(name)
    for expected in (("0", "1"), ("2", "3"), ("4", "5")):
        assert tuple(row.candidate_id for row in session.poll(max_completions=2)) == expected
    assert len(service.requests) == 2
    assert not session.active_candidate_ids


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_completion_bound_does_not_poison_session(value):
    session = NativeEvaluationSession(Service(), scorer())
    with pytest.raises(ValueError, match="max_completions"):
        session.poll(max_completions=value)
    session.admit(plan(0))
    session.stop_admission()
