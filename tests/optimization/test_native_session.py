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


def test_screening_cache_cannot_satisfy_full_candidate_evaluation():
    service = Service()
    session = NativeEvaluationSession(service, scorer())
    session.admit(replace(plan(0), stage="screening"))
    session.poll()
    for name in tuple(service.requests):
        service.complete(name)
    screening = session.poll()[0]
    with pytest.raises(ValueError, match="screening"):
        screening.require_full()
    session.admit(plan(1, key="0"))
    session.poll()
    assert len(service.requests) == 4
    for name in ("1:binance", "1:bybit"):
        service.complete(name)
    assert session.poll()[0].require_full()["fitness"]


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
