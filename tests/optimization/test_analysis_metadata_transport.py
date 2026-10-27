"""Analysis diagnostics cross the GPU/result boundary without becoming objectives."""

from copy import deepcopy
from dataclasses import replace

import pytest

from optimization.gpu.executor import (
    BacktestRequest,
    BacktestResult,
    GpuBacktestService,
    ReplayResult,
)
from optimization.native_results import CandidateEvaluation, ResultSlot


def _metadata():
    return {"metric_status": {}, "metric_metadata": {
        "definition_version": 1, "ew_half_life_days": 180.5,
        "equity_observation_count": 30}}


def test_executor_metadata_is_separate_and_copied_for_each_result():
    source = _metadata()
    expected = deepcopy(source)

    class Replay:
        def evaluate(self, candidates):
            return [ReplayResult({"cagr_fraction_strategy_equity_ew": candidate["gain_multiplier"]}, False, source)
                    for candidate in candidates]

    with GpuBacktestService(batch_size=2, max_batch_delay=1) as service:
        service.register_dataset("data", Replay())
        futures = [service.submit(BacktestRequest(str(index), "data", {"gain_multiplier": index * .01}))
                   for index in (1, 2)]
        results = [future.result(timeout=3) for future in futures]
    assert [row.metrics for row in results] == [{"cagr_fraction_strategy_equity_ew": .01}, {"cagr_fraction_strategy_equity_ew": .02}]
    assert [row.metadata for row in results] == [expected, expected]
    source["metric_metadata"]["ew_half_life_days"] = 999
    assert results[0].metadata == expected
    results[0].metadata["metric_metadata"]["equity_observation_count"] = -1
    assert results[1].metadata == expected


@pytest.mark.parametrize("invalid", [False, 1, "metadata", []])
def test_executor_invalid_metadata_is_fatal_for_complete_batch(invalid):
    class Replay:
        def evaluate(self, candidates):
            rows = [ReplayResult({"cagr_fraction_strategy_equity_ew": .01}, False, _metadata())
                    for _ in candidates]
            rows[-1] = replace(rows[-1], metadata=invalid)
            return rows

    with GpuBacktestService(batch_size=2, max_batch_delay=1) as service:
        service.register_dataset("data", Replay())
        futures = [service.submit(BacktestRequest(str(index), "data", {})) for index in (1, 2)]
        for future in futures:
            with pytest.raises(RuntimeError, match="metadata must be a mapping"):
                future.result(timeout=3)
        with pytest.raises(RuntimeError, match="service failed"):
            service.submit(BacktestRequest("new", "data", {}))


class CaptureScorer:
    def __init__(self, slots):
        self.coverage = {(slot.scenario, slot.exchange) for slot in slots}
        self.analyses = None

    def validate_coverage(self, pairs, stage):
        assert pairs == self.coverage
        assert stage == "full"

    def required_metrics(self, scenario):
        return {"cagr_fraction_strategy_equity_ew"}

    def score(self, vector, analyses, *, stage="full"):
        assert stage == "full"
        self.analyses = deepcopy(analyses)
        return {"metrics": {"objectives": {"cagr_fraction_strategy_equity_ew": .01}}, "fitness": (-.01,)}


def _slots():
    return [ResultSlot(f"{scenario}:{exchange}", f"data:{scenario}:{exchange}", scenario,
                       exchange, ("cagr_fraction_strategy_equity_ew",))
            for scenario, exchange in (("base", "first"), ("base", "second"), ("stress", "first"))]


def _result(slot, metadata):
    return BacktestResult(slot.request_id, slot.dataset_id, {"cagr_fraction_strategy_equity_ew": .01}, False, metadata)


def test_candidate_completion_preserves_each_metadata_source_and_scalar_separation():
    slots = _slots()
    scorer = CaptureScorer(slots)
    pending = CandidateEvaluation("candidate", [.5], slots, scorer)
    first_source = _metadata()
    first_expected = deepcopy(first_source)
    assert pending.add_result(_result(slots[0], first_source)) is None
    first_source["metric_metadata"]["ew_half_life_days"] = 999
    second_source = _metadata()
    second_source["metric_metadata"]["equity_observation_count"] = 31
    assert pending.add_result(_result(slots[1], second_source)) is None
    second_expected = deepcopy(second_source)
    second_source["metric_metadata"]["equity_observation_count"] = 999
    completion = pending.add_result(_result(slots[2], None)).require_full()
    assert completion["metrics"]["analysis_metadata"] == {
        "base": {"first": first_expected, "second": second_expected}
    }
    assert set(scorer.analyses) == {"base", "stress"}
    for slot in slots:
        analysis = scorer.analyses[slot.scenario][slot.exchange]
        expected = {"cagr_fraction_strategy_equity_ew": .01, "liquidated": False}
        if slot == slots[0]:
            expected.update(first_expected)
        elif slot == slots[1]:
            expected.update(second_expected)
        assert analysis == expected
    assert pending._rows == {}
    assert pending._metadata == {}
    assert pending.pending_request_ids == ()


@pytest.mark.parametrize("invalid", [False, 1, "metadata", []])
def test_candidate_invalid_metadata_never_satisfies_slot(invalid):
    slots = _slots()
    pending = CandidateEvaluation("candidate", [.5], slots, CaptureScorer(slots))
    with pytest.raises(RuntimeError, match="metadata must be a mapping"):
        pending.add_result(_result(slots[0], invalid))
    assert pending.pending_request_ids == tuple(slot.request_id for slot in slots)
    assert pending._rows == {}
    assert pending._metadata == {}
    assert pending.add_result(_result(slots[0], _metadata())) is None


def test_metadata_absence_retains_existing_candidate_payload():
    slots = _slots()
    pending = CandidateEvaluation("candidate", [.5], slots, CaptureScorer(slots))
    for slot in slots:
        completion = pending.add_result(_result(slot, None))
    assert "analysis_metadata" not in completion.require_full()["metrics"]
