import hashlib
import json

import pytest

from config.metrics import CORE_METRIC_DEFINITION_VERSION
from pareto_qualification import load_qualification, qualification_for_candidates


GROWTH = "cagr_fraction_strategy_equity"
HSL_LOSS = "hsl_panic_episode_loss_max_fraction"


@pytest.fixture
def example(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"bot":{}}\n')
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    scenario = {
        "split": "holdout",
        "execution": "stress",
        "start_date": "2026-01-01",
        "end_date": "2026-02-01",
        "inputs_sha256": "b" * 64,
    }
    result = {
        "status": "completed",
        "report": "replays/holdout/analysis.json",
        "effective_start_date": "2026-01-01T00:00:00Z",
        "effective_end_date": "2026-02-01T23:59:00Z",
        "liquidated": False,
        "metrics": {
            GROWTH: 0.35,
            HSL_LOSS: None,
            "metric_status": {HSL_LOSS: {"status": "undefined", "reason": "no_episode"}},
        },
    }
    subject = {"effective_config_sha256": "c" * 64, "results": {"stress": result}}
    evidence = {
        "schema_version": 1,
        "definition_version": CORE_METRIC_DEFINITION_VERSION,
        "engine": "rust_cpu",
        "engine_sha256": "a" * 64,
        "scenarios": {"stress": scenario},
        "candidates": {digest: subject},
    }
    return candidate, digest, evidence, scenario, result


def _write(tmp_path, evidence):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    return path


def test_qualification_only_attaches_to_exact_candidate_bytes(tmp_path, example):
    candidate, digest, evidence, _, _ = example
    evidence["candidates"]["d" * 64] = {
        "effective_config_sha256": "d" * 64,
        "results": {"stress": {"status": "unavailable", "reason": "missing data"}},
    }
    evidence["incumbent"] = evidence["candidates"][digest]
    loaded = load_qualification(_write(tmp_path, evidence))
    attached = qualification_for_candidates(loaded, [digest])
    assert list(attached["candidates"]) == [digest]
    assert attached["incumbent"] == evidence["incumbent"]
    assert attached["candidates"][digest]["results"]["stress"]["metrics"]["metric_status"] == {
        HSL_LOSS: {"status": "undefined", "reason": "no_episode"}
    }

    # The filename remains identical while the artifact identity changes.
    candidate.write_text('{"bot":{"long":{}}}\n')
    changed_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert qualification_for_candidates(loaded, [changed_digest])["candidates"] == {}
    assert load_qualification(None) is None
    assert qualification_for_candidates(None, [digest]) is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("definition_version", -1, "definition_version"),
        ("engine", "gpu", "engine"),
        ("engine_sha256", "z" * 64, "SHA-256"),
        ("incumbent", {"results": {}}, "effective_config_sha256"),
        ("candidates", {}, "candidates must be a nonempty object"),
    ],
)
def test_qualification_rejects_invalid_envelopes(tmp_path, example, field, value, message):
    _, _, evidence, _, _ = example
    evidence[field] = value
    with pytest.raises(ValueError, match=message):
        load_qualification(_write(tmp_path, evidence))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "untouched?", "split"),
        ("execution", [], "execution"),
        ("start_date", "2026-02-02", "after end_date"),
        ("end_date", "2026-02-30", "ISO 8601"),
    ],
)
def test_qualification_rejects_invalid_scenarios(tmp_path, example, field, value, message):
    _, _, evidence, scenario, _ = example
    scenario[field] = value
    with pytest.raises(ValueError, match=message):
        load_qualification(_write(tmp_path, evidence))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "approved", "status must be one of"),
        ("status", "terminated", "reason"),
        ("status", "unavailable", "reason"),
        ("report", "", "printable string"),
        ("report", "\x1b[2Jreport.json", "printable string"),
        ("effective_end_date", "2026-02-02T00:00:00Z", "within its scenario"),
        ("effective_end_date", "2025-01-01T00:00:00Z", "ordered"),
        ("effective_start_date", "2026-01-01", "UTC timestamp"),
        ("liquidated", "false", "must be a boolean"),
        ("liquidated", True, "terminated status"),
        ("metrics", {"old_alias": 2}, "known raw metrics"),
        ("metrics", {GROWTH: None}, "unavailable"),
        ("metrics", {GROWTH: float("inf")}, "malformed finite metric"),
        ("metrics", {GROWTH: 10**320}, "too large"),
        ("metrics", {GROWTH: True}, "malformed finite metric"),
        ("metrics", {GROWTH: 2, "metric_status": {GROWTH: {"status": "invalid"}}}, "must be null"),
        ("metrics", {GROWTH: None, "metric_status": {GROWTH: {"status": "missing"}}}, "must be one of"),
        ("metrics", {GROWTH: 2, "metric_status": {HSL_LOSS: {"status": "undefined"}}}, "present metrics"),
    ],
)
def test_qualification_rejects_invalid_result_facts(tmp_path, example, field, value, message):
    _, _, evidence, _, result = example
    result[field] = value
    with pytest.raises(ValueError, match=message):
        load_qualification(_write(tmp_path, evidence))


def test_qualification_preserves_early_termination_and_infinite_fact(tmp_path, example):
    _, digest, evidence, _, result = example
    result.update(
        status="terminated",
        reason="liquidation",
        liquidated=True,
        effective_end_date="2026-01-02T00:05:00Z",
        metrics={GROWTH: None, "metric_status": {GROWTH: {"status": "negative_infinity"}}},
    )
    loaded = load_qualification(_write(tmp_path, evidence))
    assert loaded["candidates"][digest]["results"]["stress"] == result


def test_qualification_rejects_duplicate_json_keys(tmp_path, example):
    _, _, evidence, _, _ = example
    path = _write(tmp_path, evidence)
    path.write_text(path.read_text().replace('"schema_version":', '"schema_version": 0, "schema_version":'))
    with pytest.raises(ValueError, match="duplicate object key"):
        load_qualification(path)
