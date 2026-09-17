"""Offline regressions for rank signal, objective evidence, and restart parity."""

import argparse
import copy
import pickle

import numpy as np
import pytest

from config.schema import get_template_config
from config_utils import add_config_arguments, format_config, update_config_with_args
from optimization.backends.gpu_backend import (
    GPU_DEFAULTS,
    _DriftMonitor,
    _ObjectiveScale,
    _drift_objective_pair,
    _recover_durable_seed_bootstrap,
    _recover_durable_validations,
    _resolve_options,
    _select_validation_indices,
)


def _monitor(proxy, exact, **options):
    proxy = np.asarray(proxy, dtype=float)
    exact = np.asarray(exact, dtype=float)
    monitor = _DriftMonitor(dict(GPU_DEFAULTS, **options), objective_count=proxy.shape[1])
    for i, (p, e) in enumerate(zip(proxy, exact)):
        monitor.add(
            float(np.mean(p)), float(np.mean(e)), probe=i >= 19, proxy_front=i < 19,
            proxy_objectives=p, exact_objectives=e,
        )
    return monitor


def _near_ties(epsilon):
    rng = np.random.default_rng(17)
    proxy = rng.normal(size=109)
    exact = proxy + 1.136 * rng.normal(size=109)
    front = np.arange(19)[:, None] / 100 - 2
    return (
        np.concatenate([front, (1 + epsilon * proxy)[:, None]]),
        np.concatenate([front, (1 + epsilon * exact)[:, None]]),
    )


@pytest.mark.parametrize("epsilon,halts", [(1.0, True), (1e-8, False), (1e-12, False)])
def test_near_ties_require_small_actual_objective_errors(epsilon, halts):
    status = _monitor(*_near_ties(epsilon)).evaluate()
    assert status["probe_rho"] == pytest.approx(0.591418774905, abs=1e-4)
    assert status["constraint_agreement"] == 1.0
    assert bool(status["halt_reason"]) == halts
    assert status["probe_objective_samples"] == 109
    assert status["probe_score_agreement"]["max_abs_error"] == pytest.approx(3.0029 * epsilon, rel=1e-3)
    if not halts:
        assert "per-objective broad probes remain sound" in status["warn_reason"]


def test_zero_tolerance_does_not_exempt_nonzero_near_tie_errors():
    assert _monitor(*_near_ties(1e-8), drift_objective_tolerance=0).evaluate()["halt_reason"]


@pytest.mark.parametrize("constant", [0.0, 1.0])
def test_identical_constant_objectives_are_not_disagreement(constant):
    values = np.full((128, 2), constant)
    status = _monitor(values, values).evaluate()
    assert np.isnan(status["probe_rho"])
    assert status["halt_reason"] is None
    assert status["warn_reason"]


def test_selection_and_monitor_preserve_perfect_opposing_objectives():
    scale = _ObjectiveScale()
    scale.fit(np.array([[-1, -1], [0, 0], [1, 1]]))
    monitor = _DriftMonitor(GPU_DEFAULTS, objective_count=2)
    for generation in range(16):
        # The single front point dominates a changing, genuinely off-front
        # population. Both individual objectives vary but their mean is zero.
        x = np.linspace(-1, 1, 109) + generation / 1000
        objectives = np.vstack([[-2, -2], np.column_stack([x, -x])])
        scores = scale.score(objectives)
        selected = _select_validation_indices(objectives, scores, total=8, probes=4)[:8]
        for index, probe, front in selected:
            monitor.add(
                scores[index], scores[index], probe=probe, proxy_front=front,
                proxy_objectives=scale.normalize(objectives)[index],
                exact_objectives=scale.normalize(objectives)[index],
            )
            assert monitor.evaluate()["halt_reason"] is None
    status = monitor.evaluate()
    assert status["probes"] == 112
    assert status["front_samples"] == 16
    assert np.isnan(status["probe_rho"])
    assert [item["rho"] for item in status["probe_objective_agreement"]] == [1.0, 1.0]


def test_sound_component_ranks_can_explain_scalar_cancellation():
    x = np.arange(128, dtype=float)
    proxy = np.column_stack([x, -x + x * 1e-3])
    exact = np.column_stack([x, -x - x * 1e-3])
    status = _monitor(proxy, exact).evaluate()
    assert status["probe_rho"] == -1.0
    assert all(item["rho"] == 1.0 for item in status["probe_objective_agreement"])
    assert status["halt_reason"] is None


@pytest.mark.parametrize("kind", ["flat_proxy", "flat_exact", "offset", "inversion", "one_bad_objective"])
def test_material_disagreement_still_halts(kind):
    x = np.arange(128, dtype=float)
    proxy = np.column_stack([x, x])
    exact = proxy.copy()
    if kind == "flat_proxy":
        proxy[:] = 0
    elif kind == "flat_exact":
        exact[:] = 0
    elif kind == "offset":
        proxy[:] = 0
        exact[:] = 1
    elif kind == "inversion":
        exact *= -1
    else:
        exact[:, 1] = -2 * x
    status = _monitor(proxy, exact).evaluate()
    assert "broad-probe rank drift" in status["halt_reason"]


@pytest.mark.parametrize("legacy_count", [1, 128])
def test_missing_objective_evidence_cannot_rescue_scalar_rank(legacy_count):
    monitor = _monitor(*_near_ties(1e-8))
    rows = list(monitor.pairs)
    for i in range(legacy_count):
        rows[-1-i] = rows[-1-i][:5]
    monitor.pairs.clear()
    monitor.pairs.extend(rows)
    assert monitor.evaluate()["halt_reason"]


@pytest.mark.parametrize("bad", [None, [], [float("nan")], [float("inf")], [1, 2], [[1]]])
def test_invalid_objective_evidence_is_rejected(bad):
    monitor = _DriftMonitor(GPU_DEFAULTS, objective_count=1)
    with pytest.raises((TypeError, ValueError), match="objectives"):
        monitor.add(1, 1, probe=True, proxy_front=False,
                    proxy_objectives=bad, exact_objectives=[1])


@pytest.mark.parametrize("mismatch_class", ["front", "probe", "all"])
def test_near_ties_cannot_bypass_constraint_gates(mismatch_class):
    proxy, exact = _near_ties(1e-8)
    monitor = _DriftMonitor(GPU_DEFAULTS, objective_count=1)
    for i, (p, e) in enumerate(zip(proxy, exact)):
        probe = i >= 19
        mismatch = mismatch_class == "all" or probe == (mismatch_class == "probe")
        monitor.add(float(p[0]), float(e[0]), probe=probe, proxy_front=not probe,
                    constraint_mismatch=mismatch, proxy_objectives=p, exact_objectives=e)
    assert "constraint agreement" in monitor.evaluate()["halt_reason"]


def test_rank_override_does_not_lower_constraint_gates():
    monitor = _DriftMonitor(dict(GPU_DEFAULTS, drift_rank_halt=0.50))
    for i in range(128):
        monitor.add(i, i, probe=i >= 20, proxy_front=i < 20, constraint_mismatch=i % 10 >= 5)
    assert "constraint agreement" in monitor.evaluate()["halt_reason"]
    assert monitor.constraint_halt == 0.6
    assert monitor.halt == 0.5
    legacy = _DriftMonitor(dict(GPU_DEFAULTS, drift_halt=0.8))
    assert legacy.constraint_halt == legacy.halt == 0.8


def test_rank_override_can_change_rank_without_changing_evidence_budget():
    config = get_template_config()
    config["optimize"]["gpu"].update(drift_probes=1, drift_window=96, drift_rank_halt=0.1)
    config["optimize"]["iters"] = 96
    options = _resolve_options(config)
    assert options["drift_halt"] == 0.6
    assert options["drift_rank_halt"] == 0.1
    assert _monitor(*_near_ties(1), drift_rank_halt=0.5).evaluate()["halt_reason"] is None


@pytest.mark.parametrize("key,value", [
    ("drift_rank_halt", 0), ("drift_rank_halt", 1.1), ("drift_rank_halt", float("nan")),
    ("drift_objective_tolerance", -1), ("drift_objective_tolerance", float("nan")),
    ("drift_objective_tolerance", float("inf")),
])
def test_invalid_signal_options_fail_at_resolution(key, value):
    config = get_template_config()
    config["optimize"]["gpu"][key] = value
    with pytest.raises(ValueError, match=key):
        _resolve_options(config)


def test_signal_options_survive_cli_and_canonical_normalization():
    config = get_template_config()
    assert config["optimize"]["gpu"]["drift_rank_halt"] is None
    assert config["optimize"]["gpu"]["drift_objective_tolerance"] == GPU_DEFAULTS["drift_objective_tolerance"]
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, config)
    args = parser.parse_args([
        "--optimize.gpu.drift_rank_halt", "0.7",
        "--optimize.gpu.drift_objective_tolerance", "0.0000001",
    ])
    update_config_with_args(config, args, verbose=False)
    options = _resolve_options(format_config(config, verbose=False))
    assert options["drift_rank_halt"] == 0.7
    assert options["drift_objective_tolerance"] == 1e-7
    assert options["drift_halt"] == 0.6


def _entries(monitor, seed=False):
    entries = []
    for i, row in enumerate(monitor.pairs):
        metadata = dict(
            schema_version=3, proxy_score=row[0], exact_score=row[1], probe=row[2],
            constraint_classification_mismatch=row[3], proxy_front=row[4],
            proxy_objectives=row[5] if len(row) == 7 else None,
            exact_objectives=row[6] if len(row) == 7 else None,
        )
        metrics = {"gpu_validation": metadata}
        if seed:
            metadata["phase"] = "seed_bootstrap"
            metrics["gpu_seed_bootstrap"] = dict(
                schema_version=1, mode="screened", source_index=i,
                exact_objectives=[row[1]], exact_violation=-1.0,
            )
        entries.append({"id": i, "metrics": metrics})
    return entries


@pytest.mark.parametrize("seed", [False, True])
def test_checkpoint_and_durable_tail_reproduce_live_gate(seed):
    monitor = _monitor(*_near_ties(1e-8))
    entries = _entries(monitor, seed=seed)
    recover = _recover_durable_seed_bootstrap if seed else _recover_durable_validations
    recovered = recover(
        entries, start_index=110, stop_index=128, objective_count=1,
        vector_from_entry=lambda entry: [entry["id"]], hash_vector=lambda vector: str(vector),
    )
    restored = _DriftMonitor(GPU_DEFAULTS, objective_count=1)
    restored.pairs.extend(pickle.loads(pickle.dumps(list(monitor.pairs)[:110])))
    restored.pairs.extend(recovered[-1])
    assert list(restored.pairs) == list(monitor.pairs)
    assert restored.evaluate() == monitor.evaluate()
    assert restored.evaluate()["halt_reason"] is None


@pytest.mark.parametrize("seed", [False, True])
@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "shape"])
def test_recovery_rejects_corrupt_new_objective_evidence(seed, corruption):
    entry = copy.deepcopy(_entries(_monitor(*_near_ties(1e-8)), seed=seed)[0])
    metadata = entry["metrics"]["gpu_validation"]
    if corruption == "missing":
        del metadata["proxy_objectives"]
    elif corruption == "nonfinite":
        metadata["proxy_objectives"] = [float("nan")]
    else:
        metadata["proxy_objectives"] = [1, 2]
    recover = _recover_durable_seed_bootstrap if seed else _recover_durable_validations
    with pytest.raises(RuntimeError, match="per-objective drift evidence"):
        recover([entry], start_index=0, stop_index=1, objective_count=1,
                vector_from_entry=lambda entry: [0], hash_vector=lambda vector: "0")


@pytest.mark.parametrize("seed", [False, True])
def test_supported_infinity_sentinel_keeps_scalar_evidence_and_persists(seed):
    scale = _ObjectiveScale()
    scale.fit(np.array([[0.0], [1.0], [2.0]]))
    monitor = _DriftMonitor(GPU_DEFAULTS, objective_count=1)
    for i in range(128):
        raw = np.array([[float("inf") if i == 40 else float(i)]])
        score = scale.score(raw)[0]
        proxy, exact = _drift_objective_pair(
            scale.normalize(raw)[0], scale.normalize(raw)[0],
            objective_count=1, allow_nonfinite=True,
        )
        if i == 40:
            assert score == 1e6
            assert proxy is exact is None
        monitor.add(score, score, probe=i >= 19, proxy_front=i < 19,
                    proxy_objectives=proxy, exact_objectives=exact)
    assert monitor.evaluate()["halt_reason"] is None
    assert monitor.evaluate()["probe_objective_samples"] == 108
    recover = _recover_durable_seed_bootstrap if seed else _recover_durable_validations
    rows = recover(
        _entries(monitor, seed), start_index=0, stop_index=128, objective_count=1,
        vector_from_entry=lambda entry: [entry["id"]], hash_vector=str,
    )[-1]
    assert rows == list(monitor.pairs)
    assert len(rows[40]) == 5


@pytest.mark.parametrize("seed", [False, True])
def test_recovery_rejects_both_vectors_truncated_to_same_length(seed):
    values = np.tile([1.0, 2.0], (128, 1))
    entry = _entries(_monitor(values, values), seed)[0]
    entry["metrics"]["gpu_validation"].update(proxy_objectives=[1.0], exact_objectives=[1.0])
    recover = _recover_durable_seed_bootstrap if seed else _recover_durable_validations
    with pytest.raises(RuntimeError, match="per-objective drift evidence"):
        recover([entry], start_index=0, stop_index=1, objective_count=2,
                vector_from_entry=lambda entry: [0], hash_vector=str)


def test_checkpoint_rejects_consistently_truncated_vectors():
    values = np.tile([1.0, 2.0], (128, 1))
    original = _monitor(values, values)
    restored = _DriftMonitor(GPU_DEFAULTS, objective_count=2)
    restored.pairs.extend([(*row[:5], row[5][:1], row[6][:1]) for row in original.pairs])
    with pytest.raises(ValueError, match="configured objective count"):
        restored.evaluate()
