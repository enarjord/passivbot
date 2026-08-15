import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from config.schema import get_template_config
from optimization.backends.gpu_backend import (
    _canonical_candidate_values,
    _canonical_vector_hash,
    _build_proxy_parameter_dicts,
    _constraint_classification_mismatch,
    _DriftMonitor,
    _ObjectiveScale,
    _recover_durable_validations,
    _update_novelty_stall,
    _spearman,
    _resolve_options,
    _restore_gpu_result_run_contract,
    _single_scenario_metric_surface,
    _select_novel_validations,
    _select_validation_indices,
    _validate_pinned_scope_bounds,
    _validate_scope,
)


class _Evaluator:
    exchanges = ["bybit"]
    shared_hlcvs_np = {"bybit": np.zeros((100, 1, 4), dtype=np.float64)}


def _long_only_ema_config():
    config = copy.deepcopy(get_template_config())
    config["live"]["strategy_kind"] = "ema_anchor"
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
    config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.0
    config["bot"]["long"]["risk"]["n_positions"] = 1
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0.0
    config["bot"]["short"]["risk"]["n_positions"] = 0
    config["bot"]["long"]["hsl"]["enabled"] = False
    config["bot"]["long"]["unstuck"]["enabled"] = False
    config["bot"]["long"]["risk"]["position_exposure_enforcer_enabled"] = False
    config["bot"]["long"]["risk"]["total_exposure_enforcer_enabled"] = False
    config["bot"]["long"]["risk"]["total_exposure_entry_gate_enabled"] = True
    config["bot"]["long"]["risk"]["we_excess_allowance_pct"] = 0.0
    config["backtest"]["suite_enabled"] = False
    return config


def test_gpu_options_are_additive_and_validate_ranges():
    config = _long_only_ema_config()
    assert _resolve_options(config)["population_size"] == 4096

    config["optimize"]["gpu"]["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["drift_window"] = 16
    config["optimize"]["gpu"]["drift_min_samples"] = 32
    with pytest.raises(ValueError, match="drift_min_samples"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["drift_window"] = 7
    config["optimize"]["gpu"]["drift_min_samples"] = 7
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["validate_per_generation"] = 1
    with pytest.raises(ValueError, match="at least 8"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["drift_window"] = 64
    config["optimize"]["iters"] = 63
    with pytest.raises(ValueError, match="optimize.iters must be at least 64"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 1
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["drift_window"] = 16
    config["optimize"]["gpu"]["drift_min_samples"] = 16
    config["optimize"]["iters"] = 15
    with pytest.raises(ValueError, match="optimize.iters must be at least 16"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["drift_window"] = 63
    with pytest.raises(ValueError, match="at least 64"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 1
    config["optimize"]["gpu"]["drift_probes"] = 4
    config["optimize"]["gpu"]["drift_window"] = 7
    config["optimize"]["gpu"]["drift_min_samples"] = 7
    with pytest.raises(ValueError, match="at least 8"):
        _resolve_options(config)


def test_cpu_backend_registry_import_does_not_import_torch():
    script = (
        "import json, sys; import optimization.backends; "
        "print(json.dumps('torch' in sys.modules))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert json.loads(result.stdout.strip()) is False


def test_gpu_result_preserves_explicit_nulls_and_bounds_for_resume():
    config = _long_only_ema_config()
    config["backtest"]["taker_fee_override"] = None
    config["optimize"]["bounds"]["long"]["strategy"]["ema_anchor"] = {
        "ema_span_0": [10, 20]
    }
    entry = copy.deepcopy(config)
    entry["backtest"]["taker_fee_override"] = 0.00055
    entry["optimize"]["bounds"]["long"]["strategy"]["ema_anchor"] = {
        "ema_span_0": [10, 20],
        "entry": {"double_down_factor": [0, 2]},
    }

    restored = _restore_gpu_result_run_contract(entry, config)

    assert restored["backtest"]["taker_fee_override"] is None
    assert restored["optimize"]["bounds"] == config["optimize"]["bounds"]


def test_single_scenario_metric_surface_supports_all_reducers():
    flattened = _single_scenario_metric_surface({"drawdown": 0.25})

    assert flattened == {
        "drawdown": 0.25,
        "drawdown_mean": 0.25,
        "drawdown_min": 0.25,
        "drawdown_max": 0.25,
        "drawdown_median": 0.25,
        "drawdown_std": 0.0,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["backtest"].__setitem__("suite_enabled", True), "suite"),
        (
            lambda config: config["backtest"].__setitem__(
                "filter_by_min_effective_cost", True
            ),
            "filter_by_min_effective_cost",
        ),
        (
            lambda config: config["live"].__setitem__(
                "market_orders_allowed", True
            ),
            "market_orders_allowed",
        ),
        (
            lambda config: config["live"].__setitem__(
                "strategy_kind", "trailing_martingale"
            ),
            "ema_anchor",
        ),
        (
            lambda config: config["bot"]["long"]["hsl"].__setitem__("enabled", True),
            "hsl",
        ),
        (
            lambda config: config["bot"]["long"]["unstuck"].__setitem__(
                "enabled", True
            ),
            "unstuck",
        ),
        (
            lambda config: config["backtest"].__setitem__(
                "btc_collateral_cap", 0.5
            ),
            "btc_collateral_cap",
        ),
        (
            lambda config: config.__setitem__(
                "coin_overrides", {"BTC": {"bot.long.risk.n_positions": 2}}
            ),
            "coin_overrides",
        ),
        (
            lambda config: config["optimize"]["fixed_runtime_overrides"].__setitem__(
                "bot.long.unstuck.enabled", True
            ),
            "fixed_runtime_overrides",
        ),
        (
            lambda config: config["live"].__setitem__(
                "max_realized_loss_pct", 0.1
            ),
            "max_realized_loss_pct",
        ),
        (
            lambda config: config["bot"]["long"]["risk"].__setitem__(
                "position_exposure_enforcer_enabled", True
            ),
            "position_exposure_enforcer_enabled",
        ),
        (
            lambda config: config["bot"]["long"]["risk"].__setitem__(
                "total_exposure_enforcer_enabled", True
            ),
            "total_exposure_enforcer_enabled",
        ),
        (
            lambda config: config["bot"]["long"]["risk"].__setitem__(
                "we_excess_allowance_pct", 0.1
            ),
            "we_excess_allowance_pct",
        ),
        (
            lambda config: config["bot"]["long"]["risk"].__setitem__(
                "total_exposure_entry_gate_enabled", False
            ),
            "total_exposure_entry_gate_enabled",
        ),
    ],
)
def test_gpu_foundation_fails_closed_for_unsupported_scope(mutate, message):
    config = _long_only_ema_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        _validate_scope(config, _Evaluator())


def test_gpu_foundation_accepts_ema_long_single():
    assert _validate_scope(_long_only_ema_config(), _Evaluator()) == "bybit"


def test_gpu_foundation_honors_approved_coins_when_disabled_side_risk_is_nonzero():
    config = _long_only_ema_config()
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 2.5
    config["bot"]["short"]["risk"]["n_positions"] = 1

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_validation_selection_includes_front_and_broad_probes():
    objectives = np.array(
        [
            [0.0, 5.0],
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [9.0, 9.0],
        ]
    )
    scores = objectives.mean(axis=1)

    selected = _select_validation_indices(objectives, scores, total=5, probes=1)

    chosen = selected[:5]
    assert len(chosen) == 5
    assert sum(is_probe for _index, is_probe in chosen) == 1
    assert all(index == 6 for index, is_probe in chosen if is_probe)
    assert len({index for index, _is_probe in selected}) == len(objectives)


def test_validation_selection_fails_without_requested_off_front_evidence():
    objectives = np.array(
        [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    )
    scores = objectives.mean(axis=1)

    with pytest.raises(RuntimeError, match="independent broad-probe evidence"):
        _select_validation_indices(objectives, scores, total=3, probes=1)


def test_validation_selection_prefers_feasible_candidates():
    objectives = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    )
    scores = objectives.mean(axis=1)
    violations = np.array([2.0, 0.0, -1.0, 3.0, 0.0])

    selected = _select_validation_indices(
        objectives, scores, violations, total=3, probes=1
    )

    assert {index for index, _probe in selected[:3]} == {1, 2, 4}


def test_validation_broad_probes_exclude_the_entire_proxy_front():
    objectives = np.array(
        [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [8.0, 8.0],
            [9.0, 9.0],
        ]
    )
    scores = objectives.mean(axis=1)

    selected = _select_validation_indices(objectives, scores, total=5, probes=2)

    assert {index for index, is_probe in selected[:5] if is_probe} == {5, 6}


def test_duplicate_broad_probe_is_replaced_by_novel_off_front_candidate():
    selections = [(0, False), (1, True), (2, False), (3, True)]

    chosen = _select_novel_validations(
        selections,
        total=2,
        probes=1,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes={"hash-1"},
        submitted_hashes=set(),
    )

    assert len(chosen) == 2
    assert sum(is_probe for _index, is_probe, _candidate, _digest in chosen) == 1
    assert chosen[0][0] == 3


def test_duplicate_broad_probes_fail_closed_when_no_novel_replacement_exists():
    with pytest.raises(RuntimeError, match="replace duplicate broad probes"):
        _select_novel_validations(
            [(0, False), (1, True)],
            total=2,
            probes=1,
            candidate_for_index=lambda index: [index],
            digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
            completed_hashes={"hash-1"},
            submitted_hashes=set(),
        )


def test_drift_monitor_needs_broad_probe_evidence_before_halting():
    options = {
        "drift_window": 64,
        "drift_min_samples": 16,
        "drift_halt": 0.6,
    }
    monitor = _DriftMonitor(options)
    for index in range(16):
        monitor.add(index, -index, probe=index < 4)

    first = monitor.evaluate()
    assert first["halt_reason"] is None
    assert first["warn_reason"]

    for index in range(16, 32):
        monitor.add(index, -index, probe=index < 24)
    second = monitor.evaluate()
    assert second["probe_rho"] < 0.0
    assert second["halt_reason"]


def test_drift_monitor_halts_when_proxy_cannot_rank_broad_probes():
    monitor = _DriftMonitor(
        {
            "drift_window": 64,
            "drift_min_samples": 16,
            "drift_halt": 0.6,
        }
    )
    for index in range(16):
        monitor.add(1.0, float(index), probe=index < 8)

    status = monitor.evaluate()

    assert np.isnan(status["rho"])
    assert np.isnan(status["probe_rho"])
    assert status["halt_reason"]


def test_drift_monitor_probe_failure_cannot_be_masked_by_high_aggregate_rho():
    monitor = _DriftMonitor(
        {
            "drift_window": 128,
            "drift_min_samples": 32,
            "drift_halt": 0.6,
        }
    )
    for index in range(56):
        monitor.add(index, index, probe=False)
    for index in range(56, 64):
        monitor.add(index, 119 - index, probe=True)

    status = monitor.evaluate()

    assert status["rho"] > 0.6
    assert status["probe_rho"] == pytest.approx(-1.0)
    assert status["halt_reason"]


def test_novelty_stall_terminates_and_resets_on_progress():
    stall = 0
    for _ in range(7):
        stall = _update_novelty_stall(stall, submitted=0, pending=0)
    with pytest.raises(RuntimeError, match="search space appears exhausted"):
        _update_novelty_stall(stall, submitted=0, pending=0)

    assert _update_novelty_stall(stall, submitted=1, pending=0) == 0
    assert _update_novelty_stall(stall, submitted=0, pending=1) == 0


def test_spearman_uses_average_ranks_for_ties():
    left = np.array([1.0, 1.0, 2.0, 3.0])
    right = np.array([4.0, 4.0, 3.0, 2.0])

    assert _spearman(left, right) == pytest.approx(-1.0)


def test_objective_scale_scores_proxy_and_exact_in_same_coordinates():
    scale = _ObjectiveScale()
    proxy = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    scale.fit(proxy)

    scores = scale.score(np.array([[1.0, 100.0], [3.0, 300.0]]))

    assert scores[0] < scores[1]


def test_gpu_candidates_use_exact_significant_digit_quantization():
    from optimization.bounds import Bound

    values = _canonical_candidate_values(
        np.array([[0.123456789, 0.61]]),
        np.array([0.0, 0.0]),
        np.array([1.0, 10.0]),
        [Bound(0.0, 1.0, None), Bound(0.0, 10.0, 0.5)],
        3,
    )

    assert values.tolist() == [[0.123, 6.0]]


def test_gpu_candidate_hash_uses_exact_significant_digit_quantization():
    from optimization.bounds import Bound

    bounds = [Bound(0.0, 1.0, None)]

    assert _canonical_vector_hash([0.1234], bounds, 3) == _canonical_vector_hash(
        [0.12349], bounds, 3
    )


def test_proxy_parameters_include_canonical_pinned_ema_values():
    from optimization.bounds import Bound

    mapped = {
        "base_qty_pct": (0, Bound(0.25, 0.25, None)),
        "offset": (1, Bound(0.0, 1.0, None)),
    }
    active = [("offset", 1, mapped["offset"][1])]

    parameters = _build_proxy_parameter_dicts(
        [0.25, 0.5], mapped, active, np.array([[0.75]])
    )

    assert parameters == [{"base_qty_pct": 0.25, "offset": 0.75}]


def test_gpu_rejects_pinned_unsupported_risk_behavior():
    from optimization.bounds import Bound

    with pytest.raises(ValueError, match="we_excess_allowance_pct"):
        _validate_pinned_scope_bounds(
            {"long_risk_we_excess_allowance_pct": Bound(0.2, 0.2, None)},
            {"long_risk_we_excess_allowance_pct": 0.2},
        )

    with pytest.raises(ValueError, match="total_exposure_enforcer_threshold"):
        _validate_pinned_scope_bounds(
            {"long_risk_total_exposure_enforcer_threshold": Bound(0.8, 0.8, None)},
            {"long_risk_total_exposure_enforcer_threshold": 0.8},
        )


def test_constraint_classification_drift_detects_feasibility_disagreement():
    assert _constraint_classification_mismatch(0.0, {"G": np.array([0.1])})
    assert _constraint_classification_mismatch(0.1, {"G": np.array([-1.0])})
    assert not _constraint_classification_mismatch(0.0, {"G": np.array([-1.0])})
    assert not _constraint_classification_mismatch(0.1, {"G": np.array([0.1])})
    assert not _constraint_classification_mismatch(0.1, {})


def test_resume_recovers_hashes_and_drift_for_results_ahead_of_checkpoint():
    entries = [
        {
            "id": index,
            "metrics": {
                "gpu_validation": {
                    "schema_version": 1,
                    "proxy_score": float(index),
                    "exact_score": float(index) + 0.5,
                    "probe": index == 3,
                    "constraint_classification_mismatch": False,
                }
            },
        }
        for index in range(4)
    ]

    recovered, drift_pairs, mismatch = _recover_durable_validations(
        entries,
        start_index=2,
        stop_index=4,
        vector_from_entry=lambda entry: [float(entry["id"])],
        hash_vector=lambda vector: f"hash-{vector[0]}",
    )

    assert recovered == {"hash-2.0", "hash-3.0"}
    assert drift_pairs == [(2.0, 2.5, False), (3.0, 3.5, True)]
    assert mismatch is None


def test_resume_hash_recovery_fails_if_durable_tail_is_missing():
    with pytest.raises(RuntimeError, match="expected 2, recovered 1"):
        _recover_durable_validations(
            [
                {
                    "id": index,
                    "metrics": {
                        "gpu_validation": {
                            "schema_version": 1,
                            "proxy_score": float(index),
                            "exact_score": float(index),
                            "probe": False,
                            "constraint_classification_mismatch": False,
                        }
                    },
                }
                for index in range(3)
            ],
            start_index=2,
            stop_index=4,
            vector_from_entry=lambda entry: [float(entry["id"])],
            hash_vector=lambda vector: f"hash-{vector[0]}",
        )


def test_resume_fails_closed_when_durable_tail_lacks_drift_evidence():
    with pytest.raises(RuntimeError, match="cannot recover proxy/exact safety evidence"):
        _recover_durable_validations(
            [{"id": 0}],
            start_index=0,
            stop_index=1,
            vector_from_entry=lambda entry: [float(entry["id"])],
            hash_vector=lambda vector: f"hash-{vector[0]}",
        )


def test_resume_recovers_durable_constraint_disagreement():
    _hashes, pairs, mismatch = _recover_durable_validations(
        [
            {
                "id": 0,
                "metrics": {
                    "gpu_validation": {
                        "schema_version": 1,
                        "proxy_score": 0.1,
                        "exact_score": 0.2,
                        "probe": True,
                        "constraint_classification_mismatch": True,
                    }
                },
            }
        ],
        start_index=0,
        stop_index=1,
        vector_from_entry=lambda entry: [float(entry["id"])],
        hash_vector=lambda vector: f"hash-{vector[0]}",
    )

    assert pairs == [(0.1, 0.2, True)]
    assert "constraint classification disagreed" in mismatch
