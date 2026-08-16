import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from config.schema import get_template_config
from optimization.bounds import Bound
from optimization.backends.gpu_backend import (
    _canonical_candidate_values,
    _canonical_vector_hash,
    _build_gpu_nsga2,
    _build_proxy_parameter_dicts,
    _constraint_classification_mismatch,
    _constraint_diagnostics,
    _format_constraint_diagnostics,
    _DriftMonitor,
    _ObjectiveScale,
    _recover_durable_validations,
    _ready_submission_prefix,
    _update_novelty_stall,
    _validation_probe_count,
    _spearman,
    _resolve_options,
    _restore_gpu_result_run_contract,
    _single_scenario_metric_surface,
    _select_novel_validations,
    _select_validation_indices,
    _validate_directional_search_space,
    _validate_pinned_scope_bounds,
    _validate_resume_evidence_budget,
    _validate_seed_side_match,
    _validate_scope,
    TRAILING_MARTINGALE_BOUND_MAP,
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


def _directional_ema_config(*, long_enabled: bool, short_enabled: bool):
    config = _long_only_ema_config()
    config["live"]["approved_coins"] = {
        "long": ["BTC"] if long_enabled else [],
        "short": ["BTC"] if short_enabled else [],
    }
    for side, enabled in (("long", long_enabled), ("short", short_enabled)):
        config["bot"][side]["risk"]["total_wallet_exposure_limit"] = (
            1.0 if enabled else 0.0
        )
        config["bot"][side]["risk"]["n_positions"] = 1 if enabled else 0
        config["bot"][side]["hsl"]["enabled"] = False
        config["bot"][side]["unstuck"]["enabled"] = False
        config["bot"][side]["risk"]["position_exposure_enforcer_enabled"] = False
        config["bot"][side]["risk"]["total_exposure_enforcer_enabled"] = False
        config["bot"][side]["risk"]["total_exposure_entry_gate_enabled"] = True
        config["bot"][side]["risk"]["we_excess_allowance_pct"] = 0.0
    return config


def _directional_tm_config(*, long_enabled: bool, short_enabled: bool):
    config = _directional_ema_config(
        long_enabled=long_enabled, short_enabled=short_enabled
    )
    config["live"]["strategy_kind"] = "trailing_martingale"
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
    config["optimize"]["gpu"]["validate_per_generation"] = 2
    with pytest.raises(ValueError, match="at least 16"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["drift_window"] = 96
    config["optimize"]["iters"] = 95
    with pytest.raises(ValueError, match="optimize.iters must be at least 96"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 2
    config["optimize"]["gpu"]["drift_probes"] = 0
    config["optimize"]["gpu"]["drift_window"] = 16
    config["optimize"]["gpu"]["drift_min_samples"] = 16
    config["optimize"]["iters"] = 15
    with pytest.raises(ValueError, match="optimize.iters must be at least 16"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 7
    config["optimize"]["gpu"]["drift_window"] = 32
    config["optimize"]["gpu"]["drift_min_samples"] = 32
    config["optimize"]["iters"] = 32
    with pytest.raises(ValueError, match="retain 8 true proxy-front validations"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 8
    with pytest.raises(ValueError, match="must be less than"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["max_pending_exact"] = 1
    with pytest.raises(ValueError, match="at least optimize.gpu"):
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
    with pytest.raises(ValueError, match="must be less than"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["drift_halt"] = 0.0
    with pytest.raises(ValueError, match="greater than zero"):
        _resolve_options(config)


def test_fresh_run_rejects_partial_suffix_without_rank_probe_budget():
    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["drift_window"] = 96
    config["optimize"]["iters"] = 97

    with pytest.raises(ValueError, match="GPU fresh run.*broad-probe"):
        _resolve_options(config)


def test_partial_validation_batch_preserves_front_evidence_ratio():
    assert _validation_probe_count(10, 10, 7) == 7
    assert _validation_probe_count(7, 10, 7) == 4
    assert _validation_probe_count(1, 10, 7) == 0


class _PendingResult:
    def __init__(self, ready):
        self._ready = ready

    def ready(self):
        return self._ready


def test_exact_results_are_consumed_only_as_ready_submission_prefix():
    first = _PendingResult(False)
    second = _PendingResult(True)
    assert _ready_submission_prefix({first: None, second: None}) == []

    first._ready = True
    second._ready = False
    third = _PendingResult(True)
    assert _ready_submission_prefix({first: None, second: None, third: None}) == [
        first
    ]


def _drift_pair(*, front: bool):
    return (0.0, 0.0, not front, False, front)


def test_resume_budget_rejects_too_few_remaining_front_samples():
    pairs = [_drift_pair(front=index < 6) for index in range(57)]
    options = {
        "drift_window": 128,
        "validate_per_generation": 8,
        "drift_probes": 4,
    }

    with pytest.raises(RuntimeError, match="proxy-front safety samples"):
        _validate_resume_evidence_budget(
            pairs,
            exact_done=57,
            exact_budget=64,
            options=options,
        )


def test_resume_budget_accepts_sufficient_recovered_and_future_evidence():
    pairs = [_drift_pair(front=index < 7) for index in range(57)]

    _validate_resume_evidence_budget(
        pairs,
        exact_done=57,
        exact_budget=64,
        options={
            "drift_window": 128,
            "validate_per_generation": 8,
            "drift_probes": 4,
        },
    )


def test_resume_budget_rejects_too_few_remaining_broad_probes():
    pairs = [_drift_pair(front=index >= 3) for index in range(57)]

    with pytest.raises(RuntimeError, match="broad-probe safety samples"):
        _validate_resume_evidence_budget(
            pairs,
            exact_done=57,
            exact_budget=64,
            options={
                "drift_window": 128,
                "validate_per_generation": 8,
                "drift_probes": 4,
            },
        )


def test_gpu_nsga2_uses_configured_pymoo_variation_operators():
    config = _long_only_ema_config()
    config["optimize"]["pymoo"]["shared"] = {
        "crossover_eta": 11.0,
        "crossover_prob_var": 0.7,
        "mutation_eta": 13.0,
        "mutation_prob_var": 0.2,
        "eliminate_duplicates": False,
    }
    algorithm = _build_gpu_nsga2(
        config,
        sampling=np.zeros((8, 5), dtype=np.float64),
        population_size=8,
        n_params=5,
    )

    assert algorithm.mating.crossover.eta.value == 11.0
    assert algorithm.mating.crossover.prob_var.value == 0.7
    assert algorithm.mating.mutation.eta.value == 13.0
    assert algorithm.mating.mutation.prob.value == 0.2
    assert type(algorithm.eliminate_duplicates).__name__ == "NoDuplicateElimination"


def test_trailing_martingale_bound_map_covers_both_directional_shapes():
    expected_suffixes = {
        "ema_span_0",
        "ema_span_1",
        "volatility_ema_span_1h",
        "volatility_ema_span_1m",
        "entry_double_down_factor",
        "entry_initial_ema_dist",
        "entry_initial_qty_pct",
        "entry_threshold_base_pct",
        "entry_threshold_we_weight",
        "entry_threshold_volatility_1h_weight",
        "entry_threshold_volatility_1m_weight",
        "entry_retracement_base_pct",
        "entry_retracement_we_weight",
        "entry_retracement_volatility_1h_weight",
        "entry_retracement_volatility_1m_weight",
        "close_qty_pct",
        "close_threshold_base_pct",
        "close_threshold_we_weight",
        "close_threshold_volatility_1h_weight",
        "close_threshold_volatility_1m_weight",
        "close_retracement_base_pct",
        "close_retracement_volatility_1h_weight",
        "close_retracement_volatility_1m_weight",
        "risk_entry_cooldown_minutes",
        "total_wallet_exposure_limit",
    }

    assert set(TRAILING_MARTINGALE_BOUND_MAP) == {
        f"{side}_{suffix}"
        for side in ("long", "short")
        for suffix in expected_suffixes
    }


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
                "strategy_kind", "trailing_grid_v7"
            ),
            "trailing_martingale",
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


@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
def test_gpu_foundation_accepts_each_directional_ema_mode(
    long_enabled, short_enabled
):
    config = _directional_ema_config(
        long_enabled=long_enabled, short_enabled=short_enabled
    )

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
def test_gpu_foundation_accepts_each_directional_trailing_martingale_mode(
    long_enabled, short_enabled
):
    config = _directional_tm_config(
        long_enabled=long_enabled, short_enabled=short_enabled
    )

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_foundation_accepts_recursive_trailing_martingale_bounds():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    for side in ("long", "short"):
        for mode in ("entry", "close"):
            config["optimize"]["bounds"][side]["strategy"][
                "trailing_martingale"
            ][mode]["retracement_base_pct"] = [-0.01, 0.01, 0.0001]

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_foundation_checks_unsupported_behavior_on_short_side():
    config = _directional_ema_config(long_enabled=False, short_enabled=True)
    config["bot"]["short"]["unstuck"]["enabled"] = True

    with pytest.raises(ValueError, match=r"bot\.short\.unstuck"):
        _validate_scope(config, _Evaluator())


def test_gpu_foundation_rejects_both_sides_disabled():
    config = _directional_ema_config(long_enabled=False, short_enabled=False)

    with pytest.raises(ValueError, match="at least one enabled side"):
        _validate_scope(config, _Evaluator())


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
    assert sum(is_probe for _index, is_probe, _front in chosen) == 1
    assert all(index == 6 for index, is_probe, _front in chosen if is_probe)
    assert len({index for index, _is_probe, _front in selected}) == len(objectives)


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

    assert {index for index, _probe, _front in selected[:3]} == {1, 2, 4}


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

    assert {
        index for index, is_probe, _front in selected[:5] if is_probe
    } == {5, 6}


def test_duplicate_broad_probe_is_replaced_by_novel_off_front_candidate():
    selections = [
        (0, False, True),
        (1, True, False),
        (2, False, True),
        (3, True, False),
    ]

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
    assert sum(
        is_probe
        for _index, is_probe, _front, _candidate, _digest in chosen
    ) == 1
    assert chosen[0][0] == 3


def test_duplicate_broad_probes_fail_closed_when_no_novel_replacement_exists():
    with pytest.raises(RuntimeError, match="replace duplicate broad probes"):
        _select_novel_validations(
            [(0, False, True), (1, True, False)],
            total=2,
            probes=1,
            candidate_for_index=lambda index: [index],
            digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
            completed_hashes={"hash-1"},
            submitted_hashes=set(),
        )


def test_validation_batch_preserves_true_front_and_off_front_classification():
    objectives = np.array(
        [[float(index), float(index)] for index in range(12)], dtype=np.float64
    )
    scores = objectives.mean(axis=1)
    selections = _select_validation_indices(
        objectives, scores, total=8, probes=4
    )

    # The complete feasible Pareto front has one member. The remaining seven
    # candidates must stay truthfully classified as broad/off-front evidence.
    chosen = _select_novel_validations(
        selections,
        total=8,
        probes=4,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes=set(),
        submitted_hashes=set(),
    )

    assert len(chosen) == 8
    assert sum(
        is_probe
        for _index, is_probe, _front, _candidate, _digest in chosen
    ) == 7


def test_all_infeasible_validation_fallback_keeps_front_membership_explicit():
    objectives = np.array(
        [[float(index), float(index)] for index in range(6)], dtype=np.float64
    )
    scores = objectives.mean(axis=1)
    violations = np.arange(1.0, 7.0)

    selected = _select_validation_indices(
        objectives, scores, violations, total=4, probes=1
    )

    assert selected[0] == (0, False, True)
    assert all(is_probe != is_front for _index, is_probe, is_front in selected)
    assert all(
        is_probe and not is_front
        for index, is_probe, is_front in selected
        if index != 0
    )


def test_validation_fails_closed_without_novel_proxy_front_evidence():
    with pytest.raises(RuntimeError, match="novel proxy-front safety evidence"):
        _select_novel_validations(
            [(0, False, True), (1, True, False), (2, True, False)],
            total=2,
            probes=1,
            candidate_for_index=lambda index: [index],
            digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
            completed_hashes={"hash-0"},
            submitted_hashes=set(),
        )


def test_validation_scans_fallbacks_for_novel_proxy_front_before_failing():
    chosen = _select_novel_validations(
        [
            (0, False, True),
            (1, True, False),
            (2, True, False),
            (3, False, True),
        ],
        total=2,
        probes=1,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes={"hash-0"},
        submitted_hashes=set(),
    )

    assert {index for index, _probe, _front, _candidate, _digest in chosen} == {
        1,
        3,
    }


def test_true_front_mismatches_halt_even_when_off_front_agreement_is_high():
    monitor = _DriftMonitor(
        {
            "drift_window": 64,
            "drift_min_samples": 32,
            "drift_halt": 0.6,
        }
    )
    for generation in range(8):
        monitor.add(
            generation,
            generation,
            probe=False,
            proxy_front=True,
            constraint_mismatch=True,
        )
        for probe in range(7):
            score = generation * 7 + probe
            monitor.add(
                score,
                score,
                probe=True,
                proxy_front=False,
                constraint_mismatch=False,
            )

    status = monitor.evaluate()

    assert status["samples"] == 64
    assert status["front_samples"] == 8
    assert status["probes"] == 56
    assert status["constraint_agreement"] == pytest.approx(0.875)
    assert status["front_constraint_agreement"] == 0.0
    assert "proxy-front constraint agreement" in status["halt_reason"]


def test_drift_monitor_needs_broad_probe_evidence_before_halting():
    options = {
        "drift_window": 64,
        "drift_min_samples": 16,
        "drift_halt": 0.6,
    }
    monitor = _DriftMonitor(options)
    for index in range(16):
        probe = index < 4
        monitor.add(index, -index, probe=probe, proxy_front=not probe)

    first = monitor.evaluate()
    assert first["halt_reason"] is None
    assert first["warn_reason"]

    for index in range(16, 32):
        probe = index < 24
        monitor.add(index, -index, probe=probe, proxy_front=not probe)
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
        probe = index < 8
        monitor.add(1.0, float(index), probe=probe, proxy_front=not probe)

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
        monitor.add(index, index, probe=False, proxy_front=True)
    for index in range(56, 64):
        monitor.add(index, 119 - index, probe=True, proxy_front=False)

    status = monitor.evaluate()

    assert status["rho"] > 0.6
    assert status["probe_rho"] == pytest.approx(-1.0)
    assert status["halt_reason"]


def test_drift_monitor_allows_isolated_broad_probe_constraint_mismatches():
    monitor = _DriftMonitor(
        {
            "drift_window": 64,
            "drift_min_samples": 16,
            "drift_halt": 0.6,
        }
    )
    for index in range(16):
        probe = index < 8
        monitor.add(
            index,
            1000 - index if probe and index < 3 else index,
            probe=probe,
            proxy_front=not probe,
            constraint_mismatch=probe and index < 3,
        )

    status = monitor.evaluate()

    assert status["probe_constraint_agreement"] == pytest.approx(0.625)
    assert status["probe_constraint_mismatches"] == 3
    assert status["probe_rank_samples"] == 5
    assert status["rho"] == pytest.approx(1.0)
    assert status["halt_reason"] is None


def test_drift_monitor_ranks_only_classification_agreeing_broad_probes():
    monitor = _DriftMonitor(
        {
            "drift_window": 128,
            "drift_min_samples": 32,
            "drift_halt": 0.6,
        }
    )
    for index in range(33):
        probe = index % 2 == 0
        mismatch = probe and index in (16, 32)
        monitor.add(
            float(index),
            float(1000 - index if mismatch else index),
            probe=probe,
            proxy_front=not probe,
            constraint_mismatch=mismatch,
        )

    status = monitor.evaluate()

    assert status["probes"] == 17
    assert status["probe_rank_samples"] == 15
    assert status["probe_rho"] == pytest.approx(1.0)
    assert status["probe_constraint_agreement"] == pytest.approx(15 / 17)
    assert status["halt_reason"] is None


def test_drift_monitor_allows_isolated_proxy_front_constraint_mismatches():
    monitor = _DriftMonitor(
        {
            "drift_window": 64,
            "drift_min_samples": 16,
            "drift_halt": 0.6,
        }
    )
    for index in range(16):
        probe = index < 8
        monitor.add(
            index,
            index,
            probe=probe,
            proxy_front=not probe,
            constraint_mismatch=not probe and index < 11,
        )

    status = monitor.evaluate()

    assert status["constraint_agreement"] == pytest.approx(0.8125)
    assert status["front_constraint_agreement"] == pytest.approx(0.625)
    assert status["front_constraint_mismatches"] == 3
    assert status["halt_reason"] is None


def test_drift_monitor_halts_on_low_proxy_front_constraint_agreement():
    monitor = _DriftMonitor(
        {
            "drift_window": 64,
            "drift_min_samples": 16,
            "drift_halt": 0.6,
        }
    )
    for index in range(16):
        probe = index < 8
        monitor.add(
            index,
            index,
            probe=probe,
            proxy_front=not probe,
            constraint_mismatch=not probe and index < 12,
        )

    status = monitor.evaluate()

    assert status["constraint_agreement"] == pytest.approx(0.75)
    assert status["front_constraint_agreement"] == pytest.approx(0.5)
    assert "proxy-front constraint agreement fell below" in status["halt_reason"]


def test_drift_monitor_halts_on_low_broad_probe_constraint_agreement():
    monitor = _DriftMonitor(
        {
            "drift_window": 64,
            "drift_min_samples": 16,
            "drift_halt": 0.6,
        }
    )
    for index in range(16):
        probe = index < 8
        monitor.add(
            index,
            index,
            probe=probe,
            proxy_front=not probe,
            constraint_mismatch=probe and index < 4,
        )

    status = monitor.evaluate()

    assert status["probe_constraint_agreement"] == pytest.approx(0.5)
    assert "constraint agreement fell below" in status["halt_reason"]


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


def test_proxy_parameters_keep_directional_names_distinct():
    from optimization.bounds import Bound

    mapped = {
        "long_offset": (0, Bound(0.0, 1.0, None)),
        "short_offset": (1, Bound(0.0, 1.0, None)),
    }
    active = [
        ("long_offset", 0, mapped["long_offset"][1]),
        ("short_offset", 1, mapped["short_offset"][1]),
    ]

    parameters = _build_proxy_parameter_dicts(
        [0.25, 0.5], mapped, active, np.array([[0.75, 0.125]])
    )

    assert parameters == [{"long_offset": 0.75, "short_offset": 0.125}]


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


def test_gpu_directional_search_space_keeps_side_enablement_fixed():
    from optimization.bounds import Bound

    approved = {"long": ["BTC"], "short": []}
    base = {
        "long_total_wallet_exposure_limit": 1.0,
        "long_n_positions": 1.0,
        "short_total_wallet_exposure_limit": 0.0,
        "short_n_positions": 0.0,
    }
    bounds = {
        "long_total_wallet_exposure_limit": Bound(0.5, 1.5, None),
        "long_n_positions": Bound(1.0, 1.0, None),
        "short_total_wallet_exposure_limit": Bound(0.0, 2.0, None),
        "short_n_positions": Bound(0.0, 1.0, None),
    }

    _validate_directional_search_space(bounds, base, approved, {"long"})

    bounds["long_total_wallet_exposure_limit"] = Bound(0.0, 1.5, None)
    with pytest.raises(ValueError, match="remain positive"):
        _validate_directional_search_space(bounds, base, approved, {"long"})

    bounds["long_total_wallet_exposure_limit"] = Bound(0.5, 1.5, None)
    bounds["long_n_positions"] = Bound(1.0, 2.0, None)
    with pytest.raises(ValueError, match="pinned at 1"):
        _validate_directional_search_space(bounds, base, approved, {"long"})


def test_gpu_directional_search_space_rejects_disabled_approved_side_activation():
    from optimization.bounds import Bound

    bounds = {
        "long_total_wallet_exposure_limit": Bound(0.5, 1.5, None),
        "long_n_positions": Bound(1.0, 1.0, None),
        "short_total_wallet_exposure_limit": Bound(0.0, 1.5, None),
        "short_n_positions": Bound(0.0, 1.0, None),
    }

    with pytest.raises(ValueError, match="short enabledness"):
        _validate_directional_search_space(
            bounds,
            {},
            {"long": ["BTC"], "short": ["BTC"]},
            {"long"},
        )


def test_gpu_rejects_optimizer_bounds_that_change_config_side_enablement():
    _validate_seed_side_match({"long"}, {"long"})

    with pytest.raises(ValueError, match="activate or disable"):
        _validate_seed_side_match({"long"}, {"long", "short"})

    with pytest.raises(ValueError, match="activate or disable"):
        _validate_seed_side_match({"long", "short"}, {"long"})


def test_constraint_classification_drift_detects_feasibility_disagreement():
    assert _constraint_classification_mismatch(0.0, {"G": np.array([0.1])})
    assert _constraint_classification_mismatch(0.1, {"G": np.array([-1.0])})
    assert not _constraint_classification_mismatch(0.0, {"G": np.array([-1.0])})
    assert not _constraint_classification_mismatch(0.1, {"G": np.array([0.1])})
    assert not _constraint_classification_mismatch(0.1, {})


def test_constraint_diagnostics_name_disagreeing_limit_values():
    check = {
        "metric": "position_held_days_max",
        "metric_key": "position_held_days_max_max",
        "mode": "greater_than",
        "bound": 60.0,
        "penalty_weight": 1.0e6,
    }
    evaluator = type("Evaluator", (), {"limit_checks": [check]})()
    exact_payload = {
        "metrics": {
            "stats": {
                "position_held_days_max": {
                    "mean": 195.0,
                    "min": 195.0,
                    "max": 195.0,
                    "std": 0.0,
                    "median": 195.0,
                }
            }
        }
    }

    diagnostics = _constraint_diagnostics(
        evaluator,
        {"position_held_days_max": 24.0},
        exact_payload,
    )

    assert diagnostics == [
        {
            "metric": "position_held_days_max",
            "metric_key": "position_held_days_max_max",
            "mode": "greater_than",
            "proxy_value": 24.0,
            "exact_value": 195.0,
            "proxy_violation": 0.0,
            "exact_violation": 135_000_000.0,
            "bound": 60.0,
        }
    ]
    detail = _format_constraint_diagnostics(diagnostics)
    assert "position_held_days_max_max" in detail
    assert "proxy=24.0 exact=195.0" in detail
    assert "bound=60.0" in detail


def test_resume_recovers_hashes_and_drift_for_results_ahead_of_checkpoint():
    entries = [
        {
            "id": index,
            "metrics": {
                "gpu_validation": {
                    "schema_version": 2,
                    "proxy_score": float(index),
                    "exact_score": float(index) + 0.5,
                    "probe": index == 3,
                    "proxy_front": index != 3,
                    "constraint_classification_mismatch": False,
                }
            },
        }
        for index in range(4)
    ]

    recovered, drift_pairs = _recover_durable_validations(
        entries,
        start_index=2,
        stop_index=4,
        vector_from_entry=lambda entry: [float(entry["id"])],
        hash_vector=lambda vector: f"hash-{vector[0]}",
    )

    assert recovered == {"hash-2.0", "hash-3.0"}
    assert drift_pairs == [
        (2.0, 2.5, False, False, True),
        (3.0, 3.5, True, False, False),
    ]


def test_resume_hash_recovery_fails_if_durable_tail_is_missing():
    with pytest.raises(RuntimeError, match="expected 2, recovered 1"):
        _recover_durable_validations(
            [
                {
                    "id": index,
                    "metrics": {
                        "gpu_validation": {
                            "schema_version": 2,
                            "proxy_score": float(index),
                            "exact_score": float(index),
                            "probe": False,
                            "proxy_front": True,
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


def test_resume_recovers_durable_front_constraint_disagreement_as_drift_evidence():
    _hashes, pairs = _recover_durable_validations(
        [
            {
                "id": 0,
                "metrics": {
                    "gpu_validation": {
                        "schema_version": 2,
                        "proxy_score": 0.1,
                        "exact_score": 0.2,
                        "probe": False,
                        "proxy_front": True,
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

    assert pairs == [(0.1, 0.2, False, True, True)]


def test_resume_records_broad_probe_constraint_disagreement_without_immediate_halt():
    _hashes, pairs = _recover_durable_validations(
        [
            {
                "id": 0,
                "metrics": {
                    "gpu_validation": {
                        "schema_version": 2,
                        "proxy_score": 0.1,
                        "exact_score": 0.2,
                        "probe": True,
                        "proxy_front": False,
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

    assert pairs == [(0.1, 0.2, True, True, False)]
