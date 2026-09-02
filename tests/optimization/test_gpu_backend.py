import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.metrics import canonicalize_metric_name
from config.overrides import OVERRIDABLE_SHARED_BOT_PATHS
from config.schema import get_template_config
from config.strategy import get_strategy_param_keys
from optimizer_overrides import optimizer_overrides
from optimization.bounds import Bound
from optimization.backends.gpu_backend import (
    _apply_gpu_optimizer_overrides,
    _ask_gpu_population,
    _canonical_candidate_values,
    _canonicalize_mirrored_hash_vector,
    _canonicalize_optimizer_override_hash_vector,
    _canonical_vector_hash,
    _build_anchor_parameter_context,
    _build_gpu_nsga2,
    _build_proxy_parameter_dicts,
    _checkpoint_signature,
    _checkpoint_gpu_interrupt,
    _constraint_classification_mismatch,
    _constraint_diagnostics,
    _deduplicate_canonical_seed_vectors,
    _disable_gpu_exact_duplicate_guard,
    _ema_multicoin_bound_map,
    _evaluate_gpu_suite_proxies,
    _evaluate_successive_halving,
    _effective_seed_bootstrap_mode,
    _format_constraint_diagnostics,
    _gpu_fixed_bound_context,
    _gpu_candidate_search_sides,
    _gpu_candidate_source_sides,
    _gpu_hsl_search_sides,
    _gpu_hsl_parameter_active,
    _gpu_lean_tm_parallelism_eligible,
    _gpu_nsga2_checkpoint_contract,
    _gpu_pinned_hsl_bound_contract,
    _gpu_profile_elapsed,
    _validate_hsl_bound_contracts,
    _validate_hsl_metric_topology,
    _gpu_suite_enabled,
    _gpu_suite_checkpoint_contract,
    _gpu_runtime_checkpoint_contract,
    _gpu_search_checkpoint_contract,
    _gpu_unstuck_parameter_active,
    _gpu_suite_scenario_override_context,
    _gpu_suite_scenario_inputs,
    _gpu_suite_search_context,
    _materialize_gpu_override_template,
    _log_gpu_profile,
    _profiled_gpu_exact_worker,
    _DriftMonitor,
    _ObjectiveScale,
    _recover_durable_seed_bootstrap,
    _recover_durable_validations,
    _ready_submission_prefix,
    _update_novelty_stall,
    _validation_probe_count,
    _spearman,
    _submit_gpu_exact_validation,
    _resolve_options,
    _apply_gpu_lean_tm_parallelism_defaults,
    _restore_gpu_result_run_contract,
    _gpu_unstuck_search_sides,
    _single_scenario_metric_surface,
    _successive_halving_survivor_indices,
    _suite_limit_metric_value,
    _trailing_martingale_multicoin_bound_map,
    _ProxyFrontValidationPending,
    _select_exact_validations,
    _select_seed_bootstrap_indices,
    _select_seed_population_indices,
    _select_validation_indices,
    _update_probe_shortfall_log,
    _validate_directional_search_space,
    _validate_dual_multicoin_metrics,
    _validate_gpu_optimizer_overrides,
    _validate_gpu_coin_overrides,
    _validate_pinned_scope_bounds,
    _validate_resume_evidence_budget,
    _validate_seed_bootstrap_plan,
    _validate_seed_side_match,
    _validate_scope,
    _validate_tm_market_mode_bounds,
    _validate_tm_market_template_bounds,
    validate_gpu_preparation_scope,
    _GPU_SUITE_METRICS_KEY,
    _GPU_SUITE_OBJECTIVES_KEY,
    _GPU_SUITE_VIOLATION_KEY,
    EMA_MULTICOIN_LONG_BOUND_MAP,
    EMA_MULTICOIN_SHORT_BOUND_MAP,
    TRAILING_MARTINGALE_BOUND_MAP,
    GPU_DEFAULTS,
    GPU_LEAN_TM_MAX_DISPATCH_CANDIDATE_BARS,
    GPU_LEAN_TM_POPULATION_SIZE,
)
from optimization.gpu.metric_registry import configured_exact_only_gpu_metrics
from optimization.gpu.service import mps_requested_metric_features
from optimization.fine_tune_anchors import ANCHOR_GENE_KEY, ANCHOR_PLAN_KEY
from optimization.warmup import build_optimizer_vector_config


class _Evaluator:
    exchanges = ["bybit"]
    shared_hlcvs_np = {"bybit": np.zeros((100, 1, 4), dtype=np.float64)}


class _MulticoinEvaluator:
    exchanges = ["bybit"]
    shared_hlcvs_np = {"bybit": np.zeros((100, 3, 4), dtype=np.float64)}


def test_gpu_generation_checks_interrupt_before_ask():
    algorithm = MagicMock()
    interrupt_check = MagicMock(side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        _ask_gpu_population(algorithm, interrupt_check)

    interrupt_check.assert_called_once_with()
    algorithm.ask.assert_not_called()


def test_gpu_exact_submission_checks_interrupt_before_apply_async():
    pool = MagicMock()
    interrupt_check = MagicMock(side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        _submit_gpu_exact_validation(pool, [1.0], interrupt_check)

    interrupt_check.assert_called_once_with()
    pool.apply_async.assert_not_called()


def test_gpu_exact_submission_uses_profiled_worker_only_when_enabled(
    monkeypatch,
):
    pool = MagicMock()
    interrupt_check = MagicMock()
    monkeypatch.setattr(
        "optimization.backends.gpu_backend.time.perf_counter", lambda: 12.5
    )

    _submit_gpu_exact_validation(
        pool, [1.0], interrupt_check, profile=True
    )

    assert pool.apply_async.call_args.args == (
        _profiled_gpu_exact_worker,
        ([1.0], 12.5),
    )


def test_gpu_profiled_exact_worker_records_actual_queue_wait(monkeypatch):
    ticks = iter((10.0, 14.0))
    monkeypatch.setattr(
        "optimization.backends.gpu_backend.time.perf_counter",
        lambda: next(ticks),
    )
    monkeypatch.setattr(
        "optimization.backends.gpu_backend._evaluate_pymoo_worker_from_globals",
        lambda vector: {"F": vector},
    )

    payload = _profiled_gpu_exact_worker([1.0], 7.0)

    assert payload["F"] == [1.0]
    assert payload["__gpu_profile_queue_wait_seconds__"] == pytest.approx(3.0)
    assert payload["__gpu_profile_worker_seconds__"] == pytest.approx(4.0)


def test_gpu_profile_log_is_structured_json(caplog):
    with caplog.at_level("INFO"):
        _log_gpu_profile(
            "generation", generation=3, timings_seconds={"wall": 1.25}
        )

    line = next(
        message
        for message in caplog.messages
        if message.startswith("[gpu-profile] ")
    )
    payload = json.loads(line.removeprefix("[gpu-profile] "))
    assert payload == {
        "schema_version": 1,
        "event": "generation",
        "generation": 3,
        "timings_seconds": {"wall": 1.25},
    }


def test_gpu_profile_elapsed_uses_monotonic_clock(monkeypatch):
    monkeypatch.setattr(
        "optimization.backends.gpu_backend.time.perf_counter", lambda: 15.5
    )
    monkeypatch.setattr(
        "optimization.backends.gpu_backend.time.time",
        lambda: (_ for _ in ()).throw(
            AssertionError("profile duration used the wall clock")
        ),
    )

    assert _gpu_profile_elapsed(10.0) == pytest.approx(5.5)


def test_gpu_interrupt_discards_incomplete_ask_tell_without_checkpointing():
    save_checkpoint = MagicMock()

    saved = _checkpoint_gpu_interrupt(
        generation_in_progress=True,
        generation=7,
        exact_done=23,
        save_checkpoint=save_checkpoint,
    )

    assert not saved
    save_checkpoint.assert_not_called()


def test_gpu_interrupt_checkpoints_complete_generation_state():
    save_checkpoint = MagicMock()

    saved = _checkpoint_gpu_interrupt(
        generation_in_progress=False,
        generation=7,
        exact_done=23,
        save_checkpoint=save_checkpoint,
    )

    assert saved
    save_checkpoint.assert_called_once_with(force=True)


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
    options = _resolve_options(config)
    assert options["auto_lean_parallelism"] is True
    assert options["population_size"] == 1024
    assert options["max_dispatch_candidate_bars"] == 1_000_000_000
    assert options["seed_bootstrap"] == {"mode": "auto", "max_exact": 128}

    config["optimize"]["gpu"]["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["max_dispatch_candidate_bars"] = 0
    with pytest.raises(ValueError, match="max_dispatch_candidate_bars"):
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


def test_gpu_seed_bootstrap_options_are_explicit_and_fail_closed():
    config = _long_only_ema_config()
    config["optimize"]["gpu"]["seed_bootstrap"] = {
        "mode": "SCREENED",
        "max_exact": 64,
    }
    assert _resolve_options(config)["seed_bootstrap"] == {
        "mode": "screened",
        "max_exact": 64,
    }

    config["optimize"]["gpu"]["seed_bootstrap"]["mode"] = "best-effort"
    with pytest.raises(ValueError, match="seed_bootstrap.mode"):
        _resolve_options(config)

    config["optimize"]["gpu"]["seed_bootstrap"] = {
        "mode": "auto",
        "max_exact": 0,
    }
    with pytest.raises(ValueError, match="max_exact"):
        _resolve_options(config)

    config["optimize"]["gpu"]["seed_bootstrap"] = {
        "mode": "auto",
        "max_exact": 64,
        "unexpected": True,
    }
    with pytest.raises(ValueError, match="unknown.*unexpected"):
        _resolve_options(config)

    config["optimize"]["gpu"]["seed_bootstrap"] = "auto"
    with pytest.raises(TypeError, match="seed_bootstrap must be an object"):
        _resolve_options(config)


def test_gpu_seed_bootstrap_auto_switches_only_at_the_exact_cap():
    policy = {"mode": "auto", "max_exact": 128}

    assert _effective_seed_bootstrap_mode(policy, 0) == "none"
    assert _effective_seed_bootstrap_mode(policy, 128) == "exact"
    assert _effective_seed_bootstrap_mode(policy, 129) == "screened"
    assert (
        _effective_seed_bootstrap_mode({"mode": "exact", "max_exact": 1}, 500)
        == "exact"
    )


def test_gpu_seed_bootstrap_deduplicates_runtime_override_equivalents():
    vectors = [[0.1, 0.2], [0.1, 0.9], [0.3, 0.4]]

    deduplicated, dropped = _deduplicate_canonical_seed_vectors(
        vectors,
        hash_vector=lambda vector: str(vector[0]),
    )

    assert deduplicated == [vectors[0], vectors[2]]
    assert dropped == 1


def test_gpu_exact_validation_disables_evaluator_duplicate_perturbation():
    base = SimpleNamespace(use_duplicate_guard=True)
    suite = SimpleNamespace(base=base)

    _disable_gpu_exact_duplicate_guard(suite)

    assert base.use_duplicate_guard is False


def test_gpu_seed_bootstrap_selection_keeps_extremes_front_and_probe_coverage():
    objectives = np.asarray(
        [
            [0.0, 10.0],
            [2.0, 7.0],
            [5.0, 5.0],
            [7.0, 2.0],
            [10.0, 0.0],
            [8.0, 8.0],
        ],
        dtype=np.float64,
    )
    scores = np.sum(objectives, axis=1)
    violations = np.full(len(objectives), -1.0, dtype=np.float64)

    selected = _select_seed_bootstrap_indices(
        objectives,
        scores,
        violations,
        total=4,
    )

    selected_ids = {index for index, _probe, _front in selected}
    assert {0, 4}.issubset(selected_ids)
    assert len(selected) == 4
    assert any(probe and not front for _index, probe, front in selected)


def test_gpu_seed_bootstrap_selection_reserves_front_and_probe_slots():
    objectives = np.asarray(
        [[float(index), float(9 - index)] for index in range(10)]
        + [[20.0 + index, 20.0 + index] for index in range(10)],
        dtype=np.float64,
    )
    scores = np.sum(objectives, axis=1)
    violations = np.full(len(objectives), -1.0, dtype=np.float64)

    selected = _select_seed_bootstrap_indices(
        objectives,
        scores,
        violations,
        total=8,
    )

    assert len(selected) == 8
    assert sum(front for _index, _probe, front in selected) == 6
    assert sum(probe for _index, probe, _front in selected) == 2
    assert {0, 9}.issubset({index for index, _probe, _front in selected})


def test_gpu_seed_bootstrap_extremes_do_not_consume_reserved_probes():
    objective_count = 8
    extremes = np.full((objective_count, objective_count), 10.0)
    for index in range(objective_count):
        extremes[index, index] = 0.0
    dominated = np.full((4, objective_count), 20.0)
    objectives = np.vstack((extremes, dominated))
    scores = np.sum(objectives, axis=1)
    violations = np.full(len(objectives), -1.0, dtype=np.float64)

    selected = _select_seed_bootstrap_indices(
        objectives,
        scores,
        violations,
        total=8,
    )

    assert sum(front for _index, _probe, front in selected) == 6
    assert sum(probe for _index, probe, _front in selected) == 2


def test_gpu_seed_bootstrap_all_infeasible_reserves_lowest_violations():
    objectives = np.asarray(
        [[0.0, 10.0], [10.0, 0.0], [6.0, 6.0], [7.0, 7.0]],
        dtype=np.float64,
    )
    scores = np.sum(objectives, axis=1)
    violations = np.asarray([10.0, 8.0, 0.01, 0.02], dtype=np.float64)

    selected = _select_seed_bootstrap_indices(
        objectives,
        scores,
        violations,
        total=2,
    )

    assert 2 in {index for index, _probe, _front in selected}


def test_gpu_seed_population_reduction_prefers_feasible_pareto_diversity():
    objectives = np.asarray(
        [[0.0, 4.0], [2.0, 2.0], [4.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    violations = np.asarray([-1.0, -1.0, -1.0, 0.5], dtype=np.float64)

    selected = _select_seed_population_indices(
        objectives,
        violations,
        count=3,
    )

    assert set(selected) == {0, 1, 2}


def test_gpu_seed_bootstrap_checkpoint_plan_fails_closed_on_pool_drift():
    vectors = [[0.1], [0.2], [0.3]]
    digest = hashlib.sha256(b"0.1\n0.2\n0.3").hexdigest()
    contract = {
        "effective_mode": "screened",
        "seed_count": 3,
        "selected_exact_count": 2,
        "all_seeds_exact": False,
        "seed_pool_sha256": digest,
    }
    kwargs = {
        "hash_vector": lambda vector: str(vector[0]),
        "proxy_metrics": {0: {}, 2: {}},
        "proxy_objectives": [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]],
        "proxy_violations": [-1.0, -1.0, -1.0],
    }

    _validate_seed_bootstrap_plan(
        vectors,
        [(0, False, True), (2, True, False)],
        [0, 2, 1],
        contract,
        **kwargs,
    )

    changed = copy.deepcopy(vectors)
    changed[1][0] = 0.25
    with pytest.raises(RuntimeError, match="does not match its seed contract"):
        _validate_seed_bootstrap_plan(
            changed,
            [(0, False, True), (2, True, False)],
            [0, 2, 1],
            contract,
            **kwargs,
        )


def test_gpu_seed_bootstrap_checkpoint_accepts_only_valid_pending_screen():
    vectors = [[0.1], [0.2], [0.3]]
    contract = {
        "effective_mode": "screened",
        "max_exact": 2,
        "seed_count": 3,
        "selected_exact_count": 2,
        "all_seeds_exact": False,
        "seed_pool_sha256": hashlib.sha256(b"0.1\n0.2\n0.3").hexdigest(),
    }

    _validate_seed_bootstrap_plan(
        vectors,
        [],
        [],
        contract,
        hash_vector=lambda vector: str(vector[0]),
        screen_complete=False,
    )

    with pytest.raises(RuntimeError, match="invalid partial evidence"):
        _validate_seed_bootstrap_plan(
            vectors,
            [(0, False, True)],
            [],
            contract,
            hash_vector=lambda vector: str(vector[0]),
            screen_complete=False,
        )


def _lean_tm_bounds(side="long"):
    bounds = {
        f"{side}_entry_retracement_base_pct": Bound(0.001, 0.02),
        f"{side}_close_retracement_base_pct": Bound(0.001, 0.02),
    }
    for suffix in (
        "entry_threshold_volatility_1h_weight",
        "entry_threshold_volatility_1m_weight",
        "entry_retracement_volatility_1h_weight",
        "entry_retracement_volatility_1m_weight",
        "close_threshold_volatility_1h_weight",
        "close_threshold_volatility_1m_weight",
        "close_retracement_volatility_1h_weight",
        "close_retracement_volatility_1m_weight",
    ):
        bounds[f"{side}_{suffix}"] = Bound(0.0, 0.0)
    return bounds


def test_gpu_lean_tm_parallelism_requires_complete_compileout_proof():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["market_orders_allowed"] = False
    config["live"]["max_realized_loss_pct"] = 1.0
    bounds = _lean_tm_bounds()

    def eligible(candidate=config, candidate_bounds=bounds, **kwargs):
        return _gpu_lean_tm_parallelism_eligible(
            candidate,
            candidate_bounds,
            kwargs.pop("enabled_sides", {"long"}),
            suite_enabled=kwargs.pop("suite_enabled", False),
            coin_count=kwargs.pop("coin_count", 1),
            requested_metric_features=kwargs.pop(
                "requested_metric_features", frozenset()
            ),
        )

    assert eligible()
    assert not eligible(enabled_sides={"long", "short"})
    assert not eligible(suite_enabled=True)
    assert not eligible(coin_count=2)
    overridden_config = copy.deepcopy(config)
    overridden_config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"hsl": {"enabled": True}}}}
    }
    assert not eligible(candidate=overridden_config)
    ema_config = copy.deepcopy(config)
    ema_config["live"]["strategy_kind"] = "ema_anchor"
    assert not eligible(candidate=ema_config)
    short_config = _directional_tm_config(
        long_enabled=False, short_enabled=True
    )
    short_config["live"]["market_orders_allowed"] = False
    short_config["live"]["max_realized_loss_pct"] = 1.0
    assert _gpu_lean_tm_parallelism_eligible(
        short_config,
        _lean_tm_bounds("short"),
        {"short"},
        suite_enabled=False,
        coin_count=1,
        requested_metric_features=frozenset(),
    )

    for feature in (
        "entry_interval",
        "strategy_eq_recovery_distribution",
        "btc_analysis",
        "btc_intraday_risk",
        "equity_balance_diff",
        "hsl_ema_tail",
        "hsl_raw_drawdown",
        "hsl_raw_tail",
        "hsl_diagnostics",
        "coin_fill_counts",
    ):
        assert not eligible(requested_metric_features={feature}), feature

    for path, value in (
        (("bot", "long", "hsl", "enabled"), True),
        (("bot", "long", "unstuck", "enabled"), True),
        (("bot", "long", "risk", "position_exposure_enforcer_enabled"), True),
        (("bot", "long", "risk", "total_exposure_enforcer_enabled"), True),
        (("live", "market_orders_allowed"), True),
        (("live", "max_realized_loss_pct"), 0.1),
    ):
        candidate = copy.deepcopy(config)
        target = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert not eligible(candidate=candidate), path

    recursive_bounds = dict(bounds)
    recursive_bounds["long_entry_retracement_base_pct"] = Bound(0.0, 0.02)
    assert not eligible(candidate_bounds=recursive_bounds)
    volatile_bounds = dict(bounds)
    volatile_bounds["long_close_threshold_volatility_1h_weight"] = Bound(
        0.0, 1.0
    )
    assert not eligible(candidate_bounds=volatile_bounds)


def test_gpu_lean_tm_parallelism_auto_sizes_only_untuned_defaults():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["market_orders_allowed"] = False
    config["live"]["max_realized_loss_pct"] = 1.0
    bounds = _lean_tm_bounds()
    options = copy.deepcopy(GPU_DEFAULTS)

    assert _apply_gpu_lean_tm_parallelism_defaults(
        options,
        config,
        bounds,
        {"long"},
        suite_enabled=False,
        coin_count=1,
        requested_metric_features=frozenset(),
        mps_chip_name="Apple M3",
    )
    assert options["population_size"] == GPU_LEAN_TM_POPULATION_SIZE
    assert (
        options["max_dispatch_candidate_bars"]
        == GPU_LEAN_TM_MAX_DISPATCH_CANDIDATE_BARS
    )

    metric_enabled = copy.deepcopy(GPU_DEFAULTS)
    assert not _apply_gpu_lean_tm_parallelism_defaults(
        metric_enabled,
        config,
        bounds,
        {"long"},
        suite_enabled=False,
        coin_count=1,
        requested_metric_features={"entry_interval"},
        mps_chip_name="Apple M3",
    )
    assert metric_enabled == GPU_DEFAULTS

    for key, value in (
        ("auto_lean_parallelism", False),
        ("population_size", 2048),
        ("batch_size", 1024),
        ("max_dispatch_candidate_bars", 500_000_000),
    ):
        tuned = copy.deepcopy(GPU_DEFAULTS)
        tuned[key] = value
        assert not _apply_gpu_lean_tm_parallelism_defaults(
            tuned,
            config,
            bounds,
            {"long"},
            suite_enabled=False,
            coin_count=1,
            requested_metric_features=frozenset(),
            mps_chip_name="Apple M3 Max",
        )

    for chip_name in ("", "Apple M2 Max", "Apple M4"):
        assert not _apply_gpu_lean_tm_parallelism_defaults(
            copy.deepcopy(GPU_DEFAULTS),
            config,
            bounds,
            {"long"},
            suite_enabled=False,
            coin_count=1,
            requested_metric_features=frozenset(),
            mps_chip_name=chip_name,
        )

    for key in ("population_size", "batch_size", "max_dispatch_candidate_bars"):
        explicit_config = copy.deepcopy(config)
        explicit_config["optimize"]["gpu"][key] = GPU_DEFAULTS[key]
        assert not _apply_gpu_lean_tm_parallelism_defaults(
            copy.deepcopy(GPU_DEFAULTS),
            explicit_config,
            bounds,
            {"long"},
            suite_enabled=False,
            coin_count=1,
            requested_metric_features=frozenset(),
            mps_chip_name="Apple M3",
        )


def test_apple_mps_chip_probe_does_not_depend_on_shell_path(monkeypatch):
    from optimization.backends import gpu_backend

    gpu_backend._apple_mps_chip_name.cache_clear()
    monkeypatch.setattr(gpu_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_backend.platform, "machine", lambda: "arm64")
    run = MagicMock(return_value=SimpleNamespace(stdout="Apple M3\n"))
    monkeypatch.setattr(gpu_backend.subprocess, "run", run)

    assert gpu_backend._apple_mps_chip_name() == "Apple M3"
    assert run.call_args.args[0] == [
        "/usr/sbin/sysctl",
        "-n",
        "machdep.cpu.brand_string",
    ]
    gpu_backend._apple_mps_chip_name.cache_clear()


@pytest.mark.parametrize(
    ("metric", "expected_features"),
    (
        ("entry_interval_hours_p95", {"entry_interval"}),
        (
            "strategy_eq_recovery_days_p99",
            {"strategy_eq_recovery_distribution"},
        ),
        ("adg_btc", {"btc_analysis"}),
        ("drawdown_worst_btc", {"btc_analysis", "btc_intraday_risk"}),
        (
            "equity_balance_diff_neg_max_usd",
            {"equity_balance_diff"},
        ),
        (
            "drawdown_worst_mean_1pct_ema_strategy_eq",
            {"hsl_ema_tail", "hsl_diagnostics"},
        ),
        (
            "drawdown_worst_mean_1pct_strategy_eq_long",
            {"hsl_raw_drawdown", "hsl_raw_tail", "hsl_diagnostics"},
        ),
        ("hard_stop_time_in_red_pct", {"hsl_diagnostics"}),
        ("fills_top_symbol_share", {"coin_fill_counts"}),
    ),
)
def test_gpu_lean_tm_metric_feature_detection(metric, expected_features):
    assert mps_requested_metric_features(
        {metric}, strategy_kind="trailing_martingale"
    ) == expected_features


def test_gpu_lean_tm_metric_feature_detection_is_torch_free(monkeypatch):
    monkeypatch.setitem(sys.modules, "optimization.gpu.metrics", None)
    assert mps_requested_metric_features(
        {"entry_interval_hours_p95"}, strategy_kind="trailing_martingale"
    ) == {"entry_interval"}


def test_gpu_successive_halving_options_are_opt_in_and_fail_closed():
    config = _long_only_ema_config()
    options = _resolve_options(config)
    assert options["successive_halving"] == {
        "enabled": False,
        "history_fractions": [0.25, 0.5, 1.0],
        "survival_fraction": 0.5,
        "min_survivors": 64,
    }

    config["optimize"]["gpu"]["successive_halving"] = {
        "enabled": True,
        "history_fractions": [0.5, 0.25, 1.0],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        _resolve_options(config)

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["successive_halving"] = {
        "enabled": True,
        "history_fractions": [0.25, 0.5, 0.999_999_999_999_5],
    }
    assert _resolve_options(config)["successive_halving"][
        "history_fractions"
    ] == [0.25, 0.5, 1.0]

    config = _long_only_ema_config()
    config["optimize"]["gpu"]["successive_halving"] = {
        "enabled": True,
        "min_survivors": 7,
    }
    with pytest.raises(ValueError, match="validate_per_generation"):
        _resolve_options(config)


def test_successive_halving_keeps_pareto_diversity_and_full_rung_evidence_only():
    candidates = [{"id": index} for index in range(8)]
    calls = []

    def evaluate_proxy(stage_candidates, *, history_fraction):
        ids = [candidate["id"] for candidate in stage_candidates]
        calls.append((history_fraction, ids))
        return [
            {
                "left": float(index),
                "right": float(7 - index),
                "violation": 0.0,
            }
            for index in ids
        ]

    def proxy_fitness(rows):
        return (
            np.asarray([[row["left"], row["right"]] for row in rows]),
            np.asarray([row["violation"] for row in rows]),
        )

    metric_rows, objectives, violations, full_indices, trace = (
        _evaluate_successive_halving(
            candidates,
            policy={
                "history_fractions": [0.25, 0.5, 1.0],
                "survival_fraction": 0.5,
                "min_survivors": 2,
            },
            evaluate_proxy=evaluate_proxy,
            proxy_fitness=proxy_fitness,
            interrupt_check=lambda: None,
        )
    )

    assert [len(ids) for _fraction, ids in calls] == [8, 4, 2]
    assert trace == [
        {
            "rung": 1,
            "history_fraction": 0.25,
            "candidate_count": 8,
            "survivor_count": 4,
        },
        {
            "rung": 2,
            "history_fraction": 0.5,
            "candidate_count": 4,
            "survivor_count": 2,
        },
        {
            "rung": 3,
            "history_fraction": 1.0,
            "candidate_count": 2,
            "survivor_count": 2,
        },
    ]
    assert len(metric_rows) == len(objectives) == len(violations) == 8
    assert len(full_indices) == 2
    assert len(np.unique(full_indices)) == 2
    assert np.all(np.isfinite(violations[full_indices]))
    assert np.all(np.isinf(np.delete(violations, full_indices)))
    assert sum(item["candidate_count"] for item in trace) == 14


def test_successive_halving_survivors_prioritize_feasibility_then_violation():
    objectives = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    violations = np.asarray([0.0, 0.0, 0.2, 0.1])

    survivors = _successive_halving_survivor_indices(
        objectives, violations, count=3
    )

    assert set(survivors[:2]) == {0, 1}
    assert survivors[2] == 3


def test_successive_halving_survivors_remain_unique_when_objectives_tie():
    survivors = _successive_halving_survivor_indices(
        np.zeros((8, 2), dtype=np.float64),
        np.zeros(8, dtype=np.float64),
        count=6,
    )

    assert survivors.tolist() == [0, 1, 2, 3, 4, 5]


def test_fresh_run_accepts_partial_suffix_with_opportunistic_probes():
    config = _long_only_ema_config()
    config["optimize"]["gpu"]["validate_per_generation"] = 8
    config["optimize"]["gpu"]["drift_probes"] = 1
    config["optimize"]["gpu"]["drift_window"] = 96
    config["optimize"]["iters"] = 97

    options = _resolve_options(config)

    assert options["drift_probes"] == 1


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


def test_resume_budget_accepts_truthful_broad_probe_scarcity():
    pairs = [_drift_pair(front=True) for _ in range(57)]

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

    contract = _gpu_nsga2_checkpoint_contract(
        config, population_size=8, n_params=5
    )
    assert contract == {
        "version": 1,
        "algorithm": "nsga2",
        "population_size": 8,
        "configured_seed": None,
        "crossover": {"operator": "sbx", "prob_var": 0.7, "eta": 11.0},
        "mutation": {"operator": "pm", "prob": 0.2, "eta": 13.0},
        "eliminate_duplicates": False,
    }


def test_gpu_nsga2_accepts_exact_seed_sampling_before_first_ask():
    from pymoo.core.problem import Problem
    from pymoo.core.termination import NoTermination

    config = _long_only_ema_config()
    initial = np.zeros((8, 2), dtype=np.float64)
    exact_seed_sampling = np.asarray(
        [[index / 10.0, (index + 1) / 10.0] for index in range(8)],
        dtype=np.float64,
    )
    algorithm = _build_gpu_nsga2(
        config,
        sampling=initial,
        population_size=8,
        n_params=2,
    )
    algorithm.setup(
        Problem(n_var=2, n_obj=1, xl=np.zeros(2), xu=np.ones(2)),
        termination=NoTermination(),
        seed=1,
        verbose=False,
    )

    algorithm.initialization.sampling = exact_seed_sampling
    population = algorithm.ask()

    np.testing.assert_allclose(population.get("X"), exact_seed_sampling)


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
        "risk_twel_enforcer_threshold",
        "risk_we_excess_allowance_pct",
        "risk_wel_enforcer_threshold",
        "total_wallet_exposure_limit",
        "unstuck_close_pct",
        "unstuck_ema_dist",
        "unstuck_loss_allowance_pct",
        "unstuck_threshold",
        "hsl_cooldown_minutes_after_red",
        "hsl_ema_span_minutes",
        "hsl_red_threshold",
    }

    assert set(TRAILING_MARTINGALE_BOUND_MAP) == {
        f"{side}_{suffix}"
        for side in ("long", "short")
        for suffix in expected_suffixes
    }


def test_cpu_runtime_imports_do_not_import_torch_or_mps_kernel():
    script = (
        "import json, sys; import backtest, passivbot, optimization.backends, "
        "tools.gpu_proxy_benchmark; "
        "print(json.dumps({"
        "'torch': 'torch' in sys.modules, "
        "'mps_kernel': 'optimization.gpu.mps_kernel' in sys.modules"
        "}))"
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

    assert json.loads(result.stdout.strip()) == {
        "torch": False,
        "mps_kernel": False,
    }


@pytest.mark.parametrize(
    "raw_optimize",
    [
        {
            "scoring": [
                {"goal": "min", "metric": "peak_recovery_days_strategy_eq"}
            ]
        },
        {
            "limits": [
                {
                    "metric": "peak_recovery_days_strategy_eq",
                    "penalize_if": "greater_than",
                    "value": 30.0,
                }
            ]
        },
        {
            "limits": {
                "penalize_if_greater_than_peak_recovery_days_strategy_eq": 30.0
            }
        },
        {
            "limits": {
                "lower_bound_peak_recovery_days_strategy_eq": 30.0
            }
        },
        {
            "limits": {
                "upper_bound_peak_recovery_days_strategy_eq": 30.0
            }
        },
        {
            "limits": (
                "--penalize_if_greater_than_peak_recovery_days_strategy_eq 30"
            )
        },
        {
            "limits": (
                '[{"metric": "peak_recovery_days_strategy_eq", '
                '"penalize_if": "greater_than", "value": 30}]'
            )
        },
        {
            "limits": (
                "[\n {metric: peak_recovery_days_strategy_eq\n"
                "  penalize_if: greater_than\n  value: 30}\n]"
            )
        },
    ],
)
def test_gpu_metric_provenance_recovers_exact_only_alias_before_canonicalization(
    raw_optimize,
):
    config = {
        "optimize": {
            "scoring": [
                {"goal": "min", "metric": "strategy_eq_recovery_days_max"}
            ],
            "limits": [],
        },
        "_raw_effective": {"optimize": raw_optimize},
    }

    assert configured_exact_only_gpu_metrics(config) == {
        "peak_recovery_days_strategy_eq"
    }


def test_gpu_metric_provenance_does_not_match_retained_side_recovery_metrics():
    config = {
        "optimize": {
            "scoring": [
                {"goal": "min", "metric": "peak_recovery_days_strategy_eq_long"}
            ],
            "limits": [],
        }
    }

    assert not configured_exact_only_gpu_metrics(config)


def test_gpu_metric_provenance_ignores_disabled_exact_only_limit():
    config = {
        "optimize": {
            "scoring": [{"goal": "max", "metric": "adg_strategy_eq"}],
            "limits": [
                {
                    "enabled": False,
                    "metric": "peak_recovery_days_strategy_eq",
                }
            ],
        }
    }

    assert not configured_exact_only_gpu_metrics(config)


def test_gpu_metric_provenance_ignores_scenario_labels_that_look_like_metrics():
    config = {
        "optimize": {
            "scoring": [
                {
                    "goal": "max",
                    "metric": "adg_strategy_eq",
                    "scenario": "fills_count",
                }
            ],
            "limits": [
                {
                    "metric": "drawdown_worst_strategy_eq",
                    "penalize_if": "greater_than",
                    "scenario": "gain_btc",
                    "value": 0.3,
                }
            ],
        }
    }

    assert not configured_exact_only_gpu_metrics(config)


def test_gpu_metric_provenance_prefers_effective_over_superseded_raw_config():
    config = {
        "optimize": {
            "scoring": [{"goal": "max", "metric": "adg_strategy_eq"}],
            "limits": [],
        },
        "_raw": {
            "optimize": {
                "scoring": [
                    {
                        "goal": "min",
                        "metric": "peak_recovery_days_strategy_eq",
                    }
                ]
            }
        },
        "_raw_effective": {
            "optimize": {
                "scoring": [
                    {"goal": "max", "metric": "adg_strategy_eq"}
                ],
                "limits": [],
            }
        },
    }

    assert not configured_exact_only_gpu_metrics(config)


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


def test_gpu_scope_allows_suite_only_when_explicitly_requested():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True

    with pytest.raises(ValueError, match="suite"):
        _validate_scope(config, _Evaluator())

    assert _validate_scope(config, _Evaluator(), allow_suite=True) == "bybit"


def test_gpu_suite_activation_comes_from_exact_evaluator_wrapper():
    config = _long_only_ema_config()
    base = object()
    suite = object()

    assert _gpu_suite_enabled(config, base, suite) is True
    assert _gpu_suite_enabled(config, base, base) is False

    config["backtest"]["suite_enabled"] = True
    with pytest.raises(TypeError, match="canonical SuiteEvaluator"):
        _gpu_suite_enabled(config, base, base)


def test_gpu_suite_inputs_materialize_one_selected_coin():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    master = np.arange(10 * 3 * 4, dtype=np.float64).reshape(10, 3, 4)
    ctx = SimpleNamespace(
        label="stress",
        overrides={},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return master, np.ones(10), [1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert len(prepared) == 1
    assert prepared[0]["hlcvs"].shape == (10, 1, 4)
    assert np.array_equal(prepared[0]["hlcvs"][:, 0], master[:, 1])
    assert prepared[0]["coin_count"] == 1


def test_gpu_suite_inputs_materialize_multicoin_subset():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    master = np.arange(10 * 4 * 4, dtype=np.float64).reshape(10, 4, 4)
    ctx = SimpleNamespace(
        label="three_coins",
        overrides={},
        exchanges=["bybit"],
        msss={"bybit": {coin: {} for coin in ("BTC", "ETH", "SOL")}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return master, np.ones(10), [0, 2, 3]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["coin_count"] == 3
    assert prepared[0]["hlcvs"].shape == (10, 3, 4)
    assert np.array_equal(prepared[0]["hlcvs"][:, 1], master[:, 2])


def test_gpu_suite_inputs_accept_dual_side_multicoin_hedge_scenario():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["live"]["hedge_mode"] = True
    config["live"]["forager_score_hysteresis_pct"] = 0.02
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["bot"]["long"]["risk"]["n_positions"] = 2
    config["bot"]["short"]["risk"]["n_positions"] = 2
    master = np.zeros((10, 3, 4), dtype=np.float64)
    ctx = SimpleNamespace(
        label="dual",
        overrides={},
        exchanges=["bybit"],
        msss={
            "bybit": {
                "BTC": {},
                "ETH": {},
                "SOL": {},
                "__meta__": {},
            }
        },
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return master, np.ones(10), [0, 1, 2]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert _gpu_suite_search_context(prepared) == (
        3,
        3,
        ("long", "short"),
    )


@pytest.mark.parametrize(
    ("overrides", "exchanges", "coin_indices", "message"),
    [
        ({"live.strategy_kind": "trailing_martingale"}, ["bybit"], [0], "outside the supported"),
    ],
)
def test_gpu_suite_inputs_reject_unsupported_scenario_scope(
    overrides, exchanges, coin_indices, message
):
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    ctx = SimpleNamespace(
        label="stress",
        overrides=overrides,
        exchanges=exchanges,
        msss={exchange: {"BTC": {}, "__meta__": {}} for exchange in exchanges},
        timestamps={exchange: np.arange(10, dtype=np.int64) for exchange in exchanges},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), coin_indices

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    with pytest.raises(ValueError, match=message):
        _gpu_suite_scenario_inputs(config, Suite())


def test_gpu_suite_inputs_materialize_each_exchange_in_one_scenario():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    exchanges = ["bybit", "binance"]
    ctx = SimpleNamespace(
        label="two_venues",
        config={"backtest": {"coin_sources": {}}},
        overrides={},
        exchanges=exchanges,
        msss={
            exchange: {
                f"{exchange}_coin": {},
                "__meta__": {},
            }
            for exchange in exchanges
        },
        timestamps={
            exchange: np.arange(10, dtype=np.int64) for exchange in exchanges
        },
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, exchange):
            width = 1 if exchange == "bybit" else 2
            return np.zeros((10, width, 4)), np.ones(10), [width - 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert [item["ctx"] for item in prepared] == [ctx, ctx]
    assert [item["exchange"] for item in prepared] == exchanges
    assert [item["coins"] for item in prepared] == [
        ["bybit_coin"],
        ["binance_coin"],
    ]
    assert all(item["coin_count"] == 1 for item in prepared)
    assert prepared[0]["config"] == prepared[1]["config"]
    assert prepared[0]["config"] is not prepared[1]["config"]


@pytest.mark.parametrize(
    ("dotted_path", "value", "resolved_path"),
    [
        (
            "backtest.dynamic_wel_by_tradability",
            False,
            ("backtest", "dynamic_wel_by_tradability"),
        ),
        ("backtest.starting_balance", 12_345.0, ("backtest", "starting_balance")),
        ("backtest.maker_fee_override", 0.0002, ("backtest", "maker_fee_override")),
        ("backtest.taker_fee_override", 0.0007, ("backtest", "taker_fee_override")),
        (
            "backtest.market_order_slippage_pct",
            0.0015,
            ("backtest", "market_order_slippage_pct"),
        ),
        (
            "backtest.filter_by_min_effective_cost",
            True,
            ("backtest", "filter_by_min_effective_cost"),
        ),
        (
            "backtest.liquidation_threshold",
            0.1,
            ("backtest", "liquidation_threshold"),
        ),
        (
            "live.forager_score_hysteresis_pct",
            0.03,
            ("live", "forager_score_hysteresis_pct"),
        ),
        ("live.hedge_mode", True, ("live", "hedge_mode")),
        ("live.hsl_signal_mode", "coin", ("live", "hsl_signal_mode")),
        (
            "live.market_order_near_touch_threshold",
            0.002,
            ("live", "market_order_near_touch_threshold"),
        ),
        (
            "live.market_orders_allowed",
            True,
            ("live", "market_orders_allowed"),
        ),
        (
            "live.max_realized_loss_pct",
            0.05,
            ("live", "max_realized_loss_pct"),
        ),
        (
            "live.pnls_max_lookback_days",
            7.0,
            ("live", "pnls_max_lookback_days"),
        ),
    ],
)
def test_gpu_suite_inputs_accept_modeled_non_bot_overrides(
    dotted_path, value, resolved_path
):
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    ctx = SimpleNamespace(
        label="modeled_runtime",
        overrides={dotted_path: value},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            target = scenario
            for part in resolved_path[:-1]:
                target = target[part]
            target[resolved_path[-1]] = value
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())
    target = prepared[0]["config"]
    for part in resolved_path:
        target = target[part]
    assert target == value


def test_gpu_suite_hsl_signal_override_revalidates_effective_topology():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    config["bot"]["long"]["hsl"]["enabled"] = True
    ctx = SimpleNamespace(
        label="invalid_signal",
        overrides={"live.hsl_signal_mode": "not-a-mode"},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["live"]["hsl_signal_mode"] = "not-a-mode"
            return scenario

    with pytest.raises(ValueError, match="coin, pside, or unified"):
        _gpu_suite_scenario_inputs(config, Suite())


def test_gpu_suite_inputs_accept_single_coin_tm_market_hsl_min_cost_and_risk():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["filter_by_min_effective_cost"] = True
    config["backtest"]["liquidation_threshold"] = 0.05
    config["live"]["market_orders_allowed"] = True
    config["live"]["hsl_signal_mode"] = "coin"
    config["live"]["pnls_max_lookback_days"] = "all"
    config["live"]["max_realized_loss_pct"] = 0.05
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["bot"]["long"]["unstuck"]["enabled"] = True
    config["bot"]["long"]["risk"][
        "position_exposure_enforcer_enabled"
    ] = True
    config["bot"]["long"]["risk"]["total_exposure_enforcer_enabled"] = True
    ctx = SimpleNamespace(
        label="tm_market_hsl",
        overrides={},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert len(prepared) == 1
    assert prepared[0]["config"]["live"]["market_orders_allowed"] is True
    assert prepared[0]["config"]["live"]["max_realized_loss_pct"] == 0.05
    assert prepared[0]["config"]["bot"]["long"]["hsl"]["enabled"] is True
    assert prepared[0]["config"]["bot"]["long"]["unstuck"]["enabled"] is True
    assert (
        prepared[0]["config"]["bot"]["long"]["risk"][
            "position_exposure_enforcer_enabled"
        ]
        is True
    )
    assert (
        prepared[0]["config"]["backtest"]["filter_by_min_effective_cost"]
        is True
    )


def test_gpu_suite_inputs_accept_scenario_local_modeled_coin_overrides():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    overrides = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {"ema_anchor": {"offset": 0.012}},
                        "risk": {"entry_cooldown_minutes": 15.0},
                    }
                }
            }
        }
    }
    ctx = SimpleNamespace(
        label="eth_override",
        overrides=overrides,
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "ETH": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), [0, 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["coin_overrides"] = copy.deepcopy(overrides["coin_overrides"])
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["overrides"] == overrides
    assert prepared[0]["config"]["coin_overrides"]["ETH"]["bot"]["long"][
        "strategy"
    ]["ema_anchor"]["offset"] == pytest.approx(0.012)


def test_gpu_suite_inputs_accept_scenario_local_tm_coin_overrides():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["backtest"]["suite_enabled"] = True
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    overrides = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "trailing_martingale": {
                                "entry": {"threshold_base_pct": 0.012}
                            }
                        },
                        "wallet_exposure_limit": 0.4,
                    }
                }
            }
        }
    }
    ctx = SimpleNamespace(
        label="eth_tm_override",
        overrides=overrides,
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "ETH": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), [0, 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["coin_overrides"] = copy.deepcopy(overrides["coin_overrides"])
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["overrides"] == overrides
    assert prepared[0]["config"]["coin_overrides"]["ETH"]["bot"]["long"][
        "strategy"
    ]["trailing_martingale"]["entry"]["threshold_base_pct"] == pytest.approx(
        0.012
    )


def test_gpu_suite_inputs_accept_modeled_scenario_coin_hsl_overrides():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    overrides = {
        "coin_overrides": {
            "ETH": {"bot": {"long": {"hsl": {"enabled": True}}}}
        }
    }
    ctx = SimpleNamespace(
        label="unsupported_coin_risk",
        overrides=overrides,
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "ETH": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), [0, 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["coin_overrides"] = copy.deepcopy(overrides["coin_overrides"])
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["config"]["coin_overrides"] == overrides["coin_overrides"]


def test_gpu_suite_inputs_reject_invalid_scenario_coin_hsl_values():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    overrides = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "hsl": {
                            "tier_ratios": {"yellow": 0.9, "orange": 0.2}
                        }
                    }
                }
            }
        }
    }
    ctx = SimpleNamespace(
        label="invalid_coin_hsl",
        overrides=overrides,
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "ETH": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), [0, 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["coin_overrides"] = copy.deepcopy(
                overrides["coin_overrides"]
            )
            return scenario

    with pytest.raises(ValueError, match="tier_ratios must satisfy"):
        _gpu_suite_scenario_inputs(config, Suite())


def test_gpu_suite_inputs_accept_scenario_one_way_mode():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["live"]["hedge_mode"] = True
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH"],
        "short": ["BTC", "ETH"],
    }
    ctx = SimpleNamespace(
        label="one_way_multi",
        overrides={"live.hedge_mode": False},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "ETH": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), [0, 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["live"]["hedge_mode"] = False
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["config"]["live"]["hedge_mode"] is False


def _suite_search_input(label, config, coin_count):
    return {
        "ctx": SimpleNamespace(label=label),
        "config": config,
        "coin_count": coin_count,
    }


def test_gpu_suite_search_context_reports_coin_count_range():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)

    assert _gpu_suite_search_context(
        [
            _suite_search_input("broad", config, 4),
            _suite_search_input("narrow", config, 2),
        ]
    ) == (2, 4, ("long",))


def test_gpu_suite_search_context_allows_legacy_single_coin_topologies():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)

    assert _gpu_suite_search_context(
        [
            _suite_search_input("base", config, 1),
            _suite_search_input("stress", config, 1),
        ]
    ) == (1, 1, None)


def test_gpu_suite_search_context_accepts_single_side_multicoin_trailing_martingale():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)

    assert _gpu_suite_search_context(
        [_suite_search_input("multi", config, 2)]
    ) == (2, 2, ("long",))


def test_gpu_suite_search_context_accepts_dual_side_multicoin_trailing_martingale():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    config["live"]["hedge_mode"] = True

    assert _gpu_suite_search_context(
        [_suite_search_input("multi", config, 2)]
    ) == (2, 2, ("long", "short"))


def test_gpu_suite_search_context_rejects_mixed_multicoin_strategy_kinds():
    ema = _directional_ema_config(long_enabled=True, short_enabled=False)
    tm = _directional_tm_config(long_enabled=True, short_enabled=False)

    with pytest.raises(ValueError, match="one supported strategy kind"):
        _gpu_suite_search_context(
            [
                _suite_search_input("ema", ema, 2),
                _suite_search_input("tm", tm, 2),
            ]
        )


def test_gpu_suite_search_context_accepts_multicoin_dual_side_scenarios():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["hedge_mode"] = True

    assert _gpu_suite_search_context(
        [
            _suite_search_input("broad", config, 4),
            _suite_search_input("narrow", config, 2),
        ]
    ) == (2, 4, ("long", "short"))


def test_gpu_suite_search_context_rejects_different_side_topologies():
    long_config = _directional_ema_config(long_enabled=True, short_enabled=False)
    short_config = _directional_ema_config(long_enabled=False, short_enabled=True)

    with pytest.raises(ValueError, match="same enabled-side topology"):
        _gpu_suite_search_context(
            [
                _suite_search_input("long", long_config, 3),
                _suite_search_input("short", short_config, 2),
            ]
        )


def test_gpu_suite_search_context_rejects_single_vs_dual_side_topology():
    long_config = _directional_ema_config(long_enabled=True, short_enabled=False)
    dual_config = _directional_ema_config(long_enabled=True, short_enabled=True)
    dual_config["live"]["hedge_mode"] = True

    with pytest.raises(ValueError, match="same enabled-side topology"):
        _gpu_suite_search_context(
            [
                _suite_search_input("long", long_config, 3),
                _suite_search_input("dual", dual_config, 3),
            ]
        )


def test_gpu_suite_scenario_overrides_shadow_candidate_parameters_last():
    config = _long_only_ema_config()
    scenario = copy.deepcopy(config)
    scenario["bot"]["long"]["strategy"]["ema_anchor"]["base_qty_pct"] = 0.025
    scenario["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.75

    fixed_bounds, fixed_parameters = _gpu_suite_scenario_override_context(
        config,
        scenario,
        {
            "bot.long.strategy.ema_anchor.base_qty_pct": 0.025,
            "bot.long.total_wallet_exposure_limit": 0.75,
        },
        {
            "long_base_qty_pct",
            "long_total_wallet_exposure_limit",
            "long_n_positions",
        },
        {
            "long_base_qty_pct": "long_base_qty_pct",
            "long_total_wallet_exposure_limit": "long_total_wallet_exposure_limit",
        },
    )

    assert fixed_bounds == {
        "long_base_qty_pct": 0.025,
        "long_total_wallet_exposure_limit": 0.75,
    }
    assert fixed_parameters == {
        "long_base_qty_pct": 0.025,
        "long_total_wallet_exposure_limit": 0.75,
    }


def test_gpu_suite_inputs_accept_and_preserve_bot_override_scope():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    overrides = {"bot.long.strategy.ema_anchor.base_qty_pct": 0.025}
    ctx = SimpleNamespace(
        label="fixed_qty",
        overrides=overrides,
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["bot"]["long"]["strategy"]["ema_anchor"]["base_qty_pct"] = 0.025
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["overrides"] == overrides
    assert (
        prepared[0]["config"]["bot"]["long"]["strategy"]["ema_anchor"][
            "base_qty_pct"
        ]
        == 0.025
    )


def test_gpu_suite_inputs_accept_ema_single_coin_total_exposure_repair_override():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    ctx = SimpleNamespace(
        label="ema_total_repair",
        overrides={"bot.long.risk.total_exposure_enforcer_enabled": True},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["bot"]["long"]["risk"][
                "total_exposure_enforcer_enabled"
            ] = True
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["config"]["bot"]["long"]["risk"][
        "total_exposure_enforcer_enabled"
    ]


def test_gpu_suite_inputs_accept_tm_total_exposure_repair_override():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["backtest"]["suite_enabled"] = True
    overrides = {
        "bot.long.risk.total_exposure_enforcer_enabled": True,
        "bot.long.risk.total_exposure_enforcer_policy": "reduce_portfolio",
        "bot.long.risk.total_exposure_enforcer_threshold": 0.8,
    }
    ctx = SimpleNamespace(
        label="portfolio_repair",
        overrides=overrides,
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            scenario = copy.deepcopy(proxy_config)
            scenario["bot"]["long"]["risk"].update(
                {
                    "total_exposure_enforcer_enabled": True,
                    "total_exposure_enforcer_policy": "reduce_portfolio",
                    "total_exposure_enforcer_threshold": 0.8,
                }
            )
            return scenario

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["overrides"] == overrides
    assert prepared[0]["config"]["bot"]["long"]["risk"] == {
        **config["bot"]["long"]["risk"],
        "total_exposure_enforcer_enabled": True,
        "total_exposure_enforcer_policy": "reduce_portfolio",
        "total_exposure_enforcer_threshold": 0.8,
    }


def test_gpu_suite_inputs_accept_combined_dataset_and_coin_sources():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    config["backtest"]["coin_sources"] = {"BTC": "bybit", "ETH": "binance"}
    ctx = SimpleNamespace(
        label="mixed_sources",
        config={
            "backtest": {"coin_sources": {"BTC": "bybit", "ETH": "binance"}}
        },
        overrides={},
        exchanges=["combined"],
        msss={
            "combined": {
                "BTC": {"exchange": "bybit", "ohlcv_source": "bybit"},
                "ETH": {"exchange": "binance", "ohlcv_source": "binance"},
                "__meta__": {},
            }
        },
        timestamps={"combined": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 2, 4)), np.ones(10), [0, 1]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["exchange"] == "combined"
    assert prepared[0]["coins"] == ["BTC", "ETH"]
    assert prepared[0]["config"]["backtest"]["coin_sources"] == {
        "BTC": "bybit",
        "ETH": "binance",
    }


@pytest.mark.parametrize("source", ["binance", "binanceusdm"])
def test_gpu_suite_inputs_accept_matching_individual_dataset_coin_source(source):
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    ctx = SimpleNamespace(
        label="binance_only",
        config={"backtest": {"coin_sources": {"BTC": source}}},
        overrides={},
        exchanges=["binance"],
        msss={"binance": {"BTC": {}, "__meta__": {}}},
        timestamps={"binance": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["exchange"] == "binance"


def test_gpu_suite_inputs_ignore_conflicting_source_for_excluded_coin():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    ctx = SimpleNamespace(
        label="bybit_btc_only",
        config={
            "backtest": {
                "coin_sources": {"BTC": "bybit", "ETH": "binance"}
            }
        },
        overrides={},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert prepared[0]["coins"] == ["BTC"]


def test_gpu_suite_inputs_reject_coin_source_outside_individual_dataset():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    ctx = SimpleNamespace(
        label="bybit_only",
        config={"backtest": {"coin_sources": {"BTC": "binance"}}},
        overrides={},
        exchanges=["bybit"],
        msss={"bybit": {"BTC": {}, "__meta__": {}}},
        timestamps={"bybit": np.arange(10, dtype=np.int64)},
    )

    class Suite:
        contexts = [ctx]

        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            raise AssertionError("conflicting source must fail before data access")

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    with pytest.raises(ValueError, match="outside prepared dataset 'bybit'"):
        _gpu_suite_scenario_inputs(config, Suite())


def test_gpu_suite_inputs_accept_one_exchange_per_scenario():
    config = _long_only_ema_config()
    config["backtest"]["suite_enabled"] = True
    contexts = [
        SimpleNamespace(
            label=exchange,
            overrides={},
            exchanges=[exchange],
            msss={exchange: {"BTC": {}, "__meta__": {}}},
            timestamps={exchange: np.arange(10, dtype=np.int64)},
        )
        for exchange in ("bybit", "binance")
    ]

    class Suite:
        @staticmethod
        def get_prepared_context_data(_ctx, _exchange):
            return np.zeros((10, 1, 4)), np.ones(10), [0]

        @staticmethod
        def build_scenario_candidate_config(proxy_config, _ctx):
            return copy.deepcopy(proxy_config)

    Suite.contexts = contexts

    prepared = _gpu_suite_scenario_inputs(config, Suite())

    assert [item["exchange"] for item in prepared] == ["bybit", "binance"]


def test_gpu_suite_proxy_rows_use_canonical_suite_scorer():
    contexts = [SimpleNamespace(label="base"), SimpleNamespace(label="stress")]

    class Proxy:
        def __init__(self, values):
            self.values = values

        def evaluate(self, candidates):
            assert len(candidates) == 2
            return [{"adg_strategy_eq": value} for value in self.values]

    class Suite:
        @staticmethod
        def score_scenario_results(results):
            values = [
                result.metrics["stats"]["adg_strategy_eq"]["mean"]
                for result in results
            ]
            return {
                "objectives": (-min(values),),
                "constraint_violation": max(values),
                "suite_metrics": {"values": values},
            }

    rows = _evaluate_gpu_suite_proxies(
        Suite(),
        [
            (contexts[0], (("bybit", Proxy([0.10, 0.20])),), {}),
            (contexts[1], (("bybit", Proxy([0.01, 0.02])),), {}),
        ],
        [{"x": 1}, {"x": 2}],
    )

    assert rows == [
        {
            _GPU_SUITE_OBJECTIVES_KEY: (-0.01,),
            _GPU_SUITE_VIOLATION_KEY: 0.10,
            _GPU_SUITE_METRICS_KEY: {"values": [0.10, 0.01]},
        },
        {
            _GPU_SUITE_OBJECTIVES_KEY: (-0.02,),
            _GPU_SUITE_VIOLATION_KEY: 0.20,
            _GPU_SUITE_METRICS_KEY: {"values": [0.20, 0.02]},
        },
    ]


def test_gpu_suite_proxy_combines_exchange_metrics_before_suite_reduction():
    context = SimpleNamespace(label="regional_subset")

    class Proxy:
        def __init__(self, values):
            self.values = values

        def evaluate(self, candidates):
            assert len(candidates) == 2
            return [{"adg_strategy_eq": value} for value in self.values]

    class Suite:
        @staticmethod
        def score_scenario_results(results):
            assert len(results) == 1
            result = results[0]
            stats = result.metrics["stats"]["adg_strategy_eq"]
            assert sorted(result.per_exchange) == ["binance", "bybit"]
            return {
                "objectives": (-stats["mean"],),
                "constraint_violation": stats["max"],
                "suite_metrics": {"stats": stats},
            }

    rows = _evaluate_gpu_suite_proxies(
        Suite(),
        [
            (
                context,
                (
                    ("bybit", Proxy([0.10, 0.40])),
                    ("binance", Proxy([0.30, 0.20])),
                ),
                {},
            )
        ],
        [{"x": 1}, {"x": 2}],
    )

    assert rows[0][_GPU_SUITE_OBJECTIVES_KEY] == pytest.approx((-0.20,))
    assert rows[0][_GPU_SUITE_VIOLATION_KEY] == pytest.approx(0.30)
    assert rows[0][_GPU_SUITE_METRICS_KEY]["stats"] == pytest.approx(
        {
            "mean": 0.20,
            "min": 0.10,
            "max": 0.30,
            "std": 0.10,
            "median": 0.20,
        }
    )
    assert rows[1][_GPU_SUITE_OBJECTIVES_KEY] == pytest.approx((-0.30,))
    assert rows[1][_GPU_SUITE_VIOLATION_KEY] == pytest.approx(0.40)


def test_gpu_suite_proxy_applies_scenario_parameter_overrides_without_mutating_candidates():
    seen = []

    class Proxy:
        def evaluate(self, candidates):
            seen.append(copy.deepcopy(candidates))
            return [
                {"adg_strategy_eq": candidate["long_base_qty_pct"]}
                for candidate in candidates
            ]

    class Suite:
        @staticmethod
        def score_scenario_results(results):
            return {
                "objectives": (0.0,),
                "constraint_violation": 0.0,
                "suite_metrics": {},
            }

    candidates = [{"long_base_qty_pct": 0.01}]
    _evaluate_gpu_suite_proxies(
        Suite(),
        [
            (
                SimpleNamespace(label="fixed"),
                (("bybit", Proxy()),),
                {"long_base_qty_pct": 0.025},
            ),
            (SimpleNamespace(label="candidate"), (("bybit", Proxy()),), {}),
        ],
        candidates,
    )

    assert seen == [
        [{"long_base_qty_pct": 0.025}],
        [{"long_base_qty_pct": 0.01}],
    ]
    assert candidates == [{"long_base_qty_pct": 0.01}]


def test_suite_limit_metric_value_respects_reducer_and_scenario():
    payload = {
        "metrics": {
            "drawdown_worst_strategy_eq": {
                "stats": {"mean": 0.2, "max": 0.4},
                "scenarios": {"base": 0.1, "stress": 0.4},
            }
        }
    }

    assert _suite_limit_metric_value(
        payload,
        {
            "metric": "drawdown_worst_strategy_eq",
            "reducer": "max",
            "scenario": None,
        },
    ) == 0.4
    assert _suite_limit_metric_value(
        payload,
        {
            "metric": "drawdown_worst_strategy_eq",
            "reducer": "mean",
            "scenario": "stress",
        },
    ) == 0.4


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["backtest"].__setitem__("suite_enabled", True), "suite"),
        (
            lambda config: config["live"].__setitem__(
                "strategy_kind", "trailing_grid_v7"
            ),
            "trailing_martingale",
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
            lambda config: config["bot"]["long"]["risk"].__setitem__(
                "position_exposure_enforcer_enabled", True
            ),
            "position_exposure_enforcer_enabled",
        ),
    ],
)
def test_gpu_foundation_fails_closed_for_unsupported_scope(mutate, message):
    config = _long_only_ema_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        _validate_scope(config, _Evaluator())


def _fake_torch_with_mps(available=True):
    return SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: available),
        )
    )


def test_gpu_preparation_preflight_rejects_btc_collateral_before_runtime_probe():
    config = _long_only_ema_config()
    config["backtest"]["btc_collateral_cap"] = 0.5
    runtime = MagicMock()

    with pytest.raises(
        ValueError,
        match=r"btc_collateral_cap=0\.0.*pymoo.*exact Rust validation",
    ):
        validate_gpu_preparation_scope(config, torch_module=runtime)

    runtime.backends.mps.is_available.assert_not_called()


def test_gpu_preparation_preflight_rejects_exact_only_metric_before_runtime_probe():
    config = _long_only_ema_config()
    config["_raw_effective"] = {
        "optimize": {
            "scoring": [
                {
                    "goal": "min",
                    "metric": "peak_recovery_days_strategy_eq",
                }
            ]
        }
    }
    runtime = MagicMock()

    with pytest.raises(ValueError, match="exact Rust backtests and analysis"):
        validate_gpu_preparation_scope(config, torch_module=runtime)

    runtime.backends.mps.is_available.assert_not_called()


def test_gpu_preparation_preflight_explains_trailing_grid_cpu_fallback():
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_grid_v7"

    with pytest.raises(
        ValueError,
        match=r"trailing_grid_v7 is deliberately outside.*pymoo",
    ):
        validate_gpu_preparation_scope(
            config,
            torch_module=_fake_torch_with_mps(),
        )


def test_gpu_preparation_preflight_rejects_halving_for_ema_or_suite():
    config = _long_only_ema_config()
    config["optimize"]["gpu"]["successive_halving"]["enabled"] = True

    with pytest.raises(ValueError, match="single-coin trailing_martingale"):
        validate_gpu_preparation_scope(
            config,
            torch_module=_fake_torch_with_mps(),
        )

    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["optimize"]["gpu"]["successive_halving"]["enabled"] = True
    with pytest.raises(ValueError, match="non-suite"):
        validate_gpu_preparation_scope(
            config,
            {"enabled": True, "scenarios": []},
            torch_module=_fake_torch_with_mps(),
        )


def test_gpu_preparation_preflight_rejects_unmodeled_suite_override_early():
    config = _long_only_ema_config()
    suite_cfg = {
        "enabled": True,
        "scenarios": [
            {
                "label": "unsupported_collateral",
                "overrides": {"backtest.btc_collateral_cap": 0.5},
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"unsupported_collateral.*btc_collateral_cap.*outside the supported modeled",
    ):
        validate_gpu_preparation_scope(
            config,
            suite_cfg,
            torch_module=_fake_torch_with_mps(),
        )


def test_gpu_preparation_preflight_validates_effective_bot_suite_values():
    config = _long_only_ema_config()
    runtime = MagicMock()
    suite_cfg = {
        "enabled": True,
        "scenarios": [
            {
                "label": "unsupported_ema_repair",
                "overrides": {
                    "bot.long.risk.position_exposure_enforcer_enabled": True,
                },
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match=(
            r"bot\.long\.risk\.position_exposure_enforcer_enabled=false"
        ),
    ):
        validate_gpu_preparation_scope(
            config,
            suite_cfg,
            torch_module=runtime,
        )

    runtime.backends.mps.is_available.assert_not_called()


def test_gpu_preparation_preflight_accepts_modeled_tm_exposure_repair_override():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    suite_cfg = {
        "enabled": True,
        "scenarios": [
            {
                "label": "tm_repair",
                "overrides": {
                    "bot.long.risk.position_exposure_enforcer_enabled": True,
                },
            }
        ],
    }

    validate_gpu_preparation_scope(
        config,
        suite_cfg,
        torch_module=_fake_torch_with_mps(),
    )


def test_gpu_preparation_preflight_requires_available_mps():
    with pytest.raises(RuntimeError, match=r"MPS is unavailable.*pymoo"):
        validate_gpu_preparation_scope(
            _long_only_ema_config(),
            torch_module=_fake_torch_with_mps(False),
        )


def test_gpu_preparation_preflight_logs_capability_contract(caplog):
    with caplog.at_level("INFO"):
        validate_gpu_preparation_scope(
            _long_only_ema_config(),
            torch_module=_fake_torch_with_mps(),
        )

    assert (
        "GPU capability preflight passed | runtime=apple_mps | "
        "strategy=ema_anchor | btc_collateral_cap=0 | max_coins_per_scenario=64"
        in caplog.text
    )


def test_gpu_foundation_accepts_ema_long_single():
    assert _validate_scope(_long_only_ema_config(), _Evaluator()) == "bybit"


def test_gpu_foundation_accepts_baseline_ema_single_coin_market_execution():
    config = _long_only_ema_config()
    config["live"]["market_orders_allowed"] = True
    config["live"]["market_order_near_touch_threshold"] = 0.002

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_foundation_accepts_baseline_tm_single_coin_market_execution():
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    config["live"]["market_order_near_touch_threshold"] = 0.002

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    ("key", "low", "high"),
    [
        ("long_close_retracement_base_pct", 0.0, 0.1),
        ("long_close_retracement_base_pct", 1.0e-50, 0.1),
        ("long_entry_retracement_base_pct", 0.0, 0.1),
        ("long_entry_retracement_base_pct", -0.1, 0.1),
        ("long_entry_retracement_base_pct", 1.0e-50, 0.1),
    ],
)
def test_gpu_tm_market_execution_accepts_entry_or_close_mode_crossing(
    key, low, high
):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        "long_entry_retracement_base_pct": Bound(0.001, 0.1),
        "long_close_retracement_base_pct": Bound(0.001, 0.1),
    }
    bounds[key] = Bound(low, high)

    _validate_tm_market_mode_bounds(bounds, {}, {"long"}, config)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.0e39])
def test_gpu_tm_market_execution_rejects_unrepresentable_mode_bounds(value):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        "long_entry_retracement_base_pct": Bound(value, value),
        "long_close_retracement_base_pct": Bound(0.001, 0.1),
    }

    with pytest.raises(ValueError, match="finite float32-representable"):
        _validate_tm_market_mode_bounds(bounds, {}, {"long"}, config)


def test_gpu_tm_market_execution_accepts_trailing_only_mode_bounds():
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        f"{side}_{phase}_retracement_base_pct": Bound(0.001, 0.1)
        for side in ("long", "short")
        for phase in ("entry", "close")
    }

    _validate_tm_market_mode_bounds(bounds, {}, {"long"}, config)


@pytest.mark.parametrize("entry_bounds", [Bound(0.0, 0.0), Bound(-0.1, 0.0)])
def test_gpu_tm_market_execution_accepts_recursive_entry_mode_bounds(
    entry_bounds,
):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        "long_entry_retracement_base_pct": entry_bounds,
        "long_close_retracement_base_pct": Bound(0.001, 0.1),
    }

    _validate_tm_market_mode_bounds(bounds, {}, {"long"}, config)


@pytest.mark.parametrize("close_bounds", [Bound(0.0, 0.0), Bound(-0.1, 0.0)])
def test_gpu_tm_market_execution_accepts_recursive_close_mode_bounds(
    close_bounds,
):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        "long_entry_retracement_base_pct": Bound(0.001, 0.1),
        "long_close_retracement_base_pct": close_bounds,
    }

    _validate_tm_market_mode_bounds(bounds, {}, {"long"}, config)


def test_gpu_multicoin_tm_market_execution_accepts_recursive_entry_mode_bounds():
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        "long_entry_retracement_base_pct": Bound(0.001, 0.1),
        "long_close_retracement_base_pct": Bound(0.001, 0.1),
    }
    bounds["long_entry_retracement_base_pct"] = Bound(0.0, 0.0)

    _validate_tm_market_mode_bounds(
        bounds, {}, {"long"}, config, coin_count=3
    )


@pytest.mark.parametrize("close_bounds", [Bound(0.0, 0.0), Bound(-0.1, 0.0)])
def test_gpu_multicoin_tm_market_execution_accepts_recursive_close_bounds(
    close_bounds,
):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    bounds = {
        "long_entry_retracement_base_pct": Bound(0.001, 0.1),
        "long_close_retracement_base_pct": close_bounds,
    }

    _validate_tm_market_mode_bounds(
        bounds, {}, {"long"}, config, coin_count=3
    )


def test_gpu_multicoin_tm_market_execution_accepts_unstuck_config():
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    config["bot"]["long"]["unstuck"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "suffix",
    [
        "unstuck_enabled",
        "risk_position_exposure_enforcer_enabled",
        "risk_total_exposure_enforcer_enabled",
    ],
)
def test_gpu_tm_market_execution_accepts_risk_ordering_bounds(side, suffix):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    key = f"{side}_{suffix}"
    bounds = {
        f"{side}_entry_retracement_base_pct": Bound(0.001, 0.1),
        f"{side}_close_retracement_base_pct": Bound(0.001, 0.1),
        key: Bound(0.0, 1.0),
    }

    _validate_tm_market_template_bounds(bounds, {}, {side}, config, [])


def test_gpu_tm_market_suite_validates_effective_scenarios_not_template():
    bounds = {
        "long_entry_retracement_base_pct": Bound(0.0, 0.1),
        "long_close_retracement_base_pct": Bound(0.001, 0.1),
        "long_unstuck_enabled": Bound(0.0, 1.0),
    }
    template = _long_only_ema_config()
    template["live"]["strategy_kind"] = "trailing_martingale"
    template["live"]["market_orders_allowed"] = True

    _validate_tm_market_template_bounds(
        bounds,
        {},
        {"long"},
        template,
        [{"config": {"effective": True}}],
    )

    scenario = copy.deepcopy(template)
    scenario["live"]["market_orders_allowed"] = False
    _validate_tm_market_mode_bounds(bounds, {}, {"long"}, scenario)

    _validate_tm_market_template_bounds(bounds, {}, {"long"}, template, [])


@pytest.mark.parametrize(
    ("mutate", "evaluator", "message"),
    [
        (
            lambda config: config["live"].__setitem__(
                "market_order_near_touch_threshold", -0.1
            ),
            _Evaluator,
            "finite non-negative",
        ),
    ],
)
def test_gpu_market_execution_fails_closed_outside_baseline_scope(
    mutate, evaluator, message
):
    config = _long_only_ema_config()
    config["live"]["market_orders_allowed"] = True
    mutate(config)

    with pytest.raises(ValueError, match=message):
        _validate_scope(config, evaluator())


def test_gpu_multicoin_tm_market_execution_accepts_trailing_baseline():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH"]
    config["live"]["market_orders_allowed"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["live"].__setitem__(
            "max_realized_loss_pct", 0.5
        ),
        lambda config: config["bot"]["long"]["unstuck"].__setitem__(
            "enabled", True
        ),
        lambda config: config["bot"]["long"]["risk"].__setitem__(
            "position_exposure_enforcer_enabled", True
        ),
        lambda config: config["bot"]["long"]["risk"].__setitem__(
            "total_exposure_enforcer_enabled", True
        ),
    ],
)
def test_gpu_tm_market_execution_accepts_risk_ordering(mutate):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["market_orders_allowed"] = True
    mutate(config)

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("risk_feature", ["loss_gate", "unstuck", "twel_enforcer"])
def test_gpu_market_execution_accepts_single_coin_ema_risk_ordering(risk_feature):
    config = _long_only_ema_config()
    config["live"]["market_orders_allowed"] = True
    if risk_feature == "loss_gate":
        config["live"]["max_realized_loss_pct"] = 0.5
    elif risk_feature == "unstuck":
        config["bot"]["long"]["unstuck"]["enabled"] = True
    else:
        config["bot"]["long"]["risk"]["total_exposure_enforcer_enabled"] = True

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("signal_mode", ["unified", "pside", "coin"])
def test_gpu_market_execution_accepts_single_coin_hsl(
    strategy_kind, signal_mode
):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["market_orders_allowed"] = True
    config["live"]["hsl_signal_mode"] = signal_mode
    config["live"]["pnls_max_lookback_days"] = "all"
    config["bot"]["long"]["hsl"]["enabled"] = True

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_gpu_market_execution_accepts_hsl_with_min_effective_cost_filter(
    strategy_kind,
):
    config = _long_only_ema_config()
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["market_orders_allowed"] = True
    config["live"]["hsl_signal_mode"] = "coin"
    config["live"]["pnls_max_lookback_days"] = "all"
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["backtest"]["filter_by_min_effective_cost"] = True

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_market_execution_accepts_dual_side_min_effective_cost_filter():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["market_orders_allowed"] = True
    config["backtest"]["filter_by_min_effective_cost"] = True

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_foundation_accepts_one_sided_single_coin_hsl():
    config = _long_only_ema_config()
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["live"]["pnls_max_lookback_days"] = "all"

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("lookback", [0.0, 30.0, "all"])
def test_gpu_hsl_accepts_conservative_pnl_lookback_envelope(lookback):
    config = _long_only_ema_config()
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["live"]["pnls_max_lookback_days"] = lookback

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_hsl_accepts_market_panic_close():
    config = _long_only_ema_config()
    config["bot"]["long"]["hsl"].update(
        {"enabled": True, "panic_close_order_type": "market"}
    )
    config["live"]["pnls_max_lookback_days"] = "all"

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_hsl_fails_closed_for_unknown_panic_close_type():
    config = _long_only_ema_config()
    config["bot"]["long"]["hsl"].update(
        {"enabled": True, "panic_close_order_type": "immediate_or_cancel"}
    )
    config["live"]["pnls_max_lookback_days"] = "all"

    with pytest.raises(ValueError, match="to be limit or market"):
        _validate_scope(config, _Evaluator())


@pytest.mark.parametrize("signal_mode", ["unified", "coin", "pside"])
def test_gpu_hsl_accepts_dual_side_single_coin_signal_modes(signal_mode):
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["bot"]["short"]["hsl"]["enabled"] = True
    config["live"]["hsl_signal_mode"] = signal_mode

    assert _validate_scope(config, _Evaluator()) == "bybit"
@pytest.mark.parametrize("signal_mode", ["unified", "pside", "coin"])
def test_gpu_hsl_accepts_one_sided_multicoin_signal_modes(signal_mode):
    config = _long_only_ema_config()
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "XRP"]
    config["live"]["hsl_signal_mode"] = signal_mode
    config["bot"]["long"]["risk"]["n_positions"] = 3
    config["bot"]["long"]["hsl"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_hsl_accepts_one_sided_multicoin_market_panic():
    config = _long_only_ema_config()
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "XRP"]
    config["live"]["hsl_signal_mode"] = "unified"
    config["bot"]["long"]["risk"]["n_positions"] = 3
    config["bot"]["long"]["hsl"].update(
        {"enabled": True, "panic_close_order_type": "market"}
    )

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_hsl_accepts_dual_side_multicoin_pside_mode():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "XRP"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["hsl_signal_mode"] = "pside"
    config["live"]["hedge_mode"] = True
    for side in ("long", "short"):
        config["bot"][side]["risk"]["n_positions"] = 3
        config["bot"][side]["hsl"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("signal_mode", ["coin", "unified"])
def test_gpu_hsl_accepts_dual_side_multicoin_fused_ema_modes(signal_mode):
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "XRP"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["hsl_signal_mode"] = signal_mode
    config["live"]["hedge_mode"] = True
    for side in ("long", "short"):
        config["bot"][side]["risk"]["n_positions"] = 3
        config["bot"][side]["hsl"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("signal_mode", ["coin", "unified"])
def test_gpu_hsl_accepts_dual_side_multicoin_tm_joint_modes(signal_mode):
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "XRP"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["hsl_signal_mode"] = signal_mode
    config["live"]["hedge_mode"] = True
    for side in ("long", "short"):
        config["bot"][side]["risk"]["n_positions"] = 3
        config["bot"][side]["hsl"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_hsl_metrics_reject_only_dual_multicoin_tier_overlap():
    _validate_hsl_metric_topology(
        {"hard_stop_panic_close_loss_sum"},
        coin_count=3,
        enabled_sides=["long"],
        hard_stop_metrics={"hard_stop_panic_close_loss_sum"},
    )

    _validate_hsl_metric_topology(
        {"hard_stop_panic_close_loss_sum", "hard_stop_triggers"},
        coin_count=3,
        enabled_sides=["long", "short"],
        hard_stop_metrics={
            "hard_stop_panic_close_loss_sum",
            "hard_stop_triggers",
        },
    )

    with pytest.raises(ValueError, match="cross-side tier overlap"):
        _validate_hsl_metric_topology(
            {"hard_stop_time_in_red_pct"},
            coin_count=3,
            enabled_sides=["long", "short"],
            hard_stop_metrics={"hard_stop_time_in_red_pct"},
        )

    with pytest.raises(ValueError, match="shared event-level account equity"):
        _validate_hsl_metric_topology(
            {"hard_stop_panic_close_loss_drawdown_pct_mean"},
            coin_count=3,
            enabled_sides=["long", "short"],
            hard_stop_metrics={
                "hard_stop_panic_close_loss_drawdown_pct_mean"
            },
        )

    _validate_hsl_metric_topology(
        {
            "hard_stop_time_in_red_pct",
            "hard_stop_panic_close_loss_drawdown_pct_mean",
        },
        coin_count=3,
        enabled_sides=["long", "short"],
        hard_stop_metrics={
            "hard_stop_time_in_red_pct",
            "hard_stop_panic_close_loss_drawdown_pct_mean",
        },
        shared_account_controller=True,
    )

    _validate_hsl_metric_topology(
        {"hard_stop_panic_close_loss_sum"},
        coin_count=1,
        enabled_sides=["long", "short"],
        hard_stop_metrics={"hard_stop_panic_close_loss_sum"},
    )


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("max_loss_pct", [0.0, 0.1, 0.999, 1.0, 2.0])
def test_gpu_foundation_accepts_single_coin_realized_loss_gate(
    side, strategy_kind, max_loss_pct
):
    builder = (
        _directional_ema_config
        if strategy_kind == "ema_anchor"
        else _directional_tm_config
    )
    config = builder(
        long_enabled=side == "long",
        short_enabled=side == "short",
    )
    config["live"]["max_realized_loss_pct"] = max_loss_pct

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("max_loss_pct", [-0.1, float("nan"), float("inf")])
def test_gpu_foundation_rejects_invalid_realized_loss_gate(max_loss_pct):
    config = _long_only_ema_config()
    config["live"]["max_realized_loss_pct"] = max_loss_pct

    with pytest.raises(ValueError, match="finite non-negative"):
        _validate_scope(config, _Evaluator())


def test_gpu_ema_realized_loss_gate_accepts_multicoin():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["live"]["max_realized_loss_pct"] = 0.1

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_tm_realized_loss_gate_accepts_multicoin():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["max_realized_loss_pct"] = 0.1

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("entry_gate", [False, True])
@pytest.mark.parametrize("allowance_mode", ["bounded", "legacy_raw"])
def test_gpu_foundation_accepts_single_coin_exposure_headroom_policy(
    strategy_kind, side, entry_gate, allowance_mode
):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=side == "long", short_enabled=side == "short")
    risk = config["bot"][side]["risk"]
    risk["we_excess_allowance_pct"] = 0.25
    risk["we_excess_allowance_mode"] = allowance_mode
    risk["total_exposure_entry_gate_enabled"] = entry_gate
    risk["total_exposure_enforcer_threshold"] = 0.8

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("suite_enabled", [False, True])
def test_gpu_foundation_accepts_tm_position_exposure_repair(side, suite_enabled):
    config = _directional_tm_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    config["bot"][side]["risk"]["position_exposure_enforcer_enabled"] = True
    config["bot"][side]["risk"]["position_exposure_enforcer_threshold"] = 0.8
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(config, _Evaluator(), allow_suite=suite_enabled) == "bybit"
    )


def test_gpu_foundation_keeps_ema_position_exposure_repair_fail_closed():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["bot"]["long"]["risk"]["position_exposure_enforcer_enabled"] = True
    config["bot"]["long"]["risk"]["position_exposure_enforcer_threshold"] = 0.8

    with pytest.raises(ValueError, match="position_exposure_enforcer_enabled"):
        _validate_scope(config, _Evaluator())


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "policy", ["reduce_overweight", "reduce_portfolio"]
)
@pytest.mark.parametrize("suite_enabled", [False, True])
def test_gpu_foundation_accepts_tm_total_exposure_repair(
    side, policy, suite_enabled
):
    config = _directional_tm_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    risk = config["bot"][side]["risk"]
    risk["total_exposure_enforcer_enabled"] = True
    risk["total_exposure_enforcer_policy"] = policy
    risk["total_exposure_enforcer_threshold"] = 0.8
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(config, _Evaluator(), allow_suite=suite_enabled)
        == "bybit"
    )


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "policy", ["reduce_overweight", "reduce_portfolio"]
)
def test_gpu_multicoin_accepts_tm_total_exposure_repair(side, policy):
    config = _directional_tm_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    config["live"]["approved_coins"][side] = ["BTC", "ETH", "SOL"]
    risk = config["bot"][side]["risk"]
    risk["n_positions"] = 2
    risk["total_exposure_enforcer_enabled"] = True
    risk["total_exposure_enforcer_policy"] = policy
    risk["total_exposure_enforcer_threshold"] = 0.8
    config["backtest"]["dynamic_wel_by_tradability"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    "repair_key",
    [
        "position_exposure_enforcer_enabled",
        "total_exposure_enforcer_enabled",
    ],
)
def test_gpu_dual_multicoin_accepts_tm_exposure_repair(repair_key):
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = True
    config["backtest"]["dynamic_wel_by_tradability"] = True
    for side in ("long", "short"):
        risk = config["bot"][side]["risk"]
        risk["n_positions"] = 2
        risk[repair_key] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_dual_multicoin_accepts_tm_coin_override_exposure_repair():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = True
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "risk": {"position_exposure_enforcer_enabled": True}
                }
            }
        }
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "policy", ["reduce_overweight", "reduce_portfolio"]
)
@pytest.mark.parametrize("suite_enabled", [False, True])
def test_gpu_foundation_accepts_ema_single_coin_total_exposure_repair(
    side, policy, suite_enabled
):
    config = _directional_ema_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    risk = config["bot"][side]["risk"]
    risk["total_exposure_enforcer_enabled"] = True
    risk["total_exposure_enforcer_policy"] = policy
    risk["total_exposure_enforcer_threshold"] = 0.8
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(config, _Evaluator(), allow_suite=suite_enabled)
        == "bybit"
    )


@pytest.mark.parametrize("hedge_mode", [False, True])
def test_gpu_foundation_accepts_ema_dual_single_coin_total_exposure_repair(
    hedge_mode,
):
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["hedge_mode"] = hedge_mode
    for side in ("long", "short"):
        risk = config["bot"][side]["risk"]
        risk["total_exposure_enforcer_enabled"] = True
        risk["total_exposure_enforcer_threshold"] = 0.8

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "policy", ["reduce_overweight", "reduce_portfolio"]
)
@pytest.mark.parametrize("suite_enabled", [False, True])
def test_gpu_multicoin_accepts_ema_total_exposure_repair(
    side, policy, suite_enabled
):
    config = _directional_ema_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"][side] = coins
    if suite_enabled:
        # Suite materialization requires symmetric coin universes even when
        # one side is disabled by a zero total-exposure budget.
        other = "short" if side == "long" else "long"
        config["live"]["approved_coins"][other] = coins
        config["bot"][other]["risk"]["total_wallet_exposure_limit"] = 0.0
    risk = config["bot"][side]["risk"]
    risk["n_positions"] = 2
    risk["total_exposure_enforcer_enabled"] = True
    risk["total_exposure_enforcer_policy"] = policy
    risk["total_exposure_enforcer_threshold"] = 0.8
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(
            config, _MulticoinEvaluator(), allow_suite=suite_enabled
        )
        == "bybit"
    )


def test_gpu_dual_multicoin_accepts_ema_total_exposure_repair():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = True
    config["backtest"]["dynamic_wel_by_tradability"] = True
    for side in ("long", "short"):
        risk = config["bot"][side]["risk"]
        risk["n_positions"] = 2
        risk["total_exposure_enforcer_enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("entry_gate", [False, True])
@pytest.mark.parametrize("allowance_mode", ["bounded", "legacy_raw"])
def test_gpu_multicoin_accepts_exposure_headroom_policy(
    strategy_kind, side, entry_gate, allowance_mode
):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=side == "long", short_enabled=side == "short")
    config["live"]["approved_coins"][side] = ["BTC", "ETH", "SOL"]
    risk = config["bot"][side]["risk"]
    risk["n_positions"] = 2
    risk["we_excess_allowance_pct"] = 0.25
    risk["we_excess_allowance_mode"] = allowance_mode
    risk["total_exposure_entry_gate_enabled"] = entry_gate
    risk["total_exposure_enforcer_threshold"] = 0.8
    config["backtest"]["dynamic_wel_by_tradability"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suite_enabled", [False, True])
@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_foundation_accepts_single_coin_min_effective_cost_filter(
    strategy_kind, suite_enabled, side
):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=side == "long", short_enabled=side == "short")
    config["backtest"]["filter_by_min_effective_cost"] = True
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(config, _Evaluator(), allow_suite=suite_enabled) == "bybit"
    )


def test_gpu_foundation_rejects_min_effective_cost_without_positive_liquidation_floor():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["backtest"]["filter_by_min_effective_cost"] = True
    config["backtest"]["liquidation_threshold"] = 0.0

    with pytest.raises(ValueError, match="proven lower balance bound"):
        _validate_scope(config, _Evaluator())


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suite_enabled", [False, True])
def test_gpu_foundation_accepts_dual_side_min_effective_cost_filter(
    strategy_kind, suite_enabled
):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=True, short_enabled=True)
    config["backtest"]["filter_by_min_effective_cost"] = True
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(config, _Evaluator(), allow_suite=suite_enabled)
        == "bybit"
    )


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suite_enabled", [False, True])
@pytest.mark.parametrize("dual_side", [False, True])
def test_gpu_foundation_accepts_multicoin_min_effective_cost_filter(
    strategy_kind, suite_enabled, dual_side
):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=True, short_enabled=dual_side)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"] if dual_side else [],
    }
    config["live"]["hedge_mode"] = True
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["backtest"]["filter_by_min_effective_cost"] = True
    config["backtest"]["suite_enabled"] = suite_enabled

    assert (
        _validate_scope(
            config,
            _MulticoinEvaluator(),
            allow_suite=suite_enabled,
        )
        == "bybit"
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_foundation_accepts_ema_single_side_multicoin(side):
    config = _directional_ema_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    config["live"]["approved_coins"][side] = ["BTC", "ETH", "SOL"]
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["bot"][side]["risk"]["n_positions"] = 2

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_multicoin_accepts_static_ema_coin_overrides(side):
    config = _directional_ema_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                side: {
                    "strategy": {"ema_anchor": {"offset": 0.02, "ema_span_0": 90}},
                    "risk": {
                        "entry_cooldown_minutes": 15,
                        "we_excess_allowance_pct": 0.25,
                    },
                    "wallet_exposure_limit": 0.4,
                    "unstuck": {
                        "enabled": True,
                        "ema_gating_enabled": False,
                        "close_pct": 0.1,
                        "ema_dist": 0.0,
                        "loss_allowance_pct": 0.02,
                        "threshold": 0.8,
                    },
                }
            }
        }
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def _nested_patch(path, value):
    result = value
    for key in reversed(tuple(path)):
        result = {key: result}
    return result


def _nested_value(value, path):
    for key in path:
        value = value[key]
    return value


@pytest.mark.parametrize(
    ("strategy_kind", "config_factory"),
    [
        ("ema_anchor", _directional_ema_config),
        ("trailing_martingale", _directional_tm_config),
    ],
)
def test_gpu_coin_override_policy_covers_cpu_backtest_effective_allowlist(
    strategy_kind, config_factory
):
    config = config_factory(long_enabled=True, short_enabled=False)
    config["live"]["hsl_signal_mode"] = "coin"
    side_config = config["bot"]["long"]
    exact_inapplicable = {
        "risk.position_exposure_enforcer_enabled",
        "risk.position_exposure_enforcer_threshold",
    }

    for dotted_path in sorted(OVERRIDABLE_SHARED_BOT_PATHS):
        if strategy_kind == "ema_anchor" and dotted_path in exact_inapplicable:
            continue
        path = tuple(dotted_path.split("."))
        config["coin_overrides"] = {
            "ETH": {
                "bot": {
                    "long": _nested_patch(path, _nested_value(side_config, path))
                }
            }
        }
        _validate_gpu_coin_overrides(
            config,
            strategy_kind=strategy_kind,
            enabled_sides=["long"],
            coin_count=3,
        )

    strategy = side_config["strategy"][strategy_kind]
    for dotted_path in get_strategy_param_keys(strategy_kind):
        path = tuple(dotted_path.split("."))
        config["coin_overrides"] = {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            strategy_kind: _nested_patch(
                                path, _nested_value(strategy, path)
                            )
                        }
                    }
                }
            }
        }
        _validate_gpu_coin_overrides(
            config,
            strategy_kind=strategy_kind,
            enabled_sides=["long"],
            coin_count=3,
        )

    for patch in (
        {"bot": {"long": {"wallet_exposure_limit": 0.4}}},
        {"live": {"forced_mode_long": "normal"}},
    ):
        config["coin_overrides"] = {"ETH": patch}
        _validate_gpu_coin_overrides(
            config,
            strategy_kind=strategy_kind,
            enabled_sides=["long"],
            coin_count=3,
        )


@pytest.mark.parametrize(
    ("strategy_kind", "config_factory"),
    [
        ("ema_anchor", _directional_ema_config),
        ("trailing_martingale", _directional_tm_config),
    ],
)
def test_gpu_coin_overrides_accept_cpu_compatible_live_only_values_with_warning(
    strategy_kind, config_factory, caplog
):
    config = config_factory(long_enabled=True, short_enabled=False)
    config["coin_overrides"] = {
        "ETH": {
            "live": {
                "forced_mode_long": "graceful_stop",
                "forced_mode_short": "panic",
                "leverage": 3,
            }
        }
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind=strategy_kind,
        enabled_sides=["long"],
        coin_count=3,
    )

    assert "CPU-compatible live-only values with no backtest effect" in caplog.text
    assert "coin_overrides.ETH.live.forced_mode_long" in caplog.text
    assert "coin_overrides.ETH.live.forced_mode_short" in caplog.text
    assert "coin_overrides.ETH.live.leverage" in caplog.text


@pytest.mark.parametrize(
    ("strategy_kind", "config_factory"),
    [
        ("ema_anchor", _directional_ema_config),
        ("trailing_martingale", _directional_tm_config),
    ],
)
@pytest.mark.parametrize(
    ("enabled_side", "disabled_side"),
    [("long", "short"), ("short", "long")],
)
def test_gpu_coin_overrides_accept_disabled_side_inert_forced_mode(
    strategy_kind, config_factory, enabled_side, disabled_side, caplog
):
    config = config_factory(
        long_enabled=enabled_side == "long",
        short_enabled=enabled_side == "short",
    )
    config["coin_overrides"] = {
        "ETH": {"live": {f"forced_mode_{disabled_side}": "graceful_stop"}}
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind=strategy_kind,
        enabled_sides=[enabled_side],
        coin_count=3,
    )

    assert (
        f"coin_overrides.ETH.live.forced_mode_{disabled_side}" in caplog.text
    )


@pytest.mark.parametrize(
    ("enabled_side", "disabled_side"),
    [("long", "short"), ("short", "long")],
)
def test_gpu_coin_overrides_reject_disabled_side_forced_normal(
    enabled_side, disabled_side
):
    config = _directional_ema_config(
        long_enabled=enabled_side == "long",
        short_enabled=enabled_side == "short",
    )
    config["coin_overrides"] = {
        "ETH": {"live": {f"forced_mode_{disabled_side}": "normal"}}
    }

    with pytest.raises(ValueError, match="do not model these paths yet"):
        _validate_gpu_coin_overrides(
            config,
            strategy_kind="ema_anchor",
            enabled_sides=[enabled_side],
            coin_count=3,
        )


@pytest.mark.parametrize(
    "risk_key",
    [
        "position_exposure_enforcer_enabled",
        "position_exposure_enforcer_threshold",
    ],
)
def test_gpu_ema_coin_overrides_reject_exact_inapplicable_position_enforcer(
    risk_key,
):
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "risk": {risk_key: config["bot"]["long"]["risk"][risk_key]}
                }
            }
        }
    }

    with pytest.raises(ValueError, match="do not model these paths yet"):
        _validate_gpu_coin_overrides(
            config,
            strategy_kind="ema_anchor",
            enabled_sides=["long"],
            coin_count=3,
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"bot": {"long": {"risk": {"n_positions": 2}}}},
        {
            "bot": {
                "long": {
                    "risk": {"we_excess_allowance_mode": "legacy_raw"}
                }
            }
        },
        {
            "bot": {
                "long": {
                    "risk": {"position_exposure_enforcer_enabled": True}
                }
            }
        },
        {"bot": {"short": {"strategy": {"ema_anchor": {"offset": 0.02}}}}},
    ],
)
def test_gpu_multicoin_coin_overrides_reject_unmodeled_leaves(patch):
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["coin_overrides"] = {"ETH": patch}

    with pytest.raises(ValueError, match="do not model these paths yet"):
        _validate_gpu_coin_overrides(
            config,
            strategy_kind="ema_anchor",
            enabled_sides=["long"],
            coin_count=3,
        )


@pytest.mark.parametrize(
    ("strategy_kind", "config_factory", "side"),
    [
        ("ema_anchor", _directional_ema_config, "long"),
        ("ema_anchor", _directional_ema_config, "short"),
        ("trailing_martingale", _directional_tm_config, "long"),
        ("trailing_martingale", _directional_tm_config, "short"),
    ],
)
def test_gpu_multicoin_accepts_enabled_side_forced_normal_override(
    strategy_kind, config_factory, side
):
    config = config_factory(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    config["coin_overrides"] = {
        "ETH": {"live": {f"forced_mode_{side}": "normal"}}
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind=strategy_kind,
        enabled_sides=[side],
        coin_count=3,
    )


def test_gpu_multicoin_accepts_complete_coin_hsl_override_group():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["live"]["hsl_signal_mode"] = "coin"
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "hsl": {
                        "enabled": True,
                        "red_threshold": 0.2,
                        "ema_span_minutes": 5.5,
                        "cooldown_minutes_after_red": 12.5,
                        "no_restart_drawdown_threshold": 0.8,
                        "restart_after_red_policy": "always",
                        "tier_ratios": {"yellow": 0.4, "orange": 0.75},
                        "orange_tier_mode": "graceful_stop",
                        "panic_close_order_type": "market",
                    }
                }
            }
        }
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind="ema_anchor",
        enabled_sides=["long"],
        coin_count=3,
    )


@pytest.mark.parametrize(
    ("hsl_patch", "match"),
    [
        ({"enabled": "yes"}, "enabled must be a boolean"),
        ({"red_threshold": -0.2}, "red_threshold must satisfy"),
        ({"ema_span_minutes": 0.0}, "ema_span_minutes must be >= 1"),
        (
            {"ema_span_minutes": float(np.finfo(np.float32).max) * 2.0},
            "representable as float32",
        ),
        ({"cooldown_minutes_after_red": -1.0}, "cooldown_minutes_after_red"),
        (
            {"cooldown_minutes_after_red": float(np.finfo(np.float32).max) * 2.0},
            "representable as float32",
        ),
        (
            {"no_restart_drawdown_threshold": 0.01},
            "no_restart_drawdown_threshold must satisfy",
        ),
        ({"restart_after_red_policy": "sometimes"}, "restart_after_red_policy"),
        ({"restart_after_red_policy": " ALWAYS "}, "restart_after_red_policy"),
        (
            {"tier_ratios": {"yellow": 0.8, "orange": 0.4}},
            "tier_ratios must satisfy",
        ),
        (
            {
                "tier_ratios": {
                    "yellow": 0.50000001,
                    "orange": 0.50000002,
                }
            },
            "remain strictly ordered.*float32",
        ),
        ({"tier_ratios": None}, "tier_ratios must be a dictionary"),
        ({"orange_tier_mode": "tp_only"}, "orange_tier_mode"),
        ({"orange_tier_mode": "GRACEFUL_STOP"}, "orange_tier_mode"),
        ({"panic_close_order_type": "makret"}, "to be limit or market"),
        ({"panic_close_order_type": " market "}, "to be limit or market"),
    ],
)
def test_gpu_multicoin_rejects_invalid_coin_hsl_values(hsl_patch, match):
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["live"]["hsl_signal_mode"] = "coin"
    config["coin_overrides"] = {
        "ETH": {"bot": {"long": {"hsl": hsl_patch}}}
    }

    with pytest.raises((TypeError, ValueError), match=match):
        _validate_gpu_coin_overrides(
            config,
            strategy_kind="ema_anchor",
            enabled_sides=["long"],
            coin_count=3,
        )


@pytest.mark.parametrize(
    ("signal_mode", "enabled_sides"),
    [("pside", ["long"]), ("unified", ["long"])],
)
def test_gpu_multicoin_coin_hsl_overrides_fail_closed_outside_one_side_coin_mode(
    signal_mode, enabled_sides
):
    config = _directional_ema_config(
        long_enabled=True, short_enabled="short" in enabled_sides
    )
    config["live"]["hsl_signal_mode"] = signal_mode
    config["coin_overrides"] = {
        "ETH": {"bot": {"long": {"hsl": {"enabled": True}}}}
    }

    with pytest.raises(ValueError, match="require live.hsl_signal_mode=coin"):
        _validate_gpu_coin_overrides(
            config,
            strategy_kind="ema_anchor",
            enabled_sides=enabled_sides,
            coin_count=3,
        )


@pytest.mark.parametrize(
    ("strategy_kind", "config_factory"),
    [
        ("ema_anchor", _directional_ema_config),
        ("trailing_martingale", _directional_tm_config),
    ],
)
def test_gpu_multicoin_coin_hsl_overrides_accept_fused_dual_side(
    strategy_kind, config_factory
):
    config = config_factory(long_enabled=True, short_enabled=True)
    config["live"]["hsl_signal_mode"] = "coin"
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {"hsl": {"enabled": True}},
                "short": {"hsl": {"enabled": True}},
            }
        }
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind=strategy_kind,
        enabled_sides=["long", "short"],
        coin_count=3,
    )


def test_gpu_coin_overrides_accept_single_coin_scope():
    config = _long_only_ema_config()
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"wallet_exposure_limit": 0.5}}}
    }

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize("signal_mode", ["unified", "pside", "coin"])
def test_gpu_single_coin_accepts_hsl_coin_override_for_every_signal_mode(
    signal_mode,
):
    config = _long_only_ema_config()
    config["live"]["hsl_signal_mode"] = signal_mode
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind="ema_anchor",
        enabled_sides=["long"],
        coin_count=1,
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_multicoin_accepts_static_tm_coin_overrides(side):
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    if side == "short":
        config = _directional_tm_config(long_enabled=False, short_enabled=True)
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                side: {
                    "strategy": {
                        "trailing_martingale": {
                            "entry": {
                                "threshold_base_pct": 0.02,
                                "retracement_base_pct": 0.005,
                            },
                            "close": {"qty_pct": 0.25},
                        }
                    },
                    "risk": {
                        "entry_cooldown_minutes": 15,
                        "we_excess_allowance_pct": 0.25,
                        "position_exposure_enforcer_enabled": True,
                        "position_exposure_enforcer_threshold": 0.8,
                    },
                    "wallet_exposure_limit": 0.4,
                    "unstuck": {
                        "enabled": True,
                        "ema_gating_enabled": False,
                        "close_pct": 0.1,
                        "ema_dist": 0.0,
                        "loss_allowance_pct": 0.02,
                        "threshold": 0.8,
                    },
                }
            }
        }
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    "patch",
    [
        {"bot": {"long": {"risk": {"n_positions": 2}}}},
    ],
)
def test_gpu_multicoin_tm_coin_overrides_reject_unmodeled_leaves(patch):
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["coin_overrides"] = {"ETH": patch}

    with pytest.raises(ValueError, match="do not model these paths yet"):
        _validate_gpu_coin_overrides(
            config,
            strategy_kind="trailing_martingale",
            enabled_sides=["long"],
            coin_count=3,
        )


@pytest.mark.parametrize("mode", ["disabled", "initial", "reentry", "all"])
def test_gpu_multicoin_tm_coin_overrides_accept_entry_ema_gate_mode(mode):
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {
                            "entry": {"ema_gate_mode": mode}
                        }
                    }
                }
            }
        }
    }

    _validate_gpu_coin_overrides(
        config,
        strategy_kind="trailing_martingale",
        enabled_sides=["long"],
        coin_count=3,
    )


def test_gpu_multicoin_foundation_accepts_fixed_wel_denominator():
    config = _long_only_ema_config()
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = False

    _validate_scope(config, _MulticoinEvaluator())


@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_multicoin_foundation_accepts_single_side_trailing_martingale(side):
    config = _directional_tm_config(
        long_enabled=side == "long", short_enabled=side == "short"
    )
    config["live"]["approved_coins"][side] = ["BTC", "ETH", "SOL"]
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["bot"][side]["risk"]["n_positions"] = 2

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_foundation_accepts_dual_side_trailing_martingale():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = True
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_foundation_accepts_dual_side_hedge_mode():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = True
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_foundation_accepts_forager_score_hysteresis():
    config = _long_only_ema_config()
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["forager_score_hysteresis_pct"] = 0.02
    config["backtest"]["dynamic_wel_by_tradability"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
def test_gpu_multicoin_ema_accepts_ordinary_market_execution(
    long_enabled, short_enabled
):
    config = _directional_ema_config(
        long_enabled=long_enabled, short_enabled=short_enabled
    )
    config["live"]["approved_coins"] = {
        side: ["BTC", "ETH", "SOL"] if enabled else []
        for side, enabled in (
            ("long", long_enabled),
            ("short", short_enabled),
        )
    }
    config["live"]["market_orders_allowed"] = True
    config["live"]["market_order_near_touch_threshold"] = 0.002

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_ema_market_execution_accepts_hsl():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["market_orders_allowed"] = True
    for side in ("long", "short"):
        config["bot"][side]["risk"]["n_positions"] = 3
        config["bot"][side]["hsl"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("feature", ["loss_gate", "twel", "unstuck", "override"])
def test_gpu_multicoin_ema_market_execution_accepts_protective_reducers(feature):
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["market_orders_allowed"] = True
    if feature == "loss_gate":
        config["live"]["max_realized_loss_pct"] = 0.05
    elif feature == "twel":
        config["bot"]["long"]["risk"][
            "total_exposure_enforcer_enabled"
        ] = True
    elif feature == "unstuck":
        config["bot"]["long"]["unstuck"]["enabled"] = True
    else:
        config["coin_overrides"] = {
            "ETH": {"bot": {"long": {"unstuck": {"enabled": True}}}}
        }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
def test_gpu_multicoin_tm_accepts_trailing_only_ordinary_market_execution(
    long_enabled, short_enabled
):
    config = _directional_tm_config(
        long_enabled=long_enabled, short_enabled=short_enabled
    )
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"] = {
        side: coins if enabled else []
        for side, enabled in (
            ("long", long_enabled),
            ("short", short_enabled),
        )
    }
    config["live"]["market_orders_allowed"] = True
    config["live"]["market_order_near_touch_threshold"] = 0.002
    for side, enabled in (("long", long_enabled), ("short", short_enabled)):
        if not enabled:
            continue
        strategy = config["bot"][side]["strategy"]["trailing_martingale"]
        strategy["entry"]["retracement_base_pct"] = 0.01
        strategy["close"]["retracement_base_pct"] = 0.01

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_tm_market_execution_accepts_hsl():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["market_orders_allowed"] = True
    for side in ("long", "short"):
        strategy = config["bot"][side]["strategy"]["trailing_martingale"]
        strategy["entry"]["retracement_base_pct"] = 0.01
        strategy["close"]["retracement_base_pct"] = 0.01
        config["bot"][side]["hsl"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("max_realized_loss_pct", [0.0, 0.05, 0.9999999995])
def test_gpu_multicoin_tm_market_execution_accepts_realized_loss_gate(
    long_enabled, short_enabled, max_realized_loss_pct
):
    config = _directional_tm_config(
        long_enabled=long_enabled, short_enabled=short_enabled
    )
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"] = {
        "long": coins if long_enabled else [],
        "short": coins if short_enabled else [],
    }
    config["live"]["market_orders_allowed"] = True
    config["live"]["max_realized_loss_pct"] = max_realized_loss_pct
    for side, enabled in (("long", long_enabled), ("short", short_enabled)):
        if not enabled:
            continue
        strategy = config["bot"][side]["strategy"]["trailing_martingale"]
        strategy["entry"]["retracement_base_pct"] = 0.01
        strategy["close"]["retracement_base_pct"] = 0.01

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("coin_override", [False, True])
def test_gpu_multicoin_tm_market_execution_accepts_unstuck(coin_override):
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["market_orders_allowed"] = True
    strategy = config["bot"]["long"]["strategy"]["trailing_martingale"]
    strategy["entry"]["retracement_base_pct"] = 0.01
    strategy["close"]["retracement_base_pct"] = 0.01
    if coin_override:
        config["coin_overrides"] = {
            "ETH": {"bot": {"long": {"unstuck": {"enabled": True}}}}
        }
    else:
        config["bot"]["long"]["unstuck"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    ("risk_key", "in_override"),
    [
        ("position_exposure_enforcer_enabled", False),
        ("position_exposure_enforcer_enabled", True),
        ("total_exposure_enforcer_enabled", False),
    ],
)
def test_gpu_multicoin_tm_market_execution_accepts_exposure_reducers(
    risk_key, in_override
):
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["market_orders_allowed"] = True
    strategy = config["bot"]["long"]["strategy"]["trailing_martingale"]
    strategy["entry"]["retracement_base_pct"] = 0.01
    strategy["close"]["retracement_base_pct"] = 0.01
    if in_override:
        config["coin_overrides"] = {
            "ETH": {"bot": {"long": {"risk": {risk_key: True}}}}
        }
    else:
        config["bot"]["long"]["risk"][risk_key] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_tm_recursive_entry_accepts_twel_entry_gate():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["market_orders_allowed"] = True
    strategy = config["bot"]["long"]["strategy"]["trailing_martingale"]
    strategy["entry"]["retracement_base_pct"] = 0.01
    strategy["close"]["retracement_base_pct"] = 0.01
    strategy["entry"]["retracement_base_pct"] = 0.0
    config["bot"]["long"]["risk"][
        "total_exposure_entry_gate_enabled"
    ] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_tm_market_execution_accepts_recursive_entry():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["market_orders_allowed"] = True
    for side in ("long", "short"):
        config["bot"][side]["risk"][
            "total_exposure_entry_gate_enabled"
        ] = False
        strategy = config["bot"][side]["strategy"]["trailing_martingale"]
        strategy["entry"]["retracement_base_pct"] = 0.0
        strategy["close"]["retracement_base_pct"] = 0.01

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_tm_market_execution_accepts_recursive_close():
    config = _directional_tm_config(long_enabled=True, short_enabled=True)
    coins = ["BTC", "ETH", "SOL"]
    config["live"]["approved_coins"] = {"long": coins, "short": coins}
    config["live"]["market_orders_allowed"] = True
    for side in ("long", "short"):
        strategy = config["bot"][side]["strategy"]["trailing_martingale"]
        strategy["entry"]["retracement_base_pct"] = 0.01
        strategy["close"]["retracement_base_pct"] = 0.0

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("branch", ["entry", "close"])
@pytest.mark.parametrize("value", [0.005, 0.0])
def test_gpu_multicoin_tm_market_execution_validates_static_override_modes(
    branch, value
):
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["live"]["market_orders_allowed"] = True
    strategy = config["bot"]["long"]["strategy"]["trailing_martingale"]
    strategy["entry"]["retracement_base_pct"] = 0.01
    strategy["close"]["retracement_base_pct"] = 0.01
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {
                            branch: {"retracement_base_pct": value}
                        }
                    }
                }
            }
        }
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_gpu_multicoin_foundation_accepts_dual_side_one_way_mode(strategy_kind):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = False
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_multicoin_foundation_accepts_dual_side_coin_overrides():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH", "SOL"],
    }
    config["live"]["hedge_mode"] = True
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True
    config["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "strategy": {"ema_anchor": {"offset": 0.02}},
                    "wallet_exposure_limit": 0.5,
                },
                "short": {
                    "strategy": {"ema_anchor": {"offset": 0.03}},
                    "risk": {"entry_cooldown_minutes": 15},
                },
            }
        }
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_gpu_multicoin_foundation_accepts_asymmetric_dual_side_coins(strategy_kind):
    builder = (
        _directional_tm_config
        if strategy_kind == "trailing_martingale"
        else _directional_ema_config
    )
    config = builder(long_enabled=True, short_enabled=True)
    config["live"]["approved_coins"] = {
        "long": ["BTC", "ETH", "SOL"],
        "short": ["BTC", "ETH"],
    }
    config["live"]["hedge_mode"] = True
    config["live"]["forager_score_hysteresis_pct"] = 0.0
    config["backtest"]["dynamic_wel_by_tradability"] = True

    config["live"]["ignored_coins"] = {
        "long": [],
        "short": ["DOGE"],
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    ("metric", "goal"),
    [
        ("equity_choppiness_w", "min"),
        ("equity_jerkiness_w", "min"),
        ("exponential_fit_error_w", "min"),
        ("volume_pct_per_day_avg_w", "max"),
    ],
)
def test_gpu_foundation_accepts_weighted_daily_series_metrics(metric, goal):
    config = _long_only_ema_config()
    config["optimize"]["scoring"] = [{"goal": goal, "metric": metric}]

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    ("metric", "goal"),
    [
        ("adg_btc", "max"),
        ("omega_ratio_w_btc", "max"),
    ],
)
def test_gpu_foundation_accepts_btc_account_metrics_without_collateral(
    metric, goal
):
    config = _long_only_ema_config()
    config["backtest"]["btc_collateral_cap"] = 0.0
    config["optimize"]["scoring"] = [{"goal": goal, "metric": metric}]

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    "metric",
    [
        "exposure_ratio_w_btc",
        "exposure_mean_ratio_w_btc",
        "calmar_ratio_w_btc",
        "sharpe_ratio_w_btc",
        "sortino_ratio_w_btc",
        "sterling_ratio_w_btc",
    ],
)
def test_gpu_foundation_excludes_btc_metrics_without_safe_proxy_surface(metric):
    pytest.importorskip("torch")
    from optimization.gpu.metrics import SUPPORTED_METRICS

    assert canonicalize_metric_name(metric) not in SUPPORTED_METRICS


@pytest.mark.parametrize(
    ("metric", "goal"),
    [
        ("drawdown_worst_btc", "min"),
        ("drawdown_worst_mean_1pct_btc", "min"),
        ("expected_shortfall_1pct_btc", "min"),
        ("sharpe_ratio_btc", "max"),
        ("sortino_ratio_btc", "max"),
        ("calmar_ratio_btc", "max"),
        ("sterling_ratio_btc", "max"),
    ],
)
def test_gpu_foundation_accepts_synchronized_btc_risk_metrics(metric, goal):
    config = _long_only_ema_config()
    config["backtest"]["btc_collateral_cap"] = 0.0
    config["optimize"]["scoring"] = [{"goal": goal, "metric": metric}]

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    ("metric", "goal"),
    [
        ("equity_balance_diff_neg_max_usd", "min"),
        ("equity_balance_diff_neg_mean_usd", "min"),
        ("paper_loss_ratio_usd", "max"),
        ("paper_loss_mean_ratio_usd", "max"),
        ("equity_balance_diff_neg_max_btc", "min"),
        ("equity_balance_diff_neg_mean_btc", "min"),
        ("paper_loss_ratio_btc", "max"),
        ("paper_loss_mean_ratio_btc", "max"),
    ],
)
def test_gpu_foundation_accepts_equity_balance_diff_metrics(metric, goal):
    config = _long_only_ema_config()
    config["backtest"]["btc_collateral_cap"] = 0.0
    config["optimize"]["scoring"] = [{"goal": goal, "metric": metric}]

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    "metric",
    [
        "entry_interval_hours_mean",
        "entry_interval_hours_median",
        "entry_interval_hours_p95",
        "entry_interval_hours_p99",
        "entry_interval_hours_max",
    ],
)
def test_gpu_foundation_accepts_entry_interval_metrics(metric):
    config = _long_only_ema_config()
    config["optimize"]["scoring"] = [{"goal": "min", "metric": metric}]

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_foundation_accepts_time_weighted_fill_gap_metric():
    config = _long_only_ema_config()
    config["optimize"]["scoring"] = [
        {"goal": "min", "metric": "fills_gap_time_weighted_mean_hours"}
    ]

    assert _validate_scope(config, _Evaluator()) == "bybit"


@pytest.mark.parametrize(
    "metric",
    [
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
        "exposure_mean_ratio_usd",
        "exposure_ratio_usd",
        "fills_gap_longest_days",
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p95_hours",
        "fills_gap_p99_hours",
        "fills_gap_time_weighted_mean_hours",
        "loss_profit_ratio",
        "peak_recovery_days_equity_usd",
        "peak_recovery_hours_equity_usd",
        "peak_recovery_days_pnl",
        "peak_recovery_days_strategy_eq",
        "peak_recovery_hours_strategy_eq",
        "peak_recovery_hours_pnl",
        "position_held_days_mean",
        "position_held_days_max",
        "position_held_hours_mean",
        "position_held_hours_max",
        "position_unchanged_days_max",
        "position_unchanged_hours_max",
        "positions_held_per_day",
        "strategy_eq_recovery_days_max",
        "strategy_eq_recovery_days_mean",
        "strategy_eq_recovery_days_median",
        "strategy_eq_recovery_days_p95",
        "strategy_eq_recovery_days_p99",
        "strategy_eq_recovery_days_mean_worst_5pct",
        "strategy_eq_recovery_days_mean_worst_1pct",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "volume_pct_per_day_avg",
        "volume_pct_per_day_avg_w",
    ],
)
def test_gpu_dual_multicoin_rejects_unreconstructable_metrics(metric):
    with pytest.raises(ValueError, match=metric):
        _validate_dual_multicoin_metrics(
            {metric, "adg_strategy_eq"},
            coin_count=3,
            enabled_sides={"long", "short"},
        )
    _validate_dual_multicoin_metrics(
        {metric, "adg_strategy_eq"},
        coin_count=3,
        enabled_sides={"long", "short"},
        shared_account_controller=True,
    )


def test_gpu_dual_multicoin_metric_gate_does_not_narrow_single_side():
    _validate_dual_multicoin_metrics(
        {
            "entry_initial_balance_pct_long",
            "entry_initial_balance_pct_short",
            "exposure_mean_ratio_usd",
            "exposure_ratio_usd",
            "fills_gap_longest_days",
            "fills_gap_mean_hours",
            "fills_gap_median_hours",
            "fills_gap_p95_hours",
            "fills_gap_p99_hours",
            "fills_gap_time_weighted_mean_hours",
            "loss_profit_ratio",
            "peak_recovery_days_equity_usd",
            "peak_recovery_hours_equity_usd",
            "peak_recovery_days_pnl",
            "peak_recovery_days_strategy_eq",
            "peak_recovery_hours_strategy_eq",
            "peak_recovery_hours_pnl",
            "position_held_days_mean",
            "position_held_days_max",
            "position_held_hours_mean",
            "position_held_hours_max",
            "position_unchanged_days_max",
            "position_unchanged_hours_max",
            "positions_held_per_day",
            "strategy_eq_recovery_days_max",
            "strategy_eq_recovery_days_mean",
            "strategy_eq_recovery_days_median",
            "strategy_eq_recovery_days_p95",
            "strategy_eq_recovery_days_p99",
            "strategy_eq_recovery_days_mean_worst_5pct",
            "strategy_eq_recovery_days_mean_worst_1pct",
            "total_wallet_exposure_max",
            "total_wallet_exposure_mean",
            "volume_pct_per_day_avg",
            "volume_pct_per_day_avg_w",
        },
        coin_count=3,
        enabled_sides={"long"},
    )


@pytest.mark.parametrize(
    "metric",
    [
        "strategy_eq_recovery_days_mean",
        "strategy_eq_recovery_days_median",
        "strategy_eq_recovery_days_p95",
        "strategy_eq_recovery_days_p99",
        "strategy_eq_recovery_days_mean_worst_5pct",
        "strategy_eq_recovery_days_mean_worst_1pct",
    ],
)
def test_gpu_recovery_distribution_metrics_accept_supported_multicoin(metric):
    for enabled in ({"long"}, {"short"}, {"long", "short"}):
        config = _directional_ema_config(
            long_enabled="long" in enabled,
            short_enabled="short" in enabled,
        )
        config["live"]["approved_coins"] = {
            side: ["BTC", "ETH", "SOL"] if side in enabled else []
            for side in ("long", "short")
        }
        config["live"]["hedge_mode"] = True
        config["live"]["forager_score_hysteresis_pct"] = 0.0
        config["backtest"]["dynamic_wel_by_tradability"] = True
        config["optimize"]["scoring"] = [
            {"goal": "min", "metric": metric}
        ]

        assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


@pytest.mark.parametrize(
    ("side", "bound_map"),
    [
        ("long", EMA_MULTICOIN_LONG_BOUND_MAP),
        ("short", EMA_MULTICOIN_SHORT_BOUND_MAP),
    ],
)
def test_gpu_multicoin_bound_map_exposes_forager_and_position_dimensions(
    side, bound_map
):
    for suffix in (
        "forager_volume_ema_span_1m",
        "forager_volatility_ema_span_1m",
        "forager_volume_drop_pct",
        "forager_score_weights_volume",
        "forager_score_weights_ema_readiness",
        "forager_score_weights_volatility",
        "n_positions",
    ):
        key = f"{side}_{suffix}"
        assert bound_map[key] == key
    assert (
        bound_map[f"{side}_risk_we_excess_allowance_pct"]
        == f"{side}_we_excess_allowance_pct"
    )
    assert (
        bound_map[f"{side}_risk_twel_enforcer_threshold"]
        == f"{side}_twel_enforcer_threshold"
    )
    for suffix in (
        "unstuck_close_pct",
        "unstuck_ema_dist",
        "unstuck_loss_allowance_pct",
        "unstuck_threshold",
        "hsl_cooldown_minutes_after_red",
        "hsl_ema_span_minutes",
        "hsl_red_threshold",
    ):
        assert bound_map[f"{side}_{suffix}"] == f"{side}_{suffix}"


@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_tm_multicoin_bound_map_exposes_strategy_forager_and_positions(side):
    bound_map = _trailing_martingale_multicoin_bound_map(side, set())

    assert (
        bound_map[f"{side}_entry_threshold_base_pct"]
        == f"{side}_entry_threshold_base_pct"
    )
    assert (
        bound_map[f"{side}_close_retracement_base_pct"]
        == f"{side}_close_retracement_base_pct"
    )
    assert (
        bound_map[f"{side}_forager_score_weights_ema_readiness"]
        == f"{side}_forager_score_weights_ema_readiness"
    )
    assert bound_map[f"{side}_n_positions"] == f"{side}_n_positions"
    assert (
        bound_map[f"{side}_risk_we_excess_allowance_pct"]
        == f"{side}_we_excess_allowance_pct"
    )
    assert (
        bound_map[f"{side}_risk_twel_enforcer_threshold"]
        == f"{side}_twel_enforcer_threshold"
    )
    for suffix in (
        "unstuck_close_pct",
        "unstuck_ema_dist",
        "unstuck_loss_allowance_pct",
        "unstuck_threshold",
        "hsl_cooldown_minutes_after_red",
        "hsl_ema_span_minutes",
        "hsl_red_threshold",
    ):
        assert bound_map[f"{side}_{suffix}"] == f"{side}_{suffix}"


def test_gpu_short_multicoin_mirror_includes_long_forager_source_dimensions():
    bound_map = _ema_multicoin_bound_map(
        "short", {"mirror_short_from_long"}
    )

    for suffix in (
        "forager_volume_ema_span_1m",
        "forager_volatility_ema_span_1m",
        "forager_volume_drop_pct",
        "forager_score_weights_volume",
        "forager_score_weights_ema_readiness",
        "forager_score_weights_volatility",
        "n_positions",
    ):
        assert f"long_{suffix}" in bound_map
        assert f"short_{suffix}" in bound_map


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


def test_gpu_foundation_accepts_single_coin_unstuck_on_short_side():
    config = _directional_ema_config(long_enabled=False, short_enabled=True)
    config["bot"]["short"]["unstuck"]["enabled"] = True

    assert _validate_scope(config, _Evaluator()) == "bybit"


def test_gpu_unstuck_search_excludes_disabled_side_genes():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)

    assert _gpu_unstuck_search_sides(config, []) == set()
    assert not _gpu_unstuck_parameter_active(
        "long_unstuck_close_pct", set()
    )
    assert _gpu_unstuck_parameter_active("long_offset", set())

    config["bot"]["short"]["unstuck"]["enabled"] = True
    search_sides = _gpu_unstuck_search_sides(config, [])
    assert search_sides == {"short"}
    assert not _gpu_unstuck_parameter_active(
        "long_unstuck_threshold", search_sides
    )
    assert _gpu_unstuck_parameter_active(
        "short_unstuck_threshold", search_sides
    )


def test_gpu_unstuck_search_keeps_genes_enabled_by_coin_override():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["coin_overrides"] = {
        "ETH": {"bot": {"long": {"unstuck": {"enabled": True}}}}
    }

    search_sides = _gpu_unstuck_search_sides(config, [])

    assert search_sides == {"long"}
    assert _gpu_unstuck_parameter_active(
        "long_unstuck_loss_allowance_pct", search_sides
    )


def test_gpu_suite_keeps_unstuck_genes_used_by_any_scenario():
    base = _directional_ema_config(long_enabled=True, short_enabled=False)
    scenario = copy.deepcopy(base)
    scenario["bot"]["long"]["unstuck"]["enabled"] = True

    assert _gpu_unstuck_search_sides(
        base, [{"config": scenario}]
    ) == {"long"}


def test_gpu_suite_keeps_mirrored_unstuck_source_genes():
    base = _directional_ema_config(long_enabled=False, short_enabled=True)
    scenario = copy.deepcopy(base)
    scenario["bot"]["short"]["unstuck"]["enabled"] = True

    search_sides = _gpu_unstuck_search_sides(
        base,
        [{"config": scenario}],
        {"mirror_short_from_long"},
    )

    assert search_sides == {"long", "short"}
    assert _gpu_unstuck_parameter_active(
        "long_unstuck_close_pct", search_sides
    )


def test_gpu_foundation_accepts_single_side_multicoin_unstuck():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    config["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL"]
    config["bot"]["long"]["risk"]["n_positions"] = 2
    config["bot"]["long"]["unstuck"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_foundation_accepts_dual_side_multicoin_unstuck():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["hedge_mode"] = True
    for side in ("long", "short"):
        config["live"]["approved_coins"][side] = ["BTC", "ETH", "SOL"]
        config["bot"][side]["risk"]["n_positions"] = 2
    config["bot"]["long"]["unstuck"]["enabled"] = True

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


def test_gpu_foundation_accepts_dual_side_multicoin_override_unstuck():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["live"]["hedge_mode"] = True
    for side in ("long", "short"):
        config["live"]["approved_coins"][side] = ["BTC", "ETH", "SOL"]
        config["bot"][side]["risk"]["n_positions"] = 2
    config["coin_overrides"] = {
        "ETH": {"bot": {"short": {"unstuck": {"enabled": True}}}}
    }

    assert _validate_scope(config, _MulticoinEvaluator()) == "bybit"


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


def test_validation_selection_uses_true_front_when_no_off_front_evidence_exists():
    objectives = np.array(
        [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    )
    scores = objectives.mean(axis=1)

    selected = _select_validation_indices(objectives, scores, total=3, probes=1)
    diversity_baseline = _select_validation_indices(
        objectives, scores, total=3, probes=0
    )

    assert len(selected) == len(objectives)
    assert all(not is_probe and is_front for _index, is_probe, is_front in selected)
    assert selected[:3] == diversity_baseline[:3]


def test_validation_selection_uses_all_available_off_front_probes():
    objectives = np.array(
        [
            [0.0, 7.0],
            [1.0, 6.0],
            [2.0, 5.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 2.0],
            [6.0, 1.0],
            [7.0, 0.0],
            [8.0, 8.0],
            [9.0, 9.0],
        ]
    )
    scores = objectives.mean(axis=1)

    selected = _select_validation_indices(objectives, scores, total=8, probes=4)
    diversity_baseline = _select_validation_indices(
        objectives, scores, total=6, probes=0
    )

    chosen = selected[:8]
    assert sum(is_probe for _index, is_probe, _front in chosen) == 2
    assert sum(is_front for _index, _probe, is_front in chosen) == 6
    assert {
        index for index, is_probe, _front in chosen if is_probe
    } == {8, 9}
    assert {
        index for index, _is_probe, is_front in chosen if is_front
    } == {index for index, _is_probe, _front in diversity_baseline[:6]}


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

    chosen = _select_exact_validations(
        selections,
        total=2,
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


def test_duplicate_broad_probe_falls_back_to_novel_true_front_candidates():
    chosen = _select_exact_validations(
        [(0, False, True), (1, True, False), (2, False, True)],
        total=2,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes={"hash-1"},
        submitted_hashes=set(),
    )

    assert [item[0] for item in chosen] == [0, 2]
    assert all(
        not is_probe and is_front
        for _index, is_probe, is_front, *_rest in chosen
    )


def test_unallocated_infeasible_fallback_does_not_restore_probe_quota():
    objectives = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [8.0, 8.0],
        ]
    )
    scores = objectives.mean(axis=1)
    selections = _select_validation_indices(
        objectives,
        scores,
        violations=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        total=3,
        probes=1,
    )

    assert selections[-1] == (4, True, False)
    chosen = _select_exact_validations(
        selections,
        total=3,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes=set(),
        submitted_hashes=set(),
    )

    assert len(chosen) == 3
    assert all(not is_probe and is_front for _index, is_probe, is_front, *_ in chosen)


def test_duplicate_fronts_do_not_expand_adaptive_probe_allocation():
    selections = [
        *((index, False, True) for index in range(6)),
        (6, True, False),
        (7, True, False),
        *((index, True, False) for index in range(8, 12)),
        *((index, False, True) for index in range(12, 17)),
    ]

    chosen = _select_exact_validations(
        selections,
        total=8,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes={f"hash-{index}" for index in range(1, 6)},
        submitted_hashes=set(),
    )

    assert len(chosen) == 8
    assert sum(is_probe for _index, is_probe, _front, *_rest in chosen) == 2
    assert sum(is_front for _index, _is_probe, is_front, *_rest in chosen) == 6
    assert {item[0] for item in chosen if item[2]} == {0, 12, 13, 14, 15, 16}


def test_probe_shortfall_logging_is_bounded_and_reports_recovery(caplog):
    with caplog.at_level("INFO"):
        state = _update_probe_shortfall_log(None, requested=4, actual=2)
        state = _update_probe_shortfall_log(state, requested=4, actual=2)
        state = _update_probe_shortfall_log(state, requested=4, actual=4)

    assert state is None
    assert caplog.text.count("fewer novel candidates") == 1
    assert "requested=4 available=2" in caplog.text
    assert caplog.text.count("allocation recovered") == 1


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
    chosen = _select_exact_validations(
        selections,
        total=8,
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


def test_validation_revalidates_completed_current_front_instead_of_relabeling_probe():
    chosen = _select_exact_validations(
        [(0, False, True), (1, True, False), (2, True, False)],
        total=2,
        candidate_for_index=lambda index: [index],
        digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
        completed_hashes={"hash-0"},
        submitted_hashes=set(),
    )

    assert [item[0] for item in chosen] == [1, 0]
    assert chosen[0][1:3] == (True, False)
    assert chosen[1][1:3] == (False, True)


def test_validation_waits_for_submitted_current_front():
    with pytest.raises(
        _ProxyFrontValidationPending,
        match="proxy-front exact validation is still in flight",
    ):
        _select_exact_validations(
            [(0, False, True), (1, True, False), (2, True, False)],
            total=2,
            candidate_for_index=lambda index: [index],
            digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
            completed_hashes=set(),
            submitted_hashes={"hash-0"},
        )


def test_validation_fails_closed_when_selector_contains_no_front_candidate():
    with pytest.raises(RuntimeError, match="truthful proxy-front safety evidence"):
        _select_exact_validations(
            [(0, True, False), (1, True, False)],
            total=2,
            candidate_for_index=lambda index: [index],
            digest_for_candidate=lambda candidate: f"hash-{candidate[0]}",
            completed_hashes=set(),
            submitted_hashes=set(),
        )


def test_validation_scans_fallbacks_for_novel_proxy_front_before_failing():
    chosen = _select_exact_validations(
        [
            (0, False, True),
            (1, True, False),
            (2, True, False),
            (3, False, True),
        ],
        total=2,
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


def test_gpu_mirror_hash_ignores_shadowed_short_genes_during_recovery():
    key_paths = [
        ("long_offset", ("bot", "long", "offset")),
        ("short_offset", ("bot", "short", "offset")),
        ("long_total_wallet_exposure_limit", ("bot", "long", "wel")),
        ("short_total_wallet_exposure_limit", ("bot", "short", "wel")),
    ]
    base_vector = [0.2, 0.7, 1.0, 0.0]
    submitted = [0.9, 0.7, 1.0, 0.0]
    recovered_from_mirrored_result = [0.9, 0.9, 1.0, 1.0]

    submitted = _canonicalize_mirrored_hash_vector(
        submitted, base_vector, key_paths
    )
    recovered = _canonicalize_mirrored_hash_vector(
        recovered_from_mirrored_result,
        base_vector,
        key_paths,
    )

    assert submitted == recovered == [0.9, 0.7, 1.0, 0.0]
    bounds = [Bound(0.0, 1.0) for _ in key_paths]
    assert _canonical_vector_hash(submitted, bounds, 6) == _canonical_vector_hash(
        recovered, bounds, 6
    )


def test_gpu_mirror_hash_neutralizes_anchor_shadow_without_long_shape_key():
    key_paths = [
        ("anchor_index", ("optimize", "fine_tune_anchor_index")),
        ("short_offset", ("bot", "short", "offset")),
    ]
    base_vector = [0.0, 0.7]

    submitted = _canonicalize_mirrored_hash_vector(
        [1.0, 0.7], base_vector, key_paths
    )
    recovered = _canonicalize_mirrored_hash_vector(
        [1.0, 0.2], base_vector, key_paths
    )

    assert submitted == recovered == [1.0, 0.7]


def test_gpu_lossless_hash_uses_effective_threshold_and_mirror_ordering():
    key_paths = [
        (
            "long_close_threshold_base_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "close",
                "threshold_base_pct",
            ),
        ),
        (
            "long_close_retracement_base_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "close",
                "retracement_base_pct",
            ),
        ),
        (
            "short_close_threshold_base_pct",
            (
                "bot",
                "short",
                "strategy",
                "trailing_martingale",
                "close",
                "threshold_base_pct",
            ),
        ),
        (
            "short_close_retracement_base_pct",
            (
                "bot",
                "short",
                "strategy",
                "trailing_martingale",
                "close",
                "retracement_base_pct",
            ),
        ),
    ]
    base_vector = [0.01, 0.02, 0.7, 0.8]
    submitted = [0.01, 0.04, 0.6, 0.9]
    recovered = [0.04, 0.04, 0.04, 0.04]
    overrides = {"mirror_short_from_long", "lossless_close_trailing"}

    submitted = _canonicalize_optimizer_override_hash_vector(
        submitted, base_vector, key_paths, overrides
    )
    recovered = _canonicalize_optimizer_override_hash_vector(
        recovered, base_vector, key_paths, overrides
    )

    assert submitted == recovered == [0.04, 0.04, 0.7, 0.8]


def test_gpu_lossless_hash_uses_selected_anchor_fixed_retracement():
    key_paths = [
        (ANCHOR_GENE_KEY, ("optimize", "fine_tune_anchor_index")),
        (
            "long_close_threshold_base_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "close",
                "threshold_base_pct",
            ),
        ),
    ]
    base_vector = [0.0, 0.01]
    anchors = [
        {"long_close_retracement_base_pct": 0.02},
        {"long_close_retracement_base_pct": 0.06},
    ]

    submitted = _canonicalize_optimizer_override_hash_vector(
        [1.0, 0.01],
        base_vector,
        key_paths,
        {"lossless_close_trailing"},
        anchor_parameter_overrides=anchors,
    )
    recovered = _canonicalize_optimizer_override_hash_vector(
        [1.0, 0.06],
        base_vector,
        key_paths,
        {"lossless_close_trailing"},
        anchor_parameter_overrides=anchors,
    )

    assert submitted == recovered == [1.0, 0.06]


def test_gpu_hash_uses_runtime_fixed_value_before_lossless_override():
    key_paths = [
        (
            "long_close_threshold_base_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "close",
                "threshold_base_pct",
            ),
        ),
        (
            "long_close_retracement_base_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "close",
                "retracement_base_pct",
            ),
        ),
    ]
    base_vector = [0.01, 0.02]

    submitted = _canonicalize_optimizer_override_hash_vector(
        [0.01, 0.02],
        base_vector,
        key_paths,
        {"lossless_close_trailing"},
        fixed_bound_values={"long_close_retracement_base_pct": 0.06},
        fixed_parameter_overrides={"long_close_retracement_base_pct": 0.06},
    )
    recovered = _canonicalize_optimizer_override_hash_vector(
        [0.06, 0.06],
        base_vector,
        key_paths,
        {"lossless_close_trailing"},
        fixed_bound_values={"long_close_retracement_base_pct": 0.06},
        fixed_parameter_overrides={"long_close_retracement_base_pct": 0.06},
    )

    assert submitted == recovered == [0.06, 0.06]


def test_gpu_short_only_mirror_keeps_long_source_genes_active():
    assert _gpu_candidate_source_sides(
        {"short"}, {"mirror_short_from_long"}
    ) == {"long", "short"}
    assert _gpu_candidate_source_sides(
        {"long"}, {"mirror_short_from_long"}
    ) == {"long"}


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


def test_gpu_proxy_applies_mirror_before_lossless_close_override():
    parameters = {
        "long_close_threshold_base_pct": 0.01,
        "long_close_retracement_base_pct": 0.02,
        "long_entry_initial_qty_pct": 0.03,
        "short_close_threshold_base_pct": 0.08,
        "short_close_retracement_base_pct": 0.09,
        "short_entry_initial_qty_pct": 0.10,
    }

    result = _apply_gpu_optimizer_overrides(
        parameters,
        {"mirror_short_from_long", "lossless_close_trailing"},
    )

    assert result["long_close_threshold_base_pct"] == pytest.approx(0.02)
    assert result["short_close_threshold_base_pct"] == pytest.approx(0.02)
    assert result["short_close_retracement_base_pct"] == pytest.approx(0.02)
    assert result["short_entry_initial_qty_pct"] == pytest.approx(0.03)


def test_gpu_proxy_parameter_builder_applies_optimizer_overrides_after_anchors():
    mapped = {
        "long_close_threshold_base_pct": (1, Bound(0.0, 1.0)),
        "long_close_retracement_base_pct": (2, Bound(0.0, 1.0)),
        "short_close_threshold_base_pct": (3, Bound(0.0, 1.0)),
        "short_close_retracement_base_pct": (4, Bound(0.0, 1.0)),
    }
    active = [(ANCHOR_GENE_KEY, 0, Bound(0.0, 1.0, 1.0))]

    parameters = _build_proxy_parameter_dicts(
        [0.0, 0.01, 0.02, 0.4, 0.5],
        mapped,
        active,
        np.asarray([[1.0]]),
        anchor_parameter_overrides=[
            {
                "long_close_threshold_base_pct": 0.03,
                "long_close_retracement_base_pct": 0.04,
            },
            {
                "long_close_threshold_base_pct": 0.05,
                "long_close_retracement_base_pct": 0.06,
            },
        ],
        optimizer_overrides={
            "mirror_short_from_long",
            "lossless_close_trailing",
        },
    )

    assert parameters == [
        {
            "long_close_threshold_base_pct": pytest.approx(0.06),
            "long_close_retracement_base_pct": pytest.approx(0.06),
            "short_close_threshold_base_pct": pytest.approx(0.06),
            "short_close_retracement_base_pct": pytest.approx(0.06),
        }
    ]


def test_gpu_proxy_parameter_builder_applies_fixed_runtime_after_tunables():
    mapped = {
        "long_close_threshold_base_pct": (0, Bound(0.0, 1.0)),
        "long_close_retracement_base_pct": (1, Bound(0.0, 1.0)),
    }
    active = [
        ("long_close_threshold_base_pct", 0, mapped["long_close_threshold_base_pct"][1]),
        (
            "long_close_retracement_base_pct",
            1,
            mapped["long_close_retracement_base_pct"][1],
        ),
    ]

    parameters = _build_proxy_parameter_dicts(
        [0.01, 0.02],
        mapped,
        active,
        np.asarray([[0.03, 0.04]]),
        fixed_parameter_overrides={"long_close_retracement_base_pct": 0.06},
        optimizer_overrides={"lossless_close_trailing"},
    )

    assert parameters == [
        {
            "long_close_threshold_base_pct": pytest.approx(0.06),
            "long_close_retracement_base_pct": pytest.approx(0.06),
        }
    ]


def test_gpu_optimizer_override_template_uses_exact_materializer():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    long_strategy = config["bot"]["long"]["strategy"]["ema_anchor"]
    short_strategy = config["bot"]["short"]["strategy"]["ema_anchor"]
    long_strategy["base_qty_pct"] = 0.123
    short_strategy["base_qty_pct"] = 0.456

    proxy_config = _materialize_gpu_override_template(
        config,
        ["mirror_short_from_long"],
    )

    assert (
        proxy_config["bot"]["short"]["strategy"]["ema_anchor"]["base_qty_pct"]
        == pytest.approx(0.123)
    )
    assert short_strategy["base_qty_pct"] == pytest.approx(0.456)


def test_gpu_runtime_template_applies_fixed_values_before_optimizer_overrides():
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.strategy.ema_anchor.base_qty_pct": 0.321
    }

    proxy_config = _materialize_gpu_override_template(
        config,
        ["mirror_short_from_long"],
    )

    assert (
        proxy_config["bot"]["long"]["strategy"]["ema_anchor"]["base_qty_pct"]
        == pytest.approx(0.321)
    )
    assert (
        proxy_config["bot"]["short"]["strategy"]["ema_anchor"]["base_qty_pct"]
        == pytest.approx(0.321)
    )


def test_gpu_runtime_template_rejects_fixed_strategy_kind_change():
    config = _long_only_ema_config()
    config["optimize"]["fixed_runtime_overrides"] = {
        "live.strategy_kind": "trailing_martingale"
    }

    with pytest.raises(ValueError, match="may not change live.strategy_kind"):
        _materialize_gpu_override_template(
            config,
            [],
        )


def test_gpu_fixed_bound_context_maps_effective_candidate_shadows():
    config = _long_only_ema_config()
    path = ("bot", "long", "strategy", "ema_anchor", "offset")
    fixed_only_path = ("bot", "long", "risk", "entry_cooldown_minutes")
    config["optimize"]["fixed_runtime_overrides"] = {
        ".".join(path): 0.123,
        ".".join(fixed_only_path): 17.0,
    }
    effective = copy.deepcopy(config)
    effective["bot"]["long"]["strategy"]["ema_anchor"]["offset"] = 0.123
    effective["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 17.0

    bound_values, parameters = _gpu_fixed_bound_context(
        config,
        effective,
        [("long_offset", path)],
        {
            "long_offset": "long_offset",
            "long_risk_entry_cooldown_minutes": "long_entry_cooldown_minutes",
        },
    )

    assert bound_values == {
        "long_offset": 0.123,
        "long_risk_entry_cooldown_minutes": 17.0,
    }
    assert parameters == {
        "long_offset": 0.123,
        "long_entry_cooldown_minutes": 17.0,
    }


def test_gpu_fixed_disabled_retracement_canonicalizes_dead_weight_genes():
    config = _directional_tm_config(long_enabled=True, short_enabled=False)
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.strategy.trailing_martingale.close.retracement_base_pct": 0.0
    }
    close_path = (
        "bot",
        "long",
        "strategy",
        "trailing_martingale",
        "close",
    )
    key_paths = [
        ("long_close_retracement_base_pct", (*close_path, "retracement_base_pct")),
        (
            "long_close_retracement_volatility_1h_weight",
            (*close_path, "retracement_volatility_1h_weight"),
        ),
        (
            "long_close_retracement_volatility_1m_weight",
            (*close_path, "retracement_volatility_1m_weight"),
        ),
    ]
    bound_map = {key: key for key, _path in key_paths}
    effective = _materialize_gpu_override_template(config, [])

    bound_values, parameters = _gpu_fixed_bound_context(
        config,
        effective,
        key_paths,
        bound_map,
    )

    assert bound_values == {
        "long_close_retracement_base_pct": 0.0,
        "long_close_retracement_volatility_1h_weight": 0.01,
        "long_close_retracement_volatility_1m_weight": 0.01,
    }
    assert parameters == bound_values
    assert _canonicalize_optimizer_override_hash_vector(
        [0.0, 37.0, 38.0],
        [0.0, 1.0, 1.0],
        key_paths,
        set(),
        fixed_bound_values=bound_values,
        fixed_parameter_overrides=parameters,
    ) == [0.0, 0.01, 0.01]

    materialized = _materialize_gpu_override_template(config, [])
    close = materialized["bot"]["long"]["strategy"]["trailing_martingale"][
        "close"
    ]
    assert close["retracement_base_pct"] == 0.0
    assert close["retracement_volatility_1h_weight"] == 0.01
    assert close["retracement_volatility_1m_weight"] == 0.01


def test_gpu_materialized_fixed_runtime_scope_accepts_single_coin_unstuck():
    config = _long_only_ema_config()
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.unstuck.enabled": True
    }
    proxy_config = _materialize_gpu_override_template(
        config,
        [],
    )

    assert _validate_scope(proxy_config, _Evaluator()) == "bybit"


def test_gpu_optimizer_override_scope_fails_closed():
    assert _validate_gpu_optimizer_overrides(
        ["mirror_short_from_long"], "ema_anchor"
    ) == {"mirror_short_from_long"}
    assert _validate_gpu_optimizer_overrides(
        ["lossless_close_trailing"], "trailing_martingale"
    ) == {"lossless_close_trailing"}

    with pytest.raises(ValueError, match="forward_tp_grid"):
        _validate_gpu_optimizer_overrides(["forward_tp_grid"], "ema_anchor")
    with pytest.raises(ValueError, match="requires.*trailing_martingale"):
        _validate_gpu_optimizer_overrides(
            ["lossless_close_trailing"], "ema_anchor"
        )


def test_gpu_anchor_context_maps_fixed_values_and_preserves_ranges():
    config = {
        ANCHOR_PLAN_KEY: {
            "fixed_keys": [
                "long_base_qty_pct",
                "long_risk_we_excess_allowance_pct",
            ],
            "anchors": [
                {
                    "fixed_values": [
                        {"key": "long_base_qty_pct", "value": 0.1},
                        {
                            "key": "long_risk_we_excess_allowance_pct",
                            "value": 0.0,
                        },
                    ]
                },
                {
                    "fixed_values": [
                        {"key": "long_base_qty_pct", "value": 0.3},
                        {
                            "key": "long_risk_we_excess_allowance_pct",
                            "value": 0.0,
                        },
                    ]
                },
            ],
        }
    }

    overrides, fixed_bounds = _build_anchor_parameter_context(
        config, {"long_base_qty_pct": "long_base_qty_pct"}
    )

    assert overrides == [
        {"long_base_qty_pct": 0.1},
        {"long_base_qty_pct": 0.3},
    ]
    assert fixed_bounds["long_base_qty_pct"] == Bound(0.1, 0.3)
    assert fixed_bounds["long_risk_we_excess_allowance_pct"] == Bound(0.0, 0.0)


def test_gpu_anchor_proxy_parameters_select_fixed_values_before_tunables():
    mapped = {"long_offset": (1, Bound(0.0, 1.0))}
    active = [
        (ANCHOR_GENE_KEY, 0, Bound(0.0, 1.0, 1.0)),
        ("long_offset", 1, mapped["long_offset"][1]),
    ]

    parameters = _build_proxy_parameter_dicts(
        [0.0, 0.5],
        mapped,
        active,
        np.array([[0.0, 0.75], [1.0, 0.125]]),
        anchor_parameter_overrides=[
            {"long_base_qty_pct": 0.1, "long_offset": 0.9},
            {"long_base_qty_pct": 0.3, "long_offset": 0.8},
        ],
    )

    assert parameters == [
        {"long_base_qty_pct": 0.1, "long_offset": 0.75},
        {"long_base_qty_pct": 0.3, "long_offset": 0.125},
    ]


@pytest.mark.parametrize(
    ("strategy_kind", "fixed_key", "fixed_path", "tunable_key", "tunable_path"),
    [
        (
            "ema_anchor",
            "long_base_qty_pct",
            ("bot", "long", "strategy", "ema_anchor", "base_qty_pct"),
            "long_offset",
            ("bot", "long", "strategy", "ema_anchor", "offset"),
        ),
        (
            "trailing_martingale",
            "long_entry_retracement_base_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "entry",
                "retracement_base_pct",
            ),
            "long_entry_initial_qty_pct",
            (
                "bot",
                "long",
                "strategy",
                "trailing_martingale",
                "entry",
                "initial_qty_pct",
            ),
        ),
    ],
)
def test_gpu_anchor_proxy_values_match_exact_candidate_materialization(
    strategy_kind, fixed_key, fixed_path, tunable_key, tunable_path
):
    config = (
        _long_only_ema_config()
        if strategy_kind == "ema_anchor"
        else _directional_tm_config(long_enabled=True, short_enabled=False)
    )
    config[ANCHOR_PLAN_KEY] = {
        "fixed_keys": [fixed_key],
        "tunable_keys": [tunable_key],
        "key_paths": [list(tunable_path)],
        "anchors": [
            {
                "source": "anchor-a.json",
                "fixed_values": [
                    {"key": fixed_key, "path": list(fixed_path), "value": 0.1}
                ],
            },
            {
                "source": "anchor-b.json",
                "fixed_values": [
                    {"key": fixed_key, "path": list(fixed_path), "value": 0.3}
                ],
            },
        ],
    }
    bound_map = (
        EMA_MULTICOIN_LONG_BOUND_MAP
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_BOUND_MAP
    )
    anchor_overrides, _fixed_bounds = _build_anchor_parameter_context(
        config, bound_map
    )
    proxy_tunable_key = bound_map[tunable_key]
    active = [
        (ANCHOR_GENE_KEY, 0, Bound(0.0, 1.0, 1.0)),
        (proxy_tunable_key, 1, Bound(0.0, 1.0)),
    ]
    vectors = [[0.0, 0.75], [1.0, 0.125]]

    proxy_parameters = _build_proxy_parameter_dicts(
        [0.0, 0.0],
        {proxy_tunable_key: (1, Bound(0.0, 1.0))},
        active,
        np.asarray(vectors),
        anchor_parameter_overrides=anchor_overrides,
    )
    exact_configs = [build_optimizer_vector_config(vector, config) for vector in vectors]

    for proxy, exact, expected_fixed, expected_tunable in zip(
        proxy_parameters,
        exact_configs,
        (0.1, 0.3),
        (0.75, 0.125),
    ):
        fixed = exact
        for part in fixed_path:
            fixed = fixed[part]
        tunable = exact
        for part in tunable_path:
            tunable = tunable[part]
        assert proxy[bound_map[fixed_key]] == fixed == expected_fixed
        assert proxy[proxy_tunable_key] == tunable == expected_tunable


def test_gpu_anchor_context_fails_closed_on_missing_fixed_values():
    config = {
        ANCHOR_PLAN_KEY: {
            "fixed_keys": ["long_base_qty_pct", "long_offset"],
            "anchors": [
                {"fixed_values": [{"key": "long_base_qty_pct", "value": 0.1}]}
            ],
        }
    }

    with pytest.raises(ValueError, match="missing fixed optimizer values"):
        _build_anchor_parameter_context(config, {})


def test_gpu_anchor_ranges_support_multicoin_exposure_headroom():
    config = {
        ANCHOR_PLAN_KEY: {
            "fixed_keys": ["long_risk_we_excess_allowance_pct"],
            "anchors": [
                {
                    "fixed_values": [
                        {
                            "key": "long_risk_we_excess_allowance_pct",
                            "value": 0.0,
                        }
                    ]
                },
                {
                    "fixed_values": [
                        {
                            "key": "long_risk_we_excess_allowance_pct",
                            "value": 0.2,
                        }
                    ]
                },
            ],
        }
    }

    _overrides, fixed_bounds = _build_anchor_parameter_context(config, {})

    _validate_pinned_scope_bounds(fixed_bounds, {}, {"long"}, coin_count=1)
    _validate_pinned_scope_bounds(fixed_bounds, {}, {"long"}, coin_count=2)


def test_gpu_anchor_ranges_cannot_change_side_enablement():
    config = {
        ANCHOR_PLAN_KEY: {
            "fixed_keys": [
                "long_n_positions",
                "long_total_wallet_exposure_limit",
            ],
            "anchors": [
                {
                    "fixed_values": [
                        {"key": "long_n_positions", "value": 1.0},
                        {
                            "key": "long_total_wallet_exposure_limit",
                            "value": 1.0,
                        },
                    ]
                },
                {
                    "fixed_values": [
                        {"key": "long_n_positions", "value": 1.0},
                        {
                            "key": "long_total_wallet_exposure_limit",
                            "value": 0.0,
                        },
                    ]
                },
            ],
        }
    }
    _overrides, fixed_bounds = _build_anchor_parameter_context(config, {})

    with pytest.raises(ValueError, match="remain positive"):
        _validate_directional_search_space(
            fixed_bounds,
            {},
            {"long": ["BTC"], "short": []},
            {"long"},
        )


def test_gpu_anchor_checkpoint_signature_tracks_ordered_fixed_values():
    active = [(ANCHOR_GENE_KEY, 0, Bound(0.0, 1.0, 1.0))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    plan = {
        "fixed_keys": ["long_base_qty_pct", "long_offset_psize_weight"],
        "tunable_keys": ["long_offset"],
        "anchors": [
            {
                "fixed_values": [
                    {"key": "long_base_qty_pct", "value": 0.1},
                    {"key": "long_offset_psize_weight", "value": 0.2},
                ]
            },
            {
                "fixed_values": [
                    {"key": "long_base_qty_pct", "value": 0.3},
                    {"key": "long_offset_psize_weight", "value": 0.4},
                ]
            },
        ],
    }
    original = _checkpoint_signature(active, scoring, anchor_plan=plan)

    edited = copy.deepcopy(plan)
    edited["anchors"][1]["fixed_values"][0]["value"] = 0.31
    reordered_anchors = copy.deepcopy(plan)
    reordered_anchors["anchors"].reverse()
    reordered_items = copy.deepcopy(plan)
    reordered_items["anchors"][0]["fixed_values"].reverse()

    assert _checkpoint_signature(active, scoring, anchor_plan=edited) != original
    assert (
        _checkpoint_signature(active, scoring, anchor_plan=reordered_anchors)
        != original
    )
    assert _checkpoint_signature(active, scoring, anchor_plan=reordered_items) == original


def test_gpu_checkpoint_signature_tracks_effective_suite_contract():
    active = [("long_base_qty_pct", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    suite = {
        "suite_enabled": True,
        "scenarios": [{"label": "base", "coins": ["BTC"]}],
        "reducer": {"default": "mean"},
        "exchanges": ["bybit"],
        "volume_normalization": True,
    }
    original = _checkpoint_signature(active, scoring, suite_contract=suite)
    changed_scenario = copy.deepcopy(suite)
    changed_scenario["scenarios"][0]["coins"] = ["ETH"]
    changed_reducer = copy.deepcopy(suite)
    changed_reducer["reducer"]["default"] = "min"
    changed_date = copy.deepcopy(suite)
    changed_date["scenarios"][0]["end_date"] = "2026-08-19"

    assert _checkpoint_signature(active, scoring) != original
    assert (
        _checkpoint_signature(active, scoring, suite_contract=changed_scenario)
        != original
    )
    assert (
        _checkpoint_signature(active, scoring, suite_contract=changed_reducer)
        != original
    )
    assert (
        _checkpoint_signature(active, scoring, suite_contract=changed_date)
        != original
    )


def test_gpu_checkpoint_signature_tracks_realized_loss_gate_contract():
    active = [("long_offset", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    config = _long_only_ema_config()
    proxy = SimpleNamespace(coin_override_contract={"values": []})
    original_contract = _gpu_runtime_checkpoint_contract(config, proxy)
    original = _checkpoint_signature(
        active, scoring, runtime_contract=original_contract
    )

    config["live"]["max_realized_loss_pct"] = 0.05
    changed_contract = _gpu_runtime_checkpoint_contract(config, proxy)

    assert changed_contract["max_realized_loss_pct"] == 0.05
    assert (
        _checkpoint_signature(active, scoring, runtime_contract=changed_contract)
        != original
    )


def test_gpu_checkpoint_signature_tracks_exact_coin_override_precision():
    active = [("long_offset", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    config = _long_only_ema_config()
    first = 0.4
    second = float(np.nextafter(first, 1.0))
    packed = float(np.float32(first))
    assert packed == float(np.float32(second))

    def signature(exact_value):
        proxy = SimpleNamespace(
            coin_override_contract={
                "values": [[packed]],
                "exact_overrides": [
                    {
                        "bot": {
                            "long": {
                                "strategy": {
                                    "ema_anchor": {"offset": exact_value}
                                }
                            }
                        }
                    }
                ],
            }
        )
        return _checkpoint_signature(
            active,
            scoring,
            runtime_contract=_gpu_runtime_checkpoint_contract(config, proxy),
        )

    assert signature(first) != signature(second)


def test_gpu_checkpoint_signature_tracks_hedge_mode_contract():
    active = [("long_offset", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    config = _directional_ema_config(long_enabled=True, short_enabled=True)
    proxy = SimpleNamespace(coin_override_contract={"values": []})
    original_contract = _gpu_runtime_checkpoint_contract(config, proxy)
    original = _checkpoint_signature(
        active, scoring, runtime_contract=original_contract
    )

    changed = copy.deepcopy(config)
    changed["live"]["hedge_mode"] = not original_contract["hedge_mode"]
    changed_contract = _gpu_runtime_checkpoint_contract(changed, proxy)

    assert changed_contract["hedge_mode"] is not original_contract["hedge_mode"]
    assert (
        _checkpoint_signature(active, scoring, runtime_contract=changed_contract)
        != original
    )


def test_gpu_checkpoint_signature_tracks_single_coin_unstuck_contract():
    active = [("long_offset", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    config = _long_only_ema_config()
    config["bot"]["long"]["unstuck"]["enabled"] = True
    proxy = SimpleNamespace(coin_override_contract=None)
    original_contract = _gpu_runtime_checkpoint_contract(config, proxy)
    original = _checkpoint_signature(
        active, scoring, runtime_contract=original_contract
    )

    edits = {
        "enabled": False,
        "ema_gating_enabled": False,
        "close_pct": 0.234,
        "ema_dist": -0.012,
        "loss_allowance_pct": 0.034,
        "threshold": 0.876,
    }
    for key, value in edits.items():
        changed = copy.deepcopy(config)
        changed["bot"]["long"]["unstuck"][key] = value
        changed_contract = _gpu_runtime_checkpoint_contract(changed, proxy)
        assert changed_contract != original_contract
        assert (
            _checkpoint_signature(
                active, scoring, runtime_contract=changed_contract
            )
            != original
        )

    changed_lookback = copy.deepcopy(config)
    changed_lookback["live"]["pnls_max_lookback_days"] = 7.0
    changed_lookback_contract = _gpu_runtime_checkpoint_contract(
        changed_lookback, proxy
    )
    assert changed_lookback_contract["pnls_max_lookback_days"] == 7.0
    assert (
        _checkpoint_signature(
            active, scoring, runtime_contract=changed_lookback_contract
        )
        != original
    )

    all_history = copy.deepcopy(config)
    all_history["live"]["pnls_max_lookback_days"] = "all"
    assert _gpu_runtime_checkpoint_contract(all_history, proxy)[
        "pnls_max_lookback_days"
    ] == -1.0

    fixed = _long_only_ema_config()
    fixed["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.unstuck.enabled": True,
        "bot.long.unstuck.threshold": 0.765,
        "live.pnls_max_lookback_days": 12.0,
    }
    effective = _materialize_gpu_override_template(fixed, [])
    effective_contract = _gpu_runtime_checkpoint_contract(effective, proxy)
    assert effective_contract["unstuck"]["long"]["enabled"] is True
    assert effective_contract["unstuck"]["long"]["threshold"] == 0.765
    assert effective_contract["pnls_max_lookback_days"] == 12.0


def test_gpu_checkpoint_signature_tracks_single_coin_hsl_contract():
    active = [("long_offset", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    config = _long_only_ema_config()
    config["bot"]["long"]["hsl"]["enabled"] = True
    proxy = SimpleNamespace(coin_override_contract=None)
    original_contract = _gpu_runtime_checkpoint_contract(config, proxy)
    original = _checkpoint_signature(
        active, scoring, runtime_contract=original_contract
    )

    edits = (
        ("live", "hsl_signal_mode", "unified"),
        ("backtest", "dynamic_wel_by_tradability", False),
        ("bot.long.risk", "n_positions", 2),
        ("bot.long.hsl", "enabled", False),
        ("bot.long.hsl", "restart_after_red_policy", "never"),
        ("bot.long.hsl", "no_restart_drawdown_threshold", 0.9),
        ("bot.long.hsl", "tier_ratio_yellow", 0.4),
        ("bot.long.hsl", "orange_tier_mode", "graceful_stop"),
        ("bot.long.hsl", "panic_close_order_type", "market"),
    )
    for parent_path, key, value in edits:
        changed = copy.deepcopy(config)
        parent = changed
        for part in parent_path.split("."):
            parent = parent[part]
        parent[key] = value
        changed_contract = _gpu_runtime_checkpoint_contract(changed, proxy)
        assert changed_contract != original_contract
        assert (
            _checkpoint_signature(
                active, scoring, runtime_contract=changed_contract
            )
            != original
        )

    pinned = {"long_hsl_red_threshold": 0.2}
    pinned_contract = _gpu_runtime_checkpoint_contract(
        config, proxy, pinned_hsl_bounds=pinned
    )
    changed_pinned_contract = _gpu_runtime_checkpoint_contract(
        config,
        proxy,
        pinned_hsl_bounds={"long_hsl_red_threshold": 0.25},
    )
    assert pinned_contract != changed_pinned_contract
    assert _checkpoint_signature(
        active, scoring, runtime_contract=pinned_contract
    ) != _checkpoint_signature(
        active, scoring, runtime_contract=changed_pinned_contract
    )


def test_gpu_hsl_gene_activity_and_pinned_contract_helpers():
    assert not _gpu_hsl_parameter_active("long_hsl_red_threshold", set())
    assert _gpu_hsl_parameter_active("long_hsl_red_threshold", {"long"})
    assert _gpu_hsl_parameter_active("long_offset", set())
    assert _gpu_pinned_hsl_bound_contract(
        {
            "long_hsl_red_threshold": Bound(0.2, 0.2),
            "long_hsl_ema_span_minutes": Bound(30.0, 90.0),
            "long_offset": Bound(0.01, 0.01),
        }
    ) == {"long_hsl_red_threshold": 0.2}

    config = _long_only_ema_config()
    config["bot"]["long"]["hsl"]["enabled"] = True
    _validate_hsl_bound_contracts(
        {
            "long_hsl_enabled": Bound(1.0, 1.0),
            "long_hsl_red_threshold": Bound(0.2, 0.8),
        },
        config,
    )
    with pytest.raises(ValueError, match="enablement to match"):
        _validate_hsl_bound_contracts(
            {"long_hsl_enabled": Bound(0.0, 0.0)}, config
        )
    with pytest.raises(ValueError, match="cannot distinguish from 1.0"):
        _validate_hsl_bound_contracts(
            {"long_hsl_red_threshold": Bound(0.9, 1.0)}, config
        )
    with pytest.raises(ValueError, match="red_threshold bounds must remain greater"):
        _validate_hsl_bound_contracts(
            {"long_hsl_red_threshold": Bound(-0.1, 0.2)}, config
        )
    with pytest.raises(ValueError, match="cooldown_minutes_after_red bounds"):
        _validate_hsl_bound_contracts(
            {
                "long_hsl_cooldown_minutes_after_red": Bound(-10.0, 10.0)
            },
            config,
        )

    override_enabled = _long_only_ema_config()
    override_enabled["coin_overrides"] = {
        "ETH": {"bot": {"long": {"hsl": {"enabled": True}}}}
    }
    with pytest.raises(ValueError, match="red_threshold bounds must remain greater"):
        _validate_hsl_bound_contracts(
            {"long_hsl_red_threshold": Bound(-0.1, 0.2)},
            override_enabled,
        )
    assert _gpu_hsl_search_sides(override_enabled, [], set()) == {"long"}

    base = _long_only_ema_config()
    scenario = copy.deepcopy(base)
    scenario["bot"]["long"]["hsl"]["enabled"] = True
    search_sides = _gpu_hsl_search_sides(
        base, [{"config": scenario}], set()
    )
    assert search_sides == {"long"}
    assert _gpu_hsl_parameter_active(
        "long_hsl_no_restart_drawdown_threshold", search_sides
    )

    switched = copy.deepcopy(base)
    switched["bot"]["long"]["risk"]["n_positions"] = 0
    switched["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.0
    switched["bot"]["short"]["risk"]["n_positions"] = 1
    switched["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 1.0
    switched["bot"]["short"]["hsl"]["enabled"] = True
    switched["live"]["approved_coins"] = {"long": [], "short": ["BTC"]}
    assert _gpu_candidate_search_sides(
        base, [{"config": switched}]
    ) == {"short"}


def test_gpu_checkpoint_signature_tracks_prepared_coin_override_contract():
    active = [("long_offset", 0, Bound(0.01, 0.1, 0.01))]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    contract = {
        "exchange": "bybit",
        "coins": ["BTC", "ETH"],
        "side": "long",
        "values": [[None] * 12, [None] * 11 + [0.4]],
    }
    original = _checkpoint_signature(
        active, scoring, runtime_contract=contract
    )
    edited = copy.deepcopy(contract)
    edited["values"][1][11] = 0.5

    assert _checkpoint_signature(active, scoring) != original
    assert (
        _checkpoint_signature(active, scoring, runtime_contract=edited)
        != original
    )
    hysteresis_edited = copy.deepcopy(contract)
    hysteresis_edited["forager_score_hysteresis_pct"] = 0.02
    assert (
        _checkpoint_signature(
            active, scoring, runtime_contract=hysteresis_edited
        )
        != original
    )


def test_gpu_checkpoint_signature_tracks_dual_side_coin_override_contract():
    active = [
        ("long_offset", 0, Bound(0.01, 0.1, 0.01)),
        ("short_offset", 1, Bound(0.01, 0.1, 0.01)),
    ]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    contract = {
        "exchange": "bybit",
        "coins": ["BTC", "ETH"],
        "sides": ["long", "short"],
        "values_by_side": {
            "long": [[None] * 12, [None] * 11 + [0.4]],
            "short": [[None] * 12, [None] * 10 + [30.0, None]],
        },
        "proxy_mode": "independent-side-hedge-v1",
    }
    original = _checkpoint_signature(active, scoring, runtime_contract=contract)
    edited = copy.deepcopy(contract)
    edited["values_by_side"]["short"][1][10] = 45.0

    assert (
        _checkpoint_signature(active, scoring, runtime_contract=edited)
        != original
    )


def test_gpu_suite_checkpoint_contract_tracks_prepared_scenario_identity():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    item = {
        "ctx": SimpleNamespace(label="stress"),
        "exchange": "bybit",
        "coins": ["BTC", "ETH"],
        "coin_count": 2,
        "config": config,
        "hlcvs": np.zeros((3, 2, 4)),
        "timestamps": np.array([1000, 2000, 3000]),
        "mss": {
            "BTC": {"exchange": "binance", "ohlcv_source": "bybit"},
            "ETH": {"exchange": "bybit"},
        },
    }
    original = _gpu_suite_checkpoint_contract(config, [item])

    assert original["prepared_scenarios"] == [
        {
            "label": "stress",
            "exchange": "bybit",
            "coins": ["BTC", "ETH"],
            "coin_count": 2,
            "strategy_kind": "ema_anchor",
            "enabled_sides": ["long"],
            "hedge_mode": False,
            "max_realized_loss_pct": 1.0,
            "pnls_max_lookback_days": 30.0,
            "unstuck": original["unstuck"],
            "hsl": original["hsl"],
            "pinned_hsl_bounds": {},
            "scenario_fixed_bound_values": {},
            "scenario_parameter_overrides": {},
            "proxy_execution": {},
            "candle_count": 3,
            "first_timestamp": 1000,
            "last_timestamp": 3000,
            "coin_sources": [
                {
                    "coin": "BTC",
                    "ohlcv_exchange": "bybit",
                    "market_settings_exchange": "binance",
                },
                {
                    "coin": "ETH",
                    "ohlcv_exchange": "bybit",
                    "market_settings_exchange": "bybit",
                },
            ],
            "coin_overrides": {},
        }
    ]

    second_exchange = copy.deepcopy(item)
    second_exchange["exchange"] = "binance"
    second_exchange["mss"]["BTC"] = {"exchange": "binance"}
    second_exchange["mss"]["ETH"] = {"exchange": "binance"}
    multi_exchange = _gpu_suite_checkpoint_contract(
        config, [item, second_exchange]
    )
    assert [
        (entry["label"], entry["exchange"])
        for entry in multi_exchange["prepared_scenarios"]
    ] == [("stress", "bybit"), ("stress", "binance")]
    assert multi_exchange != original

    changed_second_exchange = copy.deepcopy(second_exchange)
    changed_second_exchange["timestamps"] = np.array([1000, 2500, 3500])
    assert (
        _gpu_suite_checkpoint_contract(
            config, [item, changed_second_exchange]
        )
        != multi_exchange
    )

    changed_loss_gate = copy.deepcopy(item)
    changed_loss_gate["config"]["live"]["max_realized_loss_pct"] = 0.05
    changed = _gpu_suite_checkpoint_contract(config, [changed_loss_gate])

    assert changed != original
    assert changed["prepared_scenarios"][0]["max_realized_loss_pct"] == 0.05

    changed_hedge_mode = copy.deepcopy(item)
    changed_hedge_mode["config"]["live"]["hedge_mode"] = True
    changed = _gpu_suite_checkpoint_contract(config, [changed_hedge_mode])
    assert changed != original
    assert changed["prepared_scenarios"][0]["hedge_mode"] is True

    changed_base_hedge_mode = copy.deepcopy(config)
    changed_base_hedge_mode["live"]["hedge_mode"] = True
    changed = _gpu_suite_checkpoint_contract(changed_base_hedge_mode, [item])
    assert changed != original
    assert changed["hedge_mode"] is True

    changed_lookback = copy.deepcopy(item)
    changed_lookback["config"]["live"]["pnls_max_lookback_days"] = 7.0
    changed = _gpu_suite_checkpoint_contract(config, [changed_lookback])
    assert changed != original
    assert changed["prepared_scenarios"][0]["pnls_max_lookback_days"] == 7.0

    changed_base_lookback = copy.deepcopy(config)
    changed_base_lookback["live"]["pnls_max_lookback_days"] = "all"
    changed = _gpu_suite_checkpoint_contract(changed_base_lookback, [item])
    assert changed != original
    assert changed["pnls_max_lookback_days"] == -1.0

    changed_unstuck = copy.deepcopy(item)
    changed_unstuck["config"]["bot"]["long"]["unstuck"]["enabled"] = True
    assert _gpu_suite_checkpoint_contract(config, [changed_unstuck]) != original

    changed_hsl = copy.deepcopy(item)
    changed_hsl["config"]["bot"]["long"]["hsl"]["enabled"] = True
    assert _gpu_suite_checkpoint_contract(config, [changed_hsl]) != original

    changed_pinned_hsl = copy.deepcopy(item)
    changed_pinned_hsl["pinned_hsl_bounds"] = {
        "long_hsl_red_threshold": 0.25
    }
    assert (
        _gpu_suite_checkpoint_contract(config, [changed_pinned_hsl])
        != original
    )

    changed_proxy_execution = copy.deepcopy(item)
    changed_proxy_execution["proxy_checkpoint_contract"] = {
        "backtest": {"starting_balance": 20_000.0}
    }
    assert (
        _gpu_suite_checkpoint_contract(config, [changed_proxy_execution])
        != original
    )
    assert (
        _gpu_suite_checkpoint_contract(
            config,
            [item],
            pinned_hsl_bounds={"long_hsl_red_threshold": 0.25},
        )
        != original
    )

    changed_coins = copy.deepcopy(item)
    changed_coins["coins"] = ["BTC", "SOL"]
    changed_window = copy.deepcopy(item)
    changed_window["timestamps"] = np.array([1000, 2000, 4000])
    changed_source = copy.deepcopy(item)
    changed_source["mss"]["BTC"]["ohlcv_source"] = "binance"

    assert _gpu_suite_checkpoint_contract(config, [changed_coins]) != original
    assert _gpu_suite_checkpoint_contract(config, [changed_window]) != original
    assert _gpu_suite_checkpoint_contract(config, [changed_source]) != original


def test_gpu_checkpoint_signature_tracks_full_fixed_search_contract():
    key_paths = [
        ("long_offset", ("bot", "long", "strategy", "ema_anchor", "offset")),
        (
            "long_base_qty_pct",
            ("bot", "long", "strategy", "ema_anchor", "base_qty_pct"),
        ),
    ]
    bounds = [Bound(0.01, 0.1, 0.01), Bound(0.02, 0.02, 0.01)]
    base = [0.04, 0.02]
    contract = _gpu_search_checkpoint_contract(
        key_paths=key_paths,
        bounds=bounds,
        base_vector=base,
        fixed_bound_values={"long_base_qty_pct": 0.02},
        fixed_parameter_overrides={"long_base_qty_pct": 0.02},
        optimizer_overrides=set(),
        sig_digits=3,
        algorithm_contract={
            "algorithm": "nsga2",
            "population_size": 64,
            "mutation": {"prob": 0.5},
        },
        proxy_evaluation_policy={
            "enabled": False,
            "history_fractions": [0.25, 0.5, 1.0],
            "history_window": "recent_suffix_v1",
        },
    )
    assert contract["version"] == 2
    assert contract["proxy_evaluation"]["enabled"] is False
    assert contract["proxy_evaluation"]["history_window"] == "recent_suffix_v1"
    ordinary_contract = _gpu_search_checkpoint_contract(
        key_paths=key_paths,
        bounds=bounds,
        base_vector=base,
        fixed_bound_values={"long_base_qty_pct": 0.02},
        fixed_parameter_overrides={"long_base_qty_pct": 0.02},
        optimizer_overrides=set(),
        sig_digits=3,
        algorithm_contract=contract["algorithm"],
    )
    assert ordinary_contract["version"] == 1
    assert "proxy_evaluation" not in ordinary_contract
    active = [("long_offset", 0, bounds[0])]
    scoring = [{"goal": "max", "metric": "adg_strategy_eq"}]
    original = _checkpoint_signature(
        active, scoring, search_contract=contract
    )

    mutations = []
    changed_base = copy.deepcopy(contract)
    changed_base["dimensions"][1]["base"] = 0.03
    mutations.append(changed_base)
    changed_fixed = copy.deepcopy(contract)
    changed_fixed["fixed_parameter_overrides"]["long_base_qty_pct"] = 0.03
    mutations.append(changed_fixed)
    changed_digits = copy.deepcopy(contract)
    changed_digits["sig_digits"] = 4
    mutations.append(changed_digits)
    changed_override = copy.deepcopy(contract)
    changed_override["optimizer_overrides"] = ["mirror_short_from_long"]
    mutations.append(changed_override)
    changed_population = copy.deepcopy(contract)
    changed_population["algorithm"]["population_size"] = 128
    mutations.append(changed_population)
    changed_mutation = copy.deepcopy(contract)
    changed_mutation["algorithm"]["mutation"]["prob"] = 0.25
    mutations.append(changed_mutation)
    changed_proxy_policy = copy.deepcopy(contract)
    changed_proxy_policy["proxy_evaluation"]["enabled"] = True
    mutations.append(changed_proxy_policy)
    changed_history_window = copy.deepcopy(contract)
    changed_history_window["proxy_evaluation"]["history_window"] = (
        "historical_prefix_v1"
    )
    mutations.append(changed_history_window)
    changed_seed_bootstrap = _gpu_search_checkpoint_contract(
        key_paths=key_paths,
        bounds=bounds,
        base_vector=base,
        fixed_bound_values={"long_base_qty_pct": 0.02},
        fixed_parameter_overrides={"long_base_qty_pct": 0.02},
        optimizer_overrides=set(),
        sig_digits=3,
        algorithm_contract=contract["algorithm"],
        proxy_evaluation_policy=contract["proxy_evaluation"],
        seed_bootstrap_contract={
            "version": 1,
            "requested_mode": "auto",
            "effective_mode": "screened",
            "max_exact": 128,
            "seed_count": 1000,
            "selected_exact_count": 128,
            "all_seeds_exact": False,
            "seed_pool_sha256": "abc123",
        },
    )
    assert changed_seed_bootstrap["version"] == 3
    mutations.append(changed_seed_bootstrap)

    assert all(
        _checkpoint_signature(active, scoring, search_contract=changed)
        != original
        for changed in mutations
    )


def test_gpu_runtime_checkpoint_contract_tracks_prepared_proxy_execution():
    config = _long_only_ema_config()
    proxy = SimpleNamespace(
        coin_override_contract=None,
        checkpoint_contract={
            "backtest": {"starting_balance": 10_000.0},
            "markets": [{"maker_fee": 0.0004}],
        },
    )
    original = _gpu_runtime_checkpoint_contract(config, proxy)
    changed_proxy = copy.deepcopy(proxy)
    changed_proxy.checkpoint_contract["markets"][0]["maker_fee"] = 0.0005

    assert _gpu_runtime_checkpoint_contract(config, changed_proxy) != original


def test_gpu_suite_checkpoint_contract_rejects_timestamp_shape_mismatch():
    config = _directional_ema_config(long_enabled=True, short_enabled=False)
    item = {
        "ctx": SimpleNamespace(label="stress"),
        "exchange": "bybit",
        "coins": ["BTC", "ETH"],
        "coin_count": 2,
        "config": config,
        "hlcvs": np.zeros((3, 2, 4)),
        "timestamps": np.array([1000, 2000]),
    }

    with pytest.raises(ValueError, match="timestamp identity mismatch"):
        _gpu_suite_checkpoint_contract(config, [item])


def test_gpu_rejects_pinned_unsupported_exposure_repair_behavior():
    from optimization.bounds import Bound

    with pytest.raises(ValueError, match="position_exposure_enforcer_enabled"):
        _validate_pinned_scope_bounds(
            {
                "long_risk_position_exposure_enforcer_enabled": Bound(
                    1.0, 1.0, None
                )
            },
            {"long_risk_position_exposure_enforcer_enabled": 1.0},
            coin_count=2,
        )

    _validate_pinned_scope_bounds(
        {
            "long_risk_position_exposure_enforcer_enabled": Bound(
                1.0, 1.0, None
            ),
            "long_risk_wel_enforcer_threshold": Bound(0.5, 1.0, None),
        },
        {"long_risk_position_exposure_enforcer_enabled": 1.0},
        {"long"},
        coin_count=2,
        strategy_kind="trailing_martingale",
    )

    _validate_pinned_scope_bounds(
        {"long_risk_total_exposure_enforcer_enabled": Bound(1.0, 1.0, None)},
        {"long_risk_total_exposure_enforcer_enabled": 1.0},
        {"long"},
        coin_count=2,
        strategy_kind="ema_anchor",
    )

    _validate_pinned_scope_bounds(
        {"long_risk_total_exposure_enforcer_enabled": Bound(1.0, 1.0, None)},
        {"long_risk_total_exposure_enforcer_enabled": 1.0},
        {"long"},
        coin_count=1,
        strategy_kind="ema_anchor",
    )

    _validate_pinned_scope_bounds(
        {"long_risk_total_exposure_enforcer_enabled": Bound(1.0, 1.0, None)},
        {"long_risk_total_exposure_enforcer_enabled": 1.0},
        {"long"},
        coin_count=2,
        strategy_kind="trailing_martingale",
    )

    for strategy_kind in ("trailing_martingale", "ema_anchor"):
        _validate_pinned_scope_bounds(
            {
                "long_risk_total_exposure_enforcer_enabled": Bound(
                    0.0, 1.0, None
                )
            },
            {"long_risk_total_exposure_enforcer_enabled": 0.0},
            {"long", "short"},
            coin_count=2,
            strategy_kind=strategy_kind,
        )


def test_gpu_accepts_single_coin_exposure_policy_bounds():
    _validate_pinned_scope_bounds(
        {
            "long_risk_we_excess_allowance_pct": Bound(0.0, 0.5, None),
            "long_risk_twel_enforcer_threshold": Bound(0.5, 1.0, None),
        },
        {},
        {"long"},
        coin_count=1,
    )


def test_gpu_accepts_unstuck_bounds_for_single_and_dual_multicoin():
    from optimization.bounds import Bound

    bounds = {
        "long_unstuck_enabled": Bound(1.0, 1.0, None),
        "long_unstuck_close_pct": Bound(0.01, 0.2, None),
        "long_unstuck_ema_dist": Bound(-0.05, 0.05, None),
        "long_unstuck_loss_allowance_pct": Bound(0.01, 0.1, None),
        "long_unstuck_threshold": Bound(0.5, 0.95, None),
    }
    base = {"long_unstuck_enabled": 1.0}

    _validate_pinned_scope_bounds(bounds, base, {"long"}, coin_count=1)
    _validate_pinned_scope_bounds(bounds, base, {"long"}, coin_count=2)
    _validate_pinned_scope_bounds(
        bounds, base, {"long", "short"}, coin_count=2
    )


def test_gpu_accepts_hsl_bounds_for_single_and_dual_multicoin():
    bounds = {
        "long_hsl_enabled": Bound(1.0, 1.0, None),
        "long_hsl_red_threshold": Bound(0.05, 0.2, None),
        "long_hsl_ema_span_minutes": Bound(1.0, 720.0, None),
        "long_hsl_cooldown_minutes_after_red": Bound(0.0, 1440.0, None),
    }
    base = {"long_hsl_enabled": 1.0}

    _validate_pinned_scope_bounds(bounds, base, {"long"}, coin_count=2)
    _validate_pinned_scope_bounds(
        bounds, base, {"long", "short"}, coin_count=2
    )


def test_gpu_anchor_constant_twel_threshold_is_supported_for_multicoin():
    config = {
        ANCHOR_PLAN_KEY: {
            "fixed_keys": ["long_risk_twel_enforcer_threshold"],
            "anchors": [
                {
                    "fixed_values": [
                        {
                            "key": "long_risk_twel_enforcer_threshold",
                            "value": 0.8,
                        }
                    ]
                },
                {
                    "fixed_values": [
                        {
                            "key": "long_risk_twel_enforcer_threshold",
                            "value": 0.8,
                        }
                    ]
                },
            ],
        }
    }

    _overrides, fixed_bounds = _build_anchor_parameter_context(config, {})

    _validate_pinned_scope_bounds(fixed_bounds, {}, {"long"}, coin_count=2)


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


@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_multicoin_search_space_allows_bounded_n_positions(side):
    other = "short" if side == "long" else "long"
    approved = {side: ["BTC", "ETH", "SOL"], other: []}
    base = {
        f"{side}_total_wallet_exposure_limit": 1.0,
        f"{side}_n_positions": 2.0,
        f"{other}_total_wallet_exposure_limit": 0.0,
        f"{other}_n_positions": 0.0,
    }
    bounds = {
        f"{side}_total_wallet_exposure_limit": Bound(0.5, 1.5, None),
        f"{side}_n_positions": Bound(1.0, 3.0, 1.0),
        f"{other}_total_wallet_exposure_limit": Bound(0.0, 0.0, None),
        f"{other}_n_positions": Bound(0.0, 0.0, None),
    }

    _validate_directional_search_space(
        bounds, base, approved, {side}, coin_count=3
    )
    bounds[f"{side}_n_positions"] = Bound(1.0, 4.0, 1.0)
    with pytest.raises(ValueError, match=r"within \[1, 3\]"):
        _validate_directional_search_space(
            bounds, base, approved, {side}, coin_count=3
        )

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
            "scenario": None,
            "reducer": None,
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
    assert "scenario=None reducer=None" in detail


def test_constraint_diagnostics_reads_suite_reducers_and_scenarios():
    checks = [
        {
            "metric": "drawdown_worst_strategy_eq",
            "metric_key": "drawdown_worst_strategy_eq_max",
            "mode": "greater_than",
            "bound": 0.3,
            "penalty_weight": 1.0,
            "reducer": "max",
            "scenario": None,
        },
        {
            "metric": "adg_strategy_eq",
            "metric_key": "adg_strategy_eq_mean",
            "mode": "less_than",
            "bound": 0.002,
            "penalty_weight": 1.0,
            "reducer": "mean",
            "scenario": "stress",
        },
    ]
    evaluator = type("Evaluator", (), {"limit_checks": checks})()
    proxy_suite = {
        "metrics": {
            "drawdown_worst_strategy_eq": {
                "stats": {"max": 0.2},
                "scenarios": {"stress": 0.2},
            },
            "adg_strategy_eq": {
                "stats": {"mean": 0.003},
                "scenarios": {"stress": 0.003},
            },
        }
    }
    exact_suite = {
        "metrics": {
            "drawdown_worst_strategy_eq": {
                "stats": {"max": 0.4},
                "scenarios": {"stress": 0.4},
            },
            "adg_strategy_eq": {
                "stats": {"mean": 0.001},
                "scenarios": {"stress": 0.001},
            },
        }
    }

    diagnostics = _constraint_diagnostics(
        evaluator,
        {_GPU_SUITE_METRICS_KEY: proxy_suite},
        {"metrics": {"suite_metrics": exact_suite}},
    )

    assert [(item["proxy_value"], item["exact_value"]) for item in diagnostics] == [
        (0.2, 0.4),
        (0.003, 0.001),
    ]
    assert [item["exact_violation"] for item in diagnostics] == pytest.approx(
        [0.1, 0.001]
    )
    detail = _format_constraint_diagnostics(diagnostics)
    assert "scenario=None reducer=max" in detail
    assert "scenario=stress reducer=mean" in detail


def test_constraint_diagnostics_preserve_invalid_exact_suite_penalty():
    check = {
        "metric": "backtest_completion_ratio",
        "metric_key": "backtest_completion_ratio_min",
        "mode": "less_than",
        "bound": 0.99,
        "penalty_weight": 1.0,
        "reducer": "min",
        "scenario": None,
    }
    evaluator = type("Evaluator", (), {"limit_checks": [check]})()
    diagnostics = _constraint_diagnostics(
        evaluator,
        {
            _GPU_SUITE_METRICS_KEY: {
                "metrics": {
                    "backtest_completion_ratio": {
                        "stats": {"min": 1.0},
                        "scenarios": {"base": 1.0},
                    }
                }
            }
        },
        {
            "metrics": {
                "suite_metrics": {},
                "constraint_violation": 1.0e18,
                "error": "recoverable Rust failure",
            },
            "G": np.asarray([1.0e18]),
        },
    )

    assert diagnostics[0]["exact_value"] is None
    assert diagnostics[0]["exact_violation"] is None
    assert diagnostics[0]["exact_failure_penalty"] == 1.0e18
    assert "exact_failure_penalty=1e+18" in _format_constraint_diagnostics(
        diagnostics
    )


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


def test_resume_recovers_exact_and_screened_seed_bootstrap_results():
    entries = [
        {
            "id": 7,
            "metrics": {
                "gpu_seed_bootstrap": {
                    "schema_version": 1,
                    "mode": "exact",
                    "source_index": 3,
                    "exact_objectives": [0.1, 0.2],
                    "exact_violation": -1.0,
                }
            },
        },
        {
            "id": 8,
            "metrics": {
                "gpu_seed_bootstrap": {
                    "schema_version": 1,
                    "mode": "screened",
                    "source_index": 5,
                    "exact_objectives": [0.3, 0.4],
                    "exact_violation": 0.5,
                },
                "gpu_validation": {
                    "schema_version": 2,
                    "phase": "seed_bootstrap",
                    "proxy_score": 0.25,
                    "exact_score": 0.35,
                    "probe": True,
                    "proxy_front": False,
                    "constraint_classification_mismatch": True,
                },
            },
        },
    ]

    payloads, recovered, drift_pairs = _recover_durable_seed_bootstrap(
        entries,
        start_index=0,
        stop_index=2,
        vector_from_entry=lambda entry: [float(entry["id"])],
        hash_vector=lambda vector: f"hash-{vector[0]}",
    )

    assert recovered == {"hash-7.0", "hash-8.0"}
    assert payloads["hash-7.0"] == {
        "source_index": 3,
        "F": [0.1, 0.2],
        "G": [-1.0],
    }
    assert payloads["hash-8.0"]["source_index"] == 5
    assert drift_pairs == [(0.25, 0.35, True, True, False)]


def test_resume_rejects_screened_seed_without_proxy_exact_evidence():
    with pytest.raises(RuntimeError, match="without proxy/exact metadata"):
        _recover_durable_seed_bootstrap(
            [
                {
                    "id": 1,
                    "metrics": {
                        "gpu_seed_bootstrap": {
                            "schema_version": 1,
                            "mode": "screened",
                            "source_index": 0,
                            "exact_objectives": [0.1],
                            "exact_violation": -1.0,
                        }
                    },
                }
            ],
            start_index=0,
            stop_index=1,
            vector_from_entry=lambda entry: [float(entry["id"])],
            hash_vector=lambda vector: f"hash-{vector[0]}",
        )


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
