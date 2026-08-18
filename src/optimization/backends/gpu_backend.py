from __future__ import annotations

from collections import deque
from copy import deepcopy
import functools
import hashlib
import json
import logging
import math
import multiprocessing
import os
import pickle
import time
from typing import Any

import numpy as np

from config.metrics import resolve_metric_value
from limit_utils import compute_limit_violation
from metrics_schema import flatten_metric_stats
from optimization.backend_shared import load_starting_individuals
from optimization.bounds import Bound, enforce_bounds
from optimization.callback import build_pymoo_record_entry
from optimization.fine_tune_anchors import ANCHOR_GENE_KEY, get_anchor_plan
from optimization.gpu.model import gpu_side_enabled
from optimization.problem import (
    PymooAsyncRecordingRunner,
    PymooEvaluatorAdapter,
    _evaluate_pymoo_worker_from_globals,
    initialize_pymoo_worker,
)


GPU_DEFAULTS = {
    "population_size": 4096,
    "batch_size": 4096,
    "checkpoint_interval_seconds": 5.0,
    "validate_per_generation": 8,
    "drift_probes": 4,
    "drift_window": 128,
    "drift_min_samples": 32,
    "drift_halt": 0.60,
    "exact_workers": 0,
    "max_pending_exact": 0,
}

MIN_DRIFT_PROBES = 8

MAX_NOVELTY_STALL_GENERATIONS = 8

_EMA_SIDE_BOUND_SUFFIXES = {
    "base_qty_pct": "base_qty_pct",
    "ema_span_0": "ema_span_0",
    "ema_span_1": "ema_span_1",
    "entry_double_down_factor": "entry_double_down_factor",
    "offset": "offset",
    "offset_psize_weight": "offset_psize_weight",
    "offset_volatility_1h_weight": "offset_volatility_1h_weight",
    "offset_volatility_1m_weight": "offset_volatility_1m_weight",
    "offset_volatility_ema_span_1h": "offset_volatility_ema_span_1h",
    "offset_volatility_ema_span_1m": "offset_volatility_ema_span_1m",
    "risk_entry_cooldown_minutes": "entry_cooldown_minutes",
    "total_wallet_exposure_limit": "total_wallet_exposure_limit",
}

EMA_BOUND_MAP = {
    f"{side}_{bound_suffix}": f"{side}_{parameter}"
    for side in ("long", "short")
    for bound_suffix, parameter in _EMA_SIDE_BOUND_SUFFIXES.items()
}

_EMA_MULTICOIN_SIDE_BOUND_SUFFIXES = {
    "forager_volume_ema_span_1m": "forager_volume_ema_span_1m",
    "forager_volatility_ema_span_1m": "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct": "forager_volume_drop_pct",
    "forager_score_weights_volume": "forager_score_weights_volume",
    "forager_score_weights_ema_readiness": "forager_score_weights_ema_readiness",
    "forager_score_weights_volatility": "forager_score_weights_volatility",
    "n_positions": "n_positions",
}

EMA_MULTICOIN_BOUND_MAPS = {
    side: {
        **EMA_BOUND_MAP,
        **{
            f"{side}_{suffix}": f"{side}_{parameter}"
            for suffix, parameter in _EMA_MULTICOIN_SIDE_BOUND_SUFFIXES.items()
        },
    }
    for side in ("long", "short")
}

# Compatibility name for consumers of the first multicoin slice.
EMA_MULTICOIN_LONG_BOUND_MAP = EMA_MULTICOIN_BOUND_MAPS["long"]
EMA_MULTICOIN_SHORT_BOUND_MAP = EMA_MULTICOIN_BOUND_MAPS["short"]

_TM_SIDE_BOUND_SUFFIXES = {
    "ema_span_0": "ema_span_0",
    "ema_span_1": "ema_span_1",
    "volatility_ema_span_1h": "volatility_ema_span_1h",
    "volatility_ema_span_1m": "volatility_ema_span_1m",
    "entry_double_down_factor": "entry_double_down_factor",
    "entry_initial_ema_dist": "entry_initial_ema_dist",
    "entry_initial_qty_pct": "entry_initial_qty_pct",
    "entry_threshold_base_pct": "entry_threshold_base_pct",
    "entry_threshold_we_weight": "entry_threshold_we_weight",
    "entry_threshold_volatility_1h_weight": "entry_threshold_volatility_1h_weight",
    "entry_threshold_volatility_1m_weight": "entry_threshold_volatility_1m_weight",
    "entry_retracement_base_pct": "entry_retracement_base_pct",
    "entry_retracement_we_weight": "entry_retracement_we_weight",
    "entry_retracement_volatility_1h_weight": "entry_retracement_volatility_1h_weight",
    "entry_retracement_volatility_1m_weight": "entry_retracement_volatility_1m_weight",
    "close_qty_pct": "close_qty_pct",
    "close_threshold_base_pct": "close_threshold_base_pct",
    "close_threshold_we_weight": "close_threshold_we_weight",
    "close_threshold_volatility_1h_weight": "close_threshold_volatility_1h_weight",
    "close_threshold_volatility_1m_weight": "close_threshold_volatility_1m_weight",
    "close_retracement_base_pct": "close_retracement_base_pct",
    "close_retracement_volatility_1h_weight": "close_retracement_volatility_1h_weight",
    "close_retracement_volatility_1m_weight": "close_retracement_volatility_1m_weight",
    "risk_entry_cooldown_minutes": "entry_cooldown_minutes",
    "total_wallet_exposure_limit": "total_wallet_exposure_limit",
}

TRAILING_MARTINGALE_BOUND_MAP = {
    f"{side}_{bound_suffix}": f"{side}_{parameter}"
    for side in ("long", "short")
    for bound_suffix, parameter in _TM_SIDE_BOUND_SUFFIXES.items()
}

GPU_STRATEGY_BOUND_MAPS = {
    "ema_anchor": EMA_BOUND_MAP,
    "trailing_martingale": TRAILING_MARTINGALE_BOUND_MAP,
}

GPU_SUPPORTED_OPTIMIZER_OVERRIDES = {
    "lossless_close_trailing",
    "mirror_short_from_long",
}


def _validate_gpu_optimizer_overrides(overrides_list, strategy_kind: str) -> set[str]:
    overrides = set(overrides_list or [])
    unsupported = sorted(overrides - GPU_SUPPORTED_OPTIMIZER_OVERRIDES)
    if unsupported:
        raise ValueError(
            "GPU optimizer does not support optimize.enable_overrides values "
            f"{unsupported}"
        )
    if (
        "lossless_close_trailing" in overrides
        and strategy_kind != "trailing_martingale"
    ):
        raise ValueError(
            "GPU optimizer override lossless_close_trailing requires "
            "live.strategy_kind='trailing_martingale'"
        )
    return overrides


def _materialize_gpu_override_template(
    config: dict,
    overrides_list,
    *,
    finalize_fn=None,
) -> dict:
    """Apply exact runtime-finalization overrides to the proxy base config."""

    if not callable(finalize_fn):
        from optimization.warmup import _finalize_optimizer_vector_config

        finalize_fn = _finalize_optimizer_vector_config
    proxy_config = finalize_fn(
        deepcopy(config),
        overrides_list=overrides_list,
    )
    source_strategy_kind = str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
    effective_strategy_kind = (
        str(proxy_config.get("live", {}).get("strategy_kind", "")).strip().lower()
    )
    if effective_strategy_kind != source_strategy_kind:
        raise ValueError(
            "GPU optimize.fixed_runtime_overrides may not change live.strategy_kind; "
            "configure the strategy kind before optimization so the search shape and "
            "Metal kernel remain aligned"
        )
    return proxy_config


def _gpu_fixed_bound_context(
    config: dict,
    effective_config: dict,
    key_paths,
    bound_map: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve runtime-fixed optimizer bounds and their proxy parameter names."""

    from config.param_paths import (
        require_existing_config_path,
        resolve_optimizer_key_path,
    )
    from optimization.warmup import optimizer_dead_param_values

    path_to_bound_key = {
        tuple(path): bound_key for bound_key, path in key_paths
    }
    for bound_key in bound_map:
        path = resolve_optimizer_key_path(config, bound_key)
        if path is not None:
            path_to_bound_key[tuple(path)] = bound_key
    fixed_bound_values: dict[str, float] = {}
    fixed_parameters: dict[str, float] = {}
    fixed_overrides = (
        config.get("optimize", {}).get("fixed_runtime_overrides", {}) or {}
    )
    for dotted_path in fixed_overrides:
        resolved = require_existing_config_path(config, dotted_path)
        bound_key = path_to_bound_key.get(tuple(resolved))
        if bound_key is None:
            continue
        target = effective_config
        for part in resolved:
            target = target[part]
        try:
            value = float(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "GPU fixed runtime override for optimizer bound "
                f"{bound_key!r} must resolve to a numeric value"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                "GPU fixed runtime override for optimizer bound "
                f"{bound_key!r} must resolve to a finite value"
            )
        fixed_bound_values[bound_key] = value
        if bound_key in bound_map:
            fixed_parameters[bound_map[bound_key]] = value
    for bound_key, value in optimizer_dead_param_values(
        effective_config,
        globally_dead_only=True,
    ).items():
        fixed_bound_values[bound_key] = value
        if bound_key in bound_map:
            fixed_parameters[bound_map[bound_key]] = value
    return fixed_bound_values, fixed_parameters


def _mirror_short_mapping(mapping: dict) -> None:
    """Mirror effective long values/bounds into existing short-side keys."""

    for long_key, value in list(mapping.items()):
        if not long_key.startswith("long_"):
            continue
        short_key = f"short_{long_key[len('long_') :]}"
        if short_key in mapping:
            mapping[short_key] = value


def _apply_gpu_optimizer_overrides(
    parameters: dict[str, float],
    overrides: set[str],
) -> dict[str, float]:
    """Apply candidate-dependent V8 overrides without materializing Python configs."""

    if "mirror_short_from_long" in overrides:
        _mirror_short_mapping(parameters)
    if "lossless_close_trailing" in overrides:
        for side in ("long", "short"):
            threshold_key = f"{side}_close_threshold_base_pct"
            retracement_key = f"{side}_close_retracement_base_pct"
            if threshold_key in parameters and retracement_key in parameters:
                parameters[threshold_key] = max(
                    float(parameters[threshold_key]),
                    float(parameters[retracement_key]),
                )
    return parameters


def _gpu_candidate_source_sides(
    enabled_sides: set[str], overrides: set[str]
) -> set[str]:
    """Return sides whose genes can affect an enabled exact-trading side."""

    source_sides = set(enabled_sides)
    if "mirror_short_from_long" in overrides and "short" in enabled_sides:
        source_sides.add("long")
    return source_sides


def _ema_multicoin_bound_map(target_side: str, overrides: set[str]) -> dict:
    """Include all bound families that can feed the enabled multicoin side."""

    bound_map = dict(EMA_MULTICOIN_BOUND_MAPS[target_side])
    if "mirror_short_from_long" in overrides and target_side == "short":
        bound_map.update(EMA_MULTICOIN_BOUND_MAPS["long"])
    return bound_map


def _minimum_rank_evidence_samples(halt: float) -> int:
    """Total samples needed to guarantee eight comparable at agreement >= halt."""

    if not 0.0 < float(halt) <= 1.0:
        raise ValueError("GPU drift_halt must be greater than zero and at most one")
    return math.floor((MIN_DRIFT_PROBES - 1) / float(halt)) + 1


def _build_gpu_nsga2(config, *, sampling, population_size: int, n_params: int):
    """Build GPU proposal evolution with the same variation controls as pymoo CPU."""

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    from optimization.backends.pymoo_backend import (
        _resolve_mutation_prob,
        _resolve_pymoo_shared,
    )

    shared = _resolve_pymoo_shared(config)
    return NSGA2(
        pop_size=population_size,
        sampling=sampling,
        crossover=SBX(
            prob_var=float(shared["crossover_prob_var"]),
            eta=float(shared["crossover_eta"]),
        ),
        mutation=PM(
            prob=_resolve_mutation_prob(shared, n_params),
            eta=float(shared["mutation_eta"]),
        ),
        eliminate_duplicates=bool(shared["eliminate_duplicates"]),
    )

PINNED_SCOPE_BOUND_VALUES = {
    f"{side}_{suffix}": expected
    for side in ("long", "short")
    for suffix, expected in {
        "hsl_enabled": 0.0,
        "unstuck_enabled": 0.0,
        "risk_position_exposure_enforcer_enabled": 0.0,
        "risk_total_exposure_enforcer_enabled": 0.0,
        "risk_total_exposure_entry_gate_enabled": 1.0,
        "risk_twel_enforcer_threshold": 1.0,
        "risk_we_excess_allowance_pct": 0.0,
    }.items()
}

GPU_RESULT_BACKTEST_CONTRACT_KEYS = {
    "balance_sample_divider",
    "btc_collateral_cap",
    "btc_collateral_ltv_cap",
    "candle_interval_minutes",
    "coins",
    "dynamic_wel_by_tradability",
    "end_date",
    "exchanges",
    "filter_by_min_effective_cost",
    "liquidation_threshold",
    "maker_fee_override",
    "market_order_slippage_pct",
    "reducer",
    "scenarios",
    "start_date",
    "starting_balance",
    "suite_enabled",
    "taker_fee_override",
    "volume_normalization",
}

GPU_RESULT_OPTIMIZE_CONTRACT_KEYS = {
    "backend",
    "bounds",
    "compress_results_file",
    "crossover_eta",
    "crossover_probability",
    "enable_overrides",
    "fixed_params",
    "fixed_runtime_overrides",
    "gpu",
    "limits",
    "mutation_eta",
    "mutation_indpb",
    "mutation_probability",
    "objective_scenario",
    "offspring_multiplier",
    "population_size",
    "pymoo",
    "round_to_n_significant_digits",
    "scoring",
}


def _restore_gpu_result_run_contract(entry: dict, config: dict) -> dict:
    """Keep persisted critical settings byte-for-byte comparable on resume.

    ``clean_config`` intentionally fills explicit nulls from schema defaults and
    canonicalizes partial bounds. Both transformations are useful for ordinary
    config output but change strict resume inputs. Candidate strategy/risk
    values stay untouched; only invariant run-contract fields are restored.
    """

    for section_name, keys in (
        ("backtest", GPU_RESULT_BACKTEST_CONTRACT_KEYS),
        ("optimize", GPU_RESULT_OPTIMIZE_CONTRACT_KEYS),
    ):
        source = config.get(section_name, {})
        target = entry.setdefault(section_name, {})
        for key in keys:
            if key in source:
                target[key] = deepcopy(source[key])
    return entry


def _single_scenario_metric_surface(metrics: dict) -> dict:
    """Expose scalar proxy metrics through the evaluator's reducer key space."""

    flattened = dict(metrics)
    for key, value in metrics.items():
        for reducer in ("mean", "min", "max", "median"):
            flattened[f"{key}_{reducer}"] = value
        flattened[f"{key}_std"] = 0.0
    return flattened


_GPU_SUITE_OBJECTIVES_KEY = "__gpu_suite_objectives__"
_GPU_SUITE_VIOLATION_KEY = "__gpu_suite_constraint_violation__"
_GPU_SUITE_METRICS_KEY = "__gpu_suite_metrics__"


def _scalar_metric_stats(metrics: dict) -> dict:
    return {
        key: {
            "mean": float(value),
            "min": float(value),
            "max": float(value),
            "std": 0.0,
            "median": float(value),
        }
        for key, value in metrics.items()
    }


def _evaluate_gpu_suite_proxies(suite_evaluator, scenario_proxies, candidates) -> list[dict]:
    """Screen one candidate batch across suite scenarios with canonical reducers."""

    from suite_runner import ScenarioResult, SuiteScenario

    scenario_rows = []
    for ctx, proxy, parameter_overrides in scenario_proxies:
        scenario_candidates = (
            [dict(candidate, **parameter_overrides) for candidate in candidates]
            if parameter_overrides
            else candidates
        )
        scenario_rows.append((ctx, proxy.evaluate(scenario_candidates)))
    results = []
    for index in range(len(candidates)):
        scenario_results = [
            ScenarioResult(
                scenario=SuiteScenario(
                    label=ctx.label,
                    start_date=None,
                    end_date=None,
                    coins=None,
                    ignored_coins=None,
                ),
                per_exchange={},
                metrics={"stats": _scalar_metric_stats(rows[index])},
                elapsed_seconds=0.0,
                output_path=None,
            )
            for ctx, rows in scenario_rows
        ]
        scored = suite_evaluator.score_scenario_results(scenario_results)
        results.append(
            {
                _GPU_SUITE_OBJECTIVES_KEY: tuple(scored["objectives"]),
                _GPU_SUITE_VIOLATION_KEY: float(scored["constraint_violation"]),
                _GPU_SUITE_METRICS_KEY: scored["suite_metrics"],
            }
        )
    return results


def _suite_limit_metric_value(suite_payload: dict, check: dict):
    metrics = suite_payload.get("metrics", {}) if isinstance(suite_payload, dict) else {}
    metric = check["metric"]
    entry = metrics.get(metric)
    if entry is None and metric.endswith(("_usd", "_btc")):
        entry = metrics.get(metric.rsplit("_", 1)[0])
    if not isinstance(entry, dict):
        return None
    scenario = check.get("scenario")
    if scenario is not None:
        return (entry.get("scenarios") or {}).get(scenario)
    return (entry.get("stats") or {}).get(check.get("reducer") or "mean")


def _resolve_options(config: dict) -> dict:
    options = dict(GPU_DEFAULTS)
    configured = config.get("optimize", {}).get("gpu", {})
    if configured is not None and not isinstance(configured, dict):
        raise TypeError("optimize.gpu must be an object")
    for key, default in GPU_DEFAULTS.items():
        if key in (configured or {}):
            options[key] = type(default)(configured[key])
    for key in (
        "population_size",
        "batch_size",
        "validate_per_generation",
        "drift_window",
        "drift_min_samples",
    ):
        if int(options[key]) <= 0:
            raise ValueError(f"optimize.gpu.{key} must be greater than zero")
    if int(options["drift_probes"]) < 0:
        raise ValueError("optimize.gpu.drift_probes must be non-negative")
    if float(options["checkpoint_interval_seconds"]) <= 0.0:
        raise ValueError(
            "optimize.gpu.checkpoint_interval_seconds must be greater than zero"
        )
    if not 0.0 < float(options["drift_halt"]) <= 1.0:
        raise ValueError(
            "optimize.gpu.drift_halt must be greater than zero and at most one"
        )
    if int(options["drift_min_samples"]) > int(options["drift_window"]):
        raise ValueError(
            "optimize.gpu.drift_min_samples must be less than or equal to "
            "optimize.gpu.drift_window"
        )
    validations = int(options["validate_per_generation"])
    probes = int(options["drift_probes"])
    if probes >= validations:
        raise ValueError(
            "optimize.gpu.drift_probes must be less than "
            "optimize.gpu.validate_per_generation so proxy-front safety evidence "
            "is always collected"
        )
    exact_workers = int(options["exact_workers"]) or int(
        config.get("optimize", {}).get("n_cpus", 0)
    )
    effective_pending = int(options["max_pending_exact"]) or exact_workers * 2
    if effective_pending < validations:
        raise ValueError(
            "optimize.gpu.max_pending_exact (or its exact-worker default) must be "
            "at least optimize.gpu.validate_per_generation so each full validation "
            "batch retains its configured proxy-front/broad-probe allocation"
        )
    # A complete feasible proxy Pareto front may contain only one novel
    # candidate. All remaining validation slots are truthfully broad/off-front
    # evidence, so budget for the worst-case one true-front sample per full
    # generation rather than assuming a fixed front/probe split.
    required_front_window = MIN_DRIFT_PROBES * validations
    if int(options["drift_window"]) < required_front_window:
        raise ValueError(
            "optimize.gpu.drift_window must be at least "
            f"{required_front_window} to retain {MIN_DRIFT_PROBES} true "
            "proxy-front validations when a complete proxy front contributes "
            "only one novel candidate per generation"
        )
    required_evidence_window = required_front_window
    if int(options["drift_probes"]) > 0:
        required_rank_probes = _minimum_rank_evidence_samples(
            float(options["drift_halt"])
        )
        required_probe_window = max(
            required_rank_probes,
            math.ceil(required_rank_probes * validations / probes),
        )
        if int(options["drift_window"]) < required_probe_window:
            raise ValueError(
                "optimize.gpu.drift_window must be at least "
                f"{required_probe_window} to retain {required_rank_probes} broad "
                "probes, guaranteeing eight rank-comparable samples whenever "
                "the broad-probe constraint gate has not failed"
            )
        required_evidence_window = max(
            required_evidence_window, required_probe_window
        )
    exact_budget = int(config.get("optimize", {}).get("iters", 0))
    required_exact_budget = max(
        int(options["drift_min_samples"]), required_evidence_window
    )
    if exact_budget < required_exact_budget:
        raise ValueError(
            "optimize.iters must be at least "
            f"{required_exact_budget} to activate the configured GPU drift gate; "
            f"got {exact_budget}"
        )
    # Apply the same conservative rolling-suffix proof to a fresh run that is
    # used for resume. Ratio-based window sizing alone misses partial final
    # batches whose zero-probe tail can evict early evidence.
    _validate_resume_evidence_budget(
        [],
        exact_done=0,
        exact_budget=exact_budget,
        options=options,
        context="fresh run",
        error_type=ValueError,
    )
    return options


def _validation_probe_count(
    validation_count: int, validate_per_generation: int, drift_probes: int
) -> int:
    """Preserve the configured evidence ratio in a partial final batch."""
    if validation_count <= 1 or drift_probes <= 0:
        return 0
    return min(
        validation_count - 1,
        drift_probes * validation_count // validate_per_generation,
    )


def _ready_submission_prefix(pending) -> list:
    """Return only the contiguous ready prefix in submission order."""

    ready = []
    for result in pending:
        if not result.ready():
            break
        ready.append(result)
    return ready


def _validate_resume_evidence_budget(
    pairs,
    *,
    exact_done: int,
    exact_budget: int,
    options: dict,
    context: str = "resume",
    error_type: type[Exception] = RuntimeError,
) -> None:
    """Fail closed if a resumed run cannot retain mandatory front evidence.

    Broad probes are opportunistic because a complete feasible proxy front can
    truthfully leave no off-front candidates. An uninterrupted run accepts
    that geometry and keeps its independent broad-probe gates inactive until
    enough evidence exists, so resume must not invent a stronger guarantee for
    future generations. Durable probe evidence remains in ``pairs`` and is
    evaluated by ``_DriftMonitor`` exactly as it is without a restart.
    """

    remaining = max(0, int(exact_budget) - int(exact_done))
    window = int(options["drift_window"])
    validations = int(options["validate_per_generation"])

    # Recovered samples already have a durable order. Future exact results are
    # consumed strictly in submission order, so each validation generation is
    # a contiguous segment. Each future generation requests at least one true-
    # front validation, while broad probes depend on the population geometry
    # and therefore cannot be guaranteed before selection. A partially retained
    # future segment may lose its front sample first, which is the conservative
    # within-batch order.
    segments = [(1, int(bool(row[4]))) for row in pairs]
    future = remaining
    while future > 0:
        count = min(validations, future)
        segments.append((count, 1))
        future -= count

    kept = 0
    guaranteed_front = 0
    for length, front_count in reversed(segments):
        if kept >= window:
            break
        included = min(length, window - kept)
        excluded = length - included
        guaranteed_front += max(0, front_count - excluded)
        kept += included

    if guaranteed_front < MIN_DRIFT_PROBES:
        raise error_type(
            f"GPU {context} has insufficient exact budget to retain "
            f"{MIN_DRIFT_PROBES} proxy-front safety samples in the drift window: "
            f"guaranteed={guaranteed_front}, exact_done={exact_done}, "
            f"remaining={remaining}"
        )


def _validate_scope_config(
    config: dict,
    *,
    exchanges,
    coin_count: int,
    allow_suite: bool = False,
) -> str:
    if bool(config.get("backtest", {}).get("suite_enabled")) and not allow_suite:
        raise ValueError("GPU foundation does not support suite mode")
    if bool(config.get("backtest", {}).get("filter_by_min_effective_cost")):
        raise ValueError(
            "GPU foundation requires backtest.filter_by_min_effective_cost=false "
            "because the screening proxy promotes entries to exchange minimum size"
        )
    if bool(config.get("live", {}).get("market_orders_allowed")):
        raise ValueError(
            "GPU foundation requires live.market_orders_allowed=false because the "
            "screening proxy models resting maker orders only"
        )
    if float(config.get("backtest", {}).get("btc_collateral_cap", 0.0) or 0.0) > 0.0:
        raise ValueError("GPU foundation does not support backtest.btc_collateral_cap")
    if float(config.get("live", {}).get("max_realized_loss_pct", 1.0)) != 1.0:
        raise ValueError(
            "GPU foundation requires live.max_realized_loss_pct=1.0 because the "
            "screening proxy does not model the realized-loss gate"
        )
    exchanges = list(exchanges)
    if len(exchanges) != 1:
        raise ValueError(
            f"GPU foundation requires exactly one exchange, got {exchanges}"
        )
    exchange = exchanges[0]
    coin_count = int(coin_count)
    if coin_count < 1:
        raise ValueError("GPU foundation requires at least one prepared coin")
    strategy_kind = (
        str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
    )
    if strategy_kind not in GPU_STRATEGY_BOUND_MAPS:
        raise ValueError(
            "GPU foundation supports strategy_kind=ema_anchor or "
            "trailing_martingale only; "
            f"got {strategy_kind!r}"
        )
    enabled_sides = [side for side in ("long", "short") if gpu_side_enabled(config, side)]
    if not enabled_sides:
        raise ValueError("GPU foundation requires at least one enabled side")
    if coin_count > 1:
        from optimization.gpu.model import MPS_MULTICOIN_MAX_COINS

        if coin_count > MPS_MULTICOIN_MAX_COINS:
            raise ValueError(
                "GPU multicoin foundation supports at most "
                f"{MPS_MULTICOIN_MAX_COINS} coins; prepared {coin_count}"
            )
        if strategy_kind != "ema_anchor":
            raise ValueError(
                "GPU multicoin foundation currently supports strategy_kind=ema_anchor only"
            )
        if len(enabled_sides) != 1:
            raise ValueError(
                "GPU multicoin foundation currently supports exactly one enabled side"
            )
        if not bool(config.get("backtest", {}).get("dynamic_wel_by_tradability")):
            raise ValueError(
                "GPU multicoin foundation requires "
                "backtest.dynamic_wel_by_tradability=true"
            )
        if (
            float(
                config.get("live", {}).get("forager_score_hysteresis_pct", 0.0)
                or 0.0
            )
            != 0.0
        ):
            raise ValueError(
                "GPU multicoin foundation requires "
                "live.forager_score_hysteresis_pct=0"
            )
    _validate_gpu_coin_overrides(
        config,
        strategy_kind=strategy_kind,
        enabled_sides=enabled_sides,
        coin_count=coin_count,
    )
    for side in enabled_sides:
        side_config = config["bot"][side]
        if bool(side_config.get("hsl", {}).get("enabled")):
            raise ValueError(f"GPU foundation requires bot.{side}.hsl.enabled=false")
        if bool(side_config.get("unstuck", {}).get("enabled")):
            raise ValueError(
                f"GPU foundation requires bot.{side}.unstuck.enabled=false"
            )
        risk = side_config.get("risk", {})
        for key, expected in (
            ("position_exposure_enforcer_enabled", False),
            ("total_exposure_enforcer_enabled", False),
            ("total_exposure_entry_gate_enabled", True),
        ):
            if bool(risk.get(key, expected)) != expected:
                raise ValueError(
                    f"GPU foundation requires bot.{side}.risk.{key}={str(expected).lower()}"
                )
        if float(risk.get("we_excess_allowance_pct", 0.0) or 0.0) != 0.0:
            raise ValueError(
                f"GPU foundation requires bot.{side}.risk.we_excess_allowance_pct=0.0"
            )
    return exchange


def _validate_gpu_coin_overrides(
    config: dict,
    *,
    strategy_kind: str,
    enabled_sides,
    coin_count: int,
) -> None:
    """Accept only the static per-coin leaves modeled by the MPS proxy."""

    overrides = config.get("coin_overrides") or {}
    if not overrides:
        return
    if coin_count <= 1 or strategy_kind != "ema_anchor" or len(enabled_sides) != 1:
        raise ValueError(
            "GPU coin_overrides currently require multi-coin EMA Anchor with "
            "exactly one enabled side"
        )
    enabled_side = next(iter(enabled_sides))
    from optimization.gpu.model import EMA_ANCHOR_PARAM_KEYS

    strategy_keys = set(EMA_ANCHOR_PARAM_KEYS) - {
        "entry_cooldown_minutes",
        "total_wallet_exposure_limit",
    }

    def leaves(value, prefix=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, (*prefix, str(key)))
        else:
            yield prefix

    allowed = {
        ("bot", enabled_side, "risk", "entry_cooldown_minutes"),
        ("bot", enabled_side, "wallet_exposure_limit"),
    } | {
        ("bot", enabled_side, "strategy", "ema_anchor", key)
        for key in strategy_keys
    }
    unsupported = []
    for coin, patch in overrides.items():
        if not isinstance(patch, dict):
            unsupported.append(f"coin_overrides.{coin}")
            continue
        unsupported.extend(
            ".".join(("coin_overrides", str(coin), *path))
            for path in leaves(patch)
            if path not in allowed
        )
    if unsupported:
        raise ValueError(
            "GPU coin_overrides do not model these paths yet: "
            f"{sorted(unsupported)}; supported leaves are enabled-side EMA Anchor "
            "parameters, risk.entry_cooldown_minutes, and wallet_exposure_limit"
        )


def _validate_scope(
    config: dict,
    evaluator,
    *,
    allow_suite: bool = False,
) -> str:
    exchanges = list(getattr(evaluator, "exchanges", []))
    if len(exchanges) != 1:
        return _validate_scope_config(
            config,
            exchanges=exchanges,
            coin_count=0,
            allow_suite=allow_suite,
        )
    exchange = exchanges[0]
    hlcvs = evaluator.shared_hlcvs_np[exchange]
    return _validate_scope_config(
        config,
        exchanges=exchanges,
        coin_count=int(hlcvs.shape[1]),
        allow_suite=allow_suite,
    )


def _gpu_suite_scenario_inputs(proxy_config: dict, suite_evaluator) -> list[dict]:
    """Materialize fail-closed suite inputs for MPS screening."""

    from config.param_paths import require_existing_config_path

    contexts = getattr(suite_evaluator, "contexts", None)
    get_data = getattr(suite_evaluator, "get_prepared_context_data", None)
    build_config = getattr(suite_evaluator, "build_scenario_candidate_config", None)
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("GPU suite mode requires prepared optimizer scenario contexts")
    if not callable(get_data) or not callable(build_config):
        raise TypeError("GPU suite mode requires the canonical SuiteEvaluator")

    prepared = []
    expected_exchange = None
    for ctx in contexts:
        overrides = getattr(ctx, "overrides", {}) or {}
        for dotted_path in overrides:
            resolved = require_existing_config_path(proxy_config, dotted_path)
            if len(resolved) < 3 or resolved[0] != "bot" or resolved[1] not in {
                "long",
                "short",
            }:
                raise ValueError(
                    f"GPU suite scenario {ctx.label!r} override {dotted_path!r} is "
                    "outside the supported bot.long/bot.short scope"
                )
        coin_sources = (
            getattr(ctx, "config", {}).get("backtest", {}).get("coin_sources") or {}
        )
        if coin_sources:
            raise ValueError(
                f"GPU suite scenario {ctx.label!r} uses coin_sources; this slice "
                "does not model per-coin source exchanges"
            )
        exchanges = list(ctx.exchanges)
        if len(exchanges) != 1:
            raise ValueError(
                f"GPU suite scenario {ctx.label!r} requires exactly one exchange, "
                f"got {exchanges}"
            )
        exchange = exchanges[0]
        if exchange == "combined":
            raise ValueError(
                "GPU suite scenarios do not support combined multi-exchange datasets"
            )
        if expected_exchange is None:
            expected_exchange = exchange
        elif exchange != expected_exchange:
            raise ValueError(
                "GPU suite scenarios must use one shared exchange; "
                f"expected {expected_exchange!r}, got {exchange!r} in {ctx.label!r}"
            )

        hlcvs, btc, coin_indices = get_data(ctx, exchange)
        values = np.asarray(hlcvs)
        if coin_indices is not None:
            values = np.take(values, list(coin_indices), axis=1)
        values = np.ascontiguousarray(values)
        coin_count = int(values.shape[1])
        scenario_config = build_config(proxy_config, ctx)
        _validate_scope_config(
            scenario_config,
            exchanges=exchanges,
            coin_count=coin_count,
            allow_suite=True,
        )
        effective_coins = [
            str(coin) for coin in ctx.msss[exchange] if coin != "__meta__"
        ]
        if len(effective_coins) != coin_count:
            raise ValueError(
                f"GPU suite scenario {ctx.label!r} prepared coin identity "
                f"mismatch: hlcvs={coin_count}, market_settings={effective_coins}"
            )
        prepared.append(
            {
                "ctx": ctx,
                "config": scenario_config,
                "overrides": deepcopy(overrides),
                "exchange": exchange,
                "coin_count": coin_count,
                "coins": effective_coins,
                "hlcvs": values,
                "mss": ctx.msss[exchange],
                "btc": btc,
                "timestamps": ctx.timestamps.get(exchange),
            }
        )
    return prepared


def _gpu_suite_search_context(suite_inputs: list[dict]) -> tuple[int, int, str | None]:
    """Return the common candidate-space coin range and multicoin side."""

    if not suite_inputs:
        raise ValueError("GPU suite mode requires at least one prepared scenario")
    coin_counts = [int(item["coin_count"]) for item in suite_inputs]
    min_coin_count = min(coin_counts)
    max_coin_count = max(coin_counts)
    if max_coin_count == 1:
        return min_coin_count, max_coin_count, None

    strategy_kinds = {
        str(item["config"].get("live", {}).get("strategy_kind", ""))
        .strip()
        .lower()
        for item in suite_inputs
    }
    if strategy_kinds != {"ema_anchor"}:
        raise ValueError(
            "GPU multicoin suites currently require strategy_kind=ema_anchor in "
            f"every scenario; got {sorted(strategy_kinds)}"
        )
    sides_by_label = {}
    for item in suite_inputs:
        sides = tuple(
            side
            for side in ("long", "short")
            if gpu_side_enabled(item["config"], side)
        )
        sides_by_label[item["ctx"].label] = sides
        if len(sides) != 1:
            raise ValueError(
                "GPU multicoin suites require exactly one enabled side in every "
                f"scenario; {item['ctx'].label!r} has {list(sides)}"
            )
    common_sides = {sides[0] for sides in sides_by_label.values()}
    if len(common_sides) != 1:
        details = ", ".join(
            f"{label}={sides[0]}" for label, sides in sides_by_label.items()
        )
        raise ValueError(
            "GPU multicoin suites require the same enabled side in every scenario; "
            f"got {details}"
        )
    return min_coin_count, max_coin_count, common_sides.pop()


def _gpu_suite_scenario_override_context(
    base_config: dict,
    scenario_config: dict,
    overrides: dict,
    bound_keys,
    bound_map: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve exact-last scenario overrides into candidate and proxy shadows."""

    if not overrides:
        return {}, {}

    from config.param_paths import (
        require_existing_config_path,
        resolve_optimizer_key_path,
    )

    override_paths = [
        require_existing_config_path(base_config, dotted_path)
        for dotted_path in overrides
    ]

    def is_shadowed(path: tuple[str, ...]) -> bool:
        return any(
            path[: len(override_path)] == override_path
            for override_path in override_paths
        )

    def value_at(path: tuple[str, ...]):
        value = scenario_config
        for part in path:
            value = value[part]
        return value

    fixed_bound_values: dict[str, float] = {}
    fixed_parameters: dict[str, float] = {}
    for bound_key in bound_keys:
        path = resolve_optimizer_key_path(base_config, bound_key)
        if path is None or not is_shadowed(path):
            continue
        try:
            value = float(value_at(path))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "GPU suite scenario override for optimizer bound "
                f"{bound_key!r} must resolve to a numeric value"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                "GPU suite scenario override for optimizer bound "
                f"{bound_key!r} must resolve to a finite value"
            )
        fixed_bound_values[bound_key] = value
        parameter = bound_map.get(bound_key)
        if parameter is not None:
            previous = fixed_parameters.get(parameter)
            if previous is not None and not math.isclose(
                previous, value, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    "GPU suite scenario overrides resolve conflicting values for "
                    f"proxy parameter {parameter!r}"
                )
            fixed_parameters[parameter] = value
    return fixed_bound_values, fixed_parameters


def _gpu_suite_enabled(config: dict, evaluator, evaluator_for_pool) -> bool:
    enabled = evaluator_for_pool is not evaluator
    if bool(config.get("backtest", {}).get("suite_enabled")) and not enabled:
        raise TypeError("GPU suite mode requires the canonical SuiteEvaluator")
    return enabled


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including equal-value ties."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return float("nan")
    left_rank = _average_ranks(left[finite])
    right_rank = _average_ranks(right[finite])
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(
        np.sqrt((left_rank * left_rank).sum() * (right_rank * right_rank).sum())
    )
    return (
        float((left_rank * right_rank).sum() / denominator)
        if denominator
        else float("nan")
    )


class _ObjectiveScale:
    def __init__(self):
        self.median: np.ndarray | None = None
        self.spread: np.ndarray | None = None

    def fit(self, objectives: np.ndarray) -> None:
        values = np.asarray(objectives, dtype=np.float64)
        self.median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, [25.0, 75.0], axis=0)
        fallback = np.maximum(np.abs(self.median) * 0.1, 1.0e-9)
        self.spread = np.where(q75 - q25 > 1.0e-12, q75 - q25, fallback)

    def score(self, objectives: np.ndarray) -> np.ndarray:
        if self.median is None or self.spread is None:
            raise RuntimeError("GPU objective scale has not been fitted")
        values = (np.asarray(objectives, dtype=np.float64) - self.median) / self.spread
        values[~np.isfinite(values)] = 1.0e6
        return values.mean(axis=1)


class _DriftMonitor:
    MIN_PROBES = MIN_DRIFT_PROBES
    MIN_FRONT_SAMPLES = MIN_DRIFT_PROBES

    def __init__(self, options: dict):
        self.window = int(options["drift_window"])
        self.minimum = int(options["drift_min_samples"])
        self.halt = float(options["drift_halt"])
        self.pairs: deque[tuple[float, float, bool, bool, bool]] = deque(
            maxlen=self.window
        )

    def add(
        self,
        proxy_score: float,
        exact_score: float,
        *,
        probe: bool,
        proxy_front: bool,
        constraint_mismatch: bool = False,
    ) -> None:
        if bool(probe) == bool(proxy_front):
            raise ValueError(
                "GPU validation evidence must be exactly one of proxy-front "
                "or broad/off-front"
            )
        self.pairs.append(
            (
                float(proxy_score),
                float(exact_score),
                bool(probe),
                bool(constraint_mismatch),
                bool(proxy_front),
            )
        )

    def evaluate(self) -> dict:
        result = {
            "rho": float("nan"),
            "probe_rho": float("nan"),
            "front_rho": float("nan"),
            "samples": len(self.pairs),
            "probes": 0,
            "front_samples": 0,
            "rank_samples": 0,
            "probe_rank_samples": 0,
            "front_rank_samples": 0,
            "constraint_agreement": float("nan"),
            "constraint_mismatches": 0,
            "probe_constraint_agreement": float("nan"),
            "probe_constraint_mismatches": 0,
            "front_constraint_agreement": float("nan"),
            "front_constraint_mismatches": 0,
            "halt_reason": None,
            "warn_reason": None,
        }
        if len(self.pairs) < self.minimum:
            return result
        proxy = np.asarray([row[0] for row in self.pairs], dtype=np.float64)
        exact = np.asarray([row[1] for row in self.pairs], dtype=np.float64)
        probes = np.asarray([row[2] for row in self.pairs], dtype=bool)
        fronts = np.asarray(
            [row[4] for row in self.pairs],
            dtype=bool,
        )
        if np.any(probes == fronts):
            raise RuntimeError(
                "GPU drift evidence contains invalid proxy-front/broad-probe "
                "classification"
            )
        constraint_mismatches = np.asarray(
            [row[3] if len(row) > 3 else False for row in self.pairs], dtype=bool
        )
        result["constraint_mismatches"] = int(constraint_mismatches.sum())
        result["probes"] = int(probes.sum())
        result["front_samples"] = int(fronts.sum())
        # Feasibility disagreements have their own independent fail-closed
        # gates below. They are not rank-comparable: including them in
        # Spearman both double-counts the same failure and can turn otherwise
        # exact near-ties into arbitrary rank inversions.
        rank_eligible = ~constraint_mismatches
        probe_rank_eligible = probes & rank_eligible
        front_rank_eligible = fronts & rank_eligible
        result["rank_samples"] = int(rank_eligible.sum())
        result["probe_rank_samples"] = int(probe_rank_eligible.sum())
        result["front_rank_samples"] = int(front_rank_eligible.sum())
        result["rho"] = _spearman(proxy[rank_eligible], exact[rank_eligible])
        result["probe_rho"] = _spearman(
            proxy[probe_rank_eligible], exact[probe_rank_eligible]
        )
        result["front_rho"] = _spearman(
            proxy[front_rank_eligible], exact[front_rank_eligible]
        )
        result["constraint_agreement"] = 1.0 - (
            result["constraint_mismatches"] / result["samples"]
        )
        result["probe_constraint_mismatches"] = int(
            (constraint_mismatches & probes).sum()
        )
        if result["probes"]:
            result["probe_constraint_agreement"] = 1.0 - (
                result["probe_constraint_mismatches"] / result["probes"]
            )
        result["front_constraint_mismatches"] = int(
            (constraint_mismatches & fronts).sum()
        )
        if result["front_samples"]:
            result["front_constraint_agreement"] = 1.0 - (
                result["front_constraint_mismatches"] / result["front_samples"]
            )
        detail = (
            f"rho={result['rho']:.3f}, probe_rho={result['probe_rho']:.3f}, "
            f"front_rho={result['front_rho']:.3f}, samples={result['samples']}, "
            f"constraint_agreement={result['constraint_agreement']:.3f}, "
            f"probes={result['probes']}, "
            f"probe_rank_samples={result['probe_rank_samples']}, "
            f"probe_constraint_agreement={result['probe_constraint_agreement']:.3f}, "
            f"front_samples={result['front_samples']}, "
            f"front_rank_samples={result['front_rank_samples']}, "
            f"front_constraint_agreement={result['front_constraint_agreement']:.3f}"
        )
        if result["constraint_agreement"] < self.halt:
            result["halt_reason"] = (
                "GPU proxy/exact rolling constraint agreement fell below "
                f"safety threshold ({detail})"
            )
        elif result["probes"] >= self.MIN_PROBES and (
            result["probe_constraint_agreement"] < self.halt
        ):
            result["halt_reason"] = (
                "GPU proxy/exact broad-probe constraint agreement fell below "
                f"safety threshold ({detail})"
            )
        elif result["front_samples"] >= self.MIN_FRONT_SAMPLES and (
            result["front_constraint_agreement"] < self.halt
        ):
            result["halt_reason"] = (
                "GPU proxy/exact proxy-front constraint agreement fell below "
                f"safety threshold ({detail})"
            )
        elif result["probe_rank_samples"] >= self.MIN_PROBES and (
            not np.isfinite(result["probe_rho"])
            or result["probe_rho"] < self.halt
        ):
            result["halt_reason"] = (
                f"GPU proxy/exact broad-probe rank drift exceeded safety threshold ({detail})"
            )
        elif np.isfinite(result["rho"]) and result["rho"] >= self.halt:
            return result
        elif result["probe_rank_samples"] < self.MIN_PROBES:
            result["warn_reason"] = (
                "GPU drift below threshold without enough rank-comparable broad "
                f"probes ({detail})"
            )
        else:
            result["warn_reason"] = (
                f"GPU front rank is noisy but broad probes remain sound ({detail})"
            )
        return result


def _normalized_farthest_indices(values: np.ndarray, count: int) -> list[int]:
    values = np.asarray(values, dtype=np.float64)
    if count <= 0 or len(values) == 0:
        return []
    if len(values) <= count:
        return list(range(len(values)))
    low = np.nanmin(values, axis=0)
    span = np.nanmax(values, axis=0) - low
    normalized = (values - low) / np.where(span > 1.0e-12, span, 1.0)
    chosen = [int(np.argmin(np.nanmean(normalized, axis=1)))]
    distance = np.linalg.norm(normalized - normalized[chosen[0]], axis=1)
    for _ in range(count - 1):
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(
            distance, np.linalg.norm(normalized - normalized[index], axis=1)
        )
    return chosen


def _select_validation_indices(
    objectives: np.ndarray,
    scores: np.ndarray,
    violations: np.ndarray | None = None,
    *,
    total: int,
    probes: int,
) -> list[tuple[int, bool, bool]]:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    objectives = np.asarray(objectives, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if violations is None:
        violations = np.zeros(len(objectives), dtype=np.float64)
    else:
        violations = np.asarray(violations, dtype=np.float64)
    if len(objectives) != len(scores) or len(objectives) != len(violations):
        raise ValueError("GPU validation objectives, scores and violations must align")
    if total <= 0 or len(objectives) == 0:
        return []

    feasible = np.flatnonzero(np.isfinite(violations) & (violations <= 0.0))
    primary = feasible if len(feasible) else np.arange(len(objectives), dtype=np.int64)
    feasible_ids = set(map(int, feasible))
    primary_ids = set(map(int, primary))
    front_local = np.asarray(
        NonDominatedSorting().do(
            objectives[primary], only_non_dominated_front=True
        ),
        dtype=np.int64,
    )
    front = primary[front_local]
    front_ids = {int(index) for index in front}
    broad_pool = np.asarray(
        [
            int(index)
            for index in np.argsort(scores)
            if int(index) in primary_ids and int(index) not in front_ids
        ],
        dtype=np.int64,
    )
    requested_probes = min(max(0, probes), max(0, total - 1))
    # With several competing objectives the complete feasible proxy Pareto
    # front may legitimately contain nearly the entire population. Use every
    # truthful off-front probe available, then let diverse true-front members
    # fill the remaining exact quota. Never relabel a front member as a broad
    # probe merely to satisfy the configured per-generation target.
    probe_count = min(requested_probes, len(broad_pool))
    front_count = max(0, total - probe_count)
    elite_local = _normalized_farthest_indices(objectives[front], front_count)
    selected = [(int(front[index]), False, True) for index in elite_local]
    selected_ids = {index for index, _probe, _front in selected}
    if probe_count:
        positions = np.round(
            np.linspace(0, len(broad_pool) - 1, num=probe_count)
        ).astype(int)
        for position in positions:
            index = int(broad_pool[position])
            if index not in selected_ids:
                selected.append((index, True, False))
                selected_ids.add(index)

    preferred_order = sorted(
        map(int, primary), key=lambda index: (float(violations[index]), float(scores[index]))
    )
    for index in preferred_order:
        if len(selected) >= total:
            break
        if index not in selected_ids:
            is_front = index in front_ids
            selected.append((index, not is_front, is_front))
            selected_ids.add(index)

    # Return a complete preference order. The caller may skip candidates whose
    # quantized exact vectors were already evaluated and continue down this
    # list until it fills the generation's exact-validation quota.
    fallback_order = sorted(
        range(len(objectives)),
        key=lambda index: (
            0 if index in feasible_ids else 1,
            float(violations[index]) if np.isfinite(violations[index]) else float("inf"),
            float(scores[index]),
        ),
    )
    for index in fallback_order:
        if index not in selected_ids:
            is_front = index in front_ids
            selected.append((index, not is_front, is_front))
            selected_ids.add(index)
    return selected


def _select_novel_validations(
    selections,
    *,
    total: int,
    candidate_for_index,
    digest_for_candidate,
    completed_hashes,
    submitted_hashes,
):
    novel = []
    seen = set()
    novel_probe_count = 0
    novel_front_count = 0
    # The first ``total`` preferences are the selector's truthful allocation.
    # Later items replace duplicate hashes but must not change that class mix.
    target_probe_count = min(
        max(0, total - 1),
        sum(
            bool(is_probe)
            for _index, is_probe, _is_front in selections[:total]
        ),
    )
    target_front_count = max(1, total - target_probe_count)
    for index, is_probe, is_front in selections:
        candidate = candidate_for_index(index)
        digest = digest_for_candidate(candidate)
        if digest in completed_hashes or digest in submitted_hashes or digest in seen:
            continue
        seen.add(digest)
        item = (index, bool(is_probe), bool(is_front), candidate, digest)
        novel.append(item)
        novel_probe_count += int(bool(is_probe))
        novel_front_count += int(bool(is_front))
        if (
            novel_probe_count >= target_probe_count
            and novel_front_count >= target_front_count
        ):
            break

    probe_items = [item for item in novel if item[1]]
    front_items = [item for item in novel if item[2]]
    if not front_items:
        raise RuntimeError(
            "GPU validation cannot provide novel proxy-front safety evidence; "
            "the current proxy front was already evaluated or submitted"
        )
    chosen = probe_items[:target_probe_count]
    chosen_digests = {item[4] for item in chosen}
    for item in front_items:
        if len(chosen) >= total:
            break
        if item[4] not in chosen_digests:
            chosen.append(item)
            chosen_digests.add(item[4])
    return chosen


def _update_probe_shortfall_log(
    previous: tuple[int, int] | None,
    *,
    requested: int,
    actual: int,
) -> tuple[int, int] | None:
    current = (requested, actual) if actual < requested else None
    if current == previous:
        return previous
    if current is None:
        if previous is not None:
            logging.info(
                "GPU validation broad-probe allocation recovered | "
                "requested=%d available=%d",
                requested,
                actual,
            )
    else:
        logging.warning(
            "GPU validation has fewer novel candidates outside the complete feasible "
            "proxy Pareto front than requested | requested=%d available=%d | "
            "filling the exact quota with diverse true-front candidates; broad-probe "
            "gates use only truthful accumulated off-front evidence",
            requested,
            actual,
        )
    return current


def _update_novelty_stall(
    previous: int, *, submitted: int, pending: int
) -> int:
    if submitted > 0 or pending > 0:
        return 0
    current = previous + 1
    if current >= MAX_NOVELTY_STALL_GENERATIONS:
        raise RuntimeError(
            "GPU optimizer produced no novel exact candidates for "
            f"{MAX_NOVELTY_STALL_GENERATIONS} consecutive generations; "
            "the quantized search space appears exhausted before the exact budget"
        )
    return current


def _gpu_suite_checkpoint_contract(config: dict, suite_inputs=None) -> dict:
    backtest = config.get("backtest", {})
    contract = {
        key: deepcopy(backtest.get(key))
        for key in (
            "suite_enabled",
            "scenarios",
            "reducer",
            "exchanges",
            "volume_normalization",
        )
    }
    if suite_inputs is not None:
        prepared_scenarios = []
        for item in suite_inputs:
            timestamps = np.asarray(item["timestamps"]).reshape(-1)
            if len(timestamps) != len(item["hlcvs"]):
                raise ValueError(
                    f"GPU suite scenario {item['ctx'].label!r} timestamp identity "
                    f"mismatch: timestamps={len(timestamps)}, hlcvs={len(item['hlcvs'])}"
                )
            prepared_scenarios.append(
                {
                    "label": item["ctx"].label,
                    "exchange": item["exchange"],
                    "coins": list(item["coins"]),
                    "coin_count": int(item["coin_count"]),
                    "strategy_kind": str(
                        item["config"].get("live", {}).get("strategy_kind", "")
                    )
                    .strip()
                    .lower(),
                    "enabled_sides": [
                        side
                        for side in ("long", "short")
                        if gpu_side_enabled(item["config"], side)
                    ],
                    "candle_count": int(len(item["hlcvs"])),
                    "first_timestamp": (
                        int(timestamps[0]) if len(timestamps) else None
                    ),
                    "last_timestamp": (
                        int(timestamps[-1]) if len(timestamps) else None
                    ),
                    "coin_overrides": deepcopy(
                        item.get("coin_override_contract")
                        or item["config"].get("coin_overrides", {})
                    ),
                }
            )
        contract["prepared_scenarios"] = prepared_scenarios
    return contract


def _checkpoint_signature(
    active,
    scoring,
    *,
    anchor_plan=None,
    suite_contract=None,
    runtime_contract=None,
) -> str:
    payload = {
        "active": [
            [name, int(index), float(bound.low), float(bound.high), bound.step]
            for name, index, bound in active
        ],
        "scoring": scoring,
        "version": 2,
    }
    if anchor_plan is not None:
        payload["anchor_plan"] = {
            "fixed_keys": sorted(anchor_plan.get("fixed_keys") or []),
            "tunable_keys": sorted(anchor_plan.get("tunable_keys") or []),
            "anchors": [
                sorted(
                    [str(item["key"]), float(item["value"])]
                    for item in anchor.get("fixed_values") or []
                )
                for anchor in anchor_plan.get("anchors") or []
            ],
        }
    if suite_contract is not None:
        payload["suite_contract"] = suite_contract
    if runtime_contract is not None:
        payload["runtime_contract"] = runtime_contract
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_checkpoint(path: str | None, state: dict) -> None:
    if path is None:
        return
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "wb") as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save GPU optimizer checkpoint: {path}") from exc


def _canonical_candidate_values(
    rows: np.ndarray,
    low: np.ndarray,
    span: np.ndarray,
    bounds,
    sig_digits: int,
) -> np.ndarray:
    """Map normalized genes through the exact optimizer quantization contract."""

    values = np.asarray(low, dtype=np.float64) + np.clip(rows, 0.0, 1.0) * np.asarray(
        span, dtype=np.float64
    )
    return np.asarray(
        [enforce_bounds(row, bounds, sig_digits) for row in values],
        dtype=np.float64,
    )


def _canonical_vector_hash(vector, bounds, sig_digits: int) -> str:
    canonical = enforce_bounds(vector, bounds, sig_digits)
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def _canonicalize_mirrored_hash_vector(vector, base_vector, key_paths) -> list[float]:
    """Erase short genes whose exact values are overwritten by mirroring."""

    canonical = [float(value) for value in vector]
    index_by_key = {
        bound_key: index for index, (bound_key, _path) in enumerate(key_paths)
    }
    for short_key, short_index in index_by_key.items():
        if not short_key.startswith("short_"):
            continue
        canonical[short_index] = float(base_vector[short_index])
    return canonical


def _canonicalize_optimizer_override_hash_vector(
    vector,
    base_vector,
    key_paths,
    overrides: set[str],
    *,
    anchor_parameter_overrides: list[dict[str, float]] | None = None,
    fixed_bound_values: dict[str, float] | None = None,
    fixed_parameter_overrides: dict[str, float] | None = None,
) -> list[float]:
    """Hash the effective candidate while neutralizing mirrored shadow genes."""

    canonical = [float(value) for value in vector]
    index_by_key = {
        bound_key: index for index, (bound_key, _path) in enumerate(key_paths)
    }
    for bound_key, value in (fixed_bound_values or {}).items():
        if bound_key in index_by_key:
            canonical[index_by_key[bound_key]] = float(value)
    parameters = {}
    if anchor_parameter_overrides is not None:
        anchor_index = index_by_key.get(ANCHOR_GENE_KEY)
        anchor_id = (
            int(round(canonical[anchor_index])) if anchor_index is not None else 0
        )
        if anchor_id < 0 or anchor_id >= len(anchor_parameter_overrides):
            raise ValueError(
                "GPU anchored fine-tune selected invalid anchor id while hashing "
                f"{anchor_id}; available={len(anchor_parameter_overrides)}"
            )
        parameters.update(anchor_parameter_overrides[anchor_id])
    parameters.update(
        {bound_key: canonical[index] for bound_key, index in index_by_key.items()}
    )
    parameters.update(fixed_parameter_overrides or {})
    _apply_gpu_optimizer_overrides(parameters, overrides)
    for bound_key, value in parameters.items():
        if bound_key in index_by_key:
            canonical[index_by_key[bound_key]] = float(value)
    if "mirror_short_from_long" in overrides:
        canonical = _canonicalize_mirrored_hash_vector(
            canonical,
            base_vector,
            key_paths,
        )
    return canonical


def _build_proxy_parameter_dicts(
    base_vector,
    mapped,
    active,
    active_values,
    *,
    anchor_parameter_overrides: list[dict[str, float]] | None = None,
    fixed_parameter_overrides: dict[str, float] | None = None,
    optimizer_overrides: set[str] | None = None,
) -> list[dict]:
    """Include canonical pinned and active strategy values in each proxy candidate."""

    base_parameters = {
        name: float(base_vector[index]) for name, (index, _bound) in mapped.items()
    }
    anchor_columns = [
        column
        for column, (name, _index, _bound) in enumerate(active)
        if name == ANCHOR_GENE_KEY
    ]
    if len(anchor_columns) > 1:
        raise ValueError("GPU anchored fine-tune has multiple anchor genes")
    result = []
    for row in active_values:
        parameters = dict(base_parameters)
        if anchor_parameter_overrides is not None:
            anchor_id = (
                int(round(float(row[anchor_columns[0]]))) if anchor_columns else 0
            )
            if anchor_id < 0 or anchor_id >= len(anchor_parameter_overrides):
                raise ValueError(
                    "GPU anchored fine-tune selected invalid anchor id "
                    f"{anchor_id}; available={len(anchor_parameter_overrides)}"
                )
            parameters.update(anchor_parameter_overrides[anchor_id])
        elif anchor_columns:
            raise ValueError("GPU anchor gene is present without an anchored fine-tune plan")
        parameters.update(
            {
                name: float(row[column])
                for column, (name, _index, _bound) in enumerate(active)
                if name != ANCHOR_GENE_KEY
            }
        )
        parameters.update(fixed_parameter_overrides or {})
        result.append(
            _apply_gpu_optimizer_overrides(parameters, optimizer_overrides or set())
        )
    return result


def _build_anchor_parameter_context(
    config: dict, bound_map: dict[str, str]
) -> tuple[list[dict[str, float]] | None, dict[str, Bound]]:
    """Resolve anchor-fixed optimizer values without materializing every candidate config."""

    plan = get_anchor_plan(config)
    if plan is None:
        return None, {}
    fixed_keys = set(plan.get("fixed_keys") or [])
    parameter_overrides: list[dict[str, float]] = []
    values_by_key: dict[str, list[float]] = {}
    for anchor_index, anchor in enumerate(plan.get("anchors") or []):
        seen = set()
        overrides = {}
        for item in anchor.get("fixed_values") or []:
            key = item.get("key")
            if not isinstance(key, str) or key not in fixed_keys:
                raise ValueError(
                    "GPU anchored fine-tune contains an invalid fixed optimizer key "
                    f"for anchor {anchor_index}: {key!r}"
                )
            if key in seen:
                raise ValueError(
                    "GPU anchored fine-tune contains duplicate fixed optimizer key "
                    f"{key!r} for anchor {anchor_index}"
                )
            seen.add(key)
            try:
                value = float(item["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "GPU anchored fine-tune contains a non-numeric fixed value "
                    f"for {key!r} in anchor {anchor_index}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    "GPU anchored fine-tune contains a non-finite fixed value "
                    f"for {key!r} in anchor {anchor_index}"
                )
            values_by_key.setdefault(key, []).append(value)
            if key in bound_map:
                overrides[bound_map[key]] = value
        missing = sorted(fixed_keys - seen)
        if missing:
            raise ValueError(
                "GPU anchored fine-tune is missing fixed optimizer values for "
                f"anchor {anchor_index}: {missing}"
            )
        parameter_overrides.append(overrides)
    if not parameter_overrides:
        raise ValueError("GPU anchored fine-tune requires at least one anchor")
    fixed_bounds = {
        key: Bound(min(values), max(values)) for key, values in values_by_key.items()
    }
    return parameter_overrides, fixed_bounds


def _validate_pinned_scope_bounds(bound_by_key, base_by_key, enabled_sides=None) -> None:
    enabled_sides = set(enabled_sides or ("long", "short"))
    for key, expected in PINNED_SCOPE_BOUND_VALUES.items():
        if key.split("_", 1)[0] not in enabled_sides:
            continue
        bound = bound_by_key.get(key)
        values = (
            (float(bound.low), float(bound.high))
            if bound is not None
            else (float(base_by_key.get(key, expected)),) * 2
        )
        if any(not math.isclose(value, expected, abs_tol=1.0e-12) for value in values):
            raise ValueError(
                "GPU foundation requires "
                f"{key} to remain pinned at {expected}; got bounds {values}"
            )


def _validate_directional_search_space(
    bound_by_key, base_by_key, approved, enabled_sides, *, coin_count: int = 1
) -> None:
    """Keep candidate-side eligibility identical to the proxy runner's flags."""

    enabled_sides = set(enabled_sides)

    def side_approved(side: str) -> bool:
        return bool(approved.get(side, [])) if isinstance(approved, dict) else True

    def bound_edge(key: str, edge: str) -> float:
        bound = bound_by_key.get(key)
        return (
            float(getattr(bound, edge))
            if bound is not None
            else float(base_by_key.get(key, 0.0))
        )

    for side in ("long", "short"):
        can_activate = (
            side_approved(side)
            and bound_edge(f"{side}_total_wallet_exposure_limit", "high") > 0.0
            and bound_edge(f"{side}_n_positions", "high") > 0.0
        )
        if can_activate != (side in enabled_sides):
            raise ValueError(
                f"GPU foundation requires {side} enabledness to remain fixed across "
                "the full search space; pin exposure/positions or approved coins"
            )
        if side not in enabled_sides:
            continue
        if bound_edge(f"{side}_total_wallet_exposure_limit", "low") <= 0.0:
            raise ValueError(
                f"GPU foundation requires {side} total_wallet_exposure_limit "
                "to remain positive across the full search space"
            )
        n_positions = (
            bound_edge(f"{side}_n_positions", "low"),
            bound_edge(f"{side}_n_positions", "high"),
        )
        if coin_count == 1 and n_positions != (1.0, 1.0):
            raise ValueError(
                f"GPU foundation requires {side}_n_positions pinned at 1; "
                f"got bounds {n_positions}"
            )
        if coin_count > 1 and not (
            1.0 <= n_positions[0] <= n_positions[1] <= float(coin_count)
        ):
            raise ValueError(
                f"GPU multicoin foundation requires {side}_n_positions bounds "
                f"within [1, {coin_count}]; got {n_positions}"
            )


def _validate_seed_side_match(config_enabled_sides, seed_enabled_sides) -> None:
    config_enabled_sides = set(config_enabled_sides)
    seed_enabled_sides = set(seed_enabled_sides)
    if config_enabled_sides != seed_enabled_sides:
        raise ValueError(
            "GPU foundation does not allow optimizer bounds to activate or disable "
            "a side relative to the input config; "
            f"config={sorted(config_enabled_sides)}, "
            f"bounds_clamped_seed={sorted(seed_enabled_sides)}"
        )


def _constraint_classification_mismatch(proxy_violation: float, exact_payload: dict) -> bool:
    if "G" not in exact_payload:
        return False
    exact = np.asarray(exact_payload["G"], dtype=np.float64).reshape(-1)
    proxy_feasible = bool(np.isfinite(proxy_violation) and proxy_violation <= 0.0)
    exact_feasible = bool(len(exact) and np.all(np.isfinite(exact) & (exact <= 0.0)))
    return proxy_feasible != exact_feasible


def _constraint_diagnostics(evaluator, proxy_metrics: dict, exact_payload: dict) -> list[dict]:
    """Describe proxy/exact values for every active optimizer limit."""

    proxy_suite = proxy_metrics.get(_GPU_SUITE_METRICS_KEY)
    exact_metrics = exact_payload.get("metrics") or {}
    exact_suite = exact_metrics.get("suite_metrics")
    exact_constraint_violation = float(
        exact_metrics.get("constraint_violation", 0.0) or 0.0
    )
    exact_failure_penalty = (
        exact_constraint_violation
        if exact_metrics.get("error") and exact_constraint_violation > 0.0
        else None
    )
    proxy_surface = (
        None if proxy_suite is not None else _single_scenario_metric_surface(proxy_metrics)
    )
    exact_surface = (
        None
        if exact_suite is not None
        else flatten_metric_stats(exact_metrics.get("stats") or {})
    )
    diagnostics = []
    for check in getattr(evaluator, "limit_checks", []):
        proxy_value = (
            _suite_limit_metric_value(proxy_suite, check)
            if proxy_suite is not None
            else resolve_metric_value(proxy_surface, check["metric_key"])
        )
        exact_value = (
            _suite_limit_metric_value(exact_suite, check)
            if exact_suite is not None
            else resolve_metric_value(exact_surface, check["metric_key"])
        )
        exact_limit_violation = (
            None
            if exact_value is None and exact_failure_penalty is not None
            else float(compute_limit_violation(check, exact_value))
        )
        entry = {
            "metric": check["metric"],
            "metric_key": check["metric_key"],
            "scenario": check.get("scenario"),
            "reducer": check.get("reducer"),
            "mode": check["mode"],
            "proxy_value": None if proxy_value is None else float(proxy_value),
            "exact_value": None if exact_value is None else float(exact_value),
            "proxy_violation": float(compute_limit_violation(check, proxy_value)),
            "exact_violation": exact_limit_violation,
        }
        if exact_limit_violation is None:
            entry["exact_failure_penalty"] = exact_failure_penalty
        if "bound" in check:
            entry["bound"] = float(check["bound"])
        if "range" in check:
            entry["range"] = [float(value) for value in check["range"]]
        diagnostics.append(entry)
    return diagnostics


def _format_constraint_diagnostics(diagnostics: list[dict]) -> str:
    differing = [
        item
        for item in diagnostics
        if item["exact_violation"] is None
        or (item["proxy_violation"] > 0.0) != (item["exact_violation"] > 0.0)
    ]
    selected = differing or diagnostics
    return "; ".join(
        f"{item['metric_key']}: proxy={item['proxy_value']} "
        f"exact={item['exact_value']} mode={item['mode']} "
        f"bound={item.get('bound', item.get('range'))} "
        f"scenario={item.get('scenario')} reducer={item.get('reducer')}"
        + (
            f" exact_failure_penalty={item['exact_failure_penalty']}"
            if item.get("exact_failure_penalty") is not None
            else ""
        )
        for item in selected
    )


def _recover_durable_validations(
    entries,
    *,
    start_index: int,
    stop_index: int,
    vector_from_entry,
    hash_vector,
) -> tuple[set[str], list[tuple[float, float, bool, bool, bool]]]:
    """Recover candidate identities and safety evidence after a stale checkpoint."""

    recovered: set[str] = set()
    drift_pairs: list[tuple[float, float, bool, bool, bool]] = []
    consumed = 0
    for index, entry in enumerate(entries):
        if index < start_index:
            continue
        if index >= stop_index:
            break
        recovered.add(hash_vector(vector_from_entry(entry)))
        metadata = (entry.get("metrics") or {}).get("gpu_validation")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "GPU resume cannot recover proxy/exact safety evidence from "
                f"durable result {index}"
            )
        if metadata.get("schema_version") != 2:
            raise RuntimeError(
                "GPU resume found unsupported proxy/exact safety evidence in "
                f"durable result {index}"
            )
        try:
            proxy_score = float(metadata["proxy_score"])
            exact_score = float(metadata["exact_score"])
            probe = metadata["probe"]
            proxy_front = metadata["proxy_front"]
            classification_mismatch = metadata["constraint_classification_mismatch"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "GPU resume found invalid proxy/exact safety evidence in "
                f"durable result {index}"
            ) from exc
        if (
            not isinstance(probe, bool)
            or not isinstance(proxy_front, bool)
            or probe == proxy_front
            or not isinstance(classification_mismatch, bool)
        ):
            raise RuntimeError(
                "GPU resume found invalid proxy/exact safety evidence in "
                f"durable result {index}"
            )
        if not np.isfinite(proxy_score) or not np.isfinite(exact_score):
            raise RuntimeError(
                "GPU resume found non-finite proxy/exact safety evidence in "
                f"durable result {index}"
            )
        drift_pairs.append(
            (
                proxy_score,
                exact_score,
                probe,
                classification_mismatch,
                proxy_front,
            )
        )
        consumed += 1
    expected = max(0, stop_index - start_index)
    if consumed != expected:
        raise RuntimeError(
            "GPU resume could not reconstruct all durable candidate hashes: "
            f"expected {expected}, recovered {consumed}"
        )
    return recovered, drift_pairs


def run_backend(
    *,
    config: dict[str, Any],
    evaluator,
    evaluator_for_pool,
    recorder,
    overrides_list,
    duplicate_counter,
    starting_configs_path: str | None,
    constraint_fitness_cls,
    ignore_sigint_in_worker,
    get_starting_configs,
    configs_to_individuals,
    iter_starting_configs=None,
    configs_to_individuals_streaming=None,
    optimization_shape=None,
    record_individual_result=None,
    run_evolution=None,
    build_config_fn=None,
    overrides_fn=None,
    checkpoint_path: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    del duplicate_counter
    del constraint_fitness_cls
    del record_individual_result
    del run_evolution

    from config.metrics import canonicalize_metric_name
    from config.scoring import extract_objective_specs
    from optimization.gpu.metrics import SUPPORTED_METRICS
    from optimization.gpu.service import MpsMulticoinEmaProxy, MpsSingleCoinProxy
    from optimization.warmup import (
        _finalize_optimizer_vector_config,
        validate_optimizer_effective_configs,
    )

    options = _resolve_options(config)
    logging.info("GPU optimizer options: %s", options)
    validate_optimizer_effective_configs(config)

    shape = (
        optimization_shape
        if optimization_shape is not None
        else evaluator.optimization_shape
    )
    bounds = list(shape.bounds)
    key_paths = list(shape.key_paths)
    if len(bounds) != len(key_paths):
        raise ValueError("GPU optimization shape bounds/key_paths length mismatch")

    sig_digits = config["optimize"]["round_to_n_significant_digits"]
    template_vectors = configs_to_individuals(
        [config], bounds, sig_digits, optimization_shape=shape
    )
    if not template_vectors:
        raise ValueError(
            "GPU backend could not build a seed vector from the input config"
        )
    base_vector = [float(value) for value in template_vectors[0]]

    strategy_kind = str(config["live"]["strategy_kind"]).strip().lower()
    gpu_optimizer_overrides = _validate_gpu_optimizer_overrides(
        overrides_list, strategy_kind
    )
    proxy_config = _materialize_gpu_override_template(
        config,
        overrides_list,
        finalize_fn=_finalize_optimizer_vector_config,
    )
    suite_enabled = _gpu_suite_enabled(config, evaluator, evaluator_for_pool)
    suite_inputs = (
        _gpu_suite_scenario_inputs(proxy_config, evaluator_for_pool)
        if suite_enabled
        else []
    )
    if suite_enabled:
        exchange = suite_inputs[0]["exchange"]
        min_coin_count, max_coin_count, suite_multicoin_side = (
            _gpu_suite_search_context(suite_inputs)
        )
        logging.info(
            "GPU suite prepared %d scenarios | coins=%d..%d | multicoin_side=%s",
            len(suite_inputs),
            min_coin_count,
            max_coin_count,
            suite_multicoin_side or "n/a",
        )
    else:
        exchange = _validate_scope(proxy_config, evaluator)
        min_coin_count = max_coin_count = int(
            evaluator.shared_hlcvs_np[exchange].shape[1]
        )
        suite_multicoin_side = None
    if strategy_kind == "ema_anchor" and max_coin_count > 1:
        multicoin_sides = (
            [suite_multicoin_side]
            if suite_multicoin_side is not None
            else [
                side
                for side in ("long", "short")
                if gpu_side_enabled(proxy_config, side)
            ]
        )
        if len(multicoin_sides) != 1:
            raise ValueError(
                "GPU multicoin foundation requires exactly one enabled side"
            )
        bound_map = _ema_multicoin_bound_map(
            multicoin_sides[0], gpu_optimizer_overrides
        )
    else:
        bound_map = GPU_STRATEGY_BOUND_MAPS[strategy_kind]

    fixed_bound_values, fixed_parameter_overrides = _gpu_fixed_bound_context(
        config,
        proxy_config,
        key_paths,
        bound_map,
    )

    anchor_parameter_overrides, anchor_fixed_bounds = (
        _build_anchor_parameter_context(config, bound_map)
    )
    mapped_all = {
        bound_map[bound_key]: (index, bounds[index])
        for index, (bound_key, _path) in enumerate(key_paths)
        if bound_key in bound_map
    }
    missing = sorted(set(bound_map.values()) - set(mapped_all))
    if anchor_parameter_overrides is None and missing:
        raise ValueError(
            f"GPU backend could not locate {strategy_kind} bounds for {missing}"
        )
    if anchor_parameter_overrides is not None:
        missing_from_anchors = sorted(
            name
            for name in missing
            if any(name not in overrides for overrides in anchor_parameter_overrides)
        )
        if missing_from_anchors:
            raise ValueError(
                "GPU anchored fine-tune could not resolve fixed proxy parameters "
                f"for {missing_from_anchors}"
            )
    approved = proxy_config.get("live", {}).get("approved_coins", {})

    def side_approved(side: str) -> bool:
        return bool(approved.get(side, [])) if isinstance(approved, dict) else True

    base_config_enabled_sides = {
        side for side in ("long", "short") if gpu_side_enabled(proxy_config, side)
    }
    config_enabled_sides = (
        {suite_multicoin_side}
        if suite_multicoin_side is not None
        else base_config_enabled_sides
    )
    base_by_key = {
        bound_key: float(base_vector[index])
        for index, (bound_key, _path) in enumerate(key_paths)
    }
    base_by_key.update(fixed_bound_values)
    if "mirror_short_from_long" in gpu_optimizer_overrides:
        _mirror_short_mapping(base_by_key)
    if anchor_parameter_overrides is not None:
        enabled_sides = set(config_enabled_sides)
        side_values = {}
    else:
        side_values = {}
        for bound_key, value in base_by_key.items():
            if bound_key in {
                "long_total_wallet_exposure_limit",
                "long_n_positions",
                "short_total_wallet_exposure_limit",
                "short_n_positions",
            }:
                side_values[bound_key] = value

        def vector_side_enabled(side: str) -> bool:
            return (
                side_approved(side)
                and side_values.get(f"{side}_total_wallet_exposure_limit", 0.0)
                > 0.0
                and side_values.get(f"{side}_n_positions", 0.0) > 0.0
            )

        enabled_sides = {
            side for side in ("long", "short") if vector_side_enabled(side)
        }
        if suite_multicoin_side is None:
            _validate_seed_side_match(config_enabled_sides, enabled_sides)
        else:
            # CPU suite setup requires symmetric approved coin lists. Effective
            # scenario overrides establish the common single-side topology and
            # are validated below after shadowing candidate bounds.
            enabled_sides = {suite_multicoin_side}
    if not enabled_sides:
        raise ValueError(
            "GPU bounds would disable both sides for exact validation; "
            f"effective seed values: {side_values}"
        )
    candidate_source_sides = _gpu_candidate_source_sides(
        enabled_sides, gpu_optimizer_overrides
    )
    mapped = {
        name: value
        for name, value in mapped_all.items()
        if name.split("_", 1)[0] in candidate_source_sides
    }
    active = [
        (name, index, bound)
        for name, (index, bound) in sorted(mapped.items(), key=lambda item: item[1][0])
        if bound.high > bound.low
        and name not in fixed_parameter_overrides
        and not (
            "mirror_short_from_long" in gpu_optimizer_overrides
            and name.startswith("short_")
        )
    ]
    active.extend(
        (ANCHOR_GENE_KEY, index, bounds[index])
        for index, (bound_key, _path) in enumerate(key_paths)
        if bound_key == ANCHOR_GENE_KEY and bounds[index].high > bounds[index].low
    )
    active.sort(key=lambda item: item[1])
    if not active:
        raise ValueError(
            f"GPU backend found no free {strategy_kind} dimensions"
        )

    bound_by_key = {
        bound_key: bounds[index] for index, (bound_key, _path) in enumerate(key_paths)
    }
    overlap = sorted(set(bound_by_key) & set(anchor_fixed_bounds))
    if overlap:
        raise ValueError(
            "GPU anchored fine-tune marks optimizer keys as both tunable and fixed: "
            f"{overlap}"
        )
    bound_by_key.update(anchor_fixed_bounds)
    bound_by_key.update(
        {
            bound_key: Bound(value, value)
            for bound_key, value in fixed_bound_values.items()
        }
    )
    if "mirror_short_from_long" in gpu_optimizer_overrides:
        _mirror_short_mapping(bound_by_key)
    _validate_pinned_scope_bounds(bound_by_key, base_by_key, enabled_sides)

    if suite_multicoin_side is None:
        _validate_directional_search_space(
            bound_by_key,
            base_by_key,
            approved,
            enabled_sides,
            coin_count=max_coin_count,
        )

    for item in suite_inputs:
        fixed_scenario_bounds, parameter_overrides = (
            _gpu_suite_scenario_override_context(
                proxy_config,
                item["config"],
                item["overrides"],
                bound_by_key,
                bound_map,
            )
        )
        scenario_bound_by_key = dict(bound_by_key)
        scenario_base_by_key = dict(base_by_key)
        for bound_key, value in fixed_scenario_bounds.items():
            scenario_bound_by_key[bound_key] = Bound(value, value)
            scenario_base_by_key[bound_key] = value
        scenario_enabled_sides = {
            side
            for side in ("long", "short")
            if gpu_side_enabled(item["config"], side)
        }
        _validate_pinned_scope_bounds(
            scenario_bound_by_key,
            scenario_base_by_key,
            scenario_enabled_sides,
        )
        _validate_directional_search_space(
            scenario_bound_by_key,
            scenario_base_by_key,
            item["config"].get("live", {}).get("approved_coins", {}),
            scenario_enabled_sides,
            coin_count=item["coin_count"],
        )
        item["parameter_overrides"] = parameter_overrides

    for bound_key, bound in bound_by_key.items():
        if bound_key == ANCHOR_GENE_KEY or bound.high <= bound.low:
            continue
        side = bound_key.split("_", 1)[0]
        if side in {"long", "short"} and side not in enabled_sides:
            continue
        if max_coin_count == 1 and any(
            bound_key.startswith(f"{side}_forager_") for side in enabled_sides
        ):
            # Forager ranking cannot affect a one-coin backtest.
            continue
        if any(
            bound_key.startswith(f"{side}_hsl_")
            or bound_key.startswith(f"{side}_unstuck_")
            for side in enabled_sides
        ):
            # The enabling flags are not optimizer bounds. Scope validation
            # requires both features off, so these dormant values cannot affect
            # either the exact Rust backtest or the proxy.
            continue
        if (
            max_coin_count == 1
            and bound_key in {f"{side}_n_positions" for side in enabled_sides}
        ):
            continue
        if bound_key not in bound_map:
            raise ValueError(
                "GPU foundation cannot optimize active bound "
                f"{bound_key!r}; pin it or use the CPU optimizer"
            )

    specs = extract_objective_specs(config)
    metric_names = [canonicalize_metric_name(spec.metric) for spec in specs]
    limit_metrics = {
        canonicalize_metric_name(check["metric"])
        for check in getattr(evaluator, "limit_checks", [])
    }
    needed_metrics = set(metric_names) | limit_metrics | {"backtest_completion_ratio"}
    unsupported = sorted(needed_metrics - set(SUPPORTED_METRICS))
    if unsupported:
        raise ValueError(
            f"GPU foundation does not implement optimizer metrics {unsupported}; "
            "use supported metrics or the CPU optimizer"
        )

    if suite_enabled:
        scenario_proxies = []
        for item in suite_inputs:
            scenario_proxy = (
                MpsMulticoinEmaProxy
                if item["coin_count"] > 1
                else MpsSingleCoinProxy
            )(
                config=item["config"],
                hlcvs=item["hlcvs"],
                mss=item["mss"],
                btc=item["btc"],
                timestamps=item["timestamps"],
                exchange=item["exchange"],
                batch_size=int(options["batch_size"]),
                needed_metrics=needed_metrics,
            )
            item["coin_override_contract"] = getattr(
                scenario_proxy, "coin_override_contract", {}
            )
            scenario_proxies.append(
                (item["ctx"], scenario_proxy, item["parameter_overrides"])
            )

        def evaluate_proxy(candidates):
            return _evaluate_gpu_suite_proxies(
                evaluator_for_pool,
                scenario_proxies,
                candidates,
            )

    else:
        proxy_cls = MpsMulticoinEmaProxy if max_coin_count > 1 else MpsSingleCoinProxy
        proxy = proxy_cls(
            config=proxy_config,
            hlcvs=evaluator.shared_hlcvs_np[exchange],
            mss=evaluator.msss[exchange],
            btc=evaluator.shared_btc_np[exchange],
            timestamps=evaluator.timestamps.get(exchange),
            exchange=exchange,
            batch_size=int(options["batch_size"]),
            needed_metrics=needed_metrics,
        )

        def evaluate_proxy(candidates):
            return proxy.evaluate(candidates)

    active_low = np.asarray(
        [bound.low for _name, _index, bound in active], dtype=np.float64
    )
    active_high = np.asarray(
        [bound.high for _name, _index, bound in active], dtype=np.float64
    )
    active_span = active_high - active_low
    active_bounds = [bound for _name, _index, bound in active]

    def normalized_to_values(rows: np.ndarray) -> np.ndarray:
        return _canonical_candidate_values(
            rows, active_low, active_span, active_bounds, sig_digits
        )

    def parameter_dicts(rows: np.ndarray) -> list[dict]:
        values = normalized_to_values(rows)
        return _build_proxy_parameter_dicts(
            base_vector,
            mapped,
            active,
            values,
            anchor_parameter_overrides=anchor_parameter_overrides,
            fixed_parameter_overrides=fixed_parameter_overrides,
            optimizer_overrides=gpu_optimizer_overrides,
        )

    def full_vector(row: np.ndarray) -> list[float]:
        result = list(base_vector)
        values = normalized_to_values(np.asarray(row, dtype=np.float64).reshape(1, -1))[
            0
        ]
        for column, (_name, index, _bound) in enumerate(active):
            result[index] = float(values[column])
        return result

    def normalize_vector(vector) -> np.ndarray:
        values = np.asarray([float(vector[index]) for _name, index, _bound in active])
        return np.clip((values - active_low) / active_span, 0.0, 1.0)

    def vector_hash(vector) -> str:
        vector = _canonicalize_optimizer_override_hash_vector(
            vector,
            base_vector,
            key_paths,
            gpu_optimizer_overrides,
            anchor_parameter_overrides=anchor_parameter_overrides,
            fixed_bound_values=fixed_bound_values,
            fixed_parameter_overrides=fixed_parameter_overrides,
        )
        return _canonical_vector_hash(vector, bounds, sig_digits)

    from pymoo.core.problem import Problem
    from pymoo.core.termination import NoTermination

    population_size = max(8, int(options["population_size"]))
    seed = config["optimize"].get("seed")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        logging.info("GPU optimizer generated reproducible seed %d", seed)
    seed = int(seed)
    rng = np.random.default_rng(seed)
    sampling = rng.random((population_size, len(active)))
    sampling[0] = normalize_vector(base_vector)
    starting_vectors = load_starting_individuals(
        starting_configs_path=starting_configs_path,
        population_size=population_size,
        get_starting_configs=get_starting_configs,
        configs_to_individuals=configs_to_individuals,
        iter_starting_configs=iter_starting_configs,
        configs_to_individuals_streaming=configs_to_individuals_streaming,
        optimization_shape=shape,
        bounds=bounds,
        sig_digits=sig_digits,
    )
    for index, vector in enumerate(starting_vectors[: population_size - 1], start=1):
        sampling[index] = normalize_vector(vector)

    problem = Problem(
        n_var=len(active),
        n_obj=len(specs),
        n_ieq_constr=1,
        xl=np.zeros(len(active)),
        xu=np.ones(len(active)),
    )
    algorithm = _build_gpu_nsga2(
        config,
        sampling=sampling,
        population_size=population_size,
        n_params=len(active),
    )
    algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
    generation = 0
    exact_done = 0
    completed_hashes: set[str] = set()
    objective_scale = _ObjectiveScale()
    drift_monitor = _DriftMonitor(options)
    persisted_halt_reason = None
    signature = _checkpoint_signature(
        active,
        config["optimize"]["scoring"],
        anchor_plan=get_anchor_plan(config),
        suite_contract=(
            _gpu_suite_checkpoint_contract(config, suite_inputs)
            if suite_enabled
            else None
        ),
        runtime_contract=(
            None
            if suite_enabled
            else getattr(proxy, "coin_override_contract", None)
        ),
    )
    budget = int(config["optimize"]["iters"])
    if budget <= 0:
        raise ValueError("optimize.iters must be greater than zero")

    if resume:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"GPU checkpoint not found: {checkpoint_path}")
        with open(checkpoint_path, "rb") as file:
            checkpoint = pickle.load(file)
        if checkpoint.get("signature") != signature:
            raise ValueError(
                "GPU checkpoint does not match current bounds, scoring, or suite contract"
            )
        algorithm = checkpoint["algorithm"]
        seed = int(checkpoint.get("seed", seed))
        generation = int(checkpoint["generation"])
        exact_done = int(checkpoint["exact_done"])
        completed_hashes = set(checkpoint.get("completed_hashes", []))
        objective_scale.median = checkpoint.get("scale_median")
        objective_scale.spread = checkpoint.get("scale_spread")
        drift_monitor.pairs.extend(checkpoint.get("drift_pairs", []))
        persisted_halt_reason = checkpoint.get("halt_reason")
        recorded_exact = int(getattr(getattr(recorder, "store", None), "n_iters", 0))
        if recorded_exact < exact_done:
            raise RuntimeError(
                "GPU checkpoint is ahead of durable all_results.bin state: "
                f"checkpoint={exact_done}, durable={recorded_exact}"
            )
        if recorded_exact > exact_done:
            from opt_utils import load_results

            results_file = getattr(getattr(recorder, "results_file", None), "name", None)
            if not results_file:
                raise RuntimeError(
                    "GPU resume cannot recover durable candidate hashes without "
                    "all_results.bin"
                )

            def vector_from_entry(entry):
                vectors = configs_to_individuals(
                    [entry], bounds, sig_digits, optimization_shape=shape
                )
                if len(vectors) != 1:
                    raise RuntimeError(
                        "GPU resume could not reconstruct a unique vector from a "
                        "durable result"
                    )
                return vectors[0]

            recovered_hashes, recovered_pairs = _recover_durable_validations(
                load_results(results_file),
                start_index=exact_done,
                stop_index=recorded_exact,
                vector_from_entry=vector_from_entry,
                hash_vector=vector_hash,
            )
            completed_hashes.update(recovered_hashes)
            drift_monitor.pairs.extend(recovered_pairs)
            recovered_drift_status = drift_monitor.evaluate()
            if recovered_drift_status["halt_reason"]:
                raise RuntimeError(
                    "Cannot resume a GPU run stopped by its drift safety gate: "
                    f"{recovered_drift_status['halt_reason']}"
                )
            logging.warning(
                "GPU checkpoint records %d exact evaluations but all_results.bin "
                "records %d; recovered the missing durable candidate identities "
                "and safety evidence",
                exact_done,
                recorded_exact,
            )
            exact_done = recorded_exact
        if persisted_halt_reason:
            raise RuntimeError(
                "Cannot resume a GPU run stopped by its drift safety gate: "
                f"{persisted_halt_reason}"
            )
        _validate_resume_evidence_budget(
            drift_monitor.pairs,
            exact_done=exact_done,
            exact_budget=budget,
            options=options,
        )
        logging.info(
            "Resumed GPU optimizer at generation %d with %d exact evaluations",
            generation,
            exact_done,
        )

    adapter = PymooEvaluatorAdapter(evaluator_for_pool, overrides_list=overrides_list)
    workers = int(options["exact_workers"]) or int(config["optimize"]["n_cpus"])
    max_pending = int(options["max_pending_exact"]) or workers * 2
    if workers <= 0:
        raise ValueError("GPU exact validation requires at least one CPU worker")
    if max_pending <= 0:
        raise ValueError(
            "optimize.gpu.max_pending_exact must be greater than zero when set"
        )
    initializer = functools.partial(
        initialize_pymoo_worker,
        evaluator_for_pool,
        overrides_list,
        len(specs),
        adapter.has_constraints,
        seed,
        ignore_sigint_in_worker,
    )
    pool = multiprocessing.Pool(processes=workers, initializer=initializer)
    pool_workers = tuple(getattr(pool, "_pool", ()) or ())
    pending = {}
    submitted_hashes: set[str] = set()
    start_time = time.time()
    proxy_evaluations = 0
    novelty_stall_generations = 0
    last_warning = None
    last_probe_shortfall = None
    last_checkpoint_at = 0.0
    last_checkpoint_exact = exact_done

    def checkpoint_state() -> dict:
        return {
            "signature": signature,
            "algorithm": algorithm,
            "generation": generation,
            "seed": seed,
            "exact_done": exact_done,
            "completed_hashes": sorted(completed_hashes),
            "scale_median": objective_scale.median,
            "scale_spread": objective_scale.spread,
            "drift_pairs": list(drift_monitor.pairs),
            "halt_reason": persisted_halt_reason,
        }

    def maybe_save_checkpoint(*, force: bool = False) -> None:
        nonlocal last_checkpoint_at, last_checkpoint_exact
        now = time.monotonic()
        due = now - last_checkpoint_at >= float(options["checkpoint_interval_seconds"])
        if not force and (exact_done == last_checkpoint_exact or not due):
            return
        _save_checkpoint(checkpoint_path, checkpoint_state())
        last_checkpoint_at = now
        last_checkpoint_exact = exact_done

    def proxy_fitness(metric_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        objectives = np.empty((len(metric_rows), len(specs)), dtype=np.float64)
        violations = np.empty(len(metric_rows), dtype=np.float64)
        for row, metrics in enumerate(metric_rows):
            if _GPU_SUITE_OBJECTIVES_KEY in metrics:
                objective_values = np.asarray(
                    metrics[_GPU_SUITE_OBJECTIVES_KEY], dtype=np.float64
                )
                if len(objective_values) != len(specs):
                    raise ValueError(
                        "GPU suite proxy objective length mismatch: "
                        f"expected {len(specs)}, got {len(objective_values)}"
                    )
                objectives[row] = objective_values
                violations[row] = float(metrics[_GPU_SUITE_VIOLATION_KEY])
                continue
            flattened = _single_scenario_metric_surface(metrics)
            objective_values = [
                metrics[canonicalize_metric_name(spec.metric)] for spec in specs
            ]
            fitness, violation = evaluator.calc_fitness(
                flattened,
                limit_metrics=flattened,
                objective_values=objective_values,
            )
            objectives[row] = fitness
            violations[row] = float(violation)
        return objectives, violations

    def record_exact(vector, payload, *, validation_metadata: dict) -> None:
        entry = build_pymoo_record_entry(
            vector=payload.get("evaluation_vector", vector),
            metrics=payload.get("metrics") or {},
            # Preserve the normalized run config. Evaluator.config contains
            # exchange-resolved runtime values (for example an inferred taker
            # fee), which are correct for Rust execution but would make the
            # persisted first result fail strict resume-contract comparison.
            template=config,
            build_config_fn=build_config_fn,
            overrides_fn=overrides_fn,
            overrides_list=overrides_list,
        )
        entry.setdefault("metrics", {})["gpu_validation"] = validation_metadata
        recorder.record(_restore_gpu_result_run_contract(entry, config))

    def consume_ready(*, wait_for_one: bool = False) -> None:
        nonlocal exact_done, last_warning, persisted_halt_reason
        while True:
            # Preserve submission/generation order in the durable evidence
            # stream. Workers may finish out of order, but later completions
            # wait behind the oldest pending result so resume guarantees match
            # the class allocation modeled above.
            ready = _ready_submission_prefix(pending)
            if ready:
                break
            if not wait_for_one or not pending:
                return
            PymooAsyncRecordingRunner._raise_if_pool_workers_exited(pool_workers)
            time.sleep(0.05)
        for result in ready:
            (
                vector,
                proxy_score,
                proxy_violation,
                proxy_metrics,
                is_probe,
                is_proxy_front,
                digest,
            ) = pending.pop(result)
            payload = result.get()
            PymooAsyncRecordingRunner._raise_if_worker_failure(payload, exact_done)
            exact_score = float(
                objective_scale.score(np.asarray(payload["F"]).reshape(1, -1))[0]
            )
            classification_mismatch = _constraint_classification_mismatch(
                proxy_violation, payload
            )
            constraint_diagnostics = _constraint_diagnostics(
                evaluator, proxy_metrics, payload
            )
            record_exact(
                vector,
                payload,
                validation_metadata={
                    "schema_version": 2,
                    "proxy_score": float(proxy_score),
                    "exact_score": exact_score,
                    "probe": bool(is_probe),
                    "proxy_front": bool(is_proxy_front),
                    "constraint_classification_mismatch": classification_mismatch,
                    "constraint_diagnostics": constraint_diagnostics,
                },
            )
            exact_done += 1
            completed_hashes.add(digest)
            if classification_mismatch:
                logging.warning(
                    "GPU %s proxy/exact constraint classification disagreed; exact Rust "
                    "classification remains authoritative and the mismatch is rolling "
                    "drift evidence: %s",
                    "broad-probe" if is_probe else "proxy-front",
                    _format_constraint_diagnostics(constraint_diagnostics),
                )
            drift_monitor.add(
                proxy_score,
                exact_score,
                probe=is_probe,
                proxy_front=is_proxy_front,
                constraint_mismatch=classification_mismatch,
            )
            status = drift_monitor.evaluate()
            if status["warn_reason"] and status["warn_reason"] != last_warning:
                logging.warning(status["warn_reason"])
                last_warning = status["warn_reason"]
            if status["halt_reason"]:
                persisted_halt_reason = status["halt_reason"]
                maybe_save_checkpoint(force=True)
                raise RuntimeError(status["halt_reason"])
        # ResultRecorder durably flushes each exact result. Keep the companion
        # optimizer state close behind so an interruption cannot substantially
        # overrun the requested exact-evaluation budget when resumed.
        maybe_save_checkpoint(force=True)

    try:
        while exact_done < budget:
            consume_ready()
            if exact_done + len(pending) >= budget:
                consume_ready(wait_for_one=True)
                continue
            validation_count = min(
                int(options["validate_per_generation"]),
                budget - exact_done - len(pending),
            )
            if max_pending - len(pending) < validation_count:
                consume_ready(wait_for_one=True)
                continue

            population = algorithm.ask()
            rows = np.asarray(population.get("X"), dtype=np.float64)
            metric_rows = evaluate_proxy(parameter_dicts(rows))
            proxy_objectives, proxy_violations = proxy_fitness(metric_rows)
            proxy_evaluations += len(rows)
            if objective_scale.median is None:
                objective_scale.fit(proxy_objectives)
            proxy_scores = objective_scale.score(proxy_objectives)
            population.set("F", proxy_objectives)
            population.set(
                "G",
                np.where(
                    np.isfinite(proxy_violations),
                    np.where(proxy_violations > 0.0, proxy_violations, -1.0),
                    1.0e18,
                ).reshape(-1, 1),
            )
            algorithm.tell(infills=population)
            generation += 1

            probe_count = _validation_probe_count(
                validation_count,
                int(options["validate_per_generation"]),
                int(options["drift_probes"]),
            )
            selections = _select_validation_indices(
                proxy_objectives,
                proxy_scores,
                proxy_violations,
                total=validation_count,
                probes=probe_count,
            )
            novel_selections = _select_novel_validations(
                selections,
                total=validation_count,
                candidate_for_index=lambda index: full_vector(rows[index]),
                digest_for_candidate=vector_hash,
                completed_hashes=completed_hashes,
                submitted_hashes=submitted_hashes,
            )
            actual_probe_count = sum(bool(item[1]) for item in novel_selections)
            last_probe_shortfall = _update_probe_shortfall_log(
                last_probe_shortfall,
                requested=probe_count,
                actual=actual_probe_count,
            )
            submitted_this_generation = 0
            for index, is_probe, is_proxy_front, vector, digest in novel_selections:
                result = pool.apply_async(
                    _evaluate_pymoo_worker_from_globals, (vector,)
                )
                pending[result] = (
                    vector,
                    float(proxy_scores[index]),
                    float(proxy_violations[index]),
                    dict(metric_rows[index]),
                    bool(is_probe),
                    bool(is_proxy_front),
                    digest,
                )
                submitted_hashes.add(digest)
                submitted_this_generation += 1

            novelty_stall_generations = _update_novelty_stall(
                novelty_stall_generations,
                submitted=submitted_this_generation,
                pending=len(pending),
            )

            if generation == 1 or generation % 10 == 0:
                elapsed = time.time() - start_time
                logging.info(
                    "GPU optimize | gen=%d proxy=%d (%.1f/s) exact=%d inflight=%d",
                    generation,
                    proxy_evaluations,
                    proxy_evaluations / max(elapsed, 1.0e-9),
                    exact_done,
                    len(pending),
                )
            maybe_save_checkpoint()

        while pending and exact_done < budget:
            consume_ready(wait_for_one=True)
        maybe_save_checkpoint(force=True)
        logging.info(
            "GPU optimization complete | generations=%d proxy=%d exact=%d wall=%.1fs",
            generation,
            proxy_evaluations,
            exact_done,
            time.time() - start_time,
        )
        return {"pool": pool, "pool_terminated": False}
    except BaseException:
        pool.terminate()
        pool.join()
        raise
