from __future__ import annotations

from collections import deque
from copy import deepcopy
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import time
from typing import Any

import numpy as np

from optimization.bounds import enforce_bounds
from optimization.callback import build_pymoo_record_entry
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

EMA_BOUND_MAP = {
    "long_base_qty_pct": "base_qty_pct",
    "long_ema_span_0": "ema_span_0",
    "long_ema_span_1": "ema_span_1",
    "long_entry_double_down_factor": "entry_double_down_factor",
    "long_offset": "offset",
    "long_offset_psize_weight": "offset_psize_weight",
    "long_offset_volatility_1h_weight": "offset_volatility_1h_weight",
    "long_offset_volatility_1m_weight": "offset_volatility_1m_weight",
    "long_offset_volatility_ema_span_1h": "offset_volatility_ema_span_1h",
    "long_offset_volatility_ema_span_1m": "offset_volatility_ema_span_1m",
    "long_risk_entry_cooldown_minutes": "entry_cooldown_minutes",
    "long_total_wallet_exposure_limit": "total_wallet_exposure_limit",
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
    if not 0.0 <= float(options["drift_halt"]) <= 1.0:
        raise ValueError("optimize.gpu.drift_halt must be between zero and one")
    if int(options["drift_min_samples"]) > int(options["drift_window"]):
        raise ValueError(
            "optimize.gpu.drift_min_samples must be less than or equal to "
            "optimize.gpu.drift_window"
        )
    if (
        int(options["drift_probes"]) > 0
        and int(options["drift_window"]) < MIN_DRIFT_PROBES
    ):
        raise ValueError(
            "optimize.gpu.drift_window must be at least "
            f"{MIN_DRIFT_PROBES} when optimize.gpu.drift_probes is enabled"
        )
    return options


def _gpu_side_enabled(config: dict, side: str) -> bool:
    risk = config.get("bot", {}).get(side, {}).get("risk", {})
    total_exposure = float(risk.get("total_wallet_exposure_limit", 0.0) or 0.0)
    n_positions = int(round(float(risk.get("n_positions", 0) or 0)))
    if total_exposure <= 0.0 or n_positions <= 0:
        return False
    approved = config.get("live", {}).get("approved_coins", {})
    if isinstance(approved, dict):
        return bool(approved.get(side, []))
    return True


def _validate_scope(config: dict, evaluator) -> str:
    if bool(config.get("backtest", {}).get("suite_enabled")):
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
    if config.get("coin_overrides"):
        raise ValueError("GPU foundation does not support coin_overrides")
    fixed_overrides = config.get("optimize", {}).get("fixed_runtime_overrides", {}) or {}
    inert_hsl_policy_overrides = {
        "bot.long.hsl.restart_after_red_policy",
        "bot.short.hsl.restart_after_red_policy",
    }
    unsupported_overrides = sorted(set(fixed_overrides) - inert_hsl_policy_overrides)
    if unsupported_overrides:
        raise ValueError(
            "GPU foundation does not support optimize.fixed_runtime_overrides for "
            f"{unsupported_overrides}; apply the values directly to the config"
        )
    if float(config.get("live", {}).get("max_realized_loss_pct", 1.0)) != 1.0:
        raise ValueError(
            "GPU foundation requires live.max_realized_loss_pct=1.0 because the "
            "screening proxy does not model the realized-loss gate"
        )
    exchanges = list(getattr(evaluator, "exchanges", []))
    if len(exchanges) != 1:
        raise ValueError(
            f"GPU foundation requires exactly one exchange, got {exchanges}"
        )
    exchange = exchanges[0]
    hlcvs = evaluator.shared_hlcvs_np[exchange]
    if int(hlcvs.shape[1]) != 1:
        raise ValueError(
            "GPU foundation supports exactly one coin; "
            f"prepared {int(hlcvs.shape[1])}"
        )
    strategy_kind = str(config.get("live", {}).get("strategy_kind", "")).lower()
    if "ema_anchor" not in strategy_kind:
        raise ValueError(
            "GPU foundation supports strategy_kind=ema_anchor only; "
            f"got {strategy_kind!r}"
        )
    if not _gpu_side_enabled(config, "long") or _gpu_side_enabled(config, "short"):
        raise ValueError(
            "GPU foundation supports long-only configs; enable long and disable short"
        )
    long_config = config["bot"]["long"]
    if bool(long_config.get("hsl", {}).get("enabled")):
        raise ValueError("GPU foundation requires bot.long.hsl.enabled=false")
    if bool(long_config.get("unstuck", {}).get("enabled")):
        raise ValueError("GPU foundation requires bot.long.unstuck.enabled=false")
    risk = long_config.get("risk", {})
    if bool(risk.get("position_exposure_enforcer_enabled")):
        raise ValueError(
            "GPU foundation requires "
            "bot.long.risk.position_exposure_enforcer_enabled=false"
        )
    if bool(risk.get("total_exposure_enforcer_enabled")):
        raise ValueError(
            "GPU foundation requires "
            "bot.long.risk.total_exposure_enforcer_enabled=false"
        )
    if float(risk.get("we_excess_allowance_pct", 0.0) or 0.0) != 0.0:
        raise ValueError(
            "GPU foundation requires bot.long.risk.we_excess_allowance_pct=0.0"
        )
    if not bool(risk.get("total_exposure_entry_gate_enabled", True)):
        raise ValueError(
            "GPU foundation requires "
            "bot.long.risk.total_exposure_entry_gate_enabled=true"
        )
    return exchange


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

    def __init__(self, options: dict):
        self.window = int(options["drift_window"])
        self.minimum = int(options["drift_min_samples"])
        self.halt = float(options["drift_halt"])
        self.pairs: deque[tuple[float, float, bool]] = deque(maxlen=self.window)

    def add(self, proxy_score: float, exact_score: float, *, probe: bool) -> None:
        self.pairs.append((float(proxy_score), float(exact_score), bool(probe)))

    def evaluate(self) -> dict:
        result = {
            "rho": float("nan"),
            "probe_rho": float("nan"),
            "front_rho": float("nan"),
            "samples": len(self.pairs),
            "probes": 0,
            "halt_reason": None,
            "warn_reason": None,
        }
        if len(self.pairs) < self.minimum:
            return result
        proxy = np.asarray([row[0] for row in self.pairs], dtype=np.float64)
        exact = np.asarray([row[1] for row in self.pairs], dtype=np.float64)
        probes = np.asarray([row[2] for row in self.pairs], dtype=bool)
        result["probes"] = int(probes.sum())
        result["rho"] = _spearman(proxy, exact)
        result["probe_rho"] = _spearman(proxy[probes], exact[probes])
        result["front_rho"] = _spearman(proxy[~probes], exact[~probes])
        detail = (
            f"rho={result['rho']:.3f}, probe_rho={result['probe_rho']:.3f}, "
            f"front_rho={result['front_rho']:.3f}, samples={result['samples']}, "
            f"probes={result['probes']}"
        )
        if result["probes"] >= self.MIN_PROBES and (
            not np.isfinite(result["probe_rho"])
            or result["probe_rho"] < self.halt
        ):
            result["halt_reason"] = (
                f"GPU proxy/exact broad-probe rank drift exceeded safety threshold ({detail})"
            )
        elif np.isfinite(result["rho"]) and result["rho"] >= self.halt:
            return result
        elif result["probes"] < self.MIN_PROBES:
            result["warn_reason"] = (
                f"GPU drift below threshold without enough broad probes ({detail})"
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
) -> list[tuple[int, bool]]:
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
    front_count = max(0, total - probes)
    elite_local = _normalized_farthest_indices(objectives[front], front_count)
    selected = [(int(front[index]), False) for index in elite_local]
    selected_ids = {index for index, _probe in selected}

    broad_pool = np.asarray(
        [
            int(index)
            for index in np.argsort(scores)
            if int(index) in primary_ids and int(index) not in front_ids
        ],
        dtype=np.int64,
    )
    requested_probes = min(max(0, probes), max(0, total - len(selected)))
    if len(broad_pool) < requested_probes:
        raise RuntimeError(
            "GPU validation cannot provide the requested independent broad-probe "
            f"evidence: requested {requested_probes}, available {len(broad_pool)} "
            "outside the complete feasible proxy Pareto front"
        )
    probe_count = requested_probes
    if probe_count:
        positions = np.round(
            np.linspace(0, len(broad_pool) - 1, num=probe_count)
        ).astype(int)
        for position in positions:
            index = int(broad_pool[position])
            if index not in selected_ids:
                selected.append((index, True))
                selected_ids.add(index)

    preferred_order = sorted(
        map(int, primary), key=lambda index: (float(violations[index]), float(scores[index]))
    )
    for index in preferred_order:
        if len(selected) >= total:
            break
        if index not in selected_ids:
            selected.append((index, index not in front_ids))
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
            selected.append((index, index not in front_ids and index in feasible_ids))
            selected_ids.add(index)
    return selected


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


def _checkpoint_signature(active, scoring) -> str:
    payload = {
        "active": [
            [name, int(index), float(bound.low), float(bound.high), bound.step]
            for name, index, bound in active
        ],
        "scoring": scoring,
        "version": 1,
    }
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


def _build_proxy_parameter_dicts(base_vector, mapped, active, active_values) -> list[dict]:
    """Include canonical pinned and active EMA values in every proxy candidate."""

    base_parameters = {
        name: float(base_vector[index]) for name, (index, _bound) in mapped.items()
    }
    result = []
    for row in active_values:
        parameters = dict(base_parameters)
        parameters.update(
            {
                name: float(row[column])
                for column, (name, _index, _bound) in enumerate(active)
            }
        )
        result.append(parameters)
    return result


def _constraint_classification_mismatch(proxy_violation: float, exact_payload: dict) -> bool:
    if "G" not in exact_payload:
        return False
    exact = np.asarray(exact_payload["G"], dtype=np.float64).reshape(-1)
    proxy_feasible = bool(np.isfinite(proxy_violation) and proxy_violation <= 0.0)
    exact_feasible = bool(len(exact) and np.all(np.isfinite(exact) & (exact <= 0.0)))
    return proxy_feasible != exact_feasible


def _recover_completed_hashes(
    entries,
    *,
    start_index: int,
    stop_index: int,
    vector_from_entry,
    hash_vector,
) -> set[str]:
    """Recover candidate identities durably recorded after a stale checkpoint."""

    recovered: set[str] = set()
    consumed = 0
    for index, entry in enumerate(entries):
        if index < start_index:
            continue
        if index >= stop_index:
            break
        recovered.add(hash_vector(vector_from_entry(entry)))
        consumed += 1
    expected = max(0, stop_index - start_index)
    if consumed != expected:
        raise RuntimeError(
            "GPU resume could not reconstruct all durable candidate hashes: "
            f"expected {expected}, recovered {consumed}"
        )
    return recovered


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
    del iter_starting_configs
    del configs_to_individuals_streaming
    del record_individual_result
    del run_evolution

    from config.metrics import canonicalize_metric_name
    from config.scoring import extract_objective_specs
    from optimization.gpu.metrics import SUPPORTED_METRICS
    from optimization.gpu.service import MpsEmaAnchorProxy

    exchange = _validate_scope(config, evaluator)
    options = _resolve_options(config)
    logging.info("GPU optimizer options: %s", options)

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

    mapped = {
        EMA_BOUND_MAP[bound_key]: (index, bounds[index])
        for index, (bound_key, _path) in enumerate(key_paths)
        if bound_key in EMA_BOUND_MAP
    }
    missing = sorted(set(EMA_BOUND_MAP.values()) - set(mapped))
    if missing:
        raise ValueError(f"GPU backend could not locate EMA bounds for {missing}")
    active = [
        (name, index, bound)
        for name, (index, bound) in sorted(mapped.items(), key=lambda item: item[1][0])
        if bound.high > bound.low
    ]
    if not active:
        raise ValueError("GPU backend found no free EMA-anchor dimensions")

    bound_by_key = {
        bound_key: bounds[index] for index, (bound_key, _path) in enumerate(key_paths)
    }
    base_by_key = {
        bound_key: float(base_vector[index])
        for index, (bound_key, _path) in enumerate(key_paths)
    }
    approved = config.get("live", {}).get("approved_coins", {})

    def side_approved(side: str) -> bool:
        return bool(approved.get(side, [])) if isinstance(approved, dict) else True

    def maximum_bound_value(key: str) -> float:
        bound = bound_by_key.get(key)
        return (
            float(bound.high) if bound is not None else float(base_by_key.get(key, 0.0))
        )

    short_can_activate = (
        side_approved("short")
        and maximum_bound_value("short_total_wallet_exposure_limit") > 0.0
        and maximum_bound_value("short_n_positions") > 0.0
    )
    if short_can_activate:
        raise ValueError(
            "GPU foundation requires short to remain disabled across the full search space; "
            "clear live.approved_coins.short or pin short exposure/positions off"
        )

    long_n_positions_bound = bound_by_key.get("long_n_positions")
    if long_n_positions_bound is None:
        long_n_positions_values = (
            base_by_key.get("long_n_positions", 0.0),
            base_by_key.get("long_n_positions", 0.0),
        )
    else:
        long_n_positions_values = (
            float(long_n_positions_bound.low),
            float(long_n_positions_bound.high),
        )
    if long_n_positions_values != (1.0, 1.0):
        raise ValueError(
            "GPU foundation requires long_n_positions to remain pinned at 1; "
            f"got bounds {long_n_positions_values}"
        )

    for index, (bound_key, _path) in enumerate(key_paths):
        if bounds[index].high <= bounds[index].low:
            continue
        if bound_key.startswith("short_"):
            continue
        if bound_key.startswith("long_forager_"):
            # Forager ranking cannot affect a one-coin backtest.
            continue
        if bound_key.startswith("long_hsl_") or bound_key.startswith("long_unstuck_"):
            # The enabling flags are not optimizer bounds. Scope validation
            # requires both features off, so these dormant values cannot affect
            # either the exact Rust backtest or the proxy.
            continue
        if bound_key == "long_n_positions":
            continue
        if bound_key not in EMA_BOUND_MAP:
            raise ValueError(
                "GPU foundation cannot optimize active bound "
                f"{bound_key!r}; pin it or use the CPU optimizer"
            )

    side_values = {}
    for index, (bound_key, _path) in enumerate(key_paths):
        if bound_key in {
            "long_total_wallet_exposure_limit",
            "long_n_positions",
            "short_total_wallet_exposure_limit",
            "short_n_positions",
        }:
            side_values[bound_key] = float(base_vector[index])

    def vector_side_enabled(side: str) -> bool:
        return (
            side_approved(side)
            and side_values.get(f"{side}_total_wallet_exposure_limit", 0.0) > 0.0
            and side_values.get(f"{side}_n_positions", 0.0) > 0.0
        )

    vector_long = vector_side_enabled("long")
    vector_short = vector_side_enabled("short")
    if not vector_long or vector_short:
        raise ValueError(
            "GPU bounds would make exact validation disagree with the long-only proxy; "
            "pin long exposure/positions on and short exposure/positions off "
            f"(effective seed values: {side_values})"
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

    proxy = MpsEmaAnchorProxy(
        config=config,
        hlcvs=evaluator.shared_hlcvs_np[exchange],
        mss=evaluator.msss[exchange],
        btc=evaluator.shared_btc_np[exchange],
        timestamps=evaluator.timestamps.get(exchange),
        exchange=exchange,
        batch_size=int(options["batch_size"]),
        needed_metrics=needed_metrics,
    )

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
        return _build_proxy_parameter_dicts(base_vector, mapped, active, values)

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
        return _canonical_vector_hash(vector, bounds, sig_digits)

    from pymoo.algorithms.moo.nsga2 import NSGA2
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
    if starting_configs_path:
        starting_configs = get_starting_configs(starting_configs_path)
        starting_vectors = configs_to_individuals(
            starting_configs,
            bounds,
            sig_digits,
            optimization_shape=shape,
        )
        for index, vector in enumerate(
            starting_vectors[: population_size - 1], start=1
        ):
            sampling[index] = normalize_vector(vector)

    problem = Problem(
        n_var=len(active),
        n_obj=len(specs),
        n_ieq_constr=1,
        xl=np.zeros(len(active)),
        xu=np.ones(len(active)),
    )
    algorithm = NSGA2(
        pop_size=population_size, sampling=sampling, eliminate_duplicates=True
    )
    algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
    generation = 0
    exact_done = 0
    completed_hashes: set[str] = set()
    objective_scale = _ObjectiveScale()
    drift_monitor = _DriftMonitor(options)
    persisted_halt_reason = None
    signature = _checkpoint_signature(active, config["optimize"]["scoring"])

    if resume:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"GPU checkpoint not found: {checkpoint_path}")
        with open(checkpoint_path, "rb") as file:
            checkpoint = pickle.load(file)
        if checkpoint.get("signature") != signature:
            raise ValueError("GPU checkpoint does not match current bounds and scoring")
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

            completed_hashes.update(
                _recover_completed_hashes(
                    load_results(results_file),
                    start_index=exact_done,
                    stop_index=recorded_exact,
                    vector_from_entry=vector_from_entry,
                    hash_vector=vector_hash,
                )
            )
            logging.warning(
                "GPU checkpoint records %d exact evaluations but all_results.bin "
                "records %d; recovered the missing durable candidate hashes",
                exact_done,
                recorded_exact,
            )
            exact_done = recorded_exact
        if persisted_halt_reason:
            raise RuntimeError(
                "Cannot resume a GPU run stopped by its drift safety gate: "
                f"{persisted_halt_reason}"
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

    def record_exact(vector, payload) -> None:
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
        recorder.record(_restore_gpu_result_run_contract(entry, config))

    def consume_ready(*, wait_for_one: bool = False) -> None:
        nonlocal exact_done, last_warning, persisted_halt_reason
        while True:
            ready = [result for result in pending if result.ready()]
            if ready:
                break
            if not wait_for_one or not pending:
                return
            PymooAsyncRecordingRunner._raise_if_pool_workers_exited(pool_workers)
            time.sleep(0.05)
        for result in ready:
            vector, proxy_score, proxy_violation, is_probe, digest = pending.pop(result)
            payload = result.get()
            PymooAsyncRecordingRunner._raise_if_worker_failure(payload, exact_done)
            record_exact(vector, payload)
            exact_done += 1
            completed_hashes.add(digest)
            if _constraint_classification_mismatch(proxy_violation, payload):
                persisted_halt_reason = (
                    "GPU proxy/exact constraint classification disagreed for an exact "
                    f"validation: proxy_violation={proxy_violation}, "
                    f"exact_G={np.asarray(payload['G']).tolist()}"
                )
                maybe_save_checkpoint(force=True)
                raise RuntimeError(persisted_halt_reason)
            exact_score = float(
                objective_scale.score(np.asarray(payload["F"]).reshape(1, -1))[0]
            )
            drift_monitor.add(proxy_score, exact_score, probe=is_probe)
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
        budget = int(config["optimize"]["iters"])
        if budget <= 0:
            raise ValueError("optimize.iters must be greater than zero")
        while exact_done < budget:
            consume_ready()
            if exact_done + len(pending) >= budget:
                consume_ready(wait_for_one=True)
                continue

            population = algorithm.ask()
            rows = np.asarray(population.get("X"), dtype=np.float64)
            metric_rows = proxy.evaluate(parameter_dicts(rows))
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

            capacity = min(
                max_pending - len(pending),
                budget - exact_done - len(pending),
            )
            validation_count = min(int(options["validate_per_generation"]), capacity)
            probe_count = min(int(options["drift_probes"]), validation_count)
            selections = _select_validation_indices(
                proxy_objectives,
                proxy_scores,
                proxy_violations,
                total=validation_count,
                probes=probe_count,
            )
            submitted_this_generation = 0
            for index, is_probe in selections:
                if submitted_this_generation >= validation_count:
                    break
                vector = full_vector(rows[index])
                digest = vector_hash(vector)
                if digest in completed_hashes or digest in submitted_hashes:
                    continue
                result = pool.apply_async(
                    _evaluate_pymoo_worker_from_globals, (vector,)
                )
                pending[result] = (
                    vector,
                    float(proxy_scores[index]),
                    float(proxy_violations[index]),
                    bool(is_probe),
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
