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
import platform
import subprocess
import time
from typing import Any

import numpy as np

from config.metrics import resolve_metric_value
from config.pnl_lookback import parse_pnls_max_lookback_days
from limit_utils import compute_limit_violation
from metrics_schema import flatten_metric_stats
from optimization.backend_shared import (
    cancel_pending_async_results,
    load_starting_individuals,
)
from optimization.bounds import Bound, enforce_bounds
from optimization.callback import build_pymoo_record_entry
from optimization.fine_tune_anchors import ANCHOR_GENE_KEY, get_anchor_plan
from optimization.gpu.metric_registry import (
    reject_configured_exact_only_gpu_metrics,
)
from optimization.gpu.model import (
    HSL_COIN_OVERRIDE_PATHS,
    MPS_MULTICOIN_MAX_COINS,
    TRAILING_MARTINGALE_GATE_MODE_OVERRIDE_PATH,
    gpu_side_enabled,
    validate_hsl_override_patch,
    validate_hsl_signal_topology,
)
from optimization.interrupts import (
    InterruptCheck,
    OptimizerBackendInterrupted,
    no_interrupt_requested,
)
from optimization.problem import (
    PymooAsyncRecordingRunner,
    PymooEvaluatorAdapter,
    _evaluate_pymoo_worker_from_globals,
    initialize_pymoo_worker,
)
from utils import to_standard_exchange_name


GPU_DEFAULTS = {
    "auto_lean_parallelism": True,
    "population_size": 1024,
    "batch_size": 4096,
    "max_dispatch_candidate_bars": 1_000_000_000,
    "checkpoint_interval_seconds": 5.0,
    "validate_per_generation": 8,
    "drift_probes": 4,
    "drift_window": 128,
    "drift_min_samples": 32,
    "drift_halt": 0.60,
    "exact_workers": 0,
    "max_pending_exact": 0,
    "seed_bootstrap": {
        "mode": "auto",
        "max_exact": 128,
    },
    "successive_halving": {
        "enabled": False,
        "history_fractions": [0.25, 0.5, 1.0],
        "survival_fraction": 0.5,
        "min_survivors": 64,
    },
}

GPU_SEED_BOOTSTRAP_MODES = frozenset({"auto", "exact", "screened", "legacy"})
GPU_SEED_BOOTSTRAP_PROBE_FRACTION = 0.25

GPU_LEAN_TM_POPULATION_SIZE = 2304
GPU_LEAN_TM_MAX_DISPATCH_CANDIDATE_BARS = 4_500_000_000

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


def _ask_gpu_population(algorithm, interrupt_check: InterruptCheck):
    """Start an ask/tell transaction only when shutdown has not been requested."""

    interrupt_check()
    return algorithm.ask()


def _disable_gpu_exact_duplicate_guard(evaluator) -> None:
    """GPU owns canonical submission deduplication and exact revalidation."""

    base_evaluator = getattr(evaluator, "base", evaluator)
    if hasattr(base_evaluator, "use_duplicate_guard"):
        base_evaluator.use_duplicate_guard = False


def _submit_gpu_exact_validation(
    pool,
    vector,
    interrupt_check: InterruptCheck,
    *,
    profile: bool = False,
):
    """Refuse new exact CPU work once the GPU interrupt latch is set."""

    interrupt_check()
    if profile:
        submitted_at = time.perf_counter()
        return pool.apply_async(
            _profiled_gpu_exact_worker, (vector, submitted_at)
        )
    return pool.apply_async(_evaluate_pymoo_worker_from_globals, (vector,))


def _profiled_gpu_exact_worker(vector, submitted_at):
    """Attach opt-in worker time without changing persisted exact evidence."""

    started = time.perf_counter()
    payload = _evaluate_pymoo_worker_from_globals(vector)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["__gpu_profile_queue_wait_seconds__"] = max(
            0.0, started - float(submitted_at)
        )
        payload["__gpu_profile_worker_seconds__"] = time.perf_counter() - started
    return payload


def _log_gpu_profile(event: str, **payload) -> None:
    logging.info(
        "[gpu-profile] %s",
        json.dumps(
            {"schema_version": 1, "event": event, **payload},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _gpu_profile_elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - float(started))


def _checkpoint_gpu_interrupt(
    *,
    generation_in_progress: bool,
    generation: int,
    exact_done: int,
    save_checkpoint,
) -> bool:
    """Checkpoint only complete ask/tell state during graceful GPU shutdown."""

    if generation_in_progress:
        logging.info(
            "GPU interrupt received between bounded MPS dispatches; "
            "discarding the incomplete proxy generation and retaining the "
            "last safe checkpoint | generation=%d exact=%d",
            generation,
            exact_done,
        )
        return False
    save_checkpoint(force=True)
    logging.info(
        "Saved GPU interrupt checkpoint | generation=%d exact=%d",
        generation,
        exact_done,
    )
    return True


_SINGLE_COIN_EXPOSURE_BOUND_SUFFIXES = {
    "risk_we_excess_allowance_pct": "we_excess_allowance_pct",
    "risk_twel_enforcer_threshold": "twel_enforcer_threshold",
}

_SINGLE_COIN_UNSTUCK_BOUND_SUFFIXES = {
    "unstuck_close_pct": "unstuck_close_pct",
    "unstuck_ema_dist": "unstuck_ema_dist",
    "unstuck_loss_allowance_pct": "unstuck_loss_allowance_pct",
    "unstuck_threshold": "unstuck_threshold",
}

_SINGLE_COIN_HSL_BOUND_SUFFIXES = {
    "hsl_cooldown_minutes_after_red": "hsl_cooldown_minutes_after_red",
    "hsl_ema_span_minutes": "hsl_ema_span_minutes",
    "hsl_red_threshold": "hsl_red_threshold",
}

EMA_STRATEGY_BOUND_MAP = {
    f"{side}_{bound_suffix}": f"{side}_{parameter}"
    for side in ("long", "short")
    for bound_suffix, parameter in _EMA_SIDE_BOUND_SUFFIXES.items()
}

EMA_BOUND_MAP = {
    **EMA_STRATEGY_BOUND_MAP,
    **{
        f"{side}_{bound_suffix}": f"{side}_{parameter}"
        for side in ("long", "short")
        for bound_suffix, parameter in _SINGLE_COIN_EXPOSURE_BOUND_SUFFIXES.items()
    },
    **{
        f"{side}_{bound_suffix}": f"{side}_{parameter}"
        for side in ("long", "short")
        for bound_suffix, parameter in _SINGLE_COIN_UNSTUCK_BOUND_SUFFIXES.items()
    },
    **{
        f"{side}_{bound_suffix}": f"{side}_{parameter}"
        for side in ("long", "short")
        for bound_suffix, parameter in _SINGLE_COIN_HSL_BOUND_SUFFIXES.items()
    },
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
        **EMA_STRATEGY_BOUND_MAP,
        **{
            f"{side}_{bound_suffix}": f"{side}_{parameter}"
            for bound_suffix, parameter in _SINGLE_COIN_EXPOSURE_BOUND_SUFFIXES.items()
        },
        **{
            f"{side}_{suffix}": f"{side}_{parameter}"
            for suffix, parameter in _EMA_MULTICOIN_SIDE_BOUND_SUFFIXES.items()
        },
        **{
            f"{side}_{bound_suffix}": f"{side}_{parameter}"
            for bound_suffix, parameter in _SINGLE_COIN_UNSTUCK_BOUND_SUFFIXES.items()
        },
        **{
            f"{side}_{bound_suffix}": f"{side}_{parameter}"
            for bound_suffix, parameter in _SINGLE_COIN_HSL_BOUND_SUFFIXES.items()
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
    "risk_wel_enforcer_threshold": "wel_enforcer_threshold",
    "total_wallet_exposure_limit": "total_wallet_exposure_limit",
}

TRAILING_MARTINGALE_STRATEGY_BOUND_MAP = {
    f"{side}_{bound_suffix}": f"{side}_{parameter}"
    for side in ("long", "short")
    for bound_suffix, parameter in _TM_SIDE_BOUND_SUFFIXES.items()
}

TRAILING_MARTINGALE_BOUND_MAP = {
    **TRAILING_MARTINGALE_STRATEGY_BOUND_MAP,
    **{
        f"{side}_{bound_suffix}": f"{side}_{parameter}"
        for side in ("long", "short")
        for bound_suffix, parameter in _SINGLE_COIN_EXPOSURE_BOUND_SUFFIXES.items()
    },
    **{
        f"{side}_{bound_suffix}": f"{side}_{parameter}"
        for side in ("long", "short")
        for bound_suffix, parameter in _SINGLE_COIN_UNSTUCK_BOUND_SUFFIXES.items()
    },
    **{
        f"{side}_{bound_suffix}": f"{side}_{parameter}"
        for side in ("long", "short")
        for bound_suffix, parameter in _SINGLE_COIN_HSL_BOUND_SUFFIXES.items()
    },
}

TRAILING_MARTINGALE_MULTICOIN_BOUND_MAPS = {
    side: {
        **TRAILING_MARTINGALE_STRATEGY_BOUND_MAP,
        **{
            f"{side}_{bound_suffix}": f"{side}_{parameter}"
            for bound_suffix, parameter in _SINGLE_COIN_EXPOSURE_BOUND_SUFFIXES.items()
        },
        **{
            f"{side}_{suffix}": f"{side}_{parameter}"
            for suffix, parameter in _EMA_MULTICOIN_SIDE_BOUND_SUFFIXES.items()
        },
        **{
            f"{side}_{bound_suffix}": f"{side}_{parameter}"
            for bound_suffix, parameter in _SINGLE_COIN_UNSTUCK_BOUND_SUFFIXES.items()
        },
        **{
            f"{side}_{bound_suffix}": f"{side}_{parameter}"
            for bound_suffix, parameter in _SINGLE_COIN_HSL_BOUND_SUFFIXES.items()
        },
    }
    for side in ("long", "short")
}

GPU_STRATEGY_BOUND_MAPS = {
    "ema_anchor": EMA_BOUND_MAP,
    "trailing_martingale": TRAILING_MARTINGALE_BOUND_MAP,
}

GPU_CAPABILITIES_DOC = "docs/optimizing.md#deliberate-current-limitations"
GPU_SUPPORTED_STRATEGY_KINDS = frozenset(GPU_STRATEGY_BOUND_MAPS)

GPU_SUPPORTED_OPTIMIZER_OVERRIDES = {
    "lossless_close_trailing",
    "mirror_short_from_long",
}

# These scenario-local values are consumed when each MPS proxy builds the same
# canonical backtest payload as exact Rust. Keep this allowlist explicit: data
# selection and unsupported execution/risk behavior must continue to fail
# closed instead of being accepted merely because the config path exists.
GPU_SUPPORTED_SUITE_NON_BOT_OVERRIDE_PATHS = {
    ("backtest", "dynamic_wel_by_tradability"),
    ("backtest", "filter_by_min_effective_cost"),
    ("backtest", "liquidation_threshold"),
    ("backtest", "maker_fee_override"),
    ("backtest", "market_order_slippage_pct"),
    ("backtest", "starting_balance"),
    ("backtest", "taker_fee_override"),
    ("coin_overrides",),
    ("live", "forager_score_hysteresis_pct"),
    ("live", "hedge_mode"),
    ("live", "hsl_signal_mode"),
    ("live", "market_order_near_touch_threshold"),
    ("live", "market_orders_allowed"),
    ("live", "max_realized_loss_pct"),
    ("live", "pnls_max_lookback_days"),
}


def _validate_gpu_static_scope(config: dict) -> str:
    """Reject immutable GPU limitations without touching data or optional runtime state."""

    strategy_kind = (
        str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
    )
    if strategy_kind not in GPU_SUPPORTED_STRATEGY_KINDS:
        detail = (
            "trailing_grid_v7 is deliberately outside the Apple MPS scope"
            if strategy_kind == "trailing_grid_v7"
            else f"strategy_kind={strategy_kind!r} is not implemented by Apple MPS"
        )
        raise ValueError(
            "Apple MPS GPU optimization supports live.strategy_kind=ema_anchor or "
            f"trailing_martingale only; {detail}. Use optimize.backend='pymoo' or "
            f"'deap' for this configuration. See {GPU_CAPABILITIES_DOC}."
        )

    btc_collateral_cap = float(
        config.get("backtest", {}).get("btc_collateral_cap", 0.0) or 0.0
    )
    if not math.isfinite(btc_collateral_cap) or btc_collateral_cap != 0.0:
        raise ValueError(
            "Apple MPS GPU optimization does not model "
            "backtest.btc_collateral_cap; the GPU screening proxy requires "
            f"backtest.btc_collateral_cap=0.0, got {btc_collateral_cap!r}. "
            "Set it to zero or use optimize.backend='pymoo' or 'deap'; exact Rust "
            "validation cannot make an unmodeled proxy search safe. "
            f"See {GPU_CAPABILITIES_DOC}."
        )
    return strategy_kind


def _validate_gpu_data_independent_scope(
    config: dict,
    *,
    allow_suite: bool = False,
) -> tuple[str, list[str], list[str]]:
    """Validate GPU behavior which does not depend on prepared candles or coin count."""

    strategy_kind = _validate_gpu_static_scope(config)
    if bool(config.get("backtest", {}).get("suite_enabled")) and not allow_suite:
        raise ValueError("Apple MPS GPU scope validation requires allow_suite=True")
    if bool(config.get("backtest", {}).get("filter_by_min_effective_cost")):
        liquidation_threshold = float(
            config.get("backtest", {}).get("liquidation_threshold", 0.0)
        )
        if not math.isfinite(liquidation_threshold) or liquidation_threshold <= 0.0:
            raise ValueError(
                "GPU min-effective-cost filtering requires a finite positive "
                "backtest.liquidation_threshold so the proxy has a proven lower "
                "balance bound"
            )
    max_realized_loss_pct = float(
        config.get("live", {}).get("max_realized_loss_pct", 1.0)
    )
    if not math.isfinite(max_realized_loss_pct) or max_realized_loss_pct < 0.0:
        raise ValueError(
            "GPU foundation requires a finite non-negative "
            "live.max_realized_loss_pct"
        )
    enabled_sides = [
        side for side in ("long", "short") if gpu_side_enabled(config, side)
    ]
    if not enabled_sides:
        raise ValueError("GPU foundation requires at least one enabled side")
    if bool(config.get("live", {}).get("market_orders_allowed")):
        threshold = float(
            config.get("live", {}).get("market_order_near_touch_threshold", 0.001)
            or 0.0
        )
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "GPU market execution requires a finite non-negative "
                "live.market_order_near_touch_threshold"
            )
    hsl_enabled_sides = [
        side for side in enabled_sides if _gpu_hsl_side_enabled(config, side)
    ]
    if hsl_enabled_sides:
        parse_pnls_max_lookback_days(
            config.get("live", {}).get("pnls_max_lookback_days", 30.0),
            field_name="live.pnls_max_lookback_days",
        )
        for side in hsl_enabled_sides:
            hsl = config["bot"][side].get("hsl", {})
            panic_order_type = str(
                hsl.get("panic_close_order_type", "limit")
            ).strip().lower()
            if panic_order_type not in {"limit", "market"}:
                raise ValueError(
                    f"GPU HSL requires bot.{side}.hsl.panic_close_order_type "
                    f"to be limit or market, got {panic_order_type!r}"
                )
    for side in enabled_sides:
        risk = config["bot"][side].get("risk", {})
        if strategy_kind != "trailing_martingale" and bool(
            risk.get("position_exposure_enforcer_enabled", False)
        ):
            raise ValueError(
                "GPU foundation requires "
                f"bot.{side}.risk.position_exposure_enforcer_enabled=false"
            )
    return strategy_kind, enabled_sides, hsl_enabled_sides


def _validate_gpu_suite_override_paths(
    proxy_config: dict,
    *,
    label: str,
    overrides: dict,
) -> None:
    """Reject suite override paths which have no modeled MPS shadow semantics."""

    from config.param_paths import require_existing_config_path

    for dotted_path in overrides:
        resolved = require_existing_config_path(proxy_config, dotted_path)
        bot_side_override = (
            len(resolved) >= 3
            and resolved[0] == "bot"
            and resolved[1] in {"long", "short"}
        )
        if (
            not bot_side_override
            and resolved not in GPU_SUPPORTED_SUITE_NON_BOT_OVERRIDE_PATHS
        ):
            raise ValueError(
                f"Apple MPS GPU suite scenario {label!r} override "
                f"{dotted_path!r} is outside the supported modeled scenario scope "
                "and has no screening semantics. Remove the override or use "
                "optimize.backend='pymoo' or 'deap'. "
                f"See {GPU_CAPABILITIES_DOC}."
            )


def validate_gpu_preparation_scope(
    config: dict,
    suite_cfg: dict | None = None,
    *,
    torch_module=None,
) -> None:
    """Fail before historical-data preparation when immutable MPS scope is invalid."""

    reject_configured_exact_only_gpu_metrics(config)
    suite_cfg = suite_cfg or {}
    suite_enabled = bool(suite_cfg.get("enabled"))
    strategy_kind, _enabled_sides, _hsl_enabled_sides = (
        _validate_gpu_data_independent_scope(
            config,
            allow_suite=suite_enabled,
        )
    )
    halving_config = (
        config.get("optimize", {}).get("gpu", {}).get("successive_halving", {})
        or {}
    )
    if not isinstance(halving_config, dict):
        raise TypeError("optimize.gpu.successive_halving must be an object")
    if bool(halving_config.get("enabled")) and (
        strategy_kind != "trailing_martingale" or suite_enabled
    ):
        raise ValueError(
            "optimize.gpu.successive_halving currently requires a non-suite, "
            "single-coin trailing_martingale optimization"
        )
    if bool(suite_cfg.get("enabled")):
        from optimization.warmup import _apply_config_overrides

        for index, scenario in enumerate(suite_cfg.get("scenarios") or []):
            label = str(scenario.get("label") or f"scenario_{index + 1:02d}")
            overrides = scenario.get("overrides") or {}
            _validate_gpu_suite_override_paths(
                config,
                label=label,
                overrides=overrides,
            )
            scenario_config = deepcopy(config)
            _apply_config_overrides(scenario_config, overrides)
            _validate_gpu_data_independent_scope(
                scenario_config,
                allow_suite=True,
            )

    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            raise ModuleNotFoundError(
                "Apple MPS GPU optimization requires the optional 'gpu-mps' "
                "dependencies; install Passivbot with "
                "`pip install -e '.[full,gpu-mps]'`. "
                f"See {GPU_CAPABILITIES_DOC}."
            ) from exc
    if not torch_module.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS GPU optimization was requested, but MPS is unavailable in "
            "this process. Run on Apple Silicon with an MPS-enabled PyTorch build, "
            "or use optimize.backend='pymoo' or 'deap'. "
            f"See {GPU_CAPABILITIES_DOC}."
        )

    logging.info(
        "GPU capability preflight passed | runtime=apple_mps | strategy=%s | "
        "btc_collateral_cap=0 | max_coins_per_scenario=%d",
        strategy_kind,
        MPS_MULTICOIN_MAX_COINS,
    )


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


def materialize_gpu_preparation_config(config: dict) -> dict:
    """Finalize the immutable GPU template while preserving its strategy-shape guard."""

    return _materialize_gpu_override_template(
        config,
        config.get("optimize", {}).get("enable_overrides", []),
    )


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


def _gpu_candidate_search_sides(proxy_config: dict, suite_inputs) -> set[str]:
    """Return every side enabled by an effective independent scenario."""

    configs = (
        [item["config"] for item in suite_inputs]
        if suite_inputs
        else [proxy_config]
    )
    return {
        side
        for side in ("long", "short")
        if any(gpu_side_enabled(item, side) for item in configs)
    }


def _ema_multicoin_bound_map(target_side: str, overrides: set[str]) -> dict:
    """Include all bound families that can feed the enabled multicoin side."""

    bound_map = dict(EMA_MULTICOIN_BOUND_MAPS[target_side])
    if "mirror_short_from_long" in overrides and target_side == "short":
        bound_map.update(EMA_MULTICOIN_BOUND_MAPS["long"])
    return bound_map


def _trailing_martingale_multicoin_bound_map(
    target_side: str, overrides: set[str]
) -> dict:
    """Include all bound families feeding one single-side multicoin TM run."""

    bound_map = dict(TRAILING_MARTINGALE_MULTICOIN_BOUND_MAPS[target_side])
    if "mirror_short_from_long" in overrides and target_side == "short":
        bound_map.update(TRAILING_MARTINGALE_MULTICOIN_BOUND_MAPS["long"])
    return bound_map


def _minimum_rank_evidence_samples(halt: float) -> int:
    """Total samples needed to guarantee eight comparable at agreement >= halt."""

    if not 0.0 < float(halt) <= 1.0:
        raise ValueError("GPU drift_halt must be greater than zero and at most one")
    return math.floor((MIN_DRIFT_PROBES - 1) / float(halt)) + 1


def _build_gpu_nsga2(
    config,
    *,
    sampling,
    population_size: int,
    n_params: int,
    policy: dict | None = None,
):
    """Build GPU proposal evolution with the same variation controls as pymoo CPU."""

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    policy = policy or _gpu_nsga2_checkpoint_contract(
        config, population_size=population_size, n_params=n_params
    )
    return NSGA2(
        pop_size=int(policy["population_size"]),
        sampling=sampling,
        crossover=SBX(
            prob_var=float(policy["crossover"]["prob_var"]),
            eta=float(policy["crossover"]["eta"]),
        ),
        mutation=PM(
            prob=float(policy["mutation"]["prob"]),
            eta=float(policy["mutation"]["eta"]),
        ),
        eliminate_duplicates=bool(policy["eliminate_duplicates"]),
    )


def _gpu_nsga2_checkpoint_contract(
    config: dict, *, population_size: int, n_params: int
) -> dict:
    """Return the effective proposal policy serialized inside checkpoints."""

    from optimization.backends.pymoo_backend import (
        _resolve_mutation_prob,
        _resolve_pymoo_shared,
    )

    shared = _resolve_pymoo_shared(config)
    configured_seed = config.get("optimize", {}).get("seed")
    return {
        "version": 1,
        "algorithm": "nsga2",
        "population_size": int(population_size),
        "configured_seed": (
            None if configured_seed is None else int(configured_seed)
        ),
        "crossover": {
            "operator": "sbx",
            "prob_var": float(shared["crossover_prob_var"]),
            "eta": float(shared["crossover_eta"]),
        },
        "mutation": {
            "operator": "pm",
            "prob": float(_resolve_mutation_prob(shared, n_params)),
            "eta": float(shared["mutation_eta"]),
        },
        "eliminate_duplicates": bool(shared["eliminate_duplicates"]),
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


def _evaluate_gpu_suite_proxies(suite_evaluator, scenario_proxies, candidates) -> list[dict]:
    """Screen one candidate batch across suite scenarios with canonical reducers."""

    from metrics_schema import build_scenario_metrics
    from suite_runner import ScenarioResult, SuiteScenario

    scenario_rows = []
    for ctx, exchange_proxies, parameter_overrides in scenario_proxies:
        scenario_candidates = (
            [dict(candidate, **parameter_overrides) for candidate in candidates]
            if parameter_overrides
            else candidates
        )
        exchange_rows = []
        for exchange, proxy in exchange_proxies:
            rows = proxy.evaluate(scenario_candidates)
            if len(rows) != len(candidates):
                raise RuntimeError(
                    f"GPU suite scenario {ctx.label!r} exchange {exchange!r} "
                    "returned an unexpected proxy row count: "
                    f"expected {len(candidates)}, got {len(rows)}"
                )
            exchange_rows.append((exchange, rows))
        if not exchange_rows:
            raise ValueError(
                f"GPU suite scenario {ctx.label!r} has no prepared proxy datasets"
            )
        scenario_rows.append((ctx, exchange_rows))
    results = []
    for index in range(len(candidates)):
        scenario_results = []
        for ctx, exchange_rows in scenario_rows:
            per_exchange = {
                exchange: rows[index] for exchange, rows in exchange_rows
            }
            scenario_results.append(
                ScenarioResult(
                    scenario=SuiteScenario(
                        label=ctx.label,
                        start_date=None,
                        end_date=None,
                        coins=None,
                        ignored_coins=None,
                    ),
                    per_exchange=per_exchange,
                    metrics=build_scenario_metrics(per_exchange),
                    elapsed_seconds=0.0,
                    output_path=None,
                )
            )
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
    nested_options = {"seed_bootstrap", "successive_halving"}
    for key, default in GPU_DEFAULTS.items():
        if key in nested_options:
            continue
        if key in (configured or {}) and configured[key] is not None:
            options[key] = type(default)(configured[key])
    seed_bootstrap = dict(GPU_DEFAULTS["seed_bootstrap"])
    configured_seed_bootstrap = (configured or {}).get("seed_bootstrap")
    if configured_seed_bootstrap is not None and not isinstance(
        configured_seed_bootstrap, dict
    ):
        raise TypeError("optimize.gpu.seed_bootstrap must be an object")
    seed_bootstrap.update(configured_seed_bootstrap or {})
    unknown_seed_bootstrap = sorted(
        set(seed_bootstrap) - set(GPU_DEFAULTS["seed_bootstrap"])
    )
    if unknown_seed_bootstrap:
        raise ValueError(
            "unknown optimize.gpu.seed_bootstrap settings: "
            + ", ".join(unknown_seed_bootstrap)
        )
    seed_bootstrap["mode"] = str(seed_bootstrap["mode"]).strip().lower()
    if seed_bootstrap["mode"] not in GPU_SEED_BOOTSTRAP_MODES:
        allowed = ", ".join(sorted(GPU_SEED_BOOTSTRAP_MODES))
        raise ValueError(
            "optimize.gpu.seed_bootstrap.mode must be one of "
            f"{{{allowed}}}"
        )
    seed_bootstrap["max_exact"] = int(seed_bootstrap["max_exact"])
    if seed_bootstrap["max_exact"] <= 0:
        raise ValueError(
            "optimize.gpu.seed_bootstrap.max_exact must be greater than zero"
        )
    options["seed_bootstrap"] = seed_bootstrap
    halving = dict(GPU_DEFAULTS["successive_halving"])
    configured_halving = (configured or {}).get("successive_halving")
    if configured_halving is not None and not isinstance(configured_halving, dict):
        raise TypeError("optimize.gpu.successive_halving must be an object")
    halving.update(configured_halving or {})
    unknown_halving = sorted(
        set(halving) - set(GPU_DEFAULTS["successive_halving"])
    )
    if unknown_halving:
        raise ValueError(
            "unknown optimize.gpu.successive_halving settings: "
            + ", ".join(unknown_halving)
        )
    halving["enabled"] = bool(halving["enabled"])
    try:
        fractions = [float(value) for value in halving["history_fractions"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "optimize.gpu.successive_halving.history_fractions must be an array "
            "of finite fractions"
        ) from exc
    if (
        not fractions
        or any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in fractions
        )
        or any(right <= left for left, right in zip(fractions, fractions[1:]))
        or not math.isclose(fractions[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(
            "optimize.gpu.successive_halving.history_fractions must be strictly "
            "increasing finite values in (0, 1] ending at 1.0"
        )
    fractions[-1] = 1.0
    survival_fraction = float(halving["survival_fraction"])
    if not math.isfinite(survival_fraction) or not 0.0 < survival_fraction <= 1.0:
        raise ValueError(
            "optimize.gpu.successive_halving.survival_fraction must be greater "
            "than zero and at most one"
        )
    min_survivors = int(halving["min_survivors"])
    if min_survivors <= 0:
        raise ValueError(
            "optimize.gpu.successive_halving.min_survivors must be greater than zero"
        )
    halving.update(
        history_fractions=fractions,
        survival_fraction=survival_fraction,
        min_survivors=min_survivors,
    )
    options["successive_halving"] = halving
    for key in (
        "population_size",
        "batch_size",
        "max_dispatch_candidate_bars",
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
    if halving["enabled"] and min(
        min_survivors, int(options["population_size"])
    ) < validations:
        raise ValueError(
            "optimize.gpu.successive_halving.min_survivors must be at least "
            "optimize.gpu.validate_per_generation"
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
    # A complete feasible proxy Pareto front may contain only one candidate.
    # All remaining validation slots are truthfully broad/off-front evidence,
    # so budget for the worst-case one true-front sample per full generation
    # rather than assuming a fixed front/probe split. A repeated member is
    # exactly revalidated to preserve that sample.
    required_front_window = MIN_DRIFT_PROBES * validations
    if int(options["drift_window"]) < required_front_window:
        raise ValueError(
            "optimize.gpu.drift_window must be at least "
            f"{required_front_window} to retain {MIN_DRIFT_PROBES} true "
            "proxy-front validations when a complete proxy front contains "
            "only one candidate per generation"
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


def _bound_proves(
    bound_by_key: dict[str, Bound], key: str, predicate
) -> bool:
    bound = bound_by_key.get(key)
    return bool(
        bound is not None
        and predicate(float(bound.low))
        and predicate(float(bound.high))
    )


def _gpu_lean_tm_parallelism_eligible(
    config: dict,
    bound_by_key: dict[str, Bound],
    enabled_sides,
    *,
    suite_enabled: bool,
    coin_count: int,
    requested_metric_features,
) -> bool:
    """Prove the measured one-side TM kernel shape before widening dispatches."""

    if (
        suite_enabled
        or int(coin_count) != 1
        or bool(requested_metric_features)
        or bool(config.get("coin_overrides"))
        or str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
        != "trailing_martingale"
    ):
        return False
    sides = sorted(str(side) for side in enabled_sides)
    if len(sides) != 1 or sides[0] not in {"long", "short"}:
        return False
    side = sides[0]
    bot = config.get("bot", {}).get(side, {})
    risk = bot.get("risk", {})
    live = config.get("live", {})
    if (
        bool(bot.get("hsl", {}).get("enabled", False))
        or bool(bot.get("unstuck", {}).get("enabled", False))
        or bool(risk.get("position_exposure_enforcer_enabled", False))
        or bool(risk.get("total_exposure_enforcer_enabled", False))
        or bool(live.get("market_orders_allowed", False))
    ):
        return False
    try:
        max_realized_loss_pct = float(live.get("max_realized_loss_pct", 1.0))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(max_realized_loss_pct) or max_realized_loss_pct < 1.0:
        return False
    if not all(
        _bound_proves(bound_by_key, f"{side}_{suffix}", lambda value: value > 0.0)
        for suffix in (
            "entry_retracement_base_pct",
            "close_retracement_base_pct",
        )
    ):
        return False
    return all(
        _bound_proves(
            bound_by_key,
            f"{side}_{suffix}",
            lambda value: value == 0.0,
        )
        for suffix in (
            "entry_threshold_volatility_1h_weight",
            "entry_threshold_volatility_1m_weight",
            "entry_retracement_volatility_1h_weight",
            "entry_retracement_volatility_1m_weight",
            "close_threshold_volatility_1h_weight",
            "close_threshold_volatility_1m_weight",
            "close_retracement_volatility_1h_weight",
            "close_retracement_volatility_1m_weight",
        )
    )


@functools.lru_cache(maxsize=1)
def _apple_mps_chip_name() -> str:
    """Return the Apple chip name without importing optional GPU packages."""

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return ""
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _apply_gpu_lean_tm_parallelism_defaults(
    options: dict,
    config: dict,
    bound_by_key: dict[str, Bound],
    enabled_sides,
    *,
    suite_enabled: bool,
    coin_count: int,
    requested_metric_features,
    mps_chip_name: str | None = None,
) -> bool:
    """Apply the M3-tested width only when sizing is otherwise untouched."""

    if not bool(options.get("auto_lean_parallelism", True)):
        return False
    effective_chip_name = (
        _apple_mps_chip_name() if mps_chip_name is None else str(mps_chip_name)
    )
    if not effective_chip_name.startswith("Apple M3"):
        return False
    sizing_keys = (
        "population_size",
        "batch_size",
        "max_dispatch_candidate_bars",
    )
    if any(options[key] != GPU_DEFAULTS[key] for key in sizing_keys):
        return False
    configured_gpu = config.get("optimize", {}).get("gpu", {}) or {}
    if any(configured_gpu.get(key) is not None for key in sizing_keys):
        return False
    if not _gpu_lean_tm_parallelism_eligible(
        config,
        bound_by_key,
        enabled_sides,
        suite_enabled=suite_enabled,
        coin_count=coin_count,
        requested_metric_features=requested_metric_features,
    ):
        return False
    options["population_size"] = GPU_LEAN_TM_POPULATION_SIZE
    options["max_dispatch_candidate_bars"] = (
        GPU_LEAN_TM_MAX_DISPATCH_CANDIDATE_BARS
    )
    return True


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


def _tm_market_mode_value_supported(
    value, *, allow_recursive: bool = False
) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
        return False
    return allow_recursive or value > 0.0


def _validate_tm_multicoin_market_runtime_scope(
    config: dict, enabled_sides
) -> None:
    """Reject multi-coin TM market features not modeled by this slice."""

    unsupported = []
    for side in sorted(set(enabled_sides or ())):
        side_config = config.get("bot", {}).get(side, {}) or {}
        strategy = (
            side_config.get("strategy", {}).get("trailing_martingale", {}) or {}
        )
        for branch in ("entry", "close"):
            value = (strategy.get(branch, {}) or {}).get(
                "retracement_base_pct", 0.0
            )
            mode_supported = _tm_market_mode_value_supported(
                value, allow_recursive=True
            )
            if not mode_supported:
                requirement = "finite and float32-representable"
                unsupported.append(
                    f"bot.{side}.strategy.trailing_martingale.{branch}."
                    f"retracement_base_pct={value!r} "
                    f"(must remain {requirement})"
                )

    overrides = config.get("coin_overrides", {}) or {}
    if isinstance(overrides, dict):
        for coin, patch in overrides.items():
            if not isinstance(patch, dict):
                continue
            bot_patch = patch.get("bot", {}) or {}
            if not isinstance(bot_patch, dict):
                continue
            for side in sorted(set(enabled_sides or ())):
                side_patch = bot_patch.get(side, {}) or {}
                if not isinstance(side_patch, dict):
                    continue
                strategy_root = side_patch.get("strategy", {}) or {}
                strategy_patch = (
                    strategy_root.get("trailing_martingale", {}) or {}
                    if isinstance(strategy_root, dict)
                    else {}
                )
                if not isinstance(strategy_patch, dict):
                    strategy_patch = {}
                for branch in ("entry", "close"):
                    branch_patch = strategy_patch.get(branch, {}) or {}
                    if not isinstance(branch_patch, dict):
                        continue
                    if "retracement_base_pct" not in branch_patch:
                        continue
                    value = branch_patch["retracement_base_pct"]
                    mode_supported = _tm_market_mode_value_supported(
                        value, allow_recursive=True
                    )
                    if not mode_supported:
                        requirement = "finite and float32-representable"
                        unsupported.append(
                            f"coin_overrides.{coin}.bot.{side}.strategy."
                            f"trailing_martingale.{branch}."
                            f"retracement_base_pct={value!r} "
                            f"(must remain {requirement})"
                        )
    if unsupported:
        raise ValueError(
            "GPU multi-coin Trailing Martingale ordinary market execution "
            "requires finite float32-representable ordinary entry and close "
            "retracement bases and supports position/TWEL/auto-unstuck reducers; "
            "settings: "
            + ", ".join(unsupported)
        )


def _validate_scope_config(
    config: dict,
    *,
    exchanges,
    coin_count: int,
    allow_suite: bool = False,
) -> str:
    strategy_kind, enabled_sides, hsl_enabled_sides = (
        _validate_gpu_data_independent_scope(
            config,
            allow_suite=allow_suite,
        )
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
    if bool(config.get("live", {}).get("market_orders_allowed")):
        if coin_count > 1:
            if strategy_kind == "trailing_martingale":
                _validate_tm_multicoin_market_runtime_scope(
                    config, enabled_sides
                )
    if coin_count > 1:
        if coin_count > MPS_MULTICOIN_MAX_COINS:
            raise ValueError(
                "Apple MPS GPU optimization supports at most "
                f"{MPS_MULTICOIN_MAX_COINS} prepared coins per scenario; got "
                f"{coin_count}. Reduce the scenario coin universe or use "
                "optimize.backend='pymoo' or 'deap'."
            )
        if len(enabled_sides) not in (1, 2):
            raise ValueError(
                "GPU multicoin foundation requires one or two enabled sides"
            )
    _validate_gpu_coin_overrides(
        config,
        strategy_kind=strategy_kind,
        enabled_sides=enabled_sides,
        coin_count=coin_count,
    )
    if hsl_enabled_sides:
        # Directional single-coin coin mode mirrors Rust's finite fill-event
        # PnL window. Other HSL topologies retain the conservative all-history
        # screening envelope; exact Rust validation remains authoritative.
        signal_mode = str(
            config.get("live", {}).get("hsl_signal_mode", "unified")
        ).strip().lower()
        validate_hsl_signal_topology(
            signal_mode,
            coin_count=coin_count,
            enabled_side_count=len(enabled_sides),
            shared_account_controller=(
                coin_count > 1 and len(enabled_sides) == 2
            ),
        )
    return exchange


def _validate_dual_multicoin_metrics(
    needed_metrics,
    *,
    coin_count: int,
    enabled_sides,
    shared_account_controller: bool = False,
) -> None:
    """Reject metrics which cannot be reconstructed from directional summaries."""

    if (
        int(coin_count) <= 1
        or len(set(enabled_sides)) != 2
        or shared_account_controller
    ):
        return
    unsupported = sorted(
        set(needed_metrics)
        & {
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
        }
    )
    if unsupported:
        raise ValueError(
            "GPU dual-side multicoin optimization cannot safely reconstruct proxy "
            f"metrics {unsupported} from independent directional summaries; "
            "use other metrics or the CPU optimizer"
        )


def _validate_hsl_metric_topology(
    needed_metrics,
    *,
    coin_count: int,
    enabled_sides,
    hard_stop_metrics,
    shared_account_controller: bool = False,
) -> None:
    shared_account_metrics = {
        "hard_stop_halt_to_restart_equity_loss_pct",
        "hard_stop_panic_close_loss_drawdown_pct_max",
        "hard_stop_panic_close_loss_drawdown_pct_mean",
        "hard_stop_panic_close_loss_drawdown_pct_min",
    }
    tier_overlap_metrics = {
        "hard_stop_time_in_yellow_pct",
        "hard_stop_time_in_orange_pct",
        "hard_stop_time_in_red_pct",
    }
    unsupported = sorted(
        set(needed_metrics)
        & set(hard_stop_metrics)
        & (shared_account_metrics | tier_overlap_metrics)
    )
    if (
        int(coin_count) > 1
        and len(set(enabled_sides)) > 1
        and unsupported
        and not shared_account_controller
    ):
        raise ValueError(
            "GPU dual-side multi-coin HSL metrics require shared event-level "
            "account equity or minute-level cross-side tier overlap which "
            "directional summaries cannot reconstruct: "
            f"{unsupported}"
        )


def _validate_gpu_coin_overrides(
    config: dict,
    *,
    strategy_kind: str,
    enabled_sides,
    coin_count: int,
) -> None:
    """Accept modeled leaves plus explicit CPU-compatible backtest no-ops.

    ``live.leverage`` and forced modes other than ``normal`` affect live
    exchange operation, but the exact Rust backtester does not consume them.
    Keep composed live configs usable while making that no-op status visible.
    Exact override documents remain part of checkpoint identity downstream.
    """

    overrides = config.get("coin_overrides") or {}
    if not overrides:
        return
    if (
        coin_count < 1
        or strategy_kind not in {"ema_anchor", "trailing_martingale"}
        or not enabled_sides
    ):
        raise ValueError(
            "GPU coin_overrides require a prepared supported strategy with at "
            "least one enabled side"
        )
    enabled_sides = set(enabled_sides)
    from optimization.gpu.model import (
        EMA_ANCHOR_PARAM_KEYS,
        TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    )

    if strategy_kind == "ema_anchor":
        strategy_paths = {
            (key,)
            for key in set(EMA_ANCHOR_PARAM_KEYS)
            - {"entry_cooldown_minutes", "total_wallet_exposure_limit"}
        }
    else:
        strategy_paths = {
            path for _key, path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
        }
        strategy_paths.add(TRAILING_MARTINGALE_GATE_MODE_OVERRIDE_PATH)

    def leaves(value, prefix=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, (*prefix, str(key)))
        else:
            yield prefix

    def value_at(value, path):
        for key in path:
            value = value[key]
        return value

    forced_mode_paths = {
        ("live", "forced_mode_long"),
        ("live", "forced_mode_short"),
    }
    allowed = {("live", "leverage")}
    for enabled_side in enabled_sides:
        allowed.update(
            {
                ("live", f"forced_mode_{enabled_side}"),
                ("bot", enabled_side, "risk", "entry_cooldown_minutes"),
                ("bot", enabled_side, "risk", "we_excess_allowance_pct"),
                ("bot", enabled_side, "wallet_exposure_limit"),
            }
        )
        allowed.update(
            ("bot", enabled_side, "unstuck", key)
            for key in (
                "enabled",
                "ema_gating_enabled",
                "close_pct",
                "ema_dist",
                "loss_allowance_pct",
                "threshold",
            )
        )
        if strategy_kind == "trailing_martingale":
            allowed.update(
                {
                    (
                        "bot",
                        enabled_side,
                        "risk",
                        "position_exposure_enforcer_enabled",
                    ),
                    (
                        "bot",
                        enabled_side,
                        "risk",
                        "position_exposure_enforcer_threshold",
                    ),
                }
            )
        allowed.update(
            (
                "bot",
                enabled_side,
                "strategy",
                strategy_kind,
                *path,
            )
            for path in strategy_paths
        )
        allowed.update(
            ("bot", enabled_side, "hsl", *path)
            for _key, path in HSL_COIN_OVERRIDE_PATHS
        )
    unsupported = []
    backtest_inert = []
    hsl_override_paths = []
    for coin, patch in overrides.items():
        if not isinstance(patch, dict):
            unsupported.append(f"coin_overrides.{coin}")
            continue
        for path in leaves(patch):
            rendered = ".".join(("coin_overrides", str(coin), *path))
            inert_forced_mode = (
                path in forced_mode_paths and value_at(patch, path) != "normal"
            )
            if path == ("live", "leverage") or inert_forced_mode:
                backtest_inert.append(rendered)
            if len(path) >= 3 and path[0] == "bot" and path[2] == "hsl":
                hsl_override_paths.append(rendered)
            if path not in allowed and not inert_forced_mode:
                unsupported.append(rendered)
    if hsl_override_paths:
        signal_mode = str(
            config.get("live", {}).get("hsl_signal_mode", "unified")
        ).strip().lower()
        fused_dual_side = len(enabled_sides) == 2
        if coin_count > 1 and (
            signal_mode != "coin"
            or not (len(enabled_sides) == 1 or fused_dual_side)
        ):
            raise ValueError(
                "GPU per-coin HSL overrides require live.hsl_signal_mode=coin "
                "and either one enabled side or a fused dual-side proxy; "
                "unsupported paths: "
                f"{sorted(hsl_override_paths)}"
            )
        for coin, patch in overrides.items():
            if not isinstance(patch, dict):
                continue
            bot_patch = patch.get("bot", {})
            if not isinstance(bot_patch, dict):
                continue
            for side in enabled_sides:
                side_patch = bot_patch.get(side, {})
                if not isinstance(side_patch, dict):
                    continue
                hsl_patch = side_patch.get("hsl", {}) or {}
                if not isinstance(hsl_patch, dict):
                    continue
                validate_hsl_override_patch(
                    config.get("bot", {}).get(side, {}).get("hsl", {}) or {},
                    hsl_patch,
                    field_name=f"coin_overrides.{coin}.bot.{side}.hsl",
                )
    if unsupported:
        supported_risk = (
            "risk.entry_cooldown_minutes, risk.we_excess_allowance_pct"
        )
        if strategy_kind == "trailing_martingale":
            supported_risk += (
                ", risk.position_exposure_enforcer_enabled, "
                "risk.position_exposure_enforcer_threshold"
            )
        raise ValueError(
            "GPU coin_overrides do not model these paths yet: "
            f"{sorted(unsupported)}; supported leaves are enabled-side "
            f"forced_mode_<side>, {strategy_kind} parameters, {supported_risk}, "
            "unstuck parameters, HSL parameters in coin signal mode, and "
            "wallet_exposure_limit"
        )
    if backtest_inert:
        logging.warning(
            "GPU coin_overrides contain CPU-compatible live-only values with "
            "no backtest effect; exact Rust and the MPS proxy both leave these "
            "values inert, and checkpoint identity still records them: %s",
            sorted(backtest_inert),
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

    contexts = getattr(suite_evaluator, "contexts", None)
    get_data = getattr(suite_evaluator, "get_prepared_context_data", None)
    build_config = getattr(suite_evaluator, "build_scenario_candidate_config", None)
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("GPU suite mode requires prepared optimizer scenario contexts")
    if not callable(get_data) or not callable(build_config):
        raise TypeError("GPU suite mode requires the canonical SuiteEvaluator")

    prepared = []
    for ctx in contexts:
        overrides = getattr(ctx, "overrides", {}) or {}
        _validate_gpu_suite_override_paths(
            proxy_config,
            label=str(ctx.label),
            overrides=overrides,
        )
        exchanges = list(ctx.exchanges)
        if not exchanges:
            raise ValueError(
                f"GPU suite scenario {ctx.label!r} has no prepared datasets"
            )
        scenario_config = build_config(proxy_config, ctx)
        effective_coin_sources = (
            getattr(ctx, "config", {}).get("backtest", {}).get("coin_sources")
            or {}
        )
        for exchange in exchanges:
            scenario_mss = ctx.msss[exchange]
            effective_coins = [
                str(coin) for coin in scenario_mss if coin != "__meta__"
            ]
            effective_coin_set = set(effective_coins)
            if exchange != "combined":
                prepared_exchange = to_standard_exchange_name(exchange)
                conflicting_sources = {
                    str(coin): str(source)
                    for coin, source in effective_coin_sources.items()
                    if str(coin) in effective_coin_set
                    and to_standard_exchange_name(str(source)) != prepared_exchange
                }
                if conflicting_sources:
                    raise ValueError(
                        f"GPU suite scenario {ctx.label!r} assigns coin_sources "
                        f"outside prepared dataset {prepared_exchange!r}: "
                        f"{conflicting_sources}"
                    )
            hlcvs, btc, coin_indices = get_data(ctx, exchange)
            values = np.asarray(hlcvs)
            if coin_indices is not None:
                values = np.take(values, list(coin_indices), axis=1)
            values = np.ascontiguousarray(values)
            coin_count = int(values.shape[1])
            _validate_scope_config(
                scenario_config,
                exchanges=[exchange],
                coin_count=coin_count,
                allow_suite=True,
            )
            if len(effective_coins) != coin_count:
                raise ValueError(
                    f"GPU suite scenario {ctx.label!r} prepared coin identity "
                    f"mismatch on {exchange!r}: hlcvs={coin_count}, "
                    f"market_settings={effective_coins}"
                )
            prepared.append(
                {
                    "ctx": ctx,
                    "config": deepcopy(scenario_config),
                    "overrides": deepcopy(overrides),
                    "exchange": exchange,
                    "coin_count": coin_count,
                    "coins": effective_coins,
                    "hlcvs": values,
                    "mss": scenario_mss,
                    "btc": btc,
                    "timestamps": ctx.timestamps.get(exchange),
                }
            )
    return prepared


def _gpu_suite_search_context(
    suite_inputs: list[dict],
) -> tuple[int, int, tuple[str, ...] | None]:
    """Return the common candidate-space coin range and multicoin side topology."""

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
    if len(strategy_kinds) != 1 or not strategy_kinds <= {
        "ema_anchor",
        "trailing_martingale",
    }:
        raise ValueError(
            "GPU multicoin suites require one supported strategy kind in every "
            f"scenario; got {sorted(strategy_kinds)}"
        )
    strategy_kind = next(iter(strategy_kinds))
    sides_by_label = {}
    for item in suite_inputs:
        sides = tuple(
            side
            for side in ("long", "short")
            if gpu_side_enabled(item["config"], side)
        )
        sides_by_label[item["ctx"].label] = sides
        if len(sides) not in (1, 2):
            raise ValueError(
                "GPU multicoin suites require one or two enabled sides in every "
                f"scenario; {item['ctx'].label!r} has {list(sides)}"
            )
    common_topologies = set(sides_by_label.values())
    if len(common_topologies) != 1:
        details = ", ".join(
            f"{label}={list(sides)}" for label, sides in sides_by_label.items()
        )
        raise ValueError(
            "GPU multicoin suites require the same enabled-side topology in every "
            "scenario; "
            f"got {details}"
        )
    return min_coin_count, max_coin_count, common_topologies.pop()


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
    selected = np.zeros(len(values), dtype=bool)
    selected[chosen[0]] = True
    distance = np.linalg.norm(normalized - normalized[chosen[0]], axis=1)
    for _ in range(count - 1):
        available = np.flatnonzero(~selected)
        available_distances = np.where(
            np.isfinite(distance[available]), distance[available], -np.inf
        )
        index = int(available[int(np.argmax(available_distances))])
        chosen.append(index)
        selected[index] = True
        distance = np.minimum(
            distance, np.linalg.norm(normalized - normalized[index], axis=1)
        )
    return chosen


def _successive_halving_survivor_indices(
    objectives: np.ndarray,
    violations: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    """Select a deterministic constraint-aware, Pareto-diverse rung subset."""

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    objectives = np.asarray(objectives, dtype=np.float64)
    violations = np.asarray(violations, dtype=np.float64)
    if objectives.ndim != 2 or len(objectives) != len(violations):
        raise ValueError("successive-halving objectives and violations must align")
    count = min(max(0, int(count)), len(objectives))
    if count == 0:
        return np.empty(0, dtype=np.int64)

    feasible = np.flatnonzero(np.isfinite(violations) & (violations <= 0.0))
    feasible_ids = set(map(int, feasible))
    infeasible = np.asarray(
        sorted(
            (
                int(index)
                for index in range(len(objectives))
                if int(index) not in feasible_ids
            ),
            key=lambda index: (
                float(violations[index])
                if np.isfinite(violations[index])
                else float("inf"),
                index,
            ),
        ),
        dtype=np.int64,
    )
    selected: list[int] = []
    if len(feasible):
        for front_local in NonDominatedSorting().do(objectives[feasible]):
            front = feasible[np.asarray(front_local, dtype=np.int64)]
            remaining = count - len(selected)
            if remaining <= 0:
                break
            if len(front) <= remaining:
                selected.extend(map(int, front))
                continue
            diverse = _normalized_farthest_indices(objectives[front], remaining)
            selected.extend(int(front[index]) for index in diverse)
            break
    if len(selected) < count:
        selected.extend(map(int, infeasible[: count - len(selected)]))
    return np.asarray(selected, dtype=np.int64)


def _evaluate_successive_halving(
    candidates: list[dict],
    *,
    policy: dict,
    evaluate_proxy,
    proxy_fitness,
    interrupt_check: InterruptCheck,
    stage_callback=None,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Evaluate progressively longer history windows and return full-rung eligibility."""

    active = np.arange(len(candidates), dtype=np.int64)
    metric_rows: list[dict | None] = [None] * len(candidates)
    objectives = None
    violations = np.full(len(candidates), np.inf, dtype=np.float64)
    trace: list[dict] = []
    fractions = list(policy["history_fractions"])
    for rung, fraction in enumerate(fractions):
        interrupt_check()
        stage_candidates = [candidates[int(index)] for index in active]
        stage_metrics = evaluate_proxy(
            stage_candidates,
            history_fraction=float(fraction),
        )
        stage_objectives, stage_violations = proxy_fitness(stage_metrics)
        if stage_callback is not None:
            stage_callback(rung + 1, float(fraction), len(active))
        if objectives is None:
            objectives = np.full(
                (len(candidates), stage_objectives.shape[1]),
                np.nan,
                dtype=np.float64,
            )
        objectives[active] = stage_objectives
        violations[active] = stage_violations
        for local_index, source_index in enumerate(active):
            metric_rows[int(source_index)] = dict(stage_metrics[local_index])

        final_rung = rung == len(fractions) - 1
        survivor_count = len(active)
        if not final_rung:
            survivor_count = min(
                len(active),
                max(
                    int(policy["min_survivors"]),
                    int(math.ceil(len(active) * float(policy["survival_fraction"]))),
                ),
            )
        trace.append(
            {
                "rung": rung + 1,
                "history_fraction": float(fraction),
                "candidate_count": int(len(active)),
                "survivor_count": int(survivor_count),
            }
        )
        if final_rung:
            break
        local_survivors = _successive_halving_survivor_indices(
            stage_objectives,
            stage_violations,
            count=survivor_count,
        )
        active = active[local_survivors]

    if objectives is None or any(row is None for row in metric_rows):
        raise RuntimeError("successive-halving proxy evaluation produced incomplete rows")
    full_rung_indices = active.copy()
    rejected = np.ones(len(candidates), dtype=bool)
    rejected[full_rung_indices] = False
    # Partial-history rows remain useful as weak proposal evidence, but must
    # never outrank or masquerade as full-history proxy evidence.
    violations[rejected] = np.inf
    return (
        [dict(row) for row in metric_rows if row is not None],
        objectives,
        violations,
        full_rung_indices,
        trace,
    )


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

    # Return a complete preference order. The caller may skip duplicate probes,
    # seek novel front members, or deliberately revalidate an already-exact
    # current-front member without changing its truthful class.
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


def _effective_seed_bootstrap_mode(policy: dict, seed_count: int) -> str:
    """Resolve auto without silently weakening an explicit exact request."""

    mode = str(policy["mode"])
    if int(seed_count) <= 0:
        return "none"
    if mode != "auto":
        return mode
    return "exact" if int(seed_count) <= int(policy["max_exact"]) else "screened"


def _deduplicate_canonical_seed_vectors(
    vectors,
    *,
    hash_vector,
) -> tuple[list, int]:
    """Drop seeds that become identical after GPU runtime overrides."""

    deduplicated = []
    seen = set()
    for vector in vectors:
        digest = hash_vector(vector)
        if digest in seen:
            continue
        seen.add(digest)
        deduplicated.append(vector)
    return deduplicated, len(vectors) - len(deduplicated)


def _select_seed_bootstrap_indices(
    objectives: np.ndarray,
    scores: np.ndarray,
    violations: np.ndarray,
    *,
    total: int,
) -> list[tuple[int, bool, bool]]:
    """Select objective extremes, a diverse proxy front, and broad probes."""

    objectives = np.asarray(objectives, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    violations = np.asarray(violations, dtype=np.float64)
    total = min(max(0, int(total)), len(objectives))
    if total == 0:
        return []
    if objectives.ndim != 2 or len(scores) != len(objectives) or len(
        violations
    ) != len(objectives):
        raise ValueError(
            "GPU seed-bootstrap objectives, scores and violations must align"
        )

    requested_probes = min(
        total - 1,
        max(1, int(round(total * GPU_SEED_BOOTSTRAP_PROBE_FRACTION))),
    )
    preference = _select_validation_indices(
        objectives,
        scores,
        violations,
        total=total,
        probes=requested_probes,
    )
    classification = {
        int(index): (bool(is_probe), bool(is_front))
        for index, is_probe, is_front in preference
    }
    feasible = np.flatnonzero(np.isfinite(violations) & (violations <= 0.0))
    primary = feasible if len(feasible) else np.arange(len(objectives), dtype=np.int64)
    constraint_priority = []
    if len(feasible) == 0:
        # Proxy constraints can drift from exact Rust constraints, so retain
        # objective diversity, but explicitly reserve part of the exact budget
        # for the seeds closest to feasibility.  Otherwise an all-infeasible
        # objective front can crowd out the most promising constraint repairs.
        reserve = min(total, max(1, requested_probes))
        constraint_priority = sorted(
            map(int, primary),
            key=lambda index: (
                float(violations[index])
                if np.isfinite(violations[index])
                else float("inf"),
                float(scores[index]),
            ),
        )[:reserve]
    extremes = []
    for column in range(objectives.shape[1]):
        values = objectives[primary, column]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        eligible = primary[finite]
        finite_values = values[finite]
        minimum = float(np.min(finite_values))
        minima = eligible[
            np.isclose(finite_values, minimum, rtol=0.0, atol=1.0e-12)
        ]
        front_minima = np.asarray(
            [
                index
                for index in minima
                if int(index) in classification
                and classification[int(index)][1]
            ],
            dtype=np.int64,
        )
        choices = front_minima if len(front_minima) else minima
        index = int(min(choices, key=lambda item: float(scores[int(item)])))
        if index not in extremes:
            extremes.append(index)

    selected: list[tuple[int, bool, bool]] = []
    selected_ids = set()

    def append(index: int) -> None:
        if len(selected) >= total or index in selected_ids:
            return
        is_probe, is_front = classification[int(index)]
        selected.append((int(index), is_probe, is_front))
        selected_ids.add(int(index))

    front_target = max(1, total - requested_probes)
    for index in constraint_priority:
        append(index)
    for index in extremes[:front_target]:
        append(index)
    for index, _is_probe, is_front in preference:
        if is_front and sum(front for _idx, _probe, front in selected) < front_target:
            append(index)
    for index, is_probe, _is_front in preference:
        if is_probe:
            append(index)
    for index, _is_probe, _is_front in preference:
        append(index)
    return selected


def _select_seed_population_indices(
    objectives: np.ndarray,
    violations: np.ndarray,
    *,
    count: int,
) -> list[int]:
    """Reduce a seed pool with constraint-aware Pareto diversity."""

    return list(
        map(
            int,
            _successive_halving_survivor_indices(
                objectives,
                violations,
                count=count,
            ),
        )
    )


def _validate_seed_bootstrap_plan(
    starting_vectors,
    selections,
    population_indices,
    contract,
    *,
    hash_vector,
    proxy_metrics=None,
    proxy_objectives=None,
    proxy_violations=None,
    screen_complete: bool = True,
) -> None:
    """Fail closed when an incomplete checkpoint seed plan is inconsistent."""

    if not isinstance(contract, dict):
        raise RuntimeError("GPU checkpoint seed-bootstrap plan has no contract")
    seed_count = len(starting_vectors)
    hashes = [hash_vector(vector) for vector in starting_vectors]
    digest = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    if (
        int(contract.get("seed_count", -1)) != seed_count
        or contract.get("seed_pool_sha256") != digest
    ):
        raise RuntimeError(
            "GPU checkpoint seed-bootstrap plan does not match its seed contract"
        )
    mode = str(contract.get("effective_mode"))
    if mode not in {"exact", "screened"}:
        raise RuntimeError(
            "GPU checkpoint has an incomplete seed-bootstrap plan for unsupported "
            f"mode {mode!r}"
        )
    if len(hashes) != len(set(hashes)):
        raise RuntimeError(
            "GPU checkpoint seed-bootstrap plan contains canonical duplicate seeds"
        )
    expected_selected = int(contract.get("selected_exact_count", -1))
    if not screen_complete:
        if (
            mode != "screened"
            or expected_selected != min(
                seed_count, int(contract.get("max_exact", -1))
            )
            or selections
            or population_indices
            or proxy_metrics is not None
            or proxy_objectives is not None
            or proxy_violations is not None
        ):
            raise RuntimeError(
                "GPU checkpoint pending seed screen has invalid partial evidence"
            )
        return
    if expected_selected != len(selections):
        raise RuntimeError(
            "GPU checkpoint seed-bootstrap plan does not match its seed contract"
        )
    selected_ids = [int(index) for index, _probe, _front in selections]
    if (
        len(selected_ids) != len(set(selected_ids))
        or any(index < 0 or index >= seed_count for index in selected_ids)
        or len(population_indices) != len(set(map(int, population_indices)))
        or any(
            int(index) < 0 or int(index) >= seed_count
            for index in population_indices
        )
    ):
        raise RuntimeError("GPU checkpoint seed-bootstrap plan has invalid indices")
    if mode == "exact":
        if selected_ids != list(range(seed_count)) or not bool(
            contract.get("all_seeds_exact")
        ):
            raise RuntimeError(
                "GPU checkpoint exact seed-bootstrap plan is incomplete"
            )
        return
    objectives = np.asarray(proxy_objectives, dtype=np.float64)
    violations = np.asarray(proxy_violations, dtype=np.float64)
    if (
        not isinstance(proxy_metrics, dict)
        or set(map(int, proxy_metrics)) != set(selected_ids)
        or objectives.ndim != 2
        or len(objectives) != seed_count
        or violations.shape != (seed_count,)
        or bool(contract.get("all_seeds_exact"))
        or any(bool(probe) == bool(front) for _index, probe, front in selections)
    ):
        raise RuntimeError(
            "GPU checkpoint screened seed-bootstrap plan has invalid proxy evidence"
        )


class _ProxyFrontValidationPending(RuntimeError):
    """The current proxy front has exact work in flight but no ready evidence."""


def _select_exact_validations(
    selections,
    *,
    total: int,
    candidate_for_index,
    digest_for_candidate,
    completed_hashes,
    submitted_hashes,
):
    """Choose truthful exact evidence, revalidating a covered front if needed.

    Off-front candidates are never relabeled as front evidence. When duplicate
    filtering leaves no novel front member, a previously completed current-front
    candidate is deliberately re-run so the rolling front gate remains active.
    A current-front candidate that is still in flight must complete first.
    """

    novel = []
    completed_front = []
    seen = set()
    novel_probe_count = 0
    novel_front_count = 0
    pending_front = False
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
        if digest in seen:
            continue
        seen.add(digest)
        item = (index, bool(is_probe), bool(is_front), candidate, digest)
        if digest in submitted_hashes:
            pending_front = pending_front or bool(is_front)
            continue
        if digest in completed_hashes:
            if is_front:
                completed_front.append(item)
            continue
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
    if len(front_items) < target_front_count:
        front_items.extend(completed_front[: target_front_count - len(front_items)])
    if not front_items:
        if pending_front:
            raise _ProxyFrontValidationPending(
                "GPU proxy-front exact validation is still in flight"
            )
        raise RuntimeError(
            "GPU validation cannot provide truthful proxy-front safety evidence; "
            "the selector returned no novel or previously exact front candidate"
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
            "using truthful true-front candidates (including exact revalidation "
            "when necessary); broad-probe gates use only accumulated off-front evidence",
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


def _gpu_suite_checkpoint_contract(
    config: dict, suite_inputs=None, *, pinned_hsl_bounds=None
) -> dict:
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
    contract["max_realized_loss_pct"] = float(
        config.get("live", {}).get("max_realized_loss_pct", 1.0)
    )
    contract["hedge_mode"] = bool(
        config.get("live", {}).get("hedge_mode", True)
    )
    contract["pnls_max_lookback_days"] = (
        _gpu_pnls_max_lookback_days_checkpoint_value(config)
    )
    contract["unstuck"] = _gpu_unstuck_checkpoint_contract(config)
    contract["hsl"] = _gpu_hsl_checkpoint_contract(config)
    contract["pinned_hsl_bounds"] = deepcopy(pinned_hsl_bounds or {})
    if suite_inputs is not None:
        prepared_scenarios = []
        for item in suite_inputs:
            timestamps = np.asarray(item["timestamps"]).reshape(-1)
            if len(timestamps) != len(item["hlcvs"]):
                raise ValueError(
                    f"GPU suite scenario {item['ctx'].label!r} timestamp identity "
                    f"mismatch: timestamps={len(timestamps)}, hlcvs={len(item['hlcvs'])}"
                )
            source_contract = []
            scenario_mss = item.get("mss") or {}
            for coin in item["coins"]:
                metadata = scenario_mss.get(coin) or {}
                market_exchange = str(
                    metadata.get("exchange") or item["exchange"]
                )
                source_contract.append(
                    {
                        "coin": coin,
                        "ohlcv_exchange": str(
                            metadata.get("ohlcv_source") or market_exchange
                        ),
                        "market_settings_exchange": market_exchange,
                    }
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
                    "hedge_mode": bool(
                        item["config"].get("live", {}).get("hedge_mode", True)
                    ),
                    "max_realized_loss_pct": float(
                        item["config"]
                        .get("live", {})
                        .get("max_realized_loss_pct", 1.0)
                    ),
                    "pnls_max_lookback_days": (
                        _gpu_pnls_max_lookback_days_checkpoint_value(
                            item["config"]
                        )
                    ),
                    "unstuck": _gpu_unstuck_checkpoint_contract(item["config"]),
                    "hsl": _gpu_hsl_checkpoint_contract(item["config"]),
                    "pinned_hsl_bounds": deepcopy(
                        item.get("pinned_hsl_bounds", {})
                    ),
                    "scenario_fixed_bound_values": deepcopy(
                        item.get("fixed_bound_values", {})
                    ),
                    "scenario_parameter_overrides": deepcopy(
                        item.get("parameter_overrides", {})
                    ),
                    "proxy_execution": deepcopy(
                        item.get("proxy_checkpoint_contract", {})
                    ),
                    "candle_count": int(len(item["hlcvs"])),
                    "first_timestamp": (
                        int(timestamps[0]) if len(timestamps) else None
                    ),
                    "last_timestamp": (
                        int(timestamps[-1]) if len(timestamps) else None
                    ),
                    "coin_sources": source_contract,
                    "coin_overrides": deepcopy(
                        item.get("coin_override_contract")
                        or item["config"].get("coin_overrides", {})
                    ),
                }
            )
        contract["prepared_scenarios"] = prepared_scenarios
    return contract


def _gpu_runtime_checkpoint_contract(
    config: dict, proxy, *, pinned_hsl_bounds=None
) -> dict:
    return {
        "hedge_mode": bool(
            config.get("live", {}).get("hedge_mode", True)
        ),
        "max_realized_loss_pct": float(
            config.get("live", {}).get("max_realized_loss_pct", 1.0)
        ),
        "pnls_max_lookback_days": (
            _gpu_pnls_max_lookback_days_checkpoint_value(config)
        ),
        "coin_override_contract": deepcopy(
            getattr(proxy, "coin_override_contract", None)
        ),
        "unstuck": _gpu_unstuck_checkpoint_contract(config),
        "hsl": _gpu_hsl_checkpoint_contract(config),
        "pinned_hsl_bounds": deepcopy(pinned_hsl_bounds or {}),
        "proxy_execution": deepcopy(
            getattr(proxy, "checkpoint_contract", {})
        ),
    }


def _gpu_pnls_max_lookback_days_checkpoint_value(config: dict) -> float:
    return parse_pnls_max_lookback_days(
        config.get("live", {}).get("pnls_max_lookback_days", 30.0),
        field_name="live.pnls_max_lookback_days",
    ).to_backtest_days_value()


def _gpu_unstuck_checkpoint_contract(config: dict) -> dict:
    contract = {}
    for side in ("long", "short"):
        unstuck = config.get("bot", {}).get(side, {}).get("unstuck", {})
        contract[side] = {
            "enabled": bool(unstuck.get("enabled", False)),
            "ema_gating_enabled": bool(
                unstuck.get("ema_gating_enabled", True)
            ),
            "close_pct": float(unstuck.get("close_pct", 0.0)),
            "ema_dist": float(unstuck.get("ema_dist", 0.0)),
            "loss_allowance_pct": float(
                unstuck.get("loss_allowance_pct", 0.0)
            ),
            "threshold": float(unstuck.get("threshold", 0.0)),
        }
    return contract


def _gpu_hsl_checkpoint_contract(config: dict) -> dict:
    return {
        "signal_mode": str(
            config.get("live", {}).get("hsl_signal_mode", "unified")
        )
        .strip()
        .lower(),
        "dynamic_wel_by_tradability": bool(
            config.get("backtest", {}).get("dynamic_wel_by_tradability", True)
        ),
        "sides": {
            side: {
                "config": deepcopy(
                    config.get("bot", {}).get(side, {}).get("hsl", {})
                ),
                "n_positions": deepcopy(
                    config.get("bot", {})
                    .get(side, {})
                    .get("risk", {})
                    .get("n_positions")
                ),
            }
            for side in ("long", "short")
        },
    }


def _gpu_pinned_hsl_bound_contract(bound_by_key) -> dict[str, float]:
    return {
        key: float(bound.low)
        for key, bound in sorted(bound_by_key.items())
        if "_hsl_" in key
        and math.isclose(
            float(bound.low), float(bound.high), rel_tol=0.0, abs_tol=1.0e-12
        )
    }


def _gpu_hsl_side_enabled(config: dict, side: str) -> bool:
    globally_enabled = bool(
        config.get("bot", {})
        .get(side, {})
        .get("hsl", {})
        .get("enabled", False)
    )
    if globally_enabled:
        return True
    for patch in (config.get("coin_overrides") or {}).values():
        if not isinstance(patch, dict):
            continue
        hsl_patch = (
            patch.get("bot", {}).get(side, {}).get("hsl", {}) or {}
        )
        if isinstance(hsl_patch, dict) and bool(hsl_patch.get("enabled", False)):
            return True
    return False


def _validate_hsl_bound_contracts(bound_by_key, config: dict) -> None:
    float32_below_one = float(
        np.nextafter(np.float32(1.0), np.float32(0.0))
    )
    for side in ("long", "short"):
        globally_enabled = bool(
            config.get("bot", {})
            .get(side, {})
            .get("hsl", {})
            .get("enabled", False)
        )
        enabled = _gpu_hsl_side_enabled(config, side)
        enabled_bound = bound_by_key.get(f"{side}_hsl_enabled")
        if enabled_bound is not None:
            expected = float(globally_enabled)
            endpoints = (float(enabled_bound.low), float(enabled_bound.high))
            if any(
                not math.isclose(
                    value, expected, rel_tol=0.0, abs_tol=1.0e-12
                )
                for value in endpoints
            ):
                raise ValueError(
                    "GPU HSL requires pinned optimizer enablement to match the "
                    f"source bot.{side}.hsl.enabled={enabled}; got bounds {endpoints}"
                )
        if not enabled:
            continue
        red_bound = bound_by_key.get(f"{side}_hsl_red_threshold")
        if red_bound is not None and float(red_bound.low) <= 0.0:
            raise ValueError(
                f"GPU HSL {side}_hsl_red_threshold bounds must remain greater "
                f"than zero, got {(float(red_bound.low), float(red_bound.high))}"
            )
        cooldown_bound = bound_by_key.get(
            f"{side}_hsl_cooldown_minutes_after_red"
        )
        if cooldown_bound is not None and float(cooldown_bound.low) < 0.0:
            raise ValueError(
                "GPU HSL "
                f"{side}_hsl_cooldown_minutes_after_red bounds must remain "
                "non-negative, got "
                f"{(float(cooldown_bound.low), float(cooldown_bound.high))}"
            )
        for suffix in (
            "hsl_red_threshold",
            "hsl_no_restart_drawdown_threshold",
        ):
            bound = bound_by_key.get(f"{side}_{suffix}")
            if bound is None:
                continue
            low, high = float(bound.low), float(bound.high)
            if low < 1.0 and high > float32_below_one:
                raise ValueError(
                    f"GPU HSL {side}_{suffix} bounds include values which "
                    "float32 cannot distinguish from 1.0; require high <= "
                    f"{float32_below_one} or pin exactly at 1.0, got {(low, high)}"
                )


def _gpu_hsl_search_sides(
    proxy_config: dict, suite_inputs, overrides: set[str] | None = None
) -> set[str]:
    configs = (
        [item["config"] for item in suite_inputs]
        if suite_inputs
        else [proxy_config]
    )
    target_sides = {
        side
        for side in ("long", "short")
        if any(
            gpu_side_enabled(item, side)
            and _gpu_hsl_side_enabled(item, side)
            for item in configs
        )
    }
    return _gpu_candidate_source_sides(target_sides, overrides or set())


def _gpu_hsl_parameter_active(
    parameter: str, hsl_search_sides: set[str]
) -> bool:
    for side in ("long", "short"):
        if parameter.startswith(f"{side}_hsl_"):
            return side in hsl_search_sides
    return True


def _gpu_unstuck_search_sides(
    proxy_config: dict, suite_inputs, overrides: set[str] | None = None
) -> set[str]:
    """Return target and mirrored source sides whose unstuck genes affect a scenario."""

    configs = (
        [item["config"] for item in suite_inputs]
        if suite_inputs
        else [proxy_config]
    )
    target_sides = {
        side
        for side in ("long", "short")
        if any(
            gpu_side_enabled(item, side)
            and (
                bool(
                    item.get("bot", {})
                    .get(side, {})
                    .get("unstuck", {})
                    .get("enabled", False)
                )
                or any(
                    bool(
                        patch.get("bot", {})
                        .get(side, {})
                        .get("unstuck", {})
                        .get("enabled", False)
                    )
                    for patch in item.get("coin_overrides", {}).values()
                    if isinstance(patch, dict)
                )
            )
            for item in configs
        )
    }
    return _gpu_candidate_source_sides(target_sides, overrides or set())


def _gpu_unstuck_parameter_active(
    parameter: str, unstuck_search_sides: set[str]
) -> bool:
    for side in ("long", "short"):
        if parameter.startswith(f"{side}_unstuck_"):
            return side in unstuck_search_sides
    return True


def _checkpoint_signature(
    active,
    scoring,
    *,
    anchor_plan=None,
    suite_contract=None,
    runtime_contract=None,
    search_contract=None,
) -> str:
    payload = {
        "active": [
            [name, int(index), float(bound.low), float(bound.high), bound.step]
            for name, index, bound in active
        ],
        "scoring": scoring,
        "version": 3,
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
    if search_contract is not None:
        payload["search_contract"] = search_contract
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _gpu_search_checkpoint_contract(
    *,
    key_paths,
    bounds,
    base_vector,
    fixed_bound_values,
    fixed_parameter_overrides,
    optimizer_overrides,
    sig_digits,
    algorithm_contract,
    proxy_evaluation_policy=None,
    seed_bootstrap_contract=None,
) -> dict:
    """Fingerprint fixed and dormant search inputs omitted from active genes."""

    if not (len(key_paths) == len(bounds) == len(base_vector)):
        raise ValueError("GPU checkpoint search contract shape mismatch")
    contract = {
        "version": 1,
        "sig_digits": int(sig_digits),
        "dimensions": [
            {
                "key": str(bound_key),
                "path": [str(part) for part in path],
                "low": float(bound.low),
                "high": float(bound.high),
                "step": None if bound.step is None else float(bound.step),
                "base": float(base_vector[index]),
            }
            for index, ((bound_key, path), bound) in enumerate(
                zip(key_paths, bounds)
            )
        ],
        "fixed_bound_values": {
            str(key): float(value)
            for key, value in sorted((fixed_bound_values or {}).items())
        },
        "fixed_parameter_overrides": {
            str(key): float(value)
            for key, value in sorted((fixed_parameter_overrides or {}).items())
        },
        "optimizer_overrides": sorted(str(value) for value in optimizer_overrides),
        "algorithm": deepcopy(algorithm_contract),
    }
    if proxy_evaluation_policy is not None:
        contract["version"] = 2
        contract["proxy_evaluation"] = deepcopy(proxy_evaluation_policy)
    if seed_bootstrap_contract is not None:
        contract["version"] = 3
        contract["seed_bootstrap"] = deepcopy(seed_bootstrap_contract)
    return contract


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


def _validate_pinned_scope_bounds(
    bound_by_key,
    base_by_key,
    enabled_sides=None,
    *,
    coin_count: int = 1,
    strategy_kind: str | None = None,
) -> None:
    enabled_sides = set(enabled_sides or ("long", "short"))
    pinned = {}
    if strategy_kind != "trailing_martingale":
        pinned.update(
            {
                f"{side}_risk_position_exposure_enforcer_enabled": 0.0
                for side in ("long", "short")
            }
        )
    for key, expected in pinned.items():
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


def _validate_tm_market_mode_bounds(
    bound_by_key, base_by_key, enabled_sides, config, *, coin_count: int = 1
) -> None:
    if (
        str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
        != "trailing_martingale"
        or not bool(config.get("live", {}).get("market_orders_allowed"))
    ):
        return
    unsupported = []
    for side in sorted(set(enabled_sides or ())):
        for phase in ("entry", "close"):
            key = f"{side}_{phase}_retracement_base_pct"
            bound = bound_by_key.get(key)
            low, high = (
                (float(bound.low), float(bound.high))
                if bound is not None
                else (float(base_by_key.get(key, 0.0)),) * 2
            )
            if any(
                not math.isfinite(value)
                or abs(value) > float(np.finfo(np.float32).max)
                for value in (low, high)
            ):
                unsupported.append(f"{key} bounds=({low}, {high})")
    if unsupported:
        raise ValueError(
            "GPU Trailing Martingale ordinary market execution currently "
            "requires finite float32-representable entry and close retracement "
            "bounds. Unsupported bounds remain fail closed: "
            + ", ".join(unsupported)
        )


def _validate_tm_market_template_bounds(
    bound_by_key,
    base_by_key,
    enabled_sides,
    config,
    suite_inputs,
    *,
    coin_count: int = 1,
) -> None:
    """Validate a directly executed config, not a suite's override template."""

    if suite_inputs:
        return
    _validate_tm_market_mode_bounds(
        bound_by_key,
        base_by_key,
        enabled_sides,
        config,
        coin_count=coin_count,
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


def _recover_durable_seed_bootstrap(
    entries,
    *,
    start_index: int,
    stop_index: int,
    vector_from_entry,
    hash_vector,
) -> tuple[
    dict[str, dict[str, Any]],
    set[str],
    list[tuple[float, float, bool, bool, bool]],
]:
    """Recover exact seed payloads flushed after a stale bootstrap checkpoint."""

    payloads: dict[str, dict[str, Any]] = {}
    recovered: set[str] = set()
    drift_pairs: list[tuple[float, float, bool, bool, bool]] = []
    consumed = 0
    for index, entry in enumerate(entries):
        if index < start_index:
            continue
        if index >= stop_index:
            break
        metrics = entry.get("metrics") or {}
        metadata = metrics.get("gpu_seed_bootstrap")
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise RuntimeError(
                "GPU resume cannot recover seed-bootstrap evidence from "
                f"durable result {index}"
            )
        try:
            mode = str(metadata["mode"])
            source_index = int(metadata["source_index"])
            objectives = [float(value) for value in metadata["exact_objectives"]]
            violation = float(metadata["exact_violation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "GPU resume found invalid seed-bootstrap evidence in "
                f"durable result {index}"
            ) from exc
        if (
            mode not in {"exact", "screened"}
            or source_index < 0
            or not objectives
            or not all(np.isfinite(objectives))
            or not np.isfinite(violation)
        ):
            raise RuntimeError(
                "GPU resume found non-finite seed-bootstrap evidence in "
                f"durable result {index}"
            )
        digest = hash_vector(vector_from_entry(entry))
        if digest in payloads:
            raise RuntimeError(
                "GPU resume found duplicate durable seed-bootstrap evidence for "
                f"result {index}"
            )
        payloads[digest] = {
            "source_index": source_index,
            "F": objectives,
            "G": [violation],
        }
        recovered.add(digest)
        validation = metrics.get("gpu_validation")
        if mode == "screened" and validation is None:
            raise RuntimeError(
                "GPU resume found screened seed-bootstrap evidence without "
                f"proxy/exact metadata in durable result {index}"
            )
        if validation is not None:
            if (
                not isinstance(validation, dict)
                or validation.get("schema_version") != 2
                or validation.get("phase") != "seed_bootstrap"
            ):
                raise RuntimeError(
                    "GPU resume found invalid proxy/exact seed-bootstrap evidence "
                    f"in durable result {index}"
                )
            try:
                proxy_score = float(validation["proxy_score"])
                exact_score = float(validation["exact_score"])
                probe = validation["probe"]
                proxy_front = validation["proxy_front"]
                mismatch = validation["constraint_classification_mismatch"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "GPU resume found invalid proxy/exact seed-bootstrap evidence "
                    f"in durable result {index}"
                ) from exc
            if (
                not np.isfinite(proxy_score)
                or not np.isfinite(exact_score)
                or not isinstance(probe, bool)
                or not isinstance(proxy_front, bool)
                or probe == proxy_front
                or not isinstance(mismatch, bool)
            ):
                raise RuntimeError(
                    "GPU resume found invalid proxy/exact seed-bootstrap evidence "
                    f"in durable result {index}"
                )
            drift_pairs.append(
                (proxy_score, exact_score, probe, mismatch, proxy_front)
            )
        consumed += 1
    expected = max(0, stop_index - start_index)
    if consumed != expected:
        raise RuntimeError(
            "GPU resume could not reconstruct all durable seed-bootstrap results: "
            f"expected {expected}, recovered {consumed}"
        )
    return payloads, recovered, drift_pairs


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
    interrupt_check: InterruptCheck | None = None,
) -> dict[str, Any]:
    del duplicate_counter
    del constraint_fitness_cls
    del record_individual_result
    del run_evolution

    from config.metrics import canonicalize_metric_name
    from config.scoring import extract_objective_specs
    from optimization.gpu.metrics import (
        HARD_STOP_PROXY_METRICS,
        validate_gpu_metric_names,
    )
    from optimization.gpu.service import (
        MpsMulticoinProxy,
        MpsSingleCoinProxy,
        mps_requested_metric_features,
    )
    from optimization.warmup import (
        _finalize_optimizer_vector_config,
        validate_optimizer_effective_configs,
    )

    interrupt_check = interrupt_check or no_interrupt_requested
    interrupt_check()
    reject_configured_exact_only_gpu_metrics(config)
    options = _resolve_options(config)
    validate_optimizer_effective_configs(config)
    checkpoint = None
    if resume:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"GPU checkpoint not found: {checkpoint_path}")
        with open(checkpoint_path, "rb") as file:
            checkpoint = pickle.load(file)

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
        min_coin_count, max_coin_count, suite_multicoin_sides = (
            _gpu_suite_search_context(suite_inputs)
        )
        scenario_count = len({id(item["ctx"]) for item in suite_inputs})
        logging.info(
            "GPU suite prepared %d scenarios across %d datasets | "
            "coins=%d..%d | multicoin_sides=%s",
            scenario_count,
            len(suite_inputs),
            min_coin_count,
            max_coin_count,
            ",".join(suite_multicoin_sides or ()) or "n/a",
        )
    else:
        exchange = _validate_scope(proxy_config, evaluator)
        min_coin_count = max_coin_count = int(
            evaluator.shared_hlcvs_np[exchange].shape[1]
        )
        suite_multicoin_sides = None
    halving_policy = options["successive_halving"]
    if halving_policy["enabled"] and (
        suite_enabled
        or max_coin_count != 1
        or strategy_kind != "trailing_martingale"
    ):
        raise ValueError(
            "optimize.gpu.successive_halving currently requires a non-suite, "
            "single-coin trailing_martingale optimization"
        )
    if max_coin_count > 1:
        multicoin_sides = (
            list(suite_multicoin_sides)
            if suite_multicoin_sides is not None
            else [
                side
                for side in ("long", "short")
                if gpu_side_enabled(proxy_config, side)
            ]
        )
        if len(multicoin_sides) not in (1, 2):
            raise ValueError(
                "GPU multicoin foundation requires one or two enabled sides"
            )
        bound_map = {}
        for multicoin_side in multicoin_sides:
            mapper = (
                _trailing_martingale_multicoin_bound_map
                if strategy_kind == "trailing_martingale"
                else _ema_multicoin_bound_map
            )
            bound_map.update(mapper(multicoin_side, gpu_optimizer_overrides))
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
    for parameter, (_index, bound) in mapped_all.items():
        if math.isclose(
            float(bound.low), float(bound.high), rel_tol=0.0, abs_tol=1.0e-12
        ):
            fixed_parameter_overrides.setdefault(parameter, float(bound.low))
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
        set(suite_multicoin_sides)
        if suite_multicoin_sides is not None
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
        if suite_multicoin_sides is None:
            _validate_seed_side_match(config_enabled_sides, enabled_sides)
        else:
            # CPU suite setup requires symmetric approved coin lists. Effective
            # scenario overrides establish the common side topology and
            # are validated below after shadowing candidate bounds.
            enabled_sides = set(suite_multicoin_sides)
    if not enabled_sides:
        raise ValueError(
            "GPU bounds would disable both sides for exact validation; "
            f"effective seed values: {side_values}"
        )
    candidate_search_sides = _gpu_candidate_search_sides(
        proxy_config, suite_inputs
    )
    candidate_source_sides = _gpu_candidate_source_sides(
        candidate_search_sides, gpu_optimizer_overrides
    )
    unstuck_search_sides = _gpu_unstuck_search_sides(
        proxy_config, suite_inputs, gpu_optimizer_overrides
    )
    hsl_search_sides = _gpu_hsl_search_sides(
        proxy_config, suite_inputs, gpu_optimizer_overrides
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
        and _gpu_unstuck_parameter_active(name, unstuck_search_sides)
        and _gpu_hsl_parameter_active(name, hsl_search_sides)
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
    _validate_pinned_scope_bounds(
        bound_by_key,
        base_by_key,
        enabled_sides,
        coin_count=max_coin_count,
        strategy_kind=strategy_kind,
    )
    _validate_tm_market_template_bounds(
        bound_by_key,
        base_by_key,
        enabled_sides,
        proxy_config,
        suite_inputs,
        coin_count=max_coin_count,
    )
    _validate_hsl_bound_contracts(bound_by_key, proxy_config)

    if suite_multicoin_sides is None:
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
            coin_count=item["coin_count"],
            strategy_kind=strategy_kind,
        )
        _validate_tm_market_mode_bounds(
            scenario_bound_by_key,
            scenario_base_by_key,
            scenario_enabled_sides,
            item["config"],
            coin_count=item["coin_count"],
        )
        _validate_directional_search_space(
            scenario_bound_by_key,
            scenario_base_by_key,
            item["config"].get("live", {}).get("approved_coins", {}),
            scenario_enabled_sides,
            coin_count=item["coin_count"],
        )
        _validate_hsl_bound_contracts(scenario_bound_by_key, item["config"])
        item["parameter_overrides"] = parameter_overrides
        item["fixed_bound_values"] = fixed_scenario_bounds
        item["pinned_hsl_bounds"] = _gpu_pinned_hsl_bound_contract(
            scenario_bound_by_key
        )

    for bound_key, bound in bound_by_key.items():
        if bound_key == ANCHOR_GENE_KEY or bound.high <= bound.low:
            continue
        side = bound_key.split("_", 1)[0]
        if side in {"long", "short"} and side not in candidate_search_sides:
            continue
        if max_coin_count == 1 and any(
            bound_key.startswith(f"{side}_forager_")
            for side in candidate_search_sides
        ):
            # Forager ranking cannot affect a one-coin backtest.
            continue
        if any(
            bound_key.startswith(f"{side}_hsl_")
            for side in candidate_search_sides
        ):
            bound_side = bound_key.split("_", 1)[0]
            if bound_side not in hsl_search_sides:
                # Dormant HSL bounds affect neither proxy nor exact Rust.
                continue
        if (
            max_coin_count == 1
            and bound_key
            in {f"{side}_n_positions" for side in candidate_search_sides}
        ):
            continue
        if bound_key not in bound_map:
            raise ValueError(
                "GPU foundation cannot optimize active bound "
                f"{bound_key!r}; pin it or use the CPU optimizer"
            )

    specs = extract_objective_specs(config)
    metric_names = validate_gpu_metric_names(spec.metric for spec in specs)
    limit_metrics = validate_gpu_metric_names(
        check["metric"] for check in getattr(evaluator, "limit_checks", [])
    )
    needed_metrics = set(metric_names) | limit_metrics | {"backtest_completion_ratio"}
    _validate_hsl_metric_topology(
        needed_metrics,
        coin_count=max_coin_count,
        enabled_sides=enabled_sides,
        hard_stop_metrics=HARD_STOP_PROXY_METRICS,
        shared_account_controller=(
            max_coin_count > 1 and len(enabled_sides) == 2
        ),
    )
    _validate_dual_multicoin_metrics(
        needed_metrics,
        coin_count=max_coin_count,
        enabled_sides=enabled_sides,
        shared_account_controller=(
            max_coin_count > 1 and len(enabled_sides) == 2
        ),
    )

    if _apply_gpu_lean_tm_parallelism_defaults(
        options,
        proxy_config,
        bound_by_key,
        enabled_sides,
        suite_enabled=suite_enabled,
        coin_count=max_coin_count,
        requested_metric_features=mps_requested_metric_features(
            needed_metrics, strategy_kind=strategy_kind
        ),
    ):
        logging.info(
            "GPU lean Trailing Martingale parallelism selected | "
            "population=%d max_dispatch_candidate_bars=%d",
            int(options["population_size"]),
            int(options["max_dispatch_candidate_bars"]),
        )
    logging.info("GPU optimizer options: %s", options)

    if suite_enabled:
        scenario_proxy_groups = {}
        for item in suite_inputs:
            scenario_proxy = (
                MpsMulticoinProxy
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
                max_dispatch_candidate_bars=int(
                    options["max_dispatch_candidate_bars"]
                ),
                needed_metrics=needed_metrics,
                interrupt_check=interrupt_check,
            )
            item["coin_override_contract"] = getattr(
                scenario_proxy, "coin_override_contract", {}
            )
            item["proxy_checkpoint_contract"] = getattr(
                scenario_proxy, "checkpoint_contract", {}
            )
            group = scenario_proxy_groups.setdefault(
                id(item["ctx"]),
                [item["ctx"], [], item["parameter_overrides"]],
            )
            if group[2] != item["parameter_overrides"]:
                raise RuntimeError(
                    f"GPU suite scenario {item['ctx'].label!r} prepared datasets "
                    "disagree on candidate parameter overrides"
                )
            group[1].append((item["exchange"], scenario_proxy))
        scenario_proxies = [
            (ctx, tuple(exchange_proxies), parameter_overrides)
            for ctx, exchange_proxies, parameter_overrides in scenario_proxy_groups.values()
        ]
        profile_proxies = [
            proxy
            for _ctx, exchange_proxies, _parameter_overrides in scenario_proxies
            for _exchange, proxy in exchange_proxies
        ]

        def evaluate_proxy(candidates, *, history_fraction=1.0):
            if not math.isclose(
                float(history_fraction), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    "GPU suite proxy evaluation does not support partial history"
                )
            return _evaluate_gpu_suite_proxies(
                evaluator_for_pool,
                scenario_proxies,
                candidates,
            )

    else:
        proxy_cls = MpsMulticoinProxy if max_coin_count > 1 else MpsSingleCoinProxy
        proxy = proxy_cls(
            config=proxy_config,
            hlcvs=evaluator.shared_hlcvs_np[exchange],
            mss=evaluator.msss[exchange],
            btc=evaluator.shared_btc_np[exchange],
            timestamps=evaluator.timestamps.get(exchange),
            exchange=exchange,
            batch_size=int(options["batch_size"]),
            max_dispatch_candidate_bars=int(
                options["max_dispatch_candidate_bars"]
            ),
            needed_metrics=needed_metrics,
            interrupt_check=interrupt_check,
        )
        profile_proxies = [proxy]

        def evaluate_proxy(candidates, *, history_fraction=1.0):
            if float(history_fraction) < 1.0:
                history_start, trade_start = (
                    proxy.recent_window_for_history_fraction(history_fraction)
                )
                return proxy.evaluate(
                    candidates,
                    history_start_step=history_start,
                    trade_start_step=trade_start,
                )
            return proxy.evaluate(candidates)

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
    if checkpoint is not None and checkpoint.get("seed") is not None:
        seed = int(checkpoint["seed"])
    elif seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        logging.info("GPU optimizer generated reproducible seed %d", seed)
    seed = int(seed)
    rng = np.random.default_rng(seed)
    sampling = rng.random((population_size, len(active)))
    sampling[0] = normalize_vector(base_vector)
    seed_policy = options["seed_bootstrap"]
    objective_scale = _ObjectiveScale()
    seed_proxy_metrics = None
    seed_proxy_objectives = None
    seed_proxy_violations = None
    seed_proxy_scores = None
    seed_bootstrap_selections = []
    seed_population_indices = []
    seed_screen_complete = True
    checkpoint_seed_plan = (
        checkpoint.get("seed_bootstrap_plan") if checkpoint is not None else None
    )
    checkpoint_seed_contract = (
        checkpoint.get("seed_bootstrap_contract")
        if checkpoint is not None
        else None
    )
    if checkpoint_seed_contract is not None and (
        str(checkpoint_seed_contract.get("requested_mode"))
        != str(seed_policy["mode"])
        or int(checkpoint_seed_contract.get("max_exact", -1))
        != int(seed_policy["max_exact"])
    ):
        raise ValueError(
            "GPU checkpoint does not match current seed-bootstrap policy"
        )
    if checkpoint_seed_plan:
        starting_vectors = [
            [float(value) for value in vector]
            for vector in checkpoint_seed_plan["starting_vectors"]
        ]
        seed_bootstrap_mode = str(checkpoint_seed_plan["effective_mode"])
        seed_bootstrap_selections = [
            (int(index), bool(probe), bool(front))
            for index, probe, front in checkpoint_seed_plan["selections"]
        ]
        seed_population_indices = [
            int(index) for index in checkpoint_seed_plan["population_indices"]
        ]
        seed_screen_complete = bool(
            checkpoint_seed_plan.get("screen_complete", True)
        )
        seed_proxy_metrics = checkpoint_seed_plan.get("proxy_metrics")
        if checkpoint_seed_plan.get("proxy_objectives") is not None:
            seed_proxy_objectives = np.asarray(
                checkpoint_seed_plan["proxy_objectives"], dtype=np.float64
            )
            seed_proxy_violations = np.asarray(
                checkpoint_seed_plan["proxy_violations"], dtype=np.float64
            )
            objective_scale.fit(seed_proxy_objectives)
            seed_proxy_scores = objective_scale.score(seed_proxy_objectives)
        seed_bootstrap_contract = deepcopy(checkpoint_seed_contract)
        _validate_seed_bootstrap_plan(
            starting_vectors,
            seed_bootstrap_selections,
            seed_population_indices,
            seed_bootstrap_contract,
            hash_vector=vector_hash,
            proxy_metrics=seed_proxy_metrics,
            proxy_objectives=seed_proxy_objectives,
            proxy_violations=seed_proxy_violations,
            screen_complete=seed_screen_complete,
        )
    elif checkpoint_seed_contract is not None:
        starting_vectors = []
        seed_bootstrap_mode = str(
            checkpoint_seed_contract.get("effective_mode", "none")
        )
        seed_screen_complete = True
        seed_bootstrap_contract = deepcopy(checkpoint_seed_contract)
    else:
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
        if str(seed_policy["mode"]) != "legacy":
            starting_vectors, duplicate_count = (
                _deduplicate_canonical_seed_vectors(
                    starting_vectors,
                    hash_vector=vector_hash,
                )
            )
            if duplicate_count:
                logging.info(
                    "Dropped %d GPU seed configs made equivalent by runtime "
                    "overrides",
                    duplicate_count,
                )
        seed_bootstrap_mode = _effective_seed_bootstrap_mode(
            seed_policy, len(starting_vectors)
        )
        if seed_bootstrap_mode == "legacy":
            for index, vector in enumerate(
                starting_vectors[: population_size - 1], start=1
            ):
                sampling[index] = normalize_vector(vector)
        elif seed_bootstrap_mode == "exact":
            seed_bootstrap_selections = [
                (index, False, False) for index in range(len(starting_vectors))
            ]
        elif seed_bootstrap_mode == "screened":
            seed_screen_complete = False
        seed_hashes = [vector_hash(vector) for vector in starting_vectors]
        if seed_bootstrap_mode == "exact":
            selected_exact_count = len(starting_vectors)
        elif seed_bootstrap_mode == "screened":
            selected_exact_count = min(
                len(starting_vectors), int(seed_policy["max_exact"])
            )
        else:
            selected_exact_count = 0
        seed_bootstrap_contract = (
            {
                "version": 1,
                "requested_mode": str(seed_policy["mode"]),
                "effective_mode": seed_bootstrap_mode,
                "max_exact": int(seed_policy["max_exact"]),
                "seed_count": len(starting_vectors),
                "selected_exact_count": selected_exact_count,
                "all_seeds_exact": seed_bootstrap_mode == "exact",
                "seed_pool_sha256": hashlib.sha256(
                    "\n".join(seed_hashes).encode()
                ).hexdigest(),
            }
            if starting_vectors
            else None
        )
    if starting_vectors and seed_screen_complete:
        logging.info(
            "GPU seed bootstrap prepared | mode=%s seeds=%d exact=%d population=%d",
            seed_bootstrap_mode,
            len(starting_vectors),
            len(seed_bootstrap_selections),
            population_size,
        )

    problem = Problem(
        n_var=len(active),
        n_obj=len(specs),
        n_ieq_constr=1,
        xl=np.zeros(len(active)),
        xu=np.ones(len(active)),
    )
    algorithm_contract = _gpu_nsga2_checkpoint_contract(
        config,
        population_size=population_size,
        n_params=len(active),
    )
    algorithm = _build_gpu_nsga2(
        config,
        sampling=sampling,
        population_size=population_size,
        n_params=len(active),
        policy=algorithm_contract,
    )
    algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
    generation = 0
    exact_done = 0
    seed_exact_done = 0
    completed_hashes: set[str] = set()
    seed_bootstrap_payloads = {}
    seed_bootstrap_complete = seed_bootstrap_mode in {"none", "legacy"}
    drift_monitor = _DriftMonitor(options)
    persisted_halt_reason = None
    signature = _checkpoint_signature(
        active,
        config["optimize"]["scoring"],
        anchor_plan=get_anchor_plan(config),
        suite_contract=(
            _gpu_suite_checkpoint_contract(
                config,
                suite_inputs,
                pinned_hsl_bounds=_gpu_pinned_hsl_bound_contract(bound_by_key),
            )
            if suite_enabled
            else None
        ),
        runtime_contract=(
            None
            if suite_enabled
            else _gpu_runtime_checkpoint_contract(
                proxy_config,
                proxy,
                pinned_hsl_bounds=_gpu_pinned_hsl_bound_contract(bound_by_key),
            )
        ),
        search_contract=_gpu_search_checkpoint_contract(
            key_paths=key_paths,
            bounds=bounds,
            base_vector=base_vector,
            fixed_bound_values=fixed_bound_values,
            fixed_parameter_overrides=fixed_parameter_overrides,
            optimizer_overrides=gpu_optimizer_overrides,
            sig_digits=sig_digits,
            algorithm_contract=algorithm_contract,
            proxy_evaluation_policy=(
                {
                    **halving_policy,
                    "history_window": "recent_suffix_v1",
                }
                if halving_policy["enabled"]
                else None
            ),
            seed_bootstrap_contract=seed_bootstrap_contract,
        ),
    )
    budget = int(config["optimize"]["iters"])
    if budget <= 0:
        raise ValueError("optimize.iters must be greater than zero")

    if resume:
        if checkpoint.get("signature") != signature:
            raise ValueError(
                "GPU checkpoint does not match current search, scoring, suite, "
                "or prepared execution contract"
            )
        algorithm = checkpoint["algorithm"]
        seed = int(checkpoint.get("seed", seed))
        generation = int(checkpoint["generation"])
        exact_done = int(checkpoint["exact_done"])
        seed_exact_done = int(checkpoint.get("seed_exact_done", 0))
        seed_bootstrap_complete = bool(
            checkpoint.get("seed_bootstrap_complete", True)
        )
        if not seed_bootstrap_complete and not checkpoint_seed_plan:
            raise RuntimeError(
                "GPU checkpoint is missing its incomplete seed-bootstrap plan"
            )
        seed_bootstrap_payloads = dict(
            checkpoint.get("seed_bootstrap_payloads", {})
        )
        completed_hashes = set(checkpoint.get("completed_hashes", []))
        objective_scale.median = checkpoint.get("scale_median")
        objective_scale.spread = checkpoint.get("scale_spread")
        drift_monitor.pairs.extend(checkpoint.get("drift_pairs", []))
        persisted_halt_reason = checkpoint.get("halt_reason")
        recorded_exact = int(getattr(getattr(recorder, "store", None), "n_iters", 0))
        checkpoint_exact_total = seed_exact_done + exact_done
        if recorded_exact < checkpoint_exact_total:
            raise RuntimeError(
                "GPU checkpoint is ahead of durable all_results.bin state: "
                f"checkpoint={checkpoint_exact_total}, durable={recorded_exact}"
            )
        if recorded_exact > checkpoint_exact_total:
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

            if seed_bootstrap_complete:
                recovered_hashes, recovered_pairs = _recover_durable_validations(
                    load_results(results_file),
                    start_index=checkpoint_exact_total,
                    stop_index=recorded_exact,
                    vector_from_entry=vector_from_entry,
                    hash_vector=vector_hash,
                )
                exact_done += recorded_exact - checkpoint_exact_total
            else:
                (
                    recovered_seed_payloads,
                    recovered_hashes,
                    recovered_pairs,
                ) = _recover_durable_seed_bootstrap(
                    load_results(results_file),
                    start_index=checkpoint_exact_total,
                    stop_index=recorded_exact,
                    vector_from_entry=vector_from_entry,
                    hash_vector=vector_hash,
                )
                seed_bootstrap_payloads.update(recovered_seed_payloads)
                seed_exact_done += recorded_exact - checkpoint_exact_total
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
                checkpoint_exact_total,
                recorded_exact,
            )
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
            "Resumed GPU optimizer at generation %d with %d seed and %d "
            "evolution exact evaluations",
            generation,
            seed_exact_done,
            exact_done,
        )

    _disable_gpu_exact_duplicate_guard(evaluator_for_pool)
    adapter = PymooEvaluatorAdapter(evaluator_for_pool, overrides_list=overrides_list)
    profile_enabled = any(
        bool(getattr(item, "profile_enabled", False)) for item in profile_proxies
    )
    profile_totals = {
        "exact_queue_wait": 0.0,
        "exact_work": 0.0,
        "exact_result_processing": 0.0,
        "persistence": 0.0,
        "checkpointing": 0.0,
    }
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
    profile_started = time.perf_counter() if profile_enabled else 0.0
    proxy_evaluations = 0
    novelty_stall_generations = 0
    last_warning = None
    last_probe_shortfall = None
    last_checkpoint_at = 0.0
    last_checkpoint_exact = seed_exact_done + exact_done
    generation_in_progress = False

    def checkpoint_state() -> dict:
        seed_plan = None
        if not seed_bootstrap_complete and starting_vectors:
            seed_plan = {
                "effective_mode": seed_bootstrap_mode,
                "screen_complete": seed_screen_complete,
                "starting_vectors": starting_vectors,
                "selections": seed_bootstrap_selections,
                "population_indices": seed_population_indices,
                "proxy_metrics": seed_proxy_metrics,
                "proxy_objectives": (
                    None
                    if seed_proxy_objectives is None
                    else seed_proxy_objectives.tolist()
                ),
                "proxy_violations": (
                    None
                    if seed_proxy_violations is None
                    else seed_proxy_violations.tolist()
                ),
            }
        return {
            "signature": signature,
            "algorithm": algorithm,
            "generation": generation,
            "seed": seed,
            "exact_done": exact_done,
            "seed_exact_done": seed_exact_done,
            "seed_bootstrap_complete": seed_bootstrap_complete,
            "seed_bootstrap_payloads": (
                {} if seed_bootstrap_complete else seed_bootstrap_payloads
            ),
            "seed_bootstrap_contract": deepcopy(seed_bootstrap_contract),
            "seed_bootstrap_plan": seed_plan,
            "anchor_plan": deepcopy(get_anchor_plan(config)),
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
        result_count = seed_exact_done + exact_done
        if not force and (result_count == last_checkpoint_exact or not due):
            return
        profile_started = time.perf_counter() if profile_enabled else 0.0
        _save_checkpoint(checkpoint_path, checkpoint_state())
        if profile_enabled:
            profile_totals["checkpointing"] += (
                time.perf_counter() - profile_started
            )
        last_checkpoint_at = now
        last_checkpoint_exact = result_count

    def record_exact(
        vector,
        payload,
        *,
        validation_metadata: dict | None = None,
        seed_metadata: dict | None = None,
    ) -> None:
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
        if validation_metadata is not None:
            entry.setdefault("metrics", {})["gpu_validation"] = validation_metadata
        if seed_metadata is not None:
            entry.setdefault("metrics", {})["gpu_seed_bootstrap"] = seed_metadata
        recorder.record(_restore_gpu_result_run_contract(entry, config))

    def prepare_seed_proxy_screen() -> None:
        nonlocal seed_screen_complete
        nonlocal seed_proxy_metrics, seed_proxy_objectives
        nonlocal seed_proxy_violations, seed_proxy_scores
        nonlocal seed_bootstrap_selections, seed_population_indices
        if seed_bootstrap_mode != "screened" or seed_screen_complete:
            return

        # Persist the normalized canonical seed pool before the potentially
        # long Metal dispatch. An interruption can resume without the original
        # --start input; a completed screen is checkpointed again immediately.
        maybe_save_checkpoint(force=True)
        logging.info(
            "GPU seed proxy screen started | seeds=%d exact_cap=%d",
            len(starting_vectors),
            int(seed_policy["max_exact"]),
        )
        seed_proxy_started = time.perf_counter()
        seed_rows = np.asarray(
            [normalize_vector(vector) for vector in starting_vectors],
            dtype=np.float64,
        )
        proxy_metric_rows = evaluate_proxy(parameter_dicts(seed_rows))
        seed_proxy_seconds = time.perf_counter() - seed_proxy_started
        logging.info(
            "GPU seed proxy screen complete | seeds=%d wall=%.2fs rate=%.1f/s",
            len(starting_vectors),
            seed_proxy_seconds,
            len(starting_vectors) / max(seed_proxy_seconds, 1.0e-9),
        )
        seed_proxy_objectives, seed_proxy_violations = proxy_fitness(
            proxy_metric_rows
        )
        objective_scale.fit(seed_proxy_objectives)
        seed_proxy_scores = objective_scale.score(seed_proxy_objectives)
        seed_bootstrap_selections = _select_seed_bootstrap_indices(
            seed_proxy_objectives,
            seed_proxy_scores,
            seed_proxy_violations,
            total=min(len(starting_vectors), int(seed_policy["max_exact"])),
        )
        seed_proxy_metrics = {
            int(index): proxy_metric_rows[int(index)]
            for index, _probe, _front in seed_bootstrap_selections
        }
        seed_population_indices = _select_seed_population_indices(
            seed_proxy_objectives,
            seed_proxy_violations,
            count=min(len(starting_vectors), population_size - 1),
        )
        # Keep only compact objective arrays and the selected exact-validation
        # metric rows. Full per-seed metrics and normalized screen inputs can be
        # substantial for large seed archives.
        del proxy_metric_rows
        del seed_rows
        seed_screen_complete = True
        maybe_save_checkpoint(force=True)
        logging.info(
            "GPU seed bootstrap prepared | mode=%s seeds=%d exact=%d population=%d",
            seed_bootstrap_mode,
            len(starting_vectors),
            len(seed_bootstrap_selections),
            population_size,
        )

    def run_seed_bootstrap() -> None:
        nonlocal seed_exact_done, seed_bootstrap_complete, persisted_halt_reason
        if seed_bootstrap_complete:
            return

        prepare_seed_proxy_screen()

        # Preserve the normalized seed pool and proxy-screening result before
        # starting exact work. A hard interruption can then resume without
        # rereading private seed files or repeating the GPU screen.
        maybe_save_checkpoint(force=True)

        selected = []
        for source_index, is_probe, is_proxy_front in seed_bootstrap_selections:
            vector = starting_vectors[int(source_index)]
            digest = vector_hash(vector)
            if digest in seed_bootstrap_payloads:
                continue
            selected.append(
                (
                    int(source_index),
                    bool(is_probe),
                    bool(is_proxy_front),
                    vector,
                    digest,
                )
            )

        pending_seed = {}
        cursor = 0
        try:
            while cursor < len(selected) or pending_seed:
                interrupt_check()
                while cursor < len(selected) and len(pending_seed) < max_pending:
                    item = selected[cursor]
                    result = _submit_gpu_exact_validation(
                        pool,
                        item[3],
                        interrupt_check,
                        profile=profile_enabled,
                    )
                    pending_seed[result] = item
                    cursor += 1
                ready = _ready_submission_prefix(pending_seed)
                if not ready:
                    PymooAsyncRecordingRunner._raise_if_pool_workers_exited(
                        pool_workers
                    )
                    time.sleep(0.05)
                    continue
                for result in ready:
                    interrupt_check()
                    (
                        source_index,
                        is_probe,
                        is_proxy_front,
                        vector,
                        digest,
                    ) = pending_seed.pop(result)
                    payload = result.get()
                    PymooAsyncRecordingRunner._raise_if_worker_failure(
                        payload, source_index
                    )
                    validation_metadata = None
                    proxy_score = None
                    proxy_violation = None
                    classification_mismatch = False
                    if seed_bootstrap_mode == "screened":
                        proxy_score = float(seed_proxy_scores[source_index])
                        proxy_violation = float(seed_proxy_violations[source_index])
                        exact_score = float(
                            objective_scale.score(
                                np.asarray(payload["F"], dtype=np.float64).reshape(
                                    1, -1
                                )
                            )[0]
                        )
                        classification_mismatch = (
                            _constraint_classification_mismatch(
                                proxy_violation, payload
                            )
                        )
                        validation_metadata = {
                            "schema_version": 2,
                            "phase": "seed_bootstrap",
                            "proxy_score": proxy_score,
                            "exact_score": exact_score,
                            "probe": is_probe,
                            "proxy_front": is_proxy_front,
                            "constraint_classification_mismatch": (
                                classification_mismatch
                            ),
                            "constraint_diagnostics": _constraint_diagnostics(
                                evaluator,
                                seed_proxy_metrics[source_index],
                                payload,
                            ),
                        }
                        drift_monitor.add(
                            proxy_score,
                            exact_score,
                            probe=is_probe,
                            proxy_front=is_proxy_front,
                            constraint_mismatch=classification_mismatch,
                        )
                    seed_metadata = {
                        "schema_version": 1,
                        "mode": seed_bootstrap_mode,
                        "source_index": source_index,
                        "seed_count": len(starting_vectors),
                        "selected_exact_count": len(seed_bootstrap_selections),
                        "proxy_evaluated": seed_bootstrap_mode == "screened",
                        "all_seeds_exact": seed_bootstrap_mode == "exact",
                        "exact_objectives": np.asarray(
                            payload["F"], dtype=np.float64
                        ).tolist(),
                        "exact_violation": float(
                            np.asarray(payload.get("G", [-1.0])).reshape(-1)[0]
                        ),
                    }
                    record_exact(
                        vector,
                        payload,
                        validation_metadata=validation_metadata,
                        seed_metadata=seed_metadata,
                    )
                    seed_bootstrap_payloads[digest] = {
                        "source_index": source_index,
                        "F": np.asarray(payload["F"], dtype=np.float64).tolist(),
                        "G": np.asarray(
                            payload.get("G", [-1.0]), dtype=np.float64
                        ).tolist(),
                    }
                    seed_exact_done += 1
                    completed_hashes.add(digest)
                    # all_results.bin is flushed for every exact result and
                    # recovery replays durable results beyond a stale
                    # checkpoint.  Honor the configured checkpoint interval
                    # instead of rewriting the complete seed plan per seed.
                    maybe_save_checkpoint()
        except KeyboardInterrupt:
            cancel_pending_async_results(pending_seed)
            maybe_save_checkpoint(force=True)
            raise

        ordered_payloads = []
        for source_index, _is_probe, _is_proxy_front in seed_bootstrap_selections:
            digest = vector_hash(starting_vectors[int(source_index)])
            payload = seed_bootstrap_payloads.get(digest)
            if payload is None:
                raise RuntimeError(
                    "GPU seed bootstrap completed without exact evidence for "
                    f"source index {source_index}"
                )
            ordered_payloads.append(payload)
        exact_objectives = np.asarray(
            [payload["F"] for payload in ordered_payloads], dtype=np.float64
        )
        exact_violations = np.asarray(
            [
                float(np.asarray(payload["G"]).reshape(-1)[0])
                for payload in ordered_payloads
            ],
            dtype=np.float64,
        )
        exact_preference = _select_seed_population_indices(
            exact_objectives,
            exact_violations,
            count=min(len(ordered_payloads), population_size - 1),
        )
        exact_source_indices = [
            int(seed_bootstrap_selections[index][0]) for index in exact_preference
        ]
        if seed_bootstrap_mode == "exact":
            population_seed_indices = exact_source_indices
        else:
            population_seed_indices = list(exact_source_indices)
            population_seed_ids = set(population_seed_indices)
            population_seed_indices.extend(
                index
                for index in seed_population_indices
                if index not in population_seed_ids
            )
        for slot, source_index in enumerate(
            population_seed_indices[: population_size - 1], start=1
        ):
            sampling[slot] = normalize_vector(starting_vectors[source_index])
        algorithm.initialization.sampling = sampling

        status = drift_monitor.evaluate()
        if status["halt_reason"]:
            persisted_halt_reason = status["halt_reason"]
            maybe_save_checkpoint(force=True)
            raise RuntimeError(status["halt_reason"])
        seed_bootstrap_complete = True
        logging.info(
            "GPU seed bootstrap complete | mode=%s proxy=%d exact=%d "
            "population_seeds=%d",
            seed_bootstrap_mode,
            len(starting_vectors) if seed_bootstrap_mode == "screened" else 0,
            seed_exact_done,
            min(len(population_seed_indices), population_size - 1),
        )
        maybe_save_checkpoint(force=True)

    def consume_ready(*, wait_for_one: bool = False) -> None:
        nonlocal exact_done, last_warning, persisted_halt_reason
        while True:
            interrupt_check()
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
            interrupt_check()
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
            worker_seconds = (
                float(payload.pop("__gpu_profile_worker_seconds__", 0.0))
                if profile_enabled and isinstance(payload, dict)
                else 0.0
            )
            queue_wait_seconds = (
                float(payload.pop("__gpu_profile_queue_wait_seconds__", 0.0))
                if profile_enabled and isinstance(payload, dict)
                else 0.0
            )
            if profile_enabled:
                profile_totals["exact_work"] += worker_seconds
                profile_totals["exact_queue_wait"] += queue_wait_seconds
                result_processing_started = time.perf_counter()
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
            persistence_started = (
                time.perf_counter() if profile_enabled else 0.0
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
            if profile_enabled:
                persistence_seconds = time.perf_counter() - persistence_started
                profile_totals["persistence"] += persistence_seconds
                profile_totals["exact_result_processing"] += max(
                    0.0,
                    time.perf_counter()
                    - result_processing_started
                    - persistence_seconds,
                )
            exact_done += 1
            completed_hashes.add(digest)
            submitted_hashes.discard(digest)
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
        run_seed_bootstrap()
        while exact_done < budget:
            interrupt_check()
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

            # An MPS proxy may poll the latch between bounded Metal dispatches.
            # If interrupted there, the outer handler discards this incomplete
            # ask/tell transaction and retains the preceding safe checkpoint.
            generation_profile_started = (
                time.perf_counter() if profile_enabled else 0.0
            )
            ask_started = time.perf_counter() if profile_enabled else 0.0
            population = _ask_gpu_population(algorithm, interrupt_check)
            ask_seconds = (
                time.perf_counter() - ask_started if profile_enabled else 0.0
            )
            generation_in_progress = True
            rows = np.asarray(population.get("X"), dtype=np.float64)
            materialization_started = (
                time.perf_counter() if profile_enabled else 0.0
            )
            proxy_candidates = parameter_dicts(rows)
            candidate_materialization_seconds = (
                time.perf_counter() - materialization_started
                if profile_enabled
                else 0.0
            )
            proxy_started = time.perf_counter() if profile_enabled else 0.0
            proxy_profile_records = []

            def capture_halving_profile(rung, history_fraction, candidate_count):
                if not profile_enabled:
                    return
                for item in profile_proxies:
                    record = deepcopy(getattr(item, "last_profile", {}))
                    record.update(
                        successive_halving_rung=int(rung),
                        history_fraction=float(history_fraction),
                        rung_candidate_count=int(candidate_count),
                    )
                    proxy_profile_records.append(record)

            if halving_policy["enabled"]:
                (
                    metric_rows,
                    proxy_objectives,
                    proxy_violations,
                    full_rung_indices,
                    halving_trace,
                ) = _evaluate_successive_halving(
                    proxy_candidates,
                    policy=halving_policy,
                    evaluate_proxy=evaluate_proxy,
                    proxy_fitness=proxy_fitness,
                    interrupt_check=interrupt_check,
                    stage_callback=capture_halving_profile,
                )
            else:
                metric_rows = evaluate_proxy(proxy_candidates)
                proxy_objectives, proxy_violations = proxy_fitness(metric_rows)
                full_rung_indices = np.arange(len(rows), dtype=np.int64)
                halving_trace = []
            proxy_seconds = (
                time.perf_counter() - proxy_started if profile_enabled else 0.0
            )
            proxy_evaluations += (
                sum(int(item["candidate_count"]) for item in halving_trace)
                if halving_trace
                else len(rows)
            )
            if halving_trace:
                logging.info(
                    "GPU successive halving | gen=%d rungs=%s full_history=%d/%d",
                    generation + 1,
                    ",".join(
                        f"{item['history_fraction']:.0%}:{item['candidate_count']}"
                        for item in halving_trace
                    ),
                    len(full_rung_indices),
                    len(rows),
                )
            if objective_scale.median is None:
                objective_scale.fit(proxy_objectives[full_rung_indices])
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
            tell_started = time.perf_counter() if profile_enabled else 0.0
            algorithm.tell(infills=population)
            tell_seconds = (
                time.perf_counter() - tell_started if profile_enabled else 0.0
            )
            generation_in_progress = False
            generation += 1
            # PyTorch MPS may consume KeyboardInterrupt while waiting for a
            # Metal dispatch. Finish the in-progress ask/tell transaction, then
            # honor the latched signal before exact work is submitted. This is
            # also a safe point for serializing a resumable checkpoint.
            interrupt_check()

            probe_count = _validation_probe_count(
                validation_count,
                int(options["validate_per_generation"]),
                int(options["drift_probes"]),
            )
            full_rung_selections = _select_validation_indices(
                proxy_objectives[full_rung_indices],
                proxy_scores[full_rung_indices],
                proxy_violations[full_rung_indices],
                total=validation_count,
                probes=probe_count,
            )
            selections = [
                (int(full_rung_indices[index]), is_probe, is_proxy_front)
                for index, is_probe, is_proxy_front in full_rung_selections
            ]
            while True:
                try:
                    exact_selections = _select_exact_validations(
                        selections,
                        total=validation_count,
                        candidate_for_index=lambda index: full_vector(rows[index]),
                        digest_for_candidate=vector_hash,
                        completed_hashes=completed_hashes,
                        submitted_hashes=submitted_hashes,
                    )
                    break
                except _ProxyFrontValidationPending:
                    if not pending:
                        raise RuntimeError(
                            "GPU validation marked proxy-front evidence pending "
                            "without an exact job in flight"
                        )
                    consume_ready(wait_for_one=True)
                    if exact_done + len(pending) >= budget:
                        exact_selections = []
                        break
            actual_probe_count = sum(bool(item[1]) for item in exact_selections)
            last_probe_shortfall = _update_probe_shortfall_log(
                last_probe_shortfall,
                requested=probe_count,
                actual=actual_probe_count,
            )
            submitted_this_generation = 0
            for index, is_probe, is_proxy_front, vector, digest in exact_selections:
                result = _submit_gpu_exact_validation(
                    pool,
                    vector,
                    interrupt_check,
                    profile=profile_enabled,
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

            if profile_enabled:
                generation_wall_seconds = (
                    time.perf_counter() - generation_profile_started
                )
                if not halving_policy["enabled"]:
                    proxy_profile_records = [
                        deepcopy(getattr(item, "last_profile", {}))
                        for item in profile_proxies
                    ]
                proxy_profile_wall = sum(
                    float(item.get("wall_seconds", 0.0))
                    for item in proxy_profile_records
                )
                accounted_seconds = (
                    ask_seconds
                    + candidate_materialization_seconds
                    + proxy_seconds
                    + tell_seconds
                )
                _log_gpu_profile(
                    "generation",
                    generation=generation,
                    population_size=len(rows),
                    successive_halving=halving_trace,
                    full_history_candidate_count=len(full_rung_indices),
                    proxy_profiles=proxy_profile_records,
                    timings_seconds={
                        "nsga_ask": ask_seconds,
                        "candidate_materialization": (
                            candidate_materialization_seconds
                        ),
                        "proxy_evaluation": proxy_seconds,
                        "suite_or_proxy_reduction": max(
                            0.0, proxy_seconds - proxy_profile_wall
                        ),
                        "nsga_tell": tell_seconds,
                        "orchestration": max(
                            0.0, generation_wall_seconds - accounted_seconds
                        ),
                        "wall": generation_wall_seconds,
                    },
                    exact_validation_cumulative_seconds=dict(profile_totals),
                    exact_completed=exact_done,
                    exact_inflight=len(pending),
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
        if profile_enabled:
            _log_gpu_profile(
                "complete",
                generations=generation,
                proxy_evaluations=proxy_evaluations,
                exact_completed=exact_done,
                wall_seconds=_gpu_profile_elapsed(profile_started),
                exact_validation_cumulative_seconds=dict(profile_totals),
            )
        logging.info(
            "GPU optimization complete | generations=%d proxy=%d seed_exact=%d "
            "evolution_exact=%d wall=%.1fs",
            generation,
            proxy_evaluations,
            seed_exact_done,
            exact_done,
            time.time() - start_time,
        )
        return {"pool": pool, "pool_terminated": False}
    except KeyboardInterrupt:
        cancel_pending_async_results(pending)
        try:
            _checkpoint_gpu_interrupt(
                generation_in_progress=generation_in_progress,
                generation=generation,
                exact_done=exact_done,
                save_checkpoint=maybe_save_checkpoint,
            )
        except Exception:
            logging.exception(
                "Failed to save GPU checkpoint during interrupt shutdown"
            )
        pool.terminate()
        raise OptimizerBackendInterrupted(pool=pool, pool_terminated=True)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
