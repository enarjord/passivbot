"""Declared configuration serialization types and code-owned field metadata.

Types are explicit even when a default is null, omitted, or written with a
different numeric spelling. Defaults remain owned by schema.py and Rust strategy
metadata. Existing path and optimizer adapters supply bindings; this registry
does not introduce configuration aliases or alter loading/override policy.
"""

from copy import deepcopy
from dataclasses import replace

from .field_spec import CoinOverridePolicy, FieldDomain, FieldRegistry, FieldSpec, FieldType, NO_DEFAULT


def get_special_optimizer_bindings():
    """Preserve the existing optimizer special-validation membership exactly."""
    from .field_policies import SPECIAL_OPTIMIZER_KEY_PATHS

    return dict(SPECIAL_OPTIMIZER_KEY_PATHS)


def get_shared_optimizer_bindings():
    """Project existing shared bound adapters into canonical parameter paths."""
    from .optimize_bounds import SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY, _shared_bound_local_path

    paths = get_special_optimizer_bindings()
    for side in ("long", "short"):
        for group, keys in SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.items():
            for local, flat in keys.items():
                paths[f"{side}_{flat}"] = ("bot", side, group, *_shared_bound_local_path(group, local))
    for local, native in SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY["hsl"].items():
        paths[native] = ("bot", "global", "hsl", local)
    return paths


def get_coin_override_paths(hsl_signal_mode="coin"):
    """Registry-led view of the existing shared per-coin permission policy."""
    from .field_policies import CONDITIONAL_HSL_OVERRIDE_PATHS, OVERRIDABLE_SHARED_BOT_PATHS

    if hsl_signal_mode not in {"coin", "position_side", "portfolio"}:
        raise ValueError("invalid HSL scope for coin override policy")
    return (OVERRIDABLE_SHARED_BOT_PATHS if hsl_signal_mode == "coin" else
            OVERRIDABLE_SHARED_BOT_PATHS - CONDITIONAL_HSL_OVERRIDE_PATHS)


def describe_field(path):
    """Serializable metadata report for explain/editor consumers; never policy."""
    from dataclasses import asdict

    parts = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
    registry = get_field_registry()
    field = registry.find(parts)
    sparse = parts[:1] == ("coin_overrides",)
    if field is None and sparse and len(parts) > 2:
        inherited = registry.find(parts[2:])
        if inherited is not None and inherited.coin_override in {
            CoinOverridePolicy.ALLOWED, CoinOverridePolicy.COIN_HSL_ONLY,
        }:
            field = inherited
    if field is None:
        raise KeyError("unknown declared configuration field: " + ".".join(map(str, parts)))
    return {
        "path": ".".join(map(str, parts)), "types": [kind.value for kind in field.types],
        "nullable": field.nullable,
        "default": {"declared": not sparse and field.default is not NO_DEFAULT,
                    **({"value": deepcopy(field.default)} if not sparse and field.default is not NO_DEFAULT else {})},
        "scope": field.scope, "consumers": list(field.consumers),
        "unit": field.unit, "denominator": field.denominator,
        "domain": asdict(field.domain) if field.domain else None,
        "description": field.description, "source": field.source,
        "optimizer": {"eligible": False if sparse else field.optimizable, "key": field.optimize_key,
                      "when": field.optimize_when, "recommended_bounds": field.recommended_bounds},
        "coin_override": field.coin_override.value if field.coin_override else None,
    }
# One structural definition for operational field identities, current paths and
# their released-v8 origins. Internal live-value APIs address these identities;
# authored configuration accepts only the returned canonical paths.
LIVE_OPERATIONAL_PATHS = {
    name: ("live", group, name)
    for group, names in {
        "account": "user balance_override balance_hysteresis_snap_pct leverage margin_mode_preference",
        "api": "custom_endpoints_path recv_window_ms max_concurrent_api_requests market_snapshot_ticker_strategy",
        "candles": """
            candle_lock_timeout_seconds enable_archive_candle_fetch enable_forager_ws_candles
            forager_ws_candle_rest_audit_minutes inactive_coin_candle_ttl_minutes
            max_active_candle_tail_gap_minutes max_disk_candles_per_symbol_per_tf
            max_forager_candle_refresh_seconds max_forager_candle_staleness_minutes
            max_memory_candles_per_symbol max_ohlcv_fetches_per_minute
        """,
        "startup": """
            defer_broad_candle_warmup force_cold_startup risk_bootstrap_max_lookback_minutes
            startup_phase_budgets warmup_concurrency warmup_jitter_seconds
        """,
        "fills": """
            fee_conversion_max_age_ms fee_pct_fallback fee_pct_sanity_abs_max
            fills_confirmation_overlap_minutes fills_recent_overlap_minutes
        """,
        "execution": """
            execution_delay_seconds limit_order_create_max_market_dist_pct
            max_n_cancellations_per_batch max_n_creations_per_batch order_match_tolerance_pct
            order_replacement_churn_gate_activation_count order_replacement_churn_gate_market_dist_pct
            order_replacement_churn_gate_stability_minutes order_replacement_churn_gate_window_minutes
            time_in_force
        """,
        "recovery": "exchange_symbol_unavailable_cooldown_hours max_n_restarts_per_day",
    }.items()
    for name in names.split()
}
LIVE_OPERATIONAL_PATHS.update(
    {
        "forced_mode_long": ("live", "modes", "long"),
        "forced_mode_short": ("live", "modes", "short"),
        "disapproved_position_mode": ("live", "modes", "disapproved_position"),
        "recv_window_ms": ("live", "api", "recv_window_milliseconds"),
        "fee_conversion_max_age_ms": ("live", "fills", "fee_conversion_max_age_milliseconds"),
    }
)


def live_field_path(field_id: str) -> str:
    """Return an explicit canonical path for an internal operational field ID."""
    if field_id == "base_config_path":
        # Internal source provenance; never an authored operational field.
        return "live.base_config_path"
    return ".".join(LIVE_OPERATIONAL_PATHS[field_id])


# These declarations describe the current public schema. Keep numeric types
# explicit: never infer them from the type or spelling of a default value.
_FIELDS_BY_TYPE = {
    FieldType.INTEGER: {
        "backtest": "balance_sample_divider candle_interval_minutes",
        "live": """
            fee_conversion_max_age_ms leverage max_disk_candles_per_symbol_per_tf
            max_memory_candles_per_symbol max_n_cancellations_per_batch
            max_n_creations_per_batch max_n_restarts_per_day max_ohlcv_fetches_per_minute
            order_replacement_churn_gate_activation_count
            recv_window_ms warmup_concurrency
        """,
        "logging": "backup_count level",
        "monitor": "max_total_bytes price_tick_min_interval_milliseconds",
        "optimize": "iters n_cpus pareto_max_size round_to_n_significant_digits",
        "optimize.prefilter": "grid_max_entries",
        "optimize.gpu.screening": "min_survivors",
    },
    FieldType.FLOAT: {
        "backtest": """
            btc_collateral_cap equity_floor_ratio gap_tolerance_ohlcvs_minutes
            maker_fee_override market_order_slippage_pct limit_order_fill_buffer_pct
            starting_balance taker_fee_override
        """,
        "live": """
            balance_hysteresis_snap_pct candle_lock_timeout_seconds execution_delay_seconds
            exchange_symbol_unavailable_cooldown_hours fee_pct_fallback fee_pct_sanity_abs_max
            fills_confirmation_overlap_minutes fills_recent_overlap_minutes
            forager_ws_candle_rest_audit_minutes
            inactive_coin_candle_ttl_minutes limit_order_create_max_market_dist_pct
            max_active_candle_tail_gap_minutes
            max_forager_candle_refresh_seconds order_match_tolerance_pct
            order_replacement_churn_gate_market_dist_pct
            order_replacement_churn_gate_stability_minutes
            order_replacement_churn_gate_window_minutes risk_bootstrap_max_lookback_minutes
            warmup_jitter_seconds
        """,
        "logging": "max_bytes_mb memory_snapshot_interval_minutes volume_refresh_info_threshold_seconds",
        "monitor": """
            checkpoint_interval_minutes event_rotation_mb event_rotation_minutes
            retain_days snapshot_interval_seconds
        """,
        "optimize.prefilter": "adverse_directionality volatility_ema_1m volatility_ema_1h penalty_weight",
        "optimize.gpu": "checkpoint_interval_seconds",
        "optimize.gpu.screening": "survival_fraction",
        "optimize.pymoo.shared": "crossover_eta crossover_prob_var mutation_eta",
        "bot.global.loss_gate": "max_drawdown_ratio memory_ema_span_minutes",
        "bot.global.unstuck": "memory_ema_span_minutes",
        "bot.global.universe": "minimum_coin_age_days",
        "bot.global.execution": "market_order_near_touch_threshold",
        "bot.global.forager": "score_hysteresis_pct",
        "bot.global.warmup": "ratio max_minutes",
    },
    FieldType.BOOLEAN: {
        "backtest": """
            compress_cache dynamic_wel_by_tradability filter_by_min_effective_cost
            hsl_detailed_report offline suite_enabled volume_normalization
        """,
        "live": """
            defer_broad_candle_warmup enable_archive_candle_fetch enable_forager_ws_candles
            force_cold_startup
        """,
        "logging": "persist_to_file rotation",
        "monitor": """
            compress_rotated_segments emit_completed_candles enabled include_raw_fill_payloads
            retain_candles retain_fills retain_price_ticks
        """,
        "optimize": "compress_results_file write_all_results",
        "optimize.pymoo.shared": "eliminate_duplicates",
        "bot.global": "hedge_mode",
        "bot.global.universe": "filter_by_min_effective_cost",
        "bot.global.execution": "market_orders_allowed",
    },
    FieldType.STRING: {
        "": "config_version",
        "backtest": "base_dir end_date hlcvs_data_override_mode start_date",
        "live": """
            disapproved_position_mode forced_mode_long forced_mode_short margin_mode_preference
            market_snapshot_ticker_strategy time_in_force user
        """,
        "logging": "dir",
        "monitor": "root_dir",
        "optimize": "backend",
        "optimize.gpu": "tuning_mode",
        "optimize.pymoo": "algorithm",
        "optimize.pymoo.algorithms.nsga3.ref_dirs": "method",
        "bot.global.hsl": "scope",
        "bot.global": "strategy_kind",
    },
    FieldType.ARRAY: {
        "backtest": "exchanges scenarios visible_metrics",
        "logging": "live_event_debug_profiles",
        "optimize": "enable_overrides fixed_params limits scoring",
        "optimize.prefilter": "checks exposure_ratios",
        "optimize.gpu.screening": "scenarios",
    },
    FieldType.OBJECT: {
        "backtest": "coin_sources market_settings_sources",
        "backtest.market_settings": "overrides overrides_by_exchange",
        "live": "startup_phase_budgets",
        "optimize": "fixed_runtime_overrides",
        "optimize.pymoo.algorithms": "nsga2",
        "": "coin_overrides",
    },
}

_NULLABLE_FIELDS = {
    FieldType.FLOAT: (
        "backtest.btc_collateral_ltv_cap", "backtest.ew_half_life_days",
        "live.balance_override", "live.max_forager_candle_staleness_minutes",
    ),
    FieldType.INTEGER: (
        "live.max_concurrent_api_requests", "optimize.population_size", "optimize.seed",
    ),
    FieldType.STRING: (
        "backtest.organillo_csv", "backtest.organillo_sha256", "backtest.hlcvs_data_dir",
        "backtest.ohlcv_source_dir", "live.custom_endpoints_path", "optimize.objective_scenario",
    ),
}

_SHARED_SIDE_FIELDS = {
    FieldType.INTEGER: "risk.max_positions",
    FieldType.ARRAY: "forager.always_include",
    FieldType.FLOAT: """
        forager.score_weights.unilateralness forager.score_weights.ema_readiness
        forager.score_weights.volatility forager.score_weights.volume
        forager.unilateralness_ema_span_minutes forager.volatility_ema_span_minutes
        forager.volume_ema_span_minutes hsl.memory_ema_span_minutes
        hsl.cooldown_after_red_minutes hsl.drawdown_ratio_ema_span_minutes hsl.drawdown_ratio_threshold
        entry_cooldown.base_duration_minutes entry_cooldown.min_duration_minutes
        entry_cooldown.weights_minutes.exposure_ratio
        entry_cooldown.weights_minutes.adverse_directionality
        risk.position_exposure_enforcer.threshold risk.total_exposure_enforcer.threshold
        risk.total_wallet_exposure_limit risk.we_excess_allowance_pct
        unstuck.close_pct unstuck.ema_distance_fraction unstuck.ema_span_0_minutes unstuck.ema_span_1_minutes
        unstuck.loss_allowance_pct unstuck.exposure_trigger_ratio
    """,
    FieldType.BOOLEAN: """
        hsl.enabled hsl.scale_budget_with_excess_allowance
        risk.position_exposure_enforcer.enabled risk.total_exposure_enforcer.enabled
        unstuck.ema_gating_enabled unstuck.enabled
    """,
    FieldType.STRING: "hsl.panic_close_order_type risk.total_exposure_enforcer.policy",
}


def _template_value(template, path):
    node = template
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return NO_DEFAULT
        node = node[part]
    return deepcopy(node)


def get_field_registry() -> FieldRegistry:
    """Return a fresh registry; returned defaults never mutate schema defaults."""
    from .schema import get_template_config
    from .strategy_spec import get_strategy_spec, get_supported_strategy_kinds

    template = get_template_config()
    fields = []

    def add(path, types, *, nullable=False, source="python", unit=None,
            description="", optimize_key=None, domain=None, recommended_bounds=None,
            coin_override=None):
        parts = tuple(path.split("."))
        if (
            parts[0] == "live"
            and len(parts) >= 2
            and parts[1] in LIVE_OPERATIONAL_PATHS
        ):
            parts = (*LIVE_OPERATIONAL_PATHS[parts[1]], *parts[2:])
        semantic_parts = parts[2:] if parts[0] == "coin_overrides" else parts
        is_bot_field = semantic_parts[:1] == ("bot",)
        scope = None
        if is_bot_field:
            scope = semantic_parts[1] if semantic_parts[1] in {"long", "short"} else "global"
        fields.append(FieldSpec(
            path=parts, types=types if isinstance(types, tuple) else (types,),
            nullable=nullable, default=_template_value(template, parts), source=source,
            unit=unit, description=description, optimize_key=optimize_key,
            scope=scope,
            consumers=("live", "backtest", "optimize") if is_bot_field else (),
            domain=domain, order=len(fields), recommended_bounds=recommended_bounds,
            coin_override=coin_override,
        ))

    for kind, groups in _FIELDS_BY_TYPE.items():
        for prefix, names in groups.items():
            for name in names.split():
                path = f"{prefix}.{name}" if prefix else name
                add(path, kind, nullable=path in {
                    "backtest.filter_by_min_effective_cost",
                    "backtest.maker_fee_override", "backtest.taker_fee_override",
                })
    for kind, paths in _NULLABLE_FIELDS.items():
        for path in paths:
            add(path, kind, nullable=True)

    for side in ("long", "short"):
        for kind, names in _SHARED_SIDE_FIELDS.items():
            for name in names.split():
                add(f"bot.{side}.{name}", kind)
        add(f"bot.{side}.entry_cooldown.max_duration_minutes", FieldType.FLOAT,
            nullable=True, unit="minutes", description="Null means an unbounded duration.")
        # This setting exists only inside coin patches, never as a root bot field.
        add(f"coin_overrides.*.bot.{side}.wallet_exposure_limit", FieldType.FLOAT,
            coin_override=CoinOverridePolicy.ALLOWED)
    for side in ("long", "short"):
        add(f"bot.{side}.forager.always_include.*", FieldType.STRING,
            description="Persistent selection identifier; order is retained and duplicates are removed.")
    add("coin_overrides.*.override_config_path", FieldType.STRING)

    # Portfolio HSL authority is explicitly optional; no default is synthesized.
    for name in ("enabled", "scale_budget_with_excess_allowance"):
        add(f"bot.global.hsl.{name}", FieldType.BOOLEAN)
    add("bot.global.hsl.panic_close_order_type", FieldType.STRING)
    for name in ("drawdown_ratio_threshold", "drawdown_ratio_ema_span_minutes",
                 "memory_ema_span_minutes", "cooldown_after_red_minutes"):
        add(f"bot.global.hsl.{name}", FieldType.FLOAT)

    for kind in get_supported_strategy_kinds():
        spec = get_strategy_spec(kind)
        for parameter in spec["parameters"]:
            native_path = parameter["config_path"]
            path = ".".join(("bot", native_path[1], "strategy", kind, *native_path[2:]))
            add(path, FieldType.FLOAT, source="rust strategy metadata",
                optimize_key=parameter["optimize_key"],
                recommended_bounds=tuple(parameter["bounds"]))
        for parameter in spec["fixed_parameters"]:
            native_path = parameter["config_path"]
            path = ".".join(("bot", native_path[1], "strategy", kind, *native_path[2:]))
            add(path, FieldType.STRING, source="rust strategy metadata",
                domain=FieldDomain(choices=tuple(parameter["allowed_values"])))

    # Union fields have one unambiguous numeric serialization type.
    for path in ("optimize.gpu.batch_size", "optimize.gpu.max_dispatch_candidate_bars"):
        add(path, (FieldType.INTEGER, FieldType.STRING), nullable=True)
    add("optimize.pymoo.algorithms.nsga3.ref_dirs.n_partitions",
        (FieldType.INTEGER, FieldType.STRING))
    for name in ("mutation_prob", "mutation_prob_per_variable"):
        add(f"optimize.pymoo.shared.{name}", (FieldType.FLOAT, FieldType.STRING))
    for name in ("approved_coins", "ignored_coins"):
        for side in ("long", "short"):
            add(f"bot.{side}.universe.{name}", (FieldType.ARRAY, FieldType.STRING), nullable=True)
    add("backtest.reducer.*", FieldType.STRING)
    for path in ("backtest.coin_sources.*", "backtest.market_settings_sources.*",
                 "backtest.scenarios.*.coin_sources.*"):
        add(path, FieldType.STRING, nullable=True)

    # Dynamic records are typed by field, independent of example count/order.
    for prefix in ("backtest.market_settings.overrides.*",
                   "backtest.market_settings.overrides_by_exchange.*.*"):
        for name in ("price_step", "qty_step", "min_qty", "min_cost", "c_mult"):
            add(f"{prefix}.{name}", FieldType.FLOAT)
    add("optimize.limits.*.value", FieldType.FLOAT)
    add("optimize.limits.*.range", FieldType.ARRAY)
    add("optimize.limits.*.range.*", FieldType.FLOAT)
    add("optimize.limits.*.enabled", FieldType.BOOLEAN)
    for name in ("metric", "penalize_if", "reducer", "scenario"):
        add(f"optimize.limits.*.{name}", FieldType.STRING, nullable=name == "scenario")
    for name in ("goal", "metric", "reducer", "scenario"):
        add(f"optimize.scoring.*.{name}", FieldType.STRING, nullable=name == "scenario")
    for name in ("label", "start_date", "end_date"):
        add(f"backtest.scenarios.*.{name}", FieldType.STRING, nullable=True)
    # Suite coin selection also accepts a long/short object. Exchanges have
    # their own narrower string/list/null contract in _coerce_exchange_list.
    for name in ("coins", "ignored_coins"):
        add(f"backtest.scenarios.*.{name}",
            (FieldType.ARRAY, FieldType.STRING, FieldType.OBJECT), nullable=True)
    add("backtest.scenarios.*.exchanges", (FieldType.ARRAY, FieldType.STRING), nullable=True)
    add("backtest.scenarios.*.coin_sources", FieldType.OBJECT, nullable=True)
    add("backtest.scenarios.*.overrides", FieldType.OBJECT)
    add("optimize.prefilter.exposure_ratios.*", FieldType.FLOAT)
    add("optimize.prefilter.checks.*.enabled", FieldType.BOOLEAN)
    add("optimize.prefilter.checks.*.metric", FieldType.STRING)
    add("optimize.prefilter.checks.*.sides", FieldType.ARRAY)
    for name in ("min", "max", "scale", "weight"):
        add(f"optimize.prefilter.checks.*.{name}", FieldType.FLOAT,
            nullable=name in {"min", "max"})
    add("live.startup_phase_budgets.*", FieldType.OBJECT)
    for name in ("elapsed_milliseconds", "since_previous_milliseconds"):
        add(f"live.startup_phase_budgets.*.{name}", FieldType.INTEGER,
            unit="milliseconds", domain=FieldDomain(minimum=0))
    from .field_metadata import enrich_fields

    return FieldRegistry(enrich_fields(fields))
