from copy import deepcopy

from .gpu import GPU_SCREENING_DEFAULTS
from .prefilter import PREFILTER_DEFAULTS
from .optimize_bounds import get_optimize_bounds_defaults
from .bot_defaults import get_common_bot_defaults
from .strategy_spec import get_strategy_spec, get_supported_strategy_kinds

CONFIG_SCHEMA_VERSION = "v9.0.0"
SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS = frozenset()
DEFAULT_EXAMPLE_CONFIG_PATH = "configs/examples/default_trailing_martingale_long.json"
STARTER_LIVE_USER = "passivbot_default"
# A symbol suspension is temporary policy, not an indefinite timestamp. This
# generous bound also keeps hours-to-milliseconds conversion finite and well
# within the integer ranges used by persisted/structured timestamp consumers.
MAX_EXCHANGE_SYMBOL_UNAVAILABLE_COOLDOWN_HOURS = 24.0 * 365.25 * 100.0


def _side_bot_prototype():
    result = {"enabled": False, **get_common_bot_defaults()}
    # This internal shape is also used by explicit authored fixture builders.
    # Null approval retains the shared source; [] would be a concrete deny pin.
    result["universe"]["approved_coins"] = None
    return result


_template_cache = None


def get_expanded_config_prototype():
    """Return an owned template; only its process-constant construction is cached.

    Retain metadata objects so cache invalidation/reload cannot reuse their ids.
    A replaced Rust metadata snapshot rebuilds the template before returning it.
    """
    global _template_cache
    specs = tuple(get_strategy_spec(kind) for kind in get_supported_strategy_kinds())
    if (_template_cache is None or len(specs) != len(_template_cache[0])
            or any(a is not b for a, b in zip(specs, _template_cache[0]))):
        _template_cache = (specs, _build_template_config())
    return deepcopy(_template_cache[1])


def _build_template_config():
    return deepcopy(
        {
            "config_version": CONFIG_SCHEMA_VERSION,
            "backtest": {
                "acquisition_mode": "auto",
                "reducer": {"default": "mean"},
                "balance_sample_divider": 60,
                "base_dir": "backtests",
                "btc_collateral_target_ratio": 0.0,
                "max_usd_debt_to_wallet_value_ratio": None,
                "candle_interval_minutes": 1,
                "coin_sources": {},
                "compress_cache": True,
                "dynamic_wel_by_tradability": True,
                "organillo_csv": None,
                "organillo_sha256": None,
                "end_date": "now",
                "ew_half_life_days": 360.0,
                "exchanges": ["binance", "bybit"],
                "filter_by_min_effective_cost": None,
                "gap_tolerance_ohlcvs_minutes": 120,
                "hsl_detailed_report": False,
                "hlcvs_data_dir": None,
                "hlcvs_data_override_mode": "intersection",
                "equity_floor_fraction": 0.05,
                "maker_fee_override_fraction": 0.0004,
                "market_order_slippage_fraction": 0.0005,
                "limit_order_fill_buffer_fraction": 0.0005,
                "market_settings": {"overrides": {}, "overrides_by_exchange": {}},
                "market_settings_sources": {},
                "offline": False,
                "ohlcv_source_dir": None,
                "scenarios": [
                    {"label": "base"},
                    {
                        "label": "subset_1",
                        "coins": [
                            "AAVE", "APT", "ATOM", "BNB", "CC", "ENA", "GRAM",
                            "INJ", "LTC", "ONDO", "RENDER", "TAO", "XLM", "ZEC",
                        ],
                    },
                    {
                        "label": "subset_2",
                        "coins": [
                            "ADA", "ARB", "AVAX", "BTC", "DOGE", "ETH", "HBAR",
                            "KAS", "MORPHO", "POL", "SOL", "UNI", "XMR",
                        ],
                    },
                    {
                        "label": "subset_3",
                        "coins": [
                            "ALGO", "ASTER", "BCH", "BTW", "DOT", "FIL", "HYPE",
                            "LINK", "NEAR", "QNT", "SUI", "WLD", "XRP",
                        ],
                    },
                    {
                        "label": "cost_stress",
                        "overrides": {
                            "backtest.limit_order_fill_buffer_fraction": 0.001,
                            "backtest.maker_fee_override_fraction": 0.0006,
                        },
                    },
                ],
                "start_date": "2021-04-01",
                "starting_balance": 100000,
                "suite_enabled": False,
                "taker_fee_override_fraction": 0.00055,
                "volume_normalization": True,
            },
            "bot": {
                "portfolio": {
                    "strategy_kind": "trailing_martingale",
                    "hedge_mode": False,
                    "execution": {
                        "market_orders_allowed": False,
                        "market_order_near_touch_fraction": 0.001,
                    },
                    "warmup": {"activation_span_fraction": 0.3, "max_minutes": None},
                    "hsl": {"scope": "coin"},
                    "loss_gate": {
                        "max_drawdown_fraction": 1.0,
                        "memory_ema_span_minutes": 43200.0,
                    },
                    "unstuck": {"memory_ema_span_minutes": 85511.137},
                },
                "global": get_common_bot_defaults(),
                "long": _side_bot_prototype(),
                "short": _side_bot_prototype(),
            },
            "coin_overrides": {},
            "live": {
                "account": {
                    "balance_hysteresis_snap_fraction": 0.01,
                    "balance_override": None,
                    "leverage": 10,
                    "margin_mode_preference": "cross",
                    # Preserve account selection for existing authored configs.
                    "user": "bybit_01",
                },
                "api": {
                    "custom_endpoints_path": None,
                    "market_snapshot_ticker_strategy": "auto",
                    "max_concurrent_api_requests": None,
                    "recv_window_milliseconds": 5000,
                },
                "candles": {
                    "candle_lock_timeout_seconds": 10,
                    "enable_archive_candle_fetch": False,
                    "enable_forager_ws_candles": True,
                    "forager_ws_candle_rest_audit_minutes": 30,
                    "inactive_coin_candle_ttl_minutes": 10,
                    "max_active_candle_tail_gap_minutes": 10,
                    "max_disk_candles_per_symbol_per_tf": 2000000,
                    "max_forager_candle_refresh_seconds": 45,
                    "max_forager_candle_staleness_minutes": None,
                    "max_memory_candles_per_symbol": 200000,
                    "max_ohlcv_fetches_per_minute": 30,
                },
                "startup": {
                    "defer_broad_candle_warmup": True,
                    "force_cold_startup": False,
                    "risk_bootstrap_max_lookback_minutes": 129600.0,
                    "startup_phase_budgets": {},
                    "warmup_concurrency": "auto",
                    "warmup_jitter_seconds": 30,
                },
                "fills": {
                    "fee_conversion_max_age_milliseconds": 86400000,
                    "fallback_fee_fraction": 0.0002,
                    "max_abs_fee_fraction": 0.001,
                    "fills_confirmation_overlap_minutes": 60,
                    "fills_recent_overlap_minutes": 10,
                },
                "execution": {
                    "execution_delay_seconds": 2,
                    "limit_order_create_max_market_distance_fraction": 0.8,
                    "max_n_cancellations_per_batch": 5,
                    "max_n_creations_per_batch": 3,
                    "order_match_tolerance_fraction": 0.0002,
                    "order_replacement_churn_gate_activation_count": 10,
                    "order_replacement_churn_gate_market_distance_fraction": 0.005,
                    "order_replacement_churn_gate_stability_minutes": 2.0,
                    "order_replacement_churn_gate_window_minutes": 10.0,
                    "time_in_force": "good_till_cancelled",
                },
                "recovery": {
                    "exchange_symbol_unavailable_cooldown_hours": 6.0,
                    "max_n_restarts_per_day": 10,
                },
                "modes": {
                    "disapproved_position": "graceful_stop",
                    "long": None,
                    "short": None,
                },
            },
            "logging": {
                "backup_count": 5,
                "dir": "logs",
                "level": 1,
                "live_event_debug_profiles": [],
                "max_bytes_mib": 10.0,
                "memory_snapshot_interval_minutes": 30,
                "persist_to_file": True,
                "rotation": True,
                "volume_refresh_info_threshold_seconds": 30,
            },
            "monitor": {
                "checkpoint_interval_minutes": 10.0,
                "compress_rotated_segments": True,
                "emit_completed_candles": True,
                "enabled": True,
                "event_rotation_mib": 128.0,
                "event_rotation_minutes": 60.0,
                "include_raw_fill_payloads": False,
                "max_total_bytes": 1073741824,
                "price_tick_min_interval_milliseconds": 500,
                "retain_candles": True,
                "retain_days": 7.0,
                "retain_fills": True,
                "retain_price_ticks": True,
                "root_dir": "monitor",
                "snapshot_interval_seconds": 1.0,
            },
            "optimize": {
                "bounds": get_optimize_bounds_defaults(),
                **{
                    "backend": "pymoo",
                    "compress_results_file": True,
                    "enable_overrides": [],
                    "prefilter": deepcopy(PREFILTER_DEFAULTS),
                    "fixed_params": [],
                    "fixed_runtime_overrides": {},
                    "iters": 2048,
                    "n_cpus": 4,
                    "objective_scenario": None,
                    "pareto_max_size": 1000,
                    "population_size": 128,
                    "seed": None,
                    "gpu": {
                        "tuning_mode": "auto",
                        "batch_size": None,
                        "max_dispatch_candidate_bars": None,
                        "checkpoint_interval_seconds": 5.0,
                        "screening": deepcopy(GPU_SCREENING_DEFAULTS),
                    },
                    "pymoo": {
                        "algorithm": "auto",
                        "algorithms": {
                            "nsga2": {},
                            "nsga3": {
                                "ref_dirs": {
                                    "method": "das_dennis",
                                    "n_partitions": "auto",
                                }
                            },
                        },
                        "shared": {
                            "crossover_distribution_index": 20.0,
                            "crossover_probability_per_variable": 0.5,
                            "eliminate_duplicates": True,
                            "mutation_distribution_index": 20.0,
                            "mutation_probability": "auto",
                            "mutation_probability_per_variable": "auto",
                        },
                    },
                    "round_to_n_significant_digits": 8,
                    "write_all_results": True,
                    "limits": [
                        {
                            "enabled": True,
                            "metric": "backtest_completion_fraction",
                            "penalize_if": "less_than",
                            "reducer": "min",
                            "value": 1.0,
                        },
                        {
                            "enabled": False,
                            "metric": "drawdown_max_fraction_strategy_equity",
                            "penalize_if": "greater_than",
                            "reducer": "max",
                            "value": 0.50,
                        },
                        {
                            "enabled": False,
                            "metric": "peak_recovery_max_days_strategy_equity",
                            "penalize_if": "greater_than",
                            "reducer": "max",
                            "value": 240,
                        },
                    ],
                    "scoring": [
                        {
                            "goal": "max",
                            "metric": "cagr_fraction_strategy_equity_ew",
                            "reducer": "mean",
                            "scenario": None,
                        },
                        {
                            "goal": "min",
                            "metric": "ulcer_index_fraction_strategy_equity",
                            "reducer": "max",
                            "scenario": None,
                        },
                        {
                            "goal": "min",
                            "metric": "peak_recovery_max_days_strategy_equity",
                            "reducer": "max",
                            "scenario": None,
                        },
                    ],
                },
            },
        }
    )


def get_template_config():
    """Canonical authored defaults: one common policy and sparse side pins.

    Expanded runtime shapes are a separate internal prototype. In particular,
    default false activation is not an explicit side pin in this source view.
    An omitted optimizer bound fixes its effective value rather than opening a
    recommended search range.
    """
    result = get_expanded_config_prototype()
    kind = result["bot"]["portfolio"]["strategy_kind"]
    result["bot"]["global"]["strategy"] = {kind: result["bot"]["global"]["strategy"][kind]}
    result["bot"]["long"] = {}
    result["bot"]["short"] = {}
    result["optimize"]["bounds"] = {}
    return result


def get_starter_config():
    """Return the runnable authored profile for commands without a config path."""
    result = get_template_config()
    result["bot"]["long"]["enabled"] = True
    result["bot"]["short"]["enabled"] = False
    result["bot"]["global"]["universe"]["approved_coins"] = [
        "AAVE", "ADA", "ALGO", "APT", "ARB", "ASTER", "ATOM", "AVAX", "BCH", "BNB",
        "BTC", "BTW", "CC", "DOGE", "DOT", "ENA", "ETH", "FIL", "GRAM", "HBAR",
        "HYPE", "INJ", "KAS", "LINK", "LTC", "MORPHO", "NEAR", "ONDO", "POL", "QNT",
        "RENDER", "SOL", "SUI", "TAO", "UNI", "WLD", "XLM", "XMR", "XRP", "ZEC",
    ]
    result["live"]["account"]["user"] = STARTER_LIVE_USER
    kind = result["bot"]["portfolio"]["strategy_kind"]
    common_bounds = get_optimize_bounds_defaults()["long"]
    common_bounds["strategy"] = {kind: common_bounds["strategy"][kind]}
    hsl_memory = result["bot"]["global"]["hsl"]["memory_ema_span_minutes"]
    common_bounds["hsl"]["memory_ema_span_minutes"] = [hsl_memory, hsl_memory]
    result["optimize"]["bounds"] = {
        "global": common_bounds,
        "portfolio": {"unstuck": {"memory_ema_span_minutes": [10080, 129600]}},
    }
    return result
