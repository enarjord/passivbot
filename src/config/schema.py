from copy import deepcopy

from .gpu import GPU_SCREENING_DEFAULTS
from .prefilter import PREFILTER_DEFAULTS
from .optimize_bounds import get_optimize_bounds_defaults
from .strategy import get_all_strategy_defaults
from .strategy_spec import get_strategy_spec, get_supported_strategy_kinds

CONFIG_SCHEMA_VERSION = "v9.0.0"
SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS = frozenset()
DEFAULT_EXAMPLE_CONFIG_PATH = "configs/examples/default_trailing_martingale_long.json"
# A symbol suspension is temporary policy, not an indefinite timestamp. This
# generous bound also keeps hours-to-milliseconds conversion finite and well
# within the integer ranges used by persisted/structured timestamp consumers.
MAX_EXCHANGE_SYMBOL_UNAVAILABLE_COOLDOWN_HOURS = 24.0 * 365.25 * 100.0


DEFAULT_APPROVED_COINS = [
    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "SOL",
    "TRX",
    "HYPE",
    "DOGE",
    "ZEC",
    "XLM",
    "ADA",
    "LINK",
    "XMR",
    "CC",
    "BCH",
    "TON",
    "GRAM",
    "LTC",
    "HBAR",
    "SUI",
    "AVAX",
    "NEAR",
    "UNI",
    "TAO",
    "ASTER",
    "ONDO",
    "AAVE",
    "DOT",
    "WLD",
    "MNT",
    "MORPHO",
    "POL",
    "ATOM",
    "RENDER",
    "KAS",
    "ENA",
    "ALGO",
    "FIL",
    "ARB",
    "APT",
    "INJ"
]


def _get_shared_bot_defaults():
    return {
        "long": {
            "forager": {
                "always_include": [],
                "score_weights": {
                    "unilateralness": 0.0,
                    "ema_readiness": 0.21,
                    "volatility": 0.61,
                    "volume": 0.18,
                },
                "unilateralness_ema_span_1m": 60.0,
                "volatility_ema_span_1m": 2274.0,
                "volume_ema_span_1m": 310.0,
            },
            "hsl": {
                "memory_ema_span_minutes": 43200.0,
                "cooldown_minutes_after_red": 2160.0,
                "drawdown_ratio_ema_span_minutes": 720.0,
                "enabled": False,
                "scale_budget_with_excess_allowance": False,
                "panic_close_order_type": "limit",
                "drawdown_ratio_threshold": 0.15,
            },
            "entry_cooldown": {
                "base_duration_minutes": 24.1,
                "min_duration_minutes": 0.0,
                "max_duration_minutes": None,
                "weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 0.0},
            },
            "risk": {
                "max_positions": 7.0,
                "total_wallet_exposure_limit": 1.5,
                "we_excess_allowance_pct": 0.37,
                "position_exposure_enforcer": {
                    "enabled": False,
                    'threshold_ratio': 1.0,
                },
                "total_exposure_enforcer": {
                    "enabled": False,
                    "policy": "reduce_overweight",
                    'threshold_ratio': 1.0,
                },
            },
            "unstuck": {
                "close_pct": 0.041,
                "ema_distance_fraction": -0.0269,
                "ema_gating_enabled": True,
                "ema_span_0": 790.0,
                "ema_span_1": 1080.0,
                "enabled": True,
                "loss_allowance_pct": 0.0052,
                "exposure_trigger_ratio": 0.466,
            },
        },
        "short": {
            "forager": {
                "always_include": [],
                "score_weights": {
                    "unilateralness": 0.0,
                    "ema_readiness": 0.0,
                    "volatility": 0.0,
                    "volume": 0.0,
                },
                "unilateralness_ema_span_1m": 60.0,
                "volatility_ema_span_1m": 10.0,
                "volume_ema_span_1m": 60.0,
            },
            "hsl": {
                "memory_ema_span_minutes": 43200.0,
                "cooldown_minutes_after_red": 1.0,
                "drawdown_ratio_ema_span_minutes": 1.0,
                "enabled": False,
                "scale_budget_with_excess_allowance": False,
                "panic_close_order_type": "limit",
                "drawdown_ratio_threshold": 0.01,
            },
            "entry_cooldown": {
                "base_duration_minutes": 0.0,
                "min_duration_minutes": 0.0,
                "max_duration_minutes": None,
                "weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 0.0},
            },
            "risk": {
                "max_positions": 1.0,
                "total_wallet_exposure_limit": 0.0,
                "we_excess_allowance_pct": 0.0,
                "position_exposure_enforcer": {
                    "enabled": True,
                    'threshold_ratio': 0.8,
                },
                "total_exposure_enforcer": {
                    "enabled": True,
                    "policy": "reduce_overweight",
                    'threshold_ratio': 0.8,
                },
            },
            "unstuck": {
                "close_pct": 0.01,
                "ema_distance_fraction": -0.2,
                "ema_gating_enabled": True,
                "ema_span_0": 60.0,
                "ema_span_1": 60.0,
                "enabled": True,
                "loss_allowance_pct": 0.005,
                "exposure_trigger_ratio": 0.3,
            },
        },
    }


_template_cache = None


def get_template_config():
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
    strategy_defaults = get_all_strategy_defaults()
    return deepcopy(
        {
            "config_version": CONFIG_SCHEMA_VERSION,
            "backtest": {
                "reducer": {"default": "mean"},
                "balance_sample_divider": 60,
                "base_dir": "backtests",
                "btc_collateral_cap": 0.0,
                "btc_collateral_ltv_cap": None,
                "candle_interval_minutes": 1,
                "coin_sources": {},
                "compress_cache": True,
                "dynamic_wel_by_tradability": True,
                "organillo_csv": None,
                "organillo_sha256": None,
                "end_date": "now",
                "ew_half_life_days": None,
                "exchanges": ["binance", "bybit"],
                "filter_by_min_effective_cost": None,
                "gap_tolerance_ohlcvs_minutes": 120,
                "hsl_detailed_report": False,
                "hlcvs_data_dir": None,
                "hlcvs_data_override_mode": "intersection",
                "equity_floor_ratio": 0.05,
                "maker_fee_override": 0.0004,
                "market_order_slippage_pct": 0.0005,
                "limit_order_fill_buffer_pct": 0.0,
                "market_settings": {"overrides": {}, "overrides_by_exchange": {}},
                "market_settings_sources": {},
                "offline": False,
                "ohlcv_source_dir": None,
                "scenarios": [
                    {"label": "base"},
                    {
                        "label": "subset_1_top",
                        "coins": [
                            "BTC",
                            "ETH",
                            "BNB",
                            "XRP",
                            "SOL",
                            "TRX",
                            "HYPE",
                            "DOGE",
                            "XLM",
                            "XMR",
                            "ADA",
                            "CC",
                            "ZEC",
                            "LINK",
                        ],
                    },
                    {
                        "label": "subset_2_mid1",
                        "coins": [
                            "BCH",
                            "GRAM",
                            "HBAR",
                            "LTC",
                            "AVAX",
                            "SUI",
                            "CRO",
                            "NEAR",
                            "TAO",
                            "MNT",
                            "ASTER",
                            "WLD",
                            "DOT",
                        ],
                    },
                    {
                        "label": "subset_3_bottom",
                        "coins": [
                            "ONDO",
                            "UNI",
                            "MORPHO",
                            "AAVE",
                            "ATOM",
                            "RENDER",
                            "KAS",
                            "ALGO",
                            "POL",
                            "ENA",
                            "FIL",
                            "APT",
                            "ARB",
                            "INJ",
                        ],
                    },
                    {
                        "label": "subset_4_mix",
                        "coins": [
                            "BTC",
                            "XRP",
                            "HYPE",
                            "XMR",
                            "ZEC",
                            "GRAM",
                            "AVAX",
                            "NEAR",
                            "ASTER",
                            "ONDO",
                            "AAVE",
                            "KAS",
                            "ENA",
                            "ARB",
                        ],
                    },
                    {
                        "label": "subset_5_mix",
                        "coins": [
                            "ETH",
                            "SOL",
                            "DOGE",
                            "ADA",
                            "LINK",
                            "HBAR",
                            "SUI",
                            "TAO",
                            "WLD",
                            "UNI",
                            "ATOM",
                            "ALGO",
                            "FIL",
                            "INJ",
                        ],
                    },
                    {
                        "label": "subset_6_mix",
                        "coins": [
                            "BNB",
                            "TRX",
                            "XLM",
                            "CC",
                            "BCH",
                            "LTC",
                            "CRO",
                            "MNT",
                            "DOT",
                            "MORPHO",
                            "RENDER",
                            "POL",
                            "APT",
                        ],
                    },
                ],
                "start_date": "2021-04-20",
                "starting_balance": 100000,
                "suite_enabled": False,
                "taker_fee_override": 0.00055,
                "visible_metrics": [
                    "adg_strategy_eq",
                    "mdg_strategy_eq",
                    "strategy_eq_recovery_days_max",
                    "position_held_days_max",
                    "drawdown_worst_strategy_eq",
                    "volume_pct_per_day_avg",
                    "sortino_ratio_strategy_eq",
                    "sharpe_ratio_strategy_eq",
                    "loss_profit_ratio",
                    "strategy_eq_underwater_pct_mean",
                    "hard_stop_restarts_per_year",
                    "hard_stop_panic_close_loss_drawdown_pct_mean",
                ],
                "volume_normalization": True,
            },
            "bot": {
                "global": {
                    "strategy_kind": "trailing_martingale",
                    "hedge_mode": False,
                    "universe": {
                        "minimum_coin_age_days": 60.0,
                        "filter_by_min_effective_cost": True,
                    },
                    "execution": {
                        "market_orders_allowed": False,
                        "market_order_near_touch_fraction": 0.001,
                    },
                    "forager": {"score_hysteresis_pct": 0.02},
                    "warmup": {"span_multiplier": 0.3, "max_minutes": None},
                    "hsl": {"scope": "coin"},
                    "loss_gate": {
                        "max_drawdown_ratio": 1.0,
                        "memory_ema_span_minutes": 43200.0,
                    },
                    "unstuck": {"memory_ema_span_minutes": 43200.0},
                },
                "long": {
                    **deepcopy(_get_shared_bot_defaults()["long"]),
                    "strategy": strategy_defaults["long"],
                    "universe": {
                        "approved_coins": list(DEFAULT_APPROVED_COINS),
                        "ignored_coins": [],
                    },
                },
                "short": {
                    **deepcopy(_get_shared_bot_defaults()["short"]),
                    "strategy": strategy_defaults["short"],
                    "universe": {
                        "approved_coins": list(DEFAULT_APPROVED_COINS),
                        "ignored_coins": [],
                    },
                },
            },
            "coin_overrides": {},
            "live": {
                "account": {
                    "balance_hysteresis_snap_pct": 0.01,
                    "balance_override": None,
                    "leverage": 10,
                    "margin_mode_preference": "cross",
                    "user": "bybit_01",
                },
                "api": {
                    "custom_endpoints_path": None,
                    "market_snapshot_ticker_strategy": "auto",
                    "max_concurrent_api_requests": None,
                    "recv_window_ms": 5000,
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
                    "fee_conversion_max_age_ms": 86400000,
                    "fee_pct_fallback": 0.0002,
                    "fee_pct_sanity_abs_max": 0.001,
                    "fills_confirmation_overlap_minutes": 60,
                    "fills_recent_overlap_minutes": 10,
                },
                "execution": {
                    "execution_delay_seconds": 2,
                    "limit_order_create_max_market_dist_pct": 0.8,
                    "max_n_cancellations_per_batch": 5,
                    "max_n_creations_per_batch": 3,
                    "order_match_tolerance_pct": 0.0002,
                    "order_replacement_churn_gate_activation_count": 10,
                    "order_replacement_churn_gate_market_dist_pct": 0.005,
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
                "max_bytes_mb": 10.0,
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
                "event_rotation_mb": 128.0,
                "event_rotation_minutes": 60.0,
                "include_raw_fill_payloads": False,
                "max_total_bytes": 1073741824,
                "price_tick_min_interval_ms": 500,
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
                    "iters": 200000,
                    "n_cpus": 4,
                    "objective_scenario": None,
                    "pareto_max_size": 1000,
                    "population_size": None,
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
                            "crossover_eta": 20.0,
                            "crossover_prob_var": 0.5,
                            "eliminate_duplicates": True,
                            "mutation_eta": 20.0,
                            "mutation_prob": "auto",
                            "mutation_prob_per_variable": "auto",
                        },
                    },
                    "round_to_n_significant_digits": 3,
                    "write_all_results": True,
                    "limits": [
                        {
                            "enabled": True,
                            "metric": "drawdown_worst_strategy_eq",
                            "penalize_if": "greater_than",
                            "reducer": "mean",
                            "value": 0.8,
                        },
                        {
                            "enabled": True,
                            "metric": "backtest_completion_ratio",
                            "penalize_if": "less_than",
                            "value": 0.99,
                        },
                    ],
                    "scoring": [
                        {"goal": "max", "metric": "adg_strategy_eq"},
                        {"goal": "max", "metric": "mdg_strategy_eq"},
                        {"goal": "max", "metric": "sortino_ratio_strategy_eq"},
                        {"goal": "max", "metric": "volume_pct_per_day_avg"},
                        {"goal": "max", "metric": "sharpe_ratio_strategy_eq"},
                        {"goal": "min", "metric": "strategy_eq_recovery_days_max"},
                        {"goal": "min", "metric": "position_held_days_max"},
                        {"goal": "min", "metric": "drawdown_worst_strategy_eq"},
                        {"goal": "min", "metric": "loss_profit_ratio"},
                        {"goal": "min", "metric": "strategy_eq_underwater_pct_mean"},
                    ],
                },
            },
        }
    )
