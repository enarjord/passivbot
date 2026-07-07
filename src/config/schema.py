from copy import deepcopy

from .optimize_bounds import get_optimize_bounds_defaults
from .strategy import get_all_strategy_defaults


CONFIG_SCHEMA_VERSION = "v8.0.0"
DEFAULT_EXAMPLE_CONFIG_PATH = "configs/examples/default_trailing_martingale_long.json"


def _get_shared_bot_defaults():
    return {
        "long": {
            "forager": {
                "score_weights": {
                    "ema_readiness": 0.08,
                    "volatility": 0.48,
                    "volume": 0.44
                },
                "volatility_ema_span_1m": 661.0,
                "volume_drop_pct": 0.06,
                "volume_ema_span_1m": 2369.0
            },
            "hsl": {
                "cooldown_minutes_after_red": 3981.0,
                "ema_span_minutes": 644.0,
                "enabled": False,
                "no_restart_drawdown_threshold": 1,
                "orange_tier_mode": "tp_only_with_active_entry_cancellation",
                "panic_close_order_type": "limit",
                "red_threshold": 0.025,
                "restart_after_red_policy": "threshold",
                "tier_ratios": {
                    "orange": 0.75,
                    "yellow": 0.5
                }
            },
            "risk": {
                "entry_cooldown_minutes": 7.1,
                "n_positions": 5.0,
                "position_exposure_enforcer_enabled": False,
                "position_exposure_enforcer_threshold": 1.0,
                "total_exposure_enforcer_enabled": True,
                "total_exposure_enforcer_policy": "reduce_overweight",
                "total_exposure_enforcer_threshold": 1.004,
                "total_exposure_entry_gate_enabled": True,
                "total_wallet_exposure_limit": 1.5,
                "we_excess_allowance_mode": "bounded",
                "we_excess_allowance_pct": 0.5
            },
            "unstuck": {
                "close_pct": 0.028,
                "ema_dist": -0.086,
                "ema_gating_enabled": True,
                "enabled": True,
                "loss_allowance_pct": 0.0065,
                "threshold": 0.46
            }
        },
        "short": {
            "forager": {
                "score_weights": {
                    "ema_readiness": 0.0,
                    "volatility": 0.0,
                    "volume": 0.0
                },
                "volatility_ema_span_1m": 10.0,
                "volume_drop_pct": 0.02,
                "volume_ema_span_1m": 60.0
            },
            "hsl": {
                "cooldown_minutes_after_red": 1.0,
                "ema_span_minutes": 1.0,
                "enabled": False,
                "no_restart_drawdown_threshold": 1,
                "orange_tier_mode": "tp_only_with_active_entry_cancellation",
                "panic_close_order_type": "limit",
                "red_threshold": 0.01,
                "restart_after_red_policy": "threshold",
                "tier_ratios": {
                    "orange": 0.75,
                    "yellow": 0.5
                }
            },
            "risk": {
                "entry_cooldown_minutes": 0.0,
                "n_positions": 1.0,
                "position_exposure_enforcer_enabled": True,
                "position_exposure_enforcer_threshold": 0.8,
                "total_exposure_enforcer_enabled": True,
                "total_exposure_enforcer_policy": "reduce_overweight",
                "total_exposure_enforcer_threshold": 0.8,
                "total_exposure_entry_gate_enabled": True,
                "total_wallet_exposure_limit": 0.0,
                "we_excess_allowance_mode": "bounded",
                "we_excess_allowance_pct": 0.0
            },
            "unstuck": {
                "close_pct": 0.01,
                "ema_dist": -0.2,
                "ema_gating_enabled": True,
                "enabled": True,
                "loss_allowance_pct": 0.005,
                "threshold": 0.3
            }
        }
    }


def get_template_config():
    strategy_defaults = get_all_strategy_defaults()
    return deepcopy(
        {
            "config_version": CONFIG_SCHEMA_VERSION,
            "backtest": {
                "aggregate": {
                    "default": "mean"
                },
                "balance_sample_divider": 60,
                "base_dir": "backtests",
                "btc_collateral_cap": 0.0,
                "btc_collateral_ltv_cap": None,
                "candle_interval_minutes": 1,
                "coin_sources": {},
                "compress_cache": True,
                "dynamic_wel_by_tradability": True,
                "end_date": "now",
                "exchanges": [
                    "binance",
                    "bybit"
                ],
                "filter_by_min_effective_cost": False,
                "gap_tolerance_ohlcvs_minutes": 120,
                "hlcvs_data_dir": None,
                "hlcvs_data_override_mode": "intersection",
                "liquidation_threshold": 0.05,
                "maker_fee_override": 0.0004,
                "market_order_slippage_pct": 0.0005,
                "market_settings": {
                    "overrides": {},
                    "overrides_by_exchange": {}
                },
                "market_settings_sources": {},
                "ohlcv_source_dir": None,
                "scenarios": [
                    {
                        "label": "base"
                    },
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
                            "LINK"
                        ]
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
                            "DOT"
                        ]
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
                            "INJ"
                        ]
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
                            "ARB"
                        ]
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
                            "INJ"
                        ]
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
                            "APT"
                        ]
                    }
                ],
                "start_date": "2021-04-20",
                "starting_balance": 100000,
                "suite_enabled": False,
                "taker_fee_override": 0.00055,
                "visible_metrics": [
                    "adg_strategy_eq",
                    "adg_strategy_eq_w",
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
                    "hard_stop_panic_close_loss_drawdown_pct_mean"
                ],
                "volume_normalization": True
            },
            "bot": {
                "long": {
                    **deepcopy(_get_shared_bot_defaults()["long"]),
                    "strategy": strategy_defaults["long"],
                },
                "short": {
                    **deepcopy(_get_shared_bot_defaults()["short"]),
                    "strategy": strategy_defaults["short"],
                },
            },
            "coin_overrides": {},
            "live": {
                "approved_coins": {
                    "long": [
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
                        "INJ"
                    ],
                    "short": [
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
                        "INJ"
                    ]
                },
                "auto_gs": True,
                "balance_hysteresis_snap_pct": 0.01,
                "balance_override": None,
                "candle_lock_timeout_seconds": 10,
                "custom_endpoints_path": None,
                "defer_broad_candle_warmup": True,
                "enable_archive_candle_fetch": False,
                "execution_delay_seconds": 2,
                "fee_conversion_max_age_ms": 86400000,
                "fee_pct_fallback": 0.0002,
                "fee_pct_sanity_abs_max": 0.001,
                "fills_confirmation_overlap_minutes": 60,
                "fills_recent_overlap_minutes": 10,
                "filter_by_min_effective_cost": True,
                "forager_score_hysteresis_pct": 0.02,
                "force_cold_startup": False,
                "forced_mode_long": "",
                "forced_mode_short": "",
                "hedge_mode": False,
                "hsl_accept_incomplete_history": False,
                "hsl_position_during_cooldown_policy": "panic",
                "hsl_signal_mode": "coin",
                "ignored_coins": {
                    "long": [],
                    "short": []
                },
                "inactive_coin_candle_ttl_minutes": 10,
                "initial_entry_exec_max_market_dist_pct": 0.005,
                "leverage": 10,
                "limit_order_create_max_market_dist_pct": 0.8,
                "margin_mode_preference": "cross",
                "market_order_near_touch_threshold": 0.001,
                "market_orders_allowed": False,
                "market_snapshot_ticker_strategy": "auto",
                "max_active_candle_tail_gap_minutes": 10,
                "max_concurrent_api_requests": None,
                "max_disk_candles_per_symbol_per_tf": 2000000,
                "max_forager_candle_refresh_seconds": 45,
                "max_forager_candle_staleness_minutes": None,
                "max_memory_candles_per_symbol": 200000,
                "max_n_cancellations_per_batch": 5,
                "max_n_creations_per_batch": 3,
                "max_n_restarts_per_day": 10,
                "max_ohlcv_fetches_per_minute": 30,
                "max_realized_loss_pct": 1,
                "max_warmup_minutes": 0,
                "minimum_coin_age_days": 60,
                "order_match_tolerance_pct": 0.0002,
                "pnls_max_lookback_days": 30.0,
                "recv_window_ms": 5000,
                "strategy_kind": "trailing_martingale",
                "time_in_force": "good_till_cancelled",
                "user": "bybit_01",
                "warmup_concurrency": 0,
                "warmup_jitter_seconds": 30,
                "warmup_ratio": 0.3
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
                "volume_refresh_info_threshold_seconds": 30
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
                "snapshot_interval_seconds": 1.0
            },
            "optimize": {
                "bounds": get_optimize_bounds_defaults(),
                **{
                    "backend": "pymoo",
                    "compress_results_file": True,
                    "crossover_eta": 20,
                    "crossover_probability": 0.64,
                    "enable_overrides": [],
                    "fixed_params": [],
                    "fixed_runtime_overrides": {
                        "bot.long.hsl.no_restart_drawdown_threshold": 1,
                        "bot.long.hsl.restart_after_red_policy": "threshold",
                        "bot.short.hsl.no_restart_drawdown_threshold": 1,
                        "bot.short.hsl.restart_after_red_policy": "threshold"
                    },
                    "iters": 200000,
                    "mutation_eta": 20,
                    "mutation_indpb": 0.0135135135,
                    "mutation_probability": 0.34,
                    "n_cpus": 4,
                    "offspring_multiplier": 1,
                    "pareto_max_size": 1000,
                    "population_size": None,
                    "seed": None,
                    "pymoo": {
                        "algorithm": "auto",
                        "algorithms": {
                            "nsga2": {},
                            "nsga3": {
                                "ref_dirs": {
                                    "method": "das_dennis",
                                    "n_partitions": "auto"
                                }
                            }
                        },
                        "shared": {
                            "crossover_eta": 20.0,
                            "crossover_prob_var": 0.5,
                            "eliminate_duplicates": True,
                            "mutation_eta": 20.0,
                            "mutation_prob_var": "auto"
                        }
                    },
                    "round_to_n_significant_digits": 3,
                    "write_all_results": True,
                    "limits": [
                        {
                            "enabled": True,
                            "metric": "drawdown_worst_strategy_eq",
                            "penalize_if": "greater_than",
                            "stat": "mean",
                            "value": 0.8
                        },
                        {
                            "enabled": True,
                            "metric": "backtest_completion_ratio",
                            "penalize_if": "less_than",
                            "value": 0.99
                        }
                    ],
                    "scoring": [
                        {
                            "goal": "max",
                            "metric": "adg_strategy_eq"
                        },
                        {
                            "goal": "max",
                            "metric": "adg_strategy_eq_w"
                        },
                        {
                            "goal": "max",
                            "metric": "mdg_strategy_eq"
                        },
                        {
                            "goal": "max",
                            "metric": "sortino_ratio_strategy_eq"
                        },
                        {
                            "goal": "max",
                            "metric": "volume_pct_per_day_avg"
                        },
                        {
                            "goal": "max",
                            "metric": "sharpe_ratio_strategy_eq"
                        },
                        {
                            "goal": "min",
                            "metric": "strategy_eq_recovery_days_max"
                        },
                        {
                            "goal": "min",
                            "metric": "position_held_days_max"
                        },
                        {
                            "goal": "min",
                            "metric": "drawdown_worst_strategy_eq"
                        },
                        {
                            "goal": "min",
                            "metric": "loss_profit_ratio"
                        },
                        {
                            "goal": "min",
                            "metric": "strategy_eq_underwater_pct_mean"
                        }
                    ]
                },
            },
        }
    )
