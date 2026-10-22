"""Explicit common starter policy, independent of authored side presence.

The numeric profile retains the former long defaults. Both position sides have
the same fallback; activation is a separate explicit permission. Released-v8
conversion owns its frozen asymmetric defaults and does not consult this file.
"""

def get_common_bot_defaults():
    from .strategy_spec import get_common_strategy_defaults

    return {
        "enabled": False,
        "universe": {
            "approved_coins": [],
            "ignored_coins": [],
            "filter_by_min_effective_cost": True,
            "minimum_coin_age_days": 60.0,
        },
        "forager": {
            "always_include": [],
            "score_hysteresis_margin": 0.02,
            "score_weights": {
                "unilateralness": 0.0,
                "ema_readiness": 0.21,
                "volatility": 0.61,
                "volume": 0.18,
            },
            "unilateralness_ema_span_minutes": 60.0,
            "volatility_ema_span_minutes": 2274.0,
            "volume_ema_span_minutes": 310.0,
        },
        "hsl": {
            "memory_ema_span_minutes": 43200.0,
            "cooldown_after_red_minutes": 2160.0,
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
            "max_positions": 7,
            "total_wallet_exposure_limit": 1.5,
            "we_excess_allowance_fraction": 0.37,
            "position_exposure_enforcer": {"enabled": False, "threshold_ratio": 1.0},
            "total_exposure_enforcer": {
                "enabled": False,
                "policy": "reduce_overweight",
                "threshold_ratio": 1.0,
            },
        },
        "unstuck": {
            "close_fraction": 0.041,
            "ema_distance_fraction": -0.0269,
            "ema_gating_enabled": True,
            "ema_span_0_minutes": 790.0,
            "ema_span_1_minutes": 1080.0,
            "enabled": True,
            "loss_allowance_fraction": 0.0052,
            "exposure_trigger_ratio": 0.466,
        },
        "strategy": get_common_strategy_defaults(),
    }
