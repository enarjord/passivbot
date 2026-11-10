"""Explicit common starter policy, independent of authored side presence.

Both position sides inherit the starter values. Activation is a separate
explicit permission. Released-v8
conversion owns its frozen asymmetric defaults and does not consult this file.
"""

def get_common_bot_defaults():
    from .strategy_spec import get_common_strategy_defaults

    return {
        "universe": {
            "approved_coins": [],
            "ignored_coins": [],
            "filter_by_min_effective_cost": True,
            "minimum_coin_age_days": 60.0,
        },
        "forager": {
            "always_include": [],
            "score_hysteresis_margin": 0.02,
            # Unit-sum weights as used by the effective bot configuration.
            "score_weights": {
                "rms_directionality": 0.39918534095105473,
                "ema_readiness": 0.1978654292597687,
                "log_range": 0.362967082971877,
                "volume": 0.03998214681729961,
            },
            "rms_directionality_ema_span_minutes": 2169.1726,
            "log_range_ema_span_minutes": 1747.2975,
            "volume_ema_span_minutes": 518.62365,
        },
        "hsl": {
            "memory_ema_span_minutes": 43200.0,
            "cooldown_after_red_minutes": 2160.0,
            "drawdown_fraction_ema_span_minutes": 720.0,
            "enabled": False,
            "scale_budget_with_excess_allowance": False,
            "panic_close_order_type": "limit",
            "drawdown_threshold_fraction": 0.15,
        },
        "entry_cooldown": {
            "base_duration_minutes": 3.4098966,
            "min_duration_minutes": 1.0,
            "max_duration_minutes": 360.0,
            "weights_minutes": {
                "exposure_ratio": 16.628462,
                "adverse_directionality": 11.242806,
            },
        },
        "risk": {
            "max_positions": 10,
            "total_wallet_exposure_limit_ratio": 1.5,
            "we_excess_allowance_fraction": 0.39417859,
            "position_exposure_enforcer": {"enabled": False, "threshold_ratio": 1.0},
            "total_exposure_enforcer": {
                "enabled": False,
                "policy": "reduce_overweight",
                "threshold_ratio": 1.0,
            },
        },
        "unstuck": {
            "close_fraction": 0.11659791,
            "ema_distance_fraction": -0.049689319,
            "ema_gating_enabled": True,
            "ema_span_0_minutes": 513.39957,
            "ema_span_1_minutes": 2042.1798,
            "enabled": True,
            "loss_allowance_fraction": 0.0032770496,
            "exposure_trigger_ratio": 0.55288224,
        },
        "strategy": get_common_strategy_defaults(),
    }
