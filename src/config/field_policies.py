"""Dependency-light source declarations for shared field adapters.

These are the existing path and permission policies. Registry views reuse them;
metadata never adds optimizer genes or per-coin permissions.
"""

from .strategy_spec import BOT_POSITION_SIDES

SPECIAL_OPTIMIZER_KEY_PATHS = {
    "unstuck_memory_ema_span_minutes": ("bot", "portfolio", "unstuck", "memory_ema_span_minutes"),
    "loss_gate_memory_ema_span_minutes": ("bot", "portfolio", "loss_gate", "memory_ema_span_minutes"),
    "long_forager_volume_ema_span_1m": ("bot", "long", "forager", "volume_ema_span_minutes"),
    "long_forager_volatility_ema_span_1m": (
        "bot",
        "long",
        "forager",
        "volatility_ema_span_minutes",
    ),
    "long_forager_score_weights_volume": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "volume",
    ),
    "long_forager_score_weights_ema_readiness": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "ema_readiness",
    ),
    "long_forager_score_weights_volatility": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "volatility",
    ),
    "short_forager_score_weights_volume": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "volume",
    ),
    "short_forager_score_weights_ema_readiness": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "ema_readiness",
    ),
    "short_forager_score_weights_volatility": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "volatility",
    ),
    "short_forager_volume_ema_span_1m": (
        "bot",
        "short",
        "forager",
        "volume_ema_span_minutes",
    ),
    "short_forager_volatility_ema_span_1m": (
        "bot",
        "short",
        "forager",
        "volatility_ema_span_minutes",
    ),
}

for _side in BOT_POSITION_SIDES:
    for _key, _path in {
        "forager_score_weights_unilateralness": ("forager", "score_weights", "unilateralness"),
        "unilateralness_ema_span_1m": ("forager", "unilateralness_ema_span_minutes"),
        # The fixed ceiling may be null even when its search endpoints are numeric.
        "entry_cooldown_max_duration_minutes": ("entry_cooldown", "max_duration_minutes"),
        "entry_cooldown_weights_minutes_exposure_ratio": (
            "entry_cooldown",
            "weights_minutes",
            "exposure_ratio",
        ),
        "entry_cooldown_weights_minutes_adverse_directionality": (
            "entry_cooldown",
            "weights_minutes",
            "adverse_directionality",
        ),
    }.items():
        SPECIAL_OPTIMIZER_KEY_PATHS[f"{_side}_{_key}"] = ("bot", _side, *_path)


CONDITIONAL_HSL_OVERRIDE_PATHS = frozenset(
    {
        "hsl.cooldown_after_red_minutes",
        "hsl.drawdown_ratio_ema_span_minutes",
        "hsl.memory_ema_span_minutes",
        "hsl.scale_budget_with_excess_allowance",
        "hsl.enabled",
        "hsl.panic_close_order_type",
        "hsl.drawdown_ratio_threshold",
    }
)
OVERRIDABLE_SHARED_BOT_PATHS = frozenset(
    {
        "entry_cooldown.base_duration_minutes",
        "entry_cooldown.min_duration_minutes",
        "entry_cooldown.max_duration_minutes",
        "entry_cooldown.weights_minutes.exposure_ratio",
        "entry_cooldown.weights_minutes.adverse_directionality",
        "risk.position_exposure_enforcer.enabled",
        "risk.position_exposure_enforcer.threshold_ratio",
        "risk.we_excess_allowance_fraction",
        "unstuck.close_fraction",
        "unstuck.ema_distance_fraction",
        "unstuck.ema_span_0_minutes",
        "unstuck.ema_span_1_minutes",
        "unstuck.ema_gating_enabled",
        "unstuck.enabled",
        "unstuck.loss_allowance_fraction",
        "unstuck.exposure_trigger_ratio",
    }
    | CONDITIONAL_HSL_OVERRIDE_PATHS
)
OVERRIDABLE_STANDALONE_BOT_KEYS = frozenset({"wallet_exposure_limit"})


LIVE_COIN_OVERRIDE_KEYS = ("forced_mode_long", "forced_mode_short", "leverage")


def get_live_coin_override_paths():
    """Project existing operational permissions through canonical field IDs."""
    from .field_registry import LIVE_OPERATIONAL_PATHS

    return frozenset(LIVE_OPERATIONAL_PATHS[field] for field in LIVE_COIN_OVERRIDE_KEYS)
