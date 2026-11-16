"""Projection of live policy into native planning inputs."""

from typing import Any

from config.access import get_optional_config_value
from config.bot import normalize_twel_enforcer_policy
from config.strategy import (
    build_runtime_strategy_side,
    get_active_strategy_side,
    normalize_strategy_kind,
)
from live import hsl_live


def bot_params_to_rust_dict(self, pside: str, symbol: str | None) -> dict:
    """Project shared policy into Rust BotParams for a strategy calculation."""
    # Values which are configured globally (not per symbol) live under bot_value.
    global_keys = {
        "n_positions",
        "total_wallet_exposure_limit_ratio",
        "risk_twel_enforcer_enabled",
        "risk_twel_enforcer_policy",
        "risk_twel_enforcer_threshold_ratio",
    }
    bool_keys = {
        "risk_wel_enforcer_enabled",
        "risk_twel_enforcer_enabled",
        "unstuck_enabled",
        "unstuck_ema_gating_enabled",
    }
    string_keys = {
        "risk_twel_enforcer_policy",
    }
    # Strategy parameters travel in their own native envelope. BotParams
    # supplies shared risk, forager, and account policy.
    fields = [
        "forager_log_range_ema_span_minutes",
        "forager_volume_ema_span_minutes",
        "forager_score_weights",
        "risk_entry_cooldown_minutes",
        "entry_cooldown_min_duration_minutes",
        "entry_cooldown_max_duration_minutes",
        "entry_cooldown_weights_minutes",
        "rms_directionality_ema_span_minutes",
        "n_positions",
        "total_wallet_exposure_limit_ratio",
        "wallet_exposure_limit_ratio",
        "risk_wel_enforcer_enabled",
        "risk_wel_enforcer_threshold_ratio",
        "risk_twel_enforcer_enabled",
        "risk_twel_enforcer_policy",
        "risk_twel_enforcer_threshold_ratio",
        "risk_we_excess_allowance_fraction",
        "unstuck_enabled",
        "unstuck_close_fraction",
        "unstuck_ema_gating_enabled",
        "unstuck_ema_distance_fraction",
        "unstuck_ema_span_0_minutes",
        "unstuck_ema_span_1_minutes",
        "unstuck_loss_allowance_fraction",
        "unstuck_exposure_trigger_ratio",
    ]
    out: dict[str, object] = {}
    for key in fields:
        if key in global_keys:
            val = self.bot_value(pside, key)
        else:
            val = (
                self.bp(pside, key, symbol)
                if symbol is not None
                else self.bp(pside, key)
            )
        out_key = key
        if key == "forager_score_weights":
            if not isinstance(val, dict):
                raise TypeError(
                    f"bot.{pside}.forager_score_weights must be a dict, got {type(val).__name__}"
                )
            out[out_key] = {
                "volume": float(val["volume"]),
                "ema_readiness": float(val["ema_readiness"]),
                "log_range": float(val["log_range"]),
                "rms_directionality": float(val["rms_directionality"]),
            }
        elif key == "entry_cooldown_weights_minutes":
            out[out_key] = {
                name: float(val[name]) for name in ("exposure_ratio", "adverse_directionality")
            }
        elif key == "entry_cooldown_max_duration_minutes":
            out[out_key] = None if val is None else float(val)
        elif key == "n_positions":
            out[out_key] = int(round(val))
        elif key in bool_keys:
            out[out_key] = bool(val)
        elif key in string_keys:
            if key == "risk_twel_enforcer_policy":
                out[out_key] = normalize_twel_enforcer_policy(
                    val,
                    path=f"bot.{pside}.risk.total_exposure_enforcer.policy",
                )
        else:
            out[out_key] = float(val)
    policy = hsl_live.policy(self, pside, symbol)
    # The order kernel consumes only execution policy; the shared
    # hsl evaluator owns the signal and controller.
    out.update(
        hsl_enabled=policy["enabled"],
        hsl_panic_close_order_type=policy["panic_close_order_type"],
    )
    return out


def strategy_params_to_rust_dict(self, pside: str, symbol: str | None) -> dict:
    config = getattr(self, "config", {})
    strategy_kind = normalize_strategy_kind(
        get_optional_config_value(config, 'bot.portfolio.strategy_kind')
    )
    strategy_cfg = get_active_strategy_side(
        config.get("bot", {}).get(pside, {}),
        strategy_kind=strategy_kind,
        pside=pside,
    )
    symbol_override = (
        getattr(self, "coin_overrides", {})
        .get(symbol, {})
        .get("bot", {})
        .get(pside, {})
        if symbol is not None
        else {}
    )
    return build_runtime_strategy_side(
        strategy_cfg,
        strategy_kind=strategy_kind,
        pside=pside,
        override_side=symbol_override,
    )

def strategy_value(self, pside: str, key: str, symbol: str | None = None) -> Any:
    strategy_cfg = self._strategy_params_to_rust_dict(pside, symbol)
    current = strategy_cfg
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"missing required strategy key {pside}.{key}")
        current = current[part]
    return current
