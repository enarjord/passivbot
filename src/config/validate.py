from .access import require_config_dict
from .bot import validate_forager_config
from .coerce import normalize_hsl_cooldown_position_policy, normalize_hsl_signal_mode
from .shared_bot import get_grouped_bot_value
from .strategy import (
    BOT_POSITION_SIDES,
    SUPPORTED_STRATEGY_KINDS,
    get_active_strategy_side,
    normalize_strategy_kind,
)


def validate_config(config: dict, *, raw_optimize=None, verbose: bool = True, tracker=None) -> None:
    from analysis_visibility import validate_visible_metrics_config

    del raw_optimize
    require_config_dict(config, "monitor")
    strategy_kind = normalize_strategy_kind(config["live"].get("strategy_kind"))
    if strategy_kind not in SUPPORTED_STRATEGY_KINDS:
        allowed = ", ".join(sorted(SUPPORTED_STRATEGY_KINDS))
        raise ValueError(f"live.strategy_kind must be one of {{{allowed}}}; got {strategy_kind!r}")
    for pside in BOT_POSITION_SIDES:
        bot_side = require_config_dict(config, f"bot.{pside}")
        require_config_dict(bot_side, "strategy")
        entry_cooldown_minutes = float(
            get_grouped_bot_value(bot_side, "risk_entry_cooldown_minutes", 0.0) or 0.0
        )
        if entry_cooldown_minutes < 0.0:
            raise ValueError(f"bot.{pside}.risk.entry_cooldown_minutes must be >= 0.0")
        active_strategy = get_active_strategy_side(bot_side, strategy_kind=strategy_kind, pside=pside)
        if not isinstance(active_strategy, dict) or not active_strategy:
            raise ValueError(
                f"bot.{pside}.strategy.{strategy_kind} must be a non-empty dict for active strategy_kind"
            )
    normalize_hsl_signal_mode(config["live"]["hsl_signal_mode"])
    normalize_hsl_cooldown_position_policy(config["live"]["hsl_position_during_cooldown_policy"])
    monitor_cfg = require_config_dict(config, "monitor")
    if not str(monitor_cfg["root_dir"]).strip():
        raise ValueError("config.monitor.root_dir must be a non-empty string")
    validate_visible_metrics_config(config)
    validate_forager_config(config, verbose=verbose, tracker=tracker)
