from .access import require_config_dict
from .bot import validate_forager_config
from .coerce import normalize_hsl_cooldown_position_policy, normalize_hsl_signal_mode


def validate_config(config: dict, *, raw_optimize=None, verbose: bool = True, tracker=None) -> None:
    from analysis_visibility import validate_visible_metrics_config
    from optimization.config_adapter import validate_optimize_bounds_against_bot_config

    require_config_dict(config, "monitor")
    optimize_bounds = (
        raw_optimize.get("bounds")
        if isinstance(raw_optimize, dict) and isinstance(raw_optimize.get("bounds"), dict)
        else config.get("optimize", {}).get("bounds", {})
    )
    validate_optimize_bounds_against_bot_config(config["bot"], optimize_bounds)
    normalize_hsl_signal_mode(config["live"]["hsl_signal_mode"])
    normalize_hsl_cooldown_position_policy(config["live"]["hsl_position_during_cooldown_policy"])
    monitor_cfg = require_config_dict(config, "monitor")
    if not str(monitor_cfg["root_dir"]).strip():
        raise ValueError("config.monitor.root_dir must be a non-empty string")
    validate_visible_metrics_config(config)
    validate_forager_config(config, verbose=verbose, tracker=tracker)
