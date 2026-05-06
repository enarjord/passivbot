import math

from .access import require_config_dict
from .bot import validate_bot_config, validate_forager_config
from .coerce import normalize_hsl_cooldown_position_policy, normalize_hsl_signal_mode


def validate_config(
    config: dict, *, raw_optimize=None, verbose: bool = True, tracker=None
) -> None:
    from analysis_visibility import validate_visible_metrics_config
    from optimization.config_adapter import validate_optimize_bounds_against_bot_config

    require_config_dict(config, "monitor")
    optimize_bounds = (
        raw_optimize.get("bounds")
        if isinstance(raw_optimize, dict)
        and isinstance(raw_optimize.get("bounds"), dict)
        else config.get("optimize", {}).get("bounds", {})
    )
    validate_bot_config(config)
    validate_optimize_bounds_against_bot_config(config["bot"], optimize_bounds)
    normalize_hsl_signal_mode(config["live"]["hsl_signal_mode"])
    normalize_hsl_cooldown_position_policy(
        config["live"]["hsl_position_during_cooldown_policy"]
    )
    authoritative_mode = str(
        config["live"].get("authoritative_refresh_mode", "staged")
    ).lower()
    if authoritative_mode not in {"legacy", "staged"}:
        raise ValueError(
            "config.live.authoritative_refresh_mode must be one of: legacy, staged"
        )
    ticker_strategy = str(
        config["live"].get("market_snapshot_ticker_strategy", "auto")
    ).lower()
    if ticker_strategy not in {"auto", "bulk", "symbols"}:
        raise ValueError(
            "config.live.market_snapshot_ticker_strategy must be one of: auto, bulk, symbols"
        )
    try:
        hysteresis_pct = float(config["live"]["forager_score_hysteresis_pct"])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "config.live.forager_score_hysteresis_pct must be numeric"
        ) from exc
    if not math.isfinite(hysteresis_pct) or hysteresis_pct < 0.0:
        raise ValueError(
            "config.live.forager_score_hysteresis_pct must be finite and >= 0.0"
        )
    try:
        active_tail_gap_minutes = float(
            config["live"]["max_active_candle_tail_gap_minutes"]
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "config.live.max_active_candle_tail_gap_minutes must be numeric"
        ) from exc
    if not math.isfinite(active_tail_gap_minutes) or active_tail_gap_minutes <= 0.0:
        raise ValueError(
            "config.live.max_active_candle_tail_gap_minutes must be finite and > 0.0"
        )
    try:
        forager_refresh_seconds = float(config["live"]["max_forager_candle_refresh_seconds"])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "config.live.max_forager_candle_refresh_seconds must be numeric"
        ) from exc
    if not math.isfinite(forager_refresh_seconds) or forager_refresh_seconds <= 0.0:
        raise ValueError(
            "config.live.max_forager_candle_refresh_seconds must be finite and > 0.0"
        )
    max_cancellations = int(config["live"]["max_n_cancellations_per_batch"])
    max_creations = int(config["live"]["max_n_creations_per_batch"])
    if max_cancellations <= max_creations:
        raise ValueError(
            "config.live.max_n_cancellations_per_batch must be greater than "
            "config.live.max_n_creations_per_batch"
        )
    monitor_cfg = require_config_dict(config, "monitor")
    if not str(monitor_cfg["root_dir"]).strip():
        raise ValueError("config.monitor.root_dir must be a non-empty string")
    validate_visible_metrics_config(config)
    validate_forager_config(config, verbose=verbose, tracker=tracker)
