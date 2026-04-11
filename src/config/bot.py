import logging
import math
from copy import deepcopy
from typing import Optional

from pure_funcs import sort_dict_keys, str2bool

from .log_output import log_config_message
from .migrations import apply_backward_compatibility_renames
from .schema import get_template_config
from .access import require_config_dict
from .tree_ops import add_missing_keys_recursively


BOT_POSITION_SIDES = ("long", "short")

FORAGER_CANONICAL_TO_INTERNAL_BOT_KEYS = {
    "forager_volatility_ema_span": "filter_volatility_ema_span",
    "forager_volume_ema_span": "filter_volume_ema_span",
    "forager_volume_drop_pct": "filter_volume_drop_pct",
}

FORAGER_CANONICAL_TO_INTERNAL_BOUND_KEYS = {
    "long_forager_volatility_ema_span": "long_filter_volatility_ema_span",
    "long_forager_volume_ema_span": "long_filter_volume_ema_span",
    "long_forager_volume_drop_pct": "long_filter_volume_drop_pct",
    "short_forager_volatility_ema_span": "short_filter_volatility_ema_span",
    "short_forager_volume_ema_span": "short_filter_volume_ema_span",
    "short_forager_volume_drop_pct": "short_filter_volume_drop_pct",
}

def ensure_bot_defaults(
    result: dict, *, verbose: bool = True, tracker: Optional[object] = None
) -> None:
    for pside in BOT_POSITION_SIDES:
        for key, default_value in [
            ("close_trailing_qty_pct", 1.0),
            (
                "entry_trailing_double_down_factor",
                result["bot"][pside].get("entry_grid_double_down_factor", 1.0),
            ),
            (
                "forager_volatility_ema_span",
                result["bot"][pside].get(
                    "forager_volatility_ema_span",
                    result["bot"][pside].get(
                        "filter_volatility_ema_span",
                        result["bot"][pside].get(
                            "filter_rolling_window",
                            result["live"].get("ohlcv_rolling_window", 60.0),
                        ),
                    ),
                ),
            ),
            (
                "forager_volume_ema_span",
                result["bot"][pside].get(
                    "forager_volume_ema_span",
                    result["bot"][pside].get(
                        "filter_volume_ema_span",
                        result["bot"][pside].get(
                            "filter_rolling_window",
                            result["live"].get("ohlcv_rolling_window", 60.0),
                        ),
                    ),
                ),
            ),
            (
                "close_grid_markup_start",
                result["bot"][pside].get("close_grid_min_markup", 0.001)
                + result["bot"][pside].get("close_grid_markup_range", 0.001),
            ),
            (
                "close_grid_markup_end",
                result["bot"][pside].get("close_grid_min_markup", 0.001),
            ),
            (
                "forager_volume_drop_pct",
                result["live"].get("filter_relative_volume_clip_pct", 0.5),
            ),
        ]:
            if key not in result["bot"][pside]:
                result["bot"][pside][key] = default_value
                log_config_message(
                    verbose,
                    logging.INFO,
                    "adding missing backtest parameter %s %s: %s",
                    pside,
                    key,
                    default_value,
                )
                if tracker is not None:
                    tracker.add(["bot", pside, key], default_value)
        if "forager_score_weights" not in result["bot"][pside]:
            weights = {"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}
            result["bot"][pside]["forager_score_weights"] = weights
            log_config_message(
                verbose,
                logging.INFO,
                "adding missing backtest parameter %s forager_score_weights: %s",
                pside,
                weights,
            )
            if tracker is not None:
                tracker.add(["bot", pside, "forager_score_weights"], weights)


def ensure_optimize_bounds_for_bot(
    result: dict, *, verbose: bool = True, tracker: Optional[object] = None
) -> None:
    bounds = result["optimize"]["bounds"]
    for pside in BOT_POSITION_SIDES:
        for key, default_value in [
            ("close_trailing_qty_pct", [0.05, 1.0]),
            ("entry_trailing_double_down_factor", [0.01, 3.0]),
            ("forager_volatility_ema_span", [10.0, 1440.0]),
            ("forager_volume_ema_span", [10.0, 1440.0]),
            ("close_grid_markup_start", bounds.get(f"{pside}_min_markup", [0.001, 0.03])),
            ("close_grid_markup_end", bounds.get(f"{pside}_close_grid_min_markup", [0.001, 0.03])),
            ("forager_volume_drop_pct", [0.0, 1.0]),
        ]:
            opt_key = f"{pside}_{key}"
            if opt_key not in bounds:
                bounds[opt_key] = default_value
                log_config_message(
                    verbose,
                    logging.INFO,
                    "adding missing optimize parameter %s %s: %s",
                    pside,
                    opt_key,
                    default_value,
                )
                if tracker is not None:
                    tracker.add(["optimize", "bounds", opt_key], default_value)
        if "forager_score_weights" not in result["bot"][pside]:
            weights = {"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}
            result["bot"][pside]["forager_score_weights"] = weights
            if tracker is not None:
                tracker.add(["bot", pside, "forager_score_weights"], weights)
        for weight_key in ("volume", "ema_readiness", "volatility"):
            opt_key = f"{pside}_forager_score_weights_{weight_key}"
            if opt_key not in bounds:
                bounds[opt_key] = [0.0, 1.0]
                log_config_message(
                    verbose,
                    logging.INFO,
                    "adding missing optimize parameter %s %s: %s",
                    pside,
                    opt_key,
                    bounds[opt_key],
                )
                if tracker is not None:
                    tracker.add(["optimize", "bounds", opt_key], bounds[opt_key])


def normalize_forager_score_weights(weights: dict, *, path: str) -> dict:
    required_weight_keys = {"volume", "ema_readiness", "volatility"}
    if not isinstance(weights, dict):
        raise TypeError(f"{path} must be a dict")
    missing = sorted(required_weight_keys - set(weights))
    if missing:
        raise ValueError(f"{path} missing required keys: {', '.join(missing)}")
    extra = sorted(set(weights) - required_weight_keys)
    if extra:
        raise ValueError(f"{path} has unsupported keys: {', '.join(extra)}")

    normalized = {}
    total = 0.0
    for key in ("volume", "ema_readiness", "volatility"):
        try:
            value = float(weights[key])
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path}.{key} must be numeric") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{path}.{key} must be finite and non-negative")
        normalized[key] = value
        total += value

    if total <= 0.0:
        return {"volume": 1.0, "ema_readiness": 0.0, "volatility": 0.0}

    return {key: normalized[key] / total for key in ("volume", "ema_readiness", "volatility")}


def forager_score_weights_are_normalized(
    weights: dict,
    *,
    path: str,
    abs_tol: float = 1e-12,
) -> bool:
    normalized = normalize_forager_score_weights(weights, path=path)
    return all(
        math.isclose(normalized[key], weights[key], rel_tol=0.0, abs_tol=abs_tol)
        for key in ("volume", "ema_readiness", "volatility")
    )


def normalize_bot_forager_config(
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[object] = None,
) -> None:
    required_weight_keys = {"volume", "ema_readiness", "volatility"}
    for pside in BOT_POSITION_SIDES:
        bot_cfg = result["bot"][pside]
        raw_drop_pct = bot_cfg["forager_volume_drop_pct"]
        try:
            drop_pct = float(raw_drop_pct)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"bot.{pside}.forager_volume_drop_pct must be numeric") from exc
        if not math.isfinite(drop_pct) or not (0.0 <= drop_pct <= 1.0):
            raise ValueError(f"bot.{pside}.forager_volume_drop_pct must be within [0.0, 1.0]")
        if raw_drop_pct != drop_pct and tracker is not None:
            tracker.update(["bot", pside, "forager_volume_drop_pct"], raw_drop_pct, drop_pct)
        bot_cfg["forager_volume_drop_pct"] = drop_pct

        weights = bot_cfg["forager_score_weights"]
        normalized = normalize_forager_score_weights(
            weights, path=f"bot.{pside}.forager_score_weights"
        )
        if normalized != weights:
            raw_total = 0.0
            if isinstance(weights, dict):
                for key in required_weight_keys:
                    try:
                        raw_total += float(weights[key])
                    except (TypeError, ValueError, KeyError):
                        raw_total = math.nan
                        break
            if raw_total <= 0.0:
                log_config_message(
                    verbose,
                    logging.INFO,
                    "normalizing bot.%s.forager_score_weights all-zero vector to volume-only",
                    pside,
                )
            else:
                log_config_message(
                    verbose,
                    logging.INFO,
                    "normalizing bot.%s.forager_score_weights to relative unit-sum weights: %s",
                    pside,
                    normalized,
                )
            if tracker is not None:
                tracker.update(["bot", pside, "forager_score_weights"], weights, normalized)
        bot_cfg["forager_score_weights"] = normalized


def normalize_position_counts(result: dict, *, tracker: Optional[object] = None) -> None:
    for pside in BOT_POSITION_SIDES:
        current = result["bot"][pside].get("n_positions")
        rounded = int(round(current))
        if tracker is not None and current != rounded:
            tracker.update(["bot", pside, "n_positions"], current, rounded)
        result["bot"][pside]["n_positions"] = rounded


def normalize_entry_grid_inflation_flags(
    result: dict,
    *,
    tracker: Optional[object] = None,
) -> None:
    for pside in BOT_POSITION_SIDES:
        raw_value = result["bot"][pside].get("entry_grid_inflation_enabled", True)
        if isinstance(raw_value, bool):
            normalized = raw_value
        elif isinstance(raw_value, (int, float)) and raw_value in (0, 1):
            normalized = bool(raw_value)
        elif isinstance(raw_value, str):
            normalized = str2bool(raw_value)
        else:
            raise TypeError(
                "bot."
                f"{pside}.entry_grid_inflation_enabled must be a boolean, got "
                f"{type(raw_value).__name__}"
            )
        if tracker is not None and raw_value != normalized:
            tracker.update(["bot", pside, "entry_grid_inflation_enabled"], raw_value, normalized)
        result["bot"][pside]["entry_grid_inflation_enabled"] = normalized


def warn_on_deprecated_entry_grid_inflation(
    result: dict,
    *,
    verbose: bool = True,
) -> None:
    enabled_sides = [
        pside
        for pside in BOT_POSITION_SIDES
        if bool(result["bot"][pside].get("entry_grid_inflation_enabled", True))
    ]
    if not enabled_sides:
        return
    enabled_paths = ", ".join(f"bot.{pside}.entry_grid_inflation_enabled" for pside in enabled_sides)
    log_config_message(
        verbose,
        logging.WARNING,
        "%s enabled: inflated grid re-entries remain on for backwards compatibility, "
        "but this feature is scheduled for deprecation in a future version. Set the flag "
        "to false to adopt the canonical cropped-only grid behavior now.",
        enabled_paths,
    )


def validate_forager_config(
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[object] = None,
) -> None:
    del verbose, tracker
    for pside in BOT_POSITION_SIDES:
        bot_cfg = result["bot"][pside]
        drop_pct = bot_cfg["forager_volume_drop_pct"]
        try:
            drop_pct = float(drop_pct)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"bot.{pside}.forager_volume_drop_pct must be numeric") from exc
        if not math.isfinite(drop_pct) or not (0.0 <= drop_pct <= 1.0):
            raise ValueError(f"bot.{pside}.forager_volume_drop_pct must be within [0.0, 1.0]")
        pside_enabled = (
            float(bot_cfg["total_wallet_exposure_limit"]) > 0.0
            and int(round(float(bot_cfg["n_positions"]))) > 0
        )

        normalized = normalize_forager_score_weights(
            bot_cfg["forager_score_weights"],
            path=f"bot.{pside}.forager_score_weights",
        )
        if not forager_score_weights_are_normalized(
            bot_cfg["forager_score_weights"],
            path=f"bot.{pside}.forager_score_weights",
        ):
            raise ValueError(
                f"bot.{pside}.forager_score_weights must be normalized before validation"
            )

        if pside_enabled and (normalized["volume"] > 0.0 or drop_pct > 0.0):
            volume_span = float(bot_cfg["forager_volume_ema_span"])
            if not math.isfinite(volume_span) or volume_span <= 0.0:
                raise ValueError(
                    f"bot.{pside}.forager_volume_ema_span must be > 0 when "
                    "forager volume ranking or volume pruning is enabled"
                )

        if pside_enabled and normalized["volatility"] > 0.0:
            volatility_span = float(bot_cfg["forager_volatility_ema_span"])
            if not math.isfinite(volatility_span) or volatility_span <= 0.0:
                raise ValueError(
                    f"bot.{pside}.forager_volatility_ema_span must be > 0 when "
                    "forager volatility ranking is enabled"
                )


def format_bot_config(
    bot_cfg: dict,
    *,
    live_cfg: Optional[dict] = None,
    verbose: bool = True,
    tracker: Optional[object] = None,
) -> dict:
    if not isinstance(bot_cfg, dict):
        raise TypeError(f"config.bot must be a dict; got {type(bot_cfg).__name__}")
    template = get_template_config()
    result = {
        "bot": deepcopy(bot_cfg),
        "live": deepcopy(live_cfg) if isinstance(live_cfg, dict) else deepcopy(template["live"]),
        "optimize": {"bounds": {}},
    }
    for pside in BOT_POSITION_SIDES:
        if pside not in result["bot"]:
            seeded = deepcopy(template["bot"][pside])
            result["bot"][pside] = seeded
            if tracker is not None:
                tracker.add(["bot", pside], seeded)
    for path in ("bot.long", "bot.short"):
        require_config_dict(result, path)
    apply_backward_compatibility_renames(result, verbose=verbose, tracker=tracker)
    ensure_bot_defaults(result, verbose=verbose, tracker=tracker)
    add_missing_keys_recursively(
        template["bot"],
        result["bot"],
        parent=["bot"],
        verbose=verbose,
        tracker=tracker,
    )
    normalize_bot_forager_config(result, verbose=verbose, tracker=tracker)
    normalize_position_counts(result, tracker=tracker)
    normalize_entry_grid_inflation_flags(result, tracker=tracker)
    warn_on_deprecated_entry_grid_inflation(result, verbose=verbose)
    return sort_dict_keys(result["bot"])


def apply_forager_internal_aliases(result: dict) -> None:
    def _alias_bot_cfg(bot_cfg: dict) -> None:
        for canonical_key, internal_key in FORAGER_CANONICAL_TO_INTERNAL_BOT_KEYS.items():
            if canonical_key in bot_cfg and internal_key not in bot_cfg:
                bot_cfg[internal_key] = deepcopy(bot_cfg[canonical_key])
        bot_cfg.setdefault("filter_volatility_drop_pct", 0.0)

    for pside in BOT_POSITION_SIDES:
        bot_cfg = result.get("bot", {}).get(pside, {})
        if isinstance(bot_cfg, dict):
            _alias_bot_cfg(bot_cfg)

    for override in (result.get("coin_overrides") or {}).values():
        if not isinstance(override, dict):
            continue
        override_bot = override.get("bot", {})
        if not isinstance(override_bot, dict):
            continue
        for pside in BOT_POSITION_SIDES:
            bot_cfg = override_bot.get(pside, {})
            if isinstance(bot_cfg, dict):
                _alias_bot_cfg(bot_cfg)

    bounds = result.get("optimize", {}).get("bounds", {})
    if not isinstance(bounds, dict):
        return
    for canonical_key, internal_key in FORAGER_CANONICAL_TO_INTERNAL_BOUND_KEYS.items():
        if canonical_key in bounds and internal_key not in bounds:
            bounds[internal_key] = deepcopy(bounds[canonical_key])
