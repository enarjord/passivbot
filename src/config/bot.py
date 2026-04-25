import logging
import math
from copy import deepcopy
from typing import Optional

from pure_funcs import sort_dict_keys, str2bool

from .log_output import log_config_message
from .migrations import apply_backward_compatibility_renames
from .optimize_bounds import get_optimize_bounds_defaults
from .shared_bot import (
    BOT_POSITION_SIDES,
    canonicalize_shared_bot_side,
    flatten_shared_bot_side,
    get_bot_group,
    get_grouped_bot_value,
    inject_flattened_shared_bot_side,
)
from .schema import get_template_config
from .access import require_config_dict
from .tree_ops import add_missing_keys_recursively

DEFAULT_FORAGER_SCORE_WEIGHTS = {"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}
DEFAULT_HSL_TIER_RATIOS = {"yellow": 0.5, "orange": 0.75}
REQUIRED_BOT_KEYS = (
    "close_grid_markup_start",
    "close_grid_markup_end",
    "close_grid_qty_pct",
    "ema_span_0",
    "ema_span_1",
    "entry_grid_double_down_factor",
    "entry_grid_spacing_pct",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
)
CLIFF_EDGE_THRESHOLD_KEYS = (
    "risk_wel_enforcer_threshold",
    "risk_twel_enforcer_threshold",
    "unstuck_threshold",
)
CLIFF_EDGE_DUST_EPS = 1e-9
CLIFF_EDGE_WARNING_THRESHOLD = 0.1
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


def validate_unstuck_ema_dist_value(value, *, path: str, pside: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be finite")
    if pside == "long" and numeric <= -1.0:
        raise ValueError(
            f"{path} must be > -1.0; -1.0 disables auto unstuck by producing a non-positive "
            f"EMA trigger price. Use -0.99 for near-always-on behavior."
        )
    if pside == "short" and numeric >= 1.0:
        raise ValueError(
            f"{path} must be < 1.0; 1.0 disables auto unstuck by producing a non-positive "
            f"EMA trigger price. Use -0.99 for near-always-on behavior."
        )


def validate_bot_config(result: dict) -> None:
    for pside in BOT_POSITION_SIDES:
        validate_unstuck_ema_dist_value(
            get_grouped_bot_value(result["bot"][pside], "unstuck_ema_dist"),
            path=f"bot.{pside}.unstuck_ema_dist",
            pside=pside,
        )


def _bot_path(pside: str, key: str) -> str:
    return f"bot.{pside}.{key}"


def _bot_nested_path(pside: str, key: str, child: str) -> str:
    return f"bot.{pside}.{key}.{child}"


def _format_hydration_log_value(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 10) if math.isfinite(value) else value
    if isinstance(value, list):
        return [_format_hydration_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_hydration_log_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _format_hydration_log_value(item) for key, item in value.items()}
    return value


def _set_hydrated_bot_value(
    result: dict,
    *,
    pside: str,
    key: str,
    value,
    reason: str,
    verbose: bool,
    tracker: Optional[object],
    level: int = logging.INFO,
) -> None:
    result["bot"][pside][key] = deepcopy(value)
    log_config_message(
        verbose,
        level,
        "hydrating omitted %s via %s: %s",
        _bot_path(pside, key),
        reason,
        _format_hydration_log_value(value),
    )
    if tracker is not None:
        tracker.add(["bot", pside, key], value)


def _set_hydrated_bot_nested_value(
    result: dict,
    *,
    pside: str,
    key: str,
    child: str,
    value,
    reason: str,
    verbose: bool,
    tracker: Optional[object],
) -> None:
    result["bot"][pside][key][child] = deepcopy(value)
    log_config_message(
        verbose,
        logging.INFO,
        "hydrating omitted %s via %s: %s",
        _bot_nested_path(pside, key, child),
        reason,
        _format_hydration_log_value(value),
    )
    if tracker is not None:
        tracker.add(["bot", pside, key, child], value)


def _read_legacy_alias(bot_cfg: dict, *keys: str):
    for key in keys:
        if key in bot_cfg and bot_cfg[key] is not None:
            return bot_cfg[key]
    return None


def _derive_close_grid_qty_pct(bot_cfg: dict, *, path: str) -> Optional[float]:
    raw_n_closes = _read_legacy_alias(bot_cfg, "n_closes", "n_close_orders")
    if raw_n_closes is None:
        return None
    try:
        n_closes = int(round(float(raw_n_closes)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} legacy n_closes must be numeric") from exc
    if n_closes <= 0:
        raise ValueError(f"{path} legacy n_closes must round to a positive integer")
    return 1.0 / n_closes


def _bot_side_enabled(bot_cfg: dict, *, pside: str) -> bool:
    path = _bot_path(pside, "total_wallet_exposure_limit")
    try:
        total_wallet_exposure_limit = float(bot_cfg["total_wallet_exposure_limit"])
    except KeyError:
        return False
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be numeric") from exc
    if not math.isfinite(total_wallet_exposure_limit):
        raise ValueError(f"{path} must be finite")
    return total_wallet_exposure_limit > 0.0


def _hydrate_hsl_tier_ratios(
    result: dict,
    *,
    pside: str,
    verbose: bool,
    tracker: Optional[object],
) -> None:
    bot_cfg = result["bot"][pside]
    if "hsl_tier_ratios" not in bot_cfg or bot_cfg["hsl_tier_ratios"] is None:
        _set_hydrated_bot_value(
            result,
            pside=pside,
            key="hsl_tier_ratios",
            value=DEFAULT_HSL_TIER_RATIOS,
            reason="disabled HSL compatibility default",
            verbose=verbose,
            tracker=tracker,
        )
        return
    if not isinstance(bot_cfg["hsl_tier_ratios"], dict):
        return
    for child, default_value in DEFAULT_HSL_TIER_RATIOS.items():
        if child not in bot_cfg["hsl_tier_ratios"]:
            _set_hydrated_bot_nested_value(
                result,
                pside=pside,
                key="hsl_tier_ratios",
                child=child,
                value=default_value,
                reason="disabled HSL compatibility default",
                verbose=verbose,
                tracker=tracker,
            )


def _hydrate_forager_score_weights(
    result: dict,
    *,
    pside: str,
    verbose: bool,
    tracker: Optional[object],
) -> None:
    bot_cfg = result["bot"][pside]
    if "forager_score_weights" not in bot_cfg or bot_cfg["forager_score_weights"] is None:
        _set_hydrated_bot_value(
            result,
            pside=pside,
            key="forager_score_weights",
            value=DEFAULT_FORAGER_SCORE_WEIGHTS,
            reason="legacy forager scoring default",
            verbose=verbose,
            tracker=tracker,
        )
        return
    if not isinstance(bot_cfg["forager_score_weights"], dict):
        return
    for child, default_value in DEFAULT_FORAGER_SCORE_WEIGHTS.items():
        if child not in bot_cfg["forager_score_weights"]:
            _set_hydrated_bot_nested_value(
                result,
                pside=pside,
                key="forager_score_weights",
                child=child,
                value=default_value,
                reason="legacy forager scoring compatibility default",
                verbose=verbose,
                tracker=tracker,
            )


def _normalize_cliff_edge_threshold(
    value,
    *,
    path: str,
    verbose: bool,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be finite")
    if abs(numeric) < CLIFF_EDGE_DUST_EPS:
        if numeric != 0.0:
            log_config_message(
                verbose,
                logging.WARNING,
                "%s=%s is within dust tolerance %.1e; snapping to 0.0",
                path,
                numeric,
                CLIFF_EDGE_DUST_EPS,
            )
        return 0.0
    if 0.0 < numeric < CLIFF_EDGE_WARNING_THRESHOLD:
        log_config_message(
            verbose,
            logging.WARNING,
            "%s=%s is a very small positive threshold; behavior may be extremely aggressive",
            path,
            numeric,
        )
    return numeric


def normalize_cliff_edge_thresholds(
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[object] = None,
) -> None:
    for pside in BOT_POSITION_SIDES:
        risk_cfg = get_bot_group(result["bot"][pside], "risk")
        for key in CLIFF_EDGE_THRESHOLD_KEYS:
            raw_value = get_grouped_bot_value(result["bot"][pside], key)
            normalized = _normalize_cliff_edge_threshold(
                raw_value,
                path=_bot_path(pside, key),
                verbose=verbose,
            )
            if tracker is not None and raw_value != normalized:
                tracker.update(["bot", pside, key], raw_value, normalized)
            if risk_cfg:
                risk_cfg[key.removeprefix("risk_")] = normalized
            else:
                result["bot"][pside][key] = normalized


def ensure_required_bot_params_present(result: dict) -> None:
    template = get_template_config()["bot"]
    for pside in BOT_POSITION_SIDES:
        bot_cfg = result["bot"][pside]
        flat_bot_cfg = flatten_shared_bot_side(bot_cfg)
        flat_template_cfg = flatten_shared_bot_side(template[pside])
        if _bot_side_enabled(bot_cfg, pside=pside):
            missing = [
                key
                for key in REQUIRED_BOT_KEYS
                if get_grouped_bot_value(bot_cfg, key, default=None) is None
            ]
            if get_grouped_bot_value(bot_cfg, "n_positions", default=None) is None:
                missing.append("n_positions")
            if missing:
                joined = ", ".join(_bot_path(pside, key) for key in missing)
                raise ValueError(f"Missing required bot config parameter(s): {joined}")
        unhandled = sorted(set(flat_template_cfg) - set(flat_bot_cfg))
        if unhandled:
            joined = ", ".join(_bot_path(pside, key) for key in unhandled)
            raise ValueError(f"Missing explicit hydration policy for bot parameter(s): {joined}")

def ensure_bot_defaults(
    result: dict, *, verbose: bool = True, tracker: Optional[object] = None
) -> None:
    template_bot = get_template_config()["bot"]
    for pside in BOT_POSITION_SIDES:
        canonicalize_shared_bot_side(
            result["bot"][pside],
            path_prefix=("bot", pside),
            tracker=tracker,
            seed_missing_groups=True,
        )
    add_missing_keys_recursively(
        template_bot,
        result["bot"],
        parent=["bot"],
        verbose=verbose,
        tracker=tracker,
    )


def ensure_optimize_bounds_for_bot(
    result: dict, *, verbose: bool = True, tracker: Optional[object] = None
) -> None:
    del verbose
    bounds = result["optimize"]["bounds"]
    defaults = get_optimize_bounds_defaults()
    add_missing_keys_recursively(
        defaults,
        bounds,
        parent=["optimize", "bounds"],
        verbose=False,
        tracker=tracker,
    )
    for pside in BOT_POSITION_SIDES:
        forager_cfg = get_bot_group(result["bot"][pside], "forager")
        if "score_weights" not in forager_cfg:
            weights = {"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0}
            forager_cfg["score_weights"] = weights
            if tracker is not None:
                tracker.add(["bot", pside, "forager", "score_weights"], weights)


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
        return {"volume": 0.0, "ema_readiness": 1.0, "volatility": 0.0}

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
        forager_cfg = get_bot_group(result["bot"][pside], "forager")
        raw_drop_pct = forager_cfg["volume_drop_pct"]
        try:
            drop_pct = float(raw_drop_pct)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"bot.{pside}.forager.volume_drop_pct must be numeric") from exc
        if not math.isfinite(drop_pct) or not (0.0 <= drop_pct <= 1.0):
            raise ValueError(f"bot.{pside}.forager.volume_drop_pct must be within [0.0, 1.0]")
        if raw_drop_pct != drop_pct and tracker is not None:
            tracker.update(["bot", pside, "forager", "volume_drop_pct"], raw_drop_pct, drop_pct)
        forager_cfg["volume_drop_pct"] = drop_pct

        weights = forager_cfg["score_weights"]
        normalized = normalize_forager_score_weights(
            weights, path=f"bot.{pside}.forager.score_weights"
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
                    "normalizing bot.%s.forager_score_weights all-zero vector to default"
                    " ema-readiness-only ranking",
                    pside,
                )
            else:
                log_config_message(
                    verbose,
                    logging.INFO,
                    "normalizing bot.%s.forager.score_weights to relative unit-sum weights: %s",
                    pside,
                    normalized,
                )
            if tracker is not None:
                tracker.update(["bot", pside, "forager", "score_weights"], weights, normalized)
        forager_cfg["score_weights"] = normalized


def normalize_position_counts(result: dict, *, tracker: Optional[object] = None) -> None:
    for pside in BOT_POSITION_SIDES:
        canonicalize_shared_bot_side(
            result["bot"][pside],
            path_prefix=("bot", pside),
            tracker=tracker,
            seed_missing_groups=False,
        )
        risk_cfg = get_bot_group(result["bot"][pside], "risk")
        current = risk_cfg.get("n_positions")
        if current is None:
            continue
        try:
            rounded = int(round(float(current)))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"bot.{pside}.n_positions must be numeric") from exc
        if tracker is not None and current != rounded:
            tracker.update(["bot", pside, "risk", "n_positions"], current, rounded)
        risk_cfg["n_positions"] = rounded


def _parse_entry_grid_inflation_flag(raw_value, *, path: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)) and raw_value in (0, 1):
        return bool(raw_value)
    if isinstance(raw_value, str):
        return str2bool(raw_value)
    raise TypeError(f"{path} must be a boolean, got {type(raw_value).__name__}")


def strip_deprecated_entry_grid_inflation_flags(
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[object] = None,
) -> None:
    for pside in BOT_POSITION_SIDES:
        bot_cfg = result["bot"][pside]
        if "entry_grid_inflation_enabled" not in bot_cfg:
            continue
        raw_value = bot_cfg["entry_grid_inflation_enabled"]
        normalized = _parse_entry_grid_inflation_flag(
            raw_value, path=f"bot.{pside}.entry_grid_inflation_enabled"
        )
        if normalized:
            log_config_message(
                verbose,
                logging.WARNING,
                "bot.%s.entry_grid_inflation_enabled is deprecated and has no effect; removing it",
                pside,
            )
        removed = bot_cfg.pop("entry_grid_inflation_enabled")
        if tracker is not None:
            tracker.remove(["bot", pside, "entry_grid_inflation_enabled"], removed)


def strip_deprecated_coin_override_entry_grid_inflation_flags(
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[object] = None,
) -> None:
    for coin, override in (result.get("coin_overrides") or {}).items():
        if not isinstance(override, dict):
            continue
        override_bot = override.get("bot", {})
        if not isinstance(override_bot, dict):
            continue
        for pside in BOT_POSITION_SIDES:
            bot_cfg = override_bot.get(pside, {})
            if not isinstance(bot_cfg, dict) or "entry_grid_inflation_enabled" not in bot_cfg:
                continue
            raw_value = bot_cfg["entry_grid_inflation_enabled"]
            normalized = _parse_entry_grid_inflation_flag(
                raw_value, path=f"coin_overrides.{coin}.bot.{pside}.entry_grid_inflation_enabled"
            )
            if normalized:
                log_config_message(
                    verbose,
                    logging.WARNING,
                    "coin_overrides.%s.bot.%s.entry_grid_inflation_enabled is deprecated and"
                    " has no effect; removing it",
                    coin,
                    pside,
                )
            removed = bot_cfg.pop("entry_grid_inflation_enabled")
            if tracker is not None:
                tracker.remove(
                    ["coin_overrides", coin, "bot", pside, "entry_grid_inflation_enabled"],
                    removed,
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
        forager_cfg = get_bot_group(bot_cfg, "forager")
        risk_cfg = get_bot_group(bot_cfg, "risk")
        drop_pct = forager_cfg["volume_drop_pct"]
        try:
            drop_pct = float(drop_pct)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"bot.{pside}.forager.volume_drop_pct must be numeric") from exc
        if not math.isfinite(drop_pct) or not (0.0 <= drop_pct <= 1.0):
            raise ValueError(f"bot.{pside}.forager.volume_drop_pct must be within [0.0, 1.0]")
        pside_enabled = (
            float(risk_cfg["total_wallet_exposure_limit"]) > 0.0
            and int(round(float(risk_cfg["n_positions"]))) > 0
        )

        normalized = normalize_forager_score_weights(
            forager_cfg["score_weights"],
            path=f"bot.{pside}.forager.score_weights",
        )
        if not forager_score_weights_are_normalized(
            forager_cfg["score_weights"],
            path=f"bot.{pside}.forager.score_weights",
        ):
            raise ValueError(
                f"bot.{pside}.forager.score_weights must be normalized before validation"
            )

        if pside_enabled and (normalized["volume"] > 0.0 or drop_pct > 0.0):
            volume_span = float(forager_cfg["volume_ema_span"])
            if not math.isfinite(volume_span) or volume_span <= 0.0:
                raise ValueError(
                    f"bot.{pside}.forager.volume_ema_span must be > 0 when "
                    "forager volume ranking or volume pruning is enabled"
                )

        if pside_enabled and normalized["volatility"] > 0.0:
            volatility_span = float(forager_cfg["volatility_ema_span"])
            if not math.isfinite(volatility_span) or volatility_span <= 0.0:
                raise ValueError(
                    f"bot.{pside}.forager.volatility_ema_span must be > 0 when "
                    "forager volatility ranking is enabled"
                )


def format_bot_config(
    bot_cfg: dict,
    *,
    live_cfg: Optional[dict] = None,
    verbose: bool = True,
    warn_deprecations: bool = True,
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
            seeded["total_wallet_exposure_limit"] = 0.0
            result["bot"][pside] = seeded
            if tracker is not None:
                tracker.add(["bot", pside], seeded)
    for path in ("bot.long", "bot.short"):
        require_config_dict(result, path)
    apply_backward_compatibility_renames(result, verbose=verbose, tracker=tracker)
    ensure_bot_defaults(result, verbose=verbose, tracker=tracker)
    ensure_required_bot_params_present(result)
    normalize_cliff_edge_thresholds(result, verbose=verbose, tracker=tracker)
    normalize_bot_forager_config(result, verbose=verbose, tracker=tracker)
    normalize_position_counts(result, tracker=tracker)
    strip_deprecated_entry_grid_inflation_flags(
        result,
        verbose=verbose and warn_deprecations,
        tracker=tracker,
    )
    for pside in BOT_POSITION_SIDES:
        inject_flattened_shared_bot_side(result["bot"][pside])
    return sort_dict_keys(result["bot"])


def apply_forager_internal_aliases(result: dict) -> None:
    def _alias_bot_cfg(bot_cfg: dict) -> None:
        inject_flattened_shared_bot_side(bot_cfg)
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
