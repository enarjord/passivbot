import argparse
import logging
import math
import os
from copy import deepcopy
from typing import Callable

from pure_funcs import sort_dict_keys
from utils import symbol_to_coin

from .load import load_input_config, prepare_config
from .log_output import log_config_message
from .schema import get_template_config
from .shared_bot import BOT_GROUP_FIELD_MAP, canonicalize_shared_bot_side
from .strategy import (
    TRAILING_GRID_V7_FLAT_ONLY_KEYS,
    get_strategy_param_keys,
    normalize_strategy_kind,
)
from .strategy_spec import get_supported_strategy_kinds
from .transform_log import record_transform
from risk_limits import normalize_we_excess_allowance_mode


def apply_allowed_modifications(src, modifications, allowed_overrides, return_full=True):
    if return_full:
        result = deepcopy(src)
        target = result
    else:
        result = {}
        target = result

    def _has_allowed_values(allowed_subdict):
        for value in allowed_subdict.values():
            if value is True:
                return True
            if isinstance(value, dict) and _has_allowed_values(value):
                return True
        return False

    def _apply_recursive(target_dict, mod_dict, allowed_dict):
        for key, mod_value in mod_dict.items():
            if key not in allowed_dict:
                continue
            allowed_value = allowed_dict[key]
            if isinstance(allowed_value, dict) and isinstance(mod_value, dict):
                if not _has_allowed_values(allowed_value):
                    continue
                if key not in target_dict:
                    target_dict[key] = {}
                _apply_recursive(
                    target_dict[key],
                    mod_value,
                    allowed_value,
                )
                if not return_full and not target_dict[key]:
                    target_dict.pop(key, None)
            elif allowed_value is True:
                if key in {"risk_we_excess_allowance_mode", "we_excess_allowance_mode"}:
                    mod_value = normalize_we_excess_allowance_mode(mod_value)
                if return_full:
                    target_dict[key] = deepcopy(mod_value)
                else:
                    target_dict[key] = deepcopy(mod_value)

    _apply_recursive(target, modifications, allowed_overrides)
    return result


_ALLOWED_FLAT_BOT_SIDE_MODIFICATIONS = {
    "unstuck_close_pct": True,
    "unstuck_ema_dist": True,
    "unstuck_enabled": True,
    "unstuck_loss_allowance_pct": True,
    "unstuck_threshold": True,
    "wallet_exposure_limit": True,
    "risk_twel_entry_gate_enabled": False,
    "risk_wel_enforcer_enabled": True,
    "risk_wel_enforcer_threshold": True,
    "risk_we_excess_allowance_pct": True,
    "risk_we_excess_allowance_mode": True,
    "risk_twel_enforcer_enabled": False,
    "risk_twel_enforcer_policy": False,
    "risk_twel_enforcer_threshold": False,
}


def allowed_flat_bot_side_modification_keys() -> frozenset[str]:
    return frozenset(
        key for key, allowed in _ALLOWED_FLAT_BOT_SIDE_MODIFICATIONS.items() if allowed is True
    )


_UNSUPPORTED_FLAT_STRATEGY_OVERRIDE_KEYS = {
    "close_weight_volatility_1h",
    "close_weight_volatility_1m",
    "ema_span_0",
    "ema_span_1",
    "entry_volatility_ema_span_1h",
    "entry_volatility_ema_span_1m",
    "entry_weight_volatility_1h",
    "entry_weight_volatility_1m",
    "entry_we_weight",
} | TRAILING_GRID_V7_FLAT_ONLY_KEYS

_RUNTIME_GENERATED_OVERRIDE_KEYS = {
    "filter_volatility_drop_pct",
    "filter_volatility_ema_span_1m",
    "filter_volume_drop_pct",
    "filter_volume_ema_span_1m",
}


def _reject_flat_strategy_coin_overrides(overrides: dict, *, coin: str) -> None:
    if not isinstance(overrides, dict):
        return
    bot_overrides = overrides.get("bot")
    if not isinstance(bot_overrides, dict):
        return
    for pside in ("long", "short"):
        side_overrides = bot_overrides.get(pside)
        if not isinstance(side_overrides, dict):
            continue
        bad_keys = sorted(
            key
            for key in side_overrides
            if key in _UNSUPPORTED_FLAT_STRATEGY_OVERRIDE_KEYS
            or any(
                key in get_strategy_param_keys(strategy_kind)
                for strategy_kind in get_supported_strategy_kinds()
            )
        )
        if bad_keys:
            joined = ", ".join(bad_keys)
            raise ValueError(
                f"coin_overrides.{coin}.bot.{pside} contains unsupported flat strategy "
                f"override key(s): {joined}. Run `passivbot tool migrate-config-v7`, "
                f"or use coin_overrides.{coin}.bot.{pside}.strategy.<strategy_kind>.* in v8."
            )


def _allowed_bot_side_modifications() -> dict:
    def _allow_dotted_paths(keys: tuple[str, ...]) -> dict:
        result = {}
        for key in keys:
            current = result
            parts = tuple(part for part in key.split(".") if part)
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = True
        return result

    side = deepcopy(_ALLOWED_FLAT_BOT_SIDE_MODIFICATIONS)
    side["strategy"] = {
        strategy_kind: _allow_dotted_paths(keys)
        for strategy_kind in get_supported_strategy_kinds()
        for keys in (get_strategy_param_keys(strategy_kind),)
    }
    for group_name, field_map in BOT_GROUP_FIELD_MAP.items():
        grouped_allowed = {
            local_key: _ALLOWED_FLAT_BOT_SIDE_MODIFICATIONS.get(flat_key, False)
            for local_key, flat_key in field_map.items()
        }
        if any(grouped_allowed.values()):
            side[group_name] = grouped_allowed
    return side


def get_allowed_modifications():
    return {
        "bot": {
            "long": _allowed_bot_side_modifications(),
            "short": _allowed_bot_side_modifications(),
        },
        "live": {
            "forced_mode_long": True,
            "forced_mode_short": True,
            "leverage": True,
        },
    }


def set_nested_value(d: dict, p: list, v: object):
    if not p:
        raise ValueError("Path cannot be empty")
    current = d
    for key in p[:-1]:
        current = current[key]
    current[p[-1]] = v


def set_nested_value_safe(d: dict, p: list, v: object, create_missing=False):
    if not p:
        raise ValueError("Path cannot be empty")
    current = d
    for key in p[:-1]:
        if key not in current:
            if create_missing:
                current[key] = {}
            else:
                return False
        elif not isinstance(current[key], dict):
            return False
        current = current[key]
    current[p[-1]] = v
    return True


def nested_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            nested_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _unwrap_override_document(document: dict, *, source: str) -> dict:
    if not isinstance(document, dict):
        raise TypeError(f"{source} must contain a configuration object")
    nested = document.get("config")
    if isinstance(nested, dict) and ("bot" in nested or "live" in nested):
        return deepcopy(nested)
    return deepcopy(document)


def _format_override_path(coin: str, parts: tuple[str, ...]) -> str:
    return ".".join(("coin_overrides", coin, *parts))


def _extract_allowed_patch(
    document: dict,
    *,
    coin: str,
    source: str,
    strict: bool,
    strategy_kind: str,
    strip_runtime_aliases: bool = False,
) -> dict:
    """Extract explicitly supplied allowed leaves without hydrating defaults."""

    source_doc = _unwrap_override_document(document, source=source)
    _reject_flat_strategy_coin_overrides(source_doc, coin=coin)
    source_live = source_doc.get("live")
    if isinstance(source_live, dict) and "strategy_kind" in source_live:
        source_strategy_kind = normalize_strategy_kind(source_live["strategy_kind"])
        if source_strategy_kind != strategy_kind:
            raise ValueError(
                f"{source}.live.strategy_kind {source_strategy_kind!r} does not match "
                f"the global strategy_kind {strategy_kind!r}"
            )
    bot = source_doc.get("bot")
    if bot is not None and not isinstance(bot, dict):
        raise TypeError(f"{source}.bot must be a dict")
    if isinstance(bot, dict):
        for pside in ("long", "short"):
            side = bot.get(pside)
            if side is None:
                continue
            if not isinstance(side, dict):
                raise TypeError(f"{source}.bot.{pside} must be a dict")
            if strip_runtime_aliases:
                for generated_key in _RUNTIME_GENERATED_OVERRIDE_KEYS:
                    side.pop(generated_key, None)
            canonicalize_shared_bot_side(
                side,
                path_prefix=(source, "bot", pside),
                seed_missing_groups=False,
            )
            strategy = side.get("strategy")
            if strategy is None:
                continue
            if not isinstance(strategy, dict):
                raise TypeError(f"{source}.bot.{pside}.strategy must be a dict")
            unsupported = sorted(set(strategy) - set(get_supported_strategy_kinds()))
            if unsupported:
                raise ValueError(
                    f"{source}.bot.{pside}.strategy has unsupported strategy kind(s): "
                    + ", ".join(unsupported)
                )
            mismatched = sorted(kind for kind in strategy if kind != strategy_kind)
            if mismatched:
                declares_strategy_kind = (
                    isinstance(source_live, dict) and "strategy_kind" in source_live
                )
                if strict or not declares_strategy_kind:
                    raise ValueError(
                        f"{source}.bot.{pside}.strategy.{mismatched[0]} cannot override active "
                        f"strategy_kind {strategy_kind!r}"
                    )
                for kind in mismatched:
                    strategy.pop(kind, None)

    allowed = get_allowed_modifications()

    def visit(value, policy, path: tuple[str, ...]):
        if policy is True:
            if value is None:
                raise TypeError(f"{_format_override_path(coin, path)} may not be null")
            return deepcopy(value)
        if not isinstance(policy, dict):
            if strict:
                raise ValueError(f"{_format_override_path(coin, path)} is not overridable")
            return None
        if not isinstance(value, dict):
            raise TypeError(f"{_format_override_path(coin, path)} must be a dict")
        result = {}
        for key, child in value.items():
            if key not in policy:
                if strict:
                    raise ValueError(
                        f"{_format_override_path(coin, path + (key,))} is not overridable"
                    )
                continue
            child_value = visit(child, policy[key], path + (key,))
            if child_value is not None and (not isinstance(child_value, dict) or child_value):
                result[key] = child_value
        return result

    patch = {}
    for root in ("bot", "live"):
        if root not in source_doc:
            continue
        root_patch = visit(source_doc[root], allowed[root], (root,))
        if root_patch:
            patch[root] = root_patch
    if strict:
        unknown_roots = sorted(set(source_doc) - {"bot", "live", "override_config_path"})
        if unknown_roots:
            raise ValueError(
                f"coin_overrides.{coin} has unsupported key(s): " + ", ".join(unknown_roots)
            )
    return patch


def _iter_patch_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_patch_leaves(child, path + (key,))
        return
    yield path, value


def _get_nested_value(config: dict, path: tuple[str, ...]):
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_patch_leaf_types(
    config: dict, patch: dict, *, coin: str, origin: str
) -> None:
    template = get_template_config()
    for path, value in _iter_patch_leaves(patch):
        display_path = f"{_format_override_path(coin, path)} ({origin})"
        reference = _get_nested_value(config, path)
        if reference is None:
            reference = _get_nested_value(template, path)
        if isinstance(value, bool):
            if not isinstance(reference, bool):
                raise TypeError(f"{display_path} must be numeric, not a boolean")
            continue
        if isinstance(value, (int, float)):
            if isinstance(reference, (bool, str)):
                expected = "boolean" if isinstance(reference, bool) else "string"
                raise TypeError(f"{display_path} must be a {expected}, not numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{display_path} must be finite")
            continue
        if isinstance(value, str):
            if path in {
                ("live", "forced_mode_long"),
                ("live", "forced_mode_short"),
            }:
                if value:
                    try:
                        expand_PB_mode(value)
                    except Exception as exc:
                        raise ValueError(f"{display_path} has invalid mode {value!r}") from exc
            elif not isinstance(reference, str):
                raise TypeError(f"{display_path} must be numeric or boolean, not a string")
            continue
        raise TypeError(
            f"{display_path} must be a scalar value; got {type(value).__name__}"
        )

    for pside in ("long", "short"):
        side_patch = patch.get("bot", {}).get(pside, {})
        if "wallet_exposure_limit" in side_patch:
            value = side_patch["wallet_exposure_limit"]
            display_path = (
                f"coin_overrides.{coin}.bot.{pside}.wallet_exposure_limit ({origin})"
            )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{display_path} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"{display_path} must be finite and >= 0.0"
                )
    leverage = patch.get("live", {}).get("leverage")
    if leverage is not None and float(leverage) <= 0.0:
        raise ValueError(
            f"coin_overrides.{coin}.live.leverage ({origin}) must be > 0.0"
        )


def _normalize_patch_values(patch: dict, *, coin: str) -> None:
    for pside in ("long", "short"):
        risk = patch.get("bot", {}).get(pside, {}).get("risk", {})
        if isinstance(risk, dict) and "we_excess_allowance_mode" in risk:
            risk["we_excess_allowance_mode"] = normalize_we_excess_allowance_mode(
                risk["we_excess_allowance_mode"],
                path=(
                    f"coin_overrides.{coin}.bot.{pside}.risk."
                    "we_excess_allowance_mode"
                ),
            )


def _validate_effective_coin_config(
    config: dict, patch: dict, *, coin: str, origin: str
) -> None:
    effective = deepcopy(config)
    for metadata_key in (
        "_coins_sources",
        "_raw",
        "_raw_effective",
        "_transform_log",
    ):
        effective.pop(metadata_key, None)
    effective["coin_overrides"] = {}
    validation_input = get_template_config()
    nested_update(validation_input, effective)
    effective = validation_input
    for root in ("bot", "live"):
        if root in patch:
            nested_update(effective[root], deepcopy(patch[root]))
    try:
        prepare_config(
            effective,
            base_config_path=str(effective.get("live", {}).get("base_config_path") or ""),
            live_only=False,
            verbose=False,
            log_config_transforms=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise type(exc)(
            f"coin_overrides.{coin} produces an invalid config after {origin}: {exc}"
        ) from exc


def load_override_config(
    config,
    coin,
    *,
    config_loader: Callable[[str], dict] | None = None,
):
    path = config.get("coin_overrides", {}).get(coin, {}).get("override_config_path")
    if path is None:
        return {}
    if not isinstance(path, str) or not path.strip():
        raise TypeError(f"coin_overrides.{coin}.override_config_path must be a non-empty string")
    path = path.strip()
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        base_config_path = config.get("live", {}).get("base_config_path")
        if base_config_path:
            candidates.append(os.path.join(os.path.dirname(base_config_path), path))
        candidates.append(path)
    resolved_path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
    if resolved_path is None:
        attempted = ", ".join(os.path.abspath(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"coin_overrides.{coin}.override_config_path not found; tried: {attempted}"
        )
    if config_loader is not None:
        return config_loader(resolved_path)
    # Validate the document as a configuration, but return its raw explicit values.
    try:
        source, base_config_path, raw_snapshot = load_input_config(
            resolved_path, log_info=False
        )
        if not isinstance(source, dict):
            raise TypeError("configuration root must be an object")
        validation_source = get_template_config()
        nested_update(
            validation_source,
            _unwrap_override_document(
                source,
                source=f"coin_overrides.{coin}.override_config_path",
            ),
        )
        prepare_config(
            validation_source,
            base_config_path=base_config_path,
            verbose=False,
            log_config_transforms=False,
            raw_snapshot=validation_source,
        )
    except OSError as exc:
        raise type(exc)(
            f"coin_overrides.{coin}.override_config_path {resolved_path!r} "
            f"could not be read: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"coin_overrides.{coin}.override_config_path {resolved_path!r} is invalid: {exc}"
        ) from exc
    return raw_snapshot


def parse_old_coin_flags(config) -> dict:
    key_map = {
        "short_mode": ["live", "forced_mode_short"],
        "long_mode": ["live", "forced_mode_long"],
        "WE_limit_long": ["bot", "long", "wallet_exposure_limit"],
        "WE_limit_short": ["bot", "short", "wallet_exposure_limit"],
        "leverage": ["live", "leverage"],
    }
    if not isinstance(config, dict) or "live" not in config or "coin_flags" not in config["live"]:
        return {}
    flags = config["live"]["coin_flags"]
    if not isinstance(flags, dict):
        return {}
    result = {}
    for coin in flags:
        result[coin] = {}
        if not isinstance(flags[coin], str):
            continue
        parser = _build_flag_argparser()
        keysvals = vars(parser.parse_args(flags[coin].split()))
        if live_config_path := keysvals.get("live_config_path"):
            set_nested_value_safe(
                result[coin],
                ["override_config_path"],
                live_config_path,
                create_missing=True,
            )
        for key, value in keysvals.items():
            if value and key in key_map:
                set_nested_value_safe(result[coin], key_map[key], value, create_missing=True)
    return result


def parse_overrides(
    config,
    *,
    verbose=True,
    override_loader: Callable[[dict, str], dict] | None = None,
    symbol_normalizer: Callable[[str], str] | None = None,
):
    if override_loader is None:
        override_loader = load_override_config
    if symbol_normalizer is None:
        symbol_normalizer = symbol_to_coin
    result = deepcopy(config)
    if "coin_overrides" in result and not isinstance(result["coin_overrides"], dict):
        raise TypeError("coin_overrides must be a dict")
    if not result.get("coin_overrides", {}):
        result["coin_overrides"] = parse_old_coin_flags(config)
        if verbose and result["coin_overrides"]:
            log_config_message(
                verbose,
                logging.INFO,
                "Converted old coin_flags to coin_overrides: %s -> %s",
                config.get("live", {}).get("coin_flags"),
                result["coin_overrides"],
            )
    if "live" in result:
        result["live"].pop("coin_flags", None)
        result["live"].setdefault("coin_flags", {})
    normalized_overrides = {}
    normalized_sources = {}
    for coin, overrides in result["coin_overrides"].items():
        if not isinstance(coin, str):
            raise TypeError("coin_overrides keys must be strings")
        formatted_coin = symbol_normalizer(coin)
        if not formatted_coin:
            raise ValueError(f"coin_overrides.{coin} is not a valid coin or symbol")
        if formatted_coin in normalized_overrides:
            prior = normalized_sources[formatted_coin]
            raise ValueError(
                f"coin_overrides keys {prior!r} and {coin!r} both normalize to "
                f"{formatted_coin!r}"
            )
        normalized_overrides[formatted_coin] = deepcopy(overrides)
        normalized_sources[formatted_coin] = coin
        if formatted_coin != coin:
            log_config_message(
                verbose,
                logging.INFO,
                "Renamed %s -> %s for coin_overrides",
                coin,
                formatted_coin,
            )
    result["coin_overrides"] = normalized_overrides
    strategy_kind = normalize_strategy_kind(result.get("live", {}).get("strategy_kind"))
    runtime_compiled = any(
        isinstance(item, dict) and item.get("step") == "compile_runtime_config"
        for item in result.get("_transform_log", [])
    )
    for coin, overrides in result["coin_overrides"].items():
        parsed_overrides = {}
        loaded = override_loader(result, coin)
        if loaded:
            parsed_overrides = _extract_allowed_patch(
                loaded,
                coin=coin,
                source=f"coin_overrides.{coin}.override_config_path",
                strict=False,
                strategy_kind=strategy_kind,
            )
            _normalize_patch_values(parsed_overrides, coin=coin)
            _validate_patch_leaf_types(
                result,
                parsed_overrides,
                coin=coin,
                origin="override_config_path",
            )
            _validate_effective_coin_config(
                result,
                parsed_overrides,
                coin=coin,
                origin="override_config_path",
            )
        inline_patch = _extract_allowed_patch(
            overrides,
            coin=coin,
            source=f"coin_overrides.{coin}",
            strict=True,
            strategy_kind=strategy_kind,
            strip_runtime_aliases=runtime_compiled,
        )
        _normalize_patch_values(inline_patch, coin=coin)
        _validate_patch_leaf_types(
            result,
            inline_patch,
            coin=coin,
            origin="inline override",
        )
        nested_update(
            parsed_overrides,
            inline_patch,
        )
        _normalize_patch_values(parsed_overrides, coin=coin)
        _validate_effective_coin_config(
            result,
            parsed_overrides,
            coin=coin,
            origin="file and inline precedence resolution",
        )
        result.setdefault("coin_overrides", {})[coin] = parsed_overrides
        log_config_message(
            verbose,
            logging.INFO,
            "Added overrides for %s: %s",
            coin,
            sort_dict_keys(parsed_overrides),
        )
    record_transform(
        result,
        "parse_overrides",
        {"coins": sorted(result.get("coin_overrides", {}).keys())},
    )
    return result


def _build_flag_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coin_flags", add_help=False)
    parser.add_argument("-sm", type=expand_PB_mode, dest="short_mode", default=None)
    parser.add_argument("-lm", type=expand_PB_mode, dest="long_mode", default=None)
    parser.add_argument("-lw", type=float, dest="WE_limit_long", default=None)
    parser.add_argument("-sw", type=float, dest="WE_limit_short", default=None)
    parser.add_argument("-lev", type=float, dest="leverage", default=None)
    parser.add_argument("-lc", type=str, dest="live_config_path", default=None)
    return parser


def expand_PB_mode(mode: str) -> str:
    lowered = mode.lower()
    if lowered in ["gs", "graceful_stop", "graceful-stop"]:
        return "graceful_stop"
    if lowered in ["m", "manual"]:
        return "manual"
    if lowered in ["n", "normal"]:
        return "normal"
    if lowered in ["p", "panic"]:
        return "panic"
    if lowered in ["t", "tp", "tp_only", "tp-only"]:
        return "tp_only"
    raise Exception(f"unknown passivbot mode {mode}")
