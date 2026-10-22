import argparse
import logging
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Callable

from pure_funcs import sort_dict_keys
from utils import looks_like_exact_market_identifier, symbol_to_coin

from .load import load_input_config, prepare_config
from .log_output import log_config_message
from .schema import get_template_config
from .coerce import normalize_hsl_signal_mode
from .field_policies import (
    CONDITIONAL_HSL_OVERRIDE_PATHS,
    OVERRIDABLE_SHARED_BOT_PATHS,
    OVERRIDABLE_STANDALONE_BOT_KEYS,
    get_live_coin_override_paths,
)
from .shared_bot import BOT_GROUP_FIELD_MAP, canonicalize_shared_bot_side
from .strategy import (
    TRAILING_GRID_V7_FLAT_ONLY_KEYS,
    get_strategy_param_keys,
    normalize_strategy_kind,
)
from .strategy_spec import get_supported_strategy_kinds
from .transform_log import record_transform


from config.access import get_optional_config_value

def apply_allowed_modifications(
    src, modifications, allowed_overrides, return_full=True
):
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
                if return_full:
                    target_dict[key] = deepcopy(mod_value)
                else:
                    target_dict[key] = deepcopy(mod_value)

    _apply_recursive(target, modifications, allowed_overrides)
    return result


def _active_shared_bot_override_paths(hsl_signal_mode: str) -> frozenset[str]:
    from .field_registry import get_coin_override_paths

    return get_coin_override_paths(normalize_hsl_signal_mode(hsl_signal_mode))


def _flat_bot_side_modification_policy(
    *, hsl_signal_mode: str = "coin"
) -> dict[str, bool]:
    active_paths = _active_shared_bot_override_paths(hsl_signal_mode)
    policy = {
        flat_key: (
            f"{group_name}.{local_key}" in active_paths
            or any(
                path.startswith(f"{group_name}.{local_key}.") for path in active_paths
            )
        )
        for group_name, field_map in BOT_GROUP_FIELD_MAP.items()
        for local_key, flat_key in field_map.items()
    }
    policy.update({key: True for key in OVERRIDABLE_STANDALONE_BOT_KEYS})
    return policy


def allowed_flat_bot_side_modification_keys(
    *, hsl_signal_mode: str = "coin"
) -> frozenset[str]:
    return frozenset(
        key
        for key, allowed in _flat_bot_side_modification_policy(
            hsl_signal_mode=hsl_signal_mode
        ).items()
        if allowed is True
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
                f"override key(s): {joined}. Use "
                f"or use coin_overrides.{coin}.bot.{pside}.strategy.<strategy_kind>.* in final v9."
            )


def _allowed_bot_side_modifications(*, hsl_signal_mode: str = "coin") -> dict:
    def _allow_dotted_paths(keys: tuple[str, ...]) -> dict:
        result = {}
        for key in keys:
            current = result
            parts = tuple(part for part in key.split(".") if part)
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = True
        return result

    active_paths = _active_shared_bot_override_paths(hsl_signal_mode)
    side = {key: True for key in OVERRIDABLE_STANDALONE_BOT_KEYS}
    side["strategy"] = {
        strategy_kind: _allow_dotted_paths(keys)
        for strategy_kind in get_supported_strategy_kinds()
        for keys in (get_strategy_param_keys(strategy_kind),)
    }
    for group_name in BOT_GROUP_FIELD_MAP:
        relative_paths = tuple(
            path.removeprefix(f"{group_name}.")
            for path in active_paths
            if path.startswith(f"{group_name}.")
        )
        if relative_paths:
            grouped_allowed = _allow_dotted_paths(relative_paths)
            side[group_name] = grouped_allowed
    return side


def get_allowed_modifications(*, hsl_signal_mode: str = "coin"):
    live = {}
    for path in get_live_coin_override_paths():
        owner = live
        for part in path[1:-1]:
            owner = owner.setdefault(part, {})
        owner[path[-1]] = True
    return {
        "bot": {
            "long": _allowed_bot_side_modifications(hsl_signal_mode=hsl_signal_mode),
            "short": _allowed_bot_side_modifications(hsl_signal_mode=hsl_signal_mode),
        },
        "live": live,
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
        if (
            key in base_dict
            and isinstance(base_dict[key], dict)
            and isinstance(value, dict)
        ):
            nested_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _unwrap_override_document(document: dict, *, source: str, allow_legacy: bool = False) -> dict:
    if not isinstance(document, dict):
        raise TypeError(f"{source} must contain a configuration object")
    nested = document.get("config")
    if allow_legacy and isinstance(nested, dict) and ("bot" in nested or "live" in nested):
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
    hsl_signal_mode: str,
) -> dict:
    """Extract explicitly supplied allowed leaves without hydrating defaults."""

    source_doc = deepcopy(_unwrap_override_document(document, source=source))
    _reject_flat_strategy_coin_overrides(source_doc, coin=coin)
    source_bot = source_doc.get("bot")
    source_portfolio = source_bot.get("portfolio", {}) if isinstance(source_bot, dict) else {}
    if isinstance(source_portfolio, dict) and "strategy_kind" in source_portfolio:
        source_strategy_kind = normalize_strategy_kind(source_portfolio["strategy_kind"])
        if source_strategy_kind != strategy_kind:
            raise ValueError(
                f"{source}.bot.portfolio.strategy_kind {source_strategy_kind!r} does not match "
                f"the portfolio strategy_kind {strategy_kind!r}"
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
            from .hsl import validate_override_paths

            validate_override_paths(
                {"bot": {"portfolio": {"hsl": {"scope": hsl_signal_mode if strict else "coin"}}}},
                {"bot": {pside: side}},
            )
            hsl_group = side.get("hsl")
            hsl_flat_keys = set(BOT_GROUP_FIELD_MAP["hsl"].values())
            has_hsl_patch = (isinstance(hsl_group, dict) and bool(hsl_group)) or any(
                key in side for key in hsl_flat_keys
            )
            if has_hsl_patch and hsl_signal_mode != "coin":
                path = _format_override_path(coin, ("bot", pside, "hsl"))
                message = (
                    f"{path} is overridable only when the effective portfolio "
                    f"bot.portfolio.hsl.scope is 'coin'; got {hsl_signal_mode!r}"
                )
                if strict:
                    raise ValueError(message)
                logging.warning("%s; the file HSL values are ignored", message)
            if not strict:
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
                    isinstance(source_portfolio, dict) and "strategy_kind" in source_portfolio
                )
                if strict or not declares_strategy_kind:
                    raise ValueError(
                        f"{source}.bot.{pside}.strategy.{mismatched[0]} cannot override active "
                        f"strategy_kind {strategy_kind!r}"
                    )
                for kind in mismatched:
                    strategy.pop(kind, None)

    allowed = get_allowed_modifications(hsl_signal_mode=hsl_signal_mode)

    def visit(value, policy, path: tuple[str, ...]):
        if policy is True:
            from .control_sentinels import FORCED_MODE_PATHS, optional_forced_mode

            if path in FORCED_MODE_PATHS:
                return optional_forced_mode(value, path=_format_override_path(coin, path))
            if value is None and path[-2:] != ("entry_cooldown", "max_duration_minutes"):
                raise TypeError(f"{_format_override_path(coin, path)} may not be null")
            return deepcopy(value)
        if not isinstance(policy, dict):
            if strict:
                raise ValueError(f"{_format_override_path(coin, path)} is not overridable")
            return _MISSING
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
            if child_value is not _MISSING and (not isinstance(child_value, dict) or child_value):
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
        unknown_roots = sorted(
            set(source_doc) - {"bot", "live", "override_config_path"}
        )
        if unknown_roots:
            raise ValueError(
                f"coin_overrides.{coin} has unsupported key(s): "
                + ", ".join(unknown_roots)
            )
    return patch


def _iter_patch_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_patch_leaves(child, path + (key,))
        return
    yield path, value


_MISSING = object()


def _get_nested_value(config: dict, path: tuple[str, ...], *, default=None):
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _validate_patch_leaf_types(
    config: dict, patch: dict, *, coin: str, origin: str
) -> None:
    template = get_template_config()
    for path, value in _iter_patch_leaves(patch):
        display_path = f"{_format_override_path(coin, path)} ({origin})"
        from .control_sentinels import FORCED_MODE_PATHS, optional_forced_mode

        if path in FORCED_MODE_PATHS:
            optional_forced_mode(value, path=display_path)
            continue
        reference = _get_nested_value(config, path)
        if reference is None:
            reference = _get_nested_value(template, path)
        if value is None and path[-2:] == ("entry_cooldown", "max_duration_minutes"):
            continue
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
                ("live", "modes", "long"),
                ("live", "modes", "short"),
            }:
                if value:
                    try:
                        expand_PB_mode(value)
                    except Exception as exc:
                        raise ValueError(
                            f"{display_path} has invalid mode {value!r}"
                        ) from exc
            elif not isinstance(reference, str):
                raise TypeError(
                    f"{display_path} must be numeric or boolean, not a string"
                )
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
                raise ValueError(f"{display_path} must be finite and >= 0.0")
    leverage = patch.get("live", {}).get("account", {}).get("leverage")
    if leverage is not None and float(leverage) <= 0.0:
        raise ValueError(
            f"coin_overrides.{coin}.live.account.leverage ({origin}) must be > 0.0"
        )


def _validate_effective_coin_config(
    config: dict,
    patch: dict,
    *,
    coin: str,
    origin: str,
) -> dict:
    effective = deepcopy(config)
    for metadata_key in (
        "_coins_sources",
        "_raw",
        "_raw_effective",
        "_transform_log",
    ):
        effective.pop(metadata_key, None)
    # Runtime dataset metadata has no role in validating a coin's policy.
    # Normal authored loading rejects these fields before this internal check.
    effective.pop("disable_plotting", None)
    for metadata_key in ("coins", "cache_dir"):
        effective.get("backtest", {}).pop(metadata_key, None)
    # Validate this merged coin policy without recursively preparing stored
    # scenarios. The owning config and each raw file keep their full validation.
    effective.setdefault("backtest", {})["scenarios"] = []
    effective["coin_overrides"] = {}
    # The parent validates fixed coin patches against its complete coin map.
    # This isolated policy deliberately has no map, so do not reapply those
    # separately owned patches while validating one merged coin document.
    fixed = effective.get("optimize", {}).get("fixed_runtime_overrides", {})
    effective.get("optimize", {})["fixed_runtime_overrides"] = {
        key: value for key, value in fixed.items()
        if key != "coin_overrides" and not key.startswith("coin_overrides.")
    }
    for pside in ("long", "short"):
        canonicalize_shared_bot_side(
            effective.get("bot", {}).get(pside),
            path_prefix=("bot", pside),
            seed_missing_groups=False,
        )
    from .hsl import normalization_template, validate_override_paths

    validate_override_paths(effective, patch)
    validation_input = normalization_template(get_template_config(), effective)
    nested_update(validation_input, effective)
    effective = validation_input
    canonical_base = deepcopy(effective)
    for root in ("bot", "live"):
        if root in patch:
            effective_patch = deepcopy(patch[root])
            if root == "bot":
                for side in ("long", "short"):
                    effective_patch.get(side, {}).pop("wallet_exposure_limit", None)
            nested_update(effective[root], effective_patch)
    # This effective config represents one coin. Its pinned cooldown leaves
    # cannot be changed by global optimizer genes, so omit those dimensions
    # while validating the merged policy and the remaining search corners.
    from .optimize_bounds import flatten_optimize_bounds
    from .param_paths import resolve_optimizer_key_path

    pinned_cooldown_paths = {
        path for path, _ in _iter_patch_leaves(patch)
        if len(path) >= 4 and path[0] == "bot" and path[2] == "entry_cooldown"
    }
    if pinned_cooldown_paths:
        bounds = flatten_optimize_bounds(
            effective["optimize"]["bounds"],
            strategy_kind=get_optional_config_value(effective, 'bot.portfolio.strategy_kind'),
        )
        from .optimize_bounds import set_flat_optimize_bound

        effective["optimize"]["bounds"] = {}
        for key, value in bounds.items():
            if resolve_optimizer_key_path(effective, key) not in pinned_cooldown_paths:
                set_flat_optimize_bound(effective["optimize"]["bounds"], effective['bot']["portfolio"]['strategy_kind'], key, value)
    try:
        prepared = prepare_config(
            effective,
            base_config_path=str(
                effective.get("live", {}).get("base_config_path") or ""
            ),
            live_only=False,
            verbose=False,
            log_config_transforms=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise type(exc)(
            f"coin_overrides.{coin} produces an invalid config after {origin}: {exc}"
        ) from exc
    normalized_patch = {}
    for path, original_value in _iter_patch_leaves(patch):
        normalized_value = _get_nested_value(prepared, path, default=_MISSING)
        if normalized_value is _MISSING:
            # wallet_exposure_limit is a per-coin runtime value calculated after
            # canonical config preparation, so it has no canonical global leaf.
            if (
                len(path) == 3
                and path[0] == "bot"
                and path[2] == "wallet_exposure_limit"
            ):
                normalized_value = original_value
            else:
                raise ValueError(
                    f"coin_overrides.{coin} normalization lost {'.'.join(path)} after {origin}"
                )
        set_nested_value_safe(
            normalized_patch,
            list(path),
            deepcopy(normalized_value),
            create_missing=True,
        )
    return normalized_patch


_OVERRIDE_FILE_STACK = ContextVar("override_file_validation_stack", default=())


@contextmanager
def _override_file_validation(path):
    resolved = os.path.normcase(os.path.realpath(path))
    stack = _OVERRIDE_FILE_STACK.get()
    if resolved in stack:
        raise ValueError("cyclic override_config_path: " + " -> ".join((*stack, resolved)))
    token = _OVERRIDE_FILE_STACK.set((*stack, resolved))
    try:
        yield
    finally:
        _OVERRIDE_FILE_STACK.reset(token)


def _validated_file_patch(document, *, config, coin, source, resolved_path):
    """Validate the source config, then select only its explicitly authored pins."""
    from .authored import validate_authored_config
    from .migrations.v9 import require_v9_runtime_config

    raw = _unwrap_override_document(document, source=source)
    require_v9_runtime_config(config, additional_inputs=(raw,))
    if "config_version" in raw:
        require_v9_runtime_config(raw)
    strategy_kind = normalize_strategy_kind(get_optional_config_value(config, 'bot.portfolio.strategy_kind'))
    source_bot = raw.get("bot")
    source_portfolio = source_bot.get("portfolio", {}) if isinstance(source_bot, dict) else {}
    if isinstance(source_portfolio, dict) and "strategy_kind" in source_portfolio:
        if normalize_strategy_kind(source_portfolio["strategy_kind"]) != strategy_kind:
            raise ValueError(f"{source}.bot.portfolio.strategy_kind {source_portfolio['strategy_kind']!r} "
                             f"does not match the portfolio strategy_kind {strategy_kind!r}")

    validation = deepcopy(raw)
    # This is a per-coin extension, not a global canonical bot parameter.
    # Validate it before removing exactly this known leaf from the config clone.
    wallet_patch = {}
    for side in ("long", "short"):
        node = validation.get("bot", {})
        node = node.get(side) if isinstance(node, dict) else None
        if isinstance(node, dict) and "wallet_exposure_limit" in node:
            wallet_patch.setdefault("bot", {})[side] = {
                "wallet_exposure_limit": node.pop("wallet_exposure_limit")
            }
    _validate_patch_leaf_types(config, wallet_patch, coin=coin, origin=source)
    validation.setdefault("config_version", config["config_version"])
    validate_authored_config(validation, get_template_config(), policy_context=config)
    validation.setdefault("bot", {}).setdefault("portfolio", {}).setdefault("strategy_kind", strategy_kind)

    with _override_file_validation(resolved_path):
        # Recursive file references remain validated. The active path stack
        # rejects cycles and unwinds even when preparation fails.
        prepare_config(validation, base_config_path=resolved_path, verbose=False,
                       log_config_transforms=False)
        patch = _extract_allowed_patch(
            _project_declared_file_defaults(raw, strategy_kind=strategy_kind),
            coin=coin, source=source, strict=False, strategy_kind=strategy_kind,
            hsl_signal_mode=normalize_hsl_signal_mode(config.get("bot", {}).get("portfolio", {}).get("hsl", {}).get("scope", "coin")),
        )
        _validate_patch_leaf_types(config, patch, coin=coin, origin=source)
        return _validate_effective_coin_config(config, patch, coin=coin, origin=source)


def _project_declared_file_defaults(document, *, strategy_kind):
    """Project only declared, coin-eligible common file values onto each side.

    A referenced config is validated as a config first. Its global defaults may
    supply coin parameter leaves, but neither hydration nor this projection can
    authorize portfolio, activation or universe changes. Inline patches keep
    their stricter, side-only permission boundary.
    """
    result = deepcopy(document)
    bot = result.get("bot", {})
    common = bot.get("global", {}) if isinstance(bot, dict) else {}
    if not isinstance(common, dict):
        return result  # The authored validator reports malformed config trees.
    for side in ("long", "short"):
        allowed = deepcopy(get_allowed_modifications(hsl_signal_mode="coin")["bot"][side])
        allowed["strategy"] = {strategy_kind: allowed["strategy"][strategy_kind]}
        inherited = apply_allowed_modifications({}, common, allowed, return_full=False)
        if inherited:
            explicit = bot.get(side, {})
            bot[side] = nested_update(inherited, deepcopy(explicit))
    return result


def _resolve_override_config_path(config, coin):
    path = config.get("coin_overrides", {}).get(coin, {}).get("override_config_path")
    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        raise TypeError(
            f"coin_overrides.{coin}.override_config_path must be a non-empty string"
        )
    path = path.strip()
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        base_config_path = config.get("live", {}).get("base_config_path")
        if base_config_path:
            candidates.append(os.path.join(os.path.dirname(base_config_path), path))
        candidates.append(path)
    resolved_path = next(
        (candidate for candidate in candidates if os.path.isfile(candidate)), None
    )
    if resolved_path is None:
        attempted = ", ".join(os.path.abspath(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"coin_overrides.{coin}.override_config_path not found; tried: {attempted}"
        )
    return os.path.realpath(resolved_path)


def load_override_config(
    config,
    coin,
    *,
    config_loader: Callable[[str], dict] | None = None,
):
    resolved_path = _resolve_override_config_path(config, coin)
    if resolved_path is None:
        return {}
    if config_loader is not None:
        return config_loader(resolved_path)
    # Validate the document as a configuration, but return its raw explicit values.
    try:
        source, base_config_path, raw_snapshot = load_input_config(
            resolved_path, log_info=False
        )
        if not isinstance(source, dict):
            raise TypeError("configuration root must be an object")
        _validated_file_patch(
            source, config=config, coin=coin,
            source=f"coin_overrides.{coin}.override_config_path", resolved_path=resolved_path,
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
    if (
        not isinstance(config, dict)
        or "live" not in config
        or "coin_flags" not in config["live"]
    ):
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
                set_nested_value_safe(
                    result[coin], key_map[key], value, create_missing=True
                )
    return result


def normalize_coin_override_keys(
    coin_overrides, *, symbol_normalizer=None, verbose=True
):
    """Normalize authored keys without loading files or changing patch precedence."""
    if not isinstance(coin_overrides, dict):
        raise TypeError("coin_overrides must be a dict")
    if symbol_normalizer is None:
        symbol_normalizer = symbol_to_coin
    normalized_overrides = {}
    normalized_sources = {}
    for coin, overrides in coin_overrides.items():
        if not isinstance(coin, str):
            raise TypeError("coin_overrides keys must be strings")
        formatted_coin = (
            coin
            if looks_like_exact_market_identifier(coin)
            else symbol_normalizer(coin)
        )
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
    return normalized_overrides


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
    runtime_compiled = any(
        isinstance(item, dict) and item.get("step") == "compile_runtime_config"
        for item in result.get("_transform_log", [])
    )
    if runtime_compiled:
        raise ValueError(
            "parse_overrides must run before compile_runtime_config so explicit coin "
            "override keys remain distinguishable from generated runtime aliases"
        )
    if "coin_overrides" in result and not isinstance(result["coin_overrides"], dict):
        raise TypeError("coin_overrides must be a dict")
    result["coin_overrides"] = normalize_coin_override_keys(
        result["coin_overrides"], symbol_normalizer=symbol_normalizer, verbose=verbose
    )
    strategy_kind = normalize_strategy_kind(get_optional_config_value(result, 'bot.portfolio.strategy_kind'))
    live_config = result.get("live", {})
    hsl_signal_mode = normalize_hsl_signal_mode(
        result["bot"]["portfolio"]["hsl"]["scope"]
    )
    for coin, overrides in result["coin_overrides"].items():
        parsed_overrides = {}
        loaded = override_loader(result, coin)
        if loaded:
            if override_loader is load_override_config:
                resolved_path = _resolve_override_config_path(result, coin)
            else:
                reference = str(overrides.get("override_config_path") or "")
                base_path = str(live_config.get("base_config_path") or "")
                resolved_path = (os.path.join(os.path.dirname(base_path), reference)
                                 if reference and not os.path.isabs(reference) and base_path
                                 else reference or base_path or os.getcwd())
            parsed_overrides = _validated_file_patch(
                loaded,
                config=result,
                coin=coin,
                source=f"coin_overrides.{coin}.override_config_path",
                resolved_path=resolved_path,
            )
            _validate_patch_leaf_types(
                result,
                parsed_overrides,
                coin=coin,
                origin="override_config_path",
            )
            parsed_overrides = _validate_effective_coin_config(
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
            hsl_signal_mode=hsl_signal_mode,
        )
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
        parsed_overrides = _validate_effective_coin_config(
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
    from .control_sentinels import MODE_ALIASES

    lowered = mode.lower()
    if lowered in MODE_ALIASES:
        return MODE_ALIASES[lowered]
    raise Exception(f"unknown passivbot mode {mode}")
