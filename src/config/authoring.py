"""Offline authoring views and explanations, separate from authored policy.

The existing loader and override validators own accepted inputs and precedence.
These helpers never query markets or accounts and never add report metadata to
configuration objects.
"""

from copy import deepcopy
from pathlib import Path
import os

from .field_registry import get_field_registry
from .field_spec import NO_DEFAULT
from .load import prepare_config
from .overrides import parse_overrides, normalize_coin_override_keys
from .param_paths import resolve_dotted_config_path
from .result_records import extract_result_config
from .scenario_overrides import normalize_scenario_overrides
from .universe import COIN_SOURCE_FIELDS, require_side_universe_sources, set_side_universe_sources


_MISSING = object()
_UNIVERSE_PATHS = tuple(("bot", side, "universe", field)
                        for side in ("long", "short") for field in sorted(COIN_SOURCE_FIELDS))
_PROTECTED = (
    ("coin_overrides",), ("optimize", "bounds"),
    ("optimize", "fixed_params"), ("optimize", "fixed_runtime_overrides"),
    ("optimize", "enable_overrides"), ("optimize", "limits"), ("optimize", "scoring"),
    ("backtest", "scenarios"), *_UNIVERSE_PATHS,
    *(("bot", side, "forager", "always_include") for side in ("long", "short")),
    ("bot", "global", "hsl"),
)


def _get(mapping, path, default=_MISSING):
    current = mapping
    for part in path:
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, TypeError, IndexError):
                return default
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _set(mapping, path, value):
    current = mapping
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = deepcopy(value)


def _patch(config, overrides):
    result = deepcopy(config)
    for selector, value in normalize_scenario_overrides(overrides).items():
        _set(result, resolve_dotted_config_path(result, selector), value)
    return result


def _authored(source, base_config_path):
    from tools.clean_config import cleanup_config
    from config_utils import strip_config_metadata

    payload = extract_result_config(source, allow_unmarked_wrapper=True)
    # Validate the complete input before stripping metadata or pruning defaults.
    full = cleanup_config(payload, base_config_path=base_config_path, _allow_sparse=True)
    payload = strip_config_metadata(payload)
    payload.get("live", {}).pop("base_config_path", None)
    return payload, full


def _rebase_override_files(config, source, destination):
    if not source or not destination:
        return config
    base = Path(source).resolve().parent
    output = Path(destination).resolve().parent

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key.split(".")[-1] == "override_config_path" and isinstance(value, str):
                    reference = Path(value)
                    if not reference.is_absolute():
                        candidates = (base / reference, Path.cwd() / reference)
                        resolved = next((path.resolve() for path in candidates if path.is_file()), None)
                        if resolved is None:
                            raise FileNotFoundError(f"override_config_path not found: {value}")
                        node[key] = os.path.relpath(resolved, output)
                else:
                    visit(value)

    visit(config)
    return config


def compact_config(source, *, base_config_path="", destination_path=""):
    """Remove redundant authored defaults while retaining policy authority.

    Eligibility, search settings, sparse patches, scenarios, explicit nulls and
    optional HSL authorization remain explicit. The normal loader verifies the
    entire result; context-dependent defaults conservatively retain the source.
    Relative coin-override file references follow the output location.
    """
    from tools.clean_config import cleanup_config

    authored, full = _authored(source, base_config_path)
    baseline = {"config_version": authored["config_version"]}
    if "bot" in authored:
        baseline["bot"] = {side: {} for side in ("long", "short") if side in authored["bot"]}
    if "bot" in authored:
        baseline["bot"]["global"] = {"strategy_kind": full["bot"]["global"]["strategy_kind"]}
    if "live" in authored:
        baseline["live"] = {}
    if full["bot"]["global"]["hsl"]["scope"] == "portfolio":
        baseline["bot"].setdefault("global", {})["hsl"] = deepcopy(full["bot"]["global"]["hsl"])
    defaults = cleanup_config(baseline, base_config_path=base_config_path, _allow_sparse=True)
    registry = get_field_registry()
    keep_empty = {("bot",), ("live",), ("bot", "long"), ("bot", "short")}

    def prune(value, path=()):
        if value is None or path == ("config_version",) or any(
            path[:len(prefix)] == prefix for prefix in _PROTECTED
        ):
            return deepcopy(value)
        if isinstance(value, dict):
            result = {}
            for name, child in value.items():
                selected = prune(child, (*path, name))
                if selected is not _MISSING:
                    result[name] = selected
            return result if result or path in keep_empty or not path else _MISSING
        field = registry.find(path)
        if (field is not None and field.default is not NO_DEFAULT
                and value == field.default and value == _get(defaults, path)):
            return _MISSING
        return deepcopy(value)

    result = prune(authored)
    if cleanup_config(result, base_config_path=base_config_path, _allow_sparse=True) != full:
        # A default may depend on another authored field. Retain the complete
        # authored authority rather than guessing which dependency can be omitted.
        result = deepcopy(authored)
    return _rebase_override_files(result, base_config_path, destination_path)


def _resolve_coin_sources(config, *, require_complete):
    from utils import _resolve_coins_file_path, normalize_coins_source

    result = deepcopy(config)
    unresolved = []

    def check_source(value):
        if isinstance(value, dict):
            for child in value.values():
                check_source(child)
        elif isinstance(value, list):
            if len(value) == 1:
                check_source(value[0])
        elif isinstance(value, str):
            file_like = value.startswith(("./", "../", "/", "~")) or Path(value).suffix.lower() in {
                ".json", ".hjson", ".txt",
            }
            if file_like and _resolve_coins_file_path(value) is None:
                raise FileNotFoundError(f"coin-list file not found: {value}")

    for name in ("approved_coins", "ignored_coins"):
        value = require_side_universe_sources(result, name)
        check_source(value)
        resolved = normalize_coins_source(value, allow_all=name == "approved_coins")
        for side, coins in resolved.items():
            if coins == ["all"]:
                unresolved.append(f"bot.{side}.universe.{name}")
            elif require_complete and any(
                _resolve_coins_file_path(coin) is not None for coin in coins
            ):
                raise ValueError(
                    "a coin-list file remains unresolved inside a multi-item source; "
                    "use a single list-file source or explicit coin identifiers"
                )
        set_side_universe_sources(result, name, resolved)
    if require_complete and unresolved:
        raise ValueError(
            "offline resolved export cannot expand approved_coins='all'; "
            "supply explicit coin lists (market metadata would be required)"
        )
    return result, unresolved


def _resolved_context(config, *, base_config_path, require_complete=True):
    prepared = prepare_config(
        config, base_config_path=base_config_path, live_only=True,
        verbose=False, log_config_transforms=False,
    )
    prepared, unresolved = _resolve_coin_sources(prepared, require_complete=require_complete)
    return parse_overrides(prepared, verbose=False), unresolved


def resolved_config(source, *, base_config_path=""):
    """Export explicit defaults, local coin-list contents and inline coin pins.

    Dates and null inheritance retain their authored semantics. This is a local
    policy snapshot, not exchange capability discovery or a prepared dataset.
    """
    from tools.clean_config import cleanup_config

    _, full = _authored(source, base_config_path)
    resolved, _ = _resolved_context(full, base_config_path=base_config_path)
    result = cleanup_config(resolved, base_config_path=base_config_path)
    # Resolve each patch in its effective root context, preserving its original
    # sparse paths instead of copying inherited root values into the patch.
    patches = [(result["optimize"], "fixed_runtime_overrides")]
    patches.extend((scenario, "overrides") for scenario in result["backtest"]["scenarios"]
                   if "overrides" in scenario)
    for owner, name in patches:
        raw = normalize_scenario_overrides(owner[name])
        if not raw:
            continue
        effective, _ = _resolved_context(_patch(full, raw), base_config_path=base_config_path)
        for selector in list(raw):
            parts = tuple(selector.split("."))
            if parts in _UNIVERSE_PATHS:
                raw[selector] = deepcopy(_get(effective, parts))
            elif selector == "coin_overrides":
                raw[selector] = deepcopy(effective["coin_overrides"])
        owner[name] = raw
    return cleanup_config(result, base_config_path=base_config_path)


def _intersects(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _bound_target(config, path):
    """Reuse the canonical optimizer writer to identify a bounds leaf."""
    if path[:2] != ("optimize", "bounds"):
        return None
    from .optimize_bounds import flatten_optimize_bounds, set_flat_optimize_bound
    from .param_paths import iter_optimizer_key_paths

    selected = path[2:]
    if selected and str(selected[-1]).isdigit():
        selected = selected[:-1]
    kind = config["bot"]["global"]["strategy_kind"]
    bounds = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=kind)
    for key, target in iter_optimizer_key_paths(config, bounds):
        singleton = {}
        set_flat_optimize_bound(singleton, kind, key, bounds[key])
        selected_bound = _get(singleton, selected)
        if isinstance(selected_bound, list) or type(selected_bound) in (int, float):
            return target
    return None


def _event_path(event):
    return tuple(event.get("path_parts", event["path"].split(".")))


def _search_treatment(config, path, authored, *, coin_patch=None, coin=None, applied_overrides=()):
    from .optimize_bounds import flatten_optimize_bounds
    from .param_paths import iter_optimizer_key_paths, resolve_bound_selectors
    from optimization.bounds import Bound

    bounds = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=config["bot"]["global"]["strategy_kind"])
    matches = [(key, bounds[key]) for key, target in iter_optimizer_key_paths(config, bounds) if target == path]
    if not matches:
        return {"status": "not_in_bounds"}
    key, raw_bound = matches[0]
    try:
        bound = Bound.from_config(key, raw_bound)
    except Exception as exc:
        raise ValueError(f"invalid optimize bound {key!r}: {exc}") from exc
    selectors = config["optimize"].get("fixed_params", [])
    fixed_by = [selector for selector in selectors
                if key in resolve_bound_selectors(config, [selector], bounds)]
    authored_bounds = flatten_optimize_bounds(
        authored.get("optimize", {}).get("bounds", {}), strategy_kind=config["bot"]["global"]["strategy_kind"],
    )
    status = ("fixed_selector" if fixed_by else
              "fixed_bound" if key in authored_bounds and bound.low == bound.high else
              "omitted_fixed" if bound.low == bound.high else "searched")
    gene_status = status
    shadowed_by = [event for event in applied_overrides
                   if _intersects(path, _event_path(event))]
    if shadowed_by:
        status = shadowed_by[-1]["kind"]
    if coin_patch is not None and _get(coin_patch, path) is not _MISSING:
        coin_events = [event for event in applied_overrides if
                       _intersects(("coin_overrides", coin, *path), _event_path(event))]
        status = coin_events[-1]["kind"] if coin_events else "coin_override"
        shadowed_by.extend(coin_events)
    from optimizer_overrides import COUPLED_UNSTUCK_EMA_BOUND_KEYS, unstuck_ema_spans_coupled
    independent = not (key in COUPLED_UNSTUCK_EMA_BOUND_KEYS and unstuck_ema_spans_coupled(config))
    if not independent:
        gene_status = "not_in_genome"
    return {
        "status": status, "gene_status": gene_status,
        "independent_gene": independent,
        "shadowed_by": [{"kind": event["kind"], "path": event["path"]} for event in shadowed_by],
        "key": key, "bounds": deepcopy(raw_bound),
        "bounds_source": "authored" if key in authored_bounds else
                         "configured_value" if bound.low == bound.high else "loader_default_or_derived",
        "fixed_selectors": fixed_by,
    }


def explain_config(source, path, *, base_config_path="", coin=None, scenario=None, optimize=False):
    """Return a report envelope without adding provenance to authored policy."""
    authored, full = _authored(source, base_config_path)
    parts = resolve_dotted_config_path(full, path)
    if parts is None or not parts:
        raise ValueError("configuration path must not be empty")
    if coin is not None and parts[:1] == ("coin_overrides",):
        raise ValueError("--coin expects a root setting path, not a coin_overrides path")
    if coin is not None:
        coin = next(iter(normalize_coin_override_keys({coin: {}}, verbose=False)))
    bound_target = _bound_target(full, parts)
    query_target = bound_target or parts
    events = []
    root_value = _get(authored, parts)
    if root_value is not _MISSING:
        events.append({"kind": "authored", "path": ".".join(parts), "value": deepcopy(root_value)})
    else:
        events.append({"kind": "default_or_derived", "path": ".".join(parts)})
        if (len(parts) == 4 and parts[0] == "bot" and parts[1] in {"long", "short"}
                and parts[2:] == ("risk", "total_wallet_exposure_limit")
                and "bot" in authored and parts[1] not in authored["bot"]):
            events[-1]["rule"] = "omitted_side_disabled"
    context = deepcopy(full)
    dataset_selection = None
    applied_overrides = []

    def trace(name, target, dependencies, value, operation):
        event = {"kind": "optimizer_transform", "name": name,
                 "path": ".".join(target), "path_parts": list(target),
                 "dependencies": [{"path": ".".join(source), "path_parts": list(source)}
                                  for source in dependencies],
                 "operation": operation, "value": deepcopy(value)}
        applied_overrides.append(event)
        selected_target = ("coin_overrides", coin, *query_target) if coin is not None else query_target
        if _intersects(selected_target, target) or _intersects(query_target, target):
            events.append(event)

    def apply(raw, kind):
        nonlocal context
        normalized = normalize_scenario_overrides(raw)
        for selector, value in normalized.items():
            target = resolve_dotted_config_path(context, selector)
            event = {"kind": kind, "path": ".".join(target), "value": deepcopy(value)}
            applied_overrides.append(event)
            if _intersects(parts, target):
                events.append(event)
        context = _patch(context, normalized)

    if optimize:
        from optimizer_overrides import optimizer_overrides

        apply(context["optimize"]["fixed_runtime_overrides"], "fixed_runtime_override")
        transformations = context["optimize"]["enable_overrides"]
        context = optimizer_overrides(transformations, context, None, trace=trace)
        for side in ("long", "short"):
            context = optimizer_overrides(transformations, context, side, trace=trace)
    if scenario is not None:
        selected = [item for item in context["backtest"]["scenarios"] if item.get("label") == scenario]
        if len(selected) != 1:
            raise ValueError(f"scenario {scenario!r} must match exactly one explicit label")
        selected = selected[0]
        dataset_selection = {name: deepcopy(selected[name]) for name in (
            "coins", "ignored_coins", "exchanges", "coin_sources",
        ) if name in selected}
        if any(_intersects(parts, prefix) for prefix in (
            *_UNIVERSE_PATHS,
            ("backtest", "coin_sources"),
        )):
            raise ValueError(
                "scenario eligibility depends on a prepared dataset; inspect the "
                "backtest.scenarios record directly instead of guessing dataset membership"
            )
        date_patch = {f"backtest.{name}": selected[name] for name in ("start_date", "end_date")
                      if selected.get(name)}
        if selected.get("exchanges"):
            from suite_runner import _coerce_exchange_list

            date_patch["backtest.exchanges"] = _coerce_exchange_list(selected["exchanges"])
        apply(date_patch, "scenario_selection")
        apply(selected.get("overrides", {}), "scenario_override")
        if optimize:
            from optimizer_overrides import apply_coupled_unstuck_ema_spans

            context = apply_coupled_unstuck_ema_spans(context, trace=trace)
    before_files = deepcopy(context)
    effective, unresolved = _resolved_context(context, base_config_path=base_config_path, require_complete=False)
    # The same non-live loader resolves null inheritance, ignored lists and
    # dynamic dates without fetching market data.
    effective = prepare_config(effective, base_config_path=base_config_path, verbose=False, log_config_transforms=False)
    for prefix in _UNIVERSE_PATHS:
        if not _intersects(parts, prefix):
            continue
        from utils import _resolve_coins_file_path

        def references(value):
            if isinstance(value, list) and len(value) == 1:
                yield from references(value[0])
            elif isinstance(value, str):
                resolved_file = _resolve_coins_file_path(value)
                if resolved_file is not None:
                    yield str(resolved_file)

        for reference in sorted(set(references(_get(before_files, prefix)))):
            events.append({"kind": "coin_list_file", "path": ".".join(prefix), "file": reference})
    if parts == ("backtest", "filter_by_min_effective_cost") and _get(context, parts) is None:
        from .policy import effective_minimum_cost_filter

        value = effective_minimum_cost_filter(effective)
        events.append({"kind": "inherited", "path": "bot.global.universe.filter_by_min_effective_cost",
                       "value": value})
        # Explain the effective simulation policy without changing authored or
        # resolved exports, which preserve explicit null inheritance.
        effective["backtest"]["filter_by_min_effective_cost"] = value
    coin_patch = None
    if coin is not None:
        from .overrides import nested_update
        # Reuse authored-key normalization: exact markets and venue-qualified
        # identifiers must survive without a heuristic underlying substitution.
        coin_patch = effective["coin_overrides"].get(coin, {})
        inline = normalize_coin_override_keys(before_files["coin_overrides"], verbose=False).get(coin, {})
        value = _get(coin_patch, parts)
        if value is not _MISSING:
            inline_value = _get(inline, parts)
            generated = [event for event in applied_overrides if
                         _intersects(("coin_overrides", coin, *parts), _event_path(event))]
            event = (deepcopy(generated[-1]) if generated and generated[-1]["kind"] == "optimizer_transform" else
                     {"kind": "coin_override" if inline_value is not _MISSING else "override_file",
                     "path": ".".join(("coin_overrides", coin, *parts)),
                     "path_parts": ["coin_overrides", coin, *parts], "value": deepcopy(value)}
                     )
            if event["kind"] == "override_file":
                from .overrides import _resolve_override_config_path

                file_context = deepcopy(before_files)
                file_context["live"]["base_config_path"] = base_config_path
                event["file"] = _resolve_override_config_path(file_context, coin)
            if not events or events[-1] != event:
                events.append(event)
        effective = nested_update(deepcopy(effective), coin_patch)
    value = _get(effective, parts)
    if value is _MISSING:
        raise KeyError(f"unknown or inactive configuration path {path!r}")
    for event in reversed(events):
        if event["kind"] in {"inherited", "coin_list_file"}:
            # These events explain a source resolution already; comparing its
            # authored null/path with the resolved value invents normalization.
            break
        event_path = _event_path(event)
        if coin is not None and event_path[:2] == ("coin_overrides", coin):
            event_path = event_path[2:]
        if "value" not in event or parts[:len(event_path)] != event_path:
            continue
        projected = _get(event["value"], parts[len(event_path):])
        if projected is not _MISSING and projected != value:
            events.append({"kind": "normalization", "path": ".".join(parts), "value": deepcopy(value)})
        break
    registry = get_field_registry()
    field = registry.find(parts)
    inherited_field = None
    search_path = parts
    search_coin_patch = coin_patch
    if parts[:1] == ("coin_overrides",) and len(parts) > 2:
        inherited_field = registry.find(parts[2:])
        search_path = parts[2:]
        search_coin_patch = effective["coin_overrides"].get(parts[1], {})
    if field is None and coin is not None:
        field = registry.find(("coin_overrides", coin, *parts))
    parameter_field = registry.find(bound_target) if bound_target is not None else None
    default = field.default if field is not None else NO_DEFAULT
    if bound_target is not None:
        from .schema import get_template_config

        default = _get(get_template_config(), parts, NO_DEFAULT)
    metadata_field = field or inherited_field or parameter_field
    types = ([kind.value for kind in metadata_field.types] +
             (["null"] if metadata_field.nullable else [])) if metadata_field else None
    if bound_target is not None and isinstance(value, list):
        types = ["array"]
    return {
        "path": ".".join(parts), "value": deepcopy(value),
        "context": {"coin": coin, "scenario": scenario, "optimize": optimize,
                    "scope": "offline_policy"},
        "types": types,
        "unit": metadata_field.unit if metadata_field else None,
        "denominator": metadata_field.denominator if metadata_field else None,
        "scope": metadata_field.scope if metadata_field else None,
        "default": {"declared": default is not NO_DEFAULT, **({"value": deepcopy(default)} if default is not NO_DEFAULT else {})},
        "sources": events, "search": _search_treatment(
            full, bound_target or search_path, authored, coin_patch=search_coin_patch,
            coin=coin or (parts[1] if parts[:1] == ("coin_overrides",) and len(parts) > 2 else None),
            applied_overrides=applied_overrides,
        ),
        "unresolved": unresolved,
        "dataset_selection": dataset_selection,
        "bound_target": ".".join(bound_target) if bound_target is not None else None,
        "bound_element_types": [kind.value for kind in parameter_field.types] if parameter_field else None,
    }
