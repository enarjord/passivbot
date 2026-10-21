"""Explicit structural conversion from released v8 layouts to final v9."""

from copy import deepcopy
import math
import re


_RETIRED_WEIGHTED_METRIC = re.compile(r"_w(?:_|$)", re.IGNORECASE)


def replace_retired_metrics(source, *, replacements, acknowledge, changes, best_effort=False):
    """Require authored, acknowledged formula choices rather than metric aliases."""
    replacements = dict(replacements or {})
    used = set()
    ew_requested = False

    def convert(name, path):
        from config.ew import is_ew_metric

        nonlocal ew_requested
        if not isinstance(name, str) or not _RETIRED_WEIGHTED_METRIC.search(name):
            ew_requested |= isinstance(name, str) and is_ew_metric(name)
            return name
        inferred = name not in replacements and best_effort
        if inferred:
            candidate = re.sub(r"_w(?=_|$)", "_ew", name, flags=re.IGNORECASE)
            if not is_ew_metric(candidate):
                raise ValueError(f"{path}: no supported _ew replacement for {name!r}; supply --metric-replacement explicitly")
        elif name not in replacements:
            raise ValueError(f"{path}: retired metric {name!r} requires --metric-replacement {name}=NEW and an explicit formula choice")
        if not acknowledge and not best_effort:
            raise ValueError(f"{path}: metric formula replacement requires --acknowledge-metric-formula-change")
        new = candidate if inferred else replacements[name]
        if not isinstance(new, str) or not new or _RETIRED_WEIGHTED_METRIC.search(new):
            raise ValueError(f"replacement for {name!r} must name a supported non-retired metric")
        used.add(name)
        ew_requested |= is_ew_metric(new)
        if inferred or (best_effort and not acknowledge):
            from .best_effort_v9 import record_guess

            record_guess(changes, path, name, new,
                         ("adopt exponential recency formula; retired _w formula is not equivalent"
                          if inferred else "accept explicit replacement formula under best-effort consent"))
        else:
            changes.append(dict(path=path, old=name, new=new,
                                reason="acknowledged explicit metric formula replacement"))
        return new

    def visit(node, path=""):
        if isinstance(node, list):
            for index, value in enumerate(node):
                current = f"{path}[{index}]"
                if isinstance(value, str) and path.split(".")[-1] in {"scoring", "visible_metrics"}:
                    node[index] = convert(value, current)
                else:
                    visit(value, current)
            return
        if not isinstance(node, dict):
            return
        metric_map = path.split(".")[-1] in {"reducer", "limits"}
        for key in list(node):
            current = f"{path}.{key}" if path else key
            if key == "metric":
                node[key] = convert(node[key], current)
            elif metric_map:
                prefix = next((prefix for prefix in ("penalize_if_greater_than_", "penalize_if_lower_than_")
                               if key.startswith(prefix)), "")
                metric = key.removeprefix(prefix)
                renamed = prefix + convert(metric, current)
                if renamed != key:
                    value = node.pop(key)
                    if renamed in node and node[renamed] != value:
                        raise ValueError(f"{current} conflicts with selected metric {renamed}")
                    node[renamed] = value
            else:
                visit(node[key], current)
    visit(source)
    if best_effort:
        from .best_effort_v9 import record_guess

        backtest = source.setdefault("backtest", {})
        if ew_requested and backtest.get("ew_half_life_days") is None:
            record_guess(changes, "backtest.ew_half_life_days", None, 180.0,
                         "enable inferred _ew analysis with a 180-day half-life; re-evaluate objectives and limits")
            backtest["ew_half_life_days"] = 180.0
    unused = set(replacements) - used
    if unused:
        raise ValueError(f"unused metric replacements: {', '.join(sorted(unused))}")


def retire_deap_policy(config, *, acknowledge, changes, best_effort=False):
    """Preserve effective pymoo policy; DEAP algorithm changes require consent."""
    optimize = config.get("optimize", {})
    legacy_backend = optimize.get("backend", "pymoo")
    pymoo = optimize.setdefault("pymoo", {})
    shared = pymoo.setdefault("shared", {})
    if "mutation_prob_var" in shared:
        value = shared.pop("mutation_prob_var")
        if "mutation_prob" in shared and shared["mutation_prob"] != value:
            raise ValueError("optimize.pymoo.shared.mutation_prob_var conflicts with mutation_prob")
        shared["mutation_prob"] = value
        changes.append(dict(path="optimize.pymoo.shared.mutation_prob_var", old=value,
                            new={"mutation_prob": value}, reason="explicit released pymoo spelling migration"))
    if legacy_backend == "deap":
        if best_effort:
            from .released_v8 import get_released_v8_defaults
            from .best_effort_v9 import record_guess

            numeric_controls = {}
            for key in ("crossover_eta", "crossover_probability", "mutation_eta",
                        "mutation_probability", "mutation_indpb", "offspring_multiplier"):
                if key not in optimize:
                    continue
                value = optimize[key]
                if isinstance(value, bool):
                    raise ValueError(f"optimize.{key} must be numeric, not a boolean")
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"optimize.{key} must be a finite number") from exc
                if not math.isfinite(number):
                    raise ValueError(f"optimize.{key} must be a finite number")
                numeric_controls[key] = number
            defaults = get_released_v8_defaults()["optimize"]["pymoo"]["shared"]
            mapping = {"crossover_eta": "crossover_eta", "crossover_probability": "crossover_prob_var",
                       "mutation_eta": "mutation_eta", "mutation_probability": "mutation_prob",
                       "mutation_indpb": "mutation_prob_per_variable"}
            for key, default in defaults.items():
                if key in shared:
                    continue
                legacy = next((old for old, new in mapping.items() if new == key and old in optimize), None)
                value = numeric_controls[legacy] if legacy is not None else default
                if key == "mutation_prob_per_variable" and legacy == "mutation_indpb" and value == 0:
                    value = default
                shared[key] = deepcopy(value)
                record_guess(changes, f"optimize.pymoo.shared.{key}", None, value,
                             f"approximate DEAP policy using {legacy or 'released pymoo default'}; search evolution differs")
            if not acknowledge:
                record_guess(changes, "optimize.backend", "deap", "pymoo",
                             "retire DEAP in favor of pymoo; start a fresh optimization")
        if not acknowledge and not best_effort:
            raise ValueError("optimize.backend=deap requires --acknowledge-search-backend-change; pymoo changes search evolution")
        if not shared:
            raise ValueError("DEAP conversion requires an explicit optimize.pymoo.shared policy; outer genetic gates and offspring counts are not equivalent")
        optimize["backend"] = "pymoo"
        if not best_effort or acknowledge:
            changes.append(dict(path="optimize.backend", old="deap", new="pymoo",
                                reason="acknowledged search algorithm change; replacement pymoo policy controls evolution"))
    mapping = {"crossover_eta": "crossover_eta", "crossover_probability": "crossover_prob_var",
               "mutation_eta": "mutation_eta", "mutation_indpb": "mutation_prob"}
    for key in ("crossover_eta", "crossover_probability", "mutation_eta", "mutation_indpb",
                "mutation_probability", "offspring_multiplier"):
        if key not in optimize:
            continue
        value = optimize.pop(key)
        target = mapping.get(key)
        if legacy_backend != "deap" and target and target not in shared:
            if key != "mutation_indpb" or (not isinstance(value, bool) and float(value) > 0):
                shared[target] = deepcopy(value)
                changes.append(dict(path=f"optimize.{key}", old=value,
                                    new={f"optimize.pymoo.shared.{target}": value},
                                    reason="materialized effective released pymoo fallback without changing its value"))
                continue
        changes.append(dict(path=f"optimize.{key}", old=value, new=None,
                            reason="retired DEAP control; replacement pymoo policy retained"))



_RETIRED_GPU_SCALARS = (
    "auto_lean_parallelism", "drift_halt", "drift_min_samples",
    "drift_objective_tolerance", "drift_probes", "drift_rank_halt", "drift_window",
    "exact_workers", "max_pending_exact", "validate_per_generation",
)


def retire_released_gpu_policy(config, *, changes):
    """Report known released GPU search controls; keep unknown input for rejection."""
    gpu = config.get("optimize", {}).get("gpu")
    if not isinstance(gpu, dict):
        return

    def retire(node, key, path):
        if key not in node:
            return
        value = node[key]
        if isinstance(value, (dict, list)) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{path} must be a released scalar control")
        del node[key]
        changes.append(dict(path=path, old=deepcopy(value), new=None,
                            reason="retired released-v8 GPU search control; start a fresh v9 optimization"))

    for key in _RETIRED_GPU_SCALARS:
        retire(gpu, key, f"optimize.gpu.{key}")
    bootstrap = gpu.get("seed_bootstrap")
    if isinstance(bootstrap, dict):
        if not bootstrap:
            changes.append(dict(path="optimize.gpu.seed_bootstrap", old={}, new=None,
                                reason="retired released-v8 GPU search control; empty bootstrap object removed"))
        for key in ("max_exact", "mode"):
            retire(bootstrap, key, f"optimize.gpu.seed_bootstrap.{key}")
        if not bootstrap:
            del gpu["seed_bootstrap"]
    # population_size remains exclusively owned by migrate_gpu_population:
    # preserve its backend-specific precedence and the released minimum eight.


def reject_retired_gpu_selector(selector, value=None, *, context):
    """A selector is authored search authority, not a removable root default."""
    roots = tuple(f"optimize.gpu.{key}" for key in
                  (*_RETIRED_GPU_SCALARS, "seed_bootstrap", "population_size"))
    if any(selector == root or selector.startswith(root + ".") for root in roots):
        raise ValueError(
            f"{context}.{selector} selects a retired GPU control; remove this selector "
            "and choose supported top-level optimizer policy explicitly"
        )
    if isinstance(value, dict):
        for key, child in value.items():
            reject_retired_gpu_selector(f"{selector}.{key}", child, context=context)


def reject_v8_alias_conflicts(source):
    """Reject two authored spellings of one value before precedence can erase it."""
    from config.authored import _METADATA_KEYS
    from config.optimize_bounds import SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY
    from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH
    from .renames import (
        LEGACY_BOUNDS_KEYS, LEGACY_ENTRY_GRID_KEYS, LEGACY_FILTER_KEYS, LEGACY_FORAGER_KEYS,
    )

    aliases = {**LEGACY_ENTRY_GRID_KEYS, **LEGACY_FILTER_KEYS,
               **LEGACY_FORAGER_KEYS, **LEGACY_BOUNDS_KEYS}
    section_aliases = {
        ("live", "minimum_market_age_days"): "minimum_coin_age_days",
        ("live", "noisiness_rolling_mean_window_size"): "ohlcv_rolling_window",
        ("live", "ohlcvs_1m_update_after_minutes"): "inactive_coin_candle_ttl_minutes",
        ("backtest", "panic_market_slippage_pct"): "market_order_slippage_pct",
    }
    seen = {}

    def canonical_path(parts):
        # The audit is separate from transformation: aliases in one authored
        # layer must agree, while a later file/inline layer may replace them.
        parts = list(parts)
        for marker in ("overrides", "fixed_runtime_overrides"):
            if marker in parts:
                start = len(parts) - 1 - parts[::-1].index(marker) + 1
                if start < len(parts) and parts[start] in {"long", "short"}:
                    parts.insert(start, "bot")
        while parts[-1] in aliases:
            parts[-1] = aliases[parts[-1]]
        if tuple(parts[-2:]) in section_aliases:
            parts[-1] = section_aliases[tuple(parts[-2:])]
        if parts[-2:] == ["live", "memory_snapshot_interval_minutes"]:
            parts[-2] = "logging"
        if parts[-2:] == ["backtest", "exchange"]:
            parts[-1] = "exchanges"
        for index in range(len(parts) - 2):
            if parts[index] == "bot" and parts[index + 1] in {"long", "short"}:
                if parts[index + 2] in FLAT_BOT_KEY_TO_GROUP_PATH:
                    parts[index + 2:index + 3] = FLAT_BOT_KEY_TO_GROUP_PATH[parts[index + 2]]
                break
        if len(parts) >= 3 and parts[-3:-1] == ["optimize", "bounds"]:
            side, separator, flat = parts[-1].partition("_")
            if separator and side in {"long", "short"}:
                for group, mapping in SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY.items():
                    if flat in mapping:
                        local = mapping[flat]
                        local_parts = (["weights_minutes", local.removeprefix("weights_minutes_")]
                                       if local.startswith("weights_minutes_") else [local])
                        parts[-1:] = [side, group, *local_parts]
                        break
        if parts[-2:] == ["risk", "entry_cooldown_minutes"]:
            parts[-2:] = ["entry_cooldown", "base_duration_minutes"]
        for group, old, new in (
            ("risk", "n_positions", "max_positions"),
            ("unstuck", "threshold", "exposure_trigger_ratio"),
            ("unstuck", "ema_dist", "ema_distance_fraction"),
            ("hsl", "red_threshold", "drawdown_ratio_threshold"),
            ("hsl", "ema_span_minutes", "drawdown_ratio_ema_span_minutes"),
        ):
            if parts[-2:] == [group, old]:
                parts[-1] = new
        if parts[-2:-1] == ["trailing_martingale"] and parts[-1] in {"ema_span_0", "ema_span_1"}:
            parts.insert(len(parts) - 1, "entry")
        return tuple(parts)

    def visit(node, parts=()):
        if isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str):
                    raise TypeError(f"config.{'.'.join(parts)} keys must be strings")
                if not parts and key in _METADATA_KEYS:
                    continue
                visit(child, (*parts, *key.split(".")))
        elif isinstance(node, list) and any(isinstance(child, (dict, list)) for child in node):
            for index, child in enumerate(node):
                visit(child, (*parts, str(index)))
        else:
            path = canonical_path(parts)
            value = [node] if parts[-2:] == ("backtest", "exchange") and isinstance(node, str) else node
            previous = seen.get(path)
            if previous is not None and previous[1] != value:
                raise ValueError(
                    f"config.{'.'.join(parts)} conflicts with config.{'.'.join(previous[0])} "
                    f"at canonical {'.'.join(path)}"
                )
            seen[path] = (parts, value)

    visit(source)


def _migrate_shared_fixed_selector(path, *, scope):
    """Preserve released atomic shared selectors; reject changed group authority."""
    from config.param_paths import bound_path_matches_selector

    parts = tuple(part.strip() for part in path.split(".") if part.strip())
    if not parts:
        return path
    # These are the released resolver's side-prefix and shared-flat aliases.
    # A wildcard side did not resolve an authored flat shared field.
    if parts[0] in {"long", "short"} or (
        parts[0] == "*" and len(parts) > 1
        and parts[1] in {"entry_cooldown", "risk", "forager", "hsl", "unstuck", "strategy"}
    ):
        parts = ("bot", *parts)
    aliases = {
        "n_positions": ("risk", "n_positions"),
        "unstuck_threshold": ("unstuck", "threshold"),
        "hsl_ema_span_minutes": ("hsl", "ema_span_minutes"),
        "risk_entry_cooldown_minutes": ("entry_cooldown", "base_duration_minutes"),
    }
    if len(parts) == 3 and parts[:1] == ("bot",) and parts[1] in {"long", "short"}:
        if parts[2] in aliases:
            parts = (*parts[:2], *aliases[parts[2]])
    if (parts[-1:] == ("risk",) and any(
        bound_path_matches_selector(("bot", side, "risk"), parts)
        for side in ("long", "short")
    )) or (scope == "portfolio" and parts[-1:] == ("hsl",)
            and bound_path_matches_selector(("bot", "hsl", "ema_span_minutes"), parts)):
        raise ValueError(
            f"optimize.fixed_params selector {path!r} has changed group authority in v9; "
            "choose explicit canonical intended fields instead"
        )
    cooldown = parts
    if parts[-2:] == ("risk", "entry_cooldown_minutes"):
        cooldown = (*parts[:-2], "entry_cooldown", "base_duration_minutes")
    elif parts == ("entry_cooldown_minutes",):
        cooldown = ("entry_cooldown", "base_duration_minutes")
    if cooldown[-2:] == ("entry_cooldown", "base_duration_minutes"):
        if any(bound_path_matches_selector(("bot", side, *cooldown[-2:]), cooldown)
               for side in ("long", "short")):
            return ".".join(cooldown)
        return path
    renames = {
        "n_positions": ("risk", "max_positions"),
        "threshold": ("unstuck", "exposure_trigger_ratio"),
        "ema_span_minutes": ("hsl", "drawdown_ratio_ema_span_minutes"),
        "red_threshold": ("hsl", "drawdown_ratio_threshold"),
        "ema_dist": ("unstuck", "ema_distance_fraction"),
    }
    if parts[-1] not in renames:
        return path
    group, new = renames[parts[-1]]
    consumers = ([("bot", "hsl", parts[-1])]
                 if group == "hsl" and scope == "portfolio" else
                 [("bot", side, group, parts[-1]) for side in ("long", "short")])
    if any(bound_path_matches_selector(consumer, parts) for consumer in consumers):
        return ".".join((*parts[:-1], new))
    return path


def finalize_v8_layout(source, *, span, changes, acknowledge_search_backend_change=False,
                       sparse=False, best_effort=False):
    """Translate public fields without guessing independent optimizer ranges."""
    from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH

    from .shared_policy import (COIN_SOURCE_FIELDS,
                                move_shared_policy, preserve_forced_normal_selection,
                                preserve_forced_selection_overrides, shared_policy_selector)
    from .operational_live import move_operational_live, operational_live_selector

    result = deepcopy(source)
    if not sparse:
        retire_deap_policy(result, acknowledge=acknowledge_search_backend_change, changes=changes,
                           best_effort=best_effort)
        retire_released_gpu_policy(result, changes=changes)

    def record(path, old, new, reason):
        changes.append(dict(path=path, old=deepcopy(old), new=deepcopy(new), reason=reason))

    def put(node, path, value, source_path):
        target = node
        for key in path[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f"{source_path} conflicts with {'.'.join(path)}")
        key = path[-1]
        if key in target and target[key] != value:
            raise ValueError(f"{source_path} conflicts with authored {'.'.join(path)}")
        target[key] = deepcopy(value)

    def rename(node, old, new, path):
        if old not in node:
            return
        value = node.pop(old)
        put(node, (new,), value, f"{path}.{old}")
        record(f"{path}.{old}", value, {new: value}, "canonical v9 field rename")

    def policy_scope(node, inherited):
        mode = node.get("live", {}).get("hsl_signal_mode", inherited)
        mode = node.get("bot", {}).get("global", {}).get("hsl", {}).get("scope", mode)
        return "portfolio" if mode == "unified" else mode

    def memory_paths(scope):
        paths = [("bot", "global", "unstuck", "memory_ema_span_minutes"),
                 ("bot", "global", "loss_gate", "memory_ema_span_minutes")]
        return paths + ([("bot", "global", "hsl", "memory_ema_span_minutes")]
                        if scope == "portfolio" else
                        [("bot", side, "hsl", "memory_ema_span_minutes") for side in ("long", "short")])

    def selector(path, *, parameter_selector=False):
        if parameter_selector:
            converted = _migrate_shared_fixed_selector(path, scope=scope)
            if converted == path and path.split(".")[-1] in {"n_positions", "threshold", "ema_span_minutes"}:
                reject_retired_gpu_selector(path, context="optimizer selector")
                return path
            path = converted
        reject_retired_gpu_selector(path, context="optimizer selector")
        parts = path.split(".")
        if (len(parts) >= 2 and parts[0] in {"long", "short"}
                and parts[1] in FLAT_BOT_KEY_TO_GROUP_PATH):
            parts.insert(0, "bot")
        # Only a flat side-owned shared field needs grouping. Strategy and
        # already grouped paths keep their own canonical leaf names.
        if (len(parts) >= 3 and parts[0] == "bot" and parts[1] in {"long", "short"}
                and parts[2] in FLAT_BOT_KEY_TO_GROUP_PATH):
            parts[2:3] = FLAT_BOT_KEY_TO_GROUP_PATH[parts[2]]
        # Released fixed selectors may be suffixes, group-relative, or fully
        # qualified. Preserve their prefix while converting the feature leaf.
        parts = [segment for part in parts for segment in (
            part.rsplit("_", 1) if part in {
                "position_exposure_enforcer_enabled", "position_exposure_enforcer_threshold",
                "total_exposure_enforcer_enabled", "total_exposure_enforcer_threshold",
                "total_exposure_enforcer_policy",
            } else (part,)
        )]
        if len(parts) >= 2 and parts[-2] in {"position_exposure_enforcer", "total_exposure_enforcer"} and parts[-1] == "threshold":
            parts[-1] = "threshold_ratio"
        for group, old, new in (
            ("hsl", "red_threshold", "drawdown_ratio_threshold"),
            ("unstuck", "ema_dist", "ema_distance_fraction"),
        ):
            if parts[-2:] == [group, old]:
                parts[-1] = new
        path = ".".join(parts)
        for old, new in (
            ("live.hsl_signal_mode", "bot.global.hsl.scope"),
            ("live.max_realized_loss_pct", "bot.global.loss_gate.max_drawdown_ratio"),
            ("backtest.liquidation_threshold", "backtest.equity_floor_ratio"),
            ("live.auto_gs", "live.disapproved_position_mode"),
            ("bot.hsl.", "bot.global.hsl."),
            (".hsl.ema_span_minutes", ".hsl.drawdown_ratio_ema_span_minutes"),
            (".risk.n_positions", ".risk.max_positions"),
            (".risk.position_exposure_enforcer_enabled", ".risk.position_exposure_enforcer.enabled"),
            (".risk.position_exposure_enforcer_threshold", ".risk.position_exposure_enforcer.threshold_ratio"),
            (".risk.total_exposure_enforcer_enabled", ".risk.total_exposure_enforcer.enabled"),
            (".risk.total_exposure_enforcer_threshold", ".risk.total_exposure_enforcer.threshold_ratio"),
            (".risk.total_exposure_enforcer_policy", ".risk.total_exposure_enforcer.policy"),
            (".unstuck.threshold", ".unstuck.exposure_trigger_ratio"),
        ):
            path = path.replace(old, new)
        return operational_live_selector(shared_policy_selector(path))

    def patch(raw, inherited_scope, path):
        from config.scenario_overrides import normalize_scenario_overrides

        values = normalize_scenario_overrides(raw)
        scope = values.get("live.hsl_signal_mode", values.get("bot.global.hsl.scope", inherited_scope))
        scope = "portfolio" if scope == "unified" else scope
        converted = {}
        for key, value in values.items():
            authored_value = deepcopy(value)
            reject_retired_gpu_selector(key, value, context=path)
            if key.endswith(".live") and isinstance(value, dict):
                # A dotted object is replacement authority, not a flattened
                # list of fields. Convert its sparse contents in place.
                move_operational_live(
                    {"live": value},
                    record=record,
                    path=f"{path}.{key}".removesuffix(".live"),
                )
            if key.endswith((".forager.volume_drop_pct", ".risk.total_exposure_entry_gate_enabled")):
                record(f"{path}.{key}", value, None, "retired policy control")
                continue
            if key == "live.lookback_ema_span_minutes":
                for target in memory_paths(scope):
                    dotted = ".".join(target)
                    if dotted in converted and converted[dotted] != value:
                        raise ValueError(f"{path}.{key} conflicts with {dotted}")
                    converted[dotted] = deepcopy(value)
                record(f"{path}.{key}", value, converted, "split shared slow memory into independent spans")
                continue
            if key in {f"live.{field}" for field in COIN_SOURCE_FIELDS}:
                field = key.split(".")[-1]
                sources = value if isinstance(value, dict) else {side: value for side in ("long", "short")}
                if set(sources) - {"long", "short"}:
                    raise ValueError(f"{path}.{key} must use only long/short source keys")
                for side, source in sources.items():
                    target = f"bot.{side}.universe.{field}"
                    if target in converted and converted[target] != source:
                        raise ValueError(f"{path}.{key} conflicts with {target}")
                    converted[target] = deepcopy(source)
                record(f"{path}.{key}", value, converted, "side-owned universe source paths")
                continue
            new = selector(key)
            if new == "bot.global.hsl.scope" and value in {"unified", "pside"}:
                value = "portfolio" if value == "unified" else "position_side"
            if key == "live.auto_gs":
                if type(value) is not bool:
                    raise ValueError(f"{path}.{key} must be a boolean")
                value = "graceful_stop" if value else "manual"
            if new in converted and converted[new] != value:
                raise ValueError(f"{path}.{key} conflicts with {new}")
            converted[new] = deepcopy(value)
            if new != key or authored_value != value:
                record(f"{path}.{key}", authored_value, {new: value}, "canonical v9 override path")
        return converted

    scope = policy_scope(result, "coin")
    live = result.setdefault("live", {})
    for old, target in (
        ("hsl_signal_mode", ("bot", "global", "hsl", "scope")),
        ("max_realized_loss_pct", ("bot", "global", "loss_gate", "max_drawdown_ratio")),
    ):
        if old in live:
            value = live.pop(old)
            authored_value = deepcopy(value)
            if old == "hsl_signal_mode" and value in {"unified", "pside"}:
                value = "portfolio" if value == "unified" else "position_side"
            put(result, target, value, f"live.{old}")
            record(f"live.{old}", authored_value, {".".join(target): value}, "consumer-owned global risk policy")
    if "auto_gs" in live:
        value = live.pop("auto_gs")
        if type(value) is not bool:
            raise ValueError("live.auto_gs must be a boolean")
        mode = "graceful_stop" if value else "manual"
        put(live, ("disapproved_position_mode",), mode, "live.auto_gs")
        record("live.auto_gs", value, mode, "explicit disapproved-position policy")
    shared = live.pop("lookback_ema_span_minutes", span)
    if shared is not None:
        for target in memory_paths(scope):
            put(result, target, shared, "live.lookback_ema_span_minutes")
        record("live.lookback_ema_span_minutes", shared,
               {".".join(path): shared for path in memory_paths(scope)},
               "split shared slow memory into independent spans")
    bot = result.setdefault("bot", {})
    if "hsl" in bot:
        old = bot.pop("hsl")
        for key, value in old.items():
            put(result, ("bot", "global", "hsl", key), value, f"bot.hsl.{key}")
        record("bot.hsl", old, {"bot.global.hsl": old}, "portfolio HSL authority")
    rename(result.setdefault("backtest", {}), "liquidation_threshold", "equity_floor_ratio", "backtest")

    def visit(node, path=""):
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        if path.endswith(".hsl"):
            rename(node, "red_threshold", "drawdown_ratio_threshold", path)
            if node.get("scope") == "pside":
                node["scope"] = "position_side"
                record(f"{path}.scope", "pside", "position_side", "explicit position-side scope spelling")
            rename(node, "ema_span_minutes", "drawdown_ratio_ema_span_minutes", path)
        if path.endswith(".risk"):
            rename(node, "n_positions", "max_positions", path)
            for owner in ("position_exposure_enforcer", "total_exposure_enforcer"):
                if isinstance(node.get(owner), dict):
                    rename(node[owner], "threshold", "threshold_ratio", f"{path}.{owner}")
                for leaf in ("enabled", "threshold", "policy"):
                    old = f"{owner}_{leaf}"
                    if old in node:
                        value = node.pop(old)
                        target = "threshold_ratio" if leaf == "threshold" else leaf
                        put(node, (owner, target), value, f"{path}.{old}")
                        record(
                            f"{path}.{old}", value, {f"{owner}.{target}": value},
                            "explicit released-v8 exposure-enforcer object conversion",
                        )
        if path.endswith(".unstuck"):
            rename(node, "ema_dist", "ema_distance_fraction", path)
            rename(node, "threshold", "exposure_trigger_ratio", path)
        for key in list(node):
            current = f"{path}.{key}" if path else key
            if path.endswith(".strategy") and key == "trailing_grid_v7":
                if result.get("live", {}).get("strategy_kind") == key:
                    raise ValueError("trailing_grid_v7 is retired; choose a retained strategy explicitly before v9 conversion")
                record(current, node.pop(key), None, "removed inactive retired strategy subtree")
                continue
            if ((path.endswith(".forager") and key == "volume_drop_pct") or
                    (path.endswith(".risk") and key == "total_exposure_entry_gate_enabled") or
                    key in {"forager_volume_drop_pct", "filter_volume_drop_pct", "risk_twel_entry_gate_enabled"}):
                record(current, node.pop(key), None, "retired policy control")
                continue
            if key in {"overrides", "fixed_runtime_overrides"} and ("scenarios[" in path or path == "optimize"):
                node[key] = patch(node[key], scope, current)
                visit(node[key], current)
            elif key == "fixed_params" and isinstance(node[key], list):
                old = deepcopy(node[key])
                node[key] = [selector(item, parameter_selector=True) if isinstance(item, str)
                             else item for item in node[key]]
                if old != node[key]:
                    record(current, old, node[key], "canonical v9 fixed-parameter selectors")
            else:
                visit(node[key], current)
    visit(result)
    if sparse:
        from .entry_cooldown import migrate_entry_cooldown_tree
        from .entry_ema import migrate_entry_ema_tree
        from config.shared_bot import canonicalize_shared_bot_side

        migrate_entry_cooldown_tree(result)
        migrate_entry_ema_tree(result)
        move_shared_policy(result, record=record)
        move_operational_live(result, record=record)
        for side in ("long", "short"):
            canonicalize_shared_bot_side(result.get("bot", {}).get(side))
        return result
    # All released-layout structural transforms run only inside this converter.
    from . import apply_migrations, apply_backward_compatibility_renames
    from .entry_cooldown import migrate_entry_cooldown
    from .entry_ema import migrate_entry_ema_spans
    from .unstuck_ema import migrate_unstuck_ema_spans
    from config.shared_bot import canonicalize_shared_bot_side
    from config.strategy import sync_canonical_strategy_config

    for side in ("long", "short"):
        values = result.get("bot", {}).get(side, {})
        from config.shared_bot import get_grouped_path
        missing = object()
        for flat_key, path in FLAT_BOT_KEY_TO_GROUP_PATH.items():
            grouped = get_grouped_path(values, path, default=missing)
            if flat_key in values and grouped is not missing and values[flat_key] != grouped:
                raise ValueError(f"bot.{side}.{flat_key} conflicts with bot.{side}.{'.'.join(path)}")
    before = deepcopy(result)
    apply_migrations(result, verbose=False)
    apply_backward_compatibility_renames(result, verbose=False)
    migrate_entry_cooldown(result)
    move_shared_policy(result, record=record)
    migrate_entry_ema_spans(result)
    migrate_unstuck_ema_spans(result, verbose=False)
    move_operational_live(result, record=record)
    preserve_forced_normal_selection(result, record=record)
    for side in ("long", "short"):
        canonicalize_shared_bot_side(result.get("bot", {}).get(side))

    def canonicalize_coin_patches(node, path=()):
        if isinstance(node, list):
            for child in node:
                canonicalize_coin_patches(child, path)
        elif isinstance(node, dict):
            if len(path) > 2 and path[-2] == "bot" and path[-1] in ("long", "short"):
                canonicalize_shared_bot_side(node)
            for key, child in node.items():
                canonicalize_coin_patches(child, (*path, key))

    # Nested authored coin patches are converted here, before final strict
    # validation. Ordinary loading never resolves their released flat aliases.
    canonicalize_coin_patches(result)
    preserve_forced_selection_overrides(result, record=record)
    sync_canonical_strategy_config(result)
    from config.limits import normalize_limit_entries
    from config.scoring import normalize_scoring_config

    if "limits" in result.get("optimize", {}):
        result["optimize"]["limits"] = normalize_limit_entries(result["optimize"]["limits"])
    if "scoring" in result.get("optimize", {}):
        normalize_scoring_config(result, verbose=False)

    def differences(old, new, path=""):
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                current = f"{path}.{key}" if path else key
                if key not in old or key not in new:
                    record(current, old.get(key), new.get(key), "explicit released-v8 structural conversion")
                else:
                    differences(old[key], new[key], current)
        elif old != new:
            record(path, old, new, "explicit released-v8 structural conversion")
    differences(before, result)
    bounds = result.get("optimize", {}).get("bounds", {})
    if "live" in bounds and "lookback_ema_span_minutes" in bounds["live"]:
        raise ValueError("optimize.bounds.live.lookback_ema_span_minutes requires explicit independent memory bounds")
    if "hsl" in bounds:
        old = bounds.pop("hsl")
        put(bounds, ("global", "hsl"), old, "optimize.bounds.hsl")
    from config.optimize_bounds import flatten_optimize_bounds, set_flat_optimize_bound
    kind = result.get("bot", {}).get("global", {}).get("strategy_kind", "trailing_martingale")
    for side in ("long", "short"):
        forager = bounds.get(side, {}).get("forager", {})
        nested = forager.get("score_weights", {})
        for key, value in forager.items():
            if key.startswith("score_weights_"):
                name = key.removeprefix("score_weights_")
                if name in nested and value != nested[name]:
                    raise ValueError(
                        f"optimize.bounds.{side}.forager.{key} conflicts with "
                        f"optimize.bounds.{side}.forager.score_weights.{name}"
                    )
    from config.optimize_bounds import SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY
    from config.param_paths import OPTIMIZABLE_BOT_KEY_PATHS
    from config.strategy_spec import strategy_optimize_key_path_map

    known_keys = set(strategy_optimize_key_path_map(kind)) | set(OPTIMIZABLE_BOT_KEY_PATHS)
    for mapping in SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.values():
        known_keys.update(f"{side}_{key}" for side in ("long", "short") for key in mapping.values())
    known_keys.update(SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY["hsl"].values())
    for key, value in flatten_optimize_bounds(bounds, strategy_kind=kind).items():
        # Canonical unknown fields retain their authored tree and reach the
        # strict schema gate; they are not optimizer aliases to reinterpret.
        if key in known_keys:
            set_flat_optimize_bound(bounds, kind, key, value)
    for side in ("long", "short"):
        for key in list(bounds.get(side, {}).get("forager", {})):
            if key.startswith("score_weights_"):
                old = bounds[side]["forager"].pop(key)
                destination = f"optimize.bounds.{side}.forager.score_weights.{key.removeprefix('score_weights_')}"
                record(
                    f"optimize.bounds.{side}.forager.{key}", old, {destination: old},
                    "explicit released-v8 nested score-weight bound conversion",
                )
    return result
