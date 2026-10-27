"""Explicit structural conversion from released v8 layouts to final v9."""

from copy import deepcopy
import math
import re

from .released_v8_unstuck import RELEASED_V8_UNSTUCK_FLAT_PATHS


_RETIRED_WEIGHTED_METRIC = re.compile(r"_w(?:_|$)", re.IGNORECASE)
_IGNORED_BACKTEST_WARMUP_REASON = (
    "removed unused released-v8 backtest warmup cap; live.max_warmup_minutes owned the shared cap"
)
_BACKTEST_FEE_OVERRIDES = frozenset({"maker_fee_override", "taker_fee_override"})
_RELEASED_SHARED_FLAT_PATHS = {
    **RELEASED_V8_UNSTUCK_FLAT_PATHS,
    "hsl_red_threshold": ("hsl", "drawdown_threshold_fraction"),
    "hsl_ema_span_minutes": ("hsl", "drawdown_fraction_ema_span_minutes"),
    "total_wallet_exposure_limit": ("risk", "total_wallet_exposure_limit_ratio"),
    "wallet_exposure_limit": ("wallet_exposure_limit_ratio",),
    "risk_we_excess_allowance_pct": ("risk", "we_excess_allowance_fraction"),
    "risk_wel_enforcer_threshold": ("risk", "position_exposure_enforcer", "threshold_ratio"),
    "risk_twel_enforcer_threshold": ("risk", "total_exposure_enforcer", "threshold_ratio"),
    "forager_volume_ema_span_1m": ("forager", "volume_ema_span_minutes"),
    "forager_volatility_ema_span_1m": ("forager", "log_range_ema_span_minutes"),
    "unilateralness_ema_span_1m": ("forager", "unilateralness_ema_span_minutes"),
}
_RELEASED_SHARED_BOUND_PATHS = {
    **_RELEASED_SHARED_FLAT_PATHS,
    "forager_score_weights_volatility": ("forager", "score_weights", "log_range"),
}


def released_shared_bot_paths():
    """Keep released flat source keys independent of current native IDs."""
    from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH

    return {
        **FLAT_BOT_KEY_TO_GROUP_PATH,
        **_RELEASED_SHARED_FLAT_PATHS,
    }


def canonicalize_released_shared_bot_side(values):
    """Group flat released inputs only inside the explicit v8 converter."""
    from config.shared_bot import canonicalize_shared_bot_side

    if isinstance(values, dict):
        for flat, path in _RELEASED_SHARED_FLAT_PATHS.items():
            if flat not in values:
                continue
            target = values
            for part in path[:-1]:
                target = target.setdefault(part, {})
                if not isinstance(target, dict):
                    raise TypeError(f"{'.'.join(path[:-1])} must be a dict")
            value = values.pop(flat)
            if path[-1] in target and target[path[-1]] != value:
                raise ValueError(f"{flat} conflicts with {'.'.join(path)}")
            target[path[-1]] = value
    canonicalize_shared_bot_side(values)


def _is_retired_metric(name):
    return isinstance(name, str) and (
        bool(_RETIRED_WEIGHTED_METRIC.search(name))
        or name.startswith("strategy_eq_underwater_pct_")
    )


def _released_fee_override(value, path):
    """Write released fees in the target's nullable numeric representation."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"{path} must be a numeric fee or null, not a boolean; supply an explicit number if intended"
        )
    try:
        fee = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} must be a finite number or null") from exc
    if not math.isfinite(fee):
        raise ValueError(f"{path} must be finite")
    return fee if isinstance(value, str) else value


def replace_retired_metrics(source, *, replacements, acknowledge, changes, best_effort=False):
    """Convert acknowledged formulas, reporting narrowly supported opt-in guesses."""
    replacements = dict(replacements or {})
    used = set()
    ew_requested = False
    replacement_paths = set()
    outer_reducers = {}
    inferences = {}
    removals = {}
    original_source = deepcopy(source)

    def original_mapping(payload, path="backtest"):
        from config.reducers import REDUCER_ALIASES, normalize_reducer
        from .best_effort_metrics import legacy_metric_identity

        values = []
        for alias in REDUCER_ALIASES:
            if alias not in payload:
                continue
            if not isinstance(payload[alias], dict):
                raise ValueError(f"{path}.{alias} must be a mapping")
            normalized = {}
            for name, mode in payload[alias].items():
                mode = normalize_reducer(mode, path=f"{path}.{alias}.{name}")
                if mode is None:
                    raise ValueError(f"{path}.{alias}.{name} must be a reducer name, not null")
                identity = legacy_metric_identity(name) if name != "default" else name
                if identity in normalized and normalized[identity] != mode:
                    raise ValueError(f"{path}.{alias} has conflicting metric aliases for {name}")
                normalized[identity] = mode
            values.append(normalized)
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"{path} has conflicting reducer aliases")
        return values[0] if values else {}


    def original_reducer(entry, metric, kind):
        from config.reducers import REDUCER_ALIASES, LIMIT_REDUCER_ALIASES, reducer_from_aliases
        from config.limits import resolve_reducer_mode
        from .best_effort_metrics import legacy_metric_identity

        default_scenario = original_source.get("optimize", {}).get("objective_scenario") if kind == "scoring" else None
        scenario = entry.get("scenario", default_scenario)
        reducer, _ = reducer_from_aliases(entry, path=f"{kind} on {metric}", aliases=LIMIT_REDUCER_ALIASES if kind == "limits" else REDUCER_ALIASES)
        if scenario is not None:
            if not isinstance(scenario, str) or not scenario.strip():
                raise ValueError(f"{kind} on {metric}: scenario must be a nonempty string or null")
            if reducer is not None:
                raise ValueError(f"{kind} on {metric}: cannot set both a named scenario and reducer")
            return None
        if reducer is not None:
            return reducer

        backtest = original_source.get("backtest", {})
        modes = original_mapping(backtest)
        suite = backtest.get("suite")
        if isinstance(suite, dict) and (suite.get("enabled", False) or suite.get("scenarios", [])):
            suite_modes = original_mapping(suite, "backtest.suite") if any(alias in suite for alias in REDUCER_ALIASES) else {"default": "mean"}
            modes = {**modes, **suite_modes}
        return resolve_reducer_mode(legacy_metric_identity(metric), modes)

    def normalized_limit_policy(entry):
        from config.limits import normalize_limit_entries

        normalized = normalize_limit_entries([entry])[0]
        for field, direction in (("value", "greater_than"), ("range", "outside_range")):
            if field not in normalized:
                continue
            # The owner preserves inactive fields too. Validate each bound form;
            # an inactive range must never redirect conversion of an active scalar.
            if field == "value" and isinstance(normalized[field], (list, tuple)) and "range" in normalized:
                probe = {**entry, "penalize_if": "outside_range", "range": normalized[field]}
                normalized[field] = normalize_limit_entries([probe])[0]["range"]
                continue
            probe = {**entry, "penalize_if": direction}
            normalized[field] = normalize_limit_entries([probe])[0][field]
        return normalized

    def validate_entry(entry, kind, path, metric):
        """Let canonical policy owners inspect fields before retirement erases them."""
        from config.authored import _closed_fields, _LIMIT_FIELDS
        from config.reducers import LIMIT_REDUCER_ALIASES
        from config.scoring import normalize_scoring_entries

        probe = deepcopy(entry)
        probe["metric"] = "loss_profit_ratio"
        if kind == "scoring":
            normalize_scoring_entries([probe])
        else:
            _closed_fields(probe, _LIMIT_FIELDS | set(LIMIT_REDUCER_ALIASES), path)
            normalized = normalized_limit_policy(probe)
            bounds = list(normalized.get("range", []))
            if "value" in normalized:
                value = normalized["value"]
                bounds.extend(value if isinstance(value, (list, tuple)) else [value])
            if any(not math.isfinite(bound) for bound in bounds):
                raise ValueError(f"{path}: limit bounds must be finite")
        original_reducer(entry, metric, kind)

    def record_removal(path, original, reason):
        from .best_effort_v9 import record_guess
        record_guess(changes, path, original, None, reason + "; removed unsupported metric selection; review optimization objectives and constraints")

    def apply_inference(entry, metric, selector_path, kind):
        """Return a removal reason when the authored policy cannot be transformed."""
        from config.reducers import REDUCER_ALIASES, LIMIT_REDUCER_ALIASES
        from config.scoring import default_objective_goal
        from .best_effort_metrics import transform_bound
        from .best_effort_v9 import record_guess

        choice = inferences.get(selector_path)
        if choice is None:
            return None
        validate_entry(entry, kind, selector_path.removesuffix(".metric"), metric)
        if kind == "scoring" and entry.get("goal") is None and default_objective_goal(entry["metric"]) is None:
            return "inferred metric has no default scoring goal; author an explicit goal to retain this objective"
        if choice.transform == "identity":
            return None
        reducer = original_reducer(entry, metric, kind)
        before = deepcopy(entry)
        transform = choice.transform
        inactive_bound = None
        if kind == "limits":
            if transform == "annualize_daily" and reducer == "std":
                return "daily-growth dispersion has no meaningful nominal annual CAGR limit conversion"
            # Canonical normalization owns direction aliases and scalar/range forms.
            normalized = normalized_limit_policy(entry)
            if transform == "complement" and normalized["penalize_if"] == "auto":
                return "automatic long-profit limit direction cannot be safely complemented"
            if transform == "complement" and reducer == "std":
                transform = "identity"  # std(1-x) == std(x).
            range_active = normalized["penalize_if"] in {"inside_range", "outside_range"}
            inactive_field = "value" if range_active else "range"
            if inactive_field in normalized:
                inactive_bound = (inactive_field, deepcopy(before.get(inactive_field, normalized[inactive_field])))
                normalized.pop(inactive_field)
            try:
                if range_active:
                    normalized["range"] = sorted(transform_bound(value, transform) for value in normalized["range"])
                else:
                    normalized["value"] = transform_bound(normalized["value"], transform)
            except ValueError as exc:
                return str(exc)
            if transform == "complement":
                normalized["penalize_if"] = {
                    "less_than": "greater_than", "greater_than": "less_than",
                    "less_than_or_equal": "greater_than_or_equal", "greater_than_or_equal": "less_than_or_equal",
                }.get(normalized["penalize_if"], normalized["penalize_if"])
            entry.clear()
            entry.update(normalized)
        elif transform == "complement" and reducer != "std":
            entry["goal"] = {"min": "max", "max": "min"}[str(entry["goal"]).strip().lower()]
        if choice.transform == "complement" and reducer is not None:
            for alias in LIMIT_REDUCER_ALIASES if kind == "limits" else REDUCER_ALIASES:
                entry.pop(alias, None)
            entry["reducer"] = {"min": "max", "max": "min"}.get(reducer, reducer)
        if inactive_bound is not None:
            field, old = inactive_bound
            reason = "discard inactive bound field during inferred metric conversion; penalize_if selects the active scalar or range constraint"
            if field == "value" and "range" not in before:
                reason = "preserve legacy range-as-value constraint in the canonical range field during inferred metric conversion"
            record_guess(changes, selector_path.removesuffix(".metric") + "." + field, old, None,
                         reason)
        if entry != before or (kind == "limits" and choice.transform == "annualize_daily"):
            reason = {
                "complement": "complement long share to short contribution; reverse ordering and min/max reducers, while standard deviation is unchanged",
                "hours_to_days": "convert hour limits to day limits using 1/24",
                "annualize_daily": "nominally compound daily growth thresholds over 365.25 days; nonlinear scenario aggregation and changed growth formulas require fresh threshold evaluation",
            }[choice.transform]
            record_guess(changes, selector_path.removesuffix(".metric"), before, entry, reason)
        return None

    def keep_outer_reducer(node, path, reducer):
        from config.reducers import reducer_from_aliases, LIMIT_REDUCER_ALIASES
        existing, present = reducer_from_aliases(node, path=path, aliases=LIMIT_REDUCER_ALIASES)
        if present and existing != reducer:
            raise ValueError(f"{path}: released metric outer reducer {reducer!r} conflicts with authored reducer {existing!r}; choose explicitly")
        node["reducer"] = reducer
        changes.append(dict(path=f"{path}.reducer", old=existing if present else None,
                            new=reducer, reason="preserve released metric outer statistic in the reducer field"))

    def convert(name, path):
        from config.metrics import canonicalize_metric_name, reject_retired_metric, metric_base_name, KNOWN_METRICS
        from config.ew import is_ew_metric
        from .released_v8_metrics import released_v8_metric_name, RELEASED_V8_METRIC_NAMES, RELEASED_V8_STAT_SUFFIXES
        nonlocal ew_requested
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: metric selector must be a nonempty string")
        name = name.strip()
        if name not in RELEASED_V8_METRIC_NAMES:
            for stat in RELEASED_V8_STAT_SUFFIXES:
                marker = "_" + stat
                if name.endswith(marker) and (name[:-len(marker)] in RELEASED_V8_METRIC_NAMES or _is_retired_metric(name[:-len(marker)])):
                    outer_reducers[path] = stat
                    break
        explicit = name in replacements
        inferred = False
        if explicit:
            new = replacements[name]
            used.add(name)
        else:
            try:
                new = released_v8_metric_name(name)
            except ValueError:
                new = None
            if best_effort:
                from .best_effort_metrics import infer_metric
                choice = infer_metric(name)
                if choice is not None:
                    if choice.outer_reducer is not None:
                        outer_reducers[path] = choice.outer_reducer
                    if choice.target is None:
                        removals[path] = choice.reason
                        return None
                    new = choice.target
                    inferences[path] = choice
                    inferred = True
            if new is None or metric_base_name(new) not in KNOWN_METRICS:
                raise ValueError(f"{path}: retired metric {name!r} requires --metric-replacement {name}=NEW; the compact core has no automatic formula conversion")
        if not explicit and new is not None:
            new = metric_base_name(new)
        reject_retired_metric(new, path=path)
        if not acknowledge and not best_effort:
            raise ValueError(f"{path}: core definition version 1 requires --acknowledge-metric-formula-change and re-evaluation of thresholds/results")
        new = canonicalize_metric_name(new)
        replacement_paths.add(path)
        ew_requested |= is_ew_metric(new)
        reason = "core definition version 1 metric formula boundary; re-evaluate objectives, limits and stored fitness"
        if inferred:
            reason = (
                "inferred metric formula substitution: " + inferences[path].reason + "; "
                + reason
            )
        if inferred or (best_effort and not acknowledge):
            from .best_effort_v9 import record_guess
            record_guess(changes, path, name, new, reason)
        else:
            changes.append(dict(path=path, old=name, new=new, reason=reason))
        return new

    def visit(node, path=""):
        if isinstance(node, list):
            kind = path.split(".")[-1]
            scoring = kind == "scoring"
            original = deepcopy(node) if scoring else None
            replacements_before = len(replacement_paths)
            converted = []
            for index, value in enumerate(node):
                current = f"{path}[{index}]"
                if isinstance(value, str) and scoring:
                    selected = convert(value, current)
                    if selected is None:
                        record_removal(current, value, removals[current])
                        continue
                    if current in outer_reducers and current in inferences:
                        record_removal(current, value, "legacy outer-statistic shorthand has no declared scoring goal; use an objective object with an explicit goal")
                        continue
                    if current in outer_reducers or current in inferences:
                        from config.scoring import default_objective_goal
                        goal = default_objective_goal(selected)
                        if goal is None:
                            if best_effort and current in inferences:
                                record_removal(current, value, "inferred metric has no default scoring goal; author an explicit goal to retain this objective")
                                continue
                            raise ValueError(f"{current}: use an explicit scoring object with a goal for {selected!r}")
                        entry = {"metric": selected, "goal": goal}
                        if current in outer_reducers:
                            entry["reducer"] = outer_reducers[current]
                        reason = apply_inference(entry, value, current, kind)
                        if reason is not None:
                            record_removal(current, value, reason)
                            continue
                        converted.append(entry)
                    else:
                        converted.append(selected)
                elif kind in {"scoring", "limits"} and isinstance(value, dict) and "metric" in value:
                    before = deepcopy(value)
                    selector_path = current + ".metric"
                    metric = value["metric"]
                    selected = convert(metric, selector_path)
                    if selector_path in outer_reducers:
                        keep_outer_reducer(value, current, outer_reducers[selector_path])
                    if selected is None:
                        validate_entry(value, kind, current, metric)
                        record_removal(current, before, removals[selector_path])
                        continue
                    value["metric"] = selected
                    reason = apply_inference(value, metric, selector_path, kind)
                    if reason is not None:
                        record_removal(current, before, reason)
                        continue
                    converted.append(value)
                else:
                    visit(value, current)
                    converted.append(value)
            node[:] = converted
            if best_effort and scoring and original and not node:
                from .best_effort_v9 import record_guess
                record_guess(changes, path, original, [], "all unsupported objectives removed; retain explicit empty scoring without inserting replacement objectives")
            if scoring and len(replacement_paths) > replacements_before:
                from config.scoring import normalize_scoring_entries

                # Use the canonical owner's equality/conflict semantics. Formula
                # replacement can make formerly distinct objectives identical.
                metrics = [item if isinstance(item, str) else item.get("metric")
                           for item in node if isinstance(item, (str, dict))]
                duplicate = len(metrics) != len(set(metrics))
                normalized = normalize_scoring_entries(node)[0] if duplicate else None
                if normalized is not None and len(normalized) < len(node):
                    node[:] = [spec.to_config() for spec in normalized]
                    reason = (
                        "canonical scoring normalization after metric formula replacement "
                        f"coalesces identical current objectives: {len(original)} -> {len(node)}; "
                        "goals and retained scenario/reducer policy are unchanged"
                    )
                    if best_effort:
                        from .best_effort_v9 import record_guess

                        record_guess(changes, path, original, node, reason)
                    else:
                        changes.append(dict(path=path, old=original, new=deepcopy(node), reason=reason))
                    changes[-1].update(old_count=len(original), new_count=len(node))
            return
        if not isinstance(node, dict):
            return
        released_aggregate = path in {"backtest.aggregate", "backtest.suite.aggregate"}
        metric_map = released_aggregate or path.split(".")[-1] in {"reducer", "limits"} or path in {"backtest.stat", "backtest.scenario_stat", "backtest.suite.reducer", "backtest.suite.stat", "backtest.suite.scenario_stat"}
        for key in list(node):
            current = f"{path}.{key}" if path else key
            if ((key == "visible_metrics" and path.split(".")[-1] == "backtest")
                    or key == "backtest.visible_metrics"):
                changes.append(dict(path=current, old=node.pop(key), new=None,
                                    reason="removed terminal-only metric visibility setting; backtests display their complete analysis"))
                continue
            if str(key).split(".")[-1] == "limits" and isinstance(node[key], dict):
                from config.limits import _legacy_limits_dict_to_entries
                original_limits = deepcopy(node[key])
                converted_limits = []
                for limit_name, bound in original_limits.items():
                    prefix = next((part for part in ("penalize_if_greater_than_", "penalize_if_lower_than_", "lower_bound_", "upper_bound_") if limit_name.startswith(part)), "")
                    limit_path = f"{current}.{limit_name}"
                    metric = limit_name.removeprefix(prefix)
                    entry = _legacy_limits_dict_to_entries({limit_name: bound})[0]
                    selected = convert(metric, limit_path)
                    if limit_path in outer_reducers:
                        keep_outer_reducer(entry, limit_path, outer_reducers[limit_path])
                    if selected is None:
                        validate_entry(entry, "limits", limit_path, metric)
                        record_removal(limit_path, {limit_name: bound}, removals[limit_path])
                        continue
                    entry["metric"] = selected
                    reason = apply_inference(entry, metric, limit_path, "limits")
                    if reason is not None:
                        record_removal(limit_path, {limit_name: bound}, reason)
                        continue
                    converted_limits.append(entry)
                node[key] = converted_limits
                changes.append(dict(path=current, old=original_limits, new=deepcopy(converted_limits),
                                    reason="preserve released limit direction and outer reducer in explicit entries"))
            elif key == "metric" and not released_aggregate:
                selected = convert(node[key], current)
                if selected is None:
                    raise ValueError(f"{current}: unsupported metric outside objective/limit collections requires an explicit replacement")
                node[key] = selected
                if current in outer_reducers:
                    keep_outer_reducer(node, path, outer_reducers[current])
            elif metric_map and key != "default":
                prefix = "" if released_aggregate else next((
                    prefix for prefix in ("penalize_if_greater_than_", "penalize_if_lower_than_",
                                          "lower_bound_", "upper_bound_")
                    if key.startswith(prefix)), "")
                metric = key.removeprefix(prefix)
                if released_aggregate and not _is_retired_metric(metric):
                    from .released_v8_metrics import released_v8_metric_name

                    try:
                        released_v8_metric_name(metric)
                    except ValueError:
                        # Released aggregate rules also accepted custom metrics.
                        # Only this vocabulary probe permits their passthrough.
                        from .best_effort_metrics import infer_metric
                        if not best_effort or infer_metric(metric) is None:
                            continue
                selected = convert(metric, current)
                from config.reducers import normalize_reducer
                mode = normalize_reducer(node[key], path=current)
                if mode is None:
                    raise ValueError(f"{current} must be a reducer name, not null")
                if selected is None:
                    record_removal(current, {key: node.pop(key)}, removals[current])
                    continue
                renamed = prefix + selected
                if current in outer_reducers and node[key] != outer_reducers[current]:
                    raise ValueError(f"{current}: released metric outer reducer conflicts with its reducer mapping; choose explicitly")
                choice = inferences.get(current)
                if choice is not None and choice.transform == "complement":
                    new_mode = {"min": "max", "max": "min"}.get(mode, mode)
                    if new_mode != mode:
                        from .best_effort_v9 import record_guess
                        record_guess(changes, current, node[key], new_mode, "complement long-share reducer ordering for short contribution")
                        node[key] = new_mode
                if renamed != key:
                    value = node.pop(key)
                    if renamed in node and node[renamed] != value:
                        raise ValueError(f"{current} conflicts with selected metric {renamed}")
                    node[renamed] = value
            else:
                visit(node[key], current)
    if best_effort:
        # Check aliases before removing any rule could conceal their conflict.
        raw_backtest = original_source.get("backtest", {})
        if isinstance(raw_backtest, dict):
            original_mapping(raw_backtest)
            if isinstance(raw_backtest.get("suite"), dict):
                original_mapping(raw_backtest["suite"], "backtest.suite")
    visit(source)
    if best_effort:
        from .best_effort_v9 import record_guess

        backtest = source.setdefault("backtest", {})
        if ew_requested and backtest.get("ew_half_life_days") is None:
            from config.metrics import DEFAULT_EW_HALF_LIFE_DAYS
            record_guess(changes, "backtest.ew_half_life_days", None, DEFAULT_EW_HALF_LIFE_DAYS,
                         "resolve the native default CAGR EW half-life; re-evaluate objectives and limits")
            backtest["ew_half_life_days"] = DEFAULT_EW_HALF_LIFE_DAYS
    unused = set(replacements) - used
    if unused:
        raise ValueError(f"unused metric replacements: {', '.join(sorted(unused))}")


def retire_deap_policy(config, *, acknowledge, changes, best_effort=False):
    """Preserve effective pymoo policy; DEAP algorithm changes require consent."""
    from config.public_names import released_public_path

    def public_shared_target(key):
        return released_public_path(("optimize", "pymoo", "shared", key))[-1]

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
                            new={public_shared_target("mutation_prob"): value}, reason="explicit released pymoo spelling migration"))
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
                record_guess(changes, f"optimize.pymoo.shared.{public_shared_target(key)}", None, value,
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
                                    new={f"optimize.pymoo.shared.{public_shared_target(target)}": value},
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


def canonical_released_v8_path(parts, *, strategy_kind=None):
    """Identify one released authored leaf before aliases or defaults can erase it."""
    from config.optimize_bounds import SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY
    FLAT_BOT_KEY_TO_GROUP_PATH = released_shared_bot_paths()
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
            if flat in _RELEASED_SHARED_BOUND_PATHS:
                return (*parts[:-1], side, *_RELEASED_SHARED_BOUND_PATHS[flat])
            for group, mapping in SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY.items():
                if flat in mapping:
                    local = mapping[flat]
                    local_parts = (["weights_minutes", local.removeprefix("weights_minutes_")]
                                   if local.startswith("weights_minutes_") else [local])
                    parts[-1:] = [side, group, *local_parts]
                    break
    if parts[-2:] == ["risk", "entry_cooldown_minutes"]:
        parts[-2:] = ["entry_cooldown", "base_duration_minutes"]
    for owner in ("position_exposure_enforcer", "total_exposure_enforcer"):
        if parts[-2:] == ["risk", f"{owner}_threshold"]:
            parts[-1:] = [owner, "threshold_ratio"]
        elif parts[-2:] == [owner, "threshold"]:
            parts[-1] = "threshold_ratio"
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
    from config.public_names import released_public_path

    from .released_v8_strategy import released_strategy_bound_path, released_strategy_selector

    parts = released_strategy_bound_path(parts, strategy_kind=strategy_kind)
    parts = released_strategy_selector(parts, strategy_kind=strategy_kind)
    return released_public_path(parts)


def reject_v8_alias_conflicts(source):
    """Reject two authored spellings of one value before precedence can erase it."""
    from config.authored import _METADATA_KEYS

    from .released_v8_strategy import released_strategy_kind

    kind = released_strategy_kind(source, default="trailing_martingale")
    seen = {}

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
            path = canonical_released_v8_path(parts, strategy_kind=kind)
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


def released_runtime_selector(path, *, strategy_kind=None):
    """Convert an explicit released selector, including a coin-document prefix.

    Ordinary loading never calls this adapter. Referenced coin documents and
    authored override selectors must use the same conversion before lookup.
    """
    FLAT_BOT_KEY_TO_GROUP_PATH = released_shared_bot_paths()
    from config.public_names import released_public_path
    from .operational_live import operational_live_selector
    from .shared_policy import shared_policy_selector

    reject_retired_gpu_selector(path, context="optimizer selector")
    from .released_v8_strategy import released_strategy_selector

    parts = list(released_strategy_selector(path.split("."), strategy_kind=strategy_kind))
    if (len(parts) >= 2 and parts[0] in {"long", "short"}
            and parts[1] in FLAT_BOT_KEY_TO_GROUP_PATH):
        parts.insert(0, "bot")
    # Only a flat side-owned shared field needs grouping. Strategy and
    # already grouped paths keep their own canonical leaf names.
    for offset in range(len(parts) - 2):
        if (parts[offset] == "bot" and parts[offset + 1] in {"long", "short"}
                and parts[offset + 2] in FLAT_BOT_KEY_TO_GROUP_PATH):
            parts[offset + 2:offset + 3] = FLAT_BOT_KEY_TO_GROUP_PATH[parts[offset + 2]]
            break
    # Released fixed selectors may be suffixes, group-relative, or fully
    # qualified. Preserve their prefix while converting the feature leaf.
    parts = [segment for part in parts for segment in (
        part.rsplit("_", 1) if part in {
            "position_exposure_enforcer_enabled", "position_exposure_enforcer_threshold",
            "total_exposure_enforcer_enabled", "total_exposure_enforcer_threshold",
            "total_exposure_enforcer_policy",
        } else (part,)
    )]
    if (len(parts) >= 2 and parts[-2] in {
            "position_exposure_enforcer", "total_exposure_enforcer"}
            and parts[-1] == "threshold"):
        native = ("risk_wel_enforcer_threshold" if parts[-2].startswith("position_")
                  else "risk_twel_enforcer_threshold")
        parts[-1] = _RELEASED_SHARED_FLAT_PATHS[native][-1]
    for group, old, new in (
        ("hsl", "red_threshold", "drawdown_ratio_threshold"),
        ("unstuck", "ema_dist", "ema_distance_fraction"),
    ):
        if parts[-2:] == [group, old]:
            parts[-1] = new
    path = ".".join(parts)
    for old, new in (
        ("live.hsl_signal_mode", "bot.portfolio.hsl.scope"),
        ("live.max_realized_loss_pct", "bot.portfolio.loss_gate.max_drawdown_ratio"),
        ("backtest.liquidation_threshold", "backtest.equity_floor_fraction"),
        ("live.auto_gs", "live.disapproved_position_mode"),
        ("bot.hsl.", "bot.portfolio.hsl."),
        (".hsl.ema_span_minutes", ".hsl.drawdown_ratio_ema_span_minutes"),
        (".risk.n_positions", ".risk.max_positions"),
        (".risk.position_exposure_enforcer_enabled", ".risk.position_exposure_enforcer.enabled"),
        (".risk.position_exposure_enforcer_threshold", ".risk.position_exposure_enforcer.threshold"),
        (".risk.total_exposure_enforcer_enabled", ".risk.total_exposure_enforcer.enabled"),
        (".risk.total_exposure_enforcer_threshold", ".risk.total_exposure_enforcer.threshold"),
        (".risk.total_exposure_enforcer_policy", ".risk.total_exposure_enforcer.policy"),
        (".unstuck.threshold", ".unstuck.exposure_trigger_ratio"),
    ):
        path = path.replace(old, new)
    return ".".join(released_public_path(
        operational_live_selector(shared_policy_selector(path)).split(".")
    ))


def finalize_v8_layout(source, *, span, changes, acknowledge_search_backend_change=False,
                       sparse=False, best_effort=False, base_config_path=""):
    """Translate public fields without guessing independent optimizer ranges."""
    FLAT_BOT_KEY_TO_GROUP_PATH = released_shared_bot_paths()

    from .shared_policy import (COIN_SOURCE_FIELDS, is_shared_approved_source, normalize_released_coin_source,
                                minimum_cost_policy_targets, move_shared_policy, preserve_forced_normal_selection,
                                preserve_forced_selection_overrides)
    from .operational_live import move_operational_live
    from config.public_names import rewrite_released_public_names

    result = deepcopy(source)
    from .released_v8_strategy import RELEASED_V8_STRATEGY_PATH_RENAMES, released_strategy_kind

    released_kind = released_strategy_kind(
        result, default=None if sparse else "trailing_martingale"
    )
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

    # Normalize frozen strategy IDs before unstuck migration groups flat bounds
    # through current native metadata. The final public-name pass is idempotent.
    from .released_v8_strategy import rewrite_released_strategy_bounds

    rewrite_released_strategy_bounds(result, strategy_kind=released_kind, record=record)

    # Released flat bounds are source spellings, independent of current optimizer IDs.
    bound_values = result.get("optimize", {}).get("bounds", {})
    if isinstance(bound_values, dict):
        for side in ("long", "short"):
            for old, target in _RELEASED_SHARED_BOUND_PATHS.items():
                source_key = f"{side}_{old}"
                if source_key not in bound_values:
                    continue
                value = bound_values.pop(source_key)
                put(bound_values, (side, *target), value, f"optimize.bounds.{source_key}")
                record(f"optimize.bounds.{source_key}", value,
                       {f"optimize.bounds.{side}.{'.'.join(target)}": value},
                       "explicit released-v8 shared bound source conversion")

    def policy_scope(node, inherited):
        mode = node.get("live", {}).get("hsl_signal_mode", inherited)
        mode = node.get("bot", {}).get("portfolio", {}).get("hsl", {}).get("scope", mode)
        return "portfolio" if mode == "unified" else mode

    def memory_paths(scope):
        paths = [("bot", "portfolio", "unstuck", "memory_ema_span_minutes"),
                 ("bot", "portfolio", "loss_gate", "memory_ema_span_minutes")]
        return paths + ([("bot", "portfolio", "hsl", "memory_ema_span_minutes")]
                        if scope == "portfolio" else
                        [("bot", side, "hsl", "memory_ema_span_minutes") for side in ("long", "short")])

    def selector(path, *, parameter_selector=False):
        if parameter_selector:
            converted = _migrate_shared_fixed_selector(path, scope=scope)
            if converted == path and path.split(".")[-1] in {"n_positions", "threshold", "ema_span_minutes"}:
                reject_retired_gpu_selector(path, context="optimizer selector")
                return path
            path = converted
        converted = released_runtime_selector(path, strategy_kind=released_kind)
        if parameter_selector:
            from config.public_names import released_fixed_selector

            converted = released_fixed_selector(converted, strategy_kind=released_kind)
        return converted

    def patch(raw, inherited_scope, path):
        from config.scenario_overrides import normalize_scenario_overrides

        values = normalize_scenario_overrides(raw)
        scope = values.get("live.hsl_signal_mode", values.get("bot.portfolio.hsl.scope", inherited_scope))
        scope = "portfolio" if scope == "unified" else scope
        converted = {}
        for key, value in values.items():
            authored_value = deepcopy(value)
            source_parts = key.split(".")
            if (len(source_parts) in {2, 3} and source_parts[0] == "live"
                    and source_parts[1] in COIN_SOURCE_FIELDS
                    and (len(source_parts) == 2 or source_parts[2] in {"long", "short"})):
                value = normalize_released_coin_source(value, field=source_parts[1],
                    path=f"{path}.{key}", record=record, base_config_path=base_config_path,
                    side=source_parts[2] if len(source_parts) == 3 else None)
            reject_retired_gpu_selector(key, value, context=path)
            if key == "backtest.max_warmup_minutes":
                record(f"{path}.{key}", value, None, _IGNORED_BACKTEST_WARMUP_REASON)
                continue
            if key in {f"backtest.{name}" for name in _BACKTEST_FEE_OVERRIDES}:
                value = _released_fee_override(value, f"{path}.{key}")
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
                shared = field == "approved_coins" and is_shared_approved_source(value, base_config_path=base_config_path)
                if field == "approved_coins":
                    converted["bot.global.universe.approved_coins"] = deepcopy(value) if shared else []
                for side in ("long", "short"):
                    source = None if shared else sources.get(side, [])
                    if source is None and not shared:
                        source = []
                    target = f"bot.{side}.universe.{field}"
                    if target in converted and converted[target] != source:
                        raise ValueError(f"{path}.{key} conflicts with {target}")
                    converted[target] = deepcopy(source)
                record(f"{path}.{key}", value, converted, "preserved whole-source replacement with explicit shared binding or directional pins")
                continue
            if key == "live.minimum_coin_age_days":
                targets = {f"bot.{side}.universe.minimum_coin_age_days": deepcopy(value) for side in ("long", "short")}
                for target, age in targets.items():
                    if target in converted and converted[target] != age:
                        raise ValueError(f"{path}.{key} conflicts with {target}")
                    converted[target] = age
                record(f"{path}.{key}", value, targets,
                       "side-owned age gates initial entries in every selection mode; retain pre-eligibility warmup")
                continue
            targets = minimum_cost_policy_targets(key)
            if targets:
                if type(value) is not bool:
                    raise TypeError(f"{path}.{key} must be a boolean")
                for target in targets:
                    if target in converted and converted[target] != value:
                        raise ValueError(f"{path}.{key} conflicts with {target}")
                    converted[target] = value
                record(f"{path}.{key}", value, {target: value for target in targets},
                       "shared minimum-cost policy copied to both position sides")
                continue
            new = selector(key)
            if key.startswith("live.approved_coins.") and value is None:
                value = []  # Released null selected nothing; v9 null means inheritance.
            if new == "bot.portfolio.hsl.scope" and value in {"unified", "pside"}:
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
        ("hsl_signal_mode", ("bot", "portfolio", "hsl", "scope")),
        ("max_realized_loss_pct", ("bot", "portfolio", "loss_gate", "max_drawdown_ratio")),
    ):
        if old in live:
            value = live.pop(old)
            authored_value = deepcopy(value)
            if old == "hsl_signal_mode" and value in {"unified", "pside"}:
                value = "portfolio" if value == "unified" else "position_side"
            put(result, target, value, f"live.{old}")
            record(f"live.{old}", authored_value, {".".join(target): value}, "consumer-owned portfolio risk policy")
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
            put(result, ("bot", "portfolio", "hsl", key), value, f"bot.hsl.{key}")
        record("bot.hsl", old, {"bot.portfolio.hsl": old}, "portfolio HSL authority")
    rename(result.setdefault("backtest", {}), "liquidation_threshold", "equity_floor_fraction", "backtest")

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
            if path == "backtest" and key == "max_warmup_minutes":
                record(current, node.pop(key), None, _IGNORED_BACKTEST_WARMUP_REASON)
                continue
            if path == "backtest" and key in _BACKTEST_FEE_OVERRIDES:
                value = node[key]
                node[key] = _released_fee_override(value, current)
                if isinstance(value, str):
                    record(current, value, node[key], "explicit released-v8 numeric fee representation")
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

        migrate_entry_cooldown_tree(result)
        migrate_entry_ema_tree(result)
        move_shared_policy(result, record=record, base_config_path=base_config_path)
        move_operational_live(result, record=record)
        from .control_sentinels import migrate_control_sentinels

        migrate_control_sentinels(result, record=record)
        for side in ("long", "short"):
            canonicalize_released_shared_bot_side(result.get("bot", {}).get(side))
        rewrite_released_public_names(result, record=record, strategy_kind=released_kind)
        return result
    # All released-layout structural transforms run only inside this converter.
    from . import apply_migrations, apply_backward_compatibility_renames
    from .entry_cooldown import migrate_entry_cooldown
    from .entry_ema import migrate_entry_ema_spans
    from .unstuck_ema import migrate_unstuck_ema_spans
    from config.strategy import sync_canonical_strategy_config

    for side in ("long", "short"):
        values = result.get("bot", {}).get(side, {})
        from config.shared_bot import get_grouped_path
        missing = object()
        for flat_key, path in FLAT_BOT_KEY_TO_GROUP_PATH.items():
            grouped = get_grouped_path(values, path, default=missing)
            if flat_key in values and grouped is not missing and values[flat_key] != grouped:
                raise ValueError(f"bot.{side}.{flat_key} conflicts with bot.{side}.{'.'.join(path)}")
    # Released optimizer IDs must move before legacy migration helpers consult
    # the current strategy metadata to interpret flat bounds.
    from .released_v8_strategy import rewrite_released_strategy_bounds

    rewrite_released_strategy_bounds(result, strategy_kind=released_kind, record=record)
    before = deepcopy(result)
    apply_migrations(result, verbose=False)
    apply_backward_compatibility_renames(result, verbose=False)
    migrate_entry_cooldown(result)
    move_shared_policy(result, record=record, base_config_path=base_config_path)
    migrate_entry_ema_spans(result)
    from .released_v8 import get_released_v8_defaults
    from config.strategy_spec import normalize_strategy_kind

    released = get_released_v8_defaults()
    kind = normalize_strategy_kind(
        result.get("bot", {}).get("portfolio", {}).get("strategy_kind")
        or released["live"]["strategy_kind"]
    )
    migrate_unstuck_ema_spans(
        result, verbose=False,
        strategy_defaults={side: released["bot"][side]["strategy"][kind]
                           for side in ("long", "short")},
        scenario_strategy_paths={
            key: RELEASED_V8_STRATEGY_PATH_RENAMES[kind][
                ("entry", key) if kind == "trailing_martingale" else (key,)
            ]
            for key in ("ema_span_0", "ema_span_1")
        },
    )
    move_operational_live(result, record=record)
    from .control_sentinels import migrate_control_sentinels

    migrate_control_sentinels(result, record=record)
    preserve_forced_normal_selection(result, record=record)
    for side in ("long", "short"):
        canonicalize_released_shared_bot_side(result.get("bot", {}).get(side))

    def canonicalize_coin_patches(node, path=()):
        if isinstance(node, list):
            for child in node:
                canonicalize_coin_patches(child, path)
        elif isinstance(node, dict):
            if len(path) > 2 and path[-2] == "bot" and path[-1] in ("long", "short"):
                canonicalize_released_shared_bot_side(node)
            for key, child in node.items():
                canonicalize_coin_patches(child, (*path, key))

    # Nested authored coin patches are converted here, before final strict
    # validation. Ordinary loading never resolves their released flat aliases.
    canonicalize_coin_patches(result)
    preserve_forced_selection_overrides(result, record=record)
    bounds = result.get("optimize", {}).get("bounds", {})
    if "hsl" in bounds:
        old = bounds.pop("hsl")
        put(bounds, ("portfolio", "hsl"), old, "optimize.bounds.hsl")
        record("optimize.bounds.hsl", old, {"portfolio.hsl": old}, "portfolio HSL bound authority")
    rewrite_released_public_names(result, record=record, strategy_kind=kind)
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
    from config.optimize_bounds import flatten_optimize_bounds, set_flat_optimize_bound
    kind = result.get("bot", {}).get("portfolio", {}).get("strategy_kind", "trailing_martingale")
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
