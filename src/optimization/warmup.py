from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from typing import Sequence

from backtest_universe import backtest_side_enabled
from config.bot import normalize_forager_score_weights
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import require_existing_config_path, resolve_dotted_config_path, resolve_optimizer_key_path
from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH, canonicalize_shared_bot_side, get_grouped_path
from optimizer_overrides import (
    optimizer_overrides,
    materialize_coupled_scenario_spans,
    optimizer_numeric_group_source_side,
)
from optimization.bounds import Bound, enforce_bounds
from optimization.config_adapter import (
    extract_bounds_tuple_list_from_config,
    get_optimization_key_paths,
    resolve_optimization_bound_path,
)
from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY, get_anchor_plan
from optimization.shape import build_optimization_shape
from warmup_utils import compute_per_coin_warmup_minutes


def _apply_config_overrides(
    config: dict, overrides: dict, *, preserve_prepared_universe=False,
    preserve_derived_outputs=True,
) -> None:
    from config.bot_inheritance import get_bot_policy
    if get_bot_policy(config) is not None:
        _apply_inherited_config_overrides(
            config, overrides, preserve_prepared_universe=preserve_prepared_universe,
            preserve_derived_outputs=preserve_derived_outputs,
        )
        return
    from config.universe import (
        ResolvedCoinList,
        raw_side_universe_sources,
        restore_universe_sources,
        universe_sources_affected,
    )

    affected = universe_sources_affected(overrides)
    prepared = {}
    if affected and preserve_prepared_universe:
        shared = deepcopy(config["bot"]["global"]["universe"]["approved_coins"])
        sources = {
            field: raw_side_universe_sources(config, field)
            for field in ("approved_coins", "ignored_coins")
        }
        prepared = {
            (field, side): deepcopy(value)
            for side in ("long", "short")
            for field in ("approved_coins", "ignored_coins")
            if type(value := config["bot"][side]["universe"].get(field)) is ResolvedCoinList
        }
    if affected:
        restore_universe_sources(config)
    if not overrides:
        return
    from config.hsl import validate_override_paths

    validate_override_paths(config, overrides)
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str):
            raise ValueError("Override keys must be dotted strings")
        resolved = require_existing_config_path(config, dotted_path)
        parts = list(resolved)
        target = config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    from config.activation import apply_activation_patch
    apply_activation_patch(config, overrides)
    if prepared:
        same_policy = shared == config["bot"]["global"]["universe"]["approved_coins"] and all(
            sources[field] == raw_side_universe_sources(config, field) for field in sources
        )
        if not same_policy:
            if any(config.get("backtest", {}).get("coins", {}).values()):
                raise ValueError(
                    "optimizer universe source policy changed; prepare its membership before candidate replay"
                )
        else:
            for (field, side), value in prepared.items():
                if (
                    field == "approved_coins"
                    and value.source is None
                    and value.shared_source != shared
                ):
                    raise ValueError(
                        "optimizer inherited universe source requires fresh preparation"
                    )
                config["bot"][side]["universe"][field] = value
    # Public fixed/scenario patches stay canonical until explicit runtime lowering.


def _apply_inherited_config_overrides(config, overrides, *, preserve_prepared_universe,
                                      preserve_derived_outputs=True):
    from config.bot_inheritance import (
        _get, _intersects, _set, _thaw, apply_bot_config_patch, apply_bot_policy_patch,
        get_bot_policy, reconciled_bot_policy, resolve_bot_policy,
        install_bot_policy, with_prepared_bot_sources, with_normalized_bot_values,
    )
    from config.inherited_bounds import PreparedOptimizationBounds
    from config.hsl import validate_override_paths
    from config.universe import ResolvedCoinList

    policy_source = reconciled_bot_policy(config)
    if overrides and preserve_derived_outputs and (policy_source._derived or policy_source._normalized):
        # Runtime patches follow candidate transforms. Freeze their existing
        # outputs only in this disposable candidate so a later declaration
        # cannot erase an unrelated mirrored value. The reusable source and
        # its binding plan remain unchanged outside this candidate.
        candidate = deepcopy(config)
        snapshot = resolve_bot_policy(policy_source.candidate_view(), policy_source.catalog)
        snapshot = with_prepared_bot_sources(snapshot, candidate["bot"])
        snapshot = with_normalized_bot_values(snapshot, candidate["bot"])
        install_bot_policy(candidate, snapshot, effective_bot=candidate["bot"])
        canonical = _apply_inherited_config_overrides(
            candidate, overrides, preserve_prepared_universe=preserve_prepared_universe,
            preserve_derived_outputs=False,
        )
        # The snapshot determines effective patch precedence, but its derived
        # values must not become authored fixed inputs in a scenario's plan.
        effective_bot = candidate["bot"]
        bounds_source = config.get("optimize", {}).get("bounds")
        if type(bounds_source) is not PreparedOptimizationBounds:
            bounds_source = None
        install_bot_policy(candidate, policy_source)
        apply_bot_config_patch(candidate, canonical, policy_source=policy_source,
                               bounds_source=bounds_source)
        declared = [tuple(key.split(".")) for key in canonical
                    if key == "bot" or key.startswith("bot.")]
        retained = {".".join(entry.path): _get(effective_bot, entry.path[1:])
                    for entry in policy_source._derived
                    if not any(_intersects(entry.path, path) for path in declared)}
        policy = apply_bot_policy_patch(get_bot_policy(candidate), retained, channel="derived")
        policy = with_prepared_bot_sources(policy, effective_bot)
        # Formatter output belongs to unchanged logical inputs. A weight's
        # normalization depends on its complete group, including inherited pins.
        before = replace(policy_source, _normalized=()).effective_view()
        after = replace(policy, _normalized=()).effective_view()
        def same_input(left, right):
            if type(left) is not type(right):
                return False
            if isinstance(left, dict):
                return left.keys() == right.keys() and all(same_input(left[key], right[key]) for key in left)
            if isinstance(left, list):
                return len(left) == len(right) and all(same_input(a, b) for a, b in zip(left, right))
            return left == right
        changed_weights = set()
        for entry in policy_source._normalized:
            dependency = (entry.path[:4] if entry.path[2:4] == ("forager", "score_weights")
                          else entry.path)
            if same_input(_get(before, dependency[1:]), _get(after, dependency[1:])):
                _set(effective_bot, entry.path[1:], _thaw(entry.value))
            elif dependency[-1] == "score_weights":
                changed_weights.add(dependency)
        for path in changed_weights:
            _set(effective_bot, path[1:], normalize_forager_score_weights(
                _get(after, path[1:]), path=".".join(path),
            ))
        policy = with_normalized_bot_values(policy, effective_bot)
        install_bot_policy(candidate, policy, effective_bot=effective_bot)
        config.clear()
        config.update(candidate)
        return canonical
    bounds_source = config.get("optimize", {}).get("bounds")
    if type(bounds_source) is not PreparedOptimizationBounds:
        bounds_source = None
    prepared = {(field, side): deepcopy(value)
                for side in ("long", "short") for field in ("approved_coins", "ignored_coins")
                if preserve_prepared_universe and
                type(value := config["bot"][side]["universe"].get(field)) is ResolvedCoinList}
    validate_override_paths(config, overrides)
    canonical = {}
    for selector, value in overrides.items():
        path = resolve_dotted_config_path(config, selector)
        # Prepared domains are a runtime projection: an authored common range
        # need not have a corresponding global container in that projection.
        # The binding owner validates every declared field and range below.
        if bounds_source is not None and path and path[:2] == ("optimize", "bounds"):
            resolved = path
        else:
            resolved = require_existing_config_path(config, selector)
        canonical[".".join(resolved)] = deepcopy(value)
    for selector, value in canonical.items():
        is_bound_patch = selector == "optimize.bounds" or selector.startswith("optimize.bounds.")
        if (not selector.startswith("bot.") and selector != "bot"
                and not (bounds_source is not None and is_bound_patch)):
            _set_path(config, selector.split("."), value)
    apply_bot_config_patch(config, canonical, policy_source=policy_source, bounds_source=bounds_source)
    _restore_matching_optimizer_memberships(config, prepared, require_match=True)
    return canonical


def _restore_matching_optimizer_memberships(config, prepared, *, require_match):
    from config.universe import (materialize_side_universe_sources,
        raw_side_universe_sources, shared_universe_source, source_inherits_shared)

    for field in ("approved_coins", "ignored_coins"):
        sources = raw_side_universe_sources(config, field)
        shared = shared_universe_source(config, field)
        compatible = {}
        for (name, side), value in prepared.items():
            if name != field:
                continue
            if (value.source_present == (side in sources) and value.source == sources.get(side)
                    and (not source_inherits_shared(field, sources, side) or value.shared_source == shared)):
                compatible[side] = value
            elif require_match and any(config.get("backtest", {}).get("coins", {}).values()):
                raise ValueError("optimizer universe source policy changed; prepare its membership before candidate replay")
        if compatible:
            materialize_side_universe_sources(config, field, compatible, sources=sources)


def apply_optimizer_universe_sources(config: dict) -> None:
    """Install fixed source policy before acquiring an optimizer context's membership."""
    from config.universe import (
        normalize_universe_source_bindings,
        raw_side_universe_sources,
        set_side_universe_sources,
        universe_sources_affected,
    )

    overrides = config.get("optimize", {}).get("fixed_runtime_overrides", {})
    if not universe_sources_affected(overrides):
        return
    from config.bot_inheritance import get_bot_policy
    if get_bot_policy(config) is not None:
        _install_inherited_optimizer_sources(config, overrides)
        return
    effective = deepcopy(config)
    _apply_config_overrides(effective, overrides)
    config["bot"]["global"]["universe"]["approved_coins"] = deepcopy(
        effective["bot"]["global"]["universe"]["approved_coins"]
    )
    for field in ("approved_coins", "ignored_coins"):
        set_side_universe_sources(config, field, raw_side_universe_sources(effective, field))
    normalize_universe_source_bindings(config)


def _install_inherited_optimizer_sources(config, overrides):
    """Install only fixed source declarations before their one acquisition owner."""
    from config.bot_inheritance import get_bot_policy, install_bot_policy, resolve_bot_policy
    from config.universe import (COIN_SOURCE_FIELDS, ResolvedCoinList,
        normalize_universe_source_bindings, universe_source_view)

    policy = get_bot_policy(config)
    prepared = {(field, side): deepcopy(value)
                for side in ("long", "short") for field in COIN_SOURCE_FIELDS
                if type(value := config["bot"][side]["universe"].get(field)) is ResolvedCoinList}
    effective = deepcopy(config)
    _apply_config_overrides(effective, overrides)
    source = universe_source_view(config)
    replacement = universe_source_view(effective)
    for scope in ("global", "long", "short"):
        fields = replacement.get("bot", {}).get(scope, {}).get("universe", {})
        current = source.get("bot", {}).get(scope, {}).get("universe", {})
        for field in COIN_SOURCE_FIELDS:
            if field in fields:
                source["bot"].setdefault(scope, {}).setdefault("universe", {})[field] = deepcopy(fields[field])
            else:
                current.pop(field, None)
    normalize_universe_source_bindings(source)
    install_bot_policy(config, resolve_bot_policy(source["bot"], policy.catalog))
    _restore_matching_optimizer_memberships(config, prepared, require_match=False)


def _set_path(config: dict, path: Sequence[str], value) -> None:
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def _try_get_path(config: dict, path: Sequence[str]):
    target = config
    for part in path:
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]
    return target


def _canonical_dead_value(flat_bounds: dict, bound_key: str, canonical_value: float):
    raw_bound = flat_bounds.get(bound_key)
    if raw_bound is None:
        return canonical_value
    bound = Bound.from_config(bound_key, raw_bound)
    return enforce_bounds([canonical_value], [bound], sig_digits=None)[0]


def optimizer_dead_param_values(
    config: dict,
    *,
    globally_dead_only: bool = False,
) -> dict[str, float]:
    """Return exact canonical values for optimizer genes made semantically dead."""

    portfolio_cfg = config.get("bot", {}).get("portfolio", {})
    strategy_kind = (
        str(portfolio_cfg.get("strategy_kind") or "trailing_martingale").strip().lower()
    )
    if strategy_kind != "trailing_martingale":
        return {}
    flat_bounds = flatten_optimize_bounds(
        config.get("optimize", {}).get("bounds", {}),
        strategy_kind=strategy_kind,
    )
    result: dict[str, float] = {}
    fixed_paths = set()
    if globally_dead_only:
        for dotted_path in (
            config.get("optimize", {}).get("fixed_runtime_overrides", {}) or {}
        ):
            fixed_paths.add(tuple(require_existing_config_path(config, dotted_path)))
    for pside in ("long", "short"):
        close_root = ("bot", pside, "strategy", strategy_kind, "close")
        retracement_base_path = (*close_root, "retracement_base_fraction")
        try:
            retracement_base = float(_try_get_path(config, retracement_base_path))
        except (TypeError, ValueError):
            continue
        if retracement_base > 0.0:
            continue
        raw_base_bound = flat_bounds.get(f"{pside}_close_retracement_base_pct")
        base_bound = (
            None
            if raw_base_bound is None
            else Bound.from_config(
                f"{pside}_close_retracement_base_pct", raw_base_bound
            )
        )
        if (
            globally_dead_only
            and tuple(retracement_base_path) not in fixed_paths
            and (base_bound is None or base_bound.high > 0.0)
        ):
            continue
        canonical_base = _canonical_dead_value(
            flat_bounds,
            f"{pside}_close_retracement_base_pct",
            0.0,
        )
        if float(canonical_base) > 0.0:
            # A fixed runtime override may intentionally disable retracement
            # outside a positive-only search bound. The exact effective value,
            # not the inactive gene bound, owns that case.
            canonical_base = retracement_base
        result[f"{pside}_close_retracement_base_pct"] = float(canonical_base)
        for field in (
            "retracement_volatility_1h_weight",
            "retracement_volatility_1m_weight",
        ):
            path = (*close_root, field)
            if _try_get_path(config, path) is None:
                continue
            bound_key = f"{pside}_close_{field}"
            result[bound_key] = float(
                _canonical_dead_value(flat_bounds, bound_key, 0.0)
            )
    return result


def canonicalize_dead_optimizer_params(config: dict) -> None:
    from config.bot_inheritance import apply_bot_config_patch, get_bot_policy
    changes = {}
    for bound_key, value in optimizer_dead_param_values(config).items():
        # These are derived effective-side values, not authoring gene writes.
        path = resolve_optimizer_key_path(config, bound_key)
        if path is not None:
            changes[".".join(path)] = value
    if get_bot_policy(config) is not None:
        apply_bot_config_patch(config, changes, channel="derived")
    else:
        for selector, value in changes.items():
            _set_path(config, selector.split("."), value)


def _finalize_optimizer_vector_config(config: dict, overrides_list=None, *, fixed_paths=()) -> dict:
    # Internal templates may retain derived aliases while genes and anchors
    # update canonical leaves. Consume aliases before pins and mirroring, with
    # existing canonical values taking precedence over stale transport copies.
    for pside in ("long", "short"):
        side_cfg = config.get("bot", {}).get(pside)
        if not isinstance(side_cfg, dict):
            continue
        missing = object()
        for flat_key, path in FLAT_BOT_KEY_TO_GROUP_PATH.items():
            if get_grouped_path(side_cfg, path, default=missing) is not missing:
                side_cfg.pop(flat_key, None)
        canonicalize_shared_bot_side(side_cfg, path_prefix=("bot", pside))
    config.pop(ANCHOR_PLAN_KEY, None)
    runtime_overrides = config.get("optimize", {}).get("fixed_runtime_overrides", {})
    _apply_config_overrides(
        config, runtime_overrides, preserve_prepared_universe=True,
        preserve_derived_outputs=False,
    )
    fixed_value_paths = set(fixed_paths) | {
        require_existing_config_path(config, selector) for selector in runtime_overrides
    }
    if overrides_list is None:
        overrides_list = config.get("optimize", {}).get("enable_overrides", [])
    config = optimizer_overrides(overrides_list or [], config, None)
    # File and scenario materialization validates canonical configuration trees.
    # Runtime transport aliases are produced later by explicit compilation.
    for pside in (side for side in ("long", "short") if side in config.get("bot", {})):
        config = optimizer_overrides(overrides_list or [], config, pside)
    from config.bot_inheritance import get_bot_policy, install_bot_policy, with_normalized_bot_values
    normalization_policy = get_bot_policy(config)
    origins = ({origin.target: origin for origin in normalization_policy.resolve()[1]}
               if normalization_policy is not None else {})
    for pside in ("long", "short"):
        pside_cfg = config.get("bot", {}).get(pside, {})
        forager_cfg = pside_cfg.get("forager") if isinstance(pside_cfg, dict) else None
        if not isinstance(forager_cfg, dict) or "score_weights" not in forager_cfg:
            continue
        weights = forager_cfg["score_weights"]
        flat_bounds = flatten_optimize_bounds(
            config.get("optimize", {}).get("bounds", {}),
            strategy_kind=config["bot"]["portfolio"]["strategy_kind"],
        )
        source_side = optimizer_numeric_group_source_side(
            config, pside, "forager", overrides_list=overrides_list,
        )
        # Preserve fixed raw zero genes; runtime scoring still applies its existing
        # EMA-readiness fallback. Searched source groups normalize here, and
        # mirrored targets inherit that source's domain, including at all-low.
        if not any(weights.values()) and all(
            ("bot", source_side, "forager", "score_weights", name) in fixed_value_paths
            or ((origin := origins.get(("bot", source_side, "forager", "score_weights", name))) is not None
                and origin.source in fixed_value_paths)
            or (bound := Bound.from_config(key, flat_bounds[key])).low == bound.high == 0
            for name in weights
            if (key := f"{source_side}_forager_score_weights_{name}") in flat_bounds
        ):
            continue
        forager_cfg["score_weights"] = normalize_forager_score_weights(
            weights,
            path=f"bot.{pside}.forager.score_weights",
        )
    if normalization_policy is not None:
        normalized = with_normalized_bot_values(normalization_policy, config["bot"])
        install_bot_policy(config, normalized, effective_bot=config["bot"])
    canonicalize_dead_optimizer_params(config)
    materialize_coupled_scenario_spans(config)
    from config.hsl import normalize_hsl
    from config.schema import get_expanded_config_prototype

    normalization_policy = get_bot_policy(config)
    normalize_hsl(config, get_expanded_config_prototype(), verbose=False)
    if normalization_policy is not None:
        normalized = with_normalized_bot_values(normalization_policy, config["bot"])
        install_bot_policy(config, normalized, effective_bot=config["bot"])
    return config


def _build_anchored_optimizer_vector_config(
    vector: Sequence[float],
    template: dict,
    *,
    overrides_list=None,
) -> dict:
    plan = get_anchor_plan(template)
    if plan is None:
        raise ValueError("anchored fine-tune plan is missing anchors")
    key_paths = [tuple(item) for item in plan.get("key_paths") or []]
    expected_len = 1 + len(key_paths)
    assert (
        len(vector) == expected_len
    ), f"anchored individual length {len(vector)} does not match expected {expected_len}"
    anchors = plan["anchors"]
    anchor_id = int(round(float(vector[0])))
    anchor_id = max(0, min(len(anchors) - 1, anchor_id))
    anchor = anchors[anchor_id]
    config = deepcopy(template)
    config.pop(ANCHOR_PLAN_KEY, None)
    # Carry the selected plan's fixed leaf authority through finalization.
    fixed_paths = set()
    from config.bot_inheritance import reconciled_bot_policy, install_bot_policy
    from config.inherited_bounds import get_bound_plan
    binding_plan = get_bound_plan(config)
    bindings = ({binding.authoring_path: binding for binding in binding_plan.bindings}
                if binding_plan is not None else {})
    values = {}
    for item in anchor.get("fixed_values") or []:
        path = tuple(item.get("path") or ())
        if not path:
            continue
        if binding_plan is not None:
            binding = bindings[path]
            values[binding.key] = deepcopy(item.get("value"))
            fixed_paths.update(binding.effective_targets)
        else:
            _set_path(config, path, deepcopy(item.get("value")))
        fixed_paths.add(path)
    for value, path in zip(vector[1:], key_paths):
        if binding_plan is not None:
            values[bindings[path].key] = value
        else:
            _set_path(config, path, value)
    if binding_plan is not None:
        install_bot_policy(config, binding_plan.apply_values(reconciled_bot_policy(config), values))
    config["_optimizer_anchor"] = {
        "id": anchor_id,
        "source": anchor.get("source"),
        "tunable_keys": list(plan.get("tunable_keys") or []),
        "fixed_keys": list(plan.get("fixed_keys") or []),
    }
    return _finalize_optimizer_vector_config(
        config, overrides_list=overrides_list, fixed_paths=fixed_paths,
    )


def build_optimizer_vector_config(
    vector: Sequence[float],
    template: dict,
    *,
    key_paths=None,
    overrides_list=None,
) -> dict:
    if get_anchor_plan(template) is not None:
        return _build_anchored_optimizer_vector_config(
            vector,
            template,
            overrides_list=overrides_list,
        )
    config = deepcopy(template)
    if key_paths is None:
        key_paths = get_optimization_key_paths(config)
    assert len(vector) == len(
        key_paths
    ), f"individual length {len(vector)} does not match optimization key count {len(key_paths)}"
    from config.bot_inheritance import reconciled_bot_policy, install_bot_policy
    from config.inherited_bounds import get_bound_plan
    plan = get_bound_plan(config)
    if plan is not None:
        declared = {binding.key: binding.authoring_path for binding in plan.bindings}
        if any(declared.get(key) != tuple(path) for key, path in key_paths):
            raise ValueError("optimizer key paths do not match the inherited binding plan")
        updated = plan.apply_values(reconciled_bot_policy(config), {key: value for value, (key, _) in zip(vector, key_paths)})
        install_bot_policy(config, updated)
    else:
        for value, (_, path) in zip(vector, key_paths):
            _set_path(config, path, value)
    return _finalize_optimizer_vector_config(config, overrides_list=overrides_list)


def build_optimizer_max_config(config: dict) -> dict:
    bounds = extract_bounds_tuple_list_from_config(config)
    if not bounds:
        return _finalize_optimizer_vector_config(
            deepcopy(config),
            overrides_list=config.get("optimize", {}).get("enable_overrides", []) or [],
        )
    overrides_list = config.get("optimize", {}).get("enable_overrides", []) or []
    shape = build_optimization_shape(config)
    max_vector = [bound.high for bound in shape.bounds]
    return build_optimizer_vector_config(
        max_vector,
        config,
        key_paths=shape.key_paths,
        overrides_list=overrides_list,
    )


def _build_optimizer_boundary_configs(config: dict, *, rms_consumer_corner: bool = False) -> list[dict]:
    anchor_plan = get_anchor_plan(config)
    if anchor_plan is None and not rms_consumer_corner:
        return [build_optimizer_max_config(config)]
    shape = build_optimization_shape(config)
    overrides_list = config.get("optimize", {}).get("enable_overrides", []) or []
    vector = [bound.high for bound in shape.bounds]
    if rms_consumer_corner:
        # Ranking may need fewer slots; adverse cooldown may need a lower
        # base/floor. Use that consumer-enabling corner with maximal weights
        # and ceiling, then finalize so runtime overrides and anchor pins win.
        for i, ((_, path), bound) in enumerate(zip(shape.key_paths, shape.bounds)):
            if path and path[-1] in ("base_duration_minutes", "min_duration_minutes"):
                vector[i] = bound.low
            if path and path[-1] == "max_positions":
                value = bound.quantize(max(1.0, bound.low))
                if round(value) <= 0 and bound.is_stepped and bound.max_index > 0:
                    value = bound.quantize(bound.low + bound.step)
                vector[i] = value
    vectors = (
        [[float(anchor_id), *vector[1:]] for anchor_id in range(len(anchor_plan.get("anchors") or []))]
        if anchor_plan is not None else [vector]
    )
    return [
        build_optimizer_vector_config(
            candidate_vector, config, key_paths=shape.key_paths, overrides_list=overrides_list,
        )
        for candidate_vector in vectors
    ]


def validate_optimizer_effective_configs(config: dict) -> None:
    """Validate effective optimizer boundary policy before backend work."""

    from config.bot import normalize_bot_forager_config
    from config.validate import validate_config

    for candidate in _build_optimizer_boundary_configs(config):
        effective = deepcopy(candidate)
        effective.setdefault("optimize", {})["fixed_runtime_overrides"] = {}
        # Validate the canonical scoring view without changing raw candidate genes.
        # Reapplying search bounds here would undo runtime pins.
        effective["optimize"]["bounds"] = {}
        normalize_bot_forager_config(effective, verbose=False)
        validate_config(
            effective,
            raw_optimize=effective.get("optimize", {}),
            verbose=False,
            tracker=None,
        )


def validate_optimizer_dataset_intervals(config: dict, mss: dict, exchange: str) -> None:
    """Reject reachable RMS consumers only after per-coin eligibility is known."""
    if config["backtest"]["candle_interval_minutes"] == 1:
        return
    from backtest import prep_backtest_args
    from warmup_utils import rms_side_enabled
    from config.entry_cooldown import uses_adverse_rms
    from config.activation import side_enabled

    for candidate in _build_optimizer_boundary_configs(config, rms_consumer_corner=True):
        bot_params, _, _, _ = prep_backtest_args(candidate, mss, exchange)
        for side in ("long", "short"):
            eligible = sum(pair[side]["entry_eligible"] for pair in bot_params)
            for pair in bot_params:
                params = pair[side]
                ranking_possible = eligible > 1 and (
                    candidate["backtest"]["dynamic_wel_by_tradability"]
                    or eligible > bot_params[0][side]["n_positions"]
                )
                if (
                    side_enabled(candidate, side)
                    and
                    params["entry_eligible"]
                    and rms_side_enabled(params, side)
                    and (
                        (ranking_possible and params["forager_score_weights"]["unilateralness"] > 0.0)
                        or uses_adverse_rms(params)
                    )
                ):
                    raise ValueError(
                        "RMS unilateralness requires completed one-minute candles for the entire optimizer search"
                    )


def refresh_optimizer_source_bindings(config: dict) -> None:
    """Bind acquired membership facts at the owning preparation boundary."""
    from config.bot_inheritance import get_bot_policy
    from config.inherited_bounds import get_bound_plan, prepare_bound_plan

    if get_bot_policy(config) is not None and get_bound_plan(config) is not None:
        prepare_bound_plan(config)


def build_optimizer_data_config(config: dict) -> dict:
    """Return a copy whose side gates cover every optimizer-reachable side."""
    boundary_configs = _build_optimizer_boundary_configs(config)
    if not boundary_configs:
        return deepcopy(config)
    data_config = deepcopy(config)
    # Dataset sizing must see runtime pins already applied, just as candidates do.
    _apply_config_overrides(
        data_config, config.get("optimize", {}).get("fixed_runtime_overrides", {}),
        preserve_prepared_universe=True,
    )
    gate_values = {}
    for pside in ("long", "short"):
        source = next(
            (
                candidate
                for candidate in boundary_configs
                if backtest_side_enabled(candidate, pside)
            ),
            boundary_configs[0],
        )
        for field in ("n_positions", "total_wallet_exposure_limit"):
            path = resolve_optimization_bound_path(data_config, f"{pside}_{field}")
            if path is None:
                raise KeyError(
                    f"optimizer data side gate does not resolve: {pside}_{field}"
                )
            value = _try_get_path(source, path)
            if value is None:
                raise KeyError(
                    f"optimizer data side gate is missing from boundary config: {'.'.join(path)}"
                )
            gate_values[".".join(path)] = deepcopy(value)
    from config.bot_inheritance import get_bot_policy, apply_bot_policy_patch, install_bot_policy
    policy = get_bot_policy(data_config)
    if policy is not None:
        # These sizing values come from already-finalized reachable corners,
        # including runtime pins outside a sampled range. The policy owner
        # fans a shared coordinate into its inheriting sides without creating
        # new side declarations or changing the original search domains.
        install_bot_policy(data_config, apply_bot_policy_patch(policy, gate_values, channel="gene"))
    else:
        for path, value in gate_values.items():
            _set_path(data_config, path.split("."), value)
    return data_config


def compute_optimizer_per_coin_warmup_minutes(
    config: dict, *, for_trade_activation: bool = False
) -> dict:
    """Separate loaded history from the activation budget shared by candidates.

    RMS history covers the search space. RMS activation belongs to each final
    candidate's Rust entry/ranking consumer, not the shared dataset metadata.
    Other indicators retain their existing worst-case activation budget.
    """
    merged: dict[str, int] = {}
    boundary_configs = _build_optimizer_boundary_configs(config)
    if not for_trade_activation:
        # The all-high corner can clamp cooldown to a constant even though
        # lower base/floor candidates consume RMS and need its full history.
        boundary_configs += _build_optimizer_boundary_configs(config, rms_consumer_corner=True)
    for boundary_config in boundary_configs:
        warmup_map = compute_per_coin_warmup_minutes(
            boundary_config,
            for_trade_activation=for_trade_activation,
            include_rms=not for_trade_activation,
        )
        for key, value in warmup_map.items():
            merged[key] = max(int(value), int(merged.get(key, 0)))
    return merged


def compute_optimizer_backtest_warmup_minutes(config: dict) -> int:
    warmup_map = compute_optimizer_per_coin_warmup_minutes(config)
    return max((int(value) for value in warmup_map.values()), default=0)


def stamp_warmup_metadata(mss: dict, coins: Sequence[str], warmup_map: dict) -> Counter:
    """Stamp the shared trade-activation map, never the RMS history map."""
    default_warmup = int(warmup_map.get("__default__", 0))
    stamped: Counter = Counter()
    for coin in coins:
        meta = mss.get(coin)
        if not isinstance(meta, dict):
            continue
        warmup_minutes = int(warmup_map.get(coin, default_warmup))
        first_idx = int(meta.get("first_valid_index", 0))
        last_idx = int(meta.get("last_valid_index", 0))
        if first_idx > last_idx:
            trade_start = first_idx
        else:
            trade_start = min(last_idx, first_idx + warmup_minutes)
        meta["warmup_minutes"] = warmup_minutes
        meta["warmup_minutes_source"] = "activation"
        meta["trade_start_index"] = trade_start
        stamped[(warmup_minutes, trade_start)] += 1
    return stamped


__all__ = [
    "build_optimizer_data_config",
    "build_optimizer_max_config",
    "build_optimizer_vector_config",
    "canonicalize_dead_optimizer_params",
    "compute_optimizer_backtest_warmup_minutes",
    "compute_optimizer_per_coin_warmup_minutes",
    "optimizer_dead_param_values",
    "stamp_warmup_metadata",
    "validate_optimizer_effective_configs",
    "validate_optimizer_dataset_intervals",
]
