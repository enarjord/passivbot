"""Explicit offline v8 to final-v9 migration, with a diff and atomic private output."""

from argparse import ArgumentParser
from copy import deepcopy
import difflib
import json
import os
from pathlib import Path
import stat
import tempfile

from config.migrations.v9 import (
    MigrationResult, V9_SCHEMA_VERSION, _hsl_enablement_facts, _require_gate_ack,
    migrate_v8_to_v9,
)
from config.risk_memory import normalize_lookback_ema_span_minutes
from json_utils import json_dumps_streamlined


from config.access import get_optional_config_value

def _released_path_context(config):
    """Resolve v8 selectors under their released active strategy, without aliases in loading."""
    bot = config.get("bot", {})
    global_policy = bot.get("global", {})
    kind = config.get("live", {}).get("strategy_kind", "trailing_martingale")
    return {**config, "bot": {**bot, "global": {**global_policy, "strategy_kind": kind}}}


def _audit_authored_patch_conflicts(config, overrides):
    """Compare released spellings before entry migrations can choose a value."""
    from config.param_paths import resolve_dotted_config_path
    from config.shared_bot import canonical_shared_bot_path_for_flat_key

    seen = {}
    path_context = _released_path_context(config)

    def visit(node, prefix=()):
        for key, value in node.items():
            parts = (*prefix, *str(key).split("."))
            if isinstance(value, dict):
                visit(value, parts)
                continue
            path = resolve_dotted_config_path(path_context, ".".join(parts))
            if path is None:
                continue
            if len(path) == 3 and path[0] == "bot" and path[1] in {"long", "short"}:
                path = canonical_shared_bot_path_for_flat_key(path[1], path[2]) or path
            if path[-2:] == ("risk", "entry_cooldown_minutes"):
                path = (*path[:-2], "entry_cooldown", "base_duration_minutes")
            if len(path) >= 2 and path[-2] == "trailing_martingale" and path[-1] in {"ema_span_0", "ema_span_1"}:
                path = (*path[:-1], "entry", path[-1])
            if path in seen and seen[path] != value:
                raise ValueError(f"override {'.'.join(parts)!r} conflicts with canonical path {'.'.join(path)}")
            seen[path] = value

    if overrides:
        visit(overrides)


def _restore_authored_export_inputs(exported, authored):
    """Keep dynamic dates, coin sources and inheritance after effective validation."""
    from config.schema import get_template_config

    defaults = get_template_config()["backtest"]
    for key in ("start_date", "end_date"):
        exported["backtest"][key] = deepcopy(authored.get("backtest", {}).get(key, defaults[key]))
    from config.universe import raw_side_universe_sources
    for key in ("approved_coins", "ignored_coins"):
        sources = raw_side_universe_sources(authored, key)
        for side, source in sources.items():
            exported["bot"][side]["universe"][key] = deepcopy(source)

    if "bot" in authored:
        exported["bot"]["global"]["universe"]["approved_coins"] = deepcopy(authored.get("bot", {}).get("global", {}).get("universe", {}).get("approved_coins", []))

    if "filter_by_min_effective_cost" in authored.get("backtest", {}):
        exported["backtest"]["filter_by_min_effective_cost"] = authored["backtest"]["filter_by_min_effective_cost"]

    preserved_paths = {("backtest", "start_date"), ("backtest", "end_date"),
                       ("backtest", "filter_by_min_effective_cost"),
                       ("bot", "global", "universe", "approved_coins"),
                       *(("bot", side, "universe", key)
                         for side in ("long", "short") for key in ("approved_coins", "ignored_coins"))}

    def restore_patch(target, source, prefix=()):
        for key, value in source.items():
            parts = (*prefix, *key.split("."))
            if parts in preserved_paths:
                target[key] = deepcopy(value)
            elif isinstance(value, dict) and key in target and isinstance(target[key], dict):
                restore_patch(target[key], value, parts)

    restore_patch(exported["optimize"]["fixed_runtime_overrides"],
                  authored.get("optimize", {}).get("fixed_runtime_overrides", {}))
    for index, scenario in enumerate(authored.get("backtest", {}).get("scenarios", [])):
        target = exported["backtest"]["scenarios"][index]
        for key in ("start_date", "end_date"):
            if key in scenario:
                target[key] = deepcopy(scenario[key])
        restore_patch(target.get("overrides", {}), scenario.get("overrides", {}))


def _set_patch(config, overrides, *, authored=False, scenario=False):
    from config.param_paths import require_existing_config_path, resolve_dotted_config_path
    from config.scenario_overrides import normalize_scenario_overrides

    if authored:
        _audit_authored_patch_conflicts(config, overrides)
    if authored and scenario:
        from config.migrations.entry_cooldown import migrate_entry_cooldown
        from config.migrations.entry_ema import migrate_entry_ema_spans

        # Normalize released entry spellings only after their authored values
        # agree. Keep raw HSL facts for the separate authorization audit.
        context = deepcopy(config)
        context["backtest"] = {"scenarios": [{"overrides": deepcopy(overrides)}]}
        migrate_entry_cooldown(context)
        migrate_entry_ema_spans(context)
        overrides = context["backtest"]["scenarios"][0]["overrides"]

    seen = set()
    path_context = _released_path_context(config) if authored else config
    for selector, value in normalize_scenario_overrides(overrides).items():
        if scenario and (selector == "optimize" or selector.startswith("optimize.")):
            raise ValueError("scenario/runtime optimizer controls must be top-level optimize settings")
        path = (resolve_dotted_config_path(path_context, selector) if authored else
                require_existing_config_path(config, selector))
        if path is None:
            # Raw retired fields must survive until the explicit migration pass.
            path = tuple(selector.split("."))
        if path in seen:
            raise ValueError(f"override {selector!r} duplicates canonical path {'.'.join(path)}")
        seen.add(path)
        target = config
        for key in path[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f"override {selector!r} crosses a non-mapping")
        target[path[-1]] = deepcopy(value)


def _materialized_coin_patch(overrides, effective):
    """Resolve files without expanding a leaf replacement to whole-map authority.

    Scenarios run over both the ordinary and optimizer-fixed roots. Replacing
    their complete coin map with the ordinary effective map would erase fixed
    modes on unrelated coins. Preserve each authored replacement scope; a
    changed file reference necessarily replaces its materialized coin document.
    """
    roots = ("coin_overrides", "live.coin_flags")
    result = {
        key: value for key, value in overrides.items()
        if not any(key == root or key.startswith(root + ".") for root in roots)
    }
    if any(key == "live.coin_flags" or key.startswith("live.coin_flags.") for key in overrides):
        result["coin_overrides"] = deepcopy(effective["coin_overrides"])
        return result
    for key in overrides:
        parts = key.split(".")
        if parts[0] != "coin_overrides":
            continue
        if "override_config_path" in parts[2:]:
            parts = parts[:2]
        from config.migrations.final_v9 import released_runtime_selector

        # A referenced coin file has already completed sparse conversion,
        # while a file-free released coin patch still uses its source paths.
        candidates = (parts, released_runtime_selector(".".join(parts)).split("."))
        for candidate in candidates:
            value = effective
            for part in candidate:
                if not isinstance(value, dict) or part not in value:
                    break
                value = value[part]
            else:
                break
        else:
            raise ValueError(
                f"{key} is missing after released coin-file materialization"
            )
        result[".".join(parts)] = deepcopy(value)
    return result


def _materialize_files(source, *, base_config_path, span, acknowledge, changes,
                       _null_policy_enablement=None, _root_enablement=None, best_effort=False,
                       _emit_guesses=True, _change_prefix=""):
    """Use normal file resolution and file-then-inline precedence without simulation."""
    from config.overrides import (
        _extract_allowed_patch,
        _unwrap_override_document,
        load_override_config,
        nested_update,
        parse_old_coin_flags,
    )
    from config.parse import load_raw_config

    if _null_policy_enablement is None:
        _null_policy_enablement = {}
    result = deepcopy(source)
    flags = parse_old_coin_flags(result)
    old_flags = result.get("live", {}).pop("coin_flags", None)
    if not result.get("coin_overrides"):
        result["coin_overrides"] = flags
    elif flags:
        # Canonical overrides already win over legacy flags in parse_overrides.
        # Still inspect every authored file before discarding dormant policy:
        # an ignored Never selector requires the same explicit acknowledgement.
        flag_context = deepcopy(result)
        flag_context["coin_overrides"] = flags
        _materialize_files(
            flag_context, base_config_path=base_config_path, span=span,
            acknowledge=acknowledge, changes=changes,
            _root_enablement=_root_enablement, best_effort=best_effort,
            _emit_guesses=_emit_guesses,
            _change_prefix=_change_prefix + ("live.coin_flags::" if best_effort else ""),
        )
    if old_flags:
        changes.append(dict(
            path=_change_prefix + "live.coin_flags", old=old_flags, new={},
            reason="materialized legacy flags; canonical coin_overrides take precedence",
        ))
    mode = result.get("live", {}).get("hsl_signal_mode", "coin")
    mode = {"unified": "portfolio", "pside": "position_side"}.get(mode, mode)
    for coin, inline in result.get("coin_overrides", {}).items():
        if not isinstance(inline, dict):
            raise ValueError(f"coin_overrides.{coin} must be a mapping")
        if not inline.get("override_config_path"):
            continue
        context = deepcopy(result)
        context.setdefault("live", {})["base_config_path"] = base_config_path
        loaded = _unwrap_override_document(
            load_override_config(context, coin, config_loader=load_raw_config),
            source=f"coin_overrides.{coin}.override_config_path", allow_legacy=True,
        )
        # Keep all raw aliases until authorization has been checked. Later
        # explicit file/inline enablement wins over inherited facts; conflicting
        # spellings inside the winning layer remain ambiguous.
        enablement = dict(_root_enablement if _root_enablement is not None else
                          _hsl_enablement_facts({"bot": result.get("bot", {})}))
        enablement.update(_hsl_enablement_facts(loaded, inline))
        enablement = {path: values for path, values in enablement.items()
                      if path[:1] == ("bot",)}
        _null_policy_enablement.update({
            ("coin_overrides", str(coin), *path): values
            for path, values in enablement.items()
        })
        # Migrate the raw document before filtering allowed coin leaves, so a
        # dormant Never policy cannot disappear as an unrecognized current field.
        envelope = {"config_version": "v8.6.0", "bot": {}, **loaded}
        file_result = migrate_v8_to_v9(
            envelope, memory_span_minutes=span,
            acknowledge_never_removal=acknowledge,
            _null_policy_enablement=enablement,
            _coin_patch=True, best_effort=best_effort,
            _guess_log_prefix=(f"{_change_prefix}coin_overrides.{coin}.override_config_path::"
                               if _emit_guesses else None),
        )
        # Coin-only conversion deliberately avoids hydrating a full config.
        # Validate its authored structure before allowed-leaf extraction can
        # discard unknown fields from the referenced document.
        from config.authored import validate_authored_config
        from config.overrides import OVERRIDABLE_STANDALONE_BOT_KEYS
        from config.schema import get_template_config

        file_schema = get_template_config()
        for side in ("long", "short"):
            # Explicit coin allocations are declared local leaves, rather than
            # defaults in the global bot-side schema.
            file_schema["bot"][side].update({key: None for key in OVERRIDABLE_STANDALONE_BOT_KEYS})
        validate_authored_config(file_result.config, file_schema)
        # Audit inline authority in this same effective context too. Audit-only
        # scenario/optimizer calls discard the materialized result, so deferring
        # inline null validation to the final root pass would miss those modes.
        # Keep raw inline leaves for that final pass's authored change paths.
        migrate_v8_to_v9(
            {"config_version": "v8.6.0", "bot": {}, **inline}, memory_span_minutes=span,
            acknowledge_never_removal=acknowledge,
            _null_policy_enablement=enablement,
            _coin_patch=True, best_effort=best_effort,
            _guess_log_prefix=None,
        )
        migrated = file_result.config
        changes.extend({**item, "path": f"{_change_prefix}coin_overrides.{coin}.override_config_path::{item['path']}"}
                       for item in file_result.changes)
        patch = _extract_allowed_patch(
            migrated, coin=coin, source=f"coin_overrides.{coin}.override_config_path",
            strict=False, strategy_kind=result.get("live", {}).get("strategy_kind", "trailing_martingale"),
            hsl_signal_mode=mode,
        )
        for side in ("long", "short"):
            universe = migrated.get("bot", {}).get(side, {}).get("universe", {})
            if "minimum_coin_age_days" in universe:
                changes.append(dict(
                    path=f"{_change_prefix}coin_overrides.{coin}.override_config_path::bot.{side}.universe.minimum_coin_age_days",
                    old=deepcopy(universe["minimum_coin_age_days"]), new=None,
                    reason="ignored shared age policy in a released coin file; side universe admission is not coin overridable",
                ))
        inline = deepcopy(inline)
        inline.pop("override_config_path")
        from config.migrations.operational_live import move_operational_live

        def record_operational(path, old, new, reason):
            changes.append(
                dict(
                    path=f"{_change_prefix}coin_overrides.{coin}.{path}",
                    old=deepcopy(old),
                    new=deepcopy(new),
                    reason=reason,
                )
            )

        move_operational_live(inline, record=record_operational)
        # Resolve structural aliases independently on each authored patch before
        # merging. Otherwise a canonicalized file value can beat the higher-
        # precedence inline legacy value during later hydration. Keep raw HSL
        # authorization leaves for the explicit v9 migration pass below.
        from config.migrations.entry_cooldown import migrate_entry_cooldown_tree
        from config.migrations.entry_ema import migrate_entry_ema_tree
        from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH, canonicalize_shared_bot_side

        migrate_entry_cooldown_tree(inline, path=f"coin_overrides.{coin}")
        migrate_entry_ema_tree(inline, path=f"coin_overrides.{coin}")
        for side in ("long", "short"):
            values = inline.get("bot", {}).get(side)
            if not isinstance(values, dict):
                continue
            for flat, path in FLAT_BOT_KEY_TO_GROUP_PATH.items():
                if flat in values:
                    changes.append(dict(
                        path=f"{_change_prefix}coin_overrides.{coin}.bot.{side}.{flat}",
                        old=deepcopy(values[flat]),
                        new={".".join(path): deepcopy(values[flat])},
                        reason="explicit released-v8 shared field conversion before file/inline merge",
                    ))
            canonicalize_shared_bot_side(values)
            risk = values.get("risk", {})
            for owner in ("position_exposure_enforcer", "total_exposure_enforcer"):
                block = risk.get(owner)
                if isinstance(block, dict) and "threshold" in block:
                    value = block.pop("threshold")
                    if "threshold_ratio" in block and block["threshold_ratio"] != value:
                        raise ValueError(f"coin_overrides.{coin}.bot.{side}.risk.{owner}.threshold conflicts with threshold_ratio")
                    block["threshold_ratio"] = deepcopy(value)
                    changes.append(dict(
                        path=f"{_change_prefix}coin_overrides.{coin}.bot.{side}.risk.{owner}.threshold",
                        old=deepcopy(value), new={"threshold_ratio": deepcopy(value)},
                        reason="explicit released-v8 ratio rename before file/inline merge",
                    ))
            for owner, leaf in (
                ("position_exposure_enforcer", "enabled"),
                ("position_exposure_enforcer", "threshold"),
                ("total_exposure_enforcer", "enabled"),
                ("total_exposure_enforcer", "threshold"),
                ("total_exposure_enforcer", "policy"),
            ):
                old = f"{owner}_{leaf}"
                if old not in risk:
                    continue
                value = risk.pop(old)
                block = risk.setdefault(owner, {})
                target = "threshold_ratio" if leaf == "threshold" else leaf
                if not isinstance(block, dict) or (target in block and block[target] != value):
                    raise ValueError(f"coin_overrides.{coin}.bot.{side}.risk.{old} conflicts with {owner}.{target}")
                block[target] = deepcopy(value)
                changes.append(dict(
                    path=f"{_change_prefix}coin_overrides.{coin}.bot.{side}.risk.{old}",
                    old=deepcopy(value), new={f"{owner}.{target}": deepcopy(value)},
                    reason="explicit released-v8 enforcer conversion before file/inline merge",
                ))
            for group, old, new in (
                ("hsl", "red_threshold", "drawdown_ratio_threshold"),
                ("unstuck", "ema_dist", "ema_distance_fraction"),
            ):
                block = values.get(group, {})
                if old not in block:
                    continue
                value = block.pop(old)
                if new in block and block[new] != value:
                    raise ValueError(f"coin_overrides.{coin}.bot.{side}.{group}.{old} conflicts with {new}")
                block[new] = value
                changes.append(dict(
                    path=f"{_change_prefix}coin_overrides.{coin}.bot.{side}.{group}.{old}",
                    old=deepcopy(value), new={new: deepcopy(value)},
                    reason="explicit released-v8 quantity rename before file/inline merge",
                ))
        from config.public_names import rewrite_released_public_names

        rewrite_released_public_names(inline, record=record_operational)
        result["coin_overrides"][coin] = nested_update(patch, inline)
    return result


def _optimizer_root_enablement(config, enable_overrides, facts):
    """Project fact-sets through the ordinary global optimizer transforms.

    The global mirror copies HSL along with the other shared groups. Carry the
    raw fact-set as its enabled value so that the actual transform preserves
    same-layer ambiguity instead of selecting a flat/grouped spelling. The
    projection is private audit state, never an effective config or export.
    """
    from optimizer_overrides import optimizer_overrides

    projection = {
        "bot": {"global": {"strategy_kind": config.get("live", {}).get("strategy_kind", "trailing_martingale")},
                **{side: {"hsl": {"enabled": facts.get(("bot", side, "hsl", "enabled"), ())}}
                   for side in ("long", "short")}},
    }
    projection = optimizer_overrides(enable_overrides, projection, None)
    resolved = dict(facts)
    for side in ("long", "short"):
        resolved[("bot", side, "hsl", "enabled")] = projection["bot"][side]["hsl"]["enabled"]
    return resolved


def migrate(source, *, memory_span_minutes=None, memory_span_choices=None,
            acknowledge_never_removal=False,
            acknowledge_realized_loss_gate_memory_change=False,
            acknowledge_search_backend_change=False, metric_replacements=None,
            acknowledge_metric_formula_change=False, base_config_path="", best_effort=False):
    """Transform and canonically validate root, files and every effective scenario."""
    from config import prepare_config
    from config.gpu import validate_hsl_gpu_inputs
    from config.overrides import parse_overrides
    from config.scenario_overrides import normalize_scenario_overrides
    from config_utils import strip_config_metadata
    from optimizer_overrides import optimizer_overrides

    payload = source.get("config", source) if isinstance(source, dict) else source
    if not isinstance(payload, dict):
        raise TypeError("configuration must be a mapping")
    inferred_changes = []
    if best_effort and payload.get("config_version") != V9_SCHEMA_VERSION:
        from config.migrations.best_effort_v9 import record_envelope_metadata, remove_result_metadata

        wrapped = "config" in source
        if wrapped:
            record_envelope_metadata(source, inferred_changes)
        payload = deepcopy(payload)
        remove_result_metadata(payload, inferred_changes, wrapped=wrapped,
                               path_prefix="config." if wrapped else "")
    if best_effort and payload.get("config_version") != V9_SCHEMA_VERSION and memory_span_minutes is None:
        from config.migrations.best_effort_v9 import infer_memory_span

        live = payload.get("live", {})
        memory_span_minutes = live.get("lookback_ema_span_minutes")
        if memory_span_minutes is None:
            memory_span_minutes = infer_memory_span(
                live.get("pnls_max_lookback_days"), "live.pnls_max_lookback_days", inferred_changes,
                missing="pnls_max_lookback_days" not in live,
            )
    if payload.get("config_version") == V9_SCHEMA_VERSION:
        transformed = migrate_v8_to_v9(
            payload, base_config_path=base_config_path, memory_span_minutes=memory_span_minutes,
            memory_span_choices=memory_span_choices,
            acknowledge_realized_loss_gate_memory_change=acknowledge_realized_loss_gate_memory_change,
            acknowledge_search_backend_change=acknowledge_search_backend_change,
            metric_replacements=metric_replacements,
            acknowledge_metric_formula_change=acknowledge_metric_formula_change,
            best_effort=best_effort,
        )
    else:
        if memory_span_minutes is None:
            # The core owns the actionable version/span error.
            return migrate_v8_to_v9(
                payload, memory_span_minutes=None, memory_span_choices=memory_span_choices,
                acknowledge_never_removal=acknowledge_never_removal,
                acknowledge_realized_loss_gate_memory_change=acknowledge_realized_loss_gate_memory_change,
                acknowledge_search_backend_change=acknowledge_search_backend_change,
                metric_replacements=metric_replacements,
                acknowledge_metric_formula_change=acknowledge_metric_formula_change,
            )
        raw_span = (memory_span_minutes if memory_span_minutes is not None else
                    payload.get("live", {}).get("lookback_ema_span_minutes"))
        span = normalize_lookback_ema_span_minutes(raw_span) if raw_span is not None else None
        file_changes = []
        null_policy_enablement = {}
        authored = _materialize_files(
            payload, base_config_path=base_config_path, span=span,
            acknowledge=acknowledge_never_removal, changes=file_changes,
            _null_policy_enablement=null_policy_enablement, best_effort=best_effort,
        )
        roots = ("coin_overrides", "live.coin_flags")
        root_policy = {"bot": payload.get("bot", {})}
        fixed = normalize_scenario_overrides(
            payload.get("optimize", {}).get("fixed_runtime_overrides"),
        )
        optimized_payload = deepcopy(payload)
        if fixed:
            _set_patch(optimized_payload, fixed, authored=True)
        enable_overrides = payload.get("optimize", {}).get("enable_overrides") or []
        audit_optimized = bool(fixed) or "mirror_short_from_long" in enable_overrides
        optimized_root_enablement = _hsl_enablement_facts(root_policy, fixed)
        if audit_optimized:
            optimized_root_enablement = _optimizer_root_enablement(
                optimized_payload, enable_overrides, optimized_root_enablement,
            )
            materialize_fixed_coins = any(
                key == root or key.startswith(root + ".") for key in fixed for root in roots
            )
            fixed_enablement = {}
            effective_fixed = _materialize_files(
                optimized_payload, base_config_path=base_config_path, span=span,
                acknowledge=acknowledge_never_removal,
                changes=file_changes if materialize_fixed_coins else [],
                _null_policy_enablement=fixed_enablement,
                _root_enablement=optimized_root_enablement, best_effort=best_effort,
                _emit_guesses=materialize_fixed_coins,
                _change_prefix="optimize.fixed_runtime_overrides." if best_effort else "",
            )
            if materialize_fixed_coins:
                null_policy_enablement.update({
                    ("optimize", "fixed_runtime_overrides", *path): values
                    for path, values in fixed_enablement.items()
                })
                fixed = _materialized_coin_patch(fixed, effective_fixed)
                authored["optimize"]["fixed_runtime_overrides"] = fixed
        # Build each effective authored scenario before root flag conversion.
        # Otherwise generated root coin_overrides would hide scenario flags.
        for container_path in (("backtest",), ("backtest", "suite")):
            container = authored
            for key in container_path:
                container = container.get(key, {})
            for index, scenario in enumerate(container.get("scenarios", [])):
                overrides = normalize_scenario_overrides(scenario.get("overrides"))
                materialize_coin_patch = any(
                    key == root or key.startswith(root + ".")
                    for key in overrides for root in roots
                )
                # Even an enablement-only scenario changes whether an inherited
                # referenced null is removable. Audit untouched file authority
                # against that effective root before discarding the reference.
                effective = deepcopy(payload)
                _set_patch(effective, overrides, authored=True, scenario=True)
                scenario_enablement = {}
                effective = _materialize_files(
                    effective, base_config_path=base_config_path, span=span,
                    acknowledge=acknowledge_never_removal,
                    changes=file_changes if materialize_coin_patch else [],
                    _null_policy_enablement=scenario_enablement,
                    _root_enablement=_hsl_enablement_facts(root_policy, overrides), best_effort=best_effort,
                    _emit_guesses=materialize_coin_patch,
                    _change_prefix=(f"{'.'.join(container_path)}.scenarios[{index}].overrides."
                                    if best_effort else ""),
                )
                if audit_optimized:
                    optimized_scenario = deepcopy(optimized_payload)
                    _set_patch(optimized_scenario, overrides, authored=True, scenario=True)
                    scenario_optimized_enablement = dict(optimized_root_enablement)
                    scenario_optimized_enablement.update(_hsl_enablement_facts(overrides))
                    _materialize_files(
                        optimized_scenario, base_config_path=base_config_path, span=span,
                        acknowledge=acknowledge_never_removal, changes=[],
                        _root_enablement=scenario_optimized_enablement, best_effort=best_effort,
                        _emit_guesses=False,
                    )
                if materialize_coin_patch:
                    prefix = (*container_path, "scenarios", str(index), "overrides")
                    null_policy_enablement.update({
                        (*prefix, *path): values
                        for path, values in scenario_enablement.items()
                    })
                    overrides = _materialized_coin_patch(overrides, effective)
                    scenario["overrides"] = overrides
        transformed = migrate_v8_to_v9(
            authored, base_config_path=base_config_path, memory_span_minutes=span, memory_span_choices=memory_span_choices,
            acknowledge_never_removal=acknowledge_never_removal,
            acknowledge_realized_loss_gate_memory_change=acknowledge_realized_loss_gate_memory_change,
            acknowledge_search_backend_change=acknowledge_search_backend_change,
            metric_replacements=metric_replacements,
            acknowledge_metric_formula_change=acknowledge_metric_formula_change,
            _null_policy_enablement=null_policy_enablement, best_effort=best_effort,
        )
        inferred_paths = {item["path"] for item in inferred_changes if item.get("best_effort")}
        changes = [item for item in transformed.changes if item["path"] not in inferred_paths]
        transformed = MigrationResult(transformed.config, inferred_changes + file_changes + changes,
                                      transformed.warnings)
    from config.universe import normalize_universe_source_bindings

    def binding_change(path, old, new):
        transformed.changes.append(dict(path=".".join(map(str, path)), old=old, new=new,
            reason="anchored retained coin-list file binding before export"))

    normalize_universe_source_bindings(transformed.config, base_config_path=base_config_path,
                                       on_change=binding_change)
    prepared = prepare_config(
        transformed.config, verbose=False, target="canonical", runtime=None,
        base_config_path=base_config_path,
    )
    prepared = parse_overrides(prepared, verbose=False)
    optimized_base = deepcopy(prepared)
    _set_patch(optimized_base, prepared["optimize"].get("fixed_runtime_overrides", {}))
    overrides = prepared["optimize"].get("enable_overrides", [])
    optimized_base = optimizer_overrides(overrides, optimized_base, None)
    for side in ("long", "short"):
        optimized_base = optimizer_overrides(overrides, optimized_base, side)
    optimized_base = prepare_config(optimized_base, verbose=False, base_config_path=base_config_path)
    effective_configs = [prepared, optimized_base]
    for scenario in prepared["backtest"].get("scenarios", []):
        for base in (prepared, optimized_base):
            effective = deepcopy(base)
            _set_patch(effective, scenario.get("overrides"), scenario=True)
            effective_configs.append(parse_overrides(prepare_config(
                effective, verbose=False, base_config_path=base_config_path,
            ), verbose=False))
    for effective in effective_configs:
        if effective["optimize"].get("backend") == "gpu":
            validate_hsl_gpu_inputs(effective)
    exported = strip_config_metadata(prepared)
    # Keep resolution context internal after referenced coin files are inlined.
    exported.get("live", {}).pop("base_config_path", None)
    _restore_authored_export_inputs(exported, transformed.config)
    return MigrationResult(exported, transformed.changes, transformed.warnings)


def _atomic_write(target, data, *, replace=False, original_stat=None):
    target = Path(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if original_stat is not None:
            current = temporary.stat()
            if (current.st_uid, current.st_gid) != (original_stat.st_uid, original_stat.st_gid):
                os.chown(temporary, original_stat.st_uid, original_stat.st_gid)
                current = temporary.stat()
                if (current.st_uid, current.st_gid) != (original_stat.st_uid, original_stat.st_gid):
                    raise OSError("cannot preserve original configuration ownership")
            temporary.chmod(stat.S_IMODE(original_stat.st_mode))
        if replace:
            os.replace(temporary, target)
        else:
            # Atomic, exclusive publication: an existing target is never replaced.
            os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = ArgumentParser(prog="passivbot tool migrate-config-v9", description=__doc__)
    parser.add_argument("input_config", type=Path)
    parser.add_argument("output_config", type=Path, nargs="?")
    parser.add_argument("--lookback-ema-span-minutes", type=float)
    parser.add_argument("--memory-span", action="append", default=[], metavar="PATH=MINUTES")
    parser.add_argument("--acknowledge-never-removal", action="store_true")
    parser.add_argument("--acknowledge-realized-loss-gate-memory-change", action="store_true",
                        help="Explicitly adopt independent decaying realized memory for every active gate context.")
    parser.add_argument("--acknowledge-search-backend-change", action="store_true")
    parser.add_argument("--metric-replacement", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--acknowledge-metric-formula-change", action="store_true")
    parser.add_argument("--best-effort", action="store_true",
                        help="Infer replacement formulas and memory policy, logging every guess; review and backtest the result.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args(argv)
    if args.in_place and args.output_config is not None:
        parser.error("--in-place cannot be combined with output_config")
    if args.backup is not None and not args.in_place:
        parser.error("--backup requires --in-place")
    try:
        from config.parse import load_raw_config

        # Resolve every destination before the first write, including reports.
        target = (args.input_config.resolve() if args.in_place else
                  args.output_config or args.input_config.with_name(
                      args.input_config.stem + ".v9" + args.input_config.suffix))
        backup = (args.backup or target.with_name(target.stem + ".pre-v9-backup" + target.suffix)
                  if args.in_place else None)
        destinations = [path for path in (target if not args.dry_run else None,
                                          backup if not args.dry_run else None,
                                          args.report) if path is not None]
        resolved = [path.resolve() for path in destinations]
        if len(resolved) != len(set(resolved)):
            raise ValueError("output, backup and report destinations must differ")
        source_path = args.input_config.resolve()
        for path in destinations:
            if args.in_place and path is target:
                continue
            if path.resolve() == source_path:
                raise ValueError("output, backup and report must differ from input")
            if path.exists() or path.is_symlink():
                raise ValueError(f"destination already exists: {path}")

        choices = {}
        for item in args.memory_span:
            path, separator, value = item.rpartition("=")
            if not separator or not path or path in choices:
                raise ValueError("supply each --memory-span once as PATH=MINUTES")
            choices[path] = float(value)
        replacements = {}
        for item in args.metric_replacement:
            old, separator, new = item.partition("=")
            if not separator or not old or not new or old in replacements:
                raise ValueError("supply each --metric-replacement once as OLD=NEW")
            replacements[old] = new
        original_bytes = args.input_config.read_bytes()
        source = load_raw_config(str(args.input_config))
        if args.input_config.resolve() != source_path or args.input_config.read_bytes() != original_bytes:
            raise ValueError("input changed while it was being read; retry migration")
        result = migrate(
            source, memory_span_minutes=args.lookback_ema_span_minutes,
            memory_span_choices=choices, acknowledge_never_removal=args.acknowledge_never_removal,
            acknowledge_realized_loss_gate_memory_change=args.acknowledge_realized_loss_gate_memory_change,
            acknowledge_search_backend_change=args.acknowledge_search_backend_change,
            metric_replacements=replacements,
            acknowledge_metric_formula_change=args.acknowledge_metric_formula_change,
            base_config_path=str(args.input_config.resolve()), best_effort=args.best_effort,
        )
        rendered = json_dumps_streamlined(result.config, allow_nan=False) + "\n"
        report = dict(source_version=source.get("config_version"), target_version=V9_SCHEMA_VERSION,
                      changes=result.changes, warnings=list(result.warnings))
        report_rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.diff:
            before = json_dumps_streamlined(source, allow_nan=False) + "\n"
            print("".join(difflib.unified_diff(before.splitlines(True), rendered.splitlines(True),
                                            fromfile="migration input", tofile="v9 output")), end="")
        if not args.dry_run:
            if args.input_config.resolve() != source_path or args.input_config.read_bytes() != original_bytes:
                raise ValueError("input changed during migration; no configuration was written")
            if args.in_place:
                original_stat = target.stat()
                _atomic_write(backup, original_bytes, original_stat=original_stat)
                _atomic_write(target, rendered.encode(), replace=True, original_stat=original_stat)
            else:
                _atomic_write(target, rendered.encode())
            print(f"Wrote {target}")
        if args.report:
            _atomic_write(args.report, report_rendered.encode())
        else:
            print(report_rendered, end="")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
