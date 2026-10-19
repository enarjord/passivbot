"""Explicit released-v8 conversion; runtime loading requires final canonical v9."""

from copy import deepcopy
from dataclasses import dataclass
import math
import re

from config.risk_memory import normalize_lookback_ema_span_minutes


V9_SCHEMA_VERSION = "v9.0.0"

MIGRATION_COMMAND = "passivbot tool migrate-config-v9"
_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$", re.IGNORECASE)
_RETIRED_HSL_FIELDS = frozenset({"tier_ratios", "orange_tier_mode", "no_restart_drawdown_threshold"})
_RETIRED_LIVE_FIELDS = frozenset({
    "hsl_position_during_cooldown_policy", "hsl_unavailable_grace_seconds",
    "hsl_accept_incomplete_history", "risk_input_max_attempts",
})


def _retired_control(path, key):
    leaf = str(key).split(".")[-1]
    hsl_leaf = leaf.removeprefix("hsl_")
    return (leaf in _RETIRED_LIVE_FIELDS or
            (hsl_leaf in _RETIRED_HSL_FIELDS and
             (re.search(r"(^|[._])hsl([._]|$)", path) or leaf.startswith("hsl_"))))


@dataclass(frozen=True)
class MigrationResult:
    config: dict
    changes: list[dict]
    warnings: tuple[str, ...]


def _walk(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            if not str(key).startswith("_"):
                current = f"{path}.{key}" if path else str(key)
                yield current, key, value
                yield from _walk(value, current)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            current = f"{path}[{index}]"
            yield current, str(index), value
            yield from _walk(value, current)


def _semantic_hsl_path(path):
    parts = tuple(re.sub(r"\[(\d+)\]", r".\1", path).split("."))
    if parts[-1] in {"hsl_enabled", "hsl_restart_after_red_policy"}:
        parts = (*parts[:-1], "hsl", parts[-1].removeprefix("hsl_"))
    return parts


def _hsl_enablement_facts(*layers):
    """Overlay explicit enablement layers without resolving same-layer aliases.

    Missing leaves inherit. An explicitly supplied later leaf replaces the
    earlier fact-set, but duplicate spellings within one layer stay visible.
    This is migration call state, never a patch to the authored configuration.
    """
    facts = {}
    for layer in layers:
        layer_facts = {}
        for path, _, value in _walk(layer):
            parts = _semantic_hsl_path(path)
            if parts[-2:] == ("hsl", "enabled"):
                layer_facts.setdefault(parts, []).append(value)
        facts.update({path: tuple(values) for path, values in layer_facts.items()})
    return facts


def require_v9_runtime_config(config, *, additional_inputs=()):
    version = config.get("config_version")
    if version != V9_SCHEMA_VERSION:
        match = _VERSION.fullmatch(version.strip()) if isinstance(version, str) else None
        if version is not None and match is None:
            raise ValueError(
                f"config.config_version must be a semantic version like {V9_SCHEMA_VERSION}; "
                f"got {version!r}. Use explicit migration for supported v8 configurations"
            )
        if match is not None and tuple(map(int, match.groups())) > (9, 0, 0):
            raise ValueError(
                f"config.config_version {version!r} is newer than supported schema {V9_SCHEMA_VERSION}; "
                "upgrade Passivbot; explicit migration cannot load a future schema"
            )
        raise ValueError(
            f"config.config_version must be {V9_SCHEMA_VERSION}; {version!r} requires "
            f"explicit migration with {MIGRATION_COMMAND} and fresh backtests"
        )
    reject_retired_v9_fields(config, additional_inputs=additional_inputs)


def reject_retired_v9_fields(config, *, additional_inputs=()):
    """Also protect sparse config exports from silently deleting old authority."""
    for node in (config, *additional_inputs):
        for path, key, value in _walk(node):
            name = str(key)
            leaf = name.split(".")[-1]
            if (leaf in {"hsl_signal_mode", "max_realized_loss_pct", "lookback_ema_span_minutes", "auto_gs", "liquidation_threshold", "n_positions", "total_exposure_entry_gate_enabled", "risk_twel_entry_gate_enabled", "forager_volume_drop_pct", "filter_volume_drop_pct"}
                    or (leaf == "ema_span_minutes" and "hsl" in path)
                    or (leaf == "volume_drop_pct" and "forager" in path)
                    or (leaf == "threshold" and "unstuck" in path)):
                raise ValueError(f"{path} is retired; final canonical {V9_SCHEMA_VERSION} requires the v9 layout. Convert versioned v8 with {MIGRATION_COMMAND}")
            if (
                "pnls_max_lookback_days" in name
                or "realized_loss_gate_lookback_minutes" in name
                or "restart_after_red_policy" in name
                or "hsl_engine" in name
                or _retired_control(path, key)
            ):
                raise ValueError(f"{path} is removed in v9; use {MIGRATION_COMMAND}")
            if isinstance(value, str) and "fixed_params" in path and (
                "pnls_max_lookback_days" in value or "restart_after_red_policy" in value
                or "realized_loss_gate_lookback_minutes" in value
            ):
                raise ValueError(f"{path} targets a removed v9 field; use {MIGRATION_COMMAND}")
            if name.split(".")[-1] == "hsl_signal_mode" and value == "unified":
                raise ValueError(f"{path}=unified is removed in v9; use {MIGRATION_COMMAND}")


def _validate_legacy_gate_window(value, path, *, days=False):
    """Validate retired input without interpreting it as exponential memory."""
    if days and isinstance(value, str) and value.strip().lower() == "all":
        return
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite legacy duration")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} must be a finite legacy duration") from exc
    milliseconds = number * (86_400_000.0 if days else 60_000.0)
    if not math.isfinite(number) or (number < 0.0 and (days or number != -1.0)):
        all_history = "'all'" if days else "-1"
        raise ValueError(f"{path} must be >= 0, or {all_history} for legacy all history")
    if not math.isfinite(milliseconds) or round(milliseconds) > (1 << 63) - 1:
        raise ValueError(f"{path} must fit in signed 64-bit milliseconds")


def _realized_gate_migration_contexts(config):
    """Inspect only global gate consumers, including dormant authored scenarios.

    Coin-file global fields are not an allowed coin patch. Their migration uses
    private caller state to skip this audit; the complete authored configuration
    is always audited separately before canonical validation.
    """
    from config.scenario_overrides import normalize_scenario_overrides

    live = config.get("live", {})
    root = (live.get("max_realized_loss_pct", 1.0), live.get("lookback_ema_span_minutes"))

    def overlay(base, overrides):
        patch = normalize_scenario_overrides(overrides)
        return (patch.get("live.max_realized_loss_pct", base[0]),
                patch.get("live.lookback_ema_span_minutes", base[1]))

    optimized = overlay(root, config.get("optimize", {}).get("fixed_runtime_overrides"))
    contexts = [("root", root), ("optimize.fixed_runtime_overrides", optimized)]
    for prefix, container in (("backtest", config.get("backtest", {})),
                              ("backtest.suite", config.get("backtest", {}).get("suite", {}))):
        if not isinstance(container, dict):
            continue
        for index, scenario in enumerate(container.get("scenarios", [])):
            for suffix, base in (("", root), (" + fixed_runtime_overrides", optimized)):
                contexts.append((f"{prefix}.scenarios[{index}]{suffix}",
                                 overlay(base, scenario.get("overrides"))))
    active = []
    for path, (raw_fraction, span) in contexts:
        if isinstance(raw_fraction, bool):
            raise ValueError(f"{path}: live.max_realized_loss_pct must be a finite number")
        try:
            fraction = float(raw_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{path}: live.max_realized_loss_pct must be finite") from exc
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError(f"{path}: live.max_realized_loss_pct must be finite and >= 0")
        if fraction < 1.0:
            if span is None:
                raise ValueError(
                    f"{path}: active realized-loss gate requires an explicit "
                    "--lookback-ema-span-minutes choice or authored shared span"
                )
            span = normalize_lookback_ema_span_minutes(span)
            active.append(dict(context=path, max_realized_loss_pct=fraction,
                               lookback_ema_span_minutes=span))
    return active


def _require_gate_ack(config, acknowledge):
    contexts = _realized_gate_migration_contexts(config)
    if contexts and not acknowledge:
        raise ValueError(
            f"{contexts[0]['context']}: active max_realized_loss_pct changes from rolling "
            "history to shared decaying realized memory; requires "
            "--acknowledge-realized-loss-gate-memory-change and fresh backtests"
        )
    return contexts


def migrate_v8_to_v9(
    source,
    *,
    memory_span_minutes=None,
    memory_span_choices=None,
    acknowledge_never_removal=False,
    acknowledge_realized_loss_gate_memory_change=False,
    acknowledge_search_backend_change=False,
    metric_replacements=None,
    acknowledge_metric_formula_change=False,
    _null_policy_enablement=None,
    _coin_patch=False,
    best_effort=False,
    _guess_log_prefix="",
):
    """Transform authored policy without importing simulation or exchange runtimes.

    The caller materializes referenced overrides first and canonically validates
    the result afterwards. Choice keys are exact authored paths from diagnostics.
    Risk-window bounds require explicit editing instead of guessed conversion.
    """
    if not isinstance(source, dict):
        raise TypeError("configuration must be a mapping")
    from config.authored import _METADATA_KEYS

    payload = source.get("config") if "bot" not in source else source
    if not isinstance(payload, dict) or not isinstance(payload.get("bot"), dict):
        raise ValueError("expected a v8 or v9 bot configuration")
    result = deepcopy(payload)
    version = result.get("config_version")
    match = _VERSION.fullmatch(version.strip()) if isinstance(version, str) else None
    if match is None or tuple(map(int, match.groups()))[0] not in {8, 9}:
        raise ValueError("migrate-config-v9 requires versioned v8 or final canonical v9")
    if version == V9_SCHEMA_VERSION:
        require_v9_runtime_config(result)
        from config.authored import validate_authored_config
        from config.schema import get_template_config

        validate_authored_config(result, get_template_config())
        if memory_span_minutes is not None and normalize_lookback_ema_span_minutes(
            memory_span_minutes
        ) != result.get("bot", {}).get("unstuck", {}).get("memory_ema_span_minutes"):
            raise ValueError("already-v9 configuration has a different memory span; edit it explicitly")
        if memory_span_choices:
            raise ValueError("memory-span choices do not apply to an already-v9 configuration")
        if metric_replacements:
            raise ValueError("metric replacements apply only to versioned v8 migration")
        return MigrationResult(result, [], ())
    is_v8 = int(match[1]) == 8
    if not is_v8:
        raise ValueError("migrate-config-v9 supports versioned v8 and final canonical v9 only")
    from .final_v9 import reject_v8_alias_conflicts

    reject_v8_alias_conflicts(result)
    inferred_changes = []
    if best_effort and not _coin_patch:
        from .best_effort_v9 import record_envelope_metadata, remove_result_metadata

        wrapped = payload is not source
        if wrapped:
            record_envelope_metadata(source, inferred_changes)
        remove_result_metadata(result, inferred_changes, wrapped=wrapped,
                               path_prefix="config." if wrapped else "")
    if not _coin_patch:
        from .final_v9 import replace_retired_metrics

        metric_changes = inferred_changes
        replace_retired_metrics(result, replacements=metric_replacements,
                                acknowledge=acknowledge_metric_formula_change, changes=metric_changes,
                                best_effort=best_effort)
    else:
        metric_changes = []
    if best_effort and memory_span_minutes is None and not _coin_patch:
        from .best_effort_v9 import infer_memory_span

        live = result.get("live", {})
        memory_span_minutes = live.get("lookback_ema_span_minutes")
        if memory_span_minutes is None:
            memory_span_minutes = infer_memory_span(
                live.get("pnls_max_lookback_days"), "live.pnls_max_lookback_days", metric_changes,
                missing="pnls_max_lookback_days" not in live,
            )
    if is_v8 and memory_span_minutes is None and not _coin_patch:
        raise ValueError(
            "v8 risk memory is not equivalent to an EMA span; explicitly choose "
            "--lookback-ema-span-minutes and re-tune HSL/unstuck"
        )
    authored_span = (None if _coin_patch else
                     result.get("live", {}).get("lookback_ema_span_minutes"))
    root_span = (normalize_lookback_ema_span_minutes(memory_span_minutes)
                 if memory_span_minutes is not None else
                 normalize_lookback_ema_span_minutes(authored_span)
                 if authored_span is not None else None)
    if authored_span is not None and root_span != normalize_lookback_ema_span_minutes(authored_span):
        raise ValueError("authored live.lookback_ema_span_minutes conflicts with migration choice")
    choices = dict(memory_span_choices or {})
    independent_choices = {
        path: normalize_lookback_ema_span_minutes(value, field=path)
        for path, value in choices.items()
        if path.endswith(".memory_ema_span_minutes") and path.startswith("bot.")
    }
    choices = {path: value for path, value in choices.items() if path not in independent_choices}
    used_choices = set()
    changes = metric_changes

    # Read enablement before the visitor removes authored leaves. Keep duplicate
    # flat/grouped spellings visible: conflicting facts are ambiguous, not a
    # reason to choose whichever dictionary key happened to be visited first.
    enablement = _hsl_enablement_facts(result)
    # Referenced partial files need root -> file -> inline provenance before
    # filtering/canonicalization merges aliases. Only the tool supplies this
    # private exact-path context; no inherited leaf is added to the output.
    enablement.update(_null_policy_enablement or {})

    def disabled_null_policy(path):
        parts = _semantic_hsl_path(path)
        if parts[-2:] != ("hsl", "restart_after_red_policy") or "bounds" in parts:
            return False
        relative = (*parts[:-1], "enabled")
        context = ()
        if relative[:2] == ("backtest", "scenarios") and relative[3:4] == ("overrides",):
            context, relative = relative[:4], relative[4:]
        elif relative[:3] == ("backtest", "suite", "scenarios") and relative[4:5] == ("overrides",):
            context, relative = relative[:5], relative[5:]
        elif relative[:2] == ("optimize", "fixed_runtime_overrides"):
            context, relative = relative[:2], relative[2:]
        candidates = [(*context, *relative)]
        if relative[:1] == ("coin_overrides",) and len(relative) > 2:
            # A coin policy inherits its side's global policy. A scenario coin
            # map replaces the authored coin map, so do not revive an old coin
            # patch when resolving a new scenario-local coin map.
            side_policy = relative[2:]
            if not context:
                candidates.append(relative)
            candidates.extend([(*context, *side_policy), side_policy])
        elif context:
            candidates.append(relative)
        for candidate in candidates:
            values = enablement.get(candidate)
            if values is not None:
                return bool(values) and all(value is False for value in values)
        return False

    def record(path, old, new, reason):
        changes.append(dict(path=path, old=deepcopy(old), new=deepcopy(new), reason=reason))

    def visit(node, path=""):
        if isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        for key in list(node):
            if not path and key in _METADATA_KEYS:
                del node[key]
                continue
            value = node[key]
            current = f"{path}.{key}" if path else str(key)
            if best_effort and str(key).split(".")[-1] in {
                "we_excess_allowance_mode", "risk_we_excess_allowance_mode",
            }:
                from .best_effort_v9 import record_guess

                if not isinstance(value, str) or value.strip().lower() != "bounded":
                    raise ValueError(f"{current}: only retired bounded allowance mode can be removed automatically")
                del node[key]
                record_guess(changes, current, value, None,
                             "remove retired bounded-mode selector; v9 always uses bounded excess allowance",
                             log_prefix=_guess_log_prefix)
                continue
            if _retired_control(current, key):
                del node[key]
                record(current, value, None, "retired HSL control; v9 uses current RED and finite cooldown")
                continue
            if key == "fixed_params" and isinstance(value, list):
                if any(isinstance(item, str) and (
                    "pnls_max_lookback_days" in item or "restart_after_red_policy" in item
                    or "realized_loss_gate_lookback_minutes" in item
                ) for item in value):
                    raise ValueError(f"{current} targets removed v8 fields; edit selectors explicitly")
            if "restart_after_red_policy" in str(key):
                if value is None and disabled_null_policy(current):
                    del node[key]
                    record(current, value, None, "removed null restart field from a disabled v8 HSL policy")
                    continue
                if not isinstance(value, str) or value.strip().lower() not in {"always", "never"}:
                    raise ValueError(f"{current} has no v9 policy equivalent; remove it explicitly")
                if value.strip().lower() == "never" and not acknowledge_never_removal and not best_effort:
                    raise ValueError(
                        f"{current}=never requires --acknowledge-never-removal; "
                        "v9 resumes after finite cooldown"
                    )
                if value.strip().lower() == "never" and best_effort and not acknowledge_never_removal:
                    from .best_effort_v9 import record_guess

                    record_guess(changes, current, value, None,
                                 "remove permanent stop; v9 resumes after finite cooldown and admission gates",
                                 log_prefix=_guess_log_prefix)
                else:
                    record(current, value, None, "removed restart policy; only finite cooldown remains")
                del node[key]
                continue
            if "hsl_engine" in str(key):
                del node[key]
                record(current, value, None, "removed legacy engine selector")
                continue
            if "pnls_max_lookback_days" in str(key):
                if ".bounds" in current or ".fixed_params" in current or isinstance(value, (list, dict)):
                    raise ValueError(
                        f"{current} requires explicit new live.lookback_ema_span_minutes bounds; "
                        "days-to-minutes conversion is not equivalent risk policy"
                    )
                if _coin_patch:
                    _validate_legacy_gate_window(value, current, days=True)
                    del node[key]
                    record(current, value, None, "removed global memory field from a coin-only document")
                    continue
                if current == "live.pnls_max_lookback_days":
                    span = root_span
                elif current in choices:
                    span = normalize_lookback_ema_span_minutes(choices[current])
                    used_choices.add(current)
                elif best_effort:
                    from .best_effort_v9 import infer_memory_span

                    span = infer_memory_span(value, current, changes)
                else:
                    raise ValueError(f"{current} requires --memory-span '{current}=MINUTES'")
                _validate_legacy_gate_window(value, current, days=True)
                new_key = str(key).replace("pnls_max_lookback_days", "lookback_ema_span_minutes")
                if new_key.startswith("live_"):
                    new_key = "live." + new_key.removeprefix("live_")
                if new_key in node:
                    raise ValueError(f"{current} conflicts with an authored v9 risk field")
                node[new_key] = span
                del node[key]
                record(current, value, {new_key: span},
                       "explicit shared decaying-memory choice; no rolling-window equivalence")
                continue
            if "realized_loss_gate_lookback_minutes" in str(key):
                if ".bounds" in current or isinstance(value, (list, dict)):
                    raise ValueError(f"{current} is retired; choose shared-span bounds explicitly")
                _validate_legacy_gate_window(value, current)
                del node[key]
                record(current, value, None,
                       "retired rolling gate window; gate shares the existing exponential span")
                continue
            if (str(key).split(".")[-1] == "hsl_signal_mode" and
                    isinstance(value, str) and value.strip().lower() == "unified"):
                node[key] = "portfolio"
                record(current, value, "portfolio", "canonical scope rename")
            else:
                visit(value, current)

    visit(result)
    unused = set(choices) - used_choices
    if unused:
        raise ValueError(f"unused memory-span choices: {', '.join(sorted(unused))}")
    live = result.setdefault("live", {})
    if (not _coin_patch and root_span is not None and "lookback_ema_span_minutes" in live
            and live["lookback_ema_span_minutes"] != root_span):
        raise ValueError("authored live.lookback_ema_span_minutes conflicts with migration choice")
    if not _coin_patch and root_span is not None and "lookback_ema_span_minutes" not in live:
        live["lookback_ema_span_minutes"] = root_span
        record("live.lookback_ema_span_minutes", None, root_span, "explicit v9 memory choice")
    if not _coin_patch:
        active_contexts = _require_gate_ack(result, acknowledge_realized_loss_gate_memory_change or best_effort)
        if active_contexts and best_effort and not acknowledge_realized_loss_gate_memory_change:
            from .best_effort_v9 import record_guess

            for context in active_contexts:
                record_guess(changes, context["context"], None, context,
                             "adopt independent decaying realized-loss memory instead of rolling history")
        if active_contexts:
            record("live.max_realized_loss_pct", None, active_contexts,
                   "acknowledged shared realized-loss memory in every effective authored context")
    result["config_version"] = V9_SCHEMA_VERSION
    record("config_version", version, V9_SCHEMA_VERSION, "explicit v9 schema migration")
    from .final_v9 import finalize_v8_layout

    result = finalize_v8_layout(result, span=None if _coin_patch else root_span, changes=changes,
                                acknowledge_search_backend_change=acknowledge_search_backend_change,
                                sparse=_coin_patch, best_effort=best_effort)
    from config.authored import validate_authored_config
    from config.schema import get_template_config

    for path, value in independent_choices.items():
        parts = path.split(".")
        target = result
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"{path} is not an active canonical memory policy")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"{path} is not an active canonical memory policy")
        previous = target[parts[-1]]
        target[parts[-1]] = value
        record(path, previous, value, "explicit independent memory-span choice")
    if not _coin_patch:
        validate_authored_config(result, get_template_config())
    require_v9_runtime_config(result)
    return MigrationResult(
        result,
        changes,
        ("HSL ratios, drawdown memory, cooldown and unstuck replenishment changed. "
         "HSL, unstuck and the loss gate now have independent spans and shared arithmetic. "
         "Re-tune spans/allowances/bounds and freshly evaluate every seed; old fitness/checkpoints are invalid.",),
    )
