"""Explicit v8-to-v9 policy migration; ordinary runtime loading is v9-only."""

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
            if (
                "pnls_max_lookback_days" in name
                or "restart_after_red_policy" in name
                or "hsl_engine" in name
                or _retired_control(path, key)
            ):
                raise ValueError(f"{path} is removed in v9; use {MIGRATION_COMMAND}")
            if isinstance(value, str) and "fixed_params" in path and (
                "pnls_max_lookback_days" in value or "restart_after_red_policy" in value
            ):
                raise ValueError(f"{path} targets a removed v9 field; use {MIGRATION_COMMAND}")
            if name.split(".")[-1] == "hsl_signal_mode" and value == "unified":
                raise ValueError(f"{path}=unified is removed in v9; use {MIGRATION_COMMAND}")


def _gate_minutes(days, path):
    if isinstance(days, str) and days.strip().lower() == "all":
        return -1.0
    if isinstance(days, bool):
        raise ValueError(f"{path} must be finite days >= 0 or 'all'")
    try:
        number = float(days)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} must be finite days >= 0 or 'all'") from exc
    minutes = number * 1440.0
    if not math.isfinite(minutes) or number < 0.0:
        raise ValueError(f"{path} must be finite days >= 0 or 'all'")
    return minutes


def migrate_v8_to_v9(
    source,
    *,
    memory_span_minutes=None,
    memory_span_choices=None,
    acknowledge_never_removal=False,
    _null_policy_enablement=None,
):
    """Transform authored policy without importing simulation or exchange runtimes.

    The caller materializes referenced overrides first and canonically validates
    the result afterwards. Choice keys are exact authored paths from diagnostics.
    Risk-window bounds require explicit editing instead of guessed conversion.
    """
    if not isinstance(source, dict):
        raise TypeError("configuration must be a mapping")
    payload = source.get("config") if "bot" not in source else source
    if not isinstance(payload, dict) or not isinstance(payload.get("bot"), dict):
        raise ValueError("expected a v8 or v9 bot configuration")
    result = deepcopy(payload)
    version = result.get("config_version")
    match = _VERSION.fullmatch(version.strip()) if isinstance(version, str) else None
    if match is None or tuple(map(int, match.groups()))[0] not in {8, 9}:
        raise ValueError("migrate-config-v9 requires a versioned v8 configuration or v9.0.0")
    if int(match[1]) == 9:
        require_v9_runtime_config(result)
        if memory_span_minutes is not None and normalize_lookback_ema_span_minutes(
            memory_span_minutes
        ) != result.get("live", {}).get("lookback_ema_span_minutes"):
            raise ValueError("already-v9 configuration has a different memory span; edit it explicitly")
        if memory_span_choices:
            raise ValueError("memory-span choices do not apply to an already-v9 configuration")
        return MigrationResult(result, [], ())
    if memory_span_minutes is None:
        raise ValueError(
            "v8 risk memory is not equivalent to an EMA span; explicitly choose "
            "--lookback-ema-span-minutes and re-tune HSL/unstuck"
        )
    root_span = normalize_lookback_ema_span_minutes(memory_span_minutes)
    choices = dict(memory_span_choices or {})
    used_choices = set()
    changes = []

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
            if str(key).startswith("_"):
                del node[key]
                continue
            value = node[key]
            current = f"{path}.{key}" if path else str(key)
            if _retired_control(current, key):
                del node[key]
                record(current, value, None, "retired HSL control; v9 uses current RED and finite cooldown")
                continue
            if key == "fixed_params" and isinstance(value, list):
                if any(isinstance(item, str) and (
                    "pnls_max_lookback_days" in item or "restart_after_red_policy" in item
                ) for item in value):
                    raise ValueError(f"{current} targets removed v8 fields; edit selectors explicitly")
            if "restart_after_red_policy" in str(key):
                if value is None and disabled_null_policy(current):
                    del node[key]
                    record(current, value, None, "removed null restart field from a disabled v8 HSL policy")
                    continue
                if not isinstance(value, str) or value.strip().lower() not in {"always", "never"}:
                    raise ValueError(f"{current} has no v9 policy equivalent; remove it explicitly")
                if value.strip().lower() == "never" and not acknowledge_never_removal:
                    raise ValueError(
                        f"{current}=never requires --acknowledge-never-removal; "
                        "v9 resumes after finite cooldown"
                    )
                del node[key]
                record(current, value, None, "removed restart policy; only finite cooldown remains")
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
                if current == "live.pnls_max_lookback_days":
                    span = root_span
                elif current in choices:
                    span = normalize_lookback_ema_span_minutes(choices[current])
                    used_choices.add(current)
                else:
                    raise ValueError(f"{current} requires --memory-span '{current}=MINUTES'")
                new_key = str(key).replace("pnls_max_lookback_days", "lookback_ema_span_minutes")
                gate_key = str(key).replace("pnls_max_lookback_days", "realized_loss_gate_lookback_minutes")
                if new_key.startswith("live_"):
                    new_key = "live." + new_key.removeprefix("live_")
                    gate_key = "live." + gate_key.removeprefix("live_")
                if new_key in node or gate_key in node:
                    raise ValueError(f"{current} conflicts with an authored v9 risk field")
                node[new_key] = span
                node[gate_key] = _gate_minutes(value, current)
                del node[key]
                record(current, value, {new_key: span, gate_key: node[gate_key]},
                       "explicit decaying-memory choice; independent rolling gate keeps its duration")
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
    if "lookback_ema_span_minutes" in live and live["lookback_ema_span_minutes"] != root_span:
        raise ValueError("authored live.lookback_ema_span_minutes conflicts with migration choice")
    if "lookback_ema_span_minutes" not in live:
        live["lookback_ema_span_minutes"] = root_span
        record("live.lookback_ema_span_minutes", None, root_span, "explicit v9 memory choice")
    result["config_version"] = V9_SCHEMA_VERSION
    record("config_version", version, V9_SCHEMA_VERSION, "explicit v9 schema migration")
    require_v9_runtime_config(result)
    return MigrationResult(
        result,
        changes,
        ("HSL ratios, drawdown memory, cooldown and unstuck replenishment changed. "
         "Re-tune spans/allowances/bounds and freshly evaluate every seed; v8 fitness/checkpoints are invalid.",),
    )
