"""Explicit offline v8-to-v9 migration, with a diff and atomic private output."""

from argparse import ArgumentParser
from copy import deepcopy
import difflib
import json
import os
from pathlib import Path
import stat
import tempfile

from config.migrations.v9 import MigrationResult, _hsl_enablement_facts, migrate_v8_to_v9
from config.risk_memory import normalize_lookback_ema_span_minutes
from json_utils import json_dumps_streamlined


def _set_patch(config, overrides, *, authored=False, scenario=False):
    from config.param_paths import require_existing_config_path, resolve_dotted_config_path
    from config.scenario_overrides import normalize_scenario_overrides

    seen = set()
    for selector, value in normalize_scenario_overrides(overrides).items():
        if scenario and (selector == "optimize" or selector.startswith("optimize.")):
            raise ValueError("scenario/runtime optimizer controls must be top-level optimize settings")
        path = (resolve_dotted_config_path(config, selector) if authored else
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


def _materialize_files(source, *, base_config_path, span, acknowledge, changes,
                       _null_policy_enablement=None, _root_enablement=None):
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
            _root_enablement=_root_enablement,
        )
    if old_flags:
        changes.append(dict(
            path="live.coin_flags", old=old_flags, new={},
            reason="materialized legacy flags; canonical coin_overrides take precedence",
        ))
    mode = result.get("live", {}).get("hsl_signal_mode", "coin")
    mode = "portfolio" if mode == "unified" else mode
    for coin, inline in result.get("coin_overrides", {}).items():
        if not isinstance(inline, dict):
            raise ValueError(f"coin_overrides.{coin} must be a mapping")
        if not inline.get("override_config_path"):
            continue
        context = deepcopy(result)
        context.setdefault("live", {})["base_config_path"] = base_config_path
        loaded = _unwrap_override_document(
            load_override_config(context, coin, config_loader=load_raw_config),
            source=f"coin_overrides.{coin}.override_config_path",
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
        )
        # Audit inline authority in this same effective context too. Audit-only
        # scenario/optimizer calls discard the materialized result, so deferring
        # inline null validation to the final root pass would miss those modes.
        # Keep raw inline leaves for that final pass's authored change paths.
        migrate_v8_to_v9(
            {"config_version": "v8.6.0", "bot": {}, **inline}, memory_span_minutes=span,
            acknowledge_never_removal=acknowledge,
            _null_policy_enablement=enablement,
        )
        migrated = file_result.config
        changes.extend({**item, "path": f"coin_overrides.{coin}.override_config_path::{item['path']}"}
                       for item in file_result.changes)
        patch = _extract_allowed_patch(
            migrated, coin=coin, source=f"coin_overrides.{coin}.override_config_path",
            strict=False, strategy_kind=result.get("live", {}).get("strategy_kind"),
            hsl_signal_mode=mode,
        )
        inline = deepcopy(inline)
        inline.pop("override_config_path")
        # Resolve structural aliases independently on each authored patch before
        # merging. Otherwise a canonicalized file value can beat the higher-
        # precedence inline legacy value during later hydration. Keep raw HSL
        # authorization leaves for the explicit v9 migration pass below.
        from config.migrations.entry_cooldown import migrate_entry_cooldown_tree
        from config.migrations.entry_ema import migrate_entry_ema_tree

        migrate_entry_cooldown_tree(inline, path=f"coin_overrides.{coin}")
        migrate_entry_ema_tree(inline, path=f"coin_overrides.{coin}")
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
        "live": {"strategy_kind": config.get("live", {}).get("strategy_kind")},
        "bot": {side: {"hsl": {"enabled": facts.get(("bot", side, "hsl", "enabled"), ())}}
                for side in ("long", "short")},
    }
    projection = optimizer_overrides(enable_overrides, projection, None)
    resolved = dict(facts)
    for side in ("long", "short"):
        resolved[("bot", side, "hsl", "enabled")] = projection["bot"][side]["hsl"]["enabled"]
    return resolved


def migrate(source, *, memory_span_minutes=None, memory_span_choices=None,
            acknowledge_never_removal=False, base_config_path=""):
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
    if payload.get("config_version") == "v9.0.0":
        transformed = migrate_v8_to_v9(
            payload, memory_span_minutes=memory_span_minutes,
            memory_span_choices=memory_span_choices,
        )
    else:
        if memory_span_minutes is None:
            # The core owns the actionable version/span error.
            return migrate_v8_to_v9(payload, memory_span_minutes=None)
        span = normalize_lookback_ema_span_minutes(memory_span_minutes)
        file_changes = []
        null_policy_enablement = {}
        authored = _materialize_files(
            payload, base_config_path=base_config_path, span=span,
            acknowledge=acknowledge_never_removal, changes=file_changes,
            _null_policy_enablement=null_policy_enablement,
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
                _root_enablement=optimized_root_enablement,
            )
            if materialize_fixed_coins:
                null_policy_enablement.update({
                    ("optimize", "fixed_runtime_overrides", *path): values
                    for path, values in fixed_enablement.items()
                })
                fixed = {key: value for key, value in fixed.items()
                         if not any(key == root or key.startswith(root + ".") for root in roots)}
                fixed["coin_overrides"] = effective_fixed["coin_overrides"]
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
                    _root_enablement=_hsl_enablement_facts(root_policy, overrides),
                )
                if audit_optimized:
                    optimized_scenario = deepcopy(optimized_payload)
                    _set_patch(optimized_scenario, overrides, authored=True, scenario=True)
                    scenario_optimized_enablement = dict(optimized_root_enablement)
                    scenario_optimized_enablement.update(_hsl_enablement_facts(overrides))
                    _materialize_files(
                        optimized_scenario, base_config_path=base_config_path, span=span,
                        acknowledge=acknowledge_never_removal, changes=[],
                        _root_enablement=scenario_optimized_enablement,
                    )
                if materialize_coin_patch:
                    prefix = (*container_path, "scenarios", str(index), "overrides")
                    null_policy_enablement.update({
                        (*prefix, *path): values
                        for path, values in scenario_enablement.items()
                    })
                    overrides = {
                        key: value for key, value in overrides.items()
                        if not any(key == root or key.startswith(root + ".") for root in roots)
                    }
                    overrides["coin_overrides"] = effective["coin_overrides"]
                    scenario["overrides"] = overrides
        transformed = migrate_v8_to_v9(
            authored, memory_span_minutes=span, memory_span_choices=memory_span_choices,
            acknowledge_never_removal=acknowledge_never_removal,
            _null_policy_enablement=null_policy_enablement,
        )
        transformed = MigrationResult(transformed.config, file_changes + transformed.changes,
                                      transformed.warnings)
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
    return MigrationResult(strip_config_metadata(prepared), transformed.changes, transformed.warnings)


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
        backup = (args.backup or target.with_name(target.stem + ".v8-backup" + target.suffix)
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
        original_bytes = args.input_config.read_bytes()
        source = load_raw_config(str(args.input_config))
        if args.input_config.resolve() != source_path or args.input_config.read_bytes() != original_bytes:
            raise ValueError("input changed while it was being read; retry migration")
        result = migrate(
            source, memory_span_minutes=args.lookback_ema_span_minutes,
            memory_span_choices=choices, acknowledge_never_removal=args.acknowledge_never_removal,
            base_config_path=str(args.input_config.resolve()),
        )
        rendered = json_dumps_streamlined(result.config, allow_nan=False) + "\n"
        report = dict(source_version=source.get("config_version"), target_version="v9.0.0",
                      changes=result.changes, warnings=list(result.warnings))
        report_rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.diff:
            before = json_dumps_streamlined(source, allow_nan=False) + "\n"
            print("".join(difflib.unified_diff(before.splitlines(True), rendered.splitlines(True),
                                            fromfile="v8 input", tofile="v9 output")), end="")
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
