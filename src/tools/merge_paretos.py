from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config.scoring import ObjectiveSpec
from pareto_explorer import (
    ParetoCandidate,
    _normalize_objective_matrix,
    _resolve_candidate_metric_value,
    load_candidates,
    select_candidate,
)

SIDES = ("long", "short")
DEFAULT_MAX_OUTPUTS = 500


@dataclass(frozen=True)
class LoadedFront:
    index: int
    input_path: Path
    pareto_dir: Path
    candidates: list[ParetoCandidate]
    scoring_specs: list[ObjectiveSpec]
    quality_by_path: dict[Path, float]


@dataclass(frozen=True)
class SideComponent:
    side: str
    side_config: dict[str, Any]
    base_bot: dict[str, Any]
    front_index: int
    pareto_dir: Path
    candidate_path: Path
    objectives: dict[str, float]
    quality: float
    config_hash: str
    selectors: tuple[str, ...] = ()


@dataclass
class MergeStats:
    core_pair_attempts: int = 0
    core_pair_added: int = 0
    fill_pair_attempts: int = 0
    fill_pair_added: int = 0
    duplicate_pairs: int = 0
    capped: bool = False


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _entry_config(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapped = entry.get("config")
    if isinstance(wrapped, Mapping):
        return wrapped
    return entry


def _entry_bot(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    config = _entry_config(entry)
    bot = config.get("bot")
    if not isinstance(bot, Mapping):
        raise ValueError("Pareto candidate is missing config.bot/bot")
    return bot


def _entry_optimize_bounds(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    config = _entry_config(entry)
    optimize = config.get("optimize")
    if not isinstance(optimize, Mapping):
        return {}
    bounds = optimize.get("bounds")
    if not isinstance(bounds, Mapping):
        return {}
    return bounds


def _side_is_enabled(side_config: Mapping[str, Any]) -> bool:
    try:
        n_positions_raw = side_config.get("n_positions")
        exposure_limit_raw = side_config.get("total_wallet_exposure_limit")
        if n_positions_raw is None or exposure_limit_raw is None:
            return False
        n_positions = float(n_positions_raw)
        exposure_limit = float(exposure_limit_raw)
    except (TypeError, ValueError):
        return False
    return n_positions > 0.0 and exposure_limit > 0.0


def _candidate_quality(front: LoadedFront, candidate: ParetoCandidate) -> float:
    value = front.quality_by_path.get(candidate.path)
    return 0.0 if value is None else float(value)


def _load_fronts(input_paths: Sequence[Path]) -> list[LoadedFront]:
    fronts: list[LoadedFront] = []
    for idx, input_path in enumerate(input_paths):
        pareto_dir, candidates, scoring_specs = load_candidates(input_path)
        quality_by_path: dict[Path, float] = {}
        if scoring_specs:
            utilities, _lows, _highs = _normalize_objective_matrix(candidates, scoring_specs)
            for candidate, utility_row in zip(candidates, utilities):
                quality_by_path[candidate.path] = float(utility_row.mean())
        else:
            quality_by_path = {candidate.path: 0.0 for candidate in candidates}
        fronts.append(
            LoadedFront(
                index=idx,
                input_path=input_path,
                pareto_dir=pareto_dir,
                candidates=list(candidates),
                scoring_specs=list(scoring_specs),
                quality_by_path=quality_by_path,
            )
        )
    return fronts


def _component_from_candidate(
    front: LoadedFront,
    candidate: ParetoCandidate,
    side: str,
    *,
    selector: str | None = None,
) -> SideComponent | None:
    bot = _entry_bot(candidate.entry)
    raw_side_config = bot.get(side)
    if not isinstance(raw_side_config, Mapping):
        return None
    if not _side_is_enabled(raw_side_config):
        return None
    side_config = copy.deepcopy(dict(raw_side_config))
    config_hash = _json_fingerprint(side_config)
    selectors = (selector,) if selector else ()
    return SideComponent(
        side=side,
        side_config=side_config,
        base_bot=copy.deepcopy(dict(bot)),
        front_index=front.index,
        pareto_dir=front.pareto_dir,
        candidate_path=candidate.path,
        objectives=dict(candidate.objectives),
        quality=_candidate_quality(front, candidate),
        config_hash=config_hash,
        selectors=selectors,
    )


def _merge_selectors(existing: tuple[str, ...], incoming: tuple[str, ...]) -> tuple[str, ...]:
    if not incoming:
        return existing
    merged = list(existing)
    for selector in incoming:
        if selector not in merged:
            merged.append(selector)
    return tuple(merged)


def _upsert_component(
    registry: dict[str, SideComponent],
    component: SideComponent | None,
) -> None:
    if component is None:
        return
    existing = registry.get(component.config_hash)
    if existing is None:
        registry[component.config_hash] = component
        return
    selectors = _merge_selectors(existing.selectors, component.selectors)
    if selectors != existing.selectors:
        registry[component.config_hash] = SideComponent(
            side=existing.side,
            side_config=existing.side_config,
            base_bot=existing.base_bot,
            front_index=existing.front_index,
            pareto_dir=existing.pareto_dir,
            candidate_path=existing.candidate_path,
            objectives=existing.objectives,
            quality=existing.quality,
            config_hash=existing.config_hash,
            selectors=selectors,
        )


def _best_candidate_for_metric(
    candidates: Sequence[ParetoCandidate],
    spec: ObjectiveSpec,
) -> ParetoCandidate:
    values: list[tuple[float, ParetoCandidate]] = []
    for candidate in candidates:
        value = _resolve_candidate_metric_value(candidate, spec.metric)
        if value is None:
            continue
        values.append((float(value), candidate))
    if not values:
        raise ValueError(f"No candidate has objective metric {spec.metric!r}")
    if spec.goal == "max":
        return max(values, key=lambda item: item[0])[1]
    return min(values, key=lambda item: item[0])[1]


def _collect_core_components(fronts: Sequence[LoadedFront]) -> dict[str, dict[str, SideComponent]]:
    core: dict[str, dict[str, SideComponent]] = {"long": {}, "short": {}}
    for front in fronts:
        if not front.scoring_specs:
            continue
        selected: list[tuple[str, ParetoCandidate]] = []
        for spec in front.scoring_specs:
            selected.append(
                (
                    f"front{front.index}:best:{spec.metric}",
                    _best_candidate_for_metric(front.candidates, spec),
                )
            )
        for method in ("ideal", "knee"):
            result = select_candidate(front.candidates, front.scoring_specs, method=method)
            selected.append((f"front{front.index}:{method}", result.candidate))
        for selector, candidate in selected:
            for side in SIDES:
                _upsert_component(
                    core[side],
                    _component_from_candidate(front, candidate, side, selector=selector),
                )
    return core


def _collect_all_components(fronts: Sequence[LoadedFront]) -> dict[str, dict[str, SideComponent]]:
    components: dict[str, dict[str, SideComponent]] = {"long": {}, "short": {}}
    for front in fronts:
        for candidate in front.candidates:
            for side in SIDES:
                _upsert_component(
                    components[side],
                    _component_from_candidate(front, candidate, side),
                )
    return components


def _round_robin_by_front(components: Iterable[SideComponent]) -> list[SideComponent]:
    buckets: dict[int, list[SideComponent]] = {}
    for component in sorted(
        components,
        key=lambda item: (
            -item.quality,
            item.front_index,
            str(item.candidate_path),
            item.config_hash,
        ),
    ):
        buckets.setdefault(component.front_index, []).append(component)

    ordered: list[SideComponent] = []
    while buckets:
        for front_index in sorted(list(buckets)):
            bucket = buckets[front_index]
            ordered.append(bucket.pop(0))
            if not bucket:
                del buckets[front_index]
    return ordered


def _clean_number(value: float) -> int | float:
    if math.isfinite(value) and value.is_integer():
        return int(value)
    return float(value)


def _parse_bound(raw: Any) -> tuple[float, float, float | None] | None:
    if isinstance(raw, (list, tuple)):
        if len(raw) < 2:
            return None
        low_raw, high_raw = raw[0], raw[1]
        step_raw = raw[2] if len(raw) >= 3 else None
    else:
        low_raw = high_raw = raw
        step_raw = None
    try:
        low = float(low_raw)
        high = float(high_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    step: float | None = None
    if step_raw is not None:
        try:
            parsed_step = float(step_raw)
        except (TypeError, ValueError):
            parsed_step = None
        if parsed_step is not None and math.isfinite(parsed_step) and parsed_step > 0.0:
            step = parsed_step
    return low, high, step


def _merge_bounds(fronts: Sequence[LoadedFront]) -> dict[str, list[int | float]]:
    observed: dict[str, list[tuple[float, float, float | None]]] = {}
    for front in fronts:
        for candidate in front.candidates:
            bot = _entry_bot(candidate.entry)
            enabled_sides = {
                side
                for side in SIDES
                if isinstance(bot.get(side), Mapping) and _side_is_enabled(bot[side])
            }
            if not enabled_sides:
                continue
            for key, raw_bound in _entry_optimize_bounds(candidate.entry).items():
                if key.startswith("long_") and "long" not in enabled_sides:
                    continue
                if key.startswith("short_") and "short" not in enabled_sides:
                    continue
                parsed = _parse_bound(raw_bound)
                if parsed is None:
                    continue
                observed.setdefault(str(key), []).append(parsed)

    merged: dict[str, list[int | float]] = {}
    for key in sorted(observed):
        values = observed[key]
        low = min(item[0] for item in values)
        high = max(item[1] for item in values)
        steps = [item[2] for item in values if item[2] is not None]
        payload: list[int | float] = [_clean_number(low), _clean_number(high)]
        if steps:
            payload.append(_clean_number(min(steps)))
        merged[key] = payload
    return merged


def _merge_bot(long_component: SideComponent, short_component: SideComponent) -> dict[str, Any]:
    bot: dict[str, Any] = {}
    for source_bot in (long_component.base_bot, short_component.base_bot):
        for key, value in source_bot.items():
            if key in SIDES or key in bot:
                continue
            bot[key] = copy.deepcopy(value)
    bot["long"] = copy.deepcopy(long_component.side_config)
    bot["short"] = copy.deepcopy(short_component.side_config)
    return bot


def _merge_metadata(
    long_component: SideComponent,
    short_component: SideComponent,
    *,
    phase: str,
) -> dict[str, Any]:
    return {
        "tool": "passivbot tool merge-paretos",
        "phase": phase,
        "long": {
            "source": str(long_component.candidate_path),
            "pareto_dir": str(long_component.pareto_dir),
            "selectors": list(long_component.selectors),
            "objectives": long_component.objectives,
        },
        "short": {
            "source": str(short_component.candidate_path),
            "pareto_dir": str(short_component.pareto_dir),
            "selectors": list(short_component.selectors),
            "objectives": short_component.objectives,
        },
    }


def _build_merged_config(
    long_component: SideComponent,
    short_component: SideComponent,
    *,
    merged_bounds: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bot": _merge_bot(long_component, short_component),
        "merge_paretos": _merge_metadata(long_component, short_component, phase=phase),
    }
    if merged_bounds:
        config["optimize"] = {"bounds": copy.deepcopy(dict(merged_bounds))}
    return config


def _add_pair(
    outputs: list[dict[str, Any]],
    seen_pairs: set[tuple[str, str]],
    stats: MergeStats,
    long_component: SideComponent,
    short_component: SideComponent,
    *,
    max_outputs: int,
    merged_bounds: Mapping[str, Any],
    phase: str,
) -> bool:
    pair_key = (long_component.config_hash, short_component.config_hash)
    if pair_key in seen_pairs:
        stats.duplicate_pairs += 1
        return len(outputs) >= max_outputs
    seen_pairs.add(pair_key)
    outputs.append(
        _build_merged_config(
            long_component,
            short_component,
            merged_bounds=merged_bounds,
            phase=phase,
        )
    )
    if phase == "core":
        stats.core_pair_added += 1
    else:
        stats.fill_pair_added += 1
    return len(outputs) >= max_outputs


def build_merged_configs(
    fronts: Sequence[LoadedFront],
    *,
    max_outputs: int,
) -> tuple[list[dict[str, Any]], MergeStats]:
    if max_outputs < 1:
        raise ValueError("--max must be at least 1")

    all_components = _collect_all_components(fronts)
    if not all_components["long"]:
        raise ValueError("No enabled long side configs found in the input Pareto dirs")
    if not all_components["short"]:
        raise ValueError("No enabled short side configs found in the input Pareto dirs")

    core_components = _collect_core_components(fronts)
    if not core_components["long"]:
        core_components["long"] = dict(all_components["long"])
    if not core_components["short"]:
        core_components["short"] = dict(all_components["short"])

    merged_bounds = _merge_bounds(fronts)
    outputs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    stats = MergeStats()

    core_longs = _round_robin_by_front(core_components["long"].values())
    core_shorts = _round_robin_by_front(core_components["short"].values())
    for long_component in core_longs:
        for short_component in core_shorts:
            stats.core_pair_attempts += 1
            if _add_pair(
                outputs,
                seen_pairs,
                stats,
                long_component,
                short_component,
                max_outputs=max_outputs,
                merged_bounds=merged_bounds,
                phase="core",
            ):
                stats.capped = True
                return outputs, stats

    fill_longs = _round_robin_by_front(all_components["long"].values())
    fill_shorts = _round_robin_by_front(all_components["short"].values())
    if not fill_longs or not fill_shorts:
        return outputs, stats

    for offset in range(len(fill_shorts)):
        for idx, long_component in enumerate(fill_longs):
            short_component = fill_shorts[(idx + offset) % len(fill_shorts)]
            stats.fill_pair_attempts += 1
            if _add_pair(
                outputs,
                seen_pairs,
                stats,
                long_component,
                short_component,
                max_outputs=max_outputs,
                merged_bounds=merged_bounds,
                phase="fill",
            ):
                stats.capped = True
                return outputs, stats
    return outputs, stats


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [item for item in path.iterdir() if item.name != ".DS_Store"]
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path} (use --overwrite to replace JSON outputs)"
        )
    for item in existing:
        if item.is_file() and item.suffix == ".json":
            item.unlink()
        else:
            raise FileExistsError(
                f"Refusing to overwrite non-JSON output path entry: {item}. "
                "Use an empty output directory."
            )


def write_outputs(
    output_dir: Path,
    configs: Sequence[Mapping[str, Any]],
    fronts: Sequence[LoadedFront],
    stats: MergeStats,
    *,
    max_outputs: int,
    overwrite: bool,
) -> list[Path]:
    _prepare_output_dir(output_dir, overwrite=overwrite)
    width = max(4, len(str(len(configs))))
    written: list[Path] = []
    for idx, config in enumerate(configs):
        path = output_dir / f"{idx:0{width}d}_{_json_fingerprint(config)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")
        written.append(path)

    index = {
        "tool": "passivbot tool merge-paretos",
        "inputs": [str(front.pareto_dir) for front in fronts],
        "max_outputs": max_outputs,
        "count": len(configs),
        "stats": {
            "core_pair_attempts": stats.core_pair_attempts,
            "core_pair_added": stats.core_pair_added,
            "fill_pair_attempts": stats.fill_pair_attempts,
            "fill_pair_added": stats.fill_pair_added,
            "duplicate_pairs": stats.duplicate_pairs,
            "capped": stats.capped,
        },
        "files": [path.name for path in written],
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    written.append(index_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool merge-paretos",
        description=(
            "Merge two or more Pareto fronts into long/short starting configs. "
            "All positional paths except the last are input run/pareto dirs; "
            "the last path is the output dir."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input Pareto/run dirs followed by the output directory",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_OUTPUTS,
        dest="max_outputs",
        help=f"Maximum merged configs to write (default: {DEFAULT_MAX_OUTPUTS})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing JSON files in the output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merge summary without writing files",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    if len(args.paths) < 3:
        raise SystemExit("Expected at least two input Pareto/run dirs and one output dir")
    input_paths = [Path(raw).expanduser() for raw in args.paths[:-1]]
    output_dir = Path(args.paths[-1]).expanduser()
    fronts = _load_fronts(input_paths)
    configs, stats = build_merged_configs(fronts, max_outputs=args.max_outputs)

    print(f"Loaded {len(fronts)} Pareto fronts:")
    for front in fronts:
        print(f"  [{front.index}] {front.pareto_dir} ({len(front.candidates)} candidates)")
    print(f"Generated {len(configs)} merged configs (max {args.max_outputs}).")
    print(
        "Core added {core}; fill added {fill}; duplicates skipped {duplicates}.".format(
            core=stats.core_pair_added,
            fill=stats.fill_pair_added,
            duplicates=stats.duplicate_pairs,
        )
    )

    if args.dry_run:
        print(f"Dry run: no files written. Output dir would be {output_dir}")
        return 0

    written = write_outputs(
        output_dir,
        configs,
        fronts,
        stats,
        max_outputs=args.max_outputs,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(configs)} configs plus index to {output_dir.resolve()}")
    print(f"Index: {written[-1]}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return run_from_args(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
