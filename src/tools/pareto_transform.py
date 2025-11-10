"""
Utilities for mutating Pareto front artifacts (conversion, pruning, etc.).

Examples
--------
Convert legacy Pareto entries in-place (dry-run by default)::

    python -m src.tools.pareto_transform pareto/ --convert-metrics

Prune a large Pareto directory down to 500 diverse members (writing changes)::

    python -m src.tools.pareto_transform pareto/ --prune 500 --apply
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import json_dumps_streamlined


METRIC_SUFFIXES = {
    "_mean": "mean",
    "_min": "min",
    "_max": "max",
    "_std": "std",
}


@dataclass
class ParetoEntry:
    path: Path
    data: Dict[str, Any]
    changed: bool = False
    remove: bool = False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def convert_entry(entry: MutableMapping[str, Any]) -> bool:
    """
    Convert legacy ``analyses_combined`` stats/objectives payloads to the
    structured ``metrics`` schema. Returns ``True`` if the entry was mutated.
    """

    analyses = entry.get("analyses_combined")
    if not isinstance(analyses, Mapping):
        return False

    stats: Dict[str, Dict[str, float]] = {}
    objectives: Dict[str, float] = {}
    mutated = False

    for key, value in analyses.items():
        if not _is_number(value):
            continue
        for suffix, field in METRIC_SUFFIXES.items():
            if key.endswith(suffix) and len(key) > len(suffix):
                metric = key[: -len(suffix)]
                stats.setdefault(metric, {})[field] = float(value)
                mutated = True
                break
        else:
            if key.startswith("w_"):
                objectives[key] = float(value)
                mutated = True

    if not stats and not objectives:
        return False

    metrics = entry.setdefault("metrics", {})
    metrics_stats = metrics.setdefault("stats", {})
    for metric, fields in stats.items():
        metrics_stats.setdefault(metric, {}).update(fields)

    if objectives:
        metrics.setdefault("objectives", {}).update(objectives)

    entry.pop("analyses_combined", None)
    return mutated


def extract_objective_map(entry: Mapping[str, Any]) -> Dict[str, float]:
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        objectives = metrics.get("objectives")
        if isinstance(objectives, Mapping) and objectives:
            return {str(k): float(v) for k, v in objectives.items() if _is_number(v)}

    analyses = entry.get("analyses_combined")
    if isinstance(analyses, Mapping):
        return {k: float(v) for k, v in analyses.items() if k.startswith("w_") and _is_number(v)}

    return {}


def build_objective_matrix(
    entries: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    key_set: set[str] = set()
    objective_dicts: List[Dict[str, float]] = []
    valid_indices: List[int] = []
    missing_indices: List[int] = []

    for idx, entry in enumerate(entries):
        obj_map = extract_objective_map(entry)
        if obj_map:
            objective_dicts.append(obj_map)
            valid_indices.append(idx)
            key_set.update(obj_map.keys())
        else:
            missing_indices.append(idx)

    keys = sorted(key_set)
    if not keys:
        matrix = np.zeros((len(valid_indices), 0), dtype=float)
    else:
        matrix = np.array(
            [[obj_map.get(key, 0.0) for key in keys] for obj_map in objective_dicts],
            dtype=float,
        )

    return matrix, keys, valid_indices, missing_indices


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.copy()
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def farthest_point_sampling(points: np.ndarray, count: int, *, seed: int | None = None) -> List[int]:
    if count <= 0:
        return []
    total = points.shape[0]
    if count >= total:
        return list(range(total))

    rng = random.Random(seed)
    norms = np.linalg.norm(points, axis=1)
    min_norm = norms.min()
    first_candidates = np.flatnonzero(np.isclose(norms, min_norm))
    first = (
        int(rng.choice(first_candidates.tolist()))
        if first_candidates.size > 1
        else int(first_candidates[0])
    )

    selected = [first]
    distances = np.linalg.norm(points - points[first], axis=1)

    while len(selected) < count:
        max_dist = distances.max()
        candidates = np.flatnonzero(np.isclose(distances, max_dist))
        idx = int(rng.choice(candidates.tolist())) if candidates.size > 1 else int(candidates[0])
        selected.append(idx)
        new_distances = np.linalg.norm(points - points[idx], axis=1)
        distances = np.minimum(distances, new_distances)

    return sorted(set(selected))


def select_prune_indices(
    entries: Sequence[Mapping[str, Any]],
    target: int,
    *,
    seed: int | None = None,
) -> List[int]:
    """
    Determine which entry indices to keep when pruning to ``target`` members.
    Preference is given to entries lacking objective data (they are always kept),
    and the remainder are selected via farthest-point sampling for diversity.
    """

    total = len(entries)
    if target <= 0:
        return []
    if target >= total:
        return list(range(total))

    matrix, _keys, valid_indices, missing_indices = build_objective_matrix(entries)
    keep: List[int] = list(missing_indices)
    remaining = target - len(keep)

    if remaining <= 0:
        return sorted(keep[:target])

    if matrix.shape[0] == 0:
        keep.extend(valid_indices[:remaining])
        return sorted(keep)

    standardized = _standardize_matrix(matrix)
    chosen = farthest_point_sampling(standardized, min(remaining, standardized.shape[0]), seed=seed)
    keep.extend(valid_indices[idx] for idx in chosen)

    if len(keep) > target:
        keep = keep[:target]

    return sorted(sorted(set(keep)))


def _gather_targets(paths: Sequence[Path]) -> Dict[Path, List[Path]]:
    directories: Dict[Path, List[Path]] = {}
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            files = sorted(p for p in path.glob("*.json") if p.is_file())
            if files:
                directories[path] = files
        elif path.is_file() and path.suffix == ".json":
            directories.setdefault(path.parent, []).append(path)
    return directories


def _write_json(
    path: Path, data: Dict[str, Any], *, indent: int, max_inline: int, sort_keys: bool
) -> None:
    payload = json_dumps_streamlined(
        data,
        indent=indent,
        max_inline=max_inline,
        sort_keys=sort_keys,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def process_directory(
    directory: Path,
    files: Sequence[Path],
    *,
    convert_metrics: bool,
    prune_target: int | None,
    dry_run: bool,
    indent: int,
    max_inline: int,
    sort_keys: bool,
    seed: int | None,
) -> Tuple[int, int]:
    entries: List[ParetoEntry] = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        changed = convert_entry(data) if convert_metrics else False
        entries.append(ParetoEntry(path=file_path, data=data, changed=changed))

    removed = 0
    rewritten = 0

    if prune_target is not None and entries:
        keep_indices = set(
            select_prune_indices([entry.data for entry in entries], prune_target, seed=seed)
        )
        for idx, entry in enumerate(entries):
            entry.remove = idx not in keep_indices

    for entry in entries:
        if entry.remove:
            removed += 1
            action = "Would remove" if dry_run else "Removed"
            print(f"{action} {entry.path}")
            if not dry_run:
                entry.path.unlink(missing_ok=True)
            continue

        if entry.changed:
            rewritten += 1
            action = "Would rewrite" if dry_run else "Rewrote"
            print(f"{action} {entry.path}")
            if not dry_run:
                _write_json(
                    entry.path, entry.data, indent=indent, max_inline=max_inline, sort_keys=sort_keys
                )

    return rewritten, removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply bulk transformations to Pareto front JSON artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Pareto directories or JSON files to process.")
    parser.add_argument(
        "--convert-metrics",
        action="store_true",
        help="Convert legacy analyses payloads to the structured metrics schema.",
    )
    parser.add_argument(
        "--prune",
        type=int,
        metavar="N",
        help="Reduce each directory to at most N members using objective-space sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic pruning tie-breaks.",
    )
    parser.add_argument(
        "--max-inline",
        type=int,
        default=72,
        help="Inline containers up to this character length when rewriting JSON.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indentation level for rewritten JSON.",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort dictionary keys when emitting JSON.",
    )
    parser.add_argument(
        "--apply",
        dest="dry_run",
        action="store_false",
        help="Persist changes instead of running in dry-run mode.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(dry_run=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.convert_metrics and args.prune is None:
        parser.error("Specify at least one action (--convert-metrics or --prune).")

    if args.prune is not None and args.prune < 1:
        parser.error("--prune requires a value greater than zero.")

    paths = [Path(p) for p in args.paths]
    directories = _gather_targets(paths)
    if not directories:
        parser.error("No JSON files found under the provided paths.")

    total_rewritten = 0
    total_removed = 0

    for directory, files in directories.items():
        rewrote, removed = process_directory(
            directory,
            files,
            convert_metrics=args.convert_metrics,
            prune_target=args.prune,
            dry_run=args.dry_run,
            indent=args.indent,
            max_inline=args.max_inline,
            sort_keys=args.sort_keys,
            seed=args.seed,
        )
        total_rewritten += rewrote
        total_removed += removed

    mode = "Dry-run" if args.dry_run else "Applied"
    print(f"{mode} complete: {total_rewritten} rewrites, {total_removed} removals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
