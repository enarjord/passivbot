from __future__ import annotations
import os
import json
import hashlib
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import glob
import math
import time
import numpy as np
import threading
import logging
from dataclasses import dataclass
from pathlib import Path
import passivbot_rust as pbr
from config.limits import resolve_aggregate_mode
from config.metrics import canonicalize_metric_name, resolve_metric_value
from config.scoring import extract_objective_specs
from pure_funcs import calc_hash
from utils import json_dumps_streamlined
from metrics_schema import flatten_metric_stats
from optimization.bounds import Bound
from optimization.config_adapter import get_optimization_key_paths
from pareto_core import (
    compute_ideal,
    crowding_distances,
    dominates_with_violation,
    detect_latest_pareto_dir,
    extract_objectives,
    extract_violation,
    prune_front_with_extremes,
)

STAT_FIELDS = {"mean", "min", "max", "std"}


def _resolve_aggregate_mode(metric: str, aggregate_cfg: Optional[Dict[str, str]]) -> str:
    return resolve_aggregate_mode(metric, aggregate_cfg)


@dataclass(frozen=True)
class LimitSpec:
    metric: str
    field: str  # "mean", "min", "max", "std", or "auto"
    op: Callable[[float, float], bool]
    value: float


def _split_metric_field(raw_key: str) -> tuple[str, str]:
    key = raw_key.strip()
    if "." in key:
        metric, suffix = key.rsplit(".", 1)
        if suffix in STAT_FIELDS:
            return metric, suffix
    return key, "auto"


def _resolve_metric_name(metric: str, metric_map: Dict[str, str]) -> str:
    if metric.startswith("w_"):
        return canonicalize_metric_name(metric_map.get(metric, metric))
    return canonicalize_metric_name(metric)


def _resolve_limit_value(
    spec: LimitSpec,
    stats_flat: Dict[str, float],
    aggregated_values: Dict[str, float],
    objectives: Dict[str, float],
    metric_map: Dict[str, str],
) -> Optional[float]:
    metric = spec.metric
    value = resolve_metric_value(objectives, metric)
    if value is not None:
        return value
    resolved_metric = _resolve_metric_name(metric, metric_map)
    value = resolve_metric_value(objectives, resolved_metric)
    if value is not None:
        return value
    field = spec.field
    if field == "auto":
        if aggregated_values:
            value = resolve_metric_value(aggregated_values, resolved_metric)
            if value is not None:
                return value
        field = "mean"
    key = f"{resolved_metric.replace('.', '_')}_{field}"
    return resolve_metric_value(stats_flat, key)


def _suite_metrics_to_stats(
    entry: Dict[str, Any],
    aggregate_cfg: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    aggregated_values: Dict[str, float] = {}
    stats_flat: Dict[str, float] = {}
    suite_metrics = entry.get("suite_metrics") or {}
    if "metrics" in suite_metrics:
        for metric, payload in suite_metrics["metrics"].items():
            stats = payload.get("stats") or {}
            if stats:
                stats_flat.update(flatten_metric_stats({metric: stats}))
            agg = payload.get("aggregated")
            if agg is None and stats:
                mode = _resolve_aggregate_mode(metric, aggregate_cfg)
                agg = stats.get(mode, stats.get("mean"))
            if agg is not None:
                aggregated_values[metric] = agg
    elif "aggregate" in suite_metrics:
        aggregate = suite_metrics.get("aggregate") or {}
        agg_stats = aggregate.get("stats") or {}
        aggregated_values = aggregate.get("aggregated") or {}
        if not aggregated_values and agg_stats and aggregate_cfg:
            for metric, metric_stats in agg_stats.items():
                mode = _resolve_aggregate_mode(metric, aggregate_cfg)
                val = metric_stats.get(mode, metric_stats.get("mean"))
                if val is not None:
                    aggregated_values[metric] = val
        stats_flat = flatten_metric_stats(agg_stats)
    return stats_flat, aggregated_values


def _quantize_entry_params_with_bounds(
    entry: dict, bounds: Sequence[Bound], log: logging.Logger
) -> dict:
    if not isinstance(entry, dict):
        return entry

    key_paths = get_optimization_key_paths(entry)
    if len(key_paths) != len(bounds):
        log.warning(
            "ParetoStore bounds length mismatch: bounds has %d entries but optimization key list has %d params",
            len(bounds),
            len(key_paths),
        )
    for idx, (_, path) in enumerate(key_paths):
        if idx >= len(bounds):
            return entry
        target = entry
        for part in path[:-1]:
            if not isinstance(target, dict) or part not in target:
                target = None
                break
            target = target[part]
        if not isinstance(target, dict) or path[-1] not in target:
            continue
        bound = bounds[idx]
        value = target[path[-1]]
        if bound.is_stepped:
            target[path[-1]] = bound.quantize(value)
        else:
            target[path[-1]] = (
                bound.high if value > bound.high else bound.low if value < bound.low else value
            )
    return entry


def _evaluate_limits(
    specs: Sequence[LimitSpec],
    stats_flat: Dict[str, float],
    aggregated_values: Dict[str, float],
    objectives: Dict[str, float],
    metric_map: Dict[str, str],
) -> bool:
    for spec in specs:
        value = _resolve_limit_value(spec, stats_flat, aggregated_values, objectives, metric_map)
        if value is None:
            continue
        if not spec.op(value, spec.value):
            return False
    return True


class ParetoStore:
    def __init__(
        self,
        directory: str,
        sig_digits: int = 6,
        bounds: Optional[Sequence[Bound]] = None,
        flush_interval: int = 60,
        log_name: str | None = None,
        max_size: int = 300,
    ):
        self._log = logging.getLogger(log_name or __name__)
        self.directory = directory
        self.pareto_dir = os.path.join(self.directory, "pareto")
        self.sig_digits = sig_digits
        self.bounds = bounds
        self.flush_interval = flush_interval  # seconds
        self.max_size = max(1, int(max_size))
        os.makedirs(os.path.join(self.directory, "pareto"), exist_ok=True)
        # --- in-memory structures -----------------------------------------
        self._entries: dict[str, str] = {}  # hash -> file path
        self._objectives: dict[str, tuple] = {}  # hash -> objective vector
        self._violations: dict[str, float] = {}  # hash -> constraint violation
        self._front: list[str] = []  # list of hashes (Pareto set)
        self._objective_lookup: dict[tuple, str] = {}  # objective vector ➜ hash
        # ------------------------------------------------------------------
        self.n_iters = 0
        self._last_flush_ts = time.time()
        self._lock = threading.RLock()

        self.scoring_keys = None
        self.scoring_specs = None

        # bootstrap from disk if any
        self._bootstrap_from_disk()

    def add_entry(self, entry: dict, *, source_path: str | None = None) -> bool:
        """
        Add a new entry, update Pareto front in‑memory.
        Return True if the store actually changed.
        """
        self.n_iters += 1
        if self.scoring_keys is None:
            self.scoring_specs = extract_objective_specs(entry)
            self.scoring_keys = [spec.metric for spec in self.scoring_specs]
        h = calc_hash(entry)
        with self._lock:
            if h in self._entries:  # fast‑dedupe
                return False

            metrics_block = entry.get("metrics", {}) or {}
            obj, _ = extract_objectives(
                entry, scoring_keys=self.scoring_specs or entry.get("optimize", {}).get("scoring")
            )
            violation = extract_violation(entry)

            # ───────────── NEW: dedupe on the objective vector ──────────────
            existing_hash = self._objective_lookup.get(obj)
            if existing_hash:
                existing_violation = self._violations.get(existing_hash, 0.0)
                if violation >= existing_violation - 1e-12:
                    self._log.info(
                        "Dropping candidate whose obj score is already present with <= violation: %s",
                        obj,
                    )
                    return False
                else:
                    # replace existing entry with higher violation
                    self._remove_from_front(existing_hash)
            # ────────────────────────────────────────────────────────────────

            # discard if dominated by current front
            if any(
                dominates_with_violation(
                    self._objectives[idx],
                    self._violations.get(idx, 0.0),
                    obj,
                    violation,
                    objective_specs=self.scoring_specs,
                )
                for idx in self._front
            ):
                return False

            # remove dominated members
            dominated = [
                idx
                for idx in self._front
                if dominates_with_violation(
                    obj,
                    violation,
                    self._objectives[idx],
                    self._violations.get(idx, 0.0),
                    objective_specs=self.scoring_specs,
                )
            ]
            for idx in dominated:
                self._remove_from_front(idx)

            # add new member
            self._persist_entry(h, entry, source_path=source_path)
            self._objectives[h] = obj
            self._violations[h] = violation
            self._front.append(h)
            self._objective_lookup[obj] = h

            if len(self._front) > self.max_size:
                self._prune_front(len(self._front) - self.max_size)

            self._log_front_state(
                added=1,
                removed=len(dominated),
            )

            # maybe flush
            self._maybe_flush()

            return True

    def get_front(self) -> list[dict]:
        with self._lock:
            results = []
            for h in self._front:
                path = self._entries.get(h)
                if not path:
                    continue
                try:
                    with open(path) as f:
                        results.append(json.load(f))
                except FileNotFoundError:
                    continue
            return results

    def flush_now(self) -> None:
        """Force a write of the current in‑memory set to disk."""
        with self._lock:
            self._write_all_to_disk()
            self._last_flush_ts = time.time()

    def _maybe_flush(self) -> None:
        if time.time() - self._last_flush_ts >= self.flush_interval:
            self._write_all_to_disk()
            self._last_flush_ts = time.time()

    def _write_all_to_disk(self) -> None:
        if not self._front:
            for fp in glob.glob(os.path.join(self.pareto_dir, "*.json")):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            return

        live_files = set(self._entries.values())
        for fp in glob.glob(os.path.join(self.pareto_dir, "*.json")):
            if fp not in live_files:
                try:
                    os.remove(fp)
                except OSError as e:
                    self._log.warning("Could not remove obsolete Pareto file %s: %s", fp, e)

    def _bootstrap_from_disk(self) -> None:
        """
        Read existing *.json files once at start so we don’t lose old results
        when the new optimizer run appends.
        """
        for fp in glob.glob(os.path.join(self.pareto_dir, "*.json")):
            try:
                with open(fp) as f:
                    entry = json.load(f)
                self.add_entry(entry, source_path=fp)
            except Exception as e:
                print(f"bootstrap skip {fp}: {e}")

    def _log_front_state(self, *, added: int, removed: int) -> None:
        """Emit a compact one‑liner with min / max / spread per objective."""
        objs = [self._objectives[idx] for idx in self._front]

        mins = [min(col) for col in zip(*objs)]
        maxs = [max(col) for col in zip(*objs)]

        metrics = []
        for i, key in enumerate(self.scoring_keys):
            metrics.append(
                f"{key}:(" f"{pbr.round_dynamic(mins[i], 3)}," f"{pbr.round_dynamic(maxs[i], 3)}),"
            )

        line = " | ".join(metrics)
        violation_summary = ""
        if self._front:
            viols = [self._violations.get(idx, 0.0) for idx in self._front]
            if viols:
                violation_summary = (
                    f" | constraint:("
                    f"{pbr.round_dynamic(min(viols), 3)},"
                    f"{pbr.round_dynamic(max(viols), 3)})"
                )

        self._log.info(
            f"Iter: {self.n_iters} | Pareto ↑ | +{added}/-{removed} | size:{len(self._front)} | {line}{violation_summary}"
        )

    def _prune_front(self, n_prune: int) -> None:
        """Trim the Pareto front down by removing the most crowded entries."""
        if n_prune <= 0 or len(self._front) <= n_prune:
            return
        to_remove = prune_front_with_extremes(
            self._front, self._objectives, self._violations, len(self._front) - n_prune
        )
        for hash_id in to_remove:
            self._remove_from_front(hash_id)

    def _remove_from_front(self, hash_id: str) -> None:
        obj = self._objectives.pop(hash_id, None)
        if obj is not None:
            self._objective_lookup.pop(obj, None)
        self._violations.pop(hash_id, None)
        self._delete_entry_file(hash_id)
        try:
            self._front.remove(hash_id)
        except ValueError:
            pass

    def _persist_entry(self, hash_id: str, entry: dict, *, source_path: str | None = None) -> None:
        if source_path is None:
            path = os.path.join(self.pareto_dir, f"{hash_id}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(
                    json_dumps_streamlined(
                        entry,
                        indent=4,
                        max_inline=72,
                        separators=(",", ":"),
                    )
                )
            os.replace(tmp, path)
        else:
            path = source_path
            if not os.path.exists(path):
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    f.write(
                        json_dumps_streamlined(
                            entry,
                            indent=4,
                            max_inline=72,
                            separators=(",", ":"),
                        )
                    )
                os.replace(tmp, path)
        self._entries[hash_id] = path

    def _delete_entry_file(self, hash_id: str) -> None:
        path = self._entries.pop(hash_id, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def comma_separated_values_float(x):
    return [float(z) for z in x.split(",")]


def main():
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Inspect optimizer Pareto sets: discover the latest run (or analyze a\n"
            "specified pareto/ directory), compute the ideal point, rank the best\n"
            "solutions, and visualize objective correlations. Supports filtering by\n"
            "metric limits and restricting the objective vector."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 src/pareto_store.py\n"
            "  python3 src/pareto_store.py optimize_results/<run>/pareto\n"
            '  python3 src/pareto_store.py -l "peak_recovery_hours_pnl<800"\n'
            '  python3 src/pareto_store.py -l "peak_recovery_hours_pnl.max<600"\n'
            "  python3 src/pareto_store.py -o adg_btc,mdg_btc -w 0.1,0.1\n"
        ),
    )
    parser.add_argument(
        "pareto_dir",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to a pareto/ directory produced by the optimizer or suite. When\n"
            "omitted the script auto-detects the lexicographically latest run\n"
            "under optimize_results/ whose pareto/ dir has JSON candidates."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the textual summary as JSON for consumption by other tools.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=comma_separated_values_float,
        required=False,
        dest="weights",
        default=None,
        help=(
            "Comma-separated weights for the ideal-point offset. Defaults to zeros\n"
            "which corresponds to the pure component-wise ideal according to each\n"
            "objective goal. Fewer weights than objectives reuse the last provided value."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        dest="mode",
        default="weighted",
        help=(
            "Mode for ideal point computation:\n"
            "  min       – component-wise ideal according to each objective goal\n"
            "  weighted  – honour the --weights offset from the ideal\n"
            "  geomedian – geometric median in objective space"
        ),
    )
    parser.add_argument(
        "-l",
        "--limit",
        "--limits",
        dest="limits",
        action="append",
        help=(
            "Limit filters applied before ranking. Repeat for multiple expressions:\n"
            '  -l "peak_recovery_hours_pnl<800" -l "position_held_hours_max<400"\n'
            "Metrics accept optional suffixes (.min/.max/.mean/.std). Without a suffix\n"
            "suite-level aggregates (if available) are used; otherwise the mean.\n"
            "Legacy w_i identifiers are still accepted for old result files."
        ),
    )
    parser.add_argument(
        "-o",
        "--objectives",
        type=str,
        help=(
            "Restrict the objective vector to the provided comma-separated list\n"
            "(metric names for new results; legacy w_i identifiers are still accepted\n"
            "for old runs). By default all stored objectives are used."
        ),
    )
    args = parser.parse_args()

    pareto_dir = args.pareto_dir
    if not pareto_dir:
        auto_dir = detect_latest_pareto_dir()
        if auto_dir is None:
            parser.error(
                "No pareto directory specified and no valid optimize_results/<run>/pareto "
                "directory with at least one *.json candidate was found. Provide a path explicitly."
            )
        print(f"[info] Using latest pareto directory: {auto_dir}")
        pareto_dir = str(auto_dir)

    pareto_dir = pareto_dir.rstrip("/")
    entries = sorted(glob.glob(os.path.join(pareto_dir, "*.json")))
    if not entries:
        if not pareto_dir.endswith("pareto"):
            pareto_dir += "/pareto"
            entries = sorted(glob.glob(os.path.join(pareto_dir, "*.json")))
    points = []
    filenames = {}
    objective_keys: list[str] = []
    metric_names, metric_name_map = None, None
    objective_specs = None

    import operator
    import re

    OPERATORS = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "=": operator.eq,
    }

    def parse_limit_expr(expr: str) -> LimitSpec:
        for op_str in ["<=", ">=", "<", ">", "==", "="]:
            if op_str in expr:
                key, val = expr.split(op_str, 1)
                metric, field = _split_metric_field(key.strip())
                return LimitSpec(metric, field, OPERATORS[op_str], float(val.strip()))
        raise ValueError(f"Invalid limit expression: {expr}")

    limit_specs: List[LimitSpec] = []
    if args.limits:
        for expr in args.limits:
            try:
                limit_specs.append(parse_limit_expr(expr))
            except Exception as e:
                print(f"Skipping invalid limit expression '{expr}': {e}")

    for entry_path in entries:
        try:
            with open(entry_path) as f:
                entry = json.load(f)
            h = os.path.splitext(os.path.basename(entry_path))[0].split("_")[-1]
            if metric_names is None:
                objective_specs = extract_objective_specs(entry)
                metric_names = [spec.metric for spec in objective_specs]
                metric_name_map = {f"w_{i}": spec.metric for i, spec in enumerate(objective_specs)}
            metrics_block = entry.get("metrics", {}) or {}
            objective_values, extracted_keys = extract_objectives(
                entry,
                scoring_keys=objective_specs or entry.get("optimize", {}).get("scoring"),
            )
            objectives = dict(zip(extracted_keys, objective_values))
            aggregate_cfg = entry.get("backtest", {}).get("aggregate")
            stats_flat: Dict[str, float] = {}
            aggregated_values: Dict[str, float] = {}
            if "stats" in metrics_block:
                stats_flat = flatten_metric_stats(metrics_block["stats"])
            if "suite_metrics" in entry:
                stats_flat_suite, aggregated_values_suite = _suite_metrics_to_stats(
                    entry,
                    aggregate_cfg=aggregate_cfg,
                )
                stats_flat.update(stats_flat_suite)
                aggregated_values.update(aggregated_values_suite)
            if not objective_keys:
                all_objective_keys = list(extracted_keys)
                if args.objectives:
                    requested_objectives = [obj.strip() for obj in args.objectives.split(",")]
                    objective_keys = []
                    for obj_name in requested_objectives:
                        resolved_name = (metric_name_map or {}).get(obj_name, obj_name)
                        if resolved_name in all_objective_keys:
                            objective_keys.append(resolved_name)
                            continue
                        available = list(all_objective_keys)
                        print(
                            f"Warning: Objective '{obj_name}' not found. Available objectives: {available}"
                        )

                    if not objective_keys:
                        print("Error: No valid objectives found. Exiting.")
                        exit(1)
                else:
                    objective_keys = all_objective_keys
            if limit_specs and not _evaluate_limits(
                limit_specs,
                stats_flat,
                aggregated_values,
                objectives,
                metric_name_map or {},
            ):
                continue
            values = [
                resolve_metric_value(objectives, _resolve_metric_name(k, metric_name_map or {}))
                for k in objective_keys
            ]
            if all(v is not None for v in values):
                points.append((*values, h))
                filenames[h] = os.path.split(entry_path)[-1]
        except Exception as e:
            print(f"Error loading {h}: {e}")
    print(f"Found {len(entries)} Pareto members.")
    if args.objectives:
        print(f"Using objectives: {[metric_name_map.get(k, k) for k in objective_keys]}")
    if not points:
        print("No valid Pareto points found.")
        exit(0)

    values_matrix = np.array([p[:-1] for p in points])
    hashes = [p[-1] for p in points]
    if values_matrix.shape[1] != len(objective_keys):
        print("Mismatch between values and keys!")
        exit(1)

    weights = tuple([0.0] * values_matrix.shape[1]) if args.weights is None else args.weights
    if len(weights) == 1:
        weights = tuple([weights[0]] * values_matrix.shape[1])
    selected_specs = None
    if objective_specs:
        spec_by_metric = {spec.metric: spec for spec in objective_specs}
        selected_specs = [
            spec_by_metric[key] for key in objective_keys if key in spec_by_metric
        ] or None

    ideal = compute_ideal(
        values_matrix,
        mode=args.mode,
        weights=weights,
        objective_specs=selected_specs,
    )
    mins = np.min(values_matrix, axis=0)
    maxs = np.max(values_matrix, axis=0)

    norm_matrix = np.array(
        [
            [
                (v - mins[i]) / (maxs[i] - mins[i]) if maxs[i] > mins[i] else v
                for i, v in enumerate(row)
            ]
            for row in values_matrix
        ]
    )
    ideal_norm = [
        (ideal[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] > mins[i] else ideal[i]
        for i in range(len(ideal))
    ]

    dists = np.linalg.norm(norm_matrix - ideal_norm, axis=1)
    closest_idx = int(np.argmin(dists))

    print(f"Ideal point ({args.mode}{' ' + str(weights) if args.mode == 'weighted' else ''})")
    paddings = {k: len(v) for k, v in (metric_name_map or {}).items()} or {"": 0}
    paddings = {k: max(paddings.values()) - v for k, v in paddings.items()}
    for i, key in enumerate(objective_keys):
        print(f"  {metric_name_map.get(key, key)} {' ' * paddings.get(key, 0)} = {ideal[i]:.5f}")
    print(
        f"Closest to ideal: {pareto_dir}/{filenames[hashes[closest_idx]]} | norm_dist={dists[closest_idx]:.5f}"
    )
    for i, key in enumerate(objective_keys):
        print(
            f"  {metric_name_map.get(key, key)} {' ' * paddings.get(key, 0)} = {values_matrix[closest_idx][i]:.5f}"
        )

    if args.json:
        summary = {
            "n_members": len(hashes),
            "ideal": {
                metric_name_map.get(k, k): float(ideal[i]) for i, k in enumerate(objective_keys)
            },
            "closest": {
                "hash": hashes[closest_idx],
                **{
                    metric_name_map.get(k, k): float(values_matrix[closest_idx][i])
                    for i, k in enumerate(objective_keys)
                },
                "normalized_distance": float(dists[closest_idx]),
            },
        }
        print(json.dumps(summary, indent=4))

    fig = plt.figure(figsize=(12, 4))

    if len(objective_keys) == 2:
        ax = fig.add_subplot(111)
        ax.scatter(values_matrix[:, 0], values_matrix[:, 1], label="Pareto Members")
        ax.scatter(*ideal, color="green", label="Ideal Point", zorder=5)
        ax.scatter(
            values_matrix[closest_idx][0],
            values_matrix[closest_idx][1],
            color="red",
            label="Closest to Ideal",
            zorder=5,
        )
        ax.set_xlabel(metric_name_map.get(objective_keys[0], objective_keys[0]))
        ax.set_ylabel(metric_name_map.get(objective_keys[1], objective_keys[1]))
        ax.set_title("Pareto Front")
        ax.legend()
        ax.grid(True)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(objective_keys) == 3:
        import plotly.graph_objs as go
        import plotly.io as pio

        fig = go.Figure()

        # Scatter points for Pareto members
        fig.add_trace(
            go.Scatter3d(
                x=values_matrix[:, 0],
                y=values_matrix[:, 1],
                z=values_matrix[:, 2],
                mode="markers",
                marker=dict(size=4, color="blue"),
                name="Pareto Members",
                text=[f"hash: {h}" for h in hashes],
                hoverinfo="text",
            )
        )

        # Ideal point
        fig.add_trace(
            go.Scatter3d(
                x=[ideal[0]],
                y=[ideal[1]],
                z=[ideal[2]],
                mode="markers",
                marker=dict(size=8, color="green"),
                name="Ideal Point",
            )
        )

        # Closest point
        fig.add_trace(
            go.Scatter3d(
                x=[values_matrix[closest_idx][0]],
                y=[values_matrix[closest_idx][1]],
                z=[values_matrix[closest_idx][2]],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Closest to Ideal",
            )
        )

        fig.update_layout(
            title="Pareto Front (3D Interactive)",
            scene=dict(
                xaxis_title=metric_name_map.get(objective_keys[0], objective_keys[0]),
                yaxis_title=metric_name_map.get(objective_keys[1], objective_keys[1]),
                zaxis_title=metric_name_map.get(objective_keys[2], objective_keys[2]),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0.01, y=0.99),
        )

        fig.show()
    elif len(objective_keys) > 3:
        # More efficient implementation for high-dimensional Pareto fronts
        # Focus only on essential visualizations and optimize performance
        import pandas as pd

        # Convert data to pandas DataFrame for easier handling
        df = pd.DataFrame(values_matrix, columns=objective_keys)
        df["hash"] = hashes
        df["dist_from_ideal"] = dists

        # Sort by distance from ideal
        df_sorted = df.sort_values("dist_from_ideal")

        # Create a streamlined figure with just two key plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Parallel Coordinates - more efficient implementation
        ax = axes[0]

        # Only show up to the top 100 solutions to avoid clutter and improve performance
        top_indices = df_sorted.index[:200]

        # Plot in one batch for better performance
        for i in top_indices:
            if i == closest_idx:
                ax.plot(
                    range(len(objective_keys)), norm_matrix[i], "r-", linewidth=2.5, alpha=0.9, zorder=5
                )
            else:
                ax.plot(range(len(objective_keys)), norm_matrix[i], "b-", linewidth=1, alpha=0.3)

        # Plot ideal point
        ax.plot(range(len(objective_keys)), ideal_norm, "go--", linewidth=2, markersize=8)

        # Customize appearance
        ax.set_xticks(range(len(objective_keys)))
        ax.set_xticklabels(
            [metric_name_map.get(k, k) for k in objective_keys], rotation=45, ha="right"
        )
        ax.set_ylim([0, 1])
        ax.set_title(f"Parallel Coordinates (Top {len(top_indices)} Solutions)")
        ax.grid(True, alpha=0.3)

        # 2. Create a heatmap instead of a radar chart (more compatible)
        ax = axes[1]

        # Create correlation matrix
        corr_matrix = np.zeros((len(objective_keys), len(objective_keys)))
        for i, key1 in enumerate(objective_keys):
            for j, key2 in enumerate(objective_keys):
                vals1 = values_matrix[:, i]
                vals2 = values_matrix[:, j]

                # Calculate correlation
                mean1, mean2 = np.mean(vals1), np.mean(vals2)
                num = np.sum((vals1 - mean1) * (vals2 - mean2))
                den = np.sqrt(np.sum((vals1 - mean1) ** 2) * np.sum((vals2 - mean2) ** 2))

                if den != 0:
                    corr_matrix[i, j] = num / den
                else:
                    corr_matrix[i, j] = 0

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        # Add labels
        ax.set_xticks(range(len(objective_keys)))
        ax.set_yticks(range(len(objective_keys)))
        ax.set_xticklabels(
            [metric_name_map.get(k, k) for k in objective_keys], rotation=45, ha="right"
        )
        ax.set_yticklabels([metric_name_map.get(k, k) for k in objective_keys])

        # Add correlation values as text
        for i in range(len(objective_keys)):
            for j in range(len(objective_keys)):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr_matrix[i, j]) > 0.6 else "black",
                )

        ax.set_title("Objective Correlation Matrix")

        plt.tight_layout()
        # Print top 5 solutions in a compact table format
        print("\nTop 5 Solutions Closest to Ideal:")
        print("-" * 80)
        header = f"{'Rank':<5} {'Hash':<16} {'Distance':<10} " + " ".join(
            [f"{shorten_str(metric_name_map.get(k, k))[:8]:<10}" for k in objective_keys]
        )
        print(header)
        print("-" * 80)

        for rank, (idx, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            values_str = " ".join([f"{row[k]:<10.4f}" for k in objective_keys])
            print(f"{rank:<5} {row['hash']:<16} {row['dist_from_ideal']:<10.4f} {values_str}")

        # Print key insights
        print("\nKey Insights:")
        print("-" * 80)

        # Find strongly correlated objectives
        strong_correlations = []
        for i in range(len(objective_keys)):
            for j in range(i + 1, len(objective_keys)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.65:
                    relation = "positively correlated with" if corr > 0 else "trade-off with"
                    strong_correlations.append(
                        (objective_keys[i], objective_keys[j], corr, relation)
                    )

        if strong_correlations:
            print("Strong relationships between objectives:")
            for obj1, obj2, corr, relation in strong_correlations:
                name1 = metric_name_map.get(obj1, obj1)
                name2 = metric_name_map.get(obj2, obj2)
                print(f"- {name1} is {relation} {name2} (correlation: {corr:.2f})")
        else:
            print("No strong correlations found between objectives.")

        # Calculate diversity of solutions
        diversity_scores = []
        for i in range(len(objective_keys)):
            col_values = values_matrix[:, i]
            min_val = min(col_values)
            max_val = max(col_values)
            diversity = max_val - min_val
            diversity_scores.append((objective_keys[i], diversity))

        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        print("\nObjective diversity (range of values):")
        for obj, score in diversity_scores:
            name = metric_name_map.get(obj, obj)
            print(f"- {name}: {pbr.round_dynamic(score, 4)}")

        plt.show()


def shorten_str(s: str) -> str:
    return s  #''.join(c for c in s if c.lower() not in 'aeiou').replace('_', '')


if __name__ == "__main__":
    main()
