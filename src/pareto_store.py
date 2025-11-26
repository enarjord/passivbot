from __future__ import annotations
import os
import json
import hashlib
from typing import Callable, Dict, Optional, Sequence
import glob
import math
import time
import numpy as np
import threading
import logging
from dataclasses import dataclass
from pathlib import Path
import passivbot_rust as pbr
from opt_utils import round_floats
from pure_funcs import calc_hash
from utils import json_dumps_streamlined
from metrics_schema import flatten_metric_stats
from pareto_core import (
    compute_ideal,
    crowding_distances,
    dominates_with_violation,
    extract_objectives,
    extract_violation,
    prune_front_with_extremes,
)

STAT_FIELDS = {"mean", "min", "max", "std"}


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
        return metric_map.get(metric, metric)
    return metric


def _resolve_limit_value(
    spec: LimitSpec,
    stats_flat: Dict[str, float],
    aggregated_values: Dict[str, float],
    objectives: Dict[str, float],
    metric_map: Dict[str, str],
) -> Optional[float]:
    metric = spec.metric
    if metric.startswith("w_"):
        return objectives.get(metric)
    resolved_metric = _resolve_metric_name(metric, metric_map)
    field = spec.field
    if field == "auto":
        if aggregated_values and resolved_metric in aggregated_values:
            return aggregated_values.get(resolved_metric)
        field = "mean"
    key = f"{resolved_metric.replace('.', '_')}_{field}"
    return stats_flat.get(key)


def _suite_metrics_to_stats(entry: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
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
                agg = stats.get("mean")
            if agg is not None:
                aggregated_values[metric] = agg
    elif "aggregate" in suite_metrics:
        aggregate = suite_metrics.get("aggregate") or {}
        agg_stats = aggregate.get("stats") or {}
        aggregated_values = aggregate.get("aggregated") or {}
        stats_flat = flatten_metric_stats(agg_stats)
    return stats_flat, aggregated_values


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
        flush_interval: int = 60,
        log_name: str | None = None,
        max_size: int = 300,
    ):
        self._log = logging.getLogger(log_name or __name__)
        self.directory = directory
        self.pareto_dir = os.path.join(self.directory, "pareto")
        self.sig_digits = sig_digits
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

        # bootstrap from disk if any
        self._bootstrap_from_disk()

    def add_entry(self, entry: dict, *, source_path: str | None = None) -> bool:
        """
        Add a new entry, update Pareto front in‑memory.
        Return True if the store actually changed.
        """
        self.n_iters += 1
        if self.scoring_keys is None:
            self.scoring_keys = entry["optimize"]["scoring"]
        rounded = round_floats(entry, self.sig_digits)
        h = calc_hash(rounded)
        with self._lock:
            if h in self._entries:  # fast‑dedupe
                return False

            metrics_block = rounded.get("metrics", {}) or {}
            scoring_keys = self.scoring_keys or entry.get("optimize", {}).get("scoring")
            obj, _ = extract_objectives(rounded, scoring_keys=scoring_keys)
            violation = extract_violation(rounded)

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
                )
                for idx in self._front
            ):
                return False

            # remove dominated members
            dominated = [
                idx
                for idx in self._front
                if dominates_with_violation(
                    obj, violation, self._objectives[idx], self._violations.get(idx, 0.0)
                )
            ]
            for idx in dominated:
                self._remove_from_front(idx)

            # add new member
            self._persist_entry(h, rounded, source_path=source_path)
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


def detect_latest_pareto_dir(root: Path = Path("optimize_results")) -> Optional[str]:
    if not root.exists():
        return None
    latest: Optional[tuple[float, Path]] = None
    for child in sorted(root.iterdir()):
        pareto_path = child / "pareto"
        if pareto_path.is_dir():
            mtime = pareto_path.stat().st_mtime
            if latest is None or mtime > latest[0]:
                latest = (mtime, pareto_path)
    if latest is None:
        return None
    return str(latest[1])


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
            '  python3 src/pareto_store.py -l "w_4<800" "peak_recovery_hours_pnl.max<600"\n'
            "  python3 src/pareto_store.py -o adg_btc,mdg_btc -w 1,1\n"
        ),
    )
    parser.add_argument(
        "pareto_dir",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to a pareto/ directory produced by the optimizer or suite. When\n"
            "omitted the script auto-detects the most recent run under\n"
            "optimize_results/."
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
            "which corresponds to the pure component-wise minimum. Fewer weights than\n"
            "objectives reuse the last provided value."
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
            "  min       – component-wise minima (default)\n"
            "  weighted  – honour the --weights offset\n"
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
            '  -l "w_4<800" -l "position_held_hours_max<400"\n'
            "Metrics accept optional suffixes (.min/.max/.mean/.std). Without a suffix\n"
            "suite-level aggregates (if available) are used; otherwise the mean."
        ),
    )
    parser.add_argument(
        "-o",
        "--objectives",
        type=str,
        help=(
            "Restrict the objective vector to the provided comma-separated list\n"
            "(names or w_i identifiers). By default all objectives stored in the\n"
            "Pareto entry are used."
        ),
    )
    args = parser.parse_args()

    pareto_dir = args.pareto_dir
    if not pareto_dir:
        auto_dir = detect_latest_pareto_dir()
        if auto_dir is None:
            parser.error(
                "No pareto directory specified and none found under optimize_results/. "
                "Provide a path explicitly."
            )
        print(f"[info] Using latest pareto directory: {auto_dir}")
        pareto_dir = auto_dir

    pareto_dir = pareto_dir.rstrip("/")
    entries = sorted(glob.glob(os.path.join(pareto_dir, "*.json")))
    if not entries:
        if not pareto_dir.endswith("pareto"):
            pareto_dir += "/pareto"
            entries = sorted(glob.glob(os.path.join(pareto_dir, "*.json")))
    points = []
    filenames = {}
    w_keys = []
    metric_names, metric_name_map = None, None

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
                metric_names = entry.get("optimize", {}).get("scoring", [])
                metric_name_map = {f"w_{i}": name for i, name in enumerate(metric_names)}
            metrics_block = entry.get("metrics", {}) or {}
            objectives = metrics_block.get("objectives", metrics_block)
            stats_flat: Dict[str, float] = {}
            aggregated_values: Dict[str, float] = {}
            if "stats" in metrics_block:
                stats_flat = flatten_metric_stats(metrics_block["stats"])
            if "suite_metrics" in entry:
                stats_flat_suite, aggregated_values_suite = _suite_metrics_to_stats(entry)
                stats_flat.update(stats_flat_suite)
                aggregated_values.update(aggregated_values_suite)
            if not w_keys:
                all_w_keys = sorted(k for k in objectives if k.startswith("w_"))

                # Filter w_keys based on --objectives argument
                if args.objectives:
                    requested_objectives = [obj.strip() for obj in args.objectives.split(",")]
                    # Map objective names to w_keys using metric_name_map
                    if metric_names is None:
                        metric_names = entry.get("optimize", {}).get("scoring", [])
                        metric_name_map = {f"w_{i}": name for i, name in enumerate(metric_names)}

                    # Create reverse mapping: objective name -> w_key
                    reverse_map = {name: key for key, name in metric_name_map.items()}

                    # Filter w_keys to only include requested objectives
                    w_keys = []
                    for obj_name in requested_objectives:
                        if obj_name in reverse_map:
                            w_keys.append(reverse_map[obj_name])
                        else:
                            # Check if user provided w_key directly
                            if obj_name in all_w_keys:
                                w_keys.append(obj_name)
                            else:
                                print(
                                    f"Warning: Objective '{obj_name}' not found. Available objectives: {list(reverse_map.keys())}"
                                )

                    if not w_keys:
                        print("Error: No valid objectives found. Exiting.")
                        exit(1)

                    w_keys = sorted(w_keys)
                else:
                    w_keys = all_w_keys
            if limit_specs and not _evaluate_limits(
                limit_specs,
                stats_flat,
                aggregated_values,
                objectives,
                metric_name_map or {},
            ):
                continue
            values = [objectives.get(k) for k in w_keys]
            if all(v is not None for v in values):
                points.append((*values, h))
                filenames[h] = os.path.split(entry_path)[-1]
        except Exception as e:
            print(f"Error loading {h}: {e}")
    print(f"Found {len(entries)} Pareto members.")
    if args.objectives:
        print(f"Using objectives: {[metric_name_map.get(k, k) for k in w_keys]}")
    if not points:
        print("No valid Pareto points found.")
        exit(0)

    values_matrix = np.array([p[:-1] for p in points])
    hashes = [p[-1] for p in points]
    if values_matrix.shape[1] != len(w_keys):
        print("Mismatch between values and keys!")
        exit(1)

    weights = tuple([0.0] * values_matrix.shape[1]) if args.weights is None else args.weights
    if len(weights) == 1:
        weights = tuple([weights[0]] * values_matrix.shape[1])

    ideal = compute_ideal(values_matrix, mode=args.mode, weights=weights)
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
    paddings = {k: len(v) for k, v in metric_name_map.items()}
    paddings = {k: max(paddings.values()) - v for k, v in paddings.items()}
    for i, key in enumerate(w_keys):
        print(f"  {key} ({metric_name_map[key]}) {' ' * paddings[key]} = {ideal[i]:.5f}")
    print(
        f"Closest to ideal: {pareto_dir}/{filenames[hashes[closest_idx]]} | norm_dist={dists[closest_idx]:.5f}"
    )
    for i, key in enumerate(w_keys):
        print(
            f"  {key} ({metric_name_map[key]}) {' ' * paddings[key]} = {values_matrix[closest_idx][i]:.5f}"
        )

    if args.json:
        summary = {
            "n_members": len(hashes),
            "ideal": {k: float(ideal[i]) for i, k in enumerate(w_keys)},
            "closest": {
                "hash": hashes[closest_idx],
                **{k: float(values_matrix[closest_idx][i]) for i, k in enumerate(w_keys)},
                "normalized_distance": float(dists[closest_idx]),
            },
        }
        print(json.dumps(summary, indent=4))

    fig = plt.figure(figsize=(12, 4))

    if len(w_keys) == 2:
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
        ax.set_xlabel(w_keys[0])
        ax.set_ylabel(w_keys[1])
        ax.set_title("Pareto Front")
        ax.legend()
        ax.grid(True)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(w_keys) == 3:
        import plotly.graph_objs as go
        import plotly.io as pio

        # Load a config to get metric names
        sample_entry_path = entries[0]
        with open(sample_entry_path) as f:
            sample_entry = json.load(f)

        # Try to read optimize.scoring if available
        metric_names = sample_entry.get("optimize", {}).get("scoring", [])
        metric_name_map = {f"w_{i}": name for i, name in enumerate(metric_names)}

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
                xaxis_title=metric_name_map.get(w_keys[0], w_keys[0]),
                yaxis_title=metric_name_map.get(w_keys[1], w_keys[1]),
                zaxis_title=metric_name_map.get(w_keys[2], w_keys[2]),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0.01, y=0.99),
        )

        fig.show()
    elif len(w_keys) > 3:
        # More efficient implementation for high-dimensional Pareto fronts
        # Focus only on essential visualizations and optimize performance
        import pandas as pd

        # Convert data to pandas DataFrame for easier handling
        df = pd.DataFrame(values_matrix, columns=w_keys)
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
                ax.plot(range(len(w_keys)), norm_matrix[i], "r-", linewidth=2.5, alpha=0.9, zorder=5)
            else:
                ax.plot(range(len(w_keys)), norm_matrix[i], "b-", linewidth=1, alpha=0.3)

        # Plot ideal point
        ax.plot(range(len(w_keys)), ideal_norm, "go--", linewidth=2, markersize=8)

        # Customize appearance
        ax.set_xticks(range(len(w_keys)))
        ax.set_xticklabels([metric_name_map.get(k, k) for k in w_keys], rotation=45, ha="right")
        ax.set_ylim([0, 1])
        ax.set_title(f"Parallel Coordinates (Top {len(top_indices)} Solutions)")
        ax.grid(True, alpha=0.3)

        # 2. Create a heatmap instead of a radar chart (more compatible)
        ax = axes[1]

        # Create correlation matrix
        corr_matrix = np.zeros((len(w_keys), len(w_keys)))
        for i, key1 in enumerate(w_keys):
            for j, key2 in enumerate(w_keys):
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
        ax.set_xticks(range(len(w_keys)))
        ax.set_yticks(range(len(w_keys)))
        ax.set_xticklabels([metric_name_map.get(k, k) for k in w_keys], rotation=45, ha="right")
        ax.set_yticklabels([metric_name_map.get(k, k) for k in w_keys])

        # Add correlation values as text
        for i in range(len(w_keys)):
            for j in range(len(w_keys)):
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
            [f"{shorten_str(metric_name_map.get(k, k))[:8]:<10}" for k in w_keys]
        )
        print(header)
        print("-" * 80)

        for rank, (idx, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            values_str = " ".join([f"{row[k]:<10.4f}" for k in w_keys])
            print(f"{rank:<5} {row['hash']:<16} {row['dist_from_ideal']:<10.4f} {values_str}")

        # Print key insights
        print("\nKey Insights:")
        print("-" * 80)

        # Find strongly correlated objectives
        strong_correlations = []
        for i in range(len(w_keys)):
            for j in range(i + 1, len(w_keys)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.65:
                    relation = "positively correlated with" if corr > 0 else "trade-off with"
                    strong_correlations.append((w_keys[i], w_keys[j], corr, relation))

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
        for i in range(len(w_keys)):
            col_values = values_matrix[:, i]
            min_val = min(col_values)
            max_val = max(col_values)
            diversity = max_val - min_val
            diversity_scores.append((w_keys[i], diversity))

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
