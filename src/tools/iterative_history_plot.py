#!/usr/bin/env python3
"""
Quick-and-dirty visualizer for iterative backtester history logs.

Example:
    python src/tools/iterative_history_plot.py backtests/iterative/iterative_20251025_113401 \
        --output tmp/flat_vs_adg.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def _resolve_history_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "history.jsonl"
        if candidate.exists():
            return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(f"No history.jsonl found at {path}")


def _load_history(history_path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with history_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {history_path}: {exc}") from exc
    if not entries:
        raise ValueError(f"No entries found in {history_path}")
    return entries


def _collect_metric(entry: Dict, metric: str) -> float | None:
    scoring = entry.get("scoring", {})
    if metric in scoring:
        return scoring[metric].get("value")
    limits = entry.get("limits", {})
    if isinstance(limits, dict):
        if metric in limits:
            return limits[metric].get("value")
    elif isinstance(limits, list):
        for item in limits:
            if not isinstance(item, dict):
                continue
            if item.get("metric") == metric or item.get("metric_key") == metric:
                return item.get("value")
    # support analysis_combined names, if stored
    combined = entry.get("analysis_combined", {})
    if combined:
        return combined.get(metric) or combined.get(f"{metric}_mean")
    return None


def load_runs(paths: Iterable[Path]) -> List[Dict]:
    runs: List[Dict] = []
    for raw in paths:
        history_path = _resolve_history_path(raw)
        runs.extend(_load_history(history_path))
    if not runs:
        raise ValueError("No runs loaded")
    return runs


def plot_scatter(
    runs: List[Dict],
    x_metric: str,
    y_metric: str,
    output: Path | None,
    title: str | None,
) -> None:
    xs: List[float] = []
    ys: List[float] = []
    colors: List[int] = []
    annotations: List[Tuple[int, float, float]] = []

    for entry in runs:
        run_idx = entry.get("run_index") or entry.get("iteration")
        x_val = _collect_metric(entry, x_metric)
        y_val = _collect_metric(entry, y_metric)
        if x_val is None or y_val is None or run_idx is None:
            continue
        xs.append(x_val)
        ys.append(y_val)
        colors.append(run_idx)
        if entry.get("is_best"):
            annotations.append((run_idx, x_val, y_val))

    if not xs or not ys:
        raise ValueError(f"Could not collect any datapoints for metrics {x_metric} vs {y_metric}")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=40, alpha=0.8)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title or f"{y_metric} vs {x_metric}")
    cbar = plt.colorbar(scatter, ax=ax, label="Run index")

    for run_idx, x_val, y_val in annotations:
        ax.annotate(
            f"best #{run_idx}",
            xy=(x_val, y_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved plot to {output}")
    else:
        plt.show()
    plt.close(fig)
    cbar.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot metrics from iterative backtester history logs."
    )
    parser.add_argument(
        "history_paths",
        nargs="+",
        type=Path,
        help="History.jsonl file or session directory (multiple allowed).",
    )
    parser.add_argument(
        "--x-metric",
        default="peak_recovery_hours_equity_usd",
        help="Metric to plot on X axis (default: peak_recovery_hours_equity_usd).",
    )
    parser.add_argument(
        "--y-metric",
        default="adg_btc_w",
        help="Metric to plot on Y axis (default: adg_btc_w).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for saving the figure instead of showing it.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.history_paths)
    plot_scatter(
        runs,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
