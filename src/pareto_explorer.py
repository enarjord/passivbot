from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from config.limits import (
    normalize_limit_entries,
    parse_limit_cli_entries,
    resolve_aggregate_mode,
    resolve_limit_stat,
)
from config.metrics import resolve_metric_value
from config.scoring import (
    ObjectiveSpec,
    default_objective_goal,
    extract_objective_specs,
    from_engine_value,
    objective_spec_by_metric,
)
from metrics_schema import flatten_metric_stats


METHOD_ALIASES = {
    "knee": "knee",
    "k": "knee",
    "reference": "reference",
    "ref": "reference",
    "r": "reference",
    "ideal": "ideal",
    "i": "ideal",
    "utility": "utility",
    "u": "utility",
    "weighted": "utility",
    "lexicographic": "lexicographic",
    "lex": "lexicographic",
    "l": "lexicographic",
    "outranking": "outranking",
    "out": "outranking",
    "o": "outranking",
}


METHOD_DESCRIPTIONS = {
    "knee": "Approximate knee-point chooser. Picks a balanced compromise on the Pareto front.",
    "reference": "Reference-point chooser. Picks the candidate closest to user targets.",
    "ideal": "Distance-to-ideal chooser. Picks the candidate closest to the observed ideal point.",
    "utility": "Weighted utility chooser. Picks the highest weighted normalized utility.",
    "lexicographic": "Strict priority chooser. Sorts by objective priority order.",
    "outranking": "Simplified PROMETHEE-style chooser based on pairwise net preference flow.",
}


@dataclass(frozen=True)
class ParetoCandidate:
    path: Path
    entry: Dict[str, Any]
    objectives: Dict[str, float]
    stats_flat: Dict[str, float]
    aggregated_values: Dict[str, float]


@dataclass(frozen=True)
class SelectionResult:
    candidate: ParetoCandidate
    method: str
    score: float
    objective_values: Dict[str, float]
    details: Dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _public_selection_details(details: Mapping[str, Any]) -> Dict[str, Any]:
    hidden = {"ranking_order", "score_vector"}
    return {str(k): _json_ready(v) for k, v in details.items() if k not in hidden}


def _method_explanation(method: str) -> str:
    return METHOD_DESCRIPTIONS.get(method, "").strip()


def _selection_rationale_lines(result: SelectionResult) -> List[str]:
    method = result.method
    if method == "knee":
        mode = str(result.details.get("knee_mode", "")).strip()
        if mode == "hyperplane_distance":
            return ["Why this winner: strongest balanced compromise away from the extreme-anchor hyperplane."]
        if mode == "maximin_fallback":
            return ["Why this winner: strongest worst-objective utility among retained candidates."]
        return ["Why this winner: only retained candidate."]
    if method == "reference":
        return ["Why this winner: smallest weighted distance to the supplied target utilities."]
    if method == "ideal":
        return ["Why this winner: smallest weighted distance to the observed ideal point on this front."]
    if method == "utility":
        return ["Why this winner: highest weighted normalized utility after objective scaling."]
    if method == "lexicographic":
        return ["Why this winner: best on the first priority objective, then tie-broken by the next priorities."]
    if method == "outranking":
        return ["Why this winner: strongest net pairwise preference flow against the other retained candidates."]
    return []


def _summarize_anchor_files(anchor_files: Sequence[str], *, preview: int = 4) -> str:
    if len(anchor_files) <= preview:
        return ", ".join(str(item) for item in anchor_files)
    shown = ", ".join(str(item) for item in list(anchor_files)[:preview])
    hidden = len(anchor_files) - preview
    return f"{shown} ... (+{hidden} more)"


def _abbreviate_path(path: Path | str, *, max_len: int = 100) -> str:
    value = str(path)
    home = str(Path.home())
    if value.startswith(home):
        value = "~" + value[len(home) :]
    if len(value) <= max_len:
        return value
    parts = value.split(os.sep)
    if len(parts) <= 3:
        keep = max(0, max_len - 3)
        return f"...{value[-keep:]}"
    prefix = parts[0]
    suffix = parts[-2:]
    middle = parts[1:-2]
    candidate = os.sep.join([prefix, "...", *suffix])
    while len(candidate) > max_len and suffix:
        suffix = suffix[1:]
        candidate = os.sep.join([prefix, "...", *suffix]) if suffix else f"{prefix}{os.sep}..."
    if len(candidate) <= max_len:
        return candidate
    if middle:
        tail = os.sep.join(parts[-1:])
        keep = max(0, max_len - len(prefix) - len(tail) - 6)
        mid_joined = os.sep.join(middle)
        return f"{prefix}{os.sep}...{mid_joined[-keep:]}{os.sep}{tail}"
    keep = max(0, max_len - 3)
    return f"...{value[-keep:]}"


def _display_path(path: Path | str, *, base_dir: Path | None = None) -> str:
    resolved = Path(path).resolve()
    search_bases: List[Path] = []
    if base_dir is not None:
        search_bases.append(Path(base_dir).resolve())
    search_bases.append(Path.cwd().resolve())
    for base in search_bases:
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def _render_key_value_box(rows: Sequence[tuple[str, str]]) -> List[str]:
    if not rows:
        return []
    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    lines = [border]
    for key, value in rows:
        lines.append(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    lines.append(border)
    return lines


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    if not rows:
        return []
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    lines = [border, header_line, border]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |")
    lines.append(border)
    return lines


def _format_limit_entry(entry: Mapping[str, Any]) -> str:
    metric = str(entry.get("metric", "")).strip()
    mode = str(entry.get("penalize_if", "")).strip().lower()
    value = entry.get("value")
    range_value = entry.get("range")
    op_map = {
        "greater_than": ">",
        "greater_than_or_equal": ">=",
        "less_than": "<",
        "less_than_or_equal": "<=",
        "equal_to": "==",
        "not_equal": "!=",
    }
    if mode in op_map and value is not None:
        return f"{metric} {op_map[mode]} {value}"
    if mode == "outside_range" and isinstance(range_value, Sequence) and len(range_value) == 2:
        return f"{metric} outside [{range_value[0]}, {range_value[1]}]"
    if mode == "inside_range" and isinstance(range_value, Sequence) and len(range_value) == 2:
        return f"{metric} inside [{range_value[0]}, {range_value[1]}]"
    return str(dict(entry))


def _wrap_csv(values: Sequence[str], *, width: int = 104, indent: str = "") -> List[str]:
    text = ", ".join(str(value) for value in values)
    return textwrap.wrap(text, width=width, subsequent_indent=indent) or [""]


def _format_metric_value(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        magnitude = abs(float(value))
        if magnitude >= 1000:
            return f"{float(value):.1f}"
        if magnitude >= 100:
            return f"{float(value):.2f}"
        if magnitude >= 1:
            return f"{float(value):.3f}"
        return f"{float(value):.6f}"
    return str(value)


def _score_label_and_value(result: SelectionResult) -> tuple[str, str]:
    if result.method in {"ideal", "reference"}:
        return "Distance", f"{abs(float(result.score)):.6f}"
    if result.method == "utility":
        return "Utility score", f"{float(result.score):.6f}"
    if result.method == "outranking":
        return "Net flow", f"{float(result.score):.6f}"
    if result.method == "lexicographic":
        return "Lexicographic score", f"{float(result.score):.6f}"
    if result.method == "knee":
        return "Knee score", f"{float(result.score):.6f}"
    return "Score", f"{float(result.score):.6f}"


def detect_latest_pareto_dir(root: str | os.PathLike[str] = "optimize_results") -> Optional[Path]:
    base = Path(root).expanduser()
    if not base.is_dir():
        return None
    candidates = [path for path in base.glob("*/pareto") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def resolve_pareto_directory(path: str | os.PathLike[str]) -> Path:
    raw = Path(path).expanduser()
    if raw.is_dir():
        if raw.name == "pareto":
            pareto_dir = raw
        elif (raw / "pareto").is_dir():
            pareto_dir = raw / "pareto"
        else:
            pareto_dir = raw
    else:
        raise FileNotFoundError(f"Pareto path not found: {raw}")
    if not pareto_dir.is_dir():
        raise FileNotFoundError(f"Pareto directory not found: {pareto_dir}")
    return pareto_dir.resolve()


def _parse_key_value_pairs(raw_pairs: Iterable[str], *, value_name: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for raw in raw_pairs:
        token = str(raw).strip()
        if not token or "=" not in token:
            raise ValueError(f"Expected {value_name} in the form metric=value, got {raw!r}")
        metric, value = token.split("=", 1)
        metric = metric.strip()
        if not metric:
            raise ValueError(f"Expected {value_name} metric name before '=', got {raw!r}")
        try:
            parsed[metric] = float(value.strip())
        except Exception as exc:
            raise ValueError(f"Invalid numeric {value_name} value in {raw!r}") from exc
    return parsed


def _resolve_candidate_metric_value(candidate: ParetoCandidate, metric: str) -> Optional[float]:
    value = resolve_metric_value(candidate.objectives, metric)
    if value is not None:
        return float(value)
    value = resolve_metric_value(candidate.aggregated_values, metric)
    if value is not None:
        return float(value)
    mean_key = f"{metric}_mean"
    value = resolve_metric_value(candidate.stats_flat, mean_key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def parse_method_name(raw_method: str) -> str:
    method = METHOD_ALIASES.get(str(raw_method or "").strip().lower())
    if method is None:
        allowed = ", ".join(sorted(dict.fromkeys(METHOD_ALIASES.values())))
        raise ValueError(f"Unknown method {raw_method!r}; expected one of: {allowed}")
    return method


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto",
        description="Select a single candidate from a Pareto front directory.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Methods:\n"
            "  knee         Approximate balanced compromise selector.\n"
            "  reference    Closest to user targets (--target metric=value).\n"
            "  ideal        Closest to the observed ideal point.\n"
            "  utility      Highest weighted normalized utility (--weight metric=value).\n"
            "  lexicographic Strict priority order (--priority metric_a,metric_b,...).\n"
            "  outranking   Simplified PROMETHEE-style net flow selector.\n\n"
            "Limits are applied before selection. Repeat -l/--limit for multiple keep-conditions:\n"
            "  -l 'adg_strategy_eq>0.0'\n"
            "  -l 'drawdown_worst_strategy_eq<=0.35'\n"
            "  --limits '[{\"metric\":\"drawdown_worst_strategy_eq\",\"penalize_if\":\">\",\"value\":0.35}]'\n"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help="Pareto directory or optimization run directory. Defaults to the newest optimize_results/.../pareto.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="ideal",
        help="Selection method. Default: ideal.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        action="append",
        dest="limit_entries",
        default=None,
        metavar="SPEC",
        help="Repeatable keep-condition filter, using optimizer-style CLI syntax.",
    )
    parser.add_argument(
        "--limits",
        dest="limits_payload",
        default=None,
        metavar="JSON_OR_HJSON",
        help="Whole-list limit payload using canonical optimize.limits schema.",
    )
    parser.add_argument(
        "-o",
        "--objectives",
        type=str,
        default=None,
        help="Optional comma-separated subset of metrics to consider. May include stored non-scoring metrics with known min/max direction.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="METRIC=VALUE",
        help="Repeatable method weight. Used by utility, ideal, reference, and outranking.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="METRIC=VALUE",
        help="Repeatable reference-point target. Required for method=reference.",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default=None,
        help="Comma-separated objective priority order for method=lexicographic.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=1,
        metavar="N",
        help="Show the top N ranked candidates instead of only the winner. Default: 1.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser


def _extract_suite_metrics(
    entry: Mapping[str, Any],
    aggregate_cfg: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    aggregated_values: Dict[str, float] = {}
    stats_flat: Dict[str, float] = {}
    suite_metrics = entry.get("suite_metrics")
    if not isinstance(suite_metrics, Mapping):
        return stats_flat, aggregated_values
    effective_aggregate_cfg = (
        aggregate_cfg
        if aggregate_cfg is not None
        else entry.get("backtest", {}).get("aggregate")
        if isinstance(entry.get("backtest"), Mapping)
        else None
    )

    if "metrics" in suite_metrics:
        for metric, payload in suite_metrics["metrics"].items():
            if not isinstance(payload, Mapping):
                continue
            aggregated = payload.get("aggregated")
            stats = payload.get("stats") or {}
            if aggregated is None and isinstance(stats, Mapping):
                mode = resolve_aggregate_mode(str(metric), effective_aggregate_cfg)
                aggregated = stats.get(mode, stats.get("mean"))
            if isinstance(aggregated, (int, float)) and math.isfinite(float(aggregated)):
                aggregated_values[str(metric)] = float(aggregated)
            if isinstance(stats, Mapping):
                stats_flat.update(flatten_metric_stats({str(metric): dict(stats)}))
        return stats_flat, aggregated_values

    aggregate = suite_metrics.get("aggregate") or {}
    if isinstance(aggregate, Mapping):
        stats = aggregate.get("stats") or {}
        if isinstance(stats, Mapping):
            stats_flat.update(flatten_metric_stats(dict(stats)))
        aggregated = aggregate.get("aggregated") or {}
        if isinstance(aggregated, Mapping):
            for metric, value in aggregated.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    aggregated_values[str(metric)] = float(value)
        elif isinstance(stats, Mapping):
            for metric, metric_stats in stats.items():
                if not isinstance(metric_stats, Mapping):
                    continue
                mode = resolve_aggregate_mode(str(metric), effective_aggregate_cfg)
                value = metric_stats.get(mode, metric_stats.get("mean"))
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    aggregated_values[str(metric)] = float(value)
    return stats_flat, aggregated_values


def _extract_objectives(entry: Mapping[str, Any]) -> Dict[str, float]:
    scoring_specs = extract_objective_specs(entry)
    metrics_block = entry.get("metrics") or {}
    if not isinstance(metrics_block, Mapping):
        metrics_block = {}
    objective_payload = metrics_block.get("objectives") or {}
    objectives: Dict[str, float] = {}

    if isinstance(objective_payload, Mapping):
        for idx, spec in enumerate(scoring_specs):
            value = resolve_metric_value(objective_payload, spec.metric)
            if value is None:
                legacy_key = f"w_{idx}"
                if legacy_key in objective_payload:
                    value = from_engine_value(spec, float(objective_payload[legacy_key]))
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                objectives[spec.metric] = float(value)

    if len(objectives) == len(scoring_specs):
        return objectives

    stats_flat: Dict[str, float] = {}
    raw_stats = metrics_block.get("stats") or {}
    if isinstance(raw_stats, Mapping):
        stats_flat.update(flatten_metric_stats(dict(raw_stats)))
    suite_stats_flat, aggregated_values = _extract_suite_metrics(entry)
    stats_flat.update(suite_stats_flat)

    for spec in scoring_specs:
        if spec.metric in objectives:
            continue
        value = resolve_metric_value(aggregated_values, spec.metric)
        if value is not None:
            objectives[spec.metric] = float(value)
            continue
        metric_key = f"{spec.metric}_mean"
        value = resolve_metric_value(stats_flat, metric_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            objectives[spec.metric] = float(value)
    return objectives


def load_candidates(path: str | os.PathLike[str]) -> tuple[Path, List[ParetoCandidate], List[ObjectiveSpec]]:
    pareto_dir = resolve_pareto_directory(path)
    json_paths = sorted(pareto_dir.glob("*.json"))
    if not json_paths:
        raise ValueError(f"No Pareto JSON files found in {pareto_dir}")

    candidates: List[ParetoCandidate] = []
    baseline_specs: Optional[List[ObjectiveSpec]] = None
    baseline_metrics: Optional[List[str]] = None

    for entry_path in json_paths:
        with open(entry_path) as f:
            entry = json.load(f)
        specs = extract_objective_specs(entry)
        metrics = [spec.metric for spec in specs]
        if baseline_specs is None:
            baseline_specs = specs
            baseline_metrics = metrics
        elif metrics != baseline_metrics:
            raise ValueError(
                f"Inconsistent optimize.scoring in {entry_path}; expected {baseline_metrics}, got {metrics}"
            )

        metrics_block = entry.get("metrics") or {}
        stats_flat: Dict[str, float] = {}
        if isinstance(metrics_block, Mapping):
            raw_stats = metrics_block.get("stats") or {}
            if isinstance(raw_stats, Mapping):
                stats_flat.update(flatten_metric_stats(dict(raw_stats)))
        suite_stats_flat, aggregated_values = _extract_suite_metrics(entry)
        stats_flat.update(suite_stats_flat)
        objectives = _extract_objectives(entry)

        missing = [metric for metric in baseline_metrics or [] if metric not in objectives]
        if missing:
            raise ValueError(f"Missing objective values for {entry_path}: {missing}")

        candidates.append(
            ParetoCandidate(
                path=entry_path.resolve(),
                entry=entry,
                objectives=objectives,
                stats_flat=stats_flat,
                aggregated_values=aggregated_values,
            )
        )

    assert baseline_specs is not None
    return pareto_dir, candidates, baseline_specs


def _resolve_active_objective_metrics(
    scoring_specs: Sequence[ObjectiveSpec],
    candidates: Sequence[ParetoCandidate],
    *,
    objectives_arg: Optional[str],
    priority_arg: Optional[str],
    target_map: Optional[Dict[str, float]],
    method: str,
) -> List[ObjectiveSpec]:
    available_specs = objective_spec_by_metric(scoring_specs)
    available = [spec.metric for spec in scoring_specs]
    if priority_arg:
        requested = [item.strip() for item in priority_arg.split(",") if item.strip()]
    elif objectives_arg:
        requested = [item.strip() for item in objectives_arg.split(",") if item.strip()]
    elif method == "reference" and target_map:
        requested = list(target_map.keys())
    else:
        requested = available

    resolved: List[ObjectiveSpec] = []
    invalid: List[str] = []
    for raw_metric in requested:
        metric = str(raw_metric).strip()
        if not metric:
            continue
        if metric in available_specs:
            resolved.append(available_specs[metric])
            continue
        goal = default_objective_goal(metric)
        if goal is None:
            invalid.append(metric)
            continue
        if any(_resolve_candidate_metric_value(candidate, metric) is None for candidate in candidates):
            invalid.append(metric)
            continue
        resolved.append(ObjectiveSpec(metric=metric, goal=goal))

    if invalid:
        raise ValueError(
            f"Unknown or unavailable objective metric(s): {invalid}; available scoring metrics: {available}"
        )
    return resolved


def _normalize_objective_matrix(
    candidates: Sequence[ParetoCandidate],
    active_specs: Sequence[ObjectiveSpec],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_metrics = [spec.metric for spec in active_specs]
    raw = np.array(
        [
            [float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics]
            for candidate in candidates
        ],
        dtype=float,
    )
    lows = raw.min(axis=0)
    highs = raw.max(axis=0)
    spans = highs - lows
    utilities = np.ones_like(raw, dtype=float)
    for idx, metric in enumerate(active_metrics):
        span = spans[idx]
        if span <= 1e-15:
            utilities[:, idx] = 1.0
            continue
        spec = active_specs[idx]
        if spec.goal == "max":
            utilities[:, idx] = (raw[:, idx] - lows[idx]) / span
        else:
            utilities[:, idx] = (highs[idx] - raw[:, idx]) / span
    return utilities, lows, highs


def _weights_for_metrics(metrics: Sequence[str], weight_map: Optional[Dict[str, float]]) -> np.ndarray:
    weights = np.array([float((weight_map or {}).get(metric, 1.0)) for metric in metrics], dtype=float)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("At least one weight must be positive.")
    return weights / weights.sum()


def _build_ideal_point(
    candidates: Sequence[ParetoCandidate],
    active_specs: Sequence[ObjectiveSpec],
) -> Dict[str, float]:
    ideal: Dict[str, float] = {}
    for spec in active_specs:
        values = [float(_resolve_candidate_metric_value(candidate, spec.metric)) for candidate in candidates]
        ideal[spec.metric] = max(values) if spec.goal == "max" else min(values)
    return ideal


def _attach_shared_selection_details(
    result: SelectionResult,
    *,
    candidates: Sequence[ParetoCandidate],
    active_specs: Sequence[ObjectiveSpec],
) -> SelectionResult:
    ideal_point = _build_ideal_point(candidates, active_specs)
    details = dict(result.details)
    details["ideal_point"] = ideal_point
    return SelectionResult(
        candidate=result.candidate,
        method=result.method,
        score=result.score,
        objective_values=result.objective_values,
        details=details,
    )


def _ranked_rows(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    ranking_order: Sequence[int],
    score_vector: Sequence[float],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, idx in enumerate(list(ranking_order)[: max(1, int(limit))], start=1):
        candidate = candidates[int(idx)]
        rows.append(
            {
                "rank": rank,
                "score": float(score_vector[int(idx)]),
                "file": candidate.path.name,
                "hash": candidate.path.stem,
                "path": str(candidate.path),
                "objectives": {
                    metric: float(_resolve_candidate_metric_value(candidate, metric))
                    for metric in active_metrics
                },
            }
        )
    return rows


def _normalize_reference_targets(
    targets: Dict[str, float],
    active_specs: Sequence[ObjectiveSpec],
    lows: np.ndarray,
    highs: np.ndarray,
) -> np.ndarray:
    active_metrics = [spec.metric for spec in active_specs]
    target_values = []
    for idx, metric in enumerate(active_metrics):
        if metric not in targets:
            raise ValueError(f"Reference method requires target for {metric}")
        raw_value = float(targets[metric])
        low = lows[idx]
        high = highs[idx]
        span = high - low
        if span <= 1e-15:
            utility = 1.0
        else:
            spec = active_specs[idx]
            if spec.goal == "max":
                utility = (raw_value - low) / span
            else:
                utility = (high - raw_value) / span
        target_values.append(min(1.0, max(0.0, utility)))
    return np.array(target_values, dtype=float)


def _resolve_limit_value(
    candidate: ParetoCandidate,
    entry: Mapping[str, Any],
    aggregate_cfg: Mapping[str, Any] | None = None,
) -> Optional[float]:
    metric = str(entry.get("metric", "")).strip()
    if not metric:
        return None
    stat = resolve_limit_stat(
        dict(entry),
        aggregate_cfg=dict(aggregate_cfg) if aggregate_cfg else None,
    )
    if "stat" not in entry:
        value = resolve_metric_value(candidate.aggregated_values, metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    key = f"{metric}_{stat}"
    value = resolve_metric_value(candidate.stats_flat, key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if "stat" not in entry:
        fallback = _resolve_candidate_metric_value(candidate, metric)
        if isinstance(fallback, (int, float)) and math.isfinite(float(fallback)):
            return float(fallback)
    return None


def _limit_rejects(entry: Mapping[str, Any], value: float) -> bool:
    mode = str(entry.get("penalize_if", "greater_than")).strip().lower()
    if mode == "greater_than":
        return value > float(entry["value"])
    if mode == "greater_than_or_equal":
        return value >= float(entry["value"])
    if mode == "less_than":
        return value < float(entry["value"])
    if mode == "less_than_or_equal":
        return value <= float(entry["value"])
    if mode == "equal_to":
        return value == float(entry["value"])
    if mode == "not_equal":
        return value != float(entry["value"])
    if mode == "outside_range":
        low, high = entry["range"]
        return value < float(low) or value > float(high)
    if mode == "inside_range":
        low, high = entry["range"]
        return float(low) <= value <= float(high)
    raise ValueError(f"Unsupported limit mode {mode!r}")


def filter_candidates(
    candidates: Sequence[ParetoCandidate],
    *,
    limits_payload: Optional[str],
    limit_entries: Optional[Sequence[str]],
) -> tuple[List[ParetoCandidate], List[Dict[str, Any]]]:
    normalized_limits: List[Dict[str, Any]] = []
    if limits_payload is not None:
        normalized_limits.extend(normalize_limit_entries(limits_payload))
    if limit_entries:
        normalized_limits.extend(parse_limit_cli_entries(list(limit_entries)))
    normalized_limits = normalize_limit_entries(normalized_limits)
    enabled_limits = [entry for entry in normalized_limits if bool(entry.get("enabled", True))]
    if not enabled_limits:
        return list(candidates), enabled_limits
    aggregate_cfg = None
    if candidates:
        backtest_cfg = candidates[0].entry.get("backtest")
        if isinstance(backtest_cfg, Mapping):
            aggregate_cfg = backtest_cfg.get("aggregate")

    filtered: List[ParetoCandidate] = []
    for candidate in candidates:
        rejected = False
        for entry in enabled_limits:
            value = _resolve_limit_value(candidate, entry, aggregate_cfg)
            if value is None:
                continue
            if _limit_rejects(entry, value):
                rejected = True
                break
        if not rejected:
            filtered.append(candidate)
    return filtered, enabled_limits


def _select_knee(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
) -> SelectionResult:
    n_candidates, n_obj = utilities.shape
    if n_candidates == 1:
        idx = 0
        score = 0.0
        scores = np.array([score], dtype=float)
        mode = "single_candidate"
        anchor_files: list[str] = [candidates[0].path.name]
    else:
        anchor_indices = [int(np.argmax(utilities[:, j])) for j in range(n_obj)]
        unique_anchor_indices = list(dict.fromkeys(anchor_indices))
        anchors = utilities[unique_anchor_indices]
        if anchors.shape[0] >= 2 and np.linalg.matrix_rank((anchors[1:] - anchors[:1]).T) >= 1:
            base = anchors[0]
            basis = (anchors[1:] - base).T
            scores = np.zeros(n_candidates, dtype=float)
            for cand_idx in range(n_candidates):
                vec = utilities[cand_idx] - base
                coeffs, *_ = np.linalg.lstsq(basis, vec, rcond=None)
                projection = basis @ coeffs
                scores[cand_idx] = float(np.linalg.norm(vec - projection))
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            mode = "hyperplane_distance"
        else:
            scores = utilities.min(axis=1)
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            mode = "maximin_fallback"
        anchor_files = [candidates[i].path.name for i in unique_anchor_indices]
    ranking_order = list(np.argsort(-scores))
    candidate = candidates[idx]
    return SelectionResult(
        candidate=candidate,
        method="knee",
        score=score,
        objective_values={
            metric: float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics
        },
        details={
            "active_metrics": list(active_metrics),
            "selected_utilities": {metric: float(utilities[idx, j]) for j, metric in enumerate(active_metrics)},
            "minimum_selected_utility": float(np.min(utilities[idx])) if utilities.size else 0.0,
            "knee_mode": mode,
            "anchor_files": anchor_files,
            "ranking_order": ranking_order,
            "score_vector": [float(x) for x in scores],
        },
    )


def _select_ideal_like(
    method: str,
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
    weights: np.ndarray,
    target_vector: np.ndarray,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> SelectionResult:
    weighted_sq = ((utilities - target_vector) ** 2) * weights
    distances = np.sqrt(weighted_sq.sum(axis=1))
    idx = int(np.argmin(distances))
    candidate = candidates[idx]
    payload = {} if details is None else dict(details)
    payload["active_metrics"] = list(active_metrics)
    payload["selected_utilities"] = {
        metric: float(utilities[idx, j]) for j, metric in enumerate(active_metrics)
    }
    payload["target_utilities"] = {metric: float(target_vector[j]) for j, metric in enumerate(active_metrics)}
    payload["distance_components"] = {
        metric: float(weighted_sq[idx, j]) for j, metric in enumerate(active_metrics)
    }
    payload["ranking_order"] = list(np.argsort(distances))
    payload["score_vector"] = [float(-d) for d in distances]
    return SelectionResult(
        candidate=candidate,
        method=method,
        score=float(-distances[idx]),
        objective_values={
            metric: float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics
        },
        details=payload,
    )


def _select_utility(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
    weights: np.ndarray,
) -> SelectionResult:
    scores = utilities @ weights
    idx = int(np.argmax(scores))
    candidate = candidates[idx]
    return SelectionResult(
        candidate=candidate,
        method="utility",
        score=float(scores[idx]),
        objective_values={
            metric: float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics
        },
        details={
            "active_metrics": list(active_metrics),
            "weights": dict(zip(active_metrics, [float(x) for x in weights])),
            "selected_utilities": {metric: float(utilities[idx, j]) for j, metric in enumerate(active_metrics)},
            "utility_contributions": {
                metric: float(utilities[idx, j] * weights[j]) for j, metric in enumerate(active_metrics)
            },
            "ranking_order": list(np.argsort(-scores)),
            "score_vector": [float(x) for x in scores],
        },
    )


def _select_lexicographic(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
) -> SelectionResult:
    best_idx = 0
    best_key = tuple(float(utilities[0, j]) for j in range(utilities.shape[1]))
    sort_keys: list[tuple[float, ...]] = [best_key]
    for idx in range(1, len(candidates)):
        key = tuple(float(utilities[idx, j]) for j in range(utilities.shape[1]))
        sort_keys.append(key)
        if key > best_key:
            best_idx = idx
            best_key = key
    ranking_order = sorted(range(len(candidates)), key=lambda i: sort_keys[i], reverse=True)
    candidate = candidates[best_idx]
    tie_break_score = 0.0
    if utilities.shape[1] > 0:
        tie_break_score = float(sum(best_key[j] / (10 ** j) for j in range(len(best_key))))
    return SelectionResult(
        candidate=candidate,
        method="lexicographic",
        score=tie_break_score,
        objective_values={
            metric: float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics
        },
        details={
            "active_metrics": list(active_metrics),
            "priority": list(active_metrics),
            "selected_utilities": {metric: float(utilities[best_idx, j]) for j, metric in enumerate(active_metrics)},
            "ranking_order": ranking_order,
            "score_vector": [
                float(sum(sort_keys[i][j] / (10 ** j) for j in range(len(sort_keys[i]))))
                for i in range(len(candidates))
            ],
        },
    )


def _select_outranking(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
    weights: np.ndarray,
) -> SelectionResult:
    n = len(candidates)
    if n == 1:
        idx = 0
        net_flows = np.array([0.0], dtype=float)
    else:
        net_flows = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pref_ij = float(np.sum(weights * np.maximum(utilities[i] - utilities[j], 0.0)))
                pref_ji = float(np.sum(weights * np.maximum(utilities[j] - utilities[i], 0.0)))
                net_flows[i] += pref_ij - pref_ji
            net_flows[i] /= max(1, n - 1)
        idx = int(np.argmax(net_flows))
    candidate = candidates[idx]
    return SelectionResult(
        candidate=candidate,
        method="outranking",
        score=float(net_flows[idx]),
        objective_values={
            metric: float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics
        },
        details={
            "active_metrics": list(active_metrics),
            "weights": dict(zip(active_metrics, [float(x) for x in weights])),
            "selected_utilities": {metric: float(utilities[idx, j]) for j, metric in enumerate(active_metrics)},
            "ranking_order": list(np.argsort(-net_flows)),
            "score_vector": [float(x) for x in net_flows],
        },
    )


def select_candidate(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[ObjectiveSpec],
    *,
    method: str,
    objectives_arg: Optional[str] = None,
    weight_pairs: Optional[Sequence[str]] = None,
    target_pairs: Optional[Sequence[str]] = None,
    priority_arg: Optional[str] = None,
) -> SelectionResult:
    if not candidates:
        raise ValueError("No Pareto candidates available for selection.")

    normalized_method = parse_method_name(method)
    target_map = _parse_key_value_pairs(target_pairs or [], value_name="target") if target_pairs else {}
    active_specs = _resolve_active_objective_metrics(
        scoring_specs,
        candidates,
        objectives_arg=objectives_arg,
        priority_arg=priority_arg if normalized_method == "lexicographic" else None,
        target_map=target_map,
        method=normalized_method,
    )
    active_metrics = [spec.metric for spec in active_specs]
    utilities, lows, highs = _normalize_objective_matrix(candidates, active_specs)
    weight_map = _parse_key_value_pairs(weight_pairs or [], value_name="weight") if weight_pairs else {}
    weights = _weights_for_metrics(active_metrics, weight_map)

    if normalized_method == "knee":
        result = _select_knee(candidates, active_metrics, utilities)
        return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)
    if normalized_method == "reference":
        if not target_map:
            raise ValueError("Method 'reference' requires at least one --target metric=value.")
        target_vector = _normalize_reference_targets(target_map, active_specs, lows, highs)
        result = _select_ideal_like(
            "reference",
            candidates,
            active_metrics,
            utilities,
            weights,
            target_vector,
            details={"targets": target_map, "weights": dict(zip(active_metrics, weights))},
        )
        return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)
    if normalized_method == "ideal":
        target_vector = np.ones(len(active_metrics), dtype=float)
        result = _select_ideal_like(
            "ideal",
            candidates,
            active_metrics,
            utilities,
            weights,
            target_vector,
            details={"weights": dict(zip(active_metrics, weights))},
        )
        return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)
    if normalized_method == "utility":
        result = _select_utility(candidates, active_metrics, utilities, weights)
        return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)
    if normalized_method == "lexicographic":
        result = _select_lexicographic(candidates, active_metrics, utilities)
        return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)
    if normalized_method == "outranking":
        result = _select_outranking(candidates, active_metrics, utilities, weights)
        return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)
    raise ValueError(f"Unsupported selection method {normalized_method!r}")


def format_selection_result(
    pareto_dir: Path,
    *,
    candidates: Sequence[ParetoCandidate],
    loaded_count: int,
    retained_count: int,
    active_limits: Sequence[Dict[str, Any]],
    result: SelectionResult,
    show_top: int = 1,
) -> str:
    selected_filename = result.candidate.path.name
    score_label, score_value = _score_label_and_value(result)
    selected_display_path = _display_path(result.candidate.path)
    backtest_command = f"passivbot backtest {selected_display_path}"
    lines: List[str] = []
    lines.extend(
        _render_key_value_box(
            [
                ("Pareto directory", _display_path(pareto_dir)),
                ("Loaded candidates", str(loaded_count)),
                ("Retained after limits", str(retained_count)),
                ("Applied limits", str(len(active_limits))),
                ("Method", result.method),
                (score_label, score_value),
                ("Selected file", selected_filename),
                ("Selected path", selected_display_path),
            ]
        )
    )
    lines.append(f"Backtest command: {backtest_command}")
    lines.append(f"Method summary: {_method_explanation(result.method)}")
    active_metrics = result.details.get("active_metrics")
    spec_map = objective_spec_by_metric(result.candidate.entry)
    if isinstance(active_metrics, list) and active_metrics:
        lines.append("Active objectives:")
        objective_rows = []
        for metric in active_metrics:
            goal = spec_map.get(metric).goal if metric in spec_map else (default_objective_goal(metric) or "?")
            objective_rows.append([str(metric), str(goal)])
        lines.extend(_render_table(["metric", "goal"], objective_rows))
    if active_limits:
        lines.append("Limit filters:")
        for entry in active_limits:
            lines.append(f"  - {_format_limit_entry(entry)}")
    weights = result.details.get("weights")
    targets = result.details.get("targets")
    if isinstance(targets, Mapping) and targets:
        lines.append("Reference targets:")
        for metric, value in targets.items():
            lines.append(f"  - {metric} = {_format_metric_value(value)}")
    priority = result.details.get("priority")
    if isinstance(priority, list) and priority:
        lines.append(f"Priority order: {', '.join(str(metric) for metric in priority)}")
    knee_mode = result.details.get("knee_mode")
    if isinstance(knee_mode, str) and knee_mode:
        lines.append(f"Knee mode: {knee_mode}")
    anchor_files = result.details.get("anchor_files")
    if isinstance(anchor_files, list) and anchor_files:
        lines.append(
            f"Anchor files ({len(anchor_files)}): {_summarize_anchor_files([str(item) for item in anchor_files])}"
        )
    lines.extend(_selection_rationale_lines(result))
    selected_utilities = result.details.get("selected_utilities")
    utility_contributions = result.details.get("utility_contributions")
    target_utilities = result.details.get("target_utilities")
    distance_components = result.details.get("distance_components")
    ideal_point = result.details.get("ideal_point")
    if "minimum_selected_utility" in result.details:
        lines.append(
            f"Minimum selected utility: {float(result.details['minimum_selected_utility']):.6f}"
        )
    if isinstance(selected_utilities, Mapping) and selected_utilities:
        metric_rows: List[List[str]] = []
        for metric, value in result.objective_values.items():
            goal = spec_map.get(metric).goal if metric in spec_map else (default_objective_goal(metric) or "?")
            row = [
                metric,
                str(goal),
                _format_metric_value(value),
                _format_metric_value(selected_utilities.get(metric, "")),
                _format_metric_value(ideal_point.get(metric, "")) if isinstance(ideal_point, Mapping) else "",
            ]
            if isinstance(weights, Mapping) and weights:
                row.append(_format_metric_value(weights.get(metric, "")))
            if isinstance(distance_components, Mapping) and distance_components:
                row.append(_format_metric_value(distance_components.get(metric, "")))
            if isinstance(utility_contributions, Mapping) and utility_contributions:
                row.append(_format_metric_value(utility_contributions.get(metric, "")))
            if isinstance(targets, Mapping) and targets:
                row.append(_format_metric_value(targets.get(metric, "")))
            elif isinstance(target_utilities, Mapping) and target_utilities and result.method != "ideal":
                row.append(_format_metric_value(target_utilities.get(metric, "")))
            metric_rows.append(row)

        headers = ["metric", "goal", "value", "utility", "ideal"]
        if isinstance(weights, Mapping) and weights:
            headers.append("weight")
        if isinstance(distance_components, Mapping) and distance_components:
            headers.append("distance")
        if isinstance(utility_contributions, Mapping) and utility_contributions:
            headers.append("contrib")
        if isinstance(targets, Mapping) and targets:
            headers.append("target")
        elif isinstance(target_utilities, Mapping) and target_utilities and result.method != "ideal":
            headers.append("target_u")
        lines.extend(["", "Objective table:"])
        lines.extend(_render_table(headers, metric_rows))
    else:
        lines.extend(["", "Objectives:"])
        for metric, value in result.objective_values.items():
            goal = spec_map.get(metric).goal if metric in spec_map else (default_objective_goal(metric) or "?")
            lines.append(f"  {metric} ({goal}): {value}")
    ranking_order = result.details.get("ranking_order")
    score_vector = result.details.get("score_vector")
    if (
        isinstance(ranking_order, list)
        and isinstance(score_vector, list)
        and show_top > 1
        and len(ranking_order) > 1
    ):
        lines.extend(["", "Top candidates:"])
        shortlist = _ranked_rows(
            candidates,
            list(active_metrics) if isinstance(active_metrics, list) else list(result.objective_values),
            ranking_order,
            score_vector,
            limit=show_top,
        )
        top_rows = [
            [
                f"#{row['rank']}",
                f"{row['score']:.6f}",
                str(row["file"]),
                str(row["hash"]),
            ]
            for row in shortlist
        ]
        lines.extend(_render_table(["rank", "score", "file", "hash"], top_rows))
    return "\n".join(lines)


def run_from_args(args: argparse.Namespace) -> SelectionResult:
    method = parse_method_name(args.method)
    raw_path = getattr(args, "path", None)
    if not raw_path:
        latest = detect_latest_pareto_dir()
        if latest is None:
            raise FileNotFoundError(
                "No pareto path provided and no optimize_results/.../pareto directory was found."
            )
        raw_path = str(latest)
    pareto_dir, candidates, scoring_specs = load_candidates(raw_path)
    filtered_candidates, active_limits = filter_candidates(
        candidates,
        limits_payload=getattr(args, "limits_payload", None),
        limit_entries=list(getattr(args, "limit_entries", []) or []),
    )
    if not filtered_candidates:
        raise ValueError("No Pareto candidates remained after applying limits.")
    result = select_candidate(
        filtered_candidates,
        scoring_specs,
        method=method,
        objectives_arg=getattr(args, "objectives", None),
        weight_pairs=getattr(args, "weight", None),
        target_pairs=getattr(args, "target", None),
        priority_arg=getattr(args, "priority", None),
    )
    show_top = max(1, int(getattr(args, "show_top", 1) or 1))
    if getattr(args, "json_output", False):
        ranking_order = result.details.get("ranking_order") or [filtered_candidates.index(result.candidate)]
        score_vector = result.details.get("score_vector") or [result.score] * len(filtered_candidates)
        active_metrics = result.details.get("active_metrics") or list(result.objective_values)
        payload = {
            "pareto_dir": str(pareto_dir),
            "loaded_count": len(candidates),
            "retained_count": len(filtered_candidates),
            "applied_limits": _json_ready(active_limits),
            "method": result.method,
            "method_description": _method_explanation(result.method),
            "selected": {
                "file": result.candidate.path.name,
                "hash": result.candidate.path.stem,
                "path": str(result.candidate.path),
                "score": float(result.score),
                "objectives": _json_ready(result.objective_values),
                "details": _public_selection_details(result.details),
            },
            "top_candidates": _json_ready(
                _ranked_rows(
                    filtered_candidates,
                    active_metrics,
                    ranking_order,
                    score_vector,
                    limit=show_top,
                )
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            format_selection_result(
                pareto_dir,
                candidates=filtered_candidates,
                loaded_count=len(candidates),
                retained_count=len(filtered_candidates),
                active_limits=active_limits,
                result=result,
                show_top=show_top,
            )
        )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)
    return 0
