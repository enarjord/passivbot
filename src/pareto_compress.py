from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from pareto_core import detect_latest_pareto_dir
from pareto_explorer import (
    ParetoCandidate,
    _display_path,
    _format_limit_entry,
    _format_metric_value,
    _normalize_objective_matrix,
    _resolve_active_objective_metrics,
    _resolve_candidate_metric_value,
    _render_key_value_box,
    _render_table,
    filter_candidates,
    load_candidates,
)


DEFAULT_METHOD = "anchors-farthest"
METHOD_ALIASES = {
    "anchors-farthest": "anchors-farthest",
    "anchors_farthest": "anchors-farthest",
    "anchor-farthest": "anchors-farthest",
    "anchor_farthest": "anchors-farthest",
    "diverse": "anchors-farthest",
    "diversity": "anchors-farthest",
}


@dataclass(frozen=True)
class CompressedMember:
    candidate: ParetoCandidate
    selection_rank: int
    reason: str
    reason_details: List[str]
    utilities: Dict[str, float]
    objectives: Dict[str, float]
    mean_utility: float
    min_distance_to_previous: Optional[float]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _parse_method(raw: str | None) -> str:
    method = METHOD_ALIASES.get(str(raw or DEFAULT_METHOD).strip().lower())
    if method is None:
        allowed = ", ".join(sorted(dict.fromkeys(METHOD_ALIASES.values())))
        raise ValueError(f"Unknown compression method {raw!r}; expected one of: {allowed}")
    return method


def _parse_path_and_count(raw_args: Sequence[str]) -> tuple[Optional[str], int]:
    if len(raw_args) == 1:
        raw_path = None
        raw_count = raw_args[0]
    elif len(raw_args) == 2:
        raw_path = raw_args[0]
        raw_count = raw_args[1]
    else:
        raise ValueError("Expected COUNT or PATH COUNT.")
    try:
        count = int(raw_count)
    except Exception as exc:
        raise ValueError(f"Invalid member count {raw_count!r}; expected a positive integer.") from exc
    if count <= 0:
        raise ValueError("Member count must be positive.")
    return raw_path, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto-compress",
        description=(
            "Select a compact representative subset from a Pareto front without mutating "
            "the source directory."
        ),
        epilog=(
            "Examples:\n"
            "  passivbot tool pareto-compress 8\n"
            "  passivbot tool pareto-compress optimize_results/run/pareto 8\n"
            "  passivbot tool pareto-compress optimize_results/run 8 --output-dir selected_pareto_8\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "path_count",
        nargs="+",
        metavar="PATH_OR_COUNT",
        help=(
            "COUNT, or PATH COUNT. PATH may be a pareto directory or optimization run directory. "
            "If PATH is omitted, uses the latest optimize_results/<run>/pareto with JSON candidates."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        default=DEFAULT_METHOD,
        help="Compression method. Default: anchors-farthest.",
    )
    parser.add_argument(
        "-o",
        "--objectives",
        default=None,
        help="Optional comma-separated subset of objective metrics to use for compression.",
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
        "--output-dir",
        type=str,
        default=None,
        help="Copy selected JSON members and write selection.json to this directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser


def _candidate_sort_key(candidates: Sequence[ParetoCandidate], utilities: np.ndarray, idx: int) -> tuple[float, str]:
    mean_utility = float(np.mean(utilities[idx])) if utilities.size else 0.0
    return mean_utility, candidates[idx].path.name


def _best_anchor_index(
    candidates: Sequence[ParetoCandidate],
    utilities: np.ndarray,
    objective_idx: int,
) -> int:
    values = utilities[:, objective_idx]
    best_value = float(np.max(values))
    tied = [int(idx) for idx in np.flatnonzero(np.isclose(values, best_value))]
    return max(tied, key=lambda idx: _candidate_sort_key(candidates, utilities, idx))


def _ideal_anchor_index(candidates: Sequence[ParetoCandidate], utilities: np.ndarray) -> int:
    target = np.ones(utilities.shape[1], dtype=float)
    distances = np.linalg.norm(utilities - target, axis=1)
    best_distance = float(np.min(distances))
    tied = [int(idx) for idx in np.flatnonzero(np.isclose(distances, best_distance))]
    return max(tied, key=lambda idx: _candidate_sort_key(candidates, utilities, idx))


def _select_anchor_indices(
    candidates: Sequence[ParetoCandidate],
    utilities: np.ndarray,
    active_metrics: Sequence[str],
    count: int,
    *,
    selected: Optional[List[int]] = None,
    reasons: Optional[Dict[int, List[str]]] = None,
) -> tuple[List[int], Dict[int, List[str]], List[str]]:
    selected = list(selected or [])
    reasons = {idx: list(items) for idx, items in (reasons or {}).items()}
    truncated: List[str] = []
    for objective_idx, metric in enumerate(active_metrics):
        idx = _best_anchor_index(candidates, utilities, objective_idx)
        reason = f"best {metric}"
        if idx in reasons:
            reasons[idx].append(reason)
            continue
        if len(selected) >= count:
            truncated.append(reason)
            continue
        selected.append(idx)
        reasons[idx] = [reason]
    return selected, reasons, truncated


def _select_diverse_indices(
    candidates: Sequence[ParetoCandidate],
    utilities: np.ndarray,
    selected: List[int],
    reasons: Dict[int, List[str]],
    count: int,
) -> tuple[List[int], Dict[int, List[str]], Dict[int, float]]:
    min_distances: Dict[int, float] = {idx: math.inf for idx in selected}
    if count <= len(selected):
        return selected[:count], reasons, min_distances
    if not selected:
        first = max(range(len(candidates)), key=lambda idx: _candidate_sort_key(candidates, utilities, idx))
        selected.append(first)
        reasons[first] = ["highest mean objective utility"]
        min_distances[first] = math.inf

    selected_set = set(selected)
    while len(selected) < count and len(selected_set) < len(candidates):
        selected_points = utilities[selected]
        distances = np.linalg.norm(utilities[:, None, :] - selected_points[None, :, :], axis=2)
        nearest = distances.min(axis=1)
        for idx in selected_set:
            nearest[idx] = -math.inf
        best_distance = float(np.max(nearest))
        tied = [int(idx) for idx in np.flatnonzero(np.isclose(nearest, best_distance))]
        idx = max(tied, key=lambda item: _candidate_sort_key(candidates, utilities, item))
        selected.append(idx)
        selected_set.add(idx)
        reasons[idx] = ["farthest from selected members in normalized objective space"]
        min_distances[idx] = best_distance
    return selected, reasons, min_distances


def compress_candidates(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[Any],
    *,
    count: int,
    objectives_arg: Optional[str] = None,
    method: str = DEFAULT_METHOD,
) -> tuple[List[CompressedMember], List[Dict[str, Any]], List[str]]:
    if not candidates:
        raise ValueError("No Pareto candidates available for compression.")
    if count <= 0:
        raise ValueError("Member count must be positive.")
    normalized_method = _parse_method(method)
    if normalized_method != "anchors-farthest":
        raise ValueError(f"Unsupported compression method {normalized_method!r}")

    active_specs = _resolve_active_objective_metrics(
        scoring_specs,
        candidates,
        objectives_arg=objectives_arg,
        priority_arg=None,
        target_map=None,
        method="ideal",
    )
    if not active_specs:
        raise ValueError("No objective metrics available for compression.")
    active_metrics = [spec.metric for spec in active_specs]
    utilities, lows, highs = _normalize_objective_matrix(candidates, active_specs)
    effective_count = min(int(count), len(candidates))

    ideal_idx = _ideal_anchor_index(candidates, utilities)
    selected = [ideal_idx]
    reasons = {ideal_idx: ["closest to ideal point"]}
    selected, reasons, truncated_anchors = _select_anchor_indices(
        candidates,
        utilities,
        active_metrics,
        effective_count,
        selected=selected,
        reasons=reasons,
    )
    selected, reasons, min_distances = _select_diverse_indices(
        candidates,
        utilities,
        selected,
        reasons,
        effective_count,
    )

    members: List[CompressedMember] = []
    for rank, idx in enumerate(selected, start=1):
        candidate = candidates[idx]
        member_objectives = {
            metric: float(_resolve_candidate_metric_value(candidate, metric)) for metric in active_metrics
        }
        member_utilities = {metric: float(utilities[idx, col]) for col, metric in enumerate(active_metrics)}
        member_reasons = reasons.get(idx, [])
        if "closest to ideal point" in member_reasons:
            reason = "ideal_anchor"
        elif len(member_reasons) == 1 and member_reasons[0].startswith("best "):
            reason = "objective_anchor"
        elif any(item.startswith("best ") for item in member_reasons):
            reason = "objective_anchor"
        elif member_reasons == ["highest mean objective utility"]:
            reason = "utility_anchor"
        else:
            reason = "diversity_fill"
        members.append(
            CompressedMember(
                candidate=candidate,
                selection_rank=rank,
                reason=reason,
                reason_details=member_reasons,
                utilities=member_utilities,
                objectives=member_objectives,
                mean_utility=float(np.mean(utilities[idx])),
                min_distance_to_previous=min_distances.get(idx),
            )
        )

    objective_ranges = [
        {
            "metric": spec.metric,
            "goal": spec.goal,
            "min": float(lows[idx]),
            "max": float(highs[idx]),
        }
        for idx, spec in enumerate(active_specs)
    ]
    return members, objective_ranges, truncated_anchors


def _member_to_dict(member: CompressedMember) -> Dict[str, Any]:
    return {
        "selection_rank": member.selection_rank,
        "file": member.candidate.path.name,
        "hash": member.candidate.path.stem,
        "path": str(member.candidate.path),
        "reason": member.reason,
        "reason_details": list(member.reason_details),
        "objectives": dict(member.objectives),
        "utilities": dict(member.utilities),
        "mean_utility": float(member.mean_utility),
        "min_distance_to_previous": (
            None
            if member.min_distance_to_previous is None or math.isinf(float(member.min_distance_to_previous))
            else float(member.min_distance_to_previous)
        ),
    }


def _write_outputs(output_dir: Path, pareto_dir: Path, members: Sequence[CompressedMember], payload: Mapping[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir == pareto_dir.resolve():
        raise ValueError("Refusing to write compressed output into the source Pareto directory.")
    expected_files = {member.candidate.path.name for member in members}
    expected_files.add("selection.json")
    if output_dir.exists():
        unexpected_files = sorted(
            path.name for path in output_dir.iterdir() if path.is_file() and path.name not in expected_files
        )
        if unexpected_files:
            preview = ", ".join(unexpected_files[:5])
            suffix = "" if len(unexpected_files) <= 5 else f", ... (+{len(unexpected_files) - 5} more)"
            raise ValueError(
                "Output directory contains files not produced by this selection. "
                f"Use an empty output directory or remove stale files first: {preview}{suffix}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for member in members:
        destination = output_dir / member.candidate.path.name
        if destination.resolve() != member.candidate.path.resolve():
            shutil.copy2(member.candidate.path, destination)
    (output_dir / "selection.json").write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def compress_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    raw_path, count = _parse_path_and_count(getattr(args, "path_count", []))
    if not raw_path:
        latest = detect_latest_pareto_dir()
        if latest is None:
            raise FileNotFoundError(
                "No pareto path provided and no valid optimize_results/<run>/pareto directory "
                "with at least one *.json candidate was found."
            )
        raw_path = str(latest)

    pareto_dir, candidates, scoring_specs = load_candidates(raw_path)
    filtered, active_limits = filter_candidates(
        candidates,
        limits_payload=getattr(args, "limits_payload", None),
        limit_entries=list(getattr(args, "limit_entries", []) or []),
    )
    if not filtered:
        raise ValueError("No Pareto candidates remained after applying limits.")

    members, objective_ranges, truncated_anchors = compress_candidates(
        filtered,
        scoring_specs,
        count=count,
        objectives_arg=getattr(args, "objectives", None),
        method=getattr(args, "method", DEFAULT_METHOD),
    )

    output_dir_arg = getattr(args, "output_dir", None)
    output_dir = Path(output_dir_arg).expanduser() if output_dir_arg else None
    payload: Dict[str, Any] = {
        "pareto_dir": str(pareto_dir),
        "loaded_count": len(candidates),
        "retained_count": len(filtered),
        "requested_count": count,
        "selected_count": len(members),
        "method": _parse_method(getattr(args, "method", DEFAULT_METHOD)),
        "applied_limits": active_limits,
        "objectives": objective_ranges,
        "truncated_anchor_reasons": truncated_anchors,
        "selected": [_member_to_dict(member) for member in members],
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    if output_dir is not None:
        for item in payload["selected"]:
            item["output_path"] = str(output_dir / str(item["file"]))
        _write_outputs(output_dir, pareto_dir, members, payload)
    return payload


def format_compression(payload: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "Pareto Compression",
    ]
    lines.extend(
        _render_key_value_box(
            [
                ("Pareto directory", _display_path(str(payload["pareto_dir"]))),
                ("Loaded candidates", str(payload["loaded_count"])),
                ("Retained after limits", str(payload["retained_count"])),
                ("Requested members", str(payload["requested_count"])),
                ("Selected members", str(payload["selected_count"])),
                ("Method", str(payload["method"])),
            ]
        )
    )
    if payload.get("output_dir"):
        lines.append(f"Output dir: {_display_path(str(payload['output_dir']))}")
    if payload.get("applied_limits"):
        lines.append("Limit filters:")
        for entry in payload["applied_limits"]:
            lines.append(f"  - {_format_limit_entry(entry)}")
    if payload.get("truncated_anchor_reasons"):
        lines.append("Truncated anchors: " + ", ".join(str(item) for item in payload["truncated_anchor_reasons"]))

    objective_rows = [
        [
            str(item["metric"]),
            str(item["goal"]),
            _format_metric_value(item["min"]),
            _format_metric_value(item["max"]),
        ]
        for item in payload.get("objectives", [])
    ]
    if objective_rows:
        lines.extend(["", "Objective ranges:"])
        lines.extend(_render_table(["metric", "goal", "min", "max"], objective_rows))

    selected_rows = []
    for item in payload.get("selected", []):
        distance = item.get("min_distance_to_previous")
        selected_rows.append(
            [
                f"#{item['selection_rank']}",
                str(item["reason"]),
                _format_metric_value(float(item["mean_utility"])),
                "" if distance is None else _format_metric_value(float(distance)),
                str(item["file"]),
            ]
        )
    if selected_rows:
        lines.extend(["", "Selected members:"])
        lines.extend(_render_table(["rank", "reason", "mean_u", "dist", "file"], selected_rows))

    lines.append("")
    lines.append("Backtest commands:")
    for item in payload.get("selected", []):
        path = item.get("output_path") or item["path"]
        lines.append(f"  passivbot backtest {_display_path(str(path))}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = compress_from_args(args)
    if getattr(args, "json_output", False):
        print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    else:
        print(format_compression(payload))
    return 0
