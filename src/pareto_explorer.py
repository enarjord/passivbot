from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from config.limits import (
    normalize_limit_entries,
    parse_limit_cli_entries,
    resolve_reducer_mode,
    resolve_limit_basis,
)
from config.metrics import (
    CORE_METRIC_CATALOG,
    CORE_METRIC_DEFINITION_VERSION,
    resolve_metric_value,
)
from config.scoring import (
    ObjectiveSpec,
    default_objective_goal,
    default_scoring_weights,
    dominates_objectives,
    extract_objective_specs,
    objective_spec_by_metric,
)
from limit_utils import resolve_auto_limit_entries
from metrics_schema import (
    MetricAggregationError,
    copy_metric_value,
    decode_metric_value,
    encode_metric_values,
    require_metric_definition,
    flatten_metric_stats,
    raw_reduce,
)
from pareto_core import PARETO_FILENAME_PREFIX_LENGTH
from pareto_qualification import load_qualification, qualification_for_candidates


METHOD_ALIASES = {
    "knee": "knee",
    "k": "knee",
    "reference": "reference",
    "ref": "reference",
    "r": "reference",
    "aspiration": "aspiration",
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
    "aspiration": "Aspiration chooser. Minimizes the worst weighted shortfall from the targets, then the weighted signed shortfall.",
    "ideal": "Distance-to-ideal chooser. Picks the candidate closest to the observed ideal point.",
    "utility": "Weighted utility chooser. Picks the highest weighted normalized utility.",
    "lexicographic": "Strict priority chooser. Sorts by objective priority order.",
    "outranking": "Simplified PROMETHEE-style chooser based on pairwise net preference flow.",
}

WEIGHTED_METHODS = frozenset({"aspiration", "ideal", "reference", "utility", "outranking"})


FILTERED_SELECTION_MANIFEST = "selection.json"
SELECTION_RECORD_VERSION = 1


@dataclass(frozen=True)
class ParetoCandidate:
    path: Path
    entry: Dict[str, Any]
    objectives: Dict[str, float]
    stats_flat: Dict[str, float]
    aggregated_values: Dict[str, float]
    scenario: Optional[str] = None
    file_sha256: str | None = None
    source_filename: str | None = None

    @property
    def filename(self) -> str:
        return self.source_filename or self.path.name


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
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        result, extended = {}, {}
        for key, item in value.items():
            name = str(key)
            if isinstance(item, (float, np.floating)) and not math.isfinite(item):
                extended[name] = float(item)
            else:
                result[name] = _json_ready(item)
        encoded = encode_metric_values(extended)
        statuses = encoded.pop("metric_status", {})
        result.update(encoded)
        if statuses:
            result.setdefault("metric_status", {}).update(statuses)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _public_selection_details(details: Mapping[str, Any]) -> Dict[str, Any]:
    hidden = {"ranking_order", "score_vector", "secondary_score_vector", "sensitivity"}
    return {key: value for key, value in details.items() if key not in hidden}


def _method_explanation(method: str) -> str:
    return METHOD_DESCRIPTIONS.get(method, "").strip()


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


def _format_limit_entry(entry: Mapping[str, Any], *, include_scope: bool = False) -> str:
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
    if include_scope:
        scope = []
        if "reducer" in entry:
            scope.append(f"{entry['reducer']} reducer")
        if "scenario" in entry:
            scope.append(f"scenario {entry['scenario'] or 'all'}")
        if scope:
            metric += f" ({', '.join(scope)})"
    if mode in op_map and value is not None:
        return f"{metric} {op_map[mode]} {value}"
    if mode == "outside_range" and isinstance(range_value, Sequence) and len(range_value) == 2:
        return f"{metric} outside [{range_value[0]}, {range_value[1]}]"
    if mode == "inside_range" and isinstance(range_value, Sequence) and len(range_value) == 2:
        return f"{metric} inside [{range_value[0]}, {range_value[1]}]"
    return str(dict(entry))


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


def _score_label_and_value(result: SelectionResult) -> tuple[str, float]:
    if result.method in {"ideal", "reference"}:
        return "distance", abs(result.score)
    if result.method == "aspiration":
        return "worst shortfall", abs(result.score)
    labels = {"utility": "utility score", "outranking": "net flow", "knee": "knee score"}
    return labels[result.method], result.score


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


def _resolve_candidate_metric_value(
    candidate: ParetoCandidate, metric: str
) -> Optional[float]:
    value = candidate.objectives.get(metric)
    if value is not None:
        # This mapping contains already-decoded raw values, including infinities.
        return float(value)
    value = resolve_metric_value(candidate.aggregated_values, metric)
    if value is not None:
        return float(value)
    mean_key = f"{metric}_mean"
    value = resolve_metric_value(candidate.stats_flat, mean_key)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def parse_method_name(raw_method: str) -> str:
    method = METHOD_ALIASES.get(str(raw_method or "").strip().lower())
    if method is None:
        allowed = ", ".join(sorted(dict.fromkeys(METHOD_ALIASES.values())))
        raise ValueError(f"Unknown method {raw_method!r}; expected one of: {allowed}")
    return method


def add_selection_arguments(
    parser: argparse.ArgumentParser, *, objective_short: bool = False
) -> None:
    objective_flags = ("-o", "--objectives") if objective_short else ("--objectives",)
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="ideal",
        help="Selection method. Default: ideal.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Rebuild a scenario-specific sub-front from the saved suite Pareto members, "
            "then select from it using the named scenario's metric values."
        ),
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
        *objective_flags,
        type=str,
        default=None,
        help="Rebuild and normalize the front on these comma-separated objectives. May include stored non-scoring metrics with known min/max direction.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="METRIC=VALUE",
        help="Repeatable weight for ideal, utility, reference, aspiration, or outranking.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="METRIC=VALUE",
        help="Repeatable raw target for reference or aspiration.",
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
        metavar="N",
        help=(
            "Show N ranked candidates including the selected file. "
            "Default: 2 in terminal reports, 5 in JSON."
            if objective_short
            else "Include N ranked candidates in the selection record, counting the selected file. Default: 5."
        ),
    )
    parser.add_argument(
        "--scale",
        action="append",
        metavar="METRIC=VALUE",
        help="Meaningful positive change in raw metric units. Required for aspiration.",
    )
    parser.add_argument(
        "--sensitivity-weight-fraction",
        type=float,
        metavar="FRACTION",
        help="Transfer this share of total weight between pairs of active objectives. "
        "Default for weighted methods: 0.01. Use 0 to disable.",
    )
    parser.add_argument(
        "--sensitivity-target-scales",
        type=float,
        metavar="MULTIPLE",
        help="Move each aspiration target in either direction by this multiple of its scale.",
    )
    parser.add_argument(
        "--qualification",
        type=Path,
        metavar="FILE",
        help="Attach matching native CPU replay evidence from a qualification JSON sidecar.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto",
        description=(
            "Summarize a saved Pareto front and select a candidate using its scoring objectives "
            "unless you override them. CLI limits are opt-in."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Methods:\n"
            "  knee         Approximate balanced compromise selector.\n"
            "  reference    Closest to exact targets; uses the full post-limit pool.\n"
            "  aspiration   Smallest worst directional shortfall to targets in raw units.\n"
            "  ideal        Closest to the observed ideal point.\n"
            "  utility      Highest weighted normalized utility (--weight metric=value).\n"
            "  lexicographic Strict priority order (--priority metric_a,metric_b,...).\n"
            "  outranking   Simplified PROMETHEE-style net flow selector.\n\n"
            "Limits are applied before selection. Repeat -l/--limit for multiple keep-conditions:\n"
            "  -l 'cagr_fraction_strategy_equity>0.0'\n"
            "  -l 'drawdown_max_fraction_strategy_equity<=0.35'\n"
            "  --limits '[{\"metric\":\"drawdown_max_fraction_strategy_equity\",\"penalize_if\":\">\",\"value\":0.35}]'\n\n"
            "Fractional inputs use raw units (0.05 means 5%). Use --list-metrics for exact keys.\n"
            "Use an explicit path to revisit a run, or freeze the directory for repeatable selection.\n\n"
            "Outputs:\n"
            "  -s configs/selected.local.json          Copy the selected member.\n"
            "  -f optimize_results/filtered_pareto     Copy members retained after limits.\n"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help=(
            "Pareto directory, optimization run directory, or candidate JSON. Defaults to the "
            "latest populated optimize_results/<run>/pareto by directory name."
        ),
    )
    add_selection_arguments(parser, objective_short=True)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-s",
        "--save-selected",
        type=str,
        default=None,
        metavar="FILE",
        help="Copy the selected Pareto member to FILE.",
    )
    output_group.add_argument(
        "-f",
        "--save-filtered",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Copy every member retained after limits to DIR and write selection.json. "
            "Without limits, copy the full loaded set."
        ),
    )
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--front",
        action="store_true",
        help="Show active-front ranges and objective leaders without selecting a candidate.",
    )
    view_group.add_argument(
        "--list-metrics",
        action="store_true",
        help="List saved metric keys, raw units, directions, and value coverage.",
    )
    view_group.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser


def _extract_suite_metrics(entry, reducer_cfg=None, scenario_labels=None):
    from metrics_schema import (
        require_metric_definition,
        decode_metric_value,
        encode_metric_values,
        raw_reduce,
        MetricAggregationError,
    )

    suite = entry.get("suite_metrics")
    if not isinstance(suite, Mapping):
        return {}, {}
    require_metric_definition(suite)
    modes = (
        reducer_cfg
        if reducer_cfg is not None
        else entry.get("backtest", {}).get("reducer", {})
    )
    labels = tuple(scenario_labels) if scenario_labels is not None else None
    stats, aggregates = {}, {}
    for metric, payload in suite.get("metrics", {}).items():
        fields = payload.get("stats", {})
        mode = resolve_reducer_mode(metric, modes)
        if reducer_cfg is not None or labels is not None:
            scenarios = payload.get("scenarios", {})
            selected = (
                labels
                if labels is not None
                else tuple(suite.get("scenario_labels", ()))
            )
            try:
                if not selected:
                    raise MetricAggregationError("no scenario population available")
                fields = encode_metric_values(
                    {
                        mode: raw_reduce(
                            [
                                decode_metric_value(scenarios, label)
                                for label in selected
                            ],
                            mode,
                        )
                    }
                )
            except MetricAggregationError as error:
                fields = {
                    mode: None,
                    "metric_status": {
                        mode: {"status": error.status, "reason": str(error)}
                    },
                }
        stats[metric] = fields
        if reducer_cfg is None and labels is None and "aggregated" in payload:
            try:
                aggregates[metric] = decode_metric_value(payload, "aggregated")
            except MetricAggregationError:
                pass  # Unselected diagnostic remains represented by its stat status.
        elif mode in fields:
            try:
                aggregates[metric] = decode_metric_value(fields, mode)
            except MetricAggregationError:
                pass  # A selected lookup below still rejects its explicit stat status.
    return flatten_metric_stats(stats), encode_metric_values(aggregates)


def _scenario_metric_values(
    entry: Mapping[str, Any], scenario: str, *, source: Path
) -> Dict[str, float]:
    suite_metrics = entry.get("suite_metrics")
    if not isinstance(suite_metrics, Mapping):
        raise ValueError(
            f"Scenario {scenario!r} requires suite Pareto artifacts, but {source.name} "
            "has no suite_metrics payload."
        )
    require_metric_definition(suite_metrics)
    metrics = suite_metrics.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"Scenario {scenario!r} requires the current suite_metrics.metrics payload, "
            f"but {source.name} does not contain it."
        )

    values: Dict[str, float] = {}
    available_labels: set[str] = set()
    for metric, payload in metrics.items():
        if not isinstance(payload, Mapping):
            continue
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, Mapping):
            continue
        available_labels.update(
            str(label) for label in scenarios if label != "metric_status"
        )
        if scenario not in scenarios:
            continue
        copied = copy_metric_value(scenarios, scenario, output_name=str(metric))
        statuses = copied.pop("metric_status", {})
        values.update(copied)
        if statuses:
            values.setdefault("metric_status", {}).update(statuses)

    if not values:
        available = (
            ", ".join(sorted(available_labels)) if available_labels else "<none>"
        )
        raise ValueError(
            f"Unknown or unavailable scenario {scenario!r} in {source.name}. "
            f"Available scenarios: {available}"
        )
    return values



def _scenario_mean_values(values):
    means = {}
    for metric in values:
        if metric == "metric_status":
            continue
        copied = copy_metric_value(values, metric, output_name=f"{metric}_mean")
        statuses = copied.pop("metric_status", {})
        means.update(copied)
        if statuses:
            means.setdefault("metric_status", {}).update(statuses)
    return means

def project_candidates_to_scenario(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[ObjectiveSpec],
    scenario: str,
) -> List[ParetoCandidate]:
    label = str(scenario).strip()
    if not label:
        raise ValueError("--scenario requires a non-empty scenario label.")

    projected: List[ParetoCandidate] = []
    required_metrics = [spec.metric for spec in scoring_specs]
    for candidate in candidates:
        values = _scenario_metric_values(candidate.entry, label, source=candidate.path)
        objectives: Dict[str, float] = {}
        missing: List[str] = []
        for metric in required_metrics:
            if metric not in values:
                missing.append(metric)
            else:
                objectives[metric] = decode_metric_value(values, metric)
        if missing:
            raise ValueError(
                f"Scenario {label!r} is missing scoring metric(s) {missing} "
                f"in {candidate.filename}."
            )
        projected.append(
            ParetoCandidate(
                path=candidate.path,
                entry=candidate.entry,
                objectives=objectives,
                stats_flat=_scenario_mean_values(values),
                aggregated_values=values,
                scenario=label,
                file_sha256=candidate.file_sha256,
                source_filename=candidate.source_filename,
            )
        )
    return projected


def build_scenario_front(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[ObjectiveSpec],
) -> List[ParetoCandidate]:
    return build_objective_front(candidates, scoring_specs)


def build_objective_front(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[ObjectiveSpec],
) -> List[ParetoCandidate]:
    """Keep nondominated objective vectors, retaining the first member of a tie."""
    metrics = [spec.metric for spec in scoring_specs]
    front: List[ParetoCandidate] = []
    front_vectors: List[tuple[float, ...]] = []
    for candidate in candidates:
        vector = tuple(float(_resolve_candidate_metric_value(candidate, metric)) for metric in metrics)
        if vector in front_vectors:
            continue
        if any(
            dominates_objectives(existing, vector, scoring_specs)
            for existing in front_vectors
        ):
            continue
        dominated_indices = [
            idx
            for idx, existing in enumerate(front_vectors)
            if dominates_objectives(vector, existing, scoring_specs)
        ]
        for idx in reversed(dominated_indices):
            front.pop(idx)
            front_vectors.pop(idx)
        front.append(candidate)
        front_vectors.append(vector)
    return front


def _extract_objectives(entry):
    from metrics_schema import require_metric_definition
    from config.metrics import metric_stat_key

    metrics = entry.get("metrics") or {}
    require_metric_definition(metrics if metrics else entry.get("suite_metrics", {}))
    specs = extract_objective_specs(entry)
    raw = metrics.get("objectives", {})
    flat = flatten_metric_stats(metrics.get("stats", {}))
    suite_flat, aggregate = _extract_suite_metrics(entry)
    flat.update(suite_flat)
    result = {}
    for spec in specs:
        value = resolve_metric_value(raw, spec.metric)
        if value is None:
            value = resolve_metric_value(aggregate, spec.metric)
        if value is None:
            value = resolve_metric_value(
                flat, metric_stat_key(spec.metric, spec.reducer or "mean")
            )
        if value is None:
            raise ValueError(
                f"selected objective {spec.metric!r} has no current raw value; re-evaluate"
            )
        result[spec.metric] = value
    return result


class NoParetoCandidatesError(ValueError):
    """A front has no candidate entries; invalid candidate data raises separately."""


def load_candidates(path: str | os.PathLike[str]) -> tuple[Path, List[ParetoCandidate], List[ObjectiveSpec]]:
    raw_path = Path(path).expanduser()
    if raw_path.is_file():
        pareto_dir = raw_path.parent.resolve()
        json_paths = [pareto_dir / raw_path.name]
    else:
        pareto_dir = resolve_pareto_directory(raw_path)
        json_paths = sorted(pareto_dir.glob("*.json"))
    if not json_paths:
        raise NoParetoCandidatesError(f"No Pareto JSON files found in {pareto_dir}")

    candidates: List[ParetoCandidate] = []
    baseline_specs: Optional[List[ObjectiveSpec]] = None
    baseline_metrics: Optional[List[str]] = None

    for entry_path in json_paths:
        resolved_path = entry_path.resolve()
        contents = resolved_path.read_bytes()
        entry = json.loads(contents)
        if not isinstance(entry, Mapping):
            continue
        specs = extract_objective_specs(entry)
        if not specs:
            continue
        metrics = [spec.metric for spec in specs]
        if baseline_specs is None:
            baseline_specs = specs
            baseline_metrics = metrics
        elif specs != baseline_specs:
            raise ValueError(
                f"Inconsistent optimize.scoring in {entry_path}; expected "
                f"{[spec.to_config() for spec in baseline_specs]}, got "
                f"{[spec.to_config() for spec in specs]}"
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
                path=resolved_path,
                entry=entry,
                objectives=objectives,
                stats_flat=stats_flat,
                aggregated_values=aggregated_values,
                file_sha256=hashlib.sha256(contents).hexdigest(),
                source_filename=entry_path.name,
            )
        )

    if baseline_specs is None:
        raise NoParetoCandidatesError(
            f"No Pareto candidate JSON files found in {pareto_dir}; "
            "JSON artifacts without optimize.scoring were ignored."
        )
    return pareto_dir, candidates, baseline_specs


def load_selection_candidates(path: str | Path | None):
    """Load the explicit source or the latest front with actual candidate files."""
    if path:
        return load_candidates(path)
    root = Path("optimize_results")
    if root.is_dir():
        for run in sorted(root.iterdir(), key=lambda item: item.name, reverse=True):
            front = run / "pareto"
            if not front.is_dir():
                continue
            try:
                return load_candidates(front)
            except NoParetoCandidatesError:
                continue
    raise FileNotFoundError(
        "No Pareto path provided and no optimize_results/<run>/pareto directory "
        "with candidate JSON files was found."
    )


def default_pareto_goal(metric: str) -> str | None:
    """Return the direction for a requested saved metric without a scoring spec."""
    if metric == "peak_recovery_max_days_strategy_equity":
        return "min"
    return default_objective_goal(metric)


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
    elif method in {"reference", "aspiration"} and target_map:
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
        goal = default_pareto_goal(metric)
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
    if not resolved:
        raise ValueError("Choose at least one selection objective.")
    if len({spec.metric for spec in resolved}) != len(resolved):
        raise ValueError("Choose distinct selection objectives.")
    return resolved


def _raw_objective_matrix(
    candidates: Sequence[ParetoCandidate],
    active_specs: Sequence[ObjectiveSpec],
) -> np.ndarray:
    raw = np.array(
        [
            [
                _resolve_candidate_metric_value(candidate, spec.metric)
                for spec in active_specs
            ]
            for candidate in candidates
        ],
        dtype=float,
    )
    undefined = np.argwhere(np.isnan(raw))
    if undefined.size:
        row, col = undefined[0]
        raise ValueError(
            f"undefined objective {active_specs[col].metric!r} in {candidates[row].filename}"
        )
    return raw


def _normalize_objective_matrix(
    candidates: Sequence[ParetoCandidate],
    active_specs: Sequence[ObjectiveSpec],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = (2.0 / np.pi) * np.arctan(_raw_objective_matrix(candidates, active_specs))
    lows = raw.min(axis=0)
    highs = raw.max(axis=0)
    spans = highs - lows
    utilities = np.ones_like(raw, dtype=float)
    for idx, spec in enumerate(active_specs):
        span = spans[idx]
        if span <= 1e-15:
            utilities[:, idx] = 1.0
            continue
        if spec.goal == "max":
            utilities[:, idx] = (raw[:, idx] - lows[idx]) / span
        else:
            utilities[:, idx] = (highs[idx] - raw[:, idx]) / span
    return utilities, lows, highs


def _weights_for_metrics(metrics: Sequence[str], weight_map: Optional[Dict[str, float]]) -> np.ndarray:
    weights = np.array([float((weight_map or {}).get(metric, 1.0)) for metric in metrics], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and non-negative.")
    with np.errstate(over="ignore"):
        total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("At least one weight must be positive.")
    if not math.isfinite(total):
        raise ValueError("The sum of objective weights must be finite.")
    return weights / total


def _fixed_coordinates(
    raw: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        utilities = directions * (raw - origins) / scales
    if not np.all(np.isfinite(utilities)):
        raise ValueError("Fixed-scale selection requires finite scaled objective values.")
    return utilities


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
                "file": candidate.filename,
                "hash": Path(candidate.filename).stem,
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
        target = float(targets[metric])
        if math.isnan(target):
            raise ValueError(f"Reference target for {metric!r} is undefined")
        raw_value = (2.0 / math.pi) * math.atan(target)
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
    reducer_cfg: Mapping[str, Any] | None = None,
    scenario_labels: Sequence[str] | None = None,
) -> Optional[float]:
    metric = str(entry.get("metric", "")).strip()
    if not metric:
        return None
    basis = resolve_limit_basis(
        dict(entry),
        reducer_cfg=dict(reducer_cfg) if reducer_cfg else None,
    )
    if basis.scenario is not None:
        scenario_values = _scenario_metric_values(
            candidate.entry,
            basis.scenario,
            source=candidate.path,
        )
        return resolve_metric_value(scenario_values, metric)
    explicit_suite_basis = (
        candidate.scenario is not None
        and "scenario" in entry
        and entry.get("scenario") is None
    )
    applies_current_suite_reducer = reducer_cfg is not None and isinstance(
        candidate.entry.get("suite_metrics"), Mapping
    )
    applies_current_scenario_set = scenario_labels is not None and isinstance(
        candidate.entry.get("suite_metrics"), Mapping
    )
    stats_flat = candidate.stats_flat
    aggregated_values = candidate.aggregated_values
    if explicit_suite_basis or (
        (reducer_cfg is not None or scenario_labels is not None)
        and isinstance(candidate.entry.get("suite_metrics"), Mapping)
    ):
        stats_flat, aggregated_values = _extract_suite_metrics(
            candidate.entry,
            reducer_cfg=reducer_cfg,
            scenario_labels=scenario_labels,
        )
    if (
        candidate.scenario is not None
        and "reducer" in entry
        and not explicit_suite_basis
    ):
        requested_reducer = str(entry.get("reducer", "")).strip().lower()
        if requested_reducer != "mean":
            raise ValueError(
                f"Scenario {candidate.scenario!r} stores one mean value per metric; "
                f"limit reducer={requested_reducer!r} is unavailable for {candidate.filename}."
            )
    reducer = basis.reducer
    suite = candidate.entry.get("suite_metrics")
    if (
        "reducer" in entry
        and isinstance(suite, Mapping)
        and (candidate.scenario is None or explicit_suite_basis)
        and (scenario_labels is not None or "scenario_labels" in suite)
    ):
        # Saved stats contain only reducers requested by the original run. Reduce
        # the declared scenario population for this request, including its statuses.
        selected = (
            tuple(scenario_labels)
            if scenario_labels is not None
            else tuple(suite["scenario_labels"])
        )
        try:
            require_metric_definition(suite)
            if not selected:
                raise MetricAggregationError("no scenario population available")
            scenarios = suite.get("metrics", {}).get(metric, {}).get("scenarios", {})
            return raw_reduce(
                [decode_metric_value(scenarios, label) for label in selected], reducer
            )
        except MetricAggregationError as error:
            raise MetricAggregationError(
                f"Limit metric {metric!r} with reducer={reducer!r} could not be "
                f"resolved for {candidate.filename}: {error}",
                status=error.status,
            ) from error
    if "reducer" not in entry:
        value = resolve_metric_value(aggregated_values, metric)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    key = f"{metric}_{reducer}"
    value = resolve_metric_value(stats_flat, key)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    if (
        "reducer" not in entry
        and not explicit_suite_basis
        and not applies_current_suite_reducer
        and not applies_current_scenario_set
    ):
        fallback = _resolve_candidate_metric_value(candidate, metric)
        if isinstance(fallback, (int, float)) and not math.isnan(float(fallback)):
            return float(fallback)
    return None


def _format_available_limit_metrics(candidate: ParetoCandidate) -> str:
    available = sorted(
        {
            *candidate.objectives.keys(),
            *candidate.stats_flat.keys(),
            *candidate.aggregated_values.keys(),
        }
    )
    return ", ".join(available) if available else "<none>"


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
        return float(low) < value < float(high)
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
    return filter_candidates_with_limits(
        candidates,
        normalized_limits,
        scoring_weights=default_scoring_weights(),
    )


def filter_candidates_with_limits(
    candidates: Sequence[ParetoCandidate],
    limits: Sequence[Mapping[str, Any]],
    *,
    reducer_cfg: Mapping[str, Any] | None = None,
    scenario_labels: Sequence[str] | None = None,
    scoring_weights: Mapping[str, float] | None = None,
) -> tuple[List[ParetoCandidate], List[Dict[str, Any]]]:
    normalized_limits = normalize_limit_entries(list(limits))
    if scoring_weights is not None:
        normalized_limits = resolve_auto_limit_entries(
            normalized_limits,
            dict(scoring_weights),
        )
    enabled_limits = [entry for entry in normalized_limits if bool(entry.get("enabled", True))]
    if not enabled_limits:
        return list(candidates), enabled_limits

    filtered: List[ParetoCandidate] = []
    for candidate in candidates:
        rejected = False
        for entry in enabled_limits:
            value = _resolve_limit_value(
                candidate,
                entry,
                reducer_cfg,
                scenario_labels,
            )
            if value is None:
                metric = str(entry.get("metric", "")).strip() or "<missing>"
                available = _format_available_limit_metrics(candidate)
                reducer_note = (
                    f" Requested reducer: {entry['reducer']!r}."
                    if "reducer" in entry else ""
                )
                raise ValueError(
                    f"Limit metric {metric!r} could not be resolved for "
                    f"{candidate.filename}.{reducer_note} Available metrics: {available}"
                )
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
        anchor_files: list[str] = [candidates[0].filename]
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
        anchor_files = [candidates[i].filename for i in unique_anchor_indices]
    ranking_order = list(np.argsort(-scores, kind="stable"))
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


def _weighted_distance(
    utilities: np.ndarray, weights: np.ndarray, target_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        components = ((utilities - target_vector) ** 2) * weights
        squared_distances = components.sum(axis=1)
    if not np.all(np.isfinite(squared_distances)):
        raise ValueError("Weighted selection distances are not finite; check the objective scales.")
    return np.sqrt(squared_distances), components


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
    distances, weighted_sq = _weighted_distance(utilities, weights, target_vector)
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
    payload["ranking_order"] = list(np.argsort(distances, kind="stable"))
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


def _weighted_totals(utilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        scores = utilities @ weights
    if not np.all(np.isfinite(scores)):
        raise ValueError("Weighted selection totals are not finite; check the objective scales.")
    return scores


def _select_utility(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
    weights: np.ndarray,
) -> SelectionResult:
    scores = _weighted_totals(utilities, weights)
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
            "ranking_order": list(np.argsort(-scores, kind="stable")),
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
    # Positive preference minus negative preference is the weighted utility difference.
    scores = _weighted_totals(utilities, weights)
    with np.errstate(over="ignore", invalid="ignore"):
        net_flows = (
            np.zeros(1)
            if n == 1
            else (scores - np.sum(scores / n)) * (n / (n - 1))
        )
    if not np.all(np.isfinite(net_flows)):
        raise ValueError("Selection net flows are not finite; check the objective scales.")
    idx = int(np.argmax(scores))
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
            "ranking_order": list(np.argsort(-scores, kind="stable")),
            "score_vector": [float(x) for x in net_flows],
        },
    )


def _aspiration_scores(
    utilities: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    worst_shortfalls = np.max(np.maximum(-utilities, 0.0) * weights, axis=1)
    signed_shortfalls = -_weighted_totals(utilities, weights)
    return worst_shortfalls, signed_shortfalls


def _select_aspiration(
    candidates: Sequence[ParetoCandidate],
    active_metrics: Sequence[str],
    utilities: np.ndarray,
    weights: np.ndarray,
) -> SelectionResult:
    worst_shortfalls, signed_shortfalls = _aspiration_scores(utilities, weights)
    ranking = np.lexsort((signed_shortfalls, worst_shortfalls))
    idx = int(ranking[0])
    candidate = candidates[idx]
    return SelectionResult(
        candidate=candidate,
        method="aspiration",
        score=float(-worst_shortfalls[idx]),
        objective_values={
            metric: float(_resolve_candidate_metric_value(candidate, metric))
            for metric in active_metrics
        },
        details={
            "active_metrics": list(active_metrics),
            "weights": dict(zip(active_metrics, weights.tolist())),
            "shortfalls": dict(zip(active_metrics, np.maximum(-utilities[idx], 0.0).tolist())),
            "worst_weighted_shortfall": float(worst_shortfalls[idx]),
            "weighted_signed_shortfall": float(signed_shortfalls[idx]),
            "ranking_order": ranking.tolist(),
            "score_vector": (-worst_shortfalls).tolist(),
            "secondary_score_vector": (-signed_shortfalls).tolist(),
        },
    )


def _selection_sensitivity(
    candidates: Sequence[ParetoCandidate],
    result: SelectionResult,
    utilities: np.ndarray,
    weights: np.ndarray,
    target_vector: np.ndarray,
    *,
    weight_fraction: float | None,
    target_scale: float | None,
    fixed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> Dict[str, Any]:
    """Probe changes in preference while keeping the evaluated candidate pool fixed."""
    active_metrics = result.details["active_metrics"]

    def winner(coordinates: np.ndarray, scenario_weights: np.ndarray) -> str:
        if result.method == "aspiration":
            worst, signed = _aspiration_scores(coordinates, scenario_weights)
            idx = np.lexsort((signed, worst))[0]
        elif result.method in {"ideal", "reference"}:
            distances, _ = _weighted_distance(coordinates, scenario_weights, target_vector)
            idx = np.argmin(distances)
        else:  # Utility and linear outranking have the same ordering.
            idx = np.argmax(_weighted_totals(coordinates, scenario_weights))
        return candidates[int(idx)].filename

    checks: list[dict[str, Any]] = []
    if weight_fraction is not None:
        for source, source_metric in enumerate(active_metrics):
            if weight_fraction >= weights[source]:
                continue
            for dest, dest_metric in enumerate(active_metrics):
                if dest == source or weights[dest] == 0.0:
                    continue
                updated = weights.copy()
                updated[source] -= weight_fraction
                updated[dest] += weight_fraction
                checks.append(
                    {
                        "kind": "weight_transfer",
                        "from_metric": source_metric,
                        "to_metric": dest_metric,
                        "winner": winner(utilities, updated),
                    }
                )

    if target_scale is not None:
        assert fixed is not None
        raw, origins, directions, scales = fixed
        updated = utilities.copy()
        for col, metric in enumerate(active_metrics):
            origin = float(origins[col])
            step = target_scale * float(scales[col])
            for direction in (-1, 1):
                target = origin + direction * step
                if not math.isfinite(target) or target == origin:
                    raise ValueError(
                        f"Sensitivity target change for {metric!r} must be finite and representable."
                    )
                updated[:, col] = _fixed_coordinates(raw[:, col], target, directions[col], scales[col])
                checks.append(
                    {
                        "kind": "target_step",
                        "metric": metric,
                        "target": target,
                        "winner": winner(updated, weights),
                    }
                )
            updated[:, col] = utilities[:, col]

    return {
        "weight_transfer_fraction": weight_fraction,
        "target_step_scales": target_scale,
        "check_count": len(checks),
        "changed_check_count": sum(check["winner"] != result.candidate.filename for check in checks),
        "checks": checks,
    }


def select_candidate(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[ObjectiveSpec],
    *,
    method: str,
    objectives_arg: Optional[str] = None,
    weight_pairs: Optional[Sequence[str]] = None,
    target_pairs: Optional[Sequence[str]] = None,
    priority_arg: Optional[str] = None,
    scale_pairs: Optional[Sequence[str]] = None,
    sensitivity_weight_fraction: float | None = None,
    sensitivity_target_scale: float | None = None,
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
    weight_map = _parse_key_value_pairs(weight_pairs or [], value_name="weight")
    scale_map = _parse_key_value_pairs(scale_pairs or [], value_name="scale")
    if weight_map and normalized_method not in WEIGHTED_METHODS:
        raise ValueError(f"Method {normalized_method!r} does not support objective weights.")
    if target_map and normalized_method not in {"reference", "aspiration"}:
        raise ValueError(f"Method {normalized_method!r} does not support objective targets.")
    for option, values in (("weight", weight_map), ("target", target_map), ("scale", scale_map)):
        extra = sorted(values.keys() - set(active_metrics))
        if extra:
            raise ValueError(f"Unknown or inactive {option} metric(s): {', '.join(extra)}.")
    weights = _weights_for_metrics(active_metrics, weight_map)
    if sensitivity_weight_fraction is not None and (
        normalized_method not in WEIGHTED_METHODS or not 0.0 < sensitivity_weight_fraction < 1.0
    ):
        raise ValueError("Weight sensitivity requires a weighted method and a fraction between 0 and 1.")
    if sensitivity_target_scale is not None and (
        normalized_method != "aspiration"
        or not math.isfinite(sensitivity_target_scale)
        or sensitivity_target_scale <= 0.0
    ):
        raise ValueError("Target sensitivity requires aspiration and a positive finite scale multiplier.")
    if normalized_method in {"reference", "aspiration"} and not target_map:
        raise ValueError(f"Method {normalized_method!r} requires at least one --target metric=value.")
    if normalized_method == "aspiration" and not scale_map:
        raise ValueError("Method 'aspiration' requires --scale metric=value for every objective.")

    details: Dict[str, Any] = {"scaling": "atan_front_minmax"}
    fixed = None
    target_vector = np.ones(len(active_metrics), dtype=float)
    if scale_map:
        if normalized_method not in WEIGHTED_METHODS:
            raise ValueError(f"Method {normalized_method!r} does not support fixed scales.")
        for metric in active_metrics:
            scale = scale_map.get(metric)
            if scale is None or not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Fixed scale for {metric!r} must be positive and finite.")
        raw = _raw_objective_matrix(candidates, active_specs)
        nonfinite = np.argwhere(~np.isfinite(raw))
        if nonfinite.size:
            row, col = nonfinite[0]
            raise ValueError(
                f"Fixed-scale selection requires a finite {active_metrics[col]!r} "
                f"value in {candidates[row].filename}."
            )
        directions = np.array([1.0 if spec.goal == "max" else -1.0 for spec in active_specs])
        scales = np.array([scale_map[metric] for metric in active_metrics])
        if normalized_method in {"reference", "aspiration"}:
            for metric in active_metrics:
                if metric not in target_map or not math.isfinite(target_map[metric]):
                    raise ValueError(f"Fixed-scale {normalized_method} requires a finite target for {metric!r}.")
            origins = np.array([target_map[metric] for metric in active_metrics])
        else:
            origins = (raw * directions).max(axis=0) * directions
        fixed = (raw, origins, directions, scales)
        utilities = _fixed_coordinates(*fixed)
        target_vector = np.zeros(len(active_metrics), dtype=float)
        details = {"scaling": "fixed_raw", "scales": scale_map}
    else:
        utilities, lows, highs = _normalize_objective_matrix(candidates, active_specs)
        if normalized_method == "reference":
            target_vector = _normalize_reference_targets(target_map, active_specs, lows, highs)

    if normalized_method == "knee":
        result = _select_knee(candidates, active_metrics, utilities)
    elif normalized_method in {"reference", "ideal"}:
        result = _select_ideal_like(
            normalized_method,
            candidates,
            active_metrics,
            utilities,
            weights,
            target_vector,
            details={"weights": dict(zip(active_metrics, weights))},
        )
    elif normalized_method == "aspiration":
        result = _select_aspiration(candidates, active_metrics, utilities, weights)
    elif normalized_method == "utility":
        result = _select_utility(candidates, active_metrics, utilities, weights)
    elif normalized_method == "lexicographic":
        result = _select_lexicographic(candidates, active_metrics, utilities)
    elif normalized_method == "outranking":
        result = _select_outranking(candidates, active_metrics, utilities, weights)
    else:
        raise ValueError(f"Unsupported selection method {normalized_method!r}")
    if normalized_method in {"reference", "aspiration"}:
        details["targets"] = target_map
    if sensitivity_weight_fraction is not None or sensitivity_target_scale is not None:
        details["sensitivity"] = _selection_sensitivity(
            candidates,
            result,
            utilities,
            weights,
            target_vector,
            weight_fraction=sensitivity_weight_fraction,
            target_scale=sensitivity_target_scale,
            fixed=fixed,
        )
    result.details.update(details)
    return _attach_shared_selection_details(result, candidates=candidates, active_specs=active_specs)


def _format_objective_value(value: float) -> str:
    return "+inf" if value == math.inf else f"{value:.6g}"


def _report_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    right_align: Sequence[int] = (),
) -> list[str]:
    widths = [max(map(len, column)) for column in zip(headers, *rows)]
    terminal_width = shutil.get_terminal_size(fallback=(240, 24)).columns
    table_width = sum(widths) + 3 * len(widths) + 1
    if rows and table_width > terminal_width:
        # Stack the columns while keeping each exact metric key intact for copying.
        key_width = max(map(len, headers[1:]))
        value_width = max(len(value) for row in rows for value in row[1:])
        inner_width = max(key_width + value_width + 3, max(len(row[0]) for row in rows))
        value_width = inner_width - key_width - 3
        border = f"+{'-' * (inner_width + 2)}+"
        divider = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
        lines = [border]
        for name, *values in rows:
            lines.extend([f"| {name:<{inner_width}} |", divider])
            lines.extend(
                f"| {key:<{key_width}} | {value:>{value_width}} |"
                for key, value in zip(headers[1:], values)
            )
            lines.append(border)
        return lines

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def align(row):
        cells = [
            value.rjust(width) if idx in right_align else value.ljust(width)
            for idx, (value, width) in enumerate(zip(row, widths))
        ]
        return "| " + " | ".join(cells) + " |"

    return [border, align(headers), border, *(align(row) for row in rows), border]


def _command_path(path: Path) -> str:
    displayed = _display_path(path)
    return f"./{displayed}" if displayed.startswith("-") else displayed


def _report_source(pareto_dir: Path, args: argparse.Namespace) -> str:
    raw_path = getattr(args, "path", None)
    if raw_path and Path(raw_path).expanduser().is_file():
        # Keep the supplied filename, even if it is a link to a different basename.
        path = Path(raw_path).expanduser().absolute()
        if path.is_relative_to(Path.cwd()):
            path = path.relative_to(Path.cwd())
        return f"./{path}" if str(path).startswith("-") else str(path)
    return _command_path(pareto_dir)


def _plot_command(source: str, args: argparse.Namespace) -> str:
    command = ["passivbot", "tool", "pareto-plot", source, "--open"]
    for attr, flag in (
        ("method", "--method"),
        ("scenario", "--scenario"),
        ("limit_entries", "--limit"),
        ("limits_payload", "--limits"),
        ("objectives", "--objectives"),
        ("weight", "--weight"),
        ("target", "--target"),
        ("priority", "--priority"),
        ("show_top", "--show-top"),
        ("scale", "--scale"),
        ("sensitivity_weight_fraction", "--sensitivity-weight-fraction"),
        ("sensitivity_target_scales", "--sensitivity-target-scales"),
        ("qualification", "--qualification"),
    ):
        value = getattr(args, attr, None)
        if value is None or (attr == "method" and value == "ideal"):
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            token = str(item)
            if token.startswith("-"):
                command.append(f"{flag}={token}")
            else:
                command.extend([flag, token])
    return shlex.join(command)


def _report_header(source, *, loaded, retained, front, ranked=None, limits, scenario):
    counts = [f"{loaded} loaded"] if retained != loaded or front != loaded else []
    if retained != loaded:
        counts.append(f"{retained} after limits")
    if ranked is None or ranked != front:
        counts.append(f"{front} on active front")
    if ranked is not None:
        counts.append(f"{ranked} ranked")
    if limits:
        noun = "limit" if len(limits) == 1 else "limits"
        counts.append(f"{len(limits)} CLI {noun}")
    else:
        counts.append("no CLI limits")
    lines = [f"Source: {source}", f"Candidates: {' · '.join(counts)}"]
    if scenario is not None:
        lines.append(f"Scenario: {scenario} · rebuilt from saved aggregate Pareto members")
    if limits:
        lines.extend(f"  Exclude if {_format_limit_entry(entry, include_scope=True)}" for entry in limits)
    return lines


def _report_sensitivity(sensitivity, labels, selected):
    if not sensitivity or not sensitivity["check_count"]:
        return []
    checked = sensitivity["check_count"]
    changed = sensitivity["changed_check_count"]
    probes = []
    weight_fraction = sensitivity["weight_transfer_fraction"]
    if weight_fraction is not None:
        probes.append(f"{100 * weight_fraction:g}-percentage-point weight transfers")
    if sensitivity["target_step_scales"] is not None:
        probes.append(f"target shifts of {sensitivity['target_step_scales']:g} scale units")
    scope = f"on this pool ({'; '.join(probes)})"
    lines = [f"Sensitivity: {changed}/{checked} checks changed the pick {scope}."]
    if changed:
        check = next(
            check for check in sensitivity["checks"] if check["winner"] != selected
        )
        if check["kind"] == "weight_transfer":
            src = check["from_metric"]
            dest = check["to_metric"]
            step = 100 * weight_fraction
            unit = "percentage point" if step == 1 else "percentage points"
            change = f"move {step:g} {unit} of weight from {src} to {dest}"
        else:
            value = _format_objective_value(check["target"])
            change = f"set {check['metric']} target to {value}"
        winner = labels.get(check["winner"], check["winner"])
        lines.append(f"  {winner} ranks first if you {change}.")
    return lines


def _report_qualification(qualification, shortlist, labels):
    if qualification is None:
        return ["CPU replay evidence: none attached."]
    lines = ["CPU replay evidence (attached):"]
    subjects = [
        (labels.get(row["file"], row["file"]), qualification["candidates"].get(row["file_sha256"]))
        for row in shortlist
    ]
    if "incumbent" in qualification:
        subjects.append(("Incumbent", qualification["incumbent"]))
    for name, subject in subjects:
        if subject is None or not subject["results"]:
            lines.append(f"  {name}: no matching replay results")
            continue
        for scenario, replay in subject["results"].items():
            scope = qualification["scenarios"][scenario]
            result = f"{name} · {scenario} ({scope['split']}, {scope['execution']}) · {replay['status']}"
            if replay.get("liquidated"):
                result += " · liquidated"
            if replay.get("reason"):
                result += f" · {replay['reason']}"
            lines.append(f"  {result}")
    return lines


def _objective_rows(specs, *value_columns):
    rows = []
    for spec in specs:
        metric = spec.metric
        direction = "↑" if spec.goal == "max" else "↓"
        rows.append([f"{direction} {metric}", *[column(metric) for column in value_columns]])
    return rows


def _format_saved_value(values, metric):
    value = values[metric]
    if value is None:
        status = values["metric_status"][metric]["status"]
        return {"positive_infinity": "+inf", "negative_infinity": "-inf"}.get(status, status)
    return _format_objective_value(value)


def format_selection_result(
    pareto_dir: Path,
    evaluation: SelectionEvaluation,
    record: Mapping[str, Any],
    args: argparse.Namespace,
) -> str:
    result = evaluation.result
    source = _report_source(pareto_dir, args)
    labels = record["candidate_labels"]
    lines = _report_header(
        source,
        loaded=record["loaded_count"],
        retained=record["retained_count"],
        front=record["active_front_count"],
        ranked=record["selection_count"],
        limits=evaluation.active_limits,
        scenario=evaluation.scenario,
    )
    selected = result.candidate.filename
    lines.append(f"Selected file: {labels.get(selected, selected)}")

    settings = record["settings"]
    method = [result.method]
    scope = "post-limit pool" if result.method == "reference" else "active front"
    if result.method != "lexicographic":
        method.append("fixed raw scales" if settings.get("scales") else f"scaled to {scope}")
        label, value = _score_label_and_value(result)
        method.append(f"{label} {value:.4g}")
    else:
        method.append("priority in table order")
    lines.append(f"Method: {' · '.join(method)}")

    shortlist = record["top_candidates"]
    if getattr(args, "show_top", None) is None:
        shortlist = shortlist[:2]
    alternative = shortlist[1] if len(shortlist) > 1 else None
    headers = ["Objective", "Selected"]
    columns = [lambda metric: _format_objective_value(result.objective_values[metric])]
    if alternative:
        headers.append("#2")
        columns.append(lambda metric: _format_saved_value(alternative["objectives"], metric))
    targets = result.details.get("targets")
    if targets:
        headers.append("Target")
        columns.append(lambda metric: _format_objective_value(targets[metric]) if metric in targets else "—")
    best = result.details["ideal_point"]
    worst = {}
    for spec in evaluation.active_specs:
        least_favorable = max if spec.goal == "min" else min
        worst[spec.metric] = least_favorable(
            _resolve_candidate_metric_value(candidate, spec.metric)
            for candidate in evaluation.selection_pool
        )
    extrema_scope = "Pool" if result.method == "reference" else "Front"
    headers.extend([f"{extrema_scope} best", f"{extrema_scope} worst"])
    columns.extend([
        lambda metric: _format_objective_value(best[metric]),
        lambda metric: _format_objective_value(worst[metric]),
    ])
    utilities = result.details.get("selected_utilities")
    if utilities:
        headers.append("Scaled utility" if settings.get("scales") else "Utility")
        columns.append(lambda metric: f"{utilities[metric]:.6g}")
    weights = settings.get("weights")
    if weights:
        headers.append("Weight")
        columns.append(lambda metric: f"{weights[metric]:.6g}")
    lines.append("")
    rows = _objective_rows(evaluation.active_specs, *columns)
    lines.extend(_report_table(headers, rows, right_align=range(1, len(headers))))
    lines.append(f"{extrema_scope} extrema may come from different files.")
    if weights:
        lines.append("Weights are normalized across active objectives.")
    if alternative:
        lines.append("")
        lines.append("Ranked alternatives")
        for row in shortlist[1:]:
            filename = row["file"]
            lines.append(f"  #{row['rank']}  {labels.get(filename, filename)}")
            if row is not alternative:
                values = [
                    f"{metric} {_format_saved_value(row['objectives'], metric)}"
                    for metric in result.objective_values
                ]
                lines.extend(f"      {value}" for value in values)
    contributions = result.details.get("distance_components")
    if contributions and sum(contributions.values()) > 0:
        metric = max(contributions, key=contributions.get)
        share = contributions[metric] / sum(contributions.values())
        origin = "ideal" if result.method == "ideal" else "target"
        lines.extend(["", f"{metric} accounts for {share:.0%} of squared distance to {origin}."])

    sensitivity = record["sensitivity"]
    if sensitivity:
        lines.extend(_report_sensitivity(sensitivity, labels, selected))
    lines.extend(_report_qualification(record["qualification"], shortlist, labels))

    replay = ["passivbot", "backtest", _command_path(result.candidate.path)]
    scenario = evaluation.scenario
    lines.extend(["", "Backtest"])
    if scenario is not None and "," in scenario:
        lines.append("Choose this scenario in the backtest suite config. --scenarios splits labels on commas.")
    else:
        if scenario is not None:
            if scenario.startswith("-"):
                replay.append(f"--scenarios={scenario}")
            else:
                replay.extend(["--scenarios", scenario])
        lines.append(shlex.join(replay))
    lines.extend(["", "Explore in browser", _plot_command(source, args)])
    return "\n".join(lines)


def _format_inspection(pareto_dir, candidates, scoring_specs, args):
    front_view = args.front
    unsupported = [
        name for name in (
            "priority", "weight", "target", "scale", "show_top", "qualification",
            "sensitivity_weight_fraction", "sensitivity_target_scales", "save_selected", "save_filtered",
        )
        if getattr(args, name, None) is not None
    ]
    if args.method != "ideal":
        unsupported.append("method")
    if not front_view and args.objectives is not None:
        unsupported.append("objectives")
    if unsupported:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
        raise ValueError(f"Selection options do not apply to this view: {flags}")

    source = _report_source(pareto_dir, args)
    _, filtered, limits, scenario = _selection_inputs(candidates, scoring_specs, args)
    if not front_view:
        saved_goals = {spec.metric: spec.goal for spec in scoring_specs}
        rows = []
        for metric, definition in CORE_METRIC_CATALOG.items():
            available = 0
            for candidate in filtered:
                try:
                    value = _resolve_candidate_metric_value(candidate, metric)
                except MetricAggregationError:
                    continue  # Explicit unavailable facts are excluded from value coverage.
                available += value is not None
            if available:
                goal = saved_goals.get(metric) or default_pareto_goal(metric)
                rows.append([metric, definition["unit"], goal or "limit only", f"{available}/{len(filtered)}"])
        lines = [f"Source: {source}"]
        if scenario is not None:
            lines.append(f"Scenario: {scenario}")
        if limits:
            lines.append(f"Candidates: {len(filtered)} of {len(candidates)} after limits")
        lines.append("")
        lines.extend(_report_table(["Saved metric key", "Raw unit", "Goal", "Values"], rows, right_align=(3,)))
        lines.append("Fraction inputs use raw units (0.05 means 5%).")
        return "\n".join(lines)

    specs = _resolve_active_objective_metrics(
        scoring_specs, filtered, objectives_arg=args.objectives, priority_arg=None,
        target_map=None, method="ideal",
    )
    _raw_objective_matrix(filtered, specs)
    front = build_objective_front(filtered, specs)
    labels = _candidate_labels(candidates)
    extrema = {}
    for spec in specs:
        values = [_resolve_candidate_metric_value(candidate, spec.metric) for candidate in front]
        low, high = min(values), max(values)
        best = low if spec.goal == "min" else high
        name = front[values.index(best)].filename
        extrema[spec.metric] = (low, high, labels.get(name, name))
    lines = _report_header(
        source, loaded=len(candidates), retained=len(filtered), front=len(front),
        limits=limits, scenario=scenario,
    )
    rows = _objective_rows(
        specs,
        lambda metric: _format_objective_value(extrema[metric][0]),
        lambda metric: _format_objective_value(extrema[metric][1]),
        lambda metric: extrema[metric][2],
    )
    lines.append("")
    lines.extend(_report_table(["Objective", "Min", "Max", "Best file"], rows, right_align=(1, 2)))
    return "\n".join(lines)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _is_within_by_identity(path: Path, directory: Path) -> bool:
    current = path
    while True:
        try:
            if current.samefile(directory):
                return True
        except FileNotFoundError:
            pass
        if current.parent == current:
            return False
        current = current.parent


def _validate_output_path(path: Path, pareto_dir: Path, *, directory: bool) -> None:
    if (
        _is_within(path, pareto_dir)
        or _is_within_by_identity(path, pareto_dir)
        or (
            directory
            and (
                _is_within(pareto_dir, path)
                or _is_within_by_identity(pareto_dir, path)
            )
        )
    ):
        raise ValueError(
            f"Output must not overlap the source Pareto directory: {path}"
        )


def _prepare_selected_output(
    raw_path: str | os.PathLike[str] | None,
    pareto_dir: Path,
) -> Path | None:
    if raw_path is None:
        return None
    unresolved_output = Path(raw_path).expanduser()
    if unresolved_output.is_symlink():
        raise FileExistsError(f"Selected output already exists: {unresolved_output}")
    output = unresolved_output.resolve()
    _validate_output_path(output, pareto_dir, directory=False)
    if output.suffix.lower() != ".json":
        raise ValueError(f"Selected output must use a .json filename: {output}")
    if output.exists():
        if output.is_dir():
            raise IsADirectoryError(f"Selected output path is a directory: {output}")
        raise FileExistsError(f"Selected output already exists: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise NotADirectoryError(f"Selected output parent is not a directory: {output.parent}")
    return output


def _prepare_filtered_output(
    raw_path: str | os.PathLike[str] | None,
    pareto_dir: Path,
    candidates: Sequence[ParetoCandidate],
) -> Path | None:
    if raw_path is None:
        return None
    unresolved_output = Path(raw_path).expanduser()
    if unresolved_output.is_symlink():
        raise FileExistsError(f"Filtered output directory already exists: {unresolved_output}")
    output = unresolved_output.resolve()
    _validate_output_path(output, pareto_dir, directory=True)
    names = [candidate.filename for candidate in candidates]
    folded_names = [name.casefold() for name in names]
    if len(folded_names) != len(set(folded_names)):
        raise ValueError("Filtered members contain duplicate filenames.")
    if FILTERED_SELECTION_MANIFEST.casefold() in folded_names:
        raise ValueError(
            f"Filtered member filename conflicts with {FILTERED_SELECTION_MANIFEST!r}."
        )
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"Filtered output path is not a directory: {output}")
        raise FileExistsError(f"Filtered output directory already exists: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise NotADirectoryError(f"Filtered output parent is not a directory: {output.parent}")
    return output


def _verify_export_digest(candidate: ParetoCandidate, copied_file: Path) -> None:
    with copied_file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != candidate.file_sha256:
        raise ValueError(f"Pareto candidate changed after loading: {candidate.filename}")


def _stage_selected(candidate: ParetoCandidate, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=".pareto-selected-"))
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=stage,
            prefix="candidate-",
            suffix=".tmp",
        )
    except Exception:
        _remove_export_tree(stage)
        raise
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            file_descriptor = -1
            with candidate.path.open("rb") as source:
                shutil.copyfileobj(source, destination)
        _verify_export_digest(candidate, temporary)
        return temporary
    except Exception:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        _remove_selected_stage(temporary)
        raise


def _install_selected(temporary: Path, output: Path) -> None:
    try:
        if output.exists():
            raise FileExistsError(f"Selected output already exists: {output}")
        temporary.rename(output)
    finally:
        _remove_selected_stage(temporary)


def _stage_filtered(
    output: Path,
    pareto_dir: Path,
    candidates: Sequence[ParetoCandidate],
    *,
    loaded_count: int,
    active_limits: Sequence[Mapping[str, Any]],
    scenario: str | None,
    selected: ParetoCandidate,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=".pareto-filtered-"))
    try:
        for candidate in candidates:
            destination = stage / candidate.filename
            if destination.exists():
                raise ValueError(
                    f"Filtered member filename collides on the destination filesystem: "
                    f"{candidate.filename}"
                )
            shutil.copyfile(candidate.path, destination)
            _verify_export_digest(candidate, destination)
        manifest = {
            "tool": "passivbot tool pareto",
            "pareto_dir": str(pareto_dir),
            "loaded_count": int(loaded_count),
            "retained_count": len(candidates),
            "scenario": scenario,
            "applied_limits": _json_ready(active_limits),
            "selected_member": selected.filename,
            "members": [candidate.filename for candidate in candidates],
        }
        manifest_path = stage / FILTERED_SELECTION_MANIFEST
        if manifest_path.exists():
            raise ValueError(
                f"Filtered member filename conflicts with {FILTERED_SELECTION_MANIFEST!r}."
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return stage
    except Exception:
        _remove_export_tree(stage)
        raise


def _remove_export_tree(path: Path) -> None:
    shutil.rmtree(path)


def _remove_selected_stage(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.parent.exists():
        _remove_export_tree(path.parent)


def _install_filtered(stage: Path, output: Path) -> Path:
    try:
        if output.exists():
            raise FileExistsError(f"Filtered output directory already exists: {output}")
        stage.rename(output)
        return output / FILTERED_SELECTION_MANIFEST
    finally:
        if stage.exists():
            _remove_export_tree(stage)


@dataclass(frozen=True)
class SelectionEvaluation:
    selection_candidates: Sequence[ParetoCandidate]
    filtered_candidates: Sequence[ParetoCandidate]
    active_limits: Sequence[Dict[str, Any]]
    active_specs: Sequence[ObjectiveSpec]
    active_front: Sequence[ParetoCandidate]
    selection_pool: Sequence[ParetoCandidate]
    scenario: str | None
    result: SelectionResult


def _selection_inputs(candidates, scoring_specs, args):
    raw_scenario = getattr(args, "scenario", None)
    scenario = str(raw_scenario).strip() if raw_scenario is not None else None
    selection_candidates = (
        project_candidates_to_scenario(candidates, scoring_specs, scenario)
        if scenario is not None
        else candidates
    )
    filtered_candidates, active_limits = filter_candidates(
        selection_candidates,
        limits_payload=getattr(args, "limits_payload", None),
        limit_entries=list(getattr(args, "limit_entries", []) or []),
    )
    if not filtered_candidates:
        raise ValueError("No Pareto candidates remained after applying limits.")
    return selection_candidates, filtered_candidates, active_limits, scenario


def evaluate_selection(
    candidates: Sequence[ParetoCandidate],
    scoring_specs: Sequence[ObjectiveSpec],
    args: argparse.Namespace,
) -> SelectionEvaluation:
    method = parse_method_name(args.method)
    selection_candidates, filtered_candidates, active_limits, scenario = _selection_inputs(
        candidates, scoring_specs, args
    )
    option_maps = {
        name: _parse_key_value_pairs(getattr(args, name, None) or [], value_name=name)
        for name in ("weight", "target", "scale")
    }
    priority = getattr(args, "priority", None)
    if priority is not None and method != "lexicographic":
        raise ValueError("--priority requires method=lexicographic.")
    if priority is not None and getattr(args, "objectives", None):
        raise ValueError("Choose either --priority or --objectives.")
    active_specs = _resolve_active_objective_metrics(
        scoring_specs,
        filtered_candidates,
        objectives_arg=getattr(args, "objectives", None),
        priority_arg=priority,
        target_map=option_maps["target"],
        method=method,
    )
    active_metrics = [spec.metric for spec in active_specs]
    if not active_metrics or len(set(active_metrics)) != len(active_metrics):
        raise ValueError("Choose at least one distinct objective.")
    for option, values in option_maps.items():
        unknown = values.keys() - set(active_metrics)
        if unknown:
            raise ValueError(f"--{option} names inactive objectives: {', '.join(sorted(unknown))}")

    if method in WEIGHTED_METHODS:
        weights = _weights_for_metrics(active_metrics, option_maps["weight"])
        active_specs = [
            spec for spec, weight in zip(active_specs, weights) if weight > 0.0
        ]
    effective_metrics = [spec.metric for spec in active_specs]
    effective_pairs = {
        name: [f"{metric}={values[metric]}" for metric in effective_metrics if metric in values]
        for name, values in option_maps.items()
    }
    weight_step = getattr(args, "sensitivity_weight_fraction", None)
    if weight_step is None and method in WEIGHTED_METHODS:
        weight_step = 0.01

    active_front = build_objective_front(filtered_candidates, active_specs)
    # Exact reference targets can favor a dominated candidate.
    selection_pool = filtered_candidates if method == "reference" else active_front
    result = select_candidate(
        selection_pool,
        active_specs,
        method=method,
        weight_pairs=effective_pairs["weight"],
        target_pairs=effective_pairs["target"],
        scale_pairs=effective_pairs["scale"],
        sensitivity_weight_fraction=weight_step if weight_step != 0 else None,
        sensitivity_target_scale=getattr(args, "sensitivity_target_scales", None),
    )
    return SelectionEvaluation(
        selection_candidates=selection_candidates,
        filtered_candidates=filtered_candidates,
        active_limits=active_limits,
        active_specs=active_specs,
        active_front=active_front,
        selection_pool=selection_pool,
        scenario=scenario,
        result=result,
    )


def _objective_deltas(
    objectives: Mapping[str, float], selected: Mapping[str, float]
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    statuses: dict[str, dict[str, str]] = {}
    for metric, value in objectives.items():
        baseline = selected[metric]
        delta = value - baseline
        if math.isnan(delta):
            deltas[metric] = None
            statuses[metric] = {"status": "undefined", "reason": "equal_infinite_values"}
        elif math.isinf(delta) and math.isfinite(value) and math.isfinite(baseline):
            deltas[metric] = None
            statuses[metric] = {"status": "invalid", "reason": "delta_overflow"}
        else:
            deltas[metric] = delta
    if statuses:
        deltas["metric_status"] = statuses
    return deltas


def _candidate_labels(candidates: Sequence[ParetoCandidate]) -> dict[str, str]:
    """Abbreviate legacy full hashes while keeping the displayed filenames distinct."""
    filenames = {candidate.filename for candidate in candidates}
    groups: dict[str, list[str]] = {}
    for filename in sorted(filenames):
        if re.fullmatch(r"[0-9a-f]{64}\.json", filename):
            prefix = filename[:PARETO_FILENAME_PREFIX_LENGTH]
            groups.setdefault(prefix, []).append(filename)

    labels = {}
    for group in groups.values():
        for filename in group:
            for width in range(PARETO_FILENAME_PREFIX_LENGTH, 64):
                prefix = filename[:width]
                label = f"{prefix}.json"
                if label in filenames:
                    continue
                if any(other != filename and other.startswith(prefix) for other in group):
                    continue
                labels[filename] = label
                break
    return labels


def build_selection_record(
    evaluation: SelectionEvaluation, args: argparse.Namespace
) -> dict[str, Any]:
    result = evaluation.result
    pool = evaluation.selection_pool
    active_metrics = [spec.metric for spec in evaluation.active_specs]
    requested_count = getattr(args, "show_top", None)
    show_top = 5 if requested_count is None else int(requested_count)
    if show_top < 1:
        raise ValueError("--show-top must be positive.")
    rows = _ranked_rows(
        pool,
        active_metrics,
        result.details["ranking_order"],
        result.details["score_vector"],
        limit=show_top,
    )
    loaded_candidates = evaluation.selection_candidates
    fingerprints = [
        (candidate.filename, candidate.file_sha256) for candidate in loaded_candidates
    ]
    candidate_set_hash = hashlib.sha256(
        json.dumps(fingerprints, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    secondary_scores = result.details.get("secondary_score_vector")
    for row, index in zip(rows, result.details["ranking_order"]):
        row["file_sha256"] = pool[int(index)].file_sha256
        row["objective_deltas"] = _objective_deltas(row["objectives"], result.objective_values)
        if result.method != "lexicographic":
            row["score_margin"] = result.score - row["score"]
        if secondary_scores is not None:
            row["secondary_score"] = secondary_scores[int(index)]

    qualification_path = getattr(args, "qualification", None)
    qualification = (
        qualification_for_candidates(
            load_qualification(qualification_path), [row["file_sha256"] for row in rows]
        )
        if qualification_path is not None
        else None
    )
    details = _public_selection_details(result.details)
    payload = {
        "schema_version": SELECTION_RECORD_VERSION,
        "metric_definition_version": CORE_METRIC_DEFINITION_VERSION,
        "candidate_set_sha256": candidate_set_hash,
        "candidate_labels": _candidate_labels(loaded_candidates),
        "loaded_count": len(loaded_candidates),
        "retained_count": len(evaluation.filtered_candidates),
        "active_front_count": len(evaluation.active_front),
        "selection_count": len(pool),
        "normalization_scope": (
            "post_limit_pool" if result.method == "reference" else "active_front"
        ),
        "applied_limits": evaluation.active_limits,
        "objectives": [
            (
                spec.to_config()
                if evaluation.scenario is None
                else {"metric": spec.metric, "goal": spec.goal, "scenario": evaluation.scenario}
            )
            for spec in evaluation.active_specs
        ],
        "method": result.method,
        "method_description": _method_explanation(result.method),
        "settings": {
            key: details[key]
            for key in ("scaling", "weights", "scales", "targets", "priority")
            if key in details
        },
        "selected": {
            "file": result.candidate.filename,
            "hash": Path(result.candidate.filename).stem,
            "file_sha256": result.candidate.file_sha256,
            "score": float(result.score),
            "objectives": result.objective_values,
            "details": details,
        },
        "top_candidates": rows,
        "sensitivity": result.details.get("sensitivity"),
        "qualification": qualification,
    }
    if evaluation.scenario is not None:
        payload.update(
            {
                "scenario": evaluation.scenario,
                "front_scope": "saved_aggregate_pareto_members",
                "scenario_front_count": len(evaluation.active_front),
                "scenario_front_complete": False,
            }
        )
    return _json_ready(payload)


def run_from_args(args: argparse.Namespace) -> SelectionResult | None:
    pareto_dir, candidates, scoring_specs = load_selection_candidates(getattr(args, "path", None))
    if getattr(args, "front", False) or getattr(args, "list_metrics", False):
        print(_format_inspection(pareto_dir, candidates, scoring_specs, args))
        return None
    evaluation = evaluate_selection(candidates, scoring_specs, args)
    record = build_selection_record(evaluation, args)
    result = evaluation.result
    filtered_candidates = evaluation.filtered_candidates
    scenario = evaluation.scenario
    active_limits = evaluation.active_limits
    selected_output = _prepare_selected_output(
        getattr(args, "save_selected", None),
        pareto_dir,
    )
    filtered_output = _prepare_filtered_output(
        getattr(args, "save_filtered", None),
        pareto_dir,
        filtered_candidates,
    )
    selected_stage: Path | None = None
    filtered_stage: Path | None = None
    filtered_manifest: Path | None = None
    try:
        if selected_output is not None:
            selected_stage = _stage_selected(
                result.candidate,
                selected_output,
            )
        if filtered_output is not None:
            filtered_stage = _stage_filtered(
                filtered_output,
                pareto_dir,
                filtered_candidates,
                loaded_count=len(candidates),
                active_limits=active_limits,
                scenario=scenario,
                selected=result.candidate,
            )
        if selected_output is not None and selected_stage is not None:
            _install_selected(selected_stage, selected_output)
            selected_stage = None
        if filtered_output is not None and filtered_stage is not None:
            filtered_manifest = _install_filtered(
                filtered_stage,
                filtered_output,
            )
            filtered_stage = None
    finally:
        if selected_stage is not None:
            _remove_selected_stage(selected_stage)
        if filtered_stage is not None and filtered_stage.exists():
            _remove_export_tree(filtered_stage)
    if getattr(args, "json_output", False):
        payload = record
        payload["pareto_dir"] = str(pareto_dir)
        payload["selected"]["path"] = str(result.candidate.path)
        for row, index in zip(payload["top_candidates"], result.details["ranking_order"]):
            row["path"] = str(evaluation.selection_pool[int(index)].path)
        if selected_output is not None:
            payload["selected"]["saved_path"] = str(selected_output)
        if filtered_output is not None and filtered_manifest is not None:
            payload["saved_filtered"] = {
                "directory": str(filtered_output),
                "manifest": str(filtered_manifest),
                "count": len(filtered_candidates),
            }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        output = format_selection_result(pareto_dir, evaluation, record, args)
        saved_lines: List[str] = []
        if selected_output is not None:
            saved_lines.append(
                f"Saved selected member: {_display_path(selected_output)}"
            )
        if filtered_output is not None and filtered_manifest is not None:
            saved_lines.append(
                f"Saved filtered members: {len(filtered_candidates)} to "
                f"{_display_path(filtered_output)}"
            )
        if saved_lines:
            output += "\n" + "\n".join(saved_lines)
        print(output)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_from_args(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
