from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from config.metrics import split_metric_stat_suffix
from metrics_schema import (copy_metric_value, decode_metric_value, encode_metric_values, raw_reduce, MetricAggregationError)
from pareto_core import detect_latest_pareto_dir
from pareto_explorer import (
    ParetoCandidate,
    filter_candidates,
    load_candidates,
)


CONFIG_SECTIONS = ("backtest", "bot", "coin_overrides", "live", "logging", "monitor")
DEFAULT_TOP = 50


@dataclass(frozen=True)
class SeriesSummary:
    group: str
    name: str
    count: int
    missing: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    p05: Optional[float]
    p25: Optional[float]
    p75: Optional[float]
    p95: Optional[float]
    unique_count: int
    min_file: str
    max_file: str
    finite_count: int = 0
    positive_infinity_count: int = 0
    negative_infinity_count: int = 0
    unavailable_count: int = 0
    unavailable_sources: Dict[str, Any] = field(default_factory=dict)
    metric_status: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Optional[float]:
        if self.maximum is None or self.minimum is None:
            return None
        span = self.maximum - self.minimum
        return (
            None
            if math.isnan(span)
            or (
                not math.isfinite(span)
                and math.isfinite(self.maximum)
                and math.isfinite(self.minimum)
            )
            else span
        )

    @property
    def iqr(self) -> Optional[float]:
        if self.p75 is None or self.p25 is None:
            return None
        span = self.p75 - self.p25
        return None if not math.isfinite(span) else span

    @property
    def cv_abs(self) -> float:
        return (
            abs(self.std / self.mean)
            if self.std is not None and self.mean
            else math.inf
        )

    def to_dict(self) -> Dict[str, Any]:
        raw = {
            "min": self.minimum,
            "max": self.maximum,
            "range": self.span,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "iqr": self.iqr,
        }
        result = encode_metric_values(
            {name: value for name, value in raw.items() if value is not None}
        )
        statuses = {**self.metric_status, **result.get("metric_status", {})}
        for name, value in raw.items():
            if value is None:
                result[name] = None
                statuses.setdefault(
                    name,
                    {
                        "status": "undefined",
                        "reason": "distribution statistic has unavailable support",
                    },
                )
        if statuses:
            result["metric_status"] = statuses
        result.update(
            group=self.group,
            name=self.name,
            count=self.count,
            missing=self.missing,
            finite_count=self.finite_count,
            positive_infinity_count=self.positive_infinity_count,
            negative_infinity_count=self.negative_infinity_count,
            unavailable_count=self.unavailable_count,
            unavailable_sources=self.unavailable_sources,
            unique_count=self.unique_count,
            min_file=self.min_file,
            max_file=self.max_file,
        )
        return result


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _flatten_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric(value, child))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for idx, value in enumerate(obj):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(_flatten_numeric(value, child))
    elif _is_number(obj):
        out[prefix] = float(obj)
    return out


def extract_config_values(candidate: ParetoCandidate) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for section in CONFIG_SECTIONS:
        payload = candidate.entry.get(section)
        if isinstance(payload, Mapping):
            values.update(_flatten_numeric(payload, section))
    return values


def _merge_metric_fact(values, source, key, output_name=None):
    name = key if output_name is None else output_name
    copied = copy_metric_value(source, key, output_name=name)
    statuses = copied.pop("metric_status", {})
    values.update(copied)
    values.get("metric_status", {}).pop(name, None)
    if not values.get("metric_status"):
        values.pop("metric_status", None)
    if statuses:
        values.setdefault("metric_status", {}).update(statuses)


def extract_metric_values(candidate: ParetoCandidate) -> Dict[str, Any]:
    """Keep explicit unavailable facts and true endpoints with objective precedence."""
    values = {}
    for key in candidate.stats_flat:
        if key == "metric_status":
            continue
        _merge_metric_fact(values, candidate.stats_flat, key)
        base, reducer = split_metric_stat_suffix(key)
        if reducer == "mean":
            _merge_metric_fact(values, candidate.stats_flat, key, base)
    for key in candidate.aggregated_values:
        if key != "metric_status":
            _merge_metric_fact(values, candidate.aggregated_values, key)
    # Candidate objectives are already decoded raw values, including true infinities.
    objectives = encode_metric_values(candidate.objectives)
    for key in candidate.objectives:
        _merge_metric_fact(values, objectives, key)
    return values


def resolve_metric_series_value(
    candidate: ParetoCandidate, metric: str
) -> Optional[float]:
    if metric in candidate.objectives:
        return candidate.objectives[metric]
    values = extract_metric_values(candidate)
    if metric not in values:
        return None
    try:
        return decode_metric_value(values, metric)
    except MetricAggregationError:
        return None


def _collect_value_maps(
    candidates: Sequence[ParetoCandidate],
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    return (
        [extract_config_values(candidate) for candidate in candidates],
        [extract_metric_values(candidate) for candidate in candidates],
    )


def _all_names(maps: Sequence[Mapping[str, float]]) -> List[str]:
    names = set()
    for values in maps:
        names.update(key for key in values if key != "metric_status")
    return sorted(names)


def _split_csv(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _match_names(patterns: Sequence[str], names: Sequence[str]) -> List[str]:
    matched: List[str] = []
    for pattern in patterns:
        hits = [name for name in names if fnmatch.fnmatchcase(name, pattern)]
        if not hits and pattern in names:
            hits = [pattern]
        for hit in hits:
            if hit not in matched:
                matched.append(hit)
    missing = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(name, pattern) for name in names)]
    missing = [pattern for pattern in missing if pattern not in names]
    if missing:
        raise ValueError(f"No fields matched: {missing}")
    return matched


def summarize_series(
    *,
    group: str,
    name: str,
    candidates: Sequence[ParetoCandidate],
    value_maps: Sequence[Mapping[str, float]],
) -> Optional[SeriesSummary]:
    pairs, unavailable, statuses = [], {}, {}
    for idx, values in enumerate(value_maps):
        if name not in values:
            continue
        try:
            value = (
                decode_metric_value(values, name)
                if group == "metric"
                else float(values[name])
            )
        except MetricAggregationError as error:
            unavailable[candidates[idx].path.name] = values.get(
                "metric_status", {}
            ).get(name, {"status": error.status, "reason": str(error)})
            continue
        pairs.append((idx, value))
    if not pairs and not unavailable:
        return None
    raw = [value for _, value in pairs]
    finite_count = sum(math.isfinite(value) for value in raw)
    fields = {}
    for mode in ("mean", "median", "std"):
        try:
            fields[mode] = raw_reduce(raw, mode)
        except (MetricAggregationError, OverflowError) as error:
            fields[mode] = None
            statuses[mode] = {
                "status": getattr(error, "status", "invalid"),
                "reason": str(error),
            }
    for field_name, percentile in (("p05", 5), ("p25", 25), ("p75", 75), ("p95", 95)):
        if not raw or finite_count != len(raw):
            fields[field_name] = None
            statuses[field_name] = {
                "status": "undefined",
                "reason": "percentile interpolation requires a nonempty finite population",
            }
        else:
            value = float(np.percentile(raw, percentile))
            fields[field_name] = value if math.isfinite(value) else None
            if fields[field_name] is None:
                statuses[field_name] = {
                    "status": "invalid",
                    "reason": "percentile arithmetic overflow",
                }
    min_index = min(pairs, key=lambda pair: pair[1])[0] if pairs else None
    max_index = max(pairs, key=lambda pair: pair[1])[0] if pairs else None
    return SeriesSummary(
        group=group,
        name=name,
        count=len(raw),
        missing=len(candidates) - len(raw),
        minimum=min(raw) if raw else None,
        maximum=max(raw) if raw else None,
        **fields,
        unique_count=len(set(raw)),
        min_file=candidates[min_index].path.name if min_index is not None else "",
        max_file=candidates[max_index].path.name if max_index is not None else "",
        finite_count=finite_count,
        positive_infinity_count=raw.count(math.inf),
        negative_infinity_count=raw.count(-math.inf),
        unavailable_count=len(unavailable),
        unavailable_sources=unavailable,
        metric_status=statuses,
    )


def _is_varying(summary: SeriesSummary) -> bool:
    return (
        summary.unique_count > 1
        and summary.span is not None
        and abs(summary.span) > 0.0
    )


def _metric_names_for_mode(
    raw: str,
    *,
    all_metric_names: Sequence[str],
    scoring_metrics: Sequence[str],
) -> List[str]:
    mode = str(raw or "scoring").strip().lower()
    if mode == "none":
        return []
    if mode == "scoring":
        return [metric for metric in scoring_metrics if metric in all_metric_names]
    if mode == "all":
        return list(all_metric_names)
    return _match_names(_split_csv(raw), all_metric_names)


def _param_names_for_mode(
    raw: str,
    *,
    all_param_names: Sequence[str],
    summaries_by_name: Mapping[str, SeriesSummary],
) -> List[str]:
    mode = str(raw or "varying").strip().lower()
    if mode == "none":
        return []
    if mode == "all":
        return list(all_param_names)
    if mode == "varying":
        return [name for name in all_param_names if name in summaries_by_name and _is_varying(summaries_by_name[name])]
    return _match_names(_split_csv(raw), all_param_names)


def _summary_sort_key(summary: SeriesSummary) -> tuple[int, float, float, str]:
    return (
        0 if _is_varying(summary) else 1,
        -summary.cv_abs if math.isfinite(summary.cv_abs) else -1e300,
        -abs(summary.span) if summary.span is not None else 0.0,
        summary.name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto-analyze",
        description="Analyze config-parameter and metric distributions across a Pareto front.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help=(
            "Pareto directory or optimization run directory. Defaults to the latest "
            "optimize_results/<run>/pareto with JSON candidates."
        ),
    )
    parser.add_argument(
        "--params",
        default="varying",
        help=(
            "Config params to summarize: varying, all, none, or comma-separated names/globs. "
            "Default: varying."
        ),
    )
    parser.add_argument(
        "--metrics",
        default="scoring",
        help=(
            "Backtest metrics to summarize: scoring, all, none, or comma-separated names/globs. "
            "Default: scoring."
        ),
    )
    parser.add_argument(
        "--show",
        type=int,
        default=DEFAULT_TOP,
        help=f"Rows to show per group in text output. Default: {DEFAULT_TOP}.",
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
        "--corr",
        type=int,
        default=12,
        help="Show top absolute Pearson correlations between selected params and metrics. Default: 12.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Write params.csv, metrics.csv, correlations.csv, metric_correlations.csv, "
            "summary.json, and optional plots here."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write histogram and metric-vs-param scatter PNGs. Implies --output-dir if omitted.",
    )
    parser.add_argument(
        "--plot-max",
        type=int,
        default=20,
        help="Maximum number of fields to plot. Default: 20.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON to stdout.")
    return parser


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "undefined"
    if not math.isfinite(float(value)):
        return str(value)
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.5f}"
    return f"{value:.6g}"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    if not rows:
        return []
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [
        border,
        "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |",
        border,
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |")
    lines.append(border)
    return lines


def _summary_rows(summaries: Sequence[SeriesSummary], *, limit: int) -> List[List[str]]:
    rows: List[List[str]] = []
    for summary in sorted(summaries, key=lambda item: item.name)[: max(0, int(limit))]:
        rows.append(
            [
                summary.name,
                str(summary.count),
                str(summary.missing),
                str(summary.unique_count),
                _format_number(summary.minimum),
                _format_number(summary.p25),
                _format_number(summary.median),
                _format_number(summary.mean),
                _format_number(summary.p75),
                _format_number(summary.maximum),
                _format_number(summary.std),
            ]
        )
    return rows


def _series_array(
    candidates: Sequence[ParetoCandidate],
    value_maps: Sequence[Mapping[str, float]],
    name: str,
) -> np.ndarray:
    values = []
    for mapping in value_maps:
        value = mapping.get(name)
        values.append(float(value) if _is_number(value) else math.nan)
    return np.array(values, dtype=float)


def compute_correlations(
    *,
    candidates: Sequence[ParetoCandidate],
    param_maps: Sequence[Mapping[str, float]],
    metric_maps: Sequence[Mapping[str, float]],
    param_names: Sequence[str],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    del candidates
    rows: List[Dict[str, Any]] = []
    for param in param_names:
        x = _series_array([], param_maps, param)
        for metric in metric_names:
            y = _series_array([], metric_maps, metric)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 3:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            if float(np.std(x_valid)) <= 0.0 or float(np.std(y_valid)) <= 0.0:
                continue
            corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
            if math.isfinite(corr):
                rows.append({"param": param, "metric": metric, "corr": corr, "abs_corr": abs(corr), "count": int(mask.sum())})
    return sorted(rows, key=lambda item: (-float(item["abs_corr"]), str(item["param"]), str(item["metric"])))


def compute_metric_correlations(
    *,
    metric_maps: Sequence[Mapping[str, float]],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, metric_a in enumerate(metric_names):
        x = _series_array([], metric_maps, metric_a)
        for metric_b in metric_names[idx + 1 :]:
            y = _series_array([], metric_maps, metric_b)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 3:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            if float(np.std(x_valid)) <= 0.0 or float(np.std(y_valid)) <= 0.0:
                continue
            corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
            if math.isfinite(corr):
                rows.append(
                    {
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "corr": corr,
                        "abs_corr": abs(corr),
                        "count": int(mask.sum()),
                    }
                )
    return sorted(rows, key=lambda item: (-float(item["abs_corr"]), str(item["metric_a"]), str(item["metric_b"])))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(
    *,
    output_dir: Path,
    candidates: Sequence[ParetoCandidate],
    param_maps: Sequence[Mapping[str, float]],
    metric_maps: Sequence[Mapping[str, float]],
    param_summaries: Sequence[SeriesSummary],
    metric_summaries: Sequence[SeriesSummary],
    correlations: Sequence[Mapping[str, Any]],
    plot_max: int,
) -> List[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Plotting requires matplotlib from the full install.") from exc

    del candidates
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    selected = list(metric_summaries) + list(param_summaries)
    selected = sorted(selected, key=_summary_sort_key)[: max(0, int(plot_max))]
    maps_by_group = {"metric": metric_maps, "param": param_maps}
    for summary in selected:
        values = _series_array([], maps_by_group[summary.group], summary.name)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(values, bins=min(40, max(5, int(math.sqrt(values.size)))))
        ax.set_title(f"{summary.group}: {summary.name}")
        ax.set_xlabel(summary.name)
        ax.set_ylabel("candidates")
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in summary.name)
        path = plot_dir / f"{summary.group}_{safe_name}_hist.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    for corr in list(correlations)[: min(6, max(0, int(plot_max)))]:
        param = str(corr["param"])
        metric = str(corr["metric"])
        x = _series_array([], param_maps, param)
        y = _series_array([], metric_maps, metric)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x[mask], y[mask], alpha=0.75)
        ax.set_title(f"corr={float(corr['corr']):.3f}")
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        fig.tight_layout()
        safe_param = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in param)
        safe_metric = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in metric)
        path = plot_dir / f"scatter_{safe_param}__{safe_metric}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def analyze_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    raw_path = getattr(args, "path", None)
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

    param_maps, metric_maps = _collect_value_maps(filtered)
    all_param_names = _all_names(param_maps)
    all_metric_names = _all_names(metric_maps)

    all_param_summaries = {
        name: summary
        for name in all_param_names
        if (summary := summarize_series(group="param", name=name, candidates=filtered, value_maps=param_maps))
    }
    param_names = _param_names_for_mode(
        getattr(args, "params", "varying"),
        all_param_names=all_param_names,
        summaries_by_name=all_param_summaries,
    )
    scoring_metrics = [spec.metric for spec in scoring_specs]
    metric_names = _metric_names_for_mode(
        getattr(args, "metrics", "scoring"),
        all_metric_names=all_metric_names,
        scoring_metrics=scoring_metrics,
    )

    param_summaries = [all_param_summaries[name] for name in param_names if name in all_param_summaries]
    metric_summaries = [
        summary
        for name in metric_names
        if (summary := summarize_series(group="metric", name=name, candidates=filtered, value_maps=metric_maps))
    ]
    correlations = compute_correlations(
        candidates=filtered,
        param_maps=param_maps,
        metric_maps=metric_maps,
        param_names=param_names,
        metric_names=metric_names,
    )
    metric_correlations = compute_metric_correlations(
        metric_maps=metric_maps,
        metric_names=[summary.name for summary in metric_summaries],
    )

    output_dir_arg = getattr(args, "output_dir", None)
    output_dir = Path(output_dir_arg).expanduser() if output_dir_arg else None
    if getattr(args, "plots", False) and output_dir is None:
        output_dir = pareto_dir.parent / "pareto_analysis"
    written_plots: List[str] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "params.csv", [summary.to_dict() for summary in param_summaries])
        _write_csv(output_dir / "metrics.csv", [summary.to_dict() for summary in metric_summaries])
        _write_csv(output_dir / "correlations.csv", correlations)
        _write_csv(output_dir / "metric_correlations.csv", metric_correlations)
        if getattr(args, "plots", False):
            written_plots = _plot_outputs(
                output_dir=output_dir,
                candidates=filtered,
                param_maps=param_maps,
                metric_maps=metric_maps,
                param_summaries=param_summaries,
                metric_summaries=metric_summaries,
                correlations=correlations,
                plot_max=int(getattr(args, "plot_max", 20) or 20),
            )

    payload: Dict[str, Any] = {
        "pareto_dir": str(pareto_dir),
        "loaded_count": len(candidates),
        "retained_count": len(filtered),
        "applied_limits": active_limits,
        "scoring_metrics": scoring_metrics,
        "selected_params": param_names,
        "selected_metrics": metric_names,
        "params": [summary.to_dict() for summary in param_summaries],
        "metrics": [summary.to_dict() for summary in metric_summaries],
        "correlations": correlations,
        "metric_correlations": metric_correlations,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "plots": written_plots,
        "distribution_note": "Distributions retain genuine infinite endpoints and explicit unavailable counts. Correlations and plots use finite samples only; correlation rows report their admitted pair counts.",
    }
    if output_dir is not None:
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def format_analysis(payload: Mapping[str, Any], *, show: int, corr_limit: int) -> str:
    lines = [
        "Pareto Distribution Analysis",
        f"Pareto dir: {payload['pareto_dir']}",
        f"Candidates: retained {payload['retained_count']} / loaded {payload['loaded_count']}",
        "Scoring metrics: " + ", ".join(payload.get("scoring_metrics") or []),
    ]
    if payload.get("distribution_note"):
        lines.append(payload["distribution_note"])
    for summary in payload.get("metrics", []):
        if summary.get("positive_infinity_count") or summary.get("negative_infinity_count") or summary.get("unavailable_count"):
            lines.append(f"Metric support {summary['name']}: finite={summary.get('finite_count', 0)}, +infinity={summary.get('positive_infinity_count', 0)}, -infinity={summary.get('negative_infinity_count', 0)}, unavailable={summary.get('unavailable_count', 0)}")
    if payload.get("output_dir"):
        lines.append(f"Output dir: {payload['output_dir']}")
    if payload.get("plots"):
        lines.append(f"Plots: {len(payload['plots'])} PNG files")

    metric_summaries = [SeriesSummary(**_summary_to_constructor_args(item)) for item in payload.get("metrics", [])]
    param_summaries = [SeriesSummary(**_summary_to_constructor_args(item)) for item in payload.get("params", [])]

    if metric_summaries:
        lines.extend(["", "Metric Distributions"])
        rows = _summary_rows(metric_summaries, limit=show)
        lines.extend(_render_table(["metric", "n", "miss", "uniq", "min", "p25", "median", "mean", "p75", "max", "std"], rows))
        visible_metric_names = {row[0] for row in rows}
        metric_correlations = [
            item
            for item in payload.get("metric_correlations", []) or []
            if str(item["metric_a"]) in visible_metric_names and str(item["metric_b"]) in visible_metric_names
        ][: max(0, int(corr_limit))]
        if metric_correlations:
            lines.extend(["", "Metric/Metric Correlations"])
            corr_rows = [
                [
                    str(item["metric_a"]),
                    str(item["metric_b"]),
                    _format_number(float(item["corr"])),
                    str(item["count"]),
                ]
                for item in metric_correlations
            ]
            lines.extend(_render_table(["metric_a", "metric_b", "corr", "n"], corr_rows))
    if param_summaries:
        lines.extend(["", "Config Parameter Distributions"])
        rows = _summary_rows(param_summaries, limit=show)
        lines.extend(_render_table(["param", "n", "miss", "uniq", "min", "p25", "median", "mean", "p75", "max", "std"], rows))

    correlations = list(payload.get("correlations") or [])[: max(0, int(corr_limit))]
    if correlations:
        lines.extend(["", "Top Param/Metric Correlations"])
        rows = [
            [
                str(item["param"]),
                str(item["metric"]),
                _format_number(float(item["corr"])),
                str(item["count"]),
            ]
            for item in correlations
        ]
        lines.extend(_render_table(["param", "metric", "corr", "n"], rows))
    return "\n".join(lines)


def _summary_to_constructor_args(item: Mapping[str, Any]) -> Dict[str, Any]:
    def numeric(key):
        if item[key] is None and item.get("metric_status", {}).get(key, {}).get(
            "status"
        ) not in {"positive_infinity", "negative_infinity"}:
            return None
        return decode_metric_value(item, key)

    result = {
        "group": str(item["group"]),
        "name": str(item["name"]),
        "count": int(item["count"]),
        "missing": int(item["missing"]),
        "minimum": numeric("min"),
        "maximum": numeric("max"),
        "unique_count": int(item["unique_count"]),
        "min_file": str(item["min_file"]),
        "max_file": str(item["max_file"]),
    }
    result.update(
        {
            key: numeric(key)
            for key in ("mean", "median", "std", "p05", "p25", "p75", "p95")
        }
    )
    result.update(
        {
            key: item.get(key, 0)
            for key in (
                "finite_count",
                "positive_infinity_count",
                "negative_infinity_count",
                "unavailable_count",
            )
        }
    )
    result.update(
        unavailable_sources=item.get("unavailable_sources", {}),
        metric_status=item.get("metric_status", {}),
    )
    return result


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    payload = analyze_from_args(args)
    if getattr(args, "json_output", False):
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            format_analysis(
                payload,
                show=int(getattr(args, "show", DEFAULT_TOP) or DEFAULT_TOP),
                corr_limit=int(getattr(args, "corr", 12) or 0),
            )
        )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)
    return 0
