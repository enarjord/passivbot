"""Export an offline Pareto explorer with metric selection and interactive limits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence
import webbrowser

from config.metrics import canonicalize_metric_name
from config.scoring import ObjectiveSpec, default_objective_goal
from pareto_explorer import NoParetoCandidatesError, ParetoCandidate, load_candidates
from tools.pareto_plot_page import PAGE, SCRIPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto-plot",
        description="Export all saved metrics to an interactive, offline Pareto HTML explorer.",
    )
    parser.add_argument("path", nargs="?", help="Pareto directory, optimizer run directory, or candidate JSON (default: latest optimize_results/<run>/pareto with candidates)")
    parser.add_argument("metrics", nargs="*", metavar="METRIC", help="Optional initial X Y [Z] metrics; choose axes in the HTML")
    parser.add_argument("--list-metrics", action="store_true", help="List available metrics and ideal directions")
    parser.add_argument("-o", "--output", type=Path, help="HTML output (default: pareto_plots/<input-name>.html)")
    parser.add_argument("--open", action="store_true", help="Open the saved plot in the default browser")
    parser.add_argument("--force", action="store_true", help="Replace an existing HTML output file")
    return parser


def _finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _canonical_stat_key(key: str) -> str:
    metric, stat = key.rsplit("_", 1)
    return f"{canonicalize_metric_name(metric)}_{stat}"


def _canonical_values(values: dict, key_func=canonicalize_metric_name) -> dict:
    groups = {}
    for key in values:
        groups.setdefault(key_func(key), []).append(key)
    result = {}
    for canonical, aliases in groups.items():
        if canonical in values:
            result[canonical] = values[canonical]
            continue
        value = values[aliases[0]]
        if any(_finite(values[alias]) != _finite(value) for alias in aliases[1:]):
            raise ValueError(f"Conflicting aliases for metric {canonical!r}: {aliases}")
        result[canonical] = value
    return result


def build_dataset(candidates: Sequence[ParetoCandidate], specs: Sequence[ObjectiveSpec]) -> dict:
    """Keep objective > aggregate > mean precedence; expose each statistic separately."""
    if not candidates:
        raise ValueError("No Pareto candidates to plot.")
    goals = {spec.metric: spec.goal for spec in specs}
    columns = {}
    metadata = {}
    for row, candidate in enumerate(candidates):
        values = {}
        for flat_key, value in _canonical_values(candidate.stats_flat, _canonical_stat_key).items():
            metric, stat = flat_key.rsplit("_", 1)
            key = f"stats.{metric}.{stat}"
            values[key] = value
            metadata[key] = (f"{metric} [{stat}]", None if stat == "std" else goals.get(metric, default_objective_goal(metric)), "Statistics")
            if stat == "mean":
                values[metric] = value
        values.update(_canonical_values(candidate.aggregated_values))
        # Named objectives are authoritative, including any saved penalties.
        values.update(candidate.objectives)
        for key, value in values.items():
            if key not in columns:
                columns[key] = [None] * len(candidates)
            columns[key][row] = _finite(value)
            if key not in metadata:
                metadata[key] = (key, goals.get(key, default_objective_goal(key)),
                                 "Objectives" if key in goals else "Metrics")
    order = [spec.metric for spec in specs] + sorted(set(columns) - set(goals))
    metrics = []
    for key in order:
        values = columns[key]
        finite = [value for value in values if value is not None]
        label, goal, group = metadata[key]
        metrics.append(dict(key=key, label=label, goal=goal, group=group, values=values,
                            count=len(finite), min=min(finite) if finite else None,
                            max=max(finite) if finite else None))
    return dict(names=[candidate.path.name for candidate in candidates], metrics=metrics)


def _canonical_selector(name: str) -> str:
    name = name.strip()
    if name.startswith("stats.") and name.count(".") >= 2:
        metric, stat = name[len("stats."):].rsplit(".", 1)
        return f"stats.{canonicalize_metric_name(metric)}.{stat}"
    return canonicalize_metric_name(name)


def select_metrics(requested: Sequence[str], dataset: dict) -> list[str]:
    available = {metric["key"]: metric for metric in dataset["metrics"] if metric["count"]}
    names = [_canonical_selector(name) for name in requested] if requested else list(available)[:2]
    if len(names) not in (2, 3):
        raise ValueError("Choose two or three initial metrics, or omit metrics to select them in the HTML.")
    if len(set(names)) != len(names):
        raise ValueError("Choose distinct metrics for each axis.")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown or unavailable metric(s): {', '.join(unknown)}; use --list-metrics.")
    return names


def render_html(dataset: dict, selected: Sequence[str]) -> str:
    from plotly.offline import get_plotlyjs

    payload = json.dumps({**dataset, "selected": list(selected)}, allow_nan=False, ensure_ascii=True)
    # Data is inert JSON, never executable markup, including hostile filenames/metric names.
    payload = payload.replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e")
    return PAGE.replace("__PLOTLY_JS__", get_plotlyjs()).replace("__APP_SCRIPT__", SCRIPT).replace("__DATA__", payload)


def load_plot_candidates(path: str | None):
    if path is not None:
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
                # Skip empty/sidecar-only fronts; malformed candidates still fail visibly.
                continue
    raise FileNotFoundError(
        "No Pareto path provided and no optimize_results/<run>/pareto directory "
        "with candidate JSON files was found."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_metrics and args.metrics and len(args.metrics) not in (2, 3):
        parser.error("Choose two or three initial metrics, or omit metrics to select them in the HTML.")
    try:
        pareto_dir, candidates, scoring_specs = load_plot_candidates(args.path)
        dataset = build_dataset(candidates, scoring_specs)
        if args.list_metrics:
            for metric in dataset["metrics"]:
                print(f"{metric['key']} ({metric['goal'] or 'choose direction in HTML'}; {metric['count']}/{len(candidates)} values)")
            return 0
        selected = select_metrics(args.metrics, dataset)
        source = Path(args.path).expanduser().resolve() if args.path else pareto_dir
        if source.is_file():
            name = source.stem
        elif source.name == "pareto":
            name = source.parent.name
        else:
            name = source.name
        output = (args.output or Path("pareto_plots") / f"{name}.html").expanduser().absolute()
        if output.resolve().suffix.lower() not in (".html", ".htm"):
            raise ValueError("Output must have an .html or .htm extension.")
        if output.exists() and any(output.samefile(candidate.path) for candidate in candidates):
            raise ValueError("Output must not overwrite a Pareto candidate.")
        if output.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {output}; choose another path or use --force.")
        page = render_html(dataset, selected)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(page)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(candidates)} candidates and {len(dataset['metrics'])} metrics from {pareto_dir}")
    print(f"HTML: {output}")
    if args.open and not webbrowser.open(output.as_uri()):
        print("Could not open a browser automatically; open the HTML file manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
