"""Export an offline Pareto explorer with metric selection and interactive limits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence
import webbrowser

from config.metrics import (
    CORE_METRIC_CATALOG,
    CORE_METRIC_DEFINITION_VERSION,
    reject_retired_metric,
    split_metric_stat_suffix,
)
from config.scoring import ObjectiveSpec, default_objective_goal
from metrics_schema import (
    encode_metric_values,
    copy_metric_value,
    require_metric_definition,
)
from pareto_explorer import NoParetoCandidatesError, ParetoCandidate, load_candidates
from tools.pareto_plot_page import PAGE, SCRIPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto-plot",
        description="Export all saved metrics to an interactive, offline Pareto HTML explorer.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Pareto directory, optimizer run directory, or candidate JSON (default: latest optimize_results/<run>/pareto with candidates)",
    )
    parser.add_argument(
        "metrics",
        nargs="*",
        metavar="METRIC",
        help="Optional initial X Y [Z] metrics; choose axes in the HTML",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and ideal directions",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="HTML output (default: pareto_plots/<input-name>.html)",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the saved plot in the default browser"
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing HTML output file"
    )
    return parser


def _plot_value(payload: dict, key: str) -> tuple[float | None, dict | None]:
    """Retain nonfinite fact statuses without inventing plot coordinates."""
    encoded = copy_metric_value(payload, key)
    return encoded[key], encoded.get("metric_status", {}).get(key)


def build_dataset(
    candidates: Sequence[ParetoCandidate], specs: Sequence[ObjectiveSpec]
) -> dict:
    """Export current raw facts with objective > aggregate > mean precedence."""
    if not candidates:
        raise ValueError("No Pareto candidates to plot.")
    goals = {spec.metric: spec.goal for spec in specs}
    columns, statuses, metadata = {}, {}, {}
    for row, candidate in enumerate(candidates):
        require_metric_definition(
            candidate.entry.get("metrics") or candidate.entry.get("suite_metrics", {})
        )
        values = {}
        for flat_key in candidate.stats_flat:
            if flat_key == "metric_status":
                continue
            metric, stat = split_metric_stat_suffix(flat_key)
            reject_retired_metric(metric, path="plot statistic")
            if stat is None:
                raise ValueError(
                    f"Plot statistic {flat_key!r} requires an explicit outer reducer"
                )
            key = f"stats.{metric}.{stat}"
            values[key] = _plot_value(candidate.stats_flat, flat_key)
            metadata[key] = (
                metric,
                f"{metric} [{stat}]",
                (
                    None
                    if stat == "std"
                    else goals.get(metric, default_objective_goal(metric))
                ),
                "Statistics",
            )
            if stat == "mean":
                values[metric] = values[key]
        for metric in candidate.aggregated_values:
            if metric == "metric_status":
                continue
            reject_retired_metric(metric, path="plot aggregate")
            values[metric] = _plot_value(candidate.aggregated_values, metric)
        # Explorer objective values are already decoded raw facts, including infinities.
        for metric, value in candidate.objectives.items():
            reject_retired_metric(metric, path="plot objective")
            values[metric] = _plot_value(encode_metric_values({metric: value}), metric)
        for key, (value, status) in values.items():
            if key not in columns:
                columns[key], statuses[key] = [None] * len(candidates), [None] * len(
                    candidates
                )
            columns[key][row], statuses[key][row] = value, status
            if key not in metadata:
                metadata[key] = (
                    key,
                    key,
                    goals.get(key, default_objective_goal(key)),
                    "Objectives" if key in goals else "Metrics",
                )
    order = [spec.metric for spec in specs] + sorted(set(columns) - set(goals))
    metrics = []
    for key in order:
        values = columns[key]
        finite = [value for value in values if value is not None]
        base, label, goal, group = metadata[key]
        metric = dict(
            key=key,
            label=label,
            goal=goal,
            group=group,
            unit=CORE_METRIC_CATALOG[base]["unit"],
            values=values,
            count=len(finite),
            min=min(finite) if finite else None,
            max=max(finite) if finite else None,
        )
        if any(status is not None for status in statuses[key]):
            metric["statuses"] = statuses[key]
        metrics.append(metric)
    return dict(
        definition_version=CORE_METRIC_DEFINITION_VERSION,
        names=[candidate.path.name for candidate in candidates],
        metrics=metrics,
    )


def select_metrics(requested: Sequence[str], dataset: dict) -> list[str]:
    available = {
        metric["key"]: metric for metric in dataset["metrics"] if metric["count"]
    }
    names = [name.strip() for name in requested] if requested else list(available)[:2]
    if len(names) not in (2, 3):
        raise ValueError(
            "Choose two or three initial metrics, or omit metrics to select them in the HTML."
        )
    if len(set(names)) != len(names):
        raise ValueError("Choose distinct metrics for each axis.")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown or unavailable metric(s): {', '.join(unknown)}; use --list-metrics."
        )
    return names


def render_html(dataset: dict, selected: Sequence[str]) -> str:
    from plotly.offline import get_plotlyjs

    payload = json.dumps(
        {**dataset, "selected": list(selected)}, allow_nan=False, ensure_ascii=True
    )
    # Data is inert JSON, never executable markup, including hostile filenames/metric names.
    payload = (
        payload.replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e")
    )
    return (
        PAGE.replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__APP_SCRIPT__", SCRIPT)
        .replace("__DATA__", payload)
    )


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
        parser.error(
            "Choose two or three initial metrics, or omit metrics to select them in the HTML."
        )
    try:
        pareto_dir, candidates, scoring_specs = load_plot_candidates(args.path)
        dataset = build_dataset(candidates, scoring_specs)
        if args.list_metrics:
            for metric in dataset["metrics"]:
                print(
                    f"{metric['key']} ({metric['goal'] or 'choose direction in HTML'}; {metric['count']}/{len(candidates)} values)"
                )
            return 0
        selected = select_metrics(args.metrics, dataset)
        source = Path(args.path).expanduser().resolve() if args.path else pareto_dir
        if source.is_file():
            name = source.stem
        elif source.name == "pareto":
            name = source.parent.name
        else:
            name = source.name
        output = (
            (args.output or Path("pareto_plots") / f"{name}.html")
            .expanduser()
            .absolute()
        )
        if output.resolve().suffix.lower() not in (".html", ".htm"):
            raise ValueError("Output must have an .html or .htm extension.")
        if output.exists() and any(
            output.samefile(candidate.path) for candidate in candidates
        ):
            raise ValueError("Output must not overwrite a Pareto candidate.")
        if output.exists() and not args.force:
            raise FileExistsError(
                f"Output already exists: {output}; choose another path or use --force."
            )
        page = render_html(dataset, selected)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(page)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"Exported {len(candidates)} candidates and {len(dataset['metrics'])} metrics from {pareto_dir}"
    )
    print(f"HTML: {output}")
    if args.open and not webbrowser.open(output.as_uri()):
        print("Could not open a browser automatically; open the HTML file manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
