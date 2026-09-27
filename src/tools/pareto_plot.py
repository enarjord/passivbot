"""Plot saved Pareto objectives as a standalone, offline 2D or 3D HTML scatter."""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Sequence
import webbrowser

from config.metrics import canonicalize_metric_name
from config.scoring import ObjectiveSpec
from pareto_explorer import ParetoCandidate, load_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool pareto-plot",
        description="Plot two or three saved Pareto objectives in an interactive, offline HTML file.",
    )
    parser.add_argument("path", help="Pareto directory, optimizer run directory, or candidate JSON")
    parser.add_argument("metrics", nargs="*", metavar="METRIC", help="Objective names in X Y [Z] order")
    parser.add_argument("--list-metrics", action="store_true", help="List available objectives and goals")
    parser.add_argument("-o", "--output", type=Path, help="HTML output (default: pareto-plot-2d/3d.html)")
    parser.add_argument("--open", action="store_true", help="Open the saved plot in the default browser")
    parser.add_argument("--force", action="store_true", help="Replace an existing HTML output file")
    return parser


def select_metrics(requested: Sequence[str], specs: Sequence[ObjectiveSpec]) -> list[ObjectiveSpec]:
    if len(requested) not in (2, 3):
        raise ValueError("Choose exactly two or three objective metrics in X Y [Z] order.")
    names = [canonicalize_metric_name(name.strip()) for name in requested]
    if len(set(names)) != len(names):
        raise ValueError("Choose distinct objective metrics for each axis.")
    available = {spec.metric: spec for spec in specs}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown objective metric(s): {', '.join(unknown)}. "
            f"Available objectives: {', '.join(available)}"
        )
    return [available[name] for name in names]


def _axis_title(spec: ObjectiveSpec) -> str:
    # Break long native metric names at underscores without hiding the exact name.
    words = spec.metric.split("_")
    lines = [words[0]]
    for word in words[1:]:
        if len(lines[-1]) + len(word) < 30:
            lines[-1] += "_" + word
        else:
            lines[-1] += "_"
            lines.append(word)
    direction = "higher is better" if spec.goal == "max" else "lower is better"
    return "<br>".join(html.escape(line) for line in lines) + f"<br>({direction})"


def build_figure(candidates: Sequence[ParetoCandidate], specs: Sequence[ObjectiveSpec]):
    # Keep CLI help and metric discovery usable without importing the rendering dependency.
    import plotly.graph_objects as go

    if len(specs) not in (2, 3) or len({spec.metric for spec in specs}) != len(specs):
        raise ValueError("Choose two or three distinct objective metrics.")
    if not candidates:
        raise ValueError("No Pareto candidates to plot.")
    columns = [[] for _ in specs]
    for candidate in candidates:
        for column, spec in zip(columns, specs):
            value = candidate.objectives.get(spec.metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing or non-finite objective {spec.metric!r} in {candidate.path.name}")
            column.append(float(value))

    is_3d = len(specs) == 3
    hover = "<b>%{text}</b><br>" + "<br>".join(
        f"{html.escape(spec.metric)}: %{{{axis}:.8g}}" for axis, spec in zip("xyz", specs)
    ) + "<extra></extra>"
    trace_args = dict(
        x=columns[0], y=columns[1], mode="markers",
        text=[html.escape(candidate.path.name) for candidate in candidates],
        hovertemplate=hover,
        marker=dict(
            size=5 if is_3d else 9, opacity=0.85,
            color=columns[-1], colorscale="Viridis", showscale=False,
            line=dict(width=0.5, color="#ffffff"),
        ),
    )
    trace = go.Scatter3d(z=columns[2], **trace_args) if is_3d else go.Scatter(**trace_args)
    fig = go.Figure(trace)
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>Pareto trade-offs</b> · {len(candidates):,} candidates<br><sup>" +
                 ("Drag to rotate · Scroll to zoom · Hover to inspect" if is_3d else
                  "Drag to zoom · Double-click to reset · Hover to inspect") + "</sup>",
            x=0.04, xanchor="left", y=0.97, yanchor="top", font=dict(size=22),
        ),
        font=dict(family="Arial, sans-serif", size=13, color="#243449"),
        paper_bgcolor="#f6f8fb", plot_bgcolor="#ffffff",
        margin=dict(l=120, r=70, t=130, b=155),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12),
        annotations=[dict(
            text="Saved objective values (may include penalties). All saved members shown;<br>"
                 "a projection can contain dominated points. Color follows the " + ("Z" if is_3d else "Y") + " axis.",
            x=0, y=0, yshift=-115, yanchor="top", xref="paper", yref="paper", showarrow=False,
            xanchor="left", align="left", font=dict(size=11, color="#65758b"),
        )],
    )
    axis_style = dict(gridcolor="#e1e7ef", zeroline=False, tickformat=".4~g")
    if is_3d:
        fig.update_layout(scene=dict(
            **{f"{axis}axis": dict(title=dict(text=_axis_title(spec), font=dict(size=11)),
                                  **axis_style) for axis, spec in zip("xyz", specs)},
            aspectmode="cube", dragmode="orbit",
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.2)),
        ))
    else:
        fig.update_xaxes(title=dict(text=_axis_title(specs[0]), standoff=18), automargin=True, **axis_style)
        fig.update_yaxes(title=dict(text=_axis_title(specs[1]), standoff=18), automargin=True, **axis_style)
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_metrics and len(args.metrics) not in (2, 3):
        parser.error("Choose exactly two or three objective metrics in X Y [Z] order.")
    try:
        pareto_dir, candidates, scoring_specs = load_candidates(args.path)
        if args.list_metrics:
            for spec in scoring_specs:
                print(f"{spec.metric} ({spec.goal})")
            return 0
        specs = select_metrics(args.metrics, scoring_specs)
        output = (args.output or Path(f"pareto-plot-{len(specs)}d.html")).expanduser().absolute()
        if output.resolve().suffix.lower() not in (".html", ".htm"):
            raise ValueError("Output must have an .html or .htm extension.")
        # Resolve symlinks too: plotting must never overwrite an input artifact.
        if output.resolve() in {candidate.path.resolve() for candidate in candidates}:
            raise ValueError("Output must not overwrite a Pareto candidate.")
        if output.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {output}; choose another path or use --force.")
        figure = build_figure(candidates, specs)
        page = figure.to_html(
            full_html=True, include_plotlyjs=True, default_height="100vh",
            config=dict(responsive=True, scrollZoom=True, displaylogo=False,
                        toImageButtonOptions=dict(format="png", scale=2)),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(page)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Plotted {len(candidates)} candidates from {pareto_dir}")
    print(f"HTML: {output}")
    if args.open:
        if not webbrowser.open(output.as_uri()):
            print("Could not open a browser automatically; open the HTML file manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
