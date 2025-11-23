import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import re

# Ensure we can import modules from src/
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from opt_utils import load_results
from config_utils import normalize_limit_entries


def discover_runs(root: str) -> List[str]:
    pattern = os.path.join(root, "*", "pareto")
    runs = []
    for path in glob(pattern):
        run_dir = os.path.dirname(path)
        if os.path.isdir(run_dir):
            runs.append(run_dir)
    runs.sort()
    return runs


def _flatten_numeric(source: dict, prefix: str = "") -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, value in source.items():
        if not key:
            continue
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_numeric(value, path))
        elif isinstance(value, (int, float, np.number)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                flattened[path] = numeric
    return flattened


def _flatten_stats_block(stats: Dict[str, Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    if not isinstance(stats, dict):
        return flattened
    for metric, payload in stats.items():
        if not isinstance(payload, dict):
            continue
        for field, value in payload.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            key = f"{metric}_{field}"
            if prefix:
                key = f"{prefix}{key}"
            flattened[key] = numeric
    return flattened


def _ensure_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric):
        return numeric
    return None


def _extract_suite_metrics(
    entry: dict,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[str]]:
    suite_metrics = entry.get("suite_metrics")
    metrics: Dict[str, float] = {}
    scenario_values: Dict[str, Dict[str, float]] = {}
    scenario_labels: List[str] = []
    if not isinstance(suite_metrics, dict):
        return metrics, scenario_values, scenario_labels

    if "metrics" in suite_metrics:
        for metric, payload in suite_metrics["metrics"].items():
            aggregated = _ensure_float(payload.get("aggregated"))
            if aggregated is not None:
                metrics[metric] = aggregated
            stats = payload.get("stats") or {}
            for field, value in stats.items():
                numeric = _ensure_float(value)
                if numeric is not None:
                    metrics[f"{metric}_{field}"] = numeric
            for scenario, value in (payload.get("scenarios") or {}).items():
                numeric = _ensure_float(value)
                if numeric is None:
                    continue
                scenario_values.setdefault(scenario, {})[metric] = numeric
        labels = suite_metrics.get("scenario_labels")
        if isinstance(labels, list):
            scenario_labels = labels
        else:
            scenario_labels = list(scenario_values.keys())
        return metrics, scenario_values, scenario_labels

    aggregate = suite_metrics.get("aggregate", {})
    aggregated = aggregate.get("aggregated", {}) or {}
    stats = aggregate.get("stats", {}) or {}
    for metric, value in aggregated.items():
        numeric = _ensure_float(value)
        if numeric is not None:
            metrics[metric] = numeric
    metrics.update(_flatten_stats_block(stats))
    labels = suite_metrics.get("scenarios")
    if isinstance(labels, list):
        scenario_labels = labels
    return metrics, scenario_values, scenario_labels


def _extract_objectives(entry: dict) -> Dict[str, float]:
    metrics_block = entry.get("metrics") or {}
    objectives = metrics_block.get("objectives") or {}
    flattened = {}
    for key, value in objectives.items():
        numeric = _ensure_float(value)
        if numeric is not None:
            flattened[f"objective_{key}"] = numeric
    if not flattened and isinstance(metrics_block.get("stats"), dict):
        flattened.update(_flatten_stats_block(metrics_block["stats"]))
    return flattened


@dataclass
class RunData:
    dataframe: pd.DataFrame
    scenario_metrics: Dict[str, List[str]]
    scoring_metrics: List[str]
    default_limits: List[str]
    aggregated_metrics: List[str]
    param_metrics: List[str]


def load_pareto_dataframe(run_dir: str) -> RunData:
    pareto_dir = os.path.join(run_dir, "pareto")
    rows: List[Dict[str, float]] = []
    scenario_metric_map: Dict[str, set] = defaultdict(set)
    scoring_metrics: List[str] = []
    default_limits: List[str] = []
    aggregated_cols: set[str] = set()
    param_cols: set[str] = set()
    for path in sorted(glob(os.path.join(pareto_dir, "*.json"))):
        with open(path) as f:
            entry = json.load(f)
        base = {"_id": os.path.basename(path)}
        suite_values, scenario_values, scenario_labels = _extract_suite_metrics(entry)
        params = _flatten_numeric(entry.get("bot", {}), prefix="bot")
        row = {**base, **suite_values, **params}
        row.update(_extract_objectives(entry))
        for key in row:
            if key.startswith("bot."):
                param_cols.add(key)
            elif key != "_id" and not _looks_like_stat_column(key):
                aggregated_cols.add(key)
        for scenario, metric_values in scenario_values.items():
            for metric, value in metric_values.items():
                row[f"{scenario}__{metric}"] = value
                scenario_metric_map[scenario].add(metric)
        for label in scenario_labels:
            scenario_metric_map.setdefault(label, set())
        rows.append(row)
        if not scoring_metrics:
            scoring_metrics = list(entry.get("optimize", {}).get("scoring", []) or [])
            limits_cfg = entry.get("optimize", {}).get("limits", {})
            default_limits = _limits_to_exprs(limits_cfg)
    if not rows:
        return RunData(pd.DataFrame(), {}, scoring_metrics, default_limits, [], [])
    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")
    scenario_metrics = {name: sorted(metrics) for name, metrics in scenario_metric_map.items()}
    return RunData(
        df,
        scenario_metrics,
        scoring_metrics,
        default_limits,
        sorted(aggregated_cols),
        sorted(param_cols),
    )


def load_history_dataframe(run_dir: str, max_points: int = 400) -> pd.DataFrame:
    history_path = os.path.join(run_dir, "all_results.bin")
    if not os.path.exists(history_path):
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    for idx, record in enumerate(load_results(history_path)):
        suite_metrics, scenario_values, _ = _extract_suite_metrics(record)
        metrics_block = record.get("metrics") or {}
        row = {"iteration": idx}
        row.update(suite_metrics)
        if not suite_metrics:
            stats = metrics_block.get("stats")
            if isinstance(stats, dict):
                row.update(_flatten_stats_block(stats))
        row.update(_extract_objectives(record))
        for scenario, metric_values in scenario_values.items():
            for metric, value in metric_values.items():
                row[f"{scenario}__{metric}"] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if len(df) > max_points:
        step = max(len(df) // max_points, 1)
        df = df.iloc[::step]
    return df


def _select_closest_to_ideal(df: pd.DataFrame, metrics: List[str]) -> Optional[pd.Series]:
    if not metrics:
        return None
    subset = df[metrics].copy()
    # Normalize columns to [0,1]; if constant, treat distance as 0 for that metric
    normed = pd.DataFrame(index=subset.index)
    for col in metrics:
        col_min = subset[col].min()
        col_max = subset[col].max()
        if not np.isfinite(col_min) or not np.isfinite(col_max):
            normed[col] = np.nan
            continue
        if col_max > col_min:
            normed[col] = (subset[col] - col_min) / (col_max - col_min)
        else:
            normed[col] = 0.0
    normed = normed.dropna()
    if normed.empty:
        return None
    # Ideal is max (1.0) for each metric
    distances = np.sqrt(((normed - 1.0) ** 2).sum(axis=1))
    best_idx = distances.idxmin()
    return df.loc[best_idx]


LIMIT_PATTERNS = [
    (re.compile(r"^\s*(?P<key>[A-Za-z0-9_.]+)\s*<=\s*(?P<val>[-+eE0-9.]+)\s*$"), np.less_equal),
    (re.compile(r"^\s*(?P<key>[A-Za-z0-9_.]+)\s*>=\s*(?P<val>[-+eE0-9.]+)\s*$"), np.greater_equal),
    (re.compile(r"^\s*(?P<key>[A-Za-z0-9_.]+)\s*<\s*(?P<val>[-+eE0-9.]+)\s*$"), np.less),
    (re.compile(r"^\s*(?P<key>[A-Za-z0-9_.]+)\s*>\s*(?P<val>[-+eE0-9.]+)\s*$"), np.greater),
    (re.compile(r"^\s*(?P<key>[A-Za-z0-9_.]+)\s*==?\s*(?P<val>[-+eE0-9.]+)\s*$"), np.equal),
]


def _apply_limits(df: pd.DataFrame, exprs: Optional[str]) -> pd.Series:
    if not exprs:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for line in str(exprs).splitlines():
        line = line.strip()
        if not line:
            continue
        matched = False
        for pattern, op in LIMIT_PATTERNS:
            m = pattern.match(line)
            if not m:
                continue
            matched = True
            key = m.group("key")
            try:
                val = float(m.group("val"))
            except ValueError:
                continue
            if key not in df.columns:
                continue
            col = df[key]
            mask &= op(col, val)
            break
        if not matched:
            # unsupported expression; ignore
            continue
    return mask


def _looks_like_stat_column(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(("_mean", "_min", "_max", "_std"))


def _limits_to_exprs(limits_cfg: Any) -> List[str]:
    exprs: List[str] = []
    if isinstance(limits_cfg, str):
        exprs.append(limits_cfg)
        return exprs
    try:
        normalized = normalize_limit_entries(limits_cfg)
    except Exception:
        return exprs
    for entry in normalized:
        metric = entry.get("metric")
        mode = entry.get("penalize_if")
        if not metric or not mode:
            continue
        if mode == "greater_than":
            num = _ensure_float(entry.get("value"))
            if num is not None:
                exprs.append(f"{metric}<={num}")
        elif mode == "less_than":
            num = _ensure_float(entry.get("value"))
            if num is not None:
                exprs.append(f"{metric}>={num}")
        elif mode == "outside_range":
            rng = entry.get("range")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                low = _ensure_float(rng[0])
                high = _ensure_float(rng[1])
                if low is not None and high is not None:
                    exprs.append(f"{metric}>={low}")
                    exprs.append(f"{metric}<={high}")
        # inside_range cannot be represented as a simple AND expression; skip.
    return exprs


RUN_CACHE: Dict[str, RunData] = {}
HISTORY_CACHE: Dict[str, pd.DataFrame] = {}


def get_run_data(run_dir: str) -> RunData:
    if run_dir not in RUN_CACHE:
        RUN_CACHE[run_dir] = load_pareto_dataframe(run_dir)
    return RUN_CACHE[run_dir]


def get_history_dataframe(run_dir: str) -> pd.DataFrame:
    if run_dir not in HISTORY_CACHE:
        HISTORY_CACHE[run_dir] = load_history_dataframe(run_dir)
    return HISTORY_CACHE[run_dir]


def serve_dash(data_root: str, port: int = 8050):
    try:
        from dash import Dash, Input, Output, State, dcc, html, dash_table
        import plotly.express as px
    except ImportError as exc:
        raise SystemExit(
            "dash and plotly are required to run the dashboard. Install with `pip install dash plotly`."
        ) from exc

    run_dirs = discover_runs(data_root)
    if not run_dirs:
        raise SystemExit(f"No runs found under {data_root}")

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Pareto Explorer"),
                    html.Label("Run"),
                    dcc.Dropdown(
                        id="run-selection",
                        options=[{"label": os.path.basename(r), "value": r} for r in run_dirs],
                        value=run_dirs[-1],
                        clearable=False,
                    ),
                    html.Label("X metric"),
                    dcc.Dropdown(id="x-metric", placeholder="Select X metric"),
                    html.Label("Y metric"),
                    dcc.Dropdown(id="y-metric", placeholder="Select Y metric"),
                    html.Label("Histogram metric"),
                    dcc.Dropdown(id="hist-metric", placeholder="Select histogram metric"),
                    html.Label("Scenarios"),
                    dcc.Dropdown(
                        id="scenario-selection",
                        placeholder="Scenarios to include",
                        multi=True,
                    ),
                    html.Label("Scenario metric"),
                    dcc.Dropdown(id="scenario-metric", placeholder="Scenario metric"),
                    html.Label("Parameter (X axis for param scatter)"),
                    dcc.Dropdown(id="param-x", placeholder="Parameter (X axis)"),
                    html.Label("Metric (Y axis for param scatter)"),
                    dcc.Dropdown(id="metric-y", placeholder="Metric (Y axis)"),
                    html.Label("Correlation metrics"),
                    dcc.Dropdown(
                        id="correlation-metrics",
                        placeholder="Metrics for correlation heatmap",
                        multi=True,
                    ),
                    html.Label("Ideal metrics (closest config)"),
                    dcc.Dropdown(
                        id="ideal-metrics",
                        placeholder="Metrics to optimize (closest to ideal)",
                        multi=True,
                    ),
                    html.Label("Main plot metrics"),
                    dcc.Dropdown(
                        id="plot-metrics",
                        placeholder="Metrics to visualize (2D/3D/parallel depending on count)",
                        multi=True,
                    ),
                    html.Label("Limits (one per line, e.g. position_held_hours_max<800)"),
                    dcc.Textarea(
                        id="limit-expressions",
                        placeholder="Limits",
                        style={"width": "100%", "height": "80px"},
                    ),
                    html.Button("Download CSV", id="download-data"),
                    dcc.Download(id="download-dataset"),
                ],
                className="controls",
            ),
            dcc.Graph(id="main-plot"),
            dcc.Graph(id="pareto-scatter"),
            dcc.Graph(id="metric-hist"),
            dcc.Graph(id="scenario-comparison"),
            dcc.Graph(id="correlation-heatmap"),
            dcc.Graph(id="param-scatter"),
            dcc.Graph(id="history-line"),
            html.H4("Closest Config (filtered)"),
            dash_table.DataTable(
                id="best-config-table",
                page_size=5,
                style_table={"overflowX": "auto"},
            ),
            dash_table.DataTable(
                id="pareto-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
            ),
        ]
    )

    @app.callback(
        Output("x-metric", "options"),
        Output("y-metric", "options"),
        Output("hist-metric", "options"),
        Output("x-metric", "value"),
        Output("y-metric", "value"),
        Output("hist-metric", "value"),
        Input("run-selection", "value"),
    )
    def update_metric_choices(run_dir):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        numeric_cols_full = [
            c for c in df.columns if c != "_id" and pd.api.types.is_numeric_dtype(df[c])
        ]
        numeric_cols = [
            c
            for c in numeric_cols_full
            if not _looks_like_stat_column(c) and not c.startswith("bot.")
        ]
        options = [{"label": col, "value": col} for col in numeric_cols]
        preferred = [col for col in run_data.scoring_metrics if col in numeric_cols]
        default_x = preferred[0] if preferred else (numeric_cols[0] if numeric_cols else None)
        if len(preferred) > 1:
            default_y = preferred[1]
        elif preferred:
            default_y = preferred[0]
        else:
            default_y = numeric_cols[1] if len(numeric_cols) > 1 else default_x
        default_hist = default_x
        return options, options, options, default_x, default_y, default_hist

    @app.callback(
        Output("scenario-selection", "options"),
        Output("scenario-selection", "value"),
        Output("scenario-metric", "options"),
        Output("scenario-metric", "value"),
        Output("param-x", "options"),
        Output("param-x", "value"),
        Output("metric-y", "options"),
        Output("metric-y", "value"),
        Output("correlation-metrics", "options"),
        Output("correlation-metrics", "value"),
        Output("ideal-metrics", "options"),
        Output("ideal-metrics", "value"),
        Output("limit-expressions", "value"),
        Output("plot-metrics", "options"),
        Output("plot-metrics", "value"),
        Input("run-selection", "value"),
    )
    def update_metadata_controls(run_dir):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        scenario_options = [
            {"label": name, "value": name} for name in sorted(run_data.scenario_metrics)
        ]
        scenario_value = [opt["value"] for opt in scenario_options]
        scenario_metrics_set = set()
        for metrics in run_data.scenario_metrics.values():
            scenario_metrics_set.update(metrics)
        scenario_metric_options = [
            {"label": metric, "value": metric} for metric in sorted(scenario_metrics_set)
        ]
        scenario_metric_value = (
            scenario_metric_options[0]["value"] if scenario_metric_options else None
        )

        metric_cols = [
            c
            for c in run_data.aggregated_metrics
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        param_cols = [
            c
            for c in run_data.param_metrics
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        param_options = [{"label": col, "value": col} for col in param_cols]
        metric_options = [{"label": col, "value": col} for col in metric_cols]
        param_value = param_cols[0] if param_cols else None
        metric_value = metric_cols[0] if metric_cols else None

        corr_default = [opt["value"] for opt in metric_options[:5]]
        ideal_default = [
            m for m in run_data.scoring_metrics if m in {opt["value"] for opt in metric_options}
        ]
        limits_default = "\n".join(run_data.default_limits)
        plot_default = ideal_default if ideal_default else metric_cols[: min(3, len(metric_cols))]
        return (
            scenario_options,
            scenario_value,
            scenario_metric_options,
            scenario_metric_value,
            param_options,
            param_value,
            metric_options,
            metric_value,
            metric_options,
            corr_default,
            metric_options,
            ideal_default,
            limits_default,
            metric_options,
            plot_default,
        )

    @app.callback(
        Output("main-plot", "figure"),
        Output("pareto-scatter", "figure"),
        Output("metric-hist", "figure"),
        Output("pareto-table", "data"),
        Output("pareto-table", "columns"),
        Output("history-line", "figure"),
        Output("best-config-table", "data"),
        Output("best-config-table", "columns"),
        Input("run-selection", "value"),
        Input("x-metric", "value"),
        Input("y-metric", "value"),
        Input("hist-metric", "value"),
        Input("ideal-metrics", "value"),
        Input("limit-expressions", "value"),
        Input("plot-metrics", "value"),
    )
    def update_figures(
        run_dir, x_metric, y_metric, hist_metric, ideal_metrics, limit_exprs, plot_metrics
    ):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if df.empty:
            empty_fig = px.scatter()
            return empty_fig, empty_fig, empty_fig, [], [], empty_fig, [], [], empty_fig

        # Apply limits once for downstream consumers (main plot + closest config)
        limit_mask = _apply_limits(df, limit_exprs)
        df_limited = df.loc[limit_mask]

        scatter_df = df.dropna(subset=[x_metric, y_metric]) if x_metric and y_metric else df
        scatter_fig = px.scatter(
            scatter_df,
            x=x_metric,
            y=y_metric,
            hover_data=["_id"],
        )

        hist_fig = px.histogram(df, x=hist_metric) if hist_metric else px.histogram()

        table_columns = [{"name": col, "id": col} for col in df.columns[:20]]
        table_data = df.head(100).to_dict("records")

        history_df = get_history_dataframe(run_dir)
        if not history_df.empty and hist_metric in history_df.columns:
            history_fig = px.line(history_df, x="iteration", y=hist_metric)
        else:
            history_fig = px.line()

        main_fig = px.scatter()
        plot_list = plot_metrics or []
        plot_list = [m for m in plot_list if m in df.columns]
        if plot_list:
            filtered = df_limited.dropna(subset=plot_list)
            if len(plot_list) >= 4:
                main_fig = px.parallel_coordinates(filtered, dimensions=plot_list, color=plot_list[0])
            elif len(plot_list) == 3:
                main_fig = px.scatter_3d(
                    filtered,
                    x=plot_list[0],
                    y=plot_list[1],
                    z=plot_list[2],
                    hover_data=["_id"],
                )
            elif len(plot_list) == 2:
                main_fig = px.scatter(filtered, x=plot_list[0], y=plot_list[1], hover_data=["_id"])
            elif len(plot_list) == 1:
                main_fig = px.scatter(filtered, x=plot_list[0], y=plot_list[0], hover_data=["_id"])

        # compute best config based on ideal metrics and limits
        best_data: List[Dict[str, float]] = []
        best_columns: List[Dict[str, str]] = []
        if ideal_metrics:
            filtered = df_limited.dropna(subset=ideal_metrics)
            if filtered.shape[0] > 0:
                best_row = _select_closest_to_ideal(filtered, ideal_metrics)
                if best_row is not None:
                    sanitized = {k: v for k, v in best_row.items() if not _looks_like_stat_column(k)}
                    best_data = [sanitized]
                    best_columns = [{"name": k, "id": k} for k in sanitized.keys()]

        return (
            main_fig,
            scatter_fig,
            hist_fig,
            table_data,
            table_columns,
            history_fig,
            best_data,
            best_columns,
        )

    @app.callback(
        Output("scenario-comparison", "figure"),
        Input("run-selection", "value"),
        Input("scenario-selection", "value"),
        Input("scenario-metric", "value"),
    )
    def update_scenario_chart(run_dir, selected_scenarios, metric_name):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if not metric_name or df.empty:
            return px.box()
        if not selected_scenarios:
            selected_scenarios = list(run_data.scenario_metrics.keys())
        if isinstance(selected_scenarios, str):
            selected_scenarios = [selected_scenarios]
        records = []
        for scenario in selected_scenarios:
            column = f"{scenario}__{metric_name}"
            if column not in df:
                continue
            series = df[column].dropna()
            for value in series:
                records.append({"scenario": scenario, "value": value})
        if not records:
            return px.box()
        plot_df = pd.DataFrame(records)
        fig = px.box(plot_df, x="scenario", y="value", points="all")
        fig.update_layout(yaxis_title=metric_name)
        return fig

    @app.callback(
        Output("correlation-heatmap", "figure"),
        Input("run-selection", "value"),
        Input("correlation-metrics", "value"),
    )
    def update_correlation(run_dir, selected_metrics):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if not selected_metrics or len(selected_metrics) < 2:
            return px.imshow([[0]], labels=dict(x="", y="", color="corr"))
        available = [col for col in selected_metrics if col in df.columns]
        if len(available) < 2:
            return px.imshow([[0]], labels=dict(x="", y="", color="corr"))
        corr_df = df[available].dropna()
        if corr_df.empty:
            return px.imshow([[0]], labels=dict(x="", y="", color="corr"))
        matrix = corr_df.corr()
        return px.imshow(
            matrix,
            x=matrix.columns,
            y=matrix.columns,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )

    @app.callback(
        Output("param-scatter", "figure"),
        Input("run-selection", "value"),
        Input("param-x", "value"),
        Input("metric-y", "value"),
    )
    def update_param_scatter(run_dir, param_col, metric_col):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if not param_col or not metric_col or param_col not in df or metric_col not in df:
            return px.scatter()
        scatter_df = df.dropna(subset=[param_col, metric_col])
        if scatter_df.empty:
            return px.scatter()
        fig = px.scatter(scatter_df, x=param_col, y=metric_col, hover_data=["_id"])
        fig.update_layout(xaxis_title=param_col, yaxis_title=metric_col)
        return fig

    @app.callback(
        Output("download-dataset", "data"),
        Input("download-data", "n_clicks"),
        State("run-selection", "value"),
        prevent_initial_call=True,
    )
    def trigger_download(n_clicks, run_dir):
        run_data = get_run_data(run_dir)
        csv_data = run_data.dataframe.to_csv(index=False)
        filename = f"{os.path.basename(run_dir)}.csv"
        return dict(content=csv_data, filename=filename)

    app.run(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description="Pareto dashboard")
    parser.add_argument(
        "--data-root",
        default="optimize_results",
        help="Directory containing optimize_results",
    )
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    serve_dash(args.data_root, port=args.port)


if __name__ == "__main__":
    main()
