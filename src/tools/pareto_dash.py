"""Pareto Dashboard - Interactive explorer for optimization results.

Usage:
    python src/tools/pareto_dash.py --data-root optimize_results --port 8050
"""

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
    raw_configs: Dict[str, dict]  # _id -> full JSON config


def load_pareto_dataframe(run_dir: str) -> RunData:
    pareto_dir = os.path.join(run_dir, "pareto")
    rows: List[Dict[str, float]] = []
    scenario_metric_map: Dict[str, set] = defaultdict(set)
    scoring_metrics: List[str] = []
    default_limits: List[str] = []
    aggregated_cols: set[str] = set()
    param_cols: set[str] = set()
    raw_configs: Dict[str, dict] = {}

    for path in sorted(glob(os.path.join(pareto_dir, "*.json"))):
        with open(path) as f:
            entry = json.load(f)
        config_id = os.path.basename(path)
        raw_configs[config_id] = entry
        base = {"_id": config_id}
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
        return RunData(pd.DataFrame(), {}, scoring_metrics, default_limits, [], [], {})
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
        raw_configs,
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


def _extract_limit_metrics(exprs: Iterable[str]) -> List[str]:
    metrics: List[str] = []
    if not exprs:
        return metrics
    for line in exprs:
        if not line:
            continue
        for pattern, _ in LIMIT_PATTERNS:
            m = pattern.match(line)
            if m:
                key = m.group("key")
                if key not in metrics:
                    metrics.append(key)
                break
    return metrics


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
        from dash import Dash, Input, Output, State, dcc, html, dash_table, callback_context, no_update
        import dash_bootstrap_components as dbc
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Required packages missing. Install with:\n"
            "  pip install dash plotly dash-bootstrap-components"
        ) from exc

    run_dirs = discover_runs(data_root)
    if not run_dirs:
        raise SystemExit(f"No runs found under {data_root}")

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # =========================================================================
    # LAYOUT
    # =========================================================================

    def make_control_card(title: str, children: list, id_prefix: str = "") -> dbc.Card:
        """Create a collapsible control card."""
        collapse_id = f"{id_prefix}-collapse" if id_prefix else f"{title.lower().replace(' ', '-')}-collapse"
        button_id = f"{id_prefix}-toggle" if id_prefix else f"{title.lower().replace(' ', '-')}-toggle"
        return dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    title,
                    id=button_id,
                    color="link",
                    className="text-start w-100",
                ),
            ),
            dbc.Collapse(
                dbc.CardBody(children),
                id=collapse_id,
                is_open=True,
            ),
        ], className="mb-2")

    # Sidebar controls
    sidebar = html.Div([
        html.H4("Pareto Explorer", className="mb-3"),

        # Run selection (always visible)
        dbc.Card([
            dbc.CardBody([
                dbc.Label("Optimization Run"),
                dcc.Dropdown(
                    id="run-selection",
                    options=[{"label": os.path.basename(r), "value": r} for r in run_dirs],
                    value=run_dirs[-1],
                    clearable=False,
                ),
            ])
        ], className="mb-2"),

        # Filter status
        dbc.Alert(
            id="filter-status",
            color="info",
            className="mb-2 py-2",
        ),

        # Metrics section
        make_control_card("Scoring Metrics", [
            dbc.Label("X Axis"),
            dcc.Dropdown(id="x-metric", placeholder="Select X metric"),
            dbc.Label("Y Axis", className="mt-2"),
            dcc.Dropdown(id="y-metric", placeholder="Select Y metric"),
            dbc.Label("Color By", className="mt-2"),
            dcc.Dropdown(id="color-metric", placeholder="Color by metric"),
        ], "metrics"),

        # Filters section
        make_control_card("Filters", [
            dbc.Label("Limit Expressions"),
            dbc.Textarea(
                id="limit-expressions",
                placeholder="One per line, e.g.:\nprofit_sum >= 0\ndrawdown_max <= 0.5",
                style={"height": "100px", "fontFamily": "monospace", "fontSize": "12px"},
            ),
            dbc.Button("Clear Filters", id="clear-filters", color="secondary", size="sm", className="mt-2"),
        ], "filters"),

        # Scenarios section
        make_control_card("Scenarios", [
            dbc.Label("Scenarios"),
            dcc.Dropdown(
                id="scenario-selection",
                placeholder="Select scenarios",
                multi=True,
            ),
            dbc.Label("Scenario Metric", className="mt-2"),
            dcc.Dropdown(id="scenario-metric", placeholder="Metric to compare"),
        ], "scenarios"),

        # Parameters section
        make_control_card("Parameters", [
            dbc.Label("Parameter (X)"),
            dcc.Dropdown(id="param-x", placeholder="Parameter"),
            dbc.Label("Metric (Y)", className="mt-2"),
            dcc.Dropdown(id="metric-y", placeholder="Metric"),
        ], "parameters"),

    ], style={"padding": "15px", "height": "100vh", "overflowY": "auto"})

    # Main content with tabs
    main_content = html.Div([
        dcc.Tabs(id="main-tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Explorer", value="explorer"),
            dcc.Tab(label="Compare", value="compare"),
            dcc.Tab(label="Export", value="export"),
        ]),
        html.Div(id="tab-content", style={"padding": "15px"}),

        # Store for selected config
        dcc.Store(id="selected-config-id", data=None),
        dcc.Store(id="selected-configs-list", data=[]),
    ])

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3, style={"backgroundColor": "#f8f9fa", "padding": 0}),
            dbc.Col(main_content, width=9),
        ], className="g-0"),
    ], fluid=True, style={"height": "100vh"})

    # =========================================================================
    # TAB CONTENT RENDERING
    # =========================================================================

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        Input("run-selection", "value"),
        Input("selected-config-id", "data"),
    )
    def render_tab_content(tab, run_dir, selected_id):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe

        if tab == "overview":
            return render_overview_tab(run_data, df, selected_id)
        elif tab == "explorer":
            return render_explorer_tab(run_data, df, selected_id)
        elif tab == "compare":
            return render_compare_tab(run_data, df)
        elif tab == "export":
            return render_export_tab(run_data, df, selected_id)
        return html.Div("Select a tab")

    def render_overview_tab(run_data: RunData, df: pd.DataFrame, selected_id: str):
        """Overview tab with main scatter plot and summary stats."""
        n_configs = len(df)
        n_params = len(run_data.param_metrics)
        n_metrics = len(run_data.aggregated_metrics)
        n_scenarios = len(run_data.scenario_metrics)

        summary_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(str(n_configs), className="text-primary"),
                    html.P("Configs", className="mb-0"),
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(str(n_params), className="text-success"),
                    html.P("Parameters", className="mb-0"),
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(str(n_metrics), className="text-info"),
                    html.P("Metrics", className="mb-0"),
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(str(n_scenarios), className="text-warning"),
                    html.P("Scenarios", className="mb-0"),
                ])
            ]), width=3),
        ], className="mb-3")

        return html.Div([
            summary_cards,
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-scatter", style={"height": "500px"}),
                ], width=8),
                dbc.Col([
                    html.H5("Selected Config"),
                    html.Div(id="selected-config-summary"),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="histogram-plot", style={"height": "300px"}),
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="scenario-box", style={"height": "300px"}),
                ], width=6),
            ]),
        ])

    def render_explorer_tab(run_data: RunData, df: pd.DataFrame, selected_id: str):
        """Explorer tab with detailed plots and config details."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="explorer-scatter", style={"height": "450px"}),
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="param-scatter-plot", style={"height": "450px"}),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="correlation-heatmap", style={"height": "400px"}),
                ], width=6),
                dbc.Col([
                    html.H5("Config Details", className="mt-2"),
                    html.Div(id="config-details-panel", style={
                        "maxHeight": "380px",
                        "overflowY": "auto",
                        "backgroundColor": "#f8f9fa",
                        "padding": "10px",
                        "borderRadius": "5px",
                    }),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("All Configs (filtered)", className="mt-3"),
                    dash_table.DataTable(
                        id="pareto-table",
                        page_size=10,
                        sort_action="native",
                        filter_action="native",
                        row_selectable="single",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "5px", "fontSize": "12px"},
                        style_header={"fontWeight": "bold"},
                    ),
                ], width=12),
            ]),
        ])

    def render_compare_tab(run_data: RunData, df: pd.DataFrame):
        """Compare tab for side-by-side config comparison."""
        return html.Div([
            dbc.Alert(
                "Select configs by clicking on points in the scatter plots, then view comparison here.",
                color="info",
            ),
            html.H5("Selected Configs for Comparison"),
            html.Div(id="comparison-table"),
            dcc.Graph(id="radar-comparison", style={"height": "400px"}),
        ])

    def render_export_tab(run_data: RunData, df: pd.DataFrame, selected_id: str):
        """Export tab for saving configs."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H5("Export Selected Config"),
                    html.P("Click on a config in the scatter plot, then export it here."),
                    html.Div(id="export-config-id", className="mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("Copy JSON", id="copy-json-btn", color="primary"),
                        dbc.Button("Download JSON", id="download-json-btn", color="secondary"),
                    ]),
                    dcc.Download(id="download-config"),
                    html.Pre(
                        id="export-json-preview",
                        style={
                            "backgroundColor": "#f8f9fa",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "maxHeight": "500px",
                            "overflowY": "auto",
                            "fontSize": "11px",
                            "marginTop": "15px",
                        }
                    ),
                ], width=12),
            ]),
        ])

    # =========================================================================
    # CALLBACKS - Metric dropdowns
    # =========================================================================

    @app.callback(
        Output("x-metric", "options"),
        Output("y-metric", "options"),
        Output("color-metric", "options"),
        Output("x-metric", "value"),
        Output("y-metric", "value"),
        Output("color-metric", "value"),
        Input("run-selection", "value"),
    )
    def update_metric_choices(run_dir):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        numeric_cols = [
            c for c in df.columns
            if c != "_id" and pd.api.types.is_numeric_dtype(df[c])
            and not _looks_like_stat_column(c) and not c.startswith("bot.")
        ]
        options = [{"label": col, "value": col} for col in numeric_cols]
        preferred = [col for col in run_data.scoring_metrics if col in numeric_cols]
        default_x = preferred[0] if preferred else (numeric_cols[0] if numeric_cols else None)
        default_y = preferred[1] if len(preferred) > 1 else (numeric_cols[1] if len(numeric_cols) > 1 else default_x)
        default_color = preferred[2] if len(preferred) > 2 else None
        return options, options, options, default_x, default_y, default_color

    @app.callback(
        Output("scenario-selection", "options"),
        Output("scenario-selection", "value"),
        Output("scenario-metric", "options"),
        Output("scenario-metric", "value"),
        Output("param-x", "options"),
        Output("param-x", "value"),
        Output("metric-y", "options"),
        Output("metric-y", "value"),
        Output("limit-expressions", "value"),
        Input("run-selection", "value"),
    )
    def update_secondary_controls(run_dir):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe

        # Scenarios
        scenario_options = [{"label": name, "value": name} for name in sorted(run_data.scenario_metrics)]
        scenario_value = [opt["value"] for opt in scenario_options]
        scenario_metrics_set = set()
        for metrics in run_data.scenario_metrics.values():
            scenario_metrics_set.update(metrics)
        scenario_metric_options = [{"label": m, "value": m} for m in sorted(scenario_metrics_set)]
        scenario_metric_value = run_data.scoring_metrics[0] if run_data.scoring_metrics else (
            scenario_metric_options[0]["value"] if scenario_metric_options else None
        )

        # Parameters
        param_cols = [c for c in run_data.param_metrics if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        param_options = [{"label": col, "value": col} for col in param_cols]
        param_value = param_cols[0] if param_cols else None

        # Metrics for Y axis
        metric_cols = [c for c in run_data.aggregated_metrics if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        metric_options = [{"label": col, "value": col} for col in metric_cols]
        metric_value = run_data.scoring_metrics[0] if run_data.scoring_metrics and run_data.scoring_metrics[0] in metric_cols else (
            metric_cols[0] if metric_cols else None
        )

        limits_default = "\n".join(run_data.default_limits)

        return (
            scenario_options, scenario_value,
            scenario_metric_options, scenario_metric_value,
            param_options, param_value,
            metric_options, metric_value,
            limits_default,
        )

    # =========================================================================
    # CALLBACKS - Filter status
    # =========================================================================

    @app.callback(
        Output("filter-status", "children"),
        Input("run-selection", "value"),
        Input("limit-expressions", "value"),
    )
    def update_filter_status(run_dir, limit_exprs):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if df.empty:
            return "No configs loaded"
        mask = _apply_limits(df, limit_exprs)
        n_pass = mask.sum()
        n_total = len(df)
        pct = (n_pass / n_total * 100) if n_total > 0 else 0
        return f"{n_pass} / {n_total} configs pass filters ({pct:.1f}%)"

    @app.callback(
        Output("limit-expressions", "value", allow_duplicate=True),
        Input("clear-filters", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_filters(n_clicks):
        return ""

    # =========================================================================
    # CALLBACKS - Main scatter plot with click selection
    # =========================================================================

    @app.callback(
        Output("main-scatter", "figure"),
        Output("explorer-scatter", "figure"),
        Input("run-selection", "value"),
        Input("x-metric", "value"),
        Input("y-metric", "value"),
        Input("color-metric", "value"),
        Input("limit-expressions", "value"),
        Input("selected-config-id", "data"),
    )
    def update_scatter_plots(run_dir, x_metric, y_metric, color_metric, limit_exprs, selected_id):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe

        if df.empty or not x_metric or not y_metric:
            empty = px.scatter(title="Select metrics")
            return empty, empty

        mask = _apply_limits(df, limit_exprs)
        df_filtered = df.loc[mask].copy()

        if df_filtered.empty:
            empty = px.scatter(title="No configs pass filters")
            return empty, empty

        # Add selection marker
        df_filtered["_selected"] = df_filtered["_id"] == selected_id

        # Create scatter plot
        fig = px.scatter(
            df_filtered,
            x=x_metric,
            y=y_metric,
            color=color_metric if color_metric and color_metric in df_filtered.columns else None,
            hover_data=["_id"],
            title=f"{y_metric} vs {x_metric}",
            custom_data=["_id"],
        )

        # Highlight selected point
        if selected_id and selected_id in df_filtered["_id"].values:
            selected_row = df_filtered[df_filtered["_id"] == selected_id].iloc[0]
            fig.add_trace(go.Scatter(
                x=[selected_row[x_metric]],
                y=[selected_row[y_metric]],
                mode="markers",
                marker=dict(size=20, color="red", symbol="star", line=dict(width=2, color="black")),
                name="Selected",
                showlegend=True,
            ))

        fig.update_layout(clickmode="event+select")

        return fig, fig

    @app.callback(
        Output("selected-config-id", "data"),
        Input("main-scatter", "clickData"),
        Input("explorer-scatter", "clickData"),
        Input("pareto-table", "selected_rows"),
        State("run-selection", "value"),
        State("limit-expressions", "value"),
    )
    def handle_config_selection(main_click, explorer_click, table_rows, run_dir, limit_exprs):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id in ["main-scatter", "explorer-scatter"]:
            click_data = main_click if trigger_id == "main-scatter" else explorer_click
            if click_data and "points" in click_data and click_data["points"]:
                point = click_data["points"][0]
                if "customdata" in point and point["customdata"]:
                    return point["customdata"][0]

        elif trigger_id == "pareto-table" and table_rows:
            run_data = get_run_data(run_dir)
            df = run_data.dataframe
            mask = _apply_limits(df, limit_exprs)
            df_filtered = df.loc[mask]
            if table_rows[0] < len(df_filtered):
                return df_filtered.iloc[table_rows[0]]["_id"]

        return no_update

    # =========================================================================
    # CALLBACKS - Selected config display
    # =========================================================================

    @app.callback(
        Output("selected-config-summary", "children"),
        Output("config-details-panel", "children"),
        Input("selected-config-id", "data"),
        Input("run-selection", "value"),
    )
    def update_selected_config_display(selected_id, run_dir):
        if not selected_id:
            no_selection = html.P("Click on a point to select a config", className="text-muted")
            return no_selection, no_selection

        run_data = get_run_data(run_dir)
        if selected_id not in run_data.raw_configs:
            return html.P("Config not found"), html.P("Config not found")

        config = run_data.raw_configs[selected_id]
        df = run_data.dataframe
        row = df[df["_id"] == selected_id]

        # Summary card
        summary_items = [html.Strong(f"ID: {selected_id[:16]}...")]
        if not row.empty:
            for metric in run_data.scoring_metrics[:5]:
                if metric in row.columns:
                    val = row[metric].values[0]
                    if pd.notna(val):
                        summary_items.append(html.Div(f"{metric}: {val:.4f}"))
        summary = html.Div(summary_items, style={"fontSize": "12px"})

        # Detailed config - bot parameters
        bot_params = config.get("bot", {})
        details_items = [html.H6("Bot Parameters")]

        def render_params(params, prefix=""):
            items = []
            for key, value in sorted(params.items()):
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    items.append(html.Div([
                        html.Strong(f"{key}:"),
                        html.Div(render_params(value, full_key), style={"marginLeft": "15px"}),
                    ]))
                else:
                    items.append(html.Div(f"{key}: {value}", style={"fontSize": "11px"}))
            return items

        for side in ["long", "short"]:
            if side in bot_params:
                details_items.append(html.H6(f"{side.capitalize()}", className="mt-2 text-primary"))
                details_items.extend(render_params(bot_params[side]))

        # Add key metrics
        details_items.append(html.H6("Key Metrics", className="mt-3"))
        if not row.empty:
            for metric in run_data.aggregated_metrics[:10]:
                if metric in row.columns:
                    val = row[metric].values[0]
                    if pd.notna(val):
                        details_items.append(html.Div(f"{metric}: {val:.4f}", style={"fontSize": "11px"}))

        details = html.Div(details_items)
        return summary, details

    # =========================================================================
    # CALLBACKS - Other plots
    # =========================================================================

    @app.callback(
        Output("histogram-plot", "figure"),
        Input("run-selection", "value"),
        Input("x-metric", "value"),
        Input("limit-expressions", "value"),
    )
    def update_histogram(run_dir, metric, limit_exprs):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if df.empty or not metric or metric not in df.columns:
            return px.histogram(title="Select a metric")
        mask = _apply_limits(df, limit_exprs)
        df_filtered = df.loc[mask]
        fig = px.histogram(df_filtered, x=metric, title=f"Distribution of {metric}")
        return fig

    @app.callback(
        Output("scenario-box", "figure"),
        Input("run-selection", "value"),
        Input("scenario-selection", "value"),
        Input("scenario-metric", "value"),
    )
    def update_scenario_box(run_dir, selected_scenarios, metric_name):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if not metric_name or df.empty:
            return px.box(title="Select scenario metric")
        if not selected_scenarios:
            selected_scenarios = list(run_data.scenario_metrics.keys())
        records = []
        for scenario in selected_scenarios:
            column = f"{scenario}__{metric_name}"
            if column not in df:
                continue
            for value in df[column].dropna():
                records.append({"scenario": scenario, "value": value})
        if not records:
            return px.box(title="No scenario data")
        plot_df = pd.DataFrame(records)
        fig = px.box(plot_df, x="scenario", y="value", points="all", title=f"{metric_name} by Scenario")
        return fig

    @app.callback(
        Output("param-scatter-plot", "figure"),
        Input("run-selection", "value"),
        Input("param-x", "value"),
        Input("metric-y", "value"),
        Input("limit-expressions", "value"),
        Input("selected-config-id", "data"),
    )
    def update_param_scatter(run_dir, param_col, metric_col, limit_exprs, selected_id):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if not param_col or not metric_col or param_col not in df.columns or metric_col not in df.columns:
            return px.scatter(title="Select parameter and metric")
        mask = _apply_limits(df, limit_exprs)
        df_filtered = df.loc[mask].dropna(subset=[param_col, metric_col])
        if df_filtered.empty:
            return px.scatter(title="No data")
        fig = px.scatter(
            df_filtered, x=param_col, y=metric_col,
            hover_data=["_id"],
            title=f"{metric_col} vs {param_col}",
            custom_data=["_id"],
        )
        # Highlight selected
        if selected_id and selected_id in df_filtered["_id"].values:
            sel = df_filtered[df_filtered["_id"] == selected_id].iloc[0]
            fig.add_trace(go.Scatter(
                x=[sel[param_col]], y=[sel[metric_col]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name="Selected",
            ))
        return fig

    @app.callback(
        Output("correlation-heatmap", "figure"),
        Input("run-selection", "value"),
    )
    def update_correlation(run_dir):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        # Use scoring metrics + a few aggregated metrics
        metrics = run_data.scoring_metrics[:6]
        if len(metrics) < 2:
            metrics = run_data.aggregated_metrics[:6]
        available = [m for m in metrics if m in df.columns]
        if len(available) < 2:
            return px.imshow([[0]], title="Not enough metrics for correlation")
        corr_df = df[available].dropna()
        if corr_df.empty:
            return px.imshow([[0]], title="No data for correlation")
        matrix = corr_df.corr()
        fig = px.imshow(
            matrix, x=matrix.columns, y=matrix.columns,
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
            title="Metric Correlations"
        )
        return fig

    # =========================================================================
    # CALLBACKS - Pareto table
    # =========================================================================

    @app.callback(
        Output("pareto-table", "data"),
        Output("pareto-table", "columns"),
        Input("run-selection", "value"),
        Input("limit-expressions", "value"),
    )
    def update_pareto_table(run_dir, limit_exprs):
        run_data = get_run_data(run_dir)
        df = run_data.dataframe
        if df.empty:
            return [], []
        mask = _apply_limits(df, limit_exprs)
        df_filtered = df.loc[mask]

        # Select columns to show
        show_cols = ["_id"] + run_data.scoring_metrics[:5] + run_data.aggregated_metrics[:5]
        show_cols = [c for c in show_cols if c in df_filtered.columns]
        show_cols = list(dict.fromkeys(show_cols))  # Remove duplicates, keep order

        display_df = df_filtered[show_cols].copy()
        display_df["_id"] = display_df["_id"].str[:12] + "..."

        # Format numeric columns
        for col in display_df.columns:
            if col != "_id" and pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        columns = [{"name": col, "id": col} for col in show_cols]
        data = display_df.to_dict("records")
        return data, columns

    # =========================================================================
    # CALLBACKS - Export
    # =========================================================================

    @app.callback(
        Output("export-config-id", "children"),
        Output("export-json-preview", "children"),
        Input("selected-config-id", "data"),
        Input("run-selection", "value"),
    )
    def update_export_preview(selected_id, run_dir):
        if not selected_id:
            return "No config selected", "Select a config by clicking on a point in the scatter plot"

        run_data = get_run_data(run_dir)
        if selected_id not in run_data.raw_configs:
            return "Config not found", ""

        config = run_data.raw_configs[selected_id]
        # Extract just the bot config for export
        bot_config = {"bot": config.get("bot", {})}
        json_str = json.dumps(bot_config, indent=2)

        return f"Selected: {selected_id[:24]}...", json_str

    @app.callback(
        Output("download-config", "data"),
        Input("download-json-btn", "n_clicks"),
        State("selected-config-id", "data"),
        State("run-selection", "value"),
        prevent_initial_call=True,
    )
    def download_config(n_clicks, selected_id, run_dir):
        if not selected_id:
            return no_update
        run_data = get_run_data(run_dir)
        if selected_id not in run_data.raw_configs:
            return no_update
        config = run_data.raw_configs[selected_id]
        bot_config = {"bot": config.get("bot", {})}
        json_str = json.dumps(bot_config, indent=2)
        filename = f"config_{selected_id[:16]}.json"
        return dict(content=json_str, filename=filename)

    # =========================================================================
    # CALLBACKS - Collapsible sections
    # =========================================================================

    for section in ["metrics", "filters", "scenarios", "parameters"]:
        @app.callback(
            Output(f"{section}-collapse", "is_open"),
            Input(f"{section}-toggle", "n_clicks"),
            State(f"{section}-collapse", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_collapse(n_clicks, is_open, section=section):
            return not is_open

    # =========================================================================
    # RUN SERVER
    # =========================================================================

    print(f"Starting Pareto Explorer on http://localhost:{port}")
    app.run(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description="Pareto Dashboard - Interactive optimizer results explorer")
    parser.add_argument(
        "--data-root",
        default="optimize_results",
        help="Directory containing optimization runs (default: optimize_results)",
    )
    parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard on (default: 8050)")
    args = parser.parse_args()
    serve_dash(args.data_root, port=args.port)


if __name__ == "__main__":
    main()
