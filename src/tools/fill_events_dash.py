"""
Interactive dashboard for exploring cached/fresh fill events.

Features:
- Multi-account view with parallel refresh
- Cumulative/daily PnL charts
- Top symbols analysis
- Cache health monitoring (gaps, coverage)
- CSV export functionality
- Console log panel for progress feedback
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html
from dash.exceptions import PreventUpdate

# Ensure we can import modules from src/
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from fill_events_manager import (
    FillEventsManager,
    _build_fetcher_for_bot,
    _extract_symbol_pool,
    _instantiate_bot,
)
from config_utils import format_config, load_config
from logging_setup import configure_logging


# Global log buffer for UI display
_LOG_BUFFER: deque = deque(maxlen=200)
_LOG_LOCK = threading.Lock()

# Global event loop for async operations - persists across calls
_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None

# Background refresh state management
_REFRESH_STATE: Dict[str, Any] = {
    "is_running": False,
    "progress": "",
    "result": None,
    "error": None,
}
_REFRESH_LOCK = threading.Lock()
_REFRESH_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop running in a background thread."""
    global _EVENT_LOOP, _LOOP_THREAD

    if _EVENT_LOOP is not None and _EVENT_LOOP.is_running():
        return _EVENT_LOOP

    # Create new event loop in a background thread
    _EVENT_LOOP = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(_EVENT_LOOP)
        _EVENT_LOOP.run_forever()

    _LOOP_THREAD = threading.Thread(target=run_loop, daemon=True)
    _LOOP_THREAD.start()

    # Give the loop a moment to start
    time.sleep(0.1)
    return _EVENT_LOOP


def _run_async(coro):
    """Run an async coroutine on the persistent event loop."""
    loop = _get_or_create_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=600)  # 10 minute timeout


class DashLogHandler(logging.Handler):
    """Custom log handler that captures logs for the dashboard."""

    def emit(self, record):
        try:
            msg = self.format(record)
            with _LOG_LOCK:
                _LOG_BUFFER.append(msg)
        except Exception:
            pass


def _get_log_messages() -> str:
    """Get recent log messages for display."""
    with _LOG_LOCK:
        return "\n".join(_LOG_BUFFER)


def _normalize_fee_cost(fees: Optional[object]) -> float:
    if fees is None:
        return 0.0
    total = 0.0
    items: List[dict] = []
    if isinstance(fees, dict):
        items = [fees]
    elif isinstance(fees, (list, tuple)):
        items = [x for x in fees if isinstance(x, dict)]
    for entry in items:
        try:
            total += float(entry.get("cost", 0.0))
        except Exception:
            continue
    return total


def _format_datetime_str(dt: pd.Timestamp) -> str:
    """Format datetime consistently as 'YYYY-MM-DD HH:MM:SS' for display."""
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _events_to_dataframe(events: List[dict], account_label: str) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["account"] = account_label
    # Use timestamp (ms) for proper sorting, create datetime for display
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    # Create formatted string for display (consistent format across exchanges)
    df["datetime_str"] = df["datetime"].apply(_format_datetime_str)
    fees_col = df.get("fees", pd.Series([None] * len(df)))
    df["fee_cost"] = [_normalize_fee_cost(x) for x in fees_col]
    df["pnl_with_fees"] = df["pnl"] - df["fee_cost"]
    return df


def _build_managers(
    users: List[str],
    config_path: str,
    cache_root: str,
    symbols_override: Optional[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Build managers and store bot references."""
    result: Dict[str, Dict[str, Any]] = {}
    for user in users:
        try:
            config = load_config(config_path, verbose=False)
            config = format_config(config, verbose=False)
            config.setdefault("live", {})["user"] = user
            bot = _instantiate_bot(config)
            symbol_pool = _extract_symbol_pool(config, symbols_override)
            fetcher = _build_fetcher_for_bot(bot, symbol_pool)
            cache_path = Path(cache_root) / bot.exchange / bot.user
            key = f"{bot.exchange}:{bot.user}"
            result[key] = {
                "manager": FillEventsManager(
                    exchange=bot.exchange, user=bot.user, fetcher=fetcher, cache_path=cache_path
                ),
                "bot": bot,
                "exchange": bot.exchange,
                "user": bot.user,
                "config_path": config_path,
                "symbols_override": symbols_override,
                "cache_root": cache_root,
            }
        except Exception as e:
            logging.error(f"Failed to build manager for user {user}: {e}")
    return result


def _ensure_loaded(accounts: Dict[str, Dict[str, Any]]) -> None:
    """Load cached data for all accounts."""
    for key, data in accounts.items():
        try:
            _run_async(data["manager"].ensure_loaded())
        except Exception as e:
            logging.error(f"Failed to load {key}: {e}")


def _rebuild_manager(data: Dict[str, Any]) -> None:
    """Rebuild a manager with fresh bot/fetcher instances to avoid stale connections."""
    try:
        config = load_config(data["config_path"], verbose=False)
        config = format_config(config, verbose=False)
        config.setdefault("live", {})["user"] = data["user"]
        bot = _instantiate_bot(config)
        symbol_pool = _extract_symbol_pool(config, data["symbols_override"])
        fetcher = _build_fetcher_for_bot(bot, symbol_pool)
        cache_path = Path(data["cache_root"]) / bot.exchange / bot.user

        # Create new manager
        data["manager"] = FillEventsManager(
            exchange=bot.exchange, user=bot.user, fetcher=fetcher, cache_path=cache_path
        )
        data["bot"] = bot
        logging.debug(f"Rebuilt manager for {data['exchange']}:{data['user']}")
    except Exception as e:
        logging.error(f"Failed to rebuild manager for {data['exchange']}:{data['user']}: {e}")


async def _refresh_single(data: Dict[str, Any], start_ms: int, end_ms: int) -> int:
    """Refresh a single account. Returns number of events after refresh."""
    try:
        await data["manager"].refresh_range(start_ms, end_ms)
        # Reload from disk to get fresh data
        await data["manager"].ensure_loaded()
        return len(data["manager"]._events)
    except Exception as e:
        logging.error(f"Failed to refresh {data['exchange']}:{data['user']}: {e}")
        return 0


async def _refresh_all_parallel(
    accounts: Dict[str, Dict[str, Any]], selected_accounts: List[str], start_ms: int, end_ms: int
) -> Dict[str, int]:
    """Refresh all selected accounts in parallel. Returns event counts."""
    tasks = {}
    for account in selected_accounts:
        data = accounts.get(account)
        if data is None:
            continue
        tasks[account] = _refresh_single(data, start_ms, end_ms)

    results = {}
    if tasks:
        logging.info(f"Starting parallel refresh for {len(tasks)} account(s)")
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for account, result in zip(tasks.keys(), task_results):
            if isinstance(result, Exception):
                logging.error(f"Refresh failed for {account}: {result}")
                results[account] = 0
            else:
                results[account] = result
        logging.info("Parallel refresh complete")
    return results


def _refresh_range(
    accounts: Dict[str, Dict[str, Any]], selected_accounts: List[str], start_ms: int, end_ms: int
) -> Dict[str, int]:
    """Refresh accounts in parallel using persistent event loop."""
    # Rebuild managers to get fresh connections before refresh
    for account in selected_accounts:
        data = accounts.get(account)
        if data:
            _rebuild_manager(data)

    return _run_async(_refresh_all_parallel(accounts, selected_accounts, start_ms, end_ms))


def _start_background_refresh(
    accounts: Dict[str, Dict[str, Any]], selected_accounts: List[str], start_ms: int, end_ms: int
) -> bool:
    """Start a background refresh if not already running. Returns True if started."""
    with _REFRESH_LOCK:
        if _REFRESH_STATE["is_running"]:
            return False
        _REFRESH_STATE["is_running"] = True
        _REFRESH_STATE["progress"] = "Starting refresh..."
        _REFRESH_STATE["result"] = None
        _REFRESH_STATE["error"] = None

    def do_refresh():
        try:
            with _REFRESH_LOCK:
                _REFRESH_STATE["progress"] = f"Refreshing {len(selected_accounts)} account(s)..."

            # Rebuild managers to get fresh connections before refresh
            for i, account in enumerate(selected_accounts):
                data = accounts.get(account)
                if data:
                    with _REFRESH_LOCK:
                        _REFRESH_STATE["progress"] = f"Rebuilding {account} ({i+1}/{len(selected_accounts)})..."
                    _rebuild_manager(data)

            with _REFRESH_LOCK:
                _REFRESH_STATE["progress"] = "Fetching fill events from exchanges..."

            results = _run_async(_refresh_all_parallel(accounts, selected_accounts, start_ms, end_ms))

            with _REFRESH_LOCK:
                _REFRESH_STATE["result"] = results
                _REFRESH_STATE["progress"] = "Complete"
        except Exception as e:
            logging.error(f"Background refresh failed: {e}")
            with _REFRESH_LOCK:
                _REFRESH_STATE["error"] = str(e)
                _REFRESH_STATE["progress"] = f"Error: {e}"
        finally:
            with _REFRESH_LOCK:
                _REFRESH_STATE["is_running"] = False

    _REFRESH_EXECUTOR.submit(do_refresh)
    return True


def _get_refresh_state() -> Dict[str, Any]:
    """Get current refresh state (thread-safe copy)."""
    with _REFRESH_LOCK:
        return {
            "is_running": _REFRESH_STATE["is_running"],
            "progress": _REFRESH_STATE["progress"],
            "result": _REFRESH_STATE["result"],
            "error": _REFRESH_STATE["error"],
        }


def _clear_refresh_result():
    """Clear the refresh result after consuming it."""
    with _REFRESH_LOCK:
        _REFRESH_STATE["result"] = None
        _REFRESH_STATE["error"] = None


def _aggregate_accounts(
    accounts: Dict[str, Dict[str, Any]],
    selected_accounts: List[str],
    start_ms: Optional[int],
    end_ms: Optional[int],
    symbols_filter: Optional[List[str]],
) -> pd.DataFrame:
    """Aggregate events from all selected accounts into a DataFrame."""
    frames: List[pd.DataFrame] = []
    for account, data in accounts.items():
        if selected_accounts and account not in selected_accounts:
            continue
        try:
            # Ensure loaded (synchronous via persistent loop)
            _run_async(data["manager"].ensure_loaded())
            events = data["manager"].get_events(start_ms, end_ms)
            frame = _events_to_dataframe([ev.to_dict() for ev in events], account)
            if symbols_filter:
                frame = frame[frame["symbol"].isin(symbols_filter)]
            if not frame.empty:
                frames.append(frame)
        except Exception as e:
            logging.error(f"Failed to aggregate {account}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _get_coverage_summaries(
    accounts: Dict[str, Dict[str, Any]], selected_accounts: List[str]
) -> List[Dict[str, Any]]:
    """Get cache coverage summaries for selected accounts."""
    summaries = []
    for account in selected_accounts:
        data = accounts.get(account)
        if data is None:
            continue
        try:
            summary = data["manager"].get_coverage_summary()
            summary["account"] = account
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Failed to get coverage for {account}: {e}")
    return summaries


def build_figures(df: pd.DataFrame):
    """Build the main dashboard figures.

    Note: Uses raw PnL instead of pnl_with_fees since fee data is
    inconsistent/unavailable across different exchanges.
    """
    if df.empty:
        return (
            px.line(title="Cumulative PnL (no data)"),
            px.bar(title="Daily PnL (no data)"),
            px.bar(title="Top Symbols (no data)"),
            px.bar(title="PnL by Account (no data)"),
        )
    df = df.sort_values("timestamp").copy()
    df["cum_pnl"] = df.groupby("account")["pnl"].cumsum()
    cum_fig = px.line(
        df,
        x="datetime",
        y="cum_pnl",
        color="account",
        title="Cumulative Realized PnL",
        hover_data=["symbol", "pnl", "pb_order_type"],
    )
    daily = (
        df.groupby(["date", "account"], as_index=False)
        .agg({"pnl": "sum"})
        .sort_values("date")
    )
    daily_fig = px.bar(
        daily,
        x="date",
        y="pnl",
        color="account",
        title="Daily Realized PnL",
        barmode="group",
    )
    top_symbols = (
        df.groupby(["symbol", "account"], as_index=False)
        .agg({"pnl": "sum", "qty": "sum"})
        .sort_values("pnl", ascending=False)
        .head(30)
    )
    top_fig = px.bar(
        top_symbols,
        x="symbol",
        y="pnl",
        color="account",
        title="Top Symbols by Realized PnL",
    )
    # Replace fees chart with PnL by account summary
    account_pnl = (
        df.groupby("account", as_index=False)
        .agg({"pnl": "sum"})
        .sort_values("pnl", ascending=False)
    )
    account_fig = px.bar(account_pnl, x="account", y="pnl", title="Total PnL by Account")
    return cum_fig, daily_fig, top_fig, account_fig


def build_symbol_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Build a chart for a single symbol."""
    symbol_df = df[df["symbol"] == symbol].copy()
    if symbol_df.empty:
        return px.scatter(title=f"No fills for {symbol}")

    # Sort by timestamp for correct chronological ordering
    symbol_df = symbol_df.sort_values("timestamp")

    # Simple scatter plot
    fig = px.scatter(
        symbol_df,
        x="datetime",
        y="price",
        color="side",
        color_discrete_map={"buy": "blue", "sell": "red"},
        size=symbol_df["qty"].abs(),
        hover_data=["qty", "pnl", "pb_order_type", "datetime_str"],
        title=f"{symbol} Fills",
    )
    return fig


def build_cache_health_panel(summaries: List[Dict[str, Any]]) -> html.Div:
    """Build a panel showing cache health for each account."""
    if not summaries:
        return html.Div("No accounts selected", className="text-muted")

    cards = []
    for summary in summaries:
        account = summary.get("account", "Unknown")
        total_gaps = summary.get("total_gaps", 0)
        persistent_gaps = summary.get("persistent_gaps", 0)
        retryable_gaps = summary.get("retryable_gaps", 0)
        total_gap_hours = summary.get("total_gap_hours", 0)
        events_count = summary.get("events_count", 0)
        first_event = summary.get("first_event", "N/A")
        last_event = summary.get("last_event", "N/A")

        # Determine health status
        if total_gaps == 0:
            status_color = "success"
            status_text = "Healthy"
        elif retryable_gaps > 0:
            status_color = "warning"
            status_text = f"{retryable_gaps} retryable gap(s)"
        else:
            status_color = "info"
            status_text = f"{persistent_gaps} known gap(s)"

        gap_details = []
        for gap in summary.get("gaps", [])[:5]:  # Show first 5 gaps
            gap_details.append(
                html.Li(
                    f"{gap['start']} → {gap['end']} "
                    f"(retries: {gap['retry_count']}, reason: {gap['reason']})"
                )
            )

        card = dbc.Card(
            [
                dbc.CardHeader([
                    html.Strong(account),
                    dbc.Badge(status_text, color=status_color, className="ms-2"),
                ]),
                dbc.CardBody([
                    html.P(f"Events: {events_count:,}"),
                    html.P(f"First: {first_event}"),
                    html.P(f"Last: {last_event}"),
                    html.P(f"Gap hours: {total_gap_hours:.1f}"),
                    html.Ul(gap_details) if gap_details else html.P("No gaps", className="text-muted"),
                ]),
            ],
            className="mb-2",
        )
        cards.append(dbc.Col(card, md=6))

    return dbc.Row(cards)


def serve_dash(accounts: Dict[str, Dict[str, Any]], default_days: int = 30, port: int = 8050) -> None:
    """Serve the Dash application."""
    _ensure_loaded(accounts)
    now = pd.Timestamp.utcnow()
    start_default = now - pd.Timedelta(days=default_days)

    app: Dash = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # Setup log handler
    log_handler = DashLogHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(log_handler)

    app.layout = dbc.Container(
        [
            html.H2("Fill Events Dashboard"),

            # Loading overlay - shows when refresh is running
            html.Div(
                id="loading-overlay",
                style={
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100%",
                    "height": "100%",
                    "backgroundColor": "rgba(0,0,0,0.5)",
                    "zIndex": 9999,
                    "display": "none",
                    "justifyContent": "center",
                    "alignItems": "center",
                },
                children=[
                    dbc.Card(
                        [
                            dbc.CardBody([
                                html.Div([
                                    dbc.Spinner(size="lg", color="primary"),
                                    html.H4("Fetching Fill Events...", className="mt-3"),
                                    html.P(id="loading-progress", className="text-muted mb-0"),
                                ], className="text-center")
                            ])
                        ],
                        style={"width": "350px"},
                    )
                ],
            ),

            # Controls row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Accounts"),
                            dcc.Dropdown(
                                id="accounts",
                                options=[{"label": k, "value": k} for k in accounts.keys()],
                                value=list(accounts.keys()),
                                multi=True,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Symbols (optional)"),
                            dcc.Dropdown(id="symbols", options=[], value=[], multi=True),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Quick select"),
                            dcc.Dropdown(
                                id="quick-days",
                                options=[
                                    {"label": "Last 7 days", "value": 7},
                                    {"label": "Last 14 days", "value": 14},
                                    {"label": "Last 30 days", "value": 30},
                                    {"label": "Last 60 days", "value": 60},
                                    {"label": "Last 90 days", "value": 90},
                                ],
                                value=default_days,
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Date range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=(now - pd.Timedelta(days=365)).date(),
                                start_date=start_default.date(),
                                end_date=now.date(),
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Actions"),
                            html.Div([
                                dbc.Button(
                                    "Refresh", id="refresh-btn", color="primary",
                                    size="sm", className="me-1", n_clicks=0
                                ),
                                html.Span(id="refresh-status", className="ms-2 text-muted small"),
                            ])
                        ],
                        md=2,
                    ),
                ],
                className="mb-3",
            ),

            # Tabs
            dbc.Tabs(
                [
                    dbc.Tab(label="Overview", tab_id="tab-overview"),
                    dbc.Tab(label="Symbol Detail", tab_id="tab-symbol"),
                    dbc.Tab(label="Cache Health", tab_id="tab-health"),
                    dbc.Tab(label="Export", tab_id="tab-export"),
                    dbc.Tab(label="Console", tab_id="tab-console"),
                ],
                id="tabs",
                active_tab="tab-overview",
                className="mb-3",
            ),

            # Tab content container
            html.Div(id="tab-content"),

            # Data stores
            dcc.Store(id="fill-data"),
            dcc.Store(id="refresh-trigger", data=0),
            dcc.Store(id="refresh-params"),  # Store params for background refresh
            dcc.Download(id="download-csv"),

            # Interval for refresh polling and log updates
            dcc.Interval(id="refresh-poll-interval", interval=500, n_intervals=0),
            dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
        ],
        fluid=True,
    )

    # Quick select updates date range
    @app.callback(
        Output("date-range", "start_date"),
        Output("date-range", "end_date"),
        Input("quick-days", "value"),
        prevent_initial_call=True,
    )
    def update_date_range_from_quick_select(days):
        if days is None:
            raise PreventUpdate
        now = pd.Timestamp.utcnow()
        start = now - pd.Timedelta(days=days)
        return start.date(), now.date()

    # Refresh button starts background refresh
    @app.callback(
        Output("refresh-params", "data"),
        Output("refresh-btn", "disabled"),
        Input("refresh-btn", "n_clicks"),
        State("accounts", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        prevent_initial_call=True,
    )
    def start_refresh(n_clicks, selected_accounts, start_date, end_date):
        if n_clicks == 0:
            raise PreventUpdate

        selected = selected_accounts or list(accounts.keys())
        start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000) if start_date else None
        end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000) + 86400000 if end_date else None

        if start_ms is None or end_ms is None:
            logging.warning("Select date range first")
            raise PreventUpdate

        logging.info(f"Starting refresh: {start_date} to {end_date} for {len(selected)} account(s)")
        started = _start_background_refresh(accounts, selected, start_ms, end_ms)

        if not started:
            logging.warning("Refresh already in progress")
            raise PreventUpdate

        return {"selected": selected, "start_ms": start_ms, "end_ms": end_ms}, True

    # Poll for refresh completion and update loading overlay
    @app.callback(
        Output("loading-overlay", "style"),
        Output("loading-progress", "children"),
        Output("refresh-btn", "disabled", allow_duplicate=True),
        Output("refresh-trigger", "data"),
        Input("refresh-poll-interval", "n_intervals"),
        State("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def poll_refresh_state(n_intervals, trigger):
        state = _get_refresh_state()

        overlay_style = {
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0,0,0,0.5)",
            "zIndex": 9999,
            "display": "flex" if state["is_running"] else "none",
            "justifyContent": "center",
            "alignItems": "center",
        }

        if state["is_running"]:
            return overlay_style, state["progress"], True, dash.no_update

        # Check if there's a result to consume
        if state["result"] is not None or state["error"] is not None:
            _clear_refresh_result()
            return overlay_style, "", False, (trigger or 0) + 1

        return overlay_style, "", False, dash.no_update

    # Load data from cache (triggered by refresh completion or initial load)
    @app.callback(
        Output("fill-data", "data"),
        Output("symbols", "options"),
        Output("refresh-status", "children"),
        Input("refresh-trigger", "data"),
        State("accounts", "value"),
        State("symbols", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        prevent_initial_call=False,
    )
    def load_data(trigger, selected_accounts, symbols, start_date, end_date):
        selected = selected_accounts or list(accounts.keys())
        start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000) if start_date else None
        end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000) + 86400000 if end_date else None

        # Aggregate from cache
        df = _aggregate_accounts(accounts, selected, start_ms, end_ms, symbols if symbols else None)
        symbols_options = (
            [{"label": s, "value": s} for s in sorted(df["symbol"].unique())] if not df.empty else []
        )

        status = f"{len(df)} events" if not df.empty else "No data"
        return df.to_dict(orient="records"), symbols_options, status

    # Render tab content
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
        Input("fill-data", "data"),
        Input("log-interval", "n_intervals"),
        State("symbols", "value"),
        State("accounts", "value"),
    )
    def render_tab(active_tab, fill_data, n_intervals, selected_symbols, selected_accounts):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        # Only update on interval if we're on console tab
        if "log-interval" in trigger_id and active_tab != "tab-console":
            raise PreventUpdate

        if active_tab == "tab-console":
            log_text = _get_log_messages()
            return html.Div([
                html.H4("Console Output"),
                html.P("Live log messages from the fill events manager:", className="text-muted"),
                html.Pre(
                    log_text or "No log messages yet...",
                    id="console-log-content",
                    style={
                        "backgroundColor": "#1e1e1e",
                        "color": "#d4d4d4",
                        "padding": "15px",
                        "borderRadius": "5px",
                        "maxHeight": "500px",
                        "overflowY": "auto",
                        "fontFamily": "monospace",
                        "fontSize": "12px",
                    }
                ),
            ])

        df = pd.DataFrame(fill_data) if fill_data else pd.DataFrame()

        if active_tab == "tab-overview":
            if df.empty:
                return html.Div([
                    html.H4("No Data"),
                    html.P("Select accounts and date range, then click 'Refresh' to fetch data."),
                ], className="text-center mt-5")

            figs = build_figures(df)
            # Sort by timestamp (numeric) for correct ordering, use datetime_str for display
            recent = (
                df.sort_values("timestamp", ascending=False)
                .head(200)
                .to_dict(orient="records")
            )
            # Define columns: datetime_str for readable display, exclude fee_cost from main view
            # (fees are inconsistent across exchanges)
            table_columns = [
                {"name": "Time", "id": "datetime_str"},
                {"name": "Account", "id": "account"},
                {"name": "Symbol", "id": "symbol"},
                {"name": "Side", "id": "side"},
                {"name": "Pos Side", "id": "position_side"},
                {"name": "Qty", "id": "qty", "type": "numeric", "format": {"specifier": ".6f"}},
                {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": ".4f"}},
                {"name": "PnL", "id": "pnl", "type": "numeric", "format": {"specifier": ".4f"}},
                {"name": "Type", "id": "pb_order_type"},
            ]
            return html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=figs[0]), md=6),
                    dbc.Col(dcc.Graph(figure=figs[1]), md=6),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=figs[2]), md=6),
                    dbc.Col(dcc.Graph(figure=figs[3]), md=6),
                ]),
                html.H4("Recent fills", className="mt-4"),
                dash_table.DataTable(
                    columns=table_columns,
                    data=recent,
                    page_size=25,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell_conditional=[
                        {"if": {"column_id": "datetime_str"}, "width": "160px"},
                        {"if": {"column_id": "account"}, "width": "120px"},
                        {"if": {"column_id": "symbol"}, "width": "100px"},
                    ],
                ),
            ])

        elif active_tab == "tab-symbol":
            if df.empty:
                return html.Div("No data available. Please refresh first.", className="text-muted")

            unique_symbols = sorted(df["symbol"].unique())
            selected_symbol = selected_symbols[0] if selected_symbols else (unique_symbols[0] if unique_symbols else None)

            if not selected_symbol:
                return html.Div("No symbols available.", className="text-muted")

            fig = build_symbol_chart(df, selected_symbol)

            # Symbol stats (PnL without fees since fees are inconsistent)
            symbol_df = df[df["symbol"] == selected_symbol]
            stats = {
                "PnL": f"{symbol_df['pnl'].sum():.4f}",
                "Trades": len(symbol_df),
                "Buys": len(symbol_df[symbol_df["side"] == "buy"]),
                "Sells": len(symbol_df[symbol_df["side"] == "sell"]),
                "Avg Price": f"{symbol_df['price'].mean():.4f}" if not symbol_df.empty else "N/A",
            }

            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Symbol"),
                        dcc.Dropdown(
                            id="symbol-detail-dropdown",
                            options=[{"label": s, "value": s} for s in unique_symbols],
                            value=selected_symbol,
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            dbc.Badge(f"{k}: {v}", color="secondary", className="me-2 mb-1")
                            for k, v in stats.items()
                        ])
                    ], md=8),
                ], className="mb-3"),
                dcc.Graph(figure=fig),
                html.P(
                    "Tip: Select a symbol from the dropdown or filter in the main Symbols dropdown.",
                    className="text-muted mt-2",
                ),
            ])

        elif active_tab == "tab-health":
            # Load health data lazily only when this tab is active
            selected = selected_accounts or list(accounts.keys())
            health_summaries = _get_coverage_summaries(accounts, selected)
            return html.Div([
                html.H4("Cache Health Status"),
                html.P("Shows coverage and known gaps for each account's fill event cache.", className="text-muted"),
                build_cache_health_panel(health_summaries),
            ])

        elif active_tab == "tab-export":
            if df.empty:
                return html.Div("No data to export. Please refresh first.", className="text-muted")
            # Select columns for preview, use datetime_str for display
            preview_cols = ["datetime_str", "account", "symbol", "side", "position_side", "qty", "price", "pnl", "pb_order_type"]
            preview_cols = [c for c in preview_cols if c in df.columns]
            return html.Div([
                html.H4("Export Fill Events"),
                html.P(f"Ready to export {len(df)} fill events.", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Download CSV", id="btn-csv", color="primary", n_clicks=0),
                    ], md=2),
                    dbc.Col([
                        dbc.Button("Download JSON", id="btn-json", color="secondary", n_clicks=0),
                    ], md=2),
                ], className="mb-4"),
                html.H5("Preview (first 10 rows)"),
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in preview_cols],
                    data=df.head(10)[preview_cols].to_dict(orient="records"),
                    style_table={"overflowX": "auto"},
                ),
            ])

        return html.Div("Select a tab to view content.")

    @app.callback(
        Output("download-csv", "data"),
        Input("btn-csv", "n_clicks"),
        Input("btn-json", "n_clicks"),
        State("fill-data", "data"),
        prevent_initial_call=True,
    )
    def download_data(n_csv, n_json, fill_data):
        if not fill_data:
            raise PreventUpdate
        df = pd.DataFrame(fill_data)
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        # Drop internal columns, keep datetime_str as the readable timestamp
        drop_cols = ["raw", "fees", "datetime", "date", "fee_cost", "pnl_with_fees"]
        export_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        # Rename datetime_str to datetime for export clarity
        if "datetime_str" in export_df.columns:
            export_df = export_df.rename(columns={"datetime_str": "datetime"})

        if "btn-csv" in trigger:
            return dcc.send_data_frame(export_df.to_csv, "fill_events.csv", index=False)
        elif "btn-json" in trigger:
            export_data = export_df.to_dict(orient="records")
            return dict(content=json.dumps(export_data, indent=2), filename="fill_events.json")
        raise PreventUpdate

    # Run server
    logging.info(f"Starting dashboard on http://localhost:{port}")
    logging.info("Press Ctrl+C to stop the server")

    def force_exit():
        logging.info("Force exiting...")
        os._exit(0)

    def shutdown_handler(signum, frame):
        logging.info("Shutting down...")
        # Schedule force exit after 2 seconds if graceful shutdown fails
        threading.Timer(2.0, force_exit).start()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        app.run_server(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        force_exit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill events dashboard")
    parser.add_argument("--config", default="configs/template.json", help="Config path")
    parser.add_argument(
        "--users",
        required=True,
        help="Comma-separated list of live.user identifiers",
    )
    parser.add_argument(
        "--cache-root",
        default="caches/fill_events",
        help="Root directory for fill events cache",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbol override list",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Default lookback window for initial view",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Logging verbosity (warning/info/debug/trace or 0-3)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run dashboard on (default: 8050)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    users = [u.strip() for u in args.users.split(",") if u.strip()]
    accounts = _build_managers(users, args.config, args.cache_root, args.symbols)
    if not accounts:
        logging.error("No accounts could be loaded. Check your --users argument and api-keys.json")
        sys.exit(1)
    serve_dash(accounts, default_days=args.lookback_days, port=args.port)


if __name__ == "__main__":
    main()
