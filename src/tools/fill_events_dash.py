"""
Interactive dashboard for exploring cached/fresh fill events.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html

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


def _events_to_dataframe(events: List[dict], account_label: str) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["account"] = account_label
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    df["fee_cost"] = [_normalize_fee_cost(x) for x in df.get("fees", [])]
    df["pnl_with_fees"] = df["pnl"] - df["fee_cost"]
    return df


def _build_managers(
    users: List[str],
    config_path: str,
    cache_root: str,
    symbols_override: Optional[List[str]],
) -> Dict[str, FillEventsManager]:
    managers: Dict[str, FillEventsManager] = {}
    for user in users:
        config = load_config(config_path, verbose=False)
        config = format_config(config, verbose=False)
        config.setdefault("live", {})["user"] = user
        bot = _instantiate_bot(config)
        symbol_pool = _extract_symbol_pool(config, symbols_override)
        fetcher = _build_fetcher_for_bot(bot, symbol_pool)
        cache_path = Path(cache_root) / bot.exchange / bot.user
        managers[f"{bot.exchange}:{bot.user}"] = FillEventsManager(
            exchange=bot.exchange, user=bot.user, fetcher=fetcher, cache_path=cache_path
        )
    return managers


def _ensure_loaded(managers: Dict[str, FillEventsManager]) -> None:
    for m in managers.values():
        asyncio.run(m.ensure_loaded())


def _refresh_range(
    managers: Dict[str, FillEventsManager], selected_accounts: List[str], start_ms: int, end_ms: int
) -> None:
    for account in selected_accounts:
        manager = managers.get(account)
        if manager is None:
            continue
        asyncio.run(manager.refresh_range(start_ms, end_ms))


def _refresh_recent(
    managers: Dict[str, FillEventsManager], selected_accounts: List[str], days: int
) -> None:
    if days <= 0:
        return
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    _refresh_range(managers, selected_accounts, start_ms, end_ms)


def _aggregate_accounts(
    managers: Dict[str, FillEventsManager],
    selected_accounts: List[str],
    start_ms: Optional[int],
    end_ms: Optional[int],
    symbols_filter: Optional[List[str]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for account, manager in managers.items():
        if selected_accounts and account not in selected_accounts:
            continue
        asyncio.run(manager.ensure_loaded())
        events = manager.get_events(start_ms, end_ms)
        frame = _events_to_dataframe([ev.to_dict() for ev in events], account)
        if symbols_filter:
            frame = frame[frame["symbol"].isin(symbols_filter)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_figures(df: pd.DataFrame):
    if df.empty:
        return (
            px.line(title="Cumulative PnL (no data)"),
            px.bar(title="Daily PnL (no data)"),
            px.bar(title="Top Symbols (no data)"),
            px.bar(title="Fees by Account (no data)"),
        )
    df = df.sort_values("datetime")
    df["cum_pnl"] = df.groupby("account")["pnl_with_fees"].cumsum()
    cum_fig = px.line(
        df,
        x="datetime",
        y="cum_pnl",
        color="account",
        title="Cumulative realized PnL (with fees)",
        hover_data=["symbol", "pnl", "fee_cost", "pb_order_type"],
    )
    daily = (
        df.groupby(["date", "account"], as_index=False)
        .agg({"pnl_with_fees": "sum", "pnl": "sum", "fee_cost": "sum"})
        .sort_values("date")
    )
    daily_fig = px.bar(
        daily,
        x="date",
        y="pnl_with_fees",
        color="account",
        title="Daily realized PnL (with fees)",
        barmode="group",
    )
    top_symbols = (
        df.groupby(["symbol", "account"], as_index=False)
        .agg({"pnl_with_fees": "sum", "pnl": "sum", "fee_cost": "sum", "qty": "sum"})
        .sort_values("pnl_with_fees", ascending=False)
        .head(30)
    )
    top_fig = px.bar(
        top_symbols,
        x="symbol",
        y="pnl_with_fees",
        color="account",
        title="Top symbols by realized PnL (with fees)",
    )
    fees = (
        df.groupby("account", as_index=False)
        .agg({"fee_cost": "sum", "pnl": "sum"})
        .sort_values("fee_cost", ascending=False)
    )
    fees_fig = px.bar(fees, x="account", y="fee_cost", title="Fees by account")
    return cum_fig, daily_fig, top_fig, fees_fig


def serve_dash(managers: Dict[str, FillEventsManager], default_days: int = 30) -> None:
    _ensure_loaded(managers)
    now = pd.Timestamp.utcnow()
    start_default = now - pd.Timedelta(days=default_days)

    app: Dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container(
        [
            html.H2("Fill Events Dashboard"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Accounts"),
                            dcc.Dropdown(
                                id="accounts",
                                options=[{"label": k, "value": k} for k in managers.keys()],
                                value=list(managers.keys()),
                                multi=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Symbols (optional)"),
                            dcc.Dropdown(id="symbols", options=[], value=[], multi=True),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Date range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=start_default.date(),
                                start_date=start_default.date(),
                                end_date=now.date(),
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Refresh full range", id="refresh-full", color="primary", n_clicks=0
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Input(id="recent-days", type="number", value=3, min=1, step=1),
                            dbc.Button(
                                "Refresh last N days",
                                id="refresh-recent",
                                color="secondary",
                                n_clicks=0,
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="cum-pnl"), md=6),
                    dbc.Col(dcc.Graph(id="daily-pnl"), md=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="top-symbols"), md=6),
                    dbc.Col(dcc.Graph(id="fees-chart"), md=6),
                ]
            ),
            html.H4("Recent fills"),
            dash_table.DataTable(
                id="fills-table",
                columns=[
                    {"name": c, "id": c}
                    for c in [
                        "datetime",
                        "account",
                        "symbol",
                        "side",
                        "position_side",
                        "qty",
                        "price",
                        "pnl",
                        "fee_cost",
                        "pb_order_type",
                        "client_order_id",
                    ]
                ],
                page_size=25,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
            ),
            dcc.Store(id="fill-data"),
        ],
        fluid=True,
    )

    @app.callback(
        Output("fill-data", "data"),
        Output("symbols", "options"),
        Input("refresh-full", "n_clicks"),
        Input("refresh-recent", "n_clicks"),
        State("accounts", "value"),
        State("symbols", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("recent-days", "value"),
        prevent_initial_call=False,
    )
    def update_data(n_full, n_recent, accounts, symbols, start_date, end_date, recent_days):
        selected_accounts = accounts or list(managers.keys())
        start_ms = int(pd.Timestamp(start_date).timestamp() * 1000) if start_date else None
        end_ms = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else None

        ctx = dash.callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if "refresh-recent" in trigger and recent_days:
            _refresh_recent(managers, selected_accounts, int(recent_days))
        elif "refresh-full" in trigger and start_ms is not None and end_ms is not None:
            _refresh_range(managers, selected_accounts, start_ms, end_ms)

        df = _aggregate_accounts(managers, selected_accounts, start_ms, end_ms, symbols)
        symbols_options = (
            [{"label": s, "value": s} for s in sorted(df["symbol"].unique())] if not df.empty else []
        )
        return df.to_dict(orient="records"), symbols_options

    @app.callback(
        Output("cum-pnl", "figure"),
        Output("daily-pnl", "figure"),
        Output("top-symbols", "figure"),
        Output("fees-chart", "figure"),
        Output("fills-table", "data"),
        Input("fill-data", "data"),
    )
    def update_figures(data):
        if not data:
            empty_df = pd.DataFrame()
            figs = build_figures(empty_df)
            return (*figs, [])
        df = pd.DataFrame(data)
        figs = build_figures(df)
        recent = (
            df.sort_values("datetime", ascending=False)
            .head(200)
            .assign(datetime=lambda x: x["datetime"].astype(str))
            .to_dict(orient="records")
        )
        return (*figs, recent)

    app.run_server(host="0.0.0.0", port=8050, debug=False)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    users = [u.strip() for u in args.users.split(",") if u.strip()]
    managers = _build_managers(users, args.config, args.cache_root, args.symbols)
    serve_dash(managers, default_days=args.lookback_days)


if __name__ == "__main__":
    main()
