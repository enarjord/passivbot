from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

import aiohttp


def _now_ms() -> int:
    return int(time.time() * 1000.0)


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_ts_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(float(value) / 1000.0).astimezone()
    except (TypeError, ValueError, OSError):
        return str(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_age_ms(ts_ms: Any) -> str:
    if ts_ms is None:
        return "-"
    try:
        delta_ms = max(0, _now_ms() - int(float(ts_ms)))
    except (TypeError, ValueError):
        return "-"
    if delta_ms < 1000:
        return f"{delta_ms}ms"
    seconds = delta_ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def _fmt_uptime_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        seconds = int(float(value) / 1000.0)
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _fmt_pct_ratio(value: Any) -> str:
    if value is None:
        return "-"
    try:
        pct = int(round(float(value) * 100.0))
    except (TypeError, ValueError):
        return str(value)
    return f"{pct:>3d}%"


def _account_realized_value(account: dict[str, Any]) -> Any:
    realized = account.get("realized_pnl_cumsum")
    if isinstance(realized, dict):
        return realized.get("current")
    return realized


def _build_query_params(exchange: Optional[str], user: Optional[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    if exchange:
        params["exchange"] = exchange
    if user:
        params["user"] = user
    return params


def _append_query(url: str, params: dict[str, str]) -> str:
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"


def _http_to_ws(url: str) -> str:
    parts = urlsplit(url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def _read_last_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()[-max_lines:]
    except FileNotFoundError:
        return []


@dataclass
class _LogTailState:
    dev: int
    ino: int
    offset: int


@dataclass
class MonitorTuiState:
    relay_url: str
    exchange: Optional[str] = None
    user: Optional[str] = None
    focus_symbol: Optional[str] = None
    snapshot: dict[str, Any] = field(default_factory=dict)
    snapshot_seq: Optional[int] = None
    snapshot_ts_ms: Optional[int] = None
    ws_connected: bool = False
    status_text: str = "starting"
    last_error: Optional[str] = None
    last_ws_message_ts_ms: Optional[int] = None
    command_buffer: str = ""
    command_status: str = "Type 'help' for commands."
    followed_log_file: Optional[str] = None
    recent_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=12))
    recent_price_ticks: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_candles: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=12))

    def apply_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "snapshot":
            self._apply_snapshot_message(message)
            return
        if message_type == "event":
            self.recent_events.appendleft(message)
            self.last_ws_message_ts_ms = int(message.get("ts", _now_ms()))
            return
        if message_type == "history":
            self.last_ws_message_ts_ms = int(message.get("ts", _now_ms()))
            stream = str(message.get("stream", ""))
            symbol = str(message.get("symbol", "?"))
            if stream == "price_ticks":
                self.recent_price_ticks[symbol] = message
            elif stream.startswith("candles_"):
                timeframe = str(message.get("timeframe", "?"))
                self.recent_candles[f"{symbol}|{timeframe}"] = message
            return
        if message_type == "resync_required":
            self.status_text = "resync required"
            self.last_error = str(message.get("reason", "unknown"))

    def set_log_file(self, path: Optional[str]) -> None:
        self.followed_log_file = path

    def push_log_lines(self, lines: list[str]) -> None:
        for line in lines:
            cleaned = line.rstrip("\n")
            if cleaned:
                self.recent_log_lines.append(cleaned)

    def _apply_snapshot_message(self, message: dict[str, Any]) -> None:
        payload = message.get("payload")
        if isinstance(payload, dict):
            self.snapshot = payload
        self.snapshot_seq = message.get("seq")
        self.snapshot_ts_ms = message.get("ts")
        self.exchange = message.get("exchange") or self.snapshot.get("meta", {}).get("exchange") or self.exchange
        self.user = message.get("user") or self.snapshot.get("meta", {}).get("user") or self.user
        self.status_text = "snapshot refreshed"


def _active_position_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    positions = snapshot.get("positions", {})
    open_orders = snapshot.get("open_orders", {})
    market = snapshot.get("market", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(positions, dict):
        return rows
    for symbol in sorted(positions):
        entry = positions.get(symbol, {})
        if not isinstance(entry, dict):
            continue
        long_pos = entry.get("long", {}) if isinstance(entry.get("long"), dict) else {}
        short_pos = entry.get("short", {}) if isinstance(entry.get("short"), dict) else {}
        long_size = float(long_pos.get("size", 0.0) or 0.0)
        short_size = float(short_pos.get("size", 0.0) or 0.0)
        orders = open_orders.get(symbol, [])
        if not long_size and not short_size and not orders:
            continue
        last_price = None
        market_entry = market.get(symbol, {}) if isinstance(market.get(symbol), dict) else {}
        if market_entry:
            last_price = market_entry.get("last_price")
        rows.append(
            {
                "symbol": symbol,
                "label": _symbol_label(symbol),
                "long": long_pos,
                "short": short_pos,
                "orders": len(orders) if isinstance(orders, list) else 0,
                "last_price": last_price,
            }
        )
    rows.sort(
        key=lambda row: (
            abs(float((row["long"] or {}).get("wallet_exposure", 0.0) or 0.0))
            + abs(float((row["short"] or {}).get("wallet_exposure", 0.0) or 0.0)),
            abs(float((row["long"] or {}).get("size", 0.0) or 0.0))
            + abs(float((row["short"] or {}).get("size", 0.0) or 0.0)),
            row["orders"],
        ),
        reverse=True,
    )
    return rows


def _format_event_line(message: dict[str, Any]) -> str:
    parts = [_fmt_ts_ms(message.get("ts")), str(message.get("kind", "?"))]
    symbol = message.get("symbol")
    if symbol:
        parts.append(str(symbol))
    pside = message.get("pside")
    if pside:
        parts.append(f"[{pside}]")
    payload = message.get("payload", {})
    if isinstance(payload, dict):
        if "equity" in payload:
            parts.append(f"eq={_fmt_float(payload.get('equity'), 2)}")
        elif "price" in payload and "qty" in payload:
            parts.append(f"qty={_fmt_float(payload.get('qty'))}@{_fmt_float(payload.get('price'), 4)}")
        elif "status" in payload:
            parts.append(f"status={payload.get('status')}")
    return " | ".join(parts)


def _format_tick_line(symbol: str, message: dict[str, Any]) -> str:
    payload = message.get("payload", {}) if isinstance(message.get("payload"), dict) else {}
    return (
        f"{symbol:<18} last={_fmt_float(payload.get('last'), 4):>12} "
        f"age={_fmt_age_ms(message.get('ts')):>6}"
    )


def _parse_symbol_parts(symbol: str) -> tuple[str, str]:
    base = symbol
    quote = ""
    if "/" in symbol:
        base, rest = symbol.split("/", 1)
        quote = rest.split(":", 1)[0]
    return base.upper(), quote.upper()


def _symbol_label(symbol: str) -> str:
    base, quote = _parse_symbol_parts(symbol)
    if base and quote:
        return f"{base}/{quote}"
    return symbol


def _symbol_aliases(symbol: str) -> set[str]:
    base, quote = _parse_symbol_parts(symbol)
    aliases = {symbol.upper()}
    if base:
        aliases.add(base)
    if quote:
        aliases.add(f"{base}/{quote}")
        aliases.add(f"{base}{quote}")
    return aliases


def _available_focus_symbols(snapshot: dict[str, Any], state: Optional[MonitorTuiState] = None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(symbol: Any) -> None:
        if not symbol:
            return
        normalized = str(symbol)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    for row in _active_position_rows(snapshot):
        add(row.get("symbol"))
    market = snapshot.get("market", {}) if isinstance(snapshot, dict) else {}
    if isinstance(market, dict):
        for symbol in sorted(market):
            add(symbol)
    if state is not None:
        for message in state.recent_events:
            add(message.get("symbol"))
        for symbol in sorted(state.recent_price_ticks):
            add(symbol)
    return ordered


def resolve_focus_symbol_alias(
    query: str,
    snapshot: dict[str, Any],
    *,
    state: Optional[MonitorTuiState] = None,
) -> tuple[Optional[str], Optional[str]]:
    cleaned = query.strip().upper()
    if not cleaned:
        return None, "focus requires a symbol, 'auto', 'next', or 'prev'"
    matches: list[str] = []
    for symbol in _available_focus_symbols(snapshot, state):
        if cleaned in _symbol_aliases(symbol):
            matches.append(symbol)
    if not matches:
        return None, f"no symbol matched '{query}'"
    if len(matches) > 1:
        joined = ", ".join(matches[:4])
        suffix = " ..." if len(matches) > 4 else ""
        return None, f"ambiguous symbol '{query}': {joined}{suffix}"
    return matches[0], None


def _select_focus_symbol(state: MonitorTuiState, snapshot: dict[str, Any]) -> Optional[str]:
    market = snapshot.get("market", {}) if isinstance(snapshot, dict) else {}
    if state.focus_symbol:
        return state.focus_symbol if state.focus_symbol in market or not market else state.focus_symbol
    active_rows = _active_position_rows(snapshot)
    if active_rows:
        return str(active_rows[0]["symbol"])
    for message in state.recent_events:
        symbol = message.get("symbol")
        if symbol:
            return str(symbol)
    if state.recent_price_ticks:
        symbol, _ = max(
            state.recent_price_ticks.items(),
            key=lambda item: item[1].get("ts", 0),
        )
        return str(symbol)
    return None


def _filtered_recent_events(state: MonitorTuiState, focus_symbol: Optional[str]) -> list[dict[str, Any]]:
    events = list(state.recent_events)
    if not focus_symbol:
        return events[:8]
    focused = [message for message in events if message.get("symbol") == focus_symbol]
    others = [message for message in events if message.get("symbol") != focus_symbol]
    return (focused + others)[:8]


def _filtered_price_ticks(state: MonitorTuiState, focus_symbol: Optional[str]) -> list[tuple[str, dict[str, Any]]]:
    items = sorted(
        state.recent_price_ticks.items(),
        key=lambda item: item[1].get("ts", 0),
        reverse=True,
    )
    if not focus_symbol:
        return items[:8]
    focused = [item for item in items if item[0] == focus_symbol]
    others = [item for item in items if item[0] != focus_symbol]
    return (focused + others)[:8]


def _recent_order_activity(snapshot: dict[str, Any], focus_symbol: Optional[str]) -> list[dict[str, Any]]:
    recent = snapshot.get("recent", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(recent, dict):
        return []
    combined: list[dict[str, Any]] = []
    for key, action in (("order_executions", "executed"), ("order_cancellations", "canceled")):
        entries = recent.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = entry.get("symbol")
            if focus_symbol and symbol and symbol != focus_symbol:
                continue
            merged = dict(entry)
            merged["_action"] = action
            merged["_ts"] = entry.get("execution_timestamp") or entry.get("timestamp") or 0
            combined.append(merged)
    combined.sort(key=lambda entry: entry.get("_ts", 0), reverse=True)
    return combined[:6]


def _format_order_activity_line(entry: dict[str, Any]) -> str:
    symbol = str(entry.get("symbol", "?"))
    action = str(entry.get("_action", "?"))
    pside = str(entry.get("position_side", "-"))
    side = str(entry.get("side", "-"))
    qty = _fmt_float(entry.get("qty"))
    price = _fmt_float(entry.get("price"), 4)
    pb_order_type = str(entry.get("pb_order_type", "?"))
    return (
        f"{_fmt_ts_ms(entry.get('_ts'))} | {action} | {symbol} | {pside}/{side} | "
        f"{qty}@{price} | {pb_order_type}"
    )


def _format_position_side_line(row: dict[str, Any], pside: str) -> str:
    pos = row.get(pside, {}) if isinstance(row.get(pside), dict) else {}
    return (
        f"{str(row.get('label', row.get('symbol', '?'))):<12} "
        f"{pside:<5} "
        f"{_fmt_float(pos.get('size')):>9}@{_fmt_float(pos.get('price'), 4):<12} "
        f"WE={_fmt_float(pos.get('wallet_exposure'), 4):>7} | "
        f"{_fmt_pct_ratio(pos.get('wel_ratio')):>4} WEL "
        f"{_fmt_pct_ratio(pos.get('wele_ratio')):>4} WELe "
        f"{_fmt_pct_ratio(pos.get('twel_ratio')):>4} TWEL | "
        f"PA={_fmt_float(pos.get('price_action_distance'), 4):>8} "
        f"uPnL={_fmt_float(pos.get('upnl'), 3):>9} "
        f"ord={int(row.get('orders', 0) or 0):>2}"
    )


def _format_order_only_line(row: dict[str, Any]) -> str:
    return (
        f"{str(row.get('label', row.get('symbol', '?'))):<12} "
        f"{'flat':<5} "
        f"{'-':>22} "
        f"last={_fmt_float(row.get('last_price'), 4):>12} "
        f"orders={int(row.get('orders', 0) or 0):>2}"
    )


def _render_focus_panel(snapshot: dict[str, Any], focus_symbol: Optional[str]) -> list[str]:
    if not focus_symbol:
        return ["Focus", "(no focus symbol selected yet)"]
    market = snapshot.get("market", {}) if isinstance(snapshot, dict) else {}
    positions = snapshot.get("positions", {}) if isinstance(snapshot, dict) else {}
    open_orders = snapshot.get("open_orders", {}) if isinstance(snapshot, dict) else {}
    forager = snapshot.get("forager", {}) if isinstance(snapshot, dict) else {}
    market_entry = market.get(focus_symbol, {}) if isinstance(market.get(focus_symbol), dict) else {}
    pos_entry = positions.get(focus_symbol, {}) if isinstance(positions.get(focus_symbol), dict) else {}
    long_pos = pos_entry.get("long", {}) if isinstance(pos_entry.get("long"), dict) else {}
    short_pos = pos_entry.get("short", {}) if isinstance(pos_entry.get("short"), dict) else {}
    orders = open_orders.get(focus_symbol, []) if isinstance(open_orders.get(focus_symbol), list) else []
    long_forager = forager.get("long", {}) if isinstance(forager.get("long"), dict) else {}
    short_forager = forager.get("short", {}) if isinstance(forager.get("short"), dict) else {}
    lines = [f"Focus | {focus_symbol}"]
    lines.append(
        "  "
        f"last={_fmt_float(market_entry.get('last_price'), 4)} "
        f"age={_fmt_age_ms(market_entry.get('last_price_ts_ms'))} "
        f"tradable={market_entry.get('tradable', '-')} "
        f"active={market_entry.get('active_symbol', '-')} "
        f"orders={len(orders)}"
    )
    approved = market_entry.get("approved", {}) if isinstance(market_entry.get("approved"), dict) else {}
    ignored = market_entry.get("ignored", {}) if isinstance(market_entry.get("ignored"), dict) else {}
    lines.append(
        "  "
        f"long={_fmt_float(long_pos.get('size'))}@{_fmt_float(long_pos.get('price'), 4)} "
        f"short={_fmt_float(short_pos.get('size'))}@{_fmt_float(short_pos.get('price'), 4)} "
        f"approved L/S={approved.get('long', '-')}:{approved.get('short', '-')} "
        f"ignored L/S={ignored.get('long', '-')}:{ignored.get('short', '-')}"
    )
    for pside, pos in (("long", long_pos), ("short", short_pos)):
        size = float(pos.get("size", 0.0) or 0.0)
        if size == 0.0:
            continue
        lines.append(
            "  "
            f"{pside} WE={_fmt_float(pos.get('wallet_exposure'), 4)} | "
            f"{_fmt_pct_ratio(pos.get('wel_ratio'))} WEL "
            f"{_fmt_pct_ratio(pos.get('wele_ratio'))} WELe "
            f"{_fmt_pct_ratio(pos.get('twel_ratio'))} TWEL | "
            f"PA={_fmt_float(pos.get('price_action_distance'), 4)} "
            f"uPnL={_fmt_float(pos.get('upnl'), 3)}"
        )
    lines.append(
        "  "
        f"min_qty={_fmt_float(market_entry.get('min_qty'), 4)} "
        f"min_cost={_fmt_float(market_entry.get('min_cost'), 4)} "
        f"eff_min_cost={_fmt_float(market_entry.get('effective_min_cost'), 4)} "
        f"1m_age={_fmt_age_ms(market_entry.get('last_final_candle_ts_ms'))}"
    )
    lines.append(
        "  "
        f"forager selected L/S="
        f"{focus_symbol in (long_forager.get('selected_symbols') or [])}:"
        f"{focus_symbol in (short_forager.get('selected_symbols') or [])}"
    )
    return lines


def execute_tui_command(state: MonitorTuiState, raw_command: str) -> bool:
    command = raw_command.strip()
    if not command:
        state.command_status = ""
        return False
    lowered = command.lower()
    if lowered in {"quit", "exit"}:
        state.command_status = "Exiting monitor TUI."
        return True
    if lowered == "help":
        state.command_status = (
            "Commands: focus <coin|symbol>, focus auto, focus next, focus prev, clear, help, quit, exit"
        )
        return False
    if lowered == "clear":
        state.command_status = ""
        return False
    parts = command.split()
    if parts[0].lower() == "focus":
        if len(parts) == 1:
            state.command_status = f"Current focus: {state.focus_symbol or 'auto'}"
            return False
        arg = parts[1]
        arg_lower = arg.lower()
        symbols = _available_focus_symbols(state.snapshot, state)
        current = _select_focus_symbol(state, state.snapshot)
        if arg_lower == "auto":
            state.focus_symbol = None
            state.command_status = f"Focus set to auto ({_select_focus_symbol(state, state.snapshot) or '-'})"
            return False
        if arg_lower in {"next", "prev"}:
            if not symbols:
                state.command_status = "No symbols available to cycle."
                return False
            if current in symbols:
                index = symbols.index(current)
            else:
                index = 0
            if arg_lower == "next":
                index = (index + 1) % len(symbols)
            else:
                index = (index - 1) % len(symbols)
            state.focus_symbol = symbols[index]
            state.command_status = f"Focus set to {state.focus_symbol}"
            return False
        resolved, error = resolve_focus_symbol_alias(arg, state.snapshot, state=state)
        if error:
            state.command_status = error
            return False
        state.focus_symbol = resolved
        state.command_status = f"Focus set to {resolved}"
        return False
    state.command_status = f"Unknown command: {command}"
    return False


def render_screen(state: MonitorTuiState, *, width: Optional[int] = None) -> str:
    width = width or shutil.get_terminal_size((120, 40)).columns
    snapshot = state.snapshot
    meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
    account = snapshot.get("account", {}) if isinstance(snapshot, dict) else {}
    health = snapshot.get("health", {}) if isinstance(snapshot, dict) else {}
    hsl = snapshot.get("hsl", {}) if isinstance(snapshot, dict) else {}
    forager = snapshot.get("forager", {}) if isinstance(snapshot, dict) else {}
    unstuck = snapshot.get("unstuck", {}) if isinstance(snapshot, dict) else {}
    focus_symbol = _select_focus_symbol(state, snapshot)

    connection = "connected" if state.ws_connected else "disconnected"
    bot_label = f"{state.exchange or meta.get('exchange') or '?'} / {state.user or meta.get('user') or '?'}"
    lines = [
        f"Passivbot Monitor TUI | {connection} | {bot_label} | relay={state.relay_url}",
        (
            f"snapshot_seq={_fmt_int(state.snapshot_seq or meta.get('seq'))} "
            f"snapshot_age={_fmt_age_ms(state.snapshot_ts_ms or meta.get('snapshot_ts_ms'))} "
            f"ws_age={_fmt_age_ms(state.last_ws_message_ts_ms)} "
            f"focus={focus_symbol or '-'} "
            f"status={state.status_text}"
        ),
    ]
    if state.last_error:
        lines.append(f"last_error={state.last_error}")
    lines.append("")
    lines.append(
        "Account | "
        f"raw={_fmt_float(account.get('balance_raw'), 2)} "
        f"snapped={_fmt_float(account.get('balance_snapped'), 2)} "
        f"equity={_fmt_float(account.get('equity'), 2)} "
        f"realized={_fmt_float(_account_realized_value(account), 2)} "
        f"pid={_fmt_int(meta.get('pid'))}"
    )
    lines.append(
        "Health  | "
        f"uptime={_fmt_uptime_ms(health.get('uptime_ms'))} "
        f"loop={_fmt_float(health.get('last_loop_duration_ms'), 1)}ms "
        f"fills={_fmt_int(health.get('fills'))} "
        f"placed={_fmt_int(health.get('orders_placed'))} "
        f"canceled={_fmt_int(health.get('orders_cancelled'))} "
        f"errors={_fmt_int(health.get('errors_last_hour'))} "
        f"limits={_fmt_int(health.get('rate_limits'))}"
    )
    long_hsl = hsl.get("long", {}) if isinstance(hsl.get("long"), dict) else {}
    short_hsl = hsl.get("short", {}) if isinstance(hsl.get("short"), dict) else {}
    lines.append(
        "HSL     | "
        f"long={long_hsl.get('tier', '-')} halted={long_hsl.get('halted', False)} "
        f"score={_fmt_float((long_hsl.get('last_metrics') or {}).get('drawdown_score'), 4)} | "
        f"short={short_hsl.get('tier', '-')} halted={short_hsl.get('halted', False)} "
        f"score={_fmt_float((short_hsl.get('last_metrics') or {}).get('drawdown_score'), 4)}"
    )
    long_forager = forager.get("long", {}) if isinstance(forager.get("long"), dict) else {}
    short_forager = forager.get("short", {}) if isinstance(forager.get("short"), dict) else {}
    lines.append(
        "Forager | "
        f"long slots={_fmt_int((long_forager.get('slots') or {}).get('current'))}/"
        f"{_fmt_int((long_forager.get('slots') or {}).get('max'))} "
        f"selected={len(long_forager.get('selected_symbols', [])) if isinstance(long_forager.get('selected_symbols'), list) else 0} | "
        f"short slots={_fmt_int((short_forager.get('slots') or {}).get('current'))}/"
        f"{_fmt_int((short_forager.get('slots') or {}).get('max'))} "
        f"selected={len(short_forager.get('selected_symbols', [])) if isinstance(short_forager.get('selected_symbols'), list) else 0} "
        f"active={_fmt_int(health.get('positions_long'))}:{_fmt_int(health.get('positions_short'))}"
    )
    unstuck_sides = unstuck.get("sides", {}) if isinstance(unstuck.get("sides"), dict) else {}
    long_unstuck = unstuck_sides.get("long", {}) if isinstance(unstuck_sides.get("long"), dict) else {}
    short_unstuck = unstuck_sides.get("short", {}) if isinstance(unstuck_sides.get("short"), dict) else {}
    lines.append(
        "Unstuck | "
        f"open_order={unstuck.get('has_open_order', False)} "
        f"long={long_unstuck.get('status', '-')} allowance_live={_fmt_float(long_unstuck.get('allowance_live'), 4)} | "
        f"short={short_unstuck.get('status', '-')} allowance_live={_fmt_float(short_unstuck.get('allowance_live'), 4)}"
    )
    lines.append("")
    lines.extend(_render_focus_panel(snapshot, focus_symbol))
    lines.append("")
    lines.append("Active Positions / Orders")
    rows = _active_position_rows(snapshot)[:8]
    if rows:
        lines.append(
            "symbol/quote  side      size@price            WE metrics                         PA dist / uPnL / orders"
        )
        emitted = 0
        for row in rows:
            rendered_position = False
            for pside in ("long", "short"):
                pos = row.get(pside, {}) if isinstance(row.get(pside), dict) else {}
                if float(pos.get("size", 0.0) or 0.0) == 0.0:
                    continue
                lines.append(_format_position_side_line(row, pside))
                rendered_position = True
                emitted += 1
                if emitted >= 10:
                    break
            if emitted >= 10:
                break
            if not rendered_position and int(row.get("orders", 0) or 0) > 0:
                lines.append(_format_order_only_line(row))
                emitted += 1
                if emitted >= 10:
                    break
    else:
        lines.append("(no active positions or open orders)")
    lines.append("")
    lines.append("Recent Events")
    recent_events = _filtered_recent_events(state, focus_symbol)
    if recent_events:
        for message in recent_events:
            lines.append(_format_event_line(message))
    else:
        lines.append("(no websocket events seen yet)")
    lines.append("")
    lines.append("Recent Price Ticks")
    tick_messages = _filtered_price_ticks(state, focus_symbol)
    if tick_messages:
        for symbol, message in tick_messages:
            lines.append(_format_tick_line(symbol, message))
    else:
        lines.append("(no price ticks seen yet)")
    lines.append("")
    lines.append("Recent Orders")
    recent_orders = _recent_order_activity(snapshot, focus_symbol)
    if recent_orders:
        for entry in recent_orders:
            lines.append(_format_order_activity_line(entry))
    else:
        lines.append("(no recent order activity in snapshot)")
    lines.append("")
    lines.append(
        f"Recent Bot Log{f' | {state.followed_log_file}' if state.followed_log_file else ''}"
    )
    if state.recent_log_lines:
        for line in list(state.recent_log_lines)[-8:]:
            lines.append(line)
    else:
        lines.append("(no log file attached)")
    lines.append("")
    lines.append(f"Cmd: {state.command_status}")
    lines.append(f"> {state.command_buffer}")
    return "\n".join(_truncate(line, width) for line in lines)


class MonitorTuiClient:
    def __init__(
        self,
        *,
        relay_url: str,
        exchange: Optional[str] = None,
        user: Optional[str] = None,
        focus_symbol: Optional[str] = None,
        snapshot_refresh_seconds: float = 2.0,
        render_interval_ms: int = 250,
        log_file: Optional[str] = None,
        log_poll_interval_ms: int = 500,
        log_bootstrap_lines: int = 12,
    ) -> None:
        self.state = MonitorTuiState(
            relay_url=relay_url.rstrip("/"),
            exchange=exchange,
            user=user,
            focus_symbol=focus_symbol,
        )
        self.snapshot_refresh_seconds = max(0.5, float(snapshot_refresh_seconds))
        self.render_interval_ms = max(100, int(render_interval_ms))
        self.log_file = log_file
        self.log_poll_interval_ms = max(100, int(log_poll_interval_ms))
        self.log_bootstrap_lines = max(0, int(log_bootstrap_lines))
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(self._snapshot_loop(session)),
                asyncio.create_task(self._ws_loop(session)),
                asyncio.create_task(self._render_loop()),
            ]
            if self.log_file:
                tasks.append(asyncio.create_task(self._log_tail_loop(Path(self.log_file))))
            if sys.stdin.isatty():
                tasks.append(asyncio.create_task(self._command_loop()))
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def _snapshot_loop(self, session: aiohttp.ClientSession) -> None:
        params = _build_query_params(self.state.exchange, self.state.user)
        snapshot_url = _append_query(urljoin(self.state.relay_url + "/", "snapshot"), params)
        while not self._stop_event.is_set():
            try:
                async with session.get(snapshot_url) as response:
                    response.raise_for_status()
                    message = await response.json()
                self.state.apply_message(message)
                self.state.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.status_text = "snapshot refresh failed"
                self.state.last_error = str(exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.snapshot_refresh_seconds)
            except asyncio.TimeoutError:
                pass

    async def _ws_loop(self, session: aiohttp.ClientSession) -> None:
        params = _build_query_params(self.state.exchange, self.state.user)
        ws_url = _http_to_ws(_append_query(urljoin(self.state.relay_url + "/", "ws"), params))
        while not self._stop_event.is_set():
            try:
                self.state.ws_connected = False
                self.state.status_text = "connecting websocket"
                async with session.ws_connect(ws_url, heartbeat=30.0) as ws:
                    self.state.ws_connected = True
                    self.state.status_text = "websocket connected"
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(msg.data)
                            self.state.apply_message(payload)
                            continue
                        if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
                            break
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError("websocket stream error")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.last_error = str(exc)
                self.state.status_text = "websocket reconnecting"
            finally:
                self.state.ws_connected = False
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    async def _render_loop(self) -> None:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        try:
            while not self._stop_event.is_set():
                screen = render_screen(self.state)
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write(screen)
                sys.stdout.write("\n")
                sys.stdout.flush()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.render_interval_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()

    async def _command_loop(self) -> None:
        fd = sys.stdin.fileno()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()
        original_attrs = termios.tcgetattr(fd)

        def on_ready() -> None:
            try:
                chunk = os.read(fd, 64).decode(errors="ignore")
            except OSError:
                return
            for char in chunk:
                queue.put_nowait(char)

        tty.setcbreak(fd)
        loop.add_reader(fd, on_ready)
        try:
            while not self._stop_event.is_set():
                char = await queue.get()
                if char in {"\r", "\n"}:
                    should_stop = execute_tui_command(self.state, self.state.command_buffer)
                    self.state.command_buffer = ""
                    if should_stop:
                        self.stop()
                        return
                    continue
                if char in {"\x7f", "\b"}:
                    self.state.command_buffer = self.state.command_buffer[:-1]
                    continue
                if char == "\x03":
                    self.stop()
                    return
                if char == "\x1b":
                    continue
                if char.isprintable():
                    self.state.command_buffer += char
        finally:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, original_attrs)

    async def _log_tail_loop(self, path: Path) -> None:
        self.state.set_log_file(str(path))
        path_state: Optional[_LogTailState] = None
        bootstrapped = False
        while not self._stop_event.is_set():
            if not bootstrapped:
                self.state.push_log_lines(_read_last_lines(path, self.log_bootstrap_lines))
                bootstrapped = True
            try:
                stat = path.stat()
            except FileNotFoundError:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.log_poll_interval_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            file_id = (int(stat.st_dev), int(stat.st_ino))
            size = int(stat.st_size)
            if path_state is None:
                path_state = _LogTailState(file_id[0], file_id[1], size)
            else:
                reset = size < path_state.offset or (path_state.dev, path_state.ino) != file_id
                read_from = 0 if reset else path_state.offset
                if size > read_from:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            f.seek(read_from)
                            lines = f.read().splitlines()
                            new_offset = int(f.tell())
                    except FileNotFoundError:
                        lines = []
                        new_offset = read_from
                    if lines:
                        self.state.push_log_lines(lines)
                    path_state = _LogTailState(file_id[0], file_id[1], new_offset)
                else:
                    path_state = _LogTailState(file_id[0], file_id[1], size)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.log_poll_interval_ms / 1000.0
                )
            except asyncio.TimeoutError:
                pass
