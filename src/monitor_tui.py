from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from collections import deque
from copy import deepcopy
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
        return f"{float(value):.{digits}f}"
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


def _wrap_box(title: str, lines: list[str], width: int) -> list[str]:
    width = max(12, width)
    inner = max(1, width - 4)
    out = [
        "+" + "-" * (width - 2) + "+",
        f"| {_truncate(title, inner):<{inner}} |",
        "|" + "-" * (width - 2) + "|",
    ]
    if not lines:
        lines = ["-"]
    for line in lines:
        out.append(f"| {_truncate(line, inner):<{inner}} |")
    out.append("+" + "-" * (width - 2) + "+")
    return out


def _pad_lines(lines: list[str], height: int, width: int) -> list[str]:
    padded = list(lines[:height])
    while len(padded) < height:
        padded.append(" " * width)
    return [line.ljust(width)[:width] for line in padded]


def _combine_columns(left: list[str], right: list[str], left_width: int, right_width: int) -> list[str]:
    height = max(len(left), len(right))
    left_padded = _pad_lines(left, height, left_width)
    right_padded = _pad_lines(right, height, right_width)
    return [f"{l} {r}" for l, r in zip(left_padded, right_padded)]


def _render_screen_diff(previous: Optional[str], current: str) -> str:
    current_lines = current.splitlines()
    if previous is None:
        return f"\x1b[2J\x1b[H{current}\x1b[J\x1b[{len(current_lines) + 1};1H"

    previous_lines = previous.splitlines()
    max_lines = max(len(previous_lines), len(current_lines))
    out: list[str] = []
    for idx in range(max_lines):
        old_line = previous_lines[idx] if idx < len(previous_lines) else None
        new_line = current_lines[idx] if idx < len(current_lines) else ""
        if old_line == new_line:
            continue
        out.append(f"\x1b[{idx + 1};1H{new_line}\x1b[K")
    if len(current_lines) < len(previous_lines):
        out.append(f"\x1b[{len(current_lines) + 1};1H\x1b[J")
    if not out:
        return ""
    out.append(f"\x1b[{len(current_lines) + 1};1H")
    return "".join(out)


def _capture_render_data(state: "MonitorTuiState") -> dict[str, Any]:
    return {
        "snapshot": deepcopy(state.snapshot),
        "snapshot_seq": state.snapshot_seq,
        "snapshot_ts_ms": state.snapshot_ts_ms,
        "ws_connected": state.ws_connected,
        "status_text": state.status_text,
        "last_error": state.last_error,
        "last_ws_message_ts_ms": state.last_ws_message_ts_ms,
        "recent_events": deepcopy(list(state.recent_events)),
        "recent_price_ticks": deepcopy(state.recent_price_ticks),
        "recent_log_lines": deepcopy(list(state.recent_log_lines)),
        "exchange": state.exchange,
        "user": state.user,
        "focus_symbol": state.focus_symbol,
        "followed_log_file": state.followed_log_file,
        "last_submitted_command": state.last_submitted_command,
        "command_status": state.command_status,
    }


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
    last_submitted_command: str = ""
    command_status: str = "Type 'help' for commands."
    paused: bool = False
    paused_render_data: Optional[dict[str, Any]] = None
    last_rendered_screen: Optional[str] = None
    followed_log_file: Optional[str] = None
    recent_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=12))
    recent_price_ticks: dict[str, dict[str, Any]] = field(default_factory=dict)
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
            if stream == "price_ticks":
                symbol = str(message.get("symbol", "?"))
                self.recent_price_ticks[symbol] = message
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
        self.exchange = (
            message.get("exchange")
            or self.snapshot.get("meta", {}).get("exchange")
            or self.exchange
        )
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
        market_entry = market.get(symbol, {}) if isinstance(market.get(symbol), dict) else {}
        rows.append(
            {
                "symbol": symbol,
                "label": _symbol_label(symbol),
                "long": long_pos,
                "short": short_pos,
                "orders": len(orders) if isinstance(orders, list) else 0,
                "last_price": market_entry.get("last_price"),
            }
        )
    rows.sort(
        key=lambda row: (
            abs(float((row["long"] or {}).get("size", 0.0) or 0.0))
            + abs(float((row["short"] or {}).get("size", 0.0) or 0.0)),
            row["orders"],
        ),
        reverse=True,
    )
    return rows


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
    if state.focus_symbol:
        return state.focus_symbol
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


def _format_event_line(message: dict[str, Any]) -> str:
    parts = [_fmt_ts_ms(message.get("ts")), str(message.get("kind", "?"))]
    symbol = message.get("symbol")
    if symbol:
        parts.append(str(symbol))
    payload = message.get("payload", {})
    if isinstance(payload, dict):
        if "equity" in payload:
            parts.append(f"eq={_fmt_float(payload.get('equity'), 2)}")
        elif "price" in payload and "qty" in payload:
            parts.append(f"qty={_fmt_float(payload.get('qty'))}@{_fmt_float(payload.get('price'), 4)}")
        elif "status" in payload:
            parts.append(f"status={payload.get('status')}")
    return " | ".join(parts)


def _filtered_recent_events(state: MonitorTuiState, focus_symbol: Optional[str]) -> list[dict[str, Any]]:
    events = list(state.recent_events)
    if focus_symbol:
        focused = [message for message in events if message.get("symbol") == focus_symbol]
        others = [message for message in events if message.get("symbol") != focus_symbol]
        events = focused + others
    non_balance = [message for message in events if str(message.get("kind")) != "account.balance"]
    balance = [message for message in events if str(message.get("kind")) == "account.balance"]
    collapsed = non_balance[:7]
    if balance:
        collapsed.append(balance[0])
    return collapsed[:8]


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


def execute_tui_command(
    state: MonitorTuiState,
    command: str,
    *,
    dump_dir: Path | str = "tmp",
) -> bool:
    command = str(command or "").strip()
    if not command:
        return False
    state.last_submitted_command = command
    parts = command.split()
    verb = parts[0].lower()
    args = parts[1:]

    if verb in {"quit", "exit", "q"}:
        state.command_status = "Exiting monitor TUI."
        return True

    if verb == "help":
        state.command_status = (
            "Help: focus <coin|symbol> | focus auto|next|prev | "
            "pause | resume | dump | clear | help | quit"
        )
        return False

    if verb == "focus":
        if not args:
            state.command_status = "focus requires a symbol, auto, next, or prev"
            return False
        target = args[0].lower()
        available = _available_focus_symbols(state.snapshot, state)
        if target == "auto":
            state.focus_symbol = None
            state.command_status = "Focus reset to auto."
            return False
        if target in {"next", "prev"}:
            if not available:
                state.command_status = "No symbols available for focus cycling."
                return False
            current = state.focus_symbol
            try:
                index = available.index(current) if current in available else -1
            except ValueError:
                index = -1
            delta = 1 if target == "next" else -1
            state.focus_symbol = available[(index + delta) % len(available)]
            state.command_status = f"Focused {state.focus_symbol}."
            return False
        resolved, error = resolve_focus_symbol_alias(" ".join(args), state.snapshot, state=state)
        if error:
            state.command_status = error
            return False
        state.focus_symbol = resolved
        state.command_status = f"Focused {resolved}."
        return False

    if verb == "pause":
        state.paused = True
        state.paused_render_data = _capture_render_data(state)
        state.command_status = "Paused."
        return False

    if verb == "resume":
        state.paused = False
        state.paused_render_data = None
        state.command_status = "Resumed monitor TUI."
        return False

    if verb == "clear":
        state.recent_events.clear()
        state.recent_price_ticks.clear()
        state.command_status = "Cleared recent relay activity."
        return False

    if verb == "dump":
        dump_root = Path(dump_dir)
        dump_root.mkdir(parents=True, exist_ok=True)
        ts = _now_ms()
        path = dump_root / f"monitor_tui_dump_{ts}.txt"
        screen = state.last_rendered_screen or ""
        path.write_text(screen + ("\n" if screen and not screen.endswith("\n") else ""), encoding="utf-8")
        state.command_status = f"Dumped screen to {path}"
        return False

    state.command_status = f"Unknown command: {command}"
    return False


def render_screen(state: MonitorTuiState, width: Optional[int] = None) -> str:
    if width is None:
        width = max(80, shutil.get_terminal_size((120, 40)).columns - 1)

    if state.paused and state.paused_render_data is not None:
        data = state.paused_render_data
        snapshot = data.get("snapshot", {})
        snapshot_seq = data.get("snapshot_seq")
        snapshot_ts_ms = data.get("snapshot_ts_ms")
        ws_connected = bool(data.get("ws_connected"))
        status_text = str(data.get("status_text", "paused"))
        last_error = data.get("last_error")
        last_ws_message_ts_ms = data.get("last_ws_message_ts_ms")
        recent_events = deque(data.get("recent_events", []), maxlen=12)
        recent_price_ticks = dict(data.get("recent_price_ticks", {}))
        recent_log_lines = deque(data.get("recent_log_lines", []), maxlen=12)
        exchange = data.get("exchange")
        user = data.get("user")
        focus_override = data.get("focus_symbol")
        last_submitted_command = str(data.get("last_submitted_command", state.last_submitted_command))
        command_status = str(data.get("command_status", state.command_status))
    else:
        snapshot = state.snapshot
        snapshot_seq = state.snapshot_seq
        snapshot_ts_ms = state.snapshot_ts_ms
        ws_connected = state.ws_connected
        status_text = state.status_text
        last_error = state.last_error
        last_ws_message_ts_ms = state.last_ws_message_ts_ms
        recent_events = state.recent_events
        recent_price_ticks = state.recent_price_ticks
        recent_log_lines = state.recent_log_lines
        exchange = state.exchange
        user = state.user
        focus_override = state.focus_symbol
        last_submitted_command = state.last_submitted_command
        command_status = state.command_status

    focus_symbol = focus_override or _select_focus_symbol(state, snapshot)
    connection = "connected" if ws_connected else "disconnected"
    mode = "PAUSED" if state.paused else "LIVE"
    header = f"Passivbot Monitor TUI | {mode} | {connection} | {exchange or '-'} / {user or '-'}"
    if focus_symbol:
        header += f" | focus={focus_symbol}"

    account = snapshot.get("account", {}) if isinstance(snapshot, dict) else {}
    health = snapshot.get("health", {}) if isinstance(snapshot, dict) else {}
    hsl = snapshot.get("hsl", {}) if isinstance(snapshot, dict) else {}
    if isinstance(hsl, dict) and "tier" in hsl:
        hsl_tier = str(hsl.get("tier"))
    else:
        hsl_tier = "-"

    summary_lines = [
        (
            f"Account raw={_fmt_float(account.get('balance_raw'), 2)} "
            f"snapped={_fmt_float(account.get('balance_snapped'), 2)} "
            f"equity={_fmt_float(account.get('equity'), 2)}"
        ),
        (
            f"Snapshot seq={snapshot_seq or '-'} age={_fmt_age_ms(snapshot_ts_ms)} "
            f"ws_age={_fmt_age_ms(last_ws_message_ts_ms)} status={status_text}"
        ),
        (
            f"Uptime={_fmt_uptime_ms(health.get('uptime_ms'))} "
            f"fills={health.get('fills', '-')} "
            f"orders=+{health.get('orders_placed', '-')}/-{health.get('orders_cancelled', '-')} "
            f"errors={health.get('errors_last_hour', '-')} hsl={hsl_tier}"
        ),
    ]
    if last_error:
        summary_lines.append(f"Error: {last_error}")

    focus_lines: list[str] = []
    if focus_symbol:
        market = snapshot.get("market", {}) if isinstance(snapshot, dict) else {}
        positions = snapshot.get("positions", {}) if isinstance(snapshot, dict) else {}
        open_orders = snapshot.get("open_orders", {}) if isinstance(snapshot, dict) else {}
        market_entry = market.get(focus_symbol, {}) if isinstance(market.get(focus_symbol), dict) else {}
        position_entry = positions.get(focus_symbol, {}) if isinstance(positions.get(focus_symbol), dict) else {}
        long_pos = position_entry.get("long", {}) if isinstance(position_entry.get("long"), dict) else {}
        short_pos = position_entry.get("short", {}) if isinstance(position_entry.get("short"), dict) else {}
        focus_lines = [
            f"symbol={focus_symbol}",
            (
                f"last={_fmt_float(market_entry.get('last_price'), 4)} "
                f"tradable={market_entry.get('tradable', '-')}"
            ),
            (
                f"long size={_fmt_float(long_pos.get('size'))} "
                f"entry={_fmt_float(long_pos.get('price'), 4)}"
            ),
            (
                f"short size={_fmt_float(short_pos.get('size'))} "
                f"entry={_fmt_float(short_pos.get('price'), 4)}"
            ),
            f"open_orders={len(open_orders.get(focus_symbol, [])) if isinstance(open_orders.get(focus_symbol), list) else 0}",
        ]
    else:
        focus_lines = ["No focus symbol selected."]

    positions_lines: list[str] = []
    for row in _active_position_rows(snapshot)[:8]:
        positions_lines.append(
            f"{row['label']:<12} last={_fmt_float(row.get('last_price'), 4):>10} "
            f"long={_fmt_float((row.get('long') or {}).get('size')):>8} "
            f"short={_fmt_float((row.get('short') or {}).get('size')):>8} "
            f"orders={row.get('orders', 0)}"
        )
    if not positions_lines:
        positions_lines = ["(no active positions or open orders)"]

    event_messages = list(recent_events)
    if focus_symbol:
        focused = [message for message in event_messages if message.get("symbol") == focus_symbol]
        others = [message for message in event_messages if message.get("symbol") != focus_symbol]
        event_messages = focused + others
    non_balance = [message for message in event_messages if str(message.get("kind")) != "account.balance"]
    balance = [message for message in event_messages if str(message.get("kind")) == "account.balance"]
    event_messages = non_balance[:7]
    if balance:
        event_messages.append(balance[0])
    events_lines = [_format_event_line(message) for message in event_messages[:8]]
    if not events_lines:
        events_lines = ["(no recent events)"]

    market = snapshot.get("market", {}) if isinstance(snapshot, dict) else {}
    tick_items = sorted(
        recent_price_ticks.items(),
        key=lambda item: item[1].get("ts", 0),
        reverse=True,
    )
    if focus_symbol:
        focused = [item for item in tick_items if item[0] == focus_symbol]
        others = [item for item in tick_items if item[0] != focus_symbol]
        tick_items = focused + others
    tick_lines = []
    for symbol, message in tick_items[:8]:
        payload = message.get("payload", {}) if isinstance(message.get("payload"), dict) else {}
        tick_lines.append(
            f"{_symbol_label(symbol):<12} last={_fmt_float(payload.get('last'), 4):>11} age={_fmt_age_ms(message.get('ts')):>6}"
        )
    if not tick_lines:
        tick_lines = ["(no recent price ticks)"]

    log_lines = list(recent_log_lines) if recent_log_lines else ["(no local log tail)"]

    if width >= 120:
        left_width = max(50, width // 2 - 1)
        right_width = width - left_width - 1
        left = []
        left.extend(_wrap_box("Summary", summary_lines, left_width))
        left.extend(_wrap_box("Focus", focus_lines, left_width))
        left.extend(_wrap_box("Positions", positions_lines, left_width))
        right = []
        right.extend(_wrap_box("Recent Events", events_lines, right_width))
        right.extend(_wrap_box("Price Ticks", tick_lines, right_width))
        right.extend(_wrap_box(f"Bot Log | {state.followed_log_file or '-'}", log_lines, right_width))
        body = _combine_columns(left, right, left_width, right_width)
    else:
        box_width = width
        body = []
        body.extend(_wrap_box("Summary", summary_lines, box_width))
        body.extend(_wrap_box("Focus", focus_lines, box_width))
        body.extend(_wrap_box("Positions", positions_lines, box_width))
        body.extend(_wrap_box("Recent Events", events_lines, box_width))
        body.extend(_wrap_box("Price Ticks", tick_lines, box_width))
        body.extend(_wrap_box(f"Bot Log | {state.followed_log_file or '-'}", log_lines, box_width))

    footer = [
        f"Status: {command_status}",
        f"> {last_submitted_command}",
    ]
    screen = "\n".join([header, *body, *footer])
    state.last_rendered_screen = screen
    return screen


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
    ) -> None:
        self.state = MonitorTuiState(
            relay_url=relay_url,
            exchange=exchange,
            user=user,
            focus_symbol=focus_symbol,
        )
        self.snapshot_refresh_seconds = max(0.25, float(snapshot_refresh_seconds))
        self.render_interval_ms = max(50, int(render_interval_ms))
        self._stop_event = asyncio.Event()

    def _snapshot_url(self) -> str:
        params = _build_query_params(self.state.exchange, self.state.user)
        return _append_query(urljoin(self.state.relay_url.rstrip("/") + "/", "snapshot"), params)

    def _ws_url(self) -> str:
        params = _build_query_params(self.state.exchange, self.state.user)
        http_url = _append_query(urljoin(self.state.relay_url.rstrip("/") + "/", "ws"), params)
        return _http_to_ws(http_url)

    async def _snapshot_loop(self, session: aiohttp.ClientSession) -> None:
        while not self._stop_event.is_set():
            try:
                async with session.get(self._snapshot_url()) as response:
                    response.raise_for_status()
                    payload = await response.json()
                self.state.apply_message(payload)
                self.state.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.last_error = f"snapshot: {exc}"
                self.state.status_text = "snapshot error"
            await asyncio.sleep(self.snapshot_refresh_seconds)

    async def _ws_loop(self, session: aiohttp.ClientSession) -> None:
        while not self._stop_event.is_set():
            try:
                async with session.ws_connect(self._ws_url(), heartbeat=30.0) as ws:
                    self.state.ws_connected = True
                    self.state.last_error = None
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(message.data)
                            if isinstance(payload, dict):
                                self.state.apply_message(payload)
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"websocket error: {ws.exception()}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.last_error = f"ws: {exc}"
                self.state.status_text = "ws reconnecting"
            finally:
                self.state.ws_connected = False
            await asyncio.sleep(1.0)

    async def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            screen = render_screen(self.state)
            previous = getattr(self, "_previous_screen", None)
            diff = _render_screen_diff(previous, screen)
            if diff:
                sys.stdout.write(diff)
                sys.stdout.flush()
            self._previous_screen = screen
            await asyncio.sleep(self.render_interval_ms / 1000.0)

    async def _command_loop(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            await self._stop_event.wait()
            return
        while not self._stop_event.is_set():
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
            except asyncio.CancelledError:
                raise
            if line == "":
                await asyncio.sleep(0.1)
                continue
            should_stop = execute_tui_command(self.state, line)
            if should_stop:
                self._stop_event.set()

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(self._snapshot_loop(session)),
                asyncio.create_task(self._ws_loop(session)),
                asyncio.create_task(self._render_loop()),
                asyncio.create_task(self._command_loop()),
            ]
            try:
                await self._stop_event.wait()
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                sys.stdout.write("\n")
                sys.stdout.flush()
