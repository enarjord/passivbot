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
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

import aiohttp


ANSI_RESET = "\x1b[0m"
ANSI_BOLD_CYAN = "\x1b[1;36m"
ANSI_BOLD_GREEN = "\x1b[1;32m"
ANSI_BOLD_YELLOW = "\x1b[1;33m"
ANSI_BOLD_RED = "\x1b[1;31m"
ANSI_DIM = "\x1b[2m"


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


def _fmt_compact_float(
    value: Any,
    *,
    digits: int = 4,
    zero: str = "0",
    none: str = "-",
) -> str:
    if value is None:
        return none
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) < 10 ** (-digits):
        return zero
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", "-0.00", "-0.000", "-0.0000"}:
        return zero
    return text


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
        dt = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return str(value)
    return dt.strftime("%Y-%m-%d %H:%M:%SZ")


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


def _wrap_text(text: str, width: int) -> list[str]:
    if width <= 0:
        return [text]
    if not text:
        return [""]
    lines: list[str] = []
    for chunk in text.splitlines() or [text]:
        words = chunk.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _style(text: str, ansi_code: str) -> str:
    return f"{ansi_code}{text}{ANSI_RESET}"


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


def _fmt_pct_delta(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        pct = float(value) * 100.0
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if pct >= 0.0 else ""
    return f"{sign}{pct:.{digits}f}%"


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
        "recent_candles": deepcopy(state.recent_candles),
        "recent_log_lines": deepcopy(list(state.recent_log_lines)),
        "exchange": state.exchange,
        "user": state.user,
        "focus_symbol": state.focus_symbol,
        "followed_log_file": state.followed_log_file,
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


def _message_bot_key(message: dict[str, Any]) -> Optional[tuple[str, str]]:
    exchange = message.get("exchange")
    user = message.get("user")
    if exchange and user:
        return str(exchange), str(user)
    payload = message.get("payload")
    if isinstance(payload, dict):
        meta = payload.get("meta", {})
        exchange = meta.get("exchange")
        user = meta.get("user")
        if exchange and user:
            return str(exchange), str(user)
    return None


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
    last_submitted_command: str = ""
    command_status: str = "Type 'help' for commands."
    paused: bool = False
    paused_render_data: Optional[dict[str, Any]] = None
    last_rendered_screen: Optional[str] = None
    followed_log_file: Optional[str] = None
    recent_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=12))
    recent_price_ticks: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_candles: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=12))

    def apply_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "snapshot_bundle":
            self._apply_snapshot_bundle(message)
            return
        if not self._should_accept_message(message):
            return
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

    def _apply_snapshot_bundle(self, message: dict[str, Any]) -> None:
        candidates = message.get("bots", [])
        if not isinstance(candidates, list):
            return
        selected = self._select_snapshot_from_bundle(candidates)
        if selected is not None:
            self._apply_snapshot_message(selected)

    def _should_accept_message(self, message: dict[str, Any]) -> bool:
        key = _message_bot_key(message)
        if key is None:
            return True
        if self.exchange and self.user:
            return key == (self.exchange, self.user)
        self.exchange, self.user = key
        return True

    def _select_snapshot_from_bundle(self, candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if self.exchange and self.user:
            for candidate in candidates:
                if _message_bot_key(candidate) == (self.exchange, self.user):
                    return candidate
        if candidates:
            return candidates[0]
        return None


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


def _format_tick_line(symbol: str, message: dict[str, Any], market_entry: Optional[dict[str, Any]] = None) -> str:
    payload = message.get("payload", {}) if isinstance(message.get("payload"), dict) else {}
    market_entry = market_entry if isinstance(market_entry, dict) else {}
    ema_lower, ema_upper = _market_outer_band_bounds(market_entry)
    return (
        f"{_symbol_label(symbol):<12} last={_fmt_float(payload.get('last'), 4):>11} "
        f"lo={ema_lower:<10} hi={ema_upper:<10} "
        f"age={_fmt_age_ms(message.get('ts')):>6}"
    )


def _fmt_market_band_snapshot(
    market_entry: dict[str, Any],
    pside: str,
    *,
    include_trigger: bool = False,
) -> str:
    ema_bands = market_entry.get("ema_bands", {}) if isinstance(market_entry, dict) else {}
    side_bands = ema_bands.get(pside, {}) if isinstance(ema_bands, dict) else {}
    if not isinstance(side_bands, dict) or not side_bands:
        return "-"
    lower = _fmt_float(side_bands.get("lower"), 4)
    upper = _fmt_float(side_bands.get("upper"), 4)
    if include_trigger:
        trigger = _fmt_float(side_bands.get("unstuck_trigger_price"), 4)
        return f"{lower}..{upper} trg={trigger}"
    return f"{lower}..{upper}"

def _market_outer_band_bounds(market_entry: dict[str, Any]) -> tuple[str, str]:
    ema_bands = market_entry.get("ema_bands", {}) if isinstance(market_entry, dict) else {}
    if not isinstance(ema_bands, dict):
        return "-", "-"
    lowers: list[float] = []
    uppers: list[float] = []
    for side_bands in ema_bands.values():
        if not isinstance(side_bands, dict):
            continue
        try:
            lower = float(side_bands.get("lower"))
            upper = float(side_bands.get("upper"))
        except (TypeError, ValueError):
            continue
        lowers.append(lower)
        uppers.append(upper)
    if not lowers or not uppers:
        return "-", "-"
    return (
        _fmt_compact_float(min(lowers), digits=4),
        _fmt_compact_float(max(uppers), digits=4),
    )


def _render_positions_twe_summary(rows: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []
    for pside in ("long", "short"):
        total_exposure = None
        total_limit = None
        for row in rows:
            pos = row.get(pside, {}) if isinstance(row.get(pside), dict) else {}
            try:
                if total_exposure is None and "total_wallet_exposure" in pos:
                    total_exposure = abs(float(pos.get("total_wallet_exposure", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                if (
                    total_limit is None
                    and float(pos.get("total_wallet_exposure_limit", 0.0) or 0.0) > 0.0
                ):
                    total_limit = float(pos.get("total_wallet_exposure_limit", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            if total_limit is None:
                try:
                    total_we = abs(float(pos.get("total_wallet_exposure", 0.0) or 0.0))
                    twel_ratio = abs(float(pos.get("twel_ratio", 0.0) or 0.0))
                except (TypeError, ValueError):
                    total_we = 0.0
                    twel_ratio = 0.0
                if total_we > 0.0 and twel_ratio > 0.0:
                    total_limit = total_we / twel_ratio
        if total_exposure is None:
            total_exposure = 0.0
        if total_limit is not None and total_limit > 0.0:
            parts.append(
                f"{pside}={_fmt_float(total_exposure, 4)}/{_fmt_float(total_limit, 4)}"
                f" ({_fmt_pct_ratio(total_exposure / total_limit)})"
            )
        else:
            parts.append(f"{pside}={_fmt_float(total_exposure, 4)}/- (-)")
    return [f"TWE total | {' | '.join(parts)}"]


def _render_forager_panel(snapshot: dict[str, Any]) -> list[str]:
    forager = snapshot.get("forager", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(forager, dict):
        return ["(no forager snapshot data)"]
    lines: list[str] = []
    for pside in ("long", "short"):
        entry = forager.get(pside, {}) if isinstance(forager.get(pside), dict) else {}
        slots = entry.get("slots", {}) if isinstance(entry.get("slots"), dict) else {}
        selected = entry.get("selected_symbols", []) if isinstance(entry.get("selected_symbols"), list) else []
        pending = entry.get("pending_symbols", []) if isinstance(entry.get("pending_symbols"), list) else []
        next_symbol = entry.get("next_symbol")
        universe = (
            len(entry.get("candidate_universe", []))
            if isinstance(entry.get("candidate_universe"), list)
            else 0
        )
        enabled = bool(entry.get("enabled"))
        max_slots = int(slots.get("max") or 0)
        if not enabled and max_slots <= 0 and universe <= 0:
            continue
        ranking = entry.get("ranking", {}) if isinstance(entry.get("ranking"), dict) else {}
        top_total = ranking.get("top_total", {}) if isinstance(ranking.get("top_total"), dict) else {}
        top_volume = ranking.get("top_volume", {}) if isinstance(ranking.get("top_volume"), dict) else {}
        top_volatility = (
            ranking.get("top_volatility", {}) if isinstance(ranking.get("top_volatility"), dict) else {}
        )
        top_ema = (
            ranking.get("top_ema_readiness", {})
            if isinstance(ranking.get("top_ema_readiness"), dict)
            else {}
        )
        lines.append(
            f"{pside:<5} {'on' if enabled else 'off'} "
            f"slots={_fmt_int(slots.get('current'))}/{_fmt_int(slots.get('max'))} "
            f"open={_fmt_int(slots.get('open'))} sel={len(selected)} pend={len(pending)}"
        )
        next_parts = [
            f"next={_symbol_label(str(next_symbol)) if next_symbol else '-'}",
            f"dist={_fmt_pct_delta(entry.get('next_entry_distance_ratio'))}",
            f"trg={_fmt_compact_float(entry.get('next_entry_trigger_price'), digits=4)}",
            f"uni={universe}",
        ]
        lines.append(f"      {' | '.join(next_parts)}")
        ranking_parts: list[str] = []
        if top_total:
            ranking_parts.append(
                f"total={_symbol_label(str(top_total.get('symbol', '-')))}"
                f"({_fmt_compact_float(top_total.get('total_score'), digits=3)})"
            )
        if top_volume:
            ranking_parts.append(
                f"vol={_symbol_label(str(top_volume.get('symbol', '-')))}"
                f"({_fmt_compact_float(top_volume.get('raw_score'), digits=3)})"
            )
        if top_volatility:
            ranking_parts.append(
                f"vola={_symbol_label(str(top_volatility.get('symbol', '-')))}"
                f"({_fmt_compact_float(top_volatility.get('raw_score'), digits=3)})"
            )
        if top_ema:
            ranking_parts.append(
                f"ema={_symbol_label(str(top_ema.get('symbol', '-')))}"
                f"({_fmt_pct_delta(top_ema.get('raw_score'))})"
            )
        if ranking_parts:
            lines.append("      ranking:")
            lines.append(f"      {' | '.join(ranking_parts)}")
    if not lines:
        return ["(no forager snapshot data)"]
    return lines


def _render_unstuck_panel(snapshot: dict[str, Any]) -> list[str]:
    unstuck = snapshot.get("unstuck", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(unstuck, dict):
        return ["(no unstuck snapshot data)"]
    sides = unstuck.get("sides", {}) if isinstance(unstuck.get("sides"), dict) else {}
    lines = [
        f"open_order={unstuck.get('has_open_order', False)} "
        f"count={len(unstuck.get('open_orders', [])) if isinstance(unstuck.get('open_orders'), list) else 0}"
    ]
    for pside in ("long", "short"):
        entry = sides.get(pside, {}) if isinstance(sides.get(pside), dict) else {}
        status = str(entry.get("status", "-"))
        has_details = any(
            entry.get(key) is not None
            for key in (
                "allowance",
                "allowance_live",
                "next_symbol",
                "next_target_price",
                "next_target_distance_ratio",
                "next_unstuck_trigger_distance_ratio",
            )
        ) or bool(entry.get("ema_bands"))
        if status == "disabled" and not has_details:
            continue
        next_symbol = entry.get("next_symbol")
        ema_bands = entry.get("ema_bands", {}) if isinstance(entry.get("ema_bands"), dict) else {}
        ema_text = (
            f"{_fmt_compact_float(ema_bands.get('lower'), digits=4)}..{_fmt_compact_float(ema_bands.get('upper'), digits=4)}"
            if ema_bands
            else "-"
        )
        lines.append(
            f"{pside:<5} status={status} "
            f"allow={_fmt_compact_float(entry.get('allowance'), digits=4)} "
            f"live={_fmt_compact_float(entry.get('allowance_live'), digits=4)} "
            f"next={_symbol_label(str(next_symbol)) if next_symbol else '-'}"
        )
        lines.append(
            f"      band={ema_text} "
            f"trigger={_fmt_compact_float(ema_bands.get('unstuck_trigger_price'), digits=4)} "
            f"dist={_fmt_pct_delta(entry.get('next_unstuck_trigger_distance_ratio'))} "
            f"target={_fmt_compact_float(entry.get('next_target_price'), digits=4)} "
            f"target_dist={_fmt_pct_delta(entry.get('next_target_distance_ratio'))}"
        )
    return lines


def _render_trailing_panel(snapshot: dict[str, Any]) -> list[str]:
    trailing = snapshot.get("trailing", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(trailing, dict) or not trailing:
        return ["(no trailing entries/closes selected)"]
    lines: list[str] = []
    emitted = 0
    for symbol, symbol_entry in sorted(trailing.items()):
        if not isinstance(symbol_entry, dict):
            continue
        for pside in ("long", "short"):
            side_entry = symbol_entry.get(pside, {}) if isinstance(symbol_entry.get(pside), dict) else {}
            extrema = side_entry.get("extrema", {}) if isinstance(side_entry.get("extrema"), dict) else {}
            for kind in ("entry", "close"):
                entry = side_entry.get(kind, {}) if isinstance(side_entry.get(kind), dict) else {}
                if not entry:
                    continue
                lines.append(
                    f"{_symbol_label(symbol):<10} {pside:<5} {kind:<5} {entry.get('status', '-'):<18} "
                    f"px={_fmt_compact_float(entry.get('price'), digits=4):>10} "
                    f"qty={_fmt_compact_float(entry.get('qty'), digits=4):>8}"
                )
                lines.append(
                    f"      cur={_fmt_compact_float(entry.get('current_price'), digits=4)} "
                    f"thr={_fmt_compact_float(entry.get('threshold_price'), digits=4)} "
                    f"met={entry.get('threshold_met', False)} "
                    f"ret={_fmt_compact_float(entry.get('retracement_price'), digits=4)} "
                    f"met={entry.get('retracement_met', False)}"
                )
                lines.append(
                    f"      min_open={_fmt_compact_float(extrema.get('min_since_open'), digits=4)} "
                    f"max_min={_fmt_compact_float(extrema.get('max_since_min'), digits=4)} "
                    f"max_open={_fmt_compact_float(extrema.get('max_since_open'), digits=4)} "
                    f"min_max={_fmt_compact_float(extrema.get('min_since_max'), digits=4)}"
                )
                emitted += 1
                if emitted >= 4:
                    return lines
    if not lines:
        return ["(no trailing entries/closes selected)"]
    return lines


def _format_position_side_line(row: dict[str, Any], pside: str) -> str:
    pos = row.get(pside, {}) if isinstance(row.get(pside), dict) else {}
    size_text = _fmt_compact_float(pos.get("size"), digits=4)
    price_text = _fmt_compact_float(pos.get("price"), digits=4)
    return (
        f"{str(row.get('label', row.get('symbol', '?'))):<10} "
        f"{pside:<5} "
        f"{size_text:>7}@{price_text:<9} "
        f"WE={_fmt_compact_float(pos.get('wallet_exposure'), digits=4):>6} | "
        f"{_fmt_pct_ratio(pos.get('wel_ratio')):>4}/{_fmt_pct_ratio(pos.get('wele_ratio')):<4} | "
        f"PA={_fmt_compact_float(pos.get('price_action_distance'), digits=4):>7} "
        f"uPnL={_fmt_compact_float(pos.get('upnl'), digits=3):>8} "
        f"o{int(row.get('orders', 0) or 0)}"
    )


def _format_order_only_line(row: dict[str, Any]) -> str:
    return (
        f"{str(row.get('label', row.get('symbol', '?'))):<10} "
        f"{'flat':<5} "
        f"{'-':>17} "
        f"last={_fmt_compact_float(row.get('last_price'), digits=4):>10} "
        f"o{int(row.get('orders', 0) or 0)}"
    )


def _render_focus_panel(snapshot: dict[str, Any], focus_symbol: Optional[str]) -> list[str]:
    if not focus_symbol:
        return ["symbol=-", "(no focus symbol selected yet)"]
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
    lines = [f"symbol={focus_symbol}"]
    lines.append(
        f"last={_fmt_float(market_entry.get('last_price'), 4)} "
        f"age={_fmt_age_ms(market_entry.get('last_price_ts_ms'))} "
        f"tradable={market_entry.get('tradable', '-')} "
        f"active={market_entry.get('active_symbol', '-')} "
        f"orders={len(orders)}"
    )
    approved = market_entry.get("approved", {}) if isinstance(market_entry.get("approved"), dict) else {}
    ignored = market_entry.get("ignored", {}) if isinstance(market_entry.get("ignored"), dict) else {}
    lines.append(
        f"long={_fmt_float(long_pos.get('size'))}@{_fmt_float(long_pos.get('price'), 4)} "
        f"short={_fmt_float(short_pos.get('size'))}@{_fmt_float(short_pos.get('price'), 4)} "
        f"approved L/S={approved.get('long', '-')}:{approved.get('short', '-')} "
        f"ignored L/S={ignored.get('long', '-')}:{ignored.get('short', '-')}"
    )
    lines.append(
        f"min_qty={_fmt_float(market_entry.get('min_qty'), 4)} "
        f"min_cost={_fmt_float(market_entry.get('min_cost'), 4)} "
        f"eff_min_cost={_fmt_float(market_entry.get('effective_min_cost'), 4)} "
        f"1m_age={_fmt_age_ms(market_entry.get('last_final_candle_ts_ms'))}"
    )
    lines.append(
        f"forager selected L/S="
        f"{focus_symbol in (long_forager.get('selected_symbols') or [])}:"
        f"{focus_symbol in (short_forager.get('selected_symbols') or [])}"
    )
    lines.append(
        f"EMA L={_fmt_market_band_snapshot(market_entry, 'long')} "
        f"S={_fmt_market_band_snapshot(market_entry, 'short')}"
    )
    return lines


def _colorize_screen(screen: str) -> str:
    plain_lines = screen.splitlines()
    colored_lines: list[str] = []
    for idx, line in enumerate(plain_lines):
        if idx > 0 and idx + 1 < len(plain_lines):
            if plain_lines[idx - 1].startswith("+") and plain_lines[idx + 1].startswith("|-"):
                colored_lines.append(_style(line, ANSI_BOLD_CYAN))
                continue
        colored = line
        colored = colored.replace(" | connected | ", _style(" | connected | ", ANSI_BOLD_GREEN))
        colored = colored.replace(" | disconnected | ", _style(" | disconnected | ", ANSI_BOLD_RED))
        colored = colored.replace(" | LIVE | ", _style(" | LIVE | ", ANSI_BOLD_GREEN))
        colored = colored.replace(" | PAUSED | ", _style(" | PAUSED | ", ANSI_BOLD_YELLOW))
        colored = colored.replace(" long=green ", _style(" long=green ", ANSI_BOLD_GREEN))
        colored = colored.replace(" short=green ", _style(" short=green ", ANSI_BOLD_GREEN))
        colored = colored.replace(" long=yellow ", _style(" long=yellow ", ANSI_BOLD_YELLOW))
        colored = colored.replace(" short=yellow ", _style(" short=yellow ", ANSI_BOLD_YELLOW))
        colored = colored.replace(" long=red ", _style(" long=red ", ANSI_BOLD_RED))
        colored = colored.replace(" short=red ", _style(" short=red ", ANSI_BOLD_RED))
        if line.startswith("+") or line.startswith("|-"):
            colored = _style(colored, ANSI_DIM)
        colored_lines.append(colored)
    return "\n".join(colored_lines)


def execute_tui_command(
    state: MonitorTuiState,
    raw_command: str,
    *,
    dump_dir: str | Path = "tmp",
) -> bool:
    command = raw_command.strip()
    if not command:
        state.command_status = ""
        return False
    state.last_submitted_command = command
    lowered = command.lower()
    if lowered in {"quit", "exit"}:
        state.command_status = "Exiting monitor TUI."
        return True
    if lowered == "help":
        state.command_status = (
            "Help:\n"
            "focus <coin|symbol> | focus auto|next|prev\n"
            "pause | resume | dump | clear | help | quit"
        )
        return False
    if lowered == "clear":
        state.command_status = ""
        return False
    if lowered == "pause":
        if state.paused:
            state.command_status = "Already paused."
            return False
        state.paused = True
        state.paused_render_data = None
        state.command_status = "Paused. Type 'resume' to refresh or 'dump' to save the current screen."
        return False
    if lowered == "resume":
        if not state.paused:
            state.command_status = "Not paused."
            return False
        state.paused = False
        state.paused_render_data = None
        state.command_status = "Resumed monitor TUI."
        return False
    if lowered == "dump":
        base_dir = Path(dump_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        path = base_dir / f"monitor_tui_dump_{ts}.txt"
        screen = state.last_rendered_screen or render_screen(
            state, display_data=state.paused_render_data
        )
        path.write_text(screen + "\n", encoding="utf-8")
        state.command_status = f"Dumped screen to {path}"
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


def render_screen(
    state: MonitorTuiState,
    *,
    width: Optional[int] = None,
    display_data: Optional[dict[str, Any]] = None,
) -> str:
    width = max(80, width or shutil.get_terminal_size((120, 40)).columns)
    display_data = display_data or _capture_render_data(state)

    view_state = MonitorTuiState(
        relay_url=state.relay_url,
        exchange=display_data.get("exchange"),
        user=display_data.get("user"),
        focus_symbol=state.focus_symbol,
    )
    view_state.snapshot = display_data.get("snapshot", {})
    view_state.snapshot_seq = display_data.get("snapshot_seq")
    view_state.snapshot_ts_ms = display_data.get("snapshot_ts_ms")
    view_state.ws_connected = bool(display_data.get("ws_connected"))
    view_state.status_text = str(display_data.get("status_text", "starting"))
    view_state.last_error = display_data.get("last_error")
    view_state.last_ws_message_ts_ms = display_data.get("last_ws_message_ts_ms")
    view_state.recent_events = deque(display_data.get("recent_events", []), maxlen=12)
    view_state.recent_price_ticks = dict(display_data.get("recent_price_ticks", {}))
    view_state.recent_candles = dict(display_data.get("recent_candles", {}))
    view_state.recent_log_lines = deque(display_data.get("recent_log_lines", []), maxlen=12)
    view_state.followed_log_file = display_data.get("followed_log_file")

    snapshot = view_state.snapshot
    meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
    account = snapshot.get("account", {}) if isinstance(snapshot, dict) else {}
    health = snapshot.get("health", {}) if isinstance(snapshot, dict) else {}
    hsl = snapshot.get("hsl", {}) if isinstance(snapshot, dict) else {}
    forager = snapshot.get("forager", {}) if isinstance(snapshot, dict) else {}
    unstuck = snapshot.get("unstuck", {}) if isinstance(snapshot, dict) else {}
    focus_symbol = _select_focus_symbol(view_state, snapshot)

    connection = "connected" if view_state.ws_connected else "disconnected"
    mode_label = "PAUSED" if state.paused else "LIVE"
    bot_label = f"{view_state.exchange or meta.get('exchange') or '?'} / {view_state.user or meta.get('user') or '?'}"
    header_lines = [
        f"Passivbot Monitor TUI | {mode_label} | {connection} | {bot_label}",
        (
            f"relay={state.relay_url} | seq={_fmt_int(view_state.snapshot_seq or meta.get('seq'))} "
            f"| snapshot_age={_fmt_age_ms(view_state.snapshot_ts_ms or meta.get('snapshot_ts_ms'))} "
            f"| ws_age={_fmt_age_ms(view_state.last_ws_message_ts_ms)} | focus={focus_symbol or '-'}"
        ),
        f"status={view_state.status_text}",
    ]
    if view_state.last_error:
        header_lines.append(f"last_error={view_state.last_error}")

    long_hsl = hsl.get("long", {}) if isinstance(hsl.get("long"), dict) else {}
    short_hsl = hsl.get("short", {}) if isinstance(hsl.get("short"), dict) else {}

    summary_lines = [
        (
            f"Account raw={_fmt_float(account.get('balance_raw'), 2)} "
            f"snapped={_fmt_float(account.get('balance_snapped'), 2)} "
            f"equity={_fmt_float(account.get('equity'), 2)} "
            f"realized={_fmt_float(_account_realized_value(account), 2)} "
            f"pid={_fmt_int(meta.get('pid'))}"
        ),
        (
            f"Health  uptime={_fmt_uptime_ms(health.get('uptime_ms'))} "
            f"loop={_fmt_float(health.get('last_loop_duration_ms'), 1)}ms "
            f"fills={_fmt_int(health.get('fills'))} placed={_fmt_int(health.get('orders_placed'))} "
            f"canceled={_fmt_int(health.get('orders_cancelled'))} "
            f"errors={_fmt_int(health.get('errors_last_hour'))} limits={_fmt_int(health.get('rate_limits'))}"
        ),
        (
            f"HSL     long={long_hsl.get('tier', '-')} halted={long_hsl.get('halted', False)} "
            f"score={_fmt_float((long_hsl.get('last_metrics') or {}).get('drawdown_score'), 4)} | "
            f"short={short_hsl.get('tier', '-')} halted={short_hsl.get('halted', False)} "
            f"score={_fmt_float((short_hsl.get('last_metrics') or {}).get('drawdown_score'), 4)}"
        ),
    ]

    rows = _active_position_rows(snapshot)[:8]
    positions_lines = _render_positions_twe_summary(rows)
    positions_lines.append("symbol      side    size@price        WE | WEL/WELe | PA / uPnL / o")
    if rows:
        emitted = 0
        for row in rows:
            rendered_position = False
            for pside in ("long", "short"):
                pos = row.get(pside, {}) if isinstance(row.get(pside), dict) else {}
                if float(pos.get("size", 0.0) or 0.0) == 0.0:
                    continue
                positions_lines.append(_format_position_side_line(row, pside))
                rendered_position = True
                emitted += 1
                if emitted >= 10:
                    break
            if emitted >= 10:
                break
            if not rendered_position and int(row.get("orders", 0) or 0) > 0:
                positions_lines.append(_format_order_only_line(row))
                emitted += 1
                if emitted >= 10:
                    break
    else:
        positions_lines.append("(no active positions or open orders)")

    events_lines = [
        _format_event_line(message) for message in _filtered_recent_events(view_state, focus_symbol)
    ] or ["(no websocket events seen yet)"]
    ticks_lines = [
        _format_tick_line(
            symbol,
            message,
            snapshot.get("market", {}).get(symbol, {}) if isinstance(snapshot.get("market", {}), dict) else {},
        )
        for symbol, message in _filtered_price_ticks(view_state, focus_symbol)
    ] or ["(no price ticks seen yet)"]
    orders_lines = [
        _format_order_activity_line(entry)
        for entry in _recent_order_activity(snapshot, focus_symbol)
    ] or ["(no recent order activity in snapshot)"]
    logs_lines = list(view_state.recent_log_lines)[-8:] or ["(no log file attached)"]

    command_lines = [f"Mode: {'paused (panels frozen)' if state.paused else 'live'}"]
    for idx, line in enumerate(_wrap_text(state.command_status or "", max(20, width - 14))):
        prefix = "Status:" if idx == 0 else "       "
        command_lines.append(f"{prefix} {line}".rstrip())
    prompt_text = state.command_buffer or state.last_submitted_command
    command_lines.append(f"> {prompt_text}")

    output_lines = _wrap_box("Session", header_lines, width)
    if width >= 120:
        left_width = max(52, min(width - 38, int(width * 0.58)))
        right_width = max(34, width - left_width - 1)
        left_lines = (
            _wrap_box("Summary", summary_lines, left_width)
            + _wrap_box("Focus", _render_focus_panel(snapshot, focus_symbol), left_width)
            + _wrap_box("Positions", positions_lines, left_width)
        )
        log_title = "Bot Log" + (f" | {view_state.followed_log_file}" if view_state.followed_log_file else "")
        right_lines = (
            _wrap_box("Forager", _render_forager_panel(snapshot), right_width)
            + _wrap_box("Unstuck", _render_unstuck_panel(snapshot), right_width)
            + _wrap_box("Trailing", _render_trailing_panel(snapshot), right_width)
            + _wrap_box("Recent Events", events_lines, right_width)
            + _wrap_box("Price Ticks", ticks_lines, right_width)
            + _wrap_box("Recent Orders", orders_lines, right_width)
            + _wrap_box(log_title, logs_lines, right_width)
        )
        output_lines.extend(_combine_columns(left_lines, right_lines, left_width, right_width))
    else:
        output_lines.extend(_wrap_box("Summary", summary_lines, width))
        output_lines.extend(_wrap_box("Focus", _render_focus_panel(snapshot, focus_symbol), width))
        output_lines.extend(_wrap_box("Positions", positions_lines, width))
        output_lines.extend(_wrap_box("Forager", _render_forager_panel(snapshot), width))
        output_lines.extend(_wrap_box("Unstuck", _render_unstuck_panel(snapshot), width))
        output_lines.extend(_wrap_box("Trailing", _render_trailing_panel(snapshot), width))
        output_lines.extend(_wrap_box("Recent Events", events_lines, width))
        output_lines.extend(_wrap_box("Price Ticks", ticks_lines, width))
        output_lines.extend(_wrap_box("Recent Orders", orders_lines, width))
        log_title = "Bot Log" + (f" | {view_state.followed_log_file}" if view_state.followed_log_file else "")
        output_lines.extend(_wrap_box(log_title, logs_lines, width))
    output_lines.extend(_wrap_box("Command", command_lines, width))
    return "\n".join(_truncate(line, width) for line in output_lines)


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
        self._log_tail_state: Optional[_LogTailState] = None
        self._stop_event = asyncio.Event()
        self._last_painted_screen: Optional[str] = None
        if self.log_file:
            self._bootstrap_log_tail(Path(self.log_file))

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
                render_width = max(80, shutil.get_terminal_size((120, 40)).columns - 1)
                if self.state.paused:
                    if self.state.paused_render_data is None:
                        self.state.paused_render_data = _capture_render_data(self.state)
                    screen = render_screen(
                        self.state,
                        width=render_width,
                        display_data=self.state.paused_render_data,
                    )
                else:
                    self.state.paused_render_data = None
                    screen = render_screen(self.state, width=render_width)
                self.state.last_rendered_screen = screen
                painted_screen = _colorize_screen(screen)
                if painted_screen != self._last_painted_screen:
                    patch = _render_screen_diff(self._last_painted_screen, painted_screen)
                    if patch:
                        sys.stdout.write(patch)
                        sys.stdout.flush()
                    self._last_painted_screen = painted_screen
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.render_interval_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            if self._last_painted_screen is not None:
                line_count = len(self._last_painted_screen.splitlines())
                sys.stdout.write(f"\x1b[{line_count + 1};1H")
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
                    if not self.state.command_buffer and self.state.last_submitted_command:
                        self.state.last_submitted_command = ""
                    self.state.command_buffer = self.state.command_buffer[:-1]
                    continue
                if char == "\x03":
                    self.stop()
                    return
                if char == "\x1b":
                    continue
                if char.isprintable():
                    if not self.state.command_buffer and self.state.last_submitted_command:
                        self.state.last_submitted_command = ""
                    self.state.command_buffer += char
        finally:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, original_attrs)

    async def _log_tail_loop(self, path: Path) -> None:
        while not self._stop_event.is_set():
            self._poll_log_tail_once(path)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.log_poll_interval_ms / 1000.0
                )
            except asyncio.TimeoutError:
                pass

    def _bootstrap_log_tail(self, path: Path) -> None:
        self.state.set_log_file(str(path))
        self.state.push_log_lines(_read_last_lines(path, self.log_bootstrap_lines))
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._log_tail_state = None
            return
        self._log_tail_state = _LogTailState(int(stat.st_dev), int(stat.st_ino), int(stat.st_size))

    def _poll_log_tail_once(self, path: Optional[Path] = None) -> None:
        if path is None:
            if not self.log_file:
                return
            path = Path(self.log_file)
        if self.state.followed_log_file != str(path):
            self._bootstrap_log_tail(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._log_tail_state = None
            return
        file_id = (int(stat.st_dev), int(stat.st_ino))
        size = int(stat.st_size)
        if self._log_tail_state is None:
            self._log_tail_state = _LogTailState(file_id[0], file_id[1], size)
            return
        reset = size < self._log_tail_state.offset or (
            self._log_tail_state.dev,
            self._log_tail_state.ino,
        ) != file_id
        read_from = 0 if reset else self._log_tail_state.offset
        if size <= read_from:
            if reset:
                self._log_tail_state = _LogTailState(file_id[0], file_id[1], size)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(read_from)
                lines = f.read().splitlines()
                new_offset = int(f.tell())
        except FileNotFoundError:
            self._log_tail_state = None
            return
        if lines:
            self.state.push_log_lines(lines)
        self._log_tail_state = _LogTailState(file_id[0], file_id[1], new_offset)
