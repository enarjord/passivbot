from pathlib import Path
import subprocess
import sys

from monitor_tui import (
    MonitorTuiState,
    _render_screen_diff,
    execute_tui_command,
    render_screen,
    resolve_focus_symbol_alias,
)


def test_monitor_tui_state_tracks_snapshot_events_and_history():
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765")

    state.apply_message(
        {
            "type": "snapshot",
            "exchange": "bitget",
            "user": "bitget_01",
            "seq": 42,
            "ts": 1774057000000,
            "payload": {
                "meta": {"exchange": "bitget", "user": "bitget_01"},
                "account": {"balance_raw": 1000.0, "equity": 995.0},
                "health": {"uptime_ms": 90000, "fills": 3},
                "positions": {},
                "open_orders": {},
                "hsl": {"tier": "green"},
                "market": {},
            },
        }
    )
    state.apply_message(
        {
            "type": "event",
            "ts": 1774057000100,
            "kind": "bot.ready",
            "exchange": "bitget",
            "user": "bitget_01",
            "payload": {"status": "ready"},
        }
    )
    state.apply_message(
        {
            "type": "history",
            "ts": 1774057000200,
            "stream": "price_ticks",
            "kind": "price_tick",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "BTC/USDT:USDT",
            "payload": {"last": 70000.0},
        }
    )

    assert state.exchange == "bitget"
    assert state.user == "bitget_01"
    assert state.snapshot_seq == 42
    assert state.recent_events[0]["kind"] == "bot.ready"
    assert state.recent_price_ticks["BTC/USDT:USDT"]["payload"]["last"] == 70000.0


def test_monitor_tui_prefers_focus_symbol_in_rendering():
    state = MonitorTuiState(
        relay_url="http://127.0.0.1:8765",
        exchange="bitget",
        user="bitget_01",
        focus_symbol="ETH/USDT:USDT",
    )
    state.apply_message(
        {
            "type": "snapshot",
            "exchange": "bitget",
            "user": "bitget_01",
            "seq": 50,
            "ts": 1774057000000,
            "payload": {
                "meta": {"exchange": "bitget", "user": "bitget_01"},
                "account": {
                    "balance_raw": 1000.0,
                    "balance_snapped": 999.5,
                    "equity": 1001.25,
                },
                "health": {
                    "uptime_ms": 65000,
                    "fills": 5,
                    "orders_placed": 2,
                    "orders_cancelled": 1,
                    "errors_last_hour": 0,
                },
                "positions": {
                    "BTC/USDT:USDT": {
                        "long": {"size": 0.01, "price": 70000.0},
                        "short": {"size": 0.0, "price": 0.0},
                    },
                    "ETH/USDT:USDT": {
                        "long": {"size": 0.5, "price": 3500.0},
                        "short": {"size": 0.0, "price": 0.0},
                    },
                },
                "open_orders": {"ETH/USDT:USDT": [{"side": "sell", "price": 3600.0, "qty": 0.5}]},
                "hsl": {"tier": "green"},
                "market": {
                    "BTC/USDT:USDT": {"last_price": 70500.0, "tradable": True},
                    "ETH/USDT:USDT": {"last_price": 3550.0, "tradable": True},
                },
            },
        }
    )
    state.apply_message(
        {
            "type": "event",
            "ts": 1774057000100,
            "kind": "order.opened",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "ETH/USDT:USDT",
            "payload": {"price": 3600.0, "qty": 0.5},
        }
    )
    state.apply_message(
        {
            "type": "history",
            "ts": 1774057000150,
            "stream": "price_ticks",
            "kind": "price_tick",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "ETH/USDT:USDT",
            "payload": {"last": 3550.0},
        }
    )
    state.set_log_file("logs/example.log")
    state.push_log_lines(["2026-03-21T12:00:00 INFO READY"])

    rendered = render_screen(state, width=140)

    assert "Passivbot Monitor TUI | LIVE | disconnected | bitget / bitget_01 | focus=ETH/USDT:USDT" in rendered
    assert "| Summary" in rendered
    assert "Account raw=1000.00 snapped=999.50 equity=1001.25" in rendered
    assert "| Focus" in rendered
    assert "symbol=ETH/USDT:USDT" in rendered
    assert "| Positions" in rendered
    assert "ETH/USDT" in rendered
    assert "| Recent Events" in rendered
    assert "order.opened" in rendered
    assert "| Price Ticks" in rendered
    assert "ETH/USDT" in rendered
    assert "Bot Log | logs/example.log" in rendered
    assert "READY" in rendered


def test_resolve_focus_symbol_alias_accepts_coin_aliases_and_detects_ambiguity():
    snapshot = {
        "market": {
            "BTC/USDT:USDT": {},
            "ETH/USDT:USDT": {},
        }
    }
    resolved, error = resolve_focus_symbol_alias("BTC", snapshot)
    assert resolved == "BTC/USDT:USDT"
    assert error is None

    snapshot_multi = {
        "market": {
            "BTC/USDT:USDT": {},
            "BTC/USDC:USDC": {},
        }
    }
    resolved, error = resolve_focus_symbol_alias("BTC", snapshot_multi)
    assert resolved is None
    assert "ambiguous symbol" in error

    resolved, error = resolve_focus_symbol_alias("BTCUSDT", snapshot_multi)
    assert resolved == "BTC/USDT:USDT"
    assert error is None


def test_execute_tui_command_supports_focus_aliases_pause_resume_dump_and_exit(tmp_path: Path):
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765")
    state.snapshot = {
        "market": {
            "BTC/USDT:USDT": {},
            "ETH/USDT:USDT": {},
            "SOL/USDT:USDT": {},
        }
    }
    state.last_rendered_screen = "screen line one\nscreen line two"

    should_stop = execute_tui_command(state, "focus BTC")
    assert not should_stop
    assert state.focus_symbol == "BTC/USDT:USDT"

    should_stop = execute_tui_command(state, "focus next")
    assert not should_stop
    assert state.focus_symbol == "ETH/USDT:USDT"

    should_stop = execute_tui_command(state, "pause")
    assert not should_stop
    assert state.paused is True
    assert "Paused." in state.command_status

    should_stop = execute_tui_command(state, "dump", dump_dir=tmp_path)
    assert not should_stop
    dumped = sorted(tmp_path.glob("monitor_tui_dump_*.txt"))
    assert len(dumped) == 1
    assert dumped[0].read_text(encoding="utf-8") == "screen line one\nscreen line two\n"

    should_stop = execute_tui_command(state, "resume")
    assert not should_stop
    assert state.paused is False
    assert state.paused_render_data is None

    should_stop = execute_tui_command(state, "focus auto")
    assert not should_stop
    assert state.focus_symbol is None

    should_stop = execute_tui_command(state, "exit")
    assert should_stop


def test_monitor_tui_recent_events_collapses_balance_spam_and_help_status():
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765", exchange="bitget", user="bitget_01")
    state.apply_message(
        {
            "type": "snapshot",
            "exchange": "bitget",
            "user": "bitget_01",
            "seq": 1,
            "ts": 1774057000000,
            "payload": {
                "meta": {"exchange": "bitget", "user": "bitget_01"},
                "account": {},
                "health": {},
                "positions": {},
                "open_orders": {},
                "hsl": {},
                "market": {},
            },
        }
    )
    for ts in (1774057000100, 1774057000200, 1774057000300):
        state.apply_message(
            {
                "type": "event",
                "ts": ts,
                "kind": "account.balance",
                "exchange": "bitget",
                "user": "bitget_01",
                "payload": {"equity": 1000.0 + ts / 1000.0},
            }
        )
    state.apply_message(
        {
            "type": "event",
            "ts": 1774057000400,
            "kind": "order.opened",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "BTC/USDT:USDT",
            "payload": {"price": 70000.0, "qty": 0.01},
        }
    )

    execute_tui_command(state, "help")
    rendered = render_screen(state, width=120)

    assert rendered.count("account.balance") == 1
    assert "order.opened" in rendered
    assert "Status: Help:" in rendered
    assert "focus <coin|symbol> | focus auto|next|prev | pause | resume | dump | clear | help | quit" in rendered
    assert "> help" in rendered


def test_render_screen_diff_updates_only_changed_rows():
    previous = "one\ntwo\nthree"
    current = "one\nTWO\nthree"

    diff = _render_screen_diff(previous, current)

    assert "\x1b[2;1H" in diff
    assert "TWO" in diff


def test_monitor_tui_tool_help_runs_without_import_errors():
    result = subprocess.run(
        [sys.executable, "src/tools/monitor_tui.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Minimal terminal dashboard for the Passivbot monitor relay." in result.stdout
