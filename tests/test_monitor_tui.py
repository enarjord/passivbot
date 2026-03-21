from monitor_tui import MonitorTuiState, execute_tui_command, render_screen, resolve_focus_symbol_alias


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
                "meta": {
                    "exchange": "bitget",
                    "user": "bitget_01",
                    "seq": 42,
                    "snapshot_ts_ms": 1774057000000,
                },
                "account": {"balance_raw": 1000.0, "equity": 995.0},
                "health": {"uptime_ms": 90000, "fills": 3},
                "positions": {},
                "open_orders": {},
                "hsl": {},
                "forager": {},
                "unstuck": {},
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
    state.apply_message(
        {
            "type": "history",
            "ts": 1774057000300,
            "stream": "candles_1m",
            "kind": "candle.completed",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1m",
            "payload": {"c": 70100.0},
        }
    )

    assert state.exchange == "bitget"
    assert state.user == "bitget_01"
    assert state.snapshot_seq == 42
    assert state.recent_events[0]["kind"] == "bot.ready"
    assert state.recent_price_ticks["BTC/USDT:USDT"]["payload"]["last"] == 70000.0
    assert state.recent_candles["BTC/USDT:USDT|1m"]["payload"]["c"] == 70100.0


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
                "meta": {"exchange": "bitget", "user": "bitget_01", "seq": 50, "snapshot_ts_ms": 1774057000000},
                "account": {},
                "health": {},
                "positions": {
                    "BTC/USDT:USDT": {
                        "long": {
                            "size": 0.01,
                            "price": 70000.0,
                            "wallet_exposure": 0.7,
                            "wel_ratio": 3.5,
                            "wele_ratio": 2.8,
                            "twel_ratio": 0.41,
                            "price_action_distance": 0.0071,
                            "upnl": 5.0,
                        },
                        "short": {"size": 0.0, "price": 0.0},
                    },
                    "ETH/USDT:USDT": {
                        "long": {
                            "size": 0.5,
                            "price": 3500.0,
                            "wallet_exposure": 1.75,
                            "wel_ratio": 8.75,
                            "wele_ratio": 7.0,
                            "twel_ratio": 1.03,
                            "price_action_distance": 0.0143,
                            "upnl": 25.0,
                        },
                        "short": {"size": 0.0, "price": 0.0},
                    },
                },
                "open_orders": {"ETH/USDT:USDT": [{"side": "sell", "price": 3600.0, "qty": 0.5}]},
                "hsl": {},
                "forager": {"long": {"selected_symbols": ["ETH/USDT:USDT"]}, "short": {"selected_symbols": []}},
                "unstuck": {},
                "market": {
                    "BTC/USDT:USDT": {"last_price": 70500.0, "tradable": True},
                    "ETH/USDT:USDT": {
                        "last_price": 3550.0,
                        "tradable": True,
                        "active_symbol": True,
                        "approved": {"long": True, "short": False},
                        "ignored": {"long": False, "short": False},
                        "min_qty": 0.001,
                        "min_cost": 5.0,
                        "effective_min_cost": 5.1,
                    },
                },
                "recent": {
                    "order_executions": [
                        {
                            "execution_timestamp": 1774057000200,
                            "symbol": "ETH/USDT:USDT",
                            "position_side": "long",
                            "side": "buy",
                            "qty": 0.5,
                            "price": 3500.0,
                            "pb_order_type": "entry_grid_normal_long",
                        }
                    ],
                    "order_cancellations": [],
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
            "pside": "long",
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

    rendered = render_screen(state, width=160)

    assert "focus=ETH/USDT:USDT" in rendered
    assert "Focus | ETH/USDT:USDT" in rendered
    assert "long WE=1.7500" in rendered
    assert "Recent Orders" in rendered
    assert "executed | ETH/USDT:USDT" in rendered


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


def test_execute_tui_command_supports_focus_aliases_and_exit():
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765")
    state.snapshot = {
        "market": {
            "BTC/USDT:USDT": {},
            "ETH/USDT:USDT": {},
            "SOL/USDT:USDT": {},
        }
    }

    should_stop = execute_tui_command(state, "focus BTC")
    assert not should_stop
    assert state.focus_symbol == "BTC/USDT:USDT"

    should_stop = execute_tui_command(state, "focus next")
    assert not should_stop
    assert state.focus_symbol == "ETH/USDT:USDT"

    should_stop = execute_tui_command(state, "focus auto")
    assert not should_stop
    assert state.focus_symbol is None

    should_stop = execute_tui_command(state, "exit")
    assert should_stop


def test_monitor_tui_state_tracks_recent_log_lines():
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765")
    state.set_log_file("logs/example.log")
    state.push_log_lines(["line one", "", "line two"])

    assert state.followed_log_file == "logs/example.log"
    assert list(state.recent_log_lines) == ["line one", "line two"]


def test_monitor_tui_render_screen_includes_core_panels():
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765", exchange="bitget", user="bitget_01")
    state.ws_connected = True
    state.apply_message(
        {
            "type": "snapshot",
            "exchange": "bitget",
            "user": "bitget_01",
            "seq": 43,
            "ts": 1774057000000,
            "payload": {
                "meta": {
                    "exchange": "bitget",
                    "user": "bitget_01",
                    "seq": 43,
                    "snapshot_ts_ms": 1774057000000,
                },
                "account": {
                    "balance_raw": 1000.0,
                    "balance_snapped": 999.5,
                    "equity": 1001.25,
                    "realized_pnl_cumsum": {"current": 12.0},
                },
                "health": {
                    "uptime_ms": 65000,
                    "last_loop_duration_ms": 1234.0,
                    "fills": 5,
                    "orders_placed": 2,
                    "orders_cancelled": 1,
                    "errors_last_hour": 0,
                    "rate_limits": 0,
                },
                "positions": {
                    "BTC/USDT:USDT": {
                        "long": {
                            "size": 0.01,
                            "price": 70000.0,
                            "wallet_exposure": 0.7,
                            "wel_ratio": 3.5,
                            "wele_ratio": 2.8,
                            "twel_ratio": 0.41,
                            "price_action_distance": 0.0071,
                            "upnl": 5.0,
                        },
                        "short": {"size": 0.0, "price": 0.0},
                    }
                },
                "open_orders": {
                    "BTC/USDT:USDT": [{"side": "sell", "price": 71000.0, "qty": 0.01}]
                },
                "hsl": {
                    "long": {"tier": "green", "halted": False, "last_metrics": {"drawdown_score": 0.01}},
                    "short": {"tier": "green", "halted": False, "last_metrics": {"drawdown_score": 0.0}},
                },
                "forager": {
                    "long": {"slots": {"current": 4, "max": 10}, "selected_symbols": ["BTC/USDT:USDT"]},
                    "short": {"slots": {"current": 0, "max": 10}, "selected_symbols": []},
                },
                "unstuck": {
                    "has_open_order": False,
                    "sides": {
                        "long": {"status": "ok", "allowance_live": 0.0},
                        "short": {"status": "disabled", "allowance_live": 0.0},
                    },
                },
                "market": {"BTC/USDT:USDT": {"last_price": 70500.0}},
            },
        }
    )
    state.apply_message(
        {
            "type": "event",
            "ts": 1774057000500,
            "kind": "order.opened",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "BTC/USDT:USDT",
            "pside": "long",
            "payload": {"price": 71000.0, "qty": 0.01},
        }
    )
    state.apply_message(
        {
            "type": "history",
            "ts": 1774057000600,
            "stream": "price_ticks",
            "kind": "price_tick",
            "exchange": "bitget",
            "user": "bitget_01",
            "symbol": "BTC/USDT:USDT",
            "payload": {"last": 70501.0},
        }
    )
    state.set_log_file("logs/example.log")
    state.push_log_lines(["2026-03-21T12:00:00 INFO [bitget] READY"])

    rendered = render_screen(state, width=140)

    assert "Passivbot Monitor TUI | connected | bitget / bitget_01" in rendered
    assert "Account | raw=1000.00 snapped=999.50 equity=1001.25 realized=12.00" in rendered
    assert "Active Positions / Orders" in rendered
    assert "BTC/USDT" in rendered
    assert "WE= 0.7000" in rendered
    assert "PA=  0.0071" in rendered
    assert "Recent Events" in rendered
    assert "order.opened" in rendered
    assert "Recent Price Ticks" in rendered
    assert "last=  70501.0000" in rendered
    assert "Recent Bot Log | logs/example.log" in rendered
    assert "READY" in rendered


def test_monitor_tui_renders_account_realized_from_nested_payload():
    state = MonitorTuiState(relay_url="http://127.0.0.1:8765", exchange="bitget", user="bitget_01")
    state.apply_message(
        {
            "type": "snapshot",
            "exchange": "bitget",
            "user": "bitget_01",
            "seq": 44,
            "ts": 1774057000000,
            "payload": {
                "meta": {"exchange": "bitget", "user": "bitget_01", "pid": 123},
                "account": {
                    "balance_raw": 1000.0,
                    "balance_snapped": 999.5,
                    "equity": 1001.25,
                    "realized_pnl_cumsum": {"current": -46.62},
                },
                "health": {},
                "positions": {},
                "open_orders": {},
                "hsl": {},
                "forager": {},
                "unstuck": {},
            },
        }
    )

    rendered = render_screen(state, width=140)

    assert "realized=-46.62" in rendered
