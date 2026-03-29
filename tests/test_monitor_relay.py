import asyncio
import json
import subprocess
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import monitor_relay
from monitor_relay import (
    RELAY_APP_KEY,
    _handle_dashboard,
    _handle_dashboard_asset,
    _handle_health,
    _handle_snapshot,
    _handle_ws,
    create_monitor_relay_app,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")


def _append_json_line(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _make_monitor_root(tmp_path, exchange="bybit", user="user01"):
    root = tmp_path / "monitor" / exchange / user
    _write_json(
        root / "manifest.json",
        {
            "exchange": exchange,
            "user": user,
            "last_seq": 5,
            "capabilities": {"history_streams": {"fills": True, "price_ticks": True}},
        },
    )
    _write_json(
        root / "state.latest.json",
        {
            "schema_version": 1,
            "meta": {
                "exchange": exchange,
                "user": user,
                "seq": 5,
                "snapshot_ts_ms": 123456,
            },
            "account": {"balance_raw": 1000.0},
        },
    )
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "history").mkdir(parents=True, exist_ok=True)
    (root / "events" / "current.ndjson").write_text("", encoding="utf-8")
    (root / "history" / "price_ticks.current.ndjson").write_text("", encoding="utf-8")
    return tmp_path / "monitor"


def _make_request(app, path):
    return make_mocked_request("GET", path, app=app)


class FakeWebSocket:
    def __init__(self, *, close_after_messages):
        self.close_after_messages = close_after_messages
        self.closed = False
        self.messages = []

    async def prepare(self, request):
        return self

    async def send_json(self, payload):
        self.messages.append(payload)
        if len(self.messages) >= self.close_after_messages:
            self.closed = True

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_monitor_relay_health_handler_reports_available_bots(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    response = await _handle_health(_make_request(app, "/health"))

    assert response.status == 200
    data = json.loads(response.text)
    assert data["status"] == "ok"
    assert data["bots"] == [{"exchange": "bybit", "user": "user01"}]
    assert data["subscribers"]["bybit/user01"] == 0


@pytest.mark.asyncio
async def test_monitor_relay_snapshot_handler_returns_snapshot_message(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    response = await _handle_snapshot(_make_request(app, "/snapshot"))

    assert response.status == 200
    data = json.loads(response.text)
    assert data["type"] == "snapshot"
    assert data["seq"] == 5
    assert data["payload"]["account"]["balance_raw"] == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_monitor_relay_snapshot_requires_exchange_and_user_when_multiple_roots(tmp_path):
    monitor_root = _make_monitor_root(tmp_path, exchange="bybit", user="user01")
    _make_monitor_root(tmp_path, exchange="bitget", user="user02")
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    with pytest.raises(web.HTTPBadRequest) as exc_info:
        await _handle_snapshot(_make_request(app, "/snapshot"))
    assert "multiple monitor roots available" in exc_info.value.text

    response = await _handle_snapshot(
        _make_request(app, "/snapshot?exchange=bybit&user=user01")
    )
    data = json.loads(response.text)
    assert response.status == 200
    assert data["exchange"] == "bybit"
    assert data["user"] == "user01"


@pytest.mark.asyncio
async def test_monitor_relay_dashboard_handler_serves_html(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    response = await _handle_dashboard(_make_request(app, "/dashboard"))

    assert response.status == 200
    assert response.content_type == "text/html"
    assert "Passivbot Monitor Dashboard" in response.text


@pytest.mark.asyncio
async def test_monitor_relay_dashboard_asset_handler_serves_static_assets(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    request = _make_request(app, "/dashboard/assets/dashboard.js")
    request._match_info = {"name": "dashboard.js"}
    response = await _handle_dashboard_asset(request)

    assert response.status == 200
    assert response.content_type == "application/javascript"
    assert "const state" in response.text


@pytest.mark.asyncio
async def test_monitor_relay_dashboard_asset_handler_rejects_unknown_assets(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    request = _make_request(app, "/dashboard/assets/nope.js")
    request._match_info = {"name": "nope.js"}

    with pytest.raises(web.HTTPNotFound):
        await _handle_dashboard_asset(request)


@pytest.mark.asyncio
async def test_monitor_relay_publishes_first_entries_for_new_current_files_after_start(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)
    relay = app[RELAY_APP_KEY]
    key = ("bybit", "user01")
    queue = relay.subscribe(key)

    await relay.start()
    try:
        _append_json_line(
            monitor_root / "bybit" / "user01" / "history" / "candles_1h.current.ndjson",
            {
                "ts": 2200,
                "kind": "candle.completed",
                "stream": "candles_1h",
                "exchange": "bybit",
                "user": "user01",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "payload": {"close": 100500.0},
            },
        )

        await relay.poll_once()

        message = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert message["type"] == "history"
        assert message["stream"] == "candles_1h"
        assert message["timeframe"] == "1h"
        assert message["payload"]["close"] == pytest.approx(100500.0)
    finally:
        relay.unsubscribe(key, queue)
        await relay.stop()


@pytest.mark.asyncio
async def test_monitor_relay_websocket_sends_snapshot_then_live_updates(tmp_path, monkeypatch):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)
    relay = app[RELAY_APP_KEY]
    root = monitor_root / "bybit" / "user01"
    fake_ws = FakeWebSocket(close_after_messages=3)

    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda heartbeat=30.0: fake_ws)

    await relay.start()
    try:
        task = asyncio.create_task(_handle_ws(_make_request(app, "/ws")))

        for _ in range(50):
            if len(fake_ws.messages) >= 1 and relay._subscribers.get(("bybit", "user01")):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("websocket handler did not subscribe in time")

        _append_json_line(
            root / "events" / "current.ndjson",
            {
                "ts": 2000,
                "seq": 6,
                "kind": "bot.ready",
                "tags": ["bot", "lifecycle"],
                "exchange": "bybit",
                "user": "user01",
                "payload": {"status": "ready"},
            },
        )
        _append_json_line(
            root / "history" / "price_ticks.current.ndjson",
            {
                "ts": 2100,
                "kind": "price_tick",
                "stream": "price_ticks",
                "exchange": "bybit",
                "user": "user01",
                "symbol": "BTC/USDT:USDT",
                "payload": {"last": 100000.0},
            },
        )

        await relay.poll_once()
        await asyncio.wait_for(task, timeout=1.0)

        assert fake_ws.messages[0]["type"] == "snapshot"
        assert fake_ws.messages[0]["seq"] == 5
        assert fake_ws.messages[1]["type"] == "event"
        assert fake_ws.messages[1]["seq"] == 6
        assert fake_ws.messages[1]["kind"] == "bot.ready"
        assert fake_ws.messages[2]["type"] == "history"
        assert fake_ws.messages[2]["stream"] == "price_ticks"
        assert fake_ws.messages[2]["symbol"] == "BTC/USDT:USDT"
        assert fake_ws.messages[2]["payload"]["last"] == pytest.approx(100000.0)
    finally:
        await relay.stop()


@pytest.mark.asyncio
async def test_monitor_relay_websocket_replays_recent_current_file_messages_on_connect(
    tmp_path, monkeypatch
):
    monitor_root = _make_monitor_root(tmp_path)
    root = monitor_root / "bybit" / "user01"
    _append_json_line(
        root / "events" / "current.ndjson",
        {
            "ts": 1500,
            "seq": 4,
            "kind": "account.balance",
            "tags": ["account"],
            "exchange": "bybit",
            "user": "user01",
            "payload": {"equity": 999.0},
        },
    )
    _append_json_line(
        root / "history" / "price_ticks.current.ndjson",
        {
            "ts": 1600,
            "kind": "price_tick",
            "stream": "price_ticks",
            "exchange": "bybit",
            "user": "user01",
            "symbol": "BTC/USDT:USDT",
            "payload": {"last": 100100.0},
        },
    )

    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)
    relay = app[RELAY_APP_KEY]
    fake_ws = FakeWebSocket(close_after_messages=3)
    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda heartbeat=30.0: fake_ws)

    await relay.start()
    try:
        await asyncio.wait_for(_handle_ws(_make_request(app, "/ws")), timeout=1.0)

        assert fake_ws.messages[0]["type"] == "snapshot"
        assert fake_ws.messages[1]["type"] == "event"
        assert fake_ws.messages[1]["kind"] == "account.balance"
        assert fake_ws.messages[2]["type"] == "history"
        assert fake_ws.messages[2]["stream"] == "price_ticks"
        assert fake_ws.messages[2]["payload"]["last"] == pytest.approx(100100.0)
    finally:
        await relay.stop()


def test_monitor_relay_tool_help_runs_without_import_errors():
    result = subprocess.run(
        [sys.executable, "src/tools/monitor_relay.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Serve read-only Passivbot monitor snapshots and live streams." in result.stdout
