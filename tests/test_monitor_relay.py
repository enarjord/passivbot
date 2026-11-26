import asyncio
import json
import os
import subprocess
import sys

import pytest
from aiohttp import ClientWSTimeout, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

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


def _make_monitor_root(tmp_path, exchange="bybit", user="user01", snapshot_ts_ms=123456):
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
                "snapshot_ts_ms": snapshot_ts_ms,
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
        self.close_event = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.close_event.wait()
        raise StopAsyncIteration

    async def prepare(self, request):
        return self

    async def send_json(self, payload):
        self.messages.append(payload)
        if len(self.messages) >= self.close_after_messages:
            self.closed = True

    async def close(self):
        self.closed = True
        self.close_event.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [
    "https://attacker.example", "http://localhost:9999", "null", "",
    "http://localhost:8080@attacker.example", "http://localhost:8080/path",
    "http://localhost:8080?query", "http://localhost:8080#fragment",
    "http://localhost:bad", "http://user@localhost:8080", "https://localhost:8080",
])
async def test_websocket_rejects_foreign_or_invalid_origin_before_reading_data(tmp_path, origin):
    app = create_monitor_relay_app(monitor_root=str(tmp_path / "missing"))
    request = make_mocked_request("GET", "/ws", app=app,
                                  headers={"Host": "localhost:8080", "Origin": origin})
    with pytest.raises(web.HTTPForbidden):
        await _handle_ws(request)


@pytest.mark.asyncio
async def test_websocket_rejects_multiple_origin_headers(tmp_path):
    app = create_monitor_relay_app(monitor_root=str(tmp_path))
    request = make_mocked_request("GET", "/ws", app=app, headers=[
        ("Host", "localhost:8080"), ("Origin", "http://localhost:8080"),
        ("Origin", "https://attacker.example")])
    with pytest.raises(web.HTTPForbidden):
        await _handle_ws(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("host,origin", [
    ("localhost:8080", "http://localhost:8080"), ("localhost", "http://localhost:80"),
    ("[::1]:8080", "http://[::1]:8080"), ("localhost:8080", None),
])
async def test_websocket_allows_same_origin_dashboard_and_originless_clients(
    tmp_path, monkeypatch, host, origin
):
    _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(tmp_path / "monitor"),
                                   allowed_origins=[origin or f"http://{host}"])
    fake_ws = FakeWebSocket(close_after_messages=1)
    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda **_: fake_ws)
    headers = {"Host": host}
    if origin is not None:
        headers["Origin"] = origin
    response = await _handle_ws(make_mocked_request("GET", "/ws", app=app, headers=headers))
    assert response is fake_ws
    assert fake_ws.messages[0]["type"] == "snapshot"


@pytest.mark.asyncio
async def test_dns_rebinding_host_and_origin_do_not_authorize_monitor_data(tmp_path):
    app = create_monitor_relay_app(monitor_root=str(tmp_path))
    request = make_mocked_request("GET", "/ws", app=app, headers={
        "Host": "attacker.example:8765", "Origin": "http://attacker.example:8765"})
    with pytest.raises(web.HTTPForbidden):
        await _handle_ws(request)

    async def must_not_read_data(request):
        raise AssertionError("untrusted Host reached an HTTP data handler")

    with pytest.raises(web.HTTPForbidden):
        await monitor_relay._validate_monitor_host(request, must_not_read_data)


@pytest.mark.asyncio
async def test_configured_external_https_origin_works_through_http_proxy(tmp_path, monkeypatch):
    _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(tmp_path / "monitor"),
                                   allowed_origins=["https://monitor.example"])
    fake_ws = FakeWebSocket(close_after_messages=1)
    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda **_: fake_ws)
    request = make_mocked_request("GET", "/ws", app=app, headers={
        "Host": "monitor.example", "Origin": "https://monitor.example"})
    assert request.scheme == "http"
    response = await monitor_relay._validate_monitor_host(request, _handle_ws)
    assert response is fake_ws
    assert fake_ws.messages[0]["type"] == "snapshot"


@pytest.mark.parametrize("origin", ["*", "null", "http://host/path", "http://user@host", "http://host:bad"])
def test_invalid_configured_origin_fails_startup(tmp_path, origin):
    with pytest.raises(ValueError):
        create_monitor_relay_app(monitor_root=str(tmp_path), allowed_origins=[origin])


@pytest.mark.asyncio
async def test_monitor_relay_health_handler_reports_available_bots(tmp_path):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    response = await _handle_health(_make_request(app, "/health"))

    assert response.status == 200
    data = json.loads(response.text)
    assert data["status"] == "ok"
    assert data["bots"] == [
        {
            "exchange": "bybit",
            "user": "user01",
            "active": True,
            "status": "active",
            "last_activity_ts_ms": data["bots"][0]["last_activity_ts_ms"],
        }
    ]
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
async def test_monitor_relay_snapshot_returns_bundle_when_multiple_roots(tmp_path):
    monitor_root = _make_monitor_root(tmp_path, exchange="bybit", user="user01")
    _make_monitor_root(tmp_path, exchange="bitget", user="user02")
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)

    response = await _handle_snapshot(_make_request(app, "/snapshot"))
    data = json.loads(response.text)
    assert response.status == 200
    assert data["type"] == "snapshot_bundle"
    assert data["active_count"] == 2
    assert data["stale_count"] == 0
    assert len(data["bots"]) == 2
    assert {f"{entry['exchange']}/{entry['user']}" for entry in data["bots"]} == {
        "bybit/user01",
        "bitget/user02",
    }

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
@pytest.mark.parametrize("queue_size", [1, 4])
async def test_monitor_relay_mixed_burst_yields_to_ready_subscriber(tmp_path, queue_size):
    keys = [("bitget", "user01"), ("bybit", "user02")]
    monitor_root = _make_monitor_root(tmp_path, *keys[0])
    _make_monitor_root(tmp_path, *keys[1])
    relay = create_monitor_relay_app(
        monitor_root=str(monitor_root), subscriber_queue_size=queue_size
    )[RELAY_APP_KEY]
    relay._prime_offsets()
    relay._initial_prime_completed = True
    fast = relay.subscribe()
    stalled = relay.subscribe()
    expected = []
    for exchange, user in keys:
        bot = monitor_root / exchange / user
        for kind, path in [
            ("event", bot / "events/current.ndjson"),
            ("history", bot / "history/price_ticks.current.ndjson"),
        ]:
            for number in range(7):
                _append_json_line(
                    path,
                    {
                        "exchange": exchange,
                        "user": user,
                        "seq": number,
                        "ts": number,
                        "payload": {"number": number},
                    },
                )
                expected.append((kind, exchange, user, number))

    received = []

    async def consume():
        while True:
            received.append(await fast.get())

    consumer = asyncio.create_task(consume())
    try:
        await relay.poll_once()
        await asyncio.sleep(0)
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        relay.unsubscribe(fast)
        relay.unsubscribe(stalled)

    assert [row["type"] for row in received].count("resync_required") == 0
    assert [
        (row["type"], row["exchange"], row["user"], row["payload"]["number"])
        for row in received
    ] == expected
    assert stalled.get_nowait() == {
        "type": "resync_required",
        "reason": "subscriber_queue_overflow",
    }


@pytest.mark.parametrize("stream", ["events", "history"])
@pytest.mark.parametrize("split_at", ["json", "utf8", "newline"])
def test_monitor_relay_retries_unfinished_current_rows(tmp_path, stream, split_at, caplog):
    monitor_root = _make_monitor_root(tmp_path)
    relay = create_monitor_relay_app(monitor_root=str(monitor_root))[RELAY_APP_KEY]
    relay._initial_prime_completed = True
    filename = "current.ndjson" if stream == "events" else "price_ticks.current.ndjson"
    path = monitor_root / "bybit" / "user01" / stream / filename

    def row(number):
        payload = {
            "seq": number,
            "ts": number * 1000,
            "kind": "test.event",
            "stream": "price_ticks",
            "exchange": "bybit",
            "user": "user01",
            "payload": {"number": number, "label": "€"},
        }
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    first, second, third = row(1), row(2), row(3)
    path.write_bytes(first)
    assert [message["payload"]["number"] for message in relay._read_updates(path)] == [1]
    split = {
        "json": len(second) // 2,
        "utf8": second.index("€".encode("utf-8")) + 1,
        "newline": len(second) - 1,
    }[split_at]
    with path.open("ab") as file:
        file.write(second[:split])

    initial = relay._read_updates(path)
    expected = [2] if split_at == "newline" else []
    assert [message["payload"]["number"] for message in initial] == expected
    expected_offset = path.stat().st_size if split_at == "newline" else len(first)
    assert relay._path_states[path].offset == expected_offset
    replay = relay._read_recent_entries(path, 2)
    expected = [1, 2] if split_at == "newline" else [1]
    assert [entry["payload"]["number"] for entry in replay] == expected
    assert not caplog.records

    with path.open("ab") as file:
        file.write(second[split:] + third)
    expected = [3] if split_at == "newline" else [2, 3]
    assert [message["payload"]["number"] for message in relay._read_updates(path)] == expected
    assert relay._read_updates(path) == []
    assert relay._path_states[path].offset == path.stat().st_size
    replay = relay._read_recent_entries(path, 2)
    assert [entry["payload"]["number"] for entry in replay] == [2, 3]


def test_monitor_relay_skips_malformed_terminated_rows(tmp_path, caplog):
    monitor_root = _make_monitor_root(tmp_path)
    relay = create_monitor_relay_app(monitor_root=str(monitor_root))[RELAY_APP_KEY]
    relay._initial_prime_completed = True
    path = monitor_root / "bybit" / "user01" / "events" / "current.ndjson"
    path.write_bytes(b'{"seq":1}\ninvalid\n\xff\n{"seq":2}\n')

    assert [message["seq"] for message in relay._read_updates(path)] == [1, 2]
    assert relay._read_updates(path) == []
    assert relay._path_states[path].offset == path.stat().st_size
    assert [entry["seq"] for entry in relay._read_recent_entries(path, 4)] == [1, 2]
    assert len(caplog.records) == 4


@pytest.mark.parametrize("replacement", ["rotate", "truncate"])
def test_monitor_relay_resets_unfinished_row_after_file_replacement(tmp_path, replacement):
    monitor_root = _make_monitor_root(tmp_path)
    relay = create_monitor_relay_app(monitor_root=str(monitor_root))[RELAY_APP_KEY]
    relay._initial_prime_completed = True
    path = monitor_root / "bybit" / "user01" / "events" / "current.ndjson"
    path.write_bytes(b'{"seq":1,"payload":{"label":"a longer first row"}}\n{"seq":')
    assert [message["seq"] for message in relay._read_updates(path)] == [1]

    if replacement == "rotate":
        path.rename(path.with_name("rotated.ndjson"))
    path.write_bytes(b'{"seq":2}\n')
    assert [message["seq"] for message in relay._read_updates(path)] == [2]
    assert relay._read_updates(path) == []


@pytest.mark.parametrize("stream", ["events", "history"])
@pytest.mark.parametrize("complete", [False, True])
def test_monitor_relay_primes_at_start_of_unfinished_row(tmp_path, stream, complete, caplog):
    monitor_root = _make_monitor_root(tmp_path)
    relay = create_monitor_relay_app(monitor_root=str(monitor_root))[RELAY_APP_KEY]
    filename = "current.ndjson" if stream == "events" else "price_ticks.current.ndjson"
    path = monitor_root / "bybit" / "user01" / stream / filename
    first = b'{"ts":1}\n'
    second = b'{"ts":2,"payload":{"label":"' + b"a" * (80 * 1024) + b'"}}'
    split = len(second) if complete else len(second) - 3
    path.write_bytes(first + second[:split])

    relay._prime_offsets()
    assert relay._path_states[path].offset == (len(first) + len(second) if complete else len(first))
    assert relay._read_updates(path) == []

    with path.open("ab") as file:
        file.write(second[split:] + b'\n{"ts":3}\n')

    expected = [3] if complete else [2, 3]
    assert [message["ts"] for message in relay._read_updates(path)] == expected
    assert relay._read_updates(path) == []
    assert not caplog.records


def test_monitor_relay_reads_rotation_from_new_file_start(tmp_path, monkeypatch, caplog):
    monitor_root = _make_monitor_root(tmp_path)
    relay = create_monitor_relay_app(monitor_root=str(monitor_root))[RELAY_APP_KEY]
    relay._initial_prime_completed = True
    path = monitor_root / "bybit" / "user01" / "events" / "current.ndjson"
    path.write_bytes(b'{"seq":1}\n')
    assert [message["seq"] for message in relay._read_updates(path)] == [1]

    # Make the prior inode appear to have pending rows. Rotate it at the
    # same point the relay opens the stream after checking for updates.
    with path.open("ab") as file:
        file.write(b'{"seq":2}\n')
    replacement = path.with_name("replacement")
    replacement.write_bytes(b'{"seq":3,"payload":{"label":"new inode"}}\n{"seq":4}\n')
    original_open = open

    def open_after_rotation(candidate, *args, **kwargs):
        if candidate == path and replacement.exists():
            path.rename(path.with_name("rotated.ndjson"))
            replacement.rename(path)
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr("builtins.open", open_after_rotation)

    assert [message["seq"] for message in relay._read_updates(path)] == [3, 4]
    assert relay._read_updates(path) == []
    assert not caplog.records


@pytest.mark.asyncio
async def test_monitor_relay_responds_to_ping_and_releases_idle_subscription(tmp_path, monkeypatch):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root))
    relay = app[RELAY_APP_KEY]
    unsubscribe = relay.unsubscribe
    unsubscribed = asyncio.Event()

    def notify_unsubscribed(queue):
        unsubscribe(queue)
        unsubscribed.set()

    monkeypatch.setattr(relay, "unsubscribe", notify_unsubscribed)
    async with TestClient(TestServer(app, shutdown_timeout=0.5)) as client:
        ws = await client.ws_connect(
            "/ws",
            headers={"Host": "localhost:8765"},
            autoping=False,
            timeout=ClientWSTimeout(ws_close=0.5),
        )
        assert (await ws.receive_json(timeout=1.0))["type"] == "snapshot"
        await ws.ping(b"probe")
        pong = await ws.receive(timeout=1.0)
        assert pong.type is WSMsgType.PONG
        assert pong.data == b"probe"
        assert len(relay._subscribers) == 1

        await ws.close()
        await asyncio.wait_for(unsubscribed.wait(), timeout=1.0)
        assert not relay._subscribers


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
            if len(fake_ws.messages) >= 1 and any(
                subscription == ("bybit", "user01") for subscription in relay._subscribers.values()
            ):
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
async def test_monitor_relay_websocket_without_filter_streams_all_active_bots(tmp_path, monkeypatch):
    monitor_root = _make_monitor_root(tmp_path, exchange="bybit", user="user01")
    _make_monitor_root(tmp_path, exchange="bitget", user="user02")
    app = create_monitor_relay_app(monitor_root=str(monitor_root), poll_interval_ms=10)
    relay = app[RELAY_APP_KEY]
    root = monitor_root
    fake_ws = FakeWebSocket(close_after_messages=4)

    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda heartbeat=30.0: fake_ws)

    await relay.start()
    try:
        task = asyncio.create_task(_handle_ws(_make_request(app, "/ws")))

        for _ in range(50):
            if len(fake_ws.messages) >= 2 and relay._subscribers:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("websocket handler did not subscribe in time")

        _append_json_line(
            root / "bybit" / "user01" / "events" / "current.ndjson",
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
            root / "bitget" / "user02" / "events" / "current.ndjson",
            {
                "ts": 2100,
                "seq": 7,
                "kind": "bot.ready",
                "tags": ["bot", "lifecycle"],
                "exchange": "bitget",
                "user": "user02",
                "payload": {"status": "ready"},
            },
        )

        await relay.poll_once()
        await asyncio.wait_for(task, timeout=1.0)

        assert fake_ws.messages[0]["type"] == "snapshot"
        assert fake_ws.messages[1]["type"] == "snapshot"
        assert fake_ws.messages[2]["type"] == "event"
        assert fake_ws.messages[3]["type"] == "event"
        assert {f"{msg['exchange']}/{msg['user']}" for msg in fake_ws.messages[:4]} == {
            "bybit/user01",
            "bitget/user02",
        }
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


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("queue_size", [1, 1000])
async def test_monitor_relay_keeps_live_events_published_during_replay(
    tmp_path, monkeypatch, filtered, queue_size
):
    monitor_root = _make_monitor_root(tmp_path)
    _make_monitor_root(tmp_path, exchange="bitget", user="user02")
    path = monitor_root / "bybit" / "user01" / "events" / "current.ndjson"
    event = {
        "ts": 2000,
        "seq": 6,
        "kind": "bot.ready",
        "exchange": "bybit",
        "user": "user01",
        "payload": {},
    }
    _append_json_line(path, event)
    app = create_monitor_relay_app(
        monitor_root=str(monitor_root), subscriber_queue_size=queue_size
    )
    relay = app[RELAY_APP_KEY]
    snapshots = 1 if filtered else 2
    fake_ws = FakeWebSocket(close_after_messages=snapshots + 2)
    original_send = fake_ws.send_json

    async def send_json(message):
        await original_send(message)
        if message.get("type") == "event" and message.get("seq") == 6:
            last = 8 if queue_size == 1 else 7
            for seq in range(7, last + 1):
                _append_json_line(path, {**event, "seq": seq, "ts": seq * 1000})
                await relay.poll_once()

    fake_ws.send_json = send_json
    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda **_: fake_ws)
    await relay.start()
    try:
        url = "/ws?exchange=bybit&user=user01" if filtered else "/ws"
        await asyncio.wait_for(_handle_ws(_make_request(app, url)), timeout=1.0)
        assert [message["type"] for message in fake_ws.messages[:snapshots]] == ["snapshot"] * snapshots
        assert fake_ws.messages[snapshots]["seq"] == 6
        if queue_size == 1:
            assert fake_ws.messages[-1] == {
                "type": "resync_required", "reason": "subscriber_queue_overflow"
            }
        else:
            assert fake_ws.messages[-1]["seq"] == 7
        assert not relay._subscribers
    finally:
        await relay.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionResetError, asyncio.CancelledError])
async def test_monitor_relay_unsubscribes_after_initial_send_failure(tmp_path, monkeypatch, failure):
    monitor_root = _make_monitor_root(tmp_path)
    app = create_monitor_relay_app(monitor_root=str(monitor_root))
    relay = app[RELAY_APP_KEY]
    fake_ws = FakeWebSocket(close_after_messages=1)

    async def send_json(message):
        raise failure()

    fake_ws.send_json = send_json
    monkeypatch.setattr(monitor_relay.web, "WebSocketResponse", lambda **_: fake_ws)
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await _handle_ws(_make_request(app, "/ws"))
    else:
        assert await _handle_ws(_make_request(app, "/ws")) is fake_ws
    assert not relay._subscribers
    assert fake_ws.closed


def test_monitor_relay_tool_help_runs_without_import_errors():
    result = subprocess.run(
        [sys.executable, "src/tools/monitor_relay.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Serve read-only Passivbot monitor snapshots and live streams." in result.stdout


def test_monitor_relay_keeps_stale_bots_visible_before_pruning(tmp_path, monkeypatch):
    monitor_root = _make_monitor_root(tmp_path, snapshot_ts_ms=1000)
    relay = monitor_relay.MonitorRelay(monitor_root=str(monitor_root), poll_interval_ms=10)
    key = ("bybit", "user01")

    monkeypatch.setattr(relay, "_key_last_activity_ts_ms", lambda current_key: 1000)
    monkeypatch.setattr(relay, "_key_stale_after_ms", lambda current_key: 1000)
    monkeypatch.setattr(relay, "_key_prune_after_ms", lambda current_key: 5000)
    monkeypatch.setattr(monitor_relay.time, "time", lambda: 1.0)

    assert relay.active_keys() == [key]
    assert relay.visible_keys() == [key]

    monkeypatch.setattr(monitor_relay.time, "time", lambda: 2.5)

    assert relay.active_keys() == []
    assert relay.visible_keys() == [key]
    health = relay.build_health_payload()
    assert health["bots"][0]["status"] == "stale"
    assert health["bots"][0]["active"] is False

    snapshot_messages = relay.load_snapshot_messages(exchange=None, user=None)
    assert len(snapshot_messages) == 1
    assert snapshot_messages[0]["relay"]["status"] == "stale"

    monkeypatch.setattr(monitor_relay.time, "time", lambda: 7.5)

    assert relay.visible_keys() == []


def test_monitor_relay_uses_current_event_activity_when_manifest_is_stale(tmp_path, monkeypatch):
    monitor_root = _make_monitor_root(tmp_path, snapshot_ts_ms=1000)
    bot_root = monitor_root / "bybit" / "user01"
    manifest_path = bot_root / "manifest.json"
    snapshot_path = bot_root / "state.latest.json"
    events_path = bot_root / "events" / "current.ndjson"

    os.utime(manifest_path, (1.0, 1.0))
    os.utime(snapshot_path, (1.0, 1.0))
    os.utime(events_path, (199.0, 199.0))
    monkeypatch.setattr(monitor_relay.time, "time", lambda: 200.0)

    relay = monitor_relay.MonitorRelay(monitor_root=str(monitor_root), poll_interval_ms=10)

    assert relay.active_keys() == [("bybit", "user01")]
    assert relay.build_health_payload()["bots"][0]["status"] == "active"
