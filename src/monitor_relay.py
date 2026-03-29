from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aiohttp import web


MonitorKey = tuple[str, str]
DASHBOARD_STATIC_DIR = Path(__file__).resolve().parent / "monitor_dashboard_static"
_DASHBOARD_ASSETS = {
    "dashboard.css": ("dashboard.css", "text/css"),
    "dashboard.js": ("dashboard.js", "application/javascript"),
}


@dataclass
class _PathState:
    dev: int
    ino: int
    offset: int


class MonitorRelay:
    def __init__(
        self,
        *,
        monitor_root: str,
        poll_interval_ms: int = 250,
        subscriber_queue_size: int = 1000,
        ws_replay_limit: int = 50,
    ) -> None:
        self.monitor_root = Path(monitor_root).expanduser()
        self.poll_interval_ms = max(50, int(poll_interval_ms))
        self.subscriber_queue_size = max(1, int(subscriber_queue_size))
        self.ws_replay_limit = max(0, int(ws_replay_limit))
        self.started_at_monotonic = time.monotonic()
        self._path_states: dict[Path, _PathState] = {}
        self._subscribers: dict[MonitorKey, set[asyncio.Queue]] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._initial_prime_completed = False

    def discover_keys(self) -> list[MonitorKey]:
        if not self.monitor_root.exists():
            return []
        keys: list[MonitorKey] = []
        for exchange_dir in sorted(self.monitor_root.iterdir()):
            if not exchange_dir.is_dir():
                continue
            for user_dir in sorted(exchange_dir.iterdir()):
                if not user_dir.is_dir():
                    continue
                if (user_dir / "manifest.json").exists():
                    keys.append((exchange_dir.name, user_dir.name))
        return keys

    def _bot_root(self, key: MonitorKey) -> Path:
        exchange, user = key
        return self.monitor_root / exchange / user

    def _events_current_path(self, key: MonitorKey) -> Path:
        return self._bot_root(key) / "events" / "current.ndjson"

    def _history_current_paths(self, key: MonitorKey) -> list[Path]:
        history_dir = self._bot_root(key) / "history"
        if not history_dir.exists():
            return []
        return sorted(history_dir.glob("*.current.ndjson"))

    def _current_paths_for_key(self, key: MonitorKey) -> list[Path]:
        return [self._events_current_path(key), *self._history_current_paths(key)]

    def resolve_key(
        self,
        *,
        exchange: Optional[str],
        user: Optional[str],
    ) -> MonitorKey:
        keys = self.discover_keys()
        if exchange and user:
            key = (str(exchange), str(user))
            if key not in keys:
                raise FileNotFoundError(f"monitor root not found for {exchange}/{user}")
            return key
        if exchange or user:
            raise ValueError("both exchange and user are required when selecting a monitor root")
        if not keys:
            raise FileNotFoundError(f"no monitor roots found under {self.monitor_root}")
        if len(keys) == 1:
            return keys[0]
        raise LookupError("multiple monitor roots available; specify exchange and user")

    def load_snapshot(self, key: MonitorKey) -> dict:
        path = self._bot_root(key) / "state.latest.json"
        if not path.exists():
            raise FileNotFoundError(f"snapshot not found for {key[0]}/{key[1]}")
        return json.loads(path.read_text(encoding="utf-8"))

    def build_snapshot_message(self, key: MonitorKey, snapshot: dict) -> dict:
        meta = snapshot.get("meta", {}) if isinstance(snapshot, dict) else {}
        return {
            "type": "snapshot",
            "exchange": key[0],
            "user": key[1],
            "seq": meta.get("seq"),
            "ts": meta.get("snapshot_ts_ms"),
            "payload": snapshot,
        }

    def build_health_payload(self) -> dict:
        keys = self.discover_keys()
        subscribers = {
            f"{exchange}/{user}": len(self._subscribers.get((exchange, user), set()))
            for exchange, user in keys
        }
        return {
            "status": "ok",
            "monitor_root": str(self.monitor_root),
            "poll_interval_ms": self.poll_interval_ms,
            "ws_replay_limit": self.ws_replay_limit,
            "uptime_ms": int((time.monotonic() - self.started_at_monotonic) * 1000.0),
            "bots": [{"exchange": exchange, "user": user} for exchange, user in keys],
            "subscribers": subscribers,
        }

    def subscribe(self, key: MonitorKey) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.subscriber_queue_size)
        self._subscribers.setdefault(key, set()).add(queue)
        return queue

    def unsubscribe(self, key: MonitorKey, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(key)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(key, None)

    async def start(self) -> None:
        if self._poll_task is not None:
            return
        self._prime_offsets()
        self._initial_prime_completed = True
        self._stop_event.clear()
        self._poll_task = asyncio.create_task(self._run_poll_loop())

    async def stop(self) -> None:
        task = self._poll_task
        if task is None:
            return
        self._stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._poll_task = None

    def _prime_offsets(self) -> None:
        for key in self.discover_keys():
            for path in self._current_paths_for_key(key):
                if not path.exists():
                    continue
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                self._path_states[path] = _PathState(
                    dev=int(stat.st_dev),
                    ino=int(stat.st_ino),
                    offset=int(stat.st_size),
                )

    async def _run_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as exc:
                logging.error("[monitor-relay] poll loop failed: %s", exc)
            await asyncio.sleep(self.poll_interval_ms / 1000.0)

    async def poll_once(self) -> None:
        for key in self.discover_keys():
            for path in self._current_paths_for_key(key):
                for message in self._read_updates(path):
                    await self._broadcast(key, message)

    def _read_updates(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            stat = path.stat()
        except FileNotFoundError:
            return []
        file_id = (int(stat.st_dev), int(stat.st_ino))
        size = int(stat.st_size)
        state = self._path_states.get(path)
        if state is None:
            if not self._initial_prime_completed:
                self._path_states[path] = _PathState(file_id[0], file_id[1], size)
                return []
            read_from = 0
        else:
            reset = size < state.offset or (state.dev, state.ino) != file_id
            read_from = 0 if reset else state.offset
        if size == read_from:
            self._path_states[path] = _PathState(file_id[0], file_id[1], size)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(read_from)
                lines = f.readlines()
                new_offset = int(f.tell())
        except FileNotFoundError:
            return []
        self._path_states[path] = _PathState(file_id[0], file_id[1], new_offset)
        messages: list[dict] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as exc:
                logging.warning("[monitor-relay] invalid JSON in %s: %s", path, exc)
                continue
            if path.parent.name == "events":
                messages.append(self._build_event_message(entry))
            else:
                messages.append(self._build_history_message(entry))
        return messages

    def load_recent_messages(self, key: MonitorKey, *, limit: Optional[int] = None) -> list[dict]:
        per_file_limit = self.ws_replay_limit if limit is None else max(0, int(limit))
        if per_file_limit <= 0:
            return []
        messages: list[tuple[int, int, dict]] = []
        order = 0
        for path in self._current_paths_for_key(key):
            for entry in self._read_recent_entries(path, per_file_limit):
                if path.parent.name == "events":
                    message = self._build_event_message(entry)
                else:
                    message = self._build_history_message(entry)
                ts = self._message_ts_ms(message)
                messages.append((ts, order, message))
                order += 1
        messages.sort(key=lambda item: (item[0], item[1]))
        return [message for _, _, message in messages]

    def _read_recent_entries(self, path: Path, limit: int) -> list[dict]:
        if limit <= 0 or not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = list(deque((line.rstrip("\n") for line in f), maxlen=limit))
        except FileNotFoundError:
            return []
        entries: list[dict] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as exc:
                logging.warning("[monitor-relay] invalid JSON in %s: %s", path, exc)
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _message_ts_ms(self, message: dict) -> int:
        value = message.get("ts")
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _build_event_message(self, entry: dict) -> dict:
        message = {
            "type": "event",
            "seq": entry.get("seq"),
            "ts": entry.get("ts"),
            "kind": entry.get("kind"),
            "exchange": entry.get("exchange"),
            "user": entry.get("user"),
            "tags": entry.get("tags", []),
            "payload": entry.get("payload", {}),
        }
        for key in ("symbol", "pside"):
            if key in entry:
                message[key] = entry[key]
        return message

    def _build_history_message(self, entry: dict) -> dict:
        message = {
            "type": "history",
            "ts": entry.get("ts"),
            "kind": entry.get("kind"),
            "stream": entry.get("stream"),
            "exchange": entry.get("exchange"),
            "user": entry.get("user"),
            "payload": entry.get("payload", {}),
        }
        for key in ("symbol", "pside", "timeframe"):
            if key in entry:
                message[key] = entry[key]
        return message

    async def _broadcast(self, key: MonitorKey, message: dict) -> None:
        subscribers = list(self._subscribers.get(key, set()))
        if not subscribers:
            return
        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self._replace_with_resync_required(queue)

    def _replace_with_resync_required(self, queue: asyncio.Queue) -> None:
        try:
            while True:
                queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(
                {
                    "type": "resync_required",
                    "reason": "subscriber_queue_overflow",
                }
            )
        except asyncio.QueueFull:
            pass


RELAY_APP_KEY = web.AppKey("monitor_relay", MonitorRelay)


def _relay_from_app(app: web.Application) -> MonitorRelay:
    return app[RELAY_APP_KEY]


def _resolve_request_key(relay: MonitorRelay, request: web.Request) -> MonitorKey:
    return relay.resolve_key(
        exchange=request.query.get("exchange"),
        user=request.query.get("user"),
    )


async def _handle_health(request: web.Request) -> web.Response:
    relay = _relay_from_app(request.app)
    return web.json_response(relay.build_health_payload())


async def _handle_snapshot(request: web.Request) -> web.Response:
    relay = _relay_from_app(request.app)
    try:
        key = _resolve_request_key(relay, request)
        snapshot = relay.load_snapshot(key)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except LookupError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.json_response(relay.build_snapshot_message(key, snapshot))


async def _handle_dashboard(request: web.Request) -> web.Response:
    path = DASHBOARD_STATIC_DIR / "index.html"
    if not path.exists():
        raise web.HTTPNotFound(text="dashboard index not found")
    return web.Response(text=path.read_text(encoding="utf-8"), content_type="text/html")


async def _handle_dashboard_asset(request: web.Request) -> web.Response:
    asset_name = request.match_info.get("name", "")
    asset = _DASHBOARD_ASSETS.get(asset_name)
    if asset is None:
        raise web.HTTPNotFound(text=f"dashboard asset not found: {asset_name}")
    filename, content_type = asset
    path = DASHBOARD_STATIC_DIR / filename
    if not path.exists():
        raise web.HTTPNotFound(text=f"dashboard asset not found: {asset_name}")
    return web.Response(text=path.read_text(encoding="utf-8"), content_type=content_type)


async def _handle_ws(request: web.Request) -> web.StreamResponse:
    relay = _relay_from_app(request.app)
    try:
        key = _resolve_request_key(relay, request)
        snapshot = relay.load_snapshot(key)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except LookupError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc

    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    await ws.send_json(relay.build_snapshot_message(key, snapshot))
    for message in relay.load_recent_messages(key):
        await ws.send_json(message)
    queue = relay.subscribe(key)
    try:
        while not ws.closed:
            message = await queue.get()
            await ws.send_json(message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logging.info("[monitor-relay] websocket closed for %s/%s: %s", key[0], key[1], exc)
    finally:
        relay.unsubscribe(key, queue)
        await ws.close()
    return ws


async def _on_startup(app: web.Application) -> None:
    await _relay_from_app(app).start()


async def _on_cleanup(app: web.Application) -> None:
    await _relay_from_app(app).stop()


def create_monitor_relay_app(
    *,
    monitor_root: str = "monitor",
    poll_interval_ms: int = 250,
    subscriber_queue_size: int = 1000,
    ws_replay_limit: int = 50,
) -> web.Application:
    relay = MonitorRelay(
        monitor_root=monitor_root,
        poll_interval_ms=poll_interval_ms,
        subscriber_queue_size=subscriber_queue_size,
        ws_replay_limit=ws_replay_limit,
    )
    app = web.Application()
    app[RELAY_APP_KEY] = relay
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/snapshot", _handle_snapshot)
    app.router.add_get("/dashboard", _handle_dashboard)
    app.router.add_get("/dashboard/assets/{name}", _handle_dashboard_asset)
    app.router.add_get("/ws", _handle_ws)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app
