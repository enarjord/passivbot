from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    """Multi-bot monitor relay.

    Scans all passivbot instances writing snapshots under ``monitor_root`` and
    fans out their events/history to WebSocket subscribers.  A subscriber may
    opt-in to a specific bot (exchange + user) or receive a merged feed from
    every currently-active bot.

    Auto-detection rules:
    - A bot directory with a ``manifest.json`` is eligible.
    - A bot is considered *active* if its most-recent file activity falls within
      the staleness window (5× the snapshot interval, minimum 30 s).
    - The poll loop re-checks ``active_keys()`` every cycle, so bots starting
      after the relay is already running are picked up automatically, and bots
      that stop writing are silently dropped from the feed.
    """

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
        # Maps each subscriber queue to its optional key filter.
        # None means "all active bots"; a MonitorKey means a specific bot only.
        self._subscribers: dict[asyncio.Queue, Optional[MonitorKey]] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._initial_prime_completed = False

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def discover_keys(self) -> list[MonitorKey]:
        """Return all bot keys that have a manifest.json, active or not."""
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

    def _manifest_path(self, key: MonitorKey) -> Path:
        return self._bot_root(key) / "manifest.json"

    def _snapshot_path(self, key: MonitorKey) -> Path:
        return self._bot_root(key) / "state.latest.json"

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

    # ------------------------------------------------------------------
    # Staleness / activity detection
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _path_mtime_ms(self, path: Path) -> int:
        try:
            return int(path.stat().st_mtime * 1000.0)
        except FileNotFoundError:
            return 0

    def _extract_manifest_snapshot_interval_ms(self, manifest: dict[str, Any]) -> int:
        raw = manifest.get("config", {}).get("snapshot_interval_seconds")
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        return max(0, int(value * 1000.0))

    def _key_last_activity_ts_ms(self, key: MonitorKey) -> int:
        """Return the most-recent timestamp (ms) of any file activity for this key."""
        manifest = self._load_json(self._manifest_path(key))
        snapshot = self._load_json(self._snapshot_path(key))
        candidates = [
            self._path_mtime_ms(self._manifest_path(key)),
            self._path_mtime_ms(self._snapshot_path(key)),
            int(manifest.get("updated_ts_ms", 0) or 0),
            int(snapshot.get("meta", {}).get("snapshot_ts_ms", 0) or 0),
        ]
        for path in self._current_paths_for_key(key):
            candidates.append(self._path_mtime_ms(path))
        return max(candidates or [0])

    def _key_stale_after_ms(self, key: MonitorKey) -> int:
        """Return how many ms of silence before we consider this bot inactive."""
        manifest = self._load_json(self._manifest_path(key))
        snapshot_interval_ms = self._extract_manifest_snapshot_interval_ms(manifest)
        return max(30_000, snapshot_interval_ms * 5 if snapshot_interval_ms else 0)

    def _is_key_active(self, key: MonitorKey, *, now_ms: Optional[int] = None) -> bool:
        now_ms = int(time.time() * 1000.0) if now_ms is None else int(now_ms)
        last_activity_ms = self._key_last_activity_ts_ms(key)
        if last_activity_ms <= 0:
            return False
        return now_ms - last_activity_ms <= self._key_stale_after_ms(key)

    def active_keys(self) -> list[MonitorKey]:
        """Return all bot keys that are currently writing snapshots."""
        now_ms = int(time.time() * 1000.0)
        return [key for key in self.discover_keys() if self._is_key_active(key, now_ms=now_ms)]

    def matching_keys(
        self,
        *,
        exchange: Optional[str],
        user: Optional[str],
        active_only: bool = True,
    ) -> list[MonitorKey]:
        """Return keys matching the given exchange/user filter.

        If neither is specified, returns all active (or all discovered) keys.
        If both are specified, returns that single key or raises if not found.
        Specifying only one raises ValueError.
        """
        keys = self.active_keys() if active_only else self.discover_keys()
        if exchange and user:
            key = (str(exchange), str(user))
            if key not in keys:
                raise FileNotFoundError(f"monitor root not found for {exchange}/{user}")
            return [key]
        if exchange or user:
            raise ValueError("both exchange and user are required when selecting a monitor root")
        return keys

    def resolve_key(
        self,
        *,
        exchange: Optional[str],
        user: Optional[str],
    ) -> MonitorKey:
        """Resolve to a single key; raises if ambiguous or not found."""
        keys = self.matching_keys(exchange=exchange, user=user)
        if exchange and user:
            return keys[0]
        if not keys:
            raise FileNotFoundError(f"no active monitor roots found under {self.monitor_root}")
        if len(keys) == 1:
            return keys[0]
        raise LookupError("multiple monitor roots available; specify exchange and user")

    # ------------------------------------------------------------------
    # Snapshot loading
    # ------------------------------------------------------------------

    def load_snapshot(self, key: MonitorKey) -> dict:
        path = self._snapshot_path(key)
        if not path.exists():
            raise FileNotFoundError(f"snapshot not found for {key[0]}/{key[1]}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_snapshot_messages(
        self,
        *,
        exchange: Optional[str],
        user: Optional[str],
    ) -> list[dict]:
        """Load snapshot messages for all matching active bots."""
        messages: list[dict] = []
        for key in self.matching_keys(exchange=exchange, user=user):
            try:
                snapshot = self.load_snapshot(key)
            except FileNotFoundError:
                continue
            messages.append(self.build_snapshot_message(key, snapshot))
        return messages

    def load_manifest(self, key: MonitorKey) -> dict:
        path = self._manifest_path(key)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found for {key[0]}/{key[1]}")
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

    def build_snapshot_bundle(self, messages: list[dict]) -> dict:
        """Wrap multiple snapshot messages into a single bundle envelope."""
        now_ms = int(time.time() * 1000.0)
        return {
            "type": "snapshot_bundle",
            "ts": now_ms,
            "count": len(messages),
            "bots": messages,
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def build_health_payload(self) -> dict:
        discovered = self.discover_keys()
        active = set(self.active_keys())
        subscribers: dict[str, int] = {}
        for exchange, user in discovered:
            count = 0
            key = (exchange, user)
            for subscription in self._subscribers.values():
                if subscription is None or subscription == key:
                    count += 1
            subscribers[f"{exchange}/{user}"] = count
        return {
            "status": "ok",
            "monitor_root": str(self.monitor_root),
            "poll_interval_ms": self.poll_interval_ms,
            "ws_replay_limit": self.ws_replay_limit,
            "uptime_ms": int((time.monotonic() - self.started_at_monotonic) * 1000.0),
            "bots": [
                {
                    "exchange": exchange,
                    "user": user,
                    "active": (exchange, user) in active,
                    "last_activity_ts_ms": self._key_last_activity_ts_ms((exchange, user)),
                }
                for exchange, user in discovered
            ],
            "subscribers": subscribers,
        }

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, key: Optional[MonitorKey] = None) -> asyncio.Queue:
        """Return a new subscriber queue.

        Pass ``key=None`` (default) to receive events from all active bots.
        Pass a specific ``MonitorKey`` to filter to one bot.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.subscriber_queue_size)
        self._subscribers[queue] = key
        return queue

    def unsubscribe(
        self,
        queue_or_key: asyncio.Queue | MonitorKey,
        queue: Optional[asyncio.Queue] = None,
    ) -> None:
        """Remove a subscriber queue.

        Accepts either ``unsubscribe(queue)`` or the legacy two-argument form
        ``unsubscribe(key, queue)`` for backward compatibility.
        """
        if queue is None:
            target = queue_or_key
        else:
            target = queue
        self._subscribers.pop(target, None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        """Skip existing file content so we only tail new writes."""
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

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _run_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as exc:
                logging.error("[monitor-relay] poll loop failed: %s", exc)
            await asyncio.sleep(self.poll_interval_ms / 1000.0)

    async def poll_once(self) -> None:
        """Check all currently-active bots for new events/history lines."""
        for key in self.active_keys():
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

    def load_recent_messages(
        self,
        key: Optional[MonitorKey] = None,
        *,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Load recent messages for replay on WS connect.

        Pass ``key=None`` to load from all active bots (multi-bot mode).
        """
        per_file_limit = self.ws_replay_limit if limit is None else max(0, int(limit))
        if per_file_limit <= 0:
            return []
        messages: list[tuple[int, int, dict]] = []
        order = 0
        keys = [key] if key is not None else self.active_keys()
        for current_key in keys:
            for path in self._current_paths_for_key(current_key):
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
        """Send message to all subscribers interested in this key."""
        subscribers = list(self._subscribers.items())
        if not subscribers:
            return
        for queue, subscription in subscribers:
            # subscription=None means all-bots; otherwise filter to specific key
            if subscription is not None and subscription != key:
                continue
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


# ------------------------------------------------------------------
# aiohttp application
# ------------------------------------------------------------------

RELAY_APP_KEY = web.AppKey("monitor_relay", MonitorRelay)


def _relay_from_app(app: web.Application) -> MonitorRelay:
    return app[RELAY_APP_KEY]


async def _handle_health(request: web.Request) -> web.Response:
    relay = _relay_from_app(request.app)
    return web.json_response(relay.build_health_payload())


async def _handle_snapshot(request: web.Request) -> web.Response:
    """Return snapshot(s) for the requested bot(s).

    Without query params: returns all active bots (bundle if >1).
    With ``exchange`` + ``user``: returns that specific bot.
    """
    relay = _relay_from_app(request.app)
    exchange = request.query.get("exchange")
    user = request.query.get("user")
    try:
        messages = relay.load_snapshot_messages(exchange=exchange, user=user)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except LookupError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    if not messages:
        raise web.HTTPNotFound(text="no active monitor snapshots available")
    if len(messages) == 1:
        return web.json_response(messages[0])
    return web.json_response(relay.build_snapshot_bundle(messages))


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
    """WebSocket endpoint.

    Without query params: subscribes to all active bots (merged feed).
    With ``exchange`` + ``user``: subscribes to that specific bot only.

    On connect the client receives:
    1. One ``snapshot`` message per active matching bot (or a single
       ``snapshot_bundle`` if multiple).
    2. Recent replayed events/history for those bots.
    3. A live stream of new events/history as they are written.
    """
    relay = _relay_from_app(request.app)
    exchange = request.query.get("exchange")
    user = request.query.get("user")
    try:
        keys = relay.matching_keys(exchange=exchange, user=user)
        snapshot_messages = relay.load_snapshot_messages(exchange=exchange, user=user)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc)) from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except LookupError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    if not snapshot_messages:
        raise web.HTTPNotFound(text="no active monitor snapshots available")

    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)

    # Send initial snapshot(s)
    for message in snapshot_messages:
        await ws.send_json(message)

    # Determine subscription scope: specific bot or all-bots
    subscription: Optional[MonitorKey] = keys[0] if len(keys) == 1 else None

    # Replay recent messages
    for message in relay.load_recent_messages(subscription):
        await ws.send_json(message)

    queue = relay.subscribe(subscription)
    try:
        while not ws.closed:
            message = await queue.get()
            await ws.send_json(message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        scope = (
            f"{subscription[0]}/{subscription[1]}"
            if subscription is not None
            else "all-bots"
        )
        logging.info("[monitor-relay] websocket closed for %s: %s", scope, exc)
    finally:
        relay.unsubscribe(queue)
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
