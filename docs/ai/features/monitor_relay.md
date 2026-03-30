# Monitor Relay

## Contract

1. The relay is read-only and must not affect bot runtime behavior or trading decisions.
2. The monitor root on disk remains the source of truth; the relay is a network view over that data.
3. WebSocket clients must receive a snapshot first, then live-forward event/history messages.
4. Relay failures must stay contained to the relay process and must not require bot-process changes.

## Current Implementation Scope

Implemented now:

1. `src/monitor_relay.py` with:
   - monitor-root discovery by `{exchange}/{user}`
   - `GET /health`
   - `GET /snapshot`
   - `GET /ws`
   - `GET /dashboard`
   - `GET /dashboard/assets/{name}`
2. `src/tools/monitor_relay.py` CLI wrapper for local serving
3. polling of:
   - `events/current.ndjson`
   - `history/*.current.ndjson`
4. snapshot-first WebSocket bootstrap, then live `event` / `history` push messages
5. aggregate-capable subscriber queues with `exchange` + `user` tags on every message
6. websocket connect also replays a recent tail from the current event/history files so new TUI and dashboard clients can populate recent panels immediately when attaching to an already-running bot
7. the relay also serves a static browser dashboard that consumes the multiplexed `/snapshot` + `/ws` feed from the same origin without requiring a frontend build step

## Non-Obvious Details

1. Existing current files are primed to EOF on relay startup so reconnecting the relay does not replay the entire current segment by default.
2. The websocket replay tail is read directly from the current files on connect and is independent of the live poll offsets.
3. New current files created after relay startup are tailed from the beginning so their first live entries are not skipped.
4. The relay is intentionally best-effort on malformed monitor lines: invalid JSON is skipped with a warning instead of killing the process.
5. The poll loop logs relay-side failures and continues; this is acceptable because the relay is observability-only and disk remains authoritative.
6. Browser focus selection is client-side only. Query params may choose the initial bot/symbol focus, but the dashboard still attaches to the aggregate relay feed.
7. Dashboard assets are intentionally tiny static files under `src/monitor_dashboard_static/`; keep the browser reader read-only and avoid coupling it to bot internals.

## Test Focus

1. Snapshot and health handlers should work without binding sockets.
2. Aggregate snapshot/websocket handlers should work clearly when `exchange` + `user` are omitted.
3. WebSocket flow should send snapshot first, then replay backlog, then live updates.
4. New current files created after relay startup should publish their first entries.
5. Dashboard routes should serve expected static assets without binding sockets.
6. Browser dashboard should remain multi-bot first-class even when query params are used for initial focus.

## Key Code

- `src/monitor_relay.py`
- `src/monitor_dashboard_static/index.html`
- `src/monitor_dashboard_static/dashboard.css`
- `src/monitor_dashboard_static/dashboard.js`
- `src/tools/monitor_relay.py`
- `tests/test_monitor_relay.py`
- `docs/monitor.md`
