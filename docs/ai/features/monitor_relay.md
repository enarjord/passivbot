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
2. `src/tools/monitor_relay.py` CLI wrapper for local serving
3. polling of:
   - `events/current.ndjson`
   - `history/*.current.ndjson`
4. snapshot-first WebSocket bootstrap, then live `event` / `history` push messages
5. per-bot subscriber queues with `resync_required` when a subscriber falls behind

## Non-Obvious Details

1. Existing current files are primed to EOF on relay startup so reconnecting the relay does not replay the entire current segment by default.
2. New current files created after relay startup are tailed from the beginning so their first live entries are not skipped.
3. The relay is intentionally best-effort on malformed monitor lines: invalid JSON is skipped with a warning instead of killing the process.
4. The poll loop logs relay-side failures and continues; this is acceptable because the relay is observability-only and disk remains authoritative.

## Gaps Still Open

1. No auth or access control yet.
2. No per-stream or per-symbol subscription filters yet.
3. No HTTP endpoints for history/replay/rotated segments yet.
4. No client-side resync helper beyond the `resync_required` message.

## Test Focus

1. snapshot and health handlers should work without binding sockets
2. multi-root selection should fail clearly when `exchange` + `user` are omitted
3. websocket flow should send snapshot first, then live updates
4. new current files created after relay startup should publish their first entries

## Key Code

- `src/monitor_relay.py`
- `src/tools/monitor_relay.py`
- `tests/test_monitor_relay.py`
- `docs/monitor.md`
- `docs/plans/passivbot_monitor_dashboard.md`
