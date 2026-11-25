# Monitor Relay

## Contract

The relay is a read-only network view of [monitor files](monitor_persistence.md).
Keep relay failures contained to its process and bot behavior independent of relay availability.
The [operator guide](../../monitor.md) owns endpoints, origin configuration, and dashboard usage.

Before loading or sending data, require HTTP Host to match an allowed origin authority and any
WebSocket Origin to match an allowed origin. This prevents DNS rebinding and still permits
originless native clients. Configure proxy origins explicitly. Ignore forwarded headers for these
checks, which are separate from authentication.

## Delivery

Send the available snapshot envelopes first, then replay, then live event and history messages.
Prime existing current files through the last complete row at startup. Read the connection replay
tail directly from current files, independently of live poll offsets. Tail newly discovered and
rotated current files from the beginning. Bind the live byte offset to the opened file identity,
so a pathname replacement cannot reuse the preceding file's offset.

Register the bounded live queue before sending snapshots and replay. Buffer new publications during
those sends and release the subscription on every exit. Replay and queued live delivery may overlap.
Queue overflow retains the existing resync-required signal during connection setup.

Warn and skip invalid JSON lines. Log relay poll failures and continue because the monitor root on
disk remains authoritative.

An invalid unterminated final row is an unfinished append. Retain its live byte offset and retry
after more bytes arrive. Replay also defers that row. Complete JSON without a final newline remains
readable, including older files. Split UTF-8 characters must not prevent delivery of preceding rows.

Poll batches yield between groups of live messages to let subscriber writers drain during
event and history bursts. A subscriber that still overflows receives the resync-required signal.

The dashboard uses the aggregate feed. Query parameters only set initial client-side bot and symbol
focus. Keep its static reader small, read-only, and independent of bot internals.

## Validation

- snapshot, health, and static asset handlers without binding sockets
- aggregate snapshot and WebSocket without exchange and user selectors
- snapshot, replay, and live update ordering
- delivery of first entries in files created after relay startup
- multi-bot dashboard with initial-focus query parameters

## Key Code

- `src/monitor_relay.py`
- `src/monitor_dashboard_static/`
- `src/tools/monitor_relay.py`
- `tests/test_monitor_relay.py`
