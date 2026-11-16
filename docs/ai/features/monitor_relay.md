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

Send the available snapshot envelopes first, then replay, then live event and history messages. Prime existing current
files to EOF at startup. Read the connection replay tail directly from current files, independently
of live poll offsets. Tail newly discovered current files from the beginning.

Warn and skip invalid JSON lines. Log relay poll failures and continue because the monitor root on
disk remains authoritative.

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
