# Monitor Relay

Passivbot's monitor stack is split into two layers:

1. The live bot can publish local monitor artifacts under `monitor/<exchange>/<user>/`.
2. The relay can expose those artifacts over a read-only HTTP/websocket interface for local tools.

## Start The Relay

```bash
passivbot tool monitor-relay --monitor-root monitor
```

Direct script usage also works:

```bash
python src/tools/monitor_relay.py --monitor-root monitor
```

Default bind address is `127.0.0.1:8765`.

## Endpoints

- `GET /health`
  - Returns discovered monitor roots plus relay health metadata.
- `GET /snapshot`
  - Returns the latest snapshot envelope for the selected bot.
  - If multiple monitor roots exist, pass `?exchange=<exchange>&user=<user>`.
- `GET /ws`
  - Sends the latest snapshot first.
  - Replays recent messages from current event/history files.
  - Then streams live updates as new lines are appended by the publisher.

## Notes

- The relay is read-only. It does not mutate monitor artifacts.
- The websocket stream is intended for local dashboards, TUIs, and replay tooling.
- Current integration is intentionally limited to snapshot, event, and history relay. The TUI and web dashboard layers are separate follow-up steps.
