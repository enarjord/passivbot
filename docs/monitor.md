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

## Terminal TUI

The repo also includes a minimal terminal reader against the relay:

```bash
passivbot tool monitor-tui --relay-url http://127.0.0.1:8765
```

When multiple monitor roots are available on the relay, select one bot explicitly:

```bash
passivbot tool monitor-tui \
  --relay-url http://127.0.0.1:8765 \
  --exchange bitget \
  --user bitget_01 \
  --focus-symbol BTC/USDT:USDT
```

Current behavior:

1. Bootstraps current-state panels from `/snapshot`
2. Keeps recent events and price ticks live through `/ws`
3. Supports `focus`, `pause`, `resume`, `dump`, `clear`, `help`, `quit`, and `exit`
4. Renders a lightweight boxed terminal dashboard without reading bot memory or raw monitor files directly

## Monitor Dev Wrapper

For local monitor iteration there is also a convenience wrapper:

```bash
passivbot tool monitor-dev --exchange bitget --user bitget_01
```

Current behavior:

1. Reuses an existing relay if one is already healthy at the target `--relay-url`
2. Otherwise launches the relay automatically for local relay URLs
3. Selects the newest `logs/*.log` file by default unless `--log-file` is provided
4. Passes the selected bot log into the TUI for local tailing
