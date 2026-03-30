# Monitor Relay

Passivbot's monitor stack is split into two layers:

1. The live bot can publish local monitor artifacts under `monitor/<exchange>/<user>/`.
2. The relay can auto-discover those artifacts and expose them over a read-only HTTP/websocket interface for local tools.

## Start The Relay

```bash
passivbot tool monitor-relay --monitor-root monitor
```

Direct script usage also works:

```bash
python src/tools/monitor_relay.py --monitor-root monitor
```

Default bind address is `127.0.0.1:8765`.

The relay is intended to be a single long-running process per repo checkout:

1. Start it before or after any bot.
2. It auto-detects active monitor publishers under `monitor/`.
3. It keeps relaying as bots start, stop, or restart.
4. One relay can fan in any number of bots from the same repo.

## Endpoints

- `GET /health`
  - Returns discovered monitor roots, active status, and relay health metadata.
- `GET /snapshot`
  - Returns the latest snapshot envelope for one selected bot when `exchange` and `user` are provided.
  - Otherwise returns a `snapshot_bundle` containing the latest snapshot envelopes for all active bots.
- `GET /dashboard`
  - Serves a static browser dashboard bootstrapped from the same relay contract.
- `GET /ws`
  - With `exchange` and `user`, streams one selected bot.
  - Without filters, sends startup snapshots for all active bots, replays recent messages across all active bots, then streams live updates from all active bots on one websocket.
  - All snapshot, event, and history messages carry `exchange` and `user` so consumers can differentiate bots.

## Notes

- The relay is read-only. It does not mutate monitor artifacts.
- The websocket stream is intended for local dashboards, TUIs, replay tooling, and other fan-in consumers.
- The browser dashboard and TUI both consume the same read-only `/snapshot` + `/ws` surface.

## Web Dashboard

The relay also serves the main browser monitor:

```bash
passivbot tool monitor-web
```

If you prefer to manage the relay yourself:

```bash
passivbot tool monitor-relay --monitor-root monitor
```

Current behavior:

1. Connects to the aggregate `/snapshot` + `/ws` relay feed for all active bots
2. Shows a dense overview card for every active bot discovered by the relay
3. Lets operators click any bot card to focus it without restarting the dashboard
4. Keeps a focused-bot detail view for summary, focus, positions, trailing, forager, unstuck, recent events, recent ticks, and recent orders
5. Preserves a separate symbol focus inside the currently selected bot
6. Accepts optional `exchange`, `user`, and `symbol` query params for initial focus only; the dashboard still consumes the multiplexed relay feed

Example initial-focus URL:

```text
http://127.0.0.1:8765/dashboard?exchange=bitget&user=bitget_01&symbol=BTC/USDT:USDT
```

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
