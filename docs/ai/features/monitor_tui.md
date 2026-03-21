# Monitor TUI

## Contract

1. The TUI is read-only and must consume only the relay API, not bot memory or direct monitor files.
2. Current-state panels must come from `/snapshot`, not from partial event inference.
3. WebSocket data is supplemental live-forward activity for recent events/history, not a replacement for snapshot refresh.
4. TUI failures must stay isolated to the tool process.

## Current Implementation Scope

Implemented now:

1. `src/monitor_tui.py` state/render logic
2. `src/tools/monitor_tui.py` CLI wrapper
3. `src/monitor_dev.py` plus `src/tools/monitor_dev.py` local dev launcher
4. periodic `/snapshot` refresh for current-state panels
5. `/ws` consumption for recent events and price-tick activity
6. optional local bot-log tailing panel
7. optional `--focus-symbol` prioritization for one market
8. bottom command prompt with focus/quit/help commands
9. dependency-free ANSI terminal rendering
10. detailed per-side position rows with WE/WEL/WELe/TWEL, price-action distance, and uPnL from snapshot data

## Non-Obvious Details

1. The relay does not currently push fresh snapshots over websocket, so the TUI must keep polling `/snapshot`.
2. The first TUI is intentionally narrow but now includes account/health/HSL summaries, active positions/orders, focused-symbol detail, recent order activity, recent events, recent price ticks, and detailed per-side exposure metrics for active positions.
3. The local log-tail panel is for operator convenience only; it is not part of the relay protocol.
4. This tool should remain a consumer of the relay contract, not become a second monitor-file parser.

## Gaps Still Open

1. No richer keybindings beyond the current command prompt yet.
2. No per-panel filtering yet.
3. No richer history/candle visualization yet.
4. No auth flow because the relay itself has no auth yet.

## Test Focus

1. snapshot/event/history application to local state
2. stable rendering of the main screen from representative snapshot data
3. focused-symbol prioritization should affect the rendered view deterministically
4. command parsing should support focus aliases and quit aliases deterministically

## Key Code

- `src/monitor_tui.py`
- `src/tools/monitor_tui.py`
- `src/monitor_dev.py`
- `src/tools/monitor_dev.py`
- `tests/test_monitor_tui.py`
- `tests/test_monitor_dev.py`
- `docs/monitor.md`
- `docs/plans/passivbot_monitor_dashboard.md`
