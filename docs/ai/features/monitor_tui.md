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
8. bottom command prompt with focus/pause/resume/dump/quit/help commands
9. dependency-free ANSI terminal rendering
10. detailed per-side position rows with WE/WEL/WELe, price-action distance, and uPnL from snapshot data plus a box-level total-TWE summary
11. boxed single-column / two-column layout depending on terminal width
12. redraw only when the rendered frame changes, using row-diff ANSI updates instead of repainting the whole frame each interval
13. dedicated `Forager` and `Unstuck` boxes driven from snapshot data
14. recent price-tick rows also show current EMA-band snapshots from `snapshot.market[*].ema_bands`

## Non-Obvious Details

1. The relay does not currently push fresh snapshots over websocket, so the TUI must keep polling `/snapshot`.
2. The first TUI is intentionally narrow but now includes account/health/HSL summaries, active positions/orders, focused-symbol detail, dedicated forager/unstuck panels, recent order activity, recent events, recent price ticks, and detailed per-side exposure metrics for active positions.
3. `pause` intentionally freezes the data panels so users can mark/copy text, but the command footer remains live so `resume` and `dump` still show while paused.
4. `dump` writes the current rendered screen to `tmp/monitor_tui_dump_*.txt` for copy/share purposes.
5. The layout is intentionally simple ASCII boxes rather than a full terminal UI framework.
6. The local log-tail panel is for operator convenience only; it is not part of the relay protocol.
7. This tool should remain a consumer of the relay contract, not become a second monitor-file parser.

## Gaps Still Open

1. No richer keybindings beyond the current command prompt yet.
2. No per-panel filtering yet.
3. No richer history/candle visualization yet.
4. No auth flow because the relay itself has no auth yet.

## Test Focus

1. snapshot/event/history application to local state
2. stable rendering of the main screen from representative snapshot data
3. focused-symbol prioritization should affect the rendered view deterministically
4. command parsing should support focus aliases, pause/resume/dump, and quit aliases deterministically

## Key Code

- `src/monitor_tui.py`
- `src/tools/monitor_tui.py`
- `src/monitor_dev.py`
- `src/tools/monitor_dev.py`
- `tests/test_monitor_tui.py`
- `tests/test_monitor_dev.py`
- `docs/monitor.md`
- `docs/plans/passivbot_monitor_dashboard.md`
