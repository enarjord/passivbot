# Feature Docs Router

Open these only when touching the relevant subsystem.

| Feature | File | Read when |
|---------|------|-----------|
| Stock perps (HIP-3) | `stock_perps.md` | Hyperliquid stock symbols, routing, margin mode |
| Candlestick manager | `candlestick_manager.md` | OHLCV fetch/cache/synthetic candle behavior |
| Fill events manager | `fill_events_manager.md` | Fill/PnL ingestion and pagination |
| Balance routing | `balance_routing.md` | Raw vs snapped balance semantics |
| Monitor branch scope | `monitor_branch_scope.md` | Working on the extracted monitor branch or transplanting monitor work between branches |
| Monitor publisher | `monitor_publisher.md` | Monitor snapshot/event publication, schema boundaries, dashboard handoff work |
| Monitor relay | `monitor_relay.md` | Read-only monitor HTTP/WebSocket relay server |
| Monitor TUI | `monitor_tui.md` | Minimal terminal reader consuming relay snapshot + websocket data |

## Authoring Rule

Feature docs should include only:

1. Contract/invariants
2. Non-obvious edge cases
3. Test targets
4. Key code locations
