# Feature Docs Router

Open these only when touching the relevant subsystem.

| Feature | File | Read when |
|---------|------|-----------|
| Stock perps (HIP-3) | `stock_perps.md` | Hyperliquid stock symbols, routing, margin mode |
| Candlestick manager | `candlestick_manager.md` | OHLCV fetch/cache/synthetic candle behavior |
| Fill events manager | `fill_events_manager.md` | Fill/PnL ingestion and pagination |
| Balance routing | `balance_routing.md` | Raw vs snapped balance semantics |

## Authoring Rule

Feature docs should include only:

1. Contract/invariants
2. Non-obvious edge cases
3. Test targets
4. Key code locations
