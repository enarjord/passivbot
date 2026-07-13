# Feature Contract Router

Open only the contract for the subsystem being changed.

| Feature | Contract |
|---|---|
| Exchange integrations and broker attribution | `exchange_integrations.md` |
| Stock perpetuals (HIP-3) | `stock_perps.md` |
| Candles, cache continuity, and projections | `candlestick_manager.md` |
| Fill/PnL ingestion and coverage | `fill_events_manager.md` |
| Raw versus snapped balance | `balance_routing.md` |
| Monitor persistence, recovery, rotation, retention | `monitor_persistence.md` |
| Monitor relay/dashboard | `monitor_relay.md` |
| Structured live events | `live_events.md` and `../generated/live_event_registry.md` |
| Trailing diagnostics tool | `trailing_diagnostics.md` |
| Strategy schema and Rust runtime | `strategy_runtime.md` |

Feature contracts contain current invariants, failure semantics, non-obvious edge cases,
validation, and code/test locations. They do not carry progress ledgers or generic coding advice.
