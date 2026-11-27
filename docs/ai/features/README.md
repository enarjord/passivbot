# Feature Contract Router

Open only the contract for the subsystem being changed.

| Feature | Contract |
|---|---|
| Historical coin eligibility in backtests and native optimization | `organillo.md` |
| Core backtest/optimizer metric arithmetic and statuses | `backtest_metrics.md` |
| Exchange integrations and broker attribution | `exchange_integrations.md`, then the venue file under `exchanges/` |
| Stock perpetuals (HIP-3) | `stock_perps.md` |
| Candles, cache continuity, and projections | `candlestick_manager.md` |
| Fill/PnL ingestion and coverage | `fill_events_manager.md` |
| Historical runtime/fill attribution | `runtime_attribution.md` |
| Equity Hard Stop Loss episodes and cooldown boundaries | `equity_hard_stop_loss.md` |
| Raw versus snapped balance | `balance_routing.md` |
| Monitor persistence, recovery, rotation, retention | `monitor_persistence.md` |
| Monitor relay/dashboard | `monitor_relay.md` |
| Structured live events | `live_events.md`, the family file under `live_events/`, and `../generated/live_event_registry.md` |
| Trailing diagnostics tool | `trailing_diagnostics.md` |
| Strategy schema and Rust runtime | `strategy_runtime.md` |

Feature contracts contain current invariants, failure semantics, non-obvious edge cases,
validation, and code/test locations. They do not carry progress ledgers or generic coding advice.

Subsystems without an AI contract are described by their user guides, which state current
behavior but not review invariants. Read the guide and the code before changing them:

| Subsystem | Guide |
|---|---|
| GPU optimizer and backtest service | `../../gpu_backtest_service.md`, `../../optimizing.md`; active work: `../../plans/gpu_optimizer_contract.md` |
| Backtest runs and reports | `../../backtesting.md` |
| Forager selection | `../../forager.md` |
| Coin overrides | `../../coin_overrides.md` |
| Config prefilter | `../../config_prefilter.md` |
| Monitor CLI and dashboard | `../../monitor.md` |
| Container deployment | `../../container_deployment.md` |
