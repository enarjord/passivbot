# Passivbot Architecture (Lean)

## Purpose

Passivbot is split between Python orchestration and Rust trading logic.

## High-Level Map

```
Python (src/)                          Rust (passivbot-rust/src/)
├── passivbot.py (live loop)    ────►  ├── orchestrator.rs (order calc)
├── backtest.py (coordinator)   ────►  ├── backtest.rs (simulation)
├── optimize.py (optimizer)     ────►  └── analysis.rs (metrics)
├── candlestick_manager.py (OHLCV)
├── fill_events_manager.py (fills/PnL)
└── exchanges/ (API wrappers)
```

## Source-of-Truth Boundary

1. Rust (`passivbot-rust/src/`) owns order behavior.
- Includes orchestrator, entries, closes, risk, backtest simulation, and analysis metrics.
2. Python (`src/`) owns orchestration.
- Includes exchange API calls, config loading, data collection/caching, and process control.

If behavior changes in entries/closes/risk/unstuck, implement in Rust.

## Core Runtime Components

| Component | Role |
|-----------|------|
| `src/passivbot.py` | Live loop, reconciliation, execution |
| `src/backtest.py` | Historical run coordinator |
| `src/optimize.py` | Optimizer driver |
| `src/candlestick_manager.py` | OHLCV fetch/cache/synthetic candles |
| `src/fill_events_manager.py` | Fill/PnL event ingestion |
| `src/exchanges/*` | Exchange adapters |

## Data Flows

### Live

1. Fetch exchange state (positions/balance/orders).
2. Refresh candles and indicators.
3. Build orchestrator input.
4. Call Rust orchestrator.
5. Reconcile ideal vs actual orders.
6. Create/cancel orders.

### Backtest

1. Load config + date range.
2. Load/prep OHLCV.
3. Run Rust backtest.
4. Emit metrics/results.

### Optimize

1. Generate candidate configs.
2. Evaluate with Rust backtest.
3. Select next generation.

## Config Hierarchy

1. `config.live`: behavior shared by live/backtest/optimizer.
2. `config.backtest`: simulation-only settings.
3. `config.optimize`: optimizer-only settings.

Rule: if unsure, prefer `config.live`.

## Stateless Requirement

Trading decisions must remain reproducible after restart from exchange state + config.
