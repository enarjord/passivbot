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
- Includes exchange API calls, config loading, data collection/caching, reconciliation, and
  execution gating.

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
6. Gate proposed cancel/create actions.
7. Create/cancel gate-approved orders.

### Intended Live Execution Contract

Rust remains the source of truth for ideal order behavior. Python live code should orchestrate
exchange I/O, state refresh, reconciliation, and execution gating without reimplementing strategy
intent.

The intended live path is:

1. Rust proposes ideal order behavior.
2. The reconciler owns ideal-vs-actual matching and equivalence decisions, including
   `satisfied_existing` outcomes when current exchange orders already satisfy the ideal action.
3. A pure gatekeeper evaluates proposed order actions between reconciliation and exchange I/O.
4. The executor submits only gate-approved cancels/creates.

Gatekeeper decisions should be structured, not bare booleans:

1. decision: `approved`, `deferred`, `rejected`, or `satisfied_existing`
2. reason code
3. required and missing freshness surfaces
4. log severity and context

The gatekeeper evaluates concrete proposed actions, not strategy intent alone. The executor should
eventually be dumb: it should submit approved exchange requests and avoid scattered policy checks.

Use exact terminology in this boundary: `position_side`/`pside` means `long`/`short`;
`side`/`order_side` means `buy`/`sell`.

`graceful_stop` does not mean no management. It blocks new initial entries once flat, while still
allowing position management, risk-increasing DCA while a position exists, and all closes.

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
