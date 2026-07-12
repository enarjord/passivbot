# Passivbot Architecture

## Ownership Boundary

```text
Python (src/)                              Rust (passivbot-rust/src/)
├── live orchestration and exchange I/O ─►├── strategy and order calculation
├── config and data preparation          ├── risk and unstuck behavior
├── reconciliation and execution gating  ├── backtest simulation
└── backtest/optimizer coordination      └── behavioral analysis metrics
```

Rust owns trading behavior. Python may determine whether a proposed action is currently executable
from fresh exchange state, but must not reimplement strategy intent.

## Core Python Components

| Component | Responsibility |
|---|---|
| `src/passivbot.py` and `src/live/` | Live orchestration, state refresh, reconciliation, gating, execution |
| `src/backtest.py` | Historical-run coordination |
| `src/optimize.py` | Optimizer coordination |
| `src/candlestick_manager.py` | OHLCV fetch, cache, continuity, and projections |
| `src/fill_events_manager.py` | Fill/PnL ingestion and coverage |
| `src/exchanges/` | Exchange adapters and payload normalization |
| `src/config/` | Canonical config loading and access |

## Live Data Flow

1. Refresh account and relevant market state.
2. Build canonical strategy and risk inputs.
3. Ask Rust for ideal orders.
4. Reconcile ideal orders with exchange orders.
5. Evaluate concrete create/cancel actions against freshness and safety gates.
6. Submit only approved actions.
7. Confirm or classify ambiguous exchange outcomes before retrying.

Reconciliation owns equivalence decisions such as `satisfied_existing`. Gatekeeper decisions are
structured (`approved`, `deferred`, `rejected`, or `satisfied_existing`) and include stable reason
codes plus required/missing surfaces. The executor submits approved requests; it does not invent
strategy policy.

`graceful_stop` blocks new initial entries once flat, but does not disable management of an existing
position, including configured DCA and closes.

## Backtest And Optimize Flows

Backtest loads config and OHLCV, invokes the Rust simulation, and emits results. Optimize generates
candidate configs and evaluates them through the same Rust backtest contract. Differences between
live and backtest runtime inputs require explicit parity tests rather than duplicated assumptions.

## Configuration Surfaces

Use `config.live` only for behavior consumed by live and shared with backtest/optimizer.
Simulation-only settings belong in `config.backtest`; optimizer-only settings belong in
`config.optimize`. See `principles.md` for the canonical ownership rule.

## Exchange-Write Safety

Create/cancel retries must account for ambiguous outcomes: a timeout does not prove that the
exchange rejected the request. Correlation IDs, reconciliation, and authoritative confirmation
should establish the outcome before a retry could duplicate or reverse an exchange action.

Shutdown and task cancellation must not strand partially classified exchange writes. Exact
exchange-specific behavior belongs in `features/exchange_integrations.md` and focused tests.
