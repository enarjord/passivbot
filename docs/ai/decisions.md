# Architectural Decisions (Active)

Only durable decisions that still govern implementation.

## 2025-Q4: Rust Owns Trading Behavior

Decision: order/risk/unstuck behavior is implemented in Rust and shared by live/backtest.

Impact: behavior changes must land in `passivbot-rust/src/`.

## 2025-Q3: Stateless Trading Behavior

Decision: bot decisions must be reproducible after restart from exchange state + config.

Impact: no runtime-decision state that cannot be rederived.

## 2026-01: Exchange Fetch Methods Must Propagate Exceptions

Decision: exchange fetch methods do not catch and downgrade exceptions.

Impact: clear return types and preserved error context.

## 2026-01: Bybit Closed-PnL Uses Hybrid Pagination

Decision: cursor + time-window pagination with deduplication.

Impact: improved fill/PnL completeness over single-strategy pagination.

## Note

Historical/deep decision context remains in git history; keep this file short and current.
