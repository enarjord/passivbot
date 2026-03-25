# Large Account Execution Strategies

Analysis of order execution improvements for accounts where order size is a meaningful fraction of book depth.

## Current State

Rust (`entries.rs`) computes ideal `(qty, price)` pairs. Python reconciles ideal vs actual, submits as limit orders (GTC/post-only), promotes to market when price crosses spread. No awareness of book depth, fill rate, or market impact.

Key gap: `StateParams` only has top-of-book `bid`/`ask`. No depth data reaches the order calculation layer.

## Problems at Scale

- **Partial fills**: limit order sits, market moves, 30% filled, rest hangs
- **Adverse selection**: fills you get are where market moves against you
- **Information leakage**: large resting orders signal intent to front-runners
- **Slippage**: market orders eat through multiple book levels

## Approaches (ranked by feasibility for passivbot)

### Tier 1: Low effort, high impact

**A. Order Book Depth Awareness**
Check liquidity at target price before submitting. Split if order > X% of visible depth.
- Rust: add depth to `StateParams` (minor type change)
- Python: fetch depth snapshots

**B. Self-Managed Iceberg**
Cap visible order size at fraction of book depth or recent volume. Re-post on fill.
- Python-only: throttle execution of Rust's output in reconciliation layer
- No Rust changes

**C. Fill-Aware Reconciliation**
Track partial fills between cycles. Complete a partially-filled entry instead of cancel/replace.
- Python-only: state in reconciliation layer
- Related to existing `EntryInitialPartialLong/Short` types (position-based, not order-based)

### Tier 2: Medium effort

**D. Execution Scheduling Layer**
New Python module between `calc_orders_to_cancel_and_create()` and `execute_orders_parent()`.
Configurable policies: `immediate` (current), `twap`, `iceberg`, `adaptive`.

**E. Exchange-Side Trigger Orders**
Convert trailing entries to exchange-native stop-limit/trailing-stop. Survives disconnections, lower latency. Loses custom threshold+retracement logic from `calc_trailing_entry_*`.

### Tier 3: Ambitious

**F. Impact-Aware Qty in Rust**
Pass order book depth into `entries.rs`. Modify `calc_reentry_qty` / `calc_initial_entry_qty` to cap quantities at X% of book depth. Requires backtester changes to simulate impact.

## Institutional Reference

| Algorithm | Core Idea | Passivbot Relevance |
|-----------|-----------|-------------------|
| TWAP | Equal slices at fixed intervals | Bot loop is natural TWAP cadence; needs slice tracking |
| VWAP | Trade proportional to volume | Needs real-time volume (not currently available) |
| Iceberg | Hide true size, replenish on fill | Implementable in Python layer today |
| Almgren-Chriss | Optimize impact vs timing risk | Concept useful, full model overkill for crypto |
| Post-only loop | Maker-only with retry on reject | Partially supported via `time_in_force: post_only` |
| Peg / auto-reprice | Stay at top of book | Reconciliation loop does this, but tolerance suppresses small moves |

## Design Constraint

Passivbot's stateless principle ("identical after restart") conflicts with execution state (slice counts, iceberg progress). Resolution: keep Rust stateless (computes *what*), add stateful execution layer in Python (decides *how/when*). On restart, re-derive state from exchange positions + open orders.

## Priority

Biggest bang-for-buck: **iceberg + depth awareness**. Both address partial fills directly, iceberg needs zero Rust changes, depth awareness needs only minor type additions.
