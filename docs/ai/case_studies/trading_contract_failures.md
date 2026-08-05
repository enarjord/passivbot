# Trading-Contract Review Failure Modes

These cases explain why the linked canonical rules exist. They are historical rationale, not
independent policy.

## Malformed Rust Intent Is Not A Retention Signal

Scenario:

1. Rust emits a valid ideal order and live reconciliation posts it.
2. A later Rust result contains a malformed ideal order.

A tempting recommendation is to keep running and deliberately retain the earlier exchange order.
That appears conservative, but it promotes historical Python or exchange state into strategy
authority. The live bot would then be trading intent Rust did not currently emit, behavior the
backtest does not reproduce and a restart may not recover.

The correct boundary is atomic validation. A complete, well-formed Rust result may add, change, or
remove ideal orders. A malformed result is fatal before reconciliation, so it authorizes neither
cancellation nor retention policy. Existing exchange state may remain untouched because the
process stops before exchange actions; it is not adopted as current Rust intent.

Regression anchors:

- `src/live/reconciler.py` (`validate_rust_ideal_orders`)
- `tests/test_order_churn_gate.py`

## Raw Candle Freshness Is Not Canonical Readiness

Backtests consume complete historical candles. Live trading operates while exchanges publish
candles late, omit no-trade buckets, rate-limit refreshes, or return partial data. Requiring every
candidate to have a newly arrived raw candle at each Rust call can rank only whichever symbols
happened to refresh first. That creates a live-only forager bias rather than improving parity.

Live readiness therefore evaluates the canonical input each consumer needs. Documented bounded
projection, verified no-trade continuity, or age-labeled carry-forward may keep that input usable
while authoritative repair continues. Candidates with no basis, non-finite values, or excessive
age remain unavailable. Protective actions depend only on their own required surfaces.

The parity target is the same trading and risk contract under the two runtime contexts, including
the documented live recovery model—not identical raw source arrival.

Regression anchors:

- `tests/test_missing_ema_fix.py`
- `tests/test_live_candle_budget.py`
- `tests/test_passivbot_monitor.py`
