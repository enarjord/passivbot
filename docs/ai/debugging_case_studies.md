# Debugging Case Studies (Condensed)

Use these patterns for non-trivial incident work.

## Case: Bybit Missing Closed-PnL Records (2026-01)

Signal:

- Some closes had `pnl: 0.0` despite expected realized PnL.

Root cause:

- Time-window pagination missed records in high-density windows.

Fix pattern:

1. Verify raw exchange endpoint first.
2. Compare cursor vs time-window coverage.
3. Implement hybrid pagination + deduplication.

Reference:

- `src/fill_events_manager.py` (`BybitFetcher._fetch_positions_history`)

## Case: Backtest Auto-Unstuck Allowance Drifted From Live (2026-04)

Signal:

- backtests emitted `close_unstuck_*` fills that exceeded a very small configured allowance
- live behavior and user intuition were both simpler: filter recent fill events by lookback, then recompute cumsum/max from that filtered list

Root cause:

- backtest optimized the rolling realized-PnL window with a separate rebased peak/current representation
- after the lookback window slid, the stored "peak" could fall below the current rolling PnL
- that invalid state inflated `calc_auto_unstuck_allowance()` and allowed oversized unstuck orders

Fix pattern:

1. reproduce with small deterministic Rust tests first
2. assert invariants, not just artifact-specific numbers
3. define the contract in naive live terms: filter fills in window, then recompute `cumsum.max()` and `cumsum[-1]`
4. make the optimized backtest path observationally identical to that naive reference
5. add parity tests so future optimizations cannot drift silently

## Reusable Investigation Loop

1. Confirm symptom with concrete examples.
2. Compare cached/internal data vs source-of-truth API.
3. Instrument pagination/window boundaries.
4. Identify exact data-loss mechanism.
5. Implement minimal fix.
6. Add regression test and validation script.
