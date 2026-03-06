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

## Reusable Investigation Loop

1. Confirm symptom with concrete examples.
2. Compare cached/internal data vs source-of-truth API.
3. Instrument pagination/window boundaries.
4. Identify exact data-loss mechanism.
5. Implement minimal fix.
6. Add regression test and validation script.
