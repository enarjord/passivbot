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

## Case: Hyperliquid Higher-TF To 1m Synthesis (2026-03)

Signal:

- Hyperliquid historical fetches are capped at 5000 candles per timeframe.
- Hard-stop equity replay needed minute-level prices beyond native `1m` reach.
- An initial local fix only reconstructed minute closes inside one caller.

Root cause:

- The real need was not "patch this replay path."
- The real need was a reusable higher-timeframe-to-`1m` OHLCV transform.

Correct pattern:

1. When a new data transform is broadly useful, extract it immediately into a pure reusable helper instead of leaving it embedded in one caller.
2. Prefer a clean boundary like `ohlcv_{x}m_to_1m()` over ad hoc inlined logic.
3. Preserve invariant information exactly:
   - child timestamps contiguous at `1m`
   - child closes stay within parent `[low, high]`
   - child volumes sum exactly to parent volume
4. Keep synthetic fallback separate from real cached `1m` data unless persistence is explicitly intended.
5. If the transform may become performance-critical or must match Rust/backtest semantics, consider implementing it in Rust and exposing it through `passivbot_rust`.

Implementation lesson:

- Start with a pure function in a shared module.
- Use caller-specific wrappers only for orchestration and caching.
- Do not hide a reusable market-data primitive inside one risk or replay function.

Reference:

- `src/candlestick_manager.py`
- `src/passivbot.py`

## Case: Runtime Risk Knob vs Optimizer Variable (2026-03)

Signal:

- A runtime safety control (`no_restart_drawdown_threshold`) was technically optimizable.
- Optimizer pressure from `backtest_completion_ratio` made the "best" move to simply raise the threshold and avoid HSL intervention.

Root cause:

- Runtime/operator pain thresholds and strategy quality metrics are not the same class of variable.
- Keeping them in the same optimization space let the optimizer game survivability instead of improving the strategy.

Correct pattern:

1. Keep runtime/operator controls in the runtime config where they belong.
2. If optimizer should ignore a runtime control, do not force the user to mutate their live value.
3. Add optimize-time runtime overrides instead of overloading bounds.
4. Constrain the optimizer with the semantically correct metrics instead:
   - `drawdown_worst_hsl`
   - `drawdown_worst_mean_1pct_hsl`
   - `peak_recovery_hours_hsl`
5. Keep "fixed optimization params" separate from "optimize-time runtime overrides."

Reference:

- `src/optimize.py`
- `src/config_utils.py`

## Reusable Investigation Loop

1. Confirm symptom with concrete examples.
2. Compare cached/internal data vs source-of-truth API.
3. Instrument pagination/window boundaries.
4. Identify exact data-loss mechanism.
5. Implement minimal fix.
6. Add regression test and validation script.
