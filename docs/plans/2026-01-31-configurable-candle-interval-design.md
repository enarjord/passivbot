# Configurable Candle Interval for Faster Backtesting

**Date:** 2026-01-31
**Goal:** Speed up optimizer iterations by using coarser candle intervals (e.g., 5m instead of 1m)

## Motivation

Running 100K+ NSGA-II iterations requires fast individual backtests. A 365-day backtest currently processes ~525,600 candles per coin (1m resolution). Using 5m candles reduces this to ~105,120 — roughly 5x fewer iterations through the main loop.

**Trade-off accepted:** Fill sequence accuracy within each candle window is lost. For optimization purposes, this is acceptable — we care about overall strategy viability, not precise intra-minute timing.

## Design Decisions

1. **Rust handles interval-awareness** — Config spans remain in minutes; Rust adjusts alphas and timestamps internally
2. **Aggregate 1m → Nm in Python** — Uses existing 1m cache, no new download infrastructure
3. **Generic interval support** — `candle_interval_minutes` field (default 1), works for any value
4. **Aggregate once per optimization** — Not per iteration; aggregation happens before shared-memory dataset creation

## Configuration

New field in `config.backtest`:

```json
{
  "backtest": {
    "candle_interval_minutes": 1
  }
}
```

- Default: `1` (current behavior)
- Set to `5` for ~5x faster backtests
- Any positive integer that divides evenly into data length is valid

## Python Changes

### Aggregation function

```python
    def aggregate_candles(candles_1m, interval):
        """Aggregate 1m candles to coarser interval.

        Args:
            candles_1m: Array of shape (n_timesteps, n_coins, 4) for HLCV
            interval: Number of 1m candles to combine

        Returns:
            Aggregated array of shape (n_timesteps // interval, n_coins, 4)
        """
        if interval == 1:
            return candles_1m
        n = len(candles_1m)
        n_out = n // interval
        truncated = candles_1m[:n_out * interval]
        reshaped = truncated.reshape(n_out, interval, *candles_1m.shape[1:])
        return np.stack([
            reshaped[..., 0].max(axis=1),     # high: max of interval
            reshaped[..., 1].min(axis=1),     # low: min of interval
            reshaped[:, -1, ..., 2],          # close: last candle's close
            reshaped[..., 3].sum(axis=1),     # volume: sum of interval
        ], axis=-1)
```

### Integration points

**backtest.py:**
```python
interval = config["backtest"].get("candle_interval_minutes", 1)
if interval > 1:
    hlcvs = aggregate_candles(hlcvs, interval)
    start_ts = (start_ts // (interval * 60_000)) * (interval * 60_000)

# Pass to Rust
result = pbr.backtest_multi_coin(
    hlcvs,
    ...,
    candle_interval_minutes=interval,
)
```

**optimize.py:**
- Same logic, but aggregation happens once before creating shared-memory dataset
- All iterations use the pre-aggregated data

## Rust Changes

### BacktestParams

Add field:
```rust
pub candle_interval_minutes: u64,  // default 1
```

### Backtest struct

Store computed interval in milliseconds:
```rust
struct Backtest {
    interval_ms: u64,  // candle_interval_minutes * 60_000
    // ...
}
```

Initialize in constructor:
```rust
interval_ms: backtest_params.candle_interval_minutes * 60_000,
```

### EMA alpha calculation

Adjust `calc_ema_alphas()` to account for interval:
```rust
fn calc_ema_alphas(bot_params_pair: &BotParamsPair, interval: u64) -> EmaAlphas {
    let interval_f = interval as f64;

    // Spans are in minutes; divide by interval to get number of candles
    let ema_alphas_long = ema_spans_long.map(|x| 2.0 / (x / interval_f + 1.0));
    let ema_alphas_short = ema_spans_short.map(|x| 2.0 / (x / interval_f + 1.0));

    // Same adjustment for volume, log_range, and volatility alphas
    // ...
}
```

### Hardcoded 60_000 replacements

All 10 occurrences in `backtest.rs` must use `self.interval_ms`:

| Line | Current | Replace with |
|------|---------|--------------|
| 735 | `(k as u64) * 60_000` | `(k as u64) * self.interval_ms` |
| 1549 | `(k as u64) * 60_000u64` | `(k as u64) * self.interval_ms` |
| 1833 | `(k as u64) * 60_000` | `(k as u64) * self.interval_ms` |
| 2013 | `(k as u64) * 60_000` | `(k as u64) * self.interval_ms` |
| 2096 | `(k as u64) * 60_000` | `(k as u64) * self.interval_ms` |
| 2167 | `(k as u64) * 60_000` | `(k as u64) * self.interval_ms` |
| 2250 | `(k as u64) * 60_000` | `(k as u64) * self.interval_ms` |
| 2551 | `(k as u64) * 60_000u64` | `(k as u64) * self.interval_ms` |
| 2556 | `window_start_ms + 60_000` | `window_start_ms + self.interval_ms` |
| 2557 | `/ 60_000u64` | `/ self.interval_ms` |

### PyO3 binding

Update `python.rs` to accept `candle_interval_minutes` parameter and pass to `BacktestParams`.

## Files to Modify

| File | Changes |
|------|---------|
| `src/backtest.py` | Add aggregation, pass interval to Rust |
| `src/optimize.py` | Add aggregation before shared-memory setup |
| `passivbot-rust/src/types.rs` | Add `candle_interval_minutes` to `BacktestParams` |
| `passivbot-rust/src/backtest.rs` | Add `interval_ms`, update 10 hardcoded values, adjust alpha calc |
| `passivbot-rust/src/python.rs` | Expose new param in PyO3 binding |
| `configs/template.json` | Document new field |

## Not in Scope

- Live bot changes (always uses 1m from exchange)
- Downloading coarser candles directly (uses aggregation instead)
- Validation phase (re-running top candidates at 1m)

## Testing

1. Run backtest with `candle_interval_minutes: 1` — verify identical results to current behavior
2. Run backtest with `candle_interval_minutes: 5` — verify ~5x speedup, reasonable results
3. Run optimizer with `candle_interval_minutes: 5` — verify full pipeline works
4. Compare a few optimized configs between 1m and 5m validation runs to sanity-check correlation

## Implementation Notes

Completed 2026-02-01.

Key files changed:
- `passivbot-rust/src/types.rs` - Added `candle_interval_minutes` field to `BacktestParams`
- `passivbot-rust/src/python.rs` - Parse `candle_interval_minutes` from Python dict (default 1)
- `passivbot-rust/src/backtest.rs` - Store `interval_ms`, replace 10 hardcoded `60_000` values, adjust EMA alphas
- `src/backtest.py` - Added `aggregate_candles()` function, integrated into `build_backtest_payload()`
- `configs/template.json` - Document new `candle_interval_minutes` config field
- `tests/test_candle_interval.py` - Unit tests for candle aggregation
