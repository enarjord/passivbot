# Configurable Candle Interval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable coarser candle intervals (e.g., 5m) for faster optimizer iterations by aggregating 1m candles and adjusting Rust timestamp/EMA calculations.

**Architecture:** Python aggregates 1m→Nm candles before passing to Rust. Rust receives `candle_interval_minutes` in BacktestParams and adjusts all time-dependent calculations (timestamps, EMA alphas, hour bucket scans) accordingly.

**Tech Stack:** Python (numpy aggregation), Rust (backtest engine adjustments), PyO3 bindings

---

## Task 1: Add candle_interval_minutes to Rust BacktestParams

**Files:**
- Modify: `passivbot-rust/src/types.rs:157-174`

**Step 1: Add field to BacktestParams struct**

In `types.rs`, add `candle_interval_minutes` after `hedge_mode`:

```rust
#[derive(Clone, Debug)]
pub struct BacktestParams {
    pub starting_balance: f64,
    pub maker_fee: f64,
    pub coins: Vec<String>,
    pub active_coin_indices: Option<Vec<usize>>,
    pub first_timestamp_ms: u64,
    pub requested_start_timestamp_ms: u64,
    pub first_valid_indices: Vec<usize>,
    pub last_valid_indices: Vec<usize>,
    pub warmup_minutes: Vec<usize>,
    pub trade_start_indices: Vec<usize>,
    pub global_warmup_bars: usize,
    pub btc_collateral_cap: f64,
    pub btc_collateral_ltv_cap: Option<f64>,
    pub metrics_only: bool,
    pub filter_by_min_effective_cost: bool,
    pub hedge_mode: bool,
    pub candle_interval_minutes: u64,  // NEW: 1 for 1m candles (default), 5 for 5m, etc.
}
```

**Step 2: Verify it compiles**

Run: `cd passivbot-rust && cargo check 2>&1 | head -20`
Expected: Compilation errors about missing field in constructors (this is expected, we'll fix in next tasks)

**Step 3: Commit**

```bash
git add passivbot-rust/src/types.rs
git commit -m "feat(rust): add candle_interval_minutes to BacktestParams"
```

---

## Task 2: Parse candle_interval_minutes in PyO3 binding

**Files:**
- Modify: `passivbot-rust/src/python.rs:791-845`

**Step 1: Add extraction in backtest_params_from_dict**

After the `hedge_mode` extraction (around line 843), add:

```rust
        candle_interval_minutes: dict
            .get_item("candle_interval_minutes")?
            .map(|item| item.extract::<u64>())
            .transpose()?
            .unwrap_or(1),  // default to 1m candles
```

**Step 2: Verify it compiles**

Run: `cd passivbot-rust && cargo check`
Expected: PASS (or other unrelated errors from backtest.rs we'll fix next)

**Step 3: Commit**

```bash
git add passivbot-rust/src/python.rs
git commit -m "feat(rust): parse candle_interval_minutes from Python dict"
```

---

## Task 3: Add interval_ms to Backtest struct and constructor

**Files:**
- Modify: `passivbot-rust/src/backtest.rs:260-315` (struct definition)
- Modify: `passivbot-rust/src/backtest.rs:1380-1500` (constructor)

**Step 1: Add interval_ms field to Backtest struct**

Around line 265, add after `backtest_params`:

```rust
    interval_ms: u64,
```

**Step 2: Initialize interval_ms in Backtest::new**

In the constructor (around line 1410), compute and store interval_ms:

```rust
            interval_ms: backtest_params.candle_interval_minutes * 60_000,
```

**Step 3: Verify it compiles**

Run: `cd passivbot-rust && cargo check`
Expected: PASS

**Step 4: Commit**

```bash
git add passivbot-rust/src/backtest.rs
git commit -m "feat(rust): store interval_ms in Backtest struct"
```

---

## Task 4: Replace hardcoded 60_000 values with interval_ms

**Files:**
- Modify: `passivbot-rust/src/backtest.rs` (10 locations)

**Step 1: Replace all 10 occurrences**

| Line | Before | After |
|------|--------|-------|
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

**Step 2: Verify it compiles**

Run: `cd passivbot-rust && cargo check`
Expected: PASS

**Step 3: Verify no remaining 60_000 in backtest.rs**

Run: `grep -n "60.000" passivbot-rust/src/backtest.rs`
Expected: No output (all replaced)

**Step 4: Commit**

```bash
git add passivbot-rust/src/backtest.rs
git commit -m "refactor(rust): replace hardcoded 60_000 with interval_ms"
```

---

## Task 5: Adjust EMA alpha calculation for interval

**Files:**
- Modify: `passivbot-rust/src/backtest.rs:2741-2790` (calc_ema_alphas function)

**Step 1: Add interval parameter to calc_ema_alphas**

Change signature from:
```rust
fn calc_ema_alphas(bot_params_pair: &BotParamsPair) -> EmaAlphas {
```
to:
```rust
fn calc_ema_alphas(bot_params_pair: &BotParamsPair, interval: u64) -> EmaAlphas {
```

**Step 2: Adjust alpha calculations**

EMA spans are in minutes. Divide by interval to get number of candle periods:

```rust
    let interval_f = interval as f64;

    // Price EMAs - spans are in minutes, convert to candle periods
    let ema_alphas_long = ema_spans_long.map(|x| 2.0 / (x / interval_f + 1.0));
    let ema_alphas_short = ema_spans_short.map(|x| 2.0 / (x / interval_f + 1.0));
```

Apply same adjustment to volume and log_range alphas:
```rust
        vol_alpha_long: 2.0 / (bot_params_pair.long.filter_volume_ema_span as f64 / interval_f + 1.0),
        vol_alpha_short: 2.0 / (bot_params_pair.short.filter_volume_ema_span as f64 / interval_f + 1.0),
        log_range_alpha_long: 2.0 / (bot_params_pair.long.filter_volatility_ema_span as f64 / interval_f + 1.0),
        log_range_alpha_short: 2.0 / (bot_params_pair.short.filter_volatility_ema_span as f64 / interval_f + 1.0),
```

Note: `entry_volatility_logrange_ema_1h` spans are in hours, not minutes. These should NOT be adjusted since they're computed from hourly buckets, not per-candle.

**Step 3: Update call site in Backtest::new**

Around line 1387, change:
```rust
let ema_alphas: Vec<EmaAlphas> = bot_params.iter().map(|bp| calc_ema_alphas(bp)).collect();
```
to:
```rust
let interval = backtest_params.candle_interval_minutes;
let ema_alphas: Vec<EmaAlphas> = bot_params.iter().map(|bp| calc_ema_alphas(bp, interval)).collect();
```

**Step 4: Verify it compiles**

Run: `cd passivbot-rust && cargo check`
Expected: PASS

**Step 5: Commit**

```bash
git add passivbot-rust/src/backtest.rs
git commit -m "feat(rust): adjust EMA alphas for candle interval"
```

---

## Task 6: Build and verify Rust changes

**Files:** None (verification only)

**Step 1: Build release**

Run: `cd passivbot-rust && maturin develop --release 2>&1 | tail -5`
Expected: "Installed passivbot_rust-0.1.0"

**Step 2: Run existing tests**

Run: `pytest tests/ --tb=no -q 2>&1 | tail -3`
Expected: Same pass/fail count as baseline (619 passed, 3 failed)

**Step 3: Commit (if any cleanup needed)**

No commit needed if tests pass unchanged.

---

## Task 7: Add aggregate_candles function in Python

**Files:**
- Modify: `src/backtest.py` (add function near top, after imports)

**Step 1: Add aggregation function**

After the imports section (around line 95), add:

```python
def aggregate_candles(candles_1m: np.ndarray, interval: int) -> np.ndarray:
    """
    Aggregate 1m OHLCV candles to coarser interval.

    Args:
        candles_1m: Array of shape (n_timesteps, n_coins, 5) for OHLCV
        interval: Number of 1m candles to combine (e.g., 5 for 5m candles)

    Returns:
        Aggregated array of shape (n_timesteps // interval, n_coins, 5)
    """
    if interval <= 1:
        return candles_1m
    n_timesteps = candles_1m.shape[0]
    n_out = n_timesteps // interval
    if n_out == 0:
        raise ValueError(f"Not enough candles ({n_timesteps}) for interval {interval}")
    truncated = candles_1m[: n_out * interval]
    reshaped = truncated.reshape(n_out, interval, *candles_1m.shape[1:])
    # OHLCV indices: 0=open, 1=high, 2=low, 3=close, 4=volume
    aggregated = np.stack(
        [
            reshaped[:, 0, :, 0],           # open: first candle's open
            reshaped[:, :, :, 1].max(axis=1),  # high: max across interval
            reshaped[:, :, :, 2].min(axis=1),  # low: min across interval
            reshaped[:, -1, :, 3],          # close: last candle's close
            reshaped[:, :, :, 4].sum(axis=1),  # volume: sum across interval
        ],
        axis=-1,
    )
    return aggregated
```

**Step 2: Verify syntax**

Run: `python -c "from src.backtest import aggregate_candles; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/backtest.py
git commit -m "feat(python): add aggregate_candles function"
```

---

## Task 8: Integrate aggregation into build_backtest_payload

**Files:**
- Modify: `src/backtest.py:279-370` (build_backtest_payload function)

**Step 1: Read interval from config**

At the start of build_backtest_payload (after line 292), add:

```python
    candle_interval = config.get("backtest", {}).get("candle_interval_minutes", 1)
    if candle_interval < 1:
        raise ValueError(f"candle_interval_minutes must be >= 1, got {candle_interval}")
```

**Step 2: Aggregate candles if interval > 1**

After preparing hlcvs but before building the bundle, add:

```python
    # Aggregate candles if using coarser interval
    if candle_interval > 1:
        hlcvs = aggregate_candles(hlcvs, candle_interval)
        # Also aggregate timestamps and btc_usd_prices
        n_out = hlcvs.shape[0]
        if timestamps is not None:
            timestamps = timestamps[::candle_interval][:n_out]
        if btc_usd_prices is not None:
            # Use last price in each interval (matches close)
            btc_usd_prices = btc_usd_prices[candle_interval - 1 :: candle_interval][:n_out]
        # Adjust warmup and trade start indices
        for i in range(len(first_valid_indices)):
            first_valid_indices[i] = first_valid_indices[i] // candle_interval
            last_valid_indices[i] = last_valid_indices[i] // candle_interval
            trade_start_indices[i] = trade_start_indices[i] // candle_interval
            warmup_minutes[i] = warmup_minutes[i]  # Keep in minutes (Rust will adjust)
        backtest_params["global_warmup_bars"] = backtest_params["global_warmup_bars"] // candle_interval
```

**Step 3: Pass interval to backtest_params**

Before returning, add:

```python
    backtest_params["candle_interval_minutes"] = candle_interval
```

**Step 4: Adjust first_timestamp_ms**

Update the first_ts_ms calculation to account for aligned timestamps:

```python
    # Align first timestamp to interval boundary
    if candle_interval > 1 and first_ts_ms > 0:
        interval_ms = candle_interval * 60_000
        first_ts_ms = (first_ts_ms // interval_ms) * interval_ms
```

**Step 5: Verify syntax**

Run: `python -c "from src.backtest import build_backtest_payload; print('OK')"`
Expected: "OK"

**Step 6: Commit**

```bash
git add src/backtest.py
git commit -m "feat(python): integrate candle aggregation into build_backtest_payload"
```

---

## Task 9: Add config field to template.json

**Files:**
- Modify: `configs/template.json`

**Step 1: Add candle_interval_minutes field**

In the backtest section (after line 11 "filter_by_min_effective_cost"), add:

```json
        "candle_interval_minutes": 1,
```

**Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('configs/template.json')); print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add configs/template.json
git commit -m "docs: add candle_interval_minutes to template config"
```

---

## Task 10: Write integration test

**Files:**
- Create: `tests/test_candle_interval.py`

**Step 1: Write test file**

```python
"""Tests for configurable candle interval feature."""
import numpy as np
import pytest


def test_aggregate_candles_basic():
    """Test that aggregate_candles produces correct OHLCV values."""
    from src.backtest import aggregate_candles

    # Create 10 1m candles for 2 coins
    # Shape: (10, 2, 5) for OHLCV
    candles = np.zeros((10, 2, 5), dtype=np.float64)

    # Coin 0: prices 100-109, volume 1 each
    for i in range(10):
        candles[i, 0, :] = [100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i + 0.1, 1.0]

    # Coin 1: prices 200-209, volume 2 each
    for i in range(10):
        candles[i, 1, :] = [200 + i, 200 + i + 1.0, 200 + i - 1.0, 200 + i + 0.2, 2.0]

    result = aggregate_candles(candles, 5)

    assert result.shape == (2, 2, 5), f"Expected (2, 2, 5), got {result.shape}"

    # First 5m candle for coin 0 (indices 0-4):
    # open=100, high=max(100.5,101.5,102.5,103.5,104.5)=104.5, low=min(99.5,...,103.5)=99.5
    # close=104.1, volume=5
    assert result[0, 0, 0] == 100.0, "Open should be first candle's open"
    assert result[0, 0, 1] == 104.5, "High should be max of interval"
    assert result[0, 0, 2] == 99.5, "Low should be min of interval"
    assert abs(result[0, 0, 3] - 104.1) < 0.01, "Close should be last candle's close"
    assert result[0, 0, 4] == 5.0, "Volume should be sum"


def test_aggregate_candles_interval_1():
    """Test that interval=1 returns unchanged array."""
    from src.backtest import aggregate_candles

    candles = np.random.rand(100, 3, 5)
    result = aggregate_candles(candles, 1)

    assert result is candles, "interval=1 should return same array"


def test_aggregate_candles_truncates():
    """Test that incomplete final interval is dropped."""
    from src.backtest import aggregate_candles

    candles = np.random.rand(17, 2, 5)  # 17 candles, interval 5 -> 3 complete intervals
    result = aggregate_candles(candles, 5)

    assert result.shape[0] == 3, f"Expected 3 intervals, got {result.shape[0]}"


def test_aggregate_candles_error_on_insufficient():
    """Test that error is raised when not enough candles."""
    from src.backtest import aggregate_candles

    candles = np.random.rand(3, 2, 5)  # Only 3 candles

    with pytest.raises(ValueError, match="Not enough candles"):
        aggregate_candles(candles, 5)
```

**Step 2: Run test**

Run: `pytest tests/test_candle_interval.py -v`
Expected: 4 passed

**Step 3: Commit**

```bash
git add tests/test_candle_interval.py
git commit -m "test: add unit tests for candle aggregation"
```

---

## Task 11: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Build Rust with changes**

Run: `unset CONDA_PREFIX && source /home/fredb/passivbot/venv/bin/activate && cd passivbot-rust && maturin develop --release 2>&1 | tail -3`
Expected: "Installed passivbot_rust-0.1.0"

**Step 2: Run full test suite**

Run: `pytest tests/ --tb=short -q 2>&1 | tail -10`
Expected: Similar pass count to baseline, no new failures

**Step 3: Manual smoke test with 5m interval**

Create a test config with `"candle_interval_minutes": 5` and run a quick backtest to verify it works end-to-end.

---

## Task 12: Update design doc with implementation notes

**Files:**
- Modify: `docs/plans/2026-01-31-configurable-candle-interval-design.md`

**Step 1: Add "Implementation Complete" section**

At the end of the design doc, add:

```markdown
## Implementation Notes

Completed 2026-01-31.

Key files changed:
- `passivbot-rust/src/types.rs` - Added candle_interval_minutes field
- `passivbot-rust/src/python.rs` - Parse from Python dict
- `passivbot-rust/src/backtest.rs` - Store interval_ms, replace 60_000, adjust EMA alphas
- `src/backtest.py` - aggregate_candles function, integration in build_backtest_payload
- `configs/template.json` - Document new config field
- `tests/test_candle_interval.py` - Unit tests
```

**Step 2: Commit**

```bash
git add docs/plans/2026-01-31-configurable-candle-interval-design.md
git commit -m "docs: mark candle interval implementation complete"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add field to BacktestParams | types.rs |
| 2 | Parse in PyO3 binding | python.rs |
| 3 | Add interval_ms to Backtest struct | backtest.rs |
| 4 | Replace hardcoded 60_000 | backtest.rs |
| 5 | Adjust EMA alpha calculation | backtest.rs |
| 6 | Build and verify Rust | (verification) |
| 7 | Add aggregate_candles function | backtest.py |
| 8 | Integrate into build_backtest_payload | backtest.py |
| 9 | Add config field | template.json |
| 10 | Write integration test | test_candle_interval.py |
| 11 | Run full test suite | (verification) |
| 12 | Update design doc | design.md |
