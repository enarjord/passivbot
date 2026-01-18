# Fill Events Manager Session Summary

**Date:** 2026-01-16
**Purpose:** Continue work on `src/fill_events_manager.py` and `src/tools/fill_events_dash.py`

---

## Session 2: Dashboard UX Improvements

### Issues Addressed
1. **No loading indicator** - Only browser tab showed "Updating..." during refresh
2. **UI freeze during fetch** - Dashboard became unresponsive during data refresh
3. **Fees too prominent** - Fee data is inconsistent/unavailable across exchanges
4. **Datetime sorting broken** - Dates displayed inconsistently and didn't sort correctly
5. **Cache health tab slow** - Loaded on every data refresh

### Fixes Implemented

#### 1. Background Refresh with Loading Overlay
- Added `_REFRESH_STATE` global dict for tracking refresh progress
- Added `_start_background_refresh()` to run refresh in `ThreadPoolExecutor`
- Added `_get_refresh_state()` and `_clear_refresh_result()` for thread-safe state access
- Added visible loading overlay with spinner and progress text
- Dashboard remains responsive during data fetch
- Refresh button disabled while refresh is running
- 500ms polling interval checks for completion

#### 2. Standardized Datetime Format
- Added `_format_datetime_str()` helper: formats as "YYYY-MM-DD HH:MM:SS"
- Added `datetime_str` column to DataFrame for display
- Table now sorts by `timestamp` (numeric) for correct chronological order
- Display uses `datetime_str` for human-readable format
- Consistent format across all exchanges (Binance, Bybit, etc.)

#### 3. De-emphasized Fees
- Removed `fee_cost` from main fills table columns
- Changed charts from "PnL with fees" to raw "PnL"
- Replaced "Fees by Account" chart with "Total PnL by Account"
- Removed fees from symbol stats badges
- Export still includes all data, but preview shows cleaner view

#### 4. Lazy-Loaded Cache Health Tab
- Health data now loaded only when Cache Health tab is clicked
- Removed `health-data` store from layout (no longer pre-loaded)
- Faster initial load and tab switching

---

## Session 1: Initial Dashboard Fixes

### 1. Updated Specification Document
- **File:** `docs/fill_events_manager_spec.md`
- Resolved open design questions (section 10):
  - Singleton vs instance-per-bot: Chose **instance-per-bot**
  - Position flip handling: Unpack to **two FillEvents** with identical timestamps
  - Raw field: Changed from `Dict` to `List[Dict]` for multiple exchange payloads
  - Rate limiting: Use **temp file coordination** + startup jitter
- Added sections on cache self-healing, gap detection, incremental disk flushing

### 2. Updated Fill Events Manager
- **File:** `src/fill_events_manager.py`
- Changed `FillEvent.raw` field type from `Dict` to `List[Dict]`
- Added `_normalize_raw_field()` helper for backward compatibility
- Added raw field population to ALL fetchers:
  - `BitgetFetcher._normalize_fill()`
  - `BinanceFetcher._normalize_trade()`
  - `BybitFetcher._normalize_trade()`
  - `GateioFetcher._normalize_order()`
  - `KucoinFetcher._normalize_trade()`
- Added `RateLimitCoordinator` class for cross-instance coordination
- Added `FillEventCache` metadata.json support with `known_gaps`
- Added `get_coverage_summary()` method

### 3. Created/Fixed Dashboard Tool
- **File:** `src/tools/fill_events_dash.py`
- Complete rewrite to fix multiple issues:

**Key fixes:**
1. **Persistent event loop** - Created `_get_or_create_event_loop()` running in background daemon thread. All async operations use `_run_async()`. This fixed "Event loop is closed" errors on subsequent refreshes.

2. **Manager rebuilding** - `_rebuild_manager()` creates fresh bot/fetcher instances before each refresh to avoid stale CCXT connections.

3. **Simplified UI** - Replaced confusing "N days" + "Refresh N days" with:
   - Quick select dropdown ("Last 7/14/30/60/90 days")
   - Single "Refresh" button

4. **Clean shutdown** - Uses `os._exit(0)` after 2-second timeout via `threading.Timer`

5. **Tab switching fix** - Only processes log-interval updates when on Console tab (prevents freezing)

6. **Data reload** - After refresh, explicitly calls `ensure_loaded()` to reload from disk

---

## Current State

### Working Features
- Dashboard starts and loads cached events from all accounts
- Quick select dropdown updates date range
- Refresh button fetches data from exchanges (parallel for multiple accounts)
- **Loading overlay with progress text during refresh**
- **UI remains responsive during data fetch (non-blocking)**
- **Standardized datetime format (YYYY-MM-DD HH:MM:SS)**
- **Proper datetime sorting in tables**
- **Cache Health tab loads lazily (faster switching)**
- Console tab shows live logs
- All tabs (Overview, Symbol Detail, Cache Health, Export, Console) work
- Ctrl+C exits cleanly within 2 seconds
- Multiple refreshes work without "Event loop is closed" error

### Test Results (Verified)
| Test | Result |
|------|--------|
| Startup & cache loading | ✅ Works |
| Data callback (initial load) | ✅ Returns correct events |
| First refresh | ✅ No errors |
| Second refresh | ✅ No "Event loop is closed" |
| Shutdown (Ctrl+C) | ✅ Clean exit |
| Loading overlay | ✅ Visible during refresh |
| UI responsiveness | ✅ Tab switching works during refresh |
| Datetime sorting | ✅ Sorts correctly |

---

## Exchanges Status

| Exchange | Fetcher | Status |
|----------|---------|--------|
| Binance | `BinanceFetcher` | ✅ Working |
| Bitget | `BitgetFetcher` | ✅ Working |
| Bybit | `BybitFetcher` | ✅ Working |
| Hyperliquid | `HyperliquidFetcher` | ✅ Working |
| Gate.io | `GateioFetcher` | ✅ Working |
| KuCoin | `KucoinFetcher` | ✅ Working |
| OKX | Not implemented | ⏳ TODO |
| Defx | Paused | ⏸️ Not fully supported |
| Paradex | Pending | ⏸️ Awaiting testing |

---

## Next Steps (From Spec)

### Phase 1: Complete Fetcher Coverage
1. Add `OkxFetcher` - port from `exchanges/okx.py:fetch_pnls`
2. Skip Defx/Paradex for now

### Phase 2: Add Missing Tests
3. Add `GateioFetcher` unit tests
4. Add `KucoinFetcher` unit tests

### Phase 3: Integration
5. Integrate `FillEventsManager` into `src/passivbot.py` (replace `update_pnls()`)
6. Integrate into `src/tools/pareto_dash.py`

### Phase 4: Cleanup
7. Deprecate legacy `fetch_pnls` functions
8. Remove legacy implementations after stable migration

---

## Key Files

| File | Purpose |
|------|---------|
| `src/fill_events_manager.py` | Core module: FillEvent, FillEventCache, fetchers, FillEventsManager |
| `src/tools/fill_events_dash.py` | Web dashboard for fill events inspection |
| `docs/fill_events_manager_spec.md` | Working specification document |
| `tests/test_fill_events_manager.py` | Unit tests (20 tests) |
| `tests/test_fill_events.py` | Integration tests (2 tests) |

---

## Running the Dashboard

```bash
# Basic usage
python3 src/tools/fill_events_dash.py --users binance_01,bybit_01,gateio_01

# With options
python3 src/tools/fill_events_dash.py \
  --users binance_01,bybit_01 \
  --config configs/template.json \
  --lookback-days 60 \
  --port 8050 \
  --log-level info
```

---

## Important Code Patterns

### Persistent Event Loop (key fix)
```python
_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None

def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    global _EVENT_LOOP, _LOOP_THREAD
    if _EVENT_LOOP is not None and _EVENT_LOOP.is_running():
        return _EVENT_LOOP
    _EVENT_LOOP = asyncio.new_event_loop()
    def run_loop():
        asyncio.set_event_loop(_EVENT_LOOP)
        _EVENT_LOOP.run_forever()
    _LOOP_THREAD = threading.Thread(target=run_loop, daemon=True)
    _LOOP_THREAD.start()
    time.sleep(0.1)
    return _EVENT_LOOP

def _run_async(coro):
    loop = _get_or_create_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=600)
```

### Raw Field Format
```python
raw: List[Dict[str, object]] = None
# Example:
raw = [
    {"source": "fetch_my_trades", "data": {...}},
    {"source": "position_history", "data": {...}},
]
```

---

## Session 3: Shadow Mode Integration in Live Bot

### Purpose
Safely integrate FillEventsManager into the live bot by running it in parallel with the legacy `update_pnls` system. This "shadow mode" allows:
- Data accumulation and caching without affecting bot decisions
- Comparison logging to validate the new system
- Easy flip to production once validated

### Implementation

#### Config Option
Added `pnls_manager_shadow_mode` to `config.live`:
- **Location**: `configs/template.json` line 225, `src/config_utils.py` line 2014
- **Default**: `false` (opt-in)
- **When true**: FillEventsManager runs alongside legacy, logs comparisons

#### Integration Points in `src/passivbot.py`

1. **Imports** (line 27-31):
   ```python
   from fill_events_manager import (
       FillEventsManager,
       _build_fetcher_for_bot,
       _extract_symbol_pool,
   )
   ```

2. **State Variables** (line 511-518):
   ```python
   self._pnls_shadow_mode = bool(get_optional_live_value(...))
   self._pnls_manager: Optional[FillEventsManager] = None
   self._pnls_shadow_initialized = False
   self._pnls_shadow_last_comparison_ts = 0
   self._pnls_shadow_comparison_interval_ms = 60_000
   ```

3. **Shadow Methods** (lines 2238-2423):
   - `_init_pnls_shadow_manager()`: Initialize FillEventsManager with bot's fetcher
   - `_update_pnls_shadow()`: Run `refresh_latest()` in parallel with legacy
   - `_compare_pnls_shadow()`: Compare event counts, PnL sums, IDs, timestamps

4. **Main Loop Integration** (line 1088-1119):
   - `update_pos_oos_pnls_ohlcvs()` now runs shadow update in parallel via `asyncio.gather`
   - Shadow errors don't fail the bot (just logged)
   - Comparison runs after both updates complete

### Comparison Logging

The shadow comparison logs:
- Event counts (legacy vs manager)
- PnL sums (legacy vs manager)
- IDs only in one system (first 5)
- Latest timestamp differences (if >1 minute)

Log levels:
- **DEBUG**: Normal comparisons
- **INFO**: Significant differences (>10 events or >1.0 PnL diff)
- **WARNING**: Shadow update failures

### Usage

To enable shadow mode, set in your config:
```json
{
  "live": {
    "pnls_manager_shadow_mode": true,
    ...
  }
}
```

Or via CLI argument if supported.

### Cache Locations

| System | Cache Path |
|--------|------------|
| Legacy pnls | `caches/{exchange}/{user}_pnls.json` |
| FillEventsManager | `caches/fill_events/{exchange}/{user}/` |

### Testing Plan

1. **Enable shadow mode** on a test account
2. **Run for 2-3 days** accumulating data
3. **Monitor logs** for `[shadow]` prefixed messages
4. **Compare results** - should see matching event counts and PnL sums
5. **Validate edge cases**: position flips, partial fills, rate limits
6. **Flip to production** when confident

---

## User Preferences Noted

1. Prefer simple UI - single "Refresh" button instead of multiple options
2. Date range + quick select dropdown is better than separate N days input
3. Console tab for watching progress is useful
4. Clean shutdown with Ctrl+C is important
5. Need visible feedback during long operations (loading overlay preferred)
6. Fees can be de-emphasized - data is inconsistent across exchanges
7. Datetime should be consistent and human-readable (YYYY-MM-DD HH:MM:SS)
