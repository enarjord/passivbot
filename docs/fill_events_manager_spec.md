# Fill Events Manager Specification

**Status:** Draft
**Module:** `src/fill_events_manager.py`
**Goal:** Replace exchange-specific `fetch_pnls` functions with a unified `FillEventsManager`

---

## 1. Overview

The `fill_events_manager.py` module provides a unified interface for fetching, caching, and querying historical fill events across all supported exchanges. It aims to replace the various `fetch_pnls` implementations scattered across `src/exchanges/*.py` with a single, well-tested, and consistent implementation.

### Key Components

| Component | Description |
|-----------|-------------|
| `FillEvent` | Canonical dataclass representing a single fill event |
| `FillEventCache` | JSON-based cache storing fills split by UTC day |
| `BaseFetcher` | Abstract interface for exchange-specific fetchers |
| `FillEventsManager` | High-level orchestrator tying fetcher + cache together |

---

## 2. Canonical Fill Event Schema

All fill events are normalized to this schema:

```python
@dataclass(frozen=True)
class FillEvent:
    id: str                    # Unique trade/fill ID from exchange
    timestamp: int             # Unix timestamp in milliseconds
    datetime: str              # ISO 8601 datetime string
    symbol: str                # CCXT-format symbol (e.g., "BTC/USDT:USDT")
    side: str                  # "buy" or "sell" (lowercase)
    qty: float                 # Signed quantity (buy=+, sell=-)
    price: float               # Execution price
    pnl: float                 # Realized PnL for this fill
    fees: Optional[Sequence]   # Fee structure (currency, cost)
    pb_order_type: str         # Passivbot order type (e.g., "entry_grid_normal_long")
    position_side: str         # "long" or "short" (lowercase)
    client_order_id: str       # Client order ID (passivbot custom ID)
    psize: float = 0.0         # Position size after this fill (computed)
    pprice: float = 0.0        # Position VWAP after this fill (computed)
    raw: List[Dict] = None      # Original exchange payloads (multiple sources)
```

> **Note**: The `raw` field is `List[Dict]` because many exchanges require multiple API calls to construct a single FillEvent. For example:
> - One `fetch_my_trades` call for execution details
> - One position history call for realized PnL
> - One `fetch_order` call for clientOrderId
>
> Each raw payload should include a `source` key identifying its origin.

### Required Fields

All fields except `psize`, `pprice`, `fees`, and `raw` are required when constructing a `FillEvent`. The dataclass will raise `ValueError` if any required field is missing.

### Sign Conventions

| Field | Long Position | Short Position |
|-------|---------------|----------------|
| `qty` | Buy = +, Sell = - | Buy = +, Sell = - |
| `position_side` | "long" | "short" |
| `psize` | after fill, long side + signed | after fill, short side - signed |
| `side` | "buy" opens, "sell" closes | "sell" opens, "buy" closes |

---

## 3. Exchange Fetcher Status

### Implemented Fetchers (in `fill_events_manager.py`)

| Exchange | Fetcher Class | Status | Notes |
|----------|---------------|--------|-------|
| Binance | `BinanceFetcher` | Working | Merges income + trades endpoints |
| Bitget | `BitgetFetcher` | Working | Uses fill history + order detail enrichment |
| Bybit | `BybitFetcher` | Working | Combines trades + positions history |
| Hyperliquid | `HyperliquidFetcher` | Working | Uses fetch_my_trades with coalescing |
| Gate.io | `GateioFetcher` | Working | Uses fetch_closed_orders |
| KuCoin | `KucoinFetcher` | Working | Trades + positions history, local PnL computation |

### Missing Fetchers (still using legacy `fetch_pnls`)

| Exchange | Current Implementation | Priority | Notes |
|----------|------------------------|----------|-------|
| OKX | `exchanges/okx.py:fetch_pnls` | Medium | Uses `fetch_my_trades` with `fillPnl` field |
| Defx | `exchanges/defx.py:fetch_pnls` | Low | Minimal implementation, TODO for time filters |

---

## 4. Current Test Coverage

**Test files:**
- `tests/test_fill_events_manager.py` - 20 tests
- `tests/test_fill_events.py` - 2 tests (Passivbot integration)
- `tests/test_fill_events_hyperliquid_mapping.py` - 1 test

**All 23 tests pass.**

### Covered Scenarios

| Category | Coverage |
|----------|----------|
| FillEvent construction/validation | Yes |
| FillEventCache round-trip | Yes |
| BitgetFetcher pagination & enrichment | Yes |
| BinanceFetcher income/trade merging | Yes |
| BybitFetcher PnL distribution | Yes |
| HyperliquidFetcher coalescing | Yes |
| Manager refresh/refresh_latest/refresh_range | Yes |
| Gap detection | Yes |
| Detail cache usage | Yes |

### Not Yet Covered

- `GateioFetcher` unit tests
- `KucoinFetcher` unit tests
- Live integration tests
- Error recovery scenarios
- Rate limit handling edge cases

---

## 5. Architecture

### Data Flow

```
Exchange API
     |
     v
[ExchangeFetcher]  -- normalize --> List[Dict]
     |
     v
[FillEventsManager.refresh()]
     |
     +-- ensure_qty_signage() --> fix +/- signs
     +-- FillEvent.from_dict() --> validate & wrap
     +-- annotate_positions_inplace() --> compute psize/pprice
     |
     v
[FillEventCache]  -- save_days() --> {date}.json files
     |
     v
[Query APIs]  --> get_events(), get_pnl_sum(), etc.
```

### Cache Structure

```
caches/fill_events/{exchange}/{user}/
  2024-12-28.json
  2024-12-29.json
  2024-12-30.json
  ...
```

Each JSON file contains an array of `FillEvent.to_dict()` payloads for that UTC day.

---

## 6. CLI Usage

The module includes a CLI for manual cache refresh:

```bash
python src/fill_events_manager.py \
  --user myaccount \
  --config configs/myconfig.json \
  --start 2024-12-01 \
  --end 2024-12-31 \
  --log-level debug
```

---

## 7. Integration Points

### Current Integration

The `_build_fetcher_for_bot(bot, symbols)` function creates the appropriate fetcher based on `bot.exchange`. This is used by the CLI.

### Planned Integration

Replace usage of `bot.fetch_pnls()` in:
- `src/passivbot.py` - PnL tracking for live dashboard
- `src/tools/pareto_dash.py` - Historical analysis

---

## 8. Next Steps

### Phase 1: Complete Fetcher Coverage

1. **Add `OkxFetcher`**
   - Port logic from `exchanges/okx.py:fetch_pnls`
   - Use `fetch_my_trades` with `fillPnl` extraction
   - Add unit tests

2. **Add `DefxFetcher`** (if Defx support continues)
   - Port logic from `exchanges/defx.py:fetch_pnls`
   - Implement proper time filtering
   - Add unit tests

### Phase 2: Add Missing Tests

3. **Add `GateioFetcher` tests**
   - Pagination through `fetch_closed_orders`
   - Position side detection

4. **Add `KucoinFetcher` tests**
   - Trade + position history merging
   - Local PnL computation

### Phase 3: Integration

5. **Integrate into `Passivbot`**
   - Replace `fetch_pnls` calls with `FillEventsManager`
   - Use cached fills for dashboard PnL display
   - Background refresh task

6. **Integrate into `pareto_dash.py`**
   - Replace direct `fetch_pnls` usage
   - Leverage persistent cache for faster startup

### Phase 4: Cleanup

7. **Deprecate legacy `fetch_pnls`**
   - Mark as deprecated in docstrings
   - Add warnings when called directly

8. **Remove legacy implementations**
   - After migration is stable for 1+ releases

---

## 9. Exchange-Specific Notes

### Binance
- Income endpoint provides PnL without qty/price
- Trade endpoint provides qty/price without PnL
- Must merge by trade ID
- 7-day pagination windows

### Bitget
- Requires order detail lookup for `clientOid`
- Rate limit: ~120 detail calls/minute
- Concurrent detail fetching (configurable)

### Bybit
- Trades don't include PnL directly
- Positions history provides per-order PnL
- Must distribute PnL proportionally across fills

### Hyperliquid
- Same-timestamp fills must be coalesced
- Direction field determines position side
- `closedPnl` in info object

### Gate.io
- Uses `fetch_closed_orders` not trades
- `pnl` and `pnl_margin` fields
- Offset-based pagination

### KuCoin
- No direct PnL on trades
- Must use positions history + local computation
- Time-window-based pagination (24h chunks)

### OKX (TODO)
- PnL in `fillPnl` field of info
- Position side in `posSide` field
- Simple structure but needs wrapper

### Defx (TODO)
- Basic `fetch_my_trades` with `pnl` in info
- Position side inferred from pnl being zero

---

## 10. Design Decisions (Resolved)

### 10.1 Singleton vs Instance-per-Bot

**Decision:** Use **instance-per-bot** approach (non-singleton).

| Aspect | Singleton | Instance-per-Bot |
|--------|-----------|------------------|
| Cache coordination | Single cache dir, centralized | Separate cache per user |
| Rate limiting | Internal coordination | Requires external coordination |
| Lifecycle | Complex global state | Tied to bot lifecycle |
| Testing | Harder (global state) | Easier (isolated instances) |
| Memory | Shared across bots | Slightly more per bot |
| Corruption risk | Higher (concurrent writes) | Lower (isolated) |

**Rationale:** Instance-per-bot aligns with passivbot's stateless design principles. Rate limiting is handled via shared temp file coordination (see 10.4).

### 10.2 Position Flip Handling

**Decision:** Unpack position flips into **two separate FillEvents** with identical timestamps.

**Example:** Current position is 4@4.15 (long), bot fills sell order of -5, resulting position is -1@4.16 (short).

This single exchange fill becomes two FillEvents:
```python
[
    FillEvent(
        side="sell", qty=-4, position_side="long",
        pnl=0.04, timestamp=1234, # Close long
    ),
    FillEvent(
        side="sell", qty=-1, position_side="short",
        pnl=0.0, timestamp=1234,  # Open short
    ),
]
```

**Rationale:** Clean position accounting, PnL attribution is clear, analysis tools can treat each position side independently.

### 10.3 Raw Field Storage

**Decision:** Yes, store raw payloads. Use `List[Dict]` format.

**Format:**
```python
raw = [
    {"source": "fetch_my_trades", "data": {...}},
    {"source": "position_history", "data": {...}},
    {"source": "fetch_order", "data": {...}},
]
```

**Rationale:** Essential for debugging exchange integration issues. Cache size cost is acceptable given the value for troubleshooting.

### 10.4 Rate Limit Coordination

**Decision:** Implement shared temp file coordination + staggered startup jitter.

**Approach:**
1. **Temp file coordination:** A shared temp file logs recent API calls per exchange
2. **Staggered startup:** Random jitter (0-30s) when starting virgin bots

**Temp file structure:**
```
/tmp/passivbot_rate_limits/{exchange}.json
```

```json
{
  "calls": [
    {"endpoint": "fetch_my_trades", "timestamp_ms": 1705400000000, "user": "account1"},
    {"endpoint": "fetch_my_trades", "timestamp_ms": 1705400001000, "user": "account2"}
  ],
  "window_ms": 60000,
  "limits": {
    "fetch_my_trades": 120,
    "fetch_order": 60
  }
}
```

**Coordination logic:**
1. Before API call, read temp file and check current window usage
2. If approaching limit, add jitter delay (100-5000ms)
3. After API call, append entry to temp file
4. Periodically prune entries older than window

**Rationale:** Multiple virgin bots starting simultaneously on the same exchange is the high-risk scenario. Once initial cache is built, API pressure is minimal.

---

## 11. Cache Self-Healing & Gap Detection

### Gap Classification Challenge

Unlike candlesticks (predictable 60-second intervals), fill events have irregular timing. A gap could be:
- **Legitimate:** Bot was stopped, no trading occurred
- **Illegitimate:** Data fetch failed, events are missing

### Gap Detection Heuristics

1. **Time threshold:** Gaps > 12 hours trigger investigation
2. **Position discontinuity:** Position size jumps without fills → suspicious
3. **PnL discontinuity:** Wallet balance change without recorded PnL → suspicious

### Gap Metadata

```python
class KnownGap(TypedDict):
    start_ts: int           # Gap start timestamp (ms)
    end_ts: int             # Gap end timestamp (ms)
    retry_count: int        # Fetch attempts (max 3)
    reason: str             # auto_detected, fetch_failed, confirmed_legitimate, manual
    added_at: int           # When gap was first detected
    confidence: float       # 0.0=unknown, 0.3=suspicious, 0.7=likely_ok, 1.0=confirmed
```

### Gap Handling Strategy

1. **Initial detection:** Mark with `confidence=0.0`, `retry_count=0`
2. **Retry logic:** Up to 3 attempts with exponential backoff
3. **Classification:** After retries:
   - No new fills found → increase confidence toward legitimate
   - New fills found → gap was real, filled successfully
4. **Persistence:** After max retries, mark as known gap to avoid repeated fetches

### Gap Metadata Storage

Add `metadata.json` to cache directory:
```json
{
  "last_refresh_ms": 1705400000000,
  "oldest_event_ts": 1705000000000,
  "newest_event_ts": 1705399999000,
  "known_gaps": [
    {
      "start_ts": 1705200000000,
      "end_ts": 1705250000000,
      "retry_count": 3,
      "reason": "confirmed_legitimate",
      "added_at": 1705300000000,
      "confidence": 0.9
    }
  ]
}
```

---

## 12. Incremental Disk Flushing

### Problem

Initial cache building for 30 days can take 10+ minutes. If interrupted, all progress is lost.

### Solution: Flush Per Completed Day

```python
async def fetch_and_cache(self, start_ms: int, end_ms: int):
    current_day = _day_start(start_ms)
    while current_day < end_ms:
        day_events = await self._fetch_day(current_day)
        if day_events:
            self.cache.save_day(current_day, day_events)  # Atomic write
            self._update_metadata(current_day)
        current_day += DAY_MS
```

### Atomic Write Pattern

1. Write to `{date}.json.tmp`
2. Rename to `{date}.json` (atomic on most filesystems)
3. Update `metadata.json`

### Benefits

- No data loss on interruption
- Restart resumes from last complete day
- Minimal repeated work
- Progress visible via day files

---

## 13. Dashboard Tool Enhancements

### Current Features (`src/tools/fill_events_dash.py`)

- Multi-account view
- Date range filtering
- Cumulative PnL chart
- Daily PnL chart
- Top symbols by PnL
- Fees breakdown
- Fill events table

### Planned Enhancements

1. **Candlestick overlay:** Plot fills on price charts
   - Blue dots for buys, red dots for sells
   - Optional fetch of candlestick data for selected symbols

2. **Cache health panel:**
   - Show known gaps with confidence levels
   - Fetch progress indicator
   - Retry status for suspicious gaps

3. **Gap visualization:**
   - Timeline showing coverage periods
   - Highlight suspicious vs confirmed gaps

4. **Position replay:**
   - Reconstruct position history from fills
   - Show psize/pprice over time

5. **Export functionality:**
   - CSV/JSON export of filtered data

6. **Debugging aids:**
   - Show raw payloads for selected events
   - Validate field consistency
   - Check for duplicate IDs

---

## 14. Exchanges on Hold

| Exchange | Status | Reason |
|----------|--------|--------|
| Defx | Paused | Not fully supported yet |
| Paradex | Pending | Awaiting testing before implementation |

---

## Appendix A: Legacy `fetch_pnls` Locations

| File | Line | Function |
|------|------|----------|
| `exchanges/binance.py` | 238 | `fetch_pnls` |
| `exchanges/bybit.py` | 198, 318 | `fetch_pnls_sub`, `fetch_pnls` |
| `exchanges/bitget.py` | 300 | `fetch_pnls` |
| `exchanges/okx.py` | 229 | `fetch_pnls` |
| `exchanges/hyperliquid.py` | 232 | `fetch_pnls` |
| `exchanges/gateio.py` | 233 | `fetch_pnls` |
| `exchanges/kucoin.py` | 408 | `fetch_pnls` |
| `exchanges/defx.py` | 194 | `fetch_pnls` |

---

## Appendix B: Test Command

```bash
python -m pytest tests/test_fill_events*.py -v
```

---

## Changelog

| Date | Change |
|------|--------|
| 2025-01-16 | Resolved open questions: singleton, position flip, raw field, rate limiting |
| 2025-01-16 | Added sections 11-13: Cache self-healing, incremental flushing, dashboard enhancements |
| 2025-01-16 | Updated `raw` field type from `Dict` to `List[Dict]` |
| 2025-01-16 | Added section 14: Exchanges on hold (Defx, Paradex) |
