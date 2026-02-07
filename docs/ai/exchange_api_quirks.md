# Exchange API Quirks

This document catalogs known exchange API quirks, limitations, and workarounds discovered during development. When implementing exchange-specific code, check here first.

## Bybit

### Closed-PnL Pagination (Critical)

**Discovered:** 2026-01-25

**Problem:** Bybit's `/v5/position/closed-pnl` endpoint has two pagination mechanisms that behave differently:

1. **Cursor pagination** (`nextPageCursor`): Only covers ~7 days of recent data, then cursor becomes empty
2. **Time-based pagination** (`endTime`): Can reach older data but may skip records when there are >100 records in a time window

**Symptoms:**
- Close fills missing PnL (showing `pnl: 0.0`)
- Inconsistent historical data - some old records present, others missing
- CCXT's `fetch_positions_history` wrapper doesn't expose cursor, making it unreliable

**Root Cause:**
When using time-based pagination alone:
- If a time window has >100 records, only 100 are returned
- Setting `endTime = oldest_timestamp_in_batch` for next request skips records between batches
- Example: 128 records exist in window, only 100 fetched, 28 missed

**Solution:** Hybrid pagination in `BybitFetcher._fetch_positions_history`:
1. Phase 1: Use cursor pagination for recent records (efficient, no gaps)
2. Phase 2: When cursor exhausts (~7 days back), switch to time-based sliding window
3. Deduplicate by orderId to handle overlap

**Code Reference:** `src/fill_events_manager.py` - `BybitFetcher._fetch_positions_history()`

**Testing:** Verified fetching 1434 records vs 387 (cursor-only) or 1200 (time-only with gaps)

### Closed-PnL Record Timing

Each close fill on Bybit immediately generates a closed-pnl record with `avgEntryPrice`. This means:
- PnL can be computed per-fill, not just when position fully closes
- The formula: `(exit_price - avgEntryPrice) * closedSize * direction - fees`
- Old fills (>30 days) may have expired closed-pnl records on Bybit's servers

## Binance

(Add Binance-specific quirks here as discovered)

## KuCoin Futures

### OHLCV Pagination Limits + Sparse Minutes

**Discovered:** 2026-02-04

**Problem:** KuCoin futures `fetch_ohlcv` returns at most **200 rows per call**, regardless of `limit`.
In addition, 1m OHLCV appears to be **trade-only** for many symbols (illiquid contracts return
minutes only when trades occurred), which creates legitimate gaps.

**Symptoms:**
- Large synthesized zero-candle counts for illiquid symbols (TRX/CRO/XLM/DOT/AVAX, etc.)
- Pagination with limit=1000 still yields 200 rows/page
- Small internal gaps even for higher volume symbols (e.g. ZEC/XMR)

**Mitigations in Passivbot:**
- Use `limit=200` for KuCoin futures OHLCV pagination
- Overlap page boundaries by 1 candle to validate gaps between fetches
- Treat gaps **inside a single payload** as verified no-trade gaps (don’t retry)

## Bitget Futures

### OHLCV since parameter is exclusive

**Discovered:** 2026-02-04

**Problem:** Bitget `fetch_ohlcv` treats `since` as **exclusive**, which can skip the
first candle in each paginated page if you advance `since` naively.

**Symptoms:**
- Regularly spaced 1‑minute gaps at ~200‑minute intervals (page boundaries)
- Synthesized single‑minute candles even on liquid symbols

**Mitigations in Passivbot:**
- Use `limit=200` per page
- Overlap page boundaries by 1 candle
- Back up the initial `since` by 1 candle when paginating (exclusive semantics)

## General Patterns

### CCXT Wrapper Limitations

CCXT normalizes exchange APIs but sometimes loses important data:
- Pagination cursors may not be exposed
- Exchange-specific fields may be buried in `info` dict
- Always check raw response (`trade.get("info", {})`) when CCXT fields are insufficient

### Debugging Missing Data

When data appears incomplete:
1. **Query raw API directly** - bypass CCXT to see actual response
2. **Check pagination** - are there more pages? Is cursor working?
3. **Check time windows** - does the time range include the missing data?
4. **Check data retention** - how long does the exchange keep this data?
5. **Compare endpoints** - does another endpoint have the data?

### Principle: Overlap Page Boundaries to Validate Gaps

When paging by time:
- **Gaps inside a single payload** can be treated as legitimate (exchange returned no data).
- **Gaps between pages** are ambiguous unless you overlap requests. Start the next request
  at the last known valid candle (or slightly before) to confirm whether the gap is real.

See `docs/ai/debugging_case_studies.md` for detailed examples.
