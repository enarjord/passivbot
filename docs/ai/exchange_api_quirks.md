# Exchange API Quirks

Only task-specific, high-impact quirks are listed here.

## Bybit

### Closed-PnL pagination mismatch

Problem:

1. Cursor pagination has limited historical reach.
2. Time-based pagination can skip records when windows exceed page limits.

Handling in Passivbot:

1. Use hybrid pagination (cursor for recent, time-window for older).
2. Deduplicate by `orderId`.

Primary reference: `src/fill_events_manager.py` (`BybitFetcher._fetch_positions_history`).

## KuCoin Futures

### OHLCV limit behavior + sparse-minute markets

Problem:

1. Effective page size is 200 rows.
2. Illiquid symbols legitimately have missing trade minutes.

Handling:

1. Page with `limit=200`.
2. Overlap page boundaries by 1 candle to validate inter-page gaps.

## Bitget Futures

### `since` is effectively exclusive for OHLCV paging

Problem: naive paging can miss first candle in each page.

Handling:

1. Overlap boundaries by 1 candle.
2. Back up initial `since` by one candle on pagination start.

## General Guidance

1. Check raw exchange payloads when CCXT abstraction is insufficient.
2. Treat intra-page gaps and inter-page gaps differently.
3. For missing data incidents, verify source data before changing logic.
4. Treat startup exchange-config writes (`set_leverage`, `set_margin_mode`) as rate-limit-sensitive control-plane calls.
5. Pace exchange-config updates per symbol, track success per symbol, and retry failed symbols with backoff instead of treating startup config as all-or-nothing.
6. Do not fan out detached config tasks and partially await them; unhandled background failures create misleading `Task exception was never retrieved` noise and can hide which symbols still need configuration.
7. Benign exchange responses like "not modified" / "unchanged" should be handled explicitly and not logged as operational errors.
