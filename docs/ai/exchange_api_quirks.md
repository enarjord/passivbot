# Exchange API Quirks

Only task-specific, high-impact quirks are listed here.

## Broker Agreement Attribution

Problem:

1. Broker attribution is implemented differently per exchange: headers, CCXT options, order tags, or client order ids.
2. CCXT defaults may point attribution to CCXT, not Passivbot.
3. Removing broker code can silently break Passivbot broker agreements while trading continues normally.

Handling in Passivbot:

1. Treat broker-code handling as exchange-critical behavior.
2. Do not remove existing broker attribution without explicit user approval.
3. Broker-code registry loading must fail loudly on missing/invalid registry data and unknown exchange names.
4. For each broker-agreement exchange, verify the actual signed CCXT/raw request includes the required broker field/header/tag.
5. Add regression tests at the request-construction boundary when changing exchange sessions, signing, or order payload code.

## Bybit

### Broker referer header

Problem: Bybit broker attribution depends on the `Referer` header on order POST requests. CCXT derives this from `options["brokerId"]`, whose default may not be Passivbot.

Handling:

1. Set Bybit CCXT client `options["brokerId"]` from `broker_codes.hjson`.
2. Test that a signed `v5/order/create` request contains `Referer: passivbotbybit`.

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
