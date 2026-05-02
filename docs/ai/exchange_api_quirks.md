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

## Gate.io Futures

### Contract order text must start with `t-`

Problem:

1. Gate.io contract order `text` rejects values that do not start with `t-`.
2. CCXT prefixes `clientOrderId` into `text=t-...`, but raw `params["text"]` can overwrite that transformed value.

Handling:

1. Pass Passivbot custom order ids as `clientOrderId`, not raw `text`.
2. Keep broker attribution in the `X-Gate-Channel-Id` header.
3. Keep the Passivbot order-type marker inside the custom id; decoding accepts the marker inside Gate.io's `t-...` text.

### Public 1m OHLCV recent-window limit

Problem: Gate.io rejects old 1m OHLCV requests with `Candlestick too long ago. Maximum 10000 points recently are allowed`.

Handling:

1. Do not pass CCXT `until`; page forward by `since + limit`.
2. Clip 1m historical fetches to the recent-window bound and mark older spans as `no_archive`.
3. Require external OHLCV source data or another candle source for older Gate.io backtests.

## Hyperliquid

### Public candle endpoint recent-window limit

Problem: Hyperliquid `candleSnapshot` only serves the most recent 5000 candles for each timeframe, so 1m backtests older than roughly 3.5 days cannot rely on CCXT/API candles alone.

Handling:

1. Clip direct Hyperliquid 1m CCXT/API fetches to the recent 5000-minute window.
2. For older missing full-day 1m ranges from 2025-03-22 onward, use official requester-pays S3 raw node fills/trades from `hl-mainnet-node-data`, aggregate them into 1m OHLCV, and persist through the normal OHLCV cache.
3. Require complete hourly S3 coverage for a derived day; do not synthesize missing archive hours into apparently complete historical candles.
4. Require observable warnings when AWS credentials or the `lz4` decoder are unavailable; do not silently synthesize missing historical Hyperliquid data.

## General Guidance

1. Check raw exchange payloads when CCXT abstraction is insufficient.
2. Treat intra-page gaps and inter-page gaps differently.
3. For missing data incidents, verify source data before changing logic.
