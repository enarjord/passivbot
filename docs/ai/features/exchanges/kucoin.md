# KuCoin Futures Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


KuCoin Futures can reject a private request with code `400002` when its signed
timestamp is outside the accepted window. At the CCXT REST transport boundary,
retry only `futuresPrivate` GET requests once through normal throttling and signing.
A successful retry supplies fresh exchange data. If it fails, propagate the failure
to the existing caller policy. The transport logs the retry without request details.
Authentication failures and write requests retain the existing policy.

Primary reference: [KuCoin Futures error codes](https://www.kucoin.com/docs-new/error-code/futures).

## Authoritative futures book quotes

Use the public `allTickers` endpoint through CCXT's explicit
`futuresPublicGetAllTickers` method for bulk and symbol-filtered reads. The default
`contracts/active` endpoint supplies contract statistics but no bid/ask. Keep one
bulk request per read; do not fan out per symbol. Require positive finite bid,
ask and last-trade prices, reject crossed books and leave incomplete rows
unavailable to the snapshot caller. Never substitute last, mark or index price
for a missing top-of-book quote, or mark/index price for a last trade.

Primary reference: [KuCoin futures all tickers](https://www.kucoin.com/docs-new/rest/futures-trading/market-data/get-all-tickers).

## First-candle market-age discovery

KuCoin rejects `from=1`. Request daily candles from a valid millisecond timestamp
before futures launch (`2018-01-01`), with `to` explicitly set to the current time.
Pinned CCXT otherwise derives `to=since + limit * duration`; with `limit=1`, that
queries only one pre-listing day and returns no data. Retain the first returned
candle as the age basis. Empty or failed responses remain unknown and do not
bypass the configured minimum market age; zero cache entries are retried.

Primary reference: [KuCoin futures klines](https://www.kucoin.com/docs-new/rest/futures-trading/market-data/get-klines).

## IPv4 API-key whitelist transport

Problem: A dual-stack host may select IPv6 for KuCoin REST and private
WebSocket traffic even when the API key permits only the host's stable public
IPv4 address. KuCoin then rejects the first authenticated request as an IP
whitelist authentication failure.

Handling: Both KuCoin REST and WebSocket CCXT clients use IPv4-only network
connectors. Keep the host's stable public IPv4 address in the API-key whitelist.

## Private websocket token renewal

Problem:

1. KuCoin expires private futures websocket tokens during long-running sessions.
2. Pinned CCXT classifies `connectId=privateFutures` as `private` when handling the expiry message,
   so its cached futures URL keeps the rejected token and every reconnect reuses it.

Handling:

1. Clear the exact negotiated URL cache entry named by the websocket's `connectId` before CCXT
   propagates the expiry error to the reconnect loop.
2. Keep the expiry visible as a throttled websocket warning without emitting the dependency's raw
   callback traceback. The next `watch_orders` attempt must negotiate a new futures token.

## KuCoin hedge-mode refresh

Problem:

1. `set_position_mode(True)` is trading-critical setup, but broad no-op swallowing can hide a real one-way/hedge mismatch.
2. KuCoin order and fill payloads must carry `positionSide` in hedge mode; otherwise a both-sides-open account cannot safely infer an order's position side.

Handling:

1. Read the account's raw `positionMode` before a configuration write. An explicit hedge mode
   completes setup without a write. Only an explicit one-way mode permits a switch, and a
   fresh position-mode read must confirm hedge mode after a successful switch.
2. Protective startup uses the same strict mode read and requires existing hedge mode.
   Let failed reads, unrecognized modes, and failed writes propagate to the caller.
   The bounded Futures read retry applies to the position-mode read.
3. The connector keeps the exchange account in hedge mode even when `bot.portfolio.hedge_mode=false`
   disables simultaneous strategy exposure. Normalize order updates against that actual exchange
   mode, not the strategy flag.
4. Never infer a resting order's position side from the current position. Require explicit
   `info.positionSide`/`info.posSide` in exchange hedge mode; only when the exchange capability is
   actually one-way may `position_side` be derived and verified from the authoritative order side
   plus `reduceOnly` tuple.
5. In exchange hedge mode, derive entry/close-only effect from the authoritative buy/sell plus
   long/short tuple when KuCoin omits native `reduceOnly`.

Primary references: [Get Position Mode](https://www.kucoin.com/docs-new/rest/futures-trading/positions/get-position-mode)
and [Switch Position Mode](https://www.kucoin.com/docs-new/rest/futures-trading/positions/switch-position-mode).

## OHLCV limit behavior + sparse-minute markets

Problem:

1. Effective page size is 200 rows.
2. KuCoin documents that kline data is omitted for intervals with no ticks, including native
   higher-timeframe buckets.

Handling:

1. Page with `limit=200`.
2. Overlap page boundaries by 1 candle to validate inter-page gaps.
3. For native timeframes above 1m, synthesize a flat zero-volume no-trade bucket only when the gap
   is bounded by two real candles in the same successful payload and its timestamp is absent from
   the raw payload, and only up to the fixed 120-minute live connector policy. The simulation-only
   `backtest.gap_tolerance_ohlcvs_minutes` setting does not alter live readiness. A raw bucket
   rejected by candle validation remains unavailable and evicts any older cached sparse placeholder
   at that timestamp. An unidentifiable rejected row invalidates cached placeholders between the
   accepted page bounds, or across the remaining requested range when fewer than two accepted
   timestamps exist. Eviction recomputes the cached timeframe's derived index bounds. Do not
   synthesize leading, trailing, failed-fetch, oversized, or unproven between-page gaps.
4. For an unresolved 1m gap already bounded by cached real rows, retry with both boundary rows in
   the requested range. Promote the exact omission to verified no-trade continuity only if one
   successful raw payload returns both boundaries and no row inside the gap. This contextual proof
   may repair an older persistent `fetch_failed` gap. Empty, one-sided, malformed, or partially
   recovered responses remain unavailable, preserve persistent gap status, and start a five-minute
   contextual-proof cooldown rather than issuing another request on every candle read. Adjacent
   records retain independent retry clocks; all unverified fragments of the proof window must be
   eligible and both real bounds must fit one request page. Wider gaps retain ordinary repair
   instead of repeated short-cooldown proof attempts. Ordinary missing-range retry timing remains
   unchanged.
