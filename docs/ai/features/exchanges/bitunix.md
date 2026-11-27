# Bitunix Futures Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## Native connector and authentication

Bitunix is not available in the pinned CCXT release, so the production connector is a narrow
native async REST/WebSocket implementation rather than a generic `CCXTBot` exchange class.
It retains the CCXT-compatible object boundary consumed by Passivbot while implementing only the
futures operations required by live trading. Cold-cache standalone market preloads use this native
public REST client as well; they must never fall through to CCXT before the bot is constructed.
Native market records use Bitunix's documented VIP0 futures maker/taker fees as conservative
planning inputs because the trading-pairs endpoint does not publish account fee tiers.

Handling:

1. Sign private REST requests with Bitunix's documented double SHA-256 scheme. Sort query keys for
   signing, and sign the exact compact JSON body bytes sent on POST.
2. Supply `marginCoin=USDT` to account and symbol-configuration requests. Successful account data
   may be either an object or a singleton list; accept exactly those shapes.
3. Map business error envelopes to CCXT exception classes and propagate unknown failures. Keep
   request spacing below the venue's documented UID/IP rolling limit. Treat the observed
   `code=1, msg=Network Error` envelope like documented network error `10001`, and retry native
   market discovery with bounded backoff so a transient cold-start response is not permanent.
   Attach the bounded venue code to the mapped exception for structured write diagnostics; do not
   propagate response bodies, configured headers, or credentials into the public diagnostic path.
4. Authenticate the private WebSocket with its seconds-based signature, subscribe to `order`, and
   send Bitunix's application-level JSON ping while idle; transport-level WebSocket heartbeats do
   not replace the venue keepalive. Enrich each order notification from REST detail before
   publishing it. The raw push lacks enough durable close-only metadata for authoritative
   reconciliation. If REST reports that the order is not found or returns semantically invalid
   order detail, publish only that raw row as untrusted so the generic watcher requests an
   authoritative account-state refresh instead of silently discarding the transition. Treat
   non-object pushes and rows without an order ID the same way instead of dropping them before
   reconciliation. Transport failures still fail the batch and reconnect.
5. Apply custom endpoint domain rewrites and `rest.url_overrides.api` to the native REST base, and
   merge `rest.extra_headers` into every request. Reject authentication header names
   case-insensitively in configured headers so proxy or user headers cannot collide with generated
   signatures. Honor `disable_ws` for private orders and public market streams: use REST order
   polling, request only explicit bounded symbol sets through REST depth, and leave canonical
   candle refresh and repair to REST.
6. Keep `bitunix: null` explicit in `broker_codes.hjson`; there is no Passivbot broker payload for
   this connector.

Primary references: [Bitunix REST authentication](https://www.bitunix.com/api-docs/futures/common/sign.html),
[WebSocket login](https://www.bitunix.com/api-docs/futures/websocket/prepare/WebSocket.html), and
[order channel](https://www.bitunix.com/api-docs/futures/websocket/private/Order%20Channel.html).

## Hedge orders, positions, and fills

Bitunix's hedge placement request uses a venue-specific side contract:
`BUY + OPEN` opens long, `SELL + OPEN` opens short, `BUY + CLOSE` closes long, and
`SELL + CLOSE` closes short. A close additionally requires the live `positionId`. Order-detail and
fill responses instead expose the actual buy/sell action.

Handling:

1. Keep the exchange account in `HEDGE` mode. Translate Passivbot's explicit `position_side` into
   the placement-side tuple and send `reduceOnly=true` plus the cached authoritative `positionId`
   on every close.
2. Treat response `side` as the actual action. Derive long/short from action plus the boolean
   `reduceOnly`; never reuse placement-side semantics while parsing a response.
3. Accept both documented `LONG`/`SHORT` and observed `BUY`/`SELL` aliases on position rows. Keep
   all other position-side values invalid.
4. Require a positive finite quantity on every normalized order and a positive finite request
   price for every limit order and every open order. Only terminal market-order detail may omit the
   request price.
5. Bitunix has emitted `NEW_` on live order detail although its schema documents `NEW`. Normalize
   only trailing underscore padding before applying the closed order-status allowlist. Pending
   endpoint membership is itself authoritative evidence that an order remains pending, so retain
   an otherwise unknown bounded code-like transition status as open until a later complete
   pending snapshot removes it, with a rate-limited structured warning. Missing, free-form, and
   oversized statuses remain invalid, and order-detail normalization keeps the strict allowlist.
6. Page pending orders by `skip` to the required, stable reported total under one fixed `endTime`
   snapshot. Reject missing, changing, truncated, or duplicate pagination results before treating
   the account-critical open-order set as authoritative.
7. Page trade history by `skip` to the required, stable reported total under one fixed `endTime`
   snapshot. Reject missing, changing, truncated, or duplicate pagination results. Preserve
   `realizedPNL` and the fee sign so maker rebates remain positive balance impacts. Enrich empty
   fill `clientId` values through order detail, but retain the exchange-truth fill with unknown
   attribution when terminal order detail has expired. This is the canonical fill source for
   realized PnL, unstuck accounting, and HSL reconstruction.
8. Reconstruct realized wallet balance as `available + frozen + margin`. These are the disjoint
   available, order-locked, and position-margin quantities. Keep
   `crossUnrealizedPNL + isolationUnrealizedPNL` separate as mark-to-market state so price movement
   does not change Rust's realized sizing balance.

Primary references: [place order](https://www.bitunix.com/api-docs/futures/trade/place_order.html),
[pending positions](https://www.bitunix.com/api-docs/futures/position/get_pending_positions.html),
[order detail](https://www.bitunix.com/api-docs/futures/trade/get_order_detail.html), and
[history trades](https://www.bitunix.com/api-docs/futures/trade/get_history_trades.html).

## Recovery of ordinary creations

Keep the Bitunix create journal under `var/bitunix/<user>/create_attempts.sqlite3`
in the live data directory. Keep that data directory across restarts and release switches.
See [persistent live state](../../../container_deployment.md#persistent-live-state) for containers.
The [recovery contract](../../error_contract.md#live-action-readiness) applies to one outstanding
ordinary create or two members of one eligible close wave. A committed pending-order cohort can
adopt a working order by exact `clientId`. Otherwise, use exact order detail and bounded
order-history queries. Query both
canceled and non-canceled history where applicable. A terminal fill quantity must agree with
the canonical trade data by venue order ID. The journal retains the exact venue ID and the
greatest observed cumulative fill across restarts. A conflicting venue ID keeps the slot
reserved for investigation.

Bitunix describes `20007` as a lookup miss to retry later and `30042` as a duplicate
client ID. The API documents neither a definitive absence interval nor idempotent replay.
Keep an unresolved slot reserved, with bounded lookup retries and operator reminders. If it
persists, inspect the original attempt and its venue outcome before releasing it or rolling
back to a build that cannot read the journal.
[Order detail](https://www.bitunix.com/api-docs/futures/trade/get_order_detail.html),
[order history](https://www.bitunix.com/api-docs/futures/trade/get_history_orders.html),
[error codes](https://www.bitunix.com/api-docs/futures/ErrorCode/error_code.html).

For a close wave, send individual placement requests. Require the exact echoed client ID and
a unique venue order ID from the placement response before continuing. Persist that venue ID
as identity evidence and retain the reservation until authoritative recovery. An account
update on the private order stream during the wave invalidates its permission.

An existing serial journal remains readable and keeps its pending evidence. The first wave
upgrades an empty journal to schema v2 in a transaction with its first reservation. Once that
occurs, keep a v2-aware build for restart and rollback. A downgrade to a v1-only build requires
the bot to be stopped and all reservations resolved before the empty journal can be archived
and reinitialized.

## Product-admission retry policy

The native connector starts the per-symbol API-unavailable cooldown after a failed write
with structured exchange error code `710003`. The cooldown applies to the affected
symbol's entries for `live.recovery.exchange_symbol_unavailable_cooldown_hours`.
Held positions retain `tp_only` and stronger explicit modes.
Bitunix's [stock-perpetual guide](https://www.bitunix.com/hub/helpcenter/article/introduction-to-stock-perpetuals?id=211)
describes a separate product boundary; its
[public error-code table](https://www.bitunix.com/api-docs/futures/ErrorCode/error_code.html)
does not define `710003`.

An ordinary create that receives `710003` leaves its outcome unresolved and consumes the normal
execution error budget. The journal reservation defers ordinary creates across the account,
including replacement closes on other symbols. Cancellation and dedicated HSL protection
remain eligible under their own readiness, risk, and execution checks. The symbol cooldown
defers leverage and margin configuration for that symbol. An explicit reduce-only close can
bypass that configuration prerequisite while the cooldown is active. Native `CLOSE` still
requires the authoritative `positionId` and the normal account, fill, and risk guards.

Recovery of the journal reservation follows the [ordinary creation recovery contract](#recovery-of-ordinary-creations)
even after cooldown expiry or restart. See [Bitunix create recovery](../../../troubleshooting.md#bitunix-ordinary-creates-are-paused)
for operator checks.

## Live quotes and candles

The bulk REST ticker omits bid and ask. Use the official public `tickers` WebSocket for
authoritative top-of-book and last price, splitting the active market set across connections
because Bitunix permits at most 300 subscriptions per connection. Refresh those subscriptions when
the active market-ID set changes. Determine cache freshness from the bounded local receipt time
and preserve that receipt through normalization and the shared quote cache. Exchange timestamps
remain provenance. A targeted ticker request may use one-row REST depth for at most eight missing
or expired symbols, with the bid/ask midpoint labeled as its synthetic last through ticker
normalization and into snapshot provenance. Broad operation must fail instead of
fanning out one depth request per market or substituting bid/ask for a last trade. When WebSockets
are explicitly disabled, live market snapshots select this targeted REST path directly without
opening or waiting for a public ticker socket; unbounded bulk requests and requests above the
eight-symbol limit fail closed.

Live snapshots default to targeted ticker reads even with WebSockets enabled. Only requested
symbols participate in the bounded readiness wait, so an unrelated silent market cannot delay
protection of a held position. This still uses the shared websocket cache and retains its freshness
checks and capped depth fallback. It applies the caller's stricter freshness age and can repair
an established stream's stale subset without waiting for cold subscriptions. An explicit refresh
reads the requested symbols from the connector, bypassing the shared snapshot cache. A WebSocket
observation within the requested age remains eligible at its original receipt time.
`market_snapshot_ticker_strategy=bulk` remains an operator override for normal reads.

Bitunix klines return at most 200 rows. The live field names are inverted relative to their units:
`quoteVol` is base quantity and `baseVol` is quote notional; normalize `quoteVol` as CCXT base
volume so Passivbot's generic quote-volume calculation remains dimensionally correct. Missing,
non-finite, or negative base volume is invalid; an explicit zero remains valid. In live behavior,
`startTime` does not anchor a response; the venue tail-anchors at `endTime`. Derive each forward
page's `endTime` from `since + limit * timeframe`, filter to the requested bounds, sort ascending,
and deduplicate. This pagination supports live warmup, restart reconstruction, and runtime
indicators only. Bulk historical Bitunix data for backtesting or optimization is not a supported
source.

When WebSockets are enabled, multiplex the active forager candidates on the official
`market_kline_1min` channel. Deterministically shard the sorted symbol set across public sockets,
with no more than the venue's 300 subscriptions on each connection.
The push timestamp is receipt/update time, so floor it to the one-minute bucket; normalize `b` as
base volume and validate the complete OHLC row. Pass the changing open bucket through the generic
successor-candle finalization boundary. Drop a transient internally inconsistent update instead of
clamping exchange values or failing unrelated multiplexed symbols; a bounded consecutive run wakes
the affected watcher into REST fallback and suspends new rows for that symbol until the watcher
consumes the fallback signal. Startup basis, reconnect gaps, persistent malformed data, prolonged
silence, and periodic integrity checks remain REST-owned; a transport failure wakes every affected
watcher so provenance is cleared before bounded reconnect and REST fallback. Track application-data
liveness per symbol so unrelated Kline traffic and control frames cannot conceal a stalled
subscription, while also enforcing connection silence independently of the short receive polling
used for subscription reconciliation. Validate subscription acknowledgements and wake only the
rejected symbols into REST fallback when the response identifies them; an unscoped rejection
conservatively wakes the pending subscription batch.

Primary references: [ticker WebSocket](https://www.bitunix.com/api-docs/futures/websocket/public/Tickers%20Channel.html),
[REST depth](https://www.bitunix.com/api-docs/futures/market/get_depth.html), and
[Kline WebSocket](https://www.bitunix.com/api-docs/futures/websocket/public/kline%20channel.html),
and [kline API](https://www.bitunix.com/api-docs/futures/market/get_kline.html).
