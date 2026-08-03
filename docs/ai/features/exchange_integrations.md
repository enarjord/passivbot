# Exchange Integration Contracts

Only task-specific, high-impact contracts and quirks are listed here. Authenticated verification
requires explicit user approval; prefer offline request-construction tests.

## Supported Live-Exchange Boundary

The supported production live connectors are Binance, Bybit, Bitget, Bitunix, OKX, Gate.io,
KuCoin, Hyperliquid, and WEEX. The fake connector is an offline deterministic test harness, not an
exchange.

Defx is deliberately unsupported. `src/exchanges/defx.py` and the `setup_bot()` routing branch are
stale legacy placeholders retained only until a separate cleanup removes them. Their presence does
not make Defx a supported connector and must not expand feature coverage, implementation matrices,
regression requirements, or live-testing scope. The canonical live fill-event factory rejects Defx
because required realized-PnL, unstuck, and HSL replay support is absent. Do not use the Defx adapter
for live operation or authenticated probes.

Paradex is experimental and outside the supported production boundary. Its adapter and
`setup_bot()` routing branch may be used as comparative implementation or rate-limit research, but
required live fill/PnL, unstuck, and HSL replay contracts are incomplete. Do not infer production
support, implementation coverage, regression requirements, or live-testing scope from its runtime
routing branch or comparative documentation.

The generic `CCXTBot` fallback for arbitrary exchange names is also outside the supported
production boundary. It preserves compatibility for unaudited CCXT venues, but a feature requiring
authoritative order-type, close-only, remaining-quantity, or one-way position-side normalization
must use an explicit supported-connector allowlist. The generic fallback retains its legacy basic
reconciliation/tolerance path until that venue receives a connector-specific contract audit; the
separate global retirement of the old initial-entry-only distance gate does not enable the new
churn policy there.

### Temporary exchange-side symbol suspension

Some venues reject otherwise valid authenticated writes because API trading for one symbol is
temporarily unavailable. This is a connector-owned classification boundary: the shared runtime
must never infer a suspension from exception text or a portable-looking numeric code. A supported
connector may opt in only with an exact structured exchange code and a fixture-backed regression
test.

On a proven suspension, `live.exchange_symbol_unavailable_cooldown_hours` starts a per-symbol,
RAM-only cooldown (default `6.0` hours; `0` disables it; maximum `876600` hours). Values outside
that bounded conversion domain are rejected during configuration and cannot mask the original
exchange failure. A flat affected symbol is nontradable for
new planning through `graceful_stop`. An affected symbol with a position uses `tp_only`, overriding
`normal` or `graceful_stop`, so all further entries are suppressed while ordinary closes remain
available without successful leverage refresh. Explicit `panic` and `manual` modes remain stronger.
The initial failed entry/configuration write retains the normal execution failure/restart-budget
consequences. Retry-backoff cycles with no authenticated write do not count as additional failures;
the cooldown prevents repeated futile entry writes rather than hiding the fault.
Expiry restores on-demand attempts, another classified response starts a new cooldown, and process
restart deliberately clears the state and retries immediately.

Future connectors encountering an equivalent venue rule must add their own exact classifier. Do
not generalize one exchange's code, match human-readable messages, persist cooldowns across runs, or
make held positions wholly nontradable.

## Private Order Websocket Normalization

Authoritative REST open-order reconciliation remains strict. Binance and KuCoin
private websocket notifications for Passivbot-owned orders may omit native
long/short metadata; only that hint path may recover `position_side`, only in
effective hedge mode, and only when a valid Passivbot client-order marker has
no conflicting native position-side field and every supplied exchange/client
identity matches the same record in this process's emitted-order registry.
Acknowledged emitted identities remain registered for as long as the
corresponding order is present in the bot's authoritative open-order state,
even beyond the normal foreign-writer lookback. Recovered rows always force an
authoritative account refresh. Sparse foreign, explicitly one-way,
identity-conflicting, or unmarked notifications remain rejected.

Bitget UTA websocket rows use the exchange-native `holdSide` field for hedge
position attribution, while classic rows use `posSide`. Hyperliquid may deliver
an order-open websocket row before the concurrent create request returns its
exchange order ID. While a create is still freshly submitted, the connector may
briefly yield for that acknowledgement and retry the row, but recovery still
requires the exact acknowledged exchange ID and all existing contradiction
checks. Sparse Hyperliquid rows for orders already resting at process startup may
instead recover side, `position_side`, and close-only intent from one exact
exchange-ID match in the current authoritative REST open-order snapshot. A
bounded five-minute in-memory copy of those exact semantics covers terminal
updates which arrive just after reconciliation removes the order; it is rebuilt
from REST after restart and does not preserve ownership or trading intent.
Every supplied exchange-ID and client-ID alias must agree with the canonical
identity retained by the snapshot or its bounded copy. A contradictory current
snapshot row invalidates any older cached semantics for its exchange-ID aliases.
Missing, duplicate, expired, or contradictory matches remain rejected and
trigger a fresh account-state read;
authoritative snapshot contradictions must not fall back to process-local
acknowledgement evidence. A unified `reduceOnly` value of `false` or `null` is
treated as a CCXT placeholder only when the native row omits that field; an
explicit native value still must agree with the recovered semantics. All
snapshot-recovered rows request authoritative refresh because an exact exchange
ID proves semantics but not local ownership. Price, quantity, side, or order
shape alone never prove ownership.

A successful private-websocket read and a valid individual order row are separate health
boundaries. When a supported CCXT connector receives a row whose mandatory side, position-side,
quantity, or close-only semantics cannot be normalized, it discards that row, requests an
authoritative account-state refresh, and emits a bounded warning. Other valid rows from the same
websocket message remain usable. A semantic row rejection must not be reported as a transport
disconnect or consume the websocket reconnect budget; actual watch/read failures retain the
bounded reconnect backoff.

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

### WEEX broker client-order IDs

WEEX attributes both Spot and Futures API volume when `newClientOrderId` starts with
`b-{brokerId}-`. Passivbot's WEEX connector is Futures-only and the V3 Futures order endpoint
limits this field to 36 characters, so every generated ID uses
`b-{brokerId}-0xTTTT{random}` within that limit. The `0xTTTT` Passivbot order-type marker must
remain intact for ownership, reconciliation, and fill-event diagnostics. Reject invalid or
unattributed IDs before submission; do not rely on CCXT's WEEX `partner` default because CCXT
ignores it when the caller supplies a client order ID.

## Exchange Hedge Mode Versus Strategy Hedge Mode

`live.hedge_mode=false` disables simultaneous long and short strategy exposure; it does not put an
exchange account into one-way mode. Binance and Bitget connectors keep the exchange account in
hedge mode. Their private order updates must therefore normalize `position_side` and close-only
semantics from the exchange's actual mode and explicit `positionSide`/`posSide` fields even when
the strategy setting is false. Only connectors whose `hedge_mode` capability is actually false may
use the one-way side plus `reduceOnly` attribution path.

## Bybit

### Broker referer header

Problem: Bybit broker attribution depends on the `Referer` header on order POST requests. CCXT derives this from `options["brokerId"]`, whose default may not be Passivbot.

Handling:

1. Set Bybit CCXT client `options["brokerId"]` from `broker_codes.hjson`.
2. Test that a signed `v5/order/create` request contains `Referer: passivbotbybit`.

### Closed-PnL pagination mismatch

Problem:

1. Supplying only `endTime` makes Bybit search the preceding seven days.
2. Deriving the next window from a sparse page's oldest row can skip records
   between that row and the current window boundary.
3. A time window may still contain more than one page.

Handling in Passivbot:

1. Partition the requested range into explicit contiguous windows shorter than
   seven days, passing both `startTime` and `endTime`.
2. Cursor-paginate each window to exhaustion before moving to the next older
   window.
3. Deduplicate by `orderId`.
4. Propagate endpoint and pagination-progress failures; incomplete closed-PnL
   history must not be treated as a successful fetch.

Primary reference: `src/fill_events_manager.py` (`BybitFetcher._fetch_positions_history`).

## KuCoin Futures

### IPv4 API-key whitelist transport

Problem: A dual-stack host may select IPv6 for KuCoin REST and private
WebSocket traffic even when the API key permits only the host's stable public
IPv4 address. KuCoin then rejects the first authenticated request as an IP
whitelist authentication failure.

Handling: Both KuCoin REST and WebSocket CCXT clients use IPv4-only network
connectors. Keep the host's stable public IPv4 address in the API-key whitelist.

### KuCoin hedge-mode refresh

Problem:

1. `set_position_mode(True)` is trading-critical setup, but broad no-op swallowing can hide a real one-way/hedge mismatch.
2. KuCoin order and fill payloads must carry `positionSide` in hedge mode; otherwise a both-sides-open account cannot safely infer an order's position side.

Handling:

1. Treat current same-mode success as success (`code=200000`, `data.positionMode=1`).
2. Let unknown `set_position_mode` failures raise unless a verified KuCoin no-op code is added with a targeted test.
3. The connector keeps the exchange account in hedge mode even when `live.hedge_mode=false`
   disables simultaneous strategy exposure. Normalize order updates against that actual exchange
   mode, not the strategy flag.
4. Never infer a resting order's position side from the current position. Require explicit
   `info.positionSide`/`info.posSide` in exchange hedge mode; only when the exchange capability is
   actually one-way may `position_side` be derived and verified from the authoritative order side
   plus `reduceOnly` tuple.
5. In exchange hedge mode, derive entry/close-only effect from the authoritative buy/sell plus
   long/short tuple when KuCoin omits native `reduceOnly`.

### OHLCV limit behavior + sparse-minute markets

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
   recovered responses remain unavailable, preserve persistent gap status, and restart the
   persistent retry cooldown rather than issuing another contextual request on every candle read.

## Bitget Futures

### Bitget hedge-mode refresh

Problem:

1. Bitget hedge-side attribution depends on `posSide`/`holdSide` payload fields.
2. Broadly swallowing hedge-mode setup errors can mask an unsafe one-way/hedge mismatch.

Handling:

1. Treat current same-mode success as success (`code=00000`, `data.posMode=hedge_mode`).
2. Let unknown `set_position_mode` failures raise unless a verified Bitget no-op code is added with a targeted test.
3. Normalize against the actual exchange account mode, not `live.hedge_mode`, which only controls
   simultaneous strategy exposure.
4. Require explicit side-disambiguating payloads for order/fill normalization instead of defaulting to long; open orders should carry `posSide`, while fills may use `tradeSide`/`side`/`posMode`.

### UTA / Elite hedge-mode order direction

Problem:

1. Bitget UTA hedge-mode orders use `side` plus `posSide` for entries and closes.
2. `reduceOnly` is one-way-only in UTA and is rejected when combined with `posSide`.
3. UTA open-order responses may report close orders with `side=sell`, `posSide=long`,
   and `reduceOnly=NO`; deriving close direction from `reduceOnly` misclassifies them
   as entries.

Handling in Passivbot:

1. Send `posSide` and `clientOid` for UTA hedge-mode orders, but do not send
   `reduceOnly`.
2. Normalize UTA open orders from the explicit exchange/CCXT `side` field for
   buy/sell direction, and from `posSide` for long/short position side.
3. If the effective account mode is one-way but a UTA order still reports an explicit long/short
   `posSide`, derive close-only effect directly from its `side` plus `posSide` action tuple. If it
   does not report that tuple, require an authoritative native `reduceOnly` value.
4. Keep classic Bitget v2/mix `tradeSide`/`reduceOnly` handling separate.

### `since` is effectively exclusive for OHLCV paging

Problem: naive paging can miss first candle in each page.

Handling:

1. Overlap boundaries by 1 candle.
2. Back up initial `since` by one candle on pagination start.

## OKX Futures

### Long/short-mode close semantics

Problem:

1. OKX long/short mode identifies entry versus close from `side` plus `posSide`.
2. CCXT emulates reduce-only for this mode and may expose `reduceOnly=false` for a valid close.

Handling in Passivbot:

1. In effective long/short mode, normalize close-only effect from the documented `side` plus
   `posSide` action tuple.
2. In effective one-way/net mode, require an authoritative native `reduceOnly` value and verify
   one-way `position_side` against side plus close-only effect.
3. Prefer the raw exchange `info` field over a CCXT top-level default when proving close-only
   semantics.

## Gate.io Futures

### Exchange identity boundary

Passivbot's canonical exchange identity is `gateio`. This identity owns connector
routing, cache and event paths, broker attribution, and persisted state. CCXT 4.5.66
renamed only its client class to `gate`.

For compatibility, `api-keys.json` may specify either `"exchange": "gateio"` or
`"exchange": "gate"`. Passivbot normalizes the latter to `gateio` and logs that
migration. Only CCXT REST and WebSocket client construction translates `gateio` to
`gate`; do not add parallel `gate` identities to internal registries or state paths.
Normalize Gate's numeric REST account `user` value to a string before assigning it
as CCXT Pro's private futures subscription UID. Reject missing, null, empty, or
non-string/non-integer identifiers before conversion; never cache a placeholder
such as `"None"` as durable subscription state.

### Multi-currency balance semantics

Gate's `cross_available` is spendable margin, not stable account equity. Resting
orders move value between `cross_available` and `cross_order_margin`, while open
positions use `cross_initial_margin`. For multi-currency margin accounts, derive
the strategy balance from the same authoritative futures-account row as
`cross_available + cross_order_margin + cross_initial_margin -
cross_unrealised_pnl`. Require all four finite fields. Removing unrealized PnL
preserves wallet-balance semantics because Passivbot adds position PnL separately
when deriving equity. Do not feed `cross_available` alone to Rust, because
ordinary order reservation would then resize ideal orders and create
reconciliation churn. Classic accounts continue using CCXT's quote-currency
total.

### Per-symbol leverage initializes the position risk limit

Problem:

1. Gate derives a position's current risk limit from its configured leverage.
2. A contract risk-table update can leave a previously configured leverage above
   the new maximum. Gate then reports a zero risk limit and rejects every opening
   order with `RISK_LIMIT_EXCEEDED`.
3. Gate's leverage endpoint also controls margin mode: nonzero `leverage` selects
   isolated mode, while cross mode requires `leverage=0` plus
   `cross_leverage_limit`.

Handling:

1. Configure every symbol before its first order creation in a bot process.
2. Use the leverage capped by current market metadata and the configured margin
   preference.
3. Call CCXT `set_leverage` with an explicit `marginMode`; CCXT then produces the
   correct Gate parameter tuple without accidentally switching margin mode.
4. Propagate configuration failures and keep exposure-increasing creations deferred
   until configuration succeeds. Existing-position reduce-only closes do not depend
   on leverage initialization and remain eligible. Missing or invalid leverage-cap
   metadata is a symbol-scoped configuration failure: invalidate any prior configured
   marker, do not guess a cap or issue `set_leverage`, and continue market initialization
   so reduce-only closes for that symbol can still execute.
5. Every failed cycle containing a blocked entry advances the existing execution
   error/restart budget. The failure and blocked-entry scope remain visible in
   operator logs and structured events; close-only operation must not look healthy
   while entries are failing.
6. Reserve and debit the one-time signed leverage write in the account-wide order
   churn allowance before admitting the symbol's first entry creation. The
   reservation decision and configuration execution must use the same
    retry-eligibility timestamp: a backoff expiring later in that creation wave
    is deferred until the next wave rather than issuing an unreserved write.

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

### Delayed sparse-tail candles

Hyperliquid may initially omit a recent no-trade minute and later publish an
authoritative flat candle with zero volume for that same timestamp. Passivbot
must not synthesize or permanently classify the initially absent row. Recent
tail gaps use time-spaced retries, remain unavailable meanwhile, and are removed
from `known_gaps` as soon as the authoritative row is persisted.

### Candle retention and held trailing positions

Hyperliquid's `candleSnapshot` endpoint exposes only the most recent 5,000
candles. A trailing position whose latest fill predates that 1m window therefore
cannot reconstruct its extrema from a fresh host. Preserve and transfer the
existing 1m candle cache when migrating such a live position. Passivbot must
keep trailing state unavailable if the exact range from the first whole minute
after the fill is not locally present; wider-timeframe candles or an
earliest-available reset are not parity-safe substitutes. Once that range is
dense, only the bounded non-persistent open-tail projection described in the
candlestick-manager contract may bridge delayed current candles.

## Bitunix Futures

### Native connector and authentication

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
   signatures. Honor `disable_ws` for both private orders and public tickers: use REST order
   polling and request only explicit, bounded symbol sets through REST depth.
6. Keep `bitunix: null` explicit in `broker_codes.hjson`; there is no Passivbot broker payload for
   this connector.

Primary references: [Bitunix REST authentication](https://www.bitunix.com/api-docs/futures/common/sign.html),
[WebSocket login](https://www.bitunix.com/api-docs/futures/websocket/prepare/WebSocket.html), and
[order channel](https://www.bitunix.com/api-docs/futures/websocket/private/Order%20Channel.html).

### Hedge orders, positions, and fills

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
   only trailing underscore padding before applying the closed order-status allowlist.
6. Page pending orders by `skip` to the required, stable reported total under one fixed `endTime`
   snapshot. Reject missing, changing, truncated, or duplicate pagination results before treating
   the account-critical open-order set as authoritative.
7. Page trade history by `skip` to the required, stable reported total under one fixed `endTime`
   snapshot. Reject missing, changing, truncated, or duplicate pagination results. Preserve
   `realizedPNL` and the fee sign so maker rebates remain positive balance impacts. Enrich empty
   fill `clientId` values through order detail, but retain the exchange-truth fill with unknown
   attribution when terminal order detail has expired. This is the canonical fill source for
   realized PnL, unstuck accounting, and HSL replay.
8. Reconstruct realized wallet balance as
   `available + frozen + margin - crossUnrealizedPNL - isolationUnrealizedPNL`; do not feed
   mark-to-market equity into Rust sizing.

Primary references: [place order](https://www.bitunix.com/api-docs/futures/trade/place_order.html),
[pending positions](https://www.bitunix.com/api-docs/futures/position/get_pending_positions.html),
[order detail](https://www.bitunix.com/api-docs/futures/trade/get_order_detail.html), and
[history trades](https://www.bitunix.com/api-docs/futures/trade/get_history_trades.html).

### Live quotes and candles

The bulk REST ticker omits bid and ask. Use the official public `tickers` WebSocket for
authoritative top-of-book and last price, splitting the active market set across connections
because Bitunix permits at most 300 subscriptions per connection. Refresh those subscriptions when
the active market-ID set changes. Determine cache freshness from the bounded local receipt time;
retain exchange timestamps only as quote provenance so future-skewed venue timestamps cannot keep a
stale quote eligible. A targeted ticker request may use one-row REST depth for at most eight missing
symbols as a bounded startup fallback, with the bid/ask midpoint labeled as its synthetic last
through ticker normalization and into snapshot provenance. Broad operation must fail instead of
fanning out one depth request per market or substituting bid/ask for a last trade. When WebSockets
are explicitly disabled, live market snapshots select this targeted REST path directly without
opening or waiting for a public ticker socket; unbounded bulk requests and requests above the
eight-symbol limit fail closed.

Bitunix klines return at most 200 rows. The live field names are inverted relative to their units:
`quoteVol` is base quantity and `baseVol` is quote notional; normalize `quoteVol` as CCXT base
volume so Passivbot's generic quote-volume calculation remains dimensionally correct. Missing,
non-finite, or negative base volume is invalid; an explicit zero remains valid. In live behavior,
`startTime` does not anchor a response; the venue tail-anchors at `endTime`. Derive each forward
page's `endTime` from `since + limit * timeframe`, filter to the requested bounds, sort ascending,
and deduplicate. This pagination supports live warmup, restart reconstruction, and runtime
indicators only. Bulk historical Bitunix data for backtesting or optimization is not a supported
source.

Primary references: [ticker WebSocket](https://www.bitunix.com/api-docs/futures/websocket/public/Tickers%20Channel.html),
[REST depth](https://www.bitunix.com/api-docs/futures/market/get_depth.html), and
[kline API](https://www.bitunix.com/api-docs/futures/market/get_kline.html).

## WEEX Futures

### V3 hedge-order contract

Problem:

1. WEEX V3 identifies entries and closes with the combination of `side` and
   `positionSide` (`LONG` or `SHORT`). Its regular-order request does not
   document a `reduceOnly` field.
2. The unified CCXT request accepts quantity in base-asset units even though
   market metadata also exposes `contractVal`; treating that metadata as a
   contracts-to-base multiplier under-sizes orders.
3. WEEX configures position and margin mode per symbol, not account-wide.
4. WEEX `SEPARATED` mode creates split positions and rejects ordinary
   Passivbot closes with `-1054` (position ID missing). `COMBINED` mode merges
   same-direction orders into the explicit long/short positions expected by
   Passivbot and supports regular quantity-based closes.

Handling in Passivbot:

1. Send explicit `positionSide`, `newClientOrderId`, and `timeInForce`; use
   `POST_ONLY` for configured post-only orders and do not send `reduceOnly`.
2. Keep WEEX `c_mult=1.0` at the Passivbot/CCXT boundary and use the exchange's
   base-quantity precision and minimum.
3. Read the symbol's current position and margin modes, set WEEX `COMBINED`
   position mode plus the selected cross/isolated margin mode when needed, then
   set leverage. Keep Passivbot's internal long/short hedge planning enabled;
   CCXT's generic `hedged` boolean calls this WEEX mode false.
4. Treat missing or ambiguous `positionSide` on orders and fills as an error;
   do not infer it from buy/sell alone.
5. Require the raw symbol configuration to explicitly report `COMBINED` or
   `SEPARATED`; CCXT's normalized `hedged=false` is not sufficient evidence
   because it also represents missing or unknown raw mode state.
6. In `COMBINED` mode, derive an open order's close-only effect exclusively from
   `side` plus `positionSide`. Although V3 order-query responses expose a
   `reduceOnly` field, ordinary V3 placement cannot set it and valid closes have
   been observed returning `reduceOnly=false`; it is therefore not authoritative
   for reconciliation.
7. Normalize V3 account `balance` to realized wallet balance by subtracting the
   same response row's `unrealizePnl`. Rust and backtests consume balance without
   open-position mark-to-market PnL; passing WEEX equity directly changes risk
   inputs and continuously invalidates churn evidence.

Primary references: [WEEX V3 place-order API](https://www.weex.com/api-doc/contract/Transaction_API/PlaceOrder),
[current-orders API](https://www.weex.com/api-doc/contract/Transaction_API/GetCurrentOrderStatus),
and [account-balance API](https://www.weex.com/api-doc/contract/Account_API/GetAccountBalance).

### API whitelist transport

Problem: WEEX API keys accept IPv4 whitelist entries, while a dual-stack host
may select its IPv6 source address for `api-contract.weex.com`. Public market
data then works, but private endpoints reject the authenticated request with
`-1056 ILLEGAL_IP` even when the host's public IPv4 address is whitelisted.

Handling: Both the REST and WebSocket WEEX CCXT clients use IPv4-only network
connectors. Keep the host's stable public IPv4 address in the API-key whitelist;
do not interpret `-1056` as a credential, signature, or Passivbot fill-history
failure.

Primary references: [WEEX V3 error codes](https://www.weex.com/api-doc/contract/ExampleOfErrorCode)
and [WEEX API integration preparation](https://www.weex.com/api-doc/spot/QuickStart/IntegrationPreparation).

### API-unsupported symbols

WEEX error code `-1058` means that the trading pair is not supported through the API. The WEEX
connector classifies only this exact structured response for the shared temporary symbol-suspension
policy. It must not classify raw message text, nearby codes such as `-1056`, or generic permission
errors. After the configured cooldown expires, the next required exchange configuration or order
write retries the symbol; another `-1058` begins a fresh cooldown.

### Market data and CCXT compatibility

Problem:

1. WEEX's 24-hour futures ticker payload does not provide a live bid and ask,
   while its V3 book-ticker payload provides bid and ask but no last-trade price.
2. WEEX configuration mutations return the documented envelope
   `code=200, msg=success`, which CCXT 4.5.66 incorrectly classifies as an
   exchange error merely because `msg` is present.

Handling in Passivbot:

1. Fetch live quotes from the V3 contract book-ticker endpoint and reject
   missing, non-finite, non-positive, or crossed quotes. Derive `last` as the
   top-of-book midpoint and label the resulting market snapshot source
   `weex_book_ticker_mid`; downstream price consumers must not report it as a
   generic ticker or authoritative last trade.
2. Accept only the exact documented success envelope in the WEEX adapter;
   delegate every other response to CCXT's normal error mapping.

### Live OHLCV pagination and indicator inputs

Problem:

1. WEEX's recent kline endpoint returns at most 1,000 rows, includes the
   currently forming candle, and tail-anchors the response instead of honoring
   an old `since` value. Only 999 finalized candles are therefore available
   from one recent request.
2. The historical endpoint returns at most 100 rows and tail-anchors an
   over-wide time range. An unbounded request can silently skip the beginning
   of a live warmup window.
3. CCXT exposes WEEX candle volume as base volume. Passivbot's quote-volume EMA
   therefore uses the generic approximation `base_volume * (high + low + close) / 3`,
   not raw exchange quote turnover.

Handling in Passivbot:

1. Page older 1m and 1h live warmup ranges forward through bounded 100-candle
   historical windows, then switch to the recent endpoint only when its
   finalized tail covers the remaining range.
2. Exclude the forming candle and require exact finalized-candle coverage before
   publishing close, volume, quote-volume, or volatility EMAs.
3. Require exact 1m coverage before rebuilding trailing extrema or extending an
   HSL replay cache. Missing coverage marks trailing state unavailable or makes
   HSL fall back to its authoritative full replay path.
4. Keep bulk historical WEEX backtest downloading out of scope; this bounded
   paging exists for live warmup, restart reconstruction, and runtime indicators.

Primary references: [WEEX V3 current klines](https://www.weex.com/api-doc/contract/Market_API/GetKlines)
and [WEEX V3 historical klines](https://www.weex.com/api-doc/contract/Market_API/GetHistoryKlines).

### Fill-history retention and pagination

Problem: WEEX returns at most 100 trade-detail rows per request, permits at most
seven days per query, and retains up to 365 days.

Handling:

1. Split requested history into seven-day windows. Recursively bisect every
   full 100-row response into disjoint time windows until each response proves
   completeness below the limit; fail closed if one millisecond is saturated.
   Do not assume the endpoint returns oldest-first rows.
2. Preserve exchange trade and order IDs, explicit position side, realized PnL,
   and fees; enrich missing Passivbot client-order IDs from order detail.
3. Keep WEEX historical 1m backtest-data downloading out of the live adapter;
   it is not a supported WEEX data source in this release.

Primary reference: [WEEX V3 trade-detail API](https://www.weex.com/api-doc/contract/Transaction_API/GetTradeDetails).

## General Guidance

1. Check raw exchange payloads when CCXT abstraction is insufficient.
2. Treat intra-page gaps and inter-page gaps differently.
3. For missing data incidents, verify source data before changing logic.

## Validation

- Exercise actual CCXT/raw request construction for payload, header, broker, and client-ID changes.
- Use sanitized response fixtures for normalization and ambiguous-side cases.
- Test pagination overlap, deduplication, and retention boundaries with multi-page fixtures.
- Keep authenticated exchange checks outside the default suite and require explicit approval.

## Key Code And Tests

- `src/exchanges/`
- `src/fill_events_manager.py`
- `tests/exchanges/`
- `tests/ccxt_upgrade/`
