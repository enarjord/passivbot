# Exchange Integration Contracts

Only task-specific, high-impact contracts and quirks are listed here. Authenticated verification
requires explicit user approval; prefer offline request-construction tests.

## Supported Live-Exchange Boundary

The supported production live connectors are Binance, Bybit, Bitget, OKX, Gate.io, KuCoin,
Hyperliquid, and WEEX. The fake connector is an offline deterministic test harness, not an exchange.

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
must use an explicit supported-connector allowlist and leave the generic fallback on its prior
behavior until that venue receives a connector-specific contract audit.

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

### KuCoin hedge-mode refresh

Problem:

1. `set_position_mode(True)` is trading-critical setup, but broad no-op swallowing can hide a real one-way/hedge mismatch.
2. KuCoin order and fill payloads must carry `positionSide` in hedge mode; otherwise a both-sides-open account cannot safely infer an order's position side.

Current handling and planned prerequisite:

1. Treat current same-mode success as success (`code=200000`, `data.positionMode=1`).
2. Let unknown `set_position_mode` failures raise unless a verified KuCoin no-op code is added with a targeted test.
3. Current runtime prefers explicit `info.positionSide`/`info.posSide` but may still fall back to
   current-position inference. That fallback is not restart-stable enough for exact churn-gate
   reconciliation.
4. Before the churn gate is enabled for KuCoin, never infer a resting order's position side from the
   current position. Require explicit `info.positionSide`/`info.posSide` in hedge mode; in effective
   one-way mode, derive and verify `position_side` from the authoritative order side plus
   `reduceOnly` tuple.

### OHLCV limit behavior + sparse-minute markets

Problem:

1. Effective page size is 200 rows.
2. Illiquid symbols legitimately have missing trade minutes.

Handling:

1. Page with `limit=200`.
2. Overlap page boundaries by 1 candle to validate inter-page gaps.

## Bitget Futures

### Bitget hedge-mode refresh

Problem:

1. Bitget hedge-side attribution depends on `posSide`/`holdSide` payload fields.
2. Broadly swallowing hedge-mode setup errors can mask an unsafe one-way/hedge mismatch.

Handling:

1. Treat current same-mode success as success (`code=00000`, `data.posMode=hedge_mode`).
2. Let unknown `set_position_mode` failures raise unless a verified Bitget no-op code is added with a targeted test.
3. Require explicit side-disambiguating payloads for order/fill normalization instead of defaulting to long; open orders should carry `posSide`, while fills may use `tradeSide`/`side`/`posMode`.

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
3. Keep classic Bitget v2/mix `tradeSide`/`reduceOnly` handling separate.

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

Planned churn-gate prerequisite (not current runtime handling):

1. In effective long/short mode, normalize close-only effect from the documented `side` plus
   `posSide` action tuple.
2. In effective one-way/net mode, require an authoritative native `reduceOnly` value and verify
   one-way `position_side` against side plus close-only effect.
3. Prefer the raw exchange `info` field over a CCXT top-level default when proving close-only
   semantics.

Current runtime does not yet implement this complete close-only normalization. The implementation
PR must land the adapter change and focused fixtures before enabling the gate for OKX.

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

Primary reference: [WEEX V3 place-order API](https://www.weex.com/api-doc/contract/Transaction_API/PlaceOrder).

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
