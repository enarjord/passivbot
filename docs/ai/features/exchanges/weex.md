# WEEX Futures Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## V3 hedge-order contract

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

## API whitelist transport

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

## API-unsupported symbols

WEEX error code `-1058` means that the trading pair is not supported through the API. The WEEX
connector classifies only this exact structured response for the shared temporary symbol-suspension
policy. It must not classify raw message text, nearby codes such as `-1056`, or generic permission
errors. After the configured cooldown expires, the next required exchange configuration or order
write retries the symbol; another `-1058` begins a fresh cooldown.

## Market data and CCXT compatibility

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

## Live OHLCV pagination and indicator inputs

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
3. Require exact 1m coverage before rebuilding trailing extrema. HSL restart reconstruction uses
   the documented 1m/5m/15m/1h resolution ladder for the older leading prefix and remains
   authoritative for fill timestamps, realized PnL, fees, and episode boundaries.
4. Keep bulk historical WEEX backtest downloading out of scope; this bounded
   paging exists for live warmup, restart reconstruction, and runtime indicators.

Primary references: [WEEX V3 current klines](https://www.weex.com/api-doc/contract/Market_API/GetKlines)
and [WEEX V3 historical klines](https://www.weex.com/api-doc/contract/Market_API/GetHistoryKlines).

## Fill-history retention and pagination

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
