# Bybit Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## Broker referer header

Problem: Bybit broker attribution depends on the `Referer` header on order POST requests. CCXT derives this from `options["brokerId"]`, whose default may not be Passivbot.

Handling:

1. Set Bybit CCXT client `options["brokerId"]` from `broker_codes.hjson`.
2. Test that a signed `v5/order/create` request contains `Referer: passivbotbybit`.

## Closed-PnL pagination mismatch

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
