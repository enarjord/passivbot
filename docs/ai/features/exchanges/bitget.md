# Bitget Futures Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## Bitget hedge-mode refresh

Problem:

1. Bitget hedge-side attribution depends on `posSide`/`holdSide` payload fields.
2. Broadly swallowing hedge-mode setup errors can mask an unsafe one-way/hedge mismatch.

Handling:

1. Treat current same-mode success as success (`code=00000`, `data.posMode=hedge_mode`).
2. Let unknown `set_position_mode` failures raise unless a verified Bitget no-op code is added with a targeted test.
3. Normalize against the actual exchange account mode, not `bot.portfolio.hedge_mode`, which only controls
   simultaneous strategy exposure.
4. Require explicit side-disambiguating payloads for order/fill normalization instead of defaulting to long; open orders should carry `posSide`, while fills may use `tradeSide`/`side`/`posMode`.

Classic v2 hedge fills with `tradeSide=open` or `close` identify the position through the wire
`side` field. Buy identifies a long position and sell identifies a short position. Derive the
execution direction from the action and position. A one-way liquidation, settlement or ADL buy
closes a short position and its sell counterpart closes a long. Replay retained Classic executions before reconstructing cached
positions and trailing anchors. [Bitget Classic contract trade API](https://www.bitget.com/docs/catalog/classic-contract-trade/classic-contract-trade)

## UTA / Elite hedge-mode order direction

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

## UTA insufficient margin on entry

Bitget's UTA [REST code 25203](https://www.bitget.com/docs/uta/error-code/restapi)
is a definitive insufficient-margin rejection. For a position-increasing create,
record a rejected order and delay new submissions for the same symbol and position
side for 60 seconds. Exclude matching entries before create-batch capacity is
allocated, and recheck the interval at the connector. Retry only from a fresh
plan after expiry. The interval is local to the current bot instance. Submitted orders still require account
confirmation. This rejection is excluded from the bot-wide restart budget and the sustained
create-failure guard. Closes,
cancellations, and other failures retain their ordinary handling.

## UTA public candle history routing

Bitget account-mode detection enables UTA routing for authenticated account and order calls. Public
OHLCV history is account-independent and must explicitly use classic futures market-data routing;
the UTA candle endpoint may expose a shorter history and return an empty prefix which is available
from the public v2 futures history endpoint. Pass `uta=false` only on Bitget OHLCV requests. Do not
disable UTA on the shared CCXT session or on private account/order requests.

## `since` is effectively exclusive for OHLCV paging

Problem: naive paging can miss first candle in each page.

Handling:

1. Overlap boundaries by 1 candle.
2. Back up initial `since` by one candle on pagination start.
