# Gate.io Futures Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## Exchange identity boundary

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

## Listing-history discovery

Discover the first available daily candle through forward pages of at most 1,999
rows from the supported history start. A single pre-listing page can be empty
without proving that a newer contract has no history; advance to the next disjoint
page until the first candle or the current time. Reuse the market snapshot already
loaded by discovery rather than repeating market discovery on its fetch client.
Cancellation owns every child task through completion before closing transport.

## Multi-currency balance semantics

Gate's `cross_available` is spendable margin, not stable account equity. Resting
orders move value between `cross_available` and `cross_order_margin`, while open
positions use `cross_initial_margin`. For multi-currency margin accounts, select
the unique futures-account row whose settle `currency` matches the bot quote
(CCXT wraps Gate's single-account response as `info=[account]`; never prefer an
unrelated first row when currency-matched candidates exist). A singleton row is
accepted only when `currency` is omitted; an explicit mismatched currency fails
closed. Derive the strategy
wallet balance from that same authoritative row as
`cross_available + cross_order_margin + cross_initial_margin -
cross_unrealised_pnl`. Require all four finite fields. Removing unrealized PnL
preserves wallet-balance semantics because Passivbot adds position PnL separately
when deriving equity. Do not feed `cross_available` alone to Rust, because
ordinary order reservation would then resize ideal orders and create
reconciliation churn. Classic accounts continue using CCXT's quote-currency
total.

Balance-change events may publish a bounded Gate composition diagnostic for the
settle currency: reconstructed wallet amount, available margin
(`cross_available`), reserved margin (`cross_initial_margin +
cross_order_margin`), and unrealized PnL. Composition is observability-only and
must not replace the trading wallet balance.

## Per-symbol leverage initializes the position risk limit

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

## Contract order text must start with `t-`

Problem:

1. Gate.io contract order `text` rejects values that do not start with `t-`.
2. CCXT prefixes `clientOrderId` into `text=t-...`, but raw `params["text"]` can overwrite that transformed value.

Handling:

1. Pass Passivbot custom order ids as `clientOrderId`, not raw `text`.
2. Keep broker attribution in the `X-Gate-Channel-Id` header.
3. Keep the Passivbot order-type marker inside the custom id; decoding accepts the marker inside Gate.io's `t-...` text.

## Public 1m OHLCV recent-window limit

Problem: Gate.io rejects old 1m OHLCV requests with `Candlestick too long ago. Maximum 10000 points recently are allowed`.

Handling:

1. Do not pass CCXT `until`; page forward by `since + limit`.
2. Clip 1m historical fetches to the recent-window bound and mark older spans as `no_archive`.
3. Require external OHLCV source data or another candle source for older Gate.io backtests.
