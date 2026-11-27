# OKX Futures Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## WebSocket transport

Official production and demo WebSocket endpoints use port 443. Pinned CCXT defaults
may still include retired port 8443; update those official URLs before applying
operator endpoint overrides. A retired-port `notice` with code `64010` can otherwise
be classified by CCXT as an error during authentication and invalidate its pending
login future. Apply the transport correction to both private order and public candle
sessions; do not suppress arbitrary error messages or bypass authentication.

Primary reference: [OKX WebSocket port migration](https://www.okx.com/docs-v5/log_en/).

## Account wallet balance

Use the unfiltered REST trading-account balance for one account-wide USD-equivalent
wallet value. Subtract each currency's unrealized PnL from `totalEq` after
converting it to USD with that detail row's `eqUsd / eq`. Detail `upl` includes
both cross and isolated PnL. Account `upl` covers only cross margin. Use every
currency detail, regardless of its collateral flag, to match the scope of
`totalEq`.

A missing or malformed account total, missing or malformed detail, duplicate
currency, unknown detail PnL, or unproven conversion for nonzero PnL makes the
balance unavailable. An explicit zero PnL needs no conversion. An empty detail
list represents a zero wallet only with explicit zero account equity and gross
position notional and no contradictory reported isolated equity or account PnL.
A unified quote-currency total cannot prove the account-wide balance.
Staged and direct capture use the same parser.

Primary references: [OKX trading-account balance](https://www.okx.com/docs-v5/en/#trading-account-rest-api-get-balance)
and [OKX equity and PnL formulas](https://www.okx.com/help/iii-single-currency-margin-cross-margin-trading#asset-fields).

## Long/short-mode close semantics

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

## Net-mode fill attribution

Use native transaction-detail subtypes that identify an opening or closing position side,
and reject conflicts with the execution side, reduce-only evidence, or realized PnL.
An explicit reduce-only flag or a finite nonzero realized PnL identifies a reduction when
the fill lacks a side-specific subtype. A generic buy/sell subtype, a false reduce-only
flag, and zero realized PnL cannot distinguish an opening trade from a break-even close.
Reject unresolved history before advancing fill coverage. Recheck the preserved raw
evidence when loading existing net-mode cache rows and use the cache rebuild path when
the original sides cannot be recovered.

[OKX transaction details](https://www.okx.com/docs-v5/en/#order-book-trading-trade-get-transaction-details-last-3-days)
