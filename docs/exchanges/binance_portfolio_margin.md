# Binance Portfolio Margin

Binance Portfolio Margin (PM) unifies risk across spot, margin, USDⓈ-M futures, COIN-M futures,
and options into a single cross-collateral account, reachable through a separate `papi*` endpoint
family. A sub-account is either fully on PM or fully on regular margin; there is no per-position
mix. `BinancePortfolioBot` adds support for running Passivbot's existing grid engine on a PM
account unchanged, so a PM account can run Passivbot alongside other PM-native activity (for
example a separate spot+perp hedge) without a second, non-PM sub-account.

v1 covers UM futures (the grid engine's unchanged symbol universe) and margin/spot. CM futures and
options under PM are out of scope.

## Credentials

Add `"account_type": "portfolio_margin"` to the account's entry in `api-keys.json`, alongside the
existing `exchange` field:

```json
{
  "binance_pm_01": {
    "exchange": "binance",
    "account_type": "portfolio_margin",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET"
  }
}
```

This is an account-level credential property, not a strategy parameter, so it lives in
`api-keys.json` rather than a new `live.*` config field. Omitting it (or any value other than
`portfolio_margin`/`pm`) uses the regular, non-PM `BinanceBot` connector unchanged — existing
accounts are unaffected.

Use `live.user = "binance_pm_01"` in your config, same as any other account.

## What's different from regular Binance futures

Most hooks are unchanged: CCXT's installed `binance.py` already routes position fetching, order
creation/cancellation, open-order and fill queries, leverage setting, and hedge-mode setting
through the matching `papi*` endpoint once `options["portfolioMargin"]` is set, with response
field names matching the regular endpoint. `BinancePortfolioBot` sets that option once in
`create_ccxt_sessions()`; broker-code attribution from `BinanceBot.create_ccxt_sessions()` is
preserved unchanged.

The real differences:

- **Balance.** PM's balance shape differs from a regular futures account's single
  `totalCrossWalletBalance` field. `BinancePortfolioBot` reads `papiGetAccount()`'s `accountEquity`
  directly — a single, precomputed, account-wide figure that already includes unrealized PnL and
  already accounts for multi-asset collateral by construction.
- **Margin mode.** PM risk is unified account-wide; there is no per-symbol isolated/cross toggle to
  set, and CCXT's `set_margin_mode` has no PM routing to call. `update_exchange_config_by_symbols`
  sets leverage only.
- **Hedge-mode check.** CCXT's `fetch_position_mode()` has no PM routing and would read the wrong
  (regular futures) account state. Protective startup calls the raw `papiGetUmPositionSideDual()`
  endpoint directly instead.
- **No realtime order stream.** No CCXT Pro websocket class exists for PM, and the regular
  `binanceusdm` watch stream is backed by the non-PM listen-key flow, so it would not reflect a PM
  account's real state. `can_watch_orders()` returns `False`, falling back to REST polling — an
  existing, first-class supported path for exchanges without a watch stream, not PM-specific
  degradation.

## Validation

```sh
PYTHONPATH=src python -m pytest tests/test_setup_bot_fallback.py
```

Covers `setup_bot()` dispatching to `BinancePortfolioBot` on `account_type: portfolio_margin` and
confirms exchange-keyed gates (for example the order-churn gate) are unaffected by account type.

Authenticated validation (balance reads, hedge-mode and leverage setting, order placement and
cancellation across both the UM and margin legs, and a full `setup_bot()`-based boot) was run
against a funded Portfolio Margin sub-account; see `src/exchanges/binance_portfolio.py`'s class
docstring for the itemized account. Validating a new account should use a dedicated,
small-capital PM sub-account before sizing up, same as any other live connector change.
