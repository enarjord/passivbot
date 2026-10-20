# Binance Portfolio Margin: USDT UM perpetuals

Passivbot supports a restricted Binance Portfolio Margin (PM) account through
`BinancePortfolioBot`. The account must use only USDT collateral and USDT-settled
USDⓈ-M perpetuals. Cross-margin borrowing, accrued margin interest, negative
balances, non-USDT collateral or settlement, and COIN-M positions are unsupported
and cause an explicit failure before the balance is admitted to trading.

This connector does not enable general multi-asset PM, Portfolio Margin Pro,
spot/margin trading, COIN-M trading, or options. Do not share its collateral pool
with activity outside the supported domain.

## Account selection

Set `"exchange": "binance"` and `"account_type": "portfolio_margin"` in the
account entry in `api-keys.json`; `"pm"` is also accepted. Select that account with
`live.account.user`. Account type belongs to the account definition, not strategy config.
Without the PM selector, the existing Binance connector is used.

## Balance and account truth

Every balance read fetches both the complete PAPI asset balances and all COIN-M
position-risk rows. A zero COIN-M wallet or zero unrealized PnL alone does not prove
that no COIN-M position exists. Missing, duplicate, inconsistent, or non-finite
required facts remain unavailable; they never become a fabricated zero balance.

Within the supported domain, the USDT asset's `totalWalletBalance` is the wallet
balance. The connector verifies its debt-free components against
`crossMarginAsset + umWalletBalance` and verifies the cross-margin asset against
`crossMarginFree + crossMarginLocked`. UM unrealized PnL is kept separate so the
ordinary Passivbot equity calculation adds it once.

PAPI `accountEquity` and `actualEquity` are USD-valued equity figures; the former
also applies collateral rates. They are not the USDT wallet-balance input.
No FX conversion, collateral-haircut estimate, or subtraction of only the bot's
own positions is used to reinterpret them.

## Requests and execution

Raw UM position-risk rows feed the existing Binance position normalizer. Income
history uses PAPI UM income endpoints for both realized PnL and commissions.
Orders, cancellations, open orders, trades, leverage, and hedge-mode changes use
CCXT's PM routing. Broker attribution and signed quantity/position-side semantics
remain the existing Binance contracts. Per-symbol margin-mode writes are omitted;
required leverage failures propagate to the configuration retry policy.

Protective startup checks the PAPI UM hedge-mode endpoint. Private order updates
use the existing REST polling path. Detailed per-asset balance-composition
monitoring currently reports explicit unavailability for the raw PM snapshot.

## Offline validation

```sh
PYTHONPATH=src python -m pytest tests/test_binance_portfolio.py tests/test_setup_bot_fallback.py
```

Tests use synthetic responses and synthetic signing inputs. They intercept the
installed CCXT transport after signing and verify PAPI endpoint selection,
request fields, broker attribution, and failure propagation without network or
account operations. They do not claim authenticated or live execution validation.

Primary references: [PAPI account balances and account information](https://developers.binance.com/en/docs/catalog/advanced-trading-derivatives-trading-portfolio-margin/api/rest-api/account),
[PAPI trade endpoints](https://developers.binance.com/en/docs/catalog/advanced-trading-derivatives-trading-portfolio-margin/api/rest-api/trade).
