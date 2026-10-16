# Binance Portfolio Margin connector design

The implemented boundary is documented in the
[user guide](../exchanges/binance_portfolio_margin.md) and the
[exchange contract](../ai/features/exchange_integrations.md#binance-portfolio-margin).

`BinancePortfolioBot` subclasses `BinanceBot`. An explicit account-level PM selector
keeps the ordinary Binance path unchanged. CCXT's `portfolioMargin` option routes
supported unified UM calls to PAPI; raw overrides are used where the inherited
normalizer needs native rows or CCXT does not route the required account query.

Wallet balance remains denominated in the configured settlement currency and
excludes unrealized PnL. A general PM USD-equity number cannot satisfy that contract
without a proven collateral, debt, FX, and unrealized-PnL normalization model.
The initial connector therefore admits only debt-free USDT collateral with
USDT-settled UM perpetuals and independently proves absence of COIN-M positions.
It rejects unsupported account facts instead of estimating a balance.

The existing Binance fill-event fetcher retains its pagination, fee, and coverage
policies. Its account-specific raw income endpoint is selected explicitly by the
connector so PM reads cannot silently query ordinary futures history.

Orders reuse Binance broker attribution, hedge-mode position sides, reconciliation,
and HSL write coordination. No trading decisions or exchange-action permissions
are reimplemented in the adapter. REST polling uses the existing account refresh
policy; a PM private WebSocket transport and broader collateral support require
separate contracts and validation.

Offline synthetic tests exercise native response normalization, strict account
boundaries, recovery/failure behavior, and installed CCXT signed request construction.
They provide reproducible repository evidence without requiring account access.
