# Binance Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).

## Portfolio Margin

`account_type=portfolio_margin` (or `pm`) selects the Binance PM adapter while
retaining the canonical exchange identity `binance`. Its supported domain is
USDT collateral and USDT-settled UM perpetuals, without margin debt, interest,
negative balances, other collateral/settlement assets, or CM positions.

Every balance snapshot requires complete raw PAPI balance rows and an independent
all-CM-position-risk response. Within that domain, return USDT `totalWalletBalance`
only after validating its debt-free wallet components. USD-valued account equity
is not a wallet balance and must not be passed to Rust as one. Missing, malformed,
non-finite, inconsistent, or unsupported facts propagate or fail explicitly.

Raw UM position rows feed Binance normalization; realized-PnL and commission
income queries use PAPI UM history. Private orders use existing REST polling.
Preserve Binance broker fields, hedge position sides, and required configuration
failure propagation at the actual CCXT request boundary. See the
[PM guide](../../../exchanges/binance_portfolio_margin.md) for validation and limitations.
