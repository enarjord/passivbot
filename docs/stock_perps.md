# Stock Perpetuals Trading Guide

Passivbot supports Hyperliquid builder-deployed perpetuals (HIP-3). These include markets
that reference stocks, indices, and other traditional assets. The builder namespace
identifies the contract, for example `xyz:TSLA`.

## Live account and margin requirements

- Use a Hyperliquid account in `unifiedAccount` or `portfolioMargin` mode. Startup fails
  when an unsupported account mode has HIP-3 markets in its approved universe or has
  existing HIP-3 positions or open orders.
- New entries use cross margin on cross-capable markets. Market metadata determines
  whether a symbol can trade in cross mode. Isolated-only markets are ineligible for new
  positions under the current entry policy.
- Check the builder's access requirements as well as the exchange account and agent-wallet
  setup. A margin error alone does not establish why an order was rejected.

Passivbot derives wallet balance by subtracting unrealized PnL from the applicable
account equity, including core and HIP-3 PnL for the unified balance path. It uses wallet
balance for the exposure budget. `total_wallet_exposure_limit_ratio = 1.0` on a side
with a 100 USDC balance gives a 100 USDC side-level notional budget. Leverage affects
margin requirements, while the configured ratio controls the bot's exposure budget.

## Configuration

Copy the maintained template as described in [Config Workflow](config_workflow.md). Set
`live.account.user` to the configured Hyperliquid account, set `bot.portfolio.hedge_mode`
to `false`, and review the active sides and their exposure. Add the intended contracts to
`bot.global.universe.approved_coins` or a side-specific universe. Use a namespaced
identifier such as `xyz:TSLA` to distinguish contracts from different builders. Crypto
and eligible HIP-3 perpetuals can share the same universe.

This fragment shows the relevant fields to apply to the private config.

```json
{
  "config_version": "v9.0.0",
  "bot": {
    "portfolio": {
      "hedge_mode": false
    },
    "long": {
      "enabled": true,
      "universe": {
        "approved_coins": ["xyz:TSLA"]
      }
    }
  },
  "live": {
    "account": {
      "user": "hyperliquid_01"
    }
  }
}
```

Check the resolved market and the [minimum effective cost](min_effective_cost.md). The
required notional depends on market metadata and quantity precision. When the intended
first order is too small, a minimum-sized order can exceed the strategy's intended
allocation. Review this before changing `bot.<side>.universe.filter_by_min_effective_cost`.

See [Hyperliquid setup](hyperliquid_guide.md) and [Running live](live.md) for the full
config review and offline preflight workflow.

## Data Sources for Backtesting and Live Trading

Use native Hyperliquid candles for backtests of actual perp behavior. History availability
depends on the market and the sources available to the data preparation path. An explicit
`backtest.ohlcv_source_dir` supplies external candle files in the
[documented format](backtesting.md#external-ohlcv-source-dir).

For supported stock symbols, Hyperliquid archive acquisition can try an underlying-asset
data provider. The default Yahoo Finance adapter requires a separate `yfinance` install.
Passivbot also recognizes configured providers such as Finnhub and Alpha Vantage through
the private `tradfi` entry in `api-keys.json`. Provider coverage varies. Record the data
source when comparing results.

To manually probe a provider from a source checkout, run
`PYTHONPATH=src python src/tools/probe_tradfi_providers.py --provider yfinance`.
This command makes real market-data requests and uses the selected provider's
credentials when required. The offline test suite uses mocked responses.

### Synthetic Candles During Market Closure (Backtesting Model)

Underlying-asset history does not capture perpetual funding or the builder's pricing while
the underlying market is closed. In backtests with stock-perp calendar context, eligible
missing closed-market minutes can be represented as synthetic candles with
`high = low = close = previous close` and zero volume. These minutes remain tradable in
the simulator. Strict price comparisons prevent a resting order priced exactly at the
flat price from filling on that candle.

Flat synthetic candles omit overnight and weekend price movement. The next actual candle
can carry the full price change, and zero synthetic volume affects volume-based Forager
ranking. Review the synthetic-candle share and the transition to native history before
using the result to evaluate a live config.

[Hyperliquid HIP-3 reference](https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals)
