# Suite Examples

This page shows practical examples of Passivbot's flattened suite configuration under `backtest`.

Suite mode is **off by default** in both:

- `src/config/schema.py`
- `configs/examples/default_trailing_grid_long_npos10.json`

That is intentional. A fresh user should start with a single backtest or optimize run, then enable
suite mode when they specifically want multi-scenario evaluation.

## When To Use Suite Mode

Use suite mode when you want to evaluate the same config across multiple scenarios in one run, for
example:

- compare the same config across different exchanges
- compare different date windows
- compare different approved-coin sets
- compare a base config against a few targeted parameter overrides

Do not use suite mode by default just because it exists. It increases runtime and adds result
structure that is unnecessary for simple one-off backtests.

## Basic Shape

Suite configuration lives directly under `backtest`:

```json
{
  "backtest": {
    "suite_enabled": true,
    "aggregate": {
      "default": "mean"
    },
    "scenarios": [
      {"label": "base"}
    ]
  }
}
```

Supported scenario keys:

- `label`
- `start_date`
- `end_date`
- `coins`
- `ignored_coins`
- `exchanges`
- `coin_sources`
- `overrides`

## Example 1: Exchange Comparison

Compare the same bot config against different exchanges:

```json
{
  "backtest": {
    "suite_enabled": true,
    "aggregate": {"default": "mean"},
    "exchanges": ["binance", "bybit", "gateio"],
    "scenarios": [
      {"label": "binance_only", "exchanges": ["binance"]},
      {"label": "bybit_only", "exchanges": ["bybit"]},
      {"label": "gateio_only", "exchanges": ["gateio"]}
    ]
  }
}
```

Use this when you want to see whether one exchange's data/feed quality or market behavior changes
the result materially.

## Example 2: Date-Window Comparison

Compare one config across different market regimes:

```json
{
  "backtest": {
    "suite_enabled": true,
    "aggregate": {"default": "mean"},
    "scenarios": [
      {"label": "bull_2021", "start_date": "2021-01-01", "end_date": "2021-12-31"},
      {"label": "bear_2022", "start_date": "2022-01-01", "end_date": "2022-12-31"},
      {"label": "mixed_2024", "start_date": "2024-01-01", "end_date": "2024-12-31"}
    ]
  }
}
```

This is useful when a config looks strong overall but may only be good in one regime.

## Example 3: Coin-Set Comparison

Compare a large-cap basket against a smaller custom subset:

```json
{
  "backtest": {
    "suite_enabled": true,
    "aggregate": {"default": "mean"},
    "scenarios": [
      {
        "label": "large_caps",
        "coins": ["BTC", "ETH", "SOL", "BNB", "XRP"]
      },
      {
        "label": "alts",
        "coins": ["XMR", "LINK", "AVAX", "SUI", "DOT"]
      }
    ]
  }
}
```

## Example 4: Parameter Sensitivity

Compare a base config against a small number of targeted runtime overrides:

```json
{
  "backtest": {
    "suite_enabled": true,
    "aggregate": {"default": "mean"},
    "scenarios": [
      {"label": "base"},
      {
        "label": "twel_1_5",
        "overrides": {
          "bot.long.total_wallet_exposure_limit": 1.5
        }
      },
      {
        "label": "npos_5",
        "overrides": {
          "bot.long.n_positions": 5
        }
      }
    ]
  }
}
```

This is useful for directional exploration before a larger optimize run.

## Example 5: External Suite Config File

Keep your main config simple and layer suite scenarios only when needed:

```bash
passivbot backtest configs/live/my_config.json --suite y --suite-config configs/suites/regimes.json
```

This is often the cleanest workflow when:

- the same live/backtest config is reused in many contexts
- suite definitions are large or experimental
- you want to keep the base config focused on one normal run

## Aggregation Guidance

`backtest.aggregate` controls how scenario metrics are combined in `suite_summary.json` and in
optimizer suite scoring.

Good defaults:

```json
{
  "aggregate": {
    "default": "mean",
    "drawdown_worst_hsl": "max",
    "drawdown_worst_mean_1pct_hsl": "max",
    "peak_recovery_hours_hsl": "max",
    "position_held_hours_max": "max"
  }
}
```

Interpretation:

- use `mean` for general performance metrics
- use `max` for risk and recovery metrics where the worst scenario matters more than the average

## Recommended Workflow

1. Start with a single backtest or optimize run.
2. Only enable suite mode once you know what comparison you want to make.
3. Keep scenario labels short and meaningful.
4. Keep the number of scenarios small at first.
5. Use suite mode to answer a concrete question, not as a default habit.

Related docs:

- [Backtesting](backtesting.md)
- [Optimizing](optimizing.md)
- [Configuration](configuration.md)
- [Config Workflow](config_workflow.md)
