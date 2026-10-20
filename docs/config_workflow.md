# Config Workflow

This is the recommended way to work with Passivbot configs on the current config system.

## Source Of Truth

- The canonical hardcoded defaults live in `src/config/schema.py`.
- The example config `configs/examples/default_trailing_martingale_long.json` provides the maintained default strategy profile.
- New configs use the final `config_version: "v9.0.0"` layout. Normal loading rejects
  unknown/retired fields and never migrates policy implicitly. Use the explicit
  [v8 converter](v8_to_v9_migration.md) for released v8 inputs, with the required semantic
  choices and fresh backtests. Config schema and [package release versions](releases.md)
  are separate. Unreleased intermediate schemas, `pb_multi` and pre-v8 layouts are unsupported.
- Retained strategies are `trailing_martingale` and `ema_anchor`. The v7 strategy and its
  dedicated migration command have been removed.
- If you run `passivbot live`, `passivbot backtest`, or `passivbot optimize` without a config path, Passivbot starts from the in-code defaults in `src/config/schema.py`.

## Recommended Workflow

1. Copy `configs/examples/default_trailing_martingale_long.json` to a new file under
   `configs/private/`, which is ignored by Git.
2. Edit that new file for your account, market universe, and strategy changes.
3. Use `passivbot backtest` first.
4. Use `passivbot optimize` if you want to tune parameters or compare alternatives.
5. Use `passivbot live` only after the config has been tested.
6. Expect live runs to write a timestamped log file under `logs/` by default unless you set `logging.persist_to_file = false`, and to refresh `logs/{user}.log` as a stable alias to the current run. Without Windows symlink privileges, the stable path is a text pointer which Passivbot's monitor tooling follows automatically.

Example:

```bash
mkdir -p configs/private
cp configs/examples/default_trailing_martingale_long.json configs/private/my_config.json
passivbot backtest configs/private/my_config.json -s BTC -sd 2025 --suite n
passivbot optimize configs/private/my_config.json -s BTC -sd 2025 -c 4 --suite n
passivbot live configs/private/my_config.json
```

## Best Practices

Use `passivbot tool clean-config input.json output.json` to produce a full schema-aligned config,
or add `--mode live` for a lean live export. Formatting only is available with `--mode format`.
Inputs stay untouched by default; replacement requires `--in-place`. See
[Config cleanup and formatting](tools.md#config-cleanup-and-formatting) for bulk depth, HJSON,
dry-run/check modes and migration boundaries.

Backtests accept saved optimizer and backtest result files directly, including records
with a nested `config` object and known result metadata. Only known result metadata is
removed before strict validation. Recorded dataset membership is provenance;
live/scenario coin settings still control data preparation.

- Keep one normal JSON or HJSON config per strategy/account instead of relying on many CLI overrides.
- Use CLI overrides for temporary experiments, not as your main configuration workflow.
- Treat `bot`, `live`, `backtest`, and `optimize` as one config file with command-specific sections.
- Keep shared execution/risk settings in `live`. Backtests inherit `market_orders_allowed`, `market_order_near_touch_threshold`, `fee_pct_fallback`, and `fee_pct_sanity_abs_max` directly from `live`; these fields are not accepted under `backtest`.
- Risk policy belongs to its consumers: each HSL policy has its own slow and ratio-smoothing
  spans; global unstuck uses `bot.global.unstuck.memory_ema_span_minutes`; global loss gate uses
  `bot.global.loss_gate.memory_ema_span_minutes`. Acquisition uses the separately bounded
  live bootstrap cap. A rolling window is not an equivalent EMA span.
- Keep new configs on the canonical schema. Use nested active strategy paths and explicit
  [migration reports](v8_to_v9_migration.md) for released-v8 changes.
- Use `coin_overrides` for per-coin exceptions instead of cloning whole configs for minor differences.
- Leave `logging.persist_to_file = true` for normal live operations so each bot run has a durable logfile under `logs/` and Passivbot's monitor tooling can follow the stable `logs/{user}.log` alias or Windows pointer.

## What The Default Profile Is

The default profile mirrored by `configs/examples/default_trailing_martingale_long.json` is:

- trailing martingale style configuration
- long enabled with `bot.long.risk.max_positions = 7`
- short disabled with `bot.short.risk.total_wallet_exposure_limit = 0`
- `bot.long.risk.total_wallet_exposure_limit = 1.5`
- HSL present in config but disabled with `bot.{long,short}.hsl.enabled = false`
- HSL signal mode set to per-coin slots with `bot.global.hsl.scope = "coin"`
- approved-coin universe seeded to the current 41-coin default list
- optimizer backend defaulting to `pymoo`

## When To Omit A Config Path

Omitting `config_path` is useful for:

- fast smoke tests
- CLI exploration
- generating a config mentally from the canonical defaults

It is usually better to pass an explicit config file for normal backtest, optimize, and live workflows, because:

- your exact settings are easier to reproduce
- diffs are easier to review
- you avoid depending on future default changes

## Command Examples

Backtest from the canonical default:

```bash
passivbot backtest -s BTC -sd 2025 --suite n
```

Optimize from the canonical default:

```bash
passivbot optimize -s BTC -sd 2025 -c 4 --suite n
```

Live from an explicit config:

```bash
passivbot live configs/private/my_config.json
```

## Related Docs

- [Configuration](configuration.md)
- [Optimizing](optimizing.md)
- [Backtesting](backtesting.md)
