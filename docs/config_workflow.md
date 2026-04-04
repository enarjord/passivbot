# Config Workflow

This is the recommended way to work with Passivbot configs on the current config system.

## Source Of Truth

- The canonical hardcoded defaults live in `src/config/schema.py`.
- The example config `configs/examples/default_trailing_grid_long_npos10.json` mirrors those defaults exactly.
- If you run `passivbot live`, `passivbot backtest`, or `passivbot optimize` without a config path, Passivbot starts from the in-code defaults in `src/config/schema.py`.

## Recommended Workflow

1. Copy `configs/examples/default_trailing_grid_long_npos10.json` to a new file.
2. Edit that new file for your account, market universe, and strategy changes.
3. Use `passivbot backtest` first.
4. Use `passivbot optimize` if you want to tune parameters or compare alternatives.
5. Use `passivbot live` only after the config has been tested.
6. Expect live runs to write a timestamped log file under `logs/` by default unless you set `logging.persist_to_file = false`.

Example:

```bash
cp configs/examples/default_trailing_grid_long_npos10.json configs/live/my_config.json
passivbot backtest configs/live/my_config.json -s BTC -sd 2025 --suite n
passivbot optimize configs/live/my_config.json -s BTC -sd 2025 -c 4 --suite n
passivbot live configs/live/my_config.json
```

## Best Practices

- Keep one normal JSON or HJSON config per strategy/account instead of relying on many CLI overrides.
- Use CLI overrides for temporary experiments, not as your main configuration workflow.
- Treat `bot`, `live`, `backtest`, and `optimize` as one config file with command-specific sections.
- Keep new configs on the canonical schema. Do not author new configs using deprecated field names.
- Use `coin_overrides` for per-coin exceptions instead of cloning whole configs for minor differences.
- Leave `logging.persist_to_file = true` for normal live operations so each bot run has a durable logfile under `logs/`.

## What The Default Profile Is

The default profile mirrored by `configs/examples/default_trailing_grid_long_npos10.json` is:

- trailing-grid style configuration
- long enabled with `bot.long.n_positions = 10`
- short disabled with `bot.short.total_wallet_exposure_limit = 0`
- `bot.long.total_wallet_exposure_limit = 1.25`
- HSL present in config but disabled with `hsl_enabled = false`
- approved-coin universe seeded to the current default large-cap list
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
passivbot live configs/live/my_config.json
```

## Related Docs

- [Configuration](configuration.md)
- [Optimizing](optimizing.md)
- [Backtesting](backtesting.md)
