# Config Workflow

This is the recommended way to work with Passivbot configs on the current config system.

## Source Of Truth

- The canonical schema defaults hydrate omitted fields in explicit config files. An omitted
  activation flag leaves its side disabled, an omitted approval source does not grant coin
  approval, and an omitted optimizer bound fixes its effective parameter value.
- The in-code starter in `src/config/schema.py` matches the
  [shipped example](https://github.com/enarjord/passivbot/blob/v9/configs/examples/default_trailing_martingale_long.json). It declares
  activation, approval, and optimizer bounds explicitly.
- New configs use the final `config_version: "v9.0.0"` layout. Normal loading rejects
  unknown/retired fields and never migrates policy implicitly. Use the explicit
  [v8 converter](v8_to_v9_migration.md) for released v8 inputs, with the required semantic
  choices and fresh backtests. Config schema and [package release versions](releases.md)
  are separate. Unreleased intermediate schemas, `pb_multi` and pre-v8 layouts are unsupported.
- Retained strategies are `trailing_martingale` and `ema_anchor`. The v7 strategy and its
  dedicated migration command have been removed.

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
or add `--mode live` for a lean live export. Use `--mode compact` for an editable authored view
that preserves common defaults and explicit side/search/override pins, or `--mode resolved` to
inline local coin files while retaining that source authority. Expanded effective values and
origins are available in inspection output. Inspect a setting with
`passivbot tool config-explain input.json PATH`.
Formatting only is available with `--mode format`.
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
- Keep account-wide strategy selection, execution and warmup in `bot.portfolio`, shared approval sources and forager hysteresis in `bot.global`, and directional strategy/universe policy in `bot.<side>`. Backtests consume `bot.portfolio.execution.market_orders_allowed` and `market_order_near_touch_fraction`; live fee fallback/sanity settings retain their operational `live` ownership. These execution and fee fields are not accepted under `backtest`. The nullable `backtest.filter_by_min_effective_cost` override inherits each side's `bot.<side>.universe.filter_by_min_effective_cost` by default; an explicit boolean overrides both sides.
- Risk policy belongs to its consumers: each HSL policy has its own slow and ratio-smoothing
  spans; global unstuck uses `bot.portfolio.unstuck.memory_ema_span_minutes`; global loss gate uses
  `bot.portfolio.loss_gate.memory_ema_span_minutes`. Acquisition uses the separately bounded
  live bootstrap cap. A rolling window is not an equivalent EMA span.
- Keep new configs on the canonical schema. Use nested active strategy paths and explicit
  [migration reports](v8_to_v9_migration.md) for released-v8 changes.
- Use `coin_overrides` for per-coin exceptions instead of cloning whole configs for minor differences.
- Leave `logging.persist_to_file = true` for normal live operations so each bot run has a durable logfile under `logs/` and Passivbot's monitor tooling can follow the stable `logs/{user}.log` alias or Windows pointer.

## What The Default Profile Is

The in-code starter and shipped example explicitly enable longs and disable shorts. The
Trailing Martingale profile is shared through `bot.global` and inherited by each side.

- Each side has 10 slots and a total wallet exposure limit of `1.5`.
- The 40-coin approval list, forager ranking, and adaptive entry cooldown are shared.
- Coin-scope HSL is disabled.
- Backtesting uses a rolling end date and venue market settings. The five-scenario suite is
  available with `--suite`.
- CPU `pymoo` optimization uses explicit search ranges, a budget of 2,048 evaluations, and a
  population of 128. The enabled limit requires 100% backtest completion in every evaluated
  scenario.

## When To Omit A Config Path

`passivbot backtest` and `passivbot optimize` use the starter when you omit the config path.
Use the full install and either cached or accessible public market data. CLI overrides can
narrow the universe or dates for a shorter run.

To run `passivbot live` with the starter, provision the intended account in your local
`api-keys.json` under `passivbot_default`. This opts that account into the starter's long
entries. You can also select that alias explicitly. With no config path, choosing a different
alias via `-u` uses the schema defaults and leaves new entries disabled unless the CLI
explicitly supplies activation and coin approval.

Use an explicit config file to keep your exact settings reviewable and reproducible across
default changes. CLI overrides apply on top of the selected config.

## Command Examples

Backtest or optimize the starter:

```bash
passivbot backtest
passivbot optimize
```

Run a narrower backtest with CLI overrides:

```bash
passivbot backtest -s BTC -sd 2025
```

Live from an explicit config:

```bash
passivbot live configs/private/my_config.json
```

## Related Docs

- [Configuration](configuration.md)
- [Optimizing](optimizing.md)
- [Backtesting](backtesting.md)
