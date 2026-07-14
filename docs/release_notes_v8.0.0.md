# Passivbot v8.0.0

Passivbot v8 is a major, breaking release. It replaces the v7 strategy and
configuration contract, moves substantially more trading and risk behavior
into the shared Rust runtime, and makes the long-developed `v8` branch the
mainline release.

## Before upgrading

- Use Python 3.12 and a current stable Rust toolchain.
- Back up the config you currently run and keep the exact v7 revision needed
  to reproduce its behavior.
- Do not point v8 at an unreviewed v7 config. Normal v8 loading does not
  silently convert removed v7 strategy fields.
- Treat migration as a new deployment: inspect the normalized config, run
  backtests, and review live-mode, risk, approved-coin, and exchange settings
  before starting a bot.

## Config migration

The canonical v8 strategy is `trailing_martingale`. Its threshold,
retracement, quantity, and close-recursion parameters are not aliases for the
old v7 trailing-grid fields.

For users who want to preserve v7 trailing-grid semantics during the upgrade,
v8 includes the deprecated compatibility strategy `trailing_grid_v7` and an
explicit migration tool:

```bash
passivbot tool migrate-config-v7 \
  path/to/config_v7.json \
  path/to/config_v8_trailing_grid_v7.json \
  --report path/to/v7_migration_report.json
```

A clean migration writes canonical `config_version: "v8.0.0"` shape with
`live.strategy_kind = "trailing_grid_v7"`. The tool does not reinterpret the
config as `trailing_martingale`.

If the report contains dropped, unsupported, or manual-review fields, the
command returns nonzero and does not write the output by default. The
`--allow-manual-review-output` option writes a best-effort artifact for manual
work; it is not an assertion that the result is ready for live trading.

For new configs, copy:

```bash
cp configs/examples/default_trailing_martingale_long.json my_v8_config.json
```

See [Config Workflow](config_workflow.md) and
[Configuration Reference](configuration.md) for the canonical v8 shape.

## Major changes

### Shared Rust trading runtime

- Live trading, backtesting, and optimization use the Rust orchestrator for
  strategy planning and order intent.
- Strategy, order, risk, unstuck, HSL transition, and backtest behavior have
  stronger shared ownership and parity coverage.
- Required trading inputs fail closed or become explicitly unavailable rather
  than being replaced with fabricated neutral defaults.

### Strategies and portfolio controls

- `trailing_martingale` is the canonical recursive entry and close strategy.
- `ema_anchor` is available as an additional canonical strategy.
- Forager selection, wallet-exposure controls, total-wallet-exposure controls,
  unstucking, and realized-loss gates have been consolidated and hardened.
- EMA spans remain floating-point values throughout config derivation and
  runtime preparation.

### Equity Hard Stop Loss

- HSL supports unified, position-side, and per-coin signal scopes with explicit
  RED episode, cooldown, restart, and no-restart policies.
- Coin-mode startup reconstructs exchange-derived history, prioritizes held
  positions for protective readiness, and may continue non-blocking replay work
  in the background.
- Panic-close and protective management paths use dedicated readiness
  boundaries; incomplete required history remains visible and fail-closed.

Review [Equity Hard Stop Loss](equity_hard_stop_loss.md) and its risk warnings
before enabling or changing HSL settings.

### Live readiness and exchange safety

- Account and market inputs are tracked through staged freshness and planning
  snapshots before normal order planning and execution.
- Ambiguous exchange writes require authoritative confirmation before unsafe
  retries.
- Exchange adapters and fill/PnL normalization have expanded coverage,
  including Bitget UTA behavior and Hyperliquid HIP-3 stock perpetuals.

### Logging, monitoring, and events

- V8 introduces a structured live-event pipeline with bounded console, text,
  monitor, and structured-data projections.
- Events carry stable types, reason codes, correlation identifiers, redaction,
  and sink-isolation behavior for operational queries and incident analysis.
- The event architecture is part of v8. Console ownership and performance
  refinements may continue after v8.0.0 without making observability a trading
  control plane.

### Backtesting, optimization, and data

- Backtesting and optimization use the shared Rust behavior contract and the
  current nested v8 config schema.
- OHLCV preparation, coverage validation, cache integrity, suite execution,
  optimizer checkpointing, deterministic seeds, stepped bounds, and Pareto
  result handling have been expanded and hardened.
- The unified `passivbot` CLI is the preferred entry point for live,
  backtest, optimize, download, and tool commands.

## Upgrade checklist

1. Install v8 in a Python 3.12 virtual environment and rebuild the Rust
   extension.
2. Migrate the v7 config or start from the canonical v8 example.
3. Resolve every migration report item; do not rely on best-effort output as a
   live-readiness signal.
4. Re-check API account selection, exchange mode, leverage, approved and
   ignored coins, exposure limits, HSL, unstuck, and logging settings.
5. Run a bounded backtest and inspect its fills and analysis artifacts.
6. Start live trading only through the normal operational process, with logs
   and monitor output observed closely during the first cycles.

The complete detailed change ledger is in [CHANGELOG.md](../CHANGELOG.md).
