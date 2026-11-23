# Passivbot v9 release preview

V9 changes the authored config, live risk behavior, and the inputs and metrics that define
a backtest or optimizer run. Follow [release status](releases.md) when selecting a revision.

## Highlights

- The `v9.0.0` config schema groups inherited directional defaults under `bot.global` and
  account-wide policy under `bot.portfolio`. It makes side activation and the per-side coin
  universe explicit. Authored files and their overrides receive strict validation.
  [Configuration workflow](config_workflow.md)
- Each HSL policy tracks its own equity drawdown. Auto-unstuck and the portfolio loss gate
  share one account-wide drawdown from net-realized fills and one decay span at
  `bot.portfolio.risk.loss_realization.memory_ema_span_minutes`. Unstuck scales its
  allowance by the applicable total wallet exposure limit. The gate applies its cap directly
  to the account. Unstuck sizes a close toward its exposure trigger, subject to available
  allowance and venue quantity rules. Panic closes bypass the gate. Live cold starts can
  have incomplete loss history.
  [Risk model](v9_risk_model.md) and [HSL validation](hsl_live_validation.md)
- Initial entries use the earliest verified price evidence for an asset when evaluating
  minimum coin age. Forager can use RMS directionality alongside volume, EMA readiness, and
  log range. Trailing Martingale and EMA Anchor have maintained public examples.
  [Forager](forager.md) and [strategy parameters](config.bot.md)
- Backtests report native core metrics with explicit definitions and observation windows.
  Research sessions record evaluator and dataset identity, and suites can select scenarios
  and reduce metrics across them. CPU remains the default optimizer backend.
  [Metrics](metrics.md), [backtesting](backtesting.md), and [optimization](optimizing.md)
- Live execution uses fresh account cohorts and scoped readiness checks, including protective
  HSL actions while fill history settles. Bitunix can recover ambiguous ordinary placements
  from durable attempts across restarts. Live containers mount the state directory, and
  structured diagnostics expose readiness and recovery.
  [Live guide](live.md), [container state](container_deployment.md#persistent-live-state),
  and [monitoring](monitor.md)

## Upgrade from v8

1. Keep the exact running commit or tag, its environment and launch command, the original
   config and any referenced files, and the live data directory. Record the package version,
   source commit, and native runtime identity from startup.
2. Install v9 in a separate environment. Use the explicit converter on a copy of a supported
   v8 config.

   ```sh
   passivbot tool migrate-config-v9 input.v8.json output.v9.json --best-effort
   ```

3. Review the migration report. Confirm side activation, the coin universe, overrides, fees,
   and exposure. Choose the HSL scope, its decay spans, and the shared realized-loss span.
   An active v8 loss gate requires an explicit span choice and
   `--acknowledge-realized-loss-gate-memory-change` even in best-effort mode. Review the
   removed unstuck close-size setting and the exposure trigger used for sizing. A v8 rolling
   window has no unique equivalent EMA span. [Supported schemas and migration choices](v8_to_v9_migration.md)
4. Backtest the migrated policy on the intended data and scenarios. Start a fresh optimizer run
   and migrate any v8 candidate configs before reusing them as seeds. V8 fitness and checkpoints
   are tied to the old evaluation contract. [Backtesting](backtesting.md) and [optimization](optimizing.md)
5. Qualify the separately installed runtime using the [offline fake-live guide](fake_live.md)
   and the [HSL validation procedure](hsl_live_validation.md). When switching live versions,
   preserve the persistent data root. Check exchange positions, resting orders, and readiness
   against the intended policy.

## Rollback

Use the exact v8 revision that last ran the account, together with its matching native extension,
configuration, launch command, and environment. The [v8 cutover reference](releases.md#v8-rollback-reference)
identifies the master snapshot used for this upgrade review. Resolve outstanding Bitunix create
attempts before returning to a build that cannot read the journal, and verify positions, resting
orders, and protection readiness during rollback.
[Bitunix recovery](ai/features/exchange_integrations.md#recovery-of-ordinary-creations)

## GPU optimization

CUDA and Metal are opt-in experimental backends. Device precision can affect scores and feasibility
near boundaries. Run the [GPU/CPU parity tool](gpu_parity.md) on representative setups and evaluate
candidates selected for use with the Rust CPU backtest.
