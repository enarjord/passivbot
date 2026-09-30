# HSL validation

The [HSL contract](ai/features/equity_hard_stop_loss.md) defines behavior. Tests must
compare decisions and trading traces before comparing speed. No finite suite proves
that every exchange failure has been anticipated.

## Acceptance matrix

| Boundary | Required cases | Main test surfaces |
|---|---|---|
| Signal | Current-equity anchor; strict threshold; fractional EMA; same-minute replacement; current recovery | `test_hsl_signal`, `test_hsl_trace`, `test_hsl_metric_regression` |
| Reconciliation | Missing opening or final fills; duplicated/corrected fills; unordered cohorts; flat exchange position; signed shorts; overflow | `test_hsl_history`, `test_hsl_reconciler`, `test_hsl_current_flat` |
| Prices | Minute closes; deterministic coarse-candle expansion; missing prefix/interior; late corrections; no future leak | `test_hsl_prices`, `test_hsl_candles` |
| Scope and lifecycle | Coin, pside, unified; zero slots; nonzero hedged gross exposure; latest episode only; terminal RED; exposure clears cooldown; lookback expiry | `test_hsl_snapshot`, `test_hsl_evaluator`, native simulator tests |
| Current facts | Stale or invalid balance/positions/marks/orders; unavailable is distinct from GREEN; no reuse of prior permission | `test_hsl_runtime`, `test_hsl_live` |
| Execution | Bounded position/fill settlement; independent scopes; empty plans; partial closes; cancel stale panic; shutdown during awaits; restart reconstruction | `test_position_fill_sync`, `test_hsl_fake_cycle`, `test_hsl_history_timing_live`, `test_hsl_live` |
| Configuration | Explicit restart; explicit unified policy; coin overrides; CLI/scenarios; removed controls rejected as optimization targets | `test_hsl_config`, `test_hsl_optimizer_contract`, `test_migrate_hsl_config` |
| Simulation | Actual panic orders, fees/slippage, terminal boundaries, disabled HSL, liquidation, metrics after result draining | Rust backtest tests, `test_hsl_backtest_config`, `test_hsl_reporting` |
| GPU | Both strategy families and directions; scopes; sliding windows; changing budgets; full/chunked replay; bounded scratch | `tests/optimization/test_gpu_hsl_*` |

The independent Python reference is a test oracle, never a live controller or a fallback.
The offline fake exchange exercises the real orchestration and connector admission path
without credentials, account requests or real orders. Keep private operational fixtures out
of public test artifacts; use synthetic, reproducible facts.

## Build and regression checks

Run Rust tests and rebuild the extension from the same checkout before native Python tests.
Verify the loaded artifact's source fingerprint. Run the affected unit and fake-live suites,
configuration roundtrips and public examples, documentation checks, and a bounded offline
backtest/optimizer smoke. Current-head independent review and CI are separate requirements.

## Migration qualification

The package is `8.2.0.dev0`; the canonical config schema is `v8.5.0`. Package and schema
versions have different meanings. Test every supported earlier v8 schema (`v8.0.0` through
`v8.4.0`) in coin, pside and unified modes. Relabeling an old config is not a migration.
Require explicit restart choices and an explicit portfolio policy for unified mode; check
file-backed coin overrides, effective scenario/optimizer policies and removed dimensions.
Write a separate output, reload it through the normal loader and repeat migration to prove
idempotence. Failures must preserve the source and existing output. Future or unknown schemas
must remain rejected. Migration does not preserve old HSL behavior or authorize deployment.

From a source-verified environment, these offline checks cover that boundary:

```bash
PYTHONPATH=src pytest tests/test_passivbot_version.py tests/test_config_pipeline.py \
  tests/test_migrate_hsl_config.py tests/test_hsl_config.py tests/test_hsl_cli.py \
  tests/test_hsl_optimizer_contract.py tests/test_coin_overrides_hsl.py \
  tests/test_retired_hsl_controls.py tests/test_ai_docs.py
passivbot --version
passivbot tool migrate-hsl --help
```

Re-backtest migrated thresholds and reevaluate optimizer candidates; never carry forward old
fitness/checkpoints as evidence for the new calculation. Use fixed offline data and compare
HSL-disabled trading traces, current RED/recovery, terminal cooldown and all three scopes.
A package version, schema roundtrip or GREEN observation alone does not qualify execution.

## Performance

Use fixed synthetic candles, identical policy, identical outputs and a source-verified release
build. Measure HSL enabled and disabled; compact and detailed reporting; CPU simulation and
GPU screening. Compare full replay with incremental caches and replay after cache loss.
GPU float32 is approximate screening; exact Rust validation remains authoritative.

Reproducible probes live in `tests/hsl_backtest_benchmark.py`,
`tests/hsl_gpu_benchmark.py`, and `tests/hsl_live_benchmark.py`.
Report hardware, input dimensions, repetitions, elapsed time and retained trace equality.
A faster result that changes decisions, removes required checks or hides unavailable input
is not an accepted optimization.

## Live evidence

A separate [operator-approved live trial](hsl_live_validation.md) verifies exchange
behavior. Record which cases actually occurred; ordinary trading does not prove a panic
close, partial fill, restart or cooldown was exercised. Offline success does not authorize
live process changes or authenticated probes.
