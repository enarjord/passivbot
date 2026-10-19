# HSL validation

The [HSL contract](ai/features/equity_hard_stop_loss.md) defines behavior. Tests must
compare decisions and trading traces before comparing speed. No finite suite proves
that every exchange failure has been anticipated.

## Acceptance matrix

| Boundary | Required cases | Main test surfaces |
|---|---|---|
| Signal | Forward currency drawdown; contemporaneous ratios; strict threshold; fractional EMA; same-minute replacement; current recovery | `test_v9_risk_reference`, `test_hsl_signal`, `test_hsl_metric_regression` |
| Reconciliation | Missing opening or final fills; duplicated/corrected fills; unordered cohorts; flat exchange position; signed shorts; overflow | `test_hsl_history`, `test_hsl_reconciler`, `test_hsl_current_flat` |
| Prices | Minute closes; deterministic coarse-candle expansion; missing prefix/interior; late corrections; no future leak | `test_hsl_prices`, `test_hsl_candles` |
| Scope and lifecycle | Coin, pside, portfolio; zero slots; hedged gross exposure; zero episode seed; pre-RED OR post-RED flatten; fixed cooldown; no renewal | `test_hsl_snapshot`, `test_hsl_evaluator`, native simulator tests |
| Current facts | Stale or invalid balance/positions/marks/orders; unavailable is distinct from GREEN; no reuse of prior permission | `test_hsl_runtime`, `test_hsl_live` |
| Execution | Bounded position/fill settlement; independent scopes; empty plans; partial closes; cancel stale panic; shutdown during awaits; restart reconstruction | `test_position_fill_sync`, `test_hsl_fake_cycle`, `test_hsl_history_timing_live`, `test_hsl_live` |
| Configuration | Explicit v9 migration/span; Never-removal acknowledgement; portfolio policy; coin overrides; CLI/scenarios; removed controls rejected | `test_hsl_config`, `test_hsl_optimizer_contract`, `test_v9_config_migration` |
| Simulation | Actual panic orders, fees/slippage, terminal boundaries, disabled HSL, liquidation, metrics after result draining | Rust backtest tests, `test_hsl_backtest_config`, `test_hsl_reporting` |
| GPU | Both strategy families and directions; scopes; decaying state; changing budgets; full/chunked replay; constant retained risk state | `tests/optimization/test_gpu_hsl_*` |

The independent Python reference is a test oracle, never a live controller or a fallback.
The offline fake exchange exercises the real orchestration and connector admission path
without credentials, account requests or real orders. Keep private operational fixtures out
of public test artifacts; use synthetic, reproducible facts.
Bind simulated event clocks to each immutable input receipt once, while keeping UTC
freshness and I/O deadlines separate. Repeated captures cannot move an older receipt
to a later scenario time. Keep risk signals and trading outcomes strict in repeat
comparisons. Settle input reads within the current scenario bar; an unsettled read
must report an incomplete replay instead of advancing time underneath its acquisition.

## Build and regression checks

Run Rust tests and rebuild the extension from the same checkout before native Python tests.
Verify the loaded artifact's source fingerprint. Run the affected unit and fake-live suites,
configuration roundtrips and public examples, documentation checks, and a bounded offline
backtest/optimizer smoke. Current-head independent review and CI are separate requirements.

## Migration qualification

The final `v9.0.0` layout is separate from package/release versions. Released v8 configurations require
explicit migration and acknowledgment for an active realized-loss gate. V8 also needs a
chosen slow span and acknowledgement of any removed Never policy.
Test nested/file-backed coin overrides, effective scenario/optimizer policies, independent
gate-window retirement, removed dimensions and invalid f32 spans. Reload separate output and
repeat migration for idempotence. Source/backup/output collisions and failed validation must
preserve original inputs. Old checkpoints/fitness are rejected; seeds are freshly evaluated.
See [v8-to-v9 migration](v8_to_v9_migration.md). No migration authorizes deployment.

From a source-verified environment, these offline checks cover that boundary:

```bash
PYTHONPATH=src pytest tests/test_passivbot_version.py tests/test_config_pipeline.py \
  tests/test_v9_config_migration.py tests/test_hsl_config.py tests/test_hsl_cli.py \
  tests/test_hsl_optimizer_contract.py tests/test_coin_overrides_hsl.py \
  tests/test_retired_hsl_controls.py tests/test_ai_docs.py
passivbot --version
passivbot tool migrate-config-v9 --help
```

Re-backtest migrated thresholds and reevaluate optimizer candidates; never carry forward old
fitness/checkpoints as evidence for the new calculation. Use fixed offline data and compare
HSL-disabled trading traces, current RED/recovery, terminal cooldown and all three scopes.
A package version, schema roundtrip or GREEN observation alone does not qualify execution.

## Performance

Use fixed synthetic candles, identical policy, identical outputs and a source-verified release
build. Measure HSL enabled and disabled; compact and detailed reporting; CPU simulation and
GPU execution. Compare forward replay with correction rebuilds and derived-cache loss.
GPU optimization uses authoritative float32 simulations without runtime CPU validation.
Run independent source-verified GPU/CPU parity checks when simulation behavior changes.

Use the current [optimizer benchmark](optimizer_benchmark.md) and
[GPU parity tools](gpu_parity.md) for supported v9 inputs. The retired standalone
[CPU](https://github.com/enarjord/passivbot/blob/bd4265409c407d8f2345f79f844520c9e827a8dc/tests/hsl_backtest_benchmark.py),
[GPU](https://github.com/enarjord/passivbot/blob/bd4265409c407d8f2345f79f844520c9e827a8dc/tests/hsl_gpu_benchmark.py) and
[live-adapter](https://github.com/enarjord/passivbot/blob/bd4265409c407d8f2345f79f844520c9e827a8dc/tests/hsl_live_benchmark.py)
probes belong to pinned v8 source and do not qualify v9 performance.
Report hardware, input dimensions, repetitions, elapsed time and retained trace equality.
A faster result that changes decisions, removes required checks or hides unavailable input
is not an accepted optimization.

## Live evidence

A separate [operator-approved live trial](hsl_live_validation.md) verifies exchange
behavior. Record which cases actually occurred; ordinary trading does not prove a panic
close, partial fill, restart or cooldown was exercised. Offline success does not authorize
live process changes or authenticated probes.
