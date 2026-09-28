# Revised HSL sole-engine migration

Status: migration preparation. Both engines remain available and legacy remains the default.
This document defines the retirement work; it does not claim that deletion, deployment, or
exchange acceptance has happened. The tracking PR must remain draft until its acceptance
criteria are met. Passing tests on the starting revision is baseline evidence, not evidence
for a later deletion diff.

## Decision and scope

Make the existing revised HSL implementation the only HSL implementation. Preserve its
[current signal, estimator, and lifecycle](../hsl_revised.md). This is a behavioral migration
for legacy users, not a refactor that promises legacy-equivalent trading or fitness.
Do not combine it with new formulas, tighter reconstruction requirements, different settling
timers, or speculative speed changes.

The revised contract is fixed for this migration:

- Anchor reconstructed history to current equity (scope balance budget plus current UPNL).
  Smooth raw drawdown; panic only when the current minimum of raw drawdown and its EMA
  strictly exceeds the configured threshold. Historical RED is not a commitment.
- Current positions are authoritative. Rust reconciles incomplete history with explicit
  approximation diagnostics. Bad history must not resurrect the legacy all-or-nothing
  readiness gate. Fresh current inputs and action-specific admission remain required.
- A flat scope's latest episode terminal RED sample determines cooldown; renewed exposure
  clears that cooldown. Corrected evidence can revise the result. No local journal is authority.
- Forget evidence outside the configured 1–90 day lookback, including `never` restrictions.
  Use one coin-side, one side, or one portfolio controller for coin, pside, or unified mode.
- Preserve the shared bounded position-to-fill settling gate, fresh connector admission,
  current RED recovery cancellation, and Rust ownership of all trading decisions.

## Reviewable implementation order

Use one tracking draft with independently reviewable commits, or stack narrow PRs if the
reviewer prefers. Do not publish a runnable-looking default switch before its callers work.

1. **Acceptance baseline and documentation contract.** Map the retained revised tests to
   the matrix below. Capture source-verified native results before removing code. Preserve
   independent Decimal/reference tests, rather than treating old legacy results as the oracle.
2. **Configuration and caller cutover.** Route live, fake-live, standalone simulation,
   CPU optimization, GPU screening, reports and monitor output to the sole implementation.
   Update canonical template generation, normalization, CLI, overrides and resume fingerprints
   together. Exercise HSL disabled as well as enabled; disabled HSL must not acquire expensive
   history work or affect unrelated strategies.
3. **Delete legacy ownership.** Remove the legacy Python replay/supervisor and emergency
   journal, Rust signal/controller and simulator state, legacy GPU dispatch/controller,
   obsolete Python bindings, metrics and legacy-only tests. First identify helpers shared with
   revised HSL or non-HSL consumers; move those to their actual owner rather than deleting them
   by filename or by a broad text replacement. Keep ordinary strategy readiness, protective
   execution, fill reconciliation, fees, unstuck and position/fill synchronization intact.
4. **Documentation and examples.** Promote the revised contract, rewrite the canonical HSL
   feature contract, update all active user guidance and intentional public examples, and
   remove obsolete defaults/options/metrics from generated references. Historical changelog
   entries may describe legacy; active instructions must not. Update links before retiring
   the old contract. Do not publish private configurations or operational artifacts.
5. **Qualify the integrated candidate.** Re-run the matrix against the exact merge candidate,
   finish independent review and CI, then obtain the separately scoped deployment/release
   decision. A draft PR, a green baseline, and a healthy process are not migration completion.

## Proposed compatibility boundary

The final removal commit must implement and test these rules before changing the default:

| Input | Sole-engine behavior |
|---|---|
| No selector, revised-compatible configuration | Use the sole HSL engine |
| Explicit `live.hsl_engine=revised` | Accept as a compatibility spelling; no second runtime path |
| Explicit `live.hsl_engine=legacy` | Fail before account initialization, simulation, or optimizer work; explain migration |
| Enabled scope with missing or `threshold` restart policy | Require an explicit `always` or `never` choice; do not infer it |
| Unified mode without `bot.hsl` | Require an explicit portfolio policy; never copy a side or hydrate it from defaults |
| Retired tier/intervention/terminal-stop fields | Diagnose through the canonical migration boundary; document loss of their trading effect |
| Retired optimizer bounds, objectives, or runtime overrides | Reject with actionable paths; do not silently optimize inactive dimensions |
| Legacy optimizer checkpoint or cached fitness | Reject incompatible resume; explicitly supplied old candidates may be re-evaluated only after config migration |

Retaining an accepted `revised` spelling is config compatibility, not retaining legacy code.
Normalizing an old threshold number does not make it economically equivalent: users must
re-backtest/re-optimize after the signal change. Do not silently rewrite saved configurations.
The release note must state the default change and its risk-policy implications. A version/tag
or release publication requires a separate maintainer decision.

Rollback after deletion means the previous reviewed release with its compatible saved config,
not `--live.hsl_engine legacy` on the new binary. Running bots change only through an explicitly
approved restart/deployment; merging the PR is not such authorization.

## Offline configuration preparation

The current dual-engine build provides an explicit helper:

```sh
passivbot tool migrate-hsl input.json output.json --restart-policy long=always --restart-policy short=never
```

It writes a new file only after canonical validation, never overwrites an existing file,
and never starts a bot. Choices apply only to the specified scope; per-coin explicit policies
remain explicit and may still need editing. Existing `always`/`never` choices can be retained
without an override. An enabled legacy `threshold` choice must be replaced deliberately.
For unified mode, supply a complete portfolio policy as JSON through `--portfolio-policy`,
or author `bot.hsl` in the input. The helper never copies side policy into it. Retired
optimizer search dimensions/objectives must be edited explicitly before conversion succeeds.
Run the resulting config through backtests and review the changed risk semantics before use.

## Acceptance matrix

| Contract | Existing executable evidence to retain and run after deletion |
|---|---|
| Equity anchor, raw EMA, strict threshold, fractional spans, current RED recovery | `tests/test_hsl_revised_signal.py`, `tests/test_hsl_revised_trace.py`, independent `tests/test_hsl_reference*.py` |
| Missing/duplicate/ambiguous history and current authoritative positions | `tests/test_hsl_revised_reconciler.py`, `tests/test_hsl_revised_history.py`, `tests/test_hsl_revised_snapshot.py`, `tests/test_hsl_revised_current_flat.py` |
| Real protective execution without history, partial recovery cancellation, restart and cooldown | `tests/test_hsl_revised_live.py`, `tests/test_hsl_revised_fake_cycle.py`, `tests/test_hsl_revised_history_timing_live.py` |
| Position/fill settling terminates, scopes remain independent, connector writes recheck inputs | `tests/test_position_fill_sync.py`, `tests/test_position_fill_sync_fake_live.py`, `tests/test_hsl_revised_live.py` |
| Public backtest and CPU optimizer CLI, both engines' old compatibility boundary, reporting and resume | `tests/test_hsl_revised_backtest_config.py`, `tests/test_hsl_revised_config.py`, `tests/test_hsl_revised_offline_runtime.py`, `tests/test_hsl_revised_optimizer_contract.py`, `tests/test_hsl_revised_reporting.py`; replace legacy-default assertions with migration assertions |
| Single/multi-coin, modes, policies, GPU batches/chunks and public optimizer CLI | `tests/optimization/test_gpu_revised_hsl_*.py`; actual Metal and CUDA runs are separate hardware evidence |
| No HSL effects when disabled, no non-HSL trading drift | Existing strategy/order/optimizer and fake-live regressions; source-verified before/after traces with HSL disabled |
| Removed implementation cannot be selected | New loader/CLI/native-boundary rejection tests; call-site inventory after removal |
| Documentation and event contracts | AI docs checker, generated event registry check, docs tests, active-document reference audit |

Include both long and short and all three modes. Market/limit execution, zero/nonzero cooldown,
`always`/`never`, fresh-process reconstruction, fees/final accounting and lookback expiry need
explicit coverage. A flat-only test does not cover held-position protection. A finite GREEN
observation does not cover actual panic execution.

Run the baseline in an isolated checkout with the source-matched Python extension:

```sh
PYTHONPATH=src python -c 'import passivbot_rust; from rust_utils import verify_loaded_runtime_extension; print(verify_loaded_runtime_extension())'
(cd passivbot-rust && cargo test --no-default-features && cargo check --tests)
PYTHONPATH=src pytest tests/test_hsl_reference*.py tests/test_hsl_revised*.py tests/test_position_fill_sync*.py
PYTHONPATH=src pytest tests/optimization/test_gpu_revised_hsl_*.py
PYTHONPATH=src python src/tools/check_ai_docs.py
PYTHONPATH=src python src/tools/generate_live_event_registry.py --check
```

The fake exchange is offline. Hardware skips are not passes. Record exact source and native
fingerprints, commands, pass/fail/skip counts and any environment limitation in review evidence.
Do not include credentials, private configs, account identifiers or exchange telemetry there.

## Performance acceptance

Performance is a separate measurable gate, not a requirement to exhaust all conceivable
optimizations. Record before/after medians on the same machine, fixture and artifact mode:

- Ordinary and frequent-RED single-coin CPU replay; short and long lookbacks.
- Active multi-coin scopes with fills, budget changes and sliding-window boundaries.
- Metrics-only optimizer output versus normal and explicitly detailed backtest reports.
- HSL-disabled CPU/GPU workloads, one/multiple sides, small/large candidate batches,
  full/chunked replay, and exact finalist validation.
- Live capture plus admission under an advancing clock; do not increase freshness limits
  to conceal computation that outlives its inputs.

Use `tests/hsl_revised_backtest_benchmark.py`, `tests/hsl_revised_gpu_benchmark.py` and
`tests/hsl_revised_live_benchmark.py` as reproducible starting fixtures. Compare trading
traces before interpreting timing changes. A no-stop synthetic microbenchmark does not
establish worst-case performance or a complete optimizer speedup. Legacy and revised use
different mathematics, so their trading results are not expected to match.

## Completion and operator decisions

Code completion requires no executable legacy HSL path, a single documented policy, compatible
entry points, passing integrated acceptance checks, exact-head independent review and CI.
Live acceptance records which lifecycle cases actually occurred and which remain offline-only;
it does not manufacture exposure to obtain coverage. If additional live validation is desired,
the operator must approve the exact bots, candidate, saved launch commands and intervention scope.

No further risk-policy invention is required to begin the migration work. Before deployment,
the maintainer chooses the cutover targets/time and accepts the configuration changes. Before
publication, the maintainer decides release/version policy. Neither decision prevents offline
implementation or review. Keep unresolved implementation, hardware validation and live acceptance
items visible separately; do not declare the upgrade complete because one category is green.
