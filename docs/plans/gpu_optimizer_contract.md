# GPU-native optimizer development contract

## Goal

Develop a simpler, maintainable NVIDIA/CUDA-first optimizer whose GPU backtest service
returns authoritative metrics asynchronously. CPU orchestration prepares requests and
processes results without running CPU backtests during GPU optimization. Preserve CPU
optimization and standalone CPU backtests. Prefer a strong foundation and clearer
ownership over marginal speedups that add complexity.

This document contains the acceptance contract, evolving checklist, decisions, and
progress. Design choices may change when evidence supports simpler, more capable, or
more efficient implementations. Acceptance scope must not be silently weakened.

## Branch and publication policy

- Development target: `codex/gpu-native-optimizer`, initially based on master
  `00ce7d0ddf123071ae67db149753d6b3619f52ac`.
- Implementation PRs target the development branch, never master. Integration into
  master is a separate decision after acceptance. Publish completed slices as regular PRs.
- Before any merge, wait for the auto review of the current head, inspect all reported
  issues and address them with evidence and appropriate regressions. Recheck review
  metadata and comments immediately before merging; CI and author review are additional
  checks. Development-branch PRs follow the same rule. Disputed findings follow the
  repository review-adjudication runbook and remain unresolved until adjudicated.
- Keep the existing GPU screening/CPU-validation backend until its replacement passes
  the relevant acceptance gates. Do not retain both architectures indefinitely.
- Follow [repository authority and data boundaries](../../AGENTS.md). Public evidence
  uses reproducible fixtures and public examples, not private configs or host details.
- Rust remains the owner of trading semantics. GPU execution does not require Python
  to reimplement trading rules or repeat simulations on CPU.

## Scope and feature matrix

NVIDIA/CUDA is the first delivery target. Apple Metal, cloud deployment, persistent
device workers, and intracandidate cooperative parallelism are subsequent work or
optional experiments, not initial completion requirements. Multi-GPU routing must fit
behind the service boundary without exposing hardware to the evolutionary algorithm;
initial acceptance requires one device, not a multi-GPU hardware demonstration.

| Surface | Initial acceptance scope |
| --- | --- |
| Strategies | EMA Anchor and Trailing Martingale |
| Portfolios | Supported single/multicoin, long/short, hedge/one-way topologies |
| Risk | Supported HSL, unstuck, WEL/TWEL, loss gating, liquidation, entry timing |
| Overrides | Effective coin/scenario overrides, optimizer runtime overrides, mirroring and anchors |
| Data | Existing supported intervals, gaps, validity windows, warmup and delisting behavior |
| Metrics | Current supported GPU objective/limit surface; audit each approximation |
| Suites | Scenario screening, full-suite evaluation, compatible grouping and canonical reducers |
| Results | Canonical scoring/limits, Pareto/config exports and compact durable records |
| Lifecycle | Clean interruption, prompt persistence, usable checkpoint resumption |
| CPU | Existing optimize/backtest/plots remain available with optional GPU imports isolated |

Existing unsupported strategy/metric/collateral combinations may remain explicit errors.
Produce a code-backed topology/metric inventory before cutover. Missing feature coverage
is a checklist item, not permission to silently reinterpret a request.

## Ownership and service boundary

CPU orchestration owns configuration normalization, market-data acquisition, scenario
planning, candidate generation, duplicate/invalid-value rejection, scoring directions,
limit penalties, scenario reduction, evolutionary selection, Pareto updates and disk
persistence. CPU processing can use parallelism where beneficial.

The GPU service owns device input packing, compilation/specialization, dataset residency,
bounded scratch allocation, request scheduling, simulation and requested backtest-metric
calculation. It has no evolutionary algorithm, Pareto, scoring-goal, constraint-penalty,
or optimizer-checkpoint dependency. Metric names specify requested work, not search policy.

Requests refer to registered immutable datasets and fully resolved candidate parameters.
Dataset identity covers candles/timestamps, ordered coins, market settings, dates, warmup,
validity and simulation assumptions. Backend-specific buffers and device count stay inside
the service. Mutable portfolio/replay state is isolated per simulation.

Submission is asynchronous and may accept multiple requests. Results identify the request,
dataset/evaluation contract, status, compact metrics and bounded diagnostics. No full
history transfer by default. Return completions incrementally without a generation-wide
barrier; bounded internal microbatches are permitted. The orchestrator must not depend on
their size, completion order, streams or device layout.

Start with one owning service/context per device. Group compatible work for locality,
while bounding queue memory and starvation. Device errors and malformed output propagate;
never fabricate successful metrics or silently substitute CPU backtests. Preserve the
original failure if cleanup also fails.

The current service accepts `register_dataset_factory(dataset_id, factory)`, where
the factory returns a replay resource context. Its first request creates the replay on
the owning worker; repeated requests reuse it, and shutdown exits all entered contexts
on that worker. Registration and unused/cancelled datasets do not initialize device state.
Prepared-input immutability is the factory adapter's responsibility; the final dataset
registry and residency policy are still separate work. Preconstructed replay registration
remains a temporary compatibility path.

## Scheduling, tuning and specialization

Use completed-work evidence to tune batch width/delay, concurrency, replay-slot capacity,
dispatch duration and residency/scratch budgets. Retain bounded experiments, minimum
evidence, cooldowns, rollback and workload-compatible advisory measurements from existing
auto-tuning where useful. Do not preserve exact-validation pool/queue machinery after
cutover merely to reuse its tuner.

Tune CPU result-consumption and evolution-update cadence to observed bottlenecks. Cadence
may approach one candidate per update for low rates and use cohorts at high rates. Assess
evaluation-time bias and search quality when cadence changes; population/variation policy
does not become an execution autotuner's incidental side effect.

Scheduling changes must not change per-candidate simulated behavior. Cache loss, different
batch widths and different devices must not alter the intended calculation. Record
precision/compiler mode in evaluation identity instead of silently tuning accuracy.

Compile out HSL, unstuck, inactive sides, diagnostics and other proven inactive consumers.
Account for all effective coin/scenario overrides before proving inactivity. Compare
specialized/general results. Keep specialization and execution scheduling independent so
future kernel experiments do not require changes to search orchestration.

Prioritize immutable market-data reuse and compact device reductions. Bound caching of
candidate-dependent indicators; do not trade unbounded memory for hypothetical reuse.
Measure throughput, completion latency, memory, transfer overhead and useful work, not
GPU utilization alone.

## Practical parity and numerical policy

Prefer float32 where its benefits justify it. Bitwise float32/float64 identity is not
required. Use explicit absolute/relative tolerances per metric, with sample sufficiency
and documented non-finite sentinels handled intentionally. Tolerances are provisional
until baseline observations, not a global percentage applied indiscriminately.

Evaluate discrepancies case by case for material effect on trading paths, risk decisions,
feasibility and optimizer selection. Do not chase decimal noise or require perfect
long-history trajectory equality solely because event thresholds amplify rounding.
Conversely, deliberate all-history loss envelopes, conservative entry exclusions or
missing strategy behavior cannot become authoritative merely by renaming the proxy.
Resolve them or record an explicit bounded discrepancy and its supporting evidence.

Reference tests may run CPU backtests during development and release validation. They
must not run implicitly inside the GPU optimizer. Provide a repeatable parity command
with per-metric errors, scenario/candidate identity, feasibility comparison, selective
fill/state evidence where available, and machine-readable reports. Comparison failures
must be distinguishable from missing data, unsupported work and execution failure.

Parity coverage includes feature-on/off combinations, both strategies/sides, multicoin
shared accounting, boundary minimums, HSL/restart/cooldown, rolling PnL, recursive orders,
gaps/delists, and representative suites. New simulator features must extend this coverage
so removing runtime CPU validation does not remove the ability to discover divergence.

## Search, screening and persistence

GPU results directly determine evolutionary fitness and canonical result records.
Scenario screening remains an explicit CPU search-budget policy using GPU evaluations.
Screening-only observations do not masquerade as complete-suite results. Aggregate and
store a complete candidate only after all required scenario evaluations succeed.

Reject invalid requests and effective duplicate candidates before expensive work. Keep
new speculative prefilters optional until their missed-good-candidate behavior is tested.
Preserve configured bounds, floats, overrides and warmup when eliminating ineffective genes.

Persist completed candidate results promptly and persist Pareto members as they arrive.
Use bounded buffering and existing result-store contracts rather than requiring per-result
global checkpoints or a new perfect-replay journal. At interruption stop admission, collect
available completed work, flush results/Pareto state and save a usable checkpoint. Lost
in-flight work may be rerun on GPU; it must not erase already durable results.

Exact reproduction of an evolutionary trajectory is not required. Validate candidate
identity, data/config compatibility and durable results on resume. Optimizer RNG/population
state should be preserved where straightforward, without reconstructing every scheduler
or tuner event. Do not replay an incompatible old screening population as authoritative.

## Acceptance evidence

Architectural improvement with comparable performance is an acceptable outcome. Do not
declare success on utilization, isolated kernel timing or fewer evaluated scenarios alone.
Measure pinned CPU/old-GPU/new-GPU baselines with the same effective inputs and documented
precision/metric contracts. Separate startup/compilation/preparation from warm execution,
and record memory and completion tails. Include representative repeated-seed optimization
comparisons where feasible; state limits honestly.

Completion requires:

1. The agreed feature inventory is implemented or has explicit, evidence-backed accepted
   limitations; parity tooling/tests demonstrate the practical numerical contract.
2. GPU results alone drive GPU optimization, canonical limits, selection, Pareto and storage;
   tests prove no CPU simulation or hidden fallback occurs, including seed bootstrap/resume.
3. The black-box service supports bounded asynchronous work, resident reuse, specialization,
   useful adaptive tuning and lifecycle/error handling with a clean ownership boundary.
4. Scenario suites/screening, interruption and durable-result/resume behavior work end to end.
5. CPU backends and standalone backtest/plotting remain functional and dependency-isolated.
6. Obsolete screening-validation machinery is removed after replacement acceptance, with a
   documented comparison of dependencies, state and maintenance burden.
7. Proportionate Python/Rust tests, verified extension/device tests and reproducible performance
   comparisons pass; the checklist/decision log accurately records evidence and limitations.

## Evolving checklist

### Foundation and baseline

- [x] Activate the high-level goal with this contract as its detailed reference.
- [x] Refresh master and create an isolated development branch.
- [x] Commit/publish this contract on the development branch; master unchanged.
- [x] Inventory current supported topologies, metrics, deliberate approximations and direct callers.
- [x] Establish isolated NVIDIA runtime with source-fingerprint verification.
- [ ] Record reproducible CPU/GPU parity and cold/warm benchmark baselines.

### Backtest service

- [x] Implement dependency-light request/completion API and bounded asynchronous lifecycle.
- [x] Reuse existing replay engines behind a temporary adapter without changing their semantics.
- [x] Verify identity, output cardinality, backpressure, exceptions, cancellation and shutdown.
- [ ] Register immutable data, reuse packing/compilation, isolate mutable replay state.
- [ ] Demonstrate incremental completions and bounded memory on CUDA.

### Authoritative simulation and tooling

- [x] Implement standalone GPU/CPU parity tooling with structured diagnostics.
- [ ] Audit approximation inventory against representative correctness cases.
- [ ] Resolve material differences and record accepted numerical discrepancies.
- [ ] Verify requested metric surface and specialized/general kernel equivalence.

### Optimizer cutover

- [ ] Integrate GPU result scoring/limits through existing CPU-owned canonical helpers.
- [ ] Preserve suite screening, full-suite reduction, effective deduplication and seed handling.
- [ ] Tune execution and CPU result/evolution cadence without implicit numerical changes.
- [ ] Flush results/Pareto promptly; validate interruption and compatible resume.
- [ ] Prove no CPU backtest is invoked during GPU optimize/bootstrap/resume.
- [ ] Retire superseded GPU screening/validation state and keep CPU functionality intact.

### Acceptance

- [ ] Compare representative baselines and document architectural simplification.
- [ ] Run required Rust/native Python/device/CLI checks with verified builds.
- [ ] Complete development-branch PR review and CI for all delivered slices.
- [ ] Update user-facing/AI contracts for the delivered behavior on the development branch.
- [ ] Reconcile this checklist and leave master integration for a separate decision.

## Decision and progress log

### 2026-10-05 — Scope and initial design

- Accepted NVIDIA/CUDA first, Apple Metal later; retain CPU optimize/backtest/plots.
- Architectural simplification is primary; measured speedup is desirable, not mandatory.
- Practical float32 parity is evaluated by materiality, not perfect bit identity.
- GPU service is independent of evolution; CPU owns scoring, limits, suites and persistence.
- Begin with bounded asynchronous microbatching; persistent/cooperative kernels remain experiments.
- Prepare for multiple devices inside the service; do not expose device count to search policy.
- Adaptive result/evolution cadence is allowed; measure CPU bottlenecks and search effects.
- Prompt Pareto durability and clean resume matter more than exact trajectory replay.
- Existing replay engines are only transitional adapters until approximation/parity acceptance;
  adding an asynchronous wrapper does not make their current outputs authoritative.
- Current source inspection: GPU evolution tells NSGA-II proxy objectives; CPU results populate
  the authoritative archive and drift monitoring. Cutover must change this information flow.

### 2026-10-05 — First execution-service slice

- Published the contract on `codex/gpu-native-optimizer`; implementation slice uses
  `codex/gpu-native-service`, with a PR to the development branch.
- Added [`GpuBacktestService`](../../src/optimization/gpu/executor.py): one owning
  execution thread, bounded queued-plus-running admission, FIFO compatible microbatches,
  individual futures, parameter snapshots, fail-stop producer errors and clean drain/cancel.
  Cancellation must notify completion consumers even when removed before dispatch.
- The API imports no optional GPU or evolutionary dependencies. Existing replay handles
  are exclusively registered transitional adapters; optimizer integration and the final
  immutable dataset/residency interface remain outstanding.
- Nineteen offline lifecycle cases pass. Four CUDA cases cover both strategies in single-
  and multicoin replay, cold owner-thread compilation, repeated reuse, partial batches,
  request identity and matching metrics across caller threads/batch shapes.
- A bounded warm comparison uses the public deterministic benchmark fixtures, 256 candidates,
  4,096 bars, one/three coins, seed 7, dispatch/microbatch width 64, queue capacity 256 and
  1 ms accumulation delay. Five repeated runs preserve all returned metrics exactly. The
  first microbatch completes before the full accepted set; transport adds about 1–10% on
  these small workloads. This establishes baseline overhead, not an end-to-end speedup.
- Native device validation uses a rebuilt extension whose embedded source fingerprint
  matches the current Rust tree. Corrected test-only Torch leakage and a stale shader-cache
  argument assertion; all fourteen capacity-specialization comparisons pass on CUDA.
- Recorded the code-backed [cutover inventory](gpu_optimizer_inventory.md), including
  current conservative filters/loss gates and baseline disabled-HSL/dual-side unstuck
  findings. These remain acceptance work, not silently approved numerical exceptions.
- No optimizer behavior or CPU validation policy changes in this slice. Existing CPU/GPU
  backends remain selectable while authoritative simulation/parity tooling is developed.
- First PR review found lifecycle ordering issues. Detached queued cancellation before
  waking the owner thread, released admission before publishing success, removed request
  snapshots from retained future callbacks and rolled back failed thread startup. Added
  four deterministic regressions. Source-only specialization checks again run without
  Torch through an isolated module import that restores both module/package references.

### 2026-10-05 — Standalone parity tooling and measured gaps

- Added [offline parity tooling](../gpu_parity.md) with reproducible synthetic fixtures
  and prepared NPZ/config/market inputs. Comparisons use the verified native CPU engine
  and the new asynchronous service, with standard JSON, explicit per-metric policies,
  input/source identities, canonical scalar limit checks and bounded optional diagnostics.
- Missing/unsupported metrics, absent policies, non-finite values, suite-only checks
  and execution failures remain distinguishable from numerical agreement. Limit
  feasibility is compared independently; a tolerance does not hide threshold crossings.
- A 21-case, 5,760-bar, seed-7 matrix exposed small EMA fill/ADG differences and
  material multicoin TM short/passive-order differences. Both long-only TM cases
  pass the provisional policies; all cases agree on completion coverage.
  Ordinary market-order variants remove the large TM discrepancy in this fixture,
  narrowing the next investigation. Conservative minimum-cost filtering yields zero
  GPU fills in the tested dual-side cases. None of these is approved by widening a
  global tolerance or treating transport equivalence as CPU parity.
- Initial fixture revisions used template dates and then failed to account for canonical
  UTC-day end-date normalization. Aligned exclusive endpoints to midnight before
  recording the matrix; the earlier coverage discrepancy was input preparation, not
  a confirmed simulator defect. Controller toggles alone are not transition coverage.
- Scalar comparison is implemented; suite/ranking comparisons and broader stress cases
  remain required acceptance work. No optimizer cutover or implicit CPU fallback added.
- Parity-tool review identified omitted reducer defaults, unapplied prepared fixed
  runtime overrides and loss of stdout results on an optional save failure. Added
  canonical reducer/override handling and preserved results before attempting the save,
  with regressions. Unmaterialized optimizer enable-overrides are explicitly rejected.
- Reject explicitly supplied fixture-only switches in prepared-input mode, including
  switches set to fixture defaults. Prepared reports must not imply that an ignored
  execution setting was exercised; regression checks reject all eight fixture options
  before loading inputs.

### 2026-10-05 — Multicoin minimum-cost simulation semantics

- Replaced multicoin liquidation-floor/all-history-minimum admission with current
  simulated account balance and current-candle executable exchange minima. Kept dynamic
  WEL/allowance/initial-quantity overrides, preselection and one-way arbitration, and
  held-position management. Removed the permanent proxy/exact uncertainty state and
  rejection scan; the shader change removes roughly 190 net lines of obsolete machinery.
- EMA selection refreshes each candle when cost filtering is enabled, because both cash
  and current-price minima can change. Unfiltered selection cadence is unchanged in
  this slice and still requires the broader authoritative-simulation audit.
- Updated obsolete conservative-filter regressions to exercise affordable slots,
  delayed eligibility, rejected-coin isolation, per-coin overrides and side arbitration.
  A delayed-entry fixture initially also exhausted the exposure budget; made the first
  coin independently unaffordable so the test isolates cost admission from exposure.
- Independent paired CPU and GPU synthetic comparisons check that affordable filtering
  preserves each engine's unfiltered fills in all three side modes for both strategies.
  They do not conceal the separate baseline TM short/passive-order discrepancy.
- Full Rust tests and default-feature checks pass; rebuilt native source fingerprint
  matches the changed tree. Remaining single-coin conservative filters, unused packed
  maximum-cost metadata and the old optimizer configuration guard are subsequent work.
- Validation: 330 Rust tests pass (one ignored), default-feature test compilation passes,
  23 CUDA admission regressions and 37 parity-tool tests pass. Ten asynchronous-service
  and temporal-replay cases pass, including partial tails and 3/28/64-coin layouts after
  removing the obsolete serialized state. Source-only parity tests: 30 pass, seven skips.
- Reviewed minimum-cost rounding explicitly: with two markets at price 100 and minimum
  cost 5, initial quantity 0.01 admits in both engines; its immediately lower float64
  neighbor is rejected on CPU but rounds to the same admitted GPU payload. Cases 0.01%
  below/above the boundary agree. Retain native float32 admission and expose ambiguous
  threshold discontinuities, rather than adding a screening-style conservative margin
  that would also reject affordable equality. This is a bounded input-quantization
  limitation; its potentially large metric effect remains a mismatch in reports.
- Added paired boundary coverage for both strategies. These tests require agreement at
  equality and outside the ambiguous rounding cell, and truthful mismatch reporting
  when an ambiguous input produces different simulation paths. No global tolerance or
  implicit CPU fallback is introduced.

### 2026-10-05 — Passive recursive multicoin TM ladders

- The baseline generated recursive entry and close suffixes only with market orders
  enabled. Removed that restriction and passed ordinary market policy explicitly
  through suffix generation, exposure allocation and fill processing. Recursive state
  now describes strategy expansion independently of market promotion.
- Isolated close-only and entry-only experiments did not recover CPU behavior; a
  close-only change also reused a recursion flag as execution policy. The combined
  correction separates those responsibilities rather than promoting passive orders.
- In the 5,760-bar seed-7 short-only fixture, ADG relative disagreement falls from
  roughly 96.7% to 0.0412%, and fill disagreement from 60.3% to 0.0234%. Dual-side
  ADG disagreement falls to 0.0549%, with 0.0210% fill disagreement; drawdown passes
  the provisional absolute policy. Market-enabled outputs are unchanged by the fix.
- Keep strict tool policies unchanged. Dedicated regression guards recovered ADG/fill
  trajectories within 0.1% for these two fixtures; this is a fixture-specific guard,
  not a universal acceptance policy or a claim that all scalar comparisons pass.
- Extend existing exposure-ordering, partial-boundary, cooldown and fused side-mode
  regressions to passive execution. Close-group checks distinguish passive maker
  execution from unintended market promotion using independently different fee outcomes.
- Validation: 330 Rust tests pass (one ignored), default-feature test compilation and
  source-verified extension rebuild pass. CUDA passes 35 recursive-entry/market/reducer
  regressions, eight passive/market close-group cases, 47 comparator/tool tests and 34
  admission/asynchronous-service/temporal-reuse cases, including 3/28/64-coin layouts.
  Source-only comparator/tool tests: 38 pass, nine device skips. Documentation checks
  report no errors and two existing context-size warnings.
- Review corrections: label the original large parity gaps as historical and record
  the repeated matrix's current residual differences. Reject Boolean absolute/relative
  tolerance values before simulation, because JSON `true` must not become an allowed
  absolute error of 1. Four policy-field regressions and a CLI pre-execution check pass;
  the combined CUDA comparator/tool suite now passes 60 tests (source-only: 43 pass,
  seventeen device skips).

### 2026-10-05 — Worker-owned replay construction and cleanup

- Added lazy replay-resource factories to the dependency-light execution service.
  Context entry, repeated evaluation and context exit belong to the worker, with
  last-in/first-out cleanup for every initialized dataset. Unused and cancelled work
  does not create a device replay. Registration identity remains shared with the
  preconstructed compatibility adapter.
- Setup/evaluation failures fail admitted work and poison further admission. Cleanup
  failures are surfaced on close; a prior simulation/setup exception is preserved and
  secondary cleanup failure logged. Exit all resource contexts even if one cleanup fails.
- The standalone parity tool now constructs its GPU replay through this path and restores
  diagnostic hooks before disposal. Preparation timing is measured inside construction
  and excluded from the cold service-execution timing, preserving their definitions.
- Validation: 32 offline lifecycle tests pass. Four CUDA factory cases cover both
  strategies and one/three coins, proving construction/cleanup thread ownership, replay
  reuse/disposal and direct GPU metric agreement with CPU backtest calls forbidden.
  Combined lifecycle/parity/old-adapter/factory CUDA tests: 100 pass; source-only lifecycle
  and parity tests: 75 pass, seventeen device skips. Bounded real diagnostic CLI smoke
  passes with matching completion, fills, ADG and drawdown under provisional policies.
- This is an ownership foundation, not the final immutable dataset API, multi-device
  router, residency budget or optimizer cutover. Those checklist items remain open.
### 2026-10-05 — Prepared parity input binding and merge review gate

- Late review findings exposed two missed prepared-input checks: matching unsorted
  config/NPZ coins were accepted although payload construction sorted only identities;
  a requested/default exchange could disagree with config and market venues. Reproduced
  acceptance with failing regressions before changing the checks. These can invalidate
  prepared reports even when the two simulators agree; prior synthetic measurements use
  sorted identities and matching venues and are unaffected by these specific defects.
- Require canonical sorted NPZ coin order, exact declared order when present, and the
  effective single or combined exchange. Validate per-coin market venues against the
  configured sources, including explicit combined-source assignments. Reject ambiguity
  before either simulation; do not silently relabel data or reorder large arrays.
- Add a current-head auto-review gate before all development merges. Inspect reported
  comments as well as review status, address findings, and recheck immediately before
  merge. CI and author checks alone are insufficient.
- Validation: 65 comparator/tool cases pass on CUDA; the source-only run passes 49
  with 16 device skips. CLI regressions prove invalid identities fail before simulation,
  and a valid prepared nondefault-venue input reproduces its synthetic CPU/GPU metrics.
  Documentation checks have zero errors and two existing size warnings.
- Current-head auto review caught a further distinction: `exchange` identifies market
  settings while `ohlcv_source` identifies candles when they differ. Validate candle
  assignments against configured/forced data sources and settings independently against
  `market_settings_sources`; preserve the producer's documented settings-to-candle
  fallback. Reuse offline source-key reconciliation and normalize venue names. Added
  valid independent/fallback and incorrect candle/settings-source regressions before
  changing the guards. This review finding is addressed before integration.
- After source-role correction, 70 CUDA comparator/tool tests pass; source-only checks
  pass 54 with 16 device skips. No simulator or tolerance-policy changes in this slice.
- Further current-head review exposed inconsistent connector-name normalization and a
  wrapped-config declaration bypass. Normalize requested and declared exchange identities
  consistently, reject conflicting alias mappings, and use canonical flavor detection
  for both declaration checks and ignored gene bounds. Added seven source regressions
  and three paired CUDA alias/wrapper replays. All 80 comparator/tool cases pass on
  CUDA; source-only checks pass 61 with 19 device skips. Findings are addressed before
  re-review and integration; simulation and tolerance policies remain unchanged.
- Rebased onto the integrated worker-ownership and recursive-ladder changes and checked
  the merged result: 87 CUDA comparator/tool cases pass; source-only checks pass 66 with
  21 native/device skips. Documentation checks retain zero errors and two size warnings.
