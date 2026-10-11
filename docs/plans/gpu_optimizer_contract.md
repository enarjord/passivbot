# GPU-native optimizer development contract

## Goal

Develop a simpler, maintainable NVIDIA/CUDA-first optimizer whose GPU backtest service
returns authoritative metrics asynchronously. CPU orchestration prepares requests and
processes results without running CPU backtests during GPU optimization. Preserve CPU
optimization and standalone CPU backtests. Prefer a strong foundation and clearer
ownership over marginal speedups that add complexity.

This document contains the current acceptance contract and evolving checklist.
The [decision and progress log](gpu_optimizer_decision_log.md) records supporting
design history and validation. Design choices may change when evidence supports
simpler, more capable, or more efficient implementations. Acceptance scope must
not be silently weakened.

The [acceptance evidence map](gpu_optimizer_acceptance.md) separates verified ownership,
search/storage and device cases from the remaining simulator, resource and cutover gates.

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

The internal executor accepts `register_dataset_factory(dataset_id, factory)`, where
the factory returns a replay resource context. Its first request creates the replay on
the owning worker; repeated requests reuse it, and shutdown exits all entered contexts
on that worker. Registration and unused/cancelled datasets do not initialize device state.
Preconstructed replay registration remains a temporary compatibility path.

The [prepared CUDA facade](../gpu_backtest_service.md) registers `PreparedGpuDataset`
metadata and borrows shared arrays with explicit source-column identities. It owns lazy
worker-side attachment, packing reuse, compatible subset caching and a one-active-dataset/
scratch residency policy. Caller-owned shared segments remain immutable/live until
shutdown. `NativeDatasetRegistry` binds canonical CPU scenario preparation to this
interface; the native optimizer already uses it for bootstrap, screening and resume.
The CPU orchestrator owns persistent content/evaluation fingerprints, precision stamps
and resume compatibility. Broader residency and multi-GPU routing remain future service
work; simulator acceptance and legacy retirement remain separate gates.

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

Prioritize focused, meaningful HSL speed improvements before the remaining
evolving-search comparisons, so later qualification and experiments benefit from
faster backtests. Preserve risk semantics and use paired correctness/performance
evidence; a contract change requires a separate decision. This execution priority
does not replace the final comparisons or impose a universal speedup threshold.

## Practical parity and numerical policy

Prefer float32 where its benefits justify it. Bitwise float32/float64 identity is not
required. Use explicit absolute/relative tolerances per metric, with sample sufficiency
and documented non-finite sentinels handled intentionally. Tolerances are provisional
until baseline observations, not a global percentage applied indiscriminately.
The acceptance map records [case-scoped numerical decisions](gpu_optimizer_acceptance.md#practical-numerical-foundation--scoped-decisions);
these decisions do not change the standalone tool's policies or certify every metric.

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

The experimental native backend fully evaluates seeds and initial parents. For later
offspring cohorts, the existing `optimize.gpu.screening` settings select a feasibility-
and Pareto-diverse subset for full-suite GPU evaluation. Only those complete offspring
enter evolutionary survival with the existing complete parents. Selecting every scenario
or retaining every offspring bypasses the partial stage. Explicit objective/limit scenarios
must remain in the screen. `iters` retains its cohort-generation interpretation; screening
reduces the number of complete evaluations rather than extending the generation budget.
Native checkpoint version 3 stores partial selection evidence separately from fitness
and rejects pre-cutover GPU state.
Earlier experimental native checkpoints require a fresh run; saved configs remain usable seeds.

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
- [x] Record reproducible CPU/GPU parity and cold/warm benchmark baselines.
  Preserve each measurement's source and recipe; refreshed equal-budget evolving
  comparisons remain a final acceptance gate.
  - [x] Add public synthetic cohort measurements with first-use/warm scope, direct/native
    equivalence, completion latency and CPU ranking/feasibility evidence for two seeds.

### Backtest service

- [x] Implement dependency-light request/completion API and bounded asynchronous lifecycle.
- [x] Reuse existing replay engines behind a temporary adapter without changing their semantics.
- [x] Verify identity, output cardinality, backpressure, exceptions, cancellation and shutdown.
- [x] Register immutable data, reuse packing/compilation, isolate mutable replay state.
- [x] Add prepared shared-array metadata with explicit column binding and worker-owned residency.
- [x] Bind canonical standalone/lazy suite preparation to the service without copying candle histories.
- [x] Demonstrate incremental completions and bounded memory on CUDA.
  This covers admission, resident replay allocations and the measured larger service
  recipes; it is not a total-VRAM guarantee for arbitrary inputs.
  - [x] Prove incremental admission/completions and bounded replay allocations on CUDA;
    the larger service measurements record device/host/disk observations with
    explicit sampling and future-workload limitations.

### Authoritative simulation and tooling

- [x] Implement standalone GPU/CPU parity tooling with structured diagnostics.
- [x] Audit approximation inventory against representative correctness cases.
  The code-backed inventory separates definition controls, measured cases and
  policy-unassessed fields; it does not certify all 157 output values.
- [x] Resolve material differences and record accepted numerical discrepancies.
  The initial foundation has explicit case-scoped decisions; keep tight-limit and
  trajectory limitations visible. Final combined acceptance remains below.
  - [x] Align active coin HSL's empty retained-fill history with Rust's fresh
    current-position loss estimate; the subsequent factual reconstruction resolves
    the measured retained-fill and aggregate outlier.
  - [x] Develop opt-in retained factual-fill reconstruction, bounded device storage,
    actual scope episode boundaries and disposable cutoff caching. Component,
    native full/chunk/growth/discard and matched candidate evidence is recorded in
    the acceptance map. The known material HSL cohort discrepancy is resolved;
    remaining small numerical differences are accepted only for those fixtures.
  - [x] Adopt factual replay in the default native worker and version the semantic
    checkpoint contract after independent review and CI on development.
  - [x] Consolidate wider feature/lifecycle, specialization, scenario, resource and
    performance acceptance. The final affected qualification passes 38
    CUDA-enabled checks plus six CPU-entrypoint checks, carried across the
    host-only correction that passes two GCC checks. Paired warm and eight
    evolving/four-front comparisons are complete for their stated recipes;
    numerical and resource limits remain explicit. The independently checked
    old-pipeline baseline is accepted with its actual partial TM timeout; this
    does not certify every workload or numerical field. Current-head delivery
    review/CI/integration remain tracked in the
    [final foundation status](gpu_optimizer_acceptance.md#final-foundation-status),
    rather than inferred from historical component counts.
- [x] Verify requested metric surface and specialized/general kernel equivalence.
  The documented cases cover requested presence/status and the stated reducer and
  compiler controls. Fields without declared tolerances remain policy-unassessed;
  finite output is not an accuracy pass.
- [x] Qualify native exact TM initial-entry interval metrics, compact reduction,
  scratch/index bounds and temporal/retry controls: the 22-check CUDA-enabled group
  passes, and the eight-candidate twenty-day cohort preserves all five interval
  rankings and the ADG/drawdown/p99 front. This accepts that cohort only; arbitrary
  CPU/GPU fill trajectories and combined retirement qualification remain separate.
- [x] Qualify requested native EMA-tail observation capture, compact device
  reduction, scratch admission and feature-off behavior on CUDA; assess CPU-curve
  discrepancies and representative objective/limit effects on the documented cases.
  - [x] Qualify corrected capture, scopes, reporting clocks, retry/temporal isolation,
    scratch admission and terminal handling: 81 focused checks pass with CUDA enabled;
    four standalone CPU/GPU comparisons confirm unified side reports remain zero
    and measure the corrected scope residuals. The completed eight-candidate
    twenty-day companion preserves all six EMA rankings/fronts and CPU-selected
    optima, with explicitly recorded equality-boundary sensitivity. This is
    case-scoped acceptance, not a general numerical policy.
- [x] Replace hourly recovery distribution sampling with per-step GPU observations,
  budget their replay/reduction storage, and isolate mutable reduction scratch.
- [x] Restore safe disabled-HSL single-side EMA ablation and verify all returned outputs.
- [x] Prove effective candidate/coin-side unstuck EMA consumers independently of scheduling;
  specialize multicoin EMA/TM layouts and verify general/specialized outputs.
- [x] Integrate full multicoin unstuck ablation after current-head review/CI.
  Effective-consumer proofs, raw/general/temporal/cache controls, shared loss/HSL
  consumers, longer HSL panics and disabled native CLI interrupt/resume controls
  pass on CUDA; paired public-fixture measurements are recorded in acceptance.
  Independent current-head review and required CI pass; development integration
  is complete.
- [x] Integrate prepared one-side CUDA compiler specialization after review/CI.
  General/specialized raw outputs, temporal/cache transitions and native single/suite
  lifecycle controls pass. Paired compiler observations are recorded in acceptance.
  Native single-coin multicoin replay already compiles with capacity one; retain
  that shared implementation rather than adding a separate kernel without evidence.
  PR #1952 passes independent current-head review and all required CI and is
  integrated into development.

### Optimizer cutover

- [x] Integrate GPU result scoring/limits through existing CPU-owned canonical helpers.
- [x] Add incremental candidate/suite result collection using canonical CPU scoring helpers.
- [x] Add canonical CPU request preparation and bounded polling/duplicate collection helpers.
- [x] Add experimental GPU-only ask/tell CLI, seed evaluation and partial-cohort checkpoint resume.
- [x] Preserve suite screening, full-suite reduction, effective deduplication and seed handling.
  - [x] Prepare finite anchor/side-enable execution views over shared histories and restore
    checkpoint-owned anchors before native resume shape construction.
  - [x] Reuse validated scenario evidence across screening/full stages without reusing partial
    scores or skipping required full-suite collection; keep caches bounded and run-local.
  - [x] Add CPU-owned native survivor selection, full seed/bootstrap evaluation and checkpointed
    screening/promotion/full stages; exclude incomplete observations from fitness/storage.
- [x] Tune execution and CPU result/evolution cadence without implicit numerical changes.
  Completed-work and CPU-cadence controls establish the mechanism; larger busy-HSL
  measurements document its throughput/latency limits rather than optimal tuning.
  - [x] Give native factual HSL compact storage independent of the bypassed legacy
    observation tree/window; preserve the legacy layout for remaining consumers.
    Source-verified CUDA allocation and replay controls, independent review and CI
    pass; development integration is complete. Representative total-resource
    acceptance remains separate.
  - [x] Retain small learned factual-capacity estimates in dataset execution metadata
    across residency eviction, without retaining device buffers or adding checkpoints.
    Host eviction/error-policy and actual CUDA scenario/owner switches, independent
    review and CI pass; development integration is complete. Representative suite
    acceptance remains separate.
  - [x] Measure held, many-coin HSL reconstruction scaling before further launch
    tuning. Matched 2/25-coin, 512/1024/2048-bar cases show roughly quadratic
    warm replay cost with HSL enabled; wider acceptance remains separate.
  - [x] Evaluate and integrate guarded incremental reconstruction against fresh factual
    replay, including cache loss, fills, budgets, clipping, numerical conditions and
    CPU parity. Independent review and CI pass on development; broader resources
    and whole-optimizer performance remain separate gates.
    The final candidate adds disposable CUDA EMA unified controller summaries,
    capped at 64 MiB per candidate within the shared 512 MiB history/scratch
    budget. Cache ambiguity, terminal evaluation or failed admission returns to
    independent scalar replay; other default scopes remain unchanged. Final
    paired warm acceptance is complete for the two-coin EMA unified recipe:
    two alternating B1 pairs give a 3.92259× median warm ratio with stated metric
    and memory scope. This does not establish cold-start or universal speedup;
    whole-optimizer comparisons remain separate in the current status below.
  - [x] Integrate a focused strategy-neutral multicoin replay owner and compact
    physical recovery results. Affected host/CUDA/CLI and combined continuation
    validation, current-head independent review and required CI pass on development.
  - [x] Interleave bounded CPU preparation and result servicing, adapt completion grouping
    from CPU cost, and keep suite notification fan-in independent of persistence batches.
  - [x] Let execution tuning learn actual warm partial dispatches and explore smaller
    widths when growth is blocked; representative tuning quality remains open.
- [x] Add service-owned production batch tuning and prepared work/scratch dispatch limits.
  - [x] Adapt native CUDA history chunks from completed command durations within
    their existing ceilings; retain explicit observational-latency limits.
    Representative wider-cohort and longer-suite scheduling acceptance remains open.
- [x] Flush results/Pareto promptly; validate interruption and compatible resume.
- [x] Prove no CPU backtest is invoked during GPU optimize/bootstrap/resume.
- [x] Retire superseded GPU screening/validation state and keep CPU functionality intact.
  The candidate removes the old pipeline and passes the affected native/CPU
  qualification. Current-head review, CI and development integration remain
  separate requirements below.

### Acceptance

- [x] Compare representative baselines and document architectural simplification.
  The eight equal-budget CPU/native evolving runs and four common-reference
  fronts are complete for the documented 12-coin, three-scenario throughput
  recipe. Stored fitness and Pareto records match; numerical tradeoff and timing
  limits are explicit. The old screening/CPU-validation baseline is measured:
  EMA completes 40 records, while TM times out with 13 persisted records.
  Independent inspection verifies both eight-seed CPU-reference matches and
  accepts the bounded baseline while retaining its failed terminal status and
  missing full-TM scope. This is not an equal-budget old/new search-quality pass.
- [x] Run required Rust/native Python/device/CLI checks with verified builds.
  The final affected qualification carries 38 CUDA-enabled and six CPU-entrypoint
  passes at `f629b46811` across the test-only successor `1239e7bac0`, which passes
  both corrected GCC host checks. Production bytes and the verified extension
  remain unchanged; 341 Rust tests and the default build/check are attributed to
  the original `3d8581225f` build. This closes the stated qualification, not the
  comparison, numerical-unassessed or development-integration gates.
- [ ] Complete development-branch PR review and CI for all delivered slices.
- [ ] Update user-facing/AI contracts for the delivered behavior on the development branch.
- [ ] Reconcile this checklist and leave master integration for a separate decision.

Current comparison, cache-performance and delivery gates are consolidated in the
[final foundation status](gpu_optimizer_acceptance.md#final-foundation-status).
Historical component measurements retain their original revisions and scope;
fixed-candidate parity does not replace the evolving or old-pipeline comparison.

## Decision and progress log

See the [decision and progress log](gpu_optimizer_decision_log.md) for the dated
implementation decisions, review outcomes, measurements and remaining limitations.
