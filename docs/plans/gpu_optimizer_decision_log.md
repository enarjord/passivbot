# GPU-native optimizer decision and progress log

This log preserves the development history. Current requirements and completion
criteria belong to the [development contract](gpu_optimizer_contract.md); current
evidence and unresolved gates belong to the [acceptance map](gpu_optimizer_acceptance.md).
Historical observations do not supersede those requirements or prove current acceptance.

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

### 2026-10-05 — One-coin shared-account replay experiment

- Permit one coin in the existing multicoin packer and replay constructor. This enables
  the same directional/fused account implementation to accept 1..64 coins without
  changing the legacy optimizer's choice of replay engine.
- CUDA coverage: 17 cases pass, including one-coin specialized/full-capacity raw-output
  equality, all three side modes and both market policies, repeated asynchronous
  microbatches, a partial batch, candidate exposure changes and forced temporal chunks.
  CPU backtest entry points are forbidden during replay tests. Packing preserves valid
  listing/delisting windows and input arrays and rejects zero/65-coin requests.
- The six public one-coin parity fixtures run through the shared engine. EMA metrics
  match the earlier single engine; TM differences remain small under case-specific
  assessment and still fail the tool's strict policies where appropriate.
- An exploratory warm comparison uses 256 repeated candidates, 4,096 bars, widths
  32/128 and three repeats for both strategies in long/dual modes. The shared engine
  delivers approximately 35–67% of the single engine's throughput in these short
  fixtures. Fixed measurement order and repeated identical candidates limit this
  experiment; it is not an optimizer-quality or general performance benchmark.
- Keep one-coin shared replay as an internal capability rather than switching every
  request now. Restore effective kernel ablation and measure representative workloads
  before choosing the native backend's default; simplification should not silently
  impose this observed throughput loss. No extra conservative screening or CPU replay
  is added to the service.

### 2026-10-05 — Prepared datasets and bounded CUDA ownership

- Added CPU-side metadata snapshots and borrowed shared-array descriptors. Source-column
  identities are explicit and selected indices must match canonical scenario coins;
  scenario time/validity metadata stays CPU-owned. Registration neither copies full
  histories nor initializes device dependencies. Owners keep arrays immutable/live until
  service shutdown; worker views are read-only.
- Added a CUDA-only facade over the bounded asynchronous executor. A lazy worker resource
  context surrounds dataset attachments and replay lifetimes. Reuse existing packing,
  compile caches and disk-backed scenario subsets; keep one active invariant dataset and
  one owner's replay scratch on device. Compatible config variants reuse the immutable
  packed data. No evolution/scoring or CPU backtest enters this facade.
- Keep the uniform 1..64-coin replay as this internal adapter's foundation; legacy routing
  is unchanged. Its measured one-coin slowdown still requires ablation and representative
  performance work before choosing optimizer defaults.
- Validation: 121 offline lifecycle/dataset/residency/preparation cases pass. Six CUDA
  facade cases cover both strategies, one/three coins, scenario/subset reuse, direct GPU
  metric equality, setup failure and interruption with CPU backtests forbidden. Shared
  attachments and temporary files close after success/failure. A wider selected run passes
  185 cases and reproduces the existing disabled-HSL specialization failure; its runner
  and test are unchanged by this slice. Full kernel-ablation acceptance remains open.
- A bounded synthetic benchmark CLI smoke succeeds with three coins, eight candidates,
  four-request dispatches, 128 bars and one warm run. The baseline's unchanged HSL
  specialization test independently reproduces the failure on its original test source.
- This is a preparation/residency foundation, not optimizer cutover, complete evaluation
  fingerprints, host/disk admission budgets, multi-device routing or adaptive scheduling.

### 2026-10-06 — Prepared attachment failure preservation

- Current-head auto review identified that raw attachment-close callbacks could replace
  an original setup/execution exception. Added three failing regressions before changing
  ownership to resource contexts. Close every acquired attachment, propagate the original
  failure and log later cleanup failures. With no earlier error, propagate the first
  cleanup failure rather than replacing it with another close error.
- Validation: 124 offline lifecycle/dataset/residency/preparation cases pass. The combined
  prepared-dataset/facade CUDA run passes 32 cases, including real setup failure and
  interruption. No simulation semantics or result policy changes in this correction.

### 2026-10-06 — Incremental CPU scoring of GPU results

- Native replay completions now retain the simulator's actual liquidation flag separately
  from requested metrics. The legacy metric-only replay interface remains unchanged and
  explicitly has unknown status; authoritative consumers reject that unknown status.
- Added per-candidate scenario/exchange collection and canonical CPU scoring, without
  simulation calls or GPU runtime imports. Validate identities, coverage and requested
  metrics before completing candidates; preserve canonical invalid-candidate handling of
  non-finite metric sentinels. Partial scenario screenings remain explicitly separate
  from recordable full evaluations and retain named objective/limit scenarios.
- Real CUDA tests exercise mixed liquidated/non-liquidated candidates and asynchronous
  two-scenario completion scoring with CPU backtest/evaluator simulation entry points
  forbidden. The crash fixture additionally compares actual terminal flags against
  standalone CPU references outside service execution. Candidate results may complete
  independently while other candidates are still waiting.
- Validation: 69 executor/scoring/prepared-service CUDA cases pass, including 21 CPU
  scoring cases and eight real CUDA facade/integration cases. Documentation checks have
  zero errors and two existing size warnings. No Rust, kernel, or legacy CLI routing
  changes are included in this slice.
- Optimizer CLI integration, global admission/deduplication, seed handling, search cadence,
  durable result/Pareto storage and compatible resume remain open. These scoring helpers
  do not certify the entire metric surface or simulation approximation inventory.
- Enforced the requested merge discipline on the prepared-dataset slice: fix the auto
  review's finding, add regressions, answer its original thread, await clear current-head
  re-review, perform author semantic review, and wait for all required CI before dev merge.

### 2026-10-06 — Canonical CPU planning and bounded collection

- Added CPU-only preparation of effective request parameters using canonical bounds,
  optimizer/fixed policies, mirroring and exact-last scenario overrides. Keep static
  topology/coin patches in prepared datasets; reject incompatible changes before GPU
  admission. Run-local duplicate identity includes unscreened effective scenarios.
- Added bounded candidate admission, pending-duplicate fan-out and snapshot caching over
  the exclusively borrowed service. Future callbacks only enqueue notifications; CPU
  polling scores completed candidates and replenishes device work. Screenings, full
  evaluations and different screening subsets cannot share cached completion payloads.
  Stop admission/cancel waiting work while preserving fully completed candidates.
- Group interleaved requests by compatible dataset inside the service. The oldest queued
  request chooses the next dataset and other datasets retain their relative queue order.
  Full admission dispatches available work without waiting for an impossible larger batch.
  Cancellation can reselect ready work instead of retaining an empty accumulation target.
- Reproduced a material default-binding bug for both strategies: the first coin's patched
  strategy supplied global defaults to unpatched coins. Separate global preparation from
  static coin overrides and retain shared parameter encoding instead of a second packer.
- Validation: 95 combined preparation/session/scoring/executor/CUDA cases pass, followed
  by ten session cases including distinct screening-subset isolation. Wider CUDA coverage
  passes 96 with the previously reproduced disabled-HSL specialization case explicitly
  excluded. Its acceptance debt remains open. All 375 construction/adaptive-timing/HSL
  service cases and 87 parity comparator/tool cases pass after updating the construction
  fixture's metadata-only preparation boundary. Documentation checks have zero errors and
  two existing size warnings. Another 88 prepared-data/residency cases and a bounded
  benchmark CLI smoke (three coins, eight candidates, 128 bars, width four, one warm run)
  pass. No Rust or kernel source changes are included. These runs validate behavior,
  not a representative throughput or optimizer-quality improvement.
- Optimizer CLI/data-registry integration, candidate-dependent coin patches/anchor variants,
  adaptive cadence, persistence and content/precision resume identities remain open. This
  slice provides CPU orchestration components and is not a native optimizer cutover.
- The result-scoring slice passed a clear current-head auto review and all required Python
  and Rust CI before merging into development. The same gate applies to this next slice.
- During subsequent data-registry integration, a regression exposed that dynamic numeric
  exposure/position values could change a prepared kernel's side enablement. Include the
  effective enabled sides in the dataset-owned contract and reject both activation and
  deactivation before submission. The combined CPU/CUDA suite passes 98 cases after this
  correction. The prior head's clear auto review cannot approve this changed head; request
  current-head re-review and wait for its required CI before integration.

### 2026-10-06 — Bind canonical prepared scenarios to native execution

- Retain actual source-column identities in canonical suite contexts. Lazy contexts keep
  the full master order, while already-sliced contexts retain their selected source order;
  CPU evaluator behavior is unchanged and context serialization remains compact.
- Added a CPU dataset registry over canonical standalone/suite preparation. Borrow existing
  candle/BTC shared segments and own only timestamp windows, reused by content. Independent
  timestamp source ranges prevent a second time slice when canonical contexts already
  contain a selected window. Validate row alignment, column identities, required metrics
  and effective seed policies before device registration; missing inputs fail explicitly.
- Real CUDA coverage starts from canonical suite preparation with deliberately shuffled
  source columns, a smaller coin subset and a later date window. Native service/session
  metrics match independent fresh GPU references built in canonical coin/time order. CPU
  backtest and evaluator simulation entry points are forbidden throughout execution.
- Validation: 83 combined prepared-data, canonical context, registry, planning/session and
  selected CUDA facade cases pass. Cleanup coverage closes every owned window, preserves
  original caller/preparation failures and leaves borrowed histories usable. Documentation
  checks have zero errors and two existing size warnings. No Rust/kernel or CLI routing
  changes are included; search integration, persistence/precision resume identities,
  adaptive tuning and the existing simulation-ablation acceptance debt remain open.

### 2026-10-06 — Experimental GPU-only optimizer CLI

- Add the separate `gpu_native` backend while retaining CPU and old GPU options. Reuse
  canonical preparation/scoring/limits, pymoo NSGA-II/III operators and existing result/Pareto
  stores. Preserve generation/cohort search semantics while replenishing bounded service
  requests and persisting full candidate completions independently. No CPU simulation pool
  or multiprocessing manager is created on this path. The GPU service remains search-agnostic.
- Persist CPU-only algorithm state and partially evaluated seeds/cohorts atomically, without
  device handles or borrowed array names. Completed compatible fitness survives resume;
  unfinished candidates rerun on GPU. Checkpoints also bind critical run settings before
  any result exists. Native execution/precision identity distinguishes GPU fitness from CPU
  and proxy/validation fitness. Retain conservative existing source/data/config checks;
  scheduling replay and a crash-free exactly-once record stream are not required.
- A fast worker failure exposed a queue-refill race: a later failed submit could obscure
  prior successes and the original producer error. Defer that submit failure while consuming
  admitted futures; deterministic regressions cover both original-future and submit-only
  failures. Preserve primary exceptions through drain/checkpoint cleanup.
- Real offline CLI CUDA tests cover standalone and lazy scenario suites, fresh seeds, prompt
  results/Pareto writes, actual SIGINT during seed work and resume for further generations.
  CPU backtest/evaluator/raw Rust simulation entries and CPU pool/manager creation are
  forbidden. Focused search/identity/backend/session/CUDA checks pass 120 cases; broader
  optimizer/config regressions pass 456 cases. Docs have zero errors and two existing size
  warnings. Rust/kernel behavior is unchanged; validation uses the verified current extension.
- This is an initial integration, not replacement acceptance. Full suites run now; selective
  screening search policy, adaptive width/result cadence, finite static variants and
  candidate-dependent coin patches remain open, as do representative parity/performance
  comparisons and the known disabled-HSL specialized/general kernel equivalence debt.
  Retain the legacy backend until these acceptance items are resolved.

### 2026-10-06 — Restore safe disabled-HSL EMA compiler ablation

- The initial GPU-only CLI integration passed current-head auto review and every required
  Python/Rust CI check before merging into development. Master remains unchanged.
- Reproduced the old specialization test failure at selection, before output comparison:
  the sole-engine migration had forced the full layout for every dispatch. Simply restoring
  the former predicate would initialize per-coin controllers in the compact one-element
  array. Guard that history/controller binding in the Rust-owned one-side implementation.
- An offline synthetic probe also reproduced a semantic boundary: forcing compact state
  in disabled coin mode changes forced-delisting panic drawdown minimum/sum/maximum/count.
  Preserve full coin-mode state. Select compact EMA only for wholly disabled side/unified
  dispatches without enabling coin overrides; fused and TM kernels remain unspecialized.
  Broader ablation acceptance stays open rather than dropping required diagnostics.
- Compare every returned tensor against the general kernel across both sides, all three
  signal modes, forced delists and optional coin fill counts. Mixed-mode dispatches also
  match individual requests exactly, although they choose different compiled variants.
  All 26 device regressions pass. All 198 wider CUDA, parity-tool/comparator and real native
  optimizer CLI cases pass, including the formerly failing specialization case. The rebuilt,
  verified extension passes 330 Rust tests with one existing ignored benchmark; default-feature
  test compilation passes. Documentation checks have zero errors and two existing size warnings.
- A bounded warmed public benchmark fixture uses 64 candidates, 4,096 bars, seed 11 and
  ten alternating runs per variant. Median general/compact wall times are 42.4/36.0 ms
  for one coin and 209.9/197.5 ms for nine coins; kernel times are 39.6/33.2 and 207.2/194.8 ms.
  The modest improvement validates removing inactive work, not representative optimizer
  throughput or quality acceptance. Default coin-mode, TM, fused and further inactive-feature
  specializations remain future work.

### 2026-10-06 — Service-owned production batch tuning

- The disabled-HSL EMA slice passed clear current-head auto review and all required CI
  before merging into development. This work starts from that integrated source tree.
- Reuse the existing batch evidence controller inside the asynchronous service, with
  separate dataset measurements, cold/partial/failed-work exclusion, median smoothing,
  bounded trials, demand/headroom growth gates, cooldown and rollback. No calibration
  simulation is added. Width changes occur between completely validated producer batches.
  Numeric settings/off mode retain fixed ceilings; automatic settings tune in the service.
  Native CPU admission follows a bounded population window independently of device width.
- Preparation claims one cold request, discovers work and HSL/unstuck history-scratch
  bounds on its owner, and limits subsequent service batches to actual replay capacity.
  Otherwise an oversized outer batch could contain several serial device replays, delaying
  every returned future. Keep temporary runner references out of suspended resource contexts
  so inactive datasets can release device tensors and mutable scratch.
- All 142 focused policy/executor/native session/backend/device cases pass, with one existing
  skip. Six real CUDA service cases prove width changes preserve every metric/status without
  extra simulations, enforce work/scratch limits and release inactive runners. Eight real CLI
  cases cover fixed/automatic standalone/suite seeds, SIGINT, persistence and resume with CPU
  simulation/pool entry points forbidden. All 531 wider CUDA, preparation/residency, service
  and comparison-tool cases pass. Five documentation cases pass; doc checks have zero errors
  and two existing size warnings. Rust/kernel source is unchanged and the extension is verified.
- A bounded production-window comparison uses the public EMA synthetic fixture, one side,
  three coins, 32,768 bars, seed 7 and 8,192 submitted requests per run, with a rolling 1,024
  pending window and `long_base_qty_pct = 0.005 + (request_index % 512) * 0.00005`, requesting
  the fixture's default ADG, drawdown and fill-rate metrics. Fixed width 64 completes in
  99.14 s (82.63 requests/s); automatic completes
  in 64.88 s (126.27 requests/s). It accepts width 128 using unmodified 24-sample/30-second
  evidence windows, with measured baseline/trial rates 84.88/171.10 requests/s. Every request's
  metrics and terminal status match exactly. This is one sequential service comparison including
  startup, not repeated representative optimization/performance acceptance.
- Run-local evidence remains advisory and is not search/checkpoint state. Persistent calibration,
  demand-limited workload classes, admission/launch sizing, dispatch duration/delay, residency
  budgets and CPU preparation/result/evolution cadence remain open. Retain generation semantics
  and the legacy backend until full replacement acceptance.

### 2026-10-06 — CPU preparation/result pipeline

- The service-tuning slice passed clear exact-head auto review and every required CI job
  before merging into development. This subsequent CPU slice remains separately reviewable.
- Inspection and a synthetic CUDA optimization comparison show that filling a 256-candidate
  admission window before polling delays the first GPU submission by roughly 2.7 seconds.
  Submit after the first prepared candidate, then alternate CPU preparation and result
  servicing within a soft 50 ms latency target. Check interruption between preparations.
  Slow atomic preparation/scoring/storage can exceed the target; no hard deadline is claimed.
- Adapt returned completion grouping to measured CPU cost, starting at one and growing
  cautiously up to 256. Increased measured cost reduces grouping immediately. Exclude idle
  device waits, retain run-local measurements, and preserve existing evolutionary cohorts,
  candidate identities, canonical scores and checkpoint compatibility.
- Separate notification and returned-candidate bounds in the CPU session. Large suites
  need multiple notifications before yielding one candidate; cached/pending duplicates can
  yield many candidates from one result. Retain ready aliases until consumed, and preserve
  the original producer failure and earlier completed work across deferred submit failures.
- All 38 focused CPU cadence/session/backend/CUDA CLI tests pass, including a deterministic
  preparation barrier proving early submission/persistence and failure/resume after earlier
  GPU successes. All 46 additional dataset/planner/scoring/device-tuning cases and five
  documentation tests pass. No Rust/kernel code changes or CPU simulations are introduced.
- A four-run synthetic comparison alternates old/new/new/old over the public TM fixture,
  both sides, three coins, 128 bars, population/iterations 256, fixed GPU width 16 and a
  5-second checkpoint interval. Reset NumPy sampling to seed 12 before each optimization,
  alongside pymoo seed 12; retain the fixture's locked position/exposure bounds and variable
  long/short initial-quantity bounds. All 256 matched candidate metrics remain identical
  in every run. Old preparation-to-submission delays are 2.69/2.80 s; new delays are
  27.0/27.3 ms. Total times are 11.11/7.70/7.68/7.50 s, including the first run's cold
  preparation. Warm total throughput is broadly unchanged; this supports improved overlap
  and prompt result servicing, not a representative optimizer speedup claim.
- Repeat that comparison with two distinct lazy scenarios: the full three-coin 128-bar
  dataset and a 64-bar `[32:96]` window using coins 00/02, with mean suite reducers.
  Every matched metric and full-suite aggregate remains identical. Old preparation-to-submit
  delays are 4.66/4.79 s, new delays 43.0/41.8 ms; total old/new/new/old times are
  14.12/10.80/11.04/10.96 s including first-run cold startup. Warm throughput remains
  broadly unchanged in this bounded suite; larger scenario-locality cases remain open.
- Further cadence/evolution experiments, scenario-screening policy, representative parity
  and workload acceptance, and retirement of the legacy backend remain open.

### 2026-10-06 — Reuse screened simulator evidence

- The CPU pipeline slice passed clear exact-head auto review and all required CI before
  merging into development. This scenario-reuse slice remains independently reviewable.
- Add a bounded, run-local CPU cache of collector-validated simulator rows, keyed by the
  complete effective candidate identity, prepared dataset and exact request parameters.
  Promotion re-identifies and consumes those rows on the CPU poller, submits only missing scenarios, and still requires
  every complete-suite slot. Partial aggregate scores and full-candidate caches remain
  stage-separated. Cache eviction/loss only repeats GPU work, never changes calculations.
- Before collection or caching, verify a future's result matches its actual submitted
  request/dataset. Previously a mislabeled result could match a different valid slot of the
  same candidate; request-specific binding prevents wrong evidence from entering the cache.
- All 40 affected session/backend/real CLI cases and 33 registry/planning/scoring cases pass.
  Tests cover incomplete full-suite promotion, different candidate/data identities, eviction,
  immutable snapshots, malformed future binding, original failures, cancellation and resume.
  Two actual CUDA lazy-suite cases preserve independent full-result payloads. Screen-to-full
  promotion performs exactly one simulation per distinct scenario, with CPU simulations forbidden.
- This is reusable CPU collection infrastructure, not completed native screening search policy.
  Future integration must choose survivors on CPU, keep rejected partial observations out of
  full fitness/storage, persist stage progress safely and retain explicit objective/limit scenarios.
  Preserve full GPU seed/bootstrap evaluation and existing evolutionary cohort semantics.

### 2026-10-06 — Native scenario-screening search policy

- The scenario-evidence reuse slice merged into development only after clear exact-head
  automatic review and successful Python 3.12/3.14 and Rust checks. The following integration
  retains that review/CI gate and leaves the default branch unchanged.
- Extract the existing feasibility/Pareto/diversity selector into a CPU-only module shared
  with the legacy backend. Reuse existing screening configuration; add no worker-side search
  policy. Fully simulate seeds and initial parents, screen subsequent offspring, and submit
  only promoted complete offspring to pymoo survival alongside complete parents. Reject unknown
  or omitted explicit objective/limit scenarios before constructing the device service.
- Keep partial scores as compact CPU selection evidence, never evaluated fitness or result
  records. Checkpoint version 2 preserves screening completion and promoted/full-evaluation
  progress; obsolete experimental checkpoints are rejected explicitly. Lost row-cache evidence
  may repeat GPU work on resume. Cancellation still drains successes and saves usable state.
- All 763 affected legacy/native tests pass, including 18 native screening cases and 12 actual
  CUDA CLI cases. Coverage includes constrained/unconstrained NSGA-II/III, full seed reuse,
  no-op policies, malformed partial checkpoints and interruption/resume during screening,
  immediately after promotion and during full evaluation. CUDA CLI tests forbid CPU simulations
  and CPU pools and verify durable full results/Pareto with actual SIGINT and resume.
- A bounded synthetic CUDA experiment used the public Trailing Martingale, both-side fixture:
  three coins, 4,096 bars, fixture seed 7; population 32, three cohorts, search seed 12;
  GPU width 16; ADG/max and strategy drawdown/min with completion ratio at least 0.99.
  The mean-reduced suite had a full base scenario and a two-coin, 2,496-bar stress scenario.
  Screening base at fraction 0.25/minimum 4 retained 8 of each 32 offspring. In run order
  full/screened/screened/full, elapsed seconds were 16.17/12.13/12.03/12.34. Full runs stored
  96 complete candidates and submitted 96 requests per scenario. Screened runs stored 48
  complete candidates, screened 64 offspring and submitted 96 base/48 stress requests.
  Shared candidates had identical canonical metrics; final parent populations stayed at 32.
  Promotion did not repeat base simulations. Warm elapsed time was comparable despite less
  scenario work; this small experiment proves reuse/budget semantics, not speedup or search
  quality. Representative scenarios and repeated-seed Pareto quality remain acceptance work.

### 2026-10-06 — Explicit native parity execution

- Native scenario screening merged into development after clear exact-head automatic review
  and successful Python 3.12/3.14 and Rust checks. Keep the same gate for this tooling slice.
- Inspection found the standalone comparator still chose the legacy single-coin replay
  for one-coin inputs, while native optimization always uses the shared-account replay.
  Add `--gpu-engine native` to run the actual prepared CUDA service for 1..64 coins; keep
  default legacy comparison available for existing replay/Metal measurements. Reports label
  the engine even on failure and identify the replay family on successful comparison. No native
  failure falls back to legacy execution and no metric tolerance changes.
- Native comparisons allocate immutable shared inputs on the CPU, keep them alive through
  service close, and clean every allocation even if another cleanup fails. Preserve the
  original producer/consumer error. Device packing, buffers and replay construction stay
  on the owning worker. No simulation or optimizer routing changes.
- Native cold timing includes worker preparation and execution; separate preparation is
  explicitly unavailable. Bounded native diagnostics expose actual liquidation status
  alongside existing CPU fills, without exporting device internals or fabricated positions.
- The affected comparator/tool suite passes 112 tests on CUDA, including 12 native cases
  across both strategies, three side modes and one/two coins, plus two real native CLI
  cases and prepared nondefault/alias/wrapped exchange comparisons. Each native comparison
  in the 12-case matrix runs the CPU simulation once and matches independent
  shared-account replay metrics. Four allocation/consumer/cleanup regressions verify all
  shared segments are reclaimed and earlier failures preserved. Existing prepared-input
  identity checks, legacy comparison policies and measured discrepancies stay intact.
  This closes a path-selection gap in parity tooling, not the broader parity acceptance gate.
- An additional 140 existing CPU backend/preparation and artifact/Pareto plotting tests pass,
  with one environment-specific skip. This is focused compatibility coverage, not completion
  of the broader standalone CLI and dependency-isolation acceptance checklist.
- A 21-case seed-7, 5,760-bar matrix using the native service preserves completion coverage
  in every case and the two passing long-only TM fixtures. Strict measurement policies
  still expose the small residual differences recorded previously: EMA ADG relative error
  reaches 0.4911% and fill error 0.1967%; passive short/dual-side TM ADG/fill relative errors
  stay below 0.055%/0.035% in these fixtures. HSL/unstuck toggles do not prove controller
  transitions were exercised. No failures were concealed by fallback or wider tolerances.

### 2026-10-06 — Finite prepared execution views and native anchor resumption

- Native parity tooling merged into development only after completed exact-head automatic
  review with no reported findings and successful Python 3.12/3.14 and Rust checks.
- Prepare canonical representatives of finite anchor choices and side-enable boundary choices
  on the CPU. Apply optimizer policies and exact-last scenario overrides before deduplicating
  execution contracts. Register metadata views sharing immutable candle/BTC/timestamp references;
  numeric genes remain compact request parameters. The GPU service retains its existing lazy
  replay construction and one-active-dataset/one-owner-scratch residency policy.
- Select compatible views per scenario/exchange without adding evolutionary logic to the service.
  Full-candidate identity includes the selected views and all unscreened effective parameters.
  Mirror/fixed scenario policies eliminate ineffective variants. Unprepared continuous static
  inputs still fail before submission; both-sides-disabled replay remains unsupported.
- Save the fine-tune anchor plan in native checkpoints and restore it before building the
  optimizer shape. Original seed files are no longer required for resume. Existing evaluation
  identity rejects changed fixed anchor values. Older experimental anchored checkpoints without
  a stored plan cannot automatically restore anchors; use saved configs for a fresh run.
- All 1,154 affected native/legacy optimizer, preparation, executor and residency tests pass,
  including 37 finite-view and actual CUDA CLI cases. These exercise both strategies, initially
  enabled/disabled sides, shared packing with one replay scratch owner, exact-last suite policy,
  screening and interruption/resume after deleting seed files. CPU simulations and pools are
  forbidden in native search tests. Canonical request preparation already removes anchor plans
  from effective simulation configs; workers receive resolved backtest inputs, not anchor recipes.
- This implements finite supported variants rather than accepting arbitrary continuous coin
  patches. General compact coin-patch transport, representative performance/parity acceptance,
  adaptive residency budgets and legacy retirement remain open; do not weaken the acceptance gate.
- Automatic review identified positive position-count bounds that round to zero, such as
  `[0.4, 1.0]`. Enumerate canonical endpoint choices for each variable topology input, then
  deduplicate effective execution contracts. This reuses position rounding, step quantization
  and fixed/mirrored policies instead of duplicating eligibility arithmetic. Add regression
  cases for positive lower bounds, ties, stepped ranges and unreachable upper topology.

### 2026-10-06 — Controller-active parity evidence

- A separate native-service/verified-CPU experiment uses public synthetic inputs: seed 43,
  3,000 bars, two coins, both sides, one position per side, exposure limit 2, both strategies
  and all three HSL scopes. Controller EMA span is 2.5 minutes, cooldown 5 minutes, restart
  policy always. No tolerance policy or simulation code changed for this experiment.
- A red threshold of `0.000001` causes controller activity and substantial differences in
  fills, ADG and lifecycle metrics in all six cases. Repeat at `0.0005`, `0.002` and `0.01`
  rather than treating this extreme boundary test as representative numerical acceptance.
- At `0.002`, EMA cases match controller trigger rates in all scopes but disagree on time
  in red: CPU reports approximately 0.0051073 and GPU 0.0020422, a 60% relative difference.
  ADG relative differences range from 0.54% to 0.98%, and fill rates differ. At `0.01`, these
  EMA cases pass strict comparison but have no controller triggers. TM has no triggers in
  the three higher-threshold cases and retains small baseline ADG/fill differences.
- Initial source inspection shows CPU red time integrates reporting-scope state over elapsed
  minutes, whereas GPU reporting counts sampled RED tiers. Cooldown/flat-scope and observation
  timing need a focused diagnosis; this is a hypothesis, not a confirmed root cause. These
  findings reinforce the open parity acceptance item and do not justify broader tolerances.

### 2026-10-06 — Separate HSL reporting from panic intent

- Finite execution variants and checkpoint-owned anchors merged into development only
  after the rounded-position finding was fixed, answered in its original thread and
  cleared by a completed exact-head automatic review. All required CI checks passed;
  author review recorded the exact base/head/merge-base before a fresh merge gate.
- Controller tracing confirms that GPU time-in-red samples previously excluded terminal
  cooldown because they reused the current panic tier. Add a reporting-only projection
  that includes panic or halted cooldown, and use it across directional and shared-account
  samplers. Keep trading state, episode evaluation, lifecycle counters and metrics ABI intact.
- Repeat the 18-case seed-43 controller sweep before and after the change. Only six EMA
  time-in-red results change; every other requested metric stays exactly unchanged.
  At threshold `0.002`, GPU red coverage changes from 6 to 18 sampled bars out of 2,938;
  CPU remains 15 elapsed minutes out of 2,937. The residual absolute fraction difference
  is approximately `0.0010194` (0.10194 percentage points), or 16.64% relative error.
- CPU observations distinguish completed-bar valuation from scope-flat execution boundaries.
  The remaining difference needs a separate timing/episode diagnosis; including cooldown
  fixes a confirmed omission without establishing complete controller parity. Preserve the
  existing tolerance policy and open acceptance checklist.
- The broad CUDA run passes 232 cases, with 13 Metal-only skips. Four failures are
  stale expectations that disabled single-side EMA side/unified dispatches retain full
  HSL state; update those expectations to the documented compact layout while keeping
  metric-equivalence assertions. Two dual-side unstuck comparisons report 71 GPU fills
  versus 72 CPU fills. Rebuild the unchanged development baseline and reproduce both
  failures identically; they predate this reporting change and remain acceptance debt.
- All 330 Rust tests pass (one ignored), and default-feature test compilation succeeds.
  Rebuild and verify the current extension after the baseline comparison. Add actual
  replay controls restoring only the previous reporting expression, so regression checks
  can require changed red coverage with every other raw output unchanged.
- All 29 focused CUDA checks pass: four controller-state probes, six real replay controls,
  all 18 disabled-policy permutations and compact/full disabled-HSL output equivalence.
  The six controls exercise both strategies and all HSL scopes and require every raw
  output other than red sample coverage to remain exactly unchanged.

### 2026-10-06 — Restore the intended unstuck parity fixture

- The unstuck lookback fixture inherits an explicit canonical zero entry cooldown,
  then attempts to set 1,000 minutes through retired `risk.entry_cooldown_minutes`.
  Canonical precedence keeps zero, allowing repeated partial entries and changing
  the intended allowance-exhaustion experiment. Do not weaken its parity assertions.
- Set `entry_cooldown.base_duration_minutes` directly and require the CPU fill trace
  to contain only the initial entries. The complete 26-case CUDA unstuck suite passes,
  including exact fill counts and final positions for long, short and shared portfolios,
  finite/all-history budgets, scratch admission, replay reuse and interrupt reset.
- In the two-sided fixture, CPU fills change from 72 in both history modes to 21 with
  finite lookback and 10 with all history; GPU matches the corrected experiment.
  This is a test-input correction, not a trading or numerical-policy change. The original
  high-churn 71/72 observation is retained as evidence and is not proof of general parity.
- The combined HSL/reporting/unstuck/native CUDA suite passes 244 tests with 13
  Metal-only skips, including native optimizer CLI screening, interruption and resume.
  The reporting-only change merged into development after completed exact-head automatic
  review with no findings, exact-target author review and successful required CI.

### 2026-10-06 — CPU lowering for coupled candidate dependencies

- Reproduce a native-preparation gap using canonical synthetic requests: with coupled
  unstuck EMAs and a fixed coin patch, a searched strategy span of 2 is rejected while
  the prepared span of 20 succeeds. Canonical materialization changes derived coin
  span copies, even though the replay already supports request-owned span inheritance.
- Prefer CPU lowering over expanding the GPU request ABI for this dependency. Exclude
  redundant coupled coin span copies from execution identity, retain genuine strategy
  pins and bind the coupling policy explicitly. Uncoupled coin-span changes still require
  compatible prepared data. Arbitrary continuous coin-patch transport remains separate work.
- Project registered worker views to ordinary backtest sections on the CPU. Remove
  optimizer/bookkeeping metadata, omit inherited coin span copies and retain pinned
  counterparts. Immutable history references stay shared; canonical request values and
  effective candidate identity remain CPU-owned. No kernel or metric change is needed.
- Full coupled-suite CLI checks expose a resume gap: plain-replay exports contain
  candidate-dependent scenario spans that raw comparison mistakes for changed recipes.
  Materialize incoming recipes against each saved candidate before comparison. Retain
  all other scenario inputs, reject altered stored spans and leave checkpoint policy
  checks intact; this applies to CPU/legacy suites as well as native optimization.
- Final affected checks pass 185 cases on CUDA, including both strategies, standalone/lazy
  suites, scenario span overrides, enabled unstuck EMA consumers, pinned/inherited spans,
  independent effective replay comparisons, native CLI screening/interruption/resume and
  evaluation-contract guards. Another 31 legacy resume/context checks pass. Documentation
  checks pass with two existing size warnings. Rust and numerical tolerance policy are unchanged.
- The parity-fixture correction merged into development after completed exact-head automatic
  review with no findings, author semantic sign-off and successful required CI.

### 2026-10-06 — Share finite unstuck history with EMA replay

- Canonical loss-expiry fixtures expose a semantic omission: EMA shared-account GPU
  replay always uses the all-history realized-PnL peak, ignoring a finite configured
  lookback. With an eight-bar window, exact CPU long/short/dual-side runs produce
  20/23/24 fills while the baseline GPU produces 4/4/8. The all-history controls
  agree. This materially suppresses later unstuck closes, not decimal noise.
- Bind the existing account-owned rolling window in directional and fused EMA
  replays. Record fills in account order, refresh expiry before generating unstuck
  orders and fail closed on history overflow. Keep HSL and the separate conservative
  realized-loss gates on their existing independent contracts.
- Move duplicate TM history allocation/admission into the common runner. Both
  strategies specialize from effective side/coin consumer flags, including a zero
  prepared allowance which may be varied by requests. Disabled consumers and
  all-history policy omit rolling scratch. Keep per-candidate storage bounded by
  one coalesced event per candle and account for combined history admission.
- Extend exact CPU regression coverage to both strategies, one/two coins and all
  active-side topologies. Exercise native EMA service reuse with CPU simulation
  forbidden, plus replay ordering, batch limits, effective overrides and overflow.
  This slice does not establish complete controller parity or widen tolerances.
- All 71 focused unstuck cases pass, including finite history with compact/general
  disabled-HSL equivalence. The final combined run passes 573 tests with 13
  Metal-only skips, including enabled HSL, capacity specialization, service tuning
  and offline native optimizer CLI screening, interruption and resume. All 330 Rust
  tests pass (one ignored), default-feature compilation succeeds and the rebuilt
  extension is source-fingerprint verified. Documentation checks have no errors.
- Coupled-span CPU lowering merged into development only after completed exact-head
  automatic review with no findings, exact-target author review and all required CI.

### 2026-10-06 — Reproducible cohort measurements

- The finite EMA unstuck-history fix merged into development after a completed
  exact-head automatic review with no findings, exact-target author review, and
  successful Python 3.12, Python 3.14 and Rust CI. Recheck every review surface and
  current base/head immediately before the SHA-pinned merge; keep master unchanged.
- Add a public synthetic cohort benchmark for serial CPU preparation/simulation,
  direct shared-account GPU replay and native-service completion latency. Bind
  completions to their submitted identities and require exact direct/native metrics
  and liquidation status across widths and repeated cohorts.
- Retain strict per-candidate CPU/GPU comparisons. Report two-objective Pareto
  membership, pair ordering including ties, GPU-choice regret measured on CPU, and
  explicitly requested diagnostic feasibility limits. These fixed cohorts do not
  establish repeated-seed evolutionary search quality or multicore CPU throughput.
- Distinguish first-use samples from warm repetitions; retain and disclose compiler
  cache scope rather than presenting service first use after direct warmup as a cold
  compiler measurement. Expose observed batch sizes and controller evidence, not
  just configured widths. Label Torch memory separately from total device VRAM.
- Run the public default seed-7 recipe and seed 43 with widths 16/automatic and a
  fresh CuPy compiler-cache directory. Both reports identify the measured Python
  tree and verified Rust extension. Direct/native results match exactly in
  every measured cohort; automatic width 64 sees demand no greater than 16 and
  accumulates no eligible tuning evidence. Do not report this as a tuning success.
- All four CPU/GPU Pareto member sets agree and GPU maximum-ADG selections have
  zero CPU regret; EMA seed 43 has three ADG and two drawdown pair-order changes.
  Requested diagnostic feasibility limits have no flips. Strict metric gates still
  expose residual differences, so representative controller, optimizer and search
  acceptance remains open. Record recipes, timings, errors and limits in the
  [cohort benchmark documentation](../gpu_cohort_benchmark.md).
- Validation: 180 benchmark, comparator, native-service parity and CLI checks pass
  on CUDA with the current source-verified Rust extension. The slice changes only
  development tooling and documentation, not simulation kernels or optimization.
- Automatic review identified consumed tuning windows disappearing from the final
  controller snapshot. Preserve cumulative eligible samples/seconds and completed
  windows alongside the pending remainder, without changing execution decisions.
  Add a regression for completed windows, rejected trials and incomplete evidence;
  require changed-head validation and automatic re-review before integration.
  Changed-head validation passes all 181 affected CUDA parity/benchmark/CLI cases;
  source-only benchmark regressions pass 18 with two device cases deselected.

### 2026-10-06 — Cohort-tool integration and CPU preparation foundation

- The cohort benchmark merged into development in [PR #1899](https://github.com/enarjord/passivbot/pull/1899)
  after fixing the tuning-evidence review finding, completed clear changed-head
  automatic review, exact-target author sign-off and successful Python 3.12/3.14
  and Rust checks. A fresh review/CI/base/head gate preceded the SHA-pinned merge.
  The default branch and simulation semantics remain unchanged.
- Warm request streams can underfill GPU batches when canonical CPU preparation
  takes longer than the service's queue-wait allowance. Measure actual batch sizes,
  request preparation and result export separately before adding scheduling state;
  a larger fixed delay alone is not a general throughput solution.
- Candidate preparation repeatedly constructs immutable strategy/configuration
  metadata through canonical helpers. Investigate reuse of that static work while
  preserving caller-owned mutable copies, effective overrides and explicit errors.
  Keep trading decisions in their existing owners and retain CPU functionality.
- Search comparisons must report the normalized, effective genome and bounds.
  Config loading hydrates omitted bounds; a sparse input mapping does not by itself
  define a restricted search. Separate preparation/kernel timing from evolutionary
  policy and sampling differences when assessing resulting Pareto quality.
- Resolve related optimizer keys through one operation-local strategy path table
  in bounds validation, key extraction and native static execution projection.
  Share the existing single-key resolution rules, preserve ordered/duplicate keys,
  and keep mode/shape-dependent paths scoped to the unchanged input config. Empty
  and portfolio-HSL-only groups retain their lazy metadata/error behavior.
- Build the key-extraction fallback template only when bot configuration is absent.
  Prefer bounded reuse within an operation over new persistent metadata caches,
  mutable cached defaults or invalidation machinery. Preserve existing canonical
  loading, explicit bounds, overrides and simulation ownership.
- A bounded synthetic before/after check confirms identical prepared requests and
  cleaned result exports. Preparation timing improves in that fixture; export
  timing has no clear improvement. This does not establish full-search throughput,
  quality equivalence or representative performance acceptance.
- Validation: 352 configuration/path/planning/warmup checks and 320 CPU/native
  integration checks pass with the source-verified Rust extension, including CUDA
  suites, screening, anchors, interruption and resume while forbidding CPU
  simulations and worker pools in native optimization. Six documentation tests
  pass; documentation checks report zero errors and two existing size warnings.

### 2026-10-06 — Adaptive service coalescing

- CPU preparation reuse merged into development in [PR #1900](https://github.com/enarjord/passivbot/pull/1900)
  after completed clear current-head automatic review, exact-target author sign-off,
  successful Python 3.12/3.14 and Rust CI, and a fresh all-surface review/identity
  gate immediately before SHA-pinned merge. Master remains unchanged.
- Requests arrive in CPU preparation bursts. A median over individual submission
  gaps can see only nearly adjacent submissions and miss the slower gaps between
  bursts; a bounded prototype with that estimator did not improve dispatch sizes.
  Use burst-aware arrival evidence rather than a larger universal fixed delay.
- Add service-owned adaptive accumulation, separate from width and search policy.
  Learn only within-active-work arrival gaps and successful warm replay durations;
  retain cold-shape rejection, per-dataset isolation, idle-tail and absolute bounds.
  Preserve numeric delay overrides and tuning-off behavior. No new CPU/device
  coupling, simulation, metric, scoring or checkpoint state belongs in this policy.
- Compare two small seeded searches and a larger cohort against fixed accumulation.
  Completed candidate configs and metrics match exactly and every result is durable;
  the small cases benefit, while the larger case shows no clear timing improvement.
  These bounded measurements do not prove universal throughput gains or evolutionary
  quality equivalence. Retain the broader acceptance checklist and parity debts.
- Validation: 178 affected service/native/benchmark tests pass with the verified
  Rust extension on CUDA, including suites, screening, anchors, seeds and clean
  interrupted resume with CPU simulations forbidden. Source-only lifecycle and
  coalescing checks plus six documentation tests pass (82 total); documentation
  checks report zero errors and the two existing size warnings. Fixed/automatic
  search comparisons preserve all execution config sections and returned metrics.

### 2026-10-06 — Shared CPU coin-parameter encoding

- Adaptive coalescing merged into development in [PR #1901](https://github.com/enarjord/passivbot/pull/1901)
  after completed clear current-head automatic review, exact-target author sign-off,
  successful Python 3.12/3.14 and Rust checks, and a fresh all-surface review/identity
  gate immediately before the SHA-pinned merge. Master remains unchanged.
- Continuous coin patches affect more than kernel indexing: eligibility checks,
  effective RMS demand, feature ablation and truncated replays currently consume
  prepared pins. Keep incompatible changes explicit until that complete transport
  boundary is supported; finite anchor views and coupled scalar inheritance remain.
- Consolidate EMA/TM coin encoding in one CPU-only module shared by single/multicoin
  adapters. Reuse canonical payload values and override precedence, preserving NaN
  inheritance, eligibility/forced-active sentinels, coupled spans, TM gate/retracement
  encoding and exact patches separately from float32 values. Remove duplicated shared
  risk, cooldown, unstuck and HSL packing from the replay service.
- This is a smaller preparation foundation, not general continuous coin transport
  or a change to simulation, precision, kernels, scheduling, scoring or resume.
- Validation: 409 affected service, coupling, native CUDA/session/dataset and parity
  checks pass with the source-verified Rust extension. Native integration continues
  to forbid CPU simulations/pools. A bounded differential check preserves full
  matrices and exact contracts against the former encoders for both strategies,
  all HSL modes and coupled/uncoupled spans without input mutation. The CPU-only
  import/execution boundary and six documentation checks pass; AI documentation
  reports no errors and the two existing size warnings.

### 2026-10-06 — EMA realized-loss admission shares finite fill history

- Shared coin encoding merged into development in [PR #1902](https://github.com/enarjord/passivbot/pull/1902)
  after completed clear exact-head auto review, exact-target author sign-off and
  successful Python 3.12/3.14 and Rust CI. A fresh all-surface review/identity gate
  preceded the SHA-pinned merge; master remains unchanged.
- Reproduced a material admission difference: finite EMA loss caps kept blocking
  closes on GPU after old losses expired on CPU. Loss-only requests additionally
  omitted the rolling history when auto-unstuck was disabled. All-history controls
  retain their prior behavior. These are semantic differences, not float32 noise.
- Use one account-owned fill-PnL drawdown for EMA loss admission and auto-unstuck.
  Reuse existing bounded/coalesced scratch and preserve intrabar peaks. Prepare
  history for either effective consumer; omit it when both are disabled or scope
  is all history. Retain generation-time shared loss reservations and HSL accounting.
- Native one-coin and fused long/short requests use the shared-account engine.
  Retained legacy single-coin EMA and TM general loss gates still have documented
  conservative policies; do not claim those acceptance gaps are solved here.
- Validation: 487 affected CUDA checks pass across finite loss history, existing
  unstuck and loss reservations, native service/optimizer and parity integration.
  New cases compare real Rust fill counts, positions and cash, forbid CPU replay
  in native requests, and exercise consumer ablation, compact/general HSL, bounded
  batching, reuse and fatal overflow. A wider bounded loss-cap sweep agrees on
  fill counts and positions. Rust tests pass (330, one ignored), default-feature
  test targets compile, and the rebuilt extension's source stamp is verified.
  Source-only preparation and documentation checks pass; AI documentation has
  zero errors and the two existing size warnings. Tolerances are unchanged.

### 2026-10-06 — TM loss-admission reproduction and local foundation

- EMA finite loss history merged into development in [PR #1903](https://github.com/enarjord/passivbot/pull/1903)
  after completed clear exact-head automatic review, exact-target author sign-off,
  successful Python 3.12/3.14 and Rust CI, and a fresh all-surface review/identity
  gate immediately before the SHA-pinned merge. Master remains unchanged.
- Reproduced a separate TM semantic gap with offline flat-price fixtures: an ample
  configured loss allowance admits fee-only closes on CPU, while the GPU's zero-loss
  envelope rejects them. One/two-coin and long/short/fused cases differ in actual
  fills and cash. A 36-case regression matrix has 18 expected baseline failures;
  zero-budget and tight all-history controls pass. This is not precision noise.
- Keep this next change local until shared admission is implemented and validated.
  Replacing the history expression or loosening individual fill checks is insufficient.
  Rust reserves projected negative PnL for generated orders across coins and sides;
  profitable orders do not expand that reservation budget. Reducer priority/fallback,
  ordinary order iteration, executable sizing, dust and panic exemptions all matter.
- Preserve Rust's next-candle recursive expansion decision before admission. Reserve
  unfilled emitted orders as well as reachable orders, and consume admitted intent at
  fill time without checking it again against the next candle's price or cash.
- Start with an immutable grid context and a bounded iterator over duplicate-merged
  close groups. All existing group selectors share it; streaming future admission
  avoids regenerating the complete ladder for each group. A bounded differential
  CUDA probe matched 1,884 contexts and every one of 33,031 groups bitwise, including
  500-group ladders, both exposure slopes, prefix merging and market sizing. This
  validates the refactor, not the still-unimplemented loss-admission replacement.
- Explore compact persisted admission flags and bounded quantity adjustments rather
  than per-candidate full order histories. Keep the disabled gate compiled out,
  reuse finite fill history with unstuck, and include directional/fused, temporal
  replay, scratch batching, reuse and native requests with CPU simulations forbidden.
  Existing tests asserting the conservative envelope must become correct budget
  controls with Rust parity coverage; acceptance tolerances remain unchanged.
- The retained directional TM engine has loss-gate compiler specialization, while
  the native shared-account TM library does not currently expose that specialization.
  Add it at the shared-account execution boundary with the replacement; do not assume
  a legacy guard also removes native admission scratch or work.
- Refactor validation: 194 affected checks pass in the CUDA environment, including
  existing recursive fills/market reducers and finite-history/temporal replay cases,
  plus a new bounded streaming regression. Six Metal-only cases are skipped. One
  direct shader probe needed its helper-call interface updated; its dust-allocation
  assertions are unchanged and pass. Rust tests pass (330, one ignored), default-feature
  test targets compile, and the rebuilt extension is source-verified. Documentation
  checks pass with zero errors and the two existing size warnings. The admission
  replacement remains open; these results do not erase its reproduced failures.

### 2026-10-06 — Local shared-account TM close admission

- Replace the shared-account TM zero-loss envelope with one generation-time
  admission stage across coins and both sides. Finalize executable quantities,
  rank reducer alternatives globally, reserve projected negative PnL, then admit
  ordinary closes in Rust's finalized iteration order. Projected profits do not
  fund other orders; panic closes remain exempt. Fill execution consumes emitted
  intent without repeating admission at the next candle's price or balance.
- Keep persistent intent compact: one immutable grid context, 500 admission bits
  and at most two trim/dust quantity adjustments per position. Reuse a bounded
  temporary close vector across positions and reducer alternatives. Compile this
  state and stage out when the configured loss gate is disabled.
- Share the finite fill-PnL tape with auto-unstuck for TM as well as EMA. Cache
  temporal replay sizes by the compiled variant and allocate buffers by their
  actual byte size, since admission ablation changes the persisted state layout.
- CUDA validation resolves all 18 reproduced failures in the 36-case Rust comparison
  matrix. The full new admission/streaming modules pass 135 tests; an additional
  regression passes twelve long/short dust, minimum and aggregate-trim scenarios.
  Coverage includes 54 recursive slope/market/WEL/TWEL comparisons, 24 combined
  unstuck/loss history comparisons, finite/all history, compiler ablation, temporal
  replay, scratch batching, candidate variation/reordering and native requests with
  CPU simulations forbidden. Direct account cases cover unfilled reservations,
  reducer priority/fallback, panic exemption and execution-cost projection.
- Adjudicate legacy assertions that deliberately required the old conservative
  envelope: rejection controls now use zero allowance; ample allowance admits
  fee-only closes and matches Rust. Corrected controls and shader smoke pass 24
  tests. The final existing recursive/market/reducer/loss-gate slice passes 131
  checks, and shared history/HSL/temporal replay passes 62 with six Metal-only
  skips. Preparation/service tests pass 275 checks. Rust tests pass (330, one
  ignored), default-feature test compilation passes, and the rebuilt extension is
  source-verified. The real native parity CLI passes single-coin and fused synthetic
  fee-only-close comparisons, including unchanged metric policies and matching limit
  violations. Documentation checks pass with zero errors and the two existing
  size warnings. Repository-wide formatting has pre-existing unrelated differences;
  leave them outside this behavior change.
- A bounded direct CUDA measurement uses the synthetic cohort generator's seed 7,
  16 candidates, four coins, both sides and 10,080 bars on an RTX 3070 Ti Laptop GPU.
  Three warm repetitions have median times 2.44 seconds for disabled/specialized
  admission, 2.48 for disabled/general, and 2.92 for enabled allowance 0.1. Disabled
  specialized/general outputs agree. Compiler-reported local storage is respectively
  7,488, 21,664 and 21,728 bytes per thread; these are not total device-memory or
  persistent-state measurements. Timing includes raw packing/decoding, synchronization
  and output copies, excludes preparation/CPU simulation and uncontrolled first-use
  compilation, and does not establish generic throughput acceptance. Preserve this
  genuine disabled-feature ablation; broader optimizations remain future work.
- These results do not establish general simulator parity or goal completion.
  Any implementation PR still needs completed exact-head automatic review, resolved
  findings, author sign-off and CI before development-only integration.

### 2026-10-06 — Service and optimizer lifecycle acceptance evidence

- PR #1906 completed current-head automatic review, author review and all required CI
  before development-only integration. Shared-account TM loss admission now uses the
  configured finite/all history and finalized close reservations; update the inventory
  so the removed envelope is not carried forward as a native limitation.
- Add the [acceptance evidence map](gpu_optimizer_acceptance.md), including the ownership,
  state and dependency simplification against screening/CPU validation. Keep the overall
  goal and retirement gates open; representative metric approximations, resource/performance
  measurements and CPU-preserving cutover remain work.
- Add real CUDA tests for both strategies with immutable metadata/borrowed arrays,
  mutation of a queued request, immediate capacity replacement after a completed result,
  unfinished later work and repeated candidate/shape isolation. Each case compares five
  isolated references with 165 service requests. Repeated traffic stays inside its warmed
  Torch allocation envelope; this does not certify total VRAM, host RAM or disk bounds.
- Strengthen native optimizer CLI tests to read the first full record and a Pareto member
  through independent file handles before cohort completion or shutdown. Run standalone,
  suite, seed/bootstrap, screening/promotion, fixed/automatic dispatch, interruption/resume,
  anchor restoration without seed files and coupled coin-span cases with CPU simulation
  APIs and CPU worker construction forbidden.
- Validation: 42 CUDA ownership/dispatch/CLI cases pass; a final run passes the two new
  service cases and eight anchor CLI cases (52 distinct device cases across the runs).
  The focused CPU orchestration/scoring/residency corpus passes 144 checks. Six documentation
  checks pass; AI checks report zero errors and two existing size warnings, generated
  registry is current, Python syntax and changed Markdown links validate. The device
  runtime's 360 Python/Rust/shader source files match the development tree and its actual
  Rust extension passes source verification. Verification failures remain fatal; passing
  evidence uses the actual source-verified runtime.
- Close the implemented ownership, canonical scoring, suite/seed handling, prompt storage
  and no-CPU optimize/bootstrap/resume checklist items with this scoped evidence. Retain
  the broader memory and final simulator acceptance boxes; changing their scope to obtain
  completion would weaken the contract. This slice still requires current-head automatic
  review and CI before its own development integration.

### 2026-10-06 — Per-step recovery distributions and shared dispatch bounds

- The lifecycle acceptance slice (PR #1907) merged into development after completed
  clear exact-head automatic review, author sign-off and successful required CI.
  Broader metric, resource, tuning and legacy-retirement acceptance remains open.
- CPU recovery distributions measure strict time-to-exceed for every strategy-equity
  sample, including unresolved plateaus and terminal tails. Hourly GPU sampling can
  lose short recoveries independently of float32 simulation error. Requested GPU
  distributions now retain every simulation step and use the existing GPU stack/
  histogram reducer; no CPU simulation or full-history transfer enters optimization.
- Remove globally cached mutable reduction buffers. Scratch is dispatch-local on the
  input tensor's device, and replay retains only the current sample-buffer shape.
  Include samples, optional contiguous-view copying and reduction storage in history
  budgeting. Native and retained service batches share the same outer dispatch bound,
  avoiding concatenation of oversized histories after internal kernel splitting.
- These changes preserve strict comparisons and disabled-feature compilation.
  Sampling-resolution repair does not establish float32 trajectory parity or close
  the broader approximation inventory. Validation and long-fixture observations are
  recorded in the [acceptance evidence map](gpu_optimizer_acceptance.md).
- Validation: 229 CUDA replay/lifecycle/CLI checks, eleven focused recovery/truncation
  kernel checks and 445 service/tuning/parity/search checks pass; one Apple-only check
  skips on CUDA. Six documentation checks pass. All tested source/test bytes and the
  loaded Rust source stamp are verified; Rust source is unchanged.
- Repeated thirty-day public fixtures preserve input identity and all non-recovery
  metrics. Mean recovery falls from hourly GPU values 0.053182870/0.041608796 days to
  0.006702424/0.000853817 for EMA/TM, versus CPU 0.006653244/0.000852400. Both p95
  observations agree closely. Smaller mean/tail differences remain unassessed;
  eleven undefined policies keep the full reports `comparison_incomplete`.
- PR #1908 integrated into development after completed clear exact-head automatic
  review, author review and all required CI. Broader goal gates remain open.

### 2026-10-06 — Canonical traded-volume contributions and suffix reduction

- Standalone public two-day fixtures reproduce two semantic defects: applying the
  contract multiplier again scales normalized volume, and dropping partial UTC days
  loses most weighted suffix contributions in short scenarios. CPU/GPU agreement on
  synthetic multiplier-one cases alone would not expose the normalization defect.
- Shared EMA/TM replay now records actual quantity/price/post-fill-balance volume.
  Only requested weighted volume captures per-step contributions and fill presence;
  a separate Rust-owned GPU reducer applies actual-horizon suffixes. Return one
  compact value per candidate, with no CPU replay or full-history host transfer.
  Preserve disabled compilation, current-shape ownership and shared history budgeting.
- CPU preparation and metrics processing remain orchestration responsibilities;
  simulator-specific contributions and reductions remain inside the GPU service.
  Existing standalone CPU analysis is unchanged. Retained legacy directional single-
  coin helpers keep their separate approximation until replacement acceptance.
- Independent history reduction agrees closely with GPU reduction. One busy short
  TM fixture still has a 0.273% weighted-volume trajectory discrepancy and different
  fill counts. Record a case-specific regression bound and the strict standalone
  mismatch rather than widening general parity policies or claiming simulator parity.
- Repeated thirty-day public fixtures preserve input identity, all CPU metrics and
  fourteen unrelated GPU metrics. Weighted-volume discrepancies shrink from about
  4% to 0.084%/0.031% for EMA/TM; their broader materiality remains unassessed.
- The [acceptance evidence map](gpu_optimizer_acceptance.md) gives the reproducible
  recipe, resource contract and residual classification. Final-source validation passes
  817 affected Python/CUDA checks, 330 Rust tests, default-feature compile checks and
  five documentation checks. Two outdated dispatch fixtures found in a broader run
  are corrected and their enabled/disabled argument layouts verified. Actual Metal
  execution and broader resource/performance acceptance remain open. This completed
  slice still requires exact-head automatic/author review and required CI before
  development integration.

### 2026-10-06 — Fill-gap population audit and refreshed cohorts

- PR #1909 integrated into development after completed clear automatic review of
  its final commit, author sign-off and successful Python 3.12/3.14 and Rust CI.
  Master remains outside the development integration target.
- A standalone audit found that shared GPU gap histograms counted filled candles
  while CPU percentiles count every fill, including same-candle zero gaps. Restore
  those gaps from existing compact fill/histogram counts; preserve immutable output
  buffers, boundary gaps, time-weighted moments and the positive-gap bin approximation.
  No simulator changes, history exports or implicit CPU simulations are needed.
- Six two-day public EMA fixtures reproduce a one-minute p95 population error;
  restoring zeros agrees with CPU across long/short/both sides and two/four coins.
  Remaining TM positive-gap bin/trajectory differences stay visible in strict reports.
  Do not treat this correction as acceptance of all histogram approximations.
- Validation passes 180 affected metric/parity/native Python and CUDA checks,
  including ten new real-device cases, and five documentation checks. The tested
  source and unchanged loaded Rust extension are verified. This slice still requires
  completed automatic/author review and required CI before development integration.
- PR #1910 automatic review identified a restricted hardware-test payload missing
  the newly required fill count. Updated that caller, audited all direct reducer
  callers and passed both reported long/short cases plus twelve existing streamed-
  gap strategy/topology/interval cases on CUDA. The corrected head requires fresh
  automatic/author review and CI; the earlier review does not authorize its merge.
- Refreshed public seven-day cohorts retain input/candidate identities and prior
  strict comparisons/rankings. Warm direct/native throughput is comparable; all
  native outputs match direct replay and tested diagnostic limits have no flips.
  Automatic width again receives no eligible samples in sixteen-candidate cohorts.
  Sampled host/driver memory exceeds Torch-only figures; larger suite and optional-
  history costs remain explicit acceptance work in the evidence map.

### 2026-10-06 — Refresh development from master after bounded allowance/HSL changes

- Integrate master `492246b8c55bfe8cc4a45a407e66a270a7d20608`, including
  PR #1905's always-bounded excess allowance and opt-in per-side coin-HSL budget
  scaling, plus PR #1904's configuration cleanup/export tool. Keep all redesign
  work on development; this integration does not publish it to master.
- Keep CPU-owned canonical candidate parameter preparation and the development
  kernels' current-balance minimum-effective-cost behavior. Apply the new HSL
  parameter in the extracted parameter module rather than restoring old service
  preparation or conservative screening helpers removed by the redesign.
- Carry budget scaling through shared single/multicoin and directional replay;
  preserve disabled-feature compilation, streamed metrics and dispatch-local scratch.
  Add both-strategy/side parameter regressions and native GPU-only CLI, suite,
  interruption and resume regressions with scaled budgets and authored coin patches.
- Retained GPU checkpoints use master's updated signature. Native saved fitness
  already requires matching Python and Rust implementation identities, so old-source
  checkpoints cannot authorize fitness reuse after this simulator change. Existing
  benchmark/parity results remain evidence for their recorded implementation only;
  broader acceptance of the integrated simulator remains open.
- PR #1910 integrated into development after clear final-head automatic review,
  resolved feedback, author sign-off and all required CI; include it in this refresh.
- Development integration requires completed current-head automatic review,
  addressed findings, author sign-off and successful required CI.

- Validation passes 330 Rust tests (one existing ignored), default-feature compile
  checks and a rebuilt source-verified extension. Broad configuration, HSL and
  orchestration coverage passes 1,781 checks. Final-source metric, implementation-
  identity, checkpoint and native coverage passes 508 unchanged cases; corrected
  parameter/scaled-HSL coverage passes eleven cases, including all four new lifecycle
  cases. Forty-seven focused CUDA allowance, minimum-cost, disabled-feature and
  streamed-metric kernel checks pass. Six documentation checks pass.
- The new lifecycle fixtures use canonical per-coin WEL placement and decode the
  existing overlay result stream before checking fixed policy. Full Pareto exports
  and checkpoint policy are checked independently. No simulator change was needed
  beyond master's HSL policy and its integration into the extracted parameter encoder.
  Actual Metal execution and broader final simulator acceptance remain open.
- Automatic review caught a renamed HSL member in a generated selection probe.
  Correct it to `scale_hsl_budget`; the exact selection-phase regression compiles
  and passes on CUDA. Production source is unchanged by this review correction.


### 2026-10-06 — Requested-metric cohorts and numerical reference checks

- PR #1911 integrated master into development after addressing the generated HSL
  probe finding, completed final-head automatic review, author sign-off and all
  required CI. Development retains the reviewed tree; master receives no redesign.
- Extend the standalone cohort tool with requested GPU metrics, resolved explicit
  tolerance policies and repeatable canonical scalar limits. Keep the three core
  comparisons and ADG/drawdown ranking; added objectives are not silently treated
  as covered ranking dimensions. Undefined policies remain unassessed.
- Refresh four public seven-day cohorts on the integrated simulator. Core comparison
  rows, ranking and tested limit outcomes remain unchanged. Optional recovery and
  weighted-volume captures are measured separately; broader materiality, HSL-tail,
  suite resources and completed-work tuning remain open.
- A 16-versus-1+15 TM replay retains identical raw daily summaries, timestamps,
  fills and drawdowns but differs by six float64 units in weighted ADG. Allow at most
  eight float64 units only in the tool's native/direct GPU reference check; report
  each accepted discrepancy and distinguish exact agreement. Preserve CPU/GPU
  policies, identity/liquidation checks and failure on larger/non-finite differences.
- Validation passes 122 affected comparison/cohort Python and CUDA checks, plus the
  new exact-raw-replay batch-shape regression. Seven-day requested-metric cohorts
  pass execution with explicit numerical observations; CPU/GPU parity is not declared
  generally accepted. Documentation and final source checks precede publication.
- This tool slice requires completed exact-head automatic/author review and successful
  required CI before development integration. Overall legacy-retirement gates stay open.


### 2026-10-06 — Fixed-memory fill-gap refinement

- PR #1912 integrated after clear exact-head automatic/author review and all required
  CI. Latest master remains included through PR #1911; no redesign is sent to master.
- Coarse fill-gap bins changed meaningful third-objective EMA Pareto membership in
  the public seven-day cohorts. Refine counts from 128 to 512 bins rather than retain
  full histories or add an exact sorting path. Count storage grows by 1.5 KiB per
  candidate; trading state, fill populations and streamed moments are unchanged.
- Keep initial-entry intervals' separate 128-bin shader/output format and decoder.
  Update standalone kernel probes to allocate the actual fill-gap surface.
- Before/after identities and all non-gap CPU/GPU metrics match exactly. EMA p95
  maximum error falls from 3/2 minutes to 0/0.1 minute; one third-objective front
  matches CPU and the other retains one missing member. TM observations are unchanged.
  Warm cohort cost is comparable. Unknown parity policies remain unassessed; no
  universal histogram or optimizer-quality gate is declared passed.
- Preserve float32-safe edges, boundary gaps and overflow handling. Regression
  coverage includes bin boundaries, neighbouring half-hour gaps, real EMA cohorts,
  simultaneous entry/fill histograms and retained/shared GPU replay surfaces.
- This slice requires completed current-head automatic/author review and successful
  required CI before development integration. Larger suites, cold-native/orchestrator
  measurements, HSL-tail materiality and legacy retirement remain acceptance work.
- Validation passes 458 distinct affected Python/CUDA cases: 422 reducer/service/
  cohort cases and 36 retained/shared kernel cases. Rust passes 330 tests (one ignored)
  and default-feature checks; the rebuilt extension is source-verified. Six documentation
  cases pass. Two broad HSL fixtures initially failed because manual service objects
  omitted the unchanged `hsl_signal_mode` contract. Correct their initialization and
  rerun both successfully without relaxing their trading, controller or metric assertions.


### 2026-10-06 — Refresh development with shared session artifacts

- PR #1914 integrated after clear exact-head automatic/author review and successful
  required CI. Integrate master `cf3809c3d3b837861b7e8fdf5903c4fce8ceaadf`,
  including PR #1913's readable session names, setup manifests, starting-config
  snapshots and explicit suite artifact paths. PR #1905's bounded allowance and
  scaled coin-HSL policy remain included through the earlier master integration.
- Resolve only the overlapping changelog entries; retain both branches' entries.
  Preserve native CPU preparation, GPU-only bootstrap/resume, checkpoint-owned
  anchors, coupled scenario spans, screening and prompt result/Pareto persistence.
- Extend native CLI coverage to check session manifests, selected-seed counts and
  unchanged output directories/manifests on resume. Add four generated-seed cases
  spanning standalone/suite and ordinary/interrupted runs with scaled HSL enabled;
  forbid CPU simulation and CPU worker-pool creation throughout.
- Validation passes 530 focused Python/CUDA checks, including 32 native CLI lifecycle
  cases. Rust passes 330 tests (one ignored) and default-feature checks; the rebuilt
  extension is source-verified. Six documentation checks pass. No additional
  simulator or numerical-policy change is introduced by this refresh.
- Require completed current-head automatic/author review and all required CI before
  development integration. Master receives no optimizer redesign changes.


### 2026-10-06 — Explicit multiobjective evidence and sustained native traffic

- PR #1915 integrated master session artifacts after completed clear current-head
  automatic/author review and all required CI. Development preserves the reviewed
  tree and master ancestry; pending objective-vector work was retained separately.
- Add explicit min/max objective vectors to offline cohort comparison, automatically
  request their metrics and report multi-dimensional fronts, pair-order changes and
  per-axis CPU regret. Preserve default ADG/drawdown diagnostics, independent limits
  and unknown-tolerance handling. No search or GPU execution policy changes.
- Reproduce the fill-gap third-objective analysis through the public tool rather than
  separate post-processing. Raw fronts and axis extremes are diagnostic observations,
  not constraint penalties, optimizer survival or whole-search quality certification.
- Refresh sustained GPU-only completed-work evidence after fill-gap refinement with
  unchanged tuning windows: 12,288 EMA and 6,144 TM requests exactly match direct GPU
  references. Completed width-128 evidence roughly doubles width-64 throughput;
  no larger-width optimum is inferred. Shutdown releases all Torch allocations.
- Measure service caller CPU with `time.thread_time()`: 0.606/0.331 CPU seconds over
  87.756/165.652 elapsed seconds. Whole-process CPU includes worker/driver/reducers;
  repeated resolved requests do not measure full optimizer overhead. Keep large-suite,
  cold compilation, CPU evolution/storage and metric-materiality acceptance open.
- This slice requires completed current-head automatic/author review and successful
  required CI before development integration.
- Validation passes all 44 cohort-tool cases on the NVIDIA runtime, including both
  real CUDA requested-vector reports and the existing raw-replay batch-shape check.
  Four fresh eleven-metric/three-objective cohorts preserve every input/parameter
  identity, CPU/GPU metric comparison and limit outcome; their fronts reproduce the
  independent histogram analysis. Six documentation cases pass. Rust source/artifact
  is unchanged and verified; final source-byte verification precedes publication.


### 2026-10-07 — Integrated objective diagnostics and identified HSL ordering

- PR #1916 integrated at `732fb4791d47cdf8e6810d0f3fa458d5b3c3b3f5` after
  completed clear automatic/author review of head
  `46cdb50c2bde2c8161d0603f8f6a1eccdb593ec7` and all three required CI jobs.
  The reviewed tree and master ancestry are verified. Master receives no redesign.
- A public synthetic full optimizer workload uses TM, eight coins, both sides,
  10,080 minute bars, population 64, 256 requested iterations and seed 12. Three
  scenarios use all coins, alternating four coins and a two-coin subset; base-only
  screening promotes half of later generations. Two starting configs are evaluated
  on GPU. CPU simulation APIs, evaluators and worker pools are forbidden throughout.
- Fresh versus populated CUDA compiler-cache runs complete 160 full records and 192
  screenings with eight Pareto members. Canonical candidate bot/metric records match
  exactly after order-independent comparison. Elapsed time is 186.819/63.385 seconds;
  caller CPU is 24.489/25.937 seconds. Candidate preparation accounts for 18.024/19.037
  CPU seconds; scoring and evolutionary updates are much cheaper. Phase timings
  overlap and must not be added. This reveals a useful preparation target rather
  than proving an optimal result cadence or a general performance bound.
- Result cadence limits adapt from one to 77/86, but actual completion groups in this
  workload remain singletons. Preserve that distinction when assessing grouping.
  Both runs execute 584 backtests; widths remain underfilled and no completed tuning
  window is accepted. Earlier sustained service-only tuning evidence stays separate.
- Peak process RSS is about 2.15/1.08 GB, peak Torch allocation about 8.88/8.29 MB,
  and retained session files about 1.26 MB. Sampled whole-device use peaks near 2.03 GB
  and includes unrelated device allocations; it is not service-owned VRAM. Final
  Torch allocations are zero. Larger/longer workloads and EMA full-search measurements
  remain open; these observations do not close general resource acceptance.
- Trace the remaining EMA coin-HSL time-in-red discrepancy using two coins, 3,000
  bars, seed 43, threshold 0.002, EMA span 2.5 and five-minute cooldown. CPU records
  three stop/restart episodes and 15 red minutes; GPU records the same three episodes
  but 18 red bar samples. Output-only diagnostics preserve unrelated raw outputs
  and locate GPU terminal flat timestamps one candle after CPU's corresponding fills.
- CPU refreshes HSL before constructing the next orders. Shared GPU order generation
  currently precedes HSL refresh, delaying the next panic order. A bounded exported-
  shader ordering control restores the CPU fill count and brings ADG/drawdown inside
  existing strict tolerances; all seven requested disabled-HSL metrics are unchanged.
  The remaining red-time difference in that control is the sampled denominator
  (2,938 observations versus 2,937 elapsed intervals), not the removed extra panic bar.
- Do not absorb the delay into a tolerance or infer a production fix from the control.
  Correct phase ordering across native paths with trigger/recovery/cooldown, aggregate
  scopes, delisting/liquidation, specialization and interruption coverage. The control
  deliberately does not certify those combinations. Keep legacy-retirement gates open.


### 2026-10-07 — Correct HSL phase ownership and terminal boundaries

- Fetch and verify master `cf3809c3d3b837861b7e8fdf5903c4fce8ceaadf` remains
  included in development `732fb4791d47cdf8e6810d0f3fa458d5b3c3b3f5` through
  PRs #1911 and #1915. No further master merge is needed; preserve pending work.
- Advance scoped HSL after fills/indicators/slot budgets and before selection,
  one-way arbitration, unstuck and next orders in both shared EMA/TM topologies.
  Invalid controllers return the existing unavailable-HSL error before consumers;
  do not represent malformed HSL as a fabricated zero balance or liquidation.
- Separate observational reporting into a shared helper sampled after forced
  delisting and fee-inclusive final valuation. It never advances the controller.
  Preserve optional EMA-tail field/argument ablation and existing aggregate
  strategy-equity sampling eligibility. No CPU simulation joins native optimization.
- A same-bar forced terminal fill can precede the latest provisional mark. Retain
  three bounded window cursors, retract only that latest observation, and recompute
  the terminal signal using its factual timestamp. Reuse the existing lookback + 2
  storage and bounded reduction-block repair; no extra history allocation or journal.
  Ordinary out-of-order observations remain invalid. Core tests cover dense/sparse
  timestamps; production interval packing continues to scale policy into bar units.
- Extend independent Rust-controller comparison to 112 terminal cases covering
  first exposure, expiry, ring reuse and multiple reduction blocks, both restart
  policies and dense/sparse observations. Four native EMA one/two-coin long/fused
  cases restore CPU fills and pass existing ADG/drawdown/lifecycle tolerances.
  Four reporting probes verify final equity, factual terminal time and no re-observation.
- Corrected, source-verified CUDA validation passes all 166 controller/ordering/
  reporting cases. Rust passes 330 tests (one ignored) and default-feature checks;
  six documentation tests pass. An earlier phase-only broader run passes 102 cases
  with 13 hardware-specific skips; that result does not certify the final helper.
- Continue real forced-delisting/liquidation regressions, exact disabled-HSL controls,
  all-scope/strategy parity, native CLI/resume and final broader validation. Remaining
  sampled red-time denominator differences are still separate acceptance work.
  Keep this slice local until validated; require completed current-head automatic/
  author review and all required CI before integration into development.

- Final-helper validation adds 23 real forced-terminal/delist/liquidation boundary
  cases and 374 orchestration, checkpoint and parity-tool cases, including actual
  native CUDA comparisons. All pass. Original-source disabled-HSL controls compare
  every raw output exactly across both strategies and long/short/fused topologies.
- The eighteen-case ordinary scope matrix intentionally leaves red-time percentages
  unassessed. All six EMA long/fused cases pass the five assessed metrics; three
  short-only cases expose a material baseline ranking shortcut. A two-coin,
  3,000-bar, seed-43 short fixture reproduces it in the original kernel and with
  HSL disabled. Removing only the deferred-ranking return in an exported-shader
  control passes existing strict metrics; the one-coin control also passes.
  Resolve this separately before cutover; no tolerance can justify that shortcut.
- A second eighteen-case matrix adds deterministic 30% price shocks and uses 0.002
  HSL thresholds. All nine TM cases agree on stop/restart counts and drawdown;
  three long cases pass every strict metric. Short/fused TM retains small fill/ADG
  trajectory gaps. EMA long/fused lifecycle counts agree; short-only selection
  remains material. Nine checked-in native cases cover the original coin regressions
  and practical-threshold TM/all-scope and fused EMA observations with real stops.
- The near-float32-scale TM threshold of 0.000001 in the ordinary stress matrix
  retains large lifecycle differences for short/fused runs. Their cause and
  materiality remain unaccepted; the stronger-signal matrix is additional evidence,
  not a replacement for recording those failures. Neither matrix certifies full
  cutover; the completed wider and native CLI checks below stay separate.
- Enable the previously Apple-only episode-boundary probes through the shared
  runtime on CUDA, retaining Metal support and explicit probe diagnostics. Run the
  two expanded test files from an isolated temporary area while the existing
  wider/CLI job keeps immutable source. Dedicated Metal capacity/layout cases
  remain outside first-platform hardware acceptance. All 84 shared episode probes
  and the 13 expanded native/reporting cases pass on CUDA.

- Final wider HSL runner/service/replay validation passes 110 cases, with 97 skips:
  84 formerly Apple-only episode probes now pass in the separate expanded CUDA run;
  the other 13 require dedicated Metal layouts. All 32 native optimizer CLI cases
  pass, including CPU-forbidden screening, scaled HSL, interruption and resume.
- Two additional real-CUDA service tests inject HSL rejection at the producer's
  controller boundary for EMA and TM. Both futures propagate the existing
  unavailable-HSL exception; CPU simulation APIs are forbidden. Invalid HSL cannot
  silently become a usable liquidation result.
- The final Rust/kernel bytes match the tested extension source stamp. Only test
  portability, rejection coverage and the evidence ledger changed after the broad
  runs. Require a clean commit, completed exact-head automatic/author review and
  all three required CI jobs before integrating this slice into development.


### 2026-10-07 — Remove deferred EMA selection

- PR #1917 contains the validated HSL-ordering/terminal-reporting slice. Its
  current-head automatic and author reviews complete without findings. All three
  required CI jobs pass; it merges into development at
  `5f4b8284e8c229d503df0b6008c314784b46b75e`. Keep the separate selection fix local.
- Baseline CUDA ranking probes reproduce stale flat selection on both EMA sides
  without a fill or eligibility transition; corresponding TM probes pass. The
  earlier native control reproduces the material short-only metric gap with HSL
  enabled and disabled. Do not reinterpret that shortcut as numerical tolerance.
- Refresh selection from current indicators/eligibility every bar, retaining held
  positions. Remove three cached eligibility masks, initialization and previous-slot
  fields plus their invalidation logic. Hysteresis incumbency follows outstanding
  entry orders, as CPU and TM already do; previous selection alone has no authority.
- Extend current-ranking and incumbent-order probes to EMA and add twelve real
  native CPU/GPU comparisons across both directional sides, fused execution and
  disabled/coin/pside/unified HSL. Add four CPU-forbidden EMA optimizer CLI cases
  using starting seeds, prompt persistence, suite screening, interruption and resume.
  All twelve native parity cases pass against the source-verified CUDA build,
  with the single baseline numerical bound documented below. All 203 focused CUDA
  cases pass: twelve native comparisons and 191 replay/selection cases. All 36 real
  optimizer CLI cases pass, including
  four new EMA cases, with CPU simulations/pools forbidden and prompt results/
  Pareto, screening, interruption and resume verified. Rust passes 330 tests
  (one ignored) and default-feature checks for the current selection diff.

- Original-source and corrected-kernel controls produce identical assessed metrics
  for the disabled-HSL long fixture. CPU/GPU ADG is 0.0053993594/0.0053984714:
  absolute difference 0.000000888, relative about 0.0164%. Fill frequency is identical,
  neither run has HSL events, and worst drawdown differs by 0.000000177. Accept
  ADG absolute tolerance 0.000001 for this checked-in fixture only; preserve all
  other gates and the standalone parity tool's default measurement policy. This
  bounded baseline discrepancy does not explain the material short-ranking defect.
- Existing public GPU policy requires minute candles when HSL is enabled. Coarser
  disabled-HSL replay remains in scope; extending enabled HSL to coarser candles
  is not an initial cutover requirement. Sparse controller probes do not certify
  that unsupported simulation combination.

- The broader replay run exposes four outdated EMA coin-HSL isolation assertions:
  they expected both symbols flat after only one symbol stops. Four offline CPU
  controls, with packed active parameters verified, retain the healthy coin on
  both sides and for either stopped coin, matching corrected GPU replay. Require
  one healthy open position for both strategies and preserve every stop-count,
  panic-accounting and packed-slot invariance check. All fourteen isolated episode/
  directional-smoke cases pass after correcting the six outdated assertions. The
  final 203-case run passes with the same verified Rust/kernel bytes. Six
  documentation tests pass; require clean publication, completed exact-head automatic/
  author review and all three CI jobs before integrating this slice into development.

### 2026-10-07 — Align native HSL elapsed reporting

- CPU HSL reporting accumulates elapsed time using the preceding observation's RED
  state. Shared GPU reporting counted current bar samples, adding an initial
  denominator unit and potentially treating a terminal RED state as elapsed time.
- Add one shared observation clock and retain it across TM temporal dispatches.
  Reuse the two existing scalar slots for elapsed observed/RED steps; their packed
  labels remain compatible with retained directional single-coin reporting. No
  simulation/order/controller behavior or history transfer changes. HSL-disabled
  code omits observation updates and chunk-owned clock state.
- Add explicit duration/boundary probes and strict native EMA time-in-red coverage.
  Publication requires current-source CUDA, disabled controls, temporal chunking,
  lifecycle and CPU-forbidden CLI evidence; integration additionally requires
  completed current-head automatic/author review and all three CI jobs.

- Before this reporting correction, repeat the full three-scenario EMA search on
  the integrated selection source: eight coins, 10,080 minute bars, disabled HSL,
  data seed 43, optimizer seed 12, population 64, 256 iterations, two starting configs and
  base-screening survival fraction 0.5. Both cold/warm runs complete 160 full
  candidate records and 192 screens, retaining 51 Pareto members. Decode the
  incremental result stream with the canonical reader before comparison: complete
  candidate/metric records match exactly across runs. Raw message objects are
  overlays and must not be compared as complete configurations.
- Cold/warm elapsed time is 156.676/54.147 seconds; caller CPU is 23.034/22.970
  seconds, including 16.484/16.362 seconds of candidate preparation. Scoring and
  evolution remain small. Default GPU tuning receives no eligible windows in
  this underfilled search; CPU cadence limits reach 76/89 but actual groups stay
  singletons. Neither establishes an optimal batch width or consumption cadence.
- Peak process RSS is 1.91/1.08 GB, peak Torch allocation 8.44/8.53 MB, final Torch
  allocation zero, and session storage 2.68 MB. Sampled whole-device use includes
  unrelated allocations. Measurements exclude compilation/packing caches from
  session storage; other seeds and representative comparisons remain open.
- All nine enabled side/scope EMA cases fail strict time-in-red parity on the
  original sample-based reporting source; all three disabled controls pass.
  Preserve this original-source regression independently of the corrected-source
  runs. Rust passes 330 tests (one ignored), default-feature checks and six
  documentation tests for the local elapsed-observation implementation.

- Initial current-source CUDA validation passes 43 cases. Extend coverage to six
  short/shared TM scope cases: original and corrected shader sources return identical
  assessed ADG, drawdown, fills, trigger and restart metrics; corrected time-in-red
  matches CPU exactly in all six. The original denominator differs by one initial
  sample. Record bounded pre-existing trajectory discrepancies only for these shock
  fixtures: at most 0.00004505 ADG and 0.0978% fill rate. Fixture gates accept
  0.00005 absolute ADG and 0.1% relative fill rate; default parity policy, drawdown,
  trigger/restart and strict RED-time gates stay unchanged. Broader trajectory
  materiality remains separate acceptance work.

- Broader current-source CUDA controls pass 142 cases, with thirteen Apple-only
  cases skipped. They preserve every returned output across disabled HSL, general/
  specialized replay, temporal dispatches, repeated buffers, reordered candidates
  and scratch splitting, and retain invalid-input/failure propagation. All 36
  native optimizer CLI cases pass with CPU simulation APIs and worker pools forbidden,
  including seeds, suite screening, coupled/scaled policy, prompt records/Pareto,
  interruption and checkpoint resumption. The final fixture guard restricts the
  six numerical bounds explicitly to the measured two-coin shock geometry.
- Final reporting parity passes 49 cases. A focused repeat of the changed ordering
  file verifies the narrowed fixture guard; production Rust/kernel bytes remain
  unchanged. Six documentation tests, AI-document checks and the generated registry
  pass. Verify final tracked bytes and the loaded Rust artifact before publication;
  required automatic/author review and three CI jobs remain development merge gates.
  Full metric/resource/performance acceptance and legacy retirement remain open.


### 2026-10-07 — Preserve elapsed HSL reporting through liquidation

- PR #1919 automatic review identified a valid terminal-accounting gap: a liquidation
  fails the ordinary HSL reporting admission guard, dropping the last elapsed interval.
  Keep the finding unresolved until the corrected head passes device checks and review.
- Split clock advancement from tier observation. Terminal accounting advances using
  the preceding RED state without observing a new controller signal. Capture whether
  CPU would stop at the post-fill boundary before next-order/forced-delisting work:
  terminal fills use that earlier timestamp; ordinary bar-close liquidation uses the
  close timestamp. Apply the same reporting-only rule to directional/fused EMA and TM.
  Disabled-HSL variants omit these updates; trading and controller decisions are unchanged.
- Eight real CPU/native regressions pass across both strategies, long/fused operation
  and limit/market panic policies. The seven-bar fixture establishes an entry at bar 3,
  RED at bar 4, then gaps from 99.5 to 20 at bar 5. Limit panic liquidation retains
  one RED interval out of three observed intervals; terminal market panic fills retain
  zero additional elapsed intervals. All four limit cases fail on the original source.
  Two additional device probes check terminal advancement, duplicate timestamps and
  preservation of the preceding RED state. The full current-source reporting slice
  passes 69 cases, including cooldown controls and twelve native EMA scope comparisons.
  All 169 disabled/terminal/temporal controls pass after the baseline-backed assertion
  correction below. All 36 native optimizer CLI lifecycle cases pass with CPU
  simulation APIs and CPU worker-pool construction forbidden, including bootstrap,
  scenario screening, prompt persistence, interruption and checkpoint resumption.
- Rust validation passes 330 tests with one ignored test and default-feature test
  compilation. Rebuild and verify the actual extension before CUDA validation. Complete
  current-head author/automatic review and all three CI jobs remain merge requirements.

### 2026-10-07 — Measure controller-enabled scenario search

- These measurements use the elapsed-reporting implementation before the liquidation
  clock correction. Preserve that scope when comparing subsequent runs.
- Repeat the earlier eight-coin, 10,080-minute, three-scenario EMA search with coin HSL
  enabled, data seed 43 and optimizer seed 12. Apply deterministic 30% downward/upward
  shocks to coins 0/1 at bars 4,320/7,200. Request RED-time through a diagnostic limit.
  Canonical adaptive bounds adjust the nominal seed HSL values: this is an evolving
  policy workload, not a fixed 0.002-threshold policy benchmark.
- Cold/warm runs complete 160 full records and 192 screens, retaining 26 Pareto members.
  Canonically decoded complete candidate/metric records and Pareto membership agree
  exactly. Canonical effective input identities also agree; raw configuration hashes
  differ only in transformation-log bookkeeping. All 160 records contain RED-time,
  nine report a nonzero value, and the maximum is 0.0958371. Unrequested trigger/restart
  metrics cannot establish event counts.
- Elapsed time is 259.505/156.501 seconds; caller CPU is 23.553/26.095 seconds, including
  16.748/18.749 seconds of candidate preparation. Whole-process CPU is 147.860/47.134
  seconds. Peak RSS is 1.91/1.08 GB; Torch peak allocation is 39.20 MB and final allocation
  is zero. Session storage is 1.75 MB, excluding compilation/packing caches.
- CPU consumption groups remain singletons despite cadence limits of 76/84. Actual GPU
  dispatches are distinct: both runs issue 29 batches of 1–62 requests. The nominal
  automatic width is 64 and tuning receives no eligible windows. Keep default tuning
  and repeated-seed comparison acceptance open. A fixed-policy followup must fix the
  relevant search bounds explicitly; compare smaller initial widths on this workload
  before choosing a different default.


### 2026-10-07 — Prepare complete metric-surface comparison

- Verify CPU reference availability for all 157 canonical GPU-supported metric names,
  across both strategies and long/short/fused configurations. Use two coins, seed 43,
  enabled coin HSL and unstuck, threshold 0.002, span 2.5 and cooldown 5, with shocks
  at bars 1,440/1,800. This reference work is outside optimization.
- The 28,800-minute fixture returns finite values for every requested metric in all
  six cases. A 2,880-minute fixture exposes null weighted exponential-fit values for
  USD/BTC through the CPU serialization boundary: short suffixes lack fit samples.
  Use the longer fixture to assess actual metric coverage; do not treat short-fixture
  missing/sentinel values as numerical matches or substitute fabricated finite values.
- Full native GPU comparison completes after the current-source reporting, replay
  controls and CPU-forbidden optimizer CLI checks pass: all 942 metric pairs are
  present and finite. Only the four existing default tolerance policies are assessed;
  the other 153 requested metrics per case remain unassessed. Presence is not parity.
- The audit identifies two concrete HSL reporting defects: halt-to-restart loss stays
  zero despite nonzero panic loss in all six cases; post-restart retrigger percentage
  stays zero despite nonzero CPU results in the three EMA cases. Their kernel scalar
  fields have no updating producer. Resolve these before acceptance; do not widen
  tolerances around a missing calculation.
- Weighted strategy-equity ratios, trigger-drawdown means and disabled-side strict
  recovery also require case-specific assessment. Some default ADG/fill-rate gates
  fail on these long shock fixtures. Keep these results distinct from the narrower
  liquidation-clock acceptance and evaluate trajectory/materiality before choosing
  numerical policies.


### 2026-10-07 — Reconcile retained single-coin reporting controls

- The expanded control slice identifies four stale retained single-coin delisting
  assertions. All four fail identically on development and corrected exported shaders,
  with every returned tensor/scalar exactly equal. Under restart policy `never`, the
  RED terminal close remains halted; reporting includes all 1,400 subsequent samples,
  while positions are flat and each side has exactly one trigger. Replace the old
  panic-tier-based zero expectation with the factual reporting duration. Keep exact
  fill, loss, position, balance and trigger assertions; no simulation changes. The
  complete 169-case affected control slice passes with the unchanged compiled runtime.
- CPU compatibility coverage passes 128 backend/search/analysis/artifact/plot tests;
  the browser-logic test additionally passes where its JavaScript runtime is available.
  Backend/search tests use identified fake evaluators; the separate six-case real CPU
  reference audit supplies actual simulation evidence. These are distinct claims.


### 2026-10-07 — Reuse panic loss for normalized HSL reporting

- CPU `Report::metrics` defines halt-to-restart equity loss as total panic-close
  loss divided by starting balance, including unfinished halts. Decode the existing
  GPU panic-loss sum with that same formula instead of reading the dormant independent
  halt-loss scalar. No kernel, controller, transfer or trading change is necessary.
- Six two-coin, 3,000-minute seed-43 shock fixtures cover both strategies and
  long/short/fused replay. Every fixture fails on the original reducer with positive
  CPU loss and zero GPU loss; all six pass after the correction with a local absolute
  bound of 1e-5 plus relative 0.1%. General tool policy is unchanged. Synthetic
  reduction coverage deliberately supplies different panic-loss and legacy halt-loss
  scalars, verifying the actual Rust numerator and the zero-loss case.
- Post-restart retrigger reporting remains a separate open defect. Its replacement
  must account for incomplete panic exits and current GREEN recovery, rather than
  inferring every lifecycle event from completed terminal closes.


### 2026-10-07 — Measure fixed policy with a sixteen-request dispatch ceiling

- Hold both sides' coin-HSL threshold/span/cooldown at 0.002/2.5/5 through explicit
  search bounds; retain the eight-coin, 10,080-minute, seed-43 shocks, optimizer seed
  12, three scenarios, population 64 and 256 requested iterations. Forbid CPU
  simulation APIs and worker pools during both measurements.
- Cold/warm runs complete 160 full results and 192 screens with 15 Pareto members.
  Decoded candidate/metric records and Pareto membership agree exactly. All 160
  records have nonzero RED time. Unrequested trigger/restart metrics do not prove
  event counts. Cold/warm elapsed times are 330.286/230.581 seconds, caller CPU
  23.566/27.060 seconds and whole-process CPU 162.497/64.175 seconds.
- Both issue 52 actual GPU batches of 1–16 requests. An explicit batch size disables
  adaptive execution tuning, so zero tuning windows here are expected. This supplies
  a fixed-width comparison point, not adaptive-default acceptance or evidence that
  width 16 is optimal. Compare the same fixed-policy recipe at other widths before
  attributing timing differences to width; earlier adaptive-policy timings have
  different effective HSL policies.
- Peak RSS is 1.91/1.08 GB, Torch peak allocation 17.87/17.34 MB and final allocation
  zero in both runs. Stored session data is approximately 1.33 MB, excluding compile
  and packing caches. These measurements precede the lifecycle-reporting replacement.

### 2026-10-07 — Separate observational HSL lifecycle accounting

- Source inspection identifies a broader gap behind the dead retrigger scalar:
  terminal-only trigger counting misses open panic and GREEN permission recovery.
  Introduce reporting-only RED entry/restart transitions using existing state,
  separately record flat completion, and include unfinished RED/exit durations.
  Consume each scope's pending restart only once when a subsequent RED is observed.
  Renewed exposure ends the preceding terminal cooldown before a new RED is counted.
- Keep current controller action, signal, orders and all permission logic unchanged.
  Preserve feature guards, replay-state layout and packed output cardinality. Retained
  directional decoders must follow the same censored-duration reporting semantics.
- Five CPU report reference cases cover completed cooldown/retrigger, GREEN recovery,
  zero cooldown, renewed exposure and an unfinished panic. The CPU reference passes;
  all ten corresponding original-source GPU probes fail. The source-verified replacement
  passes all 20 lifecycle probes/scoped comparisons, 65 reporting cases, 169 replay
  controls and 36 CPU-forbidden optimizer lifecycle cases. Rust passes 331 tests
  with one ignored, and default-feature test compilation passes.
- The six 28,800-minute comparisons produce all 942 finite metric pairs. All seven
  assessed HSL metrics match their fixture-local absolute 1e-5 plus relative 0.1%
  bounds, including nonzero EMA retriggers and first-observed TM trigger drawdown.
  Every non-HSL GPU metric is exactly unchanged from the preceding audit. Existing
  ADG/fill-rate gate failures remain; 146 metrics per case still have no assessed
  policy. This accepts the reporting correction, not the entire metric surface.
- A separate six-case, 3,000-minute diagnostic with HSL disabled confirms ordinary
  side strategy-equity summaries are omitted: active-side raw drawdown and strict
  recovery return zero despite nonzero CPU values. Inactive constant curves also
  lose their full strict-recovery horizon. Separate ordinary equity sampling from
  HSL eligibility in the next change; retain protection ablation and CPU reference
  definitions. Weighted-ratio and long-trajectory assessments remain separate.

### 2026-10-07 — Censored HSL reporting endpoints

- Automatic review of PR #1921 identified a real endpoint defect: unfinished RED
  snapshots subtracted the bar-open equity index from a bar-close episode start.
  Derive the final reporting coordinate from the existing elapsed-observation clock.
  Normal terminal marks reach bar close; liquidation during a fill retains its fill
  timestamp. Apply that convention to shared and retained directional outputs without
  adding replay state or changing controller decisions.
- Twenty-four controlled native comparisons cover both strategies, long/shared sides,
  coin/pside/unified scopes and open limit panic versus flattened market cooldown.
  Actual CPU fill traces confirm the intended terminal exposure. All original-source
  cases report zero minutes against the CPU's one minute. The corrected source passes
  all 24; eight actual fill/mark liquidation comparisons also pass. The seven-row
  fixture's final row is lookahead, so its final simulated close is row five.
- The corrected source additionally passes 77 reporting/selection/loss checks and
  331 Rust tests (one ignored), with default-feature compilation and rebuilt-extension
  verification. The combined replay run passes 168 controls; one service request exceeds
  its 60-second observation deadline without a metric verdict. The unchanged-source
  isolated check passes in 2.83 seconds, including CPU/GPU liquidation identity.
  All 36 CPU-forbidden optimizer checks pass, covering suites, screening, prompt
  persistence, interruption and resume. No timeout is treated as terminal work or
  as successful simulation output. Six documentation checks also pass.
- A separate 3,000-minute seed-43 shock experiment with a 10,000-minute cooldown
  and one-day retained history still exposes closed-episode/history-expiry duration
  discrepancies of one to 55 minutes. It does not isolate the endpoint defect and
  is not accepted by the short controlled regression. Preserve it as a distinct
  trajectory/reporting gap for subsequent diagnosis; general parity policy is unchanged.


### 2026-10-07 — Review follow-up: scope and accounting boundaries

- The next automatic review found three reporting defects. On the published source,
  both shared shader round-trip probes miss the new trigger/restart when renewed
  exposure opens and flattens during cooldown. The six-case Rust reporting reference
  passes; align GPU reporting with the controller's exposed-or-terminal boundary.
- Two native unified comparisons preserve total trigger/restart counts but incorrectly
  attribute their restart rate to long. Transport each candidate's actual packed scope
  into metric reduction, keep portfolio totals, and exclude unified events from
  directional counters. Scope metadata stays inside the GPU execution service.
- Two retained single-coin forced-delisting comparisons report one minute against
  Rust's two; the two corresponding native shared comparisons already match. Record
  the accounting endpoint before forced delisting can be mistaken for an ordinary
  liquidation fill. Carry one explicit endpoint scalar through TM temporal replay;
  do not change order construction or controller decisions.
- The corrected source passes 331 Rust tests (one ignored), default-feature
  compilation and rebuilt-extension identity checks; 64 selected reducer/service
  cases, 58 lifecycle/endpoint/liquidation cases, 57 reporting/selection/loss cases,
  181 replay/ablation controls and all 36 CPU-forbidden optimizer lifecycle cases.
  TM temporal replay includes the new endpoint scalar. Six documentation checks pass.
- Reusing the eight ordinary fill/mark liquidation fixtures with the retained engine
  and duration-only requests yields four matches and four one-minute discrepancies.
  Four long-side comparisons against the exact published shaders reproduce identical
  metrics and positions: retained EMA makes an additional entry and retained TM misses
  the CPU's market panic close. These are existing simulation differences, not caused
  by the endpoint correction; do not change reporting to conceal different trades.
  All eight native counterparts pass. Keep the retained-engine limitation distinct
  from native acceptance and from the corrected forced-delisting endpoint.
- The PR remains unmerged until the corrected head passes automatic review and all
  required CI. Side-equity, weighted reductions and longer history-expiry trajectories
  remain independent acceptance work; general parity policies are unchanged.


### 2026-10-07 — Align shared EMA entry allocation with Rust

- Source-isolated comparisons of the accepted shared shaders and the pending
  side-equity reporting correction preserve identical portfolio drawdown and fill
  rates on three two-coin, 3,000-bar seed-43 shock fixtures. Their original
  CPU/GPU trading differences predate the reporting correction.
- A bounded trace identifies a semantic mismatch: shared GPU EMA clips each
  position at its nominal coin allocation; Rust EMA uses that allocation for clip
  sizing and inventory shift, with portfolio TWEL admission applied separately.
  At step 1600 the long-only CPU fills .072 while GPU fills .017. Removing only
  the extra coin cap restores all 1,237 fill counts and per-bar sizes within half
  a quantity step; drawdown absolute error falls from .00144623 to 7.88e-8.
- Preserve global TWEL admission, optional exposure repair, cooldown, one-way
  initial blocking and scoped HSL. The compiled allocation regression checks both
  sides crossing the nominal coin limit, portfolio clipping/rejection, an explicitly
  disabled portfolio gate and HSL entry blocking. It fails on the accepted shader.
- The CPU ideal-orders reference retains market long entries of 1.002 and .997
  units under a .2 portfolio ceiling; update the old GPU expectation from 1.998
  to 1.999 total. Short entries remain .998 each. This changes the expected strategy
  result rather than loosening its comparison tolerance.
- In the corrected shared long/short shock replay, a .004696 balance difference
  around 1010.69 crosses a nearest-step boundary at step 1453: 49.5000197 CPU
  quantity steps versus 49.4997897 GPU. A .050/.049 clip difference later changes
  two of 2,395 fills. Drawdown differs by 2.23e-7 and daily growth by 7.52e-6.
  Keep a fixture-local 0.1% fill-rate and 1e-5 absolute growth bound; long/short-only
  fill rates and all drawdown checks remain strict. The standalone parity tool's
  general policies remain unchanged; this is not broad trajectory certification.
- Side-equity sampling and weighted suffix reductions remain independent work.
  Keep the legacy optimizer until the overall acceptance gates are met.
- Validation of this isolated correction passes 331 Rust tests (one ignored),
  default-feature compilation and rebuilt-extension identity checks; six allocation
  and market-entry regressions, 13 HSL ordering cases, 12 selected replay controls,
  four native optimizer CLI/suite/interruption/resume cases and six documentation
  tests. The final replay expectations are checked against actual Rust ideal orders.
  Automatic review and required CI must clear the current head before merging.

### 2026-10-07 — Independent side strategy-equity accounting

- PR #1922 integrated into development after completed clear exact-head automatic
  and author review and all three required CI jobs. Stack the side-equity correction
  on the accepted EMA allocation behavior; do not restore the extra entry cap.
- Shared EMA and trailing-martingale replay sample ordinary raw strategy equity
  on the factual account-equity clock, independently of HSL signal eligibility.
  Unified protection retains separate long and short net cashflows and UPNL for
  ordinary side performance. Keep HSL EMA telemetry scoped to its actual signals.
- An inactive side is a constant curve: strict new-peak recovery includes the
  complete observed horizon. Requested-metric specialization still removes
  unneeded statistics; no history is transferred to the optimizer.
- For fewer than 200 observed days, the worst floor(1%) tail contains one daily
  maximum. Reuse the already requested raw maximum instead of averaging its
  histogram bin. Longer tails retain the existing approximation pending acceptance.
- Initial source-verified CUDA validation of the public seed-43 two-coin,
  3,000-bar shock matrix passes 22 of 24 side-metric scenarios. All 48 recovery
  comparisons match exactly across both strategies, trading sides and disabled,
  coin, pside and unified HSL. Disabled/unified shared trailing-martingale long
  drawdown residuals remain visible at 5.35e-5 and 5.94e-6 respectively. Do not
  change general parity policies or treat this partial matrix as acceptance.
- The initial strict run passes 55 checks and fails only those two side drawdown
  cases; all 29 HSL ordering cases and four compiled two-/199-day tail probes pass.
  Keep the numerical assessment below explicit rather than changing general policies.
- Keep recursive grid sizing as separate semantic work: Rust refreshes initial
  sizing from its simulated order-book price; the GPU helper currently retains the
  original generation price. Correcting that policy does not establish arbitrary
  trading-path identity. Do not change metric accounting to conceal distinct trades.
- The actual Rust floor API reproduces .017 from .060-.042; the existing
  GPU float32 rounding tolerance produces .018. Accept a fixture-local absolute
  drawdown bound of 1e-4 (one basis point) for the two affected long raw/tail values.
  Their observed errors are 5.35e-5 and 5.94e-6. Keep all recovery, short drawdown
  and the other 22 scenario comparisons strict, and leave general parity policies
  unchanged. This bounds practical risk in these fixtures without declaring the
  separate recursive sizing mismatch corrected or certifying arbitrary trajectories.
- Final validation passes 36 side-metric/tail/liquidation regressions, with 32
  complete comparison reports, including all eight extended actual fill/mark
  liquidation cases. Side strategy metrics retain the raw terminal mark while
  account equity is clamped. The runtime and all 922 source-manifest files are
  verified; production source is unchanged by the fixture-local policy.
- Rust passes 331 tests (one ignored), default-feature compilation passes, 183
  replay/ablation/isolation controls pass and all 36 CPU-forbidden optimizer
  lifecycle cases pass. Six documentation checks pass. Update the canonical HSL
  contract to separate ordinary equity performance from protection and distinguish
  retained CPU-validated screening from experimental native GPU-only optimization.
- Require completed current-head author/automatic review and all required CI before
  development integration. Weighted/raw portfolio reduction and broader performance,
  CPU preservation and legacy retirement acceptance remain open.

### 2026-10-07 — Distinguish portfolio strategy and account reductions

- Extend the eight public native fill/mark liquidation comparisons with portfolio
  raw drawdown, daily worst tail and underwater mean. All 24 raw-risk comparisons
  differ: CPU retains drawdown of approximately 3.157–3.210 while shared GPU
  summaries report the clamped account curve at approximately 0.951. Ordinary
  `drawdown_worst_usd` matches in all eight. Preserve account summary meaning;
  changing its terminal value globally would introduce an account-analysis defect.
- Shared EMA and trailing-martingale kernels retain requested-only raw daily
  maximum drawdown alongside account summaries. The three explicit portfolio
  raw-risk metrics consume this column; ordinary USD risk keeps account semantics.
  The column costs four bytes per observed day per candidate, enters dispatch
  scratch admission and compiles away when unrequested. No per-step history is
  transferred. A missing required summary is an error, not an account fallback.
- Eight actual fill/mark liquidation comparisons pass after this correction,
  with maximum raw-risk absolute error below 2.4e-7. Eight directional ablation
  cases across both strategies and BTC-risk settings preserve every other output
  exactly. Eight trailing-martingale temporal replay cases preserve all outputs
  exactly across chunk boundaries, including raw peak and partial-day state.
  These checks do not certify raw growth, weighted or recovery metrics.
- CPU exports distinguish ordinary account analysis from explicit strategy-equity
  analysis. Ordinary weighted account ratios recompute each suffix's peak. Raw
  strategy weighted Calmar/Sterling instead slice the full-curve drawdown series,
  retaining its peak convention. Both reconstruct suffix daily minima; whole-day
  reuse may include observations before the cutoff. Validate each family against
  its actual Rust producer before sharing a reducer. A reference limited to ordinary
  analysis does not certify explicit strategy metrics or make USD names aliases.
- Prefer requested-only compact raw daily risk alongside existing account outputs.
  Reserve resident histories for metrics that actually require chronology; include
  storage and reduction scratch in dispatch admission and tuning. Keep these
  details inside the execution service and keep compact results at the CPU boundary.
- The 24 ordinary shock cases exercise both strategies, long/short/shared sides
  and disabled, coin, pside and unified HSL. All 72 new raw-risk comparisons pass
  their explicit policies. Keep the existing one-basis-point bound for the shared
  TM disabled-HSL fixture; its corrected raw-risk error is below 4.0e-5. The other
  23 raw-risk cases remain strict; existing side-metric policies are unchanged.
- The new account check exposes a pre-existing 2.985e-6 USD drawdown residual in
  the shared EMA coin-HSL fixture. Accepted shader replay and current raw capture
  on/off return exactly the same account value. Use a fixture-local 3.1e-6 absolute
  bound for that account metric alone (about .031 basis points); preserve strict
  raw-risk checks and add actual on/off account-metric equality. General standalone
  parity policies are unchanged. Final matrix validation passes all 48 cases.
- Final validation passes 26 focused risk/liquidation/ablation checks, eight exact
  temporal checks, 183 broader controls, 36 CPU-forbidden optimizer lifecycle
  cases, four missing-summary/routing checks and the full 48-case shock matrix.
  The matrix contains 49 complete reports and 241 metric pairs. Source-verified
  Rust tests pass 331 cases (one ignored), default-feature compilation and 1,039
  affected Python checks pass, and six documentation checks pass. Verify the final
  publication tree and wait for completed current-head author/automatic review and
  required CI before development integration. Raw growth/recovery/weighted metrics,
  representative resource/performance checks and legacy retirement remain open.

## Weighted equity history foundation (2026-10-07)

- PR #1924 is merged into development after completed current-head automatic and
  author review and all required CI checks. Its merged tree matches the reviewed
  head. Master remains unchanged by this work.
- Keep two factual curves explicit. Raw strategy weighted Calmar/Sterling retain
  the full-curve peak and divide suffix contributions by ten. Ordinary account
  analysis resets peaks inside each suffix and averages its nonempty suffixes.
  No-fill accounts retain Rust defaults, but a suffix without new fills still
  consumes its changing equity. The seven equity-only weighted families therefore
  need actual total fill eligibility, not another per-step fill history.
- Add an unconnected resident-sample reducer and shared public controlled fixtures.
  A Rust test checks both actual producers against the same references for f64
  and f32-quantized inputs; GPU tests use those references on CPU Torch and CUDA.
  Cover short/empty histories, discarded peaks, partial UTC days, 240-day tails,
  negative terminal equity, zero/near-zero starts, sparse/absent fills, independent
  clocks, unused padding and interleaved views. Requested subsets skip unused
  reductions. This is reducer evidence, not simulator or optimizer acceptance.
  Source-verified validation passes 332 Rust tests (one existing ignored),
  default-feature test compilation, 153 Python checks including CPU/CUDA reference
  reductions, and six documentation checks.
- Controlled f64 comparisons agree across 840 CPU-Torch/CUDA metric pairs.
  Ordinary-scale f32 fixtures differ from the original f64 reference by at most
  2.29e-7 relatively. An artificial account near 1e-12 instead amplifies f32 input
  rounding into roughly 99.97% Calmar/Sterling differences. Check the reducer
  against matching quantized Rust inputs without weakening practical simulator
  parity policies; do not claim that f32 is uniformly harmless near singular
  metric denominators.
- Measure the actual reducer on resident interleaved f32 histories, separately
  from trading throughput. With 28,800 observations and batches 1/16/64, all seven
  families take about 30–45 ms per curve; MDG alone takes about 7–9 ms. Maximum
  measured Torch scratch above both resident input curves is about 107.4 MB at
  batch 64. These figures exclude external driver/CuPy allocations and do not
  establish whole-optimizer performance or a final admission bound.
- Next connect separately specialized raw/account capture to the existing service.
  Reduce inside the runner before it combines history sub-batches, so compact
  results accumulate without retaining all candidate histories. Budget capture,
  reduction scratch and daily work together; keep owned calendar dimensions and
  avoid device-derived host shape discovery. Preserve existing recovery capture
  meanings and account/per-exposure aliases. Validate temporal boundaries,
  liquidation, HSL modes, capture on/off behavior and real CPU/GPU replay before
  publication. This foundation alone enables no weighted capture during optimization.

### Weighted capture integration

- Connect separately specialized factual raw/account histories to shared CUDA
  runners and the native service. Request dependencies include weighted account
  ADG/MDG per exposure. Retain account aliases before installing explicit raw
  values. Missing requested compact metrics fail rather than fall back to a daily
  screening approximation. Retained Metal and legacy directional single-coin
  screening keep their existing reducer paths.
- Restore relative f32 clocks to their integer bar grid before adding the absolute
  UTC origin. Capture each factual equity observation, including liquidation, and
  reduce actual suffixes before combining history sub-batches. Preserve independent
  recovery-history semantics and request-owned calendar dimensions.
- Include requested curve storage, sequential reduction scratch, daily work and
  compact results in scratch admission. Raw/account capture separately compiles
  away when absent. A physical dispatch retains only its current weighted sample
  buffer; combined results contain compact metric vectors.
- Eighteen capture ablations across both strategies and all side combinations
  preserve every previous output exactly. Temporal replay, mixed candidate ending
  times, repeated-run clearing and bounded sub-batch controls pass. Eight actual
  CPU/GPU fill/mark liquidations across midnight pass all 120 requested weighted
  and account-exposure pairs with unchanged strict comparison policies.
- Both native service admission tests and all four weighted optimizer interruption,
  persistence and resumption cases pass with CPU simulations forbidden. The suite
  cases retain scenario screening. Affected service, backend, tuning and benchmark
  checks pass 1,012 cases (one existing skip), and six documentation checks pass.
- Across 24 ordinary public shock fixtures, actual Rust producers match reductions
  of original CPU, f32-quantized CPU and captured GPU curves in all 1,008 metric
  comparisons. The largest reducer absolute difference is below 9.1e-11. Native
  compact results and original CPU analysis both match their corresponding Rust
  references. This distinguishes reduction correctness from replay acceptance.
- Strict 1e-6 absolute plus 1e-4 relative comparisons fail in 23 of those short
  shock fixtures. Quantization alone changes long TM coin-HSL Sortino by .1314%;
  existing executable-quantity differences and accumulated replay rounding explain
  additional input-curve differences. Worst observed equity relative difference is
  .0234%, weighted growth absolute difference is .163 basis points, and weighted
  ratio relative difference is .8437%. Use documented fixture-local family bounds;
  retain strict liquidation and controlled-producer policies and unchanged general
  comparison tools. These tests do not certify arbitrary trading trajectories.
- Actual directional two-coin replay with 28,800 observations, both curves and all
  fourteen metrics uses at most about 122.4 MB capture plus Torch scratch at batch
  64, within its 162.6 MB weighted reservation. Warm shared EMA replay changes from
  about .44 to .54 seconds; TM from about .71 to .80 seconds. These synthetic
  measurements exclude external driver/CuPy allocations and do not establish whole
  optimizer throughput acceptance. Requested-only capture is the foundation for
  future reductions; tuning accounts for its real cost rather than treating it free.
- All 24 ordinary weighted shock regressions and eight exact shared shock capture
  ablations pass. PR #1925 integrated reviewed head `bedbe2a3d4` after completed
  current-head automatic review, author review and all three required CI jobs.
  Broader raw growth/recovery/other weighted families, representative
  optimizer performance and legacy retirement remain separate acceptance work.

### 2026-10-07 — Compact unweighted raw strategy growth

- Eight actual fill/mark liquidations across midnight expose a wrong source curve:
  raw strategy ADG is about -.276 on the prior GPU path when Rust reports -1.0.
  Seven ordinary raw growth/ratio metrics differ in every case, while all 64
  ordinary account comparisons pass. This is a source-selection defect, not an
  accepted decimal difference.
- Add requested-only raw daily closes/minima alongside the existing raw daily
  drawdown summary. Eleven unweighted growth, gain-quality and ratio metrics use
  that factual curve. Calmar/Sterling request drawdowns independently. Preserve
  account aliases and exposure normalization before explicit raw replacements.
  No full history is needed: capture adds eight bytes per candidate per calendar
  day, including its actual daily storage in scratch admission.
- Keep optional daily-column ownership explicit. Adding raw closes/minima and
  drawdown made the old width-based decoder mistake raw columns for BTC metrics.
  Shared decoders now receive the known BTC feature flag; synthetic and actual
  combinations cover both ownership cases.
- Current rebuilt Rust passes 332 tests with one existing ignore and default-feature
  compile coverage. Shared controlled references cover fifteen curves at f64 and
  f32 input precision, using the actual Rust producer. All 52 focused reducer,
  liquidation, capture, temporal replay, decoder and service-guard checks pass.
  Capture on/off across both strategies, all side combinations, optional raw
  risk and BTC risk preserves every previous output exactly, including weighted
  metrics and recovery. The liquidation checks retain strict comparison policies.
- On 72 original CPU, quantized CPU and actual GPU raw curves from 24 public HSL
  shock cases, all 792 same-curve producer/reducer comparisons pass. Maximum
  absolute reduction error is below 9e-13. All 264 new compact results also match
  actual Rust reductions of the previously captured GPU curves; the largest
  residual is below 1e-8 and comes from f32 daily drawdown inputs to ratios.
- Strict 1e-6 absolute plus 1e-4 relative replay comparison passes only one of those
  short shock cases. Existing input-curve differences amplify near-zero two-day
  ADG and derived ratios: the worst ADG absolute difference is .208 basis points,
  gain-quality difference is .648 basis points and expected-shortfall difference
  is .848 basis points. Near-zero TM ratios differ by less than .0005 in absolute
  units; positive participation differs by less than .051 percentage points.
  Retain strict controlled/liquidation tests and use documented fixture-local
  materiality bounds for ordinary shocks. General parity-tool policy stays unchanged.
- All 24 current-source ordinary regressions pass with those fixture-local bounds.
  Affected callers and reducers pass 1,178 checks with one existing skip; two
  existing decoder checks also pass. Four actual CUDA optimizer interruption and
  resumption cases pass with CPU simulations forbidden, covering both strategies
  and standalone/scenario-screened suites. Six documentation tests pass and checks
  report no errors. Require current-head author/automatic review and CI before
  dev integration.
  Remaining recovery, other weighted/BTC families, representative performance and
  legacy retirement remain separate acceptance work. CUDA checks do not establish
  Metal device acceptance.


### 2026-10-07 — Observed portfolio HSL EMA tails

- The current six-case, 20-day public HSL/unstuck audit returns all 157 canonical
  metrics without missing or non-finite values. Only 66 comparisons have explicit
  policies; the remaining 876 pairs are unassessed, not passes. Full metric
  acceptance, representative resource/performance measurements and legacy retirement
  remain open.
- Rust's public portfolio EMA tail uses the per-bar maximum enabled scope signal.
  The GPU proxy instead took the maximum of the two reduced side tails. These
  operations differ: a controlled 200-bar Rust case returns 0.9 overall, 0.6 long,
  and 0.45 short. This is an aggregation defect rather than f32 noise.
- Shared replay captures a bounded joint observation summary and returns one
  additional scalar. Single-side temporal replay retains it in its existing state;
  fused replay observes both sides on the same bar. Unrequested EMA tails still
  compile away. Native result reduction requires the factual portfolio payload;
  the retained directional single-coin proxy keeps its older reduction.
- Repeating all 942 metric pairs with this change alters only the two dual-side
  portfolio EMA tails. TM relative error falls from 46.34% to .145%; EMA falls from
  2.76% to 2.25%. All other 940 GPU metrics are identical. Existing side-tail
  histogram approximation and other replay differences are not accepted by this
  narrow correction. General parity-tool policies stay unchanged.
- Current-source Rust passes 332 tests with one existing ignore and default-feature
  compile coverage. All 24 new CUDA/reducer regressions pass, including controlled
  scope observations, optional scalar ownership, exact non-tail shock ablation,
  temporal partial histories and both public twenty-day comparisons.
- The affected caller corpus passes 1,159 cases across its full run and selected
  rechecks. Four old dispatch mocks were corrected for the existing optional
  weighted-equity argument. Two 60-second future observation timeouts pass on
  unchanged rechecks; they do not establish a cold-compile latency guarantee.
- Ten final decoder and fused HSL-mode smoke checks pass. Direct smoke buffers now
  use the current scalar-width constant, mock initialization includes unrequested
  weighted metrics, and service assertions verify the joint portfolio summary,
  unified event attribution and factual raw side drawdowns.
- Four actual CUDA optimizer interruption/resumption cases pass with CPU simulation
  forbidden, covering both strategies and standalone/scenario-screened suites.
  Six documentation tests pass; checks report no errors. Require current-head
  author/automatic review and all required CI before development integration.
  General metric policies, representative performance and legacy retirement remain
  open; these CUDA checks do not establish Apple Metal device acceptance.


### 2026-10-07 — Controlled shape definitions and replay materiality

- Portfolio EMA-tail PR #1927 completed exact-head author and automatic review with
  no findings and all three required CI jobs passing before development integration.
  Master remains unchanged; its current HSL changes are already incorporated.
- Extend the existing public equity-only fixture using actual Rust producer values
  for unweighted/weighted choppiness, jerkiness and exponential fit error. Fifteen
  curves, f64/f32 input precision and three fill variants cover the six USD fields.
  The same fixture checks fixed-price BTC metric routing without claiming variable
  conversion coverage. CPU/CUDA together pass 2,160 comparisons; all 178 focused
  Python checks pass. Current-source Rust passes 332 tests with one existing ignore,
  default-feature compilation and a verified rebuilt extension.
- Diagnose the largest weighted-jerkiness residual using a public twenty-day TM short
  HSL shock fixture. CPU/GPU clocks agree and GPU daily closes match its captured
  curve exactly. Actual Rust reductions of CPU, quantized CPU and GPU curves match
  the corresponding Python reductions within 1e-12. A .05869% maximum curve deviation
  accompanies a 5.205% weighted-jerkiness metric gap; rounding the CPU output curve
  alone does not reproduce it. Keep replay/materiality acceptance open and evaluate
  candidate ranking rather than accepting a percentage from a single case.
- Four sixteen-candidate, twenty-day short HSL shock cohorts (both strategies, seeds
  43/47) preserve ADG/weighted-jerkiness fronts and all pair relations with zero
  best-candidate CPU regret. All 384 async service results match direct GPU outputs
  exactly across widths 4/16/auto. Reassessing all six shape axes finds two EMA fit
  front changes and nine pair-relation changes, including numerical TM ties. One
  choppiness gap reaches 29.524% symmetrically. Preserve these findings; no universal
  replay tolerance or limit acceptance follows from selected-objective agreement.
- Native width 16/auto warm cohorts take about .88-.90 seconds for EMA and 3.52
  seconds for TM; width 4 takes 2.83-2.88 and 12.76-12.77 seconds. CPU serial cohorts
  take 2.25-2.58 and 28.13-28.22 seconds. These measurements follow warmup, include
  only sixteen requests per run and do not complete an adaptive tuning window.
  Cold compilation, full optimizer and external-resource acceptance remain open.
- This slice adds test/reference evidence only. Keep general parity policies, runtime
  behavior and all broader performance, numerical acceptance and retirement gates
  unchanged. Require current-head author/automatic review and CI before integration.

- PR #1928 automatic review identified two test weaknesses: broad nonfinite matching
  could accept NaN/negative infinity for Rust's positive-infinity sentinel, and the
  existing Rust helper used a looser absolute floor than Python. Preserve shape
  sentinel type/sign explicitly in the shared fixture and both consumers. Give the
  new Rust shape checks the same 1e-10 absolute / 1e-12 relative tolerance as Python,
  while preserving the existing other-family checks. Require fresh source validation
  and completed current-head review/CI before integration.

### 2026-10-07 — Fill the first dispatch after worker-owned preparation

- Shape-reference PR #1928 completed fresh author/automatic review, addressed both
  automatic findings and passed Rust/Python 3.12/Python 3.14 CI before development
  integration. Master remains unchanged.
- The full optimizer resource trace exposed thirteen one-candidate dispatches for
  newly prepared datasets. Their replay times can exceed subsequent large batches;
  discovering safe capacity does not require simulating that ownership claim alone.
- Reuse the executor's queue-claim logic after the factory discovers capacity. Fill
  only from already queued compatible requests, respect cancellation and the physical
  bound, and retain oldest-dataset scheduling. No new orchestration/tuner state,
  simulations, precision changes or CPU fallback are introduced.
- The paired twenty-day, sixteen-coin EMA measurement in the acceptance record
  returns identical metrics/terminal status for all 64 requests. First-cohort time
  changes from 119.959 to 55.956 seconds; repeat times remain 54.764/54.801 seconds.
  Populated shader caches and gated submission are explicit measurement limits.
  One warm full-width sample now reaches the tuner; no completed tuning window or
  optimal-width claim follows. Keep demand-limited tuning and cutover gates open.
- Require final cancellation/capacity and real CUDA optimizer interrupt/resume
  checks, current-head author/automatic review and CI before integration.

### 2026-10-07 — Isolate CPU request preparation

- Move the unchanged NumPy synthetic candle generator out of the legacy benchmark
  and share it with parity fixtures. Import the GPU device helper only when packing
  device tensors, leaving canonical parameter transport and candidate planning
  independent of execution modules.
- Four fresh-process cases prepare actual requests for both strategies, with coin
  HSL enabled/disabled and unstuck enabled, while GPU imports and CPU simulations
  are forbidden. The prior fixture and model dependencies independently reproduce
  failures under those guards. Representative short/long fixture arrays retain
  identical bytes. Preparation checks alone do not certify CPU entry points or
  authorize retirement.
- Add six permanent CPU compatibility cases: both strategies generate actual
  backtest exports and PNG plots; both CPU optimizers start and resume two real
  workers for each strategy. Install the GPU-import guard at interpreter startup
  so it applies to forkserver/spawn workers as well as fork, without changing the
  platform's default context. Preparation, planning, offline tooling, packing-cache
  and documentation checks additionally cover affected consumers. Single-coin
  CUDA packing, worker-owned multicoin execution and CPU-forbidden optimizer
  interruption/resume pass for both strategies. Require current-head author/
  automatic review and CI before development integration.

### 2026-10-07 — Matched request assessment and raw recovery input

- Before correcting the recovery input below, replay all 1,068 accepted requests
  from a six-scenario, 160-complete-candidate
  synthetic EMA Anchor search through the retained synchronous shared-account GPU
  engine. Every metric and liquidation result matches native execution exactly.
  This compares identical simulation requests, not whole legacy optimizer throughput.
- The same pre-correction candidate assessment preserves all eleven
  Pareto members and all three objective winners. Across 12,720 candidate pairs,
  95 objective orderings differ. Generous configured limits produce no feasibility
  changes; that does not certify arbitrary tight thresholds or independent searches.
- Refresh six twenty-day EMA/TM long/short/both shock comparisons: all 157 requested
  metrics are present and finite. Keep unassessed policies explicit. Historical
  inventory measurements preceding semantic corrections are not current acceptance.
- Investigate four worst recovery/drawdown cases with aligned CPU and GPU curves.
  Strict recovery is discontinuous near equal samples: quantizing one CPU curve
  changes p95 from 12.468854 to 13.434931 days. Its GPU account curve gives 13.425105
  days, while its raw strategy curve gives 12.422327 days. Neither bit identity nor
  a universal duration tolerance is an appropriate inference from that case.
- Correct shared EMA/TM recovery samples to raw realized PnL plus UPNL, matching
  Rust's strategy curve and retaining losses below the account liquidation floor.
  Compile that scalar calculation when recovery is requested even without weighted
  capture. Account metrics, trading decisions and request/result boundaries stay
  unchanged. Sixteen liquidation regression cases independently reproduce the old
  account-floor substitution with both strategies, one/both sides, mark/panic-fill
  endpoints and recovery-only/weighted-capture requests. The verified rebuilt source
  passes 194 focused device cases, 332 Rust cases (one ignored), default-feature
  compilation and six documentation cases. Six refreshed twenty-day comparisons
  contain all 157 metrics with no missing/non-finite output, and preserve all 151
  non-recovery metrics per case exactly. Four additional actual recovery-objective
  optimizer cases pass interruption/resume and prompt result/Pareto persistence
  with CPU simulation prohibited. Seven disabled-HSL policy/specialization controls
  also pass. Require current-head review/CI before integrating the correction.


### 2026-10-07 — Separate completed-episode expiry from active entry estimates

- The clipped GPU controller unconditionally added the first retained realized
  cashflow as an entry-value peak, including completed flat episodes. Controlled
  integer-valued observations show that this can preserve cooldown after the
  in-window terminal signal becomes GREEN. Thirty-eight original-source probes
  produce 24 failures and 14 controls which pass; four real coin/pside long-cooldown
  regressions also fail while two clipped active-to-terminal controls pass.
- Restrict the synthetic entry reference to active exposure and its terminal
  accounting sample. Once a completed episode's window clips, rebuild its signal
  from retained observations like the Rust controller. Bind a terminal entry estimate
  to its first retained observation, preserving it across same-time budget changes or
  scalar-cache rebuilds, and drop it when that observation expires. Keep the active
  current-entry-loss estimate, terminal classification, request/output packing and
  optional HSL compilation. Two controller scalars record the reference and its time;
  temporal replay queries the compiled state size rather than assuming a fixed layout.
- Blanket removal of entry references was rejected: current exposed inventory can
  still require the Rust estimator's entry-loss signal after opening history expires.
  Removing that reference at the terminal sample could also suppress valid cooldown.
- Same-data CPU reconstruction and frozen-episode replay isolate a separate unified
  limitation. On the public two-coin, 3,000-minute EMA seed-43 shock fixture with
  threshold/span/cooldown 0.002/2.5/10000 and one-day retained history, the latest stop
  is at minute 1555. At minute 2955, retained closing fills cause Rust reconstruction
  to supply an estimated opening basis/entry peak. Reconstructing those fills matches
  the actual CPU backtest: halted through minute 2995, normal at 2996. Evaluating the
  frozen terminal observations with the same Rust controller becomes normal at 2955.
  This is a difference in reconstructed observations, not float32 decimal error.
- The earlier matched synthetic assessment is now repeated with the raw-recovery
  correction: all 1,068 accepted GPU requests preserve every unchanged requested
  metric exactly; canonical rescoring of 160 complete six-scenario candidates selects
  the same 11 CPU/GPU Pareto members, with no actual feasibility/liquidation change or
  CPU regret at the GPU-best candidates. Recovery p95's maximum absolute difference
  decreases from 0.95625 to 0.75827 days; 94 objective-axis pair orders differ. This is
  one matched candidate set with generous actual limits, not an independent CPU search
  or general approval of tight limits. It precedes the current controller change.
- The first rebuilt 202-case controller/real-backtest run passes 200. Two EMA
  integration cases fail only the local 0.1% fill-rate bound: measured counts are
  1942/1940 and 1902/1900 (CPU/GPU), while corrected durations, RED time, drawdown
  and bounded ADG checks pass. Keep a fixture-local absolute one-fill-per-day
  bound for those two recipes; the two TM cases retain their existing local 0.1%
  bound. General parity defaults remain unchanged. Rust passes 332 tests with one
  ignored, and default-feature test compilation passes.
- Four additional probes reproduce cache-dependent permission in the first simple
  revision: a same-time flat observation or budget change rebuilds without the entry
  reference that a cached terminal signal retained. Explicit reference/time ownership
  fixes all four against the Rust initial-incomplete-episode reference. The final
  rebuilt source passes those four and the remaining 202 rolling-controller and
  actual coin/pside regression checks; Rust again passes 332 tests with one ignored,
  and default-feature test compilation passes. Wider replay controls are running.
- The wider replay run passes 197 cases and skips 13 Apple-only cases. Its 51
  failures are episode-boundary reporting assertions which expect the already
  observed RED trigger to disappear at GREEN, a held scope or an unusable terminal
  budget. Three original-controller controls reproduce one failure from each
  assertion family. Correct those expectations to retain the observed event, like
  Rust reporting, while preserving permission/flatness/cooldown checks and adding
  an explicit unchanged-permission check for the unusable terminal budget. No
  producer change is made for these reporting-fixture repairs.
- Reconcile stale service ownership prose with the implemented prepared registry,
  native optimizer integration and CPU-owned persistent evaluation identity. The
  acceptance map consolidates deliberate native approximations by actual consumer;
  it does not turn their pending numerical/materiality decisions into passes.
- All 84 repaired episode probes pass, as do 16 specialized/full-capacity and
  temporal-layout controls and four CPU-forbidden optimizer interruption/resume
  cases. The six twenty-day shock audits return all 157 requested metrics with no
  missing/non-finite values and no GPU value changes from the raw-recovery source.
  Their undefined comparison policies remain unassessed. Ten long-history recipes
  give exact duration/RED-time agreement in eight coin/pside cases. Unified EMA
  retains CPU/GPU duration 1441/1400 minutes, ADG difference 0.00003638 and worst
  drawdown difference 0.00001038; unified TM retains 1385/1386 minutes, ADG
  difference 0.00003011 and worst drawdown difference 0.00000555. Strict measurement
  mismatches remain visible; general policies are unchanged. Six documentation
  tests pass, with no checker errors and the two existing size warnings.
- Final publication metadata verification, automatic review and required CI remain mandatory
  before integration. Long unified reconstruction differences remain explicit
  acceptance work; do not widen general parity policy or declare simulator cutover
  from these controlled expiry cases alone.

### 2026-10-08 — Reproduce unified history-expiry materiality before cutover

- The completed-window correction merged into the development branch only after
  current-head automatic review, author review, all Rust/Python CI and a fresh
  base/head/merge-base and metadata gate. New 64-candidate materiality evidence
  was investigated before merge, even though the original automated checks passed.
- The [acceptance map](gpu_optimizer_acceptance.md#unified-hsl-cohort-materiality)
  records four fixed synthetic cohorts, original-controller controls and the
  additional EMA stop. Same-observation Rust replay confirms the corrected
  cooldown release. Retained-fill reconstruction still changes trading and Pareto
  membership; this is an open producer contract issue, not accepted float32 noise.
- Share explicit fixture HSL policies and ordered price shocks between parity and
  cohort tools. Validate before device work, preserve prepared-input ownership,
  record resolved synthetic recipes, and keep general tolerance/feasibility
  policies unchanged. Publish reproducible commands rather than private traces.
- Do not alter CPU/live reconstruction or add a CPU validation backtest to make
  the new GPU path appear conformant. Investigate a bounded GPU reconstruction
  that preserves Rust's current facts, history clipping and cache independence.
  Complexity and throughput must be measured before selecting a producer design.
- All 49 new recipe, validation and report-provenance checks pass. The documented
  four-cohort command reproduces all nine measured CPU/GPU values, candidate
  parameter fingerprints and rankings for all 64 candidates; native results are
  exact against direct GPU at both measured widths. An existing passive-TM
  both-side 0.1% ADG assertion remains exceeded by 0.00003094 absolute; untouched
  target-branch tools reproduce it with identical default fixture inputs. Keep
  that existing discrepancy visible without widening policy in this tooling slice.

- Automatic review identified unsupported extreme cooldowns and missing fixture
  provenance after execution failures. Reject cooldowns outside Rust's signed
  millisecond range before device access and retain resolved recipes in both
  tools' failure reports. Regression coverage includes the exclusive timestamp
  boundary, the adjacent valid value and saved strict-JSON failure reports.
- Follow-up automatic review identified finite CPU values that overflow GPU
  float32 encoding. Check policy encoding and final stressed candles before any
  simulation; preflight shocked cohort seeds before CUDA initialization. Include
  compounded shocks and underflow-to-zero regressions, while keeping valid
  fixture arrays and the ordinary success recipe unchanged.

### 2026-10-08 — Bound exact side drawdown tails by the registered timeline

- A raw side drawdown tail needs only the worst floor(1%) of observed daily
  maxima. Retain those values directly, replacing the cutoff-bin approximation.
  Compile a power-of-two capacity from the prepared UTC day count; expose no
  new optimizer setting and include the capacity in shader-cache identity.
- Keep the unfinished current day separate and merge it only during a pure
  metric query. Early liquidation/truncation uses its actual observed day count.
  Temporal replay snapshots retain the complete bounded state, and requested-only
  feature guards remove it when unused. An undersized direct shader returns an
  invalid metric rather than fabricating a partial tail.
- The change concerns metric reduction, not HSL protection or CPU/live
  reconstruction. HSL EMA tails and the retained-history reconstruction gap
  remain separate acceptance work. Verify full replay/ablation/temporal outputs,
  source-verified Python callers and resource/throughput evidence before publishing.
- All 24 new capacity/diagnostic/full-replay tests pass with CUDA using the
  source-verified Rust extension; Rust tests pass (332, one ignored) and the
  default-feature check succeeds. The matched 211-day single-side control
  preserves every non-tail output. Independent daily-sort errors fall from
  8.97e-6/0.01427 to 4.55e-13/7.45e-9 for EMA/TM; CUDA local storage falls by
  240 bytes at capacity two, with unchanged registers. Warm throughput varies
  by -1.8%/+1.1%; claim no general speedup. See the
  [acceptance evidence](gpu_optimizer_acceptance.md#exact-side-raw-daily-tails).
- Focused validation also passes 20 shader/packing, 135 metric/scoring/doc,
  12 CPU/GPU side-parity, two shared raw/EMA reducer and four actual native CLI
  checks. The CLI cases interrupt/resume standalone and suite runs for both
  strategies while forbidding CPU backtests and CPU worker creation. Current-head
  automatic review and required CI remain mandatory before development integration.

### 2026-10-08 — Separate current requirements from development history

- Keep the acceptance contract and evolving checklist in their existing location;
  link the complete dated history from a separate decision log in the same directory.
- Preserve earlier decisions and measurements verbatim. Historical evidence remains
  subject to the current acceptance map and source-specific validation limits.
- This organization changes no implementation, numerical policy, completion criterion
  or branch/review gate. Future progress entries belong in this log.

### 2026-10-08 — Bound shared-account TM temporal replay

- Source inspection found that fused dual-side TM omitted the work budget and
  interrupt callback forwarded to single-side runners. Its dispatch planner also
  excluded two sides, and small long-history CUDA batches could bypass chunking.
- Reuse the existing directional replay-state layout as the common portion of the
  fused state, adding the short side and requested portfolio EMA-tail accumulator.
  Keep the per-bar trading body and finalization unchanged; restore complete state
  across chunks, initialize external output/HSL buffers only on the first chunk,
  and finalize scores only at each candidate's actual endpoint.
- Share one Python temporal dispatch loop and query each variant's actual state
  size. Count both sides in the work envelope. CUDA TM histories above 8,192 bars
  get interrupt boundaries below the work cap too; preserve Apple activation and
  unchunked launch options. No new optimizer controls or CPU simulations.
- Local checks pass 293 service cases, eight launch-contract cases, 332 Rust tests
  with one existing ignore, and default-feature compilation. The source-verified
  extension passes all eleven new actual CUDA regressions: requested optional
  buffers, three HSL modes, hedge/one-way accounts, unequal endpoints, repeated
  chunk sizes, fatal-marker continuity/reset and an actual prepared-service future
  interrupted after a completed kernel. Three long-history raw-tail controls and
  two directional partition controls preserve every raw output. Four actual CLI
  interruption/resume cases pass while forbidding CPU backtests and worker pools.
- The matched three-coin shared-account control preserves every raw output at
  chunk sizes 8,192 and 1,024. Warm median replay times are 5.731 seconds unchunked
  and 5.768/5.780 seconds chunked; final-repeat maximum dispatch times are
  2.976/0.388 seconds. State is 7,472 bytes per candidate in that variant, CUDA
  local storage grows by 32 bytes and registers remain 255. Record the complete
  public recipe and scope in the [acceptance map](gpu_optimizer_acceptance.md#shared-account-tm-temporal-replay).
  This is continuity/overhead evidence, not optimizer throughput, an optimal chunk
  size or a general elapsed-time guarantee. Duration tuning and larger-suite
  resource acceptance remain open.
- Current-head automatic review, author review, all required Rust/Python CI and
  fresh unchanged publication metadata remain mandatory before integration.

### 2026-10-08 — Match coin HSL when all retained fills expire

- A held position with no retained fills has no evidenced historical duration in
  Rust's reconstruction. Its current size, basis, mark and budget define a fresh
  single-sample loss signal. Keeping a GPU rolling profit peak in that case can
  produce a protective close that the Rust current-position estimate does not.
- Use the existing per-coin last factual position-fill timestamp in shared-account
  EMA/TM callers. After it leaves the inclusive lookback, clear only the disposable
  signal cache and seed the current entry-loss reference. Preserve reporting and
  let the ordinary current observation retire or create RED intent. Add no replay
  state, CPU validation, optimizer controls or CPU/live semantic changes.
- All 66 new source-verified CUDA regressions pass against Rust's empty-history
  evaluator: both sides and strategies, positive/negative UPNL, fractional spans,
  scaled coin budgets, repeated observations, scalar-cache rebuilding and the
  retained-fill boundary. Unknown fill times, flat/disabled scopes and aggregate
  modes keep their preceding controller behavior. Local Rust validation passes
  332 tests with one existing ignore and default-feature compilation.
- The broader source-verified HSL suite passes 244 cases with 13 Apple-only skips;
  all 72 checkpoint contract cases also pass with that extension. Eleven fused
  temporal CUDA tests preserve all outputs across chunk sizes, unequal endpoints,
  failure markers and interrupted service futures. Four actual native CLI cases
  preserve standalone/suite results and resume with scaled coin budgets while
  forbidding CPU backtests and worker pools. Six documentation tests pass.
- This change does not solve partially retained fill reconstruction or establish
  general HSL parity, selected-config materiality, resource or replacement acceptance.
  Current-head automatic review, author review, all required CI and fresh unchanged
  publication metadata remain mandatory before integration.

- Automatic review found that direct helper probes alone did not establish caller
  wiring. Add sixteen full native replay cases across both strategies and long-only,
  short-only and fused paths. Real initial fills precede or remain within the one-day
  lookback; a later profit peak/drop distinguishes their signals. Mixed-age fused
  sides distinguish each side's timestamp. Assert actual CPU/GPU fill counts,
  per-side GPU triggers and the intended fused path, without injecting positions
  or modifying the trading simulation.
  All 82 final CUDA cases pass. Keep the original review thread open for the
  updated-head review and require fresh author review and CI before integration.

### 2026-10-08 — Specialize unused unstuck EMA consumers

- Keep consumer proofs in execution rather than evolutionary orchestration. Prove
  enabled/gating combinations over packed float32 candidates and immutable per-side
  coin flags; one possible consumer keeps the general layout. Cache both compiled
  code and temporal state size by that decision. No device readback, optimizer
  control or persisted simulator state is added.
- Remove only unused EMA band state, initialization, updates and gated branches
  in shared-account EMA/TM. Retain ordinary EMAs, ungated unstuck, selection, loss
  accounting and finite history. Default single-coin sources remain general.
  This lays a consumer-specific foundation; full unstuck and inactive-side
  ablation remain open rather than being inferred from this smaller proof.
- Ten host-proof tests and thirty source-verified CUDA controls pass. Both
  strategies, long/short/fused paths, effective coin flags, mixed candidate gates,
  weighted/recovery metrics and TM chunk/layout changes preserve every finite
  returned value exactly and preserve intentional unobserved-sample NaN masks.
  Rust passes 332 tests with one existing ignore and default-feature compilation;
  all 38 coupling, preparation-isolation and documentation checks pass against
  the current extension. See the [acceptance map](gpu_optimizer_acceptance.md#unstuck-ema-consumer-specialization).
- The broader current-build suite passes all 172 unstuck, HSL, fused temporal and
  optimizer CLI controls, including eight standalone/suite interruption/resume
  cases that forbid CPU simulations and worker pools.
- The public three-coin, both-side, two-day fixture preserves every output across
  five alternating warm general/specialized runs of 64 identical candidates.
  EMA local storage/registers decrease from 7248/202 to 6976/198; TM local storage
  decreases from 7936 to 7904 bytes and temporal state from 6000 to 5744 bytes.
  Median replay kernel times change from 0.203985 to 0.192849 seconds for EMA and
  0.588270 to 0.572162 for TM. Keep these isolated controls separate from optimizer
  throughput, total resource bounds and optimal scheduling claims.
- Current-head automatic review, author review, all required CI and fresh unchanged
  publication metadata remain mandatory before development integration.

### 2026-10-08 — Learn from warm underfilled service dispatches

- Full-batch-only evidence can leave a bounded producer unable to tune: a nominal
  width of 64 with repeated warm cohorts of 63 records no window. The service now
  uses actual successful candidate counts and rejects each actual shape's first
  use. Dataset evidence, median smoothing, production thresholds, cooldowns and
  slower-trial rollback remain execution-owned.
- When queued demand or device headroom blocks growth, allow a smaller-width
  probe using subsequent real requests. Preserve fixed widths and the retained
  screening/validation tuner's full-batch policy. Add no user knob, extra replay,
  CPU validation, numerical mode or evolutionary behavior.
- The source-verified device suite passes 151 cases with one existing skip.
  Two new EMA/TM CUDA controls compare 36 requests each against fixed execution,
  count every request once and preserve every returned metric exactly. Their
  accelerated windows test integration; default-window host cases exercise the
  production evidence thresholds. Eight existing standalone/suite CLI controls
  pass with clean interruption/resumption and CPU simulation/pools forbidden.
- See the [acceptance map](gpu_optimizer_acceptance.md#underfilled-execution-tuning).
  Representative tuning quality, duration control and resource acceptance remain
  open. Require current-head automatic review, author review, all required CI and
  fresh unchanged publication metadata before development integration.
- A follow-up caller audit found the public cohort report still classified partial
  shapes as ineligible. Update its evidence adapter and documentation, and add
  regressions for warm partial windows and invalid observations. The current-build
  reporting/policy/docs suite passes 141 cases with one existing skip and three
  cohort device cases deselected. Require fresh-head review and CI for this addition.

### 2026-10-08 — Follow the recursive simulated touch in TM sizing

- Rust advances the simulated order-book bid/ask after each entry rung and
  recomputes the initial sizing floor. The multicoin GPU helper kept its original
  sizing price, oversizing short recursive additions. Advance that anchor with
  the simulated touch while preserving the original raw price when it controls.
- Compare sixteen-rung long/short GPU helper output with canonical Rust orders.
  Before correction the short case differs on thirteen quantities; the long
  control passes. Keep this narrow strategy correction separate from ongoing
  HSL reconstruction and default-worker adoption.
- The isolated correction passes 39 actual CUDA checks: both canonical-rung
  comparisons and 37 recursive gate/market/fused controls. A bounded real native
  optimizer CLI start/resume also passes with CPU simulations forbidden. The
  rebuilt extension passes 332 Rust tests with one existing ignore, default-feature
  test compilation, source verification and five documentation checks.

### 2026-10-08 — Develop retained-fill reconstruction before native integration

- Keep the current Rust HSL history/controller contract as the reference. A GPU
  reconstruction component uses retained finite linear simulator fills, a reverse
  reduction-requirement pass and a forward inventory/basis/cashflow pass. It reads
  an ordered clipped ring and writes disposable events; it does not own evolution,
  controller permission or an earlier inferred history.
- Fractional and mixed-magnitude cases exposed cancellation in ordinary float32
  accumulation. Use compensated float32 sums for inventory, reduction requirements
  and cashflows, and a stable weighted basis update. Keep the exchange-quantum
  guard around zero rounding so a genuine remaining lot cannot be erased.
- All 22 CUDA cases in `test_gpu_hsl_history.py` pass with a separately rebuilt,
  source-verified extension. Four parameterized cases compare 996 generated and
  targeted histories with Rust across long/short and both same-time candle phases;
  18 cases reject malformed facts, ring metadata or current inputs. Compare exact
  flatness and a single remaining lot independently of magnitude-conditioned
  float32 sample/event guards. These guards do not change native metric policy.
  Rust passes 332 tests with one existing ignore and default-feature compilation.
- This component is not yet called by native backtests. Do not claim resolution of
  retained-fill or aggregate HSL discrepancies from its unit coverage. Native work
  still requires bounded factual storage, actual fill-order/flat-boundary ownership,
  scope composition, cache-loss/rebuild and temporal-state equivalence, then full
  replay and candidate-front comparisons.
- Investigate coalescing only consecutive same-pair, same-direction, same-time
  executions. Preserve the first reduction price, weighted addition basis, net
  cashflows and actual flat/reopen ordering. Recursive TM can execute hundreds of
  entry rungs in one candle, so a fixed record for every fill is an expensive
  storage baseline. No compressor, record-count bound or overflow policy has been
  accepted or integrated yet; never silently drop facts to satisfy a memory cap.


### 2026-10-08 — Own factual capture and capacity recovery inside the GPU worker

- Develop opt-in native EMA/TM factual capture before replacing HSL evaluation.
  Keep records resident and metrics payloads modest. Record actual signed fills,
  fees, post-fill positions and global chronology; aggregate controllers borrow
  pair facts. Coalesce only consecutive same-pair/direction/time executions and
  preserve actual flat boundaries. No factual history may be silently truncated.
- Reject overflow explicitly, learn larger storage within the physical scratch
  budget, and retry only GPU work. Check cancellation before changing capacity;
  propagate malformed facts and other fatal inputs without capacity retries.
  Native testing exposed missing EMA callback ownership in the first recovery
  path; make the callback explicit for both strategies and worker construction.
- Initial compile-time capacities required another kernel compilation on each
  growth. Separate runtime capacity from compiled feature enablement. Keep learned
  storage size out of compilation identity while retaining it in scratch allocation
  identity and physical candidate partitioning.
- Develop factual scope-prefix selection separately from reconstructed valuation.
  Reverse actual post-fill inventory using the simulator's exchange-quantum rule,
  retaining the current/latest completed episode and its consumed global prefix.
  Same-minute flats/reopens must remain distinct; clipped exposed prefixes cannot
  acquire an invented flat seed. Thirteen real CUDA scope cases and the 47 preceding
  pair-history/writer cases pass. They do not establish native controller parity.
- Source-verified Rust validation passes 332 tests with one existing ignore and
  default-feature compilation. Keep native capture/recovery/temporal validation
  and subsequent controller/cache integration open; do not publish this opt-in
  foundation as a completed correction of the material retained-history gap.

- The current rebuilt source passes all 110 factual checks: 100 actual CUDA and
  ten host policy cases. All forty native capture/storage/retry/interruption cases
  preserve returned metrics and complete chronology; new temporal controls prove
  that later dispatches cannot erase overflow and a fresh retry resets factual
  state. The initial short temporal fixture produced too few fills to exercise
  overflow; extend the fixture instead of weakening its prerequisite.
  Mixed fatal/overflow marker tests require fatal input propagation before any
  retry. The separate current-build host suite passes 319 checks with forty device-
  named cases deselected. The preceding capture build also passes four actual
  coupled-request comparisons and all 52 native CUDA CLI checks. Native controller
  integration and candidate-front parity remain open.

### 2026-10-08 — Rebuild scoped permission from factual observations

- Compose global fill chronology, reconstructed pair inventory and current endpoint
  facts in a Rust-owned shared GPU component. Center cashflow against the common
  endpoint; retain terminal accounting, flat seeds, estimated entry references and
  per-minute EMA semantics. No previous panic decision is an input. Debug point
  traces are optional; ordinary evaluation returns compact scalars.
- Source-verified CUDA comparisons pass 36 tests covering 375 snapshots against
  Rust's full trace and controller. Preserve exact timestamps, flatness, exposure
  and actions with local float32 guards for numerical columns. This is component
  evidence, not acceptance of the native simulator's HSL objectives or Pareto front.
  Compact output and latest terminal scores agree without a full point trace.
- The first comparison exposed a one-observation cooldown delay: pre-fill price
  sampling delayed recognition of same-time exposure. Derive lifecycle reopening
  from the retained episode's causal fill time, independently of candle valuation
  phase. Add explicit completed, reopened, missing-close, current-opening and
  same-time round-trip comparisons; do not loosen action assertions.
- Retain actual current position size/basis in the unused bytes of the existing
  factual header. Close producers know their endpoint before the caller applies
  the mutation; entry producers supply the updated weighted basis. This avoids
  coupling a common evaluator to strategy-specific position arrays or opposite-side
  fill-helper parameters. Malformed endpoints remain fatal producer facts.
  All 157 endpoint/history/scope/capture/policy checks pass, including twelve
  malformed/normal endpoint probes and actual basis inspection in forty native
  caller cases. Nine temporal controls also preserve endpoint headers exactly.
- Native wiring still needs budgeted disposable events, selected scope prefixes,
  explicit ordinary/terminal phases, temporal/checkpoint ownership and reporting
  that does not recount historical lifecycle events on each rebuild. Start from
  fresh reconstruction as the reference; add disposable caches only with rebuild
  equivalence. Do not merge this foundation as a completed HSL parity correction.

### 2026-10-08 — Attach the factual evaluator to native simulation

- Attach one shared evaluation context to the single/fused EMA/TM kernel paths.
  Selected scopes read current factual headers and reconstruct into resident scratch;
  strategy-specific position arrays and evolutionary state are outside this interface.
  Ordinary close-time and terminal fill-time observations remain distinct.
- Reserve and budget sixteen bytes of disposable events per factual capacity slot,
  rounded to allocation nodes. Capacity growth retains compiled identity and retries
  only GPU work. A separate internal runtime control distinguishes evaluation from
  capture while independent comparison work is pending.
- Rebind every controller/context after loading a temporal chunk, before observation.
  The first 33 source-verified CUDA controls pass: eighteen topology cases, nine
  full/chunk comparisons, two growth/retry cases and four disabled-policy comparisons.
  Rust passes 332 tests with one existing ignore and default-feature test compilation.
  All 119 integration checks pass: 109 CUDA and ten host policy controls, including
  existing capture/recovery and scope comparisons. Six independent stressed two-coin
  comparisons cover both strategies and all HSL scopes. The five requested HSL metrics
  agree exactly; EMA fill rates agree, while TM retains twelve missing fills and small
  worst-drawdown differences. Sub-day ADG values are zero and prove no growth parity.
  Cold-start observations include host compilation and must not be
  reported as simulator throughput.
- Keep default worker adoption and semantic checkpoint cutover pending actual metric
  and candidate-front evidence. Full reconstruction is the initial reference; add
  disposable caching only with rebuild equivalence. No correctness or performance
  acceptance is inferred from dispatch/repetition equivalence alone.

### 2026-10-08 — Assess factual replay on matched candidate cohorts

- Repeat the preceding four sixteen-candidate, 3000-bar stressed cohorts with
  identical input/candidate identities and independent CPU references. All 64
  CPU rows match the prior nine metrics exactly. For EMA seeds 7/43 and TM seed 7,
  factual replay now preserves all five HSL metrics, Pareto membership and pair
  orderings. TM seed 43 still differs materially for candidates zero/two: halt
  durations differ by 148/149 minutes and time in RED by about five percentage
  points, changing Pareto membership. Preserve these outliers for investigation;
  do not explain them away with a general float32 tolerance.
- No authored limit flips or CPU ADG regret at the GPU's best candidate occur
  in these cohorts. These limited observations do not establish general search
  equivalence. Keep default worker adoption pending the remaining trajectory,
  checkpoint, lifecycle and resource evidence.
- Correct the multicoin test fixture to produce exactly its requested coin count.
  Expand actual single/fused dispatch and temporal controls to one/two coins.
  An initial assertion incorrectly used an absent proxy config attribute; replace
  it with the prepared coin identity. This was a test failure before GPU execution.
- Full reconstruction exposes significant hot execution cost. Before introducing
  a cache, test a simpler evaluation-local shortcut for fully consumed, flat,
  exact-zero signal tails. Keep real endpoint evaluation, logical observation
  counts and complete optional debug traces. Fresh reconstruction remains the
  reference. The rebuilt optimization passes 62 composer checks against 651
  Rust snapshots, 332 Rust tests with one existing ignore, default-feature
  compilation and ten host policy checks. All 100 actual native caller checks
  also pass, for 172 component/caller/policy checks overall. Matched cohort
  output/performance comparisons remain pending; cold compilation is outside
  simulator throughput claims. The preceding integration build now passes all
  sixty corrected one/two-coin dispatch, temporal, growth and disabled controls.

### 2026-10-08 — Preserve cohort output and isolate the remaining outliers

- All 64 matched candidates preserve all nine GPU metrics exactly after flat-tail
  compaction, both initially and in three warm repeats. Warm median times for
  sixteen candidates are 85.55/72.06 seconds for EMA seeds 7/43 and 37.64/36.87
  for TM seeds 7/43. No matched warm preceding-build measurement exists; report
  these as observations, not a speedup or performance acceptance.
- Detailed CPU exports preserve every fill and all nine metrics for the two TM
  outliers. Detailed reporting still uses CPU caches. A fresh standalone Rust
  scope/controller evaluation, supplied with the captured GPU facts, reproduces
  the divergent GPU restart decisions. Candidate zero flattens one minute later
  on GPU; candidate two flattens at the same minute on both. At the earlier
  restart boundary, candidate two's reconstructed terminal EMA differs by about
  0.00000334, straddling its 0.002 threshold. Window clipping amplifies an already
  different simulation history into a material halt-duration/ranking difference.
  Investigate the first differing executions before choosing a numerical policy;
  do not widen global tolerances or blame the new scope composer for these facts.
- Early captured execution facts expose a concrete upstream semantic difference:
  multicoin TM recursive sizing freezes its initial sizing price, while Rust's
  recursive generator advances the simulated book touch and recomputes that
  floor. Generation balance stays fixed on both. The short canonical sixteen-rung
  regression fails on thirteen quantities before correction; its long control
  passes. Update the shared helper's sizing anchor with the simulated touch.
  The rebuilt source passes both regressions, 37 affected recursive gate/market/
  fused caller checks, 332 Rust tests with one existing ignore and default-feature
  test compilation. Repeating the sixteen-candidate TM seed-43 cohort resolves
  candidate zero's HSL differences and restores full Pareto membership and
  drawdown pair ordering. Candidate two still restarts 149 minutes earlier,
  leaving two RED pair-order disagreements. No authored limit flips or regret
  at the GPU's best candidate occur. The remaining outlier is not accepted;
  this partial repair does not establish general search equivalence.

### 2026-10-08 — Separate accumulation drift from HSL transport precision

- The recursive sizing repair completed independent current-head review and Rust /
  Python 3.12 / Python 3.14 CI before merging to development. Preserve the broader
  factual-HSL work separately; master remains unchanged.
- Candidate two's next discrete quantity change is a near-half-step initial order
  at minute 393. Naively accumulating its independently recorded cashflows in f32
  reproduces the balance drift. A compensated-f32 diagnostic removes that first
  change, without requiring f64 replay state or different sizing rules.
- The same sixteen-candidate diagnostic retains the full Pareto front and ADG/DD
  pair order, with no authored limit flips. The 149-minute HSL restart difference
  persists. CPU-fact f32 encoding alone does not reproduce it. Keep later trajectory
  differences and materiality open; do not generalize this into parity acceptance.

- An independent CPU-only control preserves the original outlier's full exported
  fills and nine metrics. Starting-balance perturbations of plus/minus 0.003 change
  four quantity groups each, including two observed early rounding boundaries,
  while its 1401-minute halt remains unchanged. This does not explain or accept
  the GPU's 149-minute discrepancy. Keep localizing the remaining history change
  rather than generalizing the first repaired rounding boundary.

### 2026-10-08 — Retain small cashflows in f32 account balances

- A long sequence of small encoded fees/profits loses contributions when repeatedly
  added to a much larger f32 cash balance. Small resulting balance errors can cross
  quantity-rounding boundaries. Keep f32 execution and one residual in the shared
  account; carry it with temporal state rather than changing the sizing policy.
- This reduces accumulation drift, not every f32 execution difference. The independent
  canonical sum controls distinguish encoded cashflow error from accumulation error;
  real separate device dispatches verify residual continuity. All six new CUDA
  cases fail before correction and pass afterward. Twelve canonical Rust entry
  sizing controls, seven shared-account consumers and eleven production-capacity
  temporal/interruption controls pass. Both standalone and automatic suite CLI
  start/resume checks pass with CPU simulation forbidden. All 72 checkpoint
  contracts and six documentation checks pass with the verified rebuilt extension,
  alongside 332 Rust tests (one existing ignore) and default-feature compilation.
- A grouped full-capacity order probe leaves no admission headroom for the next
  fixture on CUDA. The same sequence fails on the preceding source; both probes
  pass in separate fresh processes. Production specialized replay checks pass.
  This test-resource limitation does not justify bypassing memory admission or
  establish general resource acceptance.

### 2026-10-08 — Verify the combined numerical and factual-HSL foundation

- The compensated account correction completed exact-head independent review and
  Rust / Python 3.12 / Python 3.14 CI before merging into development. The
  unpublished factual-HSL foundation retains identical tested code after integration;
  default worker adoption and checkpoint cutover remain pending.
- The verified combined build passes 216 focused controls: 62 CUDA composer cases
  against 651 Rust snapshots, ten host capacity/error policies, sixty actual native
  factual dispatch/temporal/growth/disabled cases, six cashflow continuity cases,
  72 checkpoint contracts and six documentation checks. Rust also passes 332 tests
  with one existing ignore and default-feature test compilation.
- Repeat all 64 matched candidates against the unchanged independent CPU references.
  Initial and warm GPU output agree exactly. All four Pareto fronts, all ADG/DD pair
  orderings and authored limit classifications match. All five HSL metrics match
  for both EMA cohorts and TM seed seven; TM seed 43 candidate two retains its
  149-minute early restart and two RED pair-order disagreements. No global numeric
  acceptance follows from these observations.
- Warm seconds per sixteen-candidate cohort are 85.27/67.12 for EMA seeds 7/43 and
  36.56/34.77 for TM seeds 7/43. These describe opt-in reconstruction on the
  reference device; no matched speedup or general performance acceptance is claimed.

### 2026-10-08 — Quantize partial-entry differences without losing aligned steps

- Independent fresh HSL comparisons, crossing CPU/GPU histories, current budgets
  and candle encoding, isolate the remaining restart difference to execution facts.
  Post-shock partial-entry quantities start differing before recursive compounding.
- The canonical Rust producer floors the f64 difference `0.094 - 0.066` to 27
  quantity steps, although both quantities are aligned and their difference is
  28 steps. Actual order APIs reproduce this on both sides; all four long/short
  grid/trailing regressions fail before repair while four genuinely below-step
  controls pass. Use the existing ULP-bounded downward quantizer at the four
  partial-entry subtraction sites. Global rounding and sizing policy stay separate.
- Simpler f32 basis-expression diagnostics also repair three canonical ladder
  price ticks, but do not resolve the complete HSL outlier. Keep those prototypes
  out of production while the narrow canonical producer correction is assessed.
- The verified rebuilt extension passes 333 Rust tests (one existing ignore),
  default-feature compilation and all seventeen Python order checks. Twelve CUDA
  entry-sizing comparisons, two CPU-forbidden optimizer CLI start/resume cases,
  72 checkpoint contracts and six documentation checks pass.
- Recompute the same 64 independent CPU references against the retained verified
  factual-GPU results, whose shader implementation is unchanged by this CPU fix.
  All five HSL metrics now agree in every cohort; candidate two also agrees on
  fill count. ADG and RED pair order and authored limit classifications match.
  A small DD ordering/front difference remains in TM seed seven, with DD regret
  about 0.00000022 at the GPU's minimum-DD candidate. Preserve this numerical
  observation; resolving the material HSL outlier is not general parity acceptance.

### 2026-10-08 — Memoize only the factual scope cutoff

- Preserve fresh retained-fill reconstruction and current controller inputs. A
  simulator-owned memo stores only the factual flat-prefix cutoff, keyed by exact
  selected/exposed pair identities and the sum of monotonic fact versions. Native
  fill append, coalescence and pruning advance those versions; new candidate/retry
  initialization discards both rings and memo. Quantity quanta stay immutable for
  a prepared replay. No old signal, budget, price or permission is cached.
- Validate headers before every hit. The optional compile-time cache switch also
  removes its temporal-state fields, and actual compiled state-size queries retain
  allocation ownership. All 21 prefix/memo device controls pass, including selected
  and unrelated fills, pruning, exposure changes, explicit discard and malformed
  headers. The 62 composer cases against 651 Rust snapshots and ten host policy
  cases also pass. Rust passes 333 tests with one existing ignore and default-feature
  test compilation. Native full/chunk/growth cases and performance remain pending.
- The partial-entry correction completed current-head author and independent review
  and Rust / Python 3.12 / Python 3.14 CI before merging to development. Integrate
  it without changing the unpublished factual/cache source. Default adoption and
  checkpoint semantic cutover remain open; master is unchanged.

- Accept the remaining measured near-indifference for these four synthetic cohorts,
  rather than chasing identical f32/f64 trajectories. All five HSL metrics and the
  authored feasibility checks match; ADG/RED ordering is preserved. The single DD
  ordering/front difference has selected-candidate regret about 0.000000220.
  The observed ADG/DD/fill-rate residual envelopes are retained as bounded evidence,
  not promoted into a universal tolerance. Broader lifecycle/metric/resource gates
  still apply; this small difference alone does not block worker adoption.

- All sixty native factual dispatch, temporal continuation, bounded growth and
  disabled-policy controls pass on the cache-enabled build, plus six actual
  compensated-cashflow continuity controls. Eighteen selected checkpoint checks
  and five AI-documentation checks also pass. Cached-versus-fresh matched cohort
  measurements are running; no throughput gain is claimed from these tests.

- The first two paired cache experiments preserve every requested metric on all
  32 EMA candidates, both initially and in two warm repeats. Warm uncached/cached
  seconds per sixteen candidates are 85.25/85.27 versus 9.30/9.32 for seed seven,
  and 68.12/72.85 versus 10.54/10.61 for seed 43. The measured ratios are about
  9.16x and 6.67x. Torch allocation peaks agree in each pair; that is not a total
  driver/host memory measurement. TM measurements remain pending. Reuse only the
  cutoff, not a prior controller signal. These observations support keeping this
  small memo; they do not establish whole-optimizer or general workload speedups.

- All four alternating-process paired cohorts finish successfully. All nine
  metrics remain identical across uncached/cached runs, initial execution and two
  warm repeats, with both Python and Rust CPU backtests forbidden. The two TM
  cohorts improve from warm medians 37.64/37.77 seconds to 31.63/31.77, about
  1.19x each. Keep the small factual-cutoff memo. All five HSL metrics agree with
  the corrected independent CPU references; the previously bounded small TM
  drawdown/front difference is unchanged. Torch allocation peaks match each pair,
  without establishing total driver/host memory or full optimizer performance.

- The final history/capture recheck passes all 99 cases: 59 pair reconstruction,
  endpoint, ring and compacted-view controls (including 996 generated/targeted
  histories against Rust), and forty real native chronology, storage, temporal,
  retry and interruption controls.
- Author inspection finds that universal runner interruption wiring gives separate
  default lambdas to compatible scenarios, changing their batch keys. Use one
  shared no-op for the multicoin constructor while preserving explicitly distinct
  callback ownership. The real shared-packing/default-callback regression fails
  before correction; its explicit shared-callback control passes. Both pass after
  correction, with CPU simulation forbidden. All 54 affected suite-key/topology
  and fused-construction controls also pass. The GPU shader and compiled Rust
  artifact are unchanged by this host-side correction.


### 2026-10-08 — Select factual HSL behind the native service boundary

- The opt-in factual foundation completes current-head author and independent
  review without findings. Rust and both Python CI jobs pass; the foundation
  is integrated into the optimizer development branch.
- Enable factual replay only in the native CUDA service. Determine capture needs
  from effective dispatch parameters and coin overrides. Keep legacy screening
  mode separate; an inactive HSL dispatch omits factual storage/capture and
  preserves the existing one-side EMA disabled-HSL specialization.
- Start with a conservative worker-owned factual capacity (at most 256 records
  per pair), learn upward from rejected GPU overflow within the physical budget,
  and retain that estimate across HSL-on/off/on dispatches on the same runner.
  No new optimizer-facing tuning field or device ownership crosses the service API.
- Version native saved-fitness semantics from execution one to two. Reject old
  checkpoints instead of silently reusing their old HSL fitness; configurations
  remain usable seeds. Preserve CPU contracts and the legacy CUDA runtime contract.
- All 38 new actual device transition, override and asynchronous service controls
  pass, along with ten host retry/fatal-policy checks and both checkpoint rejection
  controls. The cutover stays local until its integrated result is validated
  and reviewed; wider evidence follows below.

- The wider native lifecycle/loss comparison passes all 42 cases against CPU
  references outside optimization. The native request path stays CPU-free. The
  documentation-adjusted source also passes 77 host session/data/tuning/retry/doc
  checks and eighteen selected checkpoint contracts. All 56 broader device
  controls also pass: 52 actual optimizer CLI cases, two canonical prepared-data
  controls and two incremental service/resource cases. Source identity is
  unchanged after the full lifecycle/CLI validation.
- A deliberate duplicate initial candidate reproduces an existing anchor-resume
  test's fixed-twelve assertion failure with fully evaluated/persisted 3/4/4
  cohorts. Count actual distinct generations, reconcile durable records and
  evaluator counts, require fully evaluated survivors and prove completed resume
  performs no new evaluation. Both duplicate and ordinary controls, plus related
  native host tests, pass (23 cases). Production search and replay are unchanged.

- The final integrated cutover passes 140 host and 144 actual device controls.
  Short-only and fused short-only coin policies are isolated. Omitting only fused
  TM short override admission in a separate negative control fails its raw HSL
  metric comparison, with the other seven override cases passing. This validates
  the new regression without changing the production implementation.

- Accept both independent worker-review findings: derive native capture from
  effective per-coin enablement, preserving strict policy validation while omitting
  unused history-readiness checks; refresh learned physical limits after successful
  replay before subsequent queue claims. Validate disabled-policy GPU outputs and
  fixed/automatic scheduling against the preceding implementation before publication.
- The corrected source passes 220 host, 146 factual replay/capture CUDA, eighteen
  execution-view (four CUDA/fourteen host) and twelve real CLI/data/service CUDA
  controls. All six disabled-policy regressions and both scheduling controls fail
  the preceding production. Preserve the single scratch-owner assertion through
  residency metadata rather than relying on a bound-method implementation detail.
  Recheck unchanged passing sources; require fresh independent review and all CI.
- Prioritize compact factual storage, dataset-owned capacity retention across
  residency eviction and representative long-held/many-coin scaling measurements.
  The current layout calculation reserves about 109.95 MiB per candidate in a
  25-coin/two-side/90-day example, only 0.61 MiB of which is factual storage. Preserve
  legacy consumers while removing that cost from the replacement path.
- Hold further launch tuning until these resource gates are addressed. Treat
  guarded incremental reconstruction as a measured follow-up requiring parity;
  shared strategy-neutral allocation/retry/dispatch extraction remains a focused
  simplification candidate, without a general backend framework.


### 2026-10-08 — Separate native factual HSL storage from observation replay

- Native factual shaders own compact headers, records and disposable events,
  without observation-window/controller storage or initialization. Preserve the
  old layout for retained observation consumers; include layout mode in compiled
  library identity and TM replay-state sizing. Disabled native HSL owns no scratch.
- Actual CUDA allocation on 25 coins, both sides and a 90-day lookback requests
  642,304 HSL bytes per candidate instead of 115,287,744 bytes. Buffer size and
  allocator requested bytes agree for both strategies. Measure allocator padding
  separately; do not claim total-resource or end-to-end throughput improvement.
- Both new device allocation regressions fail preceding production on the legacy
  allocation assertion. Source-verified Rust/default compilation, host callers,
  actual CUDA replay/partition/lifecycle/reference checks and real CLI callers pass;
  the acceptance document records the bounded validation surface. Independent
  current-head review, CI and development integration remain required.
- Continue dataset-owned capacity retention and representative long-held/many-coin
  scaling measurements before additional launch tuning. Keep the service API and
  factual signal semantics unchanged by this storage simplification.


### 2026-10-08 — Retain capacity estimates independently of residency

- Keep only small integer factual-capacity hints in service-owned dataset metadata.
  Restore them before recreated runner admission and remember successful learned
  capacity even after an HSL-off dispatch. Release device buffers normally; add no
  checkpoint or persistent replay state. Reject malformed/over-budget hints.
- Source-verified host checks pass, including four production-residency eviction
  controls with fake device transport/computation. A restoration-omission control
  makes all four fail at the reset capacity. Four actual CUDA scenario/owner-switch
  controls cover EMA/TM, unchanged metrics, zero repeated overflow retries and
  evicted-runner release. CPU backtests remain forbidden inside the native service.
- Require integrated compact-storage validation, current-head independent review
  and CI before development merge. Representative suite/total-resource acceptance
  and long-held reconstruction scaling remain separate gates.
- The reviewed compact layout is integrated with unchanged capacity production
  and host regression hashes. Combined validation passes 154 host and sixteen
  actual CUDA service controls, including all four eviction regressions. All checked
  sources remain unchanged; Rust/shaders use the reviewed compact extension.
  Require a fresh independent review and CI for this integrated head.


### 2026-10-08 — Measure the held-history reconstruction bottleneck

- Twelve source-verified native request cases cover EMA/TM, 2/25 long-side coins
  and 512/1024/2048 minute histories. Isolated CPU references prove the intended
  entry-at-64/no-close trajectory; dataset identities and five requested metrics
  agree within the stated tolerance. GPU on/off/on-repeat results agree exactly;
  all checked source files remain unchanged.
- Both strategies show roughly fourfold warm HSL-enabled cost when history doubles.
  At 25 coins and 2048 bars, warm requests take about 22.5 seconds with HSL enabled,
  versus 0.34 seconds for EMA and 0.46 seconds for TM with HSL off. This is measured
  single-candidate synthetic scaling, not universal optimizer throughput or a
  90-day simulation result. Record the public recipe and bounded resource limitations
  in acceptance; do not claim Torch counters measure all device allocations.
- Prioritize a compact guarded active-episode recurrence over further launch tuning.
  Retain fresh reconstruction as the reference/fallback for changed facts, budgets,
  clipping, causal phase and numerical concerns. Require cache-loss/temporal controls,
  paired factual and CPU parity, resource evidence and review before adoption.
  The strategy-neutral shared-runner extraction remains a separate focused follow-up.


### 2026-10-08 — Integrate resource fixes and tighten continuation guards

- Compact native factual storage and dataset-owned capacity retention are integrated
  on development through [PR #1945](https://github.com/enarjord/passivbot/pull/1945)
  and [PR #1946](https://github.com/enarjord/passivbot/pull/1946), after clean independent
  current-head review and all required CI. Master is unchanged. Reconcile the checklist
  to distinguish completed integration from remaining representative acceptance.
- The local active-episode continuation experiment passes its initial component
  comparisons, but exact native comparisons expose approximately 5e-9 short-side EMA
  diagnostic residuals. Do not count the interrupted validation as accepted parity.
- A current endpoint becomes a historical sample on the next evaluation. Require no
  same-end-minute fill and exact agreement of reconstructed final inventory/basis with
  the actual endpoint before seeding continuation; otherwise reconstruct fresh.
  Three new seed-denial regressions fail against the preceding draft without these
  guards. Corrected component/native validation and paired performance/resource
  evidence remain required before adoption.
- Extend temporal controls to both retained factual and native factual layouts;
  the earlier controls exercised only the retained factual layout. Keep the shared
  runner extraction separate from this numerical/performance change.


### 2026-10-08 — Verify stable continuation and paired held-request scaling

- Corrected validation passes 82 actual CUDA components and eighteen native
  policy/on-off-on/capacity comparisons with exact raw-output agreement. All
  checked sources remain unchanged. Three stable-prefix and six scalar-guard
  omission regressions fail as expected. Retain the strict comparisons; the
  initial draft's residuals are not an accepted numerical exception.
- Twelve paired native future cases preserve input identity, all five returned
  GPU metrics exactly and independent CPU references within 1e-6 absolute/relative
  tolerance. At 25 coins and 2048 bars, warm EMA/TM requests improve from
  22.480/22.497 seconds to 0.561/0.649 seconds. The 25-coin history-doubling
  shape becomes approximately linear; smaller cases have noisier timings.
- Torch allocation peaks match across every measured variant/phase. This does
  not measure compiler/private kernel, total driver, host or disk resources.
  Exclude cold compilation from the warm ratios and keep whole optimizer/search
  claims separate. Record the public recipe and full bounded table in acceptance.
- Continue paired mixed-candidate, actual-native temporal, lifecycle/CLI and
  multi-entry/clipping controls before publication. Keep fresh factual replay as
  reference/fallback, with no action cached and no new optimizer tuning knob.


### 2026-10-08 — Bound continuation claims with mixed candidate evidence

- Four paired cohorts preserve all nine GPU metrics for 64 distinct candidates
  exactly against preceding factual replay and between fresh/continued variants.
  Refreshed CPU references preserve exact agreement on all five HSL lifecycle
  metrics; existing trajectory differences and the TM seed-7 drawdown near-tie
  are unchanged, with zero limit flips. No numerical policy is widened.
- Busy-cohort warm medians remain essentially unchanged (EMA about 9.3/9.7
  seconds; TM about 8.5/8.2). Keep the large held-episode improvement scoped to
  stable exposed histories. Continuation adds 320/336 compiler-reported local
  bytes for EMA/TM, with 255 registers unchanged. Torch peaks match; post-replay
  whole-device free snapshots differ by about 22 MiB. These are not total peaks.
- All checked sources remain unchanged after paired comparisons. Continue
  actual-native temporal, lifecycle and real CLI tests, then representative
  multiple-entry and lookback-boundary controls before integration/review.


### 2026-10-08 — Complete continuation replay and optimizer caller checks

- Corrected continuation passes 350 broader replay checks, including 36 exact
  temporal controls across retained/native factual layouts, 42 native lifecycle/loss
  checks and twelve real CLI/data/service checks. Eight CLI combinations cover
  bootstrap/resume with CPU simulation forbidden. All checked sources remain
  unchanged; preserve the existing strict raw-output comparisons.
- Keep representative fill/clipping comparisons and total-resource evidence
  separate from these completed caller gates. Independent review and CI remain
  required for development integration.


### 2026-10-08 — Bound continuation with additional entries and clipping

- Four paired two-coin/2048-bar requests preserve every requested GPU metric
  exactly, separate CPU references within 1e-6 absolute/relative tolerance and
  matching Torch peaks. Additional-entry cases make four EMA/fourteen TM fills
  at bars 64 and 128; one-day lookback cases exercise clipping. Sources remain
  unchanged. Record the reproducible recipe and timing table in acceptance.
- Multiple-entry warm cost remains about 2.5 seconds; clipping improves from
  about 2.3 to 1.0 seconds. Retain conservative endpoint guards and the fresh
  fallback. These constant-mark cases have zero risk metrics and do not replace
  nonzero component, mixed-candidate or lifecycle evidence.
- The bounded continuation slice is ready for independent review and CI. Broader
  numerical, total-resource and whole-optimizer gates remain open; do not present
  this slice as completion of the optimizer redesign.


### 2026-10-08 — Give multicoin replay a strategy-neutral owner

- Extract the existing multicoin allocation, history, retry and result-decoding
  lifecycle into a private replay base. EMA Anchor and Trailing Martingale are
  sibling adapters; each owns its parameter layout, override columns, packing,
  library identity and kernel dispatch. Diagnostic names no longer select binary
  layouts or capabilities. Preserve existing public runner names and the separate
  directional single-coin family; introduce no general backend framework.
- Preserve fused strategy dispatch methods and shared numerical behavior. Host
  controls change diagnostic labels while checking actual layout selection, native
  HSL enablement/learned capacity and unstuck specialization. All four regressions
  fail the preceding implementation and pass the extracted owner. These are
  source-only checks, not GPU simulations.
- Require actual CUDA replay, temporal/retry, residency and real native optimizer
  callers with a verified extension, plus independent current-head review and CI,
  before integration. Keep this ownership cleanup separate from guarded HSL
  continuation and its numerical/performance acceptance.


### 2026-10-08 — Compact recovery output within shared replay ownership

- A native policy-switch control with two coins/both sides and 512 minute bars
  shows raw recovery histories escaping physical replay admission. Set a 200,000-byte
  scratch budget and retain a legal 512-record factual estimate after HSL-off work;
  a 24-request HSL-on cohort splits into one-candidate replays. Factual scratch,
  the recovery owner, cloned histories and their joined allocation coexist at
  248,388 bytes, before subsequent reduction scratch. This is a bounded history
  observation, not total device accounting or a capacity-learning reproduction.
- Let native metric-service runners reduce each accepted physical recovery history
  before sub-batch cloning/joining. Preserve raw output as the direct runner default
  for diagnostics and legacy consumers. Keep the existing GPU recurrence/reducer
  and metric semantics; do not reduce rejected factual attempts into fitness.
- Extend the focused shared-owner extraction to own this compact result transport.
  Reject malformed or ambiguous pre-reduced results; preserve the original opt-in
  raw postprocessor. Nine host transport checks pass. Updated liquidation controls
  observe reducer inputs, preserving raw-trajectory evidence without relying on
  raw histories being returned from the metric service. Actual CUDA regression,
  broader callers, independent review and CI remain required before integration.


### 2026-10-09 — Complete shared-owner and compact-result validation

- Affected validation passes 706 host/source/device checks: 154 execution/data/
  residency/tuning, 301 service, fifteen layout controls, 35 recovery, 116 factual
  replay, 53 device/transport, four optional-history, sixteen expired-history and
  twelve real CLI/data/service cases. The latter retain CPU-forbidden optimizer
  bootstrap/resume coverage. All checked sources remain unchanged after each run;
  the final derivative changes only two budget tests and the acceptance record.
- Both strategy regressions fail the preceding raw recovery transport at the
  specific 248,388-byte allocation overlap. Corrected compact transport combines
  672 bytes of recovery summaries for 24 requests and preserves results against
  raw GPU diagnostic controls. No comparison tolerance or numerical recurrence
  changes. Rust and shaders are unchanged in this slice.
- Earlier fixed-width volume/equity controls also fail preceding code: HSL-off
  releases its initial factual allowance, permitting widths six/five. Correct
  tests to assert the effective physical history envelope; do not constrain
  production scheduling merely to preserve an outdated width expectation.
- Keep guarded continuation as a separate reviewed slice. Integrate the actual
  development head, validate affected combined callers and obtain independent
  current-head review plus required CI before merging the shared-owner slice.
  Representative resources, numerical materiality and replacement retirement
  remain whole-project gates.


### 2026-10-09 — Validate cleanup against the continuation development merge

- Guarded HSL continuation is integrated on development through
  [PR #1947](https://github.com/enarjord/passivbot/pull/1947), after clean independent
  current-head review and Rust/Python 3.12/Python 3.14 CI. Master is unchanged.
  Reconcile the checklist and supersede the resolved benchmark HSL-gap wording
  while preserving the measured residuals and broader acceptance limits.
- The shared-owner/compact-result branch includes that actual development merge.
  Resolving only changelog and decision-log conflicts preserves both implementations;
  reconciliation with the subsequent development merge changes ancestry only.
- Combined source-verified validation passes 216 affected checks: fifteen layout,
  35 recovery, 134 factual replay, four optional-history, sixteen expired-history and
  twelve real CLI/data/service cases. Rust/shaders match the reviewed continuation
  runtime exactly. All 952 checked sources remain unchanged. Current-head
  independent review and CI are still required for cleanup integration.


### 2026-10-09 — Integrate shared replay ownership and measure service suites

- [PR #1948](https://github.com/enarjord/passivbot/pull/1948) integrates the
  strategy-neutral multicoin owner and compact physical recovery results on
  development after clean independent current-head review and Rust/Python
  3.12/Python 3.14 CI. The tested combined production tree is unchanged by the
  development merge. Master remains unchanged.
- Add a focused offline native service benchmark over three shared-data scenarios.
  Keep CPU/GPU parity and evolutionary ranking in their existing tools. Measure
  incremental completion, requested histories, residency, allocator/global device
  resources, process-tree RSS and disk cleanup without CPU simulations.
- Preserve default tuner evidence windows. Require completed windows per scenario
  when requested, bound their automatic extension and report insufficient evidence
  explicitly. A small underfilled cohort is insufficient evidence of tuning quality;
  broader queue demand must exercise real width trials before drawing conclusions.


### 2026-10-09 — Keep compatible demand through a tuning evidence window

- A finite-cohort controller reproduction finishes a warm evidence window with
  no backlog, although earlier work in that same window had enough compatible
  requests for growth. The preceding policy probes width two instead of eight
  from width four. Demand from the last completion alone loses this evidence.
- Retain one compatible-demand maximum per scenario for the current window;
  consume it when the window completes and reset it when the prepared ceiling
  changes. Invalid observations and unrelated scenarios cannot supply demand.
  Keep existing smoothing, cold-shape rejection, headroom, rollback and shutdown
  policy. No new simulator, checkpoint or scheduling framework is introduced.
- Five regression controls reproduce the tail failure before the change and pass
  afterward, including dataset isolation, stale-window/ceiling reset and invalid
  work. Actual CUDA/default-window resource and caller checks plus independent
  current-head review and CI remain required before development integration.


### 2026-10-09 — Validate current admission and native callers

- The preceding development code reproduces two stale assertions in the CUDA
  physical-bound control: post-replay ceiling checks produce repeated observations
  of one runner, and HSL-off releases its initial factual allowance, allowing
  seven requests to run as two then five instead of repeated width-two batches.
  Correct tests to validate the current owner-owned ceiling, request identities,
  equivalence and distinct-owner eviction/cleanup. No production change is needed
  for these preceding-code test failures.
- Four corrected actual CUDA bounds controls and twelve CPU-forbidden native
  optimizer bootstrap/resume controls pass. Together with the completed current
  tuning/host controls, 114 affected checks pass with one existing Metal-only skip.
  Both fresh strategy preparation-only CLIs keep GPU imports absent. Rust and
  shaders are unchanged. Complete the default-window resource comparisons before
  publication and current-head independent review/CI.
- Record the reproducible 33,408-result baseline separately in acceptance: all ten
  metrics match isolated GPU references exactly, one dataset is resident, shared
  arrays remain unchanged and spill cleanup/sampling succeed. This measures a
  moderate synthetic suite, not long/busy HSL or evolutionary search quality.


### 2026-10-09 — Keep bounded evidence failures explicit

- A 128-round current-demand suite completes 51,072 requests with exact requested
  metrics, unchanged arrays, clean spill removal and no sampling errors. Base
  width-128 and width-32 trials reject insufficient gains; the faster early/late
  width-128 trials still await the 30-second threshold. Extend the benchmark
  within a larger bound instead of weakening production evidence requirements.
- The final TM adaptive-accumulation caller completes 384 exact results and clean
  resource cleanup. Its sixteen-candidate cohort does not complete tuning windows;
  retain that limit rather than interpreting automatic execution as converged.
- A deliberately short EMA run preserves all 36 valid results and exits two
  for insufficient per-scenario evidence. Keep report completion separate from
  tuner evidence sufficiency. Instrumentation is installed inside its cleanup
  scope so early CUDA setup failures cannot leave the replay method replaced.
