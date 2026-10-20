# Offline optimizer benchmark suite

`passivbot tool optimizer-benchmark` freezes populations once and measures them in
two independently built checkouts. It uses synthetic market data, no accounts,
and no downloads. Benchmark workers and fresh CPU search pool children reject network
connections while permitting local Unix-domain IPC needed by multiprocessing, including
forkserver workers. Market preparation
from exchange caches or public downloads is outside its timing scope. Prepared search
also builds the recording evaluation contract once before the backend timer, matching
the production optimizer. Its timer includes backend startup, evolution, scoring,
checkpointing and result persistence; independent Rust verification follows the timer.

## Reproducible procedure

1. Pin both commits. Use clean isolated checkouts and separate Python environments
   with matching dependencies. Rebuild and verify each Rust extension using the
   [Rust runbook](ai/runbooks/rust_extension.md).
2. Generate a workload from the suite checkout. Keep that checkout outside the two
   implementations being compared. The manifest stores the seed, actual vectors,
   effective candidate configs, effective bounds, array hashes, and case policy.
3. Run smoke first. The controller measures one backend at a time, alternates
   branch order between cases/modes, and saves progress after every measurement.
4. Run a bounded pilot with more history, seeds, candidates and repetitions. Inspect
   failures and resource use before running the complete catalogue or long history.
5. Preserve the workload, report, logs, environment versions and exact commits
   together. Use fresh experiment directories; the tool refuses to overwrite them.

```sh
passivbot tool optimizer-benchmark catalogue

passivbot tool optimizer-benchmark freeze --profile smoke --output /tmp/optimizer-smoke

# Supply the checkout and Python executable for each pinned implementation.
passivbot tool optimizer-benchmark run --workload /tmp/optimizer-smoke \
  --backend baseline "$BASELINE_CHECKOUT" "$BASELINE_PYTHON" \
  --backend candidate "$CANDIDATE_CHECKOUT" "$CANDIDATE_PYTHON" \
  --modes cpu:1 cpu:4 gpu:1 gpu:8 --repeats 1 --timeout 120 \
  --output /tmp/optimizer-smoke-results

passivbot tool optimizer-benchmark freeze --profile smoke \
  --case '*/both/forager/base/*' --seeds 7 43 --candidates 16 --bars 2880 \
  --output /tmp/optimizer-pilot

passivbot tool optimizer-benchmark run --workload /tmp/optimizer-pilot \
  --backend baseline "$BASELINE_CHECKOUT" "$BASELINE_PYTHON" \
  --backend candidate "$CANDIDATE_CHECKOUT" "$CANDIDATE_PYTHON" \
  --modes cpu:1 cpu:4 gpu:1 gpu:8 gpu:16 --repeats 3 --timeout 180 \
  --output /tmp/optimizer-pilot-results

# Real production optimizer, including scoring, checkpoint and result persistence.
# A backend with a periodic drift gate needs enough iterations to exercise it.
passivbot tool optimizer-benchmark run --workload /tmp/optimizer-smoke \
  --case 'trailing_martingale/long/single/*' \
  --backend baseline "$BASELINE_CHECKOUT" "$BASELINE_PYTHON" \
  --backend candidate "$CANDIDATE_CHECKOUT" "$CANDIDATE_PYTHON" \
  --modes search-cpu:4 search-gpu:8 --search-iters 64 --timeout 180 \
  --output /tmp/optimizer-search-results

passivbot tool optimizer-benchmark compare \
  --workload /tmp/optimizer-pilot --report /tmp/optimizer-pilot-results/report.json \
  --output /tmp/optimizer-pilot-comparison.md
```

The standard/extended catalogue defaults to 128 candidates, seeds 7/43/101 and
10,080 bars. `--random-seeds N` chooses and records N fresh seeds. Reusing the saved
workload preserves the actual populations even if a future sampler changes.
For a targeted repeat, add `run --seed 7` (repeat `--seed` for several seeds) and
an optional `--case` filter. Selection keeps the original workload hash and saved
vectors; it never regenerates candidates. The report records selected entry IDs.
`--boundary-candidates` replaces the first two vectors with repaired low/high
corners, retaining random candidates. `--bounds-config` reads only `optimize.bounds`
from a canonical local config. Derived bounds and candidates remain private inputs.

Numeric genes use production `Bound.random_on_grid` and `enforce_bounds`, including
significant-digit repair. Both NumPy and Python RNG states are isolated. Cases pin
inactive sides, slot topology, disabled controls, zero weights and zero retracements.
The public short template pins exposure to zero; default short workloads instead
declare a positive 0.5..1.5 exposure range. HSL stress cases declare short signal,
threshold and cooldown ranges to exercise the controller on bounded histories.
These class-specific effective bounds are recorded, rather than silently pretending
to use an unchanged production search distribution.

## Coverage

The catalogue includes the 18 strategy/direction/universe baselines, individual
feature cases, a deterministic binary pairwise covering array, selected combined
stress cases, and an explicit CPU-only v7 strategy case. TM baselines use passive
entry and close orders (zero retracement); the entry/close trailing cases enable
retracement separately. There is no duplicate passive feature case.

| Dimension | Declared variants |
|---|---|
| Strategy and sides | TM, EMA Anchor, CPU v7; long, short, both; hedge and one-way |
| Universe | One coin, fixed multicoin, competing forager slots |
| HSL | Off, coin, pside, portfolio, market/limit panic, zero/finite cooldown, slow memory, budget scaling |
| Unstuck | Off/on, EMA gating off/on |
| Cooldown | Zero, constant, exposure/directionality weighted and clamped |
| Exposure | Position/total enforcers, both total-reduction policies, entry gate, excess allowance |
| Strategy controls | Entry/close trailing, passive behavior, EMA gate modes, volatility/exposure weights |
| Coin policy | Inheritance and sparse coin overrides |
| Selection/data | Ranking weights, unilateralness, hysteresis, validity windows, minimum cost, dynamic exposure |
| Execution | Slippage, fill buffer, realized-loss gate, coarse candles, BTC collateral capability |
| Metric work | Core metrics and completion, HSL metrics, optional weighted/recovery/gap metrics |
| Prices | Oscillating, trend, shocks |

Some entries intentionally describe CPU-only capabilities. Native GPU EMA Anchor
position enforcement, BTC collateral and v7 remain explicit `unsupported` results.
Other runtime failures are errors; the suite does not relabel them as successful skips.

Declared configuration coverage does not certify transitions. Inspect requested
HSL trigger/red-time metrics and candidate activity. There is no general counter
proving every cooldown, forager or enforcer decision, so those cases retain that
limitation. Multiple scenario/exchange suites, resume/anchor-specific qualification,
and user-supplied historical data preparation retain their existing integration
tests; this first benchmark catalogue measures one prepared exchange/scenario per
candidate. A long synthetic window is not historical market evidence.

## Interpreting results

`report.json` retains raw samples and bounded-stop/error results. `comparison.json`
contains throughput, latency, observed runner widths, per-candidate numerical
comparisons and bounded Pareto/rank diagnostics. `comparison.md` is a compact table.
Independent CPU verification of persisted search objective values is outside the
search timer. Search paths can diverge after the same initial population, so equal
iteration counts do not by themselves establish equal search quality.

- CPU serial latency includes payload preparation and simulation. Multicore replay
  uses immutable shared market arrays and includes candidate serialization in warm
  throughput. Spawn/import cost remains in first use and whole-process time.
- GPU width one submits requests serially to measure isolated service/replay latency. Batched time divided by
  candidate count is amortized cost, not individual request latency. Native request
  completion latency includes queueing and collection.
- Throughput is total completed evaluations divided by total measured wall time,
  never the unweighted mean of per-batch rates. A candidate has one scenario here.
- Frozen GPU replay selects the implementation's supported native service, or a
  supported historical direct-replay path. Direct-replay timing omits any separate
  CPU exact-validation pipeline. V9 requires the shared-account native service;
  a missing capability fails explicitly. `search-gpu` uses each implementation's
  selected production backend. Inspect the recorded path rather than inferring
  it from a branch name.
- `run --phase-profile` adds opt-in frozen-GPU phase measurements to each
  cohort and runner observation: packing, upload/clear, library compilation,
  lazy pipeline preparation, kernel execution, readback and metric reduction.
  `outside_replay_seconds` measures caller work outside replay, including service
  submission/collection; it is not a pure queue-time measurement. First use can
  spend most of its time creating a Metal pipeline after library compilation.
  Older implementations without the separate pipeline field retain a combined
  `kernel_execution_and_pipeline_preparation` phase and explicitly report that
  pipeline preparation was not measured separately. State-size probing belongs
  to compiled ABI preparation, outside replay execution.
  Profiling adds synchronization overhead; use separate unprofiled runs for
  acceptance throughput. This option does not profile production search modes.
- Search modes accept `run --host-profile` for a main-thread function profile of
  the production backend, including its preparation, scoring and persistence.
  Independent Rust verification runs after the profiler stops. Function times are
  instrumented wall time and include waits; cumulative times overlap. The report
  separately records main-thread CPU time. GPU-worker and CPU-pool threads are
  outside this profile. Use separate unprofiled campaigns for speed comparisons.
- Warm runs exclude setup and first use. First use may reuse on-disk Metal/CUDA
  compiler caches; it is not a cold-machine measurement.
- Runner-call candidate counts are observed execution widths, not measured hardware
  occupancy. For two directional runners, counts may include both side invocations.
- Process-tree RSS can double-count shared pages. Sampled peaks can miss transients.
  Metal reports driver/current allocations and recommended working set; those share
  system RAM. Whole-system available memory and swap growth are also sampled.
- Default CPU/GPU tolerances match the provisional parity tool policies. Unknown
  metric policies and missing liquidation status remain incomplete. `--policies`
  supplies explicit policies. Failed/incomplete correctness blocks an equivalent-work
  speedup claim while retaining the diagnostic timing.
- A successful command exit means measurements completed. Inspect numerical,
  constraint and ranking results separately; completion is not a parity pass.
- Pareto/rank diagnostics use ADG and drawdown and are bounded to 512 candidates.
  They do not establish repeated-seed evolutionary quality or near-limit agreement.

The controller uses POSIX process groups (macOS/Linux). Time and memory limits apply separately to each worker process tree. Timeout or
insufficient memory headroom stops that local process group and records the reason.
No other process is signalled. Harness edits during a campaign are rejected by its
source digest. Resource noise and thermal effects still require repeated paired
experiments on an otherwise idle machine.

## Autotuning

Tuning windows and cohort completion times share an origin immediately after
parameter preparation and service setup. First-use compilation and replay remain
inside that clock; preparation and setup are reported separately.

Use a sufficiently large backlog and sustained run, separately from smoke:

```sh
passivbot tool optimizer-benchmark freeze --profile standard \
  --case 'trailing_martingale/both/forager/base/*' --seeds 7 \
  --candidates 128 --bars 512 --output /tmp/optimizer-tuning

passivbot tool optimizer-benchmark run --workload /tmp/optimizer-tuning \
  --backend baseline "$BASELINE_CHECKOUT" "$BASELINE_PYTHON" \
  --backend candidate "$CANDIDATE_CHECKOUT" "$CANDIDATE_PYTHON" \
  --modes gpu:8 gpu:32 gpu:64 gpu:auto --tune-seconds 75 --timeout 180 \
  --output /tmp/optimizer-tuning-results
```

The report records actual completed production-controller windows, width decisions,
time to first window and time to three cohorts within 5% of the best measured fixed
width. Controller replay rates and caller wall throughput have different scopes.
Zero windows means insufficient tuner evidence even if cohort throughput is high.
Automatic runs stop after at most 10,000 cohorts. The sweep finds the best measured
safe setting, not a globally optimal GPU configuration. Use separate runs with
`--dispatch-budget` to compare dispatch work budgets.

Workloads, results and logs are private by default. Public contributions should
contain only the harness, synthetic recipes, tests and procedure; do not publish
operator configs, absolute local paths, raw reports or machine-specific provenance
without case-specific approval.

## Ordinary-account economics and useful discoveries

The Python APIs in `tools.optimizer_benchmark.account_metrics`, `replay_account`
and `discovery` support a separately registered economic campaign. The existing
`run` command's replay/ranking diagnostics do not automatically establish these
economic or timing criteria. Calibrate the criteria first, freeze the protocol
and input/runtime identities, and preserve unsuccessful trials as well as successes.

`extract_account_metrics` requires a full ordinary account-equity grid, a separate
pre-first-fill capital baseline, gross exposure observations, complete fills and
authoritative completion/liquidation flags. Net return is final equity divided by
initial equity minus one; geometric daily growth uses the actual observed duration.
Account drawdown includes the initial baseline. Recovery episodes measure time
from the preceding peak and explicitly identify unrecovered terminal episodes as
right-censored lower bounds. Fees and rebates are reported separately; signed fee
cashflow is already included in net realized PnL and account equity, so it is not
subtracted from return a second time. Nonpositive equity has no fabricated growth
score. The replay assumes fixed initial capital and no external transfers.

`metrics_from_native_replay` consumes full native outputs from each version's own
CPU runner. It uses ordinary account equity, without substituting HSL signals or
strategy-equity definitions. Native equity timestamp `k` stores the valuation at
`close[k]`, observable at the end of that candle. The adapter advances its clock
accordingly and retains a separate initial baseline. Expected activation/end bars
and any pre-evaluation bootstrap rows must be declared before replay. Bootstrap
rows require known-flat capital and no earlier fills; their warmup gap does not
count as evaluated coverage. Missing evaluated observations remain errors or an
explicit incomplete prefix, rather than invented flat history.

Gross exposure is the sum of absolute long and short position sizes times each
causal close price and contract multiplier, divided by positive initial capital.
Hedges remain gross, and a depleted or clamped current balance cannot hide or
inflate exposure. Fill indices keep a next-bar opening position out of the
preceding close even when their timestamps coincide. Exposure averages use
left-held close samples; they do not estimate an intrabar position/price path.
Missing marks for held inventory, missing multipliers, contradictory position
transitions and metrics-only output are unsupported evidence. Prepared valuation
inputs are checked against their frozen hash before and after extraction; the
scenario identity must also bind all other actual runner inputs.

`DiscoveryCounter` requires explicit account objective axes/reducers, economic
constraints, positive activity requirements, native-unit novelty tolerances,
required and distinct held-out inputs, per-model CPU runtime hashes, a common
starting pool, K and a time budget. It supplies no economic cutoffs. Exact
canonical effective configurations are counted once. Broad unique feasible
newcomers are reported separately from novel nondominated frontier additions
beyond the starting pool. Near copies of the initial pool or earlier useful
additions remain excluded even after their representative is dominated.
`target_measure` explicitly chooses K retained newcomer frontier choices or K
historical frontier additions; both counts remain available.

`canonical_configuration_identity` binds the full materialized policy, including
evaluation dates, venue, capital and fees. When scenarios use distinct effective
configs, use `compound_candidate_identity` on the complete label-to-config-hash
mapping. Supply `scenario_effective_configs` for exactly the frozen search and
held-out labels to `DiscoveryCounter`; each confirmation must prove both the
compound candidate hash and its selected `scenario_configuration_sha256`.
Workers verify that selected hash against the actual own-version loader output
before replay. Changing one scenario's policy invalidates the compound identity;
changing dates does not permit reusing another scenario's confirmation. The
single `effective_config` record remains available when every scenario uses that
same configuration. Fixed data-input and runtime hashes are checked separately.

Every counted discovery needs CPU confirmation under the matching model on all
required and held-out scenarios plus completed export. Held-out results must not
feed search fitness. Partial, GPU-only and unexported evidence remains pending;
failed, inactive or liquidated candidates cannot become neutral feasible scores.
Use one cold clock starting before preparation, including startup, compilation,
tuning, search, CPU confirmation, held-out evaluation and export. The bridge's
clock callback runs after account extraction; starting-pool confirmation time is
also included. Target misses are right-censored at the registered budget. The
utilities validate declarations and supplied evidence; the campaign must measure
the phases and independently verify runtime identity. No performance result or
readiness claim follows from these APIs alone.

Current workload manifests use version 2 and finite v9 HSL policy. Older manifests
and retired permanent/threshold restart feature requests require a fresh workload;
their saved fitness is not comparable under the changed risk model.
