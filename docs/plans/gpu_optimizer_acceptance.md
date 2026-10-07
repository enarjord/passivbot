# Native GPU optimizer acceptance evidence

This is an evidence map for the [development contract](gpu_optimizer_contract.md),
not a simulator certification or permission to retire the existing `gpu` backend.
The replacement remains experimental. Test coverage establishes the stated cases;
it does not establish every feature combination or production search quality.

## Ownership and lifecycle foundation

| Requirement | Evidence | Scope and remaining limits |
| --- | --- | --- |
| CPU search, GPU execution | `gpu_native_backend.py` uses ask/tell and canonical CPU scoring; `gpu.native`, `executor`, `datasets` and `residency` own execution independently of evolution | The service is reusable outside optimization. Replay adapters still retain legacy names and preparation helpers. |
| Immutable registered inputs | `test_gpu_datasets.py`, `test_gpu_service_acceptance_cuda.py` | Metadata and request parameters are snapshotted; borrowed views reject writes and source arrays remain unchanged after execution. The original owner must keep shared segments immutable and alive until service shutdown. |
| Packing reuse and mutable isolation | `test_gpu_cuda.py::test_cuda_prepared_service_reuses_packing_and_bounds_resident_scenarios`, `test_gpu_execution_tuning_cuda.py` | Compatible execution views reuse immutable packing; scenario switches release inactive runners. Candidate order, batch shape and repeated dispatches preserve GPU reference results. |
| Incremental completion and bounded admission | `test_gpu_service_acceptance_cuda.py` | A real first result returns while later replay is deliberately held; that result frees queue capacity and replacement work is accepted. Full admission still rejects excess work. Completion remains at a microbatch boundary, not an individual kernel lane. |
| Bounded replay allocations | `test_gpu_service_acceptance_cuda.py`, residency and dispatch-limit tests | Repeated requests stay inside the warmed Torch allocation envelope; one dataset and one replay's scratch are resident. This does not measure total driver/CuPy VRAM, host RSS or disk usage. |
| Failure, cancellation and cleanup | `test_gpu_executor.py`, `test_gpu_residency.py`, CUDA prepared-service failure tests | Producer failures stop admission; original exceptions survive cleanup failures. Attachments and spill files are released on setup failure, interruption and shutdown. No CPU fallback supplies a result. |
| Adaptive scheduling preserves semantics | `test_gpu_execution_tuning.py`, `test_gpu_execution_tuning_cuda.py`, `test_native_pipeline.py` | Width changes use completed production work; device dispatch respects work/scratch ceilings. CPU result grouping adapts independently. CUDA policy tests shorten evidence windows; they prove integration, not optimal default tuning. |

The focused real-device acceptance test uses each strategy, three coins, both sides,
512 synthetic minute bars, enabled unstuck and a finite realized-loss window. It
compares five isolated GPU references with 165 service requests per strategy, including
reordered candidates and changing dispatch shapes. Both cases pass on an RTX 3070 Ti
Laptop GPU with the current source-verified Rust extension. Enabled risk features in
this fixture do not imply every controller transition occurred.

The allocation test warms admitted shapes, resets Torch peak statistics and measures
later traffic against that observed envelope. It is a regression against growth with
request count. Existing residency tests separately cover scenario/view eviction and
packing reuse. Neither substitutes for representative large-suite memory measurements.

## Search, scoring and storage

`test_native_results.py` checks canonical single/suite aggregation, objective order,
limits, incomplete screening rejection and invalid-sample sentinels without simulation.
`test_native_backend.py` covers constrained/unconstrained two-objective search and
four-objective NSGA-III using an identified fake execution service. Those tests prove
CPU search behavior, not CUDA simulator parity.

`test_native_backend_cuda.py` runs the real optimizer CLI against offline synthetic
prepared inputs. It exercises standalone/suite operation, fixed/automatic dispatch,
starting configs, screening/promotion, interruption, continued resume, finite anchors
and coupled coin spans. CPU simulation APIs and CPU worker-pool construction are
replaced with failures. The first full result is read through independent result and
Pareto file handles inside the recorder callback, before cohort completion or shutdown.
Only complete candidate records enter storage and fitness. Resume uses saved anchors
after their original seed files are removed.

`test_native_datasets_cuda.py` additionally checks canonical lazy scenario preparation,
noncanonical source-column order and screening-to-full row reuse on the device.
Prepared arrays and metric collection have separate lifetimes; service shutdown precedes
release of borrowed source data. Saved fitness retains evaluation identity and execution
precision. Checkpoint rejection/failure tests remain in the CPU orchestration corpus.

Immediate file visibility is not an `fsync` or power-loss guarantee. A crash between a
flushed result and a newer checkpoint may repeat GPU work after resume. That is allowed
by the contract; this design does not add a transaction journal or perfect search replay.

## Architectural simplification

| Responsibility | Existing screening/validation optimizer | Native replacement |
| --- | --- | --- |
| Fitness and publication | GPU proxy search plus selected CPU validation and archive fitness | One complete GPU metric payload, canonical CPU scoring and direct persistence |
| Heavy work scheduling | GPU dispatch plus CPU exact-worker pools and validation queues | One bounded service, identified futures and CPU candidate collection |
| Runtime evidence | Proxy/exact pairing, drift scaling, probes and halt state | Independent parity tools outside optimization; ordinary simulation failures inside it |
| Resume state | Proxy evolution, exact-validation progress and pending validation metadata | CPU algorithm/cohort, complete fitness and separate partial screening evidence |
| Tuning | GPU batch and CPU exact-worker coordination | Service width/accumulation and independent CPU consumption cadence |

The gain is fewer authorities and independent ownership, not a promise of fewer files
while both backends remain. `gpu_native_backend.py` delegates request preparation,
collection, checkpoints and canonical scoring to focused CPU modules. CUDA code does
not select survivors, evaluate limits or update Pareto. Cohort NSGA-II/III survival still
waits for its evaluated offspring, while service completions, replenishment and storage
proceed within the cohort. Steady-state evolution is a future experiment, not an initial
acceptance requirement.

One active dataset is an intentional simple residency policy. Retaining more scenarios,
overlapping transfers or routing several GPUs can be implemented behind the service
later; they must preserve request identity, bounded admission and per-candidate behavior.
Do not add a second simulator or replay trading decisions in Python for those changes.

## Recovery distribution resolution

Requested recovery distributions retain every simulation step and use the GPU
strict time-to-exceed reducer, including plateaus and unrecovered terminal tails.
Sampling is compiled out when none of these metrics is requested. Full samples
stay on the GPU; only seven summary columns pass to host metric processing.

Dispatch-local reduction scratch replaces the global mutable buffer cache. A replay
keeps only its current sample-buffer shape. History budgeting reserves 16 bytes per
sample per candidate for samples, a possible contiguous-view copy, an index stack
and a duration histogram, plus 56 bytes for the summary and scaled output. Native
and retained service paths share the outer batch limit so internal replay splitting
does not concatenate oversized sample histories before reduction. This is a history
budget, not a complete device/host memory bound.

`test_gpu_recovery_resolution.py` covers strict plateaus, decreasing series, sparse
samples, terminal padding, one/five-minute intervals, independent CUDA-stream scratch,
and actual native dispatch under a two-candidate history budget. Twelve short native
replays compare all six distribution metrics with real CPU backtests across both
strategies, long/short/both sides and one/two coins. Native-only budget cases forbid
CPU simulation and preserve output identity across dispatches.

The public thirty-day synthetic fixture illustrates the resolution repair (all values
in days). Before/after inputs and Rust sources are identical; non-recovery metrics
are unchanged. These are observations with undefined acceptance policies, not passes.

| Strategy | Metric | CPU | Hourly GPU | Per-step GPU |
| --- | --- | ---: | ---: | ---: |
| EMA Anchor | Mean | 0.006653244 | 0.053182870 | 0.006702424 |
| EMA Anchor | p95 | 0.044444444 | 0.083333336 | 0.044444446 |
| EMA Anchor | Mean worst 1% | 0.077516757 | 0.142857149 | 0.077959850 |
| Trailing Martingale | Mean | 0.000852400 | 0.041608796 | 0.000853817 |
| Trailing Martingale | p95 | 0.001388889 | 0.041666668 | 0.001388889 |
| Trailing Martingale | Mean worst 1% | 0.001509732 | 0.041666668 | 0.001525844 |

Reproduce the current observations with this recipe, then change `--fixture` to
`trailing_martingale`. HSL is enabled but this fixture records no HSL transitions.

```sh
passivbot tool gpu-parity --fixture ema_anchor --sides both --coins 4 \
  --bars 43200 --seed 43 --hsl coin --gpu-engine native --metrics \
  adg_strategy_eq adg_strategy_eq_w backtest_completion_ratio \
  drawdown_worst_mean_1pct_strategy_eq drawdown_worst_strategy_eq \
  entry_interval_hours_p95 fills_gap_p95_hours fills_per_day \
  hard_stop_duration_minutes_mean hard_stop_post_restart_retrigger_pct \
  hard_stop_time_in_red_pct strategy_eq_recovery_days_mean \
  strategy_eq_recovery_days_mean_worst_1pct strategy_eq_recovery_days_p95 \
  volume_pct_per_day_avg_w
```

The comparison remains `comparison_incomplete`: eleven metric policies are undefined,
and existing strict ADG/fill discrepancies remain visible. Remaining mean/tail recovery
differences still require materiality assessment. The following volume slice removes
partial-day omission in shared replay; its trajectory residuals remain separate. No
general policy was widened and these slices do not establish general simulator parity.

## Traded-volume normalization and suffixes

Canonical CPU volume sums `abs(fill_qty) * fill_price / usd_total_balance` at each
actual fill, then divides by the number of filled UTC days. An additional `c_mult`
factor changes this metric's canonical definition. Shared EMA/TM GPU
accounting now follows that definition without changing sizing, fees or PnL.

Weighted volume averages the full analysis and up to nine trailing suffix analyses.
Suffix cutoffs use the actual equity sample horizon, including terminal truncation,
with the CPU's rounding and early stop for empty suffixes. Fills at a cutoff are
included. A suffix with no fills contributes zero; a filled day remains counted even
when its normalized contribution is zero. Partial UTC days must include their actual
post-cutoff fills rather than being dropped or admitting pre-cutoff volume.

When requested, shared replay captures a float2 per step: normalized volume and fill
presence. A Rust-owned GPU reducer computes the suffix averages after replay and
returns one float per candidate. Histories stay on device, retain only the current
batch shape and enter the shared dispatch budget. The budget conservatively reserves
16 bytes per step (capture plus a possible contiguous copy), plus 28 bytes per candidate
for bounds, compact output and size metadata. No capture buffer or instructions are
enabled when weighted volume is unused. The retained legacy directional single-coin
and raw daily-only helper still have their earlier approximation.

`test_gpu_weighted_volume.py` separates independent reducer-definition tests from real
CPU/native comparisons. Coverage includes midnight and intraday boundaries, one/five-
minute steps, actual short horizons, repeated cutoff indices, empty late suffixes,
zero contributions, malformed bounds, non-unit contract multipliers, both strategies,
long/short/shared sides, one/two coins, temporal chunks, candidate reordering and
changing batch shapes. CPU-forbidden native requests check optional capture, bounded
dispatch and compact outputs. Real CPU/native liquidation fixtures for both strategies
verify the shortened horizon. Metric-decoder coverage preserves other weighted metrics.

Validation passes 817 affected Python/CUDA checks with the final source, 330 Rust
tests (one existing ignored test), default-feature Rust compile checks and five
documentation checks. A broader earlier CUDA/service run passed 556 tests and skipped
one Apple-only case; its two outdated dispatch fixtures are corrected and all four
enabled/disabled CUDA/MPS argument-layout cases pass in final validation. The final
loaded extension and source/test content are verified. Actual Apple Metal execution
and complete resource/performance acceptance remain outside this NVIDIA-first slice.

One busy two-day short-only TM fixture with two coins, seed 43 and `c_mult=2` has
20,787 CPU fills versus 20,794 GPU fills. CPU weighted volume is 14.454426350 and GPU
14.493937492, a 0.273% trajectory gap. Independently reducing the captured GPU history
gives 14.493941567 (about 0.00003% from the GPU reducer). The fixture regression records
a case-specific 0.3% bound; this is not a general parity-policy approval. The remaining
fill-trajectory discrepancy and its optimizer materiality are open.

Reproduce strict standalone measurements with public synthetic inputs:

```python
from optimization.gpu.parity import MetricTolerance
from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

metrics = ("volume_pct_per_day_avg", "volume_pct_per_day_avg_w")
inputs = fixture_inputs(build_parser().parse_args([
    "--fixture", "trailing_martingale", "--sides", "short", "--coins", "2",
    "--bars", "2880", "--seed", "43",
]))
for coin, market in inputs[2].items():
    if not coin.startswith("__"):
        market["c_mult"] = 2.0
report = run_comparison(inputs, "binance", metrics,
                        {name: MetricTolerance(1e-7, 1e-4) for name in metrics},
                        gpu_engine="native")
print(report["status"], report["metrics"])
```

This stricter policy deliberately exposes the residual as a mismatch. Comparison
tools may run CPU references; native optimization still performs no CPU simulation.

The same two-day recipe with `--sides both` reproduces the normalization and suffix
repairs. Switch the fixture to `ema_anchor` for the other strategy. Measurements use
identical input identities before and after the change:

| Strategy | Metric | CPU | Previous GPU | GPU fill-suffix reduction |
| --- | --- | ---: | ---: | ---: |
| EMA Anchor | Volume average | 4.432764260 | 8.865523338 | 4.432761669 |
| EMA Anchor | Weighted volume | 2.218288739 | 0.886552334 | 2.218294144 |
| Trailing Martingale | Volume average | 30.051934663 | 60.120517731 | 30.060258865 |
| Trailing Martingale | Weighted volume | 14.535288578 | 6.012051773 | 14.539561272 |

EMA passes the strict recipe's policy. TM retains roughly 0.028%/0.029% volume
trajectory differences and remains a strict mismatch; they are not execution failures.

Repeating the thirty-day, four-coin, shared-side recipe in the recovery section leaves
input identities, CPU results and the other fourteen GPU metrics unchanged. Weighted
volume changes from 4.721090554 to 4.542897224 for EMA (CPU 4.539075277), and from
35.188868156 to 33.814136505 for TM (CPU 33.803678103). Relative discrepancies shrink
from about 4% to 0.084% and 0.031%. These long-volume observations remain unassessed
under the general policy; they do not close broader simulator/materiality acceptance.

## Fill-gap percentile populations

CPU analysis includes a gap between every fill, so additional fills in one candle
contribute zero-length gaps. Shared GPU replay streams one positive gap per filled
candle. Its decoder restores the missing zero-gap multiplicity from the existing
fill count and histogram count, without retaining or transferring fill histories.
Boundary gaps and the time-weighted second moment are unchanged. Invalid or
insufficient fill counts propagate as errors rather than becoming a percentile.

Independent reducer cases enumerate the full timestamp population, including no
fills, one fill, several filled candles and repeated same-candle fills at one/five-
minute intervals. They verify unchanged input counts and time-weighted moments.
Real native EMA comparisons use seed 43, 2,880 bars, long/short/both sides and two/four
coins; all six p95 values agree with CPU under the strict case policy. Before the
population correction, GPU p95 exceeds CPU by one minute in each case. The four-coin
both-side case changes from two minutes to the CPU's one minute.

Additional CUDA service cases exercise both strategies with one/three coins, repeated
requests and changing batch shapes while CPU simulation APIs are forbidden. The
logarithmic upper-edge approximation for positive gaps remains. Separate two-day TM
long fixtures retain p95 differences (about 3% with two coins and 10% with four) in
the strict comparison; restored zeros do not establish universal percentile parity.
No general tolerance is widened and this population repair does not close the
remaining histogram or optimizer-materiality gate.

Reproduce the EMA observation and switch sides/coin count for the other cases:

```python
from optimization.gpu.parity import MetricTolerance
from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

inputs = fixture_inputs(build_parser().parse_args([
    "--fixture", "ema_anchor", "--sides", "both", "--coins", "4",
    "--bars", "2880", "--seed", "43",
]))
name = "fills_gap_p95_hours"
report = run_comparison(inputs, "binance", (name,),
                        {name: MetricTolerance(1e-9, 1e-7)}, gpu_engine="native")
print(report["status"], report["metrics"])
```

## Requested-metric cohort evidence

The [cohort tool](../gpu_cohort_benchmark.md#optional-metric-cohort-measurements)
now requests optional histories/reductions and scalar diagnostic limit metrics from
both simulators. Explicit policy files are resolved into the report; unknown policies
remain unassessed. BTC-denominated requests enable CPU BTC analysis. The core three
comparisons and default two-objective ADG/drawdown ranking remain available;
explicit objective vectors select additional ranking dimensions.

Four public seven-day, four-coin, both-side cohorts (both strategies, seeds 7/43,
sixteen candidates) retain identical core comparisons/rankings on the integrated
bounded-allowance/HSL simulator. The eleven-metric experiment has no flips for five
selected diagnostic checks, but HSL is disabled, its EMA tail is trivially zero, and
these thresholds do not establish all constraint decisions or extra-objective rankings.
EMA recovery/gap tails and weighted-volume residuals remain visible and unassessed.
Native warm cohort cost remains close to direct replay; no tuning optimum is inferred
from underfilled automatic batches with zero eligible samples. Requested-history
Torch and externally sampled whole-process/device memory are documented separately;
large suites and cold-native execution still require measurements.

A CUDA regression checks exact raw daily summaries, timestamps, fills and drawdowns
across a 16-versus-1+15 TM replay. Weighted ADG can differ by six float64 units with
those identical inputs. The tool permits at most eight float64 units in GPU-reference
checks, records every accepted discrepancy and reports exactness separately. It does
not admit float32 replay drift or widen CPU/GPU tolerances. Missing metric keys,
identity/liquidation changes and larger/non-finite disagreements still fail. This is a
bounded numerical-reference policy, not a general simulator-parity approval.

## Fill-gap resolution and optimizer materiality

Fill-gap counts now use 512 logarithmic bins rather than 128. The count buffer is
2 KiB per candidate, an increase of 1.5 KiB; this is not a total allocation bound.
Classification still uses float32, safe upper edges and a horizon-capped overflow
bin. Finite logarithmic edges are about 3.02% apart before integer rounding and the
float32 margin. That spacing is not a universal CPU/GPU error bound. Histories are
not retained or transferred, and the streamed squared-gap moment is unchanged.
Initial-entry intervals keep their separate 128-bin format and decoder.

The same public seven-day eleven-metric cohorts reproduce the effect. Candidate
parameters, input identities, all CPU values and all ten non-gap GPU metrics are
identical before/after refinement. The original ADG/drawdown ranking and selected
limit decisions are unchanged. Add minimizing fill-gap p95 as a third objective:

| Strategy / seed | Maximum p95 error, minutes, before → after | Three-objective fronts before → after |
| --- | ---: | --- |
| EMA / 7 | 3 → 0 | Six GPU members versus eleven CPU → identical eleven members |
| EMA / 43 | 2 → 0.1 | Five GPU members versus ten CPU → nine GPU versus ten CPU |
| TM / 7 | 0 → 0 | Identical fifteen-member fronts remain |
| TM / 43 | 0 → 0 | Identical twelve-member fronts remain |

The remaining EMA/43 p95 residual is 18.55 CPU versus 18.45 GPU minutes for one
candidate. The GPU front omits candidate 9 (zero-based) from the CPU front. Existing
trajectory and core-metric differences remain; this experiment does not certify
all objective rankings. The original tool covered only ADG/drawdown. Explicit objective
vectors now make these third-objective observations directly reproducible from its reports.

Warm direct/native cohort cost remains comparable, as recorded in the
[measurement recipe](../gpu_cohort_benchmark.md#fill-gap-resolution-experiment).
Native replay retains exact direct-GPU results except the already measured six-unit
float64 weighted-ADG reduction in TM/43. No CPU policy is widened. Automatic widths
have no eligible tuning samples in these underfilled cohorts. Regression tests cover
float32 bin boundaries, distinct 29/30/31-minute gaps, the actual EMA cohorts and
simultaneously requested 512-bin fill gaps/128-bin entry intervals. Native-only cases
forbid CPU simulations. Actual Metal execution and long-gap materiality remain open.

## Explicit objective-vector diagnostics

The cohort tool accepts repeatable `--objective METRIC min|max` options, requests
those metrics and reports the resolved vector, raw Pareto fronts, pair-order
changes and CPU regret at each GPU-selected axis extreme. Without those options,
ADG/max and drawdown/min remain the default. Aliases and directions are explicit;
missing/non-finite axes leave ranking unassessed. Constraint/limit checks remain
separate. These are offline observations, not production survivor selection.

Three/four-dimensional independent cases verify dominance, ties, directions and
axis regret; real CUDA cohorts check the full requested vector in recipe/results.
The public three-objective recipe in the [cohort guide](../gpu_cohort_benchmark.md)
replaces the separate post-processing used for the fill-gap experiment. Tooling
alone does not accept the remaining one-member EMA/43 front difference or other
unassessed metric discrepancies.

## Sustained completed-work service tuning

After fill-gap refinement, a GPU-only workload exercises the default 24-sample/30-second
evidence windows without shortening them. These timing observations precede the
shared-session-artifact master integration; they measure that recorded service workload. Prepare the public cohort's sixteen configurations
with seed 7, four coins, both sides and 10,080 minute bars; request the default three
metrics. Warm a sixteen-candidate direct GPU reference, then register the same prepared
dataset with `CudaBacktestService(batch_size=None, max_pending=512)` and the cohort's
500-million candidate-bar dispatch budget. Maintain 512 outstanding requests with
unique IDs, cycling the sixteen already resolved parameter sets; replenish after
completed futures until 12,288 EMA or 6,144 TM requests finish. Check each request/
dataset/liquidation identity and all metrics against its reference. Forbid the CPU
`execute_backtest`, `run_backtest` and Rust bundle APIs during this measurement.
The public `_cohort`, `_native_dataset` and `_observe_batches` helpers provide the
fixture, shared registration and controller observations used by this recipe.

Time from before dataset registration through service shutdown, after reference
warmup. Record `time.thread_time()` for the caller and `time.process_time()` for
whole-process CPU work. Do not infer caller cost from whole-process/proc PID time.

| Strategy | Requests | End-to-end rate | Completed-window median rate, width 64 → 128 | Caller CPU / elapsed seconds | Peak Torch allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA | 12,288 | 140.024/s | 86.570/s → 171.750/s | 0.606 / 87.756 | 2,667,008 |
| TM | 6,144 | 37.090/s | 23.953/s → 47.855/s | 0.331 / 165.652 | 2,488,320 |

Both completed windows meet the unchanged sample/time minima. All 18,432 results
match GPU references exactly and final Torch allocation is zero. Eligible samples
include completed work beyond the two consumed windows; pending windows do not
establish another accepted trial. Only widths 64/128 are measured. Caller p95 latency
is about 5.912/21.424 seconds with a full 512-request queue, so throughput does not
imply low per-request latency. Whole-process CPU time is 81.105/49.207 seconds and
includes execution-worker/driver/reducer work.

This proves useful adaptation and modest service-caller cost for repeated resolved
requests. It omits candidate generation, canonical scoring, evolution, Pareto writes,
scenario switches and cold compilation. Torch-only memory excludes driver/CuPy,
host RSS and disk. Full-optimizer CPU cost, large-suite resources, search quality and
an optimal dispatch width remain separate acceptance work.

## Portfolio raw strategy risk and account risk

The CPU export distinguishes actual portfolio strategy equity from account equity,
which is clamped at the liquidation floor. Shared GPU replay keeps those meanings
separate for `drawdown_worst_strategy_eq`,
`drawdown_worst_mean_1pct_strategy_eq` and `strategy_eq_underwater_pct_mean`.
A requested-only daily maximum-drawdown column includes raw realized net cashflows
and marked UPNL, including the actual terminal mark. Ordinary USD account risk
retains its original summaries. This correction does not establish raw growth,
weighted ratios or recovery parity.

`test_gpu_portfolio_equity_sampling.py` covers eight actual fill/mark liquidation
cases across both strategies and long/shared portfolios, independent two-/200-day
risk reductions and inactive padding, and eight directional ablation cases with
BTC risk disabled/enabled. Those 26 checks pass with the source-verified runtime.
Raw liquidation-risk absolute error is below 2.4e-7. Enabling the column preserves
all other ablation outputs exactly. Eight temporal replay checks additionally
preserve every output across chunk boundaries, including partial raw daily state.

The additional daily column costs four bytes per observed day per candidate and
participates in dispatch scratch admission. Its state, capture and allocation
compile away when unrequested. Missing a requested raw summary fails before metric
processing. Only compact metrics pass through the native optimizer boundary.
The final shock matrix passes all 48 cases: 24 portfolio raw-risk cases and the
24 existing side-equity cases after extracting their shared public fixture. Its
49 comparison reports cover 241 metric pairs, including a second account-only
request that verifies exact raw-capture on/off account output in the shared EMA
coin-HSL case. That case retains a pre-existing 2.985e-6 account drawdown residual
under a fixture-local 3.1e-6 absolute bound. The shared TM disabled-HSL fixture
retains its existing one-basis-point bound; corrected portfolio raw error is below
4.0e-5. The other 23 raw-risk cases and general parity policies stay strict.

Validation also passes 331 Rust tests (one ignored), default-feature compilation,
1,039 affected Python checks, four fused-routing/missing-summary checks, 183
replay/ablation/isolation controls and 36 CPU-forbidden optimizer lifecycle cases.
Six documentation checks pass. The compiled extension and all 923 source-manifest
files are verified. Current-head author/automatic review and CI remain required
before development integration; these checks do not close broader retirement gates.

## Work still required before legacy retirement

1. Finish the code-backed approximation inventory for the actual native shared-account
   path. In particular assess requested histogram tails, recovery trajectories, partial-day
   weighting and HSL observation timing using meaningful samples and canonical limit
   decisions. The all-157-metric audit has finite results in six long shock cases,
   but exposed missing HSL retrigger reporting and weighted-ratio/trajectory differences;
   lifecycle reporting has since been corrected, while long-cooldown/history-expiry
   duration differences and side-equity/weighted-ratio gaps remain. Finite output is
   not acceptance. Keep strict measurements visible; justify bounded
   accepted differences
   by optimization/risk materiality rather than widening gates to hide failures.
2. Consolidate representative specialized/general metric and risk evidence for both
   strategies, sides, one-way/hedged accounts, gaps/delisting and scenario suites.
   Removed shared-account loss envelopes are resolved behavior, not accepted decimal
   noise. Legacy directional-only approximations need not become native limitations.
3. Refresh the [cohort measurements](../gpu_cohort_benchmark.md) after the semantic
   replacements, and measure larger suite resource use. Distinguish cold compilation,
   warm useful throughput, completed-work tuning evidence and CPU orchestration cost.
   Architectural improvement is sufficient initially; a marginal speedup is not mandatory.
4. Preserve and verify CPU optimize, standalone backtests and plot/export dependency
   isolation while removing superseded validation workers, drift/checkpoint machinery
   and obsolete controls. Publish that cutover only after the earlier gates pass.
5. Reconcile delivered user/AI contracts, complete exact-head automatic review and CI,
   and audit every completion criterion. Development acceptance does not authorize
   integration into master.
