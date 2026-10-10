# Native GPU optimizer acceptance evidence

This is an evidence map for the [development contract](gpu_optimizer_contract.md),
not a simulator certification or permission to retire the existing `gpu` backend.
The replacement remains experimental. Test coverage establishes the stated cases;
it does not establish every feature combination or production search quality.
Dated decisions and prior measurements are preserved in the
[decision and progress log](gpu_optimizer_decision_log.md).

## Ownership and lifecycle foundation

| Requirement | Evidence | Scope and remaining limits |
| --- | --- | --- |
| CPU search, GPU execution | `gpu_native_backend.py` uses ask/tell and canonical CPU scoring; `gpu.native`, `executor`, `datasets` and `residency` own execution independently of evolution | The service is reusable outside optimization. EMA/TM multicoin adapters share a private strategy-neutral allocation/retry owner with explicit parameter layouts; public legacy names and preparation helpers remain. |
| CPU request preparation isolation | `test_native_preparation_isolation.py` | Fresh processes prepare real candidate requests for both strategies with coin HSL enabled/disabled and unstuck enabled. GPU runtime, replay, Torch/CuPy and legacy benchmark imports are forbidden, as are CPU simulations. This does not replace the CPU optimizer/backtest/plot cutover checks. |
| CPU optimize/backtest/plot isolation | `test_cpu_entrypoint_isolation.py` | Both strategies run actual CPU backtests and generate analysis/fill/equity/config exports plus PNG plots. DEAP and pymoo each start and resume two real CPU workers with GPU imports forbidden at interpreter startup. Tests keep the platform's default multiprocessing context. |
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
and a duration histogram, plus 56 bytes for the summary and scaled output. Native metric-service runners reduce each accepted physical replay before internal
sub-batches clone/join outputs. Direct diagnostic and retained legacy runners keep
raw output by default. Physical replay admission includes reduction scratch; compact
logical result assembly remains separate from this history envelope. This is a
history budget, not a complete device/host memory bound.

`test_gpu_recovery_resolution.py` covers strict plateaus, decreasing series, sparse
samples, terminal padding, one/five-minute intervals, independent CUDA-stream scratch,
and actual native dispatch starting under a two-candidate history budget. When
HSL-off work releases its allowance, later widths may grow within the effective
physical envelope. Twelve short native
replays compare all six distribution metrics with real CPU backtests across both
strategies, long/short/both sides and one/two coins. Native-only budget cases forbid
CPU simulation and preserve output identity across dispatches.

Two additional policy-switch regressions use both strategies, two coins/both sides,
512 bars, a 200,000-byte history allowance and a legally seeded retained factual
capacity of 512. HSL-off work allows a logical cohort of 24; HSL-on work splits it
into single-candidate physical replays. Earlier raw clones/joined histories coexist
with factual/recovery ownership at 248,388 bytes, before subsequent reduction
scratch. Compact native results instead join a 24-by-seven f32 tensor (672 bytes),
with metrics identical across requests and to a separate raw GPU diagnostic control.
This isolates result transport, not capacity learning, and does not claim complete
device memory fits the history allowance. Failed factual attempts remain rejected
before decoding/reduction; ambiguous or malformed compact payloads fail visibly.
Both regressions fail the preceding raw transport with the specific admission
violation and pass compact transport. All 35 recovery checks and 116 factual replay
cases pass. Affected callers additionally pass 53 device/transport checks, sixteen
expired-history checks and twelve real CLI/data/service controls, alongside 470
host/layout/service checks. Four optional-history checks assert effective physical
admission; their earlier fixed-width assertions fail on preceding code when
HSL-off work legally admits widths of six (volume) or five (weighted equity).
Production is unchanged in that test-only correction, and all checked sources
remain unchanged after each run. Independent review/CI gate development integration;
these bounded checks leave representative total-resource acceptance open.

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

## Controlled account-equity shape references

`test_gpu_equity_shape.py` exercises the actual `compute_objectives` entry point for
unweighted and weighted choppiness, jerkiness and exponential fit error. Fifteen
public equity-only curves cover empty/no-fill inputs, flat and varying curves,
partial days, UTC boundaries and suffixes. The shared JSON fixture supplies actual
Rust producer values for f64 and f32-quantized input curves, with initial, sparse
and absent fills. The existing Rust producer test rechecks all added values.

CPU and CUDA each check the six USD fields and the six corresponding BTC fields
with a constant BTC price of one. All 2,160 comparisons pass with `pytest.approx(abs=1e-10, rel=1e-12)`,
with explicit positive-infinity expectations. Rust rechecks those shape references
with the same finite tolerance and sentinel type/sign. This covers BTC
metric routing with an identical curve; it does not establish variable-price
conversion or full simulator parity. The three focused Python files pass 178
checks. Current-source Rust passes 332 tests with one existing ignore, default-feature
compilation succeeds and the rebuilt extension matches the source fingerprint.

A separate twenty-day public-fixture diagnostic uses `trailing_martingale`, short,
two coins, seed 43, coin HSL and unstuck. Set both sides' HSL red threshold to .002,
EMA span to 2.5 minutes and RED cooldown to five minutes. Multiply coin zero's
OHLC prices from minute 1,440 by .7 and coin one's from minute 1,800 by 1.3. Request
all 157 metrics so feature specialization matches the broader audit. CPU/GPU
equity timestamps agree, and GPU daily closes exactly match UTC closes from its
captured account curve. On original CPU, f32-quantized CPU and actual GPU curves,
the six reductions agree with the actual Rust producer within 1e-12 absolute.

The replay curves still differ by at most .961 account units, or .05869% relative.
Weighted jerkiness is .0012049606 on CPU and .0012711257 on GPU, a 5.205% relative
gap. Quantizing the CPU curve alone yields .0012049736, so output quantization
alone does not explain the gap. Small replay differences are amplified by this
second-derivative metric. This diagnosis identifies no shape-reduction or daily
capture defect. It does not accept the remaining replay gap: candidate ranking,
limit decisions and practical materiality need assessment before a general policy.

## Shape-objective cohort measurements

Use the public `gpu_cohort_benchmark` cohort and measurement helpers with both
strategies, seeds 43/47, short, two coins, 28,800 bars, coin HSL, unstuck, sixteen
candidates and one warm repeat. Apply the HSL settings and price shocks above to
both the prepared fixture and every candidate before packing. Keep the helper's
candidate sweep: initial/base quantity `.01 + index * .001` and EMA span
`5 + index * 1.25`. Request the three default metrics and all six USD shape fields;
rank ADG/max against weighted jerkiness/min. Run service widths 4, 16 and auto.

All four cohorts produce identical CPU/GPU fronts, zero pair-order changes on
both selected objectives, and zero CPU regret at each GPU-selected best candidate.
There are 120 candidate pairs per cohort. All 384 native-service results match
same-candidate direct GPU metrics and liquidation identity exactly.

Reusing those captured metric vectors, rank ADG against each of the six shape
fields independently. Twenty-two of the 24 fronts agree. EMA seed 43 adds member
14 on weighted fit error; EMA seed 47 omits member 10 on unweighted fit error.
Nine of 2,880 shape-axis pair relations change, including four TM choppiness
near-ties with CPU regret of 2.22e-16. The other axes have zero best-candidate CPU
regret. Weighted jerkiness error stays below 1.987% in these swept cohorts, but
one EMA choppiness comparison is 614.013 versus 871.233 (29.524% symmetric relative
error). Choppiness divides total variation by absolute net change and can amplify
small replay differences near flat endpoints. Preserve these failures and front
changes; good selected-objective ranking is not acceptance of arbitrary tight limits.

| Strategy / seed | CPU serial cohort seconds | GPU direct warm seconds | Native width 4 / 16 / auto warm seconds |
| --- | ---: | ---: | ---: |
| EMA / 43 | 2.250 | .875 | 2.835 / .904 / .885 |
| EMA / 47 | 2.581 | .883 | 2.877 / .883 / .891 |
| TM / 43 | 28.126 | 3.518 | 12.764 / 3.518 / 3.518 |
| TM / 47 | 28.216 | 3.520 | 12.773 / 3.523 / 3.524 |

CPU timing includes serial preparation and simulation; GPU/native rows measure
sixteen completed requests after warmup. Native first use follows direct GPU runs,
and caches are not cleared. These are cohort measurements, not full optimization
or cold compiler benchmarks. Auto dispatch starts at width 64 but only sixteen
requests are offered; no eligible tuning window completes. This does not prove an
optimal width. Reported Torch allocations exclude driver/CuPy, host RSS and disk.
General parity policies and broader acceptance gates remain unchanged.

## Prepared first-dispatch measurement

The service now discovers a prepared dataset's physical capacity on its owning
worker, then fills the initial dispatch from already queued compatible requests.
It preserves the initial ownership claim, excludes cancelled requests, adds no
arrival wait and keeps the prepared work/scratch ceiling. This removes a forced
single-candidate simulation before the queue can use the discovered capacity.

A paired CUDA measurement uses EMA Anchor, both sides, sixteen coins, 28,800 bars,
fixture seed 43, coin HSL and unstuck. Set both sides' RED threshold to .002, EMA
span to 2.5 minutes and cooldown to five minutes. Multiply coin zero's OHLC from
bar 8,640 by .7 and coin one's from bar 17,280 by 1.3. Submit 64 candidates with
both sides' base quantity `.01 + index * .0005`. Request weighted raw ADG, weighted
account Calmar, raw recovery p95/drawdown, weighted long ADG and short MDG per
exposure, weighted volume, time in RED, joint portfolio EMA tail and completion
ratio. Use auto width, pending capacity 128 and zero accumulation delay.

For each version, evaluate the cohort twice in one service. Hold first preparation
until all 64 requests are queued, then release it; this isolates prepared-capacity
dispatch from differences in arrival timing. Use separate processes with the same
input/parameter identities and verified Rust runtime. The baseline changes only
the executor back to the development source before this scheduling change. Shader
caches remain populated; imports and fixture generation are outside the timer.

| Executor | First cohort seconds | Second cohort seconds | Actual dispatch counts |
| --- | ---: | ---: | --- |
| Before | 119.959 | 54.764 | 1, 63, 64 |
| Prepared first dispatch | 55.956 | 54.801 | 64, 64 |

All ten metric values and liquidation status match exactly for all 64 candidates
between versions and between repeats. The first cohort is 53.35% shorter in this
controlled comparison; warm throughput is essentially unchanged. The new service
records one eligible warm full-width timing sample, whereas the baseline records
none. Neither completes a tuning window. This is one saturated service workload,
not a cold-cache or full-optimizer speedup claim. Demand-limited tuning and broader
resource/performance acceptance remain open.

## Raw strategy recovery observations

Rust's strategy equity is starting balance plus factual net realized PnL and
unrealized PnL. Account liquidation clamping must not replace that curve when
sampling the six strategy recovery distribution metrics. Shared EMA/TM single-
and dual-side replay now stores raw strategy equity in the existing recovery
buffer, including when no weighted raw metric is requested. Trading decisions,
account metrics, memory layout and the separate streaming maximum-recovery
summary are unchanged.

The sixteen cases in `test_gpu_recovery_resolution.py` exercise both strategies,
long/shared sides, mark/market-panic liquidation and recovery-only/weighted-raw
capture. Each runs actual Rust first, then prohibits CPU simulation during native
execution. It checks every observed recovery sample against Rust's fourth equity
column and all six metrics against an independent strict time-to-exceed reference.
All sixteen fail on the original producer: an account floor of 50 replaces raw
terminal strategy equity of -2201.6. The rebuilt correction passes the complete
194-case recovery, raw growth, weighted capture and HSL ordering suite. Rust tests
pass 332 cases with one ignored; default-feature compilation and six documentation
checks also pass.

Four additional actual optimizer cases use the recovery p95 objective with both
strategies, standalone/screened suites and automatic dispatch width. They verify
prompt result/Pareto persistence, interruption and resume while prohibiting CPU
simulation. Seven disabled-HSL policy/specialization controls also pass.

The corrected source also completes the six twenty-day, two-coin, seed-43 shock
recipes above with all 157 requested metrics present and finite. All 151
non-recovery values per case match the preceding source exactly. The six recovery
fields remain separately measured; finite output alone is not parity acceptance.

This fixes an input definition, not every float32 trajectory difference. Four
aligned twenty-day curve diagnostics preserve the same observations and requested
non-recovery metrics. In one case recovery p95 moves from 13.425105 to 12.422327
days against CPU's 12.468854. Quantizing that CPU curve to float32 alone gives
13.434931 days: strict ordering near flat samples can amplify small rounding.
Another case retains 90 versus 91 fills and a 3.366703 versus 2.608437-day recovery
p95. Keep those residuals visible and assess optimizer materiality separately;
general standalone-tool tolerances are unchanged.

## Current native approximation inventory

This inventory follows the shared-account execution used by the native service.
It distinguishes approximate values from corrected input definitions; it does not
approve every listed metric as an objective or tight limit. Eligibility is defined
by `metric_registry.py` and `metrics.SUPPORTED_METRICS`, including canonical aliases.
An existing reducer or output field does not establish native support.

| Surface | Current calculation and evidence | Acceptance question |
| --- | --- | --- |
| Trading trajectories | Shared EMA/TM kernels use float32 strategy/account state. Native and specialized/general tests cover the stated topology, order, risk and replay cases. | Threshold rounding can alter later fills; assess risk, feasibility and selected configs rather than demand identical long trajectories. |
| Completed HSL episodes | The native worker reconstructs current scope episodes from retained simulator fill facts and causal closes, with fresh current budgets and an explicitly disposable cutoff memo. The component, temporal, matched-cohort and native lifecycle checks below distinguish this from retained observation replay, which remains on the legacy screening route. | The material HSL cohort outlier is resolved and all 42 wider native lifecycle/loss cases pass; general numerical and risk acceptance still requires the stated representative gates. |
| Active coin HSL without retained fills | Factual native reconstruction evaluates an exposed current position from its actual endpoint when clipped facts no longer establish historical exposure. The earlier 82 empty-history controls (66 probes and 16 real caller cases) validate the retained legacy observation route; factual pair/scope endpoint controls and native replay/temporal checks validate the replacement. | Current-position loss protection must remain available after expiry. Keep these route-specific controls distinct, and assess downstream float32 trajectories rather than treating component equality as universal simulator acceptance. |
| HSL EMA drawdown tails | Requested native replay captures eligible long/short/portfolio observations in bounded device histories and reduces their actual largest `max(floor(count / 100), 1)` values. The legacy observation engine retains 32 logarithmic bins. `test_gpu_portfolio_ema_tail.py` covers portfolio ownership; `test_gpu_ema_tail_resolution.py` adds independent sorted-reference, partial-clock, retry and compact sub-batch controls. | Requested capture, reporting scopes and clocks pass the focused CUDA controls and matched scope comparisons below. Exact selection on the observed GPU curve does not remove float32 trajectory differences or establish every tail objective/limit policy. |
| Side raw daily drawdown tails | Retain a bounded sorted list of daily maxima. Capacity covers the worst floor(1%) of the prepared UTC horizon and belongs to shader-cache identity. Current-day queries do not flush or mutate replay state. `test_gpu_daily_tail.py` covers selection, horizon bounds and CUDA replay isolation. | Selection is exact on the GPU's observed float32 curve; CPU/GPU curves may still differ. Full replay and matched kernel evidence are recorded below; this does not accept unrelated trajectory or HSL reconstruction differences. |
| Fill-gap percentiles | `_fill_gap_metrics` uses 512 logarithmic positive-gap bins, upper-edge decoding, actual boundary gaps and restored same-candle zero multiplicity. Multiplicity/reducer tests and the three-objective cohorts above expose the remaining residuals. | The same gap population can still be quantized; upper edges are not a universal CPU/GPU trajectory error bound. |
| Initial-entry interval percentiles | `_entry_interval_metrics` retains 128 bins for TM; streamed mean/maximum are separate. `test_gpu_metrics.py` checks totals, malformed counts, upper-edge percentiles and EMA's canonical zero case. | Median/p95/p99 are approximations even when mean/maximum agree; representative selection/limit materiality remains unassessed. |
| Recovery, weighted volume and weighted equity | Requested per-step GPU histories feed strict recovery and canonical suffix/daily reducers. Recovery-resolution, weighted-volume and weighted-equity tests isolate input definitions, cutoffs, optional capture and dispatch bounds. | Input-definition repairs do not remove float32 curve/trajectory sensitivity, especially strict recovery ordering near plateaus. Full histories remain on device. |
| Legacy-only fallbacks | Directional single-coin volume/tail helpers and daily peak-recovery helpers still exist. The native service uses shared replay, requires the observed portfolio tail, and rejects exact-only metric names. | Do not count an unused fallback as accepted native behavior or retain the old optimizer merely to preserve its approximations. |

Per-metric policy and measured feasibility/selection evidence remain required for
acceptance. The all-supported-metric audit checks presence and finite/sentinel
handling; its undefined policies are explicitly unassessed. Neither this table nor
finite output alone closes the numerical gate.

## Native EMA tail and reporting qualification

Implementation revision `b2e709a878` passes 81 focused checks with CUDA enabled,
with no failures, errors or skips. Twelve separate host reducer/scalar controls
also pass. The rebuilt Rust extension has source fingerprint
`1eb54145fa425dd86b15aedddddc80b28cdd0b1229c7c8edfb53bc0c6a15c6cf`;
341 Rust tests pass, with one existing ignore, and default-feature compilation passes.
The checks cover exact sorted selection, scope ownership, cooldown reporting,
partial and temporal replay, rejected retries, compact joining, capture admission,
disabled consumers and native CLI scoring/resumption without CPU simulation.

The eight terminal-cause comparisons include both strategies, with long-only and
both-side mark-driven and panic-fill liquidation. Their six newly covered EMA fields use
fixture-local absolute `1e-8` plus relative `2e-6` allowances, informed by the
measured rounding at panic fills. Existing lifecycle and side-equity assertions
remain unchanged. This does not widen the standalone tool's default policies.

Four standalone comparisons use public synthetic inputs: both sides, two coins,
3,000 one-minute bars, seed 43, RED threshold `.002`, EMA span `2.5` minutes,
one-day lookback and `10000`-minute cooldown, with price shocks
`(0, 1500, .7)` and `(1, 1800, 1.3)`.

| Strategy | Signal scope | Largest absolute EMA-maximum error | Largest absolute EMA-tail error |
| --- | --- | ---: | ---: |
| EMA Anchor | Unified | 7.488e-7 | 4.078e-7 |
| Trailing Martingale | Unified | 2.712e-8 | 2.192e-8 |
| EMA Anchor | Pside | 2.872e-7 | 6.022e-8 |
| Trailing Martingale | Pside | 5.018e-8 | 1.186e-8 |

Unified long/short reports are exactly zero in both engines. Pside reports retain
their own signals and a separately observed portfolio tail. These residuals are
accepted for the stated scope fixtures under the practical F32 contract. The
machine-readable comparison keeps all six fields policy-unassessed because no
general per-metric tolerances were supplied. Matched objective/limit consequences
remain a separate numerical gate; these observations do not certify arbitrary
tight limits, identical trajectories, general speedup or optimizer retirement.

Reproduce the capture and reporting controls with the source-verified extension:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/optimization/test_gpu_ema_tail_resolution.py \
  tests/optimization/test_gpu_hsl_ordering.py::test_liquidation_retains_elapsed_red_interval \
  tests/optimization/test_native_backend_cuda.py::test_native_hsl_ema_tail_cli_scores_limits_and_resumes_without_cpu
```

The 81-check qualification additionally includes the affected launch-option,
diagnostic-layout, weighted-capture, recovery and retained legacy-tail controls.

## Exact side raw daily tails

The requested side metric retains the largest daily maxima needed for the worst
`max(floor(observed_days / 100), 1)` samples. Its compile-time capacity is the
next power of two covering the prepared UTC calendar horizon. Current-day queries
are pure, and observed day counts govern early truncation. Inactive tail state is
compiled away; no equity history is transferred or CPU simulation repeated.

`test_gpu_daily_tail.py` passes all 24 checks with CUDA available: capacity boundaries,
invalid direct capacities, same-bin values, intraday and skipped-day samples, unfinished days, early
truncation, repeated queries and snapshots, plus both strategies' long/short/fused
replay ablation and capacity equivalence. TM temporal replay matches unchunked
outputs exactly. Independent float32 daily references use a 1e-7 absolute bound.

A matched whole-kernel control uses the public `_multicoin_exposure_fixture` with
long exposure, two coins, 20,167 fifteen-minute bars and 16 identical candidates.
The close curves are `100 - arange(n) / n * 25` and
`120 - arange(n) / n * 30`; other inputs/parameters are the fixture defaults.
Raw side drawdown/tail and portfolio raw daily risk are requested. The prepared
horizon is 211 UTC days, giving capacity two. Swap only the shared HSL-common
source between the histogram implementation and this implementation, clear the
shader-library cache, then measure first use and five synchronized warm replays.
Every non-tail output is identical, and all repeats within each design agree.
The separate portfolio daily summaries provide a CPU-sort reference on the same
observed GPU curve, avoiding a CPU/GPU trajectory comparison.

| Strategy | Histogram tail error | Exact tail error | Warm candidates/s, histogram → exact | CUDA local bytes, histogram → exact | Registers |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA anchor | 8.97e-6 | 4.55e-13 | 47.6 → 46.7 | 1,472 → 1,232 | 154 → 154 |
| Trailing martingale | 0.01427 | 7.45e-9 | 30.4 → 30.7 | 2,896 → 2,656 | 232 → 232 |

Errors are absolute drawdown fractions. First-use replay times are 12.0/10.2
seconds for EMA and 59.9/57.0 seconds for TM (histogram/exact), including
compilation and dispatch. This small isolated single-side control establishes
reduction accuracy and lower local storage for its capacity; it does not establish
optimizer throughput, a general speedup, or total GPU memory bounds. Larger date
ranges require larger capacity. CPU/GPU curve differences, minute EMA-tail bins
and HSL retained-fill reconstruction remain separate acceptance work.

## Shared-account TM temporal replay

Long CUDA multicoin TM histories now retain both side states, shared accounting
and requested metric accumulators between synchronized temporal dispatches. The
work envelope counts both sides. Histories above 8,192 candles get interrupt
boundaries even below the work cap; existing Apple activation remains unchanged.
Dispatch-size caps do not guarantee a wall-clock bound for every strategy workload.

Run the actual CUDA continuity/failure checks and proportional replay/CLI controls:

```bash
PYTHONPATH=src pytest tests/optimization/test_gpu_tm_fused_temporal.py -q
PYTHONPATH=src pytest tests/optimization/test_gpu_daily_tail.py -q \
  -k 'long_replay and trailing_martingale'
PYTHONPATH=src pytest tests/optimization/test_gpu_cuda.py -q \
  -k 'temporal_batch_increase and 3 and not 28 and not 64'
PYTHONPATH=src pytest tests/optimization/test_native_backend_cuda.py -q \
  -k portfolio_ema_cli
```

The continuity tests use three coins, both sides, all requested optional buffer
families, three HSL modes, hedged/one-way accounts and unequal candidate endpoints.
Raw tensors match unchunked replay exactly across repeated chunk sizes. A real
prepared-service future propagates an interrupt after its first completed chunk;
discarded partial state is reset on the next replay. Fatal-marker continuation
coverage additionally verifies that later chunks cannot erase producer failure.

A matched control uses `gpu_parity.fixture_inputs`: TM, both sides, three coins,
16,385 minute bars, seed 43, coin HSL and unstuck. HSL threshold/span/cooldown/history
are 0.002/2.5/10000/one day; shocks multiply COIN00 by 0.7 from bar 1500 and COIN01
by 1.3 from bar 1800. Eight candidates vary long initial quantity as
`0.01 + 0.004 * i`, retaining other fixture parameters. Request weighted raw/account
growth, raw daily growth/risk, long raw/EMA tails, portfolio EMA tail, BTC risk and
equity/balance difference, entry interval, weighted volume and recovery. Compare
the fused runner with no temporal budget and budgets of
`8 * 3 * 2 * chunk_bars`. Synchronize around each replay; measure first use and
three warm repeats. All returned raw outputs agree exactly in every repeat.

| History chunk | Warm median, seconds | Largest dispatch in final repeat, seconds | Dispatches | State bytes/candidate | CUDA local bytes | Registers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchunked | 5.731 | Whole replay | 1 | — | 9,280 | 255 |
| 8,192 bars | 5.768 | 2.976 | 2 | 7,472 | 9,312 | 255 |
| 1,024 bars | 5.780 | 0.388 | 16 | 7,472 | 9,312 | 255 |

Measured on an NVIDIA GeForce RTX 3070 Ti Laptop GPU. This small fixed replay
control establishes state continuity and its measured overhead, not optimizer
throughput, optimal chunk size or a general elapsed-time guarantee. Duration-based
dispatch tuning and representative larger-suite resource acceptance remain open.

## Work still required before legacy retirement

A larger native-suite pilot uses 25 coins, both sides, 11,520 minute bars,
eight candidates and unified HSL (`gpu-service-benchmark`, seed seven, one-day
lookback and threshold 0.99). Its isolated width-one reference phase does not
complete a round after more than 35 minutes and is stopped. No completed-phase
resource, throughput or parity acceptance follows. The active work also prevents
shutdown completing within six minutes of an interrupt. The tool cancels queued
requests but joins its running worker; it does not pass an interrupt callback.
Production optimization supplies its own callback, while an active CUDA kernel
still cannot observe that callback until it returns. Keep these caller and kernel
limits distinct. Profile individual completions, reconstruction/retry costs and
launch geometry before another long all-width comparison; larger busy-HSL and
long-interruption acceptance remain open.

1. Finish the code-backed approximation inventory for the actual native shared-account
   path. In particular assess requested histogram tails, recovery trajectories, partial-day
   weighting and HSL observation timing using meaningful samples and canonical limit
   decisions. A refreshed all-157-metric audit has no missing or non-finite outputs in
   six long shock cases after the lifecycle, side-equity and weighted-ratio corrections.
   Finite output is not acceptance. An independent recovery investigation also exposed
   account-equity sampling where raw strategy equity is required; the corrected
   producer now passes the liquidation regressions above. Assess its residual numerical
   differences separately. Long-cooldown/history-
   expiry duration materiality and histogram/trajectory differences still need assessment.
   Keep strict measurements visible; justify bounded
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


## Completed HSL history expiry

`test_gpu_hsl_window_expiry.py` separates a completed episode's observed peak from
an active position's estimated entry loss. Integer-valued reference streams cover
inclusive expiry at 10, 65 and 1440 minutes, fractional smoothing spans, both restart
policies and positive historical peaks. Active and terminal loss-estimate controls
use the real Rust snapshot evaluator. Same-time flat observations and budget changes
compare cached replay with forced scalar rebuilding and the Rust incomplete-episode
entry reference. An entry estimate has a first-observation lifetime; expiration must
not create another one for a completed flat episode.

The original source fails 24 rolling-controller and four real-backtest regressions;
16 related controls pass. The initial correction exposed four cache-dependent action
cases. The final source passes all 48 new cases and 158 preceding rolling/controller
checks. This evidence is separate from broader replay, temporal and optimizer checks.

The broader replay run records 197 passes, 13 Apple-only skips and 51 failures in
old episode-boundary reporting assertions. Three original-controller probes reproduce
one failure from each family. An earlier observed RED remains a historical trigger
when current permission becomes GREEN, a scope stays held, or a terminal budget is
unusable. Repair those counter expectations while retaining permission/cooldown,
scope-flatness and invalid-budget checks; the unusable-budget probe additionally
checks that the controller's prior permission is unchanged. All 84 repaired probes
pass. Sixteen specialized/full-capacity and temporal-state controls and four actual
CPU-forbidden optimizer interruption/resume cases also pass. The six twenty-day
shock audits return all 157 requested metrics with no missing/non-finite outputs;
every GPU value is unchanged from the preceding raw-recovery source. Undefined
comparison policies remain unassessed. Six documentation tests pass.

Four actual two-coin, 3000-minute, seed-43 comparisons use both sides, threshold 0.002,
EMA span 2.5, cooldown 10000 minutes and one-day retained history. Coin/pside duration
maxima match the CPU at 1439/1415 minutes for EMA and 1438/1408 for TM; RED time and
lifecycle checks remain strict. Small ADG/fill residuals use fixture-local bounds.
EMA retains two fewer fills over CPU counts 1942 and 1902, bounded by one fill per day;
TM retains the existing local 0.1% fill-rate bound. General tool policy is unchanged.

A different unified EMA fixture exposes reconstruction, rather than decimal error.
Rust's actual retained-fill evaluator matches the CPU backtest and remains halted
through minute 2995, retiring cooldown at 2996. Applying the same Rust controller to
its frozen terminal observations retires it at 2955. After clipping the opening fills,
the retained-fill evaluator supplies an estimated opening basis and entry peak. A
same-observation controller test cannot certify that both simulators produce the same
reconstructed observations. Unified long-history materiality remains separate
acceptance work; this correction does not certify arbitrary tight HSL constraints.

The final ten-recipe history measurement has exact duration and RED-time agreement
for all eight coin/pside cases. Unified EMA remains 1441/1400 minutes (CPU/GPU),
with ADG difference 0.00003638, worst drawdown difference 0.00001038 and 16 more GPU
fills. Unified TM improves from 1385/1440 to 1385/1386 minutes, with ADG difference
0.00003011, worst drawdown difference 0.00000555 and 16 more GPU fills. Both use
the same public recipe and strict measurement policies above; their mismatches are
reported without widening general policy. Same-observation correctness does not
approve the separate unified reconstruction/selection difference.

## Unified HSL cohort materiality

The reproducible command in [the cohort guide](../gpu_cohort_benchmark.md) measures
64 matched candidates: EMA/TM, seeds 7/43, 16 candidates each, two coins and both
sides, 3000 minutes, threshold/span/cooldown 0.002/2.5/10000, one-day history and
price shocks at bars 1500/1800. This is a fixed candidate cohort, not an independent
evolutionary search. Native service results agree with direct GPU under the tool's
reported float64 reduction policy. Undefined CPU/GPU metric policies remain unassessed.

No candidate changes feasibility at the authored diagnostic limits (ADG >= 0,
worst drawdown <= 0.005, RED time <= 0.5, completion >= 0.99), and all four
GPU-selected ADG extremes have zero CPU ADG regret. Nevertheless, adding RED time
as a third objective changes Pareto membership in all four cohorts. ADG/drawdown
fronts agree in three cohorts; TM seed 43 loses CPU-front candidate 0. Maximum
absolute discrepancies include 0.209738 RED-time fraction, 223 minutes in maximum
halt duration, 0.001150 ADG and 0.000974 worst drawdown. These are not all decimal
noise, and generous limit agreement does not certify arbitrary tighter limits.

Original-controller controls replace only `mps_hsl.metal` from the preceding
development source, verify the assembled shader and candidate parameter hashes,
and reuse the same CPU reference metrics. All four three-objective fronts already
differ before the correction. Mean errors improve in three cohorts, but EMA seed 43
has larger discrepancies afterward: candidate 15 now has two observed stops rather
than one. This new evidence was investigated before merging the correction.

For that candidate, original/current GPU runs first panic and flatten at minute 1526.
The original GPU restarts at 2023, the corrected GPU at 1919, and the CPU at 2050.
Replaying the GPU's identical factual terminal observations with the source-verified
Rust controller is halted at 1918 and normal at 1919. The subsequent exposure stops
again at 2252. Thus the controller correction matches Rust on the same observations;
the retained-fill reconstruction gap can still change actual trading and selection.
Do not restore a synthetic completed-entry peak to mask that gap or approve legacy
retirement from the narrower controller tests. A producer solution must preserve the
shared Rust reconstruction contract, including cache-loss/rebuild equivalence.

Warm native measurements for these 16-candidate cohorts span approximately 24–43
candidates/second. Width 16 and automatic mode are similar in this underfilled
workload; there are no completed tuning windows, so the final automatic width does
not establish an optimum. Compilation is already warm, and CPU timing is serial
preparation plus reference simulation, not CPU optimizer throughput.

The shared stress CLI reproduces all nine CPU/GPU metrics, candidate parameter
fingerprints and ranking results for all 64 measured candidates exactly; native
width-16 and automatic runs both retain exact direct-GPU values in these cases.
All 49 new recipe/validation/provenance checks pass. A wider focused run has one
remaining pre-existing passive-TM fixture failure: its both-side ADG comparison
exceeds the local 0.1% trajectory guard (CPU 0.02232715, GPU 0.02235809). Untouched
target-branch tool sources reproduce those values with identical default fixture
arrays and evaluation identity. Keep that measured discrepancy visible; this
tooling change does not widen its assertion or the general comparison policy.

## Underfilled execution tuning

The asynchronous service now records successful warm dispatches by their actual
candidate count, including partial cohorts. Each actual shape's first use remains
excluded. Dataset-owned windows retain the normal 24-sample, 30-second minimum,
median rate, cooldown and rollback policy. When demand or memory headroom prevents
growth, the service may probe a smaller width using subsequent real requests.
Fixed widths and the retained screening/validation tuner's full-batch policy keep
their existing behavior. No calibration simulations or CPU validation are added.

With a verified extension and CUDA device, run:

```bash
PYTHONPATH=src python -m pytest \
  tests/optimization/test_gpu_execution_tuning.py \
  tests/optimization/test_gpu_autotune.py \
  tests/optimization/test_gpu_executor.py \
  tests/optimization/test_gpu_coalescing.py \
  tests/optimization/test_native_pipeline.py \
  tests/optimization/test_gpu_execution_tuning_cuda.py -q -o addopts=
```

All 151 cases pass with one existing skip. Host controls cover the production
evidence thresholds, cold partial shapes, independent datasets, invalid timings,
shutdown, blocked growth and slower-trial rollback. Two added CUDA cases replay
36 requests per strategy in three-candidate cohorts against a fixed-width control.
They accelerate only the evidence thresholds, observe smaller-width trials,
count every request once and preserve all metric values exactly across both
dataset identities. The other six CUDA cases retain adaptive-width equivalence
and physical dispatch bounds. Eight existing native CLI controls preserve
standalone/suite results and interruption/resumption for both strategies with
CPU simulations and worker pools forbidden. These are correctness controls,
not evidence of an optimal width, representative throughput improvement or a
duration guarantee. Representative tuning quality and resource acceptance remain open.

The public cohort benchmark's evidence adapter uses the same actual-count,
warm-shape eligibility rules. Its cumulative sample/time and consumed-window
reports include partial cohorts without counting their first cold use or invalid
observations. All 141 cohort-reporting, execution-policy, legacy-tuner and
documentation checks pass with the current extension, one existing skip and three
cohort device cases deselected. This reporting suite is separate from the eight
service CUDA controls above.

## Unstuck EMA consumer specialization

Multicoin EMA/TM execution proves the effective enabled/gating flags across every
packed candidate, coin and active side after float32 conversion. Finite coin flags
replace candidate flags; nonfinite coin flags inherit them. One consuming combination
keeps EMA state for the entire dispatch. Unknown base flags conservatively retain it.
The proof is independent of search and scheduling, uses immutable host override views,
and performs no device readback. Its result participates in compilation and temporal
state cache identity.

When no combination consumes unstuck EMA, compilation removes the band, initialization,
updates and gated-selection branches. Ordinary strategy EMAs, ungated unstuck selection,
loss budgets and histories keep their existing contracts. The single-coin kernels remain
general. This is not full unstuck or inactive-side ablation.

Run with the current verified extension and a GPU:

```bash
PYTHONPATH=src python -m pytest \
  tests/optimization/test_gpu_unstuck_ema_specialization.py -q -o addopts=
```

Ten host-proof tests and thirty real CUDA cases
pass. Both strategies and long/short/fused paths cover disabled, ungated, gated and
coin-overridden consumers, actual fills, weighted metrics and recovery tapes. Every
returned finite value agrees exactly; intentional NaN masks also agree. TM temporal
chunks and repeated layout/batch changes preserve all outputs and reduce saved state
when specialized. Six mixed-candidate controls retain the EMA layout when only one
candidate enables its gate, then restore the compact layout on the next dispatch.

Rust validation passes 332 tests with one existing ignore and default-feature compilation.
All 38 coupling, CPU preparation-isolation and documentation checks pass with the
current extension. The broader source-verified suite passes 172 cases: finite unstuck
history and exact Rust controls, HSL empty-history helper/native callers, fused temporal
replay, and eight actual standalone/suite optimizer CLI interruption/resume cases with
CPU backtests and worker pools forbidden. These cases do not establish general simulator
parity or optimizer
throughput, and do not supersede the remaining replacement-acceptance gates.

A controlled RTX 3070 Ti Laptop measurement uses `tools.gpu_parity.fixture_inputs`:
each strategy, three coins, both sides, 2880 minute bars, seed 43, disabled HSL and
enabled ungated unstuck. Request weighted raw ADG, raw worst drawdown, recovery p95
and weighted volume. Replay 64 identical candidate rows directly, bypassing search
and deduplication. Warm both variants before five alternating measured runs each;
TM uses a 720-bar work envelope. Set `runner.unstuck_ema_specialization=False` for
the general control and `True` for the proved specialization. Read CUDA attributes
from the selected library's replay kernel and temporal bytes from the runner's
queried ABI. All returned outputs match exactly across every measured run; each
candidate has 4442 EMA or 28726 TM fills.

| Strategy | Measure | General | Specialized |
| --- | --- | ---: | ---: |
| EMA | Warm median replay kernel seconds | 0.203985 | 0.192849 |
| EMA | Compiler local bytes / registers | 7248 / 202 | 6976 / 198 |
| TM | Warm median replay kernel seconds | 0.588270 | 0.572162 |
| TM | Compiler local bytes / registers | 7936 / 255 | 7904 / 255 |
| TM | Temporal state bytes per candidate | 6000 | 5744 |

These are isolated warm replay/storage controls with identical candidate rows,
not search throughput, a cold-start benchmark, total VRAM/RAM bounds or an optimal
batch/chunk size. Compiler attributes and small timing gains are workload/device
observations, not requirements imposed on other architectures.

## Retained factual history component — development evidence

`test_gpu_hsl_history.py` compiles the Rust-owned `mps_hsl_history.metal` component
and compares its pair histories with the source-verified Rust `hsl_history`
reference. All 22 CUDA cases pass: four parameterized batches cover 996 histories,
and eighteen malformed-input cases cover quantities, prices, cashflows, current
facts and ring metadata. Generated tapes use seed 7121, 64 executions, five quantity
steps from 1e-6 to 100, clipped prefixes, wrapped storage and same-time cohorts.
Targeted cases cover empty/held histories, missing closes, reduction-only prefixes,
local quantity repair, full closes, reopenings, a remaining lot after large
inventory, and small fees surviving large cancelling cashflows.

Inventory and sample flatness must agree exactly. Quantity error remains below a
quarter of the exchange quantum. Basis, cashflow and UPNL checks use their contributing
magnitudes to distinguish float32 conditioning from a near-zero result; exact cashflow
checks additionally cover the cancellation fixture. These are component-test bounds,
not acceptance policies for optimizer objectives, feasibility or candidate ranking.
Rust reference tests pass 332 cases with one existing ignore, and default-feature
compilation passes.

The reconstruction component does not yet supply native HSL evaluation.
Opt-in native factual capture is being developed separately below; neither
component coverage nor capture proves aggregate controller observations or native
trading/Pareto parity. Bounded factual storage,
actual scope episode boundaries, cache-loss and checkpoint identity, full caller
coverage and the matched HSL candidate cohorts remain open. Existing native retained-
fill discrepancies remain classified as material until that integration is verified.


## Native factual capture and scope selection — work in progress

Native multicoin EMA/TM fill paths now support internal, opt-in factual capture.
Each worker retains signed quantity, actual execution price, gross PnL, fee,
actual post-fill position and global sequence. Its existing two-slot header also
retains the actual current size and basis, including when close accounting precedes
the caller's position mutation. Consecutive same-pair, same-direction,
same-minute executions may share a record; a reduction never merges through an
actual flat. Aggregate scopes borrow pair facts. Normal result payloads remain
metrics; full factual scratch readback belongs only to the tests.

A distinct overflow marker rejects the whole result. The worker may double its
factual capacity and repeat GPU work only within the existing one-candidate scratch
budget, checking interruption before growth and preserving the learned capacity.
Malformed facts and other fatal candidate errors must take precedence over capacity
recovery. No truncated history or partial result is admitted; no CPU backtest is
used for recovery. Capacity is a runtime storage control; only feature enablement
participates in kernel compilation. This avoids recompiling on each growth step.

The scope selector reverse-walks globally ordered factual post-fill positions on
each exchange quantity quantum. Exposed scopes retain the latest actual flat prefix;
flat scopes retain the preceding one to preserve the just-completed episode.
Its consumed sequence distinguishes a flatten from a same-minute reopen. A clipped
exposed prefix receives no invented flat seed. This selector is not yet called by
native HSL evaluation.

Thirteen real CUDA scope-selection cases and the preceding 47 pair-history/writer
cases pass. Scope cases cover coin/side/portfolio selection, same-minute reopening,
multiple completed episodes, fractional quantities, clipped exposure and invalid
current inputs. These are authored cutoff/transport controls, not full Rust
scope/controller parity.

With the current rebuilt extension, all 110 checks pass: 100 actual CUDA controls
and ten host execution-policy cases. Forty native caller cases cover EMA/TM,
long/short/fused paths and coin/side/unified modes. Capture preserves every returned
metric exactly, including NaN masks. Global fill ordinals remain complete; physical
candidate partitioning includes factual scratch. Full and temporally chunked tapes
agree exactly. Capacity growth reuses compiled identity and warm storage; cancellation
propagates before growth. Later temporal chunks preserve overflow, and a fresh retry
restores complete factual chronology and unchanged metrics. The host attempts are
explicit fakes around production capacity and decoder policy, including mixed fatal
and recoverable markers. No native optimizer parity policy is widened.

Rust passes 332 tests with one existing ignore and default-feature compilation;
319 host/preparation/service/documentation checks pass with 40 device-named cases
deselected. The preceding capture build also passes four coupled CUDA request
comparisons and all 52 native CUDA CLI checks with CPU backtests forbidden.
Controller integration, cache-loss/checkpoint equivalence, resource acceptance and
matched candidate-front comparisons remain required.

The expanded endpoint build passes all 157 checks: 147 actual CUDA and ten host
policy cases. Twelve additional endpoint probes reject malformed basis/flat
inputs and preserve factual endpoints after failure. All forty native capture
cases now inspect actual size and weighted basis; nine temporal controls also
require exact header equivalence across chunks. These fields fit in the existing
64-byte header, with 56 bytes used. Rust still passes 332 tests with one existing
ignore, including default-feature test compilation.

## Reconstructed scope/controller component — development evidence

`mps_hsl_scope.metal` composes a fresh scope from causally clipped pair histories,
current positions and aligned minute marks. It merges factual global execution
order, centers cashflow against a common current prefix, preserves exact consumed
flat boundaries, and evaluates terminal accounting before resetting each episode.
Estimated opening references do not invent an EMA observation. Current positions
without a supported historical opening discard idle observations from the current
episode while retaining earlier completed episodes.

The composer consumes the same observation phase as Rust: a pre-fill same-time
candle may value the earlier inventory, while lifecycle reopening uses the causal
fill timestamp. The initial prototype incorrectly prolonged cooldown by one
observation in this case; the Rust comparison exposed it and the episode metadata
now clears cooldown at the factual reopening time. The shared kernel source owns
this component; no Python decision formula or prior permission supplies its signal.
Optional full point traces exist only for test inspection. Ordinary callers can
request compact scalar results without allocating or returning a trace.

All 36 source-verified CUDA composer tests pass, comparing 375 scope snapshots with
Rust `hsl_trace` and `hsl_controller`: 288 generated one/two/three-pair snapshots,
three empty-current cases, sixty authored terminal/reopening cases, and 24 compact
evaluations with no point trace. These cover
long/short, clipped and wrapped histories, both candle phases, EMA spans 1/2.5/25,
always/never restart, same-minute round trips, missing closes and unexplained current
openings. Every point's timestamp, exposure, flattening and action agrees; cashflow,
UPNL, raw and EMA use explicit local float32 component bounds. These bounds do not
accept optimizer ranking, feasibility or trajectory differences.

The compact evaluations also compare final action, raw/EMA, cooldown timestamp
and latest terminal scores with Rust. A one-row unused test buffer proves that
the no-trace path writes no points and requires no full trace allocation. Normal
result payloads therefore need not expose histories for independent inspection.

An internal opt-in route now drives actual native EMA/TM kernels through this
evaluator. The default optimizer worker has not adopted it. Each selected coin/side
uses its factual endpoint header, a clipped scope prefix and disposable resident event
scratch. The physical budget reserves 16 bytes per factual capacity slot for events,
rounded to 32-byte allocation nodes. Ordinary observations use the current candle's
close time; terminal fills use only causal preceding closes or retained execution prices.
The terminal score remains available for reporting before episode reset.

All four single/fused EMA/TM entry paths attach the common evaluation context.
Temporal TM reconstructs these bindings after every chunk; serialized thread pointers
are never inputs to the next evaluation. A separate runtime control enables evaluation
without changing compiled capacity identity. Capture-only and disabled-policy controls
remain available during development. All 33 actual CUDA caller controls pass:
18 topology cases, nine chunk/full equivalence cases, two GPU-only growth cases and
four disabled-policy cases. The full 119-check integration suite passes: 109 actual
CUDA cases and ten host policy cases, including the composer and capture/recovery
controls. Rust passes 332 tests with one existing ignore and default-feature test
compilation.

Six independent stressed two-coin comparisons use both strategies, all three HSL
scopes, 256 minute bars, seed 43, threshold 0.002, EMA span 2.5 and ten-minute cooldown.
Apply persistent price shocks at bars 96/150, multiplying coins zero/one by 0.7/1.3.
Time in RED, triggers, restarts and mean/maximum halt duration agree exactly in every
case. EMA fill rates also agree. TM has twelve fewer fills and worst-drawdown residuals
up to 0.00002991. Both ADG values are zero on this sub-day fixture and therefore do
not establish growth parity. CPU references run independently; CPU execution is
forbidden during each GPU request. This is opt-in kernel evidence, not default
service adoption or general parity acceptance.

The longer diagnostic repeats four sixteen-candidate cohorts: both strategies,
seeds 7/43, both sides, two coins and 3000 minute bars. Use unified HSL, threshold
0.002, span 2.5, cooldown 10000 minutes and one-day lookback; multiply coins zero/one
by 0.7/1.3 from bars 1500/1800. Candidate generation is the public cohort tool's
quantity/EMA sweep. Input and candidate identities match the preceding materiality
experiment; all 64 newly evaluated CPU rows match its nine metrics exactly.

The factual opt-in GPU route matches all five HSL metrics, Pareto membership and
all pair orderings for EMA seeds 7/43 and TM seed 7. TM seed 43 retains two material
trajectory discrepancies: candidates zero/two differ by 148/149 halt minutes and
about five percentage points in RED. Its Pareto membership differs. No authored
limit flips occur, and CPU ADG regret at the GPU's best-ADG candidate is zero in
each cohort. Remaining numeric/fill discrepancies and the TM trajectory outliers
are not accepted by this evidence. In particular, unchanged best-candidate regret
does not establish general ranking or feasibility equivalence.

These initial GPU calls include compilation and bounded capacity retries. They
also expose hot execution cost from reconstructing flat idle tails at every
observation. Evaluation-local compaction is under validation: a trace-free scope
with every fill consumed, no reconstructed/current exposure and an exact zero
raw/EMA signal may jump to its endpoint while preserving logical observation
counts and current cooldown/lookback timing. It retains fresh reconstruction and
introduces no persistent permission cache.

The expanded 60 native caller controls pass on the preceding integration build:
36 one/two-coin topology cases, eighteen temporal equivalence cases, two growth
cases and four disabled-policy cases. The flat-tail build separately passes all
62 composer checks against 651 Rust trace/controller snapshots, including 216
long idle cases covering clipped starts, both candle phases, three EMA spans,
zero/short/long cooldowns and always/never restart. Observation counts, actions,
flat timestamps and terminal scores preserve the existing reference contract.
Its 332 Rust tests (one existing ignore), default-feature compilation and ten
host capacity/error-policy checks pass. All 100 actual native caller checks also
pass: sixty dispatch/temporal/growth/disabled controls and forty factual capture,
storage, interruption and recovery controls. Together this is 172 checks, including
162 actual CUDA cases. The same 64 matched candidates preserve all nine GPU metrics
exactly after compaction, on their initial evaluation and three warm repeats.
Warm median seconds per sixteen-candidate cohort are 85.55/72.06 for EMA seeds
7/43 and 37.64/36.87 for TM seeds 7/43. These measure the opt-in reconstruction
path, not general optimizer throughput. There is no matched warm pre-compaction
measurement, so they establish neither a speedup nor performance acceptance.

Independent outlier traces preserve CPU fills and all nine metrics when detailed
reporting is enabled. That control does not disable CPU caches. Fresh standalone
Rust reconstruction from the captured GPU facts reproduces the GPU restart
decisions: candidate zero flattens at minute 1555 rather than the CPU's 1554;
candidate two flattens at 1539 on both. At minute 2791, candidate two's reconstructed
terminal EMA is approximately 0.00199878 from GPU facts and 0.00200212 from CPU
facts, straddling the 0.002 threshold. Candidate zero's GPU facts remain halted
until minute 2940, while CPU facts release at 2791. Thus the remaining trajectories
are already different upstream of scope reconstruction; this evidence is not a
precision-policy acceptance or proof of the first divergent execution's cause.

Early execution inspection identifies a separate TM ladder bug: Rust advances
the simulated book touch and reprices its initial sizing floor at each recursive
rung; the multicoin GPU helper kept its original sizing price. A short-side
sixteen-rung canonical-order regression fails on thirteen quantities before the
correction; the long control passes. Updating that anchor passes both cases and
37 affected recursive gate, market and fused caller checks on CUDA. The rebuilt
extension also passes 332 Rust tests (one existing ignore) and default-feature
test compilation. The corrected TM seed-43 cohort restores all Pareto members
and drawdown pair ordering, and resolves candidate zero's HSL mismatch. Candidate
two retains its 149-minute early restart and two RED pair-order disagreements.
No authored limit flips occur. This evidence does not accept the remaining
trajectory discrepancy or establish general optimizer quality.

A later sizing control isolates candidate two's first remaining discrete quantity
change at minute 393. The CPU cash balance is 1020.43430834 and the f32 GPU balance
1020.43072510; the initial quantity falls just above/below 53.5 quantity steps.
Accumulating the same CPU-encoded cashflows naively in f32 produces 1020.43023682,
while summing those encoded net values in f64 preserves 1020.43430835. Simply
encoding the CPU's completed HSL facts and marks in f32 does not reproduce the
early restart, so transport rounding alone is not its explanation.

A diagnostic-only compensated-f32 account variant removes this first quantity
change. Its same sixteen-candidate cohort preserves Pareto membership and ADG/DD
pair ordering, with no authored limit flips, but retains the 149-minute restart
difference and two RED pair-order disagreements. Later rounding/trajectory
changes remain. This supports a small accumulation improvement; it does not
accept the remaining HSL discrepancy or authorize default factual-worker adoption.

The combined factual-HSL, recursive-sizing and compensated-account build passes
216 focused controls, including the actual native temporal cases and all 72
checkpoint contracts. Its matched 64-candidate comparison preserves every Pareto
front and all ADG/DD pair orderings, with no authored limit flips and identical
initial/warm GPU results. All five HSL metrics agree in three cohorts. TM seed 43
candidate two still restarts 149 minutes early, leaving two RED pair-order
disagreements unresolved. Warm seconds per sixteen candidates are 85.27/67.12 for
EMA seeds 7/43 and 36.56/34.77 for TM seeds 7/43. These are observations, not
matched speedup evidence or general numeric/search acceptance.

Subsequent independent producer checks supersede the remaining 149-minute outlier
above. CPU partial-initial subtraction can turn two mathematically aligned exchange
quantities into a representation just below their step. All four affected CPU
branches now apply the existing representation tolerance before downward rounding;
true below-step controls retain their original floor. Recomputing the 64 independent
CPU reference rows resolves all five HSL metrics in all four cohorts, all ADG/RED
pair orderings and all authored limit classifications. TM seed seven retains a tiny
worst-drawdown ordering/front difference: the CPU drawdown regret of the GPU's best
DD candidate is about 0.000000220. Accept this near-indifference for these four synthetic cohorts: no authored limit
classification flips occur, ADG/RED ordering is preserved, and selected-candidate
regret is negligible. Across all 64 candidates the maximum ADG/DD/fill-rate
residuals are about 0.00007293 / 0.000003829 / 4.903 fills per day. These describe
observed fixture-specific envelopes, not universal error bounds. Preserve the
strict comparison results; no rounding-policy change, bitwise trajectory repair
or broader search-equivalence claim follows from this bounded acceptance.

Full-window composition remains the correctness baseline. Develop a disposable
factual-cutoff memo using immutable pair identity/exposure and monotonically
advancing simulator fact versions. A cache hit skips only the reverse prefix walk;
pair reconstruction, prices, budgets, lookback and controller evaluation remain
fresh. Invalid headers still reject before a hit. The first 21 CUDA prefix/memo
controls and 62 composer cases against 651 Rust snapshots, plus ten host policy
cases, pass. All sixty native dispatch, temporal, growth/retry and disabled-policy controls
also pass, with six compensated-cashflow continuity cases. Eighteen selected
checkpoint and five documentation checks pass. All four paired cohorts preserve all nine metrics exactly in the initial and
both warm repeats, for 64 distinct candidates across both strategies and seeds
seven/43. Each fresh process forbids Python and Rust CPU backtests and reuses the
independent corrected CPU references. Alternate uncached/cached processes for each
cohort; compile the diagnostic uncached variant with
`PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED=0`, including its actual temporal-state size
queries. Inputs and candidate identities match the recipe above.

| Strategy / seed | Uncached warm median, seconds | Cached warm median, seconds | Measured ratio |
| --- | ---: | ---: | ---: |
| EMA / 7 | 85.26 | 9.31 | 9.16x |
| EMA / 43 | 70.49 | 10.57 | 6.67x |
| TM / 7 | 37.64 | 31.63 | 1.19x |
| TM / 43 | 37.77 | 31.77 | 1.19x |

Each median contains two warm observations. Torch peak allocated/reserved bytes
are unchanged within every pair; private kernel storage and total driver/host
memory are outside that measurement. Initial times include compilation and are
excluded from the warm ratios. This fixture-specific gain supports keeping the
small cutoff memo, without claiming optimizer throughput or general resource
acceptance. Checkpoint semantic cutover, default worker adoption and representative
resource/performance acceptance remain required.
No global parity policy is widened.

The subsequent source-verified history/capture recheck passes all 99 cases:
59 reconstruction, endpoint, ring and compacted-view controls, including 996
Rust-referenced histories, and forty native capture/storage/retry/temporal controls.
Universal interruption wiring uses a shared no-op callback so compatible scenarios
retain the same batch key; distinct explicit callback owners remain separate.
The real-construction regression fails before that correction and passes afterward,
with CPU simulation forbidden. All 54 affected host suite-key/topology/fused-
construction controls pass. This host-only fix preserves the tested GPU source.


### Native factual worker cutover

The local native service enables factual HSL behind its existing prepared-request
API; CPU orchestration receives the same compact results. Retained legacy screening
keeps its previous mode. Every dispatch checks effective candidate HSL flags and
coin overrides, rather than the base config alone. An HSL-off dispatch omits factual
capture and storage; existing one-side EMA disabled-HSL specialization still
applies where supported. The GPU owner starts with at most 256 factual records per
pair, grows within its scratch budget after rejected GPU work and retains that
learned estimate when subsequent HSL-off/on traffic reuses the same runner. This
initial estimate is execution policy, not a user config requirement or claim of an
optimal universal capacity. Other physical batch/history guards remain active.

The native evaluation contract advances from execution version one to two. Old
native checkpoints reject saved fitness under the previous semantics, while saved
configurations remain valid seeds. CPU evaluation contracts and the retained
screening backend's CUDA runtime contract do not change.

The source-verified local cutover passes 38 actual CUDA controls: eighteen
HSL-on/off/on transitions across both strategies, all scopes and side topologies,
eight base-off/coin-on overrides, and twelve asynchronous service requests compared
with explicit factual replay. Transition checks force growth, verify learned
capacity reuse, compare every returned native raw output and restore the original
compiled identity. HSL-off controls preserve legacy outputs and reduce scratch
cost. Python CPU simulation is forbidden in these controls, with the Rust CPU
simulation API also forbidden in actual service cases. Ten host retry/fatal-policy
checks and both old-precision/old-version checkpoint rejection controls pass.
The lifecycle and caller checks below extend this evidence; representative
metric, resource and performance acceptance remains pending.


The wider default-worker check also passes all 42 actual native HSL lifecycle
and loss-reporting comparisons against current CPU references, including unfinished
panic/halt durations, scoped trigger/restart attribution and halt-to-restart loss.
The documentation-adjusted source passes 77 host session, dataset, tuning, retry and
documentation checks plus eighteen selected checkpoint contracts. All 56 wider CUDA controls
also pass: 52 real optimizer CLI cases, two canonical prepared-dataset cases and
two incremental service controls. They cover automatic/fixed dispatch, clean
interruption/resume, scenario screening, seed anchors/coupling and current HSL
without CPU simulation during optimization. All 951 source files remain unchanged
after device validation. Representative metric/resource/performance acceptance
and reviewed development integration remain open.

The final integrated source passes 140 host/session/contract/anchor controls and
144 actual device replay/capture/variant controls. Short-only and fused short-only
coin-policy cases isolate the effective short override independently of the base
and long-side policies. A separate negative control omitting only fused TM short
override admission fails the targeted raw HSL metric comparison while the other
seven override controls pass. All 951 passing source files remain unchanged;
production code is unchanged by that coverage extension. Reviewed development
integration and representative acceptance remain separate gates.

Two current-head review findings require corrections before integration. Native
capture must use effective coin policies: a base-enabled coin scope with every
coin explicitly disabled needs no factual storage or history-readiness requirement.
Policy-value validation still applies. After successful replay grows its factual
history, the service must refresh the physical dispatch ceiling before claiming
more work; both fixed and automatic scheduling are covered.

The corrected production source passes 220 host contracts, all 146 factual replay
and capture device cases, eighteen execution-view controls (four CUDA, fourteen
host), and twelve real CUDA CLI/data/service cases. Six new disabled-coin policy
cases cover both strategies and long/short/fused execution, including restored
inheritance and rejection of malformed policy values. All six fail the preceding
production; both new fixed/automatic ceiling controls likewise fail that version.
An initial execution-view assertion depended on a bound-method implementation;
the revised assertion checks the residency-owned proxies and still requires exactly
one resident replay scratch owner. All passing source files remain unchanged.
Current-head independent review and CI remain required before integration.

### Resource design gates before further tuning

Before the compact-layout change, native factual replay retained the bypassed observation tree/window.
For 25 coins, two sides, 52 scopes, a 90-day minute lookback and 256 factual records
per pair, the actual allocation formula reserves about 109.95 MiB per candidate:
109.33 MiB of legacy observation storage and 0.61 MiB of factual storage. A 512 MiB
scratch allowance admits at most four candidates before other histories are counted.
This is an allocation calculation, not a measured device-memory peak. Give native
factual replay its own compact layout and retain the old layout only for its consumers.

The compact native layout now reserves only pair headers, retained factual records
and disposable reconstruction events. Its controller omits observation-window
state, and an HSL-off dispatch allocates neither factual scratch nor legacy rows.
The retained observation route keeps its original layout and controller. The native
layout is part of shader-cache identity, including TM's opaque replay-state sizing.

Actual CUDA allocation controls prepare the same 25-coin, two-side, 90-day input
for EMA Anchor and Trailing Martingale. Both request 642,304 bytes of native HSL
scratch versus 115,287,744 bytes for the legacy layout with factual storage appended.
Buffer sizes and the allocator's requested-byte counter agree. Allocated-block
padding is measured separately; this is HSL scratch evidence, not a total-memory
peak or whole-optimizer throughput claim. Both regressions fail the preceding
implementation at the legacy-storage assertion.

Source-verified validation includes 333 Rust tests (one existing ignore), default
test compilation, 156 host storage/service/residency/tuning controls, ten direct
shader-library/source callers, 22 targeted CUDA transition/partition controls,
326 broader replay/capture/legacy-window/execution-view controls (312 CUDA,
fourteen host), twelve actual CUDA CLI/data/service controls, two allocation
controls and 42 native lifecycle/loss controls. The partition controls compare all
returned raw metrics with factual replay using the retained layout. Lifecycle/loss
checks include CPU references outside optimization. Checked sources remain unchanged.
Independent review and required CI pass; compact storage is integrated on
development through [PR #1945](https://github.com/enarjord/passivbot/pull/1945).
Representative total-resource acceptance remains open.

Runner-local learned capacity previously disappeared when residency cleared the
runners. The service now retains only integer capacity estimates per dataset and
runner role, restores them before computing the recreated runner's physical limits,
and refreshes them after successful replay. An HSL-off request remembers the learned
capacity rather than its active zero allocation. Device buffers remain disposable;
no checkpoint state is added. Malformed and over-budget estimates fail before HSL
scratch allocation, and closing the service clears its metadata.

The source-verified change passes 154 host residency/dataset/executor/tuning checks.
Four production-residency controls use fake device transport and runner computation;
omitting only restoration makes all four fail at the reset-to-256 assertion.
Four actual CUDA service controls cover both strategies, different scenario data
and compatible-data owner switches. Initial GPU replay grows beyond its 256-record
seed; returning to the first scenario uses the same learned capacity with identical
metrics and zero overflow retries. Weak references prove that estimate retention
does not pin the evicted runner. CPU backtests are forbidden in those service checks.
Checked source files remain unchanged. Independent review and required CI pass;
capacity retention is integrated on development through
[PR #1946](https://github.com/enarjord/passivbot/pull/1946).

The compact-storage development merge is integrated without changing the checked
capacity implementation or host regressions. The combined source passes all 154
host checks and sixteen actual CUDA service checks: the four capacity-eviction
controls plus twelve existing authoritative factual-service callers. All 951
checked files remain unchanged after validation. Rust/shader sources and the
verified extension are the reviewed compact-storage build; this slice changes
only Python execution ownership and its regressions.

Matched held-position measurements below establish roughly quadratic fresh
composition cost. Guarded scalar continuation preserves the measured factual/CPU
outputs and improves stable held cases, with unchanged busy multi-entry performance.
It is integrated on development through
[PR #1947](https://github.com/enarjord/passivbot/pull/1947) after clean independent
review and required CI. Fresh reconstruction remains the explicit fallback;
matching allocation sizes alone never establishes parity. Representative useful
throughput and total host/device/disk resources remain open before further tuning.

Multicoin allocation/retry/dispatch now has a private strategy-neutral owner, with
EMA/TM as sibling adapters and explicit parameter/override layouts. Native recovery
histories reduce before logical result assembly; the bounded evidence is recorded
above. Public runner names and the separate legacy directional family remain.
This focused extraction preserves the service boundary and adds no general backend
framework. Combined validation also passes 216 affected controls: fifteen layout, 35 recovery,
134 factual replay, four optional-history, sixteen expired-history and twelve actual
CLI/data/service checks. Rust/shaders match the reviewed continuation runtime;
all 952 checked sources remain unchanged. Independent review and CI gate its
integration; these gates take priority over marginal launch gains.

### Held-position reconstruction scaling

The compact native service is measured on twelve matched synthetic cases: EMA
Anchor/TM, 2/25 long-side coins, and 512/1024/2048 minute bars. Start from the public
parity fixture with seed 7, unified HSL threshold 0.99, span 2.5 and 90-day lookback.
Replace high/low/close/volume with 100.1/99.9/100/100, except low 97 at bar 64.
EMA uses spans 10/20, offset 0.02, base quantity 0.02, doubling factor 1 and zero
inventory/volatility offset weights. TM entry uses spans 10/20, initial distance
0.01, quantity 0.02, doubling factor 1, threshold 0.9 and zero threshold weights;
close uses quantity 1, threshold 0.5 and zero threshold weights. Apply these strategy
parameters to both sides, retaining the fixture's disabled short side. Other settings,
BTC prices, markets and date alignment are unchanged. Request `adg_strategy_eq`,
`drawdown_worst_strategy_eq`, `fills_per_day`, `position_held_hours_max` and
`hard_stop_time_in_red_pct`.

Source-verified CPU references prove every coin enters at bar 64 and makes no
closes, with held durations 7.43, 15.97 and 33.03 hours. GPU requests use the native
future API, batch width one and tuning off. Each HSL-on/off/on-repeat phase contains
an initial request and two warm repeats; explicit factual-mode/capacity assertions
verify the phase switch. CPU simulations are forbidden in GPU requests and run
separately as references. Dataset identity and all five requested metrics agree
within 1e-6 absolute/relative tolerance; GPU repeats and phases agree exactly.
All 951 checked source files remain unchanged.

| Strategy / coins | Warm HSL-on, 512 bars | 1024 bars | 2048 bars |
|---|---:|---:|---:|
| EMA / 2 | 0.148 s | 0.594 s | 2.447 s |
| EMA / 25 | 1.388 s | 5.602 s | 22.450 s |
| TM / 2 | 0.148 s | 0.607 s | 2.484 s |
| TM / 25 | 1.321 s | 5.450 s | 22.500 s |

On-repeat measurements reproduce this shape. For 25 coins, HSL-off warm medians
are 0.098/0.172/0.341 s for EMA and 0.124/0.227/0.459 s for TM. Compilation is
excluded from warm timings. These single-candidate synthetic requests demonstrate
the repeated historical reconstruction bottleneck; they are not a whole-optimizer
throughput comparison or a 90-day simulation benchmark. Torch allocation counters
exclude driver/compiler allocations, and process peak RSS includes cold compilation;
total host/device/disk acceptance remains open.

Evaluate a compact, guarded active-episode cursor using the existing scope recurrence.
Fresh reconstruction remains the reference and fallback when facts, budgets, clipping,
causal phase or numerical conditions invalidate reuse. Require paired fresh/cached
outputs, reset/temporal controls, CPU parity and measured resource/performance evidence
before adoption. Do not add launch tuning or a general backend framework for this fix.


### Guarded active-episode continuation — paired development evidence

The same twelve held-position recipes above compare fresh reconstruction with
native continuation in isolated processes. Prefix each compiled source with
`PASSIVBOT_HSL_INCREMENTAL_ENABLED` set to zero or one; include TM replay-state
sizing in the same process-wide variant. Keep the native future API, width one,
tuning off and HSL-on/off/on-repeat requests unchanged. All five raw returned
metrics agree exactly across variants and repeats; independently run CPU references
agree within 1e-6 absolute/relative tolerance. Input identities agree, and all checked
source files remain unchanged after the matrix.

Each cell below is fresh / continued warm seconds, excluding first-use compilation.

| Strategy / coins | 512 bars | 1024 bars | 2048 bars |
|---|---:|---:|---:|
| EMA / 2 | 0.148 / 0.042 | 0.594 / 0.087 | 2.453 / 0.070 |
| EMA / 25 | 1.391 / 0.139 | 5.607 / 0.279 | 22.480 / 0.561 |
| TM / 2 | 0.149 / 0.076 | 0.602 / 0.055 | 2.473 / 0.074 |
| TM / 25 | 1.320 / 0.156 | 5.451 / 0.316 | 22.497 / 0.649 |

The 25-coin cases improve from roughly fourfold to twofold growth as history
doubles. On-repeat timings reproduce this shape. Small cases have noisier timings:
for example TM/2/512 improves from 0.149 to 0.076 seconds initially, but its
on-repeat continued median is 0.023 seconds. These are bounded synthetic service
requests, not whole optimizer throughput or a 90-day simulation claim. Cold
compilation remains substantial and is outside warm speedup ratios.

All recorded Torch allocated/reserved peaks match between variants. At 25 coins
and 2048 bars, requested allocator peaks are 2,646,016 bytes for EMA and 2,647,552
bytes for TM, with 4,194,304 bytes reserved for each. Compiler/private kernel storage,
driver allocations, process memory including compilation, and disk are separate
resource surfaces. Unchanged Torch peaks do not close total-resource acceptance.

Corrected source-verified validation passes 82 actual CUDA component cases, including
20 continuation/seed-denial controls and 62 preceding fresh-composer references.
Eighteen native policy/on-off-on/capacity comparisons preserve every raw output
exactly, including fields that differed in the initial draft. Three seed-guard
omission controls and six scalar-guard omission controls fail as expected without
changing production source. Rust tests and default test compilation pass with a
rebuilt, source-verified extension. No comparison tolerance was widened.

Reuse stores only scalar peak/EMA/PnL and continuation identity. It requires a known
active episode, unchanged selected factual histories, consecutive minutes, unchanged
budget/smoothing and lookback start, stable endpoint inventory/basis and causal
quotes. Missing/clipped history, same-minute fills, completed episodes, near-threshold
numerical concerns and cache loss reconstruct fresh. It retains no action permission.
The mixed candidate evidence below passes. Broader replay validation also passes
350 cases, including 36 exact temporal controls covering both legacy and native
factual layouts, one/two coins, long/short/both sides and coin/pside/unified HSL.
All 42 native lifecycle/loss checks and twelve real CLI/data/service checks also
pass. The latter include eight optimizer bootstrap/resume combinations with CPU
simulation forbidden, lazy suite preparation and incremental resource admission.
All 952 checked source files remain unchanged after the combined validation.
Representative multi-entry/clipping pairs also pass as recorded below. Total-resource
checks remain a project-wide gate; independent review and CI are required before
integration.


The paired continuation experiment also refreshes the four unified HSL cohorts
above: sixteen distinct candidates per EMA/TM seed 7/43, two coins, both sides and
3000 bars. CPU references run separately with the current verified extension.
Both GPU variants forbid CPU simulation. Each comparison has an initial replay
and two warm repeats; the same compiled variant is used for TM state sizing.

All nine GPU metrics agree exactly with preceding factual reconstruction and
across variants, as do measured limit/ranking outcomes. All five HSL lifecycle
metrics equal the refreshed CPU references exactly. Existing ADG/fill trajectory
residuals and the TM seed-7 drawdown near-tie remain visible: its CPU regret at
the GPU drawdown winner is 2.196e-7, with no limit flips. This is unchanged
fixture-specific evidence, not a new general tolerance approval.

| Cohort | Fresh warm median | Continued warm median | Fresh / continued CUDA local bytes |
|---|---:|---:|---:|
| EMA / 7 | 9.347 s | 9.253 s | 5,776 / 6,096 |
| EMA / 43 | 9.686 s | 9.572 s | 5,776 / 6,096 |
| TM / 7 | 8.508 s | 8.489 s | 6,320 / 6,656 |
| TM / 43 | 8.154 s | 8.154 s | 6,320 / 6,656 |

Treat these busy-cohort timings as essentially unchanged; do not extrapolate
the held-episode speedup to arbitrary search traffic. Registers remain 255.
Torch allocated/reserved peaks match, but post-replay whole-device free-memory
snapshots are about 22 MiB lower with continuation for each strategy. Those
snapshots include driver/compiler effects and are not device peak measurements
or exclusive ownership accounting. Record the extra private storage explicitly
while keeping representative total-resource acceptance open.


Four paired boundary requests use the same held-position recipe with two coins and
2048 bars. The multiple-entry profile adds a low of 94 at bar 128; TM uses entry
threshold 0.01, zero retracement and zero retracement weights to admit further
entries. EMA/TM make four/fourteen fills at bars 64 and 128, with no closes. The
clipped profile instead uses a one-day lookback, retaining the two entries at bar
64. All five GPU metrics agree exactly between fresh/continued variants and
repeats; separate CPU references agree within 1e-6 absolute/relative tolerance.
Torch peaks match, and all checked sources remain unchanged.

| Boundary / strategy | Fresh warm median | Continued warm median |
|---|---:|---:|
| Multiple entries / EMA | 2.494 s | 2.482 s |
| Multiple entries / TM | 2.511 s | 2.496 s |
| Clipped lookback / EMA | 2.276 s | 1.003 s |
| Clipped lookback / TM | 2.296 s | 1.017 s |

Repeated measurements reproduce these timings. Multiple-entry performance is
essentially unchanged; clipping improves by about 2.25 times in this fixture.
Constant marks produce zero risk metrics here, so these cases establish boundary
and performance behavior, not additional nonzero HSL-risk coverage. Component,
mixed-candidate and lifecycle cases supply that separate evidence. Do not relax
endpoint guards merely to obtain a larger speedup.


## Native service suite resource baseline

The [service benchmark](../gpu_service_benchmark.md) adds three-scenario observations
without CPU simulations. This baseline uses the preceding completed-work demand
policy; the window-demand correction and broader acceptance remain separate.

```bash
passivbot tool gpu-service-benchmark --strategy ema_anchor --coins 12 --bars 5760 \
  --candidates 128 --rounds 3 --tuning-windows 2 --max-rounds 128 \
  --accumulation-delay 0 --report suite.json
```

Use both sides and the public seed-seven fixture. The full scenario has twelve coins
and 5,760 minute bars; early/late scenarios each select four coins and 2,880 bars
from shared arrays, with aligned dates and validity metadata. Ten requested metrics
include recovery distributions and weighted equity/volume. Compilation/cache state
is retained between phases; these are first-use and warm observations, not fresh
cold-cache comparisons. Accumulation is fixed at zero to isolate width decisions.

On an RTX 3070 Ti Laptop GPU, all 33,408 results match the width-one references
exactly, with no reduction rounding differences. The automatic phase completes
82 rounds and at least two unchanged 24-sample/30-second evidence windows per
scenario. First-use timings, completion tails, per-window decisions and actual
batch counts remain in the report; small batches alone do not prove optimal tuning.

| Execution width | Warm requests/s | Peak Torch allocated bytes | Earlier sampled RSS bytes | Sampled global device bytes | Peak sampled packing bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.901 | 3,417,600 | 1,298,616,320 | 1,490,026,496 | 4,888,848 |
| 8 | 13.380 | 5,090,304 | 1,302,138,880 | 1,490,026,496 | 4,888,848 |
| Automatic | 68.268 | 28,797,440 | 1,304,625,152 | 1,513,095,168 | 4,888,848 |

Every owner snapshot has one resident dataset; packing reaches three reusable
entries. Shared source arrays remain unchanged, all spill files are removed on
close, and resource sampling reports no errors. One-second samples can miss short
peaks. These historical RSS observations use a leader-only child walker and may
omit worker-spawned children; they do not certify whole-process-tree memory.
Global device memory includes
driver/display/other-process allocations, while Torch peaks cover its allocator
only. CPU time covers the benchmark process and its threads, excluding compiler
children; it is not isolated orchestrator cost.

This establishes a moderate synthetic suite resource observation and useful
grouped execution. It does not establish large/long HSL resources, exclusively
owned device memory, evolutionary search quality, CPU optimizer throughput or a
globally optimal width. Those remain project acceptance work.


The current service caller also exercises TM with production adaptive accumulation:

```bash
passivbot tool gpu-service-benchmark --strategy trailing_martingale --coins 12 \
  --bars 5760 --candidates 16 --rounds 3 --report tm-suite.json
```

All 384 results match the isolated references exactly for the ten requested
metrics. Width-one, width-eight and automatic warm rates are 0.433, 3.231 and
5.991 requests/s respectively. Every owner snapshot retains one device dataset,
packing reaches three entries, arrays remain unchanged and spill cleanup succeeds
without sampling errors. Automatic Torch allocation peaks at 7,006,720 bytes. The earlier RSS sampler
has the leader-only traversal limitation described above.
This underfilled cohort checks the adaptive-accumulation caller and resources;
it completes no production tuning windows and establishes no tuning optimum.

A separate three-coin, 2,880-bar, two-candidate/two-round EMA run requests one
window per scenario but bounds execution at two rounds. All 36 results agree
exactly and cleanup succeeds. The report retains the measurements, marks automatic
evidence insufficient and exits with status two as specified. This exercises the
failure-to-meet-evidence path without treating valid simulations as failed.

The larger current-demand comparison similarly reaches its initial 128-round
bound with 51,072 exact results and clean resources, but only one completed
window each for early/late scenarios. Their width-128 trials accumulate about
22 seconds toward the unchanged 30-second threshold. A larger bounded run is
required before reporting completed trial decisions; do not shorten the evidence
threshold or present the partial run as proof of convergence.


The extended current-demand comparison uses the same EMA recipe with
`--max-rounds 256`. It finishes successfully after 151 automatic rounds, with
59,904 results matching isolated GPU references exactly and no reduction-rounding
differences. The base scenario consumes fourteen windows and retains width 64
after rejecting width-128 and width-32 trials. Early and late each complete two
windows and accept width 128: measured per-window rates increase from
207.546 to 412.129 and from 213.831 to 421.634 candidates/s respectively.
These are scenario-specific measured decisions, not a globally optimal policy.

Warm cohort rates for widths one, eight and automatic execution are 1.645, 12.131
and 71.781 requests/s. First reference rounds take 230.213 and 233.413 seconds,
slower than the earlier baseline. Different durations, dispatch shapes and
operating conditions prevent a causal whole-search speedup claim.
Automatic Torch allocation/reservation peaks are 26,851,328/50,331,648 bytes;
earlier sampled RSS/global device peaks are 1,304,571,904/1,513,095,168 bytes.
This RSS observation has the same leader-only traversal limitation.
Packing peaks at 4,888,848 bytes, with three cached entries and one resident
dataset in every owner snapshot. Arrays remain unchanged, spill files are removed
and sampling reports no errors. The checked source remains unchanged through
completion. The broader resource, HSL, parity and search-quality gates remain open.


Final explicit-failure tool checks use optimized Python for both strategy CLIs:
three coins, 2,880 bars, two candidates and two rounds, producing 72 exact results
across widths one/eight/automatic. Residency, unchanged inputs and spill cleanup
pass without sampling errors. An injected comparator rejection propagates on the
actual optimized CUDA path; replay instrumentation and CPU guards are restored
and the worker closes. All 27 focused tool/window controls pass with the verified
extension, and the final checked source remains unchanged. This strengthens
failure detection without changing simulation or comparison tolerances.


## Corrected Linux process-tree resource sampling

PR #1950 review identifies that Linux child lists belong to individual threads.
The corrected sampler traverses every task's child list and deduplicates processes.
The real worker-spawned-child regression fails preceding code and passes the fix;
all 29 focused tool/window checks pass on Linux. Earlier RSS observations above
are qualified because their leader-only traversal may omit worker children.

Refresh both strategy recipes with twelve coins, 5,760 bars, sixteen candidates
and three rounds, using production adaptive accumulation. Start with fresh CUDA
driver/CuPy compiler-cache directories, retaining state between execution phases.
All 768 results match isolated GPU references exactly, with unchanged arrays, one
resident dataset, three packing entries, clean spill removal and no sampling errors.

| Strategy | Width-one / width-eight / automatic warm requests/s | Maximum sampled process-tree RSS bytes | Automatic Torch allocated / reserved bytes | Sampled global device peak bytes | Peak packing bytes |
| --- | --- | ---: | --- | ---: | ---: |
| EMA Anchor | 1.996 / 15.308 / 29.559 | 1,629,196,288 | 7,002,624 / 29,360,128 | 1,490,026,496 | 4,888,848 |
| Trailing Martingale | 0.512 / 3.615 / 6.633 | 1,838,252,032 | 7,006,720 / 29,360,128 | 1,680,867,328 | 4,888,848 |

RSS maxima include first-use compilation; later phase warm medians retain compiler
state. The underfilled cohorts complete no tuning windows, and their observations
do not replace the larger scenario-decision evidence or broader acceptance. One-second
sampling can miss short peaks; global device memory includes unrelated allocations.
Both preceding and corrected checked sources remain unchanged after validation.


The subsequent review tightens unsupported-procfs reporting and extension identity:
missing child lists yield null RSS instead of a parent-only baseline, and skipped,
unstamped or mismatched runtime verification fails before fixture preparation.
A changed replay ceiling invalidates its old tuning window before observation,
including the completion that discovered the change; an unchanged ceiling preserves
evidence. All 65 focused Linux/CUDA controls pass. Optimized-Python EMA/TM CLI
refreshes complete 72 exact results with unchanged arrays and clean ownership/cleanup.
Checked sources remain unchanged. These small callers establish the corrected
validation path; earlier moderate resource observations retain their stated scope.

Global-device availability now likewise derives from actual usable memory/utilization
observations. Failed, empty or unsupported command output cannot advertise sampling
as available. All 71 focused Linux/CUDA controls and 72 optimized-Python caller
results pass on the final reporting correction; numerical comparison policy is unchanged.


## Full unstuck compiler specialization

The multicoin EMA Anchor and Trailing Martingale runners prove full unstuck
inactivity over packed float32 candidate flags and immutable coin overrides on
every active side. Any effective consumer or unknown consumed flag retains the
general implementation. Compiler identity includes the decision, including TM
temporal state layouts. Selection/generation and exclusive close quantity/tick
arrays are omitted when the proof succeeds; shared loss-budget and HSL consumers
remain independent. This does not specialize inactive sides or single-coin kernels.

The first source-verified CUDA matrix passes 24 specialized/general raw-output
controls across strategies, sides, coin pins, mixed candidates, temporal replay
and cache transitions, plus seven host proof controls. All returned values and
NaN masks agree exactly. Existing EMA specialization, effective history consumers,
shared realized-loss gates and disabled factual-HSL controls pass 110 additional
checks, including Rust loss-expiry comparisons. Rust tests pass 333 checks with
one existing ignored test; default-feature test compilation passes.

A paired observation uses public seed-seven parity fixtures with twelve coins,
2,880 bars, both sides, disabled HSL/unstuck and 32 candidates. Vary the long
quantity gene as `0.005 + index * 0.0001` (EMA `long_base_qty_pct`, TM
`long_entry_initial_qty_pct`), request ADG, worst strategy-equity drawdown and
fills/day, and use the fused factual proxy at fixed width 32. Compare the internal
forced-general control (`unstuck_specialization=False`) with automatic proof.
Warm both variants once, then alternate their order over six paired rounds.
CPU simulation is forbidden throughout; all 896 returned candidate metric/status
results agree exactly with the general reference. Checked sources remain unchanged.

| Strategy | General / specialized median seconds | General / specialized compiler local bytes | General / specialized registers |
| --- | --- | --- | --- |
| EMA Anchor | 0.55775 / 0.52281 | 13,952 / 13,696 | 235 / 227 |
| Trailing Martingale | 2.42668 / 2.38320 | 16,640 / 16,336 | 255 / 255 |

Observed median time decreases are approximately 6.3% and 1.8%. Compiler local
bytes/registers are reported per kernel thread, not total allocator/VRAM usage.
First-use calls take 12.87/11.71 seconds for EMA and 22.90/21.94 for TM, including
compilation/setup; their order and retained caches do not establish a paired cold
benchmark. The measured path is the proxy/kernel, not full service or search
throughput. Six additional 3,000-bar controls force active factual unified-HSL panics with
unstuck disabled and preserve all raw outputs against the general implementation.
Four disabled-unstuck native CLI controls cover both strategies, suites/screening,
interruption and resumption: they observe the disabled dispatch and forbid CPU
simulations while checking prompt persistence. Twelve existing native bootstrap/
resume CLI controls also pass after development integration. PR #1951 passes
independent current-head review and required Rust/Python CI and is integrated into
development. Wider simulator/resource acceptance remains open.

## Prepared one-side CUDA compilation

A one-side multicoin runner has a fixed prepared direction. Include that direction
in compiler/cache identity and make it constant in the Rust-owned kernel entry.
Fused two-side replay and Metal retain their general entries. The internal
`side_specialization=False` control selects the general CUDA entry. This changes
neither candidate parameters nor the numerical/checkpoint contract. Single-coin
multicoin replay already compiles with capacity one; no separate kernel is added.

Source-verified CUDA checks cover 24 raw/general cases across both strategies,
long/short/fused topologies, one/three coins and enabled/disabled factual HSL.
Every returned value and NaN mask agrees exactly, including TM temporal replay,
cache transitions and restored specialization. Eight native one-side CLI cases
cover both strategies, standalone/suite screening, prompt persistence, interruption
and resume with CPU simulations forbidden. Suites use symmetric coin eligibility
and zero exposure on the disabled side, preserving the existing suite contract.
Together with existing unstuck and CPU-entrypoint controls, 80 distinct checks
pass. Rust tests pass 333 checks with one existing ignored test; default-feature
test compilation and rebuilt-extension source verification pass.

Paired public seed-seven fixtures use one/eight coins, 2,880 bars, a single active
side, disabled HSL/unstuck, width 32 and ADG, worst strategy-equity drawdown and
fills/day. Vary the active quantity gene as `0.005 + index * 0.0001`. Warm general
and specialized variants once, then alternate order over six paired rounds.
All 3,584 candidate metric/status results agree exactly; CPU simulation is forbidden
and checked sources remain unchanged.

| Strategy / side / coins | General / specialized median seconds | General / specialized registers | Local bytes, both variants |
| --- | --- | --- | ---: |
| EMA / long / 1 | 0.04188 / 0.03779 | 162 / 160 | 576 |
| EMA / long / 8 | 0.14739 / 0.14292 | 164 / 158 | 3,536 |
| EMA / short / 1 | 0.04396 / 0.03675 | 162 / 156 | 576 |
| EMA / short / 8 | 0.15180 / 0.14317 | 164 / 158 | 3,536 |
| TM / long / 1 | 0.04922 / 0.04750 | 237 / 230 | 672 |
| TM / long / 8 | 0.19002 / 0.18484 | 217 / 214 | 4,416 |
| TM / short / 1 | 0.16350 / 0.15670 | 237 / 232 | 672 |
| TM / short / 8 | 1.24604 / 1.21615 | 217 / 220 | 4,416 |

Warm median decreases range from approximately 2.4% to 16.4% in these fixtures;
the shortest observations include substantial orchestration/reduction overhead.
Compiler local storage is unchanged and registers are not universally lower.
These are proxy/kernel observations, not whole-search throughput, total VRAM,
paired cold-cache evidence or wider simulator acceptance. PR #1952 passes
independent current-head review and required CI and is integrated into development.

## Granular larger HSL diagnostics

Isolate the base scenario's first public seed-seven EMA candidate from the larger
service recipe: 25 coins, both sides, quantity `0.01`, EMA span zero `5.0`, unified
HSL threshold `0.99`, one-day lookback and the ten service-benchmark metrics.
Use factual replay at width one, profile individual accepted attempts and retain
the same source-verified simulator. This is neither a completed suite measurement
nor population/search throughput evidence.

| Minute bars | Kernel seconds | Fills | Factual capacity | Reported replay history bytes per candidate |
| ---: | ---: | ---: | ---: | ---: |
| 2,880 | 34.115 | 495 | 256 | 977,028 |
| 5,760 | 99.473 | 1,020 | 256 | 1,311,644 |
| 11,520 | 228.941 | 1,925 | 256 | 1,980,876 |

All three complete one accepted attempt without a capacity retry. First-use
compilation is measured separately: 45.023 seconds for the initial two-day control,
then approximately 0.085 seconds for subsequent cached loads. Launch groups one
and 64 return exactly equal two-day metrics/status and take 34.319/34.115 seconds
in the kernel. That single-candidate comparison does not assess wider-cohort
geometry. History bytes are the runner's admission estimate, not total VRAM.
The large active EMA dispatch is a concrete interruption-latency limitation;
pending-future cancellation cannot shorten it. Keep long-suite and interruption
acceptance open.

Separate CPU backtests of the same public recipes take 6.219, 19.228 and 43.617
seconds. These single-candidate observations do not measure CPU/GPU population
throughput. Per-metric comparisons remain explicitly unassessed until discrepancy
materiality is evaluated; no blanket tolerance is inferred from these samples.
ADG absolute errors range from approximately `6.7e-7` to `4.3e-6`, and worst
strategy-equity drawdown errors from `9.0e-6` to `3.1e-5`. Fill rates differ too.
For four days, recovery p95 is 0.688993 days on CPU and 0.523715 on GPU, a difference
of 0.165278 days. The underlying drawdowns are small (approximately 0.000434 and
0.000403); that context does not automatically excuse an objective/limit difference.
Investigate trading-path and recovery sensitivity before accepting this case.

An independent native-service HSL-off control disables the explicit `bot.hsl`
portfolio policy and both side policies. It returns exactly the same ten GPU
metrics and CPU/GPU errors for the four-day recipe. The discrepancy therefore
persists without HSL protection. The CPU reference takes 0.161 seconds; the
15.282-second cold GPU request combines compilation/preparation/replay and does
not establish warm kernel cost. Side-policy disablement alone cannot supply this
control in unified mode. Keep numerical diagnosis separate from duration evidence
and do not reinterpret missing tolerance policy as a parity pass. All completed
checked sources remain unchanged.

A CPU-only precision-sensitivity control of the same four-day HSL-off recipe
rounds candles/BTC, market settings and floating config values to f32 separately
and together before the ordinary f64 CPU simulation. All five controls retain
the same CPU fill rate and all three recovery-duration metrics; the p95 gap stays
0.165278 days against the previously measured GPU result. This rules out those
input-rounding changes alone as an explanation. It does not test f32 arithmetic
inside the CPU simulator, identify the first divergent GPU fill, or accept the
discrepancy. Further diagnosis needs trading-path evidence rather than a blanket
precision explanation.

## Native EMA temporal replay — development evidence

Native CUDA EMA now uses the shared multicoin temporal dispatcher. Optional
state/range arguments preserve side, account, fills, day and metric state;
finalization occurs only at each candidate's actual end. Legacy EMA/Metal keep
their whole-replay entry contract. The compiled state-size query runs before
physical admission, alongside factual, unstuck and metric histories. Obsolete
state allocations are released before layout or physical-batch replacement.

Twenty EMA and eleven affected TM controls pass on actual CUDA. These cover
one-side/fused replay, disabled/coin/unified HSL, hedge/one-way selection, unequal
ends, whole/temporal cache changes, day boundaries, raw/weighted/BTC/recovery
outputs, fatal markers, discarded partial state, actual native-service futures,
factual overflow retries and compiled-size scratch admission. Eleven identified
mocked factual-policy controls pass separately. CPU simulations are forbidden
in the GPU controls. An initial service fixture had an undefined side selector
and failed before service construction; the corrected fixture passes. Rust tests
(333 passing, one existing ignored), default-feature compilation and rebuilt
source-stamp verification pass. Eight additional native EMA optimizer CLI checks
pass for standalone/suite, one-side/shared-account, interruption and resume,
with CPU simulations forbidden. Current-head independent review/CI remain gates.

Repeat the documented 25-coin two-day seed-seven recipe at width one, with the
same ten metrics and factual capacity 256. Whole and both temporal runs return
exactly equal metrics and liquidation status.

| Replay | Runner kernel-phase seconds | Dispatches | Largest observed dispatch seconds | Compiled state bytes per candidate | Reported history/state admission bytes per candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole control | 34.249 | 1 | 34.249 | 0 | 977,028 |
| Temporal first use | 33.776 | 23 | 2.841 | 34,408 | 1,011,436 |
| Temporal warm | 33.375 | 23 | 2.845 | 34,408 | 1,011,436 |

First-use preparation/compilation is separate: 45.411/47.676 seconds for
whole/temporal variants; the warm lookup is approximately 0.000018 seconds.
An interrupt after the first completed 128-bar chunk propagates in 0.046 seconds
(the observed chunk takes 0.044 seconds), returning no partial metrics. This
first-chunk check is not worst-case interruption latency. The maximum completed
chunk above is the stronger duration evidence for this case.

The initial 128-bar active-factual-HSL ceiling improves observable boundaries
without an evident total-runtime penalty in this small timing set. It is not an
optimal launch policy, a general wall-clock guarantee, a completed large-suite
measurement, or CPU/GPU numerical acceptance. Keep wider cohorts, longer suites,
adaptive duration policy and the recovery/trading-path discrepancy open.


## Adaptive native dispatch duration — development evidence

Native CUDA EMA/TM replay adjusts history chunks from synchronized, completed
production commands. A small per-replay controller targets one second, shrinks
slow work promptly and grows only after sustained fast commands. Its initial
work/HSL ceiling remains an upper bound; it adds no calibration backtests,
checkpoint state or public request fields. Native short TM replays now use the
same temporal owner. Legacy GPU and Metal dispatch policy remains unchanged.

Twelve actual CUDA checks pass for both strategies and all side topologies,
forced shrinking boundaries, restored whole replay, zero-work finalization,
unequal endpoints with HSL disabled and actual prepared-service interruption.
All compared replay outputs are exact. Unequal endpoint checks follow the
existing `n - 1` terminal-row contract.

Repeat the preceding 25-coin two-day recipe at width one. The fixed control
holds the same compiled replay's controller constant; all ten metrics and
liquidation status agree exactly across four runs.

| Replay policy | Kernel-phase seconds | Commands | Largest command seconds | Largest command after adaptive shrink seconds |
| --- | ---: | ---: | ---: | ---: |
| Fixed first | 33.598 | 23 | 2.846 | — |
| Adaptive first | 33.434 | 44 | 2.849 | 0.890 |
| Fixed warm | 33.374 | 23 | 2.848 | — |
| Adaptive warm | 33.478 | 44 | 2.853 | 0.892 |

Adaptive chunks shrink from 128 to 40 bars when the workload becomes expensive.
The first expensive command still exceeds the target. Neither this feedback
policy nor one-bar replay guarantees preemption or a maximum wall-clock latency.
Warm total runtime is similar in this bounded set; it does not demonstrate search
speedup or optimal scheduling. Compiled replay state remains 34,408 bytes per
candidate, with 1,011,436 reported history/state admission bytes. Compiler cache
loads are measured separately from kernel time. Interruption after the first
completed command propagates in 0.046 seconds and returns no partial metrics;
this is not worst-case interruption evidence. Wider cohorts, longer suites,
resource totals and the CPU/GPU trading-path discrepancy remain open.

A further 191 orchestration/tuning/CLI checks pass on the verified current Rust
runtime, with one environment skip. These include twelve real native TM/EMA
standalone/suite startup, interruption and resume cases with CPU simulations
forbidden, as well as legacy exact-worker option roundtrips.
Current-head independent review and CI remain required before integration.


## EMA close minimum-remainder regression

`test_gpu_ema_close_remainder_cuda.py` uses the public seed-seven EMA fixture,
25 coins, both sides and 5,760 minute bars, with HSL disabled, base quantity
`0.01` and EMA span zero `5.0`. It compares the corrected shared Rust producer
with the actual native request/future service through the public parity tool.
The source-level Rust test proves why an 18-step position must retain a
nine-step remainder after a nine-step clip. A second regression covers `1e-8`
minimum remainders after clips of 1, 10 and 1,000, accounting for cancellation
error at the operand scale. Accept a near-minimum subtraction discrepancy only
when aligned quantity-step counts confirm a valid remainder. This replaces
fractional-step caps, which can reject valid remainders or absorb genuine deficits
at extreme ratios. Regressions cover 13,000,000- and 40,000,000-unit clips and a
three-step remainder below a four-step minimum at a countable quadrillion-step
ratio. Recovery also requires adjacent quantity steps to remain distinguishable
at each operand's floating-point spacing. A fused residual corrects division
rounding before comparing cardinalities. Above those quantity-resolution or
countability bounds, retain the ordinary full-close decision when subtraction
reports an undersized remainder.
Genuinely undersized remainders
still trigger full closes; GPU strategy arithmetic remains unchanged.

The corrected reference produces 1,020 fills and exactly matches GPU fill rate,
completion ratio and HSL time-in-red. ADG absolute error is approximately
`2.32e-8`, worst-drawdown error `1.01e-7`, and recovery p95 error `0.0020833`
days (about three minutes). The previous four-hour p95 discrepancy therefore
does not justify copying the old CPU close artifact into GPU arithmetic.

This fixture uses a scoped five-minute absolute measurement gate for its three
strict recovery-duration distributions on a nearly flat curve. Return,
drawdown and volume gates remain much smaller; required structural metrics
match exactly. No global tolerance or matching-nonfinite policy changes. This
is a regression acceptance case, not certification of every recovery curve,
limit threshold or strategy combination.

The resolution revision passes 341 Rust tests, with one existing ignored test,
default-feature compilation, touched-file formatting, rebuilt source verification,
181 CPU caller checks without skips and five documentation checks. The earlier operand-scaled revision
passed 186 parity, Rust-backed caller and native EMA CLI lifecycle checks. Current
resolution CUDA parity/native CLI verification, independent current-head review and
CI remain integration gates. Earlier device results do not satisfy this gate.

## Larger factual-HSL service resources — EMA measurement

Use `gpu-service-benchmark --strategy ema_anchor --coins 25 --bars 5760 --candidates 4 --rounds 2 --hsl unified`, seed seven, both sides, one-day lookback and threshold 0.99. Three shared-data scenarios contain 25/full and two 8/half views. First-use and warm execution share the existing compiler caches. This is a synthetic service observation, not CPU throughput, a full optimization run or fresh cold-cache timing.

All 72 request results and statuses match the isolated width-one references exactly, with zero reduction-rounding cases. Every owner snapshot has one resident dataset; packing reaches three entries. Source arrays remain unchanged, spill files are removed after shutdown and resource sampling reports no errors.

| Width | Warm round seconds | Warm requests/s | First warm result seconds | Sampled process-tree RSS peak bytes | Sampled global device peak bytes | Torch allocated peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 684.437 | 0.017533 | 113.672 | 2,217,734,144 | 5,069,864,960 | 7,469,568 |
| 8 | 559.797 | 0.021436 | 515.282 | 1,534,681,088 | 5,069,864,960 | 10,215,936 |
| auto | 554.502 | 0.021641 | 509.100 | 1,543,335,936 | 5,069,864,960 | 10,215,936 |

Wider execution reduces warm elapsed time by approximately 18%, while delaying the first result from 114 to 509–515 seconds. Completions remain bounded by a physical microbatch. Busy many-coin HSL remains expensive; this result establishes a measurable throughput/latency tradeoff, not a promise of speedup over CPU or optimal scheduling.

The automatic phase collects one warm sample per scenario and completes zero unchanged production tuning windows. It executes actual cohorts of four under a width-64 proposal; the evidence does not establish an optimal width. Earlier sustained HSL-off measurements separately exercise completed windows.

Torch reserved peak is 25,165,824 bytes and packing peak is 10,033,620 bytes. Process-tree RSS includes compiler children; global device use includes driver/display/other processes and is not exclusive service allocation. One-second samples may miss short peaks. Whole-process CPU time includes replay/preparation/driver activity and does not isolate orchestration. History/scratch admission is not a complete VRAM guarantee. These facts preserve wider and future device-resource limitations explicitly.

The source-backed continuation limitation is precise: `replay_factual_hsl` begins at `max(minute - lookback, 0)` before applying the factual cutoff, and `hsl_advance_scope` requires the same effective start, budget and factual identity. A sliding lookback, changed fills/budget, expired history or numerical guard therefore reconstructs fresh. A held-position continuation speedup cannot be extrapolated to this workload. These observations identify possible costs; this benchmark does not isolate the fraction attributable to each condition.

These observations do not close numerical, final cutover or general resource acceptance by themselves.

## Larger factual-HSL service resources — Trailing Martingale measurement

Repeat the preceding public service recipe with `--strategy trailing_martingale`.
All 72 results and liquidation statuses match the isolated width-one references
exactly, with zero reduction-rounding cases. All phases retain one device dataset,
reach three packing entries, preserve shared source arrays, remove spill files
at shutdown and report no sampling errors. This GPU-only comparison does not
replace the pending
current-reference parity and native optimizer acceptance checks.

| Width | First-use round seconds | Warm round seconds | Warm requests/s | First warm result seconds | Sampled process-tree RSS peak bytes | Torch allocated peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17,479.624 | 17,485.691 | 0.000686 | 3,829.666 | 2,726,207,488 | 11,951,616 |
| 8 | 10,703.775 | 10,069.300 | 0.001192 | 9,426.111 | 2,663,030,784 | 28,646,400 |
| auto | 10,509.293 | 10,048.788 | 0.001194 | 9,381.709 | 2,671,534,080 | 28,646,400 |

Each round completes twelve logical requests across the three views. Wider
batching reduces warm round time by approximately 42.5%, but delays the first
result from about 64 minutes to 156–157 minutes. The absolute cost is substantial;
these results strengthen the need to profile busy HSL reconstruction and consider
usable-result latency within the existing batch policy. They do not establish
CPU-relative speedup, optimizer search quality or an acceptable general latency.

The automatic phase records two warm tuning samples, one for each half-history
view, and zero completed production evidence windows. Its base-view controller
has no retained samples. Actual physical cohorts contain four candidates; the
final proposed width is 64, with ceiling 92 for the base view and 128 for the half
views. Neither two samples nor the small configured cohort demonstrates tuning
convergence or an optimal width.

Torch reserved peaks are 25,165,824 bytes at width one and 46,137,344 bytes for
wider execution. Sampled global device peaks are 5,332,008,960, 5,352,980,480 and
5,361,369,088 bytes respectively. The preceding sampling and allocation limitations
apply: global use is not exclusive service VRAM, process-tree RSS includes compiler
children, and whole-process CPU time does not isolate orchestration. This completed
resource cohort leaves representative full-search and remaining numerical,
lifecycle and retirement gates open.

## Current factual cohort and metric-surface observations

After the minimum-remainder repair, the native factual cohort recipe compares
64 candidates: EMA Anchor and Trailing Martingale, seeds 7 and 43, sixteen
candidates per case, two coins, both sides and 3,000 minute bars. It requests
22 metrics, unified HSL at threshold 0.002, span 2.5, one-day lookback and
10,000-minute cooldown, with shocks `(0, 1500, 0.7)` and `(1, 1800, 1.3)`.
Direct native replay and the request/future service use widths sixteen and
automatic. The cohort tool and documentation checks pass 73 tests without
skips; checked sources remain unchanged. These are fixed-candidate comparisons,
not optimization runs or CPU validation of native optimizer fitness.

Across all 64 candidates, HSL mean/maximum duration, trigger/restart rates,
time in RED, completion ratio and initial-entry median/p95 agree exactly.
The ADG/drawdown/RED fronts match for both EMA cases and TM seed 43.
TM seed 7 changes one drawdown pair ordering and excludes one CPU-front member;
the CPU drawdown regret at the GPU-selected minimum is 2.1962e-7. All four
GPU-selected maximum-ADG candidates also maximize CPU ADG. The requested
ADG, drawdown, RED and completion feasibility checks agree for all candidates;
this does not establish agreement for thresholds placed inside their numerical
differences or for other objectives.

| Residual across the two seeds | EMA Anchor | Trailing Martingale |
| --- | ---: | ---: |
| Largest absolute ADG difference | 7.9057e-7 | 7.2922e-5 |
| Largest absolute worst-drawdown difference | 2.3953e-6 | 3.8286e-6 |
| Largest absolute EMA-tail drawdown difference | 1.5119e-4 | 5.2765e-5 |
| Largest relative EMA-tail drawdown difference | 9.558% | 29.051% |
| Largest absolute fill-gap p95 difference | 1 minute | 0 |
| Largest absolute initial-entry p99 difference | 0 | 0.4415 hours |

The EMA-tail reducer still averages a partially selected logarithmic bin,
whereas CPU analysis selects the actual largest samples. Its systematic
downward bias and changes to selected tail objectives remain an acceptance
item. Initial-entry and fill-gap percentile bins have separate resolution
limits; matching streamed means or HSL decisions does not prove these tails.
No general numerical policy is widened. The two-day cohort lacks a CPU
weighted exponential-fit result, so that field remains unassessed.

A separate native comparison requests all 157 supported metrics on six
twenty-day public shock fixtures: each strategy with long, short and both
sides, two coins, seed 43, coin HSL at threshold 0.002, span 2.5 and
five-minute cooldown, enabled unstuck and shocks `(0, 1440, 0.7)` and
`(1, 1800, 1.3)`. All requested CPU/GPU values are present and finite in
these cases. Of the four fields with existing comparison policies, ADG
differs beyond the strict policy in five cases and fills/day in four;
drawdown and completion pass all six. The other 153 fields per case remain
policy-unassessed. Presence, finite values and small headline errors do not
close per-metric acceptance, selection-materiality or full-search gates.
