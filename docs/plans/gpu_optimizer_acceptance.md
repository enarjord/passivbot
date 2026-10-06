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
differences and partial-day weighting still require materiality assessment. No policy
was widened and this slice does not establish general simulator parity.

## Work still required before legacy retirement

1. Finish the code-backed approximation inventory for the actual native shared-account
   path. In particular assess requested histogram tails, recovery trajectories, partial-day
   weighting and HSL observation timing using meaningful samples and canonical limit
   decisions. Keep strict measurements visible; justify bounded accepted differences
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
