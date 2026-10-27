# Prepared native GPU backtest service

`optimization.gpu.native.NativeBacktestService` is an internal execution interface under
development. CPU code registers prepared scenarios, submits identified backtest requests
and receives futures containing compact metrics and actual simulator liquidation status.
Device buffers, packing, replay handles
and residency stay inside the service. No CPU backtest or evolutionary algorithm runs
there. The experimental GPU optimizer uses this interface; independent numerical and
resource evidence is recorded below.
The [development evidence map](plans/gpu_optimizer_acceptance.md) records the ownership,
incremental-admission and optimizer persistence cases separately from outstanding parity
and cutover requirements.

The offline [service suite benchmark](gpu_service_benchmark.md) measures latency,
resource use and completed default tuning windows independently of optimization.

The service chooses Metal/MPS on supported Apple Silicon Macs and CUDA elsewhere.
`CudaBacktestService` remains an explicit CUDA-only qualification entry point.
Both devices keep one active dataset and one replay scratch owner, spill immutable
packing to run-local disk and restore it on scenario changes. Metal bounds invariants
against the remaining recommended working set and reserves driver/allocator headroom.

## Input ownership

Prepare effective config, market settings and immutable candle/BTC/timestamp arrays on
the CPU. Allocate the arrays once using `SharedArrayManager`; keep the segments immutable
and alive until the service has closed. `PreparedGpuDataset` snapshots metadata and
shared-array descriptors without attaching or copying candle histories at registration.
Read-only attachments protect worker-side views; they do not make another owner's
mutable alias safe to change during a run.

Supply `candle_coins` in the actual source-column order. Optional `coin_indices` select
those columns, and their identities/order must exactly match the scenario's sorted
`config.backtest.coins[exchange]`. Optional `time_range` selects a half-open row interval.
BTC rows remain aligned with the source candles. An already selected timestamp window
can use an independent `timestamp_range`; its selected row count must match the candle
interval. Without that explicit range, timestamps retain the original full-source layout.
The CPU must prepare config dates and market validity/warmup metadata for that effective
scenario; the service does not infer missing inputs or substitute market settings.
Requested metrics specify backtest work, not scoring directions or constraint penalties.

```python
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import BacktestRequest
from optimization.gpu.native import CudaBacktestService
from shared_arrays import SharedArrayManager

# config, markets, exchange, ordered_coins, candles, btc and timestamps
# have already been prepared for this effective scenario on the CPU.
arrays = SharedArrayManager()
try:
    specs = [arrays.create_from(value)[0] for value in (candles, btc, timestamps)]
    dataset = PreparedGpuDataset(
        config=config, markets=markets, exchange=exchange,
        hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
        candle_coins=ordered_coins, metrics=["cagr_fraction_strategy_equity", "gross_turnover_ratio_per_day"],
    )
    with CudaBacktestService() as service:
        service.register_dataset("scenario", dataset)
        future = service.submit(BacktestRequest("candidate", "scenario", {}))
        result = future.result()
        # The caller scores/reduces result.metrics with result.liquidated,
        # then records complete candidates and performs selection.
finally:
    arrays.cleanup()
```

An empty parameter mapping evaluates the dataset's base strategy parameters. The
service also accepts its materialized scalar parameter mapping;
unsupported topology changes require a separately prepared dataset. The request and
dataset IDs are caller-owned, run-local identities. The optimizer owns persistent
content/evaluation fingerprints, precision stamps and resume compatibility separately.

Metric requests use the native core catalog, with definitions, clocks and statuses in
[the backtest metric contract](ai/features/backtest_metrics.md). Each replay streams a
compact selected observation packet; the ordinary Rust core finalizer owns all
finishing formulas. Python transports the packet and configuration metadata.
`BacktestResult.metrics` contains requested canonical base names with finite numbers
or `None`. `metadata.metric_status` distinguishes genuine positive/negative infinity,
undefined quantities and invalid observations; `metadata.metric_metadata` records the
definition version and observation clocks/counts. Unrequested quantities are absent.
Limits and scenario reductions decode these statuses before using raw values.

This metric ABI currently supports one-minute USD-collateral replay. BTC collateral
and coarser metric replay fail with explicit capability errors. Market data may still
contain aligned BTC reference arrays; they do not create extra metric denominations.
Integer float32 packet counters and relative clocks must fit exact 24-bit integer
representation. Overflow cannot become a rounded duration or apparently valid score.

## CPU completion scoring

`optimization.native_results` collects identified scenario/exchange results for one
candidate independently of other candidates. `ResultSlot` declares the expected dataset,
scenario, exchange and requested metrics; `CandidateEvaluation` rejects wrong identities,
duplicate results, missing metrics or unknown terminal status. Full evaluations require
all prepared scenario/exchange pairs. Screening can cover a subset, but must retain every
explicitly selected objective/limit scenario; its completion cannot be admitted as a full
evaluation through `require_full()`.

`CanonicalResultScorer` reuses the CPU optimizer's canonical metric aggregation, suite
reducers, scoring and limits, without calling either evaluator's simulation method.
Non-finite metric sentinels follow canonical invalid-candidate scoring; malformed output
raises rather than becoming a successful evaluation. Actual simulator liquidation status
is independent of requested metrics. Replay producers must return structured metrics and a Boolean simulator terminal status.
Bare metric mappings are rejected before any request in the batch succeeds.

Search updates, durable storage and resume are the orchestrator's responsibility. Prepare
a stable scorer before submitting work and do not mutate its evaluator's scoring
configuration while results are in flight.

## CPU candidate preparation and collection

`optimization.native_planning.NativeCandidatePlanner` binds a canonical CPU evaluator to
prepared scenario/exchange datasets. It applies the existing bounds, optimizer overrides,
fixed runtime policies and exact-last scenario overrides before encoding compact request
parameters. It never calls an evaluator's simulation method. Global candidate values are
separate from dataset-owned coin patches, so the first coin's patch cannot replace defaults
for every unpatched coin. Mirrored or fixed shadow genes deduplicate by effective work.
The global `unstuck_memory_ema_span_minutes` and `loss_gate_memory_ema_span_minutes`
request parameters remain independent after overrides; changing either does not
require another prepared dataset.

`prepare(candidate_id, vector, scenarios=...)` returns a candidate plan containing requests
and expected result slots. A screening plan's duplicate identity includes unscreened
scenarios: identical screenings alone do not prove identical complete candidates. Values
outside the dynamic scalar transport remain dataset-owned; changes to feature flags,
execution modes, side enablement or materialized coin patches require compatible prepared datasets and are
rejected before submission. The canonical registry prepares finite anchor and side-enable
views before registration, sharing the same borrowed source histories. Equivalent execution
contracts share a view; exact-last scenario policies can remove ineffective choices.
Candidate-dependent continuous coin patches still require transport integration.

`optimization.gpu.coin_parameters.build_coin_override_parameters` is the shared
CPU-only coin encoder for both strategies and the single/multicoin replay adapters.
It consumes canonical resolved payloads, selects explicit pins, and preserves exact
patches for identity separately from float32 transport. It does not load device or
search libraries or simulate a backtest. Consolidating encoding does not yet make
coin patches request-owned: eligibility, RMS demand, ablation and truncated replay
must all support changing pins before continuous coin transport can be enabled.

Coupled unstuck EMA spans reuse the existing scalar transport. CPU finalization
materializes candidate/scenario dependencies; execution identity excludes only redundant
derived coin span copies and retains strategy coin pins and coupling-policy identity.
Before registration, the CPU registry projects each view to backtest sections, removes
optimizer/bookkeeping metadata, and lowers inherited coin spans to ordinary parameter
inheritance. Pinned strategy spans retain their explicit unstuck counterparts. The worker
receives an ordinary backtest view and explicit candidate values; it does not interpret
optimizer coupling policy. These metadata views share the original immutable arrays.
Saved coupled suites retain explicit scenario spans for ordinary backtest replay.
Resume compares incoming scenario recipes after resolving dependencies against each
saved candidate, while retaining checks for other scenario changes and altered stored spans.

`optimization.native_session.NativeEvaluationSession` exclusively borrows the backtest
service. The caller admits plans and polls independently completed candidates, then
performs selection and prompt persistence. Candidate admission and completed-payload
caching are bounded; pending effective duplicates share work. Cached screenings cannot
satisfy a full evaluation or a different scenario subset. Request snapshots prevent caller
mutation while work is waiting for device admission. Future callbacks only enqueue
notifications; canonical scoring runs on the CPU poller, outside GPU completion callbacks.

At interruption, stop session admission, close/drain the service, then poll completed work
and flush the caller's stores. Partially completed candidates remain unfinished and may
be rerun on GPU. Producer failures stop admission and preserve the original exception.
The CPU optimizer integrates these helpers and owns persistence and compatible-resume
identities. Their execution caches are run-local.

## Input memory and admission

GPU optimizer preparation streams compressed NumPy caches to run-owned files and borrows
read-only mappings instead of duplicating the raw history in heap and shared memory. Plain
NumPy cache files remain borrowed and are never deleted by the optimizer. Descriptors retain
file identity; replacing or modifying a prepared input fails closed. The parent releases files
after service attachments close. CPU optimizer transport continues to use shared memory.

For ordinary unsliced one-minute caches, GPU memory admission checks the NumPy header before
decoding the payload. Suites, explicit dataset slices and aggregated histories are checked after
their effective dimensions are known, before price packing. These checks retain the device's
existing invariant safety budget; they do not guarantee sufficient system RAM or include all
candidate/driver scratch. Metal trailing-martingale replay keeps five strict integer tick arrays
on disk and uploads only each temporal replay range plus previous/next-candle context. The
admitted footprint includes its maximum window. One-minute hourly ranges use lossless compact
boundary rows with virtual access at their original indices; full bars and clamped
market-execution prices remain resident for historical access. Other candle intervals retain
dense hourly ranges. Smaller histories retain resident
inputs when they fit the initial device and host memory budgets. The run captures its preference
before reading cache pages and passes it to the owner thread. After packing, clean read-only
raw file pages receive an optional OS reclaim hint in both parent and worker mappings;
future reads remain valid. Large Metal device uploads copy bounded contiguous slices and
release completed source pages, avoiding an additional active copy of the packed history.
Startup upload waits bound source lifetime; normal replay synchronization is unchanged. This preserves
the existing 45% safety budget and does not alter CUDA or other strategy layouts.
Validation uses bounded conversion and mask buffers without changing
accepted candle values or the first reported invalid location.

## Canonical prepared-data binding

`optimization.native_datasets.NativeDatasetRegistry` connects an existing canonical CPU
evaluator to the service and candidate planner. Standalone callers supply actual source
identities in `standalone_candle_coins`. Suite preparation retains `ScenarioEvalContext.candle_coins`
from the actual master/source dataset, including columns outside a selected scenario.
The registry binds those identities with the canonical time/coin indices and scenario
market metadata; missing source identities fail rather than being inferred from a subset.

Existing candle/BTC shared segments are borrowed. Only compact timestamp windows are
allocated here, with equal windows reused by content. Lazy master slices and already
sliced scenario inputs use the same service interface; registration never creates a
scenario-sized candle copy. Required metric names come from CPU scoring/limits, independently
of device handles, and effective seed policies are prepared through the canonical helpers.

The registry owns only its added timestamp segments. Nest the service inside its lifetime:
register through `registry.register(service)`, prepare plans with `registry.planner`, then
drive `NativeEvaluationSession` on the CPU. Close/drain the service before closing the
registry, and keep the original candle/BTC owner alive through both. Registry cleanup
attempts every owned window and preserves an earlier caller/preparation failure. The data bridge owns neither search nor saved-fitness compatibility.

## GPU optimizer

Select `--optimizer-backend gpu` or `optimize.backend: "gpu"` on an NVIDIA
CUDA or Apple Silicon Metal installation. CPU backends remain available. The GPU path runs GPU simulations for all fresh candidates, starting configs
and unfinished resumed candidates; it never creates a CPU simulation pool or uses CPU
backtests to validate GPU fitness. CPU preparation still uses the canonical runtime compiler.

The CPU orchestrator reuses pymoo NSGA-II/III variation and survival settings from
`optimize.pymoo`, with `optimize.population_size` and `optimize.iters` retaining their
generation-based meaning. Within a cohort it replenishes bounded GPU work, scores completions
and writes full candidate records immediately through the existing results/Pareto stores.
Evolution advances after the cohort's complete evaluations finish. Effective duplicates
share pending/cached work.

At INFO, each evaluation stage reports a compact `GPU cohort` summary with its generation,
phase and completion status. `attempts/s` divides additional completed stage attempts by that
stage's wall time. It includes prefilter and cached/aliased completions; screening attempts are
partial observations, not full fitness. Restored evaluations are excluded. `gpu_rows/s` instead
counts fresh, successfully collected service request rows per stage wall second. One attempt
can require several scenario/exchange rows, while aliases, candidate-cache hits and reused
screening rows require no fresh replay. Neither rate is pure device time or necessarily a
unique-vector rate. Failed/interrupted summaries are partial snapshots taken before shutdown
draining and do not claim successful completion of the cohort.

DEBUG detail separates admitted/cache/alias attempts, submitted request rows, validated fresh
rows and reused rows. Submission does not imply success. Main-thread `prepare`, `poll` and
`consume` wall and thread-CPU spans are disjoint: preparation covers canonical planning and
admission; polling includes submission, row fan-in, scoring and any blocking wait; consumption
includes candidate application, recording and its checkpoints. `other` covers the remaining
evaluation-stage work. Thread CPU time excludes device waiting and other threads. Serial
`ask`, `tell` and scenario-promotion boundaries are reported separately at DEBUG and lie outside
evaluation-stage wall time. These stage rates exclude startup, registry/data preparation and
serial search boundaries, so they are not a whole-command throughput measurement.

Optional owner diagnostics report completed validated batches/rows, host replay plus reduction
and validation time, claimed-request accumulation, and worker/dataset preparation. These owner
spans overlap the CPU stage wall spans and must not be added to them. Replay timing adds no
device synchronization and is not a kernel-time measurement. Tuner context retains fixed-size
aggregate counts and only the first trial selection and latest decision. Elapsed
seconds begin at tuner construction and include supply gaps; cumulative row/replay totals cover
all observed dataset classes, while each class's decision still uses its own existing evidence.
The latest class snapshot distinguishes actual count, target width, ceiling, trial state,
retained warm samples and duration thresholds. Trial selection is a future target, not proof
that a full target-width batch has executed. Capacity changes may invalidate timing evidence
without removing the completed request rows from the diagnostic totals.

Diagnostics retain no candidate history, device handles or checkpoint state and never supply
tuning, admission, cadence, screening or search decisions. INFO-enabled native optimization
enables the optional host timers; a generic fixed-policy execution service remains untimed by
default. Diagnostic snapshots and console failures cannot replace authoritative execution errors.

`optimize.gpu.screening` uses the existing scenario labels, survival fraction and minimum
survivor settings. Seeds and initial parents receive full-suite GPU evaluation. Later
offspring can be screened on a scenario subset; CPU feasibility/Pareto-diversity selection
promotes survivors to full-suite evaluation. Only complete survivors enter evolutionary
survival alongside the complete parents. Partial scores never enter fitness, stored results
or Pareto. Screening must retain explicit objective/limit scenarios. Unknown labels fail
before the device starts. Selecting every scenario or retaining every offspring bypasses
the partial stage. Screening reduces full evaluations within the configured generation
budget; it does not extend `iters` to compensate for rejected offspring.

The CPU session also retains bounded simulator-row evidence by complete effective candidate
identity, prepared dataset and exact request parameters. Screen-to-full promotion can reuse
already simulated scenarios; it still collects and validates every required full-suite slot before canonical scoring and
storage. A partial score is never reused as complete fitness. Reused rows receive the current
request identity and are consumed on the CPU poller. Future results must match their actual
submitted request before entering either collection or this cache. Each candidate-payload and
simulator-row LRU is independently limited by `cache_size`; eviction or loss only repeats GPU
work. These snapshots are run-local, without device handles or checkpoint state.

GPU requests are submitted after the first prepared candidate. The CPU session advises
the service which phase datasets can receive more work without waiting for a completion;
this hint reserves no requests or capacity and does not contribute to width-tuning demand.
Preparation and CPU result servicing then alternate within
a soft 50 ms latency target, with interruption checks between candidates. Single preparation,
scoring or storage operations can exceed that target; it is not a hard deadline. Completion
grouping starts at one candidate and grows from observed CPU work, up to 256, with immediate
reduction after increased cost. Device waiting time does not count as CPU throughput evidence.
This cadence is run-local and is not part of saved fitness or search checkpoints.

`NativeEvaluationSession.poll` independently bounds device notifications (`max_results`)
and returned full candidates (`max_completions`, defaulting to `max_results`). Large suites
can consume many notifications per candidate, while cached duplicates can produce many
completions from one notification. A small persistence batch therefore does not throttle
suite fan-in; ready aliases remain retained until their next CPU consumption call.

The native integration uses `optimize.gpu.batch_size`, `tuning_mode`,
`max_dispatch_candidate_bars` and `checkpoint_interval_seconds`. An omitted/null or `auto`
batch setting enables service-owned batch tuning in `auto`/`refresh` mode. A positive
explicit setting disables width tuning; `tuning_mode: "off"` uses a fixed ceiling of 64
when no width is supplied. These are dispatch ceilings, also bounded by prepared work
and history-scratch limits. CPU candidate admission is bounded by its population window,
independently of GPU widths; the service bounds its own queued-plus-running requests.
Legacy exact-worker, drift-probe, validation and screened-seed controls do not apply.

Request accumulation adapts independently of width in `auto`/`refresh` mode. The
service retains the last 32 within-active-work arrival gaps for each dataset, including
gaps between CPU preparation bursts. A doubled 90th-percentile gap supplies the idle
tail; successful replay duration supplies the total allowance, capped at 0.5 seconds.
Scheduling learns from the first validated replay, including first-use setup cost;
a new batch count does not restart that learning. Until completed timing and three
gaps exist, the fixed 5 ms accumulation policy remains.
This is a bounded heuristic, not a guarantee of optimal scheduling for every workload.

While the CPU producer can prepare more work, automatic accumulation can continue beyond
the ordinary 0.5-second absolute limit. Its allowance is 1.5 times successful replay
duration, capped at 300 seconds from the original accumulation start. Before any replay
has completed, an explicit producer hint receives a 60-second startup allowance. Readiness
remains authoritative through individual CPU preparation or collection pauses.
A stale hint can spend the whole allowance,
so the exclusively borrowing producer must clear it when preparation cannot continue.
Producer hints need no arrival-gap samples. Exhausted input and full CPU admission
windows seal preparation and release partial work promptly. An unrelated queued dataset
outside the active preparation set retains the ordinary short bound. These internal bounds
require no configuration knobs.

The native GPU facade's default `max_batch_delay=None` selects that policy. A numeric service
delay, including zero, retains fixed waiting; tuning-off mode defaults to 5 ms. An
explicit width can therefore remain fixed while accumulation adapts. Already buffered
work may dispatch immediately when its stream has stalled. Full queues, completed widths,
closing and cancellation retain their existing dispatch/drain behavior. This advisory
state is service-local; it changes neither submitted simulations nor saved search state.

The first request for an unprepared dataset claims one candidate, then prepares and discovers
its physical dispatch limit on the GPU owner. Before simulation, the worker fills that
first dispatch with already queued compatible requests up to the prepared limit. It does
not wait for more arrivals during this fill; cancelled queued work stays excluded. Subsequent
batches use that limit. This prevents several serial scratch/work splits from delaying
the return of an oversized outer batch. Preparation retains no inactive runner references;
switching datasets can release their tensors and mutable scratch.

Automatic Metal widths start at the existing replay cap of 512 candidates, clamped by the
prepared memory/history ceiling and service admission limit. CUDA starts at 64. The GPU
owner selects the device before creating a prepared controller; producer-side construction
remains device-independent. Explicit widths and
tuning-off defaults retain their fixed limits. Separate per-dataset
controllers measure successful replay, reductions and host results using actual candidate
counts, including partial request cohorts. Failed work and replays reporting a compiled
library cache miss do not contribute to throughput trials. Warm new counts contribute
immediately; compilation evidence aggregates feature variants and nested replay work
without enabling profiling or adding synchronization. Callers without reported compilation
evidence retain first-use rejection per count: an empty collector means unknown, not warm.
Diagnostics distinguish reported compilation from count-history rejection with `cold_source`
and keep compilation-evidence availability separate from controller readiness.
A window requires six successful warm
observations and three seconds of completed work, or three observations totaling at least
30 seconds with a median replay duration of at least ten seconds. The median requirement
prevents an isolated stall from shortening a fast workload's smoothing window. Timing
samples are grouped by actual candidate count. Each group retains up to six durations;
its median duration is weighted by its number of warm occurrences. Total actual candidates divided by the sum of
those weighted durations supplies the decision rate. This smooths stalls within like-sized
work without giving a singleton the same influence as a nearly full batch. A rarely seen
count has fewer samples to smooth. The decision rate is a robust replay-cost estimate,
not measured end-to-end optimizer throughput. The growth threshold, smaller-plateau
preference, cooldown and rollback apply. Growth also requires
observed request demand and device memory headroom. Metal growth also requires available
RAM reported by the OS to meet the larger of 256 MiB and 10% of physical RAM. Missing
or invalid OS telemetry denies a growth trial without failing completed replay work.
A growth trial is bounded by the largest compatible demand observed in that evidence
window, so it can try an intermediate
width when demand is less than twice the current width. This demand bound does not raise
the prepared physical ceiling. When growth is blocked, the service
can probe a smaller width using subsequent submitted work; a slower trial rolls back. This prevents deduplication,
screening or a bounded producer from indefinitely starving the tuner of evidence or
smaller-width experiments. Tuning submits no extra simulations and never changes precision
or search policy. Measurements are run-local; `auto` and `refresh` currently both start
fresh. Persistent advisory calibration, richer workload classes, residency budgets and
further CPU/evolution cadence experiments remain development work. This policy does not
claim a globally optimal width or a representative optimizer speedup.

Native CUDA temporal replay separately adapts history-chunk length from completed command
durations, targeting one second within existing candle/work/history ceilings. It shrinks
slow chunks and grows only after repeated fast full chunks; partial tails do not train it.
The controller resets for each physical replay and submits no extra simulations. Replay
state survives each boundary, and only complete accepted simulations produce results.
This internal duration control remains active with `tuning_mode: "off"`; that setting fixes
batch-width and accumulation policies. CPU result-consumption cadence is also independent.
One candle or a first command can exceed the duration target, so it is not a preemption or
worst-case cancellation guarantee.

Native checkpoints contain CPU search state and a partially evaluated cohort, without
service/device handles or shared-memory names. SIGINT stops admission, drains completed
work and preserves a checkpoint; completed compatible fitness is retained while unfinished
candidates are rerun on GPU. Checkpoints are replaced atomically at the configured interval
and at cohort/shutdown boundaries. A crash between a result write and a checkpoint may
cause some GPU work and result rows to be repeated after resume. Archive/progress counts
therefore include recorded attempts, rather than guaranteeing unique candidates after a
hard crash. Native evolution restores the checkpoint's cohort and generation state;
archive row counts do not determine its generation budget. Perfect replay of scheduling
is not required, and the result log, Pareto files and checkpoint are not a single transaction.
An initial zero-result checkpoint can resume before its first completed seed/candidate.
Version 3 retains compact partial screening evidence separately from fitness
and distinguishes screening, promoted full evaluation and idle stages. Row-cache loss may
repeat GPU work, while already checkpointed selection progress is retained. Earlier
experimental native and screening/validation checkpoints require a fresh run; saved result
configs can supply seeds.

Saved native fitness has an explicit device execution/precision identity alongside the
canonical data, policy, source/dependency and verified Rust identities. It cannot reuse CPU
or old GPU proxy/validation fitness. The current replay uses f32 state, integer tick boundary
encodings and f64 host preparation/metric work. Metal additionally uses f32 EW moments
and resident weighted-history reduction into compact daily summaries for f64 host analysis.
Full weighted histories never transfer to the CPU. CUDA/Metal fitness and checkpoints
are distinct. Changes to that contract require fresh
evaluation; the existing strict config resume checks also remain in force.

Host f64 ADG evaluates its power separately for each candidate, preserving the current
terminal smoothing and clamping formula. CPU tensor batch sizes and vector tails must
not change reported growth or its dependent metrics. Device-side reductions retain
their declared precision.

Native checkpoints retain fine-tune anchor definitions and restore them before optimizer
shape construction, without requiring the original seed files. Changed fixed anchor values
invalidate saved fitness. Older experimental anchored checkpoints without a stored plan
cannot restore their anchors automatically; use saved configs as seeds for a fresh run.

Finite anchor and side-enable choices use registered compatible execution views without
copying candle histories. Numeric candidate values still travel in compact requests.
Continuous changes to dataset-owned coin patches remain explicit errors. Search spaces
with a reachable both-sides-disabled execution view are rejected before evolution;
constrain at least one side to remain enabled. EMA Anchor position-exposure enforcers
on an enabled side, including coin overrides, require the CPU backend. Measured parity, specialized/general equivalence and resource limits are recorded in the
acceptance map. They certify the stated cases rather than every possible trajectory.

Prepared market execution prices use tick ABI 1. Rust selects SELL/BUY counts from the
original f64 closes and slippage before f32 packing; active counts must be uniquely
representable positive int32 values. Preparation rejects unrepresentable active prices.
Each candle/coin adds two int32 values (eight bytes). Packing remains chunked, with bounded
conversion scratch. Original f64 slippage and the ABI participate in immutable packing and
residency identity. Prepared execution contract version 6 rejects older layouts; source
and evaluation identities also prevent reuse of incompatible saved fitness.

Each replay attempt packs its current end-step limits with the small readonly sizes
prefix in a fresh control tensor. This keeps stop limits current when a batch shape is
reused and keeps the Metal argument count within the supported limit. Device admission
includes this control tensor as well as the execution tick pairs.

## Profiling time and launch shapes

`PASSIVBOT_GPU_PROFILE=1` enables optional replay phase profiles. All fields in
`timings_seconds` use host-wall clocks. The retained `kernel_execution` key includes
submission, synchronization waits and replay-loop work; it does not measure GPU-active
time. `timing_measurement` states this scope explicitly. Phase profiling already adds
synchronization, so use separate unprofiled runs for throughput comparisons.

For temporal replay, one additional host-clock read per chunk separates the kernel-call
submission interval from the existing synchronization wait. The totals appear in each
physical replay's `temporal_dispatches[].dispatch_host_timings_seconds` and in
`kernel_execution_host_subtimings_seconds`, alongside the remaining replay host-wall
time. These are children of `timings_seconds.kernel_execution`: never add the children
to that parent or to the phase total. Submission can overlap GPU work, and a wait can
include driver and scheduling costs, so neither child is a hardware kernel duration.
History sub-batches retain their separate physical records and actual dispatch counts.
`dispatch_count_scope` identifies kernel dispatches, while
`actual_dispatch_batch_sizes_scope` identifies physical candidate batches. Temporal
chunks partition each batch's history; they do not multiply completed candidates or
the reported candidate-bar work.

`dispatch_launches` records requested thread and threadgroup shapes, calculated group
counts and available compiled-pipeline limits. Missing limits or implicit group sizes
remain null. These describe the launch, not measured occupancy, register pressure or
bandwidth. Metadata is read only when profiling; replay geometry is unchanged. The
host-timing split adds no GPU events, device timestamps or synchronization calls.

## Execution and cleanup

The first accepted request creates one device residency context on its owning worker.
Dataset attachment, subset preparation, replay construction, evaluation and cleanup
belong to that worker. Unused registrations and an unused service do not initialize GPU
dependencies. Repeated requests reuse their replay and compatible immutable packing.

The current policy keeps one active dataset's tensors and one replay's scratch on the GPU.
Other packed inputs and reusable coin subsets live in run-local disk files. Scenario
switches evict device buffers while keeping reusable immutable packing. Initial invariant
inputs must fit the existing 45% free-VRAM (CUDA) or remaining recommended working-set
(Metal) budget; scratch and other allocations can
still fail and propagate. This is a bounded device-residency foundation, not a complete
host/disk admission budget or adaptive multi-device scheduler.

Core observations use the declared post-warmup trading boundary, before its first
fills, and include every supported inactive minute through the factual terminal mark.
The initial mark does not wait for the first trade or an available position slot.
Daily return moments use complete UTC closes; minute path and exposure moments
include the baseline. Same-clock terminal replacement counts once and preserves a
completed midnight close for daily returns. A fill-terminal valuation uses the latest
completed marks plus actual executions, without reading the later candle close.

The [metric contract](ai/features/backtest_metrics.md) defines 365.25-day annualization,
zero return threshold, corrected Sortino, pre-fill raw-balance turnover, and controller
HSL episode losses. Reported RED includes enforced flat cooldown. Effective enabled
zero-slot scopes remain observed and inactive; disabling every admitted effective
policy is distinct from capacity preventing trades. Metric observation never changes
controller permission, order arithmetic or account sizing.

Admission bounds queued plus running work. The oldest waiting request chooses the next
dataset; compatible queued requests form a bounded microbatch in their original relative
order. Other datasets retain their relative queue order, preventing starvation by newly
arriving work. A full queue dispatches available compatible work without waiting for more
admission. Each request receives an identified future rather than a generation-wide barrier.
`close()` drains accepted work and joins the owner. `close(cancel_pending=True)` cancels
queued work and waits for the running dispatch. An optional interrupt callback is checked
by replay execution to stop at its safe boundaries. Device, preparation and interrupt
failures fail admitted work and stop new admission; no CPU fallback supplies results.
Cleanup closes attachments and removes run-local packing/subset files, preserving an
original failure if cleanup also fails.

CUDA multicoin trailing-martingale histories above 8,192 candles use temporal
dispatches for either one side or a shared account. The complete replay state and
requested accumulators survive each boundary; partial chunks never become results.
The interrupt callback runs between synchronized chunks. Candle/work caps bound
dispatch size, not elapsed time for every candidate's strategy workload.

The transport's optional service-owned batch policy runs on its execution owner. It selects
widths within the fixed dispatch ceiling and observes only completely validated producer
results. Width changes occur between dispatches; FIFO dataset choice, cancellation,
backpressure and fail-stop producer semantics remain independent of tuning.

The shared-account engine permits 1..64 selected coins and supplies the authoritative
`gpu` optimizer backend. V9 numerical qualification uses this engine for single-coin and
multicoin workloads. Retired single-coin proxy classes and directional
EMA/TM runner constructors reject execution with a capability error directing callers
to `NativeBacktestService`, including one-coin datasets. Compiler specialization and adaptive dispatch select execution
policy within the native service.

Shared-account EMA/TM dispatches prove whether any packed candidate and effective
coin-side override can use unstuck EMA gating. When none can, compilation removes
only its EMA band, initialization, per-candle updates and gating consumers. Ungated
unstuck selection and ordinary strategy EMAs retain their existing policies;
continuous realized memory remains required when the realized-loss gate is active. Finite coin flags override candidate flags; nonfinite overrides
inherit them. The feature bit belongs to the compilation/replay-state identity,
independently of batch scheduling or search policy. Both fused sides participate
in the proof, and each replay reconstructs its state for the selected layout.

Shared-account EMA and Trailing Martingale auto-unstuck use continuous decaying
net-realized-PnL memory with `bot.portfolio.unstuck.memory_ema_span_minutes`. The global loss
gate maintains its own realized D with `bot.portfolio.loss_gate.memory_ema_span_minutes`.
Every ordinary, protective or unstuck fill contributes its net economics to each
enabled memory, including signed fees and rebates, while the economic ledger records
the fill once. Both memories exclude UPNL and survive HSL episode boundaries. The
gate uses `bot.portfolio.loss_gate.max_drawdown_fraction` directly in the allowance formula,
without a TWEL multiplier. Its configured
enablement is preserved separately from float32 packing, including fractions that
round to 1. The independent rolling window and its history scratch storage are
retired; old zero/all-history windows require explicit semantic migration.

Trailing Martingale finalizes executable close quantities before reserving projected
negative PnL across coins and both sides. It prioritizes finalized protective reducers,
then ordinary closes; projected profits cannot fund another emitted loss. Unfilled
emitted orders also reserve allowance, panic closes are exempt, and fills consume
admitted intent without repeating the loss check at a changed candle price or balance.
The disabled loss gate compiles out its admission state and work. Native single-coin
requests use this shared-account engine as well.

GPU HSL time-in-red reporting includes both current panic and terminal cooldown.
Reporting state is separate from the panic tier used by the simulation; counting
cooldown does not extend panic orders. Shared-account EMA/TM replay integrates
elapsed steps using the preceding observation's portfolio RED state, matching CPU
elapsed-time reporting. A flat execution observes its updated portfolio tier at the
fill timestamp before the following bar interval; GREEN at that boundary adds no
unobserved RED minute. First observations add no duration, terminal states add no
unobserved tail, and temporal dispatches retain the complete clock. Controller-active parity checks remain required to
assess observation timing and numerical discrepancies.

### Forward risk replay

The [v9 risk model](v9_risk_model.md) defines the CPU/device arithmetic and event contract.
Each coin/side, side or portfolio controller retains constant-size currency drawdown,
ratio EMA, causal mark and integer clock state. Ordinary replay never reconstructs a
historical fill/candle window after a fill, budget change or query. Required account and
objective histories remain separate from this risk state.

At each execution, the traded symbol's supported execution price becomes its causal
mark for both sides. Assess the old inventory, consume that fill's fee/PnL and inventory
change once, then assess the resulting inventory. Other symbols retain their latest
causal marks. Completed candle closes become observable at the candle end; forced
end-of-bar delisting follows that completed observation. No later candle close may
become an earlier fill observation.

The same authored f32 graph serves Metal and CUDA. Implicit contraction is disabled;
only explicit `fma` operations fuse. Slow and fast span coefficients must remain finite
and decaying after f32 conversion. Finite nonpositive equity is an immediate RED branch;
malformed or overflowing required economics fail the request without replacement metrics.

A flat event latches finite cooldown when the latest exposed pre-fill signal OR the
post-accounting signal is RED. Queries and later budget/mark changes cannot renew or
reclassify that deadline. There is no `never` restart policy. Configured slot counts and
authored headroom continue to determine coin budgets independently of held positions.

The canonical cooldown duration is rounded from f64 minutes to integer milliseconds.
Device transport uses two exact 24-bit float limbs, preserving carries and fractional
minute boundary decisions without converting the duration to f32 minutes. Durations
above `2^48 - 1` milliseconds are an explicit GPU capability error. Temporal dispatches
retain complete risk state and reconstruct only pointer bindings, never risk history.

Run the bounded source-verified native scope regressions with
`pytest tests/optimization/test_gpu_v9_native.py`. Risk-only actual-device witnesses are
in `tests/optimization/test_gpu_hsl_decay.py`; host policy transport tests are in
`tests/optimization/test_gpu_v9_transport.py`. These cases do not establish broad device
coverage or search speed. CUDA qualification requires actual CUDA execution separately
from Metal and CPU evidence.
