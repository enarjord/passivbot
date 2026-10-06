# Prepared CUDA backtest service

`optimization.gpu.native.CudaBacktestService` is an internal execution interface under
development. CPU code registers prepared scenarios, submits identified backtest requests
and receives futures containing compact metrics and actual simulator liquidation status.
Device buffers, packing, replay handles
and residency stay inside the service. No CPU backtest or evolutionary algorithm runs
there. An experimental optimizer integration is available below; practical simulation-parity
acceptance and replacement of the legacy GPU backend remain open.

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
        candle_coins=ordered_coins, metrics=["adg_strategy_eq", "fills_per_day"],
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
transitional replay adapter also accepts its materialized scalar parameter mapping;
unsupported topology changes require a separately prepared dataset. The request and
dataset IDs are caller-owned, run-local identities. The optimizer owns persistent
content/evaluation fingerprints, precision stamps and resume compatibility separately.

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
is independent of requested metrics. The legacy metric-only replay API remains available
but cannot supply authoritative completion status.

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
These helpers do not integrate the service into the optimizer CLI, persist results or
define content/precision identities for compatible resume. Their caches are run-local.

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
attempts every owned window and preserves an earlier caller/preparation failure. This
data bridge does not add optimizer CLI routing, search or saved-fitness compatibility.

## Experimental native optimizer

Select `--optimizer-backend gpu_native` or `optimize.backend: "gpu_native"` on an NVIDIA
CUDA installation. CPU backends and the existing `gpu` screening/validation backend remain
available. The native path runs GPU simulations for all fresh candidates, starting configs
and unfinished resumed candidates; it never creates a CPU simulation pool or uses CPU
backtests to validate GPU fitness. CPU preparation still uses the canonical runtime compiler.

The CPU orchestrator reuses pymoo NSGA-II/III variation and survival settings from
`optimize.pymoo`, with `optimize.population_size` and `optimize.iters` retaining their
generation-based meaning. Within a cohort it replenishes bounded GPU work, scores completions
and writes full candidate records immediately through the existing results/Pareto stores.
Evolution advances after the cohort's complete evaluations finish. Effective duplicates
share pending/cached work.

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

GPU submission starts after the first prepared candidate rather than waiting for the
entire CPU admission window. Preparation and CPU result servicing then alternate within
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
tail; successful warm replay duration supplies the total allowance, capped at 0.5
seconds. The first replay of each actual batch count is excluded from warm evidence.
Until warm timing and three gaps exist, the fixed 5 ms accumulation policy remains.
This is a bounded heuristic, not a guarantee of optimal scheduling for every workload.

The CUDA facade's default `max_batch_delay=None` selects that policy. A numeric service
delay, including zero, retains fixed waiting; tuning-off mode defaults to 5 ms. An
explicit width can therefore remain fixed while accumulation adapts. Already buffered
work may dispatch immediately when its stream has stalled. Full queues, completed widths,
closing and cancellation retain their existing dispatch/drain behavior. This advisory
state is service-local; it changes neither submitted simulations nor saved search state.

The first request for an unprepared dataset claims one candidate, prepares and discovers
its physical dispatch limit on the GPU owner, and returns that completion. Subsequent
batches use that limit. This prevents several serial scratch/work splits from delaying
the return of an oversized outer batch. Preparation retains no inactive runner references;
switching datasets can release their tensors and mutable scratch.

Automatic widths start at the smaller of 64 and the prepared ceiling. Separate per-dataset
controllers measure successful replay, reductions and host results; failed work, partial
tails and each width's first cold use do not contribute to throughput trials. The existing
24-sample/30-second evidence window, median smoothing, growth threshold, smaller-plateau
preference, cooldown and rollback apply. Growth also requires observed request demand and
device memory headroom. Tuning submits no extra simulations and never changes precision
or search policy. Measurements are run-local; `auto` and `refresh` currently both start
fresh. Persistent advisory calibration, demand-limited classes, dispatch duration/delay,
residency budgets and further CPU/evolution cadence experiments remain development work. This policy
does not claim a globally optimal width or a representative optimizer speedup.

Native checkpoints contain CPU search state and a partially evaluated cohort, without
service/device handles or shared-memory names. SIGINT stops admission, drains completed
work and preserves a checkpoint; completed compatible fitness is retained while unfinished
candidates are rerun on GPU. Checkpoints are replaced atomically at the configured interval
and at cohort/shutdown boundaries. A crash between a result write and a checkpoint may
cause some GPU work to be repeated after resume. Perfect replay of scheduling is not required.
An initial zero-result checkpoint can resume before its first completed seed/candidate.
Version 2 additionally retains compact partial screening evidence separately from fitness
and distinguishes screening, promoted full evaluation and idle stages. Row-cache loss may
repeat GPU work, while already checkpointed selection progress is retained. Earlier
experimental native checkpoints require a fresh run; saved result configs can supply seeds.

Saved native fitness has an explicit CUDA execution/precision identity alongside the
canonical data, policy, source/dependency and verified Rust identities. It cannot reuse CPU
or old GPU proxy/validation fitness. The current replay uses f32 state, integer tick boundary
encodings and f64 host preparation/metric work. Changes to that contract require fresh
evaluation; the existing strict config resume checks also remain in force.

Native checkpoints retain fine-tune anchor definitions and restore them before optimizer
shape construction, without requiring the original seed files. Changed fixed anchor values
invalidate saved fitness. Older experimental anchored checkpoints without a stored plan
cannot restore their anchors automatically; use saved configs as seeds for a fresh run.

Finite anchor and side-enable choices use registered compatible execution views without
copying candle histories. Numeric candidate values still travel in compact requests.
Continuous changes to dataset-owned coin patches remain explicit errors, and candidates
with both sides disabled remain unsupported by the replay. Representative parity,
specialized/general kernel equivalence, performance acceptance and adaptive tuning remain
open; the native backend does not supersede the legacy backend yet.

## Execution and cleanup

The first accepted request creates one CUDA residency context on its owning worker.
Dataset attachment, subset preparation, replay construction, evaluation and cleanup
belong to that worker. Unused registrations and an unused service do not initialize GPU
dependencies. Repeated requests reuse their replay and compatible immutable packing.

The current policy keeps one active dataset's tensors and one replay's scratch on CUDA.
Other packed inputs and reusable coin subsets live in run-local disk files. Scenario
switches evict device buffers while keeping reusable immutable packing. Initial invariant
inputs must fit the existing 45% free-VRAM budget; scratch and other allocations can
still fail and propagate. This is a bounded device-residency foundation, not a complete
host/disk admission budget or adaptive multi-device scheduler.

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

The transport's optional service-owned batch policy runs on its execution owner. It selects
widths within the fixed dispatch ceiling and observes only completely validated producer
results. Width changes occur between dispatches; FIFO dataset choice, cancellation,
backpressure and fail-stop producer semantics remain independent of tuning.

The shared-account engine now permits 1..64 selected coins. This facade uses that
implementation internally; legacy optimizer routing is unchanged. Short synthetic
one-coin measurements show a substantial throughput disadvantage against the old
single-coin implementation. Kernel ablation and representative measurements are required
before selecting the final native optimizer's default execution policy.

Single-side multicoin EMA replay selects a compact HSL layout only when every request
in the dispatch disables HSL, no effective coin override can enable it, and all signal
modes are side or unified. The specialized kernel removes controller/history binding
and per-candle HSL scans but retains aggregate forced-delisting loss diagnostics. Coin
mode retains full state even when HSL is disabled: its separate panic segments affect
reported drawdown reductions. Fused portfolios and Trailing Martingale retain their
existing layouts. Execution scheduling does not decide this semantic specialization.

Shared-account EMA realized-loss admission and auto-unstuck use one bounded fill-PnL
window selected by `live.pnls_max_lookback_days`. The history is prepared when either
consumer is enabled and compiled out when neither needs it or the scope is all history.
Loss-only requests retain the window even with auto-unstuck and HSL disabled. Shared
long/short cash and generation-time loss reservations remain unchanged. HSL histories
and Trailing Martingale's conservative realized-loss policy are separate. Native
single-coin requests use this shared-account engine as well.

GPU HSL time-in-red reporting includes both current panic and terminal cooldown.
Reporting state is separate from the panic tier used by the simulation; counting
cooldown does not extend panic orders. The GPU reduction still uses sampled bars,
whereas CPU reporting integrates elapsed scope-state time. Controller-active parity
checks remain required to assess observation timing and numerical discrepancies.
