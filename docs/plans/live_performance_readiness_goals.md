# Live Performance And Readiness Goals

This is the working checklist for making live execution approach the backtest
ideal: complete inputs available at the decision boundary, fast deterministic
planning, and prompt exchange writes when trading logic says action is needed.

Backtest remains the benchmark. Live cannot be identical because it depends on
exchange APIs, network latency, rate limits, cache repair, order confirmation,
and partial/stale data. The goal is to measure every gap, reduce avoidable
latency, and make unavoidable latency explicit in the event stream.

## How To Use This Checklist

This document is the action list for live performance and readiness work. Each
item should end in one of three outcomes:

- a measured baseline in `passivbot tool live-performance-report`;
- a behavior-preserving optimization with before/after timing evidence; or
- a documented readiness contract with tests proving that fast startup does not
  weaken trading correctness.

Any optimization that changes which order classes are allowed to proceed must
state the readiness contract explicitly. Speedups are acceptable only when they
preserve the existing trading decision semantics or when the contract is
deliberately changed and reviewed.

Use this as a living performance scorecard:

- [ ] Each observed slow path has a named owner section below.
- [ ] Each performance PR updates this file with baseline, target, and result.
- [ ] Each optimization proves whether it affects protective action, fresh
  entry, diagnostics only, or no trading behavior.
- [ ] Each live smoke leaves enough structured evidence to compare against the
  previous baseline.
- [ ] If an item is delegated, the delegate works from one checked subsection,
  opens a PR, and does not touch unrelated live behavior.

## Actionable Goal Checklist

This is the implementation checklist for the performance/readiness goal. The
rest of this document gives evidence, target contracts, and candidate PR
slices.

### Goal 1: Make The Slow Path Measurable

### Goal 2: Remove HSL Broad Replay From The Protective Critical Path

- [x] A held `coin+pside` must not wait behind unrelated flat coins before its
  exact HSL state is known.
- [x] HSL startup must expose separate states for account-critical ready,
  held-position protective ready, fresh-entry/cooldown eligibility unknown,
  and full replay ready.
- [ ] Current drawdown state takes precedence: a historical red crossing must
  not trigger a new panic if the exact current state is no longer red.
- [x] Full historical/cooldown reconstruction may continue after held positions
  are protectively ready, but fresh entries remain blocked for symbols whose
  cooldown/trading eligibility is still unknown.
- [ ] Acceptance: with 25-30 configured pairs and one held position,
  held-position protective readiness is reached in seconds, not tens of
  minutes, when required local/exchange proof is present.

- [ ] Measure cache discovery, cache decode, fill indexing, candle/timeline
  materialization, pair iteration, EMA/drawdown update, event emission, and
  exchange/backfill time separately.
- [ ] Determine whether coin-mode replay is CPU-bound Python, disk/cache IO,
  exchange backfill, repeated data conversion, or unnecessary serial dependency
  between independent pairs.
- [ ] Confirm whether all needed data was already cached in the slow Binance
  XLM incident path; if yes, explain exactly why local replay still took about
  27 minutes.
- [x] Add an offline deterministic fixture so rows/s, stage timings, and
  equivalence can be checked without live exchange access.
  - Result: `passivbot tool hsl-replay-benchmark` replays a bounded in-memory
    coin-HSL fixture through the current initializer, with distinct profiled
    timeline-rows/s and pair-rows/s, per-stage timing, counter, fixture-hash,
    final-state-hash, and side-effect-counter output. The active sparse-replay
    slice extends this to a 43,201-minute, 30-pair fixture with held and
    historical-flat pairs, cooldown/panic transitions, same-timestamp fills,
    account-balance changes, EMA smoothing, and exact dense-reference state and
    sample-count comparison. The stage-profile slice adds exclusive candidate,
    dense-reference, equivalence-comparison, fixture, history-load,
    held/background sample, current-UPnL, projection, and residual-orchestration
    timings. Production cache discovery/decode, exchange backfill, and event
    emission remain outside this in-memory benchmark and still require separate
    evidence. On the exact 43,201-minute/30-symbol fixture, the compact
    candidate run took `2.899s` and the dense-reference run took `156.390s`;
    `147.403s` of reference full-replay time was residual orchestration after
    `5.873s` of measured held/background metric sampling. Candidate and
    reference fixture contract, sample counts, and final state all matched.
- [ ] Acceptance: before optimizing, the report identifies the dominant cost
  category and provides a repeatable local benchmark.

### Goal 6: Make Warm Restart Fast But Proven

- [ ] Short downtime should not cause broad HSL, candle, or fill
  reconstruction when coverage proof is still valid.
- [ ] Warm restart should use proven canonical fill/candle cache state before scheduling broad
  repair, then reconstruct HSL authoritatively.
- [ ] A stale or missing proof for one symbol should trigger targeted repair,
  not a broad stall for every unrelated held position or forager candidate.
- [ ] Acceptance: a quick restart after a clean shutdown reuses valid local
  state and reaches protective readiness much faster than cold start.

### Goal 7: Make Shutdown Fast And Diagnosable

### Goal 8: Keep Forager Readiness Fast Without Random Fresh-Subset Bias

- [ ] Refresh the stalest eligible forager symbols regularly in the background.
- [ ] Allow bounded staleness for candidate ranking without disqualifying a
  coin merely because it was not among the freshest arbitrary subset.
- [ ] For stale-but-within-cap candidates, close EMA readiness may use bounded
  flat-close projection, while quote-volume and log-range ranking should carry
  forward latest known EMA values with age/source metadata.
- [ ] Candidates with no prior feature basis, non-finite carried values, or age
  beyond the cap are explicitly unavailable until refreshed.
- [ ] Acceptance: forager selection is both rate-limit friendly and
  non-random; stale candidate state is observable and does not weaken actual
  entry readiness.

### Goal 9: Keep Speed And Correctness Boundaries Explicit

- [ ] Reports, probes, and cache doctors may expose gaps, but must not enforce
  trading decisions unless a separate behavior PR changes the contract.
- [ ] Any readiness fallback used by live trading must be bounded, observable,
  and covered by tests.
- [ ] No neutral defaults for missing HSL, fill, candle, account, EMA, market,
  or cooldown proof.
- [ ] Acceptance: every speedup PR states whether it affects protective action,
  fresh entries, diagnostics only, or no trading behavior.

## Definition Of Done

## Current Evidence

Evidence source: VPS5 monitor/smoke data collected on 2026-06-27 while the
logging/performance report work was being merged through `v8`. Treat the
specific timings below as incident and baseline evidence, not as a guarantee
that the latest local head has been re-profiled after every subsequent
observability-only merge.

Latest incident driver: Binance `hsl_signal_mode=coin` XLM panic on
2026-06-26, with fill-event timestamp `1782492486000`. The observed startup
path spent roughly 27 minutes in HSL history reconstruction before the
protective close was posted. That is a safety-critical performance failure even
if the final replay result is correct.

3. Authoritative state refresh timings are already observable.
   - Hyperliquid staged refresh summary over 18 samples:
     `wall_ms min=2498 mean=3655 max=4166`; `surface_max_ms min=2497 mean=3549
     max=3997`; `surface_sum_ms min=3985 mean=6019 max=6620`.
   - Hyperliquid surface timings over the same summary:
     `positions_balance min=2497 mean=3549 max=3997`; `open_orders min=1488
     mean=2470 max=2769`.
   - Kucoin one startup/account refresh sample:
     `wall_ms=9600`, `surface_max_ms=9585`, `surface_sum_ms=31149`; surfaces
     were `balance=7121ms`, `positions=7210ms`, `open_orders=7233ms`,
     `fills=9585ms`.

## Required Performance Report Matrix

The live performance report should become the canonical answer to "where did
the time go?" for a live bot. Every row below should expose `count`, `min`,
`mean`, `p50`, `p95`, `max`, latest timestamp, bot identity, and trading-impact
classification when enough source events exist.

Minimum report questions the operator must be able to answer:

- [ ] How long after process start could the bot safely panic-close each held
  position?
- [ ] How long after process start could the bot safely place fresh entries?
- [ ] Which exact input or phase delayed the first possible protective action?
- [ ] Which exact input or phase delayed the first possible fresh entry?
- [ ] Was a delay caused by exchange/network IO, local cache proof, local CPU,
  disk IO, Python replay logic, Rust planning, exchange write/confirmation, or
  monitor/event-pipeline overhead?
- [ ] Did any slow background task share the critical path with protective
  actions when it should have been decoupled?
- [ ] For each restart, did warm local cache/checkpoint proof actually reduce
  startup time, or did the bot repeat broad reconstruction unnecessarily?

Trading-impact labels:

- [ ] `protective_blocker`: can delay panic/reduce-only/risk protection.
- [ ] `entry_blocker`: can delay fresh entries but not protective actions.
- [ ] `cycle_delay`: delays the full loop after readiness is established.
- [ ] `diagnostics_only`: affects logs/monitor/reporting only.
- [ ] `unknown`: missing event data; should be treated as an observability gap.

## Outcome Targets

- [ ] A held position should reach exact protective readiness in seconds, not
  minutes, after process start.
  - Initial target on the VPS5 1-vCPU profile: under `60s` for held-position
    protective HSL readiness when required cache/fill/candle proof is present.
  - Stretch target after optimized replay/checkpointing: under `10s` for warm
    restart with valid proof.

- [ ] Every performance claim should have a local/offline reproduction path.
  - Prefer copied monitor/cache fixtures and deterministic synthetic fixtures
    before relying on live exchange access.
  - VPS smoke should validate integration and real endpoint behavior, not be
    the only profiling environment.

- [ ] Operators should be able to answer "what delayed this trade?" from one
  report.
  - The report should connect startup readiness, input staleness,
    decision-boundary lag, Rust planning, Python filtering/gating, exchange
    writes, confirmation, and monitor/event-pipeline overhead.

## Performance Checklist

### P0: Readiness Contract

- [ ] Define readiness by order class, not by global startup completion.
  - Protective panic/reduce-only paths require fresh account-critical surfaces
    and the exact risk state for the held `coin+pside`.
  - Fresh entries require the broader strategy-input contract, including
    candidate freshness, EMA readiness, market snapshot freshness, and
    cooldown/trading eligibility.
  - Candidate-only stale inputs must not delay protective actions for held
    positions.

- [ ] Make every unavailable readiness state explicit.
  - Use structured events for unavailable, degraded, repairing, ready, and
    blocked states.
  - Include reason codes, affected symbols/psides, source coverage, age, and
    whether the state blocks protective actions, fresh entries, or diagnostics
    only.

- [ ] Do not use neutral defaults for trading-critical readiness.
  - Missing HSL, fill, candle, account, or EMA proof must not become zero
    drawdown, zero volume, empty fills, or ready-by-default.
  - Allowed fallbacks must be bounded, observable, and covered by tests.

### P0: HSL Protective Readiness

- [ ] Define HSL startup states explicitly.
  - `hsl_protective_unavailable`: required held-position proof is missing or
    invalid; protective HSL cannot be evaluated yet.
  - `hsl_protective_ready`: held positions have exact current HSL state and
    panic/protective decisions may proceed.
  - `hsl_entry_cooldown_unknown`: held positions are protected, but flat-symbol
    cooldown reconstruction is incomplete, so fresh HSL-gated entries remain
    blocked where affected.
  - `hsl_full_ready`: cooldown and replay state are complete for the configured
    HSL universe.

- [ ] Coin mode must not make a held coin wait behind unrelated coins.
  - A currently held `coin+pside` pair should be classified before historical
    flat pairs.
  - If exact held-pair replay reaches RED, the bot should run the existing
    protective panic supervisor immediately.

- [ ] Preserve exact HSL semantics for decisions that can trigger orders.
  - Do not replace EMA-smoothed HSL with raw drawdown unless the contract is
    explicitly changed.
  - If `ema_span_minutes > 1`, held-pair replay must produce the same runtime
    tier/drawdown state as the current full replay for that pair.

- [ ] Separate cooldown discovery from broad replay.
  - Build a fill-derived panic/cooldown index before replaying every historical
    coin.
  - Coins without current positions still need cooldown status reconstructed if
    a past panic can keep them non-tradable.
  - Coins without current positions and without relevant panic/cooldown history
    should not block protective startup.

- [ ] Add acceptance tests for protective startup.
  - A 24-pair fixture with one held late-sorting symbol must classify that held
    symbol before unrelated flat pairs.
  - Held-pair protective replay must match current full coin replay for both
    `ema_span_minutes=1` and `ema_span_minutes>1`.
  - Missing required fill/candle proof must surface an unavailable/degraded
    protective readiness state, not silently mark safe.

- [x] Replace `timeline_rows * pairs` replay with an exact lower-complexity
  path.
  - First optimization slice: finite-lookback replay now seeds pre-window state
    from fills, clamps dense candle/timeline construction to the configured
    lookback window, and skips old flat symbols with no in-window/current
    exposure.
  - Preferred shape: one pass through the timeline updating all active pair
    states that have values on that row, or per-pair sparse series built once.
  - Avoid repeated nested dict scans and repeated full fill-list scans per
    symbol.
  - Compact-memory slice deployed: cold coin replay now builds
    aligned NumPy account/pair arrays instead of the rich nested timeline and
    consumes those arrays directly. A 43,201-minute, 30-symbol builder profile
    reduced Python peak allocations from `686242590` to `73499666` bytes
    (89.3%). The fill-index prerequisite is deployed. Sparse replay keeps
    held-pair arrays dense and selects exact run, expiry, fill,
    intervention, and terminal boundaries for historical flat pairs. Its
    30-pair offline acceptance fixture reduced applied samples from `825430` to
    `43652` with identical final-state hashes and all `43201` held samples
    preserved. VPS5 then completed the four live 30-day replays in `140.746s`
    to `269.711s`, versus `601.246s` to `2279.519s` before sparse replay.
  - Preserve current RED/green/current-drawdown semantics: a historical RED
    crossing must not cause a panic now if current replay state is no longer in
    the red zone.

- [ ] Answer the current bottleneck question before broad rewrites.
  - Is startup blocked on CPU-bound Python replay, disk/cache reads, exchange
    backfill, monitor/event emission, or repeated data conversion?
  - If all needed data is cached locally, explain why replay still takes
    hundreds or thousands of seconds.
  - Confirm whether coin-mode replay currently serializes independent
    `coin+pside` pairs unnecessarily.
  - Confirm whether unrelated flat pairs can delay held-pair protective
    readiness.
  - Answered for the observed VPS incident: retained nested Python history and
    overlapping replay representations drove swap/page pressure. One-day
    synthetic replay arithmetic completed 43,200 pair-minutes in about 0.3s,
    while the 30-day rich builder retained about 686 MB of traced Python
    allocations.

- [ ] Prove whether coin-mode HSL needs cross-coin synchronization.
  - If one coin's HSL state depends only on that coin's fill/PnL and candle
    series, the held coin must not wait for unrelated coins to finish replay.
  - If any shared state exists, document it explicitly and test the dependency.
  - This decision should drive whether the first optimization is priority
    scheduling, sparse per-pair replay, multiprocessing, or a single vectorized
    pass.

- [ ] Separate cache-read speed from replay-compute speed.
  - Measure local artifact discovery, JSON/NDJSON/NPY decode, fill indexing,
    candle/timeline materialization, and replay compute separately.
  - A warm restart with complete local proof should not spend most startup time
    on exchange backfill or broad artifact rescan.
  - If disk cache reads are fast but replay is slow, optimize the replay loop.
  - If cache proof or decode is slow, add metadata indexes/checkpoints before
    changing trading logic.

- [ ] Identify the current bottleneck with a focused profile.
  - Measure time spent in fill indexing, candle/timeline construction, pair
    iteration, EMA update, drawdown/tier update, event emission, and disk/cache
    reads.
  - Run the profile on a copied local monitor/cache fixture when possible so
    optimization does not require live exchange access.
  - Report rows/s and held-pair protective elapsed time before and after each
    optimization.
  - Report whether the bottleneck is CPU-bound Python, disk/cache IO, event
    emission, or exchange/cache backfill.

- [ ] Index fill events once by `(pside, symbol)`.
  - Reuse that index for replay contracts, panic detection, position-size
    replay, realized-PnL peak/current calculations, and cooldown discovery.
  - Partial: the cold coin-HSL initializer now builds one stable pair index and
    reuses it for replay-contract inference and position-size reconstruction.
    Panic/cooldown indexing and sparse realized-PnL replay remain open.

- [ ] Parallelize only where correctness boundaries are independent.
  - Coin-mode held-pair protective reconstruction may classify independent
    `coin+pside` pairs separately, as long as shared account/fill coverage
    proof is established first.
  - Do not allow broad flat-pair replay to block a held pair that already has
    exact protective readiness.

- [ ] Prefer priority ordering before parallelism.
  - First replay currently held positions.
  - Then replay coins with active cooldown implications.
  - Then replay remaining eligible flat symbols in the background.
  - This should improve safety even on a 1-vCPU VPS where parallelism has
    limited value.

- [ ] Add structured replay timing fields.
  - Done: current full-replay events include `timeline_rows`, `pairs`,
    `held_pairs`, `cooldown_pairs`, `required_pairs`, `applied_rows`,
    `total_applied_rows`, `skipped_pairs` on completion, `rows_per_second`,
    `full_elapsed_s`, and `startup_blocking_elapsed_s`.
  - Done: the protective-readiness split emits true `protective_elapsed_s`,
    and the active scorecard slice retains that milestone separately from
    later pair progress while summarizing protective and full-replay elapsed
    milliseconds.

- [ ] Set concrete performance acceptance targets after the first optimized
  implementation.
  - Initial target: protective held-position readiness should be seconds, not
    minutes, on the VPS5 1-vCPU profile.
  - Full replay should be optimized enough that 25-30 pairs over 30 days is no
    longer a 20-40 minute operation.
  - Add regression protection for rows/s or elapsed-time regressions with a
    deterministic offline fixture.

- [x] Make the canonical HSL required-start boundary the single owner of fill and candle history
  materialization for that replay.
- [x] Preserve exact fill timestamps, realized PnL, fees, episode boundaries, and every relevant
  flat-scope cooldown while avoiding work before a proven disposable prefix.
- [x] Keep pside/unified and `threshold`/`never` full-lookback strict.
- [ ] Measure cold and warm reconstruction through the same authoritative path; do not add a
  parallel persisted replay-state compatibility matrix.

### P1: General Live Performance Report

- [x] Add `passivbot tool live-performance-report`.
  - It reads local monitor event streams only; text-log scraping is not used
    for structured timing metrics.
  - It should not contact exchanges, mutate caches, or depend on live bot
    availability.
  - It should support `--recent-minutes`, `--include-rotated`,
    `--event-tail-lines`, `--summary`, `--compact`, and bot/user filters.

- [ ] Add trading-impact annotations.
  - Mark phases that block all trading decisions.
  - Mark phases that block fresh entries but allow protective actions.
  - Mark phases that can delay panic/protective orders.
  - Mark phases that only affect diagnostics/console/dashboard.

- [x] Report decision-boundary lag.
  - For each cycle, measure how far after the relevant whole-minute boundary
    the bot started the cycle, called Rust as the current planning-ready proxy,
    produced a plan, submitted writes, and confirmed exchange state.
  - This is the main live-vs-backtest gap metric.

- [ ] Report input staleness at decision time.
  - Initial report-derived support covers account packet age at snapshot build
    and snapshot/EMA age at the Rust call boundary.
  - Remaining staleness surfaces: candle close age, market price age, config
    age, and richer symbol-scoped freshness where current events do not yet
    carry enough timestamp proof.
  - Separate strict trading blockers from stale-but-acceptable forager inputs.

- [ ] Add readiness SLA summaries.
  - Report time from process start to account ready, held-position protective
    ready, fresh-entry ready, first cycle started, first Rust call, first
    exchange write eligibility, and full background replay complete.
  - Group by exchange/user/bot so VPS-class regressions are visible before a
    panic incident.
- Status: partial. Startup phase timing aggregation is available from
  existing phase events. PR #1192 added per-bot and
  aggregate scope timing for the readiness milestones already emitted. The
  active follow-up makes current per-bot lifecycle snapshots independent of
  capped current-before-rotated traversal order while preserving historical
  aggregate distributions. PR #1194 added first-cycle, first-Rust-call, and
  first locally submitted write milestones. PR #1196 added the source event
  for true local fresh-entry eligibility, and the active consumer adds its
  current-lifecycle milestone. Actual connector invocation remains distinct.

- [ ] Add a "slowest blockers" view.
  - Rank operations by elapsed time and trading impact.
  - Separate "delayed protective action", "delayed fresh entry", "delayed
    diagnostics only", and "not on critical path".
  - Include enough event IDs/timestamps to jump from the summary into the
    structured event stream.

### P1: Runtime Cycle Speed

- [ ] Keep normal no-op cycles lean.
  - Current Hyperliquid samples were roughly `14-17s`, with `execute` around
    `6.5-7.4s`. Determine whether that execute time is real work,
    confirmation waits, sleep/rate-limit behavior, or instrumentation shape.

- [ ] Make cycle phase timings complete and unredacted where safe.
  - Current smoke output redacts some `authoritative` timing values. The event
    payload should expose numeric timing summaries while still redacting
    sensitive account payloads.

- [ ] Reduce monitor flush overhead.
  - Current Hyperliquid samples show `monitor_flush` around `1.0-1.5s`.
    Confirm whether this is disk IO, queue drain, compression/rotation, or
    synchronous snapshot work, then move heavy work off the critical path.

- [ ] Ensure protective actions are not delayed by non-critical readiness.
  - If candidate-only EMA/forager readiness is stale, protective panic,
    reduce-only, HSL, and unstuck safety paths should still proceed under their
    stricter-but-smaller data contract.

### P2: Startup And Warm Restart

- [ ] Measure cold start vs warm restart separately.
  - Track time to account ready, active-position candle ready, protective-HSL
    ready, full-HSL ready, first cycle started, first cycle completed, and first
    possible exchange write.

- [ ] Use local cache aggressively but with proof.
  - Warm restarts should avoid broad candle/fill repair when local metadata
    proves coverage.
  - Cache proof failures should be explicit and targeted, not broad blocking
    repairs by default.
  - If the bot was down only briefly and coverage proof still matches config
    and exchange state, startup should consume the existing cache/checkpoint
    before scheduling broad backfill.

- [ ] Add warm-restart acceptance fixtures.
  - Restart after a short downtime with complete cache proof should skip broad
    historical backfill and reach protective readiness quickly.
  - Restart with one stale symbol should repair that symbol, not stall every
    unrelated held position or the full forager universe.

- [ ] Leave bots running after smoke/restart operations.
  - Operational automation should stop/restart only when needed and should
    verify all expected bots are running before handing control back.

### P2: CPU And Resource Profile

- [ ] Add low-overhead process resource snapshots to the performance report.
  - CPU percent, RSS, open file descriptors if available, event queue depth,
    and monitor sink backlog.
  - Status: partial. Periodic `health.summary` now emits RSS, memory percent,
    cached non-blocking process CPU percent after the first priming sample,
    open FDs, load averages, and event-pipeline queue/drop/sink counters where
    available; `live-performance-report` aggregates those fields under
    `resource_pressure`. Loop lag and explicit sink backlog remain open
    source-event gaps.

### P2: Shutdown Latency

- [ ] Measure shutdown by stage.
  - Track signal received, exit flag set, task cancellation requested, remote
    calls cancelled or completed, monitor flush finished, and process exit.
  - Report slowest pending tasks at shutdown without logging secrets or raw
    exchange payloads.

- [ ] Make long non-critical work interruptible.
  - Big candle fetches, broad background replay, monitor scans, and forager
    refresh work should observe shutdown promptly.
  - Protective cleanup and final monitor flush may run briefly, but shutdown
    should not wait for fresh-entry-only background work.

- [ ] Add shutdown smoke expectations.
  - Repeated Ctrl+C should not be required for the normal path.
  - If a second interrupt is needed, the logs/events should identify the
    blocking task and stage.

## Target State

## Candidate PR Slices

These slices are intentionally small enough for normal review and live smoke.
Each slice should update this checklist with its result.

3. [x] Held-position protective readiness slice.
   - Classify currently held `coin+pside` pairs before unrelated flat pairs and
     emit explicit `hsl_protective_*` state events.
   - Acceptance: held-pair state matches existing full replay in tests, and a
     held late-sorting symbol is no longer delayed by broad flat-symbol replay.
   - Status: the first prerequisite centralizes post-RED episode finalization
     in Rust for backtest and Python live/history replay, including persistent
     no-restart peak/drawdown evaluation, explicit disposition, exact cooldown
     deadline, and coin-live retention of the persistent peak across restart.
     Independent preflight also found an existing coin-mode slot-budget
     denominator mismatch between live and backtest; reconcile that in a
     focused parity PR before relying on held-pair equivalence.
   - Status: the dependent parity branch now centralizes coin slot-budget and
     raw-drawdown math in Rust for live/replay and backtest. TWEL remains an
     activation/validation input, while configured live slots and intentional
     dynamic backtest slots remain caller-owned inputs.
   - Status: the sequencing slice freezes the existing replay candidate
     set and processes held pairs first, cooldown-affected pairs second, then
     remaining pairs with a bounded first-pair progress event. It deliberately
     originally kept full replay startup-blocking. The dependent readiness
     slice now releases startup after exact held-pair reconstruction, keeps
     pending pair initial entries blocked at planning and final submission,
     continues the same replay task under shutdown ownership, and emits bounded
     protective/full timing plus ready/pending pair counts. Production elapsed
     acceptance remains to be measured on VPS5 after merge.

4. [ ] Full replay lower-complexity slice.
   - Replace avoidable nested scans and repeated fill/timeline work with exact
     indexed/sparse replay.
   - Acceptance: equivalence tests pass against the old replay contract and the
     benchmark shows a material rows/s improvement.

5. [x] HSL bounded-history materialization slice.
   - Pass the canonical consumer-owned replay boundary into authoritative fill/candle preparation.
   - Result: candles and dense minute/pair structures begin at the canonical boundary. Earlier
     authoritative sparse fills remain available only to prove the boundary and seed exact balance
     and position state; they are not returned as active replay events. Full-lookback modes,
     cooldown scopes, and ambiguous evidence remain strict.

6. [ ] Shutdown and restart latency slice.
   - Make long non-critical startup/background work interruptible and report
     shutdown blockers.
   - Acceptance: repeated Ctrl+C should not be needed in the normal path, and
     slow shutdown identifies the blocking stage.

## Suggested Implementation Order

4. Optimize authoritative history materialization after the replay path is understood.
   - Use the canonical required boundary to avoid discarded work. Do not reintroduce persisted HSL
     replay state or a second compatibility/proof system.

5. Keep the live performance report as the operator-facing scorecard.
   - Every merged performance slice should update this checklist, add a
     regression test where practical, and make the report more useful for the
     next bottleneck.
