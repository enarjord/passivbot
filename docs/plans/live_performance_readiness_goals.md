# Live Performance And Readiness Goals

This is the working checklist for making live execution approach the backtest
ideal: complete inputs available at the decision boundary, fast deterministic
planning, and prompt exchange writes when trading logic says action is needed.

Backtest remains the benchmark. Live cannot be identical because it depends on
exchange APIs, network latency, rate limits, cache repair, order confirmation,
and partial/stale data. The goal is to measure every gap, reduce avoidable
latency, and make unavoidable latency explicit in the event stream.

The central readiness rule is speed with proof, not speed by assumption. A
restart may use cached data and checkpoints aggressively only when metadata
proves coverage, freshness, config compatibility, and code/schema
compatibility. If proof is missing, the bot should perform the smallest exact
repair needed for the affected order class instead of falling back to broad
blocking reconstruction.

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

## Current Priority Checklist

This is the short actionable goal list. The rest of the document explains the
evidence, contract, and candidate implementation slices.

1. [ ] Measure the slow paths with enough detail to know what is blocking
   trading.
   - [ ] Add/report min, mean, p50, p95, max, count, latest timestamp, bot, and
     trading-impact label for startup, state refresh, candle/EMA readiness, HSL
     replay, Rust planning, Python reconciliation, exchange writes,
     confirmation, monitor flush, event-pipeline overhead, and shutdown.
   - [ ] Split each startup delay into account-critical readiness, held-position
     protective readiness, fresh-entry readiness, and background/full replay.
   - [ ] Make every long operation answer whether it delayed protective action,
     fresh entries, normal cycles, or diagnostics only.

2. [ ] Fix the P0 HSL startup safety gap.
   - [ ] A held `coin+pside` must not wait behind unrelated flat coins before
     its exact HSL state is known.
   - [ ] Current drawdown state takes precedence: a historical red crossing
     must not trigger a new panic if the exact current state is no longer red.
   - [ ] Full historical/cooldown reconstruction may continue after held
     positions are protectively ready, but fresh entries remain blocked for
     symbols whose cooldown/trading eligibility is not yet known.

3. [ ] Prove the HSL replay bottleneck before optimizing broadly.
   - [ ] Measure cache discovery, cache decode, fill indexing, candle/timeline
     materialization, pair iteration, EMA/drawdown update, event emission, and
     exchange/backfill time separately.
   - [ ] Determine whether coin-mode replay is CPU-bound Python, disk/cache IO,
     exchange backfill, repeated data conversion, or unnecessary serial
     dependency between independent pairs.
   - [ ] Add an offline deterministic fixture so rows/s and equivalence can be
     checked without live exchange access.

4. [ ] Optimize exact HSL replay.
   - [ ] Replay held positions first, cooldown-affected symbols second, and
     remaining flat symbols last or in background.
   - [ ] Replace avoidable `timeline_rows * pairs` work, repeated fill scans,
     and repeated data conversions with indexed/sparse or single-pass logic
     where exact equivalence is proven.
   - [ ] Keep panic/order-triggering decisions equivalent to the current HSL
     contract unless an explicit contract change is reviewed.

5. [ ] Add resumable HSL checkpoints after exact replay is understood.
   - [ ] Treat checkpoints as performance caches only, never as authority.
   - [ ] Validate exchange, user, signal mode, risk config, schema/code version,
     fill coverage/hash, candle coverage/hash where required, market metadata,
     and last processed timestamp before resume.
   - [ ] On any ambiguity, bypass the checkpoint and perform exact replay with
     a structured reason event.

6. [ ] Make warm restarts and shutdown operationally fast.
   - [ ] Warm restart should use proven local cache/checkpoint state before
     scheduling broad repair.
   - [ ] Short downtime should not cause broad HSL/candle/fill reconstruction
     when coverage proof is still valid.
   - [ ] Ctrl+C should interrupt non-critical long work promptly and report the
     blocking task if shutdown is slow.

7. [ ] Keep observability and trading behavior separated.
   - [ ] Reports, probes, and cache doctors may prove state and expose gaps, but
     must not enforce trading decisions unless a separate behavior PR changes
     the contract.
   - [ ] Any readiness fallback used by live trading must be bounded,
     observable, and covered by tests.
   - [ ] No neutral defaults for missing HSL, fill, candle, account, EMA, or
     market proof.

## Definition Of Done

- [ ] Held-position protective readiness is reached in seconds, not tens of
  minutes, when required local/exchange proof is available.
- [ ] Full HSL replay no longer blocks immediate protective action for current
  positions.
- [ ] Warm restart with valid proof reaches protective readiness quickly and
  bypasses unnecessary broad reconstruction.
- [ ] Invalid or stale cache/checkpoint proof falls back loudly to exact repair
  without fabricating safe state.
- [ ] One `passivbot tool live-performance-report` run can explain the main
  delay categories for startup, cycle, exchange write, confirmation, and
  shutdown.
- [ ] Every merged performance slice updates this checklist with baseline,
  target, result, and review/smoke evidence.

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

1. HSL coin-mode startup replay is the current P0 latency gap.
   - Binance incident on 2026-06-26: coin HSL replay loaded at `16:19:33Z`
     with `symbols=24 pairs=24 rows=43201 fills=2704`, completed at
     `16:46:37Z` after `985965` applied rows in `1623.4s`, and XLM panic order
     was posted at `16:48:06Z`.
   - Current VPS5 restart evidence shows the same shape still present:
     Binance progress reached `275323` applied rows at `474.669s` with
     `pairs=25`, GateIO reached `254638` rows at `459.291s` with `pairs=29`,
     and OKX reached `317411` rows at `455.068s` with `pairs=26`.
   - Approximate replay speed from those samples is only `550-700` applied
     rows/s. Full replay at `25-29 * 43201` rows therefore remains a tens of
     minutes operation on the small VPS.

2. Regular runtime cycle timings are measurable, but currently sparse in the
   smoke window because several bots were still in startup HSL replay.
   - Hyperliquid had four recent `cycle.completed` events with elapsed
     `13976ms`, `14189ms`, `15103ms`, and `16703ms`.
   - Min/mean/max for those cycles: `13976ms / 14993ms / 16703ms`.
   - The slowest phase in those cycles was `execute`, about `6529-7404ms`,
     followed by `market_state`, about `1258-4071ms`, and `monitor_flush`,
     about `967-1457ms`.

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

4. Existing observability now has an initial performance report, but not yet
   all decision-boundary and input-staleness metrics.
   - Useful inputs already exist: `cycle.completed.timings_ms`,
     `state.refresh_timing`, `remote_call.* elapsed_ms`,
     `hsl.replay.* elapsed_s`, startup timing events, monitor health loop
     duration, and exchange probe endpoint latency summaries.
   - First slice added `passivbot tool live-performance-report` with timing
     groups, trading-impact labels, summary projection, and bot/user/exchange
     filters.
   - A follow-up slice added decision-boundary lag groups from current cycle
     events.
   - A follow-up slice added initial input-staleness groups from existing
     account packet, snapshot, EMA bundle, and Rust-call events.
   - A follow-up slice added HSL coin replay pair classification, applied-row,
     elapsed, and rows-per-second fields to structured replay events.
   - Missing pieces: candle close age, market price age, config age, and
     complete coverage for every order/write/shutdown stage.

5. VPS5 restart evidence after the HSL replay timing slice confirms the new
   fields are live, while also confirming the underlying speed problem remains.
   - Binance loaded `pairs=26`, `held_pairs=1`, `cooldown_pairs=1`,
     `required_pairs=20`, and `timeline_rows=43201`, then progressed at roughly
     `850-950` applied rows/s after the first minute.
   - GateIO loaded `pairs=29`, `held_pairs=1`, `cooldown_pairs=0`,
     `required_pairs=24`, and `timeline_rows=43201`; early progress peaked
     above `5000` rows/s before settling lower as replay advanced.
   - OKX loaded `pairs=27`, `held_pairs=1`, `cooldown_pairs=0`,
     `required_pairs=21`, and `timeline_rows=43201`.
   - These fields make the next optimization measurable, but they do not yet
     split protective readiness from full replay.

## Required Performance Report Matrix

The live performance report should become the canonical answer to "where did
the time go?" for a live bot. Every row below should expose `count`, `min`,
`mean`, `p50`, `p95`, `max`, latest timestamp, bot identity, and trading-impact
classification when enough source events exist.

- [ ] Startup: process start to account-critical ready.
- [ ] Startup: process start to held-position protective HSL ready.
- [ ] Startup: process start to fresh-entry ready.
- [ ] Startup: process start to first planning cycle started/completed.
- [ ] Startup: process start to first possible exchange write.
- [ ] HSL: fill/cache load time, replay build time, held-pair protective
  replay time, full replay time, checkpoint load time, checkpoint write time,
  rows/s, symbols/pairs/held pairs/cooldown pairs.
- [ ] Cache readiness: fill cache coverage proof, candle coverage proof,
  checkpoint compatibility decision, repair scope, repair elapsed time.
- [ ] Account state: balance, positions, open orders, fills, state-refresh wall
  time, surface max/sum time, retry/degraded counts.
- [ ] Market data: ticker/market price age, candle close age, EMA bundle age,
  forager feature age, candle remote fetch latency, synthetic/no-trade gap
  repair counts.
  - Status: partial. New `snapshot.built` metadata and performance-report
    groups expose planning surface ages plus market-snapshot max/mean age at
    snapshot build. Remaining work: candle close age, forager feature age, and
    symbol-scoped stale-but-acceptable candidate metadata.
- [ ] Decision boundary: whole-minute lag to cycle start, Rust input snapshot,
  Rust output, Python gate/filter, first exchange write, confirmation refresh.
- [ ] Cycle phases: market state, account state, Rust planning,
  reconciliation/gating, execution, confirmation, monitor flush, event
  pipeline overhead.
- [ ] Exchange writes: create/cancel/close/panic write latency, exchange
  response latency, ambiguous write rate, confirmation latency.
  - Status: partial. The report now derives order-wave total duration,
    create/cancel sent-to-terminal response duration, confirmation duration,
    missing-id counts, unpaired-terminal counts, and pending-start counts from
    existing structured events. Remaining work: distinguish close/panic
    subclasses and tie execution delays back to exact order class once those
    fields are consistently available in the event stream.
- [ ] Resource pressure: CPU/load, RSS, memory percent, open FDs, event queue
  depth, dropped event counters, sink errors, loop lag where available.
- [ ] Shutdown: signal to exit flag, cancellation request, blocking task names,
  final monitor flush, process exit.

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

- [ ] A broad full-HSL replay should stop being a critical-path startup blocker.
  - Initial target: full replay for `25-30` pairs over the configured lookback
    completes in under `5m` on VPS5-class hardware.
  - Stretch target: warm restart with a valid checkpoint resumes in under `60s`
    and continues exact background repair/replay when needed.

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

- [ ] Split HSL startup readiness into protective readiness and full replay.
  - Protective readiness covers current positions and active cooldown residue
    that can affect immediate risk action.
  - Full replay may continue after protective readiness, but fresh initial
    entries must remain blocked until cooldown/trading eligibility is known.

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

### P0: HSL Replay Speed

- [ ] Replace `timeline_rows * pairs` replay with an exact lower-complexity
  path.
  - Preferred shape: one pass through the timeline updating all active pair
    states that have values on that row, or per-pair sparse series built once.
  - Avoid repeated nested dict scans and repeated full fill-list scans per
    symbol.
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

- [ ] Build a deterministic HSL replay benchmark fixture.
  - Include 25-30 pairs, one or more held positions, at least one cooldown
    candidate, 30 days of one-minute rows, and realistic fill density.
  - Fixture should run offline and produce comparable output for current full
    replay, optimized replay, and checkpoint resume.
  - Output metrics: elapsed, rows/s, per-stage timings, current HSL state by
    held pair, cooldown state by affected flat pair, and equivalence diff.

- [ ] Index fill events once by `(pside, symbol)`.
  - Reuse that index for replay contracts, panic detection, position-size
    replay, realized-PnL peak/current calculations, and cooldown discovery.

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
  - Remaining: add true `protective_elapsed_s` once protective readiness is
    split from full replay.

- [ ] Set concrete performance acceptance targets after the first optimized
  implementation.
  - Initial target: protective held-position readiness should be seconds, not
    minutes, on the VPS5 1-vCPU profile.
  - Full replay should be optimized enough that 25-30 pairs over 30 days is no
    longer a 20-40 minute operation.
  - Add regression protection for rows/s or elapsed-time regressions with a
    deterministic offline fixture.

### P1: HSL Checkpointing

- [ ] Add resumable HSL checkpoints after successful exact reconstruction.
  - Checkpoints are performance caches only. They must never become an
    unverified source of trading truth.
  - Invalid, incomplete, stale, or schema-mismatched checkpoints must fall back
    to exact exchange/cache-derived reconstruction.

- [ ] Include strong proof metadata in each checkpoint.
  - Exchange, user, signal mode, pside, symbol scope, HSL config hash, bot
    config hash for relevant risk fields, code/schema version, `c_mult` and
    market metadata proof, lookback window, last processed timestamp, realized
    PnL reset timestamp, fill cache coverage metadata, fill event count/hash,
    candle coverage range/hash or known-gap proof when candles are required,
    balance baseline, and serialized HSL runtime state.

- [ ] Make checkpoint resume exact.
  - Resume from checkpoint, replay only events/candles after checkpoint
    timestamp, and compare final metrics to full replay in tests.
  - Store checkpoint write timing and resume/fallback reason in structured
    events.

- [ ] Keep checkpointing stateless in behavior.
  - A checkpoint may accelerate reconstruction, but every trading decision must
    still be reproducible from exchange state, config, cache coverage proof, and
    replay after the checkpoint boundary.
  - Checkpoint invalidation must be conservative: if any required proof is
    ambiguous, discard or bypass the checkpoint and emit the reason.

- [ ] Treat checkpoints as resumable proof, not as authority.
  - On every startup, validate the checkpoint against config, code/schema,
    fill coverage, candle coverage where required, market metadata, and last
    processed timestamp before using it.
  - Resume only from the validated boundary and replay new exchange/cache data
    after that boundary.
  - Emit checkpoint load/resume/bypass/write events with elapsed time and
    reason codes.
  - Acceptance: a warm restart with a valid checkpoint reaches held-position
    protective readiness quickly, while an invalid checkpoint falls back to
    exact replay loudly and safely.

### P1: General Live Performance Report

- [x] Add `passivbot tool live-performance-report`.
  - It reads local monitor event streams only; text-log scraping is not used
    for structured timing metrics.
  - It should not contact exchanges, mutate caches, or depend on live bot
    availability.
  - It should support `--recent-minutes`, `--include-rotated`, `--summary`,
    `--compact`, and bot/user filters.

- [ ] Aggregate min/mean/max/p50/p95/count by bot and operation.
  - Required groups: startup stages, full cycle elapsed, cycle phase timings,
    authoritative refresh wall/surface timings, remote-call endpoint latency,
    candle remote fetch latency, HSL replay/protective replay, Rust planning,
    reconciliation, execution waves, order create/cancel/confirmation, monitor
    flush, event pipeline health, and shutdown stages.

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

- [ ] Add a full live-operation duration table.
  - Include startup, account refresh, fill refresh, cache proof, HSL replay,
    candle/EMA readiness, forager feature readiness, Rust planning,
    reconciliation/gating, order execution, confirmation refresh, monitor flush,
    event-pipeline enqueue/write, and shutdown.
  - Each group should include min, max, mean, p50, p95, count, latest sample,
    bot/exchange/user, and trading-impact label.
  - The table should be usable after a live incident to identify whether the
    critical delay was before risk classification, before planning, before
    exchange write, or after exchange write.

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

- [ ] Identify CPU-bound Python loops.
  - HSL replay is currently the obvious case. Other candidates are EMA
    readiness, candle warmup/repair, forager ranking, and monitor serialization.

- [ ] Add offline synthetic benchmarks for known hot paths.
  - HSL coin replay over 30 days and 30 pairs.
  - EMA readiness over a high-cardinality forager universe.
  - Monitor event ingestion and smoke-report scan over large NDJSON segments.

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

- [ ] On restart, any currently held position reaches exact protective risk
  readiness quickly enough that panic/protective action is not delayed by broad
  universe replay.
- [ ] Fresh entries start only when their trading contract is ready, but
  candidate-only missing/stale data does not block unrelated protective
  actions.
- [ ] Operators can run one performance report and see where time is spent:
  startup, data freshness, exchange calls, Rust planning, Python reconciliation,
  exchange writes, confirmation, monitor/event pipeline, and shutdown.
- [ ] Every slow operation has a structured event with enough correlation and
  timing data to explain whether it affected trading behavior.
- [ ] HSL checkpointing reduces warm-restart replay without weakening
  stateless correctness: checkpoints accelerate reconstruction only when their
  proof matches exchange/cache-derived inputs.

## Candidate PR Slices

These slices are intentionally small enough for normal review and live smoke.
Each slice should update this checklist with its result.

1. [ ] Performance-report coverage slice.
   - Add missing report groups for resource pressure, HSL replay/protective
     readiness, candle/market/config age, and shutdown stages as source events
     become available.
   - Acceptance: `passivbot tool live-performance-report` can produce a bounded
     summary explaining the slowest startup and cycle blockers from local
     monitor data only.

2. [ ] HSL replay benchmark/profiling slice.
   - Add an offline deterministic benchmark or fixture path for coin-mode HSL
     replay using realistic pair count, fill count, and row count.
   - Acceptance: report current elapsed, rows/s, and per-stage timing without
     contacting exchanges or changing live behavior.

3. [ ] Held-position protective readiness slice.
   - Classify currently held `coin+pside` pairs before unrelated flat pairs and
     emit explicit `hsl_protective_*` state events.
   - Acceptance: held-pair state matches existing full replay in tests, and a
     held late-sorting symbol is no longer delayed by broad flat-symbol replay.

4. [ ] Full replay lower-complexity slice.
   - Replace avoidable nested scans and repeated fill/timeline work with exact
     indexed/sparse replay.
   - Acceptance: equivalence tests pass against the old replay contract and the
     benchmark shows a material rows/s improvement.

5. [ ] HSL checkpoint proof/resume slice.
   - Write and resume from conservative performance checkpoints after exact
     reconstruction.
   - Acceptance: valid checkpoint warm restart is fast; stale/incompatible
     checkpoint is bypassed with an observable reason; decisions remain
     reproducible from exchange/cache-derived truth.

6. [ ] Shutdown and restart latency slice.
   - Make long non-critical startup/background work interruptible and report
     shutdown blockers.
   - Acceptance: repeated Ctrl+C should not be needed in the normal path, and
     slow shutdown identifies the blocking stage.

## Suggested Implementation Order

1. Add the missing performance/readiness metrics first, so every optimization
   has before/after evidence.
   - Decision-boundary lag and initial input staleness are started.
   - Next metrics: HSL protective-ready elapsed time, full replay elapsed time,
     cache proof decision, candle close age, market price age, and shutdown
     blocking stage.

2. Optimize HSL coin-mode protective readiness before broad full-replay speed.
   - The highest-risk failure mode is delayed panic for a held position.
   - Full universe replay and cooldown indexing still matter, but they should
     not sit on the critical path for already-held positions.

3. Add exact HSL replay profiling and lower-complexity replay.
   - Profile first, then remove repeated scans and avoid pair-by-row nested
     work where a sparse/event-driven pass is exact.
   - Prove equivalence against current replay with fixtures before using it in
     live.

4. Add checkpoints after the exact optimized path is understood.
   - Checkpoints should make warm restarts cheap, but they should not obscure
     the baseline exact replay contract or make debugging harder.

5. Keep the live performance report as the operator-facing scorecard.
   - Every merged performance slice should update this checklist, add a
     regression test where practical, and make the report more useful for the
     next bottleneck.
