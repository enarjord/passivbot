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

## Current Evidence

Snapshot source: VPS5 monitor/smoke data on 2026-06-27 after `v8` head
`b5e08986`.

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

- [ ] Identify the current bottleneck with a focused profile.
  - Measure time spent in fill indexing, candle/timeline construction, pair
    iteration, EMA update, drawdown/tier update, event emission, and disk/cache
    reads.
  - Run the profile on a copied local monitor/cache fixture when possible so
    optimization does not require live exchange access.
  - Report rows/s and held-pair protective elapsed time before and after each
    optimization.

- [ ] Index fill events once by `(pside, symbol)`.
  - Reuse that index for replay contracts, panic detection, position-size
    replay, realized-PnL peak/current calculations, and cooldown discovery.

- [ ] Parallelize only where correctness boundaries are independent.
  - Coin-mode held-pair protective reconstruction may classify independent
    `coin+pside` pairs separately, as long as shared account/fill coverage
    proof is established first.
  - Do not allow broad flat-pair replay to block a held pair that already has
    exact protective readiness.

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
