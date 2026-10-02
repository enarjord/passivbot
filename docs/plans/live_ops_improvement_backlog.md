# Live Operations Improvement Backlog

## Purpose

This backlog captures high-value live-operations and trading-system improvement
areas discovered through production evidence. It is intentionally separate from
the finite live-logging migration. An item may use structured events without
becoming logging-overhaul scope.

The logging overhaul remains the foundation: a centralized event stream should
make the items below easier to prove, test, and operate.

This backlog does not feed the logging-overhaul loop by default. Promote an item
back into that plan only when current evidence identifies a defect in the shared
event contract, routing, boundedness, redaction, retention, sink isolation, or a
specific unmet completion criterion. A desire for another report, selector,
smoke verdict, debug profile, restart check, or operator convenience is not
enough.

## Triage Before Implementation

Classify proposed work into one primary lane:

1. **Trading behavior or safety:** strategy, reconciliation, readiness,
   exchange state, fills/PnL, HSL, order construction, or execution policy.
   Use a focused production PR and trading-contract validation.
2. **Performance or reliability:** startup, shutdown, cache, replay, remote
   calls, resource pressure, or connector throughput. Require a measured
   bottleneck and an explicit target.
3. **Operations:** restart/deploy/process control, configuration preflight,
   incident workflow, or repository preparation. Keep authority and
   state-changing actions explicit.
4. **Observability defect:** incorrect/missing evidence, unsafe payloads,
   duplicate ownership, event loss, broken correlation, or unbounded sinks.
5. **Observability consolidation:** two or more existing readers or reports can
   share one bounded implementation with a demonstrated net reduction in code.
   This may reopen the logging overhaul under resume reason 4 in
   `live_logging_overhaul_current_status.md`.
6. **Convenience:** another projection, filter, export, dashboard, or summary.
   Defer unless repeated operator use demonstrates value greater than its
   maintenance cost.

Only the observability-defect and observability-consolidation lanes may reopen
the logging-overhaul plan.

Before accepting a PR, record the observed problem, affected scope, simplest
credible fix, validation, and stopping condition. Prefer removing or
consolidating existing machinery over adding another parallel path.

Update policy:

- Keep open work in `High-Value Follow-Ups` with checkboxes, one primary lane,
  and short current status notes.
- Existing entries without a recorded primary lane are unclassified and cannot
  be selected for implementation or used to reopen the logging overhaul. Add
  the lane, evidence, simplest credible fix, validation, and stopping condition
  before selecting one.
- When a PR completes all or a meaningful first slice of an item, update the
  item status. Add a `Merged Work Log` entry only for a material milestone, not
  every incremental PR or deployment.
- Leave follow-up refinements attached to the item instead of hiding them in the
  completed log.
- Add newly discovered gaps only when backed by concrete evidence. Do not turn
  every edge case or possible refinement into queued implementation work.

Related detailed plans:

- `docs/plans/live_logging_overhaul_plan.md`
- `docs/plans/live_performance_readiness_goals.md`
- `docs/plans/live_restart_shutdown_and_warm_cache_handoff.md`

## High-Value Follow-Ups

   Target contract: operators need a preflight-visible warning/error before
   starting a config that combines `balance_override` with `unified` or `pside`
   HSL, and a carefully designed recovery path for historical panic fills that
   are known to have been created by a bad HSL model. Recovery must not silently
   ignore exchange-derived panic fills. Any invalid-panic override must be
   explicit, auditable, bounded to exact fill/time/side/symbol evidence, and
   compatible with stateless restart semantics.

   Investigation directions: extend `live-config-preflight` and/or
   `hsl-startup-preview` to flag the unsafe contract offline; add richer HSL
   baseline-source diagnostics; design an operator-owned invalid-panic marker
   if needed; and add tests proving that the default remains fail-loud and
   exchange-derived cooldown evidence is ignored only when the explicit recovery
   contract is satisfied.

   Target contract: startup must prioritize fast protective HSL classification
   for currently held positions. Full historical replay can continue for
   cooldown reconstruction, diagnostics, or non-urgent precision, but it must
   not block a conservative current-red check for held symbols. Candidate
   solutions should preserve statelessness, exchange-derived truth, and the
   current HSL semantics, while making the startup panic path bounded and
   observable.

   Investigation directions: fast-path coin-mode current drawdown from fill/PnL
   cumsum for currently held coins; position-first replay before flat-symbol
   replay; incremental/checkpointed replay artifacts that are performance
   caches only; replay indexing/vectorization; early stop once current held
   coin state proves red; and separate cooldown discovery from immediate panic
   eligibility. Any implementation must include targeted tests and structured
   startup timing evidence.

1. [x] Incident bundle generator.
   Status: initial implementation plus trace-report integration merged.
   `passivbot tool live-incident-bundle` collects local monitor event reports,
   live-event trace reports, smoke summaries, redacted log excerpts, monitor
   snapshots, config hashes, runtime metadata, and bounded event segments into a
   tarball. Bundles can also include trace reports, optional smoke-report
   process status, event-file discovery metadata, recent time windows, and
   problem-event reports that reuse the same predicate as smoke/report query
   tooling. Incident bundles also expose the common event-query scope filters
   needed to build focused one-bot or one-component bundles from a root monitor
   tree.

   Remaining refinements: richer remote smoke integration when the
   restart/smoke automation exists; keep recent-window bundle scans bounded for
   very large current monitor segments; continue cross-bot incident workflow
   improvements only where concrete operator diagnostics require them.

2. [x] Event query and timeline CLI extensions.
   Status: core filter/timeline work merged. `passivbot tool live-event-query`
   now supports event discovery, compact JSON output, current-vs-rotated segment
   selection, terse timeline rendering, and filters for event type/kind, cycle
   id, order wave id, remote call id/group id, bot id, snapshot id, plan id,
   action id, symbol, pside, reason code, status, problem-event state, and event
   time window. It also supports aggregate trace summaries over matched events
   and order waves, plus a
   dedicated order-trace reconstruction view for order waves/actions and a
   cycle-trace reconstruction view with nested order traces. Incident bundles
   now embed the existing trace-summary/order-trace reports and cycle traces
   when scoped to `--cycle-id`.

   Remaining refinements: cross-bot incident workflow and possibly a lightweight
   local event index if incident queries over very large histories remain slow.
   A 2026-06-29 VPS5 probe after PR #875 showed that
   `live-event-query --exchange gateio --user gateio_01 --include-rotated`
   still ran for roughly two minutes on a focused Gate.io/ZEC HSL query before
   manual interruption. PR #877 added mtime-based pruning for bounded
   `--since-ms`/`--recent-minutes` queries; the same VPS5 query then completed
   under a 20-second timeout wrapper, scanning 4 files and reporting
   `files_skipped_before_window=160`. If that remains too slow for larger
   incident windows, consider an event index or reverse chronological scanning
   with early stop, while preserving direct file/events-dir workflows.
   A 2026-06-30 follow-up added opt-in
   `live-event-query --event-tail-lines` for repeated recent-window queries
   over large current monitor segments. The default remains full event
   validation; bounded query output reports tail-limit metadata in
   `event_window`. Another 2026-06-30 follow-up added source, component, and
   side filters for envelope-scoped queries. A subsequent follow-up pruned
   monitor file discovery for path-shaped `--bot-id` filters while preserving
   full scans for opaque bot ids.

3. [ ] Live restart/smoke automation.
   Status: partial. The read-only `live-smoke-report` tool exists and can now
   compare running `passivbot live` processes against a tmuxp-style supervisor
   config, scope structured monitor events and parseable timestamped text logs
   to a requested time window, apply an explicit unparseable text-log policy,
   avoid traceback-prose false positives, avoid stale contextless traceback
   false positives when a time-windowed log tail starts mid-traceback, and
   summarize recent HSL/risk events from the structured stream. It also surfaces
   bounded latest problem-event
   context for selected non-hard groups such as EMA/cycle readiness
   degradation, plus aggregate problem-event groups keyed by bot/event/reason/
   status/hard flag/symbol/position side, passive remote-call health summaries,
   account-critical remote-call summaries, repository state, and a concise
   `--summary` projection for operator smoke checks. It also has a `--brief`
   projection for top-level smoke-loop counters without event groups or log
   matches. It also summarizes existing structured shutdown lifecycle events as
   `shutdown_events` in full, summary, and brief reports, and existing
   `ema.unavailable` events as `ema_readiness_health`/`ema_readiness` with
   latest candidate/unavailable counts and bounded reason/error evidence. The
   smoke report now also redacts common user/home prefixes from
   `repository.root` for shareable reports and surfaces explicit
   dropped-unparsed attention/hard counters when the opt-in log-window drop
   policy suppresses contextless hard-looking log fragments. The safe local
   exact-target restart executor and canonical repository preparation are
   implemented; remote-host orchestration and automatic force escalation remain
   outside them. The active slice composes the executor with one bounded
   post-restart monitor/log smoke window. A 2026-06-30 follow-up made the
   existing startup timing evidence visible in `live-smoke-report --summary`
   and `--brief`, so repeated smoke loops can see slow startup phases without
   opening the full report. Another 2026-06-30 follow-up made bounded text-log
   window counters visible in `--brief`, so hard/attention log counts show
   whether they came from a time-windowed scan and how much log evidence was
   skipped.
   For Rust-touching deploys, the restart flow must also make extension rebuild
   and freshness verification explicit before stopping/restarting live bots; PR
   #756 showed that the VPS has Rust under `/root/.cargo/bin` but non-login SSH
   commands need explicit `PATH`/`VIRTUAL_ENV` for `maturin develop --release`.

   Formalize the repeated VPS smoke routine: pull a branch, stop configured
   bots, measure shutdown time per bot, reload from `/root/bots_vps5.yaml`,
   wait, then summarize process liveness, git head, recent hard errors, monitor
   event counts, startup timings, and resource usage. This should be safe,
   explicit, and produce a reviewable smoke report.

   Remaining refinements: activate the bounded `sink.degraded` redaction
   follow-up and observe a fresh settled post-restart window. Remote-host control
   and any force-escalation policy remain separate review boundaries.
   The concise and brief summaries are intentionally bounded; further changes
   should target missing smoke fields rather than larger chat-facing payloads.
   2026-06-26 VPS5 deploy evidence: after PR #709, one Ctrl+C round stopped
   Binance but Kucoin, GateIO, OKX, and Hyperliquid remained as orphaned live
   processes after two Ctrl+C rounds and required SIGTERM before reload. This
   reinforces that restart orchestration needs per-bot shutdown timing, orphan
   detection, and an explicit escalation ladder. The smoke-report supervisor
   process diagnostics now make duplicate configured-command matches and
   extra/orphan-like `passivbot live` processes visible from the local process
   table before any restart orchestration.

4. [ ] Startup phase budget tracking.
   Status: partial. Startup timing and warmup cache decision events exist, and
   `live-smoke-report` now summarizes latest startup phase timings with rolling
   median/p95 baselines from local monitor events plus report-only budget
   projections against prior p95 phase baselines. PR #1269 added optional
   durable diagnostic budgets to existing timing events and gives configured
   values smoke-report precedence. PR #1270 carries those same assessments
   into `live-performance-report`; enforcement and broader
   readiness-stage coverage remain out of scope.

   Work log:
   - 2026-06-27: Added report-only startup budget projections to
     `live-smoke-report` phase summaries, comparing latest elapsed/phase
     timings against prior local p95 baselines without changing startup
     behavior or adding new runtime events.
   - 2026-07-11: Active branch `codex/v8-startup-readiness-sla` adds
     centralized readiness scope and trading-impact labels to existing startup
     timing events plus per-bot/aggregate performance and smoke projections.
     It remains additive and does not claim fresh-entry, first-Rust-call, or
     first-exchange-write readiness.
   - 2026-07-11: Follow-up branch
     `codex/v8-performance-startup-lifecycle` makes capped rotated performance
     reports retain the latest lifecycle's per-bot startup snapshot independent
     of current-before-rotated file traversal while preserving historical
     aggregate distributions.
   - 2026-07-16: PR #1266 made the brief projection aggregate existing
     elapsed/phase budget statuses so unavailable or no-baseline phases cannot
     look implicitly within budget. It remains report-only and non-enforcing.
   - 2026-07-16: PR #1269 added optional
     `live.startup_phase_budgets`, carries configured targets on existing
     `bot.startup_timing` events, and makes smoke reports prefer them over prior
     p95 projections. The values are diagnostic and never gate startup or
     trading.
   - 2026-07-16: PR #1270 makes the
     performance report retain bounded latest-lifecycle configured-budget
     assessments and aggregate their status without changing startup behavior.

5. [x] Resource pressure telemetry.
   Status: initial implementation merged. `health.summary` events now include
   process RSS, memory percent when available, open file descriptor count,
   system load averages, CPU count, and live-event pipeline queue/drop/sink
   error counters. `live-smoke-report` now also projects those existing
   event-pipeline counters into full, summary, and brief smoke reports. The
   resource-pressure path also includes process CPU percent after psutil's
   non-blocking first-sample priming, health-summary scheduling lag after the
   first heartbeat, and optional psutil-backed system memory/swap pressure
   fields for host-level pressure scans. PR #1200 added per-health-window
   processed count, queue-wait total/max, and aggregate worker sink-service
   total/max with non-consuming ordinary monitor snapshots. PR #1203 attributed
   that worker time to fixed structured/monitor sink write counts and service
   total/max. PR #1204 then split real monitor writes into fixed conversion,
   publisher lock-wait, rotation, persistence, and maintenance timing. Fresh
   VPS5 evidence attributed 47,478.85 of 53,903.942ms cumulative monitor time
   across 2,643 writes to maintenance, while a separate 1,661.187ms lock wait
   dominated the worst single write. The active follow-up coalesces the
   best-effort manifest checkpoint to the existing snapshot cadence, adds
   bounded crash-safe sequence recovery, and preserves delivery and verdicts.

   Remaining refinements: exchange-call counts, candle-fetch concurrency,
   lower-level event-loop lag if heartbeat lag proves too coarse, and
   thresholded console warnings. Keep this off console unless thresholds are
   crossed.

   Work log:
   - 2026-06-27: Added `live-smoke-report` event-pipeline health summaries
     from existing `health.summary` counters, exposing queue depth, unfinished
     work, dropped events, sink errors, degraded count, worker-not-alive count,
     and stopping count without changing smoke verdict logic.

6. [ ] Exchange health and contract probes.
   Status: partial. VPS5 smoke on 2026-06-25 surfaced Kucoin authoritative
   balance/positions/open-orders `RequestTimeout` events. `live-smoke-report`
   now passively summarizes existing `remote_call.failed` events by
   bot/reason/surface/error type, terminal remote-call elapsed-time groups, and
   terminal remote-call health groups by bot/component/kind/surface. Explicit
   read-only exchange endpoint probes are not implemented. A later VPS5 smoke on
   2026-06-26 again surfaced Kucoin authoritative REST timeouts after PR #686:
   balance/positions/open-order fetches took roughly 98-140s before
   `RequestTimeout`, with websocket ping timeouts and one timestamp/nonce
   recovery in the same period. Subsequent PR #688/#690 smokes surfaced
   slow-but-successful remote-call categories even when no terminal failures
   occurred. PR #701 added active `ticker-endpoint-probe`
   `account_critical_health` summaries for balance, positions, and open-orders
   outcomes without adding exchange calls beyond the existing read-only probe.
   A VPS5 one-repeat authenticated probe on `binance_01` validated the summary
   shape and showed a follow-up: Binance `fetch_open_orders()` without a symbol
   fails as `ExchangeError`, so lower-impact/account-only probing should use
   exchange-aware open-orders shape. PR #703 added `--account-only`,
   `--skip-my-trades`, and an open-orders symbol fallback; a VPS5 account-only
   Binance probe validated `account_critical_health` success for all three
   account-critical surfaces. PR #741 added read-only `fetch_time`
   `time_sync_health` summaries and `--skip-time-sync`, counting unsupported
   exchanges separately from actual failures. PR #743 added
   `candle_freshness_health`, derived from the existing OHLCV tail probe
   results without adding exchange calls. PR #745 adds `fill_history_health`,
   derived from the existing first-symbol `fetch_my_trades` sample without raw
   trade/order ids or extra pagination calls. PR #747 added
   `rate_limit_health` request-pressure estimates. PR #749 added an opt-in
   bounded `--fill-history-pages` sample while keeping the default one-call
   behavior. PR #751 added
   endpoint latency summaries from existing probe outcomes, including
   open-orders fallback attempts and fill-history pages, without adding exchange
   calls. PR #753 added `exchange_surface_health`, deriving exchange/user-level
   notes from already-recorded open-orders, time-sync, fill-history, and OHLCV
   tail outcomes without adding calls.

   Add/refine explicit read-only probes for each configured exchange/account
   before or during smoke only when a concrete live exchange gap needs a more
   specific surface check. The passive smoke report now also exposes top-level
   remote-call health
   success/failure/throttle totals and a filtered
   `account_critical_remote_call_health` summary for a quick operator scan.

   Work log:
   - 2026-06-30: VPS5 smoke after PR #897 surfaced one Hyperliquid
     `fills.refresh_summary` `fill_refresh_failed` event after several
     successful fill refreshes in the same 10-minute window. This was separate
     from account-critical remote-call health, which stayed green, and should
     be treated as exchange/fill-refresh health evidence rather than a deploy
     failure.

7. [ ] Live config preflight/linter.
   Status: partial. `passivbot tool live-config-preflight` now emits a
   read-only offline JSON report for one config, covering identity hints, HSL
   side settings, HSL signal mode, approved/ignored universe counts with
   bounded samples, forager slot/staleness settings, and cache-related live
   settings. It also supports an optional local `--compare` baseline config
   diff for risk-relevant HSL, universe, forager, identity, and cache-setting
   changes. It does not load API keys, contact exchanges, or enforce startup
   policy.

   Add a preflight that explains risk-relevant config changes before startup:
   HSL enabled/disabled changes, HSL signal-mode changes, new approved/ignored
   universe size, forager staleness policy, max slots, exchange/user mismatch,
   and cache compatibility. The output should be structured and should not make
   trading decisions.

   Work log:
   - 2026-06-27: Added optional read-only `--compare` diff reporting for local
     two-config preflights, covering HSL signal/enabled changes, approved and
     ignored universe count/sample deltas, forager slots/staleness, identity
     hints, and cache live settings without credentials or exchange contact.
   - 2026-06-27: Added config-only cache readiness/root-hint reporting to
     `live-config-preflight`, including candle/fill/HSL setting attention,
     explicit artifact-not-scanned notes, and derived compare-mode readiness
     deltas without cache scans, credentials, exchange contact, or startup
     enforcement.

8. [ ] HSL dry-run preview for startup.
   Status: partial. `passivbot tool hsl-startup-preview` now emits a read-only
   offline JSON report for one config plus optional local monitor event
   artifacts. It reports configured HSL settings, latest observed local HSL
   status/cooldown/drawdown-to-red fields when present, and explicitly marks
   fresh current drawdown and startup panic-order prediction unavailable instead
   of fabricating them.

   Add a non-trading preview that reconstructs current HSL state and reports
   which scopes are GREEN/RED, cooldown status, current drawdown to red,
   and whether startup would emit panic orders. This would make risky restarts
   with changed HSL configs easier to reason about before live execution.

   Work log:
   - 2026-06-26: Added first-slice local/offline `hsl-startup-preview` tool.
     Remaining gap: safe full startup replay from local fill/account artifacts
     without exchange access.

9. [ ] Reason-code registry.
   Status: partial. Initial registry slice merged in PR #645.

   Centralize reason codes and event tags enough to prevent drift. The stream is
   much easier to search when `stale_ema`, `missing_canonical_candles`,
   `exchange_time_resync`, and similar codes are stable, documented, and tested.
   This pairs directly with reason-code filtering in the event query tool.

   Work log:
   - 2026-06-25: Added shared `EventTags` and `ReasonCodes` registries for
     common live-event tags/reason codes, migrated representative emitters
     without changing emitted strings, and added registry contract tests.
   - 2026-06-25: Added a focused AI doc for the live event tag/reason-code
     registry plus a docs drift test so stable values stay discoverable.

   Remaining refinements: migrate additional stable literals as nearby event
   surfaces are touched, and expand the docs when new stable query-facing values
   are added.

10. [ ] Operator console redesign from events.
    Status: partial. PR #646 improved event-projected summaries for already
    routed execution events. PR #677 mirrored the existing execution-loop error
    burst warning into a structured `health.summary` event without changing
    console volume or restart/backoff behavior. PR #707 added a throttled
    console projection for active coin-mode HSL positions using existing
    `hsl.status` distance-to-red metrics. PR #903 made HSL status visible in
    smoke reports from the same event source without increasing console noise.

    Continue moving console output to be a projection of structured events.
    Default console should focus on fills, positions, balance, order writes,
    meaningful risk/HSL/unstuck transitions, and compact "waiting because"
    summaries. EMA/candle/cache internals should stay structured DEBUG unless
    they directly explain a blocked trading action.

    PR #1246 merged and deployed the risk-status materiality slice. It keeps every
    five-minute trailing and unstuck observation in structured/monitor sinks
    while limiting console/text projection to first observations, qualitative
    or material numeric transitions, and hourly reminders. The change is
    observability-only and leaves status calculation, planning, risk, and order
    behavior unchanged. Two natural post-restart cadences proved both durable
    detail and suppression; settled smoke was hard-green. The first visible
    trailing line still measured 311 characters, so the active
    PR #1247, `codex/compact-trailing-status-console`, compacts that formatter
    to the normal 240-character budget without changing event data or
    admission.

    PR #1250 merged and deployed the bounded HSL startup-settings projection.
    Natural forager lines measured 163-167 characters versus 310-314 before,
    and the settled smoke was hard-green. The same restart exposed five
    admitted staged-refresh timing lines at 252-305 characters. The active
    `codex/compact-state-refresh-console` slice routes only the existing
    periodic summary and detail at or above the existing ten-second threshold
    to a compact console/text projection. Complete structured timing data and
    legacy fallback remain unchanged.

    PR #1251 merged and deployed the staged-refresh projection. Three natural
    admitted lines measured 155-171 characters versus 252-305 before, zero
    legacy duplicates appeared, and the settled smoke was hard-green. A
    retained natural Binance close-EMA fallback summary measured 308
    characters. The active `codex/compact-ema-fallback-console` slice makes the
    existing `ema.fallback_used` event its compact console/text owner only for
    active close fallbacks, preserving the fifteen-minute warning cadence,
    per-cycle durable events, recovery/forager suppression, and exact legacy
    fallback.

    PR #1252 merged and deployed the close-EMA projection. Three natural lines
    measured 158-225 characters, zero post-deploy legacy duplicates appeared,
    and the settled smoke was hard-green after one bounded GateIO startup
    retry. The same logs contained 21 natural initial-entry distance-gate
    blocked lines at 222-270 characters, 20 above budget. The active
    `codex/compact-entry-distance-console` slice compacts only their existing
    structured human projection while preserving payload, admission, legacy
    fallback, gate decisions, and trading behavior.

    PR #1253 merged and deployed the initial-entry distance-gate projection.
    Sixteen natural blocked lines measured 181-190 characters, none exceeded
    240, zero legacy duplicates appeared, and the settled smoke was hard-green.
    The same exact new logs contained candle-health transition summaries at
    257-263 characters on Binance, GateIO, and OKX. The active
    `codex/compact-candle-health-console` slice bounds only that human INFO
    projection while preserving full debug diagnostics, health calculations,
    transition cadence, readiness, fetch behavior, and trading behavior.

    PR #1254 merged and deployed the candle-health projection. Four natural
    lines measured 156-211 characters, none exceeded 240, and the settled
    smoke was hard-green. The same exact new logs contained forager selection
    transitions at 247-250 characters on Binance, GateIO, and OKX. The active
    `codex/compact-forager-selection-console` slice bounds only that human INFO
    projection while preserving the structured event, full DEBUG score and
    hysteresis diagnostics, transition cadence, Rust output, selection, and
    trading behavior.

    Of two post-PR #1220 console observations, PR #1233 resolved Hyperliquid
    balance jitter by admitting console/text balance changes only when the
    snapped balance changes while retaining every raw structured event. KuCoin
    emitted paired per-symbol required-EMA warnings with overlapping
    context and a long nested error; trace producer ownership and fail-closed
    semantics before changing either line because EMA availability is
    trading-critical.

    Work log:
    - 2026-06-30: Added value-safe `live-smoke-report` HSL status projections
      for full, summary, and brief reports. The local full report keeps HSL
      magnitudes for diagnostics; shareable summary/brief output strips raw
      drawdown, distance, and threshold values.
    - 2026-06-26: Added periodic `[risk] HSL[pside:symbol] status` console
      lines for active coin-mode HSL positions, including distance to red,
      drawdown, slot budget, realized PnL peak, and unrealized PnL. The
      structured `hsl.status` event remains the durable source.
    - 2026-06-27: Added `live-smoke-report` risk/HSL log-match counters,
      splitting text-log attention/hard matches into risk and non-risk buckets
      without changing smoke verdict logic.
    - 2026-06-27: Added `live-smoke-report` hard-failure and attention source
      breakdowns, making red or attention smokes attribute their verdict to
      monitor parse errors, invalid rows, structured events, log matches, and
      process liveness without changing verdict logic.

11. [x] Order lifecycle trace completeness.
    Status: initial reconstruction view merged. The order-wave/execution event
    chain exists, `live-event-query --trace-summary` can aggregate matched event
    types/statuses/reason codes/ID scopes, and `live-event-query --order-trace`
    reconstructs order-wave/action lifecycles from existing execution events.

    Keep tightening the end-to-end chain from Rust ideal order to executable
    order, gate decision, exchange payload, exchange response, local open-order
    refresh, confirmation, and fill. The target is that any create/cancel/missing
    order can be reconstructed from one id without reading code.

    Work log:
    - 2026-06-25: Added `live-event-query --cycle-trace`, grouping matched
      events by `cycle_id` with bounded timeline samples, aggregate trace
      summaries, and nested order traces.
    - 2026-06-25: Added incident-bundle integration for trace-summary,
      order-trace, and cycle-trace reports.
    - 2026-06-26: Added structured create-filter/defer events for existing
      pre-exchange create-order gates without changing gate behavior.
    - 2026-06-26: Mirrored the existing execution-loop error-burst warning into
      a bounded structured health event with reason code
      `execution_loop_error_burst`, preserving the existing warning threshold
      and console text.

    Remaining refinements: keep tightening producer coverage as nearby event
    surfaces are touched.

12. [x] Debug profile toggles.
    Status: initial targeted profile set merged. Rust, EMA readiness,
    remote-call, candle, fills, HSL, and execution profile slices are merged.

    Add narrow runtime/debug profiles that increase event detail for one domain:
    candles, fills, HSL, Rust payloads, order execution, or exchange calls. This
    avoids code patches or globally noisy DEBUG logs when diagnosing a live
    issue.

    Work log:
    - 2026-06-26: Added `logging.live_event_debug_profiles` and
      `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`, with initial `rust` support for
      bounded Rust orchestrator input-symbol and output-order samples on
      structured events only.
    - 2026-06-26: Added the `ema` debug profile, enriching
      `ema.unavailable` structured events with bounded parsed EMA type, span,
      and inner reason summaries while keeping default events compact and
      console output unchanged.
    - 2026-06-26: Added `remote_calls` debug-profile enrichment for candle and
      authoritative remote-call events, exposing bounded payload key shape,
      parameter key names, and correlation state without raw payloads or
      console output.
    - 2026-06-27: Added `candles` debug-profile enrichment for existing candle
      tail-projection and disk-coverage events, exposing bounded key-shape,
      timeframe, window, and missing-coverage counters without raw candle rows
      or console output.
    - 2026-06-27: Added a `fills` debug-profile slice for existing fill
      refresh and fill ingestion events, exposing bounded count, coverage, and
      key-shape metadata without raw source IDs or payload values.
    - 2026-06-27: Added an `hsl` debug-profile slice for existing HSL
      status, transition, replay, red-trigger, and cooldown events, exposing
      bounded event key, metric key, and latch/cooldown state-shape metadata.
    - 2026-06-27: Added an `execution` debug-profile slice for existing
      order-wave, order-write, create-filter, and confirmation events, exposing
      bounded key-shape/counter metadata without raw order payload values.

    Remaining refinements: add new targeted profiles only as diagnostics need
    deeper live evidence.

13. [ ] Cache integrity doctor.
    Status: partial. Initial read-only local cache smoke doctor, cache-family
    summaries, candle coverage evidence, fill/HSL metadata evidence, and
    report-only warm-cache readiness evidence are merged. Deeper report-only
    metadata compatibility evidence for candle known gaps, fill coverage proof,
    and HSL artifact/timestamp compatibility is also merged.

    Work log:
    - 2026-06-25: Added `passivbot tool cache-integrity-doctor`, which reports
      local cache root presence, aggregate file/size counts, and empty/corrupt
      JSON, NDJSON, and NPY artifacts without writing or touching live behavior.
    - 2026-06-26: Added per-root and aggregate cache-family summaries plus
      family tags on cache-doctor issues.
    - 2026-06-27: Added first-slice v2 candle coverage evidence from local
      `.valid.npy` artifacts, including coverage windows, valid row counts,
      suspicious interior gap samples, and non-enforcing warm-cache evidence
      labels.
    - 2026-06-27: Added fill/HSL metadata evidence from local JSON/NDJSON
      artifacts, including fill `pnl_contract` compatibility counts, fill
      coverage timestamps, known-gap counts, and HSL/risk state timestamp
      summaries without repair or enforcement.
    - 2026-06-27: Added report-only warm-cache readiness evidence derived from
      already-scanned candle/fill/HSL metadata, including core evidence labels,
      reasons, missing families, suspicious gap counts, and per-family
      timestamp context without making startup or trading decisions.
    - 2026-06-27: Added deeper report-only metadata compatibility evidence for
      cache-integrity-doctor, including candle `index.json` known-gap
      no-trade reason counts, fill current-contract coverage proof labels, and
      HSL artifact/timestamp compatibility fields without repair or startup
      enforcement.
    - 2026-06-27: Added report-only candle boundary-gap clarity to
      cache-integrity-doctor, splitting interior gaps from boundary gaps,
      leading missing rows, and trailing shortfall rows in coverage summaries
      and warm-cache readiness without repair or startup enforcement.

    Remaining refinements: add deeper candle/fill/HSL metadata compatibility
    checks and synthetic/no-trade assumptions without making trading decisions.

14. [ ] Supervisor/process model.
    Status: partial. `live-smoke-report --supervisor-config` now reports
    expected/matched/missing `passivbot live` processes from a tmuxp-style
    config, duplicate configured-command matches, and extra/orphan-like live
    processes from local command matching. Incident bundles can include that
    smoke snapshot. The process classification is explicitly not tmux pane
    ownership.

    The tmux/tmuxp setup is workable, but repeated live smoke showed room for a
    stricter supervisor contract: clear per-bot status, bounded stop/restart,
    captured exit reason, backoff policy, and health heartbeat. This could stay
    outside the trading core but should consume the same event stream.

15. [ ] Fake-live regression scenarios.
    Status: partial. First focused offline observability regression coverage is
    underway for existing live smoke/event-pipeline health behavior.

    Build more fake-exchange/fake-live scenarios for failures repeatedly seen in
    real work: stale candles, missing EMA inputs, fill pagination gaps, timestamp
    resync, queue overflow, slow shutdown, and exchange-call ambiguity. These
    should prove observability behavior without risking live accounts.

    Work log:
    - 2026-06-27: Added a focused offline smoke-report regression for
      multi-bot event-pipeline queue/drop/sink-error health aggregation,
      proving existing queue-overflow observability without live bots,
      exchange calls, or behavior changes.
    - 2026-07-10: A direct two-step coin-HSL fake-live run reached RED, posted
      and filled the panic close, then failed the next cycle because the staged
      planner still considered `balance,fills,open_orders,positions` missing
      for the new epoch before market-snapshot refresh. The existing 29-test
      fake-live suite and seven pside HSL end-to-end scenarios remain green;
      add a dedicated coin-mode post-panic scenario and fix the epoch handoff
      separately from HSL episode-finalization math.
    - 2026-07-18: Canonical master now completes that coin-mode post-panic
      handoff: the panic close fills, authoritative account surfaces refresh,
      cooldown ends, and normal selection resumes without the former missing
      surface failure. The active regression slice persists the already-emitted
      redacted live-event envelopes in fake-live artifacts and makes the
      scenario prove a later available planning snapshot with no
      `planning.unavailable` handoff event.

16. [x] Websocket reconnect diagnostics.
    Status: completed by PR #1170.

    VPS5 smoke after PR #728 caught a fresh OKX ccxt-pro websocket callback
    traceback after `connection lost ... RequestTimeout`; the bot continued and
    the settled 2-minute smoke was green, but the current signal is only a raw
    text-log traceback. Add structured websocket reconnect/callback diagnostics
    where practical, or improve smoke classification so known dependency
    callback races are grouped with surrounding reconnect context instead of
    requiring manual log inspection.

    Work log:
    - 2026-07-10: PR #1170 adds a bounded
      structured/monitor-only reconnect event at the existing throttled logger.
      It preserves reconnect control flow and text diagnostics while excluding
      exception messages, tracebacks, payloads, and URLs from the event.

17. [ ] Forager active-symbol EMA readiness handoff.
    Status: partial. First hardening slice allows active/normal forager symbols
    to carry forward bounded cached real-candle qv/log-range EMA values during
    fill handoff. Broader create-side readiness gating and fake-live coverage
    remain open.

    VPS5 smoke after PR #735 caught an OKX recovery case where `AAVE` was
    selected/posted as a forager initial entry, filled, and became an active
    long while its forager volume/log-range EMA basis was still warming. The
    next execution loop raised one hard error:
    `missing required forager EMA for active/normal symbol AAVE/USDT:USDT:
    volume_spans= log_range_spans=996`. The bot recovered after background
    warmup completed, and the settled 2-minute smoke returned `ok=true`, but
    the transient hard restart/backoff is undesirable.

    Clarify and implement the handoff contract for a flat symbol that becomes
    active during or just after forager selection: either block the create until
    required active-symbol forager EMA inputs are provably ready, or explicitly
    carry forward bounded/stale forager feature values through the first active
    cycle with structured readiness metadata. Do not fabricate neutral
    volume/log-range values, and do not weaken protective/risk actions.

    Work log:
    - 2026-06-27: Added bounded cached qv/log-range EMA carry-forward for
      active/normal forager symbols, reusing local real-candle EMA metrics
      within the configured forager staleness cap. Candidate-only symbols still
      remain unavailable instead of receiving synthetic ranking tails.
    - 2026-07-16: Active `codex/smoke-forager-feature-health` exposes the
      producer's existing structured-only feature-unavailability evidence in
      bounded smoke-report projections. It does not change readiness or the
      create-side handoff contract.
    - 2026-08-01: Architecture-tightening work removes completed candles from
      the global staged planner barrier and lets strict-by-default Rust scope
      explicitly absent live EMA inputs to actual consumers. This addresses
      held-symbol liveness and preserves stale-order cancellation, but does not
      by itself settle flat-to-active forager ranking readiness.

18. [ ] Binance hourly hedge-mode/config refresh traceback classification.
    Status: structured event, smoke projection, performance projection, and live
    hourly emission evidence are complete. The historical Binance `-4084`
    classification remains open only if it recurs; current live evidence shows
    successful Binance refreshes. The current logging slice makes recovered
    failures explicit without changing smoke verdicts.

    VPS5 smoke after PR #892 deployed to `v8@7e7ce16f` returned hard-red from
    a non-risk text-log traceback in the Binance bot while all five live
    processes remained running and structured monitor events showed no hard
    problem event. The surrounding lines were:
    `error setting hedge mode: binanceusdm {"code":-4084,"msg":"Method is not
    allowed currently. Upcoming soon."}` and `error with maintain_hourly_cycle`.

    Target contract: recurring exchange-config maintenance failures should be
    represented as structured, bounded, exchange-surface diagnostics with clear
    severity and retry/backoff context. If a failure is trading-critical, the
    structured stream should make that explicit. If it is expected/non-critical
    on an already-running Binance futures account, it should not appear only as
    a raw traceback that makes operator smoke red without explaining whether
    trading was impaired. Do not weaken fail-loud behavior for required startup
    exchange config or order construction.

    Investigation directions: inspect the hourly `maintain_hourly_cycle` /
    hedge-mode refresh path; distinguish startup-required config from recurring
    maintenance refresh; add a structured `exchange.config_refresh` or similar
    event if the path remains in live; and adjust smoke/report classification
    only after the event contract makes criticality explicit.

    Work log:
    - 2026-06-30: PR #894 added off-console/text structured
      `exchange.config_refresh` events around hourly maintenance
      `init_markets` refresh success/failure. The event includes bounded
      sanitized failure text, `error_type`, context/operation labels, elapsed
      timing, and distinct reason codes while re-raising original refresh
      exceptions. VPS5 was pulled to `796ceb38`, but bots were not restarted,
      so live emission evidence is pending. Follow-up classification should use
      the structured event and must avoid down-classifying startup-required
      config or order-construction failures.
    - 2026-06-30: PR #896 added read-only `live-smoke-report` full, summary,
      and brief projections for `exchange.config_refresh` health. The smoke
      projection excludes raw free-text `data.error`, keeps only bounded labels,
      `error_type`, status/reason counts, and timing fields, and does not
      change smoke verdict or text-log classification. VPS5 was pulled to
      `53b8accb`; a 5-minute no-restart smoke was hard-green and showed the
      new `exchange_config_refresh` brief section with `total=0`, as expected
      before the next bot restart loads the event producer.
    - 2026-06-30: After PR #897 deployed at `aebc3667`, bots were restarted so
      running processes would load the PR #894 event producer. The following
      fresh smoke windows were otherwise clean, but `exchange_config_refresh`
      still reported `total=0` and a focused three-hour event query found no
      `exchange.config_refresh` events. This is not yet proof that the Binance
      `-4084` maintenance traceback is fixed or classified, because no sampled
      window has proven an hourly refresh occurrence after restart.
    - 2026-07-09: PR #1162 added a bounded `live-performance-report` health and
      elapsed-timing projection over existing `exchange.config_refresh` events.
      Post-deploy VPS evidence found 14 real hourly refresh events across all
      five bots: 13 succeeded and one Kucoin timeout was followed by success;
      Binance had three successes. No live `-4084` recurrence was observed.
    - 2026-07-09: Branch `codex/v8-exchange-config-refresh-recovery` adds
      latest-per-bot status, latest-failed-bot, and recovered-bot aggregates to
      smoke and performance reports so a historical timeout followed by success
      is not presented as unresolved. It does not change verdicts, retries,
      exception propagation, exchange I/O, or trading behavior.
    - 2026-07-10: Branch
      `codex/v8-exchange-config-response-diagnostics` replaces raw successful
      response rendering with one bounded, value-safe formatter across the
      shared CCXT and account-level connector call sites. Connector-specific
      failure logs and per-symbol methods that currently swallow exceptions are
      intentionally unchanged; deciding their propagation contract remains
      separate trading-critical work.
    - 2026-07-10: Branch `codex/v8-exchange-config-error-diagnostics` bounds
      the parent per-symbol retry log and connector-local exchange-config logs
      in Binance, Bitget, Defx, Hyperliquid, KuCoin, and OKX to operation,
      symbol, retry, canonical known-code, and exception-type context.
      Catch/rethrow or swallow behavior remains unchanged. Outer startup/runtime
      traceback and structured-event raw-error retention are separate
      logging-policy work; connector propagation semantics remain separate
      trading-critical work.

19. [ ] Exact-head semantic review check enforcement.
    Status: open. The autonomous review loop already re-checks a PR head before
    posting and avoids duplicate reviews, but its comments and polling cadence
    are advisory. PR #1205's final recovery head merged between polling wakes
    before every intended semantic reviewer had independently reviewed that
    exact head. This is a repository-governance race, not a defect in the merged
    monitor behavior.

    Target contract: when a semantic reviewer is configured as mandatory, expose
    its verdict as a GitHub status/check bound to the exact PR head and enforce
    that check through repository merge protection. Every new head must return
    the check to pending. Older reviews, comments, or successful checks must not
    satisfy the new head, and merge readiness must still require all configured
    current-head review and CI gates.

    Implementation directions: use deterministic metadata polling, preserve
    reviewer identity plus reviewed SHA when reconciling review history, grant
    only the GitHub permissions needed to publish the check, and make failure to
    publish visible without substituting a comment as success. Keep advisory
    review loops clearly labeled until branch protection enforces the check.

    Current evidence: after the v8.0.0 default-branch cutover on 2026-07-14,
    `master` protection enforced strict `Python 3.12` and `Rust` checks plus
    conversation resolution, but required zero formal approvals and no
    semantic-review status/check. Hermes, Grok, or self-authored `COMMENT`
    verdicts therefore remain advisory until this item is implemented. Reviewer
    schedulers must also migrate their base-branch filters and compact cache to
    the live default branch before the cutover can be considered complete.

20. [ ] Per-asset collateral, debt, and valuation balance events.
    Status: partial, OKX plus Binance plus Hyperliquid unified-account totals.
    Connector-specific slices normalize OKX's already-fetched
    `info.data[0].details`, Binance's already-fetched CCXT unified balance maps,
    and proven Hyperliquid unified `info.balances` coin/total rows into bounded
    deterministic asset rows. Hyperliquid non-unified payloads remain explicitly
    unavailable; HIP-3 position responses remain out of scope. Gate.io and
    KuCoin remain fixture/contract work. These slices add no exchange calls or
    changes to scalar balance, planning, orders, risk, or console materiality.
    Other connector parsers and any broader asset contract remain open. A
    2026-07-14 evaluation confirmed that the existing
    `balance.changed` event and console projection expose only aggregate raw
    balance, hysteresis-snapped balance, equity, deltas, and source. The
    authoritative refresh already receives the exact raw balance response
    alongside the normalized scalar, but `DataPacketMetadata` retains only a
    bounded hash/reference and the staged refresh discards the raw response
    after metadata capture. Asset quantities, explicit liabilities, per-asset
    USD values, and valuation prices therefore do not reach the event stream.

    This is not a formatter-only change. Balance response contracts differ by
    connector: examples include OKX account `details`, Bybit unified-account
    `coin` rows, Hyperliquid core/spot clearinghouse state, Gate.io
    multi-currency margin fields, Defx collateral rows, and generic CCXT
    `total`/`free`/`used` mappings. In addition, `balance.changed` currently
    triggers only when aggregate raw or snapped balance changes, so a
    collateral-composition change can be missed when the account total remains
    equal.

    Target contract: normalize the already-fetched authoritative balance
    response into a bounded `asset_balances` collection without making any new
    exchange or ticker request. Each row should identify the asset and include
    only values proven by that connector's response, such as total/net amount,
    free/used amount, explicit debt/liability, USD value, derived or reported
    USD price, collateral-enabled state, and field provenance. Missing values
    remain absent; do not infer debt from an undocumented negative field or
    invent a price. A derived price is allowed only when finite amount and USD
    value from the same coherent response make the derivation unambiguous.

    The durable event must include bounded count/truncation metadata and a
    deterministic ordering, while the console should show a shorter clean
    sample (for example the quote asset, every nonzero explicit debt, and the
    largest collateral values). Keep full raw account payloads out of events,
    console, text logs, and monitor artifacts. Emit a balance event when either
    the aggregate transition or normalized composition signature changes, so
    equal-total collateral substitutions remain observable. Diagnostic
    normalization must not change the scalar balance used for trading or add a
    new failure mode to the authoritative refresh; unsupported or malformed
    breakdown extraction must instead be explicit and bounded rather than
    silently replaced with an empty successful snapshot.

    Implementation directions: define one normalized balance-asset row
    contract and an exchange hook at the point where `capture_balance_snapshot`
    still owns the raw response; carry the normalized snapshot, not the raw
    response, through staged publication; track a deterministic composition
    signature separately from trading balance state; enrich
    `balance.changed`; and extend the dedicated console formatter with a
    bounded sanitized asset summary. Implement connector parsers in focused
    reviewable slices, starting with multi-collateral connectors that already
    expose authoritative USD values and liabilities. Keep the logging-overhaul
    loop paused while this item is only backlog work.

    Required tests: connector fixtures for quantities, USD values/prices, and
    explicit debt signs; missing/non-finite/zero fields; stable ordering and
    truncation; no raw payload leakage; no additional exchange calls; unchanged
    scalar balance behavior; composition-only event emission at equal aggregate
    balance; no duplicate event for an unchanged composition; bounded console
    formatting and sanitization; and explicit diagnostic-unavailable behavior
    for unsupported or malformed breakdowns.

    Work log:
    - 2026-07-18: Binance follow-up normalized only CCXT's documented unified
      `total`, `free`, `used`, and explicit `debt` maps from the same fetched
      response. It excludes raw `info`, arbitrary fields, non-finite values,
      and valuation inference while reusing existing bounds and signatures.
    - 2026-07-18: OKX-first slice added bounded rows for `ccy`, `cashBal`,
      `eqUsd`, `upl`, `collateralEnabled`, and explicit `liab`, with per-field
      provenance, deterministic truncation, an omitted-row-sensitive internal
      signature, and a two-row sanitized console sample. Generic, Binance, and
      Hyperliquid staged refreshes carry normalized unavailable diagnostics
      rather than raw balance payloads. No ticker/API call, scalar balance
      calculation, refresh cadence, planning, order, or risk behavior changed.

21. [ ] Historical secret-bearing text-log inventory and remediation.
    Status: partial. PR #1265 merged and deployed the bounded, value-free,
    read-only inventory and dry-run validation contract. PR #1267 recognizes
    credential query parameters in scheme-less request paths while preserving
    the same report schema. PR #1268 merged and deployed an aggregate-summary
    projection that removes per-file detail from shareable/operator output
    while preserving aggregate scan evidence. Quarantine or purge remains
    separate and requires explicit
    operator approval. A read-only VPS5 console-length audit on 2026-07-15
    accidentally admitted old untimestamped traceback fragments and confirmed
    that retained May text logs include raw private websocket URLs/tokens and
    full exchange error bodies. Current recent-window producers and smoke paths
    use bounded redacted diagnostics, but historical disk retention still
    violates the no-secret sink policy. The observed websocket tokens are
    likely short-lived, but expiry must not be treated as a redaction control.
    A value-free scan of 306 timestamped lines from the five current canonical
    logs after the PR #1249 restart found zero private-websocket-query,
    authorization, API-key-label, or raw-HTML-body matches.

    Target contract: inventory historical secret-like text-log artifacts
    without printing matched values; report only bounded counts, file identity,
    age, and stable hashes. Verify the current producers no longer emit each
    detected class, then define an operator-approved quarantine or purge plan
    that preserves the minimum forensic metadata required for incident review.
    Do not rewrite, delete, rotate, upload, or copy existing VPS artifacts
    without explicit authorization.

    Work log:
    - 2026-07-16: PR #1265's bounded VPS5 dry run scanned 40 of 1,182 files,
      identified ten retained May logs with private websocket/query credential
      classes, emitted no values or source lines, and changed no artifacts.
    - 2026-07-16: PR #1267's bounded VPS5 scan classified 144 secret-query
      matches versus 143 private-websocket-query matches, proving one additional
      non-websocket query fragment was found without exposing values.
    - 2026-07-16: PR #1268's bounded VPS5 summary scanned 40 of 1,182 files and
      8,153,519 bytes, retained the 144/143 query-class counts, omitted
      per-file paths/ages/hashes, and reported zero read or discovery errors.

    Required validation: fixtures for private websocket URLs, authorization
    material, signatures, API keys, query tokens, raw HTTP bodies, and benign
    lookalikes; value-free report output; bounded scanning of large/rotated
    logs; current-producer regression tests; and a dry-run VPS inventory before
    any destructive remediation proposal.

## Merged Work Log

## Suggested Priority

Near-term highest leverage:

1. Forager active-symbol EMA readiness handoff.
2. Exchange health and contract probes.
3. Live restart/smoke automation.
4. Operator console redesign from events.
5. Startup phase budget tracking.
6. HSL dry-run preview.
7. Cache integrity doctor coverage/readiness refinements.

These make every later live debugging session cheaper and provide direct
feedback on whether the event stream is actually answering operator questions.
