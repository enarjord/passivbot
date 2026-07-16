# Live Operations Improvement Backlog

## Purpose

This backlog captures high-value improvement areas noticed while running the
reviewed PR loop, repeatedly restarting VPS5 bots, and using the new live event
pipeline to diagnose behavior. It is intentionally higher level than the active
logging-overhaul plan. Each item below should become its own small reviewed
slice before implementation.

The logging overhaul remains the foundation: a centralized event stream should
make the items below easier to prove, test, and operate.

This backlog feeds the logging-overhaul loop selectively. Items are in-scope for
that loop when they improve diagnostics, smoke evidence, incident reconstruction,
or operator workflow needed to complete the logging work. Trading-behavior bugs
found through better observability should remain visible here, but should become
separate focused trading-path PRs unless they block observability validation.

Update policy:

- Keep open work in `High-Value Follow-Ups` with checkboxes and short status
  notes.
- When a PR completes all or a meaningful first slice of an item, update the
  item status and add an entry to `Merged Work Log`.
- Leave follow-up refinements attached to the item instead of hiding them in the
  completed log.
- Add newly discovered operations gaps here before spawning or accepting a PR for
  them.

Related detailed plans:

- `docs/plans/live_logging_overhaul_plan.md`
- `docs/plans/live_performance_readiness_goals.md`
- `docs/plans/live_restart_shutdown_and_warm_cache_handoff.md`

## High-Value Follow-Ups

0a. [ ] Critical: HSL false-panic recovery and preflight.
   Status: open. VPS3 `ebybitsub03` evidence from 2026-06-28/29 showed a wrong
   `close_panic_long` on XMR under `hsl_signal_mode=unified` with
   `balance_override=1000`. HSL reconstructed a synthetic account-level peak
   near `1213` from overridden balance plus historical realized PnL, while the
   account was not near a legitimate drawdown that should have triggered RED.
   PR #839 added the immediate fail-loud runtime guard for
   `balance_override` plus account-level HSL replay. `live-config-preflight`
   flags the same unsafe startup contract before launch, and
   `live-smoke-report --processes` now adds a read-only local config check for
   running/expected live commands. An operational gap remains for recovery and
   prevention.

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

0. [ ] Critical: HSL startup replay latency before protective panic.
   Status: open. Binance VPS5 incident evidence from 2026-06-26 shows
   `hsl_signal_mode=coin` startup history reconstruction loaded
   `symbols=24 pairs=24 rows=43201 fills=2704 panic_events=0` at
   `16:19:33Z`, then completed at `16:46:37Z` after replaying `985965` rows
   in `1623.4s`. The XLM protective close was not posted until `16:48:06Z`.
   This is too slow for a live safety path: correctness from exhaustive replay
   is valuable, but not at the cost of delaying panic/protective action by
   tens of minutes while a held coin may already be beyond red threshold.
   A 2026-06-29 VPS5 restart on `v8` head `7ce1aec9` re-confirmed the issue:
   after roughly 20 minutes, Binance, Kucoin, GateIO, and OKX were still live
   but not `READY`; their current logs had HSL coin reconstruction start lines
   and no completion lines. Process smoke stayed hard-green, so this remains a
   startup readiness latency issue rather than a process crash.

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

   Work log:
   - 2026-06-29: After PR #858 deployed to VPS5 at `8c908e72`, new structured
     `hsl.replay.progress` events showed all four forager bots reached
     `hsl_history_inputs_loaded` and `hsl_price_history_fetch_started`, but no
     bot had emitted `hsl_price_history_fetch_completed` or timeline replay
     stages in the early smoke window. This localizes the current bottleneck to
     HSL price/candle history fetching before dense timeline replay begins.
   - 2026-06-30: After PR #897 deployed at `aebc3667` and bots were restarted,
     a 20-minute smoke showed three forager bots still in active coin-HSL
     replay: Binance at pair `10/27` after `645s`, OKX at pair `22/28` after
     `1522s`, and Kucoin still in `price_history_symbol_fetch_started` after
     `1541s`. GateIO reached ZEC cooldown handling, but its startup timing
     reported `full-warmup` around `1843s`. This confirms the latency is still
     present with the current HSL config and should remain a top-priority
     trading-path optimization.
   - 2026-06-30: A later VPS5 smoke on the same deployment showed OKX finished
     coin-HSL replay after `1904.391s`, logged `hsl-ready=2462.54s`, then
     finalized `HSL[long:ZEC/USDT:USDT]` RED without an exchange order and
     entered cooldown. This is a second post-restart exchange/account example
     of the same latency pattern and strengthens the case for a position-first
     protective classification path before full cooldown replay completes.
   - 2026-06-30: PR #899 added read-only `live-smoke-report --brief`
     projection of the worst active HSL replay elapsed time, latest-event age,
     and stage counts from existing sanitized HSL replay groups. This does not
     reduce startup latency or change trading behavior, but it makes repeated
     smoke loops surface active replay latency without opening the full report.
   - 2026-06-30: PR #901 added read-only `live-smoke-report --brief`
     projection of the worst completed HSL replay elapsed time from existing
     sanitized completed replay groups. This keeps settled post-replay smoke
     useful after `active_bots` falls back to zero, while still leaving the
     actual HSL startup latency optimization open.
   - 2026-06-30: PR #903 added read-only `live-smoke-report`
     `risk_events.hsl_status` projections from existing `hsl.status` monitor
     events, with value-safe shareable summary/brief output. This makes current
     HSL tier/symbol/bot status visible in smoke loops, but does not reduce
     startup replay latency or change panic/cooldown behavior.
   - 2026-07-02: PR #988 added active HSL replay stale/long-running
     classification to `live-smoke-report`, making a missing completion event
     visible while a bot is still startup-blocked. A subsequent VPS5 deploy
     exposed Kucoin as stopped after a terminal coin-HSL startup validation
     failure, not merely slow replay.
   - 2026-07-02: PR #989 narrowed coin-mode HSL price replay strictness to
     current-position and historical panic/cooldown symbols. Kucoin no longer
     failed on a flat, non-panic historical AVAX fill and no candle cache was
     used for HSL price replay (`cm_cache=0.00 MiB`). However, the remaining
     dense row replay still took `1237.7s` and the bot reached READY only after
     about `1445s`, confirming the next fix must target the
     `timeline_minutes * pairs` replay loop rather than only candle fetching or
     missing-UPnL validation.
   - 2026-07-02: The same Kucoin restart emitted a recovered first-cycle
     `InvalidNonce` after the long startup and finalized a NEAR coin-HSL RED
     cooldown from replayed history. These should be handled as separate
     follow-ups: refresh exchange time before the first cycle after very long
     startup, and review whether smoke should distinguish expected HSL
     red/cooldown risk events from software hard failures. Branch
     `codex/v8-smoke-recovered-time-sync` addresses the smoke side for
     timestamp/nonce recovery by classifying same-cycle successful
     `exchange.time_sync` as recovered problem evidence instead of a persistent
     hard failure.
   - 2026-07-10: The held-readiness prerequisite audit found that live coin HSL
     normalizes drawdown by `balance / n_positions`, while the Rust backtest
     uses `balance * total_wallet_exposure_limit / n_positions`. The live
     contract and metric regression explicitly make sensitivity independent of
     exposure limits, so reconcile the backtest denominator in a focused
     parity PR before using full-replay equivalence as the held-readiness gate.
     The current prerequisite branch separately centralizes post-episode
     no-restart/cooldown finalization in Rust without changing trigger math.
   - 2026-07-10: PR #1174 merged and deployed the shared post-episode
     transition. The dependent denominator-parity branch now moves coin slot
     budget/raw drawdown into Rust, removes TWEL scaling from backtest, and
     preserves TWEL activation plus intentional dynamic backtest slot counts.

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
   policy suppresses contextless hard-looking log fragments. The safe restart
   orchestration contract is not implemented. A 2026-06-30 follow-up made the
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

   Work log:
   - 2026-06-26: Added first-slice plan-only
     `passivbot tool live-restart-smoke-plan`, which parses a tmuxp-style
     supervisor config and emits a structured dry-run restart/smoke plan with
     bot commands, stop/start/check phases, timeouts, escalation ladder, repo
     checks, and smoke-report command wiring. It explicitly rejects execution
     and does not signal processes, invoke tmux, SSH, pull code, start bots,
     contact exchanges, or load credentials.
   - 2026-06-27: Added process-signal safety guidance to the plan-only restart
     smoke planner after a VPS5 restart shell matched its own broad
     `passivbot live` process pattern. The planner now records that future
     execution must use exact tmux panes or exact canonical process rows, and
     rejects broad process-pattern kill/signal commands.
   - 2026-06-29: VPS5 deploy of PR #858 at `8c908e72` re-confirmed that
     shutdown responsiveness is uneven during HSL replay. Hyperliquid stopped
     promptly on the first signal, but Binance, GateIO, Kucoin, and OKX were
     still alive after roughly 20 seconds and required a second signal. Future
     restart orchestration should record per-bot shutdown timings, identify the
     current blocking phase, and apply an explicit escalation ladder while
     leaving bots running after smoke.
   - 2026-06-30: After PR #880 deployed at `74a07640`, VPS5 5-minute brief
     smoke hit a 30-second timeout wrapper and a narrower 1-minute smoke took
     roughly 15-19 seconds while reporting `ok=true`. With text logs disabled,
     the report still skipped roughly 14.7k old monitor events to summarize
     roughly 600 in-window events. First-pass fix: make `live-smoke-report`
     reuse one monitor-event parse for both monitor validation/summary and
     windowed smoke aggregates instead of parsing every event segment twice.
   - 2026-06-30: Added an opt-in `live-smoke-report --event-tail-lines`
     parser bound for repeated recent-window smoke checks over large current
     monitor segments. The default remains full monitor-event validation; the
     opt-in path reports tail-limit metadata in `event_window` so bounded smoke
     evidence is explicit.
   - 2026-06-30: PR #897 deploy required a bot restart so running processes
     would load the new `exchange.config_refresh` event producer. Kucoin did
     not exit within a 180-second Ctrl+C observation window and required SIGTERM
     before reload. The subsequent 2-minute brief smoke was green with all five
     configured bots running. This adds another concrete data point for
     per-bot shutdown timing, current-phase attribution, and a bounded
     escalation ladder in future restart automation.
   - 2026-06-30: PR #899 added worst-active-HSL replay elapsed/event-age/stage
     counters to `live-smoke-report --brief`, so short operator smoke loops can
     see whether startup HSL replay is still active and how old the active
     stage evidence is. VPS5 deploy smoke at `e1fcb038` was green, but its
     sampled window had `hsl_replay.active_bots=0`, so the active fields were
     not populated by live data in that run.

   Remaining refinements: safe pull/stop/start orchestration remains open.
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
   projections against prior p95 phase baselines. Explicit durable phase budget
   configuration/events are not implemented.

   Startup currently has timing events, but the next step is durable budget
   accounting by phase: account-critical fetches, fill/PnL refresh, active
   candle readiness, forager warmup, HSL replay, Rust planning, and READY.
   Store both current timings and rolling baselines so regressions stand out
   after short-downtime restarts.

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
   which symbols are green/yellow/red, cooldown status, current drawdown to red,
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

    PR #1207 deployed an aligned `position.changed` human transition with
    base/effective WEL utilization while preserving the complete structured
    event; a natural live Hyperliquid change verified the projection. Zero
    values in fill, balance, and entry-gate summaries are already retained
    because their numeric helper returns formatted strings; the prior
    truthiness-gap note was incorrect. PR #1210 renders frequent balance changes
    as exact raw/snapped transitions. PR #1215 removed the fill legacy/event
    duplicate after the structured projection gained timestamp, pending-PnL,
    bounded traceability, and bulk-summary semantics. PR #1216 merged and
    deployed at `13e6e484cf20b1265f2b4874b14ff7ab32d10bfd`: four natural
    periodic health lines proved compact single ownership and sane RSS, with
    `ok=true` settled smoke (`414/414` remote, `70/70` account-critical,
    five process/config checks, `12/12` fill refreshes, and zero tracked
    repository changes).
    PR #1217 then removed the natural KuCoin execution-loop error-burst
    structured/legacy duplicate while preserving the legacy fallback when no
    structured console sink exists. It merged and deployed at
    `6599fba08cadffac99ce6a1ce2bfd3f58ca3fa15`; after a real transient KuCoin
    timeout aged out, the final two-minute smoke was green with `284/284`
    remote and `62/62` account-critical calls, all five bots running, zero
    pipeline failures, and a clean tracked repository. Retained OKX logs then
    exposed the next adjacent duplicate: structured
    `execution.cancel_ambiguous_terminal` followed by the legacy full-account
    confirmation summary for the same symbol.
    PR #1218 removed that ambiguous-cancel duplicate and added the compact
    full-account-confirmation cue without changing cancellation or confirmation
    behavior. It merged and deployed at
    `4bf7706d79f2e2404f785195973d13ea49c31efb`; the settled two-minute smoke was
    green with five matching bots, `472/472` remote calls, `25/25`
    account-critical calls, `7/7` fill refreshes, four complete active HSL
    replays, zero hard/log/monitor/pipeline failures, and a clean repository.
    Fresh Binance, GateIO, and OKX logs then exposed the next adjacent duplicate:
    legacy `[entry] initial entry staged but not placed` immediately followed by
    structured `entry.initial_distance_gate_blocked`. The blocked and cleared
    structured events already carry the bounded operator context and should own
    normal console/text output while retaining legacy fallback when structured
    console infrastructure is absent.
    PR #1219 removed that entry-distance duplicate and deployed at
    `23d9e72af180e8636de7f80cdff8178a60e61937`. Its final bounded smoke was
    green with five config-valid bots, `299/299` remote and `32/32`
    account-critical calls, six fill refreshes, complete required HSL replay
    work, zero hard/log/monitor/pipeline failures, and a clean repository.
    Natural post-deploy blocked events on Binance, KuCoin, GateIO, and OKX
    proved structured single ownership. The same GateIO log exposed a separate
    duplicate: legacy `initial entry blocked by min effective cost` immediately
    followed by structured `entry.min_effective_cost_blocked`. The per-block
    structured event should own normal detail output when its console sink is
    available, while the distinct throttled aggregate summary for larger block
    sets remains operator-visible.
    PR #1220 removed that min-effective-cost duplicate and deployed at
    `9773889ecb8a396bec31e1e11c326aed9fa2cbe7`. Its final bounded smoke was
    green with five config-valid bots, `279/279` remote and `44/44`
    account-critical calls, seven fill refreshes, complete required HSL replay
    work, zero hard/log/pipeline failures, and a clean repository. Natural
    GateIO output proved structured single ownership. Static follow-up found
    direct duplicate ownership in `_log_realized_loss_gate_blocks`: the legacy
    warning immediately precedes an already console/text-routed structured
    warning. Migrate that detail with the same emitter/pipeline/sink fallback
    contract while preserving every gate decision and throttle.

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

    Two post-PR #1220 console observations remain deferred rather than folded
    into the realized-loss slice. Hyperliquid balance lines surfaced signed
    floating-point jitter near `1e-13`; decide a console-only materiality or
    display-normalization contract without changing structured balance data.
    KuCoin emitted paired per-symbol required-EMA warnings with overlapping
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

    Add a read-only doctor for candle/fill/HSL caches that reports coverage,
    metadata compatibility, corrupted shards, suspicious gaps, synthetic/no-trade
    assumptions, and whether a short restart can safely use a warm-cache path.
    This supports the separate warm-cache restart work.

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
    Status: open. A 2026-07-14 evaluation confirmed that the existing
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

21. [ ] Historical secret-bearing text-log inventory and remediation.
    Status: open. A read-only VPS5 console-length audit on 2026-07-15
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

    Required validation: fixtures for private websocket URLs, authorization
    material, signatures, API keys, query tokens, raw HTTP bodies, and benign
    lookalikes; value-free report output; bounded scanning of large/rotated
    logs; current-producer regression tests; and a dry-run VPS inventory before
    any destructive remediation proposal.

## Merged Work Log

| Date | Item | PR / Commit | Result | Remaining |
|------|------|-------------|--------|-----------|
| 2026-06-30 | #1/#2/#3 Incident bundle generator, event query, and live restart/smoke automation | pending PR / `codex/v8-incident-window-discovery-summary` | Projects `time_window.files_scanned` and `time_window.file_discovery` into compact incident-bundle output so focused bundle scoping can be verified without opening the archive | Review, merge, and VPS5 no-restart smoke pending |
| 2026-06-30 | #1/#2/#3 Incident bundle generator, event query, and live restart/smoke automation | PR #919 / `989b81c9` | Applied incident-bundle query scope filters to `time_window_report.json`, `timeline.txt`, and matched event-segment selection; VPS5 no-restart deploy stayed hard-green and a focused OKX bundle verified only OKX time-window paths, exchange/user values, and segment selection | Surface time-window discovery metadata in compact output |
| 2026-06-30 | #1/#2/#3 Incident bundle generator, event query, and live restart/smoke automation | PR #918 / `946d0757` | Added incident-bundle CLI/library filters for level, exchange, user, bot id, remote-call group, side, source, component, tag, and data_eq; VPS5 no-restart deploy stayed hard-green and a focused OKX bundle verified scoped event/problem reports from the live monitor tree | Completed by PR #919 time-window/timeline/segment scoping |
| 2026-06-30 | #1/#2/#3 Incident bundle generator, event query, and live restart/smoke automation | PR #917 / `29d026a` | Embedded a bounded `problem_event_report.json` in incident bundles by default, using the same shared problem-event predicate as `live-smoke-report` and `live-event-query --problem-events`; VPS5 no-restart deploy stayed hard-green and a bounded bundle smoke verified `problem_event_report.json` in the archive | Continue cross-bot incident workflow improvements only where concrete operator diagnostics require them |
| 2026-06-30 | #2/#3 Event query and live restart/smoke automation | PR #916 / `0f32aeff` | Added `live-event-query --problem-events` and `--hard-problem-events` using the same shared predicate as `live-smoke-report`; VPS5 no-restart deploy stayed hard-green and a focused query matched the same EMA/HSL attention groups shown by brief smoke | Completed by PR #917 incident-bundle problem-event embedding |
| 2026-06-30 | #2/#3 Event query and live restart/smoke automation | PR #915 / `aef82af9` | Projected bounded, value-safe `problem_events.groups` and `event_types` into `live-smoke-report --brief`; VPS5 no-restart deploy stayed hard-green with all five bots matched and remaining attention attributable to EMA readiness plus HSL cooldown groups | Completed by PR #916 query filters and the pending incident-bundle embedding slice |
| 2026-06-30 | #1/#3 Incident bundle generator and live restart/smoke automation | PR #914 / `9ff335e4` | Moved incident-bundle event-segment SHA hashing behind actual segment inclusion; VPS5 no-restart deploy stayed green, bounded bundle smoke completed in 9.77s with zero event-segment bytes copied, and brief smoke stayed hard-green with all five bots matched | Brief smoke still reports `attention=true` from structured problem events without bounded top-cause groups; continue safe restart orchestration |
| 2026-06-30 | #1/#3 Incident bundle generator and live restart/smoke automation | PR #913 / `d3f3264c` | Added opt-in `--event-tail-lines` to incident bundles and shared seek-tail event-row iteration for plain NDJSON across event-query, smoke-report, and incident time-window scans; VPS5 no-restart deploy stayed green and bounded bundle smoke completed in 15.51s with seek-tail metadata | Avoid remaining disabled-segment manifest hashing; continue safe restart orchestration |
| 2026-06-30 | #1/#3 Incident bundle generator and live restart/smoke automation | PR #912 / `0eb29545` | Added `--recent-minutes` to incident bundles and deployed it read-only to VPS5; bundle smoke showed the recent window active with clean hard-failure status but still many matched current-segment events | Add opt-in event-tail bounding for recent incident bundles; continue safe restart orchestration |
| 2026-06-30 | #1/#3 Incident bundle generator and live restart/smoke automation | PR #911 / `7c5c96f4` | Projected bounded event-file discovery metadata into incident-bundle event-report summaries and event-segment manifests; VPS5 no-restart deploy stayed green and bundle smoke showed matching discovery counts without copying event segments | Continue safe restart orchestration; add small incident-bundle CLI ergonomics where they reduce operator mistakes |
| 2026-06-30 | #3/#4 Live restart/smoke automation and performance readiness | PR #910 / `ac949f03` | Projected bounded event-file discovery metadata into `live-performance-report` full and summary output; VPS5 no-restart deploy stayed green and concise performance smoke showed discovery counts without file paths | Continue safe restart orchestration; carry discovery visibility into incident bundles where it explains copied segment scope |
| 2026-06-30 | #3 Live restart/smoke automation | PR #909 / `c8c51d73` | Projected bounded event-file discovery metadata into `live-smoke-report` full, summary, and brief output; VPS5 no-restart deploy stayed green and brief smoke showed discovery counts without file paths | Continue safe restart orchestration; carry discovery visibility into other read-only report surfaces where it explains scan cost |
| 2026-06-30 | #2 Event query and timeline CLI extensions | PR #908 / `0c0024a3` | Added bounded `file_discovery` metadata to `live-event-query` reports while preserving the existing list-returning discovery API; VPS5 no-restart deploy stayed green and focused query smoke showed path-pruning counts for Binance | Carry discovery visibility into smoke reports where useful; continue query-pruning/index work only where concrete smoke cost appears |
| 2026-06-30 | #2 Event query and timeline CLI extensions | PR #907 / `f792f889` | Added conservative monitor path pruning for path-shaped `live-event-query --bot-id` filters; VPS5 no-restart deploy stayed green, path-shaped bot-id query scanned one file, and opaque bot-id query preserved full-scan behavior | Continue query-pruning/index work only where concrete smoke cost appears; expose pruning/discovery metadata so operators can see when scope pruning occurred |
| 2026-06-30 | #2 Event query and timeline CLI extensions | PR #906 / `b7b34758` | Added `live-event-query --source`, `--component`, and `--side` filters plus compact `source` output; folded #903/#904 deploy evidence into the same real observability PR; VPS5 no-restart deploy stayed green and focused single-bot query smokes validated the new filter echoes | Broad parallel root-level monitor scans were too slow for routine VPS smoke; prefer focused paths and continue query-pruning/index work where concrete smoke cost appears |
| 2026-06-30 | #0/#3/#10 HSL status smoke evidence | PR #903 / `1dd115cc` | Added `risk_events.hsl_status` to `live-smoke-report` full, summary, and brief output from existing `hsl.status` events; fixed #904 by filtering shareable summary risk `latest_data` through a value-safe whitelist; VPS5 no-restart smoke stayed green and showed red HSL status counts for ZEC without magnitude fields in brief output | Actual HSL startup latency optimization remains open; broader event-driven console redesign remains open |
| 2026-06-30 | #0/#3 HSL completed replay smoke evidence | PR #901 / `9b3c29ad` | Added `hsl_replay.max_completed_elapsed_ms` to `live-smoke-report --brief`; VPS5 no-restart smoke stayed green after deploy, with no HSL replay events in the sampled window | HSL startup latency remains a trading-path optimization; this slice only surfaces completed replay latency when completed replay events are in-window |
| 2026-06-30 | #0/#3 HSL replay/startup smoke evidence | PR #899 / `e1fcb038` | Added `live-smoke-report --brief` projection for worst active HSL replay elapsed time, latest-event age, and active stage counts from existing sanitized groups; VPS5 no-restart smoke stayed green after deploy | HSL startup latency remains a trading-path optimization; this slice only surfaces active replay latency in brief smoke |
| 2026-06-30 | Logging loop scope/progress tracking | PR #898 / `05c48b5` | Recorded the retuned logging-loop boundary: backlog work is in-loop only when it helps diagnostics, smoke evidence, incident reconstruction, or logging-overhaul validation | Continue keeping scope decisions and deploy evidence current as the loop proceeds |
| 2026-06-30 | #0/#3/#18 Progress and evidence tracking | PR #897 / `aebc3667` | Recorded exchange-config-refresh smoke projection evidence; VPS5 was then restarted so live processes loaded the producer/projection, with a green settled smoke and a wider real HSL ZEC cooldown window | Prove `exchange.config_refresh` during an hourly refresh; implement HSL startup latency and safe restart orchestration separately |
| 2026-06-25 | #1 Incident bundle generator | PR #641 / `e1f99002` | Added `passivbot tool live-incident-bundle`; bundle smoke on VPS5 created an archive with redacted monitor/config evidence | Supervisor/process status and tighter remote smoke integration |
| 2026-06-25 | #2 Event query and timeline CLI extensions | PR #638 / `1b15b2d5` | Added broader live event query filters | More ID scopes still needed at that point |
| 2026-06-25 | #2 Event query and timeline CLI extensions | PR #642 / `ad36d8ea` | Added bot/snapshot/plan/action/remote-call-group filters and shared ID-key timeline rendering; VPS5 query smoke passed | Richer reconstruction views |
| 2026-06-25 | #2/#11 Event query and order trace summaries | PR #648 / `774bcf74` | Added `live-event-query --trace-summary` aggregate counts across matched events, ID scopes, symbols, sides, and order waves | Full create/cancel/missing-order reconstruction view |
| 2026-06-25 | #2/#11 Event query and order trace completeness | PR #651 / `b9f42ebd` | Added `live-event-query --order-trace` reconstruction grouped by order wave and action, with confirmation events and bounded samples | Richer cycle reconstruction and incident-bundle integration |
| 2026-06-25 | #2/#11 Event query and cycle trace completeness | PR #654 / `ff493541` | Added `live-event-query --cycle-trace` reconstruction grouped by cycle id, with bounded timelines, aggregate summaries, and nested order traces | Incident-bundle integration and cross-bot workflow |
| 2026-06-25 | #1/#2/#11 Incident bundle trace integration | PR #659 / `27931c81` | Embedded trace-summary and order-trace reports into incident bundles by default, plus cycle-trace when scoped to `--cycle-id`; VPS5 bundle smoke verified trace sections | Cross-bot incident workflow and supervisor/process context |
| 2026-06-25 | #1/#3/#14 Smoke process status | PR #661 / `72b3d931` | Added optional read-only process liveness to `live-smoke-report` and incident bundles; VPS5 smoke matched all five `/root/bots_vps5.yaml` bots | Safe restart orchestration and richer supervisor model remain open |
| 2026-06-25 | #6 Exchange health and contract probes | PR #663 / `45b0cf9e` | Added passive `remote_call.failed` summaries to `live-smoke-report`; VPS5 smoke grouped Kucoin timeouts by balance/positions/open_orders | Active read-only exchange endpoint probes remain open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #665 / `37c29359` | Added structured-event time windows to `live-smoke-report` and threaded them into incident-bundle smoke reports; VPS5 smoke used the window to separate pre-deploy failures from settled post-restart behavior | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #11 Order lifecycle trace completeness | PR #666 / `fa90623d` | Added structured create-filter/defer events for pre-exchange create-order gates, best-effort and off default console | Continue producer coverage as nearby execution surfaces are touched |
| 2026-06-26 | #2 Event query and timeline CLI extensions | PR #667 / `e5771cfa` | Added event time-window filters to `live-event-query`; query, timeline, trace-summary, order-trace, and cycle-trace views use the same scoped event set | Cross-bot incident workflow |
| 2026-06-26 | #13 Cache integrity doctor | PR #668 / `734c2de0` | Added cache-family summaries and issue family tags to the read-only cache doctor | Coverage windows, suspicious gaps, metadata compatibility, and warm-cache readiness |
| 2026-06-26 | #3 Live restart/smoke automation | PR #670 / `b74d12be` | Extended `live-smoke-report` time windows to parseable timestamped text log lines; VPS5 smoke proved stale log lines were skipped via `logs.window.lines_skipped_before` | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #671 / `34f63799` | Narrowed traceback log matching to real Python traceback headers; VPS5 smoke with logs enabled returned `ok=true`, `logs.hard_matches=0`, and all five bots matched | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3/#8 Live restart/smoke automation and HSL preview | PR #673 / `2697ff48` | Added bounded `risk_events` summaries to `live-smoke-report`; VPS5 smoke exposed GateIO ZEC long HSL RED cooldown without changing smoke health policy | Safe pull/stop/start orchestration and true HSL dry-run preview still open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #675 / `3aa1e7a7` | Added explicit `keep|drop` policy for unparseable text-log lines in smoke windows; opt-in `drop` suppresses only non-signal unparseable noise and preserves traceback/hard signals; VPS5 smoke with `drop` passed | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #10 Operator console redesign from events | PR #677 / `409f5d8e` | Mirrored existing execution-loop error burst warnings into structured `health.summary` events with `execution_loop_error_burst`, best-effort and redacted | Continue migrating high-value text logs to structured events without increasing console noise |
| 2026-06-26 | #3 Live restart/smoke automation | PR #679 / `60c9a41` | Added bounded `problem_events.latest_data` for selected smoke problem groups and timestamp-context filtering for stale unparseable log continuations; VPS5 smoke after deploy returned `ok=true`, no hard problem events, no log hard/attention matches, and all five bots matched | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #682 / `048e8595c` | Added top-level `problem_event_count` and bounded `problem_event_groups` aggregates to `live-smoke-report`; settled VPS5 smoke after deploy returned `ok=true`, no hard problem events, no log matches, no remote-call failures, and all five bots matched | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #684 / `04ca7174` | Dropped contextless unparseable log lines under explicit `--log-window-unparsed-policy drop`, preventing stale mid-traceback tails from causing false hard smoke failures; VPS5 smoke after deploy returned `ok=true`, no hard problem events, no log hard/attention matches, no remote-call failures, and all five bots matched | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3/#6 Live restart/smoke automation and exchange probes | PR #686 / `b03f4139` | Added branch/head/tracked-dirty repository metadata to `live-smoke-report`; VPS5 smoke verified `/root/passivbot` on `v8` head `9e898019`, dirty=false, all five bots matched | Kucoin authoritative endpoint timeout probes and safe pull/stop/start orchestration remain open |
| 2026-06-26 | #3/#6 Live restart/smoke automation and exchange probes | PR #688 / `9945a3d3` | Added `remote_call_timings` elapsed-time groups to `live-smoke-report`; VPS5 smoke on `11f7d142` returned `ok=true`, no hard failures, all five bots matched, and surfaced slow-but-successful candle fetch groups | Explicit exchange endpoint probes and safe pull/stop/start orchestration remain open |
| 2026-06-26 | #3/#6 Live restart/smoke automation and exchange probes | PR #690 / `dc99378a` | Added `remote_call_health` groups to `live-smoke-report`, rolling up terminal remote-call successes/failures/throttles, latency, reason/error counts, and affected symbols by bot/component/kind/surface; VPS5 smoke on `b150176f` returned `ok=true`, no hard failures, all five bots matched, and summarized 445 terminal remote calls | Explicit exchange endpoint probes and safe pull/stop/start orchestration remain open |
| 2026-06-26 | #3/#6 Live restart/smoke automation and exchange probes | PR #692 / `ac4afe3f` | Added top-level `remote_call_health` success/failure/throttle totals and percentages to `live-smoke-report`; VPS5 smoke on `c8ce4880` returned `ok=true`, no hard failures, all five bots matched, and reported total=390, succeeded=389, failed=1, throttled=0 | Explicit exchange endpoint probes and safe pull/stop/start orchestration remain open |
| 2026-06-26 | #3/#6 Live restart/smoke automation and exchange probes | PR #694 / `bebbb3f6` | Added `account_critical_remote_call_health` to `live-smoke-report`, isolating authoritative balance/positions/open-orders style endpoint health from candle/fill traffic; VPS5 smoke on `3299c1ca` returned `ok=true`, no hard failures, all five bots matched, and reported account-critical total=126, succeeded=126, failed=0, throttled=0 | Explicit exchange endpoint probes and safe pull/stop/start orchestration remain open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #696 / `f1efbe45` | Added `live-smoke-report --summary`, a concise high-signal projection of smoke health, process/repository state, problem groups, remote-call/account-critical health, and risk events; VPS5 compact summary smoke on `d850daf5` returned `ok=true`, no hard failures, all five bots matched, and account-critical total=58, succeeded=58 | Safe pull/stop/start orchestration remains open; optional row-limit/brief summary mode could reduce chat-facing output further |
| 2026-06-26 | #3 Live restart/smoke automation | PR #698 / `7c7368f3` | Redacted common user/home prefixes from smoke-report `repository.root` while preserving real git cwd use; incident-bundle `smoke_report.json` inherits the safer display field | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #699 / `4e2fcee7` | Surfaced dropped contextless unparsed attention/hard counters under `--log-window-unparsed-policy drop`; dropped attention now makes smoke `attention=true` without making stale tail fragments hard failures; VPS5 settled smoke on `d5639813` returned `ok=true`, no hard failures, all five bots matched | Safe pull/stop/start orchestration still open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #709 / `71479c61` deploy evidence | VPS5 restart smoke after the fill-cache event slice returned `ok=true`, no hard failures, all five bots matched; the restart exposed four orphaned live processes after two Ctrl+C rounds, cleared by SIGTERM before reload | Safe pull/stop/start orchestration should include shutdown timing, orphan detection, and escalation policy |
| 2026-06-26 | #3/#14 Supervisor/process diagnostics | PR #712 / `51ba92a3` | Extended `live-smoke-report --supervisor-config` to classify expected matches, missing expected commands, duplicate configured-command process matches, and extra/orphan-like `passivbot live` processes with bounded per-process metadata; VPS5 smoke showed all five configured bots matched with zero duplicate or extra live process matches; documented the read-only command-match limitation and shutdown escalation ladder as policy only | Safe pull/stop/start orchestration remains open |
| 2026-06-26 | #12 Debug profile toggles | pending PR | Added opt-in live-event debug profiles via config/env and initial Rust orchestrator structured-event enrichment with bounded input-symbol and output-order samples | Add remote-call, candle/EMA, HSL, fills, and execution profiles as needed |
| 2026-06-27 | #12 Debug profile toggles | PR #728 / `5714d36d` | Added opt-in `fills` debug-profile enrichment to existing fill refresh and ingestion events with bounded count, coverage, and key-shape metadata | HSL and execution followed in #730/#732 |
| 2026-06-27 | #12 Debug profile toggles | PR #730 / `1334982c` | Added opt-in `hsl` debug-profile enrichment to existing HSL event surfaces with bounded event key, metric key, and latch/cooldown state-shape metadata | Execution followed in #732 |
| 2026-06-27 | #12 Debug profile toggles | PR #732 / `9bc2c37f` | Added opt-in `execution` debug-profile enrichment to existing order-wave, order-write, create-filter, and confirmation events with bounded key-shape/counter metadata | Initial targeted profile set complete; add future profiles only as diagnostics require |
| 2026-06-26 | #7 Live config preflight/linter | PR #714 / `564dc0a8` | Added `passivbot tool live-config-preflight`, a local-only JSON report for one config's risk-relevant live facts with bounded coin samples and malformed-structure errors; VPS5 preflight smoke returned `ok=true` with one expected missing short-side warning | Config diffing, deeper cache compatibility checks, and live startup enforcement remain open |
| 2026-06-26 | #3 Live restart/smoke automation | PR #715 / `7b12d4b2` | Added passive `shutdown_events` summaries to full, summary, and brief `live-smoke-report` output for existing bot stopping/stage/stopped events; VPS5 no-restart smoke showed `shutdown_events.total=0`, all five bots matched, and no hard failures | Safe pull/stop/start orchestration remains open |
| 2026-06-27 | #13 Cache integrity doctor | PR #722 / `3b7e6306` | Added read-only v2 candle coverage windows and suspicious interior gap samples from local `.valid.npy` artifacts | Fill/HSL coverage/readiness and deeper metadata compatibility |
| 2026-06-27 | #13 Cache integrity doctor | PR #725 / `1ed9c466` | Added read-only fill/HSL metadata summaries from local JSON/NDJSON artifacts | Deeper metadata compatibility, synthetic/no-trade assumptions, and warm-cache readiness |
| 2026-06-27 | #13 Cache integrity doctor | PR #727 / `d8dd7246` | Added report-only warm-cache readiness evidence from already-scanned candle/fill/HSL cache metadata | Deeper metadata compatibility and synthetic/no-trade assumptions |
| 2026-06-27 | #7 Live config preflight/linter | PR #731 / `3cc2d229` | Added optional read-only `--compare` diff reporting for local two-config preflights, including HSL signal/enabled changes, universe deltas, forager slots/staleness, identity hints, and cache live settings | Deeper cache compatibility checks and live startup enforcement remain open |
| 2026-06-27 | #4 Startup phase budget tracking | PR #735 / `6f415777` | Added report-only startup budget projections to `live-smoke-report` phase summaries using prior local p95 baselines from existing monitor events; VPS5 deploy kept all five bots running and the settled smoke returned `ok=true` after unrelated transient HSL/EMA events aged out | Explicit durable budget config/events |
| 2026-06-26 | #6 Exchange health and contract probes | PR #701 / `fcda70f5` | Added `ticker-endpoint-probe` `account_critical_health` summaries for read-only balance/positions/open-orders outcomes; VPS5 Binance probe validated the summary and exposed an open-orders shape follow-up | Lower-impact/account-only mode, exchange-aware open-orders probing, clock skew/rate-limit/fill/candle probes |
| 2026-06-26 | #6 Exchange health and contract probes | PR #703 / `8fefce4b` | Added `ticker-endpoint-probe --account-only`, `--skip-my-trades`, and open-orders symbol fallback; VPS5 Binance account-only probe returned account-critical total=3, succeeded=3, and smoke stayed green | Clock skew/rate-limit/fill-pagination/candle-freshness probes |
| 2026-06-27 | #6 Exchange health and contract probes | PR #741 / `d4c28058` | Added `ticker-endpoint-probe` read-only `fetch_time` clock-skew evidence and collection-level `time_sync_health`, with unsupported exchanges separated from failures and `--skip-time-sync` as an operator escape hatch | Rate-limit/fill-pagination/candle-freshness probes |
| 2026-06-27 | #6 Exchange health and contract probes | PR #743 / `1fe1292b` | Added `ticker-endpoint-probe` `candle_freshness_health` summaries from existing 1m OHLCV tail results, including worst-symbol age and current-incomplete counts without extra exchange calls | Rate-limit/fill-pagination probes |
| 2026-06-27 | #6 Exchange health and contract probes | PR #745 / `4130155e` | Added `ticker-endpoint-probe` `fill_history_health` summaries from the existing first-symbol `fetch_my_trades` sample, including success/failure, latency, trade count, newest timestamp, and shape without raw trade/order ids or extra pagination calls; VPS5 authenticated Binance probe validated total=1, succeeded=1, failed=0 | Rate-limit/full fill-pagination coverage probes |
| 2026-06-27 | #6 Exchange health and contract probes | PR #747 / `74270454` | Added `ticker-endpoint-probe` `rate_limit_health` request-pressure estimates from existing probe outcomes and CCXT rate-limit metadata, without adding exchange calls or enforcing throttles; VPS5 authenticated Binance probe validated observed_call_count=12, private=5, public=6, concurrent=1 | Full fill-pagination coverage probes |
| 2026-06-27 | #6 Exchange health and contract probes | PR #749 / `16c25149` | Added opt-in bounded first-symbol `fetch_my_trades` pagination sampling to `ticker-endpoint-probe`, keeping default one-call behavior; VPS5 authenticated Binance probe validated pages=2/limit=2 request shape, short-page stop, call_count=1, and rate-limit accounting | Basic endpoint latency and deeper exchange-specific coverage checks |
| 2026-06-27 | #6 Exchange health and contract probes | PR #751 / `4eef3572` | Added `ticker-endpoint-probe` `endpoint_latency_health` summaries from existing probe outcomes, including open-orders fallback attempts and fill-history pages; VPS5 Binance probe validated endpoint_count=11, total=12, slowest=load_markets, and expected Binance open-orders all-symbol warning classification | Deeper exchange-specific coverage checks |
| 2026-06-27 | #6 Exchange health and contract probes | PR #753 / `0f1afc49` | Added `ticker-endpoint-probe` `exchange_surface_health` notes from existing open-orders, time-sync, fill-history, and OHLCV-tail outcomes; VPS5 Binance probe validated open-orders symbol fallback and fill-history short-page notes | Further probe expansion should be driven by concrete exchange gaps |
| 2026-06-27 | #3 Live restart/smoke automation | PR #755 / `5d9f3a5f` | Added `live-smoke-report` EMA readiness health summaries from existing `ema.unavailable` events; settled VPS5 smoke reported all five bots matched, no hard failures, no failed remote/account-critical calls, and `ema_readiness.total=11`, `bots=4`, `latest_candidate_unavailable_total=31`, `latest_unavailable_total=112` | Safe pull/stop/start orchestration still open; staged readiness diagnostics remain a useful next slice |
| 2026-06-27 | Adjacent strategy/runtime | PR #756 / `19b34138` | Added fixed trailing-martingale `ema_gate_mode` values and an unstuck EMA-gating toggle; VPS5 required an explicit Rust extension rebuild before restart, then immediate and settled smokes reported all five bots matched, no hard failures, no failed remote/account-critical calls, and only non-hard EMA readiness attention | Deployment tooling should make Rust rebuild/stamp verification explicit before live restarts |
| 2026-06-27 | #3 Live restart/smoke automation | PR #759 / `74a52ede` | Added `live-smoke-report` staged-readiness health summaries from existing staged `cycle.degraded` events; VPS5 smoke reported all five bots matched, no hard failures, no failed remote/account-critical calls, and `staged_readiness.total=4`, `bots=1`, `latest_missing_surface_total=1`, `latest_invalid_surface_total=1` | Use the new staged readiness signal to decide whether a narrow completed-candle readiness fix is warranted |
| 2026-06-27 | #3 Live restart/smoke automation | PR #760 / `31d42ea3` | Recorded staged-readiness deploy evidence; VPS5 pull required no restart, an initial smoke surfaced real HSL ZEC RED finalizations through logs and `risk_events`, and settled follow-up smokes returned `ok=true`, no hard/log failures, all five bots matched, no failed remote/account-critical calls, and persistent `staged_readiness` across up to four bots | Continue using `staged_readiness` summaries to decide whether completed-candle target-change fixes or diagnostics are warranted |
| 2026-06-27 | Adjacent staged readiness/runtime | PR #762 / `d9188b64` | Tolerated completed-candle fallback-to-normal signature shape recovery when symbol and target timestamp are unchanged; VPS5 restart smoke returned `ok=true`, all five bots matched, no hard/log failures, no failed remote/account-critical calls, and `staged_readiness.total=0` in both immediate and settled windows | Continue monitoring staged readiness; if it recurs, classify account-surface delays separately from completed-candle readiness shapes |
| 2026-06-27 | #13 Cache integrity doctor | PR #764 / `7797d038` | Added report-only metadata compatibility evidence for candle known-gap no-trade reasons, fill current-contract coverage proof, and HSL artifact/timestamp compatibility; final review follow-up made mixed no-trade/unclassified gaps explicitly partial and still unproven | Further cache-doctor refinements should stay read-only and prove rather than assume coverage |
| 2026-06-27 | #5 Resource pressure telemetry | PR #765 / `5275ab75` | Added `live-smoke-report` event-pipeline health summaries from existing `health.summary` queue/drop/sink-error counters; VPS5 30-minute smoke showed `event_pipeline.total=1`, no drops, no sink errors, and worker alive | Consider thresholded console/report warnings only after more live evidence |
| 2026-06-27 | #10 Operator console redesign from events | PR #767 / `b07d5166` | Added `live-smoke-report` risk/HSL log-match counters and bounded `category=risk|general` match labels; VPS5 smoke after deploy was green with zero hard/risk/non-risk log matches and all five bots running | Use the split counters to collect evidence before changing smoke verdict policy for real HSL RED/cooldown episodes |
| 2026-06-27 | #10 Operator console redesign from events | PR #769 / `b789e146` | Added `live-smoke-report` `hard_failure_sources` and `attention_sources` to full, summary, and brief output; VPS5 smoke after deploy was green with `hard_failure_sources.total=0`, all five bots running, no failed remote/account-critical calls, and all attention attributed to non-hard structured problem events | Use source breakdown plus risk/general log split before considering any smoke verdict policy change |
| 2026-06-25 | #3 Live restart/smoke automation | PR #639 / `86afd3b3` | Added read-only `passivbot tool live-smoke-report` | Safe pull/stop/start orchestration still open |
| 2026-06-25 | #4 Startup phase budget tracking | PR #649 / `7391d43b` | Added startup timing baselines to `live-smoke-report` from existing `bot.startup_timing` monitor events | Explicit durable budget config/events |
| 2026-06-25 | #5 Resource pressure telemetry | PR #643 / `09fd305b` | Added resource pressure and event-pipeline counters to `health.summary` | VPS5 restart/smoke pending; richer resource fields still open |
| 2026-06-25 | #9 Reason-code registry | PR #645 / `31263bb9` | Added shared `EventTags` and `ReasonCodes` registries and migrated representative live event emitters without changing emitted strings | Continue migrating stable literals as nearby event surfaces are touched |
| 2026-06-25 | #9 Reason-code registry | PR #653 / `f0a0f744` | Added focused AI docs for live event tags/reason codes and a doc drift test against the code registry | Continue migrating stable literals as nearby event surfaces are touched |
| 2026-06-25 | #10 Operator console redesign from events | PR #646 / `521832cc` | Improved event-projected console/text summaries for already-routed execution events without changing routes or console event volume | Migrate high-value stdlib text logs to structured-event projections |
| 2026-06-25 | #13 Cache integrity doctor | PR #656 / `e65597c3` | Added read-only `cache-integrity-doctor` for local cache root presence, file counts/sizes, and corrupt JSON/NDJSON/NPY artifacts | Cache-family metadata, coverage windows, suspicious gaps, and warm-cache readiness |
| 2026-06-24 | Operational restart goals | PR #619 / `e71c4f6c` | Improved shutdown progress and bounded shutdown cancel grace coverage | Broader interruptible shutdown contract remains separate work |
| 2026-06-24 | Operational restart goals | PR #622 / `29eba387` | Improved live startup warm-cache reuse | Deeper cache doctor and budget tracking remain open |
| 2026-06-30 | #3/#4 Live restart/smoke automation and startup budget tracking | PR #886 / `60c79c3a` | Exposed existing startup timing evidence in `live-smoke-report --summary` and `--brief`; VPS5 5-minute smoke stayed hard-green and showed the new `startup_timings` brief key | Safe pull/stop/start orchestration and durable startup budget config/events remain open |
| 2026-06-30 | #3 Live restart/smoke automation | PR #888 / `4b435d33` | Exposed bounded text-log window counters in `live-smoke-report --brief`; VPS5 5-minute smoke stayed hard-green and showed `logs.window.lines_skipped_before=1730` | Safe pull/stop/start orchestration remains open |
| 2026-06-30 | #3 Live restart/smoke automation | PR #890 / `1498abc9` | Exposed `event_window.enabled` in `live-smoke-report --brief`; VPS5 5-minute smoke stayed hard-green and showed `event_window.enabled=true` | Safe pull/stop/start orchestration remains open |
| 2026-06-30 | #2 Event query and timeline CLI extensions | PR #892 / `7e7ce16f` | Added `live-event-query --level` filtering for structured event severity; VPS5 query smoke matched 33 warning-level events with `ok=true` and all five bots still running | Cross-bot incident workflow remains open; Binance config-refresh traceback classification added as item #18 |
| 2026-06-30 | #18 Binance hourly hedge-mode/config refresh classification | PR #894 / `796ceb38` | Added off-console/text `exchange.config_refresh` events for hourly maintenance refresh success/failure with sanitized bounded error fields and fail-loud behavior preserved | Live emission evidence after next bot restart; smoke/report classification remains open |
| 2026-06-30 | #18 Binance hourly hedge-mode/config refresh classification | PR #896 / `53b8accb` | Added `live-smoke-report` full/summary/brief health projections for `exchange.config_refresh`, excluding raw error text; VPS5 no-restart smoke stayed hard-green and showed the new brief section | Live emission evidence after next bot restart; smoke/report classification remains open |

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
