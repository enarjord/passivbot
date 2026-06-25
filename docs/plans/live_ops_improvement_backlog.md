# Live Operations Improvement Backlog

## Purpose

This backlog captures high-value improvement areas noticed while running the
reviewed PR loop, repeatedly restarting VPS5 bots, and using the new live event
pipeline to diagnose behavior. It is intentionally higher level than the active
logging-overhaul plan. Each item below should become its own small reviewed
slice before implementation.

The logging overhaul remains the foundation: a centralized event stream should
make the items below easier to prove, test, and operate.

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
- `docs/plans/live_restart_shutdown_and_warm_cache_handoff.md`

## High-Value Follow-Ups

1. [x] Incident bundle generator.
   Status: initial implementation merged. `passivbot tool live-incident-bundle`
   collects local monitor event reports, smoke summaries, redacted log excerpts,
   monitor snapshots, config hashes, runtime metadata, and bounded event
   segments into a tarball.

   Remaining refinements: add supervisor/process status and richer remote smoke
   integration when the restart/smoke automation exists.

2. [x] Event query and timeline CLI extensions.
   Status: core filter/timeline work merged. `passivbot tool live-event-query`
   now supports event discovery, compact JSON output, current-vs-rotated segment
   selection, terse timeline rendering, and filters for event type/kind, cycle
   id, order wave id, remote call id/group id, bot id, snapshot id, plan id,
   action id, symbol, pside, reason code, and status.

   Remaining refinements: richer cycle/order reconstruction and incident-bundle
   integration.

3. [ ] Live restart/smoke automation.
   Status: partial. The read-only `live-smoke-report` tool exists, but the safe
   restart orchestration contract is not implemented.

   Formalize the repeated VPS smoke routine: pull a branch, stop configured
   bots, measure shutdown time per bot, reload from `/root/bots_vps5.yaml`,
   wait, then summarize process liveness, git head, recent hard errors, monitor
   event counts, startup timings, and resource usage. This should be safe,
   explicit, and produce a reviewable smoke report.

4. [ ] Startup phase budget tracking.
   Status: partial. Startup timing and warmup cache decision events exist, but
   durable phase budgets and rolling baselines do not.

   Startup currently has timing events, but the next step is durable budget
   accounting by phase: account-critical fetches, fill/PnL refresh, active
   candle readiness, forager warmup, HSL replay, Rust planning, and READY.
   Store both current timings and rolling baselines so regressions stand out
   after short-downtime restarts.

5. [x] Resource pressure telemetry.
   Status: initial implementation merged. `health.summary` events now include
   process RSS, memory percent when available, open file descriptor count,
   system load averages, CPU count, and live-event pipeline queue/drop/sink
   error counters.

   Remaining refinements: exchange-call counts, candle-fetch concurrency, loop
   lag, thresholded console warnings, and richer system memory/swap fields. Keep
   this off console unless thresholds are crossed.

6. [ ] Exchange health and contract probes.
   Status: open.

   Add explicit read-only probes for each configured exchange/account before or
   during smoke: balance/positions/open-orders fetch, clock skew, rate-limit
   behavior, fill pagination coverage, candle freshness, and basic endpoint
   latency. These should detect exchange/API drift before it appears as an
   opaque live failure.

7. [ ] Live config preflight/linter.
   Status: open.

   Add a preflight that explains risk-relevant config changes before startup:
   HSL enabled/disabled changes, HSL signal-mode changes, new approved/ignored
   universe size, forager staleness policy, max slots, exchange/user mismatch,
   and cache compatibility. The output should be structured and should not make
   trading decisions.

8. [ ] HSL dry-run preview for startup.
   Status: open.

   Add a non-trading preview that reconstructs current HSL state and reports
   which symbols are green/yellow/red, cooldown status, current drawdown to red,
   and whether startup would emit panic orders. This would make risky restarts
   with changed HSL configs easier to reason about before live execution.

9. [ ] Reason-code registry.
   Status: initial registry slice in progress in PR #645.

   Centralize reason codes and event tags enough to prevent drift. The stream is
   much easier to search when `stale_ema`, `missing_canonical_candles`,
   `exchange_time_resync`, and similar codes are stable, documented, and tested.
   This pairs directly with reason-code filtering in the event query tool.

   Work log:
   - 2026-06-25: Added shared `EventTags` and `ReasonCodes` registries for
     common live-event tags/reason codes, migrated representative emitters
     without changing emitted strings, and added registry contract tests.

   Remaining refinements: migrate additional stable literals as nearby event
   surfaces are touched.

10. [ ] Operator console redesign from events.
    Status: open.

    Continue moving console output to be a projection of structured events.
    Default console should focus on fills, positions, balance, order writes,
    meaningful risk/HSL/unstuck transitions, and compact "waiting because"
    summaries. EMA/candle/cache internals should stay structured DEBUG unless
    they directly explain a blocked trading action.

11. [ ] Order lifecycle trace completeness.
    Status: partial. The order-wave/execution event chain exists, but there is
    no single reconstruction view for every create/cancel/missing-order case.

    Keep tightening the end-to-end chain from Rust ideal order to executable
    order, gate decision, exchange payload, exchange response, local open-order
    refresh, confirmation, and fill. The target is that any create/cancel/missing
    order can be reconstructed from one id without reading code.

12. [ ] Debug profile toggles.
    Status: open.

    Add narrow runtime/debug profiles that increase event detail for one domain:
    candles, fills, HSL, Rust payloads, order execution, or exchange calls. This
    avoids code patches or globally noisy DEBUG logs when diagnosing a live
    issue.

13. [ ] Cache integrity doctor.
    Status: open.

    Add a read-only doctor for candle/fill/HSL caches that reports coverage,
    metadata compatibility, corrupted shards, suspicious gaps, synthetic/no-trade
    assumptions, and whether a short restart can safely use a warm-cache path.
    This supports the separate warm-cache restart work.

14. [ ] Supervisor/process model.
    Status: open.

    The tmux/tmuxp setup is workable, but repeated live smoke showed room for a
    stricter supervisor contract: clear per-bot status, bounded stop/restart,
    captured exit reason, backoff policy, and health heartbeat. This could stay
    outside the trading core but should consume the same event stream.

15. [ ] Fake-live regression scenarios.
    Status: open.

    Build more fake-exchange/fake-live scenarios for failures repeatedly seen in
    real work: stale candles, missing EMA inputs, fill pagination gaps, timestamp
    resync, queue overflow, slow shutdown, and exchange-call ambiguity. These
    should prove observability behavior without risking live accounts.

## Merged Work Log

| Date | Item | PR / Commit | Result | Remaining |
|------|------|-------------|--------|-----------|
| 2026-06-25 | #1 Incident bundle generator | PR #641 / `e1f99002` | Added `passivbot tool live-incident-bundle`; bundle smoke on VPS5 created an archive with redacted monitor/config evidence | Supervisor/process status and tighter remote smoke integration |
| 2026-06-25 | #2 Event query and timeline CLI extensions | PR #638 / `1b15b2d5` | Added broader live event query filters | More ID scopes still needed at that point |
| 2026-06-25 | #2 Event query and timeline CLI extensions | PR #642 / `ad36d8ea` | Added bot/snapshot/plan/action/remote-call-group filters and shared ID-key timeline rendering; VPS5 query smoke passed | Richer reconstruction views |
| 2026-06-25 | #3 Live restart/smoke automation | PR #639 / `86afd3b3` | Added read-only `passivbot tool live-smoke-report` | Safe pull/stop/start orchestration still open |
| 2026-06-25 | #5 Resource pressure telemetry | PR #643 / `09fd305b` | Added resource pressure and event-pipeline counters to `health.summary` | VPS5 restart/smoke pending; richer resource fields still open |
| 2026-06-24 | Operational restart goals | PR #619 / `e71c4f6c` | Improved shutdown progress and bounded shutdown cancel grace coverage | Broader interruptible shutdown contract remains separate work |
| 2026-06-24 | Operational restart goals | PR #622 / `29eba387` | Improved live startup warm-cache reuse | Deeper cache doctor and budget tracking remain open |

## Suggested Priority

Near-term highest leverage:

1. Event query/timeline CLI.
2. Live restart/smoke automation.
3. Startup phase budget tracking.
4. Resource pressure telemetry.
5. HSL dry-run preview.

These make every later live debugging session cheaper and provide direct
feedback on whether the event stream is actually answering operator questions.
