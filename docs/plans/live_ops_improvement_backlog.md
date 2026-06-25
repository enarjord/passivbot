# Live Operations Improvement Backlog

## Purpose

This backlog captures high-value improvement areas noticed while running the
reviewed PR loop, repeatedly restarting VPS5 bots, and using the new live event
pipeline to diagnose behavior. It is intentionally higher level than the active
logging-overhaul plan. Each item below should become its own small reviewed
slice before implementation.

The logging overhaul remains the foundation: a centralized event stream should
make the items below easier to prove, test, and operate.

## High-Value Follow-Ups

1. Incident bundle generator.
   Add a local tool that collects one time window or `cycle_id` into a compact
   bundle: structured events, selected text log excerpts, monitor snapshots,
   bot config hash, git head, process uptime, resource samples, and relevant
   rotated event segments. The goal is to make "send me the evidence" a single
   command instead of a manual SSH/log-grep session.

2. Event query and timeline CLI.
   Add a first-class CLI for querying the event stream by `cycle_id`,
   `order_wave_id`, `remote_call_id`, symbol, pside, reason code, and status.
   It should print a terse timeline and optionally emit JSON. This is the
   natural companion to structured events; without it, operators still need to
   know file locations and NDJSON details.

3. Live restart/smoke automation.
   Formalize the repeated VPS smoke routine: pull a branch, stop configured
   bots, measure shutdown time per bot, reload from `/root/bots_vps5.yaml`,
   wait, then summarize process liveness, git head, recent hard errors, monitor
   event counts, startup timings, and resource usage. This should be safe,
   explicit, and produce a reviewable smoke report.

4. Startup phase budget tracking.
   Startup currently has timing events, but the next step is durable budget
   accounting by phase: account-critical fetches, fill/PnL refresh, active
   candle readiness, forager warmup, HSL replay, Rust planning, and READY.
   Store both current timings and rolling baselines so regressions stand out
   after short-downtime restarts.

5. Resource pressure telemetry.
   VPS5 repeatedly showed tight memory, swap usage, and high load during
   restarts. Add low-frequency process/system resource events: RSS, open file
   count, event queue depth, event drops, exchange-call counts, candle-fetch
   concurrency, and loop lag. Keep this off console unless thresholds are
   crossed.

6. Exchange health and contract probes.
   Add explicit read-only probes for each configured exchange/account before or
   during smoke: balance/positions/open-orders fetch, clock skew, rate-limit
   behavior, fill pagination coverage, candle freshness, and basic endpoint
   latency. These should detect exchange/API drift before it appears as an
   opaque live failure.

7. Live config preflight/linter.
   Add a preflight that explains risk-relevant config changes before startup:
   HSL enabled/disabled changes, HSL signal-mode changes, new approved/ignored
   universe size, forager staleness policy, max slots, exchange/user mismatch,
   and cache compatibility. The output should be structured and should not make
   trading decisions.

8. HSL dry-run preview for startup.
   Add a non-trading preview that reconstructs current HSL state and reports
   which symbols are green/yellow/red, cooldown status, current drawdown to red,
   and whether startup would emit panic orders. This would make risky restarts
   with changed HSL configs easier to reason about before live execution.

9. Reason-code registry.
   Centralize reason codes and event tags enough to prevent drift. The stream is
   much easier to search when `stale_ema`, `missing_canonical_candles`,
   `exchange_time_resync`, and similar codes are stable, documented, and tested.

10. Operator console redesign from events.
    Continue moving console output to be a projection of structured events.
    Default console should focus on fills, positions, balance, order writes,
    meaningful risk/HSL/unstuck transitions, and compact "waiting because"
    summaries. EMA/candle/cache internals should stay structured DEBUG unless
    they directly explain a blocked trading action.

11. Order lifecycle trace completeness.
    Keep tightening the end-to-end chain from Rust ideal order to executable
    order, gate decision, exchange payload, exchange response, local open-order
    refresh, confirmation, and fill. The target is that any create/cancel/missing
    order can be reconstructed from one id without reading code.

12. Debug profile toggles.
    Add narrow runtime/debug profiles that increase event detail for one domain:
    candles, fills, HSL, Rust payloads, order execution, or exchange calls. This
    avoids code patches or globally noisy DEBUG logs when diagnosing a live
    issue.

13. Cache integrity doctor.
    Add a read-only doctor for candle/fill/HSL caches that reports coverage,
    metadata compatibility, corrupted shards, suspicious gaps, synthetic/no-trade
    assumptions, and whether a short restart can safely use a warm-cache path.
    This supports the separate warm-cache restart work.

14. Supervisor/process model.
    The tmux/tmuxp setup is workable, but repeated live smoke showed room for a
    stricter supervisor contract: clear per-bot status, bounded stop/restart,
    captured exit reason, backoff policy, and health heartbeat. This could stay
    outside the trading core but should consume the same event stream.

15. Fake-live regression scenarios.
    Build more fake-exchange/fake-live scenarios for failures repeatedly seen in
    real work: stale candles, missing EMA inputs, fill pagination gaps, timestamp
    resync, queue overflow, slow shutdown, and exchange-call ambiguity. These
    should prove observability behavior without risking live accounts.

## Suggested Priority

Near-term highest leverage:

1. Event query/timeline CLI.
2. Live restart/smoke automation.
3. Startup phase budget tracking.
4. Resource pressure telemetry.
5. HSL dry-run preview.

These make every later live debugging session cheaper and provide direct
feedback on whether the event stream is actually answering operator questions.
