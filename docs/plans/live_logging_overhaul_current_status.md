# Live Logging Overhaul Current Status

Updated: 2026-07-11.

This is the compact operational source for the active logging-overhaul loop.
Read it before the historical progress ledger. Update it whenever the active
PR, head SHA, review gate, deployed SHA, VPS state, or next action changes.

## Goal

Deliver bounded, correlated, operator-useful live observability through small
reviewable PRs, current-head review gates, and evidence-based VPS5 validation.
Rust remains the trading-logic authority; observability must not become a
trading control plane.

Estimated completion:

- core event/observability architecture: about 85%
- original logging migration: about 70%
- expanded logging, performance-readiness, restart, and ops scope: about 65%
  overall, with substantial uncertainty from the intentionally growing backlog

## Active Review Slice

- PR and publication state: query live GitHub metadata;
  `Expose HSL replay readiness scorecard`
- Branch: `codex/v8-hsl-replay-scorecard`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `6e72f374dca9f3e2d77924e84f9453bbd7561386`
- Scope: `live-performance-report` retains each bot's explicit
  `held_protective_ready` record, whitelists replay `history_format` and
  `protective_elapsed_s`, derives stage elapsed milliseconds, and adds bounded
  history-format, protective-ready, and completed-full-replay aggregates.
- Triggering evidence: after PR #1180 deployed, all four coin-HSL bots used the
  compact path and reached protective readiness in `11.237s` to `79.883s`, but
  those values and the history format required manual event queries. Kucoin's
  first full compact replay completed in `453.98s`; three broader background
  replays remained active. The operator scorecard should expose these existing
  events directly.
- Non-goals: no live event producer, replay ordering or arithmetic, HSL
  threshold/episode/cooldown behavior, cache authority, exchange call, process
  control, Rust, backtest, or smoke-verdict change.
- Local validation: all `72` live-performance-report tests and `25`
  incident-bundle integration tests pass. Python compilation and diff hygiene
  pass. Whole-file import sorting remains nonconforming at the unchanged
  baseline; this slice does not touch imports.
- Independent preflight: Terra found and Sol fixed scan-order-dependent retained
  milestones by applying the existing full event-position ordering contract.
  Delta re-review found no remaining blocker; Sol owns final adjudication.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts and run a bounded no-restart performance report
  against the existing monitor history. Running bots need no restart because
  this slice changes read-only report consumption only.

Next action:

1. Finish focused validation and independent preflight, publish the scorecard
   slice, resolve verified findings, and merge only after the exact-head gate;
   then run the declared VPS5 no-restart report smoke.

## Deployed Baseline

- Remote `v8`: `6e72f374`, PR #1180
- VPS5 repository: `6e72f374`, PR #1180; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- Immediate and fresh settled smoke reports were green: all five expected bots
  matched, hard failures were zero, and the fresh window recorded `614/614`
  remote plus `90/90` account-critical calls succeeded. One Kucoin
  `RequestTimeout` cycle failure in the wider restart window recovered before
  the fresh smoke.
- All four coin-HSL bots emitted `history_format=compact` and protective-ready
  success. Kucoin completed full replay in `453.98s`; three background replays
  remained active with no required pair work pending.
- Compared with the pre-deploy sample, aggregate RSS fell from `694840 KB` to
  `555296 KB`, all five processes moved from four `D` states to five `R`
  states, and host swap use fell from `2926 MB` to `906 MB`. CPU remained high
  while background replay continued.
- Preserve local/VPS configs, logs, monitor data, reports, and temporary files

## Review Gate

- Normal gate: all reviewers currently designated by the maintainer plus green
  CI on the exact head SHA.
- Temporary gate while Claude is rate-limited: Hermes + Grok 4.5 + green CI.
- Findings from any additional reviewer must still be verified and resolved.
- Any pushed delta requires current-head re-review.

## Agent Routing

- Sol: architecture, high-risk implementation, finding adjudication, merge,
  VPS signals/restart, and incident judgment.
- Terra: isolated low/medium-risk docs, tests, report/query tooling, and bounded
  observability implementation with explicit file scope.
- Luna or deterministic automation: metadata polling, state-change detection,
  CI/reviewer summaries, and read-only output parsing.
- Parallel PRs must be orthogonal. Dependent work waits for merge to `v8`.

## Next Slice

The coin-HSL protective-readiness split, cooperative background cadence,
current process-pressure query, and compact cold replay payload are merged and
deployed. The active slice turns the resulting readiness and full-replay timing
events into a bounded operator scorecard without changing live behavior.
Remaining candidates:

- realistic-scale replay fixtures and deeper internal-stage profiling
- lower-complexity full replay after the scorecard exposes its baseline
- unsupported configured-market and stock-perp compatibility events
- bounded operator tooling improvements sharing one code and validation surface

Do not create progress-only PRs or resume unrelated logging work from stale
worktrees.

## References

- Operating workflow: `live_logging_overhaul_pr_loop_workflow.md`
- Architecture: `live_logging_overhaul_plan.md`
- Historical evidence: `live_logging_overhaul_progress.md`
- Performance goals: `live_performance_readiness_goals.md`
- Operational backlog: `live_ops_improvement_backlog.md`
- Reviewer loop: `../ai/pr_auto_review_loop.md`
