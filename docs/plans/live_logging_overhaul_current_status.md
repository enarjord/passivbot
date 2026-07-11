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
  `Count protective readiness from completion evidence`
- Branch: `codex/v8-hsl-replay-scorecard-completion-fallback`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `77e111f7277077b4d1c4c37a562a67dad8906665`
- Scope: when a dedicated `held_protective_ready` event has rotated out of the
  selected files, `live-performance-report` derives the bounded protective
  elapsed aggregate from the retained completed replay record. It does not
  synthesize a milestone record.
- Triggering evidence: PR #1181's first no-restart VPS5 smoke correctly showed
  all four compact full-replay completions, but reported
  `protective_ready_bot_count=0` because the earlier milestone events had
  rotated while every completion record still carried `protective_elapsed_s`.
- Non-goals: no live event producer, replay ordering or arithmetic, HSL
  threshold/episode/cooldown behavior, cache authority, exchange call, process
  control, Rust, backtest, or smoke-verdict change.
- Local validation: all `73` live-performance-report tests and `25`
  incident-bundle integration tests pass. Python compilation and diff hygiene
  pass; this follow-up does not touch imports.
- Independent preflight: Terra's initial completion/startup fallback concern
  was rejected at the pre-split compatibility boundary; after the strict
  explicit-protective regression and code comment, Terra retracted the finding
  and reported no blocker. Sol owns final adjudication.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts and run a bounded no-restart performance report
  against the existing monitor history. Running bots need no restart because
  this slice changes read-only report consumption only.

Next action:

1. Finish focused validation and independent preflight, publish the completion
   fallback, resolve verified findings, and merge only after the exact-head
   gate; then rerun the declared VPS5 no-restart report smoke.

## Deployed Baseline

- Remote `v8`: `77e111f7`, PR #1181
- VPS5 repository: `77e111f7`, PR #1181; tracked status clean
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
- PR #1181 required no restart; all five bot pane PIDs remained unchanged. Its
  bounded current-segment scorecard reported four compact full-replay
  completions from `453.98s` to `1728.585s`, but exposed the active slice's
  missing completion fallback for the rotated protective milestones.
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
deployed. The active slice completes the bounded operator scorecard when early
protective milestones rotate before the later full-replay completion.
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
