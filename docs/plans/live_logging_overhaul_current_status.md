# Live Logging Overhaul Current Status

Updated: 2026-07-10.

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

- PR: query live GitHub metadata;
  `Expose current live-process pressure in smoke reports`
- Branch: `codex/v8-smoke-current-process-pressure`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `d83797b700091100fd357bcb48f90be98d0b97e4`
- Scope: aggregate already-parsed local `ps` fields under the existing
  `processes` section: state counts, uninterruptible-sleep count, and bounded
  CPU, memory, and RSS totals/maxima/reporting counts. Project the same fields
  through full, summary, and brief smoke output.
- Triggering evidence: after PR #1178, immediate and settled smokes were hard
  green with live I/O succeeding, but event-derived `resource_pressure`
  reported only Hyperliquid because replay-heavy bots could not emit timely
  `health.summary` samples. Direct process probes simultaneously showed four
  coin-HSL bots in `D` state with high RSS and swap/I/O pressure.
- Non-goals: no event producer, monitor write, process signal, verdict/attention
  threshold, exchange call, restart behavior, HSL/risk/order logic, Rust, or
  backtest change.
- Local validation: focused process parsing/projection tests plus full
  smoke-report (77), restart-smoke-plan (24), and incident-bundle (25) suites
  pass; syntax and diff checks pass. Missing RSS remains null with a zero
  reporting count, and non-finite CPU or memory samples are excluded.
- Independent preflight: one read-only audit confirmed the existing `ps`
  parser, full process report, summary/brief whitelists, compatibility fallbacks,
  and targeted test surfaces. No delegated edits were used.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull while preserving local artifacts and run a bounded
  read-only process-section smoke. No bot restart or Rust rebuild is required.

Next action:

1. Publish the read-only tooling slice, resolve verified findings, and merge
   only after the exact-head gate; then perform the declared VPS5 query smoke.

## Deployed Baseline

- Remote `v8`: `d83797b7`, PR #1178
- VPS5 repository: `d83797b7`, PR #1178; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- Immediate and settled smoke reports were green: all five expected bots
  matched, hard failures were zero, 396 remote calls and 43 account-critical
  calls succeeded across the two windows with zero failures, and no fill,
  process, event-pipeline, or text-log hard failure appeared.
- Background replay remains memory/I/O intensive: settled direct probes showed
  four coin-HSL processes in uninterruptible sleep/page wait, 29 MB/s sampled
  swap-in, 18 MB/s swap-out, 32% I/O wait, and zero idle. PR #1178 restored
  live-I/O responsiveness but intentionally did not reduce this footprint.
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

The coin-HSL protective-readiness split and cooperative background cadence are
merged and deployed. The active read-only slice makes current process pressure
visible even when periodic event-derived health samples are absent or stale.
Remaining candidates:

- realistic-scale replay fixtures and deeper internal-stage profiling
- held-position protective-readiness source events and sequencing
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
