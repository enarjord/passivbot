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

- PR: [#1180](https://github.com/enarjord/passivbot/pull/1180),
  `Compact cold coin-HSL replay memory`
- Branch: `codex/v8-hsl-direct-matrix-replay`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `c159d955f691085469858c57ed1d6ba7927d7905`
- Scope: the internal cold coin-HSL history call requests aligned compact
  timestamp, balance, account-realized, and per-pair realized/unrealized NumPy
  arrays. `NaN` preserves unavailable pair values. Public history callers and
  pside/unified replay retain the rich timeline contract. The offline benchmark
  gains held/background counters, opt-in local-scale bounds, and tracemalloc
  output.
- Triggering evidence: after PR #1179, five live processes reported aggregate
  RSS `648196 KB`; four coin-HSL processes were in uninterruptible sleep while
  swap/page pressure remained high. A deterministic 43,201-minute, 30-symbol
  local builder profile measured `686242590` rich-history peak allocation bytes
  versus `73499666` compact-history bytes, an 89.3% reduction.
- Non-goals: no HSL thresholds, episode/cooldown/no-restart semantics, cache
  authority, pair priority, readiness gate, exchange write, process signal,
  Rust, backtest, or pside/unified behavior change.
- Local validation: full coin-HSL plus benchmark suites pass; the complete
  balance-history suite passes with one unrelated known startup test excluded
  after its asynchronous shutdown fixture stalled. Realized-loss tests pass.
  Syntax and diff checks pass. Broader validation and exact counts will be
  refreshed before publication.
- Independent preflight: two read-only audits independently identified the
  nested timeline as the dominant retained allocation and recommended a private
  compact coin-only handoff. A Luna worker changed only the benchmark and its
  tests; Sol owns the live-path implementation and review.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts, restart the five configured bots with exact
  tmux/process targeting, then compare protective/full replay time, process
  states, RSS, swap, I/O wait, remote calls, and hard-failure smoke output.

Next action:

1. Finish parity and broad local validation, publish the compact replay slice,
   resolve verified findings, and merge only after the exact-head gate; then
   perform the declared controlled VPS5 restart and comparative smoke.

## Deployed Baseline

- Remote `v8`: `c159d955`, PR #1179
- VPS5 repository: `c159d955`, PR #1179; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- Immediate and settled smoke reports were green: all five expected bots
  matched, hard failures were zero, 396 remote calls and 43 account-critical
  calls succeeded across the two windows with zero failures, and no fill,
  process, event-pipeline, or text-log hard failure appeared.
- Background replay remains memory/I/O intensive: current process-section
  output shows four of five bots in `D` state, aggregate RSS `648196 KB`, and
  the earlier settled direct probes showed
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

The coin-HSL protective-readiness split, cooperative background cadence, and
current process-pressure query are merged and deployed. The active slice
removes the dominant nested Python allocation from cold coin replay while
preserving the existing risk contract.
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
