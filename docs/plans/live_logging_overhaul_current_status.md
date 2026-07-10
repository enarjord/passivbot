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

- PR #1174: `Centralize HSL episode finalization in Rust`
- Branch: `codex/v8-hsl-startup-transition-contract`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `f5395dda4de815bac4f587795a66deda8fb01bc5`
- Scope: pure Rust post-RED-episode transition shared by backtest and Python
  live/history replay. It owns caller-supplied persistent no-restart peak/
  drawdown evaluation, restart policy, explicit disposition, and exact cooldown
  deadline; coin live restart now retains that peak like pside/backtest. Python still
  owns exchange/fill-history proof, scope-flat detection, cache/latch I/O,
  orchestration, and order supervision.
- Non-goals: no RED trigger/tier math change, no signal-mode denominator
  change, no replay ordering/background concurrency change, no exchange call,
  and no new observability event.
- Local validation: Rust `207/207`; focused Python HSL/replay/binding suites
  pass; fake-live `29/29`, including seven real end-to-end pside HSL scenario
  runs; real Binance HSL backtest and four-evaluation optimizer smoke pass;
  extension source stamp and `git diff --check` pass.
- Independent preflight: identified an existing coin-mode slot-budget
  denominator mismatch between live and backtest plus a direct coin fake-live
  staged-planner epoch failure after panic flatten. Neither is introduced by
  this transition refactor; both remain focused follow-up work.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull with autostash, rebuild and verify the Rust
  extension, controlled five-bot restart, then immediate and settled bounded
  smoke reports

Next action:

1. Poll PR #1174's exact head, mergeability, CI, and required reviews.
2. Resolve any verified finding with focused regression coverage.
3. Merge only after the exact-head gate is satisfied, then perform the declared
   VPS5 restart/smoke validation.

## Deployed Baseline

- Remote `v8`: `f5395dda`, PR #1173
- VPS5 repository: `f5395dda`, PR #1173
- VPS5 expected bots: five; all are running after the controlled restart
- Latest immediate and settled smoke: `ok=true`, `hard_failures=0`, all five
  bots matched, zero failed remote/account-critical/fill-refresh calls, and
  zero text-log hard/attention matches. Four HSL replays remained active but
  were neither stale nor long-running. Five deployed
  `forager.eligibility_changed` events were observed, one per bot, with bounded
  source/list/operation/symbol payloads.
- Known VPS5 tracked edit to preserve:
  `passivbot-rust/src/equity_hard_stop_loss.rs`
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

The shared HSL episode-finalization transition is the first prerequisite for
held-position protective readiness. After it merges and is deployed, reconcile
the existing coin-mode live/backtest slot-budget denominator mismatch in its
own behavior PR, then continue with immutable candidate replay and held-first
protective sequencing. The design must preserve exact HSL reconstruction and
keep observability events out of the decision authority. Remaining candidates:

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
