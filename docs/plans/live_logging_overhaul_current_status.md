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

- PR #1172: `Add offline HSL replay benchmark`
- Branch: `codex/v8-hsl-replay-benchmark`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `4a00ff171d9a8ef04e4591c797b7ef24952bd175`
- Scope: bounded deterministic offline benchmark for the current coin-HSL
  history initializer; no live/runtime HSL behavior change
- Output: explicit timeline-row and pair-row throughput, profiled timings and
  counters, deterministic fixture/final-state hashes, and side-effect counters
- Local validation: focused benchmark plus coin-HSL suites pass (113 tests),
  direct compact CLI smoke passes, `py_compile` and `git diff --check` pass
- Independent preflight: green after correcting the pair-row throughput unit
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: none for correctness; optional pull after merge, no bot
  restart

Next action:

1. Poll live GitHub metadata for PR #1172's current head, mergeability, CI, and
   required reviews.
2. Resolve any verified finding with focused regression coverage.
3. Merge only after the exact-head gate is satisfied.

## Deployed Baseline

- Remote `v8`: `4a00ff17`, PR #1170 after workflow-doc PR #1171
- VPS5 repository: `4a00ff17`, PR #1170
- VPS5 expected bots: five; all are running after the controlled restart
- Latest immediate and settled smoke: `ok=true`, `hard_failures=0`, all five
  bots matched, zero failed remote/account-critical/fill-refresh calls, and
  zero text-log hard/attention matches. Four HSL replays remained active but
  were neither stale nor long-running.
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

The offline replay benchmark is the active independent slice. After it merges,
use its repeatable evidence to choose one review-worthy item from:

- realistic-scale replay fixtures and deeper internal-stage profiling
- held-position protective-readiness source events and sequencing
- high-value Phase 5 text-to-event migration
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
