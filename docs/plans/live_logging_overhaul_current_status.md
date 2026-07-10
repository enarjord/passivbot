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

- PR #1173: `Emit bounded forager eligibility events`
- Branch: `codex/v8-forager-eligibility-events`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `50f1dbaf582828c434f9f3f54ad482140f7deb0e`
- Scope: bounded structured/monitor-only events for existing approved/ignored
  forager membership changes; no eligibility, entry-gating, exchange, Rust,
  order, risk, or HSL behavior change
- Output: at most four aggregate events per refresh, with fixed source/list/
  operation fields and at most 12 sorted symbols per pside change row
- Local validation: focused coin-list/event-route/registry suites and the full
  fake-live suite pass; a real two-step fake-live run, `py_compile`, and `git
  diff --check` pass
- Independent preflight: green; repeated no-op refresh and `live_value` source
  coverage were added after its residual-risk review
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull with autostash, controlled five-bot restart, then
  immediate and settled bounded smoke reports

Next action:

1. Poll live GitHub metadata for PR #1173's current head, mergeability, CI, and
   required reviews.
2. Resolve any verified finding with focused regression coverage.
3. Merge only after the exact-head gate is satisfied, then perform the declared
   VPS5 restart/smoke validation.

## Deployed Baseline

- Remote `v8`: `50f1dbaf`, PR #1172
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

The forager eligibility event producer is the active Phase 5 slice. After it
merges and is deployed, the next high-value dependent slice is held-position
protective readiness. Its design must preserve exact HSL reconstruction and
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
