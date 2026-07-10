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

## Active Runtime PR

- PR #1170: `Add structured websocket reconnect events`
- Branch: `codex/v8-websocket-reconnect-events`
- Head: `40ad4a774b775a35ee1d0fcdcc0e11d985b79b33`
- Current `origin/v8`: `b7c514c0a44931ee09832bc934dce1d48705a38b`
  after PR #1163
- Scope: bounded structured/monitor-only websocket reconnect producer; no
  reconnect or trading behavior change
- CI: green on the current head
- Reviews: the prior `CHANGELOG.md` conflict findings are resolved; current-head
  Hermes and Grok delta reviews are pending
- Gate: not yet satisfied; GitHub reports `CLEAN`, but prior reviews apply to
  the superseded head
- State: runtime loop resumed; docs-only workflow PR #1171 remains orthogonal

Next action after the maintainer resumes the loop:

1. Wait for current-head Hermes and Grok delta reviews.
2. Reconfirm current `origin/v8`, mergeability, head SHA, reviews, and CI.
3. Merge PR #1170 only after the current-head gate is satisfied.
4. Pull VPS5 with autostash while preserving local artifacts.
5. Restart the five configured bots because #1170 adds a live producer.
6. Run immediate and settled bounded smoke checks without inducing a websocket
   or exchange failure.
7. Record merge/deploy evidence here and in the historical ledger.

## Deployed Baseline

- Remote `v8`: `b7c514c0`, PR #1163
- VPS5 repository: `8f836f30`, PR #1169
- VPS5 expected bots: five; all were running after the latest restart
- Latest immediate and settled smoke: `ok=true`, `hard_failures=0`, all five
  bots matched, zero failed remote/account-critical/fill-refresh calls, and
  zero text-log hard/attention matches
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

Do not start a dependent runtime slice until PR #1170 is merged and deployed.
After deployment, choose one review-worthy item from:

- high-value Phase 5 text-to-event migration
- missing performance/readiness source events that unblock measurement
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
