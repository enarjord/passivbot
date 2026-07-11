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

- PR: not yet opened; `Release coin-HSL startup after held protection`
- Branch: `codex/v8-hsl-protective-readiness`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `078a3fead1dc635877b9142cb394dd93e3747d54`
- Scope: run the existing exact held-first coin replay in one task, release
  startup after every held pair is protectively ready, and continue remaining
  pairs in the background. Keep pending pairs blocked from initial-entry
  creates at both planning and final submission while allowing cancellations
  and panic/reduce-only creates. Expose protective/full timing and pair counts.
- Non-goals: no HSL math/state contract change, replay algorithm rewrite,
  checkpoint/cache schema change, exchange-call change, backtest behavior
  change, or claim that cold history materialization itself is now fast.
- Local validation: full coin-HSL, order-orchestration, exchange-config,
  live-smoke-report, and live-event-bus suites; six focused realized-loss/HSL
  tests; nine fake-live HSL/replay scenarios; Rust `209/209`, `cargo check
  --tests`, and `cargo fmt --check`; and the deterministic 30-symbol,
  1440-minute, two-iteration replay benchmark pass. The rebuilt extension stamp
  matches this worktree; `py_compile`, reviewed added-line exception handling,
  and `git diff --check` pass.
- Independent preflight: one read-only test-surface audit identified the
  existing replay, order, fake-live, smoke, and shutdown fixtures used by this
  slice. Runtime design keeps one writer per pending pair and never rebuilds
  shared HSL state after startup release. A separate current-diff concurrency
  and safety review reported no P0-P2 findings.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull while preserving local artifacts, controlled
  five-bot restart for the live Python runtime change, then immediate and
  settled bounded smoke reports. No Rust rebuild is required.

Next action:

1. Finish author validation and publish the review-ready PR with the exact
   non-goals above.
2. Resolve verified findings and merge only after the exact-head gate, then
   perform the declared VPS5 restart/smoke validation.

## Deployed Baseline

- Remote `v8`: `078a3fea`, PR #1176
- VPS5 repository: `078a3fea`, PR #1176
- VPS5 expected bots: five; all are running after the controlled restart
- Latest immediate and settled smoke: `ok=true`, `hard_failures=0`, all five
  bots matched, zero remote/account-critical/fill-refresh failures, and zero
  text-log hard/attention matches. Four HSL replays were active and non-stale;
  KuCoin completed, and the only extra settled-window signals were non-hard
  Hyperliquid EMA/state-refresh events.
- The prior tracked Rust formatting edit was already fully present in PR #1174.
  Its patch backup and autostash were retained on VPS5; tracked status is clean.
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

The denominator-parity and immutable held/cooldown-first ordering prerequisites
are merged and deployed. The active slice splits held-position protective
readiness from full replay while keeping fresh entries blocked until their
pair-specific cooldown eligibility is known. The design preserves exact HSL
reconstruction and keeps observability events out of decision authority.
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
