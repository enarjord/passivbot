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

- PR: not yet opened; `Prioritize protective coin-HSL replay candidates`
- Branch: `codex/v8-hsl-held-first-replay`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `493a61a9cb1659927a8d895fb25597b089988527`
- Scope: freeze the existing coin-HSL replay candidates and order them held
  pairs first, cooldown-affected pairs second, then all remaining pairs while
  preserving deterministic relative order. Force one bounded first-pair
  progress event so deployed ordering is directly observable.
- Non-goals: no early startup release, background replay, protective-ready
  claim, fresh-entry readiness change, HSL math/state change, cache contract,
  exchange call, or backtest behavior change.
- Local validation: full coin-HSL suite, focused HSL metric/startup/override
  suites, fake-live marker suite (`21/21`), Rust `209/209`, and a deterministic
  30-symbol/1440-minute/two-iteration replay benchmark pass. The loaded Rust
  extension source stamp matches this worktree; `py_compile`, added-line
  silent-handling audit, and `git diff --check` pass.
- Independent preflight: confirmed pair-local HSL state and that a mere sort
  cannot yet make protective orders executable because startup still awaits
  full replay. The later readiness split needs explicit protective/full state,
  fresh-entry blocking, shutdown coordination, and dedicated source events.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull while preserving local artifacts, controlled
  five-bot restart for the live Python runtime change, then immediate and
  settled bounded smoke reports. No Rust rebuild is required.

Next action:

1. Publish the review-ready PR with the exact non-goals above.
2. Resolve verified findings and merge only after the exact-head gate, then
   perform the declared VPS5 restart/smoke validation.

## Deployed Baseline

- Remote `v8`: `493a61a9`, PR #1175
- VPS5 repository: `493a61a9`, PR #1175
- VPS5 expected bots: five; all are running after the controlled restart
- Latest immediate, settled, and fresh smoke: `ok=true`, `hard_failures=0`, all
  five bots matched, zero remote/account-critical/fill-refresh failures, and
  zero text-log hard/attention matches. The fresh window contained one
  non-hard Hyperliquid `ema.unavailable` debug event. Three HSL replays were
  active, non-stale, not long-running, and making progress.
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

The denominator-parity prerequisite is merged and deployed. The active slice
adds immutable held/cooldown-first candidate ordering without claiming early
protective readiness. After that foundation merges, split held-position
protective readiness from full replay while keeping fresh entries blocked until
their cooldown eligibility is known. The design must preserve exact HSL
reconstruction and keep observability events out of decision authority.
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
