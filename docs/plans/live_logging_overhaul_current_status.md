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
  `Keep live I/O responsive during background coin-HSL replay`
- Branch: `codex/v8-hsl-background-replay-cooperative`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `abcc422bebb0ac19fcd50fcb8efa21d77b45d41f`
- Scope: retain the existing fast 1,000-row cooperative cadence while startup
  is blocked on held-position reconstruction. After protective readiness,
  yield every 100 rows with a 10 ms pause so the same-process live event loop
  can service exchange I/O while flat/cooldown pairs continue replaying.
- Triggering evidence: after PR #1177 released startup, the four coin-HSL replay
  bots were simultaneously in `blk_io_schedule`, VPS5 showed 40-44% I/O wait,
  20-29 MB/s swap-in, 19-23 MB/s swap-out, and zero CPU idle, while fresh
  account-critical remote failures occurred. The non-replay Hyperliquid bot
  remained in normal `ep_poll` sleep.
- Non-goals: no HSL math/state, pair ordering, readiness, entry gate,
  checkpoint/cache schema, exchange-call, Rust, or backtest behavior change;
  no replay vectorization or memory-layout rewrite.
- Local validation: full coin-HSL suite; full order-orchestration and
  exchange-config suites; 15 selected HSL/coin fake-live and integration tests;
  benchmark-tool tests; syntax and diff checks. The deterministic 30-symbol,
  1440-minute, two-iteration held-position benchmark retained its state hash
  and zero side effects. A realistic flat-background probe reconstructed all
  30 pairs in 6.37 seconds with no pending pairs.
- Independent preflight: one read-only test audit confirmed the exact replay
  task, held/cooldown/flat batch boundary, existing 1,000-row cadence, and
  benchmark/test surfaces. No delegated edits were used.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull while preserving local artifacts, controlled
  five-bot restart for the live Python runtime change, then immediate and
  settled bounded smoke reports. No Rust rebuild is required.

Next action:

1. Publish the hotfix, resolve verified findings, and merge only after the
   exact-head gate; then perform the declared VPS5 restart/smoke validation.

## Deployed Baseline

- Remote `v8`: `abcc422b`, PR #1177
- VPS5 repository: `abcc422b`, PR #1177; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- Immediate post-restart smoke was green: `ok=true`, `hard_failures=0`, all five
  bots matched, and zero remote/account-critical/fill/log failures.
- Protective-ready events proved startup release for GateIO, KuCoin, and OKX
  with no held positions, while exact background replay remained active.
  Settled probes then exposed the resource-pressure incident summarized in the
  active-slice evidence above. Five bots remain alive; do not restart them
  reflexively while the dependent hotfix is under review.
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

The denominator-parity, immutable held/cooldown-first ordering, and
held-position protective-readiness split are merged and deployed. The active
hotfix bounds event-loop monopolization during the resulting background replay
without changing replay results or pair readiness.
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
