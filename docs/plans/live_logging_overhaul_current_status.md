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
  `Tolerate missing resource-pressure latest values`
- Branch: `codex/v8-resource-pressure-none`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `5526d5de21cf1b43cacc69f9b380a15bdf82e93b`
- Scope: make the read-only resource-pressure accumulator tolerate a historical
  `health.summary` field whose latest value is explicitly `null`, preserving
  valid zeroes and omitting unavailable numeric statistics according to the
  existing report contract. Add a focused regression and record PR #1184
  deployment evidence.
- Triggering evidence: after PR #1184 deployment,
  `live-performance-report --include-rotated --event-tail-lines 5000
  --max-event-files-per-bot 2 --section hsl_replay_profile` crashed in
  `_ResourcePressureAccumulator._field_stats` when `clean(None)` called
  `float(None)`. The lower-level event query with the same file/tail bounds
  remained healthy and recovered all four replay completion records.
- Non-goals: no live event producer, event parsing, monitor write, smoke verdict,
  exchange call, process control, restart behavior, HSL/risk/order behavior,
  Rust, backtest, or optimizer change.
- Local validation: the full performance-report suite, focused chronological
  and rotated regressions, Python compilation, diff hygiene, and added-line
  silent-handling scan pass. Terra implemented the isolated report/test patch;
  Luna's focused delta review reported no findings and green-lit publication;
  Sol adjudicated the contract.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts and rerun the exact rotated report query. No bot
  restart is expected because this slice changes only read-only report code.

Next action:

1. Advance the read-only report fix through focused validation, independent
   preflight, exact-head reviewers, and CI; resolve verified findings and merge
   only when the full gate is green, then rerun the rotated VPS5 query.

## Deployed Baseline

- Remote `v8`: `5526d5de`, PR #1184
- VPS5 repository: `5526d5de`, PR #1184; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- PR #1184 was approved by Hermes and Grok 4.5 on exact head `9177dfed9`, CI
  passed, and it merged as `5526d5de`. VPS5 fast-forwarded cleanly while
  preserving untracked artifacts. The five old bots stopped after the second
  exact-pane Ctrl-C; only the `passivbot` session was reloaded, and unrelated
  `misc:0.0` remained PID `434835`.
- Immediate and fresh settled smoke reports were hard-green with all five
  expected bots matched and zero hard failures. The settled window recorded
  `249/249` successful remote and `34/34` account-critical calls, process states
  `R=4, S=1`, no uninterruptible sleep, and tracked repository status clean.
- All four coin-HSL replays completed with strategy `mixed` and candidate
  reduction from `86.865%` to `93.449%`: KuCoin `140.746s`, Binance `208.098s`,
  GateIO `229.248s`, and OKX `269.711s`. The post-PR #1183 baseline was
  `601.246s` to `2279.519s`, so the deployed replay is about `77%` to `89%`
  faster while preserving dense held/ambiguous pairs.
- PR #1181 required no restart; all five bot pane PIDs remained unchanged. Its
  bounded current-segment scorecard reported four compact full-replay
  completions from `453.98s` to `1728.585s`, but exposed the active slice's
  missing completion fallback for the rotated protective milestones.
- PR #1182 required no restart and recovered protective elapsed aggregates
  from explicit completion evidence without synthesizing milestone records.
  A fresh recovery smoke was green with `279/279` remote and `66/66`
  account-critical calls successful; all five expected bots matched.
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
current process-pressure query, compact cold replay payload, bounded replay
scorecard, stable per-pair fill index, and exact sparse flat-pair replay are
merged and deployed. The active slice fixes the rotated resource-pressure
report crash exposed during post-deploy evidence collection.
Remaining candidates:

- deeper internal-stage profiling if live residual cost remains material
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
