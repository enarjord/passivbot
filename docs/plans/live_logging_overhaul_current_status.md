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
  `Index coin-HSL replay fills once`
- Branch: `codex/v8-hsl-fill-index`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `e5b4d26d3235b910c7549675f18dce865431cbb1`
- Scope: group cold coin-HSL fill history once by `(pside, symbol)` and reuse
  the stable per-pair lists for cooldown/intervention contract inference and
  position-size replay reconstruction. This removes repeated broad fill scans
  before the separate sparse minute-replay slice.
- Triggering evidence: post-PR #1180 VPS5 full replay remained between
  `453.980s` and `1728.585s`. Source inspection confirmed both repeated
  per-pair fill scans and the larger `pairs * timeline_rows` metric loop.
- Non-goals: no minute-row skipping, replay ordering or arithmetic change, HSL
  threshold/episode/cooldown behavior change, cache schema, exchange call,
  process control, Rust, backtest, or smoke-verdict change.
- Local validation: `144` coin-HSL replay, benchmark, and metric-regression
  tests pass; Python compilation and diff hygiene pass. A deterministic
  `30,000`-fill, `30`-pair comparison produced identical replay-event output
  and reduced this preprocessing substep from `0.181s` to `0.027s` locally.
- Independent preflight: Terra reported canonical cold-start history green and
  identified one conflicting-alias edge. The index now fails loudly when
  `pside` and `position_side` disagree, with a regression test. Sol owns final
  adjudication.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts, restart the five supervised bots, and compare
  replay timings plus the bounded smoke report. The initializer code is loaded
  only at process start.

Next action:

1. Finish validation and independent preflight, publish the fill-index slice,
   resolve verified findings, and merge only after the exact-head gate; then
   run the declared controlled VPS5 restart and smoke comparison.

## Deployed Baseline

- Remote `v8`: `e5b4d26d`, PR #1182
- VPS5 repository: `e5b4d26d`, PR #1182; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- Immediate and fresh settled smoke reports were green: all five expected bots
  matched, hard failures were zero, and the fresh window recorded `614/614`
  remote plus `90/90` account-critical calls succeeded. One Kucoin
  `RequestTimeout` cycle failure in the wider restart window recovered before
  the fresh smoke.
- All four coin-HSL bots emitted `history_format=compact` and protective-ready
  success. Kucoin completed full replay in `453.98s`; three background replays
  remained active with no required pair work pending.
- Compared with the pre-deploy sample, aggregate RSS fell from `694840 KB` to
  `555296 KB`, all five processes moved from four `D` states to five `R`
  states, and host swap use fell from `2926 MB` to `906 MB`. CPU remained high
  while background replay continued.
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
current process-pressure query, compact cold replay payload, and bounded replay
scorecard are merged and deployed. The active slice removes repeated broad fill
scans as the first behavior-neutral prerequisite for exact sparse full replay.
Remaining candidates:

- realistic sparse replay fixtures and dense-reference equivalence reporting
- exact lower-complexity minute replay using indexed episode/change boundaries
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
