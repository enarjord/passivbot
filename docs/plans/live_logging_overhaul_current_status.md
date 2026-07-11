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
  `Emit configured-market compatibility events`
- Branch: `codex/v8-market-compatibility-events`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `60357a0071eca8fac80c76fa64f60d328d5ed4a0`
- Scope: emit one bounded, off-console/off-text
  `config.market_compatibility` event when an approved or ignored configured
  symbol is removed because it is absent from the exchange's eligible market
  set. Preserve the existing once-per-change text logs and filtering behavior;
  classify stock-perp-looking skipped symbols without changing compatibility
  policy. Existing generic event-query, timeline, problem-event, smoke, and
  incident-bundle surfaces consume the event without a bespoke aggregator.
- Triggering evidence: current VPS5 text logs repeatedly report unsupported
  approved coins for Binance (`CRO,MNT`) and OKX (`KAS,MNT,XMR`), including the
  current July 11 log segments, but the condition is absent from structured
  monitor history and cannot be reconstructed without scraping text logs.
- Non-goals: no exchange call, eligible-market calculation, configured-coin
  filtering, stock-perp margin/account policy, Hyperliquid fatal startup path,
  isolated-only entry filtering, smoke verdict, process control, HSL/risk/order
  behavior, Rust, backtest, or optimizer change.
- Local validation: focused live-event, coin-list, smoke, query, incident,
  registry-doc, compilation, diff, and added-line silent-handling checks pass.
  Terra implemented the isolated producer/tests. Luna's independent preflight
  found per-side query provenance, durable symbol bounds/redaction, retryable
  enqueue dedupe, and changelog gaps; two delta rounds resolved all findings
  and the final preflight is green. Sol owns event-contract adjudication and
  publication.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts, restart only the five supervised bots because
  this is a live event producer, verify bounded compatibility events for the
  known Binance/OKX skips, and run immediate plus settled smoke checks.

Next action:

1. Complete the bounded configured-market event producer and regressions,
   obtain independent preflight plus exact-head reviewers and CI, resolve
   verified findings, and merge only when the full gate is green.

## Deployed Baseline

- Remote `v8`: `60357a00`; the post-#1186 delta is an unrelated example-config update
- VPS5 repository: `b9748247`, PR #1186; tracked status clean
- VPS5 expected bots: five; all remained running without restart
- PR #1186 was approved on exact head `65d61702f` by Hermes, Grok 4.5, and the
  independent Codex reviewer that found both ordering regressions; CI passed
  and it merged as `b9748247`. VPS5 fast-forwarded without restarting bots.
  The exact bounded rotated `hsl_replay_profile` report returned `ok=true`,
  zero errors/warnings, and no resource-pressure crash. A settled two-minute
  smoke was hard-green with `216/216` remote and `56/56` account-critical calls
  successful, all five expected bots matched, and no text-log hard matches.
  The original five bot PIDs remained unchanged and returned to `Rl+`; unrelated
  `misc:0.0` remained PID `434835`. An earlier smoke window caught one real
  KuCoin timeout cycle, which aged out before the settled green window.
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
scorecard, stable per-pair fill index, exact sparse flat-pair replay, and the
rotated resource-pressure report fix are merged and deployed. The active slice
makes configured-market compatibility skips available in structured history.
Remaining candidates:

- deeper internal-stage profiling if live residual cost remains material
- stock-perp account, isolated-only, and fatal-live-state compatibility events
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
