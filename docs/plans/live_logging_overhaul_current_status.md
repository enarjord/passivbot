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
  `Record HIP-3 fatal startup compatibility`
- Branch: `codex/v8-stock-perp-compatibility-events`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `b99d1b0520149299298252999864ac848c42ce42`
- Scope: before Hyperliquid's existing non-unified HIP-3 startup gate raises
  `FatalBotException`, emit and boundedly flush one off-console/off-text
  `config.market_compatibility` event. The payload aggregates only safe,
  bounded approved-position-open-order and margin-capability evidence; it does
  not retain order IDs, sizes, prices, raw payloads, config paths, or free-form
  exception text.
- Triggering evidence: the fatal gate distinguishes approved HIP-3 symbols,
  existing positions/open orders, isolated-only markets, and live isolated
  margin state in one human exception, but structured history contains only a
  generic startup error. A process can exit before queued semantic evidence is
  durable, leaving incident tooling unable to classify the compatibility
  failure without text-log parsing.
- Non-goals: no account abstraction detection, market/margin policy,
  fatal-decision or exception-message change; no generic isolated-only entry
  filter event, exchange call, configured-coin filtering, smoke verdict,
  process control, HSL/risk/order behavior, Rust, backtest, or optimizer change.
- Local validation: the focused Hyperliquid fatal-state, event-bus, smoke,
  enqueue/flush-failure, and registry-doc suite passes with 184 tests; Python
  compilation and `git diff --check` pass. Added-line silent handling is limited
  to the explicitly best-effort event enqueue/terminal flush. Independent Luna
  preflight is green after confirming approved-only diagnostics make no new
  margin-policy call and existing position/open-order paths preserve the prior
  fatal contract. Terra implemented the isolated producer/tests; Sol owns the
  terminal-event contract and publication.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts, restart the exact supervised Hyperliquid pane to
  load the Hyperliquid-only producer, and run immediate plus settled smoke
  checks. The configured VPS5 account is unified, so no fatal event is expected
  from healthy production startup; unit evidence owns the incompatible path.

Next action:

1. Complete the bounded HIP-3 fatal compatibility producer and regressions,
   obtain independent preflight plus exact-head reviewers and CI, resolve
   verified findings, and merge only when the full gate is green.

## Deployed Baseline

- Remote `v8`: `b99d1b05`, PR #1187
- VPS5 repository: `b99d1b05`, PR #1187; tracked status clean
- VPS5 expected bots: five; all running after the controlled restart
- PR #1187 was approved on exact head `74766c7cb` by Hermes, Grok 4.5, and the
  independent Codex reviewer; CI passed and it merged as `b99d1b05`. VPS5
  restarted only the five supervised bots, preserving `misc:0.0` PID `434835`.
  A bounded event query found the exact four expected per-side
  `config.market_compatibility` records: Binance `CRO,MNT` and OKX
  `KAS,MNT,XMR`, all non-hard degraded approved-list evidence. An immediate
  smoke caught a real KuCoin timeout; after it aged out, the settled smoke was
  hard-green with `370/370` remote and `26/26` account-critical calls
  successful, all five processes matched, states `R=4,S=1`, and no
  uninterruptible sleep.
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
rotated resource-pressure report fix plus configured-market skip events are
merged and deployed. The active slice makes fatal HIP-3 startup compatibility
available in durable structured history.
Remaining candidates:

- deeper internal-stage profiling if live residual cost remains material
- generic isolated-only entry-filter compatibility events
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
