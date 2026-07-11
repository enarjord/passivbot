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

- PR and publication state: unpublished; query live GitHub metadata after
  publication; `Replay flat coin-HSL history at exact change points`
- Branch: `codex/v8-hsl-sparse-replay`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `b29d5ca5fd3ffc2d75657459c0ab549e5484596d`
- Scope: compact cold coin-HSL replay keeps currently held pair history dense,
  but evaluates historical flat pairs only at exact account/pair run
  boundaries, rolling-lookback expiries, fill/marker/cooldown/required/restart
  boundaries, and the final row. Ambiguous fill replay falls back to the dense
  path. Structured replay events and the performance report expose bounded
  candidate/dense-equivalent row counts and strategy labels.
- Triggering evidence: post-PR #1183 full replay completed in `601.246s` for
  KuCoin and `914.691s` for Binance while OKX and GateIO remained active.
  The indexed-fill load stage took only `0.103s` to `0.213s`, confirming the
  per-pair minute loop is the dominant remaining local cost.
- Non-goals: no held-pair sample skipping, HSL arithmetic or threshold change,
  panic/cooldown/restart contract change, cache authority/schema, exchange
  call, process control, Rust, backtest, or smoke-verdict change.
- Local validation: the affected benchmark, coin-HSL, metric-regression, and
  performance-report suites pass. A deterministic 43,201-minute, 30-pair
  fixture with one held pair produced identical candidate/dense final-state
  hashes while reducing replay samples from `825430` to `43652`; all `43201`
  held-pair samples remained dense. The benchmark is offline and recorded zero
  network, cache-read, and cache-write side effects.
- Independent preflight: Terra identified three valid issues: held-pair density
  was data-dependent, benchmark equivalence omitted restart/cooldown state, and
  dense fallback telemetry was mislabeled. The implementation now forces held
  pairs dense, hashes the behavior-relevant runtime state, reports mixed/dense
  fallback counts, and has independent warning-clean regressions. Terra's final
  delta review reported no findings and green-lit publication; Sol adjudicated
  the fixes.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: after exact-head approval and merge, pull while
  preserving local artifacts, restart the five supervised bots, and compare
  candidate counts, full-replay timings, process pressure, and the bounded
  smoke report. The initializer code is loaded only at process start.

Next action:

1. Finish validation and independent preflight, publish the sparse-replay
   slice, resolve verified findings, and merge only after the exact-head gate;
   then run the declared controlled VPS5 restart and smoke comparison.

## Deployed Baseline

- Remote `v8`: `b29d5ca5`, PR #1183
- VPS5 repository: `b29d5ca5`, PR #1183; tracked status clean
- VPS5 expected bots: five; all are running after the controlled restart
- Immediate and fresh settled smoke reports were green: all five expected bots
  matched, hard failures were zero, and the fresh recovery window recorded
  `380/380` remote plus `42/42` account-critical calls succeeded. One Kucoin
  `RequestTimeout` cycle failure in the wider restart window recovered before
  the fresh smoke.
- All four coin-HSL bots emitted `history_format=compact` and protective-ready
  success in `13.337s` to `78.832s`. The first observed post-PR #1183 full
  completions were KuCoin at `601.246s` and Binance at `914.691s`; OKX and
  GateIO remained active without failures. The fill-index prerequisite reduced
  history-loaded work to `0.103s` to `0.213s`, but did not remove the dominant
  pair-minute loop.
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
scorecard, and stable per-pair fill index are merged and deployed. The active
slice uses exact sparse change points for historical flat-pair replay while
keeping held-pair work dense.
Remaining candidates:

- exact sparse replay deployment and VPS timing/process-pressure proof
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
