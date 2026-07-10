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

- PR #1175: `Align coin-HSL denominator across live and backtest`
- Branch: `codex/v8-hsl-coin-denominator-parity`
- Head: query live GitHub metadata; this commit cannot embed its own final SHA
  without making that value stale
- Base: `61fbd0eb14efe8a3b77535b461eb1bf7c936fe41`
- Scope: one Rust-owned coin-HSL drawdown primitive shared by Python live/replay
  and Rust backtest. Slot budget is balance divided by the caller's applicable
  slot count; TWEL remains an activation/validation input but cannot scale HSL
  sensitivity.
- Non-goals: no TWEL activation change, no dynamic-tradability slot-count
  change, no RED tier/EMA/finalization change, no replay ordering change, no
  exchange call, and no new observability event.
- Local validation: Rust `209/209` plus default-feature test compilation pass;
  focused Python HSL/replay/binding suites `139/139`; fake-live `29/29`; real
  Binance BTC backtest and pymoo optimizer smoke pass; extension source stamp,
  `py_compile`, `cargo fmt --check`, and `git diff --check` pass.
- Independent preflight: confirmed exactly two prior denominator formulas,
  identified stale user docs, and verified that dynamic effective slots are an
  intentional backtest-only input to preserve.
- Publication state, exact head, mergeability, CI, and current-head review
  verdicts: query live GitHub metadata. Do not encode those transient values in
  the same PR that contains this status file, because every correction would
  create a different head and immediately stale the embedded value.
- Expected VPS action: pull with autostash, rebuild and verify the Rust
  extension, controlled five-bot restart, then immediate and settled bounded
  smoke reports

Next action:

1. Poll PR #1175's exact head, mergeability, CI, and required reviews.
2. Resolve every verified finding with focused regression coverage.
3. Merge only after the exact-head gate is satisfied, then perform the declared
   VPS5 restart/smoke validation.

## Deployed Baseline

- Remote `v8`: `61fbd0eb`, PR #1174
- VPS5 repository: `61fbd0eb`, PR #1174
- VPS5 expected bots: five; all are running after the controlled restart
- Latest immediate, settled, and fresh smoke: `ok=true`, `hard_failures=0`, all
  five bots matched, zero account-critical/fill-refresh failures, and zero
  text-log hard/attention matches. One non-hard Hyperliquid candle rate-limit
  event in the settled window cleared in the fresh window. Four HSL replays
  remained active, non-stale, not long-running, and making progress.
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

The coin-mode denominator parity correction is the active second prerequisite
for held-position protective readiness. After it merges and is deployed,
continue with immutable candidate replay and held-first protective sequencing.
The design must preserve exact HSL reconstruction and keep observability events
out of the decision authority. Remaining candidates:

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
