# Live Logging Overhaul Current Status

Updated: 2026-07-12.

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

- Branch: `codex/v8-fresh-entry-startup-milestone`
- Base: `2f6f61da6fdd6d61c8edbf204e2690d5ebf02e8d`
- Triggering evidence: PR #1196 now emits true local fresh-entry eligibility,
  but the startup milestone report cannot yet answer when the first initial
  entry survived every local pre-connector gate.
- Scope: derive one bounded current-lifecycle
  `first_fresh_entry_eligible` startup milestone only when an
  `entry.initial_eligibility` event reports a positive integer eligible count.
  Classify it as `entry_blocker` and preserve explicit unknown output when the
  qualifying event is absent from selected history.
- Behavior boundary: read-only performance report and tests only. Do not add or
  reinterpret a trading gate, treat blocked/candidate-free/protective-only
  events as readiness, or claim connector invocation or exchange success.
- Validation: focused and full performance-report tests, Python compilation,
  `git diff --check`, added-line silent-handling audit, and independent
  preflight.
- Publication state, exact head, mergeability, CI, and current-head reviewer
  verdicts: query live GitHub metadata; do not embed self-invalidating values.
- Expected VPS action: pull while preserving local artifacts, run an exact
  bounded current-plus-rotated startup milestone report, and settled smoke.
  Do not restart bots because this slice changes only a read-only consumer.

Next action:

1. Wait for the temporary Hermes + Grok 4.5 + green-CI gate on the exact current
   PR head. After merge, pull `v8` on VPS5 without restarting bots, run the
   bounded current-plus-rotated `startup_milestones` report, and complete a
   settled smoke while verifying the bot and unrelated-process PIDs are
   unchanged.

## Deployed Baseline

- Remote `v8`: `2f6f61da6fdd6d61c8edbf204e2690d5ebf02e8d`, PR #1196
- VPS5 repository: `2f6f61da6fdd6d61c8edbf204e2690d5ebf02e8d`, PR #1196; tracked
  status clean; only expected untracked artifacts were preserved
- VPS5 expected bots: five; all running with the PR #1196 restart PIDs
  `856325/856354/856386/856420/856456`;
  unrelated `misc:0.0` remains PID `434835`
- PR #1196 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded and restarted only the five supervised panes. A focused
  query observed three bounded eligibility events with eligible,
  distance-blocked, and no-candidate outcomes. The first smoke retained a real
  KuCoin timeout; after recovery, the settled two-minute smoke was hard-green
  with `348/348` remote and `43/43` account-critical calls successful. The
  quiet one-minute smoke remained green at `175/175` and `34/34`; transient
  report-time `D` samples cleared to exact `R/S` process states.
- PR #1195 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts, preserving bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID
  `434835`. Bounded readiness and milestone reports scanned 12 files with
  zero issues; incomplete sources no longer attached stale rotated lifecycle
  data, while KuCoin retained its bounded sparse HSL context and first-cycle
  milestone at `220.123s`. The first smoke caught a real KuCoin balance
  timeout. After it aged out, the retry was green with `284/284` remote and
  `57/57` account-critical calls successful, 5/5 bots matched, no hard/log/
  monitor failures, no event-pipeline errors, and a clean tracked repository.
  Two report-time `D` samples cleared; the quiet follow-up showed all five bots
  `Rsl+`.
- PR #1194 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without process signals, preserving bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID `434835`.
  The exact bounded rotated `startup_milestones` report returned `ok=true`,
  scanned 12 files / 46,748 records with zero errors/warnings, kept truncated
  lifecycle evidence explicitly unknown, and observed KuCoin's first cycle at
  `110.653s` without claiming unseen Rust/write milestones. The first smoke
  caught one recovered Binance `InvalidNonce` and two transient `D` samples.
  After settling, all `D` states cleared and the final two-minute smoke was
  hard-green: `384/384` remote and `76/76` account-critical calls successful,
  5/5 processes matched (`R=4,S=1`), no hard/log failures, no pipeline
  drops/sink errors, and a clean tracked repository.
- PR #1193 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts, preserving bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID `434835`.
  The exact capped current-plus-rotated report returned `ok=true`, scanned 12
  files with zero errors/warnings, retained all five current per-bot lifecycle
  snapshots, and preserved historical aggregates (`account` phase count six
  versus five current bots). The settled smoke was hard-green with `322/322`
  remote and `57/57` account-critical calls successful, all five bots matched,
  no event-pipeline drops/sink errors, and a clean tracked repository. Report
  I/O briefly produced three `D` samples; all five bots returned to `R` after
  the quiet follow-up.
- PR #1192 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 replaced bot PIDs `842617/842655/842687/842721/842757` with
  `850148/850296/850370/850436/850495`, preserving `misc:0.0` PID `434835`.
  Live reports accepted five account and execution-loop scopes, four
  held-position protective scopes, three first-market-state scopes, and a
  completed background-candle scope using canonical impact labels;
  `active-candle` remained timing-only. The immediate smoke caught three real
  KuCoin balance timeouts, then the settled two-minute smoke was hard-green
  with `380/380` remote and `16/16` account-critical calls successful, all five
  configured bots matched, no pipeline drops/sink errors, and a clean tracked
  repository. Two transient `D` states during cache/replay work cleared to
  `R/S` on the quiet follow-up sample.
- PR #1191 merged as `359929007dce0b47c023a36fdef90a7106ae46da` after
  exact-head Hermes and Grok 4.5 approval plus green CI, then was pulled to
  VPS5 without restarting bots. Real optional-replay progress with
  `required_pairs=0` retained positive generic dense remaining work while
  required remaining stayed zero; exact candidate terminal records stayed
  zero. A retained legacy KuCoin terminal record without candidate totals
  reported `legacy_terminal_no_candidate_rows`, generic remaining zero, and
  retained its diagnostic dense remainder. The settled two-minute smoke was
  green with `278/278` remote and `58/58` account-critical calls successful,
  all five expected bots matched, no hard/log/monitor failures, and a clean
  repository. Large rotated-history validation briefly produced host I/O waits;
  after a quiet interval all five bot processes were `Rsl+` with their
  original PIDs.
- PR #1190 merged as `6b2da757f2fbc590c12365870475176632269021` and was deployed
  to VPS5. All five exact supervised panes were restarted, unrelated
  `misc:0.0` PID `434835` was preserved, and immediate live evidence showed the
  new scanned counters, scan rate, and candidate-row source. All four HSL
  replays completed successfully in `108.782s` to `246.091s`, reaching
  protective-ready in `9.920s` to `71.744s`; `total_scanned_rows` equaled
  `candidate_rows`, `observed_candidate_work_pct=100`, and the source was
  `candidate_rows_terminal`. The final fresh smoke was `ok=true` with `333/333`
  remote and `45/45` account-critical calls successful, all five expected bots
  matched, no hard/log/monitor failures, and a clean tracked repository. One
  process was sampled in `D`, but no process hard failure occurred. A prior
  monitor-row parse boundary was immediately revalidated clean.
  A later direct process check showed `D` cleared: Binance, KuCoin, GateIO, and
  Hyperliquid were `Ssl+/ep_poll`, OKX was `Rsl+`; `vmstat` showed `b=0` before
  a transient `b=1` with low or zero sampled iowait.
- PR #1188 was approved on exact head `363eca852` by Hermes and Grok 4.5;
  CI passed and it merged as `bd169747`. VPS5 fast-forwarded cleanly and
  gracefully restarted only `passivbot:4.0`; Hyperliquid bot PID `842779` was
  replaced by `844272`, while the other four bot PIDs and unrelated
  `misc:0.0` PID `434835` remained unchanged. The unified account reached
  startup-ready in `48.00s` and full-warmup-ready in `74.77s`. A focused query
  found zero fatal HIP-3 compatibility events, as expected. The final settled
  smoke was hard-green with all five processes matched, `327/327` remote and
  `45/45` account-critical calls successful, states `R=4,S=1`, no
  uninterruptible sleep, no hard/log failures, and tracked repository status
  clean.
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
rotated resource-pressure report fix, configured-market skip events, and fatal
HIP-3 startup compatibility are merged and deployed. PR #1189's isolated-only
initial-entry filter visibility and PR #1190's HSL replay scanned-row
throughput, PR #1191's corrected active/legacy-terminal replay estimates,
PR #1192's machine-readable startup readiness SLA semantics, PR #1193's
latest-lifecycle report ordering, and PR #1194's bounded startup action
milestones, plus PR #1195's startup consumer correctness fixes, are also merged
and deployed. The active slice adds the missing true fresh-entry eligibility
producer after all existing local pre-connector gates.

After this producer is merged and validated, select the next review-worthy
candidate from the remaining backlog, including:

- optional connector-boundary evidence if operators need actual exchange-write
  invocation rather than the existing pre-call submitted event
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
