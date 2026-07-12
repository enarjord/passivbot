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

- Branch: `codex/v8-event-pipeline-sink-attribution`
- Base: `cbf1e5ed5bf8bfed4bd967a02d853c099105eabc`
- Triggering evidence: PR #1200 measured live worker-service maxima above one
  second while drops, sink errors, and degraded counts remained zero, but the
  aggregate cannot identify whether the fixed structured or monitor sink class
  accounted for the delay.
- Scope: time each queued worker `sink.write()` attempt with a monotonic clock
  and aggregate fixed structured/monitor write counts plus service total/max
  over the existing health-summary timing window. Preserve aggregate worker
  service timing. Rotate, confirm, and restore the new counters with the same
  opaque timing token, including failed health-event enqueue and overlapping
  snapshots. Project the fixed numeric fields through event-pipeline smoke
  summaries and performance resource-pressure reports.
- Behavior boundary: observability only. Do not make timing evidence a trading
  gate, time synchronous console/text sinks, change queue capacity,
  backpressure, routing, drop/error policy, add raw payloads or unbounded
  labels, or alter order, risk, HSL, Rust, backtest, optimizer, or exchange
  behavior.
- Validation: event-bus route/timing and transactional restore tests, full
  smoke/performance projection tests, health-payload passthrough tests, fake-live
  event markers, Python compilation, `git diff --check`, added-line
  silent-handling audit, and independent preflight.
- Publication state, exact head, mergeability, CI, and current-head reviewer
  verdicts: query live GitHub metadata; do not embed self-invalidating values.
- Expected VPS action: exact five-bot graceful restart, then bounded
  `health.summary`/report evidence and immediate plus settled smoke. Preserve
  unrelated processes and local artifacts.

Next action:

1. Hold the published PR at the exact-current-head review gate until Hermes,
   Grok 4.5, and CI are green; resolve any findings with narrow delta validation
   and require fresh exact-head verdicts after every push.
2. Merge only after that gate is green, then gracefully restart the five exact
   VPS5 bot panes while preserving unrelated processes and local artifacts.
3. Verify fresh fixed structured/monitor sink timing with bounded
   `health.summary`, smoke, and performance reports, including immediate and
   settled process and repository-state checks.

## Deployed Baseline

- Remote `v8`: `cbf1e5ed5bf8bfed4bd967a02d853c099105eabc`, PR #1200
- VPS5 repository: `cbf1e5ed5bf8bfed4bd967a02d853c099105eabc`, PR #1200; tracked
  status clean; only expected untracked artifacts were preserved
- VPS5 expected bots: five; running with PR #1200 restart PIDs
  `865892/865897/865901/865903/865905`;
  unrelated `misc:0.0` remains PID `434835`
- PR #1200 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully stopped only the five exact bot
  panes; all old Python PIDs exited naturally, including KuCoin after a bounded
  uninterruptible wait. Pane PIDs and unrelated `misc:0.0` PID `434835` were
  preserved before the exact supervisor commands started the five new bots.
- Fresh smoke and performance reports projected three health windows with
  `1517` processed events, queue-wait total/max `29858.257/1077.844ms`, and
  worker-service total/max `63316.18/1033.432ms`; drops, sink errors, degraded
  counts, and unhealthy pipelines were zero. A real GateIO authoritative
  balance timeout made one intermediate smoke red. After recovery, the final
  two-minute smoke was green with `432/432` remote and `59/59`
  account-critical calls successful, all five expected processes matched,
  exact states `R/R/R/R/S`, and a clean tracked repository.
- PR #1199 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts. A bounded four-segment
  per-bot proof report returned `ok=true` with zero issues and reported all five
  current bots `proven`; cache loads preceded proof, and proof elapsed ranged
  from `5.851s` to `38.306s`.
- The first smoke retained a real KuCoin balance timeout; the next retained a
  recovered KuCoin nonce error. After both aged out, the final two-minute smoke
  was green with `283/283` remote and `39/39` account-critical calls successful,
  zero hard/log/monitor/process failures, and a clean tracked repository. A
  report-time `D` sample cleared; all five bots were `R/S` on the final exact
  state check. Bot PIDs, pane PIDs, and unrelated `misc:0.0` PID `434835`
  remained unchanged.
- PR #1198 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully stopped only the five exact bot
  panes; all old Python PIDs exited naturally. Existing pane PIDs and unrelated
  `misc:0.0` PID `434835` were preserved, then the exact supervisor commands
  started the five new bots.
- The immediate smoke retained a real KuCoin authoritative-state timeout:
  `264/267` remote and `15/18` account-critical calls succeeded. After recovery,
  the settled two-minute smoke was hard-green with `325/325` remote and `53/53`
  account-critical calls successful, all five expected processes matched, no
  hard/log/monitor/pipeline failures, no `D` states, and a clean tracked repo.
  Current post-restart segments contained neither `execution.*_sent` nor
  connector-call events, so no live order was fabricated for validation.
- PR #1197 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts. A bounded eight-segment
  Binance lifecycle report returned `ok=true` with zero issues and observed
  `first_fresh_entry_eligible` at `240.110s`, alongside first cycle, Rust call,
  and submitted write evidence. The settled two-minute smoke was hard-green
  with `208/208` remote and `53/53` account-critical calls successful, all five
  expected processes matched, no hard/log/monitor failures, and a clean tracked
  repository. One report-time `D` sample cleared; all five bots were `R` on the
  final exact-state check. Bot, pane, and unrelated `misc:0.0` PIDs remained
  unchanged.
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
latest-lifecycle report ordering, PR #1194's bounded startup action milestones,
PR #1195's startup consumer correctness fixes, PR #1196's true fresh-entry
eligibility producer, PR #1197's fresh-entry startup milestone, PR #1198's
local connector-call boundary evidence, and PR #1199's startup fill-cache proof
correlation are also merged and deployed. The active slice is selecting bounded
event-pipeline service-time evidence.

After the active service-time slice is merged and validated, select the next
review-worthy bounded operator-tooling improvement sharing one code and
validation surface.

Do not create progress-only PRs or resume unrelated logging work from stale
worktrees.

## References

- Operating workflow: `live_logging_overhaul_pr_loop_workflow.md`
- Architecture: `live_logging_overhaul_plan.md`
- Historical evidence: `live_logging_overhaul_progress.md`
- Performance goals: `live_performance_readiness_goals.md`
- Operational backlog: `live_ops_improvement_backlog.md`
- Reviewer loop: `../ai/pr_auto_review_loop.md`
