# Live Logging Overhaul Progress

This file is the running progress ledger for `docs/plans/live_logging_overhaul_plan.md`.
Keep the plan as the architecture and decision record; update this file as PRs
merge, live smoke evidence changes, or new gaps are discovered.

## Update Policy

- Update this file when a logging-overhaul slice is merged to `v8`.
- Update it when a merged slice is deployed to VPS5 and smoke-tested.
- Keep entries factual and compact: PR/commit, scope, validation, and remaining
  gap.
- Do not use this file for design churn; unresolved design details belong in the
  plan or a focused handoff doc.

## Current Status

Last updated: 2026-06-27.

Current `origin/v8` logging-overhaul head:

- `87e53dac6` merge of PR #739, `Add restart plan process signal safety`.

Current review gate:

- Composer has been stopped/retired from this loop. The normal review gate is
  now Claude + Hermes + CI. For low-risk docs/tooling-only slices, a degraded
  gate may still be used after repeated Claude absence, but that exception must
  be called out in the progress evidence.

VPS5 deployment status:

- Repository pulled through PR #696 at `d850daf5`.
- Bots were restarted from `/root/bots_vps5.yaml` after PR #677 and left
  running. The old process set stopped after about 36 seconds before the tmuxp
  reload.
- `tmuxp load` returned a nonzero attach error because SSH had no terminal, but
  it created the `passivbot` session and all five configured bot processes.
- VPS5 post-restart smoke with text logs disabled and
  `--recent-minutes 2 --supervisor-config /root/bots_vps5.yaml` reported
  `ok=true`, `hard_failures=0`, `hard_problem_event_count=0`,
  `expected_total=5`, `matched_expected=5`, and `missing_expected=[]`.
- A wider post-restart smoke saw a real GateIO HSL ZEC long RED finalization
  during startup/replay and one non-hard Kucoin candle `RequestTimeout`; the
  narrower settled-window smoke confirmed no continuing hard failures.
- PRs #670 and #671 were pulled to VPS5 without bot restart because they only
  changed read-only smoke-report tooling. A later smoke with text logs enabled
  and `--recent-minutes 2 --supervisor-config /root/bots_vps5.yaml` reported
  `ok=true`, `hard_failures=0`, `hard_problem_event_count=0`,
  `logs.hard_matches=0`, `matched_expected=5`, and `missing_expected=[]`.
- PR #673 was also pulled to VPS5 without bot restart. A later 5-minute smoke
  with text logs and process matching enabled reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `remote_call_failures.total=0`, `matched_expected=5`, and
  `missing_expected=[]`. The new `risk_events` section surfaced GateIO ZEC long
  HSL RED cooldown directly as `hsl.status` `cooldown_active`.
- PR #675 was pulled to VPS5 without bot restart. A 2-minute smoke with
  `--log-window-unparsed-policy drop` and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `remote_call_failures.total=0`, `matched_expected=5`, and
  `missing_expected=[]`.
- PR #677 was pulled to VPS5 and bots were restarted. Repeated settled
  2-minute smokes with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `matched_expected=5`, and `missing_expected=[]`. One intermediate Kucoin
  authoritative account-refresh timeout was visible in structured events and
  recovered without intervention. Remaining smoke attention came from non-hard
  EMA readiness / staged-execution degradation and GateIO ZEC HSL cooldown, not
  from the PR #677 logging migration.
- PRs #678 and #679 were pulled to VPS5 without bot restart because they only
  changed docs and read-only smoke-report tooling. A later 2-minute smoke with
  `--log-window-unparsed-policy drop`, text logs enabled, and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, and
  `missing_expected=[]`. One non-hard Kucoin candle `RequestTimeout` remained
  visible in `remote_call_failures`. The smoke output now includes bounded
  `problem_events.latest_data` for persistent non-hard `ema.unavailable` and
  `cycle.degraded` groups, and stale unparseable traceback/header fragments are
  filtered by timestamp context instead of causing false recent-window matches.
- PR #680 was pulled to VPS5 without bot restart because it only updated
  progress docs.
- PR #682 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. The first smoke after deploy surfaced a real
  transient Kucoin authoritative account-refresh `RequestTimeout` as one hard
  problem group. A later settled 2-minute smoke with
  `--log-window-unparsed-policy drop`, text logs enabled, and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `matched_expected=5`, and `missing_expected=[]`. The new
  `problem_event_groups` section grouped the remaining known non-hard
  EMA/cycle/HSL attention without requiring inspection of individual event
  samples.
- PR #683 was pulled to VPS5 without bot restart because it only updated
  progress docs.
- PR #684 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. The pre-fix smoke showed the tool could still
  hard-fail on a stale Kucoin traceback header when the tailed log slice started
  in the middle of an old traceback with no timestamp context. After PR #684, a
  1-minute smoke with `--log-window-unparsed-policy drop`, text logs enabled,
  and `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `matched_expected=5`, and `missing_expected=[]`.
- PR #686 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. The smoke report confirmed
  `repository.branch=v8`, `repository.head=9e898019`,
  `repository.dirty=false`, `tracked_changes=0`, and all five configured bots
  running. The same smoke surfaced repeated Kucoin authoritative REST
  `RequestTimeout` events unrelated to the tooling change.
- PR #688 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs and
  `/root/bots_vps5.yaml` process matching reported `ok=true`,
  `hard_failures=0`, `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `remote_call_timings.total=637`, `matched_expected=5`, and
  `missing_expected=[]`. The new timing groups surfaced slow-but-successful
  candle remote fetches, including Kucoin `M/USDT:USDT` p95/max 48047ms.
- PR #690 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `remote_call_failures.total=0`,
  `remote_call_health.total=445`, `matched_expected=5`, and
  `missing_expected=[]`. The new health groups showed the highest-volume
  terminal remote-call categories by bot/component/kind/surface, including
  slow-but-successful Binance authoritative fill fetches and candle fetch
  groups.
- PR #692 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, and
  `missing_expected=[]`. The new top-level `remote_call_health` summary was
  present with `total=390`, `succeeded=389`, `failed=1`, `throttled=0`,
  `failure_pct=0`, and `throttled_pct=0`.
- PR #694 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 5-minute smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported `ok=true`, `hard_failures=0`,
  `hard_problem_event_count=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, and
  `missing_expected=[]`. The new `account_critical_remote_call_health`
  summary was present with `total=126`, `succeeded=126`, `failed=0`,
  `throttled=0`, `failure_pct=0`, and `throttled_pct=0`.
- PR #696 was pulled to VPS5 without bot restart because it only changed
  read-only smoke-report tooling. A 2-minute compact summary smoke with text
  logs, `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml`
  process matching reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `logs.attention_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, `remote_calls.total=169`, `remote_calls.succeeded=169`,
  `account_critical_remote_calls.total=58`, and
  `account_critical_remote_calls.succeeded=58`. Remaining `attention=true`
  came from known non-hard EMA readiness / staged-execution / HSL cooldown
  groups.
- PR #681 was merged after Claude + Hermes approval and green CI, then deployed
  to VPS5 with a bot restart because it added live state-refresh event
  producers. The final rebase changed only progress-doc context relative to
  Claude's reviewed code. The restart left all five configured bots running;
  Kucoin needed a second Ctrl+C before exit. A settled 2-minute smoke reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `account_critical_remote_calls.total=35`, `succeeded=35`,
  `remote_calls.total=321`, `succeeded=321`, `matched_expected=5`, and
  `missing_expected=[]`. A direct event query confirmed
  `state.refresh_timing` and `state.refresh_progress` events on VPS5. A wider
  smoke also surfaced a real GateIO ZEC long HSL RED finalization during
  startup/replay, unrelated to the state-refresh event slice.
- PRs #698 and #699 were merged after current-head Claude + Hermes approval and
  green CI, then pulled to VPS5 without bot restart because they only changed
  read-only smoke-report tooling. Immediate post-pull smokes caught real
  transient Kucoin authoritative `RequestTimeout` bursts at 13:16Z and 13:19Z;
  a later settled 2-minute compact summary smoke on `d5639813` reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `logs.attention_matches=0`, `matched_expected=5`, `missing_expected=[]`,
  `account_critical_remote_calls.total=81`, `succeeded=81`,
  `remote_calls.total=230`, and `succeeded=230`. Remaining attention came from
  known non-hard EMA/staged readiness and GateIO ZEC HSL cooldown groups.
- PR #701 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 without bot restart because it only changed read-only
  probe tooling and docs. The pull left all five configured bots running and
  repository state clean at `91d5db9f`. Immediate smoke was red from real,
  unrelated Kucoin live issues: authoritative balance/positions/open-orders
  timeouts, a timestamp/nonce recovery, and a later fill-refresh timeout. The
  new probe summary was exercised with a one-repeat authenticated
  `ticker-endpoint-probe` on `binance_01`; it produced
  `account_critical_health.total=3`, `succeeded=2`, `failed=1`, with balance
  and positions successful and `fetch_open_orders()` failing as `ExchangeError`.
  This validates the summary projection and exposes a follow-up for
  lower-impact/account-only probing and exchange-specific open-orders shape.
- PR #703 was merged after Claude + Hermes approval on the code delta, green CI,
  and a docs-only amendment that clarified `--account-only` still loads markets
  for open-orders symbol fallback. It was pulled to VPS5 without bot restart.
  A one-repeat authenticated `ticker-endpoint-probe --account-only` on
  `binance_01` reported `account_critical_health.total=3`, `succeeded=3`,
  `failed=0`; Binance `fetch_open_orders()` first failed as `ExchangeError`,
  then succeeded through `mode=symbol_fallback`. A follow-up 2-minute smoke at
  `a2f93456` reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `matched_expected=5`, and `missing_expected=[]`.
- PR #704 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only updated progress docs. A compact
  smoke at `65589e71` reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`.
- After `/root/bots_vps5.yaml` was updated to run the new HSL forager config on
  Binance, Kucoin, GateIO, and OKX while leaving `hyperliquid_tradfi`
  unchanged, a 10-minute smoke at `65589e71` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. One non-hard Kucoin candle `RequestTimeout` recovered
  in subsequent candle calls. Remaining attention came from known non-hard
  Hyperliquid tradfi EMA/staged readiness events.
- PR #705 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changed read-only smoke-report
  tooling. A 10-minute `--brief` smoke at `4e752000` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `repository.dirty=false`,
  `remote_calls.total=606`, `remote_calls.failed=1`,
  `account_critical_remote_calls.total=90`, and
  `account_critical_remote_calls.failed=0`. The single remote failure was a
  non-hard Kucoin candle timeout; no account-critical failure or hard problem
  event was present.
- PR #706 was merged under the low-risk docs gate after Hermes approval and
  green CI, then pulled to VPS5 without bot restart because it only updated
  progress docs.
- PR #707 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 and deployed with a bot restart because it changed live
  HSL console projection. The first Ctrl+C round stopped all five bot panes
  cleanly within 20 seconds; `tmuxp load -d /root/bots_vps5.yaml` then started
  all five configured bots. A settled 5-minute `--brief` smoke at `f384edbe`
  reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, `repository.dirty=false`,
  `remote_calls.total=329`, `remote_calls.failed=0`,
  `account_critical_remote_calls.total=57`, and
  `account_critical_remote_calls.failed=0`. Remaining attention came from known
  non-hard Hyperliquid tradfi EMA/staged readiness events. No `hsl.status`
  risk events occurred in the 5-minute smoke window, so the new coin-HSL console
  projection was not exercised on VPS5 yet.
- PR #709 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 and deployed with a bot restart because it adds startup
  fill-cache structured events. The first Ctrl+C round stopped Binance, but
  Kucoin, GateIO, OKX, and Hyperliquid remained as orphaned live processes after
  two Ctrl+C rounds; SIGTERM cleared them before reload. `tmuxp load -d
  /root/bots_vps5.yaml` then started all five configured bots. A settled
  5-minute `--brief` smoke at `71479c61` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `repository.dirty=false`,
  `remote_calls.total=186`, `remote_calls.failed=0`,
  `account_critical_remote_calls.total=30`, and
  `account_critical_remote_calls.failed=0`. Direct monitor-file checks found
  `fills.refresh_summary` `fill_cache_ready` events for all five bots. Remaining
  attention came from known non-hard Hyperliquid tradfi EMA/candle readiness.
- PR #711 was merged after current-head Claude + Hermes approval and green CI,
  then deployed to VPS5 with a bot restart because it adds an exchange
  time-sync live event producer. During shutdown, the first Ctrl+C round stopped
  Binance only; a second Ctrl+C stopped OKX and Hyperliquid, while Kucoin and
  GateIO needed SIGTERM before reload. A settled 5-minute `--brief` smoke at
  `0fa6269b` reported `ok=true`, `hard_failures=0`, `logs.hard_matches=0`,
  `matched_expected=5`, `missing_expected_count=0`, `remote_calls.failed=0`,
  and `account_critical_remote_calls.failed=0`. Remaining attention came from
  known non-hard Hyperliquid tradfi EMA/cycle readiness.
- PR #712 was merged after current-head Claude + Hermes approval and green CI,
  then pulled to VPS5 without bot restart because it only changes read-only
  smoke-report tooling. A settled 5-minute `--brief`/`--summary` smoke at
  `51ba92a3` reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, `missing_expected_count=0`,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`. The
  new process diagnostics reported
  `classification_source=local_process_table_command_match`,
  `tmux_pane_ownership=not_available_from_process_table`,
  `duplicate_configured_command_matches_count=0`, and
  `extra_passivbot_live_processes_count=0`.
- PRs #714 and #715 were merged under the low-risk tooling/docs degraded gate
  after repeated Claude absence. Hermes had approved the PRs before the final
  current-base rebases, CI was green on the rebased heads, and local focused
  tests passed after conflict resolution. They were pulled to VPS5 without bot
  restart because they only changed read-only operator tooling and docs. A
  settled 5-minute `--brief` smoke at `7b12d4b2` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `repository.dirty=false`, and zero
  duplicate/extra live process matches. A bounded summary showed only known
  non-hard EMA readiness, HSL cooldown, and staged-readiness attention. The new
  `shutdown_events` section was present with `total=0` because no bot restart
  occurred. A read-only `live-config-preflight` smoke against
  `configs/forager_3pos_hsl_2026-06-26.json` returned `ok=true` with one
  warning for missing short-side bot config, and did not contact exchanges.
- PRs #719 and #720 were merged under the low-risk tooling/observability
  degraded gate after Hermes approval, green CI, and local focused validation.
  Claude did not return before merge. #719 added a plan-only restart-smoke
  planner and #720 added the opt-in `ema` live-event debug profile. Both were
  deployed to VPS5 by fast-forward pull without bot restart because #719 is
  read-only operator planning and #720 is opt-in structured event enrichment.
  A 10-minute brief smoke at `27f597be` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `repository.dirty=false`, and no
  duplicate/extra live process matches. Remaining attention came from existing
  non-hard structured problem events and HSL status. A VPS5 dry-run
  `live-restart-smoke-plan` against `/root/bots_vps5.yaml` reported `ok=true`,
  `dry_run=true`, `execution_available=false`, five planned bots, no issues,
  and rejected operations including SSH, tmux signaling, process signaling, git
  checkout/pull/fetch, `passivbot live`, exchange/API calls, and credential
  loading.
- PRs #723-#724 and #728/#730/#732 added opt-in debug-profile enrichment for
  remote calls, candles, fills, HSL, and execution events. They were deployed
  to VPS5 without bot restart where appropriate because profiles are opt-in and
  default live behavior and console volume were unchanged. Settled brief smoke
  reports after those pulls showed all five expected bots running with no hard
  failures, no log hard matches, no failed account-critical remote calls, and
  only known non-hard EMA/staged-readiness or HSL attention.
- Claude's retrospective audit for already-merged PRs #713-#722 was posted on
  PR #723. Current `v8` contains follow-up coverage: PR #726 addressed
  shutdown-message echo, EMA debug-profile scoping, and Rust debug best-effort
  isolation; PR #734 addressed shareable path hygiene, flat config reporting,
  and HSL preview scalar/path hardening; and the cache-doctor candle coverage
  path now reports length mismatches using canonical row-count authority.
- PR #735 was pulled to VPS5 without bot restart because it only changed
  read-only startup-budget smoke projections. A later smoke initially surfaced
  real HSL RED events and one OKX active-symbol forager EMA readiness handoff
  error unrelated to the smoke-tooling change; a settled follow-up smoke
  reported `ok=true`, no hard failures, no log hard matches, all five expected
  bots matched, and no failed remote/account-critical calls. PR #736 recorded
  the EMA-readiness handoff gap in the live-ops backlog.
- PR #737 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 and deployed with a bot restart because it changed live forager EMA
  readiness behavior. The first Ctrl+C round left Kucoin running; a follow-up
  SIGINT stopped it, but the restart shell killed itself because the process
  match also matched the shell command. The stale tmux session was then killed
  and `/root/bots_vps5.yaml` was reloaded, leaving all five configured bots
  running. Settled 2-minute and 5-minute smokes at `e3429ee9` both reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.
- PR #739 was merged after Claude + Hermes approval and green CI, then pulled
  to VPS5 without bot restart because it only changes plan-only restart tooling
  and docs. A settled 5-minute smoke at `87e53dac` reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.

## Phase Checklist

| Area | Status | Evidence | Remaining Work |
|------|--------|----------|----------------|
| Phase 0: event contract and routing basics | Done enough to build on | `LiveEvent`, routing, pipeline, monitor-backed sink, schema/query constants | Keep registry stable; avoid ad-hoc event names in new slices |
| Phase 1: event bus around existing structured events | Mostly done | Cycle, data packet, snapshot, planning unavailable, Rust orchestrator, order wave, fill/state events | Continue tightening tests as new producers migrate |
| Phase 2: data gatherer events | Mostly done | Account remote-call cohorts, candle tail/coverage, fill refresh summaries, cache load/flush, warmup/startup timing | Not every exchange/network call is instrumented; richer remote-call payload summaries remain incremental |
| Phase 3: Rust planning and payload refs | Partially done | Rust orchestrator called/returned events, redacted error hardening, action/planning summaries | Full raw-ref retention/debug policy still limited |
| Phase 4: order lifecycle and risk transitions | Mostly done | Order wave lifecycle, create/cancel/confirmation events, HSL/risk mode events | Expand WEL/TWEL/unstuck transition coverage as those paths are touched |
| Phase 5: migrate meaningful text logs | Partially started | Some noisy EMA console output already reduced; PR #646 improves event-projected console summaries for already-routed execution events; PR #707 restores throttled coin-mode HSL position status console lines from existing `hsl.status` metrics; PR #709 mirrors fill-cache startup readiness into off-console `fills.refresh_summary` events; PR #711 mirrors CCXT timestamp/nonce recovery into off-console `exchange.time_sync` events | Migrate high-value stdlib logs to structured-event projections without increasing console noise |
| Phase 6: gatekeeper integration | Pending | Gatekeeper remains a planned producer | Instrument gate decisions once gatekeeper work resumes |
| Operator tools | In progress | `live-event-query`, trace summaries, order trace reconstruction, cycle trace reconstruction, time-window filters, `live-smoke-report` startup baselines/process liveness/remote-call failures/remote-call timings/remote-call health groups and top-level totals/account-critical health/risk-events/shutdown-events/time windows/unparseable-log policy/brief smoke counters/supervisor duplicate-extra process diagnostics, incident bundle trace/process/time-window reports, ID filters, `ticker-endpoint-probe` account-critical health summaries and account-only mode, `live-config-preflight` offline config summaries | Cross-bot incident workflow, safe restart orchestration, active probe expansion beyond account-critical endpoints |
| Operational restart goals | Split to adjacent work | PR #619 shutdown progress; PR #622 warm-cache startup; PR #656/#668 cache integrity smoke doctor | Continue separate reviewed PRs for shutdown/warmup/cache proof improvements |

## Merged Slices

### Foundation Before PR #619

- Added the first live event pipeline pieces and monitor projections.
- Emitted structured events for forager/EMA summaries, planning defer summaries,
  HSL replay lifecycle, planning symbol state, EMA bundle starts, planned actions,
  and confirmation timeouts.
- These commits established the initial cycle/planning/execution event chain used
  by later query and smoke tooling.

### PR #619: Shutdown Progress

- Branch: `codex/v8-shutdown-progress`.
- Scope: adjacent operations improvement, not logging core.
- Result: improved shutdown progress and bounded shutdown cancel grace coverage.
- Follow-up: continue shutdown interruption work outside logging-only PRs.

### PR #621: Live Event Query Helper

- Branch: `codex/v8-live-event-query-helper`.
- Scope: shared live event query schema constants and initial query helper.
- Result: provided a stable base for later CLI filters and incident tooling.

### PR #622: Startup Warm Cache

- Branch: `codex/v8-startup-warm-cache`.
- Scope: adjacent operations improvement.
- Result: improved live startup warm-cache reuse.
- Follow-up: continue cache proof and warmup optimization separately from
  observability-only slices.

### PR #623: Live Event Query Scope

- Branch: `codex/v8-live-event-query-scope`.
- Scope: bounded live event query directory scans and rotated scan defaults.
- Result: query helper became safer on VPS-sized monitor trees.

### PR #624: EMA Console Noise

- Branch: `codex/v8-ema-console-noise`.
- Scope: console cleanup.
- Result: reduced forager EMA console noise while keeping diagnostics available
  through structured/debug paths.

### PR #625: Candle Tail Event

- Branch: `codex/v8-candle-tail-event`.
- Scope: candle/EMA readiness observability.
- Result: emitted structured candle tail projection events.

### PR #626: Event Query Filter

- Branch: `codex/v8-live-event-query-filter`.
- Scope: query tooling.
- Result: added event-type filtering to `passivbot tool live-event-query`.

### PR #627: Warmup Cache Decision Event

- Branch: `codex/v8-warmup-cache-event`.
- Scope: startup/warmup observability.
- Result: emitted structured warmup cache decision events.

### PR #628: Startup Timing Event

- Branch: `codex/v8-startup-timing-event`.
- Scope: startup timing observability.
- Result: emitted startup timing events.

### PR #629: Cache Load Events

- Branch: `codex/v8-cache-load-events`.
- Scope: cache instrumentation.
- Result: emitted candle cache load events and hardened payload building.

### PR #630: Cache Load Event Throttle

- Branch: `codex/v8-cache-load-event-throttle`.
- Scope: high-volume policy.
- Result: throttled cache load events to keep structured output bounded.

### PR #631: Cache Flush Events

- Branch: `codex/v8-cache-flush-events`.
- Scope: cache instrumentation.
- Result: emitted cache flush events.

### PR #632: HSL Transition Events

- Branch: `codex/v8-hsl-transition-events`.
- Scope: risk/HSL observability.
- Result: emitted HSL red finalization events and fixed event dedupe between
  episodes.

### PR #633: Risk Mode Events

- Branch: `codex/v8-risk-mode-events`.
- Scope: risk mode observability.
- Result: emitted risk mode change events and covered halted HSL mode events.

### PR #634: Candle Coverage Events

- Branch: `codex/v8-candle-coverage-events`.
- Scope: candle coverage audit observability.
- Result: emitted candle coverage audit events.

### PR #635: Fill Refresh Events

- Branch: `codex/v8-fill-refresh-events`.
- Scope: fill refresh observability.
- Result: emitted fill refresh summary events and covered fill refresh resync
  summaries.

### PR #636: Rust Orchestrator Event Hardening

- Branch: `codex/v8-rust-orchestrator-event-hardening`.
- Scope: event emission safety.
- Result: hardened Rust orchestrator event emission and redacted orchestrator
  error events.

### PR #637: Live Ops Improvement Backlog

- Branch: `codex/v8-live-ops-improvement-backlog`.
- Scope: process tracking.
- Result: created the living operations improvement backlog and clarified live
  event query backlog work.

### PR #638: Live Event Query Filters

- Branch: `codex/v8-live-event-query-filters`.
- Scope: query tooling.
- Result: added additional live event query filters.

### PR #639: Live Smoke Report Tool

- Branch: `codex/v8-live-smoke-report-tool`.
- Scope: operator tooling.
- Result: added read-only live smoke report tooling for monitor/log inspection.

### PR #640: Health Summary Events

- Branch: `codex/v8-health-summary-events`.
- Scope: health observability.
- Result: emitted structured health summary events.

### PR #641: Live Incident Bundle

- Branch: `codex/v8-live-incident-bundle`.
- Scope: incident tooling.
- Result: added live incident bundle tool and redacted monitor snapshots.
- VPS5 evidence: bundle smoke created an archive successfully; tool returned
  attention because live GateIO HSL RED risk events were present, not because
  bundle generation failed.

### PR #642: Live Event Query ID Scopes

- Branch: `codex/v8-live-event-query-id-scopes`.
- Scope: query tooling.
- Result: added `bot_id`, `snapshot_id`, `plan_id`, `action_id`,
  `remote_call_group_id`, and related ID filters; timeline rendering now uses
  shared event ID keys.
- VPS5 evidence: deployed to VPS5 at `ad36d8ea`; `--remote-call-group-id`
  returned correlated Kucoin authoritative remote-call events.

### PR #643: Health Resource Pressure

- Branch: `codex/v8-health-resource-pressure`.
- Scope: health observability.
- Result: enriched structured `health.summary` events with resource pressure and
  live event pipeline counters.
- Review evidence: Cursor, Hermes, and Claude approved current head
  `d34241a4`; CI green; local targeted tests passed before merge.
- VPS5 evidence: pending pull/restart/smoke.

### PR #644: Logging And Ops Progress Tracking

- Branch: `codex/v8-live-logging-progress-tracker`.
- Scope: process tracking.
- Result: added this progress ledger and converted the live operations backlog
  into a living checklist with per-item statuses and a merged-work log.

### PR #645: Reason-Code Registry Slice

- Branch: `codex/v8-reason-code-registry-slice`.
- Scope: event taxonomy and drift prevention.
- Result: added shared `EventTags` and `ReasonCodes` registries for common live
  event tags/reason codes, migrated representative emitters without changing
  emitted strings, and documented the registry rule.

### PR #646: Console Event Summaries

- Branch: `codex/v8-console-event-summaries`.
- Scope: Phase 5 console/text projection.
- Result: improved `format_console_event()` with compact operator-facing tags
  and typed summaries for order waves, order writes, confirmation results, and
  Rust planning returns. Routes and console event volume were unchanged.

### PR #648: Live Event Trace Summaries

- Branch: `codex/v8-live-event-trace-summary`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --trace-summary` to aggregate
  matched live events by event type, level, status, reason code, ID scopes,
  symbol/side, and order-wave/action coverage. Summary counts cover all matched
  events even when `--limit` truncates the returned event sample.

### PR #649: Startup Timing Baselines In Smoke Report

- Branch: `codex/v8-startup-phase-budgets`.
- Scope: adjacent operations observability.
- Result: `passivbot tool live-smoke-report` now summarizes existing
  `bot.startup_timing` monitor events into latest per-phase timings and rolling
  median/p95/min/max baselines. Latest details are redacted before smoke-report
  or incident-bundle output.

### PR #651: Live Event Order Trace View

- Branch: `codex/v8-live-event-order-trace`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --order-trace` to reconstruct
  order-wave/action lifecycles from existing structured execution events. The
  view groups by `order_wave_id` and `action_id`, reports event/status/reason
  counts, confirmation events, symbol/pside/side sets, and bounded event
  samples with shortened order/client-order references.

### PR #652: Order Trace Progress Update

- Branch: `codex/v8-progress-after-order-trace`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #651 merged.

### PR #653: Live Event Registry Documentation

- Branch: `codex/v8-reason-code-registry-docs`.
- Scope: event taxonomy documentation and drift prevention.
- Result: added `docs/ai/live_event_registry.md` for stable event tags and
  reason codes, linked it from the AI docs router/logging guide, and added a
  doc drift test that compares documented values to `EventTags`/`ReasonCodes`.

### PR #654: Live Event Cycle Trace View

- Branch: `codex/v8-live-event-cycle-trace`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --cycle-trace` to reconstruct
  matched events by `cycle_id`. Each cycle contains bounded timeline samples,
  aggregate trace summaries, and nested order traces using the existing order
  lifecycle reconstruction.

### PR #655: Cycle Trace Progress Update

- Branch: `codex/v8-progress-after-cycle-trace`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PRs #652-#654 merged.

### PR #656: Local Cache Integrity Doctor

- Branch: `codex/v8-cache-integrity-doctor-slice`.
- Scope: adjacent operations tooling.
- Result: added `passivbot tool cache-integrity-doctor`, a read-only local
  cache smoke report for root presence, aggregate file/size counts, empty
  files, and corrupt JSON/NDJSON/NPY artifacts. This is an initial cache-doctor
  slice; it does not yet prove warm-cache coverage or HSL/fill metadata
  compatibility.

### PR #658: Cache Doctor Progress Update

- Branch: `codex/v8-progress-after-cache-doctor`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #656 merged.

### PR #659: Incident Bundle Trace Reports

- Branch: `codex/v8-incident-bundle-traces`.
- Scope: incident tooling.
- Result: `passivbot tool live-incident-bundle` now embeds existing
  `live-event-query` trace-summary and order-trace reports in `event_report.json`
  by default, includes cycle-trace reconstruction when scoped to `--cycle-id`,
  and supports `--no-trace-report` for compact bundles.
- VPS5 evidence: deployed to VPS5 at `27931c81`; a read-only bundle smoke on
  monitor data produced a tarball containing `trace_summary`, `order_trace`, and
  `cycle_trace` sections. The tool returned attention because the embedded
  smoke report saw existing GateIO HSL RED and EMA readiness degradation.

### PR #661: Live Smoke Process Status

- Branch: `codex/v8-smoke-process-status`.
- Scope: operator tooling.
- Result: `passivbot tool live-smoke-report` can now include an optional
  read-only `processes` section. With `--supervisor-config`, tmuxp-style
  expected `passivbot live` commands are compared against running live
  processes and missing expected bots become smoke hard failures. Incident
  bundles pass the same process snapshot through `smoke_report.json` when
  requested.
- VPS5 evidence: deployed to VPS5 at `72b3d931`; read-only smoke using
  `/root/bots_vps5.yaml` matched all five expected bots and left them running.
  The overall smoke exit remained nonzero because Kucoin authoritative state
  fetches had recent `RequestTimeout` events, not because process liveness
  failed.

### PR #663: Remote-Call Failure Smoke Summary

- Branch: `codex/v8-smoke-remote-call-summary`.
- Scope: operator tooling.
- Result: `passivbot tool live-smoke-report` now includes a bounded
  `remote_call_failures` aggregate section built from existing
  `remote_call.failed` monitor events. Groups are keyed by
  bot/reason/surface/error type/component and include latest redacted failure
  context.
- VPS5 evidence: deployed to VPS5 at `45b0cf9e`; read-only smoke using
  `/root/bots_vps5.yaml` still matched all five expected bots and now exposed
  Kucoin authoritative endpoint timeouts directly in the smoke output:
  positions=9, open_orders=7, balance=7.

### PR #665: Live Smoke Report Time Window

- Branch: `codex/v8-smoke-report-time-window`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` can scope structured monitor
  events with `--since-ms`, `--until-ms`, or `--recent-minutes`, and reports
  explicit event-window counters. Incident bundles pass the same window into
  embedded smoke reports. Text-log scanning remains intentionally unchanged.

### PR #666: Create Filter/Defer Events

- Branch: `codex/v8-execution-defer-events`.
- Scope: execution/order lifecycle observability.
- Result: create-order pre-exchange filter/defer decisions now emit bounded
  structured events after the existing gates decide. The execution gates remain
  authoritative, event emission is best-effort, and the new routes stay off the
  default console/text projection.

### PR #667: Live Event Query Time Window

- Branch: `codex/v8-event-query-time-window`.
- Scope: operator query tooling.
- Result: `passivbot tool live-event-query` can scope matched structured events
  with `--since-ms`, `--until-ms`, or `--recent-minutes`. The same scoped event
  set feeds query output, timeline, trace summary, order trace, and cycle trace
  views, with explicit event-window counters.

### PR #668: Cache Doctor Family Summary

- Branch: `codex/v8-cache-doctor-family-summary`.
- Scope: adjacent operations tooling.
- Result: `passivbot tool cache-integrity-doctor` now includes per-root and
  aggregate cache-family summaries plus family tags on reported issues. This is
  still read-only diagnostics and does not decide whether live trading may reuse
  a warm cache.
- VPS5 evidence: deployed to VPS5 at `734c2de0`. A settled 2-minute smoke after
  restart reported all five configured bots running and no hard problem events.

### PR #670: Smoke Report Timestamped Log Window

- Branch: `codex/v8-smoke-report-log-window`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` applies `since_ms`/`until_ms` and
  `--recent-minutes` windows to parseable ISO-UTC text log lines as well as
  structured monitor events. Unparseable log lines remain visible and are
  counted in `logs.window.unparsed_ts`.
- VPS5 evidence: deployed to VPS5 at `b74d12be` without bot restart. Smoke
  showed `logs.window.lines_skipped_before=2000`, proving stale parseable log
  lines were excluded, while two current Kucoin websocket warning lines were
  still false-classified as hard due to the older traceback matcher.

### PR #671: Smoke Report Traceback Prose Filter

- Branch: `codex/v8-smoke-report-traceback-pattern`.
- Scope: operator smoke tooling.
- Result: smoke-report text-log matching now treats only real Python traceback
  headers (`Traceback (most recent call last):`) as traceback signals, avoiding
  hard/attention matches for operational prose such as "suppressing callback
  traceback".
- VPS5 evidence: deployed to VPS5 at `34f63799` without bot restart. A
  2-minute smoke with text logs enabled reported all five configured bots
  running, `logs.hard_matches=0`, and no hard problem events.

### PR #673: Live Smoke Risk Event Summary

- Branch: `codex/v8-live-smoke-risk-summary`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes a bounded
  `risk_events` aggregate built from existing structured HSL/risk events. It
  groups by bot, event type, symbol, pside, and reason, keeps the latest compact
  risk fields such as tier, mode, drawdown score, distance to red, and cooldown
  timing, and does not change smoke `ok`/`attention`/hard-failure policy.
- VPS5 evidence: deployed to VPS5 at `2697ff48` without bot restart. A
  5-minute smoke with text logs and `/root/bots_vps5.yaml` process matching
  reported all five configured bots running, no hard failures, no log hard
  matches, and exposed GateIO ZEC long HSL RED cooldown in `risk_events`.

### PR #675: Smoke Log Unparsed Policy

- Branch: `codex/v8-smoke-log-unparsed-policy`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` and embedded incident-bundle smoke
  reports now accept `--log-window-unparsed-policy keep|drop`. The default
  `keep` preserves prior behavior; opt-in `drop` suppresses only non-signal
  unparseable text-log lines when a log window is active. Signal-bearing
  unparseable lines, including Python traceback headers, remain visible and can
  still make smoke hard-fail.
- VPS5 evidence: deployed to VPS5 at `3aa1e7a7` without bot restart. A
  2-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no log matches, no remote-call failures, and
  `logs.window.unparsed_policy=drop`.

### PR #676: Smoke Log Unparsed Policy Progress

- Branch: `codex/v8-progress-after-unparsed-policy`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #675 merged and VPS5 smoke confirmed the unparseable-log window policy.

### PR #677: Execution Loop Error Burst Event

- Branch: `codex/v8-execution-loop-error-burst-event`.
- Scope: Phase 5 text-log-to-event migration.
- Result: the existing execution-loop error burst warning now emits a bounded
  structured `health.summary` event with reason code
  `execution_loop_error_burst` before the existing stdlib warning. Emission is
  best-effort, uses the existing warning threshold, redacts/caps latest error
  text, and does not change restart/backoff/trading behavior or default console
  volume.
- Review evidence: Claude and Hermes approved head `409f5d8e`; focused pytest,
  compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 at `eda7cb2f` with a full bot restart. Three
  smoke windows reported all five configured bots running, no hard failures, no
  log hard matches, and no missing expected processes. Settled windows still
  showed non-hard EMA readiness / staged-execution degradation and GateIO ZEC
  HSL cooldown, which remain separate operational signals.

### PR #678: Execution Burst Progress Update

- Branch: `codex/v8-progress-after-execution-burst-event`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #677 merged and VPS5 smoke confirmed the execution-loop error burst event
  migration.

### PR #679: Smoke Problem Event Context

- Branch: `codex/v8-smoke-problem-event-data`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded,
  allowlisted `latest_data` for relevant problem-event groups such as
  `ema.unavailable` and `cycle.degraded`, with recursive redaction and payload
  bounds. The text-log scanner also uses the last parsed timestamp as context
  for unparseable continuation lines inside active windows, so stale traceback
  fragments after old errors are skipped while current traceback signals remain
  preserved.
- Review evidence: Hermes approved the original and amended delta; CI was
  green; focused smoke-report tests, compileall, and `git diff --check` passed
  before merge. Claude did not return during repeated polls, and Composer had
  been retired, so this docs/tooling-only slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `ff714b61` without bot restart. A
  2-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no text-log hard or
  attention matches, and exposed useful `latest_data` for the remaining
  non-hard EMA/cycle readiness groups.

### PR #680: Smoke Problem Context Progress

- Branch: `codex/v8-progress-after-smoke-problem-context`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #679 merged and VPS5 smoke confirmed the bounded problem-event context.

### PR #682: Smoke Problem Event Groups

- Branch: `codex/v8-smoke-problem-event-groups`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now reports top-level
  `problem_event_count` and bounded `problem_event_groups` aggregates grouped
  by bot, event type, reason code, status, hard flag, symbol, and position
  side. Existing bounded `problem_events` samples remain available for detail.
- Review evidence: Hermes approved head `048e8595c`; CI was green; focused
  smoke-report tests, compileall, and `git diff --check` passed before merge.
  Claude did not return during repeated polls, and Composer had been retired,
  so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `998d7c9c` without bot restart. The first
  smoke surfaced a real transient Kucoin authoritative account-refresh timeout;
  a later settled 2-minute smoke reported all five configured bots running, no
  hard failures, no hard problem events, no log matches, no remote-call
  failures, and grouped the remaining known non-hard EMA/cycle/HSL attention in
  `problem_event_groups`.

### PR #683: Smoke Grouping Progress Update

- Branch: `codex/v8-progress-after-smoke-grouping`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #682 merged and VPS5 smoke confirmed grouped problem-event summaries.
- Review evidence: Hermes approved head `3fa8d819`; CI was green; `git diff
  --check` passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this docs-only slice used the degraded gate.

### PR #684: Contextless Traceback Smoke Filter

- Branch: `codex/v8-smoke-drop-contextless-traceback`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report --log-window-unparsed-policy drop`
  now skips unparseable log lines without in-window timestamp context, avoiding
  stale hard failures when the inspected log tail begins in the middle of an old
  traceback. The direct smoke-report and incident-bundle help text now describes
  the context-based drop behavior.
- Review evidence: Hermes approved original head `f01996ec` with one minor
  help-text mismatch, then approved the fixed head `04ca717`; CI was green;
  focused smoke-report tests, compileall, and `git diff --check` passed before
  merge. Claude did not return during repeated polls, and Composer had been
  retired, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `dff40001` without bot restart. A
  1-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and no remote-call failures.

### PR #686: Smoke Repository Metadata

- Branch: `codex/v8-smoke-repo-metadata`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes a best-effort
  `repository` block with worktree root, branch, head, full head, tracked-only
  dirty status, and tracked change count. Git lookup failures remain
  observational and do not affect smoke hard-failure accounting.
- Review evidence: Hermes approved head `b03f4139`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, and `git diff --check`
  passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this read-only tooling slice used the degraded
  gate.
- VPS5 evidence: deployed to VPS5 at `9e898019` without bot restart. The smoke
  report confirmed `repository.branch=v8`, `repository.head=9e898019`,
  `repository.dirty=false`, `tracked_changes=0`, and all five configured bots
  running. The same smoke surfaced a separate Kucoin operational issue:
  repeated authoritative balance/positions/open-orders `RequestTimeout` events
  with 98-140s staged refresh wall times and websocket ping timeouts.

### PR #688: Remote-Call Timing Smoke Summary

- Branch: `codex/v8-smoke-remote-call-timings`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded
  `remote_call_timings` groups for terminal remote calls that expose elapsed
  median, p95, min, max, latest ids, and latest elapsed time. The section is
  observational only and does not affect `ok`, `attention`, or trading
  behavior.
- Review evidence: Hermes approved head `9945a3d3`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, and `git diff --check`
  passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this read-only tooling slice used the degraded
  gate.
- VPS5 evidence: deployed to VPS5 at `11f7d142` without bot restart. A
  5-minute smoke with text logs and `/root/bots_vps5.yaml` process matching
  reported all five configured bots running, no hard failures, no hard problem
  events, no log hard or attention matches, no remote-call failures, and
  `remote_call_timings.total=637`.

### PR #690: Remote-Call Health Smoke Summary

- Branch: `codex/v8-smoke-remote-call-health`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded
  `remote_call_health` groups that roll up terminal remote calls by
  bot/component/kind/surface, with success/failure/throttle counts, failure and
  throttle percentages, latency summaries, reason/error counts, and bounded
  affected-symbol samples. Throttled terminal buckets are derived from
  `event_type`, while differing raw statuses remain auxiliary context.
- Review evidence: Hermes first found that `remote_call.throttled` events with
  raw `status="deferred"` were not counted as throttles, then approved fixed
  head `dc99378a`; CI was green; focused smoke-report and incident-bundle
  tests, compileall, and `git diff --check` passed before merge. Claude did
  not return during repeated polls, and Composer had been retired, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `b150176f` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, no remote-call failures, and `remote_call_health.total=445`.

### PR #692: Remote-Call Health Top-Level Totals

- Branch: `codex/v8-smoke-remote-call-health-totals`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes top-level
  `remote_call_health` success, failure, throttle, failure-percent, and
  throttle-percent totals in addition to the existing bounded per-group health
  details. This keeps operator smoke summaries scannable without changing
  `ok`, `attention`, event producers, or trading behavior.
- Review evidence: Hermes first found that the new aggregate failure/throttle
  counters could be overwritten by per-group counters, then approved fixed head
  `ac4afe3f`; CI was green; focused smoke-report and incident-bundle tests,
  compileall, and `git diff --check` passed before merge. Claude did not return
  during repeated polls, and Composer had been retired, so this read-only
  tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `c8ce4880` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and top-level `remote_call_health` totals:
  `total=390`, `succeeded=389`, `failed=1`, `throttled=0`, `failure_pct=0`,
  and `throttled_pct=0`.

### PR #694: Account-Critical Remote-Call Health

- Branch: `codex/v8-smoke-authoritative-health`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes
  `account_critical_remote_call_health`, a filtered view of terminal
  authoritative balance, position, open-order, and Hyperliquid split
  account-state surfaces. It reuses the existing bounded remote-call health
  summarizer while excluding broader candle/fill traffic.
- Review evidence: Hermes approved head `bebbb3f6`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, `git diff --check`, and
  the touched-file silent-handling audit passed before merge. Claude did not
  return during repeated polls, and Composer had been retired, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `3299c1ca` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and account-critical health totals:
  `total=126`, `succeeded=126`, `failed=0`, `throttled=0`, `failure_pct=0`,
  and `throttled_pct=0`.

### PR #696: Concise Live Smoke Summary

- Branch: `codex/v8-smoke-report-summary`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report --summary` now projects the full
  report down to high-signal smoke fields: health booleans/counters,
  repository state, monitor totals, event/log windows, process summary,
  bounded problem groups, remote-call/account-critical health, and risk events.
  `--compact` can be combined with `--summary` for short machine-readable
  output. Full report generation and exit-code behavior are unchanged.
- Review evidence: Hermes approved head `f1efbe45`; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge. Claude did not return during
  repeated polls, and Composer had been retired, so this read-only tooling
  slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `d850daf5` without bot restart. A
  2-minute compact summary smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported all five configured bots running, no hard failures, no log
  hard or attention matches, account-critical calls `total=58`,
  `succeeded=58`, and all terminal remote calls `total=169`,
  `succeeded=169`.

### PR #681: Staged Refresh Timing Events

- Branch: `codex/v8-staged-refresh-events`.
- Scope: state refresh observability.
- Result: staged refresh timing and progress logs now emit structured
  `state.refresh_timing` and `state.refresh_progress` events with bounded
  timing, plan, residual, and pending-surface data. Emission is best-effort and
  does not change refresh behavior or console log text.
- Review evidence: Claude approved code-identical head `bddf5a4c`, Hermes
  approved rebased head `4be87ce7`, and CI was green; focused
  state-refresh/smoke tests, compileall, and `git diff --check` passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `9a52d3a9` with a bot restart. A settled
  2-minute smoke reported `ok=true`, no hard failures, no log hard/attention
  matches, all five configured bots running, and all account-critical/terminal
  remote calls succeeding. A direct event query confirmed
  `state.refresh_timing` and `state.refresh_progress` events. A wider smoke
  surfaced a real GateIO ZEC HSL RED finalization during startup/replay,
  unrelated to the event slice.

### PR #698: Smoke Repository Root Redaction

- Branch: `codex/v8-smoke-report-redact-repo-root`.
- Scope: operator smoke tooling privacy.
- Result: `live-smoke-report` now redacts current-home, `/root`,
  `/home/<user>`, and `/Users/<user>` prefixes from the serialized
  `repository.root` field while continuing to run git commands against the real
  resolved repository path. Incident-bundle `smoke_report.json` inherits the
  same safer display field.
- Review evidence: current-head Claude and Hermes approved head `7c7368f3`;
  CI was green; focused smoke-report/incident-bundle tests, compileall, and
  `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 as part of the later `d5639813` pull without
  bot restart. The settled smoke reported clean repository state on `v8`, all
  five configured bots running, and no hard failures.

### PR #699: Dropped Unparsed Smoke Log Signal Counters

- Branch: `codex/v8-smoke-report-dropped-unparsed-counters`.
- Scope: operator smoke tooling.
- Result: when `--log-window-unparsed-policy drop` suppresses a contextless
  unparseable log line that still matches attention/hard patterns, smoke
  reports now expose dropped attention/hard counters and include dropped
  attention signals in `attention_count`. Dropped contextless fragments remain
  excluded from `hard_failures`, preserving the stale-tail false-positive
  suppression from PR #684.
- Review evidence: current-head Claude and Hermes approved rebased head
  `4e2fcee7`; CI was green; full smoke-report and incident-bundle tests,
  compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 at `d5639813` without bot restart. Two
  immediate smokes caught real Kucoin authoritative `RequestTimeout` bursts; a
  later settled 2-minute compact summary smoke reported `ok=true`, no hard
  failures, no log hard/attention matches, all five configured bots running,
  account-critical calls `total=81`, `succeeded=81`, and all terminal remote
  calls `total=230`, `succeeded=230`.

### PR #709: Fill Cache Ready Event

- Branch: `codex/v8-fills-cache-ready-event`.
- Scope: Phase 5 startup fill-cache observability.
- Result: startup fill-cache readiness now emits a structured
  `fills.refresh_summary` event with reason code `fill_cache_ready`, source
  `startup`, refresh mode `cache_load`, elapsed time, event count, and optional
  history scope. The existing console line remains unchanged and the structured
  event route stays off console/text.
- Review evidence: current-head Claude and Hermes approved head `f5838dfea`,
  and CI was green. Focused live-event, fill-cache init/update, monitor emitter,
  compileall, and `git diff --check` checks passed before merge.
- VPS5 evidence: deployed to VPS5 at `71479c61` with a bot restart. A settled
  5-minute brief smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, and no
  failed account-critical remote calls. Direct monitor-file checks found
  `fill_cache_ready` events for Binance, GateIO, Hyperliquid, Kucoin, and OKX.

### PR #711: Exchange Time-Sync Event

- Branch: `codex/v8-exchange-time-sync-event`.
- Scope: Phase 5 exchange timestamp/nonce recovery observability.
- Result: CCXT timestamp/nonce recovery now emits bounded
  `exchange.time_sync` events for recovery success or unavailable exchange
  hooks. The event route stays off console/text, and event emission is
  best-effort so it cannot mask the original recovery path.
- Review evidence: current-head Claude and Hermes approved head `225a0e2b8`,
  and CI was green. Focused live-event, exchange time-sync recovery, execution
  loop timestamp-error, compileall, and `git diff --check` checks passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `0fa6269b` with a bot restart. A settled
  5-minute brief smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, and no
  failed account-critical remote calls.

### PR #712: Supervisor Process Diagnostics

- Branch: `codex/v8-smoke-supervisor-process-diagnostics`.
- Scope: operator smoke tooling.
- Result: `live-smoke-report --supervisor-config` now classifies expected
  process matches, duplicate configured-command matches, and extra/orphan-like
  `passivbot live` processes from bounded local process-table metadata. The
  report explicitly states that tmux pane ownership is not available from this
  read-only process-table classifier.
- Review evidence: Claude first found the no-RSS `ps` fallback row parser
  dropped valid process rows, then approved fixed head `da39c8af`; Hermes
  approved the same fixed head, CI was green, and focused smoke-report and
  incident-bundle tests plus `git diff --check` passed before merge.
- VPS5 evidence: pulled to VPS5 at `51ba92a3` without bot restart. A settled
  5-minute brief/summary smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, no failed
  account-critical remote calls, and zero duplicate/extra live process matches.

### PR #714: Live Config Preflight Tool

- Branch: `codex/v8-live-config-preflight`.
- Scope: adjacent operator tooling.
- Result: added `passivbot tool live-config-preflight`, a read-only offline JSON
  report for one live config covering identity hints, HSL settings, approved and
  ignored universe counts with bounded samples, forager slots/staleness, and
  cache-related live settings. The tool does not load credentials, contact
  exchanges, or enforce startup policy.
- Review evidence: Hermes approved the original head `b2557db0`; after PR #713
  merged, the branch was rebased to `b51a15a2` with the same tool code plus
  resolved progress-doc context. CI was green, local focused preflight tests
  passed, and `git diff --check` passed. Claude did not return during repeated
  polls, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `7b12d4b2` without bot restart. A
  `live-config-preflight --compact` smoke against
  `configs/forager_3pos_hsl_2026-06-26.json` returned `ok=true`, reported
  bounded approved/ignored coin samples and HSL/forager/cache settings, and
  surfaced one warning for missing short-side bot config.

### PR #715: Shutdown Event Smoke Summary

- Branch: `codex/v8-smoke-shutdown-summary`.
- Scope: operator smoke tooling.
- Result: existing `bot.stopping`, `bot.shutdown.stage`, and `bot.stopped`
  structured events are now summarized as `shutdown_events` in the full,
  `--summary`, and `--brief` smoke-report projections. The change is passive and
  does not add shutdown control, process signaling, or trading behavior.
- Review evidence: Hermes approved heads `01574b4d` and `7dfa6c4a`; after
  PR #714 merged, the branch was rebased to `befade50d` with only current-base
  changelog/tool-doc context added underneath. CI was green, local
  smoke-report/incident-bundle tests and compileall passed, and `git diff
  --check` passed. Claude did not return during repeated polls, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `7b12d4b2` without bot restart. A settled
  5-minute brief/summary smoke reported no hard failures, no log hard matches,
  all five configured bots running, no failed remote/account-critical calls,
  clean tracked repository state, and `shutdown_events.total=0` because no
  restart occurred.

### PR #719: Live Restart Smoke Plan Tool

- Branch: `codex/v8-live-restart-smoke-plan`.
- Scope: adjacent operator restart/smoke tooling.
- Result: added `passivbot tool live-restart-smoke-plan`, a read-only dry-run
  planner for the repeated live restart/smoke routine. The tool parses a
  tmuxp-style supervisor config through the existing sanitized smoke-report
  parser and emits structured plan metadata, per-bot phases, repo checks, smoke
  command wiring, timeout/escalation guidance, and explicit non-execution
  policy. `--execute` is rejected; the tool does not SSH, invoke tmux, signal
  processes, pull code, start bots, contact exchanges, or load credentials.
- Review evidence: Hermes approved head `c7c4ec09`; CI was green; local
  focused restart-plan, smoke-report, CLI dispatch, compileall, and `git diff
  --check` validation passed before merge. Claude did not return before merge,
  so this plan-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `27f597be` without bot restart as part of
  the PR #719/#720 deploy. A dry-run plan against `/root/bots_vps5.yaml`
  reported `ok=true`, `dry_run=true`, `execution_available=false`, five planned
  bots, no issues, and the expected rejected operations.

### PR #720: EMA Live Event Debug Profile

- Branch: `codex/v8-ema-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `ema` as a `logging.live_event_debug_profiles` /
  `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` profile, including `ema-readiness`
  aliases. When `ema` is enabled, existing `ema.unavailable` events include
  bounded parsed EMA type, span, and inner reason summaries.
  Default events remain compact, console output is unchanged, and no
  exchange/order/risk behavior changed.
- Review evidence: Hermes approved original head `1a9a53218`; after PR #719
  merged, the branch was rebased to `b8be04654` with the same code delta, CI was
  green, and local `tests/test_live_event_bus.py`,
  `tests/test_passivbot_monitor.py`, compileall, and `git diff --check`
  validation passed. Claude did not return before merge, so this opt-in
  observability slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `27f597be` without bot restart because the
  profile is opt-in and no VPS config enabled it. A 10-minute brief smoke
  reported all five configured bots running, no hard failures, no log hard
  matches, no failed remote calls, no failed account-critical remote calls, and
  clean tracked repository state.

### PR #722: Cache Doctor Candle Coverage Evidence

- Branch: `codex/maxwell-cache-integrity-doctor-13a`.
- Scope: adjacent cache/warmup diagnostics.
- Result: `passivbot tool cache-integrity-doctor` now derives v2 candle
  coverage windows, valid row counts, suspicious interior gap samples, and
  non-enforcing warm-cache evidence labels from local `.valid.npy` artifacts.
  It remains read-only and does not change cache materialization, startup, or
  trading behavior.
- Review evidence: Hermes approved head `43f6d17b`; CI was green; Maxwell ran
  focused cache-doctor tests, compileall, `git diff --check`, and a touched-file
  silent-handling audit before opening the PR. Claude did not return during the
  merge window, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed as part of the `09ae3773` pull without bot restart.
  The 10-minute brief smoke reported all five configured bots running, clean
  tracked repository state, no hard failures, no log hard matches, no failed
  remote calls, and no failed account-critical remote calls.

### PR #727: Cache Doctor Warm-Cache Readiness Evidence

- Branch: `codex/maxwell-cache-warm-readiness`.
- Scope: adjacent cache/warmup diagnostics.
- Result: `passivbot tool cache-integrity-doctor` now adds report-only
  `warm_cache_readiness` summaries derived from already-scanned candle, fill,
  and HSL/risk cache metadata. The readiness projection is explicitly
  non-enforcing and does not change startup or trading behavior.
- Review evidence: Claude and Hermes approved head `155e3640`; CI was green;
  focused cache-doctor tests, compileall, `git diff --check`, and
  touched-file silent-handling audit passed before merge. A parent-side
  temporary-worktree validation also passed the focused test/check set.

### PR #723: Remote-Call Debug Profile

- Branch: `codex/v8-remote-call-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `remote_calls` debug-profile enrichment for candle remote-fetch
  and authoritative state-fetch events. Enrichment is bounded to key shape,
  selected timing/correlation fields, status/surface/kind, and counts; default
  events remain unchanged, console output is unchanged, and no raw payloads are
  added.
- Review evidence: Hermes approved head `e78d79b0`; CI was green; focused
  remote-call profile tests, the broader live-event/monitor suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed before
  merge. Claude did not return during the merge window, so this opt-in
  observability slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `09ae3773` without bot restart because the
  profile is opt-in and no VPS config enabled it. The same 10-minute brief
  smoke reported no hard failures, all expected bots matched, and zero failed
  remote/account-critical calls. Remaining attention came from known non-hard
  EMA/staged-readiness and HSL cooldown events.

### PR #724: Candle Debug Profile

- Branch: `codex/v8-candle-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `candles` debug-profile enrichment to existing
  `candle.tail_projected` and `candle.coverage_checked` events, exposing bounded
  key-shape, timeframe/window, and missing-coverage counters without raw candle
  arrays or default console/log changes.
- Review evidence: Hermes approved head `4454fd09`; CI was green; focused
  candle debug-profile tests, the broader live-event/monitor suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed before
  merge. Claude did not return before merge, so this opt-in observability slice
  used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `f3969dbc` without bot restart because the
  profile is opt-in and no VPS config enabled it. A 10-minute brief smoke
  reported `ok=true`, no hard failures, all five expected bots matched, clean
  tracked repository state, no failed remote calls, and no failed
  account-critical remote calls.

### PR #726: Reviewer Follow-Ups

- Branch: `codex/v8-reviewer-followups`.
- Scope: Claude retrospective follow-up for already-merged low-risk
  observability/tooling slices.
- Result: redacted shareable live-ops path fields consistently, removed
  shutdown event message echo from smoke summaries, scopes EMA debug enrichment
  to the `ema` profile only, and keeps Rust debug sample construction
  best-effort inside the event-emitter path.
- Review evidence: Claude and Hermes approved head `8f4465a9`, CI was green,
  and local focused tests plus the broader live-event/monitor/smoke/preflight
  suite, compileall, `git diff --check`, and touched-file silent-handling audit
  passed before merge.
- VPS5 evidence: deployed to VPS5 at `41961266` without bot restart because the
  slice is observability/tooling-only. A 10-minute brief smoke reported
  `ok=true`, no hard failures, all five expected bots matched, clean tracked
  repository state, no failed remote calls, and no failed account-critical
  remote calls. Remaining attention came from known non-hard EMA/staged
  readiness and HSL status/cooldown events.

### PR #728: Fills Debug Profile

- Branch: `codex/v8-fills-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `fills` debug-profile enrichment to existing
  `fills.refresh_summary` and `fill.ingested` events with bounded coverage,
  count, and key-shape metadata. Default events and console output stay
  unchanged, and no raw fill IDs, source IDs, or payload values are emitted.
- Review evidence: Claude and Hermes approved the original head `4c58a718`;
  after PR #727 merged, the branch was rebased to `8ba083f6`, CI was green, and
  both reviewers confirmed the rebased patch was unchanged. Local focused tests,
  the broader live-event/monitor suite, compileall, and `git diff --check`
  passed before merge.
- VPS5 evidence: deployed to VPS5 at `5714d36d` without bot restart because the
  slice is opt-in and no VPS config enabled it. The first 10-minute smoke was
  red only because text logs contained a fresh OKX ccxt-pro websocket callback
  traceback after a reconnect; structured monitor events had no hard failures,
  all five expected bots matched, the repository was clean, and remote-call /
  account-critical failures were zero. A settled 2-minute follow-up smoke
  reported `ok=true`, no hard failures, no log hard matches, all five expected
  bots matched, clean tracked repository state, no failed remote calls, and no
  failed account-critical remote calls.

### PR #730: HSL Debug Profile

- Branch: `codex/v8-hsl-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `hsl` debug-profile enrichment to existing HSL
  status, transition, replay, red-trigger, and cooldown events with bounded
  event key, metric key, and HSL latch/cooldown state-shape metadata. Default
  HSL events and console output remain unchanged.
- Review evidence: Hermes approved head `7cfb80ac`, CI was green, and local
  focused tests, the broader HSL/live-event suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed. Claude did
  not return before merge, so this opt-in observability slice used the degraded
  gate.
- VPS5 evidence: deployed together with PRs #731 and #732 at `9bc2c37f`.
  Settled 5-minute smoke returned `ok=true`, no hard failures, no log hard
  matches, all five expected bots matched, clean tracked repository state, no
  failed remote calls, no failed account-critical remote calls, and no monitor
  errors or warnings. Remaining attention was non-hard Hyperliquid EMA/candle
  readiness.

### PR #732: Execution Debug Profile

- Branch: `codex/v8-execution-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `execution` debug-profile enrichment to existing
  order-wave, order-write, create-filter, and confirmation events with bounded
  key-shape/counter metadata. Default execution event payloads and console
  output remain unchanged, and raw order payload values are not added.
- Review evidence: Hermes found and then approved the raw-order-value leak fix
  at `cb1f2c23`; after rebasing onto current `v8`, CI was green and focused
  execution/live-event tests, compileall, and `git diff --check` passed. Claude
  did not return before merge, so this opt-in observability slice used the
  degraded gate.
- VPS5 evidence: deployed together with PRs #730 and #731 at `9bc2c37f`.
  Settled 5-minute smoke returned `ok=true`, no hard failures, no log hard
  matches, all five expected bots matched, clean tracked repository state, no
  failed remote calls, no failed account-critical remote calls, and no monitor
  errors or warnings.

### PR #734: Retrospective Tool Hygiene

- Branch: `codex/v8-retro-tool-hygiene`.
- Scope: Claude retrospective follow-up for already-merged operator tooling.
- Result: collapsed user/deploy prefixes in shareable restart/preflight/HSL
  preview output, resolved grouped and flat bot-side config keys in preflight
  reports, and kept HSL startup-preview event data scalar and allowlisted.
- Review evidence: Claude and Hermes approved; CI was green; focused
  preflight/HSL preview/restart-plan tests, compileall, `git diff --check`, and
  touched-file audits passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  operator tooling. Existing bots were left running.

### PR #735: Startup Budget Smoke Projections

- Branch: `codex/maxwell-startup-budget-smoke`.
- Scope: adjacent startup/warmup observability.
- Result: `live-smoke-report` now projects existing `bot.startup_timing` events
  into report-only elapsed and per-phase budget evidence using recent local p95
  baselines. The slice does not enforce budgets or change startup behavior.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed without bot restart. A settled follow-up smoke
  reported all five expected bots running, no hard failures, no log hard
  matches, no failed remote calls, and no failed account-critical calls.

### PR #736: EMA Readiness Handoff Backlog

- Branch: `codex/v8-backlog-ema-readiness-race`.
- Scope: living backlog / progress tracking.
- Result: recorded the OKX/AAVE active-symbol forager EMA readiness handoff gap
  as live-ops backlog item #17 and noted that the first remediation must
  preserve the no-fabricated-ranking-values contract.
- Review evidence: Claude and Hermes approved; CI was green before merge.
- VPS5 evidence: pulled to VPS5 without bot restart; all five configured bots
  remained running.

### PR #737: Active Forager EMA Carry-Forward

- Branch: `codex/v8-forager-active-ema-projection`.
- Scope: live forager EMA readiness hardening for active/normal symbols.
- Result: active/normal forager symbols may use bounded cached real-candle
  quote-volume and log-range EMA values when the current EMA read is transiently
  unavailable, while candidate-only symbols still become unavailable and
  active/normal symbols without cached values still fail loudly. The fix keeps
  open-tail projection valid for close EMA readiness only in forager mode, so
  qv/log-range ranking inputs do not come from projected open-tail values.
- Review evidence: Hermes first found that cached metric fallback and open-tail
  projection shared one eligibility map; fixed head `6965783e` split cached
  metric and projection eligibility and added a regression covering projected
  qv/log-range values differing from cached real-candle values. Claude and
  Hermes approved the fixed head; CI was green; focused EMA tests, compileall,
  `git diff --check`, and a diff-only silent-handling audit passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `e3429ee9` with a bot restart. Settled
  2-minute and 5-minute smokes reported all five expected bots running, no hard
  failures, no log hard matches, no failed remote calls, no failed
  account-critical remote calls, and clean tracked repository state.

### PR #739: Restart Plan Process Signal Safety

- Branch: `codex/v8-restart-plan-process-safety`.
- Scope: adjacent operator restart/smoke planning.
- Result: `passivbot tool live-restart-smoke-plan` now includes a
  `process_signal_safety` contract that warns future restart automation away
  from broad `pkill -f` / `pgrep -f` live-bot process matches and toward exact
  tmux panes or exact canonical process rows. The plan remains read-only and
  execution-unavailable.
- Review evidence: Claude and Hermes approved; CI was green; focused
  restart-plan tests, compileall, `git diff --check`, and the diff-only
  touched-file silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is plan-only
  operator tooling. A settled 5-minute smoke at `87e53dac` reported all five
  expected bots running, no hard failures, no log hard matches, no failed
  remote calls, no failed account-critical remote calls, and clean tracked
  repository state.

## Current Next Steps

1. Continue Phase 5/6 by adding the next high-value event producer or debug
   profile slice without increasing default console noise. Likely candidates
   are more order/risk transition coverage or focused profile refinements only
   when live diagnostics need deeper evidence.
2. Continue active read-only exchange health probes beyond account-critical
   basics. PR #701 added account-critical health summaries and PR #703 added
   `--account-only` plus symbol fallback for open-orders. Remaining useful
   slices include clock skew, rate-limit behavior, fill pagination coverage, and
   candle freshness probes.
3. Use the persistent non-hard EMA readiness / staged-execution degradation
   visible in VPS5 smokes as the next candidate for targeted readiness
   diagnostics or a narrow fix. PRs #679 and #682 made the problem groups easier
   to inspect by surfacing bounded latest event data and aggregate groups.
4. Start the live restart/smoke automation slice if operational workflow speed
   becomes the higher leverage next step.
5. Continue cache-doctor refinements in separate adjacent PRs: deeper metadata
   compatibility checks and synthetic/no-trade assumptions.
