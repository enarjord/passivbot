# Live Python Module Split Plan

## Purpose

Refactor the Python live-bot control plane into focused `src/live/` modules without
changing trading behavior.

This is structural work. The goal is to make live account refresh, planning
freshness, market snapshot handling, reconciliation, and execution guardrails easier
to review and test. Runtime behavior, log semantics, exchange hooks, and Rust
orchestrator inputs must remain equivalent after each extraction slice.

## Current Problem

`src/passivbot.py` still owns too many live-bot concerns in one large class:

- staged account-state refresh
- freshness ledger gating
- planning snapshot validation
- market snapshot acquisition and pre-create freshness checks
- actual-vs-ideal order reconciliation
- duplicate-order and foreign-writer guardrails
- order execution batching and throttles
- runtime timing, health, and shutdown helpers

That makes live-bot changes hard to review because unrelated safety contracts are
interleaved. The module split should make each contract explicit and locally
testable while keeping `Passivbot` as the compatibility facade during the refactor.

## Non-Negotiables

- [x] Preserve live trading behavior after every commit.
- [x] Preserve Rust as the source of truth for entries, closes, risk, unstuck, and
      order calculation.
- [x] Keep Python changes limited to orchestration, exchange I/O plumbing,
      freshness gates, reconciliation, and execution safety.
- [x] Keep exchange-specific overrides in `src/exchanges/*`.
- [x] Preserve existing `Passivbot` method names as wrappers until all call sites
      and exchange overrides are audited.
- [x] Preserve current log messages unless a change is explicitly scoped as
      diagnostics cleanup and covered by tests.
- [x] Do not add local runtime state that changes decisions and cannot be
      reconstructed from exchange state plus config.
- [x] Do not introduce silent fallbacks in trading-critical paths.
- [x] Keep each extraction commit mechanical and reviewable.

## Preferred Shape

Create a `src/live/` package and move generic live orchestration there over several
small commits.

Initial module targets:

- `src/live/freshness.py`
- `src/live/market_snapshot.py`
- `src/live/planning_snapshot.py`
- `src/live/state_refresh.py`
- `src/live/planning_gates.py`
- `src/live/market_data.py`
- `src/live/reconciler.py`
- `src/live/executor.py`
- `src/live/runtime.py` only if useful after the higher-value modules are split

During extraction, prefer module-level functions that accept `bot` over new stateful
classes. This keeps the first pass close to the current `Passivbot` method bodies.
Introduce new abstractions only when the extracted module exposes real duplication
or an explicit contract boundary.

## Branch And Workflow

- [x] Branch from current `origin/master`.
- [x] Use branch: `codex/live-python-module-split-plan`.
- [ ] Keep implementation branches based on `origin/master` or this plan branch,
      depending on whether the plan has already been merged.
- [ ] Use a separate commit for each extraction phase.
- [ ] After each commit, run the targeted validation for that phase.
- [ ] Keep the checklist in this document current as implementation proceeds.

Goal mode is useful for the implementation pass if the agent is expected to keep
working across phases without stopping after the first successful extraction. If
using goal mode, set the objective to complete one phase at a time, not the entire
module split in one uninterrupted run.

## Phase 0: Baseline And Drift Audit

Purpose: establish a known behavior baseline and remove obvious config-default drift
before moving code.

- [x] Confirm branch and base:

```bash
git branch --show-current
git rev-parse HEAD origin/master
git status --short
```

- [x] Run hygiene checks:

```bash
git diff --check
./venv/bin/python -m py_compile src/passivbot.py src/freshness_ledger.py src/market_snapshot.py src/planning_snapshot.py
```

- [x] Run baseline focused tests:

```bash
./venv/bin/python -m pytest tests/test_freshness_ledger.py tests/test_market_snapshot.py -q
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -q
./venv/bin/python -m pytest tests/test_foreign_passivbot_detection.py tests/test_order_orchestration.py -q
```

- [x] Audit staged-live runtime defaults against schema/prepared config:

```bash
rg -n "get_optional_(config|live)_value|live_value\\(|setdefault\\(" src/passivbot.py src/config tests
rg -n "market_snapshot_ticker_strategy|max_forager_candle|fills_recent_overlap|fills_confirmation_overlap|staged_refresh|forager_score_hysteresis|initial_entry_exec|recv_window" src passivbot-rust tests docs
```

- [ ] If the audit finds a clear duplicate default that can drift from schema,
      fix it with focused tests before extraction.
- [x] If no actionable drift is found, mark the staged-live plan item as audited.

Exit criteria:

- [x] Baseline tests pass or known failures are recorded.
- [x] No tracked working-tree changes outside the plan/audit scope.
- [x] Any default cleanup is committed separately from module movement.

## Phase 1: Create `src/live/` And Contract Module Imports

Purpose: establish package shape with minimal behavior risk.

Recommended first pass:

- [x] Create `src/live/__init__.py`.
- [x] Move or re-export `FreshnessLedger`, `ACCOUNT_SURFACES`, and
      `LIVE_STATE_SURFACES` through `src/live/freshness.py`.
- [x] Move or re-export `MarketSnapshot` and `MarketSnapshotProvider` through
      `src/live/market_snapshot.py`.
- [x] Move or re-export `PlanningSnapshot` and related dataclasses through
      `src/live/planning_snapshot.py`.
- [x] Keep root-level compatibility shims for existing imports:
      `freshness_ledger.py`, `market_snapshot.py`, and `planning_snapshot.py`.
- [x] Update new or touched code to import from `live.*`.
- [x] Avoid broad import churn in tests unless needed.

Suggested compatibility pattern:

```python
from live.freshness import ACCOUNT_SURFACES, LIVE_STATE_SURFACES, FreshnessLedger
```

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/*.py src/freshness_ledger.py src/market_snapshot.py src/planning_snapshot.py
./venv/bin/python -m pytest tests/test_freshness_ledger.py tests/test_market_snapshot.py -q
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "planning_snapshot or market_snapshot or freshness" -q
```

Exit criteria:

- [x] Existing root imports still work.
- [x] New `live.*` imports work.
- [x] No behavior-bearing code was changed beyond import paths.

## Phase 2: Extract Staged Account Refresh

Purpose: isolate the staged authoritative account-state pipeline, currently the
highest-value extraction target.

Create:

- [x] `src/live/state_refresh.py`

Move behavior-preserving implementations behind `Passivbot` wrappers:

- [x] `refresh_authoritative_state`
- [x] `_refresh_authoritative_state_staged`
- [x] `_capture_balance_staged_snapshot`
- [x] `_capture_positions_staged_snapshot`
- [x] `_authoritative_staged_refresh_plan`
- [x] `_staged_fills_refresh_due`
- [x] `_staged_fills_can_prefetch_routine`
- [x] `_schedule_routine_fill_refresh_prefetch`
- [x] `_routine_fill_refresh_prefetch_task`
- [x] `_timed_authoritative_fetch`
- [x] `_log_staged_refresh_timings`
- [x] `_record_staged_refresh_timing_summary`
- [x] `_log_staged_refresh_progress_until`
- [x] `_fetch_authoritative_state_staged_snapshot`

Keep these as `Passivbot` hooks or wrappers because exchanges may override or rely
on method lookup:

- [x] `capture_authoritative_state_staged_snapshot`
- [x] `_staged_defer_balance_publication`
- [x] `_reconcile_balance_after_staged_refresh`
- [x] `_staged_balance_update_source`

Implementation notes:

- [x] Preserve exact return values and exception propagation.
- [x] Preserve task cancellation and `asyncio.gather(..., return_exceptions=True)`
      cleanup behavior.
- [x] Preserve log text and log levels.
- [x] Preserve `_authoritative_refresh_plan_surfaces`,
      `_authoritative_pending_confirmations`, and freshness ledger side effects.
- [x] Keep all exchange-specific staged snapshot overrides in exchange modules.

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/state_refresh.py
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "refresh_authoritative_state_staged or staged_refresh_plan or staged_account_refresh or authoritative_barrier" -q
./venv/bin/python -m pytest tests/ccxt_upgrade/test_order_contracts.py -k "staged_snapshot or capture_balance_snapshot" -q
```

Exit criteria:

- [x] Generic staged refresh request-count tests still pass.
- [x] Exchange staged snapshot tests still pass.
- [x] `Passivbot.refresh_authoritative_state()` remains the external entrypoint.
- [x] No exchange subclass behavior changed.

## Phase 3: Extract Planning Freshness And Snapshot Gates

Purpose: isolate the contract that decides whether live state is safe to hand to
Rust planning and later order creation.

Create:

- [x] `src/live/planning_gates.py`

Move behavior-preserving implementations behind wrappers:

- [x] `_staged_planner_required_surfaces`
- [x] `_staged_planner_surface_min_epochs`
- [x] `_staged_planner_precondition_state`
- [x] `_assert_staged_planner_preconditions`
- [x] `_format_staged_execution_defer_message`
- [x] `_log_staged_execution_defer`
- [x] `_defer_staged_execution_cycle`, only if it does not drag too much runtime
      loop behavior into the module
- [x] `_build_staged_planning_snapshot`
- [x] `_current_planning_snapshot_invalid_for_creations`

Implementation notes:

- [x] Do not move candle-manager internals in this phase.
- [x] Keep completed-candle signature helpers as bot-provided dependencies unless
      they become clearly separable.
- [x] Preserve planning snapshot invalid-detail structure.
- [x] Preserve all precondition failure messages and deferral semantics.

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/planning_gates.py
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "staged_planner_preconditions or planning_snapshot or pre_create_snapshot" -q
./venv/bin/python -m pytest tests/test_unstucking_safeguards.py -k "staged_orchestrator" -q
```

Exit criteria:

- [x] Rust planning is still blocked on the same stale/missing surfaces.
- [x] Pre-create planning snapshot gates still reject stale or missing state.
- [x] Deferral remains non-fatal where tests expect non-fatal staged delays.

## Phase 4: Extract Live Market Data Helpers

Purpose: isolate current bid/ask/last snapshot handling and keep completed candles
out of live price truth except for explicit, documented exceptions.

Create:

- [x] `src/live/market_data.py`

Move behavior-preserving implementations behind wrappers:

- [x] `_market_snapshot_ticker_strategy`
- [x] `_live_market_snapshot_max_age_ms`
- [x] `_get_live_market_snapshots`
- [x] `_get_orchestrator_market_snapshots`
- [x] `_record_market_snapshot_surface`
- [x] `_market_snapshot_signature`
- [x] `_market_snapshot_signature_invalid`
- [x] `_filter_fresh_market_snapshot_creations`, if this fits better here than
      executor after Phase 3

Implementation notes:

- [x] Keep Hyperliquid-specific `allMids` and HIP-3 behavior exchange-scoped.
- [x] Keep Bitget/KuCoin strategy selection behavior unchanged.
- [x] Preserve warnings for failed or stale pre-create market snapshot refresh.
- [x] Preserve cache sink behavior into candlestick manager current-price cache.

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/market_data.py src/live/market_snapshot.py
./venv/bin/python -m pytest tests/test_market_snapshot.py tests/test_hyperliquid_tickers.py tests/test_kucoin_exchange_config.py -q
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "market_snapshot or pre_create_snapshot or orchestrator_market" -q
```

Exit criteria:

- [x] Generic exchanges still reject incomplete ticker snapshots loudly.
- [x] Hyperliquid explicit snapshot exceptions remain labeled and covered.
- [x] Pre-create market snapshot refresh behavior is unchanged.

## Phase 5: Extract Reconciler And Writer Guardrails

Purpose: isolate actual-vs-ideal reconciliation helpers, recent local order
tracking, disappeared self-order guardrails, and foreign Passivbot writer detection.

Create:

- [x] `src/live/reconciler.py`

First extraction slice:

- [x] `add_to_recent_order_cancellations`
- [x] `order_was_recently_cancelled`
- [x] `order_matches_bot_cancellation`
- [x] `add_to_recent_order_executions`
- [x] `order_matches_recent_execution`
- [x] `_local_order_open_orders_confirmed`
- [x] `order_was_recently_updated`

Second extraction slice:

- [x] `_extract_order_custom_id`
- [x] `_extract_order_exchange_id`
- [x] `_canonical_passivbot_custom_id`
- [x] `_extract_order_reduce_only`
- [x] `_extract_order_float`
- [x] `_order_identity_fingerprint`
- [x] `_build_emitted_order_record`
- [x] `_emitted_order_records`
- [x] `_prune_emitted_order_custom_ids`
- [x] `_prune_foreign_passivbot_seen`
- [x] `_record_emitted_order_custom_id`
- [x] `_foreign_passivbot_detection_key`
- [x] `_order_matches_recent_emitted_record`
- [x] `_stop_for_foreign_passivbot_orders`
- [x] `_detect_foreign_passivbot_orders`

Later extraction slice, only after the identity code is stable:

- [x] actual-vs-ideal order diffing helpers
- [x] disappeared self-order state-change guardrails
- [x] duplicate-order checks

Implementation notes:

- [x] Keep detection thresholds and lookback windows unchanged.
- [x] Preserve exact custom-id canonicalization behavior.
- [x] Preserve stop behavior for repeated foreign writer detections.
- [x] Preserve symbol dirtying and account-critical refresh requests.
- [x] Do not broaden or weaken duplicate-order prevention.

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/reconciler.py
./venv/bin/python -m pytest tests/test_foreign_passivbot_detection.py -q
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "disappeared or duplicate or recent_order or emitted_order or foreign or state_change" -q
```

Exit criteria:

- [x] Foreign-writer detection still adopts this bot's emitted orders.
- [x] Ambiguous create errors are still remembered.
- [x] Disappeared self-orders still block creations and request fresh account state.
- [x] Recent execution/cancellation throttles are unchanged.

## Phase 6: Extract Executor Helpers

Purpose: isolate generic order execution orchestration after reconciliation and
market freshness gates are separated.

Create:

- [x] `src/live/executor.py`

Move behavior-preserving implementations behind wrappers:

- [x] `execute_to_exchange`
- [x] `execute_orders_parent`, if exchange subclasses do not override it in a way
      that makes wrapper lookup risky
- [x] `execute_cancellations_parent`, under the same condition
- [x] generic order batch-cap helpers
- [x] order wave bookkeeping only if it remains behavior-preserving and does not
      belong in a later diagnostics module

Do not move exchange-specific methods:

- [x] `execute_orders`
- [x] `execute_cancellations`
- [x] exchange config mutation methods
- [x] broker-code payload/header injection

Implementation notes:

- [x] Preserve debug-mode return behavior.
- [x] Preserve balance-threshold create skip behavior.
- [x] Preserve order config update gate before creation.
- [x] Preserve pre-create market snapshot filter ordering.
- [x] Preserve restart exception propagation and error-count handling.

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/executor.py
./venv/bin/python -m pytest tests/test_order_orchestration.py tests/test_exchange_config_updates.py -q
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "execute_to_exchange or execute_orders_parent or pre_create or order_wave" -q
```

Exit criteria:

- [x] Order create/cancel batch behavior is unchanged.
- [x] Exchange config gating still runs before order creation.
- [x] Market freshness gates still run before order creation.
- [x] Restart/error-count behavior is unchanged.

## Phase 7: Extract Runtime Utilities If Still Worth It

Purpose: reduce remaining `src/passivbot.py` size only where the boundary is clear.

Create only if useful:

- [ ] `src/live/runtime.py`

Potential candidates:

- [ ] shutdown helper wrappers
- [ ] health timing helpers
- [ ] silence watchdog context helpers
- [ ] maintainer lifecycle helpers
- [ ] startup timing utilities

Implementation notes:

- [ ] Do not create a dumping ground module.
- [x] Skip this phase if extraction would obscure runtime loop flow.
- [ ] Keep signal handling and CLI entrypoint behavior easy to trace.

Validation:

```bash
./venv/bin/python -m py_compile src/passivbot.py src/live/runtime.py
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -k "shutdown or health or execution_loop or watchdog" -q
```

Exit criteria:

- [ ] Runtime loop flow remains easier to read than before extraction.
- [ ] Shutdown and health behavior are unchanged.

## Optional Fake-Live Validation

Use fake-live as a parity check after state refresh, reconciler, or executor phases,
especially if the implementation touched request-count behavior or execution-loop
ordering.

- [x] Run targeted fake-live tests:

```bash
./venv/bin/python -m pytest tests/test_run_fake_live.py -m fake_live -q
```

- [x] If adding staged request-count fake-live scenarios, keep them scenario-driven
      and assert durable counts/ordering rather than incidental log formatting.
- [x] Record scenario names and key request-count expectations in this document.

## Full Final Validation

Run when all intended extraction phases are complete:

```bash
git diff --check
./venv/bin/python -m py_compile src/passivbot.py src/live/*.py
./venv/bin/python -m pytest tests/test_freshness_ledger.py tests/test_market_snapshot.py -q
./venv/bin/python -m pytest tests/test_foreign_passivbot_detection.py tests/test_order_orchestration.py tests/test_exchange_config_updates.py -q
./venv/bin/python -m pytest tests/test_passivbot_balance_split.py -q
./venv/bin/python -m pytest tests/test_unstucking_safeguards.py -q
```

If Rust-facing payloads or PyO3 wiring are touched unexpectedly, stop and reassess.
This plan should not require Rust changes.

## Review Checklist

Use this checklist for every extraction PR or commit series:

- [x] Is the diff mostly moved code plus wrappers?
- [x] Are public `Passivbot` method names still available?
- [x] Are exchange overrides still resolved through method lookup?
- [x] Did any log text or level change?
- [x] Did any exception become swallowed or downgraded?
- [x] Did any required trading input gain a neutral default fallback?
- [x] Did any state freshness gate become looser?
- [x] Did any duplicate-order or foreign-writer guardrail become weaker?
- [x] Did targeted tests cover the moved surface?
- [x] Did the commit avoid unrelated cleanup?

## Expected Payoff

After this refactor, live-bot changes should be easier to make and review because:

- account refresh behavior is localized
- planning freshness rules are explicit
- market snapshot truth is separate from candle history
- duplicate-writer and stale-state guardrails are isolated
- execution ordering is easier to audit
- future diagnostics work can attach to domain events instead of scattered log
  strings inside `src/passivbot.py`
