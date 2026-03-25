# Passivbot.py Refactor Plan

## Goal

Reduce [`src/passivbot.py`](/Users/eiriknarjord/repos/passivbot-3/src/passivbot.py) from a single 9.5k-line live-bot file into a smaller set of focused Python modules without changing trading behavior.

This is a structural refactor, not a behavior project.

## Non-Negotiables

1. Rust remains the source of truth for order behavior, risk, unstuck logic, and forager coin selection.
2. Refactors must preserve stateless live behavior after restart from exchange state + config.
3. Each extraction should be behavior-neutral, test-covered, and reviewable in isolation.
4. Avoid a giant “flag day” move. Split by subsystem and keep `Passivbot` as the composition root until the end.

## Current Shape

As of March 21, 2026:

1. [`src/passivbot.py`](/Users/eiriknarjord/repos/passivbot-3/src/passivbot.py) is `9513` lines.
2. It currently mixes:
   - bot startup and shutdown
   - exchange wiring and runtime loop control
   - equity hard stop state machine
   - monitor publisher integration
   - market/candle maintenance
   - fill/PnL refresh
   - forager payload building
   - Rust orchestrator input/output plumbing
   - order reconciliation and exchange execution
   - coin-list refresh and exchange-config refresh

That makes local reasoning, test targeting, and safe edits harder than they need to be.

## Refactor Strategy

Use a staged extraction approach:

1. extract self-contained method families into helper modules
2. keep `Passivbot` public API stable during each stage
3. move logic first, then optionally introduce helper classes only where state ownership becomes clearer
4. keep imports acyclic; helper modules should depend on `Passivbot` data, not on each other in tangled ways

The first passes should prefer plain module functions that take `self` over introducing many new classes too early.

## Target Module Map

Recommended destination layout under `src/`:

1. `passivbot.py`
   - keep only construction, top-level bot composition, and the highest-level runtime entrypoints
2. `passivbot_monitor.py`
   - monitor snapshot/event/history helpers
   - `_monitor_*`
   - `_build_monitor_*`
3. `passivbot_hsl.py`
   - full equity hard stop subsystem
   - `_equity_hard_stop_*`
4. `passivbot_runtime.py`
   - `start_bot()`
   - `run_execution_loop()`
   - `shutdown_gracefully()`
   - loop/watchdog/log-cycle coordination
5. `passivbot_exchange_state.py`
   - positions/balance/open-orders refresh
   - exchange-config refresh
   - tickers/time sync helpers
6. `passivbot_orders.py`
   - ideal-vs-actual reconciliation
   - order delta annotation
   - matching tolerances
   - create/cancel execution orchestration
7. `passivbot_orchestrator.py`
   - Rust payload building
   - EMA bundle loading
   - orchestrator mode overrides
   - executable-order finalization
8. `passivbot_forager.py`
   - candidate payload building
   - forager candle refresh
   - volume/log-range helpers
9. `passivbot_marketdata.py`
   - candle warmup
   - OHLCV maintenance
   - hourly/data-maintainer loops
10. `passivbot_coins.py`
   - approved/ignored coin list refresh
   - symbol/coin mapping helpers

This does not need to be created all at once. It is the target shape, not the first diff.

## Recommended Extraction Order

### Phase 0: Baseline Guardrails

Before moving code:

1. land a narrow smoke-test list for live-orchestration-adjacent behavior
2. identify the minimum targeted test command that must stay green after every phase
3. keep a running checklist of moved methods to avoid duplicate logic or dead imports

Recommended per-phase test floor:

```bash
pytest tests/test_passivbot_monitor.py
pytest tests/test_fill_events_manager.py
pytest tests/test_orchestrator_integration.py
pytest tests/test_orchestrator_json_api.py
pytest tests/test_passivbot_balance_split.py
```

Add narrower tests as each extracted module grows its own surface.

### Phase 1: Extract Monitor Helpers

First extraction target:

1. every `_monitor_*` helper
2. every `_build_monitor_*` helper

Why first:

1. already logically grouped
2. observability-only
3. low trading-risk surface
4. currently one of the cleanest subsystems in the file

Expected result:

1. new `src/passivbot_monitor.py`
2. `Passivbot` imports and binds monitor helpers
3. no behavior change

### Phase 2: Extract Equity Hard Stop

Second extraction target:

1. all `_equity_hard_stop_*`
2. `_hsl_*` helpers tightly coupled to that subsystem

Why second:

1. very large contiguous subsystem
2. mostly internally coherent
3. currently one of the biggest reasons the file is hard to navigate

Constraint:

1. keep the state fields on `Passivbot`
2. move logic before rethinking state containers

### Phase 3: Extract Orchestrator Payload + Forager Plumbing

Third extraction target:

1. `_load_orchestrator_ema_bundle()`
2. `calc_ideal_orders_orchestrator*()`
3. mode-override builders
4. `_to_executable_orders()`
5. `_finalize_reduce_only_orders()`
6. `build_forager_candidate_payload()`
7. `calc_volumes_and_log_ranges()`
8. forager refresh-budget helpers

Why here:

1. dense logic
2. highest-value isolation after HSL
3. easier to test once split

Constraint:

1. preserve exact Rust payload shape
2. do not change fallback/error behavior during extraction

### Phase 4: Extract Order Reconciliation + Exchange Execution

Target:

1. `calc_orders_to_cancel_and_create()`
2. reconciliation/matching helpers
3. `execute_to_exchange()`
4. create/cancel parent functions
5. order-action logging helpers
6. direct execute/cancel wrappers

Why:

1. this is another major mental chunk
2. it touches exchange side effects and deserves its own module boundary

Constraint:

1. preserve sequencing and error propagation
2. preserve logging and recent-order cache behavior

### Phase 5: Extract Exchange-State Refresh

Target:

1. positions/balance/open-orders updates
2. exchange-config refresh
3. ticker/time helpers
4. effective-min-cost refresh

Why:

1. these methods are operationally distinct from reconciliation and orchestrator code
2. easier to reuse in live-loop and tests when isolated

### Phase 6: Extract Market-Data Maintenance

Target:

1. candle warmup
2. required-index rebuild/audit helpers
3. active/inactive candle refresh loops
4. hourly/data-maintainer loops

Why:

1. candle/EMA infrastructure is already conceptually separate via `CandlestickManager`
2. Python-side orchestration around it should be grouped too

### Phase 7: Shrink `passivbot.py` to Composition Root

End state for [`src/passivbot.py`](/Users/eiriknarjord/repos/passivbot-3/src/passivbot.py):

1. constructor and core shared fields
2. top-level wiring
3. minimal high-level public entrypoints
4. imports of extracted helper modules

Target size after this phase:

1. ideally under `2500-3000` lines
2. with the largest remaining logic living in dedicated sibling modules

## Extraction Mechanics

Recommended technique per phase:

1. move a contiguous family of methods into a new module as top-level functions taking `self`
2. in `passivbot.py`, replace method bodies with imported function assignments or thin forwarding wrappers
3. once stable, optionally decide whether a helper class would improve state ownership

Pragmatic example:

```python
# src/passivbot_monitor.py
def _build_monitor_snapshot(self, *, now_ms=None):
    ...

# src/passivbot.py
from passivbot_monitor import _build_monitor_snapshot

class Passivbot:
    _build_monitor_snapshot = _build_monitor_snapshot
```

This keeps the runtime API stable while shrinking the main file quickly and safely.

## Risks To Avoid

1. introducing circular imports between helper modules
2. changing exception behavior while “just moving code”
3. accidentally moving Rust-owned trading behavior into new Python abstractions
4. over-object-orienting the split too early
5. mixing refactor and feature work in the same diff

## Review Rules

Each phase should satisfy:

1. one subsystem moved
2. no intentional behavior change
3. targeted tests green
4. no unrelated cleanup bundled in
5. follow-up commit only if the previous phase is stable

## Recommended Next Implementation Slice

Start with Phase 1:

1. create `src/passivbot_monitor.py`
2. move all `_monitor_*` and `_build_monitor_*` helpers there
3. keep `Passivbot` method names unchanged via imported bindings
4. run:

```bash
pytest tests/test_passivbot_monitor.py tests/test_monitor_publisher.py tests/test_monitor_relay.py tests/test_monitor_tui.py
```

This is the safest first cut and immediately removes a meaningful chunk from the main file.
