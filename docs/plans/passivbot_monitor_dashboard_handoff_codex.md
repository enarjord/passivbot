# Passivbot Monitor Dashboard Handoff

## Purpose

This file is a handoff for another Codex instance to continue the monitor/dashboard work in a cleaner worktree.

The actual implementation work has **not** started yet in this worktree.

What exists so far:

1. a concrete design/spec in `docs/plans/passivbot_monitor_dashboard.md`
2. exploration of the main integration seams in `src/passivbot.py` and `src/config_utils.py`
3. agreement with the user on the initial architecture and scope

## Branch / Context

- Current branch when this handoff was written: `feature/equity-hard-stop-loss_codex`
- Recent relevant commits:
  - `3879b364` update template config
  - `552128da` Pace exchange config updates and retry per symbol
  - `1a733cc5` Add fake-live coverage for HSL cooldown policies

## Important Worktree Warning

This worktree is dirty and contains many unrelated local/untracked files.

Known unrelated noise includes:

1. `configs/hsl.json` modified
2. `configs/template.json` modified
3. many untracked docs, notebooks, logs, helper scripts, and local binaries

The next Codex instance should work in a clean worktree if possible.

## What Was Decided With The User

### High-level architecture

1. Dashboard is a **separate process**
2. Bot publishes monitor data to disk
3. Dashboard reads only from that monitor data root
4. No direct shared-memory access to the bot instance
5. TUI first, web GUI later if desired

### Data publication model

The bot should publish:

1. `state.latest.json`
   - full current state only
   - no historical series
2. structured append-only event stream
3. selected history streams under the same monitor root

### Scope rule

If the live bot uses a datum in any way, it is eligible for publication.

Examples:

1. current EMAs used by logic: yes
2. full historical candle arrays inside the current snapshot: no
3. completed 1m and 1h candles as events/history streams: yes
4. price tick updates used by bot: yes
5. structured errors of any kind: yes

### Event schema direction

Use structured machine-readable event kinds like:

- `hsl.transition`
- `market.tick`
- `order.filled`

Also include human-friendly `tags`.

Recommended event envelope from the spec:

```json
{
  "ts": 1774012131634,
  "seq": 12345,
  "kind": "hsl.transition",
  "tags": ["hsl", "risk", "transition"],
  "exchange": "bybit",
  "user": "bybit_01",
  "symbol": "XMR/USDT:USDT",
  "pside": "long",
  "payload": {}
}
```

### Fill event decision

For the monitor event stream:

1. publish canonical normalized fill payloads by default
2. do **not** include raw CCXT payloads by default

Reason:

1. raw fill payloads already exist in `caches/fill_events/`
2. default raw inclusion would create unnecessary bloat

A future optional config may enable raw payload emission.

### Price tick decision

Publish tick events richly, but:

1. only when actual values change
2. only last/bid/ask/mid
3. no order book depth
4. throttle to approximately one event per symbol per `500ms`

### History loading decision

Dashboard should use only data emitted from bot startup onward by default.

Do not make historical backfill part of the MVP.

A future dashboard action like "load historical data" is acceptable later.

### Centralization rule

Do **not** litter `passivbot.py` with file writes.

Use a dedicated module, tentatively:

- `src/monitor_publisher.py`

The publisher should own:

1. manifest
2. atomic current snapshot
3. event append
4. rotation/retention
5. structured error capture

## Existing Spec

The current detailed design is in:

- `docs/plans/passivbot_monitor_dashboard.md`

That spec already contains:

1. monitor root layout
2. snapshot sections
3. event kinds
4. history stream recommendations
5. retention config ideas
6. phased rollout

## Exploration Already Done

These were inspected and identified as likely implementation seams:

### Config

`src/config_utils.py`

- template config currently has `live` and `logging` defaults around the `memory_snapshot_interval_minutes` area
- natural place to add `monitor.*` defaults and validation

### Passivbot init/runtime seams

`src/passivbot.py`

Main areas already identified:

1. `__init__`
   - good place to initialize monitor config and publisher
   - near existing logging/health config
2. startup sequence
   - `"[boot] starting execution loop..."`
   - `"[boot] READY - Bot initialization complete..."`
   - good places for `bot.start` / `bot.ready`
3. `run_execution_loop()`
   - good place for periodic snapshot flush
4. `execution_cycle()`
   - good place to rebuild current-state sections
5. `_log_mode_changes()`
   - good place for `mode.changed`
6. `_equity_hard_stop_log_transition()` and related HSL methods
   - good places for `hsl.transition`, `hsl.cooldown_*`, etc.
7. `_snapshot_actual_orders()`
   - useful existing normalization helper
8. `_log_memory_snapshot()` and health summary
   - useful source for monitor health state

### Existing hooks worth reusing

1. `debug_dump_bot_state_to_disk()` exists but is too raw and ad hoc to be the public monitor contract
2. `calc_ideal_orders_orchestrator(... return_snapshot=True)` has some existing structured snapshot logic that may be reusable conceptually, but the monitor snapshot should not be constrained by orchestrator payload shape
3. `FillEventsManager` already has canonical fill semantics and should remain the source of truth for normalized fill payloads

## Recommended Phase 1 Scope

Do **not** try to implement the full dashboard UI yet.

Phase 1 should only build the bot-side publisher and minimal monitor output.

### Deliverables

1. `monitor.enabled` config surface
2. `src/monitor_publisher.py`
3. `state.latest.json`
4. `events/current.ndjson`
5. minimal event types
6. minimal snapshot sections
7. tests

### Minimal snapshot sections for Phase 1

1. `meta`
2. `account`
3. `health`
4. `positions`
5. `open_orders`
6. `modes`
7. `hsl`

### Minimal event kinds for Phase 1

1. `bot.start`
2. `bot.ready`
3. `bot.stop`
4. `health.summary`
5. `error.bot`
6. `error.exchange`
7. `account.balance`
8. `position.changed`
9. `order.opened`
10. `order.canceled`
11. `order.filled`
12. `mode.changed`
13. `hsl.transition`
14. `hsl.cooldown_started`
15. `hsl.cooldown_ended`

Price ticks and completed candles can be added in the next increment once the publication contract is stable.

## Recommended File-by-File Implementation Order

### 1. `src/config_utils.py`

Add `monitor` defaults, likely something like:

```json
"monitor": {
  "enabled": false,
  "root_dir": "monitor",
  "snapshot_interval_seconds": 1.0,
  "checkpoint_interval_minutes": 10.0,
  "event_rotation_mb": 128,
  "event_rotation_minutes": 60,
  "retain_days": 7,
  "max_total_bytes": 1073741824,
  "retain_price_ticks": true,
  "retain_candles": true,
  "retain_fills": true,
  "compress_rotated_segments": true,
  "price_tick_min_interval_ms": 500,
  "emit_completed_candles": true,
  "include_raw_fill_payloads": false
}
```

Do validation centrally here.

### 2. `src/monitor_publisher.py`

Implement:

1. `MonitorPublisher`
2. manifest writing
3. atomic `state.latest.json`
4. append event method
5. `seq` counter
6. rotation/retention stubs or basic implementation
7. `record_error()` helper

Strong recommendation:

- add a small logging handler class here for structured `error.*` events

### 3. `src/passivbot.py`

Integrate the publisher with minimal intrusion:

1. initialize in `__init__` if `monitor.enabled`
2. emit lifecycle events at startup/shutdown
3. build a minimal monitor snapshot helper, for example:
   - `_build_monitor_snapshot()`
4. call publisher snapshot flush from the main loop
5. send mode/HSL events from existing narrow hooks

### 4. Tests

Add new tests for:

1. publisher file creation
2. atomic latest snapshot update
3. event append and `seq` monotonicity
4. minimal `Passivbot` integration with fake/test doubles
5. structured error event capture

## Good Initial Testing Strategy

1. unit tests for `monitor_publisher.py`
2. light integration tests using fake `Passivbot` objects or monkeypatches
3. only after Phase 1 is stable, consider fake-live harness emission

Fake-live integration is explicitly worth doing later, but not required for the first publisher pass.

## Important Constraints

1. Rust is still source of truth for trading behavior
2. monitor publisher is read-only / observability-only
3. publisher failures should not alter trading logic
4. avoid writing arbitrary internal Python objects as monitor schema
5. prefer canonical normalized forms already used by bot logic

## Suggested First Concrete Steps For The Next Codex Instance

1. commit the spec doc if not already committed
2. in a clean worktree, add `monitor` config defaults in `src/config_utils.py`
3. create `src/monitor_publisher.py`
4. wire only startup + periodic snapshot + error events first
5. add tests before adding more event kinds

## Notes On This Worktree

At the time of handoff:

1. no dashboard/publisher code changes were applied in this worktree
2. only the spec doc exists for this effort
3. this worktree contains unrelated local dirt and should not be trusted as the implementation base
