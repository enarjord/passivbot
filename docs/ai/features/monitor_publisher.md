# Monitor Publisher

## Contract

1. The monitor publisher is observability-only and must not affect trading decisions.
2. `state.latest.json` is the current-truth snapshot and must be written atomically.
3. Event payloads should reuse canonical normalized bot data where possible instead of dumping raw Python internals.
4. Publisher failures should be contained and must not change order/risk behavior.

## Current Implementation Scope

Implemented now:

1. centralized `monitor.*` config defaults and validation in `src/config_utils.py`
2. `src/monitor_publisher.py` with manifest, event append, atomic snapshot writes, checkpoints, basic event rotation, and retention pruning
3. live `Passivbot` hooks for:
   - lifecycle: `bot.start`, `bot.ready`, `bot.stop`
   - health: `health.summary`
   - errors: `error.bot`, `error.exchange`
   - account: `account.balance`
   - positions: `position.changed`
   - orders/fills: `order.opened`, `order.canceled`, `order.filled`
   - modes: `mode.changed`
   - HSL: `hsl.transition`, `hsl.cooldown_started`, `hsl.cooldown_ended`
4. minimal snapshot sections:
   - `meta`
   - `account`
   - `health`
   - `positions`
   - `open_orders`
   - `modes`
   - `hsl`

## Non-Obvious Details

1. The config surface is intentionally larger than the current implementation. Some knobs are placeholders for the next increment and should not be removed casually.
2. Fill events should continue to use normalized fill semantics from the live fill path. Do not switch the monitor stream to raw CCXT payloads by default.
3. The publisher owns all monitor file writes. Do not add scattered ad hoc writes elsewhere in `passivbot.py`.
4. Snapshot cadence is best-effort from the execution loop. Exact wall-clock cadence is not guaranteed during restart/backoff paths.

## Gaps Still Open

1. No `history/` streams yet for fills, ticks, or candles.
2. No market tick or completed-candle publication yet.
3. No dedicated `error.publisher` self-reporting path yet.
4. Snapshot coverage is still intentionally narrow; market/forager/unstuck/exchange-config/recent are not published yet.
5. No consumer/dashboard code has been added in-repo.

## Test Focus

1. config formatting should preserve/add `monitor.*` defaults and reject invalid values
2. publisher writes should preserve atomic snapshot replacement and monotonic event `seq`
3. live integration tests should assert monitor events are emitted from narrow existing hooks
4. future additions that introduce fallback/error behavior need explicit regression tests

## Key Code

- `src/config_utils.py`
- `src/monitor_publisher.py`
- `src/passivbot.py`
- `tests/test_monitor_publisher.py`
- `tests/test_passivbot_monitor.py`
- `docs/plans/passivbot_monitor_dashboard.md`
