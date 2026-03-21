# Monitor Publisher

## Contract

1. The monitor publisher is observability-only and must not affect trading decisions.
2. `state.latest.json` is the current-truth snapshot and must be written atomically.
3. Event payloads should reuse canonical normalized bot data where possible instead of dumping raw Python internals.
4. Publisher failures should be contained and must not change order/risk behavior.

## Current Implementation Scope

Implemented now:

1. centralized `monitor.*` config defaults and validation in `src/config_utils.py`
2. `src/monitor_publisher.py` with manifest, event append, atomic snapshot writes, checkpoints, event/history rotation, retention pruning, fill history, throttled price ticks, and completed-candle history streams
3. live `Passivbot` hooks for:
   - lifecycle: `bot.start`, `bot.ready`, `bot.stop`
   - health: `health.summary`
   - errors: `error.bot`, `error.exchange`
   - account: `account.balance`
   - positions: `position.changed`
   - orders/fills: `order.opened`, `order.canceled`, `order.filled`
   - modes: `mode.changed`
   - HSL: `hsl.transition`, `hsl.cooldown_started`, `hsl.cooldown_ended`
   - history streams:
     - normalized fill history from the live fill path
     - throttled price ticks from orchestrator price inputs
     - completed 1m/1h candle publication via a single `CandlestickManager._persist_batch()` observer
4. minimal snapshot sections:
   - `meta`
   - `account`
   - `health`
   - `positions` with per-side exposure metrics, cached last price, price-action distance, and uPnL
   - `open_orders`
   - `modes`
   - `hsl`
   - `market`
   - `forager`
   - `unstuck`
   - `recent`

## Non-Obvious Details

1. The config surface was intentionally added ahead of the implementation. The history-related knobs are now live and should keep their current semantics unless the contract is intentionally revised.
2. Fill events should continue to use normalized fill semantics from the live fill path. Do not switch the monitor stream to raw CCXT payloads by default.
3. The publisher owns all monitor file writes. Do not add scattered ad hoc writes elsewhere in `passivbot.py`.
4. Snapshot cadence is best-effort from the execution loop. Exact wall-clock cadence is not guaranteed during restart/backoff paths.
5. Completed-candle publication intentionally bootstraps with only the latest seen candle per `(symbol, timeframe)` and publishes only newer candles after that. This avoids flooding monitor history with startup warmup backfill.
6. Candle history is gated on `Passivbot._bot_ready`; warmup/startup persistence should not leak large historical batches into monitor history.

## Gaps Still Open

1. No dedicated `error.publisher` self-reporting path yet.
2. Snapshot coverage is still partial; `exchange_config` is not published yet, and richer market/decision detail can still be added later.
3. No consumer/dashboard code has been added in-repo.

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
