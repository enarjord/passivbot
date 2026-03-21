# Monitor Output

Passivbot can now publish a read-only monitor data root to disk for external tools such as a future TUI/dashboard.

This is a bot-side publisher only. No built-in dashboard UI has been added yet.

## Purpose

The monitor publisher gives you:

1. a current full-state snapshot
2. a structured append-only event stream
3. selected append-only history streams for fills, price ticks, and completed candles
4. periodic compressed checkpoints of the latest snapshot

The dashboard/process consuming this data must read only from the monitor root on disk. It should not inspect live bot memory directly.

## Enable It

Add a `monitor` section to your live config or start from the updated template:

```json
"monitor": {
  "enabled": true,
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

Current implementation note:

1. the bot validates the full `monitor.*` config surface
2. `retain_price_ticks`, `retain_candles`, `retain_fills`, `price_tick_min_interval_ms`, `emit_completed_candles`, and `include_raw_fill_payloads` are active for the current history streams
3. richer snapshot sections are still planned separately

## Output Layout

When enabled, the bot writes under:

```text
{root_dir}/
  {exchange}/
    {user}/
      manifest.json
      state.latest.json
      events/
        current.ndjson
        2026-03-20T13-08-00.ndjson.gz
      history/
        fills.current.ndjson
        fills.2026-03-20T13-08-00.ndjson.gz
        price_ticks.current.ndjson
        candles_1m.current.ndjson
        candles_1h.current.ndjson
      checkpoints/
        state.2026-03-20T13-08-00.json.gz
```

Current behavior:

1. `state.latest.json` is written atomically
2. `events/current.ndjson` is append-only
3. `history/*.current.ndjson` files are append-only for live-forward streams
4. rotated event/history segments are optionally gzipped
5. periodic snapshot checkpoints are optionally gzipped
6. retention prunes old rotated/history/checkpoint files and enforces `monitor.max_total_bytes`

## Snapshot Schema

Current Phase 1 snapshot sections:

1. `meta`
2. `account`
3. `health`
4. `positions`
5. `open_orders`
6. `modes`
7. `hsl`
8. `market`
9. `forager`
10. `unstuck`
11. `recent`

Important limits:

1. historical candle arrays are not included
2. `exchange_config` is not published yet
3. the snapshot is best-effort observability data and does not drive trading behavior

Current expansion details:

1. `market` includes current cached last price, candle refresh/finalization timestamps, min-cost metadata, approval/ignore flags, open-order/position presence, and trailing state when available
2. `forager` includes per-side candidate universe, selected symbols, slot counts, and current forager score-weight config
3. `unstuck` includes per-side allowance status plus any currently open unstuck orders
4. `recent` includes recent created and canceled orders retained by the live bot throttling caches

## Event Stream

Each event in `events/current.ndjson` uses this envelope:

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

Currently emitted event kinds:

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

Fill events use normalized fill fields from the live bot path. Raw CCXT payloads are not included by default.

## History Streams

Current history files:

1. `history/fills.current.ndjson`
2. `history/price_ticks.current.ndjson`
3. `history/candles_1m.current.ndjson`
4. `history/candles_1h.current.ndjson`

Current behavior:

1. fills reuse the same normalized fill payload used by `order.filled`
2. raw fill payloads are included only when `monitor.include_raw_fill_payloads=true`
3. price ticks are throttled per symbol by `monitor.price_tick_min_interval_ms`
4. completed candle streams bootstrap by emitting only the latest candle seen for a symbol/timeframe, then publish only newer completed candles after that
5. candle history is published only after the bot is ready, so startup warmup does not flood the monitor root with historical backfill

## Current Gaps

Not implemented yet:

1. built-in dashboard/TUI reader
2. `exchange_config` snapshot coverage
3. publisher-originated `error.publisher` events

## Related Docs

1. [Running the bot live](live.md)
2. [Configuration reference](configuration.md)
3. [Monitor dashboard plan](plans/passivbot_monitor_dashboard.md)
