# Monitor Output

Passivbot can now publish a read-only monitor data root to disk for external tools such as a future TUI/dashboard.

This is a bot-side publisher only. No built-in dashboard UI has been added yet.

## Purpose

The monitor publisher gives you:

1. a current full-state snapshot
2. a structured append-only event stream
3. periodic compressed checkpoints of the latest snapshot

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
2. not every config knob is used yet
3. `retain_price_ticks`, `retain_candles`, `retain_fills`, `price_tick_min_interval_ms`, `emit_completed_candles`, and `include_raw_fill_payloads` are reserved for later increments

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
      checkpoints/
        state.2026-03-20T13-08-00.json.gz
```

Current behavior:

1. `state.latest.json` is written atomically
2. `events/current.ndjson` is append-only
3. rotated event segments are optionally gzipped
4. periodic snapshot checkpoints are optionally gzipped
5. retention prunes old rotated/checkpoint files and enforces `monitor.max_total_bytes`

## Snapshot Schema

Current Phase 1 snapshot sections:

1. `meta`
2. `account`
3. `health`
4. `positions`
5. `open_orders`
6. `modes`
7. `hsl`

Important limits:

1. historical candle arrays are not included
2. history streams for ticks/candles/fills are not published yet
3. the snapshot is best-effort observability data and does not drive trading behavior

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

## Current Gaps

Not implemented yet:

1. built-in dashboard/TUI reader
2. tick publication
3. completed candle publication
4. dedicated history streams under `history/`
5. richer exchange-config, market, forager, unstuck, and recent snapshot sections
6. publisher-originated `error.publisher` events

## Related Docs

1. [Running the bot live](/Users/eiriknarjord/repos/passivbot-3/docs/live.md)
2. [Configuration reference](/Users/eiriknarjord/repos/passivbot-3/docs/configuration.md)
3. [Monitor dashboard plan](/Users/eiriknarjord/repos/passivbot-3/docs/plans/passivbot_monitor_dashboard.md)
