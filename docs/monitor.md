# Monitor Output

Passivbot can now publish a read-only monitor data root to disk for external tools such as a future TUI/dashboard.

An initial read-only relay server is also available for local/remote consumers. No built-in dashboard UI has been added yet.

## Purpose

The monitor publisher gives you:

1. a current full-state snapshot
2. a structured append-only event stream
3. selected append-only history streams for fills, price ticks, and completed candles
4. periodic compressed checkpoints of the latest snapshot

The dashboard/process consuming this data must read only from the monitor root on disk. It should not inspect live bot memory directly.

## Relay Server

The repo now includes a small read-only relay:

```bash
python3 src/tools/monitor_relay.py --monitor-root monitor --host 127.0.0.1 --port 8765
```

Current endpoints:

1. `GET /health`
2. `GET /snapshot`
3. `GET /ws`

Current behavior:

1. `/snapshot` returns the current `state.latest.json` envelope for one bot
2. `/ws` sends one snapshot first, then live `event` and `history` messages as current files advance
3. when multiple monitor roots exist, clients must pass both `exchange` and `user` query params
4. the relay reads only from the monitor root on disk; it does not attach to bot memory
5. the relay currently tails only `*.current.ndjson` files and does not yet serve rotated-history replay over HTTP

## Minimal TUI

The repo now also includes a minimal terminal reader against the relay:

```bash
python3 src/tools/monitor_tui.py --relay-url http://127.0.0.1:8765
```

When multiple monitor roots are available on the relay, select one bot explicitly:

```bash
python3 src/tools/monitor_tui.py \
  --relay-url http://127.0.0.1:8765 \
  --exchange bitget \
  --user bitget_01 \
  --focus-symbol BTC/USDT:USDT
```

Current behavior:

1. bootstraps current-state panels from `/snapshot`
2. refreshes `/snapshot` periodically for live state sections
3. consumes `/ws` for recent events and recent price ticks
4. can prioritize one symbol with `--focus-symbol`
5. renders a boxed terminal dashboard with current summaries, a focused-symbol detail box, a total-TWE positions box, dedicated forager and unstuck boxes, recent order activity, recent events, recent ticks, and optional local log tailing without touching bot memory directly
6. uses a two-column layout on wider terminals so the right side is filled with recent activity panels instead of staying mostly empty
7. redraws in place only when the rendered frame changes, and applies row-diff terminal updates instead of repainting the entire screen, which reduces visible flicker further
8. provides a bottom command prompt during runtime; current commands include `help`, `focus BTC`, `focus next`, `focus prev`, `focus auto`, `pause`, `resume`, `dump`, `quit`, and `exit`
9. `pause` freezes the data panels for copy/inspection while keeping the command line live, and `dump` writes the currently displayed screen to `tmp/monitor_tui_dump_*.txt`

For local monitor development, there is also a one-command wrapper:

```bash
python3 src/tools/monitor_dev.py --exchange bitget --user bitget_01
```

Current behavior:

1. reuses an existing relay if one is already healthy at the target `--relay-url`
2. otherwise launches the relay automatically
3. selects the newest `logs/*.log` file by default unless `--log-file` is provided
4. passes through `--focus-symbol` when you want to center the screen on one market
5. shows recent bot log lines inside the TUI so you do not need a separate `tail -f` terminal during iteration

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

1. `positions` now includes per-side wallet-exposure metrics (`wallet_exposure`, `wel_ratio`, `wele_ratio`, `twel_ratio`) plus `price_action_distance`, `upnl`, and cached `last_price` when available
2. `market` includes current cached last price, candle refresh/finalization timestamps, min-cost metadata, approval/ignore flags, open-order/position presence, trailing state, and current per-side EMA band snapshots when available
3. `forager` includes per-side candidate universe, selected symbols, slot counts, pending selected symbols, `next_symbol`, and current forager score-weight config
4. `unstuck` includes per-side allowance status, any currently open unstuck orders, and the latest planned unstuck symbol/target/EMA-trigger context when available
5. `recent` includes recent created and canceled orders retained by the live bot throttling caches

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

1. richer browser/mobile dashboard reader
2. relay auth, filtering/subscriptions, and replay/history HTTP endpoints
3. `exchange_config` snapshot coverage
4. publisher-originated `error.publisher` events

## Related Docs

1. [Running the bot live](live.md)
2. [Configuration reference](configuration.md)
3. [Monitor dashboard plan](plans/passivbot_monitor_dashboard.md)
