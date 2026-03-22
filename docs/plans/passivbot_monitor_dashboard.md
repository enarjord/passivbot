# Passivbot Monitor Dashboard

## Status

Bot-side publisher groundwork is implemented through history streams and initial snapshot expansion, and an initial read-only relay server now exists.

Implemented now:

1. `monitor.*` config surface and validation
2. `src/monitor_publisher.py`
3. `manifest.json`
4. atomic `state.latest.json`
5. `events/current.ndjson`
6. `history/*.current.ndjson` streams for normalized fills, throttled price ticks, and completed 1m/1h candles
7. checkpoint snapshots
8. event/history rotation and retention pruning
9. targeted tests
10. initial read-only relay server with `/health`, `/snapshot`, and `/ws`
11. initial minimal TUI reader consuming relay `/snapshot` + `/ws`
12. local dev launcher that starts the relay if needed and embeds recent bot-log tailing in the TUI
13. TUI pass 2: focused-symbol detail and recent-order panels from the existing snapshot
14. TUI command prompt with focus switching and quit aliases

Still pending:

1. richer TUI/dashboard UX beyond the initial minimal reader
2. relay auth, history/replay endpoints, and finer-grained subscriptions/filtering
3. `exchange_config` and any further snapshot/detail expansion beyond the current sections
4. publisher self-reporting via `error.publisher`

## Goal

Build a read-only live monitoring dashboard for Passivbot that:

1. shows the current bot state in-place instead of requiring the user to mentally reconstruct it from scrolling console logs
2. records a structured event stream for debugging and replay
3. stays non-intrusive to live trading logic
4. keeps Python live bot ownership limited to publishing state, not rendering UI

The dashboard itself is a separate process. The live bot only publishes structured monitor data to disk.

## Recommended Delivery Model

Use a two-layer architecture:

1. bot-side disk publisher
2. separate read-only monitor relay server

The publisher remains the source of truth.

The relay server reads only from the monitor root on disk and serves:

1. HTTP snapshot endpoints
2. WebSocket live event streams

This keeps:

1. durable local observability and replay
2. remote/mobile/web/TUI access
3. strict separation between trading logic and UI/network serving

The live bot should not serve dashboards directly from its own process.

## Core Design

The dashboard reads from a single monitor data root on disk.

The bot publishes:

1. a full current snapshot
2. a structured append-only event stream
3. selected history streams for live-forward monitoring

The dashboard does not access the live bot process memory directly.

### Relay Server Layer

Recommended shape:

1. `state.latest.json` stays the bootstrap source
2. `events/current.ndjson` plus rotated segments stay the event/history source
3. a separate relay process tails these files and exposes a stable network protocol

This relay should be:

1. read-only
2. restartable independently from the bot
3. able to resync from disk after its own restart
4. usable locally on the VPS and remotely via reverse proxy

## Non-Goals

1. Do not make the dashboard part of the live bot execution loop UI.
2. Do not scrape console logs as the dashboard data source.
3. Do not dump arbitrary `vars(self)` internals as the public monitor schema.
4. Do not duplicate all historical caches into the monitor stream by default.
5. Do not make the first version a web app.
6. Do not replace the disk publisher with direct in-bot WebSocket serving.

## Architecture

### Publisher

Add a dedicated module, tentatively:

- `src/monitor_publisher.py`

Responsibilities:

1. maintain a structured current-state view model
2. append structured events
3. rotate/compress event segments
4. write `state.latest.json` atomically
5. enforce retention policy
6. stay best-effort and non-intrusive

The publisher should be the only component writing monitor files.

Current implementation status:

1. this module now exists as `src/monitor_publisher.py`
2. manifest, event append, atomic snapshot writes, checkpoints, and retention are implemented
3. history streams now cover normalized fills, throttled price ticks, and completed 1m/1h candles

### Relay Server

Recommended new component, separate from the bot:

- `src/tools/monitor_relay.py` or similar

Responsibilities:

1. load `state.latest.json` on startup
2. tail live event/history files
3. maintain per-bot subscriber sets
4. serve snapshot and stream APIs
5. handle reconnect/resync without touching bot internals

Transport recommendation:

1. HTTP for current state/bootstrap
2. WebSocket for live push

Why:

1. WebSocket works well for browser, mobile, and TUI clients
2. HTTP makes debugging and initial bootstrap easy
3. disk remains the source of truth even if clients disconnect

Alternative transports considered:

1. SSE
   - acceptable for browser-only one-way streaming
   - weaker fit for mobile/native/TUI variety
2. polling HTTP
   - simplest fallback
   - weaker live UX and more waste
3. broker systems (MQTT/NATS/etc.)
   - likely overkill for current scope

Recommendation:

1. keep the disk-backed publisher
2. add a read-only HTTP + WebSocket relay
3. keep protocol stable and independent of file layout

Current implementation status:

1. `src/monitor_relay.py` and `src/tools/monitor_relay.py` now exist
2. the relay serves `GET /health`, `GET /snapshot`, and `GET /ws`
3. websocket clients receive one snapshot first, then live-forward `event` and `history` messages by tailing `*.current.ndjson`
4. the relay currently serves only live-forward current-file data; auth, replay/history APIs, and richer subscriptions are still pending

### Bot Integration

`Passivbot` should publish through a narrow interface, for example:

- `publisher.record_event(kind, tags, payload, ...)`
- `publisher.update_state(section, payload)`
- `publisher.flush_snapshot_if_due()`

The live bot should not perform ad hoc file writes across many methods.

Current implementation status:

1. publisher init is wired in `Passivbot.__init__`
2. startup, ready, shutdown, health, balance, position, order, fill, mode, and HSL hooks emit monitor events
3. snapshot rebuild/flush is wired from startup, loop cadence, and shutdown
4. fill history, price ticks, and completed candles are delegated through the publisher from narrow existing hooks
5. snapshot sections now include `market`, `forager`, `unstuck`, and `recent` in addition to the original minimal set

## Monitor Data Root

Recommended structure:

```text
monitor/
  {exchange}/
    {user}/
      manifest.json
      state.latest.json
      events/
        current.ndjson
        2026-03-20T13.ndjson.gz
      history/
        fills.current.ndjson
        fills.2026-03-20T13.ndjson.gz
        price_ticks.current.ndjson
        candles_1m.current.ndjson
        candles_1h.current.ndjson
      checkpoints/
        state.2026-03-20T13-08-00.json.gz
```

The dashboard should only read from this root.

The relay server should also read only from this root.

## Data Model

### 1. Current Snapshot

File:

- `state.latest.json`

Rule:

- contains the full current live state
- excludes historical time series

Examples:

- current EMA values are included
- the historical candles used to compute those EMAs are not included

Suggested top-level structure:

```json
{
  "schema_version": 1,
  "meta": {},
  "account": {},
  "health": {},
  "positions": {},
  "open_orders": {},
  "modes": {},
  "hsl": {},
  "forager": {},
  "unstuck": {},
  "market": {},
  "exchange_config": {},
  "recent": {}
}
```

Suggested sections:

Implementation note:

1. `market`, `forager`, `unstuck`, and `recent` now exist in the live snapshot
2. `exchange_config` is still pending
3. some details listed below are target-state enrichment rather than fields already published today

#### `meta`

1. exchange
2. user
3. quote
4. pid
5. bot start timestamp
6. current cycle timestamp
7. git commit if available
8. schema version

#### `account`

1. raw balance
2. snapped balance
3. equity
4. realised pnl cumsum
5. wallet exposure totals by side

#### `health`

1. uptime
2. last loop duration
3. hourly error count
4. rate-limit count
5. websocket reconnect count
6. memory snapshot metrics
7. pending exchange-config retries

#### `positions`

Per symbol and side:

1. size
2. entry price
3. wallet exposure
4. unrealised pnl if available
5. trailing metrics used by logic

#### `open_orders`

Canonical open-order view per symbol.

Include what bot uses in reconciliation:

1. side
2. position side
3. qty
4. price
5. reduce-only
6. custom id
7. order type classification if known

#### `modes`

1. effective modes per symbol and side
2. explicit overrides
3. runtime forced modes
4. symbol states returned by orchestrator if available

#### `hsl`

Per side:

1. enabled
2. current tier
3. drawdown raw
4. drawdown EMA
5. drawdown score
6. peak strategy equity
7. rolling peak strategy equity
8. cooldown state
9. pending red latch state
10. live cooldown intervention policy

#### `forager`

Per side:

1. candidate universe
2. active symbols
3. selected symbols
4. score weights after normalization
5. current volume / volatility / readiness values actually used
   - planned next, not fully published today
6. gating reasons if available
   - planned next, not fully published today

#### `unstuck`

Per side:

1. current allowance
2. balance peak and pct from peak
3. target symbols/orders if any
4. blocking reasons

#### `market`

Per symbol:

1. last traded price
2. market active flag
3. effective min cost
4. candle freshness / last finalized candle timestamps
5. approval / ignore membership
6. trailing metrics when available
7. best bid
   - planned next, not published today
8. best ask
   - planned next, not published today
9. mid
   - planned next, not published today
10. currently used EMA values
   - close EMA
   - volume EMA
   - log-range EMA
   - HSL EMA metrics if applicable
   - planned next, not published today

#### `exchange_config`

Per symbol:

1. desired leverage
2. desired margin mode
3. configured status
4. retry attempt count
5. retry-after timestamp if pending

#### `recent`

Short bounded summaries for quick UI display:

1. recent created orders
2. recent canceled orders
3. recent fills
   - planned next
4. recent mode changes
   - planned next
5. recent errors
   - planned next
6. recent HSL transitions
   - planned next

### 2. Event Stream

File:

- `events/current.ndjson`

Rule:

- append-only
- structured
- selective, not every object mutation
- rich enough that the dashboard looks alive and debugging is improved

Every event should include:

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

Recommendations:

1. keep machine parsing centered on `kind`
2. keep `tags` for human grouping and future filtering
3. use dot-namespace style for `kind`
4. keep payload schemas stable per `kind`

### 2b. Network Protocol

The relay server should not expose raw files directly as the primary client contract.

Instead expose a stable protocol with two main message families:

#### Snapshot message

```json
{
  "type": "snapshot",
  "seq": 12345,
  "payload": {}
}
```

#### Event message

```json
{
  "type": "event",
  "seq": 12346,
  "kind": "position.changed",
  "payload": {}
}
```

Optional later:

1. `history_batch`
2. `heartbeat`
3. `resync_required`
4. `error`

Client flow:

1. connect
2. receive current snapshot
3. receive live event messages
4. detect seq gaps
5. request resync or reconnect and reload snapshot

This makes the client contract independent from file segmentation/rotation details.

### Event Kinds

Initial recommended categories:

#### Bot lifecycle

1. `bot.start`
2. `bot.ready`
3. `bot.restart`
4. `bot.stop`

#### Health

1. `health.summary`
2. `health.memory`
3. `health.ws_reconnect`
4. `health.rate_limit`

#### Error stream

All bot and exchange errors should be published here in structured form.

Examples:

1. `error.bot`
2. `error.exchange`
3. `error.ws`
4. `error.orchestrator`
5. `error.publisher`

Payload should include:

1. message
2. exception class
3. context
4. symbol if relevant
5. side if relevant
6. whether it is retryable

#### Account

1. `account.balance`
2. `account.equity`
3. `account.realized_pnl`

#### Market ticks

1. `market.tick`

Payload:

1. last
2. bid
3. ask
4. mid

Rules:

1. emit only on actual change
2. no order book depth
3. throttle to at most one tick event per symbol per configured interval
   - initial recommendation: `500ms`

#### Candles

1. `market.candle.1m`
2. `market.candle.1h`

Rules:

1. publish completed candles only
2. 1m every minute for eligible symbols
3. 1h every hour for eligible symbols

#### Orders and fills

1. `order.opened`
2. `order.canceled`
3. `order.filled`
4. `fills.refresh`

For fill events, publish canonical normalized fields by default.

Do not include full raw CCXT payloads in the monitor event stream by default.

Reason:

1. raw payloads already exist in `caches/fill_events/`
2. canonical payload is enough for live monitoring
3. default raw inclusion would create unnecessary bloat

Possible future option:

- `monitor.include_raw_fill_payloads`

Default should be `false`.

#### Positions

1. `position.opened`
2. `position.changed`
3. `position.closed`

#### Modes and orchestration

1. `mode.changed`
2. `forager.selection`
3. `forager.candidates`
4. `unstuck.allowance`
5. `unstuck.selection`

#### HSL

1. `hsl.transition`
2. `hsl.red_triggered`
3. `hsl.cooldown_started`
4. `hsl.cooldown_ended`
5. `hsl.cooldown_intervention`
6. `hsl.policy_applied`

#### Exchange config

1. `exchange_config.start`
2. `exchange_config.success`
3. `exchange_config.retry`
4. `exchange_config.rate_limit`
5. `exchange_config.failure`

### 3. History Streams

These are still under the same monitor root, but separate from `events/current.ndjson`.

#### `history/price_ticks.ndjson`

Rich live-forward tick stream for charting.

#### `history/candles_1m.ndjson`

Completed 1m candles emitted from bot start onward.

#### `history/candles_1h.ndjson`

Completed 1h candles emitted from bot start onward.

#### `history/fills.ndjson`

Canonical fill history observed from bot start onward.

## Snapshot vs Stream vs Existing Caches

### Snapshot

Use for:

1. current state
2. fast dashboard startup
3. current truth

### Event stream

Use for:

1. transitions
2. recent activity
3. "alive" dashboard feel
4. debugging timeline

### Existing caches

Use selectively later for optional deeper history loading.

Recommended MVP behavior:

1. dashboard starts from current snapshot plus live-forward history only
2. dashboard does not backfill old candles/fills by default
3. later add a manual "load historical data" action if desired

This keeps startup fast and the UI responsive.

## Retention and Rotation

Retention should be configurable.

Suggested config namespace:

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

Rules:

1. rotate by time or size, whichever comes first
2. compress rotated segments
3. keep current segment uncompressed
4. retention should be enforced separately for events and history streams

## Relay API

Recommended initial endpoints:

1. `GET /snapshot`
2. `GET /health`
3. `GET /ws`

Optional later:

1. `GET /recent-events`
2. `GET /history/fills`
3. `GET /history/candles?symbol=...&timeframe=...`
4. `GET /history/ticks?symbol=...`

WebSocket behavior:

1. send snapshot immediately after connect
2. then send live event/history updates
3. optionally support subscriptions later
   - symbols
   - streams
   - sides
   - event kinds

The first version can broadcast the full stream to each client and add subscriptions later if needed.

## Performance and Safety Requirements

1. Publisher must be non-blocking relative to live trading paths.
2. File writes should be batched where possible.
3. `state.latest.json` should be written atomically.
4. If publisher fails, bot should log a monitor error event and continue unless explicitly configured otherwise.
5. Publisher must never mutate trading decisions.
6. Relay must remain read-only and must not expose trading control actions.

## Security and Deployment

Recommended assumptions for the relay:

1. authenticated access only
2. TLS via reverse proxy or equivalent
3. read-only token/auth model
4. one namespace per `{exchange}/{user}` monitor root

Suggested deployment:

1. bot writes to local monitor root on VPS
2. relay process serves that root locally
3. reverse proxy terminates TLS and forwards `/ws` and HTTP endpoints

This keeps the bot isolated from network client complexity.

## Dashboard MVP

First version should be terminal UI only.

Suggested stack:

1. `textual` if richer panes/tables/layout are desired
2. `rich` if a lighter first step is preferred

Suggested panes:

1. top status bar
2. account and health
3. positions
4. open orders
5. modes and HSL
6. forager
7. unstuck
8. recent events
9. symbol detail pane with current price and EMA distances

The dashboard should:

1. resize with terminal size
2. pre-allocate layout where practical
3. avoid noisy layout shifts

Future clients can include:

1. local TUI
2. browser dashboard
3. mobile app

All of them should consume the same relay protocol rather than custom per-client file readers.

## Bot Integration Points

Initial publisher hooks should likely be placed at:

1. startup and shutdown
2. after balance update
3. after positions update
4. after open-order refresh
5. after orchestrator output is received
6. after mode changes are applied/logged
7. after HSL state changes
8. after forager candidate/selection computation
9. after unstuck allowance/selection updates
10. after fills manager updates
11. after completed candle detection
12. after exchange-config retry/success/failure

## Testing Strategy

### Unit tests

1. snapshot schema generation
2. event serialization
3. rotation and retention
4. tick throttling
5. atomic latest-snapshot update

### Integration tests

1. live bot emits monitor files when enabled
2. monitor output remains bounded under retention policy
3. monitor publisher does not block trading loop

### Relay tests

When the relay is implemented:

1. snapshot endpoint reflects current `state.latest.json`
2. WebSocket sends snapshot first, then incremental events
3. client reconnect after relay restart can recover from disk
4. seq-gap handling forces clean resync
5. auth failures are rejected cleanly

### Fake-live harness

The fake-live harness should eventually emit the same monitor data.

This enables scenario assertions such as:

1. RED trigger emits `hsl.red_triggered`
2. cooldown start emits `hsl.cooldown_started`
3. manual intervention emits the correct `hsl.cooldown_intervention` and policy event
4. no reopen before cooldown end
5. forager selection changes appear when expected

The fake-live harness should later also drive relay-level regression tests:

1. run scenario
2. emit monitor files
3. start relay over those files
4. assert snapshot + streamed events match expected behavioral transitions

This should become an additional behavioral regression layer above unit tests.

## Phased Rollout

### Phase 1

1. create `monitor_publisher.py`
2. add monitor config
3. emit `state.latest.json`
4. emit `events/current.ndjson`
5. publish core sections:
   - meta
   - account
   - health
   - positions
   - open_orders
   - modes
   - hsl

### Phase 2

1. add forager and unstuck sections
2. add tick and completed-candle history streams
3. add fill history stream
4. add rotation/compression/retention

### Phase 3

1. build TUI dashboard
2. add symbol detail pane and charts
3. add optional manual historical backfill loading

### Phase 4

1. fake-live monitor emission
2. replay and regression assertions over monitor output
3. optional future web dashboard

## Open Questions

1. Should publisher flush events immediately or in small batches?
2. Should snapshot cadence default to each execution cycle or a fixed 1 Hz timer?
3. Should recent event summaries also be stored in `state.latest.json` or derived only by dashboard?
4. Should fill history stream be the canonical future dashboard source, or remain secondary to existing fill cache?
5. How much symbol-level EMA detail should be included in snapshot vs symbol detail history?

## Recommendation

Proceed with Phase 1 first.

That creates the stable publication contract before any dashboard rendering work begins.
