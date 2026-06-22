# Live Logging Overhaul Plan

## Problem

Live currently has three related but separate observability surfaces:

1. Standard Python console/file logs from `src/logging_setup.py`.
2. Monitor artifacts from `src/monitor_publisher.py` and `src/passivbot_monitor.py`.
3. Gatekeeper and planning diagnostics emitted as structured monitor events from live code.

These surfaces are useful, but they overlap unevenly. Operators tail console logs, monitor tooling
reads rotating artifacts, and future gatekeeper diagnostics want a richer event stream. Duplicating
each event manually into every surface will create drift and noise.

## Target Shape

Use one canonical structured event model with multiple sinks:

1. Console projection: compact human output for operators.
2. Text logfile projection: same stream as console, with bounded rotation.
3. Structured event stream: newline-delimited JSON for monitor/gatekeeper/automation consumers.
4. History streams: high-volume domain artifacts such as fills, balance, positions, candles, and
   order actions.
5. Relay/fanout: optional websocket or IPC projection for dashboards and remote monitoring.

Do not merge every surface into one physical file. Keep projections separate by audience and volume,
but emit them from one structured API so severity, tags, timestamps, symbols, reasons, and event ids
stay consistent.

## Event Contract

Each structured event should include:

- `ts`: exchange or wall-clock timestamp in milliseconds, plus UTC render in the sink if useful.
- `level`: `trace`, `debug`, `info`, `warning`, `error`, or `critical`.
- `event_type`: stable dotted name, for example `order.create`, `hsl.transition`,
  `planning.unavailable`, or `fill.ingested`.
- `tags`: ordered hierarchy such as `["risk", "hsl", "coin"]`.
- `source`: component name such as `live`, `monitor`, `gatekeeper`, `candles`, or `fills`.
- `symbol`, `pside`, `side`, `order_id`, `client_order_id` when applicable.
- `reason`: stable machine-readable reason code.
- `message`: short human-readable summary.
- `data`: bounded JSON object with event-specific context.
- `correlation_id`: optional id tying one decision/action chain together.

The console/logfile formatter should be a projection of the same event, not a separate hand-written
message where possible.

## Levels And Tags

INFO should stay tail-safe:

- order create/cancel decisions and exchange results
- fills and realized PnL summaries
- position/balance changes
- mode changes and risk-state transitions
- startup/shutdown lifecycle
- bounded fallback use

DEBUG should carry decision context:

- why an order candidate was approved/deferred/rejected
- candidate ranking details
- refresh and cache decisions that materially affect trading

TRACE should be explicit opt-in:

- raw exchange payloads
- high-frequency candle/ticker events
- full strategy input/output snapshots

Tags should be stable and hierarchical. Current examples such as `[risk]`, `[ema]`, `[fills]`,
`[candle]`, `[order]`, `[health]`, and `[forager]` should map directly to structured `tags`.

## Rotation And Retention

Use bounded disk by default:

- Text live logs: rotate by size, with configurable backup count.
- Structured events: rotate by size and age, as monitor events already do.
- History streams: rotate independently because volume differs by stream.
- Compression: enable for older structured/history segments.
- Retention: enforce total-byte and retention-day caps per bot and globally.

Defaults should be conservative enough for small VPS hosts: bounded text logs, bounded structured
events, and no unbounded debug/firehose streams unless explicitly enabled.

## Migration Plan

1. Keep current console/logfile behavior stable and turn on bounded rotation by default for new
   configs.
2. Add a small `emit_event()` facade that records structured fields and can project to Python
   logging and monitor artifacts.
3. Move existing gatekeeper/planning events to the facade first, because they already have a
   structured shape.
4. Move risk events next: HSL, TWEL, WEL, unstuck, fill-history coverage, and EMA readiness.
5. Move order lifecycle events: ideal order, reconciled action, gatekeeper decision, exchange
   request, exchange response, and observed fill.
6. Add correlation ids around each live-loop planning cycle and order action chain.
7. Add docs for event names, tags, levels, and retention settings.

## Compatibility

Existing log tailing and monitor artifact consumers should continue to work during migration.
Structured events can be added alongside current logs first. Once a component emits through the
facade, its text log line should be produced by the facade projection to avoid duplicate wording.

## Open Questions

1. Whether the monitor relay should read only structured events, or also expose selected current
   snapshots.
2. Whether gatekeeper diagnostics should be retained at DEBUG by default or only when decisions are
   non-approved.
3. Whether raw exchange payload TRACE retention should be time-limited even when enabled, to avoid
   secret and disk-risk surprises.
4. How much of the event schema should be frozen before external dashboard integrations consume it.
