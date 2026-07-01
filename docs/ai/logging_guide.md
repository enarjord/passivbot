# Logging Guide (Policy)

This file defines current logging policy. It is not a historical work log.

## Levels

| Level | Audience | Use |
|-------|----------|-----|
| INFO | Operators | Durable operational events |
| DEBUG | Developers | Decision context + diagnostics |
| TRACE | Deep debugging | Full payloads/firehose |

## INFO Must Include

1. Order create/cancel actions with reason/context.
2. Fill events with symbol/side/qty/price and PnL where relevant.
3. Position/balance state changes.
4. Mode changes meaningful to operators.
5. Unexpected errors with traceback.

## Keep Out Of INFO

1. High-frequency polling or fetch loops with no state change.
2. Internal cache noise and repeated maintenance chatter.
3. Full API payloads.

## Tagging

Use stable `[tag]` prefixes, for example:

1. `[order]`
2. `[fill]`
3. `[pos]`
4. `[balance]`
5. `[health]`
6. `[candle]`
7. `[ws]`
8. `[risk]`
9. `[ema]`

Structured live event tags and reason codes must use the shared registries in
`src/live/event_bus.py` (`EventTags`, `ReasonCodes`) when a stable value already
exists there. Add new registry values before introducing repeated literals. See
`live_event_registry.md` for the current stable values.

## Fallback Visibility

Fallbacks in critical paths log warnings with required context from `error_contract.md`.

## Live Event Debug Profiles

Use `logging.live_event_debug_profiles` or the
`PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` environment variable to opt into narrow
structured-event enrichment without increasing default console noise. Values may
be a list or comma/space separated string. `all` enables every known profile.

Supported profile names are documented in `live_event_registry.md`; keep that
registry synchronized with `src/live/event_bus.py`.

Current profile behavior:

1. `rust` adds bounded Rust orchestrator input-symbol and output-order samples
   to the existing `rust_orchestrator.called` and
   `rust_orchestrator.returned` events. Full raw Rust payload persistence remains
   disabled; hashes stay present for correlation.
2. `remote_calls`, `candles`, `ema`, `fills`, `forager`, `execution`, and `hsl`
   add bounded event-specific debug summaries such as data-key lists,
   reason/sample counts, correlation ids, and shape metadata. These summaries
   must not copy raw exchange/account payloads, credentials, or unbounded row
   data.
3. `startup` and `state` are reserved profile names for startup/state
   diagnostics. Startup lifecycle events already surface which profiles are
   enabled, and live performance reports can summarize them from existing
   monitor events.

Unknown profile names should fail visibly instead of being ignored silently.

## Live Event Console Projection

`logging.live_event_console=true` or `PASSIVBOT_LIVE_EVENT_CONSOLE=1` enables
an opt-in console sink fed by the structured live event pipeline. It is disabled
by default while legacy stdlib console logs still exist, because enabling it can
duplicate some order/execution lifecycle messages. Use it for controlled smoke
tests of event-stream console formatting before migrating default console output.
The console projection is operator-facing: it should prefer trading-relevant
summaries over internal data plumbing. HSL status events are console-visible
only when they are pside-level, red/cooldown, or tied to a held coin; routine
flat coin-universe status remains in the structured event stream but is kept off
the console.

## Review Heuristic

1. Can INFO be tailed for long periods without noise overload?
2. Can operators explain why bot is waiting vs broken?
3. Are DEBUG logs sufficient to reconstruct decision context?
