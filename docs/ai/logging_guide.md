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

The live event console sink is enabled by default and is fed by the structured
live event pipeline. Set `logging.live_event_console=false` or
`PASSIVBOT_LIVE_EVENT_CONSOLE=0` to disable it during controlled comparisons.
Some legacy stdlib console logs still exist, so the default projection may
temporarily duplicate a few order/execution lifecycle messages until those
legacy call sites are migrated.
The console projection is operator-facing and local: it should prefer
trading-relevant summaries over internal data plumbing. Because this projection
is enabled by default, account-state magnitudes such as balance, equity, PnL,
fees, or position size may appear on the operator console by default. Do not
forward console/stdout to shared or centralized log aggregation without either
opting out or applying a deliberate redaction policy. HSL status events are
console-visible only when they are pside-level, red/cooldown, or tied to a held
coin; routine flat coin-universe status remains in the structured event stream
but is kept off the console. Fill, position, and balance change events are
console-visible because they are the primary operator-facing account state
changes.
`risk.mode_changed` and `hsl.transition` events are console-visible because
mode changes and HSL tier transitions explain risk-state changes that affect
trading.
Initial-entry distance-gate, min-effective-cost, and realized-loss gate block
events are console-visible because they explain operator-relevant order
omissions: staged entries waiting for an acceptable market distance, entries
skipped because the configured slot size is below exchange-effective minimums,
and loss-realizing closes deferred by the realized-loss gate.
Periodic `health.summary` events are console-visible because they provide a
compact operator heartbeat covering uptime, loop latency, position counts,
recent order/fill activity, errors, and resource pressure. Degraded
`health.summary` events such as execution-loop error bursts must stay immediate
and must not expose raw exchange URLs or credentials in console summaries.
Startup timing events are console-visible because they explain which startup
phase is slow or complete: account state, active-candle warmup, HSL replay,
full warmup, market state, and final startup readiness. The legacy stdlib
startup timing line is only a fallback when the structured event console path
is unavailable or explicitly disabled.
Position-level `trailing.status`, `unstuck.status`, and `unstuck.selection`
events are console-visible because they explain why an existing position is
waiting, armed, triggered, over budget, selected for unstucking, or blocked by
the unstuck EMA gate. Unsupported strategy diagnostics must say so explicitly
instead of fabricating threshold/retracement prices.
Legacy stdlib unstuck status/selection lines are only fallbacks when the
structured event console path is unavailable or explicitly disabled.
Supported trailing diagnostics should include the selected mode, such as
`trailing`, `grid`, `auto_reduce`, `unstuck`, `none`, or `other`, when the next
order state is known. They should also show the effective threshold percentage
and price, retracement percentage and price, and the projected retracement price
that would apply if the threshold were reached. `forager.selection` events are
also console-visible through a throttled compact summary because they explain
which coins are being selected, retained, or skipped when forager entries are
possible.

## Review Heuristic

1. Can INFO be tailed for long periods without noise overload?
2. Can operators explain why bot is waiting vs broken?
3. Are DEBUG logs sufficient to reconstruct decision context?
