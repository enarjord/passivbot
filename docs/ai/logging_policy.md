# Logging And Observability Policy

This document owns stable logging, redaction, and sink behavior. Event values are generated in
`generated/live_event_registry.md`; event-family payload contracts live in
`features/live_events.md`. Migration progress belongs under `docs/plans/`.

## Audiences And Levels

| Level | Audience | Content |
|---|---|---|
| INFO | Operators | Durable state changes and actionable operational summaries |
| DEBUG | Developers | Bounded decision context and diagnostics |
| TRACE | Deep debugging | High-volume detail that is still subject to secret policy |

INFO includes order actions, fills, material position/balance/mode changes, startup readiness,
degraded health, and unexpected failures. Keep unchanged polling, cache maintenance chatter, and
full API payloads out of INFO.

Use stable text tags such as `[order]`, `[fill]`, `[pos]`, `[balance]`, `[health]`, `[candle]`,
`[ws]`, `[risk]`, and `[ema]`. Structured producers use code-owned `EventTypes`, `EventTags`, and
`ReasonCodes` rather than repeated string literals.

## Redaction And Boundedness

No sink may retain credentials, signatures, private keys, authorization headers, arbitrary URLs,
raw exchange/account payloads, or unbounded row collections. Treat exception messages and
tracebacks as potentially containing request or account data.

- Connector-local and structured execution events retain stable classifications, bounded context,
  and exception type—not raw responses, exception text, or tracebacks.
- Outer process/startup failure logs may include a traceback in the protected developer text log
  when needed for diagnosis, but should sanitize known request and credential material first.
- Structured event debug profiles add bounded shapes, counts, keys, correlation IDs, and samples;
  they do not enable raw-payload persistence.
- Exchange-config response formatting permits only canonical status, finite numeric leverage,
  bounded response code, and response type/presence.

## Sink Isolation

Observability-only failures must not change trading behavior. Structured, monitor, console,
dashboard, and relay failures remain isolated from the trading path unless the sink is itself an
explicitly required durability contract.

The disk monitor root is authoritative for monitor data. Relays and dashboards are read-only views.
Malformed monitor lines may be skipped with a warning because relay failure must not affect the bot.

## Console Projection

The console is local and operator-facing. It prioritizes information that explains whether the bot
is trading, waiting, degraded, or protecting an existing position:

- fills, position/balance changes, and risk-mode transitions
- create/cancel outcomes and bounded anomalies
- entry/risk gates that explain omitted actions
- health summaries and startup readiness milestones
- held-position trailing, unstuck, and HSL state
- throttled forager-selection summaries

Routine flat-universe diagnostics, high-frequency progress, and debug payload shapes stay in
structured or debug sinks. Account magnitudes can appear on the local console; do not forward
stdout to shared aggregation without a deliberate redaction policy.

Legacy stdlib lines may temporarily serve as fallback when structured console projection is
disabled or unavailable. New work should not duplicate an event in both paths without an explicit
migration reason.

## Fallback Visibility

Trading-critical fallbacks follow `error_contract.md` and include the relevant input/symbol,
reason, source, age when applicable, and consecutive-use count. Unknown debug-profile names fail
visibly rather than being ignored.

## Review Questions

1. Can operators tail INFO for long periods without noise overload?
2. Can they distinguish waiting, degraded operation, and failure?
3. Can developers correlate a decision without exposing secrets or unbounded payloads?
4. Does sink failure remain isolated from trading behavior?
