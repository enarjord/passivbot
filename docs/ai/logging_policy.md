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

### Admission Rules

Default INFO console admission is based on operator value, not merely on the producer's level.

- Emit exchange writes and outcomes, fills, position changes, safety actions, and new fatal
  failures immediately.
- Emit gates, fallbacks, connectivity, readiness, mode, and risk state on transition: entered,
  materially changed, recovered, or cleared. Unchanged repeats stay in structured/debug sinks.
- Emit one initial account snapshot. Thereafter, emit balance only for a material wallet/account
  change or an attributable settlement such as realized PnL, funding, fee, deposit, or withdrawal.
  Mark-to-market equity jitter and unchanged snapshot anchors are not console changes.
- Emit startup work as phase milestones. While a phase blocks readiness, bounded progress may appear
  at most once every 30 seconds. Per-symbol warmup, candle-index rebuild, cache maintenance, and
  successful refresh detail are DEBUG after readiness; post-ready maintenance must not use `[boot]`.
- Emit forager selection when the selected set or its availability materially changes. Persistent
  churn receives at most one aggregate every five minutes; candidate and feature detail stays off
  the console.
- Emit one periodic healthy summary per bot at most every 15 minutes. Degradation and recovery are
  immediate and do not wait for that cadence.
- Emit held-position trailing, unstuck, HSL, WEL, and TWEL state on transition or material threshold
  movement, not on every planning cycle.

The structured event remains durable even when its console projection is suppressed. Suppression
must not alter event production, trading decisions, counters, or monitor history.
Each family that uses numeric materiality or hysteresis must define that boundary explicitly and
test values on both sides; the console sink must not invent a generic threshold after emission.

### Incident Projection

The normal console projects an incident as a bounded signature, not a traceback:

1. Emit the first occurrence immediately with component, operation, exception class, status/code
   when safe, affected scope, action, and correlation id.
2. Aggregate equivalent repeats and emit a compact count at most every five minutes. Emit recovery
   once when the condition clears.
3. Keep sanitized tracebacks in the protected developer text/structured diagnostic path. A terminal
   outer-process failure may tell the operator where that detail was retained, but must not dump it
   into the normal console.

Distinct safety transitions and distinct failed exchange writes are never coalesced merely to meet
a volume target.

### Shape And Volume Budget

A normal console record should fit in 240 visible characters and no more than two terminal rows at
160 columns. Summaries use counts, maxima, and at most three samples plus `+N more`; tables, banners,
tracebacks, and unbounded symbol lists are not normal steady-state records.

After readiness, a healthy bot should target:

- at most 60 INFO records and 90 displayed rows per hour
- at most 10 INFO records in any minute, excluding direct exchange actions/outcomes, fills,
  position changes, and safety transitions
- no non-action family above 20% of steady-state INFO without an explicit operator-value rationale

These are review and VPS-smoke acceptance budgets, not hard runtime drop limits. Measure both
timestamped logical records and terminal rows after wrapping. If the budget is exceeded, change the
producer's transition, aggregation, routing, or formatting policy; do not add a global sampler that
can hide unrelated events. Startup is assessed separately because readiness milestones are bursty.

## Fallback Visibility

Trading-critical fallbacks follow `error_contract.md` and include the relevant input/symbol,
reason, source, age when applicable, and consecutive-use count. Unknown debug-profile names fail
visibly rather than being ignored.

## Review Questions

1. Can operators tail INFO for long periods without noise overload?
2. Can they distinguish waiting, degraded operation, and failure?
3. Can developers correlate a decision without exposing secrets or unbounded payloads?
4. Does sink failure remain isolated from trading behavior?
5. Is each INFO record a new operator fact, transition, action, or bounded summary?
6. Do steady-state logical-record and displayed-row measurements satisfy the console budget?
