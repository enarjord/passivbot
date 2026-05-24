# Exchange Connector And Event Contract Plan

## Purpose

Define the next architecture slice for improving Passivbot live diagnostics, exchange
connector discipline, fake-live replay, and onboarding.

The goal is not to turn Passivbot into a broad trading framework. The goal is to make the
existing focused live/backtest/optimization system easier to verify, operate, and extend across
supported exchanges.

## Core Position

Passivbot should borrow Hummingbot-like discipline at the exchange and execution boundary:

- stricter connector contracts
- explicit order/fill/selection event evidence
- exchange conformance fixtures
- better fake-live replay
- cleaner onboarding and readiness checks

Passivbot should not copy Hummingbot's broad strategy/plugin identity. Rust remains the source of
truth for order behavior, risk, unstuck, and backtest behavior. Python events are diagnostics and
evidence, not a second behavioral state machine.

## Non-Goals

- Do not add a generic strategy framework.
- Do not move entry, close, risk, or unstuck behavior out of Rust.
- Do not make event persistence a required source of future trading decisions.
- Do not add local state that changes behavior after restart.
- Do not make Docker the only supported deployment path.
- Do not put Hyperliquid private EVM keys into normal live trading config.

## Working Principles

- Trading decisions must remain reconstructible from exchange state plus config.
- Events should explain what happened, not decide what should happen.
- Missing trading-critical exchange inputs still fail loudly.
- INFO logs stay operator-readable; high-cardinality event detail belongs in structured artifacts or DEBUG/TRACE.
- Connector tests should verify the actual normalized contract Passivbot consumes.
- Broker and builder-fee attribution are exchange-critical behavior and need request-boundary tests.

## Existing Anchors

Current code already has useful pieces to build on:

- `src/exchanges/ccxt_bot.py`
  - hook taxonomy: `can_*`, `_do_*`, `_get_*`, `_normalize_*`, `_build_*`
- `src/live/executor.py`
  - order emit/cancel parent paths
- `src/live/freshness.py`
  - freshness surfaces and symbol blocks
- `src/fill_events_manager.py`
  - canonical fill/PnL ingestion
- `src/exchanges/fake.py`
  - fake exchange scenario execution
- `src/monitor_publisher.py`
  - structured monitor event publishing
- `docs/ai/ccxt_upgrade_workflow.md`
  - CCXT snapshot and fixture drift workflow
- `docs/plans/container_deployment_design.md`
  - container path as a deployment contract, not a replacement architecture

## Event Contract

### Event Role

Events are an append-only diagnostic/audit stream. They should support:

- live incident forensics
- exchange divergence analysis
- fill-to-order and cancel-to-confirmation latency metrics
- forager selection analysis
- fake-live replay assertions
- operator-facing health summaries

Events must not become a hidden control plane.

### Event Envelope

Every structured event should have a small common envelope:

- `event_type`
- `event_ts_ms`
- `exchange`
- `user`
- `symbol`, when applicable
- `position_side`, when applicable
- `side`, when applicable
- `source`
- `context`
- `order_wave_id`, when applicable
- `custom_id`, when applicable
- `exchange_order_id`, when applicable
- `client_order_id`, when applicable
- `qty`, when applicable
- `price`, when applicable
- `reduce_only`, when applicable
- `status`
- `reason`, when applicable
- `raw_ref`, optional pointer to raw/debug payload, not full payload at INFO

Prefer stable field names over exchange-specific aliases. Keep signed-qty conventions exact in
internal fields; use absolute quantities only where exchange payload contracts require it.

### Order Events

Initial event types:

- `order.emit_requested`
  - emitted before an order create call is sent
  - includes Rust/planner order context and why it survived Python execution filters
- `order.emit_accepted`
  - emitted after exchange acknowledgement strongly indicates order creation
  - includes normalized exchange order id/custom id
- `order.emit_failed_ambiguous`
  - emitted when the create call fails or returns a shape that cannot prove non-creation
  - should request authoritative `open_orders` confirmation
- `order.emit_rejected`
  - emitted when the exchange clearly rejected the order and no live order was created
- `order.cancel_requested`
  - emitted before a cancel call is sent
- `order.cancel_confirmed`
  - emitted when cancellation is confirmed or open-order reconciliation proves disappearance
- `order.cancel_failed_ambiguous`
  - emitted when cancellation result is unclear and full authoritative confirmation is needed
- `order.observed_open`
  - emitted when an open-order snapshot/websocket update observes a live order
- `order.observed_missing`
  - emitted when a previously tracked order disappears and must be classified by recent local action, fill evidence, or ambiguity

The first implementation slice should cover `emit_requested`, `emit_accepted`,
`emit_failed_ambiguous`, `cancel_requested`, and `cancel_confirmed` in the existing live executor
paths.

### Fill Events

Fill-event work should preserve the existing fill/PnL contract and add better linkage to order
events where possible:

- `fill.observed`
- `fill.canonicalized`
- `fill.migrated_cache_row`, DEBUG/artifact-level only
- `fill.pnl_repaired`, DEBUG/artifact-level only

Useful fields:

- `fill_id`
- `trade_id`
- `exchange_order_id`
- `client_order_id`
- `custom_id`
- `symbol`
- `position_side`
- `side`
- `qty`
- `price`
- `fee_paid`
- `pnl`
- `net_pnl`
- `source`
- `cache_scope`

The stored `pnl` contract should remain explicit: gross realized price PnL before fees, with fees
stored separately and net derived where displayed.

### Forager Events

Forager events are useful for explaining "why this coin and not that coin." They must be designed
to avoid INFO spam.

Initial event types:

- `forager.selection_snapshot`
  - periodic or changed-set snapshot of the selected universe
- `forager.symbol_approved`
  - symbol enters the candidate/approved set
- `forager.symbol_rejected`
  - symbol leaves or is excluded from the candidate/approved set
- `forager.slot_changed`
  - selected live slot changes from one symbol to another
- `forager.score_snapshot`
  - DEBUG/artifact-level detail for rank, volume, volatility, spread, funding, or other score inputs

INFO should focus on selected-set changes, slot changes, and periodic summaries. Full per-symbol
score detail belongs in JSONL artifacts or DEBUG/TRACE.

## Connector Contract

### Capability Matrix

Each exchange adapter should expose or be testable against a capability matrix:

- REST balance fetch
- REST positions fetch
- REST open orders fetch
- REST trade/fill history fetch
- websocket order updates
- websocket fills/trades, if supported
- hedge mode support
- margin mode support
- leverage configuration support
- reduce-only support
- client order id support
- broker attribution support
- builder-fee attribution support, if applicable
- market metadata completeness
- rate-limit model
- pagination model for fills/PnL

Capability absence is acceptable only when visible and handled by an explicit path.

### Required Normalized Fields

Connector tests should prove that Passivbot receives complete normalized fields for the surfaces it
uses:

- market metadata: `symbol`, `base`, `quote`, `settle`, `min_qty`, `min_cost`, `qty_step`,
  `price_step`, `contract_size`, `max_leverage`, margin capability
- positions: `symbol`, `position_side`, signed/internal size semantics, entry price, margin mode if available
- open orders: `symbol`, `side`, `position_side`, `qty`, `price`, `reduce_only`, custom/client id,
  exchange order id, timestamp
- order create result: enough data to classify accepted, clearly rejected, or ambiguous
- order cancel result: enough data to classify confirmed, clearly rejected, or ambiguous
- fills/trades: `symbol`, `side`, `position_side`, `qty`, `price`, fee, PnL when available,
  exchange order id, client/custom id, timestamp

Missing required fields in trading-critical paths should raise or force explicit confirmation, not
silently default.

### Conformance Tests

Add a connector conformance suite that can run offline against fixtures:

- generic CCXTBot fixture replay
- exchange-specific fixture replay
- signed request construction checks for broker/builder attribution
- open-order normalization fixtures
- position normalization fixtures
- fill/PnL normalization fixtures
- ambiguous create/cancel result fixtures
- websocket order update normalization fixtures where supported

When live snapshots are needed, keep them sanitized and out of the default test suite unless they
become durable fixtures.

## Fake-Live Replay

Fake live should become a deterministic replay harness for live behavior, not just a simple fake
exchange.

Scenario files should support:

- candles and tickers
- initial balance, positions, fills, and open orders
- resting order fills
- delayed order acknowledgement
- ambiguous create/cancel failures
- rejected orders
- websocket outage/reconnect simulation
- stale open-order snapshots
- delayed fills/PnL availability
- rate-limit and timeout injection
- forager universe changes

Replay tests should assert:

- final account/order state
- structured event sequence
- authoritative refresh barriers
- no duplicate unsafe order creation
- expected symbol blocks when state is ambiguous
- expected recovery once authoritative surfaces refresh

This gives better coverage for live bot behavior under synthetic data without requiring exchange
network access.

## Doctor And Onboarding

Add `passivbot doctor` as the operator-facing readiness layer after the contract work has a stable
base.

Suggested checks:

- config schema and migrated defaults
- account credentials load
- exchange connectivity
- required permissions
- balance/positions/open-orders/fills fetchability
- market metadata completeness
- hedge mode, leverage, margin mode readiness
- websocket support and fallback path
- broker attribution readiness
- builder-fee readiness, where relevant
- cache paths and writable directories
- fake-live dry run for the selected config, when feasible

Doctor should explain blockers in operator language and should not mutate account state unless a
subcommand explicitly says it will.

## Hyperliquid Builder Fee

Hyperliquid builder-fee support should be treated as a separate onboarding/setup flow.

Recommended shape:

- do not require the private EVM key in normal live trading config
- keep trading on API agent keys
- provide a dedicated setup command for builder-fee authorization
- make the setup command explicit about signing scope and risk
- store only non-secret readiness/status artifacts where possible
- let `passivbot doctor` verify whether the account is builder-fee ready
- add request/signature tests around the Hyperliquid builder-fee path

This keeps a sensitive one-time signing workflow separate from normal unattended live trading.

## Docker Position

Docker is useful, but should not be the first step.

`passivbot doctor` should come before a stronger Docker push because it clarifies runtime
assumptions: secrets, config, cache paths, writable directories, exchange connectivity, and
permissions. Once doctor exists, Docker can consume the same readiness contract.

Recommended stance:

- keep native local/VPS usage first-class
- keep Docker/Compose as a supported deployment target
- avoid making Docker the only official route
- use the existing container deployment design as the basis for future implementation

## Rollout Plan

### Phase 0: Audit Existing Event Surfaces

- Inventory current monitor events, logs, order wave summaries, fill events, and forager logs.
- Identify fields already available in `src/live/executor.py`, `src/passivbot.py`,
  `src/fill_events_manager.py`, and forager selection code.
- Decide whether the first event sink is monitor publisher, JSONL, logs, or a small shared helper
  feeding multiple outputs.

### Phase 1: Event Envelope And Sink

- Define a small event helper with the common envelope.
- Add tests for required envelope fields and JSON-serializable output.
- Keep INFO log emission separate from full event persistence.
- Ensure events can be disabled or bounded if artifact volume becomes high.

### Phase 2: Order Emit/Cancel Events

- Add order create requested/accepted/ambiguous/rejected events.
- Add order cancel requested/confirmed/ambiguous/rejected events.
- Link events to order wave summaries.
- Add tests for accepted, rejected, and ambiguous exchange responses.

### Phase 3: Fill Event Linkage

- Emit structured fill observed/canonicalized events.
- Link fills to order ids/custom ids where available.
- Keep cache migration/repair detail out of INFO.
- Add regression tests for fee/PnL contract visibility.

### Phase 4: Forager Selection Events

- Emit selected-set and slot-change events.
- Keep rank/score detail at DEBUG or artifact level.
- Add tests that repeated unchanged rankings do not spam INFO.
- Make event artifacts sufficient to explain cross-exchange selection divergence.

### Phase 5: Connector Conformance Suite

- Define base fixtures for markets, positions, open orders, create results, cancel results, and fills.
- Add one generic `CCXTBot` conformance path.
- Add one high-value exchange first, likely Hyperliquid or Bybit.
- Extend exchange by exchange as bugs or integration work touch those adapters.

### Phase 6: Fake-Live Replay Upgrade

- Extend fake scenarios with ambiguous order/cancel/fill timing cases.
- Assert structured event sequences.
- Add replay scenarios for duplicate-order prevention and delayed authoritative confirmation.
- Add replay scenarios for forager universe changes.

### Phase 7: Doctor

- Start with read-only checks only.
- Add exchange/account/config readiness checks.
- Add broker attribution checks.
- Add Hyperliquid builder-fee readiness check after the dedicated setup flow exists.

### Phase 8: Docker As Deployment Consumer

- Revisit Docker/Compose after doctor can validate runtime assumptions.
- Keep the container entrypoint thin and CLI-based.
- Do not fork config behavior for containers.

## First Implementation Slice

The smallest useful implementation after this plan:

1. Add a shared structured live event helper.
2. Emit order create/cancel requested and terminal/ambiguous events from the existing live executor.
3. Add unit tests for accepted, rejected, and ambiguous create/cancel outcomes.
4. Keep all behavior unchanged except diagnostics.

This slice improves observability immediately while avoiding strategy or exchange behavior changes.

## Open Questions

- Should the durable event artifact be JSONL under logs, monitor publisher output, or both?
- Should event ids be deterministic from order wave/custom id/timestamp or generated per process?
- How much raw exchange payload should be retained, and where, without leaking secrets?
- Which exchange should be the first conformance target after the generic fake/CCXT path?
- Should `passivbot doctor` live under `passivbot doctor` only, or also expose exchange-specific
  subcommands such as `passivbot doctor hyperliquid`?
