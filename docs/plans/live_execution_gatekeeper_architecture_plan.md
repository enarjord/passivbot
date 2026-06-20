# Live Execution Gatekeeper Architecture Plan

## Status

Draft architecture/spec for a future v8 live-bot refactor.

This is not an implementation checklist for one large rewrite. It defines the target shape and a
safe migration path for making live execution easier to inspect, test, and reason about.

The first implementation commitment is intentionally narrower than the full target architecture:
build the event/data-packet spine and immutable per-cycle snapshot boundary first. Gatekeeper
enforcement should remain conditional until the planning-availability contract is explicit and
tested.

## Purpose

Split the live bot into narrow components with explicit data contracts and structured decision
evidence:

1. Data handlers fetch, normalize, and packetize one data family.
2. Snapshot builder freezes a per-cycle live snapshot with data and metadata.
3. Rust receives the order-planning input and returns ideal orders.
4. Action planner reconciles ideal orders with exchange-open actual orders.
5. Gatekeeper approves, defers, rejects, or marks each planned action as already satisfied.
6. Executor performs only gate-approved exchange writes and records request/response evidence.

The primary goal is live safety and diagnosis. A live incident should be reconstructible from
structured events without reverse-engineering interleaved logs or hidden mutable state.

## Reviewer Calibration

External review agreed that the design targets real live-safety problems, but the value is
front-loaded:

- Highest value / lowest behavior risk:
  - structured event envelope
  - `DataPacket` metadata
  - immutable `LiveSnapshot` used through the whole cycle
  - `planning_unavailable` diagnostics
- Higher risk / later:
  - replacing scattered execution checks with enforced gatekeeper decisions
  - semi-lazy refresh scheduling that changes when data is refreshed

The key unresolved safety gap is downstream gatekeeper visibility. A gatekeeper can only evaluate
actions Rust emitted. If stale or missing snapshot inputs cause Rust not to emit an order it would
have emitted with fresh data, the gatekeeper cannot recover that missing action. The mitigation is
not gatekeeping alone; it is an explicit planning-availability contract before enforcement.

This means the first useful plateau is Phases 1-2 plus a planning-availability audit surface. The
full gatekeeper should be built only after that plateau is reviewed and the stale-input policy is
settled.

## Non-Goals

- Do not move entry, close, risk, HSL, or unstuck order behavior out of Rust.
- Do not create Python fallback strategy logic around stale Rust inputs.
- Do not make event persistence a source of future trading decisions.
- Do not add local runtime state that changes behavior after restart and cannot be rederived from
  exchange state plus config.
- Do not log full raw private exchange payloads at INFO by default.
- Do not make this a generic trading framework or plugin architecture.

## Core Principle

Use a stateful shell with a mostly pure core.

Stateful shell:

- exchange API clients
- caches
- latest data packets
- background refresh tasks
- rate-limit budget
- raw/debug artifact storage
- structured event sink

Pure or near-pure core:

- snapshot construction from known packet revisions
- ideal-vs-actual action planning
- gatekeeper decision rules
- executor request construction from approved actions

Each execution cycle must operate on an immutable snapshot. Background refreshes may finish while a
cycle is planning, but their results must update the state store for the next cycle, not mutate the
current cycle.

## High-Level Flow

```text
Data handlers
    -> DataPacket store
    -> SnapshotBuilder
    -> Rust orchestrator
    -> ActionPlanner
    -> Gatekeeper
    -> Executor
    -> Exchange
```

Each stage emits structured events with `cycle_id`, `snapshot_id`, and, when applicable,
`planned_action_id`.

## Data Categories

Each data category should have a dedicated handler.

### Static Bot Params

Source:

- prepared config
- coin overrides
- compiled Rust/PyO3 order-planning fields

Expected metadata:

- config path
- config hash or revision
- prepared-config timestamp
- side enablement
- pside-specific bot/risk/HSL fields

Notes:

- Static config is always present after startup validation.
- Runtime config/list reloads should create a new config revision.

### Coin Universe And Market Metadata

Source:

- prepared config approved/ignored lists
- live approved/ignored coin list files
- exchange market metadata

Expected metadata:

- config/list revision
- list read timestamp
- market metadata fetch timestamp
- symbol eligibility reason
- min qty, min cost, qty step, price step, contract multiplier, leverage/margin capability
- active/inactive market status

Policy:

- Stale or unreadable eligibility state blocks affected initial entries.
- Stale eligibility state must not block protective management of existing positions.
- If a held coin is removed from approved coins or added to ignored coins, the normal target is
  `graceful_stop` when `auto_gs=true`.
- A newly approved coin may enter the forager universe only after required market metadata and
  warmup surfaces are available.

### Positions

Source:

- authoritative exchange position fetch

Expected metadata:

- call start timestamp
- response timestamp
- exchange server timestamp if available
- raw payload reference
- normalized positions
- parsing/normalization warnings
- account-state epoch

Policy:

- Required before any exchange action.
- If positions cannot be refreshed, the bot may emit diagnostics, but must not execute creates or
  cancels.

### Open Orders

Source:

- authoritative exchange open-order fetch

Expected metadata:

- call start timestamp
- response timestamp
- raw payload reference
- normalized open orders
- order ownership classification
- malformed/ambiguous rows
- account-state epoch

Policy:

- Required before any exchange action.
- Malformed open-order rows must not be silently dropped before planning.
- Manual/external rows without Passivbot client order ids are not automatically corruption. If a row
  is otherwise valid, classify it as external/manual and log it clearly.

### Balance

Source:

- authoritative exchange balance fetch

Expected metadata:

- call start timestamp
- response timestamp
- raw payload reference
- raw balance
- effective/snapshotted balance
- account-state epoch

Policy:

- Required before any exchange action, including panic or reduce-only actions.
- Low-balance policy should block both cancellations and creations unless a future explicit
  exception is designed and tested.

### Tickers / Market Snapshot

Source:

- `fetch_ticker`, `fetch_tickers`, exchange-specific endpoints, or approved websocket cache

Expected metadata:

- call start timestamp
- response timestamp
- bid, ask, last
- source strategy
- symbol scope
- age
- exchange-specific fallback label if applicable

Policy:

- Required for symbols being traded.
- Required for symbols with positions or open orders.
- Required for initial-entry candidates before posting.
- Must not silently synthesize bid/ask/last except through a documented exchange-specific
  exception.

### Candlesticks 1m And 1h

Source:

- candlestick manager
- local cache
- remote OHLCV fetch

Expected metadata:

- requested window
- covered window
- latest completed timestamp
- synthetic/bounded gap count
- open-ended tail age
- remote calls made
- cache health
- EMA readiness for required spans

Policy:

- Required for order classes that depend on strategy indicators, trailing extrema, HSL replay, or
  forager scoring.
- Stale candles for flat symbols must not block protective management of held symbols.
- Candle unavailability must be represented explicitly. Do not convert missing candle windows into
  neutral EMA inputs.

### Fill Events

Source:

- exchange fill/trade/PnL fetchers
- fill event cache

Expected metadata:

- requested lookback
- covered start/end
- history scope
- cache contract version
- raw payload references
- canonical fill rows
- repair/quarantine status
- external/manual fill classification

Policy:

- Required for HSL, max realized loss, auto-unstuck, trailing-anchor reconstruction, and other
  fill/PnL-dependent gates.
- Corrupted/unreadable fills should use bounded retry/repair and visible degraded state.
- Do not replace missing fill/PnL history with silent zero or empty defaults.

## DataPacket Contract

Every handler should return or update a packet with a common envelope:

```text
kind
scope
revision
cycle_hint
call_started_ts_ms
response_received_ts_ms
exchange_server_ts_ms
source
value
raw_ref
raw_hash
coverage
freshness
quality
warnings
errors
```

`scope` should be one of:

- `global`
- `pside`
- `symbol`
- `symbol+pside`

`freshness` should be structured, not a bool:

```text
status: fresh | stale | missing | degraded | corrupt | unknown
age_ms
max_age_ms
reason
```

`coverage` is category-specific. Examples:

- fill events: covered start/end, scope all/window
- candles: timeframe, first/last timestamp, missing gaps, latest completed timestamp
- tickers: bid/ask/last availability and age
- market metadata: symbol validity and exchange capability completeness

Raw payload handling:

- Use `raw_ref` or `raw_hash` in normal event streams.
- Store full raw payloads only in debug artifacts or explicit trace mode.
- Redact secrets and high-risk account identifiers.

## SnapshotBuilder Contract

SnapshotBuilder produces two linked artifacts:

1. `LiveSnapshot`: metadata-rich immutable packet collection.
2. `RustInput`: the numeric/structured payload Rust currently consumes.

Rules:

- Account-critical packets are hard barriers before any exchange-write cycle:
  positions, open orders, balance.
- The snapshot may include stale/noncritical packets, but their status must be explicit.
- Do not fabricate required Rust inputs with neutral values.
- If a surface is unavailable for a symbol/order class, represent that as planning-unavailable
  metadata.
- Background refresh tasks may be scheduled while building the snapshot, but their results belong
  to a later snapshot revision.

Important failure mode:

If stale or missing data causes Rust not to emit an ideal order it would have emitted with fresh
data, the gatekeeper cannot recover that missing action. Therefore SnapshotBuilder must emit
planning-completeness diagnostics, not only rely on post-Rust gatekeeping.

Diagnostic construction and emission must be isolated from live control flow. A diagnostic failure
may remove or degrade observability for that cycle, but it must not block account refresh,
planning-snapshot capture, staged-defer logging, or protective panic planning.

Target diagnostic examples:

```text
planning_unavailable: symbol=XMR/USDT:USDT pside=long order_class=initial_entry reason=stale_1m_candles
planning_unavailable: symbol=BTC/USDT:USDT pside=long order_class=hsl_check reason=fill_history_degraded
```

## Planning Availability Contract

Planning availability is the bridge between snapshot freshness and Rust output. It must answer:

```text
For this snapshot, which symbol+pside+order_class decisions were evaluable?
Which were explicitly unavailable, degraded, or skipped before/inside Rust planning?
```

This is required because missing ideal orders are otherwise ambiguous. "Rust emitted no order" may
mean either:

- the strategy genuinely wanted no order, or
- the bot lacked fresh enough data to evaluate that order class.

Initial order classes:

- `initial_entry`
- `risk_increasing_entry`
- `take_profit_close`
- `trailing_close`
- `unstuck_close`
- `wel_twel_reduce_close`
- `hsl_panic_close`
- `entry_cancel`
- `protective_close_cancel`

Initial availability values:

- `available`
- `unavailable`
- `degraded`
- `not_applicable`
- `unknown`

Every non-`available` value needs a reason code and packet revision evidence:

```text
PlanningAvailability
  cycle_id
  snapshot_id
  symbol
  position_side
  order_class
  status
  reason_code
  required_surfaces
  packet_revisions
```

Open policy choice:

- For order classes whose Rust input cannot safely represent missing data, exclude or block the
  affected symbol/order class before Rust and emit `planning_unavailable`.
- For order classes where stale values are useful for diagnostics only, carry the stale packet in
  `LiveSnapshot` but do not let it satisfy the requirement matrix.
- Do not pass fabricated neutral defaults to Rust to force a shape.

Gatekeeper enforcement must not be treated as complete until this contract is implemented at least
in Python for live-only data surfaces, and reviewed for the Rust-owned strategy surfaces.

The passive implementation reports live-surface availability derived from the frozen snapshot for
each `(symbol, position_side, order_class)` record. It evaluates only diagnostic availability; it
must not enforce, skip, or approve live trading decisions until a later gatekeeper phase explicitly
consumes it with tests.

Availability records should name canonical readiness requirements, not raw fetch artifacts. For
example, `canonical_candles` means "the live candle handler has a continuous strategy-ready candle
series for this symbol/side through the required horizon." It does not mean "the latest REST candle
row has already appeared." Some exchanges publish completed candles late or omit no-trade candles.
The live candle handler may synthesize zero-volume candles as `[prev_close, prev_close, prev_close,
prev_close, 0]`, use them for the canonical series, avoid persisting them, and later replace them
from overlapping REST backfills. Those synthesized rows may satisfy `canonical_candles` when the
previous close is known, the gap is bounded by the candle handler policy, continuity is preserved,
and repair/backfill remains scheduled and observable. They should fail readiness only when the
canonical series cannot prove continuity, the previous close is unavailable, the gap exceeds policy,
or overlap repair repeatedly fails.

The current Python-side passive matrix is:

- `hsl_panic_close`: account surfaces plus the action symbol's market prices. It does not require
  candles, EMAs, or fill history.
- `take_profit_close`: account surfaces plus the action symbol's market prices. The non-trailing TP
  contract does not require fill history. It requires `canonical_candles` only when the planned
  action's close threshold uses volatility weighting; the current passive matrix cannot inspect that
  config/rationale yet, so enforcement must add conditional tests before consuming this class.
- `trailing_close`: account surfaces, the action symbol's market prices, canonical candles, and fill
  history. A fill resets trailing state, and trailing evaluation needs the candle span from the last
  symbol+side fill through the last strategy-ready completed minute.
- `unstuck_close`: account surfaces, the action symbol's market prices, canonical candles, and fill
  history. Current Rust unstuck close pricing uses EMA bands, and fill history feeds
  realized-PnL/loss-budget inputs.
- `wel_twel_reduce_close`: account surfaces, the action symbol's market prices, and fill history for
  `max_realized_loss_pct`. TWEL/WEL auto-reduce is portfolio/per-position repair, not trailing or
  panic, and does not require candles unless the actual Rust order construction path later consumes
  volatility/strategy indicators.
- `initial_entry`, `risk_increasing_entry`: account surfaces, the action symbol's market prices,
  canonical candles, and fill history. Canonical candles are the current live proxy for
  strategy/EMA/volatility readiness; fills feed current live entry cooldown, trailing reset, and
  realized-PnL/unstuck inputs. A separate strategy-surface stamp remains future work.
- `entry_cancel`: balance, positions, and open orders.
- `protective_close_cancel`: balance, positions, and open orders.

Stale or missing symbol-scoped surfaces affect only records that require that symbol's surface.
For example, unavailable canonical candles for a flat candidate symbol do not make protective or
normal close availability unavailable for a different positioned symbol. Account surfaces and fill
history remain global in the current snapshot contract; the target enforcement contract should
scope fill readiness by symbol+side where the data model supports it.

Monitor `planning_unavailable` events are a throttled operator-facing diagnostic emitted when
staged planning is deferred. They are useful evidence that a loop could not plan, but they are not
the complete future internal `PlanningAvailability` ledger. The future ledger must be able to
represent every symbol+pside+order-class decision, including unavailable decisions that are not
emitted to the monitor because of throttling.

`snapshot.built` monitor payloads should stay compact and emit `PlanningAvailability.summary()`.
Full record sets are available through the internal `PlanningAvailability.to_dict()` path and tests,
not emitted in routine monitor snapshots.

## Rust Orchestrator Contract

Rust remains the source of ideal order behavior.

Short-term:

- Keep the current Rust input shape where practical.
- Carry metadata parallel to Rust input in `LiveSnapshot`.
- Let gatekeeper infer required surfaces from order type, symbol, pside, and snapshot metadata.

Longer-term:

- Add Rust planning diagnostics where Rust can cheaply explain suppressed or unavailable decisions.
- Consider returning per-symbol/per-pside planning completeness for order classes Rust evaluated.
- Keep backtest/live parity contracts explicit when live-only freshness decisions differ.

Rust must not become responsible for exchange I/O freshness policy. It can expose dependencies and
diagnostics; Python gatekeeper owns exchange-write eligibility.

## ActionPlanner Contract

ActionPlanner reconciles ideal Rust orders against actual open exchange orders and produces a full
action plan.

Inputs:

- ideal orders from Rust
- normalized open orders packet
- position packet
- symbol market metadata
- replacement tolerance config
- ownership classification

Outputs:

```text
ActionPlan
  cycle_id
  snapshot_id
  ideal_order_refs
  actual_order_refs
  planned_actions[]
  retained_orders[]
  satisfied_existing[]
```

Action types:

- `create`
- `cancel`
- `replace`
- `retain`
- `satisfied_existing`
- `ignore_external`
- `manual_review_required`

Every planned action must include rationale:

```text
planned_action_id
symbol
position_side
order_side
order_type
qty
price
reduce_only
source_ideal_id
source_actual_id
reason_code
reason_detail
replacement_group_id
dependencies
```

Replacement should be modeled as a transaction, not as unrelated cancel/create actions:

```text
replace
  cancel old actual order
  create replacement ideal order
  sequencing policy
  duplicate-risk policy
```

This is required for same-cycle replace safety.

## Gatekeeper Contract

Gatekeeper evaluates concrete planned actions. It does not calculate strategy intent and it does
not perform exchange writes.

Input:

- `ActionPlan`
- `LiveSnapshot`
- freshness contract table
- exchange capability metadata
- runtime safety state

Output:

```text
GateDecision
  planned_action_id
  decision
  reason_code
  reason_detail
  required_surfaces
  missing_surfaces
  stale_surfaces
  degraded_surfaces
  packet_revisions
  severity
```

Decision values:

- `approved`
- `deferred`
- `rejected`
- `fatal`
- `satisfied_existing`

Use `deferred` for temporary missing/stale data that should trigger refresh and retry. Use
`rejected` for policy blocks. Use `fatal` for corrupted account-critical state or invariant
violations where continuing the cycle would be unsafe.

Gatekeeper rules must be order-class-specific. Example classes:

- initial entry
- risk-increasing entry / DCA
- take-profit close
- trailing close
- unstuck close
- WEL/TWEL reduce-only close
- HSL panic close
- cancel stale Passivbot entry
- cancel stale Passivbot close
- cancel malformed owned order
- ignore valid external/manual order

Cancellations need separate freshness rules. Canceling a stale owned entry may not need candles.
Canceling a protective reduce-only close may be unsafe unless the replacement/close rationale is
clear.

## Freshness Requirement Matrix

Initial target matrix:

| Requirement | Account Barrier | Initial Entry | DCA Entry | Non-Trailing TP | Trailing Close | Unstuck Close | WEL/TWEL Close | HSL Panic | Entry Cancel | Protective Close Cancel |
|-------------|-----------------|---------------|-----------|-----------------|----------------|---------------|----------------|-----------|--------------|--------------------------|
| config params | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| approved/ignored lists | no | yes | yes | no | no | no | no | no | no | no |
| market metadata | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| positions | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| open orders | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| balance | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| market prices | no | symbol | symbol | symbol | symbol | symbol | symbol | symbol | no | maybe |
| canonical candles | no | strategy | strategy | volatility-weighted | symbol+pside | symbol+pside | no | no | no | maybe |
| fill history | no | symbol+pside | symbol+pside | no | symbol+pside | yes | loss gate | no | no | maybe |

This table is intentionally not final. The implementation should turn it into explicit test cases.

Terms:

- `yes`: required globally.
- `symbol`: required for the action symbol.
- `symbol+pside`: required for the action symbol and position side.
- `strategy`: required if the action class depends on strategy indicators for that symbol/pside.
- `volatility-weighted`: required when the planned action's configured volatility weights are nonzero.
- `maybe`: depends on configured risk/HSL/trailing/unstuck mode.
- `loss gate`: required when the action is subject to `max_realized_loss_pct`.

## Semi-Lazy Refresh Model

The live loop may schedule background refreshes while using the latest known packet store.

Required constraints:

- Account-critical packets must be fresh before exchange writes.
- Background task completion must not mutate the current frozen snapshot.
- Refresh scheduling must be bounded and deduplicated by `(kind, scope)`.
- Active position/open-order symbols must have higher priority than flat broad-universe symbols.
- Stale broad forager candles must not block held-symbol protective management.
- Missing held-symbol data should trigger high-priority refresh and visible degraded state.

Suggested priority order:

1. Account-critical surfaces: positions, open orders, balance.
2. Symbols with positions.
3. Symbols with open orders.
4. Symbols under HSL/panic/cooldown supervision.
5. Forager selected slots and near-selected candidates.
6. Broad approved universe warmup.

## Executor Contract

Executor is intentionally narrow.

Inputs:

- gate-approved create/cancel/replace transactions
- exchange adapter
- request construction helpers

Responsibilities:

- construct exchange payload
- record send timestamp
- send request
- record response timestamp
- normalize response
- classify success, clear rejection, or ambiguous failure
- emit execution events
- request authoritative refresh when result is ambiguous

Executor must not:

- decide strategy intent
- silently override gatekeeper decisions
- create orders absent from the action plan
- cancel orders absent from the action plan

## Event Stream Contract

Events are append-only evidence. They are not a control plane.

Common envelope:

```text
event_type
event_ts_ms
monotonic_ts_ms
cycle_id
snapshot_id
plan_id
planned_action_id
exchange
user
symbol
position_side
order_side
source
status
reason_code
raw_ref
```

Initial event types:

- `cycle.started`
- `cycle.finished`
- `remote_call.started`
- `remote_call.succeeded`
- `remote_call.failed`
- `data_packet.updated`
- `data_packet.degraded`
- `snapshot.built`
- `planning_unavailable`
- `rust_orchestrator.returned`
- `action.planned`
- `action.retained`
- `gate.approved`
- `gate.deferred`
- `gate.rejected`
- `gate.fatal`
- `execution.create_sent`
- `execution.create_succeeded`
- `execution.create_failed`
- `execution.cancel_sent`
- `execution.cancel_succeeded`
- `execution.cancel_failed`
- `execution.ambiguous`
- `state_refresh.requested`

INFO logs should stay operator-readable. High-cardinality per-symbol and raw payload detail belongs
in JSONL artifacts or DEBUG/TRACE.

## Migration Plan

### Phase 0: Contracts Only

- Add this plan and review with external agents.
- Record the agreed scope split: Phases 1-2 are green-lit as a behavior-preserving plateau;
  gatekeeper enforcement is conditional on planning-availability work.
- Identify current call sites for snapshot building, Rust input building, reconciliation, gates, and
  executor writes.
- Define initial `DataPacket`, `LiveSnapshot`, `PlannedAction`, and `GateDecision` types in tests
  or docs before changing runtime behavior.

Exit criteria:

- No runtime behavior change.
- Current live-safety tests still pass.
- Open questions are documented.

### Phase 1: Event And Packet Scaffolding

- Wrap existing data refresh paths with packet metadata.
- Emit structured events behind a debug/config flag.
- Keep existing logs.
- Do not change gating behavior.

Exit criteria:

- Existing behavior unchanged.
- Events can reconstruct one normal execution cycle in fake-live.
- No private raw payloads emitted at INFO.

### Phase 2: Immutable Snapshot Boundary

- Freeze a `LiveSnapshot` per execution cycle.
- Keep background refresh completions out of the current cycle.
- Carry packet revisions into Rust-input construction.
- Use the frozen snapshot through Rust input building and subsequent reconciliation/execution
  diagnostics where practical.
- Emit initial planning-unavailable diagnostics for missing/stale non-account surfaces.
- Isolate diagnostic builders and emitters so diagnostics cannot change account refresh,
  snapshot capture, staged defer, or protective panic control flow.

Exit criteria:

- Fake-live can assert packet revisions and snapshot ids are stable through a cycle.
- Account-critical failures still block exchange writes.
- The bot can explain which snapshot packet revisions were used for one normal fake-live cycle.
- Diagnostic failures cannot block account refresh, staged-defer logging, normal snapshot capture,
  or protective panic snapshot capture.

### Phase 2.5: Planning Availability Contract

- Define `PlanningAvailability` records for symbol+pside+order-class evaluation.
- Emit passive availability diagnostics before gatekeeper enforcement.
- Decide, per order class, whether stale values are excluded from Rust input, included only as
  diagnostic snapshot data, or allowed as an explicit fallback.
- Add tests for silent-omission risks where missing data prevents order-class evaluation.

Exit criteria:

- "No ideal order" can be distinguished from "order class unavailable" for the initial high-risk
  classes.
- The stale-input policy is explicit enough for a reviewer to audit.
- Gatekeeper enforcement can depend on availability records rather than assuming Rust evaluated
  every order class.

### Phase 3: Explicit Action Planner

- Make reconciliation emit `PlannedAction` records for create/cancel/replace/retain.
- Record retain/satisfied-existing reasons, not only create/cancel decisions.
- Represent replace as a transaction with sequencing metadata.

Exit criteria:

- Existing order orchestration tests pass.
- New tests cover retain, replace, unknown external/manual order, and malformed owned order cases.

### Phase 4: Gatekeeper Audit Mode

- Implement the gatekeeper rule table in audit mode.
- Compare gate decisions against current behavior.
- Log divergences without changing execution.

Exit criteria:

- Divergences are reviewed and classified as intended future changes or implementation bugs.
- Gatekeeper output is stable enough for fake-live assertions.

### Phase 5: Gatekeeper Enforcement

- Route creates/cancels/replaces through gatekeeper decisions.
- Executor submits only approved actions.
- Deferred actions trigger targeted refresh scheduling.
- Enforce action classes incrementally, starting with the least ambiguous classes.

Exit criteria:

- Live-safety regression suite passes.
- Fake-live scenarios cover stale candles, stale tickers, missing fills, corrupted fills, manual
  orders, low balance, panic closes, and replacement transactions.

### Phase 6: Semi-Lazy Refresh Scheduler

- Add bounded background refresh scheduler.
- Prioritize active/held symbols over broad flat universe.
- Dedupe in-flight refreshes.
- Emit remote-call budget events.

Exit criteria:

- Startup and steady-state remote-call pressure are measurable.
- Active symbol safety is not budget-starved.
- Broad forager warmup progresses without blocking protective management.

## Test Plan

Use fake-live first, then targeted unit tests around pure components.

Required cases:

- Missing positions blocks all exchange writes.
- Missing open orders blocks all exchange writes.
- Missing balance blocks all exchange writes.
- Stale candles for flat broad-universe coin do not block held-symbol close management.
- Stale held-symbol candles block strategy-dependent entries/closes with explicit reason.
- HSL panic can proceed without candle freshness when account-critical and symbol price surfaces are
  fresh.
- Corrupted fill history triggers bounded repair/degraded decision, not silent zero history.
- Valid manual/external order without Passivbot client id is accepted as external/manual and logged.
- Replacement transaction does not duplicate create when cancel/create sequencing is uncertain.
- Gatekeeper emits approved/deferred/rejected/fatal reasons for each planned action.
- Executor ambiguous create/cancel response requests authoritative refresh.
- Event stream can reconstruct a full fake-live cycle.

## Open Questions

1. Should Rust return planning completeness diagnostics, or should Python infer all unavailable
   order classes from snapshot metadata?
2. Should `LiveSnapshot` include stale values for diagnostics only, or should stale values be
   excluded from Rust input for affected symbols/order classes?
3. How much raw payload retention is acceptable by default for live accounts?
4. Should gatekeeper rules be represented as code-only tables, JSON-like data, or typed Python
   objects?
5. Should background refresh scheduling live in a generic manager or remain split by data family?
6. Which event stream should be authoritative for monitor/UI consumers: existing monitor publisher,
   a new JSONL file, or both?
7. How should websocket freshness events integrate with REST packet revisions?

## First Implementation Slice Recommendation

Start with event and packet scaffolding around existing paths, not behavior changes.

The smallest valuable slice:

1. Define common event envelope.
2. Define `DataPacket` metadata model.
3. Wrap positions, open orders, and balance refresh packets.
4. Emit `snapshot.built` and `data_packet.updated` events in fake-live.
5. Add tests proving the current cycle uses frozen packet revisions.
6. Emit minimal `planning_unavailable` events for account-critical or obviously stale non-account
   surfaces without changing execution.
7. Add passive planning-availability reporting and diagnostic-isolation tests before any
   gatekeeper enforcement work.

This creates the diagnostic spine without changing order behavior. Gatekeeper enforcement should
wait until the event, snapshot, and planning-availability boundaries are visible and testable.
