# Live Overengineering Audit Follow-up

Status: research checkpoint for later continuation

## Objective

Continue reducing live-system complexity where the maintenance and failure cost
is disproportionate to the problem solved, without weakening trading-critical
contracts or mixing experiments into `master`.

This checkpoint combines the architectural audit, the independent external
audit, code review, and limited runtime observation. Runtime evidence was useful
for locating hot paths and noisy failure surfaces, but it was not a long,
version-frozen experiment. Every future change still needs current-code
verification and contract-specific regression coverage.

## Evidence provenance

- The independent external audit inspected
  `ad3da697b8157f5139a874a95349407a1e8d3d35` and was code/docs-only; it did not
  use VPS evidence.
- The churn simplification started from
  `c4d7023faeeb62a71bbfdd8220f903cea949b34b`, after the first two audit follow-up
  PRs were merged.
- Runtime observations were read-only snapshots of representative bots, not a
  controlled A/B run. Host-specific configs, logs, account details, and
  telemetry remain private and are intentionally absent from this repository.
- File locations below are routing hints, not permanent line references. Recheck
  the current target SHA before resuming any item.

## Completed outcomes

1. Cancel-first create deferral was narrowed to symbol and position side in
   hedge mode, and to symbol in one-way mode. Malformed unscoped cancellation
   remains conservatively account-wide.
2. The logging-overhaul resume documents and live-ops backlog were refocused so
   future work starts from the current event pipeline rather than historical
   phase narration.
3. The order-replacement churn gate was reduced to a recent-behavior economy
   filter. Stale-order cancellation and the 0.02% reconciliation tolerance
   remain authoritative.
4. The passive Python planning-availability Cartesian product and its routine
   per-snapshot event were removed. They duplicated readiness evidence, could
   not model actual Rust applicability, and were never enforcement inputs.
5. Authoritative per-surface signature, freshness, and changed-this-cohort
   ownership was consolidated in `FreshnessLedger`. Parallel `Passivbot` maps
   and sets plus both unused generation counters were removed; pending
   post-write confirmation obligations and immutable planning snapshots remain
   separate because they represent distinct contracts.
6. The scalar authoritative-refresh epoch mirror was removed. Refresh,
   planning, reconciliation, snapshot stamping, and event diagnostics now read
   the epoch directly from `FreshnessLedger`; execution-loop `cycle_id` remains
   a separate observability lifecycle.
7. The current architecture-tightening slice removes completed candles from the
   account-wide staged planning barrier. Known live EMA absence is passed to
   strict-by-default Rust as symbol-scoped availability metadata, so only actual
   EMA consumers are suppressed, stale strategy orders remain cancellable, and
   candle-independent reducers can still run.

## Decision rules for further work

Before implementing a simplification, write down:

- the invariant being protected;
- the concrete incident or observed failure it addresses;
- whether the invariant is account-wide, symbol-scoped, or position-side
  scoped;
- which layer owns the decision;
- what should happen when evidence is missing;
- the smallest counterexample that would make the proposed deletion unsafe.

Prefer deletion or consolidation when two or more layers independently infer
the same readiness, freshness, risk, or order-validity fact. Preserve
conservative barriers only where a bad partial action can change account state
or strand stale risk.

Use a PR branch for every production candidate. If exploratory instrumentation
is needed, keep it on a disposable experiment branch or in an offline tool.
The operator can run a reviewed PR checkout on one representative VPS; no
experiment requires freezing `master` for one or two weeks.

## Ranked continuation areas

### 1. Readiness and liveness ownership

Question: do global planning readiness, authoritative-surface freshness,
per-symbol readiness, and execution confirmation independently block the same
condition?

Read:

- `src/live/freshness.py`
- planning snapshot construction and consumers;
- reconciliation account barriers;
- state-change and confirmation settlement;
- live gatekeeper planning documents.

Look for:

- the same stale surface translated into multiple enums, sets, and event
  reasons;
- global blocking where only one symbol or position side is uncertain;
- recovery paths which require every layer to clear separate state;
- diagnostic snapshots that have become enforcement inputs.

Desired output: one owner for each readiness fact, a mapping of downstream
consumers, and a deletion-first proposal. Do not implement until the
account-wide versus scoped safety boundary is explicit.

Current result: `FreshnessLedger` is the sole owner of per-surface freshness
and change facts. The disappeared-self-order symbol-block state machine was
also removed after direct-caller and runtime analysis showed that the existing
full-account `N+1` confirmation barrier always settled first in normal
execution. The generic same-wave symbol latch remains because it also protects
against late websocket changes and uncertain cancellation results. The scalar
authoritative-refresh epoch mirror was removed after verifying that it had no
producer, invalidation rule, or consumer distinct from `FreshnessLedger.epoch`.

### 2. Fill and realized-PnL reconstruction

Question: has exact reconstruction become an account-wide choke point beyond
what loss controls and trailing state actually require?

Read:

- fill ingestion, cache repair, authoritative fetch, and enrichment paths;
- PnL manager and realized-loss inputs;
- position-linked fill-history readiness;
- connector pagination special cases.

Look for:

- repeated historical refetch and widening policies layered on earlier repair
  policies;
- exactness requirements applied to consumers that need only bounded or
  monotonic information;
- one malformed or delayed fill blocking unrelated symbols;
- synthetic accounting whose later reconciliation needs more machinery than
  the original benefit warrants.

Required before change: a consumer-by-consumer contract table. Risk enforcement,
balance accounting, trailing state, diagnostics, and display do not
automatically need identical fidelity.

### 3. HSL replay and restart state

Question: can restart recovery derive the required state from exchange state,
fills, candles, and config with fewer replay modes and transition flags?

Look for:

- overlapping replay/readiness state machines;
- persistence introduced only to preserve an optimization;
- old fallback states kept after newer authoritative paths were added;
- symbol-local uncertainty promoted to account-wide waiting.

Required evidence: deterministic offline restart fixtures covering flat,
existing-position, delayed-fill, and incomplete-candle cases.

### 4. Freshness, revision, and gatekeeper duplication

Question: are multiple timestamp, revision, signature, and epoch systems
protecting the same snapshot boundary?

Map every revision or epoch to:

- its producer;
- its invalidation event;
- its enforcement consumer;
- the unsafe action it prevents.

Delete any token which has no unique invalidation semantics. Keep the
gatekeeper passive until its ownership boundary is narrower than the existing
authoritative guards and its removal plan is known.

Current result: the unused authoritative surface generations were deleted.
Data-packet revisions remain diagnostics-only provenance, pending
confirmations remain cross-wave execution obligations, and planning snapshots
remain immutable handoff proofs; these are not interchangeable with freshness
epochs.

### 5. Create/cancel safeguard pipeline

Question: can the sequence of local deferrals, market checks, cancel-first
scoping, configuration sync, batch limits, connector submission, and
confirmation be expressed as one ordered policy pipeline?

The aim is not a generic framework. A good result is fewer passes, fewer
mutations on order dictionaries, and a single reason for each blocked action.
Protective market orders and stale-order cancellation remain explicit escape
and safety paths.

### 6. Logging and event surface

Resume from:

- `docs/plans/live_logging_overhaul_current_status.md`
- `docs/plans/live_logging_overhaul_plan.md`
- `docs/plans/live_logging_overhaul_progress.md`
- `docs/plans/live_ops_improvement_backlog.md`

Use the audit as a deletion lens:

- consolidate events that encode the same decision at adjacent layers;
- keep persisted events needed for postmortem reconstruction;
- remove console projections that duplicate structured events;
- prevent diagnostic emitter failure from affecting trading;
- avoid new event types until an operator query cannot be answered from the
  existing schema.

### 7. `Passivbot` facade and orchestration ownership

Question: which methods in the main class are genuine orchestration and which
are compatibility forwarding, state ownership, or subsystem implementation?

Extract only around a stable ownership boundary. Line-count reduction alone is
not a success if it creates more cross-module state or callback indirection.

### 8. Candle and EMA readiness

Question: are global candle/EMA completeness requirements stricter than the
Rust strategy inputs for the affected symbol actually require?

Keep float EMA spans and never fabricate a trading input. Seek symbol-scoped
degradation and a single readiness calculation, not broader fallback values.

## Recommended execution sequence

1. Audit readiness/liveness ownership read-only and produce the ownership map.
2. Audit fill/PnL consumers read-only and produce the fidelity table.
3. Choose one deletion candidate with a narrow invariant and offline regression
   fixture.
4. Implement it in a regular PR; review the exact head before any live trial.
5. Run that PR checkout on one representative bot for an overnight observation
   window when runtime evidence is useful.
6. Compare event and action rates with a nearby representative baseline,
   accounting for config scope and market conditions.
7. Merge only after the code-level contract and runtime behavior agree.

Avoid bundling unrelated findings. The audit succeeds when it produces fewer
states, fewer independent authorities, and fewer recovery transitions—not when
it creates a larger abstraction for describing the existing complexity.

## Evidence to collect during a PR trial

Use existing read-only observability wherever possible:

- event counts and reason distributions before and after;
- planning-cycle and execution-wave timelines;
- cancel/create attempts, accepted actions, and ambiguous outcomes;
- readiness transitions and how long each remains blocked;
- CPU, event volume, and repeated remote fetch patterns;
- examples where a scoped uncertainty becomes account-wide.

If existing events cannot answer the narrow hypothesis, first prefer an offline
replay or a temporary experiment-branch probe. Production instrumentation is
justified only when it has a durable operator use after the experiment.

## Stop conditions

Do not simplify a candidate when:

- its safety invariant cannot be stated precisely;
- the proposed replacement invents required trading data;
- a symbol-scoped failure can produce an unsafe account-wide partial action;
- backtest/live behavior would diverge materially;
- validation depends only on a quiet market interval;
- the change merely moves the same state machine to a new module.
