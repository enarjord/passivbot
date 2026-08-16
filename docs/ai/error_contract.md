# Trading-Critical Failure And Degradation Contract

This contract applies to exchange state, strategy indicators, risk inputs, fill/PnL history, order
construction, and exchange writes. It distinguishes failure of one operation from failure of the
whole bot.

## Failure Vocabulary

- **propagate**: a lower layer preserves the exception and actionable context for caller policy.
- **unavailable**: a required surface is explicitly absent or unusable; no neutral value is invented.
- **defer**: only the affected action or order class is postponed pending required inputs.
- **fail closed**: an action that cannot be evaluated safely is prevented.
- **degraded**: a documented bounded fallback is active and observable.
- **fatal**: startup or the process must stop because no safe caller policy remains.

“Fail loudly” means propagate, expose unavailability, or stop at the contract boundary appropriate
to the caller. It does not mean every failed symbol fetch must crash the whole bot.

## Required Behavior

1. Required inputs must be complete before their consumer runs.
2. Exchange fetch methods propagate exceptions; their callers own retry, restart, or scoped-defer
   policy.
3. Missing inputs block only the symbols and order classes that require them unless an
   account-critical surface is missing.
4. A fallback is permitted only when this contract or a task-approved feature contract defines its
   source, bounds, observability, and tests.
5. If an allowed fallback source is itself unavailable, fail closed or propagate according to the
   consumer boundary.
6. Validate a produced trading decision before downstream policy consumes it. A malformed Rust
   ideal-order batch is fatal before reconciliation; prior ideals, actual orders, and partial output
   are not fallbacks. When submitted inputs deterministically define an output field, validate the
   exact result against the full input policy rather than checking only a coarse permission flag.

## Forbidden Patterns

In trading-critical paths, do not:

- swallow exceptions with `pass`, `continue`, or an ignored `return_exceptions=True` result
- catch and return neutral defaults such as `0.0`, `None`, `{}`, `[]`, or `False` for required data
- use `dict.get(required_key, default)` to hide a missing required configuration or input
- pass fabricated defaults to Rust, reconciliation, a risk gate, or the executor
- reinterpret malformed or partial Rust output as current strategy intent
- treat a producer-echoed diagnostic as independent proof of a field that is deterministic from the
  submitted input; validate the echo and the decision against that input

These shapes may be valid in optional or observability-only code. Classify the consumer before
changing a match.

## Live Action Readiness

Account-critical surfaces are required before any exchange action:

1. positions
2. balance
3. open orders

Market snapshots must be fresh for the symbols acted upon. Candles and EMAs are required only for
order classes whose strategy or risk decision consumes them. Stale flat-symbol candles must not
block protective management of held symbols.

A failed urgent candle refresh is therefore an availability observation, not an account-wide
planning barrier. Live Python may explicitly mark a symbol's known missing EMA inputs or one
position side's trailing extrema as unavailable. Rust remains strict by default, including
backtests, and may scope only that explicit live absence to forager selection, one-way arbitration,
strategy, or unstuck consumers that need it. A structurally required trailing bundle paired with
`trailing_available=false` is inert transport data and must be rejected before any consuming
strategy branch can read it.
Missing ordinary strategy input produces no ideal orders for the entry or close branch that
consumes it, so normal Rust-authoritative reconciliation removes any now-stale resting orders from
that branch. The other strategy branch, independent Rust risk reducers, and panic actions continue
when their own inputs are complete. Malformed or non-finite producer output is never covered by
this degradation and remains fatal. Here, producer output means the Rust ideal-order/result
envelope. A documented live input-reader availability sentinel must be classified before its
strategy consumer and retains only its explicitly bounded recovery and scoped-defer policies.

For candle-dependent actions, gate on canonical strategy-input readiness rather than raw REST
candle arrival. A proven no-trade gap may use explicitly synthesized zero-volume continuity when
the previous close is known, overlap repair is scheduled, and the gap is within policy. Unknown
stale tails must not be persisted or treated as verified zero-volume or zero-range history. Within
the configured active-tail bound, an explicitly supported provisional in-memory projection may use
flat zero-volume rows for active strategy inputs while authoritative overlap repair continues.
Trailing-extrema reconstruction may use the same projection only for a still-open tail after dense
post-fill coverage; it must not bridge a missing reset boundary or internal minute, and the
projected rows must be discarded after that read so delayed authoritative highs and lows replace
them immediately. Forager ranking quote-volume and log-range inputs for current, remote-enabled
candidates may bridge a later-bounded internal gap when the complete gap length is within
`live.max_active_candle_tail_gap_minutes`. This is an explicitly approximate ranking-continuity
policy, not proof that the missing rows were fetched. The rows remain unresolved and retryable,
are never persisted, and are replaced by delayed authoritative candles. Cache-only candidates
remain strict across unresolved internal gaps. Existing known-gap and refresh diagnostics expose
the underlying repair state. A compact per-symbol/metric transition diagnostic additionally marks
when ranking-input calculation consumes bounded continuity and later resumes from authoritative
candles; it does not retain per-span contexts or consecutive-use counters.

Protective panic and reduce-only actions may proceed when their own account-critical and
symbol-scoped requirements are fresh, even if unrelated strategy surfaces are unavailable.

## Forager And Eligibility Inputs

Flat-symbol forager candidates may remain rankable within
`live.max_forager_candle_staleness_minutes`. Close EMA readiness may use bounded flat-close
projection. Quote-volume and log-range ranking inputs for current, remote-enabled candidates may
use flat zero-volume continuity for later-bounded internal gaps within
`live.max_active_candle_tail_gap_minutes`.
Cache-only ranking inputs instead carry forward their latest known EMA with age/source metadata;
they do not receive invented zero tails.
When the forager setting is unset, its budget-derived acceptable age must not be shorter than
`live.max_active_candle_tail_gap_minutes`; the refresh budget must not silently reduce the active
tail grace period. An explicit positive forager cap is an operator override.

Candidates with no prior feature basis, non-finite carried values, or excessive feature age are
unavailable for new entries. Do not silently rank only the subset that happened to refresh first.
Ranking-feature absence must remain scoped to forager selection: when the exact remaining eligible
candidate count fits the remaining slots, Rust may select those candidates without ranking and
must not let Python turn that unused input absence into symbol-wide non-tradability or a current
ranking-unavailable alert. Diagnostics may retain the gap as conditional until Rust reports that
ranking was required for that side.

Approved and ignored coin state is an entry-eligibility input. Stale or unreadable eligibility
blocks affected initial entries but not protective management. With `auto_gs=true`, removal of a
held coin is handled as graceful stop.

## Fill And PnL Inputs

Structural fill history and realized-PnL quality are separate readiness facts. Fill-dependent
planning may use the configured lookback only when the cache proves `history_scope=all` or a
`covered_start_ms` at or before the configured lookback start. Unknown coverage triggers observable
refresh or deferral, never a neutral history.

Pending or degraded realized PnL blocks only enabled consumers that require authoritative PnL, such
as HSL, operational auto-unstuck with positive total exposure, or the realized-loss gate. When
every such consumer is disabled, proven fill history may remain ready for structural consumers
such as fill timestamps; PnL defects remain observable and repairable but do not globally defer
planning. Enabling a PnL consumer restores the strict requirement without a neutral PnL fallback.

Corrupt or unavailable fills use bounded repair/retry and explicit degraded decisions. Valid
manual or external exchange fills without Passivbot client IDs are exchange truth unless they
violate an explicit ownership or safety policy.

## Allowed Fallback Matrix

| Input/path | Default | Allowed fallback | Required evidence |
|---|---|---|---|
| Rust ideal-order batch | Fatal before reconciliation | None | Producer-boundary regression test |
| Exchange fetch methods | Propagate | None in fetch method | Caller policy tests |
| Required EMA | Unavailable/raise | Previous EMA for the same symbol/span only when explicitly implemented | `[ema]` warning, source/age/count, regression tests |
| Risk-gating input | Fail closed | None unless explicitly approved | `[risk]` visibility and regression tests |
| Live fill fee normalization | Best-effort contract | Quote fee → fresh conversion → exchange rate → `live.fee_pct_fallback` | Per-fill metadata, warning summary, policy tests |
| Legacy fill-event cache | Repair or rebuild | Safe metadata repair or known exchange repair; otherwise quarantine and exchange rebuild | Backup path, startup migration tests |
| Other required trading input | Unavailable/raise | Only a documented, approved fallback | Visibility, bounds, regression tests |

Fallback visibility includes the relevant symbol/input, reason, source, age when applicable, and
consecutive-use count.

## Tests For Any New Fallback

1. The fallback activates only when the primary source is unavailable.
2. Use is visible with bounded context.
3. The consumer fails closed or propagates when the fallback source is unavailable.
4. Unsafe substitutions are rejected.
5. Recovery to the primary source does not leave decision-changing hidden state.
