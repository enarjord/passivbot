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

## Forbidden Patterns

In trading-critical paths, do not:

- swallow exceptions with `pass`, `continue`, or an ignored `return_exceptions=True` result
- catch and return neutral defaults such as `0.0`, `None`, `{}`, `[]`, or `False` for required data
- use `dict.get(required_key, default)` to hide a missing required configuration or input
- pass fabricated defaults to Rust, reconciliation, a risk gate, or the executor

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

For candle-dependent actions, gate on canonical strategy-input readiness rather than raw REST
candle arrival. A proven no-trade gap may use explicitly synthesized zero-volume continuity when
the previous close is known, overlap repair is scheduled, and the gap is within policy. Unknown
stale tails must not be converted to zero volume or zero range.

Protective panic and reduce-only actions may proceed when their own account-critical and
symbol-scoped requirements are fresh, even if unrelated strategy surfaces are unavailable.

## Forager And Eligibility Inputs

Flat-symbol forager candidates may remain rankable within
`live.max_forager_candle_staleness_minutes`. Close EMA readiness may use bounded flat-close
projection. Quote-volume and log-range ranking inputs carry forward their latest known EMA with
age/source metadata; they do not receive invented zero tails.

Candidates with no prior feature basis, non-finite carried values, or excessive feature age are
unavailable for new entries. Do not silently rank only the subset that happened to refresh first.

Approved and ignored coin state is an entry-eligibility input. Stale or unreadable eligibility
blocks affected initial entries but not protective management. With `auto_gs=true`, removal of a
held coin is handled as graceful stop.

## Fill And PnL Inputs

Risk consumers may use a fill/PnL lookback only when the cache proves `history_scope=all` or a
`covered_start_ms` at or before the configured lookback start. Unknown coverage triggers observable
refresh or deferral, never neutral PnL.

Corrupt or unavailable fills use bounded repair/retry and explicit degraded decisions. Valid
manual or external exchange fills without Passivbot client IDs are exchange truth unless they
violate an explicit ownership or safety policy.

## Allowed Fallback Matrix

| Input/path | Default | Allowed fallback | Required evidence |
|---|---|---|---|
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
