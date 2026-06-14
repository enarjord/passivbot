# Error Contract (Trading-Critical Paths)

This is the canonical policy for error handling and fallbacks in trading-critical code.

## Scope

Critical paths:

1. Exchange data fetches
2. EMA inputs used by orchestrator/backtester
3. Risk-gating inputs
4. Order-construction inputs

Default rule: hard-fail.

## Forbidden Patterns (Critical Paths)

1. `except Exception: pass` or `except Exception: continue`
2. Catch-and-return neutral defaults for required values (`0.0`, `None`, `{}`, `[]`, `False`)
3. `dict.get(required_key, safe_default)` for required trading inputs
4. `asyncio.gather(..., return_exceptions=True)` followed by dropped/ignored exceptions

## Required Behavior

1. Required inputs must be complete before the consumer that depends on them.
   Do not pass fabricated defaults to Rust or the live executor; if live omits
   a symbol/order class because inputs are unavailable, represent that
   unavailability explicitly.
2. If a required input fetch fails, raise immediately with actionable context.
3. If an explicitly allowed fallback exists, it must be bounded, observable, and test-covered.
4. If fallback source is unavailable, raise immediately.

## Live Order Freshness Contract

Live freshness requirements are scoped by order class and symbol. Do not treat every stale
surface as a global halt, and do not silently proceed with missing inputs required by the
specific exchange action being evaluated.

Account-critical surfaces are required before any exchange action, including panic or
reduce-only protection:

1. positions
2. balance
3. open orders

Market snapshot/ticker freshness is required for symbols being traded, especially symbols with
positions or initial-entry candidates. Candles and EMAs are required for order classes that
depend on strategy indicators, but stale candles for flat symbols must not block protective
management of held symbols.

Approved/ignored coin lists are live eligibility inputs. Stale or unreadable eligibility state
must block affected initial entries, but it must not block protective management of existing
positions. If a held coin is removed from approved coins or added to ignored coins, handle it as
graceful stop when `auto_gs=true`.

Fill/PnL history is required for order classes and risk gates that depend on it, including HSL,
max-loss, auto-unstuck, trailing, and related logic where applicable. Corrupted or unavailable
fills must use bounded repair/retry and observable degraded decisions. Do not replace missing
fill/PnL inputs with silent neutral defaults.

Manual or external fills/orders without a Passivbot client order id are not automatically
corruption. If the exchange row is otherwise valid and does not violate ownership or safety
policy, accept it as exchange truth and log it clearly as external/manual.

Hard-fail remains the default for trading-critical required inputs, but live order classes differ:
blocking a panic or reduce-only protective action can be worse than acting when that action's own
required account-critical and symbol-scoped surfaces are fresh.

## Fallback Matrix

| Path/Input | Default | Allowed Fallback | Visibility + Tests |
|------------|---------|------------------|--------------------|
| Exchange fetch methods (`fetch_balance`, `fetch_positions`, `fetch_open_orders`, etc.) | Raise | None | N/A |
| Required EMA inputs | Raise | Reuse previous EMA for same `symbol/span` only when explicitly implemented for that path | Log warning tag `[ema]` with reason/context + fallback tests |
| Risk-gating inputs | Raise | None unless explicitly approved in task | Log warning tag `[risk]` + regression tests |
| Live fill fee normalization (`fee_paid` used by accounting/risk gates) | Best-effort fee contract | Reported quote fee -> fresh ticker conversion for non-quote fee -> exchange fee rate -> `live.fee_pct_fallback` (which may be `0.0`); sanity outliers are replaced by the configured fallback | Per-fill fee metadata + `[fills]` warning summary + fee policy regression tests |
| Legacy live fill-event cache contract | Auto-migrate | Safe metadata repair or known exchange-specific repair; otherwise quarantine the old local cache and rebuild from exchange fills for the configured lookback | `[fills]`/`[fills-doctor]` warning with backup path + startup migration tests |
| Other required trading inputs | Raise | None unless documented and approved | Log warning + regression tests |

## Warning Log Fields For Fallback Use

Required fields:

1. `symbol`
2. `span` or relevant parameter name
3. `reason`
4. fallback value source
5. `age_ms` if applicable
6. consecutive fallback count

## Testing Requirements For Any New Fallback

1. Fallback is used when the primary source fails.
2. Visibility is present (warning log or explicit surfaced message).
3. Hard failure occurs when fallback source is unavailable.
4. Unsafe substitutions are guarded (for example, no silent last-price substitution for required EMA).
