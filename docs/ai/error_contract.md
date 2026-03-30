# Error Contract (Trading-Critical Paths)

This is the canonical policy for error handling and fallbacks in trading-critical code.

## Scope

Critical paths:

1. Exchange data fetches
2. EMA inputs used by orchestrator/backtester
3. Risk-gating inputs
4. Order-construction inputs
5. Coin-selection / shortlist inputs used to decide which symbols may open positions

Default rule: hard-fail.

## Forbidden Patterns (Critical Paths)

1. `except Exception: pass` or `except Exception: continue`
2. Catch-and-return neutral defaults for required values (`0.0`, `None`, `{}`, `[]`, `False`)
3. `dict.get(required_key, safe_default)` for required trading inputs
4. `asyncio.gather(..., return_exceptions=True)` followed by dropped/ignored exceptions
5. Downstream config consumers compensating for missing required params with local defaults

## Required Behavior

1. Required inputs must be complete before calling Rust orchestrator/backtester.
2. If a required input fetch fails, raise immediately with actionable context.
3. If an explicitly allowed fallback exists, it must be bounded, observable, and test-covered.
4. If fallback source is unavailable, raise immediately.
5. Config hydration, migration, renaming, and default insertion must happen in centralized config formatting/parsing code. Downstream consumers must assume correctness and must not repair missing required params locally.

## Fallback Matrix

| Path/Input | Default | Allowed Fallback | Visibility + Tests |
|------------|---------|------------------|--------------------|
| Exchange fetch methods (`fetch_balance`, `fetch_positions`, `fetch_open_orders`, etc.) | Raise | None | N/A |
| Required EMA inputs | Raise | Reuse previous EMA for same `symbol/span` only when explicitly implemented for that path and bounded by age/consecutive-fallback safeguards | Log warning tag `[ema]` with reason/context + fallback tests |
| Risk-gating inputs | Raise | None unless explicitly approved in task | Log warning tag `[risk]` + regression tests |
| Coin-selection / shortlist inputs | Raise | None unless explicitly documented and approved for that exact path | Log warning + regression tests |
| Other required trading inputs | Raise | None unless documented and approved | Log warning + regression tests |

## Warning Log Fields For Fallback Use

Required fields:

1. `symbol`
2. `span` or relevant parameter name
3. `reason`
4. fallback value source
5. `age_ms` if applicable
6. consecutive fallback count

Additional boundedness requirements for approved EMA fallback use:

1. The fallback must be stale-bounded by age.
2. The fallback must be bounded by consecutive fallback count.
3. Repeated missing EMA reuse must hard-fail once the approved bound is exceeded.

## Testing Requirements For Any New Fallback

1. Fallback is used when the primary source fails.
2. Visibility is present (warning log or explicit surfaced message).
3. Hard failure occurs when fallback source is unavailable.
4. Unsafe substitutions are guarded (for example, no silent last-price substitution for required EMA).
