# V8 Trading Logic Cleanup Checklist

Working target for `refactor/rust-strategy-runtime-plan`.

## Core behavior

- [x] Add a shared dynamic distance multiplier helper.
- [x] Use the helper for trailing grid entry distances.
- [x] Use the helper for trailing grid close distances.
- [x] Use the helper for ema anchor offset widening.
- [x] Rename runtime volatility state to `volatility_ema_1m` / `volatility_ema_1h`.
- [x] Change ema anchor inventory sensitivity to signed wallet exposure ratio.
- [x] Keep close-grid wallet-exposure behavior implicit through existing markup interpolation.
- [x] Extract common least-stuck candidate selection for TWEL enforcer and auto unstuck.

## Risk controls

- [x] Add explicit WEL enforcer enable/disable control.
- [x] Add explicit TWEL enforcer enable/disable control.
- [x] Add explicit auto-unstuck enable/disable control.
- [x] Keep WEL/TWEL enforcer threshold as a single ratio, not split trigger/reduce-to params.
- [x] Validate enabled enforcer thresholds as finite and greater than zero.
- [x] Preserve backtest `dynamic_wel_by_tradability` behavior.
- [x] Update TWEL enforcer to use a two-phase reduction contract:
  - first pass respects per-position floor as a fairness/preference rule
  - second pass continues reducing least-stuck bot-scope positions, even below the floor, until
    bot-scope TWE is at or below `TWEL * total_exposure_enforcer_threshold`
  - if min-qty/min-cost/order constraints prevent reaching target, surface a warning/diagnostic
- [x] Define `manual` mode as outside bot scope entirely:
  - no order creation
  - no order cancellation
  - excluded from bot-scope TWE/TWEL accounting
  - excluded from total exposure enforcer, position exposure enforcer, auto unstuck, and other
    bot-managed risk features
- [x] Keep WEL and TWEL enforcers as separate controls with clearer user-facing names:
  - WEL enforcer becomes `position_exposure_enforcer_*`
  - TWEL enforcer becomes `total_exposure_enforcer_*`
  - internal code may still use WEL/TWEL terminology where it is clearer
- [x] Document position exposure enforcer as both a per-position safety cap and an optional aggressive
  strategy mechanism:
  - users may deliberately set a threshold such as `0.95`
  - aggressive entries can refill position exposure back toward WEL
  - the position exposure enforcer can repeatedly trim back to `WEL * threshold`
  - unlike auto unstuck, this is not constrained by loss allowance or EMA gating
- [x] Document the risk-control stack as layered behavior:
  - close logic / negative markup: normal position management
  - auto unstuck: loss-budgeted and EMA-gated stuck-position reduction
  - position exposure enforcer: per-position WEL enforcement or aggressive trim mechanism
  - total exposure enforcer: bot-scope portfolio TWEL enforcement
  - HSL: equity-level circuit breaker

## Config/schema follow-up

- [x] Bump config schema to v8.
- [x] Decide final grouped config names for `risk`, `unstuck`, and `forager`.
- [x] Remove dev-branch-only compatibility shims from `format_config`.
- [x] Update example configs to v8 schema.
- [x] Update user docs and changelog.

## Deferred

- [ ] Forager anti-churn controls, if live/backtest behavior shows unwanted slot churn.
- [ ] Target/range volatility for forager, if users ask for excluding very high volatility coins.
