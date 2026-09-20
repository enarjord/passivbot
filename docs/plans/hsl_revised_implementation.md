# Revised HSL implementation checkpoints

This temporary implementation record follows the
[reviewed design](hsl_best_effort_redesign.md). Legacy HSL remains the trading default;
the revised numerical kernel below has no live, backtest or optimizer controller caller.

## Shared numerical kernel

`passivbot-rust/src/hsl_revised.rs` owns the pure batch drawdown calculation. Its
`hsl_revised_signal` Python binding accepts ordered `(timestamp_ms, realized, unrealized)`
currency observations, a positive budget, fractional EMA span, RED threshold and optional
entry reference. The JSON result includes equity, running peaks, raw drawdown, EMA and
strict threshold decisions for every observation. This is an experimental comparison
surface, not a complete HSL engine or an alternative execution authority.

The kernel implements current-budget rebasing, raw-drawdown EMA with its first value as
seed, same-minute replacement from the prior-minute baseline, and known boundary peaks.
A supplied minimal-history reference seeds the peak without creating a zero EMA sample.
It has no journal, cache, persistent EMA or prior-decision input. Callers remain responsible
for scope aggregation, valid current observations, clipping, reconstructing history and
controller lifecycle. A singleton without an entry reference is just a mathematical
one-point series, not the complete no-history estimator.

Subtract common realized/unrealized offsets before combining currency deltas to retain
small representable changes. Nonpositive historical peaks use the reference impairment
value of one. Arithmetic outside finite float range saturates with
`numeric_range_approximation=true`; the current endpoint is always restored to the budget.
This is explicit approximation, not a finite-precision claim of exactness for extreme
history. Invalid minimum inputs, nonfinite observations and reversed timestamps raise
rather than returning a healthy decision. The later best-effort reconstructor must turn
damaged historical fields into documented estimates before calling this numeric kernel.

## Validation and remaining work

The real-extension parity suite compares the kernel with the independent Decimal oracle:
hand-computed losses/recoveries, strict threshold equality, fractional spans, singleton
references, nonpositive peaks, repeated within-minute samples, common currency offsets,
40 deterministic 100-observation paths, finite-range behavior and input rejection.
Existing legacy Rust tests run alongside these cases.

Still required before offline completion:

- Full Rust historical reconstruction, candle-free/mixed-price composition and shared
  coin/pside/unified adapters, checked against the independent fixtures.
- Complete exchange-reconstructible controller replay, interventions, cooldown/never
  expiry, known flatten samples and execution permissions.
- Startup-only engine selection, config migration and optimizer/backtest integration.
- Source-matched extension parity through full offline fake-live scenarios, partial fills,
  current-state changes, replay recovery and cache-free restart.
- Canonical contracts, migration documentation and the live-validation/rollback checklist.

No actual live testing or legacy replacement is part of offline completion. Those are
subsequent stages with separate authorization.
