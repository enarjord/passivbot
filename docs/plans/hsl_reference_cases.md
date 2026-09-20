# Executable HSL reference cases

This is the first executable specification for the
[best-effort redesign](hsl_best_effort_redesign.md). It changes no production HSL,
config loader, optimizer, or order execution. The standard-library Python oracle
in `tests/hsl_reference.py` imports no production risk code. It is a test instrument,
not a Python trading controller or live fallback. Decimal amounts make it independent
of the future Rust floating-point implementation; integration must compare with
explicit tolerances and exact decision expectations near thresholds.

Run the focused suite with:

```sh
pytest tests/test_hsl_best_effort_reference.py tests/test_hsl_reference_fake_exchange.py
```

The fake exchange tests use the deterministic, offline `FakeCCXTClient` from the
fake-live harness. They generate real fixture-ledger fills and compare reconstructed
positions, realized PnL, fees, and current UPNL against that client's independently
maintained account state. They do not start a bot or contact an exchange. No revised
HSL has yet been wired into the full live-loop harness.

## Reference choices made concrete

- Rebase currency PnL+UPNL to the current scope budget, then compute equity peaks,
  raw drawdown, and EMA of drawdown. Trigger on strict `min(raw, EMA) > threshold`.
  Seed the first EMA from its first drawdown. Within-minute observations replace
  the EMA contribution from the prior-minute baseline; known flatten boundaries
  must be checked before episode reset/reopening.
- Preserve all supplied boundary rows in the signal output. Episode clipping is
  explicit in the test caller; an inferred flat from ambiguous/clamped quantities
  is not promoted into a proven lifecycle boundary. The old-incomplete/new-clean
  example checks that a supplied independent episode boundary isolates old damage.
- Missing-opening quantity is inferred backward from current signed exposure.
  Basis starts at the first usable retained fill price and walks forward; current
  size/basis always anchor the endpoint. Inverse contracts use harmonic entry basis
  and inverse PnL units; scope budgets must use matching settlement currency.
- Missing gross realized PnL is estimated from the reconstructed reduction and basis.
  Fees are signed balance impacts, added exactly once. Unknown fee impact is omitted
  with a reason; this is uncertainty, not proof that no fee was paid.
- Execution identity is deduplicated before reconstruction. Highest supplied revision
  wins; contradictory rows at the same revision are excluded with a reason. Known
  sequence wins within a timestamp; otherwise per-pair increases precede reductions
  with identity as tie-breaker. Ambiguity is disclosed, not a flat certificate.
- A nonpositive historical running peak has no meaningful ratio. The reference
  assigns raw drawdown 1 to that segment until a positive peak exists. Later normal
  positive-peak arithmetic resumes, including values above 1 if equity is negative.
  This is an explicit numeric reference choice for review, not existing runtime behavior.
- Minute-close sources must be available at evaluation time. Coarse source candles
  must be fully inside the window; real minutes override coarse estimates. Zigzag
  close pivots match the documented open/extrema/close path. Gaps forward-fill and
  the missing prefix backfills. The current endpoint is replaced by the fresh mark.
- In the explicit minimal-history case, seed an entry-value peak reference, not an
  extra EMA observation. With loss L and budget B, the singleton signal is L/(B+L).
  Adding actual history can legitimately change the decision by restoring smoothing.
- Cooldown expiry uses the actual flatten timestamp. `never` expiry uses the imposing
  stop timestamp. These need not be identical; tests cover exact inclusive-window
  endpoints. Zero cooldown affects waiting only. All lifecycle calls recompute from
  supplied exchange-reconstructible evidence with no prior-decision argument.

## What these tests establish

Hand-calculated examples assert complete intermediate sequences, not only booleans.
Fault cases remove old, middle, and latest fills; repeat/correct/conflict identities;
damage price, fee, and realized fields; reorder ties; lose entire round trips; and
replace candle coverage with coarse or carried prices. Restoring evidence must restore
the clean series. Missing the latest reduction has an explicit example of a missed
clean-data stop: unknown realized loss and adjusted past sizes lower the estimate.
The suite records that difference instead of equating a finite result with adequate
protection. Opposite pair paths demonstrate why aggregation must precede peaks.
Shorts, flat short endpoints, multipliers, inverse PnL, inactive coin divisors, budget
changes, and extreme finite amounts are exercised separately.

Missing round trips cannot be inferred from an unchanged current size. The fixture
asserts that lost information, rather than manufacturing a loss or claiming full
reconstruction. A finite estimate or an empty reason set is not a completeness proof.

Lifecycle examples cover both restart policies and intervention choices, no-execution
RED followed by recovery, partial panic exposure, re-panic flatten anchors, and window
expiry. `LifecycleEvidence` is an input fixture of exchange-reconstructible event times,
not a persisted latch. It does not solve uncertain event classification by itself.

## Remaining coverage before production integration

This PR supplies reference primitives and a first fault corpus. It does not claim
the comprehensive integration gates in the design are already satisfied. In particular:

- Resolve ambiguous lifecycle-anchor classification with full exchange traces, then
  compare Rust and cache-free restart against those traces. The reference permission
  algebra alone cannot certify a halt/reopen event.
- Compose multi-pair episode resets and exact flatten samples with the shared timeline;
  preserve the stated per-scope semantics and same-minute ordering.
- Specify how usable realized-loss evidence is retained when all candles are absent;
  the minimal-history and reconstructed-history examples are separate here. Do not
  treat missing candles as permission to discard known losses during integration.
- Add snapshot-skew/revision interleavings, config migration and optimizer surfaces,
  and source-correction/candle provenance diagnostics at their real integration boundaries.
- Feed these fixtures through the rebuilt Rust extension and full fake-live loop,
  including protection scheduling, outstanding-order reconciliation, and partial fills.

Small artificial intervals in the numerical/lifecycle unit cases keep expected values
readable. They are not valid production lookback configurations: enabled revised HSL
still requires the separately specified 1-90 day config validation.
