# EMA Entry Gate Mode Plan

Branch: `v8`

## Goal

Make EMA entry gating configurable for `trailing_martingale` entries instead of hard-wiring it to
initial entry orders only.

Current behavior:

- Long initial entries are priced at or below the lower EMA band:
  `min(best_bid, ema_lower * (1 - entry.initial_ema_dist))`.
- Short initial entries are priced at or above the upper EMA band:
  `max(best_ask, ema_upper * (1 + entry.initial_ema_dist))`.
- Long and short reentries are based on position price, threshold, retracement, wallet exposure,
  volatility, and best bid/ask, without an EMA-band cap/floor.

Target behavior: add one explicit config mode that decides which entry order classes receive that
same EMA-band cap/floor.

## Proposed Config

Add this field under each side's active `trailing_martingale` entry config:

```text
bot.<side>.strategy.trailing_martingale.entry.ema_gate_mode
```

Canonical values:

- `disabled`: no entry order is EMA gated.
- `initial`: only initial entry order types are EMA gated. This is the current behavior and should
  be the default.
- `reentry`: initial entry order types are not EMA gated; all non-initial entry order types are EMA
  gated.
- `all`: every entry order type is EMA gated.

Use `reentry`, not `re-entry` or `re_entry`, as the canonical value. Do not add aliases unless the
branch explicitly decides to support them.

`ema_gate_mode` is fixed config, not an optimizer parameter. This follows the v8 optimizer rule for
categorical and boolean policy fields: enums and bools stay fixed from the base/anchor config, while
only numeric strategy/risk fields are optimizer-bound candidates. Examples of fixed fields include
`hsl.enabled`, `risk.position_exposure_enforcer_enabled`,
`risk.total_exposure_enforcer_enabled`, `risk.total_exposure_enforcer_policy`,
`risk.total_exposure_entry_gate_enabled`, `risk.we_excess_allowance_mode`, and
`unstuck.enabled`.

Default:

```json
{
  "entry": {
    "ema_gate_mode": "initial"
  }
}
```

`entry.initial_ema_dist` remains the offset used by the EMA gate. When a mode does not gate an
order class, `initial_ema_dist` is ignored for that order class.

## Order Classification

Treat these as initial entry order types:

- `EntryInitialNormalLong`
- `EntryInitialPartialLong`
- `EntryInitialNormalShort`
- `EntryInitialPartialShort`

Treat these as reentry order types:

- `EntryGridNormalLong`
- `EntryGridCroppedLong`
- `EntryTrailingNormalLong`
- `EntryTrailingCroppedLong`
- `EntryGridNormalShort`
- `EntryGridCroppedShort`
- `EntryTrailingNormalShort`
- `EntryTrailingCroppedShort`

`initial` and `reentry` should follow order type semantics, not just `position.size == 0.0`. The
current code can emit an `EntryInitialPartial*` order while a small position already exists; that
order should still count as `initial`.

## Pricing Rules

Define side-specific EMA gate prices:

```text
long_ema_gate_price  = min(best_bid, ema_lower * (1 - entry.initial_ema_dist))
short_ema_gate_price = max(best_ask, ema_upper * (1 + entry.initial_ema_dist))
```

After normal entry price calculation, apply the gate only if the selected mode gates that order
class:

```text
long_final_price  = min(normal_entry_price, long_ema_gate_price)
short_final_price = max(normal_entry_price, short_ema_gate_price)
```

This means:

- `disabled`: initial longs are placed at `best_bid`; initial shorts are placed at `best_ask`.
  Reentries keep their current non-EMA threshold/retracement prices.
- `initial`: preserves current initial and initial-partial behavior. Reentries keep their current
  non-EMA threshold/retracement prices.
- `reentry`: initial and initial-partial orders are placed at touch, while grid/trailing reentries
  are additionally capped/floored by the EMA gate.
- `all`: initial, initial-partial, grid, and trailing entries are capped/floored by the EMA gate.

Do not make the EMA gate part of the trailing trigger condition itself. Threshold and retracement
state should still decide whether a trailing reentry is eligible; the EMA gate only constrains the
price of the emitted entry order after eligibility is known.

Keep current price-step rounding direction:

- Long gate target rounds down.
- Short gate target rounds up.
- Final recursive entry expansion still runs through the existing quantization step.

## Entry Expansion Effects

Recursive expansion in `calc_entries_long()` and `calc_entries_short()` should keep its existing
stop rules:

- stop on zero qty;
- stop if a trailing order was emitted after at least one previous entry;
- stop if two adjacent generated entries have the same price.

With `all` or `reentry`, EMA gating may collapse multiple reentry candidates to the same gated
price. That should be accepted and handled by the existing duplicate-price break rather than
worked around with new state.

## EMA Readiness Contract

The mode must affect live/backtest input requirements. Do not claim EMA gating is disabled while
still requiring EMA bands for that order class.

EMA bands are required for entry generation only when the requested entry order class may be gated:

| Mode | Flat initial entry | Initial partial | Reentry |
|------|--------------------|-----------------|---------|
| `disabled` | no | no | no |
| `initial` | yes | yes | no |
| `reentry` | no | no | yes |
| `all` | yes | yes | yes |

Other EMA consumers remain independent:

- `unstuck` still requires EMA bands when enabled because unstuck uses `ema_band_upper/lower`.
- One-way flat-side selection still requires candle/EMA readiness because it keeps using EMA-band
  distance as the long-vs-short tie-breaker, even when `ema_gate_mode="disabled"` means the emitted
  initial entry price itself is not EMA gated.
- Any forager or other side-selection logic that ranks candidates by EMA distance must be treated
  separately from entry-price gating and must declare its own EMA readiness requirement.
- Do not pass fabricated EMA defaults into Rust for an order path that no longer requires EMA.
  Either avoid deriving EMA bands for that path or make absence explicit in the Rust request shape.

## One-Way Flat-Side Tie-Breaking

Selected policy: keep EMA-band distance as a separate tie-break signal.

In one-way mode, when both long and short are flat and both sides are otherwise eligible, Passivbot
cannot place both initial entries at once. Today the tie-breaker uses EMA-band distance. If flat
initial entry EMA gating is disabled, the bot should still use EMA-band distance to decide which
side may place an initial entry.

This is intentionally separate from entry-price gating:

- `ema_gate_mode="disabled"` disables the EMA cap/floor on emitted initial entry prices.
- It does not disable EMA use for one-way flat long-vs-short selection.
- Therefore candlesticks and EMAs remain required in one-way mode when both sides are flat and both
  sides need a tie-breaker.

This must be documented in user-facing config docs to avoid the surprising interpretation that
disabled entry EMA gating removes every candle/EMA dependency from flat initial entry selection.

Alternatives considered:

### 1. Explicit Fixed Side Priority

Add a fixed non-optimizable policy such as `long`, `short`, or `auto_default_long` for the one-way
flat-side tie-breaker. The default can preserve today's tie-break fallback preference for long when
all else is equal.

Pros:

- Stateless and reproducible after restart.
- Does not require EMA readiness when entry EMA gating is disabled.
- Honest: it exposes that direction choice is policy, not a hidden signal.
- Easy to test and reason about.

Cons:

- Arbitrary from a market-prediction perspective.
- Can bias one-way mode toward one side for long periods unless the user configures it.

Decision: rejected because it is convenient for implementation but creates trader-visible long/short
bias.

### 2. Keep EMA Distance As A Separate Tie-Break Signal

Preserve the current EMA-distance side selection even when `ema_gate_mode` does not gate flat
initial entries.

Pros:

- Keeps current directional behavior.
- Uses an existing signal and avoids adding a new side-priority knob.
- May choose the side closer to the strategy's existing EMA reference.

Cons:

- Still requires EMA readiness for flat one-way initial entries, even though entry prices are not
  EMA gated.
- Blurs the meaning of "EMA gating disabled" unless documented as "pricing disabled, tie-breaker
  still EMA-based."
- Stale/missing EMA can still block or alter one-way flat-side selection.

Decision: selected. The separate candle/EMA readiness requirement is part of the contract.

### 3. Risk-Capacity Tie-Break

Choose the side with more available effective wallet exposure or fewer active positions under the
current side-specific risk limits.

Pros:

- Uses existing risk state rather than adding a direction signal.
- Helpful when long/short side limits differ or one side is already more constrained elsewhere.
- Does not require EMA bands.

Cons:

- Often ties when both sides are flat and symmetric.
- Not a directional signal; it chooses capacity, not expected opportunity.
- Needs a deterministic fallback, likely option 1.

### 4. Deterministic Rotation Or Hash

Alternate side by candle timestamp, symbol index, or another deterministic stateless input.

Pros:

- Spreads entries across sides over time without random state.
- Can be restart-reproducible if based only on exchange/config/time inputs.
- Does not require EMA bands.

Cons:

- Still arbitrary and can flip desired side as time advances.
- Timestamp-based rotation can cause confusing order churn near cycle boundaries.
- Symbol-hash rotation is stable but not market-aware.

### 5. Order-Book Microstructure Signal

Choose using available market snapshot features such as spread, touch movement, or order-book
imbalance if supplied by the exchange path.

Pros:

- Does not require candles or EMA bands.
- At least attempts to use current market information.

Cons:

- Many existing order-book inputs are only best bid/ask, not full depth.
- Very noisy and exchange-dependent.
- Adds a new strategy signal with weak evidence and more live/backtest parity burden.

### 6. Random Choice

Randomly choose long or short when both sides tie.

Pros:

- Simple and avoids permanent directional bias.

Cons:

- Violates the stateless reproducibility requirement unless the randomness is derived from stable
  inputs, in which case it becomes option 4.
- Harder to debug and backtest/live-match.

Recommendation: do not use random choice.

## Auto-Unstuck EMA Gating

Selected policy: add `bot.<side>.unstuck.ema_gating_enabled`.

Current auto-unstuck behavior is independent from entry pricing but still EMA-dependent. In Rust,
long unstuck requires current price to be at or above:

```text
ema_upper * (1 + unstuck.ema_dist)
```

Short unstuck requires current price to be at or below:

```text
ema_lower * (1 - unstuck.ema_dist)
```

The EMA bands are currently derived from the active strategy EMA spans, so changing entry
`ema_gate_mode` must not accidentally change auto-unstuck readiness or trigger behavior.

Auto-unstuck should get its own fixed boolean:

```text
bot.<side>.unstuck.ema_gating_enabled
```

Default: `true`.

When `true`, preserve current behavior: auto-unstuck uses the strategy EMA bands plus
`unstuck.ema_dist`.

When `false`, auto-unstuck still requires its normal loss allowance, WEL threshold, close
percentage, position, market price, and min-size inputs, but it does not require EMA bands and does
not apply the EMA trigger. `unstuck.ema_dist` remains present but is moot while
`unstuck.ema_gating_enabled=false`.

This matches the broader v8 risk-config pattern: boolean fields enable or disable a policy, while
numeric fields tune the policy when enabled.

Alternatives considered:

### A. Keep Current `unstuck.ema_dist` Only

Users can approximate "always pass" EMA gating with a very permissive finite offset.

Pros:

- No new config.
- Preserves current behavior exactly.
- Keeps optimizer/config surface small.

Cons:

- Sentinel-style values are unclear and side-sensitive.
- A value like `-1.0` is not a clean disable: for long it can make the target non-positive and skip
  unstuck entirely, while for short it tends to make the condition permissive.
- Operators cannot distinguish "wide EMA gate" from "EMA gate intentionally disabled."

Decision: rejected as the documented disable mechanism.

### B. Add `bot.<side>.unstuck.ema_gating_enabled`

Add a fixed boolean, default `true`, that controls only the auto-unstuck EMA trigger. When `false`,
auto-unstuck still requires its normal loss allowance, WEL threshold, close percentage, position,
market price, and min-size inputs, but it does not require EMA bands for the unstuck trigger.

Pros:

- Clear operator intent.
- Avoids sentinel offsets.
- Lets live readiness skip EMA bands for unstuck when the gate is disabled.
- Keeps entry EMA behavior and auto-unstuck behavior separate.

Cons:

- Adds one more fixed policy flag.
- Needs explicit tests to prove unstuck still respects loss/risk constraints when EMA gating is off.

Decision: selected.

### C. Add Independent Unstuck EMA Spans

Add separate unstuck EMA spans, for example under `bot.<side>.unstuck`, while keeping the existing
strategy EMA spans for entry gating and strategy behavior.

Pros:

- Lets auto-unstuck use a risk-management horizon independent of entry timing.
- Avoids forcing entry EMA tuning to also tune unstuck trigger sensitivity.
- Conceptually clean if users want unstuck to react to slower or faster bands than entries.

Cons:

- Adds more numeric fields and therefore potentially more optimizer surface if allowed.
- Requires additional EMA warmup/readiness plumbing in live and backtest.
- Increases candle/indicator memory and parity test surface.
- More scope than entry EMA gate mode and not needed to make `ema_gate_mode` coherent.

Decision: not now. Do not include independent unstuck EMA spans in the first entry EMA gate patch.
Consider them later only if backtests or live operations show that shared strategy EMA spans are a
real limitation. If added later, decide explicitly whether they are optimizable numeric fields or
fixed operational parameters.

## Implementation Surfaces

Rust should remain the source of truth for this behavior.

Expected Rust changes:

- Add an `EmaGateMode` enum, serialized as snake-case strings, near strategy parameter types in
  `passivbot-rust/src/strategies/mod.rs`.
- Add `ema_gate_mode: EmaGateMode` to `TrailingMartingaleEntryParams`.
- Default missing `ema_gate_mode` to `Initial` for current-behavior preservation within v8.
- Parse the new field in `passivbot-rust/src/python.rs::trailing_martingale_strategy_params_from_dict`.
- Update entry price helpers in `passivbot-rust/src/entries.rs` so initial and reentry order paths
  compute normal prices first, then apply the gate according to mode and order class.
- Update orchestrator readiness in `passivbot-rust/src/orchestrator.rs` so EMA bands are required
  only for modes/order classes that actually need them, plus existing non-entry EMA consumers.
- Keep one-way flat-side EMA-distance tie-breaking as its own EMA consumer. It must still derive EMA
  bands even when flat initial entry price gating is disabled.
- Add `unstuck_ema_gating_enabled` to shared bot params, defaulting to true, and gate the
  auto-unstuck EMA trigger/readiness on it.
- Extend Rust strategy metadata in `passivbot-rust/src/strategies/spec.rs` so the config default is
  Rust-owned. The current numeric `StrategyParameterSpec` is not enough for a string enum; add
  non-optimizable enum/default metadata rather than reintroducing Python-owned strategy defaults.

Expected Python/config changes:

- Ensure formatted configs include
  `bot.<side>.strategy.trailing_martingale.entry.ema_gate_mode`.
- Ensure optimizer bound generation does not treat `ema_gate_mode` as a numeric optimizable
  parameter. Keep all bool and enum policy fields fixed from the selected config/anchor.
- Ensure optimizer bound generation does not treat `unstuck.ema_gating_enabled` as optimizable.
- Ensure config normalization preserves the enum and rejects invalid values loudly.
- Update example configs through the normal formatting/template path after Rust metadata exposes the
  default.

Expected docs changes:

- Update `docs/ai/features/strategy_runtime.md` with the durable contract after implementation.
- Update user-facing config docs or generated field references if this repo has a generated
  strategy-field table for the current branch.
- Update `CHANGELOG.md` under Unreleased because this is user-facing strategy behavior.

## Testing Plan

Rust unit tests:

- `initial` preserves current long and short initial entry prices.
- `disabled` places long initial at bid and short initial at ask.
- `all` gates long grid reentry by `min(reentry_price, ema_gate_price)`.
- `all` gates short grid reentry by `max(reentry_price, ema_gate_price)`.
- `reentry` leaves initial entries at touch but gates grid/trailing reentries.
- Initial partial orders follow initial-mode semantics.
- Trailing reentry trigger eligibility is unchanged; only emitted price changes after eligibility.
- Duplicate-price recursive expansion still terminates when EMA gating collapses adjacent reentries.

Python/Rust boundary tests:

- Config with `ema_gate_mode` reaches Rust as the expected enum.
- Missing `ema_gate_mode` formats to `initial`.
- Invalid mode raises with the full config path.
- Strategy metadata exposes the enum default without adding a numeric optimizer bound.

Integration/readiness tests:

- With `disabled`, flat initial entries can be generated without EMA bands when no other requested
  order class needs EMA.
- With `reentry`, flat initial entries do not require EMA bands, but position reentries do.
- With `initial` or `all`, flat initial entries still require EMA bands.
- In one-way mode with both sides flat and otherwise eligible, `disabled` initial entry EMA gating
  still requires EMA bands for the EMA-distance side tie-breaker.
- The same one-way test proves EMA readiness is required for side selection but not for initial entry
  price calculation.
- Unstuck requires EMA bands when `unstuck.ema_gating_enabled=true`.
- Unstuck does not require EMA bands for its own trigger when `unstuck.ema_gating_enabled=false`,
  while still enforcing loss allowance, WEL threshold, close percentage, position, market price, and
  min-size constraints.

## Acceptance Criteria

- Default formatted configs keep current behavior through `ema_gate_mode="initial"`.
- Live and backtest order generation agree for all four modes.
- No Python-side patch reimplements strategy pricing outside Rust.
- No compatibility aliases are added for enum values unless explicitly approved.
- Missing/invalid EMA inputs fail loudly only when the active order class or another active feature
  requires EMA.
- In one-way flat-side selection, EMA inputs are required for the side tie-breaker even when initial
  entry price EMA gating is disabled.
- Existing trailing_martingale configs either format with the new default or fail with an actionable
  validation error; they do not silently change to `disabled`.

## Resolved Decisions

- `reentry` is the canonical spelling.
- `ema_gate_mode` is fixed config, not optimizable.
- All enum and boolean policy fields are fixed, including HSL, risk enforcer, risk policy, WEL
  allowance mode, TWEL entry gate, and unstuck enabled flags.
- `EntryInitialPartial*` counts as initial for EMA gate mode.
- `disabled` means flat initial entries are emitted at current best bid/ask from the exchange
  orderbook. Reentries continue to follow normal threshold/retracement trading logic.
- One-way flat-side tie-breaking keeps using EMA-band distance, so candles/EMAs are still required
  for that selection even when entry price EMA gating is disabled.
- Auto-unstuck gets `bot.<side>.unstuck.ema_gating_enabled`, a fixed boolean defaulting to `true`.
- Independent auto-unstuck EMA spans are out of scope for this patch.

## Implementation Notes

- Rust strategy metadata exposes `ema_gate_mode` through `fixed_parameters`, separate from numeric
  optimizer `parameters`.
- The one-way caveat should be worded explicitly: "entry EMA gate disabled" means emitted entry
  pricing is not EMA capped/floored; one-way side selection may still require candles/EMAs.
