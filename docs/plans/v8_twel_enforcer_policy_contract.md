# v8 Total Exposure Enforcer Policy Contract

## Status

Draft handoff for a future v8 implementation. This is not a v7 bug-fix spec.

The v7 behavior discussed in GitHub issue #600 can remain unchanged short term unless a separate
v7 mitigation is explicitly chosen. v8 is already a breaking upgrade, so the total exposure
enforcer contract can be redesigned there without preserving v7 semantics.

In this document, TWEL means total wallet exposure limit. The v8 user-facing config names are
`bot.<pside>.risk.total_wallet_exposure_limit`,
`bot.<pside>.risk.total_exposure_enforcer_threshold`, and the future
`bot.<pside>.risk.total_exposure_enforcer_policy`. Some Rust/PyO3 internals still use historical
`risk_twel_*` field names at the compiled boundary.

## Background

Current v7 behavior allows a live loop where same-side entries refill portfolio exposure while the
TWEL enforcer emits `close_auto_reduce_twel_*` orders. The root contract gap is that entry gating
uses raw `total_wallet_exposure_limit`, while the TWEL enforcer reduces at
`total_wallet_exposure_limit * risk_twel_enforcer_threshold`.

That produces a band between thresholded TWEL and raw TWEL where entries may continue even though
the TWEL enforcer is already trying to reduce the portfolio.

WEL should not be treated the same way. A WEL threshold below `1.0` may intentionally create a
per-position trim/refill loop:

```text
effective_wel = raw_twel / n_positions * (1 + risk_we_excess_allowance_pct)
wel_reduce_threshold = effective_wel * risk_wel_enforcer_threshold
```

In that contract, entries may still refill a position toward `effective_wel`, then the WEL enforcer
may trim it back toward `wel_reduce_threshold`. This can behave like an unbudgeted, aggressive
auto-unstuck mechanism. It is dangerous, but intentionally useful for some aggressive configs.

TWEL should have a different contract. Auto-unstuck is loss-budgeted and EMA-gated. WEL is a
per-position emergency trimmer. TWEL should be the portfolio exposure governor that manages borrowed
capacity created by `risk_we_excess_allowance_pct`.

## Design Goals

1. Preserve the useful excess-allowance behavior:
   - A single position may temporarily exceed `raw_twel / n_positions`.
   - Positions can borrow unused capacity from slots that have not filled yet.
   - This remains useful when excess allowance is modest and `n_positions` is high.

2. Prevent TWEL-specific entry/reduce oscillation:
   - TWEL reductions should not be immediately undone by TWEL-blind entries.
   - TWEL should not become an unbounded loss-harvesting loop.

3. Keep WEL and TWEL contracts separate:
   - WEL may deliberately oscillate around effective WEL.
   - TWEL should gate portfolio exposure and repair over-target states.

4. Keep the v8 policy surface small:
   - Avoid five or more policies unless real usage proves they are needed.
   - Prefer a small set of clear policy names.

5. Keep behavior stateless:
   - Decisions must be derived from exchange state, config, and current market inputs.
   - Do not add cooldown state or local memory to suppress entries.

## Core Definitions

Use raw TWEL for per-position sizing and thresholded TWEL for portfolio gating:

```text
raw_twel = bot.{pside}.risk.total_wallet_exposure_limit
twel_threshold = bot.{pside}.risk.total_exposure_enforcer_threshold
effective_twel = raw_twel * twel_threshold
effective_wel = raw_twel / effective_n_positions * (1 + risk_we_excess_allowance_pct)
```

Important distinction:

- `effective_wel` remains derived from raw TWEL.
- `effective_twel` governs portfolio-level entry blocking and TWEL auto-reduce target.

Example:

```text
raw_twel = 1.0
n_positions = 10
risk_we_excess_allowance_pct = 0.25
total_exposure_enforcer_threshold = 0.9

effective_wel = 1.0 / 10 * 1.25 = 0.125
effective_twel = 1.0 * 0.9 = 0.9
```

With `total_exposure_enforcer_threshold = 0.9`, the bot may still size individual positions up to
`0.125` WE, but entries are blocked or cropped once projected same-side TWE would exceed `0.9`.
This is not the same as reducing raw TWEL to `0.9`, because reducing raw TWEL would also shrink
per-position WEL.

## Global TWEL Entry Gate

If the TWEL enforcer is disabled, do not block entries via TWEL and do not emit TWEL auto-reduce
orders.

For v8, define TWEL enabled as:

```text
raw_twel > 0.0
effective_n_positions > 0
total_exposure_enforcer_threshold > 0.0
```

When TWEL is enabled, every policy must enforce:

```text
Do not allow an entry whose projected fill would make same-side TWE > effective_twel.
```

The gate should crop the last admissible entry when possible so projected TWE lands near
`effective_twel` without exceeding it. If the cropped quantity violates effective min qty or min
cost, drop the entry.

This gate is the main normal-path behavior. If the bot operates from a valid state and all entries
go through this gate, entries should not be able to trigger TWEL auto-reduce by themselves.

## Auto-Reduce Trigger

TWEL auto-reduce becomes a repair path, not the normal path:

```text
if current_same_side_TWE > effective_twel:
    emit TWEL auto-reduce orders according to total_exposure_enforcer_policy
```

Common reasons current TWE can exceed `effective_twel` even with entry gating:

- realized losses reduce account balance
- withdrawals reduce account balance
- price movement increases exposure
- WEL enforcer or auto-unstuck realizes losses
- manual orders or exchange-side state changes alter positions
- restart observes an already over-target account
- rounding, min qty, min cost, or partial fills leave the account above target

## Config Parameter

Preferred v8 parameter name:

```text
bot.long.risk.total_exposure_enforcer_policy
bot.short.risk.total_exposure_enforcer_policy
```

If temporary flat runtime fields are needed at the Python/Rust boundary, keep one canonical
user-facing field and translate to the internal representation during config compilation.

Do not add user-facing aliases unless a released-version migration requires them.

Recommended allowed values:

```text
block_entries_only
reduce_overweight
reduce_portfolio
```

Recommended default:

```text
reduce_overweight
```

## Policy: `block_entries_only`

Behavior:

1. Enforce the global TWEL entry gate at `effective_twel`.
2. Emit no TWEL auto-reduce orders.

This is the pure stop-and-wait policy. It is useful for users who want thresholded TWEL as a
portfolio entry budget, but do not want TWEL to realize losses automatically.

Threshold semantics:

- `total_exposure_enforcer_threshold` still matters.
- Entries are blocked or cropped above `raw_twel * total_exposure_enforcer_threshold`.
- Raw TWEL still controls per-position effective WEL.

## Policy: `reduce_overweight`

Behavior:

1. Enforce the global TWEL entry gate at `effective_twel`.
2. If current TWE exceeds `effective_twel`, reduce positions whose WE exceeds the thresholded
   per-slot target:

```text
overweight_target = effective_twel / effective_n_positions
candidate if WE > overweight_target
```

3. Emit auto-reduce orders for all candidates.
4. Size candidate reductions to approximate equal adverse realized loss while reducing TWE toward
   `effective_twel`.

This is the conservative portfolio repair policy. It focuses on positions consuming more than their
thresholded fair share of the portfolio budget, while avoiding healthy small positions when possible.

Example:

```text
positions WE: 0.23, 0.65, 0.41
raw_twel: 1.29
twel_threshold: 0.97
effective_twel: 1.2513
n_positions: 3
overweight_target: 0.4171
```

Only the `0.65` WE position is above `0.4171`, so only it is a candidate. The `0.41` and `0.23`
positions are below their thresholded per-slot target.

## Policy: `reduce_portfolio`

Behavior:

1. Enforce the global TWEL entry gate at `effective_twel`.
2. If current TWE exceeds `effective_twel`, all open same-side positions are candidates.
3. Emit auto-reduce orders for all candidates.
4. Size reductions to approximate equal adverse realized loss while reducing TWE toward
   `effective_twel`.

This is the true portfolio-wide deleverager. It is appropriate when the user wants
`total_exposure_enforcer_threshold < 1.0` to mean "reduce the whole same-side book if the portfolio
is over target."

Profitable positions are candidates. Their adverse realized loss is zero:

```text
adverse_loss = max(0.0, -projected_realized_pnl)
```

Do not let profitable closes create negative loss credit that permits larger losses elsewhere,
unless a future explicit policy chooses to do that.

## Equal Adverse Loss Sizing

For `reduce_overweight` and `reduce_portfolio`, the sizing objective is not equal exposure
reduction. It is equal adverse realized loss.

Rationale:

- Reducing `0.01` WE from a position 10% underwater realizes much more loss than reducing `0.01`
  WE from a position 1% underwater.
- Equal adverse loss naturally reduces more exposure from shallow-underwater positions and less
  from deep-underwater positions.
- Profitable or breakeven positions can provide exposure relief without adverse loss.

Implementation can use an iterative water-filling style algorithm:

1. Build the policy candidate set.
2. Compute required TWE reduction:

```text
required_reduction = current_TWE - effective_twel
```

3. Emit at least one reduce order per candidate.
4. Clamp each order to exchange min qty and min cost, even if that makes the order larger than its
   ideal equal-loss slice.
5. Project the resulting post-reduction state.
6. If more reduction is needed, allocate additional reductions by raising the adverse-loss target
   across remaining candidates.
7. Remove a candidate from the allocation pool when it hits full close, min/step constraints, or
   another hard cap, then redistribute the remaining reduction target.
8. Stop once projected TWE is `<= effective_twel` or no candidate can reduce further.

Small-wallet rule:

```text
If a candidate exists, keep emitting its TWEL auto-reduce order even when min qty/min cost makes the
order chunkier than the ideal equal-loss target.
```

This avoids silently concentrating all TWEL repair on only the candidates whose ideal slice happens
to clear exchange minimums.

## Realized-Loss Gate Interaction

TWEL auto-reduce orders should still pass through the existing realized-loss gate unless v8
explicitly chooses to make TWEL an emergency bypass.

Recommended first contract:

```text
max_realized_loss_pct may block lossy TWEL auto-reduce orders.
If it blocks TWEL repair while current TWE remains above effective_twel, emit a loud risk warning.
```

The warning should make clear:

- side
- current TWE
- effective TWEL target
- policy
- number of TWEL candidates
- number of TWEL orders blocked by loss gate
- projected TWE after allowed reductions

Do not silently downgrade to neutral behavior.

## Min Qty, Min Cost, Rounding, And Over-Reduction

TWEL repair is allowed to over-reduce when exchange constraints force it:

- reduce-only order qty must respect qty step
- reduce-only order cost should respect effective min cost
- if a position is smaller than effective min qty, closing the whole position remains acceptable
- all reduce orders must be capped at live position size

This means `reduce_overweight` and `reduce_portfolio` may reduce TWE below `effective_twel` on small
wallets. That is preferable to skipping candidates and repeatedly trimming only one position.

## Disabled And Invalid Config Semantics

Preserve existing inactive-side semantics:

```text
raw_twel == 0.0 means the side is intentionally inactive for TWEL purposes.
```

Invalid values should fail loudly in v8 config validation or Rust input validation:

- non-finite TWEL
- negative TWEL
- non-finite threshold
- negative threshold
- enabled TWEL with zero effective positions

If `total_exposure_enforcer_threshold <= 0.0`, treat the total exposure enforcer as disabled for
that side unless v8 chooses a stricter validation rule.

## Suggested Implementation Areas

Rust remains the source of truth for this behavior:

- `passivbot-rust/src/types.rs`
  - add the policy enum/string representation
  - expose through PyO3 JSON parsing

- `passivbot-rust/src/python.rs`
  - parse `total_exposure_enforcer_policy`
  - validate unknown policy names loudly

- `passivbot-rust/src/orchestrator.rs`
  - apply global entry gate using `effective_twel`
  - call TWEL reducer only when current TWE exceeds `effective_twel`
  - pass the selected policy to the reducer

- `passivbot-rust/src/risk.rs`
  - replace or extend `calc_twel_enforcer_actions`
  - keep candidate selection separate from sizing
  - add equal adverse-loss sizing helpers

- `src/config/`
  - add schema/defaults/normalization for v8 config
  - support any temporary alias shape if needed

- docs
  - update user-facing risk/TWEL docs
  - add `CHANGELOG.md` entry for the v8 contract change

## Tests

Add focused Rust/Python tests around the JSON orchestrator boundary:

1. Entry gate crops the last entry so projected TWE is `<= effective_twel`.
2. Entry gate drops the last entry when cropped qty is below effective min qty/cost.
3. `block_entries_only` emits no TWEL auto-reduce orders even when current TWE is above target.
4. `reduce_overweight` selects only positions with `WE > effective_twel / effective_n_positions`.
5. `reduce_overweight` emits one reduce order for every overweight candidate.
6. `reduce_portfolio` emits one reduce order for every open same-side position.
7. Equal adverse-loss sizing reduces more WE from shallow-underwater positions than from
   deep-underwater positions for the same adverse loss.
8. Profitable candidates in `reduce_portfolio` have zero adverse loss and no negative loss credit.
9. Min qty/min cost clamping still emits orders for all candidates, even when larger than ideal.
10. Realized-loss gate can block TWEL orders and surfaces diagnostics/warnings.
11. `raw_twel == 0.0` disables TWEL for that side.
12. Invalid/non-finite/negative TWEL inputs fail loudly.

## Open Questions

1. Should `total_exposure_enforcer_threshold > 1.0` be allowed in v8?
   - If allowed, `effective_twel` would exceed raw TWEL, but entries should probably still never
     exceed raw TWEL.
   - Simpler v8 rule: clamp entry cap to `min(raw_twel, effective_twel)`.

2. Should `block_entries_only` be allowed with `total_exposure_enforcer_threshold > 1.0`?
   - If the threshold is above raw TWEL, the policy becomes equivalent to raw TWEL entry gating.

3. Should TWEL repair bypass the realized-loss gate in emergency mode?
   - First recommendation is no: let the gate block, but warn loudly.
   - If future users want liquidation-avoidance behavior, add an explicit bypass config rather than
     hiding it inside `total_exposure_enforcer_policy`.

4. Should `reduce_portfolio` include positions in `tp_only` and `graceful_stop`?
   - It should include managed open positions unless mode semantics explicitly block all closes.
   - It should not override `manual` or `panic` without a clear mode contract.

5. Should policies be optimizer-searchable?
   - If yes, keep the enum small and document that policy changes can radically alter behavior.

## Summary Contract

```text
WEL enforcer:
    per-position trim/refill mechanism; may intentionally oscillate around effective WEL.

TWEL enforcer:
    portfolio governor; entries cannot refill above effective TWEL.
    auto-reduce is a repair path for already-over-target states.

total_exposure_enforcer_policy:
    controls how TWEL repair distributes reductions once current TWE is already above target.
```
