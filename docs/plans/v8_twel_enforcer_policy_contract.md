# v8 Total Exposure Enforcer Policy Contract

## Status

Draft handoff for a future v8 implementation. This is not a v7 bug-fix spec.

The v7 behavior discussed in GitHub issue #600 can remain unchanged short term unless a separate
v7 mitigation is explicitly chosen. v8 is already a breaking upgrade, so the total exposure
enforcer contract can be redesigned there without preserving v7 semantics.

In this document, TWEL means total wallet exposure limit. The v8 user-facing config names are
`bot.<pside>.risk.total_wallet_exposure_limit`,
`bot.<pside>.risk.total_exposure_entry_gate_enabled`,
`bot.<pside>.risk.total_exposure_enforcer_enabled`,
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
   - When the entry gate is enabled, TWEL reductions should not be immediately undone by
     TWEL-blind entries.
   - When the entry gate is disabled, users explicitly accept that entries may refill or exceed
     TWEL.
   - TWEL should not become an unbounded loss-harvesting loop.

3. Keep WEL and TWEL contracts separate:
   - WEL may deliberately oscillate around effective WEL.
   - TWEL entry gating is the optional portfolio entry cap.
   - TWEL auto-reduce is the optional portfolio repair path for over-target states.

4. Keep the v8 policy surface small:
   - Avoid five or more policies unless real usage proves they are needed.
   - Prefer a small set of clear policy names.

5. Keep behavior stateless:
   - Decisions must be derived from exchange state, config, and current market inputs.
   - Do not add cooldown state or local memory to suppress entries.

## Core Definitions

Use raw TWEL for per-position sizing, a capped threshold for entry gating, and thresholded TWEL for
auto-reduce repair:

```text
raw_twel = bot.{pside}.risk.total_wallet_exposure_limit
twel_threshold = bot.{pside}.risk.total_exposure_enforcer_threshold
twel_repair_target = raw_twel * twel_threshold
twel_entry_cap = min(raw_twel, twel_repair_target)
configured_wel_base = raw_twel / configured_n_positions
effective_wel = min(raw_twel, configured_wel_base * (1 + risk_we_excess_allowance_pct))
```

Important distinction:

- Live uses configured `n_positions` for WEL base sizing. It must not change WEL denominators
  because the current number of open positions differs from configured `n_positions`.
- Backtest `dynamic_wel_by_tradability` is the explicit exception. It may change runtime WEL
  denominators to handle early backtest periods with fewer tradable coins than later periods.
- Standard v8 `risk_we_excess_allowance_mode=bounded` caps `effective_wel <= raw_twel`.
- `risk_we_excess_allowance_mode=legacy_raw` exists only for improved v7 compatibility through
  `trailing_grid_v7`; it may make WEL exceed raw TWEL, but it never bypasses TWEL entry gating or
  TWEL auto-reduce.
- `twel_entry_cap` governs portfolio-level entry blocking when the entry gate is enabled.
- `twel_repair_target` governs TWEL auto-reduce when auto-reduce is enabled.
- Entry-gate TWE is computed from snapped/hysteresis balance.
- Auto-reduce TWE is computed from raw balance.

Example:

```text
raw_twel = 1.0
n_positions = 10
risk_we_excess_allowance_pct = 0.25
total_exposure_enforcer_threshold = 0.9

effective_wel = 1.0 / 10 * 1.25 = 0.125
twel_repair_target = 1.0 * 0.9 = 0.9
twel_entry_cap = min(1.0, 0.9) = 0.9
```

With `total_exposure_enforcer_threshold = 0.9`, the bot may still size individual positions up to
`0.125` WE, but entries are blocked or cropped once projected same-side TWE would exceed `0.9`
when the entry gate is enabled.
This is not the same as reducing raw TWEL to `0.9`, because reducing raw TWEL would also shrink
per-position WEL.

With `total_exposure_enforcer_threshold = 1.05`, the auto-reduce repair target is `1.05`, but the
entry cap remains raw TWEL:

```text
twel_repair_target = 1.0 * 1.05 = 1.05
twel_entry_cap = min(1.0, 1.05) = 1.0
```

That creates a deliberate buffer: the bot will not willingly enter above raw TWEL when the entry
gate is enabled, but it will not auto-reduce until same-side TWE exceeds `raw_twel * threshold`.

## Global TWEL Entry Gate

TWEL entry gating and TWEL auto-reduce are separate controls:

```text
bot.long.risk.total_exposure_entry_gate_enabled
bot.short.risk.total_exposure_entry_gate_enabled

bot.long.risk.total_exposure_enforcer_enabled
bot.short.risk.total_exposure_enforcer_enabled
```

The entry gate prevents new bot entries from pushing projected same-side TWE above
`twel_entry_cap`. Auto-reduce emits repair closes when existing same-side TWE exceeds
`twel_repair_target`.

If `total_exposure_entry_gate_enabled=false`, do not block entries via TWEL. In that mode,
positive excess allowance may allow the bot to place entries which, if filled, push same-side TWE
above raw TWEL. This is an explicit user opt-out from the TWEL entry cap and must be documented as
such.

If `total_exposure_enforcer_enabled=false`, emit no TWEL auto-reduce orders.

Combinations:

```text
entry_gate=false, enforcer=false:
    no TWEL entry cap and no TWEL auto-reduce repair

entry_gate=true, enforcer=false:
    entries cannot project above twel_entry_cap; no TWEL auto-reduce repair

entry_gate=false, enforcer=true:
    entries are not TWEL-capped; auto-reduce repairs above twel_repair_target

entry_gate=true, enforcer=true:
    entries cannot project above twel_entry_cap; auto-reduce repairs above twel_repair_target
```

For v8, the TWEL side is active for either gate when:

```text
raw_twel > 0.0
configured_n_positions > 0
total_exposure_enforcer_threshold > 0.0
```

When the entry gate is enabled, every policy must enforce:

```text
Do not allow an entry whose projected fill would make same-side TWE > twel_entry_cap.
```

The gate should crop the last admissible entry when possible so projected snapped-balance TWE lands
near `twel_entry_cap` without exceeding it. If the cropped quantity violates effective min qty or min
cost, drop the entry.

This gate is the main normal-path behavior when enabled. If the bot operates from a valid state and
all entries go through this gate, entries should not be able to push same-side TWE above raw TWEL by
themselves. If `twel_threshold < 1.0`, entries also cannot refill the band that TWEL repair is
trying to reduce.

## Auto-Reduce Trigger

TWEL auto-reduce becomes a repair path, not the normal path:

```text
if total_exposure_enforcer_enabled and current_same_side_TWE > twel_repair_target:
    emit TWEL auto-reduce orders according to total_exposure_enforcer_policy
```

Common reasons current TWE can exceed `twel_repair_target` even with entry gating:

- realized losses reduce account balance
- withdrawals reduce account balance
- price movement increases exposure
- WEL enforcer or auto-unstuck realizes losses
- manual orders or exchange-side state changes alter positions
- restart observes an already over-target account
- rounding, min qty, min cost, or partial fills leave the account above target

## Mode Semantics

TWEL entry gating applies only to bot-generated entry candidates. It does not govern external,
manual, or already-open exchange orders that are intentionally outside entry generation. Existing
same-side exchange exposure still counts toward the entry-gate TWE baseline.

Mode interactions:

- `normal`: generate entries and closes normally; TWEL entry gating applies to bot-generated
  entries when enabled.
- `graceful_stop`: block initial entries when there is no position. With an existing position,
  behave like `normal`; continuation entries remain bot-generated and should pass through TWEL entry
  gating when enabled.
- `tp_only`: generate no new bot entries, but continue close-side management for existing
  positions. Existing or operator-created non-reduce-only orders are operator risk and are not
  governed by the TWEL entry gate.
- `manual`: generate no bot entries or closes for that position side; TWEL auto-reduce should not
  override manual mode.
- `panic`: generate panic close behavior only; TWEL auto-reduce should not compete with panic mode.

TWEL auto-reduce measures current same-side TWE from all open same-side exchange positions,
including `manual` and `panic`. `reduce_portfolio` and `reduce_overweight` choose repair
candidates only from managed open positions in `normal`, `graceful_stop`, and `tp_only`; they do
not emit TWEL auto-reduce orders for `manual` or `panic`.

## Config Parameters

Preferred v8 parameter names:

```text
bot.long.risk.total_exposure_entry_gate_enabled
bot.short.risk.total_exposure_entry_gate_enabled

bot.long.risk.total_exposure_enforcer_enabled
bot.short.risk.total_exposure_enforcer_enabled

bot.long.risk.total_exposure_enforcer_threshold
bot.short.risk.total_exposure_enforcer_threshold

bot.long.risk.total_exposure_enforcer_policy
bot.short.risk.total_exposure_enforcer_policy
```

If temporary flat runtime fields are needed at the Python/Rust boundary, keep one canonical
user-facing field and translate to the internal representation during config compilation.

Do not add user-facing aliases unless a released-version migration requires them.

`total_exposure_enforcer_policy` controls only TWEL auto-reduce candidate selection. It is global
per side, not per coin. Do not allow coin overrides for TWEL entry-gate enabled, TWEL auto-reduce
enabled, TWEL threshold, or TWEL policy unless a future portfolio-level override contract is
explicitly designed.

The TWEL policy enum and TWEL boolean flags are not optimizer-searchable. Optimizer bounds remain
for numeric parameters only, with integer parameters handled explicitly where supported.

Recommended policy values:

```text
reduce_overweight
reduce_portfolio
```

Recommended default:

```text
reduce_overweight
```

## Policy: `reduce_overweight`

Behavior:

1. If the entry gate is enabled, enforce the global TWEL entry gate at `twel_entry_cap`.
2. If auto-reduce is enabled and current TWE exceeds `twel_repair_target`, reduce positions whose
   WE exceeds the thresholded per-slot target:

```text
overweight_target = twel_repair_target / configured_n_positions
candidate if WE > overweight_target
```

3. Evaluate candidates in deterministic reducer order and emit auto-reduce orders until projected
   raw-balance TWE is `<= twel_repair_target`.
4. Size candidate reductions with the initial deterministic reducer described below while reducing
   TWE toward `twel_repair_target`.

This is the conservative portfolio repair policy. It focuses on positions consuming more than their
thresholded fair share of the portfolio budget, while avoiding healthy small positions when possible.

Example:

```text
positions WE: 0.23, 0.65, 0.41
raw_twel: 1.29
twel_threshold: 0.97
twel_repair_target: 1.2513
n_positions: 3
overweight_target: 0.4171
```

Only the `0.65` WE position is above `0.4171`, so only it is a candidate. The `0.41` and `0.23`
positions are below their thresholded per-slot target.

## Policy: `reduce_portfolio`

Behavior:

1. If the entry gate is enabled, enforce the global TWEL entry gate at `twel_entry_cap`.
2. If auto-reduce is enabled and current TWE exceeds `twel_repair_target`, all open same-side
   positions are candidates.
3. Evaluate all open same-side positions as candidates, but emit auto-reduce orders only until
   projected raw-balance TWE is `<= twel_repair_target`.
4. Size reductions with the initial deterministic reducer described below while reducing TWE toward
   `twel_repair_target`.

This is the true portfolio-wide deleverager. It is appropriate when the user wants
`total_exposure_enforcer_threshold < 1.0` to mean "reduce the whole same-side book if the portfolio
is over target."

Profitable positions are candidates. Their adverse realized loss is zero:

```text
adverse_loss = max(0.0, -projected_realized_pnl)
```

Do not let profitable closes create negative loss credit that permits larger losses elsewhere,
unless a future explicit policy chooses to do that.

## Initial Reducer Sizing

For the first implementation, do not implement full equal adverse-loss water-filling. Keep the
initial reducer simpler and deterministic:

1. Build the policy candidate set.
2. Prefer profitable or breakeven candidates first.
3. Then prefer shallowest adverse-loss candidates before deeper losers.
4. Use stable symbol/order tie-breakers for deterministic output.
5. Reduce candidates until projected raw-balance TWE is `<= twel_repair_target` or no candidate can
   reduce further.
6. Emit at most one TWEL auto-reduce order per position in a single orchestrator pass.

This preserves the practical least-loss behavior while avoiding the complexity of full
equal-adverse-loss allocation in the first contract change.

## Future Equal Adverse Loss Sizing

For a later implementation, `reduce_overweight` and `reduce_portfolio` may move from deterministic
least-loss sizing to equal adverse realized-loss sizing. That future objective is not equal exposure
reduction; it is equal adverse realized loss.

Rationale:

- Reducing `0.01` WE from a position 10% underwater realizes much more loss than reducing `0.01`
  WE from a position 1% underwater.
- Equal adverse loss naturally reduces more exposure from shallow-underwater positions and less
  from deep-underwater positions.
- Profitable or breakeven positions can provide exposure relief without adverse loss.

That future implementation can use an iterative water-filling style algorithm:

1. Build the policy candidate set.
2. Compute required TWE reduction:

```text
required_reduction = current_TWE - twel_repair_target
```

3. Emit at least one reduce order per candidate.
4. Clamp each order to exchange min qty and min cost, even if that makes the order larger than its
   ideal equal-loss slice.
5. Project the resulting post-reduction state.
6. If more reduction is needed, allocate additional reductions by raising the adverse-loss target
   across remaining candidates.
7. Remove a candidate from the allocation pool when it hits full close, min/step constraints, or
   another hard cap, then redistribute the remaining reduction target.
8. Stop once projected TWE is `<= twel_repair_target` or no candidate can reduce further.

Small-wallet rule:

```text
If a candidate has a positive reduction slice, keep emitting its TWEL auto-reduce order even when
min qty/min cost makes the order chunkier than the ideal equal-loss target.
```

This avoids silently concentrating all TWEL repair on only the candidates whose positive ideal slice
happens to clear exchange minimums. Do not force a min-size order for later candidates after the
portfolio has already reached `twel_repair_target`.

## Realized-Loss Gate Interaction

`max_realized_loss_pct` is authoritative for all bot loss taking except panic close orders.
TWEL auto-reduce orders must pass through the existing realized-loss gate. They are not an
emergency bypass.

```text
max_realized_loss_pct may block lossy TWEL auto-reduce orders.
If it blocks TWEL repair while current TWE remains above twel_repair_target, emit a loud risk
warning.
```

The only loss-gate exception is `ClosePanicLong` / `ClosePanicShort`. The exemption is order-type
based: any panic close order bypasses `max_realized_loss_pct`, whether the panic came from HSL
panic-close handling or another explicit panic-mode path.

The warning should make clear:

- side
- current TWE
- TWEL repair target
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

This means `reduce_overweight` and `reduce_portfolio` may reduce TWE below `twel_repair_target` on
small wallets. That is preferable to skipping candidates and repeatedly trimming only one position.

## WEL And TWEL Priority

Emit at most one auto-reduce order per position in a single orchestrator pass.

Recommended first contract:

1. Compute TWEL auto-reduce first.
2. Compute WEL auto-reduce after TWEL.
3. If a position already has a TWEL auto-reduce order, skip WEL auto-reduce for that position.

Rationale:

- TWEL is the same-side portfolio governor.
- WEL remains a per-position trimmer for positions not already selected by portfolio repair.
- One auto-reduce order per position keeps order intent simple. TWEL/WEL auto-reduce orders are
  emitted near market price and usually fill quickly; the next loop can reassess from exchange
  state.

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
- either TWEL entry gate or TWEL auto-reduce enabled with zero configured positions

If either `total_exposure_entry_gate_enabled=true` or `total_exposure_enforcer_enabled=true`,
`total_exposure_enforcer_threshold` must be finite and `> 0.0`.

If `total_exposure_entry_gate_enabled=false`, positive excess allowance may allow bot entries to
push same-side TWE above raw TWEL. If `total_exposure_enforcer_enabled=true`, TWEL auto-reduce may
later repair that state once raw-balance TWE exceeds `twel_repair_target`. If both are false, TWEL
does not enforce the side; WEL and other risk systems still operate according to their own
contracts.

## Suggested Implementation Areas

Rust remains the source of truth for this behavior:

- `passivbot-rust/src/types.rs`
  - add `total_exposure_entry_gate_enabled`
  - add the policy enum/string representation
  - expose through PyO3 JSON parsing

- `passivbot-rust/src/python.rs`
  - parse `total_exposure_entry_gate_enabled`
  - parse `total_exposure_enforcer_policy`
  - validate unknown policy names loudly

- `passivbot-rust/src/orchestrator.rs`
  - apply global entry gate using `twel_entry_cap` only when entry gate is enabled
  - call TWEL reducer only when auto-reduce is enabled and current TWE exceeds
    `twel_repair_target`
  - keep TWEL entry gating limited to bot-generated entries
  - include all same-side open exchange positions in TWEL auto-reduce trigger/current-TWE
    measurement, but select repair candidates only from `normal`, `graceful_stop`, and `tp_only`;
    exclude `manual` and `panic` from emitted TWEL auto-reduce orders
  - keep TWEL auto-reduce subject to `max_realized_loss_pct`; keep `ClosePanic*` exempt
  - compute TWEL auto-reduce before WEL auto-reduce and skip WEL for positions selected by TWEL
  - pass the selected policy to the reducer

- `passivbot-rust/src/risk.rs`
  - replace or extend `calc_twel_enforcer_actions`
  - keep candidate selection separate from sizing
  - implement deterministic profitable/shallow-loss sizing first
  - leave full equal adverse-loss sizing as a future enhancement

- `src/config/`
  - add schema/defaults/normalization for `total_exposure_entry_gate_enabled`
  - add schema/defaults/normalization for `total_exposure_enforcer_policy`
  - do not expose TWEL booleans or policy as optimizer bounds
  - keep TWEL entry-gate enabled, TWEL auto-reduce enabled, TWEL threshold, and TWEL policy
    global per side; do not allow coin overrides
  - support any temporary alias shape if needed

- docs
  - update user-facing risk/TWEL docs
  - add `CHANGELOG.md` entry for the v8 contract change

## Tests

Add focused Rust/Python tests around the JSON orchestrator boundary:

1. Entry gate crops the last entry so projected snapped-balance TWE is `<= twel_entry_cap`.
2. Entry gate drops the last entry when cropped qty is below effective min qty/cost.
3. `total_exposure_entry_gate_enabled=false` allows entries even when projected TWE exceeds raw
   TWEL, including excess-allowance cases.
4. `total_exposure_enforcer_enabled=false` emits no TWEL auto-reduce orders even when current TWE
   is above target.
5. `total_exposure_enforcer_threshold > 1.0` keeps entry cap at raw TWEL and auto-reduce target at
   `raw_twel * threshold`.
6. `total_exposure_enforcer_threshold < 1.0` caps entries at `raw_twel * threshold`.
7. `reduce_overweight` selects only positions with `WE > twel_repair_target / configured_n_positions`.
8. `reduce_overweight` emits reduce orders for overweight candidates until projected TWE reaches
   target; it does not force min-size orders after the target is reached.
9. `reduce_portfolio` evaluates every open same-side managed position as a candidate, but emits
   reduce orders only until projected TWE reaches target.
10. Initial reducer prefers profitable/breakeven candidates, then shallowest adverse-loss
    candidates, with deterministic ties.
11. Min qty/min cost clamping still emits orders for candidates when larger than ideal.
12. Realized-loss gate can block TWEL orders and surfaces TWEL diagnostics/warnings.
13. TWEL auto-reduce is computed before WEL; WEL skips positions that already have TWEL auto-reduce.
14. `raw_twel == 0.0` disables TWEL for that side.
15. Invalid/non-finite/negative TWEL inputs fail loudly.
16. `legacy_raw` excess allowance can make WEL exceed raw TWEL, but does not bypass TWEL entry
    gating or TWEL auto-reduce.
17. `graceful_stop` blocks initial entries only; with an existing position it behaves like `normal`
    and continuation entries are TWEL-gated when the entry gate is enabled.
18. `tp_only` generates no bot entries, preserves operator entry responsibility, and still allows
    TWEL auto-reduce for managed open positions.
19. `manual` and `panic` exposure contributes to TWEL auto-reduce trigger/current-TWE
    measurement, but those positions are excluded from TWEL auto-reduce candidate selection.
20. `ClosePanic*` orders bypass `max_realized_loss_pct`; TWEL auto-reduce orders do not.

## Resolved Decisions

- TWEL repair does not bypass `max_realized_loss_pct`. The realized-loss gate may block lossy TWEL
  auto-reduce orders, and the bot must warn loudly when this leaves TWE above target.
- Panic close orders are the only realized-loss-gate exception. The exemption is tied to
  `ClosePanicLong` / `ClosePanicShort` order types.
- `reduce_portfolio` and `reduce_overweight` include managed open positions in `tp_only` and
  `graceful_stop`.
- `manual` and `panic` exposure is counted for same-side TWEL measurement, but those modes remain
  outside TWEL auto-reduce management.
- `tp_only` is manual/operator-risk for entries and managed for closes. TWEL entry gating therefore
  has no bot-generated entries to gate in `tp_only`.

## Summary Contract

```text
WEL enforcer:
    per-position trim/refill mechanism; may intentionally oscillate around effective WEL.
    computed after TWEL auto-reduce; skipped for positions already selected by TWEL.

TWEL entry gate:
    optional portfolio entry cap.
    when enabled, entries cannot project above min(raw_twel, raw_twel * threshold).
    when disabled, excess allowance may allow entries above raw TWEL.
    applies only to bot-generated entries, not operator/manual exchange orders.

TWEL auto-reduce:
    optional portfolio repair path.
    when enabled, repairs already-over-target states above raw_twel * threshold.
    subject to max_realized_loss_pct; measures manual and panic exposure but excludes manual and
    panic positions from emitted TWEL auto-reduce orders.

Panic close:
    ClosePanicLong and ClosePanicShort bypass max_realized_loss_pct.

total_exposure_enforcer_policy:
    controls how TWEL auto-reduce chooses repair candidates.
    global per side, not coin-overridable, not optimizer-searchable.
```
