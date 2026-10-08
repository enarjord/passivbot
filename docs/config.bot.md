# Bot Configuration Deep Dive

This note explains how the parameters under `config.bot.{long,short}` are consumed by
Passivbot’s Rust core.  For brevity we omit rounding to exchange precision, guard rails,
and boolean checks; the pseudo-code below mirrors the algebra in
`passivbot-rust/src/entries.rs`, `closes.rs`, and `risk.rs`.

Throughout:

* `pside ∈ {long, short}`
* `EMA_low, EMA_high` are the minima / maxima of the three EMA spans
  (`ema_span_0`, `ema_span_1`, `sqrt(ema_span_0 * ema_span_1)`).
  Strategy logic uses the active strategy's spans; auto-unstuck uses `unstuck.ema_span_0/1`.
* `pos.price`, `pos.size` are the current average entry price and signed quantity
  (`>0` long, `<0` short).
* `wallet_exposure(balance, size, price, c_mult)` returns `abs(size) * price * c_mult / balance`.
* `wel_base` abbreviates `wallet_exposure_limit` for the symbol and pside.
  In live mode this is derived from a fixed denominator (`n_positions`); in backtests it may be
  fixed or tradability-driven depending on `backtest.dynamic_wel_by_tradability`.
* Excess allowance is always bounded:
  `we_excess_effective = min(max(0, risk_we_excess_allowance_pct),
  max(0, total_wallet_exposure_limit / wel_base - 1))`.
  If `wel_base` is non-positive/non-finite, Passivbot treats the effective
  allowance and allowed exposure as zero. If `total_wallet_exposure_limit` is
  non-positive/non-finite, Passivbot grants no excess headroom.
* `wel_allowed = wel_base * (1 + we_excess_effective)`, so per-position excess allowance never
  expands a base WEL past the side's configured `total_wallet_exposure_limit`.
  An authored base WEL already above TWEL receives no excess headroom; the headroom
  clamp does not reduce that authored base itself.

## Trailing Martingale Entries

Price EMA horizons are `bot.<side>.strategy.trailing_martingale.entry.ema_span_0`
and `entry.ema_span_1`. They govern entry gating, forager entry readiness, and the one-way
entry tie-break. Ordinary closes do not consume this band. The shared
`volatility_ema_span_1m/1h` remain at strategy level because both entries and closes use them.

Schema v8.4.0 moves the old strategy-root spans into `entry`, including optimizer bounds
and coin/scenario overrides. Values and floating-point precision are preserved. Within one
source, explicit new paths win over conflicting old paths with a warning; normal file/inline
coin precedence is preserved. Old CLI paths and dotted optimizer selectors still work.
Selecting `long.strategy.entry` now includes the entry EMA horizons. Other strategies retain
their existing schema. `couple_unstuck_ema_spans` continues to derive unstuck horizons from
each coin's effective entry spans during optimization.

```text
alpha(span)         = 2 / (span + 1)
entry_threshold_vol_term =
    volatility_ema_1h * entry.threshold_volatility_1h_weight
  + volatility_ema_1m * entry.threshold_volatility_1m_weight
entry_threshold_we_term = (wel / wel_base) * entry.threshold_we_weight
entry_threshold_multiplier = max(1, 1 + entry_threshold_vol_term + entry_threshold_we_term)

entry_retracement_vol_term =
    volatility_ema_1h * entry.retracement_volatility_1h_weight
  + volatility_ema_1m * entry.retracement_volatility_1m_weight
entry_retracement_we_term = (wel / wel_base) * entry.retracement_we_weight
entry_retracement_multiplier = max(1, 1 + entry_retracement_vol_term + entry_retracement_we_term)

initial_price(pside) =
    if entry.ema_gate_mode gates initial entries:
        long  : min(best_bid, EMA_low * (1 - entry_initial_ema_dist))
        short : max(best_ask, EMA_high * (1 + entry_initial_ema_dist))
    else:
        long  : best_bid
        short : best_ask

initial_qty(balance) =
    max(min_qty,
        balance * wel_allowed * entry_initial_qty_pct / initial_price)

effective_entry_threshold =
    entry.threshold_base_pct * entry_threshold_multiplier

effective_entry_retracement =
    max(0, entry.retracement_base_pct) * entry_retracement_multiplier

next_entry_price(pside) =
    long  : pos.price * (1 - effective_entry_threshold)
    short : pos.price * (1 + effective_entry_threshold)

if entry.ema_gate_mode gates re-entries:
    next_entry_price(long)  = min(next_entry_price(long), EMA_low * (1 - entry_initial_ema_dist))
    next_entry_price(short) = max(next_entry_price(short), EMA_high * (1 + entry_initial_ema_dist))

next_entry_qty(last_fill_qty) =
    last_fill_qty * entry.double_down_factor
```

* Re-entry orders are generated until the wallet exposure implied by the next order would exceed
  `wel_allowed` (base WEL plus the TWEL-capped excess allowance, plus safeguards).
* `entry.double_down_factor > 0` multiplies each successive re-entry quantity; values
  `< 1` still increase size if the preceding fill was larger than the remaining gap to the
  exposure cap.
* When `entry.retracement_base_pct <= 0`, re-entries are passive recursive limit orders.
* When `entry.retracement_base_pct > 0`, the threshold condition must be reached first and the
  order is emitted after retracement confirmation.
* Re-entries are only normal or cropped. Near the effective exposure cap, the bot keeps the
  current order literal instead of pulling future size forward into an inflated terminal step.
* `entry.ema_gate_mode` is fixed config, not optimized. `initial` is the default and preserves the
  previous behavior. `disabled` gates no entries, `all` gates initial, partial-initial, and
  re-entry orders, and `reentry` gates re-entries only.
* In one-way mode, if both sides are flat and otherwise eligible, the long-vs-short tie-break still
  uses EMA-band distance even when `entry.ema_gate_mode = "disabled"`. Missing EMA inputs fail.

Trailing extrema are reset for the coin+pside after any fill. Passivbot tracks its own trailing
state from 1m OHLCVs and does not use exchange-native trailing order types.

## Trailing Martingale Closes

```text
close_threshold =
    close.threshold_base_pct
  + (wel / wel_base) * close.threshold_we_weight
  + volatility_ema_1h * close.threshold_volatility_1h_weight
  + volatility_ema_1m * close.threshold_volatility_1m_weight

close_retracement_multiplier =
    max(1,
        1
      + volatility_ema_1h * close.retracement_volatility_1h_weight
      + volatility_ema_1m * close.retracement_volatility_1m_weight)

close_retracement =
    max(0, close.retracement_base_pct) * close_retracement_multiplier
```

```text
if close.retracement_base_pct <= 0:
    close_price(long)  = max(best_bid, pos.price * (1 + close_threshold))
    close_price(short) = min(best_ask, pos.price * (1 - close_threshold))
else:
    triggered_when(long):
        high_since_open >= pos.price * (1 + close_threshold)
        and
        low_since_high <= high_since_open * (1 - close_retracement)
```

Close orders are recursive when `close.threshold_we_weight != 0`: compute a slice up to
`close.qty_pct`, simulate it filled, recompute `wel / wel_base`, then repeat until the position is
exhausted or the close ladder is complete. If `close.retracement_base_pct <= 0` and
`close.threshold_we_weight == 0`, all recursive closes would have the same price, so Rust emits one
full-position close instead of redundant same-price slices. The removed v7
`close_grid_markup_start` / `close_grid_markup_end` linear TP grid is intentionally not part of the
V8 strategy contract.

## Auto-Unstucking

Auto unstuck is controlled by `bot.<side>.unstuck.enabled`. When disabled, the
unstuck thresholds remain in the config but do not create orders.

`bot.<side>.unstuck.ema_gating_enabled` controls only the EMA trigger/readiness check for
auto-unstuck. It defaults to `true`. When false, auto-unstuck may trigger without EMA bands, but it
still requires loss allowance, exposure threshold, close sizing, and valid market/exchange inputs.
The field may be overridden for an individual coin+side through `coin_overrides`.

`bot.<side>.unstuck.ema_span_0` and `ema_span_1` are positive floating-point EMA spans in
minutes, independent of the active strategy spans. The band contains these two close-price EMAs
and the EMA at their geometric-mean span. Long unstucking requires price at or above the upper
band times `1 + ema_dist` (rounded up to the price tick); short unstucking requires price at or
below the lower band times `1 - ema_dist` (rounded down). These are eligibility gates; the other
unstuck conditions and account-wide candidate selection still apply.

Both spans support coin overrides and optimizer bounds under
`optimize.bounds.<side>.unstuck`. For a composed portfolio with a shared unstucker, set global
unstuck spans and leave these leaves out of the coin overrides.

When loading an older config, missing spans are copied from each side's effective active strategy,
including individual coin strategy overrides and file/inline precedence. This preserves the saved
trading behavior; saving the normalized config makes the independent spans explicit. Explicit new
unstuck spans always win. An inactive legacy side with zero strategy spans receives positive
strategy defaults with a warning to review them before enabling the side.

A former optimizer search with varying strategy spans cannot be migrated one-to-one: those genes
previously moved both bands. Migration copies fixed legacy bounds exactly; for varying ranges it
warns and fixes missing new unstuck bounds at the starting values. Set new bounds explicitly to tune them independently, or add
`couple_unstuck_ema_spans` to `optimize.enable_overrides` to restore coupled search. Migrated per-coin unstuck span overrides remain
pinned; remove those leaves deliberately to tune a shared global pair. Restart optimizer searches
rather than resuming old checkpoints after this schema change.

Apple MPS screening supports independent unstuck horizons and bounds for both supported strategies,
including per-coin overrides and candle-interval scaling. Start a fresh GPU run after upgrading;
older screening checkpoints use a different parameter layout.

Auto-unstuck requires a positive remaining realized-loss allowance. It does not wait for equity
to fall below a loss floor. The allowance is reconstructed from the account-wide realized-PnL
window configured by `live.pnls_max_lookback_days`:

```text
balance_peak = balance + (realized_pnl_cumsum_max - realized_pnl_cumsum_last)
loss_fraction = unstuck.loss_allowance_pct * total_wallet_exposure_limit
remaining_allowance = max(0, balance - balance_peak * (1 - loss_fraction))
```

If `coin_overrides.<coin>.bot.<side>.unstuck.loss_allowance_pct` is set, that
coin+side uses the override percentage in this same account-wide allowance formula.
The override does not create a per-slot budget or separate per-coin realized-PnL tracking.

With auto-unstuck enabled and positive `loss_allowance_pct`, `close_pct`, `threshold` and side
TWEL, positions must pass the exposure test and, when enabled, the EMA gate described above:

```text
effective_wel = wallet_exposure_limit * (1 + effective_we_excess_allowance_pct)
wallet_exposure = abs(position_size) * position_price * c_mult / balance
eligible_exposure = wallet_exposure / effective_wel > unstuck.threshold
```

Here `position_price` is average entry price. The threshold is an eligibility trigger only;
equality does not qualify. It is not a target remaining exposure, and Rust does not cap an
unstuck close at the quantity needed to reach it.

For the selected candidate, Rust uses current price rounded up to the price tick for a long
close or down for a short close. Before quantity rounding and other constraints:

```text
close_qty_abs = balance * effective_wel * unstuck.close_pct / (close_price * c_mult)
```

This sizes a fraction of the effective exposure budget at the close price, not a fraction of
the current position. Rust rounds quantity down, applies exchange minimums and position sizing,
and scales loss-making closes when their estimated loss exceeds the remaining allowance.
Minimum sizing can exceed the remaining allowance; see the
[loss-allowance contract](risk_management.md#auto-unstuck-loss-allowance-contract).

At unchanged balance and effective WEL, the nominal reduction in `wallet_exposure / effective_wel`
is `close_pct * position_price / close_price`, before rounding and other constraints. For example,
a long at 100% of effective WEL with `threshold = 0.90`, `close_pct = 0.12`, entry price 110 and
close price 100 would fall to about 86.8%, not 90%. Realized losses, fees or other balance changes
also change the denominator of the post-fill exposure ratio.

These formulas describe [Rust's unstuck calculation](../passivbot-rust/src/risk.rs);
[the allowance helper](../passivbot-rust/src/utils.rs) supplies the realized-loss budget.

When multiple positions are eligible, auto-unstuck chooses the least stuck
position first, defined as the lowest pside-aware relative distance between
position price and market price. TWEL enforcer uses the same selector.

`unstuck_ema_dist` must keep the EMA-derived trigger price positive:
- `bot.long.unstuck.ema_dist > -1.0`
- `bot.short.unstuck.ema_dist < 1.0`

Configs that cross those boundaries now hard-fail during validation instead of silently
disabling auto-unstuck. For near-always-on EMA triggering on either side, use a value like
`-0.99`, not `-1.0`.

## Position Exposure Enforcer

The position exposure enforcer trims individual bot-managed positions whenever
their exposure rises above the allowance-adjusted per-position cap:

```text
if not position_exposure_enforcer_enabled:
    disabled

allowed_i    = wel_base_i * (1 + we_excess_effective_i)
target_i     = allowed_i * position_exposure_enforcer_threshold
```

If `exposure_i > target_i`, the bot submits a reduce-only order sized just large
enough (with step rounding and minimum-qty guards) to bring the position back to
`target_i`. These orders are emitted as `CloseAutoReduceWel{Long,Short}` and are
returned directly from `calc_next_close_*`/`calc_closes_*`.

Setting `position_exposure_enforcer_enabled = false` disables this enforcer. When
it is enabled, `position_exposure_enforcer_threshold` must be finite and greater
than zero. Values below `1.0` force continuous trimming; values above `1.0`
create an additional grace margin.

This is both a risk control and a possible strategy mechanism. For example, a
user may deliberately set `position_exposure_enforcer_threshold = 0.95` with
aggressive entries. The strategy can refill toward the per-position limit, and
the enforcer can repeatedly trim back to `95%` of that limit. Unlike auto
unstuck, this trim is not gated by EMA distance or loss allowance.

## Total Exposure Enforcer

The total exposure enforcer repairs same-side portfolio exposure when the sum of
all open same-side exchange positions exceeds
`total_wallet_exposure_limit * total_exposure_enforcer_threshold`. Manual and
panic positions count toward this same-side exposure measurement, but they are
not repair candidates and never receive TWEL auto-reduce orders. For each
position:

```text
if not total_exposure_enforcer_enabled:
    disabled

exposure_i      = wallet_exposure(...)
overweight_i       = total_wallet_exposure_limit * total_exposure_enforcer_threshold
                     / n_positions
overweight_psize_i = overweight_i * balance / (price_i * c_mult_i)
```

While same-side `Σ exposure_i` exceeds the threshold:

1. Build the current-TWE measurement from all same-side exchange positions.
2. Build the repair candidate set from managed open positions only: `normal`,
   `graceful_stop`, and `tp_only`.
3. Under `reduce_overweight`, keep only candidates above `overweight_i`. Under
   `reduce_portfolio`, any managed open candidate can be reduced.
4. Prefer profitable or breakeven reductions first, then shallowest adverse-loss
   reductions, with stable symbol tie-breaks.
5. Emit TWEL auto-reduce orders only until projected raw-balance TWE is at or
   below the repair target.

The final reduce-only order is:

```text
qty  = sign(pside) * min(reduced_psize, |pos.size|)
price ≈ market_price
order_type = CloseAutoReduceTwel{Long,Short}
```

By construction the quantity never exceeds the live position size.
TWEL auto-reduce is computed before WEL auto-reduce. If a position receives a
TWEL auto-reduce order, the WEL enforcer skips that position for the same
scheduling cycle.

`reduce_overweight` respects the thresholded per-position target as a candidate
filter. `reduce_portfolio` is the broader deleveraging policy and may reduce a
managed position below that target when needed to bring same-side TWE back to the
repair target. Exchange min-qty, min-cost, rounding, or a lack of managed
candidates can leave the account above target.

Setting `total_exposure_enforcer_enabled = false` disables this enforcer. When it
is enabled, `total_exposure_enforcer_threshold` must be finite and greater than
zero.

Manual and panic positions are outside normal bot management: the bot does not
create or cancel ordinary orders for them, and they do not count toward active
slots, auto unstuck, or WEL/TWEL candidate selection. They still count toward
same-side TWEL measurement and toward the TWEL entry-gate baseline.

## Close Reducer Compatibility

Passivbot selects at most one protective reducer for each coin+pside in one
ideal-order batch. Competing panic, TWEL/WEL auto-reduce, and auto-unstuck
intents are consolidated by keeping the largest loss-admissible final absolute
reduction after position/minimum sizing, not their sum. If the realized-loss
gate blocks the largest non-panic intent, Passivbot tries the next-largest
intent. Equal-size ties keep panic first and otherwise prefer the
closest-to-fill candidate. A full-position panic close remains exclusive.

Ordinary strategy closes are independent reduction intent and may coexist with
the selected non-panic reducer. The reducer quantity is reserved first; ordinary
grid, trailing, or EMA-anchor closes are kept within the remaining position
quantity and trimmed furthest-from-fill first if necessary. The aggregate remains
reduce-only and capped to the position, and the realized-loss gate evaluates
the final reducer quantities largest-first across the batch before ordinary
closes in the resulting mixed close set. Live reconciliation preserves the same
reducer-first reservation if the position shrinks between planning and order
submission.

## Risk-Control Stack

Risk controls are layered from ordinary position management to emergency
intervention:

1. Close logic and negative markup handle normal position reduction.
2. Auto unstuck reduces stuck positions only when loss allowance and EMA gating permit it.
3. Position exposure enforcer trims individual positions to enforce or actively recycle per-position exposure.
4. Total exposure enforcer repairs same-side portfolio exposure using managed candidates.
5. HSL is the equity-level circuit breaker.

## Parameter Interactions at a Glance

| Parameter                                      | Primary effect                                             | Key equations |
| ---------------------------------------------- | -----------------------------------------------------------| ------------- |
| `strategy.<kind>.ema_span_*`                    | Defines the strategy's price EMA band                      | `EMA_low`, `EMA_high` |
| `unstuck.ema_span_*`                            | Defines the independent auto-unstuck EMA band              | `EMA_low`, `EMA_high` |
| `entry_grid_spacing_*`, `entry_grid_double_down_factor` | Controls grid spacing and growth of re-entry quantities | `next_grid_price`, `next_grid_qty` |
| `entry_trailing_*`                             | Adjust trailing entry triggers via exposure & volatility  | `threshold`, `retracement` |
| `close_grid_markup_*`, `close_grid_qty_pct`    | Shapes TP ladder                                          | `tp_prices`, `tp_qty` |
| `close_trailing_*`                             | Mirrors trailing entries but for exits                    | `threshold_close`, `retracement_close` |
| `unstuck_*`, `unstuck_enabled`                 | Loss realization rules                                    | `unstuck_allowed`, `close_price` |
| `position_exposure_enforcer_threshold`, `risk_we_excess_allowance_pct` | Per-position exposure cap                                | `target_i`, `qty` |
| `total_exposure_enforcer_threshold`             | Same-side portfolio exposure repair target               | `overweight_i`, `qty` |

For worked examples on a per-parameter basis, see the comments sprinkled in
`passivbot-rust/src/entries.rs` and the optimiser notebooks under `notebooks/`.
