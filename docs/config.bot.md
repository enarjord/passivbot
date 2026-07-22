# Bot Configuration Deep Dive

This note explains how the parameters under `config.bot.{long,short}` are consumed by
Passivbot’s Rust core.  For brevity we omit rounding to exchange precision, guard rails,
and boolean checks; the pseudo-code below mirrors the algebra in
`passivbot-rust/src/entries.rs`, `closes.rs`, and `risk.rs`.

Throughout:

* `pside ∈ {long, short}`
* `EMA_low, EMA_high` are the minima / maxima of the three EMA spans
  (`ema_span_0`, `ema_span_1`, `sqrt(ema_span_0 * ema_span_1)`).
* `pos.price`, `pos.size` are the current average entry price and signed quantity
  (`>0` long, `<0` short).
* `wallet_exposure(balance, size, price, c_mult)` returns `abs(size) * price * c_mult / balance`.
* `wel_base` abbreviates `wallet_exposure_limit` for the symbol and pside.
  In live mode this is derived from a fixed denominator (`n_positions`); in backtests it may be
  fixed or tradability-driven depending on `backtest.dynamic_wel_by_tradability`.
* With the default `we_excess_allowance_mode = "bounded"`,
  `we_excess_effective = min(max(0, risk_we_excess_allowance_pct),
  max(0, total_wallet_exposure_limit / wel_base - 1))`.
  If `wel_base` is non-positive/non-finite, bounded mode treats the effective
  allowance and allowed exposure as zero. If `total_wallet_exposure_limit` is
  non-positive/non-finite, bounded mode grants no excess headroom.
  With `we_excess_allowance_mode = "legacy_raw"`, the raw
  `max(0, risk_we_excess_allowance_pct)` is used instead.
* `wel_allowed = wel_base * (1 + we_excess_effective)`, so per-position excess allowance never
  expands a symbol above the side's configured `total_wallet_exposure_limit`.

## Trailing Martingale Entries

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

When aggregated realised PnL falls below the peak by more than
`unstuck_loss_allowance_pct * total_wallet_exposure_limit`, one position at a time is
selected for loss realization:

If `coin_overrides.<coin>.bot.<side>.unstuck.loss_allowance_pct` is set, that
coin+side uses the override percentage in the same account-wide allowance formula
when it is selected for unstucking. The override does not switch unstuck to a
per-slot budget and does not create separate per-coin realized-PnL tracking.

```text
unstuck_allowed = peak_balance * (1 - unstuck_loss_allowance_pct *
                                  total_wallet_exposure_limit)
if equity < unstuck_allowed:
    close_qty   = full_pos_size * unstuck_close_pct
    close_price = EMA_band_opposite *
                    (1 + sign(pside) * unstuck_ema_dist)
```

Positions become eligible when
`wallet_exposure / wel_allowed > unstuck_threshold`.

When multiple positions are eligible, auto-unstuck chooses the least stuck
position first, defined as the lowest pside-aware relative distance between
position price and market price. TWEL enforcer uses the same selector.

`unstuck_ema_dist` must keep the EMA-derived trigger price positive:
- `bot.long.unstuck_ema_dist > -1.0`
- `bot.short.unstuck_ema_dist < 1.0`

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
ideal-order batch. Panic close is exclusive. TWEL/WEL exposure repair takes
precedence over auto-unstuck, and only one competing reducer survives.

Ordinary strategy closes are independent reduction intent and may coexist with
the selected non-panic reducer. The reducer quantity is reserved first; ordinary
grid, trailing, or EMA-anchor closes are kept within the remaining position
quantity and trimmed furthest-from-fill first if necessary. The aggregate remains
reduce-only and capped to the position, and the realized-loss gate evaluates the
selected reducer before ordinary closes in the resulting mixed close set. Live
reconciliation preserves the same reducer-first reservation if the position
shrinks between planning and order submission.

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
| `ema_span_*`                                   | Defines EMA bands used by initial pricing & unstuck levels | `EMA_low`, `EMA_high` |
| `entry_grid_spacing_*`, `entry_grid_double_down_factor` | Controls grid spacing and growth of re-entry quantities | `next_grid_price`, `next_grid_qty` |
| `entry_trailing_*`                             | Adjust trailing entry triggers via exposure & volatility  | `threshold`, `retracement` |
| `close_grid_markup_*`, `close_grid_qty_pct`    | Shapes TP ladder                                          | `tp_prices`, `tp_qty` |
| `close_trailing_*`                             | Mirrors trailing entries but for exits                    | `threshold_close`, `retracement_close` |
| `unstuck_*`, `unstuck_enabled`                 | Loss realization rules                                    | `unstuck_allowed`, `close_price` |
| `position_exposure_enforcer_threshold`, `risk_we_excess_allowance_pct` | Per-position exposure cap                                | `target_i`, `qty` |
| `total_exposure_enforcer_threshold`             | Same-side portfolio exposure repair target               | `overweight_i`, `qty` |

For worked examples on a per-parameter basis, see the comments sprinkled in
`passivbot-rust/src/entries.rs` and the optimiser notebooks under `notebooks/`.
