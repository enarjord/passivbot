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
    long  : min(best_bid, EMA_low * (1 - entry_initial_ema_dist))
    short : max(best_ask, EMA_high * (1 + entry_initial_ema_dist))

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

When aggregated realised PnL falls below the peak by more than
`unstuck_loss_allowance_pct * total_wallet_exposure_limit`, one position at a time is
selected for loss realization:

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

The total exposure enforcer keeps the sum of bot-scope exposures below
`total_wallet_exposure_limit * total_exposure_enforcer_threshold`. For each
position:

```text
if not total_exposure_enforcer_enabled:
    disabled

exposure_i      = wallet_exposure(...)
floor_i         = min(wel_base_i,
                      total_wallet_exposure_limit * total_exposure_enforcer_threshold
                      / effective_n_positions)
floor_psize_i   = floor_i * balance / (price_i * c_mult_i)
```

While bot-scope `Σ exposure_i` exceeds the threshold:

1. Pick the position with the smallest relative price move from entry.
2. First reduce only exposure above `floor_i`.
3. If total exposure is still above target, continue reducing below that floor.

The final reduce-only order is:

```text
qty  = sign(pside) * min(reduced_psize, |pos.size|)
price ≈ market_price
order_type = CloseAutoReduceTwel{Long,Short}
```

By construction the quantity never exceeds the live position size.
Positions already earmarked for `CloseAutoReduceWel*` during the same scheduling cycle are skipped so that reductions do not double-up; they can be considered again on subsequent iterations once the WEL order has been filled.

The first reduction pass respects the per-position floor as a fairness rule. If
bot-scope total exposure is still above target, a second pass continues reducing
least-stuck bot-managed positions even below that floor. If exchange min-qty,
min-cost, or rounding constraints prevent reaching the target, the reducer emits
a warning.

Setting `total_exposure_enforcer_enabled = false` disables this enforcer. When it
is enabled, `total_exposure_enforcer_threshold` must be finite and greater than
zero.

Manual-mode positions are outside bot scope: the bot does not create or cancel
orders for them, and they do not count toward active slots, total exposure
accounting, auto unstuck, or either exposure enforcer.

## Risk-Control Stack

Risk controls are layered from ordinary position management to emergency
intervention:

1. Close logic and negative markup handle normal position reduction.
2. Auto unstuck reduces stuck positions only when loss allowance and EMA gating permit it.
3. Position exposure enforcer trims individual positions to enforce or actively recycle per-position exposure.
4. Total exposure enforcer keeps bot-scope portfolio exposure under the configured total limit.
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
| `total_exposure_enforcer_threshold`             | Bot-scope portfolio exposure cap                         | `floor_i`, `qty` |

For worked examples on a per-parameter basis, see the comments sprinkled in
`passivbot-rust/src/entries.rs` and the optimiser notebooks under `notebooks/`.
