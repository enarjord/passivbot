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
* `wallet_exposure(balance, size, price, c_mult)` returns `abs(size) * price / (balance * c_mult)`.
* `wel_base` abbreviates `wallet_exposure_limit` for the symbol and pside.

## Initial Entry & Grid Re-entries

```text
alpha(span)         = 2 / (span + 1)
ramp_spacing(wel)   = 1 + (wel / wel_base) * entry_grid_spacing_we_weight + vol_term
vol_term            = log_range_ema * entry_grid_spacing_volatility_weight

initial_price(pside) =
    long  : min(best_bid, EMA_low * (1 - entry_initial_ema_dist))
    short : max(best_ask, EMA_high * (1 + entry_initial_ema_dist))

initial_qty(balance) =
    max(min_qty,
        balance * wel_base * entry_initial_qty_pct / initial_price)

next_grid_price(pside, k) =
    long  : last_fill_price * (1 - entry_grid_spacing_pct * ramp_spacing(wel))^k
    short : last_fill_price * (1 + entry_grid_spacing_pct * ramp_spacing(wel))^k

next_grid_qty(last_fill_qty) =
    last_fill_qty * entry_grid_double_down_factor
```

* Grid orders are generated until the wallet exposure implied by the next order would exceed
  `wel_base` (plus safeguards).
* `entry_grid_double_down_factor > 0` multiplies each successive re-entry quantity; values
  `< 1` still increase size if the preceding fill was larger than the remaining gap to the
  exposure cap.

## Trailing Entries

Trailing entries activate after the position experiences a favourable excursion followed by
a pullback.

```text
threshold = entry_trailing_threshold_pct *
            (1 + entry_trailing_threshold_we_weight * wel_ratio
               + entry_trailing_threshold_volatility_weight * log_range_ema)

retracement = entry_trailing_retracement_pct *
              (1 + entry_trailing_retracement_we_weight * wel_ratio
                 + entry_trailing_retracement_volatility_weight * log_range_ema)

wel_ratio = wallet_exposure(...) / wel_base

triggered_when(pside == long):
    high_since_entry    > pos.price * (1 + threshold)
    and
    low_since_threshold < high_since_entry * (1 - retracement)

reentry_price(long)  = min(best_bid,
                           pos.price * (1 - threshold + retracement))
reentry_price(short) = max(best_ask,
                           pos.price * (1 + threshold - retracement))

reentry_qty = max(initial_qty, calc_reentry_qty(...) *
                 entry_trailing_double_down_factor)
```

Trailing entries are skipped when the resulting exposure would break the wallet-exposure
limit (or the TWEL enforcer, see below).

## Take-profit Grid (Close Orders)

```text
tp_prices(pside, i in [0, n)):
    step   = (close_grid_markup_end - close_grid_markup_start) / (n - 1)
    markup = close_grid_markup_start + i * step

    long  : pos.price * (1 + markup)
    short : pos.price * (1 - markup)

tp_qty(pside, i) = full_pos_size * close_grid_qty_pct
```

`n ≈ 1 / close_grid_qty_pct`.  Quantities are trimmed so the sum equals the current position
size, and any leftover exposure is assigned to the TP closest to `markup_start`.

## Trailing Closes

Trailing closes mirror the trailing-entry logic with the parameters
`close_trailing_*`.  For longs:

```text
threshold_close =
    close_trailing_threshold_pct * (1 + close_trailing_threshold_we_weight * wel_ratio)

retracement_close =
    close_trailing_retracement_pct *
    (1 + close_trailing_retracement_we_weight * wel_ratio)

triggered_when:
    high_since_entry  > pos.price * (1 + threshold_close)
    and
    low_since_high    < high_since_entry * (1 - retracement_close)

close_price = max(best_bid, high_since_entry * (1 - retracement_close))
close_qty   = full_pos_size * close_trailing_qty_pct
```

Short trailing closes invert the inequalities and use the symmetric formulas.

## Auto-Unstucking

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
`wallet_exposure / wel_base > unstuck_threshold`.

## WEL Enforcer (Auto Reduce)

The per-position wallet exposure limit (WEL) enforcer trims individual symbols
whenever their exposure rises above the allowance-adjusted cap:

```text
allowed_i    = wel_base_i * (1 + risk_we_excess_allowance_pct)
target_i     = allowed_i * risk_wel_enforcer_threshold
```

If `exposure_i > target_i`, the bot submits a reduce-only order sized just large
enough (with step rounding and minimum-qty guards) to bring the position back to
`target_i`. These orders are emitted as `CloseAutoReduceWel{Long,Short}` and are
returned directly from `calc_next_close_*`/`calc_closes_*`.

Setting `risk_wel_enforcer_threshold` below `1.0` forces a gentle, continuous
trimming behaviour; values above `1.0` create an additional grace margin.

## TWEL Enforcer (Auto Reduce)

The “Total Wallet Exposure Limit” enforcer keeps the sum of exposures below
`total_wallet_exposure_limit * risk_twel_enforcer_threshold`.  For each position:

```text
exposure_i      = wallet_exposure(...)
allowed_i       = wel_base_i * (1 + risk_we_excess_allowance_pct)
base_psize_i    = allowed_i * balance / (price_i * c_mult_i)
max_reducible_i = max(0, |pos.size_i| - base_psize_i)
```

While `Σ exposure_i` exceeds the threshold:

1. Pick the position with the smallest relative price move from entry.
2. Reduce it by `min(max_reducible_i,
                   (Σ exposure - threshold) * balance / (price_i * c_mult_i))`.

The final reduce-only order is:

```text
qty  = sign(pside) * min(reduced_psize, |pos.size|)
price ≈ market_price
order_type = CloseAutoReduceTwel{Long,Short}
```

By construction the quantity never exceeds the live position size.
Positions already earmarked for `CloseAutoReduceWel*` during the same scheduling cycle are skipped so that reductions do not double-up; they can be considered again on subsequent iterations once the WEL order has been filled.

## Parameter Interactions at a Glance

| Parameter                                      | Primary effect                                             | Key equations |
| ---------------------------------------------- | -----------------------------------------------------------| ------------- |
| `ema_span_*`                                   | Defines EMA bands used by initial pricing & unstuck levels | `EMA_low`, `EMA_high` |
| `entry_grid_spacing_*`, `entry_grid_double_down_factor` | Controls grid spacing and growth of re-entry quantities | `next_grid_price`, `next_grid_qty` |
| `entry_trailing_*`                             | Adjust trailing entry triggers via exposure & volatility  | `threshold`, `retracement` |
| `close_grid_markup_*`, `close_grid_qty_pct`    | Shapes TP ladder                                          | `tp_prices`, `tp_qty` |
| `close_trailing_*`                             | Mirrors trailing entries but for exits                    | `threshold_close`, `retracement_close` |
| `unstuck_*`                                    | Loss realization rules                                    | `unstuck_allowed`, `close_price` |
| `risk_wel_enforcer_threshold`, `risk_we_excess_allowance_pct` | Per-symbol exposure cap                                 | `target_i`, `qty` |
| `risk_twel_enforcer_threshold`, `risk_we_excess_allowance_pct` | Portfolio-wide exposure cap                              | `max_reducible_i`, `qty` |

For worked examples on a per-parameter basis, see the comments sprinkled in
`passivbot-rust/src/entries.rs` and the optimiser notebooks under `notebooks/`.
