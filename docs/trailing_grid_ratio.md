# Trailing Grid Ratio for Entries and Closes

`*_trailing_grid_ratio` (entry and close) controls how much of the allowed wallet exposure is placed
with trailing orders vs grid orders. It operates on **wallet exposure ratio**:
```
we_ratio = wallet_exposure / wallet_exposure_limit_with_allowance
         = (pos_size * pos_price / balance) / (twel/n_positions * (1 + risk_we_excess_allowance_pct))
```
This tracks “percent of position filled” (e.g., 50% filled → ratio 0.5; 20% → 0.2; 100% → 1.0).
The sign determines whether trailing or grid orders are used first.

Key points
- Let `wel = total_wallet_exposure_limit / n_positions`.
- Let `effective_wel = wel * (1 + risk_we_excess_allowance_pct)`.
- Let `we_ratio = wallet_exposure / effective_wel`.
- Ratio ≥ 1 or ≤ -1 → trailing only.
- Ratio = 0 → grid only.
- Ratio > 0 (trailing first): trailing until `we_ratio > effective_wel * trailing_grid_ratio`; afterwards grid orders until position is full.
- Ratio < 0 (grid first): grid until `we_ratio > effective_wel * (1 + trailing_grid_ratio)`; afterwards trailing orders.

Rule of thumb
- Closer to 0 → more grid.
- Closer to ±1 → more trailing.
- Positive values start with trailing then switch to grid; negative values start with grid then switch to trailing.

Examples (applies to entry and close; simplified, ignoring min_qty/min_cost rounding)

- `*_trailing_grid_ratio = 0.1` (trailing first)
  - If `we_ratio < 0.1` → trailing order.
  - Else → grid order, leaving 10% of allowed exposure for trailing.

- `*_trailing_grid_ratio = -0.1` (grid first)
  - If `we_ratio < 0.9` → grid order.
  - Else → trailing order, leaving 90% of allowed exposure for grid.

- `*_trailing_grid_ratio = 0.9` (trailing first)
  - If `we_ratio < 0.9` → trailing order.
  - Else → grid order, leaving 90% of allowed exposure for trailing.

- `*_trailing_grid_ratio = -0.9` (grid first)
  - If `we_ratio < 0.1` → grid order.
  - Else → trailing order, leaving 10% of allowed exposure for grid.
