# Minimum Effective Cost and Initial Entry Sizing

This note explains how Passivbot checks exchange minimums, computes initial entries, and how to
reason about the minimum account balance required for a given configuration.

## What is “effective min cost”?

Exchanges enforce minimum order notional per symbol. Passivbot maintains an **effective min cost** per
symbol:
- `effective_min_cost = max(exchange_min_cost, min_qty * last_price)`
- It is refreshed with `update_effective_min_cost` (after markets init and periodically).

When deciding if a symbol is tradable, the bot compares the *planned initial entry* against this
effective min cost. If it cannot meet the minimum, the symbol is excluded (leading to
“No long/short symbols are approved due to min effective cost too high…”).

## How is the initial entry computed?

For most perpetuals, `initial_notional ≈ balance * (twel / n_positions) * (1 + we_allowance) * entry_initial_qty_pct`.

With:
- `balance`: current account balance in quote currency (e.g., USDT/USDC).
- `twel = total_wallet_exposure_limit`: side-level cap; divided by `n_positions` to get per-position WE.
- `we_allowance = risk_we_excess_allowance_pct`: optional headroom multiplier on per-position WEL.
- `entry_initial_qty_pct`: fraction of the per-position exposure used for the first order.


## Pass/fail check

The bot requires:
```
initial_notional >= effective_min_cost
```
If this fails, the symbol is dropped, unless `filter_by_min_effective_cost` is set to false, in which case the bot will place the initial entry with qty modified to meet exchange's requirements. Note that this can be risky, since initial entry will then be greater than what the bot wants, which might lead to significant discrepancies between backtested and live behavior.

## Estimating required balance

Rearrange the inequality to solve for the minimum balance:

`balance_required ≈ effective_min_cost / ((twel / n_positions) * (1 + we_allowance) * entry_initial_qty_pct)`  
or  
`balance_required ≈ effective_min_cost / (effective_wel * entry_initial_qty_pct)`

Use the highest effective_min_cost among the symbols you want to trade. Example:
- `effective_min_cost = 10 USDT`
- `total_wallet_exposure_limit = 0.6`, `n_positions = 3` → `wallet_exposure_limit = 0.2`
- `risk_we_excess_allowance_pct = 0.5` (effective_wel = 0.2 * (1 + 0.5) = 0.3)
- `entry_initial_qty_pct = 0.015`
→ `balance_required >= 10 / (0.3 * 0.015) ≈ 2,222.22 USDT`

If you run multiple positions concurrently, ensure your balance comfortably exceeds this threshold so
other risk checks (total exposure, n_positions) aren’t immediately binding.

## Practical tips

- To lower the required balance (while preserving filtering):
  - Increase `entry_initial_qty_pct` (if risk allows).
  - Increase `wallet_exposure_limit` (raises per-position sizing; increases risk).
  - Reduce `n_positions` so fewer symbols compete for balance.
- Avoid disabling `filter_by_min_effective_cost` unless you accept higher initial qty.
- If using very small balances, prioritize symbols with low min cost and lower price levels.

## Where to see/debug

- Logs show the warning when all candidate symbols fail the min-cost check for a side.
- `update_effective_min_cost` runs after market init; you can inspect `effective_min_cost` per symbol
  in the cached state or by instrumenting logs if needed.
