# Risk Management in Passivbot

Passivbot is designed to manage medium- to high-frequency trading positions on perpetual futures markets. Because these markets involve leverage, rigorous risk management is required to prevent liquidation and manage "stuck" positions (bags).

This guide explains the bot's risk philosophy, from basic definitions of leverage and exposure to advanced safety mechanisms like Auto-Unstuck and Enforcers.

---

## 1. Core Concepts: Leverage & Exposure

### Leverage
On perpetual futures markets, exchanges allow you to open positions larger than your actual account balance. However, Passivbot ignores exchange leverage settings in its internal calculations. It always uses the **unleveraged wallet balance** as the reference for risk.

* **Bot Behavior:** Adjusting the leverage slider on the exchange website does **not** change the bot’s risk calculations, profit targets, or entry logic.
* **Practical Requirement:** You must set your exchange leverage high enough to support the positions the bot attempts to open. A safe rule of thumb is:

    `Exchange Leverage > (total_wallet_exposure_limit_long + total_wallet_exposure_limit_short)`

    If exchange leverage is too low, the bot will hit "insufficient margin" errors when trying to place orders.

### Wallet Exposure (WE)
To measure risk, the bot calculates the ratio of the position size to the total available balance. The formula is:

`wallet_exposure = (position_size * average_entry_price) / unleveraged_wallet_balance`

* `0.0`: No position.
* `1.0`: Position value equals 100% of the wallet balance.
* `4.0`: Position value is 4x the wallet balance (requires 4x leverage).

> **Example:** If your wallet balance is $1,000, and you hold a long position of 100 contracts at a price of $35:
> `WE = (100 * 35) / 1000 = 3.5`

### Wallet Exposure Limit (WEL)
The `wallet_exposure_limit` acts as a ceiling. If a position's exposure reaches this limit, the bot stops adding new entries to that position to prevent risk from spiraling out of control.

In a multi-coin setup, the limits are derived from the total limit:
* **Total Limit:** User sets `total_wallet_exposure_limit`.
* **Per-Position Limit:** `total_wallet_exposure_limit / n_positions`.

---

## 2. The Problem: "Getting Stuck"

In a long-only example without loss-taking, the bot buys as the price falls to lower the average entry price. However, re-entries are eventually capped by the **Wallet Exposure Limit**.

If the price continues to drop after the limit is reached, the bot becomes "stuck."
* The position has a high average entry price (e.g., $100).
* The current price action is lower (e.g., around $70).
* The bot cannot buy more to average down.
* Funding fees slowly drain the balance.

The only natural escapes are price reversion (waiting for the market to bounce back) or adding more to the stuck position as the wallet balance increases via profits from other coins.

---

## 3. The Solutions: Auto-Unstuck and Exposure Enforcers

To prevent positions from stagnating indefinitely, Passivbot employs an **Auto-Unstuck** mechanism and **Exposure Limit Enforcers**.

### A. Auto-Unstuck
Instead of a traditional stop-loss that closes the entire position at a massive loss, Auto-Unstuck chips away at the position over time. It performs controlled, partial closes at a loss to free up margin, then re-enters at lower prices.

1.  **Partial Close:** Closes a small portion of the bag to reduce exposure.
2.  **Re-entry:** Uses the freed-up exposure to buy back in at normal re-entry price (grid or trailing).
3.  **Result:** The average entry price moves closer to the market price, allowing the bot to eventually close the full position in profit on a smaller bounce.

#### Why Multi-Coin helps Auto-Unstuck
Auto-unstuck works best when running multiple coins:
* **Prioritization:** It prioritizes unstucking the **least underwater** coin first, quickly returning it to profitability.
* **Profit Offsetting:** While one coin realizes losses to unstuck, the other active coins continue generating profit, smoothing out the equity curve.
* **Capped Drawdown:** Total unstuck losses are capped at a configurable percentage below the historical peak balance (e.g., 1%). After the loss allowance is consumed, further losses can only be taken after other positions have made profits.

#### Weakness of Auto-Unstuck
Extreme black-swan events (exchange failure, stablecoin depeg, delisting, and other causes of prolonged unilateral price movement) may cause the auto-unstuck mechanism to keep taking losses and re-entering continually on an adversely moving coin. The only realistic solution to these edge cases is **human intervention** as the final backstop.

### B. Exposure Enforcers
While exposure limits prevent *new* orders, it is still possible for existing positions to swell beyond their limits (e.g., the account holds BTC as collateral and the BTC/USD price drops, the user withdraws funds while positions are maxed out, or the account realizes significant losses). To handle this, Passivbot provides three parameters that control how aggressively to cap and trim exposure.

These parameters function on top of the base limits:
* `position_exposure_enforcer_threshold` (The Position Trimmer)
* `risk_we_excess_allowance_pct` (The Buffer)
* `total_exposure_enforcer_threshold` (The Portfolio Trimmer)

#### Position Exposure Enforcer (`position_exposure_enforcer_threshold`)
This controls **per-position** trimming. If a single position's exposure exceeds:
`effective_limit * position_exposure_enforcer_threshold`
The bot issues a reduce-only order to trim it back down.

Examples:
* `0.9`: **Proactive trimming.** Keeps the position at 90% of its allowance.
* `1.0`: **Strict limit.** Trims immediately if the position exceeds the allowance.
* `1.05`: **Buffer zone.** Trims only if the position becomes greater than 5% of its allowance.

Set `position_exposure_enforcer_enabled = false` to disable this control. The
threshold must be finite and greater than zero when enabled.

This enforcer can also be used deliberately as an aggressive strategy mechanism:
with aggressive entries and a threshold such as `0.95`, the bot may refill toward
the position limit and repeatedly trim back to 95% without waiting for auto
unstuck's EMA gate or loss allowance.

#### Excess Allowance (`risk_we_excess_allowance_pct`)
In practice, the bot rarely fills all positions simultaneously. Therefore, the bot can be configured to allow exceeding individual WELs by setting `risk_we_excess_allowance_pct > 0.0` (e.g., 20% excess allowance). This can be thought of as the bot "borrowing" capacity from unfilled positions. The per-position WEL enforcer respects this expanded limit and only trims when the *effective* WEL is breached.

The raw excess is capped before use so a single position cannot receive more headroom than the side's total configured exposure:

`effective_we_excess_allowance_pct = min(max(0, risk_we_excess_allowance_pct), max(0, total_wallet_exposure_limit / wallet_exposure_limit - 1))`

`effective_limit = wallet_exposure_limit * (1 + effective_we_excess_allowance_pct)`

* **Example:** If WEL is `0.20` and allowance is `0.10` (10%), the position can grow to `0.22` before the bot considers it "full."
* **Motivation:** In a multi-coin setup, this lets the bot boost performance on active positions by utilizing the unused capacity of inactive positions.

> **Comprehensive Calculation Example:**
> Given a **$2000 balance**, `TWEL=1.0`, `excess_allowance=0.5`, `n_positions=4`, `unstuck_threshold=0.48`:
>
> 1.  **Base WEL:**
>     `base_wel = twel / n_positions`
>     `1.0 / 4 == 0.25`
>
> 2.  **Effective excess allowance:**
>     `min(max(0, excess_allowance), max(0, twel / base_wel - 1))`
>     `min(0.5, 1.0 / 0.25 - 1) == min(0.5, 3.0) == 0.5`
>
> 3.  **Per position effective exposure limit:**
>     `base_wel * (1 + effective_we_excess_allowance_pct)`
>     `0.25 * 1.5 == 0.375`
>
> 4.  **Per position allowance:**
>     `balance * effective_wel`
>     `$2000 * 0.375 == $750`
>
> **The Result:**
> The bot will stop making entries when a position's exposure hits **0.375** or when the overall account's exposure hits **1.0**.
>
> Since `0.375 * 4 > 1.0`, the bot will allow filling up the first positions' effective limits, but will gate new entries when filling those entries would lead to `twe > twel`.
>
> With `n_positions=1`, the effective excess is capped at `0.0`: `min(1 - 1, excess_allowance)`. The single position's per-position allowance therefore remains `TWEL`; raw excess does not increase it.
>
> **Auto-Unstuck Trigger:**
> Auto unstuck will begin at `effective_we_limit * unstuck_threshold`:
> `0.375 * 0.48 == 0.18` (or 48% of the full position).



* **Edge Case:**
    * *Scenario:* `TWEL = 1.0`, `n_positions = 10`, `excess_allowance = 0.5`.
    * *Base WEL:* `1.0 / 10 = 0.10`.
    * *Effective excess allowance:* `min(0.5, 1.0 / 0.10 - 1) = 0.5`.
    * *Effective per-position limit:* `0.10 * 1.5 = 0.15`.
    * If there is a market crash and 6 positions fill to their effective limit, the total exposure is `6 * 0.15 = 0.9`.
    * If a 7th position also fills to 0.15, total exposure would become `1.05`.
    * Since `1.05 > 1.0` (the TWEL), the bot will gate any new orders for that 7th position that would cause the total wallet exposure to breach the TWEL, effectively blocking the "excess" allowance for the last few positions.

#### Total Exposure Enforcer (`total_exposure_enforcer_threshold`)
This controls **total portfolio** trimming. It monitors the sum of all long (or short) exposures. If the total exceeds:
`total_wallet_exposure_limit * total_exposure_enforcer_threshold`
The bot reduces positions, starting with the **least underwater** ones first (following Auto-Unstuck logic).

* `0.95`: Trims the portfolio when it reaches 95% of the total limit.
* `1.0`: Strictly enforces the total limit.
* `> 1.0`: Allows some overflow (e.g., during extreme volatility) before forced reduction occurs.

Set `total_exposure_enforcer_enabled = false` to disable this control. The
threshold must be finite and greater than zero when enabled.

This portfolio trimming mechanism pairs well with the excess allowance. When TWE
hits TWEL and new orders are blocked, the total exposure enforcer first chips
away at least-underwater positions above their per-position floor. If total
bot-scope exposure remains above target, it has greater authority and continues
reducing least-underwater bot-managed positions even below that floor.

Manual-mode positions are outside bot scope: they do not receive bot-created
orders, bot cancellations, auto unstuck, exposure-enforcer reductions, active
slot accounting, or bot-scope TWE/TWEL accounting.

### C. Realized-Loss Gate (`live.max_realized_loss_pct`)
This is a global guardrail on **loss-realizing close orders**. It applies to all close order types, including WEL/TWEL auto-reduce and unstuck closes. Only panic closes are exempt.

The gate is anchored to the net realized PnL peak reconstructed from fill history
(`pnl + fee_paid`):

`balance_peak = current_balance + (realized_pnl_cumsum_max - realized_pnl_cumsum_last)`

`balance_floor = balance_peak * (1 - max_realized_loss_pct)`

For each candidate close order, the bot projects realized PnL if filled and blocks the order when:

`projected_balance_after_fill < balance_floor`

Behavior by value:
* `max_realized_loss_pct <= 0.0`: block all lossy closes.
* `0.0 < max_realized_loss_pct < 1.0`: allow losses only down to the configured fraction below peak balance.
* `max_realized_loss_pct >= 1.0`: disabled.

Operational notes:
* The shipped default is `1.0`, which keeps the gate disabled until you opt in explicitly.
* Live bot logs visible warnings whenever an order is blocked by this gate.
* This can intentionally block automatic reducers if they would realize too much loss.
* If you need immediate forced reduction regardless of realized loss, use panic mode.

### D. Equity Hard Stop Loss (`bot.{long,short}.hsl.*`)
This is a side-specific circuit breaker based on reconstructed strategy drawdown, not just raw exchange equity.

It exists for cases where:

1. auto-unstuck is too slow
2. the realized-loss gate is still allowing the bot to operate in a clearly degraded state
3. you want a final supervisory backstop that can close all positions on one `pside` and halt that `pside`

Behavior:

1. `yellow`: warning tier
2. `orange`: reduced-risk mode (`graceful_stop` or `tp_only_with_active_entry_cancellation`) for that `pside`
3. `red`: force panic exits, wait until all positions on that `pside` are fully closed, and halt that `pside`

Operational notes:

1. HSL is configured separately under `bot.long.hsl.*` and `bot.short.hsl.*`.
2. `live.hsl_signal_mode` defaults to the shared `unified` account-level signal, with `pside` available for side-local strategy signals and `coin` available for per-coin slot drawdown stops.
3. RED can auto-restart after `hsl_cooldown_minutes_after_red`. Terminal no-restart uses persistent cross-restart HSL drawdown.
4. In backtests, simulated market panic closes use `backtest.market_order_slippage_pct`; live market panic closes use the exchange adapter's order semantics and live exchange/CCXT slippage controls.
5. Backtests export canonical strategy-equity metrics under `*_strategy_eq`, including side-specific `*_strategy_eq_long` / `*_strategy_eq_short` metrics. Deprecated `*_hsl` metric names remain accepted as aliases for older configs/results.

See the dedicated guide:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)

---

## 4. Bankruptcy & Liquidation Technicals

Understanding the math behind liquidation helps in configuring the limits above.

* **Bankruptcy:** Occurs when `equity (balance + unrealized_pnl) == 0`. Total debt equals total assets.
* **Liquidation:** The exchange force-closes the position *before* bankruptcy to cover slippage costs.

The relationship between Wallet Exposure (WE) and the price drop required for bankruptcy is predictable:

| Wallet Exposure | Bankruptcy Price Drop |
| :--- | :--- |
| **1.0** | Price must drop to 0 (Impossible in most contexts) |
| **2.0** | 50% drop from average entry |
| **3.0** | 33.33% drop from average entry |
| **10.0** | 10% drop from average entry |

*Note: In cross-margin mode, the "Bankruptcy Price" is a function of the entire account's health, not just a single position.*

---

## 5. Diversification Strategy

Diversification is a key component of risk management. While crypto markets are highly correlated (if Bitcoin crashes, most alts follow), they do not move in perfect unison.

* **The "Coin Flip" Logic:** A thousand coin flips converge on 50/50. A single flip is binary risk.
* **Application:** It is generally safer to have 3 out of 10 bots slightly stuck (each with WE 0.1) than 1 single bot heavily stuck (WE 1.0).
* **Behavior:** The "Dead Cat Bounce" often happens at different times for different coins. By running multiple coins, the bot can use the profits from recovering coins to subsidize the unstucking of the lagging ones.
