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
* `risk_wel_enforcer_threshold` (The Position Trimmer)
* `risk_we_excess_allowance_pct` (The Buffer)
* `risk_twel_enforcer_threshold` (The Portfolio Trimmer)

#### WEL Enforcer (`risk_wel_enforcer_threshold`)
This controls **per-position** trimming. If a single position's exposure exceeds:
`effective_limit * risk_wel_enforcer_threshold`
The bot issues a reduce-only order to trim it back down.

Examples:
* `0.9`: **Proactive trimming.** Keeps the position at 90% of its allowance.
* `1.0`: **Strict limit.** Trims immediately if the position exceeds the allowance.
* `1.05`: **Buffer zone.** Trims only if the position becomes greater than 5% of its allowance.
* `<= 0`: **Disabled.** No automatic trimming; relies solely on entry logic/auto-unstuck.

#### Excess Allowance (`risk_we_excess_allowance_pct`)
In practice, the bot rarely fills all positions simultaneously. Therefore, the bot can be configured to allow exceeding individual WELs by setting `risk_we_excess_allowance_pct > 0.0` (e.g., 20% excess allowance). This can be thought of as the bot "borrowing" capacity from unfilled positions. The per-position WEL enforcer respects this expanded limit and only trims when the *effective* WEL is breached.

`effective_limit = wallet_exposure_limit * (1 + max(0, risk_we_excess_allowance_pct))`

* **Example:** If WEL is `0.20` and allowance is `0.10` (10%), the position can grow to `0.22` before the bot considers it "full."
* **Motivation:** In a multi-coin setup, this lets the bot boost performance on active positions by utilizing the unused capacity of inactive positions.

> **Comprehensive Calculation Example:**
> Given a **$2000 balance**, `TWEL=1.0`, `excess_allowance=0.5`, `n_positions=4`, `unstuck_threshold=0.48`:
>
> 1.  **Per position allowance:**
>     `((balance * twel) / n_positions) * (1 + excess_allowance)`
>     `(($2000 * 1.0) / 4) * (1 + 0.5) == $500 * 1.5 == $750`
>
> 2.  **Per position effective exposure limit:**
>     `(twel / n_position) * (1 + excess_allowance)`
>     `(1.0 / 4) * (1 + 0.5) == 0.375`
>
> **The Result:**
> The bot will stop making entries when a position's exposure hits **0.375** or when the overall account's exposure hits **1.0**.
>
> Since `0.375 * 4 > 1.0`, the bot will allow filling up the first positions' effective limits, but will gate new entries when filling those entries would lead to `twe > twel`.
>
> **Auto-Unstuck Trigger:**
> Auto unstuck will begin at `effective_we_limit * unstuck_threshold`:
> `0.375 * 0.48 == 0.18` (or 48% of the full position).



* **Edge Case:**
    * *Scenario:* `TWEL = 1.0`, `n_positions = 10`, `excess_allowance = 0.5`.
    * *Effective per-position limit:* `(1.0 / 10) * (1 + 0.5) = 0.15`.
    * If there is a market crash and 6 positions fill to their effective limit, the total exposure is `6 * 0.15 = 0.9`.
    * If a 7th position also fills to 0.15, total exposure would become `1.05`.
    * Since `1.05 > 1.0` (the TWEL), the bot will gate any new orders for that 7th position that would cause the total wallet exposure to breach the TWEL, effectively blocking the "excess" allowance for the last few positions.

#### TWEL Enforcer (`risk_twel_enforcer_threshold`)
This controls **total portfolio** trimming. It monitors the sum of all long (or short) exposures. If the total exceeds:
`total_wallet_exposure_limit * risk_twel_enforcer_threshold`
The bot reduces positions, starting with the **least underwater** ones first (following Auto-Unstuck logic).

* `0.95`: Trims the portfolio when it reaches 95% of the total limit.
* `1.0`: Strictly enforces the total limit.
* `> 1.0`: Allows some overflow (e.g., during extreme volatility) before forced reduction occurs.

This portfolio trimming mechanism pairs well with the excess allowance. When TWE hits TWEL and new orders are blocked, the TWEL enforcer will start chipping away at the least underwater positions to free up margin, even if those positions are technically within their per-position effective exposure limit.

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