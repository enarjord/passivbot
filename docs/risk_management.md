# Risk Management

## Leverage

On perpetual futures markets with leverage, Passivbot may expose more than 100% of the wallet's funds.  

Passivbot uses only (unleveraged) wallet balance in its calculations, so adjusting leverage on exchange will make no difference on risk, profit or bot behavior, as long as leverage is set high enough for the bot to make its orders according to the configuration. Practically, this means that leverage should be set at least higher than total_wallet_exposure_limit_long + total_wallet_exposure_limit_short for the bot to have enough margin to make its orders. Otherwise, the bot might encounter "insufficient margin" errors.

## Wallet Exposure

To measure a position's risk, Passivbot finds the ratio of position size (in quote) to total unleveraged balance.  
The formula for wallet exposure is

`wallet_exposure = (position_size * position_price) / unleveraged_wallet_balance`  

wallet_exposure==0.0 means no position
wallet_exposure==1.0 means 100% of unleveraged wallet balance is in position.
wallet_exposure==4.0 means 400% of unleveraged wallet balance is in position.

E.g. if wallet balance is $1000, long position size is 100.0 and position price is 35.0,  
then wallet_exposure is `100 * 35 / 1000 == 3.5`

## Wallet Exposure Limit

The wallet_exposure_limit is the limit beyond than which the bot will not allow a position's wallet_exposure to grow.

For example, if wallet_exposure_limit=0.6, the bot will not make any more entries when a position's wallet_exposure >= 0.6.

## Total Wallet Exposure Limit

User sets total_wallet_exposure_limit separately for long and short. A single position's wallet_exposure_limit is `total_wallet_exposure_limit / n_positions`.

## Exposure Allowance and Enforcers

Passivbot now exposes three related parameters that control how aggressively the bot will cap exposure, both per-position and portfolio-wide:

- `risk_we_excess_allowance_pct`
- `risk_wel_enforcer_threshold`
- `risk_twel_enforcer_threshold`

These parameters act on top of the base wallet exposure limits described above. `risk_we_excess_allowance_pct` and `risk_wel_enforcer_threshold` can be configured globally (under `bot.long` / `bot.short`) and overridden inside coin path overrides (`coin_overrides.<COIN>.bot.long/short`). `risk_twel_enforcer_threshold` applies to the whole side (all longs or all shorts) and therefore is **global-only**; coin overrides inherit the global value.

### risk_we_excess_allowance_pct — “grace room” for exposure

This value (expressed as a decimal, e.g. `0.05` = +5 %) multiplies the base wallet exposure limit before any gating or enforcer logic runs:

```
effective_limit = wallet_exposure_limit * (1 + max(0, risk_we_excess_allowance_pct))
```

The same multiplier is used for:

- entry sizing and cropping
- unstuck / partial fillers
- the per-position WEL enforcer

Example: with `wallet_exposure_limit = 0.20` and `risk_we_excess_allowance_pct = 0.10`, a position may temporarily grow to `0.22` wallet exposure before it is considered over the limit. Setting the allowance to `0.0` keeps the strict behaviour from earlier versions; increasing it lets the bot breathe a little above the nominal cap before the safety circuits engage.

### risk_wel_enforcer_threshold — per-position trimming

The WEL (Wallet Exposure Limit) enforcer monitors each open position. When a position’s actual exposure exceeds

```
effective_limit * risk_wel_enforcer_threshold
```

the bot emits a reduce-only order (tagged `close_auto_reduce_wel_*`) to bring the exposure back below the target. Values < 1.0 make the enforcer trim proactively, even before the allowance is fully consumed. A value of `1.0` enforces a hard cap precisely at the allowed limit, while `<= 0` disables WEL auto-reduction entirely.

In practical terms:

- `risk_wel_enforcer_threshold = 0.9` → keep each position around 90 % of the allowance, continuously shaving excess.
- `risk_wel_enforcer_threshold = 1.0` → allow positions to climb to the allowance but never above.
- `risk_wel_enforcer_threshold <= 0` → no automatic trimming; entries/unstuck logic alone must keep exposure in check.

### risk_twel_enforcer_threshold — portfolio trimmer

The TWEL (Total Wallet Exposure Limit) enforcer works across all positions of a side (long or short). When the **sum** of wallet exposures crosses

```
total_wallet_exposure_limit * risk_twel_enforcer_threshold
```

the bot queues reduction orders for the least underwater positions until the aggregate exposure falls back below the threshold. Much like the per-position enforcer:

- `risk_twel_enforcer_threshold < 1.0` keeps the fleet of positions running under the configured total cap (e.g. `0.95` trims whenever the portfolio reaches 95 % of the total limit).
- `risk_twel_enforcer_threshold = 1.0` enforces the limit strictly—equivalent to the old `enforce_exposure_limit=true`.
- `risk_twel_enforcer_threshold > 1.0` grants a little overflow before the enforcer fires.
- `risk_twel_enforcer_threshold <= 0` disables the TWEL enforcer entirely.

Because the TWEL enforcer uses each position’s internal limit (which already includes `risk_we_excess_allowance_pct`), both knobs interact: increasing the allowance raises the breathing room for individual symbols as well as the total pool. Pairing a higher allowance with a lower `risk_wel_enforcer_threshold`/`risk_twel_enforcer_threshold` yields “soft edges” (continuous trimming), while keeping the thresholds at `1.0` produces “hard edges” (the bot only trims on overshoot).

## Bankruptcy and liquidation

Bankruptcy is defined as when `equity == (balance + unrealized_pnl) == 0.0`, that is, when total debt is equal to total assets.

Liquidation happens when the exchange force closes a position to avoid it going into negative equity.  
This usually happens before actual bankruptcy is reached, in order for exchange to cover slippage costs.

Bankruptcy price may be calculated from position and balance.

E.g.  
If wallet_exposure==1.0, bankruptcy price is zero.  
If wallet_exposure==2.0, bankruptcy price is 50% lower than pos price.  
If wallet_exposure==3.0, bankruptcy price is 33.33% lower than pos price.  
If wallet_exposure==10.0, bankruptcy price is 10% lower than pos price.  


## Getting stuck

When a bot has no more entries left and wallet_exposure_limit is reached or exceeded, it is termed "getting stuck".  

If a long position is stuck and the price keeps falling, the distance between position price and market price grows larger, and closing the position in profit becomes less likely. Therefore it is desirable to bring position price closer to price action such that the position may be closed in profit on a small bounce.  
To achieve this, the position must be increased. However, the larger the position size, the higher the risk of liquidation, should the price keep moving the wrong way.

## Diversification

While correlation is observed in most markets in general and in crypto markets in particular (e.g. if the price of bitcoin crashes, most other cryptos tend to follow closely), it is also observed that the "dead cat" often bounces at slightly different times and at different heights. Therefore, diversifying to multiple coins helps reduce the risk of a single bad coin destroying the whole account.

A thousand coin flips will converge on 500 heads and 500 tails. One single coin flip will be either heads or tails. So it may be more desirable to end up with 3 out of 10 bots stuck, each with wallet_exposure==0.1, than with 1 single bot stuck with wallet_exposure==1.0.
