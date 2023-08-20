# Auto Unstuck

Passivbot may be configured to realize losses. Applies to recursive and neat grid modes.

## Parameters

- `auto_unstuck_wallet_exposure_threshold`  
Ratio of exposure to exposure_limit at which auto unstuck (AU) kicks in.  
`if wallet_exposure / wallet_exposure_limit > (1 - auto_unstuck_wallet_exposure_threshold): enable AU`  
E.g.  
`auto_unstuck_wallet_exposure_threshold == 0.0`: auto unstuck is disabled.  
`auto_unstuck_wallet_exposure_threshold == 0.1`: auto unstuck kicks in when exposure is 10% away from exposure_limit.  
`auto_unstuck_wallet_exposure_threshold == 0.9`: auto unstuck kicks in when exposure is 90% away from exposure_limit.  
`auto_unstuck_wallet_exposure_threshold == 1.0`: auto unstuck is always enabled.  

- `auto_unstuck_qty_pct`  
How much of max pos size to close.  
`close_cost = balance * wallet_exposure_limit * auto_unstuck_qty_pct`  
For example, if balance is $1000, `wallet_exposure_limit=0.3` and `auto_unstuck_qty_pct=0.02`:  
`close_cost == $1000 * 0.3 * 0.02 == $6`.

- `ema_span_0`, `ema_span_1`  
Bot uses three emas of spans: `[span0, (span0 * span1)**0.5, span1]`, given in minutes.  

- `auto_unstuck_ema_dist`  
Close price distance from EMA band.  
Lower auto unstuck EMA band is `min(ema0, ema1, ema2) * (1 - auto_unstuck_ema_dist)`.  
Upper auto unstuck EMA band is `max(ema0, ema1, ema2) * (1 + auto_unstuck_ema_dist)`.  

- `auto_unstuck_delay_minutes`  
Timer for unstuck closes, given in minutes.  
`if now - prev_AU_close_ts > auto_unstuck_delay: enable AU`  

## Examples
```
auto_unstuck_wallet_exposure_threshold = 0.0
auto_unstuck_qty_pct = 0.0
auto_unstuck_ema_dist = 0.0
auto_unstuck_delay_minutes = 0.0
```
Auto unstuck is disabled.

```
auto_unstuck_wallet_exposure_threshold = 0.5
auto_unstuck_qty_pct = 0.01
auto_unstuck_ema_dist = 0.0
auto_unstuck_delay_minutes = 120.0
```
If bot is 50% or more exposed, close 1% of max pos size every two hours at EMA band.

```
auto_unstuck_wallet_exposure_threshold = 1.0
auto_unstuck_qty_pct = 0.1
auto_unstuck_ema_dist = -1.0
auto_unstuck_delay_minutes = 20.0
```
If there is any position, close 10% of max pos size every 20 minutes at current market price.

```
auto_unstuck_wallet_exposure_threshold = 0.2
auto_unstuck_qty_pct = 0.03
auto_unstuck_ema_dist = 0.003
auto_unstuck_delay_minutes = 720.0
```
If bot is 20% or more exposed, close 3% of max pos size every six hours 0.3% away from EMA band.

