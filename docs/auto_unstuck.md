# Auto Unstuck

Passivbot may be configured to accept losses in order to keep position price close to price  
action without exceeding wallet exposure limit.

## Overview

Ideally, the auto unstuck would work something like this example:  
- long pos opened @ 100
- price drops, entire grid is exhausted after fills @ 97, 94, 90, 95, 89, 82, 73
- long pos price is now 85 and mark price is 72
- partial close at a loss @ 76
- reentry @ 74, pos price is now 83
- partial close at a loss @ 75
- reentry @ 73, pos price is now 80.5
- partial close at a loss @ 71.5
- reentry @ 69.5, pos price is now 77.5
- partial close at a loss @ 70.5
- reentry @ 69.5, pos price is now 75.5
- partial close at a loss @ 70.5
- normal close pos @ 76  

To achieve this partial loss, reentry, partial loss, reentry behavior,  
a band below and a band above price action are made with EMAs of different spans.  

## Parameters

- `auto_unstuck_wallet_exposure_threshold`  
Per uno of exposure limit which bot will close at a potential loss.  
If wallet exposure exceeds `wallet_exposure * (1 - auto_unstuck_wallet_exposure_threshold)`, auto unstuck mode is activated.  
If set to zero, auto unstuck is disabled.  

- `ema_span_0`, `ema_span_1`  
Bot uses three emas: `[min_span, (min_span * max_span)**0.5, max_span]`, given in minutes.  

- `auto_unstuck_ema_dist`  
Lower auto unstuck EMA band is `min(ema0, ema1, ema2) * (1 - auto_unstuck_ema_dist)`.  
Upper auto unstuck EMA band is `max(ema0, ema1, ema2) * (1 + auto_unstuck_ema_dist)`.  
For long, auto unstuck entries are at lower band, closes at upper band.  
For short, auto unstuck entries are at upper band, closes at lower band.
