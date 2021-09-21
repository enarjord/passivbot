# Risk Management

First priority to trading should be proper risk management.

## Leverage

Passivbot does not use leverage.  It uses only unleveraged balance in all its calculations.  
Leverage may be set to any value and will not affect bot behavior,  
as long as it is set high enough to avoid insufficent margin errors.

## PBR

To give a metric of a position's risk, passivbot finds the ratio of position size to available unleveraged balance.  
The formula for Position cost to Balance Ratio (PBR) is  
`pbr = (position_size * position_price) / unleveraged_wallet_balance` for linear,  
`pbr = (position_size / position_price) / unleveraged_wallet_balance` for inverse markets.

pbr==0.0 means no position
pbr==1.0 means 100% of unleveraged wallet balance is in position.
pbr==4.0 means 400% of unleveraged wallet balance is in position.

E.g. if wallet balance is $1000, linear long position size is 100.0 and position price is 35.0,  
then pbr is `100 * 35 / 1000 == 3.5`

## PBR Limit

Each bot is configured with a parameter pbr_limit, greater than which the bot will not allow a position's pbr to grow.

For example, if pbr_limit=0.6, the bot will not make any more entries when a position's pbr >= 0.6.

## Bankruptcy and liquidation

Bankruptcy is defined as when `equity (balance + unrealized_pnl) == 0.0`, that is, when total debt is equal to total assets.

Liquidation happens when the exchange force closes a position to avoid it going into negative equity.  
This usually happens before actual bankruptcy is reached, in order for exchange to cover slippage costs.

Bankruptcy price may be calculated from position and balance.

E.g.  
For linear long:
If pbr==1.0, bankruptcy price is zero.
If pbr==2.0, bankruptcy price is 50% lower than pos price.
If pbr==3.0, bankruptcy price is 33.33% lower than pos price.
If pbr==10.0, bankruptcy price is 10% lower than pos price.

For inverse long:
If pbr==1.0, bankruptcy price is 50% lower than pos price.
If pbr==2.0, bankruptcy price is 33.33% lower than pos price.
If pbr==3.0, bankruptcy price is 25% lower than pos price.
If pbr==10.0, bankruptcy price is 9.09% lower than pos price.


## Getting stuck

When a bot has no more entries left in its entry grid and pbr_limit is reached or exceeded, it is termed "getting stuck".  
If a bot is stuck in a long position and the price keeps falling, the distance between position price and market price grows larger,  
and closing the position in profit becomes less likely.

Therefore it is desirable to bring position price closer to price action such that the position may be closed in profit on a small bounce.  
This is achieved by increasing the position size at an auspicioius moment, thus bringing the position price closer to the price action.  
However, the larger the position size, the higher the risk of liquidation, should the price keep moving the wrong way.

## Multiple bots sharing sharing same wallet in cross mode

One method of risk managment is diversifying across multiple markets.  

While correlation is observed in most markets in general and in crypto markets in particular
(e.g. if the price of bitcoin crashes, most other cryptos tend to follow closely),  
it is also observed that the dead cat often bounces at slightly different times.

Since passivbot is a DCA scalper aiming to stay close to price action,  
10 bots with 0.1 pbr_limit each means less capital is required to unstuck each stuck position  
than if one bot were stuck with pbr==1.0.

