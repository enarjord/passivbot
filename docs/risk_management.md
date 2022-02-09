# Risk Management

First priority to trading should be proper risk management.

## Leverage

On futures markets with leverage, passivbot may expose more than 100% of the wallet's funds.  
Passivbot uses only (unleveraged) wallet balance in its calculations,  
so adjusting leverage on exchange will make no difference on risk, profit or bot behavior,  
as long as leverage is set high enough for the bot to make its grid according to the configuration.

## Wallet Exposure

To measure a position's risk, passivbot finds the ratio of position size to total unleveraged balance.  
The formula for wallet exposure is

`wallet_exposure = (position_size * position_price) / unleveraged_wallet_balance` for linear,  
`wallet_exposure = (position_size / position_price) / unleveraged_wallet_balance` for inverse markets.

wallet_exposure==0.0 means no position
wallet_exposure==1.0 means 100% of unleveraged wallet balance is in position.
wallet_exposure==4.0 means 400% of unleveraged wallet balance is in position.

E.g. if wallet balance is $1000, linear long position size is 100.0 and position price is 35.0,  
then wallet_exposure is `100 * 35 / 1000 == 3.5`

## Wallet Exposure Limit

Each bot is configured with a parameter wallet_exposure_limit, greater than which the bot will not allow a position's wallet_exposure to grow.

For example, if wallet_exposure_limit=0.6, the bot will not make any more entries when a position's wallet_exposure >= 0.6.

## Bankruptcy and liquidation

Bankruptcy is defined as when `equity == (balance + unrealized_pnl) == 0.0`, that is, when total debt is equal to total assets.

Liquidation happens when the exchange force closes a position to avoid it going into negative equity.  
This usually happens before actual bankruptcy is reached, in order for exchange to cover slippage costs.

Bankruptcy price may be calculated from position and balance.

E.g.  
For linear long:
If wallet_exposure==1.0, bankruptcy price is zero.  
If wallet_exposure==2.0, bankruptcy price is 50% lower than pos price.  
If wallet_exposure==3.0, bankruptcy price is 33.33% lower than pos price.  
If wallet_exposure==10.0, bankruptcy price is 10% lower than pos price.  

For inverse long:
If wallet_exposure==1.0, bankruptcy price is 50% lower than pos price.  
If wallet_exposure==2.0, bankruptcy price is 33.33% lower than pos price.  
If wallet_exposure==3.0, bankruptcy price is 25% lower than pos price.  
If wallet_exposure==10.0, bankruptcy price is 9.09% lower than pos price.  


## Getting stuck

When a bot has no more entries left in its entry grid and wallet_exposure_limit is reached or exceeded, it is termed "getting stuck".  
If a bot is stuck in a long position and the price keeps falling, the distance between position price and market price grows larger,  
and closing the position in profit becomes less likely.

Therefore it is desirable to bring position price closer to price action such that the position may be closed in profit on a small bounce.  
This is achieved by increasing the position size at an auspicioius moment, thus bringing the position price closer to the price action.  
However, the larger the position size, the higher the risk of liquidation, should the price keep moving the wrong way.

## Multiple bots sharing sharing same wallet in cross mode

While correlation is observed in most markets in general and in crypto markets in particular  
(e.g. if the price of bitcoin crashes, most other cryptos tend to follow closely),  
it is also observed that the dead cat often bounces at slightly different times and different heights.

A thousand coin flips will converge on 500 heads and 500 tails.  One single coin flip will be either heads or tails.  
Say that on average there's a 30% chance of getting stuck in the typical market crash.  
It may be more desirable to end up with 3 out of 10 bots stuck with wallet_exposure==0.1 each than with 1 single bot stuck with wallet_exposure==1.0.

