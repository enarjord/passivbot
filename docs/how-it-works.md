# How it works

A python script is given access to user's exchange account and listens to websocket stream of live trades in the selected market, automatically creating and cancelling limit buy and sell orders.

It works in futures and spot markets.  If exchange allows it, hedge mode is used to allow simultaneous long and short positions.  
In spot markets bot mimicks futures behavior, allowing only long positions with max 1x leverage.

The bot bases its decisions on account's positions, open orders and balance, and multiple EMAs of different spans.

The EMAs are used to smoothe initial entries and stop loss orders.

Bot's live behavior is deterministic and may be simulated on historical price data, using the included backtester.

Its behavior is changed by adjusting the configuration to suit a particular market and user's preference.

Configuration's parameters may be optimized to a given period of historical data by backtesting up to thousands of candidates, converging upon the config whose backtest result best satisfy specified criteria.

## Grid Trading

Passivbot may be described as a market making DCA scalping grid trader.

- Market maker: passivbot is a pure maker. It never takes orders, only makes them.

- DCA: Dollar Cost Averaging. It will typically make up to several reentries after an initial entry, in an attempt to acheive better average entry price, similar to a Martingale betting system.

- Scalping: It will typically close positions at small markups, in the range of 0.1-2.0%

- Grid trading: Its reentries may be calculated in advance and put on the books, thus making a grid of orders.


## Noise Harvesting

There are price fluctuations in most markets, and the price noise may be "harvested".  Ideally passivbot functions like a wind turbine or solar panel, passively gathering energy (money) from the environment (market).

For version specific information about configuring or using your version of the bot, refer to your version's
documentation using the version popup at the bottom of the website.

## Wallet exposure

In futures markets bot uses cross mode, so entire wallet is used as margin to any position.  This means that one bad position
may potentially liquidate entire wallet.

The bot does not use leverage in its internal logic, but the equivalent concept "wallet exposure",
which is defined as position size in terms of margin token divided by unleveraged wallet balance.

For example, 10% of available leveraged balance in a position at cross 20x leverage is equivalent to wallet_exposure==2.0,
or 100% of available leveraged balance in a position at cross 2x leverage.

Upon startup bot will automatically set leverage to a high enough value to prevent insufficient margin errors.  Other than that,
leverage is irrelevant.

wallet_exposure == 1.0 means 100% of unleveraged balance is in a position.  
wallet_exposure == 2.0 means 200% of unleveraged balance is in a position.  
wallet_exposure == 0.15 means 15% of unleveraged balance is in a position.

Bot will not allow a position's wallet_exposure to grow higher than wallet_exposure_limit (whose value is defined in live config).

You can prevent a position exceeding, say, 10% of unleveraged balance by making sure that the
wallet_exposure_limit parameter does not exceed 0.1 (==10%).  If the price of the symbol would drop to 0, you'd lose
a maximum of 10% of your wallet.

It is possible to run multiple pairs on the same account.  
For example, if wallet_exposure_limit were set to 0.05 on 10 bots, all of them combined will not expose
more than 50% of the account's unleveraged balance.

!!! Info
    Please check the [Configuration](configuration.md) page for an in-depth description of the configuration parameters available.
