# How it works

A python script is given access to user's exchange account and listens to websocket stream of live trades in the selected market, automatically creating and cancelling limit buy and sell orders.

It works primarily in futures markets, spot markets in development.  If exchange allows it, hedge mode is used to allow simultaneous long and short positions.

The bot bases its decisions on account's positions, open orders and balance, and multiple EMAs of different spans.

The EMAs are used to smoothe initial entries and stop loss orders.

Bot's live behavior is deterministic and may be simulated on historical price data, using the included backtester.

Its behavior is changed by adjusting the configuration to suit a particular market and user's preference.

Configuration's parameters may be optimized to a given period of historical data by backtesting up to thousands of candidates, converging upon the config whose backtest result best satisfy specified criteria.

## Grid Trading

passivbot may be described as a market making DCA scalping grid trader.

- Market maker: passivbot is a pure maker. It never takes orders, only makes them.

- DCA: Dollar Cost Averaging. It will typically make up to several reentries after an initial entry, in an attempt to acheive better average entry price, similar to a Matingale betting system.

- Scalping: It will typically close positions at small markups, in the range of 0.1-2.0%

- Grid trading: Its reentries may be calculated in advance and put on the books, thus making a grid of orders.


## Noise Harvesting

There are price fluctuations in most markets, and the price noise may be "harvested".  Ideally passivbot functions like a wind turbine or solar panel, passively gathering energy (money) from the environment (market).



For version specific information about configuring or using your version of the bot, refer to your version's
documentation using the version popup at the bottom of the website.