# How it works

A python script is given access to user's exchange account and listens to websocket stream of live trades in the selected market, automatically creating and cancelling limit buy and sell orders.

It works primarily in futures markets, spot markets in development.  If exchange allows it, hedge mode is used to allow simultaneous long and short positions.

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

One important thing to understand about the bot is that it trades on the futures market using cross mode. This means that any open
position is able to tap into the enture futures wallet for margin to support the position. In order to guard the wallet from
undesired exposure to position margin, it uses a concept called position-cost-to-balance-ratio (abbreviated as pbr).

If you're going to use this bot, it's important to understand this concept so you can understand and configure the bot the way you want to.

The main thing to understand is that the bot does NOT work with leverage as the exchanges do. Instead, the bot works with a concept
of a borrow cap. This borrow cap is the maximum amount of leveraged wallet you have at your disposal. The borrow cap is defined
as the maximum leverage allowed on the symbol by the exchange. This symbol represents the maximum value for the pbr_limit parameter.

If you would have a wallet of 100 USDT and traded on BTCUSDT (which has a max leverage of 125), your maximum position cost (position size & position price)
could be up to 12500 (125 * 100).

You can make sure that your position will not be allowed to exceed X% of your unleveraged balance by making sure that your
pbr_limit + pbr_stop_loss parameters do not exceeed 0.1 (=10%). If the price of the symbol would drop to 0, you'd lose
a maximum of 10% of your wallet.

The bot will makee sure that your position size does not ever surpass pbr_limit + pbr_stop_loss. When the bot crosses this value,
it reduced to position down to the specific pbr_limit.

A simpler explanation: if you have 100 bananas and use 10 of them to buy 12 apples, and theee price of apples in terms of bananas drops to zero,
your equity is 90 bananas (meaning your max exposure is 10%). This makes it very possible to run multiple pairs on the same account. 
If you were to set the pbr_limit to 0.05 on 10 bots, all of them combined will not expose more than 50% of your unleveraged balance.

!!! Info
    Please check the [Configuratio](configuration.md) page for an in-depth description of the configuration parameters available.