# Passivbot Glossary

Here follows a glossary for terms relevant to Passivbot (WIP)


- position: holding x contracts at price y. Long or short. Passivbot may hold both long and short position in the same coin at the same time. For example: A long position of 0.002 BTCUSDT @ 41,000 is equivalent to buying 0.002 BTC at price $41,000 per bitcoin.
- long position: equivalent to using USDT to buy and hold a coin. The long position profits when price goes up, and loses when price goes down.
- short position: equivalent to borrowing the coin, with USDT as collateral, and selling the coin. The short position profits when price goes down, and loses when price goes up.
- wallet exposure: the ratio of position size to account balance. For example: 0.2 ETHUSDT @ 2,250 with a wallet balance of $1,200, means a wallet exposure of (0.2 * $2,250) / $1,200 == $450 / $1,200 == 0.375 or 37.5%. Wallet exposure may be greater than 1.0 or 100%, by using leverage. 
- wallet exposure limit: the highest wallet exposure the bot will allow for a position.
- total wallet exposure limit: the sum of the wallet exposure limits for each coin being traded.
- initial_qty_pct (initial quantity percent): the percentage of a bot's allocated balance to be used for the initial entry. Typically in the 0.5-2% range. If an exchange's minimum cost per order is greater than balance * WE_limit * initial_qty_pct, the bot will not work properly.
- adg: average daily gain. E.g. adg = 0.0025 == average 0.25% gain daily.
- sharpe ratio: mean of daily gains divided by standard deviation of daily gains.
- drawdown: distance between equity peak to equity trough. E.g. if equity peak was $1000 and equity dropped to $700, then drawdown was `(1000 - 700) / 1000 == 0.3 == 30%`.
- ddown_factor (double down factor): quantity of next grid entry is position size times double down factor. E.g. if position size is 1.4 and down_factor is 0.9, then next entry quantity is 1.4 * 0.9 == 1.26.
- min_markup (minimum markup): the distance from position price to lowest take-profit price. E.g. long position price is 30.0 and min_markup is 0.005, then lowest take-profit price is `30 * (1 + 0.005) == 30.15`
- markup_range: passivbot makes a grid of TP orders from position_price + min_markup to position_price + min_markup + markup_range. E.g. if `pos_price==100`, `min_markup=0.02`, `markup_range=0.03` and `n_close_orders=7`, TP prices are [102, 102.5, 103, 103.5, 104, 104.5, 105]
- n_close_orders (number of take-profit orders): number of orders in take-profit grid.
- EMA: exponential moving average
- EMA_span: span of exponential moving average. Given in minutes.
