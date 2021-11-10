# Spot trading

Passivbot works on Binance spot markets.

Example command:  
`python3 passivbot.py binance_01 ETHBTC path/to/live_config.json -m spot -ab 0.1`  

Start the bot normally by running the script passivbot.py.  
Specify the user as normal, e.g. `binance_01`.  
Specify the symbol.  Symbols may be quoted in USDT/BUSD/BTC/ETH/BNB and so on.  Any spot symbol.  
Direct the bot to desired live config.  
Specify that the market type is spot and not futures with `--market_type spot` or `-m spot`.  

It is recommended to add kwarg `--assigned_balance x` or `-ab x` in order to fix wallet balance to a given value.  
This because in spot trading there are no positions, no wallet balance and no unrealized PnL.  There is only plain buy and sell.  
So wallet balance and position must be simulated by looking in past user trade history.  
Total spot wallet balance fetched thru REST API is total wallet equity.  
What is needed is wallet_balance defined as `quote_balance + coin_balance * pos_price`.  
Multiple spot bots sharing same wallet will be unaware of other spot bots' positions.  
As a patch for this issue, instead of computing wallet_balance by fetching info from exchange, one may set wallet_balance to some desired static value, typically slightly less than quote balance plus value in terms of quote token of all spot positions.

For example, if spot wallet has 2.5 ETH and one wants to start 5 ETH quoted spot bots,  
start 5 bots with different ETH quoted symbols:  
`python3 passivbot.py binance_01 XXXETH path/to/live_config.json -m spot -ab 2.45`  

Also note that sum of wallet_exposure_limits of all spot bots sharing same quote token shouldn't exceed 1.0.
