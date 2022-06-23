# Automatic Profit Transfer

Only Binance and Bybit USDT futures -> spot transfer supported at this time.  

```shell
python3 auto_profit_transfer.py {user_name}
```
Optional kwarg: `-p/--percentage float`:  Set percentage of profit to transfer (per uno, i.e. 0.3==30%).  Default is 0.5.  

If transfer is successful, log of IDs of transfered trades is stored in `logs/automatic_profit_transfer_log_{user}.json`.  
Delete this file to start from scratch.  
It will look max 24h into the past for fills with realized pnl to transfer.  
Execution interval is set to 1h.

