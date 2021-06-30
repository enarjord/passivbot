# Running the bot live

### Usage:

#### Supports exchanges Bybit futures and Binance futures, using hedge mode when supported

1. Add your api key and secret in file [api-keys.json](api-keys.json)
2.
```shell
python3 start_bot.py {account_name} {symbol} {path/to/config.json}
   ```

Example:

```shell
python3 start_bot.py binance_01 XMRUSDT configs/live/binance_xmrusdt.json
```

## Default configurations

There are a limited number of configurations provided by default in the repository. These configurations
are optimized over a longer period of time, and provide a good starting point for setting up your perfect
configuration. You can find these configurations in the [configs/live](configs/live) directory.

## Stopping bot

For graceful stopping of the bot, set `do_long`and `do_shrt` both to `false`, and bot will continue as normal, opening
no new positions, until all existing positions are closed.

If you have telegram configured, you can use the /stop command to stop the bot using various stop modes. 