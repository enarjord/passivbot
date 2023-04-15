# Forager

Moves passivbots around to wherever volatility is highest.  

## How it works

The script does the following:
- fetch min costs for all symbols
- filter symbols by min cost
- get 15m ohlcvs for all approved symbols
- compute volatility for the last 25 hours for all approved symbols
- get the account's currently active bots
- generate and dump yaml file with new bots and old bots with graceful-stop
- sleep for 60 secs to avoid API rate limiting due to many calls to fetch ohlcvs
- kill previous tmux session
- use tmuxp to load new tmux session
- sleep for one hour
- repeat

Volatility is defined as `ohlcvs.close.std() / ohlcvs.close.mean()`


## Usage
```shell
python3 forager.py configs/forager/config_file.hjson
```

## Requirements
In addition to requirements.txt, tmux and tmuxp

## Config


| Parameter                     | Description |
| ----------------------------- | ------------- |
| `user`						| User as defined in api-keys.json.
| `twe_long`					| Total Wallet Exposure Long.  WE_limit per bot is TWE / n_bots.
| `twe_short`					| Total Wallet Exposure Short.  WE_limit per bot is TWE / n_bots.
| `n_longs`						| Number of long bots.
| `n_shorts`					| Number of short bots.
| `max_min_cost`				| Exclude symbols whose min_cost > max_min_cost.
| `n_ohlcvs`					| Number of ohlcvs to fetch (100 candles of 15m is 25 hours).
| `ohlcv_interval`				| Default is 15m.
| `leverage`					| Set leverage.
| `price_distance_threshold`	| Don't make limit orders whose price is further away from market price than price_distance_threshold.
| `max_n_panes`					| Max number of panes per tmux window.
| `default_config_path`			| Default config to use.
| `approved_symbols_only`		| If true, allow only symbols present in live_configs_map.  If false, will use default_config_path when symbol is missing from live_configs_map
| `live_configs_map`			| Live configs to to use with particular symbols.
| `gs_mm`						| min_markup to use on bots on -gs
| `gs_mr`						| markup_range to use on bots on -gs
| `gs_lw`						| wallet_exposure_limit_long to use on bots on -gs
| `gs_sw`						| wallet_exposure_limit_short to use on bots on -gs


