# Changelog

All notable changes to this project will be documented in this file.

## [v3.5.2] - 2021-05-10
- walk forward optimization
- more advanced backtest analysis tools

## [v3.5.1] - 2021-05-09
- optimize with sliding window
- restructured dirs
- new dirs
- `backtests/{exchange}/{symbol}/optimize/`
- `backtests/{exchange}/{symbol}/plots/`
- `backtests/{exchange}/{symbol}/caches/`
- if end_date = -1 in backtest_config.hjson, downloader will make a new ticks_cache.npy for each session
- to reuse prev ticks cache, set end_date to a static date
- when optimizing, intermediate_best_result.json is dumped as usable live_config

## [v3.5.0] - 2021-05-02
- added volatility indicator
- split extract optimize.py from backtest.py
- now `python3 backtest.py backtest_config.hjson live_config.json` will backtest and plot single candidate
- `python3 optimize.py backtest_config.hjson` will optimize
- refactoring of all @njit calcs, separating them into jitted.py
- added telegram bot

## [v3.4.0] - 2021-04-14
- added binance USDT-margined backtester with stoploss
- added binance COIN-margined backtester with stoploss
- changed backtester usage -- now needs to specify whole path to .hjson config file

## [v3.3.3] - 2021-04-05

- added stop loss
- todo: backtester with stop loss

## [v3.3.2] - 2021-04-05

- changed api key format. put api key/secret in api-keys.json in main dir.
- changed name of live_settings dir to live_configs, removed subdirs binance/bybit
- changed how to use start_bot.py. see updated startup instructions
- improved backtester multiprocessing memory usage

## [v3.1.1] - 2021-04-01

- Binance inverse futures coin margined markets now supported

## [v3.3.0] - 2021-03-30

- Bybit usdt linear perpetual and Bybit inverse perpetual markets now supported
- new downloader for historical backtesting data

## [v3.2.1] - 2021-03-25

- bug fixes
- Bybit backtester improvements and bug fixes
- numba is now enabled by default, use --nojit to disable numba
- several renamings

## [v3.2.0] - 2021-03-23

- implemented particle swarm optimizationg algorithm, replacing jackrabbit
- Bybit hedge mode with inverse futures
- removed config param close_qty_pct
- removed config param balance_pct
- removed config param max_markup
- added config param markup_range

## [v3.1.0] - 2021-03-21

- removed setting min_close_qty_multiplier
- added setting close_qty_pct, which complements entry_qty_pct
- rewrote iter_long/shrt_closes
- fixed memory leak bug

## [v3.0.1] - 2021-03-18

- hedge mode backtester implemented
- emas added back

## [v3.0.0] - 2021-03-12

- Binance hedge mode implemented (Bybit not yet supported)
- emas removed
- stop loss removed

## [v2.0.3] - 2021-03-02

- new default Bybit config
- behavior change: reentry qtys may now be smaller than initial entry qty
- backtest iterates a numpy array instead of a python list of dicts for reduced ram usage

## [v2.0.2] - 2021-03-01

- more jit'ed calcs

## [v2.0.1] - 2021-02-28

- added optional just-in-time compiling for faster backtesting

## [v2.0.0] - 2021-02-23

- major update to backtester
- new backtest usage syntax
- other changes

## [v1.3.3] - 2021-02-18

- minor refactoring

## [v1.3.0] - 2021-02-17

- added indicator_settings["tick_ema"]["spread"] to live bot and backtester
    - optional setting -- ema_spread defaults to 0.0 if not present in config file

## [v1.2.1] - 2021-02-17

- backtester will cache exchange fetched settings after first run
- backtester will prevent using leverage higher than max leverage, in case max leverage set in ranges.json was too high

## [v1.2.0] - 2021-02-17

- bug fixes
- change in backtesting_notes.ipynb
    - automatic plot dump
    - other changes
- backtester now fetches relevant data from exchanges instead of user having to input them manually

## [v1.1.0] - 2021-02-16

- bug fixes v1.0.2
- updated default Bybit live settings v1.1.0

## 2021-02-12

- added indicator_settings["funding_fee_collect_mode"]
    - if true, will enter long only if predicted fundig rate is negative and enter short only if predicted funding rate
      is positive
- added indicator rsi (not finished, not active)
- changed entry_qty_pct formula
    - before initial_entry_qty = balance_ito_contracts * entry_qty_pct
    - now initial_entry_qty = balance_ito_contracts * leverage * entry_qty_pct
- added logging
- added "config_name" and "logging_level" to live settings
- added break_on condition: break if balance + pnl < starting_balance

## 2021-02-10

- renamed settings["default_qty"] to settings["entry_qty_pct"]
- settings["entry_qty_pct"] may now also be a positive value
- renamed settings["balance"] to settings["balance_pct"]
- settings["balance_pct"] may now also be a positive value
- added balance_pct to backtester. backtester will now behave like live bot, taking balance_pct into account
    - actual balance is used for liq price calc, otherwise balance * balance_pct is used

## 2021-02-09

- added classic stop loss

## 2021-02-08

- added min_close_qty_multiplier

## 2021-02-03

- backtester break conditions change
- bug fixes

## 2021-01-30

- changed backtesting results formatting
- fixed insufficient margin error
- many other fixes and changes...
- added possibility of running same backtest in two or more terminals for better cpu utilization

## 2021-01-23

- removed static mode
- added indicator ema
- rewrote backtester

## 2021-01-19

- renamed settings["margin_limit"] to settings["balance"]
- bug fixes and changes in trade data downloading
- if there already is historical trade data downloaded, run the script `rename_trade_data_csvs.py` to rename all files
