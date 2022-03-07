# Backtesting

If you want to see how a configuration will perform, you can run it through a backtest that is provided by Passivbot.
While backtesting will not give you a guarantee on the performance for the future, it does give you valuable insight
into how the bot would have performed in the past.

When you run the backtester, you should receive some output as it begins downloading the necessary price
data from the exchange (using the API keys you provided earlier). The price data is cached on the machine
and can be re-used between backtests and optimize sessions. This also means if you interrupt or close 
the process, it will continue downloading price data where it left off. 
The bot comes packaged with a downloader that allows the rapid retrieval of price data based upon
provided dates, and works independently of the backtesting unit.

!!! Warning
    The bot runs backtests on trade data, making it as accurate as possible in backtesting. Be aware that other factors
    like latency and exchange connection issues can also play a role when you run a bot live.

## Running a backtest

To execute a backtest, you can execute the following command from the root folder:

```shell
python3 backtest.py path/to/config_to_test.json
```

Running this command without any arguments other than path to live_config will make the backtester use the details provided in the `configs/backtest/default.hjson` file.
When you first checkout the project, you will need to setup your exchange credentials in the `api-keys.json` (please read [Running live](live.md) for more details).
The backtest needs a connection to the exchange to be able to download the trade data required for the backtest.

In the `configs/backtest/default.hjson`, the account name (specified in `api-keys.json`) needs to be provided. After updating
the `configs/backtest/default.hjson` with your account name, you should be able to succesfully run a backtest.

Apart from the account name, there are a number of other parameters you can specify in the backtest configuration file:

* the exchange
* the account name (must match one specified in `api-keys.json`)
* the symbol(s) to backtest on
* the latency to simulate during backtesting
* the starting balance
* the start and end date for the backtest

### Command-line arguments

Other than modifying the `default.hjson` file, it is also possible to specify a number of configuration options to use via the commandline.
One or more arguments are provided to the backtester using the following syntax on the command line:

```shell
python3 backtest.py <key> <value>
```

The following options can be provided to the backtester. Note that any argument provided will override a value specified in the backtest configuration file.

| Key | Description
| --- | -----------
| --nojit | Disables the use of numba's just in time compiler during backtests
| -b / --backtest_config | The backtest config hjson file to use<br/>**Default value:** configs/backtest/default.hjson
| -d / --download-only | Instructs the backtest to only download the data, but not dump the ticks caches to disk
| -s / --symbol | The symbol(s) to run the backtest on, separated by a `,`
| -u / --user | The name of the account used to download trade data
| --start_date | The starting date of the backtest<br/>**Syntax:** YYYY-MM-DD
| --end_date | The end date of the backtest<br/>**Syntax:** YYYY-MM-DD
| -bd / --base_dir | the base directory to place the output files in<br/>**Default:** `backtests`
| -oh / --ohlcv | use 1m ohlcv instead of 1s tick samples

## Backtest results

When the backtest is completed, the results will be shown on the console. This includes things like average daily gain,
maximum time that Passivbot was stuck etc. Apart from showing these results in the terminal, these results are
also stored in `backtests/{exchange}/{symbol}/plots/{datetime}/backtest_result.txt`. This folder will also
include the actual `live_config.json` file that was used for the plot, and several graphical plots. One of these
for example is the `balance_and_equity_sampled_{long/short}.png`, which shows how the balance and equity evolved during the course of
the backtest.

The file `balance_and_equity.png` will show how the balance and equity progressed during the period being backtested. The
blue line in the graph represents the balance, and the orange line represents the equity.

Besides this file, `whole_backtest_{long/short}.png` and a number of `backtest_XofY.png` are also created.  
The latter represent the entire backtest period, but are split up into separate files for easier inspection and zooming.

On these plots the blue dashed line is the long position price and the red dashed line is the short position price.  
Blue dots are buys (long entries or short closes) and red dots are sells (short entries or long closes).  
Red Xs are auto unstuck sells (long closes or short entries), blue Xs are auto unstuck buys (long entries or short closes)  
and in static grid mode the secondary entry is a larger green dot.

The `auto_unstuck_bands_{long/short}.png` plots show the price thresholds at which auto unstucking orders would fill.  
`initial_entry_band_{long/short}.png` shows the EMA limited initial entries.
