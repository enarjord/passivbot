# Backtesting

##Summary

1. modify `configs/backtest/default.hjson` as desired
2. run with `python3 backtest.py {path_to_live_config_to_test.json}`
3. optional args: `-b or --backtest-config`: use different backtest config

Will use numba's just in time compiler to speed up backtesting.

##Detailed explanation

You need to specify your settings for backtesting. Generally, setting your variables manually can lead to the bot going awry over time, so exercise caution when using hand-crafted configurations. PassivBot comes with a backtester, a script designed to assist you in finding the best configuration for your use case.

Utilizing the backtester is the best way to come up with new configurations, but requires that you have a basal understanding of the configuration file. The backtester's job is to look at a coin pair's price history (ETH/USDT for the last 30 days for example), examine your provided conditions (leverage, percent balance, grid spacing and so on...), and test those settings over the timeframe you selected. The bot will iterate through every trade as if it were doing it live, and return the best found results to the user. These settings (if they are profitable) are then re-used to generate new settings based upon the most likely profitable configurations. This process repeats itself as many times as the user chooses and upon completion returns the most profitable configuration for the bot, over that timeframe. There is a default backtesting configuration with the bot in every version, so examine it and set the test parameters to suit your desired results.

Note that there are two different kinds of configuration files. The formatting for a backtesting configuration file is not always the same as the formatting for a live usage configuration file. Always retain the template configuration file as a reference for the formatting of your version. If an update makes your version obsolete, you will need a reference for the formatting if you don't plan to immediately update. Some parameters have specific formatting they must abide by. For more in-depth information on configuring each of the values, refer to the version-specific documentation.

When you run the backtester, you should receive some output as it begins downloading the necessary price data from the exchange (using the API keys you provided earlier). The price data is cached on the machine and can be re-used between backtests. This also means if you interrupt or close the process, it will continue downloading price data where it left off. Newer versions of the bot come packaged with a downloader that allows the rapid retrieval of price data based upon provided dates, and works independently of the backtesting unit. Once the necessary price data has been downloaded, the backtester will begin with the starting candidate, test against the price history, and continue iterating through the ranges for each variable. Once the history for a given asset is downloaded, additional price history can simply be tacked on to the end of the cache (done automatically), cutting down testing times. The backtesting process is computed by the CPU, and can be time consuming depending on the testing period, despite optimization. Take this in to account when beginning your test.

The key to finding new, profitable configurations is using the backtester often and familiarizing yourself with the settings and ranges. Adjusting the ranges narrows the proverbial 'area' the PassivBot needs to search for good configurations, reducing the test time while potentially cutting more or less profitable settings out of the search range.