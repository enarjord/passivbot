import argparse
import asyncio
import os
from time import time

from bots.configs import BacktestConfig
from helpers.analyzers import analyze_fills
from helpers.converters import fills_to_frame, statistics_to_frame, candle_array_to_frame
from helpers.downloader import Downloader
from helpers.loaders import load_config_files, add_argparse_args, prep_config, get_strategy_and_bot_module
from helpers.misc import make_get_filepath, ts_to_date
from helpers.optimized import round_
from helpers.plotter import dump_plots
from helpers.print_functions import print_


def backtest_wrap(bot, config, data):
    # Initialize bot
    bot.init()
    bot.update_balance(config['starting_balance'])
    print_(['Number of days:', round_(config['number_of_days'], 0.1)])
    print_(['Starting balance:', config['starting_balance']])
    print_(['Backtesting...'])
    start = time()
    # Start run
    fills, statistics, accepted_orders = bot.start_websocket()
    print_([f'{time() - start:.2f} seconds elapsed'])
    fill_frame = fills_to_frame(fills)
    statistic_frame = statistics_to_frame(statistics)

    if fill_frame.empty:
        print_(['No fills'])
        return
    result = analyze_fills(fill_frame, statistic_frame, config, data[0][0], data[-1][0])
    config['result'] = result
    config['plots_dirpath'] = make_get_filepath(
        os.path.join(config['plots_dirpath'], f"{ts_to_date(time())[:19].replace(':', '')}", ''))
    fill_frame.to_csv(config['plots_dirpath'] + "fills.csv", index=False)
    candle_frame = candle_array_to_frame(data)
    print_(['Dumping plots...'])
    dump_plots(config, fill_frame, statistic_frame, candle_frame)


async def main() -> None:
    argparser = argparse.ArgumentParser(prog='PassivbotBacktest', add_help=True,
                                        description='Grid trading bot with variable strategies.')
    argparser.add_argument('live_config', type=str, help='Live config to use.')
    argparser = add_argparse_args(argparser)
    args = argparser.parse_args()
    try:
        # Load the config
        config = await prep_config(args)
        config.update(load_config_files(args.live_config))
        strategy_module, bot_module = get_strategy_and_bot_module(config, args.nojit)
        # Create the strategy configuration from the config
        strategy_config = strategy_module.convert_dict_to_config(config['strategy'])
        # Create a backtest config
        b_config = BacktestConfig(config['quantity_step'] if 'quantity_step' in config else 0.0,
                                  config['price_step'] if 'price_step' in config else 0.0,
                                  config['minimal_quantity'] if 'minimal_quantity' in config else 0.0,
                                  config['minimal_cost'] if 'minimal_cost' in config else 0.0,
                                  config['call_interval'] if 'call_interval' in config else 1.0,
                                  config['historic_tick_range'] if 'historic_tick_range' in config else 0.0,
                                  config['historic_fill_range'] if 'historic_fill_range' in config else 0.0,
                                  config['tick_interval'] if 'tick_interval' in config else 0.25,
                                  config['statistic_interval'] if 'statistic_interval' in config else 3600,
                                  config['leverage'] if 'leverage' in config else 1.0,
                                  config['symbol'] if 'symbol' in config else '',
                                  config['maker_fee'] if 'maker_fee' in config else 0.0,
                                  config['taker_fee'] if 'taker_fee' in config else 0.0,
                                  config['latency_simulation_ms'] if 'latency_simulation_ms' in config else 100,
                                  config['market_type'] if 'market_type' in config else 'futures',
                                  config['inverse'] if 'inverse' in config else False,
                                  config['contract_multiplier'] if 'contract_multiplier' in config else 1.0)

        # Create a strategy based on the strategy module and the provided class
        strategy = getattr(strategy_module, config['strategy_class'])(strategy_config)
        # Initialize some basic data
        downloader = Downloader(config)
        data = await downloader.get_candles()
        config['number_of_days'] = round_((data[-1][0] - data[0][0]) / (1000 * 60 * 60 * 24), 0.1)
        # Create the backtest bot
        bot = bot_module.BacktestBot(b_config, strategy, data)
        backtest_wrap(bot, config, data)
    except Exception as e:
        print_(['Could not start', e])


if __name__ == '__main__':
    asyncio.run(main())
