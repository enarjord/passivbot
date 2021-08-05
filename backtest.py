import argparse

import numpy as np

from helpers.loaders import load_base_config, load_module_from_file, get_strategy_definition
from helpers.print_functions import print_

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='PassivbotBacktest', add_help=True,
                                        description='Grid trading bot with variable strategies.')
    argparser.add_argument('-c', '--config', type=str, required=True, dest='c', help='Path to the config')
    args = argparser.parse_args()
    try:
        # Load the config
        config = load_base_config(args.c)
        # Create the strategy module from the specified file
        strategy_module = load_module_from_file(config['strategy_file'], 'strategy')
        # Create the strategy configuration from the config
        strategy_config = strategy_module.convert_dict_to_config(config['strategy'])
        # Get the replacement for the numba strategy specification
        replacement = get_strategy_definition(config['strategy_file'])
        replacement = ('strategy.' + replacement).replace('StrategyConfig', 'strategy.StrategyConfig')
        replacement = ('to_be_replaced_strategy', replacement)
        # Create the bot module from the file including the replacement of the strategy specification
        bot_module = load_module_from_file('bots/backtest_bot.py', 'bot', replacement, 'import strategy')
        # Create a backtest config
        b = bot_module.BacktestConfig(0.0, 0.0, 1.0, 1.0, '', 0.0, 0.0, 0.0)
        # Create a strategy based on the strategy module and the provided class
        strategy = getattr(strategy_module, config['strategy_class'])(strategy_config)
        # Initialize some basic data
        d = np.load('test_data.npy')
        # Create the backtest bot
        bot = bot_module.BacktestBot(b, strategy, d)
        # Initialize bot
        bot.init()
        # Start run
        bot.start_websocket()
    except Exception as e:
        print_(['Could not start', e])
