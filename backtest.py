import argparse
import asyncio

from bots.configs import BacktestConfig
from helpers.downloader import Downloader
from helpers.loaders import load_config_files, load_module_from_file, get_strategy_definition, add_argparse_args, \
    prep_config
from helpers.print_functions import print_


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
        b_config = BacktestConfig(config['quantity_step'] if 'quantity_step' in config else 0.0,
                                  config['price_step'] if 'price_step' in config else 0.0,
                                  config['minimal_quantity'] if 'minimal_quantity' in config else 0.0,
                                  config['minimal_cost'] if 'minimal_cost' in config else 0.0,
                                  config['call_interval'] if 'call_interval' in config else 1.0,
                                  config['historic_tick_range'] if 'historic_tick_range' in config else 0.0,
                                  config['historic_fill_range'] if 'historic_fill_range' in config else 0.0,
                                  config['leverage'] if 'leverage' in config else 1.0,
                                  config['symbol'] if 'symbol' in config else '', 0.0, 0.0,
                                  config['latency_simulation_ms'] if 'latency_simulation_ms' in config else 100,
                                  config['market_type'] if 'market_type' in config else 'futures',
                                  config['inverse'] if 'inverse' in config else False,
                                  config['contract_multiplier'] if 'contract_multiplier' in config else 1.0)

        # Create a strategy based on the strategy module and the provided class
        strategy = getattr(strategy_module, config['strategy_class'])(strategy_config)
        # Initialize some basic data
        downloader = Downloader(config)
        data = await downloader.get_candles()
        # Create the backtest bot
        bot = bot_module.BacktestBot(b_config, strategy, data)
        # Initialize bot
        bot.init()
        bot.update_balance(config['starting_balance'])
        # Start run
        fills, statistics = bot.start_websocket()
    except Exception as e:
        print_(['Could not start', e])


if __name__ == '__main__':
    asyncio.run(main())
