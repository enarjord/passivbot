import argparse

from bots.configs import BacktestConfig
from helpers.downloader import Downloader
from helpers.loaders import load_base_config, load_module_from_file, get_strategy_definition, load_exchange_settings
from helpers.print_functions import print_

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='PassivbotBacktest', add_help=True,
                                        description='Grid trading bot with variable strategies.')
    argparser.add_argument('live_config', type=str, help='Live config to use.')
    argparser.add_argument('-b', '--backtest_config', type=str, required=True, dest='backtest_config',
                           default='configs/backtest/test.hjson', help='Backtest config to use.')
    args = argparser.parse_args()
    try:
        # Load the config
        config = {}
        config.update(load_base_config(args.live_config))
        config.update(load_base_config(args.backtest_config))
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
        market_settings = await load_exchange_settings(config['exchange'], config['symbol'], config['user'],
                                                       config['market_type'])
        config.update(market_settings)

        config['session_name'] = 'some_session'
        config['caches_dirpath'] = f"backtests\\{config['exchange']}\\{config['symbol']}\\caches\\"

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
        # Start run
        bot.start_websocket()
    except Exception as e:
        print_(['Could not start', e])
