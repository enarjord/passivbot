import argparse
import asyncio
import json

from bots.base_live_bot import LiveBot
from bots.configs import LiveConfig
from helpers.loaders import load_base_config, load_module_from_file
from helpers.print_functions import print_


async def start_bot(bot: LiveBot):
    """
    Starts the three continuous functions of the bot.
    :param bot: The bot to start.
    :return:
    """
    await asyncio.gather(bot.start_heartbeat(), bot.start_user_data(), bot.start_websocket(),
                         bot.start_historic_tick_fetching(), bot.start_historic_fill_fetching())


async def main() -> None:
    argparser = argparse.ArgumentParser(prog='PassivbotLive', add_help=True,
                                        description='Grid trading bot with variable strategies.')
    argparser.add_argument('user', type=str, help='User/account_name defined in api-keys.json.')
    argparser.add_argument('symbol', type=str, help='Symbol to trade.')
    argparser.add_argument('live_config', type=str, help='Live config to use.')
    args = argparser.parse_args()
    try:
        # Load the config
        config = load_base_config(args.live_config)
        accounts = json.load(open('api-keys.json'))
        account = accounts[args.user]
        config['user'] = args.user
        config['exchange'] = account['exchange']
        config['symbol'] = args.symbol
        config['live_config'] = args.live_config
        # Create the strategy module from the specified file
        strategy_module = load_module_from_file(config['strategy_file'], 'strategy')
        # Create the strategy configuration from the config
        strategy_config = strategy_module.convert_dict_to_config(config['strategy'])
        # Create a strategy based on the strategy module and the provided class
        strategy = getattr(strategy_module, config['strategy_class'])(strategy_config)

        config = LiveConfig(config['symbol'], config['user'], config['exchange'], config['market_type'],
                            config['leverage'], config['call_interval'],
                            config['historic_tick_range'] if 'historic_tick_range' in config else 0.0,
                            config['historic_fill_range'] if 'historic_fill_range' in config else 0.0)
        if config.exchange == 'binance':
            from bots.binance import BinanceBot
            bot = BinanceBot(config, strategy)
        else:
            print_(['Exchange not supported.'], n=True)
            return
        await bot.async_init()
        await start_bot(bot)
    except Exception as e:
        print_(['Could not start', e])
        return


if __name__ == '__main__':
    asyncio.run(main())
