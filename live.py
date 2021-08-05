import argparse
import asyncio

from bots.base_live_bot import LiveBot, LiveConfig
from helpers.loaders import load_base_config, load_module_from_file
from helpers.print_functions import print_


async def start_bot(bot: LiveBot):
    """
    Starts the three continuous functions of the bot.
    :param bot: The bot to start.
    :return:
    """
    await asyncio.gather(bot.start_heartbeat(), bot.start_user_data(), bot.start_websocket())


async def main() -> None:
    argparser = argparse.ArgumentParser(prog='PassivbotLive', add_help=True,
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
        # Create a strategy based on the strategy module and the provided class
        strategy = getattr(strategy_module, config['strategy_class'])(strategy_config)
        config = LiveConfig(config['symbol'], config['user'], config['exchange'], config['leverage'],
                            config['call_interval'])
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
