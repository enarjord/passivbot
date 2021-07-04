import argparse
import asyncio

from functions import load_base_config, print_


async def start_bot(bot):
    await asyncio.gather(bot.start_heartbeat(), bot.start_user_data(), bot.start_websocket())


async def main() -> None:
    argparser = argparse.ArgumentParser(prog='GridTrader', add_help=True,
                                        description='Grid trading bot with fixed grid.')
    argparser.add_argument('-c', '--config', type=str, required=True, dest='c', help='Path to the config')
    args = argparser.parse_args()
    try:
        config = load_base_config(args.c)
        if config['exchange'] == 'binance':
            from bots.binance import BinanceBot
            bot = BinanceBot(config)
        else:
            print_(['Exchange not supported.'], n=True)
            return
        await bot.init()
        await start_bot(bot)
    except Exception as e:
        print_(['Could not start', e])
        return


if __name__ == '__main__':
    asyncio.run(main())
