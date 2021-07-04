import asyncio

from bots.binance import BinanceBot
from functions import load_base_config


async def start_bot(bot):
    await asyncio.gather(bot.start_heartbeat(), bot.start_user_data(), bot.start_websocket())


async def main() -> None:
    config = load_base_config('configs/test.hjson')
    bot = BinanceBot(config)
    await bot.init()
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())
