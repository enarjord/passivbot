import shutil
import os
import asyncio
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from downloader_gpt import OHLCVManager
from pure_funcs import symbol_to_coin
from procedures import make_get_filepath


async def copy_ohlcvs_from_old_dir():
    src = "historical_data"
    for d0 in os.listdir(src):
        if d0 == "ohlcvs_futures":
            # old binance dir
            om = OHLCVManager("binance", "2019-01-01", "")
            om.verbose = False
            await om.load_markets()
            for d1 in os.listdir(os.path.join(src, d0)):
                coin = symbol_to_coin(d1)
                if coin and om.has_coin(coin):
                    for d2 in os.listdir(os.path.join(src, d0, d1)):
                        if d2.endswith(".npy"):
                            dst = make_get_filepath(
                                os.path.join("historical_data", "ohlcvs_binanceusdm", coin, d2)
                            )
                            if not os.path.exists(dst):
                                print("copying...", os.path.join(src, d0, d1, d2), dst)
                                shutil.copy(os.path.join(src, d0, d1, d2), dst)
        elif d0 == "ohlcvs_bybit":
            om = OHLCVManager("bybit", "2019-01-01", "")
            om.verbose = False
            await om.load_markets()
            for d1 in os.listdir(os.path.join(src, d0)):
                coin = symbol_to_coin(d1)
                if coin and om.has_coin(coin):
                    for d2 in os.listdir(os.path.join(src, d0, d1)):
                        if d2.endswith(".npy"):
                            dst = make_get_filepath(
                                os.path.join("historical_data", "ohlcvs_bybit", coin, d2)
                            )
                            if not os.path.exists(dst):
                                print("copying...", os.path.join(src, d0, d1, d2), dst)
                                shutil.copy(os.path.join(src, d0, d1, d2), dst)


async def main():
    await copy_ohlcvs_from_old_dir()


if __name__ == "__main__":
    asyncio.run(main())
