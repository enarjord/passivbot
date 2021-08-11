import argparse
import asyncio
from time import sleep

from helpers.downloader import Downloader
from helpers.loaders import load_config_files
from helpers.print_functions import print_


async def main():
    argparser = argparse.ArgumentParser(prog='Downloader', description='Download ticks from exchange API.')
    argparser.add_argument('-c', '--config', type=str, required=True, dest='c', help='Path to the config')
    args = argparser.parse_args()
    try:
        # Load the config
        config = load_config_files(args.c)
        print(config)
        config['session_name'] = 'some_session'
        config['caches_dirpath'] = f"backtests\\{config['exchange']}\\{config['symbol']}\\caches\\"
        downloader = Downloader(config)
        await downloader.download_ticks()
        await downloader.prepare_files(True)
        sleep(0.1)
    except Exception as e:
        print_(['Could not start', e])
        return


if __name__ == "__main__":
    asyncio.run(main())
