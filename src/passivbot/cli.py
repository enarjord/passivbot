import argparse
import logging
import os

import passivbot.backtest
import passivbot.batch_optimize
import passivbot.bot
import passivbot.downloader
import passivbot.multi_symbol_optimize
import passivbot.optimize
from passivbot.version import __version__

logging.getLogger("telegram").setLevel(logging.CRITICAL)


def main() -> None:
    parser = argparse.ArgumentParser(prog="passivbot", description="PassivBot Crypto Trading")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--nojit", help="disable numba", action="store_true")
    subparsers = parser.add_subparsers(title="PassivBot commands")
    passivbot.bot.setup_parser(subparsers)
    passivbot.backtest.setup_parser(subparsers)
    passivbot.downloader.setup_parser(subparsers)
    passivbot.optimize.setup_parser(subparsers)
    passivbot.batch_optimize.setup_parser(subparsers)
    passivbot.multi_symbol_optimize.setup_parser(subparsers)

    # Parse the CLI arguments
    args: argparse.Namespace = parser.parse_args()

    if args.nojit:
        # Disable numba JIT compilation
        os.environ["NOJIT"] = "true"
        print("numba.njit compilation is disabled")

    # Call the right sub-parser
    args.func(args)
