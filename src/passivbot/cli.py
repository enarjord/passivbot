import argparse
import logging

import passivbot.backtest
import passivbot.batch_optimize
import passivbot.bot
import passivbot.downloader
import passivbot.multi_symbol_optimize
import passivbot.optimize
import passivbot.pso
import passivbot.pso_custom
from passivbot.version import __version__

logging.getLogger("telegram").setLevel(logging.CRITICAL)


def main() -> None:
    parser = argparse.ArgumentParser(prog="passivbot", description="PassivBot Crypto Trading")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(title="PassivBot commands")
    passivbot.bot.setup_parser(subparsers)
    passivbot.backtest.setup_parser(subparsers)
    passivbot.downloader.setup_parser(subparsers)
    passivbot.optimize.setup_parser(subparsers)
    passivbot.batch_optimize.setup_parser(subparsers)
    passivbot.multi_symbol_optimize.setup_parser(subparsers)
    passivbot.pso.setup_parser(subparsers)
    passivbot.pso_custom.setup_parser(subparsers)

    parser.parse_args()
