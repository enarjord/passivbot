import argparse
import os
import pathlib

import passivbot.backtest
import passivbot.batch_optimize
import passivbot.bot
import passivbot.downloader
import passivbot.multi_symbol_optimize
import passivbot.optimize
import passivbot.utils.logs
from passivbot.version import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="passivbot", description="PassivBot Crypto Trading")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--nojit", help="disable numba", action="store_true")
    cli_logging_params = parser.add_argument_group(
        title="Logging", description="Runtime logging configuration"
    )
    cli_logging_params.add_argument(
        "--log-level",
        choices=sorted(passivbot.utils.logs.LOG_LEVELS),
        default="warning",
        help="CLI logging level. Default: %(default)s",
    )
    cli_logging_params.add_argument(
        "--log-file", type=pathlib.Path, default=None, help="Path to logs file"
    )
    cli_logging_params.add_argument(
        "--log-file-level",
        choices=sorted(passivbot.utils.logs.LOG_LEVELS),
        default="warning",
        help="Logs file logging level. Default: %(default)s",
    )
    subparsers = parser.add_subparsers(title="PassivBot commands")
    passivbot.bot.setup_parser(subparsers)
    passivbot.backtest.setup_parser(subparsers)
    passivbot.downloader.setup_parser(subparsers)
    passivbot.optimize.setup_parser(subparsers)
    passivbot.batch_optimize.setup_parser(subparsers)
    passivbot.multi_symbol_optimize.setup_parser(subparsers)

    # Parse the CLI arguments
    args: argparse.Namespace = parser.parse_args()

    # Setup logging
    passivbot.utils.logs.setup_cli_logging(args.log_level)
    if args.log_file:
        passivbot.utils.logs.setup_logfile_logging(args.log_file, log_level=args.log_file_level)

    if args.nojit:
        # Disable numba JIT compilation
        os.environ["NOJIT"] = "true"
        print("numba.njit compilation is disabled")

    # Call the right sub-parser
    args.func(args)
