from __future__ import annotations

import argparse
import logging
import os
import pathlib
from functools import partial

import passivbot.bot
import passivbot.utils.logs
import passivbot.utils.procedures
from passivbot.version import __version__

try:
    import passivbot.backtest
    import passivbot.batch_optimize
    import passivbot.downloader
    import passivbot.multi_symbol_optimize
    import passivbot.optimize

    BACKTEST_REQUIREMENTS_MISSING = False
except ImportError:
    BACKTEST_REQUIREMENTS_MISSING = True

BACKTEST_REQUIREMENTS_MISSING_ERROR = "ATTENTION!!!! Backtesting requirements are missing."

log = logging.getLogger(__name__)


def missing_backtest_requirements(parser, args):
    parser.exit(status=2, message=BACKTEST_REQUIREMENTS_MISSING_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(prog="passivbot", description="PassivBot Crypto Trading")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--nojit", help="disable numba", action="store_true")
    config_section = parser.add_argument_group(title="Configuration Paths")
    config_section.add_argument(
        "--basedir",
        type=pathlib.Path,
        default=None,
        help=(
            "The base directory where all paths will be computed from. "
            "Defaults to the current directory"
        ),
    )
    config_section.add_argument(
        "--api-keys",
        type=pathlib.Path,
        help="Path to the `api-keys.json` file. Defaults to <basedir>/api-keys.json",
    )
    cli_logging_params = parser.add_argument_group(
        title="Logging", description="Runtime logging configuration"
    )
    cli_logging_params.add_argument(
        "--log-level",
        choices=sorted(passivbot.utils.logs.LOG_LEVELS),
        default="info",
        help="CLI logging level. Default: %(default)s",
    )
    cli_logging_params.add_argument(
        "--log-file", type=pathlib.Path, default=None, help="Path to logs file"
    )
    cli_logging_params.add_argument(
        "--log-file-level",
        choices=sorted(passivbot.utils.logs.LOG_LEVELS),
        default="info",
        help="Logs file logging level. Default: %(default)s",
    )
    subparsers = parser.add_subparsers(title="PassivBot commands", dest="subparser")
    passivbot.bot.setup_parser(subparsers)
    backtest_parser = subparsers.add_parser("backtest", help="Backtest given passivbot config.")
    downloader_parser = subparsers.add_parser(
        "downloader", help="Download ticks from exchange API."
    )
    optimize_parser = subparsers.add_parser("optimize", help="Optimize PassivBot config")
    batch_optimize_parser = subparsers.add_parser(
        "batch-optimize", help="Batch Optimize PassivBot config"
    )
    multi_symbol_optimize_parser = subparsers.add_parser(
        "multi-symbol-optimize", help="Optimize passivbot config multi symbol"
    )
    if BACKTEST_REQUIREMENTS_MISSING is False:
        passivbot.backtest.setup_parser(backtest_parser)
        passivbot.downloader.setup_parser(downloader_parser)
        passivbot.optimize.setup_parser(optimize_parser)
        passivbot.batch_optimize.setup_parser(batch_optimize_parser)
        passivbot.multi_symbol_optimize.setup_parser(multi_symbol_optimize_parser)
    else:
        for subparser in (
            backtest_parser,
            downloader_parser,
            optimize_parser,
            batch_optimize_parser,
            multi_symbol_optimize_parser,
        ):
            subparser.description = BACKTEST_REQUIREMENTS_MISSING_ERROR
            subparser.set_defaults(func=partial(missing_backtest_requirements, parser))

    # Parse the CLI arguments
    args: argparse.Namespace = parser.parse_args()

    if args.basedir is not None:
        args.basedir = args.basedir.resolve()
    else:
        args.basedir = pathlib.Path.cwd()

    # Setup logging
    passivbot.utils.logs.setup_cli_logging(args.log_level)
    if args.log_file:
        passivbot.utils.logs.setup_logfile_logging(args.log_file, log_level=args.log_file_level)

    if args.nojit:
        # Disable numba JIT compilation
        os.environ["NOJIT"] = "true"
        log.info("numba.njit compilation is disabled")
    else:
        log.info("numba.njit compilation is enabled")

    if not args.api_keys:
        args.api_keys = args.basedir / "api-keys.json"

    if args.subparser == "live":
        passivbot.bot.validate_argparse_parsed_args(parser, args)
    elif BACKTEST_REQUIREMENTS_MISSING is False:
        if args.subparser == "backtest":
            passivbot.backtest.validate_argparse_parsed_args(parser, args)
        elif args.subparser == "downloader":
            passivbot.downloader.validate_argparse_parsed_args(parser, args)
        elif args.subparser == "optimize":
            passivbot.optimize.validate_argparse_parsed_args(parser, args)
        elif args.subparser == "batch-optimize":
            passivbot.batch_optimize.validate_argparse_parsed_args(parser, args)
        elif args.subparser == "multi-symbol-optimize":
            passivbot.multi_symbol_optimize.validate_argparse_parsed_args(parser, args)

    # Call the right sub-parser
    args.func(args)
