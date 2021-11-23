from __future__ import annotations

import argparse
import logging
import os
import pathlib
from functools import partial

from pydantic import ValidationError

import passivbot.bot
import passivbot.config
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
        "-c",
        "--config",
        "--config-file",
        type=pathlib.Path,
        dest="config_files",
        default=[],
        action="append",
        help=(
            "Path to configuration file. Can be passed multiple times, but the last configuration file "
            "will be merged into the previous one, and so forth, overriding the previously defined values."
        ),
    )
    cli_logging_params = parser.add_argument_group(
        title="Logging", description="Runtime logging configuration"
    )
    cli_logging_params.add_argument(
        "--log-level",
        choices=passivbot.utils.logs.SORTED_LEVEL_NAMES,
        default=None,
        help="CLI logging level. Default: info",
    )
    cli_logging_params.add_argument(
        "--log-file", type=pathlib.Path, default=None, help="Path to logs file"
    )
    cli_logging_params.add_argument(
        "--log-file-level",
        choices=passivbot.utils.logs.SORTED_LEVEL_NAMES,
        default=None,
        help="Logs file logging level. Default: info",
    )
    runtime_params = parser.add_argument_group("Configuration Selection")
    runtime_params.add_argument(
        "-m",
        "--market",
        "--market-type",
        choices=("futures", "spot"),
        default="futures",
        help=("Select the market type to run against. Default: %default"),
    )
    runtime_params.add_argument(
        "-s",
        "--symbol",
        type=str,
        required=False,
        dest="symbol",
        default=None,
        help="The symbol to run, as specified under `symbols` on the configuration file",
    )
    runtime_params.add_argument(
        "-k",
        "--kn",
        "--key-name",
        default=None,
        dest="key_name",
        help=("The API key name as defined under `api_keys` on the configuration file."),
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

    if args.subparser == "live":
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == "backtest":
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == "downloader":
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == "optimize":
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == "batch-optimize":
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == "multi-symbol-optimize":
        config_cls = passivbot.config.LiveConfig
    else:
        parser.exit(
            status=1,
            message=(
                f"Don't know what to do regarding subparser {args.subparser}. Please fix this "
                "or file a bug report."
            ),
        )

    try:
        config: type[passivbot.config.ApiKeysConfigMixin] = config_cls.parse_files(  # type: ignore[assignment]
            *args.config_files
        )
    except ValidationError as exc:
        parser.exit(status=1, message=f"Found some errors in the configuration:\n\n{exc}\n")

    # Setup logging
    passivbot.utils.logs.setup_cli_logging(
        log_level=args.log_level or config.logging.cli.level,
        fmt=config.logging.cli.fmt,
        datefmt=config.logging.cli.datefmt,
    )
    if args.log_file or config.logging.file.path:
        passivbot.utils.logs.setup_logfile_logging(
            logfile=args.log_file or config.logging.file.path,
            log_level=args.log_file_level or config.logging.file.level,
            fmt=config.logging.file.fmt,
            datefmt=config.logging.file.datefmt,
        )

    if args.key_name not in config.api_keys:
        parser.exit(
            status=1,
            message=f"The API key name {args.key_name!r} cannot be found under `api_keys` on the configuration.",
        )

    # Set the config private attributes
    config._basedir = args.basedir
    config._market_type = args.market_type
    config._key = config.api_keys[args.key_name]

    if args.nojit:
        # Disable numba JIT compilation
        os.environ["NOJIT"] = "true"
        log.info("numba.njit compilation is disabled")
    else:
        log.info("numba.njit compilation is enabled")

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
