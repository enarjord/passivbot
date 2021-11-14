import argparse
import os
import pathlib

from pydantic import ValidationError

import passivbot.backtest
import passivbot.batch_optimize
import passivbot.bot
import passivbot.config
import passivbot.downloader
import passivbot.multi_symbol_optimize
import passivbot.optimize
import passivbot.utils.logs
import passivbot.utils.procedures
from passivbot.version import __version__


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
        choices=passivbot.utils.logs.SORTED_LEVEL_NAMES,
        default=None,
        help="CLI logging level. Default: warning",
    )
    cli_logging_params.add_argument(
        "--log-file", type=pathlib.Path, default=None, help="Path to logs file"
    )
    cli_logging_params.add_argument(
        "--log-file-level",
        choices=passivbot.utils.logs.SORTED_LEVEL_NAMES,
        default=None,
        help="Logs file logging level. Default: warning",
    )
    subparsers = parser.add_subparsers(title="PassivBot commands", dest="subparser")
    passivbot.bot.setup_parser(subparsers)
    passivbot.backtest.setup_parser(subparsers)
    passivbot.downloader.setup_parser(subparsers)
    passivbot.optimize.setup_parser(subparsers)
    passivbot.batch_optimize.setup_parser(subparsers)
    passivbot.multi_symbol_optimize.setup_parser(subparsers)

    # Parse the CLI arguments
    args: argparse.Namespace = parser.parse_args()

    if args.basedir is not None:
        args.basedir = args.basedir.resolve()
    else:
        args.basedir = pathlib.Path.cwd()

    if args.subparser == passivbot.bot.SUBPARSER_NAME:
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == passivbot.backtest.SUBPARSER_NAME:
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == passivbot.downloader.SUBPARSER_NAME:
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == passivbot.optimize.SUBPARSER_NAME:
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == passivbot.batch_optimize.SUBPARSER_NAME:
        config_cls = passivbot.config.LiveConfig
    elif args.subparser == passivbot.multi_symbol_optimize.SUBPARSER_NAME:
        config_cls = passivbot.config.LiveConfig

    try:
        config = config_cls.parse_files(*args.config_files)
    except ValidationError as exc:
        parser.exit(status=1, message=f"Found some errors in the configuration:\n\n{exc}\n")

    # Set the config private attributes
    config._basedir = args.basedir  # type: ignore[misc]

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

    if args.nojit:
        # Disable numba JIT compilation
        os.environ["NOJIT"] = "true"
        print("numba.njit compilation is disabled")

    if not args.api_keys:
        args.api_keys = args.basedir / "api-keys.json"

    if args.subparser == passivbot.bot.SUBPARSER_NAME:
        passivbot.bot.validate_argparse_parsed_args(parser, args)
    elif args.subparser == passivbot.backtest.SUBPARSER_NAME:
        passivbot.backtest.validate_argparse_parsed_args(parser, args)
    elif args.subparser == passivbot.downloader.SUBPARSER_NAME:
        passivbot.downloader.validate_argparse_parsed_args(parser, args)
    elif args.subparser == passivbot.optimize.SUBPARSER_NAME:
        passivbot.optimize.validate_argparse_parsed_args(parser, args)
    elif args.subparser == passivbot.batch_optimize.SUBPARSER_NAME:
        passivbot.batch_optimize.validate_argparse_parsed_args(parser, args)
    elif args.subparser == passivbot.multi_symbol_optimize.SUBPARSER_NAME:
        passivbot.multi_symbol_optimize.validate_argparse_parsed_args(parser, args)

    # Call the right sub-parser
    args.func(args)
