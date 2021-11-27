from __future__ import annotations

import argparse
import logging
import os
import pathlib
import traceback
from functools import partial

from pydantic import ValidationError

import passivbot.bot
import passivbot.utils.logs
import passivbot.utils.procedures
from passivbot.datastructures.config import LiveConfig
from passivbot.datastructures.config import SymbolConfig
from passivbot.exceptions import PassivBotSystemExit
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
        dest="market_type",
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
    live_parser = subparsers.add_parser("live", help="Run PassivBot Live")
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

    passivbot.bot.setup_parser(live_parser)
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

    try:
        if args.subparser == "live":
            passivbot.bot.process_argparse_parsed_args(parser, args)
        elif BACKTEST_REQUIREMENTS_MISSING is False:
            if args.subparser == "backtest":
                passivbot.backtest.process_argparse_parsed_args(parser, args)
            elif args.subparser == "downloader":
                passivbot.downloader.process_argparse_parsed_args(parser, args)
            elif args.subparser == "optimize":
                passivbot.optimize.process_argparse_parsed_args(parser, args)
            elif args.subparser == "batch-optimize":
                passivbot.batch_optimize.process_argparse_parsed_args(parser, args)
            elif args.subparser == "multi-symbol-optimize":
                passivbot.multi_symbol_optimize.process_argparse_parsed_args(parser, args)
    except AttributeError:
        # process_argparse_parsed_args was not implemented
        pass

    for config_file in list(args.config_files):
        if not config_file.exists():
            log.warning("Config file %s does not exist", config_file)
            args.config_files.remove(config_file)

    if not args.config_files:
        default_config_file = args.basedir / "configs" / "live" / "default.json"
        if not default_config_file.exists():
            parser.exit(
                status=1,
                message=(
                    f"Please pass '--config=<path/to/config.json>' at least once or create {default_config_file}"
                ),
            )
        args.config_files.append(default_config_file)

    try:
        if args.subparser == "live":
            config = LiveConfig.parse_files(*args.config_files)
        elif args.subparser == "backtest":
            config = LiveConfig.parse_files(*args.config_files)
        elif args.subparser == "downloader":
            config = LiveConfig.parse_files(*args.config_files)
        elif args.subparser == "optimize":
            config = LiveConfig.parse_files(*args.config_files)
        elif args.subparser == "batch-optimize":
            config = LiveConfig.parse_files(*args.config_files)
        elif args.subparser == "multi-symbol-optimize":
            config = LiveConfig.parse_files(*args.config_files)
        else:
            parser.exit(
                status=1,
                message=(
                    f"Don't know what to do regarding subparser {args.subparser}. Please fix this "
                    "or file a bug report."
                ),
            )
    except ValidationError as exc:
        parser.exit(status=1, message=f"Found some errors in the configuration:\n\n{exc}\n")
    except Exception:
        parser.exit(
            status=1, message=f"Failed to load the configuration:\n{traceback.format_exc()}"
        )

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

    log.info("Configuration loaded from:")
    for config_file in args.config_files:
        try:
            log.info("  - %s", config_file.relative_to(args.basedir))
        except ValueError:
            log.info("  - %s", config_file)

    if not args.symbol:
        if len(config.symbols) == 1:
            symbol_name = next(iter(config.symbols))
            log.info(
                "Defaulting to symbol %r since it's the only defined on the configuration files",
                symbol_name,
            )
            args.symbol = symbol_name
        else:
            parser.exit(
                status=1,
                message=(
                    "Since there are multiple symbols defined in the config file, "
                    "passing '--symbol' is required."
                ),
            )

    if not args.key_name:
        # If no key name was passed on the cli, default to the one defined on the config file
        # for the selected symbol
        args.key_name = config.symbols[args.symbol].key_name

    if args.key_name not in config.api_keys:
        parser.exit(
            status=1,
            message=f"The API key name {args.key_name!r} cannot be found under `api_keys` on the configuration.",
        )

    # Set the config private attributes
    config._basedir = args.basedir
    symbol: SymbolConfig = config.symbols[args.symbol]

    if args.key_name and symbol.key_name != args.key_name:
        log.info(
            "Overriding the defined key name in the selected config, %r with the key named %r",
            symbol.key_name,
            args.key_name,
        )
        symbol.key_name = args.key_name

    config._active_config = config.configs[symbol.config_name]
    config.active_config._parent = config
    config.active_config._symbol = symbol
    config.active_config._api_key = config.api_keys[symbol.key_name]
    config.active_config._market_type = args.market_type

    if args.nojit:
        # Disable numba JIT compilation
        os.environ["NOJIT"] = "true"
        log.info("numba.njit compilation is disabled")
    else:
        log.info("numba.njit compilation is enabled")

    log.info(
        "Selected configuration for symbol %r:\n%s",
        config.active_config.symbol,
        config.active_config.json(indent=2),
    )

    if args.subparser == "live":
        passivbot.bot.post_process_argparse_parsed_args(parser, args, config.active_config)
    elif BACKTEST_REQUIREMENTS_MISSING is False:
        if args.subparser == "backtest":
            passivbot.backtest.post_process_argparse_parsed_args(parser, args, config.active_config)
        elif args.subparser == "downloader":
            passivbot.downloader.post_process_argparse_parsed_args(
                parser, args, config.active_config
            )
        elif args.subparser == "optimize":
            passivbot.optimize.post_process_argparse_parsed_args(parser, args, config.active_config)
        elif args.subparser == "batch-optimize":
            passivbot.batch_optimize.post_process_argparse_parsed_args(
                parser, args, config.active_config
            )
        elif args.subparser == "multi-symbol-optimize":
            passivbot.multi_symbol_optimize.post_process_argparse_parsed_args(
                parser, args, config.active_config
            )

    # Call the right sub-parser
    try:
        args.func(config.active_config)
    except PassivBotSystemExit as exc:
        parser.exit(status=1, message=f"Error: {exc}")
