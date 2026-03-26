from __future__ import annotations

import argparse
import os
import runpy
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    module: str
    summary: str
    requires_full: bool = False


CORE_COMMANDS: dict[str, CommandSpec] = {
    "live": CommandSpec("main", "run the live trading bot"),
    "backtest": CommandSpec(
        "backtest",
        "run historical backtests (requires full install)",
        requires_full=True,
    ),
    "optimize": CommandSpec(
        "optimize",
        "run the optimizer (requires full install)",
        requires_full=True,
    ),
    "download": CommandSpec(
        "downloader",
        "download OHLCV data (requires full install)",
        requires_full=True,
    ),
}

TOOL_COMMANDS: dict[str, CommandSpec] = {
    "candle-doctor": CommandSpec(
        "tools.candle_doctor",
        "audit candle caches (requires full install)",
        requires_full=True,
    ),
    "fetch-balance": CommandSpec("tools.fetch_balance", "fetch exchange balances"),
    "fill-events-dash": CommandSpec(
        "tools.fill_events_dash",
        "launch fill events dashboard (requires full install)",
        requires_full=True,
    ),
    "fill-events-doctor": CommandSpec(
        "tools.fill_events_doctor",
        "audit fill-events cache (requires full install)",
        requires_full=True,
    ),
    "generate-mcap-list": CommandSpec(
        "tools.generate_mcap_list",
        "generate approved-coin lists by market cap (requires full install)",
        requires_full=True,
    ),
    "iterative-backtester": CommandSpec(
        "tools.iterative_backtester",
        "launch interactive iterative backtester (requires full install)",
        requires_full=True,
    ),
    "iterative-history-plot": CommandSpec(
        "tools.iterative_history_plot",
        "plot iterative history files (requires full install)",
        requires_full=True,
    ),
    "migrate-historical-data": CommandSpec(
        "tools.migrate_historical_data",
        "migrate historical data layout (requires full install)",
        requires_full=True,
    ),
    "pad-historical-daily": CommandSpec(
        "tools.pad_historical_daily",
        "pad missing daily historical data (requires full install)",
        requires_full=True,
    ),
    "pareto-dash": CommandSpec(
        "tools.pareto_dash",
        "launch Pareto dashboard (requires full install)",
        requires_full=True,
    ),
    "pareto-transform": CommandSpec(
        "tools.pareto_transform",
        "transform Pareto result data (requires full install)",
        requires_full=True,
    ),
    "streamline-json": CommandSpec("tools.streamline_json", "reformat config or result JSON"),
    "verify-hlcvs-data": CommandSpec(
        "tools.verify_hlcvs_data",
        "verify cached OHLCV datasets (requires full install)",
        requires_full=True,
    ),
}

FULL_INSTALL_MODULE_HINTS = {
    "aiohttp",
    "colorama",
    "dash",
    "dash_bootstrap_components",
    "deap",
    "dictdiffer",
    "matplotlib",
    "msgpack",
    "plotly",
    "psutil",
    "pyecharts",
    "requests",
}


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot",
        description="Passivbot unified CLI",
        epilog=(
            "Use 'passivbot <command> -h' for command-specific help.\n"
            "Base install supports live trading. Install passivbot with 'pip install -e \".[full]\"' "
            "for backtesting, optimization, downloader, and advanced tools."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for name, spec in CORE_COMMANDS.items():
        subparsers.add_parser(name, help=spec.summary)
    subparsers.add_parser("tool", help="run auxiliary tools (some require full install)")
    return parser


def _build_tool_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool",
        description="Run auxiliary Passivbot tools",
        epilog=(
            "Use 'passivbot tool <tool> -h' for tool-specific help.\n"
            "Install passivbot with 'pip install -e \".[full]\"' for tools marked as requiring the full install."
        ),
    )
    parser.add_argument("tool_name", nargs="?", help="Tool to run")
    return parser


def _restore_env_var(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _full_install_message(prog_name: str, missing_module: str | None = None) -> str:
    detail = f" Missing dependency: {missing_module}." if missing_module else ""
    return (
        f"{prog_name} requires the full Passivbot install.{detail}\n"
        "Install it with:\n"
        '  pip install -e ".[full]"\n'
    )


def _run_module(module_name: str, prog_name: str, argv: list[str], requires_full: bool = False) -> int:
    previous_argv = sys.argv[:]
    previous_prog = os.environ.get("PASSIVBOT_CLI_PROG")
    sys.argv = [prog_name, *argv]
    os.environ["PASSIVBOT_CLI_PROG"] = prog_name
    try:
        runpy.run_module(module_name, run_name="__main__")
    except ModuleNotFoundError as exc:
        if requires_full and exc.name and exc.name.split(".", 1)[0] in FULL_INSTALL_MODULE_HINTS:
            print(_full_install_message(prog_name, exc.name), file=sys.stderr)
            return 2
        raise
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        raise
    finally:
        sys.argv = previous_argv
        _restore_env_var("PASSIVBOT_CLI_PROG", previous_prog)
    return 0


def _dispatch_tool(argv: list[str]) -> int:
    parser = _build_tool_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        print("\nAvailable tools:")
        for name in sorted(TOOL_COMMANDS):
            print(f"  {name:<24} {TOOL_COMMANDS[name].summary}")
        return 0

    tool_name = argv[0]
    spec = TOOL_COMMANDS.get(tool_name)
    if spec is None:
        parser.exit(2, f"passivbot tool: unknown tool {tool_name!r}\n")
    return _run_module(
        spec.module,
        f"passivbot tool {tool_name}",
        argv[1:],
        requires_full=spec.requires_full,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_root_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    if argv[0] == "help":
        if len(argv) == 1:
            parser.print_help()
            return 0
        return main([argv[1], "-h", *argv[2:]])

    command = argv[0]
    if command == "tool":
        return _dispatch_tool(argv[1:])

    spec = CORE_COMMANDS.get(command)
    if spec is None:
        parser.exit(2, f"passivbot: unknown command {command!r}\n")
    return _run_module(
        spec.module,
        f"passivbot {command}",
        argv[1:],
        requires_full=spec.requires_full,
    )


def console_main() -> None:
    raise SystemExit(main())
