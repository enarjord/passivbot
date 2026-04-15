from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

from cli_utils import help_requested


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
        "ohlcv_download",
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
    "hyperliquid-balance-probe": CommandSpec(
        "tools.probe_hyperliquid_balance",
        "read-only Hyperliquid balance smoke test",
    ),
    "hyperliquid-order-margin-probe": CommandSpec(
        "tools.probe_hyperliquid_order_margin",
        "mutating Hyperliquid order-margin diagnostic",
    ),
    "hyperliquid-position-probe": CommandSpec(
        "tools.probe_hyperliquid_position_balance",
        "mutating Hyperliquid position/balance diagnostic",
    ),
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
    "inspect-ohlcvs": CommandSpec(
        "tools.inspect_ohlcvs",
        "inspect v2 OHLCV cache metadata and gaps (requires full install)",
        requires_full=True,
    ),
    "migrate-historical-data": CommandSpec(
        "tools.migrate_historical_data",
        "migrate historical data layout (requires full install)",
        requires_full=True,
    ),
    "monitor-relay": CommandSpec(
        "tools.monitor_relay",
        "serve monitor snapshots and live streams (requires full install)",
        requires_full=True,
    ),
    "monitor-dev": CommandSpec(
        "tools.monitor_dev",
        "launch relay if needed and attach the terminal monitor (requires full install)",
        requires_full=True,
    ),
    "monitor-web": CommandSpec(
        "tools.monitor_web",
        "launch relay if needed and keep the web dashboard available (requires full install)",
        requires_full=True,
    ),
    "monitor-tui": CommandSpec(
        "tools.monitor_tui",
        "launch terminal monitor reader (requires full install)",
        requires_full=True,
    ),
    "pad-historical-daily": CommandSpec(
        "tools.pad_historical_daily",
        "pad missing daily historical data (requires full install)",
        requires_full=True,
    ),
    "pareto": CommandSpec(
        "tools.pareto_explorer",
        "select a single candidate from a Pareto front (requires full install)",
        requires_full=True,
    ),
    "pareto-dash": CommandSpec(
        "tools.pareto_dash",
        "launch Pareto dashboard (requires full install)",
        requires_full=True,
    ),
    "pareto-explorer": CommandSpec(
        "tools.pareto_explorer",
        "select a single candidate from a Pareto front (requires full install)",
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
    "pymoo",
    "psutil",
    "pyecharts",
    "requests",
}


FULL_INSTALL_MARKER_MODULES = tuple(sorted(FULL_INSTALL_MODULE_HINTS | {"websockets"}))
ENV_MISMATCH_IGNORE_ENV = "PASSIVBOT_IGNORE_ENV_MISMATCH"
ENV_REEXEC_GUARD_ENV = "PASSIVBOT_ENV_REEXEC"


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot",
        description="Passivbot unified CLI",
        epilog=(
            "Use 'passivbot <command> -h' for command-specific help.\n"
            "Base install supports live trading. Install passivbot with "
            "'python3 -m pip install -e \".[full]\"' for backtesting, optimization, downloader, "
            "and advanced tools."
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
            "Install passivbot with 'python3 -m pip install -e \".[full]\"' for tools marked as "
            "requiring the full install."
        ),
    )
    parser.add_argument("tool_name", nargs="?", help="Tool to run")
    return parser


def _restore_env_var(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _active_env_prefix() -> Path | None:
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        raw = os.environ.get(name)
        if raw:
            return _resolve_path(raw)
    return None


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.abspath(os.path.expanduser(os.fspath(value)))
    return Path(os.path.realpath(expanded))


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _current_interpreter_prefixes() -> tuple[Path, ...]:
    prefixes: list[Path] = []
    for value in (getattr(sys, "prefix", None), getattr(sys, "exec_prefix", None)):
        if not value:
            continue
        prefixes.append(_resolve_path(value))
    return tuple(dict.fromkeys(prefixes))


def _env_bin_dir(prefix: Path) -> Path:
    return prefix / ("Scripts" if os.name == "nt" else "bin")


def _expected_console_script(prefix: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _env_bin_dir(prefix) / f"passivbot{suffix}"


def _expected_python(prefix: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _env_bin_dir(prefix) / f"python{suffix}"


def _install_command_line(command: str, prefix: Path | None = None) -> str:
    if prefix is not None:
        return f"{_expected_python(prefix)} -m pip install -e {command}"
    return f'python3 -m pip install -e {command}'


def _install_guidance(prefix: Path | None = None) -> str:
    full = '".[full]"'
    dev = '".[dev]"'
    return (
        "Install Passivbot into the active environment with one of:\n"
        f"  {_install_command_line('.', prefix)}\n"
        f"  {_install_command_line(full, prefix)}\n"
        f"  {_install_command_line(dev, prefix)}\n"
    )


def _environment_mismatch_message(prefix: Path, actual_python: Path) -> str:
    script = _resolve_path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    expected_script = _expected_console_script(prefix)
    return (
        "passivbot detected an active environment mismatch.\n"
        f"  Active environment: {prefix}\n"
        f"  Running python:     {actual_python}\n"
        f"  Running script:     {script}\n"
        f"  Expected script:    {expected_script}\n"
        "This usually means your shell resolved a stale shim or a different install.\n\n"
        f"{_install_guidance(prefix)}"
        "After installing, reactivate the environment and refresh shell command lookup "
        "(for example: 'hash -r'; with zsh also run 'rehash').\n"
        f"Set {ENV_MISMATCH_IGNORE_ENV}=1 to bypass this check intentionally.\n"
    )


def _ensure_expected_environment() -> None:
    if os.environ.get(ENV_MISMATCH_IGNORE_ENV):
        return

    prefix = _active_env_prefix()
    if prefix is None:
        return

    actual_python = _resolve_path(sys.executable)
    if _path_is_within(actual_python, prefix):
        return
    if any(current_prefix == prefix for current_prefix in _current_interpreter_prefixes()):
        return

    script = _resolve_path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    expected_script = _expected_console_script(prefix)
    if script is not None and script == expected_script:
        return

    expected_python = _expected_python(prefix)
    if expected_script.exists() and expected_python.exists() and not os.environ.get(ENV_REEXEC_GUARD_ENV):
        os.environ[ENV_REEXEC_GUARD_ENV] = "1"
        os.execv(
            str(expected_python),
            [str(expected_python), str(expected_script), *sys.argv[1:]],
        )

    raise SystemExit(_environment_mismatch_message(prefix, actual_python))


def _full_install_message(prog_name: str, missing_module: str | None = None) -> str:
    detail = f" Missing dependency: {missing_module}." if missing_module else ""
    return (
        f"{prog_name} requires the full Passivbot install.{detail}\n"
        "Install it with:\n"
        '  python3 -m pip install -e ".[full]"\n'
    )


def _missing_full_install_markers() -> list[str]:
    return [name for name in FULL_INSTALL_MARKER_MODULES if importlib.util.find_spec(name) is None]


def _is_help_request(argv: list[str]) -> bool:
    return help_requested(argv)


def _invoke_module_main(module_name: str) -> tuple[bool, int]:
    module = importlib.import_module(module_name)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        return False, 0

    result = main_fn()
    if inspect.isawaitable(result):
        result = asyncio.run(result)

    if result is None:
        return True, 0
    if isinstance(result, int):
        return True, result
    return True, 0


def _run_module(module_name: str, prog_name: str, argv: list[str], requires_full: bool = False) -> int:
    if requires_full and not _is_help_request(argv):
        if _missing_full_install_markers():
            print(_full_install_message(prog_name), file=sys.stderr)
            return 2

    previous_argv = sys.argv[:]
    previous_prog = os.environ.get("PASSIVBOT_CLI_PROG")
    sys.argv = [prog_name, *argv]
    os.environ["PASSIVBOT_CLI_PROG"] = prog_name
    try:
        ran_main, exit_code = _invoke_module_main(module_name)
        if ran_main:
            return exit_code
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
    _ensure_expected_environment()
    raise SystemExit(main())
