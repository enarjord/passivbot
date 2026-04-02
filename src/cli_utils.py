import argparse
import os


def get_cli_prog(default: str) -> str:
    override = os.environ.get("PASSIVBOT_CLI_PROG")
    if not override:
        return default
    override = override.strip()
    return override or default


def help_all_requested(argv: list[str]) -> bool:
    return "--help-all" in argv


def expand_help_all_argv(argv: list[str]) -> list[str]:
    if "--help-all" not in argv:
        return argv
    if any(arg in {"-h", "--help"} for arg in argv):
        return argv
    return [*argv, "--help"]


def build_command_parser(
    *,
    prog: str,
    description: str,
    usage: str,
    epilog: str,
) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        usage=usage,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_help_all_argument(
    parser: argparse.ArgumentParser,
    *,
    help_all: bool,
    help_text: str = "Show all config override flags, including advanced options.",
) -> None:
    parser.add_argument(
        "--help-all",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS if help_all else help_text,
    )
