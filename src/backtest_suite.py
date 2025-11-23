#!/usr/bin/env python3
"""Compatibility wrapper for the backtest suite runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from suite_runner import cli_entrypoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest suite.")
    parser.add_argument(
        "config_path",
        type=Path,
        nargs="?",
        default=Path("configs/template.json"),
        help="Path to the base Passivbot config.",
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=None,
        help="Optional path to a file containing backtest.suite overrides.",
    )
    args = parser.parse_args()
    cli_entrypoint(str(args.config_path), str(args.suite_config) if args.suite_config else None)


if __name__ == "__main__":
    main()
