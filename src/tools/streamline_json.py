#!/usr/bin/env python3
"""
Rewrites JSON files using utils.dump_json_streamlined for more compact formatting.

Usage
-----
    python -m src.tools.streamline_json path/to/file_or_directory [options]

Options
-------
    --indent INT            Base indentation level (default: 4)
    --max-inline INT        Maximum inline container length (default: 72)
    --separators ", :"      Comma/colon separators passed to json dumps (default: ", :")
    --sort-keys             Sort dictionary keys before writing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import dump_json_streamlined


def iter_json_files(target: Path):
    if target.is_file() and target.suffix == ".json":
        yield target
        return
    if target.is_dir():
        for path in sorted(target.rglob("*.json")):
            if path.is_file():
                yield path
        return
    raise FileNotFoundError(f"Target '{target}' is neither a JSON file nor directory.")


def process_file(
    path: Path,
    *,
    indent: int,
    max_inline: int,
    separators: tuple[str, str],
    sort_keys: bool,
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    with path.open("w", encoding="utf-8") as fh:
        dump_json_streamlined(
            data,
            fh,
            indent=indent,
            max_inline=max_inline,
            separators=separators,
            sort_keys=sort_keys,
        )


def parse_separators(raw: str) -> tuple[str, str]:
    value = raw.strip()
    if not value:
        return ",", ":"

    if value == ",:":
        return ",", ":"

    pieces = value.split()
    if len(pieces) == 2 and all(len(piece) == 1 for piece in pieces):
        return pieces[0], pieces[1]

    if len(value) >= 2 and "," not in value:
        return value[0], value[-1]

    parts = value.split(",")
    if len(parts) == 2 and parts[0] and parts[1].strip():
        return parts[0], parts[1].strip()

    raise argparse.ArgumentTypeError("Invalid separators format. Examples: ', :', ',:' or ' , : '")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "target",
        help="Path to a JSON file or directory containing JSON files.",
    )
    parser.add_argument("--indent", type=int, default=4, help="Indentation level (default: 4).")
    parser.add_argument(
        "--max-inline",
        type=int,
        default=72,
        help="Maximum inline container length (default: 72).",
    )
    parser.add_argument(
        "--separators",
        default=", :",
        help="Comma/colon separators, e.g. ', :' or ',:' (default: ', :').",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort dictionary keys before writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target = Path(args.target).expanduser()
    separators = parse_separators(args.separators)

    count = 0
    for file_path in iter_json_files(target):
        process_file(
            file_path,
            indent=args.indent,
            max_inline=args.max_inline,
            separators=separators,
            sort_keys=args.sort_keys,
        )
        print(f"Formatted {file_path}")
        count += 1
    if count == 0:
        print("No JSON files processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
