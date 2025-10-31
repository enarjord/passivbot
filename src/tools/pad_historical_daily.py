"""
Utility to canonicalize daily OHLCV `.npy` files by padding missing minutes.

Usage
-----
    python -m src.tools.pad_historical_daily
    python -m src.tools.pad_historical_daily historical_data/ohlcvs_binanceusdm --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable, List

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import downloader as dl


def _iter_daily_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.npy")):
        try:
            date.fromisoformat(path.stem)
        except ValueError:
            continue
        yield path


def _process_file(path: Path, dry_run: bool) -> str | None:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return f"SKIP {path}: unexpected shape {arr.shape}"

    rows = arr.shape[0]
    if rows == 1440:
        return None

    try:
        start_ts = dl.date_to_ts(path.stem)
    except Exception as exc:
        return f"SKIP {path}: cannot parse date ({exc})"

    if dry_run:
        return f"DRYRUN {path}: would pad from {rows} to 1440 rows"

    try:
        dl.dump_daily_ohlcv_data(arr, str(path), start_ts)
    except ValueError as exc:
        return f"SKIP {path}: {exc}"
    return f"FIXED {path}: padded from {rows} to 1440 rows"


def _print_progress(root: Path, idx: int, total: int) -> None:
    message = f"\rScanning {root} [{idx}/{total}]"
    sys.stdout.write(message)
    sys.stdout.flush()


def canonicalize_roots(roots: List[Path], dry_run: bool) -> None:
    messages: List[str] = []
    for root in roots:
        if not root.exists():
            messages.append(f"[WARN] {root} does not exist; skipping.")
            continue
        files = list(_iter_daily_files(root))
        total = len(files)
        if not total:
            continue
        for idx, path in enumerate(files, start=1):
            _print_progress(root, idx, total)
            result = _process_file(path, dry_run=dry_run)
            if result:
                messages.append(result)
        sys.stdout.write("\n")

    if not messages:
        print("No changes required.")
    else:
        for message in messages:
            print(message)
        if dry_run:
            print("Dry run complete; re-run without --dry-run to apply changes.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pad historical daily OHLCV files to 1440 minutes.",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("historical_data")],
        help="Root directories to scan (default: historical_data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would be modified without overwriting them.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    canonicalize_roots(args.roots, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
