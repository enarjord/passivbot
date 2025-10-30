"""
Utility for inspecting and validating cached HLCVs datasets.

Examples
--------
Summarise every dataset located under the default cache directory::

    python -m src.tools.verify_hlcvs_data summarize

Summarise a specific dataset and return the output in JSON format::

    python -m src.tools.verify_hlcvs_data summarize caches/hlcvs_data/68153d270ccd83d4 --json

Compare two datasets and list each metric that differs::

    python -m src.tools.verify_hlcvs_data compare path/to/datasetA path/to/datasetB
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DEFAULT_ROOT = Path("caches/hlcvs_data")


@dataclass
class DatasetSummary:
    path: Path
    hlcvs_hash: str
    hlcvs_shape: Optional[tuple[int, ...]]
    hlcvs_nan_count: Optional[int]
    timestamps_hash: str
    timestamps_len: Optional[int]
    timestamp_start: Optional[int]
    timestamp_end: Optional[int]
    timestamp_monotonic: Optional[bool]
    btc_prices_hash: Optional[str]
    coins_count: Optional[int]

    def to_display_dict(self) -> Dict[str, Any]:
        display = asdict(self)
        display["path"] = str(self.path)
        return display


def iter_dataset_paths(paths: Iterable[Path], follow_root: bool) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        if path.is_dir() and (path / "hlcvs.npy.gz").exists():
            resolved.append(path)
        elif path.is_dir() and follow_root:
            for candidate in sorted(path.iterdir()):
                if candidate.is_dir() and (candidate / "hlcvs.npy.gz").exists():
                    resolved.append(candidate)
        else:
            raise FileNotFoundError(f"Dataset directory not found or invalid: {path}")
    return resolved


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def load_npy_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        return np.load(fh, allow_pickle=False)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _summarize_dataset(path: Path, skip_stats: bool) -> DatasetSummary:
    hlcvs_path = path / "hlcvs.npy.gz"
    timestamps_path = path / "timestamps.npy.gz"
    btc_prices_path = path / "btc_usd_prices.npy.gz"
    coins_path = path / "coins.json"

    if not hlcvs_path.exists():
        raise FileNotFoundError(f"{path} is missing hlcvs.npy.gz")
    if not timestamps_path.exists():
        raise FileNotFoundError(f"{path} is missing timestamps.npy.gz")

    hlcvs_hash = compute_sha256(hlcvs_path)
    timestamps_hash = compute_sha256(timestamps_path)
    btc_prices_hash = compute_sha256(btc_prices_path) if btc_prices_path.exists() else None

    if skip_stats:
        return DatasetSummary(
            path=path,
            hlcvs_hash=hlcvs_hash,
            hlcvs_shape=None,
            hlcvs_nan_count=None,
            timestamps_hash=timestamps_hash,
            timestamps_len=None,
            timestamp_start=None,
            timestamp_end=None,
            timestamp_monotonic=None,
            btc_prices_hash=btc_prices_hash,
            coins_count=None,
        )

    hlcvs = load_npy_gz(hlcvs_path)
    timestamps = load_npy_gz(timestamps_path)

    if timestamps.ndim != 1:
        raise ValueError(f"{timestamps_path} expected 1D array, got shape {timestamps.shape}")
    if hlcvs.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"{path}: hlcvs rows ({hlcvs.shape[0]}) != timestamps length ({timestamps.shape[0]})"
        )

    nan_count = int(np.isnan(hlcvs).sum())
    timestamps_len = int(timestamps.shape[0])
    timestamp_start = int(timestamps[0]) if timestamps_len else None
    timestamp_end = int(timestamps[-1]) if timestamps_len else None
    timestamp_monotonic = bool(np.all(np.diff(timestamps) >= 0))
    coins_meta = load_json(coins_path)
    coins_count = len(coins_meta) if isinstance(coins_meta, list) else None

    return DatasetSummary(
        path=path,
        hlcvs_hash=hlcvs_hash,
        hlcvs_shape=tuple(int(dim) for dim in hlcvs.shape),
        hlcvs_nan_count=nan_count,
        timestamps_hash=timestamps_hash,
        timestamps_len=timestamps_len,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        timestamp_monotonic=timestamp_monotonic,
        btc_prices_hash=btc_prices_hash,
        coins_count=coins_count,
    )


def summarize(paths: List[Path], fast: bool) -> List[DatasetSummary]:
    dataset_paths = iter_dataset_paths(paths, follow_root=True)
    summaries = []
    for dataset in dataset_paths:
        try:
            summaries.append(_summarize_dataset(dataset, skip_stats=fast))
        except Exception as exc:
            print(f"[ERROR] Failed to summarise {dataset}: {exc}")
    return summaries


def print_summaries(summaries: List[DatasetSummary], json_output: bool) -> None:
    if json_output:
        payload = [summary.to_display_dict() for summary in summaries]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not summaries:
        print("No datasets summarised.")
        return

    col_headers = [
        "path",
        "rows",
        "cols",
        "hlcvs_hash",
        "timestamps_hash",
        "btc_hash",
        "nan_count",
        "start_ts",
        "end_ts",
        "monotonic",
    ]
    print("\t".join(col_headers))
    for summary in summaries:
        rows = summary.hlcvs_shape[0] if summary.hlcvs_shape else "-"
        cols = summary.hlcvs_shape[1] if summary.hlcvs_shape and len(summary.hlcvs_shape) > 1 else "-"
        if summary.timestamp_monotonic is None:
            monotonic = "-"
        else:
            monotonic = "yes" if summary.timestamp_monotonic else "no"
        line = [
            str(summary.path),
            str(rows),
            str(cols),
            summary.hlcvs_hash[:16],
            summary.timestamps_hash[:16],
            summary.btc_prices_hash[:16] if summary.btc_prices_hash else "-",
            str(summary.hlcvs_nan_count) if summary.hlcvs_nan_count is not None else "-",
            str(summary.timestamp_start or "-"),
            str(summary.timestamp_end or "-"),
            monotonic,
        ]
        print("\t".join(line))


def compare(path_a: Path, path_b: Path, fast: bool) -> None:
    summaries = summarize([path_a, path_b], fast=fast)
    if len(summaries) != 2:
        print("Comparison failed: unable to summarise both datasets.")
        return
    a, b = summaries
    print(f"Comparing {a.path} ↔ {b.path}")
    differences = []
    for field in asdict(a):
        val_a = getattr(a, field)
        val_b = getattr(b, field)
        if val_a != val_b:
            differences.append((field, val_a, val_b))
    if not differences:
        print("Datasets are identical across all tracked metrics.")
        return
    for field, val_a, val_b in differences:
        print(f"- {field}: {val_a} ≠ {val_b}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and verify cached HLCVS datasets.")
    subparsers = parser.add_subparsers(dest="command")

    summary_parser = subparsers.add_parser("summarize", help="Summarise one or more datasets.")
    summary_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_ROOT],
        help="Dataset directories or a root directory containing dataset hashes.",
    )
    summary_parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip expensive statistics (only compute file hashes).",
    )
    summary_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary data as JSON.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two datasets.")
    compare_parser.add_argument("path_a", type=Path)
    compare_parser.add_argument("path_b", type=Path)
    compare_parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip expensive statistics (only compute file hashes).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "summarize"):
        paths = args.paths if hasattr(args, "paths") else [DEFAULT_ROOT]
        summaries = summarize(paths, fast=getattr(args, "fast", False))
        print_summaries(summaries, json_output=getattr(args, "json", False))
        return

    if args.command == "compare":
        compare(args.path_a, args.path_b, fast=args.fast)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
