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
import io
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm.auto import tqdm
from datetime import datetime, UTC


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


@dataclass
class HistoricalCoinSummary:
    exchange: str
    coin: str
    files: int
    rows: int
    nan_count: int
    first_ts: Optional[int]
    last_ts: Optional[int]
    per_file_rows_ok: bool
    monotonic_within_files: bool
    contiguous_across_files: bool
    data_hash: Optional[str]

    def to_display_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProblemFile:
    exchange: str
    coin: str
    file: Path
    rows: int
    reason: str


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
    parser = argparse.ArgumentParser(
        description="Inspect and verify cached HLCVS datasets.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """
            Commonly used flags:
              summarize                  --fast  --json
              compare                    --fast
              summarize-historical       --progress  --hash  --json
              clean-historical           --progress  --apply
              hash-historical            --progress  --output PATH
              compare-historical-hashes  --apply  --local-root PATH

            Tip: every sub-command also supports -h/--help for full details.
            """
        ).strip(),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    summary_parser = subparsers.add_parser(
        "summarize",
        help="Summarise one or more datasets (--fast for hashes only, --json for machine output).",
    )
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

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two cached HLCVS datasets (use --fast to skip full stats).",
    )
    compare_parser.add_argument("path_a", type=Path)
    compare_parser.add_argument("path_b", type=Path)
    compare_parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip expensive statistics (only compute file hashes).",
    )

    historical_parser = subparsers.add_parser(
        "summarize-historical",
        help="Summarise raw historical OHLCV files (--progress, --hash, --json).",
    )
    historical_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("historical_data/ohlcvs_binanceusdm")],
        help="Historical dataset directories (exchange roots).",
    )
    historical_parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute a hash across each coin's files (slower).",
    )
    historical_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output in JSON format.",
    )
    historical_parser.add_argument(
        "--fast",
        action="store_true",
        help="Placeholder flag for parity; currently still loads arrays to compute statistics.",
    )
    historical_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while scanning coins.",
    )

    clean_parser = subparsers.add_parser(
        "clean-historical",
        help="Scan/remove problematic historical OHLCV files (--progress, --apply).",
    )
    clean_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("historical_data/ohlcvs_binanceusdm")],
        help="Historical dataset directories (exchange roots).",
    )
    clean_parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files instead of only reporting them.",
    )
    clean_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while scanning coins.",
    )

    hash_parser = subparsers.add_parser(
        "hash-historical",
        help="Compute float32 hashes for historical files (--progress, --output).",
    )
    hash_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("historical_data")],
        help="Root directories to scan (e.g. historical_data).",
    )
    hash_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON manifest.",
    )
    hash_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress while hashing files.",
    )

    compare_hash_parser = subparsers.add_parser(
        "compare-historical-hashes",
        help="Compare two hash manifests (--apply deletes mismatched local files).",
    )
    compare_hash_parser.add_argument("manifest_a", type=Path)
    compare_hash_parser.add_argument("manifest_b", type=Path)
    compare_hash_parser.add_argument(
        "--local-root",
        type=Path,
        help="Local historical_data root for deleting mismatched files when --apply is set.",
    )
    compare_hash_parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete local files that differ between manifests.",
    )

    return parser


def _summarize_coin_dir(
    exchange: str,
    coin_dir: Path,
    fast: bool,
    include_hash: bool,
) -> HistoricalCoinSummary:
    files = sorted([p for p in coin_dir.iterdir() if p.suffix == ".npy" and p.is_file()])
    total_rows = 0
    nan_count = 0
    per_file_rows_ok = True
    monotonic_within = True
    contiguous = True
    first_ts = None
    last_ts = None
    prev_last_ts = None
    digest = hashlib.sha256() if include_hash else None
    minute_ms = 60_000

    for file in files:
        data = file.read_bytes()
        if digest is not None:
            digest.update(data)
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ValueError(f"Unexpected array shape {arr.shape} in {file}")
        ts = arr[:, 0].astype(np.int64)
        if ts.size == 0:
            continue
        total_rows += ts.size
        nan_count += int(np.isnan(arr).sum())
        if arr.shape[0] != 1440:
            per_file_rows_ok = False
        diffs = np.diff(ts)
        if diffs.size and not np.all(diffs == minute_ms):
            monotonic_within = False
        if prev_last_ts is not None and ts[0] != prev_last_ts + minute_ms:
            contiguous = False
        prev_last_ts = ts[-1]
        if first_ts is None:
            first_ts = int(ts[0])
        last_ts = int(ts[-1])

    data_hash = digest.hexdigest() if digest is not None else None

    return HistoricalCoinSummary(
        exchange=exchange,
        coin=coin_dir.name,
        files=len(files),
        rows=total_rows,
        nan_count=nan_count,
        first_ts=first_ts,
        last_ts=last_ts,
        per_file_rows_ok=per_file_rows_ok,
        monotonic_within_files=monotonic_within,
        contiguous_across_files=contiguous,
        data_hash=data_hash,
    )


def _scan_problem_files(
    exchange: str,
    coin_dir: Path,
    show_progress: bool,
) -> List[ProblemFile]:
    files = sorted([p for p in coin_dir.iterdir() if p.suffix == ".npy" and p.is_file()])
    problems: List[ProblemFile] = []
    minute_ms = 60_000
    iterator = files if not show_progress else tqdm(files, desc=f"{exchange}/{coin_dir.name}")
    for file in iterator:
        try:
            arr = np.load(file, allow_pickle=False, mmap_mode="r")
        except Exception as exc:
            problems.append(
                ProblemFile(exchange, coin_dir.name, file, rows=0, reason=f"failed to load: {exc}")
            )
            continue
        reasons: List[str] = []
        rows = int(arr.shape[0]) if arr.ndim >= 1 else 0
        if rows != 1440:
            reasons.append(f"rows={rows}")
        if rows > 1:
            ts = arr[:, 0].astype(np.int64)
            diffs = np.diff(ts)
            if np.any(diffs <= 0):
                reasons.append("non-monotonic ts")
            elif np.any(diffs % minute_ms != 0):
                reasons.append("irregular spacing")
        if reasons:
            problems.append(
                ProblemFile(exchange, coin_dir.name, file, rows=rows, reason=", ".join(reasons))
            )
    return problems


def summarize_historical(
    paths: List[Path],
    fast: bool,
    include_hash: bool,
    show_progress: bool,
) -> List[HistoricalCoinSummary]:
    summaries: List[HistoricalCoinSummary] = []
    for root in paths:
        if not root.is_dir():
            raise FileNotFoundError(f"Historical data directory not found: {root}")
        exchange = root.name
        coin_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        iterator = tqdm(coin_dirs, desc=f"{exchange}", disable=not show_progress)
        for coin_dir in iterator:
            try:
                summaries.append(
                    _summarize_coin_dir(exchange, coin_dir, fast=fast, include_hash=include_hash)
                )
            except Exception as exc:
                print(f"[ERROR] Failed to summarise {coin_dir}: {exc}")
    return summaries


def clean_historical(
    paths: List[Path],
    apply_changes: bool,
    show_progress: bool,
) -> List[ProblemFile]:
    all_problems: List[ProblemFile] = []
    for root in paths:
        if not root.is_dir():
            raise FileNotFoundError(f"Historical data directory not found: {root}")
        exchange = root.name
        coin_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        iterator = tqdm(coin_dirs, desc=f"{exchange}", disable=not show_progress)
        for coin_dir in iterator:
            problems = _scan_problem_files(exchange, coin_dir, show_progress=False)
            for problem in problems:
                action = "REMOVED" if apply_changes else "FOUND"
                print(
                    f"[{action}] {problem.exchange}/{problem.coin}/{problem.file.name} -> {problem.reason}"
                )
                if apply_changes:
                    try:
                        problem.file.unlink(missing_ok=True)
                    except Exception as exc:
                        print(f"[ERROR] failed to delete {problem.file}: {exc}")
                all_problems.append(problem)
    if apply_changes:
        print(f"Deleted {len(all_problems)} files in total.")
    else:
        print(f"Identified {len(all_problems)} problematic files (dry run).")
    return all_problems


def hash_historical(paths: List[Path], show_progress: bool) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for root in paths:
        if not root.is_dir():
            raise FileNotFoundError(f"Historical data directory not found: {root}")
        files = sorted(root.rglob("*.npy"))
        iterator = tqdm(files, desc=f"hash {root.name}", disable=not show_progress)
        for file in iterator:
            rel_path = file.relative_to(root)
            digest = hashlib.sha256()
            arr = np.load(file, allow_pickle=False, mmap_mode="r")
            data = arr.astype(np.float32, copy=False).tobytes()
            digest.update(data)
            key = str(rel_path).replace("\\", "/")
            mapping[key] = digest.hexdigest()
    return mapping


def save_hash_manifest(output: Path, roots: List[Path], mapping: Dict[str, str]) -> None:
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "roots": [str(p.resolve()) for p in roots],
        "files": mapping,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    print(f"Wrote {len(mapping)} hashes to {output}")


def load_hash_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "files" not in data or not isinstance(data["files"], dict):
        raise ValueError(f"Invalid manifest format: {path}")
    return data


def compare_historical_hashes(
    manifest_a: Path,
    manifest_b: Path,
    local_root: Optional[Path],
    apply_changes: bool,
) -> None:
    data_a = load_hash_manifest(manifest_a)
    data_b = load_hash_manifest(manifest_b)
    files_a = data_a["files"]
    files_b = data_b["files"]

    keys_a = set(files_a.keys())
    keys_b = set(files_b.keys())
    common = keys_a & keys_b
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    mismatched = sorted([key for key in common if files_a[key] != files_b[key]])

    print(f"Common files with differing hashes: {len(mismatched)}")
    for key in mismatched:
        print(f"DIFF {key}")
    if only_a:
        print(f"Files only in {manifest_a}: {len(only_a)}")
    if only_b:
        print(f"Files only in {manifest_b}: {len(only_b)}")

    if apply_changes and mismatched:
        if local_root is None:
            raise ValueError("--local-root is required when --apply is used")
        deleted = 0
        for key in mismatched:
            target = local_root / key
            if target.exists():
                try:
                    target.unlink()
                    deleted += 1
                    print(f"Deleted {target}")
                except Exception as exc:
                    print(f"[ERROR] failed to delete {target}: {exc}")
            else:
                print(f"[WARN] {target} not found locally")
        print(f"Deleted {deleted} files locally.")

    print("Comparison complete. Remove the same files on the other host to keep datasets aligned.")


def print_historical_summaries(summaries: List[HistoricalCoinSummary], json_output: bool) -> None:
    if json_output:
        payload = [summary.to_display_dict() for summary in summaries]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not summaries:
        print("No historical datasets summarised.")
        return
    rows = []
    headers = [
        "exchange",
        "coin",
        "files",
        "rows",
        "nan_count",
        "first_ts",
        "last_ts",
        "per_file_rows_ok",
        "monotonic_within",
        "contiguous",
        "hash",
    ]
    rows.append(headers)
    for summary in summaries:
        rows.append(
            [
                summary.exchange,
                summary.coin,
                str(summary.files),
                str(summary.rows),
                str(summary.nan_count),
                str(summary.first_ts or "-"),
                str(summary.last_ts or "-"),
                "yes" if summary.per_file_rows_ok else "no",
                "yes" if summary.monotonic_within_files else "no",
                "yes" if summary.contiguous_across_files else "no",
                summary.data_hash[:16] if summary.data_hash else "-",
            ]
        )

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def format_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    print(format_row(rows[0]))
    for row in rows[1:]:
        print(format_row(row))


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

    if args.command == "summarize-historical":
        summaries = summarize_historical(
            args.paths,
            fast=getattr(args, "fast", False),
            include_hash=getattr(args, "hash", False),
            show_progress=getattr(args, "progress", False),
        )
        print_historical_summaries(summaries, json_output=getattr(args, "json", False))
        return

    if args.command == "clean-historical":
        clean_historical(
            args.paths,
            apply_changes=getattr(args, "apply", False),
            show_progress=getattr(args, "progress", False),
        )
        return

    if args.command == "hash-historical":
        mapping = hash_historical(args.paths, show_progress=getattr(args, "progress", False))
        save_hash_manifest(args.output, args.paths, mapping)
        return

    if args.command == "compare-historical-hashes":
        compare_historical_hashes(
            args.manifest_a,
            args.manifest_b,
            local_root=getattr(args, "local_root", None),
            apply_changes=getattr(args, "apply", False),
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
