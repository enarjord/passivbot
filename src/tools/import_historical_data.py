"""Explicit offline legacy OHLCV import. Source files are never changed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import import_legacy_range_into_store
from ohlcv_store import OhlcvStore, timeframe_to_interval_ms


def _copy_catalog_for_inspection(catalog_path: Path, scratch: Path) -> Path:
    """Read a stable DB/WAL copy; let SQLite create sidecars only in scratch."""
    def sources():
        wal = catalog_path.with_name(catalog_path.name + "-wal")
        return [catalog_path, *([wal] if wal.exists() else [])]

    def identity(path):
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        fields = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if fields(before) != fields(after):
            raise ValueError("catalog changed during dry-run inspection")
        return fields(after), digest

    originals = sources()
    before = {path: identity(path) for path in originals}
    for path in originals:
        copied = scratch / path.name
        shutil.copyfile(path, copied)
        if identity(copied)[1] != before[path][1]:
            raise ValueError("catalog changed during dry-run inspection")
    if sources() != originals or any(identity(path) != before[path] for path in originals):
        raise ValueError("catalog changed during dry-run inspection")
    return scratch / catalog_path.name


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--source-layout", choices=("daily_cache", "historical"), default="daily_cache")
    parser.add_argument("--store-root", type=Path, default=Path("caches/ohlcvs"))
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--timeframe", choices=("1m", "1h"), default="1m")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True, help="exclusive end timestamp")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="import missing rows after validation")
    mode.add_argument("--dry-run", action="store_true", help="inspect without changing source or target data (default)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.legacy_root.is_dir():
        raise ValueError("legacy root must be an existing directory")
    interval = timeframe_to_interval_ms(args.timeframe)
    if args.end_ms <= args.start_ms or args.start_ms % interval or args.end_ms % interval:
        raise ValueError(f"nonempty import range must align to {args.timeframe}")
    if args.exchange.lower() in {"defx", "paradex"}:
        raise ValueError("exchange was retired in v9")
    with TemporaryDirectory(prefix="passivbot-import-inspect-") as scratch:
        root = args.store_root
        catalog_path = root / "catalog.sqlite"
        if args.execute:
            catalog = OhlcvCatalog(catalog_path)
        elif catalog_path.exists():
            # WAL-mode snapshots may need new SQLite sidecars. All catalog
            # writes are confined to scratch; destination chunks are read only.
            catalog = OhlcvCatalog(_copy_catalog_for_inspection(catalog_path, Path(scratch)))
        elif root.exists() and any(root.iterdir()):
            raise ValueError("nonempty store has no catalog; repair it before import")
        else:
            root = Path(scratch)
            catalog = OhlcvCatalog(root / "catalog.sqlite")
        try:
            store = OhlcvStore(root, catalog)
            rows = import_legacy_range_into_store(
                store=store, legacy_root=args.legacy_root, source_layout=args.source_layout,
                exchange=args.exchange, timeframe=args.timeframe, symbol=args.symbol,
                start_ts=args.start_ms, end_ts=args.end_ms - interval, execute=args.execute,
            )
        finally:
            connection = getattr(catalog._local, "conn", None)
            if connection is not None:
                connection.close()
    print(json.dumps({"execute": args.execute, "rows_imported": rows if args.execute else 0,
                      "rows_would_import": None if args.execute else rows,
                      "existing_valid_rows_preserved": True, "source_files_changed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
