"""Explicit offline legacy OHLCV import. Source files are never changed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import import_legacy_range_into_store
from ohlcv_store import OhlcvStore


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--source-layout", choices=("daily_cache", "historical"), default="daily_cache")
    parser.add_argument("--store-root", type=Path, default=Path("caches/ohlcvs"))
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--timeframe", choices=("1m", "1h"), default="1m")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="import missing rows after validation")
    mode.add_argument("--dry-run", action="store_true", help="inspect without changing source or target data (default)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.legacy_root.is_dir():
        raise ValueError("legacy root must be an existing directory")
    if args.exchange.lower() in {"defx", "paradex"}:
        raise ValueError("exchange was retired in v9")
    with TemporaryDirectory(prefix="passivbot-import-inspect-") as scratch:
        root = args.store_root
        catalog_path = root / "catalog.sqlite"
        if args.execute:
            catalog = OhlcvCatalog(catalog_path)
        elif catalog_path.exists():
            catalog = OhlcvCatalog(catalog_path, read_only=True)
        elif root.exists() and any(root.iterdir()):
            raise ValueError("nonempty store has no catalog; repair it before import")
        else:
            root = Path(scratch)
            catalog = OhlcvCatalog(root / "catalog.sqlite")
        store = OhlcvStore(root, catalog)
        rows = import_legacy_range_into_store(
            store=store, legacy_root=args.legacy_root, source_layout=args.source_layout,
            exchange=args.exchange, timeframe=args.timeframe, symbol=args.symbol,
            start_ts=args.start_ms, end_ts=args.end_ms, execute=args.execute,
        )
    print(json.dumps({"execute": args.execute, "rows_imported": rows if args.execute else 0,
                      "rows_would_import": None if args.execute else rows,
                      "existing_valid_rows_preserved": True, "source_files_changed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
