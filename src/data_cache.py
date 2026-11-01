"""Offline, verified transfer of candle caches between filesystem roots.

The bundle format owns portable *locations*. Candle, coverage, market and prepared
input identities remain the responsibility of their existing owners.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
from typing import Iterable

import portalocker

from ohlcv_catalog import ChunkRecord, OhlcvCatalog
from ohlcv_store import MonthChunkPaths, OhlcvStore, _sanitize_symbol

BUNDLE_VERSION = 1
TABLES = ("symbols", "chunks", "gaps", "fetch_log")
GLOBAL_METADATA = (
    "first_ohlcv_timestamps_unified.json",
    "first_ohlcv_timestamps_unified.version",
    "first_ohlcv_timestamps_unified_exchange_specific.json",
    "first_ohlcv_timestamps_unified_exchange_specific_symbols.json",
    "symbol_to_coin_map.json",
    "symbol_to_coin_ambiguities.json",
)
VENUE_METADATA = ("markets.json", "first_timestamps.json", "coin_to_symbol_map.json")


class CacheTransferError(ValueError):
    """Unsafe, changed, incomplete or conflicting cache transfer input."""


def _root(value: Path) -> Path:
    path = Path(value).absolute()
    if path.is_symlink():
        raise CacheTransferError("cache root must not be a symlink")
    return path.resolve()


def _relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise CacheTransferError("bundle paths must be relative POSIX paths")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise CacheTransferError("unsafe bundle path")
    return Path(*p.parts)


def _no_symlinks(path: Path) -> None:
    for p in (path, *path.parents):
        if p.is_symlink():
            raise CacheTransferError("cache transfers reject symlinks")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _copy_stable(src: Path, dst: Path) -> dict:
    _no_symlinks(src)
    before = _fingerprint(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = _sha(dst)
    if before != _fingerprint(src) or digest != _sha(src):
        raise CacheTransferError(
            "source changed while copying; retry a quiescent snapshot"
        )
    return {"bytes": before[2], "mtime_ns": before[3], "sha256": digest}


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    conn = (
        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        if readonly
        else sqlite3.connect(path)
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA trusted_schema=OFF")
    return conn


def _table_info(conn: sqlite3.Connection, table: str):
    return [tuple(x) for x in conn.execute("PRAGMA table_info(" + table + ")")]


def _validate_catalog(
    conn: sqlite3.Connection, reference: sqlite3.Connection, *, partial=False
) -> None:
    objects = {
        (r[0], r[1])
        for r in conn.execute(
            "SELECT name,type FROM sqlite_master "
            "WHERE type IN ('table','view','trigger') "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }
    expected = {(table, "table") for table in TABLES}
    if (objects - expected) or (not partial and objects != expected):
        raise CacheTransferError("unsupported catalog objects")
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise CacheTransferError("invalid catalog integrity")
    for table in TABLES:
        kind = conn.execute(
            "SELECT type FROM sqlite_master WHERE name=?", (table,)
        ).fetchone()
        if kind is None and partial:
            continue
        if (
            kind is None
            or kind[0] != "table"
            or _table_info(conn, table) != _table_info(reference, table)
        ):
            raise CacheTransferError("unsupported catalog schema: " + table)


def _complete_catalog_snapshot(conn: sqlite3.Connection) -> None:
    """Complete only recognized owner tables in a scratch destination snapshot."""
    canonical = OhlcvCatalog(":memory:")._connect()
    try:
        _validate_catalog(conn, canonical, partial=True)
        for table in TABLES:
            if not _table_info(conn, table):
                sql = canonical.execute(
                    "SELECT sql FROM sqlite_master WHERE name=?", (table,)
                ).fetchone()[0]
                conn.execute(sql)
        conn.commit()
    finally:
        canonical.close()


@contextmanager
def _chunk_lock(body: Path, role: str):
    try:
        with portalocker.Lock(str(body) + ".lock", timeout=0, fail_when_locked=True):
            yield
    except portalocker.exceptions.LockException as exc:
        raise CacheTransferError(f"{role} chunk is being written; retry later") from exc


def _chunk_relative(row: dict, field: str) -> Path:
    suffix = ".valid.npy" if field == "valid_path" else ".npy"
    for name in (row["exchange"], row["timeframe"], _sanitize_symbol(row["symbol"])):
        if len(_relative(str(name)).parts) != 1:
            raise CacheTransferError("invalid chunk identity")
    return (
        Path("data")
        / row["exchange"]
        / row["timeframe"]
        / _sanitize_symbol(row["symbol"])
        / f"{row['year']:04d}"
        / f"{row['month']:02d}{suffix}"
    )


def _source_chunk_path(
    row: dict, field: str, root: Path, legacy_roots: Iterable[Path]
) -> Path:
    expected = _chunk_relative(row, field)
    stored = Path(row[field])
    if stored.is_absolute():
        references = (root.resolve(), *(p for p in legacy_roots))
        if not any(stored == old / expected for old in references):
            raise CacheTransferError(
                "catalog location is outside the declared source roots"
            )
    elif _relative(row[field]) != expected:
        raise CacheTransferError("catalog chunk location does not match its identity")
    actual = root / expected
    _no_symlinks(actual)
    if not actual.is_file():
        raise CacheTransferError("catalog chunk file is missing")
    return actual


def _copy_metadata(
    cache: Path, payload: Path, exchanges: list[str], files: dict, observed: dict
) -> None:
    names = [Path(n) for n in GLOBAL_METADATA]
    for exchange in exchanges:
        if len(_relative(exchange).parts) != 1:
            raise CacheTransferError("invalid exchange selection")
        market = cache / exchange / "markets.json"
        if not market.is_file():
            raise CacheTransferError(
                "selected exchange has no markets metadata: " + exchange
            )
        names.extend(Path(exchange) / n for n in VENUE_METADATA)
    for rel in names:
        source = cache / rel
        if source.exists() or source.is_symlink():
            observed[source] = _fingerprint(source)
            files[rel.as_posix()] = _copy_stable(source, payload / rel)


def _prepared_files(directory: Path) -> tuple[dict, set[str]]:
    manifest_path = directory / "manifest.json"
    _no_symlinks(manifest_path)
    if not manifest_path.is_file():
        raise CacheTransferError("prepared dataset manifest is missing")
    declared = json.loads(manifest_path.read_text())
    if (
        declared.get("dataset_kind") != "hlcvs_data"
        or declared.get("schema_version") != 1
    ):
        raise CacheTransferError("unsupported prepared dataset manifest")
    required = {
        "hlcvs",
        "btc_usd_prices",
        "timestamps",
        "coins",
        "market_specific_settings",
    }
    entries = declared.get("files")
    if not isinstance(entries, dict) or not required.issubset(entries):
        raise CacheTransferError("prepared artifact manifest is incomplete")
    allowed = {
        "hlcvs": {"hlcvs.npy", "hlcvs.npy.gz"},
        "btc_usd_prices": {"btc_usd_prices.npy", "btc_usd_prices.npy.gz"},
        "timestamps": {"timestamps.npy", "timestamps.npy.gz"},
        "coins": {"coins.json"},
        "market_specific_settings": {"market_specific_settings.json"},
        "candidate_report": {"candidate_report.json"},
    }
    names = {"manifest.json"}
    if (directory / "cache_meta.json").is_file():
        names.add("cache_meta.json")
    for key, entry in entries.items():
        if (
            key not in allowed
            or not isinstance(entry, dict)
            or entry.get("path") not in allowed[key]
        ):
            raise CacheTransferError("unsupported prepared artifact")
        name = entry["path"]
        _no_symlinks(directory / name)
        if not (directory / name).is_file():
            raise CacheTransferError("prepared artifact is missing: " + name)
        names.add(name)
    return declared, names


def _check_prepared_venues(declared: dict, chosen: list[str]) -> None:
    from utils import to_standard_exchange_name

    venues = {
        s.get(k)
        for s in declared.get("sources", {}).values()
        for k in ("ohlcv_exchange", "market_settings_exchange")
    }
    venues.add(declared.get("btc_benchmark", {}).get("exchange"))
    venues.update(declared.get("requested", {}).get("exchanges", []))
    if {to_standard_exchange_name(v) for v in venues if v} - {
        to_standard_exchange_name(v) for v in chosen
    }:
        raise CacheTransferError("prepared dataset uses an unselected exchange")


def _verify_prepared(directory: Path) -> None:
    # Use the existing logical-array/JSON verifier, including a file-backed
    # loader for compressed histories. Transport hashes do not replace it.
    from hlcvs_manifest import numpy_artifact_loader, verify_hlcvs_manifest
    from shared_arrays import SharedArrayManager

    manager = SharedArrayManager(file_backed=True)
    try:
        with numpy_artifact_loader(manager.load_numpy_artifact):
            verify_hlcvs_manifest(directory)
        manager.discard_file_pages()
    finally:
        manager.cleanup()


def export_cache(
    cache_root: Path,
    output: Path,
    *,
    exchanges: Iterable[str] = (),
    prepared_datasets: Iterable[str] = (),
    legacy_ohlcv_roots: Iterable[Path] = (),
) -> dict:
    """Create a new directory bundle. Source facts are never rewritten."""
    cache_root, output = _root(cache_root), _root(output)
    _no_symlinks(cache_root)
    _no_symlinks(output)
    if output.exists():
        raise CacheTransferError("export output must not already exist")
    if output.is_relative_to(cache_root) or cache_root.is_relative_to(output):
        raise CacheTransferError("bundle and source cache roots must be disjoint")
    raw = cache_root / "ohlcvs"
    catalog_path = raw / "catalog.sqlite"
    _no_symlinks(catalog_path)
    if not catalog_path.is_file():
        raise CacheTransferError("raw coverage catalog is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".data-cache-export-", dir=output.parent
    ) as tmp:
        stage = Path(tmp)
        payload = stage / "payload"
        (payload / "ohlcvs").mkdir(parents=True)
        source = _connect(catalog_path, readonly=True)
        snapshot = _connect(payload / "ohlcvs/catalog.sqlite")
        try:
            data_version = source.execute("PRAGMA data_version").fetchone()[0]
            source.backup(snapshot)
            expected_catalog = OhlcvCatalog(":memory:")
            _validate_catalog(snapshot, expected_catalog._connect())
            expected_catalog._connect().close()
            chosen = sorted(
                set(exchanges)
                or {
                    r[0]
                    for r in snapshot.execute("SELECT DISTINCT exchange FROM chunks")
                }
            )
            if not chosen:
                raise CacheTransferError("no chunks selected")
            for table in TABLES:
                snapshot.execute(
                    "DELETE FROM "
                    + table
                    + " WHERE exchange NOT IN ("
                    + ",".join("?" for _ in chosen)
                    + ")",
                    chosen,
                )
            rows = [
                dict(x)
                for x in snapshot.execute(
                    "SELECT * FROM chunks ORDER BY exchange,timeframe,symbol,year,month"
                )
            ]
            if not rows:
                raise CacheTransferError("no chunks selected")
            files = {}
            observed = {}
            store = OhlcvStore(raw, OhlcvCatalog(catalog_path, read_only=True))
            for row in rows:
                body = _source_chunk_path(row, "body_path", raw, legacy_ohlcv_roots)
                valid = _source_chunk_path(row, "valid_path", raw, legacy_ohlcv_roots)
                with _chunk_lock(body, "source"):
                    paths = MonthChunkPaths(body, valid)
                    chunk = replace(
                        ChunkRecord(**row), body_path=str(body), valid_path=str(valid)
                    )
                    store._validate_chunk_files(chunk, paths)
                    store._verify_chunk_checksum_unlocked(chunk, paths)
                    for field, actual in (("body_path", body), ("valid_path", valid)):
                        rel = Path("ohlcvs") / _chunk_relative(row, field)
                        observed[actual] = _fingerprint(actual)
                        files[rel.as_posix()] = _copy_stable(actual, payload / rel)
                        row[field] = _chunk_relative(row, field).as_posix()
                    snapshot.execute(
                        "UPDATE chunks SET body_path=?,valid_path=? "
                        "WHERE exchange=? AND timeframe=? AND symbol=? "
                        "AND year=? AND month=?",
                        (
                            row["body_path"],
                            row["valid_path"],
                            row["exchange"],
                            row["timeframe"],
                            row["symbol"],
                            row["year"],
                            row["month"],
                        ),
                    )
            _copy_metadata(cache_root, payload, chosen, files, observed)
            prepared = []
            for name in prepared_datasets:
                rel = _relative(name)
                if len(rel.parts) != 1:
                    raise CacheTransferError(
                        "prepared dataset must be a cache directory name"
                    )
                directory = cache_root / "hlcvs_data" / rel
                _no_symlinks(directory)
                declared, names = _prepared_files(directory)
                _check_prepared_venues(declared, chosen)
                for child in directory.iterdir():
                    _no_symlinks(child)
                    if child.is_file() and child.name in names:
                        dest = Path("hlcvs_data") / rel / child.name
                        observed[child] = _fingerprint(child)
                        files[dest.as_posix()] = _copy_stable(child, payload / dest)
                    elif child.is_dir():
                        raise CacheTransferError(
                            "nested prepared dataset directories are unsupported"
                        )
                _verify_prepared(payload / "hlcvs_data" / rel)
                prepared.append(name)
            if any(before != _fingerprint(path) for path, before in observed.items()):
                raise CacheTransferError("source chunks changed during export")
            if source.execute("PRAGMA data_version").fetchone()[0] != data_version:
                raise CacheTransferError(
                    "source catalog changed during export; retry later"
                )
            snapshot.commit()
            mode = snapshot.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            if mode != "delete":
                raise CacheTransferError("could not finalize portable catalog snapshot")
        finally:
            snapshot.close()
            source.close()
        cat = payload / "ohlcvs/catalog.sqlite"
        for suffix in ("-wal", "-shm"):
            Path(str(cat) + suffix).unlink(missing_ok=True)
        files["ohlcvs/catalog.sqlite"] = {
            "bytes": cat.stat().st_size,
            "mtime_ns": cat.stat().st_mtime_ns,
            "sha256": _sha(cat),
        }
        manifest = {
            "schema_version": BUNDLE_VERSION,
            "kind": "passivbot-data-cache",
            "catalog_path_encoding": "ohlcvs-root-relative-v1",
            "exchanges": chosen,
            "prepared_datasets": prepared,
            "files": files,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.rename(stage, output)
        return manifest


def _validated_bundle(bundle: Path) -> tuple[dict, dict[str, Path]]:
    _no_symlinks(bundle)
    manifest_path = bundle / "manifest.json"
    _no_symlinks(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != BUNDLE_VERSION
        or manifest.get("kind") != "passivbot-data-cache"
        or manifest.get("catalog_path_encoding") != "ohlcvs-root-relative-v1"
    ):
        raise CacheTransferError("unsupported cache bundle version")
    for key in ("exchanges", "prepared_datasets"):
        values = manifest.get(key)
        if (
            not isinstance(values, list)
            or any(
                not isinstance(v, str) or len(_relative(v).parts) != 1 for v in values
            )
            or len(set(values)) != len(values)
        ):
            raise CacheTransferError("invalid bundle selection")
    if not manifest["exchanges"]:
        raise CacheTransferError("bundle has no selected exchanges")
    files = manifest.get("files")
    if not isinstance(files, dict) or "ohlcvs/catalog.sqlite" not in files:
        raise CacheTransferError("bundle file manifest is incomplete")
    resolved = {}
    for name, facts in files.items():
        rel = _relative(name)
        path = bundle / "payload" / rel
        _no_symlinks(path)
        if rel.parts[0] == "hlcvs_data" and (
            len(rel.parts) != 3 or rel.parts[1] not in manifest["prepared_datasets"]
        ):
            raise CacheTransferError("prepared artifact is outside declared datasets")
        if (
            rel.parts[0] not in ("ohlcvs", "hlcvs_data")
            and rel.as_posix() not in GLOBAL_METADATA
            and not (
                len(rel.parts) == 2
                and rel.parts[0] in manifest["exchanges"]
                and rel.parts[1] in VENUE_METADATA
            )
        ):
            raise CacheTransferError("unsupported cache artifact path")
        if (
            not path.is_file()
            or type(facts.get("bytes")) is not int
            or type(facts.get("mtime_ns")) is not int
            or not isinstance(facts.get("sha256"), str)
        ):
            raise CacheTransferError("invalid or missing bundle artifact")
        if path.stat().st_size != facts["bytes"] or _sha(path) != facts["sha256"]:
            raise CacheTransferError("bundle artifact checksum mismatch: " + name)
        resolved[name] = path
    actual = set()
    for p in (bundle / "payload").rglob("*"):
        _no_symlinks(p)
        if p.is_file():
            actual.add(p.relative_to(bundle / "payload").as_posix())
    if actual != set(files):
        raise CacheTransferError("bundle contains unlisted or missing artifacts")
    for name in manifest["prepared_datasets"]:
        directory = bundle / "payload/hlcvs_data" / name
        prepared_manifest, names = _prepared_files(directory)
        _check_prepared_venues(prepared_manifest, manifest["exchanges"])
        declared = {"hlcvs_data/" + name + "/" + n for n in names}
        if declared != {n for n in files if n.startswith("hlcvs_data/" + name + "/")}:
            raise CacheTransferError(
                "prepared dataset file list is incomplete or unsupported"
            )
    return manifest, resolved


def _merge_rows(
    dest: sqlite3.Connection, incoming: sqlite3.Connection, cache_root: Path
) -> None:
    # Overlapping observations can legitimately coexist in one owner snapshot.
    # Only destination evidence absent from that snapshot can conflict with it.
    incoming_gaps = {tuple(row) for row in incoming.execute("SELECT * FROM gaps")}
    foreign_gaps = {}
    for row in dest.execute("SELECT * FROM gaps"):
        if tuple(row) not in incoming_gaps:
            foreign_gaps.setdefault(
                (row["exchange"], row["timeframe"], row["symbol"]), []
            ).append(row)
    for table in TABLES:
        info = _table_info(dest, table)
        columns = [x[1] for x in info]
        primary = [x[1] for x in sorted(info, key=lambda x: x[5]) if x[5]]
        for original in incoming.execute("SELECT * FROM " + table):
            row = dict(original)
            if table == "chunks":
                for field in ("body_path", "valid_path"):
                    expected = _chunk_relative(row, field)
                    if _relative(row[field]) != expected:
                        raise CacheTransferError(
                            "portable catalog path does not match chunk identity"
                        )
                    row[field] = str(cache_root / "ohlcvs" / expected)
            where = " AND ".join('"' + c + '"=?' for c in primary)
            existing = dest.execute(
                "SELECT * FROM " + table + " WHERE " + where, [row[c] for c in primary]
            ).fetchone()
            if existing is not None:
                if dict(existing) != row:
                    raise CacheTransferError(
                        "existing catalog facts conflict in "
                        + table
                        + "; import into a fresh cache root"
                    )
                continue
            if table == "gaps":
                overlaps = any(
                    gap["end_ts"] >= row["start_ts"]
                    and gap["start_ts"] <= row["end_ts"]
                    for gap in foreign_gaps.get(
                        (row["exchange"], row["timeframe"], row["symbol"]), ()
                    )
                )
                if overlaps:
                    raise CacheTransferError(
                        "existing gap evidence overlaps incoming facts"
                    )
            names = ",".join('"' + c + '"' for c in columns)
            dest.execute(
                "INSERT INTO "
                + table
                + " ("
                + names
                + ") VALUES ("
                + ",".join("?" for _ in columns)
                + ")",
                [row[c] for c in columns],
            )


def _install_file(source: Path, target: Path) -> None:
    # Link from a verified same-filesystem staging area. Exclusive publication
    # refuses a destination created after validation rather than replacing it.
    os.link(source, target)


def _import_cache(bundle: Path, cache_root: Path) -> dict:
    """Merge absent/identical artifacts only; conflicting facts are never overwritten.

    Use an offline destination. Existing readers/writers are not redirected to a
    new cache, and the command never rewrites existing candle or metadata files.
    """
    bundle, cache_root = _root(bundle), _root(cache_root)
    _no_symlinks(bundle)
    _no_symlinks(cache_root)
    if bundle.is_relative_to(cache_root) or cache_root.is_relative_to(bundle):
        raise CacheTransferError("bundle and destination must be disjoint")
    manifest, files = _validated_bundle(bundle)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".data-cache-import-", dir=cache_root.parent
    ) as tmp:
        stage = Path(tmp)
        payload = stage / "payload"
        for name, source in files.items():
            copied = _copy_stable(source, payload / _relative(name))
            if copied["sha256"] != manifest["files"][name]["sha256"]:
                raise CacheTransferError("bundle changed during staging")
            os.utime(
                payload / _relative(name), ns=(manifest["files"][name]["mtime_ns"],) * 2
            )
        for name in manifest["prepared_datasets"]:
            _verify_prepared(payload / "hlcvs_data" / name)
        reference_catalog = OhlcvCatalog(stage / "reference.sqlite")
        reference = reference_catalog._connect()
        incoming = _connect(payload / "ohlcvs/catalog.sqlite", readonly=True)
        try:
            _validate_catalog(incoming, reference)
            for table in TABLES:
                if any(
                    row[0] not in manifest["exchanges"]
                    for row in incoming.execute(
                        "SELECT DISTINCT exchange FROM " + table
                    )
                ):
                    raise CacheTransferError("catalog exchange is outside selection")
            expected_raw = {"ohlcvs/catalog.sqlite"}
            for row in incoming.execute("SELECT * FROM chunks"):
                row = dict(row)
                if row["exchange"] not in manifest["exchanges"]:
                    raise CacheTransferError("catalog exchange is outside selection")
                paths = []
                for field in ("body_path", "valid_path"):
                    rel = _relative(row[field])
                    if (
                        rel != _chunk_relative(row, field)
                        or (Path("ohlcvs") / rel).as_posix() not in files
                    ):
                        raise CacheTransferError(
                            "catalog references an undeclared chunk"
                        )
                    paths.append(payload / "ohlcvs" / rel)
                    expected_raw.add((Path("ohlcvs") / rel).as_posix())
                store = OhlcvStore(payload / "ohlcvs", reference_catalog)
                chunk = replace(
                    ChunkRecord(**row),
                    body_path=str(paths[0]),
                    valid_path=str(paths[1]),
                )
                store._validate_chunk_files(chunk, MonthChunkPaths(*paths))
                store._verify_chunk_checksum_unlocked(chunk, MonthChunkPaths(*paths))
            if expected_raw != {n for n in files if n.startswith("ohlcvs/")}:
                raise CacheTransferError("bundle contains unsupported raw-store files")
            old = {}
            absent = []
            for name in files:
                if name == "ohlcvs/catalog.sqlite":
                    continue
                target = cache_root / _relative(name)
                _no_symlinks(target)
                if target.exists():
                    if (
                        not target.is_file()
                        or _sha(target) != manifest["files"][name]["sha256"]
                    ):
                        raise CacheTransferError("existing artifact conflicts: " + name)
                    if (
                        name.split("/")[0] not in ("ohlcvs", "hlcvs_data")
                        and target.stat().st_mtime_ns
                        != manifest["files"][name]["mtime_ns"]
                    ):
                        raise CacheTransferError(
                            "existing metadata refresh time conflicts: " + name
                        )
                    old[target] = _fingerprint(target)
                else:
                    absent.append(name)
            # Preflight catalog conflicts in a scratch copy before touching target.
            target_db = cache_root / "ohlcvs/catalog.sqlite"
            _no_symlinks(target_db)
            existed = target_db.exists()
            # A prior process can die after exclusive reservation but before the
            # owner creates the schema. A zero-byte catalog contains no facts;
            # keep the trusted scratch schema and let the owner initialize it.
            if existed and target_db.stat().st_size:
                with _connect(target_db, readonly=True) as current:
                    current.backup(reference)
            _complete_catalog_snapshot(reference)
            reference.execute("BEGIN IMMEDIATE")
            try:
                _merge_rows(reference, incoming, cache_root)
            finally:
                reference.rollback()
            created_dirs = []
            installed = []
            publication_files = []
            dest = None
            target_catalog = None
            try:
                for name in absent + ["ohlcvs/catalog.sqlite"]:
                    parent = (cache_root / _relative(name)).parent
                    missing = []
                    while not parent.exists():
                        missing.append(parent)
                        parent = parent.parent
                    for directory in reversed(missing):
                        directory.mkdir()
                        created_dirs.append(directory)
                if not existed:
                    try:
                        fd = os.open(
                            target_db, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                        )
                    except FileExistsError as exc:
                        raise CacheTransferError(
                            "destination catalog appeared during import; retry"
                        ) from exc
                    os.close(fd)
                target_catalog = OhlcvCatalog(target_db)
                dest = target_catalog._connect()
                dest.execute("BEGIN IMMEDIATE")
                _merge_rows(dest, incoming, cache_root)
                observed = dict(old)
                remaining = set(absent)

                def publish(name):
                    target = cache_root / _relative(name)
                    # A cache root or individual artifact directory may be a
                    # mount point. Stage beside each final path, not its root.
                    descriptor, temporary_path = tempfile.mkstemp(
                        prefix=".data-cache-publish-", dir=target.parent
                    )
                    os.close(descriptor)
                    local = Path(temporary_path)
                    publication_files.append(local)
                    copied = _copy_stable(payload / _relative(name), local)
                    if copied["sha256"] != manifest["files"][name]["sha256"]:
                        raise CacheTransferError(
                            "staged artifact changed during import"
                        )
                    installed.append((target, local))
                    _install_file(local, target)
                    observed[target] = _fingerprint(target)
                    remaining.remove(name)

                chunk_names = []
                for row in incoming.execute("SELECT * FROM chunks ORDER BY body_path"):
                    names = [
                        (Path("ohlcvs") / _relative(row[field])).as_posix()
                        for field in ("body_path", "valid_path")
                    ]
                    chunk_names.append(names)
                    with _chunk_lock(cache_root / _relative(names[0]), "destination"):
                        for name in names:
                            if name in remaining:
                                publish(name)
                for name in sorted(
                    remaining, key=lambda n: (n.endswith("manifest.json"), n)
                ):
                    publish(name)
                # Bounded locks plus complete final fingerprints detect a
                # changed offline destination without retaining O(chunks) FDs.
                for names in chunk_names:
                    with _chunk_lock(cache_root / _relative(names[0]), "destination"):
                        if any(
                            _fingerprint(cache_root / _relative(name))
                            != observed[cache_root / _relative(name)]
                            for name in names
                        ):
                            raise CacheTransferError(
                                "destination changed during import"
                            )
                if any(_fingerprint(p) != before for p, before in observed.items()):
                    raise CacheTransferError("destination changed during import")
                dest.commit()
            except BaseException:
                if dest is not None:
                    dest.rollback()
                for p, staged in reversed(installed):
                    if p.exists() and (p.stat().st_dev, p.stat().st_ino) == (
                        staged.stat().st_dev,
                        staged.stat().st_ino,
                    ):
                        p.unlink()
                if target_catalog is not None:
                    target_catalog._connect().close()
                # Never unlink a database on failure. Another application may
                # have opened it or committed unrelated facts. An empty trusted
                # catalog is harmless and a retry can complete the import.
                for d in reversed(created_dirs):
                    try:
                        d.rmdir()
                    except OSError:
                        pass  # Other writers/import locks can retain a new directory.
                raise
            finally:
                if dest is not None:
                    dest.close()
                for temporary in publication_files:
                    temporary.unlink(missing_ok=True)
            return {
                "schema_version": BUNDLE_VERSION,
                "verified_files": len(files),
                "installed_files": len(installed),
                "existing_identical_files": len(old),
                "prepared_datasets": manifest["prepared_datasets"],
            }
        finally:
            incoming.close()
            reference.close()


def import_cache(bundle: Path, cache_root: Path) -> dict:
    """Serialize imports to one offline destination without replacing old facts."""
    cache_root = _root(cache_root)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    lease = cache_root.parent / ("." + cache_root.name + ".data-cache-import.lock")
    _no_symlinks(lease)
    try:
        with portalocker.Lock(str(lease), timeout=0, fail_when_locked=True):
            return _import_cache(bundle, cache_root)
    except portalocker.exceptions.LockException as exc:
        raise CacheTransferError(
            "another import owns this destination; retry later"
        ) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export/import verified offline data caches "
            "without rewriting historical facts."
        )
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    export = sub.add_parser("export", help="create a new portable directory bundle")
    export.add_argument("--cache-root", type=Path, default=Path("caches"))
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--exchange", action="append", default=[])
    export.add_argument(
        "--prepared-dataset",
        action="append",
        default=[],
        help="exact directory name under hlcvs_data; repeatable",
    )
    export.add_argument(
        "--legacy-ohlcv-root",
        type=Path,
        action="append",
        default=[],
        help="explicit original raw-cache root for a previously moved catalog",
    )
    imp = sub.add_parser(
        "import",
        help="verify and merge into an offline destination; conflicts are refused",
    )
    imp.add_argument("--bundle", type=Path, required=True)
    imp.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "export":
        result = export_cache(
            args.cache_root,
            args.output,
            exchanges=args.exchange,
            prepared_datasets=args.prepared_dataset,
            legacy_ohlcv_roots=args.legacy_ohlcv_root,
        )
        result = {
            "schema_version": result["schema_version"],
            "files": len(result["files"]),
            "exchanges": result["exchanges"],
            "prepared_datasets": result["prepared_datasets"],
        }
    else:
        result = import_cache(args.bundle, args.cache_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
