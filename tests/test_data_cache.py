from pathlib import Path
import json
import gzip
import errno
import os
import sqlite3
import subprocess
import sys

import numpy as np
import portalocker
import pytest

import data_cache
from data_cache import CacheTransferError, export_cache, import_cache
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore, month_start_ts


def make_cache(root, *, exchange="binance", symbol="BTC/USDT:USDT", close=100):
    root = root.resolve()
    catalog = OhlcvCatalog(root / "ohlcvs/catalog.sqlite")
    store = OhlcvStore(root / "ohlcvs", catalog)
    start = month_start_ts(2026, 4)
    values = np.array(
        [[close + 1, close - 1, close, 10], [close + 2, close, close + 1, 11]],
        dtype=np.float32,
    )
    store.write_rows(exchange, "1m", symbol, np.array([start, start + 60000]), values)
    market = root / exchange / "markets.json"
    market.parent.mkdir(exist_ok=True)
    market.write_text('{"fixture":"public"}\n')
    os.utime(market, ns=(1_600_000_000_123456789,) * 2)
    for name in data_cache.GLOBAL_METADATA:
        (root / name).write_text("1" if name.endswith(".version") else "{}")
        os.utime(root / name, ns=(1_600_000_000_123456789,) * 2)
    catalog.mark_gap(
        exchange=exchange,
        timeframe="1m",
        symbol=symbol,
        start_ts=start + 120000,
        end_ts=start + 180000,
        reason="unknown",
        persistent=True,
        last_attempt_at=123,
        next_retry_at=456,
        note="fixture",
    )
    catalog.record_fetch_attempt(
        exchange=exchange,
        timeframe="1m",
        symbol=symbol,
        start_ts=start,
        end_ts=start + 60000,
        attempt=1,
        outcome="ok",
        latency_ms=1,
    )
    return catalog, store, start, values


def snapshot(root):
    if not root.exists():
        return {}
    files = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
        and p.name != "catalog.sqlite"
        and not p.name.endswith(("-wal", "-shm", ".lock"))
    }
    db = root / "ohlcvs/catalog.sqlite"
    if db.is_file():
        with sqlite3.connect(db) as conn:
            files["__catalog_rows__"] = {
                t: conn.execute("SELECT * FROM " + t + " ORDER BY rowid").fetchall()
                for t in data_cache.TABLES
            }
    return files


def test_roundtrip_relocates_only_catalog_paths_and_preserves_facts(tmp_path):
    source = tmp_path / "source"
    catalog, store, start, expected = make_cache(source)
    original = [dict(x) for x in catalog._connect().execute("SELECT * FROM chunks")]
    bundle = tmp_path / "bundle"
    m = export_cache(source, bundle)
    with sqlite3.connect(bundle / "payload/ohlcvs/catalog.sqlite") as c:
        body, valid = c.execute("SELECT body_path,valid_path FROM chunks").fetchone()
    assert body == "data/binance/1m/BTC_USDT_USDT/2026/04.npy"
    assert valid.endswith("04.valid.npy")
    assert [
        dict(x) for x in catalog._connect().execute("SELECT * FROM chunks")
    ] == original
    target = tmp_path / "destination"
    result = import_cache(bundle, target)
    assert result["verified_files"] == len(m["files"])
    loaded = OhlcvStore(
        target / "ohlcvs", OhlcvCatalog(target / "ohlcvs/catalog.sqlite")
    )
    actual = loaded.read_range("binance", "1m", "BTC/USDT:USDT", start, start + 60000)
    np.testing.assert_array_equal(actual.values, expected)
    assert actual.valid.all()
    assert (target / "binance/markets.json").stat().st_mtime_ns == (
        source / "binance/markets.json"
    ).stat().st_mtime_ns
    for table in ("symbols", "gaps", "fetch_log"):
        assert [
            tuple(x)
            for x in loaded.catalog._connect().execute("SELECT * FROM " + table)
        ] == [tuple(x) for x in catalog._connect().execute("SELECT * FROM " + table)]
    assert import_cache(bundle, target)["installed_files"] == 0


def test_merge_keeps_unrelated_destination_symbol_and_venue(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    make_cache(source)
    dest, _, _, _ = make_cache(target, exchange="bybit", symbol="ETH/USDT:USDT")
    original = [tuple(x) for x in dest._connect().execute("SELECT * FROM chunks")]
    export_cache(source, tmp_path / "bundle")
    import_cache(tmp_path / "bundle", target)
    rows = [
        tuple(x)
        for x in dest._connect().execute("SELECT * FROM chunks WHERE exchange='bybit'")
    ]
    assert rows == original
    assert dest._connect().execute("SELECT count(*) FROM chunks").fetchone()[0] == 2


def test_conflicting_artifact_refused_before_any_destination_change(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    make_cache(source)
    make_cache(target, close=200)
    export_cache(source, tmp_path / "bundle")
    before = snapshot(target)
    with pytest.raises(CacheTransferError, match="conflict"):
        import_cache(tmp_path / "bundle", target)
    assert snapshot(target) == before


def test_identical_bytes_do_not_replace_different_metadata_refresh_time(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    make_cache(source)
    export_cache(source, tmp_path / "bundle")
    (target / "binance").mkdir(parents=True)
    (target / "binance/markets.json").write_bytes(
        (source / "binance/markets.json").read_bytes()
    )
    before = (target / "binance/markets.json").stat().st_mtime_ns
    with pytest.raises(CacheTransferError, match="refresh time"):
        import_cache(tmp_path / "bundle", target)
    assert (target / "binance/markets.json").stat().st_mtime_ns == before


@pytest.mark.parametrize("failure_after", [0, 1, 2])
def test_interrupted_install_rolls_back_files_and_catalog(
    tmp_path, monkeypatch, failure_after
):
    source, target = tmp_path / "source", tmp_path / "target"
    make_cache(source)
    catalog, _, _, _ = make_cache(target, exchange="bybit", symbol="ETH/USDT:USDT")
    export_cache(source, tmp_path / "bundle")
    before = snapshot(target)
    rows = {
        table: [tuple(x) for x in catalog._connect().execute("SELECT * FROM " + table)]
        for table in data_cache.TABLES
    }
    install = data_cache._install_file
    count = 0

    def failing(src, dst):
        nonlocal count
        if count == failure_after:
            raise RuntimeError("injected interruption")
        count += 1
        install(src, dst)

    monkeypatch.setattr(data_cache, "_install_file", failing)
    with pytest.raises(RuntimeError, match="injected"):
        import_cache(tmp_path / "bundle", target)
    assert snapshot(target) == before
    for table in data_cache.TABLES:
        assert [
            tuple(x) for x in catalog._connect().execute("SELECT * FROM " + table)
        ] == rows[table]


@pytest.mark.parametrize("kind", ["corrupt", "missing", "unlisted"])
def test_bad_bundle_is_rejected_before_target_creation(tmp_path, kind):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    p = tmp_path / "bundle/payload/binance/markets.json"
    if kind == "corrupt":
        p.write_text("changed")
    elif kind == "missing":
        p.unlink()
    else:
        (p.parent / "extra.json").write_text("{}")
    with pytest.raises(CacheTransferError):
        import_cache(tmp_path / "bundle", tmp_path / "target")
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "ohlcvs/../../escape",
        "ohlcvs\\escape",
        "C:/escape",
        "ohlcvs/./data",
    ],
)
def test_unsafe_manifest_paths_are_rejected(tmp_path, path):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    p = tmp_path / "bundle/manifest.json"
    m = json.loads(p.read_text())
    m["files"][path] = m["files"]["binance/markets.json"]
    p.write_text(json.dumps(m))
    with pytest.raises(CacheTransferError):
        import_cache(tmp_path / "bundle", tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_source_and_bundle_symlinks_are_rejected(tmp_path):
    make_cache(tmp_path / "source")
    market = tmp_path / "source/binance/markets.json"
    market.rename(tmp_path / "market-original")
    market.symlink_to(tmp_path / "market-original")
    with pytest.raises(CacheTransferError, match="symlink"):
        export_cache(tmp_path / "source", tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_destination_catalog_symlink_is_rejected(tmp_path):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    root = tmp_path / "target/ohlcvs"
    root.mkdir(parents=True)
    foreign = tmp_path / "foreign.sqlite"
    foreign.write_bytes(b"untouched")
    (root / "catalog.sqlite").symlink_to(foreign)
    with pytest.raises(CacheTransferError, match="symlink"):
        import_cache(tmp_path / "bundle", tmp_path / "target")
    assert foreign.read_bytes() == b"untouched"


def test_active_chunk_writer_prevents_export(tmp_path):
    c, _, start, _ = make_cache(tmp_path / "source")
    chunk = c.list_chunks("binance", "1m", "BTC/USDT:USDT", start, start)[0]
    with portalocker.Lock(chunk.body_path + ".lock", timeout=0, fail_when_locked=True):
        with pytest.raises(CacheTransferError, match="being written"):
            export_cache(tmp_path / "source", tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_mutating_source_metadata_rejects_snapshot(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    copy = data_cache._copy_metadata

    def changing(*args):
        copy(*args)
        (tmp_path / "source/binance/markets.json").write_text('{"changed":true}')

    monkeypatch.setattr(data_cache, "_copy_metadata", changing)
    with pytest.raises(CacheTransferError, match="changed"):
        export_cache(tmp_path / "source", tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_legacy_absolute_root_requires_explicit_mapping(tmp_path):
    root = tmp_path / "source"
    catalog, _, _, _ = make_cache(root)
    conn = catalog._connect()
    for field in ("body_path", "valid_path"):
        for row in conn.execute("SELECT rowid," + field + " FROM chunks").fetchall():
            old = Path(row[field])
            value = Path("/old-cache-root") / old.relative_to(root.resolve() / "ohlcvs")
            conn.execute(
                "UPDATE chunks SET " + field + "=? WHERE rowid=?",
                (str(value), row["rowid"]),
            )
    conn.commit()
    with pytest.raises(CacheTransferError, match="declared source roots"):
        export_cache(root, tmp_path / "rejected")
    export_cache(
        root, tmp_path / "bundle", legacy_ohlcv_roots=[Path("/old-cache-root")]
    )
    import_cache(tmp_path / "bundle", tmp_path / "target")


def test_cli_roundtrip_is_offline_and_relocatable(tmp_path):
    make_cache(tmp_path / "source")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    cli = [
        sys.executable,
        "-c",
        "from passivbot_cli.main import console_main; console_main()",
        "tool",
        "data-cache",
    ]
    subprocess.run(
        cli
        + [
            "export",
            "--cache-root",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "bundle"),
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        cli
        + [
            "import",
            "--bundle",
            str(tmp_path / "bundle"),
            "--cache-root",
            str(tmp_path / "target"),
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "target/ohlcvs/catalog.sqlite").is_file()


def make_prepared(
    cache, name="prepared-example", *, exchange="binance", compress=False
):
    from hlcvs_manifest import hash_logical_array, hash_json_value

    directory = cache / "hlcvs_data" / name
    directory.mkdir(parents=True)
    entries = {}
    for key in ("hlcvs", "btc_usd_prices", "timestamps"):
        array = np.zeros((2, 1), dtype=np.float64)
        filename = key + (".npy.gz" if compress else ".npy")
        if compress:
            with gzip.open(directory / filename, "wb") as stream:
                np.save(stream, array)
        else:
            np.save(directory / filename, array)
        entries[key] = {
            "path": filename,
            "sha256": hash_logical_array(array),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    for key, value in (("coins", ["BTC"]), ("market_specific_settings", {"BTC": {}})):
        (directory / (key + ".json")).write_text(json.dumps(value))
        entries[key] = {"path": key + ".json", "sha256": hash_json_value(value)}
    manifest = {
        "schema_version": 1,
        "materialization_schema_version": 1,
        "dataset_kind": "hlcvs_data",
        "config_hash": "preserved-original-id",
        "files": entries,
        "sources": {
            "BTC": {"ohlcv_exchange": exchange, "market_settings_exchange": exchange}
        },
        "btc_benchmark": {"exchange": exchange},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "cache_meta.json").write_text('{"original":true}')
    return directory


@pytest.mark.parametrize("compress", [False, True])
def test_prepared_files_keep_identity_and_exclude_unrelated_results(tmp_path, compress):
    source = tmp_path / "source"
    make_cache(source)
    prepared = make_prepared(source, compress=compress)
    original = (prepared / "manifest.json").read_bytes()
    (prepared / "checkpoint.pkl").write_bytes(b"old fitness must not transfer")
    bundle = tmp_path / "bundle"
    result = export_cache(source, bundle, prepared_datasets=["prepared-example"])
    assert not any(n.endswith(".pkl") for n in result["files"])
    import_cache(bundle, tmp_path / "target")
    target = tmp_path / "target/hlcvs_data/prepared-example"
    assert (target / "manifest.json").read_bytes() == original
    assert not (target / "checkpoint.pkl").exists()


def test_missing_prepared_artifact_prevents_export(tmp_path):
    make_cache(tmp_path / "source")
    p = make_prepared(tmp_path / "source")
    (p / "timestamps.npy").unlink()
    with pytest.raises(CacheTransferError, match="artifact is missing"):
        export_cache(
            tmp_path / "source",
            tmp_path / "bundle",
            prepared_datasets=["prepared-example"],
        )
    assert not (tmp_path / "bundle").exists()


def test_prepared_benchmark_venue_must_be_in_snapshot_selection(tmp_path):
    make_cache(tmp_path / "source")
    make_prepared(tmp_path / "source", exchange="bybit")
    with pytest.raises(CacheTransferError, match="unselected exchange"):
        export_cache(
            tmp_path / "source",
            tmp_path / "bundle",
            exchanges=["binance"],
            prepared_datasets=["prepared-example"],
        )


def test_raw_catalog_path_escape_rejected_even_with_valid_container_hash(tmp_path):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    db = bundle / "payload/ohlcvs/catalog.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("UPDATE chunks SET body_path='../escape.npy'")
    p = bundle / "manifest.json"
    m = json.loads(p.read_text())
    m["files"]["ohlcvs/catalog.sqlite"].update(
        bytes=db.stat().st_size, sha256=data_cache._sha(db)
    )
    p.write_text(json.dumps(m))
    with pytest.raises(CacheTransferError, match="path"):
        import_cache(bundle, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_extra_catalog_store_file_is_not_imported(tmp_path):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    p = bundle / "payload/ohlcvs/unrelated.py"
    p.write_text("not cache data")
    manifest_path = bundle / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["files"]["ohlcvs/unrelated.py"] = {
        "sha256": data_cache._sha(p),
        "bytes": p.stat().st_size,
        "mtime_ns": p.stat().st_mtime_ns,
    }
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(CacheTransferError, match="unsupported raw-store"):
        import_cache(bundle, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_conflicting_catalog_fact_is_not_silently_overwritten(tmp_path):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    target = tmp_path / "target"
    import_cache(bundle, target)
    with sqlite3.connect(target / "ohlcvs/catalog.sqlite") as c:
        c.execute("UPDATE gaps SET note='different factual evidence'")
    before = snapshot(target)
    with pytest.raises(CacheTransferError, match="catalog facts conflict"):
        import_cache(bundle, target)
    assert snapshot(target) == before


def test_catalog_write_during_export_rejects_snapshot(tmp_path, monkeypatch):
    catalog, _, start, _ = make_cache(tmp_path / "source")
    copy = data_cache._copy_metadata

    def changing(*args):
        copy(*args)
        catalog.record_fetch_attempt(
            exchange="binance",
            timeframe="1m",
            symbol="BTC/USDT:USDT",
            start_ts=start,
            end_ts=start + 60000,
            attempt=2,
            outcome="new fact",
        )

    monkeypatch.setattr(data_cache, "_copy_metadata", changing)
    with pytest.raises(CacheTransferError, match="catalog changed"):
        export_cache(tmp_path / "source", tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_active_destination_chunk_writer_prevents_import(tmp_path):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    target = tmp_path / "target"
    import_cache(bundle, target)
    c = OhlcvCatalog(target / "ohlcvs/catalog.sqlite")
    chunk = c.list_chunks(
        "binance",
        "1m",
        "BTC/USDT:USDT",
        month_start_ts(2026, 4),
        month_start_ts(2026, 4),
    )[0]
    before = snapshot(target)
    with portalocker.Lock(chunk.body_path + ".lock", timeout=0, fail_when_locked=True):
        with pytest.raises(
            CacheTransferError, match="destination chunk is being written"
        ):
            import_cache(bundle, target)
    assert snapshot(target) == before


def test_alias_records_sharing_a_canonical_chunk_lock_roundtrip(tmp_path):
    catalog, _, _, _ = make_cache(tmp_path / "source")
    conn = catalog._connect()
    cols = [x[1] for x in conn.execute("PRAGMA table_info(chunks)")]
    row = dict(conn.execute("SELECT * FROM chunks").fetchone())
    row["symbol"] = "BTC/USDT_USDT"
    conn.execute(
        "INSERT INTO chunks ("
        + ",".join(cols)
        + ") VALUES ("
        + ",".join("?" for _ in cols)
        + ")",
        [row[c] for c in cols],
    )
    conn.commit()
    export_cache(tmp_path / "source", tmp_path / "bundle")
    import_cache(tmp_path / "bundle", tmp_path / "target")
    with sqlite3.connect(tmp_path / "target/ohlcvs/catalog.sqlite") as c:
        assert c.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2


def test_interruption_after_link_does_not_leave_factual_files(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    install = data_cache._install_file

    def after_link(src, dst):
        install(src, dst)
        raise KeyboardInterrupt()

    monkeypatch.setattr(data_cache, "_install_file", after_link)
    with pytest.raises(KeyboardInterrupt):
        import_cache(tmp_path / "bundle", tmp_path / "target")
    after = snapshot(tmp_path / "target")
    assert set(after) == {"__catalog_rows__"}
    assert all(not rows for rows in after["__catalog_rows__"].values())
    monkeypatch.setattr(data_cache, "_install_file", install)
    assert import_cache(tmp_path / "bundle", tmp_path / "target")["installed_files"] > 0


def test_prepared_source_logical_hash_is_verified_before_export(tmp_path):
    from hlcvs_manifest import HlcvsManifestError

    make_cache(tmp_path / "source")
    prepared = make_prepared(tmp_path / "source")
    np.save(prepared / "hlcvs.npy", np.ones((2, 1)))
    with pytest.raises(HlcvsManifestError, match="hash mismatch"):
        export_cache(
            tmp_path / "source",
            tmp_path / "bundle",
            prepared_datasets=["prepared-example"],
        )
    assert not (tmp_path / "bundle").exists()


def test_container_checksum_does_not_replace_prepared_owner_validation(tmp_path):
    from hlcvs_manifest import HlcvsManifestError

    make_cache(tmp_path / "source")
    make_prepared(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle, prepared_datasets=["prepared-example"])
    artifact = bundle / "payload/hlcvs_data/prepared-example/hlcvs.npy"
    np.save(artifact, np.ones((2, 1)))
    p = bundle / "manifest.json"
    m = json.loads(p.read_text())
    m["files"]["hlcvs_data/prepared-example/hlcvs.npy"].update(
        bytes=artifact.stat().st_size, sha256=data_cache._sha(artifact)
    )
    p.write_text(json.dumps(m))
    with pytest.raises(HlcvsManifestError, match="hash mismatch"):
        import_cache(bundle, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_prepared_candidate_venues_are_required_even_if_not_selected_for_prices(
    tmp_path,
):
    make_cache(tmp_path / "source")
    prepared = make_prepared(tmp_path / "source")
    p = prepared / "manifest.json"
    m = json.loads(p.read_text())
    m["requested"] = {"exchanges": ["binance", "bybit"]}
    p.write_text(json.dumps(m))
    with pytest.raises(CacheTransferError, match="unselected exchange"):
        export_cache(
            tmp_path / "source",
            tmp_path / "bundle",
            exchanges=["binance"],
            prepared_datasets=["prepared-example"],
        )


def test_bundle_manifest_symlink_is_rejected(tmp_path):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    manifest = bundle / "manifest.json"
    manifest.rename(tmp_path / "outside-manifest.json")
    manifest.symlink_to(tmp_path / "outside-manifest.json")
    with pytest.raises(CacheTransferError, match="symlink"):
        import_cache(bundle, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_concurrent_import_lease_refuses_second_import(tmp_path):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    lease = tmp_path / ".target.data-cache-import.lock"
    with portalocker.Lock(str(lease), timeout=0, fail_when_locked=True):
        with pytest.raises(CacheTransferError, match="another import"):
            import_cache(bundle, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_catalog_created_after_preflight_is_never_deleted(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    merge = data_cache._merge_rows
    called = 0

    def creating_catalog(dest, incoming, root):
        nonlocal called
        merge(dest, incoming, root)
        called += 1
        if called == 1:
            make_cache(root, exchange="bybit", symbol="ETH/USDT:USDT")

    monkeypatch.setattr(data_cache, "_merge_rows", creating_catalog)
    with pytest.raises(CacheTransferError, match="appeared during import"):
        import_cache(bundle, tmp_path / "target")
    catalog = OhlcvCatalog(tmp_path / "target/ohlcvs/catalog.sqlite")
    assert (
        catalog._connect()
        .execute("SELECT exchange,symbol FROM chunks")
        .fetchall()[0][0]
        == "bybit"
    )
    assert catalog._connect().execute("SELECT count(*) FROM chunks").fetchone()[0] == 1


def test_interrupted_new_catalog_initialization_can_retry(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    target = tmp_path / "target"
    catalog_path = target / "ohlcvs/catalog.sqlite"
    owner = data_cache.OhlcvCatalog

    def interrupted_catalog(path, *args, **kwargs):
        if Path(path) == catalog_path:
            raise KeyboardInterrupt
        return owner(path, *args, **kwargs)

    monkeypatch.setattr(data_cache, "OhlcvCatalog", interrupted_catalog)
    with pytest.raises(KeyboardInterrupt):
        import_cache(bundle, target)
    assert catalog_path.read_bytes() == b""
    monkeypatch.setattr(data_cache, "OhlcvCatalog", owner)
    assert import_cache(bundle, target)["installed_files"] > 0
    assert (
        owner(catalog_path)
        ._connect()
        .execute("SELECT count(*) FROM chunks")
        .fetchone()[0]
        == 1
    )


def test_many_chunks_transfer_with_low_descriptor_limit(tmp_path):
    pytest.importorskip("resource")
    source = tmp_path / "source"
    for index in range(75):
        catalog, _, _, _ = make_cache(source, symbol=f"COIN{index}/USDT:USDT")
        catalog._connect().close()
    code = """
import resource
import sys
from pathlib import Path
from data_cache import export_cache, import_cache
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
limit = 64 if hard == resource.RLIM_INFINITY else min(64, hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
root = Path(sys.argv[1])
export_cache(root / 'source', root / 'bundle')
import_cache(root / 'bundle', root / 'target')
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
    with sqlite3.connect(tmp_path / "target/ohlcvs/catalog.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 75


def test_every_publication_is_staged_beside_its_target(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    link = os.link
    published = []

    def mounted_link(source, target):
        # Simulate independent mounts at artifact directories, without mounts or
        # privileges. A root/parent staging directory would raise EXDEV here.
        if Path(source).parent != Path(target).parent:
            raise OSError(errno.EXDEV, "cross-device publication")
        published.append(Path(target))
        link(source, target)

    monkeypatch.setattr(data_cache.os, "link", mounted_link)
    assert import_cache(tmp_path / "bundle", tmp_path / "target")["installed_files"]
    assert published
    assert not list((tmp_path / "target").rglob(".data-cache-publish-*"))


def test_partial_owner_schema_retry_preserves_existing_facts(tmp_path):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    target = tmp_path / "target/ohlcvs/catalog.sqlite"
    target.parent.mkdir(parents=True)
    canonical = OhlcvCatalog(":memory:")
    sql = (
        canonical._connect()
        .execute("SELECT sql FROM sqlite_master WHERE name='symbols'")
        .fetchone()[0]
    )
    canonical._connect().close()
    with sqlite3.connect(target) as conn:
        conn.execute(sql)
        conn.execute(
            "INSERT INTO symbols VALUES ('bybit','1m','OTHER/USDT:USDT',1,2,3)"
        )
    assert target.stat().st_size > 0
    import_cache(tmp_path / "bundle", tmp_path / "target")
    with sqlite3.connect(target) as conn:
        assert conn.execute(
            "SELECT first_ts,last_ts,updated_at FROM symbols WHERE exchange='bybit'"
        ).fetchone() == (1, 2, 3)
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1


def test_unrecognized_partial_schema_is_not_repaired(tmp_path):
    make_cache(tmp_path / "source")
    export_cache(tmp_path / "source", tmp_path / "bundle")
    target = tmp_path / "target/ohlcvs/catalog.sqlite"
    target.parent.mkdir(parents=True)
    with sqlite3.connect(target) as conn:
        conn.execute("CREATE TABLE symbols(unrelated TEXT)")
    before = target.read_bytes()
    with pytest.raises(CacheTransferError, match="unsupported catalog schema"):
        import_cache(tmp_path / "bundle", tmp_path / "target")
    assert target.read_bytes() == before


def test_overlapping_source_gaps_roundtrip_and_complete_a_partial_import(tmp_path):
    catalog, _, start, _ = make_cache(tmp_path / "source")
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=start + 120000,
        end_ts=start + 240000,
        reason="unknown",
        persistent=True,
        note="independent overlapping observation",
    )
    export_cache(tmp_path / "source", tmp_path / "bundle")
    target = tmp_path / "target"
    import_cache(tmp_path / "bundle", target)
    assert (
        snapshot(target)["__catalog_rows__"]["gaps"]
        == snapshot(tmp_path / "source")["__catalog_rows__"]["gaps"]
    )
    with sqlite3.connect(target / "ohlcvs/catalog.sqlite") as conn:
        conn.execute("DELETE FROM gaps WHERE end_ts=?", (start + 240000,))
    import_cache(tmp_path / "bundle", target)
    before = snapshot(target)
    assert import_cache(tmp_path / "bundle", target)["installed_files"] == 0
    assert snapshot(target) == before


def test_foreign_destination_overlap_still_refuses_new_gap_facts(tmp_path):
    catalog, _, start, _ = make_cache(tmp_path / "source")
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=start + 120000,
        end_ts=start + 240000,
        reason="unknown",
        persistent=True,
        note="source extended observation",
    )
    export_cache(tmp_path / "source", tmp_path / "bundle")
    target = tmp_path / "target"
    import_cache(tmp_path / "bundle", target)
    with sqlite3.connect(target / "ohlcvs/catalog.sqlite") as conn:
        conn.execute("DELETE FROM gaps WHERE end_ts=?", (start + 240000,))
    target_catalog = OhlcvCatalog(target / "ohlcvs/catalog.sqlite")
    target_catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=start + 200000,
        end_ts=start + 300000,
        reason="unknown",
        persistent=True,
        note="new destination observation",
    )
    before = snapshot(target)
    with pytest.raises(CacheTransferError, match="overlaps incoming"):
        import_cache(tmp_path / "bundle", target)
    assert snapshot(target) == before


def test_released_source_chunk_lock_does_not_hide_later_mutation(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    copy = data_cache._copy_metadata

    def changing(*args):
        copy(*args)
        body = next((tmp_path / "source/ohlcvs/data").rglob("04.npy"))
        with body.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"x")

    monkeypatch.setattr(data_cache, "_copy_metadata", changing)
    with pytest.raises(CacheTransferError, match="source chunks changed"):
        export_cache(tmp_path / "source", tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_bounded_destination_locks_still_reject_changed_existing_chunks(
    tmp_path, monkeypatch
):
    make_cache(tmp_path / "source")
    bundle, target = tmp_path / "bundle", tmp_path / "target"
    export_cache(tmp_path / "source", bundle)
    import_cache(bundle, target)
    missing = target / "symbol_to_coin_map.json"
    missing.unlink()
    rows_before = snapshot(target)["__catalog_rows__"]
    install = data_cache._install_file

    def changing(source, destination):
        install(source, destination)
        body = next((target / "ohlcvs/data").rglob("04.npy"))
        with body.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"x")

    monkeypatch.setattr(data_cache, "_install_file", changing)
    with pytest.raises(CacheTransferError, match="destination changed"):
        import_cache(bundle, target)
    assert not missing.exists()
    assert snapshot(target)["__catalog_rows__"] == rows_before


def test_same_filesystem_publication_moves_only_disposable_stage(tmp_path, monkeypatch):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    manifest = export_cache(tmp_path / "source", bundle)
    copy = data_cache._copy_stable
    copies = []

    def counted(source, target):
        copies.append((source, target))
        return copy(source, target)

    monkeypatch.setattr(data_cache, "_copy_stable", counted)
    import_cache(bundle, tmp_path / "target")
    # Full preflight stages every bundle file once. Same-filesystem publication
    # moves that disposable copy, without copying the whole history again.
    assert len(copies) == len(manifest["files"])
    for name, facts in manifest["files"].items():
        original = bundle / "payload" / name
        assert original.is_file()
        assert data_cache._sha(original) == facts["sha256"]
        assert original.stat().st_mtime_ns == facts["mtime_ns"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_cross_filesystem_staging_preserves_mtime_and_rollback(
    tmp_path, monkeypatch, interrupted
):
    make_cache(tmp_path / "source")
    bundle, target = tmp_path / "bundle", tmp_path / "target"
    manifest = export_cache(tmp_path / "source", bundle)
    replace = os.replace
    install = data_cache._install_file
    fallbacks = []

    def separate_mount(source, destination):
        if Path(destination).name.startswith(".data-cache-publish-"):
            fallbacks.append(source)
            raise OSError(errno.EXDEV, "fixture destination mount")
        replace(source, destination)

    installed = 0

    def interrupted_install(source, destination):
        nonlocal installed
        install(source, destination)
        installed += 1
        if installed == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(data_cache.os, "replace", separate_mount)
    if interrupted:
        monkeypatch.setattr(data_cache, "_install_file", interrupted_install)
        with pytest.raises(KeyboardInterrupt):
            import_cache(bundle, target)
        after = snapshot(target)
        assert set(after) == {"__catalog_rows__"}
        assert all(not rows for rows in after["__catalog_rows__"].values())
    else:
        import_cache(bundle, target)
        for name, facts in manifest["files"].items():
            if name != "ohlcvs/catalog.sqlite":
                assert (target / name).stat().st_mtime_ns == facts["mtime_ns"]
    assert fallbacks
    assert all(not source.exists() for source in fallbacks)
    assert not list(target.rglob(".data-cache-publish-*"))
    assert all(
        data_cache._sha(bundle / "payload" / name) == facts["sha256"]
        for name, facts in manifest["files"].items()
    )


@pytest.mark.parametrize("stage", ["lease", "preflight"])
def test_nonwritable_cache_parent_has_actionable_diagnostic(
    tmp_path, monkeypatch, stage
):
    make_cache(tmp_path / "source")
    bundle = tmp_path / "bundle"
    export_cache(tmp_path / "source", bundle)
    target = tmp_path / "target"
    target.mkdir()
    if stage == "lease":
        owner_lock = portalocker.Lock

        def denied(filename, *args, **kwargs):
            if str(filename).endswith(".data-cache-import.lock"):
                raise PermissionError("fixture nonwritable parent")
            return owner_lock(filename, *args, **kwargs)

        monkeypatch.setattr(data_cache.portalocker, "Lock", denied)
    else:
        owner_temp = data_cache.tempfile.TemporaryDirectory

        def denied(*args, **kwargs):
            if kwargs.get("prefix") == ".data-cache-import-":
                raise PermissionError("fixture nonwritable parent")
            return owner_temp(*args, **kwargs)

        monkeypatch.setattr(data_cache.tempfile, "TemporaryDirectory", denied)
    with pytest.raises(
        CacheTransferError,
        match="cache-root parent must be writable.*preflight staging",
    ):
        import_cache(bundle, target)
    assert not snapshot(target)
