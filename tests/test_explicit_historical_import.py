"""Legacy imports are explicit and cannot replace current valid market data."""

import hashlib
import json
import sqlite3
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import import_legacy_range_into_store, resolve_legacy_symbol_dir
from ohlcv_store import OhlcvStore, month_start_ts
from tools.import_historical_data import main


EXCHANGE = "binance"
SYMBOL = "BTC/USDT:USDT"
START = month_start_ts(2026, 4)
DTYPE = np.dtype([(name, kind) for name, kind in
                  [("ts", "int64"), ("o", "float32"), ("h", "float32"),
                   ("l", "float32"), ("c", "float32"), ("bv", "float32")]])


def source(root, *, malformed=False):
    path = resolve_legacy_symbol_dir(root, EXCHANGE, "1m", SYMBOL) / "2026-04-01.npy"
    path.parent.mkdir(parents=True)
    rows = [(START, 100, 101, 99, 100, 10), (START + 60_000, 101, 102, 100, 101, 11)]
    if malformed:
        rows[1] = (START + 60_000, 101, 98, 100, 101, 11)
    np.save(path, np.array(rows, dtype=DTYPE))
    return path


def store(root):
    return OhlcvStore(root, OhlcvCatalog(root / "catalog.sqlite"))


def import_rows(target, legacy, **kwargs):
    return import_legacy_range_into_store(
        store=target, legacy_root=legacy, exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL,
        start_ts=START, end_ts=START + 60_000, **kwargs,
    )


def test_explicit_import_preserves_source_current_values_and_is_idempotent(tmp_path):
    legacy = tmp_path / "legacy"
    path = source(legacy)
    before = path.read_bytes()
    target = store(tmp_path / "current")
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([START]), np.array([[201, 199, 200, 20]], dtype=np.float32))
    assert import_rows(target, legacy) == 1
    actual = target.read_range(EXCHANGE, "1m", SYMBOL, START, START + 60_000)
    np.testing.assert_array_equal(actual.valid, [True, True])
    np.testing.assert_array_equal(actual.values, [[201, 199, 200, 20], [102, 100, 101, 11]])
    assert import_rows(target, legacy) == 0
    assert path.read_bytes() == before


def test_import_rejects_malformed_source_before_target_rows_change(tmp_path):
    legacy = tmp_path / "legacy"
    source(legacy, malformed=True)
    target = store(tmp_path / "current")
    with pytest.raises(ValueError, match="invalid or unordered"):
        import_rows(target, legacy)
    assert not target.read_range(EXCHANGE, "1m", SYMBOL, START, START + 60_000).valid.any()


def test_import_rejects_corrupt_current_chunk_instead_of_rehashing_it(tmp_path):
    legacy = tmp_path / "legacy"
    source(legacy)
    target = store(tmp_path / "current")
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([START]), np.array([[201, 199, 200, 20]], dtype=np.float32))
    chunk = target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, START, START)[0]
    body = np.load(chunk.body_path, mmap_mode="r+")
    body[0, 0] = 900
    body.flush()
    del body
    digest = hashlib.sha256(open(chunk.body_path, "rb").read()).hexdigest()
    with pytest.raises(ValueError, match="checksum mismatch"):
        import_rows(target, legacy)
    assert hashlib.sha256(open(chunk.body_path, "rb").read()).hexdigest() == digest
    assert target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, START, START)[0].checksum == chunk.checksum


def test_command_dry_run_does_not_create_target(tmp_path, capsys):
    legacy = tmp_path / "legacy"
    path = source(legacy)
    before = path.read_bytes()
    target = tmp_path / "absent"
    assert main(["--legacy-root", str(legacy), "--store-root", str(target), "--exchange", EXCHANGE,
                 "--symbol", SYMBOL, "--start-ms", str(START), "--end-ms", str(START + 60_000)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows_would_import"] == 1
    assert report["rows_imported"] == 0
    assert not target.exists()
    assert path.read_bytes() == before


def historical_source(root, exchange, day, rows, *, structured=False):
    path = root / f"ohlcvs_{exchange}" / "BTC" / f"{day}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows, dtype=DTYPE if structured else np.float64))
    return path


def tree_snapshot(root):
    """Include lock files and directory metadata, without treating reads as writes."""
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.stat()
        result[str(path.relative_to(root))] = (
            stat.S_IMODE(info.st_mode), info.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
    return result


def close_fixture_catalog(target, *, rollback_journal=False):
    # A closed, checkpointed catalog also works when its whole tree is read-only.
    connection = target.catalog._connect()
    assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    if rollback_journal:
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    connection.close()
    del target.catalog._local.conn
    # Some SQLite builds retain empty WAL/shared-memory files after the last close.
    # Remove only this checkpointed synthetic fixture's now-unused sidecars.
    wal = target.root / "catalog.sqlite-wal"
    if wal.exists():
        assert wal.stat().st_size == 0
        wal.unlink()
    (target.root / "catalog.sqlite-shm").unlink(missing_ok=True)


@pytest.mark.parametrize("structured", [False, True])
def test_historical_gateio_ambiguous_volume_is_rejected_without_changes(tmp_path, structured):
    legacy = tmp_path / "historical"
    # Historical provenance does not certify units in either legacy encoding.
    volumes = (10, 3) if structured else (1000, 600)
    path = historical_source(legacy, "gateio", "2026-04-01", [
        (START, 100, 101, 99, 100, volumes[0]),
        (START + 60_000, 200, 201, 199, 200, volumes[1]),
    ], structured=structured)
    before = path.read_bytes()
    target = store(tmp_path / "current")
    target_before = tree_snapshot(target.root)
    with pytest.raises(ValueError, match="(?i)gateio"):
        import_legacy_range_into_store(
            store=target, legacy_root=legacy, source_layout="historical",
            exchange="gateio", timeframe="1m", symbol=SYMBOL,
            start_ts=START, end_ts=START + 60_000,
        )
    assert tree_snapshot(target.root) == target_before
    assert path.read_bytes() == before


def test_gateio_daily_cache_explicit_base_volume_remains_supported(tmp_path):
    legacy = tmp_path / "legacy"
    directory = resolve_legacy_symbol_dir(legacy, "gateio", "1m", SYMBOL)
    directory.mkdir(parents=True)
    path = directory / "2026-04-01.npy"
    np.save(path, np.array([(START, 100, 101, 99, 100, 10)], dtype=DTYPE))
    before = path.read_bytes()
    target = store(tmp_path / "current")
    assert import_legacy_range_into_store(
        store=target, legacy_root=legacy, exchange="gateio", timeframe="1m", symbol=SYMBOL,
        start_ts=START, end_ts=START,
    ) == 1
    actual = target.read_range("gateio", "1m", SYMBOL, START, START)
    assert actual.valid[0]
    assert actual.values[0, 3] == 10
    assert path.read_bytes() == before


@pytest.mark.parametrize("source_exchange", ["binance", "binanceusdm", "futures"])
def test_historical_cli_resolves_compact_symbol_and_exchange_alias(
    tmp_path, capsys, source_exchange,
):
    legacy = tmp_path / "historical"
    path = historical_source(legacy, source_exchange, "2026-04-01", [
        (START, 100, 101, 99, 100, 10),
        (START + 60_000, 101, 102, 100, 101, 11),
    ])
    before = path.read_bytes()
    root = tmp_path / "current"

    assert main([
        "--legacy-root", str(legacy), "--source-layout", "historical",
        "--store-root", str(root), "--exchange", EXCHANGE, "--symbol", "BTCUSDT",
        "--start-ms", str(START), "--end-ms", str(START + 120_000), "--execute",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["rows_imported"] == 2
    actual = store(root).read_range(EXCHANGE, "1m", "BTCUSDT", START, START + 60_000)
    np.testing.assert_array_equal(actual.valid, [True, True])
    np.testing.assert_allclose(actual.values[:, 3], [10, 11])
    assert path.read_bytes() == before


@pytest.mark.parametrize("execute", [False, True])
def test_cli_excludes_candle_at_exact_midnight_end(tmp_path, capsys, execute):
    legacy = tmp_path / "legacy"
    directory = resolve_legacy_symbol_dir(legacy, EXCHANGE, "1m", SYMBOL)
    directory.mkdir(parents=True)
    end = START + 24 * 60 * 60_000
    first = directory / "2026-04-01.npy"
    second = directory / "2026-04-02.npy"
    np.save(first, np.array([(end - 60_000, 100, 101, 99, 100, 10)], dtype=DTYPE))
    np.save(second, np.array([(end, 200, 201, 199, 200, 20)], dtype=DTYPE))
    before = {path: path.read_bytes() for path in (first, second)}
    root = tmp_path / "current"

    assert main([
        "--legacy-root", str(legacy), "--store-root", str(root),
        "--exchange", EXCHANGE, "--symbol", SYMBOL,
        "--start-ms", str(end - 60_000), "--end-ms", str(end),
        "--execute" if execute else "--dry-run",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows_imported" if execute else "rows_would_import"] == 1
    if execute:
        actual = store(root).read_range(EXCHANGE, "1m", SYMBOL, end - 60_000, end)
        np.testing.assert_array_equal(actual.valid, [True, False])
    else:
        assert not root.exists()
    assert all(path.read_bytes() == original for path, original in before.items())


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("forbid_lock", [False, True])
def test_existing_catalog_dry_run_preserves_entire_target_without_write_lock(
    tmp_path, capsys, monkeypatch, read_only, forbid_lock,
):
    legacy = tmp_path / "legacy"
    source_path = source(legacy)
    source_before = source_path.read_bytes()
    root = tmp_path / "current"
    target = store(root)
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([START]),
                      np.array([[201, 199, 200, 20]], dtype=np.float32))
    close_fixture_catalog(target)
    assert not (root / "catalog.sqlite-wal").exists()
    assert not (root / "catalog.sqlite-shm").exists()
    for lock in root.rglob("*.lock"):
        lock.unlink()

    def forbidden_write_lock(*args, **kwargs):
        raise AssertionError("dry-run must not acquire a store write lock")

    if forbid_lock:
        monkeypatch.setattr(OhlcvStore, "_chunk_write_lock", forbidden_write_lock)
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in [root, *root.rglob("*")]}
    if read_only:
        for path in modes:
            path.chmod(0o555 if path.is_dir() else 0o444)
    before = tree_snapshot(root)
    try:
        assert main([
            "--legacy-root", str(legacy), "--store-root", str(root),
            "--exchange", EXCHANGE, "--symbol", SYMBOL,
            "--start-ms", str(START), "--end-ms", str(START + 120_000), "--dry-run",
        ]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["rows_would_import"] == 1
        assert report["rows_imported"] == 0
        assert tree_snapshot(root) == before
        assert source_path.read_bytes() == source_before
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def test_existing_catalog_dry_run_reads_active_wal_without_changing_target(tmp_path, capsys):
    legacy = tmp_path / "legacy"
    source_path = source(legacy)
    source_before = source_path.read_bytes()
    root = tmp_path / "current"
    target = store(root)
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([START]),
                      np.array([[201, 199, 200, 20]], dtype=np.float32))
    assert (root / "catalog.sqlite-wal").stat().st_size > 0
    before = tree_snapshot(root)

    assert main([
        "--legacy-root", str(legacy), "--store-root", str(root),
        "--exchange", EXCHANGE, "--symbol", SYMBOL,
        "--start-ms", str(START), "--end-ms", str(START + 120_000), "--dry-run",
    ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["rows_would_import"] == 1
    assert report["rows_imported"] == 0
    assert tree_snapshot(root) == before
    assert source_path.read_bytes() == source_before


@pytest.mark.parametrize("malformed", [False, True])
def test_dry_run_closes_sqlite_before_scratch_cleanup_on_success_or_failure(
    tmp_path, capsys, monkeypatch, malformed,
):
    import tools.import_historical_data as command

    legacy = tmp_path / "legacy"
    path = source(legacy, malformed=malformed)
    source_before = path.read_bytes()
    root = tmp_path / "current"
    target = store(root)
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([START]),
                      np.array([[201, 199, 200, 20]], dtype=np.float32))
    close_fixture_catalog(target)
    before = tree_snapshot(root)
    connections = []
    cleanup_observations = []
    original_catalog = command.OhlcvCatalog
    original_temporary_directory = command.TemporaryDirectory

    def capture_catalog(*args, **kwargs):
        catalog = original_catalog(*args, **kwargs)
        connection = catalog._connect()
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        connections.append(connection)
        return catalog

    class CheckedTemporaryDirectory(original_temporary_directory):
        def __exit__(self, *args):
            try:
                assert connections
                for connection in connections:
                    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                        connection.execute("SELECT 1")
                cleanup_observations.append("SQLite closed before deleting scratch")
            finally:
                super().__exit__(*args)

    monkeypatch.setattr(command, "OhlcvCatalog", capture_catalog)
    monkeypatch.setattr(command, "TemporaryDirectory", CheckedTemporaryDirectory)
    argv = [
        "--legacy-root", str(legacy), "--store-root", str(root),
        "--exchange", EXCHANGE, "--symbol", SYMBOL,
        "--start-ms", str(START), "--end-ms", str(START + 120_000), "--dry-run",
    ]
    if malformed:
        with pytest.raises(ValueError, match="invalid or unordered"):
            command.main(argv)
    else:
        assert command.main(argv) == 0
        assert json.loads(capsys.readouterr().out)["rows_would_import"] == 1
    assert cleanup_observations == ["SQLite closed before deleting scratch"]
    assert len(connections) == 1
    assert tree_snapshot(root) == before
    assert path.read_bytes() == source_before


@pytest.mark.parametrize("start,end", [
    (START, START), (START, START - 60_000),
    (START + 1, START + 60_000), (START, START + 60_001),
])
def test_cli_invalid_range_fails_before_target_creation(tmp_path, start, end):
    legacy = tmp_path / "legacy"
    path = source(legacy)
    before = path.read_bytes()
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="range|align|greater|end"):
        main([
            "--legacy-root", str(legacy), "--store-root", str(root),
            "--exchange", EXCHANGE, "--symbol", SYMBOL,
            "--start-ms", str(start), "--end-ms", str(end), "--execute",
        ])
    assert not root.exists()
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["mismatch", "missing", "uncatalogued"])
def test_bad_later_month_checksum_leaves_all_target_months_and_catalog_unchanged(
    tmp_path, corruption,
):
    legacy = tmp_path / "legacy"
    directory = resolve_legacy_symbol_dir(legacy, EXCHANGE, "1m", SYMBOL)
    directory.mkdir(parents=True)
    may = month_start_ts(2026, 5)
    paths = [directory / "2026-04-30.npy", directory / "2026-05-01.npy"]
    for path, timestamp in zip(paths, [may - 60_000, may]):
        np.save(path, np.array([(timestamp, 100, 101, 99, 100, 10)], dtype=DTYPE))
    sources = {path: path.read_bytes() for path in paths}
    root = tmp_path / "current"
    target = store(root)
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([START, may]),
                      np.array([[201, 199, 200, 20], [301, 299, 300, 30]], dtype=np.float32))
    later = target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, may, may)[0]
    if corruption == "mismatch":
        body = np.load(later.body_path, mmap_mode="r+")
        body[0, 0] = 999
        body.flush()
        del body
    elif corruption == "missing":
        with target.catalog._connect() as connection:
            connection.execute("UPDATE chunks SET checksum = NULL WHERE month = 5")
    else:
        with target.catalog._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE month = 5")
    # Keep validation-only reads from changing active WAL reader-slot metadata.
    # Active and closed WAL dry-run behavior is tested separately above.
    close_fixture_catalog(target, rollback_journal=True)
    before = tree_snapshot(root)

    message = "uncatalogued" if corruption == "uncatalogued" else f"checksum {corruption}"
    with pytest.raises(ValueError, match=message):
        import_legacy_range_into_store(
            store=target, legacy_root=legacy, exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL,
            start_ts=may - 60_000, end_ts=may, execute=True,
        )

    assert tree_snapshot(root) == before
    assert all(path.read_bytes() == original for path, original in sources.items())


def test_corrupt_intervening_month_without_incoming_shard_blocks_all_writes(tmp_path):
    legacy = tmp_path / "legacy"
    directory = resolve_legacy_symbol_dir(legacy, EXCHANGE, "1m", SYMBOL)
    directory.mkdir(parents=True)
    march = month_start_ts(2026, 3)
    may = month_start_ts(2026, 5)
    paths = [directory / "2026-03-31.npy", directory / "2026-05-01.npy"]
    for path, timestamp in zip(paths, [START - 60_000, may]):
        np.save(path, np.array([(timestamp, 100, 101, 99, 100, 10)], dtype=DTYPE))
    sources = {path: path.read_bytes() for path in paths}
    target = store(tmp_path / "current")
    target.write_rows(EXCHANGE, "1m", SYMBOL, np.array([march, START, may]),
                      np.array([[201, 199, 200, 20]] * 3, dtype=np.float32))
    intervening = target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, START, START)[0]
    body = np.load(intervening.body_path, mmap_mode="r+")
    body[0, 0] = 999
    body.flush()
    del body
    close_fixture_catalog(target, rollback_journal=True)
    before = tree_snapshot(target.root)

    with pytest.raises(ValueError, match="checksum mismatch"):
        import_legacy_range_into_store(
            store=target, legacy_root=legacy, exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL,
            start_ts=START - 60_000, end_ts=may, execute=True,
        )

    assert tree_snapshot(target.root) == before
    assert all(path.read_bytes() == original for path, original in sources.items())


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_historical_duplicate_rows_are_rejected_without_rewriting_source(tmp_path, suffix):
    legacy = tmp_path / "historical"
    rows = [
        (START, 100, 101, 99, 100, 10),
        (START, 100, 101, 99, 100, 10),
    ]
    if suffix == ".npy":
        path = historical_source(legacy, EXCHANGE, "2026-04-01", rows)
    else:
        path = legacy / "ohlcvs_binance" / "BTC" / "2026-04-01.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, candles=np.array(rows, dtype=DTYPE))
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    target = store(tmp_path / "current")
    try:
        with pytest.raises(ValueError, match="duplicate|ordered"):
            import_legacy_range_into_store(
                store=target, legacy_root=legacy, source_layout="historical",
                exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL,
                start_ts=START, end_ts=START,
            )
    finally:
        assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before
    assert target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, START, START) == []


@pytest.mark.parametrize("timestamp", [START + 0.5, float("nan"), float("inf"), float(2**63)])
def test_historical_invalid_timestamp_is_rejected_before_integer_conversion(tmp_path, timestamp):
    legacy = tmp_path / "historical"
    path = historical_source(legacy, EXCHANGE, "2026-04-01", [
        (timestamp, 100, 101, 99, 100, 10),
    ])
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    target = store(tmp_path / "current")
    with pytest.raises(ValueError, match="timestamp|integer|align|finite|int64"):
        import_legacy_range_into_store(
            store=target, legacy_root=legacy, source_layout="historical",
            exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL,
            start_ts=START, end_ts=START,
        )
    assert target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, START, START) == []
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("execute", [False, True])
def test_import_rejects_cross_month_source_overlap_before_any_write(tmp_path, execute):
    legacy = tmp_path / "legacy"
    directory = resolve_legacy_symbol_dir(legacy, EXCHANGE, "1m", SYMBOL)
    directory.mkdir(parents=True)
    april_end = month_start_ts(2026, 5) - 60_000
    may_start = april_end + 60_000
    first = directory / "2026-04-30.npy"
    second = directory / "2026-05-01.npy"
    np.save(first, np.array([
        (april_end, 100, 101, 99, 100, 10),
        (may_start, 100, 101, 99, 100, 10),
    ], dtype=DTYPE))
    np.save(second, np.array([(may_start, 200, 201, 199, 200, 10)], dtype=DTYPE))
    sources = {path: path.read_bytes() for path in (first, second)}
    target = store(tmp_path / "current")

    with pytest.raises(ValueError, match="overlap or are not chronologically ordered"):
        import_legacy_range_into_store(
            store=target, legacy_root=legacy, exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL,
            start_ts=april_end, end_ts=may_start, execute=execute,
        )

    assert target.catalog.list_chunks(EXCHANGE, "1m", SYMBOL, april_end, may_start) == []
    assert all(path.read_bytes() == before for path, before in sources.items())


def test_startup_never_converts_historical_sources(tmp_path, monkeypatch):
    import candlestick_manager as cm
    import legacy_data_migrator as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("historical conversion attempted during startup")

    for name in ("standardize_cache_directories", "migrate_legacy_data_all_on_init",
                 "merge_duplicate_symbol_directories"):
        monkeypatch.setattr(legacy, name, forbidden)
    history = tmp_path / "historical_data" / "ohlcvs_binance" / "BTC"
    history.mkdir(parents=True)
    np.save(history / "2026-04-01.npy", np.array([[START, 100, 101, 99, 100, 10]], dtype=float))
    before = (history / "2026-04-01.npy").read_bytes()
    manager = cm.CandlestickManager(exchange_name=EXCHANGE, cache_dir=str(tmp_path / "caches"))
    manager.close()
    assert (history / "2026-04-01.npy").read_bytes() == before
    assert not (tmp_path / "caches" / "ohlcv" / ".migration_done").exists()


@pytest.mark.asyncio
async def test_preparation_does_not_import_legacy_shards(tmp_path):
    from hlcv_preparation import _resolve_v2_store_range_impl

    legacy = tmp_path / "legacy"
    path = source(legacy)
    before = path.read_bytes()
    target = store(tmp_path / "current")
    result = await _resolve_v2_store_range_impl(
        om=SimpleNamespace(gap_tolerance_ohlcvs_minutes=120), catalog=target.catalog,
        store=target, exchange=EXCHANGE, coin="BTC", symbol=SYMBOL,
        start_ts=START, end_ts=START + 60_000, allow_remote_fetch=False,
        local_hit_log_label="local", remote_fetch_log_label="remote",
    )
    assert result is None
    assert not target.read_range(EXCHANGE, "1m", SYMBOL, START, START + 60_000).valid.any()
    assert path.read_bytes() == before
