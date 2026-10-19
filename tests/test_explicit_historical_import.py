"""Legacy imports are explicit and cannot replace current valid market data."""

import hashlib
import json
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
    assert report["rows_would_import"] == 2
    assert report["rows_imported"] == 0
    assert not target.exists()
    assert path.read_bytes() == before


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
