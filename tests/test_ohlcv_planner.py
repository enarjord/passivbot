import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import resolve_legacy_symbol_dir
from ohlcv_planner import plan_local_symbol_range
from ohlcv_store import OhlcvStore, month_start_ts


LEGACY_DTYPE = np.dtype(
    [
        ("ts", "int64"),
        ("o", "float32"),
        ("h", "float32"),
        ("l", "float32"),
        ("c", "float32"),
        ("bv", "float32"),
    ]
)


def _write_legacy_day(root, exchange, symbol, day, rows):
    symbol_dir = resolve_legacy_symbol_dir(root, exchange, "1m", symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    np.save(symbol_dir / f"{day}.npy", np.array(rows, dtype=LEGACY_DTYPE))


def test_plan_local_symbol_range_prefers_store_complete(tmp_path):
    root = tmp_path / "caches" / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 60_000], dtype=np.int64)
    vals = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "ETH/USDT:USDT", ts, vals)

    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=tmp_path / "legacy",
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 60_000),
    )

    assert plan.status == "store_complete"
    assert plan.local_store_complete is True


def test_plan_local_symbol_range_persistent_gap_wins_over_sparse_bounds(tmp_path):
    root = tmp_path / "caches" / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 2 * 60_000], dtype=np.int64)
    vals = np.array([[101.0, 99.0, 100.0, 10.0], [103.0, 101.0, 102.0, 12.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "ETH/USDT:USDT", ts, vals)
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start + 60_000),
        end_ts=int(start + 60_000),
        reason="exchange_outage",
        persistent=True,
    )

    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=tmp_path / "legacy",
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 2 * 60_000),
    )

    assert plan.status == "blocked_by_persistent_gap"
    assert plan.blocked_by_persistent_gap is True
    assert len(plan.persistent_gaps) == 1


def test_plan_local_symbol_range_marks_legacy_importable(tmp_path):
    root = tmp_path / "caches" / "ohlcvs"
    legacy_root = tmp_path / "legacy"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    start = month_start_ts(2026, 4)
    _write_legacy_day(
        legacy_root,
        "binance",
        "ETH/USDT:USDT",
        "2026-04-01",
        [(int(start), 0.0, 101.0, 99.0, 100.0, 10.0)],
    )

    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=legacy_root,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start),
    )

    assert plan.status == "legacy_importable"
    assert plan.should_try_legacy_import is True
    assert plan.legacy_inspection is not None
    assert plan.legacy_inspection.all_days_present is True


def test_plan_local_symbol_range_allows_legacy_import_despite_persistent_gap(tmp_path):
    root = tmp_path / "caches" / "ohlcvs"
    legacy_root = tmp_path / "legacy"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    start = month_start_ts(2026, 4)
    _write_legacy_day(
        legacy_root,
        "binance",
        "ETH/USDT:USDT",
        "2026-04-01",
        [(int(start), 0.0, 101.0, 99.0, 100.0, 10.0)],
    )
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start),
        reason="exchange_outage",
        persistent=True,
    )

    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=legacy_root,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start),
    )

    assert plan.status == "legacy_importable"
    assert plan.should_try_legacy_import is True
    assert len(plan.persistent_gaps) == 1


def test_plan_local_symbol_range_marks_persistent_gap_block(tmp_path):
    root = tmp_path / "caches" / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    start = month_start_ts(2026, 4)
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 60_000),
        reason="exchange_outage",
        persistent=True,
    )

    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=tmp_path / "legacy",
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 60_000),
    )

    assert plan.status == "blocked_by_persistent_gap"
    assert plan.blocked_by_persistent_gap is True
    assert len(plan.persistent_gaps) == 1


def test_plan_local_symbol_range_marks_missing_local_when_no_store_or_legacy(tmp_path):
    root = tmp_path / "caches" / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    start = month_start_ts(2026, 4)

    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=tmp_path / "legacy",
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 60_000),
    )

    assert plan.status == "missing_local"
    assert plan.requires_remote_fetch is True
