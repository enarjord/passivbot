import math
import weakref

import numpy as np
import pytest

from optimization.gpu import residency, service


def _error(values, starts, tails):
    with pytest.raises(ValueError) as raised:
        service._require_supported_multicoin_valid_tails(values, starts, tails)
    return str(raised.value)


def test_metadata_failure_precedes_every_candle_conversion(monkeypatch):
    values = np.full((30, 2, 4), np.nan)
    original = np.asarray
    def no_conversion(value, *args, **kwargs):
        assert kwargs.get("dtype") != np.float32
        return original(value, *args, **kwargs)
    monkeypatch.setattr(service.np, "asarray", no_conversion)
    assert _error(values, [0, 7], [29, 30]) == (
        "GPU multicoin proxy requires each first_valid_idx within its prepared valid range; "
        "coin=1, first_valid_idx=7, last_valid_idx=30, candle_count=30"
    )


@pytest.mark.parametrize("bad_coin", [1, 2])
@pytest.mark.parametrize("bad_value", [np.nan, np.nextafter(0.0, 1.0), np.finfo(np.float64).max])
def test_all_forced_delist_endpoints_precede_first_valid_and_interior_failures(bad_coin, bad_value):
    values = np.ones((1605, 3, 4))
    tails = [100, 99, 98]
    values[0, 0, 0] = np.nan  # Earlier first-valid failure cannot preempt a delist.
    values[1, 0, 0] = np.nan  # Nor can an earlier interior failure.
    values[tails[2], 2, :3] = [1.234567891, 2.345678912, bad_value]
    if bad_coin == 1:
        values[tails[1], 1, :3] = [1.234567891, 2.345678912, bad_value]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = np.asarray(values[tails[bad_coin], bad_coin, :3], dtype=np.float32).tolist()
    assert _error(values, [0, 0, 0], tails) == (
        "GPU multicoin proxy requires each forced-delist final candle's packed float32 "
        "H/L/C values to remain finite and positive; "
        f"coin={bad_coin}, last_valid_idx={tails[bad_coin]}, packed_hlc={expected}"
    )


@pytest.mark.parametrize("bad_coin", [0, 1])
def test_first_valid_endpoints_preserve_coin_order_before_interior_checks(bad_coin):
    values = np.ones((200, 3, 4))
    starts, tails = [100, 20, 0], [199, 199, 199]
    values[1, 2, 0] = np.nan
    values[20, 1, :3] = [1.0, 0.0, 1.0]
    if bad_coin == 0:
        values[100, 0, :3] = [1.0, 0.0, 1.0]
    assert _error(values, starts, tails) == (
        "GPU multicoin proxy requires each first-valid candle's packed float32 "
        "H/L/C values to remain finite and positive; "
        f"coin={bad_coin}, first_valid_idx={starts[bad_coin]}, packed_hlc=[1.0, 0.0, 1.0]"
    )


@pytest.mark.parametrize("bad_cells,expected", [
    ([(8, 0), (7, 2)], (7, 2)),
    ([(7, 2), (7, 1)], (7, 1)),
    ([(6, 2), (7, 0)], (6, 2)),
])
@pytest.mark.parametrize("order", ["C", "F"])
def test_interior_failure_is_first_in_candle_coin_order_across_chunks(monkeypatch, bad_cells, expected, order):
    values = np.ones((37, 3, 4), order=order)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 3 * (3 * 22 + 8))
    for candle, coin in bad_cells:
        values[candle, coin, 1] = np.nan
    candle, coin = expected
    assert _error(values, [0, 0, 0], [36, 36, 36]) == (
        "GPU multicoin proxy requires contiguous finite positive float32 H/L/C inside "
        "every declared coin window; non-finite, finite but non-positive or "
        "float32-unrepresentable values remain fail-closed; "
        f"coin index {coin}, candle_index={candle}, valid_windows={[(0, 36)] * 3}"
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strided_windows_and_empty_coin_sentinels_ignore_outside_values(monkeypatch, dtype):
    backing = np.full((74, 8, 8), np.nan, dtype=dtype)
    values = backing[::2, ::2, ::2]
    starts, tails = [37, 0, 5, 10], [36, 20, 30, 36]
    for coin, (first, last) in enumerate(zip(starts, tails)):
        values[first:last + 1, coin, :3] = 1.0
    values[:, :, 3] = np.inf  # Volume is outside this H/L/C contract.
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 100)
    service._require_supported_multicoin_valid_tails(values, starts, tails)


def test_conversions_masks_and_indices_are_bounded_and_chunks_do_not_overlap(monkeypatch):
    candles, coins, chunk_rows = 97, 5, 3
    values = np.ones((candles, coins, 4), dtype=np.float64)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", chunk_rows * (coins * 22 + 8))
    original_asarray, original_isfinite, original_arange = np.asarray, np.isfinite, np.arange
    converted, finite_shapes, index_counts = [], [], []
    previous = None
    def asarray(value, *args, **kwargs):
        nonlocal previous
        if kwargs.get("dtype") == np.float32 and value.ndim == 3:
            assert previous is None or previous() is None
            result = original_asarray(value, *args, **kwargs)
            converted.append(result.shape)
            previous = weakref.ref(result)
            return result
        return original_asarray(value, *args, **kwargs)
    def isfinite(value, *args, **kwargs):
        finite_shapes.append(value.shape)
        return original_isfinite(value, *args, **kwargs)
    def arange(*args, **kwargs):
        result = original_arange(*args, **kwargs)
        index_counts.append(result.size)
        return result
    def no_argwhere(*args, **kwargs):
        pytest.fail("validation must not allocate an index pair for every invalid candle")
    monkeypatch.setattr(service.np, "asarray", asarray)
    monkeypatch.setattr(service.np, "isfinite", isfinite)
    monkeypatch.setattr(service.np, "arange", arange)
    monkeypatch.setattr(service.np, "argwhere", no_argwhere)
    service._require_supported_multicoin_valid_tails(values, [0] * coins, [candles - 1] * coins)
    assert len(converted) == math.ceil(candles / chunk_rows)
    assert all(shape[0] <= chunk_rows and shape[1:] == (coins, 3) for shape in converted)
    assert all(len(shape) == 1 or shape[0] <= chunk_rows for shape in finite_shapes)
    assert max(index_counts) <= chunk_rows
    assert previous() is None
