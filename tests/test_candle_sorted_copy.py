import numpy as np
import pytest

from candlestick_manager import CANDLE_DTYPE, _sorted_candle_copy


@pytest.mark.parametrize('timestamps', [[], [60000], [0, 60000, 120000], [120000, 0, 60000], [0, 0, 60000], [60000, 0, 60000, 0]])
def test_sorted_copy_matches_numpy_and_never_aliases_input(timestamps):
    # Opposing OHLC values exercise structured-field tie ordering, not only ts.
    arr = np.array([(t, 10.-i, 20.+i, 1., 5.+i, float(i)) for i, t in enumerate(timestamps)], dtype=CANDLE_DTYPE)
    expected = np.sort(arr, order='ts')
    original = arr.copy()
    for source in (arr, arr[::-1]):
        actual = _sorted_candle_copy(source)
        np.testing.assert_array_equal(actual, np.sort(source, order='ts'))
        assert not np.shares_memory(actual, source)
        if actual.size:
            actual['c'][0] = -1.
    np.testing.assert_array_equal(arr, original)
    np.testing.assert_array_equal(_sorted_candle_copy(arr), expected)


def test_large_ordered_input_avoids_sort_but_duplicates_do_not(monkeypatch):
    arr = np.zeros(100_000, dtype=CANDLE_DTYPE)
    arr['ts'] = np.arange(arr.size) * 60000
    original = np.sort
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'sort', counted)
    np.testing.assert_array_equal(_sorted_candle_copy(arr), arr)
    assert calls == []
    arr['ts'][1] = arr['ts'][0]
    np.testing.assert_array_equal(_sorted_candle_copy(arr), original(arr, order='ts'))
    assert calls == [1]


def test_sorted_copy_preserves_nan_values_and_readonly_source():
    arr = np.array([(0, float('nan'), 2., 1., 1., 0.), (0, 1., 2., 1., 1., 0.),
                    (60000, 1., 2., 1., float('nan'), 0.)], dtype=CANDLE_DTYPE)
    arr.flags.writeable = False
    actual = _sorted_candle_copy(arr)
    expected = np.sort(arr, order='ts')
    for field in CANDLE_DTYPE.names:
        np.testing.assert_array_equal(actual[field], expected[field])
    assert actual.flags.writeable
