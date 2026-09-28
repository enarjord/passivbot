"""Exact candle and gap-write equivalence; entirely offline, without timing gates."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from candlestick_manager import CandlestickManager, CANDLE_DTYPE, ONE_MIN_MS, _sorted_candle_copy


def rows(timestamps, offset=0.0):
    return np.array([(t, i+offset, i+1., i-1., -0.0, float('nan'))
                     for i, t in enumerate(timestamps)], dtype=CANDLE_DTYPE)


def reference_merge(a, b):
    if not len(a):
        return _sorted_candle_copy(b)
    if not len(b):
        return _sorted_candle_copy(a)
    combo = np.concatenate([a, b])
    combo = combo[np.argsort(combo['ts'], kind='stable')]
    keep = np.ones(len(combo), dtype=bool)
    keep[:-1] = combo['ts'][:-1] != combo['ts'][1:]
    return combo[keep]


@pytest.mark.parametrize('old,new', [
    ([], [3, 1, 1]), ([3, 1, 1], []), ([], []),
    ([1, 2, 3], [4, 5]), ([1, 2, 3], [3]), ([1, 2, 3], [3, 4, 7]),
    ([1], [1]), ([1, 2, 2], [3]), ([1, 2], [2, 2, 3]),
    ([3, 1, 2], [4]), ([1, 2, 3], [2, 4]), ([3, 4], [1, 2]),
])
def test_merge_preserves_exact_rows_and_detaches(old, new):
    a, b = rows(old), rows(new, 100.)
    a.flags.writeable = b.flags.writeable = False
    actual = CandlestickManager._merge_overwrite(None, a, b)
    assert actual.tobytes() == reference_merge(a, b).tobytes()
    assert not np.shares_memory(actual, a)
    assert not np.shares_memory(actual, b)
    assert actual.flags.writeable


def test_merge_randomized_permutations_and_duplicates():
    rng = np.random.default_rng(931)
    for _ in range(500):
        a, b = (rows(rng.integers(-10, 100, size=rng.integers(0, 100)), shift)
                for shift in (0., 200.))
        assert CandlestickManager._merge_overwrite(None, a, b).tobytes() == reference_merge(a, b).tobytes()


def test_ordered_tail_does_not_sort(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('ordered tail update must not sort the history')
    monkeypatch.setattr(np, 'argsort', fail)
    for ts in ([999], [999, 1000], [1000, 1001]):
        result = CandlestickManager._merge_overwrite(None, rows(range(1000)), rows(ts, 2000.))
        assert result[-1]['o'] == 2000.+len(ts)-1


def reference_trim(gaps, arr):
    timestamps = np.unique(arr['ts'])
    retained, changed = [], False
    for gap in gaps:
        start, end = int(gap['start_ts']), int(gap['end_ts'])
        covered = timestamps[np.searchsorted(timestamps, start):np.searchsorted(timestamps, end, side='right')]
        if not len(covered):
            retained.append(gap)
            continue
        changed = True
        next_start = start
        for ts in covered:
            if next_start <= ts - ONE_MIN_MS:
                retained.append({**gap, 'start_ts': next_start, 'end_ts': int(ts)-ONE_MIN_MS})
            next_start = int(ts)+ONE_MIN_MS
        if next_start <= end:
            retained.append({**gap, 'start_ts': next_start, 'end_ts': end})
    return changed, retained


@pytest.mark.parametrize('defer', [False, True])
def test_gap_trim_matches_reference_and_preserves_metadata(defer):
    rng = np.random.default_rng(953)
    for _ in range(100):
        gaps = [dict(start_ts=int(start), end_ts=int(start+length), retry_count=i,
                     reason='fetch_failed', last_retry_at=i*10, added_at=i)
                for i, (start, length) in enumerate(zip(rng.integers(0, 100, 30)*ONE_MIN_MS,
                                                       rng.integers(0, 10, 30)*ONE_MIN_MS))]
        # Include duplicates, unordered and unaligned timestamps, and empty input.
        arr = rows(rng.integers(0, 110*ONE_MIN_MS, rng.integers(0, 100)))
        before = deepcopy(gaps)
        saved = []
        cm = SimpleNamespace(_get_known_gaps_enhanced=lambda symbol: gaps,
                             _save_known_gaps_enhanced=lambda *args, **kwargs: saved.append((args, kwargs)))
        expected, retained = reference_trim(gaps, arr)
        assert CandlestickManager._trim_known_gaps_covered_by_rows(cm, 'A', arr, defer_index=defer) == expected
        assert saved == [ (('A', retained), {'defer_index': defer}) ] if expected else saved == []
        assert gaps == before


def test_disjoint_gaps_do_not_search_or_write(monkeypatch):
    gaps = [dict(start_ts=i*ONE_MIN_MS, end_ts=i*ONE_MIN_MS) for i in range(3000)]
    def fail(*args, **kwargs):
        pytest.fail('disjoint gaps must not search timestamps or rewrite metadata')
    cm = SimpleNamespace(_get_known_gaps_enhanced=lambda symbol: gaps, _save_known_gaps_enhanced=fail)
    monkeypatch.setattr(np, 'searchsorted', fail)
    for ts in (-ONE_MIN_MS, 3001*ONE_MIN_MS):
        assert not CandlestickManager._trim_known_gaps_covered_by_rows(cm, 'A', rows([ts]))
