"""Offline differential coverage for historical gap range indexing.

The reference class freezes the prior full-scan algorithms. All metadata, prices,
files and clocks are synthetic; performance checks count work rather than time.
"""
import copy
import json
import random
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

import candlestick_manager as module
from candlestick_manager import (
    CandlestickManager, CANDLE_DTYPE, GapEntry, ONE_MIN_MS, _GAP_MAX_RETRIES,
    GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED, _sorted_candle_copy,
)

SYMBOL = "TEST/USDT:USDT"
NOW = 1_800_000_012_345


class FullScanReference(CandlestickManager):
    """Frozen pre-index algorithms; all supporting policies use production code."""

    def _get_known_gaps_enhanced(self, symbol: str) -> List[GapEntry]:
        """Return known gaps as enhanced GapEntry objects with full metadata."""
        idx = self._ensure_symbol_index(symbol)
        gaps = idx.get("meta", {}).get("known_gaps", [])
        out: List[GapEntry] = []
        now_ms = self._now_ms()
        for it in gaps:
            try:
                # Support both old format [[start, end], ...] and new format [GapEntry, ...]
                if isinstance(it, dict):
                    # New enhanced format
                    entry: GapEntry = {
                        "start_ts": int(it.get("start_ts", 0)),
                        "end_ts": int(it.get("end_ts", 0)),
                        "retry_count": int(it.get("retry_count", 0)),
                        "reason": str(it.get("reason", GAP_REASON_AUTO)),
                        "added_at": int(it.get("added_at", now_ms)),
                        "last_retry_at": int(
                            it.get("last_retry_at", it.get("added_at", now_ms))
                        ),
                        "last_contextual_retry_at": int(
                            it.get("last_contextual_retry_at", 0)
                        ),
                    }
                    if entry["start_ts"] <= entry["end_ts"]:
                        out.append(entry)
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    # Legacy format: auto-upgrade to enhanced
                    a, b = int(it[0]), int(it[1])
                    if a <= b:
                        out.append(
                            {
                                "start_ts": a,
                                "end_ts": b,
                                "retry_count": _GAP_MAX_RETRIES,  # Assume old gaps are persistent
                                "reason": GAP_REASON_AUTO,
                                "added_at": now_ms,
                                "last_retry_at": now_ms,
                            }
                        )
            except Exception:
                continue
        return out

    def _unverified_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[Tuple[int, int]]:
        """Return unresolved unknown gaps that must remain unavailable."""
        ranges: List[Tuple[int, int]] = []
        for gap in self._get_known_gaps_enhanced(symbol):
            if str(gap.get("reason", GAP_REASON_AUTO)) not in {
                GAP_REASON_AUTO,
                GAP_REASON_FETCH_FAILED,
            }:
                continue
            overlap_start = max(int(start_ts), int(gap["start_ts"]))
            overlap_end = min(int(end_ts), int(gap["end_ts"]))
            if overlap_start <= overlap_end:
                ranges.append((overlap_start, overlap_end))
        return ranges

    def _unverified_uncovered_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[Tuple[int, int]]:
        """Return full still-missing spans from unknown gaps overlapping a range.

        Known-gap metadata may temporarily cover minutes which a later refresh
        has already populated. Width-sensitive provisional policy must measure
        the remaining contiguous outage, not either the caller-clipped overlap
        or stale outer metadata bounds.
        """
        cached = self._cache.get(symbol)
        if cached is None or cached.size == 0:
            cached = np.empty((0,), dtype=CANDLE_DTYPE)
        else:
            cached = _sorted_candle_copy(cached)
            provisional_ts = self._synthetic_timestamps.get(symbol, set())
            if provisional_ts:
                cached = cached[
                    ~np.isin(
                        cached["ts"].astype(np.int64),
                        np.asarray(tuple(provisional_ts), dtype=np.int64),
                    )
                ]

        missing: List[Tuple[int, int]] = []
        for gap in self._get_known_gaps_enhanced(symbol):
            if str(gap.get("reason", GAP_REASON_AUTO)) not in {
                GAP_REASON_AUTO,
                GAP_REASON_FETCH_FAILED,
            }:
                continue
            gap_start = int(gap["start_ts"])
            gap_end = int(gap["end_ts"])
            if gap_start > int(end_ts) or gap_end < int(start_ts):
                continue
            gap_rows = self._slice_ts_range(
                cached,
                gap_start,
                gap_end,
                assume_sorted=True,
            )
            missing.extend(self._missing_spans(gap_rows, gap_start, gap_end))

        merged: List[Tuple[int, int]] = []
        for gap_start, gap_end in sorted(missing):
            if merged and int(gap_start) <= int(merged[-1][1]) + ONE_MIN_MS:
                merged[-1] = (
                    int(merged[-1][0]),
                    max(int(merged[-1][1]), int(gap_end)),
                )
            else:
                merged.append((int(gap_start), int(gap_end)))
        return [
            (gap_start, gap_end)
            for gap_start, gap_end in merged
            if gap_start <= int(end_ts) and gap_end >= int(start_ts)
        ]

    def _due_unverified_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        now_ms: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Return retry-due unknown-gap intersections for one attempted range."""
        if now_ms is None:
            now_ms = self._now_ms()
        ranges: List[Tuple[int, int]] = []
        for gap in self._get_known_gaps_enhanced(symbol):
            if str(gap.get("reason", GAP_REASON_AUTO)) not in {
                GAP_REASON_AUTO,
                GAP_REASON_FETCH_FAILED,
            }:
                continue
            if not self._should_retry_gap(gap, now_ms=now_ms):
                continue
            overlap_start = max(int(start_ts), int(gap["start_ts"]))
            overlap_end = min(int(end_ts), int(gap["end_ts"]))
            if overlap_start <= overlap_end:
                ranges.append((overlap_start, overlap_end))
        return ranges

    def _projection_shared_content_signature(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: str,
    ) -> Tuple[Any, ...]:
        """Return content-bearing shared-cache state for a projection window.

        The index is reloaded when another process atomically replaces it.
        Shard checksums and gap coverage affect projection inputs; refresh and
        retry timestamps intentionally do not.
        """
        idx = self._ensure_symbol_index(symbol, timeframe=timeframe)
        shards = idx.get("shards", {})
        shard_state: List[Tuple[Any, ...]] = []
        if isinstance(shards, dict):
            for date_key in self._date_keys_between(start_ts, end_ts):
                shard = shards.get(date_key)
                if not isinstance(shard, dict):
                    continue
                shard_state.append(
                    (
                        str(date_key),
                        shard.get("min_ts"),
                        shard.get("max_ts"),
                        shard.get("count"),
                        shard.get("crc32"),
                    )
                )

        gap_state: List[Tuple[int, int, str]] = []
        known_gaps = idx.get("meta", {}).get("known_gaps", [])
        if isinstance(known_gaps, list):
            for gap in known_gaps:
                try:
                    if isinstance(gap, dict):
                        gap_start = int(gap.get("start_ts", 0))
                        gap_end = int(gap.get("end_ts", 0))
                        reason = str(gap.get("reason", GAP_REASON_AUTO))
                    elif isinstance(gap, (list, tuple)) and len(gap) >= 2:
                        gap_start = int(gap[0])
                        gap_end = int(gap[1])
                        reason = GAP_REASON_AUTO
                    else:
                        continue
                except Exception:
                    continue
                clipped_start = max(int(start_ts), gap_start)
                clipped_end = min(int(end_ts), gap_end)
                if clipped_start <= clipped_end:
                    gap_state.append((clipped_start, clipped_end, reason))

        return (tuple(shard_state), tuple(sorted(gap_state)))

@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Keep every file write and cleanup inside the isolated fixture directory."""
    monkeypatch.chdir(tmp_path)
    cm = CandlestickManager(exchange=None, exchange_name="hyperliquid", cache_dir=str(tmp_path / "caches"))
    cm._now_ms_callback = lambda: NOW
    yield cm
    cm._cleanup_on_exit()


def install(manager, gaps, timeframe="1m"):
    """Exercise the actual JSON ownership boundary rather than injecting a cache."""
    idx = manager._ensure_symbol_index(SYMBOL, timeframe=timeframe)
    idx["meta"]["known_gaps"] = gaps
    manager._save_index(SYMBOL, timeframe=timeframe)
    manager._index.clear()
    return manager._ensure_symbol_index(SYMBOL, timeframe=timeframe)["meta"]["known_gaps"]


def reference(manager):
    """Share factual state while substituting only the frozen full-scan methods."""
    value = copy.copy(manager)
    value.__class__ = FullScanReference
    return value


def compare(manager, start, end, now=NOW):
    """Compare coverage, full missing widths, retries and exact projection keys."""
    before = reference(manager)
    for name in ("_unverified_gap_ranges", "_unverified_uncovered_gap_ranges"):
        assert getattr(manager, name)(SYMBOL, start, end) == getattr(before, name)(SYMBOL, start, end)
    assert manager._due_unverified_gap_ranges(SYMBOL, start, end, now_ms=now) == before._due_unverified_gap_ranges(SYMBOL, start, end, now_ms=now)
    assert manager._projection_shared_content_signature(SYMBOL, start, end, timeframe="1m") == before._projection_shared_content_signature(SYMBOL, start, end, timeframe="1m")
    expected = [gap for gap in before._get_known_gaps_enhanced(SYMBOL)
                if gap["start_ts"] <= end and gap["end_ts"] >= start]
    actual = [gap for gap in manager._get_known_gaps_in_range(SYMBOL, start, end)
              if gap["start_ts"] <= end and gap["end_ts"] >= start]
    assert actual == expected


@pytest.mark.parametrize("seed", range(8))
def test_random_overlap_coverage_retry_and_signature_parity(manager, seed):
    """Unsorted, nested, duplicate and malformed records retain exact semantics."""
    rng = random.Random(seed)
    gaps = []
    for _ in range(180):
        start = rng.randrange(120) * ONE_MIN_MS
        gaps.append(dict(start_ts=start, end_ts=start + rng.randrange(25) * ONE_MIN_MS,
                         reason=rng.choice(["auto_detected", "fetch_failed", "no_trades", "manual", "no_archive"]),
                         retry_count=rng.randrange(5), added_at=NOW - rng.randrange(10**9),
                         last_retry_at=NOW - rng.randrange(10**9),
                         last_contextual_retry_at=NOW - rng.randrange(10**9)))
    gaps += [[0, 5 * ONE_MIN_MS], {"start_ts": 0, "end_ts": 150 * ONE_MIN_MS},
             {}, {"start_ts": "bad", "end_ts": 0}, {"start_ts": 0, "end_ts": 5, "retry_count": "bad"},
             [4, 2], None, "bad", [0], gaps[0].copy()]
    rng.shuffle(gaps)
    install(manager, gaps)
    arr = np.zeros(80, dtype=CANDLE_DTYPE)
    arr["ts"] = np.asarray(rng.sample(range(150), 80)) * ONE_MIN_MS
    arr["c"] = arr["o"] = arr["h"] = arr["l"] = 100.0
    manager._cache[SYMBOL] = arr
    manager._synthetic_timestamps[SYMBOL] = set(int(ts) for ts in arr["ts"][::3])
    for _ in range(60):
        start = rng.randrange(-10, 160) * ONE_MIN_MS
        compare(manager, start, start + rng.randrange(30) * ONE_MIN_MS,
                now=NOW + rng.randrange(10**9))


@pytest.mark.parametrize("mutation", [
    "set", "update", "delete", "pop_key", "popitem", "clear_record", "setdefault", "ior",
    "append", "extend", "insert", "slice", "delete_slice", "pop", "remove", "clear",
    "reverse", "sort", "iadd", "imul", "legacy", "replace_list", "replace_record",
])
def test_owned_metadata_mutations_invalidate_range_index(manager, mutation):
    """Every supported in-place mutation is visible after the range view is warm."""
    values = [dict(start_ts=9 * ONE_MIN_MS, end_ts=10 * ONE_MIN_MS, reason="fetch_failed"),
              dict(start_ts=2 * ONE_MIN_MS, end_ts=3 * ONE_MIN_MS, reason="auto_detected"),
              [4 * ONE_MIN_MS, 5 * ONE_MIN_MS]]
    gaps = install(manager, values)
    compare(manager, 0, 12 * ONE_MIN_MS)
    previous_revision = gaps._revision[0]
    new = dict(start_ts=0, end_ts=ONE_MIN_MS, reason="manual")
    if mutation == "set": gaps[0]["start_ts"] = 0
    elif mutation == "update": gaps[0].update(start_ts=0, reason="no_trades")
    elif mutation == "delete": del gaps[0]["start_ts"]
    elif mutation == "pop_key": gaps[0].pop("end_ts")
    elif mutation == "popitem": gaps[0].popitem()
    elif mutation == "clear_record": gaps[0].clear()
    elif mutation == "setdefault": gaps[0].setdefault("last_retry_at", NOW)
    elif mutation == "ior": gaps[0] |= {"start_ts": 0, "retry_count": 4}
    elif mutation == "append": gaps.append(new)
    elif mutation == "extend": gaps.extend([new])
    elif mutation == "insert": gaps.insert(0, new)
    elif mutation == "slice": gaps[1:] = [new]
    elif mutation == "delete_slice": del gaps[1:]
    elif mutation == "pop": gaps.pop()
    elif mutation == "remove": gaps.remove(gaps[0])
    elif mutation == "clear": gaps.clear()
    elif mutation == "reverse": gaps.reverse()
    elif mutation == "sort": gaps.sort(key=lambda item: str(item))
    elif mutation == "iadd": gaps += [new]
    elif mutation == "imul": gaps *= 2
    elif mutation == "legacy": gaps[2][0] = 0
    elif mutation == "replace_record": gaps[0] = new
    elif mutation == "replace_list":
        # Unowned lists must not acquire a stale identity/length certificate.
        manager._ensure_symbol_index(SYMBOL)["meta"]["known_gaps"] = [new]
        compare(manager, 0, 12 * ONE_MIN_MS)
        new["start_ts"] = 11 * ONE_MIN_MS
        new["end_ts"] = 12 * ONE_MIN_MS
    if mutation != "replace_list": assert gaps._revision[0] > previous_revision
    compare(manager, 0, 12 * ONE_MIN_MS)
    compare(manager, 0, ONE_MIN_MS)
    compare(manager, 8 * ONE_MIN_MS, 11 * ONE_MIN_MS)
    # Wrappers preserve the on-disk JSON schema and survive cold reconstruction.
    expected = json.loads(json.dumps(manager._ensure_symbol_index(SYMBOL)["meta"]["known_gaps"]))
    manager._save_index(SYMBOL)
    manager._index.clear()
    assert manager._ensure_symbol_index(SYMBOL)["meta"]["known_gaps"] == expected
    compare(manager, 0, 12 * ONE_MIN_MS)


def test_legacy_clock_defaults_and_detached_read_results(manager):
    """A warm range view freezes neither missing timestamps nor mutable results."""
    install(manager, [[0, ONE_MIN_MS], {"start_ts": 0, "end_ts": ONE_MIN_MS}])
    initial = manager._get_known_gaps_in_range(SYMBOL, 0, ONE_MIN_MS)
    initial[0]["end_ts"] = 999
    initial[1]["reason"] = "manual"
    manager._now_ms_callback = lambda: NOW + 100_000
    later = manager._get_known_gaps_in_range(SYMBOL, 0, ONE_MIN_MS)
    assert all(g["added_at"] == NOW + 100_000 and g["last_retry_at"] == NOW + 100_000 for g in later)
    assert all(g["end_ts"] == ONE_MIN_MS and g["reason"] == "auto_detected" for g in later)
    compare(manager, 0, ONE_MIN_MS)


@pytest.mark.parametrize("timeframe", ["1m", "5m"])
def test_external_atomic_replacement_and_independent_timeframes(manager, timeframe):
    """The existing shared-index check invalidates the range view after a writer."""
    install(manager, [{"start_ts": 0, "end_ts": ONE_MIN_MS}], timeframe)
    before = manager._projection_shared_content_signature(SYMBOL, 0, 3 * ONE_MIN_MS, timeframe=timeframe)
    other = CandlestickManager(exchange=None, exchange_name=manager.exchange_name, cache_dir=manager.cache_dir)
    idx = other._ensure_symbol_index(SYMBOL, timeframe=timeframe)
    idx["meta"]["known_gaps"] = [{"start_ts": 2 * ONE_MIN_MS, "end_ts": 3 * ONE_MIN_MS, "reason": "manual"}]
    other._save_index(SYMBOL, timeframe=timeframe)
    after = manager._projection_shared_content_signature(SYMBOL, 0, 3 * ONE_MIN_MS, timeframe=timeframe)
    assert after != before
    assert after == reference(manager)._projection_shared_content_signature(SYMBOL, 0, 3 * ONE_MIN_MS, timeframe=timeframe)
    assert after[1] == ((2 * ONE_MIN_MS, 3 * ONE_MIN_MS, "manual"),)
    other._cleanup_on_exit()


def test_warm_recent_window_does_not_scan_historical_gaps_or_copy_candles(manager, monkeypatch):
    """16k old gaps cause zero normalization and zero history copies on a warm hit."""
    count = 16_884
    install(manager, [dict(start_ts=i * ONE_MIN_MS, end_ts=i * ONE_MIN_MS,
                           added_at=1, last_retry_at=1, retry_count=3, reason="auto_detected")
                      for i in range(count)])
    start, end = (count + 5) * ONE_MIN_MS, (count + 10) * ONE_MIN_MS
    compare(manager, start, end)
    def unexpected(*args, **kwargs):
        raise AssertionError("warm empty window rescanned or copied historical data")
    monkeypatch.setattr(module._GapMetadataList, "_build_range_index", unexpected)
    monkeypatch.setattr(module, "_sorted_candle_copy", unexpected)
    original = manager._normalize_gap_entries
    def empty_only(values):
        assert len(values) == 0
        return original(values)
    monkeypatch.setattr(manager, "_normalize_gap_entries", empty_only)
    for _ in range(10):
        assert manager._unverified_gap_ranges(SYMBOL, start, end) == []
        assert manager._unverified_uncovered_gap_ranges(SYMBOL, start, end) == []
        assert manager._due_unverified_gap_ranges(SYMBOL, start, end) == []
        assert manager._projection_shared_content_signature(SYMBOL, start, end, timeframe="1m")[1] == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("coverage", ["none", "covered", "missing", "synthetic"])
async def test_full_candle_and_warm_ema_results_match_full_scan(manager, tmp_path, coverage):
    """Nonconstant prices retain exact candle/EMA outputs through cold and warm reads."""
    before = FullScanReference(exchange=None, exchange_name="hyperliquid",
                               cache_dir=str(tmp_path / "reference"))
    before._now_ms_callback = lambda: NOW
    end = NOW // ONE_MIN_MS * ONE_MIN_MS - ONE_MIN_MS
    rows = np.zeros(800, dtype=CANDLE_DTYPE)
    rows["ts"] = np.arange(end - 799 * ONE_MIN_MS, end + 1, ONE_MIN_MS)
    rows["c"] = 100.0 + np.sin(np.arange(800) / 13.0) * 7.0
    rows["o"] = rows["c"] + 0.1
    rows["h"] = rows["c"] + 0.7
    rows["l"] = rows["c"] - 0.5
    rows["bv"] = 2.0 + np.arange(800) % 11
    gaps = [dict(start_ts=i * ONE_MIN_MS, end_ts=i * ONE_MIN_MS,
                 reason="auto_detected", added_at=1, last_retry_at=1) for i in range(1500)]
    if coverage != "none":
        gaps.append(dict(start_ts=int(rows["ts"][600]), end_ts=int(rows["ts"][601]),
                         reason="auto_detected", added_at=NOW, last_retry_at=NOW))
    if coverage == "missing":
        rows = np.delete(rows, [600, 601])
    for cm in (manager, before):
        install(cm, copy.deepcopy(gaps))
        cm._cache[SYMBOL] = rows.copy()
        if coverage == "synthetic":
            cm._synthetic_timestamps[SYMBOL] = set(int(ts) for ts in rows["ts"][600:602])
    try:
        for _ in range(3):
            kwargs = dict(start_ts=end - 499 * ONE_MIN_MS, end_ts=end,
                          allow_remote_fetch=False, allow_provisional_internal_gaps=True)
            actual = await manager.get_candles(SYMBOL, **kwargs)
            expected = await before.get_candles(SYMBOL, **kwargs)
            np.testing.assert_array_equal(actual, expected)
        # Remove the finalized endpoint to exercise a real projected open tail.
        for cm in (manager, before):
            cm._cache[SYMBOL] = rows[:-1].copy()
            cm._projected_open_tail_ema_cache.clear()
        kwargs = dict(latest_expected_ts=end, last_cached_ts=end - ONE_MIN_MS,
                      max_tail_gap_ms=3 * ONE_MIN_MS)
        for _ in range(3):
            actual = await manager.get_projected_open_tail_ema_metrics(
                SYMBOL, {"close": [13.0, 77.0], "qv": [40.0], "log_range": [63.0]}, **kwargs)
            expected = await before.get_projected_open_tail_ema_metrics(
                SYMBOL, {"close": [13.0, 77.0], "qv": [40.0], "log_range": [63.0]}, **kwargs)
            assert actual == expected
            if coverage == "none": assert actual["close"]
    finally:
        before._cleanup_on_exit()


@pytest.mark.parametrize("insertion", ["append", "replace_record", "nested_value", "owned_alias"])
def test_inserted_container_aliases_cannot_freeze_the_range_view(manager, insertion):
    """External aliases use conservative scans; existing owned aliases stay shared."""
    gaps = install(manager, [dict(start_ts=0, end_ts=ONE_MIN_MS, reason="fetch_failed")])
    compare(manager, 0, 10 * ONE_MIN_MS)
    alias = dict(start_ts=4 * ONE_MIN_MS, end_ts=5 * ONE_MIN_MS)
    if insertion == "append": gaps.append(alias)
    elif insertion == "replace_record": gaps[0] = alias
    elif insertion == "nested_value":
        alias = [4 * ONE_MIN_MS, 5 * ONE_MIN_MS]
        gaps.append(alias)
    else:
        alias = gaps[0]
        gaps.append(alias)
        assert gaps[0] is gaps[1]
    compare(manager, 0, 10 * ONE_MIN_MS)
    if isinstance(alias, dict): alias.update(start_ts=8 * ONE_MIN_MS, end_ts=9 * ONE_MIN_MS)
    else: alias[:] = [8 * ONE_MIN_MS, 9 * ONE_MIN_MS]
    compare(manager, 0, 10 * ONE_MIN_MS)
    compare(manager, 0, ONE_MIN_MS)
    compare(manager, 8 * ONE_MIN_MS, 9 * ONE_MIN_MS)
