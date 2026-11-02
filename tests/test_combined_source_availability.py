"""Unverified source coverage must not remove a valid approved alternative."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import hlcv_preparation as hp
from simulation_data import OfflineDataError, simulation_data_policy


START = 1_774_915_200_000
END = START + 60_000
VENUES = ("binanceusdm", "bybit")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def managers():
    return {
        exchange: SimpleNamespace(
            has_coin=lambda coin: True,
            get_symbol=lambda coin: f"{coin}/USDT:USDT",
            get_market_specific_settings=lambda coin: {"symbol": f"{coin}/USDT:USDT"},
        )
        for exchange in VENUES
    }


def valid_result(exchange):
    frame = pd.DataFrame({
        "timestamp": np.asarray([START, END], dtype=np.int64),
        "high": [101., 102.], "low": [99., 100.],
        "close": [100., 101.], "volume": [10., 11.],
    })
    return exchange, frame, 2, 0, 21.


async def resolve(*, report, forced=None):
    return await hp._resolve_combined_coin(
        coin="NEW", sem=asyncio.Semaphore(1), base_start_ts=START, end_ts=END,
        first_timestamps_unified={"NEW": START}, tradfi_for_stock_perps=False,
        forced_sources={} if forced is None else {"NEW": forced},
        market_settings_sources={}, exchanges_to_consider=VENUES,
        normalization_candidate_exchanges=VENUES, om_dict=managers(),
        per_coin_warmups={}, default_warm=0, force_refetch_gaps=False,
        catalog=None, store=None, candidate_report=report,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [None, *VENUES])
async def test_valid_source_survives_unverified_alternate_history(monkeypatch, unavailable):
    async def fetch(coin, exchange, *args, **kwargs):
        if exchange == unavailable:
            raise hp.SourceHistoryUnavailable("Unverified stale leading coverage")
        return valid_result(exchange)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fetch)
    report = []
    result = await resolve(report=report)
    expected = "bybit" if unavailable == "binanceusdm" else "binanceusdm"
    assert result.coin == "NEW" and result.best_exchange == expected
    pd.testing.assert_frame_equal(result.best_df, valid_result(expected)[1])
    assert {candidate.exchange for candidate in result.candidates} == set(VENUES) - {unavailable}
    if unavailable is not None:
        failure = next(item for item in report if item["exchange"] == unavailable)
        assert failure["status"] == "ineligible"
        assert failure["reason"] == "unverified_source_history"
        assert failure["gap_class"] == "unknown"


@pytest.mark.asyncio
async def test_unverified_forced_source_is_not_replaced(monkeypatch):
    async def fetch(coin, exchange, *args, **kwargs):
        if exchange == "bybit":
            raise hp.SourceHistoryUnavailable("Unverified stale leading coverage")
        return valid_result(exchange)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fetch)
    with pytest.raises(RuntimeError, match="Forced exchange bybit failed"):
        await resolve(report=[], forced="bybit")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [hp.SourceHistoryUnavailable, OfflineDataError])
async def test_offline_source_unavailability_still_fails_closed(monkeypatch, error):
    async def fetch(coin, exchange, *args, **kwargs):
        if exchange == "bybit":
            raise error("Unverified history")
        return valid_result(exchange)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fetch)
    with simulation_data_policy({"backtest": {"offline": True}}), pytest.raises(error):
        await resolve(report=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError, RuntimeError, hp.HlcvsDataIntegrityError])
async def test_unexpected_source_error_propagates_instead_of_dropping_coin(monkeypatch, error):
    async def fetch(coin, exchange, *args, **kwargs):
        if exchange == "bybit":
            raise error("Malformed source input")
        return valid_result(exchange)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fetch)
    with pytest.raises(RuntimeError, match="Exchange bybit failed") as failure:
        await resolve(report=[])
    assert type(failure.value.__cause__) is error


@pytest.mark.asyncio
async def test_no_usable_source_retains_existing_online_no_history_policy(monkeypatch):
    async def fetch(*args, **kwargs):
        raise hp.SourceHistoryUnavailable("Unverified stale leading coverage")
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fetch)
    report = []
    assert await resolve(report=report) is None
    assert len(report) == 2 and all(item["status"] == "ineligible" for item in report)
