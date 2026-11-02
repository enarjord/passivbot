from copy import deepcopy

import pytest

import backtest
from preparation_identity import CACHE_REQUEST_SCHEMA_VERSION, gap_tolerance_minutes
from config import prepare_config
from config.schema import get_template_config


def config(tolerance):
    raw = get_template_config()
    raw["bot"]["long"]["enabled"] = True
    raw["bot"]["global"]["universe"]["approved_coins"] = ["BTC", "ETH"]
    raw["backtest"].update(start_date="2025-01", end_date="2025-02-01",
                          gap_tolerance_ohlcvs_minutes=tolerance)
    return prepare_config(raw, verbose=False)


def test_gap_tolerance_numeric_representation_has_one_versioned_cache_key(monkeypatch):
    monkeypatch.setattr(backtest, "coin_to_symbol", lambda coin, *a, **kw: coin + '/USDT:USDT')
    integer = config(120)
    floating = config(120.0)
    assert backtest.get_cache_hash(integer, "combined") == backtest.get_cache_hash(floating, "combined")
    assert type(integer["backtest"]["gap_tolerance_ohlcvs_minutes"]) is int
    assert type(floating["backtest"]["gap_tolerance_ohlcvs_minutes"]) is float
    captured = []
    monkeypatch.setattr(backtest, "calc_hash", lambda request: captured.append(request) or 'captured')
    assert backtest.get_cache_hash(integer, "combined") == 'captured'
    assert captured[0]["cache_request_schema_version"] == CACHE_REQUEST_SCHEMA_VERSION == 2
    assert type(captured[0]["gap_tolerance_ohlcvs_minutes"]) is float


def test_normalization_does_not_alias_dates_order_or_fractional_policy(monkeypatch):
    monkeypatch.setattr(backtest, "coin_to_symbol", lambda coin, *a, **kw: coin + '/USDT:USDT')
    original = config(120)
    base = backtest.get_cache_hash(original, "combined")
    fraction = deepcopy(original)
    fraction["backtest"]["gap_tolerance_ohlcvs_minutes"] = 120.25
    assert backtest.get_cache_hash(fraction, "combined") != base
    full_date = deepcopy(original)
    full_date["backtest"]["start_date"] = "2025-01-01"
    assert backtest.get_cache_hash(full_date, "combined") != base
    order = deepcopy(original)
    order["backtest"]["exchanges"] = list(reversed(order["backtest"]["exchanges"]))
    assert backtest.get_cache_hash(order, "combined") != base


def test_offline_preparation_and_online_reuse_have_the_same_candle_request_key(monkeypatch):
    monkeypatch.setattr(backtest, "coin_to_symbol", lambda coin, *a, **kw: coin + '/USDT:USDT')
    online = config(120)
    offline = deepcopy(online)
    online["backtest"]["offline"] = False
    offline["backtest"]["offline"] = True
    assert backtest.get_cache_hash(offline, "combined") == backtest.get_cache_hash(online, "combined")
    assert online["backtest"]["offline"] is False


@pytest.mark.parametrize("value", [True, '120', -1, float('nan'), float('inf')])
def test_gap_tolerance_rejects_invalid_required_input(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        gap_tolerance_minutes({"backtest": {"gap_tolerance_ohlcvs_minutes": value}})
