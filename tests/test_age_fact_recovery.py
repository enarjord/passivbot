"""Unresolved inception blocks admission without making missing facts permanent."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import numpy as np
import pytest

from config.schema import get_template_config


BASE = 1_704_067_200_000
DAY = 86_400_000


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [None, 0.0])
@pytest.mark.parametrize("exchange", ["binance", "fake"])
@pytest.mark.parametrize("sibling_response", [0.0, BASE + 1])
async def test_live_unknown_inception_retries_then_admits_without_losing_known_facts(
    monkeypatch, unknown, exchange, sibling_response,
):
    import passivbot as live

    bot = object.__new__(live.Passivbot)
    coin, symbol, known = "AAA", "AAA/USDT:USDT", "BBB/USDT:USDT"
    bot.config = get_template_config()
    bot.exchange = exchange
    bot.first_timestamps = {coin: unknown, symbol: unknown, known: BASE}
    bot.approved_coins_minus_ignored_coins = {"long": {coin, known}, "short": set()}
    bot.markets_dict = {symbol: {}, known: {}}
    bot.coin_to_symbol = lambda value, **_kwargs: symbol if value == coin else value
    bot.get_exchange_time = lambda: BASE + 61 * DAY
    lookup = AsyncMock(side_effect=[{coin: 0.0}, {coin: BASE, known: sibling_response}])
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    bot.cca = SimpleNamespace(fetch_ohlcv=AsyncMock(side_effect=[[], [[BASE, 1, 1, 1, 1, 1]]]))

    assert not bot.is_old_enough("long", symbol)
    await bot.update_first_timestamps()
    assert not bot.is_old_enough("long", symbol)
    assert bot.first_timestamps[known] == BASE
    await bot.update_first_timestamps()
    assert bot.is_old_enough("long", symbol)
    assert bot.first_timestamps[known] == BASE
    await bot.update_first_timestamps()
    if exchange == "fake":
        assert bot.cca.fetch_ohlcv.await_count == 2
    else:
        assert lookup.await_count == 2
        assert all(call.args[0] == [coin] for call in lookup.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [None, 0.0])
async def test_live_resolved_alias_repairs_unknown_symbol_without_refetch(monkeypatch, unknown):
    import passivbot as live
    coin, symbol = "AAA", "AAA/USDT:USDT"
    bot = object.__new__(live.Passivbot)
    bot.config = get_template_config()
    bot.exchange = "binance"
    bot.first_timestamps = {coin: BASE, symbol: unknown}
    bot.approved_coins_minus_ignored_coins = {"long": {coin}, "short": set()}
    bot.markets_dict = {symbol: {}}
    bot.coin_to_symbol = lambda value, **_kwargs: symbol if value == coin else value
    bot.get_exchange_time = lambda: BASE + 61 * DAY
    lookup = AsyncMock(side_effect=AssertionError("resolved aliases do not need a fetch"))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    assert not bot.is_old_enough("long", symbol)
    await bot.update_first_timestamps()
    assert bot.is_old_enough("long", symbol)
    lookup.assert_not_awaited()


def _cache_config(tmp_path, ages):
    config = get_template_config()
    config["backtest"].update(
        base_dir=str(tmp_path / "results"), compress_cache=False,
        start_date="2024-01-01", end_date="2024-01-02", exchanges=["binance"],
    )
    for side, age in zip(("long", "short"), ages):
        config["bot"][side]["universe"].update(
            minimum_coin_age_days=age, approved_coins=["AAA"],
        )
    return config


def _save_cache(bt, config, inception):
    mss = {"AAA": {"exchange": "binance", "first_valid_index": 0,
                   "last_valid_index": 2, "inception_timestamp_ms": inception}}
    arrays = (np.ones((3, 1, 4), dtype=np.float64), np.ones(3),
              np.array([BASE, BASE + 60_000, BASE + 120_000], dtype=np.int64))
    directory = bt.save_coins_hlcvs_to_cache(
        config, ["AAA"], arrays[0], "binance", mss, arrays[1], arrays[2],
        warmup_minutes=7,
    )
    return directory, arrays, mss


@pytest.mark.parametrize("unknown", [None, 0.0])
def test_automatic_cache_rebuild_publishes_recovered_inception(tmp_path, monkeypatch, unknown):
    import backtest as bt
    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_a, **_kw: coin)
    # The actual optimizer sizing owner activates the reachable short side.
    from optimization.warmup import build_optimizer_data_config
    config = _cache_config(tmp_path, (0.0, 60.0))
    config["optimize"]["bounds"] = {"short": {"risk": {
        "total_wallet_exposure_limit": [0.0, 1.0], "max_positions": [1, 1],
    }}}
    config = build_optimizer_data_config(config)
    directory, arrays, original_mss = _save_cache(bt, config, unknown)
    assert bt.load_coins_hlcvs_from_cache(config, "binance", 7) is None
    recovered_dir, _, _ = _save_cache(bt, config, BASE - 61 * DAY)
    assert recovered_dir == directory
    result = bt.load_coins_hlcvs_from_cache(config, "binance", 7)
    assert result is not None
    assert result[3]["AAA"]["inception_timestamp_ms"] == BASE - 61 * DAY
    for actual, expected in zip((result[2], result[5], result[6]), arrays):
        np.testing.assert_array_equal(actual, expected)
    assert original_mss["AAA"]["inception_timestamp_ms"] == unknown
    disk = json.loads((directory / "market_specific_settings.json").read_text())
    assert disk["AAA"]["inception_timestamp_ms"] == BASE - 61 * DAY


def test_zero_age_reuses_unknown_cache_and_explicit_override_remains_immutable(tmp_path, monkeypatch):
    import backtest as bt
    from hlcvs_override import load_hlcvs_data_override
    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_a, **_kw: coin)
    config = _cache_config(tmp_path, (0.0, 0.0))
    directory, _, _ = _save_cache(bt, config, None)
    assert bt.load_coins_hlcvs_from_cache(config, "binance", 7)[3]["AAA"]["inception_timestamp_ms"] is None
    _save_cache(bt, config, BASE)  # Sufficient zero-age caches retain their ordinary fast path.
    assert bt.load_coins_hlcvs_from_cache(config, "binance", 7)[3]["AAA"]["inception_timestamp_ms"] is None
    original_files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    explicit = deepcopy(config)
    explicit["bot"]["long"]["universe"]["minimum_coin_age_days"] = 60.0
    explicit["backtest"].update(hlcvs_data_dir=str(directory), hlcvs_data_override_mode="dataset")
    result = load_hlcvs_data_override(explicit, "binance")
    assert result[3]["AAA"]["inception_timestamp_ms"] is None
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == original_files


@pytest.mark.parametrize("context", ["root", "scenario", "fixed_root_age", "inline_only",
                                     "fixed_coin_leaf", "fixed_coin_object"])
def test_released_coin_file_and_inline_age_are_ignored_and_reported(tmp_path, context):
    from released_v8_fixtures import released_v8_policy
    from tools.migrate_config_v9 import migrate
    source = released_v8_policy()
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"live": {"minimum_coin_age_days": 7.25},
                                "bot": {"long": {"unstuck_close_pct": 0.15}}}))
    inline = {"override_config_path": file.name, "live": {"minimum_coin_age_days": 12.5},
              "bot": {"long": {"unstuck_close_pct": 0.25}}}
    if context == "inline_only":
        inline.pop("override_config_path")
    source["coin_overrides"] = {"AAA": inline}
    if context == "scenario":
        source["backtest"]["scenarios"] = [{"label": "coin", "overrides": {"coin_overrides.AAA": inline}}]
    elif context == "fixed_root_age":
        source["optimize"]["fixed_runtime_overrides"] = {"live.minimum_coin_age_days": 5.5}
    elif context == "fixed_coin_leaf":
        source["optimize"]["fixed_runtime_overrides"] = {"coin_overrides.AAA.bot.long.unstuck_close_pct": 0.25}
    elif context == "fixed_coin_object":
        source["optimize"]["fixed_runtime_overrides"] = {"coin_overrides.AAA": inline}
    before, contents = deepcopy(source), file.read_bytes()
    if context == "fixed_coin_object":
        with pytest.raises(TypeError, match="must target leaf settings"):
            migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "input.json"))
        assert source == before and file.read_bytes() == contents
        return
    result = migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "input.json"))
    assert result.config["coin_overrides"]["AAA"]["bot"]["long"]["unstuck"]["close_fraction"] == 0.25
    assert all(result.config["bot"][side]["universe"]["minimum_coin_age_days"] == 60.0 for side in ("long", "short"))
    inline_changes = [item for item in result.changes if item["path"] == "coin_overrides.AAA.live.minimum_coin_age_days"]
    assert inline_changes and all(item["old"] == 12.5 and item["new"] is None and "ignored" in item["reason"] for item in inline_changes)
    if context != "inline_only":
        assert any("override_config_path::" in item["path"] and "minimum_coin_age_days" in item["path"] and "ignored" in item["reason"] for item in result.changes)
    if context == "fixed_root_age":
        assert result.config["optimize"]["fixed_runtime_overrides"] == {f"bot.{side}.universe.minimum_coin_age_days": 5.5 for side in ("long", "short")}
    elif context == "fixed_coin_leaf":
        fixed = result.config["optimize"]["fixed_runtime_overrides"]
        assert fixed == {"coin_overrides.AAA.bot.long.unstuck.close_fraction": 0.25}
    elif context == "scenario":
        patch = result.config["backtest"]["scenarios"][0]["overrides"]["coin_overrides.AAA"]
        assert patch["bot"]["long"]["unstuck"]["close_fraction"] == 0.25
        assert "minimum_coin_age_days" not in json.dumps(patch)
    assert source == before and file.read_bytes() == contents
