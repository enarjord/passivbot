"""Suite acquisition needs facts for its effective scenarios, not just base ages."""

from copy import deepcopy
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from suite_runner import (
    apply_scenario_overrides,
    build_scenarios,
    extract_suite_config,
    load_suite_override_config,
    prepare_master_datasets,
)
from test_age_fact_recovery import BASE, DAY, _cache_config, _save_cache


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["inline", "external"])
async def test_suite_cache_recovers_scenario_inception_and_native_admission(
    tmp_path, monkeypatch, source,
):
    import backtest as bt
    import passivbot_rust as pbr
    from test_side_minimum_age import _native_payload

    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_a, **_kw: coin)
    monkeypatch.setattr(bt, "compute_per_coin_warmup_minutes", lambda _: {"__default__": 0})
    monkeypatch.setattr(bt, "compute_backtest_warmup_minutes", lambda _: 0)
    config = _cache_config(tmp_path, (0.0, 0.0))
    scenario = {"label": "positive", "overrides": {
        "bot.long.universe.minimum_coin_age_days": 5.5,
    }}
    config["backtest"]["scenarios"] = [scenario] if source == "inline" else [{"label": "zero"}]
    override = None
    if source == "external":
        file = tmp_path / "suite.json"
        file.write_text(json.dumps({"scenarios": [scenario]}))
        override = load_suite_override_config(file, source_config=config)
    scenarios, _ = build_scenarios(extract_suite_config(config, override), ["binance"])
    original = deepcopy(config)
    directory, arrays, old_mss = _save_cache(bt, config, None)
    recovered_mss = deepcopy(old_mss)
    recovered_mss["AAA"]["inception_timestamp_ms"] = BASE - 6 * DAY
    materialize = AsyncMock(return_value=(recovered_mss, arrays[2], arrays[0], arrays[1]))
    monkeypatch.setattr(bt, "try_prepare_hlcvs_v2_local", materialize)

    dataset = (await prepare_master_datasets(config, ["binance"], scenarios=scenarios))["binance"]
    assert materialize.await_count == 1
    assert dataset.mss["AAA"]["inception_timestamp_ms"] == BASE - 6 * DAY
    # Save must replace the same existing cache, despite its sufficient warmup.
    assert dataset.cache_dir == str(directory)
    disk = json.loads((directory / "market_specific_settings.json").read_text())
    assert disk["AAA"]["inception_timestamp_ms"] == BASE - 6 * DAY
    again = (await prepare_master_datasets(config, ["binance"], scenarios=scenarios))["binance"]
    assert materialize.await_count == 1
    assert again.mss["AAA"]["inception_timestamp_ms"] == BASE - 6 * DAY
    assert config == original
    assert old_mss["AAA"]["inception_timestamp_ms"] is None

    effective = deepcopy(config)
    apply_scenario_overrides(effective, scenarios[0].overrides)
    days = effective["bot"]["long"]["universe"]["minimum_coin_age_days"]
    for fact, expected in [(None, False), (dataset.mss["AAA"]["inception_timestamp_ms"], True)]:
        _, _, args = _native_payload(days, 0.0, inception=fact)
        fills = pbr.run_backtest(*args)[0]
        assert any("entry_initial" in str(fill[13]) and str(fill[13]).endswith("long")
                   for fill in fills) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("base_ages,fixed_age,scenario_age,expected", [
    ((0.0, 0.0), None, None, False),
    ((0.0, 0.0), 60.0, 0.0, False),
    ((0.0, 0.0), 0.0, 5.5, True),
    ((0.0, 0.0), None, 5.5, True),
    ((0.0, 60.0), None, 0.0, False),
])
@pytest.mark.parametrize("selector", ["leaf", "universe", "side", "bot"])
async def test_assigned_scenario_fact_demand_follows_fixed_then_scenario_owner(
    tmp_path, monkeypatch, base_ages, fixed_age, scenario_age, expected, selector,
):
    import backtest as bt
    from optimization.warmup import build_optimizer_data_config

    config = _cache_config(tmp_path, base_ages)
    config["optimize"]["bounds"] = {"long": {"risk": {
        "total_wallet_exposure_limit": [1.0, 1.0],
    }}}
    if fixed_age is not None:
        config["optimize"]["fixed_runtime_overrides"] = {
            "bot.short.universe.minimum_coin_age_days": fixed_age,
        }
    config = build_optimizer_data_config(config)
    patch = {}
    if scenario_age is not None:
        path = {"leaf": "bot.short.universe.minimum_coin_age_days",
                "universe": "bot.short.universe", "side": "bot.short", "bot": "bot"}[selector]
        value = scenario_age
        if selector != "leaf":
            value = deepcopy(config["bot"] if selector == "bot" else
                             config["bot"]["short"] if selector == "side" else
                             config["bot"]["short"]["universe"])
            target = value["short"]["universe"] if selector == "bot" else value["universe"] if selector == "side" else value
            target["minimum_coin_age_days"] = scenario_age
        patch[path] = value
    scenarios, _ = build_scenarios({"scenarios": [{"label": "age", "overrides": patch}]}, ["binance"])
    original, originals = deepcopy(config), deepcopy(scenarios)
    calls = []

    async def prepare(effective, exchange, **kwargs):
        calls.append((deepcopy(effective), kwargs))
        return (["AAA"], np.ones((2, 1, 4)), {"AAA": {"exchange": exchange}}, "", "cache",
                np.ones(2), np.array([BASE, BASE + 60_000], dtype=np.int64))

    monkeypatch.setattr(bt, "prepare_hlcvs_mss", prepare)
    await prepare_master_datasets(config, ["binance"], scenarios=scenarios)
    assert calls[0][1] == {"require_inception_facts": expected}
    assert calls[0][0] == original
    assert config == original and scenarios == originals


@pytest.mark.asyncio
async def test_combined_and_individual_fact_demand_uses_only_assigned_scenarios(tmp_path, monkeypatch):
    import backtest as bt
    config = _cache_config(tmp_path, (0.0, 0.0))
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    scenarios, _ = build_scenarios({"scenarios": [
        {"label": "combined", "exchanges": ["binance", "bybit"], "overrides": {
            "bot.short.universe.minimum_coin_age_days": 7.25}},
        {"label": "single", "exchanges": ["binance"]},
    ]}, config["backtest"]["exchanges"])
    calls = {}

    async def prepare(effective, exchange, **kwargs):
        calls[exchange] = kwargs.get("require_inception_facts")
        return (["AAA"], np.ones((2, 1, 4)), {"AAA": {"exchange": "binance"}}, "", "cache",
                np.ones(2), np.array([BASE, BASE + 60_000], dtype=np.int64))

    monkeypatch.setattr(bt, "prepare_hlcvs_mss", prepare)
    await prepare_master_datasets(config, config["backtest"]["exchanges"],
                                  scenarios=scenarios, needed_individual_exchanges={"binance"})
    assert calls == {"combined": True, "binance": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["zero_only", "scenario_zero", "immutable"])
async def test_suite_unknown_cache_reuse_and_explicit_dataset_boundaries(tmp_path, monkeypatch, policy):
    import backtest as bt
    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_a, **_kw: coin)
    monkeypatch.setattr(bt, "compute_per_coin_warmup_minutes", lambda _: {"__default__": 0})
    monkeypatch.setattr(bt, "compute_backtest_warmup_minutes", lambda _: 0)
    config = _cache_config(tmp_path, (60.0, 60.0) if policy == "scenario_zero" else (0.0, 0.0))
    patch = {}
    if policy == "scenario_zero":
        patch = {f"bot.{side}.universe.minimum_coin_age_days": 0.0 for side in ("long", "short")}
    elif policy == "immutable":
        patch = {"bot.long.universe.minimum_coin_age_days": 60.0}
    scenarios, _ = build_scenarios({"scenarios": [{"label": "age", "overrides": patch}]}, ["binance"])
    directory, arrays, mss = _save_cache(bt, config, None)
    if policy == "immutable":
        config["backtest"].update(hlcvs_data_dir=str(directory), hlcvs_data_override_mode="dataset")
    original = deepcopy(config)
    original_files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    materialize = AsyncMock(side_effect=AssertionError("this context must reuse its dataset"))
    monkeypatch.setattr(bt, "try_prepare_hlcvs_v2_local", materialize)
    dataset = (await prepare_master_datasets(config, ["binance"], scenarios=scenarios))["binance"]
    assert dataset.mss["AAA"]["inception_timestamp_ms"] is None
    materialize.assert_not_awaited()
    if policy != "immutable":
        recovered = deepcopy(mss)
        recovered["AAA"]["inception_timestamp_ms"] = BASE
        bt.save_coins_hlcvs_to_cache(
            config, ["AAA"], arrays[0], "binance", recovered, arrays[1], arrays[2],
            warmup_minutes=7, require_inception_facts=False,
        )
    assert config == original
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == original_files


@pytest.mark.asyncio
@pytest.mark.parametrize("base_ages", [(0.0, 0.0), (60.0, 0.0)])
async def test_individual_only_scenarios_do_not_require_facts_for_unused_combined_dataset(tmp_path, monkeypatch, base_ages):
    import backtest as bt
    config = _cache_config(tmp_path, base_ages)
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    scenarios, _ = build_scenarios({"scenarios": [
        {"label": "single", "exchanges": ["binance"], "coins": ["AAA"], "overrides": {
            "bot.short.universe.minimum_coin_age_days": 7.25}},
    ]}, config["backtest"]["exchanges"])
    calls = {}
    async def prepare(effective, exchange, **kwargs):
        calls[exchange] = kwargs.get("require_inception_facts")
        return (["AAA"], np.ones((2, 1, 4)), {"AAA": {"exchange": "binance"}}, "", "cache",
                np.ones(2), np.array([BASE, BASE + 60_000], dtype=np.int64))
    monkeypatch.setattr(bt, "prepare_hlcvs_mss", prepare)
    await prepare_master_datasets(config, config["backtest"]["exchanges"],
                                  scenarios=scenarios, needed_individual_exchanges={"binance"})
    assert calls == {"combined": False, "binance": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("scenarios,expected", [(None, {}), ([], {"require_inception_facts": False})])
async def test_unassigned_base_preload_retains_default_age_inference(tmp_path, monkeypatch, scenarios, expected):
    import backtest as bt
    config = _cache_config(tmp_path, (60.0, 0.0))
    calls = []
    async def prepare(effective, exchange, **kwargs):
        calls.append(kwargs)
        return (["AAA"], np.ones((2, 1, 4)), {"AAA": {"exchange": exchange}}, "", "cache",
                np.ones(2), np.array([BASE, BASE + 60_000], dtype=np.int64))
    monkeypatch.setattr(bt, "prepare_hlcvs_mss", prepare)
    await prepare_master_datasets(config, ["binance"], scenarios=scenarios)
    assert calls == [expected]
