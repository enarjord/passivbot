"""Only coins reachable through a positive side gate need inception recovery."""

from copy import deepcopy
from unittest.mock import AsyncMock

import numpy as np
import pytest

from test_age_fact_recovery import BASE, DAY, _cache_config


def _contexts(config, scenarios, columns=("AAA", "BBB")):
    from suite_runner import _prepare_scenario_universes
    return _prepare_scenario_universes(
        config, scenarios, master_coins=list(columns), master_ignored=[],
        available_exchanges=["binance"], base_coins=list(columns), base_ignored=[],
    )


def _scenario(config, scenario, context, columns=("AAA", "BBB")):
    from suite_runner import apply_scenario
    return apply_scenario(
        config, scenario, list(columns), [], ["binance"], set(columns),
        base_coins=list(columns), base_ignored=[], prepared_universe=context,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("source", ["inline", "external"])
async def test_disjoint_scenario_cache_ignores_unneeded_coin_fact(tmp_path, monkeypatch, recover, source):
    import backtest as bt
    from suite_runner import build_scenarios, prepare_master_datasets

    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_a, **_kw: coin)
    monkeypatch.setattr(bt, "compute_per_coin_warmup_minutes", lambda _: {"__default__": 0})
    monkeypatch.setattr(bt, "compute_backtest_warmup_minutes", lambda _: 0)
    config = _cache_config(tmp_path, (0.0, 0.0))
    for side in ("long", "short"):
        config["bot"][side]["universe"]["approved_coins"] = ["AAA", "BBB"]
    suite = {"scenarios": [
        {"label": "aged", "coins": ["AAA"], "overrides": {
            "bot.long.universe.minimum_coin_age_days": 7.0}},
        {"label": "zero", "coins": ["BBB"]},
    ]}
    if source == "external":
        import json
        from suite_runner import extract_suite_config, load_suite_override_config
        file = tmp_path / "suite.json"
        file.write_text(json.dumps(suite))
        config["backtest"]["scenarios"] = [{"label": "unused"}]
        suite = extract_suite_config(config, load_suite_override_config(file, source_config=config))
    scenarios, _ = build_scenarios(suite, ["binance"])
    known = BASE - 8 * DAY
    mss = {coin: {"exchange": "binance", "first_valid_index": 0, "last_valid_index": 2,
                  "inception_timestamp_ms": None if coin == "BBB" or recover else known}
           for coin in ("AAA", "BBB")}
    arrays = (np.ones((3, 2, 4)), np.ones(3), np.array([BASE, BASE + 60_000, BASE + 120_000]))
    directory = bt.save_coins_hlcvs_to_cache(
        config, ["AAA", "BBB"], arrays[0], "binance", mss, arrays[1], arrays[2], warmup_minutes=7,
    )
    recovered = deepcopy(mss)
    recovered["AAA"]["inception_timestamp_ms"] = known
    materialize = AsyncMock(
        return_value=(recovered, arrays[2], arrays[0], arrays[1]) if recover else None,
        side_effect=None if recover else AssertionError("BBB has no positive-age execution context"),
    )
    monkeypatch.setattr(bt, "try_prepare_hlcvs_v2_local", materialize)
    original = deepcopy(config)
    first = (await prepare_master_datasets(config, ["binance"], scenarios=scenarios))["binance"]
    second = (await prepare_master_datasets(config, ["binance"], scenarios=scenarios))["binance"]
    assert materialize.await_count == int(recover)
    assert first.cache_dir == second.cache_dir == str(directory)
    assert second.mss["AAA"]["inception_timestamp_ms"] == known
    assert second.mss["BBB"]["inception_timestamp_ms"] is None
    assert config == original


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("policy,expected", [
    ("allowed", {"AAA"}), ("empty", set()), ("ignored", set()), ("disabled", set()),
])
def test_context_demand_is_side_enabled_approved_and_not_ignored(tmp_path, side, policy, expected):
    from suite_runner import build_scenarios
    config = _cache_config(tmp_path, (0.0, 0.0))
    for position in ("long", "short"):
        config["bot"][position]["universe"]["approved_coins"] = ["AAA", "BBB"]
        config["bot"][position]["risk"].update(max_positions=1, total_wallet_exposure_limit=1.0)
    universe = deepcopy(config["bot"][side]["universe"])
    universe.update(minimum_coin_age_days=7.0, approved_coins=[] if policy == "empty" else ["AAA"],
                    ignored_coins=["AAA"] if policy == "ignored" else [])
    patch = {f"bot.{position}.universe": {**deepcopy(universe), "minimum_coin_age_days": 7.0 if position == side else 0.0}
             for position in ("long", "short")}
    if policy == "disabled":
        patch[f"bot.{side}.risk.total_wallet_exposure_limit"] = 0.0
    scenarios, _ = build_scenarios({"scenarios": [{"label": "side", "overrides": patch}]}, ["binance"])
    context = _contexts(config, scenarios)["side"]
    assert context.inception_coins == frozenset(expected)


@pytest.mark.parametrize("mutation", ["changed", "deleted"])
def test_scenario_file_is_resolved_once_and_reused_for_execution(tmp_path, monkeypatch, mutation):
    from pathlib import Path
    from config.universe import materialize_side_universe_sources, resolve_universe_sources, universe_source_view
    from suite_runner import build_scenarios
    file = tmp_path / "approved.json"
    file.write_text('["AAA"]')
    config = _cache_config(tmp_path, (7.0, 0.0))
    config["bot"]["global"]["universe"]["approved_coins"] = str(file)
    config["bot"]["long"]["universe"]["approved_coins"] = None
    config["bot"]["short"]["universe"]["approved_coins"] = None
    for field in ("approved_coins", "ignored_coins"):
        values, _ = resolve_universe_sources(config, field)
        materialize_side_universe_sources(config, field, values)
    file.write_text('["BBB"]')
    scenarios, _ = build_scenarios({"scenarios": [
        {"label": "unchanged"},
        {"label": "rebound", "overrides": {"bot.global.universe.approved_coins": str(file)}},
    ]}, ["binance"])
    reads = []
    read = Path.open
    def record(path, *args, **kwargs):
        if path == file and (args[0] if args else kwargs.get("mode", "r")) == "r":
            reads.append(path)
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", record)
    contexts = _contexts(config, scenarios)
    assert contexts["unchanged"].inception_coins == frozenset({"AAA"})
    assert contexts["rebound"].inception_coins == frozenset({"BBB"})
    assert len(reads) == 1  # Equal explicit assignment resolves once; omission never rereads.
    if mutation == "changed":
        file.write_text('["CCC"]')
    else:
        file.unlink()
    frozen = deepcopy(contexts)
    for _ in range(3):  # Deduplication, session description and runtime share one context.
        for scenario in scenarios:
            output, _ = _scenario(config, scenario, contexts[scenario.label])
            expected = ["AAA"] if scenario.label == "unchanged" else ["BBB"]
            assert output["bot"]["long"]["universe"]["approved_coins"] == expected
            assert universe_source_view(output)["bot"]["global"]["universe"]["approved_coins"] == str(file)
            output["bot"]["long"]["universe"]["approved_coins"].append("CCC")
    assert contexts == frozen and len(reads) == 1
    clipped, _ = _scenario(config, scenarios[0], contexts["unchanged"], ("BBB",))
    assert clipped["bot"]["long"]["universe"]["approved_coins"] == []


@pytest.mark.parametrize("change", ["age", "enabled", "source", "prepared_membership", "base_venue", "source_assignment", "json_lookalike"])
def test_prepared_universe_cannot_be_reused_for_different_recipe(tmp_path, change):
    from config.universe import materialize_side_universe_sources
    from suite_runner import build_scenarios
    config = _cache_config(tmp_path, (0.0, 0.0))
    for side in ("long", "short"):
        config["bot"][side]["universe"]["approved_coins"] = ["AAA", "BBB"]
    if change == "enabled":
        config["bot"]["long"]["universe"]["minimum_coin_age_days"] = 7.0
        config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.0
    materialize_side_universe_sources(config, "approved_coins", {"long": ["AAA"], "short": ["AAA"]})
    scenarios, _ = build_scenarios({"scenarios": [{"label": "context"}]}, ["binance"])
    context = _contexts(config, scenarios)["context"]
    changed = deepcopy(config)
    if change == "age":
        changed["bot"]["long"]["universe"]["minimum_coin_age_days"] = 7.0
    elif change == "enabled":
        changed["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.0
    elif change == "source":
        changed["bot"]["long"]["universe"]["approved_coins"] = ["BBB"]
    elif change == "prepared_membership":
        materialize_side_universe_sources(changed, "approved_coins", {"long": ["BBB"], "short": ["BBB"]})
    elif change == "base_venue":
        changed["backtest"]["exchanges"] = ["bybit"]
    elif change == "source_assignment":
        changed["backtest"]["coin_sources"] = {"AAA": "bybit"}
    else:
        context = {"membership": context.membership, "inception_coins": context.inception_coins}
    with pytest.raises(ValueError, match="does not match its source recipe"):
        _scenario(changed, scenarios[0], context)


def test_positive_numeric_changes_keep_same_acquisition_authority(tmp_path):
    from suite_runner import build_scenarios
    config = _cache_config(tmp_path, (7.0, 0.0))
    scenarios, _ = build_scenarios({"scenarios": [{"label": "context"}]}, ["binance"])
    context = _contexts(config, scenarios)["context"]
    changed = deepcopy(config)
    changed["bot"]["long"]["universe"]["minimum_coin_age_days"] = 14.0
    changed["bot"]["long"]["risk"]["total_wallet_exposure_limit"] *= 2
    result, _ = _scenario(changed, scenarios[0], context)
    assert result["bot"]["long"]["universe"]["minimum_coin_age_days"] == 14.0


@pytest.mark.parametrize("side", ["long", "short"])
def test_ordinary_inference_skips_other_side_ignored_and_disabled_members(tmp_path, side):
    import backtest as bt
    config = _cache_config(tmp_path, (0.0, 0.0))
    config["bot"][side]["risk"].update(max_positions=1, total_wallet_exposure_limit=1.0)
    config["bot"][side]["universe"].update(minimum_coin_age_days=7.0, approved_coins=["AAA", "BBB"], ignored_coins=["BBB"])
    mss = {coin: {"inception_timestamp_ms": None} for coin in ("AAA", "BBB")}
    assert bt._automatic_cache_unresolved_inception_coins(config, ["AAA", "BBB"], mss) == ["AAA"]
    config["bot"][side]["risk"]["total_wallet_exposure_limit"] = 0.0
    assert bt._automatic_cache_unresolved_inception_coins(config, ["AAA", "BBB"], mss) == []


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("venue,expected", [("binance", []), ("bybit", ["BTC"])])
def test_cache_permission_uses_actual_column_venue(tmp_path, monkeypatch, scoped, venue, expected):
    import backtest as bt
    import utils
    config = _cache_config(tmp_path, (7.0, 0.0))
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    permission = "bybit::BTC/USDT:USDT"
    config["bot"]["long"]["universe"]["approved_coins"] = [permission]
    monkeypatch.setattr(utils, "_load_coin_to_symbol_map", lambda _: {
        "BTC": ["BTC/USDT:USDT"], "BTC/USDT:USDT": ["BTC/USDT:USDT"],
    })
    mss = {"BTC": {"exchange": venue, "symbol": "BTC/USDT:USDT", "inception_timestamp_ms": None}}
    kwargs = {"inception_fact_coins": frozenset({permission})} if scoped else {}
    assert bt._automatic_cache_unresolved_inception_coins(config, ["BTC"], mss, **kwargs) == expected


def test_reconciliation_keeps_settlement_and_ambiguity_contract(monkeypatch):
    import utils
    from backtest_universe import reconcile_backtest_coin_identifiers
    markets = {"BTC/USDT:USDT": ["BTC/USDT:USDT"], "BTC/USD:BTC": ["BTC/USD:BTC"],
               "BTC": ["BTC/USDT:USDT"]}
    monkeypatch.setattr(utils, "_load_coin_to_symbol_map", lambda _: markets)
    coins = {"BTC/USDT:USDT", "BTC/USD:BTC"}
    assert reconcile_backtest_coin_identifiers(["bybit::BTC/USD:BTC"], coins, ["bybit"]) == ["BTC/USD:BTC"]
    with pytest.raises(ValueError, match="multiple prepared coins"):
        reconcile_backtest_coin_identifiers(["BTC"], {"BTC/USDT:USDT", "bybit::BTC/USDT:USDT"}, ["bybit"])


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["backtest", "optimizer"])
@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
async def test_actual_producers_reuse_once_resolved_scenario_file(
    tmp_path, monkeypatch, producer, kind,
):
    from pathlib import Path
    import suite_runner
    import optimize_suite
    from config.universe import require_side_universe_sources, universe_source_view
    from test_shared_approved_suite import (
        prepared, stub_suite_sources, dataset, payload, masks,
    )
    from backtest import execute_backtest
    from shared_arrays import SharedArrayManager

    config, data, files, _ = prepared(tmp_path, kind, "file")
    file = tmp_path / "scenario.json"
    file.write_text('["COIN02"]')
    suite = {"scenarios": [{"label": "frozen", "overrides": {
        "bot.global.universe.approved_coins": str(file),
    }}]}
    config["bot"]["long"]["universe"]["minimum_coin_age_days"] = 1.0
    module = suite_runner if producer == "backtest" else optimize_suite
    stub_suite_sources(monkeypatch, module, data)
    reads = []
    original_open = Path.open
    def record(path, *args, **kwargs):
        if path == file:
            reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", record)

    async def master(config, *args, **kwargs):
        contexts = kwargs["prepared_universes"]
        assert contexts["frozen"].inception_coins == frozenset({"COIN02"})
        assert require_side_universe_sources(kwargs["scenario_policy_config"], "approved_coins") == {
            side: ["COIN00", "COIN02"] for side in ("long", "short")
        }
        assert len(reads) == 1
        for source in (*files, file):
            source.unlink()
        return {"binance": dataset(data)}
    monkeypatch.setattr(module, "prepare_master_datasets", master)

    def verify(context):
        assert masks(context, data) == {side: [False, True] for side in ("long", "short")}
        assert universe_source_view(context)["bot"]["global"]["universe"]["approved_coins"] == str(file)
        execute_backtest(payload(context, data), context)

    if producer == "optimizer":
        manager = SharedArrayManager()
        try:
            contexts, _ = await optimize_suite.prepare_suite_contexts(config, suite, shared_array_manager=manager)
            verify(contexts[0].config)
        finally:
            manager.cleanup()
    else:
        async def run(scenario, base, datasets, coins, ignored, exchanges, available, output, **kwargs):
            context, _ = suite_runner.apply_scenario(
                base, scenario, coins, ignored, exchanges, available,
                base_coins=kwargs["base_coins"], base_ignored=kwargs["base_ignored"],
                prepared_universe=kwargs["prepared_universe"],
            )
            verify(context)
            return suite_runner.ScenarioResult(scenario, {}, {"stats": {}}, 0, None)
        monkeypatch.setattr(suite_runner, "run_backtest_scenario", run)
        monkeypatch.setattr(suite_runner, "evaluation_implementation_identity", lambda: {"test": True})
        await suite_runner.run_backtest_suite_async(
            config, suite, disable_plotting=True, suite_output_root=tmp_path / "results",
        )
    assert len(reads) == 1
