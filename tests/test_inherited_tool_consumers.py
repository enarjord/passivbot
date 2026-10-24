"""Tools choose explicit runtime profiles without changing lean authoring."""
import argparse
from copy import deepcopy
import json

import pytest

from config.bot_inheritance import bot_policy_source_view, reconciled_bot_policy
from config.inherited_bounds import get_bound_plan
from config.load import prepare_config
from config.schema import get_template_config
from config_utils import add_config_arguments, clean_config


def source(kind="trailing_martingale"):
    return {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "long": {"enabled": True}, "global": { "universe": {"approved_coins": ["AAA", "BBB"],
                    "ignored_coins": ["BBB"], "minimum_coin_age_days": 0.0}},
        "short": {"enabled": False}}}


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_explicit_preset_reopens_visible_domains_from_lean_common_weights(kind):
    from tools.optimizer_bounds import apply_recommended_bounds
    raw = source(kind)
    raw["bot"]["global"]["forager"] = {"score_weights": {
        "volume": 0.2, "volatility": 0.2, "ema_readiness": 0.2, "unilateralness": 0.4}}
    before = deepcopy(raw)
    exported = apply_recommended_bounds(raw)
    loaded = prepare_config(exported, verbose=False)
    assert len([binding for binding in get_bound_plan(loaded).bindings
                if binding.bound.low != binding.bound.high]) > 5
    for side in ("long", "short"):
        assert exported["optimize"]["bounds"][side]["forager"]["score_weights"]["unilateralness"] == [0.0, 1.0, 0.01]
    assert raw == before
    assert get_template_config()["optimize"]["bounds"] == {}


def test_common_bound_flags_are_discovered_without_authoring_any_default_range():
    parser = argparse.ArgumentParser()
    raw = get_template_config()
    registered = add_config_arguments(parser, raw)
    path = "optimize.bounds.global.risk.total_wallet_exposure_limit"
    assert path in registered
    assert getattr(parser.parse_args([]), path) is None
    args = parser.parse_args(["--" + path, "1,3,0.1"])
    assert getattr(args, path) == [1.0, 3.0, 0.1]
    assert raw["optimize"]["bounds"] == {}


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_reserved_cli_omission_does_not_create_clearing_source_pins(command):
    from config_utils import update_config_with_args
    raw = source()
    before = deepcopy(raw)
    parser = argparse.ArgumentParser()
    registered = add_config_arguments(parser, get_template_config(), command=command)
    args = parser.parse_args([])
    for side in ("long", "short"):
        assert not hasattr(args, f"bot.{side}.universe.ignored_coins")
    update_config_with_args(raw, args, allowed_keys=registered)
    assert raw["bot"] == before["bot"]


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_actual_gpu_fixture_and_explicit_benchmark_sampling_use_runtime_prototype(kind):
    from tools.gpu_parity import build_parser, fixture_inputs
    from tools.optimizer_benchmark.catalogue import Case
    from tools.optimizer_benchmark.workloads import sampled_population
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--bars", "128", "--coins", "2", "--sides", "both"]))
    assert candles.shape == (128, 2, 4)
    assert config["bot"]["long"]["strategy"][kind]
    population, vectors, bounds = sampled_population(config, Case(kind), 2, 7)
    assert len(population) == len(vectors) == 2 and bounds
    assert all(candidate["bot"]["short"]["strategy"][kind] for candidate in population)


def test_composition_reads_effective_lean_inputs_and_keeps_all_source_files(tmp_path):
    from tools.compose_coin_overrides import load_single_coin_config, compose_configs
    inputs = []
    for coin in ("AAA", "BBB"):
        raw = source()
        raw["bot"]["global"]["universe"].update(approved_coins=[coin], ignored_coins=[])
        path = tmp_path / f"{coin}.json"
        path.write_text(json.dumps(raw))
        inputs.append(load_single_coin_config(path))
    composed, report = compose_configs(inputs)
    loaded = prepare_config(composed, verbose=False)
    assert list(loaded["bot"]["long"]["universe"]["approved_coins"]) == ["AAA", "BBB"]
    assert report.coins == ["AAA", "BBB"]
    assert all(item.path.exists() for item in inputs)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_every_normalizer_plan_boundary_excludes_inactive_owner_defaults(monkeypatch, kind):
    import config.inherited_bounds as owner
    original = owner.prepare_bound_plan
    calls = []
    def checked(config, **kwargs):
        raw = reconciled_bot_policy(config).source_view()
        for side in ("long", "short"):
            assert "strategy" not in raw.get(side, {})
        calls.append(True)
        return original(config, **kwargs)
    monkeypatch.setattr(owner, "prepare_bound_plan", checked)
    prepared = prepare_config(source(kind), verbose=False)
    assert len(calls) >= 2
    assert set(prepared["bot"]["long"]["strategy"]) == {kind}


@pytest.mark.asyncio
async def test_actual_market_list_formatter_retains_omitted_sources_and_future_common_ignores(monkeypatch):
    import utils
    from config.bot_inheritance import apply_bot_config_patch
    config = prepare_config(source(), verbose=False)
    before = bot_policy_source_view(config)["bot"]
    async def no_io(*args, **kwargs):
        return None
    for name in ("reject_cross_exchange_market_identifier_collisions", "_coalesce_resolved_approved_markets", "_remove_resolved_ignored_markets"):
        monkeypatch.setattr(utils, name, no_io)
    await utils.format_approved_ignored_coins(config, ["binance"], verbose=False)
    assert bot_policy_source_view(config)["bot"] == before
    exported = clean_config(config)
    assert "universe" not in exported["bot"]["long"]
    assert "universe" not in exported["bot"]["short"]
    apply_bot_config_patch(config, {"bot.global.universe.ignored_coins": ["AAA"]})
    assert list(config["bot"]["long"]["universe"]["ignored_coins"]) == ["AAA"]


def test_complete_source_export_omits_inactive_fallback_but_keeps_explicit_inactive_pin():
    raw = source()
    assert set(clean_config(prepare_config(raw, verbose=False))["bot"]["global"]["strategy"]) == {"trailing_martingale"}
    raw["bot"]["global"]["strategy"] = {"ema_anchor": {"base_qty_fraction": 0.02}}
    exported = clean_config(prepare_config(raw, verbose=False))
    assert exported["bot"]["global"]["strategy"]["ema_anchor"]["base_qty_fraction"] == 0.02


@pytest.mark.asyncio
async def test_actual_optimizer_suite_formatter_keeps_common_gene_authority(monkeypatch):
    import optimize_suite
    import numpy as np
    from suite_runner import ExchangeDataset
    class NoSharedArrays:
        def create_from(self, _array):
            raise AssertionError("prepared master arrays must be borrowed")
    raw = source()
    raw["bot"]["global"]["universe"]["ignored_coins"] = []
    raw["backtest"] = {"start_date": "2024-01-01", "end_date": "2024-01-02",
        "exchanges": ["binance"], "suite_enabled": True, "scenarios": [{"label": "base"}]}
    raw["optimize"] = {"bounds": {"global": {"risk": {"total_wallet_exposure_limit": [1.0, 3.0]}}}}
    config = prepare_config(raw, verbose=False)
    async def no_io(*args, **kwargs):
        return {}
    async def prepared(*args, **kwargs):
        timestamps = np.arange(1441, dtype=np.int64) * 60_000 + 1704067200000
        dataset = ExchangeDataset(exchange="combined", coins=["AAA", "BBB"],
            coin_index={"AAA": 0, "BBB": 1}, coin_exchange={"AAA": "binance", "BBB": "binance"},
            available_exchanges=["binance"], hlcvs=np.ones((1441, 2, 4)),
            mss={"AAA": {}, "BBB": {}, "__meta__": {"data_interval_minutes": 1}},
            btc_usd_prices=np.ones(1441), timestamps=timestamps, cache_dir="",
            hlcvs_spec=object(), btc_spec=object())
        for coin in dataset.coins:
            dataset.mss[coin].update(qty_step=0.001, price_step=0.01, min_qty=0.001,
                                    min_cost=1.0, c_mult=1.0, maker=0.0, taker=0.0)
        return {"combined": dataset}
    for name in ("load_markets", "format_approved_ignored_coins", "reject_cross_exchange_market_identifier_collisions"):
        monkeypatch.setattr(optimize_suite, name, no_io)
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", prepared)
    contexts, _ = await optimize_suite.prepare_suite_contexts(config,
        optimize_suite.extract_suite_config(config, suite_override=None),
        shared_array_manager=NoSharedArrays())
    result = contexts[0].config
    exported = bot_policy_source_view(result)
    assert "risk" not in exported["bot"].get("long", {})
    assert "risk" not in exported["bot"].get("short", {})
    assert exported["optimize"]["bounds"] == raw["optimize"]["bounds"]
    binding = next(item for item in get_bound_plan(result).bindings
                   if item.key == "bot.global.risk.total_wallet_exposure_limit")
    assert binding.effective_targets == (("bot", "long", "risk", "total_wallet_exposure_limit"),
                                          ("bot", "short", "risk", "total_wallet_exposure_limit"))
