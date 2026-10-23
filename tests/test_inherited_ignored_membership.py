"""Ignore edits use frozen approvals acquired before eligibility filtering."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import apply_bot_config_patch, bot_policy_source_view
from config.universe import (
    get_side_universe_sources,
    materialize_universe_snapshot,
    runtime_universe_snapshot,
)


def prepared(approved, ignored=("ETH",), *, side=None):
    bot = {"global": {"universe": {
        "approved_coins": approved, "ignored_coins": list(ignored),
    }}}
    if side is not None:
        bot[side] = {"universe": {"approved_coins": ["ETH", "SOL"]}}
    return prepare_config({"config_version": "v9.0.0", "bot": bot}, verbose=False)


def patched(config, selector, value):
    result = deepcopy(config)
    assert apply_bot_config_patch(result, {selector: value})
    return prepare_config(result, verbose=False)


@pytest.mark.parametrize("value,expected", [([], ["BTC", "ETH"]), (None, ["BTC", "ETH"]), (["BTC"], ["ETH"])])
def test_common_ignore_edit_restores_acquired_literal_approvals(value, expected):
    config = prepared(["BTC", "ETH"])
    assert get_side_universe_sources(config, "approved_coins") == {side: ["BTC"] for side in ("long", "short")}
    result = patched(config, "bot.global.universe.ignored_coins", value)
    assert get_side_universe_sources(result, "approved_coins") == {side: expected for side in ("long", "short")}
    assert get_side_universe_sources(config, "approved_coins") == {side: ["BTC"] for side in ("long", "short")}


@pytest.mark.parametrize("file_change", ["delete", "replace"])
@pytest.mark.parametrize("side", [None, "long", "short"])
def test_ignore_clear_uses_frozen_file_approvals_and_independent_pins(tmp_path, file_change, side):
    path = tmp_path / "approved.json"
    path.write_text('["BTC", "ETH"]')
    config = prepared(str(path), side=side)
    source = bot_policy_source_view(config)
    if file_change == "delete":
        path.unlink()
    else:
        path.write_text('["XRP"]')
    result = patched(config, "bot.global.universe.ignored_coins", [])
    expected = {pside: ["ETH", "SOL"] if pside == side else ["BTC", "ETH"] for pside in ("long", "short")}
    assert get_side_universe_sources(result, "approved_coins") == expected
    assert bot_policy_source_view(result)["bot"]["global"]["universe"]["approved_coins"] == str(path)
    assert bot_policy_source_view(config) == source


def test_explicit_side_null_ignores_clear_only_that_side(tmp_path):
    path = tmp_path / "approved.json"
    path.write_text('["BTC", "ETH"]')
    config = prepared(str(path))
    path.unlink()
    result = patched(config, "bot.long.universe.ignored_coins", None)
    assert get_side_universe_sources(result, "approved_coins") == {"long": ["BTC", "ETH"], "short": ["BTC"]}
    source = bot_policy_source_view(result)["bot"]
    assert source["long"]["universe"]["ignored_coins"] is None
    assert "ignored_coins" not in source.get("short", {}).get("universe", {})


@pytest.mark.parametrize("available,expected", [(["BTC", "ETH"], ["BTC", "ETH"]), (["BTC"], ["BTC"])])
def test_snapshot_retains_acquired_approvals_within_prepared_columns(tmp_path, available, expected):
    path = tmp_path / "approved.json"
    path.write_text('["BTC", "ETH"]')
    config = prepared(str(path))
    snapshot = runtime_universe_snapshot(config, available_coins=available)
    path.unlink()
    target = prepare_config(bot_policy_source_view(config), live_only=True, verbose=False)
    target = materialize_universe_snapshot(target, snapshot)
    result = patched(target, "bot.global.universe.ignored_coins", [])
    assert get_side_universe_sources(result, "approved_coins") == {side: expected for side in ("long", "short")}


def test_changing_approval_source_still_requires_new_acquisition(tmp_path):
    path = tmp_path / "approved.json"
    path.write_text('["BTC", "ETH"]')
    config = prepared(str(path))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        patched(config, "bot.global.universe.approved_coins", str(tmp_path / "missing.json"))


async def formatted(tmp_path, monkeypatch, ignored, exchanges=("binance",)):
    import utils

    monkeypatch.chdir(tmp_path)
    markets = {f"{coin}/USDT:USDT": {"id": f"{coin}USDT", "swap": True,
               "linear": True, "active": True, "base": coin} for coin in ("BTC", "ETH")}
    for exchange in exchanges:
        assert utils.create_coin_symbol_map_cache(exchange, markets, verbose=False)

    async def load(*_args, **_kwargs):
        return markets

    monkeypatch.setattr(utils, "load_markets", load)
    path = tmp_path / "approved.json"
    path.write_text('["BTC", "ETH"]')
    config = prepare_config({"config_version": "v9.0.0", "bot": {"global": {"universe": {
        "approved_coins": str(path), "ignored_coins": ignored,
    }}}}, live_only=True, verbose=False)
    await utils.format_approved_ignored_coins(config, list(exchanges), verbose=False)
    path.unlink()

    def forbidden(*_args, **_kwargs):
        pytest.fail("replay consulted a mutable market cache or network")

    for name in ("load_markets", "_load_coin_to_symbol_map", "_load_symbol_to_coin_map"):
        monkeypatch.setattr(utils, name, forbidden)
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["BTCUSDT", "binance::BTCUSDT"])
async def test_alias_filter_survives_numeric_reprepare_and_partial_ignore_edit(tmp_path, monkeypatch, alias):
    config = await formatted(tmp_path, monkeypatch, [alias, "ETHUSDT"])
    assert get_side_universe_sources(config, "approved_coins") == {side: [] for side in ("long", "short")}
    result = patched(config, "bot.global.risk.total_wallet_exposure_limit", 2.0)
    assert get_side_universe_sources(result, "approved_coins") == {side: [] for side in ("long", "short")}
    result = patched(result, "bot.global.universe.ignored_coins", ["ETHUSDT"])
    assert get_side_universe_sources(result, "approved_coins") == {side: ["BTC"] for side in ("long", "short")}


@pytest.mark.asyncio
async def test_new_alias_ignore_uses_acquired_facts_not_mutable_maps(tmp_path, monkeypatch):
    config = await formatted(tmp_path, monkeypatch, [])
    result = patched(config, "bot.global.universe.ignored_coins", ["BTCUSDT"])
    assert get_side_universe_sources(result, "approved_coins") == {side: ["ETH"] for side in ("long", "short")}


@pytest.mark.asyncio
async def test_new_alias_ignore_requires_missing_preparation_facts(tmp_path, monkeypatch):
    import utils

    monkeypatch.chdir(tmp_path)
    async def forbidden(*_args, **_kwargs):
        pytest.fail("plain lists must not trigger an added market fetch")
    monkeypatch.setattr(utils, "load_markets", forbidden)
    config = prepare_config({"config_version": "v9.0.0", "bot": {"global": {"universe": {
        "approved_coins": ["BTC"], "ignored_coins": [],
    }}}}, live_only=True, verbose=False)
    await utils.format_approved_ignored_coins(config, ["binance"], verbose=False)
    assert get_side_universe_sources(config, "approved_coins") == {side: ["BTC"] for side in ("long", "short")}
    with pytest.raises(ValueError, match="prepared market alias facts unavailable.*prepare universe membership"):
        patched(config, "bot.global.universe.ignored_coins", ["BTCUSDT"])


@pytest.mark.asyncio
async def test_unrelated_cached_markets_do_not_prove_an_acquired_alias_basis(tmp_path, monkeypatch):
    import utils
    from config.universe import approval_allows_coin

    monkeypatch.chdir(tmp_path)
    markets = {"ETH/USDT:USDT": {"id": "ETHUSDT", "swap": True,
               "linear": True, "active": True, "base": "ETH"}}
    assert utils.create_coin_symbol_map_cache("binance", markets, verbose=False)
    async def forbidden(*_args, **_kwargs):
        pytest.fail("cached preparation must not fetch markets")
    monkeypatch.setattr(utils, "load_markets", forbidden)
    config = prepare_config({"config_version": "v9.0.0", "bot": {"global": {"universe": {
        "approved_coins": ["BTC"], "ignored_coins": [],
    }}}}, live_only=True, verbose=False)
    await utils.format_approved_ignored_coins(config, ["binance"], verbose=False)
    view = config["bot"]["long"]["universe"]["approved_coins"]
    assert approval_allows_coin(view, "BTC", exchange="binance", market_symbol="BTC/USDT:USDT")
    with pytest.raises(ValueError, match="prepared market alias facts unavailable"):
        patched(config, "bot.global.universe.ignored_coins", ["BTCUSDT"])


@pytest.mark.asyncio
async def test_loaded_full_metadata_can_prove_a_market_absent_on_one_venue(tmp_path, monkeypatch):
    import utils
    from config.universe import approval_allows_coin

    monkeypatch.chdir(tmp_path)
    markets = {exchange: {f"{coin}/USDT:USDT": {"id": f"{coin}USDT", "swap": True,
               "linear": True, "active": True, "base": coin}}
               for exchange, coin in (("binance", "BTC"), ("bybit", "ETH"))}
    for exchange, values in markets.items():
        assert utils.create_coin_symbol_map_cache(exchange, values, verbose=False)
    async def load(exchange, **_kwargs):
        return markets[exchange]
    monkeypatch.setattr(utils, "load_markets", load)
    config = prepare_config({"config_version": "v9.0.0", "bot": {"global": {"universe": {
        "approved_coins": ["BTC"], "ignored_coins": ["bybit::BTCUSDT"],
    }}}}, live_only=True, verbose=False)
    await utils.format_approved_ignored_coins(config, ["binance", "bybit"], verbose=False)
    view = config["bot"]["long"]["universe"]["approved_coins"]
    assert view.market_aliases.unavailable_venues == ()
    assert approval_allows_coin(view, "BTC", exchange="binance", market_symbol="BTC/USDT:USDT")
    assert not approval_allows_coin(view, "BTC", exchange="bybit", market_symbol="BTC/USDT:USDT")
    assert get_side_universe_sources(patched(config, "bot.global.universe.ignored_coins", ["binance::BTCUSDT"]),
                                     "approved_coins") == {side: [] for side in ("long", "short")}


@pytest.mark.asyncio
async def test_suite_keeps_declared_ignore_alias_normalization_facts(tmp_path, monkeypatch):
    import utils
    from suite_runner import validate_suite_side_coin_lists

    monkeypatch.chdir(tmp_path)
    markets = {f"{coin}/USDT:USDT": {"id": f"{coin}USDT", "swap": True,
               "linear": True, "active": True, "base": coin} for coin in ("BTC", "ETH")}
    assert utils.create_coin_symbol_map_cache("binance", markets, verbose=False)
    async def load(*_args, **_kwargs):
        return markets
    monkeypatch.setattr(utils, "load_markets", load)
    config = prepare_config({"config_version": "v9.0.0", "bot": {
        "global": {"universe": {"approved_coins": ["BTC"]}},
        "long": {"universe": {"ignored_coins": ["ETH"]}},
        "short": {"universe": {"ignored_coins": ["ETHUSDT"]}},
    }}, live_only=True, verbose=False)
    await utils.format_approved_ignored_coins(config, ["binance"], verbose=False)
    def forbidden(*_args, **_kwargs):
        pytest.fail("suite validation reread mutable market aliases")
    for name in ("_load_coin_to_symbol_map", "_load_symbol_to_coin_map"):
        monkeypatch.setattr(utils, name, forbidden)
    validate_suite_side_coin_lists(config, ["binance"])
    assert get_side_universe_sources(config, "approved_coins") == {side: ["BTC"] for side in ("long", "short")}


def test_unrelated_market_catalog_does_not_change_acquired_identity():
    from config.universe import PreparedMarketAliases
    from utils import _build_coin_symbol_maps

    def projection(coins):
        markets = {f"{coin}/USDT:USDT": {"id": f"{coin}USDT", "swap": True,
                   "linear": True, "active": True, "base": coin} for coin in coins}
        aliases, _ = _build_coin_symbol_maps(markets, "USDT", exchange="binance")
        return PreparedMarketAliases((("binance", tuple((key, tuple(value)) for key, value in sorted(aliases.items())),
            tuple((symbol, market["id"]) for symbol, market in sorted(markets.items()))),)).for_bound_inputs(["BTC"])

    assert projection(["BTC"]).fingerprint == projection(["BTC", "DOGE"]).fingerprint


@pytest.mark.asyncio
async def test_cache_only_alias_facts_match_after_cold_restart(tmp_path, monkeypatch):
    import utils

    monkeypatch.chdir(tmp_path)
    markets = {"BTC/USDT:USDT": {"id": "BTCUSDT", "swap": True,
               "linear": True, "active": True, "base": "BTC"}}
    assert utils.create_coin_symbol_map_cache("binance", markets, verbose=False)
    raw = {"config_version": "v9.0.0", "bot": {"global": {"universe": {"approved_coins": ["BTC"]}}}}
    async def capture():
        config = prepare_config(raw, live_only=True, verbose=False)
        await utils.format_approved_ignored_coins(config, ["binance"], verbose=False)
        return config["bot"]["long"]["universe"]["approved_coins"].market_aliases
    warm = await capture()
    utils._COIN_TO_SYMBOL_CACHE.pop("binance", None)
    cold = await capture()
    assert warm.fingerprint == cold.fingerprint
    assert warm.subtract(["BTC"], ["BTCUSDT"]) == cold.subtract(["BTC"], ["BTCUSDT"]) == []


@pytest.mark.asyncio
async def test_snapshot_alias_column_ceiling_never_restores_an_unprepared_venue(tmp_path, monkeypatch):
    config = await formatted(tmp_path, monkeypatch, ["bybit::BTCUSDT", "ETH"], ("binance", "bybit"))
    expected = {side: ["binance::BTCUSDT"] for side in ("long", "short")}
    assert get_side_universe_sources(config, "approved_coins") == expected
    snapshot = runtime_universe_snapshot(config, available_coins=["binance::BTCUSDT"])
    target = materialize_universe_snapshot(config, snapshot)
    target = patched(target, "bot.global.universe.ignored_coins", [])
    assert get_side_universe_sources(target, "approved_coins") == expected
    # A later wider dataset argument cannot widen this snapshot's acquired ceiling.
    target = materialize_universe_snapshot(target, runtime_universe_snapshot(target,
        available_coins=["binance::BTCUSDT", "bybit::BTCUSDT", "ETH"]))
    assert patched(target, "bot.global.universe.ignored_coins", None)["bot"]["long"]["universe"]["approved_coins"] == expected["long"]


@pytest.mark.asyncio
async def test_generic_column_does_not_widen_a_qualified_survivor(tmp_path, monkeypatch):
    config = await formatted(tmp_path, monkeypatch, ["bybit::BTCUSDT", "ETH"], ("binance", "bybit"))
    target = materialize_universe_snapshot(config, runtime_universe_snapshot(config, available_coins=["BTC"]))
    result = patched(target, "bot.global.risk.total_wallet_exposure_limit", 2.0)
    assert get_side_universe_sources(result, "approved_coins") == {
        side: ["binance::BTCUSDT"] for side in ("long", "short")}


def test_acquired_membership_edits_decline_fast_projection_and_fail_full_reconciliation():
    from config.bot_inheritance import reconciled_bot_policy
    from optimization.native_planning import _compile_projection_guard

    config = prepared(["BTC", "ETH"])
    view = config["bot"]["long"]["universe"]["approved_coins"]
    guard = _compile_projection_guard(view, ())
    assert guard(view)
    view.acquired_members = ("BTC", "XRP")
    assert not guard(view)
    with pytest.raises(ValueError, match="prepared membership edits"):
        reconciled_bot_policy(config)


@pytest.mark.asyncio
async def test_frozen_alias_facts_share_on_copy_and_roundtrip_worker_pickle(tmp_path, monkeypatch):
    import pickle

    config = await formatted(tmp_path, monkeypatch, ["BTCUSDT"])
    view = config["bot"]["long"]["universe"]["approved_coins"]
    assert deepcopy(view).market_aliases is view.market_aliases
    snapshot = runtime_universe_snapshot(config, available_coins=["BTC", "ETH"])
    restored = materialize_universe_snapshot(config, pickle.loads(pickle.dumps(snapshot)))
    assert get_side_universe_sources(patched(restored, "bot.global.universe.ignored_coins", []), "approved_coins") == {
        side: ["BTC", "ETH"] for side in ("long", "short")}


def test_scenario_ignore_clear_uses_frozen_approval_file(tmp_path):
    from suite_runner import SuiteScenario, apply_scenario

    path = tmp_path / "approved.json"
    path.write_text('["BTC", "ETH"]')
    config = prepared(str(path))
    path.unlink()
    scenario = SuiteScenario("clear", None, None, coins=None, ignored_coins=[])
    result, columns = apply_scenario(config, scenario, ["BTC", "ETH"], ["ETH"],
        ["binance"], {"BTC", "ETH"}, base_coins=["BTC", "ETH"], base_ignored=["ETH"], quiet=True)
    assert columns == ["BTC", "ETH"]
    assert get_side_universe_sources(result, "approved_coins") == {side: columns for side in ("long", "short")}
    assert get_side_universe_sources(config, "approved_coins") == {side: ["BTC"] for side in ("long", "short")}


@pytest.mark.asyncio
async def test_scenario_and_prepared_handoff_use_frozen_qualified_aliases(tmp_path, monkeypatch):
    from suite_runner import SuiteScenario, apply_scenario, _prepare_scenario_universes

    config = await formatted(tmp_path, monkeypatch, ["bybit::BTCUSDT", "ETH"], ("binance", "bybit"))
    scenario = SuiteScenario("base", None, None, coins=None, ignored_coins=None,
                             overrides={"bot.global.risk.total_wallet_exposure_limit": 2.0})
    kwargs = dict(master_coins=["BTC", "ETH"], master_ignored=[], available_exchanges=["binance", "bybit"],
                  available_coins={"BTC", "ETH"}, base_coins=["BTC", "ETH"], base_ignored=[])
    contexts = _prepare_scenario_universes(config, [scenario], master_coins=kwargs["master_coins"],
        master_ignored=[], available_exchanges=kwargs["available_exchanges"],
        base_coins=kwargs["base_coins"], base_ignored=[])
    for prepared_context in (None, contexts["base"]):
        result, _ = apply_scenario(config, scenario, **kwargs, quiet=True, prepared_universe=prepared_context)
        assert get_side_universe_sources(result, "approved_coins") == {
            side: ["binance::BTCUSDT"] for side in ("long", "short")}


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_native_descriptor_identity_and_ceiling_retain_preignored_acquisition(kind):
    from config.inherited_bounds import prepare_bound_plan
    from config_utils import clean_config
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.native_planning import execution_key
    from shared_arrays import SharedArraySpec
    from tools.gpu_parity import build_parser, fixture_inputs

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    config = clean_config(config)
    config["bot"]["global"]["universe"].update(
        approved_coins=["COIN00", "COIN01", "COIN02"], ignored_coins=["COIN02"])
    for side in ("long", "short"):
        config["bot"][side]["universe"]["approved_coins"] = None
        config["bot"][side]["universe"]["ignored_coins"] = ["COIN02"]
    config = prepare_config(config, verbose=False)
    coins = ["COIN00", "COIN01"]
    config["backtest"]["coins"] = {"binance": coins}
    prepare_bound_plan(config)
    specs = [SharedArraySpec(name, value.shape, value.dtype.str)
             for name, value in (("candles", candles), ("btc", btc), ("time", timestamps))]
    dataset = PreparedGpuDataset(config=config, markets=markets, exchange="binance", hlcvs=specs[0],
        btc=specs[1], timestamps=specs[2], candle_coins=["COIN00", "COIN01", "COIN02"],
        coin_indices=(0, 1), metrics=("adg_strategy_eq",))
    assert execution_key(dataset.policy_config()) == execution_key(config)
    restored = dataset.policy_config()
    assert apply_bot_config_patch(restored, {f"bot.{side}.universe.ignored_coins": [] for side in ("long", "short")})
    restored["backtest"].pop("coins")  # Authored reprepare excludes the runtime column transport.
    restored = patched(restored, "bot.global.universe.ignored_coins", [])
    assert get_side_universe_sources(restored, "approved_coins") == {side: coins for side in ("long", "short")}


@pytest.mark.asyncio
async def test_alias_and_column_fact_mutations_fail_existing_guards(tmp_path, monkeypatch):
    from config.bot_inheritance import reconciled_bot_policy
    from optimization.native_planning import _compile_projection_guard

    config = await formatted(tmp_path, monkeypatch, ["BTCUSDT"])
    for attribute, value in (("market_aliases", None), ("available_members", ("XRP",))):
        changed = deepcopy(config)
        view = changed["bot"]["long"]["universe"]["approved_coins"]
        guard = _compile_projection_guard(view, ())
        assert guard(view)
        setattr(view, attribute, value)
        assert not guard(view)
        with pytest.raises(ValueError, match="prepared membership edits"):
            reconciled_bot_policy(changed)


@pytest.mark.asyncio
async def test_acquired_alias_identity_survives_saved_result_resume(tmp_path, monkeypatch):
    import json
    import numpy as np
    from config_utils import clean_config
    from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
    from optimization.prepared_dataset_identity import PREPARED_DATASET_KEY, build_prepared_dataset_identity
    from optimize import _resume_config_mismatches
    from shared_arrays import SharedArrayManager

    config = await formatted(tmp_path, monkeypatch, ["BTCUSDT"])
    config["backtest"]["coins"] = {"binance": ["BTC", "ETH"]}
    manager = SharedArrayManager()
    try:
        candles = manager.create_from(np.full((8, 2, 4), 100.0))[0]
        btc = manager.create_from(np.full(8, 60000.0))[0]
        config[PREPARED_DATASET_KEY] = build_prepared_dataset_identity(config=config,
            hlcvs_specs={"binance": candles}, btc_usd_specs={"binance": btc},
            msss={"binance": {coin: {"symbol": f"{coin}/USDT:USDT"} for coin in ("BTC", "ETH")}},
            timestamps={"binance": np.arange(8, dtype=np.int64) * 60000})
        entry = clean_config(config)
        entry["backtest"]["coins"] = deepcopy(config["backtest"]["coins"])
        entry[CONTRACT_KEY] = build_evaluation_contract(config)
        assert _resume_config_mismatches(entry, config) == []
        saved = json.loads(json.dumps(entry))
        assert saved[CONTRACT_KEY]["prepared_data"] == config[PREPARED_DATASET_KEY]
        assert _resume_config_mismatches(saved, config) == []
    finally:
        manager.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("exchange,eligible", [("binance", True), ("bybit", False)])
async def test_qualified_policy_lowers_only_against_factual_dataset_venue(tmp_path, monkeypatch, kind, exchange, eligible):
    from backtest import build_backtest_payload
    from config.inherited_bounds import prepare_bound_plan
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.native_planning import execution_key
    from shared_arrays import SharedArraySpec
    from tools.gpu_parity import build_parser, fixture_inputs

    config = await formatted(tmp_path, monkeypatch, ["bybit::BTCUSDT", "ETH"], ("binance", "bybit"))
    assert apply_bot_config_patch(config, {"bot.portfolio.strategy_kind": kind, "bot.long.enabled": True, "bot.short.enabled": True})
    config["backtest"]["exchanges"] = [exchange]
    config = prepare_config(config, verbose=False)
    _, candles, original_markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--coins", "2", "--bars", "128",
    ]))
    markets = {coin: {**original_markets[f"COIN0{index}"], "exchange": exchange,
                     "symbol": f"{coin}/USDT:USDT", "coin": coin}
               for index, coin in enumerate(("BTC", "ETH"))}
    if "__meta__" in original_markets:
        markets["__meta__"] = deepcopy(original_markets["__meta__"])
    config["backtest"]["coins"] = {exchange: ["BTC", "ETH"]}
    prepare_bound_plan(config)
    specs = [SharedArraySpec(name, value.shape, value.dtype.str)
             for name, value in (("candles", candles), ("btc", btc), ("time", timestamps))]
    dataset = PreparedGpuDataset(config=config, markets=markets, exchange=exchange, hlcvs=specs[0],
        btc=specs[1], timestamps=specs[2], candle_coins=["BTC", "ETH"], metrics=("adg_strategy_eq",))
    assert execution_key(dataset.policy_config()) == execution_key(config)
    source = bot_policy_source_view(config)
    for runtime in (config, dataset.execution_config()):
        payload = build_backtest_payload(candles, markets, runtime, exchange, btc, timestamps)
        assert {side: [row[side]["entry_eligible"] for row in payload.bot_params_list]
                for side in ("long", "short")} == {side: [eligible, False] for side in ("long", "short")}
    assert bot_policy_source_view(config) == source
    # Clearing ignores may admit only this descriptor's actual venue/columns.
    restored = dataset.policy_config()
    restored["backtest"].pop("coins")
    restored = patched(restored, "bot.global.universe.ignored_coins", [])
    assert all(value.startswith(f"{exchange}::") for value in get_side_universe_sources(restored, "approved_coins")["long"])
