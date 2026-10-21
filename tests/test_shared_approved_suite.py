"""Suite contexts freeze execution membership without changing source policy."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from backtest import build_backtest_payload, execute_backtest
from config import prepare_config
from config.universe import (
    materialize_side_universe_sources,
    require_side_universe_sources,
    universe_source_view,
)
from config_utils import clean_config, dump_config
from suite_runner import SuiteScenario, apply_scenario
from tools.gpu_parity import build_parser, fixture_inputs


def prepared(tmp_path, strategy_kind, source_kind, *, pin=False):
    config, candles, markets, btc, timestamps = fixture_inputs(
        build_parser().parse_args(
            [
                "--fixture",
                strategy_kind,
                "--sides",
                "both",
                "--coins",
                "3",
                "--bars",
                "128",
            ]
        )
    )
    coins = list(config["backtest"]["coins"]["binance"])
    allowed = [coins[0], coins[2]]  # COIN01 is excluded by the ignored source.
    approvals, ignored = tmp_path / "approved.json", tmp_path / "ignored.json"
    approvals.write_text('["COIN00","COIN02"]')
    ignored.write_text('["COIN01"]')
    config["bot"]["global"]["universe"]["approved_coins"] = (
        "all" if source_kind == "all" else str(approvals)
    )
    for side in ("long", "short"):
        config["bot"][side]["universe"]["approved_coins"] = (
            allowed if pin and side == "short" else None
        )
        config["bot"][side]["universe"]["ignored_coins"] = str(ignored)
    config = prepare_config(
        clean_config(config),
        base_config_path=str(tmp_path / "input.json"),
        verbose=False,
    )
    config["backtest"]["coins"] = {"binance": coins}
    materialize_side_universe_sources(
        config, "approved_coins", {side: allowed for side in ("long", "short")}
    )
    materialize_side_universe_sources(
        config, "ignored_coins", {side: ["COIN01"] for side in ("long", "short")}
    )
    return (
        config,
        (candles, markets, btc, timestamps, coins),
        (approvals, ignored),
        allowed,
    )


def payload(config, data):
    candles, markets, btc, timestamps, coins = data
    selected = config["backtest"]["coins"]["binance"]
    indexes = [coins.index(coin) for coin in selected]
    selected_markets = {coin: deepcopy(markets[coin]) for coin in selected}
    if "__meta__" in markets:
        selected_markets["__meta__"] = deepcopy(markets["__meta__"])
    return build_backtest_payload(
        np.ascontiguousarray(candles[:, indexes]),
        selected_markets,
        config,
        "binance",
        btc,
        timestamps,
    )


def masks(config, data):
    result = payload(config, data)
    return {
        side: [row[side]["entry_eligible"] for row in result.bot_params_list]
        for side in ("long", "short")
    }


def scenario(
    config, data, *, overrides=None, coins=None, ignored=None, base_coins=None
):
    available = data[-1]
    return apply_scenario(
        config,
        SuiteScenario(
            "base", None, None, coins=coins, ignored_coins=ignored, overrides=overrides
        ),
        master_coins=available,
        master_ignored=["COIN01", "COIN02"],
        available_exchanges=["binance"],
        available_coins=set(available),
        base_coins=available if base_coins is None else base_coins,
        base_ignored=["COIN01"],
    )


def mutate_sources(files, mutation):
    for path in files:
        if mutation == "deleted":
            path.unlink()
        else:
            path.write_text('["COIN02"]')


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("source_kind", ["all", "file"])
@pytest.mark.parametrize("mutation", ["changed", "deleted"])
@pytest.mark.parametrize("override_kind", ["none", "numeric", "approved", "ignored"])
def test_scenario_keeps_unchanged_prepared_fields_and_source_pins(
    tmp_path, monkeypatch, kind, source_kind, mutation, override_kind
):
    config, data, files, allowed = prepared(tmp_path, kind, source_kind, pin=True)
    expected = masks(config, data)
    original_policy = universe_source_view(config)
    mutate_sources(files, mutation)
    original_open = Path.open

    def no_old_source_read(path, *args, **kwargs):
        if path in files:
            raise AssertionError("unchanged prepared coin source was reread")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_old_source_read)
    overrides = {
        "none": {},
        "numeric": {"bot.global.warmup.span_multiplier": 0.8},
        "approved": {"bot.global.universe.approved_coins": allowed},
        "ignored": {
            f"bot.{side}.universe.ignored_coins": ["COIN01"]
            for side in ("long", "short")
        },
    }[override_kind]
    result, _ = scenario(config, data, overrides=overrides)
    assert masks(result, data) == expected
    for field in ("approved_coins", "ignored_coins"):
        assert all(
            isinstance(value, list)
            for value in require_side_universe_sources(result, field).values()
        )
    policy = universe_source_view(result)
    assert policy["bot"]["long"]["universe"]["approved_coins"] is None
    assert policy["bot"]["short"]["universe"]["approved_coins"] == allowed
    if override_kind != "approved":
        assert (
            policy["bot"]["global"]["universe"]
            == original_policy["bot"]["global"]["universe"]
        )
    if override_kind != "ignored":
        assert all(
            policy["bot"][side]["universe"]["ignored_coins"]
            == original_policy["bot"][side]["universe"]["ignored_coins"]
            for side in ("long", "short")
        )
    assert universe_source_view(config) == original_policy


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("source_kind", ["all", "file"])
def test_actual_cpu_scenario_matches_frozen_base_after_source_deletion(
    tmp_path, kind, source_kind
):
    config, data, files, _ = prepared(tmp_path, kind, source_kind)
    baseline = execute_backtest(payload(config, data), config)
    mutate_sources(files, "deleted")
    result, _ = scenario(config, data)
    actual = execute_backtest(payload(result, data), result)
    np.testing.assert_array_equal(actual[0], baseline[0])
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert actual[2].keys() == baseline[2].keys()
    exported = tmp_path / "export.json"
    dump_config(result, str(exported), clean=True)
    import json

    authored = json.loads(exported.read_text())
    assert authored["bot"]["long"]["universe"]["approved_coins"] is None
    assert authored["bot"]["long"]["universe"]["ignored_coins"] == str(files[1])
    files[0].write_text('["COIN00","COIN02"]')
    files[1].write_text('["COIN01"]')
    reloaded = prepare_config(authored, base_config_path=str(exported), verbose=False)
    assert (
        universe_source_view(reloaded)["bot"]["long"]["universe"]["approved_coins"]
        is None
    )


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_explicit_scenario_lists_replace_both_sources_without_reading_deleted_base(
    tmp_path, kind
):
    config, data, files, _ = prepared(tmp_path, kind, "file")
    mutate_sources(files, "deleted")
    result, selected = scenario(config, data, coins=["COIN02"], ignored=[])
    assert selected == ["COIN02"]
    assert masks(result, data) == {"long": [True], "short": [True]}
    assert all(
        universe_source_view(result)["bot"][side]["universe"]["approved_coins"]
        == ["COIN02"]
        for side in ("long", "short")
    )


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_extra_master_columns_do_not_expand_inherited_scenario_membership(
    tmp_path, kind
):
    config, data, files, allowed = prepared(tmp_path, kind, "file")
    mutate_sources(files, "deleted")
    # The master dataset includes another scenario's explicit COIN01 column.
    result, selected = scenario(config, data, base_coins=allowed)
    assert selected == allowed
    assert require_side_universe_sources(result, "approved_coins") == {
        side: allowed for side in ("long", "short")
    }
    assert require_side_universe_sources(result, "ignored_coins") == {
        side: ["COIN01"] for side in ("long", "short")
    }
    assert masks(result, data) == {"long": [True, True], "short": [True, True]}


@pytest.mark.parametrize("ignored", ["COIN01", "file"])
def test_clean_scalar_sources_preserve_policy_through_dump_reload(tmp_path, ignored):
    config, data, files, allowed = prepared(tmp_path, "ema_anchor", "file", pin=True)
    if ignored != "file":
        for side in ("long", "short"):
            config["bot"][side]["universe"]["ignored_coins"] = ignored
        materialize_side_universe_sources(
            config, "ignored_coins", {side: ["COIN01"] for side in ("long", "short")}
        )
    original = deepcopy(config)
    cleaned = clean_config(config)
    assert clean_config(cleaned) == cleaned
    assert universe_source_view(config) == universe_source_view(original)
    assert cleaned["bot"]["long"]["universe"]["approved_coins"] is None
    assert cleaned["bot"]["short"]["universe"]["approved_coins"] == allowed
    assert cleaned["bot"]["long"]["universe"]["ignored_coins"] == (
        str(files[1]) if ignored == "file" else ignored
    )
    destination = tmp_path / "moved" / "config.json"
    destination.parent.mkdir()
    dump_config(config, str(destination), clean=True)
    from config.universe import resolve_universe_sources
    import json

    reloaded = prepare_config(
        json.loads(destination.read_text()),
        base_config_path=str(destination),
        verbose=False,
    )
    for field in ("approved_coins", "ignored_coins"):
        members, errors = resolve_universe_sources(reloaded, field)
        assert not errors
        materialize_side_universe_sources(reloaded, field, members)
    reloaded["backtest"]["coins"] = config["backtest"]["coins"]
    assert masks(reloaded, data) == masks(config, data)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_source_replacements_resolve_once_and_preserve_unaffected_field(
    tmp_path, monkeypatch, kind
):
    config, data, files, _ = prepared(tmp_path, kind, "file")
    replacement = tmp_path / "replacement.json"
    replacement.write_text('["COIN00"]')
    original_open = Path.open
    reads = []

    def read(path, *args, **kwargs):
        if path in files:
            raise AssertionError("unaffected prepared source was reread")
        if path == replacement:
            reads.append(path)
        return original_open(path, *args, **kwargs)

    mutate_sources(files, "deleted")
    monkeypatch.setattr(Path, "open", read)
    overrides = {"bot.global.universe.approved_coins": str(replacement)}
    result, _ = scenario(config, data, overrides=overrides)
    assert reads == [replacement]
    assert masks(result, data) == {
        side: [True, False, False] for side in ("long", "short")
    }
    assert require_side_universe_sources(result, "ignored_coins") == {
        side: ["COIN01"] for side in ("long", "short")
    }


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("replace_approved", [False, True])
def test_clearing_ignored_cannot_expand_frozen_approvals_without_explicit_replacement(
    tmp_path, kind, replace_approved
):
    config, data, files, _ = prepared(tmp_path, kind, "file")
    mutate_sources(files, "deleted")
    overrides = {f"bot.{side}.universe.ignored_coins": [] for side in ("long", "short")}
    if replace_approved:
        overrides["bot.global.universe.approved_coins"] = data[-1]
    result, _ = scenario(config, data, overrides=overrides)
    assert masks(result, data) == {
        side: ([True, True, True] if replace_approved else [True, False, True])
        for side in ("long", "short")
    }
    assert require_side_universe_sources(result, "ignored_coins") == {
        side: [] for side in ("long", "short")
    }


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_replacement_ignored_source_subtracts_without_rereading_approvals(
    tmp_path, kind
):
    config, data, files, _ = prepared(tmp_path, kind, "file")
    mutate_sources(files, "deleted")
    result, _ = scenario(
        config,
        data,
        overrides={
            f"bot.{side}.universe.ignored_coins": ["COIN02"]
            for side in ("long", "short")
        },
    )
    assert masks(result, data) == {
        side: [True, False, False] for side in ("long", "short")
    }


def stub_suite_sources(monkeypatch, module, data):
    """Keep real source reads/materialization; replace only market network I/O."""
    from config.universe import resolve_universe_sources

    async def no_network(*args, **kwargs):
        return {}

    async def resolve(config, *args, **kwargs):
        resolved = {}
        for field in ("approved_coins", "ignored_coins"):
            values, errors = resolve_universe_sources(config, field)
            assert not errors
            for side, members in values.items():
                if members == ["all"]:
                    values[side] = data[-1]
            resolved[field] = values
        for side in ("long", "short"):
            resolved["approved_coins"][side] = [
                coin
                for coin in resolved["approved_coins"][side]
                if coin not in resolved["ignored_coins"][side]
            ]
        for field, values in resolved.items():
            materialize_side_universe_sources(config, field, values)

    monkeypatch.setattr(module, "load_markets", no_network)
    monkeypatch.setattr(
        module, "reject_cross_exchange_market_identifier_collisions", no_network
    )
    monkeypatch.setattr(module, "format_approved_ignored_coins", resolve)


def dataset(data, specs=(None, None)):
    from suite_runner import ExchangeDataset

    candles, markets, btc, timestamps, coins = data
    return ExchangeDataset(
        "binance",
        coins,
        {coin: i for i, coin in enumerate(coins)},
        {coin: "binance" for coin in coins},
        ["binance"],
        candles,
        markets,
        btc,
        timestamps,
        "",
        hlcvs_spec=specs[0],
        btc_spec=specs[1],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("source_kind", ["all", "file"])
@pytest.mark.parametrize("mutation", ["changed", "deleted"])
async def test_optimizer_producer_freezes_memberships_through_format_context_and_replay(
    tmp_path, monkeypatch, kind, source_kind, mutation
):
    import optimize_suite
    from optimize import Evaluator, SuiteEvaluator, config_to_individual
    from optimization.native_datasets import NativeDatasetRegistry
    from shared_arrays import SharedArrayManager

    config, data, files, allowed = prepared(tmp_path, kind, source_kind)
    config["optimize"]["scoring"] = [{"metric": "adg_strategy_eq", "goal": "max"}]
    config["optimize"]["limits"] = []
    suite_config = {
        "scenarios": [
            {"label": "base"},
            {"label": "extra", "coins": ["COIN01"], "ignored_coins": []},
        ]
    }
    stub_suite_sources(monkeypatch, optimize_suite, data)
    manager = SharedArrayManager()
    suite = None
    try:
        specs = [manager.create_from(value)[0] for value in data[:1] + data[2:3]]
        base = Evaluator(
            {"binance": specs[0]},
            {"binance": specs[1]},
            {"binance": data[1]},
            config,
            timestamps={"binance": data[3]},
        )

        async def prepare_master(master, *args, **kwargs):
            assert require_side_universe_sources(master, "approved_coins") == {
                side: data[-1] for side in ("long", "short")
            }
            mutate_sources(files, mutation)
            return {"binance": dataset(data, specs)}

        monkeypatch.setattr(optimize_suite, "prepare_master_datasets", prepare_master)
        contexts, reducer = await optimize_suite.prepare_suite_contexts(
            config, suite_config, shared_array_manager=manager
        )
        assert [ctx.label for ctx in contexts] == ["base", "extra"]
        assert contexts[0].config["backtest"]["coins"]["binance"] == allowed
        assert contexts[1].config["backtest"]["coins"]["binance"] == ["COIN01"]
        ctx = contexts[0]
        assert (
            universe_source_view(ctx.config)["bot"]["long"]["universe"][
                "approved_coins"
            ]
            is None
        )
        assert require_side_universe_sources(ctx.config, "ignored_coins") == {
            side: ["COIN01"] for side in ("long", "short")
        }
        suite = SuiteEvaluator(base, contexts, reducer)
        candidate = suite.build_scenario_candidate_config(config, ctx)
        assert masks(candidate, data) == {
            side: [coin != "COIN01" for coin in allowed] for side in ("long", "short")
        }
        with NativeDatasetRegistry(suite) as registry:
            vector = config_to_individual(
                base.config, base.bounds, optimization_shape=base.optimization_shape
            )
            assert registry.planner.prepare("prepared", vector).requests
        candles, btc, indexes = suite.get_prepared_context_data(ctx, "binance")
        native = build_backtest_payload(
            candles,
            ctx.msss["binance"],
            candidate,
            "binance",
            btc,
            ctx.timestamps["binance"],
            coin_indices=indexes,
        )
        assert all(
            row["long"]["entry_eligible"] == (coin != "COIN01")
            for coin, row in zip(allowed, native.bot_params_list)
        )
        execute_backtest(native, candidate)
    finally:
        if suite is not None:
            suite.close()
        manager.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
async def test_backtest_suite_refresh_snapshot_and_preload_union_remain_separate(
    tmp_path, monkeypatch, kind
):
    import suite_runner

    config, data, files, _ = prepared(tmp_path, kind, "file")
    # Refresh at suite preparation observes this change once, before dataset preload.
    files[0].write_text('["COIN02"]')
    stub_suite_sources(monkeypatch, suite_runner, data)
    observed = {}

    async def prepare_master(master, *args, **kwargs):
        assert require_side_universe_sources(master, "approved_coins") == {
            side: ["COIN01", "COIN02"] for side in ("long", "short")
        }
        mutate_sources(files, "deleted")
        return {"binance": dataset(data)}

    async def run(
        scenario,
        base,
        datasets,
        master_coins,
        master_ignored,
        exchanges,
        available,
        output,
        **kwargs,
    ):
        context, _ = apply_scenario(
            base,
            scenario,
            master_coins=master_coins,
            master_ignored=master_ignored,
            available_exchanges=exchanges,
            available_coins=available,
            base_coins=kwargs["base_coins"],
            base_ignored=kwargs["base_ignored"],
        )
        observed[scenario.label] = (
            context["backtest"]["coins"]["binance"],
            masks(context, data),
        )
        execute_backtest(payload(context, data), context)
        return suite_runner.ScenarioResult(scenario, {}, {"stats": {}}, 0, None)

    monkeypatch.setattr(suite_runner, "prepare_master_datasets", prepare_master)
    monkeypatch.setattr(suite_runner, "run_backtest_scenario", run)
    monkeypatch.setattr(
        suite_runner, "evaluation_implementation_identity", lambda: {"test": True}
    )
    await suite_runner.run_backtest_suite_async(
        config,
        {
            "scenarios": [
                {"label": "base"},
                {"label": "extra", "coins": ["COIN01"], "ignored_coins": []},
            ]
        },
        disable_plotting=True,
        suite_output_root=tmp_path / "results",
    )
    assert observed["base"] == (
        ["COIN02"],
        {side: [True] for side in ("long", "short")},
    )
    assert observed["extra"] == (
        ["COIN01"],
        {side: [True] for side in ("long", "short")},
    )
