"""Fixed universe policy is prepared once before standalone candidate replay."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from backtest import build_backtest_payload, execute_backtest
from config.universe import (
    materialize_side_universe_sources,
    raw_side_universe_sources,
    require_side_universe_sources,
    resolve_universe_sources,
    universe_source_view,
)
from config_utils import clean_config
from tools.gpu_parity import build_parser, fixture_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize(
    "policy",
    ["all", "file", "list", "unchanged_all", "unchanged_file", "side", "ignored"],
)
async def test_actual_main_prepares_fixed_policy_before_single_evaluator_replay(
    tmp_path, monkeypatch, kind, policy
):
    import optimize
    from optimization.native_datasets import NativeDatasetRegistry
    from optimization.prepared_dataset_identity import PREPARED_DATASET_KEY

    config, candles, markets, btc, timestamps = fixture_inputs(
        build_parser().parse_args(
            ["--fixture", kind, "--sides", "both", "--coins", "3", "--bars", "128"]
        )
    )
    config = clean_config(config)
    coins = ["COIN00", "COIN01", "COIN02"]
    approved = tmp_path / "approved.json"
    approved.write_text('["COIN00","COIN02"]')
    ignored = tmp_path / "ignored.json"
    ignored.write_text('["COIN01"]')
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        '["COIN02"]' if policy in {"side", "ignored"} else '["COIN00","COIN02"]'
    )
    missing = tmp_path / "old-missing.json"
    config["bot"]["global"]["universe"]["approved_coins"] = str(missing)
    for side in ("long", "short"):
        config["bot"][side]["universe"].update(
            approved_coins=None, ignored_coins=str(ignored)
        )
    fixed = {}
    expected = {side: [True, False, True] for side in ("long", "short")}
    if policy in {"all", "file", "list"}:
        fixed["bot.global.universe.approved_coins"] = {
            "all": "all",
            "file": str(approved),
            "list": ["COIN00", "COIN02"],
        }[policy]
    elif policy.startswith("unchanged"):
        config["bot"]["global"]["universe"]["approved_coins"] = (
            "all" if policy == "unchanged_all" else str(approved)
        )
        fixed["bot.long.universe.ignored_coins"] = []
        if policy == "unchanged_all":
            expected["long"] = [True, True, True]
    elif policy == "side":
        config["bot"]["short"]["universe"]["approved_coins"] = ["COIN00"]
        fixed["bot.long.universe.approved_coins"] = str(replacement)
        expected = {"long": [False, False, True], "short": [True, False, False]}
    else:
        config["bot"]["global"]["universe"]["approved_coins"] = "all"
        for side in ("long", "short"):
            config["bot"][side]["universe"]["ignored_coins"] = str(missing)
            fixed[f"bot.{side}.universe.ignored_coins"] = str(replacement)
        expected = {side: [True, True, False] for side in ("long", "short")}
    config["bot"]["long"]["risk"]["max_positions"] = 1
    config["optimize"].update(
        bounds={"long": {"risk": {"max_positions": [1, 3]}}},
        limits=[],
        scoring=[{"metric": "adg_strategy_eq", "goal": "max"}],
        fixed_runtime_overrides=fixed,
    )
    config["backtest"].update(
        suite_enabled=False, scenarios=[], base_dir=str(tmp_path / "backtests")
    )
    source = tmp_path / "input.json"
    source.write_text(json.dumps(config))
    source_bytes = source.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        optimize.sys, "argv", ["passivbot optimize", str(source), "--suite", "n"]
    )
    monkeypatch.setattr(
        optimize.multiprocessing,
        "Manager",
        lambda: SimpleNamespace(dict=dict, shutdown=lambda: None),
    )
    acquired = []

    async def acquire(current, *args, **kwargs):
        assert current["_raw"]["optimize"]["bounds"] == config["optimize"]["bounds"]
        assert current["_raw_effective"]["optimize"]["limits"] == []
        for field in ("approved_coins", "ignored_coins"):
            values, _ = resolve_universe_sources(current, field)
            for side, value in values.items():
                if value == ["all"]:
                    values[side] = coins
            materialize_side_universe_sources(current, field, values)
        ignored_values = require_side_universe_sources(current, "ignored_coins")
        approved_values = require_side_universe_sources(current, "approved_coins")
        materialize_side_universe_sources(
            current,
            "approved_coins",
            {
                side: [
                    coin
                    for coin in approved_values[side]
                    if coin not in ignored_values[side]
                ]
                for side in ("long", "short")
            },
        )
        acquired.append(universe_source_view(current))

    async def prepare(data_config, *args, **kwargs):
        assert data_config["bot"]["long"]["risk"]["max_positions"] == 3
        for field in ("approved_coins", "ignored_coins"):
            assert all(
                type(value).__name__ == "ResolvedCoinList"
                for value in require_side_universe_sources(data_config, field).values()
            )
        # Source mutation after data preparation must never affect candidate replay.
        for path in (approved, ignored, replacement):
            path.unlink()
        return coins, candles, deepcopy(markets), "", "", btc, timestamps

    observed = []

    def backend(**kwargs):
        evaluator = kwargs["evaluator"]
        current = evaluator.config
        assert PREPARED_DATASET_KEY in current
        assert current["bot"]["long"]["risk"]["max_positions"] == 1
        assert current["_raw"]["optimize"]["bounds"] == config["optimize"]["bounds"]
        vector = optimize.config_to_individual(
            current, evaluator.bounds, optimization_shape=evaluator.optimization_shape
        )
        candidate = optimize.individual_to_config(
            vector,
            optimize.optimizer_overrides,
            [],
            current,
            key_paths=evaluator.key_paths,
        )
        native = build_backtest_payload(
            candles, evaluator.msss["binance"], candidate, "binance", btc, timestamps
        )
        actual = {
            side: [row[side]["entry_eligible"] for row in native.bot_params_list]
            for side in ("long", "short")
        }
        assert actual == expected
        execute_backtest(native, candidate)
        with NativeDatasetRegistry(
            evaluator, standalone_candle_coins={"binance": tuple(coins)}
        ) as registry:
            assert registry.planner.prepare("fixed", vector).requests
            other = tmp_path / "same-members-other-source.json"
            other.write_text(json.dumps(coins))
            current["bot"]["global"]["universe"]["approved_coins"] = str(other)
            current["optimize"]["fixed_runtime_overrides"][
                "bot.global.universe.approved_coins"
            ] = str(other)
            materialize_side_universe_sources(
                current,
                "approved_coins",
                require_side_universe_sources(candidate, "approved_coins"),
                sources=raw_side_universe_sources(current, "approved_coins"),
            )
            with pytest.raises(ValueError, match="dataset-owned execution inputs"):
                registry.planner.prepare("new-source", vector)
        observed.append(actual)
        return {"pool": None, "pool_terminated": False}

    monkeypatch.setattr(optimize, "format_approved_ignored_coins", acquire)
    monkeypatch.setattr(optimize, "prepare_hlcvs_mss", prepare)
    monkeypatch.setattr(optimize, "get_backend_runner", lambda _: backend)
    with pytest.raises(SystemExit) as exited:
        await optimize.main()
    assert exited.value.code == 0
    assert len(acquired) == len(observed) == 1
    assert source.read_bytes() == source_bytes


def test_dataset_candidate_rejects_unprepared_fixed_source_change(tmp_path):
    from test_shared_approved_native import prepared
    from optimization.warmup import build_optimizer_vector_config

    config, *_ = prepared("all", tmp_path)
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.global.universe.approved_coins": ["COIN00"]
    }
    before = deepcopy(config)
    with pytest.raises(ValueError, match="prepare.*membership"):
        build_optimizer_vector_config([], config, key_paths=[])
    assert universe_source_view(config) == universe_source_view(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("bot", [None, {}, {"long": {}}, {"short": {}}])
@pytest.mark.parametrize(
    "selector",
    [
        "bot.global.universe.approved_coins",
        "bot.long.universe.approved_coins",
        "bot.short.universe.ignored_coins",
    ],
)
@pytest.mark.parametrize("activatable", [False, True])
async def test_source_staging_preserves_original_omission_and_sparse_authority(
    tmp_path, monkeypatch, bot, selector, activatable
):
    import optimize

    original_prepare = optimize.prepare_config
    bounds = (
        {"long": {"risk": {"total_wallet_exposure_limit": [0, 2]}}}
        if activatable
        else {}
    )
    raw = {
        "config_version": "v9.0.0",
        "optimize": {
            "bounds": bounds,
            "limits": [],
            "fixed_runtime_overrides": {selector: ["COIN00"]},
        },
    }
    if bot is not None:
        raw["bot"] = deepcopy(bot)
    expected = original_prepare(raw, verbose=False)
    source = tmp_path / "sparse.json"
    source.write_text(json.dumps(raw))
    before = source.read_bytes()

    class Prepared(BaseException):
        pass

    calls = []

    def capture(current, **kwargs):
        result = original_prepare(current, **kwargs)
        calls.append(bool(kwargs.get("live_only")))
        if kwargs.get("live_only"):
            return result
        assert current["optimize"]["bounds"] == bounds
        assert current["optimize"]["limits"] == []
        assert result["_raw"] == raw
        assert result["_raw_effective"]["optimize"]["bounds"] == bounds
        for side in ("long", "short"):
            assert result["bot"][side]["risk"] == expected["bot"][side]["risk"]
            assert (
                result["optimize"]["bounds"][side]["risk"]
                == expected["optimize"]["bounds"][side]["risk"]
            )
        raise Prepared

    monkeypatch.setattr(optimize, "prepare_config", capture)
    monkeypatch.setattr(optimize.sys, "argv", ["passivbot optimize", str(source)])
    with pytest.raises(Prepared):
        await optimize.main()
    assert calls == [True, False]
    assert source.read_bytes() == before
