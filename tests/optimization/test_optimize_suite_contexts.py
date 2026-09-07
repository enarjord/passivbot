import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import optimize_suite
from config.schema import get_template_config
from suite_runner import ExchangeDataset


class _NoSharedArrayManager:
    def create_from(self, _array):
        raise AssertionError("test dataset should use lazy master specs")


class _FakeAttachment:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _stub_market_identity_validation(monkeypatch):
    async def fake_reject_cross_exchange_market_identifier_collisions(
        _identifiers, _exchanges, **_kwargs
    ):
        return None

    monkeypatch.setattr(
        optimize_suite,
        "reject_cross_exchange_market_identifier_collisions",
        fake_reject_cross_exchange_market_identifier_collisions,
    )


def _make_lazy_dataset(
    *,
    exchange="combined",
    coins=("HYPE",),
    coin_exchange=None,
    available_exchanges=None,
):
    timestamps = np.arange(1441, dtype=np.int64) * 60_000 + 1704067200000
    coins = list(coins)
    coin_exchange = coin_exchange or {coin: exchange for coin in coins}
    return ExchangeDataset(
        exchange=exchange,
        coins=coins,
        coin_index={coin: idx for idx, coin in enumerate(coins)},
        coin_exchange=coin_exchange,
        available_exchanges=available_exchanges or [exchange],
        hlcvs=np.ones((len(timestamps), len(coins), 4), dtype=np.float64),
        mss={
            **{
                coin: {
                    "exchange": coin_exchange.get(coin, exchange),
                    "first_valid_index": 0,
                    "last_valid_index": len(timestamps) - 1,
                }
                for coin in coins
            },
            "__meta__": {"data_interval_minutes": 1},
        },
        btc_usd_prices=np.ones(len(timestamps), dtype=np.float64),
        timestamps=timestamps,
        cache_dir="",
        hlcvs_spec=object(),
        btc_spec=object(),
    )


@pytest.mark.asyncio
async def test_prepare_suite_contexts_keeps_directional_scenarios_with_default_short_disabled(
    monkeypatch,
):
    _stub_market_identity_validation(monkeypatch)
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "base"},
        {"label": "long_only", "overrides": {"bot.short.total_wallet_exposure_limit": 0}},
        {"label": "short_only", "overrides": {"bot.long.total_wallet_exposure_limit": 0}},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    # Schema defaults keep shorts disabled. Optimizer candidates may enable
    # shorts later, so context preparation must not dedupe base vs long_only.
    config["bot"]["short"]["total_wallet_exposure_limit"] = 0.0

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(
        _config, _exchanges, verbose=False, **_kwargs
    ):
        return None

    captured = {}

    async def fake_prepare_master_datasets(*_args, **kwargs):
        captured["allow_internal_nan_gaps"] = kwargs["allow_internal_nan_gaps"]
        return {
            "combined": _make_lazy_dataset(
                coins=("HYPE",),
                coin_exchange={"HYPE": "binance"},
                available_exchanges=["binance", "bybit"],
            )
        }

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _reducer_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
        allow_internal_nan_gaps=True,
    )

    assert [ctx.label for ctx in contexts] == ["base", "long_only", "short_only"]
    assert captured["allow_internal_nan_gaps"] is True


def test_suite_evaluator_close_releases_context_and_master_attachments():
    from optimize import SuiteEvaluator

    context_hlcvs = _FakeAttachment()
    context_btc = _FakeAttachment()
    master_hlcvs = _FakeAttachment()
    master_btc = _FakeAttachment()

    evaluator = object.__new__(SuiteEvaluator)
    evaluator.contexts = [
        SimpleNamespace(
            attachments={
                "hlcvs": {"binance": context_hlcvs},
                "btc": {"binance": context_btc},
            }
        )
    ]
    evaluator._master_attachments = {
        "hlcvs": {"master-hlcvs": master_hlcvs},
        "btc": {"master-btc": master_btc},
    }
    evaluator._master_arrays = {
        "hlcvs": {"master-hlcvs": np.empty((1,))},
        "btc": {"master-btc": np.empty((1,))},
    }

    evaluator.close()
    evaluator.close()

    assert context_hlcvs.closed == 1
    assert context_btc.closed == 1
    assert master_hlcvs.closed == 1
    assert master_btc.closed == 1
    assert evaluator.contexts[0].attachments == {"hlcvs": {}, "btc": {}}
    assert evaluator._master_attachments == {"hlcvs": {}, "btc": {}}
    assert evaluator._master_arrays == {"hlcvs": {}, "btc": {}}


def test_suite_evaluator_pickle_strips_attached_and_cached_arrays():
    from optimize import SuiteEvaluator

    large = np.ones((1024, 1024), dtype=np.float64)
    context = optimize_suite.ScenarioEvalContext(
        label="base",
        config={},
        exchanges=["bybit"],
        hlcvs_specs={},
        btc_usd_specs={},
        msss={"bybit": {}},
        timestamps={"bybit": None},
        shared_hlcvs_np={"bybit": large},
        shared_btc_np={"bybit": large[:, 0]},
        attachments={"hlcvs": {"bybit": _FakeAttachment()}, "btc": {}},
        coin_indices={"bybit": [0]},
        overrides={},
    )
    evaluator = object.__new__(SuiteEvaluator)
    evaluator.base = None
    evaluator.contexts = [context]
    evaluator.reducer_cfg = {"default": "mean"}
    evaluator._master_attachments = {
        "hlcvs": {"master": _FakeAttachment()},
        "btc": {},
    }
    evaluator._master_arrays = {"hlcvs": {"master": large}, "btc": {}}

    payload = pickle.dumps(evaluator)
    restored = pickle.loads(payload)

    assert len(payload) < 100_000
    assert context.shared_hlcvs_np["bybit"] is large
    assert restored.contexts[0].shared_hlcvs_np == {}
    assert restored.contexts[0].shared_btc_np == {}
    assert restored.contexts[0].attachments == {"hlcvs": {}, "btc": {}}
    assert restored._master_arrays == {"hlcvs": {}, "btc": {}}
    assert restored._master_attachments == {"hlcvs": {}, "btc": {}}


@pytest.mark.asyncio
async def test_prepare_suite_contexts_master_universe_keeps_base_and_scenario_coins(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "explicit", "coins": ["DOGE"]},
        {"label": "default"},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    captured = {}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(
        _config, _exchanges, verbose=False, **_kwargs
    ):
        return None

    async def fake_prepare_master_datasets(base_config, exchanges, *_args, **_kwargs):
        captured["approved"] = list(base_config["live"]["approved_coins"]["long"])
        captured["exchanges"] = list(exchanges)
        return {
            "combined": _make_lazy_dataset(
                coins=("DOGE", "HYPE"),
                coin_exchange={"DOGE": "binance", "HYPE": "binance"},
                available_exchanges=["binance"],
            )
        }

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _reducer_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert captured["approved"] == ["DOGE", "HYPE"]
    assert captured["exchanges"] == ["binance"]
    assert [ctx.label for ctx in contexts] == ["explicit", "default"]


@pytest.mark.asyncio
async def test_prepare_suite_contexts_expands_scenario_required_exchanges(monkeypatch):
    _stub_market_identity_validation(monkeypatch)
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "bybit_only", "exchanges": ["bybit"], "coins": ["HYPE"]},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    loaded_exchanges = []
    captured = {}

    async def fake_load_markets(exchange, verbose=False):
        loaded_exchanges.append(exchange)
        return {}

    async def fake_format_approved_ignored_coins(
        _config, exchanges, verbose=False, **_kwargs
    ):
        captured["formatted_exchanges"] = list(exchanges)
        return None

    async def fake_prepare_master_datasets(_base_config, exchanges, *_args, **kwargs):
        captured["dataset_exchanges"] = list(exchanges)
        captured["needed"] = sorted(kwargs["needed_individual_exchanges"])
        return {"bybit": _make_lazy_dataset(exchange="bybit", coins=("HYPE",))}

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _reducer_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert loaded_exchanges == ["binance", "bybit"]
    assert captured["formatted_exchanges"] == ["binance", "bybit"]
    assert captured["dataset_exchanges"] == ["binance", "bybit"]
    assert captured["needed"] == ["bybit"]
    assert contexts[0].exchanges == ["bybit"]


@pytest.mark.asyncio
async def test_prepare_suite_contexts_keeps_explicit_exchange_out_of_combined_dataset(
    monkeypatch,
):
    _stub_market_identity_validation(monkeypatch)
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "bybit_only", "exchanges": ["bybit"], "coins": ["HYPE"]},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(
        _config, _exchanges, verbose=False, **_kwargs
    ):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        return {
            "combined": _make_lazy_dataset(
                coins=("HYPE",),
                coin_exchange={"HYPE": "binance"},
                available_exchanges=["binance", "bybit"],
            ),
            "bybit": _make_lazy_dataset(
                exchange="bybit",
                coins=("HYPE",),
                coin_exchange={"HYPE": "bybit"},
                available_exchanges=["bybit"],
            ),
        }

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(
        optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets
    )

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _reducer_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert contexts[0].exchanges == ["bybit"]
    assert contexts[0].msss["bybit"]["HYPE"]["exchange"] == "bybit"


@pytest.mark.asyncio
async def test_prepare_suite_contexts_rejects_unavailable_scenario_exchange(monkeypatch):
    _stub_market_identity_validation(monkeypatch)
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "bybit_only", "exchanges": ["bybit"], "coins": ["HYPE"]},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(
        _config, _exchanges, verbose=False, **_kwargs
    ):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        return {"binance": _make_lazy_dataset(exchange="binance", coins=("HYPE",))}

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    with pytest.raises(ValueError, match="requests unavailable exchange"):
        await optimize_suite.prepare_suite_contexts(
            config,
            suite_cfg,
            shared_array_manager=_NoSharedArrayManager(),
        )


@pytest.mark.asyncio
async def test_prepare_suite_contexts_rejects_scenario_with_no_usable_coins(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [{"label": "missing_coin", "coins": ["MISSING"]}]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(
        _config, _exchanges, verbose=False, **_kwargs
    ):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        return {"combined": _make_lazy_dataset(coins=("HYPE",), available_exchanges=["binance"])}

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    with pytest.raises(ValueError, match="missing_coin could not be prepared"):
        await optimize_suite.prepare_suite_contexts(
            config,
            suite_cfg,
            shared_array_manager=_NoSharedArrayManager(),
        )


@pytest.mark.asyncio
async def test_prepare_suite_contexts_rejects_asymmetric_side_coin_lists(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [{"label": "base"}]
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": ["ETH"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(
        _config, _exchanges, verbose=False, **_kwargs
    ):
        return None

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    with pytest.raises(ValueError, match="asymmetric live.approved_coins"):
        await optimize_suite.prepare_suite_contexts(
            config,
            suite_cfg,
            shared_array_manager=_NoSharedArrayManager(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["deap", "pymoo", "gpu"])
async def test_scenario_file_override_is_frozen_in_candidates_and_resume_contract(
    tmp_path, monkeypatch, backend
):
    import json
    from copy import deepcopy
    from optimize import (
        SuiteEvaluator,
        _materialize_suite_run_contract,
        _materialize_resolved_suite_dates,
        _resume_config_mismatches,
        build_backtest_payload,
        execute_backtest,
    )
    from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract

    _stub_market_identity_validation(monkeypatch)
    config = get_template_config()
    config["optimize"]["backend"] = backend
    config["backtest"].update(
        start_date="2024-01-01",
        end_date="2024-01-02",
        exchanges=["binance"],
        suite_enabled=True,
    )
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    path = tmp_path / "scenario-coin.json"
    config["backtest"]["scenarios"] = [
        {
            "label": "file-policy",
            "overrides": {
                "coin_overrides": {"HYPE": {"override_config_path": str(path)}}
            },
        }
    ]
    dataset = _make_lazy_dataset(
        coin_exchange={"HYPE": "binance"}, available_exchanges=["binance"]
    )
    dataset.mss["HYPE"].update(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=1.0,
        c_mult=1.0,
        maker=0.0,
        taker=0.0,
    )

    async def noop(*args, **kwargs):
        return None

    async def datasets(*args, **kwargs):
        return {"combined": dataset}

    monkeypatch.setattr(optimize_suite, "load_markets", noop)
    monkeypatch.setattr(optimize_suite, "format_approved_ignored_coins", noop)
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", datasets)

    async def prepare(value):
        path.write_text(
            json.dumps({"bot": {"long": {"risk": {"entry_cooldown_minutes": value}}}})
        )
        run = deepcopy(config)
        suite = optimize_suite.extract_suite_config(run, suite_override=None)
        _materialize_suite_run_contract(run, suite)
        contexts, _ = await optimize_suite.prepare_suite_contexts(
            run, suite, shared_array_manager=_NoSharedArrayManager()
        )
        _materialize_resolved_suite_dates(run, contexts)
        return run, contexts[0]

    previous, old_ctx = await prepare(37.0)
    old = {**deepcopy(previous), CONTRACT_KEY: build_evaluation_contract(previous)}
    path = tmp_path / "relocated-scenario-coin.json"
    config["backtest"]["scenarios"][0]["overrides"]["coin_overrides"]["HYPE"][
        "override_config_path"
    ] = str(path)
    unchanged, _ = await prepare(37.0)
    assert _resume_config_mismatches(old, unchanged) == []
    current, _ = await prepare(71.0)
    assert any(
        "backtest.scenarios" in x for x in _resume_config_mismatches(old, current)
    )
    assert "override_config_path" not in json.dumps(old["backtest"]["scenarios"])
    # A prepared run uses its frozen policy even after the source file changes/disappears.
    path.unlink()
    evaluator = object.__new__(SuiteEvaluator)
    candidate = evaluator.build_scenario_candidate_config(previous, old_ctx)
    assert (
        candidate["coin_overrides"]["HYPE"]["bot"]["long"]["risk"][
            "entry_cooldown_minutes"
        ]
        == 37.0
    )
    exchange = old_ctx.exchanges[0]
    payload = build_backtest_payload(
        dataset.hlcvs,
        old_ctx.msss[exchange],
        candidate,
        exchange,
        dataset.btc_usd_prices,
        dataset.timestamps,
        metrics_only=True,
    )
    assert payload.bot_params_list[0]["long"]["risk_entry_cooldown_minutes"] == 37.0
    execute_backtest(payload, candidate)
