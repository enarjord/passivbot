from copy import deepcopy
import json
import pickle
import signal

import msgpack
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
async def test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(monkeypatch, tmp_path, suite, interrupted, automatic):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    import backtest
    import optimize
    import optimize_suite
    from optimization.evaluation_contract import CONTRACT_KEY
    from suite_runner import ExchangeDataset
    from tools.gpu_parity import build_parser, fixture_inputs

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "512",
    ]))
    config["optimize"].update(backend="gpu_native", population_size=4, iters=8, seed=12)
    config["optimize"]["scoring"] = [dict(metric="adg_strategy_eq", goal="max"),
                                      dict(metric="drawdown_worst_strategy_eq", goal="min")]
    config["optimize"]["limits"] = [dict(metric="backtest_completion_ratio", penalize_if="less_than", value=0.99)]
    config["optimize"]["gpu"].update(batch_size=None if automatic else 2, checkpoint_interval_seconds=0)
    config["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        for key in ("n_positions", "total_wallet_exposure_limit"):
            value = config["bot"][side]["risk"][key]
            config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
        config["optimize"]["bounds"][f"{side}_entry_initial_qty_pct"] = [0.01, 0.05]
    config["backtest"].update(suite_enabled=suite, scenarios=[{"label": "base"},
        {"label": "window", "coins": ["COIN00", "COIN02"]}] if suite else [])
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    seeds_path = tmp_path / "seeds"
    seeds_path.mkdir()
    for index, value in enumerate((0.012, 0.03)):
        seed_config = deepcopy(config)
        seed_config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = value
        (seeds_path / f"{index}.json").write_text(json.dumps(seed_config))
    def forbidden(*_args, **_kwargs):
        pytest.fail("native optimizer CLI must never call a CPU simulation or create a CPU worker pool")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    monkeypatch.setattr(optimize.Evaluator, "evaluate", forbidden)
    monkeypatch.setattr(optimize.SuiteEvaluator, "evaluate", forbidden)
    monkeypatch.setattr(optimize.multiprocessing, "Pool", forbidden)
    monkeypatch.setattr(optimize.multiprocessing, "Manager", forbidden)
    async def offline(*_args, **_kwargs):
        return None
    async def prepared(*_args, **_kwargs):
        return (list(config["backtest"]["coins"]["binance"]), candles, deepcopy(markets),
                "", "", btc, timestamps)
    async def master(*_args, **kwargs):
        manager = kwargs["shared_array_manager"]
        source = manager.create_from(candles)[0]
        prices = manager.create_from(btc)[0]
        coins = config["backtest"]["coins"]["binance"]
        return {"combined": ExchangeDataset(
            exchange="combined", coins=coins, coin_index={coin: i for i, coin in enumerate(coins)},
            coin_exchange={coin: "binance" for coin in coins}, available_exchanges=["binance"],
            hlcvs=candles, mss=deepcopy(markets), btc_usd_prices=btc, timestamps=timestamps, cache_dir="",
            hlcvs_spec=source, btc_spec=prices,
        )}
    monkeypatch.setattr(optimize, "format_approved_ignored_coins", offline)
    monkeypatch.setattr(optimize, "prepare_hlcvs_mss", prepared)
    monkeypatch.setattr(optimize_suite, "load_markets", offline)
    monkeypatch.setattr(optimize_suite, "format_approved_ignored_coins", offline)
    monkeypatch.setattr(optimize_suite, "reject_cross_exchange_market_identifier_collisions", offline)
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", master)
    emitted = False
    record = optimize.ResultRecorder.record
    def record_and_interrupt(self, row):
        nonlocal emitted
        record(self, row)
        if interrupted and not emitted:
            emitted = True
            signal.raise_signal(signal.SIGINT)
    monkeypatch.setattr(optimize.ResultRecorder, "record", record_and_interrupt)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(optimize.sys, "argv", ["passivbot optimize", str(config_path), "--offline", "y",
                                                "--start", str(seeds_path)])
    with pytest.raises(SystemExit) as first:
        await optimize.main()
    assert first.value.code == (130 if interrupted else 0)
    results = list((tmp_path / "optimize_results").iterdir())
    assert len(results) == 1
    directory = results[0]
    def records():
        with (directory / "all_results.bin").open("rb") as source:
            return list(msgpack.Unpacker(source, raw=False, strict_map_key=False))
    assert 0 < len(records()) <= 8
    if not interrupted:
        assert len(records()) == 8
    with (directory / "checkpoint.pkl").open("rb") as source:
        state = pickle.load(source)
    assert state["completed"] == len(records())
    assert state["phase"] == ("seeds" if interrupted else "idle")
    assert state[CONTRACT_KEY]["execution"]["engine"] == "cuda_native"
    assert list(directory.rglob("*.json"))  # Pareto members were written promptly.
    monkeypatch.setattr(optimize.sys, "argv", ["passivbot optimize", str(config_path), "--offline", "y",
                                                "--resume", str(directory), "-i", "12"])
    with pytest.raises(SystemExit) as resumed:
        await optimize.main()
    assert resumed.value.code == 0
    assert len(records()) == 12
