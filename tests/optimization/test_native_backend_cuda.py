from copy import deepcopy
import json
import pickle
import signal
from pathlib import Path

import msgpack
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
async def test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(monkeypatch, tmp_path, suite, interrupted, automatic,
                                                                    screening=False, anchors=False, coupled=False, scaled_hsl=False):
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
    if screening:
        config["optimize"]["gpu"]["screening"] = dict(scenarios=["base"], min_survivors=1, survival_fraction=0.5)
    config["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        for key in ("n_positions", "total_wallet_exposure_limit"):
            value = config["bot"][side]["risk"][key]
            config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
        config["optimize"]["bounds"][f"{side}_entry_initial_qty_pct"] = [0.01, 0.05]
    if anchors:
        config["optimize"]["bounds"]["short_total_wallet_exposure_limit"] = [0, 1]
    if coupled:
        config["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
        config["optimize"]["bounds"]["long_ema_span_0"] = [2.0, 20.0]
        for side in ("long", "short"):
            config["bot"][side]["unstuck"].update(enabled=True, ema_gating_enabled=True)
        config["coin_overrides"] = {
            "COIN01": {"bot": {"long": {"unstuck": {"close_pct": 0.05}}}},
            "COIN02": {"bot": {"long": {"strategy": {"trailing_martingale": {
                "entry": {"ema_span_0": 7.5},
            }}}}},
        }
    if scaled_hsl:
        for side in ("long", "short"):
            config["bot"][side]["hsl"].update(
                enabled=True, scale_budget_with_excess_allowance=True,
            )
            config["bot"][side]["risk"]["we_excess_allowance_pct"] = 0.44
        config["coin_overrides"] = {
            "COIN00": {"bot": {"long": {
                "wallet_exposure_limit": 0.2, "risk": {"we_excess_allowance_pct": 0.1},
            }}},
        }
    config["backtest"].update(suite_enabled=suite, scenarios=[{"label": "base"},
        {"label": "window", "coins": ["COIN00", "COIN02"]}] if suite else [])
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    seeds_path = tmp_path / "seeds"
    seeds_path.mkdir()
    for index, value in enumerate((0.012, 0.03)):
        seed_config = deepcopy(config)
        seed_config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = value
        if anchors:
            seed_config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0 if index == 0 else 1
        if coupled:
            seed_config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] = 2 if index == 0 else 20
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
    first_record_persisted = False
    record = optimize.ResultRecorder.record
    def record_and_interrupt(self, row):
        nonlocal emitted, first_record_persisted
        record(self, row)
        if not first_record_persisted:
            # Read through independent file handles before shutdown, checkpoint
            # flushing or a completed cohort can hide deferred persistence.
            with open(self.results_file.name, "rb") as source:
                stored = list(msgpack.Unpacker(source, raw=False, strict_map_key=False))
            assert len(stored) == 1
            assert stored[0][CONTRACT_KEY]["execution"]["engine"] == "cuda_native"
            members = list(Path(self.store.pareto_dir).glob("*.json"))
            assert members
            assert json.loads(members[0].read_text())[CONTRACT_KEY]["execution"]["engine"] == "cuda_native"
            first_record_persisted = True
        # With screening, interrupt after the initial full generation so the
        # on-disk checkpoint exercises a partially completed screening stage.
        if interrupted and not emitted and not screening:
            emitted = True
            signal.raise_signal(signal.SIGINT)
    monkeypatch.setattr(optimize.ResultRecorder, "record", record_and_interrupt)
    if screening and interrupted:
        from optimization.backends.gpu_native_backend import _Search
        checkpoint = _Search.checkpoint
        def checkpoint_and_interrupt(self, **kwargs):
            nonlocal emitted
            checkpoint(self, **kwargs)
            if self.state["phase"] == "screening" and self.state["screened"] and not emitted:
                emitted = True
                signal.raise_signal(signal.SIGINT)
        monkeypatch.setattr(_Search, "checkpoint", checkpoint_and_interrupt)
    monkeypatch.chdir(tmp_path)
    command = ["passivbot optimize", str(config_path), "--offline", "y", "--start", str(seeds_path)]
    if anchors:
        command += ["--fine-tune-params", "long.strategy.entry.initial_qty_pct"]
    monkeypatch.setattr(optimize.sys, "argv", command)
    with pytest.raises(SystemExit) as first:
        await optimize.main()
    assert first.value.code == (130 if interrupted else 0)
    assert first_record_persisted
    results = list((tmp_path / "optimize_results").iterdir())
    assert len(results) == 1
    directory = results[0]
    def records():
        with (directory / "all_results.bin").open("rb") as source:
            return list(msgpack.Unpacker(source, raw=False, strict_map_key=False))
    assert 0 < len(records()) <= 8
    if not interrupted:
        assert len(records()) == (6 if screening else 8)
    with (directory / "checkpoint.pkl").open("rb") as source:
        state = pickle.load(source)
    assert state["completed"] == len(records())
    assert state["phase"] == (("screening" if screening else "seeds") if interrupted else "idle")
    assert state[CONTRACT_KEY]["execution"]["engine"] == "cuda_native"
    assert list(directory.rglob("*.json"))  # Pareto members were written promptly.
    if anchors:
        assert len(state["anchor_plan"]["anchors"]) == 2
        assert state["algorithm"].problem.n_var == 2
        for path in seeds_path.iterdir():
            path.unlink()
        seeds_path.rmdir()  # Resume must use checkpoint-owned anchors.
    monkeypatch.setattr(optimize.sys, "argv", ["passivbot optimize", str(config_path), "--offline", "y",
                                                "--resume", str(directory), "-i", "12"])
    with pytest.raises(SystemExit) as resumed:
        await optimize.main()
    assert resumed.value.code == 0
    assert len(records()) == (8 if screening else 12)
    if scaled_hsl:
        expanded = {}
        for row in records():
            expanded = optimize.deep_updated(expanded, row)
            assert all(expanded[CONTRACT_KEY]["bot"][side]["hsl"]["scale_budget_with_excess_allowance"]
                       for side in ("long", "short"))
            assert all(expanded["bot"][side]["hsl"]["scale_budget_with_excess_allowance"]
                       for side in ("long", "short"))
        assert state[CONTRACT_KEY]["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"]
        members = list((directory / "pareto").glob("*.json"))
        assert members
        for member in members:
            exported = json.loads(member.read_text())
            assert all(exported["bot"][side]["hsl"]["scale_budget_with_excess_allowance"]
                       for side in ("long", "short"))
    if screening:
        with (directory / "checkpoint.pkl").open("rb") as source:
            final = pickle.load(source)
        assert final["screened"] == 8 and len(final["algorithm"].pop) == 4
    if anchors:
        with (directory / "checkpoint.pkl").open("rb") as source:
            final = pickle.load(source)
        assert final["algorithm"].problem.n_var == 2
        assert final["anchor_plan"] == state["anchor_plan"]


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
async def test_native_screening_cli_cuda_preserves_full_records_and_stage_resume(monkeypatch, tmp_path,
                                                                                interrupted, automatic):
    await test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(
        monkeypatch, tmp_path, True, interrupted, automatic, screening=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suite,screening", [(False, False), (True, True)])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
async def test_native_anchor_cli_cuda_restores_side_variants_without_seed_files(monkeypatch, tmp_path,
                                                                             suite, screening, interrupted, automatic):
    await test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(
        monkeypatch, tmp_path, suite, interrupted, automatic, screening=screening, anchors=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suite,screening", [(False, False), (True, True)])
@pytest.mark.parametrize("interrupted", [False, True])
async def test_native_coupled_cli_cuda_searches_spans_with_coin_pins_and_resumes(monkeypatch, tmp_path,
                                                                            suite, screening, interrupted):
    await test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(
        monkeypatch, tmp_path, suite, interrupted, True, screening=screening, coupled=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
async def test_native_scaled_coin_hsl_cli_cuda_preserves_policy_and_resumes_without_cpu(
    monkeypatch, tmp_path, suite, interrupted,
):
    await test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(
        monkeypatch, tmp_path, suite, interrupted, True, scaled_hsl=True,
    )
