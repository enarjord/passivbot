"""Revised HSL through public backtest and CPU optimizer entry points."""
from copy import deepcopy
import json
import pickle
import socket
import os
import sys

import numpy as np
import pytest

from backtest import build_backtest_payload, execute_backtest
from optimize import Evaluator, SuiteEvaluator
from optimize_suite import ScenarioEvalContext
from shared_arrays import SharedArrayManager
from test_hsl_revised_backtest_config import inputs
from test_hsl_revised_optimizer_contract import fixed_side_bounds, threshold_vector


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("offline HSL test attempted network access")
    original_connect = socket.socket.connect
    def connect(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        return denied(sock, address)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def runtime_inputs(mode):
    cfg, markets, candles = inputs(mode)
    cfg["live"].update(warmup_ratio=1., max_warmup_minutes=1, minimum_coin_age_days=0.)
    cfg["backtest"].update(start_date="2024-01-01", end_date="2024-01-01T02:40:00")
    timestamps = 1704067200000 + np.arange(len(candles), dtype=np.int64) * 60000
    return cfg, markets, candles, timestamps


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_public_payload_runs_red_panic_and_cooldown(mode):
    cfg, markets, candles, timestamps = runtime_inputs(mode)
    original = deepcopy(cfg)
    prepared = build_backtest_payload(candles, markets, cfg, "binance",
                                     np.full(len(candles), 50000.), timestamps)
    fills, equities, analysis = execute_backtest(prepared, cfg)
    report = prepared.hard_stop_plot_data["revised"]
    assert report["mode"] == mode
    assert analysis["hard_stop_triggers"] > 0
    assert report["summary"]["panic_close_fills"] > 0
    assert report["detailed"] is False
    assert report["samples"] == []
    assert report["summary"]["restarts"] > 0
    assert {"red", "flat", "restart"} <= {row["kind"] for row in report["events"]}
    assert any("panic" in str(fill[13]) for fill in fills)
    assert equities.shape[1] == 4
    assert cfg == original


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_real_cpu_evaluator_fitness_changes_with_revised_threshold(mode):
    cfg, markets, candles, timestamps = runtime_inputs(mode)
    fixed_side_bounds(cfg)
    target = cfg["optimize"]["bounds"] if mode == "unified" else cfg["optimize"]["bounds"]["long"]
    target["hsl"] = {"red_threshold": [.01, .9]}
    cfg["optimize"]["scoring"] = ["hard_stop_triggers"]
    cfg["optimize"]["limits"] = []
    manager = SharedArrayManager()
    try:
        candle_spec, _ = manager.create_from(candles)
        btc_spec, _ = manager.create_from(np.full(len(candles), 50000.))
        evaluator = Evaluator({"binance": candle_spec}, {"binance": btc_spec},
                              {"binance": markets}, cfg, timestamps={"binance": timestamps},
                              shared_array_manager=manager)
        evaluator.use_duplicate_guard = False
        results = [evaluator.evaluate(threshold_vector(evaluator.optimization_shape, value), [])
                   for value in [.01, .9]]
        assert results[0]["fitness"][0] > 0
        assert results[1]["fitness"][0] == 0
        assert all(result["constraint_violation"] == 0 for result in results)
        assert all(result["metrics"] is not None for result in results)
        restored = pickle.loads(pickle.dumps(evaluator))
        assert restored.evaluate(threshold_vector(restored.optimization_shape, .01), [])["fitness"] == results[0]["fitness"]
        # Exercise effective scenario overrides through the complete suite evaluator.
        path = "bot.hsl.red_threshold" if mode == "unified" else "bot.long.hsl.red_threshold"
        contexts = [ScenarioEvalContext(
            label=label, config=deepcopy(cfg), exchanges=["binance"],
            hlcvs_specs={"binance": candle_spec}, btc_usd_specs={"binance": btc_spec},
            msss={"binance": markets}, timestamps={"binance": timestamps},
            shared_hlcvs_np={"binance": candles},
            shared_btc_np={"binance": np.full(len(candles), 50000.)},
            attachments={"hlcvs": {}, "btc": {}}, coin_indices={"binance": None},
            overrides={path: threshold}) for label, threshold in [("low", .01), ("high", .9)]]
        suite = SuiteEvaluator(evaluator, contexts, {"default": "mean"})
        combined = suite.evaluate(threshold_vector(evaluator.optimization_shape, .01), [])
        assert combined["constraint_violation"] == 0
        assert combined["fitness"][0] == pytest.approx(results[0]["fitness"][0] / 2)
        suite.close()
        del restored
    finally:
        manager.cleanup()


def offline_cli_config(tmp_path, monkeypatch, mode):
    from config_utils import strip_config_metadata
    from ohlcv_catalog import OhlcvCatalog
    from ohlcv_store import OhlcvStore
    from test_simulation_offline import metadata, START, SYMBOL
    from utils import ts_to_date

    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg, _, candles, timestamps = runtime_inputs(mode)
    cfg["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
    cfg["backtest"].update(offline=True, exchanges=["binance"],
                           start_date=ts_to_date(START), end_date="2024-01-02",
                           compress_cache=False, gap_tolerance_ohlcvs_minutes=0,
                           base_dir=str(tmp_path / "results"), suite_enabled=False)
    catalog = OhlcvCatalog(tmp_path / "caches/ohlcvs/catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches/ohlcvs", catalog)
    # CLI dates are day-granular. Supply one complete day plus the warmup minute.
    day_timestamps = START + np.arange(-1, 1441, dtype=np.int64) * 60000
    day_candles = np.concatenate([candles[:1, 0], candles[:, 0],
                                 np.repeat(candles[-1:, 0], 1441 - len(candles), axis=0)])
    store.write_rows("binance", "1m", SYMBOL, day_timestamps, day_candles.astype(np.float32))
    return strip_config_metadata(cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("detailed", [False, True])
async def test_backtest_cli_uses_only_offline_exchange_equivalent_data(tmp_path, monkeypatch, mode, detailed):
    from backtest import main
    cfg = offline_cli_config(tmp_path, monkeypatch, mode)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["backtest", str(config_path), "-dp", "all"] +
                        (["--backtest.hsl_detailed_report", "true"] if detailed else []))
    await main()
    artifact, = (tmp_path / "results").rglob("hsl_report.json")
    report = json.loads(artifact.read_text())
    assert report["engine"] == "revised" and report["mode"] == mode
    assert report["summary"]["panic_close_fills"] > 0
    assert report["detailed"] is detailed
    assert bool(report["samples"]) is detailed
    assert report["events"]
    assert (artifact.parent / "fills.csv").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["deap", "pymoo"])
async def test_cpu_optimizer_cli_and_resume_are_offline(tmp_path, monkeypatch, backend):
    from optimize import main
    import msgpack

    cfg = offline_cli_config(tmp_path, monkeypatch, "unified")
    fixed_side_bounds(cfg)
    cfg["optimize"].update(backend=backend, iters=2, n_cpus=1, population_size=2,
                           scoring=[{"metric": "hard_stop_triggers", "goal": "min"}], limits=[], seed=42,
                           compress_results_file=False, write_all_results=True)
    cfg["optimize"]["bounds"]["hsl"] = {"red_threshold": [.01, .1]}
    # Spawned workers and the manager also deny IP networking; local Unix IPC
    # remains available for the real multiprocessing backend.
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\n"
        "_connect = socket.socket.connect\n"
        "def denied(*a, **k): raise RuntimeError('offline optimizer attempted network')\n"
        "def connect(self, address):\n"
        "    if self.family != socket.AF_UNIX: return denied()\n"
        "    return _connect(self, address)\n"
        "socket.socket.connect = connect\n"
        "socket.getaddrinfo = denied\n")
    monkeypatch.setenv("PYTHONPATH", str(guard) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["optimize", str(config_path), "--suite", "n"])
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code == 0
    result_file, = (tmp_path / "optimize_results").rglob("all_results.bin")
    before = result_file.stat().st_size
    assert before > 0 and (result_file.parent / "checkpoint.pkl").exists()
    with result_file.open("rb") as handle:
        records = list(msgpack.Unpacker(handle, raw=False))
    assert records
    # Resume uses the real checkpoint and saved evaluation contract.
    cfg["optimize"]["iters"] = 4
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["optimize", str(config_path), "--suite", "n",
                                      "--resume", str(result_file.parent)])
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code == 0
    assert result_file.stat().st_size > before
