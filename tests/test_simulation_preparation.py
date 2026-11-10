from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

from config import prepare_config
from config.schema import get_template_config
from simulation_preparation import build_preparation_plan, prepare_optimizer_inputs


START = 1704067200000
SYMBOL = "BTC/USDT:USDT"


def source_config():
    config = get_template_config()
    config["bot"]["long"]["enabled"] = True
    config["bot"]["short"]["enabled"] = False
    config["bot"]["global"]["universe"]["approved_coins"] = ["BTC"]
    config["bot"]["global"]["universe"]["minimum_coin_age_days"] = 0
    # This short preparation fixture exercises the ordinary EMA warmup cap.
    config["bot"]["global"]["forager"]["score_weights"]["rms_directionality"] = 0.0
    config["bot"]["global"]["entry_cooldown"]["weights_minutes"] = {
        "adverse_directionality": 0.0,
        "exposure_ratio": 0.0,
    }
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = 1
    config["optimize"]["backend"] = "gpu"
    config["backtest"].update(start_date="2024-01-01", end_date="2024-01-02",
                              exchanges=["binance"], compress_cache=False)
    return config


def canonical():
    return prepare_config(source_config(), verbose=False)


def deny_network(monkeypatch):
    def denied(*a, **kw):
        pytest.fail("preparation unexpectedly attempted network access")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def tree_facts(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


def test_optimizer_plan_is_read_only_and_preserves_literal_source_policy(tmp_path, monkeypatch):
    deny_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "caches/binance/markets.json"
    p.parent.mkdir(parents=True)
    p.write_text('{}')
    os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    config = canonical()
    config["backtest"].update(suite_enabled=True, scenarios=[
        {"label": "first", "start_date": "2021-04", "end_date": "2023-01-01"},
        {"label": "subset", "coins": ["ETH", "BTC"], "start_date": "2025-01-01", "end_date": "2025-02-01"},
    ])
    before = tree_facts(tmp_path)
    raw_before = deepcopy(config["_raw"])
    torch_before = "torch" in sys.modules
    report = build_preparation_plan(config, for_optimize=True, target_cuda_free_bytes=8*2**30)
    assert report["coverage"] == "not_verified"
    assert [x["label"] for x in report["cases"]] == ["first", "subset"]
    assert report["cases"][0]["start_date_literal"] == "2021-04"
    assert report["cases"][1]["declared_identifiers"] == ["ETH", "BTC"]
    assert report["cases"][0]["warmup_minutes"] == 1
    assert report["cases"][0]["cuda_invariant_bytes_estimate"] > 0
    assert report["cases"][0]["target_cuda_45_percent_bytes"] == int(8*2**30*.45)
    assert report["metadata"][0]["mtime_ns"] == 1_000_000_000
    assert report["missing_metadata"]
    assert report["unresolved"]
    assert config["_raw"] == raw_before
    assert tree_facts(tmp_path) == before
    assert ("torch" in sys.modules) == torch_before


def test_plan_marks_dynamic_universe_unknown_and_applies_scenario_patches(tmp_path, monkeypatch):
    deny_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = source_config()
    config["bot"]["global"]["universe"]["approved_coins"] = "all"
    config["backtest"].update(suite_enabled=True, scenarios=[{
        "label": "source", "start_date": "2025-01-01", "end_date": "2025-02-01",
        "coin_sources": {"BTC": "bybit"},
        "overrides": {"backtest.end_date": "2025-03-01"},
    }])
    report = build_preparation_plan(prepare_config(config, verbose=False), for_optimize=True)
    case = report["cases"][0]
    assert case["declared_identifiers"] is None
    assert case["cuda_invariant_bytes_estimate"] is None
    assert case["end_date_literal"] == "2025-03-01"
    assert case["coin_sources"] == {"BTC": "bybit"}
    assert any("unknown" in value for value in report["unresolved"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_plan_rejects_invalid_target_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        build_preparation_plan(canonical(), target_cuda_free_bytes=budget)


def test_optimizer_plan_uses_owner_worst_case_warmup(tmp_path, monkeypatch):
    from optimization.warmup import compute_optimizer_backtest_warmup_minutes
    from simulation_preparation import optimizer_data_config
    monkeypatch.chdir(tmp_path)
    raw = source_config()
    raw["bot"]["portfolio"]["warmup"]["max_minutes"] = None
    config = prepare_config(raw, verbose=False)
    expected = compute_optimizer_backtest_warmup_minutes(optimizer_data_config(config))
    assert expected > 1
    assert build_preparation_plan(config, for_optimize=True)["cases"][0]["warmup_minutes"] == expected


@pytest.mark.parametrize("fixed", [
    {"backtest.reducer.default": "max"},
    {"backtest.reducer.default": "max", "backtest.suite_enabled": False},
])
def test_plan_uses_authored_suite_argument_not_fixed_runtime_data_view(tmp_path, monkeypatch, fixed):
    deny_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    raw = source_config()
    raw["backtest"].update(suite_enabled=True, scenarios=[{"label": "first"}, {"label": "second"}])
    raw["optimize"]["fixed_runtime_overrides"] = fixed
    result = build_preparation_plan(prepare_config(raw, verbose=False), for_optimize=True)
    assert result["suite"]["enabled"] is True
    assert result["suite"]["reducer"]["default"] == "mean"
    assert [case["label"] for case in result["cases"]] == ["first", "second"]
    assert not list(tmp_path.iterdir())


def write_metadata(root):
    from procedures import FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION
    cache = root / "caches"
    (cache / "binance").mkdir(parents=True)
    market = {"id": "BTCUSDT", "symbol": SYMBOL, "base": "BTC", "quote": "USDT", "settle": "USDT",
              "swap": True, "linear": True, "active": True, "type": "swap", "spot": False,
              "future": False, "contract": True, "maker": .0002, "taker": .0005,
              "contractSize": 1.0, "precision": {"amount": .001, "price": .1},
              "limits": {"amount": {"min": .001}, "cost": {"min": 5.0}}}
    files = {"binance/markets.json": {SYMBOL: market},
             "binance/first_timestamps.json": {"BTC": START-86400000},
             "first_ohlcv_timestamps_unified.json": {"BTC": START-86400000},
             "first_ohlcv_timestamps_unified_exchange_specific.json": {"BTC": {"binanceusdm": START-86400000}},
             "first_ohlcv_timestamps_unified_exchange_specific_symbols.json": {"BTC": {"binanceusdm": SYMBOL}}}
    for name, value in files.items():
        p = cache / name
        p.write_text(json.dumps(value))
        os.utime(p, (1, 1))
    (cache / "first_ohlcv_timestamps_unified.version").write_text(str(FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION))


@pytest.mark.asyncio
@pytest.mark.parametrize("fixed_overrides", [
    {}, {"backtest.reducer.default": "max"}, {"backtest.acquisition_mode": "rest"},
])
async def test_optimizer_download_prepares_exact_suite_and_reuses_cache_offline(tmp_path, monkeypatch, fixed_overrides):
    import backtest
    import optimize_suite
    from simulation_data import acquisition_mode
    from ohlcv_catalog import OhlcvCatalog
    from ohlcv_store import OhlcvStore
    from utils import ts_to_date

    deny_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    write_metadata(tmp_path)
    raw = source_config()
    raw["optimize"]["fixed_runtime_overrides"] = deepcopy(fixed_overrides)
    raw["backtest"].update(offline=True, start_date="2024-01-01", end_date="2024-01-03",
                          gap_tolerance_ohlcvs_minutes=0.0, suite_enabled=True, scenarios=[
                              {"label": "first", "end_date": "2024-01-02"},
                              {"label": "second", "start_date": "2024-01-02"},
                          ])
    config = prepare_config(raw, verbose=False)
    catalog = OhlcvCatalog(tmp_path / "caches/ohlcvs/catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches/ohlcvs", catalog)
    times = START + np.arange(-1, 2881, dtype=np.int64)*60000
    store.write_rows("binance", "1m", SYMBOL, times,
                     np.tile(np.array([101, 99, 100, 10], dtype=np.float32), (len(times), 1)))
    owner = optimize_suite.prepare_master_datasets
    observed_modes = []

    async def observe_owner_mode(*args, **kwargs):
        observed_modes.append(acquisition_mode())
        return await owner(*args, **kwargs)

    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", observe_owner_mode)
    report = await prepare_optimizer_inputs(config)
    assert report["consumer"] == "optimize"
    assert report["suite"]["reducer"]["default"] == "mean"
    expected_mode = fixed_overrides.get("backtest.acquisition_mode", "auto")
    assert report["acquisition_mode"] == observed_modes[0] == expected_mode
    assert config["backtest"]["acquisition_mode"] == "auto"
    assert [case["label"] for case in report["cases"]] == ["first", "second"]
    assert len(report["datasets"]) == 1
    assert report["datasets"][0]["ordered_coins"] == ["BTC"]
    assert report["datasets"][0]["manifest_sha256"]
    cache_dirs = [p.name for p in Path("caches/hlcvs_data").iterdir()]

    async def no_rebuild(*a, **kw):
        pytest.fail("optimizer download rebuilt already-prepared input")
    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", no_rebuild)
    monkeypatch.setattr(backtest, "prepare_hlcvs_combined", no_rebuild)
    again = await prepare_optimizer_inputs(config)
    assert report == again
    assert observed_modes == [expected_mode, expected_mode]
    assert acquisition_mode() == "auto"
    assert [p.name for p in Path("caches/hlcvs_data").iterdir()] == cache_dirs


@pytest.mark.asyncio
async def test_download_offline_policy_covers_metadata_preload(monkeypatch):
    import ohlcv_download
    from simulation_data import is_offline

    async def check_policy(*a, **kw):
        assert is_offline()
        raise RuntimeError("stop before acquisition")
    monkeypatch.setattr(ohlcv_download, "format_approved_ignored_coins", check_policy)
    with pytest.raises(RuntimeError, match="stop before acquisition"):
        await ohlcv_download.warm_ohlcv_caches({"backtest": {"offline": True, "exchanges": ["binance"]}})
    assert not is_offline()


def test_plan_cli_is_offline_and_does_not_materialize(tmp_path):
    raw = source_config()
    cfg = tmp_path / "input.json"
    cfg.write_text(json.dumps(raw))
    root = Path(__file__).resolve().parents[1]
    code = """
import socket, sys
import rust_utils
def denied(*a, **kw): raise AssertionError('plan attempted network or build')
socket.socket.connect = denied
socket.getaddrinfo = denied
rust_utils.check_and_maybe_compile = denied
from passivbot_cli.main import main
raise SystemExit(main(['download', sys.argv[1], '--for-optimize', '--plan', '--target-cuda-free-gib', '8']))
"""
    result = subprocess.run([sys.executable, "-c", code, str(cfg)], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(root / "src")},
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout[result.stdout.index('{'):])
    assert report["kind"] == "preparation-plan"
    assert report["consumer"] == "optimize"
    assert not (tmp_path / "caches").exists()
    assert not (tmp_path / "backtests").exists()


def test_plan_cli_refuses_stale_native_without_attempting_build(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
import sys, rust_utils
def denied(*a, **kw): raise AssertionError('plan attempted build')
rust_utils.check_and_maybe_compile = denied
rust_utils.verify_loaded_runtime_extension = lambda **kw: {'runtime_compiled_source_stamp': 'stale'}
sys.argv = ['download', '--plan']
import ohlcv_download
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(root / "src")},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert 'Read-only planning requires a current stamped Rust extension' in result.stderr
    assert 'plan attempted build' not in result.stderr
    assert not list(tmp_path.iterdir())


def test_prepared_summary_describes_non_suite_input_and_missing_manifest(tmp_path):
    from simulation_preparation import dataset_summary, describe_prepared_inputs
    config = canonical()
    dataset = dataset_summary("binance", tmp_path, ["ETH", "BTC"], 10)
    assert dataset["manifest_present"] is False
    assert dataset["manifest_sha256"] is None
    report = describe_prepared_inputs(config, datasets=[dataset])
    assert report["cases"] == [{"label": "base", "start_date_literal": "2024-01-01",
                                 "end_date_literal": "2024-01-02", "exchanges": ["binance"],
                                 "ordered_coins": {"binance": ["ETH", "BTC"]}, "rows": {"binance": 10}}]
    dataset["ordered_coins"].reverse()
    assert report["datasets"][0]["ordered_coins"] == ["ETH", "BTC"]
