import json

import numpy as np
import pytest

from tools import gpu_parity


@pytest.fixture
def require_real_passivbot_rust_module():
    import passivbot_rust
    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("real native Rust extension required")
    return passivbot_rust


def args(*options):
    return gpu_parity.build_parser().parse_args(list(options))


@pytest.mark.parametrize("option", [
    ["--sides", "long"], ["--coins", "2"], ["--bars", "5760"], ["--seed", "7"],
    ["--hsl", "disabled"], ["--unstuck"], ["--market-orders"],
    ["--filter-by-min-effective-cost"],
])
def test_prepared_inputs_reject_explicit_fixture_options_before_loading(option):
    options = args("--config", "unused.json", "--dataset", "unused.npz",
                   "--markets", "unused-markets.json", *option)
    with pytest.raises(ValueError, match="fixture-only options.*" + option[0]):
        gpu_parity.prepared_inputs(options)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hsl", ["disabled", "coin", "pside", "unified"])
def test_fixtures_are_canonical_and_keep_ordered_prepared_metadata(strategy, hsl):
    options = args("--fixture", strategy, "--bars", "128", "--sides", "both", "--hsl", hsl)
    config, candles, markets, btc, timestamps = gpu_parity.fixture_inputs(options)
    assert candles.shape == (128, 2, 4)
    assert config["backtest"]["coins"]["binance"] == ["COIN00", "COIN01"]
    assert config["live"]["strategy_kind"] == strategy
    from utils import date_to_ts
    assert date_to_ts(config["backtest"]["start_date"]) == timestamps[0]
    assert date_to_ts(config["backtest"]["end_date"]) == timestamps[-1] + 60_000
    assert config["bot"]["long"]["hsl"]["enabled"] == (hsl in {"coin", "pside"})
    again = gpu_parity.fixture_inputs(options)
    assert gpu_parity._identity(config, (candles, btc, timestamps), markets, "binance") == (
        gpu_parity._identity(again[0], (again[1], again[3], again[4]), again[2], "binance")
    )


def test_prepared_snapshot_roundtrip_and_coin_order_guard(tmp_path):
    options = args("--fixture", "trailing_martingale", "--bars", "128")
    config, candles, markets, btc, timestamps = gpu_parity.fixture_inputs(options)
    config_file, markets_file, dataset = (tmp_path / name for name in ("config.json", "markets.json", "data.npz"))
    config_file.write_text(json.dumps(config))
    markets_file.write_text(json.dumps(markets))
    np.savez(dataset, hlcvs=candles, timestamps=timestamps, btc=btc, coins=np.array(["COIN00", "COIN01"]))
    options = args("--config", str(config_file), "--dataset", str(dataset), "--markets", str(markets_file))
    prepared = gpu_parity.prepared_inputs(options)
    np.testing.assert_array_equal(prepared[1], candles)
    assert prepared[0]["backtest"]["coins"] == config["backtest"]["coins"]
    config["optimize"]["fixed_runtime_overrides"] = {"bot.long.risk.total_wallet_exposure_limit": 0.25}
    config_file.write_text(json.dumps(config))
    prepared = gpu_parity.prepared_inputs(options)
    assert prepared[0]["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.25
    np.savez(dataset, hlcvs=candles, timestamps=timestamps, btc=btc, coins=np.array(["COIN01", "COIN00"]))
    with pytest.raises(ValueError, match="coin order"):
        gpu_parity.prepared_inputs(options)


def test_main_reports_execution_failure_as_strict_json(monkeypatch, capsys):
    from types import SimpleNamespace
    import sys
    # Exercise the CLI error boundary without installing an optional GPU runtime.
    monkeypatch.setitem(sys.modules, "optimization.gpu.metrics", SimpleNamespace(
        validate_gpu_metric_names=lambda names: names,
    ))
    def failed(*_args, **_kwargs):
        raise RuntimeError("device launch failed")
    monkeypatch.setattr(gpu_parity, "run_comparison", failed)
    assert gpu_parity.main(["--fixture", "trailing_martingale", "--bars", "128"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "execution_failed"
    assert report["error"]["message"] == "device launch failed"


def test_cli_help_does_not_need_optional_gpu_runtime(capsys):
    with pytest.raises(SystemExit) as completed:
        gpu_parity.main(["--help"])
    assert completed.value.code == 0
    assert "--dataset" in capsys.readouterr().out


def test_cli_registry_exposes_offline_parity_tool():
    from passivbot_cli.main import TOOL_COMMANDS
    assert TOOL_COMMANDS["gpu-parity"].module == "tools.gpu_parity"


def test_main_uses_prepared_reducer_and_preserves_stdout_when_save_fails(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace
    import sys
    from optimization.gpu.parity import compare_limits
    monkeypatch.setitem(sys.modules, "optimization.gpu.metrics", SimpleNamespace(
        validate_gpu_metric_names=lambda names: names,
    ))
    original = gpu_parity.fixture_inputs

    def inputs(options):
        prepared = original(options)
        prepared[0]["backtest"]["reducer"]["fills_per_day"] = "std"
        prepared[0]["optimize"]["limits"] = [
            {"metric": "fills_per_day", "penalize_if": "greater_than", "value": 0}
        ]
        return prepared

    def comparison(_inputs, _exchange, _metrics, _policies, checks, **_kwargs):
        assert checks[0]["metric_key"] == "fills_per_day_std"
        feasibility = compare_limits({"fills_per_day": 100}, {"fills_per_day": 100}, checks)
        assert feasibility["cpu_feasible"] is True
        return {"passed": feasibility["passed"], "feasibility": feasibility}

    monkeypatch.setattr(gpu_parity, "fixture_inputs", inputs)
    monkeypatch.setattr(gpu_parity, "run_comparison", comparison)
    command = ["--fixture", "trailing_martingale", "--bars", "128"]
    assert gpu_parity.main(command) == 0
    assert json.loads(capsys.readouterr().out)["passed"]
    assert gpu_parity.main([*command, "--report", str(tmp_path / "missing" / "report.json")]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["passed"]
    assert "report_save_failed" in captured.err


def test_real_cuda_parity_cli(require_real_passivbot_rust_module, capsys):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    code = gpu_parity.main(["--fixture", "trailing_martingale", "--bars", "5760"])
    report = json.loads(capsys.readouterr().out)
    # This tool is meant to expose simulator differences, not conceal them to
    # make its smoke pass. Execution and the three established metrics are checked.
    assert code in (0, 1), report
    for name in gpu_parity.DEFAULT_METRICS:
        assert report["metrics"][name]["status"] == "match", report
    assert report["feasibility"]["passed"]
    assert report["rust_source_fingerprint"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_real_cuda_multicoin_filter_preserves_funded_cpu_and_gpu_runs(
    require_real_passivbot_rust_module, strategy, sides
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    options = args("--fixture", strategy, "--sides", sides)
    inputs = gpu_parity.fixture_inputs(options)
    baseline = gpu_parity.run_comparison(
        inputs, "binance", ["fills_per_day"], gpu_parity.DEFAULT_TOLERANCES
    )
    inputs[0]["backtest"]["filter_by_min_effective_cost"] = True
    filtered = gpu_parity.run_comparison(
        inputs, "binance", ["fills_per_day"], gpu_parity.DEFAULT_TOLERANCES
    )
    # Both engines independently consider these markets affordable. Filtering
    # must not introduce the old all-zero screen; other known simulator gaps
    # remain visible rather than widening the tool's CPU/GPU tolerance here.
    for engine in ("cpu", "gpu"):
        assert baseline["metrics"]["fills_per_day"][engine] > 0
        assert filtered["metrics"]["fills_per_day"][engine] == pytest.approx(
            baseline["metrics"]["fills_per_day"][engine], rel=1e-7
        )


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("initial_qty_pct, funded", [
    (0.01 * (1 - 1e-4), False), (float(np.nextafter(0.01, 0.0)), False),
    (0.01, True), (0.01 * (1 + 1e-4), True),
])
def test_real_cuda_multicoin_cost_admission_boundary_is_measured(
    require_real_passivbot_rust_module, strategy, initial_qty_pct, funded
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    inputs = gpu_parity.fixture_inputs(args(
        "--fixture", strategy, "--bars", "128", "--filter-by-min-effective-cost"
    ))
    config, candles, markets, _btc, _timestamps = inputs
    strategy_config = config["bot"]["long"]["strategy"][strategy]
    if strategy == "ema_anchor":
        strategy_config["base_qty_pct"] = initial_qty_pct
    else:
        strategy_config["entry"]["initial_qty_pct"] = initial_qty_pct
    candles[:, :, :3] = (101.0, 99.0, 100.0)
    for coin in config["backtest"]["coins"]["binance"]:
        markets[coin].update(min_cost=5.0, maker=0.0, taker=0.0)
    report = gpu_parity.run_comparison(
        inputs, "binance", ["fills_per_day"], gpu_parity.DEFAULT_TOLERANCES
    )
    metric = report["metrics"]["fills_per_day"]
    assert (metric["cpu"] > 0) is funded
    if initial_qty_pct == float(np.nextafter(0.01, 0.0)):
        # This input and exactly 0.01 have identical float32 payloads. CPU
        # rejects its sub-minimum float64 projection; GPU rounding may admit
        # it. Do not require perfect identity or conceal the discontinuity.
        if metric["gpu"] != metric["cpu"]:
            assert metric["status"] == "mismatch"
            assert not report["passed"]
    else:
        assert (metric["gpu"] > 0) is funded
        assert metric["status"] == "match", report


@pytest.mark.parametrize("sides", ["short", "both"])
def test_real_cuda_passive_recursive_ladders_recover_cpu_metrics(
    require_real_passivbot_rust_module, sides
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    inputs = gpu_parity.fixture_inputs(args("--fixture", "trailing_martingale", "--sides", sides))
    report = gpu_parity.run_comparison(
        inputs, "binance", gpu_parity.DEFAULT_METRICS, gpu_parity.DEFAULT_TOLERANCES
    )
    # This fixture exposed missing passive entry/close suffixes, losing ~60% of
    # fills and ~95% of ADG. Keep the stricter diagnostic policy unchanged and
    # separately guard the recovered trajectory to 0.1%; minor tick/float32
    # differences still appear in the report and need case-specific assessment.
    for name in ("adg_strategy_eq", "fills_per_day"):
        metric = report["metrics"][name]
        assert metric["cpu"] > 0
        assert metric["gpu"] == pytest.approx(metric["cpu"], rel=1e-3)
    assert report["metrics"]["drawdown_worst_strategy_eq"]["status"] == "match"
