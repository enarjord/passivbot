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
