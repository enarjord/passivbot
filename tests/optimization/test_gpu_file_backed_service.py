"""File-backed preparation preserves native Metal outputs and ownership."""
from pathlib import Path
import gzip

import numpy as np
import pytest

from hlcvs_manifest import load_numpy_artifact, numpy_artifact_loader
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import BacktestRequest
from optimization.gpu.native import NativeBacktestService
from shared_arrays import SharedArrayManager


@pytest.mark.parametrize("strategy,hsl", [
    ("ema_anchor", "disabled"), ("trailing_martingale", "disabled"),
    ("trailing_martingale", "coin"), ("trailing_martingale", "portfolio"),
])
def test_file_backed_native_metrics_equal_shared_memory(tmp_path, strategy, hsl):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal required")
    from tools.gpu_parity import build_parser, fixture_inputs
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--coins", "3", "--sides", "both", "--bars", "512", "--hsl", hsl,
    ]))
    metrics = ("adg_fraction_strategy_equity", "drawdown_max_fraction_strategy_equity", "backtest_completion_fraction",
               "fills_per_day", "position_held_time_weighted_mean_hours")
    results = []
    for file_backed in (False, True):
        owner = SharedArrayManager(file_backed=file_backed)
        try:
            specs = []
            for index, value in enumerate((candles, btc, timestamps)):
                if file_backed:
                    path = tmp_path / f"{index}.npy.gz"
                    with gzip.open(path, "wb") as handle:
                        np.save(handle, value, allow_pickle=False)
                    with numpy_artifact_loader(owner.load_numpy_artifact):
                        value = load_numpy_artifact(path)
                specs.append(owner.create_from(value)[0])
            dataset = PreparedGpuDataset(config=config, markets=markets, exchange="binance",
                hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
                candle_coins=config["backtest"]["coins"]["binance"], metrics=metrics)
            with NativeBacktestService(batch_size=2, max_pending=4, tuning_mode="off") as gpu:
                gpu.register_dataset("case", dataset)
                result = gpu.submit(BacktestRequest("row", "case", {})).result(timeout=180)
                results.append((result.metrics, result.liquidated))
            owned = tuple(owner._file_arrays)
        finally:
            owner.cleanup()
        if file_backed:
            assert all(not Path(name).exists() for name in owned)
    assert results[0] == results[1]
