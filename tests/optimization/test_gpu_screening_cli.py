"""GPU suites through the offline CLI, exact Rust validation and resume."""

import json
import os
import pickle
import sys

import msgpack
import numpy as np
import pytest

from test_hsl_revised_offline_runtime import deny_network, offline_cli_config, runtime_inputs
from test_simulation_offline import START, SYMBOL

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("screening", [False, True])
async def test_gpu_suite_cli_dates_exact_validation_and_resume(
    tmp_path, monkeypatch, capsys, screening
):
    from optimize import main
    from optimization.backends import gpu_backend
    from optimization.shape import build_optimization_shape
    from config.optimize_bounds import set_flat_optimize_bound
    from ohlcv_catalog import OhlcvCatalog
    from ohlcv_store import OhlcvStore

    suite_calls = []
    original_evaluate = gpu_backend._evaluate_gpu_suite_proxies

    def record_evaluate(suite, proxies, candidates, **kwargs):
        # The optimizer must allow compatible grouping even without screening.
        # These different-date scenarios still go through compatibility checks.
        assert kwargs["batch_compatible_scenarios"] is True
        suite_calls.append(
            (len(candidates), tuple(kwargs["screening_scenarios"]), kwargs["evaluation_stage"])
        )
        return original_evaluate(suite, proxies, candidates, **kwargs)

    monkeypatch.setattr(gpu_backend, "_evaluate_gpu_suite_proxies", record_evaluate)
    cfg = offline_cli_config(tmp_path, monkeypatch, "coin")
    # Add a second day so the scenario date boundaries really differ.
    _, _, candles, _ = runtime_inputs("coin")
    catalog = OhlcvCatalog(tmp_path / "caches/ohlcvs/catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches/ohlcvs", catalog)
    times = START + np.arange(1441, 2881, dtype=np.int64) * 60000
    rows = np.resize(candles[:, 0], (1440, 4)).astype(np.float32)
    store.write_rows("binance", "1m", SYMBOL, times, rows)
    cfg["backtest"].update(
        end_date="2024-01-03",
        suite_enabled=True,
        scenarios=[
            {"label": "recent", "start_date": "2024-01-02"},
            {"label": "full", "start_date": "2024-01-01"},
        ],
    )
    cfg["live"]["approved_coins"]["short"] = list(cfg["live"]["approved_coins"]["long"])
    cfg["bot"]["short"]["risk"].update(n_positions=0, total_wallet_exposure_limit=0.0)
    cfg["live"]["strategy_kind"] = "trailing_martingale"
    cfg["optimize"].update(
        backend="gpu", iters=32, n_cpus=1, population_size=8,
        scoring=[{"metric": "adg_usd", "goal": "max"}], limits=[], seed=42,
        enable_overrides=[], compress_results_file=False, write_all_results=True,
    )
    cfg["optimize"]["gpu"].update(
        population_size=8, batch_size=8, exact_workers=1,
        max_pending_exact=2, validate_per_generation=2, drift_probes=1,
        screening=dict(scenarios=["recent"] if screening else [],
                       survival_fraction=0.5, min_survivors=2),
        successive_halving={"enabled": False},
    )
    shape = build_optimization_shape(cfg)
    for key, key_path in shape.key_paths:
        value = cfg
        for part in key_path:
            value = value[part]
        set_flat_optimize_bound(cfg["optimize"]["bounds"], "trailing_martingale", key, [value, value])
    cfg["optimize"]["bounds"]["long"]["hsl"]["red_threshold"] = [0.01, 0.2]

    # Spawned exact workers deny IP networking as well; local Unix IPC is allowed.
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\n_connect=socket.socket.connect\n"
        "def denied(*a, **k): raise RuntimeError('offline GPU optimizer attempted network')\n"
        "def connect(self,address):\n"
        "    if self.family != socket.AF_UNIX: return denied()\n"
        "    return _connect(self,address)\n"
        "socket.socket.connect=connect\nsocket.getaddrinfo=denied\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(guard) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        sys, "argv", ["optimize", str(path), "--suite", "y",
                      "--optimize.gpu.screening.scenarios",
                      json.dumps(cfg["optimize"]["gpu"]["screening"]["scenarios"])]
    )
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code == 0
    artifact, = (tmp_path / "optimize_results").rglob("all_results.bin")
    with artifact.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, raw=False))
    assert records
    checkpoint = artifact.parent / "checkpoint.pkl"
    state = pickle.loads(checkpoint.read_bytes())
    log_output = capsys.readouterr().err
    assert "Removed disabled legacy" in log_output
    if screening:
        assert "stages=screening:8,full:4" in log_output
        assert (8, ("recent",), "screening") in suite_calls
        assert (4, (), "full") in suite_calls
    else:
        assert "GPU scenario screening" not in log_output
        assert (8, (), "full") in suite_calls
        assert all(labels == () and stage == "full" for _, labels, stage in suite_calls)
    assert state["generation"] > 0
    assert state["exact_done"] >= 32
    assert state["halt_reason"] is None
    assert state["optimizer_evaluation_contract"]["live"]["hsl_engine"] == "revised"
    assert all(record["live"]["hsl_engine"] == "revised" for record in records)
    assert "successive_halving" not in records[0]["optimize"]["gpu"]

    before = artifact.stat().st_size
    cfg["optimize"]["iters"] = 48
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        sys, "argv", ["optimize", str(path), "--suite", "y", "--resume", str(artifact.parent)]
    )
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code == 0
    assert artifact.stat().st_size > before
    resumed = pickle.loads(checkpoint.read_bytes())
    assert resumed["exact_done"] > state["exact_done"]
    assert resumed["optimizer_evaluation_contract"] == state["optimizer_evaluation_contract"]

    # A different screening set cannot reuse the old search state.
    cfg["optimize"]["gpu"]["screening"]["scenarios"] = ["full"]
    path.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit) as rejected:
        await main()
    assert rejected.value.code == 1
    assert "Cannot resume because critical parameters have changed" in capsys.readouterr().err
