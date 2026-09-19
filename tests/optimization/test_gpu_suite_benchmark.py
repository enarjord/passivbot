import numpy as np
import pytest

from tools import gpu_suite_benchmark as benchmark


@pytest.mark.parametrize("flag,value", [("--bars", "0"), ("--bars", "100001"),
    ("--candidates", "513"), ("--coins", "1"), ("--repeats", "6")])
def test_benchmark_rejects_unbounded_work_before_loading_gpu(monkeypatch, flag, value):
    monkeypatch.setattr(benchmark, "_require_mps_torch", lambda _: pytest.fail("GPU loaded"))
    with pytest.raises(SystemExit):
        benchmark.main([flag, value])


@pytest.mark.parametrize("corrupt", [False, True])
def test_benchmark_checks_parity_and_alternates_measurement_order(monkeypatch, corrupt):
    calls = []

    class Proxy:
        def evaluate(self, candidates):
            calls.append(len(candidates))
            return [dict(score=c["long_n_positions"] + (1 if corrupt and len(candidates) > 2 else 0),
                         optional=np.nan) for c in candidates]

    monkeypatch.setattr(benchmark, "_build_case", lambda *a, **kw: (Proxy(), [{}, {}], "fixture"))
    monkeypatch.setattr(benchmark, "synchronize", lambda: None)
    if corrupt:
        with pytest.raises(AssertionError):
            benchmark.run_benchmark(bars=120, candidates=2, coins=2, repeats=2)
        return
    report = benchmark.run_benchmark(bars=120, candidates=2, coins=2, repeats=2)
    assert calls == [1] + [2] * 9 + [18, 18] + [2] * 9
    assert report["parity"] == "exact"
    assert report["fixture_sha256"] == "fixture"
    assert len(report["wall_seconds"]["separate"]) == 2
    assert len(report["wall_seconds"]["batched"]) == 2
