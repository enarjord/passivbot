import sys

import numpy as np
import pytest

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
)
from tools.gpu_proxy_benchmark import (
    _bounded_positive,
    _fixture_sha256,
    _parameter_matrix,
    _require_mps_torch,
    build_parser,
    run_benchmark_case,
)


@pytest.mark.parametrize(
    "keys",
    (
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    ),
)
def test_gpu_proxy_benchmark_candidate_matrix_is_fixed_and_finite(keys):
    first = _parameter_matrix(keys, 8, 7)
    second = _parameter_matrix(keys, 8, 7)
    changed = _parameter_matrix(keys, 8, 8)

    assert first.shape == (8, len(keys))
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert not np.array_equal(first, changed)


def test_gpu_proxy_benchmark_rejects_non_positive_sizes():
    parser = build_parser()

    with pytest.raises(SystemExit):
        _bounded_positive(parser, "--candidates", 0, 10)
    with pytest.raises(SystemExit):
        _bounded_positive(parser, "--candidates", 11, 10)


def test_gpu_proxy_benchmark_accepts_independent_dispatch_batch_size():
    args = build_parser().parse_args(
        ["--candidates", "4096", "--dispatch-batch-size", "512"]
    )

    assert args.candidates == 4096
    assert args.dispatch_batch_size == 512


def test_gpu_proxy_benchmark_reports_profiled_dispatch_chunks(monkeypatch):
    class FakeProxy:
        def evaluate(self, _candidates):
            self.last_profile = {
                "actual_dispatch_batch_sizes": [2, 2, 2, 2],
                "cold_dispatch_count": 0,
                "dispatch_chunk_count": 4,
                "dispatch_count": 4,
                "kernel_candidate_bars": 80,
                "timings_seconds": {
                    "cold_compilation": 0.0,
                    "device_to_host": 0.1,
                    "host_overhead": 0.2,
                    "kernel_execution": 0.3,
                    "warm_library_lookup": 0.0,
                },
            }

    monkeypatch.setattr(
        "tools.gpu_proxy_benchmark._build_case",
        lambda *_args, **_kwargs: (FakeProxy(), [{}] * 8, 10, 1, 1, "fixture"),
    )

    report = run_benchmark_case(
        "ema-single-long",
        candidates=8,
        dispatch_batch_size=2,
        warm_runs=1,
        single_bars=10,
        multicoin_bars=10,
        coins=2,
        seed=7,
    )

    assert report["actual_dispatch_batch_size"] == 2
    assert report["dispatch_chunk_count"] == 4
    assert report["dispatch_count_per_run"] == 4
    assert report["kernel_candidate_bars"] == 80


def test_gpu_proxy_benchmark_reports_missing_optional_gpu_dependencies(
    monkeypatch, capsys
):
    parser = build_parser()
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(SystemExit) as exc:
        _require_mps_torch(parser)

    assert exc.value.code == 2
    assert ".[full,gpu-mps]" in capsys.readouterr().err


def test_gpu_proxy_benchmark_fixture_hash_covers_shape_dtype_and_values():
    baseline = np.asarray([[1.0, 2.0]], dtype=np.float64)

    assert _fixture_sha256(baseline) == _fixture_sha256(baseline.copy())
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline.astype(np.float32))
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline.reshape(2, 1))
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline + 1.0)
