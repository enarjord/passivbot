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
    build_parser,
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


def test_gpu_proxy_benchmark_fixture_hash_covers_shape_dtype_and_values():
    baseline = np.asarray([[1.0, 2.0]], dtype=np.float64)

    assert _fixture_sha256(baseline) == _fixture_sha256(baseline.copy())
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline.astype(np.float32))
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline.reshape(2, 1))
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline + 1.0)
