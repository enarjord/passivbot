from types import SimpleNamespace
import sys

import numpy as np
import pytest

from optimization.gpu.model import ProxyMarket, ProxyRun, build_mps_data, build_mps_multicoin_data


@pytest.fixture
def cpu_packer(monkeypatch):
    class Array(np.ndarray):
        def contiguous(self):
            return self

    calls = []

    def as_tensor(value, dtype=None, device=None):
        calls.append(device)
        return np.asarray(value, dtype=dtype).view(Array)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        as_tensor=as_tensor, float32=np.float32, int32=np.int32, mps=SimpleNamespace()
    ))
    return calls


def _run(first=0, last=5):
    return ProxyRun(1000.0, 0, 0, 0, 0, 0, 60000, 0.05, first, last)


def _market():
    return ProxyMarket(0.001, 0.01, 0.001, 1.0, 1.0, 0.0002)


@pytest.mark.parametrize("multicoin", [False, True])
@pytest.mark.parametrize("bad_hlc", [
    [np.nan, np.nan, np.nan], [101.0, 99.0, np.nan], [np.inf, 99.0, 100.0],
    [0.0, 99.0, 100.0], [101.0, 99.0, np.nextafter(0.0, 1.0)],
    [np.finfo(np.float64).max, 99.0, 100.0],
])
def test_direct_builders_reject_internal_invalid_hlc_before_gpu_allocation(cpu_packer, multicoin, bad_hlc):
    values = np.tile([101.0, 99.0, 100.0, 1.0], (6, 2, 1))
    values[3, 0, :3] = bad_hlc
    original = values.copy()
    timestamps = np.arange(6) * 60000
    with pytest.raises(ValueError, match="coin index 0, invalid candle at 3"):
        if multicoin:
            build_mps_multicoin_data(values, timestamps, [_run()] * 2, [_market()] * 2)
        else:
            build_mps_data(*[values[:, 0, column] for column in range(3)], timestamps, _run(), _market())
    assert cpu_packer == []
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("multicoin", [False, True])
def test_direct_builders_allow_unavailable_listing_prefix_and_delisting_tail(cpu_packer, multicoin):
    values = np.tile([101.0, 99.0, 100.0, 1.0], (6, 2, 1))
    values[:2, 0] = np.nan
    values[5:, 0] = np.nan
    timestamps = np.arange(6) * 60000
    if multicoin:
        packed = build_mps_multicoin_data(values, timestamps, [_run(2, 4), _run()], [_market()] * 2)
        np.testing.assert_array_equal(packed["coin_settings"][:, 6:8], [[2, 4], [0, 5]])
    else:
        packed = build_mps_data(*[values[:, 0, column] for column in range(3)], timestamps, _run(2, 4), _market())
        np.testing.assert_array_equal(packed["valid"], [False, False, True, True, True, False])
    assert cpu_packer
