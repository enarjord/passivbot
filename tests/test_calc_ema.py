import numpy as np
import passivbot_rust as pbr


def test_calc_ema_scalar():
    alpha = 0.1
    alpha_ = 0.9
    prev = 100.0
    new = 110.0
    expected = prev * alpha_ + new * alpha
    res = pbr.calc_ema(alpha, alpha_, prev, new)
    # pbr.calc_ema should return a Python float for scalar input
    assert isinstance(res, float)
    assert res == expected


def test_calc_ema_array():
    alpha = 0.2
    alpha_ = 0.8
    prev = np.array([100.0, 200.0, 300.0], dtype=float)
    new = 110.0
    expected = prev * alpha_ + new * alpha
    res = pbr.calc_ema(alpha, alpha_, prev, new)
    # pbr.calc_ema should return a numpy array for 1-D array input
    assert isinstance(res, np.ndarray)
    assert res.shape == prev.shape
    assert np.allclose(res, expected)
