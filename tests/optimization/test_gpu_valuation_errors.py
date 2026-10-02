"""Unavailable prices and unusable HSL policy inputs remain distinct fatal errors."""

import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.mps_kernel import _require_available_held_valuation


@pytest.mark.parametrize(
    "marker,message", [(-2, "held-position valuation"), (-4, "HSL controller inputs")]
)
def test_unavailable_replay_error_is_distinct_and_bounded(marker, message):
    output = torch.zeros((1000, 10))
    output[:, 9] = marker
    with pytest.raises(ValueError, match=message) as raised:
        _require_available_held_valuation(output)
    assert "candidate rows [0, 1, 2, 3, 4, 5, 6, 7] (+992 more)" in str(raised.value)
    assert len(str(raised.value)) < 260


def test_history_overflow_remains_fatal_and_liquidations_are_valid():
    output = torch.zeros((2, 10))
    output[:, 9] = torch.tensor([-1, 0])
    _require_available_held_valuation(output)
    output[1, 9] = -3
    with pytest.raises(RuntimeError, match="PnL history overflow"):
        _require_available_held_valuation(output)
