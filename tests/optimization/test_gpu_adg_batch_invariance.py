"""Host ADG arithmetic must not depend on candidate batching."""

import math

import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.metrics import _raw_strategy_equity_metrics, _smoothed_gain_adg


@pytest.mark.parametrize("size", [1, 7, 8, 9, 31, 500])
def test_cpu_f64_adg_and_sortino_match_scalar_formula_at_each_batch_position(size):
    expected = math.pow(0.5, 0.5) - 1.0
    for position in sorted({0, min(1, size - 1), min(5, size - 1), size - 1}):
        equity = torch.arange(size * 2, dtype=torch.float64).reshape(size, 2) + 100.0
        equity[position] = torch.tensor([100.0, 0.0], dtype=torch.float64)
        gain_multiplier, adg_fraction = _smoothed_gain_adg(equity, torch.ones_like(equity, dtype=torch.bool))

        assert gain_multiplier[position].item().hex() == 0.5.hex()
        assert adg_fraction[position].item().hex() == expected.hex()
        actual = _raw_strategy_equity_metrics(
            {"raw_strategy_day_end_eq": equity, "raw_strategy_day_min_eq": equity},
            {"adg_fraction_strategy_equity", "sortino_ratio_strategy_equity"},
        )
        # The sole daily loss is -1, so its downside deviation is exactly one.
        for name in ("adg_fraction_strategy_equity", "sortino_ratio_strategy_equity"):
            assert actual[name][position].item().hex() == expected.hex()


def _scalar_current_formula(equity, active):
    values = [value for value, valid in zip(equity, active) if valid]
    if len(values) < 2:
        return 0.0, 0.0
    if not values[0] > 0.0:
        return math.inf, math.inf
    # Match the existing newest-first tail accumulation and clamp exactly.
    tail = values[-3:]
    end = sum(reversed(tail)) / len(tail)
    if not end > 0.0:
        return -1.0, -1.0
    gain_multiplier = end / values[0]
    adg_fraction = math.pow(max(gain_multiplier, 1e-12), 1.0 / len(values)) - 1.0 if gain_multiplier > 0.0 else -1.0
    return gain_multiplier, adg_fraction


@pytest.mark.parametrize(
    "equity,active",
    [
        ([], []),
        ([100.0], [True]),
        ([100.0, 110.0, 120.0, 130.0], [True] * 4),
        ([100.0, 0.0], [True, True]),
        ([100.0, -100.0], [True, True]),
        ([100.0, -200.0], [True, True]),
        ([0.0, 100.0], [True, True]),
        ([-100.0, 200.0], [True, True]),
        ([math.nan, 100.0], [True, True]),
        ([100.0, math.nan], [True, True]),
        ([100.0, math.inf], [True, True]),
        ([math.inf, 100.0], [True, True]),
        ([1e-308, 1e308], [True, True]),
        ([1e308, 1e308], [True, True]),
        ([1.0, 1e-20, 1e-20, 1e-20], [True] * 4),
        ([0.0, 100.0, 0.0], [False, True, True]),
        ([100.0, math.nan, 120.0, 130.0], [True, False, True, True]),
        ([math.nan, math.inf], [False, False]),
    ],
)
def test_cpu_f64_adg_preserves_current_mask_clamp_and_sentinel_formula(equity, active):
    expected_gain, expected_adg = _scalar_current_formula(equity, active)
    for size in (1, 7, 500):
        values = torch.tensor([equity] * size, dtype=torch.float64)
        mask = torch.tensor([active] * size, dtype=torch.bool)
        gain_multiplier, adg_fraction = _smoothed_gain_adg(values, mask)
        assert gain_multiplier.dtype == adg_fraction.dtype == torch.float64
        assert gain_multiplier.device.type == adg_fraction.device.type == "cpu"
        assert [value.hex() for value in gain_multiplier.tolist()] == [expected_gain.hex()] * size
        assert [value.hex() for value in adg_fraction.tolist()] == [expected_adg.hex()] * size


def test_cpu_f64_adg_accepts_an_empty_candidate_batch():
    gain_multiplier, adg_fraction = _smoothed_gain_adg(
        torch.empty((0, 2), dtype=torch.float64), torch.empty((0, 2), dtype=torch.bool)
    )
    assert gain_multiplier.shape == adg_fraction.shape == (0,)
    assert gain_multiplier.dtype == adg_fraction.dtype == torch.float64


@pytest.mark.parametrize("size", [7, 500])
def test_cpu_f64_adg_keeps_each_candidates_mask_and_duration(size):
    cases = [
        ([0.0, 0.0, 0.0, 0.0], [False] * 4),
        ([100.0, 0.0, 0.0, 0.0], [True, False, False, False]),
        ([100.0, 0.0, 0.0, 0.0], [True, True, False, False]),
        ([100.0, 120.0, 160.0, 0.0], [True, True, True, False]),
        ([100.0, 110.0, 120.0, 130.0], [True] * 4),
        ([0.0, 100.0, 0.0, 0.0], [False, True, True, False]),
        ([0.0, 100.0, 0.0, 0.0], [True, True, False, False]),
    ]
    rows = [cases[index % len(cases)] for index in range(size)]
    gain_multiplier, adg_fraction = _smoothed_gain_adg(
        torch.tensor([values for values, _mask in rows], dtype=torch.float64),
        torch.tensor([mask for _values, mask in rows], dtype=torch.bool),
    )
    expected = [_scalar_current_formula(values, mask) for values, mask in rows]
    assert [value.hex() for value in gain_multiplier.tolist()] == [pair[0].hex() for pair in expected]
    assert [value.hex() for value in adg_fraction.tolist()] == [pair[1].hex() for pair in expected]


def test_cpu_f32_adg_retains_tensor_power(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("f32 reduction must retain tensor power")

    monkeypatch.setattr(math, "pow", forbidden)
    equity = torch.tensor([[100.0, 0.0]] * 500, dtype=torch.float32)
    gain_multiplier, adg_fraction = _smoothed_gain_adg(equity, torch.ones_like(equity, dtype=torch.bool))
    expected = torch.full((500,), 0.5, dtype=torch.float32) ** torch.full((500,), 0.5) - 1.0
    torch.testing.assert_close(gain_multiplier, torch.full_like(gain_multiplier, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(adg_fraction, expected, rtol=0, atol=0)
