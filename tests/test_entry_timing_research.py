import math

import pytest

from tools.research_entry_timing import directionality, composition_experiment


@pytest.mark.parametrize("span", [1.0, 15.5, 60.0, 240.25])
def test_ema_candidates_are_bounded_symmetric_and_return_scale_invariant(span):
    returns = [0.001, -0.003, 0.002, 0.0, -0.001] * 200
    base = directionality(returns, span)
    flipped = directionality([-r for r in returns], span)
    scaled = directionality([r * 7 for r in returns], span)
    for key in ("ew_efficiency", "ew_rms", "ew_sign"):
        assert 0.0 <= base[key] <= 1.0
        assert flipped[key] == pytest.approx(base[key])
        assert scaled[key] == pytest.approx(base[key])


def test_efficiency_remembers_direction_during_flat_tail_while_rms_decays():
    now = directionality([0.0] * 600 + [-0.1], 60.0)
    later = directionality([0.0] * 600 + [-0.1] + [0.0] * 300, 60.0)
    assert now["ew_efficiency"] == later["ew_efficiency"] == 1.0
    assert now["ew_rms"] == pytest.approx(math.sqrt(2.0 / 61.0))
    assert later["ew_rms"] < 0.002
    assert later["window_efficiency"] == 0.0
    assert directionality([0.0] * 600, 60.0)["ew_efficiency"] == 0.0


def test_composition_equivalence_and_zero_base_difference():
    result = composition_experiment()
    assert result["max_rescaled_error_minutes"] < 1e-10
    assert result["zero_base_additive_minutes"] == 15.0
    assert result["zero_base_multiplicative_minutes"] == 0.0
