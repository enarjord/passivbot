import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_rolling_peak_accepts_negative_values():
    tracker = pbr.EquityHardStopRollingPeak()

    assert tracker.update(1_000, -5.0, 1_000) == pytest.approx(-5.0)
    assert tracker.update(1_500, -2.0, 1_000) == pytest.approx(-2.0)
    assert tracker.update(2_100, -7.0, 1_000) == pytest.approx(-2.0)
