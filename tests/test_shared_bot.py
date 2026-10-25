import pytest

from config.shared_bot import flatten_shared_bot_side, require_grouped_bot_value


def test_flatten_shared_bot_side_prefers_grouped_value_over_stale_runtime_alias():
    side_cfg = {
        "hsl": {"drawdown_threshold_fraction": 1.0},
        "hsl_drawdown_threshold_fraction": 0.3,
    }

    flattened = flatten_shared_bot_side(side_cfg)

    assert flattened["hsl_drawdown_threshold_fraction"] == pytest.approx(1.0)


def test_require_grouped_bot_value_can_prefer_flat_raw_override():
    side_cfg = {
        "risk": {"total_wallet_exposure_limit_ratio": 1.0},
        "total_wallet_exposure_limit": 0.0,
    }

    assert require_grouped_bot_value(
        side_cfg,
        "long",
        "total_wallet_exposure_limit",
        prefer_flat=True,
    ) == pytest.approx(0.0)
