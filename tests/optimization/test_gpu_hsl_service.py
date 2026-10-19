"""Canonical v9 policy transport through the one-coin shared-account route."""

from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()), reason="GPU required")


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_canonical_hsl_policy_reaches_shared_native_shader(mode, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy(mode, strategy, coin_count=1, minutes=128)
    runner, result = raw(proxy, [{}])
    assert runner.hsl_capacity == 2
    assert runner.hsl_fact_capacity == 0
    for side in proxy.sides:
        assert proxy.base_params[side]["hsl_drawdown_ratio_ema_span_minutes"] == 2.5
        assert proxy.base_params[side]["hsl_memory_ema_span_minutes"] == 1440.0
        assert proxy.base_params[side]["hsl_red_threshold"] == pytest.approx(.02)
    assert torch.isfinite(result["balance"]).all()
    assert result["fill_count"].item() > 0


def test_portfolio_candidate_and_static_coin_override_precedence():
    from optimization.gpu.model import TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN
    from test_gpu_hsl_multicoin import make_proxy

    proxy = make_proxy("portfolio", coin_count=1, minutes=128)
    candidates = [{"hsl_red_threshold": .03}, {"hsl_red_threshold": .06}]
    before = deepcopy(candidates)
    for side in proxy.sides:
        matrix = proxy._parameter_matrix(candidates, side)
        np.testing.assert_allclose(matrix[:, proxy.param_keys.index("hsl_red_threshold")], [.03, .06])
    assert candidates == before
    coin = make_proxy("coin", coin_count=1, minutes=128,
                       override={"red_threshold": .07, "ema_span_minutes": 4.5})
    matrix = coin._parameter_matrix([{"long_hsl_red_threshold": .03}], "long")
    assert matrix[0, coin.param_keys.index("hsl_red_threshold")] == pytest.approx(.03)
    runner = coin.fused_runner or coin.runners["long"]
    start = TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN
    # The authored coin policy remains a separate override after candidate defaults.
    assert runner.coin_overrides[0, start + 1].item() == pytest.approx(.07)
    assert runner.coin_overrides[0, start + 2].item() == pytest.approx(4.5)


@pytest.mark.parametrize("span", [1.0, 43200.5])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_disabled_hsl_uses_no_history_for_any_valid_memory_span(span, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy("coin", strategy, coin_count=1, minutes=128,
                       memory_span=span, enabled=False)
    runner, result = raw(proxy, [{}])
    assert runner._hsl_history_bytes_per_candidate() == 0
    assert runner.hsl_fact_capacity == 0
    assert not (result["hsl_triggers_long"] + result["hsl_triggers_short"]).any()
