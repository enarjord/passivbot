"""Coin HSL must discard completed episodes before replaying a new one."""

from functools import lru_cache

import numpy as np
import pytest


torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Apple MPS and NVIDIA CUDA unavailable",
)


@lru_cache(maxsize=4)
def _boundary_library(strategy, side):
    import passivbot_rust

    source = getattr(
        passivbot_rust, f"mps_{strategy}_{side}_hsl_source_py"
    )() if strategy == "trailing_martingale" else passivbot_rust.mps_ema_anchor_source_py()
    return compile_shader(source + r"""
kernel void coin_episode_boundary(
    constant float* params [[buffer(0)]],
    device float2* values [[buffer(1)]],
    device int2* indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    HslState h = load_hsl(params, 0, 0);
    HslRollingPnlWindow window = init_hsl_rolling_pnl_window();
    record_hsl_rolling_pnl(window, values, indices, 0, 16, 0, 100, true, 0.0f);
    record_hsl_rolling_pnl(window, values, indices, 0, 16, 1, 100, true, -200.0f);
    prepare_coin_hsl_rolling_signal(h, window, values, indices, 0, 16, 1, 100, -200.0f);
    update_one_side_hsl(h, 800.0f, 1000.0f, -200.0f, 0.0f, true, false, 1.0f, 60000.0f);
    // The directional replay caller clears its window only on this return value.
    bool reset = finish_hsl_scoped_episode_at_flat(
        h, nullptr, params[11] > 0.0f, false,
        800.0f, 1000.0f, -200.0f, -200.0f, 2.0f, 60000.0f
    );
    if (reset) reset_hsl_rolling_pnl_window(window);
    output[0] = reset;
    output[1] = window.event_count;
    output[2] = h.halted;
    output[3] = h.no_restart_latched;
    output[4] = h.cooldown_until_k;
    try_restart_hsl(h, 5.0f, 800.0f);
    prepare_coin_hsl_rolling_signal(h, window, values, indices, 0, 16, 5, 100, -200.0f);
    HslSignal signal;
    bool valid = derive_hsl_signal(h, 800.0f, 1000.0f, -200.0f, 0.0f, signal);
    output[5] = valid ? signal.drawdown_raw : -1.0f;
    // Reset must preserve subsequent episode fees/losses, including same-minute fills.
    record_hsl_rolling_pnl(window, values, indices, 0, 16, 5, 100, true, -8.0f);
    prepare_coin_hsl_rolling_signal(h, window, values, indices, 0, 16, 5, 100, -208.0f);
    valid = derive_hsl_signal(h, 792.0f, 1000.0f, -208.0f, 0.0f, signal);
    output[6] = valid ? signal.drawdown_raw : -1.0f;
}
""")


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("policy", [0, 1, 2])
@pytest.mark.parametrize("still_held", [False, True])
def test_coin_red_fill_boundary_resets_window_and_preserves_restart_policy(
    strategy, side, policy, still_held
):
    params = torch.tensor(
        [1., .1, 1., 3., 1., policy, .5, .75, 0., 2., 1., float(still_held)],
        dtype=torch.float32, device=gpu_device(),
    )
    values = torch.zeros((16, 2), dtype=torch.float32, device=gpu_device())
    indices = torch.zeros((16, 2), dtype=torch.int32, device=gpu_device())
    output = torch.zeros(7, dtype=torch.float32, device=gpu_device())
    _boundary_library(strategy, side).coin_episode_boundary(
        params, values, indices, output, threads=(1, 1, 1)
    )
    actual = output.cpu().numpy()
    if still_held:
        np.testing.assert_array_equal(actual[:4], [0, 2, 0, 0])
        assert actual[5] == pytest.approx(.25)
        assert actual[6] == pytest.approx(208 / 792)
    else:
        np.testing.assert_array_equal(actual[:4], [1, 0, 1, policy == 2])
        if policy == 2:
            np.testing.assert_array_equal(actual[5:], [-1, -1])
        else:
            assert actual[4] == 5.0
            assert actual[5] == 0.0
            assert actual[6] == pytest.approx(8 / 792)
