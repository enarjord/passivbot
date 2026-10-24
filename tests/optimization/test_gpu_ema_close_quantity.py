"""Exercise the production EMA close producer at float32 remainder boundaries."""

import numpy as np
import pytest


@pytest.mark.parametrize("coin_count", [1, 2])
def test_ema_close_producer_preserves_minimum_venue_remainders(coin_count):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("Metal or CUDA required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    probe = r"""
kernel void ema_close_quantity_probe(
    constant float* cases,
    constant float* bars,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    int coins = int(cases[b * 5]);
    bool short_side = cases[b * 5 + 1] > 0.5f;
    float step = cases[b * 5 + 2];
    float size = cases[b * 5 + 3] * step;
    EmaMulticoinSideConfig config = {};
    config.base_qty_pct = 1.0f;
    config.twel = float(coins);
    config.n_positions = coins;
    EmaMulticoinSideState side;
    const int settings_offset = int(b) * coins * COIN_COLS;
    init_ema_multicoin_side_state(
        side, config, coin_settings + settings_offset, coin_overrides, coins
    );
    for (int c = 0; c < coins; ++c) {
        side.psize[c] = size;
        side.pprice[c] = 100.0f;
        side.ema0[c] = side.ema1[c] = side.ema2[c] = 100.0f;
    }
    JointPortfolioAccount account = init_joint_portfolio_account(cases[b * 5 + 4] * step * 100.0f);
    generate_ema_multicoin_side_orders(
        side, config, account, bars, touch_ticks,
        coin_settings + settings_offset, coin_overrides,
        0, coins, short_side, coins, coins, 0, false, 0.0f, -2, 0ul
    );
    for (int c = 0; c < coins; ++c) {
        output[int(b) * coins + c] = side.close_qty[c] / step;
    }
}
"""
    # Quantities are authored in venue counts; expected closes do not come
    # from repeating the shader's floating-point subtraction in the test.
    # Include quantity/cost-derived minima and representable off-grid neighbors.
    boundaries = [
        (151, 150, 1, 0, 150),  # 1.51 - 1.50 must retain 0.01 at step=0.01.
        (152, 150, 2, 0, 150),
        (152, 150, 1, 2, 150),  # Minimum cost requires a two-step remainder.
        (151, 150, 2, 0, 151),  # Real subminimum dust must still be absorbed.
        (151, 150, 1, 2, 151),
        (150, 150, 1, 0, 150),  # Exact full close.
        (1, 150, 2, 0, 1),      # Position below the minimum.
        (150.5, 150, 1, 0, 150.5),  # Off-grid dust is not snapped up.
        (151.25, 150, 1, 0, 150),   # Off-grid remainder above the minimum.
        (55, 50, 5, 0, 50),  # Packed 0.05 can exceed the product 5 * 0.01.
        (54, 50, 5, 0, 54),  # A real one-step deficit must still close fully.
        (56, 50, 5, 0, 50),
        (55, 50, 1, 4.5, 50),  # Cost minimum rounds up to five steps.
        (55, 50, 5.001, 0, 55),  # Off-grid minimum requires six steps.
        (10, 1, 5, 0, 5),  # The clip itself can be the packed minimum.
        (9, 1, 5, 0, 9),
        (11, 1, 5, 0, 5),
        (10, 1, 1, 4.5, 5),  # A cost-derived minimum-sized clip.
        (np.nextafter(np.float32(2), np.float32(0)), 1, 1, 0,
         np.nextafter(np.float32(2), np.float32(0))),
        (np.nextafter(np.float32(2), np.float32(np.inf)), 1, 1, 0, 1),
    ]
    cases, settings, expected = [], [], []
    for short in (0, 1):
        for step in (1e-12, 1e-6, 0.001, 0.01, 1.0, 10.0, 1e8):
            for size, clip, minimum, cost_steps, close in boundaries:
                cases.append((coin_count, short, step, size, clip))
                settings.extend([
                    [step, 0.01, minimum * step, cost_steps * step * 100.0,
                     1, 0, 0, 10, 0, 100, 100, 0, 0]
                ] * coin_count)
                expected.extend([close] * coin_count)
    device = gpu_device()
    arrays = [torch.tensor(value, device=device) for value in (
        np.array(cases, dtype=np.float32),
        np.array([[100, 100, 100, 1]] * coin_count, dtype=np.float32),
        np.array([[10000, 10000]] * coin_count, dtype=np.int32),
        np.array(settings, dtype=np.float32),
        np.full((coin_count, 34), np.nan, dtype=np.float32),
    )]
    output = torch.empty(len(expected), dtype=torch.float32, device=device)
    source = ("#define PASSIVBOT_HSL_CAPACITY 1\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 0\n"
              + passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe)
    compile_shader(source, cuda_coin_capacity=coin_count,
                   mps_coin_capacity=coin_count).ema_close_quantity_probe(
        *arrays, output, threads=len(cases),
    )
    synchronize()
    np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=2e-5)
