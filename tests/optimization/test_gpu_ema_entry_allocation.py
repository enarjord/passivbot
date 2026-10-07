"""EMA WEL sizes clips; portfolio TWEL and HSL govern entry admission."""

import numpy as np
import pytest


def test_ema_clips_cross_coin_allocation_without_bypassing_portfolio_gates():
    torch = pytest.importorskip("torch")
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU required")
    import passivbot_rust

    verify_loaded_runtime_extension()
    probe = r"""
kernel void ema_entry_allocation_probe(
    constant float* cases,
    constant float* bars,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    bool short_side = cases[b * 4] > 0.5f;
    EmaMulticoinSideConfig config = {};
    config.base_qty_pct = 0.02f;
    config.twel = 1.0f;
    config.n_positions = 2;
    config.twel_entry_gate_enabled = cases[b * 4 + 2] > 0.5f;
    config.twel_threshold = 1.0f;
    EmaMulticoinSideState side;
    init_ema_multicoin_side_state(side, config, coin_settings, coin_overrides, 2);
    side.psize[0] = cases[b * 4 + 1];
    side.pprice[0] = 100.0f;
    side.selected[0] = true;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    generate_ema_multicoin_side_orders(
        side, config, account, bars, touch_ticks, coin_settings, coin_overrides,
        0, 2, short_side, 2, 2, int(cases[b * 4 + 3]), false, 0.0f, -2, 0ul
    );
    output[b] = side.entry_qty[0];
}
"""
    cases = np.array([
        (side, size, gate, mode)
        for side in (0, 1)
        for size, gate, mode in (
            (4.99, 1, 0), (5.0, 1, 0), (9.95, 1, 0),
            (10.0, 1, 0), (12.0, 0, 0), (5.0, 1, 2),
        )
    ], dtype=np.float32)
    settings = np.array([
        [.001, .01, .001, 1, 1, 0, 0, 10, 0, 100, 100, 0, 0],
        [.001, .01, .001, 1, 1, 0, 0, 10, 0, 100, 100, 0, 0],
    ], dtype=np.float32)
    device = gpu_device()
    arrays = [
        torch.tensor(value, device=device) for value in (
            cases, np.array([[100, 100, 100, 100]] * 2, dtype=np.float32),
            np.array([[10000, 10000]] * 2, dtype=np.int32), settings,
            np.full((2, 32), np.nan, dtype=np.float32),
        )
    ]
    output = torch.empty(len(cases), dtype=torch.float32, device=device)
    source = ("#define PASSIVBOT_HSL_CAPACITY 1\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 0\n"
              + passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe)
    compile_shader(source, cuda_coin_capacity=2, mps_coin_capacity=2).ema_entry_allocation_probe(
        *arrays, output, threads=len(cases),
    )
    synchronize()
    for (_, size, gate, mode), qty in zip(cases, output.cpu().tolist()):
        if mode == 2 or gate and size >= 10:
            assert qty == 0
        elif gate and size == pytest.approx(9.95):
            assert .04 < qty <= .05  # The portfolio gate clips the final order.
        else:
            # Canonical Rust calc_entry_qty: 1000 * (1 / 2) * .02 / 100.
            assert qty == pytest.approx(.1, abs=1e-6)


@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_native_ema_shock_replay_matches_cpu_after_crossing_coin_allocation(sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", sides, "--coins", "2",
        "--bars", "3000", "--seed", "43", "--hsl", "disabled",
    ]))
    inputs[1][1500:, 0, :3] *= .7
    inputs[1][1800:, 1, :3] *= 1.3
    metrics = ["drawdown_worst_strategy_eq", "fills_per_day", "adg_usd"]
    policies = {name: MetricTolerance(1e-6, 1e-4) for name in metrics}
    # The measured shared-case float32 growth gap is 7.52e-6. Keep this
    # fixture-local growth bound and all drawdown checks strict.
    policies["adg_usd"] = MetricTolerance(1e-5, 0)
    if sides == "both":
        # At step 1453, balances 1010.691403/1010.686707 straddle the
        # nearest-step clip boundary: .050 CPU versus .049 GPU. This changes
        # two of 2395 fills (0.0836%); drawdown differs by only 2.23e-7.
        policies["fills_per_day"] = MetricTolerance(0, 1e-3)
    report = run_comparison(
        inputs, "binance", metrics,
        policies,
        gpu_engine="native", diagnostics=True,
    )
    assert report["diagnostics"]["cpu"]["fill_count"] > 0
    assert report["passed"], report["metrics"]
