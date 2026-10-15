"""HSL reporting integrates elapsed time using the preceding RED observation."""

import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("observations, expected", [
    ([], [0, 0]),
    ([(3, 3)], [0, 0]),
    ([(0, 0), (10, 3)], [10, 0]),
    ([(2, 3), (5, 0), (8, 3)], [6, 3]),
    ([(0, 0), (1, 3), (4, 0), (7, 0)], [7, 3]),
    ([(0, 0), (0, 3), (4, 0)], [4, 4]),
    ([(3, 3), (8, 0), (20, 3)], [17, 5]),
    ([(4, 3), (2, 0), (7, 3)], [3, 0]),
])
def test_shared_hsl_elapsed_time_uses_previous_observation(strategy, observations, expected):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    # These explicit durations follow CPU Report::advance: first observation
    # adds no interval, repeated times add none, and terminal RED adds no tail.
    source = ("#define PASSIVBOT_HSL_CAPACITY 1\n#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 0\n"
              + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
              + r"""
kernel void hsl_elapsed_probe(constant float* rows, device float* output,
    constant int* count, uint b [[thread_position_in_grid]]) {
    HslTimeObservation state = init_hsl_time_observation();
    for (int i = 0; i < count[0]; ++i) {
        record_hsl_time_observation(state, rows[i * 2], int(rows[i * 2 + 1]));
    }
    output[0] = state.observed_steps;
    output[1] = state.red_steps;
    // Carrying the complete state across dispatches must preserve the clock.
    HslTimeObservation split = init_hsl_time_observation();
    int midpoint = count[0] / 2;
    for (int i = 0; i < midpoint; ++i) {
        record_hsl_time_observation(split, rows[i * 2], int(rows[i * 2 + 1]));
    }
    HslTimeObservation resumed = split;
    for (int i = midpoint; i < count[0]; ++i) {
        record_hsl_time_observation(resumed, rows[i * 2], int(rows[i * 2 + 1]));
    }
    output[2] = resumed.observed_steps;
    output[3] = resumed.red_steps;
}
""")
    device = gpu_device()
    rows = torch.tensor(observations or [(0, 0)], dtype=torch.float32, device=device)
    count = torch.tensor([len(observations)], dtype=torch.int32, device=device)
    output = torch.zeros(4, dtype=torch.float32, device=device)
    compile_shader(source, mps_coin_capacity=1, cuda_coin_capacity=1).hsl_elapsed_probe(
        rows, output, count, threads=1
    )
    assert output.cpu().tolist() == expected + expected


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_terminal_elapsed_advance_preserves_preceding_red_state(strategy):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = ("#define PASSIVBOT_HSL_CAPACITY 1\n#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 0\n"
              + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
              + r"""
kernel void terminal_elapsed_probe(device float* output, uint b [[thread_position_in_grid]]) {
    HslTimeObservation state = init_hsl_time_observation();
    record_hsl_time_observation(state, 2, 0);
    record_hsl_time_observation(state, 3, 3);
    // A terminal mark has no new tier, but its elapsed interval remains RED.
    advance_hsl_time_observation(state, 4);
    advance_hsl_time_observation(state, 4);
    output[0] = state.observed_steps;
    output[1] = state.red_steps;
    output[2] = state.was_red ? 1 : 0;
    // A terminal fill at the preceding close has no additional interval.
    HslTimeObservation at_fill = init_hsl_time_observation();
    record_hsl_time_observation(at_fill, 2, 0);
    record_hsl_time_observation(at_fill, 3, 3);
    advance_hsl_time_observation(at_fill, 3);
    output[3] = at_fill.observed_steps;
    output[4] = at_fill.red_steps;
}
""")
    device = gpu_device()
    output = torch.zeros(5, dtype=torch.float32, device=device)
    compile_shader(source, mps_coin_capacity=1, cuda_coin_capacity=1).terminal_elapsed_probe(
        output, threads=1
    )
    assert output.cpu().tolist() == [2, 1, 1, 1, 0]
