"""Execute the authored v9 risk graph on actual supported Metal/CUDA devices.

These risk-only witnesses do not certify full native replay integration. Source
is read directly so the tests cannot accidentally exercise a stale extension.
"""

from pathlib import Path

import numpy as np
import pytest


_PROBE = r"""
kernel void v9_scope_probe(
    constant float* params, constant float* events, constant long* timestamps,
    constant int& count, device float* output, device long* deadlines,
    uint b [[thread_position_in_grid]]
) {
    if (b != 0) return;
    HslDecayPolicy policy;
    bool valid = hsl_decay_policy(params[0], params[1], params[2], long(params[3]), policy);
    HslDecayScope scope = hsl_decay_scope_init();
    for (int i = 0; i < count; ++i) {
        int e = i * 4;
        valid = valid && hsl_decay_observe(scope, policy, timestamps[i],
            events[e], events[e + 1], events[e + 2], events[e + 3] > 0.5f);
        int o = i * 7;
        output[o] = scope.drawdown;
        output[o + 1] = scope.ratio;
        output[o + 2] = scope.ema;
        output[o + 3] = float(scope.red);
        output[o + 4] = float(hsl_decay_action(scope, timestamps[i]));
        output[o + 5] = float(valid);
        // A second query must leave the state and action unchanged.
        output[o + 6] = float(hsl_decay_action(scope, timestamps[i]));
        deadlines[i] = scope.cooldown_until_ms;
    }
}

kernel void v9_realized_probe(device float* output, uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    HslDecayPolicy policy;
    bool valid = hsl_decay_policy(3.0f, 3.0f, 0.1f, 0, policy);
    float drawdown = 0.0f;
    valid = valid && hsl_decay_realized_fill(10.0f, drawdown);
    valid = valid && hsl_decay_realized_fill(-10.0f, drawdown);
    output[0] = drawdown;
    valid = valid && hsl_decay_realized_minute(policy, drawdown);
    output[1] = drawdown;
    valid = valid && hsl_decay_realized_fill(3.0f, drawdown);
    output[2] = drawdown;
    valid = valid && hsl_decay_realized_fill(20.0f, drawdown);
    output[3] = drawdown;
    output[4] = float(valid);
}

kernel void v9_allowance_probe(
    constant float* inputs, device float* output, uint b [[thread_position_in_grid]]
) {
    const int i = int(b) * 4;
    float allowance = 7.0f;
    const bool valid = hsl_decay_auto_allowance(
        inputs[i], inputs[i + 1], inputs[i + 2], inputs[i + 3], allowance);
    output[int(b) * 2] = float(valid);
    output[int(b) * 2 + 1] = allowance;
}

kernel void v9_unknown_probe(device float* output, uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    HslDecayPolicy policy;
    HslDecayScope scope = hsl_decay_scope_init();
    bool valid = hsl_decay_policy(43200.0f, 360.0f, 0.05f, 0, policy)
        && hsl_decay_seed_unknown(scope, policy, 60000, 100.0f, -10.0f);
    output[0] = scope.drawdown;
    output[1] = scope.ratio;
    output[2] = scope.ema;
    output[3] = float(scope.red);
    output[4] = float(valid);
}

kernel void v9_duration_probe(
    constant float* limbs, device long* output, uint b [[thread_position_in_grid]]
) {
    long duration;
    bool valid = hsl_decay_duration(limbs[b * 2], limbs[b * 2 + 1], duration);
    HslDecayPolicy policy;
    HslDecayScope scope = hsl_decay_scope_init();
    valid = valid && hsl_decay_policy(3.0f, 1.0f, 0.01f, duration, policy)
        && hsl_decay_observe(scope, policy, 60000, 100.0f, -20.0f, 0.0f, true)
        && hsl_decay_observe(scope, policy, 60000, 100.0f, 0.0f, -20.0f, false);
    output[b] = valid ? scope.cooldown_until_ms : -2;
}
"""


@pytest.fixture(scope="module")
def device_program():
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual Metal or CUDA execution required")
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = (Path(__file__).resolve().parents[2]
              / "passivbot-rust/src/gpu/mps_hsl_decay.metal").read_text()
    source = ("#include <metal_stdlib>\n"
              "using namespace metal;\n" + source + _PROBE)
    return torch, compile_shader(source), gpu_device(torch)


def _observe(program, events, *, slow=3.0, fast=3.0, threshold=0.06, cooldown=30000):
    torch, library, device = program
    timestamps, rows = zip(*events)
    params = torch.tensor([slow, fast, threshold, cooldown], dtype=torch.float32, device=device)
    data = torch.tensor(rows, dtype=torch.float32, device=device)
    times = torch.tensor(timestamps, dtype=torch.int64, device=device)
    output = torch.full((len(events), 7), float("nan"), dtype=torch.float32, device=device)
    deadlines = torch.empty(len(events), dtype=torch.int64, device=device)
    library.v9_scope_probe(params, data, times, len(events), output, deadlines, threads=1)
    return output.cpu().numpy(), deadlines.cpu().numpy()


def test_pre_red_post_green_latches_one_finite_deadline(device_program):
    out, deadlines = _observe(device_program, [
        (0, [100, 0, 0, 1]),
        (60000, [100, -20, 0, 1]),
        # Realize marked PnL without double-debit, and change the current budget
        # enough to turn only the terminal sample GREEN.
        (60000, [1000, 0, -20, 0]),
        (70000, [1000000, 0, 0, 0]),
        (90000, [100, 0, 0, 0]),
    ])
    np.testing.assert_array_equal(out[:, 5], 1)
    np.testing.assert_allclose(out[:3, 0], [0, 20, 20], rtol=0, atol=0)
    np.testing.assert_array_equal(out[:, 3], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(out[:, 4], [0, 3, 1, 1, 0])
    np.testing.assert_array_equal(out[:, 6], out[:, 4])
    np.testing.assert_array_equal(deadlines, [-1, -1, 90000, 90000, 90000])


def test_ratio_smoothing_keeps_contemporaneous_budget(device_program):
    out, _ = _observe(device_program, [
        (0, [100, 0, 0, 1]),
        (60000, [100, -20, 0, 1]),
        (120000, [1000, -20, 0, 1]),
    ])
    assert out[-1, 0] == 10
    assert out[-1, 1] == pytest.approx(10 / 990, rel=2e-7)
    # The earlier R=.2 retains its earlier budget, rather than renormalizing
    # an EMA of currency drawdown by the new budget.
    assert out[-1, 2] == pytest.approx(.5 * .1 + .5 * (10 / 990), rel=2e-7)


def test_eventwise_clipping_and_same_minute_fast_replacement(device_program):
    out, _ = _observe(device_program, [
        (0, [100, 0, 10, 1]),
        (0, [90, 0, -10, 1]),
        (0, [90, 0, 0, 1]),
    ])
    np.testing.assert_array_equal(out[:, 0], [0, 10, 10])
    np.testing.assert_allclose(out[:, 2], [0, .05, .05], rtol=2e-7)


def test_new_episode_zero_initializes_and_entry_fee_counts_once(device_program):
    out, deadlines = _observe(device_program, [
        (0, [99, 0, -1, 1]),
        (0, [99, 0, 0, 0]),
        (60000, [99, 0, 0, 1]),
    ], threshold=.9)
    np.testing.assert_array_equal(out[:, 0], [1, 1, 0])
    np.testing.assert_array_equal(out[:, 5], 1)
    np.testing.assert_array_equal(deadlines, -1)


def test_missing_minutes_follow_repeated_graph(device_program):
    out, _ = _observe(device_program, [
        (0, [100, 0, 0, 1]),
        (60000, [100, -20, 0, 1]),
        (240000, [100, -20, 0, 1]),
    ])
    # Minutes 2 and 3 carry the known mark; minute 4 consumes the observation.
    assert out[-1, 0] == 2.5
    assert out[-1, 5] == 1


def test_nonpositive_equity_is_red_without_fast_warmup(device_program):
    out, _ = _observe(device_program, [(0, [100, -200, 0, 1])], fast=1000, threshold=1)
    np.testing.assert_array_equal(out[0, [1, 3, 4, 5]], [1, 1, 3, 1])


@pytest.mark.parametrize("field,value", [(0, float("nan")), (1, float("inf")), (2, float("nan"))])
def test_malformed_economics_fail_explicitly(device_program, field, value):
    event = [100, 0, 0, 1]
    event[field] = value
    out, _ = _observe(device_program, [(0, event)])
    assert out[0, 5] == 0


@pytest.mark.parametrize("slow", [float("nan"), .5, 1e30])
def test_unrepresentable_memory_policy_is_rejected(device_program, slow):
    out, _ = _observe(device_program, [(0, [100, 0, 0, 1])], slow=slow)
    assert out[0, 5] == 0


def test_realized_memory_orders_clips_and_no_fill_decay(device_program):
    torch, library, device = device_program
    output = torch.empty(5, dtype=torch.float32, device=device)
    library.v9_realized_probe(output, threads=1)
    np.testing.assert_array_equal(output.cpu().numpy(), [10, 5, 2, 0, 1])


@pytest.mark.parametrize("inputs", [
    [1000, 0, 1e20, 1e20],  # finite inputs, overflowing percentage product
    [2e38, 2e38, .5, 1],  # finite inputs, overflowing remembered peak
    [1000, 0, 1e37, 1],  # finite product/peak, overflowing fused result
])
def test_realized_allowance_overflow_is_invalid_without_a_neutral_result(device_program, inputs):
    torch, library, device = device_program
    params = torch.tensor(inputs, dtype=torch.float32, device=device)
    output = torch.empty(2, dtype=torch.float32, device=device)
    library.v9_allowance_probe(params, output, threads=1)
    np.testing.assert_array_equal(output.cpu().numpy(), [0, 7])


def test_realized_allowance_finite_graph_and_clip_are_preserved(device_program):
    torch, library, device = device_program
    params = torch.tensor([[1000, 10, .01, 2], [1000, 100, .01, 1]],
                          dtype=torch.float32, device=device)
    output = torch.empty((2, 2), dtype=torch.float32, device=device)
    library.v9_allowance_probe(params, output, threads=2)
    np.testing.assert_array_equal(output.cpu().numpy(),
                                  [[1, np.float32(10.199999809265137)], [1, 0]])


def test_unknown_exposed_prefix_has_no_synthetic_fast_warmup(device_program):
    torch, library, device = device_program
    output = torch.empty(5, dtype=torch.float32, device=device)
    library.v9_unknown_probe(output, threads=1)
    np.testing.assert_allclose(output.cpu().numpy(), [10, .1, .1, 1, 1], rtol=2e-7)


def test_zero_cooldown_permits_same_timestamp_restart(device_program):
    out, deadlines = _observe(device_program, [
        (0, [100, -20, 0, 1]),
        (0, [100, 0, -20, 0]),
    ], fast=1, cooldown=0)
    assert out[-1, 5] == 1
    assert out[-1, 4] == 0
    assert deadlines[-1] == 0


def test_clock_overflow_rejects_terminal_without_mutating_state(device_program):
    latest = np.iinfo(np.int64).max
    out, deadlines = _observe(device_program, [
        (int(latest - 10), [100, -20, 0, 1]),
        (int(latest), [100, 0, -20, 0]),
    ], fast=1)
    np.testing.assert_array_equal(out[:, 0], [20, 20])
    np.testing.assert_array_equal(out[:, 5], [1, 0])
    np.testing.assert_array_equal(out[:, 4], [3, 3])
    np.testing.assert_array_equal(deadlines, -1)


def test_exact_duration_limbs_and_fractional_boundary_deadlines(device_program):
    from optimization.gpu.hsl import pack_cooldown_ms

    torch, library, device = device_program
    durations = [59999, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 48) - 1]
    minutes = [.99999165, *(duration / 60000.0 for duration in durations[1:])]
    limbs = torch.tensor([pack_cooldown_ms(value) for value in minutes],
                         dtype=torch.float32, device=device)
    output = torch.empty(len(durations), dtype=torch.int64, device=device)
    library.v9_duration_probe(limbs, output, threads=len(durations))
    np.testing.assert_array_equal(output.cpu().numpy(), [60000 + value for value in durations])
