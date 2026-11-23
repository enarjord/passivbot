"""Execute the scalar close-risk shader on a host compiler and available devices."""

from ctypes import CDLL, POINTER, c_float
from pathlib import Path
import shutil
import subprocess

import pytest


_PROBE = r"""
inline ReplayCloseProjection limit_projection(
    bool is_long, float quote, float step, float minimum
) {
    ReplayCloseProjection p;
    project_replay_close(is_long, 100.0f, quote, quote, false, 0,
        0.01f, step, minimum, 0.0f, 1.0f, 0.0f, 0.0f, p);
    return p;
}

inline void close_risk_probe(device float* out, float initial_headroom, float small_loss) {
    float qty;
    ReplayCloseProjection step = limit_projection(true, 99.0f, 0.7f, 0.0f);
    size_replay_close(500.5f, 1000.0f, 0.7f, step, true, 100.0f, false, qty);
    out[0] = qty;
    ReplayCloseProjection exact = limit_projection(false, 101.0f, 0.1f, 0.0f);
    size_replay_close(500.5f, 1000.0f, 0.1f, exact, true, 100.0f, false, qty);
    out[1] = qty;
    ReplayCloseProjection minimum = limit_projection(true, 99.0f, 1.0f, 120.0f);
    size_replay_close(500.0f, 1000.0f, 1.0f, minimum, true, 100.0f, false, qty);
    out[2] = qty;
    size_replay_close(500.0f, 1000.0f, 1.0f, minimum, true, 1.0e-30f, false, qty);
    out[3] = qty;
    size_replay_close(500.0f, 1000.0f, 1.0f, minimum, true, 0.0f, false, qty);
    out[4] = qty;
    size_replay_close(500.0f, 1000.0f, 1.0f, minimum, false, 0.0f, false, qty);
    out[5] = qty;
    ReplayCloseProjection dust = limit_projection(true, 99.0f, 0.1f, 0.5f);
    size_replay_close(0.4f, 0.4f, 0.1f, dust, true, 0.01f, true, qty);
    out[6] = qty;
    size_replay_close(0.4f, 0.6f, 0.1f, dust, true, 10.0f, true, qty);
    out[7] = qty;

    ReplayCloseProjection cause;
    project_replay_close(true, 99.0f, 99.0f, 100.0f, false, 0,
        1.0f, 0.1f, 0.0f, 0.0f, 1.0f, -0.001f, 0.002f, cause);
    out[8] = replay_close_net_pnl(cause, 3.0f);
    out[9] = cause.immediately_executable;
    ReplayCloseProjection market;
    project_replay_close(true, 99.0f, 99.0f, 100.0f, true, 99,
        1.0f, 0.1f, 0.0f, 200.0f, 1.0f, -0.001f, 0.002f, market);
    out[10] = replay_close_net_pnl(market, 3.0f);
    out[11] = market.immediately_executable;
    out[12] = market.minimum_qty;

    ReplayCloseProjection long_loss = limit_projection(true, 90.0f, 0.1f, 0.0f);
    upper_replay_unstuck_qty(10.0f, 0.1f, long_loss, 1000.0f, 1.0f, 0.5f, qty);
    out[13] = qty;
    ReplayCloseProjection short_loss = limit_projection(false, 110.0f, 0.1f, 0.0f);
    size_replay_unstuck(10.0f, 0.1f, short_loss, 1000.0f, 1.0f, 0.5f,
        100.0f, false, 0.0f, qty);
    out[14] = qty;
    ReplayCloseProjection gain = limit_projection(false, 90.0f, 0.1f, 0.0f);
    size_replay_unstuck(10.0f, 0.1f, gain, 1000.0f, 1.0f, 0.5f, 100.0f, true, 0.0f, qty);
    out[15] = qty;
    size_capped_replay_unstuck(5.2f, 10.0f, 0.1f, long_loss, 1000.0f, 1.0f, 0.5f,
        100.0f, true, 30.0f, qty);
    out[16] = qty;
    ReplayCloseProjection bad_lot = limit_projection(true, 90.0f, 0.1f, 6.0f);
    size_replay_unstuck(10.0f, 0.1f, bad_lot, 1000.0f, 1.0f, 0.5f, 100.0f, false, 0.0f, qty);
    out[17] = qty;
    size_replay_unstuck(10.0f, 0.1f, long_loss, 50.0f, 1.0f, 0.5f, 100.0f, false, 0.0f, qty);
    out[18] = qty;
    size_replay_unstuck(10.0f, 0.1f, long_loss, 1000.0f, 1.0f, 1.0f, 100.0f, false, 0.0f, qty);
    out[19] = qty;
    size_replay_unstuck(10.0f, 0.1f, gain, 1000.0f, 1.0f, 0.5f, 0.0f, false, 0.0f, qty);
    out[20] = qty;

    ReplayClosePriority worst = replay_close_priority(long_loss, 90.0f, 90.0f, 0, 10, 10.0f);
    ReplayClosePriority best = replay_close_priority(long_loss, 93.0f, 93.0f, 1, 10, 10.0f);
    out[21] = replay_close_preferred(best, worst);
    out[22] = replay_close_preferred(worst, best);
    ReplayClosePriority equal = replay_close_priority(long_loss, 90.0f, 90.0f, 1, 10, 10.0f);
    out[23] = replay_close_preferred(worst, equal);
    ReplayClosePriority crossed = replay_close_priority(long_loss, 93.0f, 94.0f, 1, 10, 10.0f);
    out[24] = replay_close_preferred(worst, crossed);
    ReplayClosePriority immediate = replay_close_priority(market, 90.0f, 90.0f, 1, 10, 10.0f);
    out[25] = replay_close_preferred(immediate, best);
    ReplayLossBudget quota = {100.0f, 0.0f};
    spend_replay_loss_budget(quota, -replay_close_net_pnl(step, out[0]));
    out[26] = replay_loss_headroom(quota);
    spend_replay_loss_budget(quota, replay_close_net_pnl(step, out[0]));
    out[27] = replay_loss_headroom(quota);
    ReplayLossBudget step_left = {nextafter(100.0f, INFINITY), 0.0f};
    spend_replay_loss_budget(step_left, -100.0f);
    out[28] = replay_loss_headroom(step_left);

    ReplayCloseProjection near_flat;
    float quote = 99.9f;
    float rebate = nextafter((quote - 100.0f) / quote, 0.0f);
    project_replay_close(true, 100.0f, quote, quote, false, 0,
        0.01f, 0.1f, 0.0f, 0.0f, 1.0f, rebate, 0.0f, near_flat);
    bool increasing_loss = true;
    float previous = 0.0f;
    for (int lot = 1; lot <= 200; ++lot) {
        float loss = -replay_close_net_pnl(near_flat, float(lot) * 0.1f);
        increasing_loss = increasing_loss && loss > previous;
        previous = loss;
    }
    out[29] = increasing_loss;
    float exact_budget = -replay_close_net_pnl(near_flat, 10.0f);
    size_replay_close(20.0f, 20.0f, 0.1f, near_flat, true, exact_budget, false, qty);
    out[30] = qty;

    ReplayLossBudget accumulated = {initial_headroom, 0.0f};
    spend_replay_loss_budget(accumulated, -small_loss);
    spend_replay_loss_budget(accumulated, -small_loss);
    out[31] = replay_loss_headroom(accumulated);
    ReplayLossBudget exhausted = accumulated;
    spend_replay_loss_budget(accumulated, -nextafter(initial_headroom, 0.0f));
    out[32] = replay_loss_headroom(accumulated);
    spend_replay_loss_budget(exhausted, -small_loss);
    spend_replay_loss_budget(exhausted, -nextafter(initial_headroom, 0.0f));
    out[33] = replay_loss_headroom(exhausted);
    size_replay_close(500.0f, 1000.0f, 1.0f, minimum, true,
        replay_loss_headroom(exhausted), false, qty);
    out[34] = qty;

    float aligned_quote = float(10030) * 0.01f;
    float aligned_touch = replay_close_priority_touch(100.3f, 10030, 10030, 0.01f);
    float off_tick_touch = replay_close_priority_touch(100.304f, 10030, 10031, 0.01f);
    out[35] = aligned_touch == aligned_quote;
    out[36] = off_tick_touch == 100.304f && off_tick_touch != aligned_quote;
}
"""

_EXPECTED = (
    100.1, 100.0, 120.0, 120.0, 0.0, 500.0, 0.4, 0.0,
    0.297, 0.0, -0.594, 1.0, 2.0,
    5.2, 5.2, 4.7, 3.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 1.0, 1.0, 1.0,
    100.0, 0.0, 2 ** -17, 1.0, 10.0,
    100.0, 2 ** -17 - 6e-6, 0.0, 0.0,
    1.0, 1.0,
)


def _probe_source():
    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    shader = (root / "mps_multicoin_common.metal").read_text()
    fill_start = shader.index("inline float ordinary_market_fill_price(")
    fill_end = shader.index("inline float resize_market_close_qty(")
    risk_start = shader.index("inline bool finite_positive(")
    risk_end = shader.index("inline int multicoin_interval_minutes(")
    return shader[fill_start:fill_end] + shader[risk_start:risk_end] + _PROBE


def test_scalar_close_risk_shader_with_host_compiler(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required")
    source = tmp_path / "close_probe.cpp"
    library = tmp_path / "close_probe.so"
    source.write_text(
        """
#include <cmath>
#include <cstring>
using namespace std;
using uint = unsigned int;
template <typename To, typename From> To as_type(From value) {
    static_assert(sizeof(To) == sizeof(From));
    To result;
    memcpy(&result, &value, sizeof(result));
    return result;
}
"""
        + _probe_source().replace("thread ", "").replace("device ", "")
        + """
extern "C" void run_close_probe(float* out, float initial_headroom, float small_loss) {
    close_risk_probe(out, initial_headroom, small_loss);
}
"""
    )
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
         str(source), "-o", str(library)],
        capture_output=True,
        text=True,
        check=True,
    )
    run = CDLL(str(library)).run_close_probe
    run.restype = None
    run.argtypes = [POINTER(c_float), c_float, c_float]
    result = (c_float * len(_EXPECTED))()
    run(result, 100.0, 3e-6)
    assert list(result) == pytest.approx(_EXPECTED, rel=2e-6, abs=1e-7)


def test_scalar_close_risk_shader_on_device():
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("actual supported GPU required")
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = "#include <metal_stdlib>\nusing namespace metal;\n" + _probe_source() + """
kernel void run_close_probe(
    constant float* input, device float* out, uint index [[thread_position_in_grid]]
) {
    if (index == 0) close_risk_probe(out, input[0], input[1]);
}
"""
    device = gpu_device(torch)
    inputs = torch.tensor([100.0, 3e-6], device=device)
    result = torch.zeros(len(_EXPECTED), device=device)
    compile_shader(source).run_close_probe(inputs, result, threads=1)
    assert result.cpu().tolist() == pytest.approx(_EXPECTED, rel=2e-6, abs=1e-7)
