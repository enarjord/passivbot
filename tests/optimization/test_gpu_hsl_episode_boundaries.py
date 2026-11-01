"""Executed HSL order origins survive later permission and report observations."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("finish", ["flat", "green", "censored"])
def test_panic_order_origin_survives_retirement_and_output_queries(finish):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("actual supported GPU required")
    from optimization.gpu.runtime import compile_shader, gpu_device

    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    source = "#pragma STDC FP_CONTRACT OFF\n#include <metal_stdlib>\nusing namespace metal;\n"
    source += "\n".join((root / name).read_text() for name in (
        "mps_hsl_decay.metal", "mps_hsl.metal", "mps_hsl_common.metal"))
    source += r"""
kernel void loss_report_probe(constant float* params, device float* output,
    constant int& finish, uint b [[thread_position_in_grid]]) {
    HslState h = load_hsl(params, 0, 0);
    bind_hsl(h, nullptr, nullptr, 0, 0, 0, 0, true, true);
    observe_hsl_event(h, 1000, 0, -400, true, 1, false);
    uint origin = h.panic_generation;
    record_hsl_panic_fill(h, -10, 1000, origin);
    record_hsl_panic_fill(h, -15, 975, origin);
    if (finish == 0) {
        observe_hsl_event(h, 600, -400, 0, false, 2, true);
    } else if (finish == 1) {
        observe_hsl_event(h, 975, -25, 0, true, 2, false);
    }
    HslOutputAggregate report = init_hsl_output_aggregate(0, 0);
    accumulate_hsl_output(report, h, false, 2);
    output[8] = report.panic_loss_drawdown_sum;
    output[9] = h.panic_event_loss;
    report = init_hsl_output_aggregate(0, 0);
    accumulate_hsl_output(report, h, false, 2);
    output[10] = report.panic_loss_drawdown_count;
    output[11] = h.panic_loss_drawdown_count;
    output[0] = report.panic_loss_drawdown_sum;
    output[1] = report.panic_loss_drawdown_count;
    output[2] = h.hsl.action;
    observe_hsl_event(h, 975, -25, 0, true, 3, false);
    observe_hsl_event(h, 800, -25, -400, true, 4, false);
    record_hsl_panic_fill(h, -40, 800, h.panic_generation);
    report = init_hsl_output_aggregate(0, 0);
    accumulate_hsl_output(report, h, false, 4);
    output[3] = report.panic_loss_drawdown_sum / report.panic_loss_drawdown_count;
    output[4] = report.panic_loss_drawdown_max;
    output[5] = report.panic_loss_drawdown_sum;
    output[6] = report.panic_loss_drawdown_count;
    output[7] = h.panic_fill_loss_sum_usd;
}
"""
    device = gpu_device()
    params = torch.tensor([1, .1, 1, 300000, 30.5, 2, 0], device=device)
    result = torch.zeros(12, device=device)
    compile_shader(source).loss_report_probe(params, result,
        ["flat", "green", "censored"].index(finish), threads=1)
    values = result.cpu().tolist()
    assert values[:2] == pytest.approx([.025, 1])
    assert values[2] == {"flat": 1, "green": 0, "censored": 3}[finish]
    assert values[3:8] == pytest.approx([.0375, .05, .075, 2, 65])
    assert values[8:] == pytest.approx([.025, 25, 1, 0])
