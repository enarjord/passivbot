"""V9 panic-loss reporting stays observational across flat/GREEN/open endings."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("finish", ["flat", "green", "censored"])
def test_panic_loss_report_finishes_segments_without_retaining_panic(finish):
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
    record_hsl_panic_fill(h, -10, 1000);
    record_hsl_panic_fill(h, -15, 975);
    if (finish == 0) {
        observe_hsl_event(h, 600, -400, 0, false, 2, true);
    } else if (finish == 1) {
        observe_hsl_event(h, 975, -25, 0, true, 2, false);
    } else {
        HslOutputAggregate report = init_hsl_output_aggregate(0, 0);
        accumulate_hsl_output(report, h, false, 2);
        output[8] = report.panic_loss_drawdown_sum;
        output[9] = h.panic_event_loss;
        report = init_hsl_output_aggregate(0, 0);
        accumulate_hsl_output(report, h, false, 2);
        output[10] = report.panic_loss_drawdown_count;
        output[11] = h.panic_loss_drawdown_count;
        finish_hsl_panic_loss(h);
    }
    output[0] = h.panic_loss_drawdown_sum;
    output[1] = h.panic_loss_drawdown_count;
    output[2] = h.hsl.action;
    finish_hsl_panic_loss(h);
    record_hsl_panic_fill(h, -40, 800);
    finish_hsl_panic_loss(h);
    output[3] = h.panic_loss_drawdown_min;
    output[4] = h.panic_loss_drawdown_max;
    output[5] = h.panic_loss_drawdown_sum;
    output[6] = h.panic_loss_drawdown_count;
    output[7] = h.panic_close_loss_sum;
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
    assert values[3:8] == pytest.approx([.025, .05, .075, 2, 65])
    if finish == "censored":
        assert values[8:] == pytest.approx([.025, 25, 1, 0])
