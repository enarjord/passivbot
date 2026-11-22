"""Single GPU realized-loss memory transport and arithmetic."""

from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import pytest

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
)


@pytest.mark.parametrize("keys", [EMA_ANCHOR_MULTICOIN_PARAM_KEYS, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS])
def test_shared_memory_span_is_candidate_owned_and_side_coherent(keys):
    pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _validate_global_memory_params
    from optimization.gpu.service import _candidate_parameter_matrix

    base = {key: 1.0 for key in keys}
    matrix = _candidate_parameter_matrix(
        [{"realized_loss_memory_ema_span_minutes": 3.0},
         {"realized_loss_memory_ema_span_minutes": 15.0}],
        keys, {"long": base, "short": base},
    )
    width = len(keys)
    assert matrix[:, keys.index("realized_loss_memory_ema_span_minutes")].tolist() == [3.0, 15.0]
    assert np.all(matrix[:, keys.index("reserved_realized_memory_slot")] == 0.0)
    _validate_global_memory_params(matrix, keys, sides=2)
    matrix[0, width + keys.index("realized_loss_memory_ema_span_minutes")] = 8.0
    with pytest.raises(ValueError, match="account-global"):
        _validate_global_memory_params(matrix, keys, sides=2)


def test_side_hsl_memory_rates_are_independent_but_portfolio_policy_is_coherent():
    pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _validate_hsl_params

    keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    width = len(keys)
    matrix = np.zeros((1, 2 * width))
    for side in range(2):
        for key, value in {"hsl_enabled": 1, "hsl_drawdown_threshold_fraction": .1,
                           "hsl_drawdown_fraction_ema_span_minutes": 3,
                           "hsl_memory_ema_span_minutes": 3 if side == 0 else 7,
                           "hsl_signal_mode": 1}.items():
            matrix[0, side * width + keys.index(key)] = value
    runner = SimpleNamespace(long_enabled=True, short_enabled=True, interval_minutes=1)
    _validate_hsl_params(runner, matrix, keys)
    matrix[:, [keys.index("hsl_signal_mode"), width + keys.index("hsl_signal_mode")]] = 0
    with pytest.raises(ValueError, match="one shared policy"):
        _validate_hsl_params(runner, matrix, keys)


_PROBE = r"""
kernel void shared_realized_event_probe(
    constant float* settings, constant float* events, device float* output,
    uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    JointPortfolioAccount account = init_joint_portfolio_account(settings[3]);
    configure_joint_realized_memory(account, settings[0], settings[1] > 0.5f, int(settings[2]));
    for (int event = 0; event < 8; ++event) {
        advance_joint_realized_memory(account, int(events[event * 2]));
        record_joint_portfolio_fill(account, events[event * 2 + 1], event % 2 == 0);
        float allowance = 0;
        bool valid = joint_realized_loss_allowance(account, settings[4], allowance);
        output[event * 5] = account.realized_memory.drawdown;
        output[event * 5 + 1] = account.balance;
        output[event * 5 + 2] = account.realized_pnl_total;
        output[event * 5 + 3] = allowance;
        output[event * 5 + 4] = float(valid && account.realized_decay_valid);
    }
}
"""


@pytest.fixture(scope="module", params=["ema", "tm"])
def shared_memory_program(request):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual supported Metal/CUDA execution required")
    from optimization.gpu.runtime import compile_shader, gpu_device
    from test_gpu_hsl_adapter import _native_source

    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    source = _native_source(root, request.param)
    device = gpu_device(torch)
    if destination := os.environ.get("PASSIVBOT_V9_EVIDENCE_DIR"):
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{request.param}-compiled-source.metal").write_text(source + _PROBE)
        (directory / f"{request.param}-device.json").write_text(json.dumps({
            "backend": device, "torch_version": str(torch.__version__),
            "dtype": "float32", "coin_capacity": 2,
            "device_name": torch.cuda.get_device_name() if device == "cuda" else "Apple MPS",
        }, indent=2) + "\n")
    return torch, compile_shader(source + _PROBE, mps_coin_capacity=2), device


@pytest.mark.parametrize("span", [3.0, 7.0])
@pytest.mark.parametrize("memory_enabled", [False, True])
@pytest.mark.parametrize("interval", [1, 5])
def test_shared_realized_memory_debits_canonical_net_fills_once(
    shared_memory_program, span, memory_enabled, interval,
):
    from v9_risk_reference import add, allowance, coefficients, f32, fma, sub

    torch, library, device = shared_memory_program
    events = [(0, -80.0), (0, 100.0), (0, -20.0), (1, -.25),
              (1, .125), (2, 0.0), (4, -2.0), (4, 3.0)]
    settings = torch.tensor([span, memory_enabled, interval, 1000.0, .1],
                            dtype=torch.float32, device=device)
    packed = torch.tensor(events, dtype=torch.float32, device=device)
    output = torch.full((8, 5), float("nan"), dtype=torch.float32, device=device)
    library.shared_realized_event_probe(settings, packed, output, threads=1)
    drawdown = 0.0
    beta = coefficients(span)[1]
    expected, previous_minute, balance, net_total = [], 0, 1000.0, 0.0
    for bar, net in events:
        minute = bar * interval
        if memory_enabled:
            for _ in range(previous_minute, minute):
                drawdown = fma(beta, drawdown, 0.0)
            drawdown = max(0.0, sub(drawdown, f32(net)))
        previous_minute = minute
        balance, net_total = add(balance, f32(net)), add(net_total, f32(net))
        expected.append([drawdown, balance, net_total, allowance(balance, .1, drawdown), 1.0])
    assert output.cpu().tolist() == expected
