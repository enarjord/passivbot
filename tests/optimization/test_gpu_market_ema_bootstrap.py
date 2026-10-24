"""Ordinary market EMA bootstrap follows the causal CPU planning boundary."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "passivbot-rust/src/gpu/mps_multicoin_common.metal"


def test_both_shared_account_strategies_bootstrap_once_at_the_usable_boundary():
    for name, prefix in (("mps_ema_anchor_multicoin_long.metal", "ema"),
                         ("mps_trailing_martingale_multicoin.metal", "tm")):
        source = (COMMON.parent / name).read_text()
        assert "bool market_ema_bootstrapped[MAX_COINS]" in source
        assert "bool hourly_ema_bootstrapped[MAX_COINS]" in source
        assert "planning_ready && k >= int(coin_settings[coin_offset + 8])" in source
        assert "if (can_bootstrap && !side.market_ema_bootstrapped[c])" in source
        assert "side.market_ema_bootstrapped[c] = true" in source
        assert source.count("k > max(global_warmup, 1) && k >= requested_start_k") >= 3
        # The existing forward suffix and independent RMS update remain explicit.
        assert "close - side.ema0[c]" in source
        assert "update_adaptive_rms(side.adaptive[c]" in source
        assert f"update_{prefix}_multicoin_side_indicators" in source


def test_risk_memories_keep_their_zero_seed_contract():
    source = COMMON.read_text()
    assert "account.unstuck_memory.drawdown = account.loss_gate_memory.drawdown = 0.0f" in source


@pytest.mark.parametrize("span,kind,k,first,values,hours,expected", [
    # Short available history: finalize 10/3, then consume 8 once -> 17/3.
    (3.0, 0, 1, 0, [2., 4., 8.], [-1., -1., -1.], 10. / 3.),
    # Span 1.5 owns ceil(3) latest observations and ignores the older outlier.
    (1.5, 0, 3, 0, [1000., 2., 4., 8.], [-1.] * 4,
     (2. * .2**2 + 4. * .2 + 8.) / (.2**2 + .2 + 1.)),
    # Listing tail excludes unavailable preceding rows.
    (3.0, 0, 3, 2, [1000., 1000., 2., 4.], [-1.] * 4, 10. / 3.),
    # Only completed hourly bucket observations participate.
    (3.0, 3, 4, 0, [1.] * 5, [-1., 2., -1., 4., -1.], 10. / 3.),
    # Original float64 count survives a span that packs to exactly 3 in float32.
    (np.nextafter(3.0, np.inf), 0, 6, 0, [1000.] + [1.] * 6, [-1.] * 7,
     8.866141732283465),
])
def test_current_metal_bootstrap_scalar_and_forward_suffix(span, kind, k, first, values, hours, expected):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("GPU unavailable")
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    # Compile only the actual authored arithmetic helper, without a native export.
    source = COMMON.read_text()
    start = source.index("inline float bootstrap_market_ema(")
    end = source.index("\n#if PASSIVBOT_RAW_STRATEGY_RISK_ENABLED", start)
    probe = "#include <metal_stdlib>\nusing namespace metal;\n" + source[start:end] + r'''
kernel void market_bootstrap_probe(constant float* bars, constant float* hours,
    constant float* inputs, constant int* counts, device float* output, uint b [[thread_position_in_grid]]) {
    if (b > 0) return;
    bool observed;
    float value = bootstrap_market_ema(bars, hours, int(inputs[0]), int(inputs[1]),
        0, 1, inputs[2], counts[0], int(inputs[3]), observed);
    output[0] = value;
    output[1] = fma(2.0f / (inputs[2] + 1.0f), 8.0f - value, value);
    output[2] = observed ? 1.0f : 0.0f;
}
'''
    device = gpu_device()
    bars = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 4, axis=1)
    output = torch.zeros(3, device=device)
    compile_shader(probe).market_bootstrap_probe(
        torch.tensor(bars, device=device), torch.tensor(hours, dtype=torch.float32, device=device),
        torch.tensor([k, first, span, kind], dtype=torch.float32, device=device),
        torch.tensor([int(np.ceil(2.0 * span))], dtype=torch.int32, device=device), output, threads=1)
    synchronize()
    assert output[0].item() == pytest.approx(expected, rel=2e-6)
    assert output[1].item() == pytest.approx(expected + 2. / (span + 1.) * (8. - expected), rel=2e-6)
    assert output[2].item() == 1.
