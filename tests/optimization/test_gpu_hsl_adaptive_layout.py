"""Current native shaders consume independent HSL/adaptive words and side strides."""

import numpy as np
import pytest

from optimization.gpu import model
from optimization.gpu.runtime import compile_shader, gpu_device

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()), reason="GPU required")


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_hsl_and_adaptive_fields_are_independent_in_native_shader(strategy):
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    prefix = strategy.upper()
    keys = getattr(model, f"{prefix}_MULTICOIN_PARAM_KEYS")
    values = dict.fromkeys(keys, 0.0)
    values.update(
        ema_span_0=2., ema_span_1=3., n_positions=3.,
        hsl_enabled=1., hsl_red_threshold=.13, hsl_drawdown_ratio_ema_span_minutes=7.,
        hsl_cooldown_ms_lo=1., hsl_cooldown_ms_hi=1.,
        hsl_memory_ema_span_minutes=1234.5, hsl_signal_mode=2.,
        unstuck_ema_span_0=11., unstuck_ema_span_1=19.,
        entry_cooldown_min_duration_minutes=2., entry_cooldown_max_duration_minutes=55.,
        entry_cooldown_exposure_weight=13., entry_cooldown_adverse_weight=17.,
        unilateralness_ema_span_1m=3.25, forager_score_weights_unilateralness=.5,
        unilateralness_window=65.,
    )
    params = np.tile(np.array([values[key] for key in keys], np.float32), 2)
    params[len(keys) + keys.index("hsl_red_threshold")] = .17
    params[len(keys) + keys.index("hsl_memory_ema_span_minutes")] = 61.5
    stem = "ema" if strategy == "ema_anchor" else "trailing_martingale"
    kind = "Ema" if strategy == "ema_anchor" else "TrailingMartingale"
    source = getattr(native, f"mps_{strategy}_multicoin_source_py")()
    probe = f"""
    kernel void packed_fields_probe(constant float* params, constant float* settings,
        constant float* overrides, device float* out, uint b [[thread_position_in_grid]]) {{
        if (b >= 2) return;
        {kind}MulticoinSideConfig config = load_{stem}_multicoin_side_config(params, int(b) * PARAM_COLS);
        {kind}MulticoinSideState side = {{}};
        init_{stem}_multicoin_side_state(side, config, settings, overrides, 1);
        AdaptiveTiming a = side.adaptive[0];
        HslState h = config.hsl_template;
        int o = int(b) * 14;
        out[o] = a.minimum; out[o+1] = a.maximum;
        out[o+2] = a.exposure_weight; out[o+3] = a.adverse_weight;
        out[o+4] = a.span; out[o+5] = float(a.window);
        out[o+6] = float(h.enabled); out[o+7] = h.red_threshold;
        out[o+8] = h.alpha; out[o+9] = float(h.decay_policy.cooldown_ms & 16777215);
        out[o+10] = float(h.signal_mode); out[o+11] = h.slot_count;
        out[o+12] = h.decay_policy.slow_alpha;
        out[o+13] = float(h.decay_policy.cooldown_ms >> 24);
    }}
    """
    pins = np.full(getattr(model, f"{prefix}_COIN_OVERRIDE_COLS"), np.nan, np.float32)
    start = getattr(model, f"{prefix}_COIN_OVERRIDE_ADAPTIVE_START")
    pins[start:start + 4] = [5., 71., 31., 37.]
    if strategy == "trailing_martingale":
        from types import SimpleNamespace
        from optimization.gpu.mps_kernel import (
            MpsTrailingMartingaleMulticoinRunner, _pack_tm_parameter_matrix,
        )
        params = _pack_tm_parameter_matrix(params.reshape(1, -1), keys, sides=2).reshape(-1)
        pins = MpsTrailingMartingaleMulticoinRunner._prepare_coin_overrides(
            SimpleNamespace(interval_minutes=1.), pins.reshape(1, -1)).reshape(-1)
    settings = np.zeros(13, np.float32)
    settings[9] = 100.
    device = gpu_device()
    output = torch.zeros((2, 14), device=device)
    library = compile_shader("#define PASSIVBOT_HSL_LOOKBACK 1\n" + source + probe,
                             mps_coin_capacity=1, cuda_coin_capacity=1)
    library.packed_fields_probe(torch.tensor(params, device=device),
        torch.tensor(settings, device=device), torch.tensor(pins, device=device), output, threads=2)
    expected = [5., 71., 31., 37., 3.25, 65., 1., .13, .25, 1., 2., 3., 2/1235.5, 1.]
    np.testing.assert_allclose(output.cpu().numpy()[0], expected, rtol=1e-6)
    expected[7], expected[12] = .17, 2/62.5
    np.testing.assert_allclose(output.cpu().numpy()[1], expected, rtol=1e-6)
