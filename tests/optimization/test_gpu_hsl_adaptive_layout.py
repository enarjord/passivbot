"""Native-exported shaders consume the merged HSL/adaptive packing on hardware."""

import numpy as np
import pytest

from optimization.gpu import model
from optimization.gpu.runtime import compile_shader, gpu_device, synchronize

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("multicoin", [False, True])
def test_hsl_and_adaptive_fields_are_independent_in_native_shader(strategy, multicoin):
    import passivbot_rust as native

    prefix = strategy.upper()
    keys = getattr(
        model, f"{prefix}_{'MULTICOIN' if multicoin else 'SINGLE_COIN'}_PARAM_KEYS"
    )
    values = {key: 0.0 for key in keys}
    values.update(
        ema_span_0=2.0,
        ema_span_1=3.0,
        hsl_enabled=1.0,
        hsl_red_threshold=0.13,
        hsl_ema_span_minutes=7.0,
        hsl_cooldown_minutes_after_red=29.0,
        hsl_restart_policy=1.0,
        hsl_signal_mode=2.0,
        hsl_slot_count=3.0,
        unstuck_ema_span_0=11.0,
        unstuck_ema_span_1=19.0,
        entry_cooldown_min_duration_minutes=2.0,
        entry_cooldown_max_duration_minutes=55.0,
        entry_cooldown_exposure_weight=13.0,
        entry_cooldown_adverse_weight=17.0,
        unilateralness_ema_span_1m=3.25,
        forager_score_weights_unilateralness=0.5,
        unilateralness_window=65.0,
    )
    params = np.array([values[key] for key in keys], np.float32)
    params = np.tile(params, 2)
    # A second side with different values catches wrong directional strides.
    params[len(keys) + keys.index("entry_cooldown_exposure_weight")] = 23.0
    if multicoin:
        source = getattr(native, f"mps_{strategy}_multicoin_source_py")()
        stem = "ema" if strategy == "ema_anchor" else "trailing_martingale"
        kind = "Ema" if strategy == "ema_anchor" else "TrailingMartingale"
        setup = f"""
        {kind}MulticoinSideConfig config = load_{stem}_multicoin_side_config(params, int(b) * PARAM_COLS);
        {kind}MulticoinSideState side = {{}};
        init_{stem}_multicoin_side_state(side, config, settings, overrides, 1);
        AdaptiveTiming a = side.adaptive[0];
        HslState h = config.hsl_template;
        """
        count = getattr(model, f"{prefix}_COIN_OVERRIDE_COLS")
        pins = np.full(count, np.nan, np.float32)
        start = getattr(model, f"{prefix}_COIN_OVERRIDE_ADAPTIVE_START")
        pins[start : start + 4] = [5.0, 71.0, 31.0, 37.0]
        expected_prefix = [5.0, 71.0, 31.0, 37.0, 3.25, 65.0]
    else:
        source = getattr(native, f"mps_{strategy}_source_py")()
        kind = "EmaSide" if strategy == "ema_anchor" else "TmSide"
        setup = f"""
        {kind} side = load_side(params, int(b) * SIDE_PARAMS, 100.0f);
        AdaptiveTiming a = side.adaptive;
        HslState h = load_hsl(params, int(b) * SIDE_PARAMS, {keys.index('hsl_enabled')});
        """
        pins = np.zeros(1, np.float32)
        expected_prefix = [2.0, 55.0, 13.0, 17.0, 3.25, 65.0]
    probe = f"""
    kernel void packed_fields_probe(constant float* params, constant float* settings,
        constant float* overrides, device float* out, uint b [[thread_position_in_grid]]) {{
        if (b >= 2) return;
        {setup}
        int o = int(b) * 12;
        out[o] = a.minimum; out[o+1] = a.maximum;
        out[o+2] = a.exposure_weight; out[o+3] = a.adverse_weight;
        out[o+4] = a.span; out[o+5] = float(a.window);
        out[o+6] = h.enabled ? 1.0f : 0.0f; out[o+7] = h.red_threshold;
        out[o+8] = h.alpha; out[o+9] = h.cooldown_minutes;
        out[o+10] = float(h.signal_mode); out[o+11] = h.slot_count;
    }}
    """
    source = (
        "#define PASSIVBOT_HSL_CAPACITY 1\n#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 0\n" + source + probe
    )
    library = compile_shader(
        source,
        mps_coin_capacity=1 if multicoin else None,
        cuda_coin_capacity=1 if multicoin else None,
    )
    device = gpu_device()
    settings = np.zeros(13, np.float32)
    settings[9] = 100.0
    output = torch.zeros((2, 12), device=device)
    library.packed_fields_probe(
        torch.tensor(params, device=device),
        torch.tensor(settings, device=device),
        torch.tensor(pins, device=device),
        output,
        threads=2,
    )
    synchronize()
    expected = expected_prefix + [1.0, 0.13, 0.25, 29.0, 2.0, 3.0]
    np.testing.assert_allclose(output.cpu().numpy()[0], expected, rtol=1e-6)
    if not multicoin:
        expected[2] = 23.0
    np.testing.assert_allclose(output.cpu().numpy()[1], expected, rtol=1e-6)
