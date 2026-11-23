"""Packed GPU replay policy checks at the shader selection boundary."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd", "none"])
@pytest.mark.parametrize(
    "gate, unstuck, coin_override, feature",
    [
        pytest.param(False, 0.0, np.nan, None, id="off"),
        pytest.param(True, 0.0, np.nan, "realized-loss close sizing", id="active-zero-gate"),
        pytest.param(False, 1.0, np.nan, "unstuck exposure sizing", id="unstuck"),
        pytest.param(False, 0.0, 1.0, "unstuck exposure sizing", id="coin-enables-unstuck"),
        pytest.param(False, 1.0, 0.0, None, id="coin-disables-unstuck"),
    ],
)
def test_multicoin_dispatch_checks_effective_close_sizing(
    monkeypatch, vendor, gate, unstuck, coin_override, feature,
):
    pytest.importorskip("torch")
    from optimization.gpu import mps_kernel as kernel

    class ReachedShaderLibrary(Exception):
        pass

    def reach_shader_library(*_args):
        raise ReachedShaderLibrary

    vendor_calls = []

    def close_sizing_vendor():
        vendor_calls.append(vendor)
        return vendor

    runner = kernel._MulticoinReplayRunner.__new__(kernel._MulticoinReplayRunner)
    runner._pack_params = np.asarray
    runner._bootstrap_coin_overrides = ()
    runner._unstuck_ema_overrides = (np.asarray([[coin_override, np.nan]], dtype=np.float32),)
    runner.interval_minutes = 1
    runner.n = 3
    runner.parameter_keys = (
        "realized_loss_memory_ema_span_minutes", "loss_gate_max_drawdown_fraction",
        "unstuck_enabled", "unstuck_ema_gating_enabled",
    )
    runner.replay_sides = 1
    runner.loss_gate_enabled = gate
    runner._hourly_first_observed = runner._hourly_coin_ranges = None
    runner.requested_start_idx = 0
    runner.run_config = SimpleNamespace(warmup_bars=1)
    runner._use_disabled_hsl_specialization = lambda _: True
    runner._library_cache_call = lambda: (None, ())
    monkeypatch.setattr(kernel, "pack_bootstrap_counts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel, "validate_hourly_bootstrap_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel, "_cached_library_with_miss", reach_shader_library)
    monkeypatch.setattr(kernel, "gpu_close_sizing_vendor", close_sizing_vendor)

    candidates = np.asarray([
        [1440.0, 0.0, 0.0, 0.0],
        [1440.0, 0.0, unstuck, 0.0],
    ], dtype=np.float64)
    if feature and vendor == "none":
        with pytest.raises(ValueError, match=rf"{feature}.*CPU backend"):
            runner._run_factual_attempt(candidates)
    else:
        with pytest.raises(ReachedShaderLibrary):
            runner._run_factual_attempt(candidates)
    assert vendor_calls == ([vendor] if feature else [])


@pytest.mark.parametrize("is_hip, expected", [(False, "nvidia"), (True, "amd")])
def test_cuda_close_sizing_checks_the_shader_compiler(monkeypatch, is_hip, expected):
    from optimization.gpu import runtime

    monkeypatch.setattr(runtime, "gpu_vendor", lambda: "nvidia")
    monkeypatch.setitem(
        sys.modules, "cupy",
        SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=is_hip))),
    )
    assert runtime.gpu_close_sizing_vendor() == expected
