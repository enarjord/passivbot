"""Inactive flat coins may skip order math, while replay behavior stays exact."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="GPU required",
)

_GUARD = "if (!selected[c] && psize[c] <= 0.0f) continue;"


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_inactive_flat_orders_preserve_scoped_hsl_and_chunk_continuation(
    monkeypatch, mode, sides,
):
    _compare_replay(monkeypatch, mode, sides, enabled=True, factual_hsl=True)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_inactive_flat_orders_preserve_disabled_hsl_and_general_replay(monkeypatch, sides):
    _compare_replay(monkeypatch, "portfolio", sides, enabled=False, factual_hsl=True)


def _compare_replay(monkeypatch, mode, sides, *, enabled, factual_hsl):
    from rust_utils import verify_loaded_runtime_extension
    from optimization.gpu import mps_kernel
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs

    verify_loaded_runtime_extension()
    # Three coins compete for one slot; held positions and flat candidates must
    # survive selection changes, order clearing and temporal state restoration.
    proxy = make_proxy(mode, sides=sides, minutes=512, coin_count=3,
                       enabled=enabled, factual_hsl=factual_hsl)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    runner.max_dispatch_candidate_bars = 3 * 3 * len(sides) * 17
    candidates = [
        {f"{side}_hsl_red_threshold": 1e-6 for side in sides},
        {f"{side}_{key}": value for side in sides for key, value in {
            "hsl_red_threshold": 0.03, "entry_cooldown_min_duration_minutes": 3.,
            "entry_cooldown_max_duration_minutes": 60.,
            "entry_cooldown_exposure_weight": 20., "entry_cooldown_adverse_weight": 30.,
            "unilateralness_ema_span_1m": 15., "forager_score_weights_unilateralness": .5,
        }.items()},
        {},
    ]
    ends = np.array([runner.n - 1, 173, 1], dtype=np.int32)
    loader = mps_kernel._trailing_martingale_multicoin_shader_library
    compile_shader = mps_kernel.compile_shader
    observed = []

    def reference_shader(source, **kwargs):
        assert source.count(_GUARD) == 1
        observed.append(True)
        return compile_shader(source.replace(_GUARD, "", 1), **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(mps_kernel, "compile_shader", reference_shader)
            loader.cache_clear()
            _, reference = raw(proxy, candidates, end_steps=ends)
        assert observed
        loader.cache_clear()
        _, actual = raw(proxy, candidates, end_steps=ends)
        equal_outputs(reference, actual)
        assert actual["fill_count"].sum().item() > 0
        if enabled:
            assert (actual["hsl_triggers_long"] + actual["hsl_triggers_short"]).sum().item() > 0
        # A new cohort must not inherit cleared orders or dormant flat fields.
        _, repeated = raw(proxy, candidates[:1])
        with monkeypatch.context() as patch:
            patch.setattr(mps_kernel, "compile_shader", reference_shader)
            loader.cache_clear()
            _, reference_repeated = raw(proxy, candidates[:1])
        equal_outputs(reference_repeated, repeated)
    finally:
        loader.cache_clear()
