"""Inherited override specialization preserves every replay output."""

from inspect import signature
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="GPU required",
)


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("overridden", [False, True])
def test_inherited_overrides_preserve_hsl_and_temporal_continuation(monkeypatch, mode, sides, overridden):
    from rust_utils import verify_loaded_runtime_extension
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs

    verify_loaded_runtime_extension()
    if overridden:
        import test_gpu_hsl_multicoin as fixtures
        prepare = fixtures.prepare_config

        def with_overrides(config, **kwargs):
            patch = {"entry_cooldown": {"min_duration_minutes": 0.0},
                     "strategy": {"trailing_martingale": {"entry": {"initial_qty_pct": 0.07}}}}
            if mode == "coin":
                patch["hsl"] = {"enabled": False, "drawdown_ratio_threshold": 0.03}
            # A short-side-only finite zero must retain lookup in fused replay.
            config["coin_overrides"] = {"AAA": {"bot": {sides[-1]: patch}}}
            return prepare(config, **kwargs)

        monkeypatch.setattr(fixtures, "prepare_config", with_overrides)
    proxy = make_proxy(mode, sides=sides, minutes=512, coin_count=3, factual_hsl=True)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    assert runner.coin_overrides_absent is not overridden
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
    _, actual = raw(proxy, candidates, end_steps=ends)
    loader, args = runner._library_cache_call()
    assert signature(loader).bind(*args).arguments["coin_overrides_absent"] is not overridden
    runner.coin_overrides_absent = False
    _, reference = raw(proxy, candidates, end_steps=ends)
    equal_outputs(reference, actual)
    assert actual["fill_count"].sum().item() > 0
    if not overridden:
        assert (actual["hsl_triggers_long"] + actual["hsl_triggers_short"]).sum().item() > 0
        assert runner._library_cache_call()[1] != args
        runner.coin_overrides_absent = True
        _, repeated = raw(proxy, candidates[:1])
        runner.coin_overrides_absent = False
        _, repeated_reference = raw(proxy, candidates[:1])
        equal_outputs(repeated_reference, repeated)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_inherited_overrides_preserve_unchunked_disabled_hsl(sides):
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs

    proxy = make_proxy("portfolio", sides=sides, minutes=128, coin_count=3,
                       enabled=False, factual_hsl=True)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    runner.max_dispatch_candidate_bars = None
    assert runner.coin_overrides_absent
    _, actual = raw(proxy, [{}])
    runner.coin_overrides_absent = False
    _, reference = raw(proxy, [{}])
    equal_outputs(reference, actual)
