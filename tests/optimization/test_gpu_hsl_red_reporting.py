"""Reporting includes halted cooldown without extending current panic intent."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="GPU unavailable",
)


@pytest.mark.parametrize(
    "family",
    ["ema_anchor", "trailing_martingale", "ema_anchor_multicoin", "trailing_martingale_multicoin"],
)
def test_report_red_covers_terminal_cooldown_but_panic_tier_does_not(family):
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = ("#define PASSIVBOT_HSL_CAPACITY 64\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
              + getattr(passivbot_rust, f"mps_{family}_source_py")()
              + r"""
kernel void red_reporting_probe(constant float* params, device HslNode* trees,
    device int* rows, device float* out, uint b [[thread_position_in_grid]]) {
    HslState h = load_hsl(params, 0, 0);
    bind_hsl(h, trees, rows, 0, 64, 1, 1440, true, true);
    observe_hsl(h, 1000, 0, 0, false, 0, false);
    out[0] = hsl_report_tier(h);
    observe_hsl(h, 1000, 0, -400, true, 1, false);
    out[1] = hsl_report_tier(h);
    out[2] = h.tier;
    finish_hsl_episode_at_flat(h, 600, 1000, -400, 2, 60000);
    out[3] = hsl_report_tier(h);
    out[4] = h.tier;
    out[5] = h.halted;
    observe_hsl(h, 600, -400, 0, false, 4, false);
    out[6] = hsl_report_tier(h);
    observe_hsl(h, 600, -400, 0, false, 7, false);
    out[7] = hsl_report_tier(h);
    out[8] = h.tier;
    out[9] = h.halted;
    // Disabled consumers must not contribute even with incidental state flags.
    h.enabled = false;
    h.halted = true;
    h.red_active_now = true;
    out[10] = hsl_report_tier(h);
}
""")
    device = gpu_device()
    params = torch.tensor([1, 0.1, 1, 5, 0, 2, 1], dtype=torch.float32, device=device)
    trees = torch.empty((36, 32), dtype=torch.uint8, device=device)
    rows = torch.empty(256, dtype=torch.int32, device=device)
    out = torch.zeros(11, dtype=torch.float32, device=device)
    compile_shader(source).red_reporting_probe(params, trees, rows, out, threads=1)
    assert out.cpu().tolist() == [0, 3, 3, 3, 0, 1, 3, 0, 0, 0, 0]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_cooldown_reporting_changes_only_red_samples_in_real_replay(monkeypatch, strategy, mode):
    import passivbot_rust
    from optimization.gpu import mps_kernel
    from test_gpu_hsl_multicoin import make_proxy, raw

    getter_name = f"mps_{strategy}_multicoin_source_py"
    getter = getattr(passivbot_rust, getter_name)
    library = getattr(mps_kernel, f"_{strategy}_multicoin_shader_library")
    candidate = {
        "long_hsl_red_threshold": 1e-6,
        "short_hsl_red_threshold": 1e-6,
        "hsl_red_threshold": 1e-6,
    }
    # A control restores only the old reporting expression in the otherwise
    # identical exported replay. Trading, episode state and counters stay intact.
    expression = "return h.enabled && (h.red_active_now || h.halted) ? 3 : 0;"
    source = getter()
    assert source.count(expression) == 1
    try:
        library.cache_clear()
        _, actual = raw(make_proxy(mode, strategy), [candidate])
        with monkeypatch.context() as control:
            control.setattr(passivbot_rust, getter_name,
                            lambda: source.replace(expression, "return h.tier;"))
            library.cache_clear()
            _, previous = raw(make_proxy(mode, strategy), [candidate])
        assert actual.keys() == previous.keys()
        assert (actual["hsl_tier_samples_red"] > previous["hsl_tier_samples_red"]).all()
        assert float((actual["hsl_triggers_long"] + actual["hsl_triggers_short"]).sum()) > 0
        for key in actual.keys() - {"hsl_tier_samples_red"}:
            if isinstance(actual[key], torch.Tensor):
                torch.testing.assert_close(actual[key], previous[key], rtol=0, atol=0,
                                           equal_nan=True, msg=key)
            else:
                assert actual[key] == previous[key], key
    finally:
        # Do not leave the reporting control cached for later hardware tests.
        library.cache_clear()
