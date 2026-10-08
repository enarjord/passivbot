"""Coin HSL must not extend an estimated opening after all fills expire."""

from functools import lru_cache
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="GPU required",
)


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return passivbot_rust


_PROBE = r"""
kernel void empty_history_probe(
    constant float* params, constant float* options, device HslNode* trees,
    device int* rows, device float* output, uint b [[thread_position_in_grid]]
) {
    HslState h = load_hsl(params, 0, 0);
    HslState control = load_hsl(params, 0, 0);
    bind_hsl(h, trees, rows, 0, 64, 1, 10, true, true);
    bind_hsl(control, trees, rows, 1, 64, 1, 10, true, true);
    h.budget_multiplier = control.budget_multiplier = options[3];
    observe_hsl(h, 1000, 0, 0, false, 0, false);
    observe_hsl(control, 1000, 0, 0, false, 0, false);
    for (int minute = 1; minute < 15; ++minute) {
        float upnl = minute < 7 ? 0 : 400;
        observe_hsl(h, 1000, 0, upnl, true, minute, false);
        observe_hsl(control, 1000, 0, upnl, true, minute, false);
    }
    for (int i = 0; i < 4; ++i) {
        int minute = i == 3 ? 18 : (i == 0 ? 15 : 16);
        float upnl = i == 0 ? options[1] : (i == 1 ? -200 : (i == 2 ? 25 : 0));
        if (options[4] > 0) h.hsl.scalar_ready = false;
        update_coin_hsl(h, 1000, 0, upnl, options[2] > 0,
            options[0], minute - 1);
        observe_hsl(control, 1000, 0, upnl, options[2] > 0, minute, false);
        int offset = i * 10;
        output[offset] = h.hsl.action;
        output[offset + 1] = h.sampled_drawdown_raw;
        output[offset + 2] = h.drawdown_ema;
        output[offset + 3] = h.triggers;
        output[offset + 4] = h.restarts;
        output[offset + 5] = h.hsl.window.count;
        output[offset + 6] = h.hsl_valid;
        output[offset + 7] = control.hsl.action;
        output[offset + 8] = control.sampled_drawdown_raw;
        output[offset + 9] = control.drawdown_ema;
    }
}
"""


@lru_cache(maxsize=None)
def _library(strategy):
    import passivbot_rust

    source = (
        "#define PASSIVBOT_HSL_CAPACITY 64\n"
        "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 10\n"
        + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
        + _PROBE
    )
    return compile_shader(source, cuda_coin_capacity=1, mps_coin_capacity=1)


def _run(strategy, *, span=2.5, upnl=50, last_fill=1, exposed=True,
         enabled=True, mode=2, slots=2, multiplier=1, reset_cache=False):
    device = gpu_device()
    params = torch.tensor(
        [enabled, .1, span, 3, 0, mode, slots], dtype=torch.float32, device=device
    )
    options = torch.tensor(
        [last_fill, upnl, exposed, multiplier, reset_cache],
        dtype=torch.float32, device=device,
    )
    trees = torch.empty((72, 32), dtype=torch.uint8, device=device)
    rows = torch.empty(256, dtype=torch.int32, device=device)
    output = torch.empty((4, 10), dtype=torch.float32, device=device)
    _library(strategy).empty_history_probe(params, options, trees, rows, output, threads=1)
    return output.cpu().numpy()


def _rust_current(pbr, side, upnl, span, minute, multiplier):
    now = minute * 60_000
    position = dict(size=4 if side == "long" else -4, basis=100,
                    mark=100 + upnl / (4 if side == "long" else -4),
                    multiplier=1, inverse=False, pside=side, quantity_step=.001)
    anchor = dict(position_at=now, revision=0, **{
        key: position[key] for key in ("size", "basis", "multiplier", "inverse", "pside")
    })
    pair = dict(symbol="COIN", position=position, position_at=now, mark_at=now,
                fills_started_at=now, fills_at=now, prices_at=now, fills=[],
                prices={str(t * 60_000): 100 if t < 7 else 300
                        for t in range(minute - 10, minute)},
                revisions=[0] * 4, fills_position_anchor=anchor)
    pair["prices"][str(now)] = position["mark"]
    snapshot = dict(global_fill_sequence=True, fills_before_same_time_price=False,
                    now=now, start=now - 10 * 60_000, balance=1000, balance_at=now,
                    config_at=now, max_current_age_ms=0, mode="coin",
                    pside=side, symbol="COIN", pairs=[pair])
    request = dict(snapshot=snapshot, slots=2, span=span, threshold=.1,
                   cooldown_ms=180_000, restart="always")
    if multiplier != 1:
        request.update(scale_budget_with_excess_allowance=True,
                       exposure_budget=dict(wallet_exposure_limit=.5,
                                            total_wallet_exposure_limit=1,
                                            we_excess_allowance_pct=multiplier - 1))
    result = json.loads(pbr.hsl_evaluate(json.dumps(request)))
    assert "estimated_current_opening" in result["reasons"]
    assert result["observations"] == 1
    return result["decision"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("span", [1, 2.5, 308.5])
@pytest.mark.parametrize("upnl", [50, -100])
@pytest.mark.parametrize("multiplier", [1, 1.5])
def test_expired_fills_use_fresh_current_estimate(
    require_real_passivbot_rust_module, strategy, side, span, upnl, multiplier,
):
    actual = _run(strategy, span=span, upnl=upnl, multiplier=multiplier)
    # Old in-window marks still have a profitable peak, but no retained fill
    # establishes exposure across them. Only current PnL may decide protection.
    assert actual[0, 8] > .3
    for i, (minute, value) in enumerate([(15, upnl), (16, -200), (16, 25), (18, 0)]):
        expected = _rust_current(require_real_passivbot_rust_module, side, value,
                                 span, minute, multiplier)
        assert actual[i, 0] == {"normal": 0, "halted": 1, "panic": 3}[expected["action"]]
        np.testing.assert_allclose(actual[i, 1:3], [expected["raw"], expected["ema"]],
                                   rtol=2e-5, atol=2e-6)
        assert actual[i, 5:7].tolist() == [1, 1]
    # Repeated GREEN and current exposure do not manufacture terminal cooldown.
    assert actual[-1, 0] == 0
    assert actual[-1, 3] == actual[-1, 4] == 1


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("kwargs", [
    {"last_fill": 5},  # inclusive left edge at the first observation
    {"last_fill": 6}, {"last_fill": -1}, {"last_fill": float("nan")},
    {"exposed": False}, {"enabled": False}, {"mode": 0}, {"mode": 1},
])
def test_retained_unknown_flat_disabled_and_aggregate_inputs_preserve_controller(
    require_real_passivbot_rust_module, strategy, kwargs,
):
    actual = _run(strategy, **kwargs)
    # At 16 the edge fill can expire, so assess the boundary at 15 only.
    np.testing.assert_array_equal(actual[0, :3], actual[0, 7:10])


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_empty_history_is_scalar_cache_independent(
    require_real_passivbot_rust_module, strategy,
):
    cached = _run(strategy)
    rebuilt = _run(strategy, reset_cache=True)
    np.testing.assert_array_equal(cached, rebuilt)


def _integrated_inputs(strategy, sides, expired_sides):
    from tools.gpu_parity import build_parser, fixture_inputs

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2", "--bars", "1530",
        "--hsl", "coin", "--hsl-red-threshold", ".003", "--hsl-ema-span-minutes", "2.5",
    ]))
    config, candles, markets, _, _ = inputs
    active = ("long", "short") if sides == "both" else (sides,)
    config["live"]["approved_coins"] = {
        side: [f"COIN{index:02}"] if side in active else []
        for index, side in enumerate(("long", "short"))
    }
    candles[:] = [100, 100, 100, 10_000]
    for index, side in enumerate(("long", "short")):
        bot = config["bot"][side]
        if side in active:
            bot["risk"]["n_positions"] = 1
        bot["hsl"]["panic_close_order_type"] = "market"
        if strategy == "ema_anchor":
            bot["strategy"][strategy].update(
                base_qty_pct=.1, ema_span_0=10_000, ema_span_1=10_000,
                offset=.2, offset_psize_weight=0, offset_volatility_1h_weight=0,
                offset_volatility_1m_weight=0,
            )
        else:
            policy = bot["strategy"][strategy]
            policy["entry"].update(
                ema_span_0=10_000, ema_span_1=10_000, initial_ema_dist=.2,
                initial_qty_pct=.1, threshold_base_pct=.5, retracement_base_pct=.5,
                threshold_volatility_1h_weight=0, threshold_volatility_1m_weight=0,
                threshold_we_weight=0, retracement_volatility_1h_weight=0,
                retracement_volatility_1m_weight=0, retracement_we_weight=0,
            )
            policy["close"].update(
                threshold_base_pct=.9, retracement_base_pct=.5,
                threshold_volatility_1h_weight=0, threshold_volatility_1m_weight=0,
                threshold_we_weight=0, retracement_volatility_1h_weight=0,
                retracement_volatility_1m_weight=0,
            )
        # The wide candle fills actual pending initial orders. Later narrow
        # candles cannot fill an ordinary close or another entry in this recipe.
        entry = 64 if side in expired_sides else 1350
        candles[entry, index, :3] = [121, 79, 100]
        peak, current = (110, 102) if side == "long" else (90, 98)
        candles[1450:1508, index, :3] = peak
        candles[1508:, index, :3] = current
    return inputs


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides,expired_sides", [
    ("long", ("long",)), ("long", ()),
    ("short", ("short",)), ("short", ()),
    ("both", ("long", "short")), ("both", ()),
    ("both", ("long",)), ("both", ("short",)),
])
def test_native_callers_use_their_own_factual_fill_history(
    require_real_passivbot_rust_module, monkeypatch, strategy, sides, expired_sides,
):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA service required")
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import run_comparison

    active = ("long", "short") if sides == "both" else (sides,)
    from optimization.gpu.mps_kernel import MpsEmaAnchorMulticoinRunner

    observed = []
    original_run = MpsEmaAnchorMulticoinRunner.run

    def capture(self, *args, **kwargs):
        output = original_run(self, *args, **kwargs)
        observed.append(dict(
            triggers={side: float(output[f"hsl_triggers_{side}"].item())
                      for side in ("long", "short")},
            fills=float(output["fill_count"].item()),
            fused="Fused" in type(self).__name__,
        ))
        return output

    monkeypatch.setattr(MpsEmaAnchorMulticoinRunner, "run", capture)
    metrics = ("hard_stop_triggers_per_year",)
    report = run_comparison(
        _integrated_inputs(strategy, sides, expired_sides), "binance", metrics,
        {name: MetricTolerance(1e-4, 1e-5) for name in metrics},
        gpu_engine="native", diagnostics=True,
    )
    # Entry and optional protective-close fills must really occur. An empty
    # simulation or a scope that never held exposure cannot satisfy this test.
    retained = set(active) - set(expired_sides)
    assert report["diagnostics"]["cpu"]["fill_count"] == len(active) + len(retained)
    assert len(observed) == 1
    assert observed[0]["triggers"] == {
        side: float(side in retained) for side in ("long", "short")
    }
    assert observed[0]["fills"] == len(active) + len(retained)
    assert observed[0]["fused"] is (sides == "both")
    assert (report["metrics"][metrics[0]]["cpu"] > 0) is bool(retained)
    assert report["passed"], report["metrics"]
