"""Per-side fixed score margins preserve replay ABI and actual selection behavior."""
from copy import deepcopy

import numpy as np
import pytest

from test_side_minimum_cost_gpu import cpu_data


@pytest.mark.parametrize("kind", ["EmaAnchor", "TrailingMartingale"])
@pytest.mark.parametrize("margins", [{"long": 0.0, "short": 0.0},
                                    {"long": 0.25, "short": 0.25},
                                    {"long": 0.001, "short": 0.5},
                                    {"long": 0.5, "short": 0.001}])
def test_directional_and_fused_hysteresis_slots_preserve_other_policy(monkeypatch, kind, margins):
    kernel, run, data = cpu_data(monkeypatch)
    activation = {"long": True, "short": False}
    costs = {"long": False, "short": True}
    for side in ("long", "short"):
        runner = getattr(kernel, f"Mps{kind}MulticoinRunner")(
            run, data, side=side, filter_by_min_effective_cost=costs,
            side_activation=activation, forager_score_hysteresis=margins)
        assert len(runner.settings) == 16
        assert runner.settings[4].item() == pytest.approx(margins[side])
        assert runner.settings[11].item() == costs[side]
        assert runner.settings[15].item() == activation[side]
    fused = getattr(kernel, f"Mps{kind}MulticoinFusedRunner")(
        run, data, filter_by_min_effective_cost=costs,
        side_activation=activation, forager_score_hysteresis=margins)
    assert len(fused.settings) == 21
    assert fused.settings[4].item() == pytest.approx(margins["long"])
    assert fused.settings[20].item() == pytest.approx(margins["short"])
    assert fused.settings[13].item() == costs["long"]
    assert fused.settings[17].item() == costs["short"]
    assert fused.settings[18:20].tolist() == [1.0, 0.0]
    assert data["coin_settings"].shape[1] == 15
    fused.settings[14], fused.settings[15] = 7.0, 1.0
    assert fused.settings[4].item() == pytest.approx(margins["long"])
    assert fused.settings[20].item() == pytest.approx(margins["short"])


@pytest.mark.parametrize("invalid", [None, True, {}, {"long": 0.0},
                                      {"long": True, "short": 0.0},
                                      {"long": 0.0, "short": float("inf")}])
def test_runner_requires_complete_finite_numeric_pair(monkeypatch, invalid):
    kernel, run, data = cpu_data(monkeypatch)
    with pytest.raises((TypeError, ValueError), match="forager_score_hysteresis"):
        kernel.MpsEmaAnchorMulticoinFusedRunner(
            run, data, filter_by_min_effective_cost={"long": False, "short": False},
            side_activation={"long": True, "short": True}, forager_score_hysteresis=invalid)


def test_both_native_shaders_and_cuda_lowering_keep_independent_margin_slots():
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension
    from optimization.gpu.cuda_kernel import cuda_source

    verify_loaded_runtime_extension()
    for source in (native.mps_ema_anchor_multicoin_source_py(),
                   native.mps_trailing_martingale_multicoin_source_py()):
        for code in (source, cuda_source(source)):
            assert "long_score_hysteresis = fmax(run_settings[4]" in code
            assert "short_score_hysteresis = fmax(run_settings[20]" in code
            assert "long_effective_n_positions, long_score_hysteresis" in code
            assert "short_effective_n_positions, short_score_hysteresis" in code
            assert "long_activation = run_settings[18]" in code
            assert "short_activation = run_settings[19]" in code



@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused", "oneway"])
@pytest.mark.parametrize("profile", ["equal", "unequal"])
def test_actual_metal_hysteresis_matches_native_selection_and_counts(kind, topology, profile):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("local Metal device unavailable")
    from backtest import build_backtest_payload, execute_backtest
    from optimization.gpu.service import SharedAccountReplay
    from rust_utils import verify_loaded_runtime_extension
    from test_side_forager_hysteresis import replay_inputs

    verify_loaded_runtime_extension()
    high_side = "short" if topology in ("short", "oneway") else "long"
    margins = {side: 0.5 if profile == "unequal" and side == high_side else 0.001
               for side in ("long", "short")}
    config, candles, markets, btc, timestamps = replay_inputs(kind, topology, margins)
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    fills, _, native = execute_backtest(payload, config)
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
                                timestamps=timestamps, exchange="binance", batch_size=1,
                                needed_metrics={"fills_per_day"})
    if topology in ("long", "short"):
        runner = proxy.runners[topology]
        matrix = proxy._parameter_matrix([{}], topology)
    else:
        cls, options = proxy._runner_specs["fused"]
        runner = cls(proxy.run, proxy.data, **options)
        matrix = np.concatenate([proxy._parameter_matrix([{}], side)
                                 for side in ("long", "short")], axis=1)
    raw = runner.run(matrix)
    assert raw["fill_count_long"].item() == native["fills_count_long"]
    assert raw["fill_count"].item() - raw["fill_count_long"].item() == native["fills_count_short"]
    if profile == "unequal":
        assert native["fills_count_" + high_side] == 0
    if len(fills):
        assert raw["first_fill_ts"].item() == int(fills[0][1]) - int(timestamps[0])
    if profile == "equal" or topology in ("fused", "oneway"):
        assert len(fills) > 0


@pytest.mark.parametrize("side", ["long", "short"])
def test_replay_checkpoint_policy_changes_for_either_fixed_margin(side):
    from optimization.gpu.service import _gpu_proxy_execution_checkpoint_contract

    policy = {"long": 0.02, "short": 0.03}
    params = {"coins": ["AAA"], "forager_score_hysteresis": policy,
              "side_activation": {"long": True, "short": True}}
    options = dict(strategy_kind="ema_anchor", exchange="binance", enabled_sides=("long", "short"),
                   hlcvs=np.ones((3, 1, 4)), timestamps=np.arange(3, dtype=np.int64),
                   backtest_params=params, exchange_params=[], base_params={})
    initial = _gpu_proxy_execution_checkpoint_contract(**options)
    assert initial["forager_hysteresis_abi"] == 1
    assert initial["backtest"]["forager_score_hysteresis"] == {"long": 0.02, "short": 0.03}
    policy[side] += 0.01
    changed = _gpu_proxy_execution_checkpoint_contract(**options)
    assert changed != initial
    assert initial["backtest"]["forager_score_hysteresis"][side] != policy[side]


@pytest.mark.parametrize("field", ["max_realized_loss_pct", "market_order_near_touch_threshold"])
def test_explicit_neutral_pair_preserves_other_runner_validation(monkeypatch, field):
    kernel, run, data = cpu_data(monkeypatch)
    with pytest.raises(ValueError, match=field):
        kernel.MpsEmaAnchorMulticoinRunner(
            run, data, side="long", forager_score_hysteresis={"long": 0.0, "short": 0.0},
            side_activation={"long": True, "short": True},
            filter_by_min_effective_cost={"long": False, "short": False},
            **{field: float("nan")})


def test_runner_hysteresis_pair_cannot_be_omitted(monkeypatch):
    kernel, run, data = cpu_data(monkeypatch)
    with pytest.raises(TypeError, match="forager_score_hysteresis"):
        kernel.MpsEmaAnchorMulticoinFusedRunner(
            run, data, side_activation={"long": True, "short": True},
            filter_by_min_effective_cost={"long": False, "short": False})
