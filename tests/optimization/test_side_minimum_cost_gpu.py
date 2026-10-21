"""GPU policy transport is directional; fused factual slots remain independent."""
from itertools import product

import numpy as np
import pytest


def cpu_data(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1 << 30)
    import optimization.gpu.runtime as runtime
    import optimization.gpu.mps_kernel as kernel
    from optimization.gpu.model import ProxyRun, ProxyMarket, build_mps_multicoin_data

    monkeypatch.setattr(runtime, "gpu_device", lambda *args: "cpu")
    monkeypatch.setattr(kernel, "gpu_device", lambda *args: "cpu")
    timestamps = np.arange(32, dtype=np.int64) * 60_000 + 1_704_067_200_000
    candles = np.empty((32, 1, 4), dtype=np.float64)
    candles[:] = [101.0, 99.0, 100.0, 1000.0]
    run = ProxyRun(1000.0, 1, 1, int(timestamps[1]), int(timestamps[1]), int(timestamps[0]), 60_000, 0.05, 0, 31)
    data = build_mps_multicoin_data(candles, timestamps, [run], [ProxyMarket(0.001, 0.01, 0.001, 50.0, 1.0, 0.0)])
    return kernel, run, data


@pytest.mark.parametrize("kind", ["EmaAnchor", "TrailingMartingale"])
@pytest.mark.parametrize("long,short", product((False, True), repeat=2))
def test_directional_and_fused_settings_keep_both_side_flags(monkeypatch, kind, long, short):
    kernel, run, data = cpu_data(monkeypatch)
    flags = {"long": long, "short": short}
    directional = getattr(kernel, f"Mps{kind}MulticoinRunner")
    for side in ("long", "short"):
        runner = directional(run, data, side=side, filter_by_min_effective_cost=flags)
        assert runner.settings[11].item() == flags[side]
        assert len(runner.settings) == 15
    fused = getattr(kernel, f"Mps{kind}MulticoinFusedRunner")(run, data, filter_by_min_effective_cost=flags)
    assert len(fused.settings) == 18
    assert fused.settings[13].item() == long
    assert fused.settings[17].item() == short
    assert fused.settings[14:16].tolist() == [0.0, 0.0]
    # Mutating factual capacity/control must not overwrite either policy flag.
    fused.settings[14], fused.settings[15] = 7, 1
    assert fused.settings[13].item() == long and fused.settings[17].item() == short


@pytest.mark.parametrize("kind", ["EmaAnchor", "TrailingMartingale"])
@pytest.mark.parametrize("invalid", [True, None, {}, {"long": True, "short": 1}])
def test_runner_rejects_malformed_policy_pair(monkeypatch, kind, invalid):
    kernel, run, data = cpu_data(monkeypatch)
    with pytest.raises(TypeError):
        getattr(kernel, f"Mps{kind}MulticoinFusedRunner")(run, data, filter_by_min_effective_cost=invalid)


def test_both_shader_and_cuda_translation_keep_independent_arbitration_inputs():
    import passivbot_rust as pbr
    from optimization.gpu.cuda_kernel import cuda_source
    for source in (pbr.mps_ema_anchor_multicoin_source_py(), pbr.mps_trailing_martingale_multicoin_source_py()):
        for translated in (source, cuda_source(source)):
            assert "filter_by_min_effective_cost_long = run_settings[13]" in translated
            assert "filter_by_min_effective_cost_short = run_settings[17]" in translated
            assert "filter_by_min_effective_cost_long, filter_by_min_effective_cost_short, account.balance" in translated
            assert "filter_by_min_effective_cost_long, balance" in translated
            assert "filter_by_min_effective_cost_short, balance" in translated


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("passing_side", ["long", "short"])
@pytest.mark.parametrize("topology", ["long", "short", "fused", "oneway"])
def test_actual_gpu_asymmetric_replay_matches_native_counts(monkeypatch, kind, passing_side, topology):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("local Metal or CUDA device unavailable")
    from backtest import build_backtest_payload, execute_backtest
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import build_parser, fixture_inputs

    sides = topology if topology in {"long", "short"} else "both"
    args = build_parser().parse_args(["--fixture", kind, "--hsl", "disabled", "--sides", sides,
                                      "--coins", "2", "--bars", "180", "--seed", "2026"])
    config, candles, markets, btc, timestamps = fixture_inputs(args)
    config["backtest"]["filter_by_min_effective_cost"] = None
    config["bot"]["global"]["hedge_mode"] = topology != "oneway"
    for side in ("long", "short"):
        bot = config["bot"][side]
        bot["universe"]["filter_by_min_effective_cost"] = side != passing_side
        bot["risk"]["max_positions"] = 1 if sides == "both" or side == sides else 0
        bot["entry_cooldown"]["base_duration_minutes"] = 1000.0
        strategy = bot["strategy"][kind]
        strategy["ema_span_0_minutes"], strategy["ema_span_1_minutes"] = 2.0, 3.0
        if kind == "ema_anchor":
            strategy.update(base_qty_fraction=0.001, offset_fraction=0.0, entry_double_down_factor=0.0)
        else:
            strategy["entry"].update(initial_qty_fraction=0.001, initial_ema_distance_fraction=0.0, double_down_factor=0.0)
    for coin, market in markets.items():
        if coin != "__meta__":
            market["min_cost"] = 100.0
    payload = build_backtest_payload(candles, markets, config, args.exchange, btc, timestamps)
    fills, _, native = execute_backtest(payload, config)
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
                                timestamps=timestamps, exchange=args.exchange, batch_size=1,
                                needed_metrics={"fills_per_day"})
    # Inspect factual transport, without promoting exact-only counts to optimizer metrics.
    runner = proxy.fused_runner if proxy.fused_runner is not None else proxy.runners[sides]
    execute = runner.run
    captures = []
    def capture(*args, **kwargs):
        output = execute(*args, **kwargs)
        captures.append(output)
        return output
    monkeypatch.setattr(runner, "run", capture)
    result = proxy.evaluate([{}])[0]
    raw = captures[0]
    long_count = raw["fill_count_long"].item()
    short_count = raw["fill_count"].item() - long_count
    assert long_count == native["fills_count_long"]
    assert short_count == native["fills_count_short"]
    assert result["fills_per_day"] == pytest.approx(native["fills_per_day"], rel=1e-5)
    entries = [str(row[13]) for row in fills if str(row[13]).startswith("entry_")]
    assert all(value.endswith("_" + passing_side) for value in entries)
    if sides == "both" or sides == passing_side:
        assert entries
