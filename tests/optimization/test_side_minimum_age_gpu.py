"""Factual side-age admission is shared by native CPU and supported GPU replays."""

import numpy as np
import pytest


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("topology", ["long", "short", "fused", "oneway"])
@pytest.mark.parametrize("eligible_side", ["long", "short"])
def test_actual_gpu_age_admission_matches_native_counts(monkeypatch, kind, topology, eligible_side):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("local Metal or CUDA device unavailable")
    from backtest import build_backtest_payload, execute_backtest
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import build_parser, fixture_inputs

    sides = topology if topology in {"long", "short"} else "both"
    args = build_parser().parse_args(["--fixture", kind, "--hsl", "disabled", "--sides", sides,
                                     "--coins", "2", "--bars", "64", "--seed", "2026"])
    config, candles, markets, btc, timestamps = fixture_inputs(args)
    config["bot"]["global"]["hedge_mode"] = topology != "oneway"
    config["bot"]["global"]["warmup"]["max_minutes"] = 1.0
    config["bot"]["global"]["universe"]["filter_by_min_effective_cost"] = False
    config["backtest"]["filter_by_min_effective_cost"] = False
    for side in ("long", "short"):
        bot = config["bot"][side]
        # The eligible side crosses its fractional age threshold during replay;
        # the opposite side stays too young for the whole selected window.
        bot["universe"]["minimum_coin_age_days"] = (5.5 / 1440 if side == eligible_side else 60.0)
        bot["risk"]["max_positions"] = 1 if sides == "both" or side == sides else 0
        bot["entry_cooldown"]["base_duration_minutes"] = 1000.0
        strategy = bot["strategy"][kind]
        spans = strategy if kind == "ema_anchor" else strategy["entry"]
        spans.update(ema_span_0_minutes=2.0, ema_span_1_minutes=3.0)
        if kind == "ema_anchor":
            strategy.update(base_qty_fraction=0.01, offset_fraction=0.0, entry_double_down_factor=0.0)
        else:
            strategy["entry"].update(initial_qty_fraction=0.01, initial_ema_distance_fraction=0.0, double_down_factor=0.0)
            for group in ("entry", "close"):
                strategy[group].update(threshold_volatility_1h_weight=0.0, retracement_volatility_1h_weight=0.0)
    for coin, market in markets.items():
        if coin != "__meta__":
            market["inception_timestamp_ms"] = int(timestamps[0])
            market["warmup_minutes"] = 1
    payload = build_backtest_payload(candles, markets, config, args.exchange, btc, timestamps)
    fills, _, native = execute_backtest(payload, config)
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
                                timestamps=timestamps, exchange=args.exchange, batch_size=1,
                                needed_metrics={"fills_per_day"})
    settings = proxy.data["coin_settings"].cpu().numpy()
    assert settings.shape == (2, 15)
    np.testing.assert_array_equal(settings[:, 6], [0, 0])
    np.testing.assert_array_equal(settings[:, 8], [1, 1])
    np.testing.assert_array_equal(settings[:, 13:], np.tile([6, 64] if eligible_side == "long" else [64, 6], (2, 1)))
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
    assert raw["fill_count_long"].item() == native["fills_count_long"]
    assert raw["fill_count"].item() - raw["fill_count_long"].item() == native["fills_count_short"]
    assert result["fills_per_day"] == pytest.approx(native["fills_per_day"], rel=1e-5)
    entries = [row for row in fills if str(row[13]).startswith("entry_")]
    assert all(str(row[13]).endswith("_" + eligible_side) for row in entries)
    assert all(int(row[1]) >= int(timestamps[7]) for row in entries)
    if sides == "both" or sides == eligible_side:
        assert entries
        assert raw["first_fill_ts"].item() == int(fills[0][1]) - int(timestamps[0])


def test_both_strategy_sources_and_cuda_translation_keep_age_before_side_arbitration():
    import passivbot_rust as pbr
    from optimization.gpu.cuda_kernel import cuda_source
    for source in (pbr.mps_ema_anchor_multicoin_source_py(), pbr.mps_trailing_martingale_multicoin_source_py()):
        for translated in (source, cuda_source(source)):
            assert "COIN_COLS = 15" in translated
            assert "short_side ? 14 : 13" in translated
            assert "long_can_generate && coin_tradable\n            && passes_multicoin_age" in translated
            assert "short_can_generate && coin_tradable\n            && passes_multicoin_age" in translated
            assert "psize[c] > 0.0f || passes_multicoin_age" in translated
