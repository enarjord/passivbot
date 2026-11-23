"""Compare device close sizing and candidate isolation with Rust replay."""

from copy import deepcopy

import pytest


@pytest.fixture
def close_sizing_device():
    torch = pytest.importorskip("torch")
    from optimization.gpu.runtime import gpu_vendor

    if gpu_vendor(torch) not in {"apple", "nvidia"}:
        pytest.skip("actual Metal or NVIDIA CUDA device required")

    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()


def assert_fill_summary(fills, result, sides, index=0):
    assert result["fill_count"][index].item() == len(fills)
    assert result["balance"][index].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if str(fill[13]).endswith(side))
        field = "psize" if side == "long" else "short_psize"
        assert result[field][index].item() == pytest.approx(abs(expected), abs=2e-5)


def fee_only_replay(strategy, sides, gate_cap):
    from test_gpu_realized_loss_lookback import loss_gate_proxy
    from test_gpu_tm_loss_admission import fee_only_close_fixture

    if strategy == "ema_anchor":
        return loss_gate_proxy(sides, max_loss_pct=gate_cap)
    return fee_only_close_fixture(sides, 2, 8.0, gate_cap)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("gate_cap", [0.0, 0.0003, 0.999999999])
def test_fee_only_closes_match_authoritative_cpu(strategy, sides, gate_cap, close_sizing_device):
    from backtest import run_backtest
    from test_gpu_hsl_multicoin import raw

    replay, inputs = fee_only_replay(strategy, sides, gate_cap)
    fills, _, _ = run_backtest(*inputs)
    assert any(str(fill[13]).startswith("entry_") for fill in fills)
    runner, result = raw(replay, [{}])
    assert runner.loss_gate_enabled
    assert not runner.dispatch_unstuck_enabled
    assert_fill_summary(fills, result, sides)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("short",), ("long", "short")])
@pytest.mark.parametrize("gate_cap", [None, 0.005])
def test_exposure_targeted_unstuck_matches_authoritative_cpu(
    strategy, sides, gate_cap, close_sizing_device,
):
    from backtest import run_backtest
    from test_gpu_hsl_multicoin import compare, raw
    from test_gpu_unstuck_lookback import make_proxy

    replay, inputs = make_proxy(sides, strategy=strategy, loss_gate_cap=gate_cap)
    fills, _, _ = run_backtest(*inputs)
    assert any(str(fill[13]).startswith("close_unstuck_") for fill in fills)
    runner, result = raw(replay, [{}])
    assert runner.dispatch_unstuck_enabled
    assert not runner.dispatch_unstuck_ema_enabled
    assert_fill_summary(fills, result, sides)

    if len(sides) == 2 and gate_cap is not None:
        runner.max_dispatch_candidate_bars = 24
        _, chunked = raw(replay, [{}], profile=True)
        assert runner.last_profile["dispatch_count"] > 1
        compare(result, chunked)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_portfolio_loss_cap_varies_by_candidate(strategy, close_sizing_device):
    from backtest import run_backtest
    from test_gpu_hsl_multicoin import raw

    sides = ("long", "short")
    replay, inputs = fee_only_replay(strategy, sides, 0.0)
    caps = (0.0, 0.0003, 0.0)
    runner, output = raw(replay, [{"loss_gate_max_drawdown_fraction": cap} for cap in caps])
    assert runner.loss_gate_enabled
    assert not runner.dispatch_unstuck_enabled
    candles, markets, config, exchange, btc, timestamps = inputs
    expected_counts = []
    for index, cap in enumerate(caps):
        candidate = deepcopy(config)
        loss_gate = candidate["bot"]["portfolio"]["risk"]["loss_realization"]["loss_gate"]
        loss_gate["max_realized_drawdown_fraction"] = cap
        fills, _, _ = run_backtest(candles, markets, candidate, exchange, btc, timestamps)
        expected_counts.append(len(fills))
        assert_fill_summary(fills, output, sides, index)
    zero_cap_count, positive_cap_count, _ = expected_counts
    assert zero_cap_count < positive_cap_count


@pytest.mark.parametrize("strategy,side", [("ema_anchor", "long"), ("trailing_martingale", "short")])
@pytest.mark.parametrize("distance,unwinds", [(-0.1, True), (0.1, False)])
def test_coin_override_enables_ema_gated_unstuck(
    strategy, side, distance, unwinds, close_sizing_device,
):
    from backtest import run_backtest
    from optimization.gpu.service import SharedAccountReplay
    from test_gpu_hsl_multicoin import raw
    from test_gpu_unstuck_lookback import make_inputs

    sides = (side,)
    inputs = make_inputs(sides, strategy=strategy, coins=1)
    candles, markets, config, exchange, btc, timestamps = inputs
    config["bot"][side]["risk"]["unstuck"]["enabled"] = False
    coin_unstuck = {
        "enabled": True,
        "ema_gating_enabled": True,
        "ema_distance_fraction": distance,
        "ema_span_0_minutes": 120.0,
        "ema_span_1_minutes": 240.0,
    }
    config["coin_overrides"] = {"BTC": {"bot": {side: {"risk": {"unstuck": coin_unstuck}}}}}
    replay = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange=exchange, batch_size=2, needed_metrics={"cagr_fraction_strategy_equity"},
    )

    fills, _, _ = run_backtest(*inputs)
    assert any(str(fill[13]).startswith("close_unstuck_") for fill in fills) is unwinds
    runner, result = raw(replay, [{}])
    assert runner.dispatch_unstuck_enabled
    assert runner.dispatch_unstuck_ema_enabled
    assert_fill_summary(fills, result, sides)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_disabled_loss_gate_ignores_dormant_cap(strategy, close_sizing_device):
    from backtest import run_backtest
    from optimization.gpu.service import SharedAccountReplay
    from test_gpu_hsl_multicoin import compare, raw

    sides = ("long", "short")
    without_cap, inputs = fee_only_replay(strategy, sides, None)
    candles, markets, config, exchange, btc, timestamps = inputs
    dormant_config = deepcopy(config)
    loss_gate = dormant_config["bot"]["portfolio"]["risk"]["loss_realization"]["loss_gate"]
    loss_gate["max_realized_drawdown_fraction"] = 0.42
    dormant_replay = SharedAccountReplay(
        config=dormant_config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange=exchange, batch_size=2, needed_metrics={
            "cagr_fraction_strategy_equity", "gross_turnover_ratio_per_day",
        },
    )

    expected_fills, _, _ = run_backtest(*inputs)
    dormant_fills, _, _ = run_backtest(candles, markets, dormant_config, exchange, btc, timestamps)
    absent_runner, expected = raw(without_cap, [{}])
    dormant_runner, result = raw(dormant_replay, [{}])
    assert not absent_runner.loss_gate_enabled and not dormant_runner.loss_gate_enabled
    assert_fill_summary(expected_fills, expected, sides)
    assert_fill_summary(dormant_fills, result, sides)
    compare(expected, result)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("btc_touch,first_coin", [(100.30, "BTC"), (100.306, "ETH")])
def test_unstuck_priority_preserves_aligned_and_off_tick_touches(
    strategy, btc_touch, first_coin, close_sizing_device, monkeypatch,
):
    from collections import Counter

    import numpy as np

    from backtest import run_backtest
    from optimization.gpu.service import SharedAccountReplay
    from test_gpu_hsl_multicoin import raw
    from test_gpu_unstuck_lookback import make_inputs

    sides = ("long",)
    inputs = make_inputs(sides, strategy=strategy, coins=2, loss_gate_cap=0.005)
    candles, markets, config, exchange, btc, timestamps = inputs
    prices = np.array([100.30, 100.40])
    candles[:, :, :3] = prices[:, None]
    if strategy == "ema_anchor":
        anchor = config["bot"]["long"]["strategy"][strategy]
        anchor["offset_fraction"] = 0.001
        # Warm the anchor high enough to keep ordinary asks outside fill range
        # while the two initial bids still land exactly at 100.30 and 100.40.
        candles[:3, :, :3] = (prices / (1.0 - anchor["offset_fraction"]))[:, None]
    # 100.306 rounds to the submitted 100.31 ask. It retains a genuine gap
    # to that ask, so ETH at its touch must lead in the off-tick case.
    candles[3:, 0, :3] = btc_touch
    candles[:, :, 0] += 0.02
    candles[:, :, 1] -= 0.02

    fills, _, _ = run_backtest(*inputs)
    first_unstuck = next(fill for fill in fills if str(fill[13]) == "close_unstuck_long")
    first_bar = int(first_unstuck[0])
    assert first_unstuck[2] == first_coin
    bars = fills[:, 0].astype(int)
    before = fills[bars < first_bar]
    through = fills[bars <= first_bar]
    assert len(before) == 2 and len(through) == 3

    construct = SharedAccountReplay._construct_runner

    def construct_counting_runner(self, key, runner_cls, kwargs):
        return construct(self, key, runner_cls, {**kwargs, "collect_coin_fill_counts": True})

    monkeypatch.setattr(SharedAccountReplay, "_construct_runner", construct_counting_runner)
    replay = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange=exchange, batch_size=2, needed_metrics={"cagr_fraction_strategy_equity"},
    )
    _, result = raw(replay, [{}, {}], end_steps=np.array([first_bar, first_bar + 1], dtype=np.int32))
    expected_counts = []
    for index, prefix in enumerate((before, through)):
        counts = Counter(fill[2] for fill in prefix)
        expected_counts.append([counts[coin] for coin in ("BTC", "ETH")])
        assert_fill_summary(prefix, result, sides, index=index)
    assert result["coin_fill_counts"].cpu().tolist() == expected_counts
