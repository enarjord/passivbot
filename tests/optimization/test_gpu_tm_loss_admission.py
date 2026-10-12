"""TM loss allowance must admit affordable closes with shared generation accounting."""

from copy import deepcopy

import pytest

from test_gpu_hsl_multicoin import compare, raw
from test_gpu_unstuck_lookback import make_proxy

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


@pytest.fixture(scope="module", autouse=True)
def verified_rust_runtime():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()


def test_tm_admission_failure_cannot_be_decoded_as_metrics():
    from optimization.gpu.mps_kernel import _require_available_held_valuation

    scalars = torch.zeros((2, 10))
    scalars[:, 9] = torch.tensor([-1.0, -5.0])
    with pytest.raises(RuntimeError, match="close-admission invariant failed"):
        _require_available_held_valuation(scalars)


def fee_only_close_fixture(sides, coins, lookback, max_loss_pct):
    from optimization.gpu.service import MpsMulticoinProxy

    original, inputs = make_proxy(
        sides, lookback, strategy="trailing_martingale", coins=coins
    )
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    config = deepcopy(config)
    config["live"]["max_realized_loss_pct"] = max_loss_pct
    candles = candles.copy()
    candles[:, :, :3] = (101.0, 99.0, 100.4)
    for side in sides:
        config["bot"][side]["unstuck"]["enabled"] = False
        # Zero exposure slope makes duplicate recursive rungs one full close.
        # Its price is flat, so only the projected fee consumes the allowance.
        config["bot"][side]["strategy"]["trailing_martingale"]["close"].update(
            qty_pct=1.0, threshold_base_pct=0.0, threshold_we_weight=0.0,
            threshold_volatility_1h_weight=0.0, threshold_volatility_1m_weight=0.0,
            retracement_base_pct=0.0, retracement_volatility_1h_weight=0.0,
            retracement_volatility_1m_weight=0.0,
        )
    inputs = candles, markets, config, exchange, btc, timestamps
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, exchange=exchange,
        btc=btc, timestamps=timestamps, batch_size=3,
        needed_metrics={"adg_strategy_eq", "fills_per_day"},
    )
    return proxy, inputs


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("lookback", [8 / 1440, "all"])
@pytest.mark.parametrize("max_loss_pct", [0.0, 0.0003, 0.1])
def test_tm_ordinary_close_uses_configured_loss_allowance(sides, coins, lookback, max_loss_pct):
    from backtest import run_backtest

    proxy, inputs = fee_only_close_fixture(sides, coins, lookback, max_loss_pct)
    _, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    expected_entries = coins * len(sides)
    assert sum(str(fill[13]).startswith("entry_") for fill in fills) == expected_entries
    expected_closes = max_loss_pct == 0.1 or (max_loss_pct == 0.0003 and lookback != "all")
    assert len(fills) == expected_entries * (2 if expected_closes else 1)
    assert output["fill_count"].item() == len(fills)
    assert output["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if fill[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("slope", [-0.02, 0.0, 0.02])
@pytest.mark.parametrize("market", [False, True])
@pytest.mark.parametrize("reducer", ["disabled", "wel", "twel"])
def test_tm_recursive_loss_plan_matches_finalized_rust_orders(sides, slope, market, reducer):
    from backtest import run_backtest
    from optimization.gpu.service import MpsMulticoinProxy

    original, inputs = fee_only_close_fixture(sides, 2, "all", 0.0004)
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    config["live"].update(
        market_orders_allowed=market, market_order_near_touch_threshold=0.01,
    )
    config["backtest"]["market_order_slippage_pct"] = 0.001
    candles[:, :, :3] = (105.0, 95.0, 100.4)
    for side in sides:
        config["bot"][side]["risk"].update(
            position_exposure_enforcer_enabled=reducer == "wel",
            position_exposure_enforcer_threshold=0.3,
            total_exposure_enforcer_enabled=reducer == "twel",
            total_exposure_enforcer_threshold=0.2,
            total_exposure_enforcer_policy="reduce_portfolio",
        )
        config["bot"][side]["strategy"]["trailing_martingale"]["close"].update(
            qty_pct=0.1, threshold_base_pct=0.005, threshold_we_weight=slope,
        )
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, exchange=exchange,
        btc=btc, timestamps=timestamps, batch_size=2,
        needed_metrics={"adg_strategy_eq"},
    )
    _, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(candles, markets, config, exchange, btc, timestamps)
    # Market sizing/fees can leave some coins without an initial entry. Compare
    # against emitted Rust intent rather than assume every eligible coin enters.
    assert len(fills) > 0
    assert output["fill_count"].item() == len(fills)
    assert output["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if fill[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("lookback", [8 / 1440, "all"])
@pytest.mark.parametrize("max_loss_pct", [0.0015, 0.1])
def test_tm_unstuck_and_general_loss_budget_share_effective_history(
    sides, coins, lookback, max_loss_pct
):
    from backtest import run_backtest
    from optimization.gpu.service import MpsMulticoinProxy

    original, inputs = make_proxy(sides, lookback, coins=coins)
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    config["live"]["max_realized_loss_pct"] = max_loss_pct
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, exchange=exchange,
        btc=btc, timestamps=timestamps, batch_size=2,
        needed_metrics={"adg_strategy_eq"},
    )
    _, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(candles, markets, config, exchange, btc, timestamps)
    if max_loss_pct == 0.1:
        assert any("close_unstuck" in str(fill[13]) for fill in fills)
    assert len(fills) >= coins * len(sides)
    assert output["fill_count"].item() == len(fills)
    assert output["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if fill[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("max_loss_pct", [0.0, 0.1, 1.0])
def test_tm_loss_history_consumer_and_compiler_ablation(sides, max_loss_pct):
    proxy, _ = fee_only_close_fixture(sides, 2, 8 / 1440, max_loss_pct)
    runner, specialized = raw(proxy, [{}, {}])
    enabled = max_loss_pct < 1.0
    assert runner.loss_gate_enabled == enabled
    assert runner.unstuck_pnl_lookback_bars == (8 if enabled else 0)
    assert runner.unstuck_pnl_capacity == (9 if enabled else 0)
    assert bool(runner._unstuck_pnl_buffers) == enabled
    if len(sides) == 1:
        runner.max_dispatch_candidate_bars = 24
        _, replayed = raw(proxy, [{}, {}])
        compare(specialized, replayed)
        specialized_bytes = runner._replay_state_bytes
    runner.loss_gate_specialization = False
    _, general = raw(proxy, [{}, {}])
    compare(specialized, general)
    if len(sides) == 1:
        assert runner._replay_state_bytes >= specialized_bytes
        if not enabled:
            assert runner._replay_state_bytes > specialized_bytes
        runner.loss_gate_specialization = True
        _, restored = raw(proxy, [{}, {}])
        compare(specialized, restored)
        assert runner._replay_state_bytes == specialized_bytes


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_tm_loss_plan_survives_replay_scratch_reuse_and_reordering(sides):
    import numpy as np

    proxy, _ = fee_only_close_fixture(sides, 2, 8 / 1440, 0.0003)
    candidates = [{}, {f"{sides[0]}_entry_initial_qty_pct": 0.3}, {}]
    runner, expected = raw(proxy, candidates)
    assert expected["balance"][0].item() != expected["balance"][1].item()
    if len(sides) == 1:
        runner.max_dispatch_candidate_bars = 24
    _, chunked = raw(proxy, candidates, end_steps=np.full(3, runner.n, dtype=np.int32))
    compare(expected, chunked)
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate() * 2
    _, split = raw(proxy, candidates, profile=True)
    compare(expected, split)
    assert runner.last_profile["candidate_batch_count"] == 2
    _, repeated = raw(proxy, candidates)
    compare(expected, repeated)
    _, reordered = raw(proxy, candidates[::-1])
    compare({k: v.flip(0) if isinstance(v, torch.Tensor) else v
             for k, v in expected.items()}, reordered)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_tm_loss_admission_never_replays_on_cpu(monkeypatch, sides, coins):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA service required")
    import backtest
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    proxy, inputs = fee_only_close_fixture(sides, coins, 8 / 1440, 0.0003)
    candidates = [{}, {f"{sides[0]}_entry_initial_qty_pct": 0.3}, {}]
    expected = proxy.evaluate_results(candidates)

    def forbidden(*args, **kwargs):
        pytest.fail("native TM admission must not execute a CPU backtest")

    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    candles, markets, config, exchange, btc, timestamps = inputs
    with _native_dataset((config, candles, markets, btc, timestamps), exchange,
                         {"adg_strategy_eq", "fills_per_day"}) as dataset:
        with CudaBacktestService(batch_size=2, tuning_mode="off") as service:
            service.register_dataset("tm-loss", dataset)
            pending = [service.submit(BacktestRequest(str(i), "tm-loss", values))
                       for i, values in enumerate(candidates)]
            actual = [future.result() for future in pending]
            repeated = service.submit(BacktestRequest("repeat", "tm-loss", {})).result()
    assert [result.metrics for result in actual] == [result.metrics for result in expected]
    assert repeated.metrics == expected[0].metrics


def test_tm_account_admission_reserves_unfilled_intent_and_reducer_fallback():
    """Exercise shared intent independently of whether next-bar fills occur."""
    import numpy as np
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize

    probe = r"""
kernel void tm_account_admission_probe(
    constant float* bars, constant int* ticks, constant float* settings,
    device float* output, uint b [[thread_position_in_grid]]
) {
    if (b >= 7) return;
    TrailingMartingaleMulticoinSideState long_side, short_side;
    for (int rank = 0; rank < 2; ++rank) {
        thread TrailingMartingaleMulticoinSideState& side = rank == 0 ? long_side : short_side;
        for (int c = 0; c < 2; ++c) {
            side.close_admission[c].valid = c == rank;
            side.close_is_hsl_panic[c] = false;
            side.close_qty[c] = 0.0f;
            if (c != rank) continue;
            thread TmCloseAdmission& source = side.close_admission[c];
            source.trailing = false;
            source.context = {
                rank == 1, 5.0f, 100.0f, 1000.0f, 1.0f,
                10000, 10000, 10000, 0.1f, 0,
                1.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.1f, 0.01f, 0.1f, 0.0f, 1.0f,
                0, 0.0f, 500, 100.0f, false, 0.0f, 5.0f
            };
            source.ordinary = {rank == 0 ? 9800 : 10200,
                              rank == 0 ? 98.0f : 102.0f, 1.0f, false};
            for (int i = 0; i < 3; ++i) source.reducers[i] = {0, 0.0f, 0.0f, false};
        }
    }
    float budget = 3.0f;
    if (b == 1) {
        // A projected profit must not finance the second position's loss.
        long_side.close_admission[0].ordinary = {10200, 102.0f, 1.0f, false};
        budget = 1.0f;
    } else if (b == 2 || b == 3) {
        long_side.close_admission[0].ordinary.qty = 0.0f;
        short_side.close_admission[1].ordinary.qty = 0.0f;
        long_side.close_admission[0].reducers[0] = {9900, 99.0f, 1.0f, false};
        short_side.close_admission[1].reducers[0] = {10100, 101.0f, 2.0f, false};
        budget = 2.5f;
        if (b == 3) {
            // Reject the largest alternative, then rank its affordable fallback
            // against the other position. Canonical coin zero wins the tie.
            short_side.close_admission[1].reducers[0] = {11000, 110.0f, 2.0f, false};
            short_side.close_admission[1].reducers[1] = {10100, 101.0f, 1.0f, false};
            budget = 1.5f;
        }
    } else if (b == 4) {
        // Panic intent is exempt, and must neither be rewritten nor reserved.
        long_side.close_is_hsl_panic[0] = true;
        long_side.close_qty[0] = 5.0f;
        budget = 1.0f;
    } else if (b == 5) {
        // A profitable quote promoted to market uses touch/slippage/taker fee.
        long_side.close_admission[0].ordinary = {10200, 102.0f, 1.0f, true};
        long_side.close_admission[0].context.generation_market_price = 99.5f;
        short_side.close_admission[1].valid = false;
        budget = 2.0f;
    } else if (b == 6) {
        long_side.close_admission[0].ordinary.price = INFINITY;
    }
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    bool success = apply_tm_multicoin_close_admission(
        &long_side, &short_side, account, bars, ticks, settings,
        0, 2, 2, budget / 1000.0f, b == 5 ? 0.01f : 0.0f
    );
    output[b * 3] = success ? 1.0f : 0.0f;
    output[b * 3 + 1] = long_side.close_qty[0];
    output[b * 3 + 2] = short_side.close_qty[1];
}
"""
    # Neither side's passive ladder crosses the next candle. Reservations must
    # nevertheless include the unfilled singleton orders emitted this candle.
    bars = np.full((2, 2, 4), 100.0, dtype=np.float32)
    ticks = np.full((2, 2, 2), 10000, dtype=np.int32)
    settings = np.zeros((2, 13), dtype=np.float32)
    settings[:, :5] = [0.1, 0.01, 0.1, 0.0, 1.0]
    settings[:, 7] = 1
    settings[:, 11] = 0.01
    source = (
        "#define PASSIVBOT_HSL_CAPACITY 1\n"
        "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 0\n"
        + passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe
    )
    device = gpu_device()
    output = torch.zeros((7, 3), device=device)
    library = compile_shader(source, cuda_coin_capacity=2, mps_coin_capacity=2)
    library.tm_account_admission_probe(
        torch.tensor(bars, device=device), torch.tensor(ticks, device=device),
        torch.tensor(settings, device=device), output, threads=(7, 1, 1),
    )
    synchronize()
    np.testing.assert_array_equal(output[:6].cpu().numpy(),
                                  [[1, 1, 0], [1, 1, 0], [1, 0, 2],
                                   [1, 1, 0], [1, 5, 0], [1, 0, 0]])
    assert output[6, 0].item() == 0.0


def test_tm_loss_admission_checks_final_dust_and_executable_minimums():
    """Reserve executable quantities, including the Rust reducer-dust regression."""
    import numpy as np
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize

    probe = r"""
kernel void tm_finalized_loss_probe(
    constant float* bars, constant int* ticks, constant float* settings,
    device float* output, uint b [[thread_position_in_grid]]
) {
    if (b >= 12) return;
    bool short_side = b >= 6;
    int mode = int(b % 6);
    float loss_price = short_side ? 110.0f : 90.0f;
    TrailingMartingaleMulticoinSideState side;
    side.close_is_hsl_panic[0] = false;
    thread TmCloseAdmission& source = side.close_admission[0];
    source.valid = true;
    source.trailing = true;
    source.context = {
        short_side, 11.0f, 100.0f, 1000.0f, 1.0f,
        10000, 10000, 10000, 10.0f, 0,
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.01f, 10.0f, 0.0f, 1.0f,
        0, 0.0f, 0, 100.0f, false, 0.0f, 11.0f
    };
    source.ordinary = {0, 0.0f, 0.0f, false};
    for (int i = 0; i < 3; ++i) source.reducers[i] = {0, 0.0f, 0.0f, false};
    float budget = mode == 0 ? 105.0f : 115.0f;
    source.reducers[0] = {int(rint(loss_price / 0.01f)), loss_price, 10.0f, false};
    if (mode >= 2) {
        source.context.psize = source.context.market_resize_psize = 5.0f;
        source.reducers[0].qty = 0.0f;
        source.ordinary = {int(rint(loss_price / 0.01f)), loss_price, 2.0f, false};
        budget = 60.0f;
    }
    if (mode == 3) {
        // Trim ordinary quantity first; preserve the independent TWEL reducer.
        source.context.min_qty = 1.0f;
        source.ordinary.qty = 4.0f;
        source.reducers[1] = {int(rint(loss_price / 0.01f)), loss_price, 3.0f, false};
    } else if (mode == 4) {
        // Drop a below-minimum leg, then charge the reducer's absorbed dust.
        source.context.psize = source.context.market_resize_psize = 3.0f;
        source.context.qty_step = 0.5f;
        source.context.min_qty = 0.5f;
        source.context.min_cost = 180.0f;
        source.ordinary = {10000, 100.0f, 0.5f, false};
        source.reducers[1] = {int(rint(loss_price / 0.01f)), loss_price, 2.0f, false};
        budget = 25.0f;
    } else if (mode == 5) {
        // Market minimum uses executable touch, not a remote quote price.
        source.context.min_qty = 1.0f;
        source.context.min_cost = 150.0f;
        source.ordinary = {20000, 200.0f, 1.0f, true};
    }
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    bool success = apply_tm_multicoin_close_admission(
        short_side ? nullptr : &side, short_side ? &side : nullptr,
        account, bars, ticks, settings, 0, 2, 1, budget / 1000.0f, 0.0f
    );
    output[b * 3] = success ? 1.0f : 0.0f;
    output[b * 3 + 1] = side.close_qty[0];
    output[b * 3 + 2] = side.secondary_close_qty[0];
}
"""
    bars = torch.full((2, 1, 4), 100.0, device=gpu_device())
    ticks = torch.full((2, 1, 2), 10000, dtype=torch.int32, device=gpu_device())
    settings = torch.zeros((1, 13), device=gpu_device())
    settings[:, :5] = torch.tensor([1.0, 0.01, 10.0, 0.0, 1.0], device=gpu_device())
    settings[:, 7] = 1
    output = torch.zeros((12, 3), device=gpu_device())
    source = (
        "#define PASSIVBOT_HSL_CAPACITY 1\n"
        "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 0\n"
        + passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe
    )
    library = compile_shader(source, cuda_coin_capacity=1, mps_coin_capacity=1)
    library.tm_finalized_loss_probe(bars, ticks, settings, output, threads=(12, 1, 1))
    synchronize()
    # The original Rust regression rejects 11 * 10 loss with allowance 105;
    # an allowance of 115 admits the full executable reducer.
    expected = [[1, 0, 0], [1, 11, 0], [1, 5, 0],
                [1, 3, 2], [1, 0, 0], [1, 0, 0]]
    np.testing.assert_array_equal(output.cpu().numpy(), expected * 2)
