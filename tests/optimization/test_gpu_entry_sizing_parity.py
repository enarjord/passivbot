"""Entry sizing must precede executable-price finalization in both engines."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def _fixture(side, coin_count, entry_kind):
    from config.schema import get_template_config

    coins = ["BTC", "ETH"][:coin_count]
    config = get_template_config()
    config["live"].update(
        strategy_kind="trailing_martingale",
        max_warmup_minutes=1,
        market_orders_allowed=False,
        approved_coins={"long": coins, "short": coins},
    )
    config["backtest"].update(
        coins={"bybit": coins}, exchanges=["bybit"], starting_balance=1000.0
    )
    for direction in ("long", "short"):
        bot = config["bot"][direction]
        bot["entry_cooldown"]["base_duration_minutes"] = 0.0
        bot["hsl"]["enabled"] = False
        bot["unstuck"]["enabled"] = False
        bot["risk"].update(
            n_positions=coin_count if direction == side else 0,
            total_wallet_exposure_limit=float(coin_count) if direction == side else 0.0,
            we_excess_allowance_pct=0.0,
            position_exposure_enforcer_enabled=False,
            total_exposure_enforcer_enabled=False,
        )
        strategy = bot["strategy"]["trailing_martingale"]
        strategy["entry"].update(
            ema_span_0=2.0,
            ema_span_1=3.0,
            ema_gate_mode="disabled",
            initial_qty_pct=0.1,
            threshold_base_pct=0.5,
            retracement_base_pct=0.1,
        )
        strategy["close"].update(threshold_base_pct=0.5, retracement_base_pct=0.1)
    if entry_kind == "reentry":
        config["bot"][side]["strategy"]["trailing_martingale"]["entry"].update(
            threshold_base_pct=0.01,
            retracement_base_pct=0.001,
            double_down_factor=0.1,
            threshold_we_weight=0.0,
            threshold_volatility_1h_weight=0.0,
            threshold_volatility_1m_weight=0.0,
            retracement_volatility_1h_weight=0.0,
            retracement_volatility_1m_weight=0.0,
            retracement_we_weight=0.0,
        )
    elif entry_kind == "minimum":
        entry = config["bot"][side]["strategy"]["trailing_martingale"]["entry"]
        entry["initial_qty_pct"] = 0.001
    count = 7
    candles = np.full((count, coin_count, 4), 100.4, dtype=np.float64)
    candles[:, :, 3] = 1.0
    candles[3, :, 0] = 102.0
    candles[3, :, 1] = 99.0
    if entry_kind == "reentry":
        close = 80.4 if side == "long" else 120.4
        candles[3:, :, 2] = close
        candles[3:, :, 0] = max(close + 2.0, 102.0)
        candles[3:, :, 1] = min(close - 2.0, 99.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    btc = np.full(count, 50_000.0)
    mss = {
        coin: dict(
            qty_step=0.001,
            price_step=1.0,
            min_qty=0.001,
            min_cost=100.0 if entry_kind == "minimum" else 0.0,
            c_mult=1.0,
            maker=0.0,
            taker=0.0,
            exchange="bybit",
            first_valid_index=0,
            last_valid_index=count - 1,
            warmup_minutes=1,
        )
        for coin in coins
    }
    mss["__meta__"] = {"requested_start_ts": int(timestamps[0])}
    return config, candles, mss, btc, timestamps


def _evaluate(side, inputs):
    from backtest import run_backtest
    from optimization.gpu.service import MpsMulticoinProxy, MpsSingleCoinProxy

    config, candles, mss, btc, timestamps = inputs
    coin_count = candles.shape[1]
    cls = MpsSingleCoinProxy if coin_count == 1 else MpsMulticoinProxy
    proxy = cls(
        config=config,
        hlcvs=candles,
        mss=mss,
        btc=btc,
        timestamps=timestamps,
        exchange="bybit",
        batch_size=1,
        needed_metrics={"adg_strategy_eq"},
    )
    outputs = []
    runner = proxy.runner if coin_count == 1 else proxy.runners[side]
    original = runner.run

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        outputs.append(out)
        return out

    runner.run = capture
    proxy.evaluate([{}])
    fills, _, _ = run_backtest(candles, mss, config, "bybit", btc, timestamps)
    return outputs[0], fills


@pytest.mark.parametrize("entry_kind", ["initial", "reentry", "minimum"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_raw_touch_entry_quantity_matches_exact_rust(side, coin_count, entry_kind):
    out, fills = _evaluate(side, _fixture(side, coin_count, entry_kind))
    expected_count = coin_count * (2 if entry_kind == "reentry" else 1)
    assert len(fills) == expected_count, fills
    if entry_kind == "initial":
        # Rust sizes at raw 100.4, then finalizes to 100 (long) or 101 (short).
        assert all(abs(float(fill[9])) == pytest.approx(0.996) for fill in fills)
    size_key = "short_psize" if side == "short" else "psize"
    assert out["fill_count"].item() == expected_count
    expected_size = sum(abs(float(fill[9])) for fill in fills)
    assert out[size_key].item() == pytest.approx(expected_size, abs=1e-5), fills


@pytest.mark.parametrize("side", ["long", "short"])
def test_flat_selection_tracks_readiness_without_fills(side):
    inputs = _fixture(side, 2, "initial")
    config, candles, _, _, _ = inputs
    bot = config["bot"][side]
    bot["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0)
    bot["forager"].update(
        score_weights={"volume": 0.0, "volatility": 0.0, "ema_readiness": 1.0},
        volume_drop_pct=0.0,
    )
    bot["strategy"]["trailing_martingale"]["entry"].update(
        ema_gate_mode="all",
        initial_ema_dist=0.1,
        ema_span_0=1000.0,
        ema_span_1=1000.0,
    )
    # Both initial orders are away from the touch. ETH then becomes more ready
    # without a fill or eligibility change, and must replace the flat incumbent.
    candles[:, :, :3] = 100.4
    price = 80.4 if side == "long" else 120.4
    candles[3:, 1, :3] = price
    candles[4:, 1, 0] = price + 2.0
    candles[4:, 1, 1] = price - 2.0
    out, fills = _evaluate(side, inputs)
    assert len(fills) == 1, fills
    assert fills[0][2] == "ETH"
    assert out["fill_count"].item() == 1
    size_key = "short_psize" if side == "short" else "psize"
    assert out[size_key].item() == pytest.approx(abs(float(fills[0][9])), abs=1e-5)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("retracement", [0.0, 0.001])
@pytest.mark.parametrize("case", [
    dict(price=70.62, balance=1102.0, wel=0.5, initial_pct=0.012,
         close_pct=0.0035, threshold=0.002, close_low=70.55, close_high=70.69,
         initial_qty=0.094, topup=0.028),
    dict(price=100.0, balance=1000.0, wel=1.0, initial_pct=0.015,
         close_pct=0.00615, threshold=0.002123, close_low=99.92, close_high=100.08,
         initial_qty=0.15, topup=0.062),
], ids=["94-minus-66-steps", "150-minus-88-steps"])
def test_partial_initial_top_up_preserves_venue_steps(side, coin_count, retracement, case):
    config, candles, mss, btc, timestamps = _fixture(side, coin_count, "initial")
    config["backtest"].update(starting_balance=case["balance"],
                             maker_fee_override=0.0, taker_fee_override=0.0)
    config["bot"][side]["risk"]["total_wallet_exposure_limit"] = case["wel"] * coin_count
    strategy = config["bot"][side]["strategy"]["trailing_martingale"]
    strategy["entry"].update(initial_qty_pct=case["initial_pct"], threshold_base_pct=0.5,
                             retracement_base_pct=retracement)
    # Exposure-dependent close prices retain separate ladder rungs. A zero
    # exposure weight would intentionally collapse to a full-position close.
    strategy["close"].update(qty_pct=case["close_pct"], threshold_base_pct=case["threshold"],
        retracement_base_pct=0.0, threshold_we_weight=-0.1,
        threshold_volatility_1h_weight=0.0, threshold_volatility_1m_weight=0.0)
    candles = np.tile(candles[:1], (12, 1, 1))
    price = case["price"]
    candles[:, :, :3] = price
    short = side == "short"
    # Trade after warmup: initial entry, one partial close, then a top-up.
    candles[4, :, 0 if short else 1] = price + 0.01 if short else price - 0.01
    candles[5, :, 1 if short else 0] = case["close_low"] if short else case["close_high"]
    candles[6, :, 0 if short else 1] = price + 0.01 if short else price - 0.01
    btc = np.full(12, 50_000.0)
    timestamps = timestamps[0] + np.arange(12, dtype=np.int64) * 60_000
    for coin, market in mss.items():
        if coin != "__meta__":
            market.update(qty_step=0.001, price_step=0.01, min_qty=0.001,
                          min_cost=1.0, last_valid_index=11)

    out, fills = _evaluate(side, (config, candles, mss, btc, timestamps))
    assert len(fills) == 3 * coin_count, fills
    topups = [fill for fill in fills if "entry_initial_partial" in fill[13]]
    assert len(topups) == coin_count, fills
    assert all(abs(float(fill[9])) == pytest.approx(case["topup"], abs=1e-12)
               for fill in topups)
    assert out["fill_count"].item() == len(fills)
    size_key = "short_psize" if short else "psize"
    assert out[size_key].item() == pytest.approx(case["initial_qty"] * coin_count, abs=1e-6)

@pytest.mark.parametrize("source_name", [
    "mps_trailing_martingale_source_py",
    "mps_trailing_martingale_multicoin_source_py",
])
def test_partial_top_up_recovers_counts_without_snapping_off_grid(source_name):
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    source = getattr(passivbot_rust, source_name)()

    def function(name):
        start = source.index("inline float " + name + "(")
        return source[start:source.index("\n}", start) + 2]

    probe = """#include <metal_stdlib>
using namespace metal;
""" + function("floor_step") + "\n" + function("aligned_qty_difference") + """
kernel void partial_top_up_probe(
    constant float* cases, device float* output,
    uint b [[thread_position_in_grid]]
) {
    output[b] = aligned_qty_difference(cases[3*b], cases[3*b+1], cases[3*b+2]);
}
"""
    # Expected differences come from authored venue counts, including counts
    # large enough for multiplication/subtraction to lose a small physical step.
    rng = np.random.default_rng(43)
    counts = [(94, 66), (150, 88), (65537, 65501), (1048575, 1048520)]
    counts += [(int(total), int(held)) for total, held in zip(
        rng.integers(1, 2**20, 64), rng.integers(0, 2**19, 64))]
    cases, expected = [], []
    for step in np.asarray([0.001, 0.25, 1e-8], dtype=np.float32):
        for total, held in counts:
            cases.append([np.float32(total) * step, np.float32(held) * step, step])
            expected.append(np.float32(max(total - held, 0)) * step)
    # Exact binary off-grid inputs must retain directional floor semantics.
    # At/above 2**24 units, require the existing physical-unit floor behavior.
    cases += [[10, 1.78125, 0.25], [10.03125, 1.75, 0.25],
              [2**22, 2**22 - 0.5, 0.25], [2**24, 2**24 - 2, 1]]
    expected += [8, 8.25, 0.5, 2]
    packed = torch.tensor(np.asarray(cases, dtype=np.float32), device=gpu_device())
    output = torch.empty(len(cases), dtype=torch.float32, device=gpu_device())
    compile_shader(probe).partial_top_up_probe(packed, output, threads=len(cases))
    synchronize()
    np.testing.assert_array_equal(output.cpu().numpy(), np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("market_allowed", [False, True])
@pytest.mark.parametrize("ladder", ["entry", "close"])
def test_passive_recursive_ladders_do_not_require_market_permission(
    side, coin_count, market_allowed, ladder
):
    config, candles, markets, btc, timestamps = _fixture(side, coin_count, "initial")
    config["live"].update(market_orders_allowed=market_allowed,
                          market_order_near_touch_threshold=0.0)
    config["backtest"].update(starting_balance=1000.0, maker_fee_override=0.0,
                             taker_fee_override=0.0, market_order_slippage_pct=0.0)
    strategy = config["bot"][side]["strategy"]["trailing_martingale"]
    strategy["entry"].update(initial_qty_pct=0.1, threshold_base_pct=0.01 if ladder == "entry" else 0.5,
                             retracement_base_pct=0.0, double_down_factor=1.0,
                             threshold_we_weight=0.0, threshold_volatility_1h_weight=0.0,
                             threshold_volatility_1m_weight=0.0)
    strategy["close"].update(threshold_base_pct=0.02005 if ladder == "close" else 0.5,
                             retracement_base_pct=0.0, qty_pct=0.01,
                             threshold_we_weight=-0.1 if ladder == "close" else 0.0,
                             threshold_volatility_1h_weight=0.0,
                             threshold_volatility_1m_weight=0.0)
    # Keep both events away from warmup and the final trading bar.
    candles = np.tile(candles[:1], (12, 1, 1))
    candles[:, :, :3] = 100.0
    btc = np.full(12, 50_000.0)
    timestamps = timestamps[0] + np.arange(12, dtype=np.int64) * 60_000
    # Keep later partial top-ups out of the close-ladder trace.
    if ladder == "close":
        config["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 100.0
    short = side == "short"
    candles[4, :, 0 if short else 1] = (
        101.4 if short else 98.6
    ) if ladder == "entry" else (100.1 if short else 99.9)
    if ladder == "close":
        candles[5, :, 1 if short else 0] = 98.65 if short else 101.35
    for coin, market in markets.items():
        if coin != "__meta__":
            market.update(qty_step=0.001, price_step=0.1, min_qty=0.001,
                          min_cost=1.0, last_valid_index=11)
    out, fills = _evaluate(side, (config, candles, markets, btc, timestamps))
    expected_count = (2 if ladder == "entry" else 4) * coin_count
    expected_size = (2.0 if short else 2.01) if ladder == "entry" else 0.7
    assert len(fills) == expected_count, fills
    assert abs(sum(float(fill[9]) for fill in fills)) == pytest.approx(
        expected_size * coin_count, abs=1e-10, rel=0
    ), fills
    expected_balance = 1000.0
    if ladder == "close":
        expected_balance += 0.36 * coin_count
    assert out["fill_count"].item() == expected_count
    key = "short_psize" if short else "psize"
    assert out[key].item() == pytest.approx(expected_size * coin_count, abs=1e-5, rel=0)
    # At most one float32 ULP per balance update; a missed tick is much larger.
    allowance = float(np.spacing(np.float32(expected_balance))) * expected_count
    assert out["balance"].item() == pytest.approx(expected_balance, abs=allowance, rel=0)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("entry_gate", [False, True])
@pytest.mark.parametrize("hysteresis", [0.0, 0.02])
def test_unfilled_forager_incumbent_survives_entry_gate(side, entry_gate, hysteresis):
    config, candles, markets, _, timestamps = _fixture(side, 2, "initial")
    config["live"].update(market_orders_allowed=False, forager_score_hysteresis_pct=hysteresis)
    config["backtest"].update(starting_balance=1000.0, maker_fee_override=0.0,
                             taker_fee_override=0.0, market_order_slippage_pct=0.0)
    bot = config["bot"][side]
    bot["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0,
                       total_exposure_entry_gate_enabled=entry_gate)
    bot["forager"].update(score_weights={"volume": 0.01, "volatility": 0.99, "ema_readiness": 0.0},
                          volume_ema_span_1m=1.0, volatility_ema_span_1m=1.0,
                          volume_drop_pct=0.0)
    bot["strategy"]["trailing_martingale"]["entry"].update(
        initial_qty_pct=0.1, initial_ema_dist=0.25, ema_gate_mode="initial",
        threshold_base_pct=0.5, retracement_base_pct=0.0,
        threshold_we_weight=0.0, threshold_volatility_1h_weight=0.0,
        threshold_volatility_1m_weight=0.0)
    candles = np.tile(candles[:1], (12, 1, 1))
    candles[:, :, :3] = 100.0
    # BTC owns the resting entry before ETH becomes slightly more attractive.
    candles[:5, :, 3] = [200.0, 100.0]
    candles[5:, :, 3] = [100.0, 200.0]
    short = side == "short"
    # Only BTC reaches its resting entry after the score change.
    candles[8, 0, 0 if short else 1] = 125.25 if short else 74.75
    btc = np.full(12, 50_000.0)
    timestamps = timestamps[0] + np.arange(12, dtype=np.int64) * 60_000
    for coin, market in markets.items():
        if coin != "__meta__":
            market.update(qty_step=0.001, price_step=0.25, min_qty=0.001,
                          min_cost=1.0, last_valid_index=11)
    out, fills = _evaluate(side, (config, candles, markets, btc, timestamps))
    expected_count = int(hysteresis > 0.0)
    assert len(fills) == expected_count, fills
    expected_size = (0.8 if short else 1.333) if expected_count else 0.0
    if expected_count:
        assert fills[0][2] == "BTC", fills
        assert abs(float(fills[0][9])) == pytest.approx(expected_size, abs=1e-12, rel=0)
    assert out["fill_count"].item() == expected_count
    key = "short_psize" if short else "psize"
    assert out[key].item() == pytest.approx(expected_size, abs=1e-5, rel=0)
