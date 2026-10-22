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
    from optimization.gpu.service import MpsMulticoinProxy

    config, candles, mss, btc, timestamps = inputs
    coin_count = candles.shape[1]
    cls = MpsMulticoinProxy
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
    runner = proxy.runners[side]
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
@pytest.mark.parametrize("market_allowed, near_touch", [(False, 0.0), (True, 0.0), (False, 0.02)])
@pytest.mark.parametrize("ladder", ["entry", "close", "close_on_tick"])
def test_passive_recursive_ladders_do_not_require_market_permission(
    side, coin_count, market_allowed, near_touch, ladder
):
    close_on_tick = ladder == "close_on_tick"
    is_close = ladder != "entry"
    config, candles, markets, btc, timestamps = _fixture(side, coin_count, "initial")
    config["live"].update(market_orders_allowed=market_allowed,
                          market_order_near_touch_threshold=near_touch)
    config["backtest"].update(starting_balance=1000.0, maker_fee_override=0.0,
                             taker_fee_override=0.0, market_order_slippage_pct=0.0)
    strategy = config["bot"][side]["strategy"]["trailing_martingale"]
    strategy["entry"].update(initial_qty_pct=0.1, threshold_base_pct=0.01 if ladder == "entry" else 0.5,
                             retracement_base_pct=0.0, double_down_factor=1.0,
                             threshold_we_weight=0.0, threshold_volatility_1h_weight=0.0,
                             threshold_volatility_1m_weight=0.0)
    strategy["close"].update(threshold_base_pct=(0.02 if close_on_tick else 0.02005) if is_close else 0.5,
                             retracement_base_pct=0.0, qty_pct=0.01,
                             threshold_we_weight=-0.1 if is_close else 0.0,
                             threshold_volatility_1h_weight=0.0,
                             threshold_volatility_1m_weight=0.0)
    # Keep both events away from warmup and the final trading bar.
    candles = np.tile(candles[:1], (12, 1, 1))
    candles[:, :, :3] = 100.0
    btc = np.full(12, 50_000.0)
    timestamps = timestamps[0] + np.arange(12, dtype=np.int64) * 60_000
    # Keep later partial top-ups out of the close-ladder trace.
    if is_close:
        config["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 100.0
    short = side == "short"
    candles[4, :, 0 if short else 1] = (
        101.4 if short else 98.6
    ) if ladder == "entry" else (100.1 if short else 99.9)
    if is_close:
        candles[5, :, 1 if short else 0] = (
            (98.75 if short else 101.25) if close_on_tick
            else (98.65 if short else 101.35)
        )
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
    if is_close:
        expected_balance += (0.33 if close_on_tick else 0.36) * coin_count
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


@pytest.mark.parametrize("source_name", [
    "mps_trailing_martingale_source_py",
    "mps_trailing_martingale_multicoin_source_py",
])
def test_partial_top_up_at_power_of_two_checks_inward_spacing(source_name):
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    source = getattr(passivbot_rust, source_name)()
    def function(name):
        start = source.index("inline float " + name + "(")
        return source[start:source.index("\n}", start) + 2]
    probe = "#include <metal_stdlib>\nusing namespace metal;\n"
    probe += function("floor_step") + "\n" + function("aligned_qty_difference")
    probe += """
kernel void inward_spacing_probe(constant float* values, device float* output,
                                 uint b [[thread_position_in_grid]]) {
    output[b] = aligned_qty_difference(values[0], values[1], values[2]);
}
"""
    device = gpu_device()
    values = torch.tensor([0.125, 0.08537521, 1e-8], dtype=torch.float32, device=device)
    output = torch.empty(1, device=device)
    compile_shader(probe).inward_spacing_probe(values, output, threads=1)
    synchronize()
    expected = np.float32(3_962_479) * np.float32(1e-8)
    assert output.item() == float(expected)


@pytest.mark.parametrize("source_name", [
    "mps_trailing_martingale_source_py",
    "mps_trailing_martingale_multicoin_source_py",
])
@pytest.mark.parametrize("short", [False, True])
def test_positive_retracement_close_forms_one_relative_distance(source_name, short):
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    source = getattr(passivbot_rust, source_name)()
    if source_name == "mps_trailing_martingale_source_py":
        branch = source.index("} else if (cr > 0.0f)", source.index("float close_target ="))
        first = source.index("close_target =", branch)
        last = source.index(";", first) + 1
        expression = source[first:last].replace("s.pprice", "price").replace("is_long", "!short_side")
    else:
        branch = source.index("} else if (close_retracement > 0.0f)", source.index("float close_target ="))
        first = source.index("close_target =", branch)
        last = source.index(";", first) + 1
        expression = source[first:last].replace("pprice[c]", "price").replace("close_threshold", "ct").replace("close_retracement", "cr")
    probe = """#include <metal_stdlib>
using namespace metal;
kernel void retracement_close_probe(constant float* values, device float* output,
                                    uint b [[thread_position_in_grid]]) {
    float price=values[0], ct=values[1], cr=values[2], step=values[3];
    bool short_side=values[4]>0;
    float close_target;
""" + expression + """
    output[0] = short_side ? floor(close_target/step+1e-6f)
                          : ceil(close_target/step-1e-6f);
}
"""
    device = gpu_device()
    values = torch.tensor([6198.13, .00865352712571621, .006685107946395874, .01, short],
                          dtype=torch.float32, device=device)
    output = torch.empty(1, device=device)
    compile_shader(probe).retracement_close_probe(values, output, threads=1)
    synchronize()
    # Decimal reference for the same represented operands; the omitted
    # intermediate 1 +/- threshold rounding changes the long target by a tick.
    from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
    p, ct, cr, step = map(lambda value: Decimal(str(float(value))), values.cpu().tolist()[:4])
    distance = -ct + cr if short else ct - cr
    expected = ((p + p * distance) / step).to_integral_value(
        rounding=ROUND_FLOOR if short else ROUND_CEILING)
    assert output.item() == int(expected)


def _decimal_boundary_inputs(side, coin_count, *, price, balance, initial_pct, qty_step,
                             factor, threshold, low, high):
    config, candles, markets, _, timestamps = _fixture(side, coin_count, "initial")
    config["backtest"].update(starting_balance=balance, maker_fee_override=0.0,
                             taker_fee_override=0.0, market_order_slippage_pct=0.0)
    strategy = config["bot"][side]["strategy"]["trailing_martingale"]
    strategy["entry"].update(initial_qty_pct=initial_pct, threshold_base_pct=threshold,
        retracement_base_pct=0.0, double_down_factor=factor, threshold_we_weight=0.0,
        threshold_volatility_1h_weight=0.0, threshold_volatility_1m_weight=0.0)
    strategy["close"].update(threshold_base_pct=0.5, retracement_base_pct=0.0,
        threshold_we_weight=0.0, threshold_volatility_1h_weight=0.0,
        threshold_volatility_1m_weight=0.0)
    candles = np.tile(candles[:1], (12, 1, 1))
    candles[:, :, :3] = price
    short = side == "short"
    candles[4, :, 0 if short else 1] = price + .01 if short else price - .01
    candles[5, :, 0 if short else 1] = high if short else low
    for coin, market in markets.items():
        if coin != "__meta__":
            market.update(qty_step=qty_step, price_step=.01, min_qty=qty_step,
                          min_cost=0.0, last_valid_index=11)
    timestamps = timestamps[0] + np.arange(12, dtype=np.int64) * 60_000
    return config, candles, markets, np.full(12, 50_000.0), timestamps


def _compound_boundary_inputs(side, coin_count, kind, offset, coin_pin):
    inputs = _decimal_boundary_inputs(side, coin_count, price=43.2, balance=1000.0,
        initial_pct=.1, qty_step=.001, factor=1., threshold=.9, low=43.19, high=43.21)
    config, candles, markets, _, _ = inputs
    values = dict(threshold_base_pct=.1 + offset, retracement_base_pct=.09375,
        retracement_volatility_1h_weight=0., retracement_volatility_1m_weight=0.)
    target = config["bot"][side]["strategy"]["trailing_martingale"][kind]
    target.update(values)
    if kind == "entry":
        target["retracement_we_weight"] = 0.
    else:
        target["qty_pct"] = 1.
    if coin_pin != "none":
        fields = ("threshold_base_pct", "retracement_base_pct") if coin_pin == "both" else (coin_pin + "_base_pct",)
        patch = {key: values[key] for key in fields}
        for key in fields:
            target[key] = .2 if key == "threshold_base_pct" else .01
        config["coin_overrides"] = {
            coin: {"bot": {side: {"strategy": {"trailing_martingale": {kind: patch.copy()}}}}}
            for coin in markets if coin != "__meta__"
        }
    if (kind == "close") != (side == "short"):
        candles[5, :, :3] = [50., 49., 49.]
        candles[6:, :, :3] = [43.475, 43., 43.47]
    else:
        candles[5, :, :3] = [38.6, 38.5, 38.55]
        candles[6:, :, :3] = [42.95, 42.925, 42.935]
    return inputs


@pytest.mark.parametrize("kind", ["entry", "close"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("offset", [0., -1e-12, 1e-12])
@pytest.mark.parametrize("coin_pin", ["none", "both", "threshold", "retracement"])
def test_compound_trailing_replay_preserves_authored_decimal_ticks(kind, side, coin_count, offset, coin_pin):
    out, fills = _evaluate(side, _compound_boundary_inputs(side, coin_count, kind, offset, coin_pin))
    assert out["fill_count"].item() == len(fills)
    exact = abs(sum(float(fill[9]) for fill in fills))
    assert out["short_psize" if side == "short" else "psize"].item() == pytest.approx(exact, abs=3e-6)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("coin_pin", [False, True])
@pytest.mark.parametrize("factor,initial_pct,qty_step,balance,expected", [
    (1.5, .0037, .001, 1000.0, .093),
    (1.15, .05, .1, 10000.0, 10.8),
    (1.15 - 1e-12, .05, .1, 10000.0, 10.7),
    (1.15 + 1e-12, .05, .1, 10000.0, 10.8),
])
def test_decimal_reentry_half_steps_match_rust_without_snapping(
    side, coin_count, coin_pin, factor, initial_pct, qty_step, balance, expected
):
    inputs = _decimal_boundary_inputs(side, coin_count, price=100.0, balance=balance,
        initial_pct=initial_pct, qty_step=qty_step, factor=factor, threshold=.01,
        low=98.9, high=101.1)
    if coin_pin:
        config, _, markets, _, _ = inputs
        config["bot"][side]["strategy"]["trailing_martingale"]["entry"]["double_down_factor"] = 0.25
        config["coin_overrides"] = {
            coin: {"bot": {side: {"strategy": {"trailing_martingale": {
                "entry": {"double_down_factor": factor}}}}}}
            for coin in markets if coin != "__meta__"
        }
    out, fills = _evaluate(side, inputs)
    assert len(fills) == coin_count * 2
    exact = abs(sum(float(fill[9]) for fill in fills))
    assert exact == pytest.approx(expected * coin_count, abs=1e-10)
    assert out["short_psize" if side == "short" else "psize"].item() == pytest.approx(exact, abs=2e-6)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("offset", [0.0, -1e-12, 1e-12])
@pytest.mark.parametrize("coin_pin", [False, True, "zeroed_modifiers"])
def test_canonical_entry_ticks_keep_authored_off_grid_distances(side, coin_count, offset, coin_pin):
    inputs = _decimal_boundary_inputs(side, coin_count, price=2.5, balance=100.0,
        initial_pct=.025, qty_step=.001, factor=1.0, threshold=.044 + offset,
        low=2.385, high=2.615)
    if coin_pin:
        config, _, markets, _, _ = inputs
        config["bot"][side]["strategy"]["trailing_martingale"]["entry"]["threshold_base_pct"] = 0.5
        config["coin_overrides"] = {
            coin: {"bot": {side: {"strategy": {"trailing_martingale": {
                "entry": {"threshold_base_pct": .044 + offset}}}}}}
            for coin in markets if coin != "__meta__"
        }
        if coin_pin == "zeroed_modifiers":
            entry = config["bot"][side]["strategy"]["trailing_martingale"]["entry"]
            entry.update(threshold_we_weight=0.25, threshold_volatility_1h_weight=1e-100)
            for override in config["coin_overrides"].values():
                override["bot"][side]["strategy"]["trailing_martingale"]["entry"].update(
                    threshold_we_weight=0.0, threshold_volatility_1h_weight=0.0)
    out, fills = _evaluate(side, inputs)
    expected_count = (1 if offset > 0 else 2) * coin_count
    assert len(fills) == expected_count
    assert out["fill_count"].item() == expected_count
    exact = abs(sum(float(fill[9]) for fill in fills))
    assert out["short_psize" if side == "short" else "psize"].item() == pytest.approx(exact, abs=2e-6)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("coin_pin", [False, True])
def test_decimal_price_proof_declines_nonzero_sub_ulp_remainders(side, coin_count, coin_pin):
    threshold = .025626895304925
    inputs = _decimal_boundary_inputs(side, coin_count, price=6532.59,
        balance=100000.0, initial_pct=.025, qty_step=.001, factor=1.0,
        threshold=threshold, low=6365.175, high=6700.005)
    if coin_pin:
        config, _, markets, _, _ = inputs
        config["bot"][side]["strategy"]["trailing_martingale"]["entry"]["threshold_base_pct"] = .5
        config["coin_overrides"] = {
            coin: {"bot": {side: {"strategy": {"trailing_martingale": {
                "entry": {"threshold_base_pct": threshold}}}}}}
            for coin in markets if coin != "__meta__"
        }
    out, fills = _evaluate(side, inputs)
    assert len(fills) == coin_count * 2
    # The decimal unit products straddle these integers by less than a f64 ULP;
    # Rust declines decimal alignment and the fused f64 target lands on the tick.
    reentries = fills[coin_count:]
    assert all(float(fill[10]) == (6700.0 if side == "short" else 6365.18)
               for fill in reentries)
    assert out["fill_count"].item() == len(fills)
    exact = abs(sum(float(fill[9]) for fill in fills))
    assert out["short_psize" if side == "short" else "psize"].item() == pytest.approx(exact, abs=2e-6)


def test_decimal_metadata_device_proofs_preserve_large_and_near_half_products():
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    source = passivbot_rust.mps_trailing_martingale_source_py()
    first = source.index("struct GpuDecimal {")
    last = source.index("// PASSIVBOT_DECIMAL_ROUNDING_END", first)
    probe = "#include <metal_stdlib>\nusing namespace metal;\n" + source[first:last] + """
kernel void decimal_proof_probe(constant float* cases, constant float* metadata,
    device float* output, uint b [[thread_position_in_grid]]) {
    GpuDecimal value = load_gpu_decimal(metadata, int(b) * 4);
    output[b] = float(round_gpu_decimal_product(uint(cases[b]), value, true, false));
}
"""
    factors = [1.15, 1.15 - 1e-12, 1.15 + 1e-12, 1.5,
               .12345674999999999, 1e-100, -1.15]
    counts = [50, 50, 50, 37, 10_000_000, 37, 37]
    expected = [58, 57, 58, 56, 1_234_567, -1, -1]
    device = gpu_device()
    packed = torch.tensor(passivbot_rust.gpu_decimal_metadata_py(factors), dtype=torch.float32, device=device)
    assert torch.isfinite(packed).all().item()
    values = torch.tensor(counts, dtype=torch.float32, device=device)
    output = torch.empty(len(counts), device=device)
    compile_shader(probe).decimal_proof_probe(values, packed, output, threads=len(counts))
    synchronize()
    assert output.cpu().tolist() == expected


def test_tm_decimal_packing_declines_original_nonzero_modifiers():
    from optimization.gpu.mps_kernel import _pack_tm_parameter_matrix, _TM_DECIMAL_WORDS
    from optimization.gpu.model import TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS as keys
    from test_gpu_mps import _tm_single_row

    params = np.asarray([_tm_single_row() * 2], dtype=np.float64)
    for side in range(2):
        for column, key in enumerate(keys):
            if key.endswith("_weight"):
                params[:, side * len(keys) + column] = 0.0
    # Underflow in the public float32 column does not license a static proof.
    params[0, keys.index("entry_threshold_we_weight")] = 1e-100
    params[0, len(keys) + keys.index("close_retracement_volatility_1m_weight")] = 1e-100
    packed = _pack_tm_parameter_matrix(params, keys, sides=2)
    width = len(keys) + _TM_DECIMAL_WORDS
    assert packed[0, keys.index("entry_threshold_we_weight")] == 0.0
    assert packed[0, len(keys) + 20:len(keys) + 24].tolist() == [1, 0, 0, 0]
    assert packed[0, width + len(keys) + 20:width + len(keys) + 24].tolist() == [0, 0, 0, 2]


def test_decimal_device_masks_apply_zero_pins_before_static_price_proof():
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize

    directional = passivbot_rust.mps_trailing_martingale_source_py()
    first = directional.index("struct GpuDecimal {")
    last = directional.index("// PASSIVBOT_DECIMAL_ROUNDING_END", first)
    multicoin = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    start = multicoin.index("inline bool tm_static_decimal_distance(")
    end = multicoin.index("inline GpuDecimal tm_decimal_override(", start)
    probe = ("#include <metal_stdlib>\nusing namespace metal;\nconstant int OVERRIDE_COLS = 73;\n"
             + directional[first:last] + multicoin[start:end] + """
kernel void zero_pin_probe(constant float* overrides, constant float* metadata,
    device float* output, uint b [[thread_position_in_grid]]) {
    bool proven = tm_static_decimal_distance(3u, overrides, int(b), 0);
    output[b] = proven ? float(canonical_gpu_relative_ticks(250,
        load_gpu_decimal(metadata, 0), invalid_gpu_decimal(), true, false, false)) : -1;
}
""")
    device = gpu_device()
    overrides = np.zeros((3, 73), dtype=np.float32)
    overrides[:, 69] = [0, 3, 39]  # Inherited, pinned zero, pinned tiny nonzero.
    packed = torch.tensor(overrides, device=device)
    metadata = torch.tensor(passivbot_rust.gpu_decimal_metadata_py([.044]),
                            dtype=torch.float32, device=device)
    output = torch.empty(3, device=device)
    compile_shader(probe).zero_pin_probe(packed, metadata, output, threads=3)
    synchronize()
    assert output.cpu().tolist() == [-1, 239, -1]


def test_compound_cancellation_declines_unproven_decimal_corrections():
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize

    source = passivbot_rust.mps_trailing_martingale_source_py()
    first = source.index("struct GpuDecimal {")
    last = source.index("// PASSIVBOT_DECIMAL_ROUNDING_END", first)
    probe = ("#include <metal_stdlib>\nusing namespace metal;\n" + source[first:last] + """
kernel void compound_fallback_probe(constant float* values, constant float* metadata,
    device float* output, uint b [[thread_position_in_grid]]) {
    bool decreasing = b == 1;
    float base = values[b * 4], step = values[b * 4 + 1];
    float threshold = values[b * 4 + 2], retracement = values[b * 4 + 3];
    int proven = canonical_gpu_relative_ticks(decreasing ? 1120 : 4320,
        load_gpu_decimal(metadata, int(b) * 8), load_gpu_decimal(metadata, int(b) * 8 + 4),
        decreasing, true, !decreasing);
    float distance = decreasing ? -threshold + retracement : threshold - retracement;
    float target = fma(base, distance, base);
    output[b * 2] = float(proven);
    output[b * 2 + 1] = decreasing ? floor(target / step) : ceil(target / step);
}
""")
    device = gpu_device()
    values = torch.tensor([[43.2, .01, .01, .0037499999999999994],
                           [11.2, .01, .01, .0037499999999999994],
                           [43.2, .01, .05788483791847438, .007884837918474379]],
                          dtype=torch.float32, device=device)
    metadata = torch.tensor(passivbot_rust.gpu_decimal_metadata_py(
        [.01, .0037499999999999994, .01, .0037499999999999994,
         .05788483791847438, .007884837918474379]),
        dtype=torch.float32, device=device)
    output = torch.empty(6, device=device)
    compile_shader(probe).compound_fallback_probe(values, metadata, output, threads=3)
    synchronize()
    # The first two rows require unsupported exponent -19. The last row has
    # valid metadata but a nonzero residual inside the f64 ambiguity envelope.
    # Neither proves a correction; physical float32 fallback remains approximate
    # (4347/1113/4536 ticks), not the Rust f64 result 4348/1112/4537.
    assert output.cpu().tolist() == [-1, 4347, -1, 1113, -1, 4536]
