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
        hsl_engine="legacy",
        approved_coins={"long": coins, "short": coins},
    )
    config["backtest"].update(
        coins={"bybit": coins}, exchanges=["bybit"], starting_balance=1000.0
    )
    for direction in ("long", "short"):
        bot = config["bot"][direction]
        bot["hsl"]["enabled"] = False
        bot["unstuck"]["enabled"] = False
        bot["risk"].update(
            n_positions=coin_count if direction == side else 0,
            total_wallet_exposure_limit=float(coin_count) if direction == side else 0.0,
            we_excess_allowance_pct=0.0,
            entry_cooldown_minutes=0.0,
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
            qty_step=0.001, price_step=1.0, min_qty=0.001,
            min_cost=100.0 if entry_kind == "minimum" else 0.0,
            c_mult=1.0, maker=0.0, taker=0.0, exchange="bybit",
            first_valid_index=0, last_valid_index=count - 1, warmup_minutes=1,
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
        config=config, hlcvs=candles, mss=mss, btc=btc, timestamps=timestamps,
        exchange="bybit", batch_size=1, needed_metrics={"adg_strategy_eq"},
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
        ema_gate_mode="all", initial_ema_dist=0.1,
        ema_span_0=1000.0, ema_span_1=1000.0,
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
