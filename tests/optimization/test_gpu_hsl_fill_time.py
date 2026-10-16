"""Flat-fill accounting can retire RED before the following bar interval."""

from copy import deepcopy

import numpy as np
import pytest

from test_gpu_entry_sizing_parity import _fixture

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU required",
)


def _recovered_panic_inputs(side, mode, strategy_kind="trailing_martingale"):
    from config.hsl import generated_template

    config, candles, markets, btc, timestamps = _fixture(side, 1, "initial")
    config = generated_template(config, mode)
    config["live"].update(strategy_kind=strategy_kind, hsl_signal_mode=mode, market_orders_allowed=True,
                          market_order_near_touch_threshold=0.0)
    config["backtest"].update(hsl_detailed_report=False, market_order_slippage_pct=0.,
                              maker_fee_override=0., taker_fee_override=0.)
    for direction in ("long", "short"):
        policy = config["bot"][direction]["hsl"]
        policy.update(enabled=direction == side and mode != "unified",
                      red_threshold=.01, ema_span_minutes=1.,
                      cooldown_minutes_after_red=5., restart_after_red_policy="always",
                      panic_close_order_type="market")
        strategy = config["bot"][direction]["strategy"]["trailing_martingale"]
        strategy["entry"].update(initial_qty_pct=.5, threshold_base_pct=.5,
                                 retracement_base_pct=0.)
        strategy["close"].update(threshold_base_pct=.5, retracement_base_pct=0.)
        config["bot"][direction]["strategy"]["ema_anchor"].update(
            base_qty_pct=.5, ema_span_0=2., ema_span_1=3., offset=.01,
            offset_psize_weight=0., entry_double_down_factor=0.,
        )
    if mode == "unified":
        config["bot"]["hsl"] = deepcopy(config["bot"][side]["hsl"])
        config["bot"]["hsl"]["enabled"] = True
    count = 16
    candles = np.full((count, 1, 4), 100., dtype=np.float64)
    candles[:, :, 3] = 1.
    if strategy_kind == "ema_anchor":
        candles[3, :, 0] = 101.
        candles[3, :, 1] = 99.
    # The close sees RED; the next market execution recovers to its basis.
    # Those observations share an accounting timestamp, so terminal GREEN
    # must not acquire an entire subsequent RED minute.
    adverse = 80. if side == "long" else 120.
    candles[5, :, :3] = adverse
    candles[5, :, 0] = max(100., adverse)
    candles[5, :, 1] = min(100., adverse)
    timestamps = timestamps[0] + np.arange(count, dtype=np.int64) * 60_000
    btc = np.full(count, 50_000.)
    markets["BTC"]["last_valid_index"] = count - 1
    return config, candles, markets, btc, timestamps


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_recovered_panic_fill_has_no_unobserved_red_minute(side, mode, strategy_kind):
    from backtest import build_backtest_payload, execute_backtest
    from config.metrics import resolve_metric_value
    from optimization.gpu.parameters import prepare_candidate_parameters
    from optimization.gpu.service import MpsMulticoinProxy
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    config, candles, markets, btc, timestamps = _recovered_panic_inputs(side, mode, strategy_kind)
    payload = build_backtest_payload(candles, markets, deepcopy(config), "bybit", btc,
                                     timestamps, metrics_only=False, skip_btc_analysis=True)
    _, _, analysis = execute_backtest(payload, config)
    report = payload.hard_stop_plot_data["hsl"]
    assert report["summary"]["triggers"] > 0
    assert report["summary"]["red_minutes"] == 0
    assert report["summary"]["observed_minutes"] > 0
    metrics = {"fills_per_day", "hard_stop_triggers_per_year", "hard_stop_time_in_red_pct"}
    proxy = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="bybit", batch_size=1,
        needed_metrics=metrics, factual_hsl=True)
    actual = proxy.evaluate([prepare_candidate_parameters(config, markets, "bybit")])[0]
    assert actual["hard_stop_time_in_red_pct"] == 0.
    for name in metrics:
        expected = resolve_metric_value(analysis, name)
        assert actual[name] == pytest.approx(expected, abs=0., rel=8*np.finfo(float).eps), name
