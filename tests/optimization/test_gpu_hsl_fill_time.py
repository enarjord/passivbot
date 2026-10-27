"""Execution-mark recovery is observed before final-fill cooldown qualification."""

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
    config['bot']["portfolio"]['strategy_kind'] = strategy_kind
    config['bot']["portfolio"]['execution']['market_orders_allowed'] = True
    config['bot']["portfolio"]['execution']['market_order_near_touch_fraction'] = 0.0
    # A power-of-two balance and integer fills make each turnover contribution
    # exactly representable in f32, retaining the strict clock/recovery comparison.
    config["backtest"].update(starting_balance=1024., hsl_detailed_report=False,
                              market_order_slippage_fraction=0.,
                              maker_fee_override_fraction=0., taker_fee_override_fraction=0.)
    for direction in ("long", "short"):
        policy = config["bot"][direction]["hsl"]
        policy.update(enabled=direction == side and mode != "portfolio",
                      drawdown_threshold_fraction=.01, drawdown_fraction_ema_span_minutes=1.,
                      cooldown_after_red_minutes=5.,
                      panic_close_order_type="market")
        strategy = config["bot"][direction]["strategy"]["trailing_martingale"]
        strategy["entry"].update(initial_qty_fraction=.5, threshold_base_fraction=.5,
                                 retracement_base_fraction=0.)
        strategy["close"].update(threshold_base_fraction=.5, retracement_base_fraction=0.)
        config["bot"][direction]["strategy"]["ema_anchor"].update(
            base_qty_fraction=.5, ema_span_0_minutes=2., ema_span_1_minutes=3., offset_fraction=.01,
            inventory_offset_exposure_coefficient=0., entry_qty_exposure_coefficient=0.,
        )
    if mode == "portfolio":
        # Preserve the portfolio scope selector while copying the fixture policy.
        config["bot"]["portfolio"]["hsl"].update(
            deepcopy(config["bot"][side]["hsl"]), enabled=True)
    count = 16
    candles = np.full((count, 1, 4), 100., dtype=np.float64)
    candles[:, :, 3] = 1.
    if strategy_kind == "ema_anchor":
        candles[3, :, 0] = 101.
        candles[3, :, 1] = 99.
    # The close sees RED; the next market execution recovers to its basis.
    # The execution-price mark recovers before the economic fill observation;
    # this GREEN pre-fill sample does not qualify a new cooldown.
    adverse = 80. if side == "long" else 120.
    candles[5, :, :3] = adverse
    candles[5, :, 0] = max(100., adverse)
    candles[5, :, 1] = min(100., adverse)
    if strategy_kind == "ema_anchor":
        # EMA entries retain their 1% resting offset. Recover exactly to that
        # entry basis so the panic fill has zero PnL and preserves the balance.
        candles[6:, :, :3] = 99. if side == "long" else 101.
    timestamps = timestamps[0] + np.arange(count, dtype=np.int64) * 60_000
    btc = np.full(count, 50_000.)
    markets["BTC"].update(last_valid_index=count - 1, qty_step=1., min_qty=1., price_step=1.)
    return config, candles, markets, btc, timestamps


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_recovered_panic_execution_mark_has_no_unobserved_red_minute(side, mode, strategy_kind):
    from backtest import build_backtest_payload, execute_backtest
    from config.metrics import resolve_metric_value
    from optimization.gpu.parameters import prepare_candidate_parameters
    from optimization.gpu.service import SharedAccountReplay
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    config, candles, markets, btc, timestamps = _recovered_panic_inputs(side, mode, strategy_kind)
    payload = build_backtest_payload(candles, markets, deepcopy(config), "bybit", btc,
                                     timestamps, metrics_only=False, metric_requests=["gross_turnover_ratio_per_day", "hsl_triggers_per_year", "hsl_red_time_fraction"])
    fills, _, analysis = execute_backtest(payload, config)
    report = payload.hard_stop_plot_data["hsl"]
    assert report["summary"]["triggers"] > 0
    assert report["summary"]["red_minutes"] == 0
    assert report["summary"]["observed_minutes"] > 0
    panic_fills = [fill for fill in fills if str(fill[13]).startswith("close_panic_")]
    assert panic_fills
    assert all(int(fill[0]) == 6 and int(fill[1]) == int(timestamps[6]) for fill in panic_fills)
    assert all(float(fill[3]) == float(fill[4]) == 0. and float(fill[5]) == 1024.
               and float(fill[9]).is_integer() and float(fill[10]).is_integer() for fill in fills)
    metrics = {"gross_turnover_ratio_per_day", "hsl_triggers_per_year", "hsl_red_time_fraction"}
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="bybit", batch_size=1,
        needed_metrics=metrics, factual_hsl=True)
    actual = proxy.evaluate([prepare_candidate_parameters(config, markets, "bybit")])[0]
    assert actual["hsl_red_time_fraction"] == 0.
    for name in metrics:
        expected = resolve_metric_value(analysis, name)
        assert actual[name] == pytest.approx(expected, abs=0., rel=8*np.finfo(float).eps), name
