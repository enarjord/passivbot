"""The actual GPU observer shares native finalization and causal sample rules."""

from functools import lru_cache
import math

import numpy as np
import pytest


def _native():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return passivbot_rust


def test_observation_packet_rejects_nonexact_integer_clocks():
    native = _native()
    fields = native.get_core_gpu_observation_catalog()["fields"]
    packet = np.zeros((1, len(fields)), dtype=np.float32)
    for name in ("start_step", "end_step", "minute_count", "fill_count"):
        malformed = packet.copy()
        malformed[0, fields.index(name)] = np.float32(2**24 + 2)
        with pytest.raises(ValueError, match="invalid GPU integer observation"):
            native.core_metrics_from_gpu_observations(
                malformed, ["backtest_completion_fraction"], 0, 60_000, 60_000, 180.0)


def _flat_packet():
    fields = _native().get_core_gpu_observation_catalog()["fields"]
    result = np.zeros((1, len(fields)), dtype=np.float32)
    for name, value in {"end_step": 1, "initial_equity": 1000, "last_equity": 1000,
                        "minute_count": 2, "exposure_count": 2,
                        "daily_first_step": -1, "daily_last_step": -1}.items():
        result[0, fields.index(name)] = value
    return result, fields


def test_inactive_effective_hsl_policy_uses_verified_elapsed_coverage():
    native = _native()
    packet, _ = _flat_packet()
    names = ["hsl_red_time_fraction", "hsl_triggers_per_year"]
    row = native.core_metrics_from_gpu_observations(packet, names, 0, 60_000, 60_000, 180.0, [True])[0]
    assert all(row[name] == 0 for name in names)
    assert not row["metric_status"]
    with pytest.raises(ValueError, match="flag count"):
        native.core_metrics_from_gpu_observations(packet, names, 0, 60_000, 60_000, 180.0, [])


@pytest.mark.parametrize("episodes", [0, 1, 3])
def test_hsl_execution_count_packet_distinguishes_triggers_and_zero_loss_fills(episodes):
    packet, fields = _flat_packet()
    for name, value in {"hsl_enabled": 1, "hsl_observed_steps": 1,
                        "hsl_trigger_count": 3, "hsl_episode_count": episodes,
                        "hsl_episode_invalid": 1}.items():
        packet[0, fields.index(name)] = value
    name = "hsl_execution_episode_count"
    row = _native().core_metrics_from_gpu_observations(
        packet, [name], 0, 60_000, 60_000)[0]
    assert row[name] == episodes
    assert row["metric_metadata"][name] == episodes
    assert not row["metric_status"]


@pytest.mark.parametrize("episodes", [math.nan, -1., .5, float(2**24 + 2)])
def test_hsl_execution_count_packet_rejects_invalid_count_without_hiding_growth(episodes):
    packet, fields = _flat_packet()
    for name, value in {"hsl_enabled": 1, "hsl_observed_steps": 1,
                        "hsl_episode_count": episodes}.items():
        packet[0, fields.index(name)] = value
    name = "hsl_execution_episode_count"
    row = _native().core_metrics_from_gpu_observations(
        packet, [name, "cagr_fraction_strategy_equity"], 0, 60_000, 60_000)[0]
    assert row[name] is None
    assert row["metric_status"][name]["status"] == "invalid"
    assert row["cagr_fraction_strategy_equity"] == 0


def test_hsl_counter_overflow_does_not_invalidate_unrelated_cagr():
    native = _native()
    packet, fields = _flat_packet()
    for name, value in {"hsl_enabled": 1, "hsl_observed_steps": 1,
                        "hsl_trigger_count": math.nan}.items():
        packet[0, fields.index(name)] = value
    row = native.core_metrics_from_gpu_observations(
        packet, ["cagr_fraction_strategy_equity", "hsl_triggers_per_year"], 0, 60_000, 60_000)[0]
    assert row["cagr_fraction_strategy_equity"] == 0
    assert row["hsl_triggers_per_year"] is None
    assert row["metric_status"]["hsl_triggers_per_year"]["status"] == "invalid"


_PROBE = r"""
kernel void core_observer_probe(constant float* equity, device float* output,
    constant int& count, constant int& terminal_replace,
    constant float& turnover_notional, uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    CoreGpuState s = init_core_gpu(equity[0], 0);
    for (int i = 1; i < count; ++i) {
        if (i == 1 && turnover_notional > 0.0f) {
            s.pre_fill_balance = 1000.0f;
            core_record_fill(s, i, -2.0f, true);
            core_record_notional(s, turnover_notional);
            s.pre_fill_balance = 998.0f;
            core_record_fill(s, i, 10.0f, true);
            core_record_notional(s, 0.0f);
        }
        core_observe_equity(s, equity[i], i, 0, 0.0f, 0.0f, 0.0f, false);
    }
    if (terminal_replace != 0)
        core_observe_equity(s, equity[0], count - 1, 0, 0.0f, 0.0f, 0.0f, true);
    s.net_profit_long = s.pending_equity - equity[0];
    write_core_gpu(s, output, 0, 0);
}
"""


@lru_cache(maxsize=4)
def _library(names):
    native = _native()
    from optimization.gpu.core_metrics import metric_mask, with_core_metrics
    from optimization.gpu.mps_kernel import _with_hsl
    from optimization.gpu.runtime import compile_shader

    source = ("#define PASSIVBOT_HSL_LOOKBACK 0\n"
              + _with_hsl(native.mps_ema_anchor_multicoin_source_py(), 1) + _PROBE)
    return compile_shader(with_core_metrics(source, metric_mask(names), 180.0),
                          mps_coin_capacity=2, cuda_coin_capacity=2)


def _probe(equity, names, *, terminal_replace=False, turnover_notional=0.0):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("actual supported GPU required")
    from optimization.gpu.core_metrics import observation_catalog, finalize_observations
    from optimization.gpu.runtime import gpu_device

    device = gpu_device()
    values = torch.tensor(equity, dtype=torch.float32, device=device)
    packet = torch.empty((1, len(observation_catalog()["fields"])), device=device)
    _library(tuple(sorted(names))).core_observer_probe(
        values, packet, len(equity), int(terminal_replace), float(turnover_notional), threads=1)
    return finalize_observations(packet, names, origin_timestamp_ms=0,
                                 observation_interval_ms=60_000,
                                 intended_end_timestamp_ms=(len(equity) - 1) * 60_000,
                                 ew_half_life_days=180.0)[0]


def test_minute_path_includes_baseline_and_pending_terminal_once():
    names = ["drawdown_max_fraction_strategy_equity", "ulcer_index_fraction_strategy_equity",
             "pain_index_fraction_strategy_equity", "peak_recovery_max_days_strategy_equity"]
    row = _probe([1000, 900, 1000, 950], names)
    assert row[names[0]] == pytest.approx(.1, abs=1e-7)
    assert row[names[1]] == pytest.approx(math.sqrt(.0125 / 4), abs=1e-7)
    assert row[names[2]] == pytest.approx(.15 / 4, abs=1e-7)
    assert row[names[3]] == pytest.approx(2 / 1440)
    assert row["metric_metadata"]["equity_observation_count"] == 4
    assert row["metric_metadata"]["peak_recovery_tail_censored"] is True
    assert "cagr_fraction_strategy_equity" not in row


@pytest.mark.parametrize("curve,censored", [([1000, 900, 1000], False),
                                           ([1000, 900, 950], True),
                                           ([1000, 1000, 1000], False)])
def test_recovery_tail_metadata_reports_observed_terminal_state(curve, censored):
    row = _probe(curve, ["peak_recovery_max_days_strategy_equity"])
    assert row["metric_metadata"]["peak_recovery_tail_censored"] is censored
    cagr = _probe(curve, ["cagr_fraction_strategy_equity"])
    assert "peak_recovery_tail_censored" not in cagr["metric_metadata"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_forced_delist_loss_is_not_an_hsl_execution_episode(strategy):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("actual supported GPU required")
    from backtest import build_backtest_payload, execute_backtest
    from optimization.gpu.service import SharedAccountReplay
    from test_gpu_entry_sizing_parity import _fixture

    config, _, markets, _, timestamps = _fixture("long", 1, "initial")
    config["bot"]["portfolio"]["strategy_kind"] = strategy
    config["bot"]["long"]["hsl"].update(enabled=True, drawdown_threshold_fraction=.9)
    config["bot"]["long"]["strategy"]["ema_anchor"].update(
        base_qty_fraction=.5, ema_span_0_minutes=2., ema_span_1_minutes=3.,
        offset_fraction=.01, inventory_offset_exposure_coefficient=0.,
        entry_qty_exposure_coefficient=0.)
    count, last_valid = 1410, 6
    candles = np.full((count, 1, 4), np.nan)
    candles[:last_valid + 1] = [100., 100., 100., 1.]
    candles[3, 0] = [100., 98., 100., 1.]
    candles[last_valid, 0] = [90., 90., 90., 1.]
    markets["BTC"]["last_valid_index"] = last_valid
    timestamps = timestamps[0] + np.arange(count, dtype=np.int64) * 60_000
    btc = np.full(count, 50_000.)
    names = ["hsl_triggers_per_year", "hsl_red_time_fraction",
             "hsl_panic_episode_loss_max_fraction", "hsl_panic_episode_loss_mean_fraction"]
    payload = build_backtest_payload(candles, markets, config, "bybit", btc, timestamps,
                                     metric_requests=names)
    fills, _, expected = execute_backtest(payload, config)
    assert len(fills) >= 2
    assert any("close_panic" in str(fill[13]) and float(fill[3]) < 0 for fill in fills)
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="bybit", batch_size=1, needed_metrics=names)
    actual = proxy.evaluate_results([{}])[0]
    assert actual.metrics == {name: 0. for name in names}
    assert {name: expected[name] for name in names} == actual.metrics
    assert actual.metadata["metric_metadata"]["hsl_execution_episode_count"] == 0
    assert expected["metric_metadata"]["hsl_execution_episode_count"] == 0


def test_complete_utc_daily_returns_use_corrected_sortino_and_omega():
    names = ["volatility_annual_fraction_strategy_equity", "sharpe_ratio_strategy_equity",
             "sortino_ratio_strategy_equity", "omega_ratio_strategy_equity"]
    curve = np.interp(np.arange(4321), [0, 1440, 2880, 4320], [1000, 1100, 880, 968])
    row = _probe(curve, names)
    returns = np.array([.1, -.2, .1])
    assert row[names[0]] == pytest.approx(returns.std(ddof=1) * math.sqrt(365.25), rel=2e-6)
    assert row[names[1]] == pytest.approx(0, abs=2e-6)
    assert row[names[2]] == pytest.approx(0, abs=2e-6)
    assert row[names[3]] == pytest.approx(1, rel=2e-6)
    assert row["metric_metadata"]["daily_return_count"] == 3


def test_daily_zero_denominators_keep_infinity_and_undefined_statuses():
    names = ["volatility_annual_fraction_strategy_equity", "sharpe_ratio_strategy_equity",
             "sortino_ratio_strategy_equity", "omega_ratio_strategy_equity"]
    rising = np.interp(np.arange(4321), [0, 1440, 2880, 4320], [1000, 2000, 4000, 8000])
    row = _probe(rising, names)
    assert row[names[0]] == 0
    for name in names[1:]:
        assert row[name] is None
        assert row["metric_status"][name]["status"] == "positive_infinity"
    flat = _probe(np.full(4321, 1000), names)
    assert flat[names[0]] == 0
    assert all(flat["metric_status"][name]["status"] == "undefined" for name in names[1:])


def test_fill_terminal_at_midnight_preserves_completed_daily_close():
    name = "omega_ratio_strategy_equity"
    row = _probe(np.linspace(1000, 1100, 1441), [name], terminal_replace=True)
    assert row[name] is None
    assert row["metric_status"][name]["status"] == "positive_infinity"
    assert row["metric_metadata"]["daily_return_count"] == 1
    assert row["metric_metadata"]["equity_observation_count"] == 1441


def test_turnover_uses_prefill_raw_balance_and_all_elapsed_days():
    names = ["gross_turnover_ratio_per_day", "loss_profit_ratio"]
    row = _probe(np.full(2881, 1000), names, turnover_notional=1200)
    assert row[names[0]] == pytest.approx(.6, rel=1e-6)
    assert row[names[1]] == pytest.approx(.2, rel=1e-6)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_selected_metrics_and_temporal_chunks_preserve_raw_economics(strategy):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("actual supported GPU required")
    _native()
    from config.metrics import CORE_METRIC_CATALOG
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import build_parser, fixture_inputs

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--coins", "2", "--sides", "both", "--bars", "128",
    ]))
    kwargs = dict(config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
                  exchange="binance", batch_size=3, factual_hsl=True)
    all_metrics = SharedAccountReplay(**kwargs, needed_metrics=set(CORE_METRIC_CATALOG))
    cagr = "cagr_fraction_strategy_equity"
    selected = SharedAccountReplay(**kwargs, needed_metrics={cagr})
    candidates = [{}, {}, {}]
    expected = all_metrics.evaluate_results(candidates)
    actual = selected.evaluate_results(candidates)
    assert all(row.metrics == {cagr: expected[i].metrics[cagr]} for i, row in enumerate(actual))
    all_runner, selected_runner = all_metrics.fused_runner, selected.fused_runner

    def raw(proxy, runner):
        matrix = np.concatenate([proxy._parameter_matrix(candidates, side)
                                 for side in ("long", "short")], axis=1)
        return {name: value.clone() for name, value in runner.run(matrix).items()
                if isinstance(value, torch.Tensor)}

    full = raw(all_metrics, all_runner)
    minimal = raw(selected, selected_runner)
    for name, value in full.items():
        if name != "core_observations":
            torch.testing.assert_close(value, minimal[name], rtol=0, atol=0, equal_nan=True)
    all_runner.max_dispatch_candidate_bars = 128
    chunked = raw(all_metrics, all_runner)
    assert all_runner._last_temporal_dispatch["dispatch_count"] > 1
    for name, value in full.items():
        torch.testing.assert_close(value, chunked[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_fill_terminal_ignores_future_close_of_another_held_coin(strategy):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("actual supported GPU required")
    _native()
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu.core_metrics import metric_mask, observation_catalog
    from optimization.gpu.model import (
        ProxyMarket, EMA_ANCHOR_MULTICOIN_PARAM_KEYS, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    )

    observations = []
    for future_close in (91.0, 99.0):
        closes = np.full((12, 2), 100.0)
        highs, lows = closes.copy(), closes.copy()
        lows[3, 0] = 49.0
        closes[5, 0] = highs[5, 0] = lows[5, 0] = 120.0
        closes[6, 0], highs[6, 0], lows[6, 0] = future_close, 120.0, 90.0
        lows[6, 1] = 49.0
        # The late coin's deliberately extreme synthetic fee depletes cash at
        # its entry. The first coin remains held; its bar-six close is unknown.
        runner, row = _multicoin_exposure_fixture(
            strategy, "long", count=12, closes=closes, highs=highs, lows=lows,
            first_valid_indices=(0, 4),
            markets=[ProxyMarket(.001, .01, .001, 0, 1, 0),
                     ProxyMarket(.001, .01, .001, 0, 1, 10)],
        )
        keys = (EMA_ANCHOR_MULTICOIN_PARAM_KEYS if strategy == "ema_anchor"
                else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
        values = {"entry_cooldown_minutes": 999.0}
        values.update({"base_qty_fraction": .25, "offset_fraction": .5}
                      if strategy == "ema_anchor" else {
                          "entry_initial_qty_fraction": .25, "entry_initial_ema_distance_fraction": .5,
                          "close_threshold_base_fraction": 1.5, "entry_reentry_qty_multiplier": 0.0})
        for name, value in values.items():
            row[keys.index(name)] = value
        runner.core_metric_mask = metric_mask({"drawdown_max_fraction_strategy_equity",
                                               "backtest_completion_fraction"})
        fields = observation_catalog()["fields"]
        runner.core_packet_cols = len(fields)
        result = runner.run(np.asarray([row], dtype=np.float64))
        assert result["balance"].item() == -250
        assert result["open_positions"].item() == 2
        assert result["fill_count"].item() == 2
        assert not result["alive"].item()
        packet = result["core_observations"][0].cpu()
        assert packet[fields.index("end_step")].item() == 6
        assert packet[fields.index("last_equity")].item() == (100 if strategy == "ema_anchor" else -200)
        observations.append(packet)
    torch.testing.assert_close(*observations, rtol=0, atol=0, equal_nan=True)
