"""Exact GPU step transport and native reconstructed metric clocks."""

from datetime import datetime, timezone

import numpy as np
import pytest


@pytest.mark.parametrize("first,last", [(54_001, 98_639), (-1, -1)])
def test_raw_equity_clock_diagnostics_preserve_indices_without_float32_timestamps(first, last):
    torch = pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _decode_outputs

    daily = torch.zeros((1, 1, 9), dtype=torch.float32)
    scalars = torch.zeros((1, 62), dtype=torch.float32)
    scalars[0, 7], scalars[0, 8] = first, last
    out = _decode_outputs(daily, scalars)
    assert out["first_eq_step"].dtype == torch.int64
    assert out["first_eq_step"].item() == first
    assert out["last_eq_step"].item() == last
    assert "first_eq_ts" not in out and "last_eq_ts" not in out


@pytest.mark.parametrize("origin", [0, 1_700_000_000_000])
def test_native_packet_reconstructs_exact_unix_clocks_without_float32_scaling(origin):
    from test_gpu_core_metrics import _flat_packet, _native

    packet, fields = _flat_packet()
    first, last = 54_001, 98_639
    for name, value in {"start_step": first, "end_step": last,
                        "minute_count": last - first + 1,
                        "exposure_count": last - first + 1}.items():
        packet[0, fields.index(name)] = value
    row = _native().core_metrics_from_gpu_observations(
        packet, ["backtest_completion_fraction"], origin, 60_000,
        origin + last * 60_000, 180.0)[0]
    metadata = row["metric_metadata"]
    assert metadata["start_timestamp_ms"] == origin + first * 60_000
    assert metadata["actual_end_timestamp_ms"] == origin + last * 60_000
    assert metadata["equity_observation_count"] == last - first + 1
    assert row["backtest_completion_fraction"] == 1


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_long_history_equity_clock_transport(strategy, sides, coin_count):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU required")
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import build_parser, fixture_inputs

    config, _, markets, _, _ = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", str(coin_count),
        "--bars", "128", "--hsl", "disabled",
    ]))
    count, first = 98_640, 54_001
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    candles = np.full((count, coin_count, 4), 100.0)
    candles[:, :, 3] = 1.0
    config['bot']["portfolio"]['warmup']['max_minutes'] = 1
    config["backtest"].update(
        start_date=datetime.fromtimestamp(int(timestamps[first]) / 1000,
                                          timezone.utc).isoformat(),
        end_date=datetime.fromtimestamp((int(timestamps[-1]) + 60_000) / 1000,
                                        timezone.utc).isoformat(),
    )
    for coin, market in markets.items():
        if coin != "__meta__":
            market.update(first_valid_index=0, last_valid_index=count - 1,
                          warmup_minutes=1, warmup_minutes_source="history")
    markets["__meta__"] = {"requested_start_ts": int(timestamps[first])}
    proxy = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets,
        btc=np.full(count, 50_000.0), timestamps=timestamps,
        exchange="binance", batch_size=1,
        needed_metrics={"cagr_fraction_strategy_equity"}, factual_hsl=True,
    )
    row = proxy.evaluate_results([{}])[0]
    metadata = row.metadata["metric_metadata"]
    assert metadata["start_timestamp_ms"] == int(timestamps[first])
    assert metadata["actual_end_timestamp_ms"] == int(timestamps[-1])
    assert metadata["intended_end_timestamp_ms"] == int(timestamps[-1])
    assert metadata["equity_observation_count"] == count - first


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_service_fill_gap_clocks_match_exact_rust(side, coin_count):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("GPU required")
    from test_gpu_entry_sizing_parity import _fixture
    from optimization.gpu.parameters import prepare_candidate_parameters
    from optimization.gpu.service import SharedAccountReplay
    from backtest import run_backtest
    from config.metrics import resolve_metric_value
    from rust_utils import verify_loaded_runtime_extension

    assert not verify_loaded_runtime_extension().get("skipped")
    config, candles, markets, btc, timestamps = _fixture(side, coin_count, "initial")
    requested = {"fills_gap_max_days"}
    _, _, analysis = run_backtest(candles, markets, config, "bybit", btc, timestamps)
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="bybit", batch_size=1,
        needed_metrics=requested, factual_hsl=True)
    actual = proxy.evaluate([prepare_candidate_parameters(config, markets, "bybit")])[0]
    for name in requested:
        assert actual[name] == pytest.approx(resolve_metric_value(analysis, name), abs=1e-8)
