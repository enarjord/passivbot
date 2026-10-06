from copy import deepcopy

import pytest

from optimization.gpu.parameters import prepare_candidate_parameters
from tools.gpu_parity import build_parser, fixture_inputs


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_candidate_parameters_use_global_values_without_leaking_first_coin_override(strategy, monkeypatch):
    import backtest
    def forbidden(*_args, **_kwargs):
        pytest.fail("parameter preparation must not simulate a CPU backtest")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    config, _candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    expected = prepare_candidate_parameters(config, markets, "binance")
    patched = deepcopy(config)
    leaf = {"entry": {"initial_qty_pct": 0.5}} if strategy == "trailing_martingale" else {"base_qty_pct": 0.5}
    patched["coin_overrides"] = {"COIN00": {"bot": {"long": {"strategy": {strategy: leaf}}}}}
    assert prepare_candidate_parameters(patched, markets, "binance") == expected
    assert patched["coin_overrides"]["COIN00"]["bot"]["long"]["strategy"][strategy] == leaf


def test_effective_disabled_side_and_canonical_adaptive_values_are_explicit(monkeypatch):
    config, _candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "long", "--coins", "2", "--bars", "128",
    ]))
    config["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = None
    parameters = prepare_candidate_parameters(config, markets, "binance")
    assert parameters["short_total_wallet_exposure_limit"] == 0
    assert parameters["short_n_positions"] == 0
    assert parameters["long_entry_cooldown_max_duration_minutes"] == -1
    assert parameters["long_unilateralness_window"] >= parameters["long_unilateralness_ema_span_1m"]
