"""Host admission of the inputs used by the inherited-override device benchmark."""

import pytest

pytest.importorskip("torch")

from optimization.gpu.parameters import prepare_candidate_parameters
from tools.gpu_inherited_overrides_benchmark import benchmark_inputs


@pytest.mark.parametrize("coins,slots", [(2, 1), (9, 7)])
def test_benchmark_inputs_pass_canonical_candidate_admission(coins, slots):
    config, candles, markets, btc, timestamps = benchmark_inputs(bars=128, coins=coins)
    # Exercise the same producer admission as the real benchmark, without
    # constructing a replay or compiling/dispatching a device kernel.
    candidate = prepare_candidate_parameters(config, markets, "binance")
    assert config["bot"]["long"]["risk"]["max_positions"] == slots
    assert "n_positions" not in config["bot"]["long"]["risk"]
    assert candidate["long_n_positions"] == slots
    assert config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert candles.shape[:2] == (128, coins)
    assert set(markets) == {"__meta__", *(f"COIN{i:02d}" for i in range(coins))}
    assert isinstance(markets["__meta__"]["requested_start_ts"], int)
    assert len(btc) == len(timestamps) == 128
