"""The coin encoder is usable by CPU preparation without optional runtimes."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_coin_encoder_import_and_execution_do_not_load_device_or_search_modules():
    source = Path(__file__).resolve().parents[2] / "src"
    program = '''
import importlib.abc
import sys
from types import SimpleNamespace
import numpy as np

forbidden = ("torch", "cupy", "pymoo", "optimization.gpu.service", "optimization.gpu.mps_kernel")
class NoDeviceOrSearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("CPU coin encoding imported " + fullname)
sys.meta_path.insert(0, NoDeviceOrSearch())
from optimization.gpu.coin_parameters import build_coin_override_parameters
from optimization.gpu.model import EMA_ANCHOR_COIN_OVERRIDE_COLS

policy = dict(enabled=False, red_threshold=0.1, ema_span_minutes=30,
              cooldown_minutes_after_red=0)
payload = SimpleNamespace(
    strategy_params_list=[{"long": {"offset": 0.125}}],
    bot_params_list=[{"long": {"n_positions": 1}}],
    backtest_params={"coins": ["COIN"], "dynamic_wel_by_tradability": True, "lookback_ema_span_minutes": 1440.0,
                     "equity_hard_stop_loss": {"engine": "hsl", "mode": "coin",
                                               "coins": {"COIN": [policy, policy]}}},
)
patch = {"bot": {"long": {"strategy": {"ema_anchor": {"offset": 0.125}}}}}
matrix, contract = build_coin_override_parameters(
    config={}, mss={}, exchange="binance", coins=["COIN"], payload=payload,
    side="long", strategy_kind="ema_anchor", resolve_override=lambda *args: patch,
)
assert matrix.shape == (1, EMA_ANCHOR_COIN_OVERRIDE_COLS)
assert np.count_nonzero(np.isfinite(matrix)) == 1
assert contract["exact_overrides"] == [patch]
patch["bot"]["long"]["strategy"]["ema_anchor"]["offset"] = 7
assert contract["exact_overrides"][0]["bot"]["long"]["strategy"]["ema_anchor"]["offset"] == 0.125
assert not any(name in sys.modules for name in forbidden)
'''
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", program], check=True, env=env,
                   capture_output=True, text=True)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("trigger", [0.0, 0.75])
def test_canonical_coin_unstuck_trigger_matches_cpu_payload(strategy, side, trigger):
    from backtest import build_backtest_payload
    from optimization.gpu.coin_parameters import build_coin_override_parameters
    from optimization.gpu import model
    from tools.gpu_parity import build_parser, fixture_inputs

    config, candles, markets, btc, timestamps = fixture_inputs(
        build_parser().parse_args([
            "--fixture", strategy, "--sides", "both", "--coins", "2", "--bars", "128",
        ])
    )
    coins = config["backtest"]["coins"]["binance"]
    for coin in coins:
        markets[coin]["symbol"] = f"{coin}/USDT:USDT"
    global_trigger = 0.5
    config["bot"][side]["unstuck"]["exposure_trigger_ratio"] = global_trigger
    patch = {"bot": {side: {"unstuck": {"exposure_trigger_ratio": trigger}}}}
    config["coin_overrides"] = {coins[0]: patch}
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    assert payload.bot_params_list[0][side]["unstuck_threshold"] == trigger
    assert payload.bot_params_list[1][side]["unstuck_threshold"] == global_trigger

    matrix, contract = build_coin_override_parameters(
        config=config, mss=markets, exchange="binance", coins=coins, payload=payload,
        side=side, strategy_kind=strategy,
    )
    column = getattr(model, f"{strategy.upper()}_COIN_OVERRIDE_UNSTUCK_START_COLUMN") + 5
    assert matrix[0, column] == trigger
    assert np.isnan(matrix[1, column])  # The unpinned coin inherits candidate policy.
    assert contract["values"][0][column] == trigger
    assert contract["values"][1][column] is None
    assert contract["exact_overrides"][0] == patch
