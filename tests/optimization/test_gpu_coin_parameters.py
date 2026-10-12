"""The coin encoder is usable by CPU preparation without optional runtimes."""

import os
from pathlib import Path
import subprocess
import sys


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
              cooldown_minutes_after_red=0, restart_after_red_policy="always")
payload = SimpleNamespace(
    strategy_params_list=[{"long": {"offset": 0.125}}],
    bot_params_list=[{"long": {"n_positions": 1}}],
    backtest_params={"coins": ["COIN"], "dynamic_wel_by_tradability": True,
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
