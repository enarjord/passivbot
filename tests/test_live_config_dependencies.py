"""Live configuration must load without optional research dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

LIVE_CONFIG_COMMAND = r"""
import importlib.abc
import socket
import sys

blocked = {
    "suite_runner", "backtest", "optimize", "optimization.evaluation_implementation",
    "optimization.backends", "packaging", "deap", "pymoo", "torch", "matplotlib",
    "pyecharts", "plotly", "dash", "dash_bootstrap_components", "dictdiffer",
}
class LiveOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError("Unavailable in live-only test: " + fullname, name=fullname)
sys.meta_path.insert(0, LiveOnlyImports())

def deny_network(*args, **kwargs):
    raise AssertionError("configuration loading attempted network access")
socket.socket.connect = deny_network
socket.getaddrinfo = deny_network

import passivbot
from config import get_template_config, prepare_config, compile_runtime_config
from config.overrides import parse_overrides

legacy, with_scenarios = (arg == "True" for arg in sys.argv[1:])
config = get_template_config()
for side in ("long", "short"):
    bot = config["bot"][side]
    bot["entry_cooldown"]["base_duration_minutes"] = 7.5
    bot["strategy"]["trailing_martingale"]["entry"].update(ema_span_0=123.75, ema_span_1=567.25)
    if legacy:
        bot["risk"]["entry_cooldown_minutes"] = bot.pop("entry_cooldown")["base_duration_minutes"]
        strategy = bot["strategy"]["trailing_martingale"]
        for key in ("ema_span_0", "ema_span_1"):
            strategy[key] = strategy["entry"].pop(key)
config["backtest"]["scenarios"] = []
if with_scenarios:
    config["backtest"]["scenarios"] = [{
        "label": "legacy",
        "overrides": {"bot": {"long": {
            "risk": {"entry_cooldown_minutes": 3.5},
            "strategy": {"trailing_martingale": {"ema_span_0": 234.5}},
        }}},
    }]

canonical = prepare_config(config, live_only=True, verbose=False)
if with_scenarios:
    assert canonical["backtest"]["scenarios"][0]["overrides"] == {
        "bot.long.entry_cooldown.base_duration_minutes": 3.5,
        "bot.long.strategy.trailing_martingale.entry.ema_span_0": 234.5,
    }
live = prepare_config(config, live_only=True, verbose=False, target="live")
live = compile_runtime_config(parse_overrides(live, verbose=False), runtime="live")
for side in ("long", "short"):
    assert live["bot"][side]["risk_entry_cooldown_minutes"] == 7.5
    assert live["bot"][side]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] == 123.75
assert not any(name in sys.modules for name in blocked)
"""


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("with_scenarios", [False, True])
def test_live_config_without_optional_dependencies(legacy, with_scenarios):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", LIVE_CONFIG_COMMAND, str(legacy), str(with_scenarios)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
