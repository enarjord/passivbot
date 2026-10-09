"""Exercise the public migration command without optional research dependencies."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from config.schema import CONFIG_SCHEMA_VERSION, get_template_config


LIVE_ONLY_COMMAND = r"""
import importlib.abc
import socket
import sys

# A fresh interpreter prevents the full test environment's imports from masking
# missing dependencies. Backend runtimes must never be loaded by static validation.
blocked = {
    "optimization.backends", "deap", "pymoo", "torch", "matplotlib",
    "pyecharts", "plotly", "dash", "dash_bootstrap_components", "dictdiffer",
}
class LiveOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError("Unavailable in live-only test: " + fullname, name=fullname)
sys.meta_path.insert(0, LiveOnlyImports())

def deny_network(*args, **kwargs):
    raise AssertionError("migration attempted network access")
socket.socket.connect = deny_network
socket.getaddrinfo = deny_network

from passivbot_cli.main import main
result = main(sys.argv[1:])
assert not any(name in sys.modules for name in blocked)
raise SystemExit(result)
"""


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("coarse", [None, "base", "scenario"])
def test_gpu_migration_works_with_live_only_dependencies(tmp_path, in_place, coarse):
    cfg = get_template_config()
    cfg["config_version"] = "v8.4.0"
    cfg["live"]["hsl_engine"] = "legacy"
    cfg["bot"]["long"]["hsl"].update(enabled=True, restart_after_red_policy="threshold")
    cfg["optimize"]["backend"] = "gpu"
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    if coarse == "base":
        cfg["backtest"]["candle_interval_minutes"] = 5
    elif coarse == "scenario":
        cfg["backtest"]["scenarios"] = [
            {"label": "coarse", "overrides": {"backtest.candle_interval_minutes": 5}}
        ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(cfg))
    before = source.read_bytes()
    output = source if in_place else tmp_path / "migrated.json"
    args = ["tool", "migrate-hsl", str(source)]
    args += ["--in-place"] if in_place else [str(output)]
    args += ["--restart-policy", "long=always"]
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", LIVE_ONLY_COMMAND, *args],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    if coarse:
        assert result.returncode == 2, result.stdout + result.stderr
        assert "GPU HSL requires 1m candles" in result.stderr
        assert source.read_bytes() == before
        if not in_place:
            assert not output.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        migrated = json.loads(output.read_text())
        assert migrated["config_version"] == CONFIG_SCHEMA_VERSION
        assert migrated["optimize"]["backend"] == "gpu"
        assert migrated["bot"]["long"]["hsl"]["restart_after_red_policy"] == "always"
        if not in_place:
            assert source.read_bytes() == before
