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
    "backtest", "passivbot", "rust_utils",
    "optimization.backends", "optimization.warmup", "suite_runner", "packaging", "deap", "pymoo", "torch", "matplotlib",
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
    cfg["bot"]["long"]["hsl"].update(enabled=True, restart_after_red_policy="always")
    for key in ("lookback_ema_span_minutes", "realized_loss_gate_lookback_minutes", "risk_bootstrap_max_lookback_minutes"):
        cfg["live"].pop(key, None)
    cfg["live"]["pnls_max_lookback_days"] = 30.0
    cfg["optimize"]["bounds"].pop("live", None)
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
    args = ["tool", "migrate-config-v9", str(source)]
    args += ["--in-place"] if in_place else [str(output)]
    args += ["--lookback-ema-span-minutes", "43200"]
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
        assert "restart_after_red_policy" not in migrated["bot"]["long"]["hsl"]
        assert migrated["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 43200.0
        assert migrated["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] == 43200.0
        for pside in ("long", "short"):
            assert migrated["bot"][pside]["hsl"]["memory_ema_span_minutes"] == 43200.0
        assert "lookback_ema_span_minutes" not in migrated["live"]
        if not in_place:
            assert source.read_bytes() == before


def test_coin_flag_file_migration_works_without_research_runtime(tmp_path):
    cfg = get_template_config()
    cfg["config_version"] = "v8.5.0"
    for key in ("lookback_ema_span_minutes", "realized_loss_gate_lookback_minutes",
                "risk_bootstrap_max_lookback_minutes"):
        cfg["live"].pop(key, None)
    cfg["live"]["pnls_max_lookback_days"] = 30.0
    cfg["optimize"]["bounds"].pop("live", None)
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    cfg["live"]["coin_flags"] = {"AAA": "-lc coin.json -lm gs"}
    cfg["backtest"]["scenarios"] = [{"label": "flags", "overrides": {
        "live.coin_flags.AAA": "-lc coin.json -lm manual",
    }}]
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": True, "restart_after_red_policy": "never",
    }}}}))
    original = coin_file.read_bytes()
    source, output = tmp_path / "source.json", tmp_path / "migrated.json"
    source.write_text(json.dumps(cfg))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", LIVE_ONLY_COMMAND, "tool", "migrate-config-v9",
         str(source), str(output), "--lookback-ema-span-minutes", "43200",
         "--acknowledge-never-removal"],
        cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    migrated = json.loads(output.read_text())
    assert coin_file.read_bytes() == original
    assert "coin_flags" not in migrated["live"]
    assert migrated["coin_overrides"]["AAA"]["live"]["forced_mode_long"] == "graceful_stop"
    scenario = migrated["backtest"]["scenarios"][0]["overrides"]
    assert scenario["coin_overrides"]["AAA"]["live"]["forced_mode_long"] == "manual"
    assert not any("coin_flags" in key for key in scenario)
