from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "container" / "entrypoint.sh"
RENDER_API_KEYS = REPO_ROOT / "container" / "render_api_keys.py"
RENDER_CONFIG = REPO_ROOT / "container" / "render_config.py"


def test_render_api_keys_script_creates_expected_payload(tmp_path):
    target = tmp_path / "api-keys.json"
    env = os.environ | {
        "PB_USER": "bitget_01",
        "PB_EXCHANGE": "bitget",
        "PB_API_KEY": "key123",
        "PB_API_SECRET": "secret123",
        "PB_API_PASSPHRASE": "phrase123",
    }
    subprocess.run(
        [sys.executable, str(RENDER_API_KEYS), str(target)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    payload = json.loads(target.read_text())
    assert payload == {
        "bitget_01": {
            "exchange": "bitget",
            "key": "key123",
            "passphrase": "phrase123",
            "secret": "secret123",
        }
    }


def test_render_config_script_merges_base_config_and_env_overrides(tmp_path):
    base_path = tmp_path / "live.json"
    base_path.write_text(
        '{"live": {"approved_coins": ["BTC"], "user": "bybit_01"}, "monitor": {"enabled": true}}\n',
        encoding="utf-8",
    )
    target = tmp_path / "config.runtime.json"
    env = os.environ | {
        "PB_USER": "bitget_01",
        "PB_CONFIG_PATH": str(base_path),
        "PB_APPROVED_COINS": "BTC,ETH,SOL",
        "PB_LOG_LEVEL": "2",
        "PB_MONITOR_ENABLED": "false",
        "PB_MONITOR_ROOT": "/data/monitor",
    }
    subprocess.run(
        [sys.executable, str(RENDER_CONFIG), str(target)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    payload = json.loads(target.read_text())
    assert payload["live"]["user"] == "bitget_01"
    assert payload["live"]["approved_coins"] == ["BTC", "ETH", "SOL"]
    assert payload["logging"]["level"] == 2
    assert payload["monitor"]["enabled"] is False
    assert payload["monitor"]["root_dir"] == "/data/monitor"


def test_entrypoint_generates_runtime_files_and_invokes_cli(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_passivbot = fake_bin / "passivbot"
    record_path = tmp_path / "invocation.json"
    fake_passivbot.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "payload = {\n"
        "    'argv': sys.argv,\n"
        "    'api_keys_path': os.path.realpath(os.path.join(os.environ['PB_APP_ROOT'], 'api-keys.json')),\n"
        "}\n"
        "with open(os.environ['FAKE_PASSIVBOT_RECORD'], 'w', encoding='utf-8') as f:\n"
        "    json.dump(payload, f)\n",
        encoding="utf-8",
    )
    fake_passivbot.chmod(fake_passivbot.stat().st_mode | stat.S_IEXEC)

    app_root = tmp_path / "app"
    runtime_root = tmp_path / "runtime"
    app_root.mkdir()
    runtime_root.mkdir()

    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PB_APP_ROOT": str(app_root),
        "PB_RUNTIME_ROOT": str(runtime_root),
        "PB_USER": "bitget_01",
        "PB_EXCHANGE": "bitget",
        "PB_API_KEY": "key123",
        "PB_API_SECRET": "secret123",
        "PB_APPROVED_COINS": "BTC,ETH",
        "PB_MONITOR_ENABLED": "false",
        "FAKE_PASSIVBOT_RECORD": str(record_path),
    }

    subprocess.run(
        [str(ENTRYPOINT), "--log-level", "info"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    invocation = json.loads(record_path.read_text())
    assert Path(invocation["argv"][0]).name == "passivbot"
    assert invocation["argv"][1:4] == ["live", str(runtime_root / "config.runtime.json"), "-u"]
    assert invocation["argv"][4:] == ["bitget_01", "--log-level", "info"]

    api_keys = json.loads((runtime_root / "api-keys.json").read_text())
    assert api_keys["bitget_01"]["exchange"] == "bitget"
    rendered_config = json.loads((runtime_root / "config.runtime.json").read_text())
    assert rendered_config["live"]["approved_coins"] == ["BTC", "ETH"]
    assert rendered_config["monitor"]["enabled"] is False
