"""Exercise schema resources through offline source/archive/wheel packaging."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path("schemas/config-v9.schema.json")

# Isolate Python packaging from the unchanged Rust build. The resulting test
# wheel is used only to verify package resources, never as a native artifact.
BUILD_PYTHON_WHEEL = """
import runpy
import socket
import sys
import types
from unittest.mock import patch
import setuptools

def offline(*args, **kwargs):
    raise AssertionError("schema packaging must remain offline")
socket.socket.connect = offline
socket.create_connection = offline
socket.getaddrinfo = offline
native_declaration = types.ModuleType("setuptools_rust")
native_declaration.Binding = types.SimpleNamespace(PyO3=None)
native_declaration.RustExtension = lambda *args, **kwargs: None
options = {}
with patch.dict(sys.modules, {"setuptools_rust": native_declaration}):
    with patch.object(setuptools, "setup", lambda **kwargs: options.update(kwargs)):
        runpy.run_path("setup.py")
options.pop("rust_extensions")
options.pop("setup_requires")
setuptools.setup(**options)
"""


def run_offline_python(code, *args, cwd, pythonpath):
    env = {**os.environ, "PYTHONPATH": str(pythonpath), "PIP_NO_INDEX": "1"}
    return subprocess.run(
        [sys.executable, "-c", code, *args], cwd=cwd, env=env,
        text=True, capture_output=True,
    )


def test_source_archive_and_wheel_ship_checked_schema_and_installed_cli_is_offline(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns(
        "__pycache__", "*.egg-info", "*.so", "*.pyd",
    ))
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in"):
        shutil.copy2(ROOT / name, project / name)
    for path in ROOT.glob("requirements*.txt"):
        shutil.copy2(path, project / path.name)
    (project / "schemas").mkdir()
    shutil.copy2(ROOT / SCHEMA, project / SCHEMA)
    expected = (ROOT / SCHEMA).read_bytes()

    archives = tmp_path / "archives"
    built = run_offline_python(
        BUILD_PYTHON_WHEEL, "sdist", "--dist-dir", str(archives),
        cwd=project, pythonpath=project / "src",
    )
    assert built.returncode == 0, built.stdout + built.stderr
    archive = next(archives.glob("passivbot-*.tar.gz"))
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive) as source:
        schema_member = next(member for member in source.getmembers()
                             if member.name.endswith("/" + SCHEMA.as_posix()))
        assert source.extractfile(schema_member).read() == expected
        source.extractall(unpacked, filter="data")
    archived_project = next(unpacked.iterdir())
    wheels = tmp_path / "wheels"
    built = run_offline_python(
        BUILD_PYTHON_WHEEL, "bdist_wheel", "--dist-dir", str(wheels),
        cwd=archived_project, pythonpath=archived_project / "src",
    )
    assert built.returncode == 0, built.stdout + built.stderr
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(wheels.glob("passivbot-*.whl"))) as wheel:
        assert wheel.read("tools/schemas/config-v9.schema.json") == expected
        wheel.extractall(installed)
    shutil.rmtree(project)
    shutil.rmtree(unpacked)
    resource = installed / "tools" / SCHEMA
    before = resource.stat().st_mtime_ns
    destination = tmp_path / "editor-schema.json"
    checked = run_offline_python(
        """
import socket
import sys
from pathlib import Path
def offline(*args, **kwargs):
    raise AssertionError("installed schema CLI must remain offline")
socket.socket.connect = offline
socket.create_connection = offline
socket.getaddrinfo = offline
from passivbot_cli.main import main
from tools.config_schema import DEFAULT_OUTPUT
assert DEFAULT_OUTPUT == Path(sys.argv[1])
assert main(["tool", "config-schema", "--check"]) == 0
assert main(["tool", "config-schema", "--output", sys.argv[2]]) == 0
assert main(["tool", "config-schema", "--field", "bot.long.unstuck.ema_span_0"]) == 0
""", str(resource), str(destination), cwd=tmp_path, pythonpath=installed,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert destination.read_bytes() == expected
    assert resource.read_bytes() == expected
    assert resource.stat().st_mtime_ns == before
    assert json.loads(destination.read_text())["title"] == "Passivbot authored v9 configuration"
