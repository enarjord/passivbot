"""The optional pytest timing report aggregates phases without logging node IDs."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_ci_timings_logs_one_record_per_test_file(tmp_path):
    (tmp_path / "test_first.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', [0, 1])\n"
        "def test_first(value):\n    assert value >= 0\n"
    )
    (tmp_path / "test_second.py").write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='example skip')\n"
        "def test_second():\n    assert False\n"
    )
    ci_path = Path(__file__).resolve().parents[1] / ".github" / "ci"
    env = {**os.environ, "PYTHONPATH": str(ci_path), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "ci_pytest_timings"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    prefix = "CI_TEST_FILE "
    rows = [json.loads(line.removeprefix(prefix)) for line in result.stdout.splitlines()
            if line.startswith(prefix)]
    assert sorted((row["file"], row["tests"]) for row in rows) == [
        ("test_first.py", 2), ("test_second.py", 1),
    ]
    assert all(row.keys() == {"file", "seconds", "tests"} and row["seconds"] >= 0 for row in rows)
    assert [row["seconds"] for row in rows] == sorted(
        (row["seconds"] for row in rows), reverse=True
    )
