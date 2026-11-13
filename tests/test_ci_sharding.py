"""CI partition completeness, ordering, error handling, and version gate wiring."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "ci"
spec = importlib.util.spec_from_file_location("ci_sharding_test_support", CI / "ci_sharding.py")
sharding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sharding)
FORMAT_VERSION = sharding.FORMAT_VERSION
selected_nodeids = sharding.selected_nodeids
shard_for_path = sharding.shard_for_path
verify_manifests = sharding.verify_manifests


VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


def _run(root, *args):
    env = {**os.environ, "PYTHONPATH": str(CI), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "ci_pytest_shard", "-c", str(root / "pytest.ini"),
         *args], cwd=root, env=env, text=True, capture_output=True, timeout=30,
    )


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = -m 'not excluded'\nmarkers =\n    excluded: default exclusion\n"
    )
    # At least one whole file in every bucket, without hardcoding fitted paths.
    buckets = {}
    candidate = 0
    while len(buckets) < 8:
        name = f"test_sample_{candidate}.py"
        buckets.setdefault(shard_for_path(f"tests/{name}", 8), name)
        candidate += 1
    for name in buckets.values():
        (tmp_path / "tests" / name).write_text(
            "import os\nfrom pathlib import Path\nimport pytest\n"
            "@pytest.mark.parametrize('value', [2, 0, 1])\n"
            "def test_order(request, value):\n"
            "    with Path(os.environ['EXECUTION_LOG']).open('a') as output:\n"
            "        output.write(request.node.nodeid + '\\n')\n"
            "@pytest.mark.excluded\n"
            "def test_excluded():\n    assert False\n"
        )
    return tmp_path


def _nodeids(result):
    return [line.strip() for line in result.stdout.splitlines()
            if line.startswith("tests/") and "::" in line]


def test_actual_pytest_shards_preserve_default_collection_order_and_exact_once_execution(suite, monkeypatch):
    baseline = _run(suite, "--collect-only", "-q")
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    collected = _nodeids(baseline)
    assert len(collected) == 24
    assert not any("test_excluded" in nodeid for nodeid in collected)
    manifests = suite / "manifests"
    for index in range(8):
        execution_log = suite / f"execution-{index}.txt"
        monkeypatch.setenv("EXECUTION_LOG", str(execution_log))
        manifest = manifests / f"shard-{index}.json"
        result = _run(suite, "-q", f"--ci-shard-index={index}", "--ci-shard-count=8",
                      f"--ci-shard-manifest={manifest}")
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(manifest.read_text())
        assert data["collected"] == collected
        assert data["selected"] == [nodeid for nodeid in collected
                                    if shard_for_path(nodeid.split("::", 1)[0], 8) == index]
        assert execution_log.read_text().splitlines() == data["selected"]
        assert len({nodeid.split("::", 1)[0] for nodeid in data["selected"]}) == 1
    assert verify_manifests(manifests, count=8, python_version=VERSION) == len(collected)
    # CLI keyword filtering also precedes sharding and manifest capture.
    selected_path = json.loads((manifests / "shard-0.json").read_text())["selected"][0]
    keyword = Path(selected_path.split("::", 1)[0]).name
    result = _run(suite, "--collect-only", "-q", "-k", keyword,
                  "--ci-shard-index=0", "--ci-shard-count=8",
                  f"--ci-shard-manifest={suite / 'filtered.json'}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads((suite / "filtered.json").read_text())["collected"]) == 3


def test_parallel_shards_preserve_collection_and_execute_once(suite, monkeypatch):
    pytest.importorskip("xdist")
    manifests = suite / "parallel-manifests"
    for index in range(4):
        execution_log = suite / f"parallel-execution-{index}.txt"
        monkeypatch.setenv("EXECUTION_LOG", str(execution_log))
        manifest = manifests / f"shard-{index}.json"
        result = _run(suite, "-q", "-p", "xdist.plugin", "-n", "2", "--dist=loadfile",
                      "--max-worker-restart=0", f"--ci-shard-index={index}",
                      "--ci-shard-count=4", f"--ci-shard-manifest={manifest}")
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(manifest.read_text())
        assert len(data["selected"]) == 6
        assert Counter(execution_log.read_text().splitlines()) == Counter(data["selected"])
    assert verify_manifests(manifests, count=4, python_version=VERSION) == 24


def test_parallel_shard_rejects_missing_completion_reports(suite):
    pytest.importorskip("xdist")
    # Suppress test execution while letting xdist report that scheduling finished.
    (suite / "conftest.py").write_text(
        "def pytest_runtest_protocol(item, nextitem):\n"
        "    return True\n"
    )
    result = _run(suite, "-q", "-p", "xdist.plugin", "-n", "2", "--dist=loadfile",
                  "--ci-shard-index=0", "--ci-shard-count=4",
                  f"--ci-shard-manifest={suite / 'missing.json'}")
    assert result.returncode != 0
    assert "Parallel shard did not complete every selected test exactly once" in result.stdout


@pytest.mark.parametrize("args", [
    ["--ci-shard-index=0"], ["--ci-shard-count=8"], ["--ci-shard-manifest=x.json"],
    ["--ci-shard-index=-1", "--ci-shard-count=8"],
    ["--ci-shard-index=8", "--ci-shard-count=8"],
    ["--ci-shard-index=0", "--ci-shard-count=0"],
    ["--ci-shard-index=0", "--ci-shard-count=-2"],
    ["--ci-shard-index=bad", "--ci-shard-count=8"],
    ["--ci-shard-index=0", "--ci-shard-count=1.5"],
])
def test_invalid_shard_options_fail_before_execution(suite, args):
    result = _run(suite, *args)
    assert result.returncode == 4, result.stdout + result.stderr


def test_empty_shard_is_an_error_and_collection_errors_are_not_hidden(suite):
    file = next((suite / "tests").glob("*.py"))
    index = (shard_for_path(file.relative_to(suite).as_posix(), 8) + 1) % 8
    result = _run(suite, str(file), f"--ci-shard-index={index}", "--ci-shard-count=8")
    assert result.returncode == 4
    assert "selected no tests" in result.stderr
    (suite / "tests" / "test_broken.py").write_text("def broken(\n")
    result = _run(suite, "--collect-only", "--ci-shard-index=0", "--ci-shard-count=8")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "SyntaxError" in result.stdout


@pytest.mark.parametrize("path", ["tests/test_ci_sharding.py", "tests/optimization/test_native_backend.py"])
def test_default_assignment_is_unsalted_sha256_of_relative_posix_file_path(path):
    expected = int.from_bytes(hashlib.sha256(path.encode("utf-8")).digest(), "big") % 8
    assert shard_for_path(path, 8) == expected


@pytest.mark.parametrize("path", ["", "/tests/test_a.py", "../test_a.py", "tests/../test_a.py",
                                  "tests/./test_a.py", "tests//test_a.py", "tests\\test_a.py"])
def test_noncanonical_paths_are_rejected(path):
    with pytest.raises(ValueError, match="canonical repository-relative POSIX"):
        shard_for_path(path, 8)


def _manifests():
    collected = [f"tests/test_sample_{index}.py::test_value[{param}]"
                 for index in range(64) for param in [2, 0, 1]]
    return [{"format_version": FORMAT_VERSION, "python_version": VERSION,
             "shard_index": index, "shard_count": 8, "collected": list(collected),
             "selected": selected_nodeids(collected, index, 8)} for index in range(8)]


@pytest.mark.parametrize("corruption", [
    "missing_manifest", "extra_manifest", "invalid_json", "unknown_field", "wrong_version",
    "wrong_count", "boolean_index", "duplicate_index", "missing_test", "extra_test",
    "wrong_assignment", "reordered_tests", "different_collection", "duplicate_selected",
    "duplicate_collected", "empty_selected", "empty_collected", "wrong_format", "boolean_format",
])
def test_coverage_gate_rejects_incomplete_inconsistent_or_misassigned_manifests(tmp_path, corruption):
    data = deepcopy(_manifests())
    if corruption == "missing_manifest":
        data.pop()
    elif corruption == "extra_manifest":
        data.append(deepcopy(data[0]))
    elif corruption == "unknown_field":
        data[0]["ignored"] = True
    elif corruption == "wrong_version":
        data[0]["python_version"] = "0.0"
    elif corruption == "wrong_count":
        data[0]["shard_count"] = 7
    elif corruption == "boolean_index":
        data[0]["shard_index"] = False
    elif corruption == "duplicate_index":
        data[0]["shard_index"] = 1
    elif corruption == "missing_test":
        data[0]["selected"].pop()
    elif corruption == "extra_test":
        data[0]["selected"].append("tests/not_collected.py::test_new")
    elif corruption == "wrong_assignment":
        data[0]["selected"].append(data[1]["selected"][0])
    elif corruption == "reordered_tests":
        data[0]["selected"].reverse()
    elif corruption == "different_collection":
        data[0]["collected"].reverse()
    elif corruption == "duplicate_selected":
        data[0]["selected"].append(data[0]["selected"][0])
    elif corruption == "duplicate_collected":
        data[0]["collected"].append(data[0]["collected"][0])
    elif corruption == "empty_selected":
        data[0]["selected"] = []
    elif corruption == "empty_collected":
        data[0]["collected"] = []
    elif corruption == "wrong_format":
        data[0]["format_version"] = 2
    elif corruption == "boolean_format":
        data[0]["format_version"] = True
    for index, manifest in enumerate(data):
        (tmp_path / f"shard-{index}.json").write_text(json.dumps(manifest))
    if corruption == "invalid_json":
        (tmp_path / "shard-0.json").write_text("{")
    with pytest.raises(ValueError):
        verify_manifests(tmp_path, count=8, python_version=VERSION)


def test_unmodified_manifest_set_and_stdlib_only_gate_cli(tmp_path):
    for index, manifest in enumerate(_manifests()):
        (tmp_path / f"shard-{index}.json").write_text(json.dumps(manifest))
    assert verify_manifests(tmp_path, count=8, python_version=VERSION) == 192
    result = subprocess.run([sys.executable, "-S", str(CI / "ci_sharding.py"), str(tmp_path),
                             "--shard-count=8", f"--python-version={VERSION}"],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "192 tests" in result.stdout


def test_workflow_preserves_version_specific_required_gates_and_shared_steps():
    main = yaml.load((ROOT / ".github/workflows/main.yml").read_text(), Loader=yaml.BaseLoader)
    shared = yaml.load((ROOT / ".github/workflows/python-tests.yml").read_text(), Loader=yaml.BaseLoader)
    for version, key in [("3.12", "python312"), ("3.14", "python314")]:
        caller = main["jobs"][f"{key}_shards"]
        assert caller["uses"] == "./.github/workflows/python-tests.yml"
        assert caller["with"] == {"python-version": version}
        gate = main["jobs"][key]
        assert gate["name"] == f"Python {version}"
        assert gate["needs"] == f"{key}_shards"
        assert gate["if"] == "${{ always() }}"
        assert gate["steps"][0]["env"]["RESULT"] == "${{ needs." + key + "_shards.result }}"
        assert gate["steps"][0]["run"] == 'test "$RESULT" = success'
        assert gate["steps"][1]["uses"].startswith("actions/checkout@")
        assert gate["steps"][2]["uses"] == "./.github/actions/verify-python-coverage"
        assert gate["steps"][2]["with"] == {"python-version": version}
    rust = main["jobs"]["rust"]
    assert rust["name"] == "Rust"
    rust_commands = [step for step in rust["steps"] if "run" in step]
    assert [step["run"] for step in rust_commands if "continue-on-error" not in step] == [
        "cargo test --locked --no-default-features", "cargo check --locked --tests",
    ]
    assert [step["run"] for step in rust_commands if step.get("continue-on-error") == "true"] == [
        "cargo fmt --all -- --check", "cargo clippy --locked --all-targets",
    ]
    lint = main["jobs"]["python_lint"]
    assert lint["name"] == "Python undefined names"
    lint_commands = lint["steps"][-1]["run"].splitlines()
    assert "ruff check --target-version py312 --select F821 src tests .github/ci" in lint_commands
    assert "ruff check --target-version py312 --select F811 src" in lint_commands
    assert "continue-on-error" not in lint["steps"][-1]
    shards = shared["jobs"]["shards"]
    assert shards["timeout-minutes"] == "90"
    assert shards["strategy"]["fail-fast"] == "false"
    assert shards["strategy"]["matrix"]["shard"] == [str(index) for index in range(4)]
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"]:
        assert shards["env"][variable] == "1"
    steps = {step["name"]: step for step in shards["steps"]}
    for name in ["Validate documentation", "Run release CLI smokes"]:
        assert steps[name]["if"] == "matrix.shard == 0"
    assert "--maxfail=10" in steps["Run Python test shard"]["run"]
    assert "--ci-shard-count=4" in steps["Run Python test shard"]["run"]
    assert "-n 2 --dist=loadfile --max-worker-restart=0" in steps["Run Python test shard"]["run"]
    assert set(shared["jobs"]) == {"shards"}
    coverage_path = ROOT / ".github/actions/verify-python-coverage/action.yml"
    coverage = yaml.load(coverage_path.read_text(), Loader=yaml.BaseLoader)
    download, verify = coverage["runs"]["steps"][-2:]
    assert download["with"]["pattern"] == "python-${{ inputs.python-version }}-shard-*"
    assert verify["env"]["PYTHON_VERSION"] == "${{ inputs.python-version }}"
    assert '--shard-count=4 --python-version="$PYTHON_VERSION"' in verify["run"]


def test_docker_cancels_only_superseded_runs_for_the_same_pr():
    docker = yaml.load((ROOT / ".github/workflows/docker.yml").read_text(), Loader=yaml.BaseLoader)
    assert docker["concurrency"] == {
        "group": "docker-images-${{ github.event_name }}-${{ github.event.pull_request.number || github.run_id }}",
        "cancel-in-progress": "true",
    }
