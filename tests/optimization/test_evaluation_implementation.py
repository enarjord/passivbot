from copy import deepcopy
from pathlib import Path

import pytest

import optimization.evaluation_implementation as implementation


def _sources(root, files):
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    for name in implementation._EVALUATOR_ROOTS:
        path = root / f"{name}.py"
        if not path.exists():
            path.write_text("")


def test_identity_follows_direct_transitive_relative_and_dynamic_imports(tmp_path):
    _sources(
        tmp_path,
        {
            "backtest.py": "from backtest_universe import enabled\n",
            "backtest_universe.py": "from policy import risk\n",
            "policy/__init__.py": "from . import risk\n",
            "policy/risk.py": "from .calc import enabled\n",
            "policy/calc.py": "from importlib import import_module as load\nload('dynamic_policy')\n",
            "dynamic_policy.py": "enabled = True\n",
        },
    )
    paths, _ = implementation.evaluator_dependencies(tmp_path)
    for name in (
        "backtest_universe.py",
        "policy/__init__.py",
        "policy/risk.py",
        "policy/calc.py",
        "dynamic_policy.py",
    ):
        assert tmp_path / name in paths
        before = implementation.python_evaluator_source_fingerprint(tmp_path)
        with (tmp_path / name).open("a") as file:
            file.write("# changed implementation\n")
        assert implementation.python_evaluator_source_fingerprint(tmp_path) != before
    before = implementation.python_evaluator_source_fingerprint(tmp_path)
    (tmp_path / "README.md").write_text("documentation change\n")
    (tmp_path / "unreferenced_connector.py").write_text("unused = True\n")
    assert implementation.python_evaluator_source_fingerprint(tmp_path) == before


def test_unresolvable_dynamic_import_falls_back_to_all_local_sources(tmp_path):
    _sources(
        tmp_path,
        {
            "optimize.py": "__import__(configured_module)\n",
            "selected_at_runtime.py": "value = 1\n",
        },
    )
    before = implementation.python_evaluator_source_fingerprint(tmp_path)
    (tmp_path / "selected_at_runtime.py").write_text("value = 2\n")
    assert implementation.python_evaluator_source_fingerprint(tmp_path) != before


def test_real_dependency_closure_includes_universe_and_shared_helpers():
    root = Path(implementation.__file__).resolve().parents[1]
    paths, external = implementation.evaluator_dependencies(root)
    for name in ("backtest_universe.py", "utils.py", "ohlcv_utils.py", "pure_funcs.py"):
        assert root / name in paths
    assert {"numpy", "torch"} <= external
    assert (
        "scipy"
        in implementation.evaluator_dependency_versions(external)["distributions"]
    )


def test_dependency_versions_track_imported_numerical_packages_only(monkeypatch):
    monkeypatch.setattr(
        implementation.metadata,
        "packages_distributions",
        lambda: {"numpy": ["numpy"], "unrelated": ["unused"]},
    )
    versions = {"numpy": "1.0", "scipy": "1.0", "unused": "1.0"}
    monkeypatch.setattr(
        implementation.metadata,
        "requires",
        lambda name: (
            ["scipy>=1.0", "missing-extra; extra == 'unused'"]
            if name == "numpy"
            else []
        ),
    )
    monkeypatch.setattr(implementation.metadata, "version", lambda name: versions[name])
    first = implementation.evaluator_dependency_versions(
        {"numpy", "os", "passivbot_rust", "not_installed"}
    )
    assert first["imports"]["not_installed"] is None
    versions["unused"] = "2.0"
    assert (
        implementation.evaluator_dependency_versions(
            {"numpy", "os", "passivbot_rust", "not_installed"}
        )
        == first
    )
    versions["scipy"] = "2.0"
    assert (
        implementation.evaluator_dependency_versions(
            {"numpy", "os", "passivbot_rust", "not_installed"}
        )
        != first
    )


def test_runtime_identity_requires_verified_source_and_artifact(monkeypatch):
    identity = implementation.evaluation_implementation_identity.__wrapped__
    runtime = {
        "runtime_compiled_source_stamp": "source",
        "runtime_compiled_sha256": "artifact",
    }
    monkeypatch.setattr(
        implementation, "verify_loaded_runtime_extension", lambda: deepcopy(runtime)
    )
    value = identity()
    assert value["rust_source_sha256"] == "source"
    assert value["rust_artifact_sha256"] == "artifact"
    runtime["runtime_compiled_sha256"] = "rebuilt-artifact"
    assert identity() != value
    runtime["runtime_compiled_source_stamp"] = None
    with pytest.raises(RuntimeError, match="source-stamped Rust extension"):
        identity()
