from copy import deepcopy

import pytest

import optimization.evaluation_implementation as implementation


def test_python_identity_tracks_evaluator_sources_but_excludes_unrelated_files(
    tmp_path,
):
    for relative in implementation._EVALUATOR_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    for package in implementation._EVALUATOR_PACKAGES:
        folder = tmp_path / package
        folder.mkdir(exist_ok=True)
        (folder / "policy.py").write_text("original\n")
    first = implementation.python_evaluator_source_fingerprint(tmp_path)
    (tmp_path / "README.md").write_text("documentation change\n")
    (tmp_path / "unrelated_live_connector.py").write_text("connector change\n")
    assert implementation.python_evaluator_source_fingerprint(tmp_path) == first
    (tmp_path / "config" / "policy.py").write_text("changed calculation\n")
    assert implementation.python_evaluator_source_fingerprint(tmp_path) != first


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
