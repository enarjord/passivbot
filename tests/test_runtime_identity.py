from dataclasses import FrozenInstanceError
import json

import pytest

import runtime_identity as runtime_identity_module
from runtime_identity import (
    RuntimeIdentity,
    build_runtime_identity,
    config_sha256,
    write_runtime_manifest,
)


def _identity(**overrides) -> RuntimeIdentity:
    values = {
        "schema_version": 1,
        "run_id": "a" * 32,
        "started_at_ms": 1_700_000_000_000,
        "passivbot_version": "8.0.0",
        "python_git_commit": "b" * 40,
        "python_git_dirty": False,
        "config_sha256": "c" * 64,
        "rust_crate_version": "0.1.0",
        "rust_source_sha256": "d" * 64,
        "rust_artifact_sha256": "e" * 64,
    }
    values.update(overrides)
    return RuntimeIdentity(**values)


def test_config_sha256_is_canonical_and_content_sensitive():
    left = {"z": {3, 1, 2}, "nested": {"b": 2, "a": [1, 2]}}
    right = {"nested": {"a": [1, 2], "b": 2}, "z": {2, 3, 1}}
    assert config_sha256(left) == config_sha256(right)
    assert config_sha256(left) != config_sha256({**right, "extra": True})


def test_build_runtime_identity_records_loaded_binary_without_exposing_config(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runtime_identity_module,
        "_git_value",
        lambda _root, *args: "f" * 40 if args[0] == "rev-parse" else "",
    )
    monkeypatch.setattr(
        runtime_identity_module,
        "_rust_build_info",
        lambda: {"crate_version": "0.2.0", "source_fingerprint": "1" * 64},
    )
    monkeypatch.setattr(
        runtime_identity_module,
        "collect_runtime_provenance",
        lambda: {"runtime_module_sha256": "2" * 64},
    )
    config = {"api_secret": "must-not-be-recorded", "threshold": 0.1}
    identity = build_runtime_identity(config, started_at_ms=123, repo_root=tmp_path)

    assert identity.python_git_commit == "f" * 40
    assert identity.python_git_dirty is False
    assert identity.rust_source_sha256 == "1" * 64
    assert identity.rust_artifact_sha256 == "2" * 64
    assert "must-not-be-recorded" not in json.dumps(identity.to_dict())
    with pytest.raises(FrozenInstanceError):
        identity.run_id = "changed"


def test_runtime_manifest_is_exclusive_and_fill_attribution_is_precise(tmp_path):
    identity = _identity()
    path = write_runtime_manifest(identity, root=tmp_path)
    assert json.loads(path.read_text()) == identity.to_dict()
    assert identity.fill_provenance()["attribution"] == "first_ingested_by_runtime"
    with pytest.raises(FileExistsError):
        write_runtime_manifest(identity, root=tmp_path)


def test_runtime_identity_collection_failure_degrades_to_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_identity_module, "_git_value", lambda *_args: None)
    monkeypatch.setattr(runtime_identity_module, "_rust_build_info", lambda: {})
    monkeypatch.setattr(
        runtime_identity_module,
        "collect_runtime_provenance",
        lambda: (_ for _ in ()).throw(OSError("unreadable artifact")),
    )
    monkeypatch.setattr(
        runtime_identity_module,
        "config_sha256",
        lambda _config: (_ for _ in ()).throw(TypeError("unhashable config")),
    )

    identity = build_runtime_identity({}, started_at_ms=123, repo_root=tmp_path)

    assert identity.python_git_commit == "unknown"
    assert identity.python_git_dirty is None
    assert identity.config_sha256 == "unknown"
    assert identity.rust_source_sha256 == "unknown"
    assert identity.rust_artifact_sha256 == "unknown"
