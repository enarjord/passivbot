from copy import deepcopy

import pytest

import optimization.evaluation_implementation as implementation


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


def test_fast_identity_covers_dynamic_sources_and_all_distribution_metadata(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    source_root = tmp_path / "src"
    source_root.mkdir()
    module = source_root / "chosen_at_runtime.py"
    module.write_text("value = 1\n")
    monkeypatch.setattr(
        implementation, "__file__", str(source_root / "optimization/identity.py")
    )
    monkeypatch.setattr(
        implementation,
        "verify_loaded_runtime_extension",
        lambda: {
            "runtime_compiled_source_stamp": "source",
            "runtime_compiled_sha256": "artifact",
        },
    )
    texts = {"A": "Name: A\nVersion: 1", "B": "Name: B\nVersion: 2"}
    distributions = [
        SimpleNamespace(
            metadata={"Name": name},
            version=str(index + 1),
            read_text=lambda field, name=name: (
                texts[name] if field == "METADATA" else None
            ),
        )
        for index, name in enumerate(texts)
    ]
    monkeypatch.setattr(
        implementation.metadata, "distributions", lambda: iter(distributions)
    )
    monkeypatch.setattr(
        implementation.metadata,
        "packages_distributions",
        lambda: pytest.fail("provider walk used"),
    )
    identity = implementation.evaluation_implementation_identity.__wrapped__
    first = identity()
    assert first["version"] == 2
    distributions.reverse()
    assert identity() == first  # Installation traversal order is not semantic.
    module.write_text("value = 2\n")
    assert identity()["python_source_sha256"] != first["python_source_sha256"]
    before = identity()
    texts["B"] += "\nRequires-Dist: new-dependency"
    assert identity()["python_dependencies"] != before["python_dependencies"]
    texts["B"] = None
    with pytest.raises(RuntimeError, match="no reproducible metadata"):
        identity()
