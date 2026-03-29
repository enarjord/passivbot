import types

from rust_utils import collect_runtime_provenance


def test_collect_runtime_provenance_uses_loaded_module_and_preferred_path(monkeypatch, tmp_path):
    runtime_path = tmp_path / "passivbot_rust_runtime.so"
    runtime_path.write_bytes(b"runtime")
    preferred_path = tmp_path / "passivbot_rust_preferred.so"
    preferred_path.write_bytes(b"preferred")

    fake_module = types.SimpleNamespace(__file__=str(runtime_path))

    monkeypatch.setattr("rust_utils.preferred_compiled_path", lambda: preferred_path)
    monkeypatch.setitem(__import__("sys").modules, "passivbot_rust", fake_module)

    provenance = collect_runtime_provenance()

    assert provenance["module_loaded"] is True
    assert provenance["runtime_module_path"] == str(runtime_path)
    assert provenance["preferred_compiled_path"] == str(preferred_path)
    assert provenance["runtime_module_sha256"] != provenance["preferred_compiled_sha256"]
    assert provenance["runtime_matches_preferred"] is False
