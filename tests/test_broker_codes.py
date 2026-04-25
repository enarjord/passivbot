import pytest

from procedures import load_broker_code, load_broker_codes


def test_load_broker_code_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSIVBOT_BROKER_CODES_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    assert load_broker_code("bybit") == "passivbotbybit"


def test_load_broker_code_uses_env_override(monkeypatch, tmp_path):
    path = tmp_path / "custom_broker_codes.hjson"
    path.write_text('{ bybit: "custom-bybit", hyperliquid: null }')
    monkeypatch.setenv("PASSIVBOT_BROKER_CODES_PATH", str(path))

    assert load_broker_code("bybit") == "custom-bybit"
    assert load_broker_code("hyperliquid") == ""


def test_load_broker_code_fails_loudly_on_unknown_exchange(monkeypatch):
    monkeypatch.delenv("PASSIVBOT_BROKER_CODES_PATH", raising=False)

    with pytest.raises(KeyError, match="bybbit"):
        load_broker_code("bybbit")


def test_load_broker_code_fails_loudly_when_env_override_lacks_exchange(monkeypatch, tmp_path):
    path = tmp_path / "custom_broker_codes.hjson"
    path.write_text('{ bybit: "custom-bybit" }')
    monkeypatch.setenv("PASSIVBOT_BROKER_CODES_PATH", str(path))

    with pytest.raises(KeyError, match="hyperliquid"):
        load_broker_code("hyperliquid")


def test_load_broker_code_rejects_invalid_entry_type(monkeypatch, tmp_path):
    path = tmp_path / "bad_type.hjson"
    path.write_text("{ bybit: 123 }")
    monkeypatch.setenv("PASSIVBOT_BROKER_CODES_PATH", str(path))

    with pytest.raises(TypeError, match="string, object, or null"):
        load_broker_code("bybit")


def test_load_broker_codes_fails_loudly_on_missing_env_path(monkeypatch, tmp_path):
    missing = tmp_path / "missing.hjson"
    monkeypatch.setenv("PASSIVBOT_BROKER_CODES_PATH", str(missing))

    with pytest.raises(FileNotFoundError, match="broker code registry not found"):
        load_broker_codes()


def test_load_broker_codes_fails_loudly_on_parse_error(monkeypatch, tmp_path):
    path = tmp_path / "broken.hjson"
    path.write_text("{ bybit: ")
    monkeypatch.setenv("PASSIVBOT_BROKER_CODES_PATH", str(path))

    with pytest.raises(RuntimeError, match="failed to load broker code registry"):
        load_broker_codes()


def test_load_broker_codes_requires_top_level_object(monkeypatch, tmp_path):
    path = tmp_path / "list.hjson"
    path.write_text('["bybit"]')
    monkeypatch.setenv("PASSIVBOT_BROKER_CODES_PATH", str(path))

    with pytest.raises(TypeError, match="top-level object"):
        load_broker_codes()
