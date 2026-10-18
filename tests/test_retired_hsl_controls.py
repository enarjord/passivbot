"""Retired recovery controls require explicit migration, even while disabled."""
import json

import pytest

from config.load import load_input_config, load_prepared_config, prepare_config
from config.schema import get_template_config
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("field", ["hsl_accept_incomplete_history", "hsl_unavailable_grace_seconds", "risk_input_max_attempts"])
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("shape", ["current", "live_only", "nested_current"])
def test_retired_recovery_controls_rejected_without_mutating_source(tmp_path, field, value, shape):
    config = get_template_config()
    config["live"][field] = value
    if shape == "live_only":
        config = {key: config[key] for key in ("config_version", "bot", "live")}
    document = {"config": config} if shape == "nested_current" else config
    path = tmp_path / "config.json"
    original = json.dumps(document)
    path.write_text(original)
    source, _, raw = load_input_config(str(path), log_info=False)
    assert source == raw == document
    with pytest.raises(ValueError, match=rf"live\.{field}.*removed in v9.*migrate-config-v9"):
        load_prepared_config(str(path), live_only=True, target="live", verbose=True, log_info=False)
    assert path.read_text() == original and raw == document

    # The offline migration owns removal; ordinary loading cannot grant a waiver.
    legacy = config.copy()
    legacy["config_version"] = "v8.6.0"
    result = migrate(legacy, memory_span_minutes=43200.0).config
    assert field not in result["live"]
    assert field not in prepare_config(result, target="canonical", runtime=None, verbose=False)["live"]
