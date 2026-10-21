"""Migration outputs remain authored policy after local source resolution."""

from copy import deepcopy
import json

import jsonschema
import pytest

from config import prepare_config
from config.json_schema import generate_schema
from config.migrations.released_v8 import get_released_v8_defaults
from passivbot_cli.main import main as cli_main
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("interface", ["programmatic", "cli"])
def test_released_migration_exports_policy_without_internal_source_context(
    tmp_path, strategy_kind, interface
):
    # Start from frozen public-v8 defaults, with one declared local coin pin.
    source = get_released_v8_defaults(strategy_kind=strategy_kind)
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    coin_path = source_dir / "coin.json"
    coin_path.write_text(
        json.dumps({"bot": {"long": {"unstuck": {"ema_span_0": 17.5}}}}),
        encoding="utf-8",
    )
    source["coin_overrides"] = {"BTC": {"override_config_path": coin_path.name}}
    source_path = source_dir / "source.v8.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    original = deepcopy(source)
    source_bytes, coin_bytes = source_path.read_bytes(), coin_path.read_bytes()

    if interface == "programmatic":
        output = migrate(
            source, best_effort=True, base_config_path=str(source_path)
        ).config
    else:
        output_path = tmp_path / "output.v9.json"
        assert (
            cli_main(
                [
                    "tool",
                    "migrate-config-v9",
                    str(source_path),
                    str(output_path),
                    "--best-effort",
                ]
            )
            == 0
        )
        output = json.loads(output_path.read_text(encoding="utf-8"))

    assert source == original
    assert source_path.read_bytes() == source_bytes
    assert coin_path.read_bytes() == coin_bytes
    assert "base_config_path" not in output["live"]
    assert output["coin_overrides"]["BTC"] == {
        "bot": {"long": {"unstuck": {"ema_span_0_minutes": 17.5}}}
    }
    jsonschema.Draft202012Validator(generate_schema()).validate(output)
    reloaded = prepare_config(output, verbose=False)
    assert reloaded["bot"]["global"]["strategy_kind"] == strategy_kind
    assert (
        reloaded["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_0_minutes"]
        == 17.5
    )
