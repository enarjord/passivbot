"""The retired migration command must not silently choose v9 risk policy."""

import pytest

from tools.migrate_hsl_config import main, migrate


def test_retired_python_migrator_has_actionable_v9_guidance():
    with pytest.raises(ValueError, match="migrate-config-v9.*lookback-ema-span-minutes"):
        migrate({"config_version": "v8.6.0"}, restart_policies={"long": "always"})


def test_retired_cli_requires_explicit_v9_migration(capsys):
    with pytest.raises(SystemExit) as error:
        main(["input.json", "output.json", "--restart-policy", "long=always"])
    assert error.value.code == 2
    assert "migrate-config-v9" in capsys.readouterr().err
