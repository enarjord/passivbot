"""Package publication and runtime configuration compatibility are separate."""
from packaging.version import Version
import pytest

from passivbot_version import __version__
from config import prepare_config
from config.schema import (CONFIG_SCHEMA_VERSION, SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS, get_template_config)


def test_package_version_is_a_valid_normalized_pep440_declaration():
    assert str(Version(__version__)) == __version__


def test_v9_runtime_schema_has_no_implicit_previous_schema_migration():
    assert Version(CONFIG_SCHEMA_VERSION.removeprefix("v")).major == 9
    assert not SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS
    legacy = get_template_config()
    legacy["config_version"] = "v8.6.0"
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(legacy, verbose=False)
