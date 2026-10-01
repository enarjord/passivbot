from packaging.version import Version

from passivbot_version import __version__
from config.schema import (
    CONFIG_SCHEMA_VERSION,
    SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS,
)


def _major(version: str) -> int:
    normalized = version.strip().lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return int(normalized.split(".", 1)[0])


def test_package_major_version_matches_config_schema_major():
    assert _major(__version__) == _major(CONFIG_SCHEMA_VERSION)


def test_v8_package_line_preserves_forward_schema_migration_boundary():
    package = Version(__version__)
    schema = Version(CONFIG_SCHEMA_VERSION.removeprefix("v"))
    assert package.release == (8, 2, 0)
    assert package.dev == 0
    assert schema.release == (8, 6, 0)
    assert "v8.5.0" in SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS
    assert CONFIG_SCHEMA_VERSION not in SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS
    assert all(
        Version(old.removeprefix("v")) < schema
        for old in SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS
    )
