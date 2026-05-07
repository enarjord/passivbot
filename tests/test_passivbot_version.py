from passivbot_version import __version__
from config.schema import CONFIG_SCHEMA_VERSION


def _major(version: str) -> int:
    normalized = version.strip().lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return int(normalized.split(".", 1)[0])


def test_package_major_version_matches_config_schema_major():
    assert _major(__version__) == _major(CONFIG_SCHEMA_VERSION)
