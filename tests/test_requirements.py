from importlib.metadata import requires, version
from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: str) -> set[str]:
    rows: set[str] = set()
    for raw_line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            rows.add(line)
    return rows


def test_live_requirements_include_psutil_for_health_telemetry():
    requirements = _requirements("requirements-live.txt")

    assert 'psutil==5.9.8; python_version < "3.14"' in requirements
    assert 'psutil==7.2.2; python_version >= "3.14"' in requirements


def _pinned_version(requirements: set[str], package_name: str) -> str:
    parsed = [Requirement(row) for row in requirements if not row.startswith("-")]
    matches = [
        requirement
        for requirement in parsed
        if requirement.name.casefold() == package_name.casefold() and requirement.marker is None
    ]
    assert len(matches) == 1
    requirement = matches[0]
    specifiers = list(requirement.specifier)
    assert len(specifiers) == 1 and specifiers[0].operator == "=="
    return specifiers[0].version


def test_project_pins_satisfy_ccxt_runtime_dependencies():
    ccxt_version = _pinned_version(_requirements("requirements-live.txt"), "ccxt")
    assert version("ccxt") == ccxt_version

    project_requirements = _requirements("requirements-live.txt") | _requirements(
        "requirements-full.txt"
    )
    active_project_pins = {
        requirement.name.casefold(): requirement.specifier
        for row in project_requirements
        if not row.startswith("-")
        and ((requirement := Requirement(row)).marker is None or requirement.marker.evaluate())
    }
    ccxt_dependencies = [
        Requirement(row)
        for row in requires("ccxt") or []
        if (Requirement(row).marker is None or Requirement(row).marker.evaluate())
    ]
    overlapping_dependencies = {
        dependency.name.casefold(): dependency for dependency in ccxt_dependencies
        if dependency.name.casefold() in active_project_pins
    }

    assert {"aiohttp", "requests"} <= overlapping_dependencies.keys()
    for package_name, dependency in overlapping_dependencies.items():
        project_specifier = active_project_pins[package_name]
        exact_project_versions = [
            specifier.version
            for specifier in project_specifier
            if specifier.operator == "=="
        ]
        assert exact_project_versions
        assert all(
            dependency.specifier.contains(project_version, prereleases=True)
            for project_version in exact_project_versions
        )
