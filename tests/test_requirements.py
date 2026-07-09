from pathlib import Path


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
