from pathlib import Path

from setuptools import setup, find_packages
from setuptools_rust import RustExtension


BASE_DIR = Path(__file__).resolve().parent


def parse_requirements(filename, _seen=None):
    """Parse a requirements file supporting nested includes."""

    if _seen is None:
        _seen = set()

    path = BASE_DIR / filename
    if path in _seen:
        return []
    _seen.add(path)

    requirements = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(f"Invalid requirements include directive: {raw_line}")
            requirements.extend(parse_requirements(parts[1].strip(), _seen))
        else:
            requirements.append(line)
    return requirements


setup(
    name="passivbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    rust_extensions=[
        RustExtension("passivbot_rust", path="passivbot-rust/Cargo.toml", binding="pyo3")
    ],
    install_requires=parse_requirements("requirements.txt"),
    setup_requires=["setuptools-rust>=1.9.0", "wheel"],
    include_package_data=True,
    zip_safe=False,
)
