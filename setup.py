from pathlib import Path

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"


def parse_requirements(filename, seen=None):
    if seen is None:
        seen = set()
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)

    requirements = []
    with path.open("r") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                nested = line[3:].strip()
                requirements.extend(parse_requirements(path.parent / nested, seen=seen))
                continue
            requirements.append(line)
    return requirements


def discover_py_modules():
    return sorted(path.stem for path in SRC_DIR.glob("*.py") if path.name != "__init__.py")


LIVE_REQUIREMENTS = parse_requirements("requirements-live.txt")
FULL_REQUIREMENTS = parse_requirements("requirements-full.txt")
DEV_REQUIREMENTS = parse_requirements("requirements-dev.txt")


setup(
    name="passivbot",
    version="7.8.5",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=discover_py_modules(),
    rust_extensions=[
        RustExtension("passivbot_rust", path="passivbot-rust/Cargo.toml", binding=Binding.PyO3)
    ],
    install_requires=LIVE_REQUIREMENTS,
    extras_require={
        "full": FULL_REQUIREMENTS,
        "dev": FULL_REQUIREMENTS + DEV_REQUIREMENTS,
    },
    setup_requires=["setuptools-rust>=1.9.0", "wheel"],
    entry_points={"console_scripts": ["passivbot=passivbot_cli.main:console_main"]},
    include_package_data=True,
    zip_safe=False,
)
