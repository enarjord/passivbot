from pathlib import Path
from functools import lru_cache
import os
from runpy import run_path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools_rust import Binding, RustExtension, build_rust


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
RUST_DIR = ROOT / "passivbot-rust"
VERSION_NS = {}
exec((SRC_DIR / "passivbot_version.py").read_text(encoding="utf-8"), VERSION_NS)
RUST_UTILS = run_path(str(SRC_DIR / "rust_utils.py"))


@lru_cache(maxsize=1)
def build_source_fingerprint():
    return RUST_UTILS["source_fingerprint"](RUST_DIR)


class BuildPyWithResources(build_py):
    """Ship the checked config schema and the expected native build fingerprint."""

    def run(self):
        super().run()
        destination = Path(self.build_lib) / "tools" / "schemas" / "config-v9.schema.json"
        self.mkpath(str(destination.parent))
        self.copy_file(str(ROOT / "schemas" / "config-v9.schema.json"), str(destination))
        stamp = Path(self.build_lib) / RUST_UTILS["WHEEL_SOURCE_STAMP"]
        self.mkpath(str(stamp.parent))
        stamp.write_text(f"{build_source_fingerprint()}\n", encoding="utf-8")

    def get_outputs(self, include_bytecode=True):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "tools" / "schemas" / "config-v9.schema.json"),
            str(Path(self.build_lib) / RUST_UTILS["WHEEL_SOURCE_STAMP"]),
        ]


class BuildRustWithSourceFingerprint(build_rust):
    def run(self):
        fingerprint = build_source_fingerprint()
        if fingerprint is None:
            raise RuntimeError("Cannot fingerprint the Rust sources for this build.")
        key = "PASSIVBOT_RUST_SOURCE_FINGERPRINT"
        previous = os.environ.get(key)
        os.environ[key] = fingerprint
        try:
            super().run()
        finally:
            if previous is None:
                os.environ.pop(key)
            else:
                os.environ[key] = previous
        if RUST_UTILS["source_fingerprint"](RUST_DIR) != fingerprint:
            raise RuntimeError("Rust sources changed during the build; build again.")


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
    version=VERSION_NS["__version__"],
    python_requires=">=3.12,<3.15,!=3.13.*",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=discover_py_modules(),
    rust_extensions=[
        RustExtension(
            "passivbot_rust", path="passivbot-rust/Cargo.toml",
            binding=Binding.PyO3, cargo_manifest_args=["--locked"],
        )
    ],
    install_requires=LIVE_REQUIREMENTS,
    extras_require={
        "full": FULL_REQUIREMENTS,
        "dev": FULL_REQUIREMENTS + DEV_REQUIREMENTS,
        "gpu-cuda": [
            "torch>=2.13,<2.14; platform_system == 'Linux'",
            "cupy-cuda12x>=13.6,<14; platform_system == 'Linux' and python_version < '3.14'",
            "cupy-cuda12x>=14.2,<15; platform_system == 'Linux' and python_version >= '3.14'",
            "nvidia-cuda-runtime-cu12==12.8.90; platform_system == 'Linux'",
            "nvidia-cuda-nvrtc-cu12==12.8.93; platform_system == 'Linux'",
        ],
        "gpu-mps": [
            "torch>=2.13,<2.14; platform_system == 'Darwin' and platform_machine == 'arm64'",
        ],
    },
    setup_requires=["setuptools-rust>=1.9.0", "wheel"],
    entry_points={"console_scripts": ["passivbot=passivbot_cli.main:console_main"]},
    include_package_data=True,
    package_data={"tools": ["backtest_dashboard_static/*.html", "backtest_dashboard_static/*.css", "backtest_dashboard_static/*.js"]},
    cmdclass={
        "build_py": BuildPyWithResources,
        "build_rust": BuildRustWithSourceFingerprint,
    },
    zip_safe=False,
)
