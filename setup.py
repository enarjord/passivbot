from setuptools import setup, find_packages
from setuptools_rust import RustExtension


def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    name="passivbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    rust_extensions=[
        RustExtension("passivbot_rust", path="passivbot-rust/Cargo.toml", binding="pyo3")
    ],
    install_requires=parse_requirements("requirements-rust.txt"),
    setup_requires=["setuptools-rust>=1.9.0", "wheel"],
    include_package_data=True,
    zip_safe=False,
)
