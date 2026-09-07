"""Version optimizer fitness code and its imported runtime dependencies."""

import ast
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement

from rust_utils import verify_loaded_runtime_extension

# Walk imports instead of maintaining an incomplete list of calculation helpers.
_EVALUATOR_ROOTS = ("optimize", "backtest", "optimize_suite", "suite_runner")


def evaluator_dependencies(source_root: Path) -> tuple[set[Path], set[str]]:
    paths, external = set(), set()
    pending = []
    dynamic_fallback = False

    def add_module(name):
        parts = name.split(".")
        if not name or not all(part.isidentifier() for part in parts):
            return
        module_file = source_root.joinpath(*parts).with_suffix(".py")
        package_file = source_root.joinpath(*parts, "__init__.py")
        target = module_file if module_file.is_file() else package_file
        if not target.is_file():
            if (
                not (source_root / parts[0]).is_dir()
                and not (source_root / f"{parts[0]}.py").is_file()
            ):
                external.add(parts[0])
            return
        pending.append(target)
        for depth in range(1, len(parts)):
            init = source_root.joinpath(*parts[:depth], "__init__.py")
            if init.is_file():
                pending.append(init)

    for root in _EVALUATOR_ROOTS:
        if not (source_root / f"{root}.py").is_file():
            raise RuntimeError(f"Missing evaluator source root: {root}")
        add_module(root)
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        relative = path.relative_to(source_root)
        package = list(relative.parts[:-1])
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        dynamic_names = {"__import__", "import_module"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join([*prefix, *([node.module] if node.module else [])])
                add_module(module)
                for alias in node.names:
                    add_module(".".join(filter(None, [module, alias.name])))
                    if node.module == "importlib" and alias.name == "import_module":
                        dynamic_names.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", "")
            )
            if called not in dynamic_names:
                continue
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                name = node.args[0].value
                if name.startswith("."):
                    # Relative dynamic imports may use an explicit package argument.
                    package_arg = next(
                        (item.value for item in node.keywords if item.arg == "package"),
                        None,
                    )
                    if package_arg is None and len(node.args) > 1:
                        package_arg = node.args[1]
                    if isinstance(package_arg, ast.Constant) and isinstance(
                        package_arg.value, str
                    ):
                        base = package_arg.value.split(".")
                    elif (
                        isinstance(package_arg, ast.Name)
                        and package_arg.id == "__package__"
                    ):
                        base = package
                    else:
                        name = None
                    if name is not None:
                        levels = len(name) - len(name.lstrip("."))
                        name = ".".join(
                            [*base[: len(base) - levels + 1], name.lstrip(".")]
                        ).rstrip(".")
                if name is not None:
                    add_module(name)
                    continue
            # An unresolvable dynamic import can reach any local module. Hash the
            # entire source tree conservatively; docs and tests remain outside it.
            if not dynamic_fallback:
                pending.extend(source_root.rglob("*.py"))
                dynamic_fallback = True
    return paths, external


def _hash_sources(source_root, paths):
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def python_evaluator_source_fingerprint(source_root: Path) -> str:
    paths, _ = evaluator_dependencies(source_root)
    return _hash_sources(source_root, paths)


def evaluator_dependency_versions(external: set[str]) -> dict:
    providers = metadata.packages_distributions()
    imports = {}
    pending = []
    for name in sorted(external - sys.stdlib_module_names - {"passivbot_rust"}):
        distributions = sorted(providers.get(name, []))
        imports[name] = distributions or None
        pending.extend(distributions)
    versions = {}
    while pending:
        distribution = pending.pop()
        canonical = distribution.lower().replace("_", "-")
        if canonical in versions:
            continue
        try:
            versions[canonical] = metadata.version(distribution)
            requirements = metadata.requires(distribution) or []
        except metadata.PackageNotFoundError:
            versions[canonical] = None
            continue
        for text in requirements:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate():
                pending.append(requirement.name)
    return {"imports": imports, "distributions": dict(sorted(versions.items()))}


@lru_cache(maxsize=1)
def evaluation_implementation_identity() -> dict:
    """Freeze verified code and imported dependency versions once per process."""
    runtime = verify_loaded_runtime_extension()
    rust_source = runtime.get("runtime_compiled_source_stamp")
    rust_artifact = runtime.get("runtime_compiled_sha256")
    if not rust_source or not rust_artifact:
        raise RuntimeError(
            "Optimizer evaluation identity requires a source-stamped Rust extension; "
            "rebuild and verify the extension before starting or resuming"
        )
    source_root = Path(__file__).resolve().parents[1]
    paths, external = evaluator_dependencies(source_root)
    return {
        "version": 1,
        "python_source_sha256": _hash_sources(source_root, paths),
        "python_runtime": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "python_dependencies": evaluator_dependency_versions(external),
        "rust_source_sha256": rust_source,
        "rust_artifact_sha256": rust_artifact,
    }
