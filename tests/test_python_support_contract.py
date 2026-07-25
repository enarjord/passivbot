import ast
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def test_setup_python_requires_matches_supported_versions():
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    python_requires = next(
        keyword.value.value
        for keyword in setup_call.keywords
        if keyword.arg == "python_requires"
        and isinstance(keyword.value, ast.Constant)
    )
    supported = SpecifierSet(python_requires)

    assert Version("3.12") in supported
    assert Version("3.13") not in supported
    assert Version("3.14") in supported
    assert Version("3.15") not in supported
