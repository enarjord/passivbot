import ast
from pathlib import Path


EXCHANGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "exchanges"


def _ccxt_exchange_classes() -> list[tuple[Path, str, set[str]]]:
    classes = []
    for path in sorted(EXCHANGE_ROOT.glob("*.py")):
        if path.name == "ccxt_bot.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {getattr(base, "id", getattr(base, "attr", "")) for base in node.bases}
            if "CCXTBot" not in bases:
                continue
            methods = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            classes.append((path, node.name, methods))
    return classes


def test_custom_legacy_balance_fetches_are_exposed_to_staged_capture():
    """Staged balance capture must not bypass exchange-specific legacy parsing."""

    offenders = []
    for path, class_name, methods in _ccxt_exchange_classes():
        if "fetch_balance" not in methods:
            continue
        if "capture_balance_snapshot" in methods or "_get_balance" in methods:
            continue
        offenders.append(f"{path.name}:{class_name}")

    assert not offenders, (
        "Exchange overrides fetch_balance(), but staged capture would bypass that legacy parser. "
        "Move parsing into _get_balance() or add capture_balance_snapshot(): "
        + ", ".join(offenders)
    )


def test_custom_legacy_position_fetches_are_exposed_to_staged_capture():
    """Staged position capture must not bypass exchange-specific legacy normalization."""

    offenders = []
    for path, class_name, methods in _ccxt_exchange_classes():
        if "fetch_positions" not in methods:
            continue
        if "capture_positions_snapshot" in methods or "_normalize_positions" in methods:
            continue
        offenders.append(f"{path.name}:{class_name}")

    assert not offenders, (
        "Exchange overrides fetch_positions(), but staged capture would bypass that legacy "
        "normalizer. Move normalization into _normalize_positions() or add "
        "capture_positions_snapshot(): "
        + ", ".join(offenders)
    )
