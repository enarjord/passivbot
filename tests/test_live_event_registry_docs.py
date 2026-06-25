from pathlib import Path
import re

from live.event_bus import EventTags, ReasonCodes


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DOC = REPO_ROOT / "docs" / "ai" / "live_event_registry.md"


def _registry_values(registry: type) -> list[str]:
    return sorted(
        value
        for name, value in vars(registry).items()
        if name.isupper() and isinstance(value, str)
    )


def _documented_values(section: str) -> list[str]:
    text = REGISTRY_DOC.read_text()
    pattern = rf"^## {re.escape(section)}\n(?P<body>.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.M | re.S)
    assert match is not None, f"missing registry doc section: {section}"
    return sorted(re.findall(r"^- `([^`]+)`$", match.group("body"), re.M))


def test_live_event_tag_docs_match_registry():
    assert _documented_values("Event Tags") == _registry_values(EventTags)


def test_live_event_reason_code_docs_match_registry():
    assert _documented_values("Reason Codes") == _registry_values(ReasonCodes)
